"""End-to-end behaviour of the xKV key cache combined with the MLP value
cache, eviction, selective attention and factor quantisation."""

import itertools
import math

import pytest
import torch
from transformers import DynamicCache

from cache.cache import CompressedCache
from cache.config import (
    BaselineCacheConfig,
    CompressedCacheConfig,
    MLPValueCacheConfig,
    SelectiveCacheConfig,
    XKVCacheConfig,
)
from model.selective_attention import install_selective_attention
from utils.turboquant import is_quantised_factor

from tests.helpers import build_llama, install_value_importance_hooks


NUM_LAYERS = 4
PROMPT_LEN = 64
DECODE_STEPS = 3
CHUNK_SIZE = 8

LOSSLESS_KEY_RATIO = 0.25
LOSSLESS_VALUE_RATIO = 1e-3


def _make_config(
    *,
    group_size,
    key_ratio=LOSSLESS_KEY_RATIO,
    value_ratio=LOSSLESS_VALUE_RATIO,
    selective=False,
    token_budget=PROMPT_LEN,
    quantise=False,
    eviction_keep_ratio=1.0,
    num_layers=NUM_LAYERS,
):
    return CompressedCacheConfig(
        key=XKVCacheConfig(
            layer_group_size=group_size,
            num_layers=num_layers,
            svd_backend="linalg",
            compression_ratio=key_ratio,
            quantise_a=quantise,
            quantise_b=quantise,
            compressor_bits=8,
        ),
        value=MLPValueCacheConfig(
            target_compression_ratio=value_ratio,
            num_epochs=0,
        ),
        selective=SelectiveCacheConfig(
            enabled=selective,
            token_budget=token_budget,
            chunk_size=CHUNK_SIZE,
            local_tokens=8,
            outlier_chunks=2,
        ),
        eviction_keep_ratio=eviction_keep_ratio,
    )


def _make_cache(config, prompt_len=PROMPT_LEN):
    return CompressedCache(
        config=config,
        cache_context={
            "padding_mask": torch.ones(1, prompt_len, dtype=torch.bool)
        },
        verbose=False,
    )


def _run(model, cache, prompt, steps, *, selective=False, evict=False):
    handles = []
    if evict:
        handles += install_value_importance_hooks(model)
    if selective:
        handles += install_selective_attention(model)
    try:
        with torch.no_grad():
            logits = [
                model(prompt, past_key_values=cache, use_cache=True).logits
            ]
            for step in steps:
                logits.append(
                    model(step, past_key_values=cache, use_cache=True).logits
                )
    finally:
        for handle in handles:
            handle.remove()
    return logits


def _prompt_and_steps(seed, prompt_len=PROMPT_LEN):
    torch.manual_seed(seed)
    prompt = torch.randint(3, 64, (1, prompt_len))
    steps = [torch.randint(3, 64, (1, 1)) for _ in range(DECODE_STEPS)]
    return prompt, steps


def _max_deviation(reference, actual):
    return max(
        (a - b).abs().max().item() for a, b in zip(reference, actual)
    )


@pytest.mark.parametrize("group_size", (1, 2, 4))
def test_lossless_settings_reproduce_the_dense_cache(group_size):
    """xKV keys + MLP values, sized so nothing is thrown away."""
    model = build_llama(NUM_LAYERS)
    prompt, steps = _prompt_and_steps(50)

    dense = _run(model, DynamicCache(), prompt, steps)
    compressed = _run(
        model, _make_cache(_make_config(group_size=group_size)), prompt, steps
    )

    assert _max_deviation(dense, compressed) < 1e-4


def test_baseline_keys_with_mlp_values_reproduce_the_dense_cache():
    """The MLP value cache on its own, with keys left uncompressed."""
    model = build_llama(NUM_LAYERS)
    prompt, steps = _prompt_and_steps(56)

    cache = _make_cache(
        CompressedCacheConfig(
            key=BaselineCacheConfig(),
            value=MLPValueCacheConfig(
                target_compression_ratio=LOSSLESS_VALUE_RATIO,
                num_epochs=0,
            ),
        )
    )
    dense = _run(model, DynamicCache(), prompt, steps)
    compressed = _run(model, cache, prompt, steps)

    assert _max_deviation(dense, compressed) < 1e-4
    for layer in cache.value_cache.layers:
        assert not layer.prefill and layer.is_compressed


@pytest.mark.parametrize("group_size", (1, 2, 4))
def test_selective_attention_with_a_full_budget_matches_dense(group_size):
    """A budget covering every chunk must not change the attention result."""
    model = build_llama(NUM_LAYERS)
    prompt, steps = _prompt_and_steps(51)
    assert PROMPT_LEN % CHUNK_SIZE == 0

    dense = _run(model, DynamicCache(), prompt, steps)
    cache = _make_cache(
        _make_config(group_size=group_size, selective=True)
    )
    compressed = _run(model, cache, prompt, steps, selective=True)

    assert cache.selective_layers, "selective state was never recorded"
    assert _max_deviation(dense, compressed) < 1e-4


@pytest.mark.parametrize("group_size", (1, 2, 4))
def test_eviction_plumbing_is_transparent_when_nothing_is_dropped(group_size):
    """Sinks plus the local window cover this prompt, so no token is lost.

    The eviction code path still runs end to end, including the RoPE position
    remapping that the key and value caches share.
    """
    model = build_llama(NUM_LAYERS)
    prompt, steps = _prompt_and_steps(52)

    dense = _run(model, DynamicCache(), prompt, steps)
    cache = _make_cache(
        _make_config(group_size=group_size, eviction_keep_ratio=0.5)
    )
    compressed = _run(model, cache, prompt, steps, evict=True)

    kept = cache.kept_positions
    assert sorted(kept) == list(range(NUM_LAYERS))
    for positions in kept.values():
        assert positions.tolist() == list(range(PROMPT_LEN))
    assert _max_deviation(dense, compressed) < 1e-4


@pytest.mark.parametrize("group_size", (1, 2))
def test_eviction_keeps_attention_sinks_and_the_local_window(group_size):
    """With a long enough prompt, eviction really does drop tokens."""
    prompt_len = 256
    keep_ratio = 0.5
    model = build_llama(NUM_LAYERS)
    prompt, steps = _prompt_and_steps(53, prompt_len=prompt_len)

    cache = _make_cache(
        _make_config(group_size=group_size, eviction_keep_ratio=keep_ratio),
        prompt_len=prompt_len,
    )
    _run(model, cache, prompt, steps, evict=True)

    sinks = set(range(8))
    local = set(range(prompt_len - 64, prompt_len))
    expected_count = max(math.ceil(prompt_len * keep_ratio), len(sinks | local))

    for layer_idx in range(NUM_LAYERS):
        kept = set(cache.kept_positions[layer_idx].tolist())
        assert sinks <= kept, f"layer {layer_idx} evicted an attention sink"
        assert local <= kept, f"layer {layer_idx} evicted the local window"
        assert len(kept) == expected_count
        assert len(kept) < prompt_len, "nothing was actually evicted"
        # Every layer in a group must agree on the compressed length.
        assert cache.key_cache.layer_states[layer_idx].compressed_len == len(
            kept
        )


@pytest.mark.parametrize("group_size", (2, 4))
def test_a_layer_group_evicts_the_same_positions(group_size):
    """xKV shares one left factor per group, so rows must stay aligned.

    Evicting per layer would leave row t holding a different token in each
    layer of the group, which is exactly the structure the shared factor is
    supposed to capture.
    """
    prompt_len = 256
    model = build_llama(NUM_LAYERS)
    prompt, steps = _prompt_and_steps(57, prompt_len=prompt_len)

    cache = _make_cache(
        _make_config(group_size=group_size, eviction_keep_ratio=0.5),
        prompt_len=prompt_len,
    )
    _run(model, cache, prompt, steps, evict=True)

    kept = {
        layer_idx: cache.kept_positions[layer_idx].tolist()
        for layer_idx in range(NUM_LAYERS)
    }
    for layer_idx in range(NUM_LAYERS):
        group_start = (layer_idx // group_size) * group_size
        assert kept[layer_idx] == kept[group_start], (
            f"layer {layer_idx} evicted different positions from its group"
        )

    # Independent groups still choose for themselves.
    if group_size < NUM_LAYERS:
        assert kept[0] != kept[group_size]


def test_layers_evict_independently_without_a_grouped_key_cache():
    """A key cache with no layer groups keeps the per-layer decision."""
    prompt_len = 256
    model = build_llama(NUM_LAYERS)
    prompt, steps = _prompt_and_steps(58, prompt_len=prompt_len)

    cache = _make_cache(
        CompressedCacheConfig(
            key=BaselineCacheConfig(),
            value=MLPValueCacheConfig(
                target_compression_ratio=LOSSLESS_VALUE_RATIO,
                num_epochs=0,
            ),
            eviction_keep_ratio=0.5,
        ),
        prompt_len=prompt_len,
    )
    _run(model, cache, prompt, steps, evict=True)

    kept = [cache.kept_positions[i].tolist() for i in range(NUM_LAYERS)]
    assert any(kept[i] != kept[0] for i in range(1, NUM_LAYERS))


def _key_reconstruction_error(group_size, *, evict, prompt_len=256, seed=59):
    """Mean relative error of the reconstructed keys against what went in."""
    from cache.keys import XKVKeysCache

    model = build_llama(NUM_LAYERS)
    prompt, steps = _prompt_and_steps(seed, prompt_len=prompt_len)
    cache = _make_cache(
        _make_config(
            group_size=group_size,
            key_ratio=2.0,
            eviction_keep_ratio=0.5 if evict else 1.0,
        ),
        prompt_len=prompt_len,
    )

    exact = {}
    original_update = XKVKeysCache.update

    def spy(self, key_states, layer_idx, cache_kwargs=None):
        if self.prefill:
            exact[layer_idx] = key_states.detach().clone()
        return original_update(self, key_states, layer_idx, cache_kwargs)

    XKVKeysCache.update = spy
    try:
        _run(model, cache, prompt, steps, evict=evict)
    finally:
        XKVKeysCache.update = original_update

    errors = []
    for layer_idx in range(NUM_LAYERS):
        target = exact[layer_idx]
        reconstructed = cache.key_cache.get_reconstructed_keys_only(layer_idx)
        errors.append(
            ((reconstructed - target).norm() / target.norm()).item()
        )
    return sum(errors) / len(errors)


@pytest.mark.parametrize("evict", (False, True))
def test_larger_layer_groups_improve_key_reconstruction(evict):
    """Grouping must help whether or not eviction is on.

    Per-layer eviction used to invert this: bigger groups made the shared
    factor worse because each layer kept a different set of tokens.
    """
    errors = {
        group_size: _key_reconstruction_error(group_size, evict=evict)
        for group_size in (1, 2, 4)
    }
    assert errors[4] < errors[2] < errors[1], errors


@pytest.mark.parametrize("group_size", (1, 2, 4))
def test_quantised_factors_stay_close_to_unquantised(group_size):
    """8-bit factor quantisation is lossy, but only mildly so."""
    model = build_llama(NUM_LAYERS)
    prompt, steps = _prompt_and_steps(54)

    reference = _run(
        model, _make_cache(_make_config(group_size=group_size)), prompt, steps
    )
    cache = _make_cache(_make_config(group_size=group_size, quantise=True))
    quantised = _run(model, cache, prompt, steps)

    for logits in quantised:
        assert torch.isfinite(logits).all()

    # The stored factors really are quantised, not silently left as tensors.
    for group_state in cache.key_cache.group_states.values():
        assert is_quantised_factor(group_state.packed_shared)
    for layer_state in cache.key_cache.layer_states.values():
        assert is_quantised_factor(layer_state.packed_right)

    scale = max(logits.abs().max().item() for logits in reference)
    deviation = _max_deviation(reference, quantised)
    assert 0.0 < deviation < 0.02 * scale, deviation

    # Quantisation is what pays for the extra compression.
    plain = _make_cache(_make_config(group_size=group_size))
    _run(model, plain, prompt, steps)
    assert cache.key_cache.comp_ratio > 2 * plain.key_cache.comp_ratio


@pytest.mark.parametrize(
    ("group_size", "evict", "selective", "quantise"),
    [
        (group_size, *flags)
        for group_size in (1, 2, 4)
        for flags in itertools.product((False, True), repeat=3)
    ],
)
def test_combinations_prefill_and_decode(
    group_size,
    evict,
    selective,
    quantise,
):
    """Every xKV + MLP combination survives prefill and several decode steps."""
    model = build_llama(NUM_LAYERS)
    prompt, steps = _prompt_and_steps(55)

    cache = _make_cache(
        _make_config(
            group_size=group_size,
            selective=selective,
            token_budget=32,
            quantise=quantise,
            eviction_keep_ratio=0.5 if evict else 1.0,
        )
    )
    logits = _run(model, cache, prompt, steps, selective=selective, evict=evict)

    assert logits[0].shape == (1, PROMPT_LEN, model.config.vocab_size)
    for step_logits in logits[1:]:
        assert step_logits.shape == (1, 1, model.config.vocab_size)
        assert torch.isfinite(step_logits).all()

    assert not cache.key_cache.prefill
    assert cache.get_seq_length(0) == PROMPT_LEN + DECODE_STEPS
    assert cache.comp_ratio > 0
    if selective:
        assert sorted(cache.selective_layers) == list(range(NUM_LAYERS))
    for layer_idx in range(NUM_LAYERS):
        assert cache.key_cache.layer_states[layer_idx].is_compressed
        assert not cache.value_cache.layers[layer_idx].prefill
