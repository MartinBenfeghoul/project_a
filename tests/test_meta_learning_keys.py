"""Meta-training on the keys the value MLP actually meets at eval time."""

import pytest
import torch
from transformers import DynamicCache

from cache.backends.mlp_values import MLPValueLayer
from cache.config import (
    CompressedCacheConfig,
    MLPValueCacheConfig,
    XKVCacheConfig,
)
from cache.core import CompressedCache
from model.meta_learning import KeyReconstructionConfig, prepare_kvs
from utils.rope import inverse_rope, model_rope_cos_sin

from tests.helpers import LLAMA3_ROPE_SCALING, build_llama


NUM_LAYERS = 4
PROMPT_LEN = 64
GROUP_SIZE = 2
COMP_RATIO = 2.0


def _key_config():
    return KeyReconstructionConfig(
        compression_ratio=COMP_RATIO,
        layer_group_size=GROUP_SIZE,
        svd_backend="linalg",
    )


def _prefill(model, cache, prompt):
    with torch.no_grad():
        model(prompt, past_key_values=cache, use_cache=True)
    return cache


def _eval_training_keys(monkeypatch, model, prompt):
    """The un-roped keys each value MLP is fitted on during eval prefill."""
    cache = CompressedCache(
        config=CompressedCacheConfig(
            key=XKVCacheConfig(
                layer_group_size=GROUP_SIZE,
                num_layers=NUM_LAYERS,
                svd_backend="linalg",
                compression_ratio=COMP_RATIO,
            ),
            value=MLPValueCacheConfig(
                target_compression_ratio=1e-3,
                num_epochs=0,
            ),
        ),
        cache_context={
            "padding_mask": torch.ones(
                prompt.size(0), prompt.size(1), dtype=torch.bool
            )
        },
        verbose=False,
    )
    captured = {}
    original = MLPValueLayer.train_mlp

    def spy(self, keys, padding_mask):
        captured[cache.value_cache.layers.index(self)] = keys.detach().clone()
        return original(self, keys, padding_mask)

    monkeypatch.setattr(MLPValueLayer, "train_mlp", spy)
    _prefill(model, cache, prompt)
    return [captured[layer_idx] for layer_idx in range(NUM_LAYERS)]


def _support_kvs(model, prompt, key_config):
    cache = _prefill(model, DynamicCache(), prompt)
    return prepare_kvs(
        cache,
        rope=model_rope_cos_sin(
            model, prompt.size(1), prompt.device, torch.float32
        ),
        dtype=torch.float32,
        key_config=key_config,
        padding_mask=torch.ones(
            prompt.size(0), prompt.size(1), dtype=torch.bool
        ),
    )


@pytest.mark.parametrize(
    "rope_scaling",
    [None, LLAMA3_ROPE_SCALING],
    ids=["plain_rope", "llama3_scaled_rope"],
)
def test_inner_loop_keys_match_the_eval_reconstruction(monkeypatch, rope_scaling):
    """The support keys are the ones eval hands the value MLP, not the exact
    keys the model computed."""
    torch.manual_seed(60)
    model = build_llama(NUM_LAYERS, rope_scaling=rope_scaling)
    prompt = torch.randint(3, 64, (1, PROMPT_LEN))

    kvs = _support_kvs(model, prompt, _key_config())
    eval_keys = _eval_training_keys(monkeypatch, model, prompt)

    assert len(kvs) == NUM_LAYERS
    for layer_idx, ((keys, _), expected) in enumerate(zip(kvs, eval_keys)):
        torch.testing.assert_close(
            keys,
            expected,
            atol=1e-5,
            rtol=1e-5,
            msg=lambda text, idx=layer_idx: f"layer {idx}: {text}",
        )


def test_reconstructed_keys_are_lossy_but_track_the_exact_keys():
    torch.manual_seed(61)
    model = build_llama(NUM_LAYERS)
    prompt = torch.randint(3, 64, (1, PROMPT_LEN))

    exact = _support_kvs(model, prompt, None)
    reconstructed = _support_kvs(model, prompt, _key_config())

    for layer_idx, ((keys, values), (recon, recon_values)) in enumerate(
        zip(exact, reconstructed)
    ):
        # Values are the targets and stay exact; only the inputs change.
        torch.testing.assert_close(values, recon_values)
        assert not torch.allclose(keys, recon, atol=1e-3, rtol=1e-3), (
            f"layer {layer_idx} keys were not compressed"
        )
        error = (recon - keys).norm() / keys.norm()
        assert error < 1.0, f"layer {layer_idx} reconstruction error {error}"


@pytest.mark.parametrize(
    "rope_scaling",
    [None, LLAMA3_ROPE_SCALING],
    ids=["plain_rope", "llama3_scaled_rope"],
)
def test_without_a_key_config_the_support_keys_stay_exact(rope_scaling):
    torch.manual_seed(62)
    model = build_llama(NUM_LAYERS, rope_scaling=rope_scaling)
    prompt = torch.randint(3, 64, (1, PROMPT_LEN))

    cache = _prefill(model, DynamicCache(), prompt)
    roped = [layer.keys.clone() for layer in cache.layers]
    kvs = prepare_kvs(
        cache,
        rope=model_rope_cos_sin(
            model, prompt.size(1), prompt.device, torch.float32
        ),
        dtype=torch.float32,
    )

    for keys, (support_keys, _) in zip(roped, kvs):
        torch.testing.assert_close(
            support_keys,
            _unrope(keys, model),
            atol=1e-6,
            rtol=1e-6,
        )


def _unrope(keys, model):
    """Undo RoPE with the model's own cos/sin, scaling included."""
    cos, sin = model_rope_cos_sin(
        model, keys.shape[2], keys.device, keys.dtype
    )
    return inverse_rope(keys, cos[:, None], sin[:, None])
