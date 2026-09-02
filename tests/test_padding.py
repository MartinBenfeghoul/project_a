"""Padding must not reach the SVD, the rank budget, or the landmark topk."""

import pytest
import torch

from cache.backends.xkv import XKVKeysCache
from cache.config import (
    BaselineCacheConfig,
    CompressedCacheConfig,
    SelectiveCacheConfig,
    XKVCacheConfig,
)
from cache.core import CompressedCache
from cache.rope import SharedRopeCache
from efficiency import adjust_rank

from tests.helpers import apply_model_rope, low_rank_keys, rope_cos_sin


def _left_pad_mask(batch_size, pad_len, seq_len):
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    mask[:, :pad_len] = False
    return mask


def _build_cache(**kwargs) -> XKVKeysCache:
    defaults = dict(
        layer_group_size=1,
        num_layers=1,
        xkv_svd_backend="linalg",
        comp_ratio=2.0,
        rope_cache=SharedRopeCache(),
    )
    return XKVKeysCache(**{**defaults, **kwargs})


def _capture_decomposition_inputs(monkeypatch):
    import cache.backends.xkv as keys_module

    captured = []
    original = keys_module.decompose_grouped_xkv_to_segment_store

    def spy(tensor, *args, **kwargs):
        captured.append(tensor.clone())
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(
        keys_module, "decompose_grouped_xkv_to_segment_store", spy
    )
    return captured


# --- the decomposition ----------------------------------------------------


def test_padded_rows_are_zeroed_before_decomposition(monkeypatch):
    torch.manual_seed(0)
    batch_size, num_heads, seq_len, head_dim, pad_len = 1, 2, 32, 16, 8
    cos, sin = rope_cos_sin(batch_size, seq_len, head_dim)
    keys = apply_model_rope(
        torch.randn(batch_size, num_heads, seq_len, head_dim), cos, sin
    )

    captured = _capture_decomposition_inputs(monkeypatch)
    cache = _build_cache()
    cache.update(
        keys,
        0,
        {
            "cos": cos,
            "sin": sin,
            "padding_mask": _left_pad_mask(batch_size, pad_len, seq_len),
        },
    )

    (decomposed,) = captured
    assert torch.count_nonzero(decomposed[..., :pad_len, :]) == 0
    # The real tokens must be left untouched.
    assert torch.count_nonzero(decomposed[..., pad_len:, :]) > 0


def test_padding_does_not_change_the_fitted_subspace():
    """Zeroing reproduces decomposing the unpadded keys, it does not
    approximate it: the valid rows reconstruct identically either way."""
    torch.manual_seed(1)
    batch_size, num_heads, valid_len, head_dim, pad_len = 1, 2, 48, 16, 16
    seq_len = valid_len + pad_len
    cos, sin = rope_cos_sin(batch_size, seq_len, head_dim)
    unroped = torch.randn(batch_size, num_heads, seq_len, head_dim)
    keys = apply_model_rope(unroped, cos, sin)

    padded_cache = _build_cache()
    padded_cache.update(
        keys,
        0,
        {
            "cos": cos,
            "sin": sin,
            "padding_mask": _left_pad_mask(batch_size, pad_len, seq_len),
        },
    )
    padded_recon = padded_cache.get_reconstructed_keys_only(0)

    # The same tokens, decomposed without ever seeing the padding. The rank
    # is pinned so the two runs are compared at equal storage.
    rank = padded_cache.layer_states[0].packed_right.shape[-2]
    unpadded_cache = _build_cache()
    unpadded_cache.decomposition = type(unpadded_cache.decomposition)(
        compression_ratio=valid_len
        * num_heads
        * head_dim
        / (rank * (valid_len + num_heads * head_dim)),
        svd_backend="linalg",
    )
    cos_valid, sin_valid = cos[:, pad_len:], sin[:, pad_len:]
    unpadded_cache.update(
        keys[..., pad_len:, :], 0, {"cos": cos_valid, "sin": sin_valid}
    )
    unpadded_recon = unpadded_cache.get_reconstructed_keys_only(0)

    assert padded_recon.size(-2) == seq_len
    # Padded rows reconstruct as zeros, real rows match the padding-free fit.
    torch.testing.assert_close(
        padded_recon[..., :pad_len, :],
        torch.zeros_like(padded_recon[..., :pad_len, :]),
        atol=1e-4,
        rtol=0,
    )
    torch.testing.assert_close(
        padded_recon[..., pad_len:, :], unpadded_recon, atol=1e-4, rtol=1e-3
    )


def test_padding_improves_reconstruction_of_the_real_tokens():
    torch.manual_seed(2)
    batch_size, num_heads, valid_len, head_dim, pad_len = 1, 4, 64, 16, 32
    seq_len = valid_len + pad_len
    cos, sin = rope_cos_sin(batch_size, seq_len, head_dim)
    keys = apply_model_rope(
        low_rank_keys(batch_size, num_heads, seq_len, head_dim, rank=12),
        cos,
        sin,
    )
    mask = _left_pad_mask(batch_size, pad_len, seq_len)

    def error(padding_mask):
        cache = _build_cache(comp_ratio=4.0)
        kwargs = {"cos": cos, "sin": sin}
        if padding_mask is not None:
            kwargs["padding_mask"] = padding_mask
        cache.update(keys, 0, kwargs)
        recon = cache.get_reconstructed_keys_only(0)[..., pad_len:, :]
        target = keys[..., pad_len:, :]
        return ((recon - target).norm() / target.norm()).item()

    assert error(mask) < error(None)


# --- the rank budget ------------------------------------------------------


def test_rank_does_not_depend_on_padding():
    """A target ratio buys the same rank however the batch was padded."""
    torch.manual_seed(11)
    num_heads, valid_len, head_dim = 2, 48, 16

    def rank_for(pad_len):
        seq_len = valid_len + pad_len
        cos, sin = rope_cos_sin(1, seq_len, head_dim)
        keys = apply_model_rope(
            torch.randn(1, num_heads, seq_len, head_dim), cos, sin
        )
        cache = _build_cache(comp_ratio=4.0)
        kwargs = {"cos": cos, "sin": sin}
        if pad_len:
            kwargs["padding_mask"] = _left_pad_mask(1, pad_len, seq_len)
        cache.update(keys, 0, kwargs)
        return cache.layer_states[0].packed_right.shape[-2]

    unpadded = rank_for(0)
    assert rank_for(16) == unpadded
    assert rank_for(64) == unpadded
    # The unpadded rank is the one adjust_rank picks from the true length.
    assert unpadded == adjust_rank(
        valid_len, num_heads * head_dim, 4.0, 0, 1, 4
    )


def test_compression_ratio_measures_against_real_tokens():
    torch.manual_seed(3)
    batch_size, num_heads, seq_len, head_dim, pad_len = 1, 2, 64, 16, 32
    cos, sin = rope_cos_sin(batch_size, seq_len, head_dim)
    keys = apply_model_rope(
        torch.randn(batch_size, num_heads, seq_len, head_dim), cos, sin
    )

    cache = _build_cache(comp_ratio=4.0)
    cache.update(
        keys,
        0,
        {
            "cos": cos,
            "sin": sin,
            "padding_mask": _left_pad_mask(batch_size, pad_len, seq_len),
        },
    )

    state = cache.layer_states[0]
    valid = seq_len - pad_len
    rank = state.packed_right.shape[-2]
    # Padding is absent from both sides: the ratio is the one this rank would
    # have achieved on a `valid`-token prompt that was never padded.
    stored = rank * (valid + num_heads * head_dim)
    expected = valid * num_heads * head_dim / stored
    assert cache.comp_ratio == pytest.approx(expected, rel=1e-4)


# --- plumbing -------------------------------------------------------------


def _selective_config(key=None, **selective):
    return CompressedCacheConfig(
        key=key
        or XKVCacheConfig(
            layer_group_size=1, num_layers=1, svd_backend="linalg"
        ),
        value=BaselineCacheConfig(),
        selective=SelectiveCacheConfig(**selective),
    )


def test_padding_mask_reaches_the_key_cache(monkeypatch):
    torch.manual_seed(4)
    batch_size, num_heads, seq_len, head_dim, pad_len = 1, 2, 32, 16, 8
    cos, sin = rope_cos_sin(batch_size, seq_len, head_dim)
    keys = apply_model_rope(
        torch.randn(batch_size, num_heads, seq_len, head_dim), cos, sin
    )
    mask = _left_pad_mask(batch_size, pad_len, seq_len)
    cache = CompressedCache(
        config=_selective_config(enabled=False),
        cache_context={"padding_mask": mask},
        verbose=False,
    )

    seen = {}
    original = cache.key_cache.update

    def spy(key_states, layer_idx, cache_kwargs=None):
        seen["padding_mask"] = (cache_kwargs or {}).get("padding_mask")
        return original(key_states, layer_idx, cache_kwargs)

    monkeypatch.setattr(cache.key_cache, "update", spy)
    cache.update(keys, torch.randn_like(keys), 0, {"cos": cos, "sin": sin})

    assert seen["padding_mask"] is not None
    torch.testing.assert_close(seen["padding_mask"], mask)


def test_padding_mask_follows_eviction():
    torch.manual_seed(5)
    batch_size, num_heads, seq_len, head_dim, pad_len = 1, 2, 64, 16, 16
    keys = torch.randn(batch_size, num_heads, seq_len, head_dim)
    mask = _left_pad_mask(batch_size, pad_len, seq_len)
    cache = CompressedCache(
        config=CompressedCacheConfig(eviction_keep_ratio=0.5),
        cache_context={"padding_mask": mask},
        verbose=False,
    )
    keep = torch.arange(0, seq_len, 2)

    aligned = cache._aligned_padding_mask(keys.index_select(-2, keep), keep)
    torch.testing.assert_close(aligned, mask.index_select(1, keep))
    assert aligned.size(-1) == keep.numel()


def test_decode_steps_carry_no_padding_mask():
    cache = CompressedCache(
        config=CompressedCacheConfig(),
        cache_context={"padding_mask": torch.ones(1, 8, dtype=torch.bool)},
        verbose=False,
    )
    assert cache._aligned_padding_mask(torch.randn(1, 2, 1, 4), None) is None


# --- selective reconstruction ---------------------------------------------


def test_landmarks_average_only_real_tokens():
    torch.manual_seed(6)
    batch_size, num_heads, seq_len, head_dim, pad_len = 1, 2, 32, 8, 12
    cache = CompressedCache(
        config=_selective_config(
            enabled=True,
            token_budget=8,
            chunk_size=8,
            local_tokens=8,
            outlier_chunks=1,
        ),
        verbose=False,
    )
    keys = torch.randn(batch_size, num_heads, seq_len, head_dim)
    mask = _left_pad_mask(batch_size, pad_len, seq_len)

    _, landmarks, valid = cache.selective._build_chunk_landmarks(keys, mask)

    # Chunk 1 straddles the boundary: 4 pad tokens then 4 real ones.
    torch.testing.assert_close(
        landmarks[:, :, 1], keys[:, :, 12:16].mean(dim=2)
    )
    # Chunk 0 is entirely padding and contributes nothing.
    assert not valid[:, :, 0].any()


def test_padding_only_chunks_are_never_selected():
    torch.manual_seed(7)
    batch_size, num_heads, seq_len, head_dim, pad_len = 1, 2, 64, 8, 32
    cache = CompressedCache(
        config=_selective_config(
            key=BaselineCacheConfig(),
            enabled=True,
            token_budget=16,
            chunk_size=8,
            local_tokens=8,
            outlier_chunks=1,
        ),
        verbose=False,
    )
    # Padding that looks exactly like an attention sink: one repeated,
    # high-norm direction that any query lines up with.
    keys = torch.randn(batch_size, num_heads, seq_len, head_dim)
    keys[:, :, :pad_len] = 20.0
    values = torch.randn_like(keys)
    mask = _left_pad_mask(batch_size, pad_len, seq_len)

    cache.selective.store_landmarks(0, keys, values, padding_mask=mask)
    state = cache.selective_layers[0]

    assert state.landmark_valid is not None
    assert not state.landmark_valid[..., : state.landmark_count].all()
    assert (state.outliers >= pad_len // 8).all()

    query = torch.full((batch_size, num_heads, 1, head_dim), 20.0)
    positions = cache.selective.select_positions(0, query)
    prefix = positions[..., : -state.exact_positions.size(-1)]
    assert (prefix >= pad_len).all()


def test_selective_rejects_right_padding():
    cache = CompressedCache(
        config=_selective_config(
            enabled=True,
            token_budget=8,
            chunk_size=8,
            local_tokens=8,
            outlier_chunks=1,
        ),
        verbose=False,
    )
    keys = torch.randn(1, 2, 32, 8)
    mask = torch.ones(1, 32, dtype=torch.bool)
    mask[:, -8:] = False

    with pytest.raises(ValueError, match="left-padded"):
        cache.selective.store_landmarks(
            0, keys, torch.randn_like(keys), padding_mask=mask
        )


def test_xkv_rejects_right_padding():
    torch.manual_seed(8)
    cos, sin = rope_cos_sin(1, 32, 16)
    keys = apply_model_rope(torch.randn(1, 2, 32, 16), cos, sin)
    mask = torch.ones(1, 32, dtype=torch.bool)
    mask[:, -8:] = False

    with pytest.raises(ValueError, match="left-padded"):
        _build_cache().update(
            keys, 0, {"cos": cos, "sin": sin, "padding_mask": mask}
        )


# --- compression ratios measure real tokens only --------------------------


def test_compression_ratio_uses_each_sequences_own_length():
    """Pad lengths differ per row, so the per-batch ratios must too."""
    torch.manual_seed(9)
    num_heads, seq_len, head_dim = 2, 64, 16
    pads = (8, 32)
    cos, sin = rope_cos_sin(2, seq_len, head_dim)
    keys = apply_model_rope(
        torch.randn(2, num_heads, seq_len, head_dim), cos, sin
    )
    mask = torch.ones(2, seq_len, dtype=torch.bool)
    for row, pad_len in enumerate(pads):
        mask[row, :pad_len] = False

    cache = _build_cache(comp_ratio=4.0)
    cache.update(keys, 0, {"cos": cos, "sin": sin, "padding_mask": mask})

    valid_lens = cache.group_states[0].valid_lens
    assert valid_lens.tolist() == [seq_len - pads[0], seq_len - pads[1]]

    rank = cache.layer_states[0].packed_right.shape[-2]
    flat_dim = num_heads * head_dim
    expected = sum(
        (seq_len - pad_len)
        * flat_dim
        / (rank * ((seq_len - pad_len) + flat_dim))
        for pad_len in pads
    ) / len(pads)
    assert cache.comp_ratio == pytest.approx(expected, rel=1e-4)


def test_turboquant_compression_ratio_excludes_padding():
    from cache.backends.turboquant import TurboQuantCache

    torch.manual_seed(10)
    batch_size, num_heads, seq_len, head_dim, pad_len = 2, 2, 64, 32, 16
    cos, sin = rope_cos_sin(batch_size, seq_len, head_dim)
    keys = apply_model_rope(
        torch.randn(batch_size, num_heads, seq_len, head_dim), cos, sin
    )
    mask = _left_pad_mask(batch_size, pad_len, seq_len)

    def ratio(padding_mask):
        cache = TurboQuantCache(compressor_bits=4, rope_cache=SharedRopeCache())
        kwargs = {"cos": cos, "sin": sin}
        if padding_mask is not None:
            kwargs["padding_mask"] = padding_mask
        cache.update(keys, 0, kwargs)
        return cache.comp_ratio

    valid = seq_len - pad_len
    # Quantisation cost is per row, so with padding off both sides the ratio
    # is a property of the quantiser and padding cannot move it.
    assert ratio(mask) == pytest.approx(ratio(None), rel=1e-6)
    assert cache_original_tokens(keys, mask) == batch_size * valid


def cache_original_tokens(keys, mask):
    from cache.backends.turboquant import TurboQuantCache

    cache = TurboQuantCache(compressor_bits=4, rope_cache=SharedRopeCache())
    cache.update(keys, 0, {"padding_mask": mask})
    return cache.layers[0].original_token_count


# --- eviction -------------------------------------------------------------


def test_eviction_keeps_each_sequences_own_sinks():
    from cache.eviction import EvictionPolicy

    torch.manual_seed(12)
    seq_len, pads = 128, (32, 8)
    policy = EvictionPolicy(keep_ratio=0.5, key_cache=None)
    mask = torch.ones(len(pads), seq_len, dtype=torch.bool)
    for row, pad_len in enumerate(pads):
        mask[row, :pad_len] = False
    importance = torch.rand(len(pads), 4, seq_len)

    keep = set(policy._build_keep_positions(importance, seq_len, mask).tolist())

    # Every sequence keeps the first `sink_tokens` tokens it actually has.
    for pad_len in pads:
        sinks = range(pad_len, pad_len + policy.sink_tokens)
        assert set(sinks) <= keep, f"missing sinks for pad_len={pad_len}"
    # Positions that are padding in every sequence are never kept.
    assert not any(pos < min(pads) for pos in keep)


def test_eviction_budget_ignores_padding():
    from cache.eviction import EvictionPolicy

    torch.manual_seed(13)
    # Sized so the keep budget binds rather than the forced sink/local floor.
    seq_len, pad_len = 512, 256
    policy = EvictionPolicy(keep_ratio=0.5, key_cache=None)
    mask = _left_pad_mask(1, pad_len, seq_len)
    importance = torch.rand(1, 4, seq_len)

    keep = policy._build_keep_positions(importance, seq_len, mask)

    # Half of the real tokens, not half of the padded length.
    assert keep.numel() == (seq_len - pad_len) // 2
    assert (keep >= pad_len).all()


def test_eviction_unchanged_without_padding():
    from cache.eviction import EvictionPolicy

    torch.manual_seed(14)
    seq_len = 512
    policy = EvictionPolicy(keep_ratio=0.5, key_cache=None)
    importance = torch.rand(1, 4, seq_len)

    keep = policy._build_keep_positions(importance, seq_len)

    assert keep.numel() == seq_len // 2
    assert set(range(policy.sink_tokens)) <= set(keep.tolist())
    assert set(range(seq_len - policy.local_tokens, seq_len)) <= set(
        keep.tolist()
    )


def test_padding_importance_does_not_leak_across_sequences():
    from cache.eviction import EvictionPolicy

    seq_len, pad_len = 64, 32
    policy = EvictionPolicy(keep_ratio=1.0, key_cache=None)
    mask = torch.ones(2, seq_len, dtype=torch.bool)
    mask[0, :pad_len] = False

    # Sequence 0's padding claims huge importance at a position where
    # sequence 1 holds a real, worthless token.
    importance = torch.zeros(2, 1, seq_len)
    importance[0, :, :pad_len] = 100.0

    score = policy._importance_score(importance, seq_len, mask)

    assert torch.count_nonzero(score[:pad_len]) == 0


def test_eviction_drops_the_same_tokens_with_or_without_padding():
    """Padding must not change which real tokens survive eviction."""
    from cache.eviction import EvictionPolicy

    torch.manual_seed(15)
    valid_len, pad_len = 4096, 2048
    seq_len = valid_len + pad_len
    importance = torch.rand(1, 4, valid_len)

    padded = EvictionPolicy(keep_ratio=0.25, key_cache=None)
    padded_importance = torch.zeros(1, 4, seq_len)
    padded_importance[..., pad_len:] = importance
    padded.set_value_importance(0, padded_importance)
    keys = torch.randn(1, 4, seq_len, 8)
    padded.apply(keys, keys.clone(), 0, _left_pad_mask(1, pad_len, seq_len))

    plain = EvictionPolicy(keep_ratio=0.25, key_cache=None)
    plain.set_value_importance(0, importance)
    unpadded = torch.randn(1, 4, valid_len, 8)
    plain.apply(unpadded, unpadded.clone(), 0)

    torch.testing.assert_close(
        padded.kept_positions[0] - pad_len, plain.kept_positions[0]
    )
    assert padded.compression_ratio == plain.compression_ratio
