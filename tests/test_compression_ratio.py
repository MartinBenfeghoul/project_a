import torch

from cache.base import SharedRopeCache
from cache.keys import XKVKeysCache
from cache.values import MLPValueCache
from numerics.quantisation import factor_nbytes

from tests.helpers import apply_model_rope, rope_cos_sin


TARGET_COMPRESSION_RATIO = 2.0
RELATIVE_TOLERANCE = 0.10
DECODE_TOKENS = 8


def _xkv_actual_compression_ratio(
    cache: XKVKeysCache,
    *,
    batch_size: int,
    num_heads: int,
    head_dim: int,
) -> float:
    logical_len = cache.get_seq_length(0)
    group_state = next(iter(cache.group_states.values()))
    dtype_size = group_state.packed_shared.element_size()
    original = (
        len(cache.layers)
        * batch_size
        * num_heads
        * logical_len
        * head_dim
        * dtype_size
    )
    factor_bytes = sum(
        factor_nbytes(segment.factor)
        for state in cache.group_states.values()
        for segments in state.shared_segments
        for segment in segments
    ) + sum(
        factor_nbytes(segment.factor)
        for state in cache.layer_states.values()
        for segments in state.segments
        for segment in segments
    )
    suffix_bytes = sum(
        layer.tensor.numel() * layer.tensor.element_size()
        for layer in cache.layers
    )
    return original / (factor_bytes + suffix_bytes)


def _mlp_actual_compression_ratio(cache: MLPValueCache) -> float:
    original_bytes = 0.0
    compressed_bytes = 0.0
    for layer in cache.layers:
        dtype_size = layer.tensor.element_size()
        suffix_bytes = layer.tensor.numel() * dtype_size
        original_bytes += (
            layer.original_token_count
            * layer.num_heads
            * layer.head_dim
            * dtype_size
            + suffix_bytes
        )
        num_stored = layer.indices.numel()
        compressed_bytes += (
            layer._num_params * dtype_size
            + layer.residual_storage_nbytes(
                num_stored,
                layer.head_dim,
                layer.tensor.dtype,
            )
            + num_stored * layer.indices.element_size()
            + suffix_bytes
        )
    return original_bytes / compressed_bytes


def _assert_near_target(actual: float) -> None:
    assert abs(actual - TARGET_COMPRESSION_RATIO) <= (
        TARGET_COMPRESSION_RATIO * RELATIVE_TOLERANCE
    ), f"achieved compression ratio {actual:.4f}"


def test_xkv_key_ratio_is_near_target_after_prefill_and_decode():
    torch.manual_seed(5)
    batch_size, num_heads, seq_len, head_dim = 1, 2, 256, 16
    cos, sin = rope_cos_sin(batch_size, seq_len, head_dim)
    cache = XKVKeysCache(
        layer_group_size=2,
        num_layers=2,
        xkv_svd_backend="linalg",
        comp_ratio=TARGET_COMPRESSION_RATIO,
        rope_cache=SharedRopeCache(),
    )

    for layer_idx in range(2):
        keys = torch.randn(batch_size, num_heads, seq_len, head_dim)
        cache.update(
            apply_model_rope(keys, cos, sin),
            layer_idx,
            {"cos": cos, "sin": sin},
        )

    prefill_ratio = _xkv_actual_compression_ratio(
        cache,
        batch_size=batch_size,
        num_heads=num_heads,
        head_dim=head_dim,
    )
    _assert_near_target(prefill_ratio)
    assert cache.comp_ratio == prefill_ratio

    cache.update_events()
    for layer_idx in range(2):
        cache.append_decode(
            torch.randn(
                batch_size,
                num_heads,
                DECODE_TOKENS,
                head_dim,
            ),
            layer_idx,
        )
    decode_ratio = _xkv_actual_compression_ratio(
        cache,
        batch_size=batch_size,
        num_heads=num_heads,
        head_dim=head_dim,
    )
    _assert_near_target(decode_ratio)
    assert decode_ratio < prefill_ratio


def test_mlp_value_ratio_is_near_target_after_prefill_and_decode():
    torch.manual_seed(6)
    batch_size, num_heads, seq_len, head_dim = 1, 2, 256, 16
    cos, sin = rope_cos_sin(batch_size, seq_len, head_dim)
    keys = apply_model_rope(
        torch.randn(batch_size, num_heads, seq_len, head_dim),
        cos,
        sin,
    )
    values = torch.randn_like(keys)
    cache = MLPValueCache(
        target_cr=TARGET_COMPRESSION_RATIO,
        num_epochs=0,
        rope_cache=SharedRopeCache(),
    )
    cache.update(
        values,
        0,
        {
            "keys": keys,
            "cos": cos,
            "sin": sin,
            "padding_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
        },
    )

    prefill_ratio = _mlp_actual_compression_ratio(cache)
    _assert_near_target(prefill_ratio)
    assert cache.calc_compression_ratio() == prefill_ratio

    cache.append_decode(
        torch.randn(batch_size, num_heads, DECODE_TOKENS, head_dim),
        0,
    )
    decode_ratio = _mlp_actual_compression_ratio(cache)
    _assert_near_target(decode_ratio)
    assert decode_ratio < prefill_ratio
