import torch

from cache.rope import SharedRopeCache
from cache.core import CompressedCache
from cache.selective import SelectiveLayerState
from cache.config import (
    BaselineCacheConfig,
    CompressedCacheConfig,
    SelectiveCacheConfig,
    XKVCacheConfig,
)
from cache.backends.xkv import XKVKeysCache
from cache.backends.mlp_values import MLPValueCache
from utils.rope import inverse_rope

from tests.helpers import apply_model_rope, gather_tokens, rope_cos_sin


def test_typed_config_builds_selective_layer_state():
    torch.manual_seed(1)
    config = CompressedCacheConfig(
        key=XKVCacheConfig(
            layer_group_size=1,
            num_layers=1,
            svd_backend="linalg",
        ),
        value=BaselineCacheConfig(),
        selective=SelectiveCacheConfig(
            enabled=True,
            token_budget=32,
            chunk_size=8,
            local_tokens=8,
            outlier_chunks=2,
        ),
    )
    cache = CompressedCache(config=config, verbose=False)
    keys = torch.randn(1, 2, 64, 16)
    values = torch.randn_like(keys)

    cache.selective.store_landmarks(0, keys, values)

    state = cache.selective_layers[0]
    assert isinstance(state, SelectiveLayerState)
    assert state.prompt_len == 64
    assert state.true_prompt_len == 64
    assert state.landmark_count == 5
    assert state.exact_keys.shape == state.exact_values.shape
    assert (
        cache.key_cache.layer_states[0].selective_overhead_bytes
        == state.key_overhead_nbytes
    )


def test_xkv_grouped_and_selected_reconstruction_match():
    torch.manual_seed(2)
    batch_size, num_heads, seq_len, head_dim = 1, 2, 64, 16
    cos, sin = rope_cos_sin(batch_size, seq_len, head_dim)
    cache = XKVKeysCache(
        layer_group_size=2,
        num_layers=2,
        xkv_svd_backend="linalg",
        comp_ratio=2.0,
        rope_cache=SharedRopeCache(),
    )

    for layer_idx in range(2):
        keys = torch.randn(batch_size, num_heads, seq_len, head_dim)
        cache.update(
            apply_model_rope(keys, cos, sin),
            layer_idx,
            {
                "cos": cos,
                "sin": sin,
                "cache_position": torch.arange(seq_len),
            },
        )

    assert cache.layer_states[0].group_last_layer == 1
    assert cache.layer_states[1].group_last_layer == 1
    assert cache.group_states[1].layer_indices == (0, 1)
    assert cache.layer_states[0].packed_right is not None
    assert cache.layer_states[1].packed_right is not None

    positions = torch.randint(0, seq_len, (batch_size, num_heads, 17))
    for layer_idx in range(2):
        full = cache._reconstruct_keys(cache.layers[layer_idx].tensor, layer_idx)
        selected = cache.retrieve_selected(layer_idx, positions)
        torch.testing.assert_close(
            selected,
            gather_tokens(full, positions),
            atol=2e-5,
            rtol=2e-5,
        )


def test_mlp_value_selected_reconstruction_matches_full_reconstruction():
    torch.manual_seed(3)
    batch_size, num_heads, seq_len, head_dim = 1, 2, 256, 16
    cos, sin = rope_cos_sin(batch_size, seq_len, head_dim)
    unroped_keys = torch.randn(batch_size, num_heads, seq_len, head_dim)
    keys = apply_model_rope(unroped_keys, cos, sin)
    values = torch.randn_like(keys)
    cache = MLPValueCache(
        target_cr=2.0,
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

    layer = cache.layers[0]
    full = layer.decompress(
        inverse_rope(keys, cos.unsqueeze(1), sin.unsqueeze(1))
    )
    positions = torch.randint(0, seq_len, (batch_size, num_heads, 29))
    selected_keys = gather_tokens(keys, positions)
    selected = cache.retrieve_selected(selected_keys, positions, 0)

    torch.testing.assert_close(
        selected,
        gather_tokens(full, positions),
        atol=2e-5,
        rtol=2e-5,
    )
