import warnings

import torch

from cache.cache import CompressedCache
from cache.keys import KEY_CACHE_CLASSES, XKVKeysCache


def build_cache(**kwargs):
    return CompressedCache(
        config=None,
        key_cache_kwargs={
            "cache_type": "xkv",
            "decomposition_method": "svd",
            "rank_selection": "energy",
            "energy_threshold": 1.0,
            "local_window": 0,
            "layer_group_size": 2,
            "num_layers": 3,
            **kwargs,
        },
        value_cache_kwargs={"cache_type": "baseline"},
        verbose=False,
    )


def test_xkv_is_registered():
    assert KEY_CACHE_CLASSES["xkv"] is XKVKeysCache


def test_xkv_roundtrip_handles_terminal_partial_group():
    torch.manual_seed(0)
    cache = build_cache()

    prefill_keys = [
        torch.randn(2, 3, 4, 5),
        torch.randn(2, 3, 4, 5),
        torch.randn(2, 3, 4, 5),
    ]
    decode_keys = [
        torch.randn(2, 3, 1, 5),
        torch.randn(2, 3, 1, 5),
        torch.randn(2, 3, 1, 5),
    ]

    for layer_idx, key_states in enumerate(prefill_keys):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            returned_keys, returned_values = cache.update(
                key_states,
                key_states,
                layer_idx,
                cache_kwargs={},
            )
        torch.testing.assert_close(returned_keys, key_states)
        torch.testing.assert_close(returned_values, key_states)

    cache.update_events()
    cache_position = torch.tensor([prefill_keys[0].size(-2)], dtype=torch.long)

    for layer_idx, key_states in enumerate(decode_keys):
        recon_keys, recon_values = cache.update(
            key_states,
            key_states,
            layer_idx,
            cache_kwargs={"cache_position": cache_position},
        )
        expected_keys = torch.cat([prefill_keys[layer_idx], key_states], dim=-2)
        expected_values = torch.cat(
            [prefill_keys[layer_idx], key_states], dim=-2
        )
        torch.testing.assert_close(recon_keys, expected_keys)
        torch.testing.assert_close(recon_values, expected_values)
