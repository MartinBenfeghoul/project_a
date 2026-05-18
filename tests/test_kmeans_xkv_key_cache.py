import pytest
import torch

from cache.keys import KEY_CACHE_CLASSES, KMeansXKVKeysCache
from utils.turboquant import is_quantised_factor


def build_cache(**kwargs):
    return KMeansXKVKeysCache(
        decomposition_method="svd",
        rank_selection="energy",
        energy_threshold=1.0,
        kmeans_cluster_size=2.0,
        kmeans_n_iter=4,
        local_window=0,
        layer_group_size=2,
        num_layers=3,
        **kwargs,
    )


def test_kmeans_xkv_is_registered():
    assert KEY_CACHE_CLASSES["kmeans_xkv"] is KMeansXKVKeysCache


def test_kmeans_xkv_rejects_non_svd_decomposition_method():
    with pytest.raises(NotImplementedError, match="supports .*'svd' only"):
        build_cache(decomposition_method="lora")


def test_kmeans_xkv_roundtrip_supports_quantised_factors():
    torch.manual_seed(0)
    cache = build_cache(quantise_a=True, quantise_b=True, compressor_bits=8)
    prefill_keys = [torch.randn(2, 3, 4, 5) for _ in range(3)]

    for layer_idx, key_states in enumerate(prefill_keys):
        returned_keys = cache.update(key_states, layer_idx, cache_kwargs={})
        torch.testing.assert_close(returned_keys, key_states)

    shared_factor = cache.shared_a[1][0][0]["factor"]
    right_factor = cache.lr_keys[0][0][0]["factor"]
    assert is_quantised_factor(shared_factor)
    assert is_quantised_factor(right_factor)

    recon_keys = cache.get_reconstructed_keys_only(0)
    assert recon_keys.shape == prefill_keys[0].shape
    assert torch.isfinite(recon_keys).all()


def test_kmeans_xkv_waits_for_group_end_before_decomposing():
    torch.manual_seed(0)
    cache = build_cache()
    prefill_keys = [torch.randn(2, 3, 4, 5) for _ in range(2)]

    returned_keys = cache.update(prefill_keys[0], layer_idx=0, cache_kwargs={})
    torch.testing.assert_close(returned_keys, prefill_keys[0])
    assert 0 not in cache.lr_keys
    assert 0 not in cache.shared_a
    assert cache.layers[0].tensor.size(-2) == prefill_keys[0].size(-2)

    returned_keys = cache.update(prefill_keys[1], layer_idx=1, cache_kwargs={})
    torch.testing.assert_close(returned_keys, prefill_keys[1])
    assert set(cache.shared_a) == {1}
    assert 0 in cache.lr_keys and 1 in cache.lr_keys
    assert cache.group_metadata[0]["group_last_layer"] == 1
    assert cache.group_metadata[1]["group_last_layer"] == 1
    assert cache.layers[0].tensor.size(-2) == 0
    assert cache.layers[1].tensor.size(-2) == 0


def test_kmeans_xkv_roundtrip_handles_terminal_partial_group():
    torch.manual_seed(1)
    cache = build_cache()
    prefill_keys = [torch.randn(2, 3, 4, 5) for _ in range(3)]
    decode_keys = [torch.randn(2, 3, 1, 5) for _ in range(3)]

    for layer_idx, key_states in enumerate(prefill_keys):
        returned_keys = cache.update(key_states, layer_idx, cache_kwargs={})
        torch.testing.assert_close(returned_keys, key_states)

    assert set(cache.shared_a) == {1, 2}
    torch.testing.assert_close(
        cache.get_reconstructed_keys_only(0),
        prefill_keys[0],
        atol=1e-4,
        rtol=1e-4,
    )

    cache.update_events()
    cache_position = torch.tensor([prefill_keys[0].size(-2)], dtype=torch.long)

    for layer_idx, key_states in enumerate(decode_keys):
        recon_keys = cache.update(
            key_states,
            layer_idx,
            cache_kwargs={"cache_position": cache_position},
        )
        expected_keys = torch.cat([prefill_keys[layer_idx], key_states], dim=-2)
        torch.testing.assert_close(
            recon_keys,
            expected_keys,
            atol=1e-4,
            rtol=1e-4,
        )


def test_kmeans_xkv_local_window_preserves_suffix():
    torch.manual_seed(2)
    cache = build_cache(local_window=2, num_layers=2)
    prefill_keys = [torch.randn(2, 3, 6, 4) for _ in range(2)]
    decode_keys = [torch.randn(2, 3, 1, 4) for _ in range(2)]

    for layer_idx, key_states in enumerate(prefill_keys):
        cache.update(key_states, layer_idx, cache_kwargs={})

    assert cache.layers[0].tensor.size(-2) == 2
    assert cache.layers[1].tensor.size(-2) == 2

    cache.update_events()
    cache_position = torch.tensor([prefill_keys[0].size(-2)], dtype=torch.long)

    for layer_idx, key_states in enumerate(decode_keys):
        recon_keys = cache.update(
            key_states,
            layer_idx,
            cache_kwargs={"cache_position": cache_position},
        )
        expected_keys = torch.cat([prefill_keys[layer_idx], key_states], dim=-2)
        torch.testing.assert_close(
            recon_keys,
            expected_keys,
            atol=1e-4,
            rtol=1e-4,
        )


def test_kmeans_xkv_update_events_reports_segment_size_summary():
    cache = build_cache()
    cache.lr_keys = {
        0: [
            [{"range": (0, 2)}, {"range": (2, 5)}],
            [{"range": (0, 4)}],
        ],
        1: [
            [{"range": (0, 1)}, {"range": (1, 5)}],
            [{"range": (0, 3)}, {"range": (3, 4)}],
        ],
    }

    assert cache.update_events() == {
        "segment_size_min": 1.0,
        "segment_size_median": 3.0,
        "segment_size_max": 4.0,
        "segment_size_std_mean": 0.75,
    }


def test_kmeans_xkv_batch_ops_are_unimplemented():
    cache = build_cache()
    with pytest.raises(NotImplementedError):
        cache.reorder_cache(torch.tensor([0]))
    with pytest.raises(NotImplementedError):
        cache.batch_repeat_interleave(2)
    with pytest.raises(NotImplementedError):
        cache.batch_select_indices(torch.tensor([0]))
