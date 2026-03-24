import torch

from cache.key_cache import KMeansLRKCache


def build_cache(**kwargs):
    return KMeansLRKCache(
        decomposition_method="svd",
        rank_selection="energy",
        energy_threshold=1.0,
        kmeans_cluster_size=2.0,
        kmeans_n_iter=4,
        local_window=0,
        **kwargs,
    )


def run_roundtrip(cache, prefill_keys, decode_keys, layer_idx=0):
    returned_prefill = cache.update(prefill_keys, layer_idx, cache_kwargs={})
    torch.testing.assert_close(returned_prefill, prefill_keys)
    cache.update_events()
    recon_keys = cache.update(decode_keys, layer_idx, cache_kwargs={})
    expected_keys = torch.cat([prefill_keys, decode_keys], dim=-2)
    torch.testing.assert_close(
        recon_keys,
        expected_keys,
        atol=1e-4,
        rtol=1e-4,
    )


def test_kmeans_concat_heads_roundtrip():
    torch.manual_seed(0)
    prefill_keys = torch.randn(2, 3, 6, 4)
    decode_keys = torch.randn(2, 3, 1, 4)
    cache = build_cache()

    run_roundtrip(cache, prefill_keys, decode_keys)


def test_kmeans_per_head_roundtrip():
    torch.manual_seed(1)
    prefill_keys = torch.randn(2, 3, 6, 4)
    decode_keys = torch.randn(2, 3, 1, 4)
    cache = build_cache(kmeans_per_head=True)

    run_roundtrip(cache, prefill_keys, decode_keys)


def test_kmeans_per_head_metadata_tracks_batch_ops():
    torch.manual_seed(2)
    prefill_keys = torch.randn(2, 3, 5, 4)
    cache = build_cache(kmeans_per_head=True)

    cache.update(prefill_keys, layer_idx=0, cache_kwargs={})
    cache.update_events()

    reorder = torch.tensor([1, 0])
    cache.reorder_cache(reorder)
    expected_prefix = prefill_keys.index_select(0, reorder)

    cache.batch_repeat_interleave(2)
    expected_prefix = expected_prefix.repeat_interleave(2, dim=0)

    select = torch.tensor([3, 1])
    cache.batch_select_indices(select)
    expected_prefix = expected_prefix[select]

    decode_keys = torch.randn(
        expected_prefix.size(0),
        expected_prefix.size(1),
        1,
        expected_prefix.size(-1),
    )
    recon_keys = cache.update(decode_keys, layer_idx=0, cache_kwargs={})
    expected_keys = torch.cat([expected_prefix, decode_keys], dim=-2)

    torch.testing.assert_close(
        recon_keys,
        expected_keys,
        atol=1e-4,
        rtol=1e-4,
    )
