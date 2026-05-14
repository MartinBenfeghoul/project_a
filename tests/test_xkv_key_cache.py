import warnings

import pytest
import torch

from cache.cache import CompressedCache
from cache.keys import KEY_CACHE_CLASSES, XKVKeysCache
from utils.matrix_decomposition import decompose_grouped_xkv_to_segment_store
from utils.turboquant import is_quantised_factor


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


def test_xkv_rejects_non_svd_decomposition_method():
    with pytest.raises(NotImplementedError, match="supports .*'svd' only"):
        build_cache(decomposition_method="lora")


def test_grouped_xkv_helper_reconstructs_each_segment():
    torch.manual_seed(0)
    grouped = torch.randn(2, 1, 5, 4)
    segment_ranges = [[(0, 2), (2, 5)], [(0, 3), (3, 5)]]

    layer_segments = decompose_grouped_xkv_to_segment_store(
        grouped,
        segment_ranges=segment_ranges,
        rank_selection="energy",
        energy_threshold=1.0,
    )

    assert len(layer_segments) == grouped.size(0)
    for batch_idx, batch_segments in enumerate(layer_segments):
        assert len(batch_segments) == len(segment_ranges[batch_idx])
        for segment, expected_range in zip(
            batch_segments, segment_ranges[batch_idx]
        ):
            assert segment["range"] == expected_range
            assert set(segment) == {"range", "factors"}
            A, B = segment["factors"]
            expected = grouped[
                batch_idx, ..., expected_range[0] : expected_range[1], :
            ]
            torch.testing.assert_close(A @ B, expected)


def test_xkv_helper_can_quantise_factors():
    torch.manual_seed(0)
    grouped = torch.randn(2, 1, 5, 4)
    segment_ranges = [[(0, 5)], [(0, 5)]]

    layer_segments = decompose_grouped_xkv_to_segment_store(
        grouped,
        segment_ranges=segment_ranges,
        rank_selection="energy",
        energy_threshold=1.0,
        quantise_a=True,
        quantise_b=True,
        compressor_bits=8,
    )

    for batch_segments in layer_segments:
        A, B = batch_segments[0]["factors"]
        assert is_quantised_factor(A)
        assert is_quantised_factor(B)


def test_xkv_roundtrip_supports_quantised_factors():
    torch.manual_seed(0)
    cache = build_cache(quantise_a=True, quantise_b=True, compressor_bits=8)
    prefill_keys = [torch.randn(2, 3, 4, 5) for _ in range(3)]

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

    shared_factor = cache.shared_a[1][0][0]["factor"]
    right_factor = cache.lr_keys[0][0][0]["factor"]
    assert is_quantised_factor(shared_factor)
    assert is_quantised_factor(right_factor)

    recon_keys = cache.get_reconstructed_keys_only(0)
    assert recon_keys.shape == prefill_keys[0].shape
    assert torch.isfinite(recon_keys).all()


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


def test_xkv_update_events_reports_segment_size_summary():
    cache = build_cache()
    cache.key_cache.lr_keys = {
        0: [[{"range": (0, 4)}], [{"range": (0, 6)}]],
        1: [[{"range": (0, 5)}], [{"range": (0, 7)}]],
    }

    assert cache.update_events() == {
        "segment_size_min": 4.0,
        "segment_size_median": 5.5,
        "segment_size_max": 7.0,
        "segment_size_std_mean": 0.0,
    }
