"""Interaction between the xKV key cache and the MLP value cache."""

import pytest
import torch

from cache.cache import CompressedCache
from cache.config import (
    CompressedCacheConfig,
    MLPValueCacheConfig,
    XKVCacheConfig,
)
from cache.values import MLPValueLayer
from utils.rope import inverse_rope

from tests.helpers import apply_model_rope, rope_cos_sin


BATCH, HEADS, SEQ_LEN, HEAD_DIM = 1, 2, 64, 16


def _build_cache(group_size, num_layers, **key_kwargs):
    return CompressedCache(
        config=CompressedCacheConfig(
            key=XKVCacheConfig(
                layer_group_size=group_size,
                num_layers=num_layers,
                svd_backend="linalg",
                compression_ratio=2.0,
                **key_kwargs,
            ),
            value=MLPValueCacheConfig(
                target_compression_ratio=2.0,
                num_epochs=0,
            ),
        ),
        verbose=False,
    )


def _spy_on_training(monkeypatch, cache):
    """Record (layer index, keys) for every value-MLP training call."""
    calls = []
    original = MLPValueLayer.train_mlp

    def spy(self, keys, padding_mask):
        calls.append((cache.value_cache.layers.index(self), keys.detach()))
        return original(self, keys, padding_mask)

    monkeypatch.setattr(MLPValueLayer, "train_mlp", spy)
    return calls


def _prefill_layer(cache, layer_idx, cos, sin, keys=None):
    keys = (
        apply_model_rope(
            torch.randn(BATCH, HEADS, SEQ_LEN, HEAD_DIM), cos, sin
        )
        if keys is None
        else keys
    )
    values = torch.randn(BATCH, HEADS, SEQ_LEN, HEAD_DIM)
    cache.update(
        keys,
        values,
        layer_idx,
        {
            "cos": cos,
            "sin": sin,
            "cache_position": torch.arange(SEQ_LEN),
            "padding_mask": torch.ones(BATCH, SEQ_LEN, dtype=torch.bool),
        },
    )
    return keys


@pytest.mark.parametrize(
    ("group_size", "num_layers"),
    [(1, 4), (2, 4), (4, 4)],
)
def test_value_mlp_trains_only_once_its_key_group_is_decomposed(
    monkeypatch,
    group_size,
    num_layers,
):
    torch.manual_seed(30)
    cos, sin = rope_cos_sin(BATCH, SEQ_LEN, HEAD_DIM)
    cache = _build_cache(group_size, num_layers)
    calls = _spy_on_training(monkeypatch, cache)

    for layer_idx in range(num_layers):
        _prefill_layer(cache, layer_idx, cos, sin)

        group_start = (layer_idx // group_size) * group_size
        group_last = min(group_start + group_size - 1, num_layers - 1)
        # Training is deferred until the whole group has been decomposed.
        expected = group_last if layer_idx == group_last else group_start - 1
        trained = [idx for idx, _ in calls]
        assert trained == list(range(expected + 1)), (
            f"after layer {layer_idx} trained {trained}, "
            f"expected layers 0..{expected}"
        )

    assert [idx for idx, _ in calls] == list(range(num_layers))


def test_deferred_layers_are_held_until_the_group_boundary(monkeypatch):
    """Non-terminal layers park their values instead of training."""
    torch.manual_seed(31)
    cos, sin = rope_cos_sin(BATCH, SEQ_LEN, HEAD_DIM)
    cache = _build_cache(group_size=2, num_layers=2)
    calls = _spy_on_training(monkeypatch, cache)

    _prefill_layer(cache, 0, cos, sin)
    assert calls == []
    assert list(cache._deferred_value_updates) == [0]

    _prefill_layer(cache, 1, cos, sin)
    assert [idx for idx, _ in calls] == [0, 1]
    assert cache._deferred_value_updates == {}


@pytest.mark.parametrize("group_size", (1, 2))
def test_value_mlp_trains_on_decomposed_unroped_keys(monkeypatch, group_size):
    torch.manual_seed(32)
    num_layers = 2
    cos, sin = rope_cos_sin(BATCH, SEQ_LEN, HEAD_DIM)
    cache = _build_cache(group_size, num_layers)
    calls = _spy_on_training(monkeypatch, cache)

    exact_roped = [
        _prefill_layer(cache, layer_idx, cos, sin)
        for layer_idx in range(num_layers)
    ]

    training_keys = dict(calls)
    assert sorted(training_keys) == list(range(num_layers))

    for layer_idx in range(num_layers):
        actual = training_keys[layer_idx]
        reconstructed_roped = cache.key_cache.get_reconstructed_keys_only(
            layer_idx
        )
        expected = inverse_rope(
            reconstructed_roped, cos.unsqueeze(1), sin.unsqueeze(1)
        )
        exact_unroped = inverse_rope(
            exact_roped[layer_idx], cos.unsqueeze(1), sin.unsqueeze(1)
        )

        # Trained against the lossy reconstruction, in the un-roped frame.
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
        assert not torch.allclose(
            actual, reconstructed_roped, atol=1e-2, rtol=1e-2
        ), "keys handed to the MLP were still roped"
        assert not torch.allclose(
            actual, exact_unroped, atol=1e-5, rtol=1e-5
        ), "keys handed to the MLP were the exact keys, not the decomposed ones"
        # ...but still tracking them: better than predicting nothing.
        error = (actual - exact_unroped).norm() / exact_unroped.norm()
        assert error < 1.0, f"layer {layer_idx} reconstruction error {error}"
