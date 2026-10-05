"""Compression ratios are measured against the uncompressed cache."""

import pytest
import torch

from cache import accounting
from cache.eviction import EvictionPolicy


class _StubBackend:
    def __init__(self, ratio: float | None):
        self._ratio = ratio

    def calc_compression_ratio(self):
        return self._ratio


class _StubState:
    def __init__(self, original: int, overhead: int):
        self.original_key_nbytes = original
        self.key_overhead_nbytes = overhead
        self.exact_value_nbytes = 0


class _StubSelective:
    def __init__(self, layers=None):
        self.layers = layers or {}
        self.scorer_nbytes = 0


class _StubRope:
    nbytes = 0


class _StubEviction:
    def __init__(self, ratio: float):
        self.compression_ratio = ratio


def _ratio(key, value, selective=None, eviction=None):
    return accounting.compression_ratio(
        _StubBackend(key),
        _StubBackend(value),
        selective or _StubSelective(),
        _StubRope(),
        eviction,
    )


def test_eviction_multiplies_the_backend_ratio():
    assert _ratio(4.0, 4.0, eviction=_StubEviction(1.0)) == pytest.approx(4.0)
    assert _ratio(4.0, 4.0, eviction=_StubEviction(2.0)) == pytest.approx(8.0)
    # Omitting the policy entirely must not change the old behaviour.
    assert _ratio(4.0, 4.0) == pytest.approx(4.0)


def test_eviction_applies_when_only_one_backend_reports():
    assert _ratio(4.0, None, eviction=_StubEviction(2.0)) == pytest.approx(8.0)
    assert _ratio(None, 4.0, eviction=_StubEviction(2.0)) == pytest.approx(8.0)
    assert _ratio(None, None, eviction=_StubEviction(2.0)) is None


def test_selective_overhead_is_not_scaled_by_eviction():
    """Landmarks and exact keys cost the same however much was evicted."""
    selective = _StubSelective({0: _StubState(original=500, overhead=100)})
    # 1000 original bytes; eviction dropped half, so the backends saw 500 and
    # stored 500/4 = 125; the 100 bytes of selective state sit on top.
    assert _ratio(4.0, 4.0, selective, _StubEviction(2.0)) == pytest.approx(
        1000 / 225
    )
    # With no eviction the backends saw all 1000 and stored 250.
    assert _ratio(4.0, 4.0, selective, _StubEviction(1.0)) == pytest.approx(
        1000 / 350
    )


def test_policy_reports_no_saving_when_disabled():
    assert (
        EvictionPolicy(keep_ratio=1.0, key_cache=None).compression_ratio == 1.0
    )


@pytest.mark.parametrize("keep_ratio", [0.5, 0.25, 0.1])
def test_ratio_matches_the_real_drop_at_prompt_lengths_used(keep_ratio):
    """`keep_ratio` stands in for a measurement, so check it is the truth.

    It only holds once the budget clears the forced sink and local windows.
    Prompts here are >= 4k, well past that floor; this pins the assumption
    so it fails loudly if those windows ever grow.
    """
    torch.manual_seed(0)
    seq_len = 4096
    policy = EvictionPolicy(keep_ratio=keep_ratio, key_cache=None)
    policy.set_value_importance(0, torch.rand(1, 4, seq_len))
    keys = torch.randn(1, 4, seq_len, 8)

    policy.apply(keys, keys.clone(), 0)

    measured = seq_len / policy.kept_positions[0].numel()
    assert policy.compression_ratio == pytest.approx(measured, rel=1e-3)


def test_ratio_holds_under_padding():
    """The keep budget is sized on real tokens, so padding cannot skew it."""
    torch.manual_seed(1)
    seq_len, pad_len = 6144, 2048
    policy = EvictionPolicy(keep_ratio=0.25, key_cache=None)
    policy.set_value_importance(0, torch.rand(1, 4, seq_len))
    keys = torch.randn(1, 4, seq_len, 8)
    mask = torch.ones(1, seq_len, dtype=torch.bool)
    mask[:, :pad_len] = False

    policy.apply(keys, keys.clone(), 0, mask)

    kept = policy.kept_positions[0]
    # Real tokens dropped, not padded length dropped.
    measured = (seq_len - pad_len) / kept.numel()
    assert policy.compression_ratio == pytest.approx(measured, rel=1e-3)
    assert (kept >= pad_len).all()


# --- eviction must not leak into the backends' own budgets ----------------


def _budget_inputs(monkeypatch):
    """Record the (rows, target_ratio) each rank decision is made from."""
    import efficiency

    calls = []
    real = efficiency.adjust_rank

    def spy(num_rows, num_columns, target_ratio, *args, **kwargs):
        calls.append((num_rows, target_ratio))
        return real(num_rows, num_columns, target_ratio, *args, **kwargs)

    monkeypatch.setattr(efficiency, "adjust_rank", spy)
    return calls


@pytest.mark.parametrize("keep_ratio", [1.0, 0.5, 0.25])
def test_rank_budget_uses_the_target_ratio_not_the_eviction_ratio(
    monkeypatch, keep_ratio
):
    """Eviction shrinks what the backend sees; it must not move the target."""
    from cache.config import (
        BaselineCacheConfig,
        CompressedCacheConfig,
        SelectiveCacheConfig,
        XKVCacheConfig,
    )
    from cache.core import CompressedCache
    from tests.helpers import apply_model_rope, rope_cos_sin

    torch.manual_seed(0)
    batch_size, num_heads, seq_len, head_dim, target = 1, 2, 512, 32, 4.0
    cos, sin = rope_cos_sin(batch_size, seq_len, head_dim)
    keys = apply_model_rope(
        torch.randn(batch_size, num_heads, seq_len, head_dim), cos, sin
    )
    calls = _budget_inputs(monkeypatch)

    cache = CompressedCache(
        config=CompressedCacheConfig(
            key=XKVCacheConfig(
                layer_group_size=1,
                num_layers=1,
                svd_backend="linalg",
                compression_ratio=target,
            ),
            value=BaselineCacheConfig(),
            selective=SelectiveCacheConfig(enabled=False),
            eviction_keep_ratio=keep_ratio,
        ),
        verbose=False,
    )
    if keep_ratio < 1.0:
        cache.set_value_importance(
            0, torch.rand(batch_size, num_heads, seq_len)
        )
    cache.update(keys, torch.randn_like(keys), 0, {"cos": cos, "sin": sin})

    ((rows, target_ratio),) = calls
    # The target is the configured one however much eviction dropped, ...
    assert target_ratio == target
    assert cache.key_cache.decomposition.compression_ratio == target
    # ... and the budget is taken against what actually reached the backend.
    kept = cache.kept_positions.get(0)
    assert rows == (seq_len if kept is None else kept.numel())


@pytest.mark.parametrize("keep_ratio", [1.0, 0.5])
def test_residual_budget_uses_the_target_ratio_not_the_eviction_ratio(
    keep_ratio,
):
    from cache.config import (
        BaselineCacheConfig,
        CompressedCacheConfig,
        MLPValueCacheConfig,
        SelectiveCacheConfig,
    )
    from cache.core import CompressedCache

    from tests.helpers import rope_cos_sin

    torch.manual_seed(1)
    batch_size, num_heads, seq_len, head_dim, target = 1, 2, 512, 16, 4.0
    cos, sin = rope_cos_sin(batch_size, seq_len, head_dim)
    keys = torch.randn(batch_size, num_heads, seq_len, head_dim)

    cache = CompressedCache(
        config=CompressedCacheConfig(
            key=BaselineCacheConfig(),
            value=MLPValueCacheConfig(
                target_compression_ratio=target, num_epochs=1
            ),
            selective=SelectiveCacheConfig(enabled=False),
            eviction_keep_ratio=keep_ratio,
        ),
        cache_context={
            "padding_mask": torch.ones(batch_size, seq_len, dtype=torch.bool)
        },
        verbose=False,
    )
    if keep_ratio < 1.0:
        cache.set_value_importance(
            0, torch.rand(batch_size, num_heads, seq_len)
        )
    cache.update(keys, torch.randn_like(keys), 0, {"cos": cos, "sin": sin})

    layer = cache.value_cache.layers[0]
    assert layer.target_cr == target
    # The residual budget is sized from the tokens that survived eviction.
    kept = cache.kept_positions.get(0)
    assert layer.original_token_count == (
        seq_len if kept is None else kept.numel()
    )
    assert cache.value_cache.calc_compression_ratio() == pytest.approx(
        target, rel=0.05
    )
