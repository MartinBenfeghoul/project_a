"""Residual sink reservation obeys padding and the existing row budget."""

import argparse
from types import SimpleNamespace

import pytest
import torch

from cache.backends.mlp_values import MLPValueLayer
from cache.config import build_value_cache, build_value_cache_config
from cache.rope import SharedRopeCache


def make_layer(sinks):
    # Unique errors, increasing with position, make ordinary MSE favor the end.
    values = torch.arange(1, 49, dtype=torch.float32).reshape(2, 2, 12, 1)
    keys = torch.zeros_like(values)
    padding = torch.tensor(
        [[False] * 2 + [True] * 10, [True] * 3 + [False] * 9]
    )
    layer = MLPValueLayer(target_cr=2, residual_sink_tokens=sinks)
    layer.lazy_initialization(values)
    layer.mlp = torch.nn.Identity()
    layer.tensor = values.clone()
    return layer, keys, values, padding


@pytest.mark.parametrize("budget", [22, 24, 26])
def test_sinks_reserve_each_head_and_sequence_within_budget(budget):
    layer, keys, values, padding = make_layer(8)
    layer.compress(keys, padding, budget)
    selected = torch.zeros(48, dtype=torch.bool)
    selected[layer.indices.long()] = True
    selected = selected.reshape(2, 2, 12)
    assert selected.sum() == budget
    assert selected[0, :, 2:10].all()
    assert selected[1, :, :3].all()  # Short sequence reserves all valid tokens.
    assert not (selected & ~padding[:, None, :]).any()
    # Remaining rows are selected by MSE, without counting sinks twice.
    candidates = [10, 11, 22, 23]
    expected_extra = candidates[-(budget - 22) :] if budget > 22 else []
    assert [i for i in candidates if selected.flatten()[i]] == expected_extra
    reconstructed = layer.decompress(keys)
    assert torch.equal(reconstructed[selected], values[selected])


def test_sinks_fail_when_mandatory_rows_exceed_budget():
    layer, keys, _, padding = make_layer(8)
    with pytest.raises(
        ValueError, match="requires 22 residual rows.*budget is 21"
    ):
        layer.compress(keys, padding, 21)
    assert not layer.is_compressed


@pytest.mark.parametrize("budget", [0, 4, 26])
def test_disabled_sinks_preserve_mse_selection(budget):
    layer, keys, values, padding = make_layer(0)
    valid = padding[:, None, :].expand(2, 2, 12).flatten().nonzero().flatten()
    expected = valid[values.flatten()[valid].topk(budget).indices].sort().values
    layer.compress(keys, padding, budget)
    assert torch.equal(layer.indices.long(), expected)


def test_cli_option_reaches_value_layer(monkeypatch):
    from lm_eval_script import parse_args

    monkeypatch.setattr(
        "sys.argv",
        [
            "lm_eval_script.py",
            "--target_cr",
            "4",
            "--no-use_residual",
            "--v_residual_sink_tokens",
            "8",
        ],
    )
    args = parse_args()
    config = build_value_cache_config(args, SimpleNamespace(), 1)
    cache, _ = build_value_cache(
        config, ddp_cache_data=None, rope_cache=SharedRopeCache(), verbose=False
    )
    assert cache._build_layer(0).residual_sink_tokens == 8


@pytest.mark.parametrize("sinks,backend", [(-1, "mlp"), (8, "baseline")])
def test_cli_rejects_invalid_sink_options(sinks, backend):
    from lm_eval_script import validate_args

    with pytest.raises(SystemExit):
        validate_args(
            argparse.ArgumentParser(),
            SimpleNamespace(v_residual_sink_tokens=sinks, v_cache_type=backend),
        )
