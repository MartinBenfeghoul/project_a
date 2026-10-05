"""The meta objective scores exactly the rows eval leaves uncorrected."""

import torch

from cache.backends.mlp_values import MLPValueLayer
from model.meta_learning import _compressed_rows, _residual_loss
from model.mlp import MLP


BATCH, HEADS, SEQ_LEN, HEAD_DIM = 2, 4, 256, 32
TARGET_CR = 4.0


def _fixture(seed):
    torch.manual_seed(seed)
    mlp = MLP(num_heads=HEADS, head_dim=HEAD_DIM)
    keys = torch.randn(BATCH, HEADS, SEQ_LEN, HEAD_DIM)
    values = torch.randn(BATCH, HEADS, SEQ_LEN, HEAD_DIM)
    return mlp, keys, values


def _scored_rows(mlp, keys, values):
    """Flat indices of the rows the meta objective holds the MLP to."""
    preds = mlp(keys)
    rows = _compressed_rows(mlp, values, TARGET_CR)
    errors = torch.nn.functional.mse_loss(
        preds, values, reduction="none"
    ).mean(dim=-1)
    threshold = torch.topk(errors.flatten(), rows, largest=False).values[-1]
    return (errors <= threshold).flatten().nonzero().flatten(), rows


def _corrected_rows(mlp, keys, values):
    """Flat indices of the rows the value cache stores a residual for."""
    layer = MLPValueLayer(target_cr=TARGET_CR, num_epochs=0)
    layer.lazy_initialization(values)
    layer.mlp = mlp
    layer._num_params = sum(p.numel() for p in mlp.parameters())
    layer.tensor = values
    padding_mask = torch.ones(BATCH, SEQ_LEN, dtype=torch.bool)
    budget = layer.compute_residual_budget(padding_mask)
    layer.compress(keys, padding_mask, budget)
    return layer.indices.long(), budget


def test_objective_scores_the_rows_the_value_cache_will_not_correct():
    mlp, keys, values = _fixture(70)

    scored, scored_count = _scored_rows(mlp, keys, values)
    corrected, budget = _corrected_rows(mlp, keys, values)

    total = BATCH * HEADS * SEQ_LEN
    assert budget > 0, "test needs a budget that actually buys residual rows"
    assert scored_count == total - budget
    # The two partitions are complementary: every row is either corrected at
    # inference or scored by the meta objective, and none is both.
    assert scored.numel() + corrected.numel() == total
    assert torch.equal(
        torch.cat([scored, corrected]).sort().values,
        torch.arange(total),
    )


def test_selection_is_global_across_the_batch():
    """One sequence may take more than its share of the residual budget."""
    mlp, keys, values = _fixture(71)
    # Make the second sequence far harder to predict than the first.
    values[1] *= 8.0

    scored, _ = _scored_rows(mlp, keys, values)
    corrected, _ = _corrected_rows(mlp, keys, values)

    rows_per_sequence = HEADS * SEQ_LEN
    corrected_per_sequence = [
        int(((corrected // rows_per_sequence) == idx).sum())
        for idx in range(BATCH)
    ]
    assert corrected_per_sequence[1] > corrected_per_sequence[0], (
        "the harder sequence should draw more of the residual budget"
    )
    # ...and the objective agrees with that split rather than halving it.
    scored_per_sequence = [
        int(((scored // rows_per_sequence) == idx).sum())
        for idx in range(BATCH)
    ]
    assert scored_per_sequence[0] > scored_per_sequence[1]
    for idx in range(BATCH):
        assert (
            scored_per_sequence[idx] + corrected_per_sequence[idx]
            == rows_per_sequence
        )
