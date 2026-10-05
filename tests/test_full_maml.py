"""Full MAML differentiates the adaptation; first-order stops at its result.

The meta gradient the two modes produce is genuinely different, so these tests
pin the exact one against finite differences rather than against each other.
"""

import torch
from omegaconf import OmegaConf

import meta_learning as meta
from model.meta_learning import _adam_step, inner_loop, trainable_params
from model.mlp import MLP

HEADS, HEAD_DIM, TOKENS = 2, 8, 16
LR, STEPS, CR = 1e-2, 2, 4.0


def _fixture(dtype=torch.float64):
    torch.manual_seed(11)
    mlps = [MLP(num_heads=HEADS, head_dim=HEAD_DIM).to(dtype) for _ in range(2)]
    kvs = [
        (
            torch.randn(1, HEADS, TOKENS, HEAD_DIM, dtype=dtype),
            torch.randn(1, HEADS, TOKENS, HEAD_DIM, dtype=dtype),
        )
        for _ in mlps
    ]
    return mlps, kvs


def _run(mlps, kvs, first_order):
    return inner_loop(
        mlps, kvs, LR, STEPS, residual_cr=CR, first_order=first_order
    )


def test_full_maml_matches_a_finite_difference_of_the_objective():
    """The exact meta gradient is the derivative of the post-adaptation
    objective with respect to the initialisation, so it must agree with a
    central difference taken through the whole inner loop."""
    mlps, kvs = _fixture()
    _, metrics = _run(mlps, kvs, first_order=False)
    analytic = metrics["param_grads"][0][0, 0, 0, 0].item()

    target = trainable_params(mlps[0])[0]
    step = 1e-6
    shifted = []
    for delta in (step, -step):
        with torch.no_grad():
            target[0, 0, 0, 0] += delta
        shifted.append(_run(mlps, kvs, first_order=False)[1]["meta_objective"])
        with torch.no_grad():
            target[0, 0, 0, 0] -= delta

    finite_difference = (shifted[0] - shifted[1]) / (2 * step)
    assert abs(analytic - finite_difference) < 1e-6 * max(1.0, abs(analytic))


def test_first_order_does_not_track_that_derivative():
    """Otherwise the flag would be doing nothing."""
    mlps, kvs = _fixture()
    exact = _run(mlps, kvs, first_order=False)[1]["param_grads"][0]
    approx = _run(mlps, kvs, first_order=True)[1]["param_grads"][0]

    assert not torch.allclose(exact.double(), approx.double(), rtol=1e-3)


def test_the_two_modes_report_the_same_adaptation():
    """Only the gradient differs. The reported losses come from the same
    inner loop, up to the epsilon that full MAML moves under the sqrt."""
    mlps, kvs = _fixture()
    exact = _run(mlps, kvs, first_order=False)[1]
    approx = _run(mlps, kvs, first_order=True)[1]

    for key in ("initial_support_loss", "final_support_loss", "meta_objective"):
        assert abs(exact[key] - approx[key]) < 1e-3 * abs(approx[key])


def test_a_dead_unit_does_not_poison_the_meta_gradient():
    """A coordinate with an exactly zero gradient leaves Adam's variance at
    zero, where sqrt has an infinite derivative. Real checkpoints have tens of
    thousands of these, so full MAML has to survive them."""
    mlps, kvs = _fixture()
    with torch.no_grad():  # silence one hidden unit so its gradients are 0
        mlps[0].weights[0][..., 0] = 0.0
        mlps[0].biases[0][..., 0] = -1.0

    _, metrics = _run(mlps, kvs, first_order=False)

    zero_grads = sum(
        int((g == 0).sum()) for g in metrics["param_grads"] if g is not None
    )
    assert zero_grads > 0, "test needs a genuinely dead unit"
    assert all(
        torch.isfinite(g).all() for g in metrics["param_grads"] if g is not None
    )


def test_the_unsafe_sqrt_is_what_would_have_poisoned_it():
    """Documents why full MAML does not reuse the first-order update."""
    param = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    grad = param * 0.0  # an exactly zero gradient that still carries a graph

    for safe, expect_finite in ((True, True), (False, False)):
        stepped, _, _ = _adam_step(
            [param], [grad], [torch.zeros(1, dtype=torch.float64)],
            [torch.zeros(1, dtype=torch.float64)], LR, 1, safe_sqrt=safe,
        )
        (second_order,) = torch.autograd.grad(
            stepped[0].sum(), param, allow_unused=True, retain_graph=True
        )
        assert torch.isfinite(second_order).all() == expect_finite


def test_run_name_separates_full_maml_checkpoints():
    values = {
        "seq_len": 4096, "inner_steps": 2, "meta_lr": 3e-4,
        "eval_target_cr": 4.0, "use_residual": False,
        "train_on_reconstructed_keys": False,
    }
    first_order = meta.build_run_name(OmegaConf.create(values))
    full = meta.build_run_name(OmegaConf.create({**values, "first_order": False}))

    assert "maml" not in first_order  # existing run directories keep their names
    assert "maml" in full and full != first_order
