"""What meta-training reports to wandb, without touching the network."""

import pytest
import torch
from omegaconf import OmegaConf

import meta_learning as ml
from utils.logging import average_metrics, init_wandb, log_epoch_metrics
from utils.rope import get_rope_theta

from tests.helpers import build_llama


NUM_LAYERS = 2
SEQ_LEN = 64


class RecordingRun:
    """Stands in for WandbRun and keeps what training asked it to log."""

    def __init__(self):
        self.logs = []
        self.summary = {}
        self.finished = False

    def log(self, metrics, step):
        self.logs.append((step, metrics))

    def summarise(self, **values):
        self.summary.update(values)

    def finish(self):
        self.finished = True

    def entries(self, prefix):
        return [
            (step, metrics)
            for step, metrics in self.logs
            if any(key.startswith(prefix) for key in metrics)
        ]


def _cfg(**overrides):
    cfg = OmegaConf.load("config/meta_learning.yaml").training
    cfg.seq_len = SEQ_LEN
    cfg.inner_steps = 1
    cfg.log_interval = 100
    cfg.inner_adaptation_dtype = "float32"
    # The logging cadence is independent of how the keys are prepared, and
    # the exact-key path keeps these tests quick.
    cfg.train_on_reconstructed_keys = False
    cfg.update(overrides)
    return cfg


@pytest.fixture(scope="module")
def model():
    return build_llama(NUM_LAYERS)


def _batches():
    while True:
        ids = torch.randint(3, 64, (1, SEQ_LEN))
        yield {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def _run_epoch(model, cfg, run, epoch=0, optimiser_steps=0):
    cfg.rope_theta = get_rope_theta(model.config)
    mlps = ml.init_mlps(model, cfg, "cpu")
    return ml.run_epoch(
        model,
        mlps,
        _batches(),
        ml.setup_optimizer(mlps, cfg),
        cfg,
        epoch,
        "cpu",
        torch.float32,
        residual_cr=float(cfg.eval_target_cr),
        key_config=ml.build_key_reconstruction_config(cfg),
        wandb_run=run,
        optimiser_steps=optimiser_steps,
    )


def test_one_log_per_optimiser_step_not_per_batch(model):
    torch.manual_seed(80)
    cfg = _cfg(batches_per_epoch=6, grad_accum_steps=3)
    run = RecordingRun()

    _, batch_count, optimiser_steps = _run_epoch(model, cfg, run)

    assert batch_count == 6
    assert optimiser_steps == 2
    steps = [step for step, _ in run.entries("train/")]
    assert steps == [1, 2], "wandb step should be the optimiser step count"


def test_a_partial_accumulation_window_is_still_logged(model):
    """The last few batches of an epoch take an optimiser step of their own."""
    torch.manual_seed(81)
    cfg = _cfg(batches_per_epoch=5, grad_accum_steps=2)
    run = RecordingRun()

    _, _, optimiser_steps = _run_epoch(model, cfg, run)

    assert optimiser_steps == 3
    assert [step for step, _ in run.entries("train/")] == [1, 2, 3]


def test_step_metrics_carry_the_adaptation_improvement(model):
    torch.manual_seed(82)
    cfg = _cfg(batches_per_epoch=2, grad_accum_steps=1)
    run = RecordingRun()

    _run_epoch(model, cfg, run)

    for _, metrics in run.entries("train/"):
        assert set(metrics) == {
            "train/initial_support_loss",
            "train/final_support_loss",
            "train/meta_objective",
            "train/adaptation_improvement",
            "train/epoch",
        }
        assert metrics["train/adaptation_improvement"] == (
            metrics["train/final_support_loss"]
            - metrics["train/initial_support_loss"]
        )


def test_optimiser_steps_continue_across_epochs(model):
    torch.manual_seed(83)
    cfg = _cfg(batches_per_epoch=4, grad_accum_steps=2)
    run = RecordingRun()

    _, _, after_first = _run_epoch(model, cfg, run, epoch=0)
    _, _, after_second = _run_epoch(
        model, cfg, run, epoch=1, optimiser_steps=after_first
    )

    assert (after_first, after_second) == (2, 4)
    assert [step for step, _ in run.entries("train/")] == [1, 2, 3, 4]


def test_epoch_metrics_and_summary():
    run = RecordingRun()
    sums = {
        "initial_support_loss": 4.0,
        "final_support_loss": 2.0,
        "meta_objective": 1.0,
    }
    log_epoch_metrics(
        run,
        average_metrics(sums, 2),
        epoch=3,
        batch_count=2,
        optimiser_steps=7,
    )

    (step, metrics), = run.entries("epoch/")
    assert step == 7
    assert metrics == {
        "epoch/avg_initial_support_loss": 2.0,
        "epoch/avg_final_support_loss": 1.0,
        "epoch/avg_meta_objective": 0.5,
        "epoch/num_batches": 2,
        "epoch/epoch": 3,
    }
    assert run.summary == {"optimiser_steps": 7, "epochs": 4}


def test_logging_is_a_no_op_when_disabled():
    config = OmegaConf.load("config/meta_learning.yaml")
    config.wandb.enabled = False

    run = init_wandb(config, "some-run")

    assert not run.enabled
    # The no-op handle still satisfies everything training calls on it.
    run.log({"train/meta_objective": 1.0}, step=1)
    run.summarise(optimiser_steps=1)
    run.finish()
