import json
import os

import torch
from dotenv import load_dotenv
from omegaconf import OmegaConf


class WandbRun:
    def __init__(self, run=None):
        self.run = run

    @property
    def enabled(self) -> bool:
        return self.run is not None

    def log(self, metrics: dict, step: int) -> None:
        if self.run is not None:
            self.run.log(metrics, step=step)

    def summarise(self, **values) -> None:
        if self.run is not None:
            self.run.summary.update(values)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()


def init_wandb(config, run_name: str) -> WandbRun:
    wandb_config = config.get("wandb", {})
    if not wandb_config.get("enabled", False):
        return WandbRun()

    import wandb

    load_dotenv()
    wandb_key = os.getenv("WANDB_KEY")
    if not wandb_key:
        raise RuntimeError(
            "WANDB_KEY environment variable is not set. Please add it to your .env file"
        )

    wandb.login(key=wandb_key)

    run = wandb.init(
        project=wandb_config.get("project"),
        entity=wandb_config.get("entity"),
        name=run_name,
        config=OmegaConf.to_container(config, resolve=True),
    )

    print(f"Logging to wandb run: {run.url or run.name}")
    return WandbRun(run)


def average_metrics(sums: dict, count: int) -> dict:
    return {metric: total / max(count, 1) for metric, total in sums.items()}


def log_step_metrics(
    wandb_run,
    window,
    count,
    epoch,
    optimiser_steps,
    grad_norm,
):
    """Log one optimiser step, averaged over its accumulated batches."""
    if wandb_run is None:
        return
    avgs = average_metrics(window, count)
    metrics = {
        "train/initial_support_loss": avgs["initial_support_loss"],
        "train/final_support_loss": avgs["final_support_loss"],
        "train/meta_objective": avgs["meta_objective"],
        "train/adaptation_improvement": (
            avgs["initial_support_loss"] - avgs["final_support_loss"]
        ),
        "train/epoch": epoch,
        "train/outer_grad_norm": grad_norm
    }
    wandb_run.log(metrics, step=optimiser_steps)


def log_epoch_metrics(wandb_run, avgs, epoch, batch_count, optimiser_steps):
    if wandb_run is None:
        return
    wandb_run.log(
        {
            "epoch/avg_initial_support_loss": avgs["initial_support_loss"],
            "epoch/avg_final_support_loss": avgs["final_support_loss"],
            "epoch/avg_meta_objective": avgs["meta_objective"],
            "epoch/num_batches": batch_count,
            "epoch/epoch": epoch,
        },
        step=optimiser_steps,
    )
    wandb_run.summarise(
        optimiser_steps=optimiser_steps,
        epochs=epoch + 1,
    )


def log_benchmark_scores(
    wandb_run,
    benchmark,
    avg_score,
    scores,
    epoch,
    optimiser_steps,
):
    if wandb_run is None:
        return
    wandb_run.log(
        {
            f"eval/{benchmark}/avg": avg_score,
            **{
                f"eval/{benchmark}/{task}": score
                for task, score in scores.items()
            },
            "eval/epoch": epoch,
        },
        step=optimiser_steps,
    )


def format_run_value(value: object) -> str:
    """Make a config value safe to embed in a directory name."""
    return str(value).replace("/", "-").replace(",", "-").replace(" ", "")


def save_attention_predictor_checkpoint(
    args,
    predictor,
    optimizer: torch.optim.Optimizer,
    layers: list[int],
    step: int,
    running: dict[str, float],
    num_updates: int,
) -> None:
    metrics = {
        key: value / max(1, num_updates) for key, value in running.items()
    }
    params = {
        "model_state_dict": predictor.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "metrics": metrics,
        "config": {
            "model_name": args.model_name,
            "seq_len": args.seq_len,
            "history_step": args.history_step,
            "block_size": args.block_size,
            "topk_blocks": args.topk_blocks,
            "layers": layers,
            "bce_weight": args.bce_weight,
        },
    }

    torch.save(
        params,
        os.path.join(args.checkpoint_dir, "model_ckpt.pt"),
    )

    metrics_path = os.path.join(args.checkpoint_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def save_checkpoint(layer_mlps, checkpoint_path, epoch):
    base, ext = os.path.splitext(checkpoint_path)
    epoch_checkpoint_path = f"{base}_epoch{epoch}{ext}"
    epoch_params = {
        f"layer_{i}": mlp.state_dict() for i, mlp in enumerate(layer_mlps)
    }
    torch.save(epoch_params, epoch_checkpoint_path)
    print(f"Checkpoint saved to {epoch_checkpoint_path}")


def get_output_path(output_path):
    i = 0
    while True:
        if not os.path.exists(output_path.format(i)):
            return output_path.format(i)
        i += 1


def save_run_config(
    args,
    layers: list[int],
    run_name: str,
    checkpoint_dir: str,
) -> None:
    config = {
        **vars(args),
        "run_name": run_name,
        "checkpoint_dir": checkpoint_dir,
        "resolved_layers": layers,
    }
    OmegaConf.save(
        OmegaConf.create(config), os.path.join(checkpoint_dir, "config.yaml")
    )


def prepare_run_directory(
    args,
    layers: list[int],
    run_name: str,
) -> str:
    """Create the run's checkpoint directory and save its resolved config."""
    args.checkpoint_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    save_run_config(args, layers, run_name, args.checkpoint_dir)
    return run_name
