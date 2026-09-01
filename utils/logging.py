import json
import os

from omegaconf import OmegaConf
import torch


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
