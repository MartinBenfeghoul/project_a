import json
import os

import numpy as np
from omegaconf import OmegaConf
import torch


class Logger:
    def __init__(self, layer_idx=None):
        self.layer_idx = layer_idx
        self.log_dict = {}

    def add_log(self, key, value):
        if key not in self.log_dict:
            self.log_dict[key] = []
        self.log_dict[key].append(value)

    def add_dict(self, input_dict):
        for key, value in input_dict.items():
            self.add_log(key, value)

    def get_log_list(self, key):
        return self.log_dict.get(key, [])

    def get_log_mean(self, key, std=False, return_dtype=np.float64):
        values = self.get_log_list(key)
        if values:
            if isinstance(values[0], (int, float)):
                res = np.mean(values, dtype=return_dtype)
                if std:
                    res = (res, np.std(values, dtype=return_dtype))
            elif isinstance(values[0], np.ndarray):
                res = np.mean(np.stack(values), axis=0, dtype=return_dtype)
                if std:
                    res = (
                        res,
                        np.std(np.stack(values), axis=0, dtype=return_dtype),
                    )
            else:
                raise ValueError(f"Unexpected value type: {type(values[0])}")
            return res
        return None

    def get_dict_mean(self, std=False):
        mean_dict = {}
        for key in self.log_dict.keys():
            mean_dict[key] = self.get_log_mean(key, std=std)
        return mean_dict

    def keys(self):
        return self.log_dict.keys()

    def values(self):
        return self.log_dict.values()

    @property
    def length(self):
        if self.log_dict:
            first_key = next(iter(self.log_dict))
            return len(self.log_dict[first_key])
        return 0

    def clear(self):
        self.log_dict = {}


def generate_run_name(config):
    """Generate a run name based on config parameters."""
    t = config.training

    if t.get("run_type") == "attention_predictor":
        return (
            f"seq{t.seq_len}_"
            f"maxb{t.max_batches}_"
            f"sps{t.samples_per_sequence}_"
            f"hist{t.history_step}_"
            f"blk{t.block_size}_"
            f"topk{t.topk_blocks}_"
            f"layers{_format_run_value(t.layers)}_"
            f"bce{t.bce_weight}"
        )

    return f"seq{t.seq_len}_" f"steps{t.inner_steps}_" f"mlr{t.meta_lr}_"


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


def save_checkpoint(layer_mlps, inner_lr_params, checkpoint_path, epoch):
    base, ext = os.path.splitext(checkpoint_path)
    epoch_checkpoint_path = f"{base}_epoch{epoch}{ext}"
    epoch_params = {
        f"layer_{i}": mlp.state_dict() for i, mlp in enumerate(layer_mlps)
    }
    if inner_lr_params is not None:
        epoch_params["inner_lr_params"] = [
            lr.detach().cpu() for lr in inner_lr_params
        ]
    torch.save(epoch_params, epoch_checkpoint_path)
    print(f"Checkpoint saved to {epoch_checkpoint_path}")


def get_output_path(output_path):
    i = 0
    while True:
        if not os.path.exists(output_path.format(i)):
            return output_path.format(i)
        i += 1


def _format_run_value(value: object) -> str:
    return str(value).replace("/", "-").replace(",", "-").replace(" ", "")


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


def make_hooks(logger):
    def post_hook(module, args, kwargs, output):
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        if input_ids is None:
            return

        pkv = output.past_key_values
        if input_ids.size(-1) > 1:
            if hasattr(pkv, "key_recon_mse"):
                key_mse = pkv.key_recon_mse
                if key_mse is not None:
                    print(f"Key reconstruction MSE: {key_mse:.6f}")
                    logger.add_log("key_recon_mse", key_mse)

            if hasattr(pkv, "value_recon_mse"):
                value_mse = pkv.value_recon_mse
                if value_mse is not None:
                    print(f"Value reconstruction MSE: {value_mse:.6f}")
                    logger.add_log("value_recon_mse", value_mse)

            if hasattr(pkv, "update_events"):
                pkv.update_events()
        elif hasattr(pkv, "comp_ratio") and not logger.recorded_cr:
            compression_ratio = pkv.comp_ratio
            if compression_ratio is not None:
                print(f"Compression ratio: {compression_ratio:.2f}")
                logger.add_log("crs", compression_ratio)
                logger.recorded_cr = True

    return post_hook


def prepare_run_directory(
    args,
    layers: list[int],
) -> str:
    training = {
        **vars(args),
        "run_type": "attention_predictor",
        "resolved_layers": layers,
    }
    run_name = generate_run_name(OmegaConf.create({"training": training}))
    args.checkpoint_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    save_run_config(args, layers, run_name, args.checkpoint_dir)
    return run_name
