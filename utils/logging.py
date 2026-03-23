import numpy as np
import json
import os
import torch

from omegaconf import OmegaConf
import wandb


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

    run_name = (
        f"seq{t.seq_len}_"
        f"steps{t.inner_steps}_"
        f"mlr{t.meta_lr}_"
        f"ilr{t.inner_lr}_"
        f"learnlr{t.get('learn_inner_lr', False)}_"
        f"learnperc{t.get('learn_target_perc', False)}_"
        f"gradaccum{t.grad_accum_steps}"
    )
    return run_name


def init_wandb(config):
    wandb_config = config.get("wandb", {})
    if not wandb_config.get("enabled", False):
        return False

    wandb_key = os.getenv("WANDB_KEY")
    if not wandb_key:
        raise RuntimeError(
            "WANDB_KEY environment variable is not set. Please add it to your .env file"
        )

    wandb.login(key=wandb_key)
    run_name = generate_run_name(config)

    wandb.init(
        project=wandb_config.get("project", "gist_vs_details"),
        entity=wandb_config.get("entity", "mixture_of_titans"),
        name=run_name,
        config=OmegaConf.to_container(config, resolve=True),
    )
    return True


def save_results(
    file_name,
    avg_nll,
    avg_nll_modified_cache,
    new_avg_nll_change_perc,
    accuracy,
    threshold,
    num_epoch,
    mem_func,
    num_token,
    num_token_per_training,
    type_of_seq,
    num_seq,
    lr,
    loss_func,
    num_changed_kv,
    percentage_changed_kv,
    layer_decomposition=None,
    seq_len=None,
):
    """Save experiment results to a JSONL file."""
    with open(file_name, "a") as f:
        f.write(
            json.dumps(
                {
                    "seq_len": seq_len,
                    "avg_nll": avg_nll,
                    "avg_nll_change_thresh": avg_nll_modified_cache,
                    "avg_nll_change_perc": new_avg_nll_change_perc,
                    "avg_accuracy_modified_cache": accuracy,
                    "threshold": threshold,
                    "num_epoch": num_epoch,
                    "mem_func": mem_func,
                    "num_token": num_token,
                    "num_token_per_training": num_token_per_training,
                    "type_of_seq": type_of_seq,
                    "num_seq": num_seq,
                    "lr": lr,
                    "loss_func": loss_func,
                    "num_changed_kv": num_changed_kv,
                    "percentage_changed_kv": percentage_changed_kv,
                    "layer_decomposition": layer_decomposition,
                }
            )
            + "\n"
        )


def log_batch(
    epoch,
    batch_idx,
    q,
    inner_metrics,
    per_layer_losses,
    inner_lr_params,
    batch_time_ms,
    global_step,
    use_wandb,
    target_perc_params=None,
):
    initial_support_loss = (
        inner_metrics["inner_losses"][0]
        if inner_metrics["inner_losses"]
        else 0.0
    )
    final_support_loss = inner_metrics["final_support_loss"] or 0.0
    generalisation_gap = q - final_support_loss

    perc_str = ""
    if target_perc_params is not None:
        mean_perc = (
            sum(p.item() * 100 for p in target_perc_params)
            / len(target_perc_params)
        )
        perc_str = f", Target Perc: {mean_perc:.1f}%"

    print(
        f"Epoch {epoch}, Batch {batch_idx}, "
        f"Query Loss: {q:.6f}, "
        f"Gen Gap: {generalisation_gap:.6f}, "
        f"Time: {batch_time_ms:.1f}ms"
        f"{perc_str}"
    )

    if use_wandb:
        log_dict = {
            "train/query_loss": q,
            "train/epoch": epoch,
            "train/batch": batch_idx,
            "train/global_step": global_step,
            "train/generalisation_gap": generalisation_gap,
            "inner/support_loss_initial": initial_support_loss,
            "inner/support_loss_final": final_support_loss,
            "inner/adaptation_improvement": initial_support_loss
            - final_support_loss,
            "perf/batch_time_ms": batch_time_ms,
        }

        if per_layer_losses:
            for layer_idx, layer_loss in enumerate(per_layer_losses):
                log_dict[f"layer/query_loss_layer_{layer_idx}"] = layer_loss

        if inner_lr_params is not None:
            lr_values = [lr.item() for lr in inner_lr_params]
            log_dict["inner/learned_lr_mean"] = sum(lr_values) / len(lr_values)
            log_dict["inner/learned_lr_min"] = min(lr_values)
            log_dict["inner/learned_lr_max"] = max(lr_values)

        if target_perc_params is not None:
            perc_values = [p.item() * 100 for p in target_perc_params]
            log_dict["meta/target_perc_mean"] = sum(perc_values) / len(perc_values)
            log_dict["meta/target_perc_min"] = min(perc_values)
            log_dict["meta/target_perc_max"] = max(perc_values)
            for layer_idx, pv in enumerate(perc_values):
                log_dict[f"meta/target_perc_layer_{layer_idx}"] = pv

        wandb.log(log_dict, step=global_step)


def save_checkpoint(layer_mlps, inner_lr_params, checkpoint_path, epoch, target_perc_params=None):
    base, ext = os.path.splitext(checkpoint_path)
    epoch_checkpoint_path = f"{base}_epoch{epoch}{ext}"
    epoch_params = {
        f"layer_{i}": mlp.state_dict() for i, mlp in enumerate(layer_mlps)
    }
    if inner_lr_params is not None:
        epoch_params["inner_lr_params"] = [
            lr.detach().cpu() for lr in inner_lr_params
        ]
    if target_perc_params is not None:
        epoch_params["target_perc_params"] = [
            p.detach().cpu() for p in target_perc_params
        ]
    torch.save(epoch_params, epoch_checkpoint_path)
    print(f"Checkpoint saved to {epoch_checkpoint_path}")

def get_output_path(output_path):
    i=0
    while True:
        if not os.path.exists(output_path.format(i)):
            return output_path.format(i)
        i+=1