import json

from omegaconf import OmegaConf
import wandb

def generate_run_name(config):
    """Generate a run name based on config parameters."""
    model_name = config.model.name.split("/")[-1]
    training = config.training

    run_name = (
        f"{model_name}_"
        f"mlr{training.meta_lr}_"
        f"ilr{training.inner_lr}_"
        f"is{training.inner_steps}_"
        f"seq{training.seq_len}"
    )
    return run_name


def init_wandb(config):
    wandb_config = config.get("wandb", {})
    if not wandb_config.get("enabled", False):
        return False

    run_name = generate_run_name(config)

    wandb.init(
        project=wandb_config.get("project", "gist-vs-details"),
        entity=wandb_config.get("entity", None),
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
):
    """Save experiment results to a JSONL file."""
    with open(file_name, "a") as f:
        f.write(
            json.dumps(
                {
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
