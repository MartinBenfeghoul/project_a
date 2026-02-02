import time

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader

from utils.tracking import init_wandb
import wandb

from utils import (
    get_model_and_tokenizer,
    avg_nll,
    VectorizedIndependentHeadMLP,
    MetaLearningDataset,
    load_data,
    meta_collate,
    )

from dotenv import load_dotenv

load_dotenv()

def inner_loop_functional(layer_mlps, kv_cache, support_slice, inner_lr, inner_steps, loss_fn=F.mse_loss, track_losses=False):
    params = [p for mlp in layer_mlps for p in mlp.parameters()]
    adapted_params = [p.clone() for p in params] # to return adapted params as new tensor (not in-place)

    inner_losses = [] if track_losses else None

    for _ in range(inner_steps):
        total_loss = 0
        param_idx = 0

        for layer_idx, layer in enumerate(kv_cache.layers):
            k = layer.keys[:, :, support_slice, :].float()
            v = layer.values[:, :, support_slice, :].float()

            mlp = layer_mlps[layer_idx]
            v_hat = functional_mlp_forward(mlp, k.float(), adapted_params, param_idx)
            param_idx += sum(1 for _ in mlp.parameters())

            total_loss += loss_fn(v_hat, v)

        if track_losses:
            inner_losses.append(total_loss.item())

        grads = torch.autograd.grad(total_loss, adapted_params, create_graph=False)

        # functional, keeps graph
        adapted_params = [p - inner_lr * g for p, g in zip(adapted_params, grads)]

    final_support_loss = None
    if track_losses:
        with torch.no_grad():
            final_loss = 0
            param_idx = 0
            for layer_idx, layer in enumerate(kv_cache.layers):
                k = layer.keys[:, :, support_slice, :].float()
                v = layer.values[:, :, support_slice, :].float()
                mlp = layer_mlps[layer_idx]
                v_hat = functional_mlp_forward(mlp, k.float(), adapted_params, param_idx)
                param_idx += sum(1 for _ in mlp.parameters())
                final_loss += loss_fn(v_hat, v).item()
            final_support_loss = final_loss

    return adapted_params, {"inner_losses": inner_losses, "final_support_loss": final_support_loss}


def functional_mlp_forward(mlp, x, all_params, start_idx):
    b, h, t, d = x.shape
    x_reshaped = x.permute(0, 1, 3, 2).reshape(b, h * d, t)

    param_idx = start_idx
    out = x_reshaped

    for layer in mlp.net:
        if isinstance(layer, nn.Conv1d):
            weight = all_params[param_idx]
            bias = all_params[param_idx + 1]
            out = F.conv1d(out, weight, bias, groups=layer.groups)
            param_idx += 2
        elif isinstance(layer, nn.GELU):
            out = F.gelu(out)

    return out.view(b, h, d, t).permute(0, 1, 3, 2)


def compute_query_loss_functional(layer_mlps, kv_cache, query_slice, adapted_params, loss_fn=F.mse_loss, track_per_layer=False):
    total_loss = 0
    param_idx = 0
    per_layer_losses = [] if track_per_layer else None

    for layer_idx, layer in enumerate(kv_cache.layers):
        k = layer.keys[:, :, query_slice, :].float()
        v = layer.values[:, :, query_slice, :].float()

        mlp = layer_mlps[layer_idx]
        v_hat = functional_mlp_forward(mlp, k.float(), adapted_params, param_idx)
        param_idx += sum(1 for _ in mlp.parameters())

        layer_loss = loss_fn(v_hat, v)
        total_loss += layer_loss

        if track_per_layer:
            per_layer_losses.append(layer_loss.item())

    return total_loss, per_layer_losses


def compute_weight_norm(params):
    total_norm = 0.0
    for p in params:
        total_norm += p.data.norm(2).item() ** 2
    return total_norm ** 0.5


def compute_update_norm(old_params, new_params):
    total_norm = 0.0
    for old_p, new_p in zip(old_params, new_params):
        total_norm += (new_p.data - old_p).norm(2).item() ** 2
    return total_norm ** 0.5


def meta_train(
    model,
    tokenizer,
    layer_mlps,
    dataloader,
    device,
    config,
    loss_fn=F.mse_loss,
    use_wandb=False,
):
    meta_lr = config.meta_lr
    inner_lr = config.inner_lr
    inner_steps = config.inner_steps
    support_ratio = config.support_ratio
    num_meta_epochs = config.num_meta_epochs
    eval_batch_size = config.eval_batch_size
    log_interval = config.get("log_interval", 10)

    all_params = [p for mlp in layer_mlps for p in mlp.parameters()]
    meta_optimizer = torch.optim.Adam(all_params, lr=meta_lr)

    global_step = 0

    for epoch in range(num_meta_epochs):
        epoch_loss = 0
        num_batches = 0
        epoch_start_time = time.time()

        for batch_idx, batch in enumerate(dataloader):
            batch_start_time = time.time()

            old_params = [p.data.clone() for p in all_params]

            meta_optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            _, seq_len = input_ids.shape

            split_idx = int(seq_len * support_ratio)
            support_slice = slice(0, split_idx)
            query_slice = slice(split_idx, seq_len)

            _, kv_cache = avg_nll(batch, model, eval_batch_size, tokenizer, device, already_tokenized=True)

            should_log = (batch_idx % log_interval == 0)

            adapted_params, inner_metrics = inner_loop_functional(
                layer_mlps, kv_cache, support_slice, inner_lr, inner_steps, loss_fn,
                track_losses=should_log
            )

            query_loss, per_layer_losses = compute_query_loss_functional(
                layer_mlps, kv_cache, query_slice, adapted_params, loss_fn,
                track_per_layer=should_log
            )

            # Backpropagate through both loops to get meta-gradient
            query_loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(all_params, float('inf'))

            meta_optimizer.step()

            query_loss_val = query_loss.item()
            epoch_loss += query_loss_val
            num_batches += 1
            global_step += 1

            batch_time_ms = (time.time() - batch_start_time) * 1000

            del kv_cache
            torch.cuda.empty_cache()

            if should_log:
                weight_norm = compute_weight_norm(all_params)
                update_norm = compute_update_norm(old_params, all_params)

                initial_support_loss = inner_metrics["inner_losses"][0] if inner_metrics["inner_losses"] else 0
                final_support_loss = inner_metrics["final_support_loss"] or 0
                generalization_gap = query_loss_val - final_support_loss

                print(
                    f"Epoch {epoch}, Batch {batch_idx}, "
                    f"Query Loss: {query_loss_val:.6f}, "
                    f"Gen Gap: {generalization_gap:.6f}, "
                    f"Time: {batch_time_ms:.1f}ms"
                )

                if use_wandb:
                    log_dict = {
                        "train/query_loss": query_loss_val,
                        "train/grad_norm": grad_norm.item(),
                        "train/epoch": epoch,
                        "train/batch": batch_idx,
                        "train/global_step": global_step,
                        "train/generalisation_gap": generalization_gap,
                        "inner/support_loss_initial": initial_support_loss,
                        "inner/support_loss_final": final_support_loss,
                        "inner/adaptation_improvement": initial_support_loss - final_support_loss,
                        "params/weight_norm": weight_norm,
                        "params/update_norm": update_norm,
                        "perf/batch_time_ms": batch_time_ms,
                    }

                    if inner_metrics["inner_losses"]:
                        for step_idx, step_loss in enumerate(inner_metrics["inner_losses"]):
                            log_dict[f"inner/support_loss_step_{step_idx}"] = step_loss

                    if per_layer_losses:
                        for layer_idx, layer_loss in enumerate(per_layer_losses):
                            log_dict[f"layer/query_loss_layer_{layer_idx}"] = layer_loss

                    wandb.log(log_dict, step=global_step)

            del old_params

        epoch_time_sec = time.time() - epoch_start_time
        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"Epoch {epoch} complete. Average Query Loss: {avg_loss:.6f}, Time: {epoch_time_sec:.1f}s")

        if use_wandb:
            wandb.log({
                "epoch/avg_query_loss": avg_loss,
                "epoch/epoch": epoch,
                "epoch/num_batches": num_batches,
                "perf/epoch_time_sec": epoch_time_sec,
            }, step=global_step)

    return layer_mlps



def load_config():
    """
    CLI args use dotlist notation, e.g.:
        python meta_learning.py training.batch_size=4 training.inner_lr=0.001
    """
    base_config = OmegaConf.load("config/meta_learning.yaml")
    cli_config = OmegaConf.from_cli()
    config = OmegaConf.merge(base_config, cli_config)
    return config

def main():
    config = load_config()
    training_config = config.training

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    use_wandb = init_wandb(config)
    if use_wandb:
        print("wandb logging enabled")

    print(f"Using device: {device}")
    print(f"Loading model: {config.model.name}")

    model, tokenizer = get_model_and_tokenizer(config.model.name, device)

    num_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    num_layers = model.config.num_hidden_layers

    print(f"Model config: {num_layers} layers, {num_kv_heads} KV heads, {head_dim} head_dim")

    if use_wandb:
        wandb.config.update({
            "model/num_layers": num_layers,
            "model/num_kv_heads": num_kv_heads,
            "model/head_dim": head_dim,
        })

    layer_mlps = [
        VectorizedIndependentHeadMLP(num_heads=num_kv_heads, head_dim=head_dim).to(device)
        for _ in range(num_layers)
    ]

    print("Loading dataset...")
    hf_dataset = load_data()

    meta_dataset = MetaLearningDataset(
        hf_dataset,
        tokenizer,
        seq_len=training_config.seq_len,
        eos_id=tokenizer.eos_token_id,
    )

    dataloader = DataLoader(
        meta_dataset,
        batch_size=training_config.batch_size,
        collate_fn=meta_collate,
    )

    print("Starting meta-training...")
    layer_mlps = meta_train(
        model=model,
        tokenizer=tokenizer,
        layer_mlps=layer_mlps,
        dataloader=dataloader,
        device=device,
        config=training_config,
        use_wandb=use_wandb,
    )

    trained_params = {f"layer_{i}": mlp.state_dict() for i, mlp in enumerate(layer_mlps)}
    checkpoint_path = training_config.checkpoint_path
    torch.save(trained_params, checkpoint_path)
    print(f"Meta-learned parameters saved to {checkpoint_path}")

    if use_wandb:
        wandb.save(checkpoint_path)
        wandb.finish()
    
    return trained_params


if __name__ == "__main__":
    trained_paramss = main()