import os
import time

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader

from utils.tracking import init_wandb
import wandb

from tqdm import tqdm

from utils import (
    get_model_and_tokenizer,
    MLP,
    MetaLearningDataset,
    load_data,
    meta_collate,
    get_loss_func,
    generate_run_name,
    )

from dotenv import load_dotenv

load_dotenv()

def inner_loop_functional(layer_mlps, kv_cache, support_slice, inner_lr, inner_steps, loss_fn=F.mse_loss, track_losses=False):
    theta = [p for mlp in layer_mlps for p in mlp.parameters()]
    phi = [p.detach().clone().requires_grad_(True) for p in theta]

    inner_losses = [] if track_losses else None

    for _ in range(inner_steps):
        total_loss = 0.0
        param_idx = 0

        for layer_idx, layer in enumerate(kv_cache.layers):
            k = layer.keys[:, :, support_slice, :].float()
            v = layer.values[:, :, support_slice, :].float()

            mlp = layer_mlps[layer_idx]
            v_hat = functional_mlp_forward(mlp, k.float(), phi, param_idx)
            param_idx += sum(1 for _ in mlp.parameters())

            total_loss += loss_fn(v_hat, v)

        if track_losses:
            inner_losses.append(float(total_loss.detach().cpu()))

        grads = torch.autograd.grad(total_loss, phi, create_graph=False, retain_graph=False)

        if isinstance(inner_lr, list):
            phi = [p - lr * g for p, lr, g in zip(phi, inner_lr, grads)]
        else:
            phi = [p - inner_lr * g for p, g in zip(phi, grads)]

    final_support_loss = None
    if track_losses:
        with torch.no_grad():
            total = 0.0
            param_idx = 0
            for layer_idx, layer in enumerate(kv_cache.layers):
                k = layer.keys[:, :, support_slice, :].float()
                v = layer.values[:, :, support_slice, :].float()
                mlp = layer_mlps[layer_idx]
                v_hat = functional_mlp_forward(mlp, k.float(), phi, param_idx)
                param_idx += sum(1 for _ in mlp.parameters())
                total += loss_fn(v_hat, v).item()
            final_support_loss = total

    return theta, phi, {"inner_losses": inner_losses, "final_support_loss": final_support_loss}


def functional_mlp_forward(mlp, x, all_params, start_idx):
    # x: [B, H, T, D]
    # params order from ParameterList: weights[0..N-1], then biases[0..N-1]
    n = mlp.num_layers
    out = x

    for i in range(n):
        weight = all_params[start_idx + i]
        bias = all_params[start_idx + n + i]
        out = torch.matmul(out, weight) + bias
        if i < n - 1:
            out = mlp.intermediate_activation(out)

    return out


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


def meta_train(
    model,
    layer_mlps,
    dataloader,
    device,
    config,
    checkpoint_path,
    loss_fn=F.mse_loss,
    use_wandb=False,
):
    meta_lr = config.meta_lr
    inner_lr = config.inner_lr
    inner_steps = config.inner_steps
    support_ratio = config.support_ratio
    num_meta_epochs = config.num_meta_epochs
    log_interval = config.get("log_interval", 10)
    batches_per_epoch = config.get("batches_per_epoch", None)
    learn_inner_lr = config.get("learn_inner_lr", False)
    grad_accum_steps = config.get("grad_accum_steps", 1)

    theta = [p for mlp in layer_mlps for p in mlp.parameters()]

    if learn_inner_lr: # TODO: maybe try doing this per head or per layer rather than per parameter
        inner_lr_params = [nn.Parameter(torch.tensor(inner_lr, dtype=torch.float32, device=device)) for _ in theta]
        meta_optimizer = torch.optim.Adam(theta + inner_lr_params, lr=meta_lr)
        print(f"Meta-learning inner LR: {len(inner_lr_params)} learnable LR params (init={inner_lr})")
    else:
        inner_lr_params = None
        meta_optimizer = torch.optim.Adam(theta, lr=meta_lr)

    model.eval()

    global_step = 0

    for epoch in range(num_meta_epochs):
        epoch_loss = 0
        num_batches = 0
        epoch_start_time = time.time()
        accum_count = 0
        meta_optimizer.zero_grad()

        for batch_idx, batch in enumerate(tqdm(dataloader, total=batches_per_epoch or len(dataloader))):
            if batches_per_epoch is not None and batch_idx >= batches_per_epoch:
                break

            batch_start_time = time.time()

            input_ids = batch["input_ids"].to(device)
            _, seq_len = input_ids.shape

            split_idx = int(seq_len * support_ratio)
            support_slice = slice(0, split_idx)
            query_slice = slice(split_idx, seq_len)

            with torch.no_grad():
                out = model(
                    input_ids=input_ids,
                    attention_mask=batch["attention_mask"].to(device),
                    use_cache=True,
                )
                kv_cache = out.past_key_values

            should_log = (batch_idx % log_interval == 0)

            theta_list, phi, inner_metrics = inner_loop_functional( # phi = adapted_params
                layer_mlps, kv_cache, support_slice,
                inner_lr_params if inner_lr_params is not None else inner_lr,
                inner_steps, loss_fn,
                track_losses=should_log
            )

            query_loss, per_layer_losses = compute_query_loss_functional(
                layer_mlps, kv_cache, query_slice, phi, loss_fn,
                track_per_layer=should_log
            )

            if inner_lr_params is not None:
                all_grads = torch.autograd.grad(query_loss, list(phi) + inner_lr_params, create_graph=False, retain_graph=False)
                g_phi = all_grads[:len(phi)]
                g_lr = all_grads[len(phi):]
            else:
                g_phi = torch.autograd.grad(query_loss, phi, create_graph=False, retain_graph=False)
                g_lr = None

            scale = 1.0 / grad_accum_steps
            for p_theta, g in zip(theta_list, g_phi):
                if g is None:
                    continue
                if p_theta.grad is None:
                    p_theta.grad = g.detach() * scale
                else:
                    p_theta.grad.add_(g.detach() * scale)

            if g_lr is not None:
                for lr_param, g in zip(inner_lr_params, g_lr):
                    if lr_param.grad is None:
                        lr_param.grad = g.detach() * scale
                    else:
                        lr_param.grad.add_(g.detach() * scale)

            accum_count += 1
            if accum_count >= grad_accum_steps:
                meta_optimizer.step()
                meta_optimizer.zero_grad()
                global_step += 1
                accum_count = 0

            q = float(query_loss.detach().cpu())
            epoch_loss += q
            num_batches += 1

            batch_time_ms = (time.time() - batch_start_time) * 1000

            del kv_cache

            if should_log:
                initial_support_loss = inner_metrics["inner_losses"][0] if inner_metrics["inner_losses"] else 0.0
                final_support_loss = inner_metrics["final_support_loss"] or 0.0
                generalisation_gap = q - final_support_loss

                print(
                    f"Epoch {epoch}, Batch {batch_idx}, "
                    f"Query Loss: {q:.6f}, "
                    f"Gen Gap: {generalisation_gap:.6f}, "
                    f"Time: {batch_time_ms:.1f}ms"
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
                        "inner/adaptation_improvement": initial_support_loss - final_support_loss,
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

                    wandb.log(log_dict, step=global_step)

        # Flush any remaining accumulated gradients at end of epoch
        if accum_count > 0:
            meta_optimizer.step()
            meta_optimizer.zero_grad()
            global_step += 1
            accum_count = 0

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

        base, ext = os.path.splitext(checkpoint_path)
        epoch_checkpoint_path = f"{base}_epoch{epoch}{ext}"
        epoch_params = {f"layer_{i}": mlp.state_dict() for i, mlp in enumerate(layer_mlps)}
        if inner_lr_params is not None:
            epoch_params["inner_lr_params"] = [lr.detach().cpu() for lr in inner_lr_params]
        torch.save(epoch_params, epoch_checkpoint_path)
        print(f"Checkpoint saved to {epoch_checkpoint_path}")

    return layer_mlps, inner_lr_params



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

    model_name = config.model.name
    model_folder = model_name.split("/")[-1]

    run_name = generate_run_name(config)
    checkpoint_dir = os.path.join("checkpoints", model_folder, run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {checkpoint_dir}/")

    config_save_path = os.path.join(checkpoint_dir, "config.yaml")
    OmegaConf.save(config, config_save_path)
    print(f"Config saved to: {config_save_path}")

    model, tokenizer = get_model_and_tokenizer(model_name, device) # TODO: check torch_dtype=torch.bfloat16

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
        MLP(num_heads=num_kv_heads, head_dim=head_dim).to(device)
        for _ in range(num_layers)
    ]

    print("Loading dataset...")
    hf_dataset = load_data()

    meta_dataset = MetaLearningDataset(
        hf_dataset,
        tokenizer,
        seq_len=training_config.seq_len, # TODO: check whether sampling sequence length from a list of possible seq_lens improves 
        eos_id=tokenizer.eos_token_id,
    )

    dataloader = DataLoader(
        meta_dataset,
        batch_size=training_config.batch_size,
        collate_fn=meta_collate,
    )

    batches_per_epoch = training_config.get("batches_per_epoch", None)
    checkpoint_path = os.path.join(checkpoint_dir, "meta_learned_mlps.pt")
    loss_fn = get_loss_func(training_config.get("loss_func", "mse"))
    print(f"Starting meta-training... (batches_per_epoch: {batches_per_epoch or 'unlimited'})")
    layer_mlps, inner_lr_params = meta_train(
        model=model,
        layer_mlps=layer_mlps,
        dataloader=dataloader,
        device=device,
        config=training_config,
        checkpoint_path=checkpoint_path,
        loss_fn=loss_fn,
        use_wandb=use_wandb,
    )

    if use_wandb:
        wandb.finish()
    
    return trained_params


if __name__ == "__main__":
    trained_params = main()