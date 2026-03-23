"""
First-order MAML-style meta-learning for per-layer MLP KV-cache compressors.

Each MLP learns to predict value vectors from key vectors. The inner loop adapts
MLP weights to a support set of KV pairs; the outer (meta) loop optimises the
initial weights so that adapted MLPs generalise to the query set.
"""
import itertools
import os
import time

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import nn
from torch.func import functional_call
from torch.utils.data import DataLoader

from utils.logging import init_wandb
import wandb

from tqdm import tqdm

from utils import (
    get_model_and_tokenizer,
    Dataset,
    PairedDataset,
    load_data,
    collate_pairs,
    get_loss_func,
    generate_run_name,
    save_checkpoint,
    log_batch,
    Logger,
)
from model.mlp import MLP
from utils import inverse_rope, compute_rope_cos_sin

from dotenv import load_dotenv

load_dotenv()


def unrope(keys: torch.Tensor, rope_theta: float) -> torch.Tensor:
    """Un-rope keys extracted from the KV cache."""
    T = keys.shape[2]
    cos, sin = compute_rope_cos_sin(
        T, keys.shape[-1], rope_theta, keys.device, keys.dtype
    )
    return inverse_rope(keys, cos, sin)


def _support_loss(layer_mlps, kv_cache, phi, loss_fn, un_rope, rope_theta):
    total = 0.0
    for layer_idx, layer in enumerate(kv_cache.layers):
        k = layer.keys.float()
        if un_rope:
            k = unrope(k, rope_theta)
        weights, biases = phi[layer_idx]
        v_hat = functional_mlp_forward(layer_mlps[layer_idx], k, weights, biases)
        total = total + loss_fn(v_hat, layer.values.float())
    return total


def _phi_from_flat(flat, layer_param_counts):
    phi, idx = [], 0
    for n_w, n_b in layer_param_counts:
        phi.append((flat[idx:idx + n_w], flat[idx + n_w:idx + n_w + n_b]))
        idx += n_w + n_b
    return phi


def _sgd_update(phi, grads, inner_lr):
    g_iter = iter(grads)
    lr_iter = iter(inner_lr) if isinstance(inner_lr, list) else None
    return [
        (
            [p - (next(lr_iter) if lr_iter else inner_lr) * next(g_iter) for p in w],
            [p - (next(lr_iter) if lr_iter else inner_lr) * next(g_iter) for p in b],
        )
        for w, b in phi
    ]


def _adam_update(phi_flat, grads, inner_lr, step, m1, m2, beta1, beta2, eps):
    t = step + 1
    updated = []
    for i, (p, g) in enumerate(zip(phi_flat, grads)):
        m1[i] = beta1 * m1[i] + (1 - beta1) * g.detach()
        m2[i] = beta2 * m2[i] + (1 - beta2) * g.detach().pow(2)
        m_hat = m1[i] / (1 - beta1 ** t)
        v_hat = m2[i] / (1 - beta2 ** t)
        lr = inner_lr[i] if isinstance(inner_lr, list) else inner_lr
        updated.append(p - lr * m_hat / (v_hat.sqrt() + eps))
    return updated


def inner_loop_functional(
    layer_mlps,
    kv_cache,
    inner_lr,
    inner_steps,
    loss_fn=F.mse_loss,
    track_losses=False,
    un_rope=False,
    rope_theta=500_000.0,
    inner_optimizer="sgd",
    inner_adam_beta1=0.9,
    inner_adam_beta2=0.999,
    inner_adam_eps=1e-8,
):
    theta = [p for mlp in layer_mlps for p in list(mlp.weights) + list(mlp.biases)]
    phi = [
        (
            [p.detach().clone().requires_grad_(True) for p in mlp.weights],
            [p.detach().clone().requires_grad_(True) for p in mlp.biases],
        )
        for mlp in layer_mlps
    ]
    layer_param_counts = [(len(mlp.weights), len(mlp.biases)) for mlp in layer_mlps]

    adam_state = None
    if inner_optimizer == "adam":
        phi_flat0 = [p for w, b in phi for p in w + b]
        adam_state = (
            [torch.zeros_like(p.detach()) for p in phi_flat0],
            [torch.zeros_like(p.detach()) for p in phi_flat0],
        )

    inner_losses = [] if track_losses else None

    for step in range(inner_steps):
        total_loss = _support_loss(layer_mlps, kv_cache, phi, loss_fn, un_rope, rope_theta)

        if track_losses:
            inner_losses.append(float(total_loss.detach().cpu()))

        phi_flat = [p for w, b in phi for p in w + b]
        grads = torch.autograd.grad(total_loss, phi_flat, create_graph=False, retain_graph=False)

        if inner_optimizer == "adam":
            m1, m2 = adam_state
            updated_flat = _adam_update(phi_flat, grads, inner_lr, step, m1, m2, inner_adam_beta1, inner_adam_beta2, inner_adam_eps)
            phi = _phi_from_flat(updated_flat, layer_param_counts)
        else:
            phi = _sgd_update(phi, grads, inner_lr)

    final_support_loss = None
    if track_losses:
        with torch.no_grad():
            final_support_loss, _ = compute_loss_functional(
                layer_mlps, kv_cache, phi, loss_fn, un_rope=un_rope, rope_theta=rope_theta,
            )
            final_support_loss = final_support_loss.item()

    return theta, phi, {"inner_losses": inner_losses, "final_support_loss": final_support_loss}


def functional_mlp_forward(mlp, x, weights, biases):
    # x: [B, H, T, D]
    params = {f"weights.{i}": w for i, w in enumerate(weights)}
    params |= {f"biases.{i}": b for i, b in enumerate(biases)}
    return functional_call(mlp, params, (x,))


def compute_loss_functional(
    layer_mlps,
    kv_cache,
    phi,
    loss_fn=F.mse_loss,
    track_per_layer=False,
    un_rope=False,
    rope_theta=500_000.0,
):
    total_loss = 0
    per_layer_losses = [] if track_per_layer else None

    for layer_idx, layer in enumerate(kv_cache.layers):
        k = layer.keys.float()
        if un_rope:
            k = unrope(k, rope_theta)
        v = layer.values.float()

        mlp = layer_mlps[layer_idx]
        weights, biases = phi[layer_idx]
        v_hat = functional_mlp_forward(mlp, k.float(), weights, biases)

        layer_loss = loss_fn(v_hat, v)
        total_loss += layer_loss

        if track_per_layer:
            per_layer_losses.append(layer_loss.item())

    return total_loss, per_layer_losses


def compute_query_loss(
    layer_mlps,
    kv_cache,
    phi,
    loss_fn=F.mse_loss,
    target_perc_params=None,
    tau=0.01,
    lambda_compression=0.0,
    track_per_layer=False,
    un_rope=False,
    rope_theta=500_000.0,
):
    if target_perc_params is not None:
        return compute_compressed_loss(
            layer_mlps, kv_cache, phi, target_perc_params,
            tau=tau, lambda_compression=lambda_compression,
            track_per_layer=track_per_layer, un_rope=un_rope, rope_theta=rope_theta,
        )
    return compute_loss_functional(
        layer_mlps, kv_cache, phi, loss_fn,
        track_per_layer=track_per_layer, un_rope=un_rope, rope_theta=rope_theta,
    )

def get_differentiable_thresh(mse_per_token, index_perc):
    B, H, T = mse_per_token.shape
    sorted_errors, _ = torch.sort(mse_per_token.detach().view(B, -1), dim=1)
    k_lo = index_perc.detach().long().clamp(0, H * T - 2)
    frac = index_perc - k_lo.float()
    thresh = torch.lerp(sorted_errors[:, k_lo], sorted_errors[:, k_lo + 1], frac).view(B, 1, 1)
    return thresh


def compute_compressed_loss(
    layer_mlps,
    kv_cache,
    phi,
    target_perc_params,
    tau=0.01,
    lambda_compression=0.0,
    track_per_layer=False,
    un_rope=False,
    rope_theta=500_000.0,
):
    total_loss = 0
    per_layer_losses = [] if track_per_layer else None

    for layer_idx, layer in enumerate(kv_cache.layers):
        k_q = layer.keys.float()
        if un_rope:
            k_q = unrope(k_q, rope_theta)
        v_q = layer.values.float()

        mlp = layer_mlps[layer_idx]
        weights, biases = phi[layer_idx]
        v_hat_q = functional_mlp_forward(mlp, k_q, weights, biases)

        B, H, T, _ = v_q.shape
        N = H * T

        mse_per_token = (v_q - v_hat_q).pow(2).mean(dim=-1)

        perc = target_perc_params[layer_idx].clamp(0.0, 1.0)
        index_perc = (perc * N).clamp(1.0, float(N - 1))
        thresh = get_differentiable_thresh(mse_per_token, index_perc)

        soft_mask = torch.sigmoid((mse_per_token.detach() - thresh) / tau)
        layer_loss = ((1.0 - soft_mask) * mse_per_token).mean()

        # penalisation to prevent trivial perc
        layer_loss = layer_loss + lambda_compression * (1.0 - perc)

        total_loss = total_loss + layer_loss

        if track_per_layer:
            per_layer_losses.append(layer_loss.item())

    return total_loss, per_layer_losses


def setup_optimizer(layer_mlps, target_perc_params, config, device):
    theta = [
        p for mlp in layer_mlps for p in list(mlp.weights) + list(mlp.biases)
    ]
    learn_inner_lr = config.get("learn_inner_lr", False)
    learn_target_perc = config.get("learn_target_perc", False)

    meta_params = list(theta)
    if learn_target_perc and target_perc_params is not None:
        meta_params = meta_params + list(target_perc_params)

    if learn_inner_lr:  # TODO: maybe try doing this per head or per layer rather than per parameter
        inner_lr_params = [
            nn.Parameter(
                torch.tensor(
                    config.inner_lr, dtype=torch.float32, device=device
                )
            )
            for _ in theta
        ]
        meta_optimizer = torch.optim.Adam(
            meta_params + inner_lr_params, lr=config.meta_lr
        )
        print(
            f"Meta-learning inner LR: {len(inner_lr_params)} learnable LR params (init={config.inner_lr})"
        )
    else:
        inner_lr_params = None
        meta_optimizer = torch.optim.Adam(meta_params, lr=config.meta_lr)
    return theta, inner_lr_params, meta_optimizer


def meta_step(
    model,
    layer_mlps,
    batch,
    inner_lr,
    inner_steps,
    loss_fn,
    inner_lr_params,
    target_perc_params,
    tau,
    lambda_compression,
    should_log,
    device,
    un_rope=False,
    rope_theta=500_000.0,
    inner_optimizer="sgd",
    inner_adam_beta1=0.9,
    inner_adam_beta2=0.999,
    inner_adam_eps=1e-8,
):
    batch_start_time = time.time()

    with torch.no_grad():
        support_out = model(
            input_ids=batch["support_input_ids"].to(device),
            attention_mask=batch["support_attention_mask"].to(device),
            use_cache=True,
        )
        support_kv = support_out.past_key_values

    theta_list, phi, inner_metrics = (
        inner_loop_functional(  # phi = adapted_params
            layer_mlps,
            support_kv,
            inner_lr_params if inner_lr_params is not None else inner_lr,
            inner_steps,
            loss_fn,
            track_losses=should_log,
            un_rope=un_rope,
            rope_theta=rope_theta,
            inner_optimizer=inner_optimizer,
            inner_adam_beta1=inner_adam_beta1,
            inner_adam_beta2=inner_adam_beta2,
            inner_adam_eps=inner_adam_eps,
        )
    )
    del support_kv

    with torch.no_grad():
        query_out = model(
            input_ids=batch["query_input_ids"].to(device),
            attention_mask=batch["query_attention_mask"].to(device),
            use_cache=True,
        )
        query_kv = query_out.past_key_values

    query_loss, per_layer_losses = compute_query_loss(
        layer_mlps,
        query_kv,
        phi,
        loss_fn=loss_fn,
        target_perc_params=target_perc_params,
        tau=tau,
        lambda_compression=lambda_compression,
        track_per_layer=should_log,
        un_rope=un_rope,
        rope_theta=rope_theta,
    )

    del query_kv
    batch_time_ms = (time.time() - batch_start_time) * 1000
    return (
        query_loss,
        theta_list,
        phi,
        inner_metrics,
        per_layer_losses,
        batch_time_ms,
    )


def accumulate_gradients(
    query_loss, theta_list, phi, inner_lr_params, target_perc_params, scale
):
    phi_flat = [p for w, b in phi for p in w + b]

    learnable_perc = [p for p in target_perc_params if p.requires_grad] if target_perc_params else []

    extra_params = []
    if inner_lr_params is not None:
        extra_params.extend(inner_lr_params)
    extra_params.extend(learnable_perc)

    all_grads = torch.autograd.grad(
        query_loss,
        phi_flat + extra_params,
        create_graph=False,
        retain_graph=False,
    )
    g_phi = all_grads[: len(phi_flat)]
    g_extra = all_grads[len(phi_flat):]

    for p_theta, g in zip(theta_list, g_phi):
        if g is None:
            continue
        if p_theta.grad is None:
            p_theta.grad = g.detach() * scale
        else:
            p_theta.grad.add_(g.detach() * scale)

    offset = 0
    if inner_lr_params is not None:
        for lr_param, g in zip(inner_lr_params, g_extra[: len(inner_lr_params)]):
            if lr_param.grad is None:
                lr_param.grad = g.detach() * scale
            else:
                lr_param.grad.add_(g.detach() * scale)
        offset += len(inner_lr_params)

    if learnable_perc:
        for perc_param, g in zip(
            learnable_perc, g_extra[offset : offset + len(learnable_perc)]
        ):
            if perc_param.grad is None:
                perc_param.grad = g.detach() * scale
            else:
                perc_param.grad.add_(g.detach() * scale)


def run_epoch(
    model,
    layer_mlps,
    data_iter,
    meta_optimizer,
    inner_lr_params,
    target_perc_params,
    config,
    epoch,
    global_step,
    loss_fn,
    device,
    use_wandb,
):
    inner_lr = config.inner_lr
    inner_steps = config.inner_steps
    log_interval = config.get("log_interval", 10)
    batches_per_epoch = config.get("batches_per_epoch", None)
    grad_accum_steps = config.get("grad_accum_steps", 1)
    tau = config.get("tau", 0.01)
    lambda_compression = config.get("lambda_compression", 0.0)
    un_rope = config.get("un_rope", False)
    rope_theta = config.get("rope_theta", 500_000.0)
    inner_optimizer = config.get("inner_optimizer", "sgd")
    inner_adam_beta1 = config.get("inner_adam_beta1", 0.9)
    inner_adam_beta2 = config.get("inner_adam_beta2", 0.999)
    inner_adam_eps = config.get("inner_adam_eps", 1e-8)

    epoch_loss = 0
    num_batches = 0
    accum_count = 0
    meta_optimizer.zero_grad()

    batch_iter = itertools.islice(data_iter, batches_per_epoch)
    for batch_idx, batch in enumerate(tqdm(batch_iter, total=batches_per_epoch)):
        should_log = batch_idx % log_interval == 0

        (
            query_loss,
            theta_list,
            phi,
            inner_metrics,
            per_layer_losses,
            batch_time_ms,
        ) = meta_step(
            model,
            layer_mlps,
            batch,
            inner_lr,
            inner_steps,
            loss_fn,
            inner_lr_params,
            target_perc_params,
            tau,
            lambda_compression,
            should_log,
            device,
            un_rope=un_rope,
            rope_theta=rope_theta,
            inner_optimizer=inner_optimizer,
            inner_adam_beta1=inner_adam_beta1,
            inner_adam_beta2=inner_adam_beta2,
            inner_adam_eps=inner_adam_eps,
        )

        accumulate_gradients(
            query_loss,
            theta_list,
            phi,
            inner_lr_params,
            target_perc_params,
            scale=1.0 / grad_accum_steps,
        )

        accum_count += 1
        if accum_count >= grad_accum_steps: # gradient update every 32 batches
            meta_optimizer.step()
            if target_perc_params is not None:
                for p in target_perc_params:
                    if p.requires_grad:
                        p.data.clamp_(0.0, 1.0)
            meta_optimizer.zero_grad()
            global_step += 1
            accum_count = 0

        epoch_loss += float(query_loss.detach().cpu())
        num_batches += 1

        if should_log:
            log_batch(
                epoch,
                batch_idx,
                float(query_loss.detach().cpu()),
                inner_metrics,
                per_layer_losses,
                inner_lr_params,
                batch_time_ms,
                global_step,
                use_wandb,
                target_perc_params=target_perc_params,
            )

    # Flush any remaining accumulated gradients at end of epoch
    if accum_count > 0:
        meta_optimizer.step()
        if target_perc_params is not None:
            for p in target_perc_params:
                if p.requires_grad:
                    p.data.clamp_(0.0, 1.0)
        meta_optimizer.zero_grad()
        global_step += 1

    return epoch_loss, num_batches, global_step


def evaluate_ruler(
    model,
    model_name,
    tokenizer,
    target_perc_params,
    training_config,
    epoch_checkpoint_path,
    device,
    epoch,
    global_step,
    use_wandb,
    num_samples=10,
):
    """Run RULER benchmark with current MLP weights and log average score to wandb."""
    from lm_eval import evaluator
    from lm_eval.tasks import TaskManager
    from lm_eval_script import CompressedCacheHFLM, get_tasks, get_device, GEN_KWARGS

    print(f"Running RULER evaluation (epoch {epoch}, {num_samples} samples)...")

    num_layers = model.config.num_hidden_layers
    num_kv_heads = model.config.num_key_value_heads

    if target_perc_params is not None:
        target_perc = [p.clamp(0.0, 1.0).item() * 100 for p in target_perc_params]
    else:
        default_perc = training_config.get("target_perc", 50.0)
        target_perc = [default_perc] * num_layers

    logger = Logger()
    logger.prefill_events = []
    logger.decode_events = []

    key_cache_kwargs = {
        "cache_type": "baseline",
        "decomposition_method": "svd",
        "comp_ratio": 1.0,
        "energy_threshold": 1.0,
        "rank_selection": "comp_ratio",
        "lr": 1e-2,
        "n_iter": 3,
        "gamma": 3.0,
        "min_size": 8.0,
    }

    value_cache_kwargs = {
        "cache_type": "mlp",
        "num_layers_per_mlp": [training_config.mlp_num_layers] * num_layers,
        "hidden_factors_per_mlp": [training_config.mlp_hidden_factor] * num_layers,
        "num_heads_per_mlp": [num_kv_heads] * num_layers,
        "per_sequence": False,
        "target_perc": target_perc,
        "target_model_num_heads": num_kv_heads,
        "lr": training_config.inner_lr,
        "device": device,
        "optimizer": "sgd",
        "loss_func": training_config.get("loss_func", "mse"),
        "num_epochs": training_config.inner_steps,
        "meta_weights_path": epoch_checkpoint_path,
        "un_rope": training_config.get("un_rope", False),
        "rope_theta": training_config.get("rope_theta", 500_000.0),
    }

    lm = CompressedCacheHFLM(
        key_cache_kwargs=key_cache_kwargs,
        value_cache_kwargs=value_cache_kwargs,
        logger=logger,
        pretrained=model,
        tokenizer=tokenizer,
        truncation=False,
        trust_remote_code=True,
    )

    ruler_tasks = get_tasks(["ruler"])
    metadata = {"tokenizer": model_name}
    tm = TaskManager(metadata=metadata)

    results = evaluator.simple_evaluate(
        model=lm,
        gen_kwargs=GEN_KWARGS,
        tasks=ruler_tasks,
        num_fewshot=0,
        batch_size=1,
        device=get_device(lm),
        task_manager=tm,
        limit=num_samples,
    )

    task_scores = {}
    for task_name, task_results in results["results"].items():
        for key, val in task_results.items():
            if isinstance(val, (int, float)) and "stderr" not in key and key not in ("alias", "samples"):
                task_scores[task_name] = val
                break

    avg_score = sum(task_scores.values()) / len(task_scores) if task_scores else 0.0
    print(f"RULER eval epoch {epoch}: avg={avg_score:.4f}, per-task={task_scores}")

    if use_wandb:
        log_data = {"benchmark/ruler_avg": avg_score, "epoch/epoch": epoch}
        for task_name, score in task_scores.items():
            log_data[f"benchmark/{task_name}"] = score
        wandb.log(log_data)

    return avg_score


def meta_train(
    model,
    model_name,
    layer_mlps,
    dataloader,
    device,
    config,
    checkpoint_path,
    tokenizer=None,
    target_perc_params=None,
    loss_fn=F.mse_loss,
    use_wandb=False,
):
    _, inner_lr_params, meta_optimizer = setup_optimizer(
        layer_mlps, target_perc_params, config, device
    )
    model.eval()
    global_step = 0
    data_iter = iter(dataloader)

    if use_wandb:
        wandb.define_metric("benchmark/*", step_metric="epoch/epoch")

    for epoch in range(config.num_meta_epochs):
        epoch_start_time = time.time()

        epoch_loss, num_batches, global_step = run_epoch(
            model,
            layer_mlps,
            data_iter,
            meta_optimizer,
            inner_lr_params,
            target_perc_params,
            config,
            epoch,
            global_step,
            loss_fn,
            device,
            use_wandb,
        )

        epoch_time_sec = time.time() - epoch_start_time
        avg_loss = epoch_loss / max(num_batches, 1)
        print(
            f"Epoch {epoch} complete. Average Query Loss: {avg_loss:.6f}, Time: {epoch_time_sec:.1f}s"
        )

        if use_wandb:
            wandb.log(
                {
                    "epoch/avg_query_loss": avg_loss,
                    "epoch/epoch": epoch,
                    "epoch/num_batches": num_batches,
                    "perf/epoch_time_sec": epoch_time_sec,
                },
                step=global_step,
            )

        save_checkpoint(layer_mlps, inner_lr_params, checkpoint_path, epoch, target_perc_params)

        eval_ruler = config.get("eval_ruler", True)
        if eval_ruler and tokenizer is not None:
            base, ext = os.path.splitext(checkpoint_path)
            epoch_checkpoint_path = f"{base}_epoch{epoch}{ext}"
            evaluate_ruler(
                model=model,
                model_name=model_name,
                tokenizer=tokenizer,
                target_perc_params=target_perc_params,
                training_config=config,
                epoch_checkpoint_path=epoch_checkpoint_path,
                device=device,
                epoch=epoch,
                global_step=global_step,
                use_wandb=use_wandb,
                num_samples=config.get("eval_ruler_samples", 10),
            )

    return layer_mlps, inner_lr_params, target_perc_params


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

    device = "cuda" if torch.cuda.is_available() else "cpu"

    use_wandb = init_wandb(config)
    if use_wandb:
        print("wandb logging enabled")

    print(f"Using device: {device}")
    print(f"Loading model: {config.model.name}")

    model_name = config.model.name
    model_folder = model_name.split("/")[-1]

    run_name = generate_run_name(config)
    checkpoint_dir_base = os.path.join("checkpoints", model_folder, run_name)
    checkpoint_dir = checkpoint_dir_base
    idx = 0
    while os.path.exists(checkpoint_dir):
        checkpoint_dir = f"{checkpoint_dir_base}_{idx}"
        idx += 1
    os.makedirs(checkpoint_dir)
    print(f"Checkpoints will be saved to: {checkpoint_dir}/")

    config_save_path = os.path.join(checkpoint_dir, "config.yaml")
    OmegaConf.save(config, config_save_path)
    print(f"Config saved to: {config_save_path}")

    model, tokenizer = get_model_and_tokenizer(
        model_name, device
    )  # TODO: check torch_dtype=torch.bfloat16

    num_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    num_layers = model.config.num_hidden_layers

    print(
        f"Model config: {num_layers} layers, {num_kv_heads} KV heads, {head_dim} head_dim"
    )

    if use_wandb:
        wandb.config.update(
            {
                "model/num_layers": num_layers,
                "model/num_kv_heads": num_kv_heads,
                "model/head_dim": head_dim,
            }
        )

    mlp_num_layers = training_config.get("mlp_num_layers", 4)
    mlp_hidden_factor = training_config.get("mlp_hidden_factor", 2)
    layer_mlps = [
        MLP(
            num_heads=num_kv_heads,
            head_dim=head_dim,
            num_layers=mlp_num_layers,
            hidden_factor=mlp_hidden_factor,
        ).to(device)
        for _ in range(num_layers)
    ]

    learn_target_perc = training_config.get("learn_target_perc", False)
    default_perc = training_config.get("target_perc", 75.0) / 100.0
    if learn_target_perc:
        target_perc_params = [
            nn.Parameter(torch.tensor(default_perc, dtype=torch.float32, device=device))
            for _ in range(num_layers)
        ]
        print(
            f"Meta-learning target_perc: {num_layers} per-layer params "
            f"(init={default_perc * 100:.1f}%)"
        )
    else:
        target_perc_params = [
            torch.tensor(default_perc, dtype=torch.float32, device=device)
            for _ in range(num_layers)
        ]
        print(f"Fixed target_perc: {default_perc * 100:.1f}% (not meta-learned)")

    print("Loading dataset...")
    hf_dataset = load_data()

    meta_dataset = PairedDataset(
        Dataset(
            hf_dataset,
            tokenizer,
            seq_len=training_config.seq_len,
            eos_id=tokenizer.eos_token_id,
        )
    )

    dataloader = DataLoader(
        meta_dataset,
        batch_size=training_config.batch_size,
        collate_fn=collate_pairs,
    )

    batches_per_epoch = training_config.get("batches_per_epoch", None)
    checkpoint_path = os.path.join(checkpoint_dir, "meta_learned_mlps.pt")
    loss_fn = get_loss_func(training_config.get("loss_func", "mse"))
    print(
        f"Starting meta-training... (batches_per_epoch: {batches_per_epoch or 'unlimited'})"
    )
    layer_mlps, inner_lr_params, target_perc_params = meta_train(
        model=model,
        model_name=model_name,
        layer_mlps=layer_mlps,
        dataloader=dataloader,
        device=device,
        config=training_config,
        checkpoint_path=checkpoint_path,
        tokenizer=tokenizer,
        target_perc_params=target_perc_params,
        loss_fn=loss_fn,
        use_wandb=use_wandb,
    )

    if use_wandb:
        wandb.finish()

    return layer_mlps, inner_lr_params, target_perc_params


if __name__ == "__main__":
    layer_mlps, inner_lr_params, target_perc_params = main()
