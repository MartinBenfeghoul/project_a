"""
First-order MAML-style meta-learning for fast support-set memorisation.

Each per-layer MLP learns to predict value vectors from key vectors. The inner
loop adapts MLP weights to a support KV cache; the outer loop also optimizes
post-adaptation support loss, so the learned initialization is selected for
rapid memorisation of the same sequence.
"""

import itertools
import os
import time

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.mlp import MLP
from model.meta_learning import (
    KeyReconstructionConfig,
    add_grad,
    inner_loop,
    prepare_kvs,
    setup_optimizer,
)
from utils.data import Dataset, collate, load_data
from utils.logging import (
    average_metrics,
    init_wandb,
    log_benchmark_scores,
    log_epoch_metrics,
    log_step_metrics,
    save_checkpoint,
)
from utils.model import extract_kv_linear_init, get_model_and_tokenizer
from utils.rope import get_rope_theta


def _metric_sums():
    return {
        "initial_support_loss": 0.0,
        "final_support_loss": 0.0,
        "meta_objective": 0.0,
    }


def _step(
    optimizer,
    count,
):
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.div_(count)
    optimizer.step()
    optimizer.zero_grad()


def run_epoch(
    model,
    mlps,
    data_iter,
    optimizer,
    cfg,
    epoch,
    device,
    inner_dtype,
    residual_cr,
    key_config=None,
    wandb_run=None,
    optimiser_steps=0,
):
    inner_lr = cfg.inner_lr
    inner_steps = cfg.inner_steps
    log_interval = cfg.log_interval
    batches_per_epoch = cfg.batches_per_epoch
    accum_steps = int(cfg.grad_accum_steps)
    rope_theta = cfg.rope_theta

    sums = _metric_sums()
    window = _metric_sums()
    batch_count = 0
    accum_count = 0
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(
        tqdm(
            itertools.islice(data_iter, batches_per_epoch),
            total=batches_per_epoch,
        )
    ):
        start = time.time()
        attention_mask = batch["attention_mask"].to(device)
        with torch.no_grad():
            output = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=attention_mask,
                use_cache=True,
            )

        kvs = prepare_kvs(
            output.past_key_values,
            rope_theta=rope_theta,
            dtype=inner_dtype,
            key_config=key_config,
            padding_mask=attention_mask,
        )

        meta_params, metrics = inner_loop(
            mlps,
            kvs,
            inner_lr,
            inner_steps,
            residual_cr=residual_cr,
        )
        del output, kvs
        batch_ms = (time.time() - start) * 1000

        for param, grad in zip(meta_params, metrics["param_grads"]):
            add_grad(param, grad)

        for name in sums:
            sums[name] += metrics[name]
            window[name] += metrics[name]
        accum_count += 1
        if accum_count >= accum_steps:
            _step(
                optimizer,
                accum_count,
            )
            optimiser_steps += 1
            log_step_metrics(
                wandb_run, window, accum_count, epoch, optimiser_steps
            )
            window = _metric_sums()
            accum_count = 0

        batch_count += 1

        if batch_idx % log_interval == 0:
            print(
                f"Epoch {epoch}, Batch {batch_idx}, "
                f"L0: {metrics['initial_support_loss']:.6f}, "
                f"L{inner_steps}: {metrics['final_support_loss']:.6f}, "
                f"Meta Objective: {metrics['meta_objective']:.6f}, "
                f"Time: {batch_ms:.1f}ms"
            )

    if accum_count > 0:
        _step(
            optimizer,
            accum_count,
        )
        optimiser_steps += 1
        log_step_metrics(
            wandb_run, window, accum_count, epoch, optimiser_steps
        )

    return sums, batch_count, optimiser_steps


def meta_train(
    model,
    name,
    mlps,
    dataloader,
    device,
    cfg,
    ckpt_path,
    tokenizer,
    wandb_run=None,
):
    optimizer = setup_optimizer(mlps, cfg)
    model.eval()
    inner_dtype = getattr(torch, str(cfg.inner_adaptation_dtype))
    print(f"FP32 meta parameters; {inner_dtype} inner adaptation")
    key_config = build_key_reconstruction_config(cfg)
    if key_config is not None:
        print(
            "Inner loop trains on xKV-reconstructed keys "
            f"(comp_ratio={key_config.compression_ratio}, "
            f"layer_group_size={key_config.layer_group_size}, "
            f"svd_backend={key_config.svd_backend})"
        )
    residual_cr = float(cfg.eval_target_cr)
    print(
        "Meta objective uses post-residual reconstruction loss "
        f"(target_cr={residual_cr})"
    )

    data_iter = iter(dataloader)
    optimiser_steps = 0
    for epoch in range(cfg.num_meta_epochs):
        start = time.time()
        sums, batch_count, optimiser_steps = run_epoch(
            model,
            mlps,
            data_iter,
            optimizer,
            cfg,
            epoch,
            device,
            inner_dtype,
            residual_cr=residual_cr,
            key_config=key_config,
            wandb_run=wandb_run,
            optimiser_steps=optimiser_steps,
        )

        epoch_sec = time.time() - start
        avgs = average_metrics(sums, batch_count)
        log_epoch_metrics(
            wandb_run, avgs, epoch, batch_count, optimiser_steps
        )
        print(
            f"Epoch {epoch} complete. "
            f"Average L0: {avgs['initial_support_loss']:.6f}, "
            f"Average L{cfg.inner_steps}: {avgs['final_support_loss']:.6f}, "
            f"Average Meta Objective: {avgs['meta_objective']:.6f}, "
            f"Time: {epoch_sec:.1f}s"
        )

        save_checkpoint(mlps, ckpt_path, epoch)

        eval_every = max(1, int(cfg.eval_interval))
        if (epoch + 1) % eval_every == 0 or (epoch + 1 == cfg.num_meta_epochs):
            base, ext = os.path.splitext(ckpt_path)
            epoch_ckpt = f"{base}_epoch{epoch}{ext}"
            for benchmark, enabled, samples in (
                ("ruler", cfg.eval_ruler, cfg.eval_ruler_samples),
                (
                    "longbench",
                    cfg.eval_longbench,
                    cfg.eval_longbench_samples,
                ),
            ):
                if not enabled:
                    continue
                eval_benchmark(
                    model=model,
                    name=name,
                    tokenizer=tokenizer,
                    cfg=cfg,
                    ckpt_path=epoch_ckpt,
                    epoch=epoch,
                    benchmark=benchmark,
                    samples=samples,
                    wandb_run=wandb_run,
                    optimiser_steps=optimiser_steps,
                )


def build_key_reconstruction_config(cfg):
    if not cfg.get("train_on_reconstructed_keys", False):
        return None
    return KeyReconstructionConfig(
        compression_ratio=float(cfg.eval_target_cr),
        layer_group_size=int(cfg.get("xkv_layer_group_size", 4)),
        svd_backend=str(cfg.get("xkv_svd_backend", "cholqr")),
    )


def build_key_cache_config(cfg, num_layers):
    from cache import BaselineCacheConfig, XKVCacheConfig

    key_config = build_key_reconstruction_config(cfg)
    if key_config is None:
        return BaselineCacheConfig()
    return XKVCacheConfig(
        compression_ratio=key_config.compression_ratio,
        layer_group_size=key_config.layer_group_size,
        svd_backend=key_config.svd_backend,
        num_layers=num_layers,
    )


def build_value_cache_config(cfg, ckpt_path, target_cr):
    """Build inference settings that match the meta-training inner loop."""
    from cache import MLPValueCacheConfig

    return MLPValueCacheConfig(
        target_compression_ratio=target_cr,
        num_epochs=cfg.inner_steps,
        meta_weights_path=ckpt_path,
        use_residual=cfg.use_residual,
    )


def eval_benchmark(
    model,
    name,
    tokenizer,
    cfg,
    ckpt_path,
    epoch,
    benchmark,
    samples,
    wandb_run=None,
    optimiser_steps=0,
):
    """Evaluate a benchmark with the current meta initialisation."""
    from lm_eval import evaluator
    from lm_eval.tasks import TaskManager
    from lm_eval_script import (
        CompressedCacheHFLM,
        GEN_KWARGS,
        get_device,
        get_tasks,
    )
    from cache import CompressedCacheConfig
    target_cr = float(cfg.eval_target_cr)
    eval_batch_size = int(getattr(cfg, "eval_batch_size", 1))
    print(
        f"Running {benchmark} evaluation (epoch {epoch}, "
        f"{samples} samples per task, target_cr={target_cr}, "
        f"batch_size={eval_batch_size})..."
    )

    lm = CompressedCacheHFLM(
        cache_config=CompressedCacheConfig(
            key=build_key_cache_config(
                cfg,
                model.config.num_hidden_layers,
            ),
            value=build_value_cache_config(
                cfg,
                ckpt_path,
                target_cr,
            ),
        ),
        pretrained=model,
        tokenizer=tokenizer,
        truncation=False,
        trust_remote_code=True,
        batch_size=eval_batch_size,
        max_batch_size=eval_batch_size,
    )

    results = evaluator.simple_evaluate(
        model=lm,
        gen_kwargs=GEN_KWARGS,
        tasks=get_tasks([benchmark]),
        num_fewshot=0,
        batch_size=eval_batch_size,
        max_batch_size=eval_batch_size,
        device=get_device(lm),
        task_manager=TaskManager(metadata={"tokenizer": name}),
        limit=samples,
    )

    scores = {}
    for task, result in results["results"].items():
        for key, value in result.items():
            if (
                isinstance(value, (int, float))
                and "stderr" not in key
                and key not in ("alias", "samples")
            ):
                scores[task] = value
                break

    avg_score = sum(scores.values()) / len(scores) if scores else 0.0
    print(
        f"{benchmark} eval epoch {epoch}: "
        f"avg={avg_score:.4f}, per-task={scores}"
    )
    log_benchmark_scores(
        wandb_run,
        benchmark,
        avg_score,
        scores,
        epoch,
        optimiser_steps,
    )


def load_config():
    """
    CLI args use dotlist notation, e.g.:
        python meta_learning.py training.batch_size=4
    """
    config_path = os.environ.get("META_CONFIG", "config/meta_learning.yaml")
    base_config = OmegaConf.load(config_path)
    cli_config = OmegaConf.from_cli()
    return OmegaConf.merge(base_config, cli_config)


def init_mlps(model, cfg, device):
    num_heads = model.config.num_key_value_heads
    dim = model.config.hidden_size // model.config.num_attention_heads
    num_layers = model.config.num_hidden_layers
    linear_init = (
        extract_kv_linear_init(model)
        if cfg.use_residual
        else [None] * num_layers
    )
    mlps = []
    for layer_idx in range(num_layers):
        mlp = MLP(
            num_heads=num_heads,
            head_dim=dim,
            use_residual=cfg.use_residual,
        ).to(device=device)
        if cfg.use_residual:
            with torch.no_grad():
                mlp.W_linear.copy_(
                    linear_init[layer_idx].to(
                        device=device,
                        dtype=mlp.W_linear.dtype,
                    )
                )
        mlps.append(mlp)
    return mlps


def build_run_name(cfg) -> str:
    """Name a meta-learning run after the knobs that define it."""
    name = f"seq{cfg.seq_len}_steps{cfg.inner_steps}_mlr{cfg.meta_lr}_"
    key_config = build_key_reconstruction_config(cfg)
    if key_config is not None:
        name += f"xkvkeys{key_config.compression_ratio}cr_"
    return name


def main():
    config = load_config()
    cfg = config.training

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    print(f"Loading model: {config.model.name}")

    model_name = config.model.name
    model_dir = model_name.split("/")[-1]

    run_name = build_run_name(cfg)
    if cfg.use_residual:
        run_name += "_perhead_residual"
    base_dir = os.path.join("checkpoints", model_dir, run_name)
    run_dir = base_dir
    idx = 0
    while os.path.exists(run_dir):
        run_dir = f"{base_dir}_{idx}"
        idx += 1
    os.makedirs(run_dir)
    print(f"Checkpoints will be saved to: {run_dir}/")

    model, tokenizer = get_model_and_tokenizer(model_name)

    num_heads = model.config.num_key_value_heads
    dim = model.config.hidden_size // model.config.num_attention_heads
    num_layers = model.config.num_hidden_layers
    cfg.rope_theta = get_rope_theta(model.config)

    config_path = os.path.join(run_dir, "config.yaml")
    OmegaConf.save(config, config_path)
    print(f"Config saved to: {config_path}")

    print(
        f"Model config: {num_layers} layers, {num_heads} KV heads, "
        f"{dim} head_dim"
    )

    mlps = init_mlps(model, cfg, device)

    data_cfg = config.data
    print(f"Loading dataset: {data_cfg.path}")
    stream = load_data(
        dataset_path=data_cfg.path,
        subset_name=data_cfg.subset,
        shuffle_buffer_size=data_cfg.shuffle_buffer_size,
    )

    dataset = Dataset(
        stream,
        tokenizer,
        seq_len=cfg.seq_len,
        eos_id=tokenizer.eos_token_id,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        collate_fn=collate,
    )

    ckpt_path = os.path.join(run_dir, "meta_mlps.pt")
    print(
        "Starting meta-training... "
        f"(batches_per_epoch: {cfg.batches_per_epoch})"
    )
    wandb_run = init_wandb(config, os.path.basename(run_dir))
    try:
        meta_train(
            model=model,
            name=model_name,
            mlps=mlps,
            dataloader=dataloader,
            device=device,
            cfg=cfg,
            ckpt_path=ckpt_path,
            tokenizer=tokenizer,
            wandb_run=wandb_run,
        )
    finally:
        wandb_run.finish()


if __name__ == "__main__":
    main()
