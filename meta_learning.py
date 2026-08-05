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
import wandb

from model.mlp import MLP
from utils import (
    Dataset,
    collate,
    extract_kv_linear_init,
    generate_run_name,
    get_model_and_tokenizer,
    load_data,
    save_checkpoint,
    init_wandb,
    add_grad,
    constrain_lrs,
    expand_lrs,
    get_rope_theta,
    inner_loop,
    prepare_kvs,
    setup_optimizer,
)


def _metric_sums():
    return {
        "initial_support_loss": 0.0,
        "final_support_loss": 0.0,
        "meta_objective": 0.0,
    }


def _step(
    optimizer,
    sums,
    count,
    epoch,
    step,
    use_wandb,
):
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.div_(count)
    optimizer.step()
    optimizer.zero_grad()
    step += 1
    if use_wandb:
        wandb.log(
            {
                "trainer/update_step": step,
                "trainer/epoch": epoch,
                **{
                    f"train/{name}": total / count
                    for name, total in sums.items()
                },
            }
        )
    return step


def run_epoch(
    model,
    mlps,
    data_iter,
    optimizer,
    cfg,
    epoch,
    step,
    device,
    use_wandb,
    inner_dtype,
    raw_lrs=None,
    residual_cr=None,
):
    inner_lr = cfg.inner_lr
    inner_steps = cfg.inner_steps
    log_interval = cfg.log_interval
    batches_per_epoch = cfg.batches_per_epoch
    accum_steps = int(cfg.grad_accum_steps)
    rope_theta = cfg.rope_theta

    sums = _metric_sums()
    batch_count = 0
    accum_count = 0
    update_sums = _metric_sums()
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(
        tqdm(
            itertools.islice(data_iter, batches_per_epoch),
            total=batches_per_epoch,
        )
    ):
        param_lrs = expand_lrs(mlps, constrain_lrs(raw_lrs, cfg))
        start = time.time()
        with torch.no_grad():
            output = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                use_cache=True,
            )

        kvs = prepare_kvs(
            output.past_key_values,
            rope_theta=rope_theta,
            dtype=inner_dtype,
        )

        meta_params, metrics = inner_loop(
            mlps,
            kvs,
            param_lrs if param_lrs is not None else inner_lr,
            inner_steps,
            outer_lrs=raw_lrs,
            residual_cr=residual_cr,
        )
        del output, kvs
        batch_ms = (time.time() - start) * 1000

        for param, grad in zip(meta_params, metrics["param_grads"]):
            add_grad(param, grad)

        if raw_lrs is not None:
            for lr_param, grad in zip(raw_lrs, metrics["lr_grads"]):
                add_grad(lr_param, grad)

        for name in sums:
            sums[name] += metrics[name]
            update_sums[name] += metrics[name]
        accum_count += 1
        if accum_count >= accum_steps:
            step = _step(
                optimizer,
                update_sums,
                accum_count,
                epoch,
                step,
                use_wandb,
            )
            accum_count = 0
            update_sums = _metric_sums()

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
        step = _step(
            optimizer,
            update_sums,
            accum_count,
            epoch,
            step,
            use_wandb,
        )

    return sums, batch_count, step


def meta_train(
    model,
    name,
    mlps,
    dataloader,
    device,
    cfg,
    ckpt_path,
    tokenizer,
    use_wandb=False,
):
    raw_lrs, optimizer = setup_optimizer(mlps, cfg)
    model.eval()
    step = 0
    inner_dtype = getattr(torch, str(cfg.inner_adaptation_dtype))
    print(f"FP32 meta parameters; {inner_dtype} inner adaptation")
    residual_cr = (
        float(cfg.eval_target_cr) if cfg.meta_loss_post_residual else None
    )
    if residual_cr is not None:
        print(
            "Meta objective uses post-residual reconstruction loss "
            f"(target_cr={residual_cr})"
        )

    if use_wandb:
        wandb.define_metric("trainer/update_step")
        wandb.define_metric("train/*", step_metric="trainer/update_step")
        wandb.define_metric("trainer/epoch")
        wandb.define_metric("epoch/*", step_metric="trainer/epoch")
        wandb.define_metric("benchmark/*", step_metric="trainer/epoch")

    data_iter = iter(dataloader)
    for epoch in range(cfg.num_meta_epochs):
        start = time.time()
        sums, batch_count, step = run_epoch(
            model,
            mlps,
            data_iter,
            optimizer,
            cfg,
            epoch,
            step,
            device,
            use_wandb,
            inner_dtype,
            raw_lrs=raw_lrs,
            residual_cr=residual_cr,
        )

        epoch_sec = time.time() - start
        avgs = {
            metric: total / max(batch_count, 1)
            for metric, total in sums.items()
        }
        print(
            f"Epoch {epoch} complete. "
            f"Average L0: {avgs['initial_support_loss']:.6f}, "
            f"Average L{cfg.inner_steps}: {avgs['final_support_loss']:.6f}, "
            f"Average Meta Objective: {avgs['meta_objective']:.6f}, "
            f"Time: {epoch_sec:.1f}s"
        )

        if use_wandb:
            wandb.log(
                {
                    **{
                        f"epoch/avg_{name}": value
                        for name, value in avgs.items()
                    },
                    "trainer/epoch": epoch,
                    "trainer/update_step": step,
                    "perf/epoch_time_sec": epoch_sec,
                },
            )

        with torch.no_grad():
            saved_lrs = expand_lrs(mlps, constrain_lrs(raw_lrs, cfg))
        save_checkpoint(mlps, saved_lrs, ckpt_path, epoch)

        eval_every = max(1, int(cfg.eval_interval))
        if (epoch + 1) % eval_every == 0 or (
            epoch + 1 == cfg.num_meta_epochs
        ):
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
                    device=device,
                    epoch=epoch,
                    step=step,
                    use_wandb=use_wandb,
                    benchmark=benchmark,
                    samples=samples,
                )


def build_cache_args(model_cfg, cfg, ckpt_path, device, target_cr):
    """Build inference settings that match the meta-training inner loop."""
    num_layers = model_cfg.num_hidden_layers
    num_heads = model_cfg.num_key_value_heads
    return {
        "cache_type": "mlp",
        "num_layers_per_mlp": [2] * num_layers,
        "hidden_factors_per_mlp": [1] * num_layers,
        "num_heads_per_mlp": [num_heads] * num_layers,
        "target_perc": [None] * num_layers,
        "per_sequence": False,
        "lr": cfg.inner_lr,
        "device": device,
        "optimizer": "adam",
        "loss_func": "mse",
        "num_epochs": cfg.inner_steps,
        "meta_weights_path": ckpt_path,
        "un_rope": True,
        "rope_theta": get_rope_theta(model_cfg),
        "use_residual": cfg.use_residual,
        "freeze_W_linear": False,
        "target_cr": target_cr,
    }


def eval_benchmark(
    model,
    name,
    tokenizer,
    cfg,
    ckpt_path,
    device,
    epoch,
    step,
    use_wandb,
    benchmark,
    samples,
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
    from utils import Logger

    target_cr = float(cfg.eval_target_cr)
    print(
        f"Running {benchmark} evaluation (epoch {epoch}, "
        f"{samples} samples per task, target_cr={target_cr})..."
    )

    logger = Logger()
    logger.prefill_events = []
    logger.decode_events = []

    lm = CompressedCacheHFLM(
        key_cache_kwargs={"cache_type": "baseline"},
        value_cache_kwargs=build_cache_args(
            model.config,
            cfg,
            ckpt_path,
            device,
            target_cr,
        ),
        eviction_keep_ratio=1.0,
        logger=logger,
        adjust_key_value_comp_ratio=False,
        pretrained=model,
        tokenizer=tokenizer,
        truncation=False,
        trust_remote_code=True,
    )

    results = evaluator.simple_evaluate(
        model=lm,
        gen_kwargs=GEN_KWARGS,
        tasks=get_tasks([benchmark]),
        num_fewshot=0,
        batch_size=1,
        max_batch_size=1,
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

    if use_wandb:
        log_data = {
            f"benchmark/{benchmark}_avg": avg_score,
            f"benchmark/{benchmark}_target_cr": target_cr,
            "trainer/epoch": epoch,
            "trainer/update_step": step,
        }
        for task, score in scores.items():
            log_data[f"benchmark/{benchmark}/{task}"] = score
        wandb.log(log_data)


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
    per_head = cfg.per_head_residual
    linear_init = (
        extract_kv_linear_init(model, per_head=per_head)
        if cfg.use_residual
        else [None] * num_layers
    )
    mlps = []
    for layer_idx in range(num_layers):
        mlp = MLP(
            num_heads=num_heads,
            head_dim=dim,
            use_residual=cfg.use_residual,
            per_head_residual=per_head,
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


def main():
    config = load_config()
    cfg = config.training

    device = "cuda" if torch.cuda.is_available() else "cpu"

    use_wandb = init_wandb(config)
    if use_wandb:
        print("wandb logging enabled")

    print(f"Using device: {device}")
    print(f"Loading model: {config.model.name}")

    model_name = config.model.name
    model_dir = model_name.split("/")[-1]

    run_name = generate_run_name(config)
    if cfg.use_residual:
        residual_type = "perhead" if cfg.per_head_residual else "joint"
        run_name += f"_{residual_type}_residual"
    base_dir = os.path.join("checkpoints", model_dir, run_name)
    run_dir = base_dir
    idx = 0
    while os.path.exists(run_dir):
        run_dir = f"{base_dir}_{idx}"
        idx += 1
    os.makedirs(run_dir)
    print(f"Checkpoints will be saved to: {run_dir}/")

    model, tokenizer = get_model_and_tokenizer(model_name, device)

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
    meta_train(
        model=model,
        name=model_name,
        mlps=mlps,
        dataloader=dataloader,
        device=device,
        cfg=cfg,
        ckpt_path=ckpt_path,
        tokenizer=tokenizer,
        use_wandb=use_wandb,
    )

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
