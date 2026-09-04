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

from model.meta_learning import (
    KeyReconstructionConfig,
    LearnedInit,
    add_grad,
    inner_loop,
    prepare_kvs,
    setup_optimizer,
)
from model.mlp import MLP
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
from utils.rope import model_rope_cos_sin


def _metric_sums():
    return {
        "initial_support_loss": 0.0,
        "final_support_loss": 0.0,
        "meta_objective": 0.0,
    }


def _step(optimizer, count):
    """Average the accumulated gradient and step."""
    grads = []
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.div_(count)
                grads.append(param.grad.detach())
    grad_norm = (
        torch.linalg.vector_norm(
            torch.stack([torch.linalg.vector_norm(g.float()) for g in grads])
        ).item()
        if grads
        else 0.0
    )
    optimizer.step()
    optimizer.zero_grad()
    return grad_norm


def _batch_rope(model, cache, seq_len):
    keys = cache.layers[0].keys
    return model_rope_cos_sin(model, seq_len, keys.device, keys.dtype)


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
        input_ids = batch["input_ids"].to(device)
        with torch.no_grad():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )

        kvs = prepare_kvs(
            output.past_key_values,
            rope=_batch_rope(
                model,
                output.past_key_values,
                input_ids.shape[1],
            ),
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
            grad_norm = _step(optimizer, accum_count)
            optimiser_steps += 1
            log_step_metrics(
                wandb_run,
                window,
                accum_count,
                epoch,
                optimiser_steps,
                grad_norm,
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
        grad_norm = _step(optimizer, accum_count)
        optimiser_steps += 1
        log_step_metrics(
            wandb_run,
            window,
            accum_count,
            epoch,
            optimiser_steps,
            grad_norm,
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

    benchmark_specs = (
        (
            "ruler",
            cfg.eval_ruler,
            cfg.eval_ruler_samples,
            cfg.get("eval_ruler_tasks", None),
        ),
        (
            "longbench",
            cfg.eval_longbench,
            cfg.eval_longbench_samples,
            cfg.get("eval_longbench_tasks", None),
        ),
    )
    eval_runtime = None

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

        eval_every = max(1, int(cfg.eval_interval))
        save_every = max(1, int(cfg.get("checkpoint_interval", 10)))
        is_last_epoch = epoch + 1 == cfg.num_meta_epochs
        should_eval = (epoch + 1) % eval_every == 0 or is_last_epoch
        if (epoch + 1) % save_every == 0 or is_last_epoch or should_eval:
            save_checkpoint(mlps, ckpt_path, epoch)

        if should_eval:
            learned_init = LearnedInit.from_modules(mlps)
            for benchmark, enabled, samples, tasks in benchmark_specs:
                if not enabled:
                    continue
                if eval_runtime is None:
                    eval_runtime = MetaEvalRuntime(model, name, tokenizer, cfg)
                eval_benchmark(
                    eval_runtime,
                    epoch=epoch,
                    benchmark=benchmark,
                    samples=samples,
                    tasks=tasks,
                    learned_init=learned_init,
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


def _flatten_task_dict(task_dict):
    """get_task_dict nests group tasks; evaluate_tasks wants them flat."""
    tasks = []
    for value in task_dict.values():
        if isinstance(value, dict):
            tasks.extend(_flatten_task_dict(value))
        else:
            tasks.append(value)
    return tasks


class MetaEvalRuntime:
    """lm-eval state reused across the periodic meta-training benchmarks."""

    def __init__(self, model, name, tokenizer, cfg):
        from lm_eval_script import CompressedCacheHFLM

        self.model = model
        self.name = name
        self.cfg = cfg
        self.batch_size = int(getattr(cfg, "eval_batch_size", 1))
        self.target_cr = float(cfg.eval_target_cr)
        self._resolved = {}
        self.lm = CompressedCacheHFLM(
            cache_config=self._cache_config(None),
            pretrained=model,
            tokenizer=tokenizer,
            truncation=False,
            trust_remote_code=True,
            batch_size=self.batch_size,
            max_batch_size=self.batch_size,
        )

    def _cache_config(self, learned_init):
        from cache import CompressedCacheConfig, MLPValueCacheConfig

        return CompressedCacheConfig(
            key=build_key_cache_config(
                self.cfg,
                self.model.config.num_hidden_layers,
            ),
            value=MLPValueCacheConfig(
                target_compression_ratio=self.target_cr,
                num_epochs=self.cfg.inner_steps,
                learned_init=learned_init,
                use_residual=self.cfg.use_residual,
            ),
        )

    def _tasks(self, benchmark, tasks):
        """Resolve a benchmark's task objects once, then reuse them."""
        if benchmark not in self._resolved:
            from lm_eval.tasks import TaskManager, get_task_dict
            from lm_eval_script import get_tasks

            metadata = {"tokenizer": self.name}
            if benchmark == "ruler":
                metadata["max_seq_lengths"] = [
                    int(length)
                    for length in self.cfg.get("eval_seq_lengths", [4096])
                ]
            names = [benchmark] if tasks is None else tasks
            if isinstance(names, str):
                names = [names]
            manager = TaskManager(metadata=metadata)
            resolved = get_task_dict(
                get_tasks(list(names), print_tasks=False), manager
            )
            self._resolved[benchmark] = (_flatten_task_dict(resolved), manager)
        return self._resolved[benchmark]

    def evaluate(self, benchmark, tasks, samples, learned_init):
        from lm_eval_script import evaluate_tasks

        eval_tasks, manager = self._tasks(benchmark, tasks)
        self.lm.set_cache_config(self._cache_config(learned_init))
        return evaluate_tasks(
            self.lm,
            eval_tasks,
            batch_size=self.batch_size,
            task_manager=manager,
            limit=samples,
        )


def eval_benchmark(
    eval_runtime,
    epoch,
    benchmark,
    samples,
    tasks=None,
    learned_init=None,
    wandb_run=None,
    optimiser_steps=0,
):
    """Evaluate a benchmark with the current meta initialisation."""
    print(
        f"Running {benchmark} evaluation (epoch {epoch}, "
        f"{samples} samples per task, target_cr={eval_runtime.target_cr}, "
        f"batch_size={eval_runtime.batch_size}, tasks={tasks or benchmark})..."
    )

    results = eval_runtime.evaluate(benchmark, tasks, samples, learned_init)

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
    parts = [
        f"seq{cfg.seq_len}",
        f"steps{cfg.inner_steps}",
        f"mlr{cfg.meta_lr}",
        f"cr{float(cfg.eval_target_cr)}",
    ]
    key_config = build_key_reconstruction_config(cfg)
    if key_config is not None:
        parts.append(f"xkvkeys{key_config.compression_ratio}cr")
    parts.append("residual" if cfg.use_residual else "noresidual")
    return "_".join(parts)


def main():
    config = load_config()
    cfg = config.training

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    print(f"Loading model: {config.model.name}")

    model_name = config.model.name
    model_dir = model_name.split("/")[-1]

    run_name = build_run_name(cfg)
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
