import os
import json
import argparse
import time
from omegaconf import OmegaConf
import torch

from lm_eval import evaluator
from lm_eval.utils import make_table
from lm_eval.tasks import TaskManager, get_task_dict

from cache import CompressedCacheHFLM
from model.attention_predictor import (
    get_attn_predictor_hook_handles,
    apply_attn_predictor_config,
)
from model.selective_attention import install_selective_attention
from utils.logging import make_hooks
from utils import (
    Logger,
    get_model_and_tokenizer,
    extract_kv_linear_init,
    get_device,
    get_device_type,
    list_of_strings,
    get_output_path,
    extract_and_save_stats,
    extract_and_save_efficiency_stats,
    log_live_value_mlp_training_wandb,
    init_lm_eval_wandb,
    filter_tasks_by_min_seq_len
)

GEN_KWARGS = {
    "do_sample": False,
    "use_cache": True,
    "logits_to_keep": 1,
}


def get_gen_kwargs(args):
    gen_kwargs = dict(GEN_KWARGS)
    if args.k_cache_type == "surprise_lr":
        gen_kwargs["logits_to_keep"] = 0
    return gen_kwargs


def get_tasks(tasks, print_tasks=True):
    if len(tasks) == 1:
        task_conf = OmegaConf.load("config/tasks.yaml")
        if tasks[0] in task_conf:
            tasks = OmegaConf.to_container(task_conf[tasks[0]], resolve=True)
    if print_tasks:
        print(f"Evaluating tasks: {tasks}")
    return tasks


@torch.no_grad()
def main(args):
    device_type = get_device_type()
    device = torch.device(device_type)

    model, tokenizer = get_model_and_tokenizer(args.model_name, device)
    selective_handles = (
        install_selective_attention(model)
        if args.selective_reconstruction
        else []
    )
    logger = Logger()
    logger.prefill_events = []
    logger.decode_events = []
    logger.recorded_k_timing = False
    logger.recorded_cr = False

    if args.dump_full_kv_dir is not None:
        if args.k_cache_type != "baseline" or args.v_cache_type != "baseline":
            raise ValueError(
                "--dump_full_kv_dir requires --k_cache_type=baseline "
                "and --v_cache_type=baseline."
            )
        tasks = get_tasks(args.tasks, print_tasks=False)
        if len(tasks) != 1 or args.limit != 1:
            raise ValueError(
                "--dump_full_kv_dir currently only supports a single task "
                "and a limit of 1 sample. "
                f"Got tasks={tasks} and limit={args.limit}."
            )
        args.use_wandb = False
        os.makedirs(args.dump_full_kv_dir, exist_ok=True)

    rope_theta = getattr(model.config, "rope_theta", 500_000.0)

    attn_predictor_hook_handles = get_attn_predictor_hook_handles(args, model)

    num_layers = model.config.num_hidden_layers
    key_cache_kwargs = {
        "cache_type": args.k_cache_type,
        "decomposition_method": args.decomposition_method,
        "local_window": args.local_window,
        "log_timing_stats": args.log_key_cache_timing,
        "comp_ratio": args.comp_ratio,
        "energy_threshold": args.energy_threshold,
        "rank_selection": args.rank_selection,
        "lr": args.k_lr,
        "decomp_n_iter": args.decomp_n_iter,
        "gamma": args.gamma,
        "min_size": 8,
        "kmeans_cluster_size": args.kmeans_cluster_size,
        "kmeans_n_iter": args.kmeans_n_iter,
        "kmeans_init": args.kmeans_init,
        "kmeans_dtype": args.kmeans_dtype,
        "kmeans_avg_heads": args.kmeans_avg_heads,
        "kmeans_per_head": args.kmeans_per_head,
        "layer_group_size": args.xkv_layer_group_size,
        "xkv_svd_backend": args.xkv_svd_backend,
        "num_layers": num_layers,
        "unrope_keys": args.un_rope,
        "selective_reconstruction": args.selective_reconstruction,
        "rope_theta": rope_theta,
        "quantise_a": args.k_quantise_a,
        "quantise_b": args.k_quantise_b,
        "compressor_bits": args.k_compressor_bits,
    }

    target_perc_per_layer = (
        args.target_perc
        if isinstance(args.target_perc, list)
        else [args.target_perc] * num_layers
    )
    value_cache_kwargs = {
        "cache_type": args.v_cache_type,
        "num_layers_per_mlp": [args.num_layers_per_mlp] * num_layers,
        "hidden_factors_per_mlp": [args.hidden_factors_per_mlp] * num_layers,
        "num_heads_per_mlp": [args.num_heads_per_mlp] * num_layers,
        "per_sequence": args.per_sequence,
        "target_perc": target_perc_per_layer,
        "lr": args.v_lr,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "optimizer": args.optimizer,
        "loss_func": args.loss_func,
        "num_epochs": args.num_epochs,
        "meta_weights_path": args.meta_weights_path,
        "value_mlp_weights_path": args.value_mlp_weights_path,
        "un_rope": args.un_rope,
        "rope_theta": rope_theta,
        "global_compression": args.global_compression,
        "use_residual": args.use_residual,
        "intermediate_activation": args.intermediate_activation,
        "linear_only": args.linear_only,
        "freeze_W_linear": args.freeze_W_linear,
        "target_cr": args.target_cr,
        "turboquant_residuals": args.v_turboquant_residuals,
        "compressor_bits": args.v_compressor_bits,
    }
    if (args.use_residual or args.linear_only) and args.v_cache_type == "mlp":
        value_cache_kwargs["W_linear_per_layer"] = extract_kv_linear_init(
            model, per_head=args.per_head_kv_linear
        )

    model.eval()

    model_baseline_mem = 0
    if args.log_efficiency_metrics and torch.cuda.is_available():
        # warm up cuda kernels before benchmarking to avoid inflated first-pass times
        print("Warming up GPU...")
        _warmup_ids = torch.ones((1, 32), dtype=torch.long, device=device)
        for _ in range(3):
            model(_warmup_ids)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        # model-weights-only footprint
        model_baseline_mem = torch.cuda.memory_allocated()

    args.tasks = get_tasks(args.tasks)
    use_wandb = init_lm_eval_wandb(args)

    pre_hook, post_hook = make_hooks(
        logger,
        measure_latency=args.log_efficiency_metrics,
        measure_gpu_memory=args.log_efficiency_metrics,
        model_baseline_mem=model_baseline_mem,
        use_wandb=use_wandb,
        dump_full_kv_dir=args.dump_full_kv_dir,
        model_name=args.model_name,
        task_name=args.tasks[0],
        rope_theta=rope_theta,
    )
    metric_hook_handles = [
        model.register_forward_pre_hook(pre_hook, with_kwargs=True),
        model.register_forward_hook(post_hook, with_kwargs=True),
    ]

    lm = CompressedCacheHFLM(
        key_cache_kwargs=key_cache_kwargs,
        value_cache_kwargs=value_cache_kwargs,
        eviction_keep_ratio=args.eviction_keep_ratio,
        logger=logger,
        adjust_key_value_comp_ratio=args.adjust_key_value_comp_ratio,
        pretrained=model,
        tokenizer=tokenizer,
        max_length=None,
        batch_size=args.batch_size,
        max_batch_size=args.batch_size,
        truncation=False,
        trust_remote_code=True,
    )
    metadata = {"tokenizer": args.model_name}
    if args.max_seq_lengths is not None:
        metadata["max_seq_lengths"] = args.max_seq_lengths
    tm = TaskManager(metadata=metadata)

    eval_tasks = args.tasks
    if args.min_seq_len is not None:
        task_dict = get_task_dict(args.tasks, tm)
        eval_tasks = filter_tasks_by_min_seq_len(
            task_dict, tokenizer, args.min_seq_len
        )

    if args.log_efficiency_metrics:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        start_time = time.perf_counter()

    results = evaluator.simple_evaluate(
        model=lm,
        gen_kwargs=get_gen_kwargs(args),
        tasks=eval_tasks,
        num_fewshot=0,
        batch_size=args.batch_size,
        max_batch_size=args.batch_size,
        device=get_device(lm),
        task_manager=tm,
        limit=args.limit,
    )

    cr_values = logger.get_log_list("crs")
    if cr_values:
        cr_mean, cr_std = logger.get_log_mean("crs", std=True)
        compression_metrics = {
            "compression_ratio_mean": float(cr_mean),
            "compression_ratio_std": float(cr_std),
            "n_compression_ratio_samples": len(cr_values),
        }
        print("Compression metrics:", compression_metrics)
        results["results"]["compression_metrics"] = compression_metrics

    for label in ["value_recon_mse", "key_recon_mse"]:
        mse_values = logger.get_log_list(label)
        if mse_values:
            mse_mean, mse_std = logger.get_log_mean(label, std=True)
            metrics = {
                "recon_mse_mean": float(mse_mean),
                "recon_mse_std": float(mse_std),
            }
            print(f"{label}:", metrics)
            results["results"][f"{label}"] = metrics

    print(make_table(results))

    results = extract_and_save_stats(logger, results)
    if args.log_efficiency_metrics:
        results = extract_and_save_efficiency_stats(
            logger, results, model_baseline_mem, start_time
        )
    results["results"]["config"] = vars(args)

    if args.debug:
        print("Debug mode — not saving results.")
        return results

    output_dir = os.path.join(
        args.output_dir, args.model_name.replace("/", "_")
    )
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir,
        f"lm_eval_kc_{args.k_cache_type}_vc_{args.v_cache_type}_ml_{bool(args.meta_weights_path)}"
        + "_{}.json",
    )
    output_path = get_output_path(output_path)
    with open(output_path, "w") as f:
        json.dump(results["results"], f, ensure_ascii=False, indent=4)
    print(f"Results saved to {output_path}")
    for handle in (
        metric_hook_handles + attn_predictor_hook_handles + selective_handles
    ):
        handle.remove()
    if use_wandb:
        import wandb

        if logger.value_mlp_log_events:
            log_live_value_mlp_training_wandb(
                logger.value_mlp_log_events,
                logger.request_idx + 1,
                log_tables=True,
            )
        wandb.finish()
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="LM eval harness script")
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
    )
    parser.add_argument("-o", "--output_dir", type=str, default="./results")
    parser.add_argument(
        "-t", "--tasks", type=list_of_strings, default=["lm_eval"]
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of samples per task.",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--max_seq_lengths",
        type=int,
        nargs="+",
        default=None,
        help="Sequence lengths for RULER tasks.",
    )
    parser.add_argument(
        "--min_seq_len",
        type=int,
        default=None,
        help="Only evaluate samples whose prompt is at least this many tokens.",
    )
    parser.add_argument("--dump_full_kv_dir", type=str, default=None)
    parser.add_argument("--log_efficiency_metrics", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--debug", action="store_true")

    # key cache
    parser.add_argument(
        "-kc", "--k_cache_type", type=str, default="surprise_lr"
    )
    parser.add_argument(
        "--decomposition_method",
        type=str,
        default="svd",
        choices=["svd", "lora"],
    )
    parser.add_argument("-r", "--comp_ratio", type=float, default=2.0)
    parser.add_argument(
        "--adjust_key_value_comp_ratio",
        action="store_true",
        help=(
            "Use different low-rank compression targets for key and value "
            "caches while preserving the requested combined --comp_ratio."
        ),
    )
    parser.add_argument("-e", "--energy_threshold", type=float, default=0.95)
    parser.add_argument(
        "--rank_selection",
        type=str,
        default="comp_ratio",
        choices=["comp_ratio", "energy"],
    )
    parser.add_argument("--k_lr", type=float, default=1e-2)
    parser.add_argument("--decomp_n_iter", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=3.0)
    parser.add_argument("--local_window", type=int, default=0)
    parser.add_argument(
        "--xkv_layer_group_size",
        type=int,
        default=4,
        help="Number of adjacent layers to jointly compress when --k_cache_type=xkv.",
    )
    parser.add_argument(
        "--xkv_svd_backend",
        choices=["cholqr", "linalg"],
        default="cholqr",
    )
    parser.add_argument("--selective_reconstruction", action="store_true")
    parser.add_argument(
        "--kmeans_cluster_size",
        type=float,
        default=None,
    )
    parser.add_argument("--kmeans_n_iter", type=int, default=3)
    parser.add_argument(
        "--kmeans_init",
        type=str,
        default="infllm",
        choices=["infllm", "random", "kmeans++"],
    )
    parser.add_argument(
        "--kmeans_dtype",
        type=str,
        default="float32",
        choices=["float16", "float32", "bfloat16"],
    )
    parser.add_argument("--kmeans_avg_heads", action="store_true")
    parser.add_argument("--kmeans_per_head", action="store_true")
    parser.add_argument("--log_key_cache_timing", action="store_true")
    parser.add_argument(
        "--k_quantise_a",
        action="store_true",
        help="Quantise low-rank key-cache A factors with TurboQuant.",
    )
    parser.add_argument(
        "--k_quantise_b",
        action="store_true",
        help="Quantise low-rank key-cache B factors with TurboQuant.",
    )
    parser.add_argument(
        "--k_compressor_bits",
        type=int,
        default=4,
        help="Bits per rotated coordinate for TurboQuant key cache or low-rank factor quantisation.",
    )

    # value cache
    parser.add_argument("-vc", "--v_cache_type", type=str, default="mlp")
    parser.add_argument("--num_layers_per_mlp", type=int, default=2)
    parser.add_argument("--hidden_factors_per_mlp", type=int, default=1)
    parser.add_argument("--num_heads_per_mlp", type=int, default=8)
    parser.add_argument("--per_sequence", action="store_true")
    parser.add_argument("--target_perc", type=int, default=85)
    parser.add_argument("--target_cr", type=float, default=None)
    parser.add_argument("--v_lr", type=float, default=1e-3)
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "adamw", "sgd"],
    )
    parser.add_argument("--loss_func", type=str, default="mse")
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--meta_weights_path", type=str, default=None)
    parser.add_argument(
        "--value_mlp_weights_path",
        type=str,
        default=None,
        help="Path to plain pretrained value-cache MLP weights from train_value_mlps.py.",
    )
    parser.add_argument(
        "--global_compression",
        action="store_true",
        help="Pool errors across all layers and apply a single global threshold instead of per-layer thresholds.",
    )
    parser.add_argument(
        "--un_rope",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Undo RoPE on keys before MLP training and inference. Use --no-un_rope to disable.",
    )
    parser.add_argument(
        "--use_residual",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add a linear residual W_linear to the MLP, initialised as pinv(W_k) @ W_v from the model's projection weights. Use --no-use_residual to disable.",
    )
    parser.add_argument(
        "--intermediate_activation",
        type=str,
        default="relu",
        help="The activation function for the MLP in the value cache.",
    )
    parser.add_argument(
        "--linear_only",
        action="store_true",
        help="Recover values directly from W_linear_init (keys @ pinv(W_k) @ W_v), skipping MLP training entirely. Requires --un_rope.",
    )
    parser.add_argument(
        "--per_head_kv_linear",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute pinv(W_k) @ W_v independently per KV head instead of jointly. Use --no-per_head_kv_linear to disable.",
    )
    parser.add_argument(
        "--freeze_W_linear",
        action="store_true",
        default=False,
        help="Freeze W_linear during MLP training.",
    )
    parser.add_argument(
        "--use_attn_predictor",
        action="store_true",
        help="Use a shared CNN attention predictor to guide value residual selection.",
    )
    parser.add_argument(
        "--attn_predictor_path",
        type=str,
        default=None,
        help="Path to a checkpoint from train_attention_predictor.py.",
    )
    parser.add_argument(
        "--eviction_keep_ratio",
        type=float,
        default=1.0,
        help="Fraction of prefill KV tokens to retain per layer. Values below 1 enable attention-predictor physical eviction.",
    )
    parser.add_argument(
        "--v_turboquant_residuals",
        action="store_true",
        help="Quantise stored MLP value residuals with TurboQuant.",
    )
    parser.add_argument(
        "--v_compressor_bits",
        type=int,
        default=2,
        help="Bits per rotated residual coordinate for TurboQuant residual coding.",
    )
    args = parser.parse_args()
    args = apply_attn_predictor_config(args)
    if args.eviction_keep_ratio < 1 and not args.use_attn_predictor:
        raise ValueError("--eviction_keep_ratio < 1 requires --use_attn_predictor.")

    print("Config for lm-eval: ", vars(args))

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
