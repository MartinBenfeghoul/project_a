import os
import json
import argparse
import time
from omegaconf import OmegaConf
import torch

from lm_eval import evaluator
from lm_eval.utils import make_table
from lm_eval.tasks import TaskManager

from cache import CompressedCacheHFLM
from model.attention_predictor import (
    get_attn_predictor_hook_handles,
    apply_attn_predictor_config,
)
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
)

GEN_KWARGS = {
    "do_sample": False,
    "use_cache": True,
    "logits_to_keep": 0,
}


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

    rope_theta = getattr(model.config, "rope_theta", args.rope_theta)

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
        "kmeans_algorithm": args.kmeans_algorithm,
        "ksubspaces_rank": args.ksubspaces_rank,
        "kmeans_avg_heads": args.kmeans_avg_heads,
        "kmeans_per_head": args.kmeans_per_head,
        "layer_group_size": args.xkv_layer_group_size,
        "num_layers": num_layers,
        "unrope_keys": args.un_rope,
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
        "un_rope": args.un_rope,
        "rope_theta": rope_theta,
        "global_compression": args.global_compression,
        "normalise_keys": args.normalise_keys,
        "use_residual": args.use_residual,
        "intermediate_activation": args.intermediate_activation,
        "linear_only": args.linear_only,
        "early_stopping_tol": args.early_stopping_tol,
        "freeze_W_linear": args.freeze_W_linear,
        "zero_init_mlp_last_layer": args.zero_init_mlp_last_layer,
        "prev_layer_init": args.prev_layer_init,
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
        logger=logger,
        adjust_key_value_comp_ratio=args.adjust_key_value_comp_ratio,
        pretrained=model,
        tokenizer=tokenizer,
        truncation=False,
        trust_remote_code=True,
    )
    metadata = {"tokenizer": args.model_name}
    if args.max_seq_lengths is not None:
        metadata["max_seq_lengths"] = args.max_seq_lengths
    tm = TaskManager(metadata=metadata)

    if args.log_efficiency_metrics:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        start_time = time.perf_counter()

    results = evaluator.simple_evaluate(
        model=lm,
        gen_kwargs=GEN_KWARGS,
        tasks=args.tasks,
        num_fewshot=0,
        batch_size=1,
        max_batch_size=1,
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
    for handle in metric_hook_handles + attn_predictor_hook_handles:
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
    parser.add_argument(
        "--max_seq_lengths",
        type=int,
        nargs="+",
        default=None,
        help="Sequence lengths for RULER tasks.",
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
    parser.add_argument("--decomp_n_iter", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=3.0)
    parser.add_argument("--local_window", type=int, default=0)
    parser.add_argument(
        "--xkv_layer_group_size",
        type=int,
        default=2,
        help="Number of adjacent layers to jointly compress when --k_cache_type=xkv.",
    )
    parser.add_argument(
        "--kmeans_cluster_size",
        type=float,
        default=None,
    )
    parser.add_argument("--kmeans_n_iter", type=int, default=3)
    parser.add_argument(
        "--kmeans_algorithm",
        type=str,
        default="kmeans",
        choices=["kmeans", "ksubspaces"],
    )
    parser.add_argument("--ksubspaces_rank", type=int, default=1)
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
        "--override_target_perc",
        action="store_true",
        help="Use --target_perc instead of the per-layer values stored in the meta-weights checkpoint.",
    )
    parser.add_argument(
        "--override_num_epochs",
        action="store_true",
        help="Use --num_epochs instead of the inner_steps value stored in the meta-weights checkpoint.",
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
        "--rope_theta",
        type=float,
        default=None,  # TODO: set default based on model config if not passed
        help="RoPE theta used to recompute cos/sin if not passed by the model (fallback only).",
    )
    parser.add_argument(
        "--normalise_keys",
        action="store_true",
        help="Normalise keys (z-score over token dim, per head) before passing to the MLP.",
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
        "--zero_init_mlp_last_layer",
        action="store_true",
        help="Initialise the final value-cache MLP layer to zero so the residual branch starts from W_linear only.",
    )
    parser.add_argument(
        "--prev_layer_init",
        action="store_true",
        help="Initialise each value-cache MLP from the previous layer's trained MLP.",
    )
    parser.add_argument(
        "--early_stopping_tol",
        type=float,
        default=None,
        help="Stop MLP training early when relative loss improvement falls below this threshold.",
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
        "--v_turboquant_residuals",
        action="store_true",
        help="Quantise stored MLP value residuals with TurboQuant.",
    )
    parser.add_argument(
        "--v_compressor_bits",
        type=int,
        default=3,
        help="Bits per rotated residual coordinate for TurboQuant residual coding.",
    )

    args = parser.parse_args()
    if args.meta_weights_path is not None:
        args = override_args_from_meta_weights(args)
    args = apply_attn_predictor_config(args)

    print("Config for lm-eval: ", vars(args))

    return args


def override_args_from_meta_weights(args):
    """Infer MLP architecture and inner-loop config from a meta-weights checkpoint."""
    # TODO: this super ugly. In the future, save MLP config in meta learning config file
    ckpt = torch.load(args.meta_weights_path, map_location="cpu")

    config_path = os.path.join(
        os.path.dirname(args.meta_weights_path), "config.yaml"
    )
    if os.path.exists(config_path):
        train_cfg = OmegaConf.load(config_path).training
        if args.override_num_epochs:
            print(
                f"[meta_weights] Ignoring num_epochs from checkpoint ({train_cfg.inner_steps}); using --num_epochs={args.num_epochs}."
            )
        else:
            args.num_epochs = train_cfg.inner_steps
        args.loss_func = train_cfg.loss_func
        args.un_rope = train_cfg.un_rope
        args.optimizer = getattr(train_cfg, "inner_optimizer", "sgd")

    layer0 = next(v for k, v in ckpt.items() if k.startswith("layer_"))
    weight_keys = sorted(k for k in layer0 if k.startswith("weights."))
    w0, wlast = layer0[weight_keys[0]], layer0[weight_keys[-1]]

    n_layers = len(weight_keys)
    n_heads = (
        w0.shape[1] if w0.dim() == 4 else w0.shape[0]
    )  # in case batch dim not present
    head_dim = wlast.shape[-1]
    hidden_factor = (w0.shape[-1] // head_dim) if n_layers > 1 else 1

    for attr, ckpt_val in [
        ("num_layers_per_mlp", n_layers),
        ("hidden_factors_per_mlp", hidden_factor),
        ("num_heads_per_mlp", n_heads),
    ]:
        cli_val = getattr(args, attr)
        if cli_val != ckpt_val:
            print(
                f"Warning: overriding --{attr} {cli_val} → {ckpt_val} to match checkpoint."
            )
        setattr(args, attr, ckpt_val)

    if "target_perc_params" in ckpt:
        raw_percs = [t.item() for t in ckpt["target_perc_params"]]
        # Old checkpoints stored target_perc_params in logit-space and have no
        # target_perc_format key. TODO: Remove once stopped using old ckpts
        if ckpt.get("target_perc_format") != "direct":
            print(
                "[meta_weights] WARNING: checkpoint has no target_perc_format key — "
                "assuming old logit-space format. Applying sigmoid to convert to percentages."
            )
            raw_percs = [
                torch.sigmoid(t).item() for t in ckpt["target_perc_params"]
            ]
        meta_percs = [v * 100 for v in raw_percs]
        if args.override_target_perc:
            print(
                f"[meta_weights] Ignoring per-layer target_perc from checkpoint "
                f"(mean={sum(meta_percs)/len(meta_percs):.1f}%); using --target_perc={args.target_perc}."
            )
        else:
            args.target_perc = meta_percs
            print(
                f"[meta_weights] Loaded per-layer target_perc: "
                f"mean={sum(meta_percs)/len(meta_percs):.1f}%, "
                f"min={min(meta_percs):.1f}%, max={max(meta_percs):.1f}%"
            )

    print(
        f"[meta_weights] Inferred: num_layers={n_layers}, hidden_factor={hidden_factor}, "
        f"num_heads={n_heads}, "
        f"num_epochs={args.num_epochs}, loss_func={args.loss_func}, optimizer overridden to sgd"
    )
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
