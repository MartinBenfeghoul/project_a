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
from utils import (
    Logger,
    get_model_and_tokenizer,
    get_device,
    get_device_type,
    list_of_strings,
    get_output_path,
    extract_and_save_timing_stats,
    extract_and_save_efficiency_stats,
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


def make_hooks(logger, uncompressed_window=0, measure_latency=False):
    """
    When measure_latency=True, cuda events are recorded to time each prefill and decode pass.
    """
    _start_evt = [None]

    _cuda = measure_latency and torch.cuda.is_available()

    def has_complete_key_cache_timings(timing_stats):
        return (
            timing_stats is not None
            and timing_stats.get("decompose_calls", 0) > 0
            and timing_stats.get("reconstruct_calls", 0) > 0
        )

    def pre_hook(module, args, kwargs):
        if _cuda:
            evt = torch.cuda.Event(enable_timing=True)
            evt.record()
            _start_evt[0] = evt

    def post_hook(module, args, kwargs, output):
        if _cuda:
            end_evt = torch.cuda.Event(enable_timing=True)
            end_evt.record()

        input_ids = kwargs.get("input_ids", args[0] if args else None)
        if input_ids is None:
            return
        seq_len = input_ids.size(-1)
        pkv = output.past_key_values
        if seq_len > 1:
            if hasattr(pkv, "update_events"):
                if uncompressed_window > 0:
                    label_ed = -uncompressed_window
                else:
                    label_ed = None
                logits = output.logits
                pkv.update_events(
                    logits[..., : -1 - uncompressed_window, :],
                    input_ids[..., 1:label_ed],
                )
            if _cuda:
                logger.prefill_events.append((_start_evt[0], end_evt))
        else:
            if hasattr(pkv, "comp_ratio") and not logger.recorded_cr:
                cr = pkv.comp_ratio
                if cr is not None:
                    print(f"Compression ratio: {cr:.2f}")
                    logger.add_log("crs", cr)
                    logger.recorded_cr = True
            if hasattr(pkv, "timing_stats") and not logger.recorded_k_timing:
                timing_stats = pkv.timing_stats
                if has_complete_key_cache_timings(timing_stats):
                    logger.add_log(
                        "key_cache_decompose_time",
                        timing_stats["decompose_time"],
                    )
                    logger.add_log(
                        "key_cache_reconstruct_time",
                        timing_stats["reconstruct_time"],
                    )
                    logger.add_log(
                        "key_cache_decompose_relative",
                        timing_stats["decompose_relative"],
                    )
                    logger.add_log(
                        "key_cache_reconstruct_relative",
                        timing_stats["reconstruct_relative"],
                    )
                    logger.recorded_k_timing = True
            if _cuda:
                logger.decode_events.append((_start_evt[0], end_evt))

    return pre_hook, post_hook


@torch.no_grad()
def main(args):
    device_type = get_device_type()
    device = torch.device(device_type)

    model, tokenizer = get_model_and_tokenizer(args.model_name, device)
    logger = Logger()
    logger.prefill_events = []
    logger.decode_events = []
    logger.recorded_k_timing = False

    key_cache_kwargs = {
        "cache_type": args.k_cache_type,
        "decomposition_method": args.decomposition_method,
        "log_timing_stats": args.log_key_cache_timing,
        "comp_ratio": args.comp_ratio,
        "energy_threshold": args.energy_threshold,
        "rank_selection": args.rank_selection,
        "lr": args.k_lr,
        "n_iter": args.n_iter,
        "gamma": args.gamma,
        "min_size": 8.0,
    }

    num_layers = model.config.num_hidden_layers
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
        "target_model_num_heads": args.target_model_num_heads,
        "lr": args.v_lr,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "optimizer": args.optimizer,
        "loss_func": args.loss_func,
        "num_epochs": args.num_epochs,
        "meta_weights_path": args.meta_weights_path,
        "un_rope": args.un_rope,
        "rope_theta": args.rope_theta,
        "global_compression": args.global_compression,
    }

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

    pre_hook, post_hook = make_hooks(
        logger, measure_latency=args.log_efficiency_metrics
    )
    model.register_forward_pre_hook(pre_hook, with_kwargs=True)
    model.register_forward_hook(post_hook, with_kwargs=True)

    lm = CompressedCacheHFLM(
        key_cache_kwargs=key_cache_kwargs,
        value_cache_kwargs=value_cache_kwargs,
        logger=logger,
        pretrained=model,
        tokenizer=tokenizer,
        truncation=False,
        trust_remote_code=True,
    )
    args.tasks = get_tasks(args.tasks)
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

    print(make_table(results))

    results = extract_and_save_timing_stats(logger, results)
    if args.log_efficiency_metrics:
        results = extract_and_save_efficiency_stats(
            logger, results, model_baseline_mem, start_time
        )

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
    parser.add_argument("--log_efficiency_metrics", action="store_true")
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
    parser.add_argument("-e", "--energy_threshold", type=float, default=0.95)
    parser.add_argument(
        "--rank_selection",
        type=str,
        default="comp_ratio",
        choices=["comp_ratio", "energy"],
    )
    parser.add_argument("--k_lr", type=float, default=1e-2)
    parser.add_argument("--n_iter", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=3.0)
    parser.add_argument("--log_key_cache_timing", action="store_true")

    # value cache
    parser.add_argument("-vc", "--v_cache_type", type=str, default="mlp")
    parser.add_argument("--num_layers_per_mlp", type=int, default=4)
    parser.add_argument("--hidden_factors_per_mlp", type=int, default=2)
    parser.add_argument("--num_heads_per_mlp", type=int, default=8)
    parser.add_argument("--per_sequence", action="store_true")
    parser.add_argument(
        "--target_perc", type=int, default=85
    )  # TODO: I think this would be better as a dictionary with number of layers for each target perc
    parser.add_argument("--target_model_num_heads", type=int, default=8)
    parser.add_argument("--v_lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", type=str, default="adam")
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
        action="store_true",
        help="Undo RoPE on keys before MLP training and inference.",
    )
    parser.add_argument(
        "--rope_theta",
        type=float,
        default=500_000.0,
        help="RoPE theta used to recompute cos/sin if not passed by the model (fallback only).",
    )

    args = parser.parse_args()
    if args.meta_weights_path is not None:
        args = override_args_from_meta_weights(args)

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
