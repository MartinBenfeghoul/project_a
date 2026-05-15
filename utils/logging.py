import time

import numpy as np
import json
import os
import torch

from dotenv import load_dotenv
from omegaconf import OmegaConf
import wandb

load_dotenv()


class Logger:
    def __init__(self, layer_idx=None):
        self.layer_idx = layer_idx
        self.log_dict = {}
        self.value_mlp_log_events = []
        self.request_idx = -1

    def add_log(self, key, value):
        if key not in self.log_dict:
            self.log_dict[key] = []
        self.log_dict[key].append(value)

    def add_value_mlp_log_events(self, logging_events):
        self.value_mlp_log_events.extend(logging_events)

    def add_dict(self, input_dict):
        for key, value in input_dict.items():
            self.add_log(key, value)

    def get_log_list(self, key):
        return self.log_dict.get(key, [])

    def get_log_mean(self, key, std=False, return_dtype=np.float64):
        values = self.get_log_list(key)
        if values:
            if isinstance(values[0], (int, float)):
                res = np.mean(values, dtype=return_dtype)
                if std:
                    res = (res, np.std(values, dtype=return_dtype))
            elif isinstance(values[0], np.ndarray):
                res = np.mean(np.stack(values), axis=0, dtype=return_dtype)
                if std:
                    res = (
                        res,
                        np.std(np.stack(values), axis=0, dtype=return_dtype),
                    )
            else:
                raise ValueError(f"Unexpected value type: {type(values[0])}")
            return res
        return None

    def get_dict_mean(self, std=False):
        mean_dict = {}
        for key in self.log_dict.keys():
            mean_dict[key] = self.get_log_mean(key, std=std)
        return mean_dict

    def keys(self):
        return self.log_dict.keys()

    def values(self):
        return self.log_dict.values()

    @property
    def length(self):
        if self.log_dict:
            first_key = next(iter(self.log_dict))
            return len(self.log_dict[first_key])
        return 0

    def clear(self):
        self.log_dict = {}
        self.value_mlp_log_events = []
        self.request_idx = -1


def generate_run_name(config):
    """Generate a run name based on config parameters."""
    t = config.training

    if t.get("run_type") == "attention_predictor":
        return (
            f"seq{t.seq_len}_"
            f"maxb{t.max_batches}_"
            f"sps{t.samples_per_sequence}_"
            f"hist{t.history_step}_"
            f"blk{t.block_size}_"
            f"topk{t.topk_blocks}_"
            f"layers{_format_run_value(t.layers)}_"
            f"bce{t.bce_weight}"
        )

    return (
        f"seq{t.seq_len}_"
        f"steps{t.inner_steps}_"
        f"mlr{t.meta_lr}_"
        f"ilr{t.inner_lr}_"
        f"learnlr{t.get('learn_inner_lr', False)}_"
        f"learnperc{t.get('learn_target_perc', False)}_"
        f"gradaccum{t.grad_accum_steps}"
    )


def init_wandb(config):
    wandb_config = config.get("wandb", {})
    if not wandb_config.get("enabled", False):
        return False

    wandb_key = os.getenv("WANDB_KEY")
    if not wandb_key:
        raise RuntimeError(
            "WANDB_KEY environment variable is not set. Please add it to your .env file"
        )

    wandb.login(key=wandb_key)
    run_name = wandb_config.get("run_name") or generate_run_name(config)
    wandb.init(
        project=wandb_config.get("project", "gist_vs_details"),
        entity=wandb_config.get("entity", "mixture_of_titans"),
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
    seq_len=None,
):
    """Save experiment results to a JSONL file."""
    with open(file_name, "a") as f:
        f.write(
            json.dumps(
                {
                    "seq_len": seq_len,
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


def save_attention_predictor_checkpoint(
    args,
    predictor,
    optimizer: torch.optim.Optimizer,
    layers: list[int],
    step: int,
    running: dict[str, float],
    num_updates: int,
) -> None:
    metrics = {
        key: value / max(1, num_updates) for key, value in running.items()
    }
    params = {
        "model_state_dict": predictor.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "metrics": metrics,
        "config": {
            "model_name": args.model_name,
            "seq_len": args.seq_len,
            "history_step": args.history_step,
            "block_size": args.block_size,
            "topk_blocks": args.topk_blocks,
            "layers": layers,
            "bce_weight": args.bce_weight,
        },
    }

    torch.save(
        params,
        os.path.join(args.checkpoint_dir, "model_ckpt.pt"),
    )

    metrics_path = os.path.join(args.checkpoint_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def log_batch(
    epoch,
    batch_idx,
    q,
    inner_metrics,
    per_layer_losses,
    inner_lr_params,
    batch_time_ms,
    global_step,
    use_wandb,
    target_perc_params=None,
):
    initial_support_loss = (
        inner_metrics["inner_losses"][0]
        if inner_metrics["inner_losses"]
        else 0.0
    )
    final_support_loss = inner_metrics["final_support_loss"] or 0.0
    generalisation_gap = q - final_support_loss

    perc_str = ""
    if target_perc_params is not None:
        mean_perc = sum(p.item() * 100 for p in target_perc_params) / len(
            target_perc_params
        )
        perc_str = f", Target Perc: {mean_perc:.1f}%"

    print(
        f"Epoch {epoch}, Batch {batch_idx}, "
        f"Query Loss: {q:.6f}, "
        f"Gen Gap: {generalisation_gap:.6f}, "
        f"Time: {batch_time_ms:.1f}ms"
        f"{perc_str}"
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
            "inner/adaptation_improvement": initial_support_loss
            - final_support_loss,
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

        if target_perc_params is not None:
            perc_values = [p.item() * 100 for p in target_perc_params]
            log_dict["meta/target_perc_mean"] = sum(perc_values) / len(
                perc_values
            )
            log_dict["meta/target_perc_min"] = min(perc_values)
            log_dict["meta/target_perc_max"] = max(perc_values)
            for layer_idx, pv in enumerate(perc_values):
                log_dict[f"meta/target_perc_layer_{layer_idx}"] = pv

        wandb.log(log_dict, step=global_step)


def save_checkpoint(
    layer_mlps, inner_lr_params, checkpoint_path, epoch, target_perc_params=None
):
    base, ext = os.path.splitext(checkpoint_path)
    epoch_checkpoint_path = f"{base}_epoch{epoch}{ext}"
    epoch_params = {
        f"layer_{i}": mlp.state_dict() for i, mlp in enumerate(layer_mlps)
    }
    if inner_lr_params is not None:
        epoch_params["inner_lr_params"] = [
            lr.detach().cpu() for lr in inner_lr_params
        ]
    if target_perc_params is not None:
        epoch_params["target_perc_params"] = [
            p.detach().cpu() for p in target_perc_params
        ]
        epoch_params["target_perc_format"] = (
            "direct"  # values in percentage-space instead of logit-space
        )
    torch.save(epoch_params, epoch_checkpoint_path)
    print(f"Checkpoint saved to {epoch_checkpoint_path}")


def make_hooks(
    logger,
    uncompressed_window=0,
    measure_latency=False,
    measure_gpu_memory=False,
    model_baseline_mem=0,
    use_wandb=False,
    dump_full_kv_dir=None,
    model_name=None,
    task_name=None,
    rope_theta=None,
):
    """
    When measure_latency=True, cuda events are recorded to time each prefill and decode pass.
    """
    _start_evt = [None]
    _prefill_mem_start = [None]

    _cuda = measure_latency and torch.cuda.is_available()
    _cuda_mem = measure_gpu_memory and torch.cuda.is_available()

    def has_complete_key_cache_timings(timing_stats):
        return (
            timing_stats is not None
            and timing_stats.get("decompose_calls", 0) > 0
            and timing_stats.get("reconstruct_calls", 0) > 0
        )

    def dump_full_kv(input_ids, pkv):
        if dump_full_kv_dir is None:
            return

        key_layers = getattr(getattr(pkv, "key_cache", None), "layers", None)
        value_layers = getattr(
            getattr(pkv, "value_cache", None), "layers", None
        )
        if key_layers is None or value_layers is None:
            raise ValueError(
                "KV dump expects baseline key/value cache layers on past_key_values."
            )

        keys = []
        values = []
        for layer_idx, (key_layer, value_layer) in enumerate(
            zip(key_layers, value_layers)
        ):
            if key_layer.tensor is None or value_layer.tensor is None:
                raise ValueError(
                    f"KV dump encountered an uninitialised cache tensor at layer {layer_idx}."
                )
            keys.append(key_layer.tensor.detach().cpu().clone())
            values.append(value_layer.tensor.detach().cpu().clone())

        output_path = os.path.join(
            dump_full_kv_dir,
            f"{model_name.replace('/', '_')}/{task_name}_raw_kv.pt",
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(
            {
                "task_name": task_name,
                "input_ids": input_ids.detach().cpu().clone(),
                "prompt_len": int(input_ids.size(-1)),
                "model_name": model_name,
                "rope_theta": rope_theta,
                "keys": torch.stack(keys, dim=0),
                "values": torch.stack(values, dim=0),
            },
            output_path,
        )
        print(f"Saved raw KV dump to {output_path}. Exiting script...")
        import sys

        sys.exit(0)

    def pre_hook(module, args, kwargs):
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        is_prefill = input_ids is not None and input_ids.size(-1) > 1
        if _cuda:
            evt = torch.cuda.Event(enable_timing=True)
            evt.record()
            _start_evt[0] = evt
        if _cuda_mem and is_prefill:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            _prefill_mem_start[0] = torch.cuda.memory_allocated()

    def post_hook(module, args, kwargs, output):
        if _cuda:
            end_evt = torch.cuda.Event(enable_timing=True)
            end_evt.record()

        input_ids = kwargs.get("input_ids", args[0] if args else None)
        if input_ids is None:
            return
        seq_len = input_ids.size(-1)
        pkv = output.past_key_values
        logging_event = None
        if seq_len > 1:
            logger.request_idx += 1
            logging_event_idx = logger.request_idx + 1
            if use_wandb:
                logging_event = {}

            if hasattr(pkv, "key_recon_mse"):
                key_mse = pkv.key_recon_mse
                if key_mse is not None:
                    print(f"Key reconstruction MSE: {key_mse:.6f}")
                    logger.add_log("key_recon_mse", key_mse)
            if logging_event is not None:
                eta_mean = getattr(pkv, "eta_mean", None)
                if eta_mean is not None:
                    logging_event["live/logging_event_idx"] = logging_event_idx
                    logging_event["live/key_cache_eta_mean"] = eta_mean
                    logging_event["live/seq_len"] = int(seq_len)
                    eta_std = getattr(pkv, "eta_std", None)
                    if eta_std is not None:
                        logging_event["live/key_cache_eta_std"] = eta_std
            if hasattr(pkv, "value_recon_mse"):
                val_mse = pkv.value_recon_mse
                if val_mse is not None:
                    print(f"Value reconstruction MSE: {val_mse:.6f}")
                    logger.add_log("value_recon_mse", val_mse)
            if hasattr(pkv, "update_events"):
                if uncompressed_window > 0:
                    label_ed = -uncompressed_window
                else:
                    label_ed = None
                logits = output.logits
                event_stats = pkv.update_events(
                    logits[..., : -1 - uncompressed_window, :],
                    input_ids[..., 1:label_ed],
                )
                if event_stats is not None:
                    for key, value in event_stats.items():
                        logger.add_log(key, value)
            if _cuda:
                logger.prefill_events.append((_start_evt[0], end_evt))
            if _cuda_mem:
                torch.cuda.synchronize()
                mem_allocated = torch.cuda.memory_allocated()
                mem_start = _prefill_mem_start[0]
                if mem_start is not None:
                    logger.add_log(
                        "prefill_gpu_mem_delta_bytes", mem_allocated - mem_start
                    )
                logger.add_log("prefill_gpu_mem_allocated_bytes", mem_allocated)
                logger.add_log(
                    "prefill_gpu_mem_overhead_bytes",
                    mem_allocated - model_baseline_mem,
                )
                _prefill_mem_start[0] = None
            dump_full_kv(input_ids, pkv)
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

        if logging_event is not None:
            value_mlp_events = getattr(pkv, "value_mlp_log_events", [])
            if value_mlp_events:
                for value_mlp_event in value_mlp_events:
                    value_mlp_event["request_idx"] = logger.request_idx
                logger.add_value_mlp_log_events(value_mlp_events)
            if logging_event:
                wandb.log(logging_event)
            if logging_event_idx % 10 == 0:
                log_live_value_mlp_training_wandb(
                    logger.value_mlp_log_events,
                    logging_event_idx,
                )

    return pre_hook, post_hook


def extract_and_save_stats(logger, results):
    decompose_time = logger.get_log_mean("key_cache_decompose_time", std=True)
    reconstruct_time = logger.get_log_mean(
        "key_cache_reconstruct_time", std=True
    )
    decompose_relative = logger.get_log_mean(
        "key_cache_decompose_relative", std=True
    )
    reconstruct_relative = logger.get_log_mean(
        "key_cache_reconstruct_relative", std=True
    )
    if decompose_time is not None and reconstruct_time is not None:
        slowest_op = (
            "decompose"
            if decompose_time[0] >= reconstruct_time[0]
            else "reconstruct"
        )
        timing_str = (
            "Key cache timings:"
            f" decompose={decompose_time[0]:.4f}+-{decompose_time[1]:.4f}s"
            f" ({decompose_relative[0]:.2%}+-{decompose_relative[1]:.2%}),"
            f" reconstruct={reconstruct_time[0]:.4f}+-{reconstruct_time[1]:.4f}s"
            f" ({reconstruct_relative[0]:.2%}+-{reconstruct_relative[1]:.2%}),"
            f" slowest={slowest_op}"
        )
        print(timing_str)
        results["results"]["key_cache_timings"] = timing_str

    event_stat_keys = [
        "segment_size_min",
        "segment_size_median",
        "segment_size_max",
        "segment_size_std_mean",
    ]
    event_stat_values = {
        key: logger.get_log_mean(key, std=True) for key in event_stat_keys
    }
    if all(value is not None for value in event_stat_values.values()):
        results["results"]["event_stats"] = {}
        for key, value in event_stat_values.items():
            results["results"]["event_stats"][key] = float(value[0])
            results["results"]["event_stats"][f"{key}_std"] = float(
                value[1]
            )
        print(
            "Logged segment sizes:"
            f" min={event_stat_values['segment_size_min'][0]:.2f}"
            f"+/-{event_stat_values['segment_size_min'][1]:.2f},"
            f" median={event_stat_values['segment_size_median'][0]:.2f}"
            f"+/-{event_stat_values['segment_size_median'][1]:.2f},"
            f" max={event_stat_values['segment_size_max'][0]:.2f}"
            f"+/-{event_stat_values['segment_size_max'][1]:.2f},"
            " avg per-sequence std="
            f"{event_stat_values['segment_size_std_mean'][0]:.2f}"
            f"+/-{event_stat_values['segment_size_std_mean'][1]:.2f} tokens"
        )

    return results


def extract_and_save_efficiency_stats(
    logger, results, model_baseline_mem, start_time
):
    torch.cuda.synchronize()
    wall_time_sec = time.perf_counter() - start_time
    peak_mem = torch.cuda.max_memory_allocated()
    prefill_ms = [s.elapsed_time(e) for s, e in logger.prefill_events]
    decode_ms = [s.elapsed_time(e) for s, e in logger.decode_events]
    prefill_mem_allocated = logger.get_log_list(
        "prefill_gpu_mem_allocated_bytes"
    )
    prefill_mem_overhead = logger.get_log_list("prefill_gpu_mem_overhead_bytes")

    efficiency_metrics = {
        "eval_wall_time_seconds": wall_time_sec,
        "eval_wall_time_minutes": wall_time_sec / 60.0,
        "gpu_peak_mem_gib": peak_mem
        / (1024**3),  # (weights + kv cache + activations)
        "gpu_kv_cache_overhead_gib": (peak_mem - model_baseline_mem)
        / (1024**3),  # (kv cache + activations)
        "prefill_gpu_mem_allocated_gib_mean": (
            sum(prefill_mem_allocated) / len(prefill_mem_allocated) / (1024**3)
            if prefill_mem_allocated
            else 0.0
        ),
        "prefill_gpu_mem_overhead_gib_mean": (
            sum(prefill_mem_overhead) / len(prefill_mem_overhead) / (1024**3)
            if prefill_mem_overhead
            else 0.0
        ),
        "prefill_latency_ms_mean": (
            sum(prefill_ms) / len(prefill_ms) if prefill_ms else 0.0
        ),
        "decode_latency_ms_mean": (
            sum(decode_ms) / len(decode_ms) if decode_ms else 0.0
        ),  # per-token
        "decode_tokens_per_sec": (
            1000.0 / (sum(decode_ms) / len(decode_ms)) if decode_ms else 0.0
        ),
        "n_prefill_passes": len(prefill_ms),
        "n_decode_steps": len(decode_ms),
    }
    print("Efficiency metrics:", efficiency_metrics)
    results["results"]["efficiency_metrics"] = efficiency_metrics
    return results


def get_output_path(output_path):
    i = 0
    while True:
        if not os.path.exists(output_path.format(i)):
            return output_path.format(i)
        i += 1


def init_lm_eval_wandb(args):
    if not getattr(args, "use_wandb", False):
        return False

    wandb_key = os.getenv("WANDB_KEY")
    if wandb_key:
        wandb.login(key=wandb_key)

    wandb.init(
        project="gist_vs_details",
        entity="mixture_of_titans",
        config=vars(args),
    )
    wandb.define_metric("live/logging_event_idx")
    wandb.define_metric("live/*", step_metric="live/logging_event_idx")
    wandb.define_metric("_value_mlp_epoch")
    return True


def sequence_length_bucket(seq_len):
    if seq_len is None:
        return "unknown"
    bounds = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
    prev = 0
    for bound in bounds:
        if seq_len <= bound:
            return f"{prev + 1:05d}-{bound:05d}"
        prev = bound
    return f"{bounds[-1] + 1:05d}+"


def log_live_value_mlp_training_wandb(
    logging_events, logging_event_idx, log_tables=False
):
    if not wandb.run or not logging_events:
        return

    table_payload = {"live/logging_event_idx": logging_event_idx}
    mse_payloads = []
    grouped = {}
    for logging_event in logging_events:
        key = (
            _wandb_key_part(logging_event.get("task_name", "unknown")),
            _wandb_key_part(
                sequence_length_bucket(logging_event.get("seq_len"))
            ),
        )
        grouped.setdefault(key, []).append(logging_event)

    for (task_name, seq_len_bucket), group_events in grouped.items():
        prefix = f"{task_name}_{seq_len_bucket}"
        if log_tables:
            table_rows = _value_mlp_event_table_rows(group_events)
            if table_rows:
                table_key = f"{prefix}/value_mlp_training_events"
                table_payload[table_key] = _value_mlp_event_table(table_rows)

        mse_key = f"{prefix}/value_recon_mse_mean"
        _define_value_mlp_mse_metric(mse_key)
        for epoch, mse_mean in _mean_mse_by_epoch_rows(group_events):
            mse_payloads.append(
                {
                    "live/logging_event_idx": logging_event_idx,
                    "_value_mlp_epoch": epoch,
                    mse_key: mse_mean,
                }
            )

        for layer_idx, rows in _mean_mse_by_layer_epoch_rows(
            group_events
        ).items():
            layer_part = _layer_key_part(layer_idx)

            mse_key = f"{prefix}/{layer_part}/value_recon_mse_mean"
            _define_value_mlp_mse_metric(mse_key)
            for epoch, mse_mean in rows:
                mse_payloads.append(
                    {
                        "live/logging_event_idx": logging_event_idx,
                        "_value_mlp_epoch": epoch,
                        mse_key: mse_mean,
                    }
                )

    if len(table_payload) > 1:
        wandb.log(table_payload)
    for payload in mse_payloads:
        wandb.log(payload)


def _define_value_mlp_mse_metric(metric_key):
    defined = getattr(_define_value_mlp_mse_metric, "_defined", set())
    if metric_key in defined:
        return
    wandb.define_metric(metric_key, step_metric="_value_mlp_epoch")
    defined.add(metric_key)
    _define_value_mlp_mse_metric._defined = defined


def _value_mlp_event_table(rows):
    return wandb.Table(
        data=rows,
        columns=[
            "request_idx",
            "layer_idx",
            "epoch",
            "value_recon_mse",
            "target_perc",
        ],
    )


def _value_mlp_event_table_rows(logging_events):
    rows = []
    for logging_event in logging_events:
        rows.append(
            [
                logging_event.get("request_idx"),
                logging_event.get("layer_idx"),
                logging_event.get("epoch"),
                float(logging_event.get("value_recon_mse")),
                float(logging_event.get("target_perc")
                )
            ]
        )
    return rows


def _mean_mse_by_epoch_rows(logging_events):
    by_epoch = {}
    for logging_event in logging_events:
        epoch = logging_event.get("epoch")
        mse = logging_event.get("value_recon_mse")
        by_epoch.setdefault(epoch, []).append(float(mse))
    return [
        [epoch, float(np.mean(mses))]
        for epoch, mses in sorted(by_epoch.items())
    ]


def _mean_mse_by_layer_epoch_rows(logging_events):
    by_layer_epoch = {}
    for logging_event in logging_events:
        layer_idx = logging_event.get("layer_idx", "unknown")
        epoch = logging_event.get("epoch")
        mse = logging_event.get("value_recon_mse")
        by_layer_epoch.setdefault(layer_idx, {}).setdefault(
            epoch, []
        ).append(float(mse))
    return {
        layer_idx: [
            [epoch, float(np.mean(mses))]
            for epoch, mses in sorted(by_epoch.items())
        ]
        for layer_idx, by_epoch in sorted(
            by_layer_epoch.items(), key=lambda item: (0, item[0])
        )
    }


def _wandb_key_part(value):
    return str(value).replace("/", "_").replace(" ", "_")


def _layer_key_part(layer_idx):
    if isinstance(layer_idx, int):
        return f"layer_{layer_idx:02d}"
    return f"layer_{_wandb_key_part(layer_idx)}"


def _format_run_value(value: object) -> str:
    return str(value).replace("/", "-").replace(",", "-").replace(" ", "")


def save_run_config(
    args,
    layers: list[int],
    run_name: str,
    checkpoint_dir: str,
) -> None:
    config = {
        **vars(args),
        "run_name": run_name,
        "checkpoint_dir": checkpoint_dir,
        "resolved_layers": layers,
    }
    OmegaConf.save(
        OmegaConf.create(config), os.path.join(checkpoint_dir, "config.yaml")
    )


def attention_predictor_config(args, layers: list[int] | None = None):
    training = {
        **vars(args),
        "run_type": "attention_predictor",
    }
    if layers is not None:
        training["resolved_layers"] = layers

    return OmegaConf.create(
        {
            "training": training,
            "wandb": {
                "enabled": args.use_wandb,
                "project": args.wandb_project,
                "entity": args.wandb_entity,
                "run_name": getattr(args, "wandb_run_name", None),
            },
        }
    )


def prepare_run_directory(
    args,
    layers: list[int],
) -> str:
    run_name = generate_run_name(attention_predictor_config(args, layers))
    args.checkpoint_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    save_run_config(args, layers, run_name, args.checkpoint_dir)
    return run_name


def init_timing_stats():
    return {
        "decompose_time": 0.0,
        "reconstruct_time": 0.0,
        "decompose_relative": 0.0,
        "reconstruct_relative": 0.0,
        "decompose_calls": 0,
        "reconstruct_calls": 0,
        "most_time_consuming": None,
    }


def sync_cuda(tensor):
    if tensor.is_cuda:
        torch.cuda.synchronize(device=tensor.device)


def update_timing_stats(cache_cls, operation, elapsed_time):
    cache_cls.timing_stats[f"{operation}_time"] += elapsed_time
    cache_cls.timing_stats[f"{operation}_calls"] += 1
    total_time = (
        cache_cls.timing_stats["decompose_time"]
        + cache_cls.timing_stats["reconstruct_time"]
    )
    if total_time > 0:
        cache_cls.timing_stats["decompose_relative"] = (
            cache_cls.timing_stats["decompose_time"] / total_time
        )
        cache_cls.timing_stats["reconstruct_relative"] = (
            cache_cls.timing_stats["reconstruct_time"] / total_time
        )
        cache_cls.timing_stats["most_time_consuming"] = max(
            ("decompose", "reconstruct"),
            key=lambda op: cache_cls.timing_stats[f"{op}_time"],
        )
