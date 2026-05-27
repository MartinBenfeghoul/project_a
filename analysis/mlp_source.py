import argparse
import csv
import json
import math
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from lm_eval import evaluator
from lm_eval.tasks import TaskManager
from lm_eval.utils import make_table
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.mlp_cache import AnalysisCompressedCache, MLPPredictionCache
from cache import CompressedCacheHFLM
from utils.args import list_of_strings
from utils.device import get_device, get_device_type
from utils.logging import extract_and_save_stats
from utils.model import get_model_and_tokenizer
from utils.results_processing.print_lm_eval import (
    LM_EVAL_TASKS,
    LONGBENCH_TASKS,
    RULER_TASKS,
)


KEY_SOURCES = [
    "prev_layer_keys",
    "prev_layer_values",
    "prev_layer_kv",
    "current_layer_values",
]
VALUE_SOURCES = [
    "prev_layer_keys",
    "prev_layer_values",
    "prev_layer_kv",
    "current_layer_keys",
]
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


class MetricLogger:
    def __init__(self):
        self.log_dict = {}
        self.request_idx = -1

    def add_log(self, key, value):
        self.log_dict.setdefault(key, []).append(value)

    def get_log_list(self, key):
        return self.log_dict.get(key, [])

    def get_log_mean(self, key, std=False):
        values = self.get_log_list(key)
        if not values:
            return None
        mean = sum(values) / len(values)
        if not std:
            return mean
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        return mean, math.sqrt(variance)


class AnalysisCompressedCacheHFLM(CompressedCacheHFLM):
    def __init__(self, key_cache_kwargs, value_cache_kwargs, logger, **kwargs):
        super().__init__(
            key_cache_kwargs=key_cache_kwargs,
            value_cache_kwargs=value_cache_kwargs,
            logger=logger,
            eviction_keep_ratio=1.0,
            adjust_key_value_comp_ratio=False,
            **kwargs,
        )

    def _make_cache(self, cache_context=None):
        return AnalysisCompressedCache(
            config=self.model.config,
            key_cache_kwargs=self._key_cache_kwargs,
            value_cache_kwargs=self._value_cache_kwargs,
            cache_context=cache_context,
            verbose=False,
        )


@contextmanager
def registered_prediction_cache():
    import cache.keys as key_module
    import cache.values as value_module

    old_key = key_module.KEY_CACHE_CLASSES.get("mlp_prediction")
    old_value = value_module.VALUE_CACHE_CLASSES.get("mlp_prediction")
    key_module.KEY_CACHE_CLASSES["mlp_prediction"] = MLPPredictionCache
    value_module.VALUE_CACHE_CLASSES["mlp_prediction"] = MLPPredictionCache
    try:
        yield
    finally:
        if old_key is None:
            key_module.KEY_CACHE_CLASSES.pop("mlp_prediction", None)
        else:
            key_module.KEY_CACHE_CLASSES["mlp_prediction"] = old_key
        if old_value is None:
            value_module.VALUE_CACHE_CLASSES.pop("mlp_prediction", None)
        else:
            value_module.VALUE_CACHE_CLASSES["mlp_prediction"] = old_value


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
    )
    parser.add_argument("-t", "--tasks", type=list_of_strings, default=["longbench"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/kv_mlp_prediction_eval"),
    )

    parser.add_argument(
        "--key_sources",
        nargs="+",
        default=KEY_SOURCES,
        choices=KEY_SOURCES,
    )
    parser.add_argument(
        "--value_sources",
        nargs="+",
        default=VALUE_SOURCES,
        choices=VALUE_SOURCES,
    )
    return parser


def condition_slug(condition: dict[str, str]) -> str:
    return f"{condition['target']}_from_{condition['source']}"


def build_conditions(args: argparse.Namespace) -> list[dict[str, str]]:
    conditions = []
    conditions.extend({"target": "keys", "source": source} for source in args.key_sources)
    conditions.extend({"target": "values", "source": source} for source in args.value_sources)
    return conditions


def flatten_metrics(results: dict[str, Any]) -> dict[str, float]:
    flat = {}
    for task_name, metrics in results.items():
        if not isinstance(metrics, dict):
            continue
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                flat[f"{task_name}/{metric_name}"] = float(value)
    return flat


def collect_summary(
    condition: dict[str, str],
    results: dict[str, Any] | None,
    *,
    tasks: list[str],
    elapsed_s: float,
    result_path: Path | None = None,
) -> dict[str, Any]:
    row = {
        **condition,
        "elapsed_s": elapsed_s,
        "result_path": str(result_path) if result_path is not None else "",
    }

    res = results.get("results", results)
    for name in ["compression_metrics", "value_recon_mse", "key_recon_mse"]:
        metrics = res.get(name)
        if isinstance(metrics, dict):
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    row[f"{name}/{key}"] = float(value)
    task_metrics = flatten_metrics(res)
    row.update(task_metrics)
    selected_scores = collect_benchmark_scores(res, tasks)
    row.update(selected_scores)
    scores = list(selected_scores.values())
    if scores:
        row["downstream_score_mean"] = sum(scores) / len(scores)
        row["n_downstream_scores"] = len(scores)
    return row


def collect_benchmark_scores(
    results: dict[str, Any],
    tasks: list[str],
) -> dict[str, float]:
    task_metrics = {**LM_EVAL_TASKS, **RULER_TASKS, **LONGBENCH_TASKS}
    scores = {}
    for task in tasks:
        metric = task_metrics.get(task)
        if metric is None:
            continue
        task_result = results.get(task)
        if not isinstance(task_result, dict):
            continue
        value = task_result.get(metric)
        if isinstance(value, (int, float)):
            scores[f"selected_score/{task}/{metric}"] = float(value)
    return scores


def write_rows_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def attach_metric_hook(model, logger: MetricLogger):
    recorded_cr = False

    def post_hook(module, args, kwargs, output):
        nonlocal recorded_cr
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        if input_ids is None:
            return

        pkv = output.past_key_values
        if input_ids.size(-1) > 1:
            logger.request_idx += 1
            for attr, label, display in [
                ("key_recon_mse", "key_recon_mse", "Key prediction MSE"),
                ("value_recon_mse", "value_recon_mse", "Value prediction MSE"),
            ]:
                value = getattr(pkv, attr, None)
                if value is not None:
                    print(f"{display}: {value:.6f}")
                    logger.add_log(label, value)
        elif not recorded_cr:
            cr = getattr(pkv, "comp_ratio", None)
            if cr is not None:
                print(f"Compression ratio: {cr:.2f}")
                logger.add_log("crs", cr)
                recorded_cr = True

    return model.register_forward_hook(post_hook, with_kwargs=True)


def make_cache_kwargs(
    *,
    condition: dict[str, str],
    num_layers: int,
    num_kv_heads: int,
    rope_theta: float | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    common = {
        "num_layers_per_mlp": [2] * num_layers,
        "hidden_factors_per_mlp": [1] * num_layers,
        "num_heads_per_mlp": [num_kv_heads] * num_layers,
        "rope_theta": rope_theta,
    }
    key_cache_kwargs = {"cache_type": "baseline"}
    value_cache_kwargs = {"cache_type": "baseline"}

    if condition["target"] == "keys":
        key_cache_kwargs = {
            **common,
            "cache_type": "mlp_prediction",
            "source": condition["source"],
            "target_is_key": True,
        }
    else:
        value_cache_kwargs = {
            **common,
            "cache_type": "mlp_prediction",
            "source": condition["source"],
            "target_is_key": False,
        }
    return key_cache_kwargs, value_cache_kwargs


def run_condition(
    args: argparse.Namespace,
    condition: dict[str, str],
    model,
    tokenizer,
    tasks: list[str],
    output_dir: Path,
) -> tuple[dict[str, Any], Path]:
    logger = MetricLogger()

    rope_theta = getattr(model.config, "rope_theta")
    num_layers = model.config.num_hidden_layers
    num_kv_heads = model.config.num_key_value_heads
    key_cache_kwargs, value_cache_kwargs = make_cache_kwargs(
        condition=condition,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        rope_theta=rope_theta,
    )

    hook_handle = attach_metric_hook(model, logger)
    try:
        lm = AnalysisCompressedCacheHFLM(
            key_cache_kwargs=key_cache_kwargs,
            value_cache_kwargs=value_cache_kwargs,
            logger=logger,
            pretrained=model,
            tokenizer=tokenizer,
            truncation=False,
            trust_remote_code=True,
        )
        results = evaluator.simple_evaluate(
            model=lm,
            gen_kwargs=GEN_KWARGS,
            tasks=tasks,
            num_fewshot=0,
            batch_size=1,
            max_batch_size=1,
            device=get_device(lm),
            limit=args.limit,
        )
        cr_values = logger.get_log_list("crs")
        if cr_values:
            cr_mean, cr_std = logger.get_log_mean("crs", std=True)
            results["results"]["compression_metrics"] = {
                "compression_ratio_mean": float(cr_mean),
                "compression_ratio_std": float(cr_std),
                "n_compression_ratio_samples": len(cr_values),
            }
        for label in ["value_recon_mse", "key_recon_mse"]:
            mse_values = logger.get_log_list(label)
            if mse_values:
                mse_mean, mse_std = logger.get_log_mean(label, std=True)
                results["results"][label] = {
                    "recon_mse_mean": float(mse_mean),
                    "recon_mse_std": float(mse_std),
                }
        print(make_table(results))
        results = extract_and_save_stats(logger, results)
        results["results"]["config"] = {
            **serialisable_config(vars(args)),
            **condition,
            "key_cache_kwargs": serialisable_config(key_cache_kwargs),
            "value_cache_kwargs": serialisable_config(value_cache_kwargs),
        }
    finally:
        hook_handle.remove()

    run_dir = output_dir / "raw" / condition_slug(condition)
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "results.json"
    with result_path.open("w") as f:
        json.dump(results["results"], f, ensure_ascii=False, indent=4)
    return results, result_path


def serialisable_config(config: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in config.items():
        if isinstance(value, torch.device):
            out[key] = str(value)
        elif isinstance(value, Path):
            out[key] = str(value)
        else:
            out[key] = value
    return out


def main(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks = get_tasks(args.tasks)
    device_type = get_device_type()
    device = torch.device(device_type)
    model, tokenizer = get_model_and_tokenizer(args.model_name, device)
    model.eval()

    rows = []
    with registered_prediction_cache():
        for condition in build_conditions(args):
            start = time.perf_counter()
            result_path = None
            print(f"Running {condition_slug(condition)}", flush=True)
            results, result_path = run_condition(
                args,
                condition,
                model,
                tokenizer,
                tasks,
                args.output_dir,
            )
            elapsed_s = time.perf_counter() - start
            rows.append(
                collect_summary(
                    condition,
                    results,
                    tasks=tasks,
                    elapsed_s=elapsed_s,
                    result_path=result_path,
                )
            )

            write_rows_jsonl(args.output_dir / "summary.jsonl", rows)
            write_rows_csv(args.output_dir / "summary.csv", rows)

    print(f"Saved summary to {args.output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
