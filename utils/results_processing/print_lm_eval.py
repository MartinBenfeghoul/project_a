import json
import argparse
from copy import deepcopy

LM_EVAL_TASKS = {
    "piqa": "acc,none",
    "arc_easy": "acc,none",
    "arc_challenge": "acc_norm,none",
    "hellaswag": "acc_norm,none",
    "winogrande": "acc,none",
    "mmlu": "acc,none",
}

RULER_TASKS = {
    "niah_single_1": "4096,none",
    "niah_single_2": "4096,none",
    "niah_single_3": "4096,none",
    "niah_multikey_1": "4096,none",
    "niah_multikey_2": "4096,none",
    "niah_multikey_3": "4096,none",
    "niah_multiquery": "4096,none",
    "niah_multivalue": "4096,none",
    "ruler_vt": "4096,none",
    "ruler_cwe": "4096,none",
    "ruler_fwe": "4096,none",
    "ruler_qa_hotpot": "4096,none",
    "ruler_qa_squad": "4096,none",
}

LONGBENCH_TASKS = {
    "longbench_2wikimqa": "qa_f1_score,none",
    "longbench_gov_report": "rouge_score,none",
    "longbench_hotpotqa": "qa_f1_score,none",
    "longbench_lcc": "code_sim_score,none",
    "longbench_multi_news": "rouge_score,none",
    "longbench_multifieldqa_en": "qa_f1_score,none",
    "longbench_musique": "qa_f1_score,none",
    "longbench_narrativeqa": "qa_f1_score,none",
    "longbench_passage_retrieval_en": "retrieval_score,none",
    "longbench_qasper": "qa_f1_score,none",
    "longbench_qmsum": "rouge_score,none",
    "longbench_repobench-p": "code_sim_score,none",
    "longbench_samsum": "rouge_score,none",
    "longbench_trec": "classification_score,none",
    "longbench_triviaqa": "qa_f1_score,none",
}

ALL_TASKS = {**LM_EVAL_TASKS, **RULER_TASKS, **LONGBENCH_TASKS}

KEY_CONFIGS = [
    "model_name",
    "tasks",
    "v_cache_type",
    "k_cache_type",
    "decomposition_method",
    "comp_ratio",
    "energy_threshold",
    "rank_selection",
    "decomp_n_iter",
    "gamma",
    "local_window",
    "xkv_layer_group_size",
    "kmeans_cluster_size",
    "kmeans_n_iter",
    "kmeans_init",
    "kmeans_dtype",
    "kmeans_avg_heads",
    "kmeans_per_head",
    "un_rope",
]


def get_task_dict(benchmark_name):
    if benchmark_name == "longbench":
        task_dict = LONGBENCH_TASKS
    elif benchmark_name == "lm_eval":
        task_dict = LM_EVAL_TASKS
    elif benchmark_name == "ruler":
        task_dict = RULER_TASKS
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")
    return task_dict


def geometric_mean(numbers):
    product = 1
    for num in numbers:
        product *= num
    return product ** (1 / len(numbers))


def process_rouge(task_results, metric):
    if metric == "rouge_geo_mean":
        scores = [
            task_results[key]
            for key in ["rouge1,none", "rouge2,none", "rougeL,none"]
        ]
        return geometric_mean(scores)
    return task_results[metric]


def get_nested_value(config, key_path):
    keys = key_path.split(".")
    value = config
    for k in keys:
        value = value.get(k, None)
        if value is None:
            return None
    return value


def format_if_numeric(value, fmt):
    return f"{value:{fmt}}" if isinstance(value, (int, float)) else value


def print_configs(config):
    key_configs = deepcopy(KEY_CONFIGS)
    print(" ===== KEY CONFIGS ===== ")
    for key in key_configs:
        value = get_nested_value(config, key)
        if value is not None:
            print(f"{key}: {value}")
    print("\n ======= RESULTS ======= ")


def parse_args():
    parser = argparse.ArgumentParser(description="Print LM evaluation results.")
    parser.add_argument(
        "-f",
        "--file_path",
        type=str,
        required=True,
        help="Path to the input JSON file with evaluation results.",
    )
    parser.add_argument(
        "-b",
        "--benchmark",
        type=str,
        default="lm_eval",
        choices=["lm_eval", "ruler", "longbench"],
        help="Benchmark to use for evaluation. Default is 'lm_eval'.",
    )
    parser.add_argument(
        "-s",
        "--show_key_configs",
        action="store_true",
        help="Whether to show key arguments.",
    )
    return parser.parse_args()


def main(benchmark, file_path, show_key_configs=False, decimal_points=4):
    task_dict = get_task_dict(benchmark)
    # Load the evaluation results from the JSON file
    with open(file_path, "r") as f:
        results = json.load(f)

    if show_key_configs:
        if "config" in results:
            print_configs(results["config"])
        else:
            print("No config found in results.\n")

    for task, metric in task_dict.items():
        if task in results:
            if isinstance(metric, list):
                for m in metric:
                    score = round(
                        process_rouge(results[task], m), decimal_points
                    )
                    print(f"{score}")
            else:
                score = round(
                    process_rouge(results[task], metric), decimal_points
                )
                print(f"{score}")
        else:
            print("")

    eff = results.get("efficiency_metrics", {})
    if eff:
        print("\nEfficiency metrics:\n")
        eval_wall_time = eff.get("eval_wall_time_minutes", "N/A")
        print(
            round(eval_wall_time, decimal_points)
            if isinstance(eval_wall_time, (int, float))
            else eval_wall_time
        )
        gpu_peak_mem = eff.get("gpu_peak_mem_gib", "N/A")
        print(
            round(gpu_peak_mem, decimal_points)
            if isinstance(gpu_peak_mem, (int, float))
            else gpu_peak_mem
        )
        gpu_kv_cache_overhead = eff.get("gpu_kv_cache_overhead_gib", "N/A")
        print(
            round(gpu_kv_cache_overhead, decimal_points)
            if isinstance(gpu_kv_cache_overhead, (int, float))
            else gpu_kv_cache_overhead
        )
        prefill_latency = eff.get("prefill_latency_ms_mean", "N/A")
        print(
            round(prefill_latency, decimal_points)
            if isinstance(prefill_latency, (int, float))
            else prefill_latency
        )
        decode_latency = eff.get("decode_latency_ms_mean", "N/A")
        print(
            round(decode_latency, decimal_points)
            if isinstance(decode_latency, (int, float))
            else decode_latency
        )

    events_stats = results.get("event_stats", {})
    if events_stats:
        print("\nEvents stats:")
        for key in events_stats:
            print(format_if_numeric(events_stats.get(key, "N/A"), ".0f"))


if __name__ == "__main__":
    args = parse_args()
    file_path = args.file_path

    main(args.benchmark, file_path, show_key_configs=args.show_key_configs)
