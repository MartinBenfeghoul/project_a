import json
import argparse


LM_EVAL_TASKS = {
    "piqa": "acc,none",
    "arc_easy": "acc,none",
    "arc_challenge": "acc_norm,none",
    "hellaswag": "acc_norm,none",
    "winogrande": "acc,none",
    "mmlu": "acc,none"
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
    "ruler_qa_squad": "4096,none"
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


def get_task_dict(benchmark_name):
    if benchmark_name == 'longbench':
        task_dict = LONGBENCH_TASKS
    elif benchmark_name == 'lm_eval':
        task_dict = LM_EVAL_TASKS
    elif benchmark_name == 'ruler':
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
    if metric == 'rouge_geo_mean':
        scores = [task_results[key] for key in ["rouge1,none", "rouge2,none", "rougeL,none"]]
        return geometric_mean(scores)
    return task_results[metric]

def parse_args():
    parser = argparse.ArgumentParser(description="Print LM evaluation results.")
    parser.add_argument(
        "-f", "--file_path", type=str, required=True,
        help="Path to the input JSON file with evaluation results."
    )
    parser.add_argument(
        "-b", "--benchmark", type=str, default='lm_eval',
        choices=['lm_eval', 'ruler', 'longbench'],
        help="Benchmark to use for evaluation. Default is 'lm_eval'."
    )
    return parser.parse_args()

def main(benchmark, file_path, decimal_points=4):
    task_dict = get_task_dict(benchmark)
    # Load the evaluation results from the JSON file
    with open(file_path, 'r') as f:
        results = json.load(f)

    for task, metric in task_dict.items():
        if task in results:
            if isinstance(metric, list):
                for m in metric:
                    score = round(process_rouge(results[task], m), decimal_points)
                    print(f"{score}")
            else:
                score = round(process_rouge(results[task], metric), decimal_points)
                print(f"{score}")
        else:
            print("")

    eff = results.get("efficiency_metrics", {})
    if eff:
        print("\nEfficiency metrics:\n")
        print(round(eff.get("eval_wall_time_minutes", "N/A"), decimal_points))
        print(round(eff.get("prefill_latency_ms_mean", "N/A"), decimal_points))
        print(round(eff.get("decode_latency_ms_mean", "N/A"), decimal_points))
        print(round(eff.get("gpu_peak_mem_gib", "N/A"), decimal_points))
        print(round(eff.get("gpu_kv_cache_overhead_gib", "N/A"), decimal_points))

if __name__ == "__main__":
    args = parse_args()
    file_path = args.file_path

    main(args.benchmark, file_path)