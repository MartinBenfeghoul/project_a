import os
import json
import argparse
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
from utils import (
    get_model_and_tokenizer,
    extract_kv_linear_init,
    get_device,
    list_of_strings,
    get_output_path,
    filter_tasks_by_min_seq_len,
)

GEN_KWARGS = {
    "do_sample": False,
    "use_cache": True,
    "logits_to_keep": 1,
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
    model, tokenizer = get_model_and_tokenizer(args.model_name)
    selective_handles = (
        install_selective_attention(model)
        if args.selective_reconstruction
        else []
    )

    attn_predictor_hook_handles = get_attn_predictor_hook_handles(args, model)

    num_layers = model.config.num_hidden_layers
    key_cache_kwargs = {
        "cache_type": args.k_cache_type,
        "comp_ratio": args.comp_ratio,
        "layer_group_size": args.xkv_layer_group_size,
        "xkv_svd_backend": args.xkv_svd_backend,
        "num_layers": num_layers,
        "selective_reconstruction": args.selective_reconstruction,
        "quantise_a": args.k_quantise_a,
        "quantise_b": args.k_quantise_b,
        "compressor_bits": args.k_compressor_bits,
    }

    value_cache_kwargs = {
        "cache_type": args.v_cache_type,
        "num_epochs": args.num_epochs,
        "meta_weights_path": args.meta_weights_path,
        "value_mlp_weights_path": args.value_mlp_weights_path,
        "use_residual": args.use_residual,
        "target_cr": args.target_cr,
        "turboquant_residuals": args.v_turboquant_residuals,
        "compressor_bits": args.v_compressor_bits,
    }
    if args.use_residual and args.v_cache_type == "mlp":
        value_cache_kwargs["W_linear_per_layer"] = extract_kv_linear_init(model)

    model.eval()

    args.tasks = get_tasks(args.tasks)

    lm = CompressedCacheHFLM(
        key_cache_kwargs=key_cache_kwargs,
        value_cache_kwargs=value_cache_kwargs,
        eviction_keep_ratio=args.eviction_keep_ratio,
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

    results = evaluator.simple_evaluate(
        model=lm,
        gen_kwargs=GEN_KWARGS,
        tasks=eval_tasks,
        num_fewshot=0,
        batch_size=args.batch_size,
        max_batch_size=args.batch_size,
        device=get_device(lm),
        task_manager=tm,
        limit=args.limit,
    )

    print(make_table(results))

    results["results"]["config"] = vars(args)

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
    for handle in attn_predictor_hook_handles + selective_handles:
        handle.remove()
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
        "-t", "--tasks", type=list_of_strings, default=["longbench"]
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
    # key cache
    parser.add_argument(
        "-kc",
        "--k_cache_type",
        choices=["baseline", "xkv", "turboquant"],
        default="xkv",
    )
    parser.add_argument("-r", "--comp_ratio", type=float, default=2.0)
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
    parser.add_argument(
        "--target_cr",
        type=float,
        default=None,
        help="sets MLP target compression ratio.",
    )
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--meta_weights_path", type=str, default=None)
    parser.add_argument(
        "--value_mlp_weights_path",
        type=str,
        default=None,
        help="Path to plain pretrained value-cache MLP weights from train_value_mlps.py.",
    )
    parser.add_argument(
        "--use_residual",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add a linear residual W_linear to the MLP, initialised as pinv(W_k) @ W_v from the model's projection weights. Use --no-use_residual to disable.",
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
    if args.v_cache_type == "mlp" and args.target_cr is None:
        parser.error("--target_cr is required when --v_cache_type=mlp.")
    if args.eviction_keep_ratio < 1 and not args.use_attn_predictor:
        raise ValueError(
            "--eviction_keep_ratio < 1 requires --use_attn_predictor."
        )

    print("Config for lm-eval: ", vars(args))

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
