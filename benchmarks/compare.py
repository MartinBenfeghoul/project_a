"""Compare exact KV, full-adaptation LowRAM, and meta-initialised LowRAM-SR."""

import argparse
import gc
import json
import math
import os
from pathlib import Path
import statistics
import time
import warnings
import torch

from benchmarks.runtime import load_model, cache_config, attention_mode
from benchmarks.tasks import retrieval_examples, score_answer
from model.unpadded_attention import attention_for_batch
from utils.model import extract_kv_linear_init, get_device


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument(
        "--methods",
        nargs="+",
        choices=["baseline", "lowram", "fast"],
        default=["baseline", "lowram"],
    )
    p.add_argument("--contexts", nargs="+", type=int, default=[4096, 16384])
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[1])
    p.add_argument("--ratios", nargs="+", type=float, default=[4])
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--meta-weights")
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--decode-tokens", type=int, default=64)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compile selective value decode for fast (default); warmup required.",
    )
    p.add_argument(
        "--output", type=Path, default=Path("results/demo/comparison.json")
    )
    args = p.parse_args(argv)
    if any(
        x < 1
        for x in args.contexts
        + args.batch_sizes
        + [args.samples, args.steps, args.repeats]
    ):
        p.error(
            "Contexts, batches, samples, steps, and repeats must be positive."
        )
    if any(not math.isfinite(x) or x <= 1 for x in args.ratios):
        p.error("Compression ratios must be finite and greater than 1.")
    if args.decode_tokens < 2 or args.warmup < 0:
        p.error("Use at least two output tokens and nonnegative warmup.")
    if args.compile and "fast" in args.methods and args.warmup < 1:
        p.error("Compilation needs at least one warmup; or pass --no-compile.")
    if "fast" in args.methods and (
        not args.meta_weights or not Path(args.meta_weights).is_file()
    ):
        p.error(
            "fast requires an existing --meta-weights checkpoint for the chosen model."
        )
    if "baseline" not in args.methods:
        p.error("Include baseline to produce a paired comparison.")
    if any(args.samples % b for b in args.batch_sizes):
        p.error(
            "--samples must be divisible by every batch size."
        )
    return args


def summarise(values):
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def measure_batch(model, tokenizer, config, examples, decode_tokens, device):
    import torch
    from cache import CompressedCache

    cuda = device.type == "cuda"
    sync = lambda: torch.cuda.synchronize(device) if cuda else None
    lengths = [len(e["input_ids"]) for e in examples]
    width = max(lengths)
    ids = torch.full(
        (len(examples), width),
        tokenizer.pad_token_id,
        dtype=torch.long,
        device=device,
    )
    mask = torch.zeros_like(ids)
    for row, example in enumerate(examples):
        ids[row, -lengths[row] :] = torch.tensor(
            example["input_ids"], device=device
        )
        mask[row, -lengths[row] :] = 1
    full_mask = torch.cat(
        [mask, mask.new_ones((len(examples), decode_tokens - 1))], dim=1
    )
    positions = torch.arange(width + decode_tokens - 1, device=device)
    position_ids = (mask.cumsum(-1) - 1).clamp_min(0)
    tokens = []
    sync()
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        start = time.perf_counter()
        cache = CompressedCache(
            config=config, cache_context={"padding_mask": mask}, verbose=False
        )
        output = model(
            input_ids=ids,
            attention_mask=mask,
            position_ids=position_ids,
            cache_position=positions[:width],
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
        next_token = output.logits[:, -1].argmax(-1, keepdim=True)
        tokens.append(next_token)
        del output
        sync()
        ttft = time.perf_counter() - start
        storage = dict(cache.prefill_compression_stats(mask))
        prefill_allocated = (
            torch.cuda.memory_allocated(device) if cuda else None
        )
        prefill_peak = torch.cuda.max_memory_allocated(device) if cuda else None
        sync()
        start = time.perf_counter()
        for step in range(decode_tokens - 1):
            output = model(
                input_ids=next_token,
                attention_mask=full_mask[:, : width + step + 1],
                position_ids=position_ids[:, -1:] + step + 1,
                cache_position=positions[width + step : width + step + 1],
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            next_token = output.logits[:, -1].argmax(-1, keepdim=True)
            tokens.append(next_token)
            del output
        sync()
        decode_s = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated(device) if cuda else None
        reserved = torch.cuda.max_memory_reserved(device) if cuda else None
        token_rows = torch.cat(tokens, dim=1).cpu().tolist()
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    answers = []
    for example, row in zip(examples, token_rows):
        end = next(
            (i for i, token in enumerate(row) if token in eos_ids), len(row)
        )
        answer = tokenizer.decode(row[:end], skip_special_tokens=True)
        answers.append(
            {
                "example_id": example["id"],
                "answer": answer,
                "output_tokens_before_eos": end,
                **score_answer(answer, example["references"]),
            }
        )
    return {
        "ttft_s": ttft,
        "decode_s": decode_s,
        "decode_tokens_per_s": len(examples) * (decode_tokens - 1) / decode_s,
        "end_to_end_tokens_per_s": len(examples)
        * decode_tokens
        / (ttft + decode_s),
        "storage": storage,
        "prefill_allocated_bytes": prefill_allocated,
        "prefill_peak_allocated_bytes": prefill_peak,
        "peak_allocated_bytes": peak,
        "peak_reserved_bytes": reserved,
        "prompt_tokens": lengths,
        "answers": answers,
    }


def aggregate(runs):
    summary = {
        key: summarise([run[key] for run in runs])
        for key in (
            "ttft_s",
            "decode_s",
            "decode_tokens_per_s",
            "end_to_end_tokens_per_s",
        )
    }
    for key in (
        "physical_compressed_bytes",
        "original_bytes",
        "physical_compression_ratio",
    ):
        summary[key] = summarise([run["storage"][key] for run in runs])
    for key in (
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "prefill_allocated_bytes",
        "prefill_peak_allocated_bytes",
    ):
        summary[key] = (
            max(run[key] for run in runs) if runs[0][key] is not None else None
        )
    answers = [
        answer
        for run in runs
        if run["repeat"] == 0
        for answer in run["answers"]
    ]
    scores = [answer["score"] for answer in answers]
    summary["accuracy"] = {
        "score": statistics.fmean(scores),
        "n": len(scores),
        "all_correct_fraction": statistics.fmean(
            a["all_correct"] for a in answers
        ),
    }
    return summary


def save(report, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(report, indent=2, allow_nan=False, default=str) + "\n")
    temp.replace(path)


def main(argv=None):
    args = parse_args(argv)

    torch.manual_seed(42)
    model, tokenizer = load_model(args.model)
    if len(set(getattr(model, "hf_device_map", {}).values())) > 1:
        raise RuntimeError(
            "The model was split across GPUs; set CUDA_VISIBLE_DEVICES to one GPU."
        )
    device = torch.device(get_device(model))
    linear = (
        extract_kv_linear_init(model)
        if any(m != "baseline" for m in args.methods)
        else None
    )
    decode_compiled = False
    if args.compile and "fast" in args.methods:
        from cache.backends.mlp_values import enable_decode_compilation

        decode_compiled = enable_decode_compilation()
    cuda = device.type == "cuda"
    report = {
        "schema_version": 1,
        "kind": "lowram_comparison",
        "metadata": {
            "model": args.model,
            "config": vars(args),
            "decode_compiled": decode_compiled,
            "gpu": torch.cuda.get_device_name(device) if cuda else None,
            "kernel_build_disabled": os.environ.get("XKV_NO_BUILD") == "1",
        },
        "examples": [],
        "comparisons": [],
    }
    max_positions = model.config.max_position_embeddings
    for context in args.contexts:
        if context + args.decode_tokens > max_positions:
            warnings.warn(
                f"{context} + output exceeds model context limit {max_positions}."
            )
        examples = retrieval_examples(
            tokenizer, context, args.samples, 42
        )
        report["examples"].extend(
            {**e, "context_tokens": context} for e in examples
        )
        for batch in args.batch_sizes:
            for method in dict.fromkeys(args.methods):
                for ratio in (
                    [1] if method == "baseline" else dict.fromkeys(args.ratios)
                ):
                    config = cache_config(
                        model,
                        method,
                        ratio,
                        args.steps,
                        args.meta_weights,
                        linear,
                    )
                    comparison = {
                        "method": method,
                        "target_ratio": ratio,
                        "context_tokens": context,
                        "batch_size": batch,
                        "adaptation_steps": (
                            0
                            if method == "baseline"
                            else 2 if method == "fast" else args.steps
                        ),
                        "status": "running",
                        "runs": [],
                    }
                    report["comparisons"].append(comparison)
                    print(
                        f"{method} / {ratio}x / {context} tokens / batch {batch}",
                        flush=True,
                    )
                    try:
                        with (
                            attention_for_batch(model, batch),
                            attention_mode(model, method),
                        ):
                            comparison["attention_implementation"] = (
                                model.config._attn_implementation
                            )
                            for _ in range(args.warmup):
                                torch.manual_seed(42)
                                measure_batch(
                                    model,
                                    tokenizer,
                                    config,
                                    examples[:batch],
                                    args.decode_tokens,
                                    device,
                                )
                            for repeat in range(args.repeats):
                                for offset in range(0, len(examples), batch):
                                    torch.manual_seed(42 + offset)
                                    run = measure_batch(
                                        model,
                                        tokenizer,
                                        config,
                                        examples[offset : offset + batch],
                                        args.decode_tokens,
                                        device,
                                    )
                                    comparison["runs"].append(
                                        {
                                            **run,
                                            "repeat": repeat,
                                            "batch_offset": offset,
                                        }
                                    )
                        comparison["summary"] = aggregate(comparison["runs"])
                        comparison["status"] = "ok"
                        summary = comparison["summary"]
                        print(
                            f"  {summary['physical_compression_ratio']['mean']:.2f}x actual; "
                            f"{summary['decode_tokens_per_s']['mean']:.1f} tok/s; "
                            f"{summary['accuracy']['score']:.1%} score",
                            flush=True,
                        )
                    except torch.OutOfMemoryError:
                        comparison["status"] = "oom"
                        comparison["error"] = (
                            "Device ran out of memory; incomplete runs are excluded."
                        )
                        print("  OOM", flush=True)
                    finally:
                        gc.collect()
                        if cuda:
                            torch.cuda.empty_cache()
                        save(report, args.output)
    for row in report["comparisons"]:
        baseline = next(
            (
                b
                for b in report["comparisons"]
                if b["method"] == "baseline"
                and b["context_tokens"] == row["context_tokens"]
                and b["batch_size"] == row["batch_size"]
                and b["status"] == "ok"
            ),
            None,
        )
        if baseline and row["status"] == "ok":
            base_score = baseline["summary"]["accuracy"]["score"]
            score = row["summary"]["accuracy"]["score"]
            row["summary"]["accuracy"]["delta_percentage_points"] = 100 * (
                score - base_score
            )
            row["summary"]["accuracy"]["recovered_percent"] = (
                100 * score / base_score if base_score else None
            )
    save(report, args.output)
    print(
        f"Saved {args.output}."
    )
    return report


if __name__ == "__main__":
    main()
