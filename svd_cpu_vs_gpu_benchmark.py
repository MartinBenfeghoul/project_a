import math
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from itertools import repeat

import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm


HEAD_DIM = 128
NUM_HEADS = 8
BATCH = 4
DTYPE = torch.bfloat16
WARMUP = 3
REPEATS = 10
NUM_SEGMENTS = 4

SOURCE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CPU_THREAD_COUNTS = sorted(
    {1, max(1, torch.get_num_threads() // 2), torch.get_num_threads()}
)
OUTER_WORKER_COUNTS = CPU_THREAD_COUNTS
SVD_EXECUTION_PLANS = {}

SHAPE_SPECS = [
    {"label": "single_16x128", "shape": (16, HEAD_DIM)},
    {"label": "single_32x128", "shape": (32, HEAD_DIM)},
    {"label": "single_64x128", "shape": (64, HEAD_DIM)},
    {"label": "single_128x128", "shape": (128, HEAD_DIM)},
    {"label": "heads_8x16x128", "shape": (NUM_HEADS, 16, HEAD_DIM)},
    {"label": "heads_8x32x128", "shape": (NUM_HEADS, 32, HEAD_DIM)},
    {"label": "heads_8x64x128", "shape": (NUM_HEADS, 64, HEAD_DIM)},
    {"label": "batch_4_heads_8x32x128", "shape": (BATCH, NUM_HEADS, 32, HEAD_DIM)},
    {"label": "batch_4_heads_8x64x128", "shape": (BATCH, NUM_HEADS, 64, HEAD_DIM)},
    {"label": "batch_1_heads_8x128x128", "shape": (1, NUM_HEADS, 128, HEAD_DIM)},
    {"label": "batch_4_heads_8x128x128", "shape": (BATCH, NUM_HEADS, 128, HEAD_DIM)},
    {"label": "batch_8_heads_32x128x128", "shape": (NUM_HEADS, 32, 128, HEAD_DIM)},
    {"label": "batch_8_heads_64x128x128", "shape": (NUM_HEADS, 64, 128, HEAD_DIM)},
]


@contextmanager
def temporary_cpu_threads(num_threads: int | None):
    if num_threads is None:
        yield
        return

    prev_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads)
    try:
        yield
    finally:
        torch.set_num_threads(prev_threads)


def synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def get_segment_bounds(seq_len: int, num_segments: int = NUM_SEGMENTS):
    num_segments = max(1, min(num_segments, seq_len))
    base = seq_len // num_segments
    rem = seq_len % num_segments

    bounds = []
    start = 0
    for i in range(num_segments):
        seg_len = base + (1 if i < rem else 0)
        end = start + seg_len
        bounds.append((start, end))
        start = end
    return bounds


def iter_current_style_segment_slices(x: torch.Tensor):
    """Mimic SurpriseLRKCache-style slicing: per batch item, then per segment."""
    segment_bounds = get_segment_bounds(x.shape[-2])
    if x.dim() >= 4:
        for b in range(x.shape[0]):
            for start, end in segment_bounds:
                yield x[b, ..., start:end, :]
    else:
        for start, end in segment_bounds:
            yield x[..., start:end, :]


def count_segment_calls(shape) -> int:
    per_sequence_calls = len(get_segment_bounds(shape[-2]))
    batch_factor = shape[0] if len(shape) >= 4 else 1
    return batch_factor * per_sequence_calls


def count_matrices_per_segment_call(shape) -> int:
    if len(shape) == 2:
        return 1
    if len(shape) == 3:
        return math.prod(shape[:-2])
    return math.prod(shape[1:-2]) or 1


def get_svd_execution_plan(device: torch.device, dtype: torch.dtype) -> dict:
    key = (device.type, str(dtype))
    if key in SVD_EXECUTION_PLANS:
        return SVD_EXECUTION_PLANS[key]

    if device.type == "cuda" and not torch.cuda.is_available():
        plan = {
            "available": False,
            "use_float32_fallback": False,
            "error": "CUDA not available",
        }
        SVD_EXECUTION_PLANS[key] = plan
        return plan

    plan = {
        "available": True,
        "use_float32_fallback": False,
        "error": "",
    }

    if dtype == torch.float32:
        SVD_EXECUTION_PLANS[key] = plan
        return plan

    probe = torch.randn((8, 8), device=device, dtype=dtype)
    try:
        torch.linalg.svd(probe, full_matrices=False)
    except RuntimeError as exc:
        if "not implemented for" not in str(exc):
            raise
        plan["use_float32_fallback"] = True

    SVD_EXECUTION_PLANS[key] = plan
    return plan


def run_svd_with_fallback(x: torch.Tensor, requested_dtype: torch.dtype):
    plan = get_svd_execution_plan(x.device, requested_dtype)
    if not plan["available"]:
        raise RuntimeError(plan["error"])

    work = x.to(torch.float32) if plan["use_float32_fallback"] else x
    U, S, Vh = torch.linalg.svd(work, full_matrices=False)

    if plan["use_float32_fallback"]:
        return (
            U.to(requested_dtype),
            S.to(requested_dtype),
            Vh.to(requested_dtype),
        )
    return U, S, Vh


def move_outputs_to_device(outputs, target_device: torch.device):
    if isinstance(outputs, tuple):
        return tuple(t.to(target_device) for t in outputs)
    return [tuple(t.to(target_device) for t in out) for out in outputs]


def run_parallel_matrix_svds(
    x: torch.Tensor,
    requested_dtype: torch.dtype,
    max_workers: int,
):
    matrices = x.reshape(-1, x.shape[-2], x.shape[-1]).unbind(0)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(
            executor.map(run_svd_with_fallback, matrices, repeat(requested_dtype))
        )


def run_segmented_pipeline(
    x_source: torch.Tensor,
    exec_device: torch.device,
    requested_dtype: torch.dtype,
    outer_parallel_workers: int | None = None,
):
    outputs = []
    for segment in iter_current_style_segment_slices(x_source):
        work = segment.to(exec_device)
        if outer_parallel_workers is None or work.dim() < 3:
            result = run_svd_with_fallback(work, requested_dtype)
        else:
            result = run_parallel_matrix_svds(
                work, requested_dtype, outer_parallel_workers
            )
        outputs.append(move_outputs_to_device(result, x_source.device))
    return outputs


def benchmark_segmented_pipeline(
    shape,
    exec_device,
    source_device,
    dtype=torch.float32,
    warmup=3,
    repeats=10,
    num_threads: int | None = None,
    outer_parallel_workers: int | None = None,
):
    exec_device = torch.device(exec_device)
    source_device = torch.device(source_device)

    matrices_per_call = count_matrices_per_segment_call(shape)
    effective_workers = (
        min(outer_parallel_workers, matrices_per_call)
        if outer_parallel_workers is not None
        else None
    )
    cpu_threads = 1 if effective_workers is not None else num_threads

    with temporary_cpu_threads(cpu_threads if exec_device.type == "cpu" else None):
        x = torch.randn(shape, device=source_device, dtype=dtype)
        plan = get_svd_execution_plan(exec_device, dtype)

        synchronize_if_cuda(source_device)
        for _ in range(warmup):
            run_segmented_pipeline(
                x,
                exec_device=exec_device,
                requested_dtype=dtype,
                outer_parallel_workers=effective_workers,
            )
        synchronize_if_cuda(source_device)

        times_ms = []
        for _ in range(repeats):
            synchronize_if_cuda(source_device)
            start = time.perf_counter()
            run_segmented_pipeline(
                x,
                exec_device=exec_device,
                requested_dtype=dtype,
                outer_parallel_workers=effective_workers,
            )
            synchronize_if_cuda(source_device)
            end = time.perf_counter()
            times_ms.append((end - start) * 1000.0)

    return {
        "mean_ms": statistics.mean(times_ms),
        "std_ms": statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "used_float32_fallback": plan["use_float32_fallback"],
        "segment_calls": count_segment_calls(shape),
        "matrices_per_segment_call": matrices_per_call,
        "effective_workers": effective_workers,
    }


def add_unavailable_device_results(row: dict, prefix: str, error: str) -> None:
    row[f"{prefix}_mean_ms"] = math.nan
    row[f"{prefix}_std_ms"] = math.nan
    row[f"{prefix}_min_ms"] = math.nan
    row[f"{prefix}_max_ms"] = math.nan
    row[f"{prefix}_used_float32_fallback"] = False
    row[f"{prefix}_segment_calls"] = math.nan
    row[f"{prefix}_matrices_per_segment_call"] = math.nan
    row[f"{prefix}_effective_workers"] = math.nan
    row[f"{prefix}_error"] = error


def build_results():
    records = []

    for spec in tqdm(SHAPE_SPECS, desc="segmented cpu-offload vs cuda"):
        row = {
            "label": spec["label"],
            "shape": str(spec["shape"]),
            "dtype": str(DTYPE).replace("torch.", ""),
        }

        cpu_stats = benchmark_segmented_pipeline(
            spec["shape"],
            exec_device="cpu",
            source_device=SOURCE_DEVICE,
            dtype=DTYPE,
            warmup=WARMUP,
            repeats=REPEATS,
        )
        row.update({f"cpu_{k}": v for k, v in cpu_stats.items()})
        row["cpu_error"] = ""

        cuda_plan = get_svd_execution_plan(torch.device("cuda"), DTYPE)
        if cuda_plan["available"]:
            cuda_stats = benchmark_segmented_pipeline(
                spec["shape"],
                exec_device="cuda",
                source_device=SOURCE_DEVICE,
                dtype=DTYPE,
                warmup=WARMUP,
                repeats=REPEATS,
            )
            row.update({f"cuda_{k}": v for k, v in cuda_stats.items()})
            row["cuda_error"] = ""
        else:
            add_unavailable_device_results(row, "cuda", cuda_plan["error"])

        row["cuda_speedup_vs_cpu"] = row["cpu_mean_ms"] / row["cuda_mean_ms"]
        records.append(row)

    return pd.DataFrame(records)


def build_cpu_thread_results():
    records = []

    for spec in tqdm(SHAPE_SPECS, desc="cpu-offload thread scaling"):
        baseline_mean = None
        for num_threads in CPU_THREAD_COUNTS:
            stats = benchmark_segmented_pipeline(
                spec["shape"],
                exec_device="cpu",
                source_device=SOURCE_DEVICE,
                dtype=DTYPE,
                warmup=WARMUP,
                repeats=REPEATS,
                num_threads=num_threads,
            )
            if num_threads == 1:
                baseline_mean = stats["mean_ms"]

            records.append(
                {
                    "label": spec["label"],
                    "shape": str(spec["shape"]),
                    "dtype": str(DTYPE).replace("torch.", ""),
                    "cpu_threads": num_threads,
                    "cpu_mean_ms": stats["mean_ms"],
                    "cpu_std_ms": stats["std_ms"],
                    "cpu_min_ms": stats["min_ms"],
                    "cpu_max_ms": stats["max_ms"],
                    "cpu_used_float32_fallback": stats[
                        "used_float32_fallback"
                    ],
                    "segment_calls": stats["segment_calls"],
                    "matrices_per_segment_call": stats[
                        "matrices_per_segment_call"
                    ],
                    "speedup_vs_1_thread": (
                        baseline_mean / stats["mean_ms"]
                        if baseline_mean is not None
                        else 1.0
                    ),
                }
            )

    return pd.DataFrame(records)


def build_cpu_outer_parallel_results():
    records = []
    batched_specs = [spec for spec in SHAPE_SPECS if len(spec["shape"]) >= 3]

    for spec in tqdm(batched_specs, desc="cpu-offload outer parallel scaling"):
        baseline_mean = None
        for max_workers in OUTER_WORKER_COUNTS:
            stats = benchmark_segmented_pipeline(
                spec["shape"],
                exec_device="cpu",
                source_device=SOURCE_DEVICE,
                dtype=DTYPE,
                warmup=WARMUP,
                repeats=REPEATS,
                outer_parallel_workers=max_workers,
            )
            if max_workers == 1:
                baseline_mean = stats["mean_ms"]

            records.append(
                {
                    "label": spec["label"],
                    "shape": str(spec["shape"]),
                    "dtype": str(DTYPE).replace("torch.", ""),
                    "outer_workers": max_workers,
                    "effective_workers": stats["effective_workers"],
                    "segment_calls": stats["segment_calls"],
                    "matrices_per_segment_call": stats[
                        "matrices_per_segment_call"
                    ],
                    "cpu_mean_ms": stats["mean_ms"],
                    "cpu_std_ms": stats["std_ms"],
                    "cpu_min_ms": stats["min_ms"],
                    "cpu_max_ms": stats["max_ms"],
                    "cpu_used_float32_fallback": stats[
                        "used_float32_fallback"
                    ],
                    "speedup_vs_1_worker": (
                        baseline_mean / stats["mean_ms"]
                        if baseline_mean is not None
                        else 1.0
                    ),
                }
            )

    return pd.DataFrame(records)


def plot_results(df: pd.DataFrame, output_path: str) -> None:
    labels = df["label"].tolist()
    x = range(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(
        [i - width / 2 for i in x],
        df["cpu_mean_ms"],
        width=width,
        yerr=df["cpu_std_ms"],
        capsize=4,
        label="CPU offload",
    )

    if df["cuda_mean_ms"].notna().any():
        ax.bar(
            [i + width / 2 for i in x],
            df["cuda_mean_ms"],
            width=width,
            yerr=df["cuda_std_ms"],
            capsize=4,
            label="CUDA",
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("runtime (ms)")
    ax.set_title("Segmented decomposition pipeline runtime")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_cpu_thread_results(df: pd.DataFrame, output_path: str) -> None:
    labels = df["label"].unique().tolist()
    x = list(range(len(labels)))
    width = 0.8 / len(CPU_THREAD_COUNTS)

    fig, ax = plt.subplots(figsize=(16, 6))
    for idx, num_threads in enumerate(CPU_THREAD_COUNTS):
        thread_df = (
            df[df["cpu_threads"] == num_threads]
            .set_index("label")
            .reindex(labels)
            .reset_index()
        )
        offsets = [i - 0.4 + width / 2 + idx * width for i in x]
        ax.bar(
            offsets,
            thread_df["cpu_mean_ms"],
            width=width,
            yerr=thread_df["cpu_std_ms"],
            capsize=4,
            label=f"CPU threads={num_threads}",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("runtime (ms)")
    ax.set_title("CPU-offload segmented pipeline by thread count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_cpu_outer_parallel_results(
    df: pd.DataFrame, output_path: str
) -> None:
    labels = df["label"].unique().tolist()
    x = list(range(len(labels)))
    width = 0.8 / len(OUTER_WORKER_COUNTS)

    fig, ax = plt.subplots(figsize=(16, 6))
    for idx, max_workers in enumerate(OUTER_WORKER_COUNTS):
        worker_df = (
            df[df["outer_workers"] == max_workers]
            .set_index("label")
            .reindex(labels)
            .reset_index()
        )
        offsets = [i - 0.4 + width / 2 + idx * width for i in x]
        ax.bar(
            offsets,
            worker_df["cpu_mean_ms"],
            width=width,
            yerr=worker_df["cpu_std_ms"],
            capsize=4,
            label=f"outer workers={max_workers}",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("runtime (ms)")
    ax.set_title("CPU-offload segmented pipeline with outer parallelism")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    print(f"torch: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    print(f"source device: {SOURCE_DEVICE}")
    print(f"cpu threads: {torch.get_num_threads()}")
    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}")
    print(
        f"benchmark config: dtype={DTYPE}, warmup={WARMUP}, repeats={REPEATS}, num_segments={NUM_SEGMENTS}"
    )
    print(f"cpu thread counts tested: {CPU_THREAD_COUNTS}")
    print(f"outer worker counts tested: {OUTER_WORKER_COUNTS}")
    print()
    if DTYPE == torch.bfloat16:
        print(
            "Note: unsupported CPU/CUDA dtypes are benchmarked via a float32 SVD fallback."
        )
        print(
            "Input slicing, device transfers, and output transfers are included in the measured runtime."
        )
        print()

    df = build_results()
    print(df.round(3).to_string(index=False))
    plot_path = "svd_segmented_cpu_offload_vs_cuda_benchmark.png"
    plot_results(df, plot_path)
    print()
    print(f"Saved plot to {plot_path}")

    print()
    print("Running CPU-offload thread scaling benchmark...")
    cpu_thread_df = build_cpu_thread_results()
    print(cpu_thread_df.round(3).to_string(index=False))
    cpu_thread_plot_path = "svd_cpu_offload_thread_scaling_benchmark.png"
    plot_cpu_thread_results(cpu_thread_df, cpu_thread_plot_path)
    print()
    print(f"Saved plot to {cpu_thread_plot_path}")

    print()
    print("Running CPU-offload outer parallel benchmark...")
    cpu_outer_df = build_cpu_outer_parallel_results()
    print(cpu_outer_df.round(3).to_string(index=False))
    cpu_outer_plot_path = "svd_cpu_offload_outer_parallel_benchmark.png"
    plot_cpu_outer_parallel_results(cpu_outer_df, cpu_outer_plot_path)
    print()
    print(f"Saved plot to {cpu_outer_plot_path}")


if __name__ == "__main__":
    main()
