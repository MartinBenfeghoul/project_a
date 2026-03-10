import math
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from tqdm import tqdm

import matplotlib.pyplot as plt
import pandas as pd
import torch


HEAD_DIM = 128
NUM_HEADS = 8
BATCH = 4
DTYPE = torch.bfloat16
WARMUP = 3
REPEATS = 10
SVD_EXECUTION_PLANS = {}
CPU_THREAD_COUNTS = sorted(
    {1, max(1, torch.get_num_threads() // 2), torch.get_num_threads()}
)
OUTER_WORKER_COUNTS = CPU_THREAD_COUNTS

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
]


def sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


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


def get_svd_execution_plan(
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
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

    if device.type == "cpu" and dtype == torch.float32:
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


def run_svd_with_fallback(
    x: torch.Tensor,
    requested_dtype: torch.dtype,
):
    plan = get_svd_execution_plan(x.device, requested_dtype)
    if plan["use_float32_fallback"]:
        x = x.to(torch.float32)
        U, S, Vh = torch.linalg.svd(x, full_matrices=False)
        return (
            U.to(requested_dtype),
            S.to(requested_dtype),
            Vh.to(requested_dtype),
        )
    return torch.linalg.svd(x, full_matrices=False)


def benchmark_svd(
    shape,
    device,
    dtype=torch.float32,
    warmup=3,
    repeats=10,
    num_threads: int | None = None,
):
    device = torch.device(device)
    with temporary_cpu_threads(num_threads if device.type == "cpu" else None):
        x = torch.randn(shape, device=device, dtype=dtype)
        plan = get_svd_execution_plan(device, dtype)

        sync_if_needed(device)
        for _ in range(warmup):
            run_svd_with_fallback(x, dtype)
        sync_if_needed(device)

        times_ms = []
        for _ in range(repeats):
            sync_if_needed(device)
            start = time.perf_counter()
            run_svd_with_fallback(x, dtype)
            sync_if_needed(device)
            end = time.perf_counter()
            times_ms.append((end - start) * 1000.0)

    return {
        "mean_ms": statistics.mean(times_ms),
        "std_ms": statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "used_float32_fallback": plan["use_float32_fallback"],
    }


def run_batched_svd_independently(
    x: torch.Tensor,
    requested_dtype: torch.dtype,
    max_workers: int,
):
    matrices = x.reshape(-1, x.shape[-2], x.shape[-1]).unbind(0)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(lambda m: run_svd_with_fallback(m, requested_dtype), matrices))


def benchmark_cpu_outer_parallel_svd(
    shape,
    dtype=torch.float32,
    warmup=3,
    repeats=10,
    max_workers=1,
):
    if len(shape) < 3:
        raise ValueError(
            f"Outer parallel benchmark requires batched input, got shape={shape}."
        )

    num_matrices = math.prod(shape[:-2])
    effective_workers = min(max_workers, num_matrices)

    with temporary_cpu_threads(1):
        x = torch.randn(shape, device="cpu", dtype=dtype)
        plan = get_svd_execution_plan(torch.device("cpu"), dtype)

        sync_if_needed(torch.device("cpu"))
        for _ in range(warmup):
            run_batched_svd_independently(x, dtype, effective_workers)
        sync_if_needed(torch.device("cpu"))

        times_ms = []
        for _ in range(repeats):
            sync_if_needed(torch.device("cpu"))
            start = time.perf_counter()
            run_batched_svd_independently(x, dtype, effective_workers)
            sync_if_needed(torch.device("cpu"))
            end = time.perf_counter()
            times_ms.append((end - start) * 1000.0)

    return {
        "mean_ms": statistics.mean(times_ms),
        "std_ms": statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "used_float32_fallback": plan["use_float32_fallback"],
        "num_matrices": num_matrices,
        "effective_workers": effective_workers,
    }


def add_device_results(
    row: dict,
    device_name: str,
    shape,
    dtype: torch.dtype,
):
    device = torch.device(device_name)
    plan = get_svd_execution_plan(device, dtype)

    if not plan["available"]:
        row[f"{device_name}_mean_ms"] = math.nan
        row[f"{device_name}_std_ms"] = math.nan
        row[f"{device_name}_min_ms"] = math.nan
        row[f"{device_name}_max_ms"] = math.nan
        row[f"{device_name}_used_float32_fallback"] = False
        row[f"{device_name}_error"] = plan["error"]
        return

    stats = benchmark_svd(
        shape,
        device=device_name,
        dtype=dtype,
        warmup=WARMUP,
        repeats=REPEATS,
    )
    row.update({f"{device_name}_{k}": v for k, v in stats.items()})
    row[f"{device_name}_error"] = ""


def build_cpu_thread_results():
    records = []

    for spec in tqdm(SHAPE_SPECS, desc="cpu thread scaling"):
        baseline_mean = None
        for num_threads in CPU_THREAD_COUNTS:
            stats = benchmark_svd(
                spec["shape"],
                device="cpu",
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
    batched_specs = [
        spec for spec in SHAPE_SPECS if len(spec["shape"]) >= 3
    ]

    for spec in tqdm(batched_specs, desc="cpu outer parallel scaling"):
        baseline_mean = None
        for max_workers in OUTER_WORKER_COUNTS:
            stats = benchmark_cpu_outer_parallel_svd(
                spec["shape"],
                dtype=DTYPE,
                warmup=WARMUP,
                repeats=REPEATS,
                max_workers=max_workers,
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
                    "num_matrices": stats["num_matrices"],
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


def build_results():
    records = []

    for spec in  tqdm(SHAPE_SPECS):
        row = {
            "label": spec["label"],
            "shape": str(spec["shape"]),
            "dtype": str(DTYPE).replace("torch.", ""),
        }

        add_device_results(row, "cpu", spec["shape"], DTYPE)
        add_device_results(row, "cuda", spec["shape"], DTYPE)
        row["cuda_speedup_vs_cpu"] = row["cpu_mean_ms"] / row["cuda_mean_ms"]

        records.append(row)

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
        label="CPU",
    )

    if df["cuda_mean_ms"].notna().any():
        ax.bar(
            [i + width / 2 for i in x],
            df["cuda_mean_ms"],
            width=width,
            yerr=df["cuda_std_ms"],
            capsize=4,
            label="GPU",
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("runtime (ms)")
    ax.set_title("torch.linalg.svd runtime: mean and standard deviation")
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
    ax.set_title("CPU SVD runtime by thread count")
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
    ax.set_title("CPU SVD runtime with outer parallelism across batch/head dims")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    print(f"torch: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    print(f"cpu threads: {torch.get_num_threads()}")
    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}")
    print(
        f"benchmark config: dtype={DTYPE}, warmup={WARMUP}, repeats={REPEATS}"
    )
    print(f"cpu thread counts tested: {CPU_THREAD_COUNTS}")
    print(f"outer worker counts tested: {OUTER_WORKER_COUNTS}")
    print()
    if DTYPE == torch.bfloat16:
        print(
            "Note: unsupported CPU/CUDA dtypes are benchmarked via a float32 SVD fallback"
        )
        print("with input/output casts included in the measured runtime.")
        print()

    df = build_results()
    print(df.round(3).to_string(index=False))
    plot_path = "svd_cpu_vs_cuda_benchmark.png"
    plot_results(df, plot_path)
    print()
    print(f"Saved plot to {plot_path}")

    print()
    print("Running CPU thread scaling benchmark...")
    cpu_thread_df = build_cpu_thread_results()
    print(cpu_thread_df.round(3).to_string(index=False))
    cpu_thread_plot_path = "svd_cpu_thread_scaling_benchmark.png"
    plot_cpu_thread_results(cpu_thread_df, cpu_thread_plot_path)
    print()
    print(f"Saved plot to {cpu_thread_plot_path}")

    print()
    print("Running CPU outer parallel benchmark across batch/head dims...")
    cpu_outer_df = build_cpu_outer_parallel_results()
    print(cpu_outer_df.round(3).to_string(index=False))
    cpu_outer_plot_path = "svd_cpu_outer_parallel_benchmark.png"
    plot_cpu_outer_parallel_results(cpu_outer_df, cpu_outer_plot_path)
    print()
    print(f"Saved plot to {cpu_outer_plot_path}")


if __name__ == "__main__":
    main()
