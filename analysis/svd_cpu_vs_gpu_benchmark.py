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
FLATTENED_HEAD_DIM = NUM_HEADS * HEAD_DIM
BATCH = 4
DTYPE = torch.bfloat16
WARMUP = 3
REPEATS = 10
NUM_SEGMENTS = 4
XKV_CHUNK_SIZES = (128, 512, 1024, 2048, 4096)
XKV_DIM_SPECS = (
    (1, FLATTENED_HEAD_DIM),
    (2, FLATTENED_HEAD_DIM * 2),
    (4, FLATTENED_HEAD_DIM * 4),
)
XKV_SUCCESSIVE_CHUNK_SIZE = 512
XKV_SUCCESSIVE_CHUNK_COUNTS = (1, 2, 4, 8)

SOURCE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CPU_THREAD_COUNTS = sorted(
    {1, max(1, torch.get_num_threads() // 2), torch.get_num_threads()}
)
OUTER_WORKER_COUNTS = CPU_THREAD_COUNTS
SVD_EXECUTION_PLANS = {}

ORIGINAL_SHAPE_SPECS = [
    {"label": "single_16x128", "shape": (16, HEAD_DIM)},
    {"label": "single_32x128", "shape": (32, HEAD_DIM)},
    {"label": "single_64x128", "shape": (64, HEAD_DIM)},
    {"label": "single_128x128", "shape": (128, HEAD_DIM)},
    {"label": "heads_8x16x128", "shape": (NUM_HEADS, 16, HEAD_DIM)},
    {"label": "heads_8x32x128", "shape": (NUM_HEADS, 32, HEAD_DIM)},
    {"label": "heads_8x64x128", "shape": (NUM_HEADS, 64, HEAD_DIM)},
    {
        "label": "batch_4_heads_8x32x128",
        "shape": (BATCH, NUM_HEADS, 32, HEAD_DIM),
    },
    {
        "label": "batch_4_heads_8x64x128",
        "shape": (BATCH, NUM_HEADS, 64, HEAD_DIM),
    },
    {
        "label": "batch_1_heads_8x128x128",
        "shape": (1, NUM_HEADS, 128, HEAD_DIM),
    },
    {
        "label": "batch_4_heads_8x128x128",
        "shape": (BATCH, NUM_HEADS, 128, HEAD_DIM),
    },
    {
        "label": "batch_8_heads_32x128x128",
        "shape": (NUM_HEADS, 32, 128, HEAD_DIM),
    },
    {
        "label": "batch_8_heads_64x128x128",
        "shape": (NUM_HEADS, 64, 128, HEAD_DIM),
    },
]

FLATTENED_HEADS_SHAPE_SPECS = [
    {
        "label": (
            f"flat_{chunk}x{feature_dim}"
            if num_layers == 1
            else f"flat_layers_{num_layers}_{chunk}x{feature_dim}"
        ),
        "shape": (chunk, feature_dim),
        "feature_dim": feature_dim,
        "num_layers": num_layers,
    }
    for num_layers, feature_dim in XKV_DIM_SPECS
    for chunk in XKV_CHUNK_SIZES
]

KMEANS_XKV_SUCCESSIVE_CHUNK_SPECS = [
    {
        "label": (
            f"succ_{num_chunks}x{XKV_SUCCESSIVE_CHUNK_SIZE}x{feature_dim}"
            if num_layers == 1
            else (
                f"succ_layers_{num_layers}_{num_chunks}x"
                f"{XKV_SUCCESSIVE_CHUNK_SIZE}x{feature_dim}"
            )
        ),
        "shape": (num_chunks * XKV_SUCCESSIVE_CHUNK_SIZE, feature_dim),
        "feature_dim": feature_dim,
        "num_layers": num_layers,
        "chunk_size": XKV_SUCCESSIVE_CHUNK_SIZE,
        "num_chunks": num_chunks,
        "segment_lengths": [XKV_SUCCESSIVE_CHUNK_SIZE] * num_chunks,
    }
    for num_layers, feature_dim in XKV_DIM_SPECS
    for num_chunks in XKV_SUCCESSIVE_CHUNK_COUNTS
]

BENCHMARK_CONFIGS = [
    {
        "name": "original",
        "description": "Original heads-separated setting",
        "shape_specs": ORIGINAL_SHAPE_SPECS,
        "output_suffix": "",
        "plot_title_suffix": "",
    },
    {
        "name": "flattened_heads",
        "description": "Flattened-heads XKVKeysCache-style setting",
        "shape_specs": FLATTENED_HEADS_SHAPE_SPECS,
        "output_suffix": "_flattened_heads",
        "plot_title_suffix": " (flattened heads / XKVKeysCache-style)",
        "split_plots_by": "feature_dim",
    },
    {
        "name": "kmeans_xkv_successive_chunks",
        "description": (
            "KMeans xKV-style successive 512-token chunks "
            "(no clustering, per-chunk decomposition)"
        ),
        "shape_specs": KMEANS_XKV_SUCCESSIVE_CHUNK_SPECS,
        "output_suffix": "_kmeans_xkv_successive_chunks",
        "plot_title_suffix": (
            " (kmeans xKV-style successive 512-token chunks, no clustering)"
        ),
        "split_plots_by": "feature_dim",
        "benchmark_type": "successive_chunks",
    },
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


def get_segment_bounds_from_lengths(segment_lengths):
    bounds = []
    start = 0
    for length in segment_lengths:
        end = start + length
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


def iter_segment_slices(x: torch.Tensor, segment_bounds):
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


def run_svd_on_device(
    segment: torch.Tensor,
    exec_device: torch.device,
    requested_dtype: torch.dtype,
):
    work = segment.to(exec_device)
    return run_svd_with_fallback(work, requested_dtype)


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
            executor.map(
                run_svd_with_fallback, matrices, repeat(requested_dtype)
            )
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

    with temporary_cpu_threads(
        cpu_threads if exec_device.type == "cpu" else None
    ):
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


def run_successive_chunk_pipeline(
    x_source: torch.Tensor,
    exec_device: torch.device,
    requested_dtype: torch.dtype,
    segment_lengths,
    cpu_segment_workers: int | None = None,
):
    segment_bounds = get_segment_bounds_from_lengths(segment_lengths)
    if segment_bounds and segment_bounds[-1][1] != x_source.shape[-2]:
        raise ValueError(
            "segment lengths must sum to the sequence dimension of the input."
        )

    segments = list(iter_segment_slices(x_source, segment_bounds))
    if (
        exec_device.type == "cpu"
        and cpu_segment_workers is not None
        and len(segments) > 1
    ):
        with ThreadPoolExecutor(max_workers=cpu_segment_workers) as executor:
            results = list(
                executor.map(
                    run_svd_on_device,
                    segments,
                    repeat(exec_device),
                    repeat(requested_dtype),
                )
            )
        return [
            move_outputs_to_device(result, x_source.device)
            for result in results
        ]

    outputs = []
    for segment in segments:
        result = run_svd_on_device(segment, exec_device, requested_dtype)
        outputs.append(move_outputs_to_device(result, x_source.device))
    return outputs


def benchmark_successive_chunk_pipeline(
    spec,
    exec_device,
    source_device,
    dtype=torch.float32,
    warmup=3,
    repeats=10,
    cpu_segment_workers: int | None = None,
):
    exec_device = torch.device(exec_device)
    source_device = torch.device(source_device)
    segment_lengths = spec["segment_lengths"]
    effective_workers = (
        min(cpu_segment_workers, len(segment_lengths))
        if cpu_segment_workers is not None
        else None
    )
    cpu_threads = 1 if effective_workers is not None else None

    with temporary_cpu_threads(
        cpu_threads if exec_device.type == "cpu" else None
    ):
        x = torch.randn(spec["shape"], device=source_device, dtype=dtype)
        plan = get_svd_execution_plan(exec_device, dtype)

        synchronize_if_cuda(source_device)
        for _ in range(warmup):
            run_successive_chunk_pipeline(
                x,
                exec_device=exec_device,
                requested_dtype=dtype,
                segment_lengths=segment_lengths,
                cpu_segment_workers=effective_workers,
            )
        synchronize_if_cuda(source_device)

        times_ms = []
        for _ in range(repeats):
            synchronize_if_cuda(source_device)
            start = time.perf_counter()
            run_successive_chunk_pipeline(
                x,
                exec_device=exec_device,
                requested_dtype=dtype,
                segment_lengths=segment_lengths,
                cpu_segment_workers=effective_workers,
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
        "segment_calls": len(segment_lengths),
        "matrices_per_segment_call": 1,
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


def add_spec_metadata(row: dict, spec: dict) -> None:
    for key in ("feature_dim", "num_layers", "chunk_size", "num_chunks"):
        if key in spec:
            row[key] = spec[key]


def build_results(shape_specs, desc: str):
    records = []

    for spec in tqdm(shape_specs, desc=desc):
        row = {
            "label": spec["label"],
            "shape": str(spec["shape"]),
            "dtype": str(DTYPE).replace("torch.", ""),
        }
        add_spec_metadata(row, spec)

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


def build_cpu_thread_results(shape_specs, desc: str):
    records = []

    for spec in tqdm(shape_specs, desc=desc):
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
                    "cpu_used_float32_fallback": stats["used_float32_fallback"],
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
            add_spec_metadata(records[-1], spec)

    return pd.DataFrame(records)


def build_cpu_outer_parallel_results(shape_specs, desc: str):
    records = []
    batched_specs = [spec for spec in shape_specs if len(spec["shape"]) >= 3]

    for spec in tqdm(batched_specs, desc=desc):
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
                    "cpu_used_float32_fallback": stats["used_float32_fallback"],
                    "speedup_vs_1_worker": (
                        baseline_mean / stats["mean_ms"]
                        if baseline_mean is not None
                        else 1.0
                    ),
                }
            )
            add_spec_metadata(records[-1], spec)

    return pd.DataFrame(records)


def build_successive_chunk_cpu_vs_gpu_results(shape_specs, desc: str):
    records = []
    cuda_plan = get_svd_execution_plan(torch.device("cuda"), DTYPE)

    for spec in tqdm(shape_specs, desc=desc):
        if cuda_plan["available"]:
            cuda_stats = benchmark_successive_chunk_pipeline(
                spec,
                exec_device="cuda",
                source_device=SOURCE_DEVICE,
                dtype=DTYPE,
                warmup=WARMUP,
                repeats=REPEATS,
            )
            cuda_error = ""
        else:
            cuda_stats = None
            cuda_error = cuda_plan["error"]

        for cpu_workers in CPU_THREAD_COUNTS:
            cpu_stats = benchmark_successive_chunk_pipeline(
                spec,
                exec_device="cpu",
                source_device=SOURCE_DEVICE,
                dtype=DTYPE,
                warmup=WARMUP,
                repeats=REPEATS,
                cpu_segment_workers=cpu_workers,
            )
            row = {
                "label": spec["label"],
                "shape": str(spec["shape"]),
                "dtype": str(DTYPE).replace("torch.", ""),
                "cpu_workers": cpu_workers,
                "cpu_mean_ms": cpu_stats["mean_ms"],
                "cpu_std_ms": cpu_stats["std_ms"],
                "cpu_min_ms": cpu_stats["min_ms"],
                "cpu_max_ms": cpu_stats["max_ms"],
                "cpu_used_float32_fallback": cpu_stats[
                    "used_float32_fallback"
                ],
                "cpu_segment_calls": cpu_stats["segment_calls"],
                "cpu_matrices_per_segment_call": cpu_stats[
                    "matrices_per_segment_call"
                ],
                "cpu_effective_workers": cpu_stats["effective_workers"],
                "cpu_error": "",
            }
            add_spec_metadata(row, spec)
            if cuda_stats is not None:
                row.update({f"cuda_{k}": v for k, v in cuda_stats.items()})
                row["cuda_error"] = cuda_error
                row["cuda_speedup_vs_cpu"] = (
                    row["cpu_mean_ms"] / row["cuda_mean_ms"]
                )
            else:
                add_unavailable_device_results(row, "cuda", cuda_error)
                row["cuda_speedup_vs_cpu"] = math.nan
            records.append(row)

    return pd.DataFrame(records)


def plot_results(
    df: pd.DataFrame, output_path: str, title_suffix: str = ""
) -> None:
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
    ax.set_title(f"Segmented decomposition pipeline runtime{title_suffix}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_cpu_thread_results(
    df: pd.DataFrame, output_path: str, title_suffix: str = ""
) -> None:
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
    ax.set_title(
        f"CPU-offload segmented pipeline by thread count{title_suffix}"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_cpu_outer_parallel_results(
    df: pd.DataFrame, output_path: str, title_suffix: str = ""
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
    ax.set_title(
        f"CPU-offload segmented pipeline with outer parallelism{title_suffix}"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_successive_chunk_cpu_vs_gpu_results(
    df: pd.DataFrame, output_path: str, title_suffix: str = ""
) -> None:
    labels = df["label"].unique().tolist()
    x = list(range(len(labels)))
    series_count = len(CPU_THREAD_COUNTS) + 1
    width = 0.8 / series_count

    fig, ax = plt.subplots(figsize=(18, 6))
    for idx, cpu_workers in enumerate(CPU_THREAD_COUNTS):
        worker_df = (
            df[df["cpu_workers"] == cpu_workers]
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
            label=f"CPU workers={cpu_workers}",
        )

    gpu_df = df.drop_duplicates(subset=["label"]).set_index("label").reindex(
        labels
    )
    if gpu_df["cuda_mean_ms"].notna().any():
        offsets = [
            i - 0.4 + width / 2 + len(CPU_THREAD_COUNTS) * width for i in x
        ]
        ax.bar(
            offsets,
            gpu_df["cuda_mean_ms"],
            width=width,
            yerr=gpu_df["cuda_std_ms"],
            capsize=4,
            label="CUDA",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("runtime (ms)")
    ax.set_title(
        f"Successive-chunk decomposition runtime by CPU workers vs CUDA{title_suffix}"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def format_group_title_suffix(
    title_suffix: str, group_column: str, group_value
) -> str:
    if group_column == "feature_dim":
        return f"{title_suffix} (feature dim={group_value})"
    return f"{title_suffix} ({group_column}={group_value})"


def plot_grouped_results(
    df: pd.DataFrame,
    output_suffix: str,
    title_suffix: str,
    group_column: str | None,
) -> None:
    if group_column is None:
        plot_path = f"svd_segmented_cpu_offload_vs_cuda_benchmark{output_suffix}.png"
        plot_results(df, plot_path, title_suffix=title_suffix)
        print()
        print(f"Saved plot to {plot_path}")
        return

    for group_value in sorted(df[group_column].dropna().unique()):
        group_df = df[df[group_column] == group_value]
        group_plot_path = (
            "svd_segmented_cpu_offload_vs_cuda_benchmark"
            f"{output_suffix}_{group_column}{group_value}.png"
        )
        group_title_suffix = format_group_title_suffix(
            title_suffix, group_column, group_value
        )
        plot_results(group_df, group_plot_path, title_suffix=group_title_suffix)
        print()
        print(f"Saved plot to {group_plot_path}")


def plot_grouped_cpu_thread_results(
    df: pd.DataFrame,
    output_suffix: str,
    title_suffix: str,
    group_column: str | None,
) -> None:
    if group_column is None:
        plot_path = f"svd_cpu_offload_thread_scaling_benchmark{output_suffix}.png"
        plot_cpu_thread_results(df, plot_path, title_suffix=title_suffix)
        print()
        print(f"Saved plot to {plot_path}")
        return

    for group_value in sorted(df[group_column].dropna().unique()):
        group_df = df[df[group_column] == group_value]
        group_plot_path = (
            f"svd_cpu_offload_thread_scaling_benchmark{output_suffix}_"
            f"{group_column}{group_value}.png"
        )
        group_title_suffix = format_group_title_suffix(
            title_suffix, group_column, group_value
        )
        plot_cpu_thread_results(
            group_df, group_plot_path, title_suffix=group_title_suffix
        )
        print()
        print(f"Saved plot to {group_plot_path}")


def plot_grouped_successive_chunk_cpu_vs_gpu_results(
    df: pd.DataFrame,
    output_suffix: str,
    title_suffix: str,
    group_column: str | None,
) -> None:
    if group_column is None:
        plot_path = f"svd_cpu_vs_gpu_successive_chunk_benchmark{output_suffix}.png"
        plot_successive_chunk_cpu_vs_gpu_results(
            df, plot_path, title_suffix=title_suffix
        )
        print()
        print(f"Saved plot to {plot_path}")
        return

    for group_value in sorted(df[group_column].dropna().unique()):
        group_df = df[df[group_column] == group_value]
        group_plot_path = (
            f"svd_cpu_vs_gpu_successive_chunk_benchmark{output_suffix}_"
            f"{group_column}{group_value}.png"
        )
        group_title_suffix = format_group_title_suffix(
            title_suffix, group_column, group_value
        )
        plot_successive_chunk_cpu_vs_gpu_results(
            group_df, group_plot_path, title_suffix=group_title_suffix
        )
        print()
        print(f"Saved plot to {group_plot_path}")


def run_successive_chunk_benchmark_config(config: dict) -> None:
    output_suffix = config["output_suffix"]
    title_suffix = config["plot_title_suffix"]
    shape_specs = config["shape_specs"]
    split_plots_by = config.get("split_plots_by")

    print(f"=== {config['description']} ===")
    print(
        f"Using per-chunk decomposition with chunk size {XKV_SUCCESSIVE_CHUNK_SIZE}."
    )
    df = build_successive_chunk_cpu_vs_gpu_results(
        shape_specs, desc=f"{config['name']} cpu workers vs cuda"
    )
    print(df.round(3).to_string(index=False))
    plot_grouped_successive_chunk_cpu_vs_gpu_results(
        df,
        output_suffix=output_suffix,
        title_suffix=title_suffix,
        group_column=split_plots_by,
    )
    print()


def run_benchmark_config(config: dict) -> None:
    if config.get("benchmark_type") == "successive_chunks":
        run_successive_chunk_benchmark_config(config)
        return

    output_suffix = config["output_suffix"]
    title_suffix = config["plot_title_suffix"]
    shape_specs = config["shape_specs"]
    split_plots_by = config.get("split_plots_by")

    print(f"=== {config['description']} ===")

    df = build_results(shape_specs, desc=f"{config['name']} cpu-offload vs cuda")
    print(df.round(3).to_string(index=False))
    plot_grouped_results(
        df,
        output_suffix=output_suffix,
        title_suffix=title_suffix,
        group_column=split_plots_by,
    )

    print()
    print("Running CPU-offload thread scaling benchmark...")
    cpu_thread_df = build_cpu_thread_results(
        shape_specs, desc=f"{config['name']} cpu thread scaling"
    )
    print(cpu_thread_df.round(3).to_string(index=False))
    plot_grouped_cpu_thread_results(
        cpu_thread_df,
        output_suffix=output_suffix,
        title_suffix=title_suffix,
        group_column=split_plots_by,
    )

    print()
    print("Running CPU-offload outer parallel benchmark...")
    cpu_outer_df = build_cpu_outer_parallel_results(
        shape_specs, desc=f"{config['name']} cpu outer parallel"
    )
    if cpu_outer_df.empty:
        print("No batched shapes for outer parallel benchmark; skipping.")
        print()
        return

    print(cpu_outer_df.round(3).to_string(index=False))
    cpu_outer_plot_path = (
        f"svd_cpu_offload_outer_parallel_benchmark{output_suffix}.png"
    )
    plot_cpu_outer_parallel_results(
        cpu_outer_df, cpu_outer_plot_path, title_suffix=title_suffix
    )
    print()
    print(f"Saved plot to {cpu_outer_plot_path}")
    print()


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

    for config in BENCHMARK_CONFIGS:
        run_benchmark_config(config)


if __name__ == "__main__":
    main()
