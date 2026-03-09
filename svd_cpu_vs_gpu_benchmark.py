import math
import statistics
import time

import pandas as pd
import torch


HEAD_DIM = 128
NUM_HEADS = 8
BATCH = 4
DTYPE = torch.float32
WARMUP = 3
REPEATS = 10

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
]


def sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_svd(
    shape,
    device,
    dtype=torch.float32,
    warmup=3,
    repeats=10,
):
    device = torch.device(device)
    x = torch.randn(shape, device=device, dtype=dtype)

    sync_if_needed(device)
    for _ in range(warmup):
        torch.linalg.svd(x, full_matrices=False)
    sync_if_needed(device)

    times_ms = []
    for _ in range(repeats):
        sync_if_needed(device)
        start = time.perf_counter()
        torch.linalg.svd(x, full_matrices=False)
        sync_if_needed(device)
        end = time.perf_counter()
        times_ms.append((end - start) * 1000.0)

    return {
        "mean_ms": statistics.mean(times_ms),
        "std_ms": statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
    }


def build_results():
    records = []

    for spec in SHAPE_SPECS:
        row = {
            "label": spec["label"],
            "shape": str(spec["shape"]),
            "dtype": str(DTYPE).replace("torch.", ""),
        }

        cpu_stats = benchmark_svd(
            spec["shape"],
            device="cpu",
            dtype=DTYPE,
            warmup=WARMUP,
            repeats=REPEATS,
        )
        row.update({f"cpu_{k}": v for k, v in cpu_stats.items()})

        if torch.cuda.is_available():
            gpu_stats = benchmark_svd(
                spec["shape"],
                device="cuda",
                dtype=DTYPE,
                warmup=WARMUP,
                repeats=REPEATS,
            )
            row.update({f"gpu_{k}": v for k, v in gpu_stats.items()})
            row["gpu_speedup_vs_cpu"] = row["cpu_mean_ms"] / row["gpu_mean_ms"]
        else:
            row["gpu_mean_ms"] = math.nan
            row["gpu_std_ms"] = math.nan
            row["gpu_min_ms"] = math.nan
            row["gpu_max_ms"] = math.nan
            row["gpu_speedup_vs_cpu"] = math.nan

        records.append(row)

    return pd.DataFrame(records)


def main():
    print(f"torch: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    print(f"cpu threads: {torch.get_num_threads()}")
    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}")
    print(
        f"benchmark config: dtype={DTYPE}, warmup={WARMUP}, repeats={REPEATS}"
    )
    print()

    df = build_results()
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
