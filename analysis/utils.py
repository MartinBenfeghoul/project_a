from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch

try:
    import pandas as pd
except ImportError:
    pd = None

from utils.matrix_decomposition import (
    DECOMP_METHODS,
    decompose_grouped_xkv_to_segment_store,
    decompose_to_segment_store,
    find_rank_wrt_cr,
    reconstruct_segments,
)
from utils.rope import compute_rope_cos_sin, inverse_rope
from utils.segmentation import (
    build_cluster_segment_ranges,
    group_keys_by_cluster,
    group_sequences_by_cluster,
    kmeans_cluster_sequences,
)

HIDDEN_COLUMNS = (
    "batch_idx",
    "seq_len",
    "compressed_len",
    "spectral_tail_curves",
    "spectral_tail_raw_curves",
)
SEQ_LENGTH_BUCKETS = [500, 1000, 2000, 4000, 8000, 10000]
SEQ_LENGTH_LABELS = {
    500: "500",
    1000: "1k",
    2000: "2k",
    4000: "4k",
    8000: "8k",
    10000: "10k",
}
PLOTS = ["heatmap", "needle"]
CLUSTERING_METRIC_PLOTS = (
    ("relative_low_rank_recon_error_bound", "bound"),
    ("relative_low_rank_recon_error", "error"),
)
CLUSTERING_METRIC_COLORS = {
    "relative_low_rank_recon_error_bound": "tab:blue",
    "relative_low_rank_recon_error": "tab:orange",
}
CLUSTERING_DIFFERENCE_COLORS = {
    "Clustered": "tab:green",
    "Random": "tab:purple",
    "Unclustered": "tab:red",
}
CLUSTERING_SET_STYLES = {
    "Clustered": {"linestyle": "-", "marker": "o"},
    "Random": {"linestyle": "-.", "marker": "^"},
    "Unclustered": {"linestyle": "--", "marker": "s"},
}
CLUSTERING_PLOT_FIGSIZE = (10, 6)


def load_results(results_file: str | Path):
    if pd is None:
        raise ImportError("pandas is required to load analysis results.")

    import json

    results = []
    with open(results_file, "r") as handle:
        for line in handle:
            if line.strip():
                results.append(json.loads(line))
    return pd.DataFrame(results)


def snap_to_bucket(n: int) -> int:
    return min(SEQ_LENGTH_BUCKETS, key=lambda bucket: abs(bucket - n))


def plot_energy_at_rank_k(energy: torch.Tensor, k: int) -> None:
    import matplotlib.pyplot as plt

    energy = energy.clone()
    if energy.dim() > 3:
        energy = energy.mean(0)
    energy = energy[..., k]

    plt.imshow(energy.cpu().numpy())
    plt.colorbar()
    plt.xlabel("Heads")
    plt.ylabel("Layers")
    plt.show()


def plot_energy_at_ranks(
    energy: torch.Tensor,
    ranks: list[int],
    comp_ratios: list[float],
    energy_threshold: float | None = None,
    ncols: int | None = None,
    figsize_per_plot: tuple[int, int] = (3, 6),
    cmap: str = "viridis",
) -> None:
    import math
    import matplotlib.pyplot as plt

    energy = energy.clone()
    if energy.dim() > 3:
        energy = energy.mean(0)

    if energy_threshold is not None:
        energy = energy.masked_fill(energy < energy_threshold, 0.0)

    n = len(ranks)
    if ncols is None:
        ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)

    stacked = torch.stack([energy[..., rank] for rank in ranks])
    vmin = stacked.min().item()
    vmax = stacked.max().item()

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows),
        squeeze=False,
    )

    ims = []
    for ax, (idx, rank) in zip(axes.flat, enumerate(ranks)):
        im = ax.imshow(
            energy[..., rank].cpu().numpy(),
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            aspect="auto",
        )
        ims.append(im)
        ax.set_title(f"Rank {rank}, CR {comp_ratios[idx]:.1f}")
        ax.set_xlabel("Heads")
        ax.set_ylabel("Layers")

    for ax in axes.flat[n:]:
        ax.axis("off")

    cax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.subplots_adjust(wspace=0.25, hspace=0.35)

    cbar = fig.colorbar(ims[0], cax=cax)
    cbar.set_label("Energy")
    plt.show()


def get_unique_save_path(save_path: str) -> str:
    import os

    if not os.path.exists(save_path.format("")):
        return save_path.format("")
    for idx in range(100):
        new_path = save_path.format(f"_{idx}")
        if not os.path.exists(new_path):
            return new_path
    raise ValueError(
        f"There appears to be at least 100 numbered variations of "
        f"{save_path}!"
    )


def plot_success_matrix(
    success_matrix,
    seq_lens,
    x_key,
    x_values,
    cache_type,
    save_path: str = "NIAH_ablations{}.png",
    crs=None,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    if crs is None:
        annot = True
        fmt = ".2f"
    else:
        annot = np.empty_like(success_matrix, dtype=object)
        for row_idx in range(success_matrix.shape[0]):
            for col_idx in range(success_matrix.shape[1]):
                annot[row_idx, col_idx] = (
                    f"{success_matrix[row_idx, col_idx]:.2f}\n"
                    f"({crs[row_idx, col_idx]:.2f})"
                )
        fmt = ""

    fig, ax = plt.subplots()
    sns.heatmap(
        success_matrix,
        annot=annot,
        fmt=fmt,
        cmap="YlGn",
        xticklabels=x_values,
        yticklabels=seq_lens,
    )

    ax.set_xlabel(x_key)
    ax.set_ylabel("Sequence Length")
    ax.set_title(f"Ablating {cache_type}")

    fig.tight_layout()
    fig.savefig(get_unique_save_path(save_path), dpi=300)


def plot_needle_results(
    path: str = "./results/needle_mse_long.jsonl",
    save_path: str = "nll_heatmap_needle_mse.png",
    columns: str = "num_token_per_training",
    values: str = "avg_accuracy_modified_cache",
    index: str = "percentage_changed_kv",
) -> None:
    if pd is None:
        raise ImportError("pandas is required to plot needle results.")

    import json
    import matplotlib.pyplot as plt
    import seaborn as sns

    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record["percentage_changed_kv"] = round(
                record["percentage_changed_kv"]
            )
            record["avg_accuracy_modified_cache"] = (
                record["avg_accuracy_modified_cache"] * 100
            )
            rows.append(record)

    if values == "avg_accuracy_modified_cache":
        vmin, vmax = 70, 98
    else:
        vmin, vmax = 2.70, 3

    df = pd.DataFrame(rows)
    heatmap_df = df.pivot(index=index, columns=columns, values=values)

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        vmin=vmin,
        vmax=vmax,
    )
    plt.title("NLL with updated KV cache", fontsize=16)
    plt.xlabel("Sequence length")
    plt.ylabel("Percentage KV changed")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_heatmap(df, output_path: str | Path, title: str | None = None) -> None:
    if pd is None:
        raise ImportError("pandas is required to plot heatmaps.")

    import matplotlib.pyplot as plt
    import seaborn as sns

    df = df.copy()
    df["seq_length"] = df["num_token"].apply(snap_to_bucket)

    pivot = df.pivot(
        index="avg_nll_change_perc",
        columns="seq_length",
        values="avg_accuracy_modified_cache",
    )
    pivot = pivot.sort_index(ascending=True)
    pivot = pivot[sorted(pivot.columns)]
    pivot.columns = [SEQ_LENGTH_LABELS[col] for col in pivot.columns]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="YlGn",
        vmin=pivot.min().min(),
        vmax=pivot.max().max(),
        cbar_kws={"label": "Accuracy"},
        ax=ax,
    )

    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Percentage of KV Changed")
    if title:
        ax.set_title(title)
    else:
        num_epochs = df.iloc[0].get("num_epoch", "?")
        ax.set_title(f"{num_epochs} Test-Time Training Epochs")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")
    print("\nSummary Table:")
    print(pivot)


def unrope_dumped_keys(keys: torch.Tensor, rope_theta: float) -> torch.Tensor:
    seq_len = keys.shape[-2]
    head_dim = keys.shape[-1]
    cos, sin = compute_rope_cos_sin(
        seq_len,
        head_dim,
        rope_theta,
        keys.device,
        keys.dtype,
    )
    return inverse_rope(keys, cos, sin)


def load_kv_dump(
    kv_dump_path: str | Path,
    *,
    unrope_keys_on_load: bool = True,
    default_rope_theta: float = 500_000.0,
) -> dict[str, Any]:
    kv_dump_path = Path(kv_dump_path)
    kvs = torch.load(kv_dump_path)
    raw_keys = kvs["keys"]
    rope_theta = kvs.get("rope_theta")
    if rope_theta is None and kvs.get("model_name"):
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(
            kvs["model_name"],
            local_files_only=True,
        )
        rope_theta = getattr(config, "rope_theta", None)
        if rope_theta is None:
            rope_parameters = getattr(config, "rope_parameters", None) or {}
            rope_theta = rope_parameters.get("rope_theta")
    rope_theta = float(default_rope_theta if rope_theta is None else rope_theta)
    keys = (
        unrope_dumped_keys(raw_keys, rope_theta)
        if unrope_keys_on_load
        else raw_keys
    )
    values = kvs["values"]
    prompt_len = int(kvs.get("prompt_len", keys.shape[-2]))
    return {
        "path": kv_dump_path,
        "kvs": kvs,
        "raw_keys": raw_keys,
        "keys": keys,
        "values": values,
        "prompt_len": prompt_len,
        "rope_theta": rope_theta,
    }


def _ensure_layer_batched_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 4:
        return tensor.unsqueeze(1)
    if tensor.dim() == 5:
        return tensor
    raise ValueError(
        "Expected tensor with shape [layers, heads, seq, dim] or "
        "[layers, batch, heads, seq, dim]."
    )


def _resolve_kmeans_dtype(kmeans_dtype: torch.dtype | str) -> torch.dtype:
    if isinstance(kmeans_dtype, str):
        try:
            kmeans_dtype = getattr(torch, kmeans_dtype)
        except AttributeError as exc:
            raise ValueError(f"Unknown kmeans dtype: {kmeans_dtype}") from exc
    if not isinstance(kmeans_dtype, torch.dtype):
        raise TypeError("kmeans_dtype must be a torch.dtype or dtype name.")
    return kmeans_dtype


def _get_cluster_count(
    seq_len: int,
    n_clusters: int,
    kmeans_cluster_size: int | None = None,
) -> int:
    if seq_len == 0:
        return 0
    if kmeans_cluster_size is not None:
        cluster_count = int(round(seq_len / kmeans_cluster_size))
    else:
        cluster_count = n_clusters
    return max(1, min(cluster_count, seq_len))


def _select_prefix(
    tensor: torch.Tensor,
    prefix_end: int | None = None,
    local_window: int = 0,
) -> tuple[torch.Tensor, int]:
    seq_len = tensor.size(-2)
    if prefix_end is None:
        prefix_end = seq_len
    if prefix_end < 0 or prefix_end > seq_len:
        raise ValueError(
            "prefix_end must be between 0 and the sequence length."
        )
    suffix_start = max(0, prefix_end - local_window)
    return tensor[..., :suffix_start, :], suffix_start


def _validate_cluster_axis(cluster_axis: str) -> None:
    if cluster_axis not in {"rows", "cols"}:
        raise ValueError("cluster_axis must be 'rows' or 'cols'.")


def _group_features_and_ranges(
    features: torch.Tensor,
    assignments: torch.Tensor,
    n_clusters: int,
) -> tuple[torch.Tensor, list[list[tuple[int, int]]]]:
    grouped_features, _, _ = group_sequences_by_cluster(features, assignments)
    segment_ranges = build_cluster_segment_ranges(
        assignments,
        n_clusters=n_clusters,
    )
    return grouped_features, segment_ranges


def _random_cluster_assignments(
    batch_size: int,
    item_count: int,
    cluster_count: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    if item_count == 0:
        return torch.empty(batch_size, 0, dtype=torch.long, device=device)

    labels = torch.arange(item_count, device=device) % cluster_count
    return torch.stack(
        [
            labels[torch.randperm(item_count, device=device)]
            for _ in range(batch_size)
        ]
    )


def _cluster_items_and_ranges(
    features: torch.Tensor,
    n_clusters: int,
    kmeans_n_iter: int,
    kmeans_init: str,
    kmeans_dtype: torch.dtype,
    *,
    kmeans_cluster_size: int | None = None,
    cluster_axis: str = "rows",
    random_assignments: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[list[tuple[int, int]]],
    int,
]:
    _validate_cluster_axis(cluster_axis)
    item_matrix = (
        features
        if cluster_axis == "rows"
        else features.transpose(1, 2).contiguous()
    )
    cluster_count = _get_cluster_count(
        item_matrix.size(1),
        n_clusters,
        kmeans_cluster_size=kmeans_cluster_size,
    )
    if random_assignments:
        assignments = _random_cluster_assignments(
            item_matrix.size(0),
            item_matrix.size(1),
            cluster_count,
            device=item_matrix.device,
        )
    else:
        assignments = kmeans_cluster_sequences(
            item_matrix,
            n_clusters=cluster_count,
            n_iter=max(1, kmeans_n_iter),
            kmeans_init=kmeans_init,
            dtype=kmeans_dtype,
        )
    grouped_items, segment_ranges = _group_features_and_ranges(
        item_matrix,
        assignments,
        cluster_count,
    )
    return (
        item_matrix,
        grouped_items,
        assignments,
        segment_ranges,
        cluster_count,
    )


def _scatter_from_grouped(
    grouped_features: torch.Tensor,
    segment_ranges: list[list[tuple[int, int]]],
) -> list[dict[str, float]]:
    metrics = []
    for batch_idx, batch_ranges in enumerate(segment_ranges):
        batch_features = grouped_features[batch_idx]
        frob_sq = batch_features.pow(2).sum()
        scatter = batch_features.new_zeros(())
        for start_idx, end_idx in batch_ranges:
            cluster = batch_features[start_idx:end_idx]
            centroid = cluster.mean(dim=0, keepdim=True)
            scatter = scatter + (cluster - centroid).pow(2).sum()

        denom = float(frob_sq.item())
        scatter_value = float(scatter.item())
        metrics.append(
            {
                "J_kmeans": scatter_value,
                "eta": scatter_value / denom if denom > 0 else 0.0,
            }
        )
    return metrics


def _group_tensor_last_dim_by_cluster(
    tensor: torch.Tensor,
    assignments: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if assignments.shape != (tensor.size(0), tensor.size(-1)):
        raise ValueError(
            f"Expected assignments shape {(tensor.size(0), tensor.size(-1))}, "
            f"got {tuple(assignments.shape)}."
        )
    permutation = torch.argsort(assignments, dim=-1)
    inverse_permutation = torch.empty_like(permutation)
    original_positions = (
        torch.arange(
            permutation.size(-1),
            device=permutation.device,
            dtype=permutation.dtype,
        )
        .unsqueeze(0)
        .expand_as(permutation)
    )
    inverse_permutation.scatter_(1, permutation, original_positions)
    gather_shape = (
        (tensor.size(0),) + (1,) * (tensor.dim() - 2) + (tensor.size(-1),)
    )
    gather_idx = permutation.view(gather_shape).expand_as(tensor)
    grouped_tensor = torch.gather(
        tensor,
        dim=tensor.dim() - 1,
        index=gather_idx,
    )
    return grouped_tensor, permutation, inverse_permutation


def _restore_tensor_last_dim(
    grouped_tensor: torch.Tensor,
    inverse_permutation: torch.Tensor,
) -> torch.Tensor:
    gather_shape = (
        (grouped_tensor.size(0),)
        + (1,) * (grouped_tensor.dim() - 2)
        + (grouped_tensor.size(-1),)
    )
    gather_idx = inverse_permutation.view(gather_shape).expand_as(
        grouped_tensor
    )
    return torch.gather(
        grouped_tensor,
        dim=grouped_tensor.dim() - 1,
        index=gather_idx,
    )


def _get_decomposition(
    decomposition_method: str = "svd",
    rank_selection: str = "comp_ratio",
    comp_ratio: float = 2.0,
    energy_threshold: float = 0.95,
    decomp_n_iter: int = 3,
    decomp_lr: float = 1e-2,
) -> tuple[Any, dict[str, Any]]:
    if decomposition_method not in DECOMP_METHODS:
        raise ValueError(
            f"Unknown decomposition_method: {decomposition_method}. "
            f"Available methods: {sorted(DECOMP_METHODS)}"
        )
    return DECOMP_METHODS[decomposition_method], {
        "rank_selection": rank_selection,
        "cr": comp_ratio,
        "energy_threshold": energy_threshold,
        "n_iter": decomp_n_iter,
        "lr": decomp_lr,
        "quantise_a": False,
        "quantise_b": False,
        "compressor_bits": 4,
    }


def _low_rank_recon_metrics(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
) -> list[dict[str, float]]:
    if original.shape != reconstructed.shape:
        raise ValueError(
            f"Shape mismatch: {original.shape} vs {reconstructed.shape}"
        )
    orig_flat = original.reshape(original.size(0), -1)
    recon_flat = reconstructed.reshape(reconstructed.size(0), -1)
    numer = (orig_flat - recon_flat).pow(2).sum(dim=-1).sqrt()
    denom = orig_flat.pow(2).sum(dim=-1).sqrt()
    metrics = []
    for idx in range(original.size(0)):
        denom_value = float(denom[idx].item())
        error_value = float(numer[idx].item())
        metrics.append(
            {
                "low_rank_recon_error": error_value,
                "relative_low_rank_recon_error": (
                    error_value / denom_value if denom_value > 0 else 0.0
                ),
            }
        )
    return metrics


def _spectral_tail_metrics(
    grouped_tensor: torch.Tensor,
    segment_ranges: list[list[tuple[int, int]]],
    *,
    cluster_axis: str,
    num_points: int = 101,
) -> list[dict[str, Any]]:
    _validate_cluster_axis(cluster_axis)
    tensor_to_decompose = (
        grouped_tensor
        if cluster_axis == "rows"
        else grouped_tensor.transpose(-2, -1).contiguous()
    )
    metrics = []
    for batch_idx, batch_ranges in enumerate(segment_ranges):
        curves = []
        raw_curves = []
        for start_idx, end_idx in batch_ranges:
            segment = tensor_to_decompose[batch_idx, ..., start_idx:end_idx, :]
            singular_values = torch.linalg.svdvals(
                segment.to(dtype=torch.float32).contiguous()
            ).reshape(-1, min(segment.size(-2), segment.size(-1)))
            for spectrum in singular_values:
                energy = spectrum.square()
                total = energy.sum()
                if total <= 0:
                    curves.append([0.0] * num_points)
                    raw_curves.append([0.0] * (energy.numel() + 1))
                    continue
                tail = torch.cat(
                    (energy.flip(0).cumsum(0).flip(0), energy.new_zeros(1))
                )
                tail = 100 * tail / total
                raw_curves.append(tail.cpu().tolist())
                curve = torch.nn.functional.interpolate(
                    tail.view(1, 1, -1),
                    size=num_points,
                    mode="linear",
                    align_corners=True,
                ).view(-1)
                curves.append(curve.cpu().tolist())
        metrics.append(
            {
                "spectral_tail_curves": curves,
                "spectral_tail_raw_curves": raw_curves,
            }
        )
    return metrics


def _merge_metric_lists(
    *metric_lists: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            key: value
            for metric_dict in metric_dicts
            for key, value in metric_dict.items()
        }
        for metric_dicts in zip(*metric_lists)
    ]


def _reconstruct_lr_segments(
    grouped_tensor: torch.Tensor,
    segment_ranges: list[list[tuple[int, int]]],
    decompose_fn: Any,
    decomp_kwargs: dict[str, Any],
    *,
    cluster_axis: str = "rows",
) -> torch.Tensor:
    _validate_cluster_axis(cluster_axis)
    tensor_to_decompose = (
        grouped_tensor
        if cluster_axis == "rows"
        else grouped_tensor.transpose(-2, -1).contiguous()
    )
    layer_segments = decompose_to_segment_store(
        tensor_to_decompose,
        decompose_fn,
        segment_ranges=segment_ranges,
        **decomp_kwargs,
    )
    reconstructed = reconstruct_segments(
        layer_segments,
        tensor_to_decompose[..., :0, :],
    )
    return (
        reconstructed
        if cluster_axis == "rows"
        else reconstructed.transpose(-2, -1).contiguous()
    )


def _reconstruct_xkv_segments(
    grouped_tensor: torch.Tensor,
    segment_ranges: list[list[tuple[int, int]]],
    decomp_kwargs: dict[str, Any],
    *,
    cluster_axis: str = "rows",
) -> torch.Tensor:
    _validate_cluster_axis(cluster_axis)
    tensor_to_decompose = (
        grouped_tensor
        if cluster_axis == "rows"
        else grouped_tensor.transpose(-2, -1).contiguous()
    )
    grouped_segments = decompose_grouped_xkv_to_segment_store(
        tensor_to_decompose.unsqueeze(1),
        segment_ranges=segment_ranges,
        **decomp_kwargs,
    )
    reconstructed = reconstruct_segments(
        grouped_segments,
        tensor_to_decompose.unsqueeze(1)[..., :0, :],
    ).squeeze(1)
    return (
        reconstructed
        if cluster_axis == "rows"
        else reconstructed.transpose(-2, -1).contiguous()
    )


def analyze_kmeans_lrk(
    tensor: torch.Tensor,
    *,
    n_clusters: int = 8,
    kmeans_cluster_size: int | None = None,
    kmeans_n_iter: int = 8,
    kmeans_init: str = "infllm",
    kmeans_dtype: torch.dtype | str = torch.float32,
    kmeans_mode: str = "per_head",
    prefix_end: int | None = None,
    local_window: int = 0,
    include_head_breakdown: bool = False,
    decomposition_method: str = "svd",
    rank_selection: str = "comp_ratio",
    comp_ratio: float = 2.0,
    energy_threshold: float = 0.95,
    decomp_n_iter: int = 3,
    decomp_lr: float = 1e-2,
    cluster_axis: str = "rows",
    random_assignments: bool = False,
) -> list[dict[str, Any]]:
    tensor = _ensure_layer_batched_tensor(tensor)
    kmeans_dtype = _resolve_kmeans_dtype(kmeans_dtype)
    _validate_cluster_axis(cluster_axis)

    if kmeans_mode not in {"concat_heads", "avg_heads", "per_head"}:
        raise ValueError(
            "kmeans_mode must be one of 'concat_heads', 'avg_heads', or "
            "'per_head'."
        )

    decompose_fn, decomp_kwargs = _get_decomposition(
        decomposition_method=decomposition_method,
        rank_selection=rank_selection,
        comp_ratio=comp_ratio,
        energy_threshold=energy_threshold,
        decomp_n_iter=decomp_n_iter,
        decomp_lr=decomp_lr,
    )

    results = []
    for layer_idx in range(tensor.size(0)):
        prefix_tensor, compressed_len = _select_prefix(
            tensor[layer_idx],
            prefix_end=prefix_end,
            local_window=local_window,
        )
        batch_size, num_heads, seq_len, head_dim = prefix_tensor.shape

        if kmeans_mode == "per_head":
            base_tensor = prefix_tensor.reshape(
                batch_size * num_heads,
                seq_len,
                head_dim,
            )
            (
                _,
                grouped_items,
                assignments,
                segment_ranges,
                cluster_count,
            ) = _cluster_items_and_ranges(
                base_tensor,
                n_clusters,
                kmeans_n_iter,
                kmeans_init,
                kmeans_dtype,
                kmeans_cluster_size=kmeans_cluster_size,
                cluster_axis=cluster_axis,
                random_assignments=random_assignments,
            )
            if cluster_axis == "rows":
                grouped_target = grouped_items
            else:
                grouped_target, _, _ = _group_tensor_last_dim_by_cluster(
                    base_tensor,
                    assignments,
                )
            scatter_metrics = _scatter_from_grouped(
                grouped_items,
                segment_ranges,
            )
            reconstructed = _reconstruct_lr_segments(
                grouped_target,
                segment_ranges,
                decompose_fn,
                decomp_kwargs,
                cluster_axis=cluster_axis,
            )
            recon_metrics = _low_rank_recon_metrics(
                grouped_target,
                reconstructed,
            )
            spectral_metrics = _spectral_tail_metrics(
                grouped_target,
                segment_ranges,
                cluster_axis=cluster_axis,
            )
            metrics = _merge_metric_lists(
                scatter_metrics,
                recon_metrics,
                spectral_metrics,
            )
            for flat_idx, metric in enumerate(metrics):
                batch_idx = flat_idx // num_heads
                head_idx = flat_idx % num_heads
                results.append(
                    {
                        "cache_type": "kmeans_lr",
                        "cluster_axis": cluster_axis,
                        "kmeans_mode": kmeans_mode,
                        "metric_scope": "head",
                        "layer_idx": layer_idx,
                        "batch_idx": batch_idx,
                        "head_idx": head_idx,
                        "seq_len": seq_len,
                        "compressed_len": compressed_len,
                        "cluster_count": cluster_count,
                        **metric,
                    }
                )
            continue

        if kmeans_mode == "avg_heads":
            token_features = prefix_tensor.mean(dim=1)
        else:
            token_features = prefix_tensor.transpose(1, 2).reshape(
                batch_size,
                seq_len,
                -1,
            )

        (
            _,
            grouped_items,
            assignments,
            segment_ranges,
            cluster_count,
        ) = _cluster_items_and_ranges(
            token_features,
            n_clusters,
            kmeans_n_iter,
            kmeans_init,
            kmeans_dtype,
            kmeans_cluster_size=kmeans_cluster_size,
            cluster_axis=cluster_axis,
            random_assignments=random_assignments,
        )

        if cluster_axis == "rows":
            grouped_target, _, _ = group_keys_by_cluster(
                prefix_tensor,
                assignments,
            )
            scatter_metrics = _scatter_from_grouped(
                grouped_items,
                segment_ranges,
            )
            reconstructed = _reconstruct_lr_segments(
                grouped_target,
                segment_ranges,
                decompose_fn,
                decomp_kwargs,
                cluster_axis=cluster_axis,
            )
            layer_recon_metrics = _low_rank_recon_metrics(
                grouped_target,
                reconstructed,
            )
            layer_spectral_metrics = _spectral_tail_metrics(
                grouped_target,
                segment_ranges,
                cluster_axis=cluster_axis,
            )
            layer_metrics = _merge_metric_lists(
                scatter_metrics,
                layer_recon_metrics,
                layer_spectral_metrics,
            )
            for batch_idx, metric in enumerate(layer_metrics):
                results.append(
                    {
                        "cache_type": "kmeans_lr",
                        "cluster_axis": cluster_axis,
                        "kmeans_mode": kmeans_mode,
                        "metric_scope": "layer",
                        "layer_idx": layer_idx,
                        "batch_idx": batch_idx,
                        "head_idx": None,
                        "seq_len": seq_len,
                        "compressed_len": compressed_len,
                        "cluster_count": cluster_count,
                        **metric,
                    }
                )

            if include_head_breakdown:
                for head_idx in range(num_heads):
                    head_scatter_metrics = _scatter_from_grouped(
                        grouped_target[:, head_idx],
                        segment_ranges,
                    )
                    head_recon_metrics = _low_rank_recon_metrics(
                        grouped_target[:, head_idx],
                        reconstructed[:, head_idx],
                    )
                    head_metrics = _merge_metric_lists(
                        head_scatter_metrics,
                        head_recon_metrics,
                    )
                    for batch_idx, metric in enumerate(head_metrics):
                        results.append(
                            {
                                "cache_type": "kmeans_lr",
                                "cluster_axis": cluster_axis,
                                "kmeans_mode": kmeans_mode,
                                "metric_scope": "head",
                                "layer_idx": layer_idx,
                                "batch_idx": batch_idx,
                                "head_idx": head_idx,
                                "seq_len": seq_len,
                                "compressed_len": compressed_len,
                                "cluster_count": cluster_count,
                                **metric,
                            }
                        )
            continue

        grouped_target, _, inverse_permutation = (
            _group_tensor_last_dim_by_cluster(
                token_features,
                assignments,
            )
        )
        scatter_metrics = _scatter_from_grouped(
            grouped_items,
            segment_ranges,
        )
        reconstructed = _reconstruct_lr_segments(
            grouped_target,
            segment_ranges,
            decompose_fn,
            decomp_kwargs,
            cluster_axis=cluster_axis,
        )
        restored_reconstructed = _restore_tensor_last_dim(
            reconstructed,
            inverse_permutation,
        )
        layer_recon_metrics = _low_rank_recon_metrics(
            grouped_target,
            reconstructed,
        )
        layer_spectral_metrics = _spectral_tail_metrics(
            grouped_target,
            segment_ranges,
            cluster_axis=cluster_axis,
        )
        layer_metrics = _merge_metric_lists(
            scatter_metrics,
            layer_recon_metrics,
            layer_spectral_metrics,
        )
        for batch_idx, metric in enumerate(layer_metrics):
            results.append(
                {
                    "cache_type": "kmeans_lr",
                    "cluster_axis": cluster_axis,
                    "kmeans_mode": kmeans_mode,
                    "metric_scope": "layer",
                    "layer_idx": layer_idx,
                    "batch_idx": batch_idx,
                    "head_idx": None,
                    "seq_len": seq_len,
                    "compressed_len": compressed_len,
                    "cluster_count": cluster_count,
                    **metric,
                }
            )

        if include_head_breakdown:
            for head_idx in range(num_heads):
                if kmeans_mode == "avg_heads":
                    head_target = prefix_tensor[:, head_idx]
                    grouped_head_target, _, _ = (
                        _group_tensor_last_dim_by_cluster(
                            head_target,
                            assignments,
                        )
                    )
                    grouped_head_items, head_segment_ranges = (
                        _group_features_and_ranges(
                            head_target.transpose(1, 2).contiguous(),
                            assignments,
                            cluster_count,
                        )
                    )
                    head_reconstructed = _reconstruct_lr_segments(
                        grouped_head_target,
                        head_segment_ranges,
                        decompose_fn,
                        decomp_kwargs,
                        cluster_axis=cluster_axis,
                    )
                    head_recon_metrics = _low_rank_recon_metrics(
                        grouped_head_target,
                        head_reconstructed,
                    )
                    head_scatter_metrics = _scatter_from_grouped(
                        grouped_head_items,
                        head_segment_ranges,
                    )
                else:
                    start_idx = head_idx * head_dim
                    end_idx = start_idx + head_dim
                    head_items, head_segment_ranges = (
                        _group_features_and_ranges(
                            token_features[..., start_idx:end_idx]
                            .transpose(1, 2)
                            .contiguous(),
                            assignments[..., start_idx:end_idx],
                            cluster_count,
                        )
                    )
                    head_scatter_metrics = _scatter_from_grouped(
                        head_items,
                        head_segment_ranges,
                    )
                    head_recon_metrics = _low_rank_recon_metrics(
                        token_features[..., start_idx:end_idx],
                        restored_reconstructed[..., start_idx:end_idx],
                    )

                head_metrics = _merge_metric_lists(
                    head_scatter_metrics,
                    head_recon_metrics,
                )
                for batch_idx, metric in enumerate(head_metrics):
                    results.append(
                        {
                            "cache_type": "kmeans_lr",
                            "cluster_axis": cluster_axis,
                            "kmeans_mode": kmeans_mode,
                            "metric_scope": "head",
                            "layer_idx": layer_idx,
                            "batch_idx": batch_idx,
                            "head_idx": head_idx,
                            "seq_len": seq_len,
                            "compressed_len": compressed_len,
                            "cluster_count": cluster_count,
                            **metric,
                        }
                    )

    return results


def _get_group_bounds(
    layer_idx: int,
    layer_group_size: int,
    num_layers: int | None = None,
) -> tuple[int, int]:
    group_start = (layer_idx // layer_group_size) * layer_group_size
    group_last = group_start + layer_group_size - 1
    if num_layers is not None:
        group_last = min(group_last, num_layers - 1)
    return group_start, group_last


def analyze_kmeans_xkv(
    tensor: torch.Tensor,
    *,
    layer_group_size: int = 2,
    num_layers: int | None = None,
    n_clusters: int = 8,
    kmeans_cluster_size: int | None = None,
    kmeans_n_iter: int = 8,
    kmeans_init: str = "infllm",
    kmeans_dtype: torch.dtype | str = torch.float32,
    prefix_end: int | None = None,
    local_window: int = 0,
    decomposition_method: str = "svd",
    rank_selection: str = "comp_ratio",
    comp_ratio: float = 2.0,
    energy_threshold: float = 0.95,
    decomp_n_iter: int = 3,
    decomp_lr: float = 1e-2,
    cluster_axis: str = "rows",
    random_assignments: bool = False,
) -> list[dict[str, Any]]:
    tensor = _ensure_layer_batched_tensor(tensor)
    kmeans_dtype = _resolve_kmeans_dtype(kmeans_dtype)
    _validate_cluster_axis(cluster_axis)

    if layer_group_size <= 0:
        raise ValueError("layer_group_size must be positive.")
    if num_layers is None:
        num_layers = tensor.size(0)
    if num_layers <= 0 or num_layers > tensor.size(0):
        raise ValueError("num_layers must be in [1, tensor.size(0)].")

    if decomposition_method != "svd":
        raise NotImplementedError(
            "KMeansXKVKeysCache-style grouped reconstruction currently "
            "supports decomposition_method='svd' only."
        )

    _, decomp_kwargs = _get_decomposition(
        decomposition_method=decomposition_method,
        rank_selection=rank_selection,
        comp_ratio=comp_ratio,
        energy_threshold=energy_threshold,
        decomp_n_iter=decomp_n_iter,
        decomp_lr=decomp_lr,
    )

    results = []
    for layer_idx in range(num_layers):
        group_start, group_last = _get_group_bounds(
            layer_idx,
            layer_group_size,
            num_layers,
        )
        if layer_idx != group_last:
            continue

        group_tensors = [tensor[i] for i in range(group_start, group_last + 1)]
        seq_len = group_tensors[-1].size(-2)
        if any(item.size(-2) != seq_len for item in group_tensors[:-1]):
            raise ValueError(
                "All layers in an xKV group must share the same cached length."
            )

        prefix_tensors = []
        split_sizes = []
        compressed_len = None
        for item in group_tensors:
            prefix_tensor, compressed_len = _select_prefix(
                item,
                prefix_end=prefix_end,
                local_window=local_window,
            )
            prefix_flat = prefix_tensor.transpose(1, 2).reshape(
                prefix_tensor.size(0),
                prefix_tensor.size(-2),
                -1,
            )
            prefix_tensors.append(prefix_flat)
            split_sizes.append(prefix_flat.size(-1))

        group_prefix = torch.cat(prefix_tensors, dim=-1)
        (
            _,
            grouped_items,
            assignments,
            segment_ranges,
            cluster_count,
        ) = _cluster_items_and_ranges(
            group_prefix,
            n_clusters,
            kmeans_n_iter,
            kmeans_init,
            kmeans_dtype,
            kmeans_cluster_size=kmeans_cluster_size,
            cluster_axis=cluster_axis,
            random_assignments=random_assignments,
        )

        if cluster_axis == "rows":
            grouped_target = grouped_items
            reconstructed_group = _reconstruct_xkv_segments(
                grouped_target,
                segment_ranges,
                decomp_kwargs,
                cluster_axis=cluster_axis,
            )
            grouped_layer_targets = torch.split(
                grouped_target,
                split_sizes,
                dim=-1,
            )
            reconstructed_layer_targets = torch.split(
                reconstructed_group,
                split_sizes,
                dim=-1,
            )
        else:
            grouped_target, _, inverse_permutation = (
                _group_tensor_last_dim_by_cluster(
                    group_prefix,
                    assignments,
                )
            )
            reconstructed_group = _reconstruct_xkv_segments(
                grouped_target,
                segment_ranges,
                decomp_kwargs,
                cluster_axis=cluster_axis,
            )
            restored_reconstructed = _restore_tensor_last_dim(
                reconstructed_group,
                inverse_permutation,
            )

        group_scatter_metrics = _scatter_from_grouped(
            grouped_items,
            segment_ranges,
        )
        group_recon_metrics = _low_rank_recon_metrics(
            grouped_target,
            reconstructed_group,
        )
        group_spectral_metrics = _spectral_tail_metrics(
            grouped_target,
            segment_ranges,
            cluster_axis=cluster_axis,
        )
        group_metrics = _merge_metric_lists(
            group_scatter_metrics,
            group_recon_metrics,
            group_spectral_metrics,
        )
        for batch_idx, metric in enumerate(group_metrics):
            results.append(
                {
                    "cache_type": "kmeans_xkv",
                    "cluster_axis": cluster_axis,
                    "metric_scope": "group",
                    "layer_idx": group_last,
                    "group_start_layer": group_start,
                    "group_last_layer": group_last,
                    "batch_idx": batch_idx,
                    "head_idx": None,
                    "seq_len": group_prefix.size(1),
                    "compressed_len": compressed_len,
                    "cluster_count": cluster_count,
                    **metric,
                }
            )

        col_offset = 0
        for offset, split_size in enumerate(split_sizes):
            actual_layer_idx = group_start + offset
            if cluster_axis == "rows":
                layer_target = grouped_layer_targets[offset]
                layer_recon = reconstructed_layer_targets[offset]
                layer_scatter_metrics = _scatter_from_grouped(
                    layer_target,
                    segment_ranges,
                )
                layer_recon_metrics = _low_rank_recon_metrics(
                    layer_target,
                    layer_recon,
                )
            else:
                layer_original = group_prefix[
                    ..., col_offset : col_offset + split_size
                ]
                layer_assignments = assignments[
                    ..., col_offset : col_offset + split_size
                ]
                layer_items, layer_segment_ranges = _group_features_and_ranges(
                    layer_original.transpose(1, 2).contiguous(),
                    layer_assignments,
                    cluster_count,
                )
                layer_scatter_metrics = _scatter_from_grouped(
                    layer_items,
                    layer_segment_ranges,
                )
                layer_recon_metrics = _low_rank_recon_metrics(
                    layer_original,
                    restored_reconstructed[
                        ..., col_offset : col_offset + split_size
                    ],
                )
                col_offset += split_size

            layer_metrics = _merge_metric_lists(
                layer_scatter_metrics,
                layer_recon_metrics,
            )
            for batch_idx, metric in enumerate(layer_metrics):
                results.append(
                    {
                        "cache_type": "kmeans_xkv",
                        "cluster_axis": cluster_axis,
                        "metric_scope": "layer",
                        "layer_idx": actual_layer_idx,
                        "group_start_layer": group_start,
                        "group_last_layer": group_last,
                        "batch_idx": batch_idx,
                        "head_idx": None,
                        "seq_len": group_prefix.size(1),
                        "compressed_len": compressed_len,
                        "cluster_count": cluster_count,
                        **metric,
                    }
                )

    return results


def _records_to_frame(
    records: list[dict[str, Any]],
    *,
    hidden_columns: tuple[str, ...] = HIDDEN_COLUMNS,
):
    if pd is None:
        return None
    return pd.DataFrame(records).drop(columns=hidden_columns, errors="ignore")


def _summary_from_records(
    label: str,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    etas = [record["eta"] for record in records]
    relative_errors = [
        record["relative_low_rank_recon_error"] for record in records
    ]
    return {
        "case": label,
        "rows": len(records),
        "mean_eta": sum(etas) / len(etas) if etas else float("nan"),
        "median_eta": _median(etas),
        "mean_relative_low_rank_recon_error": (
            sum(relative_errors) / len(relative_errors)
            if relative_errors
            else float("nan")
        ),
        "median_relative_low_rank_recon_error": _median(relative_errors),
        "min_eta": min(etas) if etas else float("nan"),
        "max_eta": max(etas) if etas else float("nan"),
    }


def _summary_from_frame(label: str, df) -> dict[str, Any]:
    return {
        "case": label,
        "rows": len(df),
        "mean_eta": df["eta"].mean(),
        "median_eta": df["eta"].median(),
        "mean_relative_low_rank_recon_error": df[
            "relative_low_rank_recon_error"
        ].mean(),
        "median_relative_low_rank_recon_error": df[
            "relative_low_rank_recon_error"
        ].median(),
        "min_eta": df["eta"].min(),
        "max_eta": df["eta"].max(),
    }


def _median(values: list[float]) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _build_case_summary_rows(
    tensor_name: str,
    cluster_axis: str,
    lrk_records: list[dict[str, Any]],
    xkv_records: list[dict[str, Any]],
    lrk_records_n1: list[dict[str, Any]],
    xkv_records_n1: list[dict[str, Any]],
    *,
    lrk_random_records: list[dict[str, Any]] | None = None,
    xkv_random_records: list[dict[str, Any]] | None = None,
    lrk_df=None,
    xkv_df=None,
    lrk_df_n1=None,
    xkv_df_n1=None,
    lrk_random_df=None,
    xkv_random_df=None,
) -> list[dict[str, Any]]:
    summary_inputs = [
        (f"{tensor_name}_lrk_{cluster_axis}", lrk_records, lrk_df),
        (f"{tensor_name}_xkv_{cluster_axis}", xkv_records, xkv_df),
        (
            f"{tensor_name}_lrk_{cluster_axis}_random",
            lrk_random_records,
            lrk_random_df,
        ),
        (
            f"{tensor_name}_xkv_{cluster_axis}_random",
            xkv_random_records,
            xkv_random_df,
        ),
        (
            f"{tensor_name}_lrk_{cluster_axis}_n_clusters_1",
            lrk_records_n1,
            lrk_df_n1,
        ),
        (
            f"{tensor_name}_xkv_{cluster_axis}_n_clusters_1",
            xkv_records_n1,
            xkv_df_n1,
        ),
    ]
    summary_rows = []
    for label, records, df in summary_inputs:
        if records is None and df is None:
            continue
        if df is not None:
            summary_rows.append(_summary_from_frame(label, df))
        else:
            summary_rows.append(_summary_from_records(label, records))
    return summary_rows


def run_analysis_case(
    tensor: torch.Tensor,
    *,
    tensor_name: str,
    cluster_axis: str,
    kmeans_cfg: dict[str, Any],
    decomposition_cfg: dict[str, Any],
    lrk_mode: str = "per_head",
    include_lrk_head_breakdown: bool = False,
    prefix_end: int | None = None,
    local_window: int = 0,
    xkv_layer_group_size: int = 4,
    xkv_num_layers: int | None = None,
    hidden_columns: tuple[str, ...] = HIDDEN_COLUMNS,
    include_random_clustering: bool = False,
) -> dict[str, Any]:
    _validate_cluster_axis(cluster_axis)

    lrk_results = analyze_kmeans_lrk(
        tensor,
        kmeans_mode=lrk_mode,
        prefix_end=prefix_end,
        local_window=local_window,
        cluster_axis=cluster_axis,
        include_head_breakdown=include_lrk_head_breakdown,
        **kmeans_cfg,
        **decomposition_cfg,
    )
    xkv_results = analyze_kmeans_xkv(
        tensor,
        layer_group_size=xkv_layer_group_size,
        num_layers=xkv_num_layers,
        prefix_end=prefix_end,
        local_window=local_window,
        cluster_axis=cluster_axis,
        **kmeans_cfg,
        **decomposition_cfg,
    )

    kmeans_cfg_n1 = {**kmeans_cfg, "n_clusters": 1, "kmeans_cluster_size": None}
    lrk_results_n1 = analyze_kmeans_lrk(
        tensor,
        kmeans_mode=lrk_mode,
        prefix_end=prefix_end,
        local_window=local_window,
        cluster_axis=cluster_axis,
        include_head_breakdown=include_lrk_head_breakdown,
        **kmeans_cfg_n1,
        **decomposition_cfg,
    )
    xkv_results_n1 = analyze_kmeans_xkv(
        tensor,
        layer_group_size=xkv_layer_group_size,
        num_layers=xkv_num_layers,
        prefix_end=prefix_end,
        local_window=local_window,
        cluster_axis=cluster_axis,
        **kmeans_cfg_n1,
        **decomposition_cfg,
    )

    lrk_results_random = None
    xkv_results_random = None
    if include_random_clustering:
        lrk_results_random = analyze_kmeans_lrk(
            tensor,
            kmeans_mode=lrk_mode,
            prefix_end=prefix_end,
            local_window=local_window,
            cluster_axis=cluster_axis,
            include_head_breakdown=include_lrk_head_breakdown,
            random_assignments=True,
            **kmeans_cfg,
            **decomposition_cfg,
        )
        xkv_results_random = analyze_kmeans_xkv(
            tensor,
            layer_group_size=xkv_layer_group_size,
            num_layers=xkv_num_layers,
            prefix_end=prefix_end,
            local_window=local_window,
            cluster_axis=cluster_axis,
            random_assignments=True,
            **kmeans_cfg,
            **decomposition_cfg,
        )

    lrk_df = _records_to_frame(lrk_results, hidden_columns=hidden_columns)
    xkv_df = _records_to_frame(xkv_results, hidden_columns=hidden_columns)
    lrk_df_n1 = _records_to_frame(
        lrk_results_n1,
        hidden_columns=hidden_columns,
    )
    xkv_df_n1 = _records_to_frame(
        xkv_results_n1,
        hidden_columns=hidden_columns,
    )
    lrk_df_random = (
        _records_to_frame(lrk_results_random, hidden_columns=hidden_columns)
        if lrk_results_random is not None
        else None
    )
    xkv_df_random = (
        _records_to_frame(xkv_results_random, hidden_columns=hidden_columns)
        if xkv_results_random is not None
        else None
    )

    summary_rows = _build_case_summary_rows(
        tensor_name,
        cluster_axis,
        lrk_results,
        xkv_results,
        lrk_results_n1,
        xkv_results_n1,
        lrk_random_records=lrk_results_random,
        xkv_random_records=xkv_results_random,
        lrk_df=lrk_df,
        xkv_df=xkv_df,
        lrk_df_n1=lrk_df_n1,
        xkv_df_n1=xkv_df_n1,
        lrk_random_df=lrk_df_random,
        xkv_random_df=xkv_df_random,
    )
    summary_df = pd.DataFrame(summary_rows) if pd is not None else None

    return {
        "case_name": f"{tensor_name}_{cluster_axis}",
        "tensor_name": tensor_name,
        "cluster_axis": cluster_axis,
        "summary_rows": summary_rows,
        "summary_df": summary_df,
        "lrk_results": lrk_results,
        "xkv_results": xkv_results,
        "lrk_results_n1": lrk_results_n1,
        "xkv_results_n1": xkv_results_n1,
        "lrk_results_random": lrk_results_random,
        "xkv_results_random": xkv_results_random,
        "lrk_df": lrk_df,
        "xkv_df": xkv_df,
        "lrk_df_n1": lrk_df_n1,
        "xkv_df_n1": xkv_df_n1,
        "lrk_df_random": lrk_df_random,
        "xkv_df_random": xkv_df_random,
    }


ALIGNMENT_MARKER_MAP = {
    ("K", "per_head"): "o",
    ("V", "per_head"): "^",
    ("K", "xkv"): "s",
    ("V", "xkv"): "D",
}


def _split_by_labels(X: torch.Tensor, labels: torch.Tensor) -> list[torch.Tensor]:
    return [X[labels == label] for label in labels.unique(sorted=True)]


def _rank_for_comp_ratio(X: torch.Tensor, comp_ratio: float) -> int:
    return find_rank_wrt_cr(comp_ratio, X.size(0), X.size(1))


def _top_right_basis(
    X: torch.Tensor,
    rank: int,
    center: bool = True,
) -> torch.Tensor:
    X = X.to(dtype=torch.float32)
    if center:
        X = X - X.mean(dim=0, keepdim=True)
    _, _, Vh = torch.linalg.svd(X.contiguous(), full_matrices=False)
    rank = min(rank, Vh.size(0))
    return Vh[:rank].T.contiguous()


def subspace_alignment_from_matrices(
    Xs: list[torch.Tensor],
    rank: int,
) -> float:
    Xs = [X for X in Xs if X.size(0) > 0]
    if len(Xs) < 2:
        return math.nan

    bases = [_top_right_basis(X, rank) for X in Xs]
    weights = [X.size(0) for X in Xs]
    numerator = torch.zeros((), device=Xs[0].device, dtype=torch.float32)
    denominator = 0

    for i in range(len(Xs)):
        for j in range(i + 1, len(Xs)):
            pair_rank = min(bases[i].size(1), bases[j].size(1))
            if pair_rank == 0:
                continue
            pair_alignment = (
                bases[i][:, :pair_rank].T @ bases[j][:, :pair_rank]
            ).pow(2).sum() / pair_rank
            weight = weights[i] * weights[j]
            numerator = numerator + weight * pair_alignment
            denominator += weight

    if denominator == 0:
        return math.nan
    return float((numerator / denominator).item())


def _centroid_separation(Xs: list[torch.Tensor]) -> float:
    Xs = [X.to(dtype=torch.float32) for X in Xs if X.size(0) > 0]
    if not Xs:
        return math.nan

    X = torch.cat(Xs, dim=0)
    global_mean = X.mean(dim=0, keepdim=True)
    total = (X - global_mean).pow(2).sum()
    if total <= 0:
        return 0.0

    between = torch.zeros((), device=X.device, dtype=torch.float32)
    for Xc in Xs:
        cluster_mean = Xc.mean(dim=0, keepdim=True)
        between = (
            between + Xc.size(0) * (cluster_mean - global_mean).pow(2).sum()
        )
    return float((between / total).item())


def _tail_energy_at_rank(X: torch.Tensor, rank: int) -> torch.Tensor:
    singular_values = torch.linalg.svdvals(
        X.to(dtype=torch.float32).contiguous()
    )
    rank = min(rank, singular_values.numel())
    return singular_values[rank:].pow(2).sum()


def _spectral_tail_delta(
    X: torch.Tensor,
    Xs: list[torch.Tensor],
    comp_ratio: float,
) -> float:
    X = X.to(dtype=torch.float32)
    denom = X.pow(2).sum()
    if denom <= 0:
        return 0.0

    clustered_tail = torch.zeros((), device=X.device, dtype=torch.float32)
    for Xc in Xs:
        if Xc.size(0) == 0:
            continue
        clustered_tail = clustered_tail + _tail_energy_at_rank(
            Xc,
            _rank_for_comp_ratio(Xc, comp_ratio),
        )

    global_tail = _tail_energy_at_rank(X, _rank_for_comp_ratio(X, comp_ratio))
    return float(((global_tail - clustered_tail) / denom).item())


def partition_alignment_diagnostics(
    X: torch.Tensor,
    labels: torch.Tensor,
    comp_ratio: float,
) -> dict[str, float | int]:
    Xs = _split_by_labels(X, labels)
    rank = _rank_for_comp_ratio(X, comp_ratio)
    return {
        "A_align": subspace_alignment_from_matrices(Xs, rank),
        "B_sep": _centroid_separation(Xs),
        "delta_tail": _spectral_tail_delta(X, Xs, comp_ratio),
        "rank": rank,
        "cluster_count": len(Xs),
    }


def _kmeans_labels_from_cfg(
    features: torch.Tensor,
    kmeans_cfg: dict[str, Any],
) -> torch.Tensor:
    cluster_count = _get_cluster_count(
        features.size(1),
        kmeans_cfg["n_clusters"],
        kmeans_cluster_size=kmeans_cfg.get("kmeans_cluster_size"),
    )
    return kmeans_cluster_sequences(
        features,
        n_clusters=cluster_count,
        n_iter=max(1, kmeans_cfg["kmeans_n_iter"]),
        kmeans_init=kmeans_cfg["kmeans_init"],
        dtype=_resolve_kmeans_dtype(kmeans_cfg["kmeans_dtype"]),
    )


def collect_per_head_partition_diagnostics(
    tensor: torch.Tensor,
    *,
    cache_name: str,
    kmeans_cfg: dict[str, Any],
    comp_ratio: float,
    prefix_end: int | None = None,
    local_window: int = 0,
) -> list[dict[str, Any]]:
    tensor = _ensure_layer_batched_tensor(tensor)
    rows = []

    with torch.no_grad():
        for layer_idx in range(tensor.size(0)):
            prefix, _ = _select_prefix(
                tensor[layer_idx],
                prefix_end=prefix_end,
                local_window=local_window,
            )
            batch_size, num_heads, seq_len, head_dim = prefix.shape
            features = prefix.reshape(batch_size * num_heads, seq_len, head_dim)
            labels = _kmeans_labels_from_cfg(features, kmeans_cfg)

            for flat_idx in range(features.size(0)):
                batch_idx = flat_idx // num_heads
                head_idx = flat_idx % num_heads
                metrics = partition_alignment_diagnostics(
                    features[flat_idx],
                    labels[flat_idx],
                    comp_ratio,
                )
                rows.append(
                    {
                        "diagnostic": "cluster_partition",
                        "scope": "per_head",
                        "cache": cache_name,
                        "layer_idx": layer_idx,
                        "batch_idx": batch_idx,
                        "head_idx": head_idx,
                        "group_start_layer": None,
                        "group_last_layer": None,
                        "seq_len": seq_len,
                        "feature_dim": head_dim,
                        **metrics,
                    }
                )

    return rows


def collect_xkv_partition_diagnostics(
    tensor: torch.Tensor,
    *,
    cache_name: str,
    kmeans_cfg: dict[str, Any],
    comp_ratio: float,
    layer_group_size: int,
    num_layers: int | None = None,
    prefix_end: int | None = None,
    local_window: int = 0,
) -> list[dict[str, Any]]:
    tensor = _ensure_layer_batched_tensor(tensor)
    if num_layers is None:
        num_layers = tensor.size(0)
    rows = []

    with torch.no_grad():
        for group_start in range(0, num_layers, layer_group_size):
            group_last = min(group_start + layer_group_size, num_layers) - 1
            prefix_layers = []
            for layer_idx in range(group_start, group_last + 1):
                prefix, _ = _select_prefix(
                    tensor[layer_idx],
                    prefix_end=prefix_end,
                    local_window=local_window,
                )
                prefix_layers.append(
                    prefix.transpose(1, 2).reshape(
                        prefix.size(0),
                        prefix.size(-2),
                        -1,
                    )
                )

            seq_lens = {prefix.size(1) for prefix in prefix_layers}
            if len(seq_lens) != 1:
                raise ValueError(
                    "All layers in an xKV group must share the same prefix "
                    "length."
                )

            features = torch.cat(prefix_layers, dim=-1)
            labels = _kmeans_labels_from_cfg(features, kmeans_cfg)
            for batch_idx in range(features.size(0)):
                metrics = partition_alignment_diagnostics(
                    features[batch_idx],
                    labels[batch_idx],
                    comp_ratio,
                )
                rows.append(
                    {
                        "diagnostic": "cluster_partition",
                        "scope": "xkv",
                        "cache": cache_name,
                        "layer_idx": group_last,
                        "batch_idx": batch_idx,
                        "head_idx": None,
                        "group_start_layer": group_start,
                        "group_last_layer": group_last,
                        "seq_len": features.size(1),
                        "feature_dim": features.size(2),
                        **metrics,
                    }
                )

    return rows


def collect_cross_layer_pooling_diagnostics(
    tensor: torch.Tensor,
    *,
    cache_name: str,
    comp_ratio: float,
    layer_group_size: int,
    num_layers: int | None = None,
    prefix_end: int | None = None,
    local_window: int = 0,
) -> list[dict[str, Any]]:
    tensor = _ensure_layer_batched_tensor(tensor)
    if num_layers is None:
        num_layers = tensor.size(0)
    rows = []

    with torch.no_grad():
        for group_start in range(0, num_layers, layer_group_size):
            group_last = min(group_start + layer_group_size, num_layers) - 1
            layer_prefixes = [
                _select_prefix(
                    tensor[layer_idx],
                    prefix_end=prefix_end,
                    local_window=local_window,
                )[0]
                for layer_idx in range(group_start, group_last + 1)
            ]
            batch_size, num_heads, seq_len, head_dim = layer_prefixes[0].shape

            for batch_idx in range(batch_size):
                pooled_layer_matrices = [
                    prefix[batch_idx]
                    .transpose(0, 1)
                    .reshape(seq_len, num_heads * head_dim)
                    for prefix in layer_prefixes
                ]
                pooled_rank = _rank_for_comp_ratio(
                    pooled_layer_matrices[0],
                    comp_ratio,
                )
                rows.append(
                    {
                        "diagnostic": "cross_layer_pooling",
                        "scope": "xkv_layer_matrix",
                        "cache": cache_name,
                        "batch_idx": batch_idx,
                        "head_idx": None,
                        "group_start_layer": group_start,
                        "group_last_layer": group_last,
                        "seq_len": seq_len,
                        "feature_dim": num_heads * head_dim,
                        "rank": pooled_rank,
                        "A_align": subspace_alignment_from_matrices(
                            pooled_layer_matrices,
                            pooled_rank,
                        ),
                        "B_sep": math.nan,
                        "delta_tail": math.nan,
                        "cluster_count": len(pooled_layer_matrices),
                    }
                )

                for head_idx in range(num_heads):
                    head_matrices = [
                        prefix[batch_idx, head_idx]
                        for prefix in layer_prefixes
                    ]
                    head_rank = _rank_for_comp_ratio(
                        head_matrices[0],
                        comp_ratio,
                    )
                    rows.append(
                        {
                            "diagnostic": "cross_layer_pooling",
                            "scope": "xkv_same_head",
                            "cache": cache_name,
                            "batch_idx": batch_idx,
                            "head_idx": head_idx,
                            "group_start_layer": group_start,
                            "group_last_layer": group_last,
                            "seq_len": seq_len,
                            "feature_dim": head_dim,
                            "rank": head_rank,
                            "A_align": subspace_alignment_from_matrices(
                                head_matrices,
                                head_rank,
                            ),
                            "B_sep": math.nan,
                            "delta_tail": math.nan,
                            "cluster_count": len(head_matrices),
                        }
                    )

    return rows


def collect_alignment_diagnostics(
    tensors: dict[str, torch.Tensor],
    *,
    kmeans_cfg: dict[str, Any],
    comp_ratio: float,
    layer_group_size: int,
    num_layers: int | None = None,
    prefix_end: int | None = None,
    local_window: int = 0,
):
    if pd is None:
        raise ImportError("pandas is required to collect alignment diagnostics.")

    rows = []
    for cache_name, tensor in tensors.items():
        rows.extend(
            collect_per_head_partition_diagnostics(
                tensor,
                cache_name=cache_name,
                kmeans_cfg=kmeans_cfg,
                comp_ratio=comp_ratio,
                prefix_end=prefix_end,
                local_window=local_window,
            )
        )
        rows.extend(
            collect_xkv_partition_diagnostics(
                tensor,
                cache_name=cache_name,
                kmeans_cfg=kmeans_cfg,
                comp_ratio=comp_ratio,
                layer_group_size=layer_group_size,
                num_layers=num_layers,
                prefix_end=prefix_end,
                local_window=local_window,
            )
        )
        rows.extend(
            collect_cross_layer_pooling_diagnostics(
                tensor,
                cache_name=cache_name,
                comp_ratio=comp_ratio,
                layer_group_size=layer_group_size,
                num_layers=num_layers,
                prefix_end=prefix_end,
                local_window=local_window,
            )
        )

    return pd.DataFrame(rows)


def summarize_alignment_diagnostics(alignment_df):
    return (
        alignment_df.groupby(["diagnostic", "scope", "cache"], dropna=False)
        .agg(
            n=("A_align", "count"),
            A_align_mean=("A_align", "mean"),
            A_align_median=("A_align", "median"),
            B_sep_mean=("B_sep", "mean"),
            B_sep_median=("B_sep", "median"),
            delta_tail_mean=("delta_tail", "mean"),
            delta_tail_median=("delta_tail", "median"),
        )
        .reset_index()
    )


def _cluster_partition_frame(alignment_df):
    return alignment_df[alignment_df["diagnostic"].eq("cluster_partition")]


def plot_alignment_tail_delta(
    alignment_df,
    *,
    figsize: tuple[float, float] = (7, 5),
    font_size: int | float = 12,
) -> Any:
    import matplotlib.pyplot as plt

    plot_df = _cluster_partition_frame(alignment_df)
    colors = {"per_head": "tab:blue", "xkv": "tab:orange"}
    markers = {"K": "o", "V": "^"}

    fig, ax = plt.subplots(figsize=figsize)
    for (scope, cache), group_df in plot_df.groupby(["scope", "cache"]):
        ax.scatter(
            group_df["A_align"],
            group_df["delta_tail"],
            label=f"{scope} {cache}",
            c=colors.get(scope, "tab:gray"),
            marker=markers.get(cache, "o"),
            alpha=0.75,
            edgecolors="none",
        )

    ax.axhline(0.0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Subspace alignment A", fontsize=font_size)
    ax.set_ylabel("Delta tail: global - clustered", fontsize=font_size)
    ax.legend(frameon=False, fontsize=font_size - 1)
    ax.tick_params(labelsize=font_size - 1)
    fig.tight_layout()
    return fig


def plot_alignment_centroid_delta(
    alignment_df,
    *,
    figsize: tuple[float, float] = (7, 5),
) -> Any:
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    cluster_df = _cluster_partition_frame(alignment_df)
    fig, ax = plt.subplots(figsize=figsize)
    max_abs_delta = cluster_df["delta_tail"].abs().max()
    if max_abs_delta <= 0:
        max_abs_delta = 1.0
    norm = TwoSlopeNorm(
        vmin=-max_abs_delta,
        vcenter=0.0,
        vmax=max_abs_delta,
    )
    for (cache, scope), group_df in cluster_df.groupby(["cache", "scope"]):
        ax.scatter(
            group_df["A_align"],
            group_df["B_sep"],
            c=group_df["delta_tail"],
            cmap="coolwarm",
            norm=norm,
            marker=ALIGNMENT_MARKER_MAP[(cache, scope)],
            alpha=0.75,
            edgecolors="none",
            label=f"{cache} {scope}",
        )

    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
    sm.set_array([])
    ax.set(xlabel="Subspace alignment A", ylabel="Centroid separation B")
    ax.legend(frameon=False)
    fig.colorbar(sm, ax=ax, label="Delta tail: global - clustered")
    fig.tight_layout()
    return fig


def plot_centroid_tail_delta(
    alignment_df,
    *,
    figsize: tuple[float, float] = (7, 5),
) -> Any:
    import matplotlib.pyplot as plt

    cluster_df = _cluster_partition_frame(alignment_df)
    fig, ax = plt.subplots(figsize=figsize)
    for (cache, scope), group_df in cluster_df.groupby(["cache", "scope"]):
        ax.scatter(
            group_df["B_sep"],
            group_df["delta_tail"],
            marker=ALIGNMENT_MARKER_MAP[(cache, scope)],
            alpha=0.75,
            edgecolors="none",
            label=f"{cache} {scope}",
        )
    ax.axhline(0.0, color="black", linewidth=1, linestyle="--")
    ax.set(
        xlabel="Centroid separation B",
        ylabel="Delta tail: global - clustered",
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def _head_representations_for_cka(
    tensor: torch.Tensor,
    prefix_end: int | None = None,
    local_window: int = 0,
) -> tuple[torch.Tensor, list[str], int, int]:
    tensor = _select_prefix(
        tensor,
        prefix_end=prefix_end,
        local_window=local_window,
    )[0]
    if tensor.dim() == 4:
        num_layers, num_heads, seq_len, head_dim = tensor.shape
        reps = tensor.reshape(num_layers * num_heads, seq_len, head_dim)
    elif tensor.dim() == 5:
        num_layers, batch_size, num_heads, seq_len, head_dim = tensor.shape
        reps = tensor.permute(0, 2, 1, 3, 4).reshape(
            num_layers * num_heads,
            batch_size * seq_len,
            head_dim,
        )
    else:
        raise ValueError(
            "Expected tensor with shape [layers, heads, seq, dim] or "
            "[layers, batch, heads, seq, dim]."
        )

    labels = [
        f"L{layer}H{head}"
        for layer in range(num_layers)
        for head in range(num_heads)
    ]
    return reps, labels, num_layers, num_heads


def _layer_representations_for_cka(
    tensor: torch.Tensor,
    prefix_end: int | None = None,
    local_window: int = 0,
) -> tuple[torch.Tensor, list[str]]:
    tensor = _select_prefix(
        tensor,
        prefix_end=prefix_end,
        local_window=local_window,
    )[0]
    if tensor.dim() == 4:
        num_layers, num_heads, seq_len, head_dim = tensor.shape
        reps = tensor.permute(0, 2, 1, 3).reshape(
            num_layers,
            seq_len,
            num_heads * head_dim,
        )
    elif tensor.dim() == 5:
        num_layers, batch_size, num_heads, seq_len, head_dim = tensor.shape
        reps = tensor.permute(0, 1, 3, 2, 4).reshape(
            num_layers,
            batch_size * seq_len,
            num_heads * head_dim,
        )
    else:
        raise ValueError(
            "Expected tensor with shape [layers, heads, seq, dim] or "
            "[layers, batch, heads, seq, dim]."
        )

    return reps, [f"L{layer}" for layer in range(num_layers)]


def linear_cka_matrix(reps: torch.Tensor) -> torch.Tensor:
    reps = reps.to(dtype=torch.float32)
    reps = reps - reps.mean(dim=1, keepdim=True)
    num_reps, _, feature_dim = reps.shape
    norms = torch.empty(num_reps, device=reps.device, dtype=torch.float32)
    for idx in range(num_reps):
        cov = reps[idx].T @ reps[idx]
        norms[idx] = cov.pow(2).sum().sqrt().clamp_min(1e-12)

    cka = torch.empty(
        num_reps,
        num_reps,
        device=reps.device,
        dtype=torch.float32,
    )
    block_size = max(1, min(16, 2_000_000 // max(1, feature_dim * feature_dim)))
    for start_i in range(0, num_reps, block_size):
        stop_i = min(start_i + block_size, num_reps)
        Xi = reps[start_i:stop_i]
        for start_j in range(start_i, num_reps, block_size):
            stop_j = min(start_j + block_size, num_reps)
            Xj = reps[start_j:stop_j]
            cross_cov = torch.einsum("itd,jte->ijde", Xi, Xj)
            values = cross_cov.pow(2).sum(dim=(-1, -2)) / (
                norms[start_i:stop_i, None] * norms[None, start_j:stop_j]
            )
            cka[start_i:stop_i, start_j:stop_j] = values
            if start_i != start_j:
                cka[start_j:stop_j, start_i:stop_i] = values.T

    return cka.clamp(0.0, 1.0).cpu()


def collect_head_cka(
    tensors: dict[str, torch.Tensor],
    *,
    prefix_end: int | None = None,
    local_window: int = 0,
) -> tuple[dict[str, torch.Tensor], list[str], int, int]:
    cka_by_cache = {}
    head_labels = None
    num_layers = None
    num_heads = None

    with torch.no_grad():
        for cache_name, tensor in tensors.items():
            reps, labels, cache_num_layers, cache_num_heads = (
                _head_representations_for_cka(
                    tensor,
                    prefix_end=prefix_end,
                    local_window=local_window,
                )
            )
            cka_by_cache[cache_name] = linear_cka_matrix(reps)
            if head_labels is None:
                head_labels = labels
                num_layers = cache_num_layers
                num_heads = cache_num_heads
            elif (
                labels != head_labels
                or cache_num_layers != num_layers
                or cache_num_heads != num_heads
            ):
                raise ValueError(
                    "All tensors must have matching layer/head dimensions."
                )

    return cka_by_cache, head_labels, num_layers, num_heads


def collect_layer_cka(
    tensors: dict[str, torch.Tensor],
    *,
    prefix_end: int | None = None,
    local_window: int = 0,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    cka_by_cache = {}
    layer_labels = None

    with torch.no_grad():
        for cache_name, tensor in tensors.items():
            reps, labels = _layer_representations_for_cka(
                tensor,
                prefix_end=prefix_end,
                local_window=local_window,
            )
            cka_by_cache[cache_name] = linear_cka_matrix(reps)
            if layer_labels is None:
                layer_labels = labels
            elif labels != layer_labels:
                raise ValueError("All tensors must have matching layers.")

    return cka_by_cache, layer_labels


def _cka_mean_std(values: torch.Tensor) -> dict[str, float | int]:
    values = values.detach().cpu().to(dtype=torch.float32)
    if values.numel() == 0:
        return {
            "n_pairs": 0,
            "CKA_mean": math.nan,
            "CKA_std": math.nan,
        }
    return {
        "n_pairs": int(values.numel()),
        "CKA_mean": float(values.mean().item()),
        "CKA_std": float(values.std(unbiased=False).item()),
    }


def _sort_cka_summary_by_mean(summary, cache_order: list[str]):
    summary = summary.copy()
    summary["_cache_order"] = summary["cache"].map(
        {cache_name: idx for idx, cache_name in enumerate(cache_order)}
    )
    return (
        summary.sort_values(
            ["_cache_order", "CKA_mean"],
            ascending=[True, False],
            na_position="last",
        )
        .drop(columns="_cache_order")
        .reset_index(drop=True)
    )


def summarize_head_cka(
    cka_by_cache: dict[str, torch.Tensor],
    *,
    num_layers: int,
    num_heads: int,
    layer_group_size: int,
    sort_by_cka_mean: bool = False,
):
    if pd is None:
        raise ImportError("pandas is required to summarize CKA metrics.")

    pair_idx = torch.triu_indices(
        num_layers * num_heads,
        num_layers * num_heads,
        offset=1,
    )
    layers = pair_idx // num_heads
    heads = pair_idx % num_heads
    groups = layers // layer_group_size
    same_head = heads[0].eq(heads[1])
    same_group = groups[0].eq(groups[1])
    masks = {
        "all heads": torch.ones(pair_idx.size(1), dtype=torch.bool),
        "same layer": layers[0].eq(layers[1]),
        "different layer": layers[0].ne(layers[1]),
        "same xKV group": same_group,
        "different xKV group": same_group.logical_not(),
        "same head index": same_head,
        "same head index, same xKV group": same_head & same_group,
        "same head index, different xKV group": (
            same_head & same_group.logical_not()
        ),
    }

    rows = []
    for cache_name, cka in cka_by_cache.items():
        cka = cka.detach().cpu()
        for setting, mask in masks.items():
            rows.append(
                {
                    "cache": cache_name,
                    "setting": setting,
                    **_cka_mean_std(
                        cka[pair_idx[0, mask], pair_idx[1, mask]]
                    ),
                }
            )
    summary = pd.DataFrame(rows)
    if sort_by_cka_mean:
        summary = _sort_cka_summary_by_mean(
            summary,
            cache_order=list(cka_by_cache),
        )
    return summary


def summarize_layer_cka(
    cka_by_cache: dict[str, torch.Tensor],
    *,
    num_layers: int,
    layer_group_size: int,
    sort_by_cka_mean: bool = False,
):
    if pd is None:
        raise ImportError("pandas is required to summarize CKA metrics.")

    pair_idx = torch.triu_indices(num_layers, num_layers, offset=1)
    groups = pair_idx // layer_group_size
    masks = {
        "all layers": torch.ones(pair_idx.size(1), dtype=torch.bool),
        "same xKV group": groups[0].eq(groups[1]),
        "different xKV group": groups[0].ne(groups[1]),
    }

    rows = []
    for cache_name, cka in cka_by_cache.items():
        cka = cka.detach().cpu()
        for setting, mask in masks.items():
            rows.append(
                {
                    "cache": cache_name,
                    "setting": setting,
                    **_cka_mean_std(
                        cka[pair_idx[0, mask], pair_idx[1, mask]]
                    ),
                }
            )
    summary = pd.DataFrame(rows)
    if sort_by_cka_mean:
        summary = _sort_cka_summary_by_mean(
            summary,
            cache_order=list(cka_by_cache),
        )
    return summary


def plot_head_cka_heatmaps(
    cka_by_cache: dict[str, torch.Tensor],
    *,
    num_layers: int,
    num_heads: int,
    figsize: tuple[float, float] = (13, 6),
) -> Any:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1,
        len(cka_by_cache),
        figsize=figsize,
        squeeze=False,
    )
    layer_centers = [
        layer * num_heads + (num_heads - 1) / 2
        for layer in range(num_layers)
    ]
    layer_boundaries = [
        layer * num_heads - 0.5 for layer in range(1, num_layers)
    ]
    for ax, (cache_name, cka) in zip(axes.flat, cka_by_cache.items()):
        im = ax.imshow(
            cka.detach().cpu().numpy(),
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
            interpolation="nearest",
        )
        ax.set_title(f"{cache_name} head CKA")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Layer")
        ax.set_xticks(layer_centers)
        ax.set_yticks(layer_centers)
        ax.set_xticklabels(range(num_layers), rotation=90, fontsize=7)
        ax.set_yticklabels(range(num_layers), fontsize=7)
        for boundary in layer_boundaries:
            ax.axhline(boundary, color="white", linewidth=0.25, alpha=0.5)
            ax.axvline(boundary, color="white", linewidth=0.25, alpha=0.5)

    fig.colorbar(im, ax=axes.ravel().tolist(), label="Linear CKA")
    return fig


def plot_layer_cka_heatmaps(
    cka_by_cache: dict[str, torch.Tensor],
    *,
    layer_labels: list[str],
    figsize: tuple[float, float] = (11, 5),
) -> Any:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1,
        len(cka_by_cache),
        figsize=figsize,
        squeeze=False,
    )
    for ax, (cache_name, cka) in zip(axes.flat, cka_by_cache.items()):
        im = ax.imshow(
            cka.detach().cpu().numpy(),
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
            interpolation="nearest",
        )
        ax.set_title(f"{cache_name} layer CKA")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Layer")
        ax.set_xticks(range(len(layer_labels)))
        ax.set_yticks(range(len(layer_labels)))
        ax.set_xticklabels(layer_labels, rotation=90, fontsize=7)
        ax.set_yticklabels(layer_labels, fontsize=7)

    fig.colorbar(im, ax=axes.ravel().tolist(), label="Linear CKA")
    return fig


def _metric_plot_frame(
    records: list[dict[str, Any]],
    *,
    metric_scope: str,
    group_columns: tuple[str, ...],
):
    if pd is None:
        raise ImportError("pandas is required to plot clustering metrics.")

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    if df.empty:
        return df

    df["relative_low_rank_recon_error_bound"] = (
        df["eta"].clip(lower=0.0).pow(0.5)
    )
    metric_columns = [name for name, _ in CLUSTERING_METRIC_PLOTS]
    df = df[df["metric_scope"].eq(metric_scope)].copy()
    for column in group_columns:
        df = df[df[column].notna()]

    if df.empty:
        return df

    return (
        df.groupby(list(group_columns), as_index=False)
        .agg({column: "mean" for column in metric_columns})
        .sort_values(list(group_columns))
    )


def _set_sparse_tick_labels(
    ax,
    labels: list[str],
    *,
    axis: str,
    max_ticks: int = 16,
    font_size: int | float = 11,
) -> None:
    import math

    if not labels:
        return

    tick_positions = list(range(len(labels)))
    if len(labels) > max_ticks:
        step = math.ceil(len(labels) / max_ticks)
        tick_positions = tick_positions[::step]
        if tick_positions[-1] != len(labels) - 1:
            tick_positions.append(len(labels) - 1)

    tick_labels = [labels[idx] for idx in tick_positions]
    if axis == "x":
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(
            tick_labels,
            rotation=45,
            ha="right",
            fontsize=font_size,
        )
    else:
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels, fontsize=font_size)


def _apply_paper_axis_style(ax, font_size: int | float) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=font_size)


def _place_external_legend(ax, font_size: int | float) -> None:
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=len(labels),
        fontsize=font_size,
        frameon=False,
    )


def _make_plot_axes(
    figsize: tuple[float, float],
    show_difference: bool,
):
    import matplotlib.pyplot as plt

    if not show_difference:
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax, None

    stacked_figsize = (figsize[0], figsize[1] * 1.5)
    fig, axes = plt.subplots(
        2,
        1,
        figsize=stacked_figsize,
        gridspec_kw={"height_ratios": [1, 1]},
    )
    return fig, axes[0], axes[1]


def _line_style_kwargs(row_label: str, metric_name: str) -> dict[str, Any]:
    style = {
        "color": CLUSTERING_METRIC_COLORS[metric_name],
        **CLUSTERING_SET_STYLES.get(
            row_label,
            CLUSTERING_SET_STYLES["Clustered"],
        ),
    }
    return style


def _difference_style_kwargs(row_label: str) -> dict[str, Any]:
    style = {
        "color": CLUSTERING_DIFFERENCE_COLORS[row_label],
        **CLUSTERING_SET_STYLES.get(
            row_label,
            CLUSTERING_SET_STYLES["Clustered"],
        ),
    }
    return style


def _plot_bound_error_ratio(
    ax,
    frame_sets,
    *,
    x_positions: list[int],
    font_size: int | float,
) -> None:
    for row_label, df in frame_sets:
        ratio = df["relative_low_rank_recon_error_bound"] / df[
            "relative_low_rank_recon_error"
        ].clip(lower=1e-12)
        ax.plot(
            x_positions,
            ratio,
            **_difference_style_kwargs(row_label),
            markersize=4,
            linewidth=1.5,
            label=f"{row_label}: bound / error",
        )

    ax.axhline(1.0, color="0.35", linewidth=1, alpha=0.8)
    ax.set_xlabel("Layer/head", fontsize=font_size)
    ax.set_ylabel("Bound / error", fontsize=font_size)
    ax.grid(True, alpha=0.25)
    _place_external_legend(ax, font_size)
    _apply_paper_axis_style(ax, font_size)


def _plot_lrk_head_metrics(
    metric_sets: list[tuple[str, list[dict[str, Any]]]],
    *,
    font_size: int | float,
    figsize: tuple[float, float],
    show_difference: bool,
) -> Any | None:
    import matplotlib.pyplot as plt

    frame_sets = []
    for label, records in metric_sets:
        df = _metric_plot_frame(
            records,
            metric_scope="head",
            group_columns=("layer_idx", "head_idx"),
        )
        if not df.empty:
            df = df.sort_values(["layer_idx", "head_idx"]).reset_index(
                drop=True
            )
            frame_sets.append((label, df))

    if not frame_sets:
        return

    base_df = frame_sets[0][1]
    max_layers = max(df["layer_idx"].nunique() for _, df in frame_sets)
    fig, ax, diff_ax = _make_plot_axes(figsize, show_difference)
    diff_positions = list(range(len(base_df)))

    for row_label, df in frame_sets:
        x_positions = list(range(len(df)))
        for metric_name, metric_label in CLUSTERING_METRIC_PLOTS:
            style = _line_style_kwargs(row_label, metric_name)
            ax.plot(
                x_positions,
                df[metric_name],
                **style,
                markersize=2,
                linewidth=1,
                label=f"{row_label}: {metric_label}",
            )

    x_labels = [
        f"L{int(row.layer_idx)} H{int(row.head_idx)}"
        for row in base_df.itertuples()
    ]
    ax.set_xlabel("Layer/head", fontsize=font_size)
    ax.set_ylabel("Error", fontsize=font_size)
    ax.grid(True, alpha=0.25)
    _place_external_legend(ax, font_size)
    _set_sparse_tick_labels(
        ax,
        x_labels,
        axis="x",
        max_ticks=min(18, max(8, max_layers)),
        font_size=font_size,
    )
    _apply_paper_axis_style(ax, font_size)

    if diff_ax is not None:
        _plot_bound_error_ratio(
            diff_ax,
            frame_sets,
            x_positions=diff_positions,
            font_size=font_size,
        )
        diff_ax.set_xlabel("Layer/head", fontsize=font_size)
        _set_sparse_tick_labels(
            diff_ax,
            x_labels,
            axis="x",
            max_ticks=min(10, max(6, max_layers)),
            font_size=font_size,
        )

    fig.tight_layout(rect=(0, 0, 1, 0.9))
    plt.show()
    return fig


def _plot_lrk_layer_mean_metrics(
    metric_sets: list[tuple[str, list[dict[str, Any]]]],
    *,
    font_size: int | float,
    figsize: tuple[float, float],
    show_difference: bool,
) -> Any | None:
    import matplotlib.pyplot as plt

    frame_sets = []
    for label, records in metric_sets:
        df = _metric_plot_frame(
            records,
            metric_scope="head",
            group_columns=("layer_idx", "head_idx"),
        )
        if df.empty:
            continue

        metric_aggs = {}
        for metric_name, _ in CLUSTERING_METRIC_PLOTS:
            metric_aggs[metric_name] = ["mean", "std"]
        layer_df = df.groupby("layer_idx", as_index=False).agg(metric_aggs)
        layer_df.columns = [
            "_".join(item).rstrip("_")
            for item in layer_df.columns.to_flat_index()
        ]
        layer_df = layer_df.fillna(0.0).sort_values("layer_idx")
        frame_sets.append((label, layer_df))

    if not frame_sets:
        return

    base_df = frame_sets[0][1]
    fig, ax, diff_ax = _make_plot_axes(figsize, show_difference)
    difference_frame_sets = []

    for row_label, df in frame_sets:
        x_positions = list(range(len(df)))
        difference_df = pd.DataFrame(
            {
                "relative_low_rank_recon_error_bound": df[
                    "relative_low_rank_recon_error_bound_mean"
                ],
                "relative_low_rank_recon_error": df[
                    "relative_low_rank_recon_error_mean"
                ],
            }
        )
        difference_frame_sets.append((row_label, difference_df))
        for metric_name, metric_label in CLUSTERING_METRIC_PLOTS:
            style = _line_style_kwargs(row_label, metric_name)
            mean = df[f"{metric_name}_mean"].to_numpy(dtype=float)
            std = df[f"{metric_name}_std"].to_numpy(dtype=float)
            ax.plot(
                x_positions,
                mean,
                **style,
                markersize=4,
                linewidth=1.5,
                label=f"{row_label}: {metric_label}",
            )
            ax.fill_between(
                x_positions,
                mean - std,
                mean + std,
                color=style["color"],
                alpha=0.25,
                linewidth=0,
            )

    x_labels = [f"L{int(row.layer_idx)}" for row in base_df.itertuples()]
    ax.set_xlabel("Layer", fontsize=font_size)
    ax.set_ylabel("Error", fontsize=font_size)
    ax.grid(True, alpha=0.25)
    _place_external_legend(ax, font_size)
    _set_sparse_tick_labels(
        ax,
        x_labels,
        axis="x",
        max_ticks=18,
        font_size=font_size,
    )
    _apply_paper_axis_style(ax, font_size)

    if diff_ax is not None:
        _plot_bound_error_ratio(
            diff_ax,
            difference_frame_sets,
            x_positions=list(range(len(base_df))),
            font_size=font_size,
        )
        diff_ax.set_xlabel("Layer", fontsize=font_size)
        _set_sparse_tick_labels(
            diff_ax,
            x_labels,
            axis="x",
            max_ticks=10,
            font_size=font_size,
        )

    fig.tight_layout(rect=(0, 0, 1, 0.9))
    plt.show()
    return fig


def _plot_xkv_group_metrics(
    metric_sets: list[tuple[str, list[dict[str, Any]]]],
    *,
    font_size: int | float,
    figsize: tuple[float, float],
    show_difference: bool,
) -> Any | None:
    import matplotlib.pyplot as plt

    frame_sets = []
    for label, records in metric_sets:
        df = _metric_plot_frame(
            records,
            metric_scope="group",
            group_columns=("group_start_layer", "group_last_layer"),
        )
        if not df.empty:
            frame_sets.append((label, df))

    if not frame_sets:
        return

    base_df = frame_sets[0][1]
    fig, ax, diff_ax = _make_plot_axes(figsize, show_difference)
    diff_positions = list(range(len(base_df)))

    for row_label, df in frame_sets:
        x_positions = list(range(len(df)))
        for metric_name, metric_label in CLUSTERING_METRIC_PLOTS:
            style = _line_style_kwargs(row_label, metric_name)
            ax.plot(
                x_positions,
                df[metric_name],
                **style,
                markersize=4,
                linewidth=1.5,
                label=f"{row_label}: {metric_label}",
            )

    group_labels = [
        f"{int(row.group_start_layer)}-{int(row.group_last_layer)}"
        for row in base_df.itertuples()
    ]
    ax.set_xlabel("Layer group", fontsize=font_size)
    ax.set_ylabel("Error", fontsize=font_size)
    ax.grid(True, alpha=0.25)
    _place_external_legend(ax, font_size)
    _set_sparse_tick_labels(
        ax,
        group_labels,
        axis="x",
        max_ticks=12,
        font_size=font_size,
    )
    _apply_paper_axis_style(ax, font_size)

    if diff_ax is not None:
        _plot_bound_error_ratio(
            diff_ax,
            frame_sets,
            x_positions=diff_positions,
            font_size=font_size,
        )
        diff_ax.set_xlabel("Layer group", fontsize=font_size)
        _set_sparse_tick_labels(
            diff_ax,
            group_labels,
            axis="x",
            max_ticks=8,
            font_size=font_size,
        )

    fig.tight_layout(rect=(0, 0, 1, 0.9))
    plt.show()
    return fig


def _plot_combined_lrk_xkv_metrics(
    lrk_metric_sets: list[tuple[str, list[dict[str, Any]]]],
    xkv_metric_sets: list[tuple[str, list[dict[str, Any]]]],
    *,
    font_size: int | float,
    figsize: tuple[float, float],
) -> Any | None:
    import matplotlib.pyplot as plt

    lrk_frame_sets = []
    for label, records in lrk_metric_sets:
        df = _metric_plot_frame(
            records,
            metric_scope="head",
            group_columns=("layer_idx", "head_idx"),
        )
        if not df.empty:
            df = df.sort_values(["layer_idx", "head_idx"]).reset_index(
                drop=True
            )
            lrk_frame_sets.append((label, df))

    xkv_frame_sets = []
    for label, records in xkv_metric_sets:
        df = _metric_plot_frame(
            records,
            metric_scope="group",
            group_columns=("group_start_layer", "group_last_layer"),
        )
        if not df.empty:
            xkv_frame_sets.append((label, df))

    if not lrk_frame_sets or not xkv_frame_sets:
        return

    fig = plt.figure(figsize=figsize)
    grid = fig.add_gridspec(
        2,
        3,
        height_ratios=(1, 1),
        width_ratios=(1, 1, 1),
    )
    lrk_ax = fig.add_subplot(grid[0, :])
    xkv_ax = fig.add_subplot(grid[1, :2])
    table_ax = fig.add_subplot(grid[1, 2])
    tick_font_size = max(font_size - 2, 8)

    for row_label, df in lrk_frame_sets:
        x_positions = list(range(len(df)))
        for metric_name, metric_label in CLUSTERING_METRIC_PLOTS:
            lrk_ax.plot(
                x_positions,
                df[metric_name],
                **_line_style_kwargs(row_label, metric_name),
                markersize=2,
                linewidth=1,
                label=f"{row_label}: {metric_label}",
            )

    lrk_base_df = lrk_frame_sets[0][1]
    lrk_labels = [
        f"L{int(row.layer_idx)} H{int(row.head_idx)}"
        for row in lrk_base_df.itertuples()
    ]
    max_layers = max(df["layer_idx"].nunique() for _, df in lrk_frame_sets)
    lrk_ax.set_xlabel("Layer/head", fontsize=font_size)
    lrk_ax.set_ylabel("Error", fontsize=font_size)
    lrk_ax.grid(True, alpha=0.25)
    _set_sparse_tick_labels(
        lrk_ax,
        lrk_labels,
        axis="x",
        max_ticks=min(18, max(8, max_layers)),
        font_size=tick_font_size,
    )
    _apply_paper_axis_style(lrk_ax, tick_font_size)

    for row_label, df in xkv_frame_sets:
        x_positions = list(range(len(df)))
        for metric_name, metric_label in CLUSTERING_METRIC_PLOTS:
            xkv_ax.plot(
                x_positions,
                df[metric_name],
                **_line_style_kwargs(row_label, metric_name),
                markersize=4,
                linewidth=1.5,
                label=f"{row_label}: {metric_label}",
            )

    xkv_base_df = xkv_frame_sets[0][1]
    group_labels = [
        f"{int(row.group_start_layer)}-{int(row.group_last_layer)}"
        for row in xkv_base_df.itertuples()
    ]
    xkv_ax.set_xlabel("Layer group", fontsize=font_size)
    xkv_ax.set_ylabel("Error", fontsize=font_size)
    xkv_ax.grid(True, alpha=0.25)
    _set_sparse_tick_labels(
        xkv_ax,
        group_labels,
        axis="x",
        max_ticks=12,
        font_size=tick_font_size,
    )
    _apply_paper_axis_style(xkv_ax, tick_font_size)

    correlation_rows = []
    for plot_label, frame_sets in (
        ("PH", lrk_frame_sets),
        ("xKV", xkv_frame_sets),
    ):
        for row_label, df in frame_sets:
            bound = df["relative_low_rank_recon_error_bound"]
            error = df["relative_low_rank_recon_error"]
            pearson = bound.corr(error, method="pearson")
            spearman = bound.corr(error, method="spearman")
            pair_label = {
                "Clustered": "Clust.",
                "Random": "Rand.",
                "Unclustered": "Unclust.",
            }.get(row_label, row_label)
            correlation_rows.append(
                (
                    plot_label,
                    pair_label,
                    f"{pearson:.2f}",
                    f"{spearman:.2f}",
                )
            )

    table_ax.axis("off")
    correlation_table = table_ax.table(
        cellText=correlation_rows,
        colLabels=("Plot", "Pair", "Pearson", "Spearman"),
        cellLoc="center",
        colLoc="center",
        colWidths=(0.14, 0.26, 0.30, 0.30),
        loc="center",
    )
    correlation_table.auto_set_font_size(False)
    correlation_table.set_fontsize(tick_font_size)
    correlation_table.scale(1, 2.53125)

    handles, labels = lrk_ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=len(labels),
        fontsize=font_size,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    plt.show()
    return fig


def _spectral_curves(
    records: list[dict[str, Any]],
    *,
    metric_scopes: tuple[str, ...],
) -> torch.Tensor | None:
    for metric_scope in metric_scopes:
        curves = [
            curve
            for record in records
            if record["metric_scope"] == metric_scope
            for curve in record.get("spectral_tail_curves", [])
        ]
        if curves:
            return torch.tensor(curves, dtype=torch.float32)
    return None


def _case_metric_sets(
    case_result: dict[str, Any],
    prefix: str,
) -> list[tuple[str, list[dict[str, Any]]]]:
    metric_sets = [("Clustered", case_result[f"{prefix}_results"])]
    random_results = case_result.get(f"{prefix}_results_random")
    if random_results is not None:
        metric_sets.append(("Random", random_results))
    metric_sets.append(("Unclustered", case_result[f"{prefix}_results_n1"]))
    return metric_sets


def _plot_relative_spectral_tail_axes(
    axes: Any,
    plot_specs: tuple[Any, ...],
    *,
    font_size: int | float,
    ylim: tuple[float, float] | None,
) -> bool:
    plotted = False
    for ax, (title, metric_sets, metric_scopes) in zip(
        axes, plot_specs, strict=True
    ):
        for label, records in metric_sets:
            curves = _spectral_curves(
                records,
                metric_scopes=metric_scopes,
            )
            if curves is None:
                continue
            plotted = True
            x_values = torch.linspace(0, 100, curves.size(1)).numpy()
            mean = curves.mean(dim=0)
            median = curves.median(dim=0).values
            color = CLUSTERING_DIFFERENCE_COLORS[label]
            ax.plot(
                x_values,
                mean.clamp_min(1e-6).numpy(),
                color=color,
                linestyle="-",
                linewidth=1.5,
                label=f"{label} mean",
            )
            ax.plot(
                x_values,
                median.clamp_min(1e-6).numpy(),
                color=color,
                linestyle="--",
                linewidth=1.5,
                label=f"{label} median",
            )
            if curves.size(0) > 1:
                std = curves.std(dim=0, unbiased=False)
                ax.fill_between(
                    x_values,
                    (mean - std).clamp(min=1e-6, max=100).numpy(),
                    (mean + std).clamp(min=1e-6, max=100).numpy(),
                    color=color,
                    alpha=0.25,
                    linewidth=0,
                )

        ax.set_title(title, fontsize=font_size)
        ax.set_xlabel("Relative rank (%)", fontsize=font_size)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.25)
        _apply_paper_axis_style(ax, font_size)
    return plotted


def _plot_relative_spectral_tails(
    lrk_metric_sets: list[tuple[str, list[dict[str, Any]]]],
    xkv_metric_sets: list[tuple[str, list[dict[str, Any]]]],
    *,
    font_size: int | float,
    figsize: tuple[float, float],
    ylim: tuple[float, float] | None,
) -> Any | None:
    import matplotlib.pyplot as plt

    plot_specs = (
        ("Per-head", lrk_metric_sets, ("head", "layer")),
        ("xKV", xkv_metric_sets, ("group",)),
    )
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    plotted = _plot_relative_spectral_tail_axes(
        axes,
        plot_specs,
        font_size=font_size,
        ylim=ylim,
    )
    if not plotted:
        plt.close(fig)
        return None

    axes[0].set_ylabel("Spectral tail mass (%)", fontsize=font_size)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(labels),
        fontsize=font_size,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    plt.show()
    return fig


def plot_combined_relative_spectral_tails(
    keys_result: dict[str, Any],
    values_result: dict[str, Any],
    *,
    font_size: int | float = 11,
    figsize: tuple[float, float] = (10, 10),
    values_ylim: tuple[float, float] | None = None,
) -> Any | None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=figsize, sharey="row")
    plotted = False
    row_plot_specs = []
    for row_axes, case_result in (
        (axes[0], keys_result),
        (axes[1], values_result),
    ):
        lrk_metric_sets = _case_metric_sets(case_result, "lrk")
        xkv_metric_sets = _case_metric_sets(case_result, "xkv")
        plot_specs = (
            ("Per-head", lrk_metric_sets, ("head", "layer")),
            ("xKV", xkv_metric_sets, ("group",)),
        )
        row_plot_specs.append(plot_specs)
        plotted |= _plot_relative_spectral_tail_axes(
            row_axes,
            plot_specs,
            font_size=font_size,
            ylim=values_ylim,
        )

    if not plotted:
        plt.close(fig)
        return None

    axes[0, 0].set_ylabel("Keys\nSpectral tail mass (%)", fontsize=font_size)
    axes[1, 0].set_ylabel("Values\nSpectral tail mass (%)", fontsize=font_size)
    inset_axes = [ax.inset_axes((0.47, 0.42, 0.5, 0.5)) for ax in axes[0]]
    _plot_relative_spectral_tail_axes(
        inset_axes,
        row_plot_specs[0],
        font_size=max(font_size - 4, 8),
        ylim=(0, 30),
    )
    for inset_ax in inset_axes:
        inset_ax.set_xlim(0, 20)
        inset_ax.set_title("")
        inset_ax.set_xlabel("")
        inset_ax.grid(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(labels),
        fontsize=font_size,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    plt.show()
    return fig


def _plot_spectral_tails(
    lrk_metric_sets: list[tuple[str, list[dict[str, Any]]]],
    xkv_metric_sets: list[tuple[str, list[dict[str, Any]]]],
    *,
    font_size: int | float,
    figsize: tuple[float, float],
    ylim: tuple[float, float] | None,
) -> Any | None:
    import matplotlib.pyplot as plt

    plot_specs = (
        ("Per-head", lrk_metric_sets, ("head", "layer")),
        ("xKV", xkv_metric_sets, ("group",)),
    )
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharey=True)
    plotted = _plot_relative_spectral_tail_axes(
        axes[0],
        plot_specs,
        font_size=font_size,
        ylim=None,
    )

    log_ylim = ylim
    if log_ylim is not None and log_ylim[0] <= 0:
        log_ylim = (1e-6, log_ylim[1])
    for ax in axes[0]:
        ax.set_yscale("log")
        if log_ylim is not None:
            ax.set_ylim(*log_ylim)

    for ax, (title, metric_sets, metric_scopes) in zip(
        axes[1], plot_specs, strict=True
    ):
        for label, records in metric_sets:
            curves = []
            for metric_scope in metric_scopes:
                curves = [
                    curve
                    for record in records
                    if record["metric_scope"] == metric_scope
                    for curve in record.get("spectral_tail_raw_curves", [])
                ]
                if curves:
                    break
            color = CLUSTERING_DIFFERENCE_COLORS[label]
            for curve_idx, curve in enumerate(curves):
                plotted = True
                ax.plot(
                    range(len(curve)),
                    torch.tensor(curve).clamp_min(1e-6).numpy(),
                    color=color,
                    linestyle=CLUSTERING_SET_STYLES[label]["linestyle"],
                    linewidth=1,
                    alpha=0.35 if label == "Clustered" else 0.6,
                    label=label if curve_idx == 0 else None,
                )

        ax.set_xlabel("Rank", fontsize=font_size)
        ax.set_yscale("log")
        if log_ylim is not None:
            ax.set_ylim(*log_ylim)
        ax.grid(True, alpha=0.25)
        _apply_paper_axis_style(ax, font_size)

    if not plotted:
        plt.close(fig)
        return None

    axes[0, 0].set_ylabel("Spectral tail mass (%)", fontsize=font_size)
    axes[1, 0].set_ylabel("Spectral tail mass (%)", fontsize=font_size)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(labels),
        fontsize=font_size,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    plt.show()
    return fig


def _plot_case_metrics(
    case_result: dict[str, Any],
    *,
    font_size: int | float,
    figsize: tuple[float, float],
    combined_figsize: tuple[float, float],
    relative_spectral_figsize: tuple[float, float],
    spectral_figsize: tuple[float, float],
    spectral_ylim: tuple[float, float] | None,
    show_difference: bool,
) -> list[Any | None]:
    lrk_metric_sets = _case_metric_sets(case_result, "lrk")
    xkv_metric_sets = _case_metric_sets(case_result, "xkv")

    figures = [
        _plot_lrk_head_metrics(
            lrk_metric_sets,
            font_size=font_size,
            figsize=figsize,
            show_difference=show_difference,
        ),
        _plot_lrk_layer_mean_metrics(
            lrk_metric_sets,
            font_size=font_size,
            figsize=figsize,
            show_difference=show_difference,
        ),
        _plot_xkv_group_metrics(
            xkv_metric_sets,
            font_size=font_size,
            figsize=figsize,
            show_difference=show_difference,
        ),
        _plot_combined_lrk_xkv_metrics(
            lrk_metric_sets,
            xkv_metric_sets,
            font_size=font_size,
            figsize=combined_figsize,
        ),
        _plot_relative_spectral_tails(
            lrk_metric_sets,
            xkv_metric_sets,
            font_size=font_size,
            figsize=relative_spectral_figsize,
            ylim=spectral_ylim,
        ),
        _plot_spectral_tails(
            lrk_metric_sets,
            xkv_metric_sets,
            font_size=font_size,
            figsize=spectral_figsize,
            ylim=spectral_ylim,
        ),
    ]
    return figures


def display_case_result(
    case_result: dict[str, Any],
    *,
    show_detail_tables: bool = False,
    plot_font_size: int | float = 11,
    plot_figsize: tuple[float, float] = CLUSTERING_PLOT_FIGSIZE,
    combined_plot_figsize: tuple[float, float] = (10, 12),
    relative_spectral_plot_figsize: tuple[float, float] = (10, 5),
    spectral_plot_figsize: tuple[float, float] = (10, 10),
    spectral_ylim: tuple[float, float] | None = None,
    show_bound_error_difference: bool = False,
) -> list[Any | None]:
    from IPython.display import display

    if case_result["summary_df"] is not None:
        display(case_result["summary_df"])
    else:
        print(case_result["summary_rows"])

    figures = _plot_case_metrics(
        case_result,
        font_size=plot_font_size,
        figsize=plot_figsize,
        combined_figsize=combined_plot_figsize,
        relative_spectral_figsize=relative_spectral_plot_figsize,
        spectral_figsize=spectral_plot_figsize,
        spectral_ylim=spectral_ylim,
        show_difference=show_bound_error_difference,
    )

    if not show_detail_tables:
        return figures

    if case_result["summary_df"] is not None:
        display(case_result["lrk_df"])
        display(case_result["xkv_df"])
        if case_result.get("lrk_df_random") is not None:
            display(case_result["lrk_df_random"])
        if case_result.get("xkv_df_random") is not None:
            display(case_result["xkv_df_random"])
        display(case_result["lrk_df_n1"])
        display(case_result["xkv_df_n1"])
        return figures

    print("First LRK result:", case_result["lrk_results"][:1])
    print("First xKV result:", case_result["xkv_results"][:1])
    print("First LRK n=1 result:", case_result["lrk_results_n1"][:1])
    print("First xKV n=1 result:", case_result["xkv_results_n1"][:1])
    return figures
