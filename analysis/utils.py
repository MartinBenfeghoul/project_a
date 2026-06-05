from __future__ import annotations

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
    reconstruct_segments,
)
from utils.rope import compute_rope_cos_sin, inverse_rope
from utils.segmentation import (
    build_cluster_segment_ranges,
    group_keys_by_cluster,
    group_sequences_by_cluster,
    kmeans_cluster_sequences,
)

HIDDEN_COLUMNS = ("batch_idx", "seq_len", "compressed_len")
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
    ("eta", "bound"),
    ("relative_low_rank_recon_error", "error"),
)
CLUSTERING_METRIC_COLORS = {
    "eta": "tab:blue",
    "relative_low_rank_recon_error": "tab:orange",
}
CLUSTERING_SET_STYLES = {
    "Clustered": {"linestyle": "-", "marker": "o"},
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
    rope_theta = float(kvs.get("rope_theta", default_rope_theta))
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


def _cluster_items_and_ranges(
    features: torch.Tensor,
    n_clusters: int,
    kmeans_n_iter: int,
    kmeans_init: str,
    kmeans_dtype: torch.dtype,
    *,
    kmeans_cluster_size: int | None = None,
    cluster_axis: str = "rows",
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


def _merge_metric_lists(
    *metric_lists: list[dict[str, float]],
) -> list[dict[str, float]]:
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
            metrics = _merge_metric_lists(scatter_metrics, recon_metrics)
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
            layer_metrics = _merge_metric_lists(
                scatter_metrics,
                layer_recon_metrics,
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
        layer_metrics = _merge_metric_lists(
            scatter_metrics,
            layer_recon_metrics,
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
        group_metrics = _merge_metric_lists(
            group_scatter_metrics,
            group_recon_metrics,
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
    lrk_df=None,
    xkv_df=None,
    lrk_df_n1=None,
    xkv_df_n1=None,
) -> list[dict[str, Any]]:
    summary_inputs = [
        (f"{tensor_name}_lrk_{cluster_axis}", lrk_records, lrk_df),
        (f"{tensor_name}_xkv_{cluster_axis}", xkv_records, xkv_df),
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

    summary_rows = _build_case_summary_rows(
        tensor_name,
        cluster_axis,
        lrk_results,
        xkv_results,
        lrk_results_n1,
        xkv_results_n1,
        lrk_df=lrk_df,
        xkv_df=xkv_df,
        lrk_df_n1=lrk_df_n1,
        xkv_df_n1=xkv_df_n1,
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
        "lrk_df": lrk_df,
        "xkv_df": xkv_df,
        "lrk_df_n1": lrk_df_n1,
        "xkv_df_n1": xkv_df_n1,
    }


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
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        fontsize=font_size,
        frameon=False,
    )


def _line_style_kwargs(row_label: str, metric_name: str) -> dict[str, Any]:
    style = {
        "color": CLUSTERING_METRIC_COLORS[metric_name],
        **CLUSTERING_SET_STYLES.get(
            row_label,
            CLUSTERING_SET_STYLES["Clustered"],
        ),
    }
    return style


def _plot_lrk_head_metrics(
    metric_sets: list[tuple[str, list[dict[str, Any]]]],
    *,
    font_size: int | float,
    figsize: tuple[float, float],
) -> None:
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
    fig, ax = plt.subplots(figsize=figsize)

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
    ax.set_ylabel("Metric value", fontsize=font_size)
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

    fig.tight_layout(rect=(0, 0, 1, 0.86))
    plt.show()


def _plot_lrk_layer_mean_metrics(
    metric_sets: list[tuple[str, list[dict[str, Any]]]],
    *,
    font_size: int | float,
    figsize: tuple[float, float],
) -> None:
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
    fig, ax = plt.subplots(figsize=figsize)

    for row_label, df in frame_sets:
        x_positions = list(range(len(df)))
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
                alpha=0.12,
                linewidth=0,
            )

    x_labels = [f"L{int(row.layer_idx)}" for row in base_df.itertuples()]
    ax.set_xlabel("Layer", fontsize=font_size)
    ax.set_ylabel("Metric value", fontsize=font_size)
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

    fig.tight_layout(rect=(0, 0, 1, 0.86))
    plt.show()


def _plot_xkv_group_metrics(
    metric_sets: list[tuple[str, list[dict[str, Any]]]],
    *,
    font_size: int | float,
    figsize: tuple[float, float],
) -> None:
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
    fig, ax = plt.subplots(figsize=figsize)

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
    ax.set_ylabel("Metric value", fontsize=font_size)
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

    fig.tight_layout(rect=(0, 0, 1, 0.86))
    plt.show()


def _plot_case_metrics(
    case_result: dict[str, Any],
    *,
    font_size: int | float,
    figsize: tuple[float, float],
) -> None:
    lrk_metric_sets = [
        ("Clustered", case_result["lrk_results"]),
        ("Unclustered", case_result["lrk_results_n1"]),
    ]
    xkv_metric_sets = [
        ("Clustered", case_result["xkv_results"]),
        ("Unclustered", case_result["xkv_results_n1"]),
    ]

    _plot_lrk_head_metrics(
        lrk_metric_sets,
        font_size=font_size,
        figsize=figsize,
    )
    _plot_lrk_layer_mean_metrics(
        lrk_metric_sets,
        font_size=font_size,
        figsize=figsize,
    )
    _plot_xkv_group_metrics(
        xkv_metric_sets,
        font_size=font_size,
        figsize=figsize,
    )


def display_case_result(
    case_result: dict[str, Any],
    *,
    show_detail_tables: bool = False,
    plot_font_size: int | float = 11,
    plot_figsize: tuple[float, float] = CLUSTERING_PLOT_FIGSIZE,
) -> None:
    from IPython.display import display

    if case_result["summary_df"] is not None:
        display(case_result["summary_df"])
    else:
        print(case_result["summary_rows"])

    _plot_case_metrics(
        case_result,
        font_size=plot_font_size,
        figsize=plot_figsize,
    )

    if not show_detail_tables:
        return

    if case_result["summary_df"] is not None:
        display(case_result["lrk_df"])
        display(case_result["xkv_df"])
        display(case_result["lrk_df_n1"])
        display(case_result["xkv_df_n1"])
        return

    print("First LRK result:", case_result["lrk_results"][:1])
    print("First xKV result:", case_result["xkv_results"][:1])
    print("First LRK n=1 result:", case_result["lrk_results_n1"][:1])
    print("First xKV n=1 result:", case_result["xkv_results_n1"][:1])
