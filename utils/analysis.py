import os
import json
import math
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

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


def load_results(results_file):
    results = []
    with open(results_file, "r") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return pd.DataFrame(results)


def snap_to_bucket(n):
    return min(SEQ_LENGTH_BUCKETS, key=lambda b: abs(b - n))


def plot_energy_at_rank_k(energy, k):
    energy = energy.clone()
    if energy.dim() > 3:
        energy = energy.mean(0)  # average over batch size
    energy = energy[..., k]

    # plot a heatmap of the energies
    plt.imshow(energy.cpu().numpy())
    plt.colorbar()
    plt.xlabel("Heads")
    plt.ylabel("Layers")
    plt.show()


def plot_energy_at_ranks(
    energy,
    ranks,
    comp_ratios,
    energy_threshold=None,
    ncols=None,
    figsize_per_plot=(3, 6),
    cmap="viridis",
):
    energy = energy.clone()
    if energy.dim() > 3:
        energy = energy.mean(0)

    if energy_threshold is not None:
        energy = energy.masked_fill(energy < energy_threshold, 0.0)

    n = len(ranks)
    if ncols is None:
        ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)

    # --- compute shared color scale ---
    stacked = torch.stack([energy[..., k] for k in ranks])
    vmin = stacked.min().item()
    vmax = stacked.max().item()

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows),
        squeeze=False,
    )

    ims = []
    for ax, (i, k) in zip(axes.flat, enumerate(ranks)):
        im = ax.imshow(
            energy[..., k].cpu().numpy(),
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            aspect="auto",
        )
        ims.append(im)
        ax.set_title(f"Rank {k}, CR {comp_ratios[i]:.1f}")
        ax.set_xlabel("Heads")
        ax.set_ylabel("Layers")

    for ax in axes.flat[n:]:
        ax.axis("off")

    # --- single shared colorbar ---
    # create dedicated colorbar axis (outside the grid)
    cax = fig.add_axes([0.90, 0.15, 0.02, 0.7])  # [left, bottom, width, height]

    fig.subplots_adjust(
        wspace=0.25,  # horizontal gap
        hspace=0.35,  # vertical gap
    )

    cbar = fig.colorbar(ims[0], cax=cax)
    cbar.set_label("Energy")

    plt.show()


def get_unique_save_path(save_path):
    if not os.path.exists(save_path.format("")):
        return save_path.format("")
    for i in range(100):
        new_path = save_path.format(f"_{i}")
        if not os.path.exists(new_path):
            return new_path
    raise ValueError(
        f"There appears to be at least 100 numbered variations of {save_path}!"
    )


def plot_success_matrix(
    success_matrix,
    seq_lens,
    x_key,
    x_values,
    cache_type,
    save_path="NIAH_ablations{}.png",
    crs=None,
):
    if crs is None:
        annot = True
        fmt = ".2f"
    else:
        annot = np.empty_like(success_matrix, dtype=object)
        for i in range(success_matrix.shape[0]):
            for j in range(success_matrix.shape[1]):
                annot[i, j] = f"{success_matrix[i, j]:.2f}\n({crs[i, j]:.2f})"
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

    save_path = get_unique_save_path(save_path)
    fig.savefig(save_path, dpi=300)


def plot_needle_results(
    path="./results/needle_mse_long.jsonl",
    save_path="nll_heatmap_needle_mse.png",
    columns="num_token_per_training",
    values="avg_accuracy_modified_cache",
    index="percentage_changed_kv",
):
    ds = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            new_l = json.loads(line)
            new_l["percentage_changed_kv"] = round(
                new_l["percentage_changed_kv"]
            )
            new_l["avg_accuracy_modified_cache"] = (
                new_l["avg_accuracy_modified_cache"] * 100
            )
            ds.append(new_l)

    if values == "avg_accuracy_modified_cache":
        vmin, vmax = 70, 98
    else:
        vmin, vmax = 2.70, 3

    df = pd.DataFrame(ds)
    heatmap_df = df.pivot(index=index, columns=columns, values=values)

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        heatmap_df, annot=True, fmt=".3f", cmap="YlOrRd", vmin=vmin, vmax=vmax
    )
    plt.title("NLL with updated KV cache", fontsize=16)
    plt.xlabel("Sequence length")
    plt.ylabel("Percentage KV changed")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_heatmap(df, output_path, title=None):
    df = df.copy()
    df["seq_length"] = df["num_token"].apply(snap_to_bucket)

    pivot = df.pivot(
        index="avg_nll_change_perc",
        columns="seq_length",
        values="avg_accuracy_modified_cache",
    )
    pivot = pivot.sort_index(ascending=True)
    pivot = pivot[sorted(pivot.columns)]

    pivot.columns = [SEQ_LENGTH_LABELS[c] for c in pivot.columns]

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


def parse_args():
    parser = argparse.ArgumentParser(description="Visualise experiment results")
    parser.add_argument(
        "-f",
        "--file_path",
        type=str,
        required=True,
        help="Path to the JSONL results file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output path for the plot (default: results_file stem + .png)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Custom title for the plot",
    )
    parser.add_argument(
        "--plot",
        type=str,
        choices=PLOTS,
        default="heatmap",
        help=f"Which plot to produce (default: heatmap). Choices: {PLOTS}",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    output_path = args.output or Path(args.file_path).stem + ".png"

    if args.plot == "heatmap":
        df = load_results(args.file_path)
        plot_heatmap(df, output_path, args.title)
    elif args.plot == "needle":
        plot_needle_results(path=args.file_path, save_path=output_path)


if __name__ == "__main__":
    main()
