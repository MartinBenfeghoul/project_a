import os
import math
import torch
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


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
        fmt = ".1f"
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


import json
import pandas as pd

if __name__ == "__main__":  # TODO: move this block to use functions above (namely plot_success_matrix - may need to adjust it)
    path = "./results/needle_mse_long.jsonl"
    save_path = "nll_heatmap_needle_mse.png"
    colums = "num_token_per_training"
    #values = "avg_nll_change_perc"
    values = "avg_accuracy_modified_cache"
    index = "percentage_changed_kv"
    ds = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            new_l = json.loads(line)
            new_l["percentage_changed_kv"] = round(new_l["percentage_changed_kv"])
            new_l["avg_accuracy_modified_cache"] = new_l["avg_accuracy_modified_cache"] * 100
            ds.append(new_l)
    if values == "avg_accuracy_modified_cache": vmin, vmax = 70, 98
    else: vmin, vmax = 2.70, 3
    df = pd.DataFrame(ds)
    heatmap_df = df.pivot(
        index=index,
        columns=colums,
        values=values,
    )
    plt.figure(figsize=(10, 6))

    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        vmin=vmin, vmax=vmax,
    )

    plt.title("NLL with updated KV cache", fontsize=16)
    plt.xlabel("Sequence length")
    plt.ylabel("Percentage KV changed")
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
