import matplotlib.pyplot as plt
import math
import torch

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
        figsize=(figsize_per_plot[0] * ncols,
                 figsize_per_plot[1] * nrows),
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
        wspace=0.25,   # horizontal gap
        hspace=0.35,   # vertical gap
    )

    cbar = fig.colorbar(ims[0], cax=cax)
    cbar.set_label("Energy")

    plt.show()