"""Qualitative visualization functions for 5D and 3D data."""

from typing import Dict, Optional

import io
from PIL import Image as PILImage
import wandb
import warnings

import matplotlib
import matplotlib.ticker as tkr
import matplotlib.pyplot as plt

import numpy as np
import torch


def force_aspect(ax, aspect=1):
    """Adjust axis aspect ratio based on image extent."""
    im = ax.get_images()
    extent = im[0].get_extent()
    ax.set_aspect(abs((extent[1] - extent[0]) / (extent[3] - extent[2])) / aspect)


def plt_to_wandb_image(fig):
    """Convert matplotlib figure to wandb Image object."""
    buf = io.BytesIO()
    fig.savefig(buf, bbox_inches="tight")
    buf.seek(0)
    img = PILImage.open(buf)
    plt.close(fig)
    return wandb.Image(img)


def plot4x4_sided(x1, x2, title="", mark_bad=False, average=True):
    """Plot prediction and ground truth side-by-side for all pairs of 5D dimensions."""
    labels = [r"v_{par}", r"v_{\mu}", r"s", r"k_x", r"k_y"]
    comb = torch.combinations(torch.arange(5), 2).tolist()

    # initialize grid
    fig, ax = plt.subplots(5, 5, figsize=(30, 14))
    for i in range(5):
        for j in range(5):
            if j == 0:
                ax[i, j].remove()
                continue
            if i == 4:
                ax[i, j].remove()
                continue
            ax_ij = ax[i, j]
            ax_ij.set_frame_on(False)
            ax_ij.tick_params(labelleft=False, labelbottom=False)
            ax_ij.set_xticks([])
            ax_ij.set_yticks([])

    fig.suptitle(title)
    c_map = matplotlib.colormaps["RdBu"]
    c_map.set_bad("k")

    # draw comparisons
    for i, j in comb:
        other = tuple([o for o in range(5) if o != i and o != j])

        # extract slices
        if average:
            x1_plot = x1[0].mean(other)
            x2_plot = x2[0].mean(other)
        else:
            x1_plot = torch.tensor(x1[0]).permute(i, j, *other).numpy()[:, :, 0, 0, 0]
            x2_plot = torch.tensor(x2[0]).permute(i, j, *other).numpy()[:, :, 0, 0, 0]

        # optional masking
        if mark_bad:
            x1_std = x1.std(other)
            x2_std = x2.std(other)
            x1_plot[x1_std == 0] = np.nan
            x2_plot[x2_std == 0] = np.nan

        ax_ij = ax[i, j]
        pos = ax_ij.get_position()

        # sub-axes layout
        plot_width = 0.475 * pos.width
        left_margin = 0.0 * pos.width
        x_left_1 = pos.x0 + left_margin
        x_left_2 = x_left_1 + plot_width
        y = pos.y0
        h = pos.height
        ax1 = fig.add_axes([x_left_1, y, plot_width, h])
        ax2 = fig.add_axes([x_left_2, y, plot_width, h])

        # scale colorbar
        vmin = min(x1_plot.min(), x2_plot.min())
        vmax = max(x1_plot.max(), x2_plot.max())

        im1 = ax1.matshow(x1_plot, cmap=c_map, vmin=vmin, vmax=vmax)
        ax2.matshow(x2_plot, cmap=c_map, vmin=vmin, vmax=vmax)

        # decorations
        cbar = fig.colorbar(
            im1, ax=[ax_ij], format=tkr.FormatStrFormatter("%.2g"), pad=0, fraction=0.05
        )
        cbar.set_ticks([vmin, (vmin + vmax) / 2, vmax])
        cbar.ax.tick_params(labelsize=12)

        if i == 0:
            ax1.set_title(r"PRED", fontsize=24)
            ax2.set_title(r"GT", fontsize=24)

        if j == 1 or (i == 1 and j == 2) or (i == 2 and j == 3) or (i == 3 and j == 4):
            ax_ij.set_ylabel(rf"${labels[i]}$", fontsize=14)

        if i == 3 or j == 1 or (i == 1 and j == 2) or (i == 2 and j == 3):
            ax_ij.set_xlabel(rf"${labels[j]}$", fontsize=14)

        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax1.tick_params(labelleft=False, labelbottom=False)
        ax2.tick_params(labelleft=False, labelbottom=False)
        force_aspect(ax1)
        force_aspect(ax2)

    return plt_to_wandb_image(fig)


def avg_flux_confidence(pred_means, pred_stds, tgt_vals, traj_ids):
    """Plot average flux predictions with confidence intervals across trajectories."""
    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(traj_ids))
    # error bars
    ax.errorbar(
        x_pos,
        pred_means,
        yerr=pred_stds,
        fmt="o",
        capsize=5,
        label="predicted (mean ± std)",
        color="blue",
        alpha=0.7,
    )
    # truth markers
    ax.plot(x_pos, tgt_vals, "rx", markersize=8, label="ground truth", zorder=3)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(traj_ids, rotation=45)
    ax.set_xlabel("trajectory id")
    ax.set_ylabel("average flux")
    ax.set_title("average flux predictions per trajectory")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return plt_to_wandb_image(fig)


def generate_val_plots(
    rollout: Dict[str, torch.Tensor],
    gt: torch.Tensor,
    phase: str,
    ts: Optional[torch.Tensor] = None,
):
    """Generate all registered validation plots for a given rollout."""
    plots = {}
    ts = f"T={ts[0].item():.2f}, " if ts is not None else ""
    val_plots_dict = {
        "df": {f"df ({ts}{phase})": plot4x4_sided},
    }

    # iterate over fields
    for key in rollout.keys():
        if key not in val_plots_dict:
            continue

        # normalize keys
        gt_key = key
        if "int" in key:
            gt_key = key.replace("_int", "")

        x = rollout[key].clone()
        y = gt[gt_key].clone()

        # ensure sequence dimension
        if x.ndim != 7:
            x = x.unsqueeze(0)

        # process zonal flow
        if y.shape[0] != 2 and key == "df":
            y = torch.cat(
                [
                    y[0::2].sum(axis=0, keepdims=True),
                    y[1::2].sum(axis=0, keepdims=True),
                ],
                dim=0,
            )

        if x.shape[1] != 2 and key == "df":
            x = torch.cat(
                [
                    x[:, 0::2].sum(axis=1, keepdims=True),
                    x[:, 1::2].sum(axis=1, keepdims=True),
                ],
                dim=1,
            )

        # execute plot functions
        for name, plot_fn in val_plots_dict[key].items():
            plots[name] = plot_fn(x[0], x2=y)

    return plots
