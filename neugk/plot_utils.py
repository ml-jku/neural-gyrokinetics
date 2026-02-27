"""Qualitative visualization functions for n-dimensional gyrokinetics data."""

import io
from typing import Dict, List, Optional, Sequence, Union

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import numpy as np
import torch
from PIL import Image as PILImage

import wandb
from neugk.utils import recombine_zf

# Standard labels for gyrokinetics dimensions
GK_LABELS = {
    6: [r"t", r"v_{\parallel}", r"\mu", r"s", r"k_x", r"k_y"],
    5: [r"v_{\parallel}", r"\mu", r"s", r"k_x", r"k_y"],
    4: [r"v_{\parallel}", r"s", r"k_x", r"k_y"],
    3: [r"k_x", r"s", r"k_y"],
}


def force_aspect(ax: plt.Axes, aspect: float = 1.0):
    """Adjust axis aspect ratio based on image extent."""
    im = ax.get_images()
    if not im:
        return
    extent = im[0].get_extent()
    ax.set_aspect(abs((extent[1] - extent[0]) / (extent[3] - extent[2])) / aspect)


def plt_to_wandb_image(fig: plt.Figure) -> wandb.Image:
    """Convert matplotlib figure to wandb Image object and close it."""
    buf = io.BytesIO()
    fig.savefig(buf, bbox_inches="tight", format="png", dpi=120, pad_inches=0.01)
    buf.seek(0)
    img = PILImage.open(buf)
    plt.close(fig)
    return wandb.Image(img)


def plot_nd(
    x: Union[np.ndarray, torch.Tensor],
    y: Optional[Union[np.ndarray, torch.Tensor]] = None,
    labels: Optional[List[str]] = None,
    cmap: str = "RdBu_r",
    aggregate: str = "mean",
    to_wandb: bool = False,
    aspect: float = 1.0,
    mark_bad: bool = False,
    **kwargs,
):
    """
    Generic n-dimensional plotting function.
    Creates a grid of 2D slices for all combinations of dimensions.
    If 'y' is provided, shows side-by-side comparison.
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if y is not None and isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()

    # Detect spatial dimensions (ndim) and channel dimension
    # If labels are provided, they define the spatial dims.
    if labels is not None:
        ndim = len(labels)
        has_channel = x.ndim > ndim
    else:
        # heuristic: if 5D/6D, assume (C, ...)
        if x.ndim in [5, 6]:
            ndim = x.ndim - 1
            has_channel = True
        else:
            ndim = x.ndim
            has_channel = False

    if ndim == 0:
        return None

    if labels is None:
        labels = GK_LABELS.get(ndim, [f"d_{i}" for i in range(ndim)])

    comb = list(torch.combinations(torch.arange(ndim), 2))
    comb_list = [c.tolist() for c in comb]

    fig, axes = plt.subplots(
        ndim,
        ndim,
        figsize=(ndim * (3.5 if y is not None else 2), ndim * 1.8),
        squeeze=False,
    )

    c_map = matplotlib.colormaps[cmap].copy()
    c_map.set_bad("gray")

    for i in range(ndim):
        for j in range(ndim):
            ax = axes[i, j]
            if [i, j] not in comb_list:
                ax.remove()
                continue

            other_dims = tuple(o for o in range(ndim) if o != i and o != j)

            def get_2d_slice(data):
                # Handle channel dimension explicitly
                d = data.sum(0) if has_channel and data.ndim > ndim else data

                # mean/std/slice over the 'other' spatial dims
                if aggregate == "mean":
                    res = d.mean(axis=other_dims)
                elif aggregate == "std":
                    res = d.std(axis=other_dims)
                elif aggregate == "slice":
                    slices = [slice(None)] * ndim
                    for o in other_dims:
                        slices[o] = d.shape[o] // 2
                    res = d[tuple(slices)]
                else:
                    res = d.mean(axis=other_dims)

                if mark_bad:
                    s = d.std(axis=other_dims)
                    res = np.where(s == 0, np.nan, res)
                return res

            xx = get_2d_slice(x)

            if y is not None:
                yy = get_2d_slice(y)
                vmin = min(np.nanmin(xx), np.nanmin(yy))
                vmax = max(np.nanmax(xx), np.nanmax(yy))

                spacer = np.full((xx.shape[0], max(1, xx.shape[1] // 15)), np.nan)
                display_img = np.concatenate([xx, spacer, yy], axis=1)
                im = ax.matshow(display_img, cmap=c_map, vmin=vmin, vmax=vmax)
            else:
                im = ax.matshow(xx, cmap=c_map)

            # Labels only on row/column boundaries for clarity
            # Y-label on the first plot of each row
            if j == i + 1:
                ax.set_ylabel(rf"${labels[i]}$", fontsize=22, labelpad=2)
            # X-label on the last plot of each column (which is on the bottom of the grid)
            if i == j - 1:
                ax.set_xlabel(rf"${labels[j]}$", fontsize=22, labelpad=2)

            ax.set_xticks([])
            ax.set_yticks([])

            force_aspect(ax, aspect=aspect * (2.1 if y is not None else 1.0))

    plt.subplots_adjust(
        left=0.01,
        right=0.99,
        bottom=0.01,
        top=0.99,
        wspace=0,
        hspace=0,
    )

    if to_wandb:
        return plt_to_wandb_image(fig)
    return fig


def plot_diag(
    gt_diag: Sequence, pred_diag: Sequence, loglog: bool = True, to_wandb: bool = False
):
    """Plot spectral diagnostics comparison."""
    n_cols = len(pred_diag)
    fig, axes = plt.subplots(
        3, n_cols, figsize=(4 * n_cols, 10), squeeze=False, constrained_layout=True
    )

    plasma = matplotlib.colormaps["plasma"]
    metrics = ["qspec", "kyspec", "kxspec"]

    for col, (gtd, pd) in enumerate(zip(gt_diag, pred_diag)):
        if not isinstance(pd, (list, tuple)):
            pd, gtd = [pd], [gtd]

        n_lines = len(pd)
        for t in range(n_lines):
            color = plasma((t + 0.1) / max(1, n_lines))
            for row, metric in enumerate(metrics):
                ax = axes[row, col]
                if metric in pd[t]:
                    ax.plot(
                        pd[t][metric][1:],
                        lw=1.5,
                        color=color,
                        label="pred" if t == 0 else None,
                    )
                if metric in gtd[t]:
                    ax.plot(
                        gtd[t][metric][1:],
                        lw=1.5,
                        color=color,
                        alpha=0.5,
                        ls="--",
                        label="gt" if t == 0 else None,
                    )

                if loglog:
                    ax.set_xscale("log")
                    ax.set_yscale("log")

                if col == 0:
                    ax.set_ylabel(metric, fontsize=12)
                if row == 0:
                    ax.set_title(f"Sample {col}", fontsize=12)

    if to_wandb:
        return plt_to_wandb_image(fig)
    return fig


def avg_flux_confidence(
    pred_means: np.ndarray,
    pred_stds: np.ndarray,
    tgt_vals: np.ndarray,
    traj_ids: List[str],
    to_wandb: bool = True,
):
    """Plot average flux predictions with confidence intervals."""
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    x_pos = np.arange(len(traj_ids))

    ax.errorbar(
        x_pos,
        pred_means,
        yerr=pred_stds,
        fmt="o",
        capsize=6,
        label="Predicted (Mean ± Std)",
        color="#1f77b4",
        mfc="white",
        mew=2,
        alpha=0.8,
    )
    ax.scatter(
        x_pos,
        tgt_vals,
        marker="x",
        s=80,
        color="#d62728",
        label="Ground Truth",
        zorder=3,
    )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(traj_ids, rotation=45, ha="right")
    ax.set_xlabel("Trajectory ID", fontsize=12)
    ax.set_ylabel("Average Flux", fontsize=12)
    ax.set_title("Flux Prediction Accuracy across Trajectories", fontsize=14)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=True, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3, ls="--")

    if to_wandb:
        return plt_to_wandb_image(fig)
    return fig


def generate_val_plots(
    rollout: Dict[str, torch.Tensor],
    gt: torch.Tensor,
    phase: str,
    ts: Optional[torch.Tensor] = None,
):
    """Generate standardized validation plots for a rollout."""
    plots = {}
    time_str = f"T={ts[0].item():.2f}, " if ts is not None else ""

    # map field names to plotting logic
    field_configs = {
        "df": {"name": f"df ({time_str}{phase})", "recombine": True, "cmap": "RdBu_r"},
        "phi": {
            "name": f"phi ({time_str}{phase})",
            "recombine": False,
            "cmap": "plasma",
        },
    }

    for key, config in field_configs.items():
        if key not in rollout:
            continue

        x = rollout[key].clone()
        y = gt[key].clone()

        # ensure sequence dimension exists
        if x.ndim != 7:
            x = x.unsqueeze(0)

        # handle zonal flow recombination if needed
        if config["recombine"]:
            if y.shape[0] != 2:
                y = recombine_zf(y, dim=0)
            if x.shape[1] != 2:
                x = recombine_zf(x, dim=1)

        # use the generic nD plotter
        x, y = x[0].squeeze(), y.squeeze()
        plots[config["name"]] = plot_nd(x, y, to_wandb=True, cmap=config["cmap"])

    return plots
