"""Validation plot helpers.

Pure-numpy upper-triangular ND view of axis-pair projections, plus
cross-section panel generation for validation.
"""

from __future__ import annotations

import io
from itertools import combinations
from typing import Optional

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from neugk_jax.utils import recombine_zf

GK_LABELS = {5: [r"v_{\parallel}", r"\mu", r"s", r"k_x", r"k_y"], 3: [r"k_x", r"s", r"k_y"]}


def _force_aspect(ax, aspect: float = 1.0):
    ims = ax.get_images()
    if not ims:
        return
    e = ims[0].get_extent()
    ax.set_aspect(abs((e[1] - e[0]) / (e[3] - e[2])) / aspect)


def _plt_to_wandb_image(fig):
    """Convert a figure to ``wandb.Image`` (or return it unchanged if wandb is
    missing). Closes the figure to free the matplotlib resources."""
    try:
        import wandb
        from PIL import Image as PILImage
    except ImportError:
        return fig
    buf = io.BytesIO()
    fig.savefig(buf, bbox_inches="tight", format="png", dpi=120, pad_inches=0.01)
    buf.seek(0)
    img = PILImage.open(buf)
    plt.close(fig)
    return wandb.Image(img)


def plot_nd(x: np.ndarray, y: Optional[np.ndarray] = None, *, cmap: str = "RdBu_r"):
    """Upper-triangular grid of 2D projections, one per axis pair.

    ``x`` (and optional ``y``) are arrays with shape ``(C?, *spatial)``, numpy or
    device arrays; a 5- or 6-dimensional array carries a leading channel axis. Each
    subplot in the upper triangle averages the non-displayed spatial axes and shows the
    resulting 2D slice; only those slices are brought to host. When ``y`` is provided
    each subplot becomes a side-by-side (pred | gt).
    """
    has_channel = x.ndim in (5, 6)
    ndim = x.ndim - 1 if has_channel else x.ndim
    labels = GK_LABELS.get(ndim, [f"d_{i}" for i in range(ndim)])
    comb = [list(c) for c in combinations(range(ndim), 2)]
    fig, axes = plt.subplots(
        ndim, ndim, figsize=(ndim * (3.5 if y is not None else 2), ndim * 1.8), squeeze=False
    )
    cmap_obj = matplotlib.colormaps[cmap].copy()
    cmap_obj.set_bad("gray")

    def _aggregate(data, other_dims):
        d = data.sum(0) if has_channel and data.ndim > ndim else data
        return np.asarray(d.mean(axis=other_dims))

    for i in range(ndim):
        for j in range(ndim):
            ax = axes[i, j]
            if [i, j] not in comb:
                ax.remove()
                continue
            other = tuple(o for o in range(ndim) if o != i and o != j)
            xx = _aggregate(x, other)
            if y is not None:
                yy = _aggregate(y, other)
                vmin = float(np.nanmin([np.nanmin(xx), np.nanmin(yy)]))
                vmax = float(np.nanmax([np.nanmax(xx), np.nanmax(yy)]))
                spacer = np.full((xx.shape[0], max(1, xx.shape[1] // 15)), np.nan)
                disp = np.concatenate([xx, spacer, yy], axis=1)
                ax.matshow(disp, cmap=cmap_obj, vmin=vmin, vmax=vmax)
            else:
                ax.matshow(xx, cmap=cmap_obj)
            if j == i + 1:
                ax.set_ylabel(rf"${labels[i]}$", fontsize=22, labelpad=2)
            if i == j - 1:
                ax.set_xlabel(rf"${labels[j]}$", fontsize=22, labelpad=2)
            ax.set_xticks([])
            ax.set_yticks([])
            _force_aspect(ax, aspect=2.1 if y is not None else 1.0)

    plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99, wspace=0, hspace=0)
    return fig


def generate_val_plots(
    rollout: dict[str, np.ndarray],
    gt: dict[str, np.ndarray],
    phase: str,
    *,
    ts: Optional[np.ndarray] = None,
) -> dict[str, object]:
    """Cross-section panels for validation.

    ``df`` is plotted with the 5D upper-triangular view (recombines the
    separate-zf channel back to 2-channel first). ``phi`` is plotted as
    its native 3D layout ``(s, k_x, k_y)``.
    """
    plots: dict[str, object] = {}
    time_str = f"T={float(ts[0]):.2f}, " if ts is not None and np.asarray(ts).size > 0 else ""
    field_configs = {
        "df": {"name": f"df ({time_str}{phase})", "recombine": True, "cmap": "RdBu_r"},
        "phi": {"name": f"phi ({time_str}{phase})", "recombine": False, "cmap": "plasma"},
    }
    for key, cfg in field_configs.items():
        if key not in rollout or key not in gt:
            continue
        x, y = rollout[key], gt[key]
        if cfg["recombine"]:
            x, y = recombine_zf(x, axis=0), recombine_zf(y, axis=0)
        x, y = x.squeeze(), y.squeeze()
        plots[cfg["name"]] = _plt_to_wandb_image(plot_nd(x, y, cmap=cfg["cmap"]))
    return plots


def avg_flux_confidence(
    pred_means: np.ndarray, pred_stds: np.ndarray, tgt_vals: np.ndarray, traj_ids: list
):
    """Per-trajectory flux mean ± std vs ground truth."""
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
    ax.scatter(x_pos, tgt_vals, marker="x", s=80, color="#d62728", label="Ground Truth", zorder=3)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(traj_ids, rotation=45, ha="right")
    ax.set_xlabel("Trajectory ID", fontsize=12)
    ax.set_ylabel("Average Flux", fontsize=12)
    ax.set_title("Flux Prediction Accuracy across Trajectories", fontsize=14)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=True, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3, ls="--")
    return _plt_to_wandb_image(fig)
