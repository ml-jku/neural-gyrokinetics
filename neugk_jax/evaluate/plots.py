"""Validation plot helpers.

Pure-numpy port of ``neugk.plot_utils.plot_nd`` + ``generate_val_plots``.
The same upper-triangular ND view of axis-pair projections the torch
pipeline produces, no upstream import.
"""

from __future__ import annotations

import io
from itertools import combinations
from typing import Optional

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from neugk_jax.utils import recombine_zf as _recombine_zf


GK_LABELS = {
    6: [r"t", r"v_{\parallel}", r"\mu", r"s", r"k_x", r"k_y"],
    5: [r"v_{\parallel}", r"\mu", r"s", r"k_x", r"k_y"],
    4: [r"v_{\parallel}", r"s", r"k_x", r"k_y"],
    3: [r"k_x", r"s", r"k_y"],
}


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
        buf = io.BytesIO()
        fig.savefig(buf, bbox_inches="tight", format="png", dpi=120, pad_inches=0.01)
        buf.seek(0)
        img = PILImage.open(buf)
        plt.close(fig)
        return wandb.Image(img)
    except Exception:
        return fig


def plot_nd(
    x: np.ndarray,
    y: Optional[np.ndarray] = None,
    *,
    labels: Optional[list[str]] = None,
    cmap: str = "RdBu_r",
    aggregate: str = "mean",
    aspect: float = 1.0,
    mark_bad: bool = False,
    to_wandb: bool = False,
):
    """Upper-triangular grid of 2D projections, one per axis pair.

    ``x`` (and optional ``y``) are arrays with shape ``(C?, *spatial)``.
    Each subplot in the upper triangle aggregates the non-displayed
    spatial axes (default: mean) and shows the resulting 2D slice. When
    ``y`` is provided each subplot becomes a side-by-side (pred | gt).
    """
    x = np.asarray(x)
    if y is not None:
        y = np.asarray(y)

    # detect spatial dims + optional leading channel
    if labels is not None:
        ndim = len(labels)
        has_channel = x.ndim > ndim
    else:
        if x.ndim in (5, 6):
            ndim = x.ndim - 1
            has_channel = True
        else:
            ndim = x.ndim
            has_channel = False

    if ndim < 2:
        # 1D: just plot a line
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(x.ravel(), label="x")
        if y is not None:
            ax.plot(y.ravel(), label="y", linestyle="--")
            ax.legend()
        return _plt_to_wandb_image(fig) if to_wandb else fig

    if labels is None:
        labels = GK_LABELS.get(ndim, [f"d_{i}" for i in range(ndim)])

    comb = [list(c) for c in combinations(range(ndim), 2)]
    fig, axes = plt.subplots(
        ndim, ndim,
        figsize=(ndim * (3.5 if y is not None else 2), ndim * 1.8),
        squeeze=False,
    )
    cmap_obj = matplotlib.colormaps[cmap].copy()
    cmap_obj.set_bad("gray")

    def _aggregate(data, other_dims):
        d = data.sum(0) if has_channel and data.ndim > ndim else data
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
            ax.set_xticks([]); ax.set_yticks([])
            _force_aspect(ax, aspect=aspect * (2.1 if y is not None else 1.0))

    plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99, wspace=0, hspace=0)
    return _plt_to_wandb_image(fig) if to_wandb else fig


def generate_val_plots(
    rollout: dict[str, np.ndarray],
    gt: dict[str, np.ndarray],
    phase: str,
    *,
    ts: Optional[np.ndarray] = None,
    to_wandb: bool = True,
) -> dict[str, object]:
    """Cross-section panels — port of ``neugk.plot_utils.generate_val_plots``.

    ``df`` is plotted with the 5D upper-triangular view (recombines the
    separate-zf channel back to 2-channel first). ``phi`` is plotted as
    its native 3D layout ``(s, k_x, k_y)``.
    """
    plots: dict[str, object] = {}
    time_str = f"T={float(ts[0]):.2f}, " if ts is not None and np.asarray(ts).size > 0 else ""
    field_configs = {
        "df":  {"name": f"df ({time_str}{phase})",  "recombine": True,  "cmap": "RdBu_r"},
        "phi": {"name": f"phi ({time_str}{phase})", "recombine": False, "cmap": "plasma"},
    }
    for key, cfg in field_configs.items():
        if key not in rollout or key not in gt:
            continue
        x = np.asarray(rollout[key])
        y = np.asarray(gt[key])
        if cfg["recombine"]:
            if y.shape[0] != 2:
                y = _recombine_zf(y, axis=0)
            axis = 1 if x.ndim == 7 else 0
            if x.shape[axis] != 2:
                x = _recombine_zf(x, axis=axis)
        if x.ndim == 7:
            x = x[0]
        x = np.squeeze(x); y = np.squeeze(y)
        fig = plot_nd(x, y, cmap=cfg["cmap"])
        plots[cfg["name"]] = _plt_to_wandb_image(fig) if to_wandb else fig
    return plots
