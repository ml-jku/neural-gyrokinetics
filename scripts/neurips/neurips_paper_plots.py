"""Shared style + save/load helpers for the NeurIPS paper figures.

Style constants are pulled from `notebooks/build_figure1_nb.py` so every panel
reads as part of the same paper aesthetic. Save format is dual: a PDF for the
figure (drop-in for LaTeX) plus a pickle of the underlying numbers (for the
multi-method comparison notebook).
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import matplotlib as mpl
import matplotlib.colors as _mc
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Palette — copied from build_figure1_nb.py so the notebook + paper agree.
# ---------------------------------------------------------------------------
COLOR_FLUX = "#CC7B7B"  # terracotta — flux traces / samples
COLOR_KY = "#8878B8"  # wisteria — W(ky) / kyspec
COLOR_QY = "#6890B5"  # steel blue — Q(ky) / qspec / fluxspec
COLOR_REF = "#202020"  # neutral dark — reference dashed lines
COLOR_GKW = "#505050"
COLOR_GYROSWIN = "#8878B8"  # purple, paper convention
COLOR_QUALIKIZ = "#B088A8"
COLOR_GYROFLOW = "#58A8A0"  # teal — our method
COLOR_TRANSIENT = "#CC7B7B"
COLOR_SATURATED = "#58A8A0"

PALETTE = {
    "flux": COLOR_FLUX,
    "ky": COLOR_KY,
    "ky_spec": COLOR_KY,
    "fluxspec": COLOR_QY,
    "qy": COLOR_QY,
    "ref": COLOR_REF,
    "gkw": COLOR_GKW,
    "gyroswin": COLOR_GYROSWIN,
    "qualikiz": COLOR_QUALIKIZ,
    "gyroflow": COLOR_GYROFLOW,
}

# Per-method colour for cross-method comparison plots. Stable for a given key.
METHOD_PALETTE = {
    "GyroFlow": COLOR_GYROFLOW,
    "GKW": COLOR_GKW,
    "GyroSwin": COLOR_GYROSWIN,
    "QuaLiKiz": COLOR_QUALIKIZ,
    "VAE": "#9c89b8",
    "VQ-VAE": "#7689bb",
    "VQ-VAE+AR": "#6868a8",
    "Diff (5D)": COLOR_GYROFLOW,
    "diff": COLOR_GYROFLOW,
}

GHOST_DARKEN = 0.65
GHOST_ALPHA = 0.4


def darken(color: str, factor: float = GHOST_DARKEN) -> tuple:
    r, g, b = _mc.to_rgb(color)
    return (r * factor, g * factor, b * factor)


def lighten(color: str, factor: float = 0.5) -> tuple:
    r, g, b = _mc.to_rgb(color)
    return (
        r + (1 - r) * factor,
        g + (1 - g) * factor,
        b + (1 - b) * factor,
    )


# ---------------------------------------------------------------------------
# Style: matplotlib rcParams. Matches build_figure1_nb.py defaults.
# ---------------------------------------------------------------------------
PAPER_RCPARAMS = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "none",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#444",
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.color": "#444",
    "ytick.color": "#444",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "lines.linewidth": 1.4,
    "lines.markersize": 4.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.18,
    "grid.linewidth": 0.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def apply_paper_style() -> None:
    """Update matplotlib rcParams to the paper style. Call once per notebook."""
    mpl.rcParams.update(PAPER_RCPARAMS)


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------
def save_fig(
    fig: plt.Figure,
    name: str,
    results_dir: str | os.PathLike,
    formats: Iterable[str] = ("pdf", "png"),
    dpi: int = 200,
) -> Dict[str, Path]:
    """Save a figure to ``<results_dir>/<name>.<ext>`` for each ext in formats.
    Returns the mapping {ext: path} of files written."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Path] = {}
    for ext in formats:
        path = results_dir / f"{name}.{ext}"
        fig.savefig(path, dpi=dpi)
        out[ext] = path
    return out


def save_method_results(
    method: str,
    payload: Mapping[str, Any],
    results_dir: str | os.PathLike,
) -> Path:
    """Pickle a per-method results dict so the comparison notebook can re-load.

    Convention for `payload`:
        {
            "method":   "<canonical name shown in plots>",
            "n_traj":   int,
            "ID_trajs": list[str],
            "OOD_trajs": list[str],
            "fids":     {level: {traj: float}},     # 4 U-Net levels
            "metrics":  {metric_name: {traj: float}},
            "summary":  {metric_name: float},       # aggregated
            "qualitative": {                        # for overview figure
                "warm_eflux":   {traj: np.ndarray},
                "warm_time":    {traj: np.ndarray},
                "gt_eflux":     {traj: np.ndarray},
                "gt_time":      {traj: np.ndarray},
                "warm_kyspec":  {traj: np.ndarray},   # (T, K)
                ...
            },
            "config":   {...},                      # cfg snapshot for reproducibility
        }
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{method}.pkl"
    with open(path, "wb") as f:
        pickle.dump(dict(payload), f)
    return path


def load_method_results(
    results_dir: str | os.PathLike,
    methods: Optional[Iterable[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Load every ``<method>.pkl`` under `results_dir` (or only the named subset)."""
    results_dir = Path(results_dir)
    out: Dict[str, Dict[str, Any]] = {}
    for path in sorted(results_dir.glob("*.pkl")):
        if methods is not None and path.stem not in methods:
            continue
        with open(path, "rb") as f:
            out[path.stem] = pickle.load(f)
    return out


# ---------------------------------------------------------------------------
# Plot primitives — reused across the per-method and comparison notebooks.
# ---------------------------------------------------------------------------
def pretty_scatter(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    color: str = COLOR_GYROFLOW,
    *,
    label: Optional[str] = None,
    diag: bool = True,
    annot: Optional[str] = None,
) -> None:
    """1:1 paper scatter. Optionally draws a y=x diagonal and an annotation."""
    ax.scatter(x, y, color=color, s=18, alpha=0.85, edgecolor="none", label=label)
    if diag:
        lo = float(min(np.nanmin(x), np.nanmin(y)))
        hi = float(max(np.nanmax(x), np.nanmax(y)))
        ax.plot([lo, hi], [lo, hi], color=COLOR_REF, alpha=0.25, lw=0.9, zorder=1)
    if annot:
        ax.text(
            0.04,
            0.96,
            annot,
            transform=ax.transAxes,
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85),
        )


def pretty_grouped_bars(
    ax: plt.Axes,
    df,
    *,
    color_by_col: bool = True,
    palette: Optional[Mapping[str, str]] = None,
    bar_width: float = 0.8,
    group_gap: float = 0.4,
) -> None:
    """Grouped bar chart for a `pandas.DataFrame` indexed by group, columned by
    series. `color_by_col` colours each series consistently across groups."""

    assert hasattr(df, "columns")
    n_groups = len(df.index)
    n_cols = len(df.columns)
    width = bar_width / max(n_cols, 1)
    palette = palette or {}
    base = np.arange(n_groups)
    for i, col in enumerate(df.columns):
        col_color = palette.get(col, METHOD_PALETTE.get(col, f"C{i}"))
        offsets = base + (i - (n_cols - 1) / 2.0) * width
        ax.bar(
            offsets,
            df[col].values,
            width=width * 0.95,
            color=col_color,
            edgecolor="black",
            linewidth=0.4,
            label=col,
        )
    ax.set_xticks(base)
    ax.set_xticklabels(df.index, rotation=20, ha="right")
    ax.legend(fontsize=8, loc="best", ncol=min(n_cols, 4))
