"""Figures from per-snapshot rows (matplotlib; `pip install -e ".[paper]"`).

`scatter_figure`: field PSNR against flux error, one point per snapshot and method, the
medians marked.  `flux_spectrum_figure`: the heat-flux spectrum of one snapshot over the
binormal modes for a few methods beside the reference.  From `results/experiments/`:
`rd_figure` (the rate-distortion sweep by bytes), `paired_figure` (GyroSplat against NF +
physics per snapshot, fitting seeds averaged), `heldout_figure` (flux fitted on half of the
modes) and `restart_figure` (solver restarts from reconstructed snapshots).
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

from neugk.gyrosplat.eval.paper.rows import METHODS, TABLE1_ORDER, medians, rows_of, select

DEFAULT_METHODS = (
    "gyrosplat",
    "splats",
    "neural_field_physics",
    "fourier",
    "fourier_objective",
    "wavelet_bior4.4",
    "pca",
)


def scatter_figure(
    rows: Iterable[dict],
    tier: int = 302,
    methods: Sequence[str] = DEFAULT_METHODS,
    ax=None,
):
    import matplotlib.pyplot as plt

    rows = list(rows)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
    else:
        fig = ax.figure
    for m in methods:
        g = rows_of(rows, tier, m)
        if not g:
            continue
        x = [max(r["flux_rel_l1"], 1e-9) for r in g]
        y = [r["psnr_f"] for r in g]
        pts = ax.scatter(x, y, s=12, alpha=0.45, label=METHODS[m])
        med = medians(g)
        ax.scatter(
            [max(med["flux_rel_l1"], 1e-9)],
            [med["psnr_f"]],
            s=90,
            marker="D",
            edgecolor="black",
            color=pts.get_facecolor()[0],
            zorder=5,
        )
    ax.set_xscale("log")
    ax.set_xlabel("heat-flux relative L1 error (nonzero modes)")
    ax.set_ylabel("distribution PSNR (dB)")
    ax.set_title(f"CR ≈ {tier}×: one point per snapshot, diamonds mark the medians")
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    return fig


def flux_spectrum_figure(
    rows: Iterable[dict],
    snapshot: str,
    tier: int = 302,
    methods: Sequence[str] = ("gyrosplat", "fourier", "wavelet_bior4.4"),
    ax=None,
):
    import matplotlib.pyplot as plt

    rows = list(rows)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
    else:
        fig = ax.figure
    truth: Optional[list] = None
    for m in methods:
        g = select(rows, tier=tier, method=m, snapshot=snapshot)
        if not g:
            continue
        r = g[0]
        truth = truth or r["flux_truth"]
        modes = range(1, len(r["flux_pred"]))
        ax.plot(modes, r["flux_pred"][1:], marker="o", ms=3, lw=1, label=METHODS[m])
    if truth is not None:
        ax.plot(
            range(1, len(truth)), truth[1:], color="black", lw=1.8, label="reference"
        )
    ax.set_xlabel("binormal mode")
    ax.set_ylabel("heat flux per mode")
    ax.set_title(f"{snapshot}, CR ≈ {tier}×")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ the experiments
# colour, marker, line style: solid lines for the codes fitted with a physics objective
STYLE = {
    "gyrosplat": ("#D55E00", "*", "-"),
    "splats": ("#5D3A9B", "^", "--"),
    "neural_field_physics": ("#009E73", "s", "-"),
    "neural_field": ("#009E73", "o", "--"),
    "fourier_objective": ("#666666", "s", "-"),
    "fourier": ("#666666", "o", "--"),
    "wavelet_haar": ("#E69F00", "o", "--"),
    "wavelet_bior4.4": ("#E69F00", "D", "--"),
    "pca": ("#CC79A7", "v", "--"),
    "zfp": ("#56B4E9", "o", "--"),
    "sz3": ("#56B4E9", "s", "--"),
}
RD_CORE = (
    "fourier",
    "neural_field",
    "splats",
    "fourier_objective",
    "neural_field_physics",
    "gyrosplat",
)
RD_ALL = (
    RD_CORE[:3] + ("wavelet_haar", "wavelet_bior4.4", "pca", "zfp", "sz3") + RD_CORE[3:]
)
# single-budget codes drawn as points in the full figure: (method, colour, marker)
RD_POINTS = (
    ("autoencoder", "#0072B2", "o"),
    ("vae", "#0072B2", "s"),
    ("vqvae", "#0072B2", "^"),
    ("wavelet_haar_objective", "#E69F00", "s"),
    ("wavelet_db4", "#E69F00", "^"),
)
TITLES = {
    "psnr_f": "distribution PSNR (dB) ↑",
    "psnr_phi": "potential PSNR (dB) ↑",
    "flux_rel_l1": "heat-flux relL1 ↓",
    "kappa_rel": "κ / κ_truth (truth = 1)",
}


def rd_figure(
    exp: dict, rows: Iterable[dict] = (), full: bool = False, min_cr: float = 99.5
):
    """The rate-distortion sweep against the compression ratio by bytes (more storage to
    the right): mean over seeds of the snapshot median, bars = seed standard deviation.  With
    `full=True`, every swept code and, as points, the codes evaluated at the two comparison
    budgets only (`rows` = the stored rows, for the AE, VAE and VQ-VAE)."""
    import matplotlib.pyplot as plt

    from neugk.gyrosplat.eval.paper.experiments import (
        RAW_BYTES,
        RD_BUDGETS,
        TIERS,
        rd_summary,
        samples,
        summary,
    )

    keys = ("psnr_f", "psnr_phi", "flux_rel_l1", "kappa_rel")
    methods = RD_ALL if full else RD_CORE
    fig, axs = plt.subplots(1, 4, figsize=(12, 3.4))
    for ax, k in zip(axs, keys):
        for m in methods:
            pts = []
            for b in RD_BUDGETS:
                v = rd_summary(exp, m, b, k)
                if v is not None and RAW_BYTES / v["bytes"] >= min_cr:
                    pts.append((RAW_BYTES / v["bytes"], v["value"], v["std"] or 0.0))
            if not pts:
                continue
            x, y, e = zip(*pts)
            colour, marker, ls = STYLE[m]
            ax.errorbar(
                x,
                y,
                yerr=e,
                color=colour,
                marker=marker,
                ls=ls,
                lw=1.6 if m == "gyrosplat" else 1.0,
                ms=8 if m == "gyrosplat" else 4,
                label=METHODS[m],
                zorder=5 if m == "gyrosplat" else 3,
            )
        if full:
            rows = list(rows)
            for m, colour, marker in RD_POINTS:
                for tier in TIERS:
                    s = samples(exp, rows, tier, m)
                    if not s:
                        continue
                    nbytes = exp["storage"]["payload_bytes"][str(tier)].get(m) or next(
                        iter(next(iter(s.values())).values())
                    ).get("bytes")
                    ax.plot(
                        RAW_BYTES / nbytes,
                        summary(s, k)["value"],
                        marker=marker,
                        color=colour,
                        ls="",
                        ms=5,
                        label=METHODS[m] if tier == TIERS[0] else None,
                    )
        ax.set_xscale("log")
        ax.set_xlim(1150, 92)
        ax.set_xticks([1000, 302, 142, 100])
        ax.set_xticklabels(["1000", "302", "142", "100"])
        ax.minorticks_off()
        if k in ("flux_rel_l1", "kappa_rel"):
            ax.set_yscale("log")
        if k == "kappa_rel":
            from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

            ax.axhline(1.0, color="black", lw=1.0)
            ax.yaxis.set_major_locator(
                FixedLocator([1, 3, 10, 30] if full else [0.8, 1, 2, 4, 8])
            )
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.yaxis.set_minor_locator(NullLocator())
        ax.set_title(TITLES[k], fontsize=9)
        ax.grid(True, alpha=0.3)
    axs[0].set_ylabel("median over 39 snapshots")
    fig.supxlabel("compression ratio by bytes (more storage →)", fontsize=9)
    handles, labels = axs[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=6 if not full else 5,
        fontsize=7,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86 if not full else 0.8))
    return fig


def paired_figure(
    exp: dict, tiers: Sequence[int] = (302, 142), other: str = "neural_field_physics"
):
    """Potential-PSNR difference GyroSplat minus another seeded method per snapshot (both
    as means over fitting seeds), sorted; the line marks the median trajectory difference."""
    import matplotlib.pyplot as plt

    from neugk.gyrosplat.eval.paper.experiments import SHORT, paired_gain

    fig, axs = plt.subplots(1, len(tiers), figsize=(5.5 * len(tiers), 3.2), sharey=True)
    for ax, tier in zip(list(axs) if len(tiers) > 1 else [axs], tiers):
        g = paired_gain(exp, tier, other)
        d = sorted(g["differences"].values())
        ax.bar(range(len(d)), d, color=["#D55E00" if x > 0 else "#999999" for x in d])
        ax.axhline(0.0, color="black", lw=0.8)
        ax.axhline(g["median"], color="#D55E00", lw=1.2, ls="--")
        lo, hi = g["interval"]
        ax.set_title(
            f"{tier}×: median over trajectories {g['median']:+.2f} dB [{lo:+.2f}, {hi:+.2f}], "
            f"positive on {g['positive_snapshots']}/{g['snapshots']} snapshots",
            fontsize=8,
        )
        ax.set_xlabel("snapshot (sorted)")
        ax.grid(True, axis="y", alpha=0.3)
    (axs[0] if len(tiers) > 1 else axs).set_ylabel(f"ΔPSNR_φ vs {SHORT[other]} (dB)")
    fig.tight_layout()
    return fig


def heldout_figure(exp: dict, snapshot: Optional[str] = None):
    """The error of the heat-flux spectrum per mode, |Q(ky) - Q_ref(ky)| / sum |Q_ref|, of one
    snapshot when the flux is fitted on the even or on the odd modes only (filled markers =
    fitted modes, hollow = held out), beside the fit on every mode."""
    import matplotlib.pyplot as plt

    rows = exp["heldout_modes"]
    snapshot = snapshot or sorted({r["snapshot"] for r in rows})[0]
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.2), sharey=True)
    for ax, parity in zip(axs, ("even", "odd")):
        r = next(x for x in rows if x["snapshot"] == snapshot and x["fitted"] == parity)
        modes = list(range(1, len(r["flux_truth"])))
        scale = sum(abs(r["flux_truth"][k]) for k in modes)
        err = {
            k: max(abs(r["flux_pred"][k] - r["flux_truth"][k]) / scale, 1e-12)
            for k in modes
        }
        ref = [
            max(abs(r["all_modes_flux_pred"][k] - r["flux_truth"][k]) / scale, 1e-12)
            for k in modes
        ]
        fit = [k for k in modes if k in r["fit_band"]]
        held = [k for k in modes if k not in r["fit_band"]]
        ax.plot(
            modes,
            ref,
            color="#009E73",
            lw=0.9,
            ls="--",
            label="flux fitted on every mode",
        )
        ax.plot(
            fit, [err[k] for k in fit], "o", color="#D55E00", ms=4, label="fitted modes"
        )
        ax.plot(
            held,
            [err[k] for k in held],
            "o",
            mfc="white",
            color="#D55E00",
            ms=4,
            label="held-out modes",
        )
        ax.set_yscale("log")
        ax.set_title(f"{snapshot}, flux fitted on the {parity} modes", fontsize=9)
        ax.set_xlabel("binormal mode")
        ax.grid(True, alpha=0.3)
    axs[0].set_ylabel("|Q - Q_ref| / Σ|Q_ref| per mode")
    axs[1].legend(fontsize=7)
    fig.tight_layout()
    return fig


def restart_figure(exp: dict, snapshot: Optional[str] = None, early: float = 5.0):
    """Heat flux against time of the solver restarted from the original snapshot and from
    its 302x reconstructions (the shaded band is the early window of the table)."""
    import matplotlib.pyplot as plt

    from neugk.gyrosplat.eval.paper.experiments import load_restarts

    runs = load_restarts(exp)
    snapshot = snapshot or sorted(runs)[0]
    states = runs[snapshot]
    style = {
        "truth": ("black", "-", "original"),
        "truth_perturbed": ("#777777", ":", "original + 1e-6 perturbation"),
        "gyrosplat": ("#D55E00", "-", "GyroSplat"),
        "neural_field_physics": ("#009E73", "-", "NF + physics"),
        "autoencoder": ("#0072B2", "-", "AE"),
    }
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    ax.axvspan(0, early, color="#000000", alpha=0.06, lw=0)
    for state, (colour, ls, label) in style.items():
        if state in states:
            ax.plot(
                states[state]["time"],
                states[state]["heat_flux"],
                color=colour,
                ls=ls,
                lw=1.6 if state == "truth" else 1.0,
                label=label,
            )
    ax.set_xlabel("simulation time")
    ax.set_ylabel("heat flux")
    ax.set_title(f"restarts from {snapshot} (shaded: the early window)", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


__all__ = [
    "flux_spectrum_figure",
    "heldout_figure",
    "paired_figure",
    "rd_figure",
    "restart_figure",
    "scatter_figure",
    "TABLE1_ORDER",
]
