"""The byte-matched comparison and the appendix experiments, from `results/experiments/`.

`results/rows/gkw_rows.jsonl` holds one fit per method and snapshot at the budgets of stored
values.  The tables here compare the methods at equal payload bytes and report every method
that has a random part as the mean over its fitting seeds of the snapshot median.  Payload
bytes count the stored values at 4 bytes and, for the sparse codes, the position of every
kept coefficient (an int32 per complex Fourier coefficient, per kept wavelet value).

The files of `results/experiments/` (one JSON object per line unless noted):

| file                      | content                                                           |
|---------------------------|-------------------------------------------------------------------|
| `seeds.jsonl`             | GyroSplat and NF + physics (seeds 0-4), NF and splats (0-2)       |
| `byte_matched.jsonl`      | Fourier and wavelet codes, with and without refinement, at equal  |
|                           | bytes, with the heat-flux views outside the objective (`view_*`)  |
| `rate_distortion.jsonl`   | the sweep over eight budgets, 1000x to 100x                       |
| `views.jsonl`             | heat-flux views outside the objective of the stored fits          |
| `heldout_modes.jsonl`     | flux fitted on the even or the odd binormal modes only            |
| `restarts/<snap>.npz`     | solver restarts from the original and the reconstructed snapshots |
| `latent_refinement.jsonl` | AE, VAE and VQ-VAE latents refined through the frozen decoders    |
| `damping.jsonl`           | damping and step count of the final Gauss-Newton refinement       |
| `timing.jsonl`            | encode and decode time, peak memory, on dedicated GPUs            |
| `operator_cost.jsonl`     | the diagnostic operator: construction time and cache sizes        |
| `storage.json`            | payload bytes per method, shared decoder sizes, sweep budgets     |

Metrics follow the row format of `rows.py`: `psnr_f`, `psnr_phi` (dB), `flux_rel_l1` (heat-flux
spectrum over the nonzero modes), `kappa_rel` (kappa / kappa_truth, one being ideal),
`f_rel_l2`, `phi_rel_l2`.  Every table function returns Markdown, or a LaTeX tabular with
`latex=True`.
"""
from __future__ import annotations

import json
import os
import random
import statistics
from typing import Dict, Iterable, List, Optional, Sequence

from neugk.gyrosplat.eval.paper.rows import (
    CITATIONS,
    LATEX_LABELS,
    METHODS,
    SNAPSHOT_CLASSES,
    TABLE1_ORDER,
    load_rows,
    quantile,
)

RAW_BYTES = 89_128_960
TIERS = (302, 142)
N_SNAPSHOTS = len(SNAPSHOT_CLASSES)
SEEDED = ("gyrosplat", "neural_field_physics", "neural_field", "splats")
BYTE_MATCHED = (
    "fourier",
    "fourier_objective",
    "wavelet_haar",
    "wavelet_bior4.4",
    "wavelet_db4",
    "wavelet_haar_objective",
)
REFINED = ("fourier_objective", "wavelet_haar_objective")
APPENDIX_ORDER = list(TABLE1_ORDER) + list(REFINED)
# the order of Table 1 (the neural field with the physics stage next to the field-only one)
MAIN_ORDER = [m for m in TABLE1_ORDER if m not in ("neural_field_physics", "splats")]
MAIN_ORDER[
    MAIN_ORDER.index("neural_field") + 1 : MAIN_ORDER.index("neural_field") + 1
] = [
    "neural_field_physics",
    "splats",
]
VIEWS = ("Q_kx", "Q_s", "Q_vpar_mu", "Q_pointwise")
CLASSES = (
    ("streamer", "Streamer"),
    ("mixed", "Mixed"),
    ("strong_zonal", "Strong zonal"),
)
RD_BUDGETS = (1000, 800, 600, 450, 302, 200, 142, 100)
RD_METHODS = ("gyrosplat", "neural_field_physics", "fourier_objective", "fourier")

# the short names of the appendix tables
SHORT = {
    "gyrosplat": "GyroSplat",
    "splats": "Splats, field only",
    "neural_field": "NF, field only",
    "neural_field_physics": "NF + physics",
    "fourier_objective": "Fourier + refinement",
    "wavelet_haar_objective": "Haar + refinement",
    "fourier": "Fourier top-K",
    "wavelet_bior4.4": "Wavelet bior4.4",
    "wavelet_haar": "Wavelet Haar",
    "wavelet_db4": "Wavelet db4",
    "autoencoder": "AE",
    "vae": "VAE",
    "vqvae": "VQ-VAE",
    "pca": "PCA",
    "zfp": "ZFP",
    "sz3": "SZ3",
}
SHORT_TEX = {**SHORT, "gyrosplat": r"\ourmethod{}", "fourier": r"Fourier top-$K$"}
_SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


# ------------------------------------------------------------------------------ loading
def load_experiments(root: str) -> dict:
    """Every row file of `results/experiments/` by name, and `storage.json`."""
    out: dict = {"root": str(root)}
    for name in (
        "seeds",
        "byte_matched",
        "rate_distortion",
        "views",
        "heldout_modes",
        "latent_refinement",
        "damping",
        "timing",
        "operator_cost",
    ):
        out[name] = load_rows(os.path.join(root, name + ".jsonl"))
    with open(os.path.join(root, "storage.json")) as fh:
        out["storage"] = json.load(fh)
    return out


def samples(
    exp: dict, rows: Iterable[dict], tier: int, method: str
) -> Dict[int, Dict[str, dict]]:
    """{seed: {snapshot: row}} of one method at one byte budget: the fitting seeds of the
    seeded methods, the byte-matched run of the Fourier and wavelet codes, and the stored
    row (`gkw_rows.jsonl`) of the AE, VAE, VQ-VAE, PCA, ZFP and SZ3.  Only seeds that cover
    every snapshot count."""
    out: Dict[int, Dict[str, dict]] = {}
    if method in SEEDED:
        for r in exp["seeds"]:
            if r["method"] == method and r["tier"] == tier:
                out.setdefault(r["seed"], {})[r["snapshot"]] = r
    elif method in BYTE_MATCHED:
        for r in exp["byte_matched"]:
            if r["method"] == method and r["tier"] == tier:
                out.setdefault(0, {})[r["snapshot"]] = r
    else:
        for r in rows:
            if r["method"] == method and r["tier"] == tier and r.get("stage") is None:
                out.setdefault(0, {})[r["snapshot"]] = r
    return {s: d for s, d in sorted(out.items()) if len(d) == N_SNAPSHOTS}


def comparison_rows(exp: dict, rows: Iterable[dict], seed: int = 0) -> List[dict]:
    """One row per method, byte budget and snapshot, for figures of single fits: the given
    fitting seed of the seeded methods, the byte-matched runs of the Fourier and wavelet
    codes, and the stored rows of the other codes."""
    rows = list(rows)
    out = []
    for tier in TIERS:
        for m in APPENDIX_ORDER + ["wavelet_db4"]:
            s = samples(exp, rows, tier, m)
            if not s:
                continue
            for snap, r in (s.get(seed) or s[min(s)]).items():
                out.append({**r, "method": m, "tier": tier, "snapshot": snap})
    return out


def summary(per_seed: Dict[int, Dict[str, dict]], key: str) -> Optional[dict]:
    """The snapshot quartiles of one metric averaged over seeds (`value` is the mean of the
    medians), and the standard deviation of the median over seeds (None for one seed)."""
    if not per_seed:
        return None
    qs = [
        [quantile([r[key] for r in d.values()], p) for p in (0.25, 0.5, 0.75)]
        for d in per_seed.values()
    ]
    med = [q[1] for q in qs]
    return {
        "value": statistics.mean(med),
        "q25": statistics.mean(q[0] for q in qs),
        "q75": statistics.mean(q[2] for q in qs),
        "std": statistics.stdev(med) if len(med) > 1 else None,
        "seeds": len(med),
    }


def _summaries(
    exp, rows, tier, method, keys=("psnr_f", "psnr_phi", "flux_rel_l1", "kappa_rel")
):
    s = samples(exp, rows, tier, method)
    return {k: summary(s, k) for k in keys} if s else None


# ------------------------------------------------------------------------------ formatting
def _num(
    x, digits: int = 3, latex: bool = False, math: bool = False, bold: bool = False
):
    """Fixed digits, or a one-decimal mantissa below 1e-3 (5.2e-7 -> 5.2x10^-7)."""
    if x is None:
        return "---" if latex else "—"
    if 0 < abs(x) < 1e-3:
        m, e = f"{x:.1e}".split("e")
        s = m + (
            r"\!\times\!10^{" + str(int(e)) + "}"
            if latex
            else "×10" + str(int(e)).translate(_SUP)
        )
        sci = True
    else:
        s, sci = f"{x:.{digits}f}", False
    if latex:
        if bold:
            s = r"\boldsymbol{" + s + "}"
        return "$" + s + "$" if (sci or math or bold) else s
    return f"**{s}**" if bold else s


def _metric(x, key: str, latex: bool = False, **kw) -> str:
    return _num(x, 2 if key.startswith("psnr") else 3, latex, **kw)


def _pm(v: Optional[dict], key: str, latex: bool) -> str:
    """mean ± seed std, or the single value."""
    if v is None:
        return _num(None, 0, latex)
    s = _metric(v["value"], key, latex)
    if v["std"] is None:
        return s
    return s + (r" $\pm$ " if latex else " ± ") + _metric(v["std"], key, latex)


def _name(method: str, latex: bool) -> str:
    return (SHORT_TEX if latex else SHORT)[method]


class _Out:
    """Rows with group headers, as Markdown or as a booktabs LaTeX tabular."""

    def __init__(self, columns, latex, spec=None, header=None):
        self.columns, self.latex = list(columns), latex
        self.spec = spec or "l" + "r" * (len(self.columns) - 1)
        self.header = header
        self.lines: List[tuple] = []

    def group(self, text: str, tex: Optional[str] = None):
        self.lines.append(("group", tex if self.latex and tex else text))

    def row(self, cells):
        self.lines.append(("row", [str(c) for c in cells]))

    def rule(self):
        self.lines.append(("rule", None))

    def space(self):
        self.lines.append(("space", None))

    def render(self) -> str:
        n = len(self.columns)
        if self.latex:
            out = [r"\begin{tabular}{" + self.spec + "}", r"\toprule"]
            out += self.header or [" & ".join(self.columns) + r" \\"]
            out.append(r"\midrule")
            for kind, x in self.lines:
                if kind == "group":
                    out.append(
                        r"\multicolumn{" + str(n) + r"}{l}{\textit{" + x + r"}}\\"
                    )
                elif kind == "rule":
                    out.append(r"\midrule")
                elif kind == "space":
                    out.append(r"\addlinespace")
                else:
                    out.append(" & ".join(x) + r" \\")
            return "\n".join(out + [r"\bottomrule", r"\end{tabular}"])
        out = ["| " + " | ".join(self.columns) + " |", "|" + "---|" * n]
        for kind, x in self.lines:
            if kind == "group":
                out.append("| *" + x + "* |" + " |" * (n - 1))
            elif kind == "row":
                out.append("| " + " | ".join(x) + " |")
        return "\n".join(out)


def _tier_group(tier: int, latex: bool, nbytes: Optional[int] = None) -> tuple:
    text = f"CR ≈ {tier}×"
    tex = rf"CR $\approx {tier}\times$"
    if nbytes:
        text += f" ({nbytes:,} bytes)"
        tex += f" ({nbytes:,} bytes)".replace(",", "{,}")
    return text, tex


# ------------------------------------------------------------------------------ main text
def main_table(
    exp: dict, rows: Iterable[dict], latex: bool = False, cite: bool = False
) -> str:
    """Table 1: every method at the two byte budgets.  Seeded methods are the mean over seeds
    of the snapshot median; bold marks the best entry of a column within a budget (the
    cancellation ratio closest to one)."""
    rows = list(rows)
    keys = ("psnr_f", "psnr_phi", "flux_rel_l1", "kappa_rel")
    digits = {"psnr_f": 2, "psnr_phi": 2, "flux_rel_l1": 4, "kappa_rel": 3}
    cols = ["method", "PSNR_f ↑", "PSNR_φ ↑", "flux relL1 ↓", "κ/κ_truth"]
    if latex:
        cols = [
            "method",
            r"PSNR$_f$\,\up",
            r"PSNR$_\phi$\,\up",
            r"flux relL1\,\down",
            r"$\krel$",
        ]
    t = _Out(cols, latex)
    for i, tier in enumerate(TIERS):
        if i:
            t.rule()
        t.group(
            *_tier_group(
                tier, latex, exp["storage"]["payload_bytes"][str(tier)]["gyrosplat"]
            )
        )
        vals = {m: _summaries(exp, rows, tier, m) for m in MAIN_ORDER}
        present = [m for m in MAIN_ORDER if vals[m]]
        best = {}
        for k in keys:
            v = [vals[m][k]["value"] for m in present]
            best[k] = (
                min(v, key=lambda x: abs(x - 1))
                if k == "kappa_rel"
                else max(v)
                if k.startswith("psnr")
                else min(v)
            )
        for m in present:
            label = METHODS[m] if not latex else LATEX_LABELS.get(m, METHODS[m])
            if latex and cite and m in CITATIONS:
                label += r"~\citep{" + CITATIONS[m] + "}"
            t.row(
                [label]
                + [
                    _num(
                        vals[m][k]["value"],
                        digits[k],
                        latex,
                        math=True,
                        bold=vals[m][k]["value"] == best[k],
                    )
                    for k in keys
                ]
            )
    return t.render()


def refinement_table(exp: dict, rows: Iterable[dict], latex: bool = False) -> str:
    """Table 3: the physics refinement on the Haar and Fourier codes and, at 302x, on the
    latents of the AE, VAE and VQ-VAE (400 Adam steps through the frozen decoders),
    beside GyroSplat."""
    rows = list(rows)
    keys = ("psnr_f", "psnr_phi", "flux_rel_l1", "kappa_rel")
    t = _Out(
        ["representation", "PSNR_f ↑", "PSNR_φ ↑", "flux relL1 ↓", "κ/κ_truth"]
        if not latex
        else [
            "representation",
            r"PSNR$_f$\,\up",
            r"PSNR$_\phi$\,\up",
            r"flux relL1\,\down",
            r"$\krel$",
        ],
        latex,
    )
    latent = latent_medians(exp)
    for i, tier in enumerate(TIERS):
        if i:
            t.rule()
        t.group(*_tier_group(tier, latex))
        for m in ("wavelet_haar_objective", "fourier_objective"):
            v = _summaries(exp, rows, tier, m)
            t.row([_name(m, latex)] + [_metric(v[k]["value"], k, latex) for k in keys])
        if tier == 302:
            for model in ("autoencoder", "vae", "vqvae"):
                v = latent[(model, "adam")]
                t.row(
                    [_name(model, latex) + " + refinement"]
                    + [_metric(v[k], k, latex) for k in keys]
                )
        v = _summaries(exp, rows, tier, "gyrosplat")
        t.row(
            [_name("gyrosplat", latex)]
            + [_metric(v[k]["value"], k, latex) for k in keys]
        )
    return t.render()


def db4_table(exp: dict, latex: bool = False) -> str:
    """Table 4: the db4 wavelet code at the two byte budgets (medians over the snapshots)."""
    keys = ("psnr_f", "psnr_phi", "flux_rel_l1", "kappa_rel")
    digits = {"psnr_f": 2, "psnr_phi": 2, "flux_rel_l1": 4, "kappa_rel": 3}
    cols = ["basis", "CR", "PSNR_f", "PSNR_φ", "flux relL1", "κ/κ_truth"]
    if latex:
        cols = ["Basis", "CR", r"PSNR$_f$", r"PSNR$_\phi$", "Flux relL1", r"$\krel$"]
    t = _Out(cols, latex)
    for tier in TIERS:
        v = _summaries(exp, [], tier, "wavelet_db4")
        t.row(
            ["db4", rf"${tier}\times$" if latex else f"{tier}×"]
            + [_num(v[k]["value"], digits[k], latex, math=True) for k in keys]
        )
    return t.render()


# ------------------------------------------------------------------------------ dispersion, seeds
def seed_dispersion_table(exp: dict, rows: Iterable[dict], latex: bool = False) -> str:
    """Table 7: the snapshot dispersion (mean median [mean 25th, mean 75th percentile] over
    seeds) of every method at both budgets, and the seed variation of the median of the four
    seeded methods (mean ± standard deviation over seeds)."""
    rows = list(rows)
    keys = ("psnr_f", "psnr_phi", "flux_rel_l1")
    t = _Out(
        ["method", "PSNR_f", "PSNR_φ", "flux relL1"]
        if not latex
        else ["Method", r"PSNR$_f$", r"PSNR$_\phi$", "Flux relL1"],
        latex,
    )
    for tier in TIERS:
        text, tex = _tier_group(tier, latex)
        t.group("Snapshot dispersion, " + text, "Snapshot dispersion, " + tex)
        for m in APPENDIX_ORDER:
            v = _summaries(exp, rows, tier, m, keys)
            if not v:
                continue
            cells = []
            for k in keys:
                q = v[k]
                inner = _metric(q["q25"], k, latex) + ", " + _metric(q["q75"], k, latex)
                cells.append(
                    _metric(q["value"], k, latex)
                    + (r" {\tiny[" + inner + "]}" if latex else f" [{inner}]")
                )
            t.row([_name(m, latex)] + cells)
        t.rule()
    t.group(
        "Fitting-seed variation: mean ± standard deviation of snapshot medians",
        r"Fitting-seed variation: mean $\pm$ standard deviation of snapshot medians",
    )
    for tier in TIERS:
        for m in SEEDED:
            v = _summaries(exp, rows, tier, m, keys)
            label = (
                _name(m, latex)
                + (rf", ${tier}\times$" if latex else f", {tier}×")
                + f" ({v['psnr_f']['seeds']} seeds)"
            )
            t.row([label] + [_pm(v[k], k, latex) for k in keys])
    return t.render()


def seed_means(exp: dict, tier: int, method: str, key: str) -> Dict[str, float]:
    """{snapshot: mean over fitting seeds} of one metric of a seeded method."""
    per: Dict[str, List[float]] = {}
    for r in exp["seeds"]:
        if r["method"] == method and r["tier"] == tier:
            per.setdefault(r["snapshot"], []).append(r[key])
    return {s: statistics.mean(v) for s, v in per.items()}


def trajectory(snapshot: str) -> str:
    """The trajectory of a snapshot ("iteration_13_t100" -> "iteration_13")."""
    return snapshot.rsplit("_t", 1)[0]


def paired_gain(
    exp: dict,
    tier: int,
    other: str = "neural_field_physics",
    key: str = "psnr_phi",
    resamples: int = 10_000,
    seed: int = 0,
) -> dict:
    """GyroSplat minus another seeded method, both as per-snapshot seed means.  Snapshots of
    one trajectory are not independent, so the differences are averaged per trajectory first;
    the median of the 27 trajectory differences gets a 95% bootstrap interval over
    trajectories.  Positive = GyroSplat higher (for the flux error, pass key="flux_rel_l1"
    and read a negative gain as GyroSplat better)."""
    a = seed_means(exp, tier, "gyrosplat", key)
    b = seed_means(exp, tier, other, key)
    common = sorted(set(a) & set(b))
    diff = {s: a[s] - b[s] for s in common}
    per: Dict[str, List[float]] = {}
    for s in common:
        per.setdefault(trajectory(s), []).append(diff[s])
    td = [statistics.mean(v) for v in per.values()]
    rng = random.Random(seed)
    boots = sorted(
        statistics.median(rng.choice(td) for _ in td) for _ in range(resamples)
    )
    lo, hi = int(0.025 * resamples) - 1, int(0.975 * resamples) - 1
    return {
        "median": statistics.median(td),
        "interval": (boots[lo], boots[hi]),
        "positive_snapshots": sum(d > 0 for d in diff.values()),
        "snapshots": len(common),
        "trajectories": len(td),
        "differences": diff,
    }


def regime_table(
    exp: dict, rows: Iterable[dict], tier: int = 302, latex: bool = False
) -> str:
    """Table 8: medians within the three turbulence regimes (12 streamer, 12 mixed, 15
    strong-zonal snapshots), averaged over the fitting seeds."""
    rows = list(rows)
    keys = ("psnr_f", "psnr_phi", "flux_rel_l1")
    heads = ("PSNR_f", "PSNR_φ", "flux")
    cols = ["method"] + [f"{title}: {h}" for _, title in CLASSES for h in heads]
    header = None
    if latex:
        header = [
            "Method & "
            + " & ".join(r"\multicolumn{3}{c}{" + title + "}" for _, title in CLASSES)
            + r" \\",
            " & " + " & ".join([r"PSNR$_f$", r"PSNR$_\phi$", "Flux"] * 3) + r" \\",
        ]
    t = _Out(cols, latex, spec="l" + "r" * 9, header=header)
    for m in APPENDIX_ORDER:
        s = samples(exp, rows, tier, m)
        if not s:
            continue
        cells = []
        for cls, _ in CLASSES:
            for k in keys:
                v = statistics.mean(
                    statistics.median(
                        r[k] for snap, r in d.items() if SNAPSHOT_CLASSES[snap] == cls
                    )
                    for d in s.values()
                )
                cells.append(_metric(v, k, latex))
        t.row([_name(m, latex)] + cells)
    return t.render()


# ------------------------------------------------------------------------------ storage
def bytes_table(exp: dict, latex: bool = False) -> str:
    """Table 9: payload bytes and the compression ratio (raw snapshot bytes / payload) of
    every compared code at the two budgets."""
    pay = exp["storage"]["payload_bytes"]
    cols = ["method", "bytes, 302×", "CR", "bytes, 142×", "CR"]
    if latex:
        cols = ["Method", r"Bytes ($302\times$)", "CR", r"Bytes ($142\times$)", "CR"]
    t = _Out(cols, latex)
    for m in APPENDIX_ORDER:
        if m not in pay["302"]:
            continue
        cells = []
        for tier in ("302", "142"):
            b = pay[tier].get(m)
            if b is None:
                cells += [_num(None, 0, latex)] * 2
                continue
            s = f"{b:,}"
            cells += [s.replace(",", "{,}") if latex else s, f"{RAW_BYTES / b:.1f}"]
        t.row([_name(m, latex)] + cells)
    return t.render()


# ------------------------------------------------------------------------------ rate-distortion
def rd_summary(exp: dict, method: str, budget: int, key: str) -> Optional[dict]:
    """Mean and standard deviation over seeds of the snapshot median at one sweep budget."""
    per: Dict[int, Dict[str, dict]] = {}
    for r in exp["rate_distortion"]:
        if r["method"] == method and r["budget"] == budget:
            per.setdefault(r["seed"], {})[r["snapshot"]] = r
    per = {s: d for s, d in per.items() if len(d) == N_SNAPSHOTS}
    if not per:
        return None
    med = [statistics.median(r[key] for r in d.values()) for d in per.values()]
    return {
        "value": statistics.mean(med),
        "std": statistics.stdev(med) if len(med) > 1 else None,
        "seeds": len(med),
        "bytes": statistics.median(
            r["bytes"] for d in per.values() for r in d.values()
        ),
    }


def rd_table(
    exp: dict,
    key: str = "psnr_phi",
    methods: Sequence[str] = RD_METHODS,
    latex: bool = False,
) -> str:
    """Tables 10-13: one metric of the sweep, two budgets per block; seeded methods as mean ±
    standard deviation over seeds 0-2 of the snapshot median."""
    cols = ["method", "first budget", "second budget"]
    if latex:
        cols = ["Method", "First budget", "Second budget"]
    t = _Out(cols, latex)
    for i in range(0, len(RD_BUDGETS), 2):
        pair = RD_BUDGETS[i : i + 2]
        if i:
            t.rule()
        t.group(
            "Budgets " + " / ".join(f"{b}×" for b in pair),
            "Budgets " + " / ".join(rf"${b}\times$" for b in pair),
        )
        for m in methods:
            t.row(
                [_name(m, latex)]
                + [_pm(rd_summary(exp, m, b, key), key, latex) for b in pair]
            )
    return t.render()


def rd_budget_table(exp: dict, latex: bool = False) -> str:
    """Table 14: the payload bytes of the eight sweep budgets (GyroSplat's stored values at
    4 bytes) and the compression ratio they give."""
    cols = ["nominal CR", "payload bytes", "CR from bytes"]
    if latex:
        cols = ["Nominal CR", "Payload bytes", "CR from bytes"]
    t = _Out(cols, latex)
    budgets = exp["storage"]["rate_distortion_budgets"]
    for b in RD_BUDGETS:
        n = budgets[str(b)]["bytes"]
        s = f"{n:,}"
        t.row(
            [
                rf"${b}\times$" if latex else f"{b}×",
                s.replace(",", "{,}") if latex else s,
                f"{RAW_BYTES / n:.1f}",
            ]
        )
    return t.render()


# ------------------------------------------------------------------------------ diagnostics
VIEW_ORDER = (
    "gyrosplat",
    "neural_field_physics",
    "fourier_objective",
    "wavelet_haar_objective",
    "splats",
    "neural_field",
    "autoencoder",
    "vae",
    "vqvae",
    "fourier",
    "pca",
    "zfp",
    "sz3",
    "wavelet_haar",
    "wavelet_bior4.4",
)


def view_medians(exp: dict, tier: int, method: str) -> Optional[dict]:
    """Median relative L1 error of the heat-flux views over the snapshots: Q(kx), Q(s),
    Q(v_par, mu) and the pointwise integrand.  The Fourier and wavelet codes are their
    byte-matched runs, the others their stored fit (one seed)."""
    if method in BYTE_MATCHED:
        g = [
            {k: r["view_" + k] for k in VIEWS}
            for r in exp["byte_matched"]
            if r["method"] == method and r["tier"] == tier
        ]
    else:
        g = [r for r in exp["views"] if r["method"] == method and r["tier"] == tier]
    if not g:
        return None
    return {k: statistics.median(r[k] for r in g) for k in VIEWS}


def views_table(exp: dict, latex: bool = False) -> str:
    """Table 15: heat-flux diagnostics outside the fitting objective."""
    cols = ["method", "Q(kx)", "Q(s)", "Q(v∥,μ)", "pointwise"]
    if latex:
        cols = ["Method", r"$Q(k_x)$", r"$Q(s)$", r"$Q(v_\parallel,\mu)$", "Pointwise"]
    t = _Out(cols, latex)
    for i, tier in enumerate(TIERS):
        if i:
            t.rule()
        t.group(*_tier_group(tier, latex))
        for m in VIEW_ORDER:
            v = view_medians(exp, tier, m)
            if v is not None:
                t.row([_name(m, latex)] + [_num(v[k], 3, latex) for k in VIEWS])
    return t.render()


def _rel_l1(pred, truth, modes) -> float:
    return sum(abs(pred[k] - truth[k]) for k in modes) / max(
        sum(abs(truth[k]) for k in modes), 1e-30
    )


def heldout_rows(exp: dict) -> List[dict]:
    """Per fit: the flux error on the fitted and on the held-out nonzero modes, and the
    change of potential PSNR against the fit of every mode."""
    out = []
    for r in exp["heldout_modes"]:
        modes = range(1, len(r["flux_truth"]))
        fitted = [k for k in modes if k in r["fit_band"]]
        held = [k for k in modes if k not in r["fit_band"]]
        out.append(
            {
                "fitted": r["fitted"],
                "snapshot": r["snapshot"],
                "fitted_flux": _rel_l1(r["flux_pred"], r["flux_truth"], fitted),
                "heldout_flux": _rel_l1(r["flux_pred"], r["flux_truth"], held),
                "all_modes_heldout_flux": _rel_l1(
                    r["all_modes_flux_pred"], r["flux_truth"], held
                ),
                "delta_psnr_phi": r["psnr_phi"] - r["all_modes_psnr_phi"],
                "delta_psnr_f": r["psnr_f"] - r["all_modes_psnr_f"],
            }
        )
    return out


def heldout_table(exp: dict, latex: bool = False) -> str:
    """Table 16: flux fitted on the even or the odd modes only (nine snapshots, 302x)."""
    cols = ["fitted modes", "fitted flux relL1", "held-out flux relL1", "ΔPSNR_φ (dB)"]
    if latex:
        cols = [
            "Fitted modes",
            "Fitted flux relL1",
            "Held-out flux relL1",
            r"$\Delta$PSNR$_\phi$ (dB)",
        ]
    t = _Out(cols, latex)
    per = heldout_rows(exp)
    for parity in ("even", "odd"):
        g = [r for r in per if r["fitted"] == parity]
        med = lambda k: statistics.median(r[k] for r in g)  # noqa: E731
        t.row(
            [
                parity.capitalize(),
                _num(med("fitted_flux"), 3, latex),
                _num(med("heldout_flux"), 3, latex),
                f"{med('delta_psnr_phi'):.2f}",
            ]
        )
    return t.render()


RESTART_STATES = ("truth_perturbed", "gyrosplat", "neural_field_physics", "autoencoder")


def load_restarts(exp: dict) -> Dict[str, dict]:
    """{snapshot: {state: {"time", "heat_flux", "ky_spectrum"}}} of the solver restarts."""
    import glob

    import numpy as np

    out: Dict[str, dict] = {}
    for path in sorted(glob.glob(os.path.join(exp["root"], "restarts", "*.npz"))):
        z = np.load(path)
        snap = os.path.basename(path)[: -len(".npz")]
        states: Dict[str, dict] = {}
        for name in z.files:
            state, q = name.split("__")
            states.setdefault(state, {})[q] = z[name]
        out[snap] = states
    return out


def restart_metrics(
    state: dict, truth: dict, early: float = 5.0, threshold: float = 0.1
) -> dict:
    """The early heat-flux error (median over 0 < t <= `early` of |Q - Q_truth| / mean |Q_truth|)
    and the departure time (first of three consecutive samples with that error above
    `threshold`) of one restart against the restart from the original snapshot.  A run that
    becomes non-finite is cut at its last finite sample (`blowup` = that time)."""
    import numpy as np

    t, q = np.asarray(state["time"]), np.asarray(state["heat_flux"])
    bad = np.where(~np.isfinite(q[1:]))[0]
    blowup = float(t[1:][bad[0]]) if bad.size else None
    if bad.size:
        t, q = t[: bad[0] + 1], q[: bad[0] + 1]
    n = min(len(t), len(truth["time"]))
    t, q, qt = t[:n], q[:n], np.asarray(truth["heat_flux"])[:n]
    window = (t > 0) & (t <= early)
    err = np.abs(q - qt) / np.nanmean(np.abs(qt[window]))
    departure, run = None, 0
    for i in range(1, n):
        run = run + 1 if err[i] > threshold else 0
        if run == 3:
            departure = float(t[i - 2])
            break
    return {
        "early": float(np.nanmedian(err[window])),
        "departure": departure,
        "blowup": blowup,
    }


def restart_table(exp: dict, latex: bool = False) -> str:
    """Table 17: early solver restarts from the 302x reconstructions (eight snapshots;
    medians over the snapshots; the departure time over the runs that depart)."""
    runs = load_restarts(exp)
    names = {
        "truth_perturbed": (
            "Original + 1e-6 perturbation",
            r"Original + $10^{-6}$ perturbation",
        ),
        "gyrosplat": ("GyroSplat", r"\ourmethod{}"),
        "neural_field_physics": ("NF + physics", "NF + physics"),
        "autoencoder": ("AE", "AE"),
    }
    cols = ["initial state", "early flux error (%)", "10% departure time", "runs"]
    if latex:
        cols = ["Initial state", r"Early flux error (\%)", r"10\% departure time"]
    t = _Out(cols, latex)
    for state in RESTART_STATES:
        m = [restart_metrics(d[state], d["truth"]) for d in runs.values() if state in d]
        dep = [x["departure"] for x in m if x["departure"] is not None]
        cells = [
            f"{100 * statistics.median(x['early'] for x in m):.1f}",
            f"{statistics.median(dep):.0f}" if dep else "—",
        ]
        if not latex:
            nonfinite = sum(x["blowup"] is not None for x in m)
            cells.append(
                f"{len(m)}" + (f", {nonfinite} non-finite" if nonfinite else "")
            )
        t.row([names[state][latex]] + cells)
    return t.render()


# ------------------------------------------------------------------------------ frozen decoders
LATENT_STAGES = (
    ("initial", "Initial"),
    ("adam", "400 Adam steps"),
    ("gauss_newton", "+ GN"),
    ("levenberg_marquardt", "+ LM"),
)


def latent_medians(
    exp: dict, adam_steps: int = 400, snapshots: Optional[Iterable[str]] = None
) -> dict:
    """{(model, stage): medians over the snapshots} of the latent refinement."""
    keep = None if snapshots is None else set(snapshots)
    groups: Dict[tuple, List[dict]] = {}
    for r in exp["latent_refinement"]:
        if r["adam_steps"] == adam_steps and (keep is None or r["snapshot"] in keep):
            groups.setdefault((r["model"], r["stage"]), []).append(r)
    keys = ("psnr_f", "psnr_phi", "flux_rel_l1", "kappa_rel") + tuple(
        "view_" + k for k in VIEWS
    )
    out = {}
    for g, rs in groups.items():
        out[g] = {k: statistics.median(r[k] for r in rs) for k in keys}
        out[g]["n"] = len(rs)
        if all("seconds" in r for r in rs):
            out[g]["seconds"] = statistics.median(r["seconds"] for r in rs)
    return out


def latent_table(exp: dict, rows: Iterable[dict], latex: bool = False) -> str:
    """Table 18: AE, VAE and VQ-VAE latents refined through their frozen decoders at 302x
    (39 snapshots, medians): the stored latent, 400 Adam steps on the field, potential and flux
    objective, and Gauss-Newton or Levenberg-Marquardt from that result, beside GyroSplat."""
    rows = list(rows)
    med = latent_medians(exp)
    cols = ["model", "fit", "PSNR_f", "PSNR_φ", "flux relL1", "κ/κ_truth", "Q(v∥,μ)"]
    if latex:
        cols = [
            "Model",
            "Fit",
            r"PSNR$_f$",
            r"PSNR$_\phi$",
            "Flux relL1",
            r"$\kappa_{\rm rel}$",
            r"$Q(v_\parallel,\mu)$",
        ]
    t = _Out(cols, latex)
    for model in ("autoencoder", "vae", "vqvae"):
        for stage, title in LATENT_STAGES:
            v = med[(model, stage)]
            t.row(
                [
                    _name(model, latex),
                    title,
                    _num(v["psnr_f"], 2, latex),
                    _num(v["psnr_phi"], 2, latex),
                    _num(v["flux_rel_l1"], 3, latex),
                    _num(v["kappa_rel"], 3, latex),
                    _num(v["view_Q_vpar_mu"], 3, latex),
                ]
            )
        t.space()
    ours = _summaries(exp, rows, 302, "gyrosplat")
    views = view_medians(exp, 302, "gyrosplat")
    t.row(
        [_name("gyrosplat", latex), f"{ours['psnr_f']['seeds']}-seed mean"]
        + [
            _metric(ours[k]["value"], k, latex)
            for k in ("psnr_f", "psnr_phi", "flux_rel_l1", "kappa_rel")
        ]
        + [_num(views["Q_vpar_mu"], 3, latex)]
    )
    return t.render()


def latent_steps_table(exp: dict, latex: bool = False) -> str:
    """Table 19: 400 against 2,000 Adam steps on the pilot snapshots that have both."""
    long = {r["snapshot"] for r in exp["latent_refinement"] if r["adam_steps"] == 2000}
    a = latent_medians(exp, 400, long)
    b = latent_medians(exp, 2000, long)
    arrow = r" $\to$ " if latex else " → "
    cols = ["model", "PSNR_φ (dB)", "flux relL1", "PSNR_f (dB)"]
    if latex:
        cols = ["Model", r"PSNR$_\phi$ (dB)", "Flux relL1", r"PSNR$_f$ (dB)"]
    t = _Out(cols, latex)
    for model in ("autoencoder", "vae", "vqvae"):
        x, y = a[(model, "adam")], b[(model, "adam")]
        t.row(
            [_name(model, latex)]
            + [
                _num(x[k], d, latex) + arrow + _num(y[k], d, latex)
                for k, d in (("psnr_phi", 2), ("flux_rel_l1", 3), ("psnr_f", 2))
            ]
        )
    return t.render()


# ------------------------------------------------------------------------------ stability
def final_fit_table(rows: Iterable[dict], latex: bool = False) -> str:
    """Table 22: checks of the stored GyroSplat fits (39 snapshots per budget, seed 0): median
    and worst flux error, the count below 1e-4, net-flux sign reversals, the count with
    0.9 <= kappa / kappa_truth <= 1.1, and the lowest potential PSNR."""
    rows = list(rows)
    cols = [
        "CR",
        "median flux",
        "worst flux",
        "< 1e-4",
        "sign flips",
        "κ within 10%",
        "min PSNR_φ",
    ]
    if latex:
        cols = [
            "CR",
            "Median flux",
            "Worst flux",
            r"$<10^{-4}$",
            "Sign flips",
            r"$\kappa$ within 10\%",
            r"Min PSNR$_\phi$",
        ]
    t = _Out(cols, latex)
    for tier in TIERS:
        g = [r for r in rows if r["method"] == "gyrosplat" and r["tier"] == tier]
        flux = [r["flux_rel_l1"] for r in g]
        flips = sum(
            (sum(r["flux_pred"][1:32]) > 0) != (sum(r["flux_truth"][1:32]) > 0)
            for r in g
        )
        t.row(
            [
                rf"${tier}\times$" if latex else f"{tier}×",
                _num(statistics.median(flux), 3, latex),
                _num(max(flux), 3, latex),
                f"{sum(x < 1e-4 for x in flux)}/{len(g)}",
                f"{flips}/{len(g)}",
                f"{sum(0.9 <= r['kappa_rel'] <= 1.1 for r in g)}/{len(g)}",
                _num(min(r["psnr_phi"] for r in g), 2, latex),
            ]
        )
    return t.render()


def damping_table(exp: dict, latex: bool = False) -> str:
    """Table 23: damping and step count of the final Gauss-Newton refinement at 302x, from
    the same saved joint-fit states of 12 snapshots; the potential-PSNR change is relative to
    those states."""
    pts: Dict[tuple, List[dict]] = {}
    for r in exp["damping"]:
        pts.setdefault((r["damping"], r["steps"]), []).append(r)
    cols = [
        "damping",
        "steps",
        "median flux",
        "worst flux",
        "median ΔPSNR_φ",
        "worst ΔPSNR_φ",
    ]
    if latex:
        cols = [
            "Damping",
            "Steps",
            "Median flux",
            "Worst flux",
            r"Median $\Delta$PSNR$_\phi$",
            r"Worst $\Delta$PSNR$_\phi$",
        ]
    t = _Out(cols, latex)
    for (mu, steps), g in sorted(pts.items()):
        flux = [r["flux_rel_l1"] for r in g]
        d = [r["psnr_phi"] - r["start_psnr_phi"] for r in g]
        t.row(
            [
                _num(mu, 4, latex),
                str(steps),
                _num(statistics.median(flux), 3, latex),
                _num(max(flux), 3, latex),
                f"{statistics.median(d):.3f}",
                f"{min(d):.3f}",
            ]
        )
    return t.render()


# ------------------------------------------------------------------------------ cost
TIMED = ("gyrosplat", "neural_field_physics", "neural_field", "splats")


def _iqr3(values: Sequence[float], latex: bool) -> str:
    v = sorted(x for x in values if x is not None)
    if not v:
        return "---" if latex else "—"
    q = statistics.quantiles(v, n=4)
    dash = "--" if latex else "–"
    return f"{statistics.median(v):.3g} [{q[0]:.3g}{dash}{q[2]:.3g}]"


def timed_rows(exp: dict, tier: int, method: str) -> List[dict]:
    """The timing rows that count: one job per dedicated GPU, without each GPU's first job
    and without jobs that shared their GPU with another process."""
    return [
        r
        for r in exp["timing"]
        if r["method"] == method and r["tier"] == tier and r["excluded"] is None
    ]


def timing_table(exp: dict, methods: Sequence[str] = TIMED, latex: bool = False) -> str:
    """Table 24: encode and decode time (median [interquartile range], seconds) and peak GPU
    memory per snapshot (nine snapshots, two seeds; codecs one seed)."""
    cols = ["method", "CR", "fits", "encode (s)", "decode (s)", "peak GB"]
    if latex:
        cols = ["Method", "CR", "Fits", "Encode (s)", "Decode (s)", "Peak GB"]
    t = _Out(cols, latex)
    for tier in TIERS:
        for m in methods:
            g = timed_rows(exp, tier, m)
            if not g:
                continue
            t.row(
                [
                    _name(m, latex),
                    rf"${tier}\times$" if latex else f"{tier}×",
                    str(len(g)),
                    _iqr3([r["encode_seconds"] for r in g], latex),
                    _iqr3([r["decode_seconds"] for r in g], latex),
                    f"{max(r['peak_gpu_gb'] or 0 for r in g):.1f}",
                ]
            )
    return t.render()


STAGES = (
    ("fit", "Distribution fit"),
    ("physics_fit", "Potential fit"),
    ("carriers", "Gabor + joint fit"),
    ("gauss_newton", "Final GN refinement"),
)


def stage_table(exp: dict, latex: bool = False) -> str:
    """Table 25: median time per GyroSplat stage (the medians need not sum to the total)."""
    cols = (
        ["stage", "302× (s)", "142× (s)"]
        if not latex
        else ["Stage", r"$302\times$ (s)", r"$142\times$ (s)"]
    )
    t = _Out(cols, latex)
    g = {tier: timed_rows(exp, tier, "gyrosplat") for tier in TIERS}
    for key, title in STAGES:
        t.row(
            [title]
            + [
                f"{statistics.median(r['stage_seconds'][key] for r in g[tier]):.0f}"
                for tier in TIERS
            ]
        )
    t.row(
        ["Total encoding"]
        + [
            f"{statistics.median(r['encode_seconds'] for r in g[tier]):.0f}"
            for tier in TIERS
        ]
    )
    return t.render()


def operator_cost(exp: dict) -> dict:
    """Medians over the snapshots: building the diagnostic operator (geometry, field solve,
    reference readouts), the bytes it holds, and the Gauss-Newton Jacobian at each budget
    (31 modes x real and imaginary part x Gabor amplitudes, float64)."""
    ops = exp["operator_cost"]
    med = lambda k: statistics.median(r[k] for r in ops)  # noqa: E731
    out = {
        "construct_seconds": med("construct_seconds"),
        "construct_peak_gb": med("construct_peak_gb"),
        "reference_bytes": med("truth_target_bytes"),
        "geometry_bytes": med("geometry_bytes"),
        "snapshot_bytes": med("snapshot_field_bytes"),
    }
    for tier in TIERS:
        n = statistics.median(
            r["gabor_amplitudes"] for r in timed_rows(exp, tier, "gyrosplat")
        )
        out[f"gabor_amplitudes_{tier}"] = n
        out[f"jacobian_bytes_{tier}"] = 31 * 2 * n * 8
    return out


# ------------------------------------------------------------------------------ relative errors
REL_ORDER = (
    "gyrosplat",
    "neural_field_physics",
    "neural_field",
    "splats",
    "fourier_objective",
    "wavelet_haar_objective",
    "fourier",
    "wavelet_bior4.4",
    "wavelet_haar",
    "autoencoder",
    "vae",
    "vqvae",
    "pca",
    "zfp",
    "sz3",
)
REL_NAMES = {
    "gyrosplat": "GyroSplat",
    "neural_field": "NF",
    "fourier": "Fourier top-K",
    "wavelet_bior4.4": "Wavelet top-K (bior4.4)",
    "wavelet_haar": "Wavelet top-K (Haar)",
    "autoencoder": "Autoencoder",
}


def relative_error_table(exp: dict, rows: Iterable[dict], latex: bool = False) -> str:
    """Table 26: relative L2 errors ||u_hat - u|| / ||u|| of the distribution and the potential,
    in percent, at the two byte budgets (seeded methods: mean over seeds of the median)."""
    rows = list(rows)
    pct = r"\%" if latex else "%"
    cols = ["method", "f, 302×", "φ, 302×", "f, 142×", "φ, 142×"]
    if latex:
        cols = [
            "Method",
            r"$f$, $302\times$",
            r"$\phi$, $302\times$",
            r"$f$, $142\times$",
            r"$\phi$, $142\times$",
        ]
    t = _Out(cols, latex)
    for m in REL_ORDER:
        cells = []
        for tier in TIERS:
            s = samples(exp, rows, tier, m)
            if not s:
                cells += [_num(None, 0, latex)] * 2
                continue
            for k in ("f_rel_l2", "phi_rel_l2"):
                cells.append(f"{100 * summary(s, k)['value']:.0f}{pct}")
        t.row([REL_NAMES.get(m, SHORT[m])] + cells)
    return t.render()


__all__ = [
    "APPENDIX_ORDER",
    "RD_BUDGETS",
    "MAIN_ORDER",
    "bytes_table",
    "comparison_rows",
    "damping_table",
    "db4_table",
    "final_fit_table",
    "heldout_rows",
    "heldout_table",
    "latent_medians",
    "latent_steps_table",
    "latent_table",
    "load_experiments",
    "load_restarts",
    "main_table",
    "operator_cost",
    "paired_gain",
    "rd_budget_table",
    "rd_summary",
    "rd_table",
    "refinement_table",
    "regime_table",
    "relative_error_table",
    "restart_metrics",
    "restart_table",
    "samples",
    "seed_dispersion_table",
    "seed_means",
    "stage_table",
    "summary",
    "timing_table",
    "view_medians",
    "views_table",
]
