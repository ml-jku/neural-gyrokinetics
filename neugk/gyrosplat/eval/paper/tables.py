"""The paper's tables from per-snapshot rows (Markdown, or LaTeX with `latex=True`).

Every function takes the list of rows (`load_rows`) and returns one string.  Medians are
taken per metric over the snapshots of a group; quantiles interpolate linearly on the
(n - 1) p grid; the cancellation column is the per-snapshot ratio kappa / kappa_truth, one
being ideal; sign flips count snapshots whose predicted net flux over the nonzero modes has
the wrong sign.  Bold marks the best entry of a column within a budget: the largest PSNR,
the smallest flux error, the cancellation ratio closest to one.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

from neugk.gyrosplat.eval.paper.rows import (
    CITATIONS,
    CLASSES,
    DISPERSION_ORDER,
    LADDER,
    LADDER_MAIN,
    LATEX_LABELS,
    METHODS,
    STAGE_TITLES,
    TABLE1_ORDER,
    TIERS,
    medians,
    means,
    quantile,
    rows_of,
    sign_flips,
)

FIELDS = ("psnr_f", "psnr_phi", "flux_rel_l1", "kappa_rel")
DIGITS = (2, 2, 4, 3)
HEADERS = ("PSNR_f ↑", "PSNR_φ ↑", "flux relL1 ↓", "κ/κ_truth")
LATEX_HEADERS = (
    r"PSNR$_f$\,\up",
    r"PSNR$_\phi$\,\up",
    r"flux relL1\,\down",
    r"$\krel$",
)
BUDGETS = {302: "73,728", 142: "156,669"}
_SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


# ------------------------------------------------------------------------------ formatting
def _num(x, digits: int, latex: bool = False, bold: bool = False) -> str:
    if x is None:
        return r"\textemdash" if latex else "—"
    if 0 < abs(x) < 1e-3:
        a, b = f"{x:.2e}".split("e")
        e = str(int(b))
        s = a + (r"\!\times\!10^{" + e + "}" if latex else "×10" + e.translate(_SUP))
    else:
        s = f"{x:.{digits}f}"
    if latex:
        return "$" + (r"\boldsymbol{" + s + "}" if bold else s) + "$"
    return f"**{s}**" if bold else s


def _cr(x, latex: bool = False) -> str:
    if x is None:
        return r"\textemdash" if latex else "—"
    return f"${x:.1f}\\times$" if latex else f"{x:.1f}×"


def _iqr(values: Sequence[float], digits: int, latex: bool = False) -> str:
    if not values:
        return r"\textemdash" if latex else "—"
    q = [quantile(values, p) for p in (0.25, 0.5, 0.75)]
    if max(abs(v) for v in q) < 1e-3 and min(abs(v) for v in q) > 0:
        import math

        e = math.floor(math.log10(q[1]))
        v = [x / 10**e for x in q]
        if latex:
            return rf"${v[1]:.2f}\,[{v[0]:.2f},{v[2]:.2f}]\!\times\!10^{{{e}}}$"
        return f"{v[1]:.2f} [{v[0]:.2f}, {v[2]:.2f}]×10{str(e).translate(_SUP)}"
    if latex:
        return f"${q[1]:.{digits}f}\\, [{q[0]:.{digits}f},{q[2]:.{digits}f}]$"
    return f"{q[1]:.{digits}f} [{q[0]:.{digits}f}, {q[2]:.{digits}f}]"


def _label(method: str, latex: bool, cite: bool = False) -> str:
    if not latex:
        return METHODS[method]
    s = LATEX_LABELS.get(method, METHODS[method])
    if cite and method in CITATIONS:
        s += r"~\citep{" + CITATIONS[method] + "}"
    return s


def _best(values: List[Optional[dict]]) -> dict:
    """Per field, the best median among the groups (None-safe)."""
    present = [m for m in values if m]
    best = {}
    if not present:
        return best
    for k in FIELDS[:-1]:
        vals = [m[k] for m in present if m.get(k) is not None]
        if vals:
            best[k] = max(vals) if "psnr" in k else min(vals)
    vals = [m["kappa_rel"] for m in present if m.get("kappa_rel") is not None]
    if vals:
        best["kappa_rel"] = min(vals, key=lambda v: abs(v - 1))
    return best


def _cells(m: Optional[dict], best: Optional[dict], latex: bool) -> List[str]:
    return [
        _num(
            None if m is None else m.get(k),
            d,
            latex,
            bool(best and m and m.get(k) is not None and m.get(k) == best.get(k)),
        )
        for k, d in zip(FIELDS, DIGITS)
    ]


class _Table:
    """Rows of cells with group headers, rendered as Markdown or as a LaTeX tabular."""

    def __init__(self, columns: Sequence[str], latex: bool, spec: Optional[str] = None):
        self.columns = list(columns)
        self.latex = latex
        self.spec = spec or ("l" + "r" * (len(columns) - 1))
        self.lines: List[tuple] = []

    def group(self, text: str, latex_text: Optional[str] = None):
        self.lines.append(("group", latex_text if self.latex and latex_text else text))

    def row(self, cells: Sequence[str]):
        self.lines.append(("row", list(cells)))

    def rule(self):
        self.lines.append(("rule", None))

    def render(self) -> str:
        n = len(self.columns)
        if self.latex:
            out = [r"\begin{tabular}{" + self.spec + "}", r"\toprule"]
            out.append(" & ".join(self.columns) + r" \\")
            out.append(r"\midrule")
            for kind, payload in self.lines:
                if kind == "group":
                    out.append(r"\multicolumn{" + str(n) + r"}{l}{" + payload + r"}\\")
                elif kind == "rule":
                    out.append(r"\addlinespace")
                else:
                    out.append(" & ".join(payload) + r" \\")
            out += [r"\bottomrule", r"\end{tabular}"]
            return "\n".join(out)
        out = ["| " + " | ".join(self.columns) + " |", "|" + "---|" * n]
        for kind, payload in self.lines:
            if kind == "group":
                out.append("| *" + payload + "*" + " |" * n)
            elif kind == "rule":
                continue
            else:
                out.append("| " + " | ".join(payload) + " |")
        return "\n".join(out)


def _headers(latex: bool, first: str, cr: bool = False, sign: bool = False):
    cols = [first] + (["CR"] if cr else []) + list(LATEX_HEADERS if latex else HEADERS)
    if sign:
        cols.append(r"wrong sign" if latex else "wrong sign")
    return cols


def _tier_title(tier: int, latex: bool, budget: bool = True) -> str:
    if latex:
        s = r"\emph{CR ${\approx}" + str(tier) + r"\times$"
        return s + (
            r" ($" + BUDGETS[tier].replace(",", "{,}") + r"$ stored reals)}"
            if budget
            else "}"
        )
    return f"CR ≈ {tier}×" + (f" ({BUDGETS[tier]} stored reals)" if budget else "")


# ------------------------------------------------------------------------------ the tables
def families_table(
    rows: Iterable[dict],
    tiers: Sequence[int] = TIERS,
    methods: Optional[Sequence[str]] = None,
    latex: bool = False,
    cite: bool = False,
) -> str:
    """Table 1: medians over the snapshots for every method at both budgets."""
    rows = list(rows)
    methods = list(methods or TABLE1_ORDER)
    t = _Table(_headers(latex, "method"), latex)
    for i, tier in enumerate(tiers):
        if i:
            t.rule()
        t.group(_tier_title(tier, latex))
        med = {m: medians(rows_of(rows, tier, m)) for m in methods}
        present = [m for m in methods if med[m]]
        best = _best([med[m] for m in present])
        for m in present:
            t.row(
                [_label(m, latex, cite and tier == tiers[0])]
                + _cells(med[m], best, latex)
            )
    return t.render()


def _ladder_rows(rows, entries, tier: int = 302, aggregate=medians):
    return {
        (recipe, stage): aggregate(rows_of(rows, tier, recipe=recipe, stage=stage))
        for _, recipe, stage, *_ in entries
    }


def ladder_table(
    rows: Iterable[dict],
    extended: bool = False,
    latex: bool = False,
    mean: bool = False,
) -> str:
    """The ablation ladder at 302x: the seven rows of the main table, or the extended ladder
    of 23 rows with the achieved compression ratio (`extended=True`).  Medians, or
    arithmetic means with `mean=True`."""
    rows = list(rows)
    agg = means if mean else medians
    if extended:
        t = _Table(_headers(latex, "stage / ablation", cr=True), latex)
        values = _ladder_rows(rows, LADDER, 302, agg)
        previous = None
        for group, recipe, stage, label in LADDER:
            if group != previous:
                if previous is not None:
                    t.rule()
                title = STAGE_TITLES[group]
                if latex and group != "A":
                    t.group(
                        title,
                        r"\stagebadge{stage"
                        + {"1": "one", "2": "two", "3": "three"}[group]
                        + "}{"
                        + group
                        + r"}\quad\textbf{"
                        + title
                        + "}",
                    )
                else:
                    t.group(title, r"\textbf{" + title + "}")
                previous = group
            m = values[(recipe, stage)]
            t.row(
                [label, _cr(None if m is None else m["CR"], latex)]
                + _cells(m, None, latex)
            )
        return t.render()
    t = _Table(_headers(latex, "configuration within each stage"), latex)
    values = _ladder_rows(rows, LADDER_MAIN, 302, agg)
    previous = None
    for group, recipe, stage, label, alternative in LADDER_MAIN:
        if group != previous:
            if previous is not None:
                t.rule()
            t.group(STAGE_TITLES[group], r"\textbf{" + STAGE_TITLES[group] + "}")
            previous = group
        name = (
            (r"\textcolor{black!60}{" + label + "}" if latex else f"({label})")
            if alternative
            else label
        )
        if latex and label == "GyroSplat":
            name = r"\ourmethod{}"
        t.row([name] + _cells(values[(recipe, stage)], None, latex))
    return t.render()


def transfer_table(
    rows: Iterable[dict], tiers: Sequence[int] = TIERS, latex: bool = False
) -> str:
    """Table 3: the physics refinement on the Haar and Fourier codes beside GyroSplat."""
    rows = list(rows)
    methods = ["wavelet_haar_objective", "fourier_objective", "gyrosplat"]
    t = _Table(_headers(latex, "representation"), latex)
    for i, tier in enumerate(tiers):
        if i:
            t.rule()
        t.group(_tier_title(tier, latex, budget=False))
        med = {m: medians(rows_of(rows, tier, m)) for m in methods}
        best = _best(list(med.values()))
        for m in methods:
            t.row([_label(m, latex)] + _cells(med[m], best, latex))
    return t.render()


def wavelet_table(
    rows: Iterable[dict], tiers: Sequence[int] = TIERS, latex: bool = False
) -> str:
    """Table 4: the wavelet variants with their wrong-sign counts."""
    rows = list(rows)
    t = _Table(_headers(latex, "basis", cr=True, sign=True), latex)
    for tier in tiers:
        for m, name in (
            ("wavelet_haar", "Haar"),
            ("wavelet_db4", "db4"),
            ("wavelet_db8", "db8"),
        ):
            g = rows_of(rows, tier, m)
            med = medians(g)
            flips = f"{sign_flips(g)}/{len(g)}" if g else _num(None, 0, latex)
            t.row(
                [name, _cr(None if med is None else med["CR"], latex)]
                + _cells(med, None, latex)
                + [flips]
            )
    return t.render()


def per_class_table(
    rows: Iterable[dict],
    tier: int = 302,
    methods: Sequence[str] = DISPERSION_ORDER,
    latex: bool = False,
) -> str:
    """Table 7: medians within the snapshot classes."""
    rows = list(rows)
    cols = ["family"]
    for _, title in CLASSES:
        cols += [f"{title}: {h}" for h in (LATEX_HEADERS if latex else HEADERS)]
    t = _Table(cols, latex, spec="l" + "r" * 12)
    for m in methods:
        g = rows_of(rows, tier, m)
        if not g:
            continue
        cells = []
        for cls, _ in CLASSES:
            cells += _cells(medians([r for r in g if r["class"] == cls]), None, latex)
        t.row([_label(m, latex)] + cells)
    return t.render()


def dispersion_table(
    rows: Iterable[dict],
    tiers: Sequence[int] = TIERS,
    methods: Sequence[str] = DISPERSION_ORDER,
    latex: bool = False,
) -> str:
    """Table 8: median [25th, 75th percentile] per method and budget, with sign flips."""
    rows = list(rows)
    cols = (
        ["family", "CR"]
        + list(LATEX_HEADERS if latex else HEADERS)
        + [r"sign flips\,\down" if latex else "sign flips ↓"]
    )
    t = _Table(cols, latex, spec="lcccccc")
    for i, tier in enumerate(tiers):
        if i:
            t.rule()
        t.group(_tier_title(tier, latex, budget=False))
        for m in methods:
            g = rows_of(rows, tier, m)
            if not g:
                continue
            med = medians(g)
            cells = [_cr(med["CR"], latex)]
            cells += [_iqr([r[k] for r in g], d, latex) for k, d in zip(FIELDS, DIGITS)]
            cells.append(str(sign_flips(g)))
            t.row([_label(m, latex)] + cells)
    return t.render()


def storage_table(
    configs: Sequence[str], raw_reals: int = 22_282_240, latex: bool = False
) -> str:
    """The storage accounting of the recipes: base splats, Gabor groups, expanded basis size,
    stored reals (17 per base splat, 14 per shared envelope, 3 per carrier, 4 normalization
    values) and the compression ratio against `raw_reals` (the GKW snapshot)."""
    from neugk.gyrosplat.config import load_config
    from neugk.gyrosplat.recipe import budget

    cols = [
        "recipe",
        "base splats",
        "Gabor groups",
        "basis functions",
        "stored reals",
        "CR",
    ]
    t = _Table(cols, latex, spec="lrrrrr")
    for path in configs:
        cfg = load_config(path)
        n_base, n_groups = budget(cfg)
        k = cfg.carriers.tied_k if cfg.carriers.kind == "tied_gabor" else 0
        per_group = (14 + 3 * k) if cfg.carriers.share_envelopes else 17 * k
        stored = cfg.reals_per_atom * n_base + per_group * n_groups + 4
        t.row(
            [
                path.rsplit("/", 1)[-1],
                f"{n_base:,}",
                f"{n_groups:,}",
                f"{n_base + k * n_groups:,}",
                f"{stored:,}",
                _cr(raw_reals / stored, latex),
            ]
        )
    return t.render()


def compare_table(
    paper: Iterable[dict],
    runs: Iterable[dict],
    tier: int,
    methods: Sequence[str],
    latex: bool = False,
) -> str:
    """Medians of the paper's rows and of this package's re-run over the snapshots the
    re-run covers, method by method."""
    paper, runs = list(paper), list(runs)
    cols = (
        ["method", "n"]
        + [f"{h} paper" for h in HEADERS]
        + [f"{h} package" for h in HEADERS]
    )
    t = _Table(cols, latex, spec="lr" + "r" * 8)
    for m in methods:
        new = rows_of(runs, tier, m)
        if not new:
            continue
        snaps = {r["snapshot"] for r in new}
        old = [r for r in rows_of(paper, tier, m) if r["snapshot"] in snaps]
        t.row(
            [_label(m, latex), str(len(new))]
            + _cells(medians(old), None, latex)
            + _cells(medians(new), None, latex)
        )
    return t.render()
