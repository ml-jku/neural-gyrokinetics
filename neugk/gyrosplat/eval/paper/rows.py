"""Per-snapshot result rows: the rows behind the paper's tables, and how to make more.

One row is one method, or one stage of the ablation ladder, on one snapshot at one nominal
budget.  `results/rows/gkw_rows.jsonl` holds the rows of the paper (one JSON object per
line); `package_rows` turns this package's own result files (`python -m neugk.gyrosplat fit`, `baseline`,
`ablate`) into the same format so that they drop into every table and figure.

Fields: `snapshot`, `class` (streamer / mixed / strong_zonal), `tier` (302 or 142, the
nominal budget), `method` (a key of METHODS, or None for an ablation stage), `recipe` and
`stage` (a key of LADDER and its stage index; the deployed recipe has stage 0 after the
joint fit and stage 1 after the Gauss-Newton polish, which is also `method = "gyrosplat"`),
`CR` (raw reals over stored reals), `psnr_f` and `psnr_phi` (peak PSNR of the distribution
on its stored grid and of the potential, in dB), `flux_rel_l1` and `flux_rel_l2` (errors of
the heat-flux spectrum over the nonzero binormal modes), `kappa`, `kappa_truth`, `kappa_rel`
(gross-to-net cancellation ratio of the flux integrand over those modes, the reference's,
and their ratio), `f_rel_l2`, `phi_rel_l2`, `flux_pred` and `flux_truth` (the heat-flux
spectrum per binormal mode, index 0 = zonal), `stored_reals`, `source`.
"""
from __future__ import annotations

import json
import os
import re
import statistics
from typing import Dict, Iterable, List, Optional, Sequence

TIERS = (302, 142)
METRICS = (
    "CR",
    "psnr_f",
    "psnr_phi",
    "flux_rel_l1",
    "kappa_rel",
    "f_rel_l2",
    "phi_rel_l2",
    "flux_rel_l2",
    "kappa",
    "kappa_truth",
)

METHODS: Dict[str, str] = {
    "wavelet_bior4.4": "Wavelet top-K (bior4.4)",
    "wavelet_haar": "Wavelet top-K (Haar)",
    "wavelet_db4": "Wavelet top-K (db4)",
    "wavelet_db8": "Wavelet top-K (db8)",
    "pca": "PCA, per snapshot",
    "zfp": "ZFP",
    "sz3": "SZ3",
    "fourier": "Fourier top-K",
    "autoencoder": "Autoencoder",
    "vae": "VAE",
    "vqvae": "VQ-VAE",
    "neural_field": "Neural field (field objective)",
    "splats": "GyroSplat (field objective only)",
    "neural_field_physics": "Neural field (+ physics stage)",
    "gyrosplat": "GyroSplat (ours)",
    "fourier_objective": "Fourier + refinement",
    "wavelet_haar_objective": "Haar + refinement",
}
LATEX_LABELS = {
    "wavelet_bior4.4": r"Wavelet top-$K$ (bior4.4)",
    "wavelet_haar": r"Wavelet top-$K$ (Haar)",
    "wavelet_db4": r"Wavelet top-$K$ (db4)",
    "wavelet_db8": r"Wavelet top-$K$ (db8)",
    "fourier": r"Fourier top-$K$",
    "splats": r"\ourmethod{} (field objective only)",
    "neural_field_physics": r"Neural field ($+$ physics stage)",
    "gyrosplat": r"\ourmethod{} (ours)",
    "fourier_objective": r"Fourier $+$ refinement",
    "wavelet_haar_objective": r"Haar $+$ refinement",
}
CITATIONS = {
    "wavelet_bior4.4": "cohen1992biorthogonal",
    "wavelet_haar": "daubechies1992",
    "pca": "eckart1936approximation",
    "zfp": "lindstrom2014zfp",
    "sz3": "liang2022sz3",
    "fourier": "devore1998nonlinear",
    "autoencoder": "pinc2026",
    "vae": "pinc2026",
    "vqvae": "oord2018vqvae,pinc2026",
    "neural_field": "pinc2026",
    "neural_field_physics": "pinc2026",
}
# the order of Table 1
TABLE1_ORDER = [
    "wavelet_bior4.4",
    "wavelet_haar",
    "pca",
    "zfp",
    "sz3",
    "fourier",
    "autoencoder",
    "vae",
    "vqvae",
    "neural_field",
    "splats",
    "neural_field_physics",
    "gyrosplat",
]
# the order of the per-class and dispersion tables
DISPERSION_ORDER = [
    "gyrosplat",
    "splats",
    "neural_field",
    "neural_field_physics",
    "autoencoder",
    "vae",
    "vqvae",
    "wavelet_bior4.4",
    "wavelet_haar",
    "wavelet_db4",
    "wavelet_db8",
    "pca",
    "zfp",
    "sz3",
    "fourier",
    "fourier_objective",
    "wavelet_haar_objective",
]
# what `python -m neugk.gyrosplat fit`, `baseline` and `ablate` regenerate; the rest are stored rows only
PACKAGE_METHODS = {
    "gyrosplat",
    "splats",
    "fourier",
    "fourier_objective",
    "wavelet_bior4.4",
    "wavelet_haar",
    "wavelet_db4",
    "wavelet_db8",
    "wavelet_haar_objective",
    "pca",
    "zfp",
    "sz3",
    "neural_field",
}

# the ablation ladder at 302x: (stage group, recipe, stage index, label)
LADDER = [
    ("1", "floor", 0, "floor (global norm, random placement)"),
    ("1", "zonal", 0, "+ zonal-split normalization"),
    ("1", "importance_init", 0, "+ importance placement, exact amplitude init"),
    ("1", "geometry_warmup", 0, "+ geometry warmup (field only)"),
    ("1", "potential_fit", 0, "+ physics fine-tune (phi on, flux off)"),
    ("2", "gabor_init", 0, "+ Gabor splats"),
    ("2", "deployed", 0, "+ phi-aware observable co-fit"),
    ("3", "deployed", 1, "+ flux Gauss-Newton, GyroSplat (deployed)"),
    ("A", "field_solve", 0, "co-fit -> field-optimal solve (field_solve)"),
    ("A", "gn_without_joint_fit", 0, "polish without co-fit (gn_nocofit)"),
    ("A", "noky", 0, "no modulation, equal bytes, to fine-tune (noky)"),
    ("A", "random_init", 0, "random placement, after field fitting (random_init)"),
    ("A", "random_full", 0, "random placement, end-to-end (random_full)"),
    ("A", "no_warmstart", 0, "field fit without amplitude warm start (no_warmstart)"),
    ("A", "flux_during_potential_fit", 0, "flux term on at the fine-tune (flux_on)"),
    (
        "A",
        "all_gabor",
        0,
        "all Gabor splats, after field and potential fitting (all_gabor)",
    ),
    ("A", "all_gabor_full", 0, "all Gabor splats, end-to-end (all_gabor_full)"),
    ("A", "frac40", 0, "40% Gabor allocation, after amplitude initialization (frac40)"),
    ("A", "frac60", 0, "60% Gabor allocation, after amplitude initialization (frac60)"),
    ("A", "s7_pack", 0, "quantize to fp16 (0.5x bytes) (s7_pack)"),
    (
        "A",
        "fp32match",
        0,
        "fp32 at matched parameter storage (N_eq = 2168) (fp32match)",
    ),
    ("A", "noky_full", 0, "base splats only, un-masked flux chain (noky_full)"),
    (
        "A",
        "flux_refinement_all_splats",
        0,
        "deployed splats, un-masked flux chain (base_gnall)",
    ),
]
# the seven rows of the main ablation table: (stage group, recipe, stage index, label, alternative)
LADDER_MAIN = [
    ("1", "importance_init", 0, "Gaussian splats", False),
    ("1", "potential_fit", 0, "Potential objective", False),
    ("1", "flux_during_potential_fit", 0, "Potential fit with flux loss", True),
    ("2", "deployed", 0, "Joint fit of distribution, potential and flux", False),
    ("3", "flux_refinement_all_splats", 0, "Gauss-Newton on all splats", True),
    ("3", "gn_without_joint_fit", 0, "No joint fit", True),
    ("3", "deployed", 1, "GyroSplat", False),
]
STAGE_TITLES = {
    "1": "Fit base splats to the distribution and potential",
    "2": "Add Gabor splats and fit both groups jointly",
    "3": "Refine Gabor amplitudes at fixed geometry",
    "A": "Alternative configurations",
}
CLASSES = (
    ("streamer", "streamer"),
    ("mixed", "mixed"),
    ("strong_zonal", "strong zonal"),
)
SNAPSHOT_CLASSES = {
    "extra_iteration_0_t150": "mixed",
    "extra_iteration_15_t150": "strong_zonal",
    "extra_iteration_19_t150": "strong_zonal",
    "extra_iteration_23_t150": "strong_zonal",
    "extra_iteration_26_t150": "strong_zonal",
    "extra_iteration_31_t150": "mixed",
    "extra_iteration_36_t150": "strong_zonal",
    "extra_iteration_5_t150": "strong_zonal",
    "iteration_100_t100": "strong_zonal",
    "iteration_100_t150": "strong_zonal",
    "iteration_100_t200": "strong_zonal",
    "iteration_115_t100": "mixed",
    "iteration_131_t150": "mixed",
    "iteration_13_t100": "streamer",
    "iteration_13_t150": "mixed",
    "iteration_13_t200": "mixed",
    "iteration_200_t100": "strong_zonal",
    "iteration_200_t150": "strong_zonal",
    "iteration_200_t200": "strong_zonal",
    "iteration_204_t32": "mixed",
    "iteration_214_t20": "mixed",
    "iteration_235_t100": "mixed",
    "iteration_26_t32": "strong_zonal",
    "iteration_280_t20": "mixed",
    "iteration_280_t25": "streamer",
    "iteration_280_t60": "streamer",
    "iteration_432_t32": "mixed",
    "iteration_454_t10": "streamer",
    "iteration_50_t32": "mixed",
    "iteration_520_t15": "streamer",
    "iteration_520_t5": "streamer",
    "iteration_5_t20": "streamer",
    "iteration_5_t32": "streamer",
    "iteration_681_t32": "strong_zonal",
    "iteration_711_t20": "streamer",
    "iteration_711_t45": "streamer",
    "iteration_877_t15": "streamer",
    "iteration_877_t20": "streamer",
    "iteration_893_t32": "strong_zonal",
}


def load_rows(path: str) -> List[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def select(rows: Iterable[dict], **where) -> List[dict]:
    """The rows whose fields equal the keyword values."""
    return [r for r in rows if all(r.get(k) == v for k, v in where.items())]


def rows_of(
    rows: Iterable[dict],
    tier: int,
    method: Optional[str] = None,
    recipe: Optional[str] = None,
    stage: Optional[int] = None,
) -> List[dict]:
    """One method, or one ladder row (recipe and stage), at one tier."""
    if method is not None:
        return select(rows, tier=tier, method=method)
    return select(rows, tier=tier, recipe=recipe, stage=stage)


def medians(rows: Sequence[dict], keys: Sequence[str] = METRICS) -> Optional[dict]:
    """Per-key medians over the rows (None when there are no rows)."""
    rows = list(rows)
    if not rows:
        return None
    out = {"n": len(rows)}
    for k in keys:
        vals = [r[k] for r in rows if r.get(k) is not None]
        out[k] = statistics.median(vals) if vals else None
    return out


def means(rows: Sequence[dict], keys: Sequence[str] = METRICS) -> Optional[dict]:
    """Per-key arithmetic means over the rows."""
    rows = list(rows)
    if not rows:
        return None
    out = {"n": len(rows)}
    for k in keys:
        vals = [r[k] for r in rows if r.get(k) is not None]
        out[k] = statistics.mean(vals) if vals else None
    return out


def quantile(values: Sequence[float], p: float) -> float:
    """Linear interpolation on the (n - 1) p grid."""
    vals = sorted(values)
    x = (len(vals) - 1) * p
    i = int(x)
    if i >= len(vals) - 1:
        return vals[-1]
    return vals[i] * (1 - (x - i)) + vals[i + 1] * (x - i)


def net_flux(row: dict):
    """(predicted, reference) heat flux summed over the nonzero modes."""
    return sum(row["flux_pred"][1:]), sum(row["flux_truth"][1:])


def sign_flips(rows: Iterable[dict]) -> int:
    """Snapshots whose predicted net flux has the opposite sign to the reference's."""
    count = 0
    for r in rows:
        pred, truth = net_flux(r)
        count += pred * truth < 0
    return count


def grouped(rows: Iterable[dict]) -> Dict[tuple, List[dict]]:
    """rows -> {(tier, method or (recipe, stage)): rows}."""
    out: Dict[tuple, List[dict]] = {}
    for r in rows:
        key = (r["tier"], r["method"] or (r["recipe"], r["stage"]))
        out.setdefault(key, []).append(r)
    return out


# ------------------------------------------------------------- this package's result files
_TIER_OF_N_EQUIV = {4337: 302, 9216: 142, 2168: 302, 4608: 302}
_TIER_OF_BUDGET = {73728: 302, 73732: 302, 156669: 142, 156673: 142}


def _metrics(score: dict) -> dict:
    """The row metrics from a result file's score dict."""
    kappa = score.get("kappa_nonzero", (score.get("kappa") or [None])[0])
    truth = score.get("kappa_truth_nonzero", (score.get("kappa_gt") or [None])[0])
    stored = score.get("stored_reals")
    return {
        "CR": score.get("CR"),
        "stored_reals": stored.get("total") if isinstance(stored, dict) else stored,
        "psnr_f": score.get("psnr_f", score.get("psnr_paper_all")),
        "psnr_phi": score.get("psnr_phi", score.get("phi_psnr")),
        "psnr_phi_signed": score.get("psnr_phi_signed", score.get("phi_psnr")),
        "f_rel_l2": score.get("f_rel_l2"),
        "phi_rel_l2": score.get("phi_rel_l2"),
        "flux_rel_l1": score.get(
            "flux_rel_l1", (score.get("flux_relL1_band") or [None])[0]
        ),
        "flux_rel_l2": score.get("flux_rel_l2"),
        "kappa": kappa,
        "kappa_truth": truth,
        "kappa_rel": score.get(
            "kappa_rel", kappa / truth if kappa is not None and truth else None
        ),
        "flux_pred": score.get("flux_pred"),
        "flux_truth": score.get("flux_truth"),
    }


def _row(payload: dict, score: dict, tier, method, recipe, stage, source) -> dict:
    snapshot = payload.get("snapshot")
    return {
        "dataset": payload.get("dataset", "gkw"),
        "snapshot": snapshot,
        "class": SNAPSHOT_CLASSES.get(snapshot),
        "tier": tier,
        "method": method,
        "recipe": recipe,
        "stage": stage,
        **_metrics(score),
        "source": source,
    }


def package_rows(paths: Iterable[str], tier: Optional[int] = None) -> List[dict]:
    """Rows from this package's result files: `fit` (the deployed stages 0 and 1, and the
    16-bit row as `s7_pack`), `baseline` (the method and its objective variant), `ablate`
    (the control).  The tier is read from the recipe size or the budget unless given."""
    out = []
    given = tier
    for path in paths:
        tier = given
        with open(path) as fh:
            payload = json.load(fh)
        rows = payload.get("rows") or []
        source = os.path.basename(path)
        cfg = payload.get("config") or {}
        named = re.search(r"_(302|142)x_", source)
        if named and tier is None:
            tier = int(named.group(1))
        if "control" in payload:
            t = tier or _TIER_OF_N_EQUIV.get(cfg.get("n_equiv"))
            control = payload["control"]
            for i, r in enumerate(rows):
                method = "gyrosplat" if control == "deployed" and i == 1 else None
                out.append(_row(payload, r, t, method, control, i, source))
        elif "method" in payload:
            t = tier or _TIER_OF_BUDGET.get(
                int(round(payload.get("budget_reals") or 0))
            )
            m = payload["method"]
            base = m[:-3] if m.endswith("_gn") else m
            if base == "wavelet":
                base = f"wavelet_{payload.get('wavelet') or 'haar'}"
            names = [base, base + "_objective"]
            for r, name in zip(rows, names):
                out.append(_row(payload, r, t, name, None, None, source))
        else:
            t = tier or _TIER_OF_N_EQUIV.get(cfg.get("n_equiv"))
            for i, r in enumerate(rows):
                if i < 2:
                    method = "gyrosplat" if i == 1 else None
                    out.append(_row(payload, r, t, method, "deployed", i, source))
                else:
                    out.append(_row(payload, r, t, None, "s7_pack", 0, source))
    return out
