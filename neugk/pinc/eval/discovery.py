"""Shared eval helpers: checkpoint discovery, traditional-compressor calibration, per-snapshot
metrics and the per-method result-file convention. Imported by the eval drivers in scripts/ so the
(traj, timestep) set, the matched-CR calibration and the result schema are defined in ONE place.
"""

import os
import re
import glob
import json
import math
from collections import defaultdict
from functools import partial

import numpy as np

from neugk.pinc.neural_fields.data import CycloneNFDataset  # noqa: F401  re-export
from neugk.pinc.eval import trad
from neugk.pinc.eval.metrics import integrate
from neugk.physics.diagnostics import velocity_moment_errors

# NF checkpoint prefixes: density-only warmup vs PINC-trained (physics losses).
NF_PREFIX = {"nf": "best_mlp", "nf-pinc": "best_int_mlp"}

# traditional method -> (fn, knob name, monotone grid of knob values).
# CR increases along the grid order (so we can pick the value closest to target).
TRAD = {
    "zfp": (trad.zfp_recon, "tolerance", np.logspace(0, 6, 30)),
    "wavelet": (trad.wavelet_recon, "threshold", np.logspace(-1, 3, 30)),
    "pca": (trad.pca_recon, "n_components", list(range(40, 0, -1))),
    "jpeg2000": (trad.jpeg2000_recon, "quality", np.logspace(1.7, -2, 30)),
    "sz3": (trad.sz3_recon, "error_bound", np.logspace(0, 6, 30)),
}


def discover(ckpt_dir, prefix):
    """{traj: {t: path}} and the NF compression ratio, for one checkpoint prefix."""
    rx = re.compile(re.escape(prefix) + r"_(iteration_\d+)_t(\d+)_x(\d+)\.pt$")
    weights, cr = defaultdict(dict), None
    for p in glob.glob(os.path.join(ckpt_dir, prefix + "_*.pt")):
        m = rx.search(os.path.basename(p))
        if m:
            weights[m.group(1)][int(m.group(2))] = p
            cr = int(m.group(3))
    return weights, cr


def calibrate(name, df, target_cr):
    """Pick the knob value whose CR is closest to target_cr (log distance)."""
    fn, knob, grid = TRAD[name]
    best = None
    for v in grid:
        try:
            _, _, size = fn(df, **{knob: (int(v) if knob == "n_components" else float(v))})
        except Exception:
            continue
        if not size or size <= 0:  # e.g. wavelet threshold so high all coeffs vanish
            continue
        cr = df.nbytes / size
        d = abs(math.log(cr) - math.log(target_cr))
        if best is None or d < best[0]:
            best = (d, v, cr)
    if best is None:
        return None
    _, v, cr = best
    return partial(fn, **{knob: (int(v) if knob == "n_components" else float(v))}), cr, v


def metrics_for(pred, gt, geom, csize):
    """Per-snapshot metrics (df/phi PSNR, flux L1, achieved CR, velocity moments). No EPE -- the
    full-sequence schema (EPE + spectral WD) is evaluate_method in runner.py."""
    dev = next(iter(geom.values())).device  # keep pred/gt with geom (FluxIntegral)
    pred = pred.detach().to(dev).float()
    gt = gt.detach().to(dev).float()
    mse = float(((pred - gt) ** 2).mean())
    p_phi, p_ef = integrate(pred, geom)
    g_phi, g_ef = integrate(gt, geom)
    phi_mse = float(((p_phi - g_phi) ** 2).mean())
    out = dict(
        df_psnr=float(10 * math.log10(gt.max() ** 2 / mse)) if mse > 0 else float("inf"),
        df_l1=float((pred - gt).abs().mean()),
        phi_psnr=(
            float(10 * math.log10(g_phi.max() ** 2 / phi_mse)) if phi_mse > 0 else float("inf")
        ),
        phi_l1=float((p_phi - g_phi).abs().mean()),
        flux_l1=float(abs(p_ef.sum() - g_ef.sum())),
        cr=float(gt.numel() * 4 / csize) if csize else float("inf"),
    )
    out.update(velocity_moment_errors(pred, gt, geom))
    return out


def subsample(ts, per_traj):
    """Pick `per_traj` timesteps evenly across a trajectory (0 = all)."""
    ts = sorted(ts)
    if not per_traj or len(ts) <= per_traj:
        return ts
    if per_traj == 1:
        return [ts[len(ts) // 2]]  # representative middle (turbulent) snapshot
    idx = np.linspace(0, len(ts) - 1, per_traj).round().astype(int)
    return [ts[i] for i in sorted(set(idx))]


def fname(outdir, label):
    """Per-method result file (label may contain '-', which is filename-safe)."""
    return os.path.join(outdir, f"metrics_{label}.json")


def summarize(outdir, labels):
    import statistics as st

    print(f"\n==== eval summary -> {outdir}/metrics_<method>.json ====")
    for lbl in labels:
        fp = fname(outdir, lbl)
        if not os.path.exists(fp):
            continue
        rows = json.load(open(fp))
        ok = [r for r in rows if "error" not in r]
        if not ok:
            print(f"  {lbl:10s}  (no successful rows of {len(rows)})")
            continue
        agg = {k: st.median([r[k] for r in ok]) for k in ("df_psnr", "phi_psnr", "flux_l1", "cr")}
        print(
            f"  {lbl:10s} n={len(ok):3d} median: df_psnr={agg['df_psnr']:.2f} "
            f"phi_psnr={agg['phi_psnr']:.2f} flux_l1={agg['flux_l1']:.4f} CR={agg['cr']:.0f}x"
        )
