"""Evaluation runner: scores each reconstructor over the test trajectories and
stores results so the notebooks only have to load and plot.

Results are pickled incrementally (per method) to ``out_dir``:
- ``full_metrics.pkl``: ``{method: {metric: (mean, std)}}``
- ``full_diagnostics.pkl``: ``{method: {traj: [per-timestep spectra dicts]}}``

Single-device by default. For scale, shard the trajectory list across GPUs with
one process per GPU (as in ``neugk/pinc/nf_main.py``) and merge the per-rank
pickles; ``evaluate_method`` is process-safe.
"""

import os
import pickle
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np

from neugk.pinc.neural_fields.data import CycloneNFDataset
from neugk.pinc.eval.metrics import (
    ml_eval,
    integrate,
    spectral_diagnostics,
    time_averaged_spectral_metrics,
    temporal_epe,
)
from neugk.pinc.eval.reconstructors import Reconstructor


def _gt_snapshots(gt: CycloneNFDataset):
    if gt.ndim > 5:
        return [gt.full_df[:, t] for t in range(gt.full_df.shape[1])]
    return [gt.full_df]


def evaluate_method(
    reconstructor: Reconstructor,
    trajectories: Sequence[str],
    timesteps: Sequence[int],
    path: str,
    backend: str = "gds",
    device: str = "cuda",
):
    metrics: Dict[str, list] = defaultdict(list)
    diagnostics_per_traj: Dict[str, list] = {}

    for traj in trajectories:
        gt = CycloneNFDataset(
            traj,
            timesteps=list(timesteps),
            path=path,
            backend=backend,
            realpotens=True,
            normalize=None,
        )
        gt_dfs = _gt_snapshots(gt)
        ds = gt.ds

        dfs, csize = reconstructor.reconstruct(traj, timesteps, gt, device)
        if csize:
            metrics["cr"].append(float(gt.full_df.nbytes) / csize)

        pred_diags, gt_diags = [], []
        for pred_df, gt_df in zip(dfs, gt_dfs):
            pred_diags.append(spectral_diagnostics(pred_df, gt.geom, ds))
            if reconstructor.name == "GT":
                continue
            gt_diags.append(spectral_diagnostics(gt_df, gt.geom, ds))
            p_phi, p_ef = integrate(pred_df, gt.geom)
            g_phi, g_ef = integrate(gt_df, gt.geom)
            for k, v in ml_eval(pred_df, gt_df, p_phi, g_phi, p_ef, g_ef).items():
                metrics[k].append(v)

        # store as {key: [per-timestep arrays]} for the visualization notebooks
        diagnostics_per_traj[traj] = (
            {k: [d[k] for d in pred_diags] for k in pred_diags[0]} if pred_diags else {}
        )
        if reconstructor.name != "GT":
            metrics.setdefault("endpoint", []).append(temporal_epe(gt_dfs, dfs))
            for k, v in time_averaged_spectral_metrics(pred_diags, gt_diags).items():
                metrics[k].append(v)

    agg = {
        k: (float(np.mean(v)), float(np.std(v))) for k, v in metrics.items() if len(v)
    }
    return agg, diagnostics_per_traj


def run_scaling(
    reconstructor_groups: List[tuple],  # List[(family_name, List[Reconstructor])]
    trajectories: Sequence[str],
    timesteps: Sequence[int],
    path: str,
    backend: str = "gds",
    device: str = "cuda",
    out_dir: str = "pinc_eval_results",
):
    """Evaluate multiple size variants per compression family for rate-distortion curves.

    Each reconstructor in a group is a different size / compression-rate variant
    of the same method family (e.g. NF at CR 1163, 2418, ... or AE at CR 302, 1208, ...).
    Results are saved as ``scaling.pkl``::

        {family_name: [{"cr": 1163, "psnr": 38.4, "l1": 0.023, ...}, ...]}

    sorted by CR within each family, so notebooks can load and plot directly.
    """
    os.makedirs(out_dir, exist_ok=True)
    scaling_path = os.path.join(out_dir, "scaling.pkl")
    scaling = (
        pickle.load(open(scaling_path, "rb")) if os.path.exists(scaling_path) else {}
    )

    for family_name, reconstructors in reconstructor_groups:
        if family_name in scaling:
            print(f"[scaling] {family_name}: already computed, skipping")
            continue
        entries = []
        for r in reconstructors:
            agg, _ = evaluate_method(r, trajectories, timesteps, path, backend, device)
            cr = agg.get("cr", (None,))[0]
            # flatten (mean, std) → mean only for rate-distortion plots
            entry = {"cr": cr, "name": r.name}
            entry.update({k: v[0] for k, v in agg.items()})
            entries.append(entry)
            print(
                f"[scaling]   {r.name}: cr={cr:.1f}" if cr else f"[scaling]   {r.name}"
            )
        entries.sort(key=lambda e: e.get("cr", 0) or 0)
        scaling[family_name] = entries
        # incremental save: survive long multi-family runs
        with open(scaling_path, "wb") as f:
            pickle.dump(scaling, f)
        print(f"[scaling] {family_name}: {len(entries)} variants → {scaling_path}")
    return scaling


def run_eval(
    reconstructors: List[Reconstructor],
    trajectories: Sequence[str],
    timesteps: Sequence[int],
    path: str,
    backend: str = "gds",
    device: str = "cuda",
    out_dir: str = "pinc_eval_results",
):
    os.makedirs(out_dir, exist_ok=True)
    m_path = os.path.join(out_dir, "full_metrics.pkl")
    d_path = os.path.join(out_dir, "full_diagnostics.pkl")
    full_m = pickle.load(open(m_path, "rb")) if os.path.exists(m_path) else {}
    full_d = pickle.load(open(d_path, "rb")) if os.path.exists(d_path) else {}

    for r in reconstructors:
        agg, diags = evaluate_method(r, trajectories, timesteps, path, backend, device)
        full_m[r.name], full_d[r.name] = agg, diags
        with open(m_path, "wb") as f:  # incremental: survive long multi-method runs
            pickle.dump(full_m, f)
        with open(d_path, "wb") as f:
            pickle.dump(full_d, f)
        print(f"[eval] {r.name}: {len(diags)} trajectories -> {m_path}")
    return full_m, full_d
