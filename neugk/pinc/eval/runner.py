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
import torch

from neugk.pinc.neural_fields.data import CycloneNFDataset
from neugk.pinc.eval.metrics import (
    ml_eval,
    integrate,
    spectral_diagnostics,
    time_averaged_spectral_metrics,
    temporal_epe,
)
from neugk.pinc.eval.reconstructors import Reconstructor
from neugk.physics.diagnostics import velocity_moment_errors


def _gt_snapshots(gt: CycloneNFDataset):
    if gt.ndim > 5:
        return [gt.full_df[:, t] for t in range(gt.full_df.shape[1])]
    return [gt.full_df]


@torch.no_grad()  # eval never needs gradients; without this the forwards retain
# autograd graphs and OOM the GPU on the wide low-CR nets (then illegal-access cascade).
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
        # run the metric physics (FluxIntegral FFTs, spectra) on-device; the FluxIntegral
        # bessel/i0 init stays cpu-isolated internally (B300 special-fn crash), the rest is gpu-safe.
        geom = {k: v.to(device) for k, v in gt.geom.items()}
        for pred_df, gt_df in zip(dfs, gt_dfs):
            pred_df, gt_df = pred_df.to(device), gt_df.to(device)
            pred_diags.append(spectral_diagnostics(pred_df, geom, ds))
            if reconstructor.name == "GT":
                continue
            gt_diags.append(spectral_diagnostics(gt_df, geom, ds))
            p_phi, p_ef = integrate(pred_df, geom)
            g_phi, g_ef = integrate(gt_df, geom)
            for k, v in ml_eval(pred_df, gt_df, p_phi, g_phi, p_ef, g_ef).items():
                metrics[k].append(v)
            # held-out velocity-moment reconstruction fidelity (not in training loss)
            for k, v in velocity_moment_errors(pred_df, gt_df, geom).items():
                metrics[k].append(v)

        # store as {key: [per-timestep arrays]} for the visualization notebooks.
        # spectra come back as torch tensors; convert to numpy here (storage boundary
        # only, not in the metric math). also keep the matching GT spectra inline
        # (kyspec_gt/qspec_gt/...) so the cascade plot can overlay pred vs GT.
        _np = lambda t: t.detach().cpu().numpy()
        traj_diag = (
            {k: [_np(d[k]) for d in pred_diags] for k in pred_diags[0]} if pred_diags else {}
        )
        if gt_diags:
            traj_diag.update(
                {f"{k}_gt": [_np(d[k]) for d in gt_diags] for k in gt_diags[0]}
            )
        diagnostics_per_traj[traj] = traj_diag
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
            # failproof: a variant that cannot reach its target CR (traditional knob
            # out of range) or whose NF was never trained must not kill the family.
            try:
                agg, _ = evaluate_method(r, trajectories, timesteps, path, backend, device)
            except Exception as e:
                print(f"[scaling]   {r.name}: skipped ({type(e).__name__}: {e})")
                continue
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
