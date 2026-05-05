"""Helpers for FID across multiple GyroSwin latent sources.

Builds the diffusion runner + the GyroSwin checkpoint side-by-side, samples
real validation snapshots and matched diffusion samples once, then re-uses the
same df batches to evaluate FID under several latent-extraction choices
(bottleneck, full multiscale flux_head, individual flux_head levels, ...).
"""
from __future__ import annotations

import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm

from notebooks.neurips_diff_eval import (
    compute_fid,
    compute_statistics,
    extract_gyroswin_latents,
    gyroswin_has_flux_head,
)
from notebooks.neurips_generate_table1 import (
    COND_META_MAP,
    _build_diff_runner,
    _notebook_safe,
    _traj_basename,
    free_cuda,
)
from neugk.dataset import get_data
from neugk.utils import expand_as


# ---------------------------------------------------------------------------
# Latent-source spec
# ---------------------------------------------------------------------------
@dataclass
class LatentSource:
    """One latent-extraction recipe to evaluate.

    name  : short label used in plots and the table
    source: 'bottleneck', 'flux_head', 'decoder', 'skip', or 'phi'
    level : interpretation depends on `source`
        * 'bottleneck' / 'phi' — ignored
        * 'flux_head'  — multiscale level index (0 = bottleneck mix, 1+ = decoder
                         up-blocks). None -> concat all levels (default).
        * 'decoder'    — Pythonic up-block index. None -> -1 (last up-block,
                         one level before the 5D output, à la InceptionV3 pool3).
        * 'skip'       — Pythonic down-block skip index. None -> deepest skip.
    pool  : optional spatial reduction passed to extract_gyroswin_latents
            for sources that accept it ('bottleneck', 'skip', 'phi'). Use
            'amax' or 'mean' to collapse the 5D activation to a per-channel
            vector — strongly recommended at low sample counts (FID with
            K~64 vs raw flattened activations is under-determined).
    """
    name: str
    source: str
    level: Optional[int] = None
    pool: Optional[str] = None

    def kwargs(self):
        kw = {"source": self.source}
        if self.source == "flux_head" and self.level is not None:
            kw["flux_head_level"] = self.level
        elif self.source == "decoder":
            kw["decoder_level"] = -1 if self.level is None else self.level
        elif self.source == "skip" and self.level is not None:
            kw["decoder_level"] = self.level
        if self.pool is not None:
            kw["pool"] = self.pool
        return kw


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
def _load_old_gyroswin(checkpoint_dir, trainset, device):
    """Load the legacy monkey-patched GyroSwin checkpoint."""
    sys.path.insert(0, str(Path(__file__).parent))
    from neurips_gyroswin_eval import load_gyroswin_model
    gs_model, gs_cfg, _ = load_gyroswin_model(
        checkpoint_dir, dataset=trainset, device=device,
    )
    return gs_model, gs_cfg


def _load_new_gyroswin(checkpoint_dir, trainset, device, model_snapshot="best.pth"):
    """Load a current-codebase GyroSwin via `neugk.gyroswin.models.get_model`.
    Reuses the diffusion runner's trainset (matching `active_keys` /
    `resolution`); GyroSwin-specific knobs come from the checkpoint cfg."""
    import omegaconf
    from neugk.gyroswin.models import get_model as get_gyroswin_model
    cfg = omegaconf.OmegaConf.load(os.path.join(checkpoint_dir, "config.yaml"))
    gs_model = get_gyroswin_model(cfg, dataset=trainset).to(device).eval()
    ckpt = torch.load(os.path.join(checkpoint_dir, model_snapshot),
                      map_location=device, weights_only=False)
    gs_model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=True)
    return gs_model, cfg


def _load_old_gs_norm_stats(checkpoint_dir):
    """Old GyroSwin checkpoints bundle `normalization_stats.pkl` directly."""
    path = os.path.join(checkpoint_dir, "normalization_stats.pkl")
    with open(path, "rb") as f:
        stats = pickle.load(f)
    # Expected layout: stats[<field>]["full"]["mean"|"std"] (numpy arrays).
    return stats


def _load_new_gs_norm_stats(checkpoint_dir, data_path):
    """New (current-codebase) GyroSwin checkpoint: build its trainset just to
    read the stats pkl that the dataset normalizer auto-loads. The trainset
    is dropped immediately afterwards."""
    import omegaconf
    cfg = _notebook_safe(omegaconf.OmegaConf.load(
        os.path.join(checkpoint_dir, "config.yaml"),
    ))
    cfg.dataset.path = str(data_path)
    cfg.dataset.gds_override = True
    print(f"  loading GS norm stats for new ckpt ({checkpoint_dir}) ...")
    datasets, _, _ = get_data(cfg, rank=0)
    trainset = datasets[0]
    stats = {
        "df": {"full": {
            "mean": np.asarray(trainset.stats["df"]["full"]["mean"]),
            "std":  np.asarray(trainset.stats["df"]["full"]["std"]),
        }}
    }
    del datasets, trainset
    free_cuda()
    return stats


def _gs_stats_to_tensors(gs_stats, ref_tensor):
    """Convert GS df stats into (mean, std) tensors broadcast-compatible with
    `ref_tensor`. `ref_tensor` is just used for dtype/device; the shapes follow
    the underlying numpy arrays in `gs_stats` and `expand_as` prepends size-1
    dims to match `ref_tensor`'s rank.
    """
    mean = torch.as_tensor(
        np.asarray(gs_stats["df"]["full"]["mean"]),
        dtype=ref_tensor.dtype, device=ref_tensor.device,
    )
    std = torch.as_tensor(
        np.asarray(gs_stats["df"]["full"]["std"]),
        dtype=ref_tensor.dtype, device=ref_tensor.device,
    )
    return expand_as(mean, ref_tensor), expand_as(std, ref_tensor)


def setup(
    diff_ckpt_dir,
    ae_checkpoint,
    data_path,
    valid_traj_h5_names,
    device,
    *,
    gyroswin_checkpoint=None,       # legacy (monkey-patched) checkpoint
    gyroswin_checkpoint_new=None,   # current-codebase checkpoint
    model_snapshot="best.pth",
):
    """Build the diffusion runner and load 1-2 GyroSwin variants for FID.

    Returns
    -------
    runner : diffusion runner used to collect real + matched-diff df samples.
    gyroswins : dict[str, dict] keyed by variant ('old' for the monkey-patched
        legacy checkpoint, 'new' for the current-codebase checkpoint). Each
        value is ``{"model": nn.Module, "cfg": DictConfig, "cond_keys": [...]}``.
        Variants whose checkpoint is None are omitted; at least one is required.
    """
    if not gyroswin_checkpoint and not gyroswin_checkpoint_new:
        raise ValueError(
            "supply at least one of gyroswin_checkpoint / gyroswin_checkpoint_new"
        )

    runner = _build_diff_runner(
        diff_ckpt_dir, ae_checkpoint, data_path,
        valid_traj_h5_names, model_snapshot, device,
    )

    gyroswins = {}
    if gyroswin_checkpoint:
        gs_model, gs_cfg = _load_old_gyroswin(
            gyroswin_checkpoint, runner.trainset, device,
        )
        if not gyroswin_has_flux_head(gs_model):
            raise RuntimeError(
                f"GyroSwin (old) at {gyroswin_checkpoint} has no flux_head; "
                "only source='bottleneck' will work."
            )
        gyroswins["old"] = {
            "model": gs_model, "cfg": gs_cfg,
            "cond_keys": sorted(list(gs_cfg.model.conditioning)),
            "norm_stats": _load_old_gs_norm_stats(gyroswin_checkpoint),
        }
    if gyroswin_checkpoint_new:
        gs_model, gs_cfg = _load_new_gyroswin(
            gyroswin_checkpoint_new, runner.trainset, device, model_snapshot,
        )
        if not gyroswin_has_flux_head(gs_model):
            raise RuntimeError(
                f"GyroSwin (new) at {gyroswin_checkpoint_new} has no flux_head."
            )
        gyroswins["new"] = {
            "model": gs_model, "cfg": gs_cfg,
            "cond_keys": sorted(list(gs_cfg.model.conditioning)),
            "norm_stats": _load_new_gs_norm_stats(
                gyroswin_checkpoint_new,
                data_path=runner.cfg.dataset.path,
            ),
        }
    return runner, gyroswins


# ---------------------------------------------------------------------------
# Sample collection
# ---------------------------------------------------------------------------
def _traj_label(fpath):
    m = re.search(r"iteration_(\d+)", fpath)
    return f"iter_{m.group(1)}" if m else os.path.basename(fpath)


def _denormalize_with_dataset(dataset, fi, df_norm):
    """Convert a per-fi normalized df tensor into physical units using the
    dataset's stored scale/shift -- bypasses any AE decode the Cyclone-AE
    valset does in `denormalize` (we already have a 5D df, not a latent)."""
    scale, shift = dataset._get_scale_shift(fi, "df", df_norm)
    return df_norm * scale + shift


def collect_real(runner, n_per_traj=None, max_total=None, seed=0,
                 physical=False):
    """Stratified pick of validation snapshots: up to `n_per_traj` evenly
    spaced snapshots per trajectory, optionally capped at `max_total`.

    Returns a dict keyed by file index `fi` with:
        df         : list[Tensor]   model-space df (normalized, ready for forward)
                                    or physical-space df if `physical=True`
        flat_idx   : list[int]      flat valset index for each entry
        label      : str            "iter_<id>" label

    Pass `physical=True` if you intend to feed `extract_features` a
    `gs_norm_stats` argument so it can renormalize into the GyroSwin
    checkpoint's space.
    """
    valset = runner.valsets[0]
    by_fi = defaultdict(list)
    for idx in range(len(valset)):
        fi, _ = valset.flat_index_to_file_and_tstep[idx]
        by_fi[fi].append(idx)

    sel_per_fi = {}
    for fi, idxs in by_fi.items():
        if n_per_traj is None or len(idxs) <= n_per_traj:
            sel_per_fi[fi] = list(idxs)
        else:
            step = len(idxs) / n_per_traj
            sel_per_fi[fi] = [idxs[int(k * step)] for k in range(n_per_traj)]

    if max_total is not None:
        flat = [(fi, i) for fi in sel_per_fi for i in sel_per_fi[fi]]
        if len(flat) > max_total:
            rng = np.random.default_rng(seed)
            keep = rng.choice(len(flat), size=max_total, replace=False)
            keep_set = {flat[k] for k in keep}
            sel_per_fi = {fi: [i for i in sel_per_fi[fi] if (fi, i) in keep_set]
                          for fi in sel_per_fi}

    out = {}
    for fi, idxs in sorted(sel_per_fi.items()):
        if not idxs:
            continue
        dfs = [valset[i].df for i in idxs]
        if physical:
            dfs = [_denormalize_with_dataset(valset, fi, t) for t in dfs]
        out[fi] = {
            "df": dfs,
            "flat_idx": list(idxs),
            "label": _traj_label(valset.files[fi]),
        }
    n_total = sum(len(v["df"]) for v in out.values())
    print(f"  real samples: {n_total} across {len(out)} trajs "
          f"({ {fi: len(v['df']) for fi, v in out.items()} })  "
          f"[{'physical' if physical else 'AE-normalized'}]")
    return out


def split_by_traj_set(by_fi, runner, trajectories_id, trajectories_ood,
                      trajectories_test=None):
    """Partition a `{fi: ...}` dict (real or gen) into ID/OOD/TEST subsets,
    matching basenames against the trajectory lists.

    Returns ``{"ID": {fi: entry}, "OOD": ..., "TEST": ...}``. Splits whose
    trajectory list is empty are still emitted as empty dicts so downstream
    code can rely on the keys. Trajectories that match no list are dropped
    (with a warning).

    Backward-compatible: callers passing only ID + OOD get the same two
    splits as before, plus an empty ``"TEST"`` entry.
    """
    valset = runner.valsets[0]
    id_bases   = {t.replace("_ifft_realpotens", "") for t in (trajectories_id or [])}
    ood_bases  = {t.replace("_ifft_realpotens", "") for t in (trajectories_ood or [])}
    test_bases = {t.replace("_ifft_realpotens", "") for t in (trajectories_test or [])}
    out = {"ID": {}, "OOD": {}, "TEST": {}}
    skipped = []
    for fi, entry in by_fi.items():
        base = _traj_basename(valset.files[fi])
        if base in id_bases:
            out["ID"][fi] = entry
        elif base in ood_bases:
            out["OOD"][fi] = entry
        elif base in test_bases:
            out["TEST"][fi] = entry
        else:
            skipped.append(base)
    if skipped:
        print(f"  [split] dropped {len(skipped)} unmatched trajectories: {skipped}")
    print(f"  [split] ID: {len(out['ID'])} trajs | OOD: {len(out['OOD'])} trajs"
          f" | TEST: {len(out['TEST'])} trajs")
    return out


@torch.no_grad()
def generate_diff(runner, real_by_fi, n_denoising_steps=15, batch_size=32,
                  physical=False):
    """For each trajectory in `real_by_fi`, sample as many diffusion samples as
    real ones (using that trajectory's conditioning). Returns the same dict
    layout but with key 'df' holding the decoded gen df tensors (cpu).

    Pass `physical=True` to denormalize the decoded df via the diffusion
    runner's trainset stats; combine with `extract_features(...,
    gs_norm_stats=...)` to evaluate FID in GyroSwin-checkpoint space."""
    valset = runner.valsets[0]
    cond_keys = sorted(runner.cfg.model.conditioning)

    out = {}
    for fi, entry in real_by_fi.items():
        meta = valset.metadata[fi]
        cond_vals = [float(np.squeeze(meta[COND_META_MAP.get(k, k)])) for k in cond_keys]
        cond_t = torch.tensor(cond_vals, dtype=torch.float32, device=runner.device)
        n = len(entry["df"])

        gen_dfs = []
        for i in range(0, n, batch_size):
            bs = min(batch_size, n - i)
            c = cond_t.unsqueeze(0).expand(bs, -1)
            decoded = runner.sample(c, steps=n_denoising_steps, latent_only=False)
            for b in range(bs):
                t = decoded["df"][b].cpu()
                if physical:
                    t = _denormalize_with_dataset(valset, fi, t)
                gen_dfs.append(t)
        out[fi] = {"df": gen_dfs, "label": entry["label"]}
    return out


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def _trajectory_condition(runner, keys, fi, n):
    """Build the (n, len(keys)) per-trajectory conditioning tensor.

    `keys` should NOT include 'timestep' — that's a per-snapshot scalar that
    `extract_gyroswin_latents` fills from `default_timestep` when absent.
    """
    meta = runner.valsets[0].metadata[fi]
    vals = [float(np.squeeze(meta[COND_META_MAP.get(k, k)])) for k in keys]
    return torch.tensor(vals, dtype=torch.float32).unsqueeze(0).expand(n, -1).contiguous()


@torch.no_grad()
def extract_features(
    gs_model, by_fi, runner, gs_cond_keys, latent_source: LatentSource,
    batch_size=8, device="cuda", desc="extract",
    gs_norm_stats=None,
):
    """Run all df samples through the GyroSwin extractor for one latent source.

    If `gs_norm_stats` is given, `by_fi` is assumed to hold **physical-space**
    df tensors and they are renormalized into GyroSwin-checkpoint space
    (`(df - gs_mean) / gs_std`) right before the forward pass — i.e. fed to
    GyroSwin in the exact normalization it was trained on. If `gs_norm_stats`
    is None we feed `by_fi` through as-is (legacy behaviour).

    Returns dict: fi -> {'feats': (n, D) np.ndarray, 'label': str}.
    """
    # 'timestep' is a per-snapshot scalar pulled from a different metadata
    # field (`meta["timesteps"]`); drop it from the per-trajectory condition
    # vector and let extract_gyroswin_latents fill it with default_timestep.
    nontime_keys = [k for k in gs_cond_keys if k != "timestep"]

    out = {}
    extractor_kw = latent_source.kwargs()
    for fi, entry in by_fi.items():
        dfs = entry["df"]
        n = len(dfs)
        cond_t = _trajectory_condition(runner, nontime_keys, fi, n)
        feats = []
        for i in range(0, n, batch_size):
            batch_df = torch.stack(dfs[i:i+batch_size])
            if gs_norm_stats is not None:
                gs_mean, gs_std = _gs_stats_to_tensors(gs_norm_stats, batch_df)
                batch_df = (batch_df - gs_mean) / gs_std
            batch_cond = cond_t[i:i+batch_size]
            f = extract_gyroswin_latents(
                gs_model, batch_df, device=device, condition=batch_cond,
                cond_keys=nontime_keys, **extractor_kw,
            )
            feats.append(f)
        out[fi] = {"feats": np.concatenate(feats, axis=0), "label": entry["label"]}
    return out


# ---------------------------------------------------------------------------
# FID matrices
# ---------------------------------------------------------------------------
def _fit_pca(feats_dict, n_components):
    pooled = np.concatenate([v["feats"] for v in feats_dict.values()], axis=0)
    n = min(n_components, pooled.shape[0], pooled.shape[1])
    pca = PCA(n_components=n).fit(pooled)
    return pca, pooled.shape[1]


def compute_fid_set(real_feats, gen_feats, n_components=64):
    """Compute global FID, GT-vs-GT pairwise FID, and GT-vs-Pred pairwise FID.

    PCA is fit on the pooled real+gen features so all matrices share a
    common reduction.
    """
    fis = sorted(real_feats.keys())
    labels = [real_feats[fi]["label"] for fi in fis]
    n_traj = len(fis)

    # shared PCA over real+gen so all FIDs are on the same axes
    combined = {f"r{fi}": real_feats[fi] for fi in fis}
    combined.update({f"g{fi}": gen_feats[fi] for fi in fis})
    pca, raw_dim = _fit_pca(combined, n_components)

    real_pca = {fi: pca.transform(real_feats[fi]["feats"]) for fi in fis}
    gen_pca  = {fi: pca.transform(gen_feats[fi]["feats"])  for fi in fis}

    # Global: pool everything
    real_all = np.concatenate([real_pca[fi] for fi in fis], axis=0)
    gen_all  = np.concatenate([gen_pca[fi]  for fi in fis], axis=0)
    mu_r, sig_r = compute_statistics(real_all)
    mu_g, sig_g = compute_statistics(gen_all)
    fid_global = compute_fid(mu_r, sig_r, mu_g, sig_g)

    # Per-traj stats
    stats_real = {fi: compute_statistics(real_pca[fi]) for fi in fis}
    stats_gen  = {fi: compute_statistics(gen_pca[fi])  for fi in fis}

    fid_gt_gt = np.full((n_traj, n_traj), np.nan)
    fid_gt_diff = np.full((n_traj, n_traj), np.nan)
    for i, fi in enumerate(fis):
        mu_ri, cov_ri = stats_real[fi]
        for j, fj in enumerate(fis):
            mu_rj, cov_rj = stats_real[fj]
            mu_gj, cov_gj = stats_gen[fj]
            fid_gt_gt[i, j]   = compute_fid(mu_ri, cov_ri, mu_rj, cov_rj)
            fid_gt_diff[i, j] = compute_fid(mu_ri, cov_ri, mu_gj, cov_gj)

    return {
        "labels": labels,
        "fid_global": fid_global,
        "fid_gt_gt": fid_gt_gt,
        "fid_gt_diff": fid_gt_diff,
        "raw_dim": raw_dim,
        "pca_dim": pca.n_components_,
        "explained_var": float(pca.explained_variance_ratio_.sum()),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _draw_heatmap(ax, M, labels, title, fmt="{:.1f}"):
    n = len(labels)
    vmax = np.nanmax(M) if np.isfinite(M).any() else 1.0
    im = ax.imshow(M, cmap="YlOrRd", vmin=0, vmax=vmax)
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=8)
    for i in range(n):
        for j in range(n):
            if np.isfinite(M[i, j]):
                color = "white" if M[i, j] > 0.6 * vmax else "black"
                weight = "bold" if i == j else "normal"
                ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center",
                        fontsize=8, fontweight=weight, color=color)
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, label="FID")


def _result_label(key):
    """Render a result-dict key as a short string for plot/table labels.
    Accepts either a plain str or a (latent, split) tuple."""
    if isinstance(key, tuple):
        return f"{key[0]} ({key[1]})"
    return str(key)


def plot_heatmap_grid(results, title=None):
    """Plot a 2-column grid: rows = latent sources (or (latent, split) keys),
    col 0 = GT-vs-GT, col 1 = GT-vs-Diff."""
    n_rows = len(results)
    fig, axes = plt.subplots(n_rows, 2, figsize=(13, 6 * n_rows), squeeze=False)
    for r, (key, res) in enumerate(results.items()):
        name = _result_label(key)
        labels = res["labels"]
        _draw_heatmap(
            axes[r, 0], res["fid_gt_gt"], labels,
            f"{name}: GT vs GT  (raw dim={res['raw_dim']}, "
            f"pca={res['pca_dim']}, var={res['explained_var']:.0%})",
        )
        _draw_heatmap(
            axes[r, 1], res["fid_gt_diff"], labels,
            f"{name}: GT vs Diff  (global FID = {res['fid_global']:.2f})",
        )
        axes[r, 1].set_xlabel("Diff (col traj conditioning)", fontsize=9)
        axes[r, 1].set_ylabel("GT (row traj)", fontsize=9)
    if title:
        fig.suptitle(title, fontsize=12, y=1.001)
    fig.tight_layout()
    return fig


def build_table(results):
    """Per-(latent, split) summary table: global FID, mean diagonal, mean
    off-diagonal. Keys may be plain str or (latent, split) tuples; the
    output index reflects that structure."""
    rows = []
    has_split = any(isinstance(k, tuple) for k in results)
    for key, res in results.items():
        gt_diff = res["fid_gt_diff"]
        n = gt_diff.shape[0]
        diag = np.diag(gt_diff)
        off = gt_diff.copy(); np.fill_diagonal(off, np.nan)
        gt_gt = res["fid_gt_gt"]
        gt_gt_off = gt_gt.copy(); np.fill_diagonal(gt_gt_off, np.nan)

        if isinstance(key, tuple):
            latent, split = key
        else:
            latent, split = key, None

        rows.append({
            "latent":                 latent,
            "split":                  split,
            "raw_dim":                res["raw_dim"],
            "pca_dim":                res["pca_dim"],
            "var_explained":          res["explained_var"],
            "global_FID":             res["fid_global"],
            "diag_FID(GT_i,Diff_i)":  float(np.nanmean(diag)),
            "off_FID(GT_i,Diff_j)":   float(np.nanmean(off))    if n > 1 else np.nan,
            "GT-GT off-diag":         float(np.nanmean(gt_gt_off)) if n > 1 else np.nan,
        })
    df = pd.DataFrame(rows)
    index_cols = ["latent", "split"] if has_split else ["latent"]
    if not has_split:
        df = df.drop(columns=["split"])
    return df.set_index(index_cols)
