"""Per-source df samplers for the Table-2 FID notebook.

Each `sample_<source>` function builds the model, generates df samples that
match the per-trajectory layout of the diffusion-runner valset, **denormalizes
them back to physical units using the source AE's own training stats**, and
tears the model down. Output layout is the same
``{fi: {"df": list[Tensor], "label": str}}`` dict that
``notebooks.neurips_fid_gyroswin_latents.extract_features`` consumes.

Why physical space: the FID feature extractor (`extract_features`) takes a
`gs_norm_stats=` argument; when set, it renormalizes the physical-space df
into the GyroSwin checkpoint's normalization right before the forward pass.
That's the normalization GyroSwin was actually trained on, so the features
are 1:1 comparable across all 7 generative sources.

Cached AE-trainset stats (module-level dict) are shared across calls, so
running both `sample_vqvae` and `sample_ar` only builds the VQ-VAE trainset
once instead of twice.
"""
from __future__ import annotations

import gc
import os
import pickle
from pathlib import Path

import numpy as np
import omegaconf
import torch
from tqdm import tqdm

from neugk.dataset import get_data
from neugk.pinc.autoencoders.ae_utils import load_autoencoder
from neugk.pinc.autoencoders.gk_autoencoders import Swin5DVAE, Swin5DVQVAE
from neugk.pinc.generate import (
    compute_codebook_prior,
    get_conditioning,
    sample_vae_prior,
    sample_vqvae_random,
)

from notebooks.neurips_generate_table1 import (
    COND_META_MAP,
    _ar_sample_decode,
    _build_ar_model,
    _build_diff_runner,
    _notebook_safe,
    _release_module,
    free_cuda,
)
from notebooks.neurips_diff_eval import compute_fid, compute_statistics


# ---------------------------------------------------------------------------
# AE-trainset stats cache (avoids rebuilding the same trainset twice when
# multiple samplers share an AE checkpoint, e.g. VQ-VAE for both `sample_vqvae`
# and `sample_ar`).
# ---------------------------------------------------------------------------
_AE_STATS_CACHE = {}


def _ae_df_stats(ckpt_dir, data_path):
    """Return the AE trainset's df mean/std numpy arrays. Cached per
    `(realpath(ckpt_dir), realpath(data_path))`."""
    key = (os.path.realpath(ckpt_dir),
           os.path.realpath(str(data_path)) if data_path else None)
    if key in _AE_STATS_CACHE:
        return _AE_STATS_CACHE[key]
    train_cfg = _notebook_safe(omegaconf.OmegaConf.load(
        os.path.join(ckpt_dir, "config.yaml"),
    ))
    if data_path is not None:
        train_cfg.dataset.path = str(data_path)
    print(f"  loading AE trainset stats ({ckpt_dir}) ...")
    datasets, _, _ = get_data(train_cfg, rank=0)
    trainset = datasets[0]
    mean = np.asarray(trainset.stats["df"]["full"]["mean"])
    std  = np.asarray(trainset.stats["df"]["full"]["std"])
    del datasets, trainset
    gc.collect()
    _AE_STATS_CACHE[key] = (mean, std)
    return mean, std


def clear_ae_stats_cache():
    """Drop the cached AE-trainset stats (call this if you re-run the
    notebook end-to-end with different paths)."""
    _AE_STATS_CACHE.clear()


def _denormalize(df_norm, mean_np, std_np):
    """Convert AE-normalized df to physical: x_phys = x * std + mean. The
    numpy stats are broadcast-aligned to df_norm via `expand_as` (prepends
    size-1 dims until ranks match)."""
    from neugk.utils import expand_as
    mean = expand_as(torch.as_tensor(mean_np, dtype=df_norm.dtype, device=df_norm.device), df_norm)
    std  = expand_as(torch.as_tensor(std_np,  dtype=df_norm.dtype, device=df_norm.device), df_norm)
    return df_norm * std + mean


def _build_pinc_ae(ckpt_dir, data_path, device):
    """Load the PINC AE without rebuilding its trainset (we only need its
    stats, which `_ae_df_stats` caches separately)."""
    train_cfg = _notebook_safe(omegaconf.OmegaConf.load(
        os.path.join(ckpt_dir, "config.yaml"),
    ))
    if data_path is not None:
        train_cfg.dataset.path = str(data_path)
    print(f"  loading AE weights ({ckpt_dir}) ...")
    model, _, _ = load_autoencoder(ckpt_dir, device)
    return model.to(device).eval(), train_cfg


def _decoder_cond_keys(train_cfg):
    """Effective decoder conditioning for a PINC AE config (matches
    `evaluate_generative`'s logic)."""
    model_cfg = (train_cfg.autoencoder if "autoencoder" in train_cfg
                 else train_cfg.model)
    enc = list(getattr(model_cfg, "encoder_conditioning", []) or [])
    dec = list(getattr(model_cfg, "decoder_conditioning", []) or [])
    return sorted(set(enc) | set(dec))


def _vae_latent_stats(ckpt_dir, device):
    """Load (mean, std) for the VAE prior. Returns (0, 1) if no precomputed
    stats are bundled with the checkpoint."""
    new = os.path.join(ckpt_dir, "train_latent_stats.pt")
    old = os.path.join(ckpt_dir, "train_latent_stats.pkl")
    if os.path.exists(new):
        s = torch.load(new, weights_only=True)
        return s["z_mean"].to(device), s["z_std"].to(device)
    if os.path.exists(old):
        d = pickle.load(open(old, "rb"))
        arr = np.concatenate([d[k]["x"] for k in d.keys()], axis=0)
        return (torch.tensor(np.mean(arr, axis=0), device=device),
                torch.tensor(np.std(arr, axis=0), device=device))
    return torch.tensor(0.0, device=device), torch.tensor(1.0, device=device)


# ---------------------------------------------------------------------------
# PINC-family samplers (VAE / VQ-VAE / VQ-VAE+AR)
# ---------------------------------------------------------------------------
@torch.no_grad()
def sample_vae(vae_ckpt_dir, ref_runner, real_by_fi, *,
               n_per_traj=32, batch_size=32, data_path=None,
               device=torch.device("cuda")):
    """Sample VAE prior per traj; return physical-space df dict."""
    df_mean, df_std = _ae_df_stats(vae_ckpt_dir, data_path)
    model, train_cfg = _build_pinc_ae(vae_ckpt_dir, data_path, device)
    assert isinstance(model, Swin5DVAE), \
        f"Expected VAE at {vae_ckpt_dir}, got {type(model).__name__}"
    pad_axes = model.get_pad_axes(model.base_resolution)
    cond_keys = _decoder_cond_keys(train_cfg)
    z_mean, z_std = _vae_latent_stats(vae_ckpt_dir, device)
    valset = ref_runner.valsets[0]

    out = {}
    try:
        for fi, entry in tqdm(list(real_by_fi.items()), desc="  vae sample"):
            meta = valset.metadata[fi]
            cond = (get_conditioning(cond_keys, meta, device)
                    if cond_keys else None)
            dfs = []
            remaining = int(n_per_traj)
            while remaining > 0:
                bs = min(batch_size, remaining)
                gen = sample_vae_prior(model, bs, cond, pad_axes, device,
                                       mean=z_mean, std=z_std).float()
                for b in range(bs):
                    dfs.append(_denormalize(gen[b].cpu(), df_mean, df_std))
                remaining -= bs
            out[fi] = {"df": dfs, "label": entry["label"]}
    finally:
        _release_module(model)
        del model
        free_cuda()
    return out


@torch.no_grad()
def sample_vqvae(vqvae_ckpt_dir, vq_index_pkl, ref_runner, real_by_fi, *,
                 n_per_traj=32, batch_size=32, data_path=None,
                 device=torch.device("cuda")):
    """Sample VQ-VAE codes from the empirical codebook prior; return
    physical-space df dict."""
    df_mean, df_std = _ae_df_stats(vqvae_ckpt_dir, data_path)
    model, train_cfg = _build_pinc_ae(vqvae_ckpt_dir, data_path, device)
    assert isinstance(model, Swin5DVQVAE), \
        f"Expected VQ-VAE at {vqvae_ckpt_dir}, got {type(model).__name__}"
    pad_axes = model.get_pad_axes(model.base_resolution)
    cond_keys = _decoder_cond_keys(train_cfg)
    vq_prior = compute_codebook_prior(vq_index_pkl, model.vq.codebook_size) \
        if vq_index_pkl else None
    valset = ref_runner.valsets[0]

    out = {}
    try:
        for fi, entry in tqdm(list(real_by_fi.items()), desc="  vqvae sample"):
            meta = valset.metadata[fi]
            cond = (get_conditioning(cond_keys, meta, device)
                    if cond_keys else None)
            dfs = []
            remaining = int(n_per_traj)
            while remaining > 0:
                bs = min(batch_size, remaining)
                gen = sample_vqvae_random(model, bs, cond, pad_axes, device,
                                          prior=vq_prior).float()
                for b in range(bs):
                    dfs.append(_denormalize(gen[b].cpu(), df_mean, df_std))
                remaining -= bs
            out[fi] = {"df": dfs, "label": entry["label"]}
    finally:
        _release_module(model)
        del model
        free_cuda()
    return out


@torch.no_grad()
def sample_ar(ar_ckpt_dir, ae_checkpoint, ref_runner, real_by_fi, *,
              n_per_traj=32, batch_size=8, data_path=None,
              device=torch.device("cuda")):
    """Sample VQ-VAE-decoded df from the AR transformer; return physical-space
    df dict. The AE used for decoding is `ae_checkpoint`, so we denormalize
    with the *AE's* stats (cached -- shared with `sample_vqvae` if you call
    both)."""
    df_mean, df_std = _ae_df_stats(ae_checkpoint, data_path)
    ar_model, ae_model, ar_cfg, _ = _build_ar_model(
        ar_ckpt_dir, device, data_path=data_path, ae_checkpoint=ae_checkpoint,
    )
    cond_keys = sorted(ar_cfg.model.conditioning)
    valset = ref_runner.valsets[0]

    out = {}
    try:
        for fi, entry in tqdm(list(real_by_fi.items()), desc="  ar sample"):
            meta = valset.metadata[fi]
            cond = (get_conditioning(cond_keys, meta, device)
                    if cond_keys else None)
            dfs = []
            remaining = int(n_per_traj)
            while remaining > 0:
                bs = min(batch_size, remaining)
                gen = _ar_sample_decode(ar_model, ae_model, ar_cfg,
                                        cond, bs, device).float()
                for b in range(bs):
                    dfs.append(_denormalize(gen[b].cpu(), df_mean, df_std))
                remaining -= bs
            out[fi] = {"df": dfs, "label": entry["label"]}
    finally:
        _release_module(ar_model)
        _release_module(ae_model)
        del ar_model, ae_model
        free_cuda()
    return out


# ---------------------------------------------------------------------------
# Diffusion samplers (return physical-space df via the runner's valset stats)
# ---------------------------------------------------------------------------
def _denormalize_via_dataset(dataset, fi, df_norm):
    """Stat-only denormalize against a Cyclone(AE) dataset, bypassing the
    autoencoder.decode path that `dataset.denormalize` takes for AE valsets."""
    scale, shift = dataset._get_scale_shift(fi, "df", df_norm)
    return df_norm * scale + shift


@torch.no_grad()
def sample_diff_inplace(ref_runner, real_by_fi, *,
                        n_per_traj=32, n_denoising_steps=15, batch_size=32):
    """Sample the *reference* diffusion runner directly. Output is in
    physical units (denormalized via ref_runner.valsets[0] stats)."""
    valset = ref_runner.valsets[0]
    cond_keys = sorted(ref_runner.cfg.model.conditioning)
    out = {}
    for fi, entry in tqdm(list(real_by_fi.items()), desc="  diff (ref) sample"):
        meta = valset.metadata[fi]
        cond_vals = [float(np.squeeze(meta[COND_META_MAP.get(k, k)])) for k in cond_keys]
        cond_t = torch.tensor(cond_vals, dtype=torch.float32, device=ref_runner.device)
        dfs = []
        remaining = int(n_per_traj)
        while remaining > 0:
            bs = min(batch_size, remaining)
            c = cond_t.unsqueeze(0).expand(bs, -1)
            decoded = ref_runner.sample(c, steps=n_denoising_steps, latent_only=False)
            for b in range(bs):
                dfs.append(_denormalize_via_dataset(valset, fi, decoded["df"][b].cpu()))
            remaining -= bs
        out[fi] = {"df": dfs, "label": entry["label"]}
    return out


@torch.no_grad()
def sample_diff(diff_ckpt_dir, ae_checkpoint, ref_runner, real_by_fi, *,
                n_per_traj=32, n_denoising_steps=15, batch_size=32,
                data_path=None,
                device=torch.device("cuda"), model_snapshot="best.pth"):
    """Build a fresh diffusion runner for a non-ref variant (EDM / DDPM /
    secondary FLOW), generate samples per traj, return physical-space dfs.

    Denormalization uses **the new runner's** valset stats, not ref_runner's.
    This is the correct behaviour even when the underlying AE differs."""
    valid_h5 = [os.path.basename(f) for f in ref_runner.valsets[0].files]
    runner = _build_diff_runner(diff_ckpt_dir, ae_checkpoint, data_path,
                                valid_h5, model_snapshot, device)
    cond_keys = sorted(runner.cfg.model.conditioning)
    new_valset = runner.valsets[0]
    ref_valset = ref_runner.valsets[0]
    # Map fi (in ref valset) -> fi' (in new valset) by basename.
    base_to_new_fi = {os.path.basename(f): i for i, f in enumerate(new_valset.files)}

    out = {}
    try:
        label = os.path.basename(diff_ckpt_dir)
        for fi, entry in tqdm(list(real_by_fi.items()), desc=f"  {label} sample"):
            meta = ref_valset.metadata[fi]
            new_fi = base_to_new_fi.get(os.path.basename(ref_valset.files[fi]), fi)
            cond_vals = [float(np.squeeze(meta[COND_META_MAP.get(k, k)])) for k in cond_keys]
            cond_t = torch.tensor(cond_vals, dtype=torch.float32, device=runner.device)
            dfs = []
            remaining = int(n_per_traj)
            while remaining > 0:
                bs = min(batch_size, remaining)
                c = cond_t.unsqueeze(0).expand(bs, -1)
                decoded = runner.sample(c, steps=n_denoising_steps, latent_only=False)
                for b in range(bs):
                    dfs.append(_denormalize_via_dataset(new_valset, new_fi, decoded["df"][b].cpu()))
                remaining -= bs
            out[fi] = {"df": dfs, "label": entry["label"]}
    finally:
        _release_module(getattr(runner, "model", None))
        _release_module(getattr(runner, "autoencoder", None))
        del runner
        free_cuda()
    return out


# ---------------------------------------------------------------------------
# Nearest-neighbour-in-training-set sampler
# ---------------------------------------------------------------------------
@torch.no_grad()
def sample_nn(ref_runner, real_by_fi, *,
              n_per_traj=32, saturated_phase_start=120, seed=42,
              min_dist=1e-6, verbose=False):
    """For each eval trajectory in `real_by_fi`, find the nearest
    training trajectory in normalized-condition space and pull
    `n_per_traj` saturated-phase frames from it.

    Output is in **physical** space (frames are pulled from
    `ref_runner.trainset` and denormalized via that trainset's stats).

    Training entries whose normalized-condition distance to the eval
    trajectory is below `min_dist` are skipped — this avoids returning the
    eval trajectory itself if a matching parameter-scan point happens to
    sit in the training set (`dist=0` case)."""
    trainset = ref_runner.trainset
    valset = ref_runner.valsets[0]
    cond_keys = sorted(ref_runner.cfg.model.conditioning)

    n_files = len(trainset.metadata)
    train_conds = np.array([
        [float(np.squeeze(trainset.metadata[f_id][COND_META_MAP.get(k, k)])) for k in cond_keys]
        for f_id in range(n_files)
    ])
    train_conds_std = train_conds.std(axis=0, keepdims=True) + 1e-8
    train_conds_norm = train_conds / train_conds_std

    flat_by_file = {}
    for flat, (f_id, t_id) in trainset.flat_index_to_file_and_tstep.items():
        flat_by_file.setdefault(f_id, []).append((flat, t_id))

    rng = np.random.default_rng(seed)
    out = {}
    for fi, entry in real_by_fi.items():
        meta = valset.metadata[fi]
        eval_cond = np.array([
            float(np.squeeze(meta[COND_META_MAP.get(k, k)])) for k in cond_keys
        ]).reshape(1, -1)
        eval_cond_norm = eval_cond / train_conds_std
        dists = np.linalg.norm(train_conds_norm - eval_cond_norm, axis=1)
        # Pick the smallest distance that's strictly above min_dist (skips
        # exact-match training entries; the smallest *valid* index wins).
        order = np.argsort(dists)
        nn_train_fi = next((int(j) for j in order if dists[int(j)] > min_dist), None)
        if nn_train_fi is None:
            print(f"  [nn] no training trajectory with dist > {min_dist} for "
                  f"eval fi={fi} ({entry['label']}); skipping")
            continue
        n_skipped = int(np.sum(dists[order[:order.tolist().index(nn_train_fi)]] <= min_dist))

        offset = trainset.offsets[nn_train_fi]
        sat_t_min = max(0, saturated_phase_start - offset)
        sat_flat = [flat for flat, t in flat_by_file.get(nn_train_fi, [])
                    if t >= sat_t_min]
        if not sat_flat:
            print(f"  [nn] no saturated frames for train file {nn_train_fi}; "
                  f"skipping eval traj fi={fi} ({entry['label']})")
            continue
        n = min(n_per_traj, len(sat_flat))
        chosen = rng.choice(sat_flat, size=n, replace=False)
        dfs = []
        for idx in chosen:
            sample = trainset.__getitem__(int(idx), get_normalized=True, override_latens=True)
            dfs.append(_denormalize_via_dataset(trainset, nn_train_fi, sample.df.cpu()))
        if verbose:
            extra = f" (skipped {n_skipped} dist<={min_dist} match{'es' if n_skipped != 1 else ''})" if n_skipped else ""
            print(f"  [nn] eval fi={fi} -> train fi={nn_train_fi} "
                  f"(dist={dists[nn_train_fi]:.4f}, n_frames={n}){extra}")
        out[fi] = {"df": dfs, "label": entry["label"]}
    return out


# ---------------------------------------------------------------------------
# Split-half FID (appendix tab:fid_splithalf)
# ---------------------------------------------------------------------------
def split_half_fid(real_feats, gen_full_feats, *, k, n_components=64, seed=42):
    """Compute split-half FID on a single trajectory.

    `real_feats`     : (N_ref, D) real-sample features in some latent space.
    `gen_full_feats` : (>=2K, D) generated-sample features. The first 2K
                       rows are randomly partitioned into halves H_A, H_B.

    All three FIDs (full, A, B) share a single PCA basis fit on the pooled
    (real + full-gen) features so the values are directly comparable. PCA
    is dropped if `n_components is None`.

    Returns a dict with `fid_full`, `fid_a`, `fid_b`, `delta_abs` (|A-B|),
    `delta_rel` (|A-B|/full), and bookkeeping fields.
    """
    real_feats = np.asarray(real_feats)
    gen_full_feats = np.asarray(gen_full_feats)
    n_full = gen_full_feats.shape[0]
    if n_full < 2 * k:
        raise ValueError(
            f"split_half_fid: need >=2K={2*k} gen samples, got {n_full}"
        )
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_full)
    a_idx = perm[:k]
    b_idx = perm[k:2 * k]

    if n_components is None:
        proj_real = real_feats
        proj_full = gen_full_feats
        pca_dim = real_feats.shape[1]
        var_explained = 1.0
    else:
        from sklearn.decomposition import PCA
        pooled = np.concatenate([real_feats, gen_full_feats], axis=0)
        n_comp = min(n_components, *pooled.shape)
        pca = PCA(n_components=n_comp).fit(pooled)
        proj_real = pca.transform(real_feats)
        proj_full = pca.transform(gen_full_feats)
        pca_dim = pca.n_components_
        var_explained = float(pca.explained_variance_ratio_.sum())

    proj_a = proj_full[a_idx]
    proj_b = proj_full[b_idx]

    mu_r, sig_r       = compute_statistics(proj_real)
    mu_full, sig_full = compute_statistics(proj_full)
    mu_a, sig_a       = compute_statistics(proj_a)
    mu_b, sig_b       = compute_statistics(proj_b)

    fid_full = compute_fid(mu_r, sig_r, mu_full, sig_full)
    fid_a    = compute_fid(mu_r, sig_r, mu_a,    sig_a)
    fid_b    = compute_fid(mu_r, sig_r, mu_b,    sig_b)
    delta = abs(fid_a - fid_b)
    return {
        "fid_full":      fid_full,
        "fid_a":         fid_a,
        "fid_b":         fid_b,
        "delta_abs":     delta,
        "delta_rel":     delta / fid_full if fid_full > 0 else float("nan"),
        "n_ref":         int(real_feats.shape[0]),
        "n_full":        int(n_full),
        "k":             int(k),
        "pca_dim":       int(pca_dim),
        "var_explained": var_explained,
    }
