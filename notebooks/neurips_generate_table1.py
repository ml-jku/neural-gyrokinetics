"""Unified ID/OOD generative-eval helpers.

One source of truth for the avg_flux_rmse table across all models:

  * VAE / VQ-VAE / VQ-VAE+AR  (PINC-style, via neugk.pinc.generate)
  * Diff (flow-matching)      (5D + optional linear probes)
  * GyroSwin                  (AR rollout, optional)

Each `evaluate_*` function below builds its model, runs over ID + OOD
trajectories, caches results to disk, and tears the model down so the next
chapter starts fresh on the GPU. All metrics flow through
`notebooks.neurips_diff_eval.print_aggregate_metrics` -> rows are 1-to-1
comparable.
"""

import gc
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import omegaconf
from torch.utils._pytree import tree_map
from tqdm import tqdm

from neugk.dataset import get_data
from neugk.diffusion import get_diffusion_runner
from neugk.diffusion.models import get_diffusion_model
from neugk.integrals import FluxIntegral
from neugk.pinc.autoencoders.ae_utils import load_autoencoder
from neugk.pinc.autoencoders.gk_autoencoders import Swin5DVAE, Swin5DVQVAE
from neugk.pinc.generate import (
    compute_codebook_prior,
    compute_physics,
    denormalize,
    evaluate_generative,
    evaluate_ground_truth,
    get_conditioning,
    get_geometry,
    load_inference_config,
)
from neugk.utils import expand_as, recombine_zf, separate_zf

from notebooks.neurips_diff_eval import (
    fit_probes,
    generate_latents,
    print_aggregate_metrics,
)


# ---------------------------------------------------------------------------
# Default trajectory lists. Same as pinc_generate / gyroswin_generate /
# diff_generate_table so the rows align across notebooks.
# ---------------------------------------------------------------------------
TRAJECTORIES_ID = [
    "iteration_262_ifft_realpotens",
    "iteration_135_ifft_realpotens",
    "iteration_8_ifft_realpotens",
    "iteration_232_ifft_realpotens",
    "iteration_148_ifft_realpotens",
    "iteration_115_ifft_realpotens",
]
TRAJECTORIES_OOD = [
    "ood_iteration_0_ifft_realpotens",
    "ood_iteration_1_ifft_realpotens",
    "ood_iteration_2_ifft_realpotens",
    "ood_iteration_3_ifft_realpotens",
    "ood_iteration_4_ifft_realpotens",
]

COND_META_MAP = {"itg": "ion_temp_grad", "dg": "density_grad"}


# ---------------------------------------------------------------------------
# Generic utilities
# ---------------------------------------------------------------------------
def free_cuda():
    """Aggressive GPU cleanup. Multiple gc passes are needed because PyTorch
    modules often hold cyclic references that the cycle collector clears
    only after a couple of iterations."""
    for _ in range(3):
        gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _gpu_mem_str():
    if not torch.cuda.is_available():
        return "(no cuda)"
    a = torch.cuda.memory_allocated() / 1024**3
    r = torch.cuda.memory_reserved()  / 1024**3
    return f"alloc={a:.2f}G reserved={r:.2f}G"


def _release_module(mod):
    """Move a torch.nn.Module's parameters & buffers to CPU in-place so the
    GPU memory becomes immediately reclaimable, then delete the reference."""
    if mod is None:
        return
    try:
        mod.cpu()
    except Exception:
        pass


def _cached_or_run(output_path, run_fn):
    if output_path and os.path.exists(output_path):
        print(f"  [cache hit ] {output_path}")
        return torch.load(output_path, weights_only=False)
    result = run_fn()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(result, output_path)
        print(f"  [cache save] {output_path}")
    return result


def _add_meta_targets(gt_res, meta):
    """Add meta-side fluxspec/kyspec targets so probe rows can use them."""
    for spec_key in ("kyspec", "fluxspec"):
        if spec_key in meta:
            arr = np.asarray(meta[spec_key])
            if arr.ndim >= 1 and arr.shape[0] > 1:
                arr = arr[-80:].mean(0)
            gt_res[f"meta_{spec_key}_mean"] = torch.as_tensor(
                arr, dtype=torch.float32
            )
    return gt_res


def _add_full_alias(gt_res):
    """Add gt[k]['full'] (== ['all']) so R^2 SS_tot has a long series."""
    for k, v in gt_res.items():
        if isinstance(v, dict) and "all" in v and "full" not in v:
            v["full"] = v["all"]
    return gt_res


def _notebook_safe(cfg):
    cfg.ddp.enable = False
    if "deepspeed" in cfg:
        cfg.deepspeed.enable = False
    cfg.training.num_workers = 0
    cfg.training.pin_memory = False
    if "logging" in cfg:
        cfg.logging.writer = None
    return cfg


# ===========================================================================
# PINC family: VAE / VQ-VAE / VQ-VAE+AR
# ===========================================================================
def _build_pinc_model_and_stats(ckpt_dir, device):
    train_cfg = _notebook_safe(omegaconf.OmegaConf.load(os.path.join(ckpt_dir, "config.yaml")))
    print(f"  building trainset ({ckpt_dir}) ...")
    datasets, _, _ = get_data(train_cfg, rank=0)
    trainset = datasets[0]
    norm_stats = {
        "df_mean": np.asarray(trainset.stats["df"]["full"]["mean"]),
        "df_std":  np.asarray(trainset.stats["df"]["full"]["std"]),
    }
    # Drop the trainset (and the AE it may carry as `trainset.autoencoder` for
    # diffusion configs) before loading our own AE -- otherwise we double up.
    del datasets, trainset
    gc.collect()
    print(f"  loading AE weights ...")
    model, _, _ = load_autoencoder(ckpt_dir, device)
    model = model.to(device).eval()
    return model, train_cfg, norm_stats


def _build_ar_model(ar_ckpt_dir, device):
    ar_cfg = _notebook_safe(omegaconf.OmegaConf.load(os.path.join(ar_ckpt_dir, "config.yaml")))
    ae_ckpt = ar_cfg.ae_checkpoint
    print(f"  building trainset (AR config: ae_checkpoint={ae_ckpt}) ...")
    datasets, _, _ = get_data(ar_cfg, rank=0)
    trainset = datasets[0]
    norm_stats = {
        "df_mean": np.asarray(trainset.stats["df"]["full"]["mean"]),
        "df_std":  np.asarray(trainset.stats["df"]["full"]["std"]),
    }
    print(f"  loading AE (VQ-VAE) ...")
    ae_model, _, _ = load_autoencoder(ae_ckpt, device)
    ae_model = ae_model.to(device).eval()
    assert isinstance(ae_model, Swin5DVQVAE), \
        f"AR requires a VQ-VAE AE, got {type(ae_model).__name__}"
    print(f"  building AR transformer ...")
    ar_model = get_diffusion_model(ar_cfg, ae_model, trainset)
    ar_model = ar_model.to(device).eval()
    state = torch.load(os.path.join(ar_ckpt_dir, "best.pth"),
                       map_location=device, weights_only=True)
    ar_model.load_state_dict(state["model_state_dict"])
    # Drop trainset (it carries its own AE on diffusion configs -> double-load
    # of the VQ-VAE on GPU otherwise) and the loaded state dict.
    del datasets, trainset, state
    gc.collect()
    return ar_model, ae_model, ar_cfg, norm_stats


def _run_pinc_trajectories(model, ckpt_dir, train_cfg, norm_stats, trajectories,
                           inference_cfg, n_samples, batch_size, device,
                           integrator, vq_prior=None):
    cli = SimpleNamespace(trajectories=trajectories, n_samples=n_samples,
                          batch_size=batch_size, device=str(device))
    inf_cfg = load_inference_config(inference_cfg, cli)
    print(f"  n_samples={n_samples} bs={batch_size} ({len(inf_cfg['trajectories'])} trajs)")
    results = {}
    for traj in inf_cfg["trajectories"]:
        meta_path = os.path.join(inf_cfg["root"], traj, "metadata.pkl")
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        gen = evaluate_generative(model, ckpt_dir, train_cfg, inf_cfg, meta,
                                  norm_stats, device, integrator, vq_prior=vq_prior)
        gt = evaluate_ground_truth(meta, traj, train_cfg, inf_cfg, device, integrator)
        gt = _add_full_alias(_add_meta_targets(gt, meta))
        results[traj] = {"gen": gen, "gt": gt}
        # Per-traj cleanup keeps GPU cache from fragmenting across the loop.
        free_cuda()
    return results


def _ar_sample_decode(ar_model, ae_model, ar_cfg, condition, batch_size, device):
    ar_sub = ar_cfg.model.get("ar", {})
    temperature = ar_sub.get("temperature", 1.0)
    top_k = ar_sub.get("top_k", None)
    cond = (condition.unsqueeze(0).expand(batch_size, -1).to(device)
            if condition is not None else None)
    indices = ar_model.generate(condition=cond, temperature=temperature,
                                top_k=top_k, device=device)
    return ae_model.decode_from_indices(indices, condition=cond)["df"]


def _run_ar_trajectories(ar_model, ae_model, ar_cfg, norm_stats, trajectories,
                         inference_cfg, n_samples, batch_size, device, integrator):
    cli = SimpleNamespace(trajectories=trajectories, n_samples=n_samples,
                          batch_size=batch_size, device=str(device))
    inf_cfg = load_inference_config(inference_cfg, cli)
    sep_zf = ar_cfg.dataset.separate_zf
    cond_keys = list(ar_cfg.model.conditioning)
    print(f"  n_samples={n_samples} bs={batch_size} ({len(inf_cfg['trajectories'])} trajs)")
    results = {}
    for traj in inf_cfg["trajectories"]:
        meta_path = os.path.join(inf_cfg["root"], traj, "metadata.pkl")
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        condition = get_conditioning(cond_keys, meta, device) if cond_keys else None
        geometry = get_geometry(meta)

        accum = defaultdict(list)
        remaining = n_samples
        while remaining > 0:
            bs = min(batch_size, remaining)
            gen_df = _ar_sample_decode(ar_model, ae_model, ar_cfg, condition, bs, device).float()
            denorm = torch.stack([
                denormalize(df=gen_df[b], norm_stats=norm_stats) for b in range(bs)
            ])
            geom_b = tree_map(lambda g: g.unsqueeze(0).expand(bs, *g.shape), geometry)
            phys = compute_physics(denorm.cpu(), geom_b, sep_zf, integrator)
            for k, v in phys.items():
                accum[k].append(v.cpu())
            remaining -= bs

        gen_res = {}
        for k, tensors in accum.items():
            stacked = torch.cat(tensors, dim=0)
            gen_res[k] = {"all": stacked,
                          "mean": stacked.mean(dim=0),
                          "std":  stacked.std(dim=0)}

        gt_res = evaluate_ground_truth(meta, traj, ar_cfg, inf_cfg, device, integrator)
        gt_res = _add_full_alias(_add_meta_targets(gt_res, meta))
        results[traj] = {"gen": gen_res, "gt": gt_res}
        free_cuda()  # keep GPU cache tight across trajectories
    return results


def evaluate_vae(ckpt_dir, inference_cfg, *,
                 trajectories_id=TRAJECTORIES_ID,
                 trajectories_ood=TRAJECTORIES_OOD,
                 n_samples=128, batch_size=64,
                 output_path=None,
                 device=torch.device("cuda")):
    """Evaluate a Swin5DVAE on ID + OOD. Returns {vae_id, vae_ood}."""
    print("\n" + "=" * 60 + "\n  VAE\n" + "=" * 60)
    print(f"  GPU before: {_gpu_mem_str()}")
    def _run():
        integrator = FluxIntegral(flux_fields=True, spectral_df=False, spectral_potens=True)
        model, train_cfg, norm_stats = _build_pinc_model_and_stats(ckpt_dir, device)
        assert isinstance(model, Swin5DVAE), \
            f"Expected VAE at {ckpt_dir}, got {type(model).__name__}"
        try:
            out = {
                "vae_id":  _run_pinc_trajectories(model, ckpt_dir, train_cfg, norm_stats,
                                                  trajectories_id, inference_cfg, n_samples,
                                                  batch_size, device, integrator),
                "vae_ood": _run_pinc_trajectories(model, ckpt_dir, train_cfg, norm_stats,
                                                  trajectories_ood, inference_cfg, n_samples,
                                                  batch_size, device, integrator),
            }
        finally:
            _release_module(model)
            del model, train_cfg, norm_stats
            free_cuda()
        return out
    out = _cached_or_run(output_path, _run)
    print(f"  GPU after:  {_gpu_mem_str()}")
    return out


def evaluate_vqvae(ckpt_dir, inference_cfg, *,
                   vq_index_pkl=None,
                   trajectories_id=TRAJECTORIES_ID,
                   trajectories_ood=TRAJECTORIES_OOD,
                   n_samples=128, batch_size=64,
                   output_path=None,
                   device=torch.device("cuda")):
    """Evaluate a Swin5DVQVAE on ID + OOD. Returns {vqvae_id, vqvae_ood}."""
    print("\n" + "=" * 60 + "\n  VQ-VAE\n" + "=" * 60)
    print(f"  GPU before: {_gpu_mem_str()}")
    def _run():
        integrator = FluxIntegral(flux_fields=True, spectral_df=False, spectral_potens=True)
        model, train_cfg, norm_stats = _build_pinc_model_and_stats(ckpt_dir, device)
        assert isinstance(model, Swin5DVQVAE), \
            f"Expected VQ-VAE at {ckpt_dir}, got {type(model).__name__}"
        vq_prior = None
        if vq_index_pkl:
            vq_prior = compute_codebook_prior(vq_index_pkl, model.vq.codebook_size)
            nz = int((vq_prior > 0).sum().item())
            print(f"  empirical codebook prior: {nz}/{model.vq.codebook_size} codes used")
        try:
            out = {
                "vqvae_id":  _run_pinc_trajectories(model, ckpt_dir, train_cfg, norm_stats,
                                                    trajectories_id, inference_cfg, n_samples,
                                                    batch_size, device, integrator,
                                                    vq_prior=vq_prior),
                "vqvae_ood": _run_pinc_trajectories(model, ckpt_dir, train_cfg, norm_stats,
                                                    trajectories_ood, inference_cfg, n_samples,
                                                    batch_size, device, integrator,
                                                    vq_prior=vq_prior),
            }
        finally:
            _release_module(model)
            del model, train_cfg, norm_stats, vq_prior
            free_cuda()
        return out
    out = _cached_or_run(output_path, _run)
    print(f"  GPU after:  {_gpu_mem_str()}")
    return out


def evaluate_ar(ar_ckpt_dir, inference_cfg, *,
                trajectories_id=TRAJECTORIES_ID,
                trajectories_ood=TRAJECTORIES_OOD,
                n_samples=128, batch_size=64,
                output_path=None,
                device=torch.device("cuda")):
    """Evaluate AR-transformer + VQ-VAE decoder on ID + OOD.
    Returns {ar_id, ar_ood}."""
    print("\n" + "=" * 60 + "\n  AR (over VQ-VAE codes)\n" + "=" * 60)
    print(f"  GPU before: {_gpu_mem_str()}")
    def _run():
        integrator = FluxIntegral(flux_fields=True, spectral_df=False, spectral_potens=True)
        ar_model, ae_model, ar_cfg, norm_stats = _build_ar_model(ar_ckpt_dir, device)
        try:
            out = {
                "ar_id":  _run_ar_trajectories(ar_model, ae_model, ar_cfg, norm_stats,
                                               trajectories_id, inference_cfg,
                                               n_samples, batch_size, device, integrator),
                "ar_ood": _run_ar_trajectories(ar_model, ae_model, ar_cfg, norm_stats,
                                               trajectories_ood, inference_cfg,
                                               n_samples, batch_size, device, integrator),
            }
        finally:
            _release_module(ar_model)
            _release_module(ae_model)
            del ar_model, ae_model, ar_cfg, norm_stats
            free_cuda()
        return out
    out = _cached_or_run(output_path, _run)
    print(f"  GPU after:  {_gpu_mem_str()}")
    return out


# ===========================================================================
# Diff (flow-matching) family
# ===========================================================================
def _build_diff_runner(diff_ckpt_dir, ae_checkpoint, data_path,
                       valid_traj_h5_names, model_snapshot, device):
    pcfg = omegaconf.OmegaConf.load(os.path.join(diff_ckpt_dir, "config.yaml"))
    pcfg.output_path = diff_ckpt_dir
    pcfg.dataset.path = str(data_path)
    pcfg.ae_checkpoint = os.path.dirname(ae_checkpoint)
    pcfg.dataset.gds_override = True
    pcfg.dataset.validation_trajectories = valid_traj_h5_names
    pcfg.dataset.val_subsample = 1
    pcfg.dataset.eval_cond_filters = {}
    pcfg.validation.probe = {"targets": []}
    pcfg.logging.writer = None
    pcfg.logging.tqdm = True
    pcfg.training.num_workers = 0
    pcfg.training.pin_memory = False
    pcfg.ddp.enable = False
    pcfg.deepspeed.enable = False

    runner = get_diffusion_runner(rank=0, cfg=pcfg, world_size=1)
    torch.use_deterministic_algorithms(False)
    ckpt = torch.load(os.path.join(diff_ckpt_dir, model_snapshot),
                      map_location=device, weights_only=False)
    runner.model.load_state_dict(ckpt["model_state_dict"])
    runner.model.eval()
    print(f"  diff ckpt epoch={ckpt.get('epoch', '?')}, "
          f"params={sum(p.numel() for p in runner.model.parameters())/1e6:.1f}M")
    return runner


def _traj_basename(path):
    b = os.path.basename(path)
    for suf in ("_ifft_realpotens", "_ifft"):
        if b.endswith(suf):
            b = b[: -len(suf)]
    if b.endswith(".h5"):
        b = b[:-3]
    return b


@torch.no_grad()
def _diff_evaluate_trajectory(runner, fi, n_samples, gen_batch_size,
                              n_denoising_steps, integrator, device):
    """Sample latents, decode -> df, run physics on the decoded df. Stores raw
    latents under gen['_latents'] for chapter-B probe application.
    """
    valset = runner.valsets[0]
    sep_zf = runner.cfg.dataset.separate_zf
    cond_keys = sorted(runner.cfg.model.conditioning)

    fpath = valset.files[fi]
    traj  = _traj_basename(fpath)
    meta  = valset.metadata[fi]
    cond_vals = [float(np.squeeze(meta[COND_META_MAP.get(k, k)])) for k in cond_keys]
    cond = torch.tensor(cond_vals, dtype=torch.float32).unsqueeze(0)

    accum = defaultdict(list)
    all_latents = []
    remaining = n_samples
    while remaining > 0:
        bs = min(gen_batch_size, remaining)
        c = cond.expand(bs, -1).to(device)
        z = runner.sample(c, steps=n_denoising_steps, latent_only=True)
        all_latents.append(z.detach().cpu().numpy().reshape(z.shape[0], -1))

        decoded = runner.autoencoder.decode(z, condition=c)
        pred_df = decoded["df"].cpu()
        for b in range(bs):
            pred_df[b] = valset.denormalize(fi, df=pred_df[b])
        if sep_zf and pred_df.shape[1] > 2:
            pred_df = recombine_zf(pred_df, dim=1)

        geom = tree_map(lambda g: torch.as_tensor(g, dtype=torch.float64),
                        meta["geometry"])
        geom_b = tree_map(lambda g: g.unsqueeze(0).expand(bs, *g.shape), geom)
        phys = compute_physics(pred_df.float().cpu(), geom_b, separate_zf=False,
                               integrator=integrator)
        for k, v in phys.items():
            accum[k].append(v.cpu())
        remaining -= bs

    gen_res = {
        k: {"all": torch.cat(v), "mean": torch.cat(v).mean(0), "std": torch.cat(v).std(0)}
        for k, v in accum.items()
    }
    gen_res["_latents"] = np.concatenate(all_latents, axis=0)

    # GT physics + meta-side spectra targets.
    gt_subdir = f"{traj}_ifft_realpotens"
    gt_res = evaluate_ground_truth(meta, gt_subdir, runner.cfg,
                                   {"root": str(runner.cfg.dataset.path)},
                                   device, integrator)
    gt_res = _add_full_alias(_add_meta_targets(gt_res, meta))
    return traj, {"gen": gen_res, "gt": gt_res}


def _fit_diff_probes(runner, n_components, alpha, subsample, seed,
                     gen_batch_size, n_denoising_steps):
    """Fit linear probes on training latents (flux + cond + spectra)."""
    from sklearn.linear_model import Ridge

    print("  fitting probes ...")
    cond_keys = sorted(runner.cfg.model.conditioning)
    all_keys = list(runner.trainset.precomputed_latents.keys())
    X_ae_full = np.stack([
        np.array(runner.trainset.precomputed_latents[k]["x"]).reshape(-1)
        for k in all_keys
    ])
    y_flux_full = np.array([
        float(np.squeeze(runner.trainset.precomputed_latents[k]["flux"]))
        for k in all_keys
    ])
    C_full = np.stack([
        np.array([
            float(np.squeeze(runner.trainset.precomputed_latents[k][ck]))
            for ck in cond_keys
        ])
        for k in all_keys
    ])

    if subsample and subsample < len(X_ae_full):
        sub_idx = np.random.RandomState(seed).choice(
            len(X_ae_full), subsample, replace=False,
        )
        X_ae    = X_ae_full[sub_idx]
        y_flux  = y_flux_full[sub_idx]
        C_train = C_full[sub_idx]
        sub_keys = [all_keys[i] for i in sub_idx]
    else:
        X_ae, y_flux, C_train = X_ae_full, y_flux_full, C_full
        sub_keys = all_keys

    X_gen = generate_latents(runner, C_train, batch_size=gen_batch_size,
                             steps=n_denoising_steps)
    probes = fit_probes(X_ae, X_gen, y_flux, cond_keys, C_train,
                        n_components=n_components, alpha=alpha)
    pca = probes["pca"]

    # Spectra probes (kyspec/fluxspec from meta, kxspec from integrator)
    SPEC_META = ["kyspec", "fluxspec"]
    SPEC_INTEG = ["kxspec"]
    integrator_sp = FluxIntegral(flux_fields=True, spectral_df=False, spectral_potens=True)
    sep_zf = runner.cfg.dataset.separate_zf
    trainset = runner.trainset
    lookup = trainset.file_and_tstep_to_flat_index

    def _stats(key):
        s = trainset.stats.get(key, {}).get("full", {})
        m = np.asarray(s.get("mean", 0.0), dtype=np.float32)
        d = np.asarray(s.get("std", 1.0), dtype=np.float32)
        return m, np.where(d > 0, d, 1.0)

    meta_stats = {k: _stats(k) for k in SPEC_META}
    raw_targets = {k: [] for k in SPEC_META + SPEC_INTEG}
    valid_rows = []
    for row, (fi, ti) in enumerate(tqdm(sub_keys, desc="  spec targets")):
        if (fi, ti) not in lookup:
            continue
        meta_t = trainset.metadata[fi]
        if any(k not in meta_t for k in SPEC_META):
            continue
        vals_meta = {}
        for k in SPEC_META:
            v = np.asarray(meta_t[k])
            if v.ndim >= 1 and v.shape[0] > 1:
                v = v[ti]
            vals_meta[k] = v
        flat_idx = lookup[(fi, ti)]
        sample = trainset.__getitem__(flat_idx, override_latens=True)
        scale, shift = trainset._get_scale_shift(fi, "df", sample.df)
        df_t = (sample.df * scale + shift).unsqueeze(0)
        if sep_zf and df_t.shape[1] > 2:
            df_t = recombine_zf(df_t, dim=1)
        geom_t = trainset.get_batch_geometry(torch.tensor([fi]))
        geom_t = tree_map(lambda g: torch.as_tensor(g, dtype=torch.float64), geom_t)
        phys = compute_physics(df_t.float().cpu(), geom_t, separate_zf=False,
                               integrator=integrator_sp)
        kx_val = phys["kxspec"].squeeze(0).cpu().numpy().reshape(-1)
        for k, v in vals_meta.items():
            raw_targets[k].append(v)
        raw_targets["kxspec"].append(kx_val)
        valid_rows.append(row)

    all_specs = {}
    for k in SPEC_META:
        raw_log = np.log1p(np.stack(raw_targets[k]))
        m, d = meta_stats[k]
        y = ((raw_log - m) / d).astype(np.float32)
        if y.ndim > 2:
            y = y.reshape(y.shape[0], -1)
        all_specs[k] = {"y": y, "mean": m, "std": d}
    for k in SPEC_INTEG:
        raw_log = np.log1p(np.stack(raw_targets[k]))
        m = raw_log.mean(axis=0).astype(np.float32)
        d = raw_log.std(axis=0).astype(np.float32)
        d = np.where(d > 0, d, 1.0)
        all_specs[k] = {"y": ((raw_log - m) / d).astype(np.float32), "mean": m, "std": d}

    valid_rows = np.asarray(valid_rows)
    X_ae_pca  = pca.transform(X_ae[valid_rows])
    X_gen_pca = pca.transform(X_gen[valid_rows])

    spec_probes = {}
    for k, spec in all_specs.items():
        y = spec["y"]
        pae  = Ridge(alpha=alpha).fit(X_ae_pca, y)
        pgen = Ridge(alpha=alpha).fit(X_gen_pca, y)
        spec_probes[k] = {"probe_ae": pae, "probe_gen": pgen,
                          "mean": spec["mean"], "std": spec["std"]}

    return {"probes": probes, "spec_probes": spec_probes, "pca": pca}


def _apply_diff_probes(group, probe_pack):
    """Add probe_<key>_<ae|gen> entries to gen[] for each trajectory."""
    probes = probe_pack["probes"]
    spec_probes = probe_pack["spec_probes"]
    pca = probe_pack["pca"]
    for traj, res in group.items():
        gen = res["gen"]
        if "_latents" not in gen:
            continue
        X = gen["_latents"]
        X_pca = pca.transform(X)
        gen["probe_flux_ae"]  = probes["flux"]["probe_ae"].predict(X_pca)
        gen["probe_flux_gen"] = probes["flux"]["probe_gen"].predict(X_pca)
        for sk, sp in spec_probes.items():
            for v in ("ae", "gen"):
                pred_norm = sp[f"probe_{v}"].predict(X_pca)
                pred_log  = pred_norm * sp["std"] + sp["mean"]
                gen[f"probe_{sk}_{v}"] = np.expm1(pred_log)
    return group


def evaluate_diff(diff_ckpt_dir, ae_checkpoint, *,
                  data_path,
                  trajectories_id=TRAJECTORIES_ID,
                  trajectories_ood=TRAJECTORIES_OOD,
                  n_samples=128, gen_batch_size=64, n_denoising_steps=20,
                  with_probes=True,
                  probe_n_components=64, probe_alpha=1.0,
                  probe_subsample=256, probe_seed=0,
                  model_snapshot="best.pth",
                  output_path=None,
                  device=torch.device("cuda")):
    """Evaluate a flow-matching diffusion model on ID + OOD.

    Returns {diff_id, diff_ood}. With probes=True the gen dicts also carry
    probe_flux_<ae|gen>, probe_kxspec_<ae|gen>, probe_kyspec_<ae|gen>,
    probe_fluxspec_<ae|gen> arrays of shape (N, ...) for the probe rows.
    """
    print("\n" + "=" * 60 + "\n  Diff (flow-matching)\n" + "=" * 60)
    print(f"  GPU before: {_gpu_mem_str()}")
    def _run():
        integrator = FluxIntegral(flux_fields=True, spectral_df=False, spectral_potens=True)
        valid_h5 = [
            *(t.replace("_ifft_realpotens", "") + ".h5" for t in trajectories_id),
            *(t.replace("_ifft_realpotens", "") + ".h5" for t in trajectories_ood),
        ]
        runner = _build_diff_runner(diff_ckpt_dir, ae_checkpoint, data_path,
                                    valid_h5, model_snapshot, device)
        try:
            valset = runner.valsets[0]
            id_bases  = {t.replace("_ifft_realpotens", "") for t in trajectories_id}
            ood_bases = {t.replace("_ifft_realpotens", "") for t in trajectories_ood}

            out = {"diff_id": {}, "diff_ood": {}}
            for fi, fpath in enumerate(valset.files):
                base = _traj_basename(fpath)
                if base in id_bases:
                    key = "diff_id"
                elif base in ood_bases:
                    key = "diff_ood"
                else:
                    continue
                print(f"  [{key}] {base}")
                _, res = _diff_evaluate_trajectory(
                    runner, fi, n_samples, gen_batch_size, n_denoising_steps,
                    integrator, device,
                )
                out[key][base] = res
                free_cuda()

            if with_probes:
                probe_pack = _fit_diff_probes(
                    runner, probe_n_components, probe_alpha, probe_subsample,
                    probe_seed, gen_batch_size, n_denoising_steps,
                )
                for k in ("diff_id", "diff_ood"):
                    _apply_diff_probes(out[k], probe_pack)
                del probe_pack
        finally:
            _release_module(getattr(runner, "model", None))
            _release_module(getattr(runner, "autoencoder", None))
            del runner
            free_cuda()
        return out
    out = _cached_or_run(output_path, _run)
    print(f"  GPU after:  {_gpu_mem_str()}")
    return out


# ===========================================================================
# GyroSwin (optional)
# ===========================================================================
def _build_gyroswin_runner(diff_dir, ae_checkpoint, data_path,
                           pinned_traj_h5, device):
    pcfg = omegaconf.OmegaConf.load(os.path.join(diff_dir, "config.yaml"))
    pcfg.output_path = diff_dir
    pcfg.dataset.path = str(data_path)
    pcfg.ae_checkpoint = os.path.dirname(ae_checkpoint)
    pcfg.dataset.gds_override = True
    pcfg.dataset.validation_trajectories = [pinned_traj_h5]
    pcfg.dataset.val_subsample = 1
    pcfg.dataset.eval_cond_filters = {}
    pcfg.validation.probe = {"targets": []}
    pcfg.logging.writer = None
    pcfg.logging.tqdm = True
    pcfg.training.num_workers = 0
    pcfg.training.pin_memory = False
    pcfg.ddp.enable = False
    pcfg.deepspeed.enable = False
    runner = get_diffusion_runner(rank=0, cfg=pcfg, world_size=1)
    # get_diffusion_runner re-enables deterministic algorithms internally;
    # CuBLAS in the GyroSwin forward pass crashes under that flag unless
    # CUBLAS_WORKSPACE_CONFIG is set, so disable it here (same as
    # _build_diff_runner does for the diff path).
    torch.use_deterministic_algorithms(False)
    return runner


def _gyroswin_load_df_bin(data_dir, idx, res, sep_zf):
    bin_path = data_dir / "data" / f"timestep_{idx:05d}.bin"
    arr = np.fromfile(bin_path, dtype=np.float32).reshape(2, *res)
    if sep_zf:
        arr = separate_zf(arr, dim=0)
    return torch.tensor(arr)


def _gyroswin_scale_shift(gs_stats, field, ref):
    s = gs_stats[field]["full"]
    mean = np.asarray(s["mean"], dtype=np.float32)
    std  = np.asarray(s["std"],  dtype=np.float32)
    shift = expand_as(torch.as_tensor(mean, dtype=ref.dtype, device=ref.device), ref)
    scale = expand_as(torch.as_tensor(std,  dtype=ref.dtype, device=ref.device), ref)
    return scale, shift


@torch.no_grad()
def _gyroswin_ar_rollout(gs_model, gs_stats, gs_cond_keys, traj, n_steps,
                         offset, fluxavg_mean, fluxavg_std,
                         data_prep, sep_zf, device, return_df=True):
    data_dir = data_prep / traj
    with open(data_dir / "metadata.pkl", "rb") as f:
        meta = pickle.load(f)
    t_grid = np.asarray(meta["timesteps"])
    n_act = min(n_steps, len(t_grid) - offset)
    res = meta["resolution"]

    df0 = _gyroswin_load_df_bin(data_dir, offset, res, sep_zf).unsqueeze(0).to(device)
    df_scale, df_shift = _gyroswin_scale_shift(gs_stats, "df", df0)
    params = {
        k: torch.tensor(np.asarray(meta[COND_META_MAP.get(k, k)]).reshape(1),
                        dtype=torch.float32, device=device)
        for k in gs_cond_keys if k != "timestep"
    }
    inputs = {"df": (df0 - df_shift) / df_scale}
    flux_pred = np.empty(n_act, dtype=np.float32)
    df_snaps = [] if return_df else None
    for step in range(n_act):
        params["timestep"] = torch.tensor(
            [float(t_grid[offset + step])], dtype=torch.float32, device=device,
        )
        out = gs_model(**inputs, **params)
        inputs["df"] = out["df"].clone()
        flux_norm = float(out["flux"].squeeze().detach().cpu())
        flux_pred[step] = flux_norm * fluxavg_std + fluxavg_mean
        if return_df:
            df_snaps.append((out["df"] * df_scale + df_shift).squeeze(0).cpu())
    return dict(meta=meta, n_steps=n_act, offset=offset,
                t_grid=t_grid[offset:offset + n_act].astype(np.float32),
                flux_pred=flux_pred,
                df_pred=torch.stack(df_snaps) if return_df else None)


def _gyroswin_compute_physics_batched(df_snaps, meta, sep_zf):
    integrator = FluxIntegral(flux_fields=True, spectral_df=False, spectral_potens=True)
    geometry = get_geometry(meta)
    bs = df_snaps.shape[0]
    geom_b = tree_map(lambda g: g.unsqueeze(0).expand(bs, *g.shape), geometry)
    return compute_physics(df_snaps.cpu().float(), geom_b, sep_zf, integrator)


def _gyroswin_evaluate_trajectory(traj, n_steps, offset, fluxavg_mean, fluxavg_std,
                                   data_prep, sep_zf, gs_model, gs_stats, gs_cond_keys,
                                   device):
    roll = _gyroswin_ar_rollout(
        gs_model, gs_stats, gs_cond_keys, traj, n_steps, offset,
        fluxavg_mean, fluxavg_std, data_prep, sep_zf, device, return_df=True,
    )
    n = roll["n_steps"]
    meta = roll["meta"]

    gen_phys = _gyroswin_compute_physics_batched(roll["df_pred"], meta, sep_zf)
    gen_phys["eflux"] = torch.from_numpy(roll["flux_pred"]).float()

    full_n = len(meta["timesteps"]) - offset
    if full_n != n:
        gt_df_full = torch.stack([
            _gyroswin_load_df_bin(data_prep / traj, offset + k, meta["resolution"], sep_zf)
            for k in range(full_n)
        ]).float()
    else:
        gt_df_full = torch.stack([
            _gyroswin_load_df_bin(data_prep / traj, offset + k, meta["resolution"], sep_zf)
            for k in range(n)
        ]).float()
    gt_phys_full = _gyroswin_compute_physics_batched(gt_df_full, meta, sep_zf)
    gt_phys_full["eflux"] = gt_phys_full["eflux"].float()

    def _stats(t):
        return {"mean": t.mean(dim=0) if t.dim() > 0 else t,
                "std":  (t.std(dim=0) if t.dim() > 1
                         else (t.std() if t.dim() == 1 else torch.zeros_like(t))),
                "all":  t,
                "full": t}

    gen_res = {k: {"all": v if isinstance(v, torch.Tensor) else torch.as_tensor(v),
                   "mean": (v.mean(dim=0) if isinstance(v, torch.Tensor) and v.dim() > 0
                             else v),
                   "std":  (v.std(dim=0) if isinstance(v, torch.Tensor) and v.dim() > 1
                             else torch.zeros_like(v) if isinstance(v, torch.Tensor)
                             else torch.tensor(0.0))}
               for k, v in gen_phys.items()}
    gt_res = {k: _stats(v if isinstance(v, torch.Tensor) else torch.as_tensor(v))
              for k, v in gt_phys_full.items()}
    gt_res = _add_meta_targets(gt_res, meta)
    return {"gen": gen_res, "gt": gt_res, "t_grid": roll["t_grid"]}


def evaluate_gyroswin(gyroswin_checkpoint, diff_dir, ae_checkpoint, *,
                      data_prep,
                      trajectories_id=TRAJECTORIES_ID,
                      trajectories_ood=TRAJECTORIES_OOD,
                      n_steps=128,
                      fluxavg_mean=92.6521, fluxavg_std=45.026,
                      output_path=None,
                      device=torch.device("cuda")):
    """Evaluate GyroSwin AR-rollout on ID + OOD.
    Loads the checkpoint via the bundled `notebooks.neurips_gyroswin_eval` helper
    (same one gyroswin_generate.ipynb uses)."""
    print("\n" + "=" * 60 + "\n  GyroSwin (AR rollout)\n" + "=" * 60)
    print(f"  GPU before: {_gpu_mem_str()}")
    def _run():
        # Local imports so non-gyroswin runs don't need this on the path.
        sys.path.insert(0, str(Path(__file__).parent))
        from neurips_gyroswin_eval import load_gyroswin_model

        pinned = trajectories_id[0].replace("_ifft_realpotens", "") + ".h5"
        runner = _build_gyroswin_runner(diff_dir, ae_checkpoint, data_prep,
                                        pinned, device)
        gs_model, gs_cfg, _ = load_gyroswin_model(
            gyroswin_checkpoint, dataset=runner.trainset, device=device,
        )
        with open(os.path.join(gyroswin_checkpoint, "normalization_stats.pkl"), "rb") as f:
            gs_stats = pickle.load(f)
        offset    = int(gs_cfg.dataset.get("offset", 80))
        sep_zf    = bool(gs_cfg.dataset.separate_zf)
        cond_keys = sorted(list(gs_cfg.model.conditioning))

        try:
            out = {"gyroswin_id": {}, "gyroswin_ood": {}}
            for label, trajs in (("gyroswin_id", trajectories_id),
                                 ("gyroswin_ood", trajectories_ood)):
                for traj in trajs:
                    print(f"  [{label}] {traj}")
                    out[label][traj] = _gyroswin_evaluate_trajectory(
                        traj, n_steps, offset, fluxavg_mean, fluxavg_std,
                        data_prep, sep_zf, gs_model, gs_stats, cond_keys, device,
                    )
                free_cuda()
        finally:
            _release_module(gs_model)
            _release_module(getattr(runner, "model", None))
            _release_module(getattr(runner, "autoencoder", None))
            del gs_model, runner, gs_stats
            free_cuda()
        return out
    out = _cached_or_run(output_path, _run)
    print(f"  GPU after:  {_gpu_mem_str()}")
    return out


# ===========================================================================
# Table builder
# ===========================================================================
_RENAME_TO_COL = {
    # 5D row keys (integrator-side targets except for fluxspec)
    "eflux_RMSE":             "eflux_RMSE",
    "kxspec_RMSE":            "kxspec_RMSE",
    "kyspec_RMSE":            "kyspec_RMSE",
    "qspec_RMSE":             "fluxspec_RMSE",
    # Probe row keys
    "probe_flux_ae_RMSE":     "eflux_RMSE",
    "probe_flux_gen_RMSE":    "eflux_RMSE",
    "probe_kxspec_ae_RMSE":   "kxspec_RMSE",
    "probe_kxspec_gen_RMSE":  "kxspec_RMSE",
    "probe_kyspec_ae_RMSE":   "kyspec_RMSE",
    "probe_kyspec_gen_RMSE":  "kyspec_RMSE",
    "probe_fluxspec_ae_RMSE":  "fluxspec_RMSE",
    "probe_fluxspec_gen_RMSE": "fluxspec_RMSE",
}


def _row_metrics_5d(group):
    """Standard 5D-row metrics: integrator-side eflux/kxspec/kyspec, plus
    qspec mapped to meta_fluxspec_mean as the fluxspec column. Same shape
    as diff_generate_table's `5D` row."""
    return print_aggregate_metrics(
        group,
        scalar_keys=["eflux"],
        spec_keys=["kxspec", "kyspec"],
        extra_keys=[("qspec", "meta_fluxspec_mean")],
    )


def _row_metrics_probe(group, variant):
    """Probe-row metrics. variant in {'ae', 'gen'}."""
    return print_aggregate_metrics(
        group,
        scalar_keys=[(f"probe_flux_{variant}", "eflux")],
        spec_keys=[(f"probe_kxspec_{variant}", "kxspec"),
                   (f"probe_kyspec_{variant}", "meta_kyspec_mean")],
        extra_keys=[(f"probe_fluxspec_{variant}", "meta_fluxspec_mean")],
    )


def _coerce_table_row(model_label, split_label, raw_metrics):
    return {
        "model": model_label,
        "split": split_label,
        **{_RENAME_TO_COL[k]: v
           for k, v in raw_metrics.items()
           if k in _RENAME_TO_COL},
    }


def build_summary_table(all_results, *, with_diff_probes=True):
    """Single DataFrame with rows for each present model & split.

    Recognised group keys in `all_results`:
      vae_id / vae_ood, vqvae_id / vqvae_ood, ar_id / ar_ood,
      diff_id / diff_ood (also produces probe rows when probe_* keys present),
      gyroswin_id / gyroswin_ood.

    Columns: eflux_RMSE, kxspec_RMSE, kyspec_RMSE, fluxspec_RMSE.
    """
    rows = []

    for model_label, gid, good in (
        ("VAE",      "vae_id",      "vae_ood"),
        ("VQ-VAE",   "vqvae_id",    "vqvae_ood"),
        ("AR",       "ar_id",       "ar_ood"),
        ("Diff 5D",  "diff_id",     "diff_ood"),
    ):
        for split, key in (("ID", gid), ("OOD", good)):
            g = all_results.get(key)
            if not g:
                continue
            rows.append(_coerce_table_row(model_label, split, _row_metrics_5d(g)))

    if with_diff_probes:
        for variant in ("ae", "gen"):
            for split, key in (("ID", "diff_id"), ("OOD", "diff_ood")):
                g = all_results.get(key)
                if not g:
                    continue
                # Skip if probe predictions are not present.
                probe_present = any(
                    f"probe_flux_{variant}" in res["gen"]
                    for res in g.values()
                )
                if not probe_present:
                    continue
                rows.append(_coerce_table_row(
                    f"Diff probe ({variant})", split,
                    _row_metrics_probe(g, variant),
                ))

    for split, key in (("ID", "gyroswin_id"), ("OOD", "gyroswin_ood")):
        g = all_results.get(key)
        if not g:
            continue
        rows.append(_coerce_table_row("GyroSwin", split, _row_metrics_5d(g)))

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index(["model", "split"])
    cols = ["eflux_RMSE", "kxspec_RMSE", "kyspec_RMSE", "fluxspec_RMSE"]
    return df[[c for c in cols if c in df.columns]]
