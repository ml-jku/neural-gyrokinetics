"""Unified ID/OOD generative-eval helpers.

avg_flux_rmse table across all models:

  * VAE / VQ-VAE / VQ-VAE+AR  (PINC-style, via neugk.pinc.generate)
  * Diff (flow-matching)      (5D + optional linear probes)
  * GyroSwin                  (AR rollout, optional)

Each `evaluate_*` function below builds its model, runs over ID + OOD
trajectories, caches results to disk, and tears the model down so the next
start is fresh on the GPU. All metrics flow through
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

TRAJECTORIES_TEST = [
    "iteration_13_ifft_realpotens",
    "iteration_100_ifft_realpotens",
    "iteration_200_ifft_realpotens",
]
TRAJECTORIES_ID = [
    "iteration_262_ifft_realpotens",
    "iteration_131_ifft_realpotens",
    "iteration_8_ifft_realpotens",
    "iteration_235_ifft_realpotens",
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

# Standard splits surfaced by the evaluators and tables. Tuples of
# (display_label, group_suffix, default_traj_list). Each evaluator iterates
# this list, runs trajectories per non-empty split, and stores results under
# `<model>_<group_suffix>` keys; build_summary_table / build_recon_table
# surface one row per split per model. Add a fourth tuple here to introduce
# yet another split with no further code changes.
SUMMARY_SPLITS = (
    ("ID",   "id",   TRAJECTORIES_ID),
    ("OOD",  "ood",  TRAJECTORIES_OOD),
    ("TEST", "test", TRAJECTORIES_TEST),
)


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


class _GpuTimer:
    """Context manager that times a block of GPU+CPU work in wall-clock seconds.

    `torch.cuda.synchronize()` brackets the block when CUDA is in use so the
    measurement actually covers kernel completion (kernels are otherwise
    asynchronous). On CPU the syncs are no-ops.
    """
    def __init__(self):
        self.elapsed = 0.0

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed = time.perf_counter() - self._t0
        return False


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
    if output_path and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        try:
            cached = torch.load(output_path, weights_only=False)
            print(f"  [cache hit ] {output_path}")
            return cached
        except (RuntimeError, EOFError, OSError, pickle.UnpicklingError) as e:
            print(f"  [cache CORRUPT] {output_path}  "
                  f"({type(e).__name__}: {e}); regenerating")
            try:
                os.replace(output_path, output_path + ".corrupt")
            except OSError:
                pass
    elif output_path and os.path.exists(output_path):
        print(f"  [cache empty] {output_path}; regenerating")
        try:
            os.replace(output_path, output_path + ".corrupt")
        except OSError:
            pass
    result = run_fn()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(result, output_path)
        print(f"  [cache save] {output_path}")
    return result


def _resolve_splits(trajectories_id, trajectories_ood, trajectories_test=None):
    """Return [(suffix, trajs), ...] for non-empty splits, in canonical
    SUMMARY_SPLITS order. Empty splits are omitted so cached results don't
    grow stub keys when a split isn't requested."""
    by_label = {"id": trajectories_id, "ood": trajectories_ood,
                "test": trajectories_test}
    out = []
    for _label, suffix, _default in SUMMARY_SPLITS:
        trajs = by_label.get(suffix)
        if trajs:
            out.append((suffix, list(trajs)))
    return out


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
def _build_pinc_model_and_stats(ckpt_dir, device, data_path=None):
    train_cfg = _notebook_safe(omegaconf.OmegaConf.load(os.path.join(ckpt_dir, "config.yaml")))
    if data_path is not None:
        train_cfg.dataset.path = str(data_path)
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


def _build_ar_model(ar_ckpt_dir, device, data_path=None, ae_checkpoint=None):
    ar_cfg = _notebook_safe(omegaconf.OmegaConf.load(os.path.join(ar_ckpt_dir, "config.yaml")))
    if data_path is not None:
        ar_cfg.dataset.path = str(data_path)
    # Override the AE checkpoint baked into the AR config (often a stale HPC
    # path) with whatever the caller actually has on disk.
    if ae_checkpoint is not None:
        ar_cfg.ae_checkpoint = str(ae_checkpoint)
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
                           integrator, vq_prior=None, data_path=None):
    cli = SimpleNamespace(trajectories=trajectories, n_samples=n_samples,
                          batch_size=batch_size, device=str(device))
    inf_cfg = load_inference_config(inference_cfg, cli)
    if data_path is not None:
        inf_cfg["root"] = str(data_path)
    print(f"  n_samples={n_samples} bs={batch_size} ({len(inf_cfg['trajectories'])} trajs)")
    results = {}
    for traj in inf_cfg["trajectories"]:
        meta_path = os.path.join(inf_cfg["root"], traj, "metadata.pkl")
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        timing = {}
        gen = evaluate_generative(model, ckpt_dir, train_cfg, inf_cfg, meta,
                                  norm_stats, device, integrator, vq_prior=vq_prior,
                                  timing_out=timing)
        gen["_gen_time_s"] = float(timing.get("gen_time_s", 0.0))
        gen["_n_samples"]  = int(timing.get("n_samples", n_samples))
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
                         inference_cfg, n_samples, batch_size, device, integrator,
                         data_path=None):
    cli = SimpleNamespace(trajectories=trajectories, n_samples=n_samples,
                          batch_size=batch_size, device=str(device))
    inf_cfg = load_inference_config(inference_cfg, cli)
    if data_path is not None:
        inf_cfg["root"] = str(data_path)
    sep_zf = ar_cfg.dataset.separate_zf
    cond_keys = sorted(ar_cfg.model.conditioning)
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
        gen_time_s = 0.0
        n_done = 0
        while remaining > 0:
            bs = min(batch_size, remaining)
            with _GpuTimer() as _t:
                gen_df = _ar_sample_decode(ar_model, ae_model, ar_cfg, condition, bs, device).float()
            gen_time_s += _t.elapsed
            n_done += bs
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
        gen_res["_gen_time_s"] = float(gen_time_s)
        gen_res["_n_samples"]  = int(n_done)

        gt_res = evaluate_ground_truth(meta, traj, ar_cfg, inf_cfg, device, integrator)
        gt_res = _add_full_alias(_add_meta_targets(gt_res, meta))
        results[traj] = {"gen": gen_res, "gt": gt_res}
        free_cuda()  # keep GPU cache tight across trajectories
    return results


def evaluate_vae(ckpt_dir, inference_cfg, *,
                 trajectories_id=TRAJECTORIES_ID,
                 trajectories_ood=TRAJECTORIES_OOD,
                 trajectories_test=TRAJECTORIES_TEST,
                 n_samples=128, batch_size=64,
                 output_path=None,
                 data_path=None,
                 device=torch.device("cuda")):
    """Evaluate a Swin5DVAE on ID + OOD (+ TEST if non-empty). Returns
    one entry per non-empty split keyed `vae_<id|ood|test>`."""
    print("\n" + "=" * 60 + "\n  VAE\n" + "=" * 60)
    print(f"  GPU before: {_gpu_mem_str()}")
    splits = _resolve_splits(trajectories_id, trajectories_ood, trajectories_test)
    def _run():
        integrator = FluxIntegral(flux_fields=True, spectral_df=False, spectral_potens=True)
        model, train_cfg, norm_stats = _build_pinc_model_and_stats(ckpt_dir, device, data_path)
        assert isinstance(model, Swin5DVAE), \
            f"Expected VAE at {ckpt_dir}, got {type(model).__name__}"
        try:
            out = {
                f"vae_{suffix}": _run_pinc_trajectories(
                    model, ckpt_dir, train_cfg, norm_stats,
                    trajs, inference_cfg, n_samples,
                    batch_size, device, integrator,
                    data_path=data_path,
                )
                for suffix, trajs in splits
            }
        finally:
            _release_module(model)
            del model, train_cfg, norm_stats
            free_cuda()
        return out
    out = _cached_or_run(output_path, _run)
    print(f"  GPU after:  {_gpu_mem_str()}")
    return out


# Quantities computed per batch in evaluate_reconstruction. "df" is the
# point-wise reconstruction error in physical space; the rest are squared
# errors on FluxIntegral outputs (eflux scalar + phi/q spectra). qspec is
# renamed to "fluxspec" in the table to match the summary table convention.
_RECON_QUANTITIES = ("df", "eflux", "kxspec", "kyspec", "qspec")
_RECON_QTY_TO_COL = {
    "df": "df_RMSE",
    "eflux": "eflux_RMSE",
    "kxspec": "kxspec_RMSE",
    "kyspec": "kyspec_RMSE",
    "qspec": "fluxspec_RMSE",
}


@torch.no_grad()
def evaluate_reconstruction(ckpt_dir, inference_cfg, *,
                            trajectories_id=TRAJECTORIES_ID,
                            trajectories_ood=TRAJECTORIES_OOD,
                            trajectories_test=TRAJECTORIES_TEST,
                            data_path=None,
                            batch_size=4,
                            device=torch.device("cuda")):
    """Per-trajectory reconstruction RMSE for an AE / VAE / VQ-VAE, on both
    the raw distribution function `df` and the physics integrals computed
    from it (energy flux scalar `eflux`, phi spectra `kxspec`/`kyspec`, and
    energy-flux spectrum `qspec`).

    Streams the post-offset df binaries of each requested split's trajectories
    through the model (encode + decode, eval mode), denormalizes both
    prediction and target with the trainset's df stats, and per batch:
      * accumulates point-wise squared error on df (physical space).
      * runs `FluxIntegral(...)` on both pred and target and accumulates
        squared error on each integrator output.

    Per-trajectory RMSE for each quantity = `sqrt(sum_sq / n_elems)`, in raw
    integrator units. Reuses `_build_pinc_model_and_stats` so model,
    normalization, and conditioning handling stay consistent with the
    generation chapters.

    Returns:
        {
            "n_params_M": float,
            "id_rmse":   {<quantity>: {<traj>: float}, ...},  # if non-empty
            "ood_rmse":  {<quantity>: {<traj>: float}, ...},  # if non-empty
            "test_rmse": {<quantity>: {<traj>: float}, ...},  # if non-empty
        }
        where `<quantity>` is one of `_RECON_QUANTITIES`. Splits with no
        trajectories are omitted.
    """
    print("\n" + "=" * 60 +
          f"\n  Reconstruction\n  ckpt_dir={ckpt_dir}\n" + "=" * 60)
    print(f"  GPU before: {_gpu_mem_str()}")

    splits = _resolve_splits(trajectories_id, trajectories_ood, trajectories_test)

    model, train_cfg, norm_stats = _build_pinc_model_and_stats(
        ckpt_dir, device, data_path,
    )
    n_params_M = sum(p.numel() for p in model.parameters()) / 1e6
    sep_zf = bool(train_cfg.dataset.separate_zf)
    offset = int(train_cfg.dataset.offset)

    integrator = FluxIntegral(
        flux_fields=True, spectral_df=False, spectral_potens=True,
    )

    all_trajs = [t for _suffix, trajs in splits for t in trajs]
    cli = SimpleNamespace(
        trajectories=all_trajs,
        n_samples=1, batch_size=batch_size, device=str(device),
    )
    inf_cfg = load_inference_config(inference_cfg, cli)
    if data_path is not None:
        inf_cfg["root"] = str(data_path)

    # Encoder/decoder conditioning union (matches evaluate_generative).
    model_key = "autoencoder" if hasattr(train_cfg, "autoencoder") else "model"
    model_cfg = getattr(train_cfg, model_key)
    cond_keys = sorted(set(getattr(model_cfg, "decoder_conditioning", []))
                       | set(getattr(model_cfg, "encoder_conditioning", [])))

    shift_t = torch.as_tensor(np.asarray(norm_stats["df_mean"]),
                              dtype=torch.float32, device=device)
    scale_t = torch.as_tensor(np.asarray(norm_stats["df_std"]),
                              dtype=torch.float32, device=device)
    df_shape = (2, 32, 8, 16, 85, 32)  # raw bin layout; separate_zf expands ch=0

    def _process(traj):
        meta_path = os.path.join(inf_cfg["root"], traj, "metadata.pkl")
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        condition = (
            get_conditioning(cond_keys, meta, device) if cond_keys else None
        )
        # Geometry pulled per-trajectory; expanded to batch shape on each iter.
        geometry = get_geometry(meta)

        gt_path = os.path.join(inf_cfg["root"], traj, "data")
        timesteps = sorted(
            f for f in os.listdir(gt_path) if f.startswith("timestep")
        )[offset:]

        sq_sum  = {q: 0.0 for q in _RECON_QUANTITIES}
        n_elems = {q: 0   for q in _RECON_QUANTITIES}
        for i in tqdm(range(0, len(timesteps), batch_size),
                      desc=f"  {traj}", unit="batch"):
            batch_files = timesteps[i:i + batch_size]
            dfs = []
            for ts in batch_files:
                arr = np.fromfile(os.path.join(gt_path, ts),
                                  dtype=np.float32).reshape(df_shape)
                if sep_zf:
                    arr = separate_zf(arr, dim=0)
                dfs.append(arr)
            df_phys = torch.from_numpy(np.stack(dfs)).to(device)
            B = df_phys.shape[0]

            shift_b = expand_as(shift_t, df_phys)
            scale_b = expand_as(scale_t, df_phys)
            df_norm = (df_phys - shift_b) / scale_b

            cond_b = (condition.unsqueeze(0).expand(B, -1).to(device)
                      if condition is not None else None)
            kwargs = {} if cond_b is None else {"condition": cond_b}
            recon_norm = model(df_norm, **kwargs)["df"]
            recon_phys = recon_norm * scale_b + shift_b

            # Point-wise df RMSE in physical space.
            err_df = (recon_phys - df_phys).pow(2)
            sq_sum["df"]  += float(err_df.sum().item())
            n_elems["df"] += df_phys.numel()

            # Physics integrals on both. compute_physics expects (B, C, ...);
            # geometry must be batched with leading dim B. Run on CPU since
            # the integrator uses float64 Bessel functions via NVRTC.
            geom_b = tree_map(
                lambda g: g.unsqueeze(0).expand(B, *g.shape), geometry,
            )
            phys_pred = compute_physics(
                recon_phys.float().cpu(), geom_b, sep_zf, integrator,
            )
            phys_gt = compute_physics(
                df_phys.float().cpu(), geom_b, sep_zf, integrator,
            )
            for k in ("eflux", "kxspec", "kyspec", "qspec"):
                err = (phys_pred[k] - phys_gt[k]).pow(2)
                sq_sum[k]  += float(err.sum().item())
                n_elems[k] += err.numel()

        return {q: float((sq_sum[q] / n_elems[q]) ** 0.5)
                for q in _RECON_QUANTITIES if n_elems[q] > 0}

    out = {"n_params_M": float(n_params_M)}
    for suffix, _ in splits:
        out[f"{suffix}_rmse"] = {q: {} for q in _RECON_QUANTITIES}
    try:
        for suffix, trajs in splits:
            split_key = f"{suffix}_rmse"
            for traj in trajs:
                per_qty = _process(traj)
                for q, v in per_qty.items():
                    out[split_key][q][traj] = v
    finally:
        _release_module(model)
        del model, train_cfg, norm_stats
        free_cuda()
    print(f"  GPU after:  {_gpu_mem_str()}")
    return out


def build_recon_table(recon_results):
    """Build a reconstruction-error table from
    `{model_label: <evaluate_reconstruction output>}`. Rows are
    `(model, split)` pairs over ID / OOD / TEST (matching the summary table
    layout); columns are `params_M` plus `<quantity>_RMSE` /
    `<quantity>_RMSE_std` for each of `df`, `eflux`, `kxspec`, `kyspec`,
    `fluxspec` (= `qspec`). Std is sample std (ddof=1) of per-trajectory
    RMSEs across that split. Splits absent from a model's results are
    skipped (no row emitted).
    """
    rows = []
    for label, r in recon_results.items():
        params_M = float(r.get("n_params_M", float("nan")))
        for split_label, split_suffix, _default in SUMMARY_SPLITS:
            split_key = f"{split_suffix}_rmse"
            split = r.get(split_key)
            if not split:
                continue
            row = {"model": label, "split": split_label, "params_M": params_M}
            for qty in _RECON_QUANTITIES:
                vals = list(split.get(qty, {}).values())
                col = _RECON_QTY_TO_COL[qty]
                if not vals:
                    row[col] = float("nan")
                    row[f"{col}_std"] = float("nan")
                else:
                    row[col] = float(np.mean(vals))
                    row[f"{col}_std"] = (
                        float(np.std(vals, ddof=1)) if len(vals) >= 2 else 0.0
                    )
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index(["model", "split"])
    cols = ["params_M"]
    for qty in _RECON_QUANTITIES:
        col = _RECON_QTY_TO_COL[qty]
        cols += [col, f"{col}_std"]
    return df[[c for c in cols if c in df.columns]]


def format_recon_latex(recon_results, *,
                       quantities=("df",),
                       splits=None,
                       caption=None,
                       label=None,
                       precision=3):
    """LaTeX recon-error table with one column-group per (split, quantity).

    `quantities` is a string ("df") or a list/tuple of strings from
    {"df", "eflux", "kxspec", "kyspec", "qspec"}. Each renders as
    `len(splits)` columns -- one per split (ID/OOD/TEST by default) -- so
    the table is ``2 + len(splits) * len(quantities)`` columns wide.

    `splits` is a list of split labels to show: any subset of
    {"ID", "OOD", "TEST"} in the order you want them rendered. Defaults to
    all splits actually present in the results (in SUMMARY_SPLITS order).

    `precision` selects significant-digit precision via ``:.<precision>g``,
    so big spectral values gracefully fall back to scientific notation
    rendered as ``<m>\\times 10^{<e>}``.
    """
    if isinstance(quantities, str):
        quantities = (quantities,)
    quantities = tuple(quantities)
    n = len(quantities)

    # Determine which splits to render: caller-supplied, otherwise auto-detect
    # the splits that any model has results for, ordered by SUMMARY_SPLITS.
    _split_specs = [(label_, suffix) for label_, suffix, _ in SUMMARY_SPLITS]
    if splits is None:
        present = set()
        for r in recon_results.values():
            for label_, suffix in _split_specs:
                if r.get(f"{suffix}_rmse"):
                    present.add(label_)
        active_splits = [(lab, suf) for lab, suf in _split_specs if lab in present]
    else:
        wanted = {s.upper() for s in splits}
        active_splits = [(lab, suf) for lab, suf in _split_specs if lab in wanted]
    if not active_splits:
        active_splits = [("ID", "id"), ("OOD", "ood")]
    n_splits = len(active_splits)

    if caption is None:
        caption = (
            "Reconstruction RMSE in physical (denormalised) units, "
            "evaluated for each autoencoder backbone underlying the latent "
            "generative models. Parameter counts in millions (M). ID and "
            "OOD entries denote the mean $\\pm$ standard deviation across "
            "trajectories of the corresponding test split."
        )
    if label is None:
        label = "tab:ae_recon_rmse"

    quantity_label = {
        "df":     r"{Recon}_{\mathrm{RMSE}}",
        "eflux":  r"\bar{Q}_{\mathrm{RMSE}}",
        "kxspec": r"k_x\text{-spec}_{\mathrm{RMSE}}",
        "kyspec": r"k_y\text{-spec}_{\mathrm{RMSE}}",
        "qspec":  r"Q\text{-spec}_{\mathrm{RMSE}}",
    }

    def _agg(r, split_key, qty):
        vals = list(r.get(split_key, {}).get(qty, {}).values())
        if not vals:
            return float("nan"), float("nan")
        m = float(np.mean(vals))
        s = float(np.std(vals, ddof=1)) if len(vals) >= 2 else 0.0
        return m, s

    def _fmt(x):
        if not np.isfinite(x):
            return "---"
        s = f"{x:.{precision}g}"
        if "e" in s:
            mant, exp = s.split("e")
            return rf"{mant}\times 10^{{{int(exp)}}}"
        return s

    def _cell(mean, std):
        if not np.isfinite(mean):
            return "---"
        return rf"${_fmt(mean)}_{{\pm {_fmt(std)}}}$"

    col_spec = "lc" + "c" * (n_splits * n)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
    ]
    head = [
        r"\multirow{2}{*}{\textbf{Method}}",
        r"\multirow{2}{*}{\textbf{Params (M)}}",
    ]
    for qty in quantities:
        ql = quantity_label.get(qty, qty)
        head.append(rf"\multicolumn{{{n_splits}}}{{c}}{{${ql}\downarrow$}}")
    lines.append(" & ".join(head) + r" \\")
    lines.append(" ".join(
        rf"\cmidrule(lr){{{3 + n_splits * i}-{2 + n_splits * (i + 1)}}}"
        for i in range(n)
    ))
    sub = ["", ""]
    for _ in quantities:
        for split_label_, _suf in active_splits:
            sub.append(rf"\textbf{{{split_label_}}}")
    lines.append(" & ".join(sub) + r" \\")
    lines.append(r"\midrule")
    for model_label, r in recon_results.items():
        params = float(r.get("n_params_M", float("nan")))
        cells = [model_label, f"{params:.1f}"]
        for qty in quantities:
            for _split_label, suffix in active_splits:
                m, s = _agg(r, f"{suffix}_rmse", qty)
                cells.append(_cell(m, s))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def evaluate_vqvae(ckpt_dir, inference_cfg, *,
                   vq_index_pkl=None,
                   trajectories_id=TRAJECTORIES_ID,
                   trajectories_ood=TRAJECTORIES_OOD,
                   trajectories_test=TRAJECTORIES_TEST,
                   n_samples=128, batch_size=64,
                   output_path=None,
                   data_path=None,
                   device=torch.device("cuda")):
    """Evaluate a Swin5DVQVAE on ID + OOD (+ TEST if non-empty). Returns
    one entry per non-empty split keyed `vqvae_<id|ood|test>`."""
    print("\n" + "=" * 60 + "\n  VQ-VAE\n" + "=" * 60)
    print(f"  GPU before: {_gpu_mem_str()}")
    splits = _resolve_splits(trajectories_id, trajectories_ood, trajectories_test)
    def _run():
        integrator = FluxIntegral(flux_fields=True, spectral_df=False, spectral_potens=True)
        model, train_cfg, norm_stats = _build_pinc_model_and_stats(ckpt_dir, device, data_path)
        assert isinstance(model, Swin5DVQVAE), \
            f"Expected VQ-VAE at {ckpt_dir}, got {type(model).__name__}"
        vq_prior = None
        if vq_index_pkl:
            vq_prior = compute_codebook_prior(vq_index_pkl, model.vq.codebook_size)
            nz = int((vq_prior > 0).sum().item())
            print(f"  empirical codebook prior: {nz}/{model.vq.codebook_size} codes used")
        try:
            out = {
                f"vqvae_{suffix}": _run_pinc_trajectories(
                    model, ckpt_dir, train_cfg, norm_stats,
                    trajs, inference_cfg, n_samples,
                    batch_size, device, integrator,
                    vq_prior=vq_prior, data_path=data_path,
                )
                for suffix, trajs in splits
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
                ae_checkpoint=None,
                trajectories_id=TRAJECTORIES_ID,
                trajectories_ood=TRAJECTORIES_OOD,
                trajectories_test=TRAJECTORIES_TEST,
                n_samples=128, batch_size=64,
                output_path=None,
                data_path=None,
                device=torch.device("cuda")):
    """Evaluate AR-transformer + VQ-VAE decoder on ID + OOD (+ TEST if
    non-empty). Returns one entry per non-empty split keyed
    `ar_<id|ood|test>`.

    `ae_checkpoint` overrides `ar_cfg.ae_checkpoint` (which often points at a
    stale training-time HPC path). Pass the same VQ-VAE directory you use for
    `evaluate_vqvae`.
    """
    print("\n" + "=" * 60 + "\n  AR (over VQ-VAE codes)\n" + "=" * 60)
    print(f"  GPU before: {_gpu_mem_str()}")
    splits = _resolve_splits(trajectories_id, trajectories_ood, trajectories_test)
    def _run():
        integrator = FluxIntegral(flux_fields=True, spectral_df=False, spectral_potens=True)
        ar_model, ae_model, ar_cfg, norm_stats = _build_ar_model(
            ar_ckpt_dir, device, data_path, ae_checkpoint=ae_checkpoint,
        )
        try:
            out = {
                f"ar_{suffix}": _run_ar_trajectories(
                    ar_model, ae_model, ar_cfg, norm_stats,
                    trajs, inference_cfg,
                    n_samples, batch_size, device, integrator,
                    data_path=data_path,
                )
                for suffix, trajs in splits
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
    sample_time_s = 0.0
    decode_time_s = 0.0
    n_done = 0
    while remaining > 0:
        bs = min(gen_batch_size, remaining)
        c = cond.expand(bs, -1).to(device)
        # Time denoising and decoding separately so probe rows can report
        # sample+probe (probes work on latents) and the Diff 5D row can report
        # sample+decode (5D physics needs the decoded df).
        with _GpuTimer() as _t_sample:
            z = runner.sample(c, steps=n_denoising_steps, latent_only=True)
        sample_time_s += _t_sample.elapsed
        all_latents.append(z.detach().cpu().numpy().reshape(z.shape[0], -1))

        with _GpuTimer() as _t_decode:
            decoded = runner.autoencoder.decode(z, condition=c)
        decode_time_s += _t_decode.elapsed
        n_done += bs

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
    gen_res["_sample_time_s"] = float(sample_time_s)
    gen_res["_decode_time_s"] = float(decode_time_s)
    gen_res["_gen_time_s"]    = float(sample_time_s + decode_time_s)
    gen_res["_n_samples"]     = int(n_done)

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
    """Add probe_<key>_<ae|gen> entries to gen[] for each trajectory.

    Records `_probe_time_s` per trajectory (PCA project + ridge predict for all
    samples in that trajectory) so the probe summary rows can report
    sample+diffusion+probe wall time.
    """
    probes = probe_pack["probes"]
    spec_probes = probe_pack["spec_probes"]
    pca = probe_pack["pca"]
    for traj, res in group.items():
        gen = res["gen"]
        if "_latents" not in gen:
            continue
        X = gen["_latents"]
        with _GpuTimer() as _t:
            X_pca = pca.transform(X)
            gen["probe_flux_ae"]  = probes["flux"]["probe_ae"].predict(X_pca)
            gen["probe_flux_gen"] = probes["flux"]["probe_gen"].predict(X_pca)
            for sk, sp in spec_probes.items():
                for v in ("ae", "gen"):
                    pred_norm = sp[f"probe_{v}"].predict(X_pca)
                    pred_log  = pred_norm * sp["std"] + sp["mean"]
                    gen[f"probe_{sk}_{v}"] = np.expm1(pred_log)
        gen["_probe_time_s"] = float(_t.elapsed)
    return group


def evaluate_diff(diff_ckpt_dir, ae_checkpoint, *,
                  data_path,
                  trajectories_id=TRAJECTORIES_ID,
                  trajectories_ood=TRAJECTORIES_OOD,
                  trajectories_test=TRAJECTORIES_TEST,
                  n_samples=128, gen_batch_size=64, n_denoising_steps=20,
                  with_probes=True,
                  probe_n_components=64, probe_alpha=1.0,
                  probe_subsample=256, probe_seed=0,
                  model_snapshot="best.pth",
                  output_path=None,
                  device=torch.device("cuda")):
    """Evaluate a flow-matching diffusion model on ID + OOD (+ TEST if
    non-empty).

    Returns one entry per non-empty split keyed `diff_<id|ood|test>`. With
    probes=True the gen dicts also carry probe_flux_<ae|gen>,
    probe_kxspec_<ae|gen>, probe_kyspec_<ae|gen>, probe_fluxspec_<ae|gen>
    arrays of shape (N, ...) for the probe rows.
    """
    print("\n" + "=" * 60 + "\n  Diff (flow-matching)\n" + "=" * 60)
    print(f"  GPU before: {_gpu_mem_str()}")
    splits = _resolve_splits(trajectories_id, trajectories_ood, trajectories_test)
    def _run():
        integrator = FluxIntegral(flux_fields=True, spectral_df=False, spectral_potens=True)
        valid_h5 = [
            t.replace("_ifft_realpotens", "") + ".h5"
            for _suffix, trajs in splits for t in trajs
        ]
        runner = _build_diff_runner(diff_ckpt_dir, ae_checkpoint, data_path,
                                    valid_h5, model_snapshot, device)
        try:
            valset = runner.valsets[0]
            # Map trajectory base -> (output key, group dict).
            base_to_key = {}
            out = {}
            for suffix, trajs in splits:
                key = f"diff_{suffix}"
                out[key] = {}
                for t in trajs:
                    base_to_key[t.replace("_ifft_realpotens", "")] = key

            for fi, fpath in enumerate(valset.files):
                base = _traj_basename(fpath)
                key = base_to_key.get(base)
                if key is None:
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
                for k in out:
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


@torch.no_grad()
def evaluate_diff_n_sweep(diff_ckpt_dir, ae_checkpoint, *,
                          data_path,
                          n_denoising_steps_list,
                          trajectories_id=TRAJECTORIES_ID,
                          trajectories_ood=TRAJECTORIES_OOD,
                          trajectories_test=TRAJECTORIES_TEST,
                          n_samples=128, gen_batch_size=64,
                          model_snapshot="best.pth",
                          output_path_fn=None,
                          device=torch.device("cuda")):
    """Sweep `n_denoising_steps` over `n_denoising_steps_list` while loading
    the diffusion runner ONCE.

    Use this for ablations over the number of Euler steps -- spinning up
    `evaluate_diff` per N reloads the model + autoencoder + trainset every
    iteration, which dominates wall time. Here the runner is built once and
    we just call `runner.sample(..., steps=n)` per N value.

    `output_path_fn(n) -> str | None` is an optional callable that returns a
    cache path for each N value; cached N's are loaded directly and skipped
    from the rollout pass. Probes are not fit (probe inference depends on N
    in non-trivial ways; if you need them, use `evaluate_diff` separately).

    Returns: {n_denoising_steps: {"diff_id": {<traj>: ...}, "diff_ood": ...}}.
    """
    print("\n" + "=" * 60 +
          f"\n  Diff (flow-matching, N sweep over {list(n_denoising_steps_list)})\n"
          + "=" * 60)
    print(f"  GPU before: {_gpu_mem_str()}")

    # Load whatever's already cached, defer the rest to one runner build.
    # Corrupted caches (e.g. from an interrupted run) are skipped with a
    # warning instead of crashing -- the N is queued for re-generation.
    results = {}
    pending = []
    for n in n_denoising_steps_list:
        path = output_path_fn(n) if output_path_fn else None
        if path and os.path.exists(path) and os.path.getsize(path) > 0:
            try:
                cached = torch.load(path, weights_only=False)
                results[n] = cached
                print(f"  [cache hit ] N={n}: {path}")
                continue
            except (RuntimeError, EOFError, OSError, pickle.UnpicklingError) as e:
                print(f"  [cache CORRUPT] N={n}: {path}  ({type(e).__name__}: {e}); will regenerate")
                # Move the bad file aside so a fresh save can take its slot.
                try:
                    os.replace(path, path + ".corrupt")
                except OSError:
                    pass
        elif path and os.path.exists(path):
            print(f"  [cache empty] N={n}: {path}; will regenerate")
            try:
                os.replace(path, path + ".corrupt")
            except OSError:
                pass
        pending.append(n)

    if not pending:
        print(f"  GPU after:  {_gpu_mem_str()}")
        return results

    integrator = FluxIntegral(
        flux_fields=True, spectral_df=False, spectral_potens=True,
    )
    splits = _resolve_splits(trajectories_id, trajectories_ood, trajectories_test)
    valid_h5 = [
        t.replace("_ifft_realpotens", "") + ".h5"
        for _suffix, trajs in splits for t in trajs
    ]
    runner = _build_diff_runner(
        diff_ckpt_dir, ae_checkpoint, data_path,
        valid_h5, model_snapshot, device,
    )
    try:
        valset = runner.valsets[0]
        # Map trajectory base -> output key, computed once.
        base_to_key = {}
        for suffix, trajs in splits:
            key = f"diff_{suffix}"
            for t in trajs:
                base_to_key[t.replace("_ifft_realpotens", "")] = key

        for n in pending:
            print(f"\n  --- N = {n} ---")
            out = {f"diff_{suffix}": {} for suffix, _ in splits}
            for fi, fpath in enumerate(valset.files):
                base = _traj_basename(fpath)
                key = base_to_key.get(base)
                if key is None:
                    continue
                print(f"  [{key}] {base}")
                _, res = _diff_evaluate_trajectory(
                    runner, fi, n_samples, gen_batch_size, n,
                    integrator, device,
                )
                out[key][base] = res
                free_cuda()

            results[n] = out
            path = output_path_fn(n) if output_path_fn else None
            if path:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                torch.save(out, path)
                print(f"  [cache save] N={n}: {path}")
    finally:
        _release_module(getattr(runner, "model", None))
        _release_module(getattr(runner, "autoencoder", None))
        del runner
        free_cuda()
    print(f"  GPU after:  {_gpu_mem_str()}")
    return results


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
                         offset, flux_mean, flux_std,
                         data_prep, sep_zf, device, return_df=True,
                         flux_key="flux"):
    """`flux_key` is the output dict key the GyroSwin uses for its flux head;
    "flux" for per-step flux (old codebase + the GyroSwin_tiny ckpt) and
    "fluxavg" for the warm-start checkpoints that supervise the time-averaged
    flux directly. `flux_mean` / `flux_std` must match that key's normalization.
    """
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
        flux_norm = float(out[flux_key].squeeze().detach().cpu())
        flux_pred[step] = flux_norm * flux_std + flux_mean
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


def _gyroswin_evaluate_trajectory(traj, n_steps, offset, flux_mean, flux_std,
                                   data_prep, sep_zf, gs_model, gs_stats, gs_cond_keys,
                                   device, flux_key="flux"):
    with _GpuTimer() as _t_roll:
        roll = _gyroswin_ar_rollout(
            gs_model, gs_stats, gs_cond_keys, traj, n_steps, offset,
            flux_mean, flux_std, data_prep, sep_zf, device,
            return_df=True, flux_key=flux_key,
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
    # GyroSwin rollout = 1 "sample" per trajectory (autoregressive over the
    # whole horizon), so n_samples=1 and time_per_sample = total rollout time.
    gen_res["_gen_time_s"] = float(_t_roll.elapsed)
    gen_res["_n_samples"]  = 1
    gt_res = {k: _stats(v if isinstance(v, torch.Tensor) else torch.as_tensor(v))
              for k, v in gt_phys_full.items()}
    gt_res = _add_meta_targets(gt_res, meta)
    return {"gen": gen_res, "gt": gt_res, "t_grid": roll["t_grid"]}


def _scalar_stat(arr, name):
    """Coerce a stats entry (np.ndarray | scalar | list) to a Python float.

    `flux` and `fluxavg` are scalars per timestep, so the corresponding entry
    in `trainset.stats[<field>]["full"]` should be shape () or (1,). If the
    stored stats are larger (recompute disagreed with the field's dimensionality
    -- usually a stale or wrong pickle), reduce via `mean()` and print a
    warning rather than blowing up so the user can inspect the result.
    """
    a = np.asarray(arr).reshape(-1)
    if a.size == 0:
        raise ValueError(f"empty {name} stats")
    if a.size > 1:
        print(f"  WARNING: {name} stats has shape {np.asarray(arr).shape}; "
              f"reducing to scalar via mean of {a.size} values.")
        return float(a.mean())
    return float(a.item())


def evaluate_gyroswin_new(gyroswin_ckpt_dir, *,
                          data_path,
                          variant="new",
                          trajectories_id=TRAJECTORIES_ID,
                          trajectories_ood=TRAJECTORIES_OOD,
                          trajectories_test=TRAJECTORIES_TEST,
                          n_steps=128,
                          start_timestep=80,
                          flux_mean=None, flux_std=None,
                          output_path=None,
                          device=torch.device("cuda")):
    """Evaluate a GyroSwin trained with the *current* codebase on ID + OOD.

    Loads via `neugk.gyroswin.models.get_model` -- no monkey-patching, same
    pattern as the other generative chapters. The new GyroSwinMultitask
    predicts `flux` per-step (not `fluxavg`), so denormalization uses
    trainset.stats["flux"]["full"].

    `start_timestep` (default 80) is the trajectory frame the rollout starts
    from. Pass 0 to seed from the initial condition, 10 for the 10th frame, etc.

    `variant` is a short name (e.g. "warm", "cold", "new") that becomes part
    of the returned dict keys (`gyroswin_<variant>_id` / `_ood`) and the label
    in the summary table (`GyroSwin (<variant>)`). Use distinct variants when
    you want several GyroSwin checkpoints to coexist in `all_results`.

    `flux_mean` / `flux_std` (optional) override the trainset's flux
    normalization stats. Use this if the dataset's auto-computed flux stats
    look wrong (e.g. coming back with the wrong shape) and you want to pin
    explicit values for the rollout's flux denormalization.
    """
    splits = _resolve_splits(trajectories_id, trajectories_ood, trajectories_test)
    print("\n" + "=" * 60 +
          f"\n  GyroSwin ({variant}, AR rollout)\n" + "=" * 60)
    print(f"  GPU before: {_gpu_mem_str()}")
    def _run():
        from neugk.gyroswin.models import get_model as get_gyroswin_model

        cfg = _notebook_safe(omegaconf.OmegaConf.load(
            os.path.join(gyroswin_ckpt_dir, "config.yaml"),
        ))
        cfg.dataset.path = str(data_path)
        cfg.dataset.gds_override = True

        print(f"  building trainset for {gyroswin_ckpt_dir} ...")
        datasets, _, _ = get_data(cfg, rank=0)
        trainset = datasets[0]

        print(f"  building GyroSwin model ...")
        gs_model = get_gyroswin_model(cfg, dataset=trainset).to(device).eval()
        ckpt = torch.load(os.path.join(gyroswin_ckpt_dir, "best.pth"),
                          map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        gs_model.load_state_dict(state, strict=True)
        print(f"  GyroSwin ckpt epoch={ckpt.get('epoch', '?')}, "
              f"params={sum(p.numel() for p in gs_model.parameters())/1e6:.1f}M")

        # Some checkpoints (the warm-start ones) supervise time-averaged flux
        # via `fluxavg` instead of per-step `flux`. The model's `outputs` list
        # tells us which output key to read in the rollout, and the matching
        # trainset.stats[<key>] has the right normalization shape (the
        # "wrong" key is computed via the probe-targets path and ends up as a
        # per-trajectory vector, which is what surfaced as the
        # `only length-1 arrays can be converted to Python scalars` error).
        model_outputs = list(getattr(gs_model, "outputs", []))
        flux_key = next((k for k in ("flux", "fluxavg") if k in model_outputs), None)
        if flux_key is None:
            raise RuntimeError(
                f"GyroSwin model has no flux output (model.outputs={model_outputs}). "
                "Cannot run AR rollout without a flux head."
            )
        print(f"  flux output key: {flux_key!r} (model.outputs={model_outputs})")

        # Flux denorm stats: caller override > trainset.stats[<flux_key>].
        if flux_mean is not None and flux_std is not None:
            flux_mean_v = float(flux_mean)
            flux_std_v  = float(flux_std)
            print(f"  using user-provided {flux_key} stats: "
                  f"mean={flux_mean_v:.4g}, std={flux_std_v:.4g}")
        else:
            stats_entry = trainset.stats.get(flux_key, {}).get("full")
            if not (stats_entry and "mean" in stats_entry and "std" in stats_entry):
                raise RuntimeError(
                    f"trainset.stats[{flux_key!r}] missing -- "
                    f"pass flux_mean= / flux_std= explicitly."
                )
            flux_mean_v = _scalar_stat(stats_entry["mean"], f"{flux_key}.full.mean")
            flux_std_v  = _scalar_stat(stats_entry["std"],  f"{flux_key}.full.std")

        # Reshape df stats so they look like the old gs_stats pickle layout
        # (gs_stats[<field>]["full"]["mean"|"std"]) consumed by
        # `_gyroswin_scale_shift`.
        gs_stats = {"df": {"full": {
            "mean": np.asarray(trainset.stats["df"]["full"]["mean"]),
            "std":  np.asarray(trainset.stats["df"]["full"]["std"]),
        }}}
        del datasets

        offset    = int(start_timestep)
        sep_zf    = bool(cfg.dataset.separate_zf)
        cond_keys = sorted(list(cfg.model.conditioning))

        try:
            out = {f"gyroswin_{variant}_{suffix}": {} for suffix, _ in splits}
            for suffix, trajs in splits:
                label = f"gyroswin_{variant}_{suffix}"
                for traj in trajs:
                    print(f"  [{label}] {traj} (start_t={offset})")
                    out[label][traj] = _gyroswin_evaluate_trajectory(
                        traj, n_steps, offset, flux_mean_v, flux_std_v,
                        Path(data_path), sep_zf, gs_model, gs_stats, cond_keys, device,
                        flux_key=flux_key,
                    )
                free_cuda()
        finally:
            _release_module(gs_model)
            del gs_model, trainset, gs_stats
            free_cuda()
        return out
    out = _cached_or_run(output_path, _run)
    print(f"  GPU after:  {_gpu_mem_str()}")
    return out


def load_gyroswin_xxl_results(autoreg_root, *,
                              data_path,
                              trajectories_id=TRAJECTORIES_ID,
                              trajectories_ood=TRAJECTORIES_OOD,
                              trajectories_test=TRAJECTORIES_TEST,
                              offset=80,
                              n_last_steps=80):
    """Load pre-computed AR-rollout outputs from a `gyroswin_xxl_*/autoreg_t0/`
    directory and assemble result groups compatible with `build_summary_table`.

    Layout expected per trajectory under `<autoreg_root>/<traj>/best/`:
      * `K{step}/flux`            — per-step model flux-head scalar (text).
                                    Same convention as the existing `GyroSwin`
                                    row, which puts the flux-head output in the
                                    `eflux` slot. The dump's `K{step}/eflux`
                                    file is the integrator-derived energy flux
                                    (different field) and is *not* used here.
      * `K{step}/kyspec`          — per-step predicted (32,) ky spectrum (text)
      * `K{step}/eflux_spectra`   — per-step predicted (32,) eflux spectrum (= qspec)

    `K{step}` is timestep-indexed; step starts at offset+1 (i.e. K81 with
    `offset=80`) and runs to the end of the trajectory. We don't recompute
    GT integrals here -- the GT references are pulled straight from the
    trajectory's `metadata.pkl` under `data_path`. `kxspec` isn't dumped, so
    that column comes out NaN for this row.

    Trajectory naming:
      * ID/TEST: `<traj_base>` under `autoreg_root/`, e.g. `iteration_148`.
             The matching metadata dir is `<traj_base>_ifft_realpotens` under
             `data_path`.
      * OOD: under `autoreg_root/ood/<traj_base>/best/`. The matching
             metadata dir is `ood_<traj_base>_ifft_realpotens`.

    Trajectories that don't have a corresponding K* dump are skipped with a
    note (some autoreg dumps are smaller than the standard ID/OOD set).

    `n_last_steps` (default 80) restricts metric aggregation to the final
    `n_last_steps` per-step predictions of each trajectory (and the matching
    GT slice). The GyroSwin XXL row is computed only over the saturated
    regime in this convention, matching the `last-80` averaging used by
    `_add_meta_targets`. Pass `None` to keep all rolled-out steps.

    Returns: {"gyroswin_xxl_<id|ood|test>": {<traj>: {gen, gt}, ...}, ...}
    (one entry per non-empty split).
    """
    autoreg_root = str(autoreg_root)
    data_path = Path(data_path)
    splits = _resolve_splits(trajectories_id, trajectories_ood, trajectories_test)
    out = {f"gyroswin_xxl_{suffix}": {} for suffix, _ in splits}

    # Build (group_key, traj_label, autoreg_subdir, meta_dir) tuples.
    plan = []
    for suffix, trajs in splits:
        group_key = f"gyroswin_xxl_{suffix}"
        for t in trajs:
            base = t.replace("_ifft_realpotens", "")
            if suffix == "ood" and base.startswith("ood_"):
                # `ood_iteration_0_ifft_realpotens` -> autoreg `ood/iteration_0`.
                sub = "ood/" + base[len("ood_"):]
            else:
                sub = base
            plan.append((group_key, t, sub, t))

    print(f"\n{'=' * 60}\n  GyroSwin XXL (load cached AR rollouts)\n{'=' * 60}")
    for group_key, traj_label, sub, meta_subdir in plan:
        best_dir = os.path.join(autoreg_root, sub, "best")
        meta_pkl = data_path / meta_subdir / "metadata.pkl"
        if not os.path.isdir(best_dir):
            print(f"  [skip] {traj_label}: no autoreg dump at {best_dir}")
            continue
        if not meta_pkl.exists():
            print(f"  [skip] {traj_label}: no metadata at {meta_pkl}")
            continue

        Ks = sorted(
            (d for d in os.listdir(best_dir) if d.startswith("K") and d[1:].isdigit()),
            key=lambda s: int(s[1:]),
        )
        if not Ks:
            print(f"  [skip] {traj_label}: no K* dirs under {best_dir}")
            continue

        # Per-step predictions. Use `K{step}/flux` (model flux-head scalar),
        # not `K{step}/eflux` (integrator-derived) -- matches the existing
        # GyroSwin row's convention where gen_phys["eflux"] is overridden with
        # the flux-head output.
        try:
            pred_eflux = np.array([
                float(open(os.path.join(best_dir, k, "flux")).read().strip())
                for k in Ks
            ], dtype=np.float32)
            pred_ky = np.stack([
                np.loadtxt(os.path.join(best_dir, k, "kyspec"))
                for k in Ks
            ]).astype(np.float32)
            pred_es = np.stack([
                np.loadtxt(os.path.join(best_dir, k, "eflux_spectra"))
                for k in Ks
            ]).astype(np.float32)
        except FileNotFoundError as e:
            print(f"  [skip] {traj_label}: {e}")
            continue

        n = len(Ks)
        with open(meta_pkl, "rb") as f:
            meta = pickle.load(f)

        # Older trajectory metadata sometimes calls the per-step flux series
        # 'fluxes' instead of 'flux'. Accept either; skip the trajectory if
        # neither is present.
        if "flux" in meta:
            flux_key = "flux"
        elif "fluxes" in meta:
            flux_key = "fluxes"
        else:
            print(f"  [skip] {traj_label}: metadata has neither 'flux' nor 'fluxes'")
            continue

        # GT references: `meta[<key>][offset:offset+n]` aligns one-to-one with
        # the predicted K-steps. The slice clamps if the metadata is shorter
        # than offset+n.
        gt_flux_arr = np.asarray(meta[flux_key])[offset:offset + n].astype(np.float32)
        gt_ky_arr = (
            np.asarray(meta["kyspec"])[offset:offset + n].astype(np.float32)
            if "kyspec" in meta else None
        )

        # Restrict to the last `n_last_steps` (saturated-regime convention).
        # The pred and GT series are already index-aligned, so the same tail
        # slice applies to both.
        if n_last_steps is not None and n_last_steps < n:
            pred_eflux = pred_eflux[-n_last_steps:]
            pred_ky    = pred_ky[-n_last_steps:]
            pred_es    = pred_es[-n_last_steps:]
            gt_flux_arr = gt_flux_arr[-n_last_steps:]
            if gt_ky_arr is not None:
                gt_ky_arr = gt_ky_arr[-n_last_steps:]
            n_kept = n_last_steps
        else:
            n_kept = n

        pred_eflux_t = torch.as_tensor(pred_eflux)
        pred_ky_t    = torch.as_tensor(pred_ky)
        pred_es_t    = torch.as_tensor(pred_es)

        gen = {
            "eflux":  {"all":  pred_eflux_t,
                       "mean": pred_eflux_t.mean(),
                       "std":  pred_eflux_t.std()},
            "kyspec": {"all":  pred_ky_t,
                       "mean": pred_ky_t.mean(0),
                       "std":  pred_ky_t.std(0)},
            "qspec":  {"all":  pred_es_t,
                       "mean": pred_es_t.mean(0),
                       "std":  pred_es_t.std(0)},
            # Single autoregressive rollout per trajectory; no on-demand timing
            # available (these are pre-dumped outputs).
            "_n_samples": 1,
        }

        gt_flux_t = torch.as_tensor(gt_flux_arr)
        gt = {
            "eflux":  {"mean": gt_flux_t.mean(),
                       "std":  gt_flux_t.std(),
                       "all":  gt_flux_t,
                       "full": gt_flux_t},
        }
        if gt_ky_arr is not None:
            gt_ky_t = torch.as_tensor(gt_ky_arr)
            gt["kyspec"] = {"mean": gt_ky_t.mean(0),
                            "std":  gt_ky_t.std(0),
                            "all":  gt_ky_t,
                            "full": gt_ky_t}
        # `_add_meta_targets` adds meta_kyspec_mean / meta_fluxspec_mean (last-80
        # convention) so the qspec-vs-meta_fluxspec_mean column has a target.
        # Already silently skips keys missing from this trajectory's metadata.
        gt = _add_meta_targets(gt, meta)
        gt = _add_full_alias(gt)

        out[group_key][traj_label] = {"gen": gen, "gt": gt}
        notes = []
        if flux_key != "flux":
            notes.append(f"meta key='{flux_key}'")
        if gt_ky_arr is None:
            notes.append("no GT kyspec")
        note_str = f"  [{', '.join(notes)}]" if notes else ""
        steps_str = (
            f"n_steps={n_kept}/{n} (last-{n_last_steps})"
            if n_last_steps is not None and n_last_steps < n
            else f"n_steps={n}"
        )
        print(f"  [{group_key}] {traj_label}: {steps_str}, "
              f"pred eflux mean={float(pred_eflux_t.mean()):.4g}, "
              f"gt eflux mean={float(gt_flux_t.mean()):.4g}{note_str}")

    return out


def evaluate_gyroswin(gyroswin_checkpoint, diff_dir, ae_checkpoint, *,
                      data_prep,
                      trajectories_id=TRAJECTORIES_ID,
                      trajectories_ood=TRAJECTORIES_OOD,
                      trajectories_test=TRAJECTORIES_TEST,
                      n_steps=128,
                      fluxavg_mean=92.6521, fluxavg_std=45.026,
                      output_path=None,
                      device=torch.device("cuda")):
    """Evaluate GyroSwin AR-rollout on ID + OOD (+ TEST if non-empty).
    Loads the checkpoint via the bundled `notebooks.neurips_gyroswin_eval`
    helper (same one gyroswin_generate.ipynb uses). Returns one entry per
    non-empty split keyed `gyroswin_<id|ood|test>`."""
    print("\n" + "=" * 60 + "\n  GyroSwin (AR rollout)\n" + "=" * 60)
    print(f"  GPU before: {_gpu_mem_str()}")
    splits = _resolve_splits(trajectories_id, trajectories_ood, trajectories_test)
    if not splits:
        raise ValueError("evaluate_gyroswin: at least one of "
                         "trajectories_id/ood/test must be non-empty.")
    def _run():
        # Local imports so non-gyroswin runs don't need this on the path.
        sys.path.insert(0, str(Path(__file__).parent))
        from neurips_gyroswin_eval import load_gyroswin_model

        # `pinned` only feeds the runner's validation dataset; pick any traj
        # we actually evaluate (first split, first traj).
        pinned = splits[0][1][0].replace("_ifft_realpotens", "") + ".h5"
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
            out = {f"gyroswin_{suffix}": {} for suffix, _ in splits}
            for suffix, trajs in splits:
                label = f"gyroswin_{suffix}"
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
    "eflux_RMSE_std":         "eflux_RMSE_std",
    "kxspec_RMSE":            "kxspec_RMSE",
    "kxspec_RMSE_std":        "kxspec_RMSE_std",
    "kxspec_WD":              "kxspec_WD",
    "kxspec_WD_std":          "kxspec_WD_std",
    "kyspec_RMSE":            "kyspec_RMSE",
    "kyspec_RMSE_std":        "kyspec_RMSE_std",
    "kyspec_WD":              "kyspec_WD",
    "kyspec_WD_std":          "kyspec_WD_std",
    "qspec_RMSE":             "fluxspec_RMSE",
    "qspec_RMSE_std":         "fluxspec_RMSE_std",
    # Probe row keys
    "probe_flux_ae_RMSE":      "eflux_RMSE",
    "probe_flux_ae_RMSE_std":  "eflux_RMSE_std",
    "probe_flux_gen_RMSE":     "eflux_RMSE",
    "probe_flux_gen_RMSE_std": "eflux_RMSE_std",
    "probe_kxspec_ae_RMSE":      "kxspec_RMSE",
    "probe_kxspec_ae_RMSE_std":  "kxspec_RMSE_std",
    "probe_kxspec_ae_WD":        "kxspec_WD",
    "probe_kxspec_ae_WD_std":    "kxspec_WD_std",
    "probe_kxspec_gen_RMSE":     "kxspec_RMSE",
    "probe_kxspec_gen_RMSE_std": "kxspec_RMSE_std",
    "probe_kxspec_gen_WD":       "kxspec_WD",
    "probe_kxspec_gen_WD_std":   "kxspec_WD_std",
    "probe_kyspec_ae_RMSE":      "kyspec_RMSE",
    "probe_kyspec_ae_RMSE_std":  "kyspec_RMSE_std",
    "probe_kyspec_ae_WD":        "kyspec_WD",
    "probe_kyspec_ae_WD_std":    "kyspec_WD_std",
    "probe_kyspec_gen_RMSE":     "kyspec_RMSE",
    "probe_kyspec_gen_RMSE_std": "kyspec_RMSE_std",
    "probe_kyspec_gen_WD":       "kyspec_WD",
    "probe_kyspec_gen_WD_std":   "kyspec_WD_std",
    "probe_fluxspec_ae_RMSE":      "fluxspec_RMSE",
    "probe_fluxspec_ae_RMSE_std":  "fluxspec_RMSE_std",
    "probe_fluxspec_gen_RMSE":     "fluxspec_RMSE",
    "probe_fluxspec_gen_RMSE_std": "fluxspec_RMSE_std",
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


def _coerce_table_row(model_label, split_label, raw_metrics, time_per_sample_s=None):
    row = {
        "model": model_label,
        "split": split_label,
        **{_RENAME_TO_COL[k]: v
           for k, v in raw_metrics.items()
           if k in _RENAME_TO_COL},
    }
    if time_per_sample_s is not None:
        row["time_per_sample_s"] = time_per_sample_s
    return row


def _mean_time_per_sample(group, *, include_probe=False):
    """Average per-sample wall-clock time across the trajectories in `group`.

    Each gen dict carries `_n_samples` (samples drawn) and timing fields:
      * `_gen_time_s` — total generation wall time the row should report
        (Diff 5D: sample+decode; VAE/VQVAE/AR: sample+decode in one call;
        GyroSwin: whole rollout, n_samples=1).
      * `_sample_time_s` (Diff only) — denoising-only time, no decode.
      * `_decode_time_s` (Diff only) — decoder-only time.
      * `_probe_time_s`  (Diff only) — probe predict time over that traj's
        samples.

    With `include_probe=True` the probe row reports `sample + probe` per
    sample -- decode is intentionally excluded because the probes operate on
    latents and would otherwise overcount the decoder. Falls back to
    `_gen_time_s + _probe_time_s` only when `_sample_time_s` isn't recorded
    (older caches). Returns NaN for groups that have no timing fields.
    """
    per_traj = []
    for res in group.values():
        gen = res.get("gen", {})
        n = gen.get("_n_samples")
        if not n:
            continue
        if include_probe:
            sample_t = gen.get("_sample_time_s", gen.get("_gen_time_s"))
            probe_t  = gen.get("_probe_time_s", 0.0)
            if sample_t is None:
                continue
            per_sample = (float(sample_t) + float(probe_t)) / float(n)
        else:
            t = gen.get("_gen_time_s")
            if t is None:
                continue
            per_sample = float(t) / float(n)
        per_traj.append(per_sample)
    if not per_traj:
        return float("nan")
    return float(np.mean(per_traj))


def build_summary_table(all_results, *, with_diff_probes=True):
    """Single DataFrame with rows for each present model & split.

    Recognised group keys in `all_results`: `<model>_<split>` where
    `<split>` ∈ {id, ood, test} (per `SUMMARY_SPLITS`). Per-model bases:
      vae, vqvae, ar, diff (+ probe rows when probe_* keys present),
      gyroswin (old monkey-patched checkpoint),
      gyroswin_xxl (cached AR rollouts), and any number of
      gyroswin_<variant> groups from evaluate_gyroswin_new(..., variant=...).
    Splits not present in `all_results` are silently skipped.

    Columns: eflux_RMSE, kxspec_RMSE/WD, kyspec_RMSE/WD, fluxspec_RMSE,
    plus `*_std` siblings and `time_per_sample_s`.
    """
    rows = []
    splits = SUMMARY_SPLITS  # (label, suffix, _default_traj_list)

    for model_label, base in (
        ("VAE",      "vae"),
        ("VQ-VAE",   "vqvae"),
        ("AR",       "ar"),
        ("Diff 5D",  "diff"),
    ):
        for split_label, suffix, _ in splits:
            g = all_results.get(f"{base}_{suffix}")
            if not g:
                continue
            rows.append(_coerce_table_row(
                model_label, split_label, _row_metrics_5d(g),
                time_per_sample_s=_mean_time_per_sample(g),
            ))

    if with_diff_probes:
        for variant in ("ae", "gen"):
            for split_label, suffix, _ in splits:
                g = all_results.get(f"diff_{suffix}")
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
                    f"Diff probe ({variant})", split_label,
                    _row_metrics_probe(g, variant),
                    time_per_sample_s=_mean_time_per_sample(g, include_probe=True),
                ))

    for split_label, suffix, _ in splits:
        g = all_results.get(f"gyroswin_{suffix}")
        if not g:
            continue
        rows.append(_coerce_table_row(
            "GyroSwin", split_label, _row_metrics_5d(g),
            time_per_sample_s=_mean_time_per_sample(g),
        ))

    for split_label, suffix, _ in splits:
        g = all_results.get(f"gyroswin_xxl_{suffix}")
        if not g:
            continue
        rows.append(_coerce_table_row(
            "GyroSwin XXL", split_label, _row_metrics_5d(g),
            time_per_sample_s=_mean_time_per_sample(g),
        ))

    # Auto-detect any number of `gyroswin_<variant>_<split>` groups (set by
    # `evaluate_gyroswin_new(..., variant=...)`). Each becomes a
    # `GyroSwin (<variant>)` row. `gyroswin_<split>` (plain "GyroSwin") and
    # `gyroswin_xxl_<split>` are handled above and excluded.
    _RESERVED_GS_VARIANTS = {"xxl"}
    _split_suffixes = [suffix for _label, suffix, _ in splits]
    _gs_variants = set()
    for key in all_results:
        if not key.startswith("gyroswin_"):
            continue
        for suffix in _split_suffixes:
            if key.endswith(f"_{suffix}"):
                tail = key[len("gyroswin_"):-(len(suffix) + 1)]
                if tail and tail not in _RESERVED_GS_VARIANTS:
                    _gs_variants.add(tail)
                break
    for variant in sorted(_gs_variants):
        for split_label, suffix, _ in splits:
            g = all_results.get(f"gyroswin_{variant}_{suffix}")
            if not g:
                continue
            rows.append(_coerce_table_row(
                f"GyroSwin ({variant})", split_label, _row_metrics_5d(g),
                time_per_sample_s=_mean_time_per_sample(g),
            ))

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index(["model", "split"])
    # Mean and std interleaved so each metric reads "value (± spread across
    # trajectories)" left-to-right. Std is sample std (ddof=1) of per-traj RMSE.
    cols = [
        "eflux_RMSE",    "eflux_RMSE_std",
        "kxspec_RMSE",   "kxspec_RMSE_std",   "kxspec_WD",   "kxspec_WD_std",
        "kyspec_RMSE",   "kyspec_RMSE_std",   "kyspec_WD",   "kyspec_WD_std",
        "fluxspec_RMSE", "fluxspec_RMSE_std",
        "time_per_sample_s",
    ]
    return df[[c for c in cols if c in df.columns]]

