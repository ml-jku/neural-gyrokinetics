"""Standalone generative evaluation for PINC VAE and VQ-VAE autoencoders.

Evaluates generative capabilities by sampling from the prior:
  - VAE: sample z ~ N(0, I), decode to 5D distribution function.
  - VQ-VAE: sample random codebook indices, decode to 5D distribution function.

Then compute physics integrals for flux and spectra on the decoded fields.
Outputs averaged fluxes (particle, heat) and spectra (ky, kx, heat flux spectrum)
per trajectory.

Usage:
    python -m neugk.pinc.generate --ckpt /path/to/checkpoint
    python -m neugk.pinc.generate --ckpt /path/to/checkpoint --include_ood --n_samples 16
    python -m neugk.pinc.generate --ckpt /path/to/checkpoint --trajectories iteration_13.h5
"""

import os
import argparse
from math import prod
from collections import defaultdict
import pickle
import yaml
import torch
from torch.utils._pytree import tree_map
import numpy as np
from tqdm import tqdm

from neugk.pinc.autoencoders.ae_utils import load_autoencoder
from neugk.pinc.autoencoders.gk_autoencoders import Swin5DVAE, Swin5DVQVAE
from neugk.integrals import FluxIntegral
from neugk.utils import recombine_zf
from neugk.plot_utils import plot_nd


KEY_MAP = {
    "itg": "ion_temp_grad",
    "dg": "density_grad",
}

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generative evaluation for PINC VAE / VQ-VAE autoencoders."
    )
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint directory or .pth file.")
    parser.add_argument("--config", default="configs/pinc_inference.yaml", help="Inference config YAML.")
    parser.add_argument("--trajectories", nargs="+", default=None, help="Override trajectory list.")
    parser.add_argument("--n_samples", type=int, default=None, help="Override number of samples.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--norm_stats", default=None, help="Path to dataset norm_stats .pkl file (RunningMeanStd dict).")
    parser.add_argument("--output", default=None, help="Path to save results .pt file (default: <ckpt_dir>/generative_eval.pt).")
    parser.add_argument(
        "--vq_index_pkl",
        default=None,
        help="Path to precomputed training VQ indices pkl "
        "(e.g. diff_train_indices_*_vqvae<id>.pkl). If provided, VQ-VAE generation "
        "samples indices from the global empirical codebook distribution instead of "
        "a uniform prior.",
    )
    return parser.parse_args()


def load_inference_config(path, cli_args):
    """Load YAML config and apply CLI overrides."""
    with open(path) as f:
        cfg = yaml.safe_load(f)

    if cli_args.trajectories is not None:
        cfg["trajectories"] = cli_args.trajectories
    if cli_args.n_samples is not None:
        cfg["n_samples"] = cli_args.n_samples
    if cli_args.batch_size is not None:
        cfg["batch_size"] = cli_args.batch_size
    if cli_args.device is not None:
        cfg["device"] = cli_args.device
    return cfg


def get_conditioning(conditions, metadata, device):
    """Build a conditioning vector for a trajectory from its metadata."""
    cond_list = []
    for k in conditions:
        val = metadata[KEY_MAP.get(k, k)]
        if hasattr(val, "__len__"):
            val = val.squeeze()
        cond_list.append(torch.tensor(val, dtype=torch.float32))
    return torch.stack(cond_list, dim=-1).to(device)


def get_geometry(metadata, dtype=torch.float64):
    """Get geometry dict for a single trajectory file."""
    geometry = metadata["geometry"]
    return tree_map(lambda g: torch.as_tensor(g).to(dtype=dtype), geometry)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_vae_prior(model, batch_size, condition, pad_axes, device, mean=0.0, std=1.0):
    """Sample z ~ N(0, I) and decode."""
    grid_size = model.bottleneck_grid_size
    latent_dim = model.bottleneck_dim
    z = torch.randn(batch_size, *grid_size, latent_dim, device=device) * std + mean
    cond = condition.unsqueeze(0).expand(batch_size, -1).to(device) if condition is not None else None
    return model.decode(z, pad_axes, condition=cond)["df"]


def compute_codebook_prior(index_pkl_path, codebook_size):
    """Build a global empirical categorical over codebook indices from a training pkl.

    The pkl is the dict produced by CycloneDiff.precompute_latents:
      {(f_idx, t_idx): {"x": int64 ndarray of shape (seq_len,), ...}}.
    Returns a 1D float tensor of length codebook_size summing to 1.
    """
    with open(index_pkl_path, "rb") as f:
        latents = pickle.load(f)
    all_idx = np.concatenate([v["x"].ravel() for v in latents.values()])
    counts = np.bincount(all_idx, minlength=codebook_size).astype(np.float64)
    if counts.sum() == 0:
        raise ValueError(f"No indices found in {index_pkl_path}")
    probs = counts / counts.sum()
    return torch.from_numpy(probs).float()


@torch.no_grad()
def sample_vqvae_random(model, batch_size, condition, pad_axes, device, prior=None):
    """Sample codebook indices and decode.

    If `prior` is None, indices are drawn uniformly. If `prior` is a 1D tensor of
    length codebook_size, indices are drawn i.i.d. per position from that
    categorical (global empirical codebook usage).
    """
    grid_size = model.bottleneck_grid_size
    num_tokens = prod(grid_size)
    embedding_dim = model.middle_vq_downproj.out_features

    if prior is None:
        indices = torch.randint(
            0, model.vq.codebook_size, (batch_size, num_tokens), device=device,
        )
    else:
        indices = torch.multinomial(
            prior.to(device), batch_size * num_tokens, replacement=True,
        ).view(batch_size, num_tokens)
    codes = model.vq.get_codes_from_indices(indices)
    z = codes.view(batch_size, *grid_size, embedding_dim)
    cond = condition.unsqueeze(0).expand(batch_size, -1) if condition is not None else None
    return model.decode(z, pad_axes, condition=cond)["df"]


# ---------------------------------------------------------------------------
# Physics integrals and spectra
# ---------------------------------------------------------------------------

def compute_spectra(phi_fft, eflux_field):
    """Compute kx, ky power spectra and heat-flux spectrum from integrated fields."""
    kxspec = torch.sum(torch.abs(phi_fft) ** 2, dim=(1, 3))
    kyspec = torch.sum(torch.abs(phi_fft) ** 2, dim=(1, 2))

    qspec = eflux_field.sum(
        (1, 2, 3, 4) if eflux_field.dim() == 6 else (0, 1, 2, 3)
    )
    if qspec.dim() == 1:
        qspec = qspec.unsqueeze(0)

    return {"kxspec": kxspec, "kyspec": kyspec, "qspec": qspec}


@torch.no_grad()
def compute_physics(df, geometry, separate_zf, integrator):
    """Compute fluxes and spectra for a decoded distribution function."""
    df = df.float()
    if separate_zf and df.shape[1] > 2:
        df = recombine_zf(df, dim=1)

    phi_spec, (pflux, eflux_field, vflux) = integrator(geometry, df)
    spectra = compute_spectra(phi_spec, eflux_field.squeeze())

    agg_axes = tuple(i for i in range(1, len(pflux.shape)))
    return {
        "pflux": pflux.sum(agg_axes),
        "eflux": eflux_field.sum(agg_axes),
        "vflux": vflux.sum(agg_axes),
        **spectra,
    }

def denormalize(df, norm_stats):
    """Denormalize a decoded distribution function using trajectory statistics."""
    mean = torch.tensor(norm_stats["df_mean"], device=df.device)
    std = torch.tensor(norm_stats["df_std"], device=df.device)
    return df * std + mean

# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate_generative(model, ckpt_dir, cfg, inf_cfg, metadata, norm_stats, device, integrator, vq_prior=None, *, timing_out=None):
    """Generate samples for a single trajectory and return averaged physics results.

    Samples are generated from the prior (VAE) or random codebook indices (VQ-VAE),
    decoded, denormalized using the trajectory's statistics, and then physics
    integrals (flux, spectra) are computed.

    If `timing_out` is a dict, the cumulative wall time of the sample+decode
    block (excluding physics integrals) is added under "gen_time_s" and the
    total sample count under "n_samples".
    """
    import time as _time
    is_vae = isinstance(model, Swin5DVAE)
    is_vqvae = isinstance(model, Swin5DVQVAE)
    if not (is_vae or is_vqvae):
        raise ValueError(
            f"Generative evaluation requires a VAE or VQ-VAE model, "
            f"got {type(model).__name__}"
        )

    n_samples = inf_cfg.get("n_samples", 8)
    batch_size = inf_cfg.get("batch_size", 8)
    use_amp = inf_cfg.get("use_amp", False)
    separate_zf = cfg.dataset.separate_zf
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    model.eval()
    pad_axes = model.get_pad_axes(model.base_resolution)
    # Get conditioning and geometry for this trajectory
    model_key = "autoencoder" if hasattr(cfg, "autoencoder") else "model"
    model_cfg = getattr(cfg, model_key)
    decoder_conds = sorted(set(model_cfg.decoder_conditioning) | set(getattr(model_cfg, "encoder_conditioning", [])))

    condition = (
        get_conditioning(decoder_conds, metadata, device)
        if decoder_conds
        else None
    )
    geometry = get_geometry(metadata)
    # Generate in batches
    accum = defaultdict(list)
    remaining = n_samples

    while remaining > 0:
        bs = min(batch_size, remaining)

        if timing_out is not None and torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.synchronize()
        _t0 = _time.perf_counter()

        with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp and device.type == "cuda"):
            if is_vae:
                mean = 0.0
                std = 1.0
                stats_path = os.path.join(ckpt_dir, "train_latent_stats.pt")
                old_stats_path = os.path.join(ckpt_dir, "train_latent_stats.pkl")
                if os.path.exists(stats_path):
                    lat_stats = torch.load(stats_path, weights_only=True)
                    mean = lat_stats["z_mean"].to(device)
                    std = lat_stats["z_std"].to(device)
                elif os.path.exists(old_stats_path):
                    train_lats = pickle.load(open(old_stats_path, "rb"))
                    train_lats = np.concatenate([train_lats[k]["x"] for k in train_lats.keys()], axis=0)
                    mean = torch.tensor(np.mean(train_lats, axis=0), device=device)
                    std = torch.tensor(np.std(train_lats, axis=0), device=device)
                gen_df = sample_vae_prior(model, bs, condition, pad_axes, device, mean=mean, std=std)
            else:
                gen_df = sample_vqvae_random(model, bs, condition, pad_axes, device, prior=vq_prior)

        gen_df = gen_df.float()

        # Denormalize each sample using the trajectory's statistics
        denorm_dfs = []
        for b in range(bs):
            denorm_dfs.append(denormalize(df=gen_df[b], norm_stats=norm_stats))
        gen_df_denorm = torch.stack(denorm_dfs)

        if timing_out is not None:
            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.synchronize()
            timing_out["gen_time_s"] = timing_out.get("gen_time_s", 0.0) + (_time.perf_counter() - _t0)
            timing_out["n_samples"]  = timing_out.get("n_samples", 0) + bs

        # Add batch dim and expand to match batch size; keep on CPU because
        # integrators use float64 Bessel functions that need NVRTC on CUDA.
        geom_batch = tree_map(
            lambda g: g.unsqueeze(0).expand(bs, *g.shape),
            geometry,
        )
        physics = compute_physics(
            gen_df_denorm.cpu(), geom_batch, separate_zf, integrator,
        )
        for k, v in physics.items():
            accum[k].append(v.cpu())

        remaining -= bs

    # Aggregate over all samples
    results = {}
    for k, tensors in accum.items():
        stacked = torch.cat(tensors, dim=0)
        results[k] = {
            "mean": stacked.mean(dim=0),
            "std": stacked.std(dim=0),
            "all": stacked,
        }
    return results

def evaluate_ground_truth(metadata, trajectory, train_cfg, inf_cfg, device, integrator):
    """Compute physics integrals for the ground truth distribution function."""
    geometry = get_geometry(metadata)
    gt_path = os.path.join(inf_cfg["root"], trajectory, "data")
    timesteps = [f for f in sorted(os.listdir(gt_path)) if f.startswith("timestep")]
    timesteps = timesteps[train_cfg.dataset.offset:]

    df_batch = []
    for ts in tqdm(timesteps, desc=f"Loading GT for {trajectory}", unit="timestep"):
        df = np.fromfile(os.path.join(gt_path, ts), dtype=np.float32)
        df = torch.from_numpy(df).reshape((2,32,8,16,85,32)).unsqueeze(0)
        df_batch.append(df)

    # GT physics runs on CPU (integrator uses float64 Bessel via NVRTC), so
    # no need to bounce through the GPU — doing so OOMs when another process
    # is using the device.
    df_batch = torch.stack(df_batch, dim=0).float()
    bs = df_batch.shape[0]
    geom_batch = tree_map(
        lambda g: g.unsqueeze(0).expand(bs, *g.shape),
        geometry,
    )

    physics = compute_physics(
        df_batch, geom_batch, separate_zf=False, integrator=integrator,
    )
    results = {}
    for k, tensors in physics.items():
        results[k] = {
            "mean": tensors.mean(dim=0),
            "std": tensors.std(dim=0),
            "all": tensors,
        }
    return results

def print_comparison(all_results):
    """Compare generated vs ground-truth results aggregated across all trajectories."""
    print(f"\n{'='*60}")
    print(f"  Aggregate metrics across {len(all_results)} trajectories")
    print(f"{'='*60}")

    # Collect per-sample metrics across all trajectories
    all_eflux_rmse = []
    all_eflux_r2 = []
    all_spec_rmse = {"kxspec": [], "kyspec": []}

    for res in all_results.values():
        gen, gt = res["gen"], res["gt"]
        gt_eflux_mean = gt["eflux"]["mean"]
        gt_eflux_all = gt["eflux"]["all"]
        gen_eflux = gen["eflux"]["all"]

        # Per-sample RMSE (generated vs GT mean)
        per_sample_se = (gen_eflux - gt_eflux_mean.unsqueeze(0)).pow(2)
        if per_sample_se.dim() > 1:
            per_sample_rmse = per_sample_se.mean(dim=tuple(range(1, per_sample_se.dim()))).sqrt()
        else:
            per_sample_rmse = per_sample_se.sqrt()
        all_eflux_rmse.append(per_sample_rmse)

        # R2: use GT timestep variance as SS_tot, per-sample residuals vs GT mean
        ss_tot = (gt_eflux_all - gt_eflux_all.mean()).pow(2).sum()
        if ss_tot > 0:
            per_sample_ss_res = (gen_eflux - gt_eflux_mean.unsqueeze(0)).pow(2)
            if per_sample_ss_res.dim() > 1:
                per_sample_ss_res = per_sample_ss_res.sum(dim=tuple(range(1, per_sample_ss_res.dim())))
            all_eflux_r2.append(1 - per_sample_ss_res / ss_tot)

        # Per-sample spectral RMSE
        for spec_name in ["kxspec", "kyspec"]:
            if spec_name in gen and spec_name in gt:
                gt_spec = gt[spec_name]["mean"]
                gen_spec_all = gen[spec_name]["all"]
                per_sample_spec_se = (gen_spec_all - gt_spec.unsqueeze(0)).pow(2)
                per_sample_spec_rmse = per_sample_spec_se.mean(dim=tuple(range(1, per_sample_spec_se.dim()))).sqrt()
                all_spec_rmse[spec_name].append(per_sample_spec_rmse)

    # Aggregate: concatenate all per-sample values, compute mean +/- SE
    all_eflux_rmse = torch.cat(all_eflux_rmse)
    n = len(all_eflux_rmse)
    eflux_rmse = all_eflux_rmse.mean().item()
    eflux_se = (all_eflux_rmse.std() / (n ** 0.5)).item()
    print(f"  eflux RMSE: {eflux_rmse:.6g} +/- {eflux_se:.6g} (SE, n={n})")

    if all_eflux_r2:
        all_eflux_r2 = torch.cat(all_eflux_r2)
        r2 = all_eflux_r2.mean().item()
        r2_se = (all_eflux_r2.std() / (len(all_eflux_r2) ** 0.5)).item()
    else:
        r2, r2_se = float("nan"), float("nan")
    print(f"  eflux R2:   {r2:.6g} +/- {r2_se:.6g} (SE)")

    spec_metrics = {}
    for spec_name in ["kxspec", "kyspec"]:
        if all_spec_rmse[spec_name]:
            vals = torch.cat(all_spec_rmse[spec_name])
            m = vals.mean().item()
            se = (vals.std() / (len(vals) ** 0.5)).item()
            spec_metrics[spec_name] = (m, se)
            print(f"  {spec_name} RMSE: {m:.6g} +/- {se:.6g} (SE)")

    # LaTeX table row
    kx_rmse, kx_se = spec_metrics.get("kxspec", (float("nan"), float("nan")))
    ky_rmse, ky_se = spec_metrics.get("kyspec", (float("nan"), float("nan")))
    latex = (
        f"  LaTeX: VAE prior "
        f"& ${eflux_rmse:.4g} \\pm {eflux_se:.4g}$ "
        f"& ${r2:.4g} \\pm {r2_se:.4g}$ "
        f"& ${kx_rmse:.4g} \\pm {kx_se:.4g}$ "
        f"& ${ky_rmse:.4g} \\pm {ky_se:.4g}$ \\\\"
    )
    print(latex)
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    inf_cfg = load_inference_config(args.config, args)

    device = torch.device(inf_cfg.get("device", "cuda"))
    ckpt_dir = args.ckpt if os.path.isdir(args.ckpt) else os.path.dirname(args.ckpt)

    # Load model
    print(f"Loading model from {args.ckpt} ...")
    model, _, train_cfg = load_autoencoder(args.ckpt, device)
    model = model.to(device)
    model.eval()

    # Load normalization stats
    if args.norm_stats:
        stats_path = args.norm_stats
    else:
        # Auto-discover: look for a *_stats.pkl in the dataset root that starts with "df_"
        data_root = train_cfg.dataset.path
        candidates = sorted(
            f for f in os.listdir(data_root)
            if f.startswith("df_") and f.endswith("_stats.pkl")
        )
        if not candidates:
            raise FileNotFoundError(
                f"No df_*_stats.pkl found in {data_root}. "
                "Pass --norm_stats /path/to/stats.pkl explicitly."
            )
        stats_path = os.path.join(data_root, candidates[0])
        if len(candidates) > 1:
            print(f"Warning: multiple stats files found, using {candidates[0]}")

    print(f"Loading normalization stats from {stats_path} ...")
    raw_stats = pickle.load(open(stats_path, "rb"))
    norm_axes = tuple(train_cfg.dataset.normalization.df.agg_axes)
    mean, var, *_ = raw_stats["df"].aggregate_stats(
        raw_stats["df"].mean, raw_stats["df"].var, raw_stats["df"].min, raw_stats["df"].max, agg_axes=norm_axes
    )
    std = np.sqrt(var)
    stats = {
        "df_mean": mean,
        "df_std": std,
    }

    model_type = type(model).__name__
    print(f"Model type: {model_type}")
    if hasattr(model, "get_compression_info"):
        info = model.get_compression_info()
        print(f"Compression info: {info}")

    # Build empirical codebook prior for VQ-VAE, if requested
    vq_prior = None
    if args.vq_index_pkl is not None:
        if not isinstance(model, Swin5DVQVAE):
            print(f"Warning: --vq_index_pkl given but model is {model_type}; ignoring.")
        else:
            print(f"Loading VQ index pkl from {args.vq_index_pkl} ...")
            vq_prior = compute_codebook_prior(args.vq_index_pkl, model.vq.codebook_size)
            nz = int((vq_prior > 0).sum().item())
            print(f"Empirical codebook prior: {nz}/{model.vq.codebook_size} codes used")

    # Determine trajectories
    trajectories = list(inf_cfg["trajectories"])
    output_path = args.output if args.output else os.path.join(ckpt_dir, "generative_eval.pt")

    if os.path.exists(output_path):
        print(f"Loading cached results from {output_path}")
        all_results = torch.load(output_path, weights_only=False)
    else:
        all_results = {}
        integrator = FluxIntegral(flux_fields=True, spectral_df=False, spectral_potens=True)
        for traj in trajectories:
            meta = pickle.load(open(os.path.join(inf_cfg["root"], traj, "metadata.pkl"), "rb"))
            results = evaluate_generative(model, ckpt_dir, train_cfg, inf_cfg, meta, stats, device, integrator, vq_prior=vq_prior)
            gt_res = evaluate_ground_truth(meta, traj, train_cfg, inf_cfg, device, integrator)
            all_results[traj] = {"gen": results, "gt": gt_res}

        torch.save(all_results, output_path)
        print(f"Results saved to {output_path}")

    print_comparison(all_results)


if __name__ == "__main__":
    main()
