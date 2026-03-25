"""Generate samples from a trained diffusion checkpoint and save them as H5 files."""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import random
import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from diffusers import DDPMScheduler
from omegaconf import OmegaConf
from torch.distributions import Laplace, Normal, StudentT
from torch.utils.data import DataLoader

from neugk.dataset import CycloneAEDataset, CycloneVAEDataset
from neugk.dataset.backend import H5Backend, KvikIOBackend
from neugk.diffusion.models import get_diffusion_model
from neugk.integrals import get_integrals
from neugk.pinc.autoencoders.ae_utils import load_autoencoder
from neugk.utils import RunningMeanStd


@dataclass
class GenerationTarget:
    name: str
    condition: torch.Tensor
    denorm_file_index: Optional[int]
    geometry: Optional[Dict[str, np.ndarray]]
    source_metadata: Optional[Dict[str, Any]] = None


def _to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    return bool(value)


def _basename_without_suffix(path: str) -> str:
    base = os.path.basename(path)
    for suffix in ["_ifft_realpotens.h5", "_ifft.h5", ".h5"]:
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return base


def _get_backend(name: str):
    backend_name = str(name).lower()
    if backend_name == "h5":
        return H5Backend(rank=0)
    if backend_name == "gds":
        # For generation we keep GDS disabled for stability/portability.
        return KvikIOBackend(rank=0, use_kvikio=False)
    raise ValueError(f"Unsupported backend: {name}")


def _resolve_checkpoint_path(ckpt_dir: str, snapshot_name: str) -> str:
    ckpt_path = os.path.join(ckpt_dir, snapshot_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint snapshot not found: {ckpt_path}")
    return ckpt_path


def _maybe_to_numpy_dict(geometry: Optional[Dict[str, Any]]) -> Optional[Dict[str, np.ndarray]]:
    if geometry is None:
        return None
    out: Dict[str, np.ndarray] = {}
    for k, v in geometry.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().cpu().numpy()
        else:
            out[k] = np.array(v)
    return out


def _extract_condition_from_meta(meta: Dict[str, Any], cond_keys: Sequence[str]) -> torch.Tensor:
    alias_map = {
        "itg": ["itg", "ion_temp_grad", "ion_temperature_gradient"],
        "dg": ["dg", "density_grad", "density_gradient"],
        "s_hat": ["s_hat", "shat"],
        "q": ["q", "safety_factor"],
    }

    vals: List[float] = []
    for cond_key in cond_keys:
        candidates = alias_map.get(cond_key, [cond_key])
        found_key = next((k for k in candidates if k in meta), None)
        if found_key is None:
            raise KeyError(
                f"Missing conditioning key '{cond_key}' in metadata. "
                f"Tried aliases {candidates}. Available keys: {sorted(list(meta.keys()))[:40]}"
            )
        vals.append(float(np.squeeze(meta[found_key])))

    return torch.tensor(vals, dtype=torch.float32)


def _canonicalize_generated_df(df: torch.Tensor) -> torch.Tensor:
    # Some models include a singleton sequence axis for bundle_seq_length=1.
    if df.ndim >= 3 and df.shape[1] == 1 and df.shape[2] in (2, 4):
        df = df.squeeze(1)
    return df


def _recombine_separate_zf(df: torch.Tensor, separate_zf: bool) -> torch.Tensor:
    if separate_zf and df.ndim >= 2 and df.shape[1] == 4:
        return df[:, [0, 1]] + df[:, [2, 3]]
    return df


def _build_sampler(model, autoencoder, cfg, device: torch.device, latent_scale: float):
    diff_cfg = cfg.model.diffusion
    formulation = str(diff_cfg.get("formulation", "ddpm")).lower()

    if "ddpm" in formulation:
        sched_cfg = diff_cfg.get("scheduler", cfg.model.get("scheduler", {}))
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=sched_cfg.get("num_train_timesteps", 1000),
            beta_start=sched_cfg.get("beta_start", 0.0001),
            beta_end=sched_cfg.get("beta_end", 0.02),
            beta_schedule=sched_cfg.get("beta_schedule", "linear"),
            prediction_type=sched_cfg.get("prediction_type", "epsilon"),
        )

        @torch.no_grad()
        def _sample(condition: torch.Tensor, steps: Optional[int] = None):
            model.eval()
            autoencoder.eval()
            bs = condition.shape[0]
            n_steps = int(steps or noise_scheduler.config.num_train_timesteps)
            latents = torch.randn((bs, *model.latent_shape), device=device)
            noise_scheduler.set_timesteps(n_steps)

            for t in noise_scheduler.timesteps:
                t_batch = torch.full((bs,), t, device=device, dtype=torch.long)
                pred = model(latents, tstep=t_batch, condition=condition)
                latents = noise_scheduler.step(pred, t, latents).prev_sample

            decoded = autoencoder.decode(latents / latent_scale, condition=condition)
            model.train()
            return decoded

        return _sample

    if "flow" in formulation:
        noise_dist = str(diff_cfg.get("noise_distribution", "gaussian")).lower()
        distr_gauss = Normal(
            torch.tensor(0.0, device=device),
            torch.tensor(1.0, device=device),
        )
        nu = float(getattr(cfg.model, "nu", 5.0))
        laplace_scale = float(getattr(cfg.model, "laplace_scale", 0.1))
        mix_ratio = float(getattr(cfg.model, "mix_ratio", 0.5))
        distr_student = StudentT(torch.tensor(nu, device=device))
        distr_laplace = Laplace(
            torch.tensor(0.0, device=device),
            torch.tensor(laplace_scale, device=device),
        )
        n_train_steps = int(diff_cfg.scheduler.get("num_train_timesteps", 1000))
        continuous_time = bool(diff_cfg.get("continuous_time", True))

        def _prior(shape):
            if noise_dist == "gaussian":
                return distr_gauss.sample(shape)
            if noise_dist == "mixture":
                laplace_samples = distr_laplace.sample(shape)
                student_samples = distr_student.sample(shape)
                mask = torch.bernoulli(torch.full(shape, mix_ratio, device=device))
                return torch.where(mask.bool(), laplace_samples, student_samples)
            raise ValueError(
                f"Unsupported flow prior noise_distribution '{noise_dist}'. "
                "Expected 'gaussian' or 'mixture'."
            )

        @torch.no_grad()
        def _sample(condition: torch.Tensor, steps: Optional[int] = None):
            model.eval()
            autoencoder.eval()
            bs = condition.shape[0]
            n_steps = int(steps or 50)
            x = _prior((bs, *model.latent_shape)).to(device)
            t_steps = torch.linspace(0.0, 1.0, n_steps + 1, device=device)

            for i in range(n_steps):
                t_curr = t_steps[i]
                t_next = t_steps[i + 1]
                dt = t_next - t_curr

                if continuous_time:
                    t_batch = torch.full((bs,), t_curr.item(), device=device)
                else:
                    t_idx = int(t_curr.item() * (n_train_steps - 1))
                    t_batch = torch.full((bs,), t_idx, device=device, dtype=torch.long)

                v_pred = model(x, tstep=t_batch, condition=condition)
                x = x + v_pred * dt

            decoded = autoencoder.decode(x / latent_scale, condition=condition)
            model.train()
            return decoded

        return _sample

    if "edm" in formulation or "karras" in formulation:
        sigma_data = float(diff_cfg.get("sigma_data", 1.0))
        sigma_min = float(diff_cfg.get("sigma_min", 0.002))
        sigma_max = float(diff_cfg.get("sigma_max", 80.0))
        rho = float(diff_cfg.get("rho", 7.0))

        def _preconditioned_forward(x: torch.Tensor, sigma: torch.Tensor, condition):
            expand_dims = [-1] + [1] * (x.ndim - 1)
            c_skip = (sigma_data**2) / (sigma**2 + sigma_data**2)
            c_out = sigma * sigma_data / (sigma**2 + sigma_data**2).sqrt()
            c_in = 1.0 / (sigma**2 + sigma_data**2).sqrt()
            c_noise = 0.25 * sigma.log()
            c_skip = c_skip.view(*expand_dims)
            c_out = c_out.view(*expand_dims)
            c_in = c_in.view(*expand_dims)
            f_theta = model(x * c_in, tstep=c_noise, condition=condition)
            return c_skip * x + c_out * f_theta

        @torch.no_grad()
        def _sample(condition: torch.Tensor, steps: Optional[int] = None):
            model.eval()
            autoencoder.eval()
            bs = condition.shape[0]
            n_steps = int(steps or 5)

            step_indices = torch.arange(n_steps, dtype=torch.float32, device=device)
            sigma_max_rho = sigma_max ** (1 / rho)
            sigma_min_rho = sigma_min ** (1 / rho)
            sigmas = (
                sigma_max_rho
                + step_indices / (n_steps - 1) * (sigma_min_rho - sigma_max_rho)
            ) ** rho
            sigmas = torch.cat([sigmas, torch.zeros_like(sigmas[:1])])

            x = torch.randn((bs, *model.latent_shape), device=device) * sigma_max
            for i in range(len(sigmas) - 1):
                sigma_hat = sigmas[i]
                sigma_next = sigmas[i + 1]
                sigma_batch = torch.full((bs,), sigma_hat, device=device)
                denoised = _preconditioned_forward(x, sigma_batch, condition)
                d_i = (x - denoised) / sigma_hat
                x_next = x + d_i * (sigma_next - sigma_hat)
                if sigma_next != 0:
                    sigma_next_batch = torch.full((bs,), sigma_next, device=device)
                    denoised_next = _preconditioned_forward(
                        x_next, sigma_next_batch, condition
                    )
                    d_prime = (x_next - denoised_next) / sigma_next
                    x_next = x + 0.5 * (d_i + d_prime) * (sigma_next - sigma_hat)
                x = x_next

            decoded = autoencoder.decode(x / latent_scale, condition=condition)
            model.train()
            return decoded

        return _sample

    if "jit" in formulation:
        sched_cfg = diff_cfg.get("scheduler", cfg.model.get("scheduler", {}))
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=sched_cfg.get("num_train_timesteps", 1000),
            beta_start=sched_cfg.get("beta_start", 0.0001),
            beta_end=sched_cfg.get("beta_end", 0.02),
            beta_schedule=sched_cfg.get("beta_schedule", "linear"),
            prediction_type=sched_cfg.get("prediction_type", "epsilon"),
        )
        n_train_steps = int(noise_scheduler.config.num_train_timesteps)

        @torch.no_grad()
        def _sample(condition: torch.Tensor, steps: Optional[int] = None):
            model.eval()
            autoencoder.eval()
            bs = condition.shape[0]
            n_steps = int(steps or 1)
            xt = torch.randn((bs, *model.latent_shape), device=device)
            t_indices = torch.linspace(n_train_steps - 1, 0, n_steps)

            for i in range(n_steps):
                t_val = int(t_indices[i])
                t_batch = torch.full((bs,), t_val, device=device, dtype=torch.long)
                x0_pred = model(xt, tstep=t_batch, condition=condition)
                if i < n_steps - 1:
                    t_next = int(t_indices[i + 1])
                    noise = torch.randn_like(x0_pred)
                    xt = noise_scheduler.add_noise(
                        x0_pred, noise, torch.tensor([t_next], device=device)
                    )

            decoded = autoencoder.decode(x0_pred / latent_scale, condition=condition)
            model.train()
            return decoded

        return _sample

    raise ValueError(f"Unsupported diffusion formulation: {formulation}")


def _make_train_dataset(
    diff_cfg,
    backend_name: str,
    data_path: str,
    normalization_stats: Optional[Dict[str, Any]] = None,
    dataset_class=CycloneAEDataset,
    latent_sampling_mode: Optional[str] = None,
):
    backend = _get_backend(backend_name)
    kwargs = dict(
        backend=backend,
        conditions=list(diff_cfg.model.conditioning),
        active_keys=list(diff_cfg.dataset.active_keys),
        fields_to_load=["df"],
        path=data_path,
        split="train",
        trajectories=diff_cfg.dataset.training_trajectories,
        normalization=OmegaConf.to_container(diff_cfg.dataset.normalization),
        normalization_scope=diff_cfg.dataset.normalization_scope,
        normalization_stats=normalization_stats,
        spatial_ifft=diff_cfg.dataset.spatial_ifft,
        separate_zf=diff_cfg.dataset.separate_zf,
        real_potens=diff_cfg.dataset.real_potens,
        offset=diff_cfg.dataset.offset,
        bundle_seq_length=diff_cfg.model.bundle_seq_length,
        minmax_beta1=diff_cfg.dataset.minmax_beta1,
        minmax_beta2=diff_cfg.dataset.minmax_beta2,
        log_transform=diff_cfg.dataset.log_transform,
        decouple_mu=diff_cfg.dataset.norm_decouple_mu,
        subsample=diff_cfg.dataset.subsample,
        rank=0,
        num_workers=0,
    )
    if dataset_class is CycloneVAEDataset:
        kwargs["latent_sampling_mode"] = str(
            latent_sampling_mode or diff_cfg.dataset.get("latent_sampling_mode", "stochastic")
        )
    return dataset_class(**kwargs)


def _make_generation_dataset(
    diff_cfg,
    backend_name: str,
    data_path: str,
    trajectories,
    normalization_stats,
    dataset_class=CycloneAEDataset,
    latent_sampling_mode: Optional[str] = None,
):
    backend = _get_backend(backend_name)
    kwargs = dict(
        backend=backend,
        conditions=list(diff_cfg.model.conditioning),
        active_keys=list(diff_cfg.dataset.active_keys),
        fields_to_load=["df"],
        path=data_path,
        split="val",
        trajectories=trajectories,
        normalization=OmegaConf.to_container(diff_cfg.dataset.normalization),
        normalization_scope=diff_cfg.dataset.normalization_scope,
        normalization_stats=normalization_stats,
        spatial_ifft=diff_cfg.dataset.spatial_ifft,
        separate_zf=diff_cfg.dataset.separate_zf,
        real_potens=diff_cfg.dataset.real_potens,
        offset=diff_cfg.dataset.offset,
        bundle_seq_length=diff_cfg.model.bundle_seq_length,
        minmax_beta1=diff_cfg.dataset.minmax_beta1,
        minmax_beta2=diff_cfg.dataset.minmax_beta2,
        log_transform=diff_cfg.dataset.log_transform,
        decouple_mu=diff_cfg.dataset.norm_decouple_mu,
        subsample=1,
        rank=0,
        num_workers=0,
    )
    if dataset_class is CycloneVAEDataset:
        kwargs["latent_sampling_mode"] = str(
            latent_sampling_mode or diff_cfg.dataset.get("latent_sampling_mode", "stochastic")
        )
    return dataset_class(**kwargs)


def _get_latent_scale(cfg, trainset, autoencoder, device: torch.device) -> float:
    latent_scale_override = cfg.inference.get("latent_scale", None)
    if latent_scale_override is not None:
        return float(latent_scale_override)

    loader = DataLoader(
        trainset,
        batch_size=int(cfg.latent_stats.batch_size),
        shuffle=False,
        num_workers=int(cfg.latent_stats.num_workers),
        collate_fn=trainset.collate,
        pin_memory=False,
    )
    trainset.precompute_latents(
        rank=0,
        dataloader=loader,
        autoencoder=autoencoder,
        device=device,
    )
    return float(1.0 / np.sqrt(trainset.latent_stats.var).item())


def _cache_section(cfg) -> Dict[str, Any]:
    cache_cfg = cfg.get("cache", None)
    if cache_cfg is None:
        return {}
    return OmegaConf.to_container(cache_cfg, resolve=True)


def _discover_cached_file(
    data_path: str,
    stem: str,
    extension: str,
    offset: int,
    decouple_mu: bool,
) -> Optional[str]:
    pattern = os.path.join(data_path, f"{stem}*{extension}")
    candidates = glob.glob(pattern)
    if not candidates:
        return None

    filtered = []
    for p in candidates:
        b = os.path.basename(p)
        if f"offset{offset}" not in b:
            continue
        if decouple_mu and "_mu_" not in b:
            continue
        if not decouple_mu and "_mu_" in b:
            continue
        filtered.append(p)

    if not filtered:
        filtered = candidates

    filtered.sort(key=os.path.getmtime, reverse=True)
    return filtered[0]


def _maybe_aggregate(mean, var, min_, max_, agg_axes):
    if agg_axes is None:
        return mean, var, min_, max_
    if len(mean.shape) > max(agg_axes):
        return RunningMeanStd.aggregate_stats(mean, var, min_, max_, agg_axes=agg_axes)
    return mean, var, min_, max_


def _extract_df_stats(stats_obj, agg_axes=None):
    if isinstance(stats_obj, dict) and "df" in stats_obj:
        df_stats = stats_obj["df"]
        if isinstance(df_stats, dict) and "full" in df_stats:
            df_stats = df_stats["full"]

        if isinstance(df_stats, dict) and all(k in df_stats for k in ["mean", "std", "min", "max"]):
            mean = np.array(df_stats["mean"], dtype=np.float32)
            var = np.array(df_stats["std"], dtype=np.float32) ** 2
            min_ = np.array(df_stats["min"], dtype=np.float32)
            max_ = np.array(df_stats["max"], dtype=np.float32)
            mean, var, min_, max_ = _maybe_aggregate(mean, var, min_, max_, agg_axes)
            return mean, np.sqrt(var), min_, max_

        if hasattr(df_stats, "mean") and hasattr(df_stats, "var") and hasattr(df_stats, "min") and hasattr(df_stats, "max"):
            mean = np.array(df_stats.mean, dtype=np.float32)
            var = np.array(df_stats.var, dtype=np.float32)
            min_ = np.array(df_stats.min, dtype=np.float32)
            max_ = np.array(df_stats.max, dtype=np.float32)
            mean, var, min_, max_ = _maybe_aggregate(mean, var, min_, max_, agg_axes)
            return mean, np.sqrt(var), min_, max_

    if hasattr(stats_obj, "mean") and hasattr(stats_obj, "var") and hasattr(stats_obj, "min") and hasattr(stats_obj, "max"):
        mean = np.array(stats_obj.mean, dtype=np.float32)
        var = np.array(stats_obj.var, dtype=np.float32)
        min_ = np.array(stats_obj.min, dtype=np.float32)
        max_ = np.array(stats_obj.max, dtype=np.float32)
        mean, var, min_, max_ = _maybe_aggregate(mean, var, min_, max_, agg_axes)
        return mean, np.sqrt(var), min_, max_

    if isinstance(stats_obj, dict) and all(k in stats_obj for k in ["mean", "var", "min", "max"]):
        mean = np.array(stats_obj["mean"], dtype=np.float32)
        var = np.array(stats_obj["var"], dtype=np.float32)
        min_ = np.array(stats_obj["min"], dtype=np.float32)
        max_ = np.array(stats_obj["max"], dtype=np.float32)
        mean, var, min_, max_ = _maybe_aggregate(mean, var, min_, max_, agg_axes)
        return mean, np.sqrt(var), min_, max_

    raise ValueError("Unsupported normalization stats format")


def _load_normalization_stats(norm_stats_path: str, diff_cfg) -> Dict[str, Any]:
    with open(norm_stats_path, "rb") as fh:
        loaded_stats = pickle.load(fh)

    df_norm_cfg = diff_cfg.dataset.normalization.df
    df_agg_axes = tuple(df_norm_cfg.agg_axes) if df_norm_cfg.get("agg_axes", None) else None
    df_mean, df_std, df_min, df_max = _extract_df_stats(loaded_stats, agg_axes=df_agg_axes)

    return {
        "df": {
            "full": {
                "mean": df_mean,
                "std": df_std,
                "min": df_min,
                "max": df_max,
            }
        }
    }


def _latent_scale_from_cached_pkl(
    latent_pkl_path: str,
    latent_sampling_mode: str = "stochastic",
) -> float:
    with open(latent_pkl_path, "rb") as fh:
        loaded = pickle.load(fh)

    precomputed_latents = loaded["samples"] if isinstance(loaded, dict) and "samples" in loaded else loaded

    latent_running = None
    for sample in precomputed_latents.values():
        if "mu" in sample and "var" in sample:
            mu = np.asarray(sample["mu"], dtype=np.float32)
            var = np.clip(np.asarray(sample["var"], dtype=np.float32), 1e-12, None)
            norm_axes = tuple(range(0, mu.ndim))

            if latent_sampling_mode == "stochastic":
                x_mean = np.mean(mu, axis=norm_axes, keepdims=True)
                mu2_mean = np.mean(mu**2, axis=norm_axes, keepdims=True)
                var_mean = np.mean(var, axis=norm_axes, keepdims=True)
                x_var = var_mean + mu2_mean - x_mean**2
                std = np.sqrt(var)
                x_min = np.min(mu - 3.0 * std, axis=norm_axes, keepdims=True)
                x_max = np.max(mu + 3.0 * std, axis=norm_axes, keepdims=True)
            else:
                x_mean = np.mean(mu, axis=norm_axes, keepdims=True)
                x_var = np.var(mu, axis=norm_axes, keepdims=True)
                x_min = np.min(mu, axis=norm_axes, keepdims=True)
                x_max = np.max(mu, axis=norm_axes, keepdims=True)
        else:
            x = np.asarray(sample["x"], dtype=np.float32)
            norm_axes = tuple(range(0, x.ndim))
            x_mean = np.mean(x, axis=norm_axes, keepdims=True)
            x_var = np.var(x, axis=norm_axes, keepdims=True)
            x_min = np.min(x, axis=norm_axes, keepdims=True)
            x_max = np.max(x, axis=norm_axes, keepdims=True)
        if latent_running is None:
            latent_running = RunningMeanStd(shape=x_mean.shape)
        latent_running.update(x_mean, x_var, x_min, x_max)

    return float(1.0 / np.sqrt(latent_running.var).item())


def _load_geometry_template(
    data_path: str,
    backend_name: str,
    trajectory_name: str,
    spatial_ifft: bool,
    split_into_bands: Optional[int],
    real_potens: bool,
) -> Dict[str, np.ndarray]:
    backend = _get_backend(backend_name)
    traj_path = os.path.join(data_path, trajectory_name)
    formatted = backend.format_path(
        traj_path,
        spatial_ifft=spatial_ifft,
        split_into_bands=split_into_bands,
        real_potens=real_potens,
    )
    meta = backend.read_metadata(formatted, input_fields=["df"])
    return _maybe_to_numpy_dict(meta.get("geometry", {})) or {}


def _first_meta_value(meta: Dict[str, Any], keys: Sequence[str]) -> Optional[np.ndarray]:
    for k in keys:
        if k in meta and meta[k] is not None:
            return np.asarray(meta[k])
    return None


def _prepare_gt_tensor(gt_value: np.ndarray, target_shape: Sequence[int]) -> Optional[torch.Tensor]:
    gt = torch.as_tensor(np.asarray(gt_value), dtype=torch.float32)
    gt = torch.squeeze(gt)
    while gt.ndim < len(target_shape):
        gt = gt.unsqueeze(0)

    if gt.ndim != len(target_shape):
        return None

    for dim, tdim in zip(gt.shape, target_shape):
        if dim not in (1, tdim):
            return None

    return gt


def _compute_optional_metrics(
    gen_df: torch.Tensor,
    source_meta: Optional[Dict[str, Any]],
    geometry: Optional[Dict[str, np.ndarray]],
    calculate_metrics: bool,
    precomputed_flux_preds: Optional[np.ndarray] = None,
    precomputed_phi_stack: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    if not calculate_metrics:
        return {}

    if source_meta is None:
        warnings.warn(
            "CALCULATE_MSE_METRICS=True but source metadata is unavailable for this target. "
            "Skipping metric computation."
        )
        return {}

    metric_meta: Dict[str, Any] = {}
    gt_df_mean = _first_meta_value(source_meta, ["df_mean"])
    gt_phi_mean = _first_meta_value(source_meta, ["phi_mean", "phi_mean_field"])
    gt_flux_mean_arr = _first_meta_value(source_meta, ["flux_mean"])
    flux_series = _first_meta_value(source_meta, ["fluxes", "flux"])

    gt_flux_mean: Optional[float] = None
    if gt_flux_mean_arr is not None:
        gt_flux_mean = float(np.mean(np.asarray(gt_flux_mean_arr, dtype=np.float32)))
    elif flux_series is not None:
        flux_values = np.asarray(flux_series, dtype=np.float32).reshape(-1)
        if flux_values.size > 1:
            flux_values = flux_values[1:]
        if flux_values.size >= 80:
            flux_values = flux_values[-80:]
        gt_flux_mean = float(np.mean(flux_values))
    else:
        warnings.warn(
            "Ground-truth flux mean unavailable in metadata (expected one of: flux_mean, fluxes, flux). "
            "Skipping flux mean metrics."
        )

    if gt_df_mean is not None:
        gt_df_tensor = _prepare_gt_tensor(gt_df_mean, gen_df.shape[1:])
        if gt_df_tensor is None:
            warnings.warn(
                f"Cannot align metadata df_mean shape {np.asarray(gt_df_mean).shape} "
                f"to generated df shape {tuple(gen_df.shape[1:])}. Skipping df MSE metrics."
            )
        else:
            df_mse_per_sample = ((gen_df.float() - gt_df_tensor.unsqueeze(0)) ** 2).flatten(1).mean(dim=1)
            metric_meta["mse_df_vs_gt_mean_per_sample"] = (
                df_mse_per_sample.detach().cpu().numpy().astype(np.float32)
            )
            metric_meta["mse_df_vs_gt_mean"] = np.array(
                [float(df_mse_per_sample.mean().item())], dtype=np.float32
            )
    else:
        warnings.warn(
            "Ground-truth df mean unavailable in metadata (expected key: df_mean). "
            "Skipping df MSE metrics."
        )

    needs_integrals = gt_phi_mean is not None or gt_flux_mean is not None
    if needs_integrals and geometry is None:
        warnings.warn(
            "Geometry is required for phi/flux metrics but is missing. "
            "Skipping phi/flux metric computation."
        )
        return metric_meta

    if not needs_integrals:
        if gt_phi_mean is None:
            warnings.warn(
                "Ground-truth phi mean unavailable in metadata (expected one of: phi_mean, phi_mean_field). "
                "Skipping phi metrics."
            )
        return metric_meta

    try:
        if precomputed_flux_preds is not None and precomputed_phi_stack is not None:
            flux_preds_np = np.asarray(precomputed_flux_preds, dtype=np.float32)
            phi_stack = precomputed_phi_stack
        else:
            geom_tensors = {
                k: torch.as_tensor(v, dtype=torch.float32)
                for k, v in (geometry or {}).items()
            }
            flux_preds: List[float] = []
            phi_preds: List[torch.Tensor] = []
            for i in range(gen_df.shape[0]):
                phi_i, (_, eflux_i, _) = get_integrals(
                    gen_df[i].float().cpu(), geom_tensors
                )
                flux_preds.append(float(torch.as_tensor(eflux_i).sum().item()))
                phi_preds.append(torch.as_tensor(phi_i, dtype=torch.float32).cpu())

            flux_preds_np = np.asarray(flux_preds, dtype=np.float32)
            phi_stack = torch.stack(phi_preds, dim=0)

        metric_meta["pred_flux_per_sample"] = flux_preds_np
        metric_meta["pred_flux_mean"] = np.array([float(np.mean(flux_preds_np))], dtype=np.float32)

        phi_mean_pred = phi_stack.mean(dim=0)
        metric_meta["pred_phi_mean"] = phi_mean_pred.numpy().astype(np.float32)

        if gt_flux_mean is not None:
            flux_mse_per_sample = (flux_preds_np - np.float32(gt_flux_mean)) ** 2
            metric_meta["gt_flux_mean"] = np.array([gt_flux_mean], dtype=np.float32)
            metric_meta["mse_flux_vs_gt_mean_per_sample"] = flux_mse_per_sample.astype(np.float32)
            metric_meta["mse_flux_vs_gt_mean"] = np.array(
                [float(np.mean(flux_mse_per_sample))], dtype=np.float32
            )

        if gt_phi_mean is not None:
            gt_phi_tensor = _prepare_gt_tensor(gt_phi_mean, phi_stack.shape[1:])
            if gt_phi_tensor is None:
                warnings.warn(
                    f"Cannot align metadata phi_mean shape {np.asarray(gt_phi_mean).shape} "
                    f"to predicted phi shape {tuple(phi_stack.shape[1:])}. Skipping phi MSE metrics."
                )
            else:
                phi_mse_per_sample = ((phi_stack - gt_phi_tensor.unsqueeze(0)) ** 2).flatten(1).mean(dim=1)
                metric_meta["mse_phi_vs_gt_mean_per_sample"] = (
                    phi_mse_per_sample.detach().cpu().numpy().astype(np.float32)
                )
                metric_meta["mse_phi_vs_gt_mean"] = np.array(
                    [float(phi_mse_per_sample.mean().item())], dtype=np.float32
                )
    except Exception as exc:
        warnings.warn(f"Failed to compute phi/flux metrics from generated fields: {exc}")

    return metric_meta


def _to_np_float32(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


def _compute_array_stats(arr: np.ndarray) -> Dict[str, np.ndarray]:
    arr_f = _to_np_float32(arr)
    return {
        "mean": _to_np_float32(np.mean(arr_f, axis=0)),
        "var": _to_np_float32(np.var(arr_f, axis=0)),
        "std": _to_np_float32(np.std(arr_f, axis=0)),
        "min": _to_np_float32(np.min(arr_f, axis=0)),
        "max": _to_np_float32(np.max(arr_f, axis=0)),
    }


def _metadata_value_or_default(
    source_meta: Optional[Dict[str, Any]],
    key: str,
    default: np.ndarray,
) -> np.ndarray:
    if source_meta is None or key not in source_meta or source_meta[key] is None:
        return default
    return np.asarray(source_meta[key])


def _compute_generated_phi_flux(
    gen_df: torch.Tensor,
    geometry: Optional[Dict[str, np.ndarray]],
) -> tuple[torch.Tensor, np.ndarray]:
    if geometry is None:
        raise ValueError(
            "Geometry is required to compute generated phi/flux metadata. "
            "For custom conditions, provide generation.geometry_template."
        )

    geom_tensors = {
        k: torch.as_tensor(v, dtype=torch.float32) for k, v in geometry.items()
    }
    flux_preds: List[float] = []
    phi_preds: List[torch.Tensor] = []
    for i in range(gen_df.shape[0]):
        phi_i, (_, eflux_i, _) = get_integrals(gen_df[i].float().cpu(), geom_tensors)
        flux_preds.append(float(torch.as_tensor(eflux_i).sum().item()))
        phi_preds.append(torch.as_tensor(phi_i, dtype=torch.float32).cpu())

    phi_stack = torch.stack(phi_preds, dim=0)
    pred_flux = np.asarray(flux_preds, dtype=np.float32)
    return phi_stack, pred_flux


def _write_metadata_subgroup(h5_file: Any, group_name: str, values: Dict[str, Any]) -> None:
    group_path = f"metadata/{group_name}"
    grp = h5_file[group_path] if group_path in h5_file else h5_file.create_group(group_path)
    for key, raw_value in values.items():
        if raw_value is None:
            continue
        value = raw_value
        if isinstance(value, str):
            value = np.bytes_(value)
        elif isinstance(value, (list, tuple)):
            value = np.asarray(value)
        if key in grp:
            continue
        grp.create_dataset(key, data=value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to generation config YAML.",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)

    seed = int(cfg.inference.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device_name = str(cfg.inference.get("device", "cuda"))
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable, falling back to CPU.")
        device_name = "cpu"
    device = torch.device(device_name)

    diff_ckp_dir = str(cfg.diffusion_checkpoint_dir)
    diff_cfg_path = os.path.join(diff_ckp_dir, "config.yaml")
    if not os.path.exists(diff_cfg_path):
        raise FileNotFoundError(f"Diffusion config not found: {diff_cfg_path}")
    diff_cfg = OmegaConf.load(diff_cfg_path)

    default_data_path = str(diff_cfg.dataset.path)
    # Legacy override: applies to both stats/training and generation unless specific overrides are set.
    legacy_data_path_override = cfg.get("data_path", None)
    stats_data_path_override = cfg.get("stats_data_path", None)
    generation_data_path_override = cfg.get("generation_data_path", None)

    stats_data_path = str(
        stats_data_path_override
        if stats_data_path_override is not None
        else (
            legacy_data_path_override
            if legacy_data_path_override is not None
            else default_data_path
        )
    )
    generation_data_path = str(
        generation_data_path_override
        if generation_data_path_override is not None
        else (
            legacy_data_path_override
            if legacy_data_path_override is not None
            else default_data_path
        )
    )
    backend_override = cfg.get("backend", None)
    backend_name = str(backend_override) if backend_override is not None else str(diff_cfg.dataset.get("backend", "h5"))
    model_snapshot = str(cfg.get("model_snapshot", "ckp.pth"))
    ckpt_path = _resolve_checkpoint_path(diff_ckp_dir, model_snapshot)

    ae_ckp = cfg.get("ae_checkpoint", None)
    if ae_ckp is None:
        ae_ckp = diff_cfg.get("ae_checkpoint", None)
    if ae_ckp is None:
        raise ValueError("AE checkpoint path must be provided in config or diffusion config.")

    print(f"Loading AE from: {ae_ckp}")
    autoencoder, _, _ = load_autoencoder(str(ae_ckp), device=device)
    autoencoder.checkpoint_path = os.path.abspath(str(ae_ckp))
    autoencoder = autoencoder.to(device)
    autoencoder.eval()

    is_vae_autoencoder = all(
        hasattr(autoencoder, key) for key in ["compute_kl_loss", "reparameterize"]
    )
    dataset_class = CycloneVAEDataset if is_vae_autoencoder else CycloneAEDataset
    latent_sampling_mode = str(
        cfg.generation.get(
            "latent_sampling_mode",
            diff_cfg.dataset.get("latent_sampling_mode", "stochastic"),
        )
    )

    print(
        f"Generation latent dataset mode: {'VAE' if is_vae_autoencoder else 'AE'}"
    )
    if is_vae_autoencoder:
        print(f"Generation latent sampling mode: {latent_sampling_mode}")

    cache_cfg = _cache_section(cfg)
    norm_stats_path = cache_cfg.get("norm_stats_path", cfg.get("norm_stats_path", None))
    latent_pkl_path = cache_cfg.get("latent_pkl_path", cfg.get("latent_pkl_path", None))

    if norm_stats_path is None:
        norm_stats_path = _discover_cached_file(
            data_path=stats_data_path,
            stem="diff_df",
            extension="stats.pkl",
            offset=int(diff_cfg.dataset.offset),
            decouple_mu=bool(diff_cfg.dataset.norm_decouple_mu),
        )
    if latent_pkl_path is None:
        latent_pkl_path = _discover_cached_file(
            data_path=stats_data_path,
            stem="diff_train_latents",
            extension="latents.pkl",
            offset=int(diff_cfg.dataset.offset),
            decouple_mu=bool(diff_cfg.dataset.norm_decouple_mu),
        )

    normalization_stats_override = None
    if norm_stats_path is not None and os.path.exists(norm_stats_path):
        print(f"Using cached normalization stats: {norm_stats_path}")
        normalization_stats_override = _load_normalization_stats(norm_stats_path, diff_cfg)
    else:
        print("No cached normalization stats found, will use dataset pipeline.")

    print("Building train dataset for normalization and latent scaling...")
    trainset = _make_train_dataset(
        diff_cfg=diff_cfg,
        backend_name=backend_name,
        data_path=stats_data_path,
        normalization_stats=normalization_stats_override,
        dataset_class=dataset_class,
        latent_sampling_mode=latent_sampling_mode,
    )

    latent_scale_override = cfg.inference.get("latent_scale", None)
    if latent_scale_override is not None:
        latent_scale = float(latent_scale_override)
    elif latent_pkl_path is not None and os.path.exists(latent_pkl_path):
        print(f"Using cached latent stats: {latent_pkl_path}")
        latent_scale = _latent_scale_from_cached_pkl(
            latent_pkl_path,
            latent_sampling_mode=latent_sampling_mode,
        )
    else:
        print("No cached latent stats found, precomputing latent scale from train data...")
        latent_scale = _get_latent_scale(cfg, trainset, autoencoder, device=device)
    print(f"Using latent_scale: {latent_scale:.8f}")

    print("Building diffusion model...")
    model = get_diffusion_model(diff_cfg, autoencoder, trainset)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    model.to(device).eval()
    print(f"Loaded diffusion snapshot: {ckpt_path}")

    sample_fn = _build_sampler(
        model=model,
        autoencoder=autoencoder,
        cfg=diff_cfg,
        device=device,
        latent_scale=latent_scale,
    )

    trajectories = cfg.generation.get("trajectories", None)
    cond_entries = cfg.generation.get("conditions", None)
    has_traj = trajectories is not None and trajectories != []
    has_cond = cond_entries is not None and len(cond_entries) > 0
    if not has_traj and not has_cond:
        raise ValueError("Provide at least one of generation.trajectories or generation.conditions.")

    cond_keys = list(diff_cfg.model.conditioning)
    targets: List[GenerationTarget] = []
    gen_dataset = None

    if has_traj:
        print(f"Building generation dataset from trajectories: {trajectories}")
        gen_dataset = _make_generation_dataset(
            diff_cfg=diff_cfg,
            backend_name=backend_name,
            data_path=generation_data_path,
            trajectories=trajectories,
            normalization_stats=(
                normalization_stats_override
                if normalization_stats_override is not None
                else getattr(trainset, "stats", None)
            ),
            dataset_class=dataset_class,
            latent_sampling_mode=latent_sampling_mode,
        )
        for file_idx, file_path in enumerate(gen_dataset.files):
            meta = gen_dataset.metadata[file_idx]
            condition = _extract_condition_from_meta(meta, cond_keys)
            geometry = _maybe_to_numpy_dict(meta.get("geometry", None))
            name = _basename_without_suffix(file_path)
            targets.append(
                GenerationTarget(
                    name=name,
                    condition=condition,
                    denorm_file_index=file_idx,
                    geometry=geometry,
                    source_metadata=meta,
                )
            )

    if has_cond:
        if diff_cfg.dataset.normalization_scope != "dataset":
            raise ValueError(
                "Custom conditions are only supported with dataset-level normalization "
                "(normalization_scope=dataset)."
            )

        geometry_template = cfg.generation.get("geometry_template", None)
        geometry = None
        if geometry_template is not None:
            geometry = _load_geometry_template(
                data_path=generation_data_path,
                backend_name=backend_name,
                trajectory_name=str(geometry_template),
                spatial_ifft=bool(diff_cfg.dataset.spatial_ifft),
                split_into_bands=diff_cfg.dataset.get("split_into_bands", None),
                real_potens=bool(diff_cfg.dataset.real_potens),
            )

        for idx, entry in enumerate(cond_entries):
            entry = OmegaConf.to_container(entry, resolve=True)
            name = str(entry.get("name", f"condition_{idx}"))
            cond_values = [float(entry[k]) for k in cond_keys]
            targets.append(
                GenerationTarget(
                    name=name,
                    condition=torch.tensor(cond_values, dtype=torch.float32),
                    denorm_file_index=0,
                    geometry=geometry,
                    source_metadata=None,
                )
            )

    out_dir = str(cfg.generation.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    h5_backend = H5Backend(rank=0)

    n_samples = int(cfg.generation.n_samples)
    chunk_size = int(cfg.generation.chunk_size)
    sample_steps = cfg.generation.get("steps", None)
    sample_steps = int(sample_steps) if sample_steps is not None else None
    save_normalized = _to_bool(cfg.generation.get("save_normalized", False), default=False)
    calculate_mse_metrics = _to_bool(
        cfg.generation.get("calculate_mse_metrics", False), default=False
    )
    output_prefix = str(cfg.generation.get("output_prefix", "generated"))
    generation_compute_seconds = 0.0
    total_generated_samples = 0

    print(f"Generating {n_samples} samples per target for {len(targets)} target(s)...")
    for target in targets:
        chunks = []
        target_compute_seconds = 0.0
        for start in range(0, n_samples, chunk_size):
            bs = min(chunk_size, n_samples - start)
            cond = target.condition.to(device).unsqueeze(0).expand(bs, -1).contiguous()

            compute_t0 = time.perf_counter()
            decoded = sample_fn(cond, steps=sample_steps)
            df = decoded["df"].detach().cpu()
            df = _canonicalize_generated_df(df)

            if not save_normalized:
                denorm_idx = target.denorm_file_index if target.denorm_file_index is not None else 0
                df = trainset.denormalize(denorm_idx, df=df)
                df = _recombine_separate_zf(df, separate_zf=bool(diff_cfg.dataset.separate_zf))
            chunk_compute_seconds = time.perf_counter() - compute_t0
            target_compute_seconds += chunk_compute_seconds
            generation_compute_seconds += chunk_compute_seconds
            total_generated_samples += int(df.shape[0])

            chunks.append(df)
            del decoded, cond
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        gen_df = torch.cat(chunks, dim=0)
        phi_stack, pred_flux = _compute_generated_phi_flux(
            gen_df=gen_df,
            geometry=target.geometry,
        )
        metric_metadata = _compute_optional_metrics(
            gen_df=gen_df,
            source_meta=target.source_metadata,
            geometry=target.geometry,
            calculate_metrics=calculate_mse_metrics,
            precomputed_flux_preds=pred_flux,
            precomputed_phi_stack=phi_stack,
        )
        df_np = gen_df.numpy().astype(np.float32)
        df_stats = _compute_array_stats(df_np)

        pred_flux = np.asarray(pred_flux, dtype=np.float32)
        flux_stats = _compute_array_stats(pred_flux)
        phi_stats = _compute_array_stats(phi_stack.numpy().astype(np.float32))
        phi_mean = phi_stats["mean"]
        phi_var = phi_stats["var"]
        phi_std = phi_stats["std"]
        phi_min = phi_stats["min"]
        phi_max = phi_stats["max"]

        source_timesteps = None
        if target.source_metadata is not None and "timesteps" in target.source_metadata:
            source_timesteps = np.asarray(target.source_metadata["timesteps"])
        if source_timesteps is not None and source_timesteps.shape[0] > 0:
            start_idx = 80 if source_timesteps.shape[0] > 80 else 0
            source_slice = source_timesteps[start_idx:]

            if source_slice.shape[0] >= gen_df.shape[0]:
                timesteps = source_slice[: gen_df.shape[0]]
            else:
                base_ts = source_slice
                if base_ts.shape[0] >= 2:
                    dt = base_ts[-1] - base_ts[-2]
                elif source_timesteps.shape[0] >= 2:
                    dt = source_timesteps[-1] - source_timesteps[-2]
                else:
                    dt = np.array(1.0, dtype=base_ts.dtype)
                n_extra = gen_df.shape[0] - base_ts.shape[0]
                extra = base_ts[-1] + dt * np.arange(1, n_extra + 1)
                timesteps = np.concatenate([base_ts, extra])
        else:
            timesteps = np.arange(gen_df.shape[0], dtype=np.float64)

        source_kyspec = np.zeros((gen_df.shape[0],), dtype=np.float32)
        source_fluxspec = np.zeros((gen_df.shape[0],), dtype=np.float32)

        ion_temp_grad = np.array([np.nan], dtype=np.float32)
        density_grad = np.array([np.nan], dtype=np.float32)
        s_hat = np.array([np.nan], dtype=np.float32)
        q = np.array([np.nan], dtype=np.float32)

        cond_lookup = {
            "itg": "ion_temp_grad",
            "dg": "density_grad",
            "s_hat": "s_hat",
            "q": "q",
        }
        cond_values = {
            "ion_temp_grad": ion_temp_grad,
            "density_grad": density_grad,
            "s_hat": s_hat,
            "q": q,
        }
        for i, cond_key in enumerate(cond_keys):
            mapped = cond_lookup.get(cond_key)
            if mapped is not None:
                cond_values[mapped] = np.array([float(target.condition[i].item())], dtype=np.float32)

        if target.source_metadata is not None:
            for key in ["ion_temp_grad", "density_grad", "s_hat", "q"]:
                if np.isnan(cond_values[key]).all() and key in target.source_metadata:
                    cond_values[key] = np.asarray(target.source_metadata[key], dtype=np.float32).reshape(-1)

        sample_shape = tuple(gen_df.shape[1:])
        out_path = os.path.join(out_dir, f"{output_prefix}_{target.name}.h5")

        # Always rewrite target file on reruns so metadata (including metrics) stays in sync.
        if os.path.exists(out_path):
            os.remove(out_path)

        base_metadata = {
            "trajectory": np.bytes_(target.name),
            "split": np.bytes_("generated"),
            "timesteps": timesteps,
            "fluxes": pred_flux,
            "flux": pred_flux,
            "resolution": np.array(sample_shape[1:], dtype=np.int32),
            "kyspec": source_kyspec,
            "fluxspec": source_fluxspec,
            "ion_temp_grad": cond_values["ion_temp_grad"],
            "density_grad": cond_values["density_grad"],
            "s_hat": cond_values["s_hat"],
            "q": cond_values["q"],
            "n_samples": np.array([gen_df.shape[0]], dtype=np.int32),
            "sample_steps": np.array([
                sample_steps if sample_steps is not None else -1
            ], dtype=np.int32),
            "conditioning_keys": np.bytes_(",".join(cond_keys)),
            "bundle_seq_length": np.array([int(diff_cfg.model.bundle_seq_length)], dtype=np.int32),
            "latent_scale": np.array([latent_scale], dtype=np.float32),
            "stats_data_path": np.bytes_(str(stats_data_path)),
            "generation_data_path": np.bytes_(str(generation_data_path)),
            "data_path": np.bytes_(str(generation_data_path)),
            "saved_normalized": np.array([1 if save_normalized else 0], dtype=np.int32),
            "calculate_mse_metrics": np.array([1 if calculate_mse_metrics else 0], dtype=np.int32),
            "df_mean": df_stats["mean"],
            "df_var": df_stats["var"],
            "df_std": df_stats["std"],
            "df_min": df_stats["min"],
            "df_max": df_stats["max"],
            "phi_mean": phi_mean,
            "phi_var": phi_var,
            "phi_std": phi_std,
            "phi_min": phi_min,
            "phi_max": phi_max,
            "flux_mean": flux_stats["mean"],
            "flux_var": flux_stats["var"],
            "flux_std": flux_stats["std"],
            "flux_min": flux_stats["min"],
            "flux_max": flux_stats["max"],
        }

        for i, cond_key in enumerate(cond_keys):
            cond_value = np.array([float(target.condition[i].item())], dtype=np.float32)
            if cond_key not in {"itg", "dg", "s_hat", "q"}:
                base_metadata[f"condition_{cond_key}"] = cond_value
        if target.geometry is not None:
            base_metadata["geometry"] = target.geometry
        for mk, mv in metric_metadata.items():
            base_metadata[mk] = mv

        provenance_metadata = {
            "ae_checkpoint": str(ae_ckp),
            "diffusion_checkpoint_dir": str(diff_ckp_dir),
            "diffusion_snapshot": str(model_snapshot),
            "diffusion_formulation": str(diff_cfg.model.diffusion.formulation),
            "noise_distribution": str(diff_cfg.model.diffusion.noise_distribution),
            "stats_data_path": str(stats_data_path),
            "generation_data_path": str(generation_data_path),
            "conditioning_keys": ",".join(cond_keys),
            "raw_config_json": json.dumps(OmegaConf.to_container(cfg, resolve=True), default=str),
        }

        with h5_backend.create(out_path) as f:
            h5_backend.write_metadata(f, base_metadata)
            _write_metadata_subgroup(f, "provenance", provenance_metadata)
            for i in range(gen_df.shape[0]):
                h5_backend.write_df(
                    f,
                    str(i).zfill(5),
                    gen_df[i].numpy().astype(np.float32),
                )

        print(f"Saved {gen_df.shape[0]} samples -> {out_path}")
        target_rate = gen_df.shape[0] / target_compute_seconds if target_compute_seconds > 0 else float("inf")
        print(
            f"Compute time [{target.name}] (diffusion+decode+denorm): "
            f"{target_compute_seconds:.2f}s ({target_rate:.2f} samples/s)"
        )
        if metric_metadata:
            if "mse_df_vs_gt_mean" in metric_metadata:
                print(
                    f"Metrics [{target.name}] df MSE vs GT mean: "
                    f"{float(metric_metadata['mse_df_vs_gt_mean'][0]):.6e}"
                )
            if "mse_flux_vs_gt_mean" in metric_metadata:
                print(
                    f"Metrics [{target.name}] flux-mean MSE vs GT mean: "
                    f"{float(metric_metadata['mse_flux_vs_gt_mean'][0]):.6e}"
                )
            if "mse_phi_vs_gt_mean" in metric_metadata:
                print(
                    f"Metrics [{target.name}] phi-mean MSE vs GT mean: "
                    f"{float(metric_metadata['mse_phi_vs_gt_mean'][0]):.6e}"
                )

    overall_rate = (
        total_generated_samples / generation_compute_seconds
        if generation_compute_seconds > 0
        else float("inf")
    )
    print(
        "Generation compute summary "
        "(diffusion+decode+denorm only; excludes model/data loading and H5 writing): "
        f"{generation_compute_seconds:.2f}s for {total_generated_samples} samples "
        f"({overall_rate:.2f} samples/s)"
    )
    print("Generation complete.")


if __name__ == "__main__":
    main()