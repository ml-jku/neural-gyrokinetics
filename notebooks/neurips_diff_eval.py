"""Helper functions for the NeurIPS diffusion evaluation notebook."""

import re
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.stats import pearsonr

from neugk.integrals import FluxIntegral
from neugk.utils import recombine_zf


# ---------------------------------------------------------------------------
# Space transforms (model <-> solver)
# ---------------------------------------------------------------------------

def to_model_space(df_spectral, separate_zf=True):
    """complex128 spectral -> float32 real-space (C-channel, with optional separate_zf)."""
    from neugk.utils import separate_zf as _separate_zf
    df_np = np.fft.ifftshift(np.array(df_spectral), axes=(3,))
    df_real = np.fft.ifftn(df_np, axes=(3, 4), norm="forward")
    out = np.stack([df_real.real, df_real.imag]).astype(np.float32)
    if separate_zf:
        out = _separate_zf(out, dim=0)
    return out


def from_model_space(df_model, separate_zf=True):
    """float32 real-space (C-channel) -> complex128 spectral, recombining zf if needed."""
    import jax.numpy as jnp
    if separate_zf:
        df_model = recombine_zf(df_model, dim=0)
    df_complex = (df_model[0] + 1j * df_model[1]).astype(np.complex128)
    df_spectral = np.fft.fftn(df_complex, axes=(3, 4), norm="forward")
    return jnp.asarray(np.fft.fftshift(df_spectral, axes=(3,)), dtype=jnp.complex128)


# ---------------------------------------------------------------------------
# Flux UQ evaluation
# ---------------------------------------------------------------------------

def flux_uq_for_valset(runner, valset, n_samples_per_traj=32, steps=10):
    """Generate N samples per trajectory, decode, denormalize, compute flux via integrals.

    Returns:
        traj_ids, pred_means, pred_stds, tgt_vals
    """
    runner.model.eval()
    grouped_tgts = defaultdict(list)
    traj_conditions = {}
    separate_zf = runner.cfg.dataset.separate_zf

    # extract per-trajectory info from metadata (no disk reads)
    cond_keys = sorted(runner.cfg.model.conditioning)
    # map short conditioning names to metadata keys
    _cond_meta_map = {"itg": "ion_temp_grad", "dg": "density_grad"}
    for fi, fpath in enumerate(valset.files):
        meta = valset.metadata[fi]
        match = re.search(r"(?:ood_)?iteration_\d+", fpath)
        traj_id = match.group() if match else f"file_{fi}"
        avg_flux = float(np.mean(meta["flux"][-80:]))
        n_samples = valset.file_num_samples[fi] if fi < len(valset.file_num_samples) else 1
        for _ in range(n_samples):
            grouped_tgts[traj_id].append(avg_flux)
        if traj_id not in traj_conditions:
            cond = torch.tensor(
                [float(np.squeeze(meta[_cond_meta_map.get(k, k)])) for k in cond_keys],
                dtype=torch.float32,
            )
            traj_conditions[traj_id] = (cond.unsqueeze(0), fi)

    if not traj_conditions:
        print("Warning: no trajectories found.")
        return [], np.array([]), np.array([]), np.array([])

    print(f"Found {len(traj_conditions)} trajectories")

    integrator = FluxIntegral(real_potens=False)
    integrator.cpu()

    grouped_preds = {}
    for traj_id, (cond, fi) in traj_conditions.items():
        cond_batch = cond.to(runner.device).expand(n_samples_per_traj, -1)
        geometry = valset.get_batch_geometry(
            torch.full((n_samples_per_traj,), fi, dtype=torch.long)
        )

        pred_fluxes = []
        chunk = min(8, n_samples_per_traj)
        for start in range(0, n_samples_per_traj, chunk):
            end = min(start + chunk, n_samples_per_traj)
            c = cond_batch[start:end]
            with torch.no_grad():
                preds = runner.sample(c, steps=steps, latent_only=False)
                pred_df = preds["df"]
                for b in range(pred_df.shape[0]):
                    pred_df[b] = valset.denormalize(fi, df=pred_df[b])
                if separate_zf and pred_df.shape[1] > 2:
                    pred_df = recombine_zf(pred_df, dim=1)
                geom_chunk = {k: v[start:end] for k, v in geometry.items()}
                pred_df = pred_df.cpu()
                _, (_, eflux, _) = integrator(geom_chunk, pred_df)
                pred_fluxes.extend(eflux.cpu().numpy().flatten().tolist())

        grouped_preds[traj_id] = pred_fluxes
        print(
            f"  {traj_id}: pred_flux mean={np.mean(pred_fluxes):.4f}, "
            f"GT avg_flux mean={np.mean(grouped_tgts[traj_id]):.4f}"
        )

    traj_ids = sorted(grouped_preds.keys())
    pred_means = np.array([np.mean(grouped_preds[i]) for i in traj_ids])
    pred_stds = np.array([np.std(grouped_preds[i]) for i in traj_ids])
    tgt_vals = np.array([np.mean(grouped_tgts[i]) for i in traj_ids])
    return traj_ids, pred_means, pred_stds, tgt_vals


def plot_flux_confidence(traj_ids, pred_means, pred_stds, tgt_vals, title=""):
    """Errorbar plot: predicted flux mean +/- std vs ground truth per trajectory."""
    if len(traj_ids) == 0:
        print("No trajectories to plot.")
        return plt.figure()
    fig, ax = plt.subplots(
        figsize=(max(8, len(traj_ids) * 0.6), 5), constrained_layout=True
    )
    x_pos = np.arange(len(traj_ids))
    ax.errorbar(
        x_pos, pred_means, yerr=pred_stds, fmt="o", capsize=6,
        label="Predicted (Mean +/- Std)", color="#1f77b4", mfc="white", mew=2, alpha=0.8,
    )
    ax.scatter(
        x_pos, tgt_vals, marker="x", s=80, color="#d62728", label="Ground Truth", zorder=3
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(traj_ids, rotation=45, ha="right")
    ax.set_xlabel("Trajectory ID")
    ax.set_ylabel("Average Flux")
    ax.set_title(title or "Flux Prediction Accuracy")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=True, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3, ls="--")
    return fig


# ---------------------------------------------------------------------------
# FID computation
# ---------------------------------------------------------------------------

def compute_statistics(samples):
    """Compute mean and covariance of (N, D) samples."""
    mu = np.mean(samples, axis=0)
    sigma = np.cov(samples, rowvar=False)
    return mu, sigma


def compute_fid(mu1, sigma1, mu2, sigma2):
    """Frechet distance between two multivariate Gaussians."""
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            print(f"Warning: imaginary component {np.max(np.abs(covmean.imag)):.6f}")
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))


@torch.no_grad()
def extract_gyroswin_latents(model, df_batch, device="cuda", **kwargs):
    """Extract bottleneck features from a GyroSwin (Swin5DUnet) encoder.

    Runs: patch_encode -> down_blocks -> middle (bottleneck).
    Returns flattened bottleneck features (B, D).

    Args:
        model: Swin5DUnet or wrapper with .patch_encode, .down_blocks, .middle
        df_batch: (B, C, ...) distribution function tensor
        device: target device
        **kwargs: conditioning kwargs (e.g. condition=..., timestep=...)
    """
    model.eval()
    x = df_batch.to(device)

    # Resolve the actual model if wrapped in DDP or similar
    m = model.module if hasattr(model, "module") else model

    # Use df_unet directly (works for both GyroSwin and GyroSwinMultitask)
    unet = m.df_unet if hasattr(m, "df_unet") else m

    cond = unet.condition(kwargs) if hasattr(unet, "condition") else {}
    # FiLM layers require a condition tensor even when unused — supply zeros
    if not cond and hasattr(unet, "cond_embed") and unet.cond_embed is not None:
        n_cond = len(unet.condition_keys) if hasattr(unet, "condition_keys") else 5
        dummy = torch.zeros(x.shape[0], n_cond, device=device)
        cond = {"condition": unet.cond_embed(dummy)}

    x, pad_axes = unet.patch_encode(x)
    for blk in unet.down_blocks:
        x, _ = blk(x, **cond)
    if hasattr(unet, "middle_pe"):
        x = unet.middle_pe(x)
    x = unet.middle(x, **cond)

    return x.flatten(1).cpu().numpy()


def compute_fid_on_latents(
    real_samples,
    gen_samples,
    feature_extractor,
    device="cuda",
    batch_size=16,
    n_components=None,
    **kwargs,
):
    """Compute FID between real and generated samples using a feature extractor.

    Args:
        real_samples: list/array of (C, ...) tensors (real distribution functions)
        gen_samples: list/array of (C, ...) tensors (generated distribution functions)
        feature_extractor: callable(batch, device, **kwargs) -> (B, D) numpy array
        device: target device
        batch_size: batch size for feature extraction
        n_components: optional PCA dimensionality reduction
        **kwargs: extra kwargs passed to feature_extractor

    Returns:
        fid: float
        real_feats: (N, D) numpy array
        gen_feats: (N, D) numpy array
    """
    from sklearn.decomposition import PCA

    def _extract_all(samples):
        feats = []
        for i in range(0, len(samples), batch_size):
            batch = torch.stack(samples[i:i + batch_size])
            feats.append(feature_extractor(batch, device=device, **kwargs))
        return np.concatenate(feats, axis=0)

    real_feats = _extract_all(real_samples)
    gen_feats = _extract_all(gen_samples)

    max_components = min(real_feats.shape[0], real_feats.shape[1])
    if n_components is not None and n_components < real_feats.shape[1]:
        n_components = min(n_components, max_components)
        pca = PCA(n_components=n_components)
        real_feats = pca.fit_transform(real_feats)
        gen_feats = pca.transform(gen_feats)

    mu_r, sig_r = compute_statistics(real_feats)
    mu_g, sig_g = compute_statistics(gen_feats)
    fid = compute_fid(mu_r, sig_r, mu_g, sig_g)
    return fid, real_feats, gen_feats


# ---------------------------------------------------------------------------
# Warm restart helpers
# ---------------------------------------------------------------------------

def run_trajectory_pair(df_gt, df_warm, geometry, params, pre, state_init,
                        n_steps=1000, label="", chunk_size=1, backend="cuda",
                        mixed_precision=True, print_every=500, log_every=1):
    """Run GT and warm-start trajectories in parallel via vmap.

    Returns (log_gt, log_warm) dicts with keys: time, kx_spec, ky_spec, eflux.
    """
    import jax
    import jax.numpy as jnp
    import dataclasses
    from gyaradax.solver import GKState, mode_amplitude
    from gyaradax.simulate import _compute_phi_for_init, gksolve, gk_run_batched
    from gyaradax.integrals import get_integrals
    from gyaradax.diag import get_diagnostics

    params = dataclasses.replace(params, backend=backend, mixed_precision=mixed_precision)

    nky = len(geometry["krho"])
    t_start = float(state_init.time)

    def _make_state(df_init):
        phi0 = _compute_phi_for_init(df_init, geometry, params)
        amp0 = mode_amplitude(phi0, geometry, params.norm_eps)
        return GKState(
            time=jnp.array(t_start, dtype=jnp.float64),
            step=jnp.array(0, dtype=jnp.int32),
            accumulated_norm_factor=jnp.ones(nky, dtype=jnp.float64),
            window_start_amp=amp0,
            last_growth_rate=jnp.zeros(nky, dtype=jnp.float64),
        )

    # run both trajectories — gksolve is jitted so second call hits cache
    dfs = [df_gt, df_warm]
    states = [_make_state(df_gt), _make_state(df_warm)]
    logs = [{"time": [], "kx_spec": [], "ky_spec": [], "eflux": []} for _ in range(2)]

    # initial diagnostics
    for b in range(2):
        phi, fluxes = get_integrals(dfs[b], geometry, params=params, pre=pre,
                                     adiabatic_electrons=params.adiabatic_electrons)
        diags = get_diagnostics(phi, fluxes, states[b])
        for k in logs[b]:
            logs[b][k].append(np.array(diags[k]))

    # run in chunks
    steps_done = 0
    while steps_done < n_steps:
        n = min(chunk_size, n_steps - steps_done)
        for b in range(2):
            dfs[b], (phi, fluxes), states[b] = gksolve(
                dfs[b], geometry, params, states[b], n_steps=n, pre=pre)
        steps_done += n

        # log diagnostics every log_every steps
        if steps_done % log_every == 0 or steps_done == n_steps:
            for b in range(2):
                phi_b, fluxes_b = get_integrals(
                    dfs[b], geometry, params=params, pre=pre,
                    adiabatic_electrons=params.adiabatic_electrons)
                diags = get_diagnostics(phi_b, fluxes_b, states[b])
                for k in logs[b]:
                    logs[b][k].append(np.array(diags[k]))

        # print progress less frequently
        if steps_done % print_every == 0 or steps_done == n_steps:
            q_gt = float(logs[0]["eflux"][-1])
            q_warm = float(logs[1]["eflux"][-1])
            t_now = float(states[0].time)
            ky_gt = np.log10(np.maximum(logs[0]["ky_spec"][-1], 1e-30))
            ky_w = np.log10(np.maximum(logs[1]["ky_spec"][-1], 1e-30))
            r_ky = pearsonr(ky_w, ky_gt)[0] if len(ky_gt) > 1 else 0.0
            print(
                f"  [{label}] {steps_done}/{n_steps}  t={t_now:.3f}"
                f"  Q_gt={q_gt:.4e}  Q_warm={q_warm:.4e}"
                f"  r(ky)={r_ky:.3f}"
            )

    log_gt = {k: np.array(v) for k, v in logs[0].items()}
    log_warm = {k: np.array(v) for k, v in logs[1].items()}
    # store final df states (spectral, JAX arrays -> numpy)
    log_gt["df_final"] = np.array(dfs[0])
    log_warm["df_final"] = np.array(dfs[1])
    return log_gt, log_warm


def run_trajectory(df_init, geometry, params, pre, state_init, n_steps=1000, label="",
                    chunk_size=1, backend="cuda", mixed_precision=True,
                    print_every=500, log_every=1):
    """Run a single trajectory. Convenience wrapper."""
    import jax.numpy as jnp
    import dataclasses
    from gyaradax.solver import GKState, mode_amplitude
    from gyaradax.simulate import _compute_phi_for_init, gksolve
    from gyaradax.integrals import get_integrals
    from gyaradax.diag import get_diagnostics

    params = dataclasses.replace(params, backend=backend, mixed_precision=mixed_precision)
    nky = len(geometry["krho"])
    t_start = float(state_init.time)
    phi0 = _compute_phi_for_init(df_init, geometry, params)
    amp0 = mode_amplitude(phi0, geometry, params.norm_eps)
    state = GKState(
        time=jnp.array(t_start, dtype=jnp.float64),
        step=jnp.array(0, dtype=jnp.int32),
        accumulated_norm_factor=jnp.ones(nky, dtype=jnp.float64),
        window_start_amp=amp0,
        last_growth_rate=jnp.zeros(nky, dtype=jnp.float64),
    )

    log = {"time": [], "kx_spec": [], "ky_spec": [], "eflux": []}
    phi, fluxes = get_integrals(df_init, geometry, params=params, pre=pre,
                                 adiabatic_electrons=params.adiabatic_electrons)
    diags = get_diagnostics(phi, fluxes, state)
    for k in log:
        log[k].append(np.array(diags[k]))

    steps_done = 0
    while steps_done < n_steps:
        n = min(chunk_size, n_steps - steps_done)
        df_init, (phi, fluxes), state = gksolve(df_init, geometry, params, state, n_steps=n, pre=pre)
        steps_done += n
        if steps_done % log_every == 0 or steps_done == n_steps:
            diags = get_diagnostics(phi, fluxes, state)
            for k in log:
                log[k].append(np.array(diags[k]))
        if steps_done % print_every == 0 or steps_done == n_steps:
            print(f"  [{label}] {steps_done}/{n_steps}  t={float(state.time):.3f}  Q={float(diags['eflux']):.4e}")

    return {k: np.array(v) for k, v in log.items()}


def load_reference_flux(gkw_dir, iteration):
    """Load saturated heat flux stats from GKW fluxes.dat."""
    path = os.path.join(gkw_dir, f"iteration_{iteration}", "fluxes.dat")
    data = np.loadtxt(path)
    eflux = data[:, 1]  # heat flux is second column
    tail = eflux[-80 * 3:]  # last 240 steps for robust std
    return float(np.mean(tail)), float(np.std(tail))


def time_to_convergence(log_warm, log_gt, time_axis, window=50,
                        flux_n_std=3.0, flux_threshold=0.6,
                        spec_threshold=0.95, verbose=True,
                        ref_flux_mean=None, ref_flux_std=None):
    """Compute TTC for flux and ky-spectrum separately.

    Returns:
        dict with keys 'flux' and 'ky_spec', each the time (in simulation units)
        until the criterion is met, or np.inf.

    Flux criterion: rolling fraction of warm flux within ref mean +/- flux_n_std * std.
    Spectrum criterion: rolling-average Pearson(log ky_spec) >= spec_threshold.

    If ref_flux_mean/ref_flux_std are provided, use those (from fluxes.dat).
    Otherwise fall back to GT log tail.
    """
    if ref_flux_mean is not None and ref_flux_std is not None:
        gt_flux_mean = ref_flux_mean
        gt_flux_std = ref_flux_std
    else:
        n_gt = len(log_gt["eflux"])
        tail = min(80, n_gt)
        gt_flux_tail = log_gt["eflux"][-tail:]
        gt_flux_mean = np.mean(gt_flux_tail)
        gt_flux_std = np.std(gt_flux_tail)
    flux_band = flux_n_std * gt_flux_std

    n = min(len(log_warm["eflux"]), len(time_axis))
    effective_window = min(window, max(5, n // 5))

    # --- flux in-band ---
    flux_in_band = np.array([
        abs(log_warm["eflux"][i] - gt_flux_mean) < flux_band
        for i in range(n)
    ], dtype=float)

    # --- ky spectrum pearson ---
    n_spec = min(len(log_warm["ky_spec"]), len(log_gt["ky_spec"]), n)
    ky_corr = np.zeros(n_spec)
    for i in range(n_spec):
        ky_w = np.log10(np.maximum(log_warm["ky_spec"][i], 1e-30))
        ky_g = np.log10(np.maximum(log_gt["ky_spec"][i], 1e-30))
        ky_corr[i] = pearsonr(ky_w, ky_g)[0] if len(ky_w) > 1 else 0.0

    # --- find TTC for each ---
    def _rolling_ttc(signal, threshold, win):
        for i in range(len(signal) - win):
            if np.mean(signal[i : i + win]) >= threshold:
                return float(time_axis[i] - time_axis[0])
        return np.inf

    ttc_flux = _rolling_ttc(flux_in_band, flux_threshold, effective_window)
    ttc_spec = _rolling_ttc(ky_corr, spec_threshold, effective_window)

    if verbose:
        n_in = int(flux_in_band.sum())
        mean_corr = float(np.mean(ky_corr)) if n_spec > 0 else 0.0
        print(f"    TTC flux: gt_mean={gt_flux_mean:.3e}, gt_std={gt_flux_std:.3e}, "
              f"band=±{flux_n_std:.0f}σ ({flux_band:.3e}), "
              f"in_band={n_in}/{n} ({n_in/max(n,1):.0%}), "
              f"window={effective_window}, ttc={ttc_flux:.3f}")
        print(f"    TTC spec: mean_r(ky)={mean_corr:.3f}, "
              f"window={effective_window}, ttc={ttc_spec:.3f}")

    return {"flux": ttc_flux, "ky_spec": ttc_spec}


# ---------------------------------------------------------------------------
# Correlation plots
# ---------------------------------------------------------------------------

def plot_correlation_grid(extended_metrics, metric_names=None, x_axes=None):
    """Plot correlation grid: columns = (FID, TTC), rows = error metrics.

    Args:
        extended_metrics: dict[iteration -> dict of metric values]
        metric_names: list of (key, label) tuples for y-axes
        x_axes: list of (key, label) tuples for x-axes (columns)
    """
    if metric_names is None:
        metric_names = [
            ("rel_err", "Rel. model-space error"),
            ("probe_flux_rmse", "Probing flux RMSE"),
            ("integral_flux_err", "Integral flux rel. error"),
            ("kyspec_rmse_init", "ky-spec RMSE (log, init)"),
            ("kxspec_rmse_init", "kx-spec RMSE (log, init)"),
            ("kyspec_pearson_init", "ky-spec Pearson (init)"),
        ]
    if x_axes is None:
        x_axes = [
            ("fid", "GyroSwin-FID"),
            ("ttc", r"Time to Convergence $[v_{th}/R]$"),
        ]

    n_rows = len(metric_names)
    n_cols = len(x_axes)
    iters = sorted(extended_metrics.keys())

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows), squeeze=False)

    for col, (x_key, x_label) in enumerate(x_axes):
        x_vals = np.array([extended_metrics[it][x_key] for it in iters])
        for row, (y_key, y_label) in enumerate(metric_names):
            ax = axes[row, col]
            y_vals = np.array([extended_metrics[it].get(y_key, np.nan) for it in iters])
            mask = np.isfinite(x_vals) & np.isfinite(y_vals)
            if mask.sum() < 2:
                ax.text(
                    0.5, 0.5, "insufficient data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="gray",
                )
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)
                continue
            xv, yv = x_vals[mask], y_vals[mask]
            ax.scatter(xv, yv, s=60, zorder=3, edgecolors="k", linewidth=0.5)
            for xi, yi, it in zip(xv, yv, np.array(iters)[mask]):
                ax.annotate(
                    str(it), (xi, yi), textcoords="offset points",
                    xytext=(6, 4), fontsize=8, color="gray",
                )
            if mask.sum() >= 3:
                coeffs = np.polyfit(xv, yv, 1)
                x_fit = np.linspace(xv.min(), xv.max(), 50)
                ax.plot(x_fit, np.polyval(coeffs, x_fit), "r--", lw=1, alpha=0.7)
                r, p = pearsonr(xv, yv)
                ax.text(
                    0.05, 0.95, f"r={r:.2f}, p={p:.2f}",
                    transform=ax.transAxes, fontsize=9, va="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                )
            ax.set_xlabel(x_label if row == n_rows - 1 else "")
            ax.set_ylabel(y_label)
            ax.grid(True, alpha=0.15)
            if row < n_rows - 1:
                ax.set_xticklabels([])

    axes[0, 0].set_title("vs. GyroSwin-FID", fontweight="bold", fontsize=12)
    if n_cols > 1:
        axes[0, 1].set_title("vs. Time to Convergence", fontweight="bold", fontsize=12)

    fig.suptitle("Error Metric Correlations", fontweight="bold", fontsize=14, y=1.01)
    fig.tight_layout()
    return fig
