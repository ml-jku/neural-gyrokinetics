"""NeurIPS diffusion evaluation helpers."""

import re
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.stats import pearsonr

from neugk.utils import recombine_zf


def set_seed(seed):
    """Seed python random, numpy, and torch (CPU + CUDA) for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_model_space(df_spectral, separate_zf=True):
    """Complex128 spectral -> float32 model-space (with optional separate_zf)."""
    from neugk.utils import separate_zf as _separate_zf

    # match preprocessing convention: fftshift before ifft (see preprocess.py:261, augment.py:46)
    df_np = np.fft.fftshift(np.array(df_spectral), axes=(3,))
    df_real = np.fft.ifftn(df_np, axes=(3, 4), norm="forward")
    out = np.stack([df_real.real, df_real.imag]).astype(np.float32)
    if separate_zf:
        out = _separate_zf(out, dim=0)
    return out


def from_model_space(df_model, separate_zf=True):
    """Float32 model-space -> complex128 spectral."""
    import jax.numpy as jnp

    if separate_zf:
        df_model = recombine_zf(df_model, dim=0)
    df_complex = (df_model[0] + 1j * df_model[1]).astype(np.complex128)
    df_spectral = np.fft.fftn(df_complex, axes=(3, 4), norm="forward")
    # inverse of preprocessing's fftshift is ifftshift (see augment.py:reverse_ifft)
    return jnp.asarray(np.fft.ifftshift(df_spectral, axes=(3,)), dtype=jnp.complex128)


def compute_statistics(samples, reg=1e-6):
    mu = np.mean(samples, axis=0)
    sigma = np.cov(samples, rowvar=False)
    if sigma.ndim == 2:
        sigma += np.eye(sigma.shape[0]) * reg
    return mu, sigma


def compute_fid(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            print(f"warning: imaginary component {np.max(np.abs(covmean.imag)):.6f}")
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))


def gyroswin_has_flux_head(model):
    """True if the loaded GyroSwin model exposes a usable flux_head."""
    m = model.module if hasattr(model, "module") else model
    return hasattr(m, "flux_head") and m.flux_head is not None


@torch.no_grad()
def extract_gyroswin_latents(
    model,
    df_batch,
    device="cuda",
    condition=None,
    source="bottleneck",
    cond_keys=None,
    default_timestep=150.0,
    flux_head_level=None,
    decoder_level=None,
    pool=None,
    **kwargs,
):
    """Extract feature vectors from a GyroSwin model for FID / MMD.

    source:
      "bottleneck"  — df_unet encoder middle-block activations.
                      Returns the flattened (B, C*spatial) tensor by default
                      (high-D, ~5e5 for an XXL model). Pass `pool="amax"` or
                      `pool="mean"` to spatially pool first → (B, C), the same
                      reduction flux_head uses; that's what you want for FID
                      with limited samples.
      "flux_head"   — multiscale pre-MLP flux_head latents (one pooled vector
                      per resolution level). Physics-targeted: these are the
                      features the model uses to regress the transport flux.
                      Pass ``flux_head_level=None`` (default) to concatenate
                      all levels; pass an integer index ``k`` to return only
                      level ``k`` (level 0 = bottleneck mix, level 1 = first
                      decoder up-block, level 2 = next, ...).
      "decoder"     — df_unet decoder up-block activations, captured via a
                      forward hook on ``df_unet.up_blocks[decoder_level]``.
                      Pythonic indexing: ``decoder_level=-1`` (default) is
                      the last up-block — i.e. the df features one level
                      before the final 5D output (analogue of InceptionV3
                      pool3 in classic FID). ``decoder_level=-2`` is two
                      levels before the 5D output, ``decoder_level=0`` is
                      the first up-block, etc.

    For source in {'flux_head', 'decoder'}, the full model.forward runs and
    a hook captures the chosen activation. `condition` is expected as a
    (B, len(cond_keys)) tensor whose columns correspond to `cond_keys`; if
    `cond_keys` omits 'timestep' we fill it with `default_timestep`.
    """
    model.eval()
    x = df_batch.to(device)
    m = model.module if hasattr(model, "module") else model

    if source == "bottleneck":
        unet = m.df_unet if hasattr(m, "df_unet") else m
        embed = unet.cond_embed if hasattr(unet, "cond_embed") else None
        if condition is None:
            raise ValueError("condition must be provided for GyroSwin feature extraction")
        c = condition.to(device)
        if embed is not None:
            if c.shape[-1] < embed.n_cond:
                ts = torch.empty(c.shape[0], embed.n_cond - c.shape[-1], device=device).uniform_(
                    100, 200
                )
                c = torch.cat([c, ts], dim=-1)
            cond = {"condition": embed(c)}
        else:
            cond = {}

        x, _ = unet.patch_encode(x)
        for blk in unet.down_blocks:
            x, _ = blk(x, **cond)
        if hasattr(unet, "middle_pe"):
            x = unet.middle_pe(x)
        x = unet.middle(x, **cond)
        # Optional spatial pool (analogue of flux_head's amax/mean reduction).
        # x is (B, ...spatial..., C); reduce over spatial axes 1..ndim-1.
        if pool == "amax":
            x = x.amax(axis=list(range(1, x.ndim - 1)))
        elif pool == "mean":
            x = x.mean(axis=list(range(1, x.ndim - 1)))
        return x.reshape(x.shape[0], -1).cpu().numpy()

    elif source == "flux_head":
        if not gyroswin_has_flux_head(m):
            raise RuntimeError("model has no flux_head; cannot use source='flux_head'")
        if condition is None or cond_keys is None:
            raise ValueError("source='flux_head' requires both `condition` and `cond_keys`")
        c = condition.to(device)
        n_keys = len(cond_keys)
        cols = c.shape[-1]
        if cols == n_keys:
            # no timestep in the tensor — fill with default
            cond_kwargs = {k: c[:, i] for i, k in enumerate(cond_keys)}
            cond_kwargs.setdefault(
                "timestep",
                torch.full((c.shape[0],), float(default_timestep), device=device, dtype=c.dtype),
            )
        elif cols == n_keys + 1:
            # caller appended timestep as the last column
            cond_kwargs = {k: c[:, i] for i, k in enumerate(cond_keys)}
            cond_kwargs["timestep"] = c[:, -1]
        else:
            raise ValueError(
                f"condition has {cols} cols; expected {n_keys} or {n_keys + 1} "
                f"(cond_keys = {list(cond_keys)}, optional trailing timestep)"
            )

        captured = {}

        def _hook(_mod, inputs, _output):
            captured["lats"] = inputs[0]

        handle = m.flux_head.register_forward_hook(_hook)
        try:
            _ = m(x, **cond_kwargs)
            lats = captured.get("lats")
            if lats is None:
                raise RuntimeError("forward pass did not invoke flux_head")
            flat_per_level = [l.reshape(l.shape[0], -1) for l in lats]
            if flux_head_level is None:
                flat = torch.cat(flat_per_level, dim=-1)
            else:
                if not 0 <= flux_head_level < len(flat_per_level):
                    raise IndexError(
                        f"flux_head_level={flux_head_level} out of range "
                        f"[0, {len(flat_per_level)})"
                    )
                flat = flat_per_level[flux_head_level]
            return flat.cpu().numpy()
        finally:
            handle.remove()

    elif source == "skip":
        if condition is None or cond_keys is None:
            raise ValueError("source='skip' requires both `condition` and `cond_keys`")
        c = condition.to(device)
        n_keys = len(cond_keys)
        cols = c.shape[-1]
        if cols == n_keys:
            cond_kwargs = {k: c[:, i] for i, k in enumerate(cond_keys)}
            cond_kwargs.setdefault(
                "timestep",
                torch.full((c.shape[0],), float(default_timestep), device=device, dtype=c.dtype),
            )
        elif cols == n_keys + 1:
            cond_kwargs = {k: c[:, i] for i, k in enumerate(cond_keys)}
            cond_kwargs["timestep"] = c[:, -1]
        else:
            raise ValueError(f"condition has {cols} cols; expected {n_keys} or {n_keys + 1}")

        # Capture skip connection from `df_unet.down_blocks[level]`. Down-block
        # forward returns `(x_down, x_pre)` — the second element is the skip
        # routed to the matching up-block; that's what the U-Net uses.
        down_blocks = m.df_unet.down_blocks
        n_down = len(down_blocks)
        # Default to the **deepest** skip (closest to the bottleneck).
        lvl = (n_down - 1) if decoder_level is None else decoder_level
        if not -n_down <= lvl < n_down:
            raise IndexError(f"skip level={lvl} out of range for {n_down} down_blocks")
        target = down_blocks[lvl]

        captured = {}

        def _hook(_mod, _inputs, output):
            # Down block returns (x_down, x_pre); take x_pre, the actual skip.
            if isinstance(output, (tuple, list)) and len(output) >= 2:
                captured["feat"] = output[1]
            else:
                captured["feat"] = output[0] if isinstance(output, (tuple, list)) else output

        handle = target.register_forward_hook(_hook)
        try:
            _ = m(x, **cond_kwargs)
            feat = captured.get("feat")
            if feat is None:
                raise RuntimeError(f"forward pass did not invoke down_blocks[{lvl}]")
            if pool == "amax":
                feat = feat.amax(axis=list(range(1, feat.ndim - 1)))
            elif pool == "mean":
                feat = feat.mean(axis=list(range(1, feat.ndim - 1)))
            return feat.reshape(feat.shape[0], -1).cpu().numpy()
        finally:
            handle.remove()

    elif source == "phi":
        if condition is None or cond_keys is None:
            raise ValueError("source='phi' requires both `condition` and `cond_keys`")
        c = condition.to(device)
        n_keys = len(cond_keys)
        cols = c.shape[-1]
        if cols == n_keys:
            cond_kwargs = {k: c[:, i] for i, k in enumerate(cond_keys)}
            cond_kwargs.setdefault(
                "timestep",
                torch.full((c.shape[0],), float(default_timestep), device=device, dtype=c.dtype),
            )
        elif cols == n_keys + 1:
            cond_kwargs = {k: c[:, i] for i, k in enumerate(cond_keys)}
            cond_kwargs["timestep"] = c[:, -1]
        else:
            raise ValueError(f"condition has {cols} cols; expected {n_keys} or {n_keys + 1}")

        # Aggregated phi latents: capture the phi-bottleneck activation
        # (`phi_middle` in SwinXNetMultitask). Spatial-pool by default to
        # avoid the huge flat dim — same `pool` kwarg as `bottleneck`.
        target = None
        for attr in ("phi_middle", "phi_middle_post"):
            if hasattr(m, attr):
                target = getattr(m, attr)
                break
        if target is None and hasattr(m, "phi_unet"):
            target = getattr(m.phi_unet, "middle", None)
        if target is None:
            raise RuntimeError("model exposes no phi-middle module to hook")

        captured = {}

        def _hook(_mod, _inputs, output):
            captured["feat"] = output[0] if isinstance(output, (tuple, list)) else output

        handle = target.register_forward_hook(_hook)
        try:
            _ = m(x, **cond_kwargs)
            feat = captured.get("feat")
            if feat is None:
                raise RuntimeError("forward pass did not invoke phi_middle")
            if pool == "amax":
                feat = feat.amax(axis=list(range(1, feat.ndim - 1)))
            elif pool == "mean":
                feat = feat.mean(axis=list(range(1, feat.ndim - 1)))
            return feat.reshape(feat.shape[0], -1).cpu().numpy()
        finally:
            handle.remove()

    elif source == "decoder":
        if condition is None or cond_keys is None:
            raise ValueError("source='decoder' requires both `condition` and `cond_keys`")
        c = condition.to(device)
        n_keys = len(cond_keys)
        cols = c.shape[-1]
        if cols == n_keys:
            cond_kwargs = {k: c[:, i] for i, k in enumerate(cond_keys)}
            cond_kwargs.setdefault(
                "timestep",
                torch.full((c.shape[0],), float(default_timestep), device=device, dtype=c.dtype),
            )
        elif cols == n_keys + 1:
            cond_kwargs = {k: c[:, i] for i, k in enumerate(cond_keys)}
            cond_kwargs["timestep"] = c[:, -1]
        else:
            raise ValueError(
                f"condition has {cols} cols; expected {n_keys} or {n_keys + 1} "
                f"(cond_keys = {list(cond_keys)}, optional trailing timestep)"
            )

        up_blocks = m.df_unet.up_blocks
        n_up = len(up_blocks)
        lvl = -1 if decoder_level is None else decoder_level
        if not -n_up <= lvl < n_up:
            raise IndexError(f"decoder_level={lvl} out of range for {n_up} up_blocks")
        target = up_blocks[lvl]

        captured = {}

        def _hook(_mod, _inputs, output):
            feat = output[0] if isinstance(output, (tuple, list)) else output
            captured["feat"] = feat

        handle = target.register_forward_hook(_hook)
        try:
            _ = m(x, **cond_kwargs)
            feat = captured.get("feat")
            if feat is None:
                raise RuntimeError(f"forward pass did not invoke up_blocks[{lvl}]")
            return feat.flatten(1).cpu().numpy()
        finally:
            handle.remove()

    else:
        raise ValueError(
            f"unknown source: {source!r} (use 'bottleneck', 'flux_head', or 'decoder')"
        )


def compute_fid_on_latents(
    real_samples,
    gen_samples,
    feature_extractor,
    device="cuda",
    batch_size=16,
    n_components=None,
    real_conditions=None,
    gen_conditions=None,
    **kwargs,
):
    from sklearn.decomposition import PCA

    def _extract_all(samples, conditions=None):
        feats = []
        for i in range(0, len(samples), batch_size):
            batch = torch.stack(samples[i : i + batch_size])
            kw = dict(kwargs)
            if conditions is not None:
                kw["condition"] = torch.stack(conditions[i : i + batch_size])
            feats.append(feature_extractor(batch, device=device, **kw))
        return np.concatenate(feats, axis=0)

    real_feats = _extract_all(real_samples, real_conditions)
    gen_feats = _extract_all(gen_samples, gen_conditions)

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


def run_trajectories(
    df_gt,
    df_warms,
    geometry,
    params,
    pre,
    state_init,
    n_steps=1000,
    labels=None,
    chunk_size=1,
    backend="cuda",
    mixed_precision=True,
    print_every=500,
    log_every=1,
):
    """Run GT once and multiple warm-start trajectories.

    Parameters
    ----------
    df_gt : array
        Ground-truth spectral distribution function.
    df_warms : list of arrays
        Warm-start ICs to compare against GT.
    labels : list of str, optional
        Names for each warm-start (for printing). Defaults to ["warm_0", ...].

    Returns
    -------
    log_gt : dict
    log_warms : list of dict  (same order as df_warms)
    """
    import jax.numpy as jnp
    import dataclasses
    from gyaradax.solver import GKState, mode_amplitude
    from gyaradax.simulate import _compute_phi_for_init, gksolve
    from gyaradax.integrals import get_integrals
    from gyaradax.diag import get_diagnostics

    params = dataclasses.replace(params, backend=backend, mixed_precision=mixed_precision)

    nky = len(geometry["krho"])
    t_start = float(state_init.time)
    n_warm = len(df_warms)
    if labels is None:
        labels = [f"warm_{i}" for i in range(n_warm)]

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

    # index 0 = GT, 1..N = warm-starts
    dfs = [df_gt] + list(df_warms)
    states = [_make_state(df) for df in dfs]
    ntraj = 1 + n_warm
    logs = [{"time": [], "kx_spec": [], "ky_spec": [], "eflux": []} for _ in range(ntraj)]

    for b in range(ntraj):
        phi, fluxes = get_integrals(
            dfs[b],
            geometry,
            params=params,
            pre=pre,
            adiabatic_electrons=params.adiabatic_electrons,
        )
        diags = get_diagnostics(phi, fluxes, states[b])
        for k in logs[b]:
            logs[b][k].append(np.array(diags[k]))

    steps_done = 0
    while steps_done < n_steps:
        n = min(chunk_size, n_steps - steps_done)
        for b in range(ntraj):
            dfs[b], (phi, fluxes), states[b] = gksolve(
                dfs[b], geometry, params, states[b], n_steps=n, pre=pre
            )
        steps_done += n

        if steps_done % log_every == 0 or steps_done == n_steps:
            for b in range(ntraj):
                phi_b, fluxes_b = get_integrals(
                    dfs[b],
                    geometry,
                    params=params,
                    pre=pre,
                    adiabatic_electrons=params.adiabatic_electrons,
                )
                diags = get_diagnostics(phi_b, fluxes_b, states[b])
                for k in logs[b]:
                    logs[b][k].append(np.array(diags[k]))

        if steps_done % print_every == 0 or steps_done == n_steps:
            q_gt = float(logs[0]["eflux"][-1])
            t_now = float(states[0].time)
            parts = [f"Q_gt={q_gt:.4e}"]
            ky_gt = np.log10(np.maximum(logs[0]["ky_spec"][-1], 1e-30))
            for w in range(n_warm):
                q_w = float(logs[1 + w]["eflux"][-1])
                ky_w = np.log10(np.maximum(logs[1 + w]["ky_spec"][-1], 1e-30))
                r_ky = pearsonr(ky_w, ky_gt)[0] if len(ky_gt) > 1 else 0.0
                parts.append(f"Q_{labels[w]}={q_w:.4e} r={r_ky:.3f}")
            print(f"  {steps_done}/{n_steps}  t={t_now:.3f}  " + "  ".join(parts))

    out_logs = []
    for b in range(ntraj):
        log = {k: np.array(v) for k, v in logs[b].items()}
        log["df_final"] = np.array(dfs[b])
        out_logs.append(log)

    return out_logs[0], out_logs[1:]


def run_trajectory_pair(
    df_gt,
    df_warm,
    geometry,
    params,
    pre,
    state_init,
    n_steps=1000,
    label="",
    chunk_size=1,
    backend="cuda",
    mixed_precision=True,
    print_every=500,
    log_every=1,
):
    """Run GT and a single warm-start trajectory. Convenience wrapper around run_trajectories."""
    log_gt, log_warms = run_trajectories(
        df_gt,
        [df_warm],
        geometry,
        params,
        pre,
        state_init,
        n_steps=n_steps,
        labels=[label],
        chunk_size=chunk_size,
        backend=backend,
        mixed_precision=mixed_precision,
        print_every=print_every,
        log_every=log_every,
    )
    return log_gt, log_warms[0]


def run_trajectory(
    df_init,
    geometry,
    params,
    pre,
    state_init,
    n_steps=1000,
    label="",
    chunk_size=1,
    backend="cuda",
    mixed_precision=True,
    print_every=500,
    log_every=1,
):
    """Run a single trajectory."""
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
    phi, fluxes = get_integrals(
        df_init,
        geometry,
        params=params,
        pre=pre,
        adiabatic_electrons=params.adiabatic_electrons,
    )
    diags = get_diagnostics(phi, fluxes, state)
    for k in log:
        log[k].append(np.array(diags[k]))

    steps_done = 0
    while steps_done < n_steps:
        n = min(chunk_size, n_steps - steps_done)
        df_init, (phi, fluxes), state = gksolve(
            df_init, geometry, params, state, n_steps=n, pre=pre
        )
        steps_done += n
        if steps_done % log_every == 0 or steps_done == n_steps:
            diags = get_diagnostics(phi, fluxes, state)
            for k in log:
                log[k].append(np.array(diags[k]))
        if steps_done % print_every == 0 or steps_done == n_steps:
            print(
                f"  [{label}] {steps_done}/{n_steps}  t={float(state.time):.3f}  Q={float(diags['eflux']):.4e}"
            )

    return {k: np.array(v) for k, v in log.items()}


def load_reference_flux(gkw_dir, iteration):
    """Heat flux stats from GKW fluxes.dat (last 240 steps)."""
    samples = load_reference_flux_samples(gkw_dir, iteration)
    return float(np.mean(samples)), float(np.std(samples))


def load_reference_flux_samples(gkw_dir, iteration, n_tail=240):
    """Raw GKW heat-flux samples (last `n_tail` steps of fluxes.dat).

    Used by the stationary-distribution divergences below as the reference
    sample $\\{Q_\\text{GKW}\\}$.
    """
    path = os.path.join(gkw_dir, f"iteration_{iteration}", "fluxes.dat")
    data = np.loadtxt(path)
    eflux = data[:, 1]
    return np.asarray(eflux[-n_tail:], dtype=np.float64)


# ---------------------------------------------------------------------------
# Stationary-distribution divergences (paper §5.2 lines 630–639).
#
# All metrics operate on two 1-D samples drawn from the *stationary window* of
# each trajectory. Vector spectra are handled by `spec_divergence` below,
# which broadcasts a 1-D metric across modes.
# ---------------------------------------------------------------------------


def stationary_window(x, frac=0.5):
    """Return the last `frac` fraction of a 1-D series (the post-saturation
    window we feed to the two-sample tests). frac=0.5 → second half."""
    x = np.asarray(x).ravel()
    if x.size == 0:
        return x
    k = max(1, int(round(frac * x.size)))
    return x[-k:]


# --- 1-D scalar divergences (flux) -----------------------------------------


def flux_ks_pvalue(x_warm, x_ref):
    """Two-sample Kolmogorov–Smirnov p-value. Higher = more indistinguishable
    from the reference distribution. Paper's primary $\\tau_Q$ test."""
    from scipy.stats import ks_2samp

    return float(ks_2samp(np.asarray(x_warm), np.asarray(x_ref)).pvalue)


def flux_ad_statistic(x_warm, x_ref):
    """Anderson–Darling k-sample statistic (k=2). Heavier tails of $Q$ get
    higher weight than under KS — paper alternative."""
    from scipy.stats import anderson_ksamp

    try:
        return float(anderson_ksamp([np.asarray(x_warm), np.asarray(x_ref)]).statistic)
    except Exception:
        # ad raises if either sample has < 2 distinct values
        return np.nan


def flux_wasserstein(x_warm, x_ref):
    """1-D Wasserstein-1 distance (Earth-mover). Calibrated divergence
    alternative to the KS p-value."""
    from scipy.stats import wasserstein_distance

    return float(wasserstein_distance(np.asarray(x_warm), np.asarray(x_ref)))


def gelman_rubin_R(x_warm, x_ref):
    """Two-chain Gelman–Rubin $\\hat R$ on the full window. < 1.1 ≈ converged.

    Standard MCMC mixing diagnostic with the warm-started run and the
    long cold-started GKW reference treated as the two chains."""
    chains = [np.asarray(x_warm, dtype=np.float64), np.asarray(x_ref, dtype=np.float64)]
    n = min(len(c) for c in chains)
    if n < 2:
        return np.nan
    chains = np.stack([c[-n:] for c in chains])  # (m, n)
    means = chains.mean(axis=1)
    grand = means.mean()
    B = n * np.sum((means - grand) ** 2) / (chains.shape[0] - 1)
    W = np.mean(np.var(chains, axis=1, ddof=1))
    if W <= 0:
        return np.nan
    var_hat = (n - 1) / n * W + B / n
    return float(np.sqrt(var_hat / W))


def gelman_rubin_t_curve(x_warm, x_ref):
    """$\\hat R(t)$ as a function of trajectory length (paper-stated, used
    for the convergence-time interpretation $\\hat R(t) < 1.1$).

    Returns an array of $\\hat R$ computed on prefixes of length 8, 16, ...,
    up to min(len(warm), len(ref)).
    """
    n = min(len(x_warm), len(x_ref))
    ts = []
    Rs = []
    t = 8
    while t <= n:
        Rs.append(gelman_rubin_R(np.asarray(x_warm)[-t:], np.asarray(x_ref)[-t:]))
        ts.append(t)
        t = min(n, t * 2) if t * 2 <= n else n + 1
    return np.asarray(ts), np.asarray(Rs)


def flux_autocorr(x, max_lag=40):
    """Centered autocorrelation $C_Q(\\tau) = \\langle Q(t)Q(t+\\tau)\\rangle - \\langle Q\\rangle^2$
    for $\\tau = 0, \\dots, \\text{max\\_lag}$ (normalized by the variance)."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    var = x.var()
    if var <= 0 or len(x) <= max_lag + 1:
        return np.full(max_lag + 1, np.nan)
    out = np.empty(max_lag + 1)
    for tau in range(max_lag + 1):
        out[tau] = np.mean(x[: len(x) - tau] * x[tau:]) / var
    return out


def flux_struct_fn(x, max_lag=40):
    """Second-order structure function $S_2(\\tau) = \\langle (Q(t+\\tau) - Q(t))^2 \\rangle$."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) <= max_lag + 1:
        return np.full(max_lag + 1, np.nan)
    out = np.empty(max_lag + 1)
    for tau in range(max_lag + 1):
        out[tau] = np.mean((x[tau:] - x[: len(x) - tau]) ** 2)
    return out


def autocorr_l1(x_warm, x_ref, max_lag=40, kind="autocorr"):
    """L1 distance between warm and reference correlation/structure curves
    (paper random-walk-consistency check)."""
    fn = flux_autocorr if kind == "autocorr" else flux_struct_fn
    a = fn(x_warm, max_lag)
    b = fn(x_ref, max_lag)
    if np.any(~np.isfinite(a)) or np.any(~np.isfinite(b)):
        return np.nan
    return float(np.mean(np.abs(a - b)))


def sliced_wasserstein(X_warm, X_ref, n_projections=64, seed=0):
    """Sliced Wasserstein-1 between two multivariate samples by averaging
    1-D $W_1$ over `n_projections` random unit directions."""
    rng = np.random.default_rng(seed)
    X_warm = np.asarray(X_warm)
    X_ref = np.asarray(X_ref)
    if X_warm.ndim == 1:
        X_warm = X_warm[:, None]
        X_ref = X_ref[:, None]
    d = X_warm.shape[1]
    P = rng.standard_normal((d, n_projections))
    P /= np.linalg.norm(P, axis=0, keepdims=True) + 1e-12
    proj_w = X_warm @ P  # (n_warm, n_proj)
    proj_r = X_ref @ P
    from scipy.stats import wasserstein_distance

    vals = [wasserstein_distance(proj_w[:, i], proj_r[:, i]) for i in range(n_projections)]
    return float(np.mean(vals))


# --- vector divergences (spectra) ------------------------------------------


def _per_mode_apply(metric_fn, X_warm, X_ref):
    """Apply a 1-D metric per mode-axis column. Returns an array of length
    K (number of spectral modes)."""
    X_warm = np.asarray(X_warm)
    X_ref = np.asarray(X_ref)
    if X_warm.ndim == 1:
        return np.asarray([metric_fn(X_warm, X_ref)])
    K = X_warm.shape[-1]
    out = np.empty(K)
    for k in range(K):
        out[k] = metric_fn(X_warm[..., k], X_ref[..., k])
    return out


_SPEC_METRIC_TABLE = {
    "ks": flux_ks_pvalue,
    "ad": flux_ad_statistic,
    "w1": flux_wasserstein,
}


def spec_divergence(X_warm, X_ref, kind="wasserstein"):
    """Per-mode 1-D divergence aggregated over a vector spectrum.

    Returns ``{"per_mode": (K,) array, "mean", "median", "frac_indistinguishable"}``.
    The last field is the fraction of modes with KS p-value ≥ 0.05 — only
    populated when `kind == "ks"`, otherwise NaN.
    """
    if kind not in _SPEC_METRIC_TABLE:
        raise ValueError(f"unknown kind: {kind!r}; choose from {list(_SPEC_METRIC_TABLE)}")
    per_mode = _per_mode_apply(_SPEC_METRIC_TABLE[kind], X_warm, X_ref)
    finite = per_mode[np.isfinite(per_mode)]
    return {
        "per_mode": per_mode,
        "mean": float(np.mean(finite)) if finite.size else np.nan,
        "median": float(np.median(finite)) if finite.size else np.nan,
        "frac_indistinguishable": (float(np.mean(per_mode >= 0.05)) if kind == "ks" else np.nan),
    }


# --- driver: compute everything for a (warm, ref) pair ---------------------

_FLUX_SCALAR_DIVERGENCES = {
    "flux_ks_p": flux_ks_pvalue,
    "flux_ad": flux_ad_statistic,
    "flux_w1": flux_wasserstein,
    "flux_R": gelman_rubin_R,
}


def compute_distribution_divergences(
    log_run,
    log_gt,
    *,
    ref_flux_samples=None,
    frac=0.5,
    max_lag=40,
    spec_keys=("ky_spec", "kx_spec", "fluxspec"),
):
    """Compute every flux + spectra divergence for one trajectory pair.

    Parameters
    ----------
    log_run : dict
        Output of `run_trajectory_pair` for the warm (or cold) run. Must
        contain ``eflux`` and at least one of `spec_keys`.
    log_gt : dict
        GT-side log from the same call (its late-time window is the spectra
        reference).
    ref_flux_samples : np.ndarray or None
        Reference flux samples, e.g. from `load_reference_flux_samples`. If
        None, falls back to `stationary_window(log_gt["eflux"], frac)`.
    frac : float
        Stationary-window fraction (default 0.5 = second half).

    Returns
    -------
    dict
        Flat metric dict ready to drop into `warm_results[iter][run_label]`.
        Keys: `flux_<metric>` (scalars), `<spec>_<metric>_{mean,median,frac}`.
    """
    out = {}
    flux_warm = stationary_window(log_run["eflux"], frac)
    flux_ref = (
        np.asarray(ref_flux_samples)
        if ref_flux_samples is not None
        else stationary_window(log_gt["eflux"], frac)
    )

    for name, fn in _FLUX_SCALAR_DIVERGENCES.items():
        try:
            out[name] = float(fn(flux_warm, flux_ref))
        except Exception as e:
            out[name] = np.nan
            out[f"{name}_error"] = repr(e)
    try:
        out["flux_autocorr_L1"] = float(
            autocorr_l1(flux_warm, flux_ref, max_lag=max_lag, kind="autocorr")
        )
        out["flux_struct_L1"] = float(
            autocorr_l1(flux_warm, flux_ref, max_lag=max_lag, kind="struct")
        )
    except Exception as e:
        out["flux_autocorr_L1"] = np.nan
        out["flux_struct_L1"] = np.nan
        out["flux_autocorr_L1_error"] = repr(e)

    for spec_key in spec_keys:
        if spec_key not in log_run or spec_key not in log_gt:
            continue
        Xw = stationary_window_2d(np.asarray(log_run[spec_key]), frac)
        Xr = stationary_window_2d(np.asarray(log_gt[spec_key]), frac)
        if Xw.size == 0 or Xr.size == 0:
            continue
        for kind in ("ks", "ad", "w1"):
            d = spec_divergence(Xw, Xr, kind=kind)
            out[f"{spec_key}_{kind}_mean"] = d["mean"]
            out[f"{spec_key}_{kind}_median"] = d["median"]
            if kind == "ks":
                out[f"{spec_key}_{kind}_frac"] = d["frac_indistinguishable"]
        out[f"{spec_key}_sliced_w1"] = sliced_wasserstein(Xw, Xr)
    return out


def stationary_window_2d(X, frac=0.5):
    """Last `frac` fraction along the time axis (axis 0) of a 2-D (T, K) array."""
    X = np.asarray(X)
    if X.ndim == 1:
        return stationary_window(X, frac)
    if X.size == 0:
        return X
    T = X.shape[0]
    k = max(1, int(round(frac * T)))
    return X[-k:]


def time_to_convergence(
    log_warm,
    log_gt,
    time_axis,
    window=50,
    flux_n_std=3.0,
    flux_threshold=0.6,
    spec_threshold=0.95,
    verbose=True,
    ref_flux_mean=None,
    ref_flux_std=None,
):
    """TTC for flux and ky-spectrum separately. Returns dict with 'flux' and 'ky_spec'."""
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

    flux_in_band = np.array(
        [abs(log_warm["eflux"][i] - gt_flux_mean) < flux_band for i in range(n)],
        dtype=float,
    )

    n_spec = min(len(log_warm["ky_spec"]), len(log_gt["ky_spec"]), n)
    ky_corr = np.zeros(n_spec)
    for i in range(n_spec):
        ky_w = np.log10(np.maximum(log_warm["ky_spec"][i], 1e-30))
        ky_g = np.log10(np.maximum(log_gt["ky_spec"][i], 1e-30))
        ky_corr[i] = pearsonr(ky_w, ky_g)[0] if len(ky_w) > 1 else 0.0

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
        print(
            f"    ttc flux: gt_mean={gt_flux_mean:.3e}, gt_std={gt_flux_std:.3e}, "
            f"band=±{flux_n_std:.0f}σ ({flux_band:.3e}), "
            f"in_band={n_in}/{n} ({n_in/max(n,1):.0%}), "
            f"window={effective_window}, ttc={ttc_flux:.3f}"
        )
        print(
            f"    ttc spec: mean_r(ky)={mean_corr:.3f}, "
            f"window={effective_window}, ttc={ttc_spec:.3f}"
        )

    return {"flux": ttc_flux, "ky_spec": ttc_spec}


def remap_gyroswin_checkpoint(old_sd, new_model, encoder_only=True, verbose=True):
    """Map state_dict keys from an older GyroSwinMultitask checkpoint to the current
    GyroSwinMultitask model architecture.

    Key differences handled:
    1. Old has single ``cond_embed`` per unet; new has split ``enc_cond_embed`` +
       ``dec_cond_embed``.  When the new model shares the same object for all three
       (``cond_embed`` / ``enc_cond_embed`` / ``dec_cond_embed``), loading one
       canonical key populates all aliases automatically.
    2. Old stores plain RPB tensors (``.rpb``, ``.rpb_idx``) and a CPB MLP as
       direct children of ``WindowAttention``.  New wraps them inside an ``RPB``
       sub-module: ``.rpb.cpb_mlp``, ``.rpb.rpb``, ``.rpb.rpb_idx``.
    3. Old uses FiLM conditioning (``.conditioning.N.modulation``); new may use DiT
       (``.blocks.N.dit.modulation``).  When the new model also uses FiLM, keys map
       directly.  FiLM -> DiT is a shape mismatch (2*dim vs ~6*dim); those are
       skipped with a warning.
    4. Old depth may differ from new depth (e.g. 4 vs 2).  Blocks beyond the new
       depth are skipped.
    5. Old checkpoint contains aliased keys (e.g. ``df_down_blocks.*`` duplicates
       ``df_unet.down_blocks.*``); we de-duplicate on the canonical prefix.  The
       new model's ``state_dict()`` also has aliases (``df_down_blocks.*`` mirrors
       ``df_unet.down_blocks.*``); loading a canonical key populates the alias
       because they share the same underlying tensor.
    6. Old has no ``gated_attention`` or ``qk_norm`` layers; those stay at init.

    Args:
        old_sd: The state_dict loaded from the old checkpoint.
        new_model: A freshly constructed current-architecture model instance.
        encoder_only: Only load encoder path (patch_embed -> down_blocks -> middle)
                      plus cond_embed, vel_pe, vspace_attn, and cross-attention
                      mixing layers.  Decoder / unpatch weights are skipped.
        verbose: Print detailed mapping report.

    Returns:
        A dict suitable for ``new_model.load_state_dict(result, strict=False)``.
    """
    new_sd = new_model.state_dict()
    new_keys = set(new_sd.keys())
    mapped = {}
    skipped_shape = []
    skipped_depth = []
    skipped_cond_type = []
    skipped_decoder = []
    skipped_missing = []
    rpb_buffer_skipped = []

    # -------------------------------------------------------------------
    # helpers: detect which new-model keys are aliases of other keys
    # (point to the same underlying tensor)
    # -------------------------------------------------------------------
    _ptr_to_keys = {}
    for k, v in new_sd.items():
        ptr = v.data_ptr()
        _ptr_to_keys.setdefault(ptr, []).append(k)

    def _is_alias_covered(key):
        """True if *key* shares storage with another key already in *mapped*."""
        ptr = new_sd[key].data_ptr()
        for sibling in _ptr_to_keys.get(ptr, []):
            if sibling != key and sibling in mapped:
                return True
        return False

    # -------------------------------------------------------------------
    # 1. Canonicalize old keys: drop aliased short-form prefixes
    #    (df_down_blocks.* -> df_unet.down_blocks.*, etc.)
    # -------------------------------------------------------------------
    canonical = {}
    old_alias_prefixes = [
        ("df_down_blocks.", "df_unet.down_blocks."),
        ("df_up_blocks.", "df_unet.up_blocks."),
        ("phi_down_blocks.", "phi_unet.down_blocks."),
        ("phi_up_blocks.", "phi_unet.up_blocks."),
    ]
    for old_key in old_sd:
        is_alias = False
        for short, long in old_alias_prefixes:
            if old_key.startswith(short):
                long_key = long + old_key[len(short) :]
                if long_key in old_sd:
                    is_alias = True
                    break
        if not is_alias:
            canonical[old_key] = old_sd[old_key]

    # -------------------------------------------------------------------
    # 2. Build key transformation
    # -------------------------------------------------------------------
    def _remap_key(old_key):
        """Apply all key-renaming rules, returning the new key."""
        k = old_key

        # 2a. cond_embed -> enc_cond_embed
        #     When the new model aliases all three, loading enc_cond_embed is enough.
        for unet_prefix in ("df_unet.", "phi_unet."):
            old_ce = unet_prefix + "cond_embed."
            if k.startswith(old_ce):
                suffix = k[len(old_ce) :]
                return unet_prefix + "enc_cond_embed." + suffix
        if k.startswith("cond_embed."):
            return "enc_cond_embed." + k[len("cond_embed.") :]

        # 2b. RPB: hoist cpb_mlp / rpb / rpb_idx into RPB sub-module
        if ".attn.cpb_mlp." in k:
            k = k.replace(".attn.cpb_mlp.", ".attn.rpb.cpb_mlp.")
        if ".attn.rpb_idx" in k and ".attn.rpb.rpb_idx" not in k:
            k = k.replace(".attn.rpb_idx", ".attn.rpb.rpb_idx")
        # Bare .attn.rpb (the buffer, not a sub-key of .attn.rpb.*)
        if ".attn.rpb" in k and ".attn.rpb." not in k:
            idx = k.rindex(".attn.rpb")
            k = k[:idx] + ".attn.rpb.rpb"

        # 2c. FiLM conditioning.N.modulation -> keep as-is when new model uses FiLM,
        #     or remap to blocks.N.dit.modulation when new model uses DiT.
        #     Try the FiLM path first; fall through to DiT remap if absent.
        if ".conditioning." in k and ".modulation." in k:
            # The FiLM key already matches the new FiLM layout.
            # If the new model uses DiT instead, the FiLM key won't exist,
            # so we also compute the DiT-remapped key as a fallback.
            film_key = k  # already correct for FiLM
            # Build DiT fallback:
            parts = k.split(".conditioning.")
            prefix = parts[0]
            rest = parts[1]
            rest_parts = rest.split(".", 1)
            block_idx = rest_parts[0]
            param_suffix = rest_parts[1]
            dit_key = prefix + ".blocks." + block_idx + ".dit." + param_suffix

            if film_key in new_keys:
                return film_key
            return dit_key

        return k

    # -------------------------------------------------------------------
    # 3. Scope control (encoder-only)
    # -------------------------------------------------------------------
    decoder_prefixes = (
        "df_unet.up_blocks.",
        "df_unet.unpatch.",
        "phi_unet.up_blocks.",
        "phi_unet.unpatch.",
    )

    # -------------------------------------------------------------------
    # 4. Apply mapping
    # -------------------------------------------------------------------
    for old_key, old_val in canonical.items():
        new_key = _remap_key(old_key)
        if new_key is None:
            skipped_missing.append(old_key)
            continue

        # Skip decoder if encoder_only
        if encoder_only and any(new_key.startswith(p) for p in decoder_prefixes):
            skipped_decoder.append(old_key)
            continue

        # Check existence in new model
        if new_key not in new_keys:
            # RPB buffers / attn_mask may be non-persistent -> skip gracefully
            if ".rpb.rpb" in new_key or ".rpb.rpb_idx" in new_key or ".attn_mask" in new_key:
                rpb_buffer_skipped.append(f"{old_key} -> {new_key}")
                continue
            skipped_missing.append(f"{old_key} -> {new_key} (not in new model)")
            continue

        # Shape check
        new_shape = new_sd[new_key].shape
        old_shape = old_val.shape
        if old_shape != new_shape:
            if ".dit.modulation." in new_key:
                skipped_cond_type.append(
                    f"{old_key} -> {new_key}: " f"FiLM {tuple(old_shape)} vs DiT {tuple(new_shape)}"
                )
            elif ".blocks." in new_key:
                skipped_depth.append(
                    f"{old_key} -> {new_key}: {tuple(old_shape)} vs {tuple(new_shape)}"
                )
            else:
                skipped_shape.append(
                    f"{old_key} -> {new_key}: {tuple(old_shape)} vs {tuple(new_shape)}"
                )
            continue

        mapped[new_key] = old_val

    # -------------------------------------------------------------------
    # 5. Report
    # -------------------------------------------------------------------
    if verbose:
        mapped_keys = set(mapped.keys())
        # Count how many new-model keys are covered (directly or via alias)
        n_covered = sum(1 for k in new_keys if k in mapped_keys or _is_alias_covered(k))
        n_new = len(new_keys)
        print(
            f"Checkpoint remapping: {len(mapped)} keys mapped, "
            f"{n_covered}/{n_new} new-model params covered "
            f"({n_covered/max(n_new,1)*100:.1f}%)"
        )
        if skipped_cond_type:
            print(f"\n  FiLM -> DiT conditioning mismatch ({len(skipped_cond_type)} keys):")
            for s in skipped_cond_type[:8]:
                print(f"    {s}")
            if len(skipped_cond_type) > 8:
                print(f"    ... and {len(skipped_cond_type)-8} more")
        if skipped_depth:
            print(f"\n  Depth mismatch ({len(skipped_depth)} keys):")
            for s in skipped_depth[:5]:
                print(f"    {s}")
        if skipped_shape:
            print(f"\n  Shape mismatch ({len(skipped_shape)} keys):")
            for s in skipped_shape[:10]:
                print(f"    {s}")
        if rpb_buffer_skipped:
            print(
                f"\n  RPB/mask buffers skipped (non-persistent in new model): "
                f"{len(rpb_buffer_skipped)} keys"
            )
        if skipped_decoder and encoder_only:
            print(f"\n  Decoder keys skipped (encoder_only=True): {len(skipped_decoder)}")

        uncovered = sorted(k for k in new_keys if k not in mapped_keys and not _is_alias_covered(k))
        if uncovered:
            print(f"\n  New model keys left at init ({len(uncovered)}):")
            for k in uncovered[:15]:
                print(f"    {k}")
            if len(uncovered) > 15:
                print(f"    ... and {len(uncovered)-15} more")

    return mapped


COND_META_MAP = {"itg": "ion_temp_grad", "dg": "density_grad", "s_hat": "s_hat", "q": "q"}

# gyaradax params attribute name for each conditioning key
COND_PARAM_MAP = {"itg": "rlt", "dg": "rln", "s_hat": "shat", "q": "q"}


def cond_from_params(params, cond_keys):
    """Extract conditioning vector from gyaradax params object.

    Returns 1-D numpy array of shape (len(cond_keys),).
    """
    return np.array([float(getattr(params, COND_PARAM_MAP[k])) for k in cond_keys])


def build_train_cond_index(runner, verbose=True):
    """Build normalised conditioning matrix from the training set.

    Applies training_cond_filters from the runner config and returns a dict
    that can be passed to :func:`find_nearest_nn`.

    Returns
    -------
    dict with keys:
        cond_keys   : list[str]
        train_conds : (N, C) raw conditioning values
        train_conds_norm : (N, C) std-normalised conditioning
        train_conds_std  : (1, C) per-key std used for normalisation
        train_file_ids   : list[int] — f_id in runner.trainset for each row
    """
    from omegaconf import OmegaConf

    cond_keys = sorted(runner.cfg.model.conditioning)
    cfg_filters = OmegaConf.to_container(
        runner.cfg.dataset.training_cond_filters or {}, resolve=True
    )
    cond_thresh = runner.cfg.dataset.offset if runner.cfg.dataset.offset > 0 else 80

    def _passes_filter(meta):
        for filter_key, rng in cfg_filters.items():
            parts = filter_key.split("_", 1)
            where, field = (parts[0], parts[1]) if len(parts) == 2 else (None, parts[0])
            if field not in meta:
                continue
            val = meta[field]
            if field == "flux":
                val = float(np.mean(val[:cond_thresh] if where == "first" else val[-cond_thresh:]))
            rng = [rng] if not isinstance(rng[0], (list, tuple)) else rng
            if not any(lo <= val <= hi for lo, hi in rng):
                return False
        return True

    n_files = len(runner.trainset.metadata)
    valid_ids = [f_id for f_id in range(n_files) if _passes_filter(runner.trainset.metadata[f_id])]

    train_conds = np.zeros((len(valid_ids), len(cond_keys)))
    train_file_ids = []
    for row, f_id in enumerate(valid_ids):
        meta = runner.trainset.metadata[f_id]
        for j, k in enumerate(cond_keys):
            train_conds[row, j] = float(np.squeeze(meta[COND_META_MAP[k]]))
        train_file_ids.append(f_id)

    train_conds_std = train_conds.std(axis=0, keepdims=True) + 1e-8
    train_conds_norm = train_conds / train_conds_std

    if verbose:
        print(f"Training trajectories: {n_files} total, {len(valid_ids)} pass flux filter")
        print(f"  filters: {cfg_filters}")
        print(f"  cond keys ({len(cond_keys)}): {cond_keys}")

    return dict(
        cond_keys=cond_keys,
        train_conds=train_conds,
        train_conds_norm=train_conds_norm,
        train_conds_std=train_conds_std,
        train_file_ids=train_file_ids,
    )


def ic_single_diffusion(runner, params, cond_keys, n_steps=10):
    """Sample one diffusion IC, return denormalised raw model-space array."""
    cond = torch.tensor(
        [cond_from_params(params, cond_keys)],
        dtype=torch.float32,
        device=runner.device,
    )
    runner.model.eval()
    with torch.no_grad():
        df = runner.sample(cond, latent_only=False, steps=n_steps)["df"][0].cpu().numpy()
    s, sh = runner.trainset._get_scale_shift(0, "df", torch.tensor(df))
    return df * s.numpy() + sh.numpy()


def ic_repr_diffusion(
    runner, params, cond_keys, geometry, pre, state_init, n_samples=8, n_steps=10, verbose=True
):
    """Sample N diffusion candidates; return the one whose initial eflux is closest
    to the batch mean. Returns (df_model, info_dict)."""
    import dataclasses
    import jax.numpy as jnp
    from gyaradax.solver import GKState, mode_amplitude
    from gyaradax.simulate import _compute_phi_for_init
    from gyaradax.integrals import get_integrals
    from gyaradax.diag import get_diagnostics

    params_jax = dataclasses.replace(params, backend="jax", mixed_precision=True)
    nky = len(geometry["krho"])
    t_start = float(state_init.time)
    cond = torch.tensor(
        [cond_from_params(params, cond_keys)],
        dtype=torch.float32,
        device=runner.device,
    )

    dfs, efluxes = [], []
    runner.model.eval()
    with torch.no_grad():
        for i in range(n_samples):
            df = runner.sample(cond, latent_only=False, steps=n_steps)["df"][0].cpu().numpy()
            s, sh = runner.trainset._get_scale_shift(0, "df", torch.tensor(df))
            df = df * s.numpy() + sh.numpy()
            df_spec = from_model_space(df)
            phi0 = _compute_phi_for_init(df_spec, geometry, params_jax)
            amp0 = mode_amplitude(phi0, geometry, params_jax.norm_eps)
            state = GKState(
                time=jnp.array(t_start, dtype=jnp.float64),
                step=jnp.array(0, dtype=jnp.int32),
                accumulated_norm_factor=jnp.ones(nky, dtype=jnp.float64),
                window_start_amp=amp0,
                last_growth_rate=jnp.zeros(nky, dtype=jnp.float64),
            )
            phi, fluxes = get_integrals(
                df_spec,
                geometry,
                params=params_jax,
                pre=pre,
                adiabatic_electrons=params_jax.adiabatic_electrons,
            )
            diags = get_diagnostics(phi, fluxes, state)
            efluxes.append(float(diags["eflux"]))
            dfs.append(df)
            if verbose:
                print(f"    [{i+1}/{n_samples}] eflux={efluxes[-1]:.4e}")

    mean_eflux = float(np.mean(efluxes))
    best_idx = int(np.argmin(np.abs(np.array(efluxes) - mean_eflux)))
    if verbose:
        print(f"    mean={mean_eflux:.4e}, repr_idx={best_idx} (eflux={efluxes[best_idx]:.4e})")
    return dfs[best_idx], dict(
        efluxes=efluxes, mean_eflux=mean_eflux, best_idx=best_idx, n_samples=n_samples
    )


def plot_method_comparison(results, methods=None, method_styles=None):
    """Flux + ky-spectrum comparison across warmstart methods.

    Parameters
    ----------
    results : dict[iteration] -> {log_gt, ref_mean, ref_std, methods: {name: {log_warm, ttc_flux, ...}}}
    methods : list[str] | None
        Which method names to plot (default: all present in the first iteration).
    method_styles : dict[name] -> dict of matplotlib kwargs (color, ls, lw, ...)
    """
    default_styles = {
        "diffusion": dict(color="#9c27b0", ls="-.", lw=1.1, label_prefix="diffusion (single)"),
        "repr_diffusion": dict(color="#2196f3", ls="-", lw=1.4, label_prefix="repr-diffusion"),
        "nn": dict(color="#e76f51", ls="--", lw=1.1, label_prefix="NN train"),
    }
    styles = {**default_styles, **(method_styles or {})}

    iterations = list(results.keys())
    if methods is None:
        methods = list(results[iterations[0]]["methods"].keys())

    n = len(iterations)
    fig, axes = plt.subplots(n, 2, figsize=(14, 4 * n), squeeze=False)

    for row, it in enumerate(iterations):
        res = results[it]
        lg = res["log_gt"]
        t = lg["time"]
        ref_mean, ref_std = res["ref_mean"], res["ref_std"]

        # flux panel
        ax = axes[row, 0]
        ax.plot(t, lg["eflux"], "k", lw=0.8, alpha=0.45, label="GT")
        for m in methods:
            if m not in res["methods"]:
                continue
            mres = res["methods"][m]
            st = styles.get(m, dict(lw=1.1, label_prefix=m))
            lbl = f"{st.get('label_prefix', m)}  TTC={mres['ttc_flux']:.1f}"
            ax.plot(
                t,
                mres["log_warm"]["eflux"],
                color=st.get("color"),
                ls=st.get("ls", "-"),
                lw=st.get("lw", 1.1),
                label=lbl,
            )
        ax.axhspan(ref_mean - ref_std, ref_mean + ref_std, color="k", alpha=0.07, label="ref ±1σ")
        ax.axhline(ref_mean, color="k", ls=":", lw=0.8)
        ax.set_title(f"iter {it}: flux")
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.15)
        ax.set_xlabel(r"time $[v_{th}/R]$")

        # ky spectrum (final snapshot)
        ax = axes[row, 1]
        ky_gt = np.log10(np.maximum(lg["ky_spec"][-1], 1e-30))
        ax.plot(ky_gt, "k", lw=1, alpha=0.55, label="GT")
        for m in methods:
            if m not in res["methods"]:
                continue
            mres = res["methods"][m]
            st = styles.get(m, dict(lw=1.1, label_prefix=m))
            ky = np.log10(np.maximum(mres["log_warm"]["ky_spec"][-1], 1e-30))
            ax.plot(
                ky,
                color=st.get("color"),
                ls=st.get("ls", "-"),
                lw=st.get("lw", 1.1),
                label=st.get("label_prefix", m),
            )
        ax.set_title(f"iter {it}: $k_y$ spectrum (final)")
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.15)
        ax.set_xlabel("$k_y$ mode")

    fig.tight_layout()
    return fig


def find_nearest_nn(
    params, cond_index, runner, saturated_phase_start=120, seed=42, iteration=0, verbose=True
):
    """Find nearest-neighbour training trajectory and sample a saturated-phase IC.

    Parameters
    ----------
    params : gyaradax params object
    cond_index : dict returned by :func:`build_train_cond_index`
    runner : diffusion runner (for trainset access)
    saturated_phase_start : minimum absolute h5 timestep index for IC sampling
    seed : base RNG seed (combined with iteration)
    iteration : evaluation iteration (for seed offset and printing)

    Returns
    -------
    dict with keys:
        df_model : np.ndarray — raw model-space IC (not normalised)
        nn_file  : str — path to the NN training file
        nn_cond  : np.ndarray — conditioning of the NN file
        nn_dist  : float — normalised L2 distance
        orig_t   : int — original h5 timestep index of the sampled IC
        flat_idx : int — flat index in runner.trainset
    """
    cond_keys = cond_index["cond_keys"]
    eval_cond = cond_from_params(params, cond_keys).reshape(1, -1)
    eval_cond_norm = eval_cond / cond_index["train_conds_std"]
    dists = np.linalg.norm(cond_index["train_conds_norm"] - eval_cond_norm, axis=1)

    nn_row = int(np.argmin(dists))
    nn_idx = cond_index["train_file_ids"][nn_row]
    nn_file = runner.trainset.files[nn_idx]
    nn_cond = cond_index["train_conds"][nn_row]

    # Collect flat indices in the saturated phase
    offset = runner.trainset.offsets[nn_idx]
    sat_t_min = max(0, saturated_phase_start - offset)
    sat_flat_indices = [
        flat_idx
        for flat_idx, (fi, ti) in runner.trainset.flat_index_to_file_and_tstep.items()
        if fi == nn_idx and ti >= sat_t_min
    ]
    assert (
        len(sat_flat_indices) > 0
    ), f"No saturated-phase samples for file {nn_idx} (offset={offset}, sat_t_min={sat_t_min})"

    rng = np.random.default_rng(seed=seed + iteration)
    chosen_flat_idx = int(rng.choice(sat_flat_indices))
    chosen_fi, chosen_ti = runner.trainset.flat_index_to_file_and_tstep[chosen_flat_idx]
    orig_ti = chosen_ti + offset

    sample = runner.trainset.__getitem__(
        chosen_flat_idx, get_normalized=False, override_latens=True
    )
    df_model = sample.df.numpy()

    if verbose:
        print(f"  NN file   : {os.path.basename(nn_file)}")
        print(f"  NN cond   : {dict(zip(cond_keys, nn_cond.round(4)))}")
        print(f"  L2 dist (normalised): {dists[nn_row]:.4f}")
        print(f"  Sampled t_index={chosen_ti} (original h5 index={orig_ti})")

    return dict(
        df_model=df_model,
        nn_file=nn_file,
        nn_cond=nn_cond,
        nn_dist=float(dists[nn_row]),
        orig_t=orig_ti,
        flat_idx=chosen_flat_idx,
    )


def collect_latents(precomputed_latents, cond_keys):
    """Extract latent vectors, flux, and conditioning from precomputed latents dict."""
    lats, flux, cond = [], [], []
    for (fi, ti), s in precomputed_latents.items():
        lats.append(np.array(s["x"]).reshape(-1))
        flux.append(float(np.squeeze(s["flux"])))
        cond.append(np.array([float(np.squeeze(s[k])) for k in cond_keys]))
    return np.stack(lats), np.array(flux), np.stack(cond)


def generate_latents(runner, conditions, batch_size=32, steps=10):
    """Generate latent vectors from conditioning using the diffusion model."""
    import torch
    from tqdm import tqdm

    gen_lats = []
    runner.model.eval()
    for i in tqdm(range(0, len(conditions), batch_size), desc="generating latents"):
        c = torch.tensor(conditions[i : i + batch_size], dtype=torch.float32, device=runner.device)
        with torch.no_grad():
            z = runner.sample(c, latent_only=True, steps=steps)
            gen_lats.append(z.cpu().numpy().reshape(z.shape[0], -1))
    return np.concatenate(gen_lats)


def fit_probes(X_ae, X_gen, y, cond_keys, cond_targets, n_components=256, alpha=1.0):
    """Fit PCA + Ridge probes on AE and gen latents separately.

    Returns dict with pca, probes, predictions, and RMSE for flux and each cond key.
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error

    print(f"fitting PCA ({n_components} components)...")
    pca = PCA(n_components=min(n_components, X_ae.shape[0], X_ae.shape[1])).fit(X_ae)
    X_ae_pca = pca.transform(X_ae)
    X_gen_pca = pca.transform(X_gen)

    print("fitting ridge probes...")
    probe_ae = Ridge(alpha=alpha).fit(X_ae_pca, y)
    probe_gen = Ridge(alpha=alpha).fit(X_gen_pca, y)
    pred_ae = probe_ae.predict(X_ae_pca)
    pred_gen = probe_gen.predict(X_gen_pca)
    rmse_ae = np.sqrt(mean_squared_error(y, pred_ae))
    rmse_gen = np.sqrt(mean_squared_error(y, pred_gen))

    cond_probe_ae = Ridge(alpha=alpha).fit(X_ae_pca, cond_targets)
    cond_probe_gen = Ridge(alpha=alpha).fit(X_gen_pca, cond_targets)
    cond_pred_ae = cond_probe_ae.predict(X_ae_pca)
    cond_pred_gen = cond_probe_gen.predict(X_gen_pca)
    cond_rmse = {}
    for j, k in enumerate(cond_keys):
        cond_rmse[k] = {
            "ae": np.sqrt(mean_squared_error(cond_targets[:, j], cond_pred_ae[:, j])),
            "gen": np.sqrt(mean_squared_error(cond_targets[:, j], cond_pred_gen[:, j])),
        }

    return {
        "pca": pca,
        "flux": {
            "probe_ae": probe_ae,
            "probe_gen": probe_gen,
            "pred_ae": pred_ae,
            "pred_gen": pred_gen,
            "rmse_ae": rmse_ae,
            "rmse_gen": rmse_gen,
        },
        "cond": {
            "probe_ae": cond_probe_ae,
            "probe_gen": cond_probe_gen,
            "pred_ae": cond_pred_ae,
            "pred_gen": cond_pred_gen,
            "rmse": cond_rmse,
        },
    }


def plot_probes(y_true, probes, cond_keys, cond_targets, X_ae, X_gen):
    """Plot probe scatter plots (flux + cond + PCA) in a 2-row grid."""
    from sklearn.decomposition import PCA

    n_cond = len(cond_keys)
    ncols = max(3, (n_cond + 2 + 1) // 2)
    fig, axes = plt.subplots(2, ncols, figsize=(5 * ncols, 8))
    axes = axes.flatten()

    fl = probes["flux"]
    ax = axes[0]
    ax.scatter(y_true, fl["pred_ae"], s=8, alpha=0.3, label=f"ae RMSE={fl['rmse_ae']:.3f}")
    ax.scatter(
        y_true,
        fl["pred_gen"],
        s=12,
        alpha=0.5,
        marker="x",
        label=f"gen RMSE={fl['rmse_gen']:.3f}",
    )
    lim = [y_true.min(), y_true.max()]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set(xlabel="GT flux", ylabel="predicted", title="flux probe")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.15)

    cd = probes["cond"]
    for j, k in enumerate(cond_keys):
        ax = axes[1 + j]
        ax.scatter(
            cond_targets[:, j],
            cd["pred_ae"][:, j],
            s=8,
            alpha=0.3,
            label=f"ae RMSE={cd['rmse'][k]['ae']:.3f}",
        )
        ax.scatter(
            cond_targets[:, j],
            cd["pred_gen"][:, j],
            s=12,
            alpha=0.5,
            marker="x",
            label=f"gen RMSE={cd['rmse'][k]['gen']:.3f}",
        )
        lim = [cond_targets[:, j].min(), cond_targets[:, j].max()]
        ax.plot(lim, lim, "k--", lw=0.8)
        ax.set(xlabel=f"GT {k}", ylabel="predicted", title=f"{k} probe")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.15)

    pca2d = PCA(n_components=2).fit(X_ae)
    z_ae = pca2d.transform(X_ae)
    z_gen = pca2d.transform(X_gen)
    ax = axes[1 + n_cond]
    ax.scatter(z_ae[:, 0], z_ae[:, 1], s=8, alpha=0.3, label="ae")
    ax.scatter(z_gen[:, 0], z_gen[:, 1], s=12, alpha=0.4, marker="x", label="gen")
    ax.set(xlabel="PC1", ylabel="PC2", title="latent PCA")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.15)

    for i in range(2 + n_cond, len(axes)):
        axes[i].set_visible(False)
    fig.tight_layout()
    return fig


def encode_valset(valset, autoencoder, cond_keys, device, batch_size=32):
    """Encode validation set through AE in batches. Returns (latents, conditions, flux_gt, file_indices)."""
    flat_idx_map = {(f, t): idx for idx, (f, t) in valset.flat_index_to_file_and_tstep.items()}
    dfs, conds_t, cond_all, flux_gt, fi_list = [], [], [], [], []

    from tqdm import tqdm

    print(f"loading {len(valset.files)} val trajectories...")
    for fi in range(len(valset.files)):
        meta = valset.metadata[fi]
        gt_flux = float(np.mean(meta["flux"][-80:]))
        cond_vals = [float(np.squeeze(meta[COND_META_MAP.get(k, k)])) for k in cond_keys]
        for t_idx in range(valset.file_num_samples[fi]):
            if (fi, t_idx) not in flat_idx_map:
                continue
            sample = valset[flat_idx_map[(fi, t_idx)]]
            if sample.df is None:
                continue
            dfs.append(sample.df)
            conds_t.append(
                sample.conditioning
                if sample.conditioning is not None
                else torch.zeros(len(cond_keys))
            )
            cond_all.append(cond_vals)
            flux_gt.append(gt_flux)
            fi_list.append(fi)
    print(f"  collected {len(dfs)} samples")

    lats = []
    autoencoder.eval()
    for i in tqdm(range(0, len(dfs), batch_size), desc="encoding val through AE"):
        batch_df = torch.stack(dfs[i : i + batch_size]).to(device)
        batch_cond = torch.stack(conds_t[i : i + batch_size]).to(device)
        with torch.no_grad():
            z, _ = autoencoder.encode(batch_df, condition=batch_cond)
        lats.append(z.cpu().numpy().reshape(z.shape[0], -1))

    return np.concatenate(lats), np.array(cond_all), np.array(flux_gt), fi_list


def plot_val_probe(valset, pred_ae, pred_gen, y_gt, fi_list, rmse_ae, rmse_gen):
    """Per-trajectory errorbar plot for validation probe results."""
    fi_labels = {}
    for fi in sorted(set(fi_list)):
        m = re.search(r"iteration_(\d+)", valset.files[fi])
        fi_labels[fi] = f"iter_{m.group(1)}" if m else f"f{fi}"

    per_file = {}
    for i, fi in enumerate(fi_list):
        lbl = fi_labels[fi]
        if lbl not in per_file:
            per_file[lbl] = {"ae": [], "gen": [], "gt": y_gt[i]}
        per_file[lbl]["ae"].append(pred_ae[i])
        per_file[lbl]["gen"].append(pred_gen[i])

    labels = sorted(per_file.keys())
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.2), 5))
    x_pos = np.arange(len(labels))
    w = 0.25
    for j, lbl in enumerate(labels):
        d = per_file[lbl]
        ax.errorbar(
            j - w / 2,
            np.mean(d["ae"]),
            yerr=np.std(d["ae"]),
            fmt="o",
            capsize=5,
            color="#1f77b4",
            mfc="white",
            mew=1.5,
            alpha=0.8,
        )
        ax.errorbar(
            j + w / 2,
            np.mean(d["gen"]),
            yerr=np.std(d["gen"]),
            fmt="s",
            capsize=5,
            color="#ff7f0e",
            mfc="white",
            mew=1.5,
            alpha=0.8,
        )
        ax.scatter(j, d["gt"], marker="x", s=80, color="#d62728", zorder=3)
    ax.errorbar(
        [],
        [],
        fmt="o",
        color="#1f77b4",
        mfc="white",
        mew=1.5,
        label=f"ae RMSE={rmse_ae:.3f}",
    )
    ax.errorbar(
        [],
        [],
        fmt="s",
        color="#ff7f0e",
        mfc="white",
        mew=1.5,
        label=f"gen RMSE={rmse_gen:.3f}",
    )
    ax.scatter([], [], marker="x", s=80, color="#d62728", label="gt flux")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set(xlabel="trajectory", ylabel="flux", title="flux probe on validation")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3, ls="--")
    fig.tight_layout()
    return fig


def plot_correlation_grid(extended_metrics, metric_names=None, x_axes=None):
    if metric_names is None:
        metric_names = [
            ("rel_err", "rel model-space error"),
            ("probe_flux_rmse", "probing flux rmse"),
            ("integral_flux_err", "integral flux rel error"),
            ("kyspec_rmse_init", "ky-spec rmse (log, init)"),
            ("kxspec_rmse_init", "kx-spec rmse (log, init)"),
            ("kyspec_pearson_init", "ky-spec pearson (init)"),
        ]
    if x_axes is None:
        x_axes = [
            ("fid", "gyroswin-fid"),
            ("ttc", r"ttc $[v_{th}/R]$"),
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
                    0.5,
                    0.5,
                    "insufficient data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=10,
                    color="gray",
                )
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)
                continue
            xv, yv = x_vals[mask], y_vals[mask]
            ax.scatter(xv, yv, s=60, zorder=3, edgecolors="k", linewidth=0.5)
            for xi, yi, it in zip(xv, yv, np.array(iters)[mask]):
                ax.annotate(
                    str(it),
                    (xi, yi),
                    textcoords="offset points",
                    xytext=(6, 4),
                    fontsize=8,
                    color="gray",
                )
            if mask.sum() >= 3:
                coeffs = np.polyfit(xv, yv, 1)
                x_fit = np.linspace(xv.min(), xv.max(), 50)
                ax.plot(x_fit, np.polyval(coeffs, x_fit), "r--", lw=1, alpha=0.7)
                r, p = pearsonr(xv, yv)
                ax.text(
                    0.05,
                    0.95,
                    f"r={r:.2f}, p={p:.2f}",
                    transform=ax.transAxes,
                    fontsize=9,
                    va="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                )
            ax.set_xlabel(x_label if row == n_rows - 1 else "")
            ax.set_ylabel(y_label)
            ax.grid(True, alpha=0.15)
            if row < n_rows - 1:
                ax.set_xticklabels([])

    axes[0, 0].set_title("vs gyroswin-fid", fontweight="bold", fontsize=12)
    if n_cols > 1:
        axes[0, 1].set_title("vs ttc", fontweight="bold", fontsize=12)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Shared aggregate-metric API used by pinc_generate / gyroswin_generate /
# diff_generate_table notebooks. One source of truth so the ID/OOD tables
# from each notebook are 1-to-1 comparable.
#
# Convention (matches the existing _agg_metrics_avg / avg_flux_rmse pattern):
#   per traj :  pred_mean = gen[key]['all'].mean(axis=0)        # collapse axis 0
#               diff      = pred_mean - gt[key]['mean']
#               sq_err    = diff ** 2                            # scalar or (D,)
#   pool     :  concat sq_err across trajectories
#   final    :  RMSE = sqrt(mean(pool))
#   R^2      :  per traj  1 - bias^2 / Var(gt full series); mean across trajs
#
# Both `gen[key]` and `gt[key]` may be either a dict ({all,mean,std[,full]})
# or a raw tensor / ndarray -- see `_extract_pred` / `_extract_gt` below.
# ---------------------------------------------------------------------------


def _to_tensor(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.float()
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)


def _extract_pred(gen_entry):
    """Predictions: (N, ...) where axis 0 is the sample / time axis."""
    if gen_entry is None:
        return None
    if isinstance(gen_entry, dict):
        return _to_tensor(gen_entry.get("all"))
    return _to_tensor(gen_entry)


def _extract_gt_mean(gt_entry):
    """Per-trajectory reference: scalar or (D,)."""
    if gt_entry is None:
        return None
    if isinstance(gt_entry, dict):
        return _to_tensor(gt_entry.get("mean"))
    return _to_tensor(gt_entry)


def _extract_gt_full(gt_entry):
    """Full GT time series for SS_tot in R^2; falls back to ['all']."""
    if gt_entry is None or not isinstance(gt_entry, dict):
        return None
    return _to_tensor(gt_entry.get("full", gt_entry.get("all")))


def aggregate_metric(group, get_pred, get_gt, get_gt_full=None):
    """Core: avg_flux_rmse-style aggregation over a {traj: {gen, gt}} group.

    Args:
        group: dict mapping trajectory name -> {'gen': ..., 'gt': ...}.
        get_pred(gen_dict) -> tensor (N, ...) | None      predictions per traj
        get_gt(gt_dict)    -> tensor (...)    | None      per-traj reference
        get_gt_full(gt_dict) -> tensor (T,...) | None     for R^2 SS_tot

    Returns:
        {"RMSE": float, "R2": float, "n_trajs": int}
    """
    sq_errs = []
    r2_vals = []
    for res in group.values():
        gen, gt = res.get("gen", {}), res.get("gt", {})
        pred = get_pred(gen)
        target = get_gt(gt)
        if pred is None or target is None:
            continue
        pred_mean = pred.mean(dim=0)
        diff = (pred_mean - target).flatten()
        sq_errs.append(diff.pow(2))

        if get_gt_full is not None:
            gt_full = get_gt_full(gt)
            if gt_full is not None:
                ss_tot = (gt_full - gt_full.mean()).pow(2).sum()
                if ss_tot > 0:
                    r2_vals.append(1.0 - diff.pow(2).sum() / ss_tot)

    if not sq_errs:
        return {"RMSE": float("nan"), "R2": float("nan"), "n_trajs": 0}

    rmse = float(torch.cat(sq_errs).mean().sqrt())
    r2 = (
        float(torch.stack([torch.as_tensor(r) for r in r2_vals]).mean())
        if r2_vals
        else float("nan")
    )
    return {"RMSE": rmse, "R2": r2, "n_trajs": len(sq_errs)}


def _resolve_key(k):
    """`k` is either a string (use it for both pred and gt) or a tuple
    (pred_key, gt_key). Returns (pred_key, gt_key, column_name)."""
    if isinstance(k, tuple):
        return k[0], k[1], k[0]
    return k, k, k


def print_aggregate_metrics(
    group,
    label=None,
    scalar_keys=("eflux",),
    spec_keys=("kxspec", "kyspec"),
    extra_keys=(),
):
    """Compute + print avg_flux_rmse-style metrics for the standard layout
    used by all three ID/OOD notebooks.

    Each entry in `scalar_keys` / `spec_keys` / `extra_keys` is either a string
    (used as both pred_key and gt_key) or a (pred_key, gt_key) tuple. The
    tuple form is for probe predictions whose name differs from the GT field
    (e.g. ('probe_flux_ae', 'eflux'), or ('probe_kyspec_ae', 'meta_kyspec_mean')).

    For each key:
      * predictions  = gen[pred_key]['all']  (or gen[pred_key] if a raw array)
      * gt reference = gt[gt_key]['mean']    (or gt[gt_key] if a raw value)
      * R^2 SS_tot   = gt[gt_key]['full']    (or gt[gt_key]['all'] as fallback,
                                              only computed for scalar_keys)

    Returns dict {f"{pred_key}_RMSE": float, "{pred_key}_R2": float (scalar only), ...}.
    """
    if label:
        print(f"\n{'=' * 60}")
        print(f"  {label}  (avg_flux_rmse aggregate, n_trajs={len(group)})")
        print(f"{'=' * 60}")

    out = {}
    for is_scalar, keys in (
        (True, scalar_keys),
        (False, spec_keys),
        (False, extra_keys),
    ):
        for k in keys:
            pred_key, gt_key, col = _resolve_key(k)
            m = aggregate_metric(
                group,
                get_pred=lambda gen, p=pred_key: _extract_pred(gen.get(p)),
                get_gt=lambda gt, g=gt_key: _extract_gt_mean(gt.get(g)),
                get_gt_full=(
                    (lambda gt, g=gt_key: _extract_gt_full(gt.get(g))) if is_scalar else None
                ),
            )
            out[f"{col}_RMSE"] = m["RMSE"]
            if is_scalar:
                out[f"{col}_R2"] = m["R2"]
            if label:
                r2_str = f"  R2={m['R2']:.6g}" if is_scalar and m["n_trajs"] else ""
                print(
                    f"  {col:>20s}  RMSE = {m['RMSE']:.6g}" f"   (n_trajs={m['n_trajs']}){r2_str}"
                )
    return out
