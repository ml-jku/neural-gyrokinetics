"""Warm-restart evaluation utilities.

Splits out of `neurips_diff_eval.py` the pieces concerned with running
GT-vs-warm gyrokinetic trajectories and scoring the resulting stationary
distributions. Three buckets:

1. **Trajectory drivers** — `run_trajectories`, `run_trajectory_pair`,
   `run_trajectory` (gyaradax / GKW solver wrappers).
2. **Reference loaders + window helpers** — `load_reference_flux*`,
   `stationary_window`, `stationary_window_2d`.
3. **Stationary-distribution divergences** between warm and reference
   samples in the window:
   * Two-sample tests: `flux_ks_pvalue`, `flux_ad_statistic`,
     `flux_wasserstein`, `sliced_wasserstein`, `flux_mmd_rbf`,
     `flux_energy_distance`, `flux_cdf_linf`, `flux_cdf_l1`.
   * MCMC-mixing: `gelman_rubin_R`, `gelman_rubin_t_curve`.
   * Time-structure / random-walk consistency:
     `flux_autocorr`, `flux_struct_fn`, `autocorr_l1`.
   * Spectra-specific window comparisons (operate on `(T, K)` arrays):
     `spec_divergence` (per-mode aggregate), `mean_log_spectrum_pearson`,
     `mean_log_spectrum_l2`, `time_avg_spectrum_kl`,
     `spec_cosine_curve`.
   * Driver: `compute_distribution_divergences`.

Also kept here for legacy use: `time_to_convergence` (TTC) — superseded
by the divergence-based metrics above (paper §5.2 lines 630–631 explains
why the running-mean band threshold is degenerate under a good warm-start).
"""
from __future__ import annotations

import os

import numpy as np
from scipy.stats import pearsonr


# ---------------------------------------------------------------------------
# Trajectory drivers
# ---------------------------------------------------------------------------

def _resolved_spectra_jax(phi, df, gt):
    """ky/kx-resolved kxspec, kyspec, qspec from gyaradax `phi` and `df`.

    Mirrors `neugk.pinc.generate.compute_spectra` (which produced `meta["kyspec"]`
    / `meta["fluxspec"]` at preprocessing time), so results land in meta units
    without the 16× gyaradax/meta rescale.

    Inputs:
        phi : jnp.ndarray, shape (s, kx, ky)
        df  : jnp.ndarray, shape (vpar, mu, s, kx, ky)  (adiabatic, 5D)
              or (nsp, vpar, mu, s, kx, ky) for kinetic
        gt  : dict from `gyaradax.integrals.geom_tensors(geometry, params=params)`
              (or the cached `pre["geom_tensors"]`).

    Returns: dict with numpy arrays
        "kxspec" (nkx,), "kyspec" (nky,), "qspec" (nky,) [or (nsp, nky) kinetic]
    """
    import jax.numpy as jnp
    from einops import rearrange

    # |phi|^2 spectra (sum over parallel + the orthogonal mode axis)
    phi_abs2 = jnp.abs(phi) ** 2
    if phi.ndim == 3:                     # (s, kx, ky)
        kxspec = jnp.sum(phi_abs2, axis=(0, 2))
        kyspec = jnp.sum(phi_abs2, axis=(0, 1))
    else:
        raise ValueError(f"unexpected phi.ndim={phi.ndim}")

    # eflux per-mode — replicates gyaradax.calculate_fluxes math up to (but
    # not including) the final scalar sum.
    bn       = gt["bn"]
    parseval = gt["parseval"]
    ints     = gt["ints"]
    efun     = gt["efun"]
    krho     = gt["krho"]
    bessel   = gt["bessel"]
    vpgr     = gt["vpgr"]
    mugr     = gt["mugr"]
    intvp    = gt["intvp"]
    intmu    = gt["intmu"]
    d2X      = gt["d2X"]

    if df.ndim == 5:                      # adiabatic
        phi_expanded = rearrange(phi, "s x y -> 1 1 s x y")
    elif df.ndim == 6:                    # kinetic
        phi_expanded = rearrange(phi, "s x y -> 1 1 1 s x y")
    else:
        raise ValueError(f"unexpected df.ndim={df.ndim}")
    phi_gyro = bessel * phi_expanded

    dum  = parseval * ints * (efun * krho) * df
    dum1 = dum * jnp.conj(phi_gyro)
    dum2 = dum1 * bn
    d3v  = d2X * intmu * bn * intvp
    eflux_field = d3v * (vpgr ** 2 * jnp.imag(dum1) + 2.0 * mugr * jnp.imag(dum2))

    # `parseval`/`bessel` are 6-D in gyaradax, so `eflux_field` is 6-D even for
    # 5-D `df`. Sum over all axes except the last (ky) — robust to either
    # shape — and drop any singleton dims left over.
    sum_axes = tuple(range(eflux_field.ndim - 1))
    qspec = jnp.squeeze(jnp.sum(eflux_field, axis=sum_axes))

    return {
        "kxspec": np.asarray(kxspec),
        "kyspec": np.asarray(kyspec),
        "qspec":  np.asarray(qspec),
    }


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
    log_resolved_spectra=True,
):
    """Run GT once and multiple warm-start trajectories. Returns
    (log_gt, [log_warm, ...])."""
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

    dfs = [df_gt] + list(df_warms)
    states = [_make_state(df) for df in dfs]
    ntraj = 1 + n_warm
    log_keys = ["time", "kx_spec", "ky_spec", "eflux"]
    if log_resolved_spectra:
        log_keys += ["fluxspec"]
    logs = [{k: [] for k in log_keys} for _ in range(ntraj)]

    gt_cached = pre["geom_tensors"] if (pre is not None and "geom_tensors" in pre) else None
    if log_resolved_spectra and gt_cached is None:
        from gyaradax.integrals import geom_tensors as _gt
        gt_cached = _gt(geometry, params=params)

    def _log_step(b, phi, fluxes, df_now):
        diags = get_diagnostics(phi, fluxes, states[b])
        logs[b]["time"].append(np.array(diags["time"]))
        logs[b]["eflux"].append(np.array(diags["eflux"]))
        if log_resolved_spectra:
            res = _resolved_spectra_jax(phi, df_now, gt_cached)
            logs[b]["kx_spec"].append(res["kxspec"])
            logs[b]["ky_spec"].append(res["kyspec"])
            logs[b]["fluxspec"].append(res["qspec"])
        else:
            logs[b]["kx_spec"].append(np.array(diags["kx_spec"]))
            logs[b]["ky_spec"].append(np.array(diags["ky_spec"]))

    for b in range(ntraj):
        phi, fluxes = get_integrals(
            dfs[b], geometry, params=params, pre=pre,
            adiabatic_electrons=params.adiabatic_electrons,
        )
        _log_step(b, phi, fluxes, dfs[b])

    steps_done = 0
    while steps_done < n_steps:
        n = min(chunk_size, n_steps - steps_done)
        for b in range(ntraj):
            dfs[b], (phi, fluxes), states[b] = gksolve(
                dfs[b], geometry, params, states[b], n_steps=n, pre=pre,
            )
        steps_done += n
        if steps_done % log_every == 0 or steps_done == n_steps:
            for b in range(ntraj):
                phi_b, fluxes_b = get_integrals(
                    dfs[b], geometry, params=params, pre=pre,
                    adiabatic_electrons=params.adiabatic_electrons,
                )
                _log_step(b, phi_b, fluxes_b, dfs[b])
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
    df_gt, df_warm, geometry, params, pre, state_init,
    n_steps=1000, label="", chunk_size=1, backend="cuda",
    mixed_precision=True, print_every=500, log_every=1,
    log_resolved_spectra=True,
):
    """Run GT and a single warm-start trajectory."""
    log_gt, log_warms = run_trajectories(
        df_gt, [df_warm], geometry, params, pre, state_init,
        n_steps=n_steps, labels=[label], chunk_size=chunk_size,
        backend=backend, mixed_precision=mixed_precision,
        print_every=print_every, log_every=log_every,
        log_resolved_spectra=log_resolved_spectra,
    )
    return log_gt, log_warms[0]


def run_trajectory(
    df_init, geometry, params, pre, state_init,
    n_steps=1000, label="", chunk_size=1, backend="cuda",
    mixed_precision=True, print_every=500, log_every=1,
    log_resolved_spectra=True,
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
    log_keys = ["time", "kx_spec", "ky_spec", "eflux"]
    if log_resolved_spectra:
        log_keys += ["fluxspec"]
    log = {k: [] for k in log_keys}

    gt_cached = pre["geom_tensors"] if (pre is not None and "geom_tensors" in pre) else None
    if log_resolved_spectra and gt_cached is None:
        from gyaradax.integrals import geom_tensors as _gt
        gt_cached = _gt(geometry, params=params)

    def _log_step(phi, fluxes, df_now, state):
        diags = get_diagnostics(phi, fluxes, state)
        log["time"].append(np.array(diags["time"]))
        log["eflux"].append(np.array(diags["eflux"]))
        if log_resolved_spectra:
            res = _resolved_spectra_jax(phi, df_now, gt_cached)
            log["kx_spec"].append(res["kxspec"])
            log["ky_spec"].append(res["kyspec"])
            log["fluxspec"].append(res["qspec"])
        else:
            log["kx_spec"].append(np.array(diags["kx_spec"]))
            log["ky_spec"].append(np.array(diags["ky_spec"]))

    phi, fluxes = get_integrals(
        df_init, geometry, params=params, pre=pre,
        adiabatic_electrons=params.adiabatic_electrons,
    )
    _log_step(phi, fluxes, df_init, state)

    steps_done = 0
    while steps_done < n_steps:
        n = min(chunk_size, n_steps - steps_done)
        df_init, (phi, fluxes), state = gksolve(
            df_init, geometry, params, state, n_steps=n, pre=pre,
        )
        steps_done += n
        if steps_done % log_every == 0 or steps_done == n_steps:
            _log_step(phi, fluxes, df_init, state)
        if steps_done % print_every == 0 or steps_done == n_steps:
            print(f"  [{label}] {steps_done}/{n_steps}  t={float(state.time):.3f}  "
                  f"Q={float(log['eflux'][-1]):.4e}")
    out = {k: np.array(v) for k, v in log.items()}
    return out


# ---------------------------------------------------------------------------
# Reference loaders + window helpers
# ---------------------------------------------------------------------------

def load_reference_flux(gkw_dir, iteration):
    """Heat-flux mean/std over the last 240 steps of GKW fluxes.dat."""
    samples = load_reference_flux_samples(gkw_dir, iteration)
    return float(np.mean(samples)), float(np.std(samples))


def load_reference_flux_samples(gkw_dir, iteration, n_tail=240):
    """Raw GKW heat-flux samples (last `n_tail` rows of fluxes.dat)."""
    path = os.path.join(gkw_dir, f"iteration_{iteration}", "fluxes.dat")
    data = np.loadtxt(path)
    eflux = data[:, 1]
    return np.asarray(eflux[-n_tail:], dtype=np.float64)


def stationary_window(x, frac=0.5):
    """Last `frac` fraction of a 1-D series (post-saturation window)."""
    x = np.asarray(x).ravel()
    if x.size == 0:
        return x
    k = max(1, int(round(frac * x.size)))
    return x[-k:]


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


# ---------------------------------------------------------------------------
# Two-sample distributional divergences (1-D scalar — flux)
# ---------------------------------------------------------------------------

def flux_ks_pvalue(x_warm, x_ref):
    """KS two-sample p-value (higher = more indistinguishable). Paper $\\tau_Q$."""
    from scipy.stats import ks_2samp
    return float(ks_2samp(np.asarray(x_warm), np.asarray(x_ref)).pvalue)


def flux_ad_statistic(x_warm, x_ref):
    """Anderson–Darling 2-sample statistic (heavier tail weighting than KS)."""
    from scipy.stats import anderson_ksamp
    try:
        return float(anderson_ksamp(
            [np.asarray(x_warm), np.asarray(x_ref)],
        ).statistic)
    except Exception:
        return np.nan


def flux_wasserstein(x_warm, x_ref):
    """1-D Wasserstein-1 distance."""
    from scipy.stats import wasserstein_distance
    return float(wasserstein_distance(np.asarray(x_warm), np.asarray(x_ref)))


def flux_cdf_linf(x_warm, x_ref):
    """L∞ between empirical CDFs (the KS *statistic*; complements the p-value
    which depends on sample sizes)."""
    from scipy.stats import ks_2samp
    return float(ks_2samp(np.asarray(x_warm), np.asarray(x_ref)).statistic)


def flux_cdf_l1(x_warm, x_ref, n_grid=512):
    """L1 between empirical CDFs evaluated on a shared `n_grid` grid spanning
    the joint sample range. Equivalent to W₁ for sorted scalars but useful
    when you want a calibrated 'CDF area' rather than transport cost."""
    a = np.sort(np.asarray(x_warm).ravel())
    b = np.sort(np.asarray(x_ref).ravel())
    if a.size == 0 or b.size == 0:
        return np.nan
    lo = min(a[0], b[0])
    hi = max(a[-1], b[-1])
    if hi == lo:
        return 0.0
    grid = np.linspace(lo, hi, n_grid)
    Fa = np.searchsorted(a, grid, side="right") / a.size
    Fb = np.searchsorted(b, grid, side="right") / b.size
    return float(np.trapz(np.abs(Fa - Fb), grid))


def flux_mmd_rbf(x_warm, x_ref, sigma=None):
    """MMD² with an RBF kernel, sigma defaults to the median heuristic.

    A two-sample-test alternative that's smoother than KS/AD and
    well-defined for any 1-D distribution (no ties / continuity assumption)."""
    a = np.asarray(x_warm, dtype=np.float64).ravel()
    b = np.asarray(x_ref,  dtype=np.float64).ravel()
    if a.size < 2 or b.size < 2:
        return np.nan
    if sigma is None:
        joint = np.concatenate([a, b])
        d = np.abs(joint[:, None] - joint[None, :])
        sigma = max(np.median(d), 1e-12)
    g = 1.0 / (2 * sigma * sigma)

    def _k(u, v):
        d2 = (u[:, None] - v[None, :]) ** 2
        return np.exp(-g * d2)

    Kxx = _k(a, a); np.fill_diagonal(Kxx, 0.0)
    Kyy = _k(b, b); np.fill_diagonal(Kyy, 0.0)
    Kxy = _k(a, b)
    n, m = a.size, b.size
    return float(
        Kxx.sum() / (n * (n - 1))
        + Kyy.sum() / (m * (m - 1))
        - 2.0 * Kxy.mean()
    )


def flux_mannwhitney_u_p(x_warm, x_ref):
    """Mann–Whitney U two-sample (a.k.a. Wilcoxon rank-sum) p-value.

    Tests stochastic ordering — under the null both samples come from the
    same distribution. Rank-based (no continuity assumption beyond ties)
    so it is robust on heavy-tailed flux histograms where KS can over- or
    under-react. Higher p = more indistinguishable.
    """
    from scipy.stats import mannwhitneyu
    try:
        return float(mannwhitneyu(
            np.asarray(x_warm), np.asarray(x_ref), alternative="two-sided",
        ).pvalue)
    except Exception:
        return np.nan


def flux_mannwhitney_u_stat(x_warm, x_ref):
    """Mann–Whitney U statistic (rank-sum). Lower-bounded by 0; depends on
    sample sizes — pair with the p-value or normalise by `n*m`."""
    from scipy.stats import mannwhitneyu
    try:
        u = mannwhitneyu(
            np.asarray(x_warm), np.asarray(x_ref), alternative="two-sided",
        ).statistic
        return float(u) / (len(x_warm) * len(x_ref))
    except Exception:
        return np.nan


def flux_wilcoxon_signed_rank_p(x_warm, x_ref):
    """Paired Wilcoxon signed-rank p-value. Only meaningful when both samples
    have the same length and are paired in time (e.g. snapshot-by-snapshot).
    Returns NaN otherwise."""
    from scipy.stats import wilcoxon
    a = np.asarray(x_warm).ravel()
    b = np.asarray(x_ref).ravel()
    if a.size != b.size or a.size < 1:
        return np.nan
    try:
        return float(wilcoxon(a, b, zero_method="wilcox", alternative="two-sided").pvalue)
    except Exception:
        return np.nan


def flux_arima_param_l2(x_warm, x_ref, order=(2, 0, 2)):
    """Fit ARIMA(p,d,q) to both samples, return L2 distance between AR + MA + intercept
    coefficients. Robust signal of "do the two stochastic processes share the same
    dynamics" beyond marginal distribution. NaN if fitting fails or arrays too short.
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError:
        return np.nan
    a = np.asarray(x_warm, dtype=np.float64).ravel()
    b = np.asarray(x_ref, dtype=np.float64).ravel()
    p, d, q = order
    min_len = max(p + d + q + 5, 20)
    if a.size < min_len or b.size < min_len:
        return np.nan
    import warnings
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.filterwarnings("ignore", category=UserWarning,
                                     module="statsmodels")
            ma = ARIMA(a, order=order).fit(disp=False).params
            mb = ARIMA(b, order=order).fit(disp=False).params
    except Exception:
        return np.nan
    n = min(len(ma), len(mb))
    return float(np.linalg.norm(ma[:n] - mb[:n]))


def flux_r2_histogram(x_warm, x_ref, bins=32):
    """R² between histogram bin counts of the two samples on a shared edge grid.
    Captures point-wise distributional fidelity (high = same shape; ≤ 0 = worse than
    predicting the mean). Robust to scale via density=True normalisation."""
    a = np.asarray(x_warm, dtype=np.float64).ravel()
    b = np.asarray(x_ref, dtype=np.float64).ravel()
    if a.size < 4 or b.size < 4:
        return np.nan
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    if hi == lo:
        return 1.0 if np.allclose(a.mean(), b.mean()) else 0.0
    edges = np.linspace(lo, hi, bins + 1)
    ha, _ = np.histogram(a, bins=edges, density=True)
    hb, _ = np.histogram(b, bins=edges, density=True)
    ss_res = float(np.sum((hb - ha) ** 2))
    ss_tot = float(np.sum((hb - hb.mean()) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-30)


def spec_r2_meanlog(X_warm, X_ref, eps=1e-30):
    """R² between time-averaged log-spectra across modes. The point-wise spectral
    analogue of `mean_log_spectrum_pearson`: 1 = perfectly reconstructed shape,
    0 = no better than predicting the mean across modes."""
    a = np.log10(np.maximum(np.asarray(X_warm).mean(axis=0), eps))
    b = np.log10(np.maximum(np.asarray(X_ref).mean(axis=0),  eps))
    if a.size < 2:
        return np.nan
    ss_res = float(np.sum((b - a) ** 2))
    ss_tot = float(np.sum((b - b.mean()) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-30)


def flux_energy_distance(x_warm, x_ref):
    """Energy distance E²(P, Q) = 2·E|X−Y| − E|X−X'| − E|Y−Y'|.

    Two-sample test that is metric on probability measures with finite first
    moment; cheap O(n·m) but stable on heavy tails."""
    a = np.asarray(x_warm, dtype=np.float64).ravel()
    b = np.asarray(x_ref,  dtype=np.float64).ravel()
    if a.size < 2 or b.size < 2:
        return np.nan
    cross = np.abs(a[:, None] - b[None, :]).mean()
    aa    = np.abs(a[:, None] - a[None, :]).mean()
    bb    = np.abs(b[:, None] - b[None, :]).mean()
    return float(max(2 * cross - aa - bb, 0.0))


# ---------------------------------------------------------------------------
# Paper-spec τ_Q: first-passage convergence times
# (paper §5.2 eq for τ_Q = min{t : p_KS({Q(s)}_{s≤t}, {Q_GKW}) ≥ α}, α = 0.05)
# ---------------------------------------------------------------------------

def _first_passage_prefix(
    x_warm, x_ref, time_axis, predicate, *, t_min=8, log_every=2,
):
    """Walk prefixes of `x_warm` (lengths t_min, t_min·log_every, ...) and
    return the smallest `time_axis[t-1] - time_axis[0]` at which
    `predicate(prefix, x_ref)` holds. inf if never satisfied within the run.

    `time_axis` is the absolute time array aligned with `x_warm`."""
    n = min(len(x_warm), len(time_axis))
    t = t_min
    while t <= n:
        prefix = np.asarray(x_warm)[:t]
        if predicate(prefix, x_ref):
            return float(time_axis[t - 1] - time_axis[0])
        nxt = t * log_every if t * log_every > t else t + 1
        t = nxt if nxt <= n else n + 1
    return np.inf


def tau_q_ks(x_warm, x_ref, time_axis, alpha=0.05, **kw):
    """Paper's primary τ_Q (KS): first prefix-time at which the KS p-value
    against the GKW reference is ≥ alpha."""
    return _first_passage_prefix(
        x_warm, x_ref, time_axis,
        lambda a, b: flux_ks_pvalue(a, b) >= alpha, **kw,
    )


def tau_q_ad(x_warm, x_ref, time_axis, threshold=2.5, **kw):
    """τ_Q variant using Anderson–Darling: first prefix-time at which the AD
    statistic drops below `threshold` (≈ 2.5 corresponds to p ≳ 0.05)."""
    return _first_passage_prefix(
        x_warm, x_ref, time_axis,
        lambda a, b: flux_ad_statistic(a, b) <= threshold, **kw,
    )


def tau_q_wasserstein(x_warm, x_ref, time_axis, threshold=None, **kw):
    """τ_Q variant using Wasserstein-1: first prefix-time at which W1 drops
    below `threshold`. If threshold is None we use 0.1·std(x_ref) as a
    self-calibrating choice."""
    if threshold is None:
        threshold = 0.1 * float(np.std(x_ref) + 1e-12)
    return _first_passage_prefix(
        x_warm, x_ref, time_axis,
        lambda a, b: flux_wasserstein(a, b) <= threshold, **kw,
    )


def tau_q_gelman_rubin(x_warm, x_ref, time_axis, threshold=1.1, **kw):
    """τ_Q via Gelman–Rubin: first prefix-time at which R̂ < threshold (1.1)."""
    return _first_passage_prefix(
        x_warm, x_ref, time_axis,
        lambda a, b: gelman_rubin_R(a, b) < threshold, **kw,
    )


def tau_q_mannwhitney(x_warm, x_ref, time_axis, alpha=0.05, **kw):
    """τ_Q via Mann–Whitney rank-sum (Wilcoxon): first prefix-time at which the
    test fails to reject (p ≥ alpha)."""
    return _first_passage_prefix(
        x_warm, x_ref, time_axis,
        lambda a, b: flux_mannwhitney_u_p(a, b) >= alpha, **kw,
    )


# ---------------------------------------------------------------------------
# MCMC-mixing diagnostics
# ---------------------------------------------------------------------------

def gelman_rubin_R(x_warm, x_ref):
    """Two-chain Gelman–Rubin $\\hat R$ on the full window. <1.1 ≈ converged."""
    chains = [np.asarray(x_warm, dtype=np.float64), np.asarray(x_ref, dtype=np.float64)]
    n = min(len(c) for c in chains)
    if n < 2:
        return np.nan
    chains = np.stack([c[-n:] for c in chains])
    means = chains.mean(axis=1)
    grand = means.mean()
    B = n * np.sum((means - grand) ** 2) / (chains.shape[0] - 1)
    W = np.mean(np.var(chains, axis=1, ddof=1))
    if W <= 0:
        return np.nan
    var_hat = (n - 1) / n * W + B / n
    return float(np.sqrt(var_hat / W))


def gelman_rubin_t_curve(x_warm, x_ref):
    """$\\hat R(t)$ on prefixes of length 8, 16, ... up to min(len)."""
    n = min(len(x_warm), len(x_ref))
    ts, Rs = [], []
    t = 8
    while t <= n:
        Rs.append(gelman_rubin_R(np.asarray(x_warm)[-t:], np.asarray(x_ref)[-t:]))
        ts.append(t)
        t = min(n, t * 2) if t * 2 <= n else n + 1
    return np.asarray(ts), np.asarray(Rs)


# ---------------------------------------------------------------------------
# Time-structure / random-walk consistency
# ---------------------------------------------------------------------------

def flux_autocorr(x, max_lag=40):
    """Centered autocorrelation $C_Q(\\tau)/C_Q(0)$, $\\tau = 0, \\dots, $`max_lag`."""
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
    """Second-order structure function $S_2(\\tau)$."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) <= max_lag + 1:
        return np.full(max_lag + 1, np.nan)
    out = np.empty(max_lag + 1)
    for tau in range(max_lag + 1):
        out[tau] = np.mean((x[tau:] - x[: len(x) - tau]) ** 2)
    return out


def autocorr_l1(x_warm, x_ref, max_lag=40, kind="autocorr"):
    """L1 distance between warm/ref autocorr (or structure-fn) curves."""
    fn = flux_autocorr if kind == "autocorr" else flux_struct_fn
    a = fn(x_warm, max_lag)
    b = fn(x_ref, max_lag)
    if np.any(~np.isfinite(a)) or np.any(~np.isfinite(b)):
        return np.nan
    return float(np.mean(np.abs(a - b)))


# ---------------------------------------------------------------------------
# Multivariate divergences and spectra-window comparisons
# ---------------------------------------------------------------------------

def sliced_wasserstein(X_warm, X_ref, n_projections=64, seed=0):
    """Sliced Wasserstein-1 by averaging 1-D W₁ over `n_projections` random
    unit directions."""
    from scipy.stats import wasserstein_distance
    rng = np.random.default_rng(seed)
    X_warm = np.asarray(X_warm)
    X_ref = np.asarray(X_ref)
    if X_warm.ndim == 1:
        X_warm = X_warm[:, None]
        X_ref = X_ref[:, None]
    d = X_warm.shape[1]
    P = rng.standard_normal((d, n_projections))
    P /= np.linalg.norm(P, axis=0, keepdims=True) + 1e-12
    proj_w = X_warm @ P
    proj_r = X_ref @ P
    vals = [
        wasserstein_distance(proj_w[:, i], proj_r[:, i])
        for i in range(n_projections)
    ]
    return float(np.mean(vals))


def _per_mode_apply(metric_fn, X_warm, X_ref):
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
    "mmd_rbf": flux_mmd_rbf,
    "energy": flux_energy_distance,
}


def spec_divergence(X_warm, X_ref, kind="w1"):
    """Per-mode 1-D divergence aggregated over a vector spectrum.

    Returns ``{"per_mode", "mean", "median", "frac_indistinguishable"}``.
    The last field is the fraction of modes with KS p-value ≥ 0.05 — only
    populated when `kind == "ks"`.
    """
    if kind not in _SPEC_METRIC_TABLE:
        raise ValueError(
            f"unknown kind: {kind!r}; choose from {list(_SPEC_METRIC_TABLE)}"
        )
    per_mode = _per_mode_apply(_SPEC_METRIC_TABLE[kind], X_warm, X_ref)
    finite = per_mode[np.isfinite(per_mode)]
    return {
        "per_mode": per_mode,
        "mean":     float(np.mean(finite))   if finite.size else np.nan,
        "median":   float(np.median(finite)) if finite.size else np.nan,
        "frac_indistinguishable": (
            float(np.mean(per_mode >= 0.05)) if kind == "ks" else np.nan
        ),
    }


def mean_log_spectrum_pearson(X_warm, X_ref, eps=1e-30):
    """Pearson correlation between time-averaged log-spectra over the window.

    A direct generalisation of the spectrum-Pearson check used in TTC, but
    averaged over the whole window rather than evaluated snapshot-by-snapshot.
    """
    a = np.log10(np.maximum(np.asarray(X_warm).mean(axis=0), eps))
    b = np.log10(np.maximum(np.asarray(X_ref).mean(axis=0),  eps))
    if a.size < 2:
        return np.nan
    r, _ = pearsonr(a, b)
    return float(r)


def mean_log_spectrum_l2(X_warm, X_ref, eps=1e-30):
    """L2 distance between time-averaged log-spectra (lower = closer)."""
    a = np.log10(np.maximum(np.asarray(X_warm).mean(axis=0), eps))
    b = np.log10(np.maximum(np.asarray(X_ref).mean(axis=0),  eps))
    return float(np.linalg.norm(a - b) / np.sqrt(a.size))


def time_avg_spectrum_kl(X_warm, X_ref, eps=1e-30):
    """KL(p_warm || p_ref) of time-averaged spectra treated as discrete
    probability distributions over modes (normalised to sum 1)."""
    a = np.asarray(X_warm).mean(axis=0)
    b = np.asarray(X_ref).mean(axis=0)
    a = np.clip(a, eps, None); a = a / a.sum()
    b = np.clip(b, eps, None); b = b / b.sum()
    return float(np.sum(a * (np.log(a) - np.log(b))))


def spec_cosine_curve(X_warm, X_ref, eps=1e-30):
    """Mean cosine similarity over snapshots between log-spectra. Higher
    is closer; a `cold` start with a transient yields ≪1 early then climbs.
    For a window-only evaluation we just average across the window."""
    a = np.log10(np.maximum(np.asarray(X_warm), eps))
    b = np.log10(np.maximum(np.asarray(X_ref),  eps))
    if a.shape != b.shape or a.ndim != 2:
        return np.nan
    num = (a * b).sum(axis=-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12
    return float(np.mean(num / den))


def running_mean_drift(x_warm, x_ref):
    """|⟨x_warm⟩ − ⟨x_ref⟩| / |⟨x_ref⟩| — calibrated relative drift in the
    first moment. The simplest possible window comparison; useful as a
    baseline (the metric TTC was originally trying to capture)."""
    a = float(np.mean(x_warm))
    b = float(np.mean(x_ref))
    return float(abs(a - b) / max(abs(b), 1e-12))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

# Canonical 6-metric set: applied uniformly to flux + each spec_key. These
# are the ONLY metrics the driver emits — extra helpers above remain available
# for direct use but are deliberately excluded from the default output.
CANONICAL_METRICS = {
    "w1":       flux_wasserstein,
    "mmd":      flux_mmd_rbf,
    "ks_p":     flux_ks_pvalue,
    "ad":       flux_ad_statistic,
    "arima_l2": flux_arima_param_l2,
    "r2_hist":  flux_r2_histogram,
}


def compute_distribution_divergences(
    log_run, log_gt, *, ref_flux_samples=None, warm_frac=0.95,
    max_lag=40, spec_keys=("ky_spec", "fluxspec"),
):
    """Divergences between a warm-started run and the GT saturated reference.

    Two-sample distributional tests on each mode's time-series: the (short)
    warm trajectory and the (long) GT saturated tail are both treated as
    samples from a stationary distribution; per-mode KS/AD/W1/MMD score
    whether they come from the same one. Sample counts may differ.

    Warm side: drop the leading `1 - warm_frac` (default 5%); keep the rest. The GT side is provided
    *pre-sliced* in `log_gt` (and `ref_flux_samples` for the flux), so the
    caller picks the right tail (e.g. last 80 GKW snapshots for spectra,
    last 240 fluxes.dat rows for flux).

    Parameters
    ----------
    log_run : dict
        Output of `run_trajectory` / `run_trajectory_pair` for the warm
        run. Must contain ``eflux`` and at least one of `spec_keys`.
    log_gt : dict
        GT saturated-state reference. Each `spec_keys[i]` entry is a (T, K)
        array of the *already tail-sliced* GT snapshots (we do not re-slice
        it here). May also include "eflux" as a fallback when
        `ref_flux_samples` is None.
    ref_flux_samples : np.ndarray or None
        GT reference flux samples (e.g. `load_reference_flux_samples`,
        last 240 rows of fluxes.dat). If None, falls back to
        `log_gt["eflux"]` taken as-is.
    warm_frac : float
        Stationary-window fraction of the *warm* run (default 0.5 =
        second half).
    spec_keys : tuple of str
        Spectra fields to compute divergences for.

    Returns
    -------
    dict
        Flat metric dict.
    """
    out = {}
    flux_warm = stationary_window(log_run["eflux"], warm_frac)
    if ref_flux_samples is not None:
        flux_ref = np.asarray(ref_flux_samples)
    elif "eflux" in log_gt:
        flux_ref = np.asarray(log_gt["eflux"]).ravel()
    else:
        flux_ref = None

    if flux_ref is not None:
        for name, fn in CANONICAL_METRICS.items():
            try:
                out[f"flux_{name}"] = float(fn(flux_warm, flux_ref))
            except Exception as e:
                out[f"flux_{name}"] = np.nan
                out[f"flux_{name}_error"] = repr(e)

    for spec_key in spec_keys:
        if spec_key not in log_run or spec_key not in log_gt:
            continue
        Xw = stationary_window_2d(np.asarray(log_run[spec_key]), warm_frac)
        Xr = np.asarray(log_gt[spec_key])  # caller already tail-sliced
        if Xw.size == 0 or Xr.size == 0:
            continue
        for name, fn in CANONICAL_METRICS.items():
            try:
                if name == "r2_hist":
                    # spectra: use R² of mean-log spectrum across modes instead
                    # of histogram (more meaningful for a vector spectrum).
                    out[f"{spec_key}_r2_meanlog"] = spec_r2_meanlog(Xw, Xr)
                else:
                    per_mode = _per_mode_apply(fn, Xw, Xr)
                    finite = per_mode[np.isfinite(per_mode)]
                    out[f"{spec_key}_{name}_mean"] = (
                        float(np.mean(finite)) if finite.size else np.nan
                    )
            except Exception as e:
                k = f"{spec_key}_{'r2_meanlog' if name == 'r2_hist' else name + '_mean'}"
                out[k] = np.nan
                out[f"{k}_error"] = repr(e)
    return out


# ---------------------------------------------------------------------------
# Legacy: time-to-convergence (paper §5.2 calls this degenerate; keep for
# back-compat / sanity rows).
# ---------------------------------------------------------------------------

def time_to_convergence(
    log_warm, log_gt, time_axis,
    window=50, flux_n_std=3.0, flux_threshold=0.6, spec_threshold=0.95,
    verbose=True, ref_flux_mean=None, ref_flux_std=None,
):
    """Legacy ±nσ band TTC for flux + ky-spectrum Pearson. Kept for the
    pre-divergence FID-vs-TTC scatter; new metrics live in
    `compute_distribution_divergences`."""
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
        ky_g = np.log10(np.maximum(log_gt["ky_spec"][i],   1e-30))
        ky_corr[i] = pearsonr(ky_w, ky_g)[0] if len(ky_w) > 1 else 0.0

    def _rolling_ttc(signal, threshold, win):
        for i in range(len(signal) - win):
            if np.mean(signal[i : i + win]) >= threshold:
                return float(time_axis[i] - time_axis[0])
        return np.inf

    ttc_flux = _rolling_ttc(flux_in_band, flux_threshold, effective_window)
    ttc_spec = _rolling_ttc(ky_corr,      spec_threshold, effective_window)

    if verbose:
        n_in = int(flux_in_band.sum())
        mean_corr = float(np.mean(ky_corr)) if n_spec > 0 else 0.0
        print(
            f"    ttc flux: gt_mean={gt_flux_mean:.3e}, gt_std={gt_flux_std:.3e}, "
            f"band=±{flux_n_std:.0f}σ ({flux_band:.3e}), "
            f"in_band={n_in}/{n} ({n_in/max(n,1):.0%}), "
            f"window={effective_window}, ttc={ttc_flux:.3f}"
        )
        print(f"    ttc spec: mean_r(ky)={mean_corr:.3f}, "
              f"window={effective_window}, ttc={ttc_spec:.3f}")
    return {"flux": ttc_flux, "ky_spec": ttc_spec}
