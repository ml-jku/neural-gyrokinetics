"""NeurIPS diffusion evaluation helpers."""

import re
import os
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.stats import pearsonr

from neugk.integrals import FluxIntegral
from neugk.utils import recombine_zf


def to_model_space(df_spectral, separate_zf=True):
    """Complex128 spectral -> float32 model-space (with optional separate_zf)."""
    from neugk.utils import separate_zf as _separate_zf

    df_np = np.fft.ifftshift(np.array(df_spectral), axes=(3,))
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
    return jnp.asarray(np.fft.fftshift(df_spectral, axes=(3,)), dtype=jnp.complex128)


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


@torch.no_grad()
def extract_gyroswin_latents(model, df_batch, device="cuda", condition=None, **kwargs):
    """Bottleneck features from GyroSwin encoder."""
    model.eval()
    x = df_batch.to(device)

    m = model.module if hasattr(model, "module") else model
    unet = m.df_unet if hasattr(m, "df_unet") else m

    embed = unet.cond_embed if hasattr(unet, "cond_embed") else None
    if condition is None:
        raise ValueError("condition must be provided for GyroSwin feature extraction")
    c = condition.to(device)
    if embed is not None:
        if c.shape[-1] < embed.n_cond:
            ts = torch.empty(c.shape[0], embed.n_cond - c.shape[-1], device=device).uniform_(100, 200)
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

    return x.flatten(1).cpu().numpy()


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
    """Run GT and warm-start trajectories, return (log_gt, log_warm)."""
    import jax.numpy as jnp
    import dataclasses
    from gyaradax.solver import GKState, mode_amplitude
    from gyaradax.simulate import _compute_phi_for_init, gksolve
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

    dfs = [df_gt, df_warm]
    states = [_make_state(df_gt), _make_state(df_warm)]
    logs = [{"time": [], "kx_spec": [], "ky_spec": [], "eflux": []} for _ in range(2)]

    for b in range(2):
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
        for b in range(2):
            dfs[b], (phi, fluxes), states[b] = gksolve(dfs[b], geometry, params, states[b], n_steps=n, pre=pre)
        steps_done += n

        if steps_done % log_every == 0 or steps_done == n_steps:
            for b in range(2):
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
    log_gt["df_final"] = np.array(dfs[0])
    log_warm["df_final"] = np.array(dfs[1])
    return log_gt, log_warm


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
    """Heat flux stats from GKW fluxes.dat (last 240 steps)."""
    path = os.path.join(gkw_dir, f"iteration_{iteration}", "fluxes.dat")
    data = np.loadtxt(path)
    eflux = data[:, 1]
    tail = eflux[-80 * 3 :]
    return float(np.mean(tail)), float(np.std(tail))


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
        print(f"    ttc spec: mean_r(ky)={mean_corr:.3f}, " f"window={effective_window}, ttc={ttc_spec:.3f}")

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
                skipped_depth.append(f"{old_key} -> {new_key}: {tuple(old_shape)} vs {tuple(new_shape)}")
            else:
                skipped_shape.append(f"{old_key} -> {new_key}: {tuple(old_shape)} vs {tuple(new_shape)}")
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
            print(f"\n  RPB/mask buffers skipped (non-persistent in new model): " f"{len(rpb_buffer_skipped)} keys")
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


COND_META_MAP = {"itg": "ion_temp_grad", "dg": "density_grad"}


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
            conds_t.append(sample.conditioning if sample.conditioning is not None else torch.zeros(len(cond_keys)))
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
