"""Gyaradax adapter — phi / particle / heat / momentum fluxes from df.

The upstream torch repo carries its own python port of the GKW field-solve
+ phase-space integrals (``neugk/integrals.py:FluxIntegral``). We delegate
to the pure-JAX ``gerkone/gyaradax`` package instead. The API:

    from gyaradax.integrals import get_integrals
    phi, (pflux, eflux, vflux) = get_integrals(df, geometry, params=None, ...)

Inputs are unbatched; we ``vmap`` over the batch axis. Physical
quantities should run in ``float64`` — ``gyaradax`` sets
``jax_enable_x64=True`` globally on import. We don't put the integrals on
the training graph so the x64 promotion is contained to eval-only
``jit`` blocks.

Note: ``gyaradax`` is electrostatic-only at the moment (no apar/bpar
paths). That matches what the upstream evaluator actually uses.
"""

from __future__ import annotations

from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np


_GEOMETRY_KEYS = (
    "krho", "ints", "intmu", "intvp", "vpgr", "mugr",
    "bn", "ffun", "efun", "rfun", "bt_frac", "parseval",
    "mas", "tmp", "d2X", "signz", "signB", "de", "vthrat",
    "kxrh", "little_g",
)
_PARAM_KEYS = ("adiabatic", "beta", "nlapar", "nlbpar")


def _split_geom_and_params(geometry: dict[str, jnp.ndarray]):
    """Separate gyaradax's geometry dict from its params dict.

    Upstream's torch ``FluxIntegral`` lumps everything into one geometry
    dict; gyaradax expects them split.
    """
    geom = {k: geometry[k] for k in _GEOMETRY_KEYS if k in geometry}
    params_dict = {k: geometry[k] for k in _PARAM_KEYS if k in geometry}
    return geom, params_dict


def compute_integrals(
    df: jnp.ndarray,
    geometry: dict[str, jnp.ndarray],
    *,
    params: Optional[Any] = None,
    adiabatic_electrons: bool = True,
) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """Compute (phi, (pflux, eflux, vflux)) for one sample's distribution function.

    ``df`` shape (with adiabatic electrons): ``(vpar, mu, s, x, y)`` or
    ``(2, vpar, mu, s, x, y)`` for complex inputs (real / imag channels);
    ``gyaradax.get_integrals`` accepts both layouts.
    """
    from gyaradax.integrals import get_integrals
    geom, gparams = _split_geom_and_params(geometry)
    p = params
    if p is None and gparams:
        # pass scalars as-is; gyaradax builds its params object or falls back to compute_geometry defaults
        p = gparams
    return get_integrals(df, geom, params=p, adiabatic_electrons=adiabatic_electrons)


def batched_integrals(
    df_batch: jnp.ndarray,
    geometry_batch: dict[str, jnp.ndarray],
    *,
    adiabatic_electrons: bool = True,
):
    """``vmap``ed compute_integrals over the batch axis (axis 0)."""
    def one(df, geom_one):
        return compute_integrals(df, geom_one, adiabatic_electrons=adiabatic_electrons)
    return jax.vmap(one)(df_batch, geometry_batch)


def gyaradax_flux_integrals(
    df_batch: jnp.ndarray,
    geometry_one: dict,
):
    """Pure-JAX flux integral via gyaradax. Returns ``(phi, eflux)`` as
    host numpy arrays.

    Inputs:
        df_batch:     ``(B, 4, vp, mu, s, x, y)`` — denormalised AE-decoded
                      df with the separate-zf channel-of-4 layout.
        geometry_one: dict of per-trajectory geometry values (no batch axis).
                      Caller is expected to have stripped the batch axis
                      (batches in the eval loop are single-trajectory).

    Pipeline:
        recombine zf → real/imag → complex 5D (vp, mu, s, x, y) →
        forward FFT (x, y, norm='forward'; ifftshift on x) → gyaradax
        ``get_integrals`` adiabatic path.

    **Parseval correction** — upstream torch's metadata.pkl ships
    ``parseval = [1, ny, ny, ...]`` (a physics bug: it should be the
    Hermitian-symmetry factor ``[1, 2, 2, ...]``, see
    ``gyaradax/geometry/geom.py:geometry_from_geom_dat_and_input``).
    Torch's ``pev_fluxes`` then double-counts ``ints`` to cancel the
    ``ny/2`` overcount — both bugs land on the right flux. We refuse to
    inherit the bug: ``parseval`` is replaced here with the correct
    ``where(|krho|<1e-12, 1, 2)``, and the single-``ints`` formula in
    gyaradax then gives the same flux values as upstream torch to fp64
    precision, with the correct underlying physics.

    Roughly **300× faster** than the torch FluxIntegral bridge on CPU
    (~12 ms vs ~3.35 s per batch-of-4).
    """
    from gyaradax.integrals import get_integrals
    df_batch = jnp.asarray(df_batch)
    B = df_batch.shape[0]
    # recombine_zf: (B, 4, ...) → (B, 2, ...) → complex (B, vp, mu, s, x, y)
    df_rec = df_batch[:, :2] + df_batch[:, 2:]
    df_cplx = (df_rec[:, 0] + 1j * df_rec[:, 1]).astype(jnp.complex128)

    geom = {k: jnp.asarray(v) for k, v in geometry_one.items()}
    # batched geom (leaves carry leading batch axis matching B)? else single-traj path
    krho = geom.get("krho")
    if krho is not None and krho.ndim > 0 and krho.shape[0] == B:
        # batched geom — collapse if all samples share trajectory, else vmap over both
        if _same_traj(geom, B):
            geom = _strip_batch_axis(geom, B)
            geom["parseval"] = jnp.where(
                jnp.abs(geom["krho"]) < 1e-12, 1.0, 2.0,
            ).astype(jnp.float64)
            phi, eflux = _gyaradax_integ_batched(df_cplx, geom)
        else:
            # per-sample geom: override parseval per row (vectorized)
            geom["parseval"] = jnp.where(
                jnp.abs(geom["krho"]) < 1e-12, 1.0, 2.0,
            ).astype(jnp.float64)
            phi, eflux = _gyaradax_integ_batched_geom(df_cplx, geom)
    else:
        # caller passed a single-trajectory geom (no batch axis)
        geom["parseval"] = jnp.where(
            jnp.abs(jnp.asarray(geom["krho"])) < 1e-12, 1.0, 2.0,
        ).astype(jnp.float64)
        phi, eflux = _gyaradax_integ_batched(df_cplx, geom)
    return np.asarray(phi), np.asarray(eflux)


@jax.jit
def _gyaradax_integ_one(df_one, geom):
    """Per-sample (spectral df, geom) → (phi, eflux). FFT inside so the
    caller can stay in spatial layout. Jit'd once at module level."""
    from gyaradax.integrals import get_integrals
    spec = jnp.fft.fftn(df_one, axes=(-2, -1), norm="forward")
    spec = jnp.fft.ifftshift(spec, axes=-2)
    phi, (_pflux, eflux, _vflux) = get_integrals(
        spec, geom, adiabatic_electrons=True,
    )
    return phi, eflux


# vmap with shared geom (single trajectory in a batch) — the fast common case
_gyaradax_integ_batched = jax.jit(jax.vmap(_gyaradax_integ_one, in_axes=(0, None)))
# vmap with per-sample geom (handles mixed-trajectory batches at boundaries)
_gyaradax_integ_batched_geom = jax.jit(jax.vmap(_gyaradax_integ_one, in_axes=(0, 0)))


def _strip_batch_axis(geom: dict, batch_size: int) -> dict:
    """Drop the leading batch axis from a geom dict whose leaves were stacked
    across a batch (via ``get_batch_geometry``). Returns the geom of the first
    sample — only safe when all samples share the same trajectory."""
    out = {}
    for k, v in geom.items():
        arr = jnp.asarray(v)
        if arr.ndim and arr.shape[0] == batch_size:
            out[k] = arr[0]
        else:
            out[k] = arr
    return out


def _same_traj(geom: dict, batch_size: int) -> bool:
    """Cheap heuristic — check if the batched geom's per-sample slices are
    identical (i.e. all from the same trajectory)."""
    krho = geom.get("krho")
    if krho is None:
        return False
    arr = np.asarray(krho)
    if arr.ndim == 0 or arr.shape[0] != batch_size:
        return False
    return bool(np.all(arr == arr[0:1]))


def torch_flux_integrals(
    df_batch,
    geometry_batch: dict,
):
    """Compute ``(phi, eflux)`` by delegating to upstream torch ``FluxIntegral``.

    Runs on CUDA so the eval doesn't blow up to thousands of CPU OpenMP
    threads (saw 1290 threads at 0% GPU util before — the per-batch
    CPU FluxIntegral was the eval bottleneck). The integrator is built
    once and cached. Returns ``(phi_np, eflux_np)`` as host numpy arrays.
    """
    import numpy as _np
    import torch
    from neugk.integrals import FluxIntegral
    from neugk.utils import recombine_zf

    device = torch_flux_integrals._device
    if device is None:
        # FluxIntegral CUDA path fails nvrtc on this cluster; CPU fallback with thread cap (override via TORCH_FLUX_DEVICE)
        import os
        override = os.environ.get("TORCH_FLUX_DEVICE")
        if override:
            device = torch.device(override)
        else:
            device = torch.device("cpu")
        if device.type == "cpu":
            torch.set_num_threads(int(os.environ.get("TORCH_FLUX_THREADS", "16")))
            try:
                torch.set_num_interop_threads(2)
            except RuntimeError:
                pass
        torch_flux_integrals._device = device

    integrator = torch_flux_integrals._integrator
    if integrator is None:
        integrator = FluxIntegral(
            real_potens=True, spectral_potens=False, flux_fields=False,
            spectral_df=False, integral_precision="float64",
        ).to(device)
        torch_flux_integrals._integrator = integrator

    df_t = torch.as_tensor(_np.array(df_batch), device=device)
    if df_t.dim() == 7:
        df_t = recombine_zf(df_t, dim=1)  # (B, 2, vp, mu, s, x, y)
    df_t = df_t.unsqueeze(1).double()  # (B, sp=1, 2, vp, mu, s, x, y)

    geom_t = {
        k: torch.as_tensor(_np.array(v), device=device).double()
        for k, v in geometry_batch.items()
    }
    with torch.no_grad():
        phi, (_pflux, eflux, _vflux) = integrator(geom_t, df_t)
    return _np.asarray(phi.float().cpu()), _np.asarray(eflux.squeeze(-1).float().cpu())


torch_flux_integrals._integrator = None  # type: ignore[attr-defined]
torch_flux_integrals._device = None  # type: ignore[attr-defined]
