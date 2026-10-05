"""Zonal-flow / spectral turbulence metrics for validation.

Numpy implementation built on top of the gyaradax integrals adapter.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    # ordinal ranks (spectra are continuous, so no ties), then pearson on the ranks
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return _pearson(ra, rb)


def _wasserstein_1d(u: np.ndarray, v: np.ndarray) -> float:
    # 1D w1 for equal-length, uniform-weight samples == mean|sorted(u)-sorted(v)|
    return float(np.abs(np.sort(u) - np.sort(v)).mean())


def _zonal_profiles(phi_spec: np.ndarray, geom: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """GKW diagnos_zfshear trio from the spectral potential (s, kx, ky).

    zfphi is the flux-surface average (ints weights) of the zonal (ky=0) mode;
    zfflow/zfshear are its first/second radial derivatives (i*kx in spectral x),
    the E x B zonal flow and its shear rate (Dannert & Jenko, PoP 2005).
    """
    ints = np.asarray(geom["ints"], dtype=np.float64).reshape(-1, 1)
    zon = (phi_spec[:, :, 0] * ints).sum(0)  # (kx,) complex
    kx = np.asarray(geom["kxrh"], dtype=np.float64)

    def prof(z):
        return np.fft.ifft(np.fft.ifftshift(z), norm="forward").real

    return {"zfphi": prof(zon), "zfflow": prof(1j * kx * zon), "zfshear": prof(-(kx**2) * zon)}


def diagnostics(phi_fft_: np.ndarray, eflux_field: np.ndarray, ds: float) -> Dict[str, np.ndarray]:
    """Turbulence diagnostics from the potential FFT and the heat-flux field.

    The last three axes of ``phi_fft_`` are ``(nx, *, ny)``; ``kxspec`` sums the y axis,
    ``kyspec`` sums the x (``*``) axis, and both sum the nx axis (the whole field line).
    """
    power = phi_fft_.real**2 + phi_fft_.imag**2
    return {
        "kxspec": power.sum(axis=(-3, -1)) * ds,
        "kyspec": power.sum(axis=(-3, -2)) * ds,
        # heat-flux spectrum: sum everything except the trailing wavenumber axis
        "qspec": (
            eflux_field.sum(axis=tuple(range(eflux_field.ndim - 1)))
            if eflux_field.ndim >= 2
            else eflux_field.sum()
        ),
    }


def spectral_diagnostics(
    df_batch: np.ndarray, geom: Dict[str, np.ndarray], ds: float
) -> List[Dict[str, np.ndarray]]:
    from neugk_jax.evaluate.integrals import gyaradax_spectral_fields

    phi_spec, eflux = gyaradax_spectral_fields(df_batch, geom)
    return _diagnostics_from_fields(phi_spec, eflux, geom, ds)


def _diagnostics_from_fields(phi_spec, eflux, geom, ds) -> List[Dict[str, np.ndarray]]:
    out: List[Dict[str, np.ndarray]] = []
    for b in range(phi_spec.shape[0]):
        d = diagnostics(phi_spec[b], eflux[b], ds=ds)
        d.update(_zonal_profiles(phi_spec[b], geom))
        out.append(d)
    return out


_ZF_KEYS = ("zfphi", "zfflow", "zfshear")


def _rl2(p, g) -> float:
    return float(np.linalg.norm(p - g) / (np.linalg.norm(g) + 1e-12))


def spectral_sums(
    pred_diags: List[Dict[str, np.ndarray]], gt_diags: List[Dict[str, np.ndarray]]
) -> Dict[str, np.ndarray]:
    """Additive per-trajectory statistics of paired snapshot diagnostics."""
    out: Dict[str, np.ndarray] = {"n": np.asarray(float(len(pred_diags)))}
    for key in ("kyspec", "qspec"):
        out[f"{key}_p"] = np.stack([np.asarray(d[key], np.float64) for d in pred_diags]).sum(0)
        out[f"{key}_g"] = np.stack([np.asarray(d[key], np.float64) for d in gt_diags]).sum(0)
    if "zfphi" in pred_diags[0]:
        for key in _ZF_KEYS:
            out[f"{key}_rl2"] = np.asarray(
                sum(_rl2(p[key], g[key]) for p, g in zip(pred_diags, gt_diags))
            )
        out["zf_er"] = np.asarray(
            sum(
                float((p["zfphi"] ** 2).sum() / ((g["zfphi"] ** 2).sum() + 1e-12))
                for p, g in zip(pred_diags, gt_diags)
            )
        )
    return out


def add_spectral_sums(acc: Optional[Dict[str, np.ndarray]], new: Dict[str, np.ndarray]):
    if acc is None:
        return new
    return {k: acc[k] + new[k] for k in acc}


def metrics_from_spectral_sums(sums: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Pearson/Spearman/Wasserstein/L1 on the time-averaged ky and Q spectra, zonal-flow errors."""
    n = float(sums["n"])
    out: Dict[str, float] = {}
    for key in ("kyspec", "qspec"):
        p, g = sums[f"{key}_p"] / n, sums[f"{key}_g"] / n
        out[f"{key}_pc"] = float(_pearson(p, g))
        out[f"{key}_sc"] = float(_spearman(p, g))
        out[f"{key}_l1"] = float(np.abs(p - g).sum())
        out[f"{key}_rl2"] = _rl2(p, g)
        out[f"{key}_rl1"] = float(np.abs(p - g).sum() / (np.abs(g).sum() + 1e-12))
        pn, gn = p / (p.sum() + 1e-12), g / (g.sum() + 1e-12)
        out[f"{key}_wd"] = float(_wasserstein_1d(pn, gn))
    if "zf_er" in sums:
        for key in _ZF_KEYS:
            out[f"{key}_rl2"] = float(sums[f"{key}_rl2"]) / n
        out["zf_energy_err"] = abs(float(sums["zf_er"]) / n - 1)
    return out


def time_averaged_spectral_metrics(
    pred_diags: List[Dict[str, np.ndarray]], gt_diags: List[Dict[str, np.ndarray]]
) -> Dict[str, float]:
    """Spectral metrics of one trajectory's paired snapshot diagnostics."""
    return metrics_from_spectral_sums(spectral_sums(pred_diags, gt_diags))


def spectral_sums_layout(n_ky: int) -> Dict[str, tuple]:
    """Shapes of the :func:`spectral_sums` entries for ``n_ky`` binormal modes."""
    return {
        "n": (),
        "kyspec_p": (n_ky,),
        "kyspec_g": (n_ky,),
        "qspec_p": (n_ky,),
        "qspec_g": (n_ky,),
        **{f"{k}_rl2": () for k in _ZF_KEYS},
        "zf_er": (),
    }


def pack_spectral_store(
    store: Dict[int, Dict[str, np.ndarray]], n_files: int, n_ky: int
) -> np.ndarray:
    """Fixed-shape ``(n_files, D)`` array of per-trajectory sums (zeros where absent)."""
    layout = spectral_sums_layout(n_ky)
    out = np.zeros((n_files, sum(int(np.prod(s)) for s in layout.values())), np.float64)
    for fid, sums in store.items():
        out[fid] = np.concatenate([np.asarray(sums[k], np.float64).reshape(-1) for k in layout])
    return out


def unpack_spectral_store(packed: np.ndarray, n_ky: int) -> Dict[int, Dict[str, np.ndarray]]:
    layout = spectral_sums_layout(n_ky)
    store = {}
    for fid, row in enumerate(packed):
        parts, i = {}, 0
        for k, shape in layout.items():
            size = int(np.prod(shape))
            parts[k] = row[i : i + size].reshape(shape)
            i += size
        if parts["n"] > 0:
            store[fid] = parts
    return store


# evaluator glue — shared between the ae and diffusion evaluators
def accumulate_spectral_diagnostics(
    store: Dict[int, Dict[str, np.ndarray]],
    df_pred,
    df_tgt,
    file_idx: np.ndarray,
    val_ds: Any,
    valid: Optional[np.ndarray] = None,
) -> bool:
    """Add per-trajectory spectral sums of a pred/target batch into ``store``.

    ``df_pred``/``df_tgt`` are denormalised batches (host or device); each row's spectral
    fields are computed once with its trajectory's geometry over the whole fixed-shape
    batch, and only the small fields come to host. Rows where ``valid`` is False are
    skipped. Returns ``False`` (without touching ``store``) when the dataset metadata
    carries no ``ds`` so the caller can warn once.
    """
    from neugk_jax.evaluate.integrals import gyaradax_spectral_fields

    file_idx = np.asarray(file_idx)
    valid = np.ones(len(file_idx), bool) if valid is None else np.asarray(valid, bool)
    fids = np.unique(file_idx[valid])
    ds_vals = {int(f): val_ds.get_ds(int(f)) for f in fids}
    if any(v is None for v in ds_vals.values()):
        return False
    if not len(fids):
        return True
    geoms = {
        int(f): {k: np.asarray(v) for k, v in val_ds.metadata[int(f)]["geometry"].items()}
        for f in fids
    }
    rows = [geoms[int(f) if int(f) in geoms else int(fids[0])] for f in file_idx]
    batched = {k: np.stack([g[k] for g in rows]) for k in rows[0]}
    fields = [gyaradax_spectral_fields(src, batched, per_sample=True) for src in (df_pred, df_tgt)]
    for fid in fids:
        idx = np.where((file_idx == fid) & valid)[0]
        geom, ds_val = geoms[int(fid)], ds_vals[int(fid)]
        (pp, pe), (gp, ge) = fields
        sums = spectral_sums(
            _diagnostics_from_fields(pp[idx], pe[idx], geom, ds_val),
            _diagnostics_from_fields(gp[idx], ge[idx], geom, ds_val),
        )
        store[int(fid)] = add_spectral_sums(store.get(int(fid)), sums)
    return True


def merged_spectral_metrics(store: Dict[int, Dict[str, np.ndarray]]) -> Dict[str, float]:
    """Per-trajectory time-averaged spectral metrics, mean over trajectories."""
    per_traj = [metrics_from_spectral_sums(s) for s in store.values() if float(s["n"]) > 0]
    if not per_traj:
        return {}
    return {k: float(np.mean([m[k] for m in per_traj])) for k in per_traj[0]}
