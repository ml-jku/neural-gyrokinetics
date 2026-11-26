"""Baseline codecs, every one scored through the same backend decode as the atoms.

A baseline takes the represented object H (spectral) and a budget of stored reals and
returns a reconstruction of H; the backend then decodes and scores it exactly as it does
the atom codec's output.  The Fourier code is linear in its stored values and has a
"+ objective" variant: the same Gauss-Newton on the flux rows, moving the kept
coefficients (`gn_on_code.py`).

    fourier   top-K coefficients of the full Fourier transform      (+ objective)
    wavelet   top-K coefficients of a 5-axis wavelet transform, one per real channel
              (+ objective for the Haar basis, through the differentiable Haar code)
    pca       centred PCA per real channel at the rank closest to the budget
    zfp, sz3  the error-bounded compressors, searched to the budget (zfpy, pysz)
    splats    plain Gaussian splats with a field objective only
    neural_field  one MLP over the grid coordinates, field objective only

Wavelet, PCA, ZFP and SZ3 see the two stored real channels as separate arrays, with the
rate the closest reachable one, as in the evaluation of PINC.
"""
from __future__ import annotations

import copy
import time
from typing import Dict, Optional

import torch

from neugk.gyrosplat.config import RecipeConfig
from neugk.gyrosplat.model.pack import compression_ratios

METHODS = (
    "fourier",
    "fourier_gn",
    "wavelet",
    "wavelet_gn",
    "pca",
    "zfp",
    "sz3",
    "splats",
    "neural_field",
)


def budget_from_recipe(cfg: RecipeConfig) -> int:
    """Independent atom parameters plus normalization, in float32 units per species.

    Pass an explicit budget when replaying historical baseline results, which used
    the old shared-envelope estimate. Parameters at 16 bits buy half as many reals.
    """
    from neugk.gyrosplat.recipe import budget, reals_per_carrier

    n_base, n_groups = budget(cfg)
    n_carr = n_groups * cfg.carriers.tied_k if cfg.carriers.kind == "tied_gabor" else 0
    reals = n_base * cfg.reals_per_atom + n_carr * reals_per_carrier(cfg)
    return int(round(reals * cfg.storage_bits / 32)) + 4


def run_baseline(
    method: str,
    backend,
    cfg: RecipeConfig,
    budget: Optional[int] = None,
    device: str = "cuda",
    log=print,
) -> Dict[str, object]:
    from neugk.gyrosplat.recipe import _snr_per_species

    t0 = time.time()
    budget = budget or budget_from_recipe(cfg)
    name, with_objective = (
        (method[:-3], True) if method.endswith("_gn") else (method, False)
    )
    if name == "splats":
        from neugk.gyrosplat.baselines.splats import field_only_splats

        return field_only_splats(backend, cfg, budget, device, log)

    code = None  # set by the codes that are linear in their stored values
    if name == "fourier":
        from neugk.gyrosplat.baselines.fourier import fourier_topk

        axes = tuple(backend.real_axes) if cfg.baselines.fourier_all_axes else ()
        H_hat, achieved, code = fourier_topk(backend.gt, budget, axes)
    elif name == "neural_field":
        from neugk.gyrosplat.baselines.neural_field import neural_field

        H_hat, achieved, _ = neural_field(
            backend, backend.to_real(backend.gt), budget, cfg, log
        )
    elif name in ("wavelet", "pca", "zfp", "sz3"):
        real = backend.to_real(backend.gt)
        if name == "pca":
            from neugk.gyrosplat.baselines.pca import pca_channels

            H_hat, achieved, _ = pca_channels(backend, real, budget)
        elif name == "wavelet" and with_objective:
            from neugk.gyrosplat.baselines.haar_code import haar_linear_code

            if cfg.baselines.wavelet != "haar" or cfg.baselines.wavelet_level != 1:
                raise ValueError(
                    "the objective variant exists for the one-level Haar basis"
                )
            H_hat, achieved, code, _ = haar_linear_code(backend, real, budget)
        elif name == "wavelet":
            from neugk.gyrosplat.baselines.wavelet import wavelet_topk

            rec, achieved, _ = wavelet_topk(
                real,
                budget,
                wavelet=cfg.baselines.wavelet,
                level=cfg.baselines.wavelet_level,
            )
            H_hat = backend.to_spectral(rec)
        else:
            from neugk.gyrosplat.baselines.error_bounded import error_bounded

            rec, achieved, _ = error_bounded(real, budget, codec=name)
            H_hat = backend.to_spectral(rec)
        del real
    else:
        raise ValueError(f"unknown baseline {method}")
    if with_objective and code is None:
        raise ValueError(f"{name} is not linear in stored values: no objective variant")

    def scored(tag, H):
        with torch.no_grad():
            row = backend.score(backend.decode(H))
            row["H_snr_dB"] = _snr_per_species(backend.gt, H - backend.gt)
        row["tag"] = tag
        stored = {"atoms": float(sum(achieved))}
        stored["total"] = stored["atoms"]
        row.update(
            {"stored_reals": stored, **compression_ratios(backend.raw_reals, stored)}
        )
        return row

    rows = [scored(name, H_hat)]
    log(f"  [{name}] " + _short(rows[-1]))
    del H_hat
    if with_objective:
        from neugk.gyrosplat.baselines.gn_on_code import (
            descend_on_code,
            gauss_newton_on_code,
        )

        if cfg.baselines.objective_warmup > 0:
            code = descend_on_code(
                backend, code, cfg.baselines.objective_warmup, log=log
            )
        H_obj, _ = gauss_newton_on_code(
            backend,
            code,
            copy.copy(cfg.gn),
            log,
            relative_damping=cfg.baselines.objective_relative_damping,
        )
        rows.append(scored(f"{name} + objective", H_obj))
        log(f"  [{name} + objective] " + _short(rows[-1]))
    return {
        "method": method,
        "budget_reals": budget,
        "achieved_reals": [float(a) for a in achieved],
        "rows": rows,
        "seconds": time.time() - t0,
        "config": {"name": method},
        "wavelet": cfg.baselines.wavelet if name == "wavelet" else None,
    }


def _short(row: dict) -> str:
    fl = row["flux_relL1_band"]
    return (
        f"psnr_f {row.get('psnr_paper_all', float('nan')):.1f} phi {row['phi_psnr']:.2f} flux "
        + "/".join(f"{x:.4f}" for x in fl)
        + f" CR {row['CR']:.0f}x"
    )
