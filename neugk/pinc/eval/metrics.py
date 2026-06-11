"""Metrics for the PINC compression evaluation.

Reuses the shared physics diagnostics (`neugk.physics`) so the eval and the
training losses measure the same quantities. Ported from the running logic of
`notebooks/01_pinc_evaluation_hf.ipynb`, structured for batch/scale use.
"""

from typing import Dict, List, Optional, Sequence

import torch

from neugk.physics import diagnostics
from neugk.physics.integrals import FluxIntegral
from neugk.pinc.neural_fields.nf_utils import endpoint_error, optical_flow_5d


def _batch_geom(geom: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.unsqueeze(0) for k, v in geom.items()}


def _pearson(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a, b = a - a.mean(), b - b.mean()
    return (a * b).sum() / (a.norm() * b.norm() + 1e-30)


def _spearman(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # ordinal ranks (== scipy average ranks when there are no ties; spectra are
    # continuous so ties don't occur), then Pearson on the ranks.
    ra = a.argsort().argsort().to(a.dtype)
    rb = b.argsort().argsort().to(b.dtype)
    return _pearson(ra, rb)


def _wasserstein_1d(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # 1D W1 for equal-length, uniform-weight samples == mean|sorted(u)-sorted(v)|;
    # matches scipy.stats.wasserstein_distance(u, v) on the (same-length) spectra.
    return (u.sort().values - v.sort().values).abs().mean()


def ml_eval(
    pred_df: torch.Tensor,
    gt_df: torch.Tensor,
    pred_phi: torch.Tensor,
    gt_phi: torch.Tensor,
    pred_eflux: torch.Tensor,
    gt_eflux: torch.Tensor,
    compressed_size: Optional[int] = None,
) -> Dict[str, float]:
    """Per-snapshot reconstruction + integral metrics (all on pred_df's device)."""
    dev = pred_df.device
    gt_df = gt_df.to(dev)
    pred_phi, gt_phi = pred_phi.to(dev), gt_phi.to(dev)
    pred_eflux, gt_eflux = pred_eflux.to(dev), gt_eflux.to(dev)
    mse = ((pred_df - gt_df) ** 2).mean()
    phi_mse = ((pred_phi - gt_phi) ** 2).mean()
    m = {
        "mse": float(mse),
        "l1": float((pred_df - gt_df).abs().mean()),
        "psnr": float(10 * torch.log10(gt_df.max() ** 2 / mse)),
        "phi_mse": float(phi_mse),
        "phi_l1": float((pred_phi - gt_phi).abs().mean()),
        "phi_psnr": float(10 * torch.log10(gt_phi.max() ** 2 / phi_mse)),
        # heat flux Q is the phase-space integral; match the training loss
        # (sum the field to the scalar Q, then L1), not a per-cell mean.
        "eflux_l1": float((pred_eflux.sum() - gt_eflux.sum()).abs()),
    }
    if compressed_size is not None:
        m["bpp"] = (compressed_size * 8) / pred_df.numel()
    return m


def integrate(df: torch.Tensor, geom: Dict[str, torch.Tensor]):
    """Real-space potential and heat-flux field for the ml metrics."""
    integ = FluxIntegral(real_potens=True, flux_fields=True)
    phi, (_, eflux, _) = integ(_batch_geom(geom), df.unsqueeze(0))
    return phi.squeeze(0), eflux.squeeze(0)


def spectral_diagnostics(
    df: torch.Tensor, geom: Dict[str, torch.Tensor], ds: float
) -> Dict[str, torch.Tensor]:
    """Turbulence spectra (kxspec/kyspec/qspec) for one snapshot, as on-device tensors."""
    integ = FluxIntegral(real_potens=True, flux_fields=True, spectral_potens=True)
    phi_spec, (_, eflux, _) = integ(_batch_geom(geom), df.unsqueeze(0))
    d = diagnostics(phi_spec.squeeze(), eflux.squeeze(), ds=ds)
    return {k: v.detach() for k, v in d.items()}


def time_averaged_spectral_metrics(
    pred_diags: List[Dict[str, torch.Tensor]],
    gt_diags: List[Dict[str, torch.Tensor]],
) -> Dict[str, float]:
    """Pearson/Spearman/Wasserstein/L1 on the time-averaged ky and Q spectra."""
    out: Dict[str, float] = {}
    for key in ("kyspec", "qspec"):
        p = torch.stack([d[key] for d in pred_diags], 0).mean(0)
        g = torch.stack([d[key] for d in gt_diags], 0).mean(0)
        out[f"{key}_pc"] = float(_pearson(p, g))
        out[f"{key}_sc"] = float(_spearman(p, g))
        out[f"{key}_l1"] = float((p - g).abs().sum())
        pn, gn = p / (p.sum() + 1e-12), g / (g.sum() + 1e-12)
        out[f"{key}_wd"] = float(_wasserstein_1d(pn, gn))
    return out


def temporal_epe(
    gt_dfs: Sequence[torch.Tensor], pred_dfs: Sequence[torch.Tensor]
) -> float:
    """End-point error of the optical flow over a snapshot sequence (>=2 frames)."""
    if len(gt_dfs) < 2:
        return float("nan")
    # stack on-device; optical_flow_5d is torch/GPU (was scipy-CPU, minutes/traj).
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = torch.stack([d.to(dev, torch.float32) for d in gt_dfs])
    p = torch.stack([d.to(dev, torch.float32) for d in pred_dfs])
    return float(endpoint_error(g, p, optical_flow_5d))


# direction of improvement, for table formatting downstream
DIRECTION = {
    "l1": "min",
    "mse": "min",
    "psnr": "max",
    "bpp": "min",
    "cr": "max",
    "phi_l1": "min",
    "phi_psnr": "max",
    "eflux_l1": "min",
    "endpoint": "min",
    "kyspec_pc": "max",
    "qspec_pc": "max",
    "kyspec_sc": "max",
    "qspec_sc": "max",
    "kyspec_l1": "min",
    "qspec_l1": "min",
    "kyspec_wd": "min",
    "qspec_wd": "min",
    "density_l1": "min",
    "momentum_l1": "min",
    "energy_l1": "min",
    "free_energy_err": "min",
}
