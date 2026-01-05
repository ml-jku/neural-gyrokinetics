from typing import Optional, Dict

import torch
import torch.nn.functional as F

from neugk.integrals import FluxIntegral
from neugk.pinc.neural_fields.nf_utils import phi_fft


def get_integrals(
    pred: torch.Tensor,
    geom: torch.Tensor,
    phi: Optional[torch.Tensor] = None,
    flux_fields: bool = False,
    spectral_df: bool = False,
    spectral_potens: bool = False,
):
    if pred.shape[0] != 2:
        pred = pred[[0, 1]] + pred[[2, 3]]
    geom = {k: g.unsqueeze(0).to(pred.device) for k, g in geom.items()}
    integrator = FluxIntegral(
        real_potens=False,
        flux_fields=flux_fields,
        spectral_df=spectral_df,
        spectral_potens=spectral_potens,
    )
    integrator.to(pred.device)
    phi, (pflux, eflux, vflux) = integrator(geom, df=pred.unsqueeze(0))
    phi = phi.squeeze()
    return phi, (pflux, eflux, vflux)


def diagnostics(
    phi_fft: torch.Tensor,
    eflux_field: torch.Tensor,
    ds: float,
    zf_mode: int = 0,
    aggregate: str = "mid",
):
    diag = {}
    nx, _, ny = phi_fft.shape
    # compute kxspec
    kxspec = torch.sum(torch.abs(phi_fft) ** 2, dim=(2,)) * ds
    if aggregate == "mean":
        diag["kxspec"] = torch.sum(kxspec, dim=0)
    if aggregate == "mid":
        diag["kxspec"] = kxspec[kxspec.shape[0] // 2]
    # compute kyspec
    kyspec = torch.sum(torch.abs(phi_fft) ** 2, dim=(1,)) * ds
    if aggregate == "mean":
        diag["kyspec"] = torch.sum(kyspec, dim=0)
    if aggregate == "mid":
        diag["kyspec"] = kyspec[kyspec.shape[0] // 2]
    # compute zf profile from 5D
    fourier_zf = phi_fft.clone()
    # mask everything except the zf_mode
    fourier_zf[..., :zf_mode] = 0.0
    fourier_zf[..., zf_mode + 1 :] = 0.0
    fourier_zf = torch.fft.fftshift(fourier_zf, dim=(0,))
    diag["phi_zf"] = torch.fft.irfftn(
        fourier_zf, dim=(0, 2), norm="forward", s=[nx, ny]
    )
    # compute flux spectrum
    diag["qspec"] = eflux_field.sum((0, 1, 2, 3))
    return diag


def integral_losses(
    pred_df: torch.Tensor,
    gt_df: torch.Tensor,
    geom: Dict[str, torch.Tensor],
    device: torch.device,
    use_flux_fields: bool = False,
    use_spectral: bool = False,
    timestep: Optional[int] = None,
    return_fields: bool = False,
) -> torch.Tensor:
    losses = {}
    # reconstruct field
    pred_phi, (pred_pflux, pred_eflux, _) = get_integrals(
        pred_df,
        geom,
        flux_fields=True,
        spectral_df=use_spectral,
    )
    gt_phi, (_, gt_eflux, _) = get_integrals(
        gt_df,
        geom,
        flux_fields=True,
        spectral_df=use_spectral,
    )
    # df loss
    losses["df loss"] = F.mse_loss(pred_df, gt_df.to(device))
    # flux loss
    pred_pflux, pred_eflux = pred_pflux.to(device), pred_eflux.to(device)
    pred_df, gt_eflux = pred_df.to(device), gt_eflux.to(device)
    if use_flux_fields:
        flux_loss = (pred_pflux**2).sum() + F.l1_loss(
            pred_eflux, gt_eflux, reduction="sum"
        )
    else:
        flux_loss = pred_pflux.sum() ** 2 + F.l1_loss(pred_eflux.sum(), gt_eflux.sum())
    losses["flux loss"] = flux_loss
    # potential loss
    phi_shift, phi_scale = gt_phi.mean(), gt_phi.std()
    pred_phi = (pred_phi.to(device) - phi_shift) / phi_scale
    gt_phi = (gt_phi.to(device) - phi_shift) / phi_scale
    losses["phi loss"] = F.l1_loss(pred_phi, gt_phi)
    losses["phi mse"] = F.mse_loss(pred_phi, gt_phi)

    if return_fields:
        return (
            losses,
            (pred_phi, gt_phi),
            (pred_eflux[0], gt_eflux[0]),
        )
    else:
        return losses


def spectra_losses(
    pred_df: torch.Tensor,
    pred_phi: torch.Tensor,
    pred_eflux: torch.Tensor,
    gt_df: torch.Tensor,
    gt_phi: torch.Tensor,
    gt_eflux: torch.Tensor,
    ds: float,
    monotonicity_tol: float = 0.1,
    aggregate: str = "mean",
) -> torch.Tensor:
    pred_eflux, gt_eflux = pred_eflux.squeeze(), gt_eflux.squeeze()
    losses = {}
    pred_phi_fft = phi_fft(pred_phi)
    gt_phi_fft = phi_fft(gt_phi)
    pred_diag = diagnostics(pred_phi_fft, pred_eflux, ds=ds, aggregate=aggregate)
    gt_diag = diagnostics(gt_phi_fft, gt_eflux, ds=ds, aggregate=aggregate)
    # spectral trace loss
    losses.update({f"{k} loss": F.l1_loss(pred_diag[k], gt_diag[k]) for k in pred_diag})
    # enforce negative derivatives (log space, after peak)
    # kernel = torch.ones(1, 1, 5, device=pred_df.device) / 5
    for k in ["qspec", "kyspec"]:
        spec = torch.nan_to_num(torch.log1p(pred_diag[k]), 0.0)
        # spec_gt = torch.nan_to_num(torch.log1p(gt_diag[k]), 0.0)
        peak_idx = torch.argmax(spec).item()
        tail = spec[peak_idx:]
        # isotonic loss
        # spec = F.conv1d(spec[None, None, :], kernel, padding=1)[0, 0]
        tail_sorted, _ = torch.sort(tail, descending=True)
        losses[f"{k} monotonicity loss"] = F.l1_loss(tail, tail_sorted)
        # # finite differences
        # diff = (tail[1:] - tail[:-1]) - monotonicity_tol
        # losses[f"{k} monotonicity loss"] = torch.clamp(diff, min=0.0).mean()

        # # finite differences (automatic tol)
        # tail_gt = spec_gt[peak_idx:]
        # diff_pred = tail[1:] - tail[:-1]
        # diff_gt = tail_gt[1:] - tail_gt[:-1]
        # tol = torch.clamp(diff_gt.max(), min=0.0).item()
        # losses[f"{k} monotonicity loss"] = torch.clamp(diff_pred - tol, min=0.0).sum()
    # mass conservation loss (works in real space? fft is linear so yeah)
    losses["mass loss"] = F.l1_loss(pred_df.sum(), gt_df.sum())
    return losses, (gt_diag, pred_diag)
