"""Shared gyrokinetic physics losses and turbulence diagnostics.

Single source of truth for the physics-informed loss terms used by both the
autoencoder path (``neugk/pinc/losses.py``) and the neural-field path
(``neugk/pinc/neural_fields/gk_losses.py``).

Reductions follow the neural-field reference (the implementation that produced
the reported diagnostics): ``kxspec`` sums the y axis, ``kyspec`` sums the x
axis, ``qspec`` sums the heat-flux field, ``phi_zf`` uses a real ``irfftn``, and
the monotonicity penalty is sort-based. The functions are batch-agnostic: an
optional leading batch dimension is preserved, the last three axes are taken as
the ``(nx, *, ny)`` spectral/spatial axes.
"""

from typing import Dict, Optional, Sequence, MutableMapping

import torch
import torch.nn.functional as F

from neugk.physics.integrals import get_integrals

EPS = 1e-8


# --------------------------------------------------------------------------- #
# Loss-type dispatch (ported from PINCLossWrapper, made standalone)
# --------------------------------------------------------------------------- #
def compute_data_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_type: str = "mse",
    eps: float = EPS,
    reduction: str = "mean",
    complex_metrics=None,
) -> torch.Tensor:
    """Reconstruction-style loss with a configurable norm."""
    if loss_type == "mse":
        return F.mse_loss(pred, target, reduction=reduction)
    if loss_type == "l1":
        return F.l1_loss(pred, target, reduction=reduction)
    if loss_type == "huber":
        return F.huber_loss(pred, target, reduction=reduction)
    if loss_type == "smooth_l1":
        return F.smooth_l1_loss(pred, target, reduction=reduction)
    if loss_type == "relative_mse":
        if reduction == "mean":
            return torch.sum((pred - target) ** 2) / (torch.sum(target**2) + eps)
        return (pred - target) ** 2 / (target**2 + eps)
    if loss_type == "relative_l1":
        return (torch.abs(pred - target) / (torch.abs(target) + eps)).mean()
    if loss_type == "log_error":
        return F.mse_loss(
            torch.log(torch.abs(pred) + eps),
            torch.log(torch.abs(target) + eps),
            reduction=reduction,
        )
    if loss_type == "log_l1_error":
        return F.l1_loss(
            torch.log(torch.abs(pred) + eps),
            torch.log(torch.abs(target) + eps),
            reduction=reduction,
        )
    if loss_type == "log_cosh":
        return torch.log(torch.cosh(pred - target)).mean()
    if "complex" in loss_type and complex_metrics is not None:
        p_c, t_c = complex_metrics.to_complex(pred), complex_metrics.to_complex(target)
        return (
            complex_metrics.complex_mse(p_c, t_c).mean()
            if "mse" in loss_type
            else complex_metrics.complex_l1(p_c, t_c).mean()
        )
    raise ValueError(f"unknown data loss type: {loss_type}")


def compute_integral_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_type: str = "mse",
    eps: float = EPS,
    loss_name: str = "flux_int",
    dataset_stats: Optional[Dict] = None,
    ema_state: Optional[MutableMapping] = None,
) -> torch.Tensor:
    """Integral-quantity loss; supports dataset-normalised and adaptive variants."""
    if loss_type == "mse":
        return F.mse_loss(pred, target)
    if loss_type in ["relative_mse", "relative_l1", "log_error"]:
        return compute_data_loss(pred, target, loss_type=loss_type, eps=eps)

    if loss_type == "adaptive_relative":
        alpha = 0.01
        key = loss_name.split("_")[0]
        ema_state = ema_state if ema_state is not None else {}
        curr_mean = torch.abs(target).mean().item()
        ema_state[key] = (
            curr_mean
            if key not in ema_state
            else alpha * curr_mean + (1 - alpha) * ema_state[key]
        )
        return ((pred - target) / max(ema_state[key], eps)).pow(2).mean()

    if loss_type in ["int_norm_mse", "int_norm_l1"]:
        stats = dataset_stats or {}
        skey = "flux_std" if "flux" in loss_name else "phi_std"
        if skey in stats:
            norm_err = (pred - target) / max(float(stats[skey]), eps)
            return (
                norm_err.pow(2) if "mse" in loss_type else torch.abs(norm_err)
            ).mean()
        return compute_data_loss(pred, target, loss_type=loss_type, eps=eps)

    raise ValueError(f"unknown integral loss type: {loss_type}")


def compute_spectral_loss(
    pred: torch.Tensor, target: torch.Tensor, loss_type: str = "l1", eps: float = EPS
) -> torch.Tensor:
    """Spectral-trace loss; supports normalised and log-space variants."""
    if loss_type in ["l1", "mse", "relative_l1", "relative_mse"]:
        return compute_data_loss(pred, target, loss_type=loss_type, eps=eps)
    if "normalized" in loss_type:
        scale = torch.mean(torch.abs(target)) + eps
        return (
            F.l1_loss(pred / scale, target / scale)
            if "l1" in loss_type
            else F.mse_loss(pred / scale, target / scale)
        )
    if "log" in loss_type:
        if "relative" in loss_type:
            pred, target = pred / (pred.sum() + eps), target / (target.sum() + eps)
        pl, tl = torch.log(torch.abs(pred) + eps), torch.log(torch.abs(target) + eps)
        return F.l1_loss(pl, tl) if "l1" in loss_type else F.mse_loss(pl, tl)
    raise ValueError(f"unknown spectral loss type: {loss_type}")


# --------------------------------------------------------------------------- #
# Spectral diagnostics (neural-field reference, generalised over a batch dim)
# --------------------------------------------------------------------------- #
def phi_fft(phi: torch.Tensor, norm: str = "forward") -> torch.Tensor:
    """FFT of the electrostatic potential over the (x, y) axes.

    Accepts a real 2-channel representation (``re``/``im`` on a leading channel
    axis) or an already-complex tensor, batched or unbatched. The transform is
    taken over the first and last spatial axes ``(x, ..., y)`` with an
    ``fftshift`` on x, matching ``nf_utils.phi_fft``.
    """
    phi = phi.float() if not torch.is_complex(phi) else phi
    if not torch.is_complex(phi):
        # 2-channel complex: channel axis is 0 (unbatched) or 1 (batched)
        if phi.shape[0] == 2 and phi.ndim in (3, 4):
            phi = torch.view_as_complex(phi.movedim(0, -1).contiguous())
        elif phi.ndim >= 4 and phi.shape[1] == 2:
            phi = torch.view_as_complex(phi.movedim(1, -1).contiguous())
        else:
            phi = phi.to(torch.complex64)
    x_dim = phi.ndim - 3
    return torch.fft.fftshift(
        torch.fft.fftn(phi, dim=(x_dim, phi.ndim - 1), norm=norm), dim=(x_dim,)
    )


def diagnostics(
    phi_fft_: torch.Tensor,
    eflux_field: torch.Tensor,
    ds: float,
    zf_mode: int = 0,
    aggregate: str = "mid",
) -> Dict[str, torch.Tensor]:
    """Turbulence diagnostics from the potential FFT and the heat-flux field.

    The last three axes of ``phi_fft_`` are ``(nx, *, ny)``. ``kxspec`` sums the
    y axis, ``kyspec`` sums the x (``*``) axis; an optional leading batch axis is
    preserved. ``aggregate`` selects how the remaining nx axis is collapsed:
    ``"mean"`` sums it, ``"mid"`` takes the central slice, ``"none"`` keeps it.
    """
    diag: Dict[str, torch.Tensor] = {}
    nx, ny = phi_fft_.shape[-3], phi_fft_.shape[-1]
    # torch.abs on complex tensors fails on Blackwell GPUs (nvrtc arch error).
    # Use manual sqrt(re²+im²) instead.
    if torch.is_complex(phi_fft_):
        power = (phi_fft_.real ** 2 + phi_fft_.imag ** 2)
    else:
        power = torch.abs(phi_fft_) ** 2

    kxspec = power.sum(dim=-1) * ds  # reduce y -> (..., nx, mid)
    kyspec = power.sum(dim=-2) * ds  # reduce mid -> (..., nx, ny)

    def _agg(spec):  # collapse the nx axis (now at -2)
        if aggregate == "mean":
            return spec.sum(dim=-2)
        if aggregate == "mid":
            return spec.index_select(
                -2, torch.tensor([nx // 2], device=spec.device)
            ).squeeze(-2)
        return spec

    diag["kxspec"] = _agg(kxspec)
    diag["kyspec"] = _agg(kyspec)

    # zonal-flow profile: keep only zf_mode along y, real inverse transform
    fourier_zf = phi_fft_.clone()
    fourier_zf[..., :zf_mode] = 0.0
    fourier_zf[..., zf_mode + 1 :] = 0.0
    x_dim = phi_fft_.ndim - 3
    fourier_zf = torch.fft.fftshift(fourier_zf, dim=(x_dim,))
    diag["phi_zf"] = torch.fft.irfftn(
        fourier_zf, dim=(x_dim, phi_fft_.ndim - 1), norm="forward", s=[nx, ny]
    )

    # heat-flux spectrum: sum the flux field over the first four axes
    # (velocity/species, s, x), leaving the trailing wavenumber axis (NF reference).
    diag["qspec"] = (
        eflux_field.sum(dim=(0, 1, 2, 3))
        if eflux_field.ndim >= 4
        else eflux_field.sum()
    )
    return diag


def monotonicity_loss(
    pred_diag: Dict[str, torch.Tensor],
    keys: Sequence[str] = ("qspec", "kyspec"),
) -> Dict[str, torch.Tensor]:
    """Sort-based isotonic penalty: the post-peak tail (log space) should decay.

    Works on a single spectrum (1D) or a batch of spectra (leading batch dim);
    the per-row penalties are averaged.
    """
    losses: Dict[str, torch.Tensor] = {}
    for k in keys:
        if k not in pred_diag:
            continue
        x = torch.nan_to_num(torch.log1p(pred_diag[k]), 0.0)
        rows = x.reshape(1, -1) if x.ndim <= 1 else x.reshape(x.shape[0], -1)
        terms = []
        for r in rows:
            if r.numel() < 2:
                continue
            tail = r[torch.argmax(r).item() :]
            tail_sorted, _ = torch.sort(tail, descending=True)
            terms.append(F.l1_loss(tail, tail_sorted))
        losses[f"{k} monotonicity loss"] = (
            torch.stack(terms).mean() if terms else torch.zeros((), device=x.device)
        )
    return losses


def mass_loss(pred_df: torch.Tensor, gt_df: torch.Tensor) -> torch.Tensor:
    """Total-mass conservation: L1 on the summed distribution function."""
    return F.l1_loss(pred_df.sum(), gt_df.sum())


# --------------------------------------------------------------------------- #
# Integral losses (neural-field reference path)
# --------------------------------------------------------------------------- #
def integral_losses(
    pred_df: torch.Tensor,
    gt_df: torch.Tensor,
    geom: Dict[str, torch.Tensor],
    device: torch.device,
    use_flux_fields: bool = False,
    use_spectral: bool = False,
    timestep: Optional[int] = None,
    return_fields: bool = False,
):
    """Particle/heat-flux and (normalised) potential losses for an unbatched snapshot."""
    losses = {}
    pred_phi, (pred_pflux, pred_eflux, _) = get_integrals(
        pred_df, geom, flux_fields=True, spectral_df=use_spectral
    )
    gt_phi, (_, gt_eflux, _) = get_integrals(
        gt_df, geom, flux_fields=True, spectral_df=use_spectral
    )

    losses["df loss"] = F.mse_loss(pred_df, gt_df.to(device))

    pred_pflux, pred_eflux = pred_pflux.to(device), pred_eflux.to(device)
    gt_eflux = gt_eflux.to(device)
    if use_flux_fields:
        flux_loss = (pred_pflux**2).sum() + F.l1_loss(
            pred_eflux, gt_eflux, reduction="sum"
        )
    else:
        flux_loss = pred_pflux.sum() ** 2 + F.l1_loss(pred_eflux.sum(), gt_eflux.sum())
    losses["flux loss"] = flux_loss

    phi_shift, phi_scale = gt_phi.mean(), gt_phi.std()
    pred_phi = (pred_phi.to(device) - phi_shift) / phi_scale
    gt_phi = (gt_phi.to(device) - phi_shift) / phi_scale
    losses["phi loss"] = F.l1_loss(pred_phi, gt_phi)
    losses["phi mse"] = F.mse_loss(pred_phi, gt_phi)

    if return_fields:
        return losses, (pred_phi, gt_phi), (pred_eflux[0], gt_eflux[0])
    return losses


def spectra_losses(
    pred_df: torch.Tensor,
    pred_phi: torch.Tensor,
    pred_eflux: torch.Tensor,
    gt_df: torch.Tensor,
    gt_phi: torch.Tensor,
    gt_eflux: torch.Tensor,
    ds: float,
    aggregate: str = "mean",
):
    """Spectral-trace, monotonicity and mass losses for an unbatched snapshot."""
    pred_eflux, gt_eflux = pred_eflux.squeeze(), gt_eflux.squeeze()
    pred_diag = diagnostics(phi_fft(pred_phi), pred_eflux, ds=ds, aggregate=aggregate)
    gt_diag = diagnostics(phi_fft(gt_phi), gt_eflux, ds=ds, aggregate=aggregate)

    losses = {f"{k} loss": F.l1_loss(pred_diag[k], gt_diag[k]) for k in pred_diag}
    losses.update(monotonicity_loss(pred_diag))
    losses["mass loss"] = mass_loss(pred_df, gt_df)
    return losses, (gt_diag, pred_diag)
