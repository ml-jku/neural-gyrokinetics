"""Heat-flux views outside the fitting objective.

The physics stage fits the heat-flux spectrum Q(ky) over the nonzero binormal modes.  The
views below integrate the same signed flux integrand, laid out (v_par, mu, s, kx, ky), over
other axes: the radially resolved flux Q(kx), the flux along the field line Q(s), the
velocity-space flux Q(v_par, mu), and the integrand itself (pointwise), always over the
nonzero modes.  A reconstruction can match Q(ky) exactly and still miss these views; the
appendix reports their relative L1 errors.

    from neugk.gyrosplat.eval.diagnostics import view_errors
    errors = view_errors(backend, decoded)      # {"Q_kx": ..., "Q_s": ..., ...}

`flux_views` works on any integrand tensor; `view_errors` takes a backend with `readouts`
(the GKW backend: the host's `get_integrals`) and a decoded spectral object.
"""
from __future__ import annotations

from typing import Dict

import torch

VIEWS = ("Q_kx", "Q_s", "Q_vpar_mu", "Q_pointwise")


def flux_views(integrand: torch.Tensor) -> Dict[str, torch.Tensor]:
    """The views of a heat-flux integrand (v_par, mu, s, kx, ky) over the nonzero ky."""
    if integrand.dim() != 5:
        raise ValueError(
            f"expected (v_par, mu, s, kx, ky), got {tuple(integrand.shape)}"
        )
    q = integrand[..., 1:].double()
    return {
        "Q_ky": q.sum((0, 1, 2, 3)),
        "Q_kx": q.sum((0, 1, 2, 4)),
        "Q_s": q.sum((0, 1, 3, 4)),
        "Q_vpar_mu": q.sum((2, 3, 4)),
        "Q_pointwise": q,
    }


def relative_l1(pred: torch.Tensor, ref: torch.Tensor) -> float:
    """sum |pred - ref| / sum |ref|."""
    err = (pred.double() - ref.double()).abs().sum()
    return float(err / ref.double().abs().sum().clamp_min(1e-300))


@torch.no_grad()
def view_errors(backend, decoded: torch.Tensor) -> Dict[str, float]:
    """Relative L1 errors of the views of a decoded spectral object against the backend's
    reference snapshot."""
    field = backend.to_real(decoded)[0]
    _, _, integrand = backend.readouts(field)
    pred, ref = flux_views(integrand), flux_views(backend.integrand_gt)
    return {k: relative_l1(pred[k], ref[k]) for k in VIEWS}
