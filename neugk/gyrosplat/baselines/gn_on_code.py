"""The physics stage applied to a linear code.

A linear code stores complex values and decodes them linearly, here the kept Fourier
coefficients.  The flux rows are then the same quadratic functions of the stored values as
they are of the atoms' amplitudes, and the same damped Gauss-Newton applies, with the
damping taken relative to the largest diagonal entry of J J^T.  The Jacobian is the
per-mode row gradient pulled back through the decoder by autograd.

`descend_on_code` is the Adam warm-up on the backend's physics terms that the Fourier
control takes first: it starts too far from the flux target for the Gauss-Newton alone.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

import torch

from neugk.gyrosplat.config import GaussNewtonConfig
from neugk.gyrosplat.model.gn import row_gradients


@dataclass
class LinearCode:
    """values: per species, the stored complex values that may move (1-D).
    decode: values -> the spectral object (n_species, ...), linear and differentiable."""

    values: List[torch.Tensor]
    decode: Callable[[List[torch.Tensor]], torch.Tensor]


def _unknowns(code: LinearCode):
    """The values as real pairs, and the decoder in those units."""
    theta = [torch.view_as_real(v).clone() for v in code.values]

    def decode(th):
        return code.decode([torch.view_as_complex(t) for t in th])

    return theta, decode


def gauss_newton_on_code(
    backend,
    code: LinearCode,
    cfg: GaussNewtonConfig,
    log=print,
    tol: float = 1e-7,
    relative_damping: float = 1e-8,
    return_code: bool = False,
):
    """Damped Gauss-Newton on the code's values against the flux rows; stops early once the row
    residual falls below `tol`.  The damping is `relative_damping` times the largest diagonal
    entry of J J^T. Returns (H, history), or (H, history, fitted_code) when
    return_code=True, so the exact final coefficients can be serialized."""
    theta, decode = _unknowns(code)
    sizes = [t.numel() for t in theta]
    history = []
    for it in range(cfg.steps):
        with torch.no_grad():
            H = decode(theta)
            r = backend.flux_rows_target() - backend.flux_rows(H)
        history.append(float(r.norm()))
        log(f"  gn step {it + 1}/{cfg.steps}  |r| {history[-1]:.4e}")
        if history[-1] < tol:
            break
        G = row_gradients(backend, H)
        del H
        th = [t.clone().requires_grad_(True) for t in theta]
        Hd = decode(th)
        J = torch.zeros(r.numel(), sum(sizes), dtype=torch.float64, device=r.device)
        for b, i in enumerate(backend.band):
            plane = Hd[..., i]
            for k in range(G[b].shape[0]):
                pulled = (torch.conj(G[b][k]) * plane).real.sum()
                grads = torch.autograd.grad(pulled, th, retain_graph=True)
                J[backend.row_index(b, k)] = torch.cat(
                    [g.reshape(-1) for g in grads]
                ).double()
        del Hd, G
        R = r.numel()
        JJt = J @ J.T
        mu = relative_damping * JJt.diagonal().max()
        step = J.T @ torch.linalg.solve(
            JJt + mu * torch.eye(R, device=J.device, dtype=torch.float64), r
        )
        offset = 0
        for t, n in zip(theta, sizes):
            t += step[offset : offset + n].view_as(t).to(t.dtype)
            offset += n
        del J, step
    with torch.no_grad():
        H = decode(theta)
        info = {"residual": history}
        if return_code:
            fitted = LinearCode(
                [torch.view_as_complex(t).detach().clone() for t in theta], code.decode
            )
            return H, info, fitted
        return H, info


def descend_on_code(
    backend, code: LinearCode, steps: int, lr: float = 3e-3, log=print
) -> LinearCode:
    """Adam with a cosine schedule on the code's values against the backend's field, potential
    and flux terms, each relative to its starting value, with weights 1, 1/2 and 1/2.  The
    values are taken in orthonormal-transform units (`backend.spectral_unit`), in which the
    learning rate is absolute."""
    unit = float(getattr(backend, "spectral_unit", 1.0))
    params = [
        torch.view_as_real(v * unit).clone().requires_grad_(True) for v in code.values
    ]
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    start = None
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        H = code.decode([torch.view_as_complex(p) / unit for p in params])
        terms = backend.physics_terms(backend.to_real(H), with_flux=True)
        if start is None:
            start = {k: max(float(v.detach()), 1e-12) for k, v in terms.items()}
        loss = (
            terms["field"] / start["field"]
            + 0.5 * terms["potential"] / start["potential"]
            + 0.5 * terms["flux"] / start["flux"]
        )
        loss.backward()
        opt.step()
        sched.step()
        if step % max(1, steps // 4) == 0:
            log(f"  warm-up {step}/{steps}  flux {terms['flux'].item():.4f}")
    values = [(torch.view_as_complex(p.detach()) / unit).clone() for p in params]
    return LinearCode(values, code.decode)
