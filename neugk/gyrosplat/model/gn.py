"""The physics stage: Gauss-Newton on the carrier amplitudes against the flux rows.

The flux rows are quadratic in the represented object and the object is linear in the
amplitudes, so a few damped Gauss-Newton steps on the rows converge fast.  The Jacobian is
one backward pass per row through the full render (`jacobian_autograd`), exact through the
host readout; `row_gradients` gives the per-mode row gradients the "+ objective" baselines
pull back through their own decoders.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch

from neugk.gyrosplat.model.atoms import AtomBank
from neugk.gyrosplat.config import GaussNewtonConfig
from neugk.gyrosplat.model.normalize import NormStats, denormalize
from neugk.gyrosplat.model.synth import render_linear, to_channels


@dataclass
class SpectralModel:
    """Everything needed to go from amplitudes to the spectral object: the banks, their grids,
    the normalisation statistics and the backend's spectral transform."""

    banks: List[AtomBank]
    vel_grids: List[torch.Tensor]
    phys_grids: List[torch.Tensor]
    stats: List[NormStats]
    backend: object
    chunk: int = 16384

    @property
    def shapes(self):
        return [b.shape for b in self.banks]

    def fields(self, amps: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        """Per-species (2, ...) de-normalised real channels, differentiable in `amps`."""
        out = []
        for a, bank, vel, phys, st in zip(
            amps, self.banks, self.vel_grids, self.phys_grids, self.stats
        ):
            out.append(
                denormalize(
                    to_channels(
                        render_linear(a, bank, vel, phys, self.chunk), bank.shape
                    ),
                    st,
                )
            )
        return out

    def spectral(self, amps: Sequence[torch.Tensor]) -> torch.Tensor:
        return self.backend.to_spectral(self.fields(amps))

    def current_amps(self) -> List[torch.Tensor]:
        return [b.amps.data for b in self.banks]

    def set_amps(self, flat: torch.Tensor):
        i = 0
        for b in self.banks:
            n = 2 * b.N
            b.amps.data.copy_(flat[i : i + n].view(-1, 2).to(b.amps.dtype))
            i += n

    def flat_amps(self) -> torch.Tensor:
        return torch.cat([b.amps.data.reshape(-1) for b in self.banks]).double()

    def column_mask(self, carriers_only: bool) -> torch.Tensor:
        if not carriers_only:
            return torch.ones(
                sum(2 * b.N for b in self.banks),
                dtype=torch.float64,
                device=self.banks[0].mu.device,
            )
        return torch.cat(
            [b.carrier_mask.repeat_interleave(2) for b in self.banks]
        ).double()


# ----------------------------------------------------------------------------- Jacobians
def jacobian_autograd(model: SpectralModel) -> tuple:
    """(J (R, 2 N_total) float64, residual r) with one backward pass per row through the render."""
    backend = model.backend
    with torch.enable_grad():
        amps = [b.amps.data.clone().requires_grad_(True) for b in model.banks]
        rows = backend.flux_rows(model.spectral(amps))
        r = (backend.flux_rows_target() - rows).detach()
        R = rows.numel()
        cols = []
        for k in range(R):
            grads = torch.autograd.grad(rows[k], amps, retain_graph=(k < R - 1))
            cols.append(torch.cat([g.reshape(-1) for g in grads]).double())
    return torch.stack(cols), r


def row_gradients(backend, H: torch.Tensor) -> List[torch.Tensor]:
    """d rows / d H for every supervised mode: a list over the band of (n_rows_per_mode, ...) tensors."""
    out = []
    for b, i in enumerate(backend.band):
        with torch.enable_grad():
            hs = H[..., i].contiguous().detach().clone().requires_grad_(True)
            rows_i = backend.mode_rows(hs, i)
            eye = torch.eye(rows_i.numel(), device=hs.device, dtype=rows_i.dtype)
            try:
                G = torch.autograd.grad(
                    rows_i,
                    hs,
                    grad_outputs=eye,
                    is_grads_batched=True,
                    retain_graph=True,
                )[0]
            except Exception:
                G = torch.stack(
                    [
                        torch.autograd.grad(
                            rows_i[k], hs, retain_graph=(k < rows_i.numel() - 1)
                        )[0]
                        for k in range(rows_i.numel())
                    ]
                )
        out.append(G.detach())
    return out


# ----------------------------------------------------------------------------- the loop
def gauss_newton(model: SpectralModel, cfg: GaussNewtonConfig, log=print) -> dict:
    """Damped Gauss-Newton steps on the amplitudes against the flux rows."""
    mask = model.column_mask(cfg.carriers_only)
    history = []
    for it in range(cfg.steps):
        J, r = jacobian_autograd(model)
        Jm = J * mask[None, :]
        R = r.numel()
        step = Jm.T @ torch.linalg.solve(
            Jm @ Jm.T + cfg.mu * torch.eye(R, device=J.device, dtype=torch.float64), r
        )
        model.set_amps(model.flat_amps() + step)
        history.append(float(r.norm()))
        log(f"  gn step {it + 1}/{cfg.steps}  |r| {history[-1]:.4e}")
        del J, Jm, r, step
    return {"residual": history}
