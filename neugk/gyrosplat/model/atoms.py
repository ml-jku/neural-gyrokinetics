"""The atom bank.

An atom is a separable Gaussian, one 2-D Gaussian in velocity times one 3-D Gaussian in
space, with a complex amplitude and an optional plane-wave carrier along the binormal axis:

    a_n(v, s) = c_n * G(v; mu_v, L_v) * G(s; mu_s, L_s) * exp(i ky (y - mu_y))

Coordinates are grid indices divided by the axis length, so every axis lives in [0, 1) and a
carrier with ky = 2*pi*m sits exactly on Fourier mode m.  The Cholesky factors are stored
flat with a softplus-positive diagonal.  Because the field is linear in the amplitudes and
separable on a tensor-product grid, the bank exposes its velocity factor V (n_v x N) and its
spatial factors (n_s x N) for a block of spatial rows; everything else (rendering, the
amplitude solve, the Gauss-Newton) is built on those factors.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F

VEL = slice(0, 2)  # velocity axes of a coordinate row
PHYS = slice(2, 5)  # spatial axes
YIDX = 4  # the binormal axis (the spectral axis; the carrier axis)


def lower_triangular(flat: torch.Tensor, d: int) -> torch.Tensor:
    """(N, d(d+1)/2) flat parameters -> (N, d, d) lower-triangular factors with a positive diagonal."""
    n = flat.shape[0]
    L = flat.new_zeros(n, d, d)
    idx = torch.arange(d, device=flat.device)
    L[:, idx, idx] = torch.clamp(F.softplus(flat[:, :d]), min=1e-5)
    if d > 1:
        r, c = torch.tril_indices(d, d, offset=-1, device=flat.device)
        L[:, r, c] = flat[:, d:]
    return L


def inv_softplus(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = x.clamp(min=eps)
    return x + torch.log(-torch.expm1(-x))


def make_grid(shape: Sequence[int], device="cpu") -> torch.Tensor:
    """The tensor-product coordinate grid, (n_voxel, 5), each axis as index / length."""
    axes = [torch.arange(n, device=device, dtype=torch.float32) / n for n in shape]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 5)


def subgrids(grid: torch.Tensor, shape: Sequence[int]):
    """The velocity (n_v, 2) and spatial (n_s, 3) sub-grids of a tensor-product grid."""
    n_s = shape[2] * shape[3] * shape[4]
    return grid[::n_s, VEL].contiguous(), grid[:n_s, PHYS].contiguous()


class AtomBank(torch.nn.Module):
    """N atoms with independent amplitudes/frequencies and optionally shared geometry.

    Geometry has N rows by default. With envelope_group_size=k, it has n_base
    base rows followed by one row per contiguous group of k carriers. Autograd
    sums the carriers' geometry gradients into that single trainable row.
    """

    def __init__(
        self,
        mu,
        L_phys_raw,
        L_vel_raw,
        amps,
        ky=None,
        n_base: Optional[int] = None,
        shape: Optional[Sequence[int]] = None,
        envelope_group_size: int = 1,
    ):
        super().__init__()
        n = amps.shape[0]
        self.N = n
        self.n_base = n if n_base is None else int(n_base)
        self.envelope_group_size = int(envelope_group_size)
        if self.envelope_group_size < 1 or not 0 <= self.n_base <= n:
            raise ValueError("Invalid base count or envelope group size")
        if self.n_carriers % self.envelope_group_size:
            raise ValueError("Carrier count must be divisible by envelope group size")
        n_geometry = self.n_base + self.n_carriers // self.envelope_group_size
        if (tuple(mu.shape), tuple(L_phys_raw.shape), tuple(L_vel_raw.shape)) != (
            (n_geometry, 5),
            (n_geometry, 6),
            (n_geometry, 3),
        ):
            raise ValueError("Geometry rows do not match the declared envelope sharing")
        self.mu = torch.nn.Parameter(mu)
        self.L_phys_raw = torch.nn.Parameter(L_phys_raw)
        self.L_vel_raw = torch.nn.Parameter(L_vel_raw)
        self.amps = torch.nn.Parameter(amps)
        self.ky = torch.nn.Parameter(
            torch.zeros(n, device=mu.device) if ky is None else ky, requires_grad=False
        )
        self.shape = None if shape is None else tuple(int(s) for s in shape)

    # -- derived quantities ---------------------------------------------------------------
    def expanded_geometry(self, value: torch.Tensor) -> torch.Tensor:
        """Per-atom view of a unique geometry tensor, with differentiable sharing."""
        if self.envelope_group_size == 1:
            return value
        return torch.cat(
            (
                value[: self.n_base],
                value[self.n_base :].repeat_interleave(self.envelope_group_size, dim=0),
            )
        )

    def L_phys(self) -> torch.Tensor:
        return self.expanded_geometry(lower_triangular(self.L_phys_raw, 3))

    def L_vel(self) -> torch.Tensor:
        return self.expanded_geometry(lower_triangular(self.L_vel_raw, 2))

    @property
    def geometry_carrier_mask(self) -> torch.Tensor:
        """Routing mask for unique geometry rows (amplitudes use carrier_mask)."""
        return torch.arange(self.mu.shape[0], device=self.mu.device) >= self.n_base

    @property
    def carrier_mask(self) -> torch.Tensor:
        m = torch.zeros(self.N, dtype=torch.bool, device=self.mu.device)
        m[self.n_base :] = True
        return m

    @property
    def n_carriers(self) -> int:
        return self.N - self.n_base

    # -- the separable factors ------------------------------------------------------------
    def velocity_factor(self, vel_grid: torch.Tensor) -> torch.Tensor:
        """V (n_v, N): the velocity Gaussian of every atom on the velocity grid."""
        Lv = self.L_vel()
        dv = vel_grid.unsqueeze(1) - self.expanded_geometry(self.mu)[:, VEL].unsqueeze(
            0
        )
        uv = torch.linalg.solve_triangular(
            Lv, dv.permute(1, 2, 0), upper=False
        ).permute(2, 0, 1)
        return torch.exp(-0.5 * (uv**2).sum(-1))

    def spatial_factors(self, phys_block: torch.Tensor):
        """(Pre, Pim), each (n_block, N): the spatial Gaussian times cos / sin of the carrier phase."""
        Lp = self.L_phys()
        mu = self.expanded_geometry(self.mu)
        dp = phys_block.unsqueeze(1) - mu[:, PHYS].unsqueeze(0)
        up = torch.linalg.solve_triangular(
            Lp, dp.permute(1, 2, 0), upper=False
        ).permute(2, 0, 1)
        P = torch.exp(-0.5 * (up**2).sum(-1))
        y = phys_block[:, YIDX - 2]
        theta = self.ky.unsqueeze(0) * (y.unsqueeze(1) - mu[:, YIDX].unsqueeze(0))
        return P * torch.cos(theta), P * torch.sin(theta)

    def factors(self, vel_grid: torch.Tensor, phys_block: torch.Tensor):
        Pre, Pim = self.spatial_factors(phys_block)
        return self.velocity_factor(vel_grid), Pre, Pim

    # -- bookkeeping ----------------------------------------------------------------------
    def subset(self, idx: torch.Tensor) -> "AtomBank":
        """Independent copy of selected atoms (no gradient link).

        Shared geometry is expanded for this copy, used by the frozen-geometry
        subset amplitude solve; the source bank's sharing is preserved.
        """
        return AtomBank(
            self.expanded_geometry(self.mu.data)[idx].clone(),
            self.expanded_geometry(self.L_phys_raw.data)[idx].clone(),
            self.expanded_geometry(self.L_vel_raw.data)[idx].clone(),
            self.amps.data[idx].clone(),
            self.ky.data[idx].clone(),
            n_base=int((idx < self.n_base).sum()),
            shape=self.shape,
        )

    def state(self) -> dict:
        return {
            "mu": self.mu.data.cpu(),
            "L_phys_raw": self.L_phys_raw.data.cpu(),
            "L_vel_raw": self.L_vel_raw.data.cpu(),
            "amps": self.amps.data.cpu(),
            "ky": self.ky.data.cpu(),
            "n_base": self.n_base,
            "shape": self.shape,
            "envelope_group_size": self.envelope_group_size,
        }

    @classmethod
    def from_state(cls, d: dict, device="cpu") -> "AtomBank":
        return cls(
            d["mu"].to(device),
            d["L_phys_raw"].to(device),
            d["L_vel_raw"].to(device),
            d["amps"].to(device),
            d["ky"].to(device),
            n_base=d.get("n_base"),
            shape=d.get("shape"),
            envelope_group_size=d.get("envelope_group_size", 1),
        )

    def share_carrier_envelopes(self, group_size: int) -> "AtomBank":
        """Lossless conversion of exactly equal envelopes into unique parameters.

        Rejects divergent fitted envelopes; this is not a projection or an average.
        Normally called once immediately after grouped carrier initialization.
        """
        if group_size < 1 or self.n_carriers % group_size:
            raise ValueError("Invalid envelope group size")
        compact = []
        for value in (self.mu, self.L_phys_raw, self.L_vel_raw):
            expanded = self.expanded_geometry(value.detach())
            grouped = expanded[self.n_base :].reshape(-1, group_size, expanded.shape[1])
            first = grouped[:, :1]
            if not torch.equal(grouped, first.expand_as(grouped)):
                raise ValueError("Cannot share divergent fitted envelopes losslessly")
            compact.append(torch.cat((expanded[: self.n_base], first[:, 0])).clone())
        return AtomBank(
            *compact,
            self.amps.detach().clone(),
            self.ky.detach().clone(),
            n_base=self.n_base,
            shape=self.shape,
            envelope_group_size=group_size,
        )


def concat(base: AtomBank, extra: AtomBank) -> AtomBank:
    """Append `extra` as carriers to `base` (the result's n_base is base.N)."""
    if base.envelope_group_size != 1 or extra.envelope_group_size != 1:
        raise ValueError(
            "Concatenate independent banks before declaring envelope sharing"
        )
    return AtomBank(
        torch.cat([base.mu.data, extra.mu.data]),
        torch.cat([base.L_phys_raw.data, extra.L_phys_raw.data]),
        torch.cat([base.L_vel_raw.data, extra.L_vel_raw.data]),
        torch.cat([base.amps.data, extra.amps.data]),
        torch.cat([base.ky.data, extra.ky.data]),
        n_base=base.N,
        shape=base.shape,
    )
