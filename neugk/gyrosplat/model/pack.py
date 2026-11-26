"""Byte accounting and model files.

The compression ratio is the number of stored reals in the raw snapshot divided by the
number of reals the representation needs: 17 per atom (centre 5, spatial Cholesky 6,
velocity Cholesky 3, amplitude 2, carrier wavenumber 1). Envelopes are independent by
default. With structural sharing, each group stores 14 geometry reals once, plus
3 reals per carrier. Counts are read from the actual bank, not inferred from a preset.
Atom parameters
are counted at `bits` per real (32 in the paper's rows; `storage_bits: 16` rounds them to
half precision first, so the metrics see what a reader of a 16-bit file gets).
`packed_bytes` counts the parameter blob after zlib, excluding normalization and headers.
It is not the size of the complete `save_model` checkpoint.
"""
from __future__ import annotations

import zlib
from typing import Dict, List, Optional, Sequence

import torch

from neugk.gyrosplat.model.atoms import AtomBank


def stored_reals(
    banks: Sequence[AtomBank],
    reals_per_atom: int = 17,
    reals_per_carrier: Optional[float] = None,
    bits: int = 32,
) -> Dict[str, float]:
    """Actual trainable representation at `bits`, in 32-bit units.

    Plain Gaussian baselines may omit their known-zero wavenumbers (16 reals).
    Geometry is counted once only when the bank structurally enforces sharing.
    """
    _validate_layout(banks, reals_per_atom)
    if reals_per_carrier is not None:
        for b in banks:
            expected = 3 + 14 / b.envelope_group_size
            if b.n_carriers and abs(reals_per_carrier - expected) > 1e-10:
                raise ValueError(
                    f"This bank requires {expected:g} reals per carrier (17 reals if independent)"
                )
    atoms = sum(
        sum(p.numel() for p in (b.mu, b.L_phys_raw, b.L_vel_raw, b.amps, b.ky))
        - (17 - reals_per_atom) * b.n_base
        for b in banks
    )
    atoms = atoms * bits / 32.0
    return {"atoms": atoms, "total": atoms}


def compression_ratios(raw_reals: int, stored: Dict[str, float]) -> Dict[str, float]:
    return {"CR": raw_reals / max(stored["total"], 1)}


@torch.no_grad()
def quantize_banks(banks: Sequence[AtomBank], bits: int = 16) -> None:
    """Round every atom parameter through the storage precision, in place."""
    if bits >= 32:
        return
    for b in banks:
        for p in (b.mu, b.L_phys_raw, b.L_vel_raw, b.amps, b.ky):
            p.data.copy_(p.data.to(torch.float16).to(p.dtype))


@torch.no_grad()
def _validate_layout(banks, reals_per_atom):
    if reals_per_atom not in (16, 17):
        raise ValueError("An atom requires 17 reals, or 16 with known-zero wavenumber")
    if reals_per_atom == 16 and any(
        torch.count_nonzero(b.ky[: b.n_base]).item() for b in banks
    ):
        raise ValueError("Cannot omit nonzero base wavenumbers")


@torch.no_grad()
def packed_parameters(
    banks: Sequence[AtomBank],
    bits: int = 32,
    tied_k: Optional[int] = None,
    *,
    reals_per_atom: int = 17,
) -> bytes:
    """zlib-compressed parameters, retaining every unique fitted envelope.

    `tied_k` is accepted for old callers but never drops envelope rows. Sharing
    must be declared in the bank's representation, not by this packing hint. The blob
    is ordered by bank, then base/carriers, then mu/L_phys/L_vel/amps/ky. Shapes,
    normalization and configuration are external; use save_model for a checkpoint.
    """
    _validate_layout(banks, reals_per_atom)
    dtype = torch.float16 if bits <= 16 else torch.float32
    chunks = []
    for b in banks:
        base = slice(0, b.n_base)
        chunks += [
            b.mu.data[base],
            b.L_phys_raw.data[base],
            b.L_vel_raw.data[base],
            b.amps.data[base],
        ]
        if reals_per_atom == 17:
            chunks.append(b.ky.data[base, None])
        if b.n_carriers == 0:
            continue
        carr = slice(b.n_base, b.N)
        chunks += [
            b.mu.data[carr],
            b.L_phys_raw.data[carr],
            b.L_vel_raw.data[carr],
            b.amps.data[carr],
            b.ky.data[carr, None],
        ]
    values = torch.cat([c.reshape(-1) for c in chunks]) if chunks else torch.empty(0)
    blob = values.to(dtype).cpu().numpy().tobytes()
    return zlib.compress(blob, 9)


def packed_bytes(banks, bits=32, tied_k=None, *, reals_per_atom=17) -> int:
    """Compressed parameter bytes, excluding normalization and checkpoint headers."""
    return len(packed_parameters(banks, bits, tied_k, reals_per_atom=reals_per_atom))


def save_model(
    path: str, banks: List[AtomBank], stats, cfg, extra: Optional[dict] = None
):
    from omegaconf import OmegaConf

    payload = {
        "banks": [b.state() for b in banks],
        "stats": [vars(s) for s in stats],
        "config": OmegaConf.to_container(OmegaConf.structured(cfg)),
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_model(path: str, device="cpu"):
    from neugk.gyrosplat.model.normalize import NormStats

    p = torch.load(path, map_location="cpu", weights_only=False)
    banks = [AtomBank.from_state(s, device) for s in p["banks"]]
    stats = [NormStats(**s) for s in p["stats"]]
    return banks, stats, p["config"], p["extra"]
