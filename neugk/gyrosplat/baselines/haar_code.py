"""Differentiable, one-level periodized Haar codes for physics refinement.

Coefficient selection matches the PyWavelets baseline, including odd grid sizes.
The decoder is a separable Torch inverse, so the same physical objective can
differentiate with respect to retained real coefficients on the GPU.
"""
from __future__ import annotations

import math
import numpy as np
import torch

from neugk.gyrosplat.baselines.gn_on_code import LinearCode
from neugk.gyrosplat.baselines.wavelet import _forward, _largest


def inverse_haar(coefficients: torch.Tensor, shape) -> torch.Tensor:
    """Inverse of PyWavelets' one-level, periodized coeffs_to_array layout."""
    result = coefficients
    for axis in reversed(range(len(shape))):
        low, high = result.chunk(2, dim=axis)
        even = (low + high) / math.sqrt(2)
        odd = (low - high) / math.sqrt(2)
        result = torch.stack((even, odd), dim=axis + 1).flatten(axis, axis + 1)
        result = result.narrow(axis, 0, shape[axis])
    return result


def haar_linear_code(backend, real_fields, budget_reals: int):
    """Return (reconstruction, achieved counts, LinearCode, portable layout).

    The complex values in LinearCode pack pairs of independent real coefficients;
    they are not complex Haar coefficients. An odd count gets one unused zero pad.
    The layout and the final code values suffice to decode a stored representation."""
    parts, values, achieved = [], [], []
    for f in real_fields:
        arrays = [
            _forward(ch.detach().cpu().numpy().astype(np.float64), "haar", 1)[0]
            for ch in f
        ]
        coefficients = np.stack(arrays)
        mask = _largest(np.abs(coefficients), budget_reals)
        indices = np.flatnonzero(mask.reshape(-1))
        v = torch.as_tensor(
            coefficients.reshape(-1)[indices], dtype=torch.float32, device=f.device
        )
        achieved.append(v.numel())
        if v.numel() % 2:
            v = torch.cat((v, v.new_zeros(1)))
        values.append(torch.view_as_complex(v.reshape(-1, 2).contiguous()))
        parts.append(
            dict(
                shape=list(f.shape[1:]),
                coefficient_shape=list(coefficients.shape),
                indices=torch.as_tensor(indices, dtype=torch.int64, device=f.device),
            )
        )

    def decode(vals):
        fields = []
        for v, part in zip(vals, parts):
            real = torch.view_as_real(v).reshape(-1)[: part["indices"].numel()]
            flat = real.new_zeros(math.prod(part["coefficient_shape"]))
            coefficients = flat.index_put((part["indices"],), real).reshape(
                part["coefficient_shape"]
            )
            fields.append(
                torch.stack([inverse_haar(ch, part["shape"]) for ch in coefficients])
            )
        return backend.to_spectral(fields)

    code = LinearCode(values, decode)
    with torch.no_grad():
        reconstruction = decode(values)
    return reconstruction, achieved, code, parts


def code_from_saved(backend, parts, values, device="cpu"):
    """Reconstruct the differentiable decoder from saved indices and code values."""
    layout = [{**p, "indices": p["indices"].to(device).long()} for p in parts]

    def decode(vals):
        fields = []
        for v, part in zip(vals, layout):
            real = torch.view_as_real(v).reshape(-1)[: part["indices"].numel()]
            flat = real.new_zeros(math.prod(part["coefficient_shape"]))
            coefficients = flat.index_put((part["indices"],), real).reshape(
                part["coefficient_shape"]
            )
            fields.append(
                torch.stack([inverse_haar(ch, part["shape"]) for ch in coefficients])
            )
        return backend.to_spectral(fields)

    return LinearCode([v.to(device) for v in values], decode)
