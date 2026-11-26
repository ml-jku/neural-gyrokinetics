"""Wavelet top-K on the real-space object: a separable PyWavelets transform over all five
axes of each stored real channel (periodisation, one level), the K largest real coefficients
of both channels kept one by one, as in the evaluation of PINC.
"""
from __future__ import annotations

from typing import List

import numpy as np
import torch


def _forward(arr: np.ndarray, wavelet: str, level: int):
    import pywt

    coeffs = pywt.wavedecn(arr, wavelet, mode="periodization", level=level)
    flat, slices = pywt.coeffs_to_array(coeffs)
    return flat, slices


def _inverse(flat: np.ndarray, slices, wavelet: str, shape):
    import pywt

    coeffs = pywt.array_to_coeffs(flat, slices, output_format="wavedecn")
    return pywt.waverecn(coeffs, wavelet, mode="periodization")[
        tuple(slice(0, n) for n in shape)
    ]


def wavelet_topk(
    real_fields: List[torch.Tensor],
    budget_reals: int,
    wavelet: str = "bior4.4",
    level: int = 1,
):
    """real_fields: per species (2, ...) channels.  Returns (reconstructions, achieved reals, extra)."""
    out, achieved = [], []
    for f in real_fields:
        re = f[0].cpu().numpy().astype(np.float64)
        im = f[1].cpu().numpy().astype(np.float64)
        a_re, slices = _forward(re, wavelet, level)
        a_im, _ = _forward(im, wavelet, level)
        both = _largest(np.stack([np.abs(a_re), np.abs(a_im)]), budget_reals)
        m_re, m_im = both[0], both[1]
        achieved.append(int(both.sum()))
        r_re = _inverse(np.where(m_re, a_re, 0.0), slices, wavelet, re.shape)
        r_im = _inverse(np.where(m_im, a_im, 0.0), slices, wavelet, im.shape)
        rec = [torch.as_tensor(r, dtype=torch.float32) for r in (r_re, r_im)]
        out.append(torch.stack(rec, 0).to(f.device))
    return out, achieved, {"wavelet": wavelet, "level": level}


def _largest(score: np.ndarray, k: int) -> np.ndarray:
    """Mask of the k largest entries."""
    flat = score.reshape(-1)
    k = min(k, flat.size)
    m = np.zeros(flat.size, dtype=bool)
    if k > 0:
        m[np.argpartition(-flat, k - 1)[:k]] = True
    return m.reshape(score.shape)
