"""ZFP and SZ3 on the real-space object, searched to the byte budget.

The two channels go into one float32 array of shape (2, velocity, s . y, x), the layout of
PINC's evaluation, and the error bound is searched in log space for the
compressed size closest to 4 x budget reals: the rate of these codecs moves in steps, so the
closest reachable rate can lie on either side of the target.  The achieved reals are the
compressed bytes divided by four: byte-true, unlike the value-count accounting of the top-K
codes."""
from __future__ import annotations

import math
from typing import List

import numpy as np
import torch


def _zfp(arr: np.ndarray, tol: float):
    import zfpy

    c = zfpy.compress_numpy(arr, tolerance=tol)
    return len(c), zfpy.decompress_numpy(c)


def _sz3(arr: np.ndarray, eb: float):
    from pysz import sz as SZ, szConfig, szErrorBoundMode

    cfg = szConfig(tuple(int(x) for x in arr.shape))
    cfg.errorBoundMode = szErrorBoundMode.ABS
    cfg.absErrorBound = float(eb)
    comp, _ = SZ.compress(arr, cfg)
    dec, _ = SZ.decompress(comp, arr.dtype, arr.shape)
    return int(np.asarray(comp).nbytes), np.asarray(dec, dtype=np.float32)


def _closest(compress_fn, target_bytes: int, lo: float, hi: float, tol=0.02, iters=14):
    best = None
    llo, lhi = math.log(lo), math.log(hi)
    for _ in range(iters):
        p = math.exp(0.5 * (llo + lhi))
        nb, dec = compress_fn(p)
        err = abs(math.log(nb / target_bytes))
        if best is None or err < best[0]:
            best = (err, p, nb, dec)
        if err < tol:
            break
        if nb > target_bytes:
            llo = math.log(p)
        else:
            lhi = math.log(p)
    return best[1:]


def error_bounded(
    real_fields: List[torch.Tensor], budget_reals: int, codec: str = "zfp"
):
    fn = _zfp if codec == "zfp" else _sz3
    out, achieved, extra = [], [], {"codec": codec, "bounds": []}
    for f in real_fields:
        shape = tuple(f.shape[1:])
        scale = float(f.abs().max())
        n0, n1, n2, n3, n4 = shape
        arr = f.permute(0, 1, 2, 3, 5, 4).reshape(2, n0 * n1, n2 * n4, n3)
        arr = np.ascontiguousarray(arr.cpu().numpy().astype(np.float32))
        # the closest rate can sit at a bound far above the largest value (ZFP's rate
        # halves with every doubling of the tolerance)
        p, nb, dec = _closest(
            lambda b: fn(arr, b), 4 * budget_reals, 1e-6 * scale, 1e3 * scale
        )
        rec = torch.as_tensor(np.ascontiguousarray(dec))
        rec = rec.reshape(2, n0, n1, n2, n4, n3).permute(0, 1, 2, 3, 5, 4)
        out.append(rec.reshape((2,) + shape).contiguous().to(f.device))
        achieved.append(int(math.ceil(nb / 4)))
        extra["bounds"].append(p)
    return out, achieved, extra
