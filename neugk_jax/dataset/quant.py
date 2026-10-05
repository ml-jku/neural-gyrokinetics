"""Quantized shard format of the preprocessed dataset.

A quantized shard sits next to its fp32 ``foo.bin`` as ``foo.<bits>.bin``::

    fp16 / bf16:   raw 16-bit values, no header
    i8:            float32 scale (4 bytes) || raw int8 values
    i4:            float32 scale (4 bytes) || raw uint8 nibble-packed
                   (two int4 values per byte: low nibble = index 2k,
                    high nibble = index 2k+1)
"""

from __future__ import annotations

import os

import numpy as np

from neugk_jax.utils import atomic_write

SUFFIX = {"fp16": ".fp16.bin", "bf16": ".bf16.bin", "i8": ".i8.bin", "i4": ".i4.bin"}
# bytes of the float32 scale in front of the integer payloads
HEADER_BYTES = 4


def has_header(bits: str) -> bool:
    return bits in ("i8", "i4")


def payload_size(bits: str, n_elems: int) -> int:
    return (n_elems + 1) // 2 if bits == "i4" else n_elems


def sibling(fp32_path: str, bits: str) -> str:
    # foo.bin -> foo.<bits>.bin
    return fp32_path.removesuffix(".bin") + SUFFIX[bits]


def resolve(fp32_path: str, prefer: str) -> tuple[str, str]:
    """``(path, bits)`` to read: the ``prefer`` sibling when it exists, else the fp32 shard."""
    if prefer != "fp32":
        cand = sibling(fp32_path, prefer)
        if os.path.exists(cand):
            return cand, prefer
    return fp32_path, "fp32"


def quantize(arr_f32: np.ndarray, bits: str) -> tuple[np.ndarray, np.float32 | None]:
    """Quantize a flat fp32 array to ``bits`` precision.

    Returns ``(payload, scale)``. ``scale`` is ``None`` for IEEE 16-bit
    formats (the dtype itself encodes magnitude). For int8/int4 it's the
    per-tensor symmetric quantization scale (``max(|x|) / qmax``).
    """
    if bits == "fp16":
        return arr_f32.astype(np.float16), None
    if bits == "bf16":
        from ml_dtypes import bfloat16

        return arr_f32.astype(bfloat16), None
    if bits in ("i8", "i4"):
        qmin, qmax = (-128, 127) if bits == "i8" else (-8, 7)
        mx = float(np.max(np.abs(arr_f32)))
        scale = np.float32(mx / qmax) if mx > 0 else np.float32(1.0)
        q = np.clip(np.round(arr_f32 / scale), qmin, qmax).astype(np.int8)
        if bits == "i8":
            return q, scale
        # nibble-pack: two int4 values per byte, low nibble = idx 2k, high = 2k+1
        if q.size % 2:
            q = np.concatenate([q, np.zeros(1, dtype=np.int8)])
        lo = (q[0::2].astype(np.uint8)) & 0x0F
        hi = (q[1::2].astype(np.uint8)) & 0x0F
        return ((hi << 4) | lo).astype(np.uint8), scale
    raise ValueError(f"unknown bits={bits!r}; expected one of {list(SUFFIX)}")


def dequantize(
    payload: np.ndarray, scale: np.float32 | None, bits: str, n_elems: int
) -> np.ndarray:
    """Inverse of :func:`quantize` — returns fp32."""
    if bits in ("fp16", "bf16"):
        return payload.astype(np.float32)
    if bits == "i8":
        return payload.astype(np.float32) * float(scale)
    if bits == "i4":
        lo = payload & 0x0F
        hi = (payload >> 4) & 0x0F
        # sign-extend 4-bit two's complement
        lo = np.where(lo >= 8, lo.astype(np.int8) - 16, lo.astype(np.int8))
        hi = np.where(hi >= 8, hi.astype(np.int8) - 16, hi.astype(np.int8))
        out = np.empty(payload.size * 2, dtype=np.int8)
        out[0::2] = lo
        out[1::2] = hi
        return out[:n_elems].astype(np.float32) * float(scale)
    raise ValueError(f"unknown bits={bits!r}")


def payload_dtype(bits: str):
    if bits == "bf16":
        from ml_dtypes import bfloat16

        return bfloat16
    return {"fp16": np.float16, "i8": np.int8, "i4": np.uint8}[bits]


def write(dst: str, payload: np.ndarray, scale: np.float32 | None) -> int:
    """Atomic write of one quantized shard. Returns bytes written."""

    def dump(f):
        if scale is not None:
            f.write(np.float32(scale).tobytes())
        f.write(payload.tobytes())

    atomic_write(dst, dump)
    return os.path.getsize(dst)


def read(path: str, bits: str, n_elems: int) -> np.ndarray:
    """Read and dequantize a quantized shard of ``n_elems`` values."""
    with open(path, "rb") as f:
        scale = (
            np.frombuffer(f.read(HEADER_BYTES), dtype=np.float32)[0] if has_header(bits) else None
        )
        payload = np.frombuffer(f.read(), dtype=payload_dtype(bits))
    if payload.size != payload_size(bits, n_elems):
        raise IOError(
            f"{path}: expected {payload_size(bits, n_elems)} {bits} values, got {payload.size}"
        )
    return dequantize(payload, scale, bits, n_elems)


def roundtrip(arr_f32: np.ndarray, bits: str) -> np.ndarray:
    payload, scale = quantize(arr_f32.ravel(), bits)
    return dequantize(payload, scale, bits, arr_f32.size).reshape(arr_f32.shape)
