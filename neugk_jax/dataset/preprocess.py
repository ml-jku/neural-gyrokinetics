"""Dataset preprocessing for the JAX port.

Currently a thin entrypoint with mode dispatch (similar to upstream
``neugk/dataset/preprocess.py``'s ``--metadata_only`` / ``--geometry_only``
flags). The full raw GKW → fp32 preprocessing path is still on the torch
side (it does the GKW IFFT, geometry parsing, flux verification against
the legacy integrator) and is left intentionally torch-bound for now —
all the moving parts live in upstream ``neugk/`` and reproducing them
brings in `load_geometry`, `K_files`, `parse_input_dat`, ... none of
which the JAX trainer needs at runtime.

What this file does cover today: take an existing fp32 ``.bin`` shard
and produce side-by-side quantized siblings (``.bf16.bin``, ``.fp16.bin``,
``.i8.bin``, ``.i4.bin``) used by the dataloader's
``prefer_dtype="…"`` graceful-fallback path.

Layout per file::

    fp16 / bf16:   raw 16-bit values, no header
    i8:            float32 scale (4 bytes) || raw int8 values
    i4:            float32 scale (4 bytes) || raw uint8 nibble-packed
                   (two int4 values per byte: low nibble = index 2k,
                    high nibble = index 2k+1)

Reads round-trip via ``neugk_jax.dataset.backend._read_quantized`` which
dequantizes back to fp32 transparently.

Usage::

    python -m neugk_jax.dataset.preprocess --mode=quantize \\
        --path /local00/bioinf/galletti/preprocessed_kvikio \\
        --trajs 'iteration_{0-299}_ifft_realpotens' \\
        --bits bf16 --num-workers 8

    # idempotent — skips files whose quantized sibling already exists
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


# ---------- file naming -------------------------------------------------
_DTYPE_SUFFIX = {
    "fp16": ".fp16.bin",
    "bf16": ".bf16.bin",
    "i8":   ".i8.bin",
    "i4":   ".i4.bin",
}


def quantized_sibling(fp32_path: str, bits: str) -> str:
    """``foo.bin`` → ``foo.<dtype>.bin`` (the side-by-side quantized shard)."""
    if not fp32_path.endswith(".bin"):
        return fp32_path + _DTYPE_SUFFIX[bits]
    return fp32_path[:-4] + _DTYPE_SUFFIX[bits]


# ---------- quantize / dequantize core ---------------------------------
def quantize_array(arr_f32: np.ndarray, bits: str) -> tuple[np.ndarray, np.float32 | None]:
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
    if bits == "i8":
        qmax = 127
        mx = float(np.max(np.abs(arr_f32)))
        scale = np.float32(mx / qmax) if mx > 0 else np.float32(1.0)
        q = np.clip(np.round(arr_f32 / scale), -128, 127).astype(np.int8)
        return q, scale
    if bits == "i4":
        qmax = 7
        mx = float(np.max(np.abs(arr_f32)))
        scale = np.float32(mx / qmax) if mx > 0 else np.float32(1.0)
        q = np.clip(np.round(arr_f32 / scale), -8, 7).astype(np.int8)
        # nibble-pack: two int4 values per byte. low nibble = idx 2k, high = 2k+1.
        if q.size % 2:
            q = np.concatenate([q, np.zeros(1, dtype=np.int8)])
        lo = (q[0::2].astype(np.uint8)) & 0x0F
        hi = (q[1::2].astype(np.uint8)) & 0x0F
        packed = (hi << 4) | lo
        return packed.astype(np.uint8), scale
    raise ValueError(f"unknown bits={bits!r}; expected one of {list(_DTYPE_SUFFIX)}")


def dequantize_array(payload: np.ndarray, scale: np.float32 | None, bits: str, n_elems: int) -> np.ndarray:
    """Inverse of :func:`quantize_array` — returns fp32."""
    if bits == "fp16":
        return payload.astype(np.float32)
    if bits == "bf16":
        return payload.astype(np.float32)
    if bits == "i8":
        return payload.astype(np.float32) * float(scale)
    if bits == "i4":
        # unpack nibbles
        lo = payload & 0x0F
        hi = (payload >> 4) & 0x0F
        # sign-extend 4-bit two's complement
        lo = np.where(lo >= 8, lo.astype(np.int8) - 16, lo.astype(np.int8))
        hi = np.where(hi >= 8, hi.astype(np.int8) - 16, hi.astype(np.int8))
        out = np.empty(payload.size * 2, dtype=np.int8)
        out[0::2] = lo
        out[1::2] = hi
        out = out[:n_elems]  # drop the padding from quantize_array if any
        return out.astype(np.float32) * float(scale)
    raise ValueError(f"unknown bits={bits!r}")


# ---------- on-disk I/O ------------------------------------------------
def write_quantized(dst: str, payload: np.ndarray, scale: np.float32 | None) -> int:
    """Atomic-ish write of one quantized shard. Returns bytes written."""
    tmp = dst + ".tmp"
    with open(tmp, "wb") as f:
        if scale is not None:
            f.write(np.float32(scale).tobytes())
        f.write(payload.tobytes())
    n = os.path.getsize(tmp)
    os.replace(tmp, dst)
    return n


def read_quantized(path: str, bits: str, n_elems: int, *, return_raw: bool = False):
    """Read a quantized shard and (optionally) dequantize.

    ``return_raw=True`` returns ``(payload, scale)`` without dequantizing —
    used by the GPU-side bf16 / fp16 paths that prefer to keep the read in
    its original dtype until after dlpack hand-off to JAX.
    """
    with open(path, "rb") as f:
        scale = None
        if bits in ("i8", "i4"):
            scale = np.frombuffer(f.read(4), dtype=np.float32)[0]
        if bits == "fp16":
            payload = np.frombuffer(f.read(), dtype=np.float16)
        elif bits == "bf16":
            from ml_dtypes import bfloat16
            payload = np.frombuffer(f.read(), dtype=bfloat16)
        elif bits == "i8":
            payload = np.frombuffer(f.read(), dtype=np.int8)
        elif bits == "i4":
            payload = np.frombuffer(f.read(), dtype=np.uint8)
        else:
            raise ValueError(f"unknown bits={bits!r}")
    if return_raw:
        return payload, scale
    return dequantize_array(payload, scale, bits, n_elems)


# ---------- traversal --------------------------------------------------
def _resolve_trajs(root: str, spec) -> list[str]:
    """Expand a brace pattern (or accept an explicit list) → traj dir basenames."""
    if isinstance(spec, list) and len(spec) != 1:
        return list(spec)
    if isinstance(spec, list):
        spec = spec[0]
    m = re.match(r"^(.*?)\{([^}]+)\}(.*?)$", spec)
    if not m:
        return [spec]
    prefix, ranges_str, suffix = m.groups()
    nums = []
    for part in ranges_str.split(","):
        if "-" in part:
            lo, hi = map(int, part.split("-"))
            nums.extend(range(lo, hi + 1))
        else:
            nums.append(int(part))
    return [f"{prefix}{n}{suffix}" for n in nums]


def _src_bins(data_dir: str) -> list[str]:
    """List fp32 .bin sources (timestep + poten) inside ``traj/data``."""
    if not os.path.isdir(data_dir):
        return []
    out = []
    for name in os.listdir(data_dir):
        if not name.endswith(".bin"):
            continue
        if any(name.endswith(suf) for suf in _DTYPE_SUFFIX.values()):
            continue
        if not (name.startswith("timestep_") or name.startswith("poten_")):
            continue
        out.append(os.path.join(data_dir, name))
    return sorted(out)


def _quantize_file(src: str, bits: str, force: bool) -> tuple[str, int, str]:
    dst = quantized_sibling(src, bits)
    if os.path.exists(dst) and not force:
        return src, 0, "skip"
    try:
        arr = np.fromfile(src, dtype=np.float32)
        payload, scale = quantize_array(arr, bits)
        n = write_quantized(dst, payload, scale)
        return src, n, "written"
    except Exception as e:
        return src, 0, f"error: {e}"


def _process_traj(traj_dir: str, bits: str, force: bool) -> tuple[str, int, int, int]:
    files = _src_bins(os.path.join(traj_dir, "data"))
    n_written = n_skipped = bytes_written = 0
    for src in files:
        _, n, status = _quantize_file(src, bits, force=force)
        if status == "written":
            n_written += 1
            bytes_written += n
        elif status == "skip":
            n_skipped += 1
        else:
            print(f"  [{traj_dir}] {os.path.basename(src)}: {status}", file=sys.stderr)
    return traj_dir, n_written, n_skipped, bytes_written


def run_quantize(
    *, path: str, trajs: str | Sequence[str], bits: str,
    num_workers: int = 4, force: bool = False,
) -> None:
    traj_basenames = _resolve_trajs(path, trajs)
    traj_dirs = [os.path.join(path, n) for n in traj_basenames]
    traj_dirs = [d for d in traj_dirs if os.path.isdir(d)]
    if not traj_dirs:
        print(f"no trajectory dirs matched under {path}")
        sys.exit(1)
    print(f"quantizing {len(traj_dirs)} trajectories to {bits} from {path}")
    t0 = time.perf_counter()
    total_w = total_s = total_b = 0
    with ThreadPoolExecutor(max_workers=max(1, num_workers)) as ex:
        futures = {ex.submit(_process_traj, d, bits, force): d for d in traj_dirs}
        for i, fut in enumerate(as_completed(futures), 1):
            d, nw, ns, bw = fut.result()
            total_w += nw; total_s += ns; total_b += bw
            elapsed = time.perf_counter() - t0
            rate = total_b / max(elapsed, 1e-6) / 1e9
            print(
                f"  [{i}/{len(traj_dirs)}] {Path(d).name:<40}  "
                f"written={nw:4d}  skip={ns:4d}  bytes={bw / 1e9:6.2f} GB  "
                f"rate={rate:5.2f} GB/s",
                flush=True,
            )
    elapsed = time.perf_counter() - t0
    print(
        f"\ndone — {total_w} files written, {total_s} skipped, "
        f"{total_b / 1e9:.2f} GB in {elapsed:.0f}s "
        f"({total_b / max(elapsed, 1e-6) / 1e9:.2f} GB/s)"
    )


# ---------- CLI --------------------------------------------------------
def main(argv: Iterable[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--mode",
        choices=("quantize",),
        default="quantize",
        help="Currently only ``quantize`` is implemented. The full GKW raw → "
             "fp32 preprocessing path stays on the torch side for now "
             "(``neugk/dataset/preprocess.py``).",
    )
    ap.add_argument("--path", default="/local00/bioinf/galletti/preprocessed_kvikio")
    ap.add_argument(
        "--trajs", nargs="+", default=["iteration_{0-299}_ifft_realpotens"],
        help="brace pattern (single string) OR explicit list of trajectory dirs",
    )
    ap.add_argument(
        "--bits", choices=tuple(_DTYPE_SUFFIX), default="bf16",
        help="quantization target (fp16 / bf16 / i8 / i4)",
    )
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing quantized shards")
    args = ap.parse_args(argv)
    if args.mode == "quantize":
        run_quantize(
            path=args.path, trajs=args.trajs, bits=args.bits,
            num_workers=args.num_workers, force=args.force,
        )


if __name__ == "__main__":
    main()
