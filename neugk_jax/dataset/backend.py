"""I/O backend for the cyclone preprocessed dataset (.bin per-timestep).

The upstream torch project uses two backends: ``H5Backend`` (for the
canonical HDF5 files) and ``KvikIOBackend`` (for a directory of float32
``.bin`` files plus a ``metadata.pkl``). The user told us KvikIOBackend
is the only one we need; it's also the format the AE precompute pipeline
already writes to.

We default to ``NumpyBackend`` (``np.fromfile``) because it's portable
across nodes without cupy and just-as-fast on most setups. ``KvikIOBackend``
wraps cupy + kvikio for GPU-direct reads where available — it exists for
parity with the torch path and is gated behind an import guard.

All readers return host-side ``numpy`` arrays — JAX consumes them via
``jnp.asarray(...)`` when batching, which is zero-copy on a single host.
"""

from __future__ import annotations

import contextlib
import os
import pickle
import re
from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence

import numpy as np


def read_bin(file: str, shape: tuple, dtype=np.float32) -> np.ndarray:
    """Read a flat binary into a contiguous ``np.ndarray`` of given shape."""
    arr = np.fromfile(file, dtype=dtype)
    expected = int(np.prod(shape))
    if arr.size != expected:
        raise IOError(f"{file}: expected {expected} elements, got {arr.size}")
    return arr.reshape(shape)


_DTYPE_SUFFIX = {
    "fp16": ".fp16.bin",
    "bf16": ".bf16.bin",
    "i8":   ".i8.bin",
    "i4":   ".i4.bin",
}


def _quantized_sibling(fp32_path: str, dtype: str) -> str:
    suf = _DTYPE_SUFFIX[dtype]
    return fp32_path[:-4] + suf if fp32_path.endswith(".bin") else fp32_path + suf


# kept for back-compat; new code uses ``_quantized_sibling``
def _bf16_sibling(fp32_path: str) -> str:
    return _quantized_sibling(fp32_path, "bf16")


def _quantize_roundtrip(arr_f32: np.ndarray, dtype: str) -> np.ndarray:
    """Round-trip an fp32 array through the requested quantization and back to fp32.

    Used by the loader's on-the-fly fallback: when ``prefer_dtype`` is set but the
    pre-computed quantized shard isn't on disk, we still want the model to see
    inputs with the same precision the shard would have produced.
    """
    from neugk_jax.dataset.preprocess import quantize_array, dequantize_array
    payload, scale = quantize_array(arr_f32, dtype)
    return dequantize_array(payload, scale, dtype, arr_f32.size).astype(np.float32, copy=False)


def _resolve_dtyped_path(fp32_path: str, prefer_dtype: str | None) -> tuple[str, str]:
    """Pick the path to actually read from.

    ``prefer_dtype`` is one of ``fp32`` / ``fp16`` / ``bf16`` / ``i8`` /
    ``i4`` (or ``None`` / ``False`` for fp32). Falls back to fp32 silently
    when the preferred sibling is missing — that's the graceful
    fp32-fallback the dataset config relies on.

    Returns ``(path, mode)`` where ``mode`` is one of ``fp32``, ``fp16``,
    ``bf16``, ``i8``, ``i4``.
    """
    if prefer_dtype and prefer_dtype != "fp32":
        cand = _quantized_sibling(fp32_path, prefer_dtype)
        if os.path.exists(cand):
            return cand, prefer_dtype
    return fp32_path, "fp32"


class DataBackend(ABC):
    @abstractmethod
    def is_valid(self, path: str) -> bool: ...

    @abstractmethod
    def format_path(
        self, path: str, spatial_ifft: bool,
        split_into_bands: Optional[int] = None,
        real_potens: bool = True,
    ) -> str: ...

    @abstractmethod
    def read_metadata(
        self, path: str, input_fields: Sequence[str] = ("df",),
        lightweight: bool = False,
    ) -> dict: ...

    @abstractmethod
    @contextlib.contextmanager
    def open(self, path: str): ...

    @abstractmethod
    def read_df(
        self, f: Any, timestamp: str, shape: Sequence[int],
        active_keys: Optional[Sequence[int]] = None,
    ) -> np.ndarray: ...

    @abstractmethod
    def read_phi(self, f: Any, timestamp: str, shape: Sequence[int]) -> np.ndarray: ...


class NumpyBackend(DataBackend):
    """Plain numpy reader for the per-timestep .bin layout.

    Directory layout::

        traj_dir/
        ├── metadata.pkl
        ├── metadata_light.pkl  (optional; written by the upstream code)
        └── data/
            ├── timestep_00000.bin
            ├── timestep_00001.bin
            ├── timestep_00000.bf16.bin  (optional, ``scripts/quantize_bf16.py``)
            ├── poten_00000.bin
            └── ...

    ``prefer_bf16=True`` reads the ``.bf16.bin`` sibling when present and
    falls back to fp32 silently when it isn't (graceful regression).
    """

    def __init__(
        self, *, prefer_dtype: str | None = None,
        quantize_fallback: bool = True, prefer_bf16: bool = False,
    ):
        # back-compat: prefer_bf16=True → prefer_dtype="bf16"
        if prefer_bf16 and not prefer_dtype:
            prefer_dtype = "bf16"
        self.prefer_dtype = prefer_dtype or "fp32"
        # if true and the preferred sibling is missing, read fp32 and round-trip
        # through the preferred dtype on-the-fly so the model sees the same
        # precision regardless of whether the bf16/fp16/... shards exist
        self.quantize_fallback = quantize_fallback
        # keep the legacy attribute name for any code still reading it
        self.prefer_bf16 = (self.prefer_dtype == "bf16")

    def _strip_h5(self, path: str) -> str:
        return path.removesuffix("/").removesuffix(".h5")

    def is_valid(self, path: str) -> bool:
        return os.path.isdir(self._strip_h5(path))

    def format_path(
        self, path: str, spatial_ifft: bool,
        split_into_bands: Optional[int] = None,
        real_potens: bool = True,
    ) -> str:
        path = self._strip_h5(path)
        if spatial_ifft:
            if split_into_bands:
                tag = f"_ifft_separate_zf_{split_into_bands}bands_realpotens"
            else:
                tag = "_ifft_realpotens" if real_potens else "_ifft"
            if tag not in path:
                path = path + tag
        return path

    def read_metadata(
        self, path: str, input_fields: Sequence[str] = ("df",),
        lightweight: bool = False,
    ) -> dict:
        path = self._strip_h5(path)
        light_path = os.path.join(path, "metadata_light.pkl")
        full_path = os.path.join(path, "metadata.pkl")
        if lightweight and os.path.exists(light_path):
            with open(light_path, "rb") as f:
                meta = pickle.load(f)
        else:
            with open(full_path, "rb") as f:
                meta = pickle.load(f)
            if lightweight:
                drop = {"df_min", "df_max", "df_var", "df_mean", "df_std",
                        "phi_min", "phi_max", "phi_var"}
                meta = {k: v for k, v in meta.items() if k not in drop}
        # fill in missing geometry scalars with safe defaults
        if "geometry" in meta:
            g = meta["geometry"]
            for k in ("adiabatic", "de", "beta", "nlapar", "nlbpar"):
                if k not in g:
                    g[k] = np.array(1.0, dtype=np.float64)
            # gyaradax needs ffun (flux-surface function); stub with ones for CYCLONE s-α at ε→0
            if "ffun" not in g and "ints" in g:
                g["ffun"] = np.ones_like(np.asarray(g["ints"]), dtype=np.float64)
        return meta

    @contextlib.contextmanager
    def open(self, path: str):
        # yield the directory path as the "file handle" for per-timestep reads
        yield self._strip_h5(path)

    def _read_dtyped(self, fp32_path: str, shape: Sequence[int]) -> np.ndarray:
        """Read fp32 or a quantized sibling; always returns fp32 numpy.

        Quantized formats (fp16/bf16/i8/i4) are dequantized inline on the
        host so downstream code (normalize, separate_zf, FFT integrals)
        stays fp32-only.

        When ``prefer_dtype`` is set but the sibling is missing and
        ``quantize_fallback=True``, the fp32 file is read and immediately
        round-tripped through the preferred dtype — the model sees identical
        precision whether the on-disk shard exists or not.
        """
        path, mode = _resolve_dtyped_path(fp32_path, self.prefer_dtype)
        if mode == "fp32" and self.prefer_dtype not in (None, "fp32") and self.quantize_fallback:
            # fp32 sibling exists but bf16/fp16/... was requested → quantize on-the-fly
            arr = read_bin(path, tuple(shape))
            return _quantize_roundtrip(arr.ravel(), self.prefer_dtype).reshape(shape)
        if mode == "fp32":
            return read_bin(path, tuple(shape))
        from neugk_jax.dataset.preprocess import read_quantized
        expected = int(np.prod(shape))
        arr = read_quantized(path, mode, expected)
        if arr.size != expected:
            raise IOError(f"{path}: expected {expected} {mode} elements, got {arr.size}")
        return arr.reshape(shape).astype(np.float32, copy=False)

    def read_df(
        self, f_dir: str, timestamp: str, shape: Sequence[int],
        active_keys: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        fp = os.path.join(f_dir, "data", f"timestep_{timestamp}.bin")
        k = self._read_dtyped(fp, tuple(shape))
        if active_keys is None or (len(active_keys) == 2 and tuple(active_keys) == (0, 1)):
            return k
        return k[list(active_keys)]

    def read_phi(self, f_dir: str, timestamp: str, shape: Sequence[int]) -> np.ndarray:
        fp = os.path.join(f_dir, "data", f"poten_{timestamp}.bin")
        return self._read_dtyped(fp, tuple(shape))


class KvikIOBackend(NumpyBackend):
    """GPU-direct reads via cupy + kvikio (NVIDIA GDS).

    Default backend — the upstream torch loader uses kvikio for the same
    reason: disk → GPU bypassing the host buffer. Two modes:

    * ``return_jax=True`` (default): returns a ``jax.Array`` on GPU via
      ``jax.dlpack.from_dlpack(cp_arr.toDlpack())``. Zero-copy from cupy.
      Requires the dataloader to run single-process (``num_workers=0``).
    * ``return_jax=False``: copies back to host as ``numpy.ndarray``.
      Slower per-sample but compatible with multi-worker dataloaders. Use
      this when num_workers > 0 or when running on a node without GPUs.

    Falls back to ``NumpyBackend`` (``np.fromfile``) if cupy / kvikio aren't
    installed or ``use_kvikio=False``.
    """

    def __init__(
        self,
        rank: int = 0,
        *,
        use_kvikio: bool = True,
        return_jax: bool = True,
        prefer_dtype: str | None = None,
        prefer_bf16: bool = False,
    ):
        super().__init__(prefer_dtype=prefer_dtype, prefer_bf16=prefer_bf16)
        self.rank = rank
        self.use_kvikio = use_kvikio
        self.return_jax = return_jax
        if use_kvikio:
            try:
                import cupy  # noqa: F401
                import kvikio  # noqa: F401
            except ImportError:
                self.use_kvikio = False
                self.return_jax = False

    def _cp_read(self, file: str, shape: tuple, *, dtype="fp32"):
        """kvikio CuFile read into a cupy buffer. The on-disk layout varies
        with ``dtype``; we read the raw bytes into a buffer of the right
        size and let the caller reinterpret/dequantize."""
        import cupy as cp
        import kvikio
        n_elems = int(np.prod(shape))
        if dtype == "fp32":
            buf_dtype, buf_size = cp.float32, n_elems
        elif dtype in ("fp16", "bf16"):
            buf_dtype, buf_size = cp.uint16, n_elems  # 2 bytes/elem; reinterpret post-DLPack
        elif dtype == "i8":
            # 4-byte fp32 scale header + n_elems int8 values
            buf_dtype, buf_size = cp.int8, n_elems + 4
        elif dtype == "i4":
            # 4-byte fp32 scale header + ceil(n_elems / 2) packed uint8 nibbles
            buf_dtype, buf_size = cp.uint8, ((n_elems + 1) // 2) + 4
        else:
            raise ValueError(f"unknown dtype={dtype!r}")
        with cp.cuda.Device(self.rank):
            gpu = cp.empty(buf_size, dtype=buf_dtype)
            with kvikio.CuFile(file, "r") as fh:
                fh.read(gpu)
        return gpu

    def _dequantize_gpu(self, gpu, mode: str, shape: tuple):
        """Turn a raw cupy buffer into an fp32 jax.Array (or numpy if
        ``return_jax=False``). Handles the per-dtype layout described in
        :mod:`neugk_jax.dataset.preprocess`.
        """
        import cupy as cp
        import jax.dlpack as jdlp
        import jax.lax as lax
        import jax.numpy as jnp
        n_elems = int(np.prod(shape))
        if mode == "fp32":
            if self.return_jax:
                return jdlp.from_dlpack(gpu.reshape(shape))
            return cp.asnumpy(gpu.reshape(shape))
        if mode == "fp16":
            f16 = gpu.view(cp.float16).reshape(shape)
            if self.return_jax:
                return jdlp.from_dlpack(f16).astype(jnp.float32)
            return cp.asnumpy(f16).astype(np.float32)
        if mode == "bf16":
            u16 = gpu.reshape(shape)
            if self.return_jax:
                a = jdlp.from_dlpack(u16)
                return lax.bitcast_convert_type(a, jnp.bfloat16).astype(jnp.float32)
            # host fallback
            from ml_dtypes import bfloat16
            return cp.asnumpy(u16).view(bfloat16).astype(np.float32)
        if mode == "i8":
            # split header / payload on device
            scale = float(cp.asnumpy(gpu[:4]).view(np.float32)[0])
            payload = gpu[4:].view(cp.int8).reshape(shape)
            if self.return_jax:
                a = jdlp.from_dlpack(payload).astype(jnp.float32)
                return a * jnp.float32(scale)
            return cp.asnumpy(payload).astype(np.float32) * np.float32(scale)
        if mode == "i4":
            # dequantize on host — i4 unpacking is fiddly and not a hot path
            scale = float(cp.asnumpy(gpu[:4]).view(np.float32)[0])
            packed = cp.asnumpy(gpu[4:].view(cp.uint8))
            from neugk_jax.dataset.preprocess import dequantize_array
            arr = dequantize_array(packed, np.float32(scale), "i4", n_elems).reshape(shape)
            if self.return_jax:
                return jnp.asarray(arr)
            return arr
        raise ValueError(f"unknown mode={mode!r}")

    def _read(self, file: str, shape: tuple):
        path, mode = _resolve_dtyped_path(file, self.prefer_dtype)
        # on-the-fly quantize fallback: fp32 on disk but bf16/fp16/... requested
        if mode == "fp32" and self.prefer_dtype not in (None, "fp32") and self.quantize_fallback:
            if not self.use_kvikio:
                arr = read_bin(path, shape)
            else:
                gpu = self._cp_read(path, tuple(shape), dtype="fp32")
                # host round-trip is simpler than a GPU bf16 emulation on cupy 14
                import cupy as cp
                arr = cp.asnumpy(gpu).reshape(shape)
            quantized = _quantize_roundtrip(arr.ravel(), self.prefer_dtype).reshape(shape)
            if self.return_jax and self.use_kvikio:
                import jax.numpy as jnp
                return jnp.asarray(quantized)
            return quantized
        if not self.use_kvikio:
            if mode == "fp32":
                return read_bin(path, shape)
            from neugk_jax.dataset.preprocess import read_quantized
            n_elems = int(np.prod(shape))
            return read_quantized(path, mode, n_elems).reshape(shape).astype(np.float32, copy=False)
        gpu = self._cp_read(path, tuple(shape), dtype=mode)
        return self._dequantize_gpu(gpu, mode, tuple(shape))

    def read_df(
        self, f_dir: str, timestamp: str, shape: Sequence[int],
        active_keys: Optional[Sequence[int]] = None,
    ):
        fp = os.path.join(f_dir, "data", f"timestep_{timestamp}.bin")
        k = self._read(fp, tuple(shape))
        if active_keys is None or (len(active_keys) == 2 and tuple(active_keys) == (0, 1)):
            return k
        return k[list(active_keys)]

    def read_phi(self, f_dir: str, timestamp: str, shape: Sequence[int]):
        fp = os.path.join(f_dir, "data", f"poten_{timestamp}.bin")
        return self._read(fp, tuple(shape))


def resolve_trajectories(path: str, trajectories) -> list[str]:
    """Expand a trajectories spec (string with ``{1-5}`` ranges, or list)."""
    if isinstance(trajectories, str):
        match = re.match(r"^(.*?)\{([^}]+)\}(.*?)$", trajectories)
        if not match:
            return [os.path.join(path, trajectories)]
        prefix, ranges_str, suffix = match.groups()
        nums = []
        for part in ranges_str.split(","):
            if "-" in part:
                lo, hi = map(int, part.split("-"))
                nums.extend(range(lo, hi + 1))
            else:
                nums.append(int(part))
        return [os.path.join(path, f"{prefix}{n}{suffix}") for n in nums]
    return [os.path.join(path, t) for t in trajectories]
