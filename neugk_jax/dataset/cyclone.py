"""Unified ``CycloneDataset`` for both AE training and latent diffusion.

This is a deliberate consolidation of the upstream torch ``CycloneDataset``
(``neugk/dataset/cyclone.py``) and ``CycloneAEDataset``
(``neugk/dataset/cyclone_diff.py``):

* **Same** file resolution / metadata loading / conditioning extraction.
* **Same** per-timestep flat indexing and normalisation conventions.
* **Drops** the SimSiam / VAE / VQVAE / LinearCyclone / Coordinate
  variants — not in the user's requested scope.
* ``mode="ae"`` returns raw distribution-function tensors; ``mode="diff"``
  returns precomputed latents (after running ``precompute_latents``).

Returns host-side numpy arrays in a frozen ``CycloneSample`` dataclass.
JAX consumes them via ``jnp.asarray`` when the dataloader stacks a batch.
"""

from __future__ import annotations

import os
import pickle
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Any, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from neugk_jax.dataset.backend import DataBackend, KvikIOBackend, NumpyBackend, resolve_trajectories
from neugk_jax.utils import RunningMeanStd, separate_zf as separate_zf_fn


@dataclass(frozen=True)
class CycloneSample:
    """A single dataset item. ``df`` is the raw distribution function (mode='ae')
    or a precomputed latent tensor (mode='diff')."""

    df: np.ndarray | None
    phi: np.ndarray | None
    flux: np.ndarray
    avg_flux: np.ndarray
    timestep: np.ndarray
    file_index: np.ndarray
    timestep_index: np.ndarray
    conditioning: np.ndarray | None
    # raw scalar conditions (also packed into conditioning if requested)
    itg: np.ndarray
    dg: np.ndarray
    s_hat: np.ndarray
    q: np.ndarray


def collate(batch: Sequence[CycloneSample]) -> CycloneSample:
    """Stack a list of samples into a batched ``CycloneSample``.

    Detects the array flavour from the first sample's ``df``: ``jax.Array``
    inputs (kvikio + ``return_jax=True``) are stacked via ``jnp.stack`` to
    keep the batch on GPU; everything else goes through ``np.stack``.
    """
    use_jax = isinstance(batch[0].df, jax.Array)
    stacker = jnp.stack if use_jax else np.stack
    def stack(key: str):
        vals = [getattr(s, key) for s in batch]
        if vals[0] is None:
            return None
        return stacker(vals)
    return CycloneSample(
        df=stack("df"),
        phi=stack("phi"),
        flux=stack("flux"),
        avg_flux=stack("avg_flux"),
        timestep=stack("timestep"),
        file_index=stack("file_index"),
        timestep_index=stack("timestep_index"),
        conditioning=stack("conditioning"),
        itg=stack("itg"),
        dg=stack("dg"),
        s_hat=stack("s_hat"),
        q=stack("q"),
    )


class CycloneDataset:
    """Map-style dataset over preprocessed gyrokinetics trajectories.

    Parameters
    ----------
    path, trajectories
        Directory + trajectory spec (string with ``{1-5,7}`` ranges or list).
    split
        ``"train"`` or ``"val"``. ``val`` may use ``partial_holdouts`` to
        carve out the tail of each trajectory.
    fields_to_load
        Subset of ``("df", "phi")`` to read from disk. ``flux`` is always
        derivable from metadata.
    conditions
        Scalar fields to pack into ``CycloneSample.conditioning`` (e.g.
        ``("itg", "dg", "s_hat", "q")``).
    mode
        ``"ae"`` returns raw df reads; ``"diff"`` returns precomputed latents
        (after calling :func:`precompute_latents`).
    normalization, normalization_scope, normalization_stats
        Same semantics as upstream — see ``neugk/dataset/cyclone.py``.
    separate_zf, decouple_mu
        Optional channel-axis preprocessing matching the AE config.
    bundle_seq_length
        Time-bundling stride. Default ``1`` (one timestep per sample).
    offset
        Number of leading timesteps to skip per trajectory.
    """

    def __init__(
        self,
        *,
        path: str,
        split: str = "train",
        trajectories: Optional[Any] = None,
        partial_holdouts: Optional[dict] = None,
        fields_to_load: Sequence[str] = ("df",),
        conditions: Sequence[str] = ("itg", "dg", "s_hat", "q"),
        mode: str = "ae",
        normalization: Optional[dict] = None,
        normalization_scope: str = "dataset",
        normalization_stats: Optional[dict | str] = None,
        spatial_ifft: bool = True,
        real_potens: bool = True,
        cond_filters: Optional[dict] = None,
        bundle_seq_length: int = 1,
        offset: int = 0,
        tail_offset: int = 0,
        subsample: int = 1,
        separate_zf: bool = False,
        decouple_mu: bool = False,
        backend: Optional[DataBackend] = None,
        num_workers: int = 0,
        rank: int = 0,
    ):
        assert split in ("train", "val")
        assert mode in ("ae", "diff")
        self.path = path
        self.split = split
        self.fields_to_load = list(fields_to_load)
        # sort alphabetically to keep conditioning slot order consistent across runs
        self.conditions = sorted(conditions)
        self.mode = mode
        self.normalization = normalization
        self.normalization_scope = normalization_scope
        self.normalization_stats = normalization_stats
        self.spatial_ifft = spatial_ifft
        self.real_potens = real_potens
        self.cond_filters = cond_filters or {}
        self.bundle_seq_length = bundle_seq_length
        self.offset = offset
        self.tail_offset = tail_offset
        self.subsample = subsample
        self.separate_zf = separate_zf
        self.decouple_mu = decouple_mu
        # default to KvikIOBackend for GPU-direct reads; falls back to NumpyBackend transparently
        self.backend = backend or KvikIOBackend(rank=rank)
        self.num_workers = num_workers
        self.rank = rank
        self.partial_holdouts = partial_holdouts or {}

        # latent storage (mode="diff"); filled by precompute_latents()
        self.precomputed_latents: dict[tuple[int, int], dict] | None = None
        self.latent_stats: RunningMeanStd | None = None

        if trajectories is not None:
            if split == "val" and self.partial_holdouts:
                raw = [os.path.join(path, k) for k in self.partial_holdouts]
            else:
                raw = resolve_trajectories(path, trajectories)
        else:
            raw = [
                os.path.join(path, n)
                for n in os.listdir(path)
                if self.backend.is_valid(os.path.join(path, n))
            ]
        self.files = sorted({
            self.backend.format_path(f, spatial_ifft, None, real_potens) for f in raw
            if self.backend.is_valid(self.backend.format_path(f, spatial_ifft, None, real_potens))
        })
        if not self.files:
            raise RuntimeError(f"no trajectories found under {path}")

        # metadata loads are I/O bound and tiny; cap workers at 16
        with ThreadPoolExecutor(max_workers=max(1, min(16, num_workers or 8))) as ex:
            metas = list(ex.map(
                lambda f: self.backend.read_metadata(f, self.fields_to_load),
                self.files,
            ))
        self.metadata: dict[int, dict] = {}
        kept_files = []
        for fp, meta in zip(self.files, metas):
            if self._passes_cond_filter(meta):
                fid = len(kept_files)
                kept_files.append(fp)
                # OOD trajectories use "fluxes" key instead of "flux" — normalise here
                if "flux" not in meta and "fluxes" in meta:
                    meta["flux"] = meta["fluxes"]
                self.metadata[fid] = meta
        self.files = kept_files

        self.flat_index_to_file_and_tstep: dict[int, tuple[int, int]] = {}
        self.file_and_tstep_to_flat_index: dict[tuple[int, int], int] = {}
        self.file_num_timesteps: list[int] = []
        flat = 0
        for fid, meta in self.metadata.items():
            timesteps = meta["timesteps"][offset:]
            if tail_offset > 0:
                timesteps = timesteps[:-tail_offset]
            n = len(timesteps[::subsample]) - bundle_seq_length * 2 + 1
            self.file_num_timesteps.append(len(timesteps[::subsample]))
            for t_idx in range(max(0, n)):
                self.flat_index_to_file_and_tstep[flat] = (fid, t_idx * subsample)
                self.file_and_tstep_to_flat_index[(fid, t_idx * subsample)] = flat
                flat += 1
        self.length = flat

        # resolution: assume same across files
        self.resolution = tuple(self.metadata[0]["resolution"])
        self.df_shape = (2, *self.resolution)
        self.phi_resolution = (self.resolution[3], self.resolution[2], self.resolution[4])

        # normalisation stats: prefer normalization_stats if provided, fall back to per-trajectory metadata moments
        self.stats = self._build_stats()


    def _passes_cond_filter(self, meta: dict) -> bool:
        for cond_name, cond_range in self.cond_filters.items():
            where = None
            if "_" in cond_name:
                where, cond_name = cond_name.split("_", 1)
            if cond_name not in meta:
                return False
            cond = meta[cond_name]
            if not isinstance(cond_range[0], (list, tuple)):
                cond_range = [cond_range]
            if cond_name == "flux":
                bound = self.offset if self.offset > 0 else 80
                cond = float(np.mean(cond[:bound] if where == "first" else cond[-bound:]))
            if not any(lo <= cond <= hi for lo, hi in cond_range):
                return False
        return True


    def _build_stats(self) -> dict[str, dict]:
        """Construct ``stats[field][fid|'full']`` from metadata moments.

        ``normalization_stats`` can be:

        * a ``dict`` already in the ``stats[field][key]`` form — used as-is;
        * a ``str`` / ``Path`` pointing at the upstream stats pickle
          (``RunningMeanStd`` per field) — we load and apply the
          per-field ``agg_axes`` from ``self.normalization`` to collapse
          the full per-element stats into the aggregation requested by
          the model config.
        """
        if isinstance(self.normalization_stats, (str, os.PathLike)):
            return self._load_stats_pkl(self.normalization_stats)
        if self.normalization_stats is not None:
            return self.normalization_stats
        if self.normalization is None:
            return {}
        out: dict[str, dict] = {k: defaultdict(dict) for k in self.fields_to_load}
        agg: dict[str, RunningMeanStd] = {k: RunningMeanStd() for k in self.fields_to_load}
        for fid, meta in self.metadata.items():
            for k in self.fields_to_load:
                if f"{k}_mean" in meta:
                    mean = meta[f"{k}_mean"]
                    var = meta[f"{k}_std"] ** 2
                    mn = meta.get(f"{k}_min", mean)
                    mx = meta.get(f"{k}_max", mean)
                    out[k][fid] = {"mean": mean, "std": np.sqrt(var), "min": mn, "max": mx}
                    n = len(meta["timesteps"])
                    agg[k].update(mean, var, mn, mx, count=n)
        for k in self.fields_to_load:
            if agg[k].count:
                out[k]["full"] = {
                    "mean": np.asarray(agg[k].mean, dtype=np.float32),
                    "std": np.asarray(np.sqrt(agg[k].var), dtype=np.float32),
                    "min": np.asarray(agg[k].min, dtype=np.float32),
                    "max": np.asarray(agg[k].max, dtype=np.float32),
                }
        return out

    def _load_stats_pkl(self, path) -> dict[str, dict]:
        """Load upstream ``RunningMeanStd``-pkl and apply per-field ``agg_axes``.

        Returns the stats dict in our internal format
        (``stats[field]['full']`` with mean / std / min / max numpy arrays).
        """
        with open(path, "rb") as f:
            raw = pickle.load(f)
        out: dict[str, dict] = {}
        for k, rms in raw.items():
            mean = np.asarray(rms.mean, dtype=np.float64)
            var = np.asarray(rms.var, dtype=np.float64)
            mn = np.asarray(rms.min, dtype=np.float64)
            mx = np.asarray(rms.max, dtype=np.float64)
            agg = None
            if self.normalization and k in self.normalization:
                agg = self.normalization[k].get("agg_axes")
            if agg:
                agg = tuple(int(a) for a in agg)
                # keepdims so the result broadcasts directly against the data tensor
                new_mean = mean.mean(axis=agg, keepdims=True)
                new_var = (
                    var.mean(axis=agg, keepdims=True)
                    + mean.var(axis=agg, keepdims=True)
                )
                new_min = mn.min(axis=agg, keepdims=True)
                new_max = mx.max(axis=agg, keepdims=True)
                mean, var, mn, mx = new_mean, new_var, new_min, new_max
            out.setdefault(k, {})["full"] = {
                "mean": mean.astype(np.float32),
                "std": np.sqrt(var).astype(np.float32),
                "min": mn.astype(np.float32),
                "max": mx.astype(np.float32),
            }
        return out

    def _get_scale_shift(self, fid: int, field: str, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.normalization_scope == "sample" or self.normalization is None:
            return np.float32(1.0), np.float32(0.0)
        key = "full" if self.normalization_scope == "dataset" else fid
        stats = self.stats.get(field, {}).get(key, {})
        if not stats:
            return np.float32(1.0), np.float32(0.0)
        nt = self.normalization[field]["type"]
        if nt == "zscore":
            mean = np.asarray(stats["mean"], dtype=np.float32)
            std = np.asarray(stats["std"], dtype=np.float32)
            return std, mean
        if nt == "minmax":
            lo = np.asarray(stats["min"], dtype=np.float32)
            hi = np.asarray(stats["max"], dtype=np.float32)
            beta1 = self.normalization[field].get("minmax_beta1", 8)
            beta2 = self.normalization[field].get("minmax_beta2", 4)
            scale = (hi - lo) / beta1
            shift = lo + scale * beta2
            return scale, shift
        raise ValueError(nt)

    def normalize(self, fid: int, *, df=None, phi=None, flux=None) -> np.ndarray:
        x, field = self._unpack_field(df=df, phi=phi, flux=flux)
        scale, shift = self._get_scale_shift(fid, field, x)
        return (x - shift) / scale

    def denormalize(self, fid: int, *, df=None, phi=None, flux=None) -> np.ndarray:
        x, field = self._unpack_field(df=df, phi=phi, flux=flux)
        scale, shift = self._get_scale_shift(fid, field, x)
        return x * scale + shift

    @staticmethod
    def _unpack_field(*, df=None, phi=None, flux=None):
        if df is not None: return df, "df"
        if phi is not None: return phi, "phi"
        if flux is not None: return flux, "flux"
        raise ValueError("provide exactly one of df, phi, flux")


    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> CycloneSample:
        fid, t_idx = self.flat_index_to_file_and_tstep[index]
        if self.mode == "diff" and self.precomputed_latents is not None:
            return self._get_latent_sample(fid, t_idx)
        return self._get_ae_sample(fid, t_idx)

    def _get_ae_sample(self, fid: int, t_idx: int) -> CycloneSample:
        meta = self.metadata[fid]
        original_t = t_idx + self.offset
        f_path = self.files[fid]
        df = phi = None
        with self.backend.open(f_path) as handle:
            t_str = str(original_t).zfill(5)
            if "df" in self.fields_to_load:
                df = self.backend.read_df(handle, t_str, self.df_shape, [0, 1])
                if self.separate_zf:
                    df = separate_zf_fn(df, axis=0)
            if "phi" in self.fields_to_load:
                phi = self.backend.read_phi(handle, t_str, self.phi_resolution)

        flux = np.asarray(meta["flux"][original_t], dtype=np.float32)
        timestep = np.asarray(meta["timesteps"][original_t], dtype=np.float32)
        if df is not None and self.normalization is not None:
            df = self.normalize(fid, df=df)
            df = df.astype(jnp.float32) if isinstance(df, jax.Array) else df.astype(np.float32)
        if phi is not None and self.normalization is not None:
            phi = self.normalize(fid, phi=phi)
            phi = phi.astype(jnp.float32) if isinstance(phi, jax.Array) else phi.astype(np.float32)

        return self._build_sample(fid, t_idx, df, phi, flux, timestep, meta)

    def _get_latent_sample(self, fid: int, t_idx: int) -> CycloneSample:
        cached = self.precomputed_latents[(fid, t_idx)]
        meta = self.metadata[fid]
        flux = np.asarray(cached.get("flux", meta["flux"][t_idx + self.offset]), dtype=np.float32)
        timestep = np.asarray(cached.get("timestep", meta["timesteps"][t_idx + self.offset]), dtype=np.float32)
        return self._build_sample(
            fid, t_idx, cached["x"].astype(np.float32),
            cached.get("phi"), flux, timestep, meta,
        )

    def _build_sample(self, fid, t_idx, df, phi, flux, timestep, meta) -> CycloneSample:
        itg = np.asarray(np.squeeze(meta["ion_temp_grad"]), dtype=np.float32)
        dg = np.asarray(np.squeeze(meta["density_grad"]), dtype=np.float32)
        s_hat = np.asarray(np.squeeze(meta["s_hat"]), dtype=np.float32)
        q = np.asarray(np.squeeze(meta["q"]), dtype=np.float32)
        cond_vec = None
        if self.conditions:
            packed = []
            local = {"itg": itg, "dg": dg, "s_hat": s_hat, "q": q}
            for k in self.conditions:
                v = local.get(k)
                if v is None:
                    v = np.asarray(np.squeeze(meta[k]), dtype=np.float32)
                packed.append(np.atleast_1d(v))
            cond_vec = np.concatenate(packed).astype(np.float32)

        avg = float(np.mean(meta["flux"][-80:]))
        return CycloneSample(
            df=df, phi=phi,
            flux=flux,
            avg_flux=np.float32(avg),
            timestep=timestep,
            file_index=np.int64(fid),
            timestep_index=np.int64(t_idx),
            conditioning=cond_vec,
            itg=itg, dg=dg, s_hat=s_hat, q=q,
        )

    def get_avg_flux(self, fid: int) -> float:
        return float(np.mean(self.metadata[fid]["flux"][-80:]))

    def get_batch_geometry(self, file_indices: np.ndarray) -> dict[str, np.ndarray]:
        """Stack per-file geometry into a batched dict."""
        geoms = [self.metadata[int(f)]["geometry"] for f in file_indices]
        keys = geoms[0].keys()
        return {k: np.stack([np.ascontiguousarray(g[k]) for g in geoms]) for k in keys}


    @staticmethod
    def collate(batch: Sequence[CycloneSample]) -> CycloneSample:
        return collate(batch)
