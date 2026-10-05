"""Evaluator base: fixed-shape batch iteration, the dataset normalization table on device, flux
integrals and masked metric accumulation with cross-process sync over a fixed key set.

Each evaluator is built once per run; its batch plan, normalization and geometry tables
are fixed, and its per-batch forward is a module-level jitted function.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from neugk_jax.evaluate.integrals import flux_integral, precompute_geometry, require_geometry
from neugk_jax.losses import per_sample_mse, per_sample_rel_l2
from neugk_jax.training.data import BatchLoader, eval_plans
from neugk_jax.training.ddp import (
    DistributedInfo,
    init_distributed,
    local_view,
    replicate_local,
    shard_local,
)
from neugk_jax.utils import recombine_zf, to_dict


def validation_cfg(cfg) -> dict:
    return to_dict(cfg.get("validation")) if hasattr(cfg, "get") else {}


class GeometryCache:
    """:func:`precompute_geometry` tensors of a dataset's trajectories, computed once each."""

    def __init__(self, ds):
        self.ds = ds
        self._geoms: dict[int, dict] = {}

    def _one(self, fid: int) -> dict:
        if fid not in self._geoms:
            g = self.ds.metadata[fid].get("geometry")
            require_geometry(g)
            self._geoms[fid] = precompute_geometry(g)
        return self._geoms[fid]

    def stack(self, fids: Sequence[int]) -> dict[str, np.ndarray]:
        rows = [self._one(int(f)) for f in fids]
        return {k: np.stack([r[k] for r in rows]) for k in rows[0]}

    def table(self) -> dict[str, np.ndarray]:
        return self.stack(range(len(self.ds.files)))


def recon_metrics(pred, x, pred_d, tgt_d) -> dict:
    return {
        "df_mse": per_sample_mse(pred, x),
        "df_rel_l2": per_sample_rel_l2(recombine_zf(pred_d, axis=1), recombine_zf(tgt_d, axis=1)),
    }


def take_rows(table: dict, fids) -> dict:
    return jax.tree_util.tree_map(lambda a: a[fids], table)


def integrate(geom, fids, df, phi=None):
    """Batched flux integral of a denormalized df (separate-zf layouts are recombined)."""
    g = take_rows(geom, fids)
    df = recombine_zf(df, axis=1)
    if phi is None:
        return jax.vmap(flux_integral)(g, df)
    return jax.vmap(flux_integral)(g, df, phi)


def accumulate(acc: dict, values: Mapping[str, jnp.ndarray], weight) -> dict:
    """Add the ``weight``-masked per-sample ``values`` and the weight total into ``acc``."""
    out = dict(acc)
    for k, v in values.items():
        out[k] = acc[k] + jnp.sum(v * weight)
    out["_n"] = acc["_n"] + jnp.sum(weight)
    return out


class BaseEvaluator:
    """Owns the fixed evaluation batch plan, the device tables and metric reduction.

    ``batch_size`` is per local device. Subclasses define ``metric_keys`` and
    ``__call__(model, *, epoch) -> (metrics, plots)``; ``validation.eval_integrals``
    defaults to their ``integrals_default``.
    """

    integrals_default: bool = False

    def __init__(
        self,
        cfg: Any,
        *,
        val_ds: Any,
        dist: Optional[DistributedInfo] = None,
        batch_size: int = 1,
        loader: Optional[BatchLoader] = None,
        indices: Optional[Sequence[int]] = None,
        max_batches: Optional[int] = None,
    ):
        self.cfg = cfg
        self.vcfg = validation_cfg(cfg)
        self.ds = val_ds
        self.dist = dist or init_distributed()
        self.batch_size = int(batch_size) * self.dist.local_device_count
        self.loader = loader or BatchLoader()
        idx = range(len(val_ds)) if indices is None else indices
        self.plans = eval_plans(
            self.dist, idx, self.batch_size, max_batches, val_ds.flat_index_to_file_and_tstep
        )
        self.norm = replicate_local(self.dist, val_ds.norm)
        self.eval_integrals = bool(self.vcfg.get("eval_integrals", self.integrals_default))
        self._geometry = None

    @property
    def is_rank0(self) -> bool:
        return self.dist.is_rank0

    @property
    def geometry(self) -> dict:
        if self._geometry is None:
            self._geometry = replicate_local(self.dist, GeometryCache(self.ds).table())
        return self._geometry

    def place(self, batch):
        return shard_local(self.dist, batch)

    def local_model(self, model):
        return local_view(self.dist, model)

    def zeros(self, keys: Sequence[str], shape: tuple = ()) -> dict:
        return replicate_local(self.dist, {k: jnp.zeros(shape, jnp.float32) for k in keys})

    def reduce(self, acc: dict) -> dict[str, float]:
        return self.sum_processes({k: float(v) for k, v in jax.device_get(acc).items()})

    def finalize(self, sums: Mapping[str, float], keys: Sequence[str]) -> dict[str, float]:
        return {k: sums[k] / max(sums["_n"], 1.0) for k in keys}

    def spectra(self, store: dict, pred_d, tgt_d, plan) -> None:
        from neugk_jax.evaluate.metrics import accumulate_spectral_diagnostics

        accumulate_spectral_diagnostics(store, pred_d, tgt_d, plan.fids, self.ds, plan.mask > 0)

    @staticmethod
    def plot_time(batch) -> np.ndarray:
        return np.asarray(jax.device_get(batch["timestep"][:1])).reshape(-1)

    def spectra_available(self, requested: bool) -> bool:
        """``requested`` unless a validation trajectory's metadata lacks the ``ds`` spacing."""
        if requested and any(self.ds.get_ds(f) is None for f in range(len(self.ds.files))):
            if self.is_rank0:
                print(
                    "[evaluate] eval_spectra requested but metadata has no 'ds'; "
                    "skipping spectral metrics"
                )
            return False
        return requested

    def spectral_metrics(self, store: dict) -> dict[str, float]:
        """Spectral metrics of the per-trajectory sums of every process (collective)."""
        from neugk_jax.evaluate import metrics as m

        if self.dist.num_processes > 1:
            n_ky = int(self.ds.resolution[-1])
            packed = self.sum_process_arrays(m.pack_spectral_store(store, len(self.ds.files), n_ky))
            store = m.unpack_spectral_store(packed, n_ky)
        return m.merged_spectral_metrics(store)

    def sum_process_arrays(self, arr: np.ndarray) -> np.ndarray:
        if self.dist.num_processes <= 1:
            return arr
        from jax.experimental import multihost_utils

        return np.asarray(multihost_utils.process_allgather(arr)).sum(axis=0)

    def sum_processes(self, host: dict[str, float]) -> dict[str, float]:
        """Sum host scalars over processes; every process passes the same key set."""
        if self.dist.num_processes <= 1:
            return host
        keys = sorted(host)
        tot = self.sum_process_arrays(np.asarray([host[k] for k in keys], dtype=np.float64))
        return {k: float(v) for k, v in zip(keys, tot)}

    def __call__(self, model: Any, *, epoch: int) -> tuple[dict[str, float], dict[str, Any]]:
        raise NotImplementedError
