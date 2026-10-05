"""Fixed-shape batch plans and a multi-threaded prefetching batch loader."""

from __future__ import annotations

import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence

import numpy as np

from neugk_jax.dataset.cyclone import CycloneSample
from neugk_jax.training.ddp import DistributedInfo, eval_batch_owner, process_batch_indices


@dataclass(frozen=True)
class BatchPlan:
    """Sample indices of one batch, a validity mask, its global batch number and, for
    dataset batches, the ``(file index, timestep index)`` of every row."""

    indices: np.ndarray
    mask: np.ndarray
    number: int
    fids: Optional[np.ndarray] = None
    t_idx: Optional[np.ndarray] = None


def train_plans(
    dist: DistributedInfo, n: int, per_device: int, perm: np.ndarray
) -> list[BatchPlan]:
    """This process's share of each full global batch of a shuffled epoch (the partial tail is dropped)."""
    gbs = per_device * dist.device_count
    plans = []
    for b, start in enumerate(range(0, n - gbs + 1, gbs)):
        local = process_batch_indices(dist, perm[start : start + gbs])
        plans.append(BatchPlan(local, np.ones(len(local), np.float32), b))
    return plans


def eval_plans(
    dist: Optional[DistributedInfo],
    indices: Sequence[int],
    batch_size: int,
    max_batches: Optional[int] = None,
    index: Optional[Mapping[int, tuple[int, int]]] = None,
) -> list[BatchPlan]:
    """Fixed-size batches over ``indices`` owned by this process; the last one is padded and masked.

    ``index`` (flat index -> ``(fid, t_idx)``) fills the plans' ``fids`` and ``t_idx``.
    """
    indices = np.asarray(indices, dtype=np.int64)
    n_batches = -(-len(indices) // batch_size)
    if max_batches is not None:
        n_batches = min(n_batches, int(max_batches))
    plans = []
    for b in range(n_batches):
        if dist is not None and not eval_batch_owner(dist, b):
            continue
        sel = indices[b * batch_size : (b + 1) * batch_size]
        mask = np.zeros(batch_size, np.float32)
        mask[: len(sel)] = 1.0
        sel = np.concatenate([sel, np.full(batch_size - len(sel), sel[-1])])
        fids = t_idx = None
        if index is not None:
            fids, t_idx = (np.asarray(a) for a in zip(*(index[int(i)] for i in sel)))
        plans.append(BatchPlan(sel, mask, b, fids, t_idx))
    return plans


def stack_fields(samples: Sequence[CycloneSample], fields: Sequence[str]) -> dict[str, Any]:
    """Stack the named sample fields in their array namespace; host ints become int32.

    Device arrays (``KvikIOBackend`` frames) stay on device; absent fields are left out.
    """
    out = {}
    for f in fields:
        vals = [getattr(s, f) for s in samples]
        if vals[0] is None:
            continue
        v = vals[0].__array_namespace__().stack(vals)
        if isinstance(v, np.ndarray) and v.dtype.kind in "iu":
            v = v.astype(np.int32)
        out[f] = v
    return out


class BatchLoader:
    """Prefetches batches with ``prefetch`` batches in flight and ``workers`` sample-reading threads.

    ``load(ds, indices, read)`` builds the host/device batch tree from sample indices
    (``read`` maps ``ds.__getitem__`` over the reader pool), ``place`` moves it to its
    final sharding; both run in the prefetch threads. Yields ``(plan, batch, wait_ms)``.
    """

    def __init__(self, *, workers: int = 4, prefetch: int = 2):
        self.prefetch = max(1, int(prefetch))
        self._readers = ThreadPoolExecutor(max_workers=max(1, int(workers)))
        self._batches = ThreadPoolExecutor(max_workers=self.prefetch)

    def read(self, ds, indices) -> list:
        return list(self._readers.map(lambda i: ds[int(i)], indices))

    def map(self, fn: Callable, items) -> list:
        return list(self._readers.map(fn, items))

    def iterate(
        self, ds, plans: Sequence[BatchPlan], load: Callable, place: Callable
    ) -> Iterator[tuple[BatchPlan, Any, float]]:
        def job(plan):
            batch = load(ds, plan.indices, self.read)
            batch["mask"] = plan.mask
            return place(batch)

        pending: deque = deque()
        it = iter(plans)
        for plan in it:
            pending.append((plan, self._batches.submit(job, plan)))
            if len(pending) >= self.prefetch:
                break
        while pending:
            plan, fut = pending.popleft()
            t0 = time.perf_counter()
            batch = fut.result()
            wait = (time.perf_counter() - t0) * 1e3
            nxt = next(it, None)
            if nxt is not None:
                pending.append((nxt, self._batches.submit(job, nxt)))
            yield plan, batch, wait

    def close(self) -> None:
        self._readers.shutdown(wait=False, cancel_futures=True)
        self._batches.shutdown(wait=False, cancel_futures=True)
