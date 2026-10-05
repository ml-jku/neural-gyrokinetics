"""Distributed setup for jax.distributed (SLURM or torchrun) and data-parallel placement helpers.

One global ``Mesh`` with a single data axis spans every device of every process. Training
batches are loaded per process (``process_batch_indices``) and assembled into global arrays
(``shard_batch``); model and optimizer state are replicated (``replicate``). Evaluation runs
per process on a local mesh (``local_view``, ``shard_local``) with batches assigned
round-robin (``eval_batch_owner``). ``training.batch_size`` is per device, so the global
batch is ``batch_size * device_count``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P


@dataclass
class DistributedInfo:
    process_id: int
    num_processes: int
    local_device_count: int
    local_rank: int
    mesh: Mesh

    @property
    def is_rank0(self) -> bool:
        return self.process_id == 0

    @property
    def device_count(self) -> int:
        return self.mesh.size

    @property
    def local_mesh(self) -> Mesh:
        return Mesh(np.asarray(jax.local_devices()), self.mesh.axis_names)


def _torchrun_env() -> dict | None:
    if int(os.environ.get("WORLD_SIZE", "1")) <= 1 or "RANK" not in os.environ:
        return None
    return dict(
        coordinator_address=f"{os.environ.get('MASTER_ADDR', 'localhost')}:"
        f"{os.environ.get('MASTER_PORT', '29500')}",
        num_processes=int(os.environ["WORLD_SIZE"]),
        process_id=int(os.environ["RANK"]),
        # torchrun starts one process per gpu
        local_device_ids=[int(os.environ.get("LOCAL_RANK", "0"))],
    )


def _slurm_multiprocess() -> bool:
    return "SLURM_JOB_ID" in os.environ and int(os.environ.get("SLURM_NTASKS", "1")) > 1


def init_distributed(*, axis_name: str = "dp") -> DistributedInfo:
    """Initialise jax.distributed under torchrun or multi-task SLURM, else single process."""
    if not jax.distributed.is_initialized():
        tr = _torchrun_env()
        if tr is not None:
            jax.distributed.initialize(**tr)
        elif _slurm_multiprocess():
            # jax's slurm cluster detection resolves the coordinator from the nodelist
            jax.distributed.initialize()
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
    return DistributedInfo(
        process_id=jax.process_index(),
        num_processes=jax.process_count(),
        local_device_count=jax.local_device_count(),
        local_rank=local_rank,
        mesh=Mesh(np.asarray(jax.devices()), (axis_name,)),
    )


def barrier(name: str) -> None:
    if jax.process_count() > 1:
        from jax.experimental import multihost_utils

        multihost_utils.sync_global_devices(name)


def global_batch_size(dist: DistributedInfo, per_device: int) -> int:
    return per_device * dist.device_count


def process_batch_indices(dist: DistributedInfo, window: np.ndarray) -> np.ndarray:
    # contiguous per-process slice of one global batch window
    per_proc = len(window) // dist.num_processes
    return window[dist.process_id * per_proc : (dist.process_id + 1) * per_proc]


def eval_batch_owner(dist: DistributedInfo, batch_idx: int) -> bool:
    return batch_idx % dist.num_processes == dist.process_id


def _is_array(x) -> bool:
    return isinstance(x, jax.Array | np.ndarray | np.generic)


def _assemble(x, sharding: NamedSharding, n_rows_global: int):
    # per-device slices of the process-local rows, placed and stitched into one global array
    shape = (n_rows_global, *x.shape[1:])
    index_map = sharding.addressable_devices_indices_map(shape)
    offset = min(idx[0].start or 0 for idx in index_map.values())
    shards = []
    for dev, idx in index_map.items():
        lo, hi = (idx[0].start or 0) - offset, (idx[0].stop or n_rows_global) - offset
        shards.append(jax.device_put(x[lo:hi], dev))
    return jax.make_array_from_single_device_arrays(shape, sharding, shards)


def _put_rows(tree, mesh: Mesh, n_procs: int):
    if mesh.size == 1:
        return _put(tree, mesh.devices.flat[0])
    sharding = NamedSharding(mesh, P(mesh.axis_names[0]))
    return jax.tree_util.tree_map(
        lambda x: _assemble(x, sharding, x.shape[0] * n_procs) if _is_array(x) else x, tree
    )


def shard_batch(dist: DistributedInfo, tree):
    return _put_rows(tree, dist.mesh, dist.num_processes)


def shard_local(dist: DistributedInfo, tree):
    return _put_rows(tree, dist.local_mesh, 1)


def _replicated_sharding(mesh: Mesh):
    if mesh.size == 1:
        return jax.sharding.SingleDeviceSharding(mesh.devices.flat[0])
    return NamedSharding(mesh, P())


def _put(tree, sharding):
    return jax.tree_util.tree_map(
        lambda x: jax.device_put(x, sharding) if _is_array(x) else x, tree
    )


def replicate(dist: DistributedInfo, tree):
    if dist.num_processes > 1:
        tree = jax.device_get(tree)
    return _put(tree, _replicated_sharding(dist.mesh))


def replicate_local(dist: DistributedInfo, tree):
    return _put(tree, _replicated_sharding(dist.local_mesh))


def local_view(dist: DistributedInfo, tree):
    """Process-local replicated view of replicated arrays.

    Globally replicated arrays reuse their local shards; arrays on fewer devices are
    copied onto the local mesh.
    """
    mesh = dist.local_mesh
    rep = _replicated_sharding(mesh)

    def view(x):
        if not isinstance(x, jax.Array) or x.sharding.device_set == set(mesh.devices.flat):
            return x
        if len(x.addressable_shards) != mesh.size:
            return jax.device_put(x, rep)
        shards = [s.data for s in x.addressable_shards]
        if mesh.size == 1:
            return shards[0]
        return jax.make_array_from_single_device_arrays(x.shape, rep, shards)

    return jax.tree_util.tree_map(view, tree)
