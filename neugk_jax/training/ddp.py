"""Distributed setup helpers for jax.distributed (SLURM + torchrun-style).

Mirrors the upstream ``main.py`` dispatch: detect whether we're under a
multi-process launcher, set up ``jax.distributed`` accordingly, build a
single-axis ``Mesh`` over the host's local devices, and expose a
``shard_data`` helper that puts a leading-axis-sharded array onto that
mesh. Multi-host coordinates via ``SLURM_NODELIST``'s first host.
"""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


@dataclass
class DistributedInfo:
    process_id: int
    num_processes: int
    local_device_count: int
    mesh: Mesh

    @property
    def is_rank0(self) -> bool:
        return self.process_id == 0


def _slurm_coordinator() -> tuple[str, int, int] | None:
    """Resolve (host, num_processes, process_id) from SLURM env vars."""
    if "SLURM_JOB_ID" not in os.environ:
        return None
    n_procs = int(os.environ.get("SLURM_NTASKS", "1"))
    pid = int(os.environ.get("SLURM_PROCID", "0"))
    if n_procs == 1:
        return None
    nodelist = os.environ.get("SLURM_NODELIST", "")
    # take the first hostname from the nodelist (handles "node[01-04]" syntax loosely)
    first = nodelist.split(",")[0].split("[")[0].strip()
    return first or "localhost", n_procs, pid


def _torchrun_coordinator() -> tuple[str, int, int] | None:
    """Resolve from RANK / WORLD_SIZE / MASTER_ADDR if set."""
    if "RANK" not in os.environ:
        return None
    return (
        os.environ.get("MASTER_ADDR", "localhost"),
        int(os.environ["WORLD_SIZE"]),
        int(os.environ["RANK"]),
    )


def init_distributed(*, port: int = 29500, axis_name: str = "dp") -> DistributedInfo:
    """Initialise jax.distributed if launched under SLURM or torchrun.

    Falls back to single-process / local-devices mode if neither launcher
    is detected. Always returns a populated ``DistributedInfo`` with a
    single-axis ``Mesh`` covering this host's local devices.
    """
    info = _torchrun_coordinator() or _slurm_coordinator()
    if info is not None:
        host, n, pid = info
        coord = f"{host}:{port}"
        if not jax.distributed.is_initialized():
            jax.distributed.initialize(
                coordinator_address=coord,
                num_processes=n,
                process_id=pid,
            )
        local_devices = jax.local_devices()
    else:
        n, pid = 1, 0
        local_devices = jax.local_devices()
    mesh = Mesh(jax.devices(), (axis_name,))
    return DistributedInfo(
        process_id=pid,
        num_processes=n,
        local_device_count=len(local_devices),
        mesh=mesh,
    )


def data_sharding(mesh: Mesh, axis_name: str = "dp") -> NamedSharding:
    """``NamedSharding`` that places the leading axis on ``axis_name`` and replicates the rest."""
    return NamedSharding(mesh, P(axis_name))


def replicated(mesh: Mesh) -> NamedSharding:
    """``NamedSharding`` that fully replicates over the mesh."""
    return NamedSharding(mesh, P())


def all_reduce_mean(x):
    """Cross-device mean (use inside ``shard_map``)."""
    return jax.lax.pmean(x, axis_name="dp")
