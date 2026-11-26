"""Loader for the GKW (cyclone base case) snapshots in the preprocessed layout.

A trajectory directory `<root>/<trajectory>_ifft_realpotens/` holds `metadata.pkl` (or
`metadata_light.pkl`) with the resolution (at the top level, or under `layout` in the nested
metadata layout) and the geometry tensors the integrals need, and
`data/timestep_NNNNN.bin`, one float32 array (2, v_par, mu, s, x, y) per time step.  Only
the data reading lives here; the physics operators are the host `neugk` package's.
"""
from __future__ import annotations

import os
import pickle

import numpy as np
import torch

from neugk.gyrosplat.data import Snapshot


def load_gkw_snapshot(
    root: str,
    trajectory: str,
    timestep: int,
    device: str = "cuda",
    raw_root: str = None,
) -> Snapshot:
    d = os.path.join(root, f"{trajectory}_ifft_realpotens")
    if not os.path.isdir(d):
        raise FileNotFoundError(d)
    meta_path = os.path.join(d, "metadata.pkl")
    if not os.path.exists(meta_path):
        meta_path = os.path.join(d, "metadata_light.pkl")
    with open(meta_path, "rb") as fh:
        meta = pickle.load(fh)
    # the resolution sits at the top level, or under "layout" in the nested metadata layout
    resolution = meta["resolution"] if "resolution" in meta else meta["layout"]["resolution"]
    shape = (2, *resolution)
    raw = np.fromfile(
        os.path.join(d, "data", f"timestep_{timestep:05d}.bin"), dtype=np.float32
    )
    field = torch.from_numpy(raw.reshape(shape)).float()

    geom = {
        k: torch.as_tensor(np.array(v)).squeeze(0)
        for k, v in meta["geometry"].items()
        if k != "ds"  # the nested layout also stores the parallel spacing here
    }
    for k in (
        "adiabatic",
        "de",
        "beta",
        "nlapar",
        "nlbpar",
    ):  # defaults the integrals expect
        geom.setdefault(k, torch.tensor(1.0, dtype=torch.float64))
    ds = _parallel_spacing(meta, trajectory, raw_root)
    return Snapshot(
        [field],
        name=f"{trajectory}_t{timestep}",
        meta={"geom": geom, "ds": ds, "trajectory": trajectory, "timestep": timestep},
    )


def _parallel_spacing(meta: dict, trajectory: str, raw_root: str = None) -> float:
    """The spacing of the parallel grid: from the raw run's `sgrid` when available, else from
    the (uniform) parallel integration weights in the metadata."""
    if raw_root:
        try:
            sgrid = np.loadtxt(
                os.path.join(raw_root, trajectory.split("_ifft")[0], "sgrid")
            )
            return float(sgrid[1] - sgrid[0])
        except OSError:
            pass
    w = np.asarray(meta["geometry"]["ints"]).ravel()
    if w.size > 1 and np.allclose(w, w[0], rtol=1e-9, atol=0):
        return float(w[0])
    raise ValueError("cannot infer the parallel grid spacing; pass raw_root")
