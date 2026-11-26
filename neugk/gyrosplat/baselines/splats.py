"""Gaussian splats with a field objective only, at the same stored bytes as the full recipe:
the textbook fit.  One global z-score, centres scattered uniformly, a flat-rate Adam on the
mean squared error, no amplitude solve, no carriers, no physics stages, and 16 reals per
atom because a plain Gaussian needs no carrier wavenumber.
"""
from __future__ import annotations

import copy
import time

from neugk.gyrosplat.config import RecipeConfig


def field_only_splats(
    backend, cfg: RecipeConfig, budget_reals: int, device: str, log=print
):
    from neugk.gyrosplat.recipe import compress

    c = copy.deepcopy(cfg)
    c.carriers.kind = "none"
    c.physics.epochs = 0
    c.gn.steps = 0
    c.storage_bits = 32
    c.normalize = "global"
    c.init.method = "random"
    c.fit.loss = "mse"
    c.fit.schedule = "flat"
    c.fit.steps = 1000
    c.fit.warm_start = False
    c.fit.polish_steps = 0
    c.varpro.after_fit = False
    c.reals_per_atom = cfg.reals_per_atom - 1
    c.n_equiv = max(1, budget_reals // c.reals_per_atom)
    t0 = time.time()
    model = compress(backend, c, device=device)
    rows = model.info["rows"]
    for r in rows:
        r["tag"] = "splats (field objective only)"
    return {
        "method": "splats",
        "budget_reals": budget_reals,
        "achieved_reals": [float(rows[-1]["stored_reals"]["atoms"])],
        "rows": rows,
        "seconds": time.time() - t0,
        "config": {"name": "splats"},
        "n_atoms": model.info["n_atoms"],
    }
