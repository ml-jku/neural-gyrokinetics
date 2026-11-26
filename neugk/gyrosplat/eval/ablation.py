"""The rows of the paper's ablation ladder, one command each.

    python -m neugk.gyrosplat ablate --dataset gkw --control noky --config configs/gyrosplat/recipe/gkw_302x.yaml data.root=... \\
        data.trajectory=iteration_280 data.timestep=25

Most controls are the recipe with a few overrides (`CONTROLS`).  Four are not expressible
that way and are built here from the recipe's own stages: the field-only amplitude solve at
the initial Gabor geometry (`field_solve`), the two all-Gabor banks that put the whole
budget into shared-envelope groups before stage 1 (`all_gabor`, `all_gabor_full`), and the
half-precision round trip of the deployed model (`s7_pack`, the recipe's `storage_bits=16`
row).  `deployed` returns the two rows of the recipe itself: after the joint fit (stage 0)
and after the Gauss-Newton polish (stage 1, the paper's method).
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

import torch
from omegaconf import OmegaConf

import neugk.gyrosplat.recipe as recipe
from neugk.gyrosplat.model.atoms import AtomBank, make_grid, subgrids
from neugk.gyrosplat.config import RecipeConfig
from neugk.gyrosplat.model.gn import SpectralModel, gauss_newton
from neugk.gyrosplat.model.normalize import normalize
from neugk.gyrosplat.model.synth import to_separable
from neugk.gyrosplat.model.train import descend, physics_fit

STAGE_1 = ["carriers.kind=none", "physics.epochs=0", "gn.steps=0"]
FLAT_MSE = ["fit.loss=mse", "fit.schedule=flat", "fit.polish_steps=0"]
NO_COFIT = ["carriers.cofit=none", "varpro.potential_weight=0"]
ALL_GABOR = ["carriers.fraction=1", "fit.chunk=4096", "varpro.chunk=4096"] + NO_COFIT

CONTROLS: Dict[str, List[str]] = {
    # the cumulative stages of the ladder
    "floor": STAGE_1
    + ["normalize=global", "init.method=random", "fit.warm_start=false"]
    + FLAT_MSE,
    "zonal": STAGE_1 + ["init.method=random", "fit.warm_start=false"] + FLAT_MSE,
    "importance_init": STAGE_1 + FLAT_MSE,
    "geometry_warmup": STAGE_1,
    "potential_fit": ["carriers.kind=none", "gn.steps=0"],
    "gabor_init": NO_COFIT + ["gn.steps=0"],
    "deployed": [],
    # the alternative configurations
    "field_solve": NO_COFIT + ["gn.steps=0"],  # then the field-only solve, below
    "gn_without_joint_fit": NO_COFIT,
    "noky": ["carriers.kind=none", "n_equiv=4608", "reals_per_atom=16", "gn.steps=0"],
    "random_init": STAGE_1 + FLAT_MSE + ["init.method=random"],
    "random_full": ["init.method=random"],
    "no_warmstart": STAGE_1 + FLAT_MSE + ["fit.warm_start=false"],
    "flux_during_potential_fit": [
        "carriers.kind=none",
        "gn.steps=0",
        "physics.flux_weight=1",
    ],
    "all_gabor": ALL_GABOR + ["gn.steps=0"],
    "all_gabor_full": ALL_GABOR + ["gn.carriers_only=false"],
    "frac40": ["carriers.fraction=0.4"] + NO_COFIT + ["gn.steps=0"],
    "frac60": ["carriers.fraction=0.6"] + NO_COFIT + ["gn.steps=0"],
    "s7_pack": ["storage_bits=16"],
    "fp32match": ["n_equiv=2168"],
    "noky_full": [
        "carriers.kind=none",
        "n_equiv=4608",
        "reals_per_atom=16",
        "gn.carriers_only=false",
    ],
    "flux_refinement_all_splats": ["carriers.cofit=none", "gn.carriers_only=false"],
}


def control_config(cfg: RecipeConfig, control: str) -> RecipeConfig:
    """The recipe with the control's overrides applied."""
    if control not in CONTROLS:
        raise ValueError(f"unknown control {control}; choose from {sorted(CONTROLS)}")
    merged = OmegaConf.merge(
        OmegaConf.structured(cfg), OmegaConf.from_dotlist(CONTROLS[control])
    )
    return OmegaConf.to_object(merged)


def _grids(bank: AtomBank, device: str):
    grid = make_grid(bank.shape, device)
    vel, phys = subgrids(grid, bank.shape)
    return grid, vel, phys


def _all_gabor_bank(backend, cfg: RecipeConfig, device: str):
    """The whole budget as shared-envelope Gabor groups, fitted through stage 1."""
    raw = backend.target_fields()[0].to(device)
    norm, stats = normalize(raw, cfg.normalize)
    target = to_separable(norm)
    shape = tuple(raw.shape[1:])
    grid = make_grid(shape, device)
    vel, phys = subgrids(grid, shape)
    empty = AtomBank(
        torch.empty(0, 5, device=device),
        torch.empty(0, 6, device=device),
        torch.empty(0, 3, device=device),
        torch.empty(0, 2, device=device),
        shape=shape,
        n_base=0,
    )
    bank = recipe._graft(empty, 0, backend, cfg, grid)
    recipe._solve(bank, target, vel, phys, cfg)
    descend(bank, target, vel, phys, cfg.fit)
    physics_fit(
        [bank], [stats], [vel], [phys], backend, cfg.physics, chunk=cfg.fit.chunk
    )
    return recipe.Model([bank], [stats]), target, vel, phys


def run_control(
    control: str, backend, cfg: RecipeConfig, device: Optional[str] = None
) -> Dict[str, object]:
    """One ladder row on one snapshot: {"control", "rows", "config", "seconds"}."""
    device = device or backend.device
    t0 = time.time()
    ccfg = control_config(cfg, control)
    torch.manual_seed(ccfg.seed)
    if control == "deployed":
        model = recipe.compress(backend, ccfg, device=device)
        rows = model.info["rows"][:2]
    elif control == "s7_pack":
        model = recipe.compress(backend, ccfg, device=device)
        rows = [model.info["rows"][-1]]
    elif control == "field_solve":
        model = recipe.compress(backend, ccfg, device=device)
        for bank, stats, raw in zip(model.banks, model.stats, backend.target_fields()):
            _, vel, phys = _grids(bank, device)
            target, check = normalize(raw.to(device), ccfg.normalize)
            assert vars(check) == vars(stats)
            recipe._solve(bank, to_separable(target), vel, phys, ccfg)
        rows = [recipe.score(model, backend, ccfg, control)]
    elif control in ("all_gabor", "all_gabor_full"):
        model, target, vel, phys = _all_gabor_bank(backend, ccfg, device)
        if control == "all_gabor_full":
            bank = model.banks[0]
            term = recipe.potential_term(
                model.banks,
                model.stats,
                [vel],
                [phys],
                backend,
                0,
                ccfg.varpro.potential_weight,
                ccfg.fit.chunk,
            )
            recipe._solve(bank, target, vel, phys, ccfg, linear_terms=[term])
            gauss_newton(
                SpectralModel(
                    model.banks,
                    [vel],
                    [phys],
                    model.stats,
                    backend,
                    chunk=ccfg.fit.chunk,
                ),
                ccfg.gn,
                log=recipe._log(ccfg),
            )
        rows = [recipe.score(model, backend, ccfg, control)]
    else:
        model = recipe.compress(backend, ccfg, device=device)
        rows = [model.info["rows"][-1]]
    for bank in model.banks:
        if bank.n_carriers and ccfg.carriers.share_envelopes:
            assert bank.envelope_group_size == ccfg.carriers.tied_k
    return {
        "control": control,
        "overrides": list(CONTROLS[control]),
        "rows": rows,
        "n_atoms": [b.N for b in model.banks],
        "n_carriers": [b.n_carriers for b in model.banks],
        "config": OmegaConf.to_container(OmegaConf.structured(ccfg)),
        "seconds": time.time() - t0,
    }
