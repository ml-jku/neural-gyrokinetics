"""The staged recipe: compress, decode, score.

    normalise -> place base atoms -> amplitude solve -> descend -> [amplitude solve]
    -> physics fit on every atom parameter (field and potential)
    -> graft grouped carriers -> amplitude solve (the carriers against the base's residual)
    -> potential-aware amplitude solve -> routed co-fit
    -> Gauss-Newton on the flux rows -> [round to the storage precision] -> score

Every stage reads its knobs from a `RecipeConfig`; a stage switched off in the recipe is
skipped.  `compress` returns the fitted banks and an `info` dict with timings and the byte
accounting.  `score` runs the backend's readouts on the decoded object.  The banks of the
species are fitted independently; the physics stages couple them through the fields.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from neugk.gyrosplat.model.atoms import AtomBank, make_grid, subgrids
from neugk.gyrosplat.model.carriers import graft_tied_gabor, tied_budget
from neugk.gyrosplat.config import RecipeConfig
from neugk.gyrosplat.model.gn import SpectralModel, gauss_newton
from neugk.gyrosplat.model.init import field_init, random_init
from neugk.gyrosplat.model.normalize import NormStats, denormalize, normalize
from neugk.gyrosplat.model.pack import (
    compression_ratios,
    packed_bytes,
    quantize_banks,
    stored_reals,
)
from neugk.gyrosplat.model.solve import solve_amplitudes, summary_jacobian
from neugk.gyrosplat.model.synth import (
    fit_snr_db,
    render,
    render_linear,
    to_channels,
    to_separable,
)
from neugk.gyrosplat.model.train import descend, gradient_ratio, physics_fit, routed_cofit


@dataclass
class Model:
    """The fitted banks, their normalisation and the run's bookkeeping."""

    banks: List[AtomBank]
    stats: List[NormStats]
    info: Dict[str, object] = field(default_factory=dict)


def _log(cfg):
    return (lambda *a: print(*a, flush=True)) if cfg.verbose else (lambda *a: None)


def spectral_of(model: Model, backend, chunk: int = 16384) -> torch.Tensor:
    """The represented spectral object at the current amplitudes."""
    fields = []
    for bank, st in zip(model.banks, model.stats):
        grid = make_grid(bank.shape, bank.mu.device)
        vel, phys = subgrids(grid, bank.shape)
        fields.append(
            denormalize(to_channels(render(bank, vel, phys, chunk), bank.shape), st)
        )
    return backend.to_spectral(fields)


def decode(model: Model, backend) -> torch.Tensor:
    """The distribution function the physics is scored on."""
    return backend.decode(spectral_of(model, backend))


def score(
    model: Model, backend, cfg: Optional[RecipeConfig] = None, tag: str = ""
) -> Dict[str, object]:
    with torch.no_grad():
        H = spectral_of(model, backend)
        row = backend.score(backend.decode(H))
        row["H_snr_dB"] = _snr_per_species(backend.gt, H - backend.gt)
    row["tag"] = tag
    if cfg is not None:
        row.update(_accounting(model, backend, cfg))
    return row


def _snr_per_species(ref, err):
    dims = tuple(range(1, ref.dim()))
    num = (ref.abs() ** 2).sum(dim=dims, dtype=torch.float64)
    den = (err.abs() ** 2).sum(dim=dims, dtype=torch.float64).clamp_min(1e-300)
    return (10 * torch.log10(num / den)).cpu().tolist()


# ------------------------------------------------------------------------------ budgets, bytes
def budget(cfg: RecipeConfig, s: int = 0) -> Tuple[int, int]:
    """(base atoms, carrier groups) from the historical allocation rule.

    `n_equiv` is a preset size, not the achieved storage. Preserve this allocation
    to reproduce existing fits; accounting counts all unique stored envelope rows.
    """
    cc = cfg.carriers
    if cc.kind == "tied_gabor":
        return tied_budget(cfg.n_equiv, cc.fraction, cc.tied_k, cfg.reals_per_atom)
    return cfg.n_equiv, 0


def reals_per_carrier(cfg: RecipeConfig) -> float:
    """Storage of the selected geometry parameterization."""
    if cfg.carriers.kind == "tied_gabor" and cfg.carriers.share_envelopes:
        return 3 + 14 / cfg.carriers.tied_k
    return 17.0


def _accounting(model: Model, backend, cfg: RecipeConfig) -> dict:
    stored = stored_reals(model.banks, cfg.reals_per_atom, bits=cfg.storage_bits)
    raw = backend.raw_reals
    # All four normalization scalars are exactly representable in float32.
    stored["normalization"] = 4 * len(model.stats)
    stored["total"] += stored["normalization"]
    nbytes = packed_bytes(
        model.banks,
        cfg.storage_bits,
        reals_per_atom=cfg.reals_per_atom,
    )
    return {
        "stored_reals": stored,
        **compression_ratios(raw, stored),
        "packed_bytes": nbytes,
        "CR_packed": raw * 4 / max(nbytes, 1),
        "storage_accounting": "shared_envelopes_v1"
        if any(b.envelope_group_size > 1 for b in model.banks)
        else "independent_envelopes_v2",
        "CR_scope": "atom parameters plus four float32 normalization scalars per species; excludes headers",
        "packed_bytes_scope": "zlib atom parameters only; excludes normalization and headers",
    }


# ------------------------------------------------------------------------------ stages
def _solve(bank, target, vel, phys, cfg: RecipeConfig, **kw):
    return solve_amplitudes(
        bank,
        target,
        vel,
        phys,
        cfg.varpro.ridge,
        cfg.varpro.per_column,
        cfg.varpro.chunk,
        **kw,
    )


def _place(target, grid, shape, n: int, cfg: RecipeConfig) -> AtomBank:
    if cfg.init.method == "random":
        return random_init(
            n, shape, target.device, seed=cfg.seed, sigma_cells=cfg.init.sigma_cells
        )
    return field_init(
        n, target, grid, shape, seed=cfg.seed, sigma_cells=cfg.init.sigma_cells
    )


def _graft(bank: AtomBank, s: int, backend, cfg: RecipeConfig, grid) -> AtomBank:
    cc = cfg.carriers
    _, n_carr = budget(cfg, s)
    tail = (
        list(cc.tail_bins)
        if cc.tail_bins
        else list(backend.band)[len(backend.band) // 2 :]
    )
    return graft_tied_gabor(bank, grid, n_carr, cc, tail, seed=cfg.seed + s)


def potential_term(
    banks, stats, vels, physs, backend, s: int, weight: float, chunk: int = 16384
):
    """The backend's potential summary as a linear function of species s's amplitudes (the other
    species held fixed): (M, target, weight) for `solve_amplitudes`.  The summary is affine in
    the amplitudes through the normalisation offsets, so the offset is moved to the target."""

    def summary(amps):
        fields = []
        for j, (b, st, v, p) in enumerate(zip(banks, stats, vels, physs)):
            a = amps if j == s else b.amps.detach()
            fields.append(
                denormalize(to_channels(render_linear(a, b, v, p, chunk), b.shape), st)
            )
        return backend.potential_summary(fields)

    M = summary_jacobian(summary, banks[s])
    with torch.no_grad():
        offset = summary(torch.zeros_like(banks[s].amps))
    return M, (backend.summary_target - offset).double(), weight


# ------------------------------------------------------------------------------ the pipeline
def compress(backend, cfg: RecipeConfig, device: Optional[str] = None) -> Model:
    """Fit the recipe to the backend's target fields.  Returns the Model (banks, stats, info)."""
    log = _log(cfg)
    t0 = time.time()
    stage_t = {}
    device = device or backend.device
    torch.manual_seed(cfg.seed)
    banks, stats, targets, vels, physs, grids = [], [], [], [], [], []

    # -- per species: normalise, place, descend, solve ----------------------------------------
    for s, raw in enumerate(backend.target_fields()):
        shape = tuple(raw.shape[1:])
        norm, st = normalize(raw.to(device), cfg.normalize)
        target = to_separable(norm)
        del norm
        grid = make_grid(shape, device)
        vel, phys = subgrids(grid, shape)
        n_base, _ = budget(cfg, s)
        bank = _place(target, grid, shape, n_base, cfg)
        if cfg.fit.warm_start:
            _solve(bank, target, vel, phys, cfg)
        descend(bank, target, vel, phys, cfg.fit)
        if cfg.varpro.after_fit:
            _solve(bank, target, vel, phys, cfg)
        log(
            f"[{s}] base bank N {bank.N}: fit SNR {fit_snr_db(bank, target, vel, phys):.2f} dB ({time.time() - t0:.0f}s)"
        )
        for lst, item in (
            (banks, bank),
            (stats, st),
            (targets, target),
            (vels, vel),
            (physs, phys),
            (grids, grid),
        ):
            lst.append(item)
    stage_t["fit"] = time.time() - t0

    # -- the physics fit (every parameter, all species, the backend's terms) --------------------
    if cfg.physics.epochs > 0:
        t1 = time.time()
        physics_fit(
            banks,
            stats,
            vels,
            physs,
            backend,
            cfg.physics,
            chunk=cfg.fit.chunk,
            log=log,
        )
        stage_t["physics_fit"] = time.time() - t1
        log(f"physics fit: {cfg.physics.epochs} epochs ({time.time() - t0:.0f}s)")

    # -- carriers -------------------------------------------------------------------------------
    cc = cfg.carriers
    live = cc.kind != "none" and cc.fraction > 0
    gate = "none"
    if live and cc.flux_gate > 0 and hasattr(backend, "flux_magnitude"):
        q = backend.flux_magnitude()
        gate = f"{'inert' if q < cc.flux_gate else 'live'} (band flux {q:.3g})"
    inert = gate.startswith("inert")
    if live:
        t1 = time.time()
        for s in range(len(banks)):
            bank = banks[s] = _graft(banks[s], s, backend, cfg, grids[s])
            if inert:
                continue
            cols = (
                torch.nonzero(bank.carrier_mask).reshape(-1)
                if cc.residual_solve
                else None
            )
            _solve(bank, targets[s], vels[s], physs[s], cfg, columns=cols)
            log(
                f"[{s}] + {bank.n_carriers} {cc.kind} carriers -> N {bank.N}: fit SNR "
                f"{fit_snr_db(bank, targets[s], vels[s], physs[s]):.2f} dB ({time.time() - t0:.0f}s)"
            )
        if not inert and cfg.varpro.potential_weight > 0:
            for s, bank in enumerate(banks):
                term = potential_term(
                    banks,
                    stats,
                    vels,
                    physs,
                    backend,
                    s,
                    cfg.varpro.potential_weight,
                    cfg.fit.chunk,
                )
                _solve(bank, targets[s], vels[s], physs[s], cfg, linear_terms=[term])
            log(f"potential-aware amplitude solve ({time.time() - t0:.0f}s)")
        if not inert and cc.cofit == "routed":
            ratio = gradient_ratio(banks, stats, vels, physs, backend, cfg.fit.chunk)
            epochs = cc.routed_steps_hard if ratio > cc.hard_ratio else cc.routed_steps
            routed_cofit(
                banks, stats, vels, physs, backend, cc, epochs, cfg.fit.chunk, log
            )
            log(
                f"routed co-fit: {epochs} epochs, gradient ratio {ratio:.1f} ({time.time() - t0:.0f}s)"
            )
            stage_t["gradient_ratio"] = ratio
        stage_t["carriers"] = time.time() - t1
    fit_snr = [
        fit_snr_db(b, t, v, p) for b, t, v, p in zip(banks, targets, vels, physs)
    ]
    del targets
    model = Model(banks, stats)
    rows = [score(model, backend, cfg, "fit")]
    log(_fmt(rows[-1]))

    # -- the physics stage ---------------------------------------------------------------------
    if (
        cfg.gn.steps > 0
        and not inert
        and any(b.n_carriers > 0 or not cfg.gn.carriers_only for b in banks)
    ):
        sm = SpectralModel(banks, vels, physs, stats, backend)
        t1 = time.time()
        hist = gauss_newton(sm, cfg.gn, log=log)
        stage_t["gauss_newton"] = time.time() - t1
        rows.append(score(model, backend, cfg, "+ gauss-newton"))
        log(_fmt(rows[-1]))
        model.info["gn"] = hist

    # -- storage precision ---------------------------------------------------------------------
    if cfg.storage_bits < 32:
        quantize_banks(model.banks, cfg.storage_bits)
        rows.append(score(model, backend, cfg, f"+ {cfg.storage_bits}-bit parameters"))
        log(_fmt(rows[-1]))
    model.info.update(
        {
            "rows": rows,
            "fit_snr_dB": fit_snr,
            "seconds": time.time() - t0,
            "stage_seconds": stage_t,
            "n_atoms": [b.N for b in banks],
            "n_carriers": [b.n_carriers for b in banks],
            "carrier_gate": gate,
        }
    )
    return model


def _fmt(row: dict) -> str:
    keys = ("psnr_paper_all", "phi_psnr")
    parts = [
        f"{k} {row[k]:.2f}" for k in keys if k in row and isinstance(row[k], float)
    ]
    if "flux_relL1_band" in row:
        fl = row["flux_relL1_band"]
        fl = fl.get("energy", fl) if isinstance(fl, dict) else fl
        parts.append("flux " + "/".join(f"{x:.4f}" for x in fl))
    if "kappa" in row:
        parts.append("kappa " + "/".join(f"{x:.2f}" for x in row["kappa"]))
    if "CR" in row:
        parts.append(f"CR {row['CR']:.0f}x")
    return f"  [{row.get('tag', '')}] " + "  ".join(parts)
