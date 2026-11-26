"""CPU tests on the synthetic snapshot: the render adjoint, the amplitude solves, the
Gauss-Newton and the recipe end to end."""
import math

import torch

from neugk.gyrosplat.model.atoms import make_grid, subgrids
from neugk.gyrosplat.backends.synthetic import SyntheticBackend
from neugk.gyrosplat.config import load_config
from neugk.gyrosplat.datasets.synthetic import make_synthetic_snapshot
from neugk.gyrosplat.model.gn import SpectralModel, gauss_newton, jacobian_autograd
from neugk.gyrosplat.model.init import field_init
from neugk.gyrosplat.model.normalize import normalize
from neugk.gyrosplat.recipe import compress
from neugk.gyrosplat.model.solve import solve_amplitudes
from neugk.gyrosplat.model.synth import render, render_linear, residual_power, to_separable

SHAPE = (6, 4, 5, 12, 12)


def _snapshot(seed=0):
    return make_synthetic_snapshot(SHAPE, seed=seed)


def _bank(n=10, seed=1):
    snap = _snapshot(seed)
    target = to_separable(snap.fields[0])
    grid = make_grid(SHAPE)
    vel, phys = subgrids(grid, SHAPE)
    return field_init(n, target, grid, SHAPE, seed=seed), target, vel, phys


def test_render_linear_matches_render_and_adjoint():
    bank, target, vel, phys = _bank()
    torch.manual_seed(0)
    bank.ky.data[5:] = 2 * math.pi * torch.randint(1, 4, (bank.N - 5,)).float()
    ref = render(bank, vel, phys, chunk=17)
    amps = bank.amps.data.clone().requires_grad_(True)
    out = render_linear(amps, bank, vel, phys, chunk=17)
    assert torch.allclose(out, ref, atol=1e-6)
    g = torch.randn_like(out)
    (out * g).sum().backward()
    # adjoint check: <render(a), g> == <a, render^T g>
    lhs = float((out.detach() * g).sum())
    rhs = float((amps.grad * amps.detach()).sum())
    assert abs(lhs - rhs) < 1e-4 * max(abs(lhs), 1.0)


def test_amplitude_solve_recovers_generating_amplitudes():
    bank, _, vel, phys = _bank(n=8)
    truth = bank.amps.data.clone()
    target = render(bank, vel, phys)
    bank.amps.data.zero_()
    solve_amplitudes(bank, target, vel, phys, ridge=1e-9, per_column=True, chunk=50)
    assert residual_power(bank, target, vel, phys) < 1e-8 * float((target**2).mean())
    assert torch.allclose(bank.amps.data, truth, atol=1e-3, rtol=1e-3)


def test_subset_solve_moves_only_the_requested_atoms():
    bank, target, vel, phys = _bank(n=12)
    bank.n_base = 8
    bank.ky.data[8:] = 2 * math.pi * torch.tensor([1.0, 2.0, 3.0, 2.0])
    solve_amplitudes(bank, target, vel, phys, ridge=1e-6, per_column=False, chunk=64)
    base = bank.amps.data[:8].clone()
    bank.amps.data[8:] = 0.0
    before = residual_power(bank, target, vel, phys)
    cols = torch.nonzero(bank.carrier_mask).reshape(-1)
    solve_amplitudes(
        bank, target, vel, phys, ridge=1e-6, per_column=False, chunk=64, columns=cols
    )
    assert torch.equal(bank.amps.data[:8], base)
    assert residual_power(bank, target, vel, phys) <= before * (1 + 1e-6)


def test_potential_aware_solve_pulls_the_summary_to_the_truth():
    from neugk.gyrosplat.recipe import potential_term

    snap = _snapshot(5)
    backend = SyntheticBackend(snap, band=[1, 2, 3])
    norm, st = normalize(snap.fields[0], "zonal")
    target = to_separable(norm)
    grid = make_grid(SHAPE)
    vel, phys = subgrids(grid, SHAPE)
    bank = field_init(10, target, grid, SHAPE, seed=5)

    def summary_error():
        from neugk.gyrosplat.model.normalize import denormalize
        from neugk.gyrosplat.model.synth import to_channels

        fields = [denormalize(to_channels(render(bank, vel, phys), SHAPE), st)]
        return float(
            (backend.potential_summary(fields) - backend.summary_target).norm()
        )

    solve_amplitudes(bank, target, vel, phys, ridge=1e-3, per_column=False, chunk=64)
    plain = bank.amps.data.clone()
    e_plain = summary_error()
    term = potential_term([bank], [st], [vel], [phys], backend, 0, 0.0, chunk=64)
    solve_amplitudes(
        bank,
        target,
        vel,
        phys,
        ridge=1e-3,
        per_column=False,
        chunk=64,
        linear_terms=[term],
    )
    assert torch.allclose(bank.amps.data, plain, atol=1e-5)
    M, t, _ = potential_term([bank], [st], [vel], [phys], backend, 0, 100.0, chunk=64)
    solve_amplitudes(
        bank,
        target,
        vel,
        phys,
        ridge=1e-3,
        per_column=False,
        chunk=64,
        linear_terms=[(M, t, 100.0)],
    )
    e_heavy = summary_error()
    # with a heavy weight the summary error approaches the best the bank's span allows
    best = float(
        (M @ torch.linalg.lstsq(M, t.unsqueeze(1)).solution.squeeze(1) - t).norm()
    )
    assert e_heavy < e_plain
    assert e_heavy < 1.1 * best + 1e-3


def test_gauss_newton_reduces_the_flux_rows():
    """The autograd Jacobian and a few damped steps on the carrier amplitudes."""
    from neugk.gyrosplat.config import GaussNewtonConfig

    snap = _snapshot(4)
    backend = SyntheticBackend(snap, band=[1, 2, 3])
    norm, st = normalize(snap.fields[0], "zonal")
    target = to_separable(norm)
    grid = make_grid(SHAPE)
    vel, phys = subgrids(grid, SHAPE)
    bank = field_init(12, target, grid, SHAPE, seed=4)
    bank.n_base = 8
    bank.ky.data[8:] = 2 * math.pi * torch.tensor([1.0, 2.0, 3.0, 2.0])
    solve_amplitudes(bank, target, vel, phys, ridge=1e-3, per_column=False, chunk=64)
    sm = SpectralModel([bank], [vel], [phys], [st], backend, chunk=64)
    J, r = jacobian_autograd(sm)
    assert J.shape == (len(backend.band) * 2, 2 * bank.N)
    hist = gauss_newton(sm, GaussNewtonConfig(steps=6, mu=1e-4), log=lambda *a: None)
    assert hist["residual"][-1] < 0.5 * hist["residual"][0]


def test_recipe_runs_end_to_end():
    import os

    snap = _snapshot(6)
    backend = SyntheticBackend(snap, band=[1, 2, 3, 4])
    path = os.path.join(
        os.path.dirname(__file__), "..", "..", "configs", "gyrosplat", "recipe", "synthetic.yaml"
    )
    cfg = load_config(
        path,
        [
            "verbose=false",
            "fit.steps=20",
            "physics.epochs=3",
            "carriers.routed_steps=3",
            "gn.steps=4",
        ],
    )
    model = compress(backend, cfg, device="cpu")
    rows = model.info["rows"]
    tags = [r["tag"] for r in rows]
    assert tags == ["fit", "+ gauss-newton", "+ 16-bit parameters"]
    assert rows[1]["flux_relL1_band"][0] < rows[0]["flux_relL1_band"][0]
    assert model.info["gn"]["residual"][-1] < model.info["gn"]["residual"][0]
    bank = model.banks[0]
    assert bank.n_base == 25 and bank.n_carriers == 33
    assert torch.all(bank.ky.data[: bank.n_base] == 0)
    assert rows[-1]["CR"] > 0 and rows[-1]["packed_bytes"] > 0
    assert "physics_fit" in model.info["stage_seconds"]
