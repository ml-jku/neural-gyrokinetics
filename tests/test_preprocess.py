"""Preprocessing transforms, statistics, raw readers, potential rewrite and quantization."""

from __future__ import annotations

import importlib.util
import os
import pickle

import numpy as np
import pytest

from neugk_jax.dataset import preprocess as P
from neugk_jax.evaluate import fourier as F

needs_gyaradax = pytest.mark.skipif(
    importlib.util.find_spec("gyaradax") is None, reason="gyaradax not installed"
)

NVP, NMU, NS, NKX, NKY = 4, 2, 4, 7, 4


def _geometry():
    rng = np.random.default_rng(0)
    krho = np.linspace(0.0, 0.6, NKY)
    return {
        "krho": krho,
        "kxrh": np.fft.fftshift(np.fft.fftfreq(NKX)) * 2.0,
        "ints": np.full(NS, 1.0 / NS),
        "intmu": np.full(NMU, 0.5),
        "intvp": np.full(NVP, 0.25),
        "vpgr": np.linspace(-2, 2, NVP),
        "mugr": np.linspace(0.1, 1.0, NMU),
        "bn": 1.0 + 0.1 * rng.random(NS),
        "efun": 0.5 + 0.1 * rng.random(NS),
        "rfun": np.ones(NS),
        "bt_frac": np.ones(NS),
        "little_g": np.stack([np.ones(NS), 0.1 * np.ones(NS), np.ones(NS)], -1),
        "parseval": np.array([1.0] + [float(NKY)] * (NKY - 1)),
        "signz": np.ones(1),
        "vthrat": np.ones(1),
        "tmp": np.ones(1),
        "mas": np.ones(1),
        "de": np.ones(1),
        "d2X": np.array(1.0),
        "signB": np.array(1.0),
        "adiabatic": np.array(1.0),
        "beta": np.array(0.0),
        "nlapar": np.array(0.0),
        "nlbpar": np.array(0.0),
    }


def _spectral_df(rng, shape=(NVP, NMU, NS, NKX, NKY)):
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)


def _centred_one_sided(g, nky):
    # kx-centred (fftshift over x) one-sided ky spectrum of a real (x, s, y) field
    spec = np.fft.fftshift(np.fft.rfftn(g, axes=(0, 2), norm="forward"), axes=(0,))
    out = np.zeros((*spec.shape[:2], nky), dtype=spec.dtype)
    out[..., : spec.shape[-1]] = spec[..., :nky]
    return out


@pytest.mark.parametrize("nkx", [85, 84])
def test_phi_fft_to_real_inverts_kx_centring(nkx):
    g = np.random.default_rng(1).standard_normal((nkx, 3, 32))
    spec = _centred_one_sided(g, 32)
    np.testing.assert_allclose(F.spec_to_phi(spec), g, atol=1e-10)
    wrong = np.fft.irfftn(
        np.fft.fftshift(spec, axes=(0,)), axes=(0, 2), norm="forward", s=[nkx, 32]
    )
    assert np.allclose(wrong, g) == (nkx % 2 == 0)


def test_phi_to_spc_extracts_centred_window_odd_nkx():
    rng = np.random.default_rng(2)
    nkx, ns, nky, nx, ny = 85, 3, 32, 135, 96
    spec = _centred_one_sided(rng.standard_normal((nkx, ns, 2 * nky)), nky)
    full = np.zeros((nx, ns, ny), dtype=complex)
    x0, y0, c = (nx - nkx) // 2, ny // 2, nkx // 2
    full[x0 : x0 + nkx, :, y0 : y0 + nky] = spec
    for ky in range(1, nky):
        for kx in range(-c, c + 1):
            full[x0 + c - kx, :, y0 - ky] = np.conj(spec[c + kx, :, ky])
    phi = np.fft.ifftn(np.fft.ifftshift(full, axes=(0, 2)), axes=(0, 2), norm="forward")
    assert np.max(np.abs(phi.imag)) < 1e-10
    got = F.phi_to_spec(phi.real, (nkx, ns, nky))
    np.testing.assert_allclose(got, spec, atol=1e-10)
    real = F.spec_to_phi(got)
    np.testing.assert_allclose(
        F.phi_to_spec(real, got.shape)[..., : nky // 2], spec[..., : nky // 2], atol=1e-10
    )


def test_df_spectrum_roundtrip_and_gkw_convention():
    k = _spectral_df(np.random.default_rng(3))
    real = F.spec_to_df(k)
    assert real.shape == (2, NVP, NMU, NS, NKX, NKY) and real.dtype == np.float32
    np.testing.assert_allclose(F.df_to_spec(real), k, atol=1e-5)
    shifted = np.fft.ifftn(np.fft.fftshift(k, axes=(3,)), axes=(3, 4), norm="forward")
    np.testing.assert_array_equal(real, np.stack([shifted.real, shifted.imag]).astype(np.float32))
    import jax.numpy as jnp

    np.testing.assert_allclose(np.asarray(F.spec_to_df(jnp.asarray(k))), real, atol=1e-5)
    raw = np.stack([k.real, k.imag]).astype(np.float32)
    assert P.check_ifft(real, raw)
    assert not P.check_ifft(real, raw + 1e-3)


def test_split_modes_recombine_and_check():
    from neugk_jax.utils import recombine_zf

    k = _spectral_df(np.random.default_rng(4))
    for bands in (None, 2):
        parts = np.concatenate(P._split_modes(k, bands), axis=0)
        assert parts.shape[0] == 2 * (2 if bands is None else 1 + bands)
        np.testing.assert_allclose(recombine_zf(parts, axis=0), F.spec_to_df(k), atol=1e-5)
        assert P.check_ifft(parts, np.stack([k.real, k.imag]), zf_separated=True)


def test_running_stats_push_matches_merge():
    from neugk_jax.utils import RunningStats

    xs = np.random.default_rng(5).standard_normal((6, 3, 4)) * 1e-3 + 0.2
    ref, got = RunningStats(prior_count=1e-4), RunningStats(prior_count=1e-4)
    for x in xs:
        ref.merge(x, np.zeros_like(x), x, x)
        got.push(x)
    for name in ("mean", "var", "min", "max"):
        np.testing.assert_allclose(getattr(got, name), getattr(ref, name), rtol=1e-12)
    np.testing.assert_allclose(got.mean, xs.sum(0) / (len(xs) + 1e-4), rtol=1e-12)
    scalar = RunningStats(prior_count=1e-4)
    for x in xs[:, 0, 0]:
        scalar.push(x)
    np.testing.assert_allclose(scalar.mean, got.mean[0, 0], rtol=1e-12)
    pooled = got.moments(axes=(1,))
    np.testing.assert_allclose(
        pooled["var"], got.var.mean(1, keepdims=True) + got.mean.var(1, keepdims=True)
    )


def test_expand_spec_and_resolve(tmp_path):
    from neugk_jax.dataset.backend import expand_spec

    assert expand_spec("it_{0-2,5}_x") == ["it_0_x", "it_1_x", "it_2_x", "it_5_x"]
    assert expand_spec(["a", "b"]) == ["a", "b"]
    assert expand_spec(["b_{1-2}"]) == ["b_1", "b_2"]
    for n in ("b_ifft_realpotens", "a_ifft_realpotens", "c_other"):
        (tmp_path / n).mkdir()
    assert [os.path.basename(p) for p in P.resolve_traj_dirs(str(tmp_path))] == [
        "a_ifft_realpotens",
        "b_ifft_realpotens",
    ]


def test_gkw_text_readers(tmp_path):
    (tmp_path / "input.dat").write_text(
        "&control\n nlapar = .true.\n/\n&species\n mass = 1.0, z = 1.0, rlt = 6.9\n/\n"
        "&species\n mass = 2.7e-4, z = -1\n/\n"
    )
    cfg = P.parse_input_dat(str(tmp_path / "input.dat"))
    assert cfg["control"]["nlapar"] == ".true." and cfg["species"]["rlt"] == 6.9
    assert cfg["species0"]["z"] == -1 and P._gkw_bool(cfg["control"]["nlapar"]) == 1.0
    (tmp_path / "geom.dat").write_text("kthnorm\n 3.5\nbn\n 1.0 1.1\n 1.2\n")
    geom = P.load_geom_dat(str(tmp_path / "geom.dat"))
    np.testing.assert_array_equal(geom["bn"], [1.0, 1.1, 1.2])
    np.testing.assert_array_equal(geom["kthnorm"], [3.5])
    (tmp_path / "K02").touch()
    (tmp_path / "K01").touch()
    (tmp_path / "K01.dat").write_text(" TIME = 1.25,\n")
    (tmp_path / "10").touch()
    (tmp_path / "9").touch()
    assert P.k_files(str(tmp_path)) == ["K01", "K02", "9", "10"]
    assert P.read_dump_time(str(tmp_path / "K01.dat")) == 1.25


@pytest.mark.parametrize("bits", ["bf16", "fp16", "i8", "i4"])
def test_quantize_roundtrip(tmp_path, bits):
    from neugk_jax.dataset import quant

    x = np.random.default_rng(6).standard_normal(1001).astype(np.float32)
    payload, scale = quant.quantize(x, bits)
    dst = quant.sibling(str(tmp_path / "timestep_00000.bin"), bits)
    quant.write(dst, payload, scale)
    y = quant.read(dst, bits, x.size)
    tol = {"bf16": 1e-2, "fp16": 1e-3, "i8": 2e-2, "i4": 0.3}[bits]
    assert np.max(np.abs(y - x)) <= tol * np.max(np.abs(x))
    np.testing.assert_array_equal(quant.roundtrip(x, bits), y)
    with pytest.raises(IOError):
        quant.read(dst, bits, x.size + 2)


def test_field_solver_spectrum_sums_to_flux_and_matches_flux_integral():
    import jax
    import jax.numpy as jnp

    from neugk_jax.evaluate.integrals import flux_integral, precompute_geometry

    geom = _geometry()
    df = F.spec_to_df(_spectral_df(np.random.default_rng(7)))
    solver = P.FieldSolver(geom)
    phi, eflux = solver(df)
    assert phi.shape == (NKX, NS, NKY) and phi.dtype == np.float32
    np.testing.assert_allclose(solver.flux_spectrum(df).sum(), eflux, rtol=1e-10)
    phi32, (_, ef32, _) = jax.jit(flux_integral)(precompute_geometry(geom), jnp.asarray(df))
    np.testing.assert_allclose(phi, np.asarray(phi32), rtol=1e-4, atol=1e-5 * np.abs(phi).max())
    np.testing.assert_allclose(eflux, float(ef32), rtol=1e-4)


def _synthetic_traj(root, n=3):
    traj = os.path.join(root, "iteration_0_ifft_realpotens")
    os.makedirs(os.path.join(traj, "data"))
    rng = np.random.default_rng(8)
    for i in range(n):
        F.spec_to_df(_spectral_df(rng)).tofile(os.path.join(traj, "data", f"timestep_{i:05d}.bin"))
        np.zeros((NKX, NS, NKY), np.float32).tofile(
            os.path.join(traj, "data", f"poten_{i:05d}.bin")
        )
    meta = {
        "resolution": (NVP, NMU, NS, NKX, NKY),
        "geometry": _geometry(),
        "timesteps": np.arange(n, dtype=float),
        "extra": "kept",
        **{
            f"phi_{k}": np.zeros((NKX, NS, NKY), np.float32)
            for k in ("mean", "var", "std", "min", "max")
        },
        "df_mean": np.zeros(1, np.float32),
    }
    P.write_metadata(traj, meta)
    return traj


def test_rewrite_poten_backs_up_and_resolves(tmp_path):
    traj = _synthetic_traj(str(tmp_path / "data"))
    backup = str(tmp_path / "backup")
    msg = P.rewrite_poten(traj, backup)
    assert "rewrote 3" in msg and "skip" in P.rewrite_poten(traj, backup)
    bdir = os.path.join(backup, os.path.basename(traj))
    assert sorted(os.listdir(bdir)) == [
        "DONE",
        "metadata.pkl",
        "metadata_light.pkl",
        "poten_00000.bin",
        "poten_00001.bin",
        "poten_00002.bin",
    ]
    assert not np.fromfile(os.path.join(bdir, "poten_00001.bin"), np.float32).any()
    solver = P.FieldSolver(_geometry())
    phis = []
    for i in range(3):
        df = np.fromfile(os.path.join(traj, "data", f"timestep_{i:05d}.bin"), np.float32)
        phi = solver(df.reshape(2, NVP, NMU, NS, NKX, NKY))[0]
        np.testing.assert_array_equal(
            np.fromfile(os.path.join(traj, "data", f"poten_{i:05d}.bin"), np.float32), phi.ravel()
        )
        phis.append(phi)
    with open(os.path.join(traj, "metadata.pkl"), "rb") as f:
        meta = pickle.load(f)
    with open(os.path.join(traj, "metadata_light.pkl"), "rb") as f:
        light = pickle.load(f)
    assert meta["extra"] == "kept" and meta["phi_mean"].dtype == np.float32
    np.testing.assert_allclose(meta["phi_mean"], np.sum(phis, 0) / (3 + 1e-4), rtol=1e-5)
    np.testing.assert_allclose(meta["phi_max"], np.max(phis, 0))
    np.testing.assert_array_equal(light["phi_std"], meta["phi_std"])
    assert "phi_var" not in light
