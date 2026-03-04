import pytest
import torch
import numpy as np
import os

from neugk.integrals import FluxIntegral, get_integrals
from neugk.utils import load_geometry, K_files, poten_files
from neugk.dataset.preprocess import phi_to_spc, phi_fft_to_real


@pytest.fixture(
    params=[
        "v3_kiteration_991_double_rlt",
        "v3_kiteration_991_half_rlt",
        "v3_kiteration_991_ntsks128",
    ]
)
def kinetic_dir(request):
    return f"/restricteddata/ukaea/gyrokinetics/raw/kinetic_electrons/{request.param}"


@pytest.fixture(params=[0, 8, 13, 131, 200])
def adiabatic_dir(request):
    return f"/restricteddata/ukaea/gyrokinetics/raw/iteration_{request.param}"


def test_flux_integral_shapes(adiabatic_dir):
    if not os.path.exists(adiabatic_dir):
        pytest.skip(f"Test directory {adiabatic_dir} not found.")

    geom = load_geometry(adiabatic_dir)
    # Check if geometry loaded correctly
    assert "adiabatic" in geom
    assert "de" in geom
    geom["adiabatic"] = torch.tensor(1.0, dtype=geom["adiabatic"].dtype)
    geom = {k: g.unsqueeze(0) for k, g in geom.items()}

    integrator = FluxIntegral()

    # Mock DF with shapes matching the loaded geometry
    y = geom["krho"].shape[1]
    s = geom["ints"].shape[1]
    vmu = geom["intmu"].shape[1]
    vpar = geom["intvp"].shape[1]
    x = geom["kxrh"].shape[1]

    df = torch.randn(1, 2, vpar, vmu, s, x, y, dtype=torch.float64)

    phi, (pflux, eflux, vflux) = integrator(geom, df)

    assert phi.shape == (1, 2, x, s, y)
    assert pflux.ndim == 1
    assert eflux.ndim == 1
    assert vflux.ndim == 1


def test_flux_integral_species_shapes(kinetic_dir):
    if not os.path.exists(kinetic_dir):
        pytest.skip(f"Test directory {kinetic_dir} not found.")

    geom = load_geometry(kinetic_dir)
    geom["adiabatic"] = torch.tensor(0.0, dtype=geom["adiabatic"].dtype)
    geom = {k: g.unsqueeze(0) for k, g in geom.items()}

    integrator = FluxIntegral()

    # Mock DF with shapes matching the loaded geometry
    y = geom["krho"].shape[1]
    s = geom["ints"].shape[1]
    vmu = geom["intmu"].shape[1]
    vpar = geom["intvp"].shape[1]
    x = geom["kxrh"].shape[1]

    sp = 2
    df = torch.randn(1, sp, 2, vpar, vmu, s, x, y, dtype=torch.float64)

    phi, (pflux, eflux, vflux) = integrator(geom, df)

    # Potential shouldn't have species axis
    assert phi.shape == (1, 2, x, s, y)

    # Fluxes should have species axis (batch, sp)
    assert pflux.shape == (1, sp)
    assert eflux.shape == (1, sp)
    assert vflux.shape == (1, sp)


@pytest.mark.parametrize("idx", [10, 20, 50, 80, 100, -1, -5])
def test_flux_integral_real_data_adiabatic(adiabatic_dir, idx):
    if not os.path.exists(adiabatic_dir):
        pytest.skip(f"Test directory {adiabatic_dir} not found.")

    geom = load_geometry(adiabatic_dir)
    # The dataset preprocess usually operates with float32 but geometry is loaded float64
    # keep geom float64 and df complex128 internally to match GKW precision exactly

    # read helper vars
    sgrid = np.loadtxt(f"{adiabatic_dir}/sgrid")
    xphi = np.loadtxt(f"{adiabatic_dir}/xphi")
    krho = np.loadtxt(f"{adiabatic_dir}/krho")
    vpgr = np.loadtxt(f"{adiabatic_dir}/vpgr.dat")
    ns = sgrid.shape[1] if len(sgrid.shape) > 1 else sgrid.shape[0]
    nx, ny = xphi.shape[1], xphi.shape[0]
    nkx, nky = krho.shape[1], krho.shape[0]
    nvpar, nmu = vpgr.shape[1], vpgr.shape[0]
    resolution = (nvpar, nmu, ns, nkx, nky)

    # K files and Potens
    ks = K_files(adiabatic_dir)
    potens, _ = poten_files(adiabatic_dir)

    if idx >= len(ks) or idx < -len(ks):
        pytest.skip(f"Index {idx} out of range for available K files ({len(ks)}).")

    k_file = ks[idx]
    pot_file = potens[idx]

    # load df
    with open(f"{adiabatic_dir}/{k_file}", "rb") as fid:
        ff = np.fromfile(fid, dtype=np.float64)
    # GKW output is (2, par, mu, s, kx, ky)
    knth = np.reshape(ff, (2, *resolution), order="F").astype("float32")

    # load potentials to construct a real potential
    a = np.loadtxt(f"{adiabatic_dir}/{pot_file}")
    phi_raw = np.reshape(a, (nx, ns, ny), order="F").astype("float32")
    spc_file = pot_file.replace("Poten", "Spc3d")
    b = np.loadtxt(f"{adiabatic_dir}/{spc_file}")
    gt_spc = np.reshape(b, (nkx, ns, nky), order="F")
    phi_fft_unpadded = phi_to_spc(phi_raw, gt_spc, out_shape=(nkx, ns, nky))
    phi_real = phi_fft_to_real(phi_fft_unpadded, out_shape=phi_fft_unpadded.shape)

    df_tensor = torch.tensor(knth, dtype=torch.float64)
    # The get_integrals function expects (c, par, mu, s, x, y)
    phi_pred, (_, eflux_pred, _) = get_integrals(
        df_tensor,
        geom,
        spectral_df=True,  # the original dump is in k-space already
    )

    # get the exact timestamp for this K file
    time_val = None
    with open(f"{adiabatic_dir}/{k_file}.dat", "r") as file:
        for line in file:
            line_split = line.split("=")
            if line_split[0].strip() == "TIME":
                time_val = float(line_split[1].strip().strip(",").strip())
                break

    orig_times = np.loadtxt(f"{adiabatic_dir}/time.dat")
    ts_idx = np.isclose(orig_times, time_val).nonzero()[0][0]

    fluxes = np.loadtxt(f"{adiabatic_dir}/fluxes.dat")[:, 1]
    orig_flux = fluxes[ts_idx]

    # Validate flux
    assert np.isclose(
        eflux_pred.sum().item(), orig_flux, rtol=1e-3, atol=1e-6
    ), f"Flux mismatch: {eflux_pred.sum().item()} vs {orig_flux}"


@pytest.mark.parametrize("idx", [10, 20, 50, 80, 100, -1, -5])
def test_flux_integral_real_data_kinetic(kinetic_dir, idx):
    if not os.path.exists(kinetic_dir):
        pytest.skip(f"Test directory {kinetic_dir} not found.")

    geom = load_geometry(kinetic_dir)

    # read helper vars
    sgrid = np.loadtxt(f"{kinetic_dir}/sgrid")
    xphi = np.loadtxt(f"{kinetic_dir}/xphi")
    krho = np.loadtxt(f"{kinetic_dir}/krho")
    vpgr = np.loadtxt(f"{kinetic_dir}/vpgr.dat")
    ns = sgrid.shape[1] if len(sgrid.shape) > 1 else sgrid.shape[0]
    nx, ny = xphi.shape[1], xphi.shape[0]
    nkx, nky = krho.shape[1], krho.shape[0]
    nvpar, nmu = vpgr.shape[1], vpgr.shape[0]
    resolution = (nvpar, nmu, ns, nkx, nky)

    ks = K_files(kinetic_dir)
    potens, _ = poten_files(kinetic_dir)

    if idx >= len(ks) or idx < -len(ks):
        pytest.skip(f"Index {idx} out of range for available K files ({len(ks)}).")

    k_file = ks[idx]
    pot_file = potens[idx]

    # load df
    with open(f"{kinetic_dir}/{k_file}", "rb") as fid:
        ff = np.fromfile(fid, dtype=np.float64)

    # GKW output for kinetic electrons is (c=2, par, mu, s, kx, ky, sp)
    sp = 2
    knth = np.reshape(ff, (2, *resolution, sp), order="F").astype("float32")
    # Move sp to the front to match PyTorch convention (sp, c, par, mu, s, x, y)
    knth = np.moveaxis(knth, -1, 0)

    df_tensor = torch.tensor(knth, dtype=torch.float64)
    phi_pred, (pflux_pred, eflux_pred, vflux_pred) = get_integrals(
        df_tensor,
        geom,
        spectral_df=True,
    )

    # get the exact timestamp for this K file
    time_val = None
    with open(f"{kinetic_dir}/{k_file}.dat", "r") as file:
        for line in file:
            line_split = line.split("=")
            if line_split[0].strip() == "TIME":
                time_val = float(line_split[1].strip().strip(",").strip())
                break

    orig_times = np.loadtxt(f"{kinetic_dir}/time.dat")
    ts_idx = np.isclose(orig_times, time_val).nonzero()[0][0]

    # For multiple species, fluxes are stored per species.
    # The columns in fluxes.dat are typically: pflux_1, eflux_1, vflux_1, pflux_2, eflux_2, vflux_2
    fluxes = np.loadtxt(f"{kinetic_dir}/fluxes.dat")

    # Validate all fluxes
    orig_pflux = fluxes[ts_idx, [0, 3]]
    orig_eflux = fluxes[ts_idx, [1, 4]]
    orig_vflux = fluxes[ts_idx, [2, 5]]

    pred_pflux = pflux_pred.detach().cpu().numpy().flatten()
    pred_eflux = eflux_pred.detach().cpu().numpy().flatten()
    pred_vflux = vflux_pred.detach().cpu().numpy().flatten()

    # Heat flux validation
    assert np.allclose(
        pred_eflux, orig_eflux, rtol=1e-2, atol=1e-5
    ), f"Heat flux mismatch at idx {idx}: {pred_eflux} vs {orig_eflux}"

    # Particle flux validation
    assert np.allclose(
        pred_pflux, orig_pflux, rtol=1e-2, atol=1e-5
    ), f"Particle flux mismatch at idx {idx}: {pred_pflux} vs {orig_pflux}"

    # Momentum flux validation
    assert np.allclose(
        pred_vflux, orig_vflux, rtol=1e-2, atol=1e-5
    ), f"Momentum flux mismatch at idx {idx}: {pred_vflux} vs {orig_vflux}"
