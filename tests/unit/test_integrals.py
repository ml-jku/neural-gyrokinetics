import pytest
import torch
import numpy as np
import os

from neugk.integrals import FluxIntegral, get_integrals
from neugk.utils import load_geometry, K_files, poten_files
from neugk.dataset.preprocess import phi_to_spc, phi_fft_to_real


@pytest.fixture
def base_dir():
    return "/restricteddata/ukaea/gyrokinetics/raw/kinetic_electrons/v3_kiteration_991_double_rlt"


@pytest.fixture
def adiabatic_dir():
    return "/restricteddata/ukaea/gyrokinetics/raw/iteration_13"


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


def test_flux_integral_species_shapes(base_dir):
    if not os.path.exists(base_dir):
        pytest.skip(f"Test directory {base_dir} not found.")

    geom = load_geometry(base_dir)
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


@pytest.mark.parametrize("idx", [10, 50, 100, -1])
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

    # check if predicted potential matches the ground truth (Spc3d)
    phi_pred_np = phi_pred.detach().cpu().numpy()
    phi_pred_np = phi_pred_np[0] + 1j * phi_pred_np[1]

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
        eflux_pred.sum().item(), orig_flux, rtol=0.0, atol=1e-2
    ), f"Flux mismatch: {eflux_pred.sum().item()} vs {orig_flux}"


@pytest.mark.skip(
    reason="Kinetic fluxes require electromagnetic components (A_parallel, B_parallel) which are not yet implemented in FluxIntegral"
)
def test_flux_integral_real_data_kinetic(base_dir):
    if not os.path.exists(base_dir):
        pytest.skip(f"Test directory {base_dir} not found.")

    geom = load_geometry(base_dir)

    # read helper vars
    sgrid = np.loadtxt(f"{base_dir}/sgrid")
    xphi = np.loadtxt(f"{base_dir}/xphi")
    krho = np.loadtxt(f"{base_dir}/krho")
    vpgr = np.loadtxt(f"{base_dir}/vpgr.dat")
    ns = sgrid.shape[1] if len(sgrid.shape) > 1 else sgrid.shape[0]
    nx, ny = xphi.shape[1], xphi.shape[0]
    nkx, nky = krho.shape[1], krho.shape[0]
    nvpar, nmu = vpgr.shape[1], vpgr.shape[0]
    resolution = (nvpar, nmu, ns, nkx, nky)

    ks = K_files(base_dir)
    potens, _ = poten_files(base_dir)
    idx = 50  # pick a random intermediate index
    k_file = ks[idx]
    pot_file = potens[idx]

    # load df
    with open(f"{base_dir}/{k_file}", "rb") as fid:
        ff = np.fromfile(fid, dtype=np.float64)

    # GKW output for kinetic electrons is (c=2, par, mu, s, kx, ky, sp)
    sp = 2
    knth = np.reshape(ff, (2, *resolution, sp), order="F").astype("float32")
    # Move sp to the front to match PyTorch convention (sp, c, par, mu, s, x, y)
    knth = np.moveaxis(knth, -1, 0)

    # load potentials to construct a real potential
    a = np.loadtxt(f"{base_dir}/{pot_file}")
    phi_raw = np.reshape(a, (nx, ns, ny), order="F").astype("float32")
    spc_file = pot_file.replace("Poten", "Spc3d")
    b = np.loadtxt(f"{base_dir}/{spc_file}")
    gt_spc = np.reshape(b, (nkx, ns, nky), order="F")
    phi_fft_unpadded = phi_to_spc(phi_raw, gt_spc, out_shape=(nkx, ns, nky))
    phi_real = phi_fft_to_real(phi_fft_unpadded, out_shape=phi_fft_unpadded.shape)

    # get_integrals expects (c, x, s, y) where c=2 (real and imaginary parts)
    # since we used irfftn, the result is purely real, so imag part is 0
    phi_real_complex = np.stack(
        [phi_real, np.zeros_like(phi_real)], axis=0
    )  # (2, x, s, y)

    df_tensor = torch.tensor(knth, dtype=torch.float64)
    phi_real_tensor = torch.tensor(phi_real_complex, dtype=torch.float32)
    phi_pred, (pflux_pred, eflux_pred, vflux_pred) = get_integrals(
        df_tensor,
        geom,
        phi=phi_real_tensor,
        spectral_df=True,
    )

    # get the exact timestamp for this K file
    time_val = None
    with open(f"{base_dir}/{k_file}.dat", "r") as file:
        for line in file:
            line_split = line.split("=")
            if line_split[0].strip() == "TIME":
                time_val = float(line_split[1].strip().strip(",").strip())
                break

    orig_times = np.loadtxt(f"{base_dir}/time.dat")
    ts_idx = np.isclose(orig_times, time_val).nonzero()[0][0]

    # For multiple species, fluxes are stored per species.
    # The columns in fluxes.dat are typically: pflux_1, eflux_1, vflux_1, pflux_2, eflux_2, vflux_2
    fluxes = np.loadtxt(f"{base_dir}/fluxes.dat")
    orig_eflux = fluxes[ts_idx, [1, 4]]

    # Validate flux for both species
    # To match GKW fluxes, the raw integrals must be multiplied by species-specific constants
    # The pure phase space integration d3v uses normalized velocities.
    # The ExB drift brings a factor of v_th,s (actually just mass and temp scaling).
    # Based on GKW gyro-bohm normalizations for the decomposed species energy flux:
    # Q_s = raw_flux * n_s * T_s * v_th,s / (n_ref * T_ref * v_th,ref)
    pred_eflux = eflux_pred.detach().cpu().numpy()

    # The flux also lacks the A_parallel and B_parallel components which are non-zero
    # for kinetic electrons. Thus, they will still not perfectly match the total ExB+EM flux.
    assert np.allclose(
        pred_eflux, orig_eflux, rtol=0.0, atol=1e-2
    ), f"Flux mismatch: {pred_eflux} vs {orig_eflux}"
