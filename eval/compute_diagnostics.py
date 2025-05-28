import sys
sys.path.append('..')
import os
from argparse import ArgumentParser
import glob
from utils import (
    phi_integral,
    load_geometry,
    pev_flux_df_phi,
    parse_input_dat,
    K_files,
    poten_files
)
import numpy as np
import torch

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--path", required=True, type=str)
    return parser.parse_args()

def main(args):
    path = args.path[:-1] if args.path.endswith("/") else args.path
    src_root = "/restricteddata/ukaea/gyrokinetics/raw"
    if "ood" in path:
        src_root = os.path.join(src_root, "ood")

    if src_root in args.path:
        sim = path.split("/")[-1]
        print(f"Computing diagnostics for original sim {sim}")
        k_dirs = K_files(path)
        potens = poten_files(path)
        k_poten_dict = dict(zip(k_dirs, potens))
        compute_for_raw = True
        dump_path = os.path.join(path, "python_diagnostics")
        os.makedirs(dump_path, exist_ok=True)
    else:
        sim = path.split("/")[-2]
        k_dirs = [d for d in glob.glob(os.path.join(path, "*")) if os.path.isdir(d) and not "plots" in d
                  if d.split("/")[-1].startswith("K")]
        compute_for_raw = False

    geometry = load_geometry(os.path.join(src_root, sim))
    # load grid variables
    xphi = np.loadtxt(os.path.join(src_root, sim, "xphi"))
    sgrid = np.loadtxt(os.path.join(src_root, sim, 'sgrid'))
    vpgr = np.loadtxt(os.path.join(src_root, sim, 'vpgr.dat'))
    ds = sgrid[1] - sgrid[0]
    nvpar, nmu = vpgr.shape[1], vpgr.shape[0]
    ns = sgrid.shape[0]
    nx, ny = xphi.shape[1], xphi.shape[0]
    # load config
    config = parse_input_dat(os.path.join(src_root, sim, "input.dat"))
    nkx = config["gridsize"]["nx"]
    nky = config["gridsize"]["nmod"]

    for dir in k_dirs:

        if compute_for_raw:
            pred_file = os.path.join(path, dir)
            dir = os.path.join(dump_path, dir)
            os.makedirs(dir, exist_ok=True)
        else:
            pred_file = os.path.join(dir, "FDS")

        # Load the full distribution function data
        with open(pred_file, "rb") as fid:
            ff = np.fromfile(fid, dtype=np.float64)

        resolution = (nvpar, nmu, ns, nkx, nky)
        # Reshape the distribution function (copy for speeed in stat computation)
        knth = np.reshape(ff, (2, *resolution), order="F").astype("float32").copy()
        phi_fft = phi_integral(torch.tensor(knth), geometry).numpy()

        # phi is complex => pad and transform to real space again
        xpad = (nx - nkx) // 2 + 1
        padded = np.zeros((nx, ns, ny)).astype(phi_fft.dtype)
        padded[xpad:xpad + nkx, :, :nky] = phi_fft
        phi = np.fft.fftshift(padded, axes=(0,))
        phi = np.fft.irfftn(phi, axes=(0, 2), norm="forward", s=[nx, ny])

        with open(os.path.join(dir, "Poten_FDS"), "wb") as f:
            print(f"Writing {dir}")
            f.write(phi)

        # compute kyspec
        W = np.sum(np.abs(phi_fft) ** 2, axis=(1,)) * ds
        W = np.sum(W, axis=0)
        np.savetxt(os.path.join(dir, "kyspec"), W)

        # compute zf profile from 5D
        fourier_zf = phi_fft.copy()
        fourier_zf[..., 1:] = 0.
        padded = np.zeros((nx, ns, ny)).astype(phi_fft.dtype)
        padded[xpad:xpad + nkx, :, :nky] = fourier_zf
        fourier_zf = np.fft.fftshift(padded, axes=(0,))
        phi_zf = np.fft.irfftn(fourier_zf, axes=(0, 2), norm="forward", s=[135, 96])
        with open(os.path.join(dir, "df_zf_profile"), "wb") as f:
            f.write(phi_zf)

        # compute zf profile from 3D potens
        if compute_for_raw:
            pred_file = os.path.join(path, dir, k_poten_dict[dir])
        else:
            pred_file = os.path.join(path, dir, "Poten")

        # Load dumped phi
        with open(pred_file, "rb") as fid:
            ff = np.fromfile(fid, dtype=np.float64)
        resolution = (ns, nkx, nky)
        real_phi = np.reshape(ff, (2, *resolution), order="F").astype("float32").copy()

        phi_fft = np.fft.fftn(real_phi, axes=(0, 2), norm="forward")
        # compute zf profile from 3D
        fourier_zf = phi_fft.copy()
        fourier_zf[..., 1:] = 0.
        padded = np.zeros((nx, ns, ny)).astype(phi_fft.dtype)
        padded[xpad:xpad + nkx, :, :nky] = fourier_zf
        fourier_zf = np.fft.fftshift(padded, axes=(0,))
        phi_zf = np.fft.irfftn(fourier_zf, axes=(0, 2), norm="forward", s=[135, 96])
        with open(os.path.join(dir, "poten_zf_profile"), "wb") as f:
            f.write(phi_zf)

        # compute flux spectrum
        df = np.moveaxis(knth, 0, -1).copy()
        df = df.view(dtype=np.complex64).squeeze()
        df = torch.tensor(df)
        _, eflux, _ = pev_flux_df_phi(df, torch.tensor(phi_fft), geometry, aggregate=False)
        flux_spectra = eflux.sum((0,1,2,3)).numpy()
        np.savetxt(os.path.join(dir, "eflux_spectra"), flux_spectra)
        np.savetxt(os.path.join(dir, "eflux"), [flux_spectra.sum()])


if __name__ == '__main__':
    args = parse_args()
    main(args)