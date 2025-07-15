import os
from argparse import ArgumentParser
import glob

import numpy as np
import torch

from neugk.utils import (
    phi_integral,
    load_geometry,
    pev_flux_df_phi,
    parse_input_dat,
    K_files,
    poten_files,
)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--path", required=True, type=str)
    parser.add_argument("--zf_mode", type=int, default=0)
    return parser.parse_args()


def dump_df_diagnostics(
    df, geometry, ds, nx, ny, ns, nkx, nky, dir, args, reduced=False
):

    phi_fft = phi_integral(torch.tensor(df), geometry).numpy()

    # phi is complex => pad and transform to real space again
    xpad = (nx - nkx) // 2 + 1
    padded = np.zeros((nx, ns, ny)).astype(phi_fft.dtype)
    padded[xpad : xpad + nkx, :, :nky] = phi_fft
    phi = np.fft.fftshift(padded, axes=(0,))
    phi = np.fft.irfftn(phi, axes=(0, 2), norm="forward", s=[nx, ny])

    filename = "Poten_FDS"
    if reduced:
        filename += "_reduced"
    with open(os.path.join(dir, filename), "wb") as f:
        print(f"Writing {dir}")
        f.write(phi)

    # compute kyspec
    W = np.sum(np.abs(phi_fft) ** 2, axis=(1,)) * ds
    W = np.sum(W, axis=0)
    filename = "kyspec"
    if reduced:
        filename += "_reduced"
    np.savetxt(os.path.join(dir, filename), W)

    # compute zf profile from 5D
    fourier_zf = phi_fft.copy()
    # mask everything except the zf_mode
    fourier_zf[..., : args.zf_mode] = 0.0
    fourier_zf[..., args.zf_mode + 1 :] = 0.0
    padded = np.zeros((nx, ns, ny)).astype(phi_fft.dtype)
    padded[xpad : xpad + nkx, :, :nky] = fourier_zf
    fourier_zf = np.fft.fftshift(padded, axes=(0,))
    phi_zf = np.fft.irfftn(fourier_zf, axes=(0, 2), norm="forward", s=[nx, ny])
    filename = f"df_zf_{args.zf_mode}mode_profile"
    if reduced:
        filename += "_reduced"
    with open(os.path.join(dir, filename), "wb") as f:
        f.write(phi_zf)

    # compute flux spectrum
    df = np.moveaxis(df, 0, -1).copy()
    df = df.view(dtype=np.complex64).squeeze()
    df = torch.tensor(df)
    _, eflux, _ = pev_flux_df_phi(df, torch.tensor(phi_fft), geometry, aggregate=False)
    flux_spectra = eflux.sum((0, 1, 2, 3)).numpy()
    filename = "eflux_spectra"
    if reduced:
        filename += "_reduced"
    np.savetxt(os.path.join(dir, filename), flux_spectra)
    filename = "eflux"
    if reduced:
        filename += "_reduced"
    np.savetxt(os.path.join(dir, filename), [flux_spectra.sum()])


def dump_phi_diagnostics(phi, nx, ns, ny, nkx, nky, dir, args, reduced=False, pad=True):
    phi_fft = np.fft.fftn(phi, axes=(0, 2), norm="forward")
    # compute zf profile from 3D
    fourier_zf = phi_fft.copy()
    # mask everything except the zf_mode
    fourier_zf[..., : args.zf_mode] = 0.0
    fourier_zf[..., args.zf_mode + 1 :] = 0.0
    if pad:
        padded = np.zeros((nx, ns, ny)).astype(phi_fft.dtype)
        xpad = (nx - nkx) // 2 + 1
        padded[xpad : xpad + nkx, :, :nky] = fourier_zf
    else:
        padded = fourier_zf
    fourier_zf = np.fft.fftshift(padded, axes=(0,))
    phi_zf = np.fft.irfftn(fourier_zf, axes=(0, 2), norm="forward", s=[nx, ny])
    filename = f"poten_zf_{args.zf_mode}mode_profile"
    if reduced:
        filename += "_reduced"
    with open(os.path.join(dir, filename), "wb") as f:
        f.write(phi_zf)


def main(args):
    path = args.path[:-1] if args.path.endswith("/") else args.path
    src_root = "/restricteddata/ukaea/gyrokinetics/raw"
    if "ood" in path:
        src_root = os.path.join(src_root, "ood")

    if src_root in args.path:
        sim = path.split("/")[-1]
        print(f"Computing diagnostics for original sim {sim}")
        k_dirs = K_files(path)
        potens, _ = poten_files(path)
        k_poten_dict = dict(zip(k_dirs, potens))
        compute_for_raw = True
        dump_path = os.path.join(path, "python_diagnostics")
        os.makedirs(dump_path, exist_ok=True)
    else:
        sim = path.split("/")[-2]
        k_dirs = [
            d
            for d in glob.glob(os.path.join(path, "*"))
            if os.path.isdir(d) and "plots" not in d
            if d.split("/")[-1].startswith("K")
        ]
        compute_for_raw = False

    geometry = load_geometry(os.path.join(src_root, sim))
    # load grid variables
    xphi = np.loadtxt(os.path.join(src_root, sim, "xphi"))
    sgrid = np.loadtxt(os.path.join(src_root, sim, "sgrid"))
    vpgr = np.loadtxt(os.path.join(src_root, sim, "vpgr.dat"))
    ds = sgrid[1] - sgrid[0]
    nvpar, nmu = vpgr.shape[1], vpgr.shape[0]
    ns = sgrid.shape[0]
    nx, ny = xphi.shape[1], xphi.shape[0]
    # load config
    config = parse_input_dat(os.path.join(src_root, sim, "input.dat"))
    nkx = config["gridsize"]["nx"]
    nky = config["gridsize"]["nmod"]

    pred_df = []
    pred_poten = []
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
        if not compute_for_raw:
            knth = np.reshape(ff, (2, *resolution)).astype("float32").copy()
        else:
            # order changed after inference
            knth = np.reshape(ff, (2, *resolution), order="F").astype("float32").copy()
        pred_df.append(knth)

        dump_df_diagnostics(knth, geometry, ds, nx, ny, ns, nkx, nky, dir, args)

        # compute zf profile from 3D potens
        if compute_for_raw:
            pred_file = os.path.join(path, k_poten_dict[dir.split("/")[-1]])
        else:
            pred_file = os.path.join(path, dir, "Poten")

        # Load dumped phi
        if not compute_for_raw:
            with open(pred_file, "rb") as fid:
                ff = np.fromfile(fid, dtype=np.float64)
            resolution = (nkx, ns, nky)
            # order is different if we load from predicted files
            real_phi = np.reshape(ff, resolution).astype("float32").copy()
        else:
            ff = np.loadtxt(pred_file, dtype=np.float64)
            resolution = (nx, ns, ny)
            real_phi = np.reshape(ff, resolution, order="F").astype("float32").copy()
        pred_poten.append(real_phi)

        dump_phi_diagnostics(
            real_phi, nx, ns, ny, nkx, nky, dir, args, pad=not compute_for_raw
        )

    # average over predicted df and phi
    pred_poten = np.mean(pred_poten, axis=0)
    pred_df = np.mean(pred_df, axis=0)
    dir = "/" + os.path.join(*dir.split("/")[:-1])

    # again compute diagnostics
    dump_df_diagnostics(
        pred_df, geometry, ds, nx, ny, ns, nkx, nky, dir, args, reduced=True
    )
    dump_phi_diagnostics(
        real_phi, nx, ns, ny, nkx, nky, dir, args, reduced=True, pad=not compute_for_raw
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
