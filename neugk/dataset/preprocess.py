import warnings
import sys

sys.path.append("..")
import os
from tqdm import tqdm

import numpy as np
import h5py
import torch
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from argparse import ArgumentParser

from neugk.utils import (
    RunningMeanStd,
    pev_flux_df_phi,
    load_geometry,
    K_files,
    poten_files,
    parse_input_dat,
)
from neugk.integrals import FluxIntegral

parser = ArgumentParser()
parser.add_argument("--debug", action="store_true")
parser.add_argument("--num_workers", type=int, default=10)
parser.add_argument("--assemble_to_shard", action="store_true")
args = parser.parse_args()

ROOT = "/projects/u6eb/gyrokinetics"


def do_ifft(knth):
    knth = np.fft.ifftn(knth, axes=(3, 4), norm="forward")
    knth = np.stack([knth.real, knth.imag]).squeeze().astype("float32")
    return knth


def get_stats(filenames, spatial_ifft=False, separate_zf=False, per_mode_norm=False):
    running_stats = None
    old_running_stats = None
    for filename in filenames:
        dir_in = f"{ROOT}/raw/{filename}"
        dir_out = f"{ROOT}/preprocessed/{filename}"
        if not os.path.exists(dir_out):
            os.makedirs(dir_out)
        # create h5 file with timestamps and field data
        ifft_tag = "_ifft" if spatial_ifft else ""
        zf_tag = "_separate_zf" if separate_zf else ""
        per_mode_tag = "_per_mode" if per_mode_norm else ""
        h5_filename = f"{dir_out}/{filename}{ifft_tag}{zf_tag}{per_mode_tag}.h5"
        if os.path.exists(h5_filename):
            if old_running_stats is None:
                # load old stats
                old_file = h5py.File(h5_filename, "r")
                old_mean = old_file["metadata"]["k_mean"][:]
                old_var = old_file["metadata"]["k_std"][:] ** 2
                old_min = old_file["metadata"]["k_min"][:]
                old_max = old_file["metadata"]["k_max"][:]
                old_running_stats = RunningMeanStd(shape=old_mean.shape)
                old_running_stats.mean = old_mean
                old_running_stats.var = old_var
                old_running_stats.min = old_min
                old_running_stats.max = old_max

            print(f"File {h5_filename} already exists, skipping...")
            continue

        ks = K_files(dir_in)
        potens, ts_slices = poten_files(dir_in)
        # get timestamps
        ts = []
        for k in ks:
            # load corresponding timestep
            with open(f"{dir_in}/{k}.dat", "r") as file:
                for line in file:
                    line_split = line.split("=")
                    if line_split[0].strip() == "TIME":
                        time = float(line_split[1].strip().strip(",").strip())
                        ts.append(time)

        # read helper vars
        sgrid = np.loadtxt(f"{dir_in}/sgrid")
        krho = np.loadtxt(f"{dir_in}/krho")
        vpgr = np.loadtxt(f"{dir_in}/vpgr.dat")
        # number of parallel direction grid points
        ns = sgrid.shape[1] if len(sgrid.shape) > 1 else sgrid.shape[0]
        # number of modes in x and y direction
        nkx, nky = krho.shape[1], krho.shape[0]
        # get velocity space resolutions
        nvpar, nmu = vpgr.shape[1], vpgr.shape[0]

        resolution = (nvpar, nmu, ns, nkx, nky)
        if running_stats is None:
            running_stats = RunningMeanStd(shape=(2, 1, 1, 1, 1, nky))

        for idx, (k, pot) in tqdm(
            enumerate(zip(ks, potens)),
            f"Computing normalization for {filename}...",
            total=len(ks),
        ):
            # Load the full distribution function data
            with open(f"{dir_in}/{k}", "rb") as fid:
                ff = np.fromfile(fid, dtype=np.float64)

            # Reshape the distribution function (copy for speeed in stat computation)
            knth = np.reshape(ff, (2, *resolution), order="F").astype("float32").copy()

            mean = np.mean(knth, axis=(1, 2, 3, 4), keepdims=True)
            var = np.var(knth, axis=(1, 2, 3, 4), keepdims=True)
            min = np.min(knth, axis=(1, 2, 3, 4), keepdims=True)
            max = np.max(knth, axis=(1, 2, 3, 4), keepdims=True)
            running_stats.update(mean, var, min, max)

    if old_running_stats is not None:
        # combine new stats with old stats
        running_stats.combine(old_running_stats)

    return running_stats


def check_ifft(transformed, orig, zf_separated=False):
    if zf_separated:
        real_parts = transformed[::2]
        imag_parts = transformed[1::2]
        sum_real = np.sum(real_parts, axis=0)
        sum_imag = np.sum(imag_parts, axis=0)
        orig_ifft = np.concatenate(
            [np.expand_dims(sum_real, 0), np.expand_dims(sum_imag, 0)], axis=0
        )
    else:
        orig_ifft = transformed
    orig_ifft = np.moveaxis(orig_ifft, 0, -1).copy()
    orig_ifft = orig_ifft.view(dtype=np.complex64)
    orig_ifft = np.fft.fftn(orig_ifft, axes=(3, 4), norm="forward")
    orig_ifft = np.fft.ifftshift(orig_ifft, axes=(3,))
    orig_ifft = np.stack([orig_ifft.real, orig_ifft.imag]).squeeze().astype("float32")
    return np.allclose(orig_ifft, orig, rtol=0, atol=1e-5)


def _check_spc(abs_phi_fft, spc):
    return np.allclose(abs_phi_fft, spc, rtol=0.0, atol=1e-3)


def phi_to_spc(phi, gt_spc, out_shape, norm: str = "forward"):
    phi_fft = np.fft.fftn(phi, axes=(0, 2), norm=norm)
    phi_fft = np.fft.fftshift(phi_fft, axes=(0, 2))
    phi_fft = phi_fft[..., phi_fft.shape[-1] // 2 :]
    nkx, _, nky = out_shape
    xpad = (phi_fft.shape[0] - nkx) // 2
    xpad = xpad + 1 if (phi_fft.shape[0] % 2 == 0) else xpad
    phi_fft = phi_fft[xpad : nkx + xpad, :, :nky]
    assert _check_spc(np.abs(phi_fft), gt_spc), "Spectral space of Phi incorrect"
    # phi_fft = np.fft.ifftshift(phi_fft, axes=(0,))
    # phi_fft = np.fft.ifftn(phi_fft, axes=(0, 2), norm=norm)
    # spc = np.fft.fftn(phi_fft, axes=(0, 2), norm=norm)
    # spc = np.fft.fftshift(spc, axes=(0,))
    # assert _check_spc(np.abs(spc), gt_spc), "Spectral space of phi incorrect"
    # phi = np.stack([phi_fft.real, phi_fft.imag]).astype("float32")
    return phi_fft


def phi_fft_to_real(fft, out_shape, norm: str = "forward"):
    if fft.shape != out_shape:
        nkx, _, nky = out_shape
        nx, _, ny = fft.shape
        xpad = (nkx - nx) // 2 + 1
        padded = np.zeros(out_shape).astype(fft.dtype)
        padded[xpad : xpad + nx, :, :ny] = fft
    else:
        nkx, _, nky = fft.shape
        padded = fft
    phi = np.fft.fftshift(padded, axes=(0,))
    phi_ifft = np.fft.irfftn(phi, axes=(0, 2), norm=norm, s=[nkx, nky])
    return phi_ifft


def preprocess(
    filename,
    spatial_ifft=False,
    separate_zf=False,
    split_into_bands=None,
):
    assert not (
        separate_zf and not spatial_ifft
    ), "Need to perform IFFT to maintain shapes for separate_zf"
    dir_in = f"{ROOT}/raw/{filename}"
    dir_out = f"{ROOT}/preprocessed"
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)
    filename = filename.replace("/", "_")

    # create h5 file with timestamps and field data
    ifft_tag = "_ifft" if spatial_ifft else ""
    zf_tag = "_separate_zf" if separate_zf else ""
    split_into_bands_tag = f"_{split_into_bands}bands" if split_into_bands else ""
    h5_filename = f"{dir_out}/{filename}{ifft_tag}{zf_tag}{split_into_bands_tag}_realpotens.h5"
    write_mode = "w"
    if os.path.exists(h5_filename):
        print(f"File {h5_filename} already exists, skipping...")
        return h5_filename, True

    ks = K_files(dir_in.replace("_Lin", ""))
    potens, _ = poten_files(dir_in.replace("_Lin", ""))
    k_dir = dir_in.replace("_Lin", "")
    if not len(ks):
        # load k dump files of other sim, they are sampled the same anyways
        # this is only for extracting the correct flux timesteps
        ks = K_files("/restricteddata/ukaea/gyrokinetics/raw/iteration_0")
        potens, _ = poten_files("/restricteddata/ukaea/gyrokinetics/raw/iteration_0")
        k_dir = "/restricteddata/ukaea/gyrokinetics/raw/iteration_0"
    # get timestamps
    ts = []
    for k in ks:
        # load corresponding timestep
        with open(f"{k_dir}/{k}.dat", "r") as file:
            for line in file:
                line_split = line.split("=")
                if line_split[0].strip() == "TIME":
                    time = float(line_split[1].strip().strip(",").strip())
                    ts.append(time)
    timesteps = np.array(ts)
    
    # read helper vars
    sgrid = np.loadtxt(f"{dir_in}/sgrid")
    xphi = np.loadtxt(f"{dir_in}/xphi")
    krho = np.loadtxt(f"{dir_in}/krho")
    vpgr = np.loadtxt(f"{dir_in}/vpgr.dat")
    # number of parallel direction grid points
    ns = sgrid.shape[1] if len(sgrid.shape) > 1 else sgrid.shape[0]
    # number of x, y grid points (in real space)
    nx, ny = xphi.shape[1], xphi.shape[0]
    # number of modes in x and y direction
    nkx, nky = krho.shape[1], krho.shape[0]
    # get velocity space resolutions
    nvpar, nmu = vpgr.shape[1], vpgr.shape[0]

    resolution = (nvpar, nmu, ns, nkx, nky)

    # always load nonlinear fluxes
    fluxes = np.loadtxt(f"{dir_in.replace('_Lin', '')}/fluxes.dat")[:, 1]
    orig_fluxes = fluxes.copy()
    # print(ks)
    if not "Lin" in h5_filename:
        orig_times = np.loadtxt(f"{dir_in.replace('_Lin', '')}/time.dat")
        ts_slices = [np.isclose(orig_times, t).nonzero()[0][0] for t in timesteps]
        fluxes = fluxes[ts_slices]
        orig_fluxes = fluxes.copy()
    
    fluxes = np.clip(fluxes, a_min=0., a_max=None)
    # load parameters
    config = parse_input_dat(f"{dir_in}/input.dat")
    ion_temp_grad = config["species"]["rlt"]
    density_grad = config["species"]["rln"]
    s_hat = config["geom"]["shat"]
    q = config["geom"]["q"]

    shape = tuple(
        [
            resolution[ax - 1]
            for ax in np.arange(1, len(resolution) + 1)
        ]
    )
    if zf_tag:
        if split_into_bands:
            df_stats = RunningMeanStd(shape=((split_into_bands + 1) * 2,) + shape)
        else:
            df_stats = RunningMeanStd(shape=(4,) + shape)
    else:
        df_stats = RunningMeanStd(shape=(2,) + shape)
        phi_stats = RunningMeanStd((1,1,1))
        flux_stats = RunningMeanStd((1,))

    ks = K_files(dir_in.replace("_Lin", ""))
    potens, _ = poten_files(dir_in)
    if "Lin" in h5_filename:
        # if linear sim, only take last timestep
        ks = ["FDS"]
        potens = [potens[-1]]
        kyspec = np.loadtxt(dir_in.replace("_Lin", "/kyspec"))
        growth_rate = np.loadtxt(os.path.join(dir_in, "growth.dat"))[-1, :]
        ky_frequencies = np.loadtxt(os.path.join(dir_in, "frequencies.dat"))[-1, :]


    with h5py.File(h5_filename, write_mode) as file:

        geometry = load_geometry(dir_in)
        # group for metadata (e.g. timesteps)
        metadata_group = file.create_group("metadata")
        metadata_group.create_dataset("timesteps", data=timesteps)
        metadata_group.create_dataset("resolution", data=resolution)
        metadata_group.create_dataset("ion_temp_grad", data=ion_temp_grad, shape=(1,))
        metadata_group.create_dataset("density_grad", data=density_grad, shape=(1,))
        metadata_group.create_dataset("fluxes", data=fluxes)
        metadata_group.create_dataset("s_hat", data=s_hat, shape=(1,))
        metadata_group.create_dataset("q", data=q, shape=(1,))
        if "Lin" in h5_filename:
            metadata_group.create_dataset("kyspec", data=kyspec)
            metadata_group.create_dataset("growth_rate", data=growth_rate)
            metadata_group.create_dataset("frequency", data=ky_frequencies)
        geometry_group = file.create_group("geometry")
        np_geom = {
            k: geometry[k].numpy() if isinstance(geometry[k], torch.Tensor) 
            else geometry[k]
            for k in geometry.keys()
        }
        for key in np_geom.keys():
            geometry_group.create_dataset(key, data=np_geom[key])

        # group for our 6D field data
        data_group = file.create_group("data")
        for idx, (k, pot) in tqdm(
            enumerate(zip(ks, potens)),
            f"Processing {filename} -> {h5_filename}",
            total=len(ks),
        ):
            # Load the full distribution function data
            with open(f"{dir_in}/{k}", "rb") as fid:
                ff = np.fromfile(fid, dtype=np.float64)

            # Reshape the distribution function (copy for speeed in stat computation)
            knth = np.reshape(ff, (2, *resolution), order="F").astype("float32").copy()
            orig_knth = knth.copy()
            if spatial_ifft:
                knth = np.moveaxis(knth, 0, -1).copy()
                knth = knth.view(dtype=np.complex64)
                knth = np.fft.fftshift(knth, axes=(3,))
                separated_modes = []
                if separate_zf:
                    knth_zf = knth.copy()
                    knth_no_zf = knth.copy()
                    knth_zf[..., 1:, :] = 0.0
                    ifft_knth_zf = do_ifft(knth_zf)
                    separated_modes.append(ifft_knth_zf)
                    knth_no_zf[..., 0, :] = 0.0
                    if split_into_bands:
                        modes_per_channel = nky // split_into_bands
                        for band in range(split_into_bands):
                            cur_knth = np.zeros_like(knth_no_zf)
                            offset = 1 + band * modes_per_channel
                            if (split_into_bands - 1) == band:
                                # last band contains all remaining frequencies
                                cur_knth[..., offset:, :] = knth_no_zf[..., offset:, :]
                            else:
                                cur_knth[
                                    ..., offset : offset + modes_per_channel, :
                                ] = knth_no_zf[
                                    ..., offset : offset + modes_per_channel, :
                                ]
                            ifft_knth = do_ifft(cur_knth)
                            separated_modes.append(ifft_knth)
                    else:
                        ifft_knth_no_zf = do_ifft(knth_no_zf)
                        separated_modes.append(ifft_knth_no_zf)

                    knth = np.concatenate(separated_modes, axis=0)
                else:
                    knth = do_ifft(knth)

                assert check_ifft(
                    knth.copy(), orig_knth.copy()
                ), "Error transforming back to original space"

            # load the potential field
            a = np.loadtxt(f"{dir_in}/{pot}")
            phi = np.reshape(a, (nx, ns, ny), order="F").astype("float32").copy()
            if not "Lin" in h5_filename:
                spc_file = pot.replace("Poten", "Spc3d")
                b = np.loadtxt(f"{dir_in}/{spc_file}")
                gt_spc = np.reshape(b, (nkx, ns, nky), order="F")
            else:
                gt_spc = None
            phi_fft_unpadded = phi_to_spc(phi, out_shape=(nkx, ns, nky), gt_spc=gt_spc)
            phi = phi_fft_to_real(phi_fft_unpadded, out_shape=phi_fft_unpadded.shape)

            df = np.moveaxis(orig_knth, 0, -1).copy()
            df = df.view(dtype=np.complex64).squeeze()
            df = torch.tensor(df)
            
            if not "Lin" in h5_filename:
                # do not compute integral for linear sims => it will fail!
                phi_fft_unpadded = torch.tensor(phi_fft_unpadded)
                
                _, eflux, _ = pev_flux_df_phi(df, phi_fft_unpadded, geometry, aggregate=False)
                if not np.isclose(eflux.sum().item(), orig_fluxes[idx], rtol=0., atol=1e-4):
                    warnings.warn(
                        f"Flux integral does not match original flux! Computed: {eflux.sum().item()}, Original: {orig_fluxes[idx]}"
                    )
                # assert np.isclose(eflux.sum().item(), orig_fluxes[idx], rtol=0., atol=1.0), "Strong deviation for flux!!"

            # update running averages
            df_stats.update(knth, np.zeros_like(knth), knth, knth)
            flux_stats.update(fluxes[idx], np.zeros_like(fluxes[idx]), fluxes[idx], fluxes[idx])
            phi_stats.update(phi, np.zeros_like(phi), phi, phi)
            # Add the reshaped data as a dataset to the "data" group
            k_name = "timestep_" + str(idx).zfill(5)
            if k_name not in file["data"].keys():
                data_group.create_dataset(k_name, data=knth)
            poten_name = "poten_" + str(idx).zfill(5)
            if poten_name not in file["data"].keys():
                data_group.create_dataset(poten_name, data=phi)
            
        metadata_group.create_dataset("df_mean", data=df_stats.mean)
        metadata_group.create_dataset("df_std", data=np.sqrt(df_stats.var))
        metadata_group.create_dataset("df_min", data=df_stats.min)
        metadata_group.create_dataset("df_max", data=df_stats.max)
        metadata_group.create_dataset("phi_mean", data=phi_stats.mean)
        metadata_group.create_dataset("phi_std", data=np.sqrt(phi_stats.var))
        metadata_group.create_dataset("phi_min", data=phi_stats.min)
        metadata_group.create_dataset("phi_max", data=phi_stats.max)
        metadata_group.create_dataset("flux_mean", data=flux_stats.mean)
        metadata_group.create_dataset("flux_std", data=np.sqrt(flux_stats.var))
        metadata_group.create_dataset("flux_min", data=flux_stats.min)
        metadata_group.create_dataset("flux_max", data=flux_stats.max)
    
    return h5_filename, False


IFFT = True
separate_zf = False
split_into_bands = None
ifft_tag = "_ifft" if IFFT else ""
zf_tag = "_separate_zf" if separate_zf else ""
split_into_bands_tag = f"_{split_into_bands}bands" if split_into_bands else ""
# datasets = [file for file in os.listdir(f"{ROOT}/raw") if file.endswith("Lin")]
datasets = [f"iteration_{i}" for i in range(301)]
# df = pd.read_csv("/system/user/publicwork/fpaische/plasmamodelling/misc/original_unstable.csv")
# originals = df["name"].values
# originals = [
#     "iteration_262",
#     "iteration_115",
#     "iteration_148",
#     "iteration_235",
#     "iteration_131",
#     "iteration_8",
#     "ood/iteration_0",
#     "ood/iteration_1",
#     "ood/iteration_2",
#     "ood/iteration_3",
#     "ood/iteration_4",
# ]
# datasets = [f"{name}_Lin" for name in originals]

if not args.debug:
    # if we don't debug, we launch multiprocessing
    preprocess_fns = partial(
        preprocess,
        spatial_ifft=IFFT,
        separate_zf=separate_zf,
        split_into_bands=split_into_bands,
    )

    with ThreadPoolExecutor(args.num_workers) as executor:
        # indices in parallel, collect results in list
        returns = tqdm(
            executor.map(preprocess_fns, datasets),
            total=len(datasets),
            desc="Preprocessing data...",
        )

    for filename, skipped in returns:
        if skipped:
            print(f"{filename} was skipped!")
else:
    for f in datasets:
        try:
            h5_filename, skipped = preprocess(
                f,
                spatial_ifft=IFFT,
                separate_zf=separate_zf,
                split_into_bands=split_into_bands,
            )
        except Exception as e:
            print(f"Error processing {f}: {e}, deleting incomplete file if exists...")
            ifft_tag = "_ifft" if IFFT else ""
            h5_filename = f"{ROOT}/preprocessed/{f}{ifft_tag}_realpotens.h5"
            if os.path.exists(h5_filename):
                os.remove(h5_filename)
            continue

        # set rwx permissions
        try:
            os.chmod(h5_filename, 0o777)
        except PermissionError:
            pass

        if not skipped:
            # read in the structure and example field of the created h5 file
            with h5py.File(h5_filename, "r") as h5f:
                # Read the "metadata/timesteps" dataset
                timesteps = len(h5f["data"])
                rlt = h5f["metadata/ion_temp_grad"][:]
                timestep_0 = h5f["data/timestep_00000"][:]
                mean, std = h5f["metadata/df_mean"][0], h5f["metadata/df_std"][0]
                min_, max_ = h5f["metadata/df_min"][0], h5f["metadata/df_max"][0]
                print(
                    f"{h5_filename}:\n "
                    f"\tpoints: {timesteps}, shape of timestep_00000: {timestep_0.shape}\n"
                    f"\trlt: {rlt}\n"
                )

if args.assemble_to_shard:
    # After multiprocessing finishes:
    with h5py.File(f"{ROOT}/preprocessed/master_shard.h5", "w") as master:
        for small_h5 in glob.glob(f"{ROOT}/preprocessed/*.h5"):
            with h5py.File(small_h5, "r") as source:
                # This is a very fast metadata-level copy
                master.copy(source, small_h5.replace(".h5", ""))