import os
import queue
import warnings

from tqdm import tqdm
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np
import torch

from neugk.utils import (
    RunningMeanStd,
    load_geometry,
    K_files,
    poten_files,
    parse_input_dat,
)
from neugk.integrals import get_integrals

from neugk.dataset.backend import H5Backend, KvikIOBackend, DataBackend


def do_ifft(knth):
    knth = np.fft.ifftn(knth, axes=(3, 4), norm="forward")
    knth = np.stack([knth.real, knth.imag]).squeeze().astype("float32")
    return knth


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


def phi_to_spc(phi, gt_spc, out_shape, norm="forward"):
    phi_fft = np.fft.fftn(phi, axes=(0, 2), norm=norm)
    phi_fft = np.fft.fftshift(phi_fft, axes=(0, 2))
    phi_fft = phi_fft[..., phi_fft.shape[-1] // 2 :]
    nkx, _, nky = out_shape
    xpad = (phi_fft.shape[0] - nkx) // 2
    xpad = xpad + 1 if (phi_fft.shape[0] % 2 == 0) else xpad
    phi_fft = phi_fft[xpad : nkx + xpad, :, :nky]
    assert _check_spc(np.abs(phi_fft), gt_spc), "Spectral space of Phi incorrect"
    return phi_fft


def phi_fft_to_real(fft, out_shape, norm="forward"):
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
    filename: str,
    backend: DataBackend,
    spatial_ifft: bool = False,
    separate_zf: bool = False,
    split_into_bands=None,
    root: str = "/restricteddata/ukaea/gyrokinetics",
    target_dir: str = "/local00/bioinf/galletti",
    position_queue: queue.Queue = None,
):
    # Grab a dedicated row for this worker's progress bar (default to 0 if single-threaded)
    pos = position_queue.get() if position_queue is not None else 0

    try:
        assert not (
            separate_zf and not spatial_ifft
        ), "Need to perform IFFT to maintain shapes for separate_zf"

        target_dir = root if target_dir is None else target_dir
        dir_in = f"{root}/raw/{filename}"

        if isinstance(backend, KvikIOBackend):
            dir_out = f"{target_dir}/preprocessed_kvikio"
        else:
            dir_out = f"{target_dir}/preprocessed"

        os.makedirs(dir_out, exist_ok=True)
        safe_filename = filename.replace("/", "_")

        # format path via backend
        base_path = os.path.join(dir_out, safe_filename)
        out_path = backend.format_path(
            base_path, spatial_ifft, split_into_bands, real_potens=True
        )

        if backend.exists(out_path):
            return out_path, True

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
            with open(f"{dir_in.replace('_Lin', '')}/{k}.dat", "r") as file:
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
        if not "Lin" in out_path:
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
            
        df_stats = RunningMeanStd()
        phi_stats = RunningMeanStd()
        flux_stats = RunningMeanStd()

        if "Lin" in out_path:
            # if linear sim, only take last timestep
            ks = ["FDS"]
            potens = [potens[-1]]
            kyspec = np.loadtxt(dir_in.replace("_Lin", "/kyspec"))
            growth_rate = np.loadtxt(os.path.join(dir_in, "growth.dat"))[-1, :]
            ky_frequencies = np.loadtxt(os.path.join(dir_in, "frequencies.dat"))[-1, :]

        geometry = load_geometry(dir_in)
        np_geom = {
            k: (geometry[k].numpy() if hasattr(geometry[k], "numpy") else geometry[k])
            for k in geometry.keys()
        }

        metadata = {
            "timesteps": timesteps,
            "resolution": resolution,
            "ion_temp_grad": np.array([ion_temp_grad]),
            "density_grad": np.array([density_grad]),
            "fluxes": fluxes,
            "s_hat": np.array([s_hat]),
            "q": np.array([q]),
            "geometry": np_geom,
        }

        with backend.create(out_path) as f:
            innter_pbar = zip(ks, potens)
            if args.tqdm:
                innter_pbar = tqdm(
                    innter_pbar, desc=filename, total=len(ks), position=pos, leave=False
                )

            for idx, (k, pot) in enumerate(innter_pbar):
                # load df
                with open(f"{dir_in}/{k}", "rb") as fid:
                    ff = np.fromfile(fid, dtype=np.float64)

                knth = (
                    np.reshape(ff, (2, *resolution), order="F").astype("float32").copy()
                )
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
                                    cur_knth[..., offset:, :] = knth_no_zf[
                                        ..., offset:, :
                                    ]
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
                if not "Lin" in out_path:
                    spc_file = pot.replace("Poten", "Spc3d")
                    b = np.loadtxt(f"{dir_in}/{spc_file}")
                    gt_spc = np.reshape(b, (nkx, ns, nky), order="F")
                else:
                    gt_spc = None
                phi_fft_unpadded = phi_to_spc(phi, gt_spc, out_shape=(nkx, ns, nky))
                phi = phi_fft_to_real(
                    phi_fft_unpadded, out_shape=phi_fft_unpadded.shape
                )

                if not "Lin" in out_path:
                    # do not compute integral for linear sims => it will fail!
                    # phi_fft_unpadded = torch.tensor(phi_fft_unpadded)
                    df = torch.tensor(knth)
                    _, (_, eflux, _) = get_integrals(df, geometry)
                    if not np.isclose(eflux.sum().item(), orig_fluxes[idx], rtol=0., atol=1e-4):
                        warnings.warn(
                            f"Flux integral does not match original flux! Computed: {eflux.sum().item()}, Original: {orig_fluxes[idx]}"
                        )
                    assert np.isclose(eflux.sum().item(), orig_fluxes[idx], rtol=0., atol=1e-2), "Strong deviation for flux!!"

                # append stats to metadata dictionary
                df_stats.update(knth, np.zeros_like(knth), knth, knth)
                flux_stats.update(fluxes[idx], np.zeros_like(fluxes[idx]), fluxes[idx], fluxes[idx])
                phi_stats.update(phi, np.zeros_like(phi), phi, phi)

                # write to disk
                backend.write_df(f, str(idx).zfill(5), df=knth)
                backend.write_phi(f, str(idx).zfill(5), phi=phi)

            # append stats to metadata dictionary
            metadata["df_mean"] = df_stats.mean
            metadata["df_var"] = df_stats.var
            metadata["df_std"] = np.sqrt(df_stats.var)
            metadata["df_min"] = df_stats.min
            metadata["df_max"] = df_stats.max

            metadata["phi_mean"] = phi_stats.mean
            metadata["phi_var"] = phi_stats.var
            metadata["phi_std"] = np.sqrt(phi_stats.var)
            metadata["phi_min"] = phi_stats.min
            metadata["phi_max"] = phi_stats.max

            metadata["flux_mean"] = flux_stats.mean
            metadata["flux_var"] = flux_stats.var
            metadata["flux_std"] = np.sqrt(flux_stats.var)
            metadata["flux_min"] = flux_stats.min
            metadata["flux_max"] = flux_stats.max

            # dump metadata as last operation
            backend.write_metadata(f, metadata)

        return out_path, False

    finally:
        # Free up the terminal row for the next job
        if position_queue is not None:
            position_queue.put(pos)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--tqdm", action="store_true")
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument(
        "--backend", type=str, choices=["hdf5", "kvikio"], default="kvikio"
    )
    parser.add_argument("--target_dir", type=str, default="/local00/bioinf/galletti")
    parser.add_argument(
        "--root", type=str, default="/restricteddata/ukaea/gyrokinetics"
    )
    args = parser.parse_args()

    IFFT = True
    separate_zf = False
    split_into_bands = None

    datasets = ["iteration_0", "iteration_1", "iteration_13"]

    if args.backend == "kvikio":
        backend = KvikIOBackend(use_kvikio=False)
    else:
        backend = H5Backend()

    if not args.debug:
        # Define how many terminal rows we need (one per thread)
        num_threads = min(len(datasets), args.num_workers)
        position_queue = queue.Queue()
        for i in range(1, num_threads + 1):
            position_queue.put(i)

        preprocess_fns = partial(
            preprocess,
            backend=backend,
            spatial_ifft=IFFT,
            separate_zf=separate_zf,
            split_into_bands=split_into_bands,
            root=args.root,
            target_dir=args.target_dir,
            position_queue=position_queue,
        )

        returns = []
        with ThreadPoolExecutor(num_threads) as executor:
            # Main progression bar pinned strictly to the top row (position=0)
            pbar = executor.map(preprocess_fns, datasets)
            if args.tqdm:
                pbar = tqdm(
                    pbar,
                    total=len(datasets),
                    desc="Overall Progress",
                    position=0,
                    leave=True,
                )
            for res in pbar:
                returns.append(res)

        # Cleanly print out the skipped files at the very end to avoid messing up the UI
        skipped_files = [f for f, skipped in returns if skipped]
        if skipped_files:
            print(f"\nSkipped {len(skipped_files)} trajectories (already processed).")

    else:
        for f in datasets:
            out_path, skipped = preprocess(
                f,
                backend=backend,
                spatial_ifft=IFFT,
                separate_zf=separate_zf,
                split_into_bands=split_into_bands,
                root=args.root,
                target_dir=args.target_dir,
                position_queue=None,
            )

            try:
                if isinstance(backend, H5Backend):
                    os.chmod(out_path, 0o777)
            except PermissionError:
                pass

            if skipped:
                print(f"Skipped {f}: already exists.")
            else:
                meta = backend.read_metadata(out_path)
                timesteps = len(meta["timesteps"])
                mean, std = meta["df_mean"][0].mean(), meta["df_std"][0].mean()
                print(
                    f"{out_path}:\n "
                    f"\tpoints: {timesteps}\n"
                    f"\tmean: {mean:2f}, std: {std:2f}\n"
                )
