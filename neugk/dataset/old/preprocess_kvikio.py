import warnings
import os
import pickle
from tqdm import tqdm

import numpy as np
import torch

from neugk.utils import (
    RunningMeanStd,
    pev_flux_df_phi,
    load_geometry,
    K_files,
    poten_files,
    parse_input_dat,
)


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


def phi_to_spc(phi, gt_spc, out_shape, norm: str = "forward"):
    phi_fft = np.fft.fftn(phi, axes=(0, 2), norm=norm)
    phi_fft = np.fft.fftshift(phi_fft, axes=(0, 2))
    phi_fft = phi_fft[..., phi_fft.shape[-1] // 2 :]
    nkx, _, nky = out_shape
    xpad = (phi_fft.shape[0] - nkx) // 2
    xpad = xpad + 1 if (phi_fft.shape[0] % 2 == 0) else xpad
    phi_fft = phi_fft[xpad : nkx + xpad, :, :nky]
    assert _check_spc(np.abs(phi_fft), gt_spc), "Spectral space of Phi incorrect"
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
    root: str,
    filename: str,
    spatial_ifft: bool = False,
    separate_zf: bool = False,
    split_into_bands=None,
    norm_axes=(1, 2, 3, 4, 5),
    debug: bool = False,
    target_dir: str = None,
):
    assert not (
        separate_zf and not spatial_ifft
    ), "Need to perform IFFT to maintain shapes for separate_zf"
    target_dir = root if target_dir is None else target_dir
    #
    dir_in = f"{root}/raw/{filename}"
    dir_out = f"{target_dir}/preprocessed_kvikio"
    os.makedirs(dir_out, exist_ok=True)
    filename = filename.replace("/", "_")

    # create directory names instead of h5 files
    ifft_tag = "_ifft" if spatial_ifft else ""
    zf_tag = "_separate_zf" if separate_zf else ""
    split_into_bands_tag = f"_{split_into_bands}bands" if split_into_bands else ""

    # This is now a DIRECTORY, not a file
    out_dir = f"{dir_out}/{filename}{ifft_tag}{zf_tag}{split_into_bands_tag}_realpotens"
    data_dir = os.path.join(out_dir, "data")
    meta_path = os.path.join(out_dir, "metadata.pkl")

    if os.path.exists(meta_path) and not debug:
        # If metadata exists, assume we already processed this
        return out_dir, True

    os.makedirs(data_dir, exist_ok=True)

    ks = K_files(dir_in.replace("_Lin", ""))
    potens, _ = poten_files(dir_in.replace("_Lin", ""))
    # get timestamps
    ts = []
    for k in ks:
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
    ns = sgrid.shape[1] if len(sgrid.shape) > 1 else sgrid.shape[0]
    nx, ny = xphi.shape[1], xphi.shape[0]
    nkx, nky = krho.shape[1], krho.shape[0]
    nvpar, nmu = vpgr.shape[1], vpgr.shape[0]

    resolution = (nvpar, nmu, ns, nkx, nky)

    fluxes = np.loadtxt(f"{dir_in.replace('_Lin', '')}/fluxes.dat")[:, 1]
    orig_times = np.loadtxt(f"{dir_in.replace('_Lin', '')}/time.dat")
    ts_slices = [np.isclose(orig_times, t).nonzero()[0][0] for t in timesteps]
    fluxes = fluxes[ts_slices]
    orig_fluxes = fluxes.copy()
    fluxes = np.clip(fluxes, a_min=0.0, a_max=None)

    config = parse_input_dat(f"{dir_in}/input.dat")
    ion_temp_grad = config["species"]["rlt"]
    density_grad = config["species"]["rln"]
    s_hat = config["geom"]["shat"]
    q = config["geom"]["q"]

    shape = tuple(
        [
            1 if ax in norm_axes else resolution[ax - 1]
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
        phi_stats = RunningMeanStd((1, 1, 1))
        flux_stats = RunningMeanStd((1,))

    if "Lin" in out_dir:
        # if linear sim, only take last timestep
        ks = [ks[-1]]
        potens = [potens[-1]]

    # TODO
    geometry = load_geometry(dir_in)
    np_geom = {
        k: (geometry[k] if type(geometry[k]) != torch.Tensor else np.array(geometry[k]))
        for k in geometry.keys()
    }

    # Prepare dictionary for our non-tensor metadata
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

    for idx, (k, pot) in tqdm(
        enumerate(zip(ks, potens)),
        f"Processing {filename} -> {out_dir}",
        total=len(ks),
    ):
        with open(f"{dir_in}/{k}", "rb") as fid:
            ff = np.fromfile(fid, dtype=np.float64)

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
                            cur_knth[..., offset:, :] = knth_no_zf[..., offset:, :]
                        else:
                            cur_knth[..., offset : offset + modes_per_channel, :] = (
                                knth_no_zf[..., offset : offset + modes_per_channel, :]
                            )
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
        spc_file = pot.replace("Poten", "Spc3d")
        b = np.loadtxt(f"{dir_in}/{spc_file}")
        gt_spc = np.reshape(b, (nkx, ns, nky), order="F")
        phi_fft_unpadded = phi_to_spc(phi, gt_spc, out_shape=(nkx, ns, nky))
        phi = phi_fft_to_real(phi_fft_unpadded, out_shape=phi_fft_unpadded.shape)

        df = np.moveaxis(orig_knth, 0, -1).copy()
        df = df.view(dtype=np.complex64).squeeze()
        df = torch.tensor(df)

        if "Lin" not in out_dir:
            phi_fft_unpadded = torch.tensor(phi_fft_unpadded)
            _, eflux, _ = pev_flux_df_phi(
                df, phi_fft_unpadded, geometry, aggregate=False
            )

            try:
                assert np.isclose(
                    eflux.sum().item(), orig_fluxes[idx], rtol=0.0, atol=1e-4
                ), "Flux integral failed..."
            except:
                warnings.warn("Flux integral failed...")

        # update running averages
        df_stats.update(
            np.mean(knth, axis=norm_axes, keepdims=True),
            np.var(knth, axis=norm_axes, keepdims=True),
            np.min(knth, axis=norm_axes, keepdims=True),
            np.max(knth, axis=norm_axes, keepdims=True),
        )
        flux_stats.update(fluxes[idx], fluxes[idx], fluxes[idx], fluxes[idx])
        phi_stats.update(
            np.mean(phi, axis=(0, 1, 2), keepdims=True),
            np.var(phi, axis=(0, 1, 2), keepdims=True),
            np.min(phi, axis=(0, 1, 2), keepdims=True),
            np.max(phi, axis=(0, 1, 2), keepdims=True),
        )

        # WRITE RAW BINARIES
        # Use ascontiguousarray to ensure memory is contiguous before dumping
        k_name = os.path.join(data_dir, "timestep_" + str(idx).zfill(5) + ".bin")
        if not os.path.exists(k_name):
            np.ascontiguousarray(knth).tofile(k_name)

        poten_name = os.path.join(data_dir, "poten_" + str(idx).zfill(5) + ".bin")
        if not os.path.exists(poten_name):
            np.ascontiguousarray(phi).tofile(poten_name)

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

    # dump metadata via pickle
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)

    return out_dir, False


if __name__ == "__main__":
    originals = [
        "iteration_13",
        "iteration_58",
        "iteration_70",
        "iteration_54",
        "iteration_75",
        "iteration_36",
        "iteration_49",
        "iteration_84",
        "iteration_51",
        "iteration_7",
        "iteration_74",
        "iteration_55",
        "iteration_97",
        "iteration_43",
        "iteration_33",
        "iteration_93",
        "iteration_24",
        "iteration_60",
        "iteration_47",
        "iteration_62",
        "iteration_46",
        "iteration_39",
        "iteration_8",
        "iteration_20",
        "iteration_48",
        "iteration_98",
        "iteration_80",
        "iteration_73",
        "iteration_23",
        "iteration_40",
        "iteration_41",
        "iteration_76",
        "iteration_15",
        "iteration_37",
        "iteration_85",
        "iteration_14",
        "iteration_79",
        "iteration_94",
        "iteration_71",
        "iteration_86",
        "iteration_90",
        "iteration_96",
        "iteration_88",
        "iteration_0",
        "iteration_2",
        "iteration_82",
        "iteration_21",
        "iteration_57",
        "iteration_95",
        "iteration_29",
    ]

    for f in originals:
        out_dir, skipped = preprocess(
            root="/restricteddata/ukaea/gyrokinetics/",
            filename=f,
            spatial_ifft=True,
            separate_zf=False,
            norm_axes=(1, 2, 3, 4, 5),
            target_dir="/local00/bioinf/galletti",
        )
        if skipped:
            print(f"skipped {f}: already exists.")

        if not skipped:
            meta_path = os.path.join(out_dir, "metadata.pkl")
            data_dir = os.path.join(out_dir, "data")

            with open(meta_path, "rb") as mf:
                meta = pickle.load(mf)

            timesteps = len(os.listdir(data_dir)) // 2

            ts0_path = os.path.join(data_dir, "timestep_00000.bin")
            timestep_0_bytes = np.fromfile(ts0_path, dtype=np.float32)

            mean, std = meta["df_mean"][0].mean(), meta["df_std"][0].mean()
            print(
                f"{out_dir}:\n "
                f"\tpoints: {timesteps}, raw float32 size of timestep_00000.bin: {timestep_0_bytes.size}\n"
                f"\tmean: {mean:2f}, std: {std:2f}\n"
            )
