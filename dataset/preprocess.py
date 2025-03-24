import os
import numpy as np
import h5py
import re
from tqdm import tqdm

from utils import RunningMeanStd

ROOT = "/restricteddata/ukaea/gyrokinetics"

def K_files(directory):
    files = os.listdir(directory)
    digit_files = sorted(
        [file for file in files if file.isdigit()], key=lambda x: int(x)
    )
    k_files = sorted(
        [file for file in files if file.startswith("K") and not file.endswith(".dat")]
    )
    return k_files + digit_files


def poten_files(directory):
    files = os.listdir(directory)
    poten_files = sorted([file for file in files if file.startswith("Poten")])
    timestep_slices = [int(f.replace("Poten", "")) for f in poten_files]
    return poten_files, np.array(timestep_slices) - 1

def parse_input_dat(file_path):
    parsed_data = {}
    with open(file_path, "r") as file:
        content = file.read()
    # split the content by section headers (e.g., &SPECIES, &SPCGENERAL, etc.)
    sections = re.split(r"&\w+", content)
    # get all the headers by finding the section names
    section_headers = re.findall(r"&(\w+)", content)
    # remove comments
    sections = [
        section.strip() for section in sections if section[0] != "!" and section.strip()
    ]
    for header, section in zip(section_headers, sections):
        section_dict = {}
        params = re.findall(r"(\w+)\s*=\s*([-\d\.e\w]+)", section)
        for param, value in params:
            try:
                section_dict[param] = (
                    float(value) if "e" in value or "." in value else int(value)
                )
            except ValueError:
                section_dict[param] = value.strip()
        while header in parsed_data:
            header = f"{header}0"
        parsed_data[header] = section_dict

    return parsed_data

def do_ifft(knth):
    knth = np.fft.ifftn(knth, axes=(3, 4))
    knth = np.stack([knth.real, knth.imag]).squeeze().astype("float32")
    return knth

def get_stats(filenames, spatial_ifft=False, separate_zf=False, per_mode_norm=False):
    running_stats = None
    old_running_stats = None
    for filename in filenames:
        dir_in = f"{ROOT}/raw/{filename}"
        dir_out = "/local00/bioinf/gyrokinetics/preprocessed"
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

def check_ifft(transformed, orig):
    real_parts = transformed[::2]
    imag_parts = transformed[1::2]
    sum_real = np.sum(real_parts, axis=0)
    sum_imag = np.sum(imag_parts, axis=0)
    orig_ifft = np.concatenate([np.expand_dims(sum_real, 0), np.expand_dims(sum_imag, 0)], axis=0)
    orig_ifft = np.moveaxis(orig_ifft, 0, -1).copy()
    orig_ifft = orig_ifft.view(dtype=np.complex64)
    orig_ifft = np.fft.fftn(orig_ifft, axes=(3, 4))
    orig_ifft = np.fft.ifftshift(orig_ifft, axes=(3,))
    orig_ifft = np.stack([orig_ifft.real, orig_ifft.imag]).squeeze().astype("float32")
    return np.allclose(orig_ifft, orig, rtol=0, atol=1e-5)

def preprocess(filename, spatial_ifft=False, separate_zf=False, stats=None, per_mode_norm=False,
               split_into_bands=None, norm_axes=(1, 2, 3, 4, 5)):
    assert not (separate_zf and not spatial_ifft), "Need to perform IFFT to maintain shapes for separate_zf"
    dir_in = f"{ROOT}/raw/{filename}"
    dir_out = f"{ROOT}/preprocessed"
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    # create h5 file with timestamps and field data
    ifft_tag = "_ifft" if spatial_ifft else ""
    zf_tag = "_separate_zf" if separate_zf else ""
    per_mode_tag = "_per_mode" if per_mode_norm else ""
    split_into_bands_tag = f"_{split_into_bands}bands" if split_into_bands else ""
    # axes_desc = ["nvpar", "nmu", "ns", "nkx", "nky"]
    # norm_axes_tag = f"_norm_{"_".join([axes_desc[ax-1] for ax in norm_axes])}"
    h5_filename = f"{dir_out}/{filename}{ifft_tag}{zf_tag}{per_mode_tag}{split_into_bands_tag}.h5"
    if os.path.exists(h5_filename):
        if per_mode_norm:
            # update stats with new stats
            with h5py.File(h5_filename, "r+") as file:
                file["metadata"]["k_mean"] = stats.mean
                file["metadata"]["k_var"] = np.sqrt(stats.var)
                file["metadata"]["k_min"] = stats.min
                file["metadata"]["k_max"] = stats.max

        print(f"File {h5_filename} already exists, skipping...")
        return h5_filename, True

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

    # load fluxes
    fluxes = np.loadtxt(f"{dir_in}/fluxes.dat")[:, 1]
    print(ks)
    try:
        fluxes = fluxes[ts_slices]
    except IndexError:
        print("Mismatch between Poten ts and K ts")
        orig_times = np.loadtxt(f"{dir_in}/time.dat")
        ts_slices = [np.isclose(orig_times, t).nonzero()[0][0] for t in timesteps]
        fluxes = fluxes[ts_slices]

    # load parameters
    config = parse_input_dat(f"{dir_in}/input.dat")
    ion_temp_grad = config["SPECIES"]["rlt"]
    if not per_mode_norm:
        shape = tuple([1 if ax in norm_axes else resolution[ax-1] for ax in np.arange(1,len(resolution)+1)])
        if zf_tag:
            if split_into_bands:
                stats = RunningMeanStd(shape=((split_into_bands+1)*2,)+shape)
            else:
                stats = RunningMeanStd(shape=(4,)+shape)
        else:
            stats = RunningMeanStd(shape=(2,)+shape)

    with (h5py.File(h5_filename, "w") as file):

        # group for metadata (e.g. timesteps)
        metadata_group = file.create_group("metadata")
        metadata_group.create_dataset("timesteps", data=timesteps)
        metadata_group.create_dataset("fluxes", data=fluxes)
        metadata_group.create_dataset("resolution", data=resolution)
        metadata_group.create_dataset("ion_temp_grad", data=ion_temp_grad, shape=(1,))

        # group for our 6D field data
        data_group = file.create_group("data")
        for idx, (k, pot) in tqdm(
            enumerate(zip(ks, potens)),
            f"Processing {filename} -> {filename + ifft_tag + zf_tag + per_mode_tag + split_into_bands_tag + ".h5"}",
            total=len(ks),
        ):
            # Load the full distribution function data
            with open(f"{dir_in}/{k}", "rb") as fid:
                ff = np.fromfile(fid, dtype=np.float64)

            # Reshape the distribution function (copy for speeed in stat computation)
            knth = np.reshape(ff, (2, *resolution), order="F").astype("float32").copy()
            orig_knth = knth.copy()
            if spatial_ifft:
                if per_mode_norm:
                    knth = (knth - stats.mean) / np.sqrt(stats.var)
                knth = np.moveaxis(knth, 0, -1).copy()
                knth = knth.view(dtype=np.complex64)
                knth = np.fft.fftshift(knth, axes=(3,))
                separated_modes = []
                if separate_zf:
                    knth_zf = knth.copy()
                    knth_no_zf = knth.copy()
                    knth_zf[..., 1:, :] = 0.
                    ifft_knth_zf = do_ifft(knth_zf)
                    separated_modes.append(ifft_knth_zf)
                    knth_no_zf[..., 0, :] = 0.
                    if split_into_bands:
                        modes_per_channel = nky // split_into_bands
                        for band in range(split_into_bands):
                            cur_knth = np.zeros_like(knth_no_zf)
                            offset = 1 + band * modes_per_channel
                            if (split_into_bands - 1) == band:
                                # last band contains all remaining frequencies
                                cur_knth[..., offset:, :] = knth_no_zf[..., offset:, :]
                            else:
                                cur_knth[..., offset:offset+modes_per_channel, :] = knth_no_zf[..., offset:offset+modes_per_channel, :]
                            ifft_knth = do_ifft(cur_knth)
                            separated_modes.append(ifft_knth)
                    else:
                        ifft_knth_no_zf = do_ifft(knth_no_zf)
                        separated_modes.append(ifft_knth_no_zf)

                    knth = np.concatenate(separated_modes, axis=0)
                    assert check_ifft(knth.copy(), orig_knth), "Error transforming back to original space"
                else:
                    knth = do_ifft(knth)

            if not per_mode_norm:
                # update running averages
                stats.update(np.mean(knth, axis=norm_axes, keepdims=True),
                             np.var(knth, axis=norm_axes, keepdims=True),
                             np.min(knth, axis=norm_axes, keepdims=True),
                             np.max(knth, axis=norm_axes, keepdims=True))

            # Add the reshaped data as a dataset to the "data" group
            k_name = "timestep_" + str(idx).zfill(5)
            data_group.create_dataset(k_name, data=knth)

            # load the potential field
            a = np.loadtxt(f"{dir_in}/{pot}")
            phi = np.reshape(a, (nx, ns, ny), order="F").astype("float32").copy()
            poten_name = "poten_" + str(idx).zfill(5)
            data_group.create_dataset(poten_name, data=phi)

        metadata_group.create_dataset("k_mean", data=stats.mean)
        metadata_group.create_dataset("k_std", data=np.sqrt(stats.var))
        metadata_group.create_dataset("k_min", data=stats.min)
        metadata_group.create_dataset("k_max", data=stats.max)

        return h5_filename, False

IFFT = True
separate_zf = False
per_mode_norm = False
split_into_bands = None
norm_axes = (1,2,3,4,5)
ifft_tag = "_ifft" if IFFT else ""
zf_tag = "_separate_zf" if separate_zf else ""
per_mode_tag = "_per_mode" if per_mode_norm else ""
split_into_bands_tag = f"_{split_into_bands}bands" if split_into_bands else ""
# axes_desc = ["nvpar", "nmu", "ns", "nkx", "nky"]
# norm_axes_tag = f"_norm_{"_".join([axes_desc[ax-1] for ax in norm_axes])}"
datasets = [
    "cyclone17_2",
    "cyclone4_2_2",
    "cyclone20_2",
    "cyclone5_2_diffInit",
    "cyclone5_2_diffInit3",
    "cyclone8_2",
    "cyclone10_2",
    "cyclone8_2_diffInit3",
    "cyclone8_2_diffInit2",
    "cyclone9_2",
    "cyclone5_2",
    "cyclone18_2",
    "cyclone9_2_diffInit",
    "cyclone12_2_diffInit",
    "cyclone15_2",
    "cyclone14_2",
    "cyclone22_2_diffInit3",
    "cyclone9_2_diffInit3",
    "cyclone12_2_diffInit3",
    "cyclone20_2_diffInit",
    "cyclone5_2_diffInit2",
    "cyclone9_2_diffInit2",
    "cyclone8_2_diffInit",
    "cyclone6_2",
    "cyclone20_2_diffInit2",
    "cyclone22_2_diffInit2",
    "cyclone13_2",
    "cyclone16_2",
    "cyclone7_2",
    "cyclone11_2",
    "cyclone12_2",
    "cyclone20_2_diffInit3",
    "cyclone21_2",
    "cyclone12_2_diffInit2",
    "cyclone22_2",
    "cyclone22_2_diffInit",
    "cyclone19_2"
]


stats = get_stats(datasets, IFFT, separate_zf, per_mode_norm) if per_mode_norm else None
for f in datasets:
    h5_filename, skipped = preprocess(f, spatial_ifft=IFFT, separate_zf=separate_zf, stats=stats, per_mode_norm=per_mode_norm,
                             split_into_bands=split_into_bands, norm_axes=norm_axes)
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
            mean, std = h5f["metadata/k_mean"][0], h5f["metadata/k_std"][0]
            min_, max_ = h5f["metadata/k_min"][0], h5f["metadata/k_max"][0]
            print(
                f"{h5_filename}:\n "
                f"\tpoints: {timesteps}, shape of timestep_00000: {timestep_0.shape}\n"
                f"\trlt: {rlt}\n"
            )


# for filename in datasets:
#     h5_filename = f"{filename}{ifft_tag}{zf_tag}{per_mode_tag}{split_into_bands_tag}.h5"
#     os.system(f"rsync -ah --info=progress {ROOT}/preprocessed/{h5_filename} /local00/bioinf/gyrokinetics/preprocessed/{h5_filename}")