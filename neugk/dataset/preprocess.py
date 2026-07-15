import os
import queue
import sys
import time
import warnings

from tqdm import tqdm
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from neugk.utils import (
    RunningMeanStd,
    load_geometry,
    K_files,
    poten_files,
    parse_input_dat,
)
from neugk.physics.integrals import get_integrals

from neugk.dataset.backend import H5Backend, KvikIOBackend, DataBackend


# bf16 sibling conversion: mirrors JAX port (neugk_jax/dataset/preprocess.py); for an existing fp32 .bin shard, write a side-by-side .bf16.bin sibling with raw bfloat16 values (no header/scale; dtype encodes magnitude); dataloader reads these in place of fp32, falls back silently if absent

_BF16_SUFFIX = ".bf16.bin"


def quantized_sibling(fp32_path: str, bits: str = "bf16") -> str:
    """``foo.bin`` -> ``foo.bf16.bin`` (the side-by-side quantized shard)."""
    assert bits == "bf16", f"only bf16 supported here, got {bits!r}"
    if fp32_path.endswith(".bin"):
        return fp32_path[:-4] + _BF16_SUFFIX
    return fp32_path + _BF16_SUFFIX


def quantize_array(arr_f32: np.ndarray, bits: str = "bf16") -> np.ndarray:
    """Quantize a flat fp32 array to ``bits`` precision (bf16 only)."""
    assert bits == "bf16", f"only bf16 supported here, got {bits!r}"
    from ml_dtypes import bfloat16

    return arr_f32.astype(bfloat16)


def write_quantized(dst: str, payload: np.ndarray) -> int:
    """Atomic-ish write of one quantized shard. Returns bytes written."""
    tmp = dst + ".tmp"
    with open(tmp, "wb") as f:
        f.write(payload.tobytes())
    n = os.path.getsize(tmp)
    os.replace(tmp, dst)
    return n


def _src_bins(data_dir: str) -> list:
    """List fp32 .bin sources (timestep + poten) inside ``traj/data``."""
    if not os.path.isdir(data_dir):
        return []
    out = []
    for name in os.listdir(data_dir):
        if not name.endswith(".bin"):
            continue
        if name.endswith(_BF16_SUFFIX):
            continue
        if not (name.startswith("timestep_") or name.startswith("poten_")):
            continue
        out.append(os.path.join(data_dir, name))
    return sorted(out)


def _quantize_file(src: str, force: bool) -> tuple:
    dst = quantized_sibling(src, "bf16")
    if os.path.exists(dst) and not force:
        return src, 0, "skip"
    try:
        arr = np.fromfile(src, dtype=np.float32)
        payload = quantize_array(arr, "bf16")
        n = write_quantized(dst, payload)
        return src, n, "written"
    except Exception as e:  # noqa: BLE001
        return src, 0, f"error: {e}"


def _process_traj_bf16(traj_dir: str, force: bool) -> tuple:
    files = _src_bins(os.path.join(traj_dir, "data"))
    n_written = n_skipped = bytes_written = 0
    for src in files:
        _, n, status = _quantize_file(src, force)
        if status == "written":
            n_written += 1
            bytes_written += n
        elif status == "skip":
            n_skipped += 1
        else:
            print(f"  [{traj_dir}] {os.path.basename(src)}: {status}", file=sys.stderr)
    return traj_dir, n_written, n_skipped, bytes_written


def convert_trajs_to_bf16(
    traj_dirs: Sequence[str], *, num_workers: int = 4, force: bool = False
) -> None:
    """Write ``.bf16.bin`` siblings for an explicit list of trajectory dirs.

    Idempotent: skips files whose bf16 sibling already exists (unless
    ``force``). The fp32 originals are never touched.
    """
    traj_dirs = [d for d in traj_dirs if os.path.isdir(d)]
    if not traj_dirs:
        print("no trajectory dirs matched")
        sys.exit(1)
    print(f"converting {len(traj_dirs)} trajectories to bf16 siblings")
    t0 = time.perf_counter()
    total_w = total_s = total_b = 0
    with ThreadPoolExecutor(max_workers=max(1, num_workers)) as ex:
        futures = {ex.submit(_process_traj_bf16, d, force): d for d in traj_dirs}
        for i, fut in enumerate(as_completed(futures), 1):
            d, nw, ns, bw = fut.result()
            total_w += nw
            total_s += ns
            total_b += bw
            elapsed = time.perf_counter() - t0
            rate = total_b / max(elapsed, 1e-6) / 1e9
            print(
                f"  [{i}/{len(traj_dirs)}] {Path(d).name:<40}  "
                f"written={nw:4d}  skip={ns:4d}  bytes={bw / 1e9:6.2f} GB  "
                f"rate={rate:5.2f} GB/s",
                flush=True,
            )
    elapsed = time.perf_counter() - t0
    print(
        f"\ndone -- {total_w} files written, {total_s} skipped, "
        f"{total_b / 1e9:.2f} GB in {elapsed:.0f}s "
        f"({total_b / max(elapsed, 1e-6) / 1e9:.2f} GB/s)"
    )


def do_ifft(knth):
    # inverse FFT along kx, ky; extract real/imag channels
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
    raw_subdir: str = "raw",
    target_dir: str = "/local00/bioinf/galletti",
    position_queue: queue.Queue = None,
    metadata_only: bool = False,
    geometry_only: bool = False,
):
    # Grab a dedicated row for this worker's progress bar (default to 0 if single-threaded)
    pos = position_queue.get() if position_queue is not None else 0

    try:
        assert not (
            separate_zf and not spatial_ifft
        ), "need to perform IFFT to maintain shapes for separate_zf"

        target_dir = root if target_dir is None else target_dir
        dir_in = f"{root}/{raw_subdir}/{filename}"

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

        if backend.exists(out_path) and not (metadata_only or geometry_only):
            return out_path, True

        ks = K_files(dir_in.replace("_Lin", ""))
        potens, _ = poten_files(dir_in.replace("_Lin", ""))
        # k_dir = dir_in.replace("_Lin", "")
        if not len(ks):
            # load k dump files from other sim (sampled the same way); extract correct flux timesteps
            ks = K_files("/restricteddata/ukaea/gyrokinetics/raw/iteration_0")
            potens, _ = poten_files(
                "/restricteddata/ukaea/gyrokinetics/raw/iteration_0"
            )
            # k_dir = "/restricteddata/ukaea/gyrokinetics/raw/iteration_0"
        # extract timestamps
        ts = []
        for k in ks:
            # load timestep
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
        # parallel direction grid points
        ns = sgrid.shape[1] if len(sgrid.shape) > 1 else sgrid.shape[0]
        # x, y grid points (in real space)
        nx, ny = xphi.shape[1], xphi.shape[0]
        # modes in x and y direction
        nkx, nky = krho.shape[1], krho.shape[0]
        # velocity space resolutions
        nvpar, nmu = vpgr.shape[1], vpgr.shape[0]

        resolution = (nvpar, nmu, ns, nkx, nky)

        # load nonlinear fluxes
        fluxes = np.loadtxt(f"{dir_in.replace('_Lin', '')}/fluxes.dat")[:, 1]
        orig_fluxes = fluxes.copy()
        if "Lin" not in out_path:
            # extract timesteps matching nonlinear times
            orig_times = np.loadtxt(f"{dir_in.replace('_Lin', '')}/time.dat")
            ts_slices = [np.isclose(orig_times, t).nonzero()[0][0] for t in timesteps]
            fluxes = fluxes[ts_slices]
            orig_fluxes = fluxes.copy()

        # clip negative fluxes
        fluxes = np.clip(fluxes, a_min=0.0, a_max=None)
        # load parameters
        config = parse_input_dat(f"{dir_in}/input.dat")
        ion_temp_grad = config["species"]["rlt"]
        density_grad = config["species"]["rln"]
        s_hat = config["geom"]["shat"]
        q = config["geom"]["q"]

        geometry = load_geometry(dir_in)
        np_geom = {
            k: (geometry[k].numpy() if hasattr(geometry[k], "numpy") else geometry[k])
            for k in geometry.keys()
        }

        kyspec = np.loadtxt(f"{dir_in}/kyspec")[ts_slices]
        fluxspec = np.loadtxt(f"{dir_in}/eflux_spectra.dat")[ts_slices]

        metadata = {
            "timesteps": timesteps,
            "resolution": resolution,
            "ds": float(
                np.ravel(sgrid)[1] - np.ravel(sgrid)[0]
            ),  # parallel-grid spacing
            "ion_temp_grad": np.array([ion_temp_grad]),
            "density_grad": np.array([density_grad]),
            "flux": fluxes,
            "s_hat": np.array([s_hat]),
            "q": np.array([q]),
            "geometry": np_geom,
            "kyspec": kyspec,
            "fluxspec": fluxspec,
        }

        if geometry_only:
            if backend.exists(out_path):
                # load existing metadata to preserve stats if they exist
                old_metadata = backend.read_metadata(out_path)
                stats_keys = [
                    "df_mean",
                    "df_var",
                    "df_std",
                    "df_min",
                    "df_max",
                    "phi_mean",
                    "phi_var",
                    "phi_std",
                    "phi_min",
                    "phi_max",
                    "flux_mean",
                    "flux_var",
                    "flux_std",
                    "flux_min",
                    "flux_max",
                ]
                for k in stats_keys:
                    if k in old_metadata:
                        metadata[k] = old_metadata[k]

            with backend.create(out_path) as f:
                backend.write_metadata(f, metadata)
            return out_path, False

        df_stats = RunningMeanStd()
        phi_stats = RunningMeanStd()
        flux_stats = RunningMeanStd()

        if "Lin" in out_path:
            # linear sim: only take last timestep
            ks = ["FDS"]
            potens = [potens[-1]]
            # kyspec = np.loadtxt(dir_in.replace("_Lin", "/kyspec"))
            # growth_rate = np.loadtxt(os.path.join(dir_in, "growth.dat"))[-1, :]
            # ky_frequencies = np.loadtxt(os.path.join(dir_in, "frequencies.dat"))[-1, :]

        with backend.create(out_path) as f:
            innter_pbar = zip(ks, potens)
            if args.tqdm:
                innter_pbar = tqdm(
                    innter_pbar, desc=filename, total=len(ks), position=pos, leave=False
                )

            for idx, (k, pot) in enumerate(innter_pbar):
                # load distribution function
                with open(f"{dir_in}/{k}", "rb") as fid:
                    ff = np.fromfile(fid, dtype=np.float64)

                knth = (
                    np.reshape(ff, (2, *resolution), order="F").astype("float32").copy()
                )
                orig_knth = knth.copy()

                if spatial_ifft:
                    # move channels axis to end, view as complex, shift FFT
                    knth = np.moveaxis(knth, 0, -1).copy()
                    knth = knth.view(dtype=np.complex64)
                    knth = np.fft.fftshift(knth, axes=(3,))
                    separated_modes = []
                    if separate_zf:
                        # separate zero-flow (ky=0) from turbulent modes
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
                    ), "error transforming back to original space"

                # load potential field
                a = np.loadtxt(f"{dir_in}/{pot}")
                phi = np.reshape(a, (nx, ns, ny), order="F").astype("float32").copy()
                if "Lin" not in out_path:
                    spc_file = pot.replace("Poten", "Spc3d")
                    b = np.loadtxt(f"{dir_in}/{spc_file}")
                    gt_spc = np.reshape(b, (nkx, ns, nky), order="F")
                else:
                    gt_spc = None
                phi_fft_unpadded = phi_to_spc(phi, gt_spc, out_shape=(nkx, ns, nky))
                phi = phi_fft_to_real(
                    phi_fft_unpadded, out_shape=phi_fft_unpadded.shape
                )

                if "Lin" not in out_path:
                    # skip integral for linear sims (would fail)
                    df = torch.tensor(knth)
                    _, (_, eflux, _) = get_integrals(df, geometry)
                    if not np.isclose(
                        eflux.sum().item(), orig_fluxes[idx], rtol=0.0, atol=1e-2
                    ):
                        warnings.warn(
                            "Flux integral does not match original flux! "
                            f"Computed: {eflux.sum().item()}, Original: {orig_fluxes[idx]}"
                        )
                    assert np.isclose(
                        eflux.sum().item(), orig_fluxes[idx], rtol=0.0, atol=1.0
                    ), "strong deviation for flux"

                # accumulate statistics
                df_stats.update(knth, np.zeros_like(knth), knth, knth)
                flux_stats.update(
                    fluxes[idx], np.zeros_like(fluxes[idx]), fluxes[idx], fluxes[idx]
                )
                phi_stats.update(phi, np.zeros_like(phi), phi, phi)

                # write to disk
                if not metadata_only:
                    backend.write_df(f, str(idx).zfill(5), df=knth)
                    backend.write_phi(f, str(idx).zfill(5), phi=phi)

            # add stats to metadata
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

            # write metadata as final step
            backend.write_metadata(f, metadata)

        return out_path, False
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return out_path, False
    finally:
        # free up terminal row for next job
        if position_queue is not None:
            position_queue.put(pos)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--tqdm", action="store_true")
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument(
        "--metadata_only",
        action="store_true",
        help="Only update metadata.pkl and stats without writing field data.",
    )
    parser.add_argument(
        "--geometry_only",
        action="store_true",
        help="Only update geometry in metadata without processing field data or stats.",
    )
    parser.add_argument(
        "--backend", type=str, choices=["hdf5", "kvikio"], default="kvikio"
    )
    parser.add_argument("--target_dir", type=str, default="/local00/bioinf/galletti")
    parser.add_argument(
        "--root", type=str, default="/restricteddata/ukaea/gyrokinetics"
    )
    parser.add_argument(
        "--raw_subdir",
        type=str,
        default="raw",
        help="Subdirectory under root containing the raw simulation folders.",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=300,
        help="Number of iterations to process (iteration_0 .. iteration_N-1).",
    )
    parser.add_argument(
        "--to_bf16",
        action="store_true",
        help="Skip the full raw->fp32 pipeline; instead write .bf16.bin siblings "
        "for already-preprocessed trajectory dirs (see --bf16_trajs). Idempotent.",
    )
    parser.add_argument(
        "--bf16_trajs",
        type=str,
        nargs="+",
        default=None,
        help="Trajectory dir basenames (or a single brace pattern like "
        "'iteration_{0-59}_ifft_realpotens') under --target_dir/preprocessed_kvikio "
        "to convert when --to_bf16 is set. Defaults to all *_ifft_realpotens dirs.",
    )
    parser.add_argument(
        "--bf16_force",
        action="store_true",
        help="Overwrite existing .bf16.bin siblings.",
    )
    args = parser.parse_args()

    # bf16 sibling conversion path (does not touch fp32 originals)
    if args.to_bf16:
        root_dir = os.path.join(args.target_dir, "preprocessed_kvikio")

        def _resolve_bf16_trajs(spec):
            import re as _re

            if spec is None:
                return sorted(
                    os.path.join(root_dir, n)
                    for n in os.listdir(root_dir)
                    if n.endswith("_ifft_realpotens")
                    and os.path.isdir(os.path.join(root_dir, n))
                )
            if isinstance(spec, list) and len(spec) != 1:
                return [os.path.join(root_dir, n) for n in spec]
            s = spec[0] if isinstance(spec, list) else spec
            m = _re.match(r"^(.*?)\{([^}]+)\}(.*?)$", s)
            if not m:
                return [os.path.join(root_dir, s)]
            prefix, ranges_str, suffix = m.groups()
            nums = []
            for part in ranges_str.split(","):
                if "-" in part:
                    lo, hi = map(int, part.split("-"))
                    nums.extend(range(lo, hi + 1))
                else:
                    nums.append(int(part))
            return [os.path.join(root_dir, f"{prefix}{n}{suffix}") for n in nums]

        traj_dirs = _resolve_bf16_trajs(args.bf16_trajs)
        convert_trajs_to_bf16(
            traj_dirs, num_workers=args.num_workers, force=args.bf16_force
        )
        sys.exit(0)

    IFFT = True
    separate_zf = False
    split_into_bands = None

    datasets = [f"iteration_{i}" for i in range(args.num_iterations)]

    if args.backend == "kvikio":
        backend = KvikIOBackend(use_kvikio=False)
    else:
        backend = H5Backend()

    if not args.debug:
        # allocate terminal rows for parallel workers
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
            raw_subdir=args.raw_subdir,
            target_dir=args.target_dir,
            position_queue=position_queue,
            metadata_only=args.metadata_only,
            geometry_only=args.geometry_only,
        )

        returns = []
        with ThreadPoolExecutor(num_threads) as executor:
            # main progress bar pinned to top row (position=0)
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

        # print skipped files at end to avoid UI clutter
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
                raw_subdir=args.raw_subdir,
                target_dir=args.target_dir,
                position_queue=None,
                metadata_only=args.metadata_only,
                geometry_only=args.geometry_only,
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
