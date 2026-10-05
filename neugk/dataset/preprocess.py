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


def read_k_spectral(path: str, resolution) -> tuple:
    """A raw GKW K dump as float32 ``(2, *resolution)`` and as the ky-shifted complex64 spectrum."""
    with open(path, "rb") as fid:
        ff = np.fromfile(fid, dtype=np.float64)
    knth = np.reshape(ff, (2, *resolution), order="F").astype("float32").copy()
    spec = np.moveaxis(knth, 0, -1).copy().view(dtype=np.complex64)
    return knth, np.fft.fftshift(spec, axes=(3,))


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
    # ifftshift inverts the kx-centring of phi_to_spc (fftshift is off by one bin for odd nkx)
    phi = np.fft.ifftshift(padded, axes=(0,))
    phi_ifft = np.fft.irfftn(phi, axes=(0, 2), norm=norm, s=[nkx, nky])
    return phi_ifft


def field_solve_phi(df: np.ndarray, geometry) -> np.ndarray:
    """Real-space potential (x, s, y) solved from a real-space df, as in the evaluation."""
    from neugk.physics.integrals import FluxIntegral

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    geom = {k: torch.as_tensor(g).unsqueeze(0).to(dev) for k, g in geometry.items()}
    phi, _ = FluxIntegral(real_potens=True, flux_fields=True)(geom, torch.as_tensor(df).unsqueeze(0).to(dev))
    return phi.squeeze(0).cpu().numpy().astype(np.float32)


def resolve_traj_dirs(root_dir, spec):
    """Trajectory dirs under root_dir from basenames or one brace pattern (default: all *_ifft_realpotens)."""
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


def rewrite_poten(traj_dir: str, backup_dir: str) -> str:
    """Overwrite poten_*.bin of a preprocessed kvikio trajectory with the field solve of its df.

    The originals and both metadata files are copied to backup_dir/<name> first; the per-trajectory
    phi statistics are recomputed. Resumable: trajectories with a DONE marker are skipped.
    """
    import glob
    import pickle
    import shutil

    torch.set_num_threads(2)
    name = os.path.basename(traj_dir.rstrip("/"))
    bdir = os.path.join(backup_dir, name)
    if os.path.exists(os.path.join(bdir, "DONE")):
        return f"{name}: skip (done)"
    os.makedirs(bdir, exist_ok=True)
    data = os.path.join(traj_dir, "data")
    potens = sorted(glob.glob(os.path.join(data, "poten_*.bin")))
    for p in potens + [os.path.join(traj_dir, m) for m in ("metadata.pkl", "metadata_light.pkl")]:
        b = os.path.join(bdir, os.path.basename(p))
        if os.path.exists(p) and not os.path.exists(b):
            shutil.copy2(p, b)

    from neugk.pinc.neural_fields.data import CycloneNFDataset

    t0 = int(os.path.basename(potens[0])[6:11])
    nf = CycloneNFDataset(name.replace("_ifft_realpotens", ""), timesteps=[t0], normalize=None,
                          path=os.path.dirname(traj_dir.rstrip("/")), backend="kvikio", realpotens=True)
    shape = tuple(nf.full_df.shape[-6:]) if nf.full_df.ndim > 6 else tuple(nf.full_df.shape)
    stats = RunningMeanStd()
    for p in potens:
        idx = os.path.basename(p)[6:11]
        df = np.fromfile(os.path.join(data, f"timestep_{idx}.bin"), dtype=np.float32).reshape(shape)
        with torch.no_grad():
            phi = field_solve_phi(df, nf.geom)
        assert phi.nbytes == os.path.getsize(p), (p, phi.shape)
        phi.tofile(p + ".tmp")
        os.replace(p + ".tmp", p)
        stats.update(phi, np.zeros_like(phi), phi, phi)

    new = {"phi_mean": stats.mean, "phi_var": stats.var, "phi_std": np.sqrt(stats.var),
           "phi_min": stats.min, "phi_max": stats.max}
    for m in ("metadata.pkl", "metadata_light.pkl"):
        mp = os.path.join(traj_dir, m)
        if os.path.exists(mp):
            meta = pickle.load(open(mp, "rb"))
            for k in [k for k in new if k in meta]:
                meta[k] = np.asarray(new[k], dtype=np.asarray(meta[k]).dtype)
            with open(mp + ".tmp", "wb") as f:
                pickle.dump(meta, f)
            os.replace(mp + ".tmp", mp)
    open(os.path.join(bdir, "DONE"), "w").write(f"{len(potens)}\n")
    return f"{name}: rewrote {len(potens)} potentials"


def restore_fp32(
    traj_dir: str, out_root: str, timesteps: Sequence[int], raw_root: str
) -> str:
    """Rebuild the fp32 df shards of a bf16-only trajectory at ``timesteps`` under ``out_root``.

    Each shard is redone from the raw GKW K dump with the preprocess transform and must round to
    the stored bf16 sibling bit for bit; metadata and potentials are symlinked from ``traj_dir``.
    Resumable: existing shards are skipped.
    """
    import pickle

    from ml_dtypes import bfloat16

    name = os.path.basename(traj_dir.rstrip("/"))
    raw = os.path.join(raw_root, name.replace("_ifft_realpotens", ""))
    out = os.path.join(out_root, name)
    os.makedirs(os.path.join(out, "data"), exist_ok=True)
    for m in ("metadata.pkl", "metadata_light.pkl"):
        if os.path.exists(os.path.join(traj_dir, m)) and not os.path.lexists(os.path.join(out, m)):
            os.symlink(os.path.join(traj_dir, m), os.path.join(out, m))
    meta = pickle.load(open(os.path.join(traj_dir, "metadata_light.pkl"), "rb"))
    resolution = tuple(int(r) for r in meta["resolution"])
    ks = K_files(raw)
    n_new = 0
    for t in timesteps:
        dst = os.path.join(out, "data", f"timestep_{int(t):05d}.bin")
        stored = os.path.join(traj_dir, "data", f"timestep_{int(t):05d}{_BF16_SUFFIX}")
        if int(t) >= len(ks) or not os.path.exists(stored):
            continue
        for p in (f"poten_{int(t):05d}.bin", f"poten_{int(t):05d}{_BF16_SUFFIX}"):
            src = os.path.join(traj_dir, "data", p)
            if os.path.exists(src) and not os.path.lexists(os.path.join(out, "data", p)):
                os.symlink(src, os.path.join(out, "data", p))
        if os.path.exists(dst):
            continue
        df = do_ifft(read_k_spectral(os.path.join(raw, ks[int(t)]), resolution)[1])
        ref = np.fromfile(stored, dtype=np.uint16)
        if not np.array_equal(df.astype(bfloat16).ravel().view(np.uint16), ref):
            raise RuntimeError(f"{name} t={t}: rebuilt df does not round to the stored bf16 shard")
        df.tofile(dst + ".tmp")
        os.replace(dst + ".tmp", dst)
        n_new += 1
    return f"{name}: {n_new} fp32 shards rebuilt"


def solver_df_to_realspace(df_spec: np.ndarray) -> np.ndarray:
    """gyaradax spectral df (vpar,mu,s,kx,ky) -> cyclone real-space (2,vpar,mu,s,x,y).

    Exact inverse of the generate-side ``model_space_to_solver_df`` (fftshift over
    kx, inverse spatial FFT over (kx,ky)). The df is stored raw/unscaled, in the
    same amplitude convention as the GKW-preprocessed data (so a model-generated
    df integrates to the correct flux). The gyaradax/GKW flux discrepancy lives in
    the geometry ``parseval`` factor, which is corrected there, not on the df.
    """
    un = np.fft.fftshift(df_spec, axes=(3,))
    phys = np.fft.ifftn(un, axes=(3, 4), norm="forward")
    return np.stack([phys.real, phys.imag]).astype("float32")


def preprocess_gyaradax(
    traj_dir: str,
    backend: DataBackend,
    target_dir: str = "/local00/bioinf/galletti",
    out_name: str = None,
    metadata_only: bool = False,
    verify: bool = True,
    flux_atol: float = 1.0,
    show_tqdm: bool = False,
    device: str = "cpu",
):
    """Convert a gyaradax run folder (step_*.npz + config.yaml + geometry.pkl) into
    the cyclone/kvikio format (real-space 2-channel df, real phi, metadata).

    The df is scaled to the neugk integral convention and its flux is verified
    against the gyaradax-reported heat flux. Spectra (kyspec/fluxspec) and stats
    are recomputed consistently. Returns the output path.
    """
    import glob
    import pickle

    import torch
    from omegaconf import OmegaConf

    from neugk.physics.integrals import FluxIntegral

    traj_dir = str(traj_dir)
    name = out_name or os.path.basename(os.path.normpath(traj_dir))
    if isinstance(backend, KvikIOBackend):
        dir_out = f"{target_dir}/preprocessed_kvikio"
    else:
        dir_out = f"{target_dir}/preprocessed"
    os.makedirs(dir_out, exist_ok=True)
    out_path = backend.format_path(
        os.path.join(dir_out, name), spatial_ifft=True, split_into_bands=None, real_potens=True
    )

    cfg = OmegaConf.load(os.path.join(traj_dir, "config.yaml"))
    with open(os.path.join(traj_dir, "geometry.pkl"), "rb") as fh:
        np_geom = pickle.load(fh)

    # electrostatic adiabatic switches, which the computed geometry omits
    adiabatic = 1.0 if bool(cfg.grid.get("adiabatic_electrons", True)) else 0.0
    np_geom["adiabatic"] = np.array(adiabatic, dtype=np.float64)
    np_geom["beta"] = np.array(float(cfg.physics.get("beta", 0.0)), dtype=np.float64)
    np_geom["nlapar"] = np.array(0.0, dtype=np.float64)
    np_geom["nlbpar"] = np.array(0.0, dtype=np.float64)

    ints = np.asarray(np_geom["ints"])
    nvpar = len(np.asarray(np_geom["intvp"]))
    nmu = len(np.asarray(np_geom["intmu"]))
    ns = len(ints)
    nkx = len(np.asarray(np_geom["kxrh"]))
    nky = len(np.asarray(np_geom["krho"]))

    # gkw parseval convention: [1, 2ns, 2ns, ...] where gyaradax stores [1, 2, 2, ...]
    parseval = np.asarray(np_geom["parseval"], dtype=np.float64).copy()
    parseval[1:] *= float(ns)
    np_geom["parseval"] = parseval
    resolution = (nvpar, nmu, ns, nkx, nky)
    sgrid = np.asarray(np_geom["sgrid"]).ravel()
    ds = float(sgrid[1] - sgrid[0]) if sgrid.size > 1 else 1.0 / ns

    steps = sorted(glob.glob(os.path.join(traj_dir, "step_*.npz")))
    if not steps:
        raise FileNotFoundError(f"no step_*.npz dumps in {traj_dir}")

    # torch geometry for the neugk flux integral (+ ES adiabatic defaults it omits)
    geom_t = {
        k: torch.as_tensor(np.asarray(v), dtype=torch.float64)
        for k, v in np_geom.items()
        if np.asarray(v).dtype.kind in "fiu"
    }
    for k, val in {"adiabatic": 1.0, "beta": 0.0, "nlapar": 0.0, "nlbpar": 0.0, "de": 1.0}.items():
        geom_t.setdefault(k, torch.tensor(val, dtype=torch.float64))
    integrator = FluxIntegral(real_potens=True, spectral_potens=False, flux_fields=True).to(device)
    geom_b = {k: g.unsqueeze(0).to(device) for k, g in geom_t.items()}

    times, fluxes, kyspecs, fluxspecs = [], [], [], []
    df_stats, phi_stats, flux_stats = RunningMeanStd(), RunningMeanStd(), RunningMeanStd()

    with backend.create(out_path) as f:
        iterator = enumerate(steps)
        if show_tqdm:
            iterator = tqdm(iterator, total=len(steps), desc=name, leave=False)
        for idx, step_path in iterator:
            d = np.load(step_path)
            df_real = solver_df_to_realspace(d["df"])
            dft = torch.as_tensor(df_real).unsqueeze(0).to(device)
            phi_r, (pflux, eflux, vflux) = integrator(geom_b, df=dft)
            eflux = eflux[0]  # (vpar,mu,s,kx,ky) per-mode
            eflux_total = float(eflux.sum().item())
            reported = float(d["fluxes"][1])
            if verify and not np.isclose(eflux_total, reported, rtol=0.0, atol=flux_atol):
                warnings.warn(
                    f"{name} step {int(d['step'])}: flux {eflux_total:.4f} != reported "
                    f"{reported:.4f}"
                )
            # per-ky heat flux spectrum: sum the per-mode flux over all but the ky axis
            fluxspec = np.asarray(
                eflux.sum(dim=tuple(range(eflux.ndim - 1))).cpu(), dtype=np.float32
            )
            kyspecs.append(np.asarray(d["ky_spec"], dtype=np.float32))
            fluxspecs.append(fluxspec)
            times.append(float(d["time"]))
            fluxes.append(reported)

            phi = np.asarray(phi_r[0].cpu(), dtype=np.float32)
            df_stats.update(df_real, np.zeros_like(df_real), df_real, df_real)
            phi_stats.update(phi, np.zeros_like(phi), phi, phi)
            flux_stats.update(
                np.float32(reported), np.float32(0.0), np.float32(reported), np.float32(reported)
            )
            if not metadata_only:
                backend.write_df(f, str(idx).zfill(5), df=df_real)
                backend.write_phi(f, str(idx).zfill(5), phi=phi)

        metadata = {
            "timesteps": np.asarray(times),
            "resolution": resolution,
            "ds": ds,
            "ion_temp_grad": np.array([float(cfg.physics.rlt)]),
            "density_grad": np.array([float(cfg.physics.rln)]),
            "flux": np.clip(np.asarray(fluxes), a_min=0.0, a_max=None),
            "s_hat": np.array([float(cfg.geometry.shat)]),
            "q": np.array([float(cfg.geometry.q)]),
            "geometry": np_geom,
            "kyspec": np.asarray(kyspecs),
            "fluxspec": np.asarray(fluxspecs),
            "df_mean": df_stats.mean, "df_var": df_stats.var, "df_std": np.sqrt(df_stats.var),
            "df_min": df_stats.min, "df_max": df_stats.max,
            "phi_mean": phi_stats.mean, "phi_var": phi_stats.var, "phi_std": np.sqrt(phi_stats.var),
            "phi_min": phi_stats.min, "phi_max": phi_stats.max,
            "flux_mean": flux_stats.mean, "flux_var": flux_stats.var,
            "flux_std": np.sqrt(flux_stats.var), "flux_min": flux_stats.min, "flux_max": flux_stats.max,
        }
        backend.write_metadata(f, metadata)
    return out_path


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
                orig_knth, spec = read_k_spectral(f"{dir_in}/{k}", resolution)
                knth = orig_knth.copy()

                if spatial_ifft:
                    knth = spec
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
                    # the stored potential must be the field solve of the stored df
                    phi_int = field_solve_phi(knth, geometry)
                    rel = np.linalg.norm(phi - phi_int) / np.linalg.norm(phi_int)
                    assert rel < 1e-2, f"poten {pot} does not match the field solve of {k} (rel-L2 {rel:.3e})"

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
        "--trajs_file",
        type=str,
        default=None,
        help="File with one trajectory name per line; overrides --num_iterations "
        "(used to exclude zero-flux / config-excluded trajectories).",
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
    parser.add_argument(
        "--restore_fp32",
        type=str,
        default=None,
        help="Output root: rebuild the fp32 df shards of the --bf16_trajs at --restore_timesteps "
        "from the raw dumps there (checked bit for bit against the stored bf16 shards).",
    )
    parser.add_argument(
        "--restore_timesteps", type=int, nargs="+", default=None, help="Frame indices to restore."
    )
    parser.add_argument(
        "--restore_windows",
        type=str,
        default=None,
        help="Transition windows json: restore t0..t1 (step 2) of each trajectory instead.",
    )
    parser.add_argument(
        "--rewrite_poten",
        action="store_true",
        help="Overwrite poten_*.bin of already-preprocessed trajectories (see --bf16_trajs) with the "
        "field solve of their df, backing up the originals to --poten_backup. Resumable.",
    )
    parser.add_argument(
        "--poten_backup",
        type=str,
        default=None,
        help="Backup directory for --rewrite_poten (required).",
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["gkw", "gyaradax"],
        default="gkw",
        help="Raw format: 'gkw' (K-files/input.dat/geom.dat) or 'gyaradax' "
        "(step_*.npz + config.yaml + geometry.pkl).",
    )
    parser.add_argument(
        "--gyaradax_dirs",
        type=str,
        nargs="+",
        default=None,
        help="gyaradax run folders to convert when --source gyaradax (absolute paths).",
    )
    args = parser.parse_args()

    # gyaradax -> cyclone conversion path
    if args.source == "gyaradax":
        if not args.gyaradax_dirs:
            print("--source gyaradax requires --gyaradax_dirs", file=sys.stderr)
            sys.exit(1)
        backend = KvikIOBackend(use_kvikio=False) if args.backend == "kvikio" else H5Backend()
        for traj_dir in args.gyaradax_dirs:
            out = preprocess_gyaradax(
                traj_dir,
                backend=backend,
                target_dir=args.target_dir,
                metadata_only=args.metadata_only,
            )
            meta = backend.read_metadata(out)
            print(
                f"{out}: {len(meta['timesteps'])} steps, "
                f"df mean/std {meta['df_mean'].mean():.3e}/{meta['df_std'].mean():.3e}, "
                f"flux mean {float(np.mean(meta['flux'])):.3f}",
                flush=True,
            )
        sys.exit(0)

    # fp32 df shards of bf16-only trajectories, rebuilt from the raw dumps into another root
    if args.restore_fp32:
        from multiprocessing import get_context

        import json

        root_dir = os.path.join(args.target_dir, "preprocessed_kvikio")
        raw_root = os.path.join(args.root, args.raw_subdir)
        if args.restore_windows:
            # per-trajectory onset windows t0..t1, step 2 (the transition set of nf_main)
            windows = json.load(open(args.restore_windows))
            jobs = [
                (os.path.join(root_dir, f"{k}_ifft_realpotens"), list(range(int(v["t0"]), int(v["t1"]) + 1, 2)))
                for k, v in windows.items()
                if not v.get("bad", False)
            ]
        else:
            jobs = [(d, args.restore_timesteps) for d in resolve_traj_dirs(root_dir, args.bf16_trajs)]
        with get_context("spawn").Pool(args.num_workers) as pool:
            results = [
                pool.apply_async(restore_fp32, (d, args.restore_fp32, ts, raw_root)) for d, ts in jobs
            ]
            for r in results:
                print(r.get(), flush=True)
        sys.exit(0)

    # potential rewrite path (only poten_*.bin and the phi stats change)
    if args.rewrite_poten:
        from multiprocessing import get_context

        if not args.poten_backup:
            print("--rewrite_poten requires --poten_backup", file=sys.stderr)
            sys.exit(1)
        root_dir = os.path.join(args.target_dir, "preprocessed_kvikio")
        traj_dirs = resolve_traj_dirs(root_dir, args.bf16_trajs)
        with get_context("spawn").Pool(args.num_workers) as pool:
            for msg in pool.imap_unordered(partial(rewrite_poten, backup_dir=args.poten_backup), traj_dirs):
                print(msg, flush=True)
        sys.exit(0)

    # bf16 sibling conversion path (does not touch fp32 originals)
    if args.to_bf16:
        root_dir = os.path.join(args.target_dir, "preprocessed_kvikio")

        traj_dirs = resolve_traj_dirs(root_dir, args.bf16_trajs)
        convert_trajs_to_bf16(
            traj_dirs, num_workers=args.num_workers, force=args.bf16_force
        )
        sys.exit(0)

    IFFT = True
    separate_zf = False
    split_into_bands = None

    if args.trajs_file:
        with open(args.trajs_file) as fh:
            datasets = [line.strip() for line in fh if line.strip()]
    else:
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
