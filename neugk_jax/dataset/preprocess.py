"""Dataset preprocessing: raw simulations to the kvikio layout, potential rewrites, quantization.

Modes (``python -m neugk_jax.dataset.preprocess --mode=<mode>``):

* ``preprocess``: raw GKW runs (K-files, Poten/Spc3d dumps, input.dat, geom.dat) to
  ``<target>/preprocessed_kvikio/<traj>_ifft_realpotens/`` with ``data/timestep_XXXXX.bin``
  (fp32 ``(2, vpar, mu, s, x, y)``), ``data/poten_XXXXX.bin`` (fp32 real ``(x, s, y)``),
  ``metadata.pkl`` and ``metadata_light.pkl``.
* ``rewrite-phi``: overwrite ``poten_*.bin`` of preprocessed trajectories with the field solve
  of their df (backups, resumable ``DONE`` markers, recomputed phi statistics).
* ``gyaradax``: gyaradax run folders (``step_*.npz`` + ``config.yaml`` + ``geometry.pkl``)
  to the same layout, with heat-flux verification.
* ``quantize``: side-by-side quantized siblings of the fp32 shards (``.bf16.bin``,
  ``.fp16.bin``, ``.i8.bin``, ``.i4.bin``, layout in :mod:`neugk_jax.dataset.quant`) read
  by the dataloader's ``prefer_dtype`` path.

Usage::

    python -m neugk_jax.dataset.preprocess --mode=preprocess --trajs 'iteration_{0-9}' \\
        --root /path/to/gkw --target-dir /path/to/out
    python -m neugk_jax.dataset.preprocess --mode=rewrite-phi \\
        --path /path/to/out/preprocessed_kvikio --poten-backup /path/to/backup
    python -m neugk_jax.dataset.preprocess --mode=gyaradax --gyaradax-dirs /path/to/run \\
        --target-dir /path/to/out
    python -m neugk_jax.dataset.preprocess --mode=quantize \\
        --path /path/to/out/preprocessed_kvikio \\
        --trajs 'iteration_{0-299}_ifft_realpotens' --bits bf16 --num-workers 8

``--root`` (raw GKW root holding ``<raw-subdir>/<run>``), ``--target-dir`` and ``--path``
default to ``$NEUGK_RAW_ROOT``, ``$NEUGK_TARGET_DIR`` and
``$NEUGK_TARGET_DIR/preprocessed_kvikio``; without them the flags are required.
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import re
import shutil
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from neugk_jax.dataset import quant
from neugk_jax.dataset.backend import (
    LIGHT_DROP_KEYS,
    NumpyBackend,
    expand_spec,
    frame_name,
    load_meta,
    meta_path,
    save_meta,
)
from neugk_jax.evaluate.fourier import df_to_spec, phi_to_spec, spec_to_df, spec_to_phi
from neugk_jax.utils import RunningStats, atomic_write, progress, recombine_zf

RAW_ROOT = os.environ.get("NEUGK_RAW_ROOT")
TARGET_DIR = os.environ.get("NEUGK_TARGET_DIR")
KVIKIO_SUBDIR = "preprocessed_kvikio"


def resolve_traj_dirs(root_dir: str, spec=None) -> list[str]:
    """Trajectory dirs under ``root_dir`` from basenames or one brace pattern.

    ``spec=None`` selects every ``*_ifft_realpotens`` directory.
    """
    if spec is None:
        return sorted(
            os.path.join(root_dir, n)
            for n in os.listdir(root_dir)
            if n.endswith("_ifft_realpotens") and os.path.isdir(os.path.join(root_dir, n))
        )
    return [os.path.join(root_dir, n) for n in expand_spec(spec)]


def _src_bins(data_dir: str) -> list[str]:
    """List fp32 .bin sources (timestep + poten) inside ``traj/data``."""
    if not os.path.isdir(data_dir):
        return []
    out = []
    for name in os.listdir(data_dir):
        if not name.endswith(".bin"):
            continue
        if any(name.endswith(suf) for suf in quant.SUFFIX.values()):
            continue
        if not (name.startswith("timestep_") or name.startswith("poten_")):
            continue
        out.append(os.path.join(data_dir, name))
    return sorted(out)


def _quantize_file(src: str, bits: str, force: bool) -> tuple[str, int, str]:
    dst = quant.sibling(src, bits)
    if os.path.exists(dst) and not force:
        return src, 0, "skip"
    payload, scale = quant.quantize(np.fromfile(src, dtype=np.float32), bits)
    return src, quant.write(dst, payload, scale), "written"


def _process_traj(traj_dir: str, bits: str, force: bool) -> tuple[str, int, int, int]:
    files = _src_bins(os.path.join(traj_dir, "data"))
    n_written = n_skipped = bytes_written = 0
    for src in files:
        _, n, status = _quantize_file(src, bits, force=force)
        if status == "written":
            n_written += 1
            bytes_written += n
        elif status == "skip":
            n_skipped += 1
        else:
            print(f"  [{traj_dir}] {os.path.basename(src)}: {status}", file=sys.stderr)
    return traj_dir, n_written, n_skipped, bytes_written


def run_quantize(
    *, path: str, trajs: str | Sequence[str], bits: str, num_workers: int = 4, force: bool = False
) -> None:
    traj_dirs = [d for d in resolve_traj_dirs(path, trajs) if os.path.isdir(d)]
    if not traj_dirs:
        print(f"no trajectory dirs matched under {path}")
        sys.exit(1)
    print(f"quantizing {len(traj_dirs)} trajectories to {bits} from {path}")
    t0 = time.perf_counter()
    total_w = total_s = total_b = 0
    with ThreadPoolExecutor(max_workers=max(1, num_workers)) as ex:
        futures = {ex.submit(_process_traj, d, bits, force): d for d in traj_dirs}
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
        f"\ndone — {total_w} files written, {total_s} skipped, "
        f"{total_b / 1e9:.2f} GB in {elapsed:.0f}s "
        f"({total_b / max(elapsed, 1e-6) / 1e9:.2f} GB/s)"
    )


def parse_input_dat(file_path: str) -> dict:
    """Parse a GKW ``input.dat`` namelist into ``{section: {param: value}}``."""
    parsed = {}
    with open(file_path, "r") as file:
        content = file.read()
    sections = re.split(r"&\w+", content)
    headers = re.findall(r"&(\w+)", content)
    sections = [s.strip() for s in sections if len(s) and s[0] != "!" and s.strip()]
    for header, section in zip(headers, sections):
        section_dict = {}
        for param, value in re.findall(r"(\w+)\s*=\s*([-\d\.e\w]+)", section):
            try:
                section_dict[param] = float(value) if "e" in value or "." in value else int(value)
            except ValueError:
                section_dict[param] = value.strip()
        # repeated sections (species) get a trailing 0 per repeat
        while header in parsed:
            header = f"{header}0"
        parsed[header] = section_dict
    return parsed


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def load_geom_dat(file_path: str) -> dict:
    """Parse a GKW ``geom.dat`` into scalars and float64 arrays keyed by block name."""
    data, key, values = {}, None, []
    with open(file_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        if len(parts) == 1 and not _is_number(parts[0]):
            if key is not None:
                data[key] = np.array(values, dtype=np.float64)
            key, values = parts[0], []
        else:
            values.extend(map(float, parts))
    if key is not None:
        data[key] = np.array(values, dtype=np.float64)
    return data


def _gkw_bool(val) -> float:
    if isinstance(val, str):
        v = val.lower().strip()
        if v == ".true.":
            return 1.0
        if v == ".false.":
            return 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def _default_mu_grid(n: int = 8, mumax: float = 4.5) -> tuple[np.ndarray, np.ndarray]:
    dvperp = np.sqrt(2.0 * mumax) / n
    vperp = (np.arange(n + 1) - 0.5) * dvperp
    mugr = vperp**2 / 2.0
    intmu = np.abs(np.pi * ((vperp + 0.5 * dvperp) ** 2 - (vperp - 0.5 * dvperp) ** 2))
    return mugr[1:], intmu[1:]


def load_geometry(directory: str) -> dict[str, np.ndarray]:
    """Flux-integral geometry of one GKW run as float64 numpy arrays."""
    f64 = np.float64
    geom = load_geom_dat(os.path.join(directory, "geom.dat"))
    inp = parse_input_dat(os.path.join(directory, "input.dat"))
    g = {k: np.array(1.0, dtype=f64) for k in ("signz", "vthrat", "tmp", "mas", "d2X", "signB")}
    control = inp.get("control", {})
    g["nlapar"] = np.array(_gkw_bool(control.get("nlapar", 0.0)), dtype=f64)
    g["nlbpar"] = np.array(_gkw_bool(control.get("nlbpar", 0.0)), dtype=f64)
    g["beta"] = np.array(float(inp.get("parameters", {}).get("beta", 0.0)), dtype=f64)

    num_sp = 1
    for sec in inp.values():
        if "number_of_species" in sec:
            num_sp = int(sec["number_of_species"])
            break
    species = [inp[k] for k in inp if k.startswith("species")][:num_sp]
    if not species:
        species = [{}]
    for key, name in (("mas", "mass"), ("tmp", "temp"), ("de", "dens"), ("signz", "z")):
        g[key] = np.array([sp.get(name, 1.0) for sp in species], dtype=f64)
    g["vthrat"] = np.sqrt(g["tmp"] / g["mas"])
    g["adiabatic"] = np.array(1.0, dtype=f64)

    kxrh = np.loadtxt(os.path.join(directory, "kxrh"))[0]
    krho = np.loadtxt(os.path.join(directory, "krho")).T[0] / geom["kthnorm"]
    g["kxrh"] = np.asarray(kxrh, dtype=f64)
    g["krho"] = np.asarray(krho, dtype=f64)
    g["parseval"] = np.array([1.0] + [float(len(krho))] * (len(krho) - 1), dtype=f64)

    mugr_d, intmu_d = _default_mu_grid()
    if os.path.exists(os.path.join(directory, "intmu.dat")):
        intmu = np.loadtxt(os.path.join(directory, "intmu.dat"))
        intmu = intmu[:, 0] if intmu.ndim == 2 else intmu
    else:
        intmu = intmu_d
    g["intmu"] = np.asarray(intmu, dtype=f64)
    if os.path.exists(os.path.join(directory, "vperp.dat")):
        vperp = np.loadtxt(os.path.join(directory, "vperp.dat"))
        vperp = vperp[:, 0] if vperp.ndim == 2 else vperp
        mugr = vperp**2 / 2.0
    else:
        mugr = mugr_d
    g["mugr"] = np.asarray(mugr, dtype=f64)
    g["intvp"] = np.asarray(np.loadtxt(os.path.join(directory, "intvp.dat"))[0], dtype=f64)
    g["vpgr"] = np.asarray(np.loadtxt(os.path.join(directory, "vpgr.dat"))[0], dtype=f64)

    sgrid = np.loadtxt(os.path.join(directory, "sgrid"))
    ints = np.concatenate([np.array([0.0]), np.diff(sgrid)])
    ints[0] = ints[1]
    g["ints"] = ints.astype(f64)
    g["efun"] = np.asarray(-geom["E_eps_zeta"], dtype=f64)
    g["little_g"] = np.stack([geom["g_zeta_zeta"], geom["g_eps_zeta"], geom["g_eps_eps"]], -1)
    g["bn"] = np.asarray(geom["bn"], dtype=f64)
    g["bt_frac"] = np.asarray(geom["Bt_frac"], dtype=f64)
    g["rfun"] = np.asarray(geom["R"], dtype=f64)
    if len(g["de"]) > 1:
        g["adiabatic"] = np.array(0.0, dtype=f64)
    else:
        g["adiabatic"] = np.array(np.squeeze(geom.get("adiabatic", 1.0)), dtype=f64)
    for k in ("mas", "tmp", "de", "signz", "vthrat"):
        if k in geom:
            g[k] = np.asarray(geom[k], dtype=f64)
    return g


def k_files(directory: str) -> list[str]:
    # k* dumps sorted, then numeric names by value
    files = os.listdir(directory)
    digits = sorted((f for f in files if f.isdigit()), key=int)
    ks = sorted(f for f in files if f.startswith("K") and not f.endswith(".dat"))
    return ks + digits


def poten_files(directory: str) -> tuple[list[str], np.ndarray]:
    pots = sorted(f for f in os.listdir(directory) if f.startswith("Poten"))
    return pots, np.array([int(f.replace("Poten", "")) for f in pots]) - 1


def read_dump_time(dat_path: str) -> float:
    with open(dat_path, "r") as file:
        for line in file:
            parts = line.split("=")
            if parts[0].strip() == "TIME":
                return float(parts[1].strip().strip(",").strip())
    raise ValueError(f"no TIME entry in {dat_path}")


def load_k_dump(path: str, resolution: tuple) -> np.ndarray:
    # fp32 (2, vpar, mu, s, kx, ky), kx zero-centred
    ff = np.fromfile(path, dtype=np.float64)
    return np.reshape(ff, (2, *resolution), order="F").astype("float32").copy()


def check_ifft(
    transformed: np.ndarray, orig: np.ndarray, zf_separated: bool = False, atol: float = 1e-5
) -> bool:
    """True when the real-space df transforms back onto the raw spectral dump within ``atol``."""
    spec = df_to_spec(recombine_zf(transformed, axis=0) if zf_separated else transformed)
    err_re = np.max(np.abs(spec.real.astype(np.float32) - orig[0]))
    err_im = np.max(np.abs(spec.imag.astype(np.float32) - orig[1]))
    return bool(max(err_re, err_im) <= atol)


def _check_spc(abs_phi_fft: np.ndarray, spc: np.ndarray) -> bool:
    return np.allclose(abs_phi_fft, spc, rtol=0.0, atol=1e-3)


class FieldSolver:
    """Jitted field solve, heat flux and per-ky heat-flux spectrum of real-space dfs.

    Built once per trajectory geometry; ``x64`` runs the solve in float64 (potentials are
    returned as fp32 ``(x, s, y)``).
    """

    _jits: dict = {}

    def __init__(self, geometry: dict, x64: bool = True):
        import jax
        import jax.numpy as jnp

        from neugk_jax.evaluate.integrals import flux_integral, flux_spectrum, precompute_geometry

        self.x64 = x64
        self.dtype = np.float64 if x64 else np.float32
        if not FieldSolver._jits:
            FieldSolver._jits["solve"] = jax.jit(flux_integral)
            FieldSolver._jits["spectrum"] = jax.jit(flux_spectrum)
        gt = precompute_geometry(geometry, dtype=self.dtype)
        with jax.enable_x64(x64):
            self.geom = {k: jnp.asarray(v) for k, v in gt.items()}

    def __call__(self, df: np.ndarray) -> tuple[np.ndarray, float]:
        import jax
        import jax.numpy as jnp

        with jax.enable_x64(self.x64):
            phi, (_, eflux, _) = self._jits["solve"](self.geom, jnp.asarray(df, self.dtype))
            return np.asarray(phi, dtype=np.float32), float(eflux)

    def flux_spectrum(self, df: np.ndarray) -> np.ndarray:
        import jax
        import jax.numpy as jnp

        with jax.enable_x64(self.x64):
            out = self._jits["spectrum"](self.geom, jnp.asarray(df, self.dtype))
            return np.asarray(out, dtype=np.float64)


def _new_stats() -> RunningStats:
    # prior count of the stored dataset statistics
    return RunningStats(prior_count=1e-4)


def _stats_dict(prefix: str, stats: RunningStats, dtype=None) -> dict:
    return {f"{prefix}_{k}": v for k, v in stats.moments(dtype).items()}


def write_metadata(traj_dir: str, metadata: dict) -> None:
    save_meta(os.path.join(traj_dir, "metadata"), metadata, ".pkl")
    light = {k: v for k, v in metadata.items() if k not in LIGHT_DROP_KEYS}
    save_meta(os.path.join(traj_dir, "metadata_light"), light, ".pkl")


def _write_bin(path: str, arr: np.ndarray) -> None:
    if not os.path.exists(path):
        np.ascontiguousarray(arr).tofile(path)


def _merge_old_metadata(traj_dir: str, metadata: dict) -> dict:
    old = load_meta(os.path.join(traj_dir, "metadata"))
    if old is None:
        return metadata
    return {**metadata, **{k: v for k, v in old.items() if k not in metadata}}


def preprocess(
    filename: str,
    spatial_ifft: bool = True,
    separate_zf: bool = False,
    split_into_bands: Optional[int] = None,
    root: Optional[str] = RAW_ROOT,
    raw_subdir: str = "raw",
    target_dir: Optional[str] = TARGET_DIR,
    metadata_only: bool = False,
    geometry_only: bool = False,
    phi_source: str = "field_solve",
    max_timesteps: Optional[int] = None,
    x64: bool = True,
    show_tqdm: bool = False,
    position: int = 0,
) -> tuple[str, bool]:
    """Convert one raw GKW run into the kvikio layout. Returns ``(out_path, skipped)``.

    Every dump is checked: the real-space df transforms back onto the K-file, the potential
    spectrum matches ``Spc3d``, the heat flux of the df matches ``fluxes.dat`` and the GKW
    potential matches the field solve of the df. ``phi_source`` picks the stored potential:
    ``"field_solve"`` (the field solve of the stored df) or ``"gkw"`` (the ``Poten`` dump).
    ``max_timesteps`` truncates the trajectory (data, series and statistics consistently).
    """
    assert spatial_ifft, "only the real-space (ifft) layout is supported"
    assert phi_source in ("field_solve", "gkw"), phi_source
    if "Lin" in filename:
        raise ValueError(f"{filename}: linear runs are not converted by preprocess")
    if root is None:
        raise ValueError("preprocess needs the raw GKW root (root= or $NEUGK_RAW_ROOT)")
    target_dir = root if target_dir is None else target_dir
    dir_in = f"{root}/{raw_subdir}/{filename}"
    dir_out = os.path.join(target_dir, KVIKIO_SUBDIR)
    os.makedirs(dir_out, exist_ok=True)
    backend = NumpyBackend(split_into_bands=split_into_bands)
    out_path = backend.trajectory_path(os.path.join(dir_out, filename.replace("/", "_")))
    if backend.exists(out_path) and not (metadata_only or geometry_only):
        return out_path, True

    ks = k_files(dir_in)
    potens, _ = poten_files(dir_in)
    if not ks:
        # dump names follow the same sampling as iteration_0
        ref = f"{root}/{raw_subdir}/iteration_0"
        ks = k_files(ref)
        potens, _ = poten_files(ref)
    if max_timesteps is not None:
        ks, potens = ks[:max_timesteps], potens[:max_timesteps]
    timesteps = np.array([read_dump_time(f"{dir_in}/{k}.dat") for k in ks])

    sgrid = np.loadtxt(f"{dir_in}/sgrid")
    xphi = np.loadtxt(f"{dir_in}/xphi")
    krho = np.loadtxt(f"{dir_in}/krho")
    vpgr = np.loadtxt(f"{dir_in}/vpgr.dat")
    ns = sgrid.shape[1] if len(sgrid.shape) > 1 else sgrid.shape[0]
    nx, ny = xphi.shape[1], xphi.shape[0]
    nkx, nky = krho.shape[1], krho.shape[0]
    nvpar, nmu = vpgr.shape[1], vpgr.shape[0]
    resolution = (nvpar, nmu, ns, nkx, nky)

    fluxes = np.loadtxt(f"{dir_in}/fluxes.dat")[:, 1]
    orig_times = np.loadtxt(f"{dir_in}/time.dat")
    ts_slices = [np.isclose(orig_times, t).nonzero()[0][0] for t in timesteps]
    orig_fluxes = fluxes[ts_slices].copy()
    fluxes = np.clip(orig_fluxes, a_min=0.0, a_max=None)

    config = parse_input_dat(f"{dir_in}/input.dat")
    geometry = load_geometry(dir_in)
    metadata = {
        "timesteps": timesteps,
        "resolution": resolution,
        "ds": float(np.ravel(sgrid)[1] - np.ravel(sgrid)[0]),
        "ion_temp_grad": np.array([config["species"]["rlt"]]),
        "density_grad": np.array([config["species"]["rln"]]),
        "flux": fluxes,
        "s_hat": np.array([config["geom"]["shat"]]),
        "q": np.array([config["geom"]["q"]]),
        "geometry": geometry,
        "kyspec": np.loadtxt(f"{dir_in}/kyspec")[ts_slices],
        "fluxspec": np.loadtxt(f"{dir_in}/eflux_spectra.dat")[ts_slices],
    }

    if geometry_only:
        os.makedirs(os.path.join(out_path, "data"), exist_ok=True)
        write_metadata(out_path, _merge_old_metadata(out_path, metadata))
        return out_path, False

    solver = FieldSolver(geometry, x64=x64)
    df_stats, phi_stats, flux_stats = _new_stats(), _new_stats(), _new_stats()
    os.makedirs(os.path.join(out_path, "data"), exist_ok=True)
    it = progress(
        enumerate(zip(ks, potens)),
        show_tqdm,
        desc=filename,
        total=len(ks),
        position=position,
        leave=False,
    )
    for idx, (k, pot) in it:
        knth = load_k_dump(f"{dir_in}/{k}", resolution)
        orig_knth = knth.copy()
        knth = np.moveaxis(knth, 0, -1).copy().view(dtype=np.complex64)
        if separate_zf:
            knth = np.concatenate(_split_modes(knth, split_into_bands), axis=0)
        else:
            knth = spec_to_df(knth)
        assert check_ifft(
            knth, orig_knth, zf_separated=separate_zf
        ), "error transforming back to original space"

        a = np.loadtxt(f"{dir_in}/{pot}")
        phi_gkw = np.reshape(a, (nx, ns, ny), order="F").astype("float32").copy()
        b = np.loadtxt(f"{dir_in}/{pot.replace('Poten', 'Spc3d')}")
        gt_spc = np.reshape(b, (nkx, ns, nky), order="F")
        phi_fft = phi_to_spec(phi_gkw, (nkx, ns, nky))
        assert _check_spc(np.abs(phi_fft), gt_spc), "Spectral space of Phi incorrect"
        phi_gkw = spec_to_phi(phi_fft).astype(np.float32)

        phi_int, eflux = solver(recombine_zf(knth, axis=0))
        if not np.isclose(eflux, orig_fluxes[idx], rtol=0.0, atol=1e-2):
            warnings.warn(
                f"Flux integral does not match original flux! "
                f"Computed: {eflux}, Original: {orig_fluxes[idx]}"
            )
        assert np.isclose(eflux, orig_fluxes[idx], rtol=0.0, atol=1.0), "strong deviation for flux"
        rel = np.linalg.norm(phi_gkw - phi_int) / np.linalg.norm(phi_int)
        assert rel < 1e-2, f"poten {pot} does not match the field solve of {k} (rel-L2 {rel:.3e})"
        phi = phi_int if phi_source == "field_solve" else phi_gkw

        df_stats.push(knth)
        flux_stats.push(fluxes[idx])
        phi_stats.push(phi)
        if not metadata_only:
            _write_bin(os.path.join(out_path, "data", frame_name("timestep", idx) + ".bin"), knth)
            _write_bin(os.path.join(out_path, "data", frame_name("poten", idx) + ".bin"), phi)

    metadata.update(_stats_dict("df", df_stats, np.float32))
    metadata.update(_stats_dict("phi", phi_stats, np.float32))
    metadata.update({k: np.float64(v) for k, v in _stats_dict("flux", flux_stats).items()})
    if metadata_only:
        metadata = _merge_old_metadata(out_path, metadata)
    write_metadata(out_path, metadata)
    return out_path, False


def _split_modes(knth: np.ndarray, split_into_bands: Optional[int]) -> list[np.ndarray]:
    """Zonal (ky=0) and turbulent (optionally ky-banded) real-space parts of a spectral df."""

    def to_df(shifted):
        return spec_to_df(np.fft.ifftshift(shifted, axes=(3,)))

    knth = np.fft.fftshift(knth, axes=(3,))
    nky = knth.shape[4]
    zf, no_zf = knth.copy(), knth.copy()
    zf[..., 1:, :] = 0.0
    no_zf[..., 0, :] = 0.0
    out = [to_df(zf)]
    if not split_into_bands:
        return out + [to_df(no_zf)]
    per = nky // split_into_bands
    for band in range(split_into_bands):
        cur = np.zeros_like(no_zf)
        lo = 1 + band * per
        hi = None if band == split_into_bands - 1 else lo + per
        cur[..., lo:hi, :] = no_zf[..., lo:hi, :]
        out.append(to_df(cur))
    return out


def rewrite_poten(traj_dir: str, backup_dir: str, x64: bool = True) -> str:
    """Overwrite ``poten_*.bin`` of a preprocessed trajectory with the field solve of its df.

    The originals and both metadata files are copied to ``backup_dir/<name>`` first and the
    phi statistics are recomputed in both metadata files. Trajectories with a ``DONE``
    marker in their backup are skipped.
    """
    name = os.path.basename(traj_dir.rstrip("/"))
    bdir = os.path.join(backup_dir, name)
    if os.path.exists(os.path.join(bdir, "DONE")):
        return f"{name}: skip (done)"
    os.makedirs(bdir, exist_ok=True)
    data = os.path.join(traj_dir, "data")
    potens = sorted(glob.glob(os.path.join(data, "poten_[0-9][0-9][0-9][0-9][0-9].bin")))
    metas = [
        p
        for p in (meta_path(os.path.join(traj_dir, m)) for m in ("metadata", "metadata_light"))
        if p is not None
    ]
    for p in potens + metas:
        b = os.path.join(bdir, os.path.basename(p))
        if not os.path.exists(b):
            shutil.copy2(p, b)

    meta = NumpyBackend().read_metadata(traj_dir)
    shape = (2, *meta["resolution"])
    solver = FieldSolver(meta["geometry"], x64=x64)
    stats = _new_stats()
    for p in potens:
        df_path = os.path.join(data, os.path.basename(p).replace("poten_", "timestep_"))
        phi = solver(np.fromfile(df_path, dtype=np.float32).reshape(shape))[0]
        assert phi.nbytes == os.path.getsize(p), (p, phi.shape)
        atomic_write(p, phi.tofile)
        stats.push(phi)

    new = _stats_dict("phi", stats)
    for mp in metas:
        base, ext = os.path.splitext(mp)
        m = load_meta(base)
        for k in [k for k in new if k in m]:
            m[k] = np.asarray(new[k], dtype=np.asarray(m[k]).dtype)
        save_meta(base, m, ext)
    with open(os.path.join(bdir, "DONE"), "w") as f:
        f.write(f"{len(potens)}\n")
    return f"{name}: rewrote {len(potens)} potentials"


def preprocess_gyaradax(
    traj_dir: str,
    target_dir: Optional[str] = TARGET_DIR,
    metadata_only: bool = False,
    show_tqdm: bool = False,
    x64: bool = True,
) -> str:
    """Convert a gyaradax run folder (``step_*.npz`` + ``config.yaml`` + ``geometry.pkl``).

    Writes the kvikio layout (real-space 2-channel df, field-solved real phi, metadata with
    kyspec/fluxspec and statistics) with the geometry in the GKW convention. The flux of each
    df is checked against the gyaradax-reported heat flux. Returns the output path.
    """
    from omegaconf import OmegaConf

    traj_dir = str(traj_dir)
    name = os.path.basename(os.path.normpath(traj_dir))
    if target_dir is None:
        raise ValueError("preprocess_gyaradax needs target_dir (or $NEUGK_TARGET_DIR)")
    dir_out = os.path.join(target_dir, KVIKIO_SUBDIR)
    os.makedirs(dir_out, exist_ok=True)
    out_path = NumpyBackend().trajectory_path(os.path.join(dir_out, name))

    cfg = OmegaConf.load(os.path.join(traj_dir, "config.yaml"))
    with open(os.path.join(traj_dir, "geometry.pkl"), "rb") as fh:
        np_geom = pickle.load(fh)
    adiabatic = 1.0 if bool(cfg.grid.get("adiabatic_electrons", True)) else 0.0
    np_geom["adiabatic"] = np.array(adiabatic, dtype=np.float64)
    np_geom["beta"] = np.array(float(cfg.physics.get("beta", 0.0)), dtype=np.float64)
    np_geom["nlapar"] = np.array(0.0, dtype=np.float64)
    np_geom["nlbpar"] = np.array(0.0, dtype=np.float64)

    ints = np.asarray(np_geom["ints"])
    ns = len(ints)
    resolution = (
        len(np.asarray(np_geom["intvp"])),
        len(np.asarray(np_geom["intmu"])),
        ns,
        len(np.asarray(np_geom["kxrh"])),
        len(np.asarray(np_geom["krho"])),
    )
    # the flux kernel weights ints twice, so the ky factor carries one 1/ints = ns
    if not np.allclose(ints, 1.0 / ns):
        raise NotImplementedError("gyaradax import assumes a uniform s grid (ints == 1/ns)")
    parseval = np.asarray(np_geom["parseval"], dtype=np.float64).copy()
    parseval[1:] *= float(ns)
    np_geom["parseval"] = parseval
    sgrid = np.asarray(np_geom["sgrid"]).ravel()
    ds = float(sgrid[1] - sgrid[0]) if sgrid.size > 1 else 1.0 / ns

    steps = sorted(glob.glob(os.path.join(traj_dir, "step_*.npz")))
    if not steps:
        raise FileNotFoundError(f"no step_*.npz dumps in {traj_dir}")
    solver = FieldSolver(np_geom, x64=x64)

    times, fluxes, kyspecs, fluxspecs = [], [], [], []
    df_stats, phi_stats, flux_stats = _new_stats(), _new_stats(), _new_stats()
    os.makedirs(os.path.join(out_path, "data"), exist_ok=True)
    for idx, step_path in progress(
        enumerate(steps), show_tqdm, total=len(steps), desc=name, leave=False
    ):
        d = np.load(step_path)
        df_real = spec_to_df(d["df"])
        phi, eflux_total = solver(df_real)
        reported = float(d["fluxes"][1])
        if not np.isclose(eflux_total, reported, rtol=0.0, atol=1.0):
            warnings.warn(
                f"{name} step {int(d['step'])}: flux {eflux_total:.4f} != reported "
                f"{reported:.4f}"
            )
        fluxspecs.append(solver.flux_spectrum(df_real).astype(np.float32))
        kyspecs.append(np.asarray(d["ky_spec"], dtype=np.float32))
        times.append(float(d["time"]))
        fluxes.append(reported)
        df_stats.push(df_real)
        phi_stats.push(phi)
        flux_stats.push(reported)
        if not metadata_only:
            _write_bin(
                os.path.join(out_path, "data", frame_name("timestep", idx) + ".bin"), df_real
            )
            _write_bin(os.path.join(out_path, "data", frame_name("poten", idx) + ".bin"), phi)

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
        **_stats_dict("df", df_stats, np.float32),
        **_stats_dict("phi", phi_stats, np.float32),
        **{k: np.float32(v) for k, v in _stats_dict("flux", flux_stats).items()},
    }
    write_metadata(out_path, metadata)
    return out_path


def _gkw_datasets(args) -> list[str]:
    if args.trajs_file:
        with open(args.trajs_file) as fh:
            return [line.strip() for line in fh if line.strip()]
    if args.trajs:
        return expand_spec(args.trajs)
    return [f"iteration_{i}" for i in range(args.num_iterations)]


def _run_preprocess(args) -> None:
    datasets = _gkw_datasets(args)
    kwargs = dict(
        spatial_ifft=True,
        separate_zf=args.separate_zf,
        split_into_bands=args.split_into_bands,
        root=args.root,
        raw_subdir=args.raw_subdir,
        target_dir=args.target_dir,
        metadata_only=args.metadata_only,
        geometry_only=args.geometry_only,
        phi_source=args.phi_source,
        max_timesteps=args.max_timesteps,
        x64=not args.fp32,
        show_tqdm=args.tqdm,
    )

    def one(i_name):
        i, name = i_name
        try:
            return (
                name,
                *preprocess(name, position=1 + i % max(1, args.num_workers), **kwargs),
                None,
            )
        except (OSError, ValueError, AssertionError, IndexError, KeyError) as e:
            return name, None, False, e

    skipped = []
    with ThreadPoolExecutor(max(1, min(len(datasets), args.num_workers))) as ex:
        for name, out_path, was_skipped, err in ex.map(one, enumerate(datasets)):
            if err is not None:
                print(f"Error processing {name}: {err}", file=sys.stderr, flush=True)
            elif was_skipped:
                skipped.append(name)
            else:
                meta = load_meta(os.path.join(out_path, "metadata"))
                msg = f"{out_path}: {len(meta['timesteps'])} points"
                if "df_mean" in meta:
                    msg += (
                        f", mean {meta['df_mean'][0].mean():.2e}, "
                        f"std {meta['df_std'][0].mean():.2e}"
                    )
                print(msg, flush=True)
    if skipped:
        print(f"Skipped {len(skipped)} trajectories (already processed).")


def main(argv: Iterable[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--mode", choices=("preprocess", "rewrite-phi", "gyaradax", "quantize"), default="quantize"
    )
    ap.add_argument(
        "--path",
        default=TARGET_DIR and os.path.join(TARGET_DIR, KVIKIO_SUBDIR),
        help="preprocessed dataset root (rewrite-phi, quantize)",
    )
    ap.add_argument(
        "--trajs",
        nargs="+",
        default=None,
        help="brace pattern (single string) OR explicit list of trajectories; "
        "raw run names for preprocess, trajectory dirs otherwise",
    )
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--tqdm", action="store_true")
    ap.add_argument("--fp32", action="store_true", help="field solve in float32 (default float64)")
    g = ap.add_argument_group("preprocess")
    g.add_argument("--root", default=RAW_ROOT)
    g.add_argument("--raw-subdir", default="raw")
    g.add_argument(
        "--target-dir",
        default=TARGET_DIR,
        help="output root; data goes to <target-dir>/preprocessed_kvikio",
    )
    g.add_argument(
        "--num-iterations",
        type=int,
        default=300,
        help="iteration_0 .. iteration_N-1 when neither --trajs nor --trajs-file is set",
    )
    g.add_argument("--trajs-file", default=None, help="file with one raw run name per line")
    g.add_argument(
        "--metadata-only",
        action="store_true",
        help="only rewrite metadata (with statistics), no field data",
    )
    g.add_argument(
        "--geometry-only",
        action="store_true",
        help="only rewrite metadata without statistics (existing ones are kept)",
    )
    g.add_argument("--phi-source", choices=("field_solve", "gkw"), default="field_solve")
    g.add_argument("--max-timesteps", type=int, default=None)
    g.add_argument("--separate-zf", action="store_true")
    g.add_argument("--split-into-bands", type=int, default=None)
    g = ap.add_argument_group("rewrite-phi")
    g.add_argument("--poten-backup", default=None, help="backup directory (required)")
    g = ap.add_argument_group("gyaradax")
    g.add_argument("--gyaradax-dirs", nargs="+", default=None)
    g = ap.add_argument_group("quantize")
    g.add_argument(
        "--bits",
        choices=tuple(quant.SUFFIX),
        default="bf16",
        help="quantization target (fp16 / bf16 / i8 / i4)",
    )
    g.add_argument("--force", action="store_true", help="overwrite existing quantized shards")
    args = ap.parse_args(argv)
    required = {
        "quantize": ("path",),
        "rewrite-phi": ("path",),
        "preprocess": ("root",),
        "gyaradax": ("target_dir",),
    }[args.mode]
    for name in required:
        if getattr(args, name) is None:
            ap.error(
                f"--mode={args.mode} needs --{name.replace('_', '-')} "
                "(or the NEUGK_RAW_ROOT / NEUGK_TARGET_DIR environment variables)"
            )

    if args.mode == "quantize":
        run_quantize(
            path=args.path,
            trajs=args.trajs or ["iteration_{0-299}_ifft_realpotens"],
            bits=args.bits,
            num_workers=args.num_workers,
            force=args.force,
        )
    elif args.mode == "preprocess":
        _run_preprocess(args)
    elif args.mode == "rewrite-phi":
        if not args.poten_backup:
            ap.error("--mode=rewrite-phi requires --poten-backup")
        traj_dirs = resolve_traj_dirs(args.path, args.trajs)
        with ThreadPoolExecutor(max(1, args.num_workers)) as ex:
            futures = [
                ex.submit(rewrite_poten, d, args.poten_backup, not args.fp32) for d in traj_dirs
            ]
            for fut in as_completed(futures):
                print(fut.result(), flush=True)
    elif args.mode == "gyaradax":
        if not args.gyaradax_dirs:
            ap.error("--mode=gyaradax requires --gyaradax-dirs")
        for traj_dir in args.gyaradax_dirs:
            out = preprocess_gyaradax(
                traj_dir,
                target_dir=args.target_dir,
                metadata_only=args.metadata_only,
                show_tqdm=args.tqdm,
                x64=not args.fp32,
            )
            meta = load_meta(os.path.join(out, "metadata"))
            print(
                f"{out}: {len(meta['timesteps'])} steps, "
                f"df mean/std {meta['df_mean'].mean():.3e}/{meta['df_std'].mean():.3e}, "
                f"flux mean {float(np.mean(meta['flux'])):.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
