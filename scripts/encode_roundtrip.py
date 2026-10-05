# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Validate the gyaradax-5D -> gyroflow-latent encoder against its decoder.

`gyaradax_gyroflow_transport_model.encode_spectral_df` is the exact mirror of
the decode half the transport model already uses for its flux integrals. This
script measures how much of a real turbulent state survives the round trip:

  spectral df -> real space -> separate-zf -> z-score -> AE encode
             -> AE decode -> denormalize -> recombine -> spectral df

Two independent sources on the GyroSwin grid (32, 8, 16, 85, 32):

  * GKW K-dumps of iteration_13, an AE *validation* trajectory — exactly the
    representation the autoencoder was trained on;
  * a saturated gyaradax nonlinear end-state at the same operating point
    (`--nl-state`; `--make-nl-state PATH` runs it and writes the npz).

Reported per source: NMSE of the transform alone (must be ~1e-16, it is an
exact inverse), NMSE of the full AE round trip in normalized-df space and in
gyaradax spectral space, the same split per channel group (zonal vs the rest),
and the ion heat flux computed from the original and the round-tripped df.

Usage (from the repo root, under the repo venv):

    .venv/bin/python experiments/gyaradax_ql_basic/encode_roundtrip.py \
        --device 6 --make-nl-state /scratch/nl_state.npz
    .venv/bin/python experiments/gyaradax_ql_basic/encode_roundtrip.py \
        --device 6 --timesteps 150,200,250 --nl-state /scratch/nl_state.npz
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import paths  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TORAX_ROOT = os.path.dirname(os.path.dirname(HERE))

RAW_DIR = os.path.join(paths.gkw_raw_dir, "iteration_13")
META_PATH = os.path.join(paths.preprocessed_dir,
                         "iteration_13_ifft_realpotens", "metadata_light.pkl")
GF_BASE = paths.ckpt_dir
GF_STATS = paths.df_stats
K_SHAPE = (2, 32, 8, 16, 85, 32)
GRID = dict(nvpar=32, nmu=8, ns=16, nkx=85, nky=32, ikxspace=5, krhomax=1.4)
EPS_TRAIN = 0.19


def _parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--device", default="6", help="CUDA_VISIBLE_DEVICES")
  p.add_argument("--timesteps", default="150,200,250")
  p.add_argument("--nl-state", default="", help="npz with a gyaradax 5D df")
  p.add_argument(
      "--make-nl-state",
      default="",
      help=(
          "run a nonlinear gyaradax point at the iteration_13 parameters,"
          " write the saturated 5D df to this npz, and exit"
      ),
  )
  p.add_argument("--out", default=os.path.join(HERE, "encode_roundtrip.json"))
  return p.parse_args()


_ARGS = _parse_args()
os.environ.setdefault("CUDA_VISIBLE_DEVICES", _ARGS.device)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
if TORAX_ROOT not in sys.path:
  sys.path.insert(0, TORAX_ROOT)

import torax  # noqa: E402  F401

# torax appends an XLA flag this jaxlib rejects; an empty XLA_FLAGS also aborts
_flags = (
    os.environ.get("XLA_FLAGS", "")
    .replace("--xla_cpu_opt_preset=FAST_COMPILE", "")
    .strip()
)
if _flags:
  os.environ["XLA_FLAGS"] = _flags
else:
  os.environ.pop("XLA_FLAGS", None)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from gyaradax import integrals as gyx_integrals  # noqa: E402
from torax._src.transport_model import gyaradax_gyroflow_transport_model as gf  # noqa: E402
from torax._src.transport_model import gyaradax_ql_transport_model as ql_lib  # noqa: E402


class _GridCfg:
  """Attribute bag `gyaradax_geometry_at` reads the grid sizes off."""

  ns, nkx, nky, nvpar, nmu = 16, 85, 32, 32, 8
  ikxspace, krhomax = 5, 1.4


def k_file_names(raw: str) -> list[str]:
  """GKW K-dump listing convention: K01..K99 then the bare digit files."""
  files = os.listdir(raw)
  digit = sorted([f for f in files if f.isdigit()], key=int)
  named = sorted(
      [f for f in files if f.startswith("K") and not f.endswith(".dat")]
  )
  return named + digit


def load_k_dump(raw: str, ts: int) -> jnp.ndarray:
  """One GKW K-dump -> complex (vpar, mu, s, kx, ky), fortran-ordered fp64."""
  path = os.path.join(raw, k_file_names(raw)[ts])
  arr = np.reshape(np.fromfile(path, dtype=np.float64), K_SHAPE, order="F")
  return jnp.asarray(arr[0] + 1j * arr[1])


def nmse(pred, ref) -> float:
  return float(jnp.sum(jnp.abs(pred - ref) ** 2) / jnp.sum(jnp.abs(ref) ** 2))


def make_nl_state(path: str) -> None:
  """Run a nonlinear gyaradax point to saturation and dump its 5D df."""
  import time

  from gyaradax.geometry import compute_geometry
  from gyaradax.params import GKParams
  from gyaradax.quasilinear import point_eval

  with open(META_PATH, "rb") as f:
    meta = pickle.load(f)
  rlt = float(np.squeeze(meta["ion_temp_grad"]))
  rln = float(np.squeeze(meta["density_grad"]))
  q = float(np.squeeze(meta["q"]))
  shat = float(np.squeeze(meta["s_hat"]))
  geom = compute_geometry(
      q=q,
      shat=shat,
      eps=EPS_TRAIN,
      vpar_max=3.0,
      nperiod=1,
      kxmax=0.0,
      krhomax=GRID["krhomax"],
      ikxspace=GRID["ikxspace"],
      signB=1.0,
      Rref=100.0,
      geom_type="circ",
      nvpar=GRID["nvpar"],
      nmu=GRID["nmu"],
      ns=GRID["ns"],
      nkx=GRID["nkx"],
      nky=GRID["nky"],
  )
  params = GKParams(
      dt=0.01,
      naverage=100,
      disp_par=1.0,
      disp_vp=0.2,
      disp_x=0.1,
      disp_y=0.1,
      idisp=2,
      drive_scale=1.0,
      non_linear=True,
      disable_per_ky_norm=False,
      adaptive_dt=True,
      cfl_safety=0.95,
      mixed_precision=True,
      backend="cuda",
      adiabatic_electrons=True,
      finit="cosine2",
      amp_init=1e-4,
      rlt=rlt,
      rln=rln,
      mas=1.0,
      tmp=1.0,
      de=1.0,
      signz=1.0,
      vthrat=1.0,
      shat=shat,
      q=q,
      eps=EPS_TRAIN,
      dgrid=1.0,
      tgrid=1.0,
  )
  shape = (
      GRID["nvpar"],
      GRID["nmu"],
      GRID["ns"],
      GRID["nkx"],
      GRID["nky"],
  )
  start = time.time()
  q_i, _qe, _pfe, df = point_eval.nl_at_point(
      params,
      geom,
      shape,
      n_steps=15000,
      tail_blocks=10,
      block=500,
      return_df=True,
  )
  print(
      f"Qi={float(q_i):.4g} GB in {time.time() - start:.0f}s"
      f"  max|df|={float(np.abs(np.asarray(df)).max()):.4g}",
      flush=True,
  )
  np.savez(
      path,
      df=np.asarray(df),
      qi=float(q_i),
      rlt=rlt,
      rln=rln,
      q=q,
      shat=shat,
      eps=EPS_TRAIN,
  )
  print("wrote", path)


def main() -> None:
  if _ARGS.make_nl_state:
    make_nl_state(_ARGS.make_nl_state)
    return
  gf.enable_legacy_swin_residual()
  ae, _dit = gf._load_models(  # noqa: SLF001
      f"{GF_BASE}/ae327/best.eqx", f"{GF_BASE}/dit948/best.eqx"
  )
  stats = gf.load_df_stats(GF_STATS)
  mean, std = stats

  with open(META_PATH, "rb") as f:
    meta = pickle.load(f)
  q = float(np.squeeze(meta["q"]))
  shat = float(np.squeeze(meta["s_hat"]))
  geom = ql_lib.gyaradax_geometry_at(
      q=jnp.asarray(q),
      shat=jnp.asarray(shat),
      eps=jnp.asarray(EPS_TRAIN),
      config=_GridCfg,
      topology=ql_lib._get_topology_cached(85, 32, 5, 16),  # noqa: SLF001
  )
  geom_tensors = gyx_integrals.geom_tensors(geom)

  @jax.jit
  def eflux(spec):
    _phi, (_p, e, _v) = gyx_integrals.get_integrals(
        spec, geom, adiabatic_electrons=True, geom=geom_tensors
    )
    return e

  encode = jax.jit(lambda s: gf.encode_spectral_df(ae, s, stats))
  decode = jax.jit(lambda z: gf.decode_latents_to_spectral(ae, z, stats))

  sources = []
  for ts in [int(t) for t in _ARGS.timesteps.split(",") if t.strip()]:
    sources.append((f"gkw_iteration_13_ts{ts}", load_k_dump(RAW_DIR, ts)))
  if _ARGS.nl_state:
    raw = np.load(_ARGS.nl_state)
    sources.append(("gyaradax_nl_endstate", jnp.asarray(raw["df"])))

  rows = []
  for name, spec in sources:
    df_norm = gf.spectral_to_denormalized_df(spec)
    df_norm = ((df_norm - mean) / std).astype(jnp.float32)
    # the transform pair alone: an exact inverse, so this bounds fp round-off
    transform_only = nmse(
        gf.denormalized_df_to_spectral(
            df_norm.astype(jnp.float64) * std + mean
        ),
        spec,
    )
    latents = encode(spec)
    spec_hat = decode(latents)
    df_hat = ae.decode(latents)["df"]
    row = {
        "source": name,
        "latent_shape": list(latents.shape),
        "max_abs_df": float(jnp.max(jnp.abs(spec))),
        "transform_only_nmse": transform_only,
        "df_norm_nmse": nmse(df_hat, df_norm),
        "df_norm_nmse_zonal": nmse(df_hat[:2], df_norm[:2]),
        "df_norm_nmse_nonzonal": nmse(df_hat[2:], df_norm[2:]),
        "spectral_nmse": nmse(spec_hat, spec),
        "eflux_orig": float(eflux(spec)),
        "eflux_roundtrip": float(eflux(spec_hat)),
    }
    row["eflux_ratio"] = row["eflux_roundtrip"] / row["eflux_orig"]
    rows.append(row)
    print(
        f"{name:28s} transform={transform_only:.2e} "
        f"df_nmse={row['df_norm_nmse']:.4f} "
        f"spec_nmse={row['spectral_nmse']:.4f} "
        f"Qi {row['eflux_orig']:.3f} -> {row['eflux_roundtrip']:.3f} "
        f"({row['eflux_ratio']:.3f})",
        flush=True,
    )

  summary = {
      "q": q,
      "shat": shat,
      "eps": EPS_TRAIN,
      "ae_checkpoint": f"{GF_BASE}/ae327/best.eqx",
      "df_stats": GF_STATS,
      "rows": rows,
  }
  with open(_ARGS.out, "w") as f:
    json.dump(summary, f, indent=1)
  print("wrote", _ARGS.out)


if __name__ == "__main__":
  main()
