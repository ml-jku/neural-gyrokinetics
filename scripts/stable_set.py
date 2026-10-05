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

"""Linear-stability labels plus gyroflow latents for the QLKNN C_stab term.

The probe fit set has no stable state (its lowest Q_i is 1.05), so nothing in
the loss pins the critical gradient. This builds the missing half: conditions
sampled across the DiT's (rln, rlt, q, s_hat) box, weighted toward low R/L_Ti
and toward reversed and strongly positive shear, each labelled by a linear
gyaradax run, and gyroflow latents sampled for every stable one.

Stability is read from the amplitude ratio of the per-ky potential power
between two blocks of a per-ky-normalization-free linear run, not from
`GKState.last_growth_rate`: once a damped mode decays into round-off the
per-window rate becomes noise of either sign, while the long-baseline ratio
stays negative.

Writes `stable_latents.npz`; labels are checkpointed to
`stable_labels_partial.npz` after every point so an eviction costs one run.

Usage (from the repo root, under the repo venv):

    CUDA_VISIBLE_DEVICES=4 .venv/bin/python \
        experiments/gyaradax_ql_basic/stable_set.py
"""

from __future__ import annotations

import dataclasses
import os
import time

import paths  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TORAX_ROOT = os.path.dirname(os.path.dirname(HERE))

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import torax  # noqa: E402, F401

# torax appends an XLA flag this jaxlib rejects; an empty XLA_FLAGS also aborts
_FLAGS = (
    os.environ.get("XLA_FLAGS", "")
    .replace("--xla_cpu_opt_preset=FAST_COMPILE", "")
    .strip()
)
if _FLAGS:
  os.environ["XLA_FLAGS"] = _FLAGS
else:
  os.environ.pop("XLA_FLAGS", None)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

GF_BASE = paths.ckpt_dir
GF_STATS = paths.df_stats
OUT_NPZ = os.path.join(HERE, "stable_latents.npz")
PARTIAL_NPZ = os.path.join(HERE, "stable_labels_partial.npz")

GK_GRID = dict(nvpar=32, nmu=8, ns=16, nkx=85, nky=32)
SHAPE = (32, 8, 16, 85, 32)
EPS_TRAIN = 0.19
DT_MAX = 0.01
BLOCK, N_BLOCKS, REF_BLOCK = 200, 10, 2
GAMMA_TOL = 0.0
N_COND = 260
SEED = 7
N_SAMPLES, SAMPLER_STEPS = 16, 10
# the 5x5 evaluation grid, excluded from the sample
GRID_RLT = (3.0, 5.0, 7.0, 9.0, 11.0)
GRID_SHAT = (-1.0, -0.25, 0.5, 1.25, 2.0)
GRID_RLN, GRID_Q = 1.2, 1.7


def sample_conditions() -> np.ndarray:
  """(N, 4) conditions (rln, rlt, q, shat), weighted toward marginal cases."""
  rng = np.random.default_rng(SEED)
  n = N_COND
  low = rng.random(n) < 0.7
  rlt = np.where(low, rng.uniform(2.0, 6.0, n), rng.uniform(6.0, 9.0, n))
  q_low = rng.random(n) < 0.6
  q = np.where(q_low, rng.uniform(1.5, 3.0, n), rng.uniform(3.0, 7.0, n))
  shat = np.where(
      rng.random(n) < 0.85,
      rng.uniform(-1.5, 3.0, n),
      rng.uniform(3.0, 5.0, n),
  )
  rln = rng.uniform(0.0, 3.0, n)
  cond = np.stack([rln, rlt, q, shat], axis=1)
  grid = {(GRID_RLN, r, GRID_Q, s) for r in GRID_RLT for s in GRID_SHAT}
  keep = [i for i, c in enumerate(cond) if tuple(np.round(c, 6)) not in grid]
  return cond[keep].astype(np.float64)


def build_point(rln, rlt, q, shat):
  """(GKParams, geom) for one linear point on the gyroflow grid."""
  from gyaradax.geometry import compute_geometry
  from gyaradax.params import GKParams

  geom = compute_geometry(
      q=float(q), shat=float(shat), eps=EPS_TRAIN, vpar_max=3.0, nperiod=1,
      kxmax=0.0, krhomax=1.4, ikxspace=5, signB=1.0, Rref=100.0,
      geom_type="circ", **GK_GRID)
  params = GKParams(
      dt=DT_MAX, naverage=100, disp_par=1.0, disp_vp=0.2, disp_x=0.1,
      disp_y=0.1, idisp=2, drive_scale=1.0, non_linear=False,
      disable_per_ky_norm=True, adaptive_dt=False, cfl_safety=0.95,
      mixed_precision=True, backend="cuda", adiabatic_electrons=True,
      finit="cosine2", amp_init=1e-4,
      rlt=float(rlt), rln=float(rln), mas=1.0, tmp=1.0, de=1.0, signz=1.0,
      vthrat=1.0, shat=float(shat), q=float(q), eps=EPS_TRAIN,
      dgrid=1.0, tgrid=1.0)
  return params, geom


def _log_power(phi):
  """log per-ky potential power, factored through the peak to avoid overflow."""
  amp = np.abs(phi)
  peak = float(amp.max())
  axes = tuple(range(amp.ndim - 1))
  if not np.isfinite(peak) or peak <= 0.0:
    with np.errstate(divide="ignore", invalid="ignore"):
      return np.log(np.sum(amp ** 2, axis=axes))
  with np.errstate(divide="ignore"):
    return 2.0 * np.log(peak) + np.log(np.sum((amp / peak) ** 2, axis=axes))


def growth_rates(rln, rlt, q, shat):
  """Per-ky growth rate from the potential power ratio, plus the dt used."""
  from gyaradax.cfl import estimate_linear_timestep
  from gyaradax.quasilinear import point_eval
  from gyaradax.solver import default_state, gksolve, linear_precompute

  params, geom = build_point(rln, rlt, q, shat)
  pre = linear_precompute(geom, params)
  dt_cfl = float(estimate_linear_timestep(pre, params))
  dt = min(dt_cfl, DT_MAX) if np.isfinite(dt_cfl) else DT_MAX
  params = dataclasses.replace(params, dt=dt)
  df = point_eval.initial_df(*SHAPE)
  state = default_state(nky=SHAPE[-1])
  logp, cum = [], 0.0
  for _ in range(N_BLOCKS):
    df, (phi, _f), state = gksolve(
        df, geom, params, state, n_steps=BLOCK, pre=pre)
    logp.append(_log_power(np.asarray(phi)) + 2.0 * cum)
    # rescale between blocks: with no per-ky norm the fastest mode overflows f64
    scale = float(np.max(np.abs(np.asarray(df))))
    if np.isfinite(scale) and scale > 0.0:
      df = df / scale
      cum += float(np.log(scale))
  logp = np.asarray(logp)
  span = dt * BLOCK * (N_BLOCKS - 1 - REF_BLOCK)
  with np.errstate(invalid="ignore"):
    gamma = 0.5 * (logp[-1] - logp[REF_BLOCK]) / span
  return np.where(np.isnan(gamma), -np.inf, gamma), dt


def label_conditions(cond):
  """Linear gamma(ky) for every condition, resuming from the checkpoint."""
  n = len(cond)
  gamma = np.full((n, SHAPE[-1]), np.nan)
  dts = np.full(n, np.nan)
  done = np.zeros(n, dtype=bool)
  if os.path.exists(PARTIAL_NPZ):
    prev = np.load(PARTIAL_NPZ)
    if len(prev["done"]) == n and np.allclose(prev["cond"], cond):
      gamma = prev["gamma"].copy()
      dts, done = prev["dt"].copy(), prev["done"].copy()
      print(f"resuming: {int(done.sum())}/{n} labelled", flush=True)
  t0 = time.time()
  for i in range(n):
    if done[i]:
      continue
    g, dt = growth_rates(*cond[i])
    gamma[i], dts[i], done[i] = g, dt, True
    np.savez(PARTIAL_NPZ, cond=cond, gamma=gamma, dt=dts, done=done)
    if i % 10 == 0 or i == n - 1:
      el = time.time() - t0
      rate = el / max(int(done.sum()), 1)
      print(f"[{int(done.sum())}/{n}] rlt={cond[i][1]:5.2f} "
            f"shat={cond[i][3]:+5.2f} "
            f"q={cond[i][2]:4.2f} gmax={np.max(gamma[i][1:]):+.4f} "
            f"({el / 60:.0f} min, {rate:.0f} s/pt, "
            f"eta {rate * (n - done.sum()) / 60:.0f} min)", flush=True)
  return gamma, dts


def sample_latents(cond, rows):
  """Pooled gyroflow latents for the given condition rows, neugk pooling."""
  from torax._src.transport_model import gyaradax_gyroflow_transport_model as gf

  model = gf.GyaradaxGyroflowConfig(
      ae_checkpoint_path=f"{GF_BASE}/ae327/best.eqx",
      dit_checkpoint_path=f"{GF_BASE}/dit948/best.eqx",
      df_stats_path=GF_STATS,
      n_samples=N_SAMPLES,
      sampler_steps=SAMPLER_STEPS,
  ).build_transport_model()
  sample = jax.jit(lambda c, k: model._sample_latents(c, k))  # noqa: SLF001

  pooled = []
  t0 = time.time()
  for n, i in enumerate(rows):
    key = jax.random.fold_in(jax.random.PRNGKey(SEED), int(i))
    lat = sample(jnp.asarray(cond[i], dtype=jnp.float32), key)
    pooled.append(
        np.asarray(lat.reshape(lat.shape[0], -1, lat.shape[-1])).mean(1))
    if n % 20 == 0:
      print(f"latents {n + 1}/{len(rows)} "
            f"({time.time() - t0:.0f}s)", flush=True)
  return np.stack(pooled)


def main() -> None:
  cond = sample_conditions()
  print(f"{len(cond)} conditions, grid={SHAPE}, dt<= {DT_MAX}, "
        f"{BLOCK * N_BLOCKS} steps each", flush=True)
  gamma, dts = label_conditions(cond)
  gmax = np.max(gamma[:, 1:], axis=1)
  stable = gmax <= GAMMA_TOL
  rows = np.flatnonzero(stable)
  print(f"stable {int(stable.sum())}/{len(cond)}  "
        f"|gmax|<0.02: {int(np.sum(np.abs(gmax) < 0.02))}", flush=True)
  pooled = sample_latents(cond, rows)
  np.savez_compressed(
      OUT_NPZ,
      cond=cond,
      gamma=gamma,
      gamma_max=gmax,
      stable=stable,
      stable_index=rows,
      pooled_stable=pooled,
      dt=dts,
      n_samples=N_SAMPLES,
      sampler_steps=SAMPLER_STEPS,
      seed=SEED,
      gamma_tol=GAMMA_TOL,
      block=BLOCK,
      n_blocks=N_BLOCKS,
      ref_block=REF_BLOCK,
  )
  print(f"wrote {OUT_NPZ}", flush=True)


if __name__ == "__main__":
  main()
