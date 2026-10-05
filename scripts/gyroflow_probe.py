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

"""Linear probe from gyroflow latents to the ion heat flux.

Upstream neugk fits its probes exactly this way (`neugk/evaluate.py`
`collect_latents` + `run_probing_evaluation`): global-average-pool the latent
over its spatial axes, append a bias column, solve the normal equations for a
z-scored target, and report RMSE against the denormalized truth. We reproduce
that recipe on the gyroflow latents, with the ion heat flux as the target, and
compare it against the route the transport model actually uses — decode the
same latents and run gyaradax's flux integral.

Data: `ds_small_full.npz` (259 GKW trajectories, `gkw_raw`) is the fit set and
`ds_eval_b6.npz` (59 trajectories from a later batch the DiT never saw) is the
held-out set. `Y` is the tail-averaged nonlinear ion heat flux in GKW gyroBohm
units; `F` carries (rlt_i, rln_i, rlt_e, rln_e, shat, q, eps, beta), from which
the DiT conditioning (dg, itg, q, s_hat) = (rln_i, rlt_i, q, shat) is built in
the sorted-name order the model trained with.

Stages:

    --stage latents   sample latents per row (and, for --flux-files, decode
                      them and integrate the flux); writes a resumable npz
    --stage fit       fit the probes and score every route on the held-out set

Usage (from the repo root, under the repo venv):

    .venv/bin/python experiments/gyaradax_ql_basic/gyroflow_probe.py \
        --stage latents --device 6 --n-samples 16
    .venv/bin/python experiments/gyaradax_ql_basic/gyroflow_probe.py --stage fit
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import paths

HERE = os.path.dirname(os.path.abspath(__file__))
TORAX_ROOT = os.path.dirname(os.path.dirname(HERE))

DS_DIR = paths.dataset_dir
FIT_FILE = "ds_small_full"
HOLDOUT_FILE = "ds_eval_b6"
GF_BASE = paths.ckpt_dir
GF_STATS = paths.df_stats
EPS_TRAIN = 0.19
SEED = 7
# ridge strengths swept by leave-one-condition-out CV on the fit set
RIDGE_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4)


def _parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--stage", default="fit", choices=("latents", "fit"))
  p.add_argument("--device", default="6", help="CUDA_VISIBLE_DEVICES")
  p.add_argument("--n-samples", type=int, default=16)
  p.add_argument("--sampler-steps", type=int, default=10)
  p.add_argument(
      "--flux-files",
      default=HOLDOUT_FILE,
      help="comma list of dataset stems to also decode + flux-integrate",
  )
  p.add_argument("--limit", type=int, default=0, help="debug: first N rows")
  p.add_argument("--out-dir", default=HERE)
  return p.parse_args()


_ARGS = _parse_args()
if _ARGS.stage == "latents":
  os.environ.setdefault("CUDA_VISIBLE_DEVICES", _ARGS.device)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
if TORAX_ROOT not in sys.path:
  sys.path.insert(0, TORAX_ROOT)

import numpy as np


def latent_path(stem: str) -> str:
  return os.path.join(_ARGS.out_dir, f"probe_latents_{stem}.npz")


def load_dataset(stem: str):
  """(cond (N,4), Y (N,), src_id (N,)) for one dataset stem."""
  d = np.load(os.path.join(DS_DIR, f"{stem}.npz"), allow_pickle=True)
  names = [str(n) for n in d["feature_names"]]
  f = d["F"]
  col = {n: f[:, i] for i, n in enumerate(names)}
  # sorted-name order (dg, itg, q, s_hat) — the order the DiT trained with
  cond = np.stack(
      [col["rln_i"], col["rlt_i"], col["q"], col["shat"]], axis=1
  ).astype(np.float32)
  return cond, np.asarray(d["Y"], dtype=np.float64), d["src_id"]


# ------------------------------------------------------------------ latents


def _init_jax():
  """Import torax, repair XLA_FLAGS, then hand back jax."""
  import torax  # noqa: F401

  flags = (
      os.environ.get("XLA_FLAGS", "")
      .replace("--xla_cpu_opt_preset=FAST_COMPILE", "")
      .strip()
  )
  if flags:
    os.environ["XLA_FLAGS"] = flags
  else:
    os.environ.pop("XLA_FLAGS", None)
  import jax

  return jax


def run_latents() -> None:
  """Sample latents (and optionally fluxes) for both datasets, resumably."""
  jax = _init_jax()
  import jax.numpy as jnp
  from torax._src.transport_model import gyaradax_gyroflow_transport_model as gf
  from torax._src.transport_model import gyaradax_ql_transport_model as ql_lib

  model = gf.GyaradaxGyroflowConfig(
      ae_checkpoint_path=f"{GF_BASE}/ae327/best.eqx",
      dit_checkpoint_path=f"{GF_BASE}/dit948/best.eqx",
      df_stats_path=GF_STATS,
      n_samples=_ARGS.n_samples,
      sampler_steps=_ARGS.sampler_steps,
  ).build_transport_model()
  flux_stems = {s for s in _ARGS.flux_files.split(",") if s.strip()}

  sample = jax.jit(lambda c, k: model._sample_latents(c, k))  # noqa: SLF001

  def flux_of(latents, geom):
    df = model._decode_latents(latents)  # noqa: SLF001
    _p, e = model._per_sample_fluxes(df, geom)  # noqa: SLF001
    return e

  flux = jax.jit(flux_of)

  for stem in (FIT_FILE, HOLDOUT_FILE):
    cond, y, src = load_dataset(stem)
    if _ARGS.limit:
      cond, y, src = cond[: _ARGS.limit], y[: _ARGS.limit], src[: _ARGS.limit]
    want_flux = stem in flux_stems
    pooled, fluxes = [], []
    t0 = time.time()
    for i, c in enumerate(cond):
      cond_j = jnp.asarray(c)
      key = jax.random.fold_in(jax.random.PRNGKey(SEED), i)
      latents = sample(cond_j, key)
      # neugk pooling: (..., C) -> mean over every axis but the channel
      pooled.append(
          np.asarray(
              latents.reshape(latents.shape[0], -1, latents.shape[-1])
          ).mean(1)
      )
      if want_flux:
        geom = ql_lib.gyaradax_geometry_at(
            q=cond_j[2],
            shat=cond_j[3],
            eps=jnp.asarray(EPS_TRAIN),
            config=model,
            topology=model.topology,
        )
        fluxes.append(np.asarray(flux(latents, geom), dtype=np.float64))
      if i % 10 == 0:
        rate = (time.time() - t0) / (i + 1)
        print(f"{stem} {i + 1}/{len(cond)} ({rate:.1f} s/row)", flush=True)
    out = dict(
        cond=cond,
        y=y,
        src_id=src,
        pooled=np.stack(pooled),
        n_samples=_ARGS.n_samples,
        sampler_steps=_ARGS.sampler_steps,
        seed=SEED,
    )
    if want_flux:
      out["flux_samples"] = np.stack(fluxes)
    np.savez_compressed(latent_path(stem), **out)
    print(f"wrote {latent_path(stem)} in {time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------ probes


def _design(x: np.ndarray) -> np.ndarray:
  """Append the bias column neugk's probe fits (`x_train_b`)."""
  return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)


def _fit(x: np.ndarray, y: np.ndarray, ridge: float = 0.0) -> np.ndarray:
  xb = _design(x)
  if ridge <= 0.0:
    return np.linalg.pinv(xb) @ y
  gram = xb.T @ xb
  # bias column left unpenalized
  pen = ridge * np.eye(gram.shape[0])
  pen[-1, -1] = 0.0
  return np.linalg.solve(gram + pen, xb.T @ y)


def _scores(pred: np.ndarray, truth: np.ndarray) -> dict:
  resid = pred - truth
  ss_tot = float(np.sum((truth - truth.mean()) ** 2))
  pos = np.maximum(pred, 1e-3)
  return {
      "rmse": float(np.sqrt(np.mean(resid**2))),
      "r2": float(1.0 - np.sum(resid**2) / ss_tot),
      "log_rmse": float(
          np.sqrt(
              np.mean((np.log10(pos) - np.log10(np.maximum(truth, 1e-3))) ** 2)
          )
      ),
  }


def _cv_ridge(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
  """Pick the ridge strength by 5-fold CV over conditions (not samples)."""
  uniq = np.unique(groups)
  rng = np.random.default_rng(SEED)
  folds = rng.permutation(len(uniq)) % 5
  fold_of = {g: folds[i] for i, g in enumerate(uniq)}
  gid = np.asarray([fold_of[g] for g in groups])
  best, best_err = RIDGE_GRID[0], np.inf
  for ridge in RIDGE_GRID:
    err = 0.0
    for k in range(5):
      tr, te = gid != k, gid == k
      w = _fit(x[tr], y[tr], ridge)
      err += float(np.sum((_design(x[te]) @ w - y[te]) ** 2))
    if err < best_err:
      best, best_err = ridge, err
  return best


def _per_condition(pred_samples: np.ndarray) -> np.ndarray:
  """Average a per-sample prediction back to one number per condition."""
  return pred_samples.mean(1)


def run_fit() -> None:
  fit = np.load(latent_path(FIT_FILE), allow_pickle=True)
  hold = np.load(latent_path(HOLDOUT_FILE), allow_pickle=True)
  n_s = int(fit["n_samples"])

  y_fit, y_hold = fit["y"], hold["y"]
  # z-score the target exactly like neugk's probe (dataset flux stats)
  y_mean, y_std = float(y_fit.mean()), float(y_fit.std())
  results = {}

  def register(name, pred_hold, pred_fit=None, extra=None) -> None:
    row = _scores(pred_hold, y_hold)
    if pred_fit is not None:
      row["fit"] = _scores(pred_fit, y_fit)
    if extra:
      row.update(extra)
    results[name] = row
    in_sample = (
        f"  in-sample R2={row['fit']['r2']:+.3f}" if "fit" in row else ""
    )
    print(
        f"{name:34s} RMSE={row['rmse']:7.2f}  R2={row['r2']:+.3f}  "
        f"logRMSE={row['log_rmse']:.3f}{in_sample}"
        + (f"  [{extra}]" if extra else ""),
        flush=True,
    )

  # ---- latent probes, neugk recipe: per-sample rows, pooled latent + bias
  x_fit = fit["pooled"].reshape(-1, fit["pooled"].shape[-1])
  x_hold = hold["pooled"].reshape(-1, hold["pooled"].shape[-1])
  groups = np.repeat(np.arange(len(y_fit)), n_s)
  t_fit = np.repeat((y_fit - y_mean) / y_std, n_s)

  w = _fit(x_fit, t_fit)
  pred = _design(x_hold) @ w * y_std + y_mean
  pred_in = _design(x_fit) @ w * y_std + y_mean
  register(
      "probe_latent_ols",
      _per_condition(pred.reshape(-1, n_s)),
      _per_condition(pred_in.reshape(-1, n_s)),
  )

  ridge = _cv_ridge(x_fit, t_fit, groups)
  w = _fit(x_fit, t_fit, ridge)
  pred = _design(x_hold) @ w * y_std + y_mean
  pred_in = _design(x_fit) @ w * y_std + y_mean
  register(
      "probe_latent_ridge",
      _per_condition(pred.reshape(-1, n_s)),
      _per_condition(pred_in.reshape(-1, n_s)),
      {"ridge": ridge},
  )

  # ---- same probe on log10 flux: the target spans two decades
  t_log = np.repeat(np.log10(np.maximum(y_fit, 1e-3)), n_s)
  ridge_log = _cv_ridge(x_fit, t_log, groups)
  w = _fit(x_fit, t_log, ridge_log)
  pred = 10.0 ** (_design(x_hold) @ w)
  pred_in = 10.0 ** (_design(x_fit) @ w)
  register(
      "probe_latent_ridge_log",
      _per_condition(pred.reshape(-1, n_s)),
      _per_condition(pred_in.reshape(-1, n_s)),
      {"ridge": ridge_log},
  )

  # ---- condition-mean pooling: one latent per condition instead of n_samples
  xm_fit, xm_hold = fit["pooled"].mean(1), hold["pooled"].mean(1)
  ridge_m = _cv_ridge(xm_fit, (y_fit - y_mean) / y_std, np.arange(len(y_fit)))
  w = _fit(xm_fit, (y_fit - y_mean) / y_std, ridge_m)
  register(
      "probe_condmean_ridge",
      _design(xm_hold) @ w * y_std + y_mean,
      _design(xm_fit) @ w * y_std + y_mean,
      {"ridge": ridge_m},
  )

  # ---- control: the 4 conditioning scalars the latents were generated from
  c_fit, c_hold = fit["cond"].astype(np.float64), hold["cond"].astype(
      np.float64
  )
  ridge_c = _cv_ridge(c_fit, (y_fit - y_mean) / y_std, np.arange(len(y_fit)))
  w = _fit(c_fit, (y_fit - y_mean) / y_std, ridge_c)
  register(
      "control_conditioning_linear",
      _design(c_hold) @ w * y_std + y_mean,
      _design(c_fit) @ w * y_std + y_mean,
      {"ridge": ridge_c},
  )

  # ---- the route the transport model uses: decode the latents, integrate
  if "flux_samples" in hold.files:
    fit_flux = (
        fit["flux_samples"].mean(1) if "flux_samples" in fit.files else None
    )
    register("flux_integral_route", hold["flux_samples"].mean(1), fit_flux)

  out = os.path.join(_ARGS.out_dir, "gyroflow_probe.json")
  with open(out, "w") as f:
    json.dump(
        {
            "fit_file": FIT_FILE,
            "holdout_file": HOLDOUT_FILE,
            "n_fit": int(len(y_fit)),
            "n_holdout": int(len(y_hold)),
            "n_samples": n_s,
            "latent_features": int(x_fit.shape[1]),
            "results": results,
        },
        f,
        indent=1,
    )
  print("wrote", out)


if __name__ == "__main__":
  if _ARGS.stage == "latents":
    run_latents()
  else:
    run_fit()
