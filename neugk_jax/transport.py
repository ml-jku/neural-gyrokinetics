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

"""Gyroflow (latent flow-matching diffusion) as a TORAX transport model.

Builds on BaseGKWGyaradaxPlugin, which upstream TORAX does not carry yet
(google-deepmind/torax#2189), so this needs the gerkone/torax fork --
install the `transport` extra.

Wraps the neugk-jax "gyroflow" generative pipeline trained on the GyroSwin
255-run nonlinear adiabatic-electron GKW dataset: a frozen Swin5D
autoencoder maps 5D distribution-function snapshots to a latent grid, and a
DiT velocity field — conditioned ONLY on the local operational parameters
(R/L_n, R/L_T, q, s_hat) — is integrated from Gaussian noise to a latent
sample (rectified flow matching). Per rho_match radius: build the local
condition vector from TORAX profiles, draw n_samples snapshots, decode and
denormalize them to df, and compute fluxes with gyaradax's field solve +
flux integrals on the training flux-tube geometry evaluated at the local
(q, s_hat). Sample-averaged fluxes are GKW gyroBohm and are converted to
TORAX units exactly like the other gyaradax plugins (gyaradax_normalization).

The training data never scans radial location or eps (fixed at 0.19), so no
radial-profile information beyond the local scalars is passed to the model,
and the flux-integral geometry keeps the training eps rather than the local
one.

The available (neurips26) checkpoints were trained with the doubled swin
shortcut; neugk-jax carries that as `SwinBlock.legacy_double_shortcut`, which
`build_ae_from_config` enables by default, so no patching is needed here.
"""

import dataclasses
import os
import pickle
from functools import lru_cache
from typing import Annotated, Any, Dict, Literal, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from gyaradax import integrals as gyx_integrals
from torax._src.physics import psi_calculations
from torax._src.torax_pydantic import torax_pydantic
from torax._src.transport_model import gyaradax_base as base_lib
from torax._src.transport_model import gyaradax_diagnostics as diag_lib  # noqa: E501

from neugk_jax import translate as neugk_translate
from neugk_jax.diffusion.flow_matching import euler_sample
from neugk_jax.models import build as neugk_build
from neugk_jax.training import checkpoint as checkpoint_lib

# GyroSwin dataset GKW grid; fixed by the AE/DiT training data
_GYROSWIN_GRID = dict(
    nvpar=32, nmu=8, ns=16, nkx=85, nky=32, ikxspace=5, krhomax=1.4
)
_EPS_TRAIN = 0.19

# GyroSwin scan box; conditions outside it are extrapolation
_RLT_TRAIN = (1.0, 12.0)
_RLN_TRAIN = (0.0, 7.0)
_Q_TRAIN = (1.0, 9.0)
_SHAT_TRAIN = (0.5, 5.0)
_FLUX_CLIP = 1e3

# 1/sqrt(mean latent variance) over the ae327 training latents
_DEFAULT_LATENT_SCALE = 0.047809

# zscore aggregation axes of the training normalization (all but mu)
_STATS_AGG_AXES = (0, 1, 3, 4, 5)


class _StatsUnpickler(pickle.Unpickler):
  """Unpickler resolving upstream neugk stats classes to attribute shims."""

  def find_class(self, module, name):
    if module.split(".")[0] in ("neugk", "neugk_jax"):
      return type(name, (), {})
    return super().find_class(module, name)


@lru_cache(maxsize=4)
def load_df_stats(stats_path: str) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Per-mu (mean, std) of the training df, broadcastable to decoded df."""
  with open(stats_path, "rb") as f:
    raw = _StatsUnpickler(f).load()
  entry = raw["df"]
  get = entry.get if isinstance(entry, dict) else lambda k: getattr(entry, k)
  mean = np.asarray(get("mean"), dtype=np.float64)
  var = np.asarray(get("var"), dtype=np.float64)
  agg_mean = mean.mean(axis=_STATS_AGG_AXES, keepdims=True)
  agg_var = var.mean(axis=_STATS_AGG_AXES, keepdims=True) + mean.var(
      axis=_STATS_AGG_AXES, keepdims=True
  )
  return jnp.asarray(agg_mean), jnp.asarray(np.sqrt(agg_var))


def _load_ckpt(template, path: str):
  """Load .eqx directly; anything else translates via torch (needs torch)."""
  if path.endswith(".eqx"):
    return checkpoint_lib.load_model_only(path, template)
  return neugk_translate.load_or_translate(template, path)


@lru_cache(maxsize=2)
def _load_models(
    ae_checkpoint_path: str,
    dit_checkpoint_path: str,
    legacy_swin_residual: bool = True,
):
  """Frozen AE + DiT pair, each built from the config.yaml next to it."""
  ae_cfg = os.path.join(os.path.dirname(ae_checkpoint_path), "config.yaml")
  ae = _load_ckpt(
      neugk_build.build_ae_from_config(
          ae_cfg,
          key=jax.random.PRNGKey(0),
          legacy_double_shortcut=legacy_swin_residual,
      ),
      ae_checkpoint_path,
  )
  dit_cfg = os.path.join(os.path.dirname(dit_checkpoint_path), "config.yaml")
  dit = _load_ckpt(
      neugk_build.build_dit_from_config(
          dit_cfg, ae, key=jax.random.PRNGKey(0)
      ),
      dit_checkpoint_path,
  )
  return ae, dit


def local_conditions(rho_idx, ql_inputs, core_profiles, geo, clip=True):
  """Local (rln, rlt, q, shat) scalars, optionally clipped to the training box.

  Only local operational parameters — the model was never trained with any
  radial-profile or rho-dependent shape information.

  Clipping keeps the conditioning inside the GyroSwin scan box, but the box
  floor s_hat = 0.5 collapses every reversed-shear radius onto one point:
  measured at (rlt 7, rln 1.2, q 1.7), the raw model gives Q = 1.7 GB at
  s_hat = -1 versus 14.3 at s_hat = 0.5, i.e. it does reproduce reversed-shear
  stabilisation (nonlinear gyaradax gives exactly 0 there) and clipping throws
  that away. Pass clip=False to condition on the true local values.
  """
  smag_face = psi_calculations.calc_s_rmid(geo, core_profiles.psi)
  rln = ql_inputs.lref_over_lne[rho_idx]
  rlt = ql_inputs.lref_over_lti[rho_idx]
  q = core_profiles.q_face[rho_idx]
  shat = smag_face[rho_idx]
  if clip:
    rln = jnp.clip(rln, *_RLN_TRAIN)
    rlt = jnp.clip(rlt, *_RLT_TRAIN)
    q = jnp.clip(q, *_Q_TRAIN)
    shat = jnp.clip(shat, *_SHAT_TRAIN)
  return rln, rlt, q, shat


def denormalized_df_to_spectral(df_norm: jnp.ndarray) -> jnp.ndarray:
  """Denormalized separate-zf df (..., 4|2, vpar, mu, s, x, y) -> gyaradax spectral.

  Recombines the zonal split, forms the complex analytic field and applies the
  GyroSwin dataset's spatial->spectral convention. Batch axes are optional.
  """
  if df_norm.shape[-6] == 4:
    # recombine the separate-zf channel-of-4 layout back to (re, im)
    df_norm = df_norm[..., :2, :, :, :, :, :] + df_norm[..., 2:, :, :, :, :, :]
  df = df_norm[..., 0, :, :, :, :, :] + 1j * df_norm[..., 1, :, :, :, :, :]
  spec = jnp.fft.fftn(df, axes=(-2, -1), norm="forward")
  return jnp.fft.ifftshift(spec, axes=-2)


def spectral_to_denormalized_df(spec: jnp.ndarray) -> jnp.ndarray:
  """Exact inverse of `denormalized_df_to_spectral` (separate-zf, 4 channels).

  gyaradax evolves df in (kx, ky) with kx zero-centered and ky one-sided, so
  the inverse transform yields a *complex* real-space field whose real and
  imaginary parts are the dataset's two channels, matching the preprocessed
  training bins.
  """
  real = jnp.fft.ifftn(
      jnp.fft.fftshift(spec, axes=-2), axes=(-2, -1), norm="forward"
  )
  df = jnp.stack([real.real, real.imag], axis=-6)
  # separate_zf: [zf, df - zf] with zf the ky-mean broadcast back (neugk_jax.utils)
  zf = jnp.broadcast_to(df.mean(axis=-1, keepdims=True), df.shape)
  return jnp.concatenate([zf, df - zf], axis=-6)


def encode_spectral_df(ae, spec: jnp.ndarray, df_stats) -> jnp.ndarray:
  """gyaradax 5D spectral df -> gyroflow latents (what `euler_sample` returns).

  The exact mirror of the decode half used for the fluxes: spectral -> real
  space -> separate-zf 4-channel layout -> per-mu z-score with the shipped
  training stats -> AE encoder. `spec` is one unbatched (vpar, mu, s, kx, ky)
  snapshot on the GyroSwin grid; `df_stats` is `load_df_stats`' (mean, std).
  """
  mean, std = df_stats
  df_norm = (spectral_to_denormalized_df(spec) - mean) / std
  return ae.encode(df_norm.astype(jnp.float32))[0]


def decode_latents_to_spectral(
    ae, latents: jnp.ndarray, df_stats
) -> jnp.ndarray:
  """gyroflow latents -> gyaradax 5D spectral df (inverse of encode_spectral_df)."""
  mean, std = df_stats
  df_norm = ae.decode(latents)["df"]
  return denormalized_df_to_spectral(df_norm.astype(jnp.float64) * std + mean)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=False)
class GyaradaxGyroflowTransportModel(base_lib.BaseGKWGyaradaxPlugin):
  """Diffusion-sampled turbulence snapshots -> gyaradax flux integrals."""

  ae_checkpoint_path: str = ""
  dit_checkpoint_path: str = ""
  df_stats_path: str = ""
  n_samples: int = 4
  sampler_steps: int = 10
  sampler_method: str = "euler"
  latent_scale: float = _DEFAULT_LATENT_SCALE
  eps_train: float = _EPS_TRAIN
  seed: int = 0
  legacy_swin_residual: bool = True
  clip_conditions: bool = False
  latent_dump_dir: str = ""
  latent_dump_max_calls: int = 0
  latent_dump_decoded: int = 0
  latent_dump_decoded_max_calls: int = 0

  @classmethod
  def from_config(cls, cfg) -> "GyaradaxGyroflowTransportModel":
    for name in ("ae_checkpoint_path", "dit_checkpoint_path", "df_stats_path"):
      if not getattr(cfg, name):
        raise ValueError(f"gyaradax-gyroflow requires '{name}'")
    model = cls(
        rho_match=tuple(cfg.rho_match),
        ae_checkpoint_path=cfg.ae_checkpoint_path,
        dit_checkpoint_path=cfg.dit_checkpoint_path,
        df_stats_path=cfg.df_stats_path,
        n_samples=cfg.n_samples,
        sampler_steps=cfg.sampler_steps,
        sampler_method=cfg.sampler_method,
        latent_scale=(
            cfg.latent_scale
            if cfg.latent_scale is not None
            else _DEFAULT_LATENT_SCALE
        ),
        eps_train=cfg.eps_train,
        seed=cfg.seed,
        legacy_swin_residual=cfg.legacy_swin_residual,
        clip_conditions=getattr(cfg, "clip_conditions", False),
        diagnostics_path=getattr(cfg, "diagnostics_path", None) or "",
        latent_dump_dir=getattr(cfg, "latent_dump_dir", None) or "",
        latent_dump_max_calls=getattr(cfg, "latent_dump_max_calls", 0),
        latent_dump_decoded=getattr(cfg, "latent_dump_decoded", 0),
        latent_dump_decoded_max_calls=getattr(
            cfg, "latent_dump_decoded_max_calls", 0
        ),
        **_GYROSWIN_GRID,
    )
    # warm the host-side caches (topology, checkpoints, stats) outside jit
    _ = model.topology
    _ = model.models
    _ = model.df_stats
    return model

  @property
  def models(self):
    return _load_models(
        self.ae_checkpoint_path,
        self.dit_checkpoint_path,
        self.legacy_swin_residual,
    )

  @property
  def df_stats(self):
    return load_df_stats(self.df_stats_path)

  def _sample_latents(self, cond: jnp.ndarray, key) -> jnp.ndarray:
    """n_samples flow-matching latents for one condition (pre-decoder)."""
    _ae, dit = self.models
    cond_b = jnp.broadcast_to(cond, (self.n_samples, cond.shape[-1]))
    return euler_sample(
        lambda x, t, c: dit(x, t, c),
        key=key,
        shape=(self.n_samples, *dit.latent_shape),
        cond=cond_b,
        steps=self.sampler_steps,
        latent_scale=self.latent_scale,
        method=self.sampler_method,
    )

  def _decode_latents(self, latents: jnp.ndarray) -> jnp.ndarray:
    """Latents -> normalized separate-zf df snapshots."""
    ae, _dit = self.models
    # sequential over samples: vmap here multiplies with the radii vmap and OOMs at n_samples>=16
    return jax.lax.map(lambda z: ae.decode(z)["df"], latents)

  def _sample_normalized_df(self, cond: jnp.ndarray, key) -> jnp.ndarray:
    """n_samples normalized separate-zf df snapshots for one condition."""
    return self._decode_latents(self._sample_latents(cond, key))

  def _per_sample_fluxes(
      self, df_norm: jnp.ndarray, geom: Dict[str, Any]
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Per-sample (pflux, eflux) in GKW gyroBohm units, shape (n_samples,)."""
    mean, std = self.df_stats
    spec = denormalized_df_to_spectral(df_norm.astype(jnp.float64) * std + mean)
    gt = gyx_integrals.geom_tensors(geom)

    def one(sample):
      _phi, (pflux, eflux, _vflux) = gyx_integrals.get_integrals(
          sample, geom, adiabatic_electrons=True, geom=gt
      )
      return pflux, eflux

    return jax.lax.map(one, spec)

  def _fluxes_from_df(
      self, df_norm: jnp.ndarray, geom: Dict[str, Any]
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Sample-averaged (pflux, eflux) in GKW gyroBohm units."""
    pflux, eflux = self._per_sample_fluxes(df_norm, geom)
    return jnp.mean(pflux), jnp.mean(eflux)

  def _latent_payload(self, latents, df_norm) -> Dict[str, jnp.ndarray]:
    """Latents, and optionally the paired decoded snapshots in fp16."""
    out = {"latents": latents}
    n = int(self.latent_dump_decoded)
    if n > 0:
      mean, std = self.df_stats
      out["df_norm"] = df_norm[:n].astype(jnp.float16)
      out["df_norm_mean"] = jnp.asarray(mean, jnp.float32)
      out["df_norm_std"] = jnp.asarray(std, jnp.float32)
    return out

  def _per_radius_from_profiles(
      self, rho_idx, ql_inputs, core_profiles, geo
  ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    rln, rlt, q, shat = local_conditions(
        rho_idx, ql_inputs, core_profiles, geo, clip=self.clip_conditions
    )
    # sorted-name order (dg, itg, q, s_hat) — the order the DiT trained with
    cond = jnp.stack([rln, rlt, q, shat]).astype(jnp.float32)
    # flux-tube geometry at the local (q, s_hat); eps pinned to training value
    geom = base_lib.gyaradax_geometry_at(
        q=q,
        shat=shat,
        eps=jnp.asarray(self.eps_train),
        config=self,
        topology=self.topology,
    )
    key = jax.random.fold_in(jax.random.PRNGKey(self.seed), rho_idx)
    if self.latent_dump_dir:
      # keep the latents alive for the dump instead of only their decoding
      latents = self._sample_latents(cond, key)
      df_norm = self._decode_latents(latents)
    else:
      latents, df_norm = None, self._sample_normalized_df(cond, key)
    pflux, eflux = self._per_sample_fluxes(df_norm, geom)
    q_i = jnp.clip(jnp.mean(eflux), 0.0, _FLUX_CLIP)
    diag_lib.record(
        self.sink,
        {
            "model": type(self).__name__,
            "n_samples": self.n_samples,
            "sampler_steps": self.sampler_steps,
            "sampler_method": self.sampler_method,
        },
        {
            "rho_idx": rho_idx,
            "rho_face": jnp.asarray(geo.rho_face_norm)[rho_idx],
            "cond": cond,
            "qi_gb": q_i,
            "qi_gb_samples": eflux,
            "qi_gb_std": jnp.std(eflux),
            "pflux_gb": jnp.mean(pflux),
        },
        latents=None if latents is None else self._latent_payload(latents, df_norm),
    )
    # adiabatic electrons: qe aliased to qi, no electron particle flux
    return q_i, q_i, jnp.asarray(0.0)


class GyaradaxGyroflowConfig(base_lib.BaseGKWGyaradaxConfig):
  """Config for the gyaradax-gyroflow transport model.

  Attributes:
    model_name: transport model selector. Hardcoded to 'gyaradax-gyroflow'.
    ae_checkpoint_path: Swin5D autoencoder checkpoint (.eqx, or torch .pth
      which is translated on the fly and requires torch). A config.yaml
      describing the architecture must sit next to the checkpoint.
    dit_checkpoint_path: latent DiT checkpoint; same conventions as the AE.
    df_stats_path: pickle with the training df normalization statistics
      (upstream RunningMeanStd format or a {'df': {'mean', 'var'}} dict);
      used to denormalize decoded snapshots before the flux integral.
    n_samples: diffusion samples drawn (and averaged over) per radius.
    sampler_steps: flow-matching integration steps per sample. 10 is the
      calibrated value: 20 raises the flux ~12 percent and over-predicts.
    sampler_method: ODE scheme ('euler', 'midpoint', 'heun', 'rk4').
    latent_scale: latent whitening scale of the flow-matching model; None
      uses the value measured on the ae327 training latents.
    eps_train: inverse aspect ratio of the training dataset, used for the
      flux-integral geometry (the model is not conditioned on eps).
    seed: PRNG seed for the diffusion sampler (folded with the radius index).
    legacy_swin_residual: build the AE with the doubled swin shortcut
      forward the neurips26 checkpoints were trained with (process-global
      patch of neugk-jax; without it those weights decode to near-zero
      fields). Set False for checkpoints retrained after the upstream fix.
    clip_conditions: clamp the conditioning into the training hull before
      sampling. Off by default: the chi conversion divides the returned flux by
      the TRUE gradient, so clamping the gradient up asks the model about a
      different plasma than the one its answer is converted for. At a radius
      with R/L_Ti = 0.05 the clamp reports the flux for R/L_Ti = 1.0 (1.89 vs
      1.40 gB), inflating chi further on top of an already diverging ratio.
      Enabling it also pins s_hat at the hull edge of 0.5, hiding the model's
      own reversed-shear behaviour.
    diagnostics_path: when set, append one JSON row per (transport call,
      radius) to this `.jsonl`: conditioning vector, pre-unit-conversion
      fluxes, the per-sample flux spread, sampler steps / n_samples and
      wallclock. Written host-side through io_callback, so it is jit-safe.
    latent_dump_dir: when set, write one `.npz` per (transport call, radius)
      into this directory holding the sampled latents `euler_sample` returned
      (pre-decoder) alongside the conditioning vector, the radius and the
      fluxes they decoded to. Sized `n_samples * prod(latent_shape)` floats
      per file (~4.7 MB at n_samples=16 on the neurips26 checkpoints), so cap
      long runs with `latent_dump_max_calls`.
    latent_dump_decoded: how many of the n_samples decoded 5D snapshots to save
      alongside the latents, in fp16. Each costs ~89 MB per call and radius, so
      keep it small; 0 saves latents only.
    latent_dump_decoded_max_calls: stop saving decoded snapshots after this many
      transport calls while latents keep being written. 0 means no separate
      limit.
    latent_dump_max_calls: stop dumping latents after this many transport
      calls; 0 (default) means no cap.
    Remaining attributes are inherited from BaseGKWGyaradaxConfig.
  """

  model_name: Annotated[
      Literal["gyaradax-gyroflow"], torax_pydantic.JAX_STATIC
  ] = "gyaradax-gyroflow"
  ae_checkpoint_path: Annotated[str, torax_pydantic.JAX_STATIC] = ""
  dit_checkpoint_path: Annotated[str, torax_pydantic.JAX_STATIC] = ""
  df_stats_path: Annotated[str, torax_pydantic.JAX_STATIC] = ""
  n_samples: Annotated[int, torax_pydantic.JAX_STATIC] = 4
  sampler_steps: Annotated[int, torax_pydantic.JAX_STATIC] = 20
  sampler_method: Annotated[str, torax_pydantic.JAX_STATIC] = "euler"
  latent_scale: Annotated[Optional[float], torax_pydantic.JAX_STATIC] = None
  eps_train: Annotated[float, torax_pydantic.JAX_STATIC] = _EPS_TRAIN
  seed: Annotated[int, torax_pydantic.JAX_STATIC] = 0
  legacy_swin_residual: Annotated[bool, torax_pydantic.JAX_STATIC] = True
  clip_conditions: Annotated[bool, torax_pydantic.JAX_STATIC] = False
  latent_dump_dir: Annotated[Optional[str], torax_pydantic.JAX_STATIC] = None
  latent_dump_max_calls: Annotated[int, torax_pydantic.JAX_STATIC] = 0
  latent_dump_decoded: Annotated[int, torax_pydantic.JAX_STATIC] = 0
  latent_dump_decoded_max_calls: Annotated[int, torax_pydantic.JAX_STATIC] = 0

  def build_transport_model(self) -> GyaradaxGyroflowTransportModel:
    return GyaradaxGyroflowTransportModel.from_config(self)

