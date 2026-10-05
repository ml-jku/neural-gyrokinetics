"""Model construction from configs: ``Swin5DAE`` / ``Swin5DVQVAE``, ``DiT`` and ``GyroSwinMultitask``.

Every builder takes a YAML path, an OmegaConf config or a mapping with a ``model`` section
(and a ``dataset`` section for the resolution and the zonal-flow layout); the run builders
(:func:`build_ae`, :func:`build_dit`, :func:`build_gyroswin`) take an in-memory run config
and the dataset it trains on. Config keys the port does not implement raise.
"""

from __future__ import annotations

import copy
from functools import partial
from typing import Mapping, Optional, Sequence

import jax
import jax.numpy as jnp

from neugk_jax.utils import to_dict

# df grid (vp, mu, s, x, y) of the cyclone dataset and the release checkpoints
RESOLUTION = (32, 8, 16, 85, 32)


def force_f32(model):
    """Cast every f64 ``jax.Array`` leaf in ``model`` down to f32.

    ``eqx.nn.Linear`` initialises with the JAX default dtype, which is f64
    whenever ``jax_enable_x64`` is on (gyaradax flips it on import).
    """
    return jax.tree_util.tree_map(
        lambda x: (
            x.astype(jnp.float32) if isinstance(x, jax.Array) and x.dtype == jnp.float64 else x
        ),
        model,
    )


def validate_keys(name: str, section: Mapping, accepted, fixed: Optional[Mapping] = None) -> None:
    """Raise on keys of config ``section`` outside ``accepted`` and ``fixed``, or a ``fixed``
    key (one supported value) set to anything else."""
    fixed = fixed or {}
    unknown = set(section) - set(accepted) - set(fixed)
    if unknown:
        raise ValueError(f"unknown {name} keys: {sorted(unknown)}")
    for k, v in fixed.items():
        if k in section and section[k] != v:
            raise NotImplementedError(f"{name}.{k}={section[k]!r} (only {v!r} is supported)")


def _df_channels(dataset: Mapping) -> int:
    # real/imag, doubled by the zonal-flow split
    return 2 * (2 if dataset.get("separate_zf", False) else 1)


def _legacy_shortcut(mcfg: Mapping, override: Optional[bool]) -> bool:
    return bool(mcfg.get("legacy_swin_shortcut", False)) if override is None else override


_AE_VIT_KEYS = {
    "num_heads",
    "depth",
    "drop_path",
    "qkv_bias",
    "qk_norm",
    "use_rpb",
    "gated_attention",
    "modulation",
    "gradient_checkpoint",
}
_AE_PATCH_KEYS = {
    "patch_size",
    "window_size",
    "merging_depth",
    "unmerging_depth",
    "merging_hidden_ratio",
    "unmerging_hidden_ratio",
    "c_multiplier",
}
_AE_BOTTLENECK_KEYS = {"dim", "depth", "num_heads", "normalized_latent", "norm_learnable"}
_NO_PE = {"use_abs_pe": False, "use_rope": False}


def ae_conditioning(mcfg: Mapping) -> tuple[list[str], list[str]]:
    # (encoder, decoder) condition names, each defaulting to model.conditioning
    cond = list(mcfg.get("conditioning") or [])
    enc = mcfg.get("encoder_conditioning")
    dec = mcfg.get("decoder_conditioning")
    return (cond if enc is None else list(enc)), (cond if dec is None else list(dec))


def _check_ae(mcfg: Mapping, conditioned: bool) -> None:
    vit = mcfg.get("vit", {})
    unsupported = {
        f"model.model_type={mcfg.get('model_type')!r}": mcfg.get("model_type", "ae")
        not in ("ae", "vqvae"),
        f"model.vit.modulation={vit.get('modulation')!r}": conditioned
        and vit.get("modulation", "dit") != "dit",
        "model.flux_head": bool((mcfg.get("flux_head") or {}).get("enable")),
        f"model.act_fn={mcfg.get('act_fn')!r}": mcfg.get("act_fn", "GELU") != "GELU",
        f"model.norm_fn={mcfg.get('norm_fn')!r}": mcfg.get("norm_fn", "RMSNorm")
        not in ("RMSNorm", "LayerNorm"),
    }
    bad = [k for k, v in unsupported.items() if v]
    if bad:
        raise NotImplementedError(", ".join(bad))


def build_ae_from_config(
    cfg_path,
    *,
    key,
    resolution: Optional[Sequence[int]] = None,
    legacy_double_shortcut: Optional[bool] = None,
):
    """``Swin5DAE`` (``Swin5DVQVAE`` for ``model.model_type: vqvae``) of a config.

    ``model.encoder_conditioning`` / ``model.decoder_conditioning`` condition each path; an
    absent ``model.norm_fn`` is RMSNorm for the AE and LayerNorm for the VQ-VAE.
    ``legacy_double_shortcut`` (the doubled swin residual) defaults to
    ``model.legacy_swin_shortcut``, else False.
    """
    from neugk_jax.pinc import Swin5DAE, Swin5DVQVAE

    cfg = to_dict(cfg_path)
    mcfg = cfg["model"]
    vq = mcfg.get("model_type") == "vqvae"
    enc_cond, dec_cond = ae_conditioning(mcfg)
    _check_ae(mcfg, bool(enc_cond or dec_cond))
    vit, patch, bn = mcfg.get("vit", {}), mcfg.get("patch", {}), mcfg.get("bottleneck", {})
    validate_keys("model.vit", vit, _AE_VIT_KEYS, _NO_PE)
    validate_keys("model.patch", patch, _AE_PATCH_KEYS)
    validate_keys("model.bottleneck", bn, _AE_BOTTLENECK_KEYS)
    dataset = cfg.get("dataset", {})
    depth = vit["depth"]
    cls = partial(Swin5DVQVAE, vq_config=mcfg.get("vq") or {}) if vq else Swin5DAE
    return force_f32(
        cls(
            space=5,
            decouple_mu=mcfg.get("decouple_mu", True),
            dim=mcfg["latent_dim"],
            base_resolution=list(resolution or dataset.get("resolution") or RESOLUTION),
            in_channels=dataset.get("in_channels", _df_channels(dataset)),
            out_channels=dataset.get("out_channels", _df_channels(dataset)),
            patch_size=patch["patch_size"],
            window_size=patch["window_size"],
            depth=depth,
            num_heads=vit["num_heads"],
            num_layers=mcfg.get(
                "num_layers", len(depth) if isinstance(depth, (list, tuple)) else 4
            ),
            bottleneck_dim=bn.get("dim"),
            bottleneck_depth=bn.get("depth", 2),
            bottleneck_num_heads=bn.get("num_heads", 2),
            hidden_mlp_ratio=mcfg.get("hidden_mlp_ratio", 2.0),
            merging_hidden_ratio=patch.get("merging_hidden_ratio", 8.0),
            unmerging_hidden_ratio=patch.get("unmerging_hidden_ratio", 8.0),
            merging_depth=patch.get("merging_depth", 2),
            unmerging_depth=patch.get("unmerging_depth", 2),
            c_multiplier=int(patch.get("c_multiplier", 2)),
            drop_path=float(vit.get("drop_path", 0.1)),
            normalized_latent=bn.get("normalized_latent", False),
            qkv_bias=vit.get("qkv_bias", False),
            qk_norm=vit.get("qk_norm", True),
            use_rpb=vit.get("use_rpb", True),
            gated_attention=vit.get("gated_attention", False),
            norm_affine=False,
            rms_norm=mcfg.get("norm_fn", "LayerNorm" if vq else "RMSNorm") == "RMSNorm",
            legacy_double_shortcut=_legacy_shortcut(mcfg, legacy_double_shortcut),
            use_checkpoint=bool(vit.get("gradient_checkpoint", False)),
            encoder_conditioning=enc_cond,
            decoder_conditioning=dec_cond,
            key=key,
        )
    )


def build_dit_from_config(cfg_path, ae, *, key):
    """``DiT`` of a config whose latent grid and width match the AE bottleneck."""
    from neugk_jax.diffusion.dit import DiT

    mcfg = to_dict(cfg_path)["model"]
    vit = mcfg["vit"]
    return force_f32(
        DiT(
            z_dim=int(ae.bottleneck_dim),
            dim=mcfg["latent_dim"],
            grid_size=tuple(ae.bottleneck_grid_size),
            depth=vit["depth"],
            num_heads=vit["num_heads"],
            n_cond=len(mcfg.get("conditioning", []) or []),
            key=key,
            mlp_ratio=float(vit.get("mlp_ratio", 2.0)),
            drop_path=float(vit.get("drop_path", 0.1)),
        )
    )


_GYROSWIN_SWIN_KEYS = {
    "patch_size",
    "window_size",
    "num_heads",
    "depth",
    "gradient_checkpoint",
    "merging_hidden_ratio",
    "unmerging_hidden_ratio",
    "c_multiplier",
    "patch_skip",
    "modulation",
    "use_rpb",
    "qk_norm",
    "gated_attention",
    "norm_fn",
    "flux_reduce",
    "flux_num_heads",
    "flux_depth",
    "flux_conditioning",
    "detach_flux_latents",
    "detach_phi_cross_latents",
    "attn_drop",
    "flux_drop",
    # phi grids are derived from the df grids
    "phi_patch_size",
    "phi_window_size",
    # no effect on the model: unused by the multitask model, or init-only
    "norm_output",
    "drop_path",
    "init_weights",
    "patching_init_weights",
    "cond_init_weights",
}
_GYROSWIN_FIXED_SWIN = {
    "swin_bottleneck": True,
    "latent_cross_attn": True,
    "use_abs_pe": False,
    "use_rope": False,
    "cosine_attn": False,
    "act_fn": "GELU",
    "flux_depth": 1,
    "flux_reduce": "max",
}


def _check_gyroswin(mcfg: dict, dataset: dict, training: dict) -> None:
    swin = mcfg["swin"]
    validate_keys("model.swin", swin, _GYROSWIN_SWIN_KEYS, _GYROSWIN_FIXED_SWIN)
    unsupported = {
        "model.swin.modulation": swin.get("modulation", "film") not in ("film", "dit"),
        "model.swin.norm_fn": swin.get("norm_fn", "LayerNorm") not in ("LayerNorm", "RMSNorm"),
        "model.bundle_seq_length > 1": int(mcfg.get("bundle_seq_length", 1)) > 1,
        "model.num_layers != 1": int(mcfg.get("num_layers", 1)) != 1,
        "dataset.real_potens=false (complex phi)": not dataset.get("real_potens", True),
        "training.predict_delta": bool(training.get("predict_delta", False)),
        "training.pushforward unrolls": any(
            int(u) > 0 for u in (training.get("pushforward") or {}).get("unrolls") or []
        ),
    }
    bad = [k for k, v in unsupported.items() if v]
    if bad:
        raise NotImplementedError(", ".join(bad))


def build_gyroswin_from_config(
    cfg_path,
    *,
    key,
    resolution: Optional[Sequence[int]] = None,
    legacy_double_shortcut: Optional[bool] = None,
):
    """``GyroSwinMultitask`` of a config; outputs and the flux head follow the loss config,
    ``legacy_double_shortcut`` defaults to ``model.legacy_swin_shortcut``, else False."""
    from neugk_jax.gyroswin.models.gyroswin import GyroSwinMultitask
    from neugk_jax.training.loss_scheduler import LossConfig

    cfg = to_dict(cfg_path)
    mcfg = cfg["model"] if "model" in cfg else cfg
    swin = mcfg["swin"]
    dataset = cfg.get("dataset", {}) or {}
    _check_gyroswin(mcfg, dataset, cfg.get("training", {}) or {})
    losses = LossConfig(
        mcfg.get("loss_weights"), mcfg.get("extra_loss_weights"), mcfg.get("loss_scheduler")
    )
    outputs = losses.outputs or ("df", "phi")
    if "phi" not in outputs:
        raise NotImplementedError("gyroswin without a phi output")
    channels = _df_channels(dataset)
    model = GyroSwinMultitask(
        dim=mcfg["latent_dim"],
        df_base_resolution=resolution or dataset.get("resolution") or RESOLUTION,
        df_patch_size=swin["patch_size"],
        df_window_size=swin["window_size"],
        depth=swin["depth"],
        num_heads=swin["num_heads"],
        in_channels=channels,
        out_channels=channels,
        num_layers=int(mcfg.get("num_layers", 1)),
        c_multiplier=swin.get("c_multiplier", 2),
        merging_hidden_ratio=swin.get("merging_hidden_ratio", 4.0),
        unmerging_hidden_ratio=swin.get("unmerging_hidden_ratio", 8.0),
        decouple_mu=mcfg.get("decouple_mu", True),
        patch_skip=swin.get("patch_skip", True),
        use_rpb=swin.get("use_rpb", True),
        # gyroswin swin blocks default qk_norm/gated-attention off, unlike the ae
        qk_norm=swin.get("qk_norm", False),
        gated_attention=swin.get("gated_attention", False),
        cond_mode=swin.get("modulation", "film"),
        rms_norm=(swin.get("norm_fn") == "RMSNorm"),
        drop_path=float(mcfg.get("drop_path") if mcfg.get("drop_path") is not None else 0.1),
        flux_num_heads=swin.get("flux_num_heads", 4),
        flux_depth=swin.get("flux_depth", 1),
        flux_conditioning=bool(swin.get("flux_conditioning", False)),
        flux_drop=float(swin.get("flux_drop", 0.1)),
        attn_drop=float(swin.get("attn_drop", 0.1)),
        detach_flux_latents=bool(swin.get("detach_flux_latents", False)),
        detach_phi_cross_latents=bool(swin.get("detach_phi_cross_latents", False)),
        use_phi="phi" in outputs,
        flux_key=losses.flux_key,
        n_cond=len(mcfg.get("conditioning", []) or []),
        use_checkpoint=bool(swin.get("gradient_checkpoint", False)),
        legacy_double_shortcut=_legacy_shortcut(mcfg, legacy_double_shortcut),
        key=key,
    )
    return force_f32(model)


# swin keys of the release configs that no model code reads
_RELEASE_DEAD_SWIN_KEYS = ("flux_layernorms",)


def release_config(cfg_path, *, resolution: Optional[Sequence[int]] = None) -> dict:
    """Builder config of a GyroSwin release checkpoint config (``ml-jku/gyroswin_*``).

    Drops the swin keys no model code reads, sets ``legacy_swin_shortcut=False`` (the
    release checkpoints use the single swin residual), and keeps ``separate_zf`` /
    ``real_potens`` of the dataset section with ``resolution`` (default :data:`RESOLUTION`).
    """
    cfg = copy.deepcopy(to_dict(cfg_path))
    mcfg = cfg["model"]
    for k in _RELEASE_DEAD_SWIN_KEYS:
        mcfg["swin"].pop(k, None)
    mcfg["legacy_swin_shortcut"] = False
    ds = cfg.get("dataset") or {}
    dataset = {
        "separate_zf": bool(ds.get("separate_zf", True)),
        "real_potens": bool(ds.get("real_potens", True)),
        "resolution": [int(r) for r in (resolution or RESOLUTION)],
    }
    return {"model": mcfg, "dataset": dataset}


def build_release_gyroswin(cfg_path, *, key, resolution: Optional[Sequence[int]] = None):
    return build_gyroswin_from_config(release_config(cfg_path, resolution=resolution), key=key)


def run_config(cfg, ds=None) -> dict:
    """``{"model", "dataset", "training"}`` plain dict of a run config; ``ds`` fixes resolution and zf."""
    out = {k: to_dict(cfg.get(k)) for k in ("model", "dataset", "training")}
    if ds is not None:
        out["dataset"]["resolution"] = [int(r) for r in ds.resolution]
        out["dataset"]["separate_zf"] = bool(ds.separate_zf)
    return out


def build_ae(cfg, ds, *, key):
    return build_ae_from_config(run_config(cfg, ds), key=key)


def build_dit(cfg, ae, *, key):
    return build_dit_from_config(run_config(cfg), ae, key=key)


def build_gyroswin(cfg, ds, *, key):
    return build_gyroswin_from_config(run_config(cfg, ds), key=key)
