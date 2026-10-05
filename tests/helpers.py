"""Shared test helpers: synthetic cyclone trajectories, tiny AE configs and models."""

from __future__ import annotations

import pickle
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from omegaconf import OmegaConf

RES = (4, 4, 4, 16, 8)
COND = jnp.asarray([1.0, 2.0, 1.4, 0.8])

# torch module names of a few adapted linears of the tiny ae
TARGETS = [
    "dec_cond_embed.mlp.0",
    "middle_post.blocks.0.attn.qkv",
    "up_blocks.0.swin_att.blocks.1.mlp.mlp.3",
    "unpatch.expansion.mlp.0",
    "down_blocks.0.swin_att.blocks.0.attn.rpb.cpb_mlp.mlp.3",
]


def make_geometry(resolution):
    """Shape-consistent single-species geometry for gyaradax on a tiny grid."""
    vp, mu, s, x, y = resolution
    ones = ("mas", "tmp", "de", "d2X", "signz", "signB", "vthrat")
    return {
        "krho": np.arange(y, dtype=np.float64) * 0.5,  # ky=0 zonal mode at index 0
        "kxrh": (np.arange(x, dtype=np.float64) - x // 2) * 0.4,
        "ints": np.full(s, 1.0 / s, dtype=np.float64),
        "intmu": np.linspace(0.1, 0.4, mu, dtype=np.float64),
        "intvp": np.full(vp, 0.5, dtype=np.float64),
        "vpgr": np.linspace(-1.5, 1.5, vp, dtype=np.float64),
        "mugr": np.linspace(0.1, 1.6, mu, dtype=np.float64),
        "bn": np.ones(s, dtype=np.float64),
        "ffun": np.ones(s, dtype=np.float64),
        "efun": np.ones(s, dtype=np.float64),
        "rfun": np.ones(s, dtype=np.float64),
        "bt_frac": np.ones(s, dtype=np.float64),
        "parseval": np.where(np.arange(y) == 0, 1.0, 2.0).astype(np.float64),
        "little_g": np.tile(np.array([1.0, 0.0, 1.0]), (s, 1)),
        **{k: np.ones((1,), dtype=np.float64) for k in ones},
    }


def make_traj(root: Path, name: str, *, n_t: int, resolution=RES, **meta_extra) -> dict:
    """Random-df trajectory ``<name>_ifft_realpotens`` in the numpy layout; returns its metadata."""
    traj = Path(root) / f"{name}_ifft_realpotens"
    data = traj / "data"
    data.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(hash(name) & 0xFFFFFFFF)
    for t in range(n_t):
        rng.standard_normal((2, *resolution)).astype(np.float32).tofile(
            data / f"timestep_{t:05d}.bin"
        )
    meta = {
        "timesteps": np.arange(n_t, dtype=np.float64),
        "flux": rng.standard_normal(n_t).astype(np.float32),
        "ion_temp_grad": np.array([2.3], dtype=np.float32),
        "density_grad": np.array([1.1], dtype=np.float32),
        "s_hat": np.array([0.8], dtype=np.float32),
        "q": np.array([1.4], dtype=np.float32),
        "resolution": np.array(resolution),
        "ds": np.float64(0.0625),
        "geometry": make_geometry(resolution),
        **meta_extra,
    }
    with open(traj / "metadata.pkl", "wb") as f:
        pickle.dump(meta, f)
    return meta


def tiny_ae_cfg(path, resolution=RES, out_path=None):
    """Runner config of a tiny unconditioned AE on ``iteration_0`` (train) / ``iteration_1`` (val)."""
    patch = {"patch_size": [2, 0, 2, 4, 2], "window_size": [2, 0, 2, 2, 2], "merging_depth": 1}
    patch.update(unmerging_depth=1, merging_hidden_ratio=2.0, unmerging_hidden_ratio=2.0)
    vit = {"num_heads": [2], "depth": [1], "use_rpb": False, "gated_attention": False}
    vit.update(qk_norm=False, qkv_bias=False)
    training = {"batch_size": 1, "n_epochs": 1, "learning_rate": 3e-4, "weight_decay": 0.0}
    training.update(final_learning_rate=1e-6, clip_grad=True, clip_to=1.0, exclude_from_wd=[])
    return OmegaConf.create(
        {
            "workflow": "ae",
            "seed": 0,
            "output_path": str(out_path or Path(path) / "out"),
            "model": {
                "name": "ae",
                "decouple_mu": True,
                "latent_dim": 16,
                "patch": {**patch, "c_multiplier": 1},
                "vit": vit,
                "bottleneck": {"dim": 8, "depth": 1, "num_heads": 2, "normalized_latent": False},
                "hidden_mlp_ratio": 2.0,
            },
            "dataset": {
                "name": "cyclone",
                "path": str(path),
                "backend": "numpy",
                "training_trajectories": "iteration_0",
                "validation_trajectories": "iteration_1",
                "input_fields": ["df"],
                "conditions": ["itg", "dg", "s_hat", "q"],
                "separate_zf": False,
                "offset": 0,
                "normalization": None,
                "resolution": list(resolution),
            },
            "training": training,
            "validation": {
                "validate_every_n_epochs": 1,
                "eval_integrals": False,
                "eval_sampling": False,
            },
            "logging": {"mode": "disabled", "tqdm": False},
        }
    )


def tiny_ae_model_cfg(**extra) -> dict:
    """``model`` config of a tiny decoder-conditioned AE (doubled swin residual)."""
    patch = {"patch_size": [2, 0, 2, 4, 2], "window_size": [2, 0, 2, 2, 2], "merging_depth": 2}
    patch.update(unmerging_depth=2, merging_hidden_ratio=2.0, unmerging_hidden_ratio=2.0)
    vit = {"num_heads": [2], "depth": [2], "use_rpb": True, "gated_attention": True}
    vit.update(qk_norm=True, modulation="dit", drop_path=0.0)
    return {
        "name": "ae",
        "latent_dim": 16,
        "decoder_conditioning": ["itg", "dg", "s_hat", "q"],
        "legacy_swin_shortcut": True,
        "patch": {**patch, "c_multiplier": 1},
        "vit": vit,
        "bottleneck": {"dim": 8, "depth": 1, "num_heads": 2},
        **extra,
    }


def tiny_ae(**extra):
    from neugk_jax.models.build import build_ae_from_config

    cfg = {"model": tiny_ae_model_cfg(**extra), "dataset": {"resolution": RES, "separate_zf": True}}
    return build_ae_from_config(cfg, key=jr.PRNGKey(0))


def tiny_vq_cfg(quantizer="vq", **extra) -> dict:
    vq = {"quantizer": quantizer, "codebook_size": 32, "embedding_dim": 8, "levels": [4, 3, 3]}
    m = tiny_ae_model_cfg(name="vqvae", model_type="vqvae", norm_fn="RMSNorm", vq=vq, **extra)
    m.update(legacy_swin_shortcut=False, loss_weights={"df": 1.0, "vq_commit": 1.0})
    return m


GYROSWIN_RES = [8, 2, 4, 10, 8]
GYROSWIN_CFG = {
    "model": {
        "name": "gyroswin_multi",
        "latent_dim": 16,
        "num_layers": 1,
        "decouple_mu": True,
        "conditioning": ["timestep", "itg"],
        "drop_path": 0.0,
        "loss_weights": {"df": 1.0, "phi": 0.1, "flux": 0.0, "fluxavg": 0.0},
        "loss_scheduler": {"fluxavg": {"type": "linear", "start": 1, "end": 1}},
        "swin": {
            "patch_size": [2, 1, 2, 5, 2],
            "window_size": [2, 1, 2, 2, 2],
            "phi_patch_size": [1, 5, 2],
            "phi_window_size": [2, 2, 2],
            "num_heads": 2,
            "depth": 1,
            "merging_hidden_ratio": 2.0,
            "unmerging_hidden_ratio": 2.0,
            "c_multiplier": 2,
            "flux_num_heads": 2,
            "flux_depth": 1,
            "flux_reduce": "max",
            "modulation": "dit",
            "norm_fn": "RMSNorm",
            "act_fn": "GELU",
            "swin_bottleneck": True,
            "latent_cross_attn": True,
            "use_abs_pe": False,
            "use_rope": False,
            "cosine_attn": False,
            "init_weights": "kaiming_uniform",
            "flux_conditioning": True,
        },
    },
    "dataset": {"separate_zf": True, "real_potens": True, "resolution": GYROSWIN_RES},
    "training": {"predict_delta": False, "pushforward": {"unrolls": [0, 0]}},
}
