"""Regression guards: legacy residual flag, rms_norm forwarding, weight-decay masking."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import pytest
from omegaconf import OmegaConf


def test_swin_layer_forwards_rms_norm():
    from neugk_jax.models.swin import swin_layer
    from neugk_jax.models.utils import RMSNorm

    grid, win, dim = (4, 8), (2, 4), 8
    layer = swin_layer(dim, 2, 2, grid, win, key=jr.PRNGKey(2), rms_norm=True)
    assert all(isinstance(b.norm1, RMSNorm) and isinstance(b.norm2, RMSNorm) for b in layer.blocks)
    layer_ln = swin_layer(dim, 1, 2, grid, win, key=jr.PRNGKey(2), rms_norm=False)
    assert not isinstance(layer_ln.blocks[0].norm1, RMSNorm)
    layer_legacy = swin_layer(dim, 1, 2, grid, win, key=jr.PRNGKey(2), legacy_double_shortcut=True)
    assert layer_legacy.blocks[0].legacy_double_shortcut


def test_weight_decay_mask_and_coupling():
    from neugk_jax.training.runner import build_optimizer, weight_decay_mask

    params = {"cond_embed": jnp.ones(3), "blocks": {"w": jnp.ones(3)}}
    mask = weight_decay_mask(params, ["cond"])
    assert mask == {"cond_embed": False, "blocks": {"w": True}}
    assert not any(jax.tree_util.tree_leaves(weight_decay_mask(params, "all")))

    tcfg = OmegaConf.create({"weight_decay": 0.1, "exclude_from_wd": ["cond"], "clip_grad": False})
    zero = jax.tree_util.tree_map(jnp.zeros_like, params)
    for decoupled in (False, True):
        opt = build_optimizer(optax.constant_schedule(1e-2), tcfg, params, decoupled=decoupled)
        upd, _ = opt.update(zero, opt.init(params), params)
        # excluded leaves see no decay, the rest shrink
        assert jnp.all(upd["cond_embed"] == 0)
        assert jnp.all(upd["blocks"]["w"] < 0)


def _zero_mlp(blk):
    zeroed = jax.tree_util.tree_map(lambda a: jnp.zeros_like(a) if eqx.is_array(a) else a, blk.mlp)
    return eqx.tree_at(lambda b: b.mlp, blk, zeroed)


def test_legacy_double_shortcut_swin_block():
    from neugk_jax.models.swin import SwinBlock

    x = jr.normal(jr.PRNGKey(0), (4, 8, 8))
    single = SwinBlock(8, 2, (4, 8), (2, 4), key=jr.PRNGKey(1))
    legacy = SwinBlock(8, 2, (4, 8), (2, 4), key=jr.PRNGKey(1), legacy_double_shortcut=True)
    x_res1 = _zero_mlp(single)(x)
    # zeroed mlp leaves the post-attention residual: single -> x_res1, legacy -> 2*x_res1
    assert jnp.allclose(_zero_mlp(legacy)(x), 2.0 * x_res1, atol=1e-5)
    assert jnp.allclose(legacy(x) - single(x), x_res1, atol=1e-5)


def test_layers_forward_legacy_flag():
    from neugk_jax.models.swin import swin_layer

    kw = dict(key=jr.PRNGKey(0), legacy_double_shortcut=True)
    for cond in ({}, {"cond_dim": 6, "cond_mode": "film"}):
        layer = swin_layer(8, 2, 2, (4, 8), (2, 4), **cond, **kw)
        assert all(b.legacy_double_shortcut for b in layer.blocks)


def test_layers_reject_unknown_kwargs():
    from neugk_jax.models.swin import swin_layer
    from neugk_jax.models.vit import vit_layer

    common = dict(key=jr.PRNGKey(0), rms_nrom=True)
    for cond in ({}, {"cond_dim": 4}, {"cond_dim": 4, "cond_mode": "film"}):
        with pytest.raises(TypeError):
            swin_layer(8, 1, 2, (4, 4), (2, 2), **cond, **common)
        with pytest.raises(TypeError):
            vit_layer(8, 1, 2, **cond, **common)


def test_ae_backbone_has_no_dead_middle():
    from neugk_jax.pinc import Swin5DAE

    ae = Swin5DAE(
        decouple_mu=True,
        dim=8,
        base_resolution=[4, 4, 4, 16, 8],
        in_channels=2,
        out_channels=2,
        patch_size=[2, 0, 2, 4, 2],
        window_size=[2, 0, 2, 2, 2],
        depth=[1],
        num_heads=[2],
        num_layers=1,
        bottleneck_dim=8,
        bottleneck_depth=1,
        bottleneck_num_heads=2,
        c_multiplier=1,
        key=jr.PRNGKey(0),
    )
    assert ae.backbone.middle is None and ae.backbone.middle_upscale is None
    assert ae(jnp.zeros((2, 4, 4, 4, 16, 8)))["df"].shape == (2, 4, 4, 4, 16, 8)


def test_ae_dit_builders_accept_mappings(tmp_path):
    import yaml

    from neugk_jax.models.build import build_ae_from_config, build_dit_from_config

    ae_cfg = {
        "model": {
            "latent_dim": 8,
            "num_layers": 1,
            "decouple_mu": True,
            "patch": {
                "patch_size": [2, 0, 2, 4, 2],
                "window_size": [2, 0, 2, 2, 2],
                "c_multiplier": 1,
                "merging_depth": 1,
                "unmerging_depth": 1,
            },
            "vit": {"num_heads": [2], "depth": [1], "drop_path": 0.3},
            "bottleneck": {"dim": 8, "depth": 1, "num_heads": 2},
        },
        "dataset": {"separate_zf": False, "resolution": [4, 4, 4, 16, 8]},
    }
    path = tmp_path / "ae.yaml"
    path.write_text(yaml.safe_dump(ae_cfg))
    ae_path = build_ae_from_config(str(path), key=jr.PRNGKey(0))
    ae = build_ae_from_config(OmegaConf.create(ae_cfg), key=jr.PRNGKey(0))
    assert jax.tree_util.tree_structure(ae) == jax.tree_util.tree_structure(ae_path)
    assert ae.middle_pre.blocks[0].drop_path.rate == 0.3
    dit_cfg = {
        "model": {
            "latent_dim": 16,
            "conditioning": ["itg", "dg"],
            "vit": {"num_heads": 2, "depth": 1},
        }
    }
    dit = build_dit_from_config(dit_cfg, ae, key=jr.PRNGKey(0))
    blk = dit.backbone.blocks[0]
    assert blk.mlp.layers[0].weight.shape[0] == 2 * 16 and blk.drop_path.rate == 0.1
    assert dit.cond_embed is not None


def test_conditioning_slots_follow_sorted_names():
    import numpy as np

    from neugk_jax.training.runner import conditioning_slots

    ds_conds = sorted(["itg", "dg", "s_hat", "q", "timestep"])
    for order in (["itg", "dg", "s_hat", "q"], ["q", "s_hat", "dg", "itg"]):
        slots = conditioning_slots(ds_conds, order)
        assert [ds_conds[i] for i in slots] == ["dg", "itg", "q", "s_hat"]
    assert conditioning_slots(ds_conds, []) is None
    assert np.asarray(conditioning_slots(ds_conds, ["timestep"])).tolist() == [4]


def test_builders_default_to_the_single_swin_residual():
    import copy

    from helpers import GYROSWIN_CFG, tiny_ae_model_cfg

    from neugk_jax.models.build import build_ae_from_config, build_gyroswin_from_config

    ae_cfg = {"model": tiny_ae_model_cfg(), "dataset": {"resolution": [4, 4, 4, 16, 8]}}
    del ae_cfg["model"]["legacy_swin_shortcut"]
    gs_cfg = copy.deepcopy(GYROSWIN_CFG)
    # film-modulated gyroswin: its plain swin blocks carry the residual option
    gs_cfg["model"]["swin"]["modulation"] = "film"
    cases = (
        (build_ae_from_config, ae_cfg, lambda m: m.backbone.down_blocks[0]),
        (build_gyroswin_from_config, gs_cfg, lambda m: m.df_unet.down_blocks[0]),
    )
    for build, cfg, block in cases:
        for legacy in (None, True):
            if legacy:
                cfg["model"]["legacy_swin_shortcut"] = True
            blk = block(build(cfg, key=jr.PRNGKey(0))).swin.blocks[0]
            assert blk.legacy_double_shortcut is bool(legacy), build.__name__
