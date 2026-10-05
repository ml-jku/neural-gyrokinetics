"""Shared training-step behavior: buffers are neither differentiated nor decayed."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from omegaconf import OmegaConf

from neugk_jax.models.embeddings import APE
from neugk_jax.models.swin import swin_layer
from neugk_jax.models.utils import trainable_mask
from neugk_jax.training.runner import build_optimizer, train_update


def test_buffers_frozen_under_weight_decay():
    layer = swin_layer(8, 2, 2, (4, 8), (2, 4), key=jr.PRNGKey(0), use_rpb=True)
    model = (layer, APE(8, (4, 8), key=jr.PRNGKey(1)))
    mask = trainable_mask(model)
    blk = layer.blocks[1]
    assert blk.attn_mask is not None and blk.attn.rpb is not None
    frozen = [blk.attn_mask, blk.attn.rpb.rpb, blk.attn.rpb.rpb_idx]
    tcfg = OmegaConf.create({"weight_decay": 0.1, "clip_grad": False})
    opt = build_optimizer(optax.constant_schedule(1e-2), tcfg, model, decoupled=False)
    opt_state = opt.init(eqx.filter(model, mask))
    x = jr.normal(jr.PRNGKey(1), (4, 8, 8))

    def loss_fn(m):
        return jnp.mean(m[1](m[0](x, inference=True)) ** 2)

    new, _, _ = train_update(model, opt_state, loss_fn, opt, mask)
    nblk = new[0].blocks[1]
    after = [nblk.attn_mask, nblk.attn.rpb.rpb, nblk.attn.rpb.rpb_idx]
    for a, b in zip(frozen, after):
        assert jnp.array_equal(a, b)
    moved = jax.tree_util.tree_map(
        lambda a, b: not jnp.array_equal(a, b), eqx.filter(model, mask), eqx.filter(new, mask)
    )
    assert all(jax.tree_util.tree_leaves(moved))
