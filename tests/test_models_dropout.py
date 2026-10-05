"""Dropout is identity in inference, stochastic in training, and adds no parameters."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from neugk_jax.gyroswin.models.gyroswin import GyroSwinMultitask
from neugk_jax.gyroswin.models.x_layers import FluxDecoder, MixingBlock, QueryPool, velocity_pool
from neugk_jax.models.attention import MultiHeadCrossAttention, MultiHeadSelfAttention
from neugk_jax.models.utils import MLP, dropout

RES = (8, 2, 4, 10, 8)


def _shapes(model):
    return [x.shape for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))]


def _check(fwd):
    train = [fwd(jr.PRNGKey(i), False) for i in range(1, 9)]
    assert any(not jnp.allclose(train[0], t) for t in train[1:])
    assert jnp.array_equal(fwd(jr.PRNGKey(1), True), fwd(None, True))
    assert jnp.array_equal(fwd(None, False), fwd(None, True))


def test_dropout_fn():
    x = jnp.ones((1000,))
    assert jnp.array_equal(dropout(x, 0.5, key=jr.PRNGKey(0), inference=True), x)
    assert jnp.array_equal(dropout(x, 0.0, key=jr.PRNGKey(0), inference=False), x)
    y = dropout(x, 0.5, key=jr.PRNGKey(0), inference=False)
    assert set(jnp.unique(y).tolist()) <= {0.0, 2.0}
    assert 0.3 < float(jnp.mean(y == 0.0)) < 0.7


def test_self_attention_dropout():
    x = jr.normal(jr.PRNGKey(3), (12, 16))
    attn = MultiHeadSelfAttention(16, 2, key=jr.PRNGKey(0), attn_drop=0.5)
    base = MultiHeadSelfAttention(16, 2, key=jr.PRNGKey(0))
    assert _shapes(attn) == _shapes(base)
    assert jnp.allclose(attn(x), base(x))
    _check(lambda k, inf: attn(x, key=k, inference=inf))


def test_cross_attention_dropout():
    left, right = jr.normal(jr.PRNGKey(3), (6, 16)), jr.normal(jr.PRNGKey(4), (9, 8))
    attn = MultiHeadCrossAttention(16, 8, 2, key=jr.PRNGKey(0), attn_drop=0.5, proj_drop=0.2)
    base = MultiHeadCrossAttention(16, 8, 2, key=jr.PRNGKey(0))
    assert _shapes(attn) == _shapes(base)
    assert jnp.allclose(attn(left, right), base(left, right))
    _check(lambda k, inf: attn(left, right, key=k, inference=inf))


def test_mlp_dropout():
    x = jr.normal(jr.PRNGKey(3), (5, 8))
    mlp = MLP([8, 16, 4], key=jr.PRNGKey(0), drop=0.5)
    assert _shapes(mlp) == _shapes(MLP([8, 16, 4], key=jr.PRNGKey(0)))
    _check(lambda k, inf: mlp(x, key=k, inference=inf))


def test_mixing_vspace_flux_dropout():
    left, right = jr.normal(jr.PRNGKey(3), (2, 3, 16)), jr.normal(jr.PRNGKey(4), (4, 8))
    mix = MixingBlock(16, 8, 2, key=jr.PRNGKey(0), attn_drop=0.5, drop=0.5)
    assert _shapes(mix) == _shapes(MixingBlock(16, 8, 2, key=jr.PRNGKey(0)))
    _check(lambda k, inf: mix(left, right, key=k, inference=inf))

    df = jr.normal(jr.PRNGKey(5), (4, 2, 3, 2, 16))
    vs = QueryPool(16, 8, 2, key=jr.PRNGKey(0), attn_drop=0.5)
    _check(lambda k, inf: velocity_pool(vs, df, key=k, inference=inf))

    head = FluxDecoder([16], [8], 2, 1, key=jr.PRNGKey(0), attn_drop=0.5, drop=0.5)
    _check(
        lambda k, inf: head([head.mix(0, left, right, key=k, inference=inf)], key=k, inference=inf)
    )


def _gyroswin(**kw):
    return GyroSwinMultitask(
        dim=16,
        df_base_resolution=RES,
        df_patch_size=[2, 1, 2, 5, 2],
        df_window_size=[2, 1, 2, 2, 2],
        depth=1,
        num_heads=2,
        in_channels=4,
        out_channels=4,
        num_layers=1,
        merging_hidden_ratio=2.0,
        unmerging_hidden_ratio=2.0,
        flux_key="fluxavg",
        n_cond=2,
        flux_num_heads=2,
        drop_path=0.0,
        key=jr.PRNGKey(0),
        **kw,
    )


def test_gyroswin_dropout_no_new_params():
    model = _gyroswin(attn_drop=0.5, flux_drop=0.5)
    assert _shapes(model) == _shapes(_gyroswin(attn_drop=0.0, flux_drop=0.0))
    x = jr.normal(jr.PRNGKey(1), (4, *RES))
    cond = jnp.array([0.3, -0.2])
    for out_key in ("df", "phi", "fluxavg"):
        _check(lambda k, inf: model(x, cond, key=k, inference=inf)[out_key])
