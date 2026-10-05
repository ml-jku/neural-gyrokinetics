"""Drop-path is stochastic in training (key + inference=False) and off in inference."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr

from neugk_jax.diffusion.dit import DiT
from neugk_jax.pinc.swin5d_ae import Swin5DAE


def _check(fwd):
    a, b = fwd(jr.PRNGKey(1), False), fwd(jr.PRNGKey(2), False)
    assert not jnp.allclose(a, b)
    assert jnp.array_equal(fwd(jr.PRNGKey(1), True), fwd(None, True))


def test_ae_drop_path():
    res = (4, 4, 4, 16, 8)
    ae = Swin5DAE(
        decouple_mu=True,
        dim=16,
        base_resolution=list(res),
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
        drop_path=0.5,
        key=jr.PRNGKey(0),
    )
    x = jr.normal(jr.PRNGKey(3), (2, *res))
    _check(lambda k, inf: ae(x, key=k, inference=inf)["df"])


def test_dit_drop_path():
    dit = DiT(
        z_dim=4,
        dim=16,
        grid_size=(2, 2, 2),
        depth=2,
        num_heads=2,
        n_cond=0,
        drop_path=0.5,
        key=jr.PRNGKey(0),
    )
    x = jr.normal(jr.PRNGKey(3), (2, 2, 2, 4))
    _check(lambda k, inf: dit(x, jnp.asarray(0.3), key=k, inference=inf))
