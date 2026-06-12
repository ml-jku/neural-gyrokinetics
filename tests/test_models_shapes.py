"""M1 verification: forward pass shape checks for all model components."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from neugk_jax.models import (
    APE,
    ContinuousConditionEmbed,
    DiT,
    DiTLayer,
    DiTSwinLayer,
    Film,
    LayerNorm,
    Linear,
    MLP,
    PatchEmbed,
    PatchExpand,
    PatchMerge,
    SwinLayer,
    Swin5DUnet,
    ViTLayer,
    pad_to_blocks,
    unpad,
)
from neugk_jax.autoencoders import Swin5DAE




def test_linear_shapes():
    lyr = Linear(8, 4, key=jr.PRNGKey(0))
    out = lyr(jnp.zeros((3, 5, 8)))
    assert out.shape == (3, 5, 4)


def test_layernorm():
    lyr = LayerNorm(8)
    x = jr.normal(jr.PRNGKey(1), (2, 7, 8))
    out = lyr(x)
    assert out.shape == x.shape
    assert jnp.allclose(out.mean(-1), 0.0, atol=1e-5)


def test_mlp():
    lyr = MLP([8, 16, 4], key=jr.PRNGKey(0))
    out = lyr(jnp.zeros((3, 8)))
    assert out.shape == (3, 4)


def test_film():
    lyr = Film(cond_dim=6, dim=8, key=jr.PRNGKey(0))
    x = jr.normal(jr.PRNGKey(1), (4, 5, 8))
    cond = jr.normal(jr.PRNGKey(2), (6,))
    out = lyr(x, cond)
    assert out.shape == x.shape




def test_ape():
    pe = APE(8, (4, 6))
    x = jnp.zeros((4, 6, 8))
    assert pe(x).shape == (4, 6, 8)


def test_ape_3d():
    pe = APE(16, (3, 5, 7))
    x = jnp.zeros((3, 5, 7, 16))
    out = pe(x)
    assert out.shape == x.shape


def test_continuous_condition_embed():
    emb = ContinuousConditionEmbed(32, n_cond=4, key=jr.PRNGKey(0), cond_dim=64)
    out = emb(jnp.array([0.1, 0.5, -0.3, 1.0]))
    assert out.shape == (64,)




def test_pad_to_blocks():
    x = jnp.zeros((7, 13, 4))
    padded, pads = pad_to_blocks(x, (4, 5))
    assert padded.shape[0] % 4 == 0
    assert padded.shape[1] % 5 == 0
    restored = unpad(padded, pads, (7, 13))
    assert restored.shape[:2] == (7, 13)


def test_patch_embed():
    pe = PatchEmbed(
        base_resolution=(16, 24, 16),
        patch_size=(4, 4, 4),
        in_channels=2,
        embed_dim=32,
        key=jr.PRNGKey(0),
    )
    x = jnp.zeros((16, 24, 16, 2))
    out = pe(x)
    assert out.shape == (4, 6, 4, 32)


def test_patch_merge_then_expand():
    grid = (8, 12, 4)
    merge = PatchMerge(dim=16, grid_size=grid, key=jr.PRNGKey(0), c_multiplier=2)
    x = jr.normal(jr.PRNGKey(1), (*grid, 16))
    y = merge(x)
    assert y.shape == (*merge.target_grid_size, merge.out_dim)
    expand = PatchExpand(
        dim=merge.out_dim,
        grid_size=merge.target_grid_size,
        key=jr.PRNGKey(2),
        c_multiplier=2,
        expand_by=2,
        target_grid_size=grid,
    )
    z = expand(y)
    assert z.shape == (*grid, expand.out_dim)




@pytest.mark.parametrize("space,grid,window", [
    (2, (8, 8), (4, 4)),
    (3, (8, 12, 8), (4, 4, 4)),
])
def test_swin_layer(space, grid, window):
    lyr = SwinLayer(
        space=space, dim=32, depth=2, num_heads=4,
        grid_size=grid, window_size=window, key=jr.PRNGKey(0),
    )
    x = jr.normal(jr.PRNGKey(1), (*grid, 32))
    out = lyr(x, inference=True)
    assert out.shape == x.shape


def test_vit_layer():
    lyr = ViTLayer(
        space=3, dim=32, depth=2, num_heads=4,
        grid_size=(2, 3, 4), key=jr.PRNGKey(0),
    )
    x = jr.normal(jr.PRNGKey(1), (2, 3, 4, 32))
    assert lyr(x, inference=True).shape == x.shape


def test_dit_layer():
    lyr = DiTLayer(
        space=3, dim=32, depth=2, num_heads=4,
        grid_size=(2, 3, 4), key=jr.PRNGKey(0), cond_dim=64,
    )
    x = jr.normal(jr.PRNGKey(1), (2, 3, 4, 32))
    cond = jr.normal(jr.PRNGKey(2), (64,))
    assert lyr(x, cond, inference=True).shape == x.shape


def test_dit_swin_layer():
    grid = (8, 8, 4)
    lyr = DiTSwinLayer(
        space=3, dim=32, depth=2, num_heads=4,
        grid_size=grid, window_size=(4, 4, 2),
        key=jr.PRNGKey(0), cond_dim=64,
    )
    x = jr.normal(jr.PRNGKey(1), (*grid, 32))
    cond = jr.normal(jr.PRNGKey(2), (64,))
    assert lyr(x, cond, inference=True).shape == x.shape




def test_swin_5d_unet_no_decouple():
    """Smaller-than-real 5D backbone: shape preservation."""
    space = 5
    base = (4, 4, 4, 16, 8)  # vp, mu, s, x, y
    model = Swin5DUnet(
        space=space,
        decouple_mu=False,
        dim=16,
        base_resolution=base,
        in_channels=2,
        out_channels=2,
        patch_size=(2, 2, 2, 4, 2),
        window_size=(2, 2, 2, 2, 2),
        depth=2, num_heads=2, num_layers=2,
        middle_depth=1, middle_num_heads=2,
        merging_depth=1, unmerging_depth=1,
        merging_hidden_ratio=2.0, unmerging_hidden_ratio=2.0,
        hidden_mlp_ratio=2.0,
        key=jr.PRNGKey(0),
    )
    x = jr.normal(jr.PRNGKey(1), (2, *base))
    out = model(x)
    assert out.shape == x.shape


def test_swin5d_ae_decouple_mu():
    """5D AE with decouple_mu collapses mu into channels (Swin5DAE config)."""
    base = (4, 4, 4, 16, 8)  # vp, mu, s, x, y
    ae = Swin5DAE(
        space=5,
        decouple_mu=True,
        dim=16,
        base_resolution=base,
        in_channels=2,
        out_channels=2,
        patch_size=(2, 0, 2, 4, 2),
        window_size=(2, 0, 2, 2, 2),
        depth=2, num_heads=2, num_layers=2,
        middle_depth=1, middle_num_heads=2,
        bottleneck_dim=24, bottleneck_depth=1, bottleneck_num_heads=2,
        merging_depth=1, unmerging_depth=1,
        merging_hidden_ratio=2.0, unmerging_hidden_ratio=2.0,
        hidden_mlp_ratio=2.0,
        key=jr.PRNGKey(0),
    )
    x = jr.normal(jr.PRNGKey(1), (2, *base))
    z, _ = ae.encode(x)
    assert z.shape == (*ae.bottleneck_grid_size, ae.bottleneck_dim)
    out = ae(x)
    assert out["df"].shape == x.shape


def test_dit_forward():
    grid = (2, 4, 2)
    z_dim = 16
    dim = 32
    model = DiT(
        space=3, z_dim=z_dim, dim=dim, grid_size=grid,
        depth=2, num_heads=4, n_cond=4,
        key=jr.PRNGKey(0),
    )
    x = jr.normal(jr.PRNGKey(1), (*grid, z_dim))
    out = model(x, tstep=jnp.float32(0.5), condition=jnp.array([0.1, 0.2, -0.3, 1.0]))
    assert out.shape == x.shape
    assert model.latent_shape == (*grid, z_dim)



def test_swin5d_ae_vmapped_batch():
    """Vmap over a batch axis works without extra plumbing."""
    base = (4, 4, 4, 16, 8)
    ae = Swin5DAE(
        space=5, decouple_mu=True, dim=8,
        base_resolution=base, in_channels=2, out_channels=2,
        patch_size=(2, 0, 2, 4, 2), window_size=(2, 0, 2, 2, 2),
        depth=1, num_heads=2, num_layers=2,
        middle_depth=1, middle_num_heads=2,
        bottleneck_dim=16, bottleneck_depth=1, bottleneck_num_heads=2,
        merging_depth=1, unmerging_depth=1,
        merging_hidden_ratio=2.0, unmerging_hidden_ratio=2.0,
        hidden_mlp_ratio=2.0,
        key=jr.PRNGKey(0),
    )
    batch = jr.normal(jr.PRNGKey(1), (3, 2, *base))
    out = jax.vmap(lambda x: ae(x)["df"])(batch)
    assert out.shape == batch.shape
