import pytest
import torch
import torch.nn as nn
from neugk.models.gk_unet import SwinNDUnet
from neugk.models.nd_vit.patching import PatchEmbed, PatchExpand


@pytest.mark.parametrize("space", [2, 3])
def test_patch_embed_expand_shapes(space):
    # Test that patching and unpatching (expansion) are shape-consistent
    embed_dim = 16
    patch_size = [2] * space
    base_resolution = [8] * space
    in_channels = 2

    # input: [B, C, ...grid]
    x = torch.randn(1, in_channels, *base_resolution)
    # PatchEmbed expects [B, ...grid, C] internally for non-conv,
    # but the forward assertion says all([xs == res for xs, res in zip(x.shape[1:-1], self.base_resolution)])
    # which implies input is [B, ...grid, C]. Let's check PatchEmbed.forward

    # Actually PatchEmbed.forward:
    # assert all([xs == res for xs, res in zip(x.shape[1:-1], self.base_resolution)])
    # x = self.proj(x)

    # Let's fix input shape to [B, ...grid, C]
    x_grid_last = x.permute(0, *range(2, space + 2), 1)

    embed = PatchEmbed(
        space=space,
        base_resolution=base_resolution,
        patch_size=patch_size,
        embed_dim=embed_dim,
        in_channels=in_channels,
        flatten=True,
    )

    latent = embed(x_grid_last)
    # expected latent: [B, num_patches, embed_dim]
    num_patches = (8 // 2) ** space
    assert latent.shape == (1, num_patches, embed_dim)

    expand = PatchExpand(
        space=space,
        dim=embed_dim,
        grid_size=[4] * space,
        expand_by=patch_size,
        out_channels=in_channels,
    )

    out = expand(latent)
    # output is [B, ...grid, out_channels]
    assert out.shape == x_grid_last.shape


@pytest.mark.parametrize(
    "space, res, num_layers",
    [
        (2, [16, 16], 2),
        (3, [8, 8, 8], 1),
    ],
)
def test_swin_nd_unet_forward(space, res, num_layers):
    # Test SwinNDUnet basic forward pass and shape consistency
    model = SwinNDUnet(
        space=space,
        dim=16,
        base_resolution=res,
        patch_size=2,
        window_size=2,
        num_layers=num_layers,
        depth=1,
        num_heads=2,
        in_channels=2,
        out_channels=2,
    )

    x = torch.randn(1, 2, *res)
    out = model(x)

    assert out.shape == x.shape


def test_swin_nd_unet_conditioning():
    # Test that DiT-style conditioning changes outputs
    space = 2
    res = [16, 16]
    # We need to mock condition_keys because it seems missing in some paths or expected as attribute
    model = SwinNDUnet(
        space=space,
        dim=16,
        base_resolution=res,
        patch_size=2,
        window_size=2,
        num_layers=1,
        depth=1,
        num_heads=2,
        conditioning=["ion_temp_grad"],
        modulation="dit",
    )

    # Force set condition_keys if missing (it should be set in __init__ though)
    if not hasattr(model, "condition_keys"):
        model.condition_keys = ["ion_temp_grad"]

    x = torch.randn(1, 2, *res)
    # SwinNDUnet.forward(self, x, **kwargs) -> calls self.condition(kwargs)

    cond_val = torch.randn(1, 1)

    out1 = model(x, ion_temp_grad=cond_val)
    out2 = model(x, ion_temp_grad=cond_val * 2.0)

    assert out1.shape == x.shape
    assert not torch.allclose(out1, out2)


def test_patch_expand_dims():
    space = 2
    grid_size = [4, 4]
    x = torch.randn(1, 16, 16)  # [B, L, C]

    expand = PatchExpand(
        space=space, dim=16, grid_size=grid_size, expand_by=2, out_channels=32
    )

    out = expand(x)
    # [B, H, W, C] -> [1, 8, 8, 32]
    assert out.shape == (1, 8, 8, 32)
