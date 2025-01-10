from einops import rearrange
from kappamodules.transformer import PerceiverPoolingBlock, PrenormBlock, PerceiverBlock
from kappamodules.layers import ContinuousSincosEmbed
from torch import nn
import torch

from models.nd_vit.patching import PatchEmbed, PatchUnmerging, pad_to_blocks, unpad
from models.nd_vit.positional import PositionalEmbedding


class LatentEncoder(nn.Module):
    def __init__(
        self,
        space: int,
        dim: int,
        patch_size,
        base_resolution,
        in_channels: int = 2,
        num_heads: int = 8,
        depth: int = 2,
        num_latent_tokens: int = 128,
    ):
        super().__init__()

        padded_base_resolution, _ = pad_to_blocks(base_resolution, patch_size)

        self.patch_embed = PatchEmbed(
            space=space,
            base_resolution=padded_base_resolution,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=dim,
            flatten=False,
            mlp_ratio=1.0,
        )

        self.grid_size = self.patch_embed.grid_size
        self.patch_size = patch_size
        self.pos_embed = PositionalEmbedding(
            dim, grid_size=self.grid_size, learnable=True, init_weights="rand"
        )

        # blocks
        self.blocks = nn.Sequential(
            *[PrenormBlock(dim=dim, num_heads=num_heads) for _ in range(depth)],
        )

        self.pooling_perc = PerceiverPoolingBlock(
            dim=dim,
            num_heads=num_heads,
            num_query_tokens=num_latent_tokens,
            perceiver_kwargs={"kv_dim": dim},
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        # pad to patch blocks
        x = rearrange(x, "b c ... -> b ... c")
        x, pad_axes = pad_to_blocks(x, self.patch_size)

        x = self.patch_embed(x)
        x = self.pos_embed(x)

        # flatten
        x = rearrange(x, "b v1 v2 s x y c -> b (v1 v2 s x y) c")

        x = self.blocks(x)
        # to latent tokens
        x = self.pooling_perc(x)
        x = self.norm(x)
        return x, pad_axes


class LatentDecoder(nn.Module):
    def __init__(
        self,
        space: int,
        dim: int,
        base_resolution,
        grid_size,
        patch_size,
        out_channels: int = 2,
        num_heads: int = 8,
    ):
        super().__init__()

        self.grid_size = grid_size
        self.base_resolution = base_resolution

        self.pos_embed = ContinuousSincosEmbed(
            dim=dim,
            ndim=space,
        )

        self.query_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.perc = PerceiverBlock(dim=dim, kv_dim=dim, num_heads=num_heads)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        # unpatch
        self.unpatch = PatchUnmerging(
            space,
            dim,
            grid_size=grid_size,
            expand_by=patch_size,
            out_channels=out_channels,
            flatten=False,
            norm_layer=None,
            mlp_ratio=1.0,
        )

    def forward(self, x, output_pos, pad_axes):

        q = self.pos_embed(output_pos)
        q = self.query_proj(q)

        x = self.perc(q=q, kv=x)
        x = self.norm(x)

        # expand patches to original size
        x = rearrange(
            x,
            "b (v1 v2 s x y) c -> b v1 v2 s x y c",
            v1=self.grid_size[0],
            v2=self.grid_size[1],
            s=self.grid_size[2],
            x=self.grid_size[3],
            y=self.grid_size[4],
        )

        x = self.unpatch(x)

        # unpad output
        x = unpad(x, pad_axes, self.base_resolution)
        # return as image
        x = rearrange(x, "b ... c -> b c ...")

        return x


class CompressionPerc(nn.Module):
    def __init__(
        self,
        space: int,
        dim: int,
        patch_size,
        base_resolution,
        in_channels: int = 2,
        out_channels: int = 2,
        num_heads: int = 8,
        encoder_depth: int = 2,
        approximator_depth: int = 2,
        num_latent_tokens: int = 128,
    ):
        super().__init__()

        self.encoder = LatentEncoder(
            space,
            dim,
            patch_size,
            base_resolution,
            in_channels=in_channels,
            num_heads=num_heads,
            depth=encoder_depth,
            num_latent_tokens=num_latent_tokens,
        )

        self.approximator = nn.Sequential(
            *[
                PrenormBlock(
                    dim=dim,
                    num_heads=num_heads,
                )
                for _ in range(approximator_depth)
            ],
        )
        # nD positions in the downsampled space
        grid_axes = [torch.arange(res) for res in self.encoder.grid_size]
        mesh = torch.meshgrid(*grid_axes, indexing="ij")
        grid_pos = torch.stack(mesh, axis=-1).reshape(-1, len(self.encoder.grid_size))
        grid_pos = 2 * (grid_pos / (torch.tensor(self.encoder.grid_size) - 1)) - 1

        self.register_buffer("grid_pos", grid_pos.unsqueeze(0))

        self.decoder = LatentDecoder(
            space,
            dim,
            out_channels=out_channels,
            base_resolution=base_resolution,
            grid_size=self.encoder.grid_size,
            patch_size=patch_size,
        )

    def forward(self, x, timestep):

        x, pad_axes = self.encoder(x)

        x = self.approximator(x)

        x = self.decoder(x, self.grid_pos, pad_axes)

        return x

    def patch_encode(self, x):
        return self.encoder(x)

    def patch_decode(self, z, pad_axes):
        return self.decoder(z, self.grid_pos, pad_axes)
