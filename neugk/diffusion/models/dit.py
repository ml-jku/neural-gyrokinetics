from typing import Sequence, Union, Optional, Type

import torch
from torch import nn
from functools import partial

from neugk.models.nd_vit.vit_layers import ViTLayer, DiTLayer, FilmViTLayer
from neugk.models.nd_vit.positional import PositionalEmbedding
from neugk.models.nd_vit.patching import (
    PatchEmbed,
    PatchExpand,
    pad_to_blocks,
    unpad,
)


class DiT(nn.Module):
    def __init__(
        self,
        space: int,
        z_dim: int,
        dim: int,
        base_resolution: Sequence[int],
        time_embed: nn.Module,
        patch_size: Union[Sequence[int], int] = 4,
        depth: int = 2,
        num_heads: int = 4,
        drop_path: float = 0.1,
        hidden_mlp_ratio: float = 2.0,
        use_checkpoint: bool = False,
        merging_hidden_ratio: float = 8.0,
        unmerging_hidden_ratio: float = 8.0,
        cond_embed: Optional[nn.Module] = None,
        modulation: str = "dit",
        act_fn: nn.Module = nn.GELU,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        expand_act_fn: nn.Module = nn.LeakyReLU,
        init_weights: str = "xavier_uniform",
        patching_init_weights: str = "xavier_uniform",
        norm_output: bool = False,
        patch_skip: bool = False,
    ):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = [patch_size] * space

        self.patch_size = patch_size
        self.init_weights = init_weights
        self.patching_init_weights = patching_init_weights
        self.base_resolution = base_resolution
        self.norm_output = norm_output
        self.patch_skip = patch_skip

        num_heads = num_heads if isinstance(num_heads, int) else sum(num_heads)
        self.num_heads = num_heads
        depth = depth if isinstance(depth, int) else sum(depth)
        self.depth = depth

        grid_size = base_resolution
        if patch_size is not None and patch_size != 1 and sum(patch_size) != 1:
            padded_base_resolution, _ = pad_to_blocks(base_resolution, patch_size)
            self.patch_embed = PatchEmbed(
                space=space,
                base_resolution=padded_base_resolution,
                patch_size=patch_size,
                in_channels=dim,
                embed_dim=dim,
                flatten=False,
                mlp_ratio=merging_hidden_ratio,
                act_fn=act_fn,
            )
            self.unpatch = PatchExpand(
                space,
                dim,
                grid_size=self.patch_embed.grid_size,
                expand_by=patch_size,
                out_channels=dim,
                flatten=False,
                norm_layer=None,
                mlp_ratio=unmerging_hidden_ratio,
                act_fn=expand_act_fn,
                patch_skip=self.patch_skip,
                cond_dim=self.cond_embed.cond_dim if self.cond_embed else None,
            )

            grid_size = self.patch_embed.grid_size

        self.latent_shape = (*grid_size, z_dim)
        # diffusion time conditioning
        self.time_embed = time_embed
        cond_dim = self.time_embed.cond_dim
        # parameter conditioning
        if cond_embed:
            self.cond_embed = cond_embed
            cond_dim += self.cond_embed.cond_dim
        if modulation == "dit":
            GlobalLayer = partial(DiTLayer, cond_dim=cond_dim)
        if modulation == "film":
            GlobalLayer = partial(FilmViTLayer, cond_dim=cond_dim)

        self.encoder = nn.Sequential(nn.Linear(z_dim, dim, bias=False), act_fn())
        self.ape = PositionalEmbedding(z_dim, grid_size, init_weights="sincos")
        self.backbone = GlobalLayer(
            space,
            dim,
            grid_size=grid_size,
            depth=depth,
            num_heads=num_heads,
            drop_path=drop_path,
            mlp_ratio=hidden_mlp_ratio,
            use_checkpoint=use_checkpoint,
            norm_layer=norm_layer,
            act_fn=act_fn,
        )
        self.decoder = nn.Sequential(nn.Linear(dim, z_dim, bias=False), act_fn())

        self.reset_parameters()

    def reset_parameters(self):
        # patching
        if hasattr(self, "patch_embed"):
            self.patch_embed.reset_parameters(self.patching_init_weights)
            self.unpatch.reset_parameters(self.patching_init_weights)
        # conditioning
        if hasattr(self, "cond_embed") and self.cond_embed is not None:
            self.cond_embed.reset_parameters(self.init_weights)
        # backbone
        self.backbone.reset_parameters(self.init_weights)

    def patch_encode(self, x: torch.Tensor) -> torch.Tensor:
        # pad to patch blocks
        x, pad_axes = pad_to_blocks(x, self.patch_size)
        # linear flat patch embedding
        x = self.patch_embed(x)
        return x, pad_axes

    def patch_decode(
        self,
        z: torch.Tensor,
        pad_axes: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # expand patches to original size
        x = self.unpatch(z, condition)
        # unpad output
        x = unpad(x, pad_axes, self.base_resolution)
        return x

    def forward(
        self,
        x: torch.Tensor,
        tstep: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # compress to patch space
        if hasattr(self, "patch_embed"):
            x, pad_axes = self.patch_encode(x)
            if self.patch_skip:
                first_res = x.clone()

        tstep = self.time_embed(tstep)

        if condition is not None:
            condition = self.cond_embed(condition)
            condition = torch.cat([tstep, condition], dim=-1)
        else:
            condition = tstep

        x = self.ape(x)

        x = self.encoder(x)
        x = self.backbone(x, condition=condition)
        x = self.decoder(x)

        # expand to original
        if hasattr(self, "patch_embed"):
            if self.patch_skip:
                x = torch.cat([x, first_res], -1)

            x = self.patch_decode(x, pad_axes, condition=condition)
        return x
