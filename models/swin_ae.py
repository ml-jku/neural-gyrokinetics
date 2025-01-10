from typing import Sequence, Union, Optional

import numpy as np
from einops import rearrange
import torch
from torch import nn
from functools import partial

from models.nd_vit.swin_layers import SwinLayer, ModulatedSwinLayer, LayerModes
from models.nd_vit.positional import PositionalEmbedding
from models.nd_vit.patching import PatchEmbed, PatchUnmerging, pad_to_blocks, unpad
from models.swin_unet import SwinUnet, SwinBlockDown, SwinBlockUp


class SwinAE(SwinUnet):
    """N-dimensional shifted window transformer autoecoder implementation (v1/v2)."""

    def __init__(
        self,
        space: int,
        dim: int,
        base_resolution: Sequence[int],
        patch_size: Union[Sequence[int], int] = 4,
        window_size: Union[Sequence[int], int] = 5,
        depth: Union[Sequence[int], int] = 2,
        up_depth: Optional[Union[Sequence[int], int]] = None,
        num_heads: Union[Sequence[int], int] = 4,
        up_num_heads: Optional[Union[Sequence[int], int]] = None,
        in_channels: int = 2,
        out_channels: int = 2,
        num_layers: int = 4,
        abs_pe: bool = False,
        c_multiplier: int = 2,
        conv_patch: bool = False,
        drop_path: float = 0.1,
        hidden_mlp_ratio: float = 2.0,
        use_checkpoint: bool = False,
        patching_hidden_ratio: float = 8.0,
        conditioning: Optional[nn.Module] = None,
        act_fn: nn.Module = nn.GELU,
        expand_act_fn: nn.Module = nn.LeakyReLU,
        init_weights: str = "xavier_uniform",
        patching_init_weights: str = "xavier_uniform",
    ):
        super().__init__(
            space=space,
            dim=dim,
            base_resolution=base_resolution,
            patch_size=patch_size,
            window_size=window_size,
            depth=depth,
            up_depth=up_depth,
            num_heads=num_heads,
            up_num_heads=up_num_heads,
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            abs_pe=abs_pe,
            c_multiplier=c_multiplier,
            conv_patch=conv_patch,
            drop_path=drop_path,
            hidden_mlp_ratio=hidden_mlp_ratio,
            use_checkpoint=use_checkpoint,
            patching_hidden_ratio=patching_hidden_ratio,
            conditioning=conditioning,
            act_fn=act_fn,
            expand_act_fn=expand_act_fn,
            init_weights=init_weights,
            patching_init_weights=patching_init_weights,
        )

        from models.nd_vit.vit_layers import ViTLayer

        del self.middle

        # middle_dim = dim * c_multiplier ** num_layers

        # self.middle_pe = PositionalEmbedding(middle_dim, self.grid_sizes[-1], learnable=True)
        # self.middle = ViTLayer(
        #     space,
        #     middle_dim,
        #     grid_size=self.grid_sizes[-1],
        #     depth=4,
        #     num_heads=8,
        #     drop_path=drop_path,
        #     mlp_ratio=hidden_mlp_ratio,
        #     mode=LayerModes.SEQUENCE,
        #     use_checkpoint=use_checkpoint,
        #     act_fn=act_fn,
        # )

        for i in range(num_layers):
            del self.up_blocks[i].proj_concat

        self.reset_parameters()

    def reset_parameters(self):
        # patching
        self.patch_embed.reset_parameters(self.patching_init_weights)
        self.unpatch.reset_parameters(self.patching_init_weights)
        # conditioning
        self.cond_embed.reset_parameters(self.init_weights)
        # backbone
        for up_blk, down_blk in zip(self.up_blocks, self.down_blocks):
            up_blk.reset_parameters(self.init_weights)
            down_blk.reset_parameters(self.init_weights)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:

        cond = kwargs.get("timestep")
        if cond is not None and self.cond_embed is not None:
            # embed conditioning is e.g. sincos
            cond = {"condition": self.cond_embed(cond)}
        else:
            cond = {}

        # pad to patch blocks
        x, pad_axes = self.patch_encode(x)

        # down path
        for blk in self.down_blocks:
            x = blk(x, return_skip=False, **cond)

        # # middle
        # x = self.middle_pe(x)
        # x = self.middle(x)
        x = self.middle_upscale(x)

        # down path
        for blk in self.up_blocks:
            x = blk(x, **cond)

        # expand patches to original size
        x = self.patch_decode(x, pad_axes)
        return x

    def patch_encode(self, x: torch.Tensor) -> torch.Tensor:
        # pad to patch blocks
        x = rearrange(x, "b c ... -> b ... c")
        x, pad_axes = pad_to_blocks(x, self.patch_size)

        # linear flat patch embedding
        x = self.patch_embed(x)
        return x, pad_axes

    def patch_decode(self, z: torch.Tensor, pad_axes: torch.Tensor) -> torch.Tensor:
        # expand patches to original size
        x = self.unpatch(z)

        # unpad output
        x = unpad(x, pad_axes, self.base_resolution)
        # return as image
        x = rearrange(x, "b ... c -> b c ...")

        return x
