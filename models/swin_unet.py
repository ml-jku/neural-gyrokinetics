from typing import Sequence, Union, Optional

import numpy as np
from einops import rearrange
import torch
from torch import nn
from functools import partial
from math import ceil

from models.nd_swin import (
    PositionalEmbedding,
    PatchEmbed,
    PatchMerging,
    PatchUnmerging,
    SwinLayer,
    SwinLayerModes,
    pad_to_blocks,
    unpad
)


class SwinBlockDown(nn.Module):
    def __init__(
        self,
        space: int,
        dim: int,
        grid_size: Sequence[int],
        window_size: Sequence[int],
        num_heads: int,
        depth: int,
        downsample: bool = False,
        c_multiplier: int = 2,
        abs_pe: bool = False,
        drop_path: float = 0.1,
        hidden_mlp_ratio: float = 2.0,
        learnable_pos_embed: bool = False,
        use_checkpoint: bool = True,
        act_fn: nn.Module = nn.GELU
    ):
        super().__init__()

        self.window_size = window_size
        self.abs_pe = abs_pe
        self.dim = dim
        self.grid_size = grid_size

        if abs_pe:
            self.pos_embed = PositionalEmbedding(
                dim, grid_size, learnable=learnable_pos_embed
            )

        self.swin_att = SwinLayer(
            space,
            dim,
            depth=depth,
            resolution=self.grid_size,
            num_heads=num_heads,
            window_size=window_size,
            drop_path=drop_path,
            mlp_ratio=hidden_mlp_ratio,
            use_checkpoint=use_checkpoint,
            resample=PatchMerging if downsample else None,
            c_multiplier=c_multiplier,
            mode=SwinLayerModes.DOWNSAMPLE if downsample else SwinLayerModes.SEQUENCE,
            act_fn=act_fn
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor (B, D, H, ..., C)

        Returns:
            Tensor (B, D, H, ..., C)
        """
        if self.abs_pe:
            x = self.pos_embed(x)

        return self.swin_att(x)


class SwinBlockUp(nn.Module):
    def __init__(
        self,
        space: int,
        dim: int,
        grid_size: Sequence[int],
        window_size: Sequence[int],
        num_heads: int,
        depth: int,
        target_grid_size: Optional[Sequence[int]] = None,
        upsample: bool = False,
        abs_pe: bool = False,
        drop_path: float = 0.1,
        hidden_mlp_ratio: float = 2.0,
        c_multiplier: int = 2,
        learnable_pos_embed: bool = False,
        use_checkpoint: bool = False,
        act_fn: nn.Module = nn.GELU,
        patching_hidden_ratio: float = 8.0,
    ):
        super().__init__()

        self.space = space
        self.abs_pe = abs_pe
        self.dim = dim
        self.upsample = upsample
        self.grid_size = grid_size

        if abs_pe:
            self.pos_embed = PositionalEmbedding(
                dim, grid_size, learnable=learnable_pos_embed
            )

        if upsample:
            upsample_fn = partial(
                PatchUnmerging,
                expand_by=2,
                grid_size=grid_size,
                target_grid_size=target_grid_size,
                mlp_ratio=patching_hidden_ratio,
                act_fn=act_fn
            )
            mode = SwinLayerModes.UPSAMPLE
            dim_next = dim // c_multiplier  # latent mapped down in upsample
        else:
            upsample_fn = None
            mode = SwinLayerModes.SEQUENCE
            dim_next = dim  # no latent map down
        self.swin_att = nn.Sequential(
            SwinLayer(
                space,
                2 * dim,  # concat skip connection
                mode=mode,
                num_heads=num_heads,
                depth=depth,
                drop_path=drop_path,
                resolution=grid_size,
                mlp_ratio=hidden_mlp_ratio,
                window_size=window_size,
                resample=upsample_fn,
                c_multiplier=c_multiplier,
                use_checkpoint=use_checkpoint,
                act_fn=act_fn
            ),
            act_fn(),
            nn.Linear(2 * dim_next, dim_next, bias=False),  # project down
        )

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor (B, D, H, ..., C)
            s: Tensor (B, D, H, ..., C) unet skip connection

        Returns:
            Tensor (B, D, H, ..., C)
        """
        # TODO not ideal and not the right place (solves patch merging padding)
        # s, _ = pad_to_shape(s, x.shape[1:-1])

        assert (
            all(x_s == s_s for x_s, s_s in zip(x.shape, s.shape)) and x.ndim == s.ndim
        )

        x = torch.cat([x, s], -1)

        if self.abs_pe:
            x = self.pos_embed(x)
        x = self.swin_att(x)

        return x


class SwinUnet(nn.Module):
    def __init__(
        self,
        space: int,
        dim: int,
        img_size: Sequence[int],
        patch_size: Union[Sequence[int], int] = 4,
        window_size: Union[Sequence[int], int] = 5,
        depth: Union[Sequence[int], int] = 2,
        up_depth: Optional[Union[Sequence[int], int]] = None,
        num_heads: Union[Sequence[int], int] = 4,
        up_num_heads: Optional[Union[Sequence[int], int]] = None,
        in_channels: int = 3,
        out_channels: int = 3,
        num_layers: int = 4,
        learnable_pos_embed: bool = False,
        downsample: Union[Sequence[bool], bool] = False,
        c_multiplier: int = 2,
        conv_patch: bool = False,
        drop_path: float = 0.1,
        middle_depth: int = 4,
        middle_num_heads: int = 8,
        hidden_mlp_ratio: float = 2.0,
        abs_pe: bool = False,
        use_checkpoint: bool = False,
        patching_hidden_ratio: float = 8.0,
        act_fn: nn.Module = nn.GELU,
        expand_act_fn: nn.Module = nn.LeakyReLU,
    ):
        super().__init__()

        if isinstance(patch_size, int):
            patch_size = [patch_size] * space

        if isinstance(window_size, int):
            window_size = [window_size] * space

        self.patch_size = patch_size
        self.window_size = window_size
        self.abs_pe = abs_pe
        self.img_size = img_size
        padded_img_size, _ = pad_to_blocks(img_size, patch_size)

        if isinstance(downsample, bool):
            downsample = [downsample] * num_layers
        if isinstance(num_heads, int):
            num_heads = [num_heads] * num_layers
        if isinstance(depth, int):
            depth = [depth] * num_layers

        assert len(downsample) == num_layers

        self.patch_embed = PatchEmbed(
            space=space,
            img_size=padded_img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=dim,
            flatten=False,
            use_conv=conv_patch,
            mlp_ratio=patching_hidden_ratio,
            act_fn=act_fn
        )

        # down
        c_multiplier = [c_multiplier if downsample[i] else 1 for i in range(num_layers)]
        grid_size = self.patch_embed.grid_size
        acc_multi = [1] + list(np.cumprod(c_multiplier))  # values are pre-downsample
        down_dims = [dim * m for m in acc_multi]
        grid_sizes = [[ceil(g / m) for g in grid_size] for m in acc_multi]

        down_blocks = []
        for i in range(num_layers):
            down_blocks.append(
                SwinBlockDown(
                    space,
                    down_dims[i],
                    grid_size=grid_sizes[i],
                    depth=depth[i],
                    window_size=window_size,
                    num_heads=num_heads[i],
                    downsample=downsample[i],
                    abs_pe=abs_pe,
                    drop_path=drop_path,
                    learnable_pos_embed=learnable_pos_embed,
                    use_checkpoint=use_checkpoint,
                    hidden_mlp_ratio=hidden_mlp_ratio,
                    act_fn=act_fn
                )
            )
        self.down_blocks = nn.ModuleList(down_blocks)

        # middle
        self.middle = SwinLayer(
            space,
            down_dims[-1],
            resolution=grid_sizes[-1],
            depth=middle_depth,
            num_heads=middle_num_heads,
            window_size=window_size,
            resample=None,
            drop_path=drop_path,
            mlp_ratio=hidden_mlp_ratio,
            mode=SwinLayerModes.SEQUENCE,
            use_checkpoint=use_checkpoint,
            act_fn=act_fn
        )
        if abs_pe:
            self.middle_pe = PositionalEmbedding(down_dims[-1], grid_sizes[-1])

        # up
        upsample = downsample[::-1]
        up_dims = down_dims[::-1]
        up_grid_sizes = grid_sizes[::-1]
        # TODO better place to put patch merging padding
        # up_grid_sizes = [pad_to_blocks(g, space * (2,))[0] for g in up_grid_sizes]

        up_depth = up_depth if up_depth is not None else depth[::-1]
        up_num_heads = up_num_heads if up_num_heads is not None else num_heads[::-1]

        up_blocks = []
        for i in range(num_layers):
            up_blocks.append(
                SwinBlockUp(
                    space,
                    up_dims[i],
                    grid_size=up_grid_sizes[i],
                    target_grid_size=up_grid_sizes[i + 1],
                    window_size=window_size,
                    num_heads=up_num_heads[i],
                    depth=up_depth[i],
                    upsample=upsample[i],
                    abs_pe=abs_pe,
                    drop_path=drop_path,
                    learnable_pos_embed=learnable_pos_embed,
                    hidden_mlp_ratio=hidden_mlp_ratio,
                    use_checkpoint=use_checkpoint,
                    patching_hidden_ratio=patching_hidden_ratio,
                    act_fn=act_fn
                )
            )

        self.up_blocks = nn.ModuleList(up_blocks)

        self.unpatch = PatchUnmerging(
            space,
            up_dims[-1],
            grid_size=up_grid_sizes[-1],
            expand_by=patch_size,
            out_channels=out_channels,
            flatten=False,
            use_conv=conv_patch,
            norm_layer=None,
            mlp_ratio=patching_hidden_ratio,
            act_fn=expand_act_fn,
            patch_skip=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pad to patch blocks
        x = rearrange(x, "b c ... -> b ... c")
        x, pad_axes = pad_to_blocks(x, self.patch_size)

        # linear flat patch embedding
        x = self.patch_embed(x)
        # patch_skip = x

        # down path
        feature_maps = []

        for blk in self.down_blocks:
            x = blk(x)
            feature_maps.append(x)

        # middle block
        if self.abs_pe:
            x = self.middle_pe(x)
        x = self.middle(x)

        # down path
        feature_maps = feature_maps[::-1]
        for i, blk in enumerate(self.up_blocks):
            x = blk(x, s=feature_maps[i])

        # expand patches to original size
        x = self.unpatch(x) # torch.cat([x, patch_skip], -1))

        # unpad output
        x = unpad(x, pad_axes, self.img_size)
        # return as image
        x = rearrange(x, "b ... c -> b c ...")
        return x

    def autoencoder(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b c ... -> b ... c")
        x, pad_axes = pad_to_blocks(x, self.patch_size)

        z = self.patch_embed(x)
        x = self.unpatch(z)

        x = unpad(x, pad_axes, self.img_size)
        x = rearrange(x, "b ... c -> b c ...")
        return x
