from typing import Sequence, Union, Optional, Type

import numpy as np
from einops import rearrange
import torch
from torch import nn
from functools import partial

from models.nd_vit.swin_layers import SwinLayer, ModulatedSwinLayer, LayerModes
from models.nd_vit.positional import PositionalEmbedding
from models.nd_vit.patching import (
    PatchEmbed,
    PatchMerging,
    PatchUnmerging,
    pad_to_blocks,
    unpad,
)


class SwinBlockDown(nn.Module):
    """N-dimensional shifted window transformer downsample block.

    `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`
    (arxiv.org/pdf/2103.14030)

    Args:
        space (int): Number of input/output dimensions.
        dim (int): latent dimension. Divided by `c_multiplier` at output.
        grid_size (tuple(int)): Input resolution.
        window_size (int | tuple(int)): Window size for the shifted window attention.
        depth (int): Number of swin transformer layers.
        num_heads (int): Number of attention heads in the swin layers.
        abs_pe (bool): Add absolute positional encoding to the input. Default is False.
        learnable_pos_embed (bool): Learnable APE (if abs_pe is set). Default is False.
        drop_path (float): Stochastic depth drop rate. Default is 1/10.
        hidden_mlp_ratio (float): Expansion rate for transformer MLPs. Default is 2.0
        c_multiplier (int): Latent dimensions expansions after downsample. Default is 2.
        use_checkpoint (bool): Gradient checkpointing (saves memory). Default is False.
        act_fn (callable): Activation function. Default is nn.GELU.
        swin_attention_layer (nn.Module): Type for the swin attention layer.
    """

    def __init__(
        self,
        space: int,
        dim: int,
        grid_size: Sequence[int],
        window_size: Sequence[int],
        num_heads: int,
        depth: int,
        abs_pe: bool = False,
        learnable_pos_embed: bool = False,
        drop_path: float = 0.1,
        hidden_mlp_ratio: float = 2.0,
        c_multiplier: int = 2,
        use_checkpoint: bool = True,
        act_fn: nn.Module = nn.GELU,
        swin_attention_layer: Type = SwinLayer,
    ):
        super().__init__()

        self.window_size = window_size
        self.dim = dim
        self.grid_size = grid_size

        if abs_pe:
            self.pos_embed = PositionalEmbedding(
                dim, grid_size, learnable=learnable_pos_embed
            )
        # TODO move downsample here?
        self.swin_att = swin_attention_layer(
            space,
            dim,
            depth=depth,
            grid_size=self.grid_size,
            num_heads=num_heads,
            window_size=window_size,
            drop_path=drop_path,
            mlp_ratio=hidden_mlp_ratio,
            use_checkpoint=use_checkpoint,
            resample_fn=PatchMerging,
            c_multiplier=c_multiplier,
            mode=LayerModes.DOWNSAMPLE,
            act_fn=act_fn,
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x: Tensor (B, D, H, ..., C)

        Returns:
            Tensor (B, D, H, ..., C)
        """
        if hasattr(self, "pos_embed"):
            x = self.pos_embed(x)

        return self.swin_att(x, **kwargs, return_skip=True)


class SwinBlockUp(nn.Module):
    """N-dimensional shifted window transformer upscale block.

    `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`
    (arxiv.org/pdf/2103.14030)

    Args:
        space (int): Number of input/output dimensions.
        dim (int): latent dimension. Divided by `c_multiplier` at output.
        grid_size (tuple(int)): Input resolution.
        window_size (int | tuple(int)): Window size for the shifted window attention.
        depth (int): Number of swin transformer layers.
        num_heads (int): Number of attention heads in the swin layers.
        target_grid_size (tuple(int)): Output resolution (after upsample).
        abs_pe (bool): Add absolute positional encoding to the input. Default is False.
        learnable_pos_embed (bool): Learnable APE (if abs_pe is set). Default is False.
        drop_path (float): Stochastic depth drop rate. Default is 1/10.
        hidden_mlp_ratio (float): Expansion rate for transformer MLPs. Default is 2.0
        c_multiplier (int): Latent dimensions expansions after downsample. Default is 2.
        use_checkpoint (bool): Gradient checkpointing (saves memory). Default is False.
        act_fn (callable): Activation function. Default is nn.GELU.
        patching_hidden_ratio (float): Expansion rate for patching MLPs. Default is 8.0
        swin_attention_layer (nn.Module): Type for the swin attention layer.
        conv_upsample (bool): Use transposed convolutions to unpatch. Default is False.
        mode (LayerModes): Specify which operation to perform in the up-layer.
    """

    def __init__(
        self,
        space: int,
        dim: int,
        grid_size: Sequence[int],
        window_size: Sequence[int],
        depth: int,
        num_heads: int,
        target_grid_size: Optional[Sequence[int]] = None,
        abs_pe: bool = False,
        learnable_pos_embed: bool = False,
        drop_path: float = 0.1,
        hidden_mlp_ratio: float = 2.0,
        c_multiplier: int = 2,
        use_checkpoint: bool = False,
        act_fn: nn.Module = nn.GELU,
        patching_hidden_ratio: float = 8.0,
        swin_attention_layer: Type = SwinLayer,
        conv_upsample: bool = False,
        mode: LayerModes = LayerModes.UPSAMPLE,
    ):
        super().__init__()

        self.space = space
        self.dim = dim
        self.grid_size = grid_size

        if abs_pe:
            self.pos_embed = PositionalEmbedding(
                dim, grid_size, learnable=learnable_pos_embed
            )

        if mode == LayerModes.UPSAMPLE:
            upsample_fn = partial(
                PatchUnmerging,
                expand_by=2,
                target_grid_size=target_grid_size,
                mlp_ratio=patching_hidden_ratio,
                act_fn=act_fn,
                use_conv=conv_upsample,
            )
        elif mode == LayerModes.SEQUENCE:
            upsample_fn = None
        # NOTE: project down concat dimension first to save params
        self.proj_concat = nn.Sequential(nn.Linear(2 * dim, dim), act_fn())
        # TODO move upsample here?
        self.swin_att = swin_attention_layer(
            space,
            dim,
            num_heads=num_heads,
            depth=depth,
            drop_path=drop_path,
            grid_size=grid_size,
            mlp_ratio=hidden_mlp_ratio,
            window_size=window_size,
            resample_fn=upsample_fn,
            c_multiplier=c_multiplier,
            use_checkpoint=use_checkpoint,
            act_fn=act_fn,
            mode=mode,
        )

    def forward(self, x: torch.Tensor, s: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x: Tensor (B, D, H, ..., C)
            s: Tensor (B, D, H, ..., C) unet skip connection

        Returns:
            Tensor (B, D, H, ..., C)
        """
        assert (
            all(x_s == s_s for x_s, s_s in zip(x.shape, s.shape)) and x.ndim == s.ndim
        )

        x = torch.cat([x, s], -1)
        # concat to hidden dim
        x = self.proj_concat(x)

        if hasattr(self, "pos_embed"):
            x = self.pos_embed(x)

        return self.swin_att(x, **kwargs)


class SwinUnet(nn.Module):
    """N-dimensional shifted window transformer UNet implementation (v1/v2). The number
    of spatial/temporal dimensions is set with the argument `space` and the model is
    built accordingly.

    `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`
    (arxiv.org/pdf/2103.14030)

    Args:
        space (int): Number of input/output dimensions.
        dim (int): latent dimension. Multiplied by `c_multiplier` for every downsample.
        base_resolution (tuple(int)): Input grid size.
        patch_size (int | tuple(int)): Patch size. Default is 4 (across all dimensions).
        window_size (int | tuple(int)): Window size for the shifted window attention.
                        Default is 5 (across all dimensions).
        depth (int | tuple(int)): Depth at each (down/up) Swin Transformer layer.
        up_depth (int | tuple(int)): Depth at each UP Swin Transformer layer.
        num_heads (int | tuple(int)): Number of attention heads in each swin layer.
        up_num_heads (int | tuple(int)): Number of attention heads in each UP layer.
        in_channels (int): Number of input channels. Default is 2.
        out_channels (int): Number of output channels. Default is 2.
        num_layers (int): Number of down/up layers. Each layer applies a down/up-sample.
                        Default is 4.
        abs_pe (bool): Add absolute positional encoding to the input. Default is False.
        c_multiplier (int): Latent dimensions expansions after downsample. Default is 2.
        conv_patch (bool): Use convolutions to patch and unpatch (only 2D or 3D).
                        Default is False.
        drop_path (float): Stochastic depth drop rate. Default is 1/10.
        middle_depth (int): Number of layers in the bottleneck. Default is 4.
        middle_num_heads (int): Attention heads in the bottleneck. Default is 8.
        hidden_mlp_ratio (float): Expansion rate for transformer MLPs. Default is 2.0
        use_checkpoint (bool): Gradient checkpointing (saves memory). Default is False.
        patching_hidden_ratio (float): Expansion rate for patching MLPs. Default is 2.0
        conditioning (bool): Allow (Film) conditioning of swin layers. Default is False.
                        If set, a `timestep` must be passed to the forward call.
        act_fn (callable): Activation function. Default is nn.GELU.
        expand_act_fn (callable): Activation function for the patch expansion. Default
                        is nn.LeakyRelu. Better if nonzero in the negative regime.
    """

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
        middle_depth: int = 4,
        middle_num_heads: int = 8,
        hidden_mlp_ratio: float = 2.0,
        use_checkpoint: bool = False,
        patching_hidden_ratio: float = 8.0,
        conditioning: Optional[nn.Module] = None,
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
        self.base_resolution = base_resolution
        padded_base_resolution, _ = pad_to_blocks(base_resolution, patch_size)

        if isinstance(num_heads, int):
            num_heads = [num_heads] * num_layers
        if isinstance(depth, int):
            depth = [depth] * num_layers

        assert len(num_heads) == len(depth) == num_layers

        # set conditioning and layer type
        SwinAttentionLayer = SwinLayer
        self.cond_embed = conditioning
        if self.cond_embed is not None:
            SwinAttentionLayer = partial(
                ModulatedSwinLayer, cond_dim=self.cond_embed.cond_dim
            )

        self.patch_embed = PatchEmbed(
            space=space,
            base_resolution=padded_base_resolution,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=dim,
            flatten=False,
            use_conv=conv_patch,
            mlp_ratio=patching_hidden_ratio,
            act_fn=act_fn,
        )

        # down
        acc_multi = [1] + list(np.cumprod([c_multiplier] * num_layers))
        # pre-downsample dims
        down_dims = [int(dim * m) for m in acc_multi]

        assert all([(d % h) == 0 for d, h in zip(down_dims, num_heads)])

        grid_sizes = [self.patch_embed.grid_size]
        down_blocks = []
        for i in range(num_layers):
            block = SwinBlockDown(
                space,
                down_dims[i],
                grid_size=grid_sizes[i],
                depth=depth[i],
                window_size=window_size,
                num_heads=num_heads[i],
                abs_pe=abs_pe,
                drop_path=drop_path,
                learnable_pos_embed=False,
                use_checkpoint=use_checkpoint,
                hidden_mlp_ratio=hidden_mlp_ratio,
                act_fn=act_fn,
                swin_attention_layer=SwinAttentionLayer,
            )
            down_blocks.append(block)
            grid_sizes.append(block.swin_att.resampled_grid_size)

        self.down_blocks = nn.ModuleList(down_blocks)
        self.grid_sizes = grid_sizes

        # middle
        self.middle = SwinAttentionLayer(
            space,
            down_dims[-1],
            grid_size=grid_sizes[-1],
            depth=middle_depth,
            num_heads=middle_num_heads,
            window_size=window_size,
            resample_fn=None,
            drop_path=drop_path,
            mlp_ratio=hidden_mlp_ratio,
            mode=LayerModes.SEQUENCE,
            use_checkpoint=use_checkpoint,
            act_fn=act_fn,
        )

        if abs_pe:
            self.middle_pe = PositionalEmbedding(down_dims[-1], grid_sizes[-1])

        self.middle_upscale = PatchUnmerging(
            space=space,
            dim=down_dims[-1],
            grid_size=grid_sizes[-1],
            target_grid_size=grid_sizes[-2],
            mlp_ratio=patching_hidden_ratio,
            act_fn=act_fn,
            use_conv=conv_patch,
        )

        # up
        up_dims = down_dims[::-1][1:]
        up_grid_sizes = grid_sizes[::-1][1:]
        # patch merging padding
        # up_grid_sizes = [pad_to_blocks(g, space * (2,))[0] for g in up_grid_sizes]

        up_depth = up_depth if up_depth is not None else depth[::-1]
        up_num_heads = up_num_heads if up_num_heads is not None else num_heads[::-1]

        up_blocks = []
        for i in range(num_layers - 1):
            up_blocks.append(
                SwinBlockUp(
                    space,
                    up_dims[i],
                    grid_size=up_grid_sizes[i],
                    target_grid_size=up_grid_sizes[i + 1],
                    window_size=window_size,
                    num_heads=up_num_heads[i],
                    depth=up_depth[i],
                    abs_pe=abs_pe,
                    drop_path=drop_path,
                    hidden_mlp_ratio=hidden_mlp_ratio,
                    use_checkpoint=use_checkpoint,
                    patching_hidden_ratio=patching_hidden_ratio,
                    act_fn=act_fn,
                    swin_attention_layer=SwinAttentionLayer,
                    conv_upsample=conv_patch,
                )
            )
        # last up block (no upsample)
        up_blocks.append(
            SwinBlockUp(
                space,
                up_dims[-1],
                grid_size=up_grid_sizes[-1],
                window_size=window_size,
                num_heads=up_num_heads[-1],
                depth=up_depth[-1],
                abs_pe=abs_pe,
                drop_path=drop_path,
                hidden_mlp_ratio=hidden_mlp_ratio,
                use_checkpoint=use_checkpoint,
                act_fn=act_fn,
                swin_attention_layer=SwinAttentionLayer,
                mode=LayerModes.SEQUENCE,
            )
        )
        self.up_blocks = nn.ModuleList(up_blocks)

        # unpatch
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
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:

        cond = kwargs.get("timestep")
        if cond is not None and self.cond_embed is not None:
            # embed conditioning is e.g. sincos
            cond = {"condition": self.cond_embed(cond)}
        else:
            cond = {}

        # pad to patch blocks
        x = rearrange(x, "b c ... -> b ... c")
        x, pad_axes = pad_to_blocks(x, self.patch_size)

        # linear flat patch embedding
        x = self.patch_embed(x)

        # down path
        feature_maps = []

        for blk in self.down_blocks:
            x, x_pre = blk(x, **cond)
            feature_maps.append(x_pre)

        # middle block
        if hasattr(self, "middle_pe"):
            x = self.middle_pe(x)
        x = self.middle(x, **cond)

        x = self.middle_upscale(x)

        # down path
        feature_maps = feature_maps[::-1]
        for i, blk in enumerate(self.up_blocks):
            x = blk(x, s=feature_maps[i], **cond)

        # expand patches to original size
        x = self.unpatch(x)

        # unpad output
        x = unpad(x, pad_axes, self.base_resolution)
        # return as image
        x = rearrange(x, "b ... c -> b c ...")
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
