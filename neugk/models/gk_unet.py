"""Neural gyrokinetics swin unet."""

from typing import Sequence, Union, Optional, Type, Tuple, Dict, List

from einops import rearrange
import torch
from torch import nn
from functools import partial

from neugk.models.layers import seq_weight_init
from neugk.models.nd_vit.swin_layers import (
    SwinLayer,
    DiTSwinLayer,
    FilmSwinLayer,
)
from neugk.models.nd_vit.vit_layers import (
    LayerModes,
    ViTLayer,
    DiTLayer,
    FilmViTLayer,
)
from neugk.models.nd_vit.patching import (
    PatchEmbed,
    PatchMerge,
    PatchExpand,
    pad_to_blocks,
    unpad,
)
from neugk.models.nd_vit.positional import APE


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
        use_abs_pe (bool): Absolute positional encoding to the input. Default is False.
        learnable_pos_embed (bool): Learnable APE (if use_abs_pe set). Default is False.
        drop_path (float): Stochastic depth drop rate. Default is 1/10.
        hidden_mlp_ratio (float): Expansion rate for transformer MLPs. Default is 2.0
        c_multiplier (int): Latent dimensions expansions after downsample. Default is 2.
        use_checkpoint (bool): Gradient checkpointing (saves memory). Default is False.
        act_fn (callable): Activation function. Default is nn.GELU.
        norm_layer (nn.Module): Normalization layer type. Default is nn.LayerNorm.
        LayerType (nn.Module): Type for the swin attention layer.
    """

    def __init__(
        self,
        space: int,
        dim: int,
        grid_size: Sequence[int],
        window_size: Sequence[int],
        num_heads: int,
        depth: int,
        use_abs_pe: bool = False,
        learnable_pos_embed: bool = False,
        drop_path: float = 0.1,
        hidden_mlp_ratio: float = 2.0,
        c_multiplier: int = 2,
        use_checkpoint: bool = True,
        act_fn: nn.Module = nn.GELU,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        LayerType: Type = SwinLayer,
        init_weights: str = "xavier_uniform",
    ):
        super().__init__()

        self.window_size = window_size
        self.dim = dim
        self.grid_size = grid_size

        if use_abs_pe:
            self.pos_embed = APE(
                dim, grid_size, learnable=learnable_pos_embed, init_weights="sincos"
            )

        self.swin_att = LayerType(
            space,
            dim,
            depth=depth,
            grid_size=self.grid_size,
            num_heads=num_heads,
            window_size=window_size,
            drop_path=drop_path,
            mlp_ratio=hidden_mlp_ratio,
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
            act_fn=act_fn,
        )

        self.resampled_grid_size = grid_size
        self.downsample = PatchMerge(
            space=space,
            dim=dim,
            grid_size=grid_size,
            norm_layer=norm_layer,
            c_multiplier=c_multiplier,
        )
        self.resampled_grid_size = self.downsample.target_grid_size
        self.out_dim = self.downsample.out_dim

        if init_weights:
            self.reset_parameters(init_weights)

    def reset_parameters(self, init_weights):
        if hasattr(self, "pos_embed"):
            self.pos_embed.reset_parameters()
        self.swin_att.reset_parameters(init_weights)
        self.downsample.reset_parameters(init_weights)

    def forward(
        self, x: torch.Tensor, return_skip: bool = True, **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        """
        Args:
            x: Tensor (B, D, H, ..., C)

        Returns:
            Tensor (B, D, H, ..., C)
        """
        if hasattr(self, "pos_embed"):
            x = self.pos_embed(x)

        x = self.swin_att(x, **kwargs)

        x_merged = self.downsample(x)
        # return skip connection
        return (x_merged, x) if return_skip else x_merged


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
        use_abs_pe (bool): Add absolute positional encoding to the input. Default is False.
        learnable_pos_embed (bool): Learnable APE (if use_abs_pe is set). Default is False.
        drop_path (float): Stochastic depth drop rate. Default is 1/10.
        hidden_mlp_ratio (float): Expansion rate for transformer MLPs. Default is 2.0
        c_multiplier (int): Latent dimensions expansions after downsample. Default is 2.
        use_checkpoint (bool): Gradient checkpointing (saves memory). Default is False.
        act_fn (callable): Activation function. Default is nn.GELU.
        patching_hidden_ratio (float): Expansion rate for patching MLPs. Default is 8.0
        conv_upsample (bool): Use transposed convolutions to unpatch. Default is False.
        norm_layer (nn.Module): Normalization layer type. Default is nn.LayerNorm.
        LayerType (nn.Module): Type for the swin attention layer.
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
        use_abs_pe: bool = False,
        learnable_pos_embed: bool = False,
        drop_path: float = 0.1,
        hidden_mlp_ratio: float = 2.0,
        c_multiplier: int = 2,
        use_checkpoint: bool = False,
        act_fn: nn.Module = nn.GELU,
        conv_upsample: bool = False,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        LayerType: Type = SwinLayer,
        mode: LayerModes = LayerModes.UPSAMPLE,
        init_weights: Optional[str] = "xavier_uniform",
    ):
        super().__init__()

        self.space = space
        self.dim = dim
        self.grid_size = grid_size

        if use_abs_pe:
            self.pos_embed = APE(
                dim, grid_size, learnable=learnable_pos_embed, init_weights="sincos"
            )

        # NOTE: project down concat dimension first to save params
        self.proj_concat = nn.Sequential(nn.Linear(2 * dim, dim), act_fn())
        self.swin_att = LayerType(
            space,
            dim,
            num_heads=num_heads,
            depth=depth,
            drop_path=drop_path,
            grid_size=grid_size,
            mlp_ratio=hidden_mlp_ratio,
            window_size=window_size,
            use_checkpoint=use_checkpoint,
            act_fn=act_fn,
        )
        if mode == LayerModes.UPSAMPLE:
            self.upsample = PatchExpand(
                space=space,
                dim=dim,
                grid_size=grid_size,
                norm_layer=norm_layer,
                c_multiplier=c_multiplier,
                expand_by=2,
                target_grid_size=target_grid_size,
                mlp_depth=1,  # inner unmerges as linear layers
                use_conv=conv_upsample,
            )
            self.resampled_grid_size = self.upsample.target_grid_size
        elif mode == LayerModes.SEQUENCE:
            self.upsample = None
            self.resampled_grid_size = grid_size

        if init_weights:
            self.reset_parameters(init_weights)

    def reset_parameters(self, init_weights):
        if hasattr(self, "proj_concat"):
            if init_weights == "torch" or init_weights is None:
                pass
            elif init_weights == "xavier_uniform":
                self.proj_concat.apply(seq_weight_init(nn.init.xavier_uniform_))
            elif init_weights == "kaiming_uniform":
                self.proj_concat.apply(
                    seq_weight_init(
                        partial(
                            nn.init.kaiming_uniform_,
                            nonlinearity="relu",
                            mode="fan_in",
                            a=0,
                        )
                    )
                )
            elif init_weights in ["truncnormal", "truncnormal002"]:
                self.proj_concat.apply(seq_weight_init(nn.init.trunc_normal_))
            else:
                raise NotImplementedError

        if hasattr(self, "pos_embed"):
            self.pos_embed.reset_parameters()
        self.swin_att.reset_parameters(init_weights)
        if self.upsample is not None:
            self.upsample.reset_parameters(init_weights)

    def forward(
        self,
        x: torch.Tensor,
        s: Optional[torch.Tensor] = None,
        return_skip: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            x: Tensor (B, D, H, ..., C)
            s: Tensor (B, D, H, ..., C) unet skip connection. Can be None (e.g. AE).

        Returns:
            Tensor (B, D, H, ..., C)
        """
        if s is not None:
            assert (
                all(x_s == s_s for x_s, s_s in zip(x.shape, s.shape))
                and x.ndim == s.ndim
            )
            # concat to hidden dim and project to latent dim
            x = self.proj_concat(torch.cat([x, s], -1))

        if hasattr(self, "pos_embed"):
            x = self.pos_embed(x)

        x = self.swin_att(x, **kwargs)

        x_upsampled = x
        if self.upsample is not None:
            x_upsampled = self.upsample(x)
        return (x_upsampled, x) if return_skip else x_upsampled


class SwinNDUnet(nn.Module):
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
        use_abs_pe (bool): Add absolute positional encoding to the input. Default is False.
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
        use_abs_pe: bool = False,
        c_multiplier: int = 2,
        conv_patch: bool = False,
        drop_path: float = 0.1,
        middle_depth: int = 2,
        middle_num_heads: int = 8,
        hidden_mlp_ratio: float = 2.0,
        use_checkpoint: bool = False,
        merging_hidden_ratio: float = 8.0,
        unmerging_hidden_ratio: float = 8.0,
        merging_depth: int = 2,
        unmerging_depth: int = 2,
        conditioning: Optional[List[str]] = None,
        cond_embed: Optional[nn.Module] = None,
        enc_cond_embed: Optional[nn.Module] = None,
        dec_cond_embed: Optional[nn.Module] = None,
        encoder_conditioning: Optional[List[str]] = None,
        decoder_conditioning: Optional[List[str]] = None,
        modulation: str = "dit",
        act_fn: nn.Module = nn.GELU,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        expand_act_fn: nn.Module = nn.LeakyReLU,
        init_weights: str = "xavier_uniform",
        patching_init_weights: str = "xavier_uniform",
        cond_init_weights: str = "normal_smallvar",
        norm_output: bool = False,
        patch_skip: bool = False,
        swin_bottleneck: bool = False,
        use_rpb: bool = True,
        use_rope: bool = False,
        gated_attention: bool = False,
        cosine_attn: bool = False,
        qk_norm: bool = False,
        mid_norm_learnable: bool = True,
    ):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = [patch_size] * space

        if isinstance(window_size, int):
            window_size = [window_size] * space

        self.space = space
        self.patch_size = patch_size
        self.window_size = window_size
        self.init_weights = init_weights
        self.patching_init_weights = patching_init_weights
        self.cond_init_weights = cond_init_weights
        self.base_resolution = base_resolution
        self.norm_output = norm_output
        self.patch_skip = patch_skip
        self.problem_dim = in_channels
        self.act_fn = act_fn
        padded_base_resolution, pad_axes = pad_to_blocks(base_resolution, patch_size)
        self.pad_axes = [int(p) for p in pad_axes]

        self.use_abs_pe = use_abs_pe
        self.use_rpb = use_rpb
        self.use_rope = use_rope
        self.gated_attention = gated_attention
        self.cosine_attn = cosine_attn

        if isinstance(num_heads, int):
            num_heads = [num_heads] * num_layers
        if isinstance(depth, int):
            depth = [depth] * num_layers

        assert len(num_heads) == len(depth) == num_layers
        self.num_heads = num_heads
        self.depth = depth

        # set layer type and conditioning
        self.cond_embed = cond_embed
        self.enc_cond_embed = enc_cond_embed
        self.dec_cond_embed = dec_cond_embed

        if self.enc_cond_embed is None and encoder_conditioning is None:
            self.enc_cond_embed = cond_embed
            self.encoder_condition_keys = sorted(conditioning) if conditioning else []
        else:
            self.encoder_condition_keys = (
                sorted(encoder_conditioning) if encoder_conditioning else []
            )

        if self.dec_cond_embed is None and decoder_conditioning is None:
            self.dec_cond_embed = cond_embed
            self.decoder_condition_keys = sorted(conditioning) if conditioning else []
        else:
            self.decoder_condition_keys = (
                sorted(decoder_conditioning) if decoder_conditioning else []
            )

        if modulation == "dit":
            ModulatedSwinLayer = DiTSwinLayer
            ModulatedViTLayer = DiTLayer
        elif modulation == "film":
            ModulatedSwinLayer = FilmSwinLayer
            ModulatedViTLayer = FilmViTLayer
        else:
            raise ValueError(f"Unknown modulation type: {modulation}")

        def get_layer_types(embed, swin_bottleneck):
            if embed is not None:
                # Set conditioning parameters
                M_SwinLayer = partial(ModulatedSwinLayer, cond_dim=embed.cond_dim)
                M_ViTLayer = partial(ModulatedViTLayer, cond_dim=embed.cond_dim)

                Local = M_SwinLayer
                Global = M_SwinLayer if swin_bottleneck else M_ViTLayer
            else:
                Local = SwinLayer
                Global = SwinLayer if swin_bottleneck else ViTLayer

            Local = partial(
                Local,
                use_rpb=use_rpb,
                use_rope=use_rope,
                gated_attention=gated_attention,
                cosine_attn=cosine_attn,
                qk_norm=qk_norm,
            )
            if swin_bottleneck:
                Global = partial(
                    Global,
                    window_size=window_size,
                    use_rpb=use_rpb,
                    use_rope=use_rope,
                    gated_attention=gated_attention,
                    cosine_attn=cosine_attn,
                    qk_norm=qk_norm,
                )
            else:
                Global = partial(
                    Global,
                    use_rope=use_rope,
                    gated_attention=gated_attention,
                    qk_norm=qk_norm,
                )
            return Local, Global

        self.EncoderLocalLayerType, self.EncoderGlobalLayerType = get_layer_types(
            self.enc_cond_embed, swin_bottleneck
        )
        self.DecoderLocalLayerType, self.DecoderGlobalLayerType = get_layer_types(
            self.dec_cond_embed, swin_bottleneck
        )
        self.enc_cond_dim = self.enc_cond_embed.cond_dim if self.enc_cond_embed else -1
        self.dec_cond_dim = self.dec_cond_embed.cond_dim if self.dec_cond_embed else -1

        self.condition_keys = sorted(
            list(set(self.encoder_condition_keys) | set(self.decoder_condition_keys))
        )

        # Precompute indices for tensor-based conditioning (e.g. PINC)
        self.enc_indices = (
            [self.condition_keys.index(k) for k in self.encoder_condition_keys]
            if self.encoder_condition_keys
            else None
        )
        self.dec_indices = (
            [self.condition_keys.index(k) for k in self.decoder_condition_keys]
            if self.decoder_condition_keys
            else None
        )

        self.patch_embed = PatchEmbed(
            space=space,
            base_resolution=padded_base_resolution,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=dim,
            flatten=False,
            use_conv=conv_patch,
            mlp_ratio=merging_hidden_ratio,
            act_fn=act_fn,
            mlp_depth=merging_depth,
        )

        # down path
        grid_sizes = [self.patch_embed.grid_size]
        down_blocks = []
        down_dims = [dim]
        for i in range(num_layers):
            block = SwinBlockDown(
                space,
                down_dims[i],
                grid_size=grid_sizes[i],
                depth=depth[i],
                window_size=window_size,
                num_heads=num_heads[i],
                use_abs_pe=use_abs_pe,
                drop_path=drop_path,
                learnable_pos_embed=False,
                use_checkpoint=use_checkpoint,
                hidden_mlp_ratio=hidden_mlp_ratio,
                c_multiplier=c_multiplier,
                act_fn=act_fn,
                norm_layer=norm_layer,
                LayerType=self.EncoderLocalLayerType,
            )
            down_blocks.append(block)
            down_dims.append(block.out_dim)
            grid_sizes.append(block.resampled_grid_size)

        self.down_blocks = nn.ModuleList(down_blocks)
        self.grid_sizes = grid_sizes
        self.down_dims = down_dims

        # middle/bottleneck
        bottleneck_norm_layer = (
            partial(norm_layer, elementwise_affine=mid_norm_learnable)
            if norm_layer
            else None
        )
        self.middle = self.DecoderGlobalLayerType(
            space,
            down_dims[-1],
            grid_size=grid_sizes[-1],
            depth=middle_depth,
            num_heads=middle_num_heads,
            drop_path=drop_path,
            mlp_ratio=hidden_mlp_ratio,
            use_checkpoint=use_checkpoint,
            norm_layer=bottleneck_norm_layer,
            act_fn=act_fn,
        )

        if use_abs_pe:
            self.middle_pe = APE(down_dims[-1], grid_sizes[-1])

        self.middle_upscale = PatchExpand(
            space=space,
            dim=down_dims[-1],
            grid_size=grid_sizes[-1],
            target_grid_size=grid_sizes[-2],
            c_multiplier=c_multiplier,
            use_conv=conv_patch,
            mlp_depth=1,  # inner unmerges as linear layers
        )

        # up path
        up_dims = down_dims[::-1][1:]
        up_grid_sizes = grid_sizes[::-1][1:]

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
                    use_abs_pe=use_abs_pe,
                    drop_path=drop_path,
                    hidden_mlp_ratio=hidden_mlp_ratio,
                    c_multiplier=c_multiplier,
                    use_checkpoint=use_checkpoint,
                    act_fn=act_fn,
                    norm_layer=norm_layer,
                    LayerType=self.DecoderLocalLayerType,
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
                use_abs_pe=use_abs_pe,
                drop_path=drop_path,
                hidden_mlp_ratio=hidden_mlp_ratio,
                use_checkpoint=use_checkpoint,
                act_fn=act_fn,
                norm_layer=norm_layer,
                LayerType=self.DecoderLocalLayerType,
                mode=LayerModes.SEQUENCE,
            )
        )
        self.up_blocks = nn.ModuleList(up_blocks)

        # unpatch
        self.unpatch = PatchExpand(
            space,
            up_dims[-1],
            grid_size=up_grid_sizes[-1],
            expand_by=patch_size,
            out_channels=out_channels,
            flatten=False,
            use_conv=conv_patch,
            norm_layer=None,
            mlp_ratio=unmerging_hidden_ratio,
            act_fn=expand_act_fn,
            patch_skip=self.patch_skip,
            cond_dim=(self.dec_cond_embed.cond_dim if self.dec_cond_embed else None),
            mlp_depth=unmerging_depth,
        )
        self.reset_parameters()

    def reset_parameters(self):
        # patching
        self.patch_embed.reset_parameters(self.patching_init_weights)
        self.unpatch.reset_parameters(self.patching_init_weights)
        # conditioning
        if hasattr(self, "cond_embed") and self.cond_embed is not None:
            self.cond_embed.reset_parameters(self.cond_init_weights)
        if hasattr(self, "encoder_cond_embed") and self.enc_cond_embed is not None:
            self.enc_cond_embed.reset_parameters(self.cond_init_weights)
        if hasattr(self, "decoder_cond_embed") and self.decoder_cond_embed is not None:
            self.decoder_cond_embed.reset_parameters(self.cond_init_weights)

        # backbone
        for up_blk in self.up_blocks:
            up_blk.reset_parameters(self.init_weights)
        for down_blk in self.down_blocks:
            down_blk.reset_parameters(self.init_weights)
        self.middle.reset_parameters(self.init_weights)
        self.middle_upscale.reset_parameters(self.init_weights)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # compress to patch space
        x, pad_axes = self.patch_encode(x)
        if self.patch_skip:
            first_res = x.clone()
        # backbone
        enc_cond = self.condition(
            kwargs,
            self.enc_cond_embed,
            self.encoder_condition_keys,
            indices=self.enc_indices,
        )
        dec_cond = self.condition(
            kwargs,
            self.decoder_cond_embed,
            self.decoder_condition_keys,
            indices=self.dec_indices,
        )

        # down path
        feature_maps = []
        for blk in self.down_blocks:
            x, x_pre = blk(x, **enc_cond)
            feature_maps.append(x_pre)
        # middle block
        if hasattr(self, "middle_pe"):
            x = self.middle_pe(x)
        x = self.middle(x, **dec_cond)
        x = self.middle_upscale(x)
        # up path
        feature_maps = feature_maps[::-1]
        for i, blk in enumerate(self.up_blocks):
            x = blk(x, s=feature_maps[i], **dec_cond)
        # expand to original
        if self.patch_skip:
            x = torch.cat([x, first_res], -1)
        return self.patch_decode(x, pad_axes, **dec_cond)

    def get_pad_axes(self, resolution: Sequence[int]) -> List[int]:
        _, pad_axes = pad_to_blocks(resolution, self.patch_size)
        return pad_axes

    def patch_encode(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b c ... -> b ... c")
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
        # return as image
        x = rearrange(x, "b ... c -> b c ...")
        return x

    def condition(
        self,
        kwconds: Dict[str, torch.Tensor],
        embed: Optional[nn.Module] = None,
        keys: Optional[List[str]] = None,
        indices: Optional[List[int]] = None,
    ) -> Dict:
        embed = embed or self.cond_embed
        keys = keys or self.condition_keys

        if embed is None:
            return {}

        # 1. Handle raw tensor passed in kwconds['condition'] (e.g. from PINC forward)
        if "condition" in kwconds and isinstance(kwconds["condition"], torch.Tensor):
            raw_tensor = kwconds["condition"]
            if indices is not None:
                cond = raw_tensor[:, indices]
            elif raw_tensor.shape[-1] == len(keys):
                cond = raw_tensor
            else:
                return {}
            return {"condition": embed(cond)}

        # 2. Handle individual variables in kwconds (e.g. from GyroSwin forward)
        if not keys:
            return {}

        selected = {k: v for k, v in kwconds.items() if k in keys}
        if len(selected) == 0:
            return {}

        assert keys == sorted(list(selected.keys())), (
            "Mismatch in conditioning keys "
            f"{keys} != {sorted(list(selected.keys()))}"
        )

        cond = torch.cat(
            [selected[k].view(selected[k].shape[0], -1) for k in keys],
            dim=-1,
        )
        return {"condition": embed(cond)}


class Swin5DUnet(SwinNDUnet):
    def __init__(
        self,
        decouple_mu: bool = False,
        **kwargs,
    ):
        full_in_channels = kwargs["in_channels"]
        kwargs["space"] = 5
        full_resolution = list(kwargs["base_resolution"])
        if decouple_mu:
            kwargs["space"] = 4
            # adjust patch and window size
            patch_size = kwargs["patch_size"]
            kwargs["patch_size"] = [patch_size[0]] + patch_size[2:]
            window_size = kwargs["window_size"]
            kwargs["window_size"] = [window_size[0]] + window_size[2:]
            decoupled_dim = full_resolution[1]
            # adjust resolution and channels
            kwargs["base_resolution"] = [full_resolution[0]] + full_resolution[2:]
            kwargs["in_channels"] = full_in_channels * decoupled_dim
            kwargs["out_channels"] = kwargs["out_channels"] * decoupled_dim
            vel_pe_resolution = [1, decoupled_dim, 1, 1, 1]

        super().__init__(**kwargs)
        self.decouple_mu = decouple_mu
        self.full_resolution = full_resolution
        self.original_problem_dim = full_in_channels
        if decouple_mu:
            self.decoupled_dim = decoupled_dim
            # positional information for velocity mixing
            self.vel_pe = APE(full_in_channels, vel_pe_resolution, True)

    def forward(self, x, **kwargs):
        return {"df": super().forward(x, **kwargs)}

    def patch_encode(self, df: torch.Tensor):
        df = rearrange(df, "b c ... -> b ... c")
        # decouple mu and add positional information
        if self.decouple_mu:
            df = self.vel_pe(df)
            df = rearrange(df, "b vp mu ... c -> b vp ... (c mu)")
        # pad to patch blocks
        df, pad_axes = pad_to_blocks(df, self.patch_size)
        # linear flat patch embedding
        df = self.patch_embed(df)
        return df, pad_axes

    def patch_decode(
        self,
        z: torch.Tensor,
        pad_axes: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # expand patches to original size
        df = self.unpatch(z, condition)
        # unpad output
        df = unpad(df, pad_axes, self.base_resolution)
        # return as image
        df = rearrange(df, "b ... c -> b c ...")
        if self.decouple_mu:
            df = rearrange(
                df, "b (c mu) vp ... -> b c vp mu ...", mu=self.decoupled_dim
            )
        return df
