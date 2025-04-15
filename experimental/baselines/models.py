from typing import Sequence, Union, Optional, Type, Dict

from einops import rearrange
import torch
from torch import nn
from functools import partial

from models.utils import MLP, ContinuousConditionEmbed
from models.nd_vit.vit_layers import ViTLayer, DiTLayer, FilmViTLayer, LayerModes
from models.nd_vit.swin_layers import SwinLayer, DiTSwinLayer, FilmSwinLayer
from models.nd_vit.positional import PositionalEmbedding
from experimental.swin_xnet import FluxDecoder
from models.swin_unet import SwinBlockDown, SwinBlockUp
from models.nd_vit.patching import (
    PatchEmbed,
    PatchUnmerging,
    pad_to_blocks,
    unpad,
)

from experimental.baselines.fno import FNOLayer, FilmFNOLayer


class PhiUnet(nn.Module):
    def __init__(
        self,
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
        middle_depth: int = 2,
        middle_num_heads: int = 8,
        hidden_mlp_ratio: float = 2.0,
        use_checkpoint: bool = False,
        merging_hidden_ratio: float = 8.0,
        unmerging_hidden_ratio: float = 8.0,
        conditioning: Optional[nn.Module] = None,
        modulation: str = "dit",
        block_type: str = "swin",
        act_fn: nn.Module = nn.GELU,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        expand_act_fn: nn.Module = nn.LeakyReLU,
        init_weights: str = "xavier_uniform",
        patching_init_weights: str = "xavier_uniform",
        patch_skip: bool = False,
        swin_bottleneck: bool = False,
    ):
        super().__init__()

        space = 3

        if isinstance(patch_size, int):
            patch_size = [patch_size] * space

        if isinstance(window_size, int):
            window_size = [window_size] * space

        self.patch_size = patch_size
        self.window_size = window_size
        self.init_weights = init_weights
        self.patching_init_weights = patching_init_weights
        self.base_resolution = base_resolution
        self.patch_skip = patch_skip
        padded_base_resolution, _ = pad_to_blocks(base_resolution, patch_size)

        if isinstance(num_heads, int):
            num_heads = [num_heads] * num_layers
        if isinstance(depth, int):
            depth = [depth] * num_layers

        assert len(num_heads) == len(depth) == num_layers

        # set layer type and conditioning
        if block_type == "swin":
            LocalLayer = SwinLayer
            GlobalLayer = SwinLayer if swin_bottleneck else ViTLayer
        if block_type == "fno":
            LocalLayer = FNOLayer
            GlobalLayer = FNOLayer
        self.cond_embed = conditioning
        if self.cond_embed is not None:
            if block_type == "swin":
                if modulation == "dit":
                    ModulatedSwinLayer = DiTSwinLayer
                    ModulatedViTLayer = DiTLayer
                if modulation == "film":
                    ModulatedSwinLayer = FilmSwinLayer
                    ModulatedViTLayer = FilmViTLayer
            if block_type == "fno":
                ModulatedSwinLayer = FilmFNOLayer
                ModulatedViTLayer = FilmFNOLayer
            LocalLayer = partial(ModulatedSwinLayer, cond_dim=self.cond_embed.cond_dim)
            if swin_bottleneck:
                GlobalLayer = partial(
                    ModulatedSwinLayer,
                    cond_dim=self.cond_embed.cond_dim,
                    window_size=window_size,
                )
            else:
                GlobalLayer = partial(
                    ModulatedViTLayer, cond_dim=self.cond_embed.cond_dim
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
                abs_pe=abs_pe,
                drop_path=drop_path,
                learnable_pos_embed=False,
                use_checkpoint=use_checkpoint,
                hidden_mlp_ratio=hidden_mlp_ratio,
                c_multiplier=c_multiplier,
                act_fn=act_fn,
                norm_layer=norm_layer,
                LayerType=LocalLayer,
            )
            down_blocks.append(block)
            down_dims.append(block.out_dim)
            grid_sizes.append(block.resampled_grid_size)

        self.down_blocks = nn.ModuleList(down_blocks)
        self.grid_sizes = grid_sizes
        self.down_dims = down_dims

        # middle/bottleneck
        self.middle = GlobalLayer(
            space,
            down_dims[-1],
            grid_size=grid_sizes[-1],
            depth=middle_depth,
            num_heads=middle_num_heads,
            drop_path=drop_path,
            mlp_ratio=hidden_mlp_ratio,
            use_checkpoint=use_checkpoint,
            norm_layer=norm_layer,
            act_fn=act_fn,
        )

        if abs_pe:
            self.middle_pe = PositionalEmbedding(down_dims[-1], grid_sizes[-1])

        self.middle_upscale = PatchUnmerging(
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
                    abs_pe=abs_pe,
                    drop_path=drop_path,
                    hidden_mlp_ratio=hidden_mlp_ratio,
                    c_multiplier=c_multiplier,
                    use_checkpoint=use_checkpoint,
                    act_fn=act_fn,
                    norm_layer=norm_layer,
                    LayerType=LocalLayer,
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
                norm_layer=norm_layer,
                LayerType=LocalLayer,
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
            mlp_ratio=unmerging_hidden_ratio,
            act_fn=expand_act_fn,
            patch_skip=self.patch_skip,
            cond_dim=self.cond_embed.cond_dim if self.cond_embed else None,
        )

        self.flux_head = FluxDecoder(
            self.down_dims[::-1],
            num_heads=8,
            drop=0.1,
            attn_drop=0.1,
        )

        self.reset_parameters()

    def reset_parameters(self):
        # patching
        self.patch_embed.reset_parameters(self.patching_init_weights)
        self.unpatch.reset_parameters(self.patching_init_weights)
        # conditioning
        if hasattr(self, "cond_embed") and self.cond_embed is not None:
            self.cond_embed.reset_parameters(self.init_weights)
        # backbone
        for up_blk, down_blk in zip(self.up_blocks, self.down_blocks):
            up_blk.reset_parameters(self.init_weights)
            down_blk.reset_parameters(self.init_weights)
        self.middle.reset_parameters(self.init_weights)
        self.middle_upscale.reset_parameters(self.init_weights)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # compress to patch space
        x, pad_axes = self.patch_encode(x)
        if self.patch_skip:
            first_res = x.clone()

        # backbone
        cond = self.condition(kwargs)

        # down path
        feature_maps = []
        flux_lats = []
        for blk in self.down_blocks:
            x, x_pre = blk(x, **cond)
            feature_maps.append(x_pre)

        # middle block
        if hasattr(self, "middle_pe"):
            x = self.middle_pe(x)
        x = self.middle(x, **cond)
        flux_lats.append(self.flux_head.mix(0, x))
        x = self.middle_upscale(x)

        # up path
        feature_maps = feature_maps[::-1]
        for i, blk in enumerate(self.up_blocks):
            x, x_ = blk(x, s=feature_maps[i], return_skip=True, **cond)
            flux_lats.append(self.flux_head.mix(i + 1, x_))

        # expand to original
        if self.patch_skip:
            x = torch.cat([x, first_res], -1)

        x = self.patch_decode(x, cond["condition"], pad_axes)

        flux = self.flux_head(flux_lats)

        return x, flux

    def patch_encode(self, x: torch.Tensor) -> torch.Tensor:
        # pad to patch blocks
        x = rearrange(x, "b c ... -> b ... c")
        x, pad_axes = pad_to_blocks(x, self.patch_size)
        # linear flat patch embedding
        x = self.patch_embed(x)
        return x, pad_axes

    def patch_decode(
        self, z: torch.Tensor, cond: torch.Tensor, pad_axes: torch.Tensor
    ) -> torch.Tensor:
        # expand patches to original size
        x = self.unpatch(z, cond)
        # unpad output
        x = unpad(x, pad_axes, self.base_resolution)
        # return as image
        x = rearrange(x, "b ... c -> b c ...")
        return x

    def condition(self, kwconds) -> Dict:
        if len(kwconds) == 0:
            return {}
        cond = kwconds.get("timestep")
        cond = cond.unsqueeze(-1)
        refine_step = kwconds.get("refinement_step", None)
        if refine_step is not None:
            cond = torch.cat([cond, refine_step.unsqueeze(-1)], dim=-1)
        itg = kwconds.get("itg", None)
        if itg is not None:
            cond = torch.cat([cond, itg.unsqueeze(-1)], dim=-1)
        if cond is not None and self.cond_embed is not None:
            # embed conditioning is e.g. sincos
            return {"condition": self.cond_embed(cond)}
        else:
            return {}


class FluxLSTM(nn.Module):
    def __init__(self, dim: int, n_cond: int, num_layers: int):
        super().__init__()

        cond_embed = ContinuousConditionEmbed(32, n_cond)
        self.cond_embed = nn.Sequential(
            cond_embed, MLP([cond_embed.cond_dim, dim], dropout_prob=0.1)
        )
        self.encoder = nn.Linear(1, dim, bias=False)
        self.lstm = nn.LSTM(2 * dim, dim, num_layers=num_layers)
        self.readout = MLP([dim, 1], dropout_prob=0.1)

    def forward(self, flux_seq, ts, itg):
        n_ts = flux_seq.shape[1]
        cond = self.cond_embed(torch.stack([ts, itg], -1))
        flux_seq = self.encoder(flux_seq)
        x = torch.cat([flux_seq, cond.unsqueeze(1).repeat(1, n_ts, 1)], -1)
        x, _ = self.lstm(x)
        return self.readout(x[:, -1])


class FluxMLP(nn.Module):
    def __init__(self, dim: int, n_cond: int):
        super().__init__()

        self.cond_embed = ContinuousConditionEmbed(32, n_cond)
        self.mlp = MLP([self.cond_embed.cond_dim, dim, 1], dropout_prob=0.1)

    def forward(self, ts, itg):
        x = self.cond_embed(torch.stack([ts, itg], -1))
        return self.mlp(x)


class QLKNN(nn.Module):
    def __init__(self, n_cond: int):
        super().__init__()
        # TODO load QLKNN weights and match / hardcode other conditions

    def forward(self, itg):
        pass
