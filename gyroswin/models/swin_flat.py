from typing import Sequence, Union, Optional, Type, Dict

from einops import rearrange
import torch
from torch import nn
from functools import partial

from neugk.models.nd_vit.swin_layers import SwinLayer, DiTSwinLayer, FilmSwinLayer
from neugk.models.nd_vit.positional import PositionalEmbedding
from neugk.models.nd_vit.patching import (
    PatchEmbed,
    PatchUnmerging,
    pad_to_blocks,
    unpad,
)


class SwinFlat(nn.Module):
    def __init__(
        self,
        space: int,
        dim: int,
        base_resolution: Sequence[int],
        patch_size: Union[Sequence[int], int] = 4,
        window_size: Union[Sequence[int], int] = 5,
        depth: int = 2,
        num_heads: int = 4,
        in_channels: int = 2,
        out_channels: int = 2,
        abs_pe: bool = False,
        conv_patch: bool = False,
        drop_path: float = 0.1,
        hidden_mlp_ratio: float = 2.0,
        use_checkpoint: bool = False,
        unmerging_hidden_ratio: float = 8.0,
        conditioning: Optional[nn.Module] = None,
        modulation: str = "dit",
        act_fn: nn.Module = nn.GELU,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        expand_act_fn: nn.Module = nn.LeakyReLU,
        init_weights: str = "xavier_uniform",
        patching_init_weights: str = "xavier_uniform",
        patch_skip: bool = False,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.base_resolution = base_resolution
        self.patch_skip = patch_skip
        # set layer type and conditioning
        Layer = SwinLayer
        self.cond_embed = conditioning
        if self.cond_embed is not None:
            if modulation == "dit":
                ModulatedSwinLayer = DiTSwinLayer
            if modulation == "film":
                ModulatedSwinLayer = FilmSwinLayer
            Layer = partial(ModulatedSwinLayer, cond_dim=self.cond_embed.cond_dim)

        padded_base_resolution, _ = pad_to_blocks(base_resolution, patch_size)

        self.patch_embed = PatchEmbed(
            space=space,
            base_resolution=padded_base_resolution,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=dim,
            flatten=False,
            use_conv=conv_patch,
            mlp_ratio=8.0,
            act_fn=act_fn,
        )

        # middle/bottleneck
        self.swin = Layer(
            space,
            dim,
            grid_size=self.patch_embed.grid_size,
            window_size=window_size,
            depth=depth,
            num_heads=num_heads,
            drop_path=drop_path,
            mlp_ratio=hidden_mlp_ratio,
            use_checkpoint=use_checkpoint,
            norm_layer=norm_layer,
            act_fn=act_fn,
        )

        if abs_pe:
            self.ape = PositionalEmbedding(dim, self.patch_embed.grid_size)

        # unpatch
        self.unpatch = PatchUnmerging(
            space,
            dim,
            grid_size=self.patch_embed.grid_size,
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

        self.init_weights = init_weights
        self.patching_init_weights = patching_init_weights
        self.reset_parameters()

    def reset_parameters(self):
        # patching
        self.patch_embed.reset_parameters(self.patching_init_weights)
        self.unpatch.reset_parameters(self.patching_init_weights)
        # conditioning
        if hasattr(self, "cond_embed") and self.cond_embed is not None:
            self.cond_embed.reset_parameters(self.init_weights)
        self.swin.reset_parameters(self.init_weights)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # compress to patch space
        x, pad_axes = self.patch_encode(x)
        if self.patch_skip:
            first_res = x.clone()

        # backbone
        cond = self.condition(kwargs)

        x = self.swin(x, **cond)

        # expand to original
        if self.patch_skip:
            x = torch.cat([x, first_res], -1)

        x = self.patch_decode(x, cond["condition"], pad_axes)

        return x

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
