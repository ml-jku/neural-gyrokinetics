from typing import Dict, Sequence

import torch
import torch.nn as nn
from einops import rearrange

from neuralop.models import FNO, FNO3d

from neugk.models.nd_vit.patching import (
    PatchEmbed,
    PatchExpand,
    pad_to_blocks,
    unpad,
)


class Df5DTFNO(FNO):
    def __init__(
        self,
        dim: int,
        base_resolution: Sequence[int],
        in_channels: int = 2,
        out_channels: int = 2,
        num_layers: int = 4,
        mode_scale: int = 4,
    ):
        super().__init__(
            n_modes=[r // mode_scale for r in base_resolution],
            hidden_channels=dim,
            in_channels=in_channels,
            out_channels=out_channels,
            n_layers=num_layers,
            factorization="tucker",
            implementation="factorized",
            rank=0.05,
        )
        self.base_resolution = tuple(base_resolution)

    def forward(self, df: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        _ = kwargs
        df = super().forward(df, output_shape=self.base_resolution)
        return {"df": df}


class DfVSpace3DTFNO(FNO3d):
    def __init__(
        self,
        dim: int,
        base_resolution: Sequence[int],
        in_channels: int = 2,
        out_channels: int = 2,
        num_layers: int = 4,
        mode_scale: int = 2,
    ):
        super().__init__(
            n_modes_height=base_resolution[2] // mode_scale,
            n_modes_width=base_resolution[3] // mode_scale,
            n_modes_depth=base_resolution[4] // mode_scale,
            hidden_channels=dim,
            # vspace to channels
            in_channels=in_channels * base_resolution[0] * base_resolution[1],
            out_channels=out_channels * base_resolution[0] * base_resolution[1],
            n_layers=num_layers,
            factorization="tucker",
            implementation="factorized",
            rank=0.05,
        )
        self.base_resolution = tuple(base_resolution)

    def forward(self, df: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        _ = kwargs
        vp, vm = df.shape[2], df.shape[3]
        df = rearrange(df, "b c vp vm s x y -> b (c vp vm) s x y")
        df = super().forward(df, output_shape=self.base_resolution[2:])
        df = rearrange(df, "b (c vp vm) s x y -> b c vp vm s x y", vp=vp, vm=vm)
        return {"df": df}


class DfLocal5DTFNO(nn.Module):
    """Local-FNO https://arxiv.org/pdf/2411.11348"""

    def __init__(
        self,
        dim: int,
        base_resolution: Sequence[int],
        patch_size: Sequence[int],
        in_channels: int = 2,
        out_channels: int = 2,
        num_layers: int = 4,
    ):
        super().__init__()

        padded_base_resolution, _ = pad_to_blocks(base_resolution, patch_size)
        self.base_resolution = tuple(base_resolution)
        self.patch_size = patch_size

        self.patch = PatchEmbed(
            space=5,
            base_resolution=padded_base_resolution,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=dim,
            flatten=False,
            mlp_depth=1,
        )
        self.fno = Df5DTFNO(
            dim,
            base_resolution=self.patch.grid_size,
            in_channels=dim,
            out_channels=dim,
            num_layers=num_layers,
            mode_scale=1,  # use every mode in patched space
        )
        self.unpatch = PatchExpand(
            space=5,
            dim=dim,
            grid_size=self.patch.grid_size,
            expand_by=patch_size,
            out_channels=out_channels,
            flatten=False,
            norm_layer=None,
            mlp_depth=1,
        )

    def forward(self, df: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        _ = kwargs
        df = rearrange(df, "b c ... -> b ... c")
        df, pad_axes = pad_to_blocks(df, self.patch_size)
        df = self.patch(df)
        df = rearrange(df, "b ... c -> b c ...")
        df = self.fno(df)["df"]
        df = rearrange(df, "b c ... -> b ... c")
        df = self.unpatch(df)
        df = unpad(df, pad_axes, self.base_resolution)
        df = rearrange(df, "b ... c -> b c ...")
        return {"df": df}
