from typing import Type, Optional, Sequence, Union, Tuple

from itertools import product
from math import ceil
import numpy as np
from einops import rearrange
import torch
import torch.nn.functional as F
from torch import nn


def pad_to_shape(
    x: torch.Tensor, shape: Sequence[int]
) -> Tuple[torch.Tensor, Sequence[int]]:
    pad_axes = []
    if any(x1 != x2 for x1, x2 in zip(x.shape[1:-1], shape)):
        for i, s in enumerate(shape):
            assert s >= x.shape[1 + i]
            pad_axes.append(0)
            pad_axes.append(s - x.shape[1 + i])  # +1 for batch

        # last tuple first in
        pad_axes = pad_axes[::-1]
        if any([p > 0 for p in pad_axes]):
            x = F.pad(x, (0, 0, *pad_axes))

    return x, pad_axes


def pad_to_blocks(
    x: Union[Tuple[int], torch.Tensor], blocks: Sequence[int]
) -> Tuple[torch.Tensor, Sequence[int]]:
    x_shp = x.shape if isinstance(x, torch.Tensor) else (None, *x)
    pad_axes = []
    for i, w in enumerate(blocks):
        pad_axes.append((w - x_shp[1 + i] % w) % w)  # +1 for batch
        pad_axes.append(0)

    if isinstance(x, torch.Tensor):
        # pad tensor
        # last tuple first in
        pad_axes = pad_axes[::-1]
        if any([p > 0 for p in pad_axes]):
            x = F.pad(x, (0, 0, *pad_axes))
    else:
        # compute padded shape
        x = tuple([x[i] + pad_axes[i * 2] for i in range(len(x))])

    return x, pad_axes


def unpad(
    x: torch.Tensor, pad_axes: Sequence[int], base_grid: Sequence[int]
) -> torch.Tensor:
    if any([p > 0 for p in pad_axes]):
        # unpad to original resolution
        x = x[:, *[slice(0, r) for r in base_grid], :].contiguous()

    return x


class PatchEmbed(nn.Module):
    def __init__(
        self,
        space: int,
        img_size: Sequence[int],
        patch_size: Sequence[int],
        in_chans: int = 2,
        embed_dim: int = 24,
        norm_layer: nn.Module = None,
        flatten: bool = True,
        use_conv: bool = False,
        act_fn: nn.Module = nn.LeakyReLU,
        mlp_ratio: float = 8.0,
    ):
        assert len(img_size) == space, f"Image size must be {space}D"
        assert len(patch_size) == space, f"Patch size must be {space}D"

        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = [ceil(img_size[i] / patch_size[i]) for i in range(space)]
        self.embed_dim = embed_dim
        self.flatten = flatten
        self.use_conv = use_conv
        self.space = space
        self.in_chans = in_chans

        hidden_dim = int(embed_dim * mlp_ratio)

        if use_conv:
            if space == 2:
                Conv = nn.Conv2d
            elif space == 3:
                Conv = nn.Conv3d
            else:
                raise NotImplementedError

            self.patch = Conv(
                in_chans,
                embed_dim,
                kernel_size=patch_size,
                stride=patch_size,
                bias=False,
            )
        else:
            self.patch = nn.Sequential(
                nn.Linear(in_chans * np.prod(patch_size), hidden_dim),
                act_fn(),
                nn.Linear(hidden_dim, embed_dim, bias=False),
            )

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        """
        Args:
            x: Tensor (B, C, ...)

        Returns:
            Tensor (B, ..., C)
        """
        x = self.proj(x)
        if self.flatten:
            x = rearrange(x, "b ... c -> b (...) c")
        x = self.norm(x)
        return x

    def proj(self, x):
        if self.use_conv:
            x = rearrange(x, "b ... c -> b c ...")
            x = self.patch(x)
            x = rearrange(x, "b c ... -> b ... c")
        else:
            patch_view = []
            patch_permute_even = []
            patch_permute_odd = []
            for i in range(self.space):
                # + 1 for batch dimension
                patch_view.append(ceil(x.shape[1 + i] / self.patch_size[i]))
                patch_view.append(self.patch_size[i])
                patch_permute_even.append(1 + i * 2)
                patch_permute_odd.append(1 + i * 2 + 1)

            patch_permute = patch_permute_even + patch_permute_odd

            b, c = x.shape[0], x.shape[-1]
            # TODO check if permute is correct or transposes image?
            x = x.view(b, *patch_view, c).permute(0, *patch_permute, -1)
            # flatten patches
            x = x.flatten(1 + self.space)
            # linear flat-patch projection
            x = self.patch(x)

        return x


class PatchMerging(nn.Module):
    """
    Patch merging layer.
    """

    def __init__(
        self,
        space: int,
        dim: int,
        grid_size: Sequence[int],
        norm_layer: Type[nn.LayerNorm] = nn.LayerNorm,
        c_multiplier: int = 2,
    ) -> None:
        """
        Args:
            space: spatial dimensionality.
            dim: number of feature channels.
            norm_layer: normalization layer.
        """

        super().__init__()
        self.space = space
        self.dim = dim

        # NOTE only merge those with more than two patches -> otherwise risk of nans
        self.merge_subspace = [g > 2 for g in grid_size]
        self.grid_size = grid_size
        # grid resolution after patch merging
        self.target_grid_size = [
            ceil(g / 2) if m else g for g, m in zip(grid_size, self.merge_subspace)
        ]

        n_merges = sum(self.merge_subspace)
        self.norm = norm_layer(2**n_merges * dim) if norm_layer else nn.Identity()
        self.reduction = nn.Linear(2**n_merges * dim, c_multiplier * dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # must pad to even shape
        # TODO for now pad to multiple of 2. can do better?
        x, _ = pad_to_blocks(x, self.space * (2,))

        # even/odd shifted patch selection along all axes (only for accepted subspaces)
        subspaces01 = []
        for sub in product(*[[0, 1] if sub else [0] for sub in self.merge_subspace]):
            merge_axis = []
            for i, s in enumerate(sub):
                if self.merge_subspace[i]:
                    # alternated slice
                    merge_axis.append(slice(s, None, 2))
                else:
                    # cannot split this patch
                    merge_axis.append(slice(0, None))
            subspaces01.append(merge_axis)

        x = torch.cat([x[:, *merge_ax, :] for merge_ax in subspaces01], -1)

        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchUnmerging(nn.Module):
    def __init__(
        self,
        space: int,
        dim: int,
        grid_size: Sequence[int],
        expand_by: Optional[Union[Sequence[int], int]] = None,
        target_grid_size: Optional[Sequence[int]] = None,
        norm_layer: Type[nn.LayerNorm] = nn.LayerNorm,
        c_multiplier: int = 2,
        out_channels: Optional[int] = None,
        flatten: bool = False,
        use_conv: bool = False,
        act_fn: nn.Module = nn.LeakyReLU,
        mlp_ratio: float = 8.0,
        patch_skip: bool = False,
    ):
        super().__init__()
        assert expand_by is not None or target_grid_size is not None

        self.space = space
        self.dim = dim
        self.grid_size = grid_size
        self.flatten = flatten
        self.use_conv = use_conv

        if target_grid_size is not None:
            self.target_grid_size = target_grid_size
            # target_grid_size overwrites expand_by (uncudes padding from patch merging)
            expand_by = [ceil(t / g) for g, t in zip(grid_size, target_grid_size)]

        if isinstance(expand_by, int):
            expand_by = (expand_by,) * space
        self.expand_by = expand_by

        if target_grid_size is None:
            self.target_grid_size = [g * e for g, e in zip(grid_size, expand_by)]

        hidden_dim = int(np.prod(expand_by) * mlp_ratio)

        # NOTE out_channels overrides c_multiplier
        dim_out = dim // c_multiplier if out_channels is None else out_channels

        if use_conv:
            if space == 2:
                Conv = nn.ConvTranspose2d
            elif space == 3:
                Conv = nn.ConvTranspose3d
            else:
                raise NotImplementedError

            self.expansion = Conv(
                dim, dim_out, kernel_size=grid_size, stride=grid_size, bias=False
            )
        else:
            self.expansion = nn.Sequential(
                nn.Linear(dim * (2 if patch_skip else 1), hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, np.prod(expand_by) * dim_out),
            )

        self.norm = norm_layer(dim_out) if norm_layer else nn.Identity()

    def forward(self, x):
        ndim = x.ndim
        if ndim < 4:
            b, c = x.shape[0], x.shape[-1]
            x = x.view(b, *self.grid_size, c)

        if self.use_conv:
            x = rearrange(x, "b ... c -> b c ...")
            x = self.expansion(x)
            x = rearrange(x, "b c ... -> b ... c")
        else:
            x = self.up_proj(x)

        # must unpad beause of patch merging
        # TODO for now. can do better?
        _, pad_axes = pad_to_blocks(self.target_grid_size, self.space * (2,))
        x = unpad(x, pad_axes, self.target_grid_size)

        if self.flatten:
            x = rearrange(x, "b ... c -> b (...) c")

        x = self.norm(x)

        return x

    def up_proj(self, x):
        b = x.shape[0]
        # linear expansion of patches
        x = self.expansion(x)
        patch_permute = []
        for i in range(self.space):
            patch_permute.append(1 + i)
            patch_permute.append(1 + i + self.space)
        x = x.view(b, *self.grid_size, *self.expand_by, -1)
        x = x.permute(0, *patch_permute, -1)
        # recover patch size by flattening patch count and size in pairs
        for i in range(self.space):
            x = x.flatten(i + 1, i + 2)
        return x
