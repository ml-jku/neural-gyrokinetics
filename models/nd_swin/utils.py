from typing import Type, Optional, Sequence, Callable, Union, Tuple

from itertools import product
from math import ceil
import numpy as np
from einops import rearrange
import torch
import torch.nn.functional as F
from torch import nn


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False."""

    def __init__(
        self,
        ndim: int = -1,
        weight: bool = True,
        bias: bool = False,
        eps: float = 1e-5,
        residual_weight: bool = True,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(ndim)) if weight else None
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps
        self.residual_weight = residual_weight
        self.ndim = ndim
        self.reset_parameters()

    @property
    def weight_proxy(self) -> torch.Tensor:
        if self.weight is None:
            return None
        if self.residual_weight:
            return 1.0 + self.weight
        else:
            return self.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            normalized_shape=(self.ndim,),
            weight=self.weight_proxy,
            bias=self.bias,
            eps=self.eps,
        )

    def reset_parameters(self):
        if self.weight_proxy is not None:
            if self.residual_weight:
                nn.init.zeros_(self.weight)
            else:
                nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


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
        pad_axes.append(0)
        pad_axes.append((w - x_shp[1 + i] % w) % w)  # +1 for batch

    if isinstance(x, torch.Tensor):
        # pad tensor
        # last tuple first in
        pad_axes = pad_axes[::-1]
        if any([p > 0 for p in pad_axes]):
            x = F.pad(x, (0, 0, *pad_axes))
    else:
        # compute padded shape
        x = tuple([x[i] + pad_axes[1 + i * 2] for i in range(len(x))])

    return x, pad_axes


def unpad(
    x: torch.Tensor, pad_axes: Sequence[int], base_resolution: Sequence[int]
) -> torch.Tensor:
    if any([p > 0 for p in pad_axes]):
        # unpad to original resolution
        x = x[:, *[slice(0, r) for r in base_resolution], :].contiguous()

    return x


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: Sequence[int],
        patch_size: Sequence[int],
        in_chans: int = 2,
        embed_dim: int = 24,
        norm_layer: Callable = None,
        flatten: bool = True,
        use_conv: bool = False,
        space: int = 2,
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

        if use_conv:
            from convNd.convNd import convNd  # noqa

            self.patch = convNd(
                in_chans,
                embed_dim,
                num_dims=space,
                kernel_size=patch_size,
                stride=patch_size,
                padding=0,
            )
        else:
            self.patch = nn.Linear(in_chans * np.prod(patch_size), embed_dim)

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


class PositionalEmbedding(nn.Module):
    """
    Absolute positional embedding module
    """

    def __init__(
        self, dim: int, grid_size: tuple, learnable: bool = False, init: str = "sincos"
    ) -> None:
        super().__init__()
        d, h, w, u, v = grid_size
        self.grid_size = grid_size

        if init == "rand":
            pos_embed = torch.zeros(1, dim, d, h, w, u, v)
            nn.init.trunc_normal_(pos_embed, std=0.02)
        if init == "sincos":
            try:
                from kappamodules.functional.pos_embed import (
                    get_sincos_pos_embed_from_seqlens,
                )
            except ImportError:
                raise ImportError("pip install kappamodules")

            pos_embed = get_sincos_pos_embed_from_seqlens(grid_size, dim)[None]

        if learnable:
            self.pos_embed = nn.Parameter(pos_embed)
        else:
            self.register_buffer("pos_embed", pos_embed)

    def forward(self, x):
        ndim = x.ndim
        if ndim < 4:
            b, c = x.shape[0], x.shape[-1]
            x = x.view(b, *self.grid_size, c)
        x = x + self.pos_embed
        if ndim < 4:
            x = rearrange(x, "b ... c -> b (...) c")

        return x


class PatchMerging(nn.Module):
    """
    Patch merging layer.
    """

    def __init__(
        self,
        space: int,
        dim: int,
        norm_layer: Type[LayerNorm] = nn.LayerNorm,
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

        self.norm = norm_layer(2**space * dim)
        self.reduction = nn.Linear(2**space * dim, c_multiplier * dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # must pad to even shape
        # TODO for now pad to multiple of 2. can do better?
        x, _ = pad_to_blocks(x, self.space * (2,))
        x = torch.cat(
            [
                x[:, *[slice(i, None, 2) for i in idxs], :]
                for idxs in product(*[range(2) for _ in range(self.space)])
            ],
            -1,
        )
        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchUnmerging(nn.Module):
    def __init__(
        self,
        space: int,
        dim: int,
        grid_size: Sequence[int],
        expand_by: Union[Sequence[int], int],
        target_grid_size: Optional[Sequence[int]] = None,
        norm_layer: Type[LayerNorm] = LayerNorm,
        c_multiplier: int = 2,
        out_channels: Optional[int] = None,
        flatten: bool = False,
        use_conv: bool = False,
    ):
        super().__init__()
        self.space = space
        self.dim = dim
        self.grid_size = grid_size
        if isinstance(expand_by, int):
            expand_by = (expand_by,) * space
        self.expand_by = expand_by
        self.flatten = flatten
        self.use_conv = use_conv
        self.target_grid_size = (
            target_grid_size
            if target_grid_size
            else [g * e for g, e in zip(grid_size, expand_by)]
        )

        # NOTE out_channels overrides c_multiplier
        dim_out = dim // c_multiplier if out_channels is None else out_channels

        if use_conv:
            from convNd.convNd import convNd

            self.expansion = convNd(
                dim,
                dim_out,
                num_dims=space,
                kernel_size=tuple(grid_size),
                stride=tuple(grid_size),
                padding=(0,) * space,
                is_transposed=True,
                use_bias=False,
            )
        else:
            self.expansion = nn.Linear(dim, np.prod(expand_by) * dim_out, bias=False)

        if norm_layer is not None:
            self.norm = norm_layer(dim // c_multiplier)

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

        if hasattr(self, "norm"):
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


class DropPath(nn.Module):
    """Stochastic drop paths per sample for residual blocks.
    Based on:
    https://github.com/rwightman/pytorch-image-models
    """

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True) -> None:
        """
        Args:
            drop_prob: drop path probability.
            scale_by_keep: scaling by non-dropped probability.
        """
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

        if not (0 <= drop_prob <= 1):
            raise ValueError("Drop path prob should be between 0 and 1.")

    def drop_path(
        self,
        x,
        drop_prob: float = 0.0,
        training: bool = False,
        scale_by_keep: bool = True,
    ):
        if drop_prob == 0.0 or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

    def forward(self, x):
        return self.drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


# TODO check this out
class ResdualDropPath(nn.Sequential):
    """
    Efficiently drop paths (Stochastic Depth) per sample such that dropped samples are not processed.
    This is a subclass of nn.Sequential and can be used either as standalone Module or like nn.Sequential.
    Examples::
        >>> # use as nn.Sequential module
        >>> sequential_droppath = DropPath(nn.Linear(4, 4), drop_prob=0.2)
        >>> y = sequential_droppath(torch.randn(10, 4))

        >>> # use as standalone module
        >>> standalone_layer = nn.Linear(4, 4)
        >>> standalone_droppath = DropPath(drop_prob=0.2)
        >>> y = standalone_droppath(torch.randn(10, 4), standalone_layer)
    """

    def __init__(
        self,
        *args,
        drop_prob: float = 0.0,
        scale_by_keep: bool = True,
        stochastic_drop_prob: bool = False,
        drop_prob_tolerance: float = 0.01,
    ):
        super().__init__(*args)
        assert 0.0 <= drop_prob < 1.0
        self._drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
        self.stochastic_drop_prob = stochastic_drop_prob
        self.drop_prob_tolerance = drop_prob_tolerance

    @property
    def drop_prob(self):
        return self._drop_prob

    @drop_prob.setter
    def drop_prob(self, value):
        assert 0.0 <= value < 1.0
        self._drop_prob = value

    @property
    def keep_prob(self):
        return 1.0 - self.drop_prob

    def forward(self, x, residual_path=None, residual_path_kwargs=None):
        assert (len(self) == 0) ^ (residual_path is None)
        residual_path_kwargs = residual_path_kwargs or {}
        if self.drop_prob == 0.0 or not self.training:
            if residual_path is None:
                return x + super().forward(x, **residual_path_kwargs)
            else:
                return x + residual_path(x, **residual_path_kwargs)
        bs = len(x)
        # for small batchsizes its not possible to do it efficiently
        # e.g. batchsize 2 with drop_rate=0.05 would drop 1 sample and therefore increase the drop_rate to 0.5
        # resolution: fall back to inefficient version
        keep_count = max(int(bs * self.keep_prob), 1)
        # allow some level of tolerance
        actual_keep_prob = keep_count / bs
        drop_path_delta = self.keep_prob - actual_keep_prob
        # if drop_path_delta > self.drop_prob_tolerance:
        #     warnings.warn(
        #         f"efficient stochastic depth (DropPath) would change drop_path_rate by {drop_path_delta:.4f} "
        #         f"because the batchsize is too small to accurately drop {bs - keep_count} samples per forward pass"
        #         f" -> forcing stochastic_drop_prob=True drop_path_rate={self.drop_prob}"
        #     )

        # inefficient drop_path
        if self.stochastic_drop_prob or drop_path_delta > self.drop_prob_tolerance:
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = x.new_empty(shape).bernoulli_(self.keep_prob)
            if self.scale_by_keep:
                random_tensor.div_(self.keep_prob)
            if residual_path is None:
                return x + super().forward(x, **residual_path_kwargs) * random_tensor
            else:
                return x + residual_path(x, **residual_path_kwargs) * random_tensor

        # generate indices to keep (propagated through transform path)
        scale = bs / keep_count
        perm = torch.randperm(bs, device=x.device)[:keep_count]

        # propagate
        if self.scale_by_keep:
            alpha = scale
        else:
            alpha = 1.0
        # reduce kwargs (e.g. used for DiT block where scale/shift/gate is passed and also has to be reduced)
        residual_path_kwargs = {
            key: value[perm] if torch.is_tensor(value) else value
            for key, value in residual_path_kwargs.items()
        }
        if residual_path is None:
            residual = super().forward(x[perm], **residual_path_kwargs)
        else:
            residual = residual_path(x[perm], **residual_path_kwargs)
        return torch.index_add(
            x.flatten(start_dim=1),
            dim=0,
            index=perm,
            source=residual.to(x.dtype).flatten(start_dim=1),
            alpha=alpha,
        ).view_as(x)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob, 3):0.3f}"
