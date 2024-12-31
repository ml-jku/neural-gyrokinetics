from einops import rearrange
import torch
from torch import nn


class PositionalEmbedding(nn.Module):
    """
    Absolute positional embedding module
    """

    def __init__(
        self, dim: int, grid_size: tuple, learnable: bool = False, init: str = "sincos"
    ) -> None:
        super().__init__()
        self.grid_size = grid_size

        if init == "rand":
            pos_embed = torch.zeros(1, *grid_size, dim)
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
