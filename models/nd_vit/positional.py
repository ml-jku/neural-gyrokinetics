from typing import Sequence

from einops import rearrange
import torch
from torch import nn


class PositionalEmbedding(nn.Module):
    """
    Absolute positional embedding module
    """

    def __init__(
        self,
        dim: int,
        grid_size: tuple,
        learnable: bool = False,
        init_weights: str = "sincos",
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.dim = dim
        self.learnable = learnable
        self.init_weights = init_weights

        pos_embed = torch.zeros(1, *self.grid_size, self.dim)
        if learnable:
            self.pos_embed = nn.Parameter(pos_embed)
        else:
            self.register_buffer("pos_embed", pos_embed)

        self.reset_parameters()

    def reset_parameters(self):
        if self.init_weights == "rand":
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        elif self.init_weights == "sincos":
            try:
                from kappamodules.functional.pos_embed import (
                    get_sincos_pos_embed_from_seqlens as sincos_pos_embed,
                )
            except ImportError:
                raise ImportError("pip install kappamodules")

            pos_embed = sincos_pos_embed(self.grid_size, self.dim)[None]
            # replace param
            if isinstance(self.pos_embed, nn.Parameter):
                self.pos_embed.data.copy_(pos_embed)
            else:
                self.pos_embed.copy_(pos_embed)
        else:
            raise NotImplementedError

    def forward(self, x):
        ndim = x.ndim
        if ndim < 4:
            b, c = x.shape[0], x.shape[-1]
            x = x.view(b, *self.grid_size, c)
        x = x + self.pos_embed
        if ndim < 4:
            x = rearrange(x, "b ... c -> b (...) c")

        return x


class RotaryQueryKeyPE(torch.nn.Module):
    """https://github.com/limefax/rope-nd"""

    def __init__(self, dim: int, grid_size: Sequence[int], base: float = 10000):
        super().__init__()

        k_max = dim // (2 * len(grid_size))
        assert (
            dim % k_max == 0 and k_max > 1
        ), f"dim ({dim}) not divisible by 2 * len(grid_size) (={2 * len(grid_size)})"
        # tensor of angles to use
        theta_ks = 1 / (base ** (torch.arange(k_max) / k_max))
        # create a stack of angles multiplied by position
        angles = torch.cat(
            [
                t.unsqueeze(-1) * theta_ks
                for t in torch.meshgrid(
                    [torch.arange(d) for d in grid_size], indexing="ij"
                )
            ],
            dim=-1,
        )
        # convert to complex number to allow easy rotation
        rotations = torch.polar(torch.ones_like(angles), angles)
        # store in a buffer so it can be saved in model parameters
        self.register_buffer("rotations", rotations)

    def forward(self, x):
        # convert input into complex numbers to perform rotation
        x = torch.view_as_complex(x.view(*x.shape[:-1], -1, 2))
        pe_x = self.rotations.unsqueeze(0) * x
        return torch.view_as_real(pe_x).flatten(-2)
