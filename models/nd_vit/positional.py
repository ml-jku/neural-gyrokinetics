from typing import Sequence

from einops import rearrange
import torch
from torch import nn
from warnings import warn


class PositionalEmbedding(nn.Module):
    """
    Absolute positional embedding module
    """

    def __init__(
        self,
        dim: int,
        grid_size: tuple,
        learnable: bool = False,
        init_weights: str = "rand",
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

        if dim < len(grid_size) and init_weights == "sincos":
            init_weights = "rand"
            warn(
                "Sincos initialization only works if len(grid_size) < dim"
                "(returns zero padding otherwise). Switching to random"
            )

        self.reset_parameters()

    def reset_parameters(self):
        if self.init_weights == "zeros":
            pass
        if self.init_weights == "rand":
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        elif self.init_weights == "sincos":
            try:
                from kappamodules.functional.pos_embed import (
                    get_sincos_pos_embed_from_seqlens as sincos_pos_embed,
                )
            except ImportError:
                raise ImportError("pip install kappamodules")
            # NOTE not reccomended when dim << 5, needs padding (returns zeros)
            pos_embed = sincos_pos_embed(self.grid_size, self.dim)
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


class RotaryPE(torch.nn.Module):
    """https://github.com/limefax/rope-nd"""

    def __init__(
        self,
        dim: int,
        grid_size: Sequence[int],
        base: float = 10000,
        learnable: bool = False,
    ):
        super().__init__()

        k_max = dim // (2 * len(grid_size))
        self.grid_size = grid_size
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
        if learnable:
            self.rotations = nn.Parameter(rotations)
        else:
            self.register_buffer("rotations", rotations)

    def forward(self, x):
        flatten = False
        if x.ndim < len(self.grid_size) + 3:
            # reshape to grid for angle multiplication
            flatten = True
            b, heads, _, c = x.shape
            x = x.view(b, heads, *self.grid_size, c)
        # convert input into complex numbers to perform rotation
        x = torch.view_as_complex(x.view(*x.shape[:-1], -1, 2))
        # broadcast batch and head (correct?)
        pe_x = self.rotations[None, None] * x
        pe_x = torch.view_as_real(pe_x).flatten(-2)
        if flatten:
            pe_x = rearrange(pe_x, "b heads ... c -> b heads (...) c")
        return pe_x


class RealRotaryPE(nn.Module):
    """Rotary Positional Embedding with rotation matrix (no complex numbers)."""

    def __init__(
        self,
        dim: int,
        grid_size: Sequence[int],
        base: float = 10000,
        learnable: bool = False,
    ):
        super().__init__()

        k_max = dim // (2 * len(grid_size))
        self.grid_size = grid_size
        assert (
            dim % k_max == 0 and k_max > 1
        ), f"dim ({dim}) not divisible by 2 * len(grid_size) (={2 * len(grid_size)})"
        # Compute theta_ks as inverse frequencies
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
        # rotation matrix instead of polar
        rotations = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        if learnable:
            self.rotations = nn.Parameter(rotations)
        else:
            self.register_buffer("rotations", rotations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flatten = False
        original_shape = x.shape
        if x.ndim < len(self.grid_size) + 3:
            # reshape to grid for angle multiplication
            flatten = True
            b, heads, _, c = x.shape
            x = x.view(b, heads, *self.grid_size, c)
        # reshape into pairs for re / im parts
        x = x.view(*x.shape[:-1], -1, 2)  # [..., c//2, 2]
        x_re, x_im = x[..., 0], x[..., 1]
        rot_cos = self.rotations[None, None, ..., 0]
        rot_sin = self.rotations[None, None, ..., 1]
        # apply rotation with real arithmetic
        pe_x_re = x_re * rot_cos - x_im * rot_sin
        pe_x_im = x_re * rot_sin + x_im * rot_cos
        pe_x = torch.stack([pe_x_re, pe_x_im], dim=-1)
        pe_x = pe_x.flatten(-2)
        if flatten:
            pe_x = pe_x.view(*original_shape)
        return pe_x
