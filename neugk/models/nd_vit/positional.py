from typing import Sequence

from functools import partial
import numpy as np
from einops import rearrange
import torch
from torch import nn
from warnings import warn

from neugk.models.layers import MLP


class PositionalEmbedding(nn.Module):
    """
    Absolute positional embedding module
    """

    def __init__(
        self,
        dim: int,
        grid_size: Sequence[int],
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


class RPB(nn.Module):
    """Swinv2 relative position bias (RPB) to learn token distances."""

    def __init__(self, space: int, grid_size: Sequence[int], num_heads: int):
        super().__init__()

        self.num_heads = num_heads
        self.cpb_mlp = MLP(
            [space, 512, num_heads],
            act_fn=partial(nn.ReLU, inplace=True),
            bias=[True, False],
        )
        # get relative_coords_table
        coords_nd = []
        for w in grid_size:
            coords_nd.append(torch.arange(-(w - 1), w, dtype=torch.float32))

        rpb = torch.stack(torch.meshgrid(*coords_nd, indexing="ij"))
        for i in range(space):
            rpb[i] = rpb[i] / (grid_size[i] - 1)
        rpb = rearrange(rpb, "d ... -> ... d").unsqueeze(0)
        # normalize to -8, 8
        rpb = 8 * rpb
        rpb = torch.sign(rpb) * torch.log2(torch.abs(rpb) + 1.0) / np.log2(8)
        self.register_buffer("rpb", rpb)  # NOTE: fsdp does not shard buffer

        # index with distances
        grid = torch.stack(
            torch.meshgrid(*[torch.arange(w) for w in grid_size], indexing="ij")
        )  # (space, wD, wH, wW, wU, wV)
        dists = grid.flatten(1).unsqueeze(-1) - grid.flatten(1).unsqueeze(1)

        for i in range(space):
            center = max(np.prod([(2 * w - 1) for w in grid_size[(i + 1) :]]), 1)
            dists[i] = (dists[i] + grid_size[i] - 1) * center

        self.register_buffer("rpb_idx", dists.sum(0))

    def reset_parameters(self, init_weights: str):
        pass  # TODO

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # rpb from swinv2
        sl = x.shape[2]
        rpb = self.cpb_mlp(self.rpb).view(-1, self.num_heads)
        rpb = rpb[self.rpb_idx.flatten()].view(sl, sl, self.num_heads)
        rpb = 16 * torch.sigmoid(rpb)
        return rearrange(rpb, "slx sly h -> h slx sly").unsqueeze(0).contiguous()


class RotaryPE(nn.Module):
    """https://github.com/limefax/rope-nd"""

    def __init__(
        self,
        dim: int,
        grid_size: Sequence[int],
        base: float = 10_000,
        learnable: bool = False,
        use_complex: bool = True,
    ):
        super().__init__()

        k_max = dim // (2 * len(grid_size))
        self.grid_size = grid_size
        self.use_complex = use_complex
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
        if use_complex:
            # convert to complex number to allow easy rotation
            rotations = torch.polar(torch.ones_like(angles), angles)
        else:
            # use real rotation matrix no complex numbers (for bfloat16)
            rotations = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)

        if learnable:
            self.rotations = nn.Parameter(rotations)
        else:
            self.register_buffer("rotations", rotations)

    def forward(self, x: torch.Tensor, flatten: bool = False) -> torch.Tensor:
        if x.ndim < len(self.grid_size) + 3:
            flatten = True
            b, heads, _, c = x.shape
            # reshape to grid for angle multiplication
            x = x.view(b, heads, *self.grid_size, c)

        if self.use_complex:
            # convert input into complex numbers to perform rotation
            x = torch.view_as_complex(x.view(*x.shape[:-1], -1, 2))
            # broadcast batch and head (correct?)
            pe_x = self.rotations[None, None] * x
            pe_x = torch.view_as_real(pe_x).flatten(-2)
        else:
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
            pe_x = rearrange(pe_x, "b heads ... c -> b heads (...) c")
        return pe_x
