from typing import Sequence, Optional

import numpy as np

import torch
from torch import nn

from neugk.models.layers import (
    ContinuousConditionEmbed,
    IntegerConditionEmbed,
    IntegerSincosConditionEmbed,
)


class Sine(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        is_first: bool = False,
        w0: float = 30.0,
        skip: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        # self.register_buffer("w0", torch.tensor(w0))
        self.w0 = w0
        self.is_first = is_first
        self.skip = skip

        self.in_dim = in_dim
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        if self.is_first:
            self.linear.weight.uniform_(-1 / self.in_dim, 1 / self.in_dim)
        else:
            self.linear.weight.uniform_(
                -np.sqrt(6 / self.in_dim) / self.w0,
                np.sqrt(6 / self.in_dim) / self.w0,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = torch.sin(self.w0 * self.linear(x))
        if self.skip:
            return res + x
        return x


class SineClip(Sine):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = 0.5 * torch.sin(self.w0 * self.linear(x) + 1)
        if self.skip:
            return res + x
        return x


class SIREN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_layers: int,
        dim: int,
        linear_readout: bool = True,
        first_w0: float = 1.0,
        hidden_w0: float = 10.0,
        readout_w0: float = 1.0,
        embed_type: str = "lin",
        skips: bool = False,
        clip_out: bool = False,
        bias: bool = True,
        grid_size: Optional[Sequence[int]] = None,
    ):
        super().__init__()

        self.dim = dim
        self.out_dim = out_dim
        self.n_layers = n_layers
        self.embed_type = embed_type

        if embed_type == "sincos_continuous":
            self.coord_embed = ContinuousConditionEmbed(
                dim // 4, in_dim, max_wavelength=500, act_fn=nn.Identity
            )
            embed_dim = self.coord_embed.cond_dim
        elif embed_type == "sincos_discrete":
            self.coord_embed = IntegerSincosConditionEmbed(
                dim,
                in_dim,
                max_size=grid_size,
                use_mlp=False,
            )
            embed_dim = self.coord_embed.cond_dim
        elif embed_type == "discrete":
            self.coord_embed = IntegerConditionEmbed(
                dim,
                in_dim,
                max_size=grid_size,
                use_mlp=False,
            )
            embed_dim = self.coord_embed.cond_dim
        else:
            self.coord_embed = Sine(in_dim, dim, is_first=True, w0=first_w0, skip=False)
            embed_dim = dim

        net = []
        skip0 = embed_dim == dim
        blk = Sine(embed_dim, dim, is_first=False, w0=hidden_w0, skip=skip0, bias=bias)
        net.append(blk)
        for _ in range(n_layers - 1):
            blk = Sine(dim, dim, is_first=False, w0=hidden_w0, skip=skips, bias=bias)
            net.append(blk)
        self.net = nn.ModuleList(net)

        if linear_readout:
            dtype = torch.float
            self.readout = nn.Linear(dim, out_dim, dtype=dtype)
            with torch.no_grad():
                const = np.sqrt(6 / dim) / max(hidden_w0, 1e-12)
                self.readout.weight.uniform_(-const, const)
        else:
            if clip_out:
                self.readout = SineClip(dim, out_dim, w0=readout_w0, skip=skips)
            else:
                self.readout = Sine(dim, out_dim, w0=readout_w0, skip=skips, bias=bias)

    def forward(self, coords):
        x = self.coord_embed(coords)
        for layer in self.net:
            x = layer(x)
        x = self.readout(x)
        return x
