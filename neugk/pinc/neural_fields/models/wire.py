from typing import Sequence, Optional

import numpy as np
import torch
from torch import nn
from einops import rearrange

import math

from neugk.models.layers import (
    ContinuousConditionEmbed,
    IntegerConditionEmbed,
    IntegerSincosConditionEmbed,
)


def complex_linear_init_(
    layer: torch.nn.Linear,
    numerator: float = 6,
    mode: str = "fan_in",
    distribution: str = "uniform",
    is_real: bool = False,
):
    fan_in = layer.in_features
    fan_out = layer.out_features
    if mode == "fan_in":
        denom = fan_in
    elif mode == "fan_out":
        denom = fan_out
    elif mode == "fan_avg":
        denom = (fan_in + fan_out) / 2
    else:
        raise ValueError(f"invalid mode: {mode}")
    if distribution == "uniform":
        bound = math.sqrt(numerator / denom)
        real = torch.empty_like(layer.weight, dtype=torch.float).uniform_(-bound, bound)
        imag = torch.empty_like(layer.weight, dtype=torch.float).uniform_(-bound, bound)
    elif distribution == "normal":
        std = math.sqrt(numerator / denom)
        real = torch.empty_like(layer.weight, dtype=torch.float).normal_(0, std)
        imag = torch.empty_like(layer.weight, dtype=torch.float).normal_(0, std)
    elif distribution == "uniform_squared":
        bound = numerator / denom
        real = torch.empty_like(layer.weight, dtype=torch.float).uniform_(-bound, bound)
        imag = torch.empty_like(layer.weight, dtype=torch.float).uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution: {distribution}")
    if is_real:
        layer.weight.data = real
    else:
        layer.weight.data = real + 1j * imag
    if layer.bias is not None:
        layer.bias.data.zero_()


class ComplexGaborLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        w0: float = 10.0,
        sigma0: float = 10.0,
        is_first: bool = False,
        learnable: bool = False,
        skip: bool = False,
    ):
        super().__init__()
        self.skip = skip
        self.is_first = is_first

        dtype = torch.float if is_first else torch.cfloat

        self.w0 = nn.Parameter(w0 * torch.ones(1), learnable)
        self.sigma0 = nn.Parameter(sigma0 * torch.ones(1), learnable)

        self.linear = nn.Linear(in_dim, out_dim, bias=bias, dtype=dtype)

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        if self.is_first:
            numerator = 1
            distribution = "uniform_squared"
            is_real = True
        else:
            numerator = 6 / self.w0**2
            distribution = "uniform"
            is_real = False
        complex_linear_init_(
            self.linear,
            numerator=numerator,
            distribution=distribution,
            is_real=is_real,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        scale_x = self.w0 * self.linear(x)
        scale_y = self.sigma0 * self.linear(x)

        freq_term = torch.exp(1j * scale_x)
        gauss_term = torch.exp(
            -(self.sigma0**2) * (scale_x.abs() ** 2 + scale_y.abs() ** 2)
        )
        x = freq_term * gauss_term

        if self.skip:
            return res + x
        return x


class WIRE(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_layers: int = 2,
        dim: int = 128,
        first_w0: float = 1.0,
        hidden_w0: float = 3.0,
        readout_w0: float = 3.0,
        s0: float = 1.0,
        learnable_w0_s0: bool = False,
        complex_out: bool = False,
        skips: bool = False,
        embed_type: str = "linear",
        grid_size: Optional[Sequence[int]] = None,
    ):
        super().__init__()

        # complex hidden dim
        dim = int(dim / np.sqrt(2))
        self.complex_out = complex_out
        self.embed_type = embed_type

        if not complex_out:
            out_dim = 2 * out_dim

        if embed_type == "sincos_continuous":
            self.coord_embed = ContinuousConditionEmbed(
                dim // 4, in_dim, max_wavelength=500
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
        elif embed_type == "linear":
            self.coord_embed = nn.Identity()
            embed_dim = in_dim
        else:
            raise NotImplementedError(f"embed: {embed_type}")

        self.coord_embed = nn.Sequential(
            self.coord_embed,
            ComplexGaborLayer(
                embed_dim,
                dim,
                w0=first_w0,
                sigma0=s0,
                learnable=learnable_w0_s0,
                is_first=True,
                skip=False,
            ),
        )

        net = []
        for _ in range(n_layers):
            net.append(
                ComplexGaborLayer(
                    dim,
                    dim,
                    w0=hidden_w0,
                    sigma0=s0,
                    learnable=learnable_w0_s0,
                    is_first=False,
                    skip=skips,
                )
            )
        self.net = nn.Sequential(*net)

        self.readout = ComplexGaborLayer(
            dim,
            out_dim,
            w0=readout_w0,
            sigma0=s0,
            learnable=learnable_w0_s0,
            is_first=False,
            skip=False,
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        x = self.coord_embed(coords)
        x = self.net(x)
        x = self.readout(x)

        if self.complex_out:
            x = torch.view_as_real(x).squeeze()
            if x.ndim > 2:
                x = rearrange(x, "b ... m c -> b ... (m c)")
        else:
            x = x.real
        return x
