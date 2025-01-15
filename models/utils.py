from typing import Optional, Sequence, Union

from einops import rearrange
import torch
from torch import nn
from kappamodules.functional.pos_embed import get_sincos_1d_from_seqlen


def seq_weight_init(weight_init_fn, bias_init_fn=None):
    if bias_init_fn is None:
        bias_init_fn = nn.init.zeros_

    def _apply(m):
        if isinstance(m, nn.Linear):
            weight_init_fn(m.weight)
            if hasattr(m, "bias") and m.bias is not None:
                bias_init_fn(m.bias)

    return _apply


class MLP(nn.Module):
    def __init__(
        self,
        latents: Sequence[int],
        act_fn: nn.Module = nn.GELU,
        bias: Union[bool, Sequence[bool]] = True,
        dropout_prob: float = 0.0,
    ):
        super().__init__()
        if isinstance(bias, bool):
            bias = [bias] * (len(latents) - 1)
        dropout = nn.Dropout(dropout_prob) if dropout_prob > 0 else nn.Identity()
        mlp = []
        for i, (lat_i, lat_i2) in enumerate(zip(latents, latents[1:])):
            mlp.append(nn.Linear(lat_i, lat_i2, bias=bias[i]))
            mlp.append(dropout)
            if i != len(latents) - 2:
                mlp.append(act_fn())
        self.mlp = nn.Sequential(*mlp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class IntegerConditionEmbed(nn.Module):
    def __init__(self, dim: int, max_size: int, init_weights: Optional[str] = None):
        super().__init__()
        cond_dim = dim * 4
        self.max_size = max_size
        self.dim = dim
        self.cond_dim = cond_dim
        self.register_buffer("cond_embed", get_sincos_1d_from_seqlen(max_size, dim))
        self.mlp = nn.Sequential(
            nn.Linear(dim, cond_dim),
            nn.SiLU(),
        )

        if init_weights is not None:
            self.reset_parameters(init_weights)

    def reset_parameters(self, init_weights):
        if init_weights == "torch" or init_weights is None:
            pass
        elif init_weights == "xavier_uniform":
            self.mlp.apply(seq_weight_init(nn.init.xavier_uniform_))
        elif init_weights in ["truncnormal", "truncnormal002"]:
            self.mlp.apply(seq_weight_init(nn.init.trunc_normal_))
        else:
            raise NotImplementedError

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        # checks + preprocess
        assert cond.numel() == len(cond)
        cond = cond.flatten().long()
        return self.mlp(self.cond_embed[cond])


class ContinuousConditionEmbed(nn.Module):
    def __init__(
        self,
        dim: int,
        n_cond: int,
        max_wavelength: int = 10_000,
        init_weights: Optional[str] = None,
    ):
        super().__init__()
        self.dim = dim
        self.n_cond = n_cond
        self.ndim_padding = dim % n_cond
        dim_per_ndim = (dim - self.ndim_padding) // n_cond
        self.sincos_padding = dim_per_ndim % 2
        self.max_wavelength = max_wavelength
        self.padding = self.ndim_padding + self.sincos_padding * n_cond
        cond_per_wave = (self.dim - self.padding) // n_cond
        assert cond_per_wave > 0
        self.register_buffer(
            "omega",
            1.0 / max_wavelength ** (torch.arange(0, cond_per_wave, 2) / cond_per_wave),
        )
        self.cond_dim = 4 * dim
        self.mlp = nn.Sequential(
            nn.Linear(dim + self.padding, self.cond_dim),
            nn.SiLU(),
        )

        if init_weights is not None:
            self.reset_parameters(init_weights)

    def reset_parameters(self, init_weights):
        if init_weights == "torch" or init_weights is None:
            pass
        elif init_weights == "xavier_uniform":
            self.mlp.apply(seq_weight_init(nn.init.xavier_uniform_))
        elif init_weights in ["truncnormal", "truncnormal002"]:
            self.mlp.apply(seq_weight_init(nn.init.trunc_normal_))
        else:
            raise NotImplementedError

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        assert self.n_cond == cond.shape[-1], f"{self.n_cond} != {cond.shape[-1]}"
        out = cond.unsqueeze(-1) @ self.omega.unsqueeze(0)
        emb = torch.concat([torch.sin(out), torch.cos(out)], dim=-1)
        emb = rearrange(emb, "... ncond cdim -> ... (ncond cdim)")
        if self.padding > 0:
            padding = torch.zeros(
                *emb.shape[:-1], self.padding, device=emb.device, dtype=emb.dtype
            )
            emb = torch.concat([emb, padding], dim=-1)
        emb = self.mlp(emb)
        return emb


class Film(nn.Module):
    def __init__(self, cond_dim: int, dim_out: int):
        super().__init__()

        self.dim_cond = cond_dim
        self.dim_out = dim_out
        self.modulation = nn.Linear(cond_dim, dim_out * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        mod = self.modulation(cond)
        # broadcast to x
        scale, shift = mod.reshape(
            mod.shape[0], *(1,) * (x.ndim - cond.ndim), *mod.shape[1:]
        ).chunk(2, dim=-1)
        return x * (scale + 1) + shift
