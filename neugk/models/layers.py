"""Generic utility layers."""

from typing import Optional, Sequence, Union

from functools import partial
from einops import rearrange
import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn import functional as F
import torch.distributed as dist

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
        last_act_fn: Optional[nn.Module] = None,
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
        if last_act_fn is not None:
            mlp.append(last_act_fn())
        self.mlp = nn.Sequential(*mlp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class AttentionDecoder(nn.Module):
    def __init__(
        self,
        q_dim: int,
        num_heads: int,
        qkv_bias: bool = False,
        kv_dim: Optional[int] = None,
        out_dim: Optional[int] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        init_weights: Optional[str] = None,
    ):

        super().__init__()
        self.q_dim = q_dim
        self.kv_dim = kv_dim if kv_dim else q_dim
        self.out_dim = out_dim if out_dim else q_dim
        self.num_heads = num_heads
        self.head_dim = q_dim // num_heads
        self.attn_drop = attn_drop
        self.qkv_bias = qkv_bias

        self.q = nn.Linear(q_dim, q_dim, bias=qkv_bias)
        self.kv = nn.Linear(self.kv_dim, q_dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(q_dim, self.out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if init_weights:
            self.reset_parameters(init_weights)

    def reset_parameters(self, init_weights):
        if init_weights == "torch" or init_weights is None:
            return
        elif init_weights == "xavier_uniform":
            init_weights_fn = nn.init.xavier_uniform_
        elif init_weights == "kaiming_uniform":
            init_weights_fn = partial(
                nn.init.kaiming_uniform_, nonlinearity="relu", mode="fan_in", a=0
            )
        elif init_weights in ["truncnormal", "truncnormal002"]:
            init_weights_fn = nn.init.trunc_normal_
        else:
            raise NotImplementedError

        init_weights_fn(self.q.weight)
        init_weights_fn(self.kv.weight)
        if self.qkv_bias:
            nn.init.zeros_(self.q.bias)
            nn.init.zeros_(self.kv.bias)
        init_weights_fn(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, left: torch.Tensor, right: Optional[torch.Tensor] = None):
        b, c = left.shape[0], left.shape[-1]
        grid_size = left.shape[1:-1]
        left = rearrange(left, "b ... c -> b (...) c")  # (b, n, c)
        if right is None:
            right = left
        else:
            right = rearrange(right, "b ... c -> b (...) c")  # (b, m, c)
        # qkv embeddings from inputs
        q = rearrange(self.q(left), "b n (h c) -> b h n c", h=self.num_heads)
        k, v = rearrange(
            self.kv(right), "b m (t h c) -> t b h m c", t=2, h=self.num_heads
        )
        # avoid misaligned strides error
        if dist.is_initialized():
            with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
                x = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=(self.attn_drop if self.training else 0.0)
                )
        else:
            x = F.scaled_dot_product_attention(
                q, k, v, dropout_p=(self.attn_drop if self.training else 0.0)
            )
        # attention readout
        x = rearrange(x, "b k n c -> b n (k c)")
        x = self.proj(x)
        x = self.proj_drop(x)
        # back to original shape
        x = x.view(b, *grid_size, c)
        return x


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
        elif init_weights == "kaiming_uniform":
            self.mlp.apply(
                seq_weight_init(
                    partial(
                        nn.init.kaiming_uniform_,
                        nonlinearity="relu",
                        mode="fan_in",
                        a=0,
                    )
                )
            )
        elif init_weights == "normal_smallvar":
            self.mlp.apply(
                seq_weight_init(partial(nn.init.normal_(mean=0.0, std=1e-3)))
            )
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
            nn.Linear(dim, self.cond_dim),
            nn.SiLU(),
        )

        if init_weights is not None:
            self.reset_parameters(init_weights)

    def reset_parameters(self, init_weights):
        if init_weights == "torch" or init_weights is None:
            pass
        elif init_weights == "xavier_uniform":
            self.mlp.apply(seq_weight_init(nn.init.xavier_uniform_))
        elif init_weights == "kaiming_uniform":
            self.mlp.apply(
                seq_weight_init(
                    partial(
                        nn.init.kaiming_uniform_,
                        nonlinearity="relu",
                        mode="fan_in",
                        a=0,
                    )
                )
            )
        elif init_weights == "normal_smallvar":
            self.mlp.apply(
                seq_weight_init(partial(nn.init.normal_, mean=0.0, std=1e-3))
            )
        elif init_weights in ["truncnormal", "truncnormal002"]:
            self.mlp.apply(seq_weight_init(nn.init.trunc_normal_))
        else:
            raise NotImplementedError

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        if cond.ndim == 1:
            cond = cond.unsqueeze(-1)
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
    def __init__(self, cond_dim: int, dim: int):
        super().__init__()

        self.dim_cond = cond_dim
        self.dim = dim
        self.modulation = nn.Linear(cond_dim, dim * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        mod = self.modulation(cond)
        # broadcast to x
        scale, shift = mod.reshape(
            mod.shape[0], *(1,) * (x.ndim - cond.ndim), *mod.shape[1:]
        ).chunk(2, dim=-1)
        return x * (scale + 1) + shift


class DiT(nn.Module):
    def __init__(
        self,
        dim: int,
        cond_dim: int,
        gate_indices=None,
        init_weights="xavier_uniform",
        init_gate_zero=False,
    ):
        super().__init__()
        self.dim = dim
        self.cond_dim = cond_dim
        # NOTE: 6 for (scale1, shift1, gate1, scale2, shift2, gate2)
        self.modulation = nn.Linear(cond_dim, 6 * dim)
        self.init_gate_zero = init_gate_zero
        self.gate_indices = gate_indices
        if init_weights is not None:
            self.reset_parameters(init_weights)

    def reset_parameters(self, init_weights):
        if init_weights == "torch":
            pass
        elif init_weights == "xavier_uniform":
            nn.init.xavier_uniform_(self.modulation.weight)
        elif init_weights == "kaiming_uniform":
            nn.init.kaiming_uniform_(
                self.modulation.weight, nonlinearity="relu", mode="fan_in", a=0
            )
        elif init_weights in ["truncnormal", "truncnormal002"]:
            self.modulation.apply(nn.init.trunc_normal_)
        else:
            raise NotImplementedError

        if self.init_gate_zero:
            assert self.gate_indices is not None
            for gate_index in self.gate_indices:
                start = self.dim * gate_index
                end = self.dim * (gate_index + 1)
                with torch.no_grad():
                    self.modulation.weight[start:end] = 0
                    self.modulation.bias[start:end] = 0

    def forward(self, cond):
        return self.modulation(cond).chunk(6, dim=1)

    @staticmethod
    def modulate_scale_shift(x, scale, shift):
        scale = scale.reshape(
            scale.shape[0], *(1,) * (x.ndim - scale.ndim), *scale.shape[1:]
        )
        shift = shift.reshape(
            shift.shape[0], *(1,) * (x.ndim - shift.ndim), *shift.shape[1:]
        )
        return x * (1 + scale) + shift

    @staticmethod
    def modulate_gate(x, gate):
        gate = gate.reshape(
            gate.shape[0], *(1,) * (x.ndim - gate.ndim), *gate.shape[1:]
        )
        return gate * x
