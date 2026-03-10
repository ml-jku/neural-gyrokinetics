from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import get_peft_model, LoraConfig, TaskType, EvaConfig

from neugk.pinc.peft_utils import find_linear_layers


class NFPeftWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        coords: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        del kwargs
        if coords is None and input_ids is not None:
            coords = input_ids
        return self.model(coords)


def get_lora_neural_field(model: nn.Module, cfg):
    wrapped_model = NFPeftWrapper(model)

    eva_cfg = EvaConfig(
        rho=2.0,
        tau=0.99,
        use_label_mask=False,
        whiten=False,
        adjust_scaling_factors=True,
    )

    config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=getattr(cfg, "lora_rank", 2),
        lora_alpha=16,
        lora_dropout=0.0,
        init_lora_weights="eva",
        eva_config=eva_cfg,
        target_modules=[n for n, _ in find_linear_layers(wrapped_model)],
    )

    return get_peft_model(wrapped_model, config)


class LoRAAdapt(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rank: int = 4,
        alpha: float = 1.0,
        skip: bool = True,
    ):
        super().__init__()
        self.rank = rank
        self.skip = skip
        self.alpha = alpha / rank

        self.lora_u = nn.Parameter(torch.zeros(rank, in_dim))
        self.lora_v = nn.Parameter(torch.zeros(out_dim, rank))
        nn.init.normal_(self.lora_u, mean=0.0, std=1 / rank)
        nn.init.zeros_(self.lora_v)

    def forward(self, x):
        skip = x if self.skip else 0
        return skip + self.alpha * (x @ self.lora_u.T) @ self.lora_v.T


class LoRALinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 1.0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha / rank

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

        # lora
        self.lora_u = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_v = nn.Parameter(torch.zeros(out_features, rank))

        nn.init.kaiming_uniform_(self.weight, a=5 ** (1 / 2))
        nn.init.normal_(self.lora_u, mean=0.0, std=1.0 / rank)
        nn.init.zeros_(self.lora_v)

    def forward(self, x):
        # standard linear
        result = F.linear(x, self.weight, self.bias)
        # low-rank update
        delta = F.linear(x, self.lora_u, None)
        delta = F.linear(delta, self.lora_v, None)
        return result + self.alpha * delta


class LinearCoordEmbed(nn.Module):
    def __init__(
        self,
        ndim: int,
        dim: int,
        act_fn: nn.Module,
        fused: bool = True,
        lora_rank: Optional[int] = None,
    ):
        super().__init__()

        self.cond_dim = dim
        self.ndim = ndim
        self.lora_rank = lora_rank

        if lora_rank is None:
            if ndim == 6 and not fused:
                self.embed = nn.Linear(ndim - 1, dim, bias=False)
                self.t_embed = nn.Linear(1, dim, bias=False)
            else:
                self.embed = nn.Linear(ndim, dim, bias=False)
        else:
            if ndim == 6 and not fused:
                self.embed = LoRALinear(ndim - 1, dim, rank=lora_rank, bias=False)
                self.t_embed = LoRALinear(1, dim, rank=lora_rank, bias=False)
            else:
                self.embed = LoRALinear(ndim, dim, rank=lora_rank, bias=False)
        self.act_fn = act_fn()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "t_embed"):
            x = self.embed(x[..., :-1]) + self.t_embed(x[..., -1:])
        else:
            x = self.embed(x)
        return self.act_fn(x)
