from typing import Sequence, Optional, Union, Callable, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from neugk.models.layers import (
    ContinuousConditionEmbed,
    IntegerConditionEmbed,
    IntegerSincosConditionEmbed,
    Film,
)
from neugk.pinc.neural_fields.models.utils import LinearCoordEmbed


class MLPBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: nn.Module,
        act_fn: Optional[nn.Module] = None,
        bias: bool = True,
        skip: bool = False,
    ):
        super().__init__()
        self.skip = skip and (in_dim == out_dim)
        layers = [nn.Linear(in_dim, out_dim, bias=bias), dropout]
        if act_fn is not None:
            layers.append(act_fn())
        self.block = nn.Sequential(*layers)

        self.reset_parameters()

    def forward(self, x):
        res = x
        x = self.block(x)
        if self.skip:
            x = res + x
        return x

    def reset_parameters(self):
        for layer in self.block:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)


class MLP(nn.Module):
    def __init__(
        self,
        latents: Sequence[int],
        act_fn: nn.Module = nn.GELU,
        last_act_fn: Optional[nn.Module] = None,
        bias: Union[bool, Sequence[bool]] = True,
        dropout_prob: float = 0.0,
        use_checkpoint: bool = False,
        skips: bool = False,
        conditioner: Optional[Callable] = None,
    ):
        super().__init__()
        if isinstance(bias, bool):
            bias = [bias] * (len(latents) - 1)
        dropout = nn.Dropout(dropout_prob) if dropout_prob > 0 else nn.Identity()
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList()

        for i, (lat_i, lat_i2) in enumerate(zip(latents, latents[1:])):
            self.blocks.append(
                MLPBlock(
                    lat_i,
                    lat_i2,
                    act_fn=act_fn if i != len(latents) - 2 else None,
                    dropout=dropout,
                    bias=bias[i],
                    skip=skips,
                )
            )

        self.last_act_fn = last_act_fn() if last_act_fn else None
        self.conditioner = conditioner

        self.reset_parameters()

    def forward(
        self, x: torch.Tensor, cond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
            if cond is not None and self.conditioner is not None:
                # NOTE could call it once (faster)
                x = self.conditioner(x, cond)
        if self.last_act_fn:
            x = self.last_act_fn(x)
        return x

    def reset_parameters(self):
        for block in self.blocks:
            block.reset_parameters()


class MLPNF(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_layers: int = 2,
        dim: int = 512,
        act_fn: nn.Module = nn.SiLU,
        use_checkpoint: bool = False,
        embed_type: str = "linear",
        skips: bool = False,
        use_z_functa: bool = False,
        grid_size: Tuple[int],
    ):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dim = dim
        self.act_fn = act_fn
        self.embed_type = embed_type

        if embed_type == "sincos_continuous":
            self.coord_embed = ContinuousConditionEmbed(
                dim // 4, in_dim, max_wavelength=500
            )
        elif embed_type == "sincos_discrete":
            self.coord_embed = IntegerSincosConditionEmbed(
                dim,
                in_dim,
                max_size=grid_size,
                use_mlp=False,
            )
        elif embed_type == "discrete":
            self.coord_embed = IntegerConditionEmbed(
                dim,
                in_dim,
                max_size=grid_size,
                use_mlp=False,
            )
        elif embed_type == "linear":
            self.coord_embed = LinearCoordEmbed(in_dim, dim, act_fn=act_fn, fused=True)
        else:
            raise NotImplementedError(f"embed: {embed_type}")

        self.conditioner = self.z_functa = None
        if use_z_functa:
            self.z_functa = nn.Parameter(torch.zeros((1, self.dim)))
            self.conditioner = Film(dim, dim, shift=False, modulation=nn.Identity())

        embed_dim = self.coord_embed.cond_dim
        self.net = MLP(
            [embed_dim, *[dim] * (n_layers - 1)],
            act_fn=act_fn,
            last_act_fn=act_fn,
            use_checkpoint=use_checkpoint,
            skips=skips,
            conditioner=self.conditioner,
        )
        self.readout = nn.Linear(dim, out_dim)

        self.reset_parameters()

    def reset_parameters(self):
        if isinstance(self.coord_embed, nn.Sequential):
            for layer in self.coord_embed:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        self.net.reset_parameters()
        nn.init.xavier_uniform_(self.readout.weight)
        if self.readout.bias is not None:
            nn.init.zeros_(self.readout.bias)

    def forward(
        self, coords: torch.Tensor, cond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # NOTE cond could also be used, would require a shared hypernetwork
        coords = self.coord_embed(coords)
        cond = cond if cond is not None else self.z_functa
        x = self.net(coords, cond)
        x = self.readout(x)
        return x
