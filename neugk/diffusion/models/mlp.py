from typing import Sequence, Optional

from math import prod

import torch
from torch import nn

from neugk.models.layers import MLP


class LatentMLP(nn.Module):
    def __init__(
        self,
        z_dim: int,
        dim: int,
        base_resolution: Sequence[int],
        time_embed: nn.Module,
        cond_embed: Optional[nn.Module] = None,
        act_fn: nn.Module = nn.GELU,
    ):
        super().__init__()

        self.base_resolution = base_resolution
        self.latent_shape = (*base_resolution, z_dim)
        self.z_dim = z_dim
        # diffusion time conditioning
        self.time_embed = time_embed
        cond_dim = self.time_embed.cond_dim
        # parameter conditioning
        if cond_embed:
            self.cond_embed = cond_embed
            cond_dim += self.cond_embed.cond_dim
        # flat latent mlp
        self.mlp = MLP([z_dim + cond_dim, dim, z_dim], act_fn=act_fn)

    def forward(
        self,
        x: torch.Tensor,
        tstep: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        tstep = self.time_embed(tstep)
        if condition is not None:
            c = self.cond_embed(condition)
            c = torch.cat([tstep, c], dim=-1)
        else:
            c = tstep
        c = c.view(c.shape[0], *[1] * (x.ndim - c.ndim), c.shape[-1])
        c = c.expand(*x.shape[:-1], -1)
        return self.mlp(torch.cat([x, c], dim=-1))
