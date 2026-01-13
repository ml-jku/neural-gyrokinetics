from typing import Dict

import torch
from torch import nn

from neugk.models.gk_unet import Swin5DUnet


class Swin5DDiffUnet(Swin5DUnet):
    def __init__(self, *args, time_embed: nn.Module, **kwargs):
        if "cond_embed" in kwargs:
            kwargs["cond_embed"].cond_dim += time_embed.cond_dim
        super().__init__(*args, **kwargs)

        # TODO(diff) hacky way to include time_embed cond dim in modulation layers
        self.cond_embed.cond_dim -= time_embed.cond_dim

        self.time_embed = time_embed

        self.latent_shape = (self.original_problem_dim, *self.full_resolution)

    def condition(self, kwconds: Dict[str, torch.Tensor]) -> Dict:
        tstep = kwconds["tstep"]
        condition = kwconds.get("condition", None)

        tstep = self.time_embed(tstep)

        if condition is not None:
            condition = self.cond_embed(condition)
            condition = torch.cat([tstep, condition], dim=-1)
        else:
            condition = tstep

        return {"condition": condition}

    def forward(self, x: torch.Tensor, **kwargs):
        x = self._shape_correction(x)
        df = super().forward(x, **kwargs)["df"]
        return self._inverse_shape_correction(df)

    def _shape_correction(self, x: torch.Tensor, gamma: float = 0.5):
        return torch.sign(x) * (torch.abs(x) ** gamma)

    def _inverse_shape_correction(self, x: torch.Tensor, gamma: float = 0.5):
        return torch.sign(x) * (torch.abs(x) ** (1 / gamma))
