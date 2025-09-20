from typing import Tuple, Dict
from torch import nn
import torch
from einops import rearrange

from neugk.models.layers import MLP, ContinuousEmbed


class PointNet(nn.Module):
    def __init__(self, dim, n_dims, n_channels, condition_keys):
        super().__init__()
        self.condition_keys = condition_keys
        self.pos_embed = ContinuousEmbed(dim=dim,
                                         n_cond=n_dims,
                                         init_weights="xavier_uniform")
        
        self.cond_embed = ContinuousEmbed(dim=dim*2,
                                          n_cond=len(self.condition_keys),
                                          init_weights="xavier_uniform")
        
        # embed re/img part of last timestep as additional context to predict change for next timestep
        self.last_input_embed = MLP([n_channels, self.pos_embed.cond_dim])
        self.in_block = MLP([dim * 2, dim * 2, dim * 2])
        self.max_block = MLP([dim * 2, dim * 8, dim * 32])
        self.out_block = MLP([dim * (2 + 32), dim * 16, dim * 4])
        self.encoder = MLP([dim * 2, dim * 4, dim * 2])
        self.decoder = MLP([dim * 2, dim * 4, n_channels])
        self.ff = nn.Linear(dim * 4, dim * 2)

    def condition(self, kwconds: Dict[str, torch.Tensor]) -> Dict:
        # drop input fields
        kwconds = {k: v for k, v in kwconds.items() if k in self.condition_keys}
        if len(kwconds) == 0:
            return {}

        assert sorted(self.condition_keys) == sorted(list(kwconds.keys())), (
            "Mismatch in conditioning keys "
            f"{self.condition_keys} != {sorted(list(kwconds.keys()))}"
        )
        cond = torch.cat(
            [kwconds[k].unsqueeze(-1) for k in self.condition_keys], dim=-1
        )
        if self.cond_embed is not None:
            # embed conditioning is e.g. sincos
            return {"condition": self.cond_embed(cond)}
        else:
            return {}

    def forward(
        self, df: torch.Tensor, position: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        x = self.pos_embed(position)
        df = self.last_input_embed(df.transpose(1,2))
        x = torch.cat([x, df], dim=-1)
        x = x + self.condition(kwargs)["condition"]

        x = self.encoder(x) + x
        x = self.in_block(x)

        global_coef = self.max_block(x)
        global_coef = torch.max(global_coef, dim=1)[0]
        x = torch.cat([x, global_coef.repeat(x.shape[1], 1).unsqueeze(0)], dim=-1)
        x = self.out_block(x)
        x = self.ff(x)
        x = self.decoder(x)

        return {"df": rearrange(x, "bs n c -> bs c n")}