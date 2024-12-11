import torch
from torch import nn
from kappamodules.functional.pos_embed import get_sincos_1d_from_seqlen


class IntegerConditionEmbed(nn.Module):
    def __init__(self, dim, max_size):
        super().__init__()
        cond_dim = dim * 4
        self.max_size = max_size
        self.dim = dim
        self.cond_dim = cond_dim
        self.register_buffer(
            "cond_embed",
            get_sincos_1d_from_seqlen(seqlen=max_size, dim=dim),
        )
        self.mlp = nn.Sequential(
            nn.Linear(dim, cond_dim),
            nn.SiLU(),
        )

    def forward(self, condition):
        # checks + preprocess
        assert condition.numel() == len(condition)
        condition = condition.flatten().long()
        return self.mlp(self.cond_embed[condition])
    
    
class Film(nn.Module):
    def __init__(self, cond_dim, dim_out):
        super().__init__()
        
        self.dim_cond = cond_dim
        self.dim_out = dim_out
        self.modulation = nn.Linear(cond_dim, dim_out * 2)

    def forward(self, x, cond):
        mod = self.modulation(cond)
        # broadcast to x
        scale, shift = mod.reshape(mod.shape[0], *(1,) * (x.ndim - cond.ndim), *mod.shape[1:]).chunk(2, dim=-1)
        return x * (scale + 1) + shift