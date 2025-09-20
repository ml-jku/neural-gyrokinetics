from functools import partial
from typing import Optional, List, Dict

import torch
import torch.nn as nn
from einops import einsum, rearrange
from torch.nn import functional as F

from models.utils import ContinuousEmbed, MLP, AttentionDecoder, DiT


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int = 128,
        num_heads: int = 4,
        dropout_prob: float = 0.1,
        attn_dropout_prob: float = 0.0,
        act_fn: nn.Module = nn.SiLU,
        mlp_ratio: float = 4.0,
        slice_base: int = 64,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()

        self.dim = dim

        self.norm1 = norm_layer(dim)
        self.attn = AttentionDecoder(
            q_dim=dim,
            kv_dim=dim,
            out_dim=dim,
            num_heads=num_heads,
            proj_drop=dropout_prob,
            attn_drop=attn_dropout_prob,
        )

        self.norm2 = norm_layer(dim)
        self.mlp = MLP([dim, int(dim * mlp_ratio), dim], act_fn=act_fn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ViT-like structure
        # attention
        x = x + self.attn(self.norm1(x))
        # mlp
        x = x + self.mlp(self.norm2(x))
        return x


class DiTransformerBlock(TransformerBlock):
    def __init__(self, cond_dim: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dit = DiT(self.dim, cond_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x = super().forward(x)
        scale1, shift1, gate1, scale2, shift2, gate2 = self.dit(cond)
        # modulated attention
        x1 = self.dit.modulate_scale_shift(self.norm1(x), scale1, shift1)
        x2 = self.dit.modulate_gate(self.attn(x1), gate1) + x
        # modulated mlp
        x3 = self.dit.modulate_scale_shift(self.norm2(x2), scale2, shift2)
        x4 = self.dit.modulate_gate(self.mlp(x3), gate2) + x2
        return x4


class Transformer(nn.Module):
    def __init__(
        self,
        condition_keys: List[str],
        latent_channels: int = 128,
        output_channels: int = 17,
        act_fn: nn.Module = nn.SiLU,
        dropout_prob: float = 0.1,
        attn_dropout_prob: float = 0.1,
        space: int = 2,
        dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        slice_base: int = 64,
        mlp_ratio: float = 2.0,
    ):
        super().__init__()

        self.space = space
        self.output_channels = output_channels

        assert (dim % num_heads) == 0
        n_conds = len(condition_keys)
        self.cond_embed = ContinuousEmbed(dim=256, n_cond=n_conds)
        self.condition_keys = condition_keys

        self.conditioning = nn.Sequential(
            MLP(
                [256, 256 // 2, 256 // 4, latent_channels],
                act_fn=act_fn,
                last_act_fn=act_fn,
                dropout_prob=dropout_prob,
            ),
        )

        # encode positions to latent
        self.coord_embed = ContinuousEmbed(dim=dim, n_cond=space)
        self.last_input_embed = MLP([2, self.coord_embed.cond_dim])
        self.encoder = MLP([dim * 2, dim], act_fn=act_fn)

        # processor ("physics attention")
        BlockType = partial(DiTransformerBlock, latent_channels)

        blocks = []
        for _ in range(num_layers):
            block = BlockType(
                dim=dim,
                num_heads=num_heads,
                dropout_prob=dropout_prob,
                attn_dropout_prob=attn_dropout_prob,
                act_fn=act_fn,
                mlp_ratio=mlp_ratio,
                slice_base=slice_base,
            )
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)


        # decode latent to fields (+ positions)
        self.decoder = MLP(
            [dim, output_channels],
            act_fn,
            dropout_prob=dropout_prob,
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # NOTE reproduce original initialization?
        if isinstance(m, nn.Linear):
            # from timm.models.layers import trunc_normal_
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

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
        self,
        df: torch.Tensor,
        position: torch.Tensor,
        **kwargs,
    ):
        # conditioning
        latent_vector = self.conditioning(self.condition(kwargs)["condition"])
        cond = {"cond": latent_vector}

        # embed coordinates of last timestep df
        df = self.last_input_embed(df.transpose(1,2))

        # encoder
        x = self.coord_embed(position)
        x = torch.cat([x, df], dim=-1)
        x = self.encoder(x)  # (B, N, C)

        # transolver
        for block in self.blocks:
            x = block(x, **cond)

        # decoder
        x = self.decoder(x)

        return {"df": rearrange(x, "bs n c -> bs c n")}