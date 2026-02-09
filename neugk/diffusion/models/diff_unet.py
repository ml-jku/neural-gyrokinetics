from typing import Dict, Union, Sequence

import torch
from torch import nn
from einops import rearrange

from neugk.gyroswin.models.baselines.fno import DfVSpace3DTFNO
from neugk.models.gk_unet import Swin5DUnet, SwinNDUnet
from neugk.models.nd_vit.patching import PatchEmbed, PatchExpand, pad_to_blocks
from neugk.models.layers import Film


class Swin5DDiffUnet(Swin5DUnet):
    def __init__(self, *args, time_embed: nn.Module, **kwargs):
        # include time_embed cond dim in modulation layers
        # if "cond_embed" in kwargs:
        #     kwargs["cond_embed"].cond_dim += time_embed.cond_dim
        super().__init__(*args, **kwargs)
        # if "cond_embed" in kwargs:
        #     self.cond_embed.cond_dim -= time_embed.cond_dim
        self.time_embed = time_embed
        self.latent_shape = (self.original_problem_dim, *self.full_resolution)

    def condition(self, kwconds: Dict[str, torch.Tensor]) -> Dict:
        tstep = kwconds["tstep"]
        condition = kwconds.get("condition", None)
        tstep = self.time_embed(tstep)
        condition = tstep

        # if condition is not None:
        #     condition = self.cond_embed(condition)
        #     condition = torch.cat([tstep, condition], dim=-1)
        # else:
        #     condition = tstep

        return {"condition": condition}

    def forward(self, x: torch.Tensor, **kwargs):
        df = super().forward(x, **kwargs)["df"]
        return df


class Basic5DDiff(nn.Module):
    def __init__(
        self,
        dim: int,
        base_resolution: Sequence[int],
        max_tsteps: int,
        in_channels: int = 2,
        out_channels: int = 2,
        patch_size: Union[Sequence[int], int] = 4,
        act_fn: nn.Module = nn.GELU,
    ):
        super().__init__()

        space = 5

        self.time_embed = nn.Embedding(max_tsteps, 2 * dim)
        padded_base_resolution, pad_axes = pad_to_blocks(base_resolution, patch_size)
        self.pad_axes = [int(p) for p in pad_axes]

        self.patch_embed = PatchEmbed(
            space=space,
            base_resolution=padded_base_resolution,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=dim,
            flatten=False,
            mlp_ratio=2.0,
            act_fn=act_fn,
            mlp_depth=2,
        )
        self.unpatch = PatchExpand(
            space,
            dim,
            grid_size=self.patch_embed.grid_size,
            expand_by=patch_size,
            out_channels=out_channels,
            flatten=False,
            norm_layer=None,
            mlp_ratio=2.0,
            act_fn=act_fn,
            cond_dim=self.time_embed.embedding_dim,
            mlp_depth=2,
        )

        self.latent_shape = (in_channels, *base_resolution)

    def forward(self, x: torch.Tensor, **kwargs):
        x = rearrange(x, "b c ... -> b ... c")
        x = self.patch_embed(x)
        cond = self.time_embed(kwargs["tstep"])
        x = self.unpatch(x, cond=cond)
        return rearrange(x, "b ... c -> b c ...")


class FNO5DDiff(DfVSpace3DTFNO):
    def __init__(self, *args, max_tsteps: int, **kwargs):
        super().__init__(*args, **kwargs)

        self.time_embed = nn.Embedding(max_tsteps, 256)
        self.film = nn.ModuleList(
            [Film(256, self.hidden_channels) for _ in range(self.n_layers)]
        )
        vspace = self.base_resolution[0] * self.base_resolution[1]
        self.latent_shape = (self.in_channels // vspace, *self.base_resolution)

    def forward(self, df: torch.Tensor, **kwargs):
        cond = self.time_embed(kwargs["tstep"])
        # put vspace in channels
        vp, vm = df.shape[2], df.shape[3]
        x = rearrange(df, "b c vp vm s x y -> b (c vp vm) s x y")
        # fno stuff
        if self.positional_embedding is not None:
            x = self.positional_embedding(x)
        x = self.lifting(x)
        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)
        for i in range(self.n_layers):
            x = self.fno_blocks(x, i)
            # modulation
            x = rearrange(x, "b (c vp vm) s x y -> b s x y (c vp vm)")
            x = self.film[i](x, cond)
            x = rearrange(x, "b s x y (c vp vm) -> b (c vp vm) s x y")
        if self.domain_padding is not None:
            x = self.domain_padding.unpad(x)
        x = self.projection(x)
        # back to 5D
        return rearrange(x, "b (c vp vm) s x y -> b c vp vm s x y", vp=vp, vm=vm)


# class Swin3DDiffUnet(SwinNDUnet):
#     def __init__(self, *args, time_embed: nn.Module, **kwargs):
#         kwargs["space"] = 3
#         super().__init__(*args, **kwargs)
#         self.time_embed = time_embed
#         self.latent_shape = (self.problem_dim, *self.base_resolution)

#     def condition(self, kwconds: Dict[str, torch.Tensor]) -> Dict:
#         tstep = kwconds["tstep"]
#         tstep = self.time_embed(tstep)
#         return {"condition": tstep}

#     def forward(self, x: torch.Tensor, **kwargs):
#         x = super().forward(x, **kwargs)
#         return x


class Swin3DDiffUnet(DfVSpace3DTFNO):
    def __init__(self, *args, max_tsteps: int, **kwargs):
        kwargs["base_resolution"] = (1, 1, *kwargs["base_resolution"])
        super().__init__(*args, **kwargs)
        self.base_resolution = [
            self.base_resolution[2],
            self.base_resolution[3],
            self.base_resolution[4],
        ]

        self.time_embed = nn.Embedding(max_tsteps, 256)
        self.film = nn.ModuleList(
            [Film(256, self.hidden_channels) for _ in range(self.n_layers)]
        )
        self.latent_shape = (self.in_channels, *self.base_resolution)

    def forward(self, x: torch.Tensor, **kwargs):
        cond = self.time_embed(kwargs["tstep"])
        # fno stuff
        if self.positional_embedding is not None:
            x = self.positional_embedding(x)
        x = self.lifting(x)
        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)
        for i in range(self.n_layers):
            x = self.fno_blocks(x, i)
            # modulation
            x = rearrange(x, "b c x s y -> b x s y c")
            x = self.film[i](x, cond)
            x = rearrange(x, "b x s y c -> b c x s y")
        if self.domain_padding is not None:
            x = self.domain_padding.unpad(x)
        return self.projection(x)
