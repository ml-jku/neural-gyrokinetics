from typing import Sequence, Union, Optional, Tuple

from torch.nn import functional as F
import torch
from torch import nn
from einops import rearrange

from models.swin_unet import SwinUnet
from models.utils import Film


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        init_weights: Optional[str] = None,
    ) -> None:

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_drop = attn_drop
        self.qkv_bias = qkv_bias

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if init_weights:
            self.reset_parameters(init_weights)

    def reset_parameters(self, init_weights):
        if init_weights == "torch" or init_weights is None:
            return
        elif init_weights == "xavier_uniform":
            init_weights_fn = nn.init.xavier_uniform_
        elif init_weights in ["truncnormal", "truncnormal002"]:
            init_weights_fn = nn.init.trunc_normal_
        else:
            raise NotImplementedError

        init_weights_fn(self.qkv.weight)
        if self.qkv_bias:
            nn.init.zeros_(self.qkv.bias)
        init_weights_fn(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, left: torch.Tensor, right: torch.Tensor):
        b, c = left.shape[0], left.shape[-1]
        grid_size = left.shape[1:-1]
        left = rearrange(left, "b ... c -> b (...) c")  # (b, n, c)
        right = rearrange(right, "b ... c -> b (...) c")  # (b, m, c)
        q = rearrange(self.q(left), "b n (h c) -> b h n c", h=self.num_heads)
        k, v = rearrange(
            self.kv(right), "b n (t h c) -> t b h n c", t=2, h=self.num_heads
        )

        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop)

        # attention readout
        x = rearrange(x, "b k n c -> b n (k c)")
        x = self.proj(x)
        x = self.proj_drop(x)
        # back to original shape
        x = x.view(b, *grid_size, c)
        return x


class LatentFusion(nn.Module):
    def __init__(
        self,
        latent_resolution: Sequence[int],
        latent_dim: int,
        dropout_prob: float = 0.1,
    ):
        super().__init__()
        self.latent_resolution = latent_resolution
        self.latent_dim = latent_dim
        self.cond_dim = 4 * latent_dim
        # TODO do cross attention instead
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, self.cond_dim),
            nn.Dropout(dropout_prob),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO pool out space -> bad
        x = rearrange(x, "b ... c -> b (...) c").mean(1, keepdim=True)
        return self.mlp(x)


class SwinXnet(nn.Module):
    def __init__(
        self,
        dim: int,
        df_base_resolution: Sequence[int],
        phi_base_resolution: Sequence[int],
        df_patch_size: Union[Sequence[int], int] = 4,
        phi_patch_size: Union[Sequence[int], int] = 4,
        df_window_size: Union[Sequence[int], int] = 5,
        phi_window_size: Union[Sequence[int], int] = 5,
        depth: Union[Sequence[int], int] = 2,
        num_heads: Union[Sequence[int], int] = 4,
        in_channels: int = 2,
        out_channels: int = 2,
        num_layers: int = 4,
        abs_pe: bool = False,
        c_multiplier: int = 2,
        drop_path: float = 0.1,
        use_checkpoint: bool = False,
        merging_hidden_ratio: float = 8.0,
        unmerging_hidden_ratio: float = 8.0,
        conditioning: Optional[nn.Module] = None,
        modulation: str = "dit",
        act_fn: nn.Module = nn.GELU,
    ):
        super().__init__()

        self.df_unet = SwinUnet(
            5,
            dim=dim,
            base_resolution=df_base_resolution,
            patch_size=df_patch_size,
            window_size=df_window_size,
            depth=depth,
            num_heads=num_heads,
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            use_checkpoint=use_checkpoint,
            drop_path=drop_path,
            abs_pe=abs_pe,
            hidden_mlp_ratio=2.0,
            c_multiplier=c_multiplier,
            modulation=modulation,
            merging_hidden_ratio=merging_hidden_ratio,
            unmerging_hidden_ratio=unmerging_hidden_ratio,
            conditioning=conditioning,
            act_fn=act_fn,
        )

        self.phi_unet = SwinUnet(
            3,
            dim=dim,
            base_resolution=phi_base_resolution,
            patch_size=phi_patch_size,
            window_size=phi_window_size,
            depth=depth,
            num_heads=num_heads,
            in_channels=1,
            out_channels=1,
            num_layers=num_layers,
            use_checkpoint=use_checkpoint,
            drop_path=drop_path,
            abs_pe=abs_pe,
            conv_patch=True,
            hidden_mlp_ratio=2.0,
            c_multiplier=c_multiplier,
            modulation=modulation,
            merging_hidden_ratio=merging_hidden_ratio,
            unmerging_hidden_ratio=unmerging_hidden_ratio,
            conditioning=conditioning,
            act_fn=act_fn,
        )

        # self.df_cond_embed = LatentFusion(
        #     self.phi_unet.grid_sizes[-1], self.phi_unet.down_dims[-1]
        # )
        # self.df_fusion = Film(self.df_cond_embed.cond_dim, self.df_unet.down_dims[-1])
        # self.phi_cond_embed = LatentFusion(
        #     self.df_unet.grid_sizes[-1], self.df_unet.down_dims[-1]
        # )
        # self.phi_fusion = Film(
        #     self.phi_cond_embed.cond_dim, self.phi_unet.down_dims[-1]
        # )
        self.df_cross_att = CrossAttention(self.df_unet.down_dims[-1], 8)
        self.phi_cross_att = CrossAttention(self.phi_unet.down_dims[-1], 8)

    def forward(
        self, df: torch.Tensor, phi: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # compress to patch space
        df, df_pad_axes = self.df_unet.patch_encode(df)
        phi, phi_pad_axes = self.phi_unet.patch_encode(phi)

        # parameter conditioning
        df_cond = self.df_unet.condition(kwargs)
        phi_cond = self.phi_unet.condition(kwargs)

        # down paths
        df_feature_maps = []
        phi_feature_maps = []
        for df_blk, phi_blk in zip(self.df_unet.down_blocks, self.phi_unet.down_blocks):
            df, df_pre = df_blk(df, **df_cond)
            phi, phi_pre = phi_blk(phi, **phi_cond)
            df_feature_maps.append(df_pre)
            phi_feature_maps.append(phi_pre)

        # middle blocks + latent fusion
        # TODO do at all Unet levels
        df = self.df_unet.middle_pe(df)
        phi = self.phi_unet.middle_pe(phi)

        # df = self.df_fusion(df, self.df_cond_embed(phi))
        # phi = self.phi_fusion(phi, self.phi_cond_embed(df))
        df, phi = self.df_cross_att(df, phi), self.phi_cross_att(phi, df)

        df = self.df_unet.middle(df, **df_cond)
        phi = self.phi_unet.middle(phi, **phi_cond)
        df = self.df_unet.middle_upscale(df)
        phi = self.phi_unet.middle_upscale(phi)

        # up path
        df_feature_maps = df_feature_maps[::-1]
        phi_feature_maps = phi_feature_maps[::-1]
        for i, (df_blk, phi_blk) in enumerate(
            zip(self.df_unet.up_blocks, self.phi_unet.up_blocks)
        ):
            df = df_blk(df, s=df_feature_maps[i], **df_cond)
            phi = phi_blk(phi, s=phi_feature_maps[i], **phi_cond)

        # expand to original
        df = self.df_unet.patch_decode(df, df_cond["condition"], df_pad_axes)
        phi = self.phi_unet.patch_decode(phi, phi_cond["condition"], phi_pad_axes)

        return df, phi
