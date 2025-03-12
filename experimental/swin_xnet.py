from typing import Sequence, Union, Optional, Tuple, Type

from torch.nn import functional as F
import torch
from torch import nn
from einops import rearrange

from models.swin_unet import SwinUnet
from models.nd_vit.drop import DropPath
from models.utils import Film, MLP, seq_weight_init


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


class LatentMixingTransformer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        use_checkpoint: bool = False,
        act_fn: nn.Module = nn.GELU,
        init_weights: Optional[str] = None,
    ):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        self.init_weights = init_weights

        self.norm1 = norm_layer(dim) if norm_layer is not None else nn.Identity()
        self.attn = CrossAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            init_weights=init_weights,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim) if norm_layer is not None else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = MLP([dim, mlp_hidden_dim, dim], act_fn=act_fn, dropout_prob=drop)

        if init_weights:
            self.reset_parameters(init_weights)

    def reset_parameters(self, init_weights):
        if init_weights == "torch" or init_weights is None:
            pass
        elif init_weights == "xavier_uniform":
            self.mlp.apply(seq_weight_init(nn.init.xavier_uniform_))
        elif init_weights in ["truncnormal", "truncnormal002"]:
            self.mlp.apply(seq_weight_init(nn.init.trunc_normal_))
        else:
            raise NotImplementedError

        self.attn.reset_parameters(init_weights)

    def forward_part1(self, left: torch.Tensor, right: torch.Tensor):
        x = self.norm1(self.attn(left, right))
        return x

    def forward_part2(self, x):
        x = self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward(self, left: torch.Tensor, right: Optional[torch.Tensor] = None):
        right = right if right is not None else left
        shortcut = left
        x = self.forward_part1(left, right)
        x = shortcut + self.drop_path(x)
        x = x + self.forward_part2(x)
        return x


class LatentMixingMLP(nn.Module):
    def __init__(
        self, dim: int, right_dim: int, right_grid: int, dropout_prob: float = 0.1
    ):
        super().__init__()

        self.latent_resolution = right_grid
        self.right_dim = right_dim
        self.cond_dim = 4 * right_dim
        self.embed_mlp = nn.Sequential(
            nn.Linear(right_dim, self.cond_dim),
            nn.Dropout(dropout_prob),
            nn.SiLU(),
        )

        self.modulate = Film(self.cond_dim, dim)

    def forward(self, left: torch.Tensor, right: torch.Tensor):
        # pool out spatials
        right = rearrange(right, "b ... c -> b (...) c").mean(1, keepdim=True)
        right = self.embed_mlp(right)
        return self.modulate(left, right)


class FluxDecoder(LatentMixingTransformer):
    def forward(self, left: torch.Tensor, right: Optional[torch.Tensor] = None):
        x = super().forward(left, right)
        # average spatials out
        return x.mean(axis=list(range(1, x.ndim - 1)))


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
        patch_skip: bool = False,
        latent_cross_attn: bool = True,
        flux_head: bool = True,
    ):
        super().__init__()

        self.patch_skip = patch_skip

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
            hidden_mlp_ratio=8.0,
            c_multiplier=c_multiplier,
            modulation=modulation,
            merging_hidden_ratio=merging_hidden_ratio,
            unmerging_hidden_ratio=unmerging_hidden_ratio,
            conditioning=conditioning,
            act_fn=act_fn,
            patch_skip=patch_skip,
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
            hidden_mlp_ratio=8.0,
            c_multiplier=2,
            modulation=modulation,
            merging_hidden_ratio=merging_hidden_ratio,
            unmerging_hidden_ratio=unmerging_hidden_ratio,
            conditioning=conditioning,
            act_fn=act_fn,
            patch_skip=patch_skip,
        )
        self.df_up_blocks = self.df_unet.up_blocks
        self.df_down_blocks = self.df_unet.down_blocks
        self.phi_up_blocks = self.phi_unet.up_blocks
        self.phi_down_blocks = self.phi_unet.down_blocks

        self.flux_blocks = None
        self.flux_readout = None
        if flux_head:
            flux_blocks = []
            flux_latent_size = 0
            for phi_dim in self.phi_unet.down_dims[::-1]:
                flux_blocks.append(FluxDecoder(phi_dim, 8, attn_drop=0.1))
                flux_latent_size += phi_dim

            self.flux_blocks = nn.ModuleList(flux_blocks)
            self.flux_readout = MLP([flux_latent_size, flux_latent_size // 2, 1])

        if latent_cross_attn:
            # down/middle direction
            df_mixing = []
            phi_mixing = []
            for df_dim, phi_dim in zip(self.df_unet.down_dims, self.phi_unet.down_dims):
                df_mixing.append(LatentMixingTransformer(df_dim, 8, attn_drop=0.1))
                phi_mixing.append(LatentMixingTransformer(phi_dim, 8, attn_drop=0.1))
            self.df_mixing = nn.ModuleList(df_mixing)
            self.phi_mixing = nn.ModuleList(phi_mixing)
            # up direction
            df_mixing_up = []
            phi_mixing_up = []
            for df_blk, phi_blk in zip(self.df_up_blocks, self.phi_up_blocks):
                df_mixing_up.append(
                    LatentMixingTransformer(df_blk.dim, 8, attn_drop=0.1)
                )
                phi_mixing_up.append(
                    LatentMixingTransformer(phi_blk.dim, 8, attn_drop=0.1)
                )
            self.df_mixing_up = nn.ModuleList(df_mixing_up)
            self.phi_mixing_up = nn.ModuleList(phi_mixing_up)

            df_patch_dim = self.df_unet.unpatch.dim * (2 if patch_skip else 1)
            phi_patch_dim = self.phi_unet.unpatch.dim * (2 if patch_skip else 1)
            self.df_mixing_unpatch = LatentMixingTransformer(
                df_patch_dim, 8, attn_drop=0.1
            )
            self.phi_mixing_unpatch = LatentMixingTransformer(
                phi_patch_dim, 8, attn_drop=0.1
            )
        else:
            df_mixing = []
            phi_mixing = []
            for (df_dim, phi_dim), (df_grid, phi_grid) in zip(
                zip(self.df_unet.down_dims, self.phi_unet.down_dims),
                zip(self.df_unet.grid_sizes, self.phi_unet.grid_sizes),
            ):
                df_mixing.append(LatentMixingMLP(df_dim, phi_dim, phi_grid))
                phi_mixing.append(LatentMixingMLP(phi_dim, df_dim, df_grid))
            self.df_mixing = nn.ModuleList(df_mixing)
            self.phi_mixing = nn.ModuleList(phi_mixing)

    def forward(
        self, df: torch.Tensor, phi: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # compress to patch space
        df, df_pad_axes = self.df_unet.patch_encode(df)
        phi, phi_pad_axes = self.phi_unet.patch_encode(phi)
        if self.patch_skip:
            df0 = df.clone()
            phi0 = phi.clone()

        # parameter conditioning
        df_cond = self.df_unet.condition(kwargs)
        phi_cond = self.phi_unet.condition(kwargs)

        # down paths
        df_feature_maps = []
        phi_feature_maps = []
        for df_blk, phi_blk, df_mix, phi_mix in zip(
            self.df_down_blocks,
            self.phi_down_blocks,
            self.df_mixing[:-1],
            self.phi_mixing[:-1],
        ):
            # mix latents
            df, phi = df_mix(df, phi), phi_mix(phi, df)
            # down blocks
            df, df_pre = df_blk(df, **df_cond)
            phi, phi_pre = phi_blk(phi, **phi_cond)
            df_feature_maps.append(df_pre)
            phi_feature_maps.append(phi_pre)

        # middle blocks + latent fusion
        flux_lats = []
        df = self.df_unet.middle_pe(df)
        phi = self.phi_unet.middle_pe(phi)

        df, phi = self.df_mixing[-1](df, phi), self.phi_mixing[-1](phi, df)

        df = self.df_unet.middle(df, **df_cond)
        phi = self.phi_unet.middle(phi, **phi_cond)

        if self.flux_blocks is not None:
            flux_lats.append(self.flux_blocks[0](phi, df))

        df = self.df_unet.middle_upscale(df)
        phi = self.phi_unet.middle_upscale(phi)

        # up path
        df_feature_maps = df_feature_maps[::-1]
        phi_feature_maps = phi_feature_maps[::-1]
        for i, (df_blk, phi_blk, df_mix, phi_mix) in enumerate(
            zip(
                self.df_up_blocks,
                self.phi_up_blocks,
                self.df_mixing_up,
                self.phi_mixing_up,
            )
        ):
            # mix latents
            df, phi = df_mix(df, phi), phi_mix(phi, df)
            # up blocks
            df = df_blk(df, s=df_feature_maps[i], **df_cond)
            phi = phi_blk(phi, s=phi_feature_maps[i], **phi_cond)
            if self.flux_blocks is not None:
                flux_lats.append(self.flux_blocks[i + 1](phi, df))

        # expand to original
        if self.patch_skip:
            df = torch.cat([df, df0], -1)
            phi = torch.cat([phi, phi0], -1)

        # expand to original
        df, phi = self.df_mixing_unpatch(df, phi), self.phi_mixing_unpatch(phi, df)
        df = self.df_unet.patch_decode(df, df_cond["condition"], df_pad_axes)
        phi = self.phi_unet.patch_decode(phi, phi_cond["condition"], phi_pad_axes)

        flux = None
        if self.flux_readout is not None:
            flux = self.flux_readout(torch.cat(flux_lats, dim=-1))

        return df, phi, flux
