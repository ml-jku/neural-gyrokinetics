from typing import Sequence, Union, Optional, Tuple, Type

from torch.nn import functional as F
import torch
from torch import nn
from einops import rearrange

from models.swin_unet import SwinUnet
from models.nd_vit.drop import DropPath
from models.nd_vit.positional import PositionalEmbedding
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


class MixingBlock(nn.Module):
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
        act_fn: nn.Module = nn.GELU,
        init_weights: Optional[str] = None,
    ):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
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

    def forward(self, left: torch.Tensor, right: Optional[torch.Tensor] = None):
        right = right if right is not None else left
        x = left + self.drop_path(self.norm1(self.attn(left, right)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class LatentMixingTransformer(nn.Module):
    def __init__(
        self,
        space: int,
        dim: int,
        depth: int,
        num_heads: int,
        grid_size: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop_path: Union[Sequence[float], float] = 0.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_fn: nn.Module = nn.GELU,
        init_weights: Optional[str] = None,
    ):

        super().__init__()

        if isinstance(drop_path, float):
            drop_path = [drop_path] * depth

        self.depth = depth
        self.dim = dim
        self.grid_size = grid_size
        self.drop_path = drop_path

        assert dim % num_heads == 0

        self.blocks = nn.ModuleList(
            [
                MixingBlock(
                    space,
                    dim=dim,
                    grid_size=grid_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i],
                    norm_layer=norm_layer,
                    act_fn=act_fn,
                )
                for i in range(depth)
            ]
        )

        if init_weights is not None:
            self.reset_parameters(init_weights)

    def reset_parameters(self, init_weights):
        for blk in self.blocks:
            blk.reset_parameters(init_weights)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(left, right)
        return x


class FluxDecoder(nn.Module):
    def __init__(
        self,
        dims: Sequence[int],
        num_heads: int,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_fn: nn.Module = nn.GELU,
        init_weights: Optional[str] = None,
    ):
        super().__init__()

        flux_blocks = []
        flux_latent_size = 0
        for dim in dims:
            flux_blocks.append(
                MixingBlock(
                    dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path,
                    norm_layer=norm_layer,
                    act_fn=act_fn,
                    init_weights=init_weights,
                )
            )
            flux_latent_size += dim

        self.blocks = nn.ModuleList(flux_blocks)
        self.flux_mlp = MLP(
            [flux_latent_size, flux_latent_size // 2, 1],
            # last_act_fn=nn.Softplus,
            dropout_prob=drop,
        )

    def mix(self, i: int, left: torch.Tensor, right: Optional[torch.Tensor] = None):
        x = self.blocks[i].forward(left, right)
        # pool spatials
        return x.amax(axis=list(range(1, x.ndim - 1)))

    def forward(self, flux_latents: Sequence[torch.Tensor]):
        return self.flux_mlp(torch.cat(flux_latents, dim=-1))


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
        flux_head: bool = True,
        decouple_mu: bool = False,
        separate_zf: bool = False,
        zf_norm: bool = False,
        flux_drop: float = 0.1,
    ):
        super().__init__()

        self.patch_skip = patch_skip
        self.decouple_mu = decouple_mu
        self.separate_zf = separate_zf
        self.df_base_resolution = [int(r) for r in df_base_resolution]
        self.df_space = 5
        self.phi_space = 3

        if separate_zf:
            in_channels = 2 * in_channels
            out_channels = 2 * out_channels
            self.zf_norm = nn.LayerNorm(in_channels) if zf_norm else nn.Identity()

        if decouple_mu:
            self.df_space = 4
            df_full_resolution = self.df_base_resolution
            df_patch_size = [df_patch_size[0]] + df_patch_size[2:]
            df_window_size = [df_window_size[0]] + df_window_size[2:]
            self.df_deoupled_dim = df_full_resolution[1]
            self.df_base_resolution = [df_full_resolution[0]] + df_full_resolution[2:]
            # positional information for velocity mixing
            self.vel_pe = PositionalEmbedding(in_channels, list(df_full_resolution))

            in_channels = in_channels * self.df_deoupled_dim
            out_channels = out_channels * self.df_deoupled_dim

        self.df_unet = SwinUnet(
            self.df_space,
            dim=dim,
            base_resolution=self.df_base_resolution,
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
            self.phi_space,
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

        self.flux_head = None
        if flux_head:
            self.flux_head = FluxDecoder(
                self.phi_unet.down_dims[::-1],
                num_heads=8,
                drop=flux_drop,
                attn_drop=0.1,
            )

        # down/middle direction
        df_mix = []
        phi_mix = []
        for df_dim, phi_dim in zip(self.df_unet.down_dims, self.phi_unet.down_dims):
            df_mix.append(MixingBlock(df_dim, 8, attn_drop=0.1))
            phi_mix.append(MixingBlock(phi_dim, 8, attn_drop=0.1))
        self.df_mix = nn.ModuleList(df_mix)
        self.phi_mix = nn.ModuleList(phi_mix)
        # up direction
        df_mix_up = []
        phi_mix_up = []
        for df_blk, phi_blk in zip(self.df_up_blocks, self.phi_up_blocks):
            df_mix_up.append(MixingBlock(df_blk.dim, 8, attn_drop=0.1))
            phi_mix_up.append(MixingBlock(phi_blk.dim, 8, attn_drop=0.1))
        self.df_mix_up = nn.ModuleList(df_mix_up)
        self.phi_mix_up = nn.ModuleList(phi_mix_up)

        df_patch_dim = self.df_unet.unpatch.dim * (2 if patch_skip else 1)
        phi_patch_dim = self.phi_unet.unpatch.dim * (2 if patch_skip else 1)
        self.df_mix_unpatch = MixingBlock(df_patch_dim, 8, attn_drop=0.1)
        self.phi_mix_unpatch = MixingBlock(phi_patch_dim, 8, attn_drop=0.1)

    def forward(
        self, df: torch.Tensor, phi: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        df, phi, df_pad_axes, phi_pad_axes = self.patch_encode(df, phi)

        if self.patch_skip:
            df0 = df.clone()
            phi0 = phi.clone()

        # parameter conditioning
        # TODO why have two?
        df_cond = self.df_unet.condition(kwargs)
        phi_cond = self.phi_unet.condition(kwargs)

        # down paths
        df_features = []
        phi_features = []
        for df_blk, phi_blk, df_mix, phi_mix in zip(
            self.df_down_blocks,
            self.phi_down_blocks,
            self.df_mix[:-1],
            self.phi_mix[:-1],
        ):
            # mix latents
            df, phi = df_mix(df, phi), phi_mix(phi, df)
            # down blocks
            df, df_pre = df_blk(df, **df_cond)
            phi, phi_pre = phi_blk(phi, **phi_cond)
            df_features.append(df_pre)
            phi_features.append(phi_pre)

        # middle blocks + latent mixing
        flux_lats = []
        df = self.df_unet.middle_pe(df)
        phi = self.phi_unet.middle_pe(phi)

        df, phi = self.df_mix[-1](df, phi), self.phi_mix[-1](phi, df)

        df = self.df_unet.middle(df, **df_cond)
        phi = self.phi_unet.middle(phi, **phi_cond)

        if self.flux_head is not None:
            flux_lats.append(self.flux_head.mix(0, phi, df))

        df = self.df_unet.middle_upscale(df)
        phi = self.phi_unet.middle_upscale(phi)

        # up path
        df_features = df_features[::-1]
        phi_features = phi_features[::-1]
        for i, (df_blk, phi_blk, df_mix, phi_mix) in enumerate(
            zip(self.df_up_blocks, self.phi_up_blocks, self.df_mix_up, self.phi_mix_up)
        ):
            # mix latents
            df, phi = df_mix(df, phi), phi_mix(phi, df)
            # up blocks
            df, df_ = df_blk(df, s=df_features[i], return_skip=True, **df_cond)
            phi, phi_ = phi_blk(phi, s=phi_features[i], return_skip=True, **phi_cond)
            # multiscale flux latents
            if self.flux_head is not None:
                flux_lats.append(self.flux_head.mix(i + 1, phi_, df_))

        # expand to original
        if self.patch_skip:
            df = torch.cat([df, df0], -1)
            phi = torch.cat([phi, phi0], -1)

        df, phi = self.patch_decode(
            df,
            phi,
            df_cond=df_cond["condition"],
            phi_cond=phi_cond["condition"],
            df_pad_axes=df_pad_axes,
            phi_pad_axes=phi_pad_axes,
        )

        flux = None
        if self.flux_head is not None:
            flux = self.flux_head(flux_lats)
        return df, phi, flux

    def patch_encode(self, df: torch.Tensor, phi: torch.Tensor):
        if self.separate_zf:
            # split zonal flow
            zf = df.mean(-1, True).repeat(1, 1, 1, 1, 1, 1, df.shape[-1])
            no_zf = df - zf
            df = torch.cat([zf, no_zf], dim=1)  # stack on channels
            df = rearrange(df, "b c ... -> b ... c")
            df = self.zf_norm(df)
            df = rearrange(df, "b ... c -> b c ...")

        # decouple mu and add positional information
        if self.decouple_mu:
            df = rearrange(df, "b c ... -> b ... c")
            df = self.vel_pe(df)
            df = rearrange(df, "b vp mu ... c -> b (c mu) vp ...")
        # compress to patch space
        df, df_pad_axes = self.df_unet.patch_encode(df)
        phi, phi_pad_axes = self.phi_unet.patch_encode(phi)

        return df, phi, df_pad_axes, phi_pad_axes

    def patch_decode(
        self,
        zdf: torch.Tensor,
        zphi: torch.Tensor,
        df_cond: torch.Tensor,
        phi_cond: torch.Tensor,
        df_pad_axes: Sequence,
        phi_pad_axes: Sequence,
    ):
        # final mixing
        zdf, zphi = self.df_mix_unpatch(zdf, zphi), self.phi_mix_unpatch(zphi, zdf)
        # expand to original
        df = self.df_unet.patch_decode(zdf, df_cond, df_pad_axes)
        phi = self.phi_unet.patch_decode(zphi, phi_cond, phi_pad_axes)
        # move mu back to spatials
        if self.decouple_mu:
            df = rearrange(
                df, "b (c mu) vp ... -> b c vp mu ...", mu=self.df_deoupled_dim
            )
        # recompose zonal flow
        if self.separate_zf:
            df = torch.cat([df[:, 0::2].sum(1, True), df[:, 1::2].sum(1, True)], dim=1)

        return df, phi
