from typing import Sequence, Union, Optional, Type

import torch
from torch import nn
from einops import rearrange
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn import functional as F
import torch.distributed as dist

from models.nd_vit.drop import DropPath
from models.utils import MLP, seq_weight_init, AttentionDecoder


class MixingBlock(nn.Module):
    def __init__(
        self,
        left_dim: int,
        right_dim: int,
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
        self.left_dim = left_dim
        self.right_dim = right_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.init_weights = init_weights

        self.norm1 = norm_layer(left_dim) if norm_layer is not None else nn.Identity()
        self.attn = AttentionDecoder(
            q_dim=left_dim,
            kv_dim=right_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            init_weights=init_weights,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(left_dim) if norm_layer is not None else nn.Identity()
        mlp_hidden_dim = int(left_dim * mlp_ratio)

        self.mlp = MLP(
            [left_dim, mlp_hidden_dim, left_dim], act_fn=act_fn, dropout_prob=drop
        )

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
        left_dim: int,
        right_dim: int,
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
        self.left_dim = left_dim
        self.right_dim = right_dim
        self.grid_size = grid_size
        self.drop_path = drop_path

        assert left_dim % num_heads == 0

        self.blocks = nn.ModuleList(
            [
                MixingBlock(
                    space,
                    left_dim=left_dim,
                    right_dim=right_dim,
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
        left_dims: Sequence[int],
        right_dims: Sequence[int],
        num_heads: int,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_fn: nn.Module = nn.GELU,
        init_weights: Optional[str] = None,
        detach_latents: bool = False,
        reduction: str = "max",
    ):
        super().__init__()
        self.detach_latents = detach_latents
        self.reduction = reduction
        flux_blocks = []
        reduction_blocks = []
        flux_latent_size = 0
        for left_dim, right_dim in zip(left_dims, right_dims):
            flux_blocks.append(
                MixingBlock(
                    left_dim,
                    right_dim,
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
            if self.reduction == "integral":
                reduction_blocks.append(
                    RSpaceReduce(
                        dim=left_dim,
                        out_dim=left_dim,
                        num_heads=8,
                        attn_drop=0.1,
                        init_weights="xavier_uniform",
                    )
                )
            flux_latent_size += left_dim
        self.blocks = nn.ModuleList(flux_blocks)
        self.reductions = nn.ModuleList(reduction_blocks)

        self.flux_mlp = MLP(
            [flux_latent_size, flux_latent_size // 2, 1],
            # last_act_fn=nn.Softplus,
            dropout_prob=drop,
        )

    def mix(self, i: int, left: torch.Tensor, right: Optional[torch.Tensor] = None):
        if self.detach_latents:
            left = left.detach()
            right = right.detach()
        x = self.blocks[i].forward(left, right)
        # pool spatials
        if self.reduction == "max":
            x = x.amax(axis=list(range(1, x.ndim - 1)))
        elif self.reduction == "mean":
            x = x.mean(axis=list(range(1, x.ndim - 1)))
        else:
            x = self.reductions[i].forward(x)
        return x

    def forward(self, flux_latents: Sequence[torch.Tensor]):
        flux = self.flux_mlp(torch.cat(flux_latents, dim=-1))
        return flux.squeeze(1)


class VSpaceReduce(AttentionDecoder):
    def __init__(
        self,
        dim: int,
        out_dim: int,
        num_heads: int,
        decouple_mu: bool = False,
        gain: float = 1e-2,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        init_weights: Optional[str] = None,
    ):
        super().__init__(
            q_dim=dim,
            kv_dim=dim,
            out_dim=out_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            init_weights=init_weights,
        )
        del self.q
        # learned token to integrate vspace (query projection not needed)
        integral_token = gain * torch.randn(1, 1, dim)
        self.register_buffer("integral_token", integral_token)

        self.decouple_mu = decouple_mu
        self.out_dim = out_dim

    def forward(self, df: torch.Tensor):
        if self.decouple_mu:
            b, _, ns, nx, ny, _ = df.shape
            df = rearrange(df, "b vpar s x y c -> (b s x y) vpar c")
        else:
            b, _, _, ns, nx, ny, _ = df.shape
            df = rearrange(df, "b vpar mu s x y c -> (b s x y) (vpar mu) c")

        # qkv embeddings from inputs
        df = df.contiguous()
        assert df.is_contiguous() and self.integral_token.is_contiguous(), "Tensors not contiguous."
        q = rearrange(self.integral_token, "b n (h c) -> b h n c", h=self.num_heads)
        k, v = rearrange(self.kv(df), "b n (t h c) -> t b h n c", t=2, h=self.num_heads)
        phi = F.scaled_dot_product_attention(
            q, k, v, None, dropout_p=(self.attn_drop if self.training else 0.0)
        )
        phi = rearrange(phi, "b k n c -> b n (k c)")
        phi = self.proj(phi)
        phi = self.proj_drop(phi)
        return phi.view(b, ns, nx, ny, self.out_dim)


class RSpaceReduce(AttentionDecoder):
    def __init__(
            self,
            dim: int,
            out_dim: int,
            num_heads: int,
            gain: float = 1e-2,
            qkv_bias: bool = False,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
            init_weights: Optional[str] = None,
    ):
        super().__init__(
            q_dim=dim,
            kv_dim=dim,
            out_dim=out_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            init_weights=init_weights,
        )
        del self.q
        # learned token to integrate vspace (query projection not needed)
        integral_token = gain * torch.randn(1, 1, dim)
        self.register_buffer("integral_token", integral_token)
        self.out_dim = out_dim

    def forward(self, phi: torch.Tensor):
        b, ns, nx, ny, _ = phi.shape
        phi = rearrange(phi, "b s x y c -> b (s x y) c")

        # qkv embeddings from inputs
        phi = phi.contiguous()
        assert phi.is_contiguous() and self.integral_token.is_contiguous(), "Tensors not contiguous."
        q = rearrange(self.integral_token, "b n (h c) -> b h n c", h=self.num_heads)
        k, v = rearrange(self.kv(phi), "b n (t h c) -> t b h n c", t=2, h=self.num_heads)
        phi = F.scaled_dot_product_attention(
            q, k, v, None, dropout_p=(self.attn_drop if self.training else 0.0)
        )
        phi = rearrange(phi, "b k n c -> b n (k c)")
        phi = self.proj(phi)
        phi = self.proj_drop(phi)
        return phi.view(b, self.out_dim)
