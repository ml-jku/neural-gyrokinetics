from typing import Optional, Type, Sequence, Union

from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from torch.nn.attention import SDPBackend, sdpa_kernel
import torch.distributed as dist

from models.nd_vit.drop import DropPath
from models.utils import Film, MLP, DiT, seq_weight_init


class LayerModes(Enum):
    DOWNSAMPLE = "Downsample"
    UPSAMPLE = "Upsample"
    SEQUENCE = "Sequence"


class PatchAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        grid_size: Sequence[int],
        num_heads: int,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        init_weights: Optional[str] = None,
    ) -> None:

        super().__init__()
        self.dim = dim
        self.grid_size = grid_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_drop = attn_drop
        self.qkv_bias = qkv_bias

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
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

    def forward(self, x):
        """Forward function.
        Args:
            x: input features with shape of (B, N, C)
        """
        b, c = x.shape[0], x.shape[-1]
        grid_size = x.shape[1:-1]
        x = x.flatten(1, -2)
        qkv = rearrange(
            self.qkv(x),
            "b n (three heads c) -> three b heads n c",
            three=3,
            heads=self.num_heads,
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        if dist.is_initialized():
            with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
                x = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=(self.attn_drop if self.training else 0.0)
                )
        else:
            x = F.scaled_dot_product_attention(
                q, k, v, dropout_p=(self.attn_drop if self.training else 0.0)
            )

        # attention readout
        x = rearrange(x, "b k n c -> b n (k c)")
        x = self.proj(x)
        x = self.proj_drop(x)
        # back to original shape
        x = x.view(b, *grid_size, c)
        return x


class VisionTransformerBlock(nn.Module):
    """
    Args:
        dim (int): Number of hidden channels.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Expansion ratio of the mlp hidden dimension. Default is 2.
        qkv_bias (bool): Add a learnable bias to query, key, value. Default is False.
        drop (float): Attention output dropout rate. Detault is 0.
        attn_drop (float): Attention dropout rate. Default is 0.
        drop_path (float): Stochastic depth drop rate. Default is 0.
        norm_layer (nn.Module): Normalization layer type. Default is nn.LayerNorm.
        use_checkpoint (bool): Gradient checkpointing (saves memory). Default is False.
        act_fn (callable): Activation function. Default is nn.GELU.
    """

    def __init__(
        self,
        space: int,
        dim: int,
        grid_size: Sequence[int],
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
    ) -> None:

        super().__init__()
        self.space = space
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        self.init_weights = init_weights

        self.norm1 = (
            norm_layer(dim, elementwise_affine=True)
            if norm_layer is not None
            else nn.Identity()
        )
        self.attn = PatchAttention(
            dim,
            grid_size=grid_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = (
            norm_layer(dim, elementwise_affine=True)
            if norm_layer is not None
            else nn.Identity()
        )
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

    def forward_part1(self, x):
        x = self.attn(self.norm1(x))
        return x

    def forward_part2(self, x):
        x = self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward(self, x):
        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x, use_reentrant=False)
        else:
            x = self.forward_part1(x)
        x = shortcut + self.drop_path(x)
        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x, use_reentrant=False)
        else:
            x = x + self.forward_part2(x)
        return x


class DiTVisionTransformerBlock(VisionTransformerBlock):
    """DiT conditioned Vision Transformer block."""

    def __init__(self, *args, cond_dim: int = 2, **kwargs):

        super().__init__(*args, **kwargs)

        self.dit = DiT(self.dim, cond_dim)

        if self.init_weights:
            self.reset_parameters(self.init_weights)

    def reset_parameters(self, init_weights):
        super().reset_parameters(init_weights)
        self.dit.reset_parameters(init_weights)

    def forward_part1(self, x: torch.Tensor, scale_shift_gate: Sequence[torch.Tensor]):
        scale, shift, gate = scale_shift_gate
        x = self.dit.modulate_scale_shift(self.norm1(x), scale, shift)
        x = self.attn(x)
        return self.dit.modulate_gate(x, gate)

    def forward_part2(self, x: torch.Tensor, scale_shift_gate: Sequence[torch.Tensor]):
        scale, shift, gate = scale_shift_gate
        x = self.dit.modulate_scale_shift(self.norm2(x), scale, shift)
        x = self.dit.modulate_gate(self.mlp(x), gate)
        x = self.drop_path(x)
        return x

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        scale1, shift1, gate1, scale2, shift2, gate2 = self.dit(cond)
        mod1 = scale1, shift1, gate1
        mod2 = scale2, shift2, gate2

        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x, mod1, use_reentrant=False)
        else:
            x = self.forward_part1(x, scale_shift_gate=mod1)
        x = shortcut + self.drop_path(x)
        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(
                self.forward_part2, x, mod2, use_reentrant=False
            )
        else:
            x = x + self.forward_part2(x, scale_shift_gate=mod2)
        return x


class ViTLayer(nn.Module):
    """
    Basic Vision Transformer layer.

    Args:
        space (int): Number of input/output dimensions.
        dim (int): Number of hidden channels.
        depth (int): Number of transformer layers.
        num_heads (int): Number of attention heads.
        grid_size (tuple(int)): Input resolution.
        mode (LayerModes): Mark layer operation.
        mlp_ratio (float): Expansion ratio of the mlp hidden dimension. Default is 2.
        qkv_bias (bool): Add a learnable bias to query, key, value. Default is False.
        drop_path (float | tuple(float)): Stochastic depth drop rate. Default is 0.
        drop (float): Attention output dropout rate. Detault is 0.
        attn_drop (float): Attention dropout rate. Default is 0.
        norm_layer (nn.Module): Normalization layer type. Default is nn.LayerNorm.
        c_multiplier (int): Latent dimensions expansions after downsample. Default is 2.
        resample_fn (nn.Module): Optional resampling layer, applied after attention.
        use_checkpoint (bool): Gradient checkpointing (saves memory). Default is False.
        act_fn (callable): Activation function. Default is nn.GELU.
    """

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
        use_checkpoint: bool = False,
        act_fn: nn.Module = nn.GELU,
        init_weights: Optional[str] = None,
    ) -> None:

        super().__init__()

        if isinstance(drop_path, float):
            drop_path = [drop_path] * depth

        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.dim = dim
        self.grid_size = grid_size
        self.drop_path = drop_path

        assert dim % num_heads == 0

        self.blocks = nn.ModuleList(
            [
                VisionTransformerBlock(
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
                    use_checkpoint=use_checkpoint,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x


class FilmViTLayer(ViTLayer):
    """Film-conditioned Vision Transformer layer."""

    def __init__(self, *args, cond_dim: int, **kwargs):
        super().__init__(*args, **kwargs)

        self.conditioning = nn.ModuleList(
            [Film(cond_dim, self.dim) for _ in range(len(self.blocks))]
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        for blk, cond in zip(self.blocks, self.conditioning):
            x = cond(x, condition)
            x = blk(x)
        return x


class DiTLayer(ViTLayer):
    """DiT-conditioned Vision Transformer layer."""

    def __init__(
        self,
        space: int,
        dim: int,
        cond_dim: int,
        depth: int,
        num_heads: int,
        grid_size: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop_path: Union[Sequence[float], float] = 0.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        use_checkpoint: bool = False,
        act_fn: nn.Module = nn.GELU,
        init_weights: Optional[str] = None,
    ):
        super().__init__(
            space=space,
            dim=dim,
            depth=depth,
            num_heads=num_heads,
            grid_size=grid_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_path=drop_path,
            drop=drop,
            attn_drop=attn_drop,
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
            act_fn=act_fn,
            init_weights=init_weights,
        )

        self.blocks = nn.ModuleList(
            [
                DiTVisionTransformerBlock(
                    space,
                    dim=dim,
                    num_heads=num_heads,
                    grid_size=self.grid_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=self.drop_path[i],
                    norm_layer=norm_layer,
                    use_checkpoint=use_checkpoint,
                    act_fn=act_fn,
                    cond_dim=cond_dim,
                )
                for i in range(depth)
            ]
        )

        if init_weights:
            self.reset_parameters(init_weights)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x, cond=condition)
        return x
