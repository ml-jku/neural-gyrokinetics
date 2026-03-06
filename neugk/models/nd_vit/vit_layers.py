from typing import Optional, Type, Sequence, Union
from functools import partial

from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange

from neugk.models.nd_vit.drop import DropPath
from neugk.models.layers import Film, MLP, DiT, seq_weight_init, Gate
from neugk.models.nd_vit.positional import RotaryPE


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
        qk_norm: bool = False,
        dim_out: Optional[int] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        init_weights: Optional[str] = None,
        use_rope: bool = False,
        gated_attention: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out if dim_out else dim
        self.grid_size = grid_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_drop = attn_drop
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm
        self.use_rope = use_rope
        self.gated_attention = gated_attention

        if use_rope:
            # TODO use real only with bf16
            self.rope = RotaryPE(self.head_dim, grid_size, use_complex=False)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, self.dim_out)
        self.proj_drop = nn.Dropout(proj_drop)

        if gated_attention:
            self.gate = Gate(self.head_dim)

        if qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-8)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-8)

        if init_weights:
            self.reset_parameters(init_weights)

    def reset_parameters(self, init_weights):
        if init_weights == "torch" or init_weights is None:
            return
        elif init_weights == "xavier_uniform":
            init_weights_fn = nn.init.xavier_uniform_
        elif init_weights == "kaiming_uniform":
            init_weights_fn = partial(
                nn.init.kaiming_uniform_, nonlinearity="relu", mode="fan_in", a=0
            )
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
        b, grid_size = x.shape[0], x.shape[1:-1]
        x = x.flatten(1, -2)
        qkv = rearrange(
            self.qkv(x),
            "b n (three heads c) -> three b heads n c",
            three=3,
            heads=self.num_heads,
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)

        if self.use_rope:
            # rotary positional embedding (faster, sparse mask)
            q, k = self.rope(q), self.rope(k)

        attn_drop = self.attn_drop if self.training else 0.0

        x = F.scaled_dot_product_attention(q, k, v, dropout_p=attn_drop)

        if self.gated_attention:
            # gated headwise attention before readout (https://arxiv.org/pdf/2505.06708)
            x = self.gate(x, g=q)

        # attention readout
        x = rearrange(x, "b k n c -> b n (k c)")
        x = self.proj(x)
        x = self.proj_drop(x)
        # back to original shape
        x = x.view(b, *grid_size, self.dim_out)
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
        dim_out: Optional[int] = None,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        use_checkpoint: bool = False,
        act_fn: nn.Module = nn.GELU,
        init_weights: Optional[str] = None,
        use_rope: bool = False,
        gated_attention: bool = False,
    ):
        super().__init__()
        self.space = space
        self.dim = dim
        self.dim_out = dim_out if dim_out else dim
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
            dim=dim,
            dim_out=self.dim_out,
            grid_size=grid_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=drop,
            use_rope=use_rope,
            gated_attention=gated_attention,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = (
            norm_layer(self.dim_out, elementwise_affine=True)
            if norm_layer is not None
            else nn.Identity()
        )
        mlp_hidden_dim = int(self.dim_out * mlp_ratio)

        # TODO define behavior
        self.skip = (
            nn.Linear(dim, self.dim_out, bias=False)
            if self.dim_out != dim
            else nn.Identity()
        )
        self.mlp = MLP(
            [self.dim_out, mlp_hidden_dim, self.dim_out], act_fn, dropout_prob=drop
        )

        if init_weights:
            self.reset_parameters(init_weights)

    def reset_parameters(self, init_weights):
        if init_weights == "torch" or init_weights is None:
            pass
        elif init_weights == "xavier_uniform":
            self.mlp.apply(seq_weight_init(nn.init.xavier_uniform_))
        elif init_weights == "kaiming_uniform":
            self.mlp.apply(
                seq_weight_init(
                    partial(
                        nn.init.kaiming_uniform_,
                        nonlinearity="relu",
                        mode="fan_in",
                        a=0,
                    )
                )
            )
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
        shortcut = self.skip(x)
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x, use_reentrant=False)
        else:
            x = self.forward_part1(x)
        x = shortcut + self.drop_path(x)
        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part2, x, use_reentrant=False)
        else:
            x = self.forward_part2(x)
        x = shortcut + x
        return x


class DiTVisionTransformerBlock(VisionTransformerBlock):
    """DiT conditioned Vision Transformer block."""

    def __init__(self, *args, cond_dim: int = 2, **kwargs):
        super().__init__(*args, **kwargs)

        self.dit = DiT(self.dim, dim2=self.dim_out, cond_dim=cond_dim)

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

        shortcut = self.skip(x)
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x, mod1, use_reentrant=False)
        else:
            x = self.forward_part1(x, scale_shift_gate=mod1)
        x = shortcut + self.drop_path(x)
        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part2, x, mod2, use_reentrant=False)
        else:
            x = self.forward_part2(x, scale_shift_gate=mod2)
        x = shortcut + x
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
        dim_out: Optional[int] = None,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        drop_path: Union[Sequence[float], float] = 0.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        use_checkpoint: bool = False,
        act_fn: nn.Module = nn.GELU,
        init_weights: Optional[str] = None,
        use_rope: bool = False,
        gated_attention: bool = False,
        TransformerBlockType: Type[nn.Module] = VisionTransformerBlock,
    ):
        super().__init__()

        if isinstance(drop_path, float):
            drop_path = [drop_path] * depth

        self.space = space
        self.dim = dim
        self.dim_out = dim_out if dim_out else dim
        self.depth = depth
        self.num_heads = num_heads
        self.grid_size = grid_size
        self.drop = drop
        self.drop_path = drop_path
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm
        self.attn_drop = attn_drop
        self.norm_layer = norm_layer
        self.use_checkpoint = use_checkpoint
        self.act_fn = act_fn
        self.use_rope = use_rope
        self.gated_attention = gated_attention
        self.init_weights = init_weights

        assert dim % num_heads == 0

        self.blocks = nn.ModuleList(
            [
                TransformerBlockType(
                    space,
                    dim=dim,
                    dim_out=dim_out if i == (depth - 1) else dim,
                    grid_size=grid_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i],
                    norm_layer=norm_layer,
                    use_checkpoint=use_checkpoint,
                    act_fn=act_fn,
                    use_rope=use_rope,
                    gated_attention=gated_attention,
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

    def __init__(self, *args, cond_dim: int, **kwargs):
        kwargs.pop("TransformerBlockType", None)
        DiTransformer = partial(DiTVisionTransformerBlock, cond_dim=cond_dim)

        super().__init__(*args, TransformerBlockType=DiTransformer, **kwargs)
        self.cond_dim = cond_dim

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x, cond=condition)
        return x
