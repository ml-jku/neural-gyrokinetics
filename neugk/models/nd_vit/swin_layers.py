"""
based on: Liu et al., 
Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
https://arxiv.org/abs/2103.14030
"""

from typing import Type, Sequence, Union, Optional

from itertools import product
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from math import ceil
from functools import partial
from torch.nn.attention import SDPBackend, sdpa_kernel
import torch.distributed as dist

from neugk.models.nd_vit.drop import DropPath
from neugk.models.nd_vit.positional import RPB, RotaryPE
from neugk.models.nd_vit.patching import unpad, pad_to_blocks
from neugk.models.layers import Film, seq_weight_init, MLP, DiT, Gate


def window_partition(x, window_size):
    """Window partition operation is n- dimensions.

    Partition tokens into their respective windows

     Args:
        x: input tensor (B, H, W, ..., C)

        window_size: local window size.


    Returns:
        windows: (B * num_windows, window_size ** space, C)
    """

    win_split = []
    win_permute_even = []
    win_permute_odd = []
    for i, w in enumerate(window_size):
        # number of windows in ith dimension
        win_split.append(x.shape[1 + i] // w)  # + 1 for batch dimension
        # window size in ith dimension
        win_split.append(w)
        win_permute_even.append(1 + i * 2)
        win_permute_odd.append(1 + i * 2 + 1)

    win_permute = win_permute_even + win_permute_odd

    b, c = x.shape[0], x.shape[-1]
    windows = x.view(b, *win_split, c)
    windows = windows.permute(0, *win_permute, -1)
    windows = windows.flatten(0, len(window_size))  # flatten batch and num windows
    windows = windows.flatten(1, -2)  # flatten windows

    return windows  # (b * num windows, prod(window_size), c)


def window_reverse(windows, window_size, dims):
    """Window reconstruction/reverse operation in n- dimensions.

     Args:
        windows: windows tensor (B * num_windows, window_size[0], ..., C)
        window_size: local window size.
        dims: dimension (B, H, W, ...).

    Returns:
        x: (B, H, W, ..., C)
    """

    space = len(window_size)
    win_split_count = []
    win_split_win = []
    win_permute = []
    for i, (w, d) in enumerate(zip(window_size, dims[1:])):
        # number of windows in ith dimension
        win_split_count.append(d // w)
        # window size in ith dimension
        win_split_win.append(w)
        win_permute.append(1 + i)
        win_permute.append(1 + i + space)

    b = dims[0]
    x = windows.view(b, *win_split_count, *win_split_win, -1)
    x = x.permute(0, *win_permute, -1)
    # recompose by flattening window count and size in pairs
    for i in range(space):
        x = x.flatten(i + 1, i + 2)
    return x


def get_window_size(grid_size, window_size, shift_size=None):
    """Window size correction.

    Args:
       grid_size: input size.
       window_size: local window size.
       shift_size: window shifting size.
    """

    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(grid_size)):
        if grid_size[i] <= window_size[i]:
            use_window_size[i] = grid_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


def compute_mask(dims, window_size, shift_size):
    """Computing region masks for n- dimension shifted windows.

    Args:
       dims: dimension values.
       window_size: local window size.
       shift_size: shift size.

    Returns:
        attn_mask: (B * num_windows, window_size ** space, C), window-wise attention
                    masks to account for shifted rolled masks outside the frame.
    """

    img_mask = torch.zeros((1, *dims, 1))
    cnt = 0

    axis_slices = [
        (slice(-s), slice(-w, -s), slice(-s, None))
        for w, s in zip(window_size, shift_size)
    ]

    for ax_idx in product(*axis_slices):
        img_mask[:, *ax_idx, :] = cnt
        cnt += 1

    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100))
    attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))

    mask_windows = window_partition(img_mask, window_size)

    return attn_mask


class WindowAttention(nn.Module):
    """
    Window based multi-head self attention module with relative position bias.

    Args:
        dim (int): Number of latent channels.
        num_heads (int): Number of attention heads.
        window_size (int | tuple(int)): Local window size.
        qkv_bias (bool): Add a learnable bias to query, key, value. Default is False.
        attn_drop (float): Attention dropout rate. Default is 0.
        proj_drop (float): Dropout rate of output. Default is 0.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Sequence[int],
        qkv_bias: bool = False,
        qk_norm: bool = False,
        dim_out: Optional[int] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        init_weights: Optional[str] = None,
        use_rpb: bool = True,
        use_rope: bool = False,
        cosine_attn: bool = False,
        gated_attention: bool = False,
    ):

        super().__init__()
        self.dim = dim
        self.dim_out = dim_out if dim_out else dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_drop = attn_drop
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm
        space = len(window_size)
        self.use_rpb = use_rpb
        self.use_rope = use_rope
        self.cosine_attn = cosine_attn
        self.gated_attention = gated_attention

        if use_rpb:
            self.rpb = RPB(space, window_size, num_heads)

        if use_rope:
            # TODO use real only with bf16
            self.rope = RotaryPE(self.head_dim, window_size, use_complex=False)

        if cosine_attn:
            # for swinv2 cosine similarity attention
            self.logit_scale = nn.Parameter(
                torch.log(10 * torch.ones((num_heads, 1, 1)))
            )
            self.register_buffer(
                "max_logits", torch.log(torch.tensor(1.0 / 0.01)), persistent=False
            )
            self.attn_drop = nn.Dropout(attn_drop)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, self.dim_out)
        self.proj_drop = nn.Dropout(proj_drop)

        if gated_attention:
            self.gate = Gate(self.head_dim)

        if qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim, self.head_dim)
            self.k_norm = nn.RMSNorm(self.head_dim, self.head_dim)

        if init_weights:
            self.reset_parameters(init_weights)

    def reset_parameters(self, init_weights: str):
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
        if self.use_rpb:
            self.rpb.reset_parameters(init_weights)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, N, N) or None
        """
        qkv = rearrange(
            self.qkv(x),
            "b n (three heads c) -> three b heads n c",
            three=3,
            heads=self.num_heads,
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)

        # expand mask to head and batch
        if mask is not None:
            mask = mask.unsqueeze(1)  # head dimension
            mask = mask.repeat(q.shape[0] // mask.shape[0], 1, 1, 1)  # batch dimension

        # positional encoding
        if self.use_rpb:
            rpb = self.rpb(q)
            mask = mask + rpb if mask is not None else rpb
        if self.use_rope:
            # rotary positional embedding (faster, sparse mask)
            q, k = self.rope(q), self.rope(k)

        # window attention
        if not self.cosine_attn:
            # swinv1 sdpa attention
            attn_drop = self.attn_drop if self.training else 0.0
            if dist.is_initialized():
                with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
                    x = F.scaled_dot_product_attention(q, k, v, mask, attn_drop)
            else:
                x = F.scaled_dot_product_attention(q, k, v, mask, attn_drop)
        if self.cosine_attn:
            # swinv2 cosine similarity attention
            attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
            logit_scale = torch.clamp(self.logit_scale, max=self.max_logits).exp()
            attn = attn * logit_scale
            attn = attn + mask
            attn = self.attn_drop(F.softmax(attn, dim=-1))
            x = attn @ v

        if self.gated_attention:
            # gated headwise attention before readout (https://arxiv.org/pdf/2505.06708)
            x = self.gate(x, g=q)

        # attention readout
        x = rearrange(x, "b k n c -> b n (k c)")
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer block.

    Handles shifts and window partitions, applies window-wise attention with the
    configured roll attention mask, reverses the windows and undoes the roll and shift,
    and applies the output MLP.


    Args:
        dim (int): Number of hidden channels.
        num_heads (int): Number of attention heads.
        window_size (tuple(int)): Local window size.
        shift_size (tuple(int)): Window shift size (for each dimension).
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
        num_heads: int,
        grid_size: Sequence[int],
        window_size: Sequence[int],
        shift_size: Sequence[int],
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
        use_rpb: bool = True,
        use_rope: bool = False,
        gated_attention: bool = False,
    ):

        super().__init__()
        self.space = space
        self.dim = dim
        self.dim_out = dim_out if dim_out else dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.norm_layer = norm_layer
        self.use_checkpoint = use_checkpoint
        self.init_weights = init_weights

        self.window_size, self.shift_size = get_window_size(
            grid_size, window_size, shift_size
        )

        assert len(self.window_size) == len(self.shift_size) == space

        self.norm1 = (
            norm_layer(self.dim_out, elementwise_affine=False)
            if norm_layer is not None
            else nn.Identity()
        )
        self.attn = WindowAttention(
            dim=dim,
            dim_out=self.dim_out,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=drop,
            use_rpb=use_rpb,
            use_rope=use_rope,
            gated_attention=gated_attention,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = (
            norm_layer(self.dim_out, elementwise_affine=False)
            if norm_layer is not None
            else nn.Identity()
        )
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.skip = (
            nn.Linear(dim, self.dim_out, bias=False)
            if self.dim_out != dim
            else nn.Identity()
        )
        self.mlp = MLP(
            [self.dim_out, mlp_hidden_dim, self.dim_out], act_fn, dropout_prob=drop
        )

        # precompute attention mask
        if any([s > 0 for s in self.shift_size]):
            window_size, shift_size = get_window_size(
                grid_size, self.window_size, self.shift_size
            )
            mask_dims = [ceil(grid_size[i] / w) * w for i, w in enumerate(window_size)]
            attn_mask = compute_mask(mask_dims, window_size, shift_size)
            attn_mask = attn_mask.requires_grad_(False)
            self.register_buffer("attn_mask", attn_mask, persistent=False)
        else:
            self.attn_mask = None

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

    def forward_part1(self, x: torch.Tensor):
        grid_size = x.shape[1:-1]
        # pad to windows
        x, pad_axes = pad_to_blocks(x, self.window_size)
        # shift roll
        if any(i > 0 for i in self.shift_size):
            x = torch.roll(
                x,
                shifts=[-s for s in self.shift_size],
                dims=list(range(1, self.space + 1)),
            )
        x_windows = window_partition(x, self.window_size)
        # shifted window attention
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        # reshape to window/grid
        attn_windows = attn_windows.view(-1, *(self.window_size + (x.shape[-1],)))
        x = window_reverse(attn_windows, self.window_size, x.shape[:-1])
        # invert shift roll
        if any(i > 0 for i in self.shift_size):
            x = torch.roll(
                x,
                shifts=[s for s in self.shift_size],
                dims=list(range(1, self.space + 1)),
            )
        x = unpad(x, pad_axes, grid_size)
        # NOTE swinv2 attention norm
        if not isinstance(self, DiTSwinTransformerBlock):
            x = self.norm1(x)
        return x

    def forward_part2(self, x: torch.Tensor):
        # NOTE swinv2 mlp norm
        x = self.norm2(self.drop_path(self.mlp(x)))
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
            x = x + checkpoint.checkpoint(self.forward_part2, x, use_reentrant=False)
        else:
            x = x + self.forward_part2(x)
        x = shortcut + x
        return x


class DiTSwinTransformerBlock(SwinTransformerBlock):
    """DiT conditioned Swin Transformer block."""

    def __init__(self, *args, cond_dim: int = 2, **kwargs):

        super().__init__(*args, **kwargs)

        # applied before attention projection, needs input dims
        self.norm1 = (
            self.norm_layer(self.dim, elementwise_affine=False)
            if self.norm_layer is not None
            else nn.Identity()
        )

        self.dit = DiT(self.dim, dim2=self.dim_out, cond_dim=cond_dim)

        if self.init_weights:
            self.reset_parameters(self.init_weights)

    def reset_parameters(self, init_weights):
        super().reset_parameters(init_weights)
        self.dit.reset_parameters(init_weights)

    def forward_part1(
        self,
        x: torch.Tensor,
        scale_shift_gate: Sequence[torch.Tensor],
    ):
        scale, shift, gate = scale_shift_gate
        x = self.dit.modulate_scale_shift(self.norm1(x), scale, shift)
        x = super().forward_part1(x)
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


class SwinLayer(nn.Module):
    """
    Basic Swin Transformer layer.

    Pre-computes attention mask, applies swin attention and applies an optional down/up-
    sample layer.

    Args:
        space (int): Number of input/output dimensions.
        dim (int): Number of hidden channels.
        depth (int): Number of swin transformer layers.
        num_heads (int): Number of attention heads.
        grid_size (tuple(int)): Input resolution.
        window_size (tuple(int)): Local window size.
        mlp_ratio (float): Expansion ratio of the mlp hidden dimension. Default is 2.
        qkv_bias (bool): Add a learnable bias to query, key, value. Default is False.
        drop_path (float | tuple(float)): Stochastic depth drop rate. Default is 0.
        drop (float): Attention output dropout rate. Detault is 0.
        attn_drop (float): Attention dropout rate. Default is 0.
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
        window_size: Sequence[int],
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
        use_rpb: bool = True,
        use_rope: bool = False,
        gated_attention: bool = False,
        depth_shifts: bool = False,
        TransformerBlockType: Type[nn.Module] = SwinTransformerBlock,
    ):
        super().__init__()

        self.window_size = window_size
        if depth_shifts:
            # intermediate values as depth allows
            shift_sizes = list(
                zip(
                    *[
                        torch.linspace(0, w // 2, depth, dtype=torch.int).tolist()
                        for w in window_size
                    ]
                )
            )
        else:
            # zero shift interleaved with half shifts
            shift_sizes = [(0,) * space, tuple(i // 2 for i in window_size)] * depth

        if isinstance(drop_path, float):
            drop_path = [drop_path] * depth

        self.space = space
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.dim_out = dim_out if dim_out is not None else dim
        self.grid_size = grid_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm
        self.shift_sizes = shift_sizes
        self.drop_path = drop_path
        self.drop = drop
        self.attn_drop = attn_drop
        self.norm_layer = norm_layer
        self.use_checkpoint = use_checkpoint
        self.act_fn = act_fn
        self.init_weights = init_weights
        self.use_rpb = use_rpb
        self.use_rope = use_rope
        self.gated_attention = gated_attention

        assert dim % num_heads == 0

        blocks = []
        for i in range(depth):
            swin = TransformerBlockType(
                space,
                dim=dim,
                dim_out=self.dim_out if i == depth - 1 else None,
                num_heads=num_heads,
                grid_size=grid_size,
                window_size=window_size,
                shift_size=self.shift_sizes[i],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i],
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
                act_fn=act_fn,
                use_rpb=use_rpb,
                use_rope=use_rope,
                gated_attention=gated_attention,
            )
            blocks.append(swin)
        self.blocks = nn.ModuleList(blocks)

        if init_weights:
            self.reset_parameters(init_weights)

    def reset_parameters(self, init_weights):
        for blk in self.blocks:
            blk.reset_parameters(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x


class FilmSwinLayer(SwinLayer):
    """Film-conditioned Swin Transformer layer."""

    def __init__(self, *args, cond_dim: int, **kwargs):
        super().__init__(*args, **kwargs)

        self.conditioning = nn.ModuleList(
            [Film(cond_dim, self.dim) for _ in range(len(self.blocks))]
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        for blk, cond in zip(self.blocks, self.conditioning):
            x = cond(x, cond=condition)
            x = blk(x)
        return x


class DiTSwinLayer(SwinLayer):
    """DiT-conditioned Swin Transformer layer."""

    def __init__(self, *args, cond_dim: int, **kwargs):
        kwargs.pop("TransformerBlockType", None)
        DiTransformer = partial(DiTSwinTransformerBlock, cond_dim=cond_dim)

        super().__init__(*args, TransformerBlockType=DiTransformer, **kwargs)
        self.cond_dim = cond_dim

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x, cond=condition)
        return x
