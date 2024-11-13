import itertools
from typing import Optional, Sequence, Type, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm
from einops import rearrange

from utils import look_up_option
from .utils import SUPPORTED_DROPOUT_MODE


class DropPath(nn.Module):
    """Stochastic drop paths per sample for residual blocks.
    Based on:
    https://github.com/rwightman/pytorch-image-models
    """

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True) -> None:
        """
        Args:
            drop_prob: drop path probability.
            scale_by_keep: scaling by non-dropped probability.
        """
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

        if not (0 <= drop_prob <= 1):
            raise ValueError("Drop path prob should be between 0 and 1.")

    def drop_path(self, x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
        if drop_prob == 0.0 or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

    def forward(self, x):
        return self.drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

class MLPBlock(nn.Module):
    """
    A multi-layer perceptron block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """

    def __init__(
        self, hidden_size: int, mlp_dim: int, dropout_rate: float = 0.0, act: tuple | str = "GELU", dropout_mode="vit"
    ) -> None:
        """
        Args:
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer. If 0, `hidden_size` will be used.
            dropout_rate: fraction of the input units to drop.
            act: activation type and arguments. Defaults to GELU. Also supports "GEGLU" and others.
            dropout_mode: dropout mode, can be "vit" or "swin".
                "vit" mode uses two dropout instances as implemented in
                https://github.com/google-research/vision_transformer/blob/main/vit_jax/models.py#L87
                "swin" corresponds to one instance as implemented in
                https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_mlp.py#L23
                "vista3d" mode does not use dropout.

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")
        mlp_dim = mlp_dim or hidden_size
        try:
            self.fn = getattr(nn, act)()
        except AttributeError:
            raise AttributeError(f"Activation fn {act} does not exist!")
        self.linear1 = nn.Linear(hidden_size, mlp_dim) if self.fn != "GEGLU" else nn.Linear(hidden_size, mlp_dim * 2)
        self.linear2 = nn.Linear(mlp_dim, hidden_size)
        # Use Union[nn.Dropout, nn.Identity] for type annotations
        self.drop1: Union[nn.Dropout, nn.Identity]
        self.drop2: Union[nn.Dropout, nn.Identity]

        dropout_opt = look_up_option(dropout_mode, SUPPORTED_DROPOUT_MODE)
        if dropout_opt == "vit":
            self.drop1 = nn.Dropout(dropout_rate)
            self.drop2 = nn.Dropout(dropout_rate)
        elif dropout_opt == "swin":
            self.drop1 = nn.Dropout(dropout_rate)
            self.drop2 = self.drop1
        elif dropout_opt == "vista3d":
            self.drop1 = nn.Identity()
            self.drop2 = nn.Identity()
        else:
            raise ValueError(f"dropout_mode should be one of {SUPPORTED_DROPOUT_MODE}")

    def forward(self, x):
        x = self.fn(self.linear1(x))
        x = self.drop1(x)
        x = self.linear2(x)
        x = self.drop2(x)
        return x

class PatchEmbed5D(nn.Module):
    def __init__(
        self,
        img_size: Tuple,
        patch_size=Tuple,
        in_chans=2,
        embed_dim=24,
        norm_layer=None,
        flatten=True,
    ):
        assert len(img_size) == 5, "Image size must corresponds to (h, w, d, u, v)"
        assert len(patch_size) == 5, "Patch size must corresponds to (h, w, d, u, v)"

        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = [img_size[i] // patch_size[i] for i in range(5)]
        self.embed_dim = embed_dim
        self.flatten = flatten

        self.fc = nn.Linear(in_features=in_chans * np.prod(patch_size), out_features=embed_dim)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # torch.cuda.nvtx.range_push("PatchEmbed")
        assert (
            all([a == b for a, b in zip(x.shape[2:], self.img_size)]) and 
            len(self.img_size) == len(x.shape[2:])
        ), f"Input image shape {x.shape[2:]} doesn't match model ({self.img_size})."
        
        x = self.proj(x)
        if self.flatten:
            x = rearrange(x, "b c pD pH pW pU pV -> b (pD pH pW pU pV) c")
        x = self.norm(x)
        # torch.cuda.nvtx.range_pop()
        return x

    def proj(self, x):
        sD, sH, sW, sU, sV = self.patch_size

        x = rearrange(
            x,
            "b c (pD sD) (pH sH) (pW sW) (pU sU) (pV sV) "
            "-> b pD pH pW pU pV (sD sH sW sU sV c)",
            sD=sD, sH=sH, sW=sW, sU=sU, sV=sV
        )
        
        x = self.fc(x)
        x = rearrange(
            x,
            "b  pD pH pW pU pV emb -> b emb pD pH pW pU pV",
            emb=self.embed_dim
        ).contiguous()
        return x


class PositionalEmbedding(nn.Module):
    """
    Absolute positional embedding module
    """

    def __init__(
        self, dim: int, patch_dim: tuple
    ) -> None:

        super().__init__()
        self.dim = dim
        self.patch_dim = patch_dim
        d, h, w, u, v = patch_dim
        self.pos_embed = nn.Parameter(torch.zeros(1, dim, d, h, w, u, v))
        # self.time_embed = nn.Parameter(torch.zeros(1, dim, 1, 1, 1, t))

        # trunc_normal_(self.pos_embed, std=0.02)
        # trunc_normal_(self.time_embed, std=0.02)


    def forward(self, x):

        x = x + self.pos_embed
        # x = x + self.time_embed[:, :, :, :, :, :t]

        return x


def window_partition(x, window_size):
    """window partition operation based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

    Partition tokens into their respective windows

     Args:
        x: input tensor (B, D, H, W, U, V, C)

        window_size: local window size.


    Returns:
        windows: (B * num_windows, window_size ** 5, C)
    """
    b, d, h, w, u, v, c = x.shape
    x = x.view(
        b,
        d // window_size[0],  # number of windows in depth dimension
        window_size[0],  # window size in depth dimension
        h // window_size[1],  # number of windows in height dimension
        window_size[1],  # window size in height dimension
        w // window_size[2],  # number of windows in width dimension
        window_size[2],  # window size in width dimension
        u // window_size[3],  # number of windows in v1 dimension
        window_size[3],  # window size in v1 dimension
        v // window_size[4],  # number of windows in v2 dimension
        window_size[4],  # window size in v2 dimension
        c,
    )
    # flatten windows on batch
    windows = rearrange(
        x,
        "b nD wD nH wH nW wW nU wU nV wV c -> "
        "(b nD nH nW nU nV) (wD wH wW wU wV) c"
    )
    return windows


def window_reverse(windows, window_size, dims):
    """window reverse operation based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        windows: windows tensor (B*num_windows, window_size, window_size, C)
        window_size: local window size.
        dims: dimension values.

    Returns:
        x: (B, D, H, W, U, V, C)
    """

    b, d, h, w, u, v = dims
    x = windows.view(
        b,
        torch.div(d, window_size[0], rounding_mode="floor"),
        torch.div(h, window_size[1], rounding_mode="floor"),
        torch.div(w, window_size[2], rounding_mode="floor"),
        torch.div(u, window_size[3], rounding_mode="floor"),
        torch.div(v, window_size[4], rounding_mode="floor"),
        window_size[0],
        window_size[1],
        window_size[2],
        window_size[3],
        window_size[4],
        -1,
    )
    x = rearrange(
        x,
        "b nD nH nW nU nV wD wH wW wU wV c -> "
        "b (nD wD) (nH wH) (nW wW) (nU wU) (nV wV) c"
    )
    return x


def get_window_size(x_size, window_size, shift_size=None):
    """Computing window size based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        x_size: input size.
        window_size: local window size.
        shift_size: window shifting size.
    """

    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


class WindowAttention5D(nn.Module):
    """
    Window based multi-head self attention module with relative position bias based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Sequence[int],
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            num_heads: number of attention heads.
            window_size: local window size.
            qkv_bias: add a learnable bias to query, key, value.
            attn_drop: attention dropout rate.
            proj_drop: dropout rate of output.
        """

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.attn_drop = attn_drop

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask):
        """Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, N, N) or None
        """
        qkv = rearrange(self.qkv(x), "b n (three heads c) -> three b heads n c", three=3, heads=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        with nn.attention.sdpa_kernel(nn.attention.SDPBackend.EFFICIENT_ATTENTION):
            x = F.scaled_dot_product_attention(q, k, v, mask, dropout_p=self.attn_drop)
        x = rearrange(x, "b k n c -> b n (k c)")
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock5D(nn.Module):
    """
    Swin Transformer block based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Sequence[int],
        shift_size: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: str = "GELU",
        norm_layer: Type[LayerNorm] = nn.LayerNorm,
        use_checkpoint: bool = False,
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            num_heads: number of attention heads.
            window_size: local window size.
            shift_size: window shift size.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            drop_path: stochastic depth rate.
            act_layer: activation layer.
            norm_layer: normalization layer.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
        """

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention5D(
            dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLPBlock(hidden_size=dim, mlp_dim=mlp_hidden_dim, act=act_layer, dropout_rate=drop, dropout_mode="swin")

    def forward_part1(self, x, mask_matrix):
        b, d, h, w, u, v, c = x.shape
        window_size, shift_size = get_window_size((d, h, w, u, v), self.window_size, self.shift_size)
        x = self.norm1(x)
        pad_d0 = pad_h0 = pad_w0 = pad_u0 = pad_v0 = 0
        pad_d1 = (window_size[0] - d % window_size[0]) % window_size[0]
        pad_h1 = (window_size[1] - h % window_size[1]) % window_size[1]
        pad_w1 = (window_size[2] - w % window_size[2]) % window_size[2]
        pad_u1 = (window_size[3] - u % window_size[3]) % window_size[3]
        pad_v1 = (window_size[4] - v % window_size[4]) % window_size[4]
        x = F.pad(x, (0, 0, pad_v0, pad_v1, pad_u0, pad_u1, pad_w0, pad_w1, pad_h0, pad_h1, pad_d0, pad_d1))  # last tuple first in
        _, dp, hp, wp, up, vp, _ = x.shape
        dims = [b, dp, hp, wp, up, vp]
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(
                x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2], -shift_size[3], -shift_size[4]), dims=(1, 2, 3, 4, 5)
            )
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        x_windows = window_partition(shifted_x, window_size)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, *(window_size + (c,)))
        shifted_x = window_reverse(attn_windows, window_size, dims)
        if any(i > 0 for i in shift_size):
            x = torch.roll(
                shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2], shift_size[3], shift_size[4]), dims=(1, 2, 3, 4, 5)
            )
        else:
            x = shifted_x

        if pad_d1 > 0 or pad_h1 > 0 or pad_w1 > 0 or pad_u1 > 0 or pad_v1 > 0:
            x = x[:, :d, :h, :w, :u, :v, :].contiguous()

        return x

    def forward_part2(self, x):
        x = self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward(self, x, mask_matrix):
        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix, use_reentrant=False)
        else:
            x = self.forward_part1(x, mask_matrix)
        x = shortcut + self.drop_path(x)
        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x, use_reentrant=False)
        else:
            x = x + self.forward_part2(x)
        return x


class PatchMergingV2(nn.Module):
    """
    Patch merging layer based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self, dim: int, norm_layer: Type[LayerNorm] = nn.LayerNorm, spatial_dims: int = 3, c_multiplier: int = 2
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            norm_layer: normalization layer.
            spatial_dims: number of spatial dims.
        """

        super().__init__()
        self.dim = dim

        # Skip dimension reduction on the temporal dimension

        self.reduction = nn.Linear(2 ** 5 * dim, c_multiplier * dim, bias=False)
        self.norm = norm_layer(2 ** 5 * dim)

    def forward(self, x):
        x_shape = x.size()
        b, d, h, w, u, v, c = x_shape
        x = torch.cat(
            [x[:, i::2, j::2, k::2, m::2, h::2, :] for i, j, k, m, h in itertools.product(range(2), range(2), range(2), range(2), range(2))],
            -1,
        )
        x = self.norm(x)
        x = self.reduction(x)

        return x


def compute_mask(dims, window_size, shift_size, device, dtype):
    """Computing region masks based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        dims: dimension values.
        window_size: local window size.
        shift_size: shift size.
        device: device.
    """

    cnt = 0

    d, h, w, u, v = dims
    img_mask = torch.zeros((1, d, h, w, u, v, 1), device=device, dtype=dtype)
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                for u in slice(-window_size[3]), slice(-window_size[3], -shift_size[3]), slice(-shift_size[3], None):
                    for v in slice(-window_size[4]), slice(-window_size[4], -shift_size[4]), slice(-shift_size[4], None):
                        img_mask[:, d, h, w, u, v, :] = cnt
                        cnt += 1

    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    return attn_mask


class Swin5DLayer(nn.Module):
    """
    Basic Swin Transformer layer in one stage based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: Sequence[int],
        drop_path: list,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        norm_layer: Type[LayerNorm] = nn.LayerNorm,
        c_multiplier: int = 2,
        downsample: Optional[nn.Module] = None,
        use_checkpoint: bool = False,
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            depth: number of layers in each stage.
            num_heads: number of attention heads.
            window_size: local window size.
            drop_path: stochastic depth rate.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            norm_layer: normalization layer.
            downsample: an optional downsampling layer at the end of the layer.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
        """

        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.no_shift = tuple(0 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock5D(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=self.no_shift if (i % 2 == 0) else self.shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    use_checkpoint=use_checkpoint,
                )
                for i in range(depth)
            ]
        )
        self.downsample = downsample
        if callable(self.downsample):
            self.downsample = downsample(
                dim=dim, norm_layer=norm_layer, spatial_dims=len(self.window_size), c_multiplier=c_multiplier
            )

    def forward(self, x):
        b, c, d, h, w, u, v = x.size()
        window_size, shift_size = get_window_size((d, h, w, u, v), self.window_size, self.shift_size)
        x = rearrange(x, "b c d h w u v -> b d h w u v c")
        dp = int(np.ceil(d / window_size[0])) * window_size[0]
        hp = int(np.ceil(h / window_size[1])) * window_size[1]
        wp = int(np.ceil(w / window_size[2])) * window_size[2]
        up = int(np.ceil(u / window_size[3])) * window_size[3]
        vp = int(np.ceil(v / window_size[4])) * window_size[4]
        attn_mask = None
        # attn_mask = compute_mask([dp, hp, wp, up, vp], window_size, shift_size, x.device, x.dtype)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = x.view(b, d, h, w, u, v, -1)
        if self.downsample is not None:
            x = self.downsample(x)
        x = rearrange(x, "b d h w u v c -> b c d h w u v")

        return x


class Swin5DModel(nn.Module):
    def __init__(self, patch_size, img_size, dim=16, in_channels=2, out_channels=2, num_layers=2):
        super().__init__()
        
        self.patch_size = patch_size
        self.out_channels = out_channels
        
        self.patch_embed = PatchEmbed5D(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=dim,
            flatten=False,
        )
        
        pos_embeds = []
        patch_dim = self.patch_embed.grid_size
        for _ in range(num_layers):
            # pos_embeds.append(PositionalEmbedding(dim, patch_dim))
            pos_embeds.append(nn.Identity())
        self.pos_embeds = nn.ModuleList(pos_embeds)
        
        convs = []
        for _ in range(num_layers):
            convs.append(Swin5DLayer(dim, depth=2, num_heads=2, window_size=[7, 7, 7, 4, 4], drop_path=0.1, mlp_ratio=2, use_checkpoint=True))
        
        self.convs = nn.ModuleList(convs)
        
        self.patch_up = nn.Linear(dim, np.prod(patch_size) * out_channels)
        
    def readout(self, x):
        x = rearrange(x, "b c pD pH pW pU pV -> b pD pH pW pU pV c")
        x = self.patch_up(x)
        sD, sH, sW, sU, sV = self.patch_size
        x = rearrange(
            x,
            "b pD pH pW pU pV (sD sH sW sU sV out) "
            "-> b out (pD sD) (pH sH) (pW sW) (pU sU) (pV sV)",
            sD=sD, sH=sH, sW=sW, sU=sU, sV=sV, out=self.out_channels
        )
        return x
    
    def forward(self, x):
        """
        Args:
            x: Tensor (B, C, D, H, W, U, V)

        Returns:
            Tensor (B, C, D, H, W, U, V)
        """
        x = self.patch_embed(x)
        for pos, conv in zip(self.pos_embeds, self.convs):
            x = pos(x)
            x = conv(x)
        
        x = self.readout(x)
        return x
