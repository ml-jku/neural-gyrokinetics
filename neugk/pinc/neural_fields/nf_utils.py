from typing import Optional, Sequence, Tuple, Callable

import numpy as np
from itertools import product
from copy import deepcopy

from einops import rearrange
import torch
import torch.nn as nn
from neugk.utils import recombine_zf
from zipnn import ZipNN
import zfpy
from scipy.ndimage import convolve


ACTS = {
    "silu": nn.SiLU,
    "gelu": nn.GELU,
    "mish": nn.Mish,
    "relu": nn.ReLU,
    "lrelu": nn.LeakyReLU,
}


def to_complex(x: torch.Tensor) -> torch.Tensor:
    assert x.shape[0] == 2, x.shape
    x = rearrange(x, "c ... -> ... c").contiguous()
    return torch.view_as_complex(x).squeeze()


def to_real(x: torch.Tensor) -> torch.Tensor:
    return rearrange(torch.view_as_real(x), "... c -> c ...").squeeze()


def df_fft(df: torch.Tensor, norm: str = "forward"):
    if df.shape[0] > 2:
        df = recombine_zf(df, dim=0)
    df = to_complex(df)
    df = torch.fft.fftn(df, dim=(-2, -1), norm=norm)
    df = torch.fft.fftshift(df, dim=(-2,))
    return to_real(df)


def df_ifft(df: torch.Tensor, norm: str = "forward"):
    if df.shape[0] > 2:
        df = recombine_zf(df, dim=0)
    df = to_complex(df)
    df = torch.fft.ifftshift(df, dim=(-2,))
    df = torch.fft.ifftn(df, dim=(-2, -1), norm=norm)
    return to_real(df)


def sample_field(
    model: nn.Module,
    data,
    device: torch.device,
    timestep: Optional[int] = None,
    full: bool = False,
    point_batch: int = 2**20,
) -> torch.Tensor:
    """Sample the neural field over the full grid.

    Fully vectorized: the grid is flattened to a point cloud and run through the
    model in contiguous chunks of ``point_batch`` points, then reshaped back.
    ``full=True`` does it in a single forward (may OOM for large grids).
    """
    model = model.to(device)

    grid = data.grid
    if timestep is not None and data.ndim == 6:
        grid = grid[timestep]
    field_shape = grid.shape[:-1]  # (vpar, vmu, s, x, y) [+ leading t]

    coords = grid.reshape(-1, grid.shape[-1]).to(device)
    n = coords.shape[0]
    bs = n if full else min(point_batch, n)
    outs = [model(coords[i : i + bs]) for i in range(0, n, bs)]
    out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=0)  # (N, c)
    x = out.reshape(*field_shape, -1).movedim(-1, 0)  # (c, *field_shape)

    # denormalise; drop the leading (time) stats axis when sampling one frame.
    # index it only if stats are per-timestep (size > 1), else it broadcasts.
    scale, shift = data.scale["df"].to(device), data.shift["df"].to(device)
    if timestep is not None and scale.ndim == x.ndim + 1:
        ti = timestep if scale.shape[1] > 1 else 0
        scale, shift = scale[:, ti], shift[:, ti]
    x = x * scale + shift
    if x.shape[0] == 2:
        return x
    return sum(x.chunk(x.shape[0] // 2))


def compress_weights(
    model: nn.Module, method: str = "zfp", tolerance: Optional[float] = None
):
    # TODO currently compresses each weight vector separately, zipnn can do better
    state_dict = model.state_dict()
    weights = {k: v.cpu() for k, v in state_dict.items()}
    compressed_weights = {}
    original_size = 0
    compressed_size = 0
    # compress
    if method == "zipnn":
        zpn = ZipNN(input_format="torch")
    for k, arr in weights.items():
        if method == "zfp":
            arr_bytes = zfpy.compress_numpy(arr.numpy(), tolerance=tolerance)
            compressed_size += len(arr_bytes)
        if method == "zipnn":
            arr_bytes = zpn.compress(arr)
            compressed_size += len(arr_bytes)
        if "quantize" in method:
            if method.endswith("8"):
                arr_bytes = arr.to(dtype=torch.float8_e4m3fn)
            if method.endswith("16"):
                arr_bytes = arr.to(dtype=torch.float16)
            compressed_size += arr_bytes.nbytes
        original_size += arr.nbytes
        compressed_weights[k] = arr_bytes
    # decompress
    decompressed_weights = {}
    for k, compressed_arr in compressed_weights.items():
        if method == "zfp":
            arr = torch.from_numpy(zfpy.decompress_numpy(compressed_arr))
        if method == "zipnn":
            arr = zpn.decompress(compressed_arr)
        if "quantize" in method:
            arr = compressed_arr.to(dtype=torch.float32)
        decompressed_weights[k] = arr
    # TODO not ideal
    model_compressed = deepcopy(model)
    state_dict = model_compressed.state_dict()
    for k in state_dict.keys():
        state_dict[k] = (
            decompressed_weights[k].to(state_dict[k].device).type(state_dict[k].dtype)
        )
    model_compressed.load_state_dict(state_dict)
    return model_compressed, original_size, compressed_size


def load_nf(path: str, device):
    from neugk.pinc.neural_fields.models.utils import get_lora_neural_field
    from neugk.pinc.neural_fields.models.siren import SIREN
    from neugk.pinc.neural_fields.models.wire import WIRE
    from neugk.pinc.neural_fields.models.mlp import MLPNF

    ckp = torch.load(path, map_location=device, weights_only=False)
    cfg = ckp["cfg"]
    ndim = 5
    nchannels = 2 if getattr(cfg, "ky_filter", "base") == "base" else 10

    if cfg.name == "siren":
        model = SIREN(
            ndim,
            nchannels,
            n_layers=cfg.n_layers,
            dim=cfg.dim,
            first_w0=cfg.first_w0,
            hidden_w0=cfg.hidden_w0,
            readout_w0=cfg.readout_w0,
            skips=cfg.skips,
            embed_type=cfg.embed_type,
            clip_out=False,
        )
    if cfg.name == "wire":
        model = WIRE(
            ndim,
            nchannels // 2,
            n_layers=cfg.n_layers,
            dim=cfg.dim,
            complex_out=False,
            real_out=False,
            skips=cfg.skips,
            learnable_w0_s0=True,
        )
    if cfg.name == "mlp":
        model = MLPNF(
            ndim,
            nchannels,
            n_layers=cfg.n_layers,
            dim=cfg.dim,
            act_fn=ACTS[cfg.act_fn],
            use_checkpoint=False,
            skips=cfg.skips,
            embed_type=cfg.embed_type,
        )

    if getattr(cfg, "use_lora", False) and "int" in path:
        model = get_lora_neural_field(model, cfg)

    # remove torch.compile artifacts
    state_dict = {}
    for k, v in ckp["state_dict"].items():
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod.") :]
        state_dict[k] = v

    model.load_state_dict(state_dict)

    return model


def optical_flow_5d(
    x: np.ndarray,
    deltas: Optional[np.ndarray] = None,
    alpha: float = 1.0,
    n_iters: int = 50,
    kernel_size: Optional[Tuple[int]] = None,
):
    """
    Iterative Horn-Schunck 5D optical flow. Enforces smoothness via local averaging.
    """
    # channel mean to get intensity
    x_intensity = x.mean(axis=0)
    # temporal derivative and mid-point intensity
    x1 = x_intensity[:-1]
    x2 = x_intensity[1:]
    # account for deltas
    deltas = deltas if deltas is not None else np.ones(x1.shape[0])
    deltas = deltas.reshape(-1, *[1] * 5)
    xt = (x2 - x1) / deltas
    x_mid = 0.5 * (x1 + x2)

    # generalized 5d spatial gradients
    grads = np.gradient(x_mid, axis=(1, 2, 3, 4, 5))
    # denominator (alpha^2 + |grad|^2)
    sum_squared_grads = sum(g**2 for g in grads)
    denominator = alpha**2 + sum_squared_grads
    velocity = np.zeros((5, *xt.shape))
    if n_iters == 0:
        # non iterative (normal flow, local only)
        for i, g in enumerate(grads):
            velocity[i] = -g * xt / denominator
    else:
        # iterative smoothing with averaging kernel (global approx)
        kernel_size = kernel_size if kernel_size else (3, 3, 3, 3, 3)
        kernel = np.zeros((1, *kernel_size))
        neighbor_weight = 1.0 / (5 * 2)
        # star stencil
        for d in range(5):
            idx_l = [k // 2 for k in kernel_size]
            idx_r = [k // 2 for k in kernel_size]
            idx_l[d] -= 1
            idx_r[d] += 1
            kernel[(0, *idx_l)] = neighbor_weight
            kernel[(0, *idx_r)] = neighbor_weight
        for _ in range(n_iters):
            # compute local averages for each component
            u_avg = np.stack([convolve(u, kernel, mode="constant") for u in velocity])
            # compute brightness consistency update
            grad_dot_u_avg = sum(grads[i] * u_avg[i] for i in range(5))
            # horn schunck update
            for i in range(5):
                velocity[i] = u_avg[i] - grads[i] * (grad_dot_u_avg + xt) / denominator
    return velocity


def endpoint_error(x1: np.ndarray, x2: np.ndarray, optical_flow_fn: Callable):
    """Endpoint error (EPE) between optical flow fields of two sequences."""
    return float(np.mean((optical_flow_fn(x1) - optical_flow_fn(x2)) ** 2))
