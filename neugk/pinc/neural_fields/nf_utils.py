from typing import Optional, Sequence, Tuple, Callable

import numpy as np
import scipy.linalg
from itertools import product
from copy import deepcopy

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
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


def phi_fft(phi: torch.Tensor, norm: str = "forward"):
    phi = to_complex(phi)
    phi = torch.fft.fftn(phi, dim=(0, 2), norm=norm)
    phi = torch.fft.fftshift(phi, dim=(0,))
    return phi


def phi_ifft(spc: torch.Tensor, norm: str = "forward"):
    spc = to_complex(spc)
    spc = torch.fft.ifftshift(spc, dim=(0,))
    spc = torch.fft.ifftn(spc, dim=(0, 2), norm=norm)
    return to_real(spc)


def sample_field(
    model: nn.Module,
    data,
    device: torch.device,
    timestep: Optional[int] = None,
    full: bool = False,
) -> torch.Tensor:
    blocks = []
    if timestep is None:
        # x = torch.zeros((2, 32, 8, 16, 85, 64), device=device)
        x = torch.zeros_like(data.df, device=device)
        if data.ndim == 6:
            timesteps = list(range(data.grid.shape[0]))
            blocks.append(timesteps)
    else:
        x = torch.zeros_like(data.df[:, 0], device=device)
        blocks.append([timestep])
    # sequential over vpar
    blocks.append(list(range(data.grid.shape[1 if data.ndim == 6 else 0])))
    if data.ndim == 6:
        # and vmu (for larger systems, save ram)
        blocks.append(list(range(data.grid.shape[2 if data.ndim == 6 else 1])))
    model = model.to(device)
    if not full:
        for idxs in product(*blocks):
            coords = data.grid[*idxs].to(device)
            if timestep is None:
                x[:, *idxs] = rearrange(model(coords), "... c -> c ...")
            else:
                x[:, *idxs[1:]] = rearrange(model(coords), "... c -> c ...")
    else:
        coords = data.grid.to(device)
        x = rearrange(model(coords), "... c -> c ...")
    scale = data.scale["df"][..., *[None] * (x.ndim - 2)].to(device)
    shift = data.shift["df"][..., *[None] * (x.ndim - 2)].to(device)
    x = x * scale + shift
    # recompose modes if separated
    if x.shape[0] == 2:
        return x
    else:
        return sum(x.chunk(x.shape[0] // 2))


def load_pretrained_lora(
    model, lora, hot_params: Sequence[str] = ["lora"], lora_postact: bool = True
):
    model_state = model.state_dict()
    lora_state = lora.state_dict()

    for name, param in lora_state.items():
        if name in model_state and model_state[name].shape == param.shape:
            param.copy_(model_state[name])
        elif name not in model_state:
            print(f"{name} not in source model")
            hot_params.append(name)
        else:
            print(f"{name}, {model_state[name].shape} != {param.shape}")
            hot_params.append(name)

    for name, param in lora.named_parameters():
        is_hot = all(h not in name for h in hot_params)
        if lora_postact:
            if is_hot:
                param.requires_grad_(False)
        else:
            if is_hot and not any(n in name for n in ["lora_u", "lora_v"]):
                param.requires_grad_(False)

    if not lora_postact:
        lora.load_state_dict(lora_state, strict=False)
    else:
        lora.load_state_dict(lora_state)

    return lora


def network_distance(model1: nn.Module, model2: nn.Module, distance_metric: str = "l2"):
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()
    shared_keys = set(state_dict1.keys()) & set(state_dict2.keys())

    distances = []
    for key in shared_keys:
        p1 = state_dict1[key].flatten()
        p2 = state_dict2[key].flatten()

        if distance_metric == "l1":
            dist = torch.abs(p1 - p2).mean()
        elif distance_metric == "l2":
            dist = torch.norm(p1 - p2, p=2)
        elif distance_metric == "cosine":
            dist = 1 - F.cosine_similarity(p1.unsqueeze(0), p2.unsqueeze(0))
        elif distance_metric == "max":
            dist = torch.abs(p1 - p2).max()
        elif distance_metric == "kl":
            log_p1 = F.log_softmax(p1, dim=0)
            p2 = F.softmax(p2, dim=0)
            dist = F.kl_div(log_p1, p2, reduction="batchmean")
        elif distance_metric == "wasserstein":
            dist = torch.abs(torch.sort(p1)[0] - torch.sort(p2)[0]).mean()
        else:
            raise ValueError(f"Unknown metric: {distance_metric}")
        distances.append(dist)

    return torch.stack(distances).mean().item() if distances else 0.0


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
    nchannels = 2 if cfg.ky_filter == "base" else 10  # TODO

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


def exact_dmd(x: np.ndarray, r: int = None, dt: float = 0.4):
    """Dynamic Mode Decomposition of an nD sequences of shape (time, d1, d2, ...)."""
    x = rearrange(x.mean(1), "t ... -> (...) t")
    x, xp = x[:, :-1], x[:, 1:]
    u, s, vt = scipy.linalg.svd(x, full_matrices=False)
    if r is None:
        r = np.searchsorted(np.cumsum(s) / np.sum(s), 0.9) + 1
        r = min(r, len(s))
    # truncate
    ur = u[:, :r]
    sr = s[:r]
    vr = vt.conj().T[:, :r]
    sigma_inv = np.diag(1.0 / sr)
    atilde = ur.conj().T @ xp @ vr @ sigma_inv
    # discrete-time eigenvalues
    mu, w = scipy.linalg.eig(atilde)
    # dmd modes (continuous-time eigenvalues)
    phi = xp @ vr @ sigma_inv @ w
    phi = phi / np.linalg.norm(phi, axis=0, keepdims=True)
    # dmd values
    lam = np.log(mu) / dt
    # t0 mode amplitudes
    b = scipy.linalg.lstsq(phi, x[:, 0])[0]
    return {
        "lam": lam,
        "mu": mu,
        "modes": phi,
        "amps": b,
        "r": r,
        "freqs": np.imag(lam) / (2 * np.pi),
    }


def base_optical_flow(x: np.ndarray):
    """Optical flow between frames using Horn-Schunck finite differences.
    Introduces ghost flows because of reshaping to 2D.
    """
    nvp, nmu, ns, nx, ny = x.shape[2:]
    x = rearrange(x, "c t vp mu s x y -> t c (vp mu) (s x y)")
    x_avg = x.mean(axis=1)
    x1 = x_avg[:-1]
    x2 = x_avg[1:]
    grad_x1 = np.gradient(x1, axis=(1, 2))
    grad_x2 = np.gradient(x2, axis=(1, 2))
    xx = 0.5 * (grad_x1[1] + grad_x2[1])
    xy = 0.5 * (grad_x1[0] + grad_x2[0])
    # temporal derivative
    xt = x2 - x1
    # Horn–Schunck
    alpha = 1.0
    denominator = xx**2 + xy**2 + alpha**2
    u = -xx * xt / denominator
    v = -xy * xt / denominator
    flows = np.stack([u, v], axis=1)
    return rearrange(
        flows,
        "t c (vp mu) (s x y) -> c t vp mu s x y",
        vp=nvp,
        mu=nmu,
        s=ns,
        x=nx,
        y=ny,
    )


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


def endpoint_error(
    x1: np.ndarray, x2: np.ndarray, optical_flow_fn: Callable = base_optical_flow
):
    """Endpoint error (EPE) between optical flow fields of two sequences."""
    return float(np.mean((optical_flow_fn(x1) - optical_flow_fn(x2)) ** 2))
