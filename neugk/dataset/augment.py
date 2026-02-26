import torch
import einops
from typing import Optional, Callable

def noise_transform(std: float = 1e-4, accumulated: bool = True, window_size: int = 1):
    def _noise(x: torch.Tensor) -> torch.Tensor:
        # this assumes channels normalized uniformly
        if accumulated and window_size > 1:
            x = einops.rearrange(x, "b (t c) ... -> b t c ...", t=window_size)
            sequence_noise = torch.normal(0, std / (window_size**2), size=x.shape)
            sequence_noise = torch.cumsum(sequence_noise, dim=1).to(x.device)
            return (x + sequence_noise).flatten(1, 2)
        else:
            return x + torch.normal(0, std, size=(x.shape), device=x.device)

    return _noise

def reverse_ifft(x, zf_separated=False):
    # Ensure input is a torch.Tensor
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(x)

    if zf_separated:
        real_parts = x[:, ::2]
        imag_parts = x[:, 1::2]
        sum_real = torch.sum(real_parts, dim=1)
        sum_imag = torch.sum(imag_parts, dim=1)
        x = torch.stack([sum_real, sum_imag], dim=1)

    # FFT is batch-wise, i.e. dim=0 is batch
    x = x.permute(0, *range(2, x.ndim), 1).contiguous()
    x = torch.view_as_complex(x)
    x = torch.fft.fftn(x, dim=(-2, -1), norm="forward")
    x = torch.fft.ifftshift(x, dim=(-2,))
    x = torch.stack([x.real, x.imag], dim=0).squeeze()
    x = x.permute(1, 0, *range(2, x.ndim))
    return x.to(torch.float32)

def ifft(x):
    x = x.permute(0, *range(2, x.ndim), 1).contiguous()
    x = torch.view_as_complex(x)
    x = torch.fft.ifftn(x, dim=(-2, -1), norm="forward")
    x = torch.stack([x.real, x.imag]).squeeze().to(torch.float32)
    x = x.permute(1, 0, *range(2, x.ndim))
    return x

def de_normalize(x, file_idx, denormalize_fn):
    # de/normalize physics data keys
    x = torch.stack(
        [
            denormalize_fn(f, x[b])
            for b, f in enumerate(file_idx.tolist())
        ]
    )
    return x

def separate_zf(x):
    nky = x.shape[-1]
    zf = torch.repeat_interleave(x.mean(dim=-1, keepdim=True), repeats=nky, dim=-1)
    x = torch.cat([zf, x - zf], dim=1)
    return x

def mask_modes(
        mask_ratio: float, 
        is_fourier: bool = False, 
        zf_separated: bool = False, 
        weights: Optional[torch.Tensor] = None,
        rescale: bool = True,
        mask_zero_mode: bool = True,
        denormalize_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        normalize_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
    assert 0.0 <= mask_ratio <= 1.0, "mask_ratio must be in [0, 1]"
    if weights is not None and not isinstance(weights, torch.Tensor):
        weights = torch.from_numpy(weights).to(device)

    def _mask(x: torch.Tensor, file_idx: torch.Tensor) -> torch.Tensor:
        device = x.device
        x_tgt = x.clone()
        if not is_fourier:
            # x was dumped in real space, denormalize and transform to fourier again
            if denormalize_fn is not None:
                x = de_normalize(x, file_idx, denormalize_fn)
            x = reverse_ifft(x, zf_separated=zf_separated)
        
        nky = x.shape[-1]
        if weights is None:
            probs = torch.full((nky,), mask_ratio, device=device)
        else:
            normalized_weights = weights / weights.mean()
            probs = mask_ratio * normalized_weights
            probs = torch.clamp(probs, 0.0, 1.0)

        # Handle zero mode separately for linear sims
        if not mask_zero_mode:
            probs[0] = 0.0

        # Sample Bernoulli mask (1 = keep, 0 = drop)
        keep_prob = 1.0 - probs
        mask_1d = torch.bernoulli(keep_prob)

        # Optional rescaling to preserve expected energy
        if rescale:
            scale = torch.where(keep_prob > 0, 1.0 / keep_prob, torch.zeros_like(keep_prob))
            mask_1d = mask_1d * scale

        # Reshape for broadcasting
        shape = [1] * x.ndim
        shape[-1] = nky
        mask = mask_1d.view(shape)
        x_masked = x * mask
        if not is_fourier:
            x_masked = ifft(x_masked)
            if zf_separated:
                # remove zf again if it was removed originally
                x_masked = separate_zf(x_masked)
            if normalize_fn is not None:
                x_masked = de_normalize(x_masked, file_idx, normalize_fn)
        return x_masked, x_tgt

    return _mask