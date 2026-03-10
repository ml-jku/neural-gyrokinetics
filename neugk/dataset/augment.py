import torch
import einops
from typing import Optional, Callable, Dict
from enum import Enum
from functools import partial


class MaskStrategy(str, Enum):
    RANDOM = "random"
    LOW_FROM_HIGH = "low_from_high"  # mask low modes, predict them from high
    HIGH_FROM_LOW = "high_from_low"  # mask high modes, predict them from low
    ZONAL_FLOW = "zonal_flow"  # mask zero mode only
    MIXED = "mixed"  # uniform coin flip over the four above


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
    x = torch.stack([x.real, x.imag], dim=0)
    x = x.permute(1, 0, *range(2, x.ndim))
    return x.to(torch.float32)


def ifft(x):
    x = x.permute(0, *range(2, x.ndim), 1).contiguous()
    x = torch.view_as_complex(x)
    x = torch.fft.ifftn(x, dim=(-2, -1), norm="forward")
    x = torch.stack([x.real, x.imag], dim=0)
    x = x.permute(1, 0, *range(2, x.ndim))
    return x.to(torch.float32)


def de_normalize(x, file_idx, denormalize_fn):
    # de/normalize physics data keys
    x = torch.stack([denormalize_fn(f, x[b]) for b, f in enumerate(file_idx.tolist())])
    return x


def separate_zf(x):
    nky = x.shape[-1]
    zf = torch.repeat_interleave(x.mean(dim=-1, keepdim=True), repeats=nky, dim=-1)
    x = torch.cat([zf, x - zf], dim=1)
    return x


def _build_mask(
    nky: int,
    strategy: MaskStrategy,
    mask_ratio: float,
    cutoff: int,
    weights: Optional[torch.Tensor],
    mask_zero_mode: bool,
    rescale: bool,
    device: torch.device,
) -> torch.Tensor:
    """Return a 1-D mask of shape (nky,) where 1 = keep, 0 = drop.

    If rescale is True, kept modes are scaled up so expected energy is preserved.
    """
    if strategy == MaskStrategy.RANDOM:
        if weights is None:
            probs = torch.full((nky,), mask_ratio, device=device)
        else:
            w = weights.to(device)
            probs = mask_ratio * (w / w.mean())
            probs = probs.clamp(0.0, 1.0)

        if not mask_zero_mode:
            probs[0] = 0.0

        keep_prob = 1.0 - probs
        mask = torch.bernoulli(keep_prob)

        if rescale:
            scale = torch.where(
                keep_prob > 0, 1.0 / keep_prob, torch.zeros_like(keep_prob)
            )
            mask = mask * scale

    elif strategy == MaskStrategy.LOW_FROM_HIGH:
        # Mask out modes [0, cutoff), keep [cutoff, nky)
        mask = torch.ones(nky, device=device)
        mask[:cutoff] = 0.0
        if not mask_zero_mode:
            mask[0] = 1.0
        if rescale:
            n_kept = mask.sum().clamp(min=1)
            mask = mask * (nky / n_kept)

    elif strategy == MaskStrategy.HIGH_FROM_LOW:
        # Keep modes [0, cutoff), mask out [cutoff, nky)
        mask = torch.ones(nky, device=device)
        mask[cutoff:] = 0.0
        # always rescale energy for high->low to not plateau
        n_kept = mask.sum().clamp(min=1)
        full_scale = nky / n_kept
        scale = 1.0 + torch.rand(1, device=device).item() * (full_scale - 1.0)
        mask = mask * scale

    elif strategy == MaskStrategy.ZONAL_FLOW:
        # Mask only the zero mode
        mask = torch.ones(nky, device=device)
        mask[0] = 0.0
        if rescale:
            mask = mask * (nky / (nky - 1))

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return mask


def _sample_strategy(
    strategy: MaskStrategy,
    mix_weights: Dict[MaskStrategy, float],
) -> MaskStrategy:
    """If strategy is MIXED, sample a leaf strategy according to mix_weights."""
    if strategy != MaskStrategy.MIXED:
        return strategy

    strategies = list(mix_weights.keys())
    probs = torch.tensor([mix_weights[s] for s in strategies])
    probs = probs / probs.sum()
    idx = torch.multinomial(probs, 1).item()
    return strategies[idx]


def mask_modes(
    mask_ratio: float,
    strategy: str | MaskStrategy = MaskStrategy.RANDOM,
    is_fourier: bool = False,
    zf_separated: bool = False,
    weights: Optional[torch.Tensor] = None,
    rescale: bool = True,
    mask_zero_mode: bool = True,
    cutoff: Optional[int] = None,
    mix_weights: Optional[Dict[str | MaskStrategy, float]] = None,
    denormalize_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    normalize_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
):
    assert 0.0 <= mask_ratio <= 1.0, "mask_ratio must be in [0, 1]"
    if weights is not None and not isinstance(weights, torch.Tensor):
        weights = torch.from_numpy(weights).to(device)

    # Resolve mix weights
    _mix = {}
    _leaf = [s for s in MaskStrategy if s != MaskStrategy.MIXED]
    if strategy == MaskStrategy.MIXED:
        if mix_weights is not None:
            for k, v in mix_weights.items():
                _mix[MaskStrategy(k) if isinstance(k, str) else k] = v
        else:
            _mix = {s: 1.0 / len(_leaf) for s in _leaf}
        assert all(
            s in _leaf for s in _mix
        ), f"mix_weights keys must be leaf strategies, got {list(_mix.keys())}"

    def _mask(x: torch.Tensor, file_idx: torch.Tensor) -> torch.Tensor:
        device = x.device
        x_tgt = x.clone()
        if not is_fourier:
            # x was dumped in real space, denormalize and transform to fourier again
            if denormalize_fn is not None:
                x = de_normalize(x, file_idx, denormalize_fn)
            x = reverse_ifft(x, zf_separated=zf_separated)

        nky = x.shape[-1]
        # select strategy
        chosen = _sample_strategy(strategy, _mix)

        mask_1d = _build_mask(
            nky=nky,
            strategy=chosen,
            mask_ratio=mask_ratio,
            weights=weights,
            mask_zero_mode=mask_zero_mode,
            rescale=rescale,
            cutoff=cutoff if cutoff is not None else nky // 2,
            device=device,
        )

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
                x_masked = de_normalize(
                    x_masked, file_idx, partial(normalize_fn, return_stats=False)
                )
        return x_masked, x_tgt, mask, chosen

    return _mask
