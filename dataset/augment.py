import torch
import einops


def noise_transform(std: float = 1e-4, accumulated: bool = True, window_size: int = 1):
    def _noise(x: torch.Tensor) -> torch.Tensor:
        # this assumes channels normalized uniformly
        if accumulated and window_size > 1:
            x = einops.rearrange(x, "b (t c) x y -> b t c x y", t=window_size)
            sequence_noise = torch.normal(0, std / (window_size**2), size=x.shape)
            sequence_noise = torch.cumsum(sequence_noise, dim=1)
            return (x + sequence_noise).flatten(1, 2)
        else:
            return x + torch.normal(0, std, size=(x.shape), device=x.device)

    return _noise
