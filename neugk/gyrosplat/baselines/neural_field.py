"""Per-snapshot neural field: one MLP per species that maps a grid point to the two real
channels of the object, trained on this snapshot alone with a field objective.

Architecture and schedule follow the density phase of PINC: one embedding
table per grid axis (the coordinates are integers), the embeddings concatenated, residual
SiLU blocks, a linear readout; AdamW with a cosine schedule stepped once per epoch, the MSE
on the z-scored channels, and every epoch drawn as a random subset of the grid whose share
grows linearly over the epochs.  The width is the largest one whose parameter count fits the
budget.  Stored per species: the parameters and the z-score scalars.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from neugk.gyrosplat.config import NeuralFieldConfig


class _Block(nn.Module):
    def __init__(self, n_in: int, n_out: int, activation: bool):
        super().__init__()
        self.linear = nn.Linear(n_in, n_out)
        self.activation = nn.SiLU() if activation else nn.Identity()
        self.skip = n_in == n_out
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        y = self.activation(self.linear(x))
        return x + y if self.skip else y


class NeuralField(nn.Module):
    """Flat grid index -> (re, im).  The index is split into its integer coordinates inside
    the module, so no coordinate table is kept."""

    def __init__(self, grid: Sequence[int], width: int, layers: int):
        super().__init__()
        per_axis = _embedding_width(width, len(grid)) // len(grid)
        self.embed = nn.ModuleList([nn.Embedding(int(n), per_axis) for n in grid])
        sizes = [per_axis * len(grid)] + [width] * (layers - 1)
        self.blocks = nn.ModuleList(
            [
                _Block(a, b, activation=i < len(sizes) - 2)
                for i, (a, b) in enumerate(zip(sizes, sizes[1:]))
            ]
        )
        self.readout = nn.Linear(width, 2)
        nn.init.xavier_uniform_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)
        strides = np.cumprod([1] + [int(n) for n in grid[:0:-1]])[::-1]
        self.strides = [int(s) for s in strides]

    def forward(self, index):
        parts, rest = [], index
        for table, stride in zip(self.embed, self.strides):
            coordinate = torch.div(rest, stride, rounding_mode="floor")
            parts.append(table(coordinate))
            rest = rest - coordinate * stride
        x = torch.cat(parts, dim=-1)
        for block in self.blocks:
            x = block(x)
        return self.readout(F.silu(x))


def _embedding_width(width: int, n_axes: int) -> int:
    return round(width / n_axes) * n_axes


def parameter_count(grid: Sequence[int], width: int, layers: int) -> int:
    embedded = _embedding_width(width, len(grid))
    sizes = [embedded] + [width] * (layers - 1)
    tables = sum(int(n) * (embedded // len(grid)) for n in grid)
    blocks = sum(a * b + b for a, b in zip(sizes, sizes[1:]))
    return tables + blocks + 2 * width + 2


def solve_width(grid: Sequence[int], budget: int, layers: int):
    """The width with the most parameters within the budget (the count is not monotone in the
    width, because the embedding width is rounded to a multiple of the number of axes)."""
    best = max(
        (w for w in range(8, 4096) if parameter_count(grid, w, layers) <= budget),
        key=lambda w: parameter_count(grid, w, layers),
        default=None,
    )
    if best is None:
        raise ValueError(f"no neural field of {layers} layers fits {budget} parameters")
    return best, parameter_count(grid, best, layers)


def fit_field(
    target: torch.Tensor, grid, width: int, nf: NeuralFieldConfig, seed: int, log
):
    """Train one field on `target` (points, 2), already z-scored.  Returns the module."""
    torch.manual_seed(seed)
    device, n_points = target.device, target.shape[0]
    model = NeuralField(grid, width, nf.layers).to(device)
    net = torch.compile(model) if nf.compile else model
    opt = torch.optim.AdamW(model.parameters(), nf.lr, weight_decay=nf.weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, nf.epochs, 1e-12)
    shares = np.linspace(nf.subsample[0], nf.subsample[1], nf.epochs)
    for epoch, share in enumerate(shares):
        n_batches = max(1, int(n_points * share) // nf.batch)
        take = min(n_batches * nf.batch, n_points)
        order = torch.randperm(n_points, device=device)[:take].view(n_batches, -1)
        total = torch.zeros((), device=device)
        for index in order:
            loss = F.mse_loss(net(index), target[index])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.detach()
        schedule.step()
        log(f"    epoch {epoch + 1}/{nf.epochs}  mse {float(total) / n_batches:.4e}")
    return model.eval()


@torch.no_grad()
def evaluate(model: NeuralField, n_points: int, chunk: int = 1 << 20) -> torch.Tensor:
    device = next(model.parameters()).device
    out = torch.empty((n_points, 2), device=device)
    for a in range(0, n_points, chunk):
        b = min(a + chunk, n_points)
        out[a:b] = model(torch.arange(a, b, device=device))
    return out


def neural_field(
    backend, real_fields: List[torch.Tensor], budget_reals: int, cfg, log=print
):
    """Returns (H_hat, achieved reals per species, {"width", "parameters"})."""
    nf = cfg.baselines.nf
    torch.set_float32_matmul_precision("high")
    fields, achieved, extra = [], [], {"width": [], "parameters": []}
    for s, f in enumerate(real_fields):
        grid = tuple(int(n) for n in f.shape[1:])
        if nf.width is None:
            width, n_params = solve_width(grid, budget_reals, nf.layers)
        else:
            width, n_params = nf.width, parameter_count(grid, nf.width, nf.layers)
        # z-score per channel, over the whole grid or separately for every index of one axis
        over = [d for d in range(1, f.dim()) if d - 1 != nf.zscore_axis]
        mean = f.mean(over, keepdim=True, dtype=torch.float64)
        std = (
            (f.double() - mean).pow(2).mean(over, keepdim=True).sqrt().clamp_min(1e-30)
        )
        target = ((f - mean.float()) / std.float()).reshape(2, -1).T.contiguous()
        log(f"  [neural field, species {s}] width {width}, {n_params} parameters")
        model = fit_field(target, grid, width, nf, cfg.seed + 1000 * s, log)
        rec = evaluate(model, target.shape[0]).T.reshape(f.shape)
        fields.append((rec * std.float() + mean.float()).contiguous())
        achieved.append(n_params + mean.numel() + std.numel())
        extra["width"].append(width)
        extra["parameters"].append(n_params)
        del model, target, rec
    with torch.no_grad():
        H_hat = backend.to_spectral(fields)
    return H_hat, achieved, extra
