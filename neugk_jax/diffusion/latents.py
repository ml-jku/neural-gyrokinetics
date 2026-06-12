"""Precompute and cache AE latents for the diffusion training mode.

After training the AE (M4), the diffusion training (M5) operates on
the AE's bottleneck latents instead of the raw distribution functions.
We encode every sample once, cache the result, and the dataset serves
the latents in mode='diff' instead of doing a costly forward through
the encoder on each step.
"""

from __future__ import annotations

import hashlib
import os
import pickle
from pathlib import Path
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np

from neugk_jax.utils import RunningMeanStd


def _tqdm(*args, **kwargs):
    try:
        from tqdm import tqdm
        return tqdm(*args, **kwargs)
    except ImportError:
        return args[0] if args else iter(())


def _cache_path(dataset, split: str, ae_tag: str) -> Path:
    """Deterministic name for a precomputed-latents pickle on disk."""
    basenames = sorted(os.path.basename(f) for f in dataset.files)
    h = hashlib.sha256("".join(basenames).encode()).hexdigest()[:12]
    name = f"diff_{split}_latents_offset{dataset.offset}_{h}_{ae_tag}.pkl"
    return Path(dataset.path) / name


def precompute_latents(
    dataset,
    *,
    encode_fn: Callable,
    ae_tag: str,
    batch_size: int = 4,
    device=None,
    overwrite: bool = False,
) -> None:
    """Encode the entire dataset through ``encode_fn`` and cache the result.

    ``encode_fn`` is called as ``encode_fn(df_batch, cond_batch) -> latent_batch``
    where ``df_batch`` has shape ``(B, C, *resolution)`` and ``latent_batch``
    has shape ``(B, *latent_grid, latent_channels)``. The dataset is mutated
    in place: ``dataset.precomputed_latents`` is populated and the mode is
    flipped to ``"diff"`` so subsequent ``__getitem__`` calls return cached
    latents instead of raw df reads.
    """
    cache_file = _cache_path(dataset, dataset.split, ae_tag)
    if cache_file.exists() and not overwrite:
        with open(cache_file, "rb") as f:
            dataset.precomputed_latents = pickle.load(f)
        dataset.mode = "diff"
        _compute_latent_stats(dataset)
        return

    latents_dict: dict[tuple[int, int], dict] = {}
    n = len(dataset)
    indices = list(range(n))
    pbar = _tqdm(range(0, n, batch_size), desc=f"precompute {dataset.split} latents")
    for start in pbar:
        sl = indices[start : start + batch_size]
        # gather a batch from the (currently mode='ae') dataset
        samples = [dataset[i] for i in sl]
        df_batch = jnp.stack([jnp.asarray(s.df) for s in samples])
        cond_batch = None
        if samples[0].conditioning is not None:
            cond_batch = jnp.stack([jnp.asarray(s.conditioning) for s in samples])
        z = encode_fn(df_batch, cond_batch)
        z_np = np.asarray(z)
        for i, s in zip(sl, samples):
            fid = int(s.file_index)
            t_idx = int(s.timestep_index)
            latents_dict[(fid, t_idx)] = {
                "x": z_np[sl.index(i)],
                "flux": np.asarray(s.flux),
                "timestep": np.asarray(s.timestep),
                "phi": np.asarray(s.phi) if s.phi is not None else None,
                "conditioning": np.asarray(s.conditioning) if s.conditioning is not None else None,
            }

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "wb") as f:
        pickle.dump(latents_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    dataset.precomputed_latents = latents_dict
    dataset.mode = "diff"
    _compute_latent_stats(dataset)


def load_precomputed_latents(dataset, pickle_path: str | Path) -> None:
    """Populate ``dataset.precomputed_latents`` from a pre-existing pickle.

    Bypasses the ``encode_fn`` path in :func:`precompute_latents` when the
    cache has been computed externally (e.g. by the upstream torch
    pipeline). The pickle is expected to be ``dict[(file_idx, t_idx),
    dict]`` with the same per-entry schema written by
    :func:`precompute_latents` — at minimum ``x``, ``flux``, ``timestep``;
    optionally ``phi`` and the scalar conditioning fields.
    """
    p = Path(pickle_path)
    if not p.exists():
        raise FileNotFoundError(p)
    with open(p, "rb") as f:
        dataset.precomputed_latents = pickle.load(f)
    dataset.mode = "diff"
    _compute_latent_stats(dataset)


def _compute_latent_stats(dataset) -> None:
    """Running stats over all cached latents — used to scale by 1/std at train time."""
    stats = None
    for s in dataset.precomputed_latents.values():
        x = s["x"]
        mean = np.mean(x, keepdims=True)
        var = np.var(x, keepdims=True)
        mn = np.min(x, keepdims=True)
        mx = np.max(x, keepdims=True)
        if stats is None:
            stats = RunningMeanStd(shape=mean.shape)
        stats.update(mean, var, mn, mx, count=1)
    dataset.latent_stats = stats
