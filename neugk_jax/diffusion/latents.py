"""Precompute and cache AE latents for the diffusion training mode.

Diffusion training operates on the AE's bottleneck latents instead of
the raw distribution functions. Encodes every sample once, caches the
result, and the dataset serves the latents in mode='diff' instead of
running the encoder on each step.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import warnings
from pathlib import Path
from typing import Callable, Optional

import jax.numpy as jnp
import numpy as np

from neugk_jax.dataset.cyclone import COND_META_KEYS
from neugk_jax.training.ddp import barrier
from neugk_jax.utils import RunningStats, atomic_write, progress


def latent_cache_path(
    dataset, split: str, ae_checkpoint: str, *, decouple_mu: bool = False, kind: str = "latents"
) -> Path:
    """Cache file for a split's latents (``kind="latents"``) or VQ code indices (``"indices"``).

    ``<path>/diff_<split>_<kind>_offset<o>[_mu]_<sha256(sorted basenames)[:12]>_<kind>_<tag><run>.pkl``
    where ``<tag>`` is ``ae`` for latents and ``vqvae`` for indices and ``<run>`` is the last ``_``
    field of the AE run directory (a checkpoint file resolves to its directory).
    """
    if kind not in ("latents", "indices"):
        raise ValueError(f"unknown cache kind {kind!r}")
    basenames = sorted(os.path.basename(f) for f in dataset.files)
    file_hash = hashlib.sha256("".join(basenames).encode()).hexdigest()[:12]
    run_dir = os.path.abspath(str(ae_checkpoint))
    if os.path.isfile(run_dir):
        run_dir = os.path.dirname(run_dir)
    segments = [
        "diff",
        f"{split}_{kind}",
        f"offset{dataset.offset}",
        "mu" if decouple_mu else "",
        file_hash,
        kind,
        ("ae" if kind == "latents" else "vqvae") + run_dir.split("_")[-1],
    ]
    return Path(dataset.path) / ("_".join(filter(None, segments)) + ".pkl")


def cache_meta_path(cache_file: str | Path) -> Path:
    cache_file = Path(cache_file)
    return cache_file.with_name(cache_file.name + ".meta.json")


def latent_cache_meta(
    dataset, ae_checkpoint: str | os.PathLike, *, normalization_stats=None
) -> dict:
    """Provenance of a latent cache: AE checkpoint file identity and the dataset preprocessing."""
    ae = Path(ae_checkpoint).resolve()
    st = ae.stat()
    stats = normalization_stats if isinstance(normalization_stats, (str, os.PathLike)) else None
    return {
        "ae_checkpoint": str(ae),
        "ae_size": int(st.st_size),
        "ae_mtime": float(st.st_mtime),
        "normalization_stats": None if stats is None else str(Path(stats).resolve()),
        "separate_zf": bool(dataset.separate_zf),
        "offset": int(dataset.offset),
    }


def write_cache_meta(cache_file: str | Path, meta: dict) -> None:
    text = json.dumps(meta, indent=2, sort_keys=True)
    atomic_write(cache_meta_path(cache_file), lambda f: f.write(text), mode="w")


def check_cache_meta(cache_file: str | Path, meta: Optional[dict]) -> None:
    """Raise if the sidecar of ``cache_file`` disagrees with ``meta``; warn if it has none."""
    if meta is None:
        return
    path = cache_meta_path(cache_file)
    if not path.exists():
        warnings.warn(
            f"{cache_file} has no {path.name}; its AE and preprocessing cannot be "
            "checked, only its index and latent shape"
        )
        return
    saved = json.loads(path.read_text())
    diff = {k: (saved.get(k), v) for k, v in meta.items() if saved.get(k) != v}
    if diff:
        lines = "\n".join(f"  {k}: cache={a!r} run={b!r}" for k, (a, b) in sorted(diff.items()))
        raise ValueError(
            f"latent cache {cache_file} was built for a different AE or "
            f"preprocessing:\n{lines}\ndelete it to re-encode, or point "
            "dataset.latents_cache_* at a matching cache"
        )


def precompute_latents(
    dataset,
    *,
    encode_fn: Callable,
    cache_file: str | Path,
    batch_size: int = 4,
    meta: Optional[dict] = None,
    latent_shape=None,
) -> None:
    """Encode the dataset through ``encode_fn`` into ``cache_file`` (or load it if present).

    ``encode_fn(df_batch, cond_batch) -> latent_batch`` maps ``(B, C, *resolution)`` to
    ``(B, *latent_grid, latent_channels)``, or to integer code grids stored flattened as int64.
    Entries:
    ``{(fid, t_idx): {"x", "phi", "flux", "timestep", <one raw scalar per condition>}}``.
    Process 0 encodes and writes atomically, with ``meta`` (see :func:`latent_cache_meta`)
    as a ``<cache>.meta.json`` sidecar; the others wait. Every process then loads the cache
    through :func:`load_precomputed_latents`, which checks it against ``meta`` and this
    dataset, and the dataset is switched to ``mode="diff"``.
    """
    import jax

    cache_file = Path(cache_file)
    if not cache_file.exists() and jax.process_index() == 0:
        latents_dict = _encode_all(dataset, encode_fn, batch_size)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        if meta is not None:
            write_cache_meta(cache_file, meta)
        atomic_write(
            cache_file, lambda f: pickle.dump(latents_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        )
    barrier(f"latents:{cache_file.name}")
    load_precomputed_latents(dataset, cache_file, latent_shape=latent_shape, meta=meta)


def _encode_all(dataset, encode_fn: Callable, batch_size: int) -> dict:
    from neugk_jax.training.data import eval_plans, stack_fields

    latents_dict: dict[tuple[int, int], dict] = {}
    # the tail batch repeats its last sample so the encoder sees one batch shape
    plans = eval_plans(None, range(len(dataset)), batch_size)
    for plan in progress(plans, True, desc=f"precompute {dataset.split} latents"):
        samples = [dataset[int(i)] for i in plan.indices]
        batch = {
            k: jnp.asarray(v) for k, v in stack_fields(samples, ("df", "conditioning")).items()
        }
        z_np = np.asarray(encode_fn(batch["df"], batch.get("conditioning")))
        if z_np.dtype.kind in "iu":
            z_np = z_np.reshape(len(samples), -1).astype(np.int64)
        for b, s in enumerate(samples[: int(plan.mask.sum())]):
            entry = {
                "x": z_np[b],
                "phi": np.asarray(s.phi) if s.phi is not None else None,
                "flux": np.asarray(s.flux, dtype=np.float32),
                "timestep": np.asarray(s.timestep, dtype=np.float32),
            }
            if s.conditioning is not None:
                cond = np.asarray(s.conditioning, dtype=np.float32)
                for k, name in enumerate(dataset.conditions):
                    entry[name] = cond[k]
            latents_dict[(int(s.file_index), int(s.timestep_index))] = entry
    return latents_dict


def latent_arrays(dataset) -> tuple[np.ndarray, np.ndarray | None]:
    """Latents ``(N, *latent_shape)`` and conditioning ``(N, n_cond)`` in flat-index order."""
    keys = [dataset.flat_index_to_file_and_tstep[i] for i in range(len(dataset))]
    z = np.stack([np.asarray(dataset.precomputed_latents[k]["x"]) for k in keys])
    z = z if z.dtype.kind in "iu" else z.astype(np.float32)
    if not dataset.conditions:
        return z, None
    return z, np.stack([dataset.conditioning(f, dataset.get_timestep(f, t)) for f, t in keys])


def load_precomputed_latents(
    dataset, pickle_path: str | Path, *, latent_shape=None, meta: Optional[dict] = None
) -> None:
    """Populate ``dataset.precomputed_latents`` from a cache pickle and switch to ``mode="diff"``.

    The pickle is ``dict[(file_idx, t_idx), dict]`` with the per-entry schema written by
    :func:`precompute_latents` (at minimum ``x``, ``flux``, ``timestep``; optionally ``phi``
    and the scalar conditions). With ``meta`` the ``<cache>.meta.json`` sidecar must match
    (see :func:`check_cache_meta`). The cache is verified against this dataset: every
    indexed ``(fid, t_idx)`` must be present, the scalar conditions it carries must agree
    with the trajectory metadata and the latents must have ``latent_shape`` when given.
    A cache keyed by a different file ordering is re-keyed onto this dataset's fids
    (see :func:`remap_latent_cache`) instead of failing.
    """
    p = Path(pickle_path)
    if not p.exists():
        raise FileNotFoundError(p)
    check_cache_meta(p, meta)
    with open(p, "rb") as f:
        cache = pickle.load(f)
    try:
        verify_latent_cache(dataset, cache)
    except ValueError:
        cache = remap_latent_cache(dataset, cache)
    verify_latent_cache(dataset, cache, latent_shape=latent_shape)
    dataset.precomputed_latents = cache
    dataset.mode = "diff"
    _compute_latent_stats(dataset)


def remap_latent_cache(dataset, cache: dict, *, tol: float = 1e-3) -> dict:
    """Re-key a cache built with a different file ordering onto this dataset's fids.

    Trajectories are matched on their ``(itg, dg, s_hat, q)`` tuple, which must be
    unique on both sides; the per-entry ``timestep`` is then checked against the
    trajectory metadata so a wrong pairing cannot slip through.
    """

    def _tuple_of(get):
        return np.array([float(np.squeeze(get(k))) for k in COND_META_KEYS])

    ours = {}
    for fid, meta in dataset.metadata.items():
        key = tuple(np.round(_tuple_of(lambda k: meta[COND_META_KEYS[k]]), 6))
        if key in ours:
            raise ValueError(
                f"trajectories {ours[key]} and {fid} share conditions {key}; "
                "cannot remap the cache by conditions"
            )
        ours[key] = fid
    keys = np.array(list(ours))
    fids = list(ours.values())

    cache_fids = sorted({k[0] for k in cache})
    steps = {f: sorted(t for cf, t in cache if cf == f) for f in cache_fids}
    mapping: dict[int, int] = {}
    for cfid in cache_fids:
        entry = cache[(cfid, steps[cfid][0])]
        if not all(k in entry for k in COND_META_KEYS):
            raise ValueError("cache entries carry no scalar conditions; cannot remap")
        d = np.abs(keys - _tuple_of(lambda k: entry[k])).max(axis=1)
        j = int(np.argmin(d))
        if d[j] > tol:
            raise ValueError(
                f"cache trajectory {cfid} matches no dataset trajectory "
                f"(closest distance {d[j]:.3g})"
            )
        mapping[cfid] = fids[j]
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("cache -> dataset trajectory matching is not bijective")

    out = {}
    for (cfid, t), entry in cache.items():
        fid = mapping[cfid]
        if "timestep" in entry:
            want = float(dataset.metadata[fid]["timesteps"][t + dataset.offset])
            if not np.isclose(float(entry["timestep"]), want, rtol=1e-4):
                raise ValueError(
                    f"remapped cache entry ({cfid}->{fid}, t={t}) has timestep "
                    f"{float(entry['timestep'])} but the trajectory has {want}"
                )
        out[(fid, t)] = entry
    return out


def verify_latent_cache(dataset, cache: dict, *, latent_shape=None) -> None:
    """Raise unless ``cache`` matches this dataset's index and file ordering."""
    wanted = set(dataset.flat_index_to_file_and_tstep.values())
    missing = sorted(wanted - set(cache))
    if missing:
        raise ValueError(
            f"latent cache is missing {len(missing)} of {len(wanted)} (fid, t_idx) entries, "
            f"e.g. {missing[:5]} — the trajectory selection or offset does not match "
            "the cache (cache holds "
            f"{len({k[0] for k in cache})} trajectories x {len({k[1] for k in cache})} steps)"
        )
    any_key = next(iter(wanted))
    x = cache[any_key]["x"]
    if latent_shape is not None and tuple(x.shape) != tuple(latent_shape):
        raise ValueError(
            f"latent cache shape {tuple(x.shape)} != model latent {tuple(latent_shape)}"
        )
    # the scalar conditions pin the fid -> trajectory mapping
    bad = []
    for fid in sorted({k[0] for k in wanted}):
        entry = cache[(fid, sorted(t for f, t in wanted if f == fid)[0])]
        for short, meta_key in COND_META_KEYS.items():
            if short not in entry:
                continue
            want = float(np.squeeze(dataset.metadata[fid][meta_key]))
            got = float(np.squeeze(entry[short]))
            if not np.isclose(want, got, rtol=1e-4, atol=1e-6):
                bad.append((fid, short, want, got))
    if bad:
        raise ValueError(
            f"latent cache conditions disagree for {len(bad)} (fid, field) pairs, e.g. "
            f"{bad[:4]} — the cache was built with a different file ordering or filter set"
        )


def _compute_latent_stats(dataset) -> None:
    """Running stats over all cached latents — used to scale by 1/std at train time."""
    stats = RunningStats(prior_count=0.0)
    for s in dataset.precomputed_latents.values():
        x = s["x"]
        stats.merge(*(f(x, keepdims=True) for f in (np.mean, np.var, np.min, np.max)))
    dataset.latent_stats = stats
