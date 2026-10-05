"""``CycloneDataset`` construction from a ``dataset`` config section."""

from __future__ import annotations

from typing import Any, Optional, Sequence

from neugk_jax.dataset.backend import make_backend
from neugk_jax.dataset.cyclone import DEFAULT_CONDITIONS, CycloneDataset
from neugk_jax.utils import to_dict


def build_dataset(
    dcfg,
    *,
    split: str,
    dist=None,
    mode: str = "ae",
    fields: Optional[Sequence[str]] = None,
    conditions: Optional[Sequence[str]] = None,
    prefer_dtype: Optional[str] = None,
    stats: Optional[dict] = None,
    **overrides: Any,
) -> CycloneDataset:
    """Dataset for ``split`` ("train" or "val") from the ``dataset`` config.

    ``stats`` (an already loaded ``CycloneDataset.stats``) replaces
    ``normalization_stats`` so the pickle is read once per run.
    ``lightweight_metadata`` (read ``metadata_light``, without the per-trajectory df
    moments) defaults on when normalization stats are given.
    """
    norm_stats = dcfg.get("normalization_stats")
    if norm_stats is not None and not isinstance(norm_stats, str):
        norm_stats = to_dict(norm_stats)
    normalization = to_dict(dcfg.get("normalization")) or None
    lightweight = bool(dcfg.get("lightweight_metadata", norm_stats is not None))
    if lightweight and normalization is not None and norm_stats is None:
        raise ValueError(
            "lightweight_metadata drops the per-trajectory df moments; set "
            "normalization_stats or disable it"
        )
    trajectories = dcfg.training_trajectories if split == "train" else dcfg.validation_trajectories
    filters = dcfg.get("training_cond_filters" if split == "train" else "eval_cond_filters")
    kwargs = dict(
        path=dcfg.path,
        split=split,
        trajectories=trajectories if isinstance(trajectories, str) else list(trajectories),
        fields_to_load=tuple(fields or dcfg.get("input_fields", ("df",))),
        conditions=tuple(
            conditions if conditions is not None else dcfg.get("conditions", DEFAULT_CONDITIONS)
        ),
        mode=mode,
        separate_zf=bool(dcfg.get("separate_zf", False)),
        normalization=normalization,
        normalization_scope=dcfg.get("normalization_scope", "dataset"),
        normalization_stats=stats if stats is not None else norm_stats,
        offset=int(dcfg.get("offset", 0)),
        subsample=int(
            dcfg.get("subsample", 1) if split == "train" else dcfg.get("val_subsample", 1)
        ),
        cond_filters=to_dict(filters) or None,
        backend=make_backend(
            dcfg,
            local_rank=dist.local_rank if dist else 0,
            prefer_dtype=prefer_dtype,
            lightweight_metadata=lightweight,
        ),
        rank=dist.process_id if dist else 0,
    )
    kwargs.update(overrides)
    return CycloneDataset(**kwargs)


def build_splits(
    dcfg, *, dist=None, train_dtype: Optional[str] = None, val_overrides=None, **kwargs
) -> tuple[CycloneDataset, CycloneDataset]:
    """Train and val datasets sharing one normalization-stats load."""
    train = build_dataset(dcfg, split="train", dist=dist, prefer_dtype=train_dtype, **kwargs)
    stats = train.stats if dcfg.get("normalization_stats") is not None else None
    val = build_dataset(
        dcfg, split="val", dist=dist, stats=stats, **{**kwargs, **(val_overrides or {})}
    )
    return train, val
