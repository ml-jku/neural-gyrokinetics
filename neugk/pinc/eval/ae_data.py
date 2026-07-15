"""Autoencoder eval support: the trained AE checkpoints to score, and a builder that returns a
per-trajectory val dataset sharing the checkpoint's TRAINING normalization stats (so AE rows line
up with the NF / traditional / PIGS rows). Kept out of the drivers so the AE wiring lives in one place.
"""

from neugk.dataset.cyclone_diff import CycloneAEDataset
from neugk.dataset.backend import H5Backend, KvikIOBackend

# AE checkpoints to evaluate: label -> (checkpoint path, load_peft, vqvae)
AE_CKPTS = {
    "AE-PRETRAIN": (
        "/system/user/publicwork/galletti/pinc_revival/AE1k_PRE/best.pth",
        False,
        False,
    ),
    "PINC-AE": (
        "/system/user/publicwork/galletti/pinc_revival/20260606_215226_877/best.pth",
        True,
        False,
    ),
    "PINC-AE-LAST": (
        "/system/user/publicwork/galletti/pinc_revival/20260606_215226_877/ckp.pth",
        True,
        False,
    ),
}


def _ns_get(ns, key, default=None):
    """getattr on the SimpleNamespace config that load_autoencoder returns."""
    return getattr(ns, key, default)


def build_make_val_dataset(config, path, backend):
    """Return make_val_dataset(traj)->CycloneAEDataset sharing the checkpoint's TRAINING
    normalization stats.

    Mirrors neugk/dataset/__init__.py get_data() `workflow == "pinc"` branch: conditioning is the
    union of encoder/decoder conditioning; the val set reuses the train set's `.stats` (loaded from
    the on-disk agg cache, not recomputed per call).
    """
    ds = config.dataset
    model = config.model

    enc_cond = list(_ns_get(model, "encoder_conditioning", []) or [])
    dec_cond = list(_ns_get(model, "decoder_conditioning", []) or [])
    conditioning = sorted(set(enc_cond) | set(dec_cond))

    # normalization is a SimpleNamespace; CycloneDataset wants a plain dict. reconstruct it.
    def ns_to_dict(o):
        if hasattr(o, "__dict__"):
            return {k: ns_to_dict(v) for k, v in vars(o).items()}
        if isinstance(o, list):
            return [ns_to_dict(v) for v in o]
        return o

    normalization = ns_to_dict(ds.normalization)

    # load the SAME fields as training (df,phi,flux) so the dataset-scope stats cache key matches the
    # on-disk agg cache (skip the multi-minute recompute). reconstruct only reads sample.df.
    common = dict(
        active_keys=list(ds.active_keys),
        fields_to_load=["df", "phi", "flux"],
        probe_targets=[],
        path=path,
        normalization=normalization,
        normalization_scope=ds.normalization_scope,
        spatial_ifft=ds.spatial_ifft,
        bundle_seq_length=_ns_get(model, "bundle_seq_length", 1),
        log_transform=ds.log_transform,
        split_into_bands=_ns_get(ds, "split_into_bands", None),
        minmax_beta1=ds.minmax_beta1,
        minmax_beta2=ds.minmax_beta2,
        separate_zf=ds.separate_zf,
        real_potens=ds.real_potens,
        decouple_mu=ds.norm_decouple_mu,
        num_workers=0,
        conditions=conditioning,
    )

    def make_backend():
        if backend == "h5":
            return H5Backend(0)
        # val without gds (get_data does the same: KvikIOBackend(rank, use_kvikio=False))
        return KvikIOBackend(0, use_kvikio=False)

    # dataset-scope stats: prefer the cached agg-stats pkl directly (the 235-traj recompute is a
    # multi-hour job on network mounts). fall back to building the train set only if no cache.
    import os
    import glob as _glob
    import pickle as _pickle
    import numpy as _np

    _cache = sorted(_glob.glob(os.path.join(path, "*df01345_phi012_agg_stats.pkl")))
    if _cache:
        _rms = _pickle.load(open(_cache[-1], "rb"))  # {field: RunningMeanStd}
        train_stats = {
            fld: {"full": {"mean": r.mean, "std": _np.sqrt(r.var), "min": r.min, "max": r.max}}
            for fld, r in _rms.items()
        }
    else:
        train = CycloneAEDataset(
            backend=make_backend(),
            split="train",
            trajectories=ds.training_trajectories,
            cond_filters=ns_to_dict(_ns_get(ds, "training_cond_filters", None)),
            subsample=ds.subsample,
            offset=_ns_get(ds, "offset", 0),
            timestep_std_filter=_ns_get(ds, "timestep_std_filter", None),
            **common,
        )
        train_stats = train.stats
        del train

    def make_val_dataset(traj):
        # one-trajectory val set; subsample=1/offset=0 so flat index == absolute timestep.
        return CycloneAEDataset(
            backend=make_backend(),
            split="val",
            trajectories=[traj],
            normalization_stats=train_stats,
            cond_filters=None,
            subsample=1,
            offset=0,
            timestep_std_filter=None,
            **common,
        )

    return make_val_dataset
