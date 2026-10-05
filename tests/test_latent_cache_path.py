"""Latent cache naming, sidecar provenance checks and AE dataset consistency."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from neugk_jax.diffusion.latents import latent_cache_path


def test_cache_path_dir_and_file_agree(tmp_path):
    run = tmp_path / "20260405_022851_327"
    run.mkdir()
    (run / "best.pth").write_bytes(b"")
    ds = SimpleNamespace(
        files=["/d/iteration_1_ifft_realpotens", "/d/iteration_0_ifft_realpotens"],
        offset=80,
        path=str(tmp_path),
    )
    by_dir = latent_cache_path(ds, "train", run, decouple_mu=True)
    by_file = latent_cache_path(ds, "train", run / "best.pth", decouple_mu=True)
    assert by_dir == by_file and by_dir.parent == tmp_path
    assert by_dir.name.startswith("diff_train_latents_offset80_mu_")
    assert by_dir.name.endswith("_latents_ae327.pkl")


def _tiny_dataset(root):
    from helpers import make_traj

    from neugk_jax.dataset import CycloneDataset, NumpyBackend

    make_traj(root, "iteration_0", n_t=4, resolution=(4, 4, 4, 16, 8))
    return CycloneDataset(path=str(root), trajectories="iteration_0", backend=NumpyBackend())


def _encode(df, _cond):
    import jax.numpy as jnp

    return jnp.zeros((df.shape[0], 2, 3)) + df.mean()


def test_cache_sidecar_checks_ae_and_preprocessing(tmp_path):
    import pytest

    from neugk_jax.diffusion.latents import (
        cache_meta_path,
        latent_cache_meta,
        load_precomputed_latents,
        precompute_latents,
    )

    ds = _tiny_dataset(tmp_path)
    ae = tmp_path / "ae.eqx"
    ae.write_bytes(b"weights")
    meta = latent_cache_meta(ds, ae)
    cache = tmp_path / "latents.pkl"
    precompute_latents(
        ds, encode_fn=_encode, cache_file=cache, batch_size=2, meta=meta, latent_shape=(2, 3)
    )
    assert ds.mode == "diff" and cache_meta_path(cache).exists()
    # a matching sidecar loads; a different ae or offset raises
    load_precomputed_latents(_tiny_dataset(tmp_path), cache, latent_shape=(2, 3), meta=meta)
    ae.write_bytes(b"retrained weights")
    with pytest.raises(ValueError, match="ae_size"):
        precompute_latents(
            _tiny_dataset(tmp_path),
            encode_fn=_encode,
            cache_file=cache,
            meta=latent_cache_meta(ds, ae),
            latent_shape=(2, 3),
        )
    with pytest.raises(ValueError, match="offset"):
        load_precomputed_latents(_tiny_dataset(tmp_path), cache, meta={**meta, "offset": 80})
    # an existing cache is always verified against the model latent shape
    with pytest.raises(ValueError, match="latent cache shape"):
        load_precomputed_latents(_tiny_dataset(tmp_path), cache, latent_shape=(4, 3), meta=meta)


def test_cache_without_sidecar_warns_and_verifies(tmp_path):
    import pickle

    import pytest

    from neugk_jax.diffusion.latents import load_precomputed_latents

    ds = _tiny_dataset(tmp_path)
    cache = {
        k: {"x": np.zeros((2, 3), np.float32), "flux": np.float32(0), "timestep": np.float32(0)}
        for k in ds.flat_index_to_file_and_tstep.values()
    }
    path = tmp_path / "torch_cache.pkl"
    path.write_bytes(pickle.dumps(cache))
    with pytest.warns(UserWarning, match="meta.json"):
        load_precomputed_latents(ds, path, latent_shape=(2, 3), meta={"offset": 0})
    del cache[(0, 0)]
    path.write_bytes(pickle.dumps(cache))
    with pytest.warns(UserWarning), pytest.raises(ValueError):
        load_precomputed_latents(_tiny_dataset(tmp_path), path, meta={"offset": 0})


def test_ae_dataset_mismatch_raises():
    import pytest

    from neugk_jax.diffusion.runner import check_ae_dataset

    norm = {"df": {"type": "zscore", "agg_axes": [0, 1, 3, 4, 5]}}
    ae = {
        "separate_zf": True,
        "offset": 80,
        "normalization": {**norm, "phi": {"type": "zscore"}},
        "normalization_stats": "/a/../a/stats.pkl",
    }
    ok = {
        "separate_zf": True,
        "offset": 80,
        "normalization": norm,
        "normalization_stats": "/a/stats.pkl",
    }
    check_ae_dataset(ae, ok)
    for key, value in (
        ("offset", 0),
        ("separate_zf", False),
        ("normalization", {"df": {"type": "minmax"}}),
        ("normalization_stats", "/b/stats.pkl"),
    ):
        with pytest.raises(ValueError, match=key):
            check_ae_dataset(ae, {**ok, key: value})
    with pytest.warns(UserWarning, match="normalization_stats"):
        check_ae_dataset({**ae, "normalization_stats": None}, ok)
