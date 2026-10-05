"""VQ-VAE: quantizers (assignment, straight-through, EMA, dead codes), model, runner, code cache."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from helpers import COND, RES, make_traj, tiny_ae_cfg, tiny_vq_cfg
from omegaconf import OmegaConf

from neugk_jax.models.utils import trainable_mask
from neugk_jax.pinc.quantizers import FSQ, LFQ, VectorQuantizer


def _vq(cosine=False, k=16, d=4, thr=0.0):
    kw = dict(decay=0.9, threshold_ema_dead_code=thr, use_cosine_sim=cosine)
    vq = VectorQuantizer(d, k, key=jr.PRNGKey(0), **kw)
    embed = jr.normal(jr.PRNGKey(1), (k, d))
    embed = embed / jnp.linalg.norm(embed, axis=-1, keepdims=True) if cosine else embed
    return eqx.tree_at(lambda q: (q.embed, q.embed_avg), vq, (embed, embed))


@pytest.mark.parametrize("cosine", [False, True])
def test_nearest_code_assignment(cosine):
    vq = _vq(cosine)
    z = jr.normal(jr.PRNGKey(3), (64, 4))
    q, idx, aux = vq(z)
    e, x = np.asarray(vq.embed), np.asarray(z)
    if cosine:
        ref = np.argmax(x / np.linalg.norm(x, axis=-1, keepdims=True) @ e.T, axis=-1)
    else:
        ref = np.argmin(((x[:, None] - e[None]) ** 2).sum(-1), axis=-1)
    np.testing.assert_array_equal(idx, ref)
    np.testing.assert_allclose(q, e[ref], rtol=1e-6)
    assert aux == {} and not any(jax.tree_util.tree_leaves(trainable_mask(vq)))


def test_straight_through_and_fsq_codes():
    for qz in (_vq(), LFQ(16)):
        z = jr.normal(jr.PRNGKey(4), (8, qz.dim))
        c = jr.normal(jr.PRNGKey(5), z.shape)
        g = jax.grad(lambda z_: jnp.sum(qz(z_, inference=False)[0] * c))(z)
        np.testing.assert_allclose(g, c, rtol=1e-6)
    fsq = FSQ([8, 5, 5, 3])
    z = 3 * jr.normal(jr.PRNGKey(6), (32, 4))
    q, idx, _ = fsq(z, inference=False)
    np.testing.assert_allclose(fsq.codes(idx), q, atol=1e-6)
    assert int(idx.min()) >= 0 and int(idx.max()) < fsq.codebook_size == 600
    g = jax.grad(lambda z_: jnp.sum(fsq(z_, inference=False)[0]))(z)
    hw = np.asarray([4, 2, 2, 1], np.float32)
    np.testing.assert_allclose(g, jax.grad(lambda z_: jnp.sum(fsq.bound(z_) / hw))(z), rtol=1e-5)


@pytest.mark.parametrize("m", [40, 400])
def test_ema_update_and_dead_code_reseeding(m):
    vq = _vq(k=64, d=4, thr=2.0)
    vq = eqx.tree_at(lambda q: q.cluster_size, vq, jnp.full((64,), 2.05))
    flat = jr.normal(jr.PRNGKey(9), (m, 4))
    idx = np.asarray(vq(flat)[1])
    new, n = vq.ema_update(flat, idx, jr.PRNGKey(1))
    x = np.asarray(flat, np.float64)
    esum = np.zeros((64, 4))
    np.add.at(esum, idx, x)
    cs = 2.05 + 0.1 * (np.bincount(idx, minlength=64) - 2.05)
    ea = np.asarray(vq.embed_avg) + 0.1 * (esum - np.asarray(vq.embed_avg))
    embed = ea / ((cs + vq.eps) / (cs.sum() + 64 * vq.eps) * cs.sum())[:, None]
    expired = cs < 2.0
    assert int(n) == int(expired.sum()) > 0 and not expired.all()
    e_new = np.asarray(new.embed)
    np.testing.assert_allclose(new.cluster_size, np.where(expired, 2.0, cs), rtol=1e-5)
    np.testing.assert_allclose(e_new[~expired], embed[~expired], rtol=1e-4, atol=1e-6)
    hit = [np.flatnonzero(np.isclose(x, r, atol=1e-6).all(axis=1)) for r in e_new[expired]]
    assert all(len(h) for h in hit)
    if m >= 64:
        assert len({int(h[0]) for h in hit}) == len(hit)
    np.testing.assert_allclose(np.asarray(new.embed_avg)[expired], 2.0 * e_new[expired])


@pytest.mark.parametrize("quantizer,enc", [("vq", []), ("fsq", ["q", "itg"]), ("lfq", [])])
def test_vqvae_forward_and_decode_from_indices(quantizer, enc):
    from neugk_jax.models.build import build_ae_from_config

    cfg = {
        "model": tiny_vq_cfg(quantizer, encoder_conditioning=enc),
        "dataset": {"resolution": RES},
    }
    model = build_ae_from_config(cfg, key=jr.PRNGKey(0))
    assert (model.enc_cond_embed is None) == (not enc)
    x = jr.normal(jr.PRNGKey(1), (2, *RES))
    out = model(x, COND)
    assert out["df"].shape == x.shape and out["vq_indices"].shape == model.bottleneck_grid_size
    again = model.decode_from_indices(out["vq_indices"], COND)["df"]
    np.testing.assert_allclose(again, out["df"], rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(model.encode_indices(x, COND), out["vq_indices"])
    assert not np.allclose(model(x, COND.at[0].set(-3.0))["df"], out["df"])


def test_vq_translation_names():
    from neugk_jax.translate import _ae_name_map

    assert "vq._codebook.embed" in _ae_name_map("vq.embed")
    assert "vq._codebook.cluster_size" in _ae_name_map("vq.cluster_size")
    assert "middle_vq_downproj.weight" in _ae_name_map("middle_downproj.inner.weight")


@pytest.fixture
def vq_cfg(tmp_path):
    make_traj(tmp_path, "iteration_0", n_t=8)
    make_traj(tmp_path, "iteration_1", n_t=4)
    cfg = tiny_ae_cfg(tmp_path)
    cfg.workflow = "vqvae"
    cfg.training.batch_size = 2
    return cfg


@pytest.mark.parametrize("quantizer", ["vq", "fsq"])
def test_vqvae_runner_step_and_eval(vq_cfg, quantizer):
    from neugk_jax.pinc.runner import VQVAERunner
    from neugk_jax.training.ddp import local_view, shard_batch
    from neugk_jax.training.runner import train_step

    vq_cfg.model = OmegaConf.create(tiny_vq_cfg(quantizer, encoder_conditioning=["q"]))
    r = VQVAERunner(vq_cfg, output_path=vq_cfg.output_path)
    batch = r.load_batch(r.train_ds, [0, 1] * r.dist.local_device_count, r.loader.read)
    assert batch["conditioning"].shape[-1] == 4
    before = jax.device_get(local_view(r.dist, r.model).vq)
    inputs = (shard_batch(r.dist, batch), r.ctx, jr.PRNGKey(0))
    r.model, r.opt_state, logs = train_step(inputs, r.model, r.opt_state, r.spec)
    assert {"total", "df", "vq_commit", "codebook_usage"} <= set(logs) and "state" not in logs
    assert np.isfinite(float(logs["total"]))
    if quantizer == "vq":
        vq = local_view(r.dist, r.model).vq
        assert float(logs["dead_codes"]) > 0 and float(jnp.sum(vq.cluster_size)) > 0
        assert not np.allclose(vq.embed, before.embed)
    metrics, _ = r.evaluate(1)
    assert {"df_mse", "df_rel_l2", "codebook_usage", "codebook_perplexity"} <= set(metrics)
    assert 0 < metrics["codebook_usage"] <= 1 and metrics["codebook_perplexity"] >= 1
    vq_cfg.model.loss_weights.flux_int = 1.0
    with pytest.raises(ValueError, match="flux_int"):
        VQVAERunner(vq_cfg, output_path=vq_cfg.output_path)


def test_precompute_vq_indices(vq_cfg, tmp_path):
    from neugk_jax.dataset.factory import build_dataset
    from neugk_jax.diffusion.latents import latent_arrays, latent_cache_path
    from neugk_jax.models.build import build_ae_from_config
    from neugk_jax.pinc.vqvae import precompute_vq_indices
    from neugk_jax.training.runner import conditioning_slots

    ds = build_dataset(vq_cfg.dataset, split="train", mode="ae")
    ae = build_ae_from_config(
        {"model": tiny_vq_cfg(), "dataset": {"resolution": RES}}, key=jr.PRNGKey(0)
    )
    (tmp_path / "run_123").mkdir()
    cache = latent_cache_path(ds, "train", tmp_path / "run_123", kind="indices")
    assert cache.name.startswith("diff_train_indices_") and cache.name.endswith("_vqvae123.pkl")
    slots = conditioning_slots(ds.conditions, ae.condition_keys)
    precompute_vq_indices(ds, ae, cache, cond_slots=slots, batch_size=3)
    z, _ = latent_arrays(ds)
    assert ds.mode == "diff" and z.dtype == np.int64
    assert z.shape == (len(ds), int(np.prod(ae.bottleneck_grid_size)))
    s = ds.with_mode("ae")[2]
    ref = ae.encode_indices(jnp.asarray(s.df), jnp.asarray(s.conditioning)[slots])
    np.testing.assert_array_equal(z[2], np.asarray(ref).reshape(-1))
