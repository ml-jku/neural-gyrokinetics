"""VQ-VAE against the torch reference: quantizers, one EMA codebook step and a tiny random model."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
import torch
from torch_ref import RES, STUB_DS, torch_ae_cfg, torch_doubles_swin_shortcut

from neugk_jax.pinc.quantizers import FSQ, LFQ, VectorQuantizer


def _pair(cosine, thr, k=32, d=8, cluster_size=3.0):
    from neugk.pinc.autoencoders.vector_quantize import VectorQuantize

    torch.manual_seed(0)
    kw = dict(decay=0.9, commitment_weight=0.25, threshold_ema_dead_code=thr)
    t = VectorQuantize(dim=d, codebook_size=k, use_cosine_sim=cosine, **kw)
    cb = t._codebook
    e = torch.randn(1, k, d)
    with torch.no_grad():
        cb.embed.copy_(torch.nn.functional.normalize(e, dim=-1) if cosine else e)
        cb.embed_avg.copy_(cb.embed)
        cb.cluster_size.fill_(cluster_size)
    vq = VectorQuantizer(d, k, key=jr.PRNGKey(0), use_cosine_sim=cosine, **kw)
    bufs = tuple(jnp.asarray(b.numpy()[0]) for b in (cb.embed, cb.embed_avg, cb.cluster_size))
    return t.train(), eqx.tree_at(lambda q: (q.embed, q.embed_avg, q.cluster_size), vq, bufs)


@pytest.mark.parametrize("cosine", [False, True])
def test_ema_step_matches_torch(cosine):
    t, vq = _pair(cosine, thr=0)
    x = torch.randn(2, 50, 8)
    tq, tidx, tloss = t(x)
    jq, jidx, aux = vq(jnp.asarray(x.numpy().reshape(-1, 8)), inference=False)
    np.testing.assert_array_equal(jidx, tidx.numpy().reshape(-1))
    np.testing.assert_allclose(jq, tq.detach().numpy().reshape(-1, 8), rtol=1e-5, atol=1e-6)
    assert float(vq.batch_loss({"commit": aux["commit"][None]})) == pytest.approx(
        float(tloss.detach())
    )
    new, n = vq.ema_update(aux["z"], jidx, jr.PRNGKey(0))
    cb = t._codebook
    for j, ref in ((new.cluster_size, cb.cluster_size), (new.embed_avg, cb.embed_avg)):
        np.testing.assert_allclose(j, ref.numpy()[0], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(new.embed, cb.embed.numpy()[0], rtol=1e-5, atol=1e-6)
    assert int(n) == 0


@pytest.mark.parametrize("cosine", [False, True])
def test_dead_code_replacement_matches_torch(cosine):
    t, vq = _pair(cosine, thr=2, cluster_size=2.05)
    x = torch.randn(1, 40, 8)
    t(x)
    _, jidx, aux = vq(jnp.asarray(x.numpy()[0]), inference=False)
    new, n = vq.ema_update(aux["z"], jidx, jr.PRNGKey(0))
    cb = t._codebook
    np.testing.assert_allclose(new.cluster_size, cb.cluster_size.numpy()[0], rtol=1e-5)
    # ema cluster size 1.845 + 0.1 * hits: codes hit at most once expire
    expired = np.bincount(np.asarray(jidx), minlength=32) <= 1
    assert int(n) == int(expired.sum()) and 0 < int(n) < 32
    t_embed = cb.embed.numpy()[0]
    np.testing.assert_allclose(np.asarray(new.embed)[~expired], t_embed[~expired], atol=1e-6)
    # both re-seed the expired codes with (normalized) batch tokens
    rows = np.asarray(aux["z"])
    for emb in (np.asarray(new.embed)[expired], t_embed[expired]):
        assert all(np.isclose(rows, r, atol=1e-6).all(axis=1).any() for r in emb)


def test_fsq_and_lfq_match_torch():
    from neugk.pinc.autoencoders.vector_quantize import FSQ as TorchFSQ
    from neugk.pinc.autoencoders.vector_quantize import LFQ as TorchLFQ

    levels = [8, 8, 8, 5, 5, 5]
    x = 2 * torch.randn(300, 6)
    tq, tidx, _ = TorchFSQ(levels)(x)
    jq, jidx, _ = FSQ(levels)(jnp.asarray(x.numpy()))
    np.testing.assert_array_equal(jidx, tidx.numpy())
    np.testing.assert_allclose(jq, tq.numpy(), atol=1e-6)
    t_codes = TorchFSQ(levels).indices_to_codes(tidx).numpy()
    np.testing.assert_allclose(FSQ(levels).codes(jidx), t_codes, atol=1e-6)

    t = TorchLFQ(codebook_size=256, entropy_loss_weight=0.1, commitment_weight=0.25).train()
    x = 0.1 * torch.randn(2, 30, 8)
    _, tidx, tloss = t(x)
    lfq = LFQ(256, entropy_loss_weight=0.1, commitment_weight=0.25)
    out = [lfq(jnp.asarray(xi), inference=False) for xi in x.numpy()]
    np.testing.assert_array_equal(np.stack([o[1] for o in out]), tidx.numpy())
    aux = {k: jnp.stack([o[2][k] for o in out]) for k in out[0][2]}
    assert float(lfq.batch_loss(aux)) == pytest.approx(float(tloss), rel=1e-4)


@pytest.mark.parametrize("quantizer", ["vq", "fsq", "lfq"])
def test_tiny_vqvae_translates_and_matches_torch(quantizer):
    from neugk.pinc.autoencoders import get_autoencoder
    from omegaconf import OmegaConf

    from neugk_jax.models.build import build_ae_from_config
    from neugk_jax.translate import ae_state_template, export_ae_state, translate_ae

    vq = {"quantizer": quantizer, "codebook_size": 64, "embedding_dim": 8, "levels": [4, 3, 3]}
    cfg = torch_ae_cfg(name="vqvae", model_type="vqvae", vq=vq)
    torch.manual_seed(0)
    tmodel = get_autoencoder(OmegaConf.create(cfg), STUB_DS, rank=None).eval()
    state = {k: v.detach().numpy().copy() for k, v in tmodel.state_dict().items()}
    legacy = torch_doubles_swin_shortcut()
    jmodel = build_ae_from_config(cfg, key=jr.PRNGKey(0), legacy_double_shortcut=legacy)
    jmodel, missing, unused = translate_ae(jmodel, state, strict=True)
    assert not missing and not unused
    x = np.random.default_rng(0).standard_normal((4, *RES)).astype(np.float32)
    cond = np.asarray([1.6, 2.6, 8.5, 0.6], np.float32)
    with torch.no_grad():
        tout = tmodel(torch.from_numpy(x)[None], condition=torch.from_numpy(cond)[None])
    with jax.default_matmul_precision("highest"):
        jout = jmodel(jnp.asarray(x), jnp.asarray(cond))
    tdf = tout["df"][0].numpy()
    assert np.linalg.norm(np.asarray(jout["df"]) - tdf) <= 1e-5 * np.linalg.norm(tdf)
    np.testing.assert_array_equal(jout["vq_indices"], tout["vq_indices"][0].numpy())
    exported = export_ae_state(jmodel, ae_state_template(jmodel))
    tmodel.load_state_dict(
        {k: torch.from_numpy(v.copy()) for k, v in exported.items()}, strict=True
    )


@pytest.mark.parametrize("model_type", ["ae", "vqvae"])
def test_lora_strategy_targets_match_torch(model_type):
    from neugk.pinc.autoencoders import get_autoencoder
    from neugk.pinc.peft_utils import _STRATEGY_GROUPS, get_target_modules_for_lora
    from omegaconf import OmegaConf

    from neugk_jax.models.build import build_ae_from_config
    from neugk_jax.translate import _LORA_STRATEGIES, _ae_module_name, ae_lora_paths

    vq = {"quantizer": "vq", "codebook_size": 64, "embedding_dim": 8}
    extra = {"name": "vqvae", "model_type": "vqvae", "vq": vq} if model_type == "vqvae" else {}
    cfg = torch_ae_cfg(**extra)
    tmodel = get_autoencoder(OmegaConf.create(cfg), STUB_DS, rank=None)
    jmodel = build_ae_from_config(cfg, key=jr.PRNGKey(0))
    assert set(_LORA_STRATEGIES) == set(_STRATEGY_GROUPS)
    for strategy in _STRATEGY_GROUPS:
        paths = ae_lora_paths(jmodel, {"strategy": strategy})
        names = {_ae_module_name(p, vq=model_type == "vqvae") for p in paths}
        assert names == set(get_target_modules_for_lora(tmodel, strategy)), strategy
