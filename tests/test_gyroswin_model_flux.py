"""GyroSwin flux head conditioning, flux reductions and builder config validation."""

from __future__ import annotations

import copy

import jax.numpy as jnp
import jax.random as jr
import pytest
import yaml
from helpers import GYROSWIN_CFG, GYROSWIN_RES

from neugk_jax.gyroswin.models.x_layers import FluxDecoder
from neugk_jax.models.build import build_gyroswin_from_config
from neugk_jax.translate import named_leaves


def _build(tmp_path, edit=None):
    cfg = copy.deepcopy(GYROSWIN_CFG)
    if edit:
        edit(cfg)
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return build_gyroswin_from_config(str(path), key=jr.PRNGKey(0))


def test_flux_decoder_conditioning_changes_output():
    left, right = jr.normal(jr.PRNGKey(1), (3, 2, 16)), jr.normal(jr.PRNGKey(2), (5, 8))
    head = FluxDecoder([16], [8], 2, 1, key=jr.PRNGKey(0), n_cond=3)
    names = [n for n, _ in named_leaves(head)]
    assert "blocks.0.cond_embed.mlp.0.inner.weight" in names
    assert "blocks.0.conditioning.0.modulation.inner.weight" in names
    a = head([head.mix(0, left, right, jnp.array([0.1, 0.2, 0.3]))])
    b = head([head.mix(0, left, right, jnp.array([1.0, -2.0, 0.5]))])
    assert a.shape == (1,) and not jnp.allclose(a, b)
    plain = FluxDecoder([16], [8], 2, 1, key=jr.PRNGKey(0))
    assert not plain.use_cond and plain.blocks[0].cond_embed is None


def test_flux_max_reduction():
    left, right = jr.normal(jr.PRNGKey(1), (3, 2, 16)), jr.normal(jr.PRNGKey(2), (5, 8))
    head = FluxDecoder([16], [8], 2, 1, key=jr.PRNGKey(0))
    assert head.mix(0, left, right).shape == (16,)


def test_builder_flux_conditioning(tmp_path):
    model = _build(tmp_path)
    assert model.flux_key == "fluxavg" and model.flux_head.use_cond
    x = jr.normal(jr.PRNGKey(1), (4, *GYROSWIN_RES))
    out = model(x, jnp.array([0.3, -0.2]))
    assert out["fluxavg"].shape == (1,)
    assert out["df"].shape == x.shape and out["phi"].shape == (
        GYROSWIN_RES[3],
        GYROSWIN_RES[2],
        GYROSWIN_RES[4],
    )
    plain = _build(tmp_path, lambda c: c["model"]["swin"].update(flux_conditioning=False))
    assert not plain.flux_head.use_cond


@pytest.mark.parametrize(
    "edit,exc",
    [
        (lambda c: c["model"]["swin"].update(use_rope=True), NotImplementedError),
        (lambda c: c["model"]["swin"].update(swin_bottleneck=False), NotImplementedError),
        (lambda c: c["model"]["swin"].update(act_fn="SiLU"), NotImplementedError),
        (lambda c: c["model"]["swin"].update(modulation="adaln"), NotImplementedError),
        (lambda c: c["model"]["swin"].update(bogus_key=1), ValueError),
        (lambda c: c["model"]["swin"].update(flux_reduce="mean"), NotImplementedError),
        (lambda c: c["model"].update(bundle_seq_length=2), NotImplementedError),
        (lambda c: c["dataset"].update(real_potens=False), NotImplementedError),
        (lambda c: c["training"].update(predict_delta=True), NotImplementedError),
        (lambda c: c["training"]["pushforward"].update(unrolls=[0, 2]), NotImplementedError),
    ],
)
def test_builder_rejects_unsupported(tmp_path, edit, exc):
    with pytest.raises(exc):
        _build(tmp_path, edit)


def test_builder_accepts_mappings(tmp_path):
    from omegaconf import OmegaConf

    ref = _build(tmp_path)
    for src in (copy.deepcopy(GYROSWIN_CFG), OmegaConf.create(copy.deepcopy(GYROSWIN_CFG))):
        model = build_gyroswin_from_config(src, key=jr.PRNGKey(0))
        assert [n for n, _ in named_leaves(model)] == [n for n, _ in named_leaves(ref)]
