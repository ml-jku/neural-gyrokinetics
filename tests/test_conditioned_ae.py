"""Conditioned Swin5DAE: condition routing and order, translation names."""

from __future__ import annotations

import jax.random as jr
import numpy as np
import pytest
from helpers import COND, RES, tiny_ae, tiny_ae_model_cfg


def test_decoder_conditioned_ae_uses_the_condition():
    ae = tiny_ae()
    assert ae.condition_keys == ("dg", "itg", "q", "s_hat") and ae.enc_cond_embed is None
    assert ae.middle_post.modulated and not ae.middle_pre.modulated
    x = jr.normal(jr.PRNGKey(0), (4, *RES))
    out = ae(x, COND, return_latent=True)
    assert out["df"].shape == x.shape and not np.allclose(out["df"], ae(x, COND.at[0].add(1))["df"])
    np.testing.assert_array_equal(out["latent"], ae.encode(x))
    with pytest.raises(ValueError):
        ae(x)


def test_encoder_conditioning_picks_its_slots():
    ae = tiny_ae(encoder_conditioning=["q", "itg"])
    assert ae.enc_indices == (1, 2) and ae.dec_indices == (0, 1, 2, 3)
    x = jr.normal(jr.PRNGKey(0), (4, *RES))
    z1 = ae.encode(x, COND)
    # dg (slot 0) only reaches the decoder
    np.testing.assert_array_equal(z1, ae.encode(x, COND.at[0].set(-3.0)))
    assert not np.allclose(z1, ae.encode(x, COND.at[2].set(-3.0)))


def test_translation_names_and_unported_options():
    from neugk_jax.translate import _ae_name_map

    name = "backbone.up_blocks.0.swin.blocks.0.mod.proj.inner.weight"
    assert "up_blocks.0.swin_att.blocks.0.dit.modulation.weight" in _ae_name_map(name)
    with pytest.raises(NotImplementedError):
        tiny_ae(vit=dict(tiny_ae_model_cfg()["vit"], modulation="film"))
