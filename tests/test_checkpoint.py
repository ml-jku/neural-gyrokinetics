"""JAX checkpoint round trips: model-only and full training state."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from neugk_jax.pinc import Swin5DAE
from neugk_jax.training.checkpoint import (
    CheckpointState,
    load_checkpoint,
    load_model_only,
    save_checkpoint,
    save_model_only,
)


def _toy_ae(key):
    return Swin5DAE(
        space=5,
        decouple_mu=True,
        dim=16,
        base_resolution=(4, 4, 4, 16, 8),
        in_channels=2,
        out_channels=2,
        patch_size=(2, 0, 2, 4, 2),
        window_size=(2, 0, 2, 2, 2),
        depth=2,
        num_heads=2,
        num_layers=2,
        bottleneck_dim=24,
        bottleneck_depth=1,
        bottleneck_num_heads=2,
        merging_depth=1,
        unmerging_depth=1,
        merging_hidden_ratio=2.0,
        unmerging_hidden_ratio=2.0,
        hidden_mlp_ratio=2.0,
        key=key,
    )


def test_save_model_only_roundtrip(tmp_path):
    ae = _toy_ae(jr.PRNGKey(0))
    x = jr.normal(jr.PRNGKey(1), (2, 2, 4, 4, 4, 16, 8))
    out_before = jax.vmap(lambda xi: ae(xi)["df"])(x)

    path = tmp_path / "ae.eqx"
    save_model_only(path, ae)
    template = _toy_ae(jr.PRNGKey(42))  # different init — proves leaves do override
    ae_restored = load_model_only(path, template)
    out_after = jax.vmap(lambda xi: ae_restored(xi)["df"])(x)
    assert jnp.allclose(out_before, out_after, atol=1e-6)


def test_full_checkpoint_roundtrip(tmp_path):
    """Full training-state snapshot: model + opt state + epoch + loss."""
    ae = _toy_ae(jr.PRNGKey(0))
    x = jr.normal(jr.PRNGKey(1), (2, 2, 4, 4, 4, 16, 8))
    out_before = jax.vmap(lambda xi: ae(xi)["df"])(x)

    # fake opt state — just a pytree mirroring the model leaves
    import equinox as eqx

    fake_opt_state = jax.tree_util.tree_map(
        lambda a: jnp.zeros_like(a), eqx.filter(ae, eqx.is_array)
    )
    state = CheckpointState(model=ae, opt_state=fake_opt_state, epoch=3, loss=0.123)
    path = tmp_path / "ckp.eqx"
    save_checkpoint(path, state)

    template = _toy_ae(jr.PRNGKey(42))
    loaded = load_checkpoint(path, template)
    assert loaded.epoch == 3
    assert loaded.loss == pytest.approx(0.123)
    out_after = jax.vmap(lambda xi: loaded.model(xi)["df"])(x)
    assert jnp.allclose(out_before, out_after, atol=1e-6)
