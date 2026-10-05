"""LoRA adapters: init, merge, strategy targets and the torch export round trip."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from helpers import RES, TARGETS, tiny_ae, tiny_vq_cfg

from neugk_jax.models.lora import LoRALinear, get_path, merge_lora, module_paths
from neugk_jax.models.utils import Linear


def test_lora_linear_init_forward_and_merge():
    base = Linear(12, 5, key=jr.PRNGKey(0))
    lin = LoRALinear(base, r=4, alpha=0.5, key=jr.PRNGKey(1))
    assert lin.scale == 0.125 and lin.lora_A.shape == (4, 12) and lin.lora_B.shape == (5, 4)
    assert float(jnp.abs(lin.lora_A).max()) <= 1 / np.sqrt(12) and not lin.lora_B.any()
    x = jr.normal(jr.PRNGKey(2), (3, 12))
    np.testing.assert_array_equal(lin(x), base(x))
    lin = eqx.tree_at(lambda m: m.lora_B, lin, jr.normal(jr.PRNGKey(3), (5, 4)))
    np.testing.assert_allclose(lin.merged()(x), lin(x), rtol=1e-5, atol=1e-6)
    assert not np.allclose(lin(x), base(x))


def test_lora_strategy_targets():
    from neugk_jax.translate import _ae_module_name, ae_lora_paths

    ae = tiny_ae()
    names = {_ae_module_name(p) for p in ae_lora_paths(ae, {"strategy": "attention_mlp"})}
    assert set(TARGETS) <= names
    assert "middle_downproj" not in names and not any("modulation" in n for n in names)
    bottleneck = ae_lora_paths(ae, {"strategy": "bottleneck_only"})
    assert sorted(bottleneck) == ["middle_downproj", "middle_upproj"]
    excluded = ae_lora_paths(ae, {"strategy": "attention_mlp", "exclude_patterns": ["qkv"]})
    assert excluded and not any("qkv" in p for p in excluded)
    with pytest.raises(KeyError):
        ae_lora_paths(ae, {"target_modules": ["not.a.module"]})
    default = ae_lora_paths(ae, {})
    assert set(default) == set(ae_lora_paths(ae, {"strategy": "comprehensive"}))
    assert {"middle_downproj", "middle_upproj"} <= set(default)


def test_vqvae_lora_names():
    from neugk_jax.models.build import build_ae_from_config
    from neugk_jax.translate import _ae_module_name, ae_lora_paths

    cfg = {"model": tiny_vq_cfg(), "dataset": {"resolution": RES}}
    vq = build_ae_from_config(cfg, key=jr.PRNGKey(0))
    assert not ae_lora_paths(vq, {"strategy": "bottleneck_only"})
    names = [
        _ae_module_name(p, vq=True)
        for p in ae_lora_paths(vq, {"target_modules": ["middle_vq_upproj"]})
    ]
    assert names == ["middle_vq_upproj"]


def test_export_merges_adapters_and_round_trips():
    from neugk_jax.translate import (
        ae_state_template,
        attach_ae_lora,
        export_ae_state,
        named_leaves,
        translate_ae,
    )

    ae = tiny_ae()
    template = ae_state_template(ae)
    lora = {"target_modules": TARGETS, "r": 2, "lora_alpha": 1.0}
    model = attach_ae_lora(ae, lora, key=jr.PRNGKey(1))
    for i, p in enumerate(module_paths(model, LoRALinear)):
        b = 0.1 * jr.normal(jr.PRNGKey(i), get_path(model, p).lora_B.shape)
        model = eqx.tree_at(lambda m, p=p: get_path(m, p).lora_B, model, b)
    state = export_ae_state(model, template)
    assert set(state) == set(template)
    back, _, _ = translate_ae(tiny_ae(), state, strict=True)
    for (_, a), (_, b) in zip(named_leaves(back), named_leaves(merge_lora(model))):
        np.testing.assert_array_equal(a, b)
    peft = export_ae_state(model, template, peft_format=True)
    assert len(peft) == len(template) + 2 * len(TARGETS)
    assert "middle_post.blocks.0.attn.qkv.base_layer.weight" in peft
    assert "middle_post.blocks.0.attn.qkv.lora_B.default.weight" in peft


def test_vqvae_export_round_trips():
    from neugk_jax.models.build import build_ae_from_config
    from neugk_jax.translate import ae_state_template, attach_ae_lora, export_ae_state, translate_ae

    cfg = {"model": tiny_vq_cfg(), "dataset": {"resolution": RES}}
    vq = build_ae_from_config(cfg, key=jr.PRNGKey(0))
    template = ae_state_template(vq)
    assert {"middle_vq_downproj.weight", "vq._codebook.embed", "vq._codebook.initted"} <= set(
        template
    )
    lora = {"target_modules": ["middle_vq_downproj"], "r": 2, "lora_alpha": 1.0}
    peft = export_ae_state(attach_ae_lora(vq, lora, key=jr.PRNGKey(1)), template, peft_format=True)
    assert "middle_vq_downproj.lora_A.default.weight" in peft
    assert "middle_vq_downproj.base_layer.weight" in peft
    back, missing, unused = translate_ae(vq, export_ae_state(vq, template), strict=True)
    assert not missing and not unused
