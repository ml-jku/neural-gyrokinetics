"""Conditioned AE, LoRA adapters and the AE export against a random tiny torch PINC autoencoder."""

from __future__ import annotations

import copy

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from torch_ref import RES, STUB_DS, torch_ae_cfg, torch_doubles_swin_shortcut

TARGETS = [
    "dec_cond_embed.mlp.0",
    "middle_post.blocks.0.attn.qkv",
    "middle_post.blocks.0.mlp.mlp.3",
    "up_blocks.0.swin_att.blocks.1.attn.proj",
    "unpatch.expansion.mlp.3",
    "down_blocks.0.swin_att.blocks.0.attn.rpb.cpb_mlp.mlp.0",
    "middle_upscale.expansion.mlp.0",
]


def torch_ae():
    from neugk.pinc.autoencoders import get_autoencoder
    from omegaconf import OmegaConf

    return get_autoencoder(OmegaConf.create(torch_ae_cfg()), STUB_DS, rank=None).eval()


@pytest.fixture(scope="module")
def tiny():
    import torch

    from neugk_jax.models.build import build_ae_from_config
    from neugk_jax.translate import translate_ae

    torch.manual_seed(0)
    tmodel = torch_ae()
    state = {k: v.detach().numpy().copy() for k, v in tmodel.state_dict().items()}
    legacy = torch_doubles_swin_shortcut()
    jmodel = build_ae_from_config(torch_ae_cfg(), key=jr.PRNGKey(0), legacy_double_shortcut=legacy)
    jmodel, missing, unused = translate_ae(jmodel, state, strict=True)
    assert not missing and not unused
    x = np.random.default_rng(0).standard_normal((4, *RES)).astype(np.float32)
    cond = np.asarray([1.6, 2.6, 8.5, 0.6], np.float32)
    return tmodel, jmodel, state, x, cond


def _torch_forward(tmodel, x, cond):
    import torch

    with torch.no_grad():
        out = tmodel(torch.from_numpy(x)[None], condition=torch.from_numpy(cond)[None])
    return out["df"][0].numpy()


def _rel(a, b):
    return float(np.linalg.norm(np.asarray(a) - b) / np.linalg.norm(b))


def _with_lora(tmodel, jmodel):
    import torch
    from neugk.pinc.peft_utils import attach_peft_adapters

    from neugk_jax.models.lora import set_adapters
    from neugk_jax.translate import ae_lora_paths, attach_ae_lora

    tm = attach_peft_adapters(copy.deepcopy(tmodel), r=2, lora_alpha=0.5, target_modules=TARGETS)
    mods = dict(tm.named_modules())
    gen = torch.Generator().manual_seed(1)
    adapters = {}
    for name, path in zip(TARGETS, ae_lora_paths(jmodel, {"target_modules": TARGETS})):
        with torch.no_grad():
            mods[name].lora_B["default"].weight.normal_(0.0, 0.3, generator=gen)
        ab = (mods[name].lora_A["default"].weight, mods[name].lora_B["default"].weight)
        adapters[path] = tuple(w.detach().numpy() for w in ab)
    lora = {"target_modules": TARGETS, "r": 2, "lora_alpha": 0.5}
    return tm.eval(), set_adapters(attach_ae_lora(jmodel, lora, key=jr.PRNGKey(1)), adapters)


def test_conditioned_ae_and_lora_forward_match_torch(tiny):
    tmodel, jmodel, _, x, cond = tiny
    with jax.default_matmul_precision("highest"):
        ref = _torch_forward(tmodel, x, cond)
        assert _rel(jmodel(jnp.asarray(x), jnp.asarray(cond))["df"], ref) < 1e-5
        tm, jm = _with_lora(tmodel, jmodel)
        t_lora = _torch_forward(tm, x, cond)
        assert _rel(t_lora, ref) > 1e-3
        assert _rel(jm(jnp.asarray(x), jnp.asarray(cond))["df"], t_lora) < 1e-5


def test_export_loads_into_torch_strict(tiny):
    import torch

    from neugk_jax.translate import export_ae_state

    tmodel, jmodel, state, x, cond = tiny
    tm, jm = _with_lora(tmodel, jmodel)
    fresh = torch_ae()
    exported = export_ae_state(jm, state)
    fresh.load_state_dict({k: torch.from_numpy(v) for k, v in exported.items()}, strict=True)
    assert _rel(_torch_forward(fresh, x, cond), _torch_forward(tm, x, cond)) < 1e-5
