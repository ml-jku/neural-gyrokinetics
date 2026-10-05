"""GyroSwin release checkpoint ``ml-jku/gyroswin_large`` translated to JAX against torch.

Weights come from the Hugging Face hub; the model is built from the shipped release config
(``configs/checkpoints/gyroswin_large.yaml``) with the single swin residual the release
checkpoints use. Compares translation coverage and the forward pass (df, phi, fluxavg) on
a fixed random input and on the public snapshot ``iteration_8.h5`` at highest matmul
precision.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
import yaml

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "checkpoints" / "gyroswin_large.yaml"
RES = (32, 8, 16, 85, 32)
ALIASES = ("enc_cond_embed.", "dec_cond_embed.")
COS_MIN = 0.9999


@pytest.fixture(scope="module", autouse=True)
def highest_precision():
    import torch

    prev = jax.config.jax_default_matmul_precision
    jax.config.update("jax_default_matmul_precision", "highest")
    tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    yield
    jax.config.update("jax_default_matmul_precision", prev)
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32


@pytest.fixture(scope="module")
def state(hf_large_weights):
    from neugk_jax.translate import load_torch_state

    return load_torch_state(hf_large_weights)


@pytest.fixture(scope="module")
def translated(state):
    from neugk_jax.models.build import build_release_gyroswin
    from neugk_jax.translate import translate_gyroswin

    return translate_gyroswin(build_release_gyroswin(CONFIG, key=jr.PRNGKey(0)), state)


@pytest.fixture(scope="module")
def torch_model(hf_large_weights, single_swin_residual):
    import torch
    from neugk.gyroswin.models import get_model
    from omegaconf import OmegaConf

    mcfg = yaml.safe_load(CONFIG.read_text())["model"]
    cfg = OmegaConf.create(
        {
            "model": mcfg,
            "logging": {"model_summary": False},
            "dataset": {
                "input_fields": ["df"],
                "separate_zf": True,
                "real_potens": True,
                "active_keys": ["re", "im"],
            },
        }
    )

    class _Dataset:
        active_keys = ["re", "im"]
        resolution = RES
        phi_resolution = (RES[3], RES[2], RES[4])

    model = get_model(cfg, dataset=_Dataset())
    res = model.load_state_dict(
        torch.load(hf_large_weights, map_location="cpu", weights_only=True), strict=False
    )
    # the encoder/decoder condition embeds alias the shared one
    assert not res.unexpected_keys and all(any(a in k for a in ALIASES) for k in res.missing_keys)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(device).eval(), sorted(mcfg["conditioning"]), device


def _cos(a, b) -> float:
    a, b = np.ravel(a).astype(np.float64), np.ravel(b).astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def _compare(translated, torch_model, x, cond):
    import torch

    model, keys, device = torch_model
    jm = translated[0]
    with torch.no_grad():
        t = model(
            torch.from_numpy(x)[None].to(device),
            **{k: torch.tensor([[float(cond[i])]], device=device) for i, k in enumerate(keys)},
        )
    j = eqx_forward(jm, jnp.asarray(x), jnp.asarray(cond))
    assert set(j) == set(t) == {"df", "phi", "fluxavg"}
    out = {
        k: (
            _cos(t[k][0].cpu().numpy(), j[k]),
            float(t[k].cpu().numpy().ravel()[0]),
            float(np.ravel(j[k])[0]),
        )
        for k in j
    }
    print(
        {k: f"1-cos={1 - c:.2e}" for k, (c, _, _) in out.items()},
        f"fluxavg torch={out['fluxavg'][1]:.6f} jax={out['fluxavg'][2]:.6f}",
    )
    for k in ("df", "phi"):
        assert out[k][0] >= COS_MIN, (k, out[k][0])
    assert out["fluxavg"][2] == pytest.approx(out["fluxavg"][1], rel=1e-4, abs=1e-5)


@jax.jit
def eqx_forward(model, x, cond):
    return model(x, cond, inference=True)


def test_release_config_matches_the_torch_release(torch_repo_root):
    release = torch_repo_root / "configs" / "checkpoints" / "gyroswin_large" / "config.yaml"
    if not release.exists():
        pytest.skip(f"no release config at {release}")
    assert (
        yaml.safe_load(CONFIG.read_text())["model"] == yaml.safe_load(release.read_text())["model"]
    )


def test_translation_is_complete(state, translated):
    _, missing, unused = translated
    assert not missing, missing[:5]
    used = {state[k].tobytes() for k in set(state) - set(unused)}
    # torch registers shared modules under several names; the outer flux cond embed is dead
    real = [k for k in unused if state[k].tobytes() not in used]
    assert all(k.startswith("flux_head.cond_embed.") for k in real), real
    assert translated[0].flux_key == "fluxavg" and translated[0].flux_head.use_cond


def test_forward_parity_random_input(translated, torch_model):
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((4, *RES)) * 0.1).astype(np.float32)
    cond = rng.standard_normal(len(torch_model[1])).astype(np.float32)
    _compare(translated, torch_model, x, cond)


def test_forward_parity_real_sample(translated, torch_model, hf_sample):
    from neugk_jax.dataset import CycloneDataset, H5Backend

    ds = CycloneDataset(
        path=hf_sample.root,
        trajectories=[hf_sample.name],
        backend=H5Backend(),
        separate_zf=True,
        conditions=torch_model[1],
        normalization={"df": {"type": "zscore"}},
        normalization_stats=hf_sample.stats,
    )
    s = ds.sample(0, 0)
    assert ds.conditions == torch_model[1]
    _compare(translated, torch_model, np.asarray(s.df, np.float32), np.asarray(s.conditioning))
