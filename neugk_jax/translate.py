"""Torch → Equinox checkpoint translation.

Centralizes the AE/DiT translation logic so:

- ``scripts/translate_*ckpt.py`` are thin CLI wrappers
- the notebook can call ``load_or_translate(template, ckpt)`` and stay agnostic
- the rename tables (AE vs DiT) live next to a single shared translate loop

Public API:

- ``load_torch_state(.pth)`` → ``dict[str, np.ndarray]``
- ``build_ae_from_config(cfg, key)`` → ``Swin5DAE`` (f32-forced)
- ``build_dit_from_config(cfg, ae, key)`` → ``DiT`` (f32-forced)
- ``translate_ae(model, state)`` and ``translate_dit(model, state)``
- ``load_or_translate(template, ckpt_path)`` — dispatches on suffix + template type
"""

from __future__ import annotations

import copy
import re
from typing import Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import yaml


_LAYERS_RE = re.compile(r"\.layers\.(\d+)")
_NON_PERSISTENT = (".attn_mask", ".rel_pos", ".rpb", ".rpb_idx", ".omega")
_AE_DEAD = ("backbone.middle.",)


def force_f32(model):
    """Cast every f64 ``jax.Array`` leaf in ``model`` down to f32.

    ``eqx.nn.Linear`` initialises with the JAX default dtype, which is f64
    whenever ``jax_enable_x64`` is on (gyaradax flips it on import). Letting
    those f64 leaves through to the sampler breaks ``jax.lax.scan`` and
    silently doubles every cudnn/cublas kernel cost.
    """
    return jax.tree_util.tree_map(
        lambda x: x.astype(jnp.float32)
        if isinstance(x, jax.Array) and x.dtype == jnp.float64 else x,
        model,
    )


def load_torch_state(path: str) -> dict[str, np.ndarray]:
    """Open a torch ``.pth`` on CPU and return a flat numpy dict."""
    import torch
    blob = torch.load(path, map_location="cpu", weights_only=False)
    sd = blob["model_state_dict"] if isinstance(blob, dict) and "model_state_dict" in blob else blob
    if any(k.startswith("module.") for k in sd):
        sd = {k.removeprefix("module."): v for k, v in sd.items()}
    out = {}
    for k, v in sd.items():
        t = v.detach().cpu()
        if t.dtype == torch.bfloat16:
            t = t.float()
        out[k] = t.numpy()
    return out


def iter_leaves(tree, prefix: str = ""):
    """Walk an equinox tree, yielding ``(dotted_name, leaf)`` for every array."""
    if isinstance(tree, (jax.Array, np.ndarray)):
        yield prefix.lstrip("."), tree
        return
    if isinstance(tree, (list, tuple)):
        for i, v in enumerate(tree):
            yield from iter_leaves(v, f"{prefix}.{i}")
        return
    if isinstance(tree, dict):
        for k, v in tree.items():
            yield from iter_leaves(v, f"{prefix}.{k}")
        return
    if hasattr(tree, "__dataclass_fields__"):
        for fname in tree.__dataclass_fields__:
            try:
                v = getattr(tree, fname)
            except AttributeError:
                continue
            yield from iter_leaves(v, f"{prefix}.{fname}")


def _is_non_persistent(name: str) -> bool:
    return any(name.endswith(s) for s in _NON_PERSISTENT)


def _is_dead_leaf(name: str) -> bool:
    return any(s in name for s in _AE_DEAD)


def _ae_name_map(jax_name: str) -> list[str]:
    """AE JAX-name → candidate torch state_dict keys."""
    base = jax_name
    base = base.replace(".inner.", ".")
    if base.startswith("backbone."):
        base = base[len("backbone."):]
    base = base.replace(".swin.", ".swin_att.")
    base = base.replace(".downsample.proj.", ".downsample.reduction.")
    base = base.replace(".gate.proj.", ".gate.gate.1.")
    base = _LAYERS_RE.sub(lambda m: f".mlp.{int(m.group(1)) * 3}", base)
    return [jax_name, base]


def _dit_name_map(jax_name: str) -> list[str]:
    """DiT JAX-name → candidate torch state_dict keys."""
    base = jax_name.replace(".inner.", ".")
    cand = [base]
    base2 = base.replace(".mod.proj.", ".dit.modulation.")
    base2 = _LAYERS_RE.sub(lambda m: f".mlp.{int(m.group(1)) * 3}", base2)
    if base2 not in cand:
        cand.append(base2)
    return cand


def _gyroswin_name_map(jax_name: str) -> list[str]:
    """GyroSwin JAX-name → candidate torch state_dict keys.

    Reuses the AE renames inside each U-Net subtree, plus the cross-attention
    layer name mappings that are unique to gyroswin (``MixingBlock``,
    ``VSpaceReduce`` — the ``.kv.`` / ``.proj.`` / ``integral_token`` fields
    already align with torch names).
    """
    base = jax_name.replace(".inner.", ".")
    # the .swin. → .swin_att. rename only applies inside the U-Net subtrees
    base = base.replace(".swin.", ".swin_att.")
    base = base.replace(".downsample.proj.", ".downsample.reduction.")
    base = base.replace(".gate.proj.", ".gate.gate.1.")
    base = _LAYERS_RE.sub(lambda m: f".mlp.{int(m.group(1)) * 3}", base)
    return [jax_name, base]


def _apply_replacements(model, replacements):
    new_model = copy.deepcopy(model)
    for dotted, value in replacements.items():
        parts = dotted.split(".")
        node = new_model
        for p_ in parts[:-1]:
            node = (node[int(p_)] if isinstance(node, list)
                    else (node[p_] if isinstance(node, dict) else getattr(node, p_)))
        leaf = getattr(node, parts[-1])
        object.__setattr__(node, parts[-1], jnp.asarray(value, dtype=leaf.dtype))
    return new_model


def _translate(model, torch_state, name_map, *, strict: bool = False):
    leaves = list(iter_leaves(model))
    used = set()
    missing = []
    replacements = {}
    for name, leaf in leaves:
        matched = False
        for cand in name_map(name):
            if cand in torch_state:
                tw = torch_state[cand]
                if tuple(tw.shape) == tuple(leaf.shape):
                    replacements[name] = tw; used.add(cand); matched = True; break
                # torch APE has a leading singleton batch axis — squeeze it
                if (tw.ndim == leaf.ndim + 1 and tw.shape[0] == 1
                        and tuple(tw.shape[1:]) == tuple(leaf.shape)):
                    replacements[name] = np.asarray(tw).squeeze(0)
                    used.add(cand); matched = True; break
        if not matched and not _is_non_persistent(name) and not _is_dead_leaf(name):
            missing.append((name, tuple(leaf.shape)))
    unused = sorted(set(torch_state) - used)
    if strict and (missing or unused):
        raise RuntimeError(f"translate strict: missing={len(missing)}, unused={len(unused)}")
    return _apply_replacements(model, replacements), missing, unused


def translate_ae(model, torch_state, *, strict: bool = False):
    """Translate an upstream torch AE state_dict onto an Equinox ``Swin5DAE``."""
    return _translate(model, torch_state, _ae_name_map, strict=strict)


def translate_dit(model, torch_state, *, strict: bool = False):
    """Translate an upstream torch DiT state_dict onto an Equinox ``DiT``."""
    return _translate(model, torch_state, _dit_name_map, strict=strict)


def translate_gyroswin(model, torch_state, *, strict: bool = False):
    """Translate an upstream torch GyroSwinMultitask state_dict onto our equinox port."""
    return _translate(model, torch_state, _gyroswin_name_map, strict=strict)


def build_ae_from_config(
    cfg_path: str, *, key, resolution: Optional[Sequence[int]] = None,
):
    """Construct a ``Swin5DAE`` from a Hydra YAML config (upstream or local)."""
    from neugk_jax.autoencoders import Swin5DAE
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    mcfg = cfg["model"]
    vit, patch, bn = mcfg.get("vit", {}), mcfg.get("patch", {}), mcfg.get("bottleneck", {})
    dataset = cfg.get("dataset", {})
    base_resolution = resolution or dataset.get("resolution") or (32, 8, 16, 85, 32)
    depth = vit["depth"]
    n_layers = mcfg.get("num_layers", len(depth) if isinstance(depth, (list, tuple)) else 4)
    sep_zf = dataset.get("separate_zf", False)
    return force_f32(Swin5DAE(
        space=5,
        decouple_mu=mcfg.get("decouple_mu", True),
        dim=mcfg["latent_dim"],
        base_resolution=list(base_resolution),
        in_channels=dataset.get("in_channels", 2 * (2 if sep_zf else 1)),
        out_channels=dataset.get("out_channels", 2 * (2 if sep_zf else 1)),
        patch_size=patch["patch_size"],
        window_size=patch["window_size"],
        depth=depth,
        num_heads=vit["num_heads"],
        num_layers=n_layers,
        middle_depth=mcfg.get("middle_depth", 2),
        middle_num_heads=mcfg.get("middle_num_heads", 8),
        bottleneck_dim=bn.get("dim"),
        bottleneck_depth=bn.get("depth", 2),
        bottleneck_num_heads=bn.get("num_heads", 2),
        hidden_mlp_ratio=mcfg.get("hidden_mlp_ratio", 2.0),
        merging_hidden_ratio=patch.get("merging_hidden_ratio", 8.0),
        unmerging_hidden_ratio=patch.get("unmerging_hidden_ratio", 8.0),
        merging_depth=patch.get("merging_depth", 2),
        unmerging_depth=patch.get("unmerging_depth", 2),
        c_multiplier=int(patch.get("c_multiplier", 2)),
        normalized_latent=bn.get("normalized_latent", False),
        qkv_bias=vit.get("qkv_bias", False),
        qk_norm=vit.get("qk_norm", False),
        use_rpb=vit.get("use_rpb", True),
        gated_attention=vit.get("gated_attention", False),
        norm_affine=False,
        key=key,
    ))


def build_dit_from_config(cfg_path: str, ae, *, key):
    """Construct a ``DiT`` whose dims match an existing AE's bottleneck."""
    from neugk_jax.diffusion.dit import DiT
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    mcfg = cfg["model"]
    vit = mcfg["vit"]
    grid = tuple(ae.bottleneck_grid_size)
    return force_f32(DiT(
        space=len(grid),
        z_dim=int(ae.bottleneck_dim),
        dim=mcfg["latent_dim"],
        grid_size=grid,
        depth=vit["depth"],
        num_heads=vit["num_heads"],
        n_cond=len(mcfg.get("conditioning", []) or []),
        key=key,
        mlp_ratio=vit.get("mlp_ratio", 2.0),
        drop_path=vit.get("drop_path", 0.1),
    ))


def load_or_translate(template, ckpt_path: str):
    """``.eqx`` → load; ``.pth`` → on-the-fly translate. Returns the model."""
    from neugk_jax.training.checkpoint import load_model_only
    from neugk_jax.diffusion.dit import DiT
    from neugk_jax.gyroswin.models.gyroswin import GyroSwinMultitask

    if ckpt_path.endswith(".eqx"):
        return load_model_only(ckpt_path, template)
    state = load_torch_state(ckpt_path)
    if isinstance(template, GyroSwinMultitask):
        fn = translate_gyroswin
    elif isinstance(template, DiT):
        fn = translate_dit
    else:
        fn = translate_ae
    model, missing, unused = fn(template, state)
    print(f"  translated torch -> jax: {len(missing)} missing, {len(unused)} unused")
    return model
