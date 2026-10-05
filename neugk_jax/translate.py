"""Torch → Equinox checkpoint translation (weights only; templates come from ``models.build``).

- ``load_torch_state(.pth)`` → ``dict[str, np.ndarray]``
- ``translate_ae`` / ``translate_dit`` / ``translate_gyroswin(model, state)`` → ``(model,
  missing, unused)``: every array leaf takes the first torch key among its candidate names
  with a matching shape
- ``load_or_translate(template, ckpt_path)`` — dispatches on suffix + template type
- ``attach_ae_lora(model, lora_cfg, key)`` — LoRA adapters on the AE linears of a peft config
- ``ae_state_template(model)`` — checkpoint keys and shapes of a JAX AE
- ``export_ae_state(model, template)`` — JAX AE (LoRA merged, or peft keys) to a torch state_dict
"""

from __future__ import annotations

import re
from collections.abc import Mapping

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

_LAYERS_RE = re.compile(r"\.layers\.(\d+)")
_NON_PERSISTENT = (".attn_mask", ".rel_pos", ".rpb", ".rpb_idx", ".omega")
_VQ_BUFFER_RE = re.compile(r"^vq\.(embed|embed_avg|cluster_size)$")
# checkpoint keys without a model leaf: skipped on translation, written as ones on export
_UNMAPPED_KEYS = ("vq._codebook.initted",)
# checkpoint tensors with a leading singleton axis the model leaf drops
_LEADING_AXIS_RE = re.compile(r"(pos_embed|^vq\._codebook\.(embed|embed_avg|cluster_size))$")


def _unimportable(e: Exception) -> bool:
    # a pickled class whose module is missing or fails at import (deepspeed without cuda)
    return (
        isinstance(e, (ImportError, AttributeError)) or type(e).__name__ == "MissingCUDAException"
    )


def _stub_pickle_module():
    """``pickle`` shim whose unpickler stubs classes it cannot import.

    Replaces any class that fails to import (e.g. deepspeed's ``LossScaler``,
    which needs a CUDA toolchain) with a permissive placeholder, so tensor
    data can still be read out.
    """
    import pickle
    import types

    mod = types.ModuleType("neugk_jax_stub_pickle")
    mod.__dict__.update(pickle.__dict__)
    mod.__name__ = "neugk_jax_stub_pickle"

    class _StubUnpickler(pickle.Unpickler):
        def find_class(self, mod_name, name):
            try:
                return super().find_class(mod_name, name)
            except Exception as e:
                if not _unimportable(e):
                    raise
                # permissive ctor: enums/scalers are rebuilt as ``Cls(value)``
                return type(
                    name,
                    (),
                    {
                        "__init__": lambda self, *a, **k: None,
                        "__setstate__": lambda self, state: None,
                    },
                )

    mod.Unpickler = _StubUnpickler
    return mod


def load_torch_state(path: str) -> dict[str, np.ndarray]:
    """Open a torch ``.pth`` on CPU and return a flat numpy dict."""
    import torch

    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        if not _unimportable(e):
            raise
        # trainer-side objects (e.g. deepspeed loss scalers) whose modules do not import here
        blob = torch.load(
            path, map_location="cpu", weights_only=False, pickle_module=_stub_pickle_module()
        )
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


def _key_name(k) -> str:
    for attr in ("name", "idx", "key"):
        if hasattr(k, attr):
            return str(getattr(k, attr))
    return str(k)


def named_leaves(model) -> list[tuple[str, jax.Array]]:
    flat, _ = jax.tree_util.tree_flatten_with_path(model)
    return [(".".join(map(_key_name, p)), leaf) for p, leaf in flat if eqx.is_array(leaf)]


def _is_non_persistent(name: str) -> bool:
    return any(name.endswith(s) for s in _NON_PERSISTENT)


# jax -> torch renames: equinox's wrapped Linear/LayerNorm, the u-net modules, DiT modulation
_INNER = ((".inner.", "."),)
_UNET = (
    (".swin.", ".swin_att."),
    (".downsample.proj.", ".downsample.reduction."),
    (".gate.proj.", ".gate.gate.1."),
)
_DIT = ((".mod.proj.", ".dit.modulation."),)
# torch wraps proj_concat in an nn.Sequential, so the param sits at proj_concat.0.*
_PROJ_CONCAT = ((".proj_concat.", ".proj_concat.0."),)


def _name_map(*renames, strip_prefix: str = ""):
    """Candidate torch keys of a jax leaf name: itself, then ``renames`` applied in order
    (after the ``.inner.`` unwrap and ``strip_prefix``), with ``layers.i`` -> ``mlp.3i``."""

    def candidates(jax_name: str) -> list[str]:
        base = jax_name.replace(*_INNER[0]).removeprefix(strip_prefix)
        for old, new in renames:
            base = base.replace(old, new)
        return [jax_name, _LAYERS_RE.sub(lambda m: f".mlp.{int(m.group(1)) * 3}", base)]

    return candidates


_ae_base_map = _name_map(*_UNET, *_DIT, strip_prefix="backbone.")


def _ae_name_map(jax_name: str) -> list[str]:
    own, base = _ae_base_map(jax_name)
    base = _VQ_BUFFER_RE.sub(r"vq._codebook.\1", base)
    if base.startswith(("middle_downproj.", "middle_upproj.")):
        return [own, base, base.replace("middle_", "middle_vq_", 1)]
    return [own, base]


_dit_name_map = _name_map(*_DIT)
_gyroswin_name_map = _name_map(*_UNET, *_DIT, *_PROJ_CONCAT)


def _translate(model, torch_state, name_map, *, strict: bool = False):
    leaves = named_leaves(model)
    used = set()
    missing = []
    replacements = {}
    for name, leaf in leaves:
        matched = False
        for cand in name_map(name):
            if cand in torch_state:
                tw = torch_state[cand]
                if tuple(tw.shape) == tuple(leaf.shape):
                    replacements[name] = tw
                    used.add(cand)
                    matched = True
                    break
                # torch ape has a leading singleton batch axis — squeeze it
                if (
                    tw.ndim == leaf.ndim + 1
                    and tw.shape[0] == 1
                    and tuple(tw.shape[1:]) == tuple(leaf.shape)
                ):
                    replacements[name] = np.asarray(tw).squeeze(0)
                    used.add(cand)
                    matched = True
                    break
        if not matched and not _is_non_persistent(name):
            missing.append((name, tuple(leaf.shape)))
    unused = sorted(set(torch_state) - used - set(_UNMAPPED_KEYS))
    if strict and (missing or unused):
        raise RuntimeError(f"translate strict: missing={len(missing)}, unused={len(unused)}")
    flat, treedef = jax.tree_util.tree_flatten_with_path(model)
    new = [
        jnp.asarray(replacements[n], leaf.dtype) if n in replacements else leaf
        for n, leaf in ((".".join(map(_key_name, p)), leaf) for p, leaf in flat)
    ]
    return jax.tree_util.tree_unflatten(treedef, new), missing, unused


def translate_ae(model, torch_state, *, strict: bool = False):
    return _translate(model, torch_state, _ae_name_map, strict=strict)


def translate_dit(model, torch_state, *, strict: bool = False):
    return _translate(model, torch_state, _dit_name_map, strict=strict)


def translate_gyroswin(model, torch_state, *, strict: bool = False):
    return _translate(model, torch_state, _gyroswin_name_map, strict=strict)


def load_or_translate(template, ckpt_path: str, *, strict: bool = False):
    """``.eqx`` → load; ``.pth`` → on-the-fly translate (``strict`` raises on any key mismatch)."""
    from neugk_jax.diffusion.dit import DiT
    from neugk_jax.gyroswin.models.gyroswin import GyroSwinMultitask
    from neugk_jax.training.checkpoint import load_model_only

    if ckpt_path.endswith(".eqx"):
        return load_model_only(ckpt_path, template)
    state = load_torch_state(ckpt_path)
    if isinstance(template, GyroSwinMultitask):
        fn = translate_gyroswin
    elif isinstance(template, DiT):
        fn = translate_dit
    else:
        fn = translate_ae
    model, missing, unused = fn(template, state, strict=strict)
    print(f"  translated torch -> jax: {len(missing)} missing, {len(unused)} unused")
    return model


def report(model, torch_state: dict, missing, unused, limit: int = 20) -> None:
    """Print the translation coverage and the first ``limit`` missing / unused names."""
    total = len(named_leaves(model))
    print(f"translated leaves: {total - len(missing)} / {total}")
    if missing:
        print(f"missing JAX leaves ({len(missing)}):")
        for n, s in missing[:limit]:
            print(f"  {n}  shape={s}")
    if unused:
        print(f"unused torch keys ({len(unused)}):")
        for n in unused[:limit]:
            print(f"  {n}  shape={torch_state[n].shape}")


def _is_vq(model) -> bool:
    from neugk_jax.pinc import Swin5DVQVAE

    return isinstance(model, Swin5DVQVAE)


def _ae_key(jax_name: str, *, vq: bool = False) -> str:
    cands = _ae_name_map(jax_name)
    return cands[-1] if vq else cands[1]


def _ae_module_name(path: str, *, vq: bool = False) -> str:
    return _ae_key(f"{path}.inner.weight", vq=vq).removesuffix(".weight")


def ae_state_template(model) -> dict[str, np.ndarray]:
    """Checkpoint keys and arrays of a (LoRA-free) JAX AE, as :func:`export_ae_state` writes them."""
    from neugk_jax.pinc.quantizers import VectorQuantizer

    vq = _is_vq(model)
    leaves = named_leaves(model)
    out = {_ae_key(n, vq=vq): np.asarray(v) for n, v in leaves if not _is_non_persistent(n)}
    out = {k: v[None] if _LEADING_AXIS_RE.search(k) else v for k, v in out.items()}
    if vq and isinstance(model.vq, VectorQuantizer):
        out["vq._codebook.initted"] = np.ones((1,), np.float32)
    return out


LORA_DEFAULT_STRATEGY = "comprehensive"
_LORA_STRATEGIES = {
    "comprehensive": ("attention_qkv", "attention_proj", "mlp_layers", "bottleneck"),
    "attention_mlp": ("attention_qkv", "attention_proj", "mlp_layers"),
    "mlp_only": ("mlp_layers",),
    "attention_only": ("attention_qkv", "attention_proj"),
    "bottleneck_only": ("bottleneck",),
    "modulation_only": ("modulation",),
    "all_except_attention": ("mlp_layers", "bottleneck", "patch_embed", "modulation", "downsample"),
}
_MLP_PATTERNS = ("mlp.mlp.", "mlp.", "cond_embed.mlp", "cpb_mlp.mlp", "feed_forward")


def _classify_linear(name: str) -> str:
    if "patch_embed.patch.mlp" in name:
        return "patch_embed"
    if "qkv" in name:
        return "attention_qkv"
    if "attn.proj" in name or "attention.proj" in name:
        return "attention_proj"
    if any(p in name for p in _MLP_PATTERNS):
        return "mlp_layers"
    if any(p in name for p in ("middle_downproj", "middle_upproj", "bottleneck")):
        return "bottleneck"
    if "modulation" in name:
        return "modulation"
    if "downsample" in name or "upsample" in name:
        return "downsample"
    return "other"


def ae_lora_paths(model, lora_cfg: Mapping) -> list[str]:
    """JAX paths of the AE linears a ``peft.lora`` config adapts.

    ``target_modules`` lists checkpoint module names; without it ``strategy`` (default
    ``comprehensive``) selects the linear groups (minus names containing an
    ``exclude_patterns`` entry), in name order.
    """
    from neugk_jax.models.lora import module_paths
    from neugk_jax.models.utils import Linear

    vq = _is_vq(model)
    by_name = {_ae_module_name(p, vq=vq): p for p in module_paths(model, Linear)}
    targets = lora_cfg.get("target_modules")
    if not targets:
        strategy = lora_cfg.get("strategy", LORA_DEFAULT_STRATEGY)
        if strategy not in _LORA_STRATEGIES:
            raise ValueError(
                f"unknown lora strategy {strategy!r}; one of {sorted(_LORA_STRATEGIES)}"
            )
        exclude = lora_cfg.get("exclude_patterns") or ()
        targets = sorted(
            n
            for n in by_name
            if _classify_linear(n) in _LORA_STRATEGIES[strategy]
            and not any(e in n for e in exclude)
        )
    missing = sorted(set(targets) - set(by_name))
    if missing:
        raise KeyError(f"{len(missing)} adapter targets have no linear in the model: {missing[:5]}")
    return [by_name[m] for m in targets]


def attach_ae_lora(model, lora_cfg: Mapping, *, key):
    from neugk_jax.models.lora import attach_lora

    paths = ae_lora_paths(model, lora_cfg)
    return attach_lora(
        model, paths, r=int(lora_cfg["r"]), alpha=float(lora_cfg["lora_alpha"]), key=key
    )


def export_ae_state(
    model, template: Mapping[str, np.ndarray], *, peft_format: bool = False
) -> dict[str, np.ndarray]:
    """Torch state_dict of a JAX AE, keyed and shaped like the base ``template`` state_dict.

    Adapters are merged into the base weights, or with ``peft_format`` kept separable as
    ``<module>.base_layer.*`` plus ``<module>.lora_{A,B}.default.weight``. Raises unless every
    template key is written exactly once.
    """
    from neugk_jax.models.lora import LoRALinear, merge_lora, module_paths

    vq = _is_vq(model)
    adapted = set()
    if peft_format:
        adapted = {_ae_module_name(p, vq=vq) for p in module_paths(model, LoRALinear)}
    expected = dict(template)
    for mod in adapted:
        for k in [k for k in expected if k.rsplit(".", 1)[0] == mod]:
            expected[f"{mod}.base_layer.{k.rsplit('.', 1)[1]}"] = expected.pop(k)
    out = {}
    for name, leaf in named_leaves(model if peft_format else merge_lora(model)):
        if _is_non_persistent(name):
            continue
        arr = np.asarray(jax.device_get(leaf), dtype=np.float32)
        if name.endswith((".lora_A", ".lora_B")):
            mod, which = name.rsplit(".", 1)
            out[f"{_ae_module_name(mod, vq=vq)}.{which}.default.weight"] = arr
            continue
        cands = [c.replace(".base.", ".base_layer.") for c in _ae_name_map(name)]
        key = next((c for c in cands if c in expected), None)
        if key is None or key in out:
            raise KeyError(f"no unique torch key for {name}")
        tshape = tuple(expected[key].shape)
        if [d for d in arr.shape if d != 1] != [d for d in tshape if d != 1]:
            raise ValueError(f"{name}: {arr.shape} does not fit {key} {tshape}")
        out[key] = arr.reshape(tshape)
    for key in _UNMAPPED_KEYS:
        if key in expected:
            out[key] = np.ones(expected[key].shape, np.float32)
    missing = sorted(set(expected) - set(out))
    if missing:
        raise KeyError(f"{len(missing)} torch keys not exported: {missing[:5]}")
    return out


def save_torch_checkpoint(path: str, state: Mapping[str, np.ndarray], **meta) -> None:
    import torch

    sd = {k: torch.from_numpy(np.array(v)) for k, v in state.items()}
    torch.save({"model_state_dict": sd, **meta}, path)
