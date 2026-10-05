"""Low-rank adapters (LoRA) on ``Linear`` layers."""

from __future__ import annotations

import math
from typing import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from neugk_jax.models.utils import Linear


class LoRALinear(eqx.Module):
    """``base(x) + scale * (x A^T) B^T`` with ``A`` (r, in) kaiming-uniform and ``B`` (out, r) zero."""

    base: Linear
    lora_A: jax.Array
    lora_B: jax.Array
    scale: float = eqx.field(static=True)

    def __init__(self, base: Linear, *, r: int, alpha: float, key):
        out_dim, in_dim = base.weight.shape
        # kaiming-uniform with a = sqrt(5): bound = 1 / sqrt(fan_in)
        bound = 1.0 / math.sqrt(in_dim)
        self.base = base
        self.lora_A = jr.uniform(key, (r, in_dim), base.weight.dtype, -bound, bound)
        self.lora_B = jnp.zeros((out_dim, r), base.weight.dtype)
        self.scale = float(alpha) / float(r)

    def merged(self) -> Linear:
        delta = jnp.matmul(self.lora_B, self.lora_A, precision=jax.lax.Precision.HIGHEST)
        return eqx.tree_at(
            lambda m: m.inner.weight, self.base, self.base.weight + self.scale * delta
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        a, b = self.lora_A.astype(x.dtype), self.lora_B.astype(x.dtype)
        return self.base(x) + self.scale * jnp.matmul(jnp.matmul(x, a.T), b.T)


def get_path(node, path: str):
    for p in path.split("."):
        node = node[int(p)] if isinstance(node, (list, tuple)) else getattr(node, p)
    return node


def module_paths(tree, cls, prefix: str = "") -> list[str]:
    """Dotted attribute paths of every ``cls`` instance in an equinox tree."""
    if isinstance(tree, cls):
        return [prefix]
    if isinstance(tree, (list, tuple)):
        items = list(enumerate(tree))
    elif isinstance(tree, eqx.Module):
        items = [(n, getattr(tree, n, None)) for n in tree.__dataclass_fields__]
    else:
        return []
    return [p for k, v in items for p in module_paths(v, cls, f"{prefix}.{k}".lstrip("."))]


def attach_lora(model, paths: Sequence[str], *, r: int, alpha: float, key):
    """Replace the ``Linear`` at each dotted attribute path of ``model`` with a ``LoRALinear``."""
    paths = list(paths)
    for path, k in zip(paths, jr.split(key, max(len(paths), 1))):
        base = get_path(model, path)
        if not isinstance(base, Linear):
            raise TypeError(f"{path} is a {type(base).__name__}, not a Linear")
        lora = LoRALinear(base, r=r, alpha=alpha, key=k)
        model = eqx.tree_at(lambda m, p=path: get_path(m, p), model, lora)
    return model


def set_adapters(model, adapters: dict):
    """``model`` with ``{path: (lora_A, lora_B)}`` copied into its adapters."""
    for path, (a, b) in adapters.items():
        model = eqx.tree_at(
            lambda m, p=path: (get_path(m, p).lora_A, get_path(m, p).lora_B),
            model,
            (jnp.asarray(a), jnp.asarray(b)),
        )
    return model


def _is_lora(x) -> bool:
    return isinstance(x, LoRALinear)


def lora_mask(model):
    def visit(m, mk):
        return eqx.tree_at(lambda t: (t.lora_A, t.lora_B), mk, (True, True)) if _is_lora(m) else mk

    mask = jax.tree_util.tree_map(lambda _: False, model)
    return jax.tree_util.tree_map(visit, model, mask, is_leaf=_is_lora)


def merge_lora(model):
    return jax.tree_util.tree_map(
        lambda m: m.merged() if _is_lora(m) else m, model, is_leaf=_is_lora
    )
