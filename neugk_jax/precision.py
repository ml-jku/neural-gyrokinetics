"""Mixed-precision policy.

Three dtypes are tracked:

* ``param_dtype`` — parameter storage dtype. Always ``float32``: optimizer
  states (mu/nu) need fp32 to remain stable; bf16 lacks enough mantissa for
  long training runs.
* ``compute_dtype`` — activation dtype. ``bfloat16`` during training (model
  forward cast at the boundary), ``float32`` for eval.
* ``integral_dtype`` — flux/potential integrals. **Always ``float64``** —
  these are physical quantities computed once per eval step and we don't
  trade their accuracy for speed.

Modules are dtype-agnostic for activations: they compute in the dtype of
their inputs. ``Linear`` casts the weight to the input's dtype on the fly,
so parameters stay in fp32 while compute runs in bf16. ``LayerNorm`` always
internally upcasts to fp32 around the variance computation (numerically
safer; matches the torch convention) and casts back to the input dtype on
output.

The policy is global, with a contextmanager for scoped overrides.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class Policy:
    param_dtype: jnp.dtype = jnp.float32
    compute_dtype: jnp.dtype = jnp.float32
    integral_dtype: jnp.dtype = jnp.float64

    def cast_compute(self, x):
        if x is None:
            return None
        return x.astype(self.compute_dtype)

    def cast_param(self, x):
        if x is None:
            return None
        return x.astype(self.param_dtype)


_POLICY: Policy = Policy()


def get_policy() -> Policy:
    return _POLICY


def set_policy(policy: Policy) -> None:
    global _POLICY
    _POLICY = policy


@contextlib.contextmanager
def use_policy(policy: Policy):
    global _POLICY
    old = _POLICY
    _POLICY = policy
    try:
        yield policy
    finally:
        _POLICY = old


def bf16_training_policy() -> Policy:
    """Bf16 activations, fp32 params, f64 integrals."""
    return Policy(
        param_dtype=jnp.float32,
        compute_dtype=jnp.bfloat16,
        integral_dtype=jnp.float64,
    )


def fp32_policy() -> Policy:
    return Policy(
        param_dtype=jnp.float32,
        compute_dtype=jnp.float32,
        integral_dtype=jnp.float64,
    )


def cast_to_compute(tree):
    """Cast every float array in a pytree to the active compute dtype."""
    p = _POLICY
    def _cast(x):
        if isinstance(x, jax.Array) and jnp.issubdtype(x.dtype, jnp.floating):
            return x.astype(p.compute_dtype)
        return x
    return jax.tree_util.tree_map(_cast, tree)


def cast_to_param(tree):
    """Cast every float array in a pytree to the active param dtype (fp32)."""
    p = _POLICY
    def _cast(x):
        if isinstance(x, jax.Array) and jnp.issubdtype(x.dtype, jnp.floating):
            return x.astype(p.param_dtype)
        return x
    return jax.tree_util.tree_map(_cast, tree)
