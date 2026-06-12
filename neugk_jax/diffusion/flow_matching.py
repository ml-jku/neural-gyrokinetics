"""Latent rectified flow matching (Gaussian prior, continuous time, OT-coupled).

Mirrors ``FlowMatchingRunner`` in the upstream torch repo
(``neugk/diffusion/run.py:552-617``):

* ``x0 ~ N(0, I)``
* ``x1 = encoded_df * latent_scale``
* ``t ~ sigmoid(N(0, 1))`` (continuous time path)
* optional minibatch optimal transport (Hungarian) couples ``x0`` and
  ``x1`` across the batch before training
* ``xt = t * x1 + (1 - t) * x0`` and ``v_target = x1 - x0``
* train: ``MSE(model(xt, t, cond), v_target)``
* sample: Euler integration over ``[0, 1]`` of the learned velocity field

The flow-matching loss/training is independent of the autoencoder
specifics — caller passes the latent batch directly.
"""

from __future__ import annotations

from typing import Callable, Optional

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np


def sample_prior(key, shape, dtype=jnp.float32):
    """Gaussian prior x0."""
    return jr.normal(key, shape, dtype=dtype)


def sample_time(key, batch: int, dtype=jnp.float32):
    """``t ~ sigmoid(N(0, 1))`` — biases mass toward the middle of [0, 1]."""
    return jax.nn.sigmoid(jr.normal(key, (batch,), dtype=dtype))


def minibatch_ot(x0: jnp.ndarray, x1: jnp.ndarray) -> jnp.ndarray:
    """Optimal-transport coupling of ``x0`` and ``x1`` across the batch axis.

    Uses scipy's Hungarian algorithm via a host callback — fine for the
    small batch sizes flow matching typically uses (upstream torch does
    the same scipy call).

    Bug fix vs upstream: ``scipy.optimize.linear_sum_assignment`` returns
    ``(row_ind, col_ind)`` where ``row_ind`` is always the identity
    permutation for a square cost matrix. Upstream's ``x0[row_ind]`` is
    therefore a no-op, silently disabling OT coupling during training.
    The correct permutation is over ``col_ind``: pair ``x0[i]`` with
    ``x1[col_ind[i]]``. Equivalently we re-order ``x0`` so that the
    returned ``x0_new[i]`` is the OT-match for the original ``x1[i]``,
    which is ``x0[argsort(col_ind)]``.
    """
    import scipy.optimize
    bs = x0.shape[0]
    x0_flat = x0.reshape(bs, -1)
    x1_flat = x1.reshape(bs, -1)
    cost = jnp.linalg.norm(x0_flat[:, None, :] - x1_flat[None, :, :], axis=-1)
    cost_np = np.asarray(cost)
    _, col = scipy.optimize.linear_sum_assignment(cost_np)
    perm = np.argsort(col)
    return x0[jnp.asarray(perm)]


def fm_forward_loss(
    model_fn: Callable,
    latents: jnp.ndarray,
    cond: Optional[jnp.ndarray],
    *,
    key,
    latent_scale: float = 1.0,
    use_ot: bool = True,
) -> jnp.ndarray:
    """One flow-matching training step (returns the scalar loss).

    ``model_fn(xt, t_scalar, cond_per_sample)`` is the *per-sample* DiT
    forward — caller vmaps the model over the batch.
    """
    bs = latents.shape[0]
    k_prior, k_t = jr.split(key, 2)
    x1 = latents * latent_scale
    x0 = sample_prior(k_prior, x1.shape, dtype=x1.dtype)
    if use_ot:
        x0 = minibatch_ot(x0, x1)
    t = sample_time(k_t, bs, dtype=x1.dtype)
    t_b = t.reshape(-1, *[1] * (x1.ndim - 1))
    xt = t_b * x1 + (1.0 - t_b) * x0
    target_v = x1 - x0
    # vmap over the batch — model_fn is per-sample
    pred = jax.vmap(model_fn)(xt, t, cond) if cond is not None else jax.vmap(model_fn)(xt, t)
    return jnp.mean((pred - target_v) ** 2)


def euler_sample(
    model_fn: Callable,
    *,
    key,
    shape: tuple[int, ...],
    cond: Optional[jnp.ndarray] = None,
    steps: int = 10,
    latent_scale: float = 1.0,
    dtype=jnp.float32,
) -> jnp.ndarray:
    """Integrate the velocity field over ``[0, 1]`` with Euler steps.

    Fused as a single ``jax.lax.scan`` so the whole sampling roll-out is
    one jit'd kernel — avoids the per-step host-device sync the previous
    Python ``for`` loop incurred. ``shape = (B, *latent_grid, z_dim)``
    matches the encoder's output. Returns a sample in the data scale
    (divides by ``latent_scale`` at the end to undo the encoder's whitening).
    """
    bs = shape[0]
    x0 = sample_prior(key, shape, dtype=dtype)
    t_grid = jnp.linspace(0.0, 1.0, steps + 1, dtype=dtype)
    dts = t_grid[1:] - t_grid[:-1]
    ts = t_grid[:-1]

    if cond is not None:
        def step(x, ti_dti):
            ti, dti = ti_dti
            t = jnp.full((bs,), ti, dtype=dtype)
            v = jax.vmap(model_fn)(x, t, cond)
            return x + v * dti, None
    else:
        def step(x, ti_dti):
            ti, dti = ti_dti
            t = jnp.full((bs,), ti, dtype=dtype)
            v = jax.vmap(model_fn)(x, t)
            return x + v * dti, None

    x, _ = jax.lax.scan(step, x0, (ts, dts))
    return x / latent_scale
