"""Latent rectified flow matching (Gaussian prior, continuous time, OT-coupled).

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

from neugk_jax.losses import masked_mean, per_sample_mse


def sample_prior(key, shape, dtype=jnp.float32):
    return jr.normal(key, shape, dtype=dtype)


def sample_time(key, batch: int, dtype=jnp.float32):
    # t ~ sigmoid(n(0, 1))
    return jax.nn.sigmoid(jr.normal(key, (batch,), dtype=dtype))


def minibatch_ot(x0: jnp.ndarray, x1: jnp.ndarray) -> jnp.ndarray:
    """Optimal-transport coupling of ``x0`` and ``x1`` across the batch axis.

    Uses scipy's Hungarian algorithm via ``jax.pure_callback``, so the coupling
    also works inside a jit'd training step — fine for the small batch sizes flow
    matching typically uses.

    ``scipy.optimize.linear_sum_assignment`` returns ``(row_ind, col_ind)``
    where ``row_ind`` is the identity permutation for a square cost matrix,
    so the OT pairing is ``x0[i] <-> x1[col_ind[i]]``. Re-ordering ``x0`` so
    that ``x0_new[i]`` is the match for ``x1[i]`` gives ``x0[argsort(col_ind)]``.
    """
    bs = x0.shape[0]
    x0_flat = x0.reshape(bs, -1)
    x1_flat = x1.reshape(bs, -1)
    # ||a-b||^2 = |a|^2 + |b|^2 - 2a.b: one gemm, no (B, B, D) intermediate
    sq0 = jnp.sum(x0_flat**2, axis=-1)
    sq1 = jnp.sum(x1_flat**2, axis=-1)
    cost = jnp.sqrt(jnp.maximum(sq0[:, None] + sq1[None, :] - 2.0 * (x0_flat @ x1_flat.T), 0.0))

    def _assign(cost_np):
        import scipy.optimize

        _, col = scipy.optimize.linear_sum_assignment(np.asarray(cost_np))
        return np.argsort(col).astype(np.int32)

    perm = jax.pure_callback(_assign, jax.ShapeDtypeStruct((bs,), jnp.int32), cost)
    return x0[perm]


def fm_forward_loss(
    model_fn: Callable,
    latents: jnp.ndarray,
    cond: Optional[jnp.ndarray],
    *,
    key,
    latent_scale: float = 1.0,
    use_ot: bool = True,
    dropout_key=None,
    mask: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """One flow-matching training step (returns the scalar loss).

    ``model_fn(xt, t_scalar, cond_per_sample)`` is the *per-sample* DiT
    forward — caller vmaps the model over the batch. With ``dropout_key`` it is called as
    ``model_fn(xt, t, cond, key)`` with one key per sample. ``mask`` (``(B,)``) averages
    the loss over the rows it marks.
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
    args = (xt, t) if cond is None else (xt, t, cond)
    if dropout_key is not None:
        args = (*args, jr.split(dropout_key, bs))
    pred = jax.vmap(model_fn)(*args)
    if mask is None:
        return jnp.mean((pred - target_v) ** 2)
    return masked_mean(per_sample_mse(pred, target_v), mask)


def dit_flow_loss(model, z, cond, key, *, latent_scale, use_ot, train: bool, mask=None):
    """:func:`fm_forward_loss` of a DiT ``model(x, t, cond, key=, inference=)`` on latents ``z``.

    ``train`` turns on dropout/drop-path with per-sample keys split off ``key``.
    """
    fm_key, drop_key = jr.split(key)

    def fwd(x, t, *rest):
        c = rest[0] if cond is not None else None
        k = rest[-1] if train else None
        return model(x, t, c, key=k, inference=not train)

    return fm_forward_loss(
        fwd,
        z,
        cond,
        key=fm_key,
        latent_scale=latent_scale,
        use_ot=use_ot,
        dropout_key=drop_key if train else None,
        mask=mask,
    )


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
    """Euler-integrate the velocity field over ``[0, 1]``.

    Fused as a single ``jax.lax.scan`` so the whole sampling roll-out is
    one jit'd kernel. ``shape = (B, *latent_grid, z_dim)`` matches the
    encoder's output. Returns a sample in the data scale (divides by
    ``latent_scale`` at the end to undo the encoder's whitening).
    """
    bs = shape[0]
    x0 = sample_prior(key, shape, dtype=dtype)
    t_grid = jnp.linspace(0.0, 1.0, steps + 1, dtype=dtype)
    dts = t_grid[1:] - t_grid[:-1]
    ts = t_grid[:-1]

    if cond is not None:

        def velocity(x, ti):
            return jax.vmap(model_fn)(x, jnp.full((bs,), ti, dtype=dtype), cond)
    else:

        def velocity(x, ti):
            return jax.vmap(model_fn)(x, jnp.full((bs,), ti, dtype=dtype))

    def step(x, ti_dti):
        ti, dti = ti_dti
        return x + velocity(x, ti) * dti, None

    x, _ = jax.lax.scan(step, x0, (ts, dts))
    return x / latent_scale
