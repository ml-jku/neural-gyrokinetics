"""PINN-residual baseline for neural-field compression.

Alternative to train_pinc: instead of the PINC integral/spectral diagnostic
losses, train the neural field on the gyrokinetic-equation residual -- apply the
gyrokinetic RHS operator (JAX solver `gyaradax`) to the reconstruction and match
it to the RHS of the ground truth -- plus a Sobolev (derivative-matching) loss.

gyaradax is JAX, the neural field is torch. The whole complex chain
(real df -> complex spectral -> RHS -> residual scalar) runs in JAX and is
exposed to torch as a REAL-valued autograd.Function via dlpack (zero-copy) +
jax.vjp. Keeping the complex math in JAX also sidesteps the Blackwell (sm_100)
nvrtc failure on torch complex autograd.
"""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".2")

from copy import deepcopy

import torch
import torch.nn.functional as F

# gyaradax (JAX) is an OPTIONAL dependency, only needed for the PINN baseline.
# Import it at module load (not lazily) because torch.compile, run during the
# density warmup, hooks the import machinery and shadows a later import. If it
# is absent, this module still imports; the PINN entry points raise on use.
try:
    import gyaradax  # noqa: F401  (configures jax x64 on import)
    import jax
    import jax.numpy as jnp
    from gyaradax import gk_from_gkw_dir
    from gyaradax.solver import g_to_f, _compute_fields
    from gyaradax.backends import create_ops

    _GYARADAX_ERR = None
except Exception as _e:  # ImportError, or a JAX/CUDA init failure
    jax = jnp = None
    _GYARADAX_ERR = _e

from neugk.pinc.neural_fields.nf_utils import sample_field


def _require_gyaradax():
    if _GYARADAX_ERR is not None:
        raise ImportError(
            "The PINN-residual baseline needs gyaradax (+jax). Install it from "
            "https://github.com/gerkone/gyaradax. Original import error: "
            f"{_GYARADAX_ERR!r}"
        )

RAW_ROOT = "/restricteddata/ukaea/gyrokinetics/raw"


def _resolve_raw_dir(trajectory: str) -> str:
    name = trajectory.replace(".h5", "").split("_ifft")[0].split("_Lin")[0]
    for cand in (f"{RAW_ROOT}/{name}", f"{RAW_ROOT}/{name}_Lin"):
        if os.path.isfile(os.path.join(cand, "input.dat")):
            return cand
    raise FileNotFoundError(f"no GKW run dir for {trajectory} under {RAW_ROOT}")


def load_gk_rhs(trajectory: str):
    """Gyrokinetic RHS operator for a trajectory: rhs(f_spectral_jax) -> jax array."""
    _require_gyaradax()
    d = _resolve_raw_dir(trajectory)
    _, geometry, params, _, pre = gk_from_gkw_dir(d)
    ops = create_ops(
        pre, backend=params.backend,
        use_z2z=getattr(params, "use_z2z", False),
        mixed_precision=getattr(params, "mixed_precision", False),
    )

    def rhs(dg):  # mirrors solver.gkstep_single._rhs (linear; +nonlinear if enabled)
        phi, apar, bpar = _compute_fields(dg, geometry, params, pre)
        df = g_to_f(dg, apar, params, pre) if apar is not None else dg
        r = ops.linear_rhs(df, phi, geometry, params, pre, apar=apar, bpar=bpar)
        if getattr(params, "non_linear", False):
            r = r + ops.nonlinear_term_iii(dg, phi, geometry)
        return r

    return jax.jit(rhs), {"dir": d, "geometry": geometry, "params": params, "pre": pre}


def _to_spectral_jax(df_real):
    """JAX: NF real-space df (2, vpar, mu, s, x, y) -> complex spectral
    (vpar, mu, s, kx, ky), matching FluxIntegral._df_fft (the validated path)."""
    z = (df_real[0] + 1j * df_real[1]).astype(jnp.complex128)
    z = jnp.fft.fftn(z, axes=(-2, -1), norm="forward")
    return jnp.fft.ifftshift(z, axes=(-2,))


def make_residual_loss(rhs, gt_df):
    """Relative gyrokinetic-RHS residual loss in JAX. gt_df: torch (2,...) real.
    Returns loss_jax(df_real_jax) -> real scalar, with rhs(gt) precomputed."""
    gt_j = jax.dlpack.from_dlpack(gt_df.detach().contiguous())
    rhs_gt = rhs(_to_spectral_jax(gt_j))
    denom = jnp.real(jnp.vdot(rhs_gt, rhs_gt)) + 1e-30

    def loss_jax(df_real):
        diff = rhs(_to_spectral_jax(df_real)) - rhs_gt
        return jnp.real(jnp.vdot(diff, diff)) / denom

    return jax.jit(loss_jax)


class _ResidualLoss(torch.autograd.Function):
    """Wrap the JAX real-scalar residual loss as a torch op (jax.vjp backward)."""
    @staticmethod
    def forward(ctx, df, loss_jax):
        v, ctx.vjp = jax.vjp(loss_jax, jax.dlpack.from_dlpack(df.detach().contiguous()))
        return torch.from_dlpack(v).clone()

    @staticmethod
    def backward(ctx, g):
        (gx,) = ctx.vjp(jnp.asarray(g.detach().item()))
        return torch.from_dlpack(gx).clone().to(g.dtype), None


def sobolev_loss(pred, gt):
    """H1 Sobolev loss: value MSE + first-derivative MSE over all field axes."""
    loss = F.mse_loss(pred, gt)
    for d in range(1, pred.ndim):  # skip channel axis 0
        loss = loss + F.mse_loss(torch.diff(pred, dim=d), torch.diff(gt, dim=d))
    return loss


def train_pinn(
    model, n_epochs, data, optim, sched, device, trajectory,
    w_sobolev=1.0, w_residual=1.0, eval_every=2, use_print=True,
):
    """PINN baseline: train the NF on the gyrokinetic RHS residual + Sobolev loss
    (no PINC integral/spectral losses). One snapshot (per-trajectory operator)."""
    rhs, meta = load_gk_rhs(trajectory)
    data.to(device)
    model.to(device)
    gt_df = data.full_df.to(device).double()
    loss_jax = make_residual_loss(rhs, gt_df)
    best_loss, best_model, best_e, losses = torch.inf, None, 0, []
    for e in range(n_epochs):
        model.train()
        pred = sample_field(model, data, device).double()
        res = _ResidualLoss.apply(pred, loss_jax)
        sob = sobolev_loss(pred, gt_df)
        loss = w_sobolev * sob + w_residual * res
        optim.zero_grad()
        loss.backward()
        optim.step()
        if sched is not None:
            sched.step()
        rec = {"train/loss": float(loss), "train/sobolev": float(sob), "train/residual": float(res)}
        if eval_every > 0 and (e % eval_every == 0 or e == n_epochs - 1):
            with torch.no_grad():
                df_l1 = float((sample_field(model, data, device).double() - gt_df).abs().mean())
            rec["val/df_l1"] = df_l1
            if df_l1 < best_loss:
                best_loss, best_e, best_model = df_l1, e, deepcopy(model)
        losses.append(rec)
        if use_print:
            print(f"[{e}] " + ", ".join(f"{k}: {v:.5f}" for k, v in rec.items()))
    if best_model is None:
        best_model, best_e = deepcopy(model), n_epochs - 1
    return model, best_model, losses, best_e
