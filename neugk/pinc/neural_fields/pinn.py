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
from functools import lru_cache

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".2")

from copy import deepcopy

import torch
import torch.nn.functional as F

from neugk.pinc.neural_fields.nf_utils import sample_field

# gyaradax (JAX) optional; imported lazily to avoid half-init after torch.compile
jax = jnp = None
gk_from_gkw_dir = create_ops = _compute_fields = g_to_f = None
_GYARADAX_ERR = None
_GYARADAX_LOADED = False


def _load_gyaradax():
    global jax, jnp, gk_from_gkw_dir, create_ops, _compute_fields, g_to_f
    global _GYARADAX_ERR, _GYARADAX_LOADED
    if _GYARADAX_LOADED:
        return
    _GYARADAX_LOADED = True
    import sys

    # strip CWD/..-relative sys.path entries shadowing gyaradax editable install
    sys.modules.pop("gyaradax", None)
    bad = {os.path.abspath(p) for p in (".", "..", os.getcwd(),
                                        os.path.dirname(os.getcwd()))}
    saved_path = sys.path[:]
    sys.path[:] = [p for p in sys.path
                   if p not in ("", ".", "..") and os.path.abspath(p) not in bad]
    try:
        import jax as _jax
        import jax.numpy as _jnp
        from gyaradax import gk_from_gkw_dir as _gk
        from gyaradax.backends import create_ops as _ops
        from gyaradax.solver import _compute_fields as _cf, g_to_f as _g2f

        # persistent XLA cache on /tmp (not home, which filled 50GB); gyaradax bakes per-trajectory constants (~1GB each); 1s compile floor keeps only RHS kernel
        _jax.config.update(
            "jax_compilation_cache_dir",
            os.environ.get("PINN_JAX_CACHE", "/tmp/pinn_jax"),
        )
        _jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
        _jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
        jax, jnp = _jax, _jnp
        gk_from_gkw_dir, create_ops = _gk, _ops
        _compute_fields, g_to_f = _cf, _g2f
        _GYARADAX_ERR = None
    except Exception as e:  # ImportError, or a JAX/CUDA init failure
        import traceback

        _GYARADAX_ERR = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        # drop half-initialized modules so later retry (fresh worker) is clean
        for m in [k for k in sys.modules if k == "gyaradax" or k.startswith("gyaradax.")]:
            sys.modules.pop(m, None)
    finally:
        sys.path[:] = saved_path


def _require_gyaradax():
    _load_gyaradax()
    if _GYARADAX_ERR is not None:
        raise ImportError(
            "The PINN-residual baseline needs gyaradax (+jax). Install it from "
            "https://github.com/gerkone/gyaradax.\nOriginal import traceback:\n"
            f"{_GYARADAX_ERR}"
        )


RAW_ROOT = "/restricteddata/ukaea/gyrokinetics/raw"


def _resolve_raw_dir(trajectory: str) -> str:
    name = trajectory.replace(".h5", "").split("_ifft")[0].split("_Lin")[0]
    for cand in (f"{RAW_ROOT}/{name}", f"{RAW_ROOT}/{name}_Lin"):
        if os.path.isfile(os.path.join(cand, "input.dat")):
            return cand
    raise FileNotFoundError(f"no GKW run dir for {trajectory} under {RAW_ROOT}")


@lru_cache(maxsize=8)
def load_gk_rhs(trajectory: str):
    """Gyrokinetic RHS operator for a trajectory: rhs(f_spectral_jax) -> jax array; cached per trajectory; jit amortized over snapshots; persistent cache makes re-runs fast."""
    _require_gyaradax()
    d = _resolve_raw_dir(trajectory)
    _, geometry, params, _, pre = gk_from_gkw_dir(d)
    ops = create_ops(
        pre, backend=params.backend,
        use_z2z=getattr(params, "use_z2z", False),
        mixed_precision=getattr(params, "mixed_precision", False),
    )

    def rhs(dg):  # mirrors solver.gkstep_single._rhs: linear (+nonlinear if enabled)
        phi, apar, bpar = _compute_fields(dg, geometry, params, pre)
        df = g_to_f(dg, apar, params, pre) if apar is not None else dg
        r = ops.linear_rhs(df, phi, geometry, params, pre, apar=apar, bpar=bpar)
        if getattr(params, "non_linear", False):
            r = r + ops.nonlinear_term_iii(dg, phi, geometry)
        return r

    return jax.jit(rhs), {"dir": d, "geometry": geometry, "params": params, "pre": pre}


def _to_spectral_jax(df_real):
    """JAX: NF real-space df (2, vpar, mu, s, x, y) -> complex spectral (vpar, mu, s, kx, ky), matches FluxIntegral._df_fft."""
    z = (df_real[0] + 1j * df_real[1]).astype(jnp.complex128)
    z = jnp.fft.fftn(z, axes=(-2, -1), norm="forward")
    return jnp.fft.ifftshift(z, axes=(-2,))


@lru_cache(maxsize=8)
def _residual_loss_core(rhs):
    """jit(loss(df_real, rhs_gt, denom)); compiles once per trajectory, reused for all snapshots; per-snapshot args traced (not baked in HLO) for persistent cache."""
    def loss(df_real, rhs_gt, denom):
        diff = rhs(_to_spectral_jax(df_real)) - rhs_gt
        return jnp.real(jnp.vdot(diff, diff)) / denom

    return jax.jit(loss)


def make_residual_loss(rhs, gt_df):
    """Relative gyrokinetic-RHS residual loss in JAX; gt_df torch (2,...) real; returns loss_jax(df_real_jax) -> real scalar."""
    gt_j = jax.dlpack.from_dlpack(gt_df.detach().contiguous())
    rhs_gt = rhs(_to_spectral_jax(gt_j))
    denom = jnp.real(jnp.vdot(rhs_gt, rhs_gt)) + 1e-30
    core = _residual_loss_core(rhs)

    # bind per-snapshot ground truth as traced args for identical HLO per trajectory
    def loss_jax(df_real):
        return core(df_real, rhs_gt, denom)

    return loss_jax


class _ResidualLoss(torch.autograd.Function):
    """Wrap JAX real-scalar residual loss as torch op with jax.vjp backward."""

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
    model,
    n_epochs,
    data,
    optim,
    sched,
    device,
    trajectory,
    w_sobolev=1.0,
    w_residual=1.0,
    eval_every=2,
    use_print=True,
):
    """Train NF on gyrokinetic RHS residual + Sobolev loss (no PINC integral/spectral losses); per-trajectory operator."""
    from neugk.pinc.neural_fields.nf_train import nf_eval  # same val metrics as PINC/density

    _require_gyaradax()
    # pin all jax ops to same GPU as torch tensors; dlpack bridge mismatches otherwise; loss.backward covers it too
    idx = device.index if getattr(device, "index", None) is not None else 0
    with jax.default_device(jax.devices()[idx]):
        rhs, meta = load_gk_rhs(trajectory)
        data.to(device)
        model.to(device)
        gt_df = data.full_df.to(device).double()
        loss_jax = make_residual_loss(rhs, gt_df)
        best_psnr, best_model, best_e, losses = -torch.inf, None, 0, []
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
                # full physics eval (df/phi psnr, flux, spectra); matches train_pinc/density
                rec.update(
                    {
                        f"val/{k}": v
                        for k, v in nf_eval(model, data, device=device, use_flux_fields=False).items()
                    }
                )
                if rec["val/df psnr"] > best_psnr:
                    best_psnr, best_e, best_model = rec["val/df psnr"], e, deepcopy(model)
            losses.append(rec)
            if use_print:
                print(f"[{e}] " + ", ".join(f"{k}: {v:.5f}" for k, v in rec.items()))
        if best_model is None:
            best_model, best_e = deepcopy(model), n_epochs - 1
    return model, best_model, losses, best_e
