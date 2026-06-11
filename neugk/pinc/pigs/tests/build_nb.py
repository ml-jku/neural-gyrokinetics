"""Build notebooks/06_pigs.ipynb (clean JSON). Run locally (no GPU)."""
import json
def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(keepends=True)}
C = []

C.append(md(r"""
# PIGS: Physics-Inspired Gaussian Splats

Compression of one 5D gyrokinetic snapshot into a fixed Gaussian budget. The method **ladder** (each rung
adds one verified ingredient):

| rung | call | what it adds |
|---|---|---|
| **base** | `compress_base` | VANILLA Gaussian splatting: joint AdamW MSE on minibatches — no warm-start, no separable trick. The slow reference we started from. |
| **fast** | `compress_fast` | amp warm-start (amplitudes are linear → closed-form LS) + separable warmup (Huber) + fused polish. Density only. |
| **gPINC** | `compress_gpinc` | fast + physics fine-tune (df/φ/spectra losses) on the **separable** reconstruction (~10× faster than dense, bit-identical losses). Recommended reconstruction model. |
| **PINC** | `compress_pinc` | fast + the SAME physics fine-tune but **dense** — same quality, kept only as the speed reference for gPINC. |
| **PIGS** | `compress_pigs` | gPINC + flux graft + frozen-base flux refine (POST). Production graft = **tied carrier groups**: stratified-coverage envelopes each carrying K=9 carriers at the tail k_y with shared μ/Σ (~3.7× carriers/byte; measured ~20× better flux than free carriers). Budget-neutral: `n_total = n_base + n_flux` atom-equivalents. |

All models are returned **fp16-rounded** (on-disk-fair eval) with timing in `info`.
- **§1 Quick start** — run the ladder, results table, diagnostic plots.
- **§2 Under the hood + ablations** — stage rationale; CR sweep (how far can compression go), warmup steps,
  gPINC-vs-PINC speed, flux_lambda Pareto, flux_frac budget split.
"""))

C.append(md("## §1 — Setup & data"))
C.append(code(r"""
%load_ext autoreload
%autoreload 2
import os, sys
sys.path.append("..")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda"
"""))
C.append(code(r"""
%matplotlib inline
import torch, numpy as np, pandas as pd, matplotlib.pyplot as plt
import neugk.pinc.pigs as pigs
from neugk.pinc.neural_fields import CycloneNFDataset

DATA_PATH = "/projects/u6eb/gyrokinetics/preprocessed_kvikio"
def load(traj="iteration_13", t=100):
    d = CycloneNFDataset(traj, timesteps=t, path=DATA_PATH, backend="kvikio",
                         realpotens=True, normalize="zscore", normalize_coords=True)
    d.to(torch.device(device)); return d
data = load()
print("snapshot bytes:", data.full_df.nbytes, " grid:", tuple(data.df.shape))
"""))

C.append(md(r"""
## Run the ladder
`n=1000` atoms ≈ CR 2786× at fp16 (`pigs.n_for_cr(data, cr)` inverts a target CR to N). `seed=0` makes runs
reproducible (it also resets the loader's in-place shuffle hysteresis).
"""))
C.append(code(r"""
n_total = 1000

m_base,  info_base  = pigs.compress_base (data, n_total, device, seed=0)   # vanilla (slow reference)
m_fast,  info_fast  = pigs.compress_fast (data, n_total, device, seed=0)   # fast density
m_gpinc, info_gpinc = pigs.compress_gpinc(data, n_total, device, seed=0)   # fast + gPINC (recommended)
m_pinc,  info_pinc  = pigs.compress_pinc (data, n_total, device, seed=0)   # fast + dense PINC (speed ref, ~10 min)
m_pigs,  info_pigs  = pigs.compress_pigs (data, n_total=n_total, flux_lambda=3.0, device=device, seed=0)
"""))
C.append(md(r"""
## Results table — the ladder at one glance
**Metric note — `flux_relL1`**: relative L1 of the **k_y-resolved heat-flux spectrum** `Q(k_y)` (the eflux
field integrated over velocity space and (s,x), per binormal mode, band k_y 1–15). It is NOT the eflux-field
error and NOT the single scalar total heat flux (the scalar sign-cancels and is degenerate — it can stay ~1
while the spectrum is perfect, and vice versa). `flux_dom` = band 1–6, `flux_tail` = band 7–15. **`eflux_int_rel`** = relative error of the
INTEGRATED scalar heat flux Q_tot=Σ_ky Q(ky) — the physical flux value (signed; can cancel, so read it
together with `flux_relL1`, not instead of it).
PSNR higher = better; flux_* lower = better.
"""))
C.append(code(r"""
results = {
    "base (vanilla GS)":   {**info_base,  **pigs.evaluate(m_base,  data, device)},
    "fast (density)":      {**info_fast,  **pigs.evaluate(m_fast,  data, device)},
    "gPINC (separable)":   {**info_gpinc, **pigs.evaluate(m_gpinc, data, device)},
    "PINC (dense, ref)":   {**info_pinc,  **pigs.evaluate(m_pinc,  data, device)},
    "PIGS (full)":         {**info_pigs,  **pigs.evaluate(m_pigs,  data, device)},
}
df = pd.DataFrame(results).T
cols = ["CR", "time_s", "PSNR(f)", "PSNR(phi)", "kyspec L1", "qspec L1", "flux_dom", "flux_tail", "flux_relL1", "eflux_int_rel"]
df[[c for c in cols if c in df.columns]]
"""))
C.append(md(r"""
Expected reads: **base→fast** = +PSNR(f) at a fraction of the time (warm-start + separable);
**fast→gPINC** = φ goes from ~−4 to ~+18 (physics losses); **gPINC vs PINC** = same quality, ~10× time gap;
**gPINC→PIGS** = flux_relL1 drops several× (the graft + POST), small φ/f cost.
"""))

C.append(md("## Visualizations — GT vs reconstruction"))
C.append(code(r"""
with torch.no_grad():
    f_base  = pigs.reconstruct(m_base,  data, device)
    f_gpinc = pigs.reconstruct(m_gpinc, data, device)
    f_pigs  = pigs.reconstruct(m_pigs,  data, device)
gt = data.full_df.to(device)
"""))
C.append(code(r"""
from neugk.pinc.neural_fields.nf_train import eval_diagnose
fig_df, fig_eflux, fig_potens, fig_diag = eval_diagnose(data, torch.device(device), pred_df=f_pigs)
fig_df.suptitle("distribution f — GT vs PIGS")
fig_potens.suptitle("potential phi — GT vs PIGS")
fig_diag.suptitle("k_y / k_x / q spectra — GT vs PIGS")
plt.show()
"""))
C.append(code(r"""
# Spectral comparison across rungs. df is COMPLEX (the 2 channels are Re/Im), so the spectrum is the
# full complex FFT -> all 32 distinct ky modes (an rfft of the real channels would only show 17).
from neugk.pinc.neural_fields.nf_utils import to_complex
fig, axes = plt.subplots(1, 2, figsize=(15, 4))
methods = [("GT", gt), ("base", f_base), ("gPINC", f_gpinc), ("PIGS", f_pigs)]
colors = ["black", "blue", "green", "red"]
for ax, dim, mean_dims, name in [(axes[0], -1, (0, 1, 2, 3), "kyspec (binormal, 32 modes)"),
                                 (axes[1], -2, (0, 1, 2, 4), "kxspec (radial)")]:
    for (mn, field), c in zip(methods, colors):
        spec = torch.fft.fft(to_complex(field), dim=dim).abs().mean(dim=mean_dims).cpu().numpy()
        ax.semilogy(np.arange(len(spec)), spec + 1e-8, label=mn, color=c, lw=1.5)
    ax.set_title(name); ax.set_xlabel("bin"); ax.grid(alpha=0.3); ax.legend()
axes[0].axvline(7, color="gray", ls="--", alpha=0.4)
plt.show()
"""))

C.append(md(r"""
## §2 — Under the hood

### What each stage does (and why it works)
- **base** (`train_default`): importance-sampled init + joint AdamW MSE on shuffled voxel batches. Slow
  because every step touches a 200k-voxel batch through the dense forward; no structure exploited.
- **fast** (`train_fast`): (i) the field is **linear in the amplitudes** → closed-form least-squares
  warm-start; (ii) **block-diagonal Σ** → each Gaussian factorizes over (velocity × space) → one exact
  full-grid step is a GEMM (~17 ms vs ~6 s dense); (iii) short fused polish. **Huber** loss preserves the
  subdominant modes that energy-greedy MSE starves.
- **gPINC / PINC** (`train_pinc`): physics multi-loss (df, flux, φ, kyspec, qspec, monotonicity).
  `mode='gpinc'` evaluates it on the separable reconstruction — **bit-identical losses, ~10× faster**;
  `mode='dense'` is the same thing through the dense forward (the speed reference).
- **PIGS graft + POST** (`tied_centers` + `add_atoms` + `refine_flux`): the heat flux `Q(ky)=Im(df_k·conj(φ_k))`
  lives at tail ky the plain Gaussians can't reach (a real Gaussian's y-spectrum sits at ky=0; one Gabor
  carrier = one independent per-bin cross-phase DOF). Production graft = **tied carrier groups**: envelopes
  placed for **coverage** (stratified over the grid — measured ~6× better flux and +4–6 dB φ than residual
  hotspots), each carrying K=9 carriers at the tail bins with **shared μ/Σ** (14+3K params vs 17K → ~3.7×
  carriers per byte → ~20× better flux at the same bytes). Complex amps warm-started in closed form; then
  **freeze the φ-optimal base** and POST-refine the carrier amps only (tying + on-bin ky stay exact):
  `raw` (match Q, min total flux) or `wnorm` (match W=Q/|φ|², φ-preserving tail-only); `flux_lambda` slides
  the Pareto. `tied=False` gives the older free-carrier graft.
"""))

C.append(md(r"""
## Ablations

### A1 — How far can compression go? (CR sweep)
Quality vs compression ratio for the recommended model (`compress_gpinc`). Low CR = many Gaussians (slow,
heavy): for N > 6000 we skip the fused polish (`polish_epochs=0`) to avoid the (B,N) memory blow-up.
"""))
C.append(code(r"""
CRS = [4000, 2000, 1000, 500, 250, 100]     # 100x is heavy (~28k atoms) -- drop it for a quick look
cr_rows = []
for cr in CRS:
    n = pigs.n_for_cr(data, cr)
    kw = dict(polish_epochs=0) if n > 6000 else {}
    m, info = pigs.compress_gpinc(data, n, device, seed=0, verbose=False, **kw)
    e = pigs.evaluate(m, data, device)
    cr_rows.append({"target_CR": cr, "n": n, "CR": info["CR"], "t_s": info["time_s"],
                    **{k: round(v, 3) for k, v in e.items()}})
    print(f"CR={cr:5d} n={n:6d} t={info['time_s']:6.1f}s f={e['PSNR(f)']:.2f} phi={e['PSNR(phi)']:.2f} "
          f"flux={e['flux_relL1']:.3f}")
df_cr = pd.DataFrame(cr_rows); df_cr
"""))
C.append(code(r"""
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].semilogx(df_cr["CR"], df_cr["PSNR(f)"], "-o", label="PSNR(f)")
ax[0].semilogx(df_cr["CR"], df_cr["PSNR(phi)"], "-s", label="PSNR(phi)")
ax[0].set_xlabel("compression ratio (log)"); ax[0].set_ylabel("PSNR [dB]"); ax[0].legend(); ax[0].grid(alpha=0.3)
ax[0].set_title("reconstruction vs compression")
ax[1].loglog(df_cr["CR"], df_cr["flux_relL1"], "-o", color="tab:red")
ax[1].set_xlabel("compression ratio (log)"); ax[1].set_ylabel("flux_relL1 (lower=better)"); ax[1].grid(alpha=0.3)
ax[1].set_title("heat flux vs compression")
plt.tight_layout(); plt.show()
"""))

C.append(md(r"""
### A2 — Density warmup steps (real sweep via kwargs passthrough)
`compress_fast(..., warmup_steps=s)` forwards to `train_fast`. The curve flattens ~step 500–600.
"""))
C.append(code(r"""
abl = []
for s in [100, 250, 500, 1000, 2000]:
    m, info = pigs.compress_fast(data, n_total, device, warmup_steps=s, seed=0, verbose=False)
    e = pigs.evaluate(m, data, device)
    abl.append({"warmup_steps": s, "t_s": info["time_s"], "PSNR(f)": round(e["PSNR(f)"], 2)})
    print(f"steps={s:5d}: t={info['time_s']:5.1f}s f={e['PSNR(f)']:.2f}dB")
pd.DataFrame(abl)
"""))

C.append(md(r"""
### A3 — gPINC vs dense PINC (the speed reference)
Same physics losses, same epochs — only the reconstruction path differs (already run in §1):
"""))
C.append(code(r"""
pd.DataFrame({
    "gPINC (separable)": {"time_s": info_gpinc["time_s"], **{k: round(v,3) for k,v in pigs.evaluate(m_gpinc, data, device).items()}},
    "PINC (dense)":      {"time_s": info_pinc["time_s"],  **{k: round(v,3) for k,v in pigs.evaluate(m_pinc, data, device).items()}},
}).T
"""))

C.append(md(r"""
### A4 — PIGS flux_lambda Pareto (raw vs wnorm)
Sweep the POST knob on the PRODUCTION (tied) recipe. `seed=0` makes the rows share identical base/graft
randomness, so the differences are purely the objective.
"""))
C.append(code(r"""
rows = []
for mode, lams in [("raw", [1.0, 3.0, 10.0]), ("wnorm", [1.0, 3.0])]:
    for L in lams:
        m, info = pigs.compress_pigs(data, n_total=n_total, flux_mode=mode, flux_lambda=L,
                                     device=device, seed=0, verbose=False)
        e = pigs.evaluate(m, data, device)
        rows.append({"mode": mode, "lambda": L, "t_s": info["time_s"],
                     "PSNR(phi)": round(e["PSNR(phi)"], 2), "flux_tail": e["flux_tail"], "flux_all": e["flux_relL1"]})
        print(rows[-1])
dfp = pd.DataFrame(rows); dfp
"""))
C.append(code(r"""
fig, ax = plt.subplots(figsize=(6.5, 4.5))
for mode, mk, c in [("raw", "o", "tab:green"), ("wnorm", "s", "tab:blue")]:
    s = dfp[dfp["mode"] == mode]; ax.plot(s["flux_all"], s["PSNR(phi)"], "-" + mk, c=c, label=mode)
ax.set_xlabel("flux_all (lower=better)"); ax.set_ylabel("PSNR(phi) [dB]")
ax.set_title("phi-vs-flux Pareto: raw = min flux, wnorm = phi-preserving tail-only")
ax.legend(); ax.grid(alpha=0.3); plt.show()
"""))

C.append(md(r"""
### A5 — Budget split (flux_frac): how many atoms to reserve for flux?
"""))
C.append(code(r"""
frac_rows = []
for frac in [0.1, 0.25, 0.5]:
    m, info = pigs.compress_pigs(data, n_total=n_total, flux_frac=frac, device=device, seed=0, verbose=False)
    e = pigs.evaluate(m, data, device)
    frac_rows.append({"flux_frac": frac, "n_base": info["n_base"], "n_flux": info["n_flux"],
                      "t_s": info["time_s"], "PSNR(f)": round(e["PSNR(f)"], 2),
                      "PSNR(phi)": round(e["PSNR(phi)"], 2), "flux_relL1": round(e["flux_relL1"], 4)})
    print(frac_rows[-1])
pd.DataFrame(frac_rows)
"""))

C.append(md(r"""
## Notes
- **Self-contained**: `neugk.pinc.pigs` depends only on the shared `neugk` data loader + physics losses
  (no prior gsplat iterations).
- **PIGS = density → gPINC → flux graft → POST refine** (no hand-in-hand stage — measured: the dedicated
  frozen-base refine is what delivers the flux; see `gsplat_speed/RESULTS.md` for the full study).
- **gPINC speed**: the separable reconstruction is bit-identical through the FFT physics losses, ~10× faster.
- **Budget-neutral flux**: at fixed `n_total`, the flux atoms replace Gaussians (file size unchanged;
  each Gabor atom stores 17 params vs 16 — the +1 is its carrier ky).
- **fp16**: all returned models are fp16-rounded (verified ~lossless, even for the flux cross-phase), so
  `evaluate()` reflects what is on disk.
"""))

nb = {"cells": C, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = "/home/u6eb/gutenbru.u6eb/plasmamodelling/notebooks/06_pigs.ipynb"
json.dump(nb, open(out, "w"), indent=1)
print("wrote", out, "cells:", len(C))
