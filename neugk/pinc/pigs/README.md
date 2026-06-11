# pigs — Physics-Inspired Gaussian Splats (5D gyrokinetic compression)

Self-contained (only the shared `neugk` data loader + physics losses). One snapshot → a fixed budget of
anisotropic 5D Gaussians; the full model also gets the **heat flux** (the df–φ cross-phase) right.

## The method ladder
| rung | call | adds | typical (N≈1200, GH200) |
|---|---|---|---|
| **base** | `compress_base` | vanilla GS: joint AdamW MSE minibatches, no tricks | ~32 dB f, φ unsolved, ~90 s |
| **fast** | `compress_fast` | amp warm-start (amps are linear → closed-form LS) + separable warmup (Huber) + fused polish | ~36 dB f, ~30 s |
| **gPINC** | `compress_gpinc` | physics fine-tune (df/φ/spectra) on the **separable** reconstruction | φ ~18–21, +~70 s |
| **PINC** | `compress_pinc` | same fine-tune, **dense** — same quality; the ~10× speed reference | +~12 min |
| **PIGS** | `compress_pigs` | flux graft (**tied carrier groups**: stratified-coverage envelopes × K=9 tail carriers w/ shared μ/Σ, complex amp warm-start) + frozen-base **POST** flux refine (amps only) | flux_relL1 ~0.55→**~0.003–0.01** (seed 0) |

```python
import neugk.pinc.pigs as pigs
m, info = pigs.compress_pigs(data, n_total=1000, flux_lambda=3.0, seed=0)
pigs.evaluate(m, data, "cuda")          # PSNR(f)/PSNR(phi)/kyspec/qspec + flux dom/tail/all
n = pigs.n_for_cr(data, target_cr=1000) # file-size control up front
```

## Why each stage works (short)
- **fast**: block-diagonal Σ ⇒ each Gaussian factorizes over (velocity × space) ⇒ one exact full-grid step
  is a GEMM (~17 ms vs ~6 s dense); Huber preserves the subdominant modes energy-greedy MSE starves.
- **gPINC**: the separable reconstruction is bit-identical through the FFT physics losses → same training,
  ~10× faster than dense PINC.
- **PIGS**: a real Gaussian's y-spectrum sits at ky=0, so the flux-carrying tail (ky 7–15) is unreachable
  (uncertainty principle). A grafted Gabor carrier `G·exp(i·ky·(y−μ_y))` = one exact ky mode = one
  independent per-bin cross-phase DOF. Production graft = **tied groups** (`tied=True` default): envelopes
  placed for COVERAGE (stratified; ~6× better flux + 4–6 dB φ than residual hotspots), each carrying K=9
  carriers with shared μ/Σ (14+3K vs 17K params → ~3.7× carriers/byte → ~20× better flux at equal bytes).
  Freeze the φ-optimal base, POST-refine the carrier amps only (tying + on-bin ky stay exact):
  `flux_mode='raw'` (min total flux) or `'wnorm'` (W=Q/|φ|², φ-preserving tail-only); `flux_lambda` slides
  the φ-vs-flux Pareto. **Budget-neutral**: `n_total = n_base + n_flux` atom-equivalents.
- **flux_relL1** = relative L1 of the ky-resolved heat-flux spectrum Q(ky), band 1–15 (NOT the eflux-field
  error, NOT the degenerate scalar total flux). dom = ky 1–6, tail = ky 7–15.

## Practical notes
- All models returned **fp16-rounded** (verified ~lossless incl. the flux cross-phase) → `evaluate` is on-disk-fair.
- `seed=0` ⇒ bit-reproducible (also resets the loader's in-place shuffle hysteresis).
- Each `compress_*` frees the CUDA cache at entry — running the ladder in sequence on one GPU is safe
  (dense PINC peaks near the full GPU).
- Density-fit kwargs pass through: `compress_fast(data, n, warmup_steps=500, polish_epochs=0, loss="mse")`.
- For large N (>~6000) skip the fused polish (`polish_epochs=0`) — the polish batch materializes (B,N) memory.

Notebook: `notebooks/06_pigs.ipynb` (§1 ladder + plots, §2 rationale + ablations: CR sweep, warmup,
gPINC-vs-PINC, flux_lambda Pareto, flux_frac). Full study + negative results: `../gsplat_speed/RESULTS.md`.
