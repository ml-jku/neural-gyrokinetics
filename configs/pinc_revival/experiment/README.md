# PINC revival: autoencoder configs

Hydra experiment configs for the generalizing autoencoders (one model for all
snapshots), composed onto `configs/main.yaml`.

Run:
```
python main.py --config-dir configs/pinc_revival experiment=<name>
```
All use `dataset=pinc` (train split, 235 trajectories), `model=pinc/{ae,vqvae}`, the
paper training schedule (400 epochs, bs 16, lr 3e-4), and `workflow=pinc_autoencoder`.

## Quantizer flavors (VQ-VAE)

The discrete-bottleneck variants are all **in-training** quantizers, not post-hoc:
the quantization (codebook lookup / rounding / sign / residual selection) happens
inside the forward pass with a straight-through estimator, so the encoder/decoder
learn around it end-to-end. Selected by `model.vq.quantizer`; implemented in
`neugk/pinc/autoencoders/vector_quantize.py` and wired in `gk_autoencoders.py`
(`Swin5DVQVAE`) via `self.vq`.

| config | `vq.quantizer` | what it is | key knobs |
|---|---|---|---|
| `vqvae` | `vq` (euclidean) | learned codebook, EMA + commitment loss | `codebook_size`, `embedding_dim` |
| `vqvae_cosine` | `vq` + `codebook_type: cosine` | cosine-sim codebook | `codebook_size` |
| `vqvae_fsq` | `fsq` | Finite Scalar Quantization (no codebook/EMA, no collapse) | `levels` (implicit cb = prod(levels)) |
| `vqvae_lfq` | `lfq` | Lookup-Free Quantization (MAGVIT-v2), sign bits + entropy reg | `codebook_size` (pow2), `entropy_loss_weight` |
| `vqvae_rvq` | `rvq` | Residual VQ: stack of codebooks on residuals | `num_quantizers`, `codebook_size` |

All five build through `get_autoencoder` and produce a reconstruction plus
`out["vq_indices"]` (verified, CPU). RVQ stores `num_quantizers` codes per token;
the others store 1. `levels: [8,8,8,5,5,5]` gives an implicit codebook of 64000.

## Compression-ratio (CR) levels

CR = (input bytes) / (latent bytes), input stored as float32 (4 bytes).

- **Continuous AE** (`ae_*`): latent is a dense float tensor
  `prod(bottleneck_grid) * bottleneck.dim` values (stored float32). So
  `CR = prod(base_resolution_4d) * in_channels * 4 / (prod(grid) * dim * 4)`.
  Two knobs: `patch.patch_size` + number of vit layers set the token grid;
  `bottleneck.dim` sets the latent width.
- **VQ-VAE** (`vqvae_*`): latent is the integer index grid, stored int16 (2 bytes),
  `prod(grid) * codes_per_token` indices. So
  `CR = prod(base_resolution_4d) * in_channels * 4 / (prod(grid) * codes_per_token * 2)`.
  CR is **independent of `bottleneck.dim` and `codebook_size`** — it is set purely by
  the number of tokens and codes/token (1, or `num_quantizers` for RVQ). The codebook
  size only changes the bits/index (and thus a tighter entropy-coded bound), not this
  index-count accounting, which matches the eval reconstructors' int16 storage.

Resolution: full field `(32, 8, 16, 85, 32)`, `separate_zf=True` so
`problem_dim = 2 + 2 = 4`. `decouple_mu` folds the mu axis (8) into the channels, so
the AE sees base_resolution `[32, 16, 85, 32]` with `4*8 = 32` effective input
channels. Input elements = `32*16*85*32 * 32 = 47,513,600`.

### How ~100x (and beyond) compression is reached
Take the basic `ae` (continuous): grid `[4,2,9,4] = 288` tokens, `bottleneck.dim 256`.
`CR = 47,513,600 * 4 / (288 * 256 * 4) = 47,513,600 / 73,728 ~= 644x` of pointwise
data. The lever is the token grid: each `patch_size` factor and each vit
down-merge layer multiplies the spatial reduction, and `bottleneck.dim` trades latent
width linearly. To hit a target CR you pick a `patch_size`/layer count for the grid
and then set `bottleneck.dim` (AE) — for the VQ-VAE you only pick the grid, since the
index accounting ignores the width. ~100x is the low end (coarse-ish grid, wide
latent); the configs below span 200x–100000x.

### Measured CRs (one real forward at full resolution; `tests/_tune_cr.py`)
| config | target | grid (tokens) | latent | measured CR |
|---|---|---|---|---|
| `ae_200x` | 200x | `[4,2,9,4]` (288) | dim 768 | **201x** |
| `ae_1000x` | 1000x | `[4,2,9,4]` (288) | dim 154 | **1005x** |
| `ae_10000x` | 10000x | `[2,2,5,2]` (40) | dim 112 | **9947x** |
| `vqvae_1000x` | 1000x | `[16,8,43,16]` (88064) | 1 code/tok | **1012x** |
| `vqvae_10000x` | 10000x | `[8,8,9,16]` (9216) | 1 code/tok | **9671x** |
| `vqvae_100000x` | 100000x | `[4,4,9,8]` (1152) | 1 code/tok | **77369x** |

All within ~1.5x of the filename target. `vqvae_100000x` lands at 77369x because the
token grid is discrete: 1152 tokens is the best-centered choice; the next-coarser grid
(576 tokens) gives 154738x = 1.5x over target.

## Physics-loss modes (`training.physics_mode`)

Two ways to bring the PINC physics losses (flux/phi integral + spectral) into the
generalizing AE. Routed in `neugk/pinc/run.py` (`PINCRunner._resolve_physics_mode`).

- **`scheduled`** (single stage, no finetune) — `ae_scheduled.yaml`. One
  `stage=autoencoder` run. The df reconstruction loss is active from step 0; the
  physics-loss weights start at 0 and are linearly ramped in by the `loss_scheduler`
  block (`start_fraction` -> `end_fraction`). The runner turns each `loss_scheduler`
  entry into a per-step weight via `get_linear_burn_in_fn`, fed to `PINCLossWrapper`.
- **`two_stage`** (pretrain then PEFT adapt) — `ae_two_stage_pretrain.yaml` +
  `ae_two_stage_peft.yaml`. Stage 1 (`stage=autoencoder`) trains the AE on the df
  reconstruction loss only (physics terms must be absent, enforced by the runner).
  Stage 2 (`workflow=pinc_peft`, `stage=peft`, `ae_checkpoint=<stage1 run>`) loads the
  frozen AE, attaches LoRA/EVA adapters (`model.peft`), freezes the base weights, and
  finetunes only the adapters against the physics losses.

Both validated on CPU (`tests/_validate_physics_modes.py`): the scheduler holds the
physics weight at 0 until 50% then ramps to 1e-4 by 70%; a df-recon step runs
(scheduled / stage 1); and LoRA adapters attach (base frozen, only adapters trainable)
and finetune (stage 2).
