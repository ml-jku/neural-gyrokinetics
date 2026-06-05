# PINC revival: autoencoder configs

Hydra experiment configs for the generalizing autoencoders (one model for all
snapshots), composed onto `configs/main.yaml`. They live under `configs/experiment/`
because AE training is Hydra-composed (`experiment=...`), unlike the neural-field
sweep which uses flat configs in `configs/pinc_revival/` read directly by `nf_main`.

Run:
```
python main.py experiment=pinc_revival/<name>
```
All use `dataset=pinc` (train split, 235 trajectories), `model=pinc/{ae,vqvae}`, the
paper training schedule (400 epochs, bs 16, lr 3e-4, cosine), and `workflow=pinc_autoencoder`.

## Main set (runnable today)
| config | model | quantizer | status |
|---|---|---|---|
| `ae` | pinc/ae | none (continuous latent) | ready |
| `vqvae` | pinc/vqvae | euclidean codebook | ready — the basic VQ-VAE for the main table |
| `vqvae_cosine` | pinc/vqvae | cosine codebook | ready |

## Modern VQ flavor ablations (need a quantizer code hook)
These are the modern discrete-bottleneck variants reviewers asked about. They are
all **in-training** quantizers, not post-hoc: the quantization (rounding / sign /
residual selection) happens inside the forward pass with a straight-through
estimator, so the encoder/decoder learn around it end-to-end. A post-hoc scheme
would instead train a continuous AE and quantize the latents afterward — that is a
different (weaker) setup and not what these configs do.

| config | flavor | knob | needs |
|---|---|---|---|
| `vqvae_fsq` | Finite Scalar Quantization | `vq.levels` | no codebook/EMA, no collapse |
| `vqvae_lfq` | Lookup-Free Quantization (MAGVIT-v2) | `vq.codebook_size` (pow2), entropy | scales to huge codebooks |
| `vqvae_rvq` | Residual VQ | `vq.num_quantizers` | finer recon at fixed codebook |

**Code required:** `neugk/pinc/autoencoders/vector_quantize.py` only implements the
euclidean and cosine codebooks; `gk_autoencoders.py` selects them via
`vq.codebook_type`. To enable the flavors, add a `vq.quantizer` branch there that
builds the corresponding lucidrains class (`FSQ`, `LFQ`, `ResidualVQ` from
`vector-quantize-pytorch`) and projects the bottleneck to its expected dim. Until
that hook exists, the flavor configs fall back to the euclidean codebook.

Recommendation: run `ae`, `vqvae` (euclidean), and `vqvae_cosine` now for the main
comparison; add the FSQ/LFQ/RVQ ablations as a smaller follow-up once the quantizer
hook is in (FSQ is the highest-value, lowest-risk one to add first).
