# neugk-jax

JAX/Equinox port of the gyrokinetic Swin5D autoencoder + latent flow matching pipeline.

Self-contained — does **not** depend on the surrounding torch codebase.

## Install

```bash
pip install -e ".[cuda,gyro,dev]"
```

`gyaradax` provides the JAX flux integrals used in evaluation (electrostatic only).
`cupy-cuda12x` + `kvikio-cu12` enable GPU-direct reads from the binary dataset
(optional; CPU fallback via `np.fromfile` is available).

## Layout

- `neugk_jax/models/` — equinox modules (MLP, embeddings, patching, attention, Swin/ViT, gk_unet, DiT)
- `neugk_jax/autoencoders/` — Swin5DAE
- `neugk_jax/diffusion/` — flow matching
- `neugk_jax/dataset/` — unified CycloneDataset (ae / diff modes)
- `neugk_jax/training/` — runner, schedulers, distributed setup, checkpoint, logging
- `neugk_jax/evaluate/` — base evaluator, AE/diffusion evaluators, gyaradax integrals adapter
- `configs/` — Hydra configs (mirror upstream layout)
- `main.py` — Hydra entrypoint (mirrors the upstream `main.py`)
- `scripts/` — `translate_*ckpt.py`, `eval_diffusion.py`, `plot_reconstruction.py`,
  `quantization_error_study.py`, `benchmark_{ae_train,dataloader}.py`
- `tests/`

## Milestones

1. **M1** — Skeleton + models forward (shape tests, fwd/bwd benchmark)
2. **M2** — Torch→Orbax checkpoint translator + AE parity (<1e-4)
3. **M3** — Dataset + loaders (parity vs torch)
4. **M4** — AE training (single + multi-GPU + multi-node)
5. **M5** — Flow matching training + eval (gyaradax integrals)
