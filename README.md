# neugk-jax

JAX/Equinox port of the gyrokinetic Swin5D autoencoder + latent flow matching pipeline.

Self-contained — does **not** depend on the surrounding torch codebase.

## Install

```bash
pip install -e ".[cuda,gyro,dev]"
```

`gyaradax` provides the JAX flux integrals used in evaluation (electrostatic only).
`cupy-cuda12x` + `kvikio-cu12` enable GPU-direct reads from the binary dataset
(optional; CPU fallback via `np.fromfile` is available). `h5py` (extra `h5`) reads
single-file `.h5` trajectories; the `dev` extra adds `h5py` and `huggingface_hub` for the
public data tests.

## Layout

- `neugk_jax/models/` — equinox modules (MLP, embeddings, patching, attention, Swin/ViT, gk_unet, DiT)
- `neugk_jax/pinc/` — Swin5DAE (optionally encoder / decoder conditioned), the PINC-AE
  LoRA fine-tune on the physics losses (`experiment=pinc_revival`) and Swin5DVQVAE with EMA VQ /
  FSQ / LFQ quantizers (`experiment=vqvae`)
- `neugk_jax/diffusion/` — flow matching
- `neugk_jax/gyroswin/` — GyroSwin multitask model, runner and rollout evaluator
- `neugk_jax/dataset/` — CycloneDataset (ae / diff / next modes), binary and h5 backends
- `neugk_jax/training/` — runner, schedulers, distributed setup, checkpoint, logging
- `neugk_jax/evaluate/` — base evaluator, flux integrals, spectral metrics
- `configs/` — Hydra configs; `configs/checkpoints/` holds release model configs
- `main.py` — Hydra entrypoint
- `scripts/` — `translate_ckpt.py` (AE / DiT / GyroSwin), `eval_diffusion.py`, `export_pinc_torch.py`
- `docs/metrics.md` — validation metric definitions and renames
- `tests/`

## Running

Dataset paths are required: `export NEUGK_DATA=/path/to/preprocessed` (or `dataset.path=...`),
and `experiment=diffusion` / `experiment=pinc_revival` need `ae_checkpoint=<ae run dir>`.

## Tests

```bash
python -m pytest                                  # unit tests + public real-data / parity tests
XLA_FLAGS=--xla_force_host_platform_device_count=2 JAX_PLATFORMS=cpu python -m pytest  # 2 devices
```

- Unit tests use synthetic data only.
- Public real-data tests (`tests/test_hf_data.py`) download the CBC snapshot
  `ml-jku/gyroswin_cbc_id_ood/preprocessed/iteration_8.h5` (~90 MB) into the Hugging Face
  cache; they skip without hub access.
- Torch parity tests (`tests/parity/`) need `torch` and the torch `neugk` repository
  (`NEUGK_TORCH_REPO`, default: the parent directory) and skip otherwise. The GyroSwin release
  parity (`test_hf_gyroswin_parity.py`) downloads `ml-jku/gyroswin_large` (~4 GB) on first use;
  run it on a GPU.
- `tests/local/` and `scripts/local/` hold machine-local checks against private checkpoints
  and datasets; they are excluded via `.git/info/exclude` and never committed.

## Milestones

1. **M1** — Skeleton + models forward (shape tests, fwd/bwd benchmark)
2. **M2** — Torch→Orbax checkpoint translator + AE parity (<1e-4)
3. **M3** — Dataset + loaders (parity vs torch)
4. **M4** — AE training (single + multi-GPU + multi-node)
5. **M5** — Flow matching training + eval (gyaradax integrals)
