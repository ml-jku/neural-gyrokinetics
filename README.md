# Neural Gyrokinetics

Machine learning tools to accelerate and compress high-dimensional plasma turbulence simulations. A growing collection of research code at the intersection of scientific machine learning and gyrokinetics.

- <img src="pages/imgs/gyroswin_icon.png" alt="GyroSwin Icon" height="12px"> <strong>[GyroSwin](https://arxiv.org/abs/2510.07314)</strong>: a 5D neural surrogate for nonlinear gyrokinetics. [blogpost](https://ml-jku.github.io/blog/2025/gyroswin/)
- <img src="pages/imgs/pinc_icon.png" alt="PINC Icon" height="12px"> <strong>[PINC](https://arxiv.org/abs/2602.04758)</strong>: physics-informed neural compression of plasma data. [blogpost](https://ml-jku.github.io/blog/2026/pinc/)
- <strong>Diffusion</strong>: latent flow matching over the learned representations for generative sampling.
- <strong>[gyaradax](https://arxiv.org/abs/2604.06085)</strong>: a differentiable JAX gyrokinetic solver. [repo](https://github.com/gerkone/gyaradax) · [blogpost](https://gerkone.github.io/blog/gyaradax.html)

This repository keeps growing as the project expands.

## Who is this for?
Researchers at the intersection of scientific machine learning and plasma physics, or anyone working on accelerating high-dimensional simulations.

## Models and data
GyroSwin checkpoints are on the Hub in three sizes: [small](https://huggingface.co/ml-jku/gyroswin_small) | [medium](https://huggingface.co/ml-jku/gyroswin_medium) | [large](https://huggingface.co/ml-jku/gyroswin_large), with the in- and out-of-distribution evaluation cases at [gyroswin_cbc_id_ood](https://huggingface.co/datasets/ml-jku/gyroswin_cbc_id_ood).

The PINC compression test set and model checkpoints (neural fields and autoencoders) are at [pinc_gkw](https://huggingface.co/datasets/gerkone/pinc_gkw). The complete trajectory set, with the per-trajectory GKW `input.dat` and generation config for reproducibility, is at [gyrokinetic-adiabatic-256traj](https://huggingface.co/datasets/gerkone/gyrokinetic-adiabatic-256traj).

GyroSwin inference from the Hub:
```
python -m neugk.gyroswin.eval.inference_from_hf
```
This fetches the data and weights and rolls out autoregressively, writing each prediction (`df`, `phi`, `flux`) to `predictions/`. Select the checkpoint with `--ckpt`.

## Running
Experiments use Hydra configs (`configs/{dataset,logging,model,training,validation}`); training and evaluation run through `main.py`. To sample from a trained diffusion checkpoint:
```
python generate_samples.py --config configs/generation/generate_samples.yaml
```
setting `diffusion_checkpoint_dir`, `ae_checkpoint`, and `generation.{trajectories,output_dir}` in the config.

## Components

<img src="pages/imgs/gyroswin_icon.png" alt="GyroSwin Icon" height="16px"> <strong>GyroSwin</strong> is a 5D vision transformer with shifted-window linear attention that captures the full nonlinear dynamics of gyrokinetic turbulence, predicting turbulent transport at a fraction of the simulation cost while preserving physics that quasilinear models miss. See the [blogpost](https://ml-jku.github.io/blog/2025/gyroswin/).

<img src="pages/imgs/pinc_icon.png" alt="PINC Icon" height="16px"> <strong>PINC</strong> compresses gyrokinetic turbulence data by up to 70,000x while preserving physical diagnostics, with a unified pipeline to assess how well compressors retain spatial and temporal turbulence. See the [blogpost](https://ml-jku.github.io/blog/2026/pinc/).

<strong>gyaradax</strong> is a differentiable JAX implementation of the gyrokinetic right-hand side, used for the physics losses and the PINN-residual baseline. [repo](https://github.com/gerkone/gyaradax) · [paper](https://arxiv.org/abs/2604.06085) · [blogpost](https://gerkone.github.io/blog/gyaradax.html)

## Project structure
```
📁 configs                            # Experiment configs (Hydra)

📁 neugk
├── 📁 gyroswin                       # GyroSwin surrogate
│   ├── 📁 eval                       # rollout evaluation and hub inference
│   ├── 📁 models                     # GyroSwin and baseline architectures
│   └── 📄 run.py                     # GyroSwin runner
│
├── 📁 pinc                           # physics-informed neural compression
│   ├── 📁 autoencoders               # 5D Swin AE, VAE, VQ-VAE, VAPOR baseline
│   ├── 📁 neural_fields              # per-snapshot neural fields (incl. PINN-residual baseline)
│   ├── 📁 eval                       # compression metrics, reconstructors, traditional baselines
│   ├── 📄 losses.py                  # PINC integral and spectral losses
│   ├── 📄 peft_utils.py              # LoRA/EVA adapters for the fine-tuning stage
│   ├── 📄 nf_main.py                 # neural field parallel runner
│   └── 📄 run.py                     # PINC autoencoder runner
│
├── 📁 diffusion                      # latent flow matching / diffusion
│   ├── 📁 models                     # AR transformer, DiT, diffusion UNet
│   └── 📄 run.py                     # diffusion runner
│
├── 📁 physics                        # gyrokinetic integrals and diagnostics
│   ├── 📄 integrals.py               # potential and flux phase-space integrals
│   └── 📄 diagnostics.py             # spectra and turbulence diagnostics
│
├── 📁 dataset                        # dataset classes, backends and preprocessing
├── 📁 models                         # shared nD ViT / Swin building blocks
├── 📄 evaluate.py                    # base evaluator
├── 📄 losses.py                      # common losses and gradient balancer
└── 📄 runner.py                      # base runner class

📄 main.py                            # training / evaluation entry point
📄 generate_samples.py                # diffusion sampling entry point
```

## Citing

```
@inproceedings{paischer2025gyroswin,
    title={GyroSwin: 5D Surrogates for Gyrokinetic Plasma Turbulence Simulations},
    author={Fabian Paischer and Gianluca Galletti and William Hornsby and Paul Setinek and Lorenzo Zanisi and Naomi Carey and Stanislas Pamela and Johannes Brandstetter},
    booktitle={Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2025, NeurIPS 2025, San Diego, CA, USA, December 02 - 07, 2025},
    year={2025}
}
```

```
@misc{galletti2026pinc,
      title={Physics-Informed Neural Compression of High-Dimensional Plasma Data},
      author={Gianluca Galletti and Gerald Gutenbrunner and Sandeep S. Cranganore and William Hornsby and Lorenzo Zanisi and Naomi Carey and Stanislas Pamela and Johannes Brandstetter and Fabian Paischer},
      year={2026},
      eprint={2602.04758},
      archivePrefix={arXiv},
      primaryClass={physics.plasm-ph},
}
```
