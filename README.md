# Neural Gyrokinetics

Machine learning tools to accelerate and compress high-dimensional plasma turbulence simulations. A growing collection of research code at the intersection of scientific machine learning and gyrokinetics.

- <img src="pages/imgs/gyroswin_icon.png" alt="GyroSwin Icon" height="12px"> <strong>[GyroSwin](https://arxiv.org/abs/2510.07314)</strong>: a 5D neural surrogate for nonlinear gyrokinetics. [blogpost](https://ml-jku.github.io/blog/2025/gyroswin/)
- <img src="pages/imgs/pinc_icon.png" alt="PINC Icon" height="12px"> <strong>[PINC](https://arxiv.org/abs/2602.04758)</strong>: physics-informed neural compression of plasma data. [blogpost](https://ml-jku.github.io/blog/2026/pinc/)
- <img src="pages/imgs/gyrosplat_icon.png" alt="GyroSplat Icon" height="12px"> <strong>[GyroSplat](https://openreview.net/forum?id=xZMuTU83eC)</strong>: 5D snapshots as Gaussian and Gabor splats, with the potential and the heat flux computed from the stored splats. [code](neugk/gyrosplat)
- <img src="pages/imgs/gyroflow_icon.png" alt="GyroFlow Icon" height="12px"> <strong>[GyroFlow](https://arxiv.org/abs/2607.13022)</strong>: latent flow matching that samples statistically steady 5D turbulence without simulating the transient. [code](neugk/diffusion)
- <strong>[gyaradax](https://arxiv.org/abs/2604.06085)</strong>: a differentiable JAX gyrokinetic solver. [repo](https://github.com/gerkone/gyaradax) · [blogpost](https://gerkone.github.io/blog/gyaradax.html)

This repository keeps growing as the project expands.

## Who is this for?
Researchers at the intersection of scientific machine learning and plasma physics, or anyone working on accelerating high-dimensional simulations.

## Models and data
GyroSwin checkpoints are on the Hub in three sizes: [small](https://huggingface.co/ml-jku/gyroswin_small) | [medium](https://huggingface.co/ml-jku/gyroswin_medium) | [large](https://huggingface.co/ml-jku/gyroswin_large), with the in- and out-of-distribution evaluation cases at [gyroswin_cbc_id_ood](https://huggingface.co/datasets/ml-jku/gyroswin_cbc_id_ood).

The PINC compression test set and model checkpoints (neural fields and autoencoders) are at [pinc_gkw](https://huggingface.co/datasets/gerkone/pinc_gkw). The complete adiabatic trajectory set (bf16 distribution functions and potentials, metadata, GKW `input.dat`) is at [🤗 cbc-gyroswin-256traj](https://huggingface.co/datasets/gerkone/cbc-gyroswin-256traj), with the held-out evaluation trajectories under `id/` and `ood/`. Download the evaluation sets only:
```python
from huggingface_hub import snapshot_download
snapshot_download("gerkone/cbc-gyroswin-256traj", repo_type="dataset",
                  allow_patterns=["id/*", "ood/*"], local_dir="data/preprocessed_kvikio")
```

For GyroSplat the per-snapshot result rows behind its tables are in [`neugk/gyrosplat/results`](neugk/gyrosplat/results). Checkpoints might be added later.

## Reproducing GyroSwin
With the evaluation sets downloaded as above, from the repository root:
```
python scripts/gyroswin/eval_gyroswin.py --ckpt ml-jku/gyroswin_large --data-root data/preprocessed_kvikio
```
This rolls out the hub checkpoint (`--ckpt`, small / medium / large) on the 6 in- and 5 out-of-distribution cases of the paper and prints the time-averaged heat-flux error per case and the RMSE per split (Table 2). `--split id|ood|all` evaluates every trajectory under `id/` and `ood/` instead. Single rollouts that write each prediction (`df`, `phi`, `flux`) to `predictions/` run with `python -m neugk.gyroswin.eval.inference_from_hf --ckpt <checkpoint>`.

## Reproducing GyroSplat
The notebooks in `notebooks/gyrosplat` rebuild every table and figure from prior calculated results. To fit a single snapshot run something like:
```
python -m neugk.gyrosplat fit --dataset gkw --config configs/gyrosplat/recipe/gkw_302x.yaml data.root=<snapshots> data.trajectory=iteration_13 data.timestep=100
```
`scripts/gyrosplat/reproduce_gkw.sh <snapshots> <results_dir>` runs GyroSplat and the baselines on the 39 evaluation snapshots at both file sizes. See the [GyroSplat README](neugk/gyrosplat/README.md).

## Running
Experiments use Hydra configs (`configs/{dataset,logging,model,training,validation}`); training and evaluation run through `main.py`. To sample from a trained diffusion checkpoint:
```
python generate_samples.py configs/generation/generate_samples.yaml
```
setting `diffusion_checkpoint_dir`, `ae_checkpoint`, `data_path` and `generation.{trajectories,output_dir}` in the config. With `normalization_file` and `inference.latent_scale` set, no training data is needed.
`scripts/gyroflow/eval_gyroflow.py <sampling config>` samples the `id/` and `ood/` sets and reports the time-averaged heat-flux error.

## Components

<img src="pages/imgs/gyroswin_icon.png" alt="GyroSwin Icon" height="16px"> <strong>GyroSwin</strong> is a 5D vision transformer with shifted-window linear attention that captures the full nonlinear dynamics of gyrokinetic turbulence, predicting turbulent transport at a fraction of the simulation cost while preserving physics that quasilinear models miss. See the [blogpost](https://ml-jku.github.io/blog/2025/gyroswin/).

<img src="pages/imgs/pinc_icon.png" alt="PINC Icon" height="16px"> <strong>PINC</strong> compresses gyrokinetic turbulence data by up to 70,000x while preserving physical diagnostics, with a unified pipeline to assess how well compressors retain spatial and temporal turbulence. See the [blogpost](https://ml-jku.github.io/blog/2026/pinc/).

<img src="pages/imgs/gyrosplat_icon.png" alt="GyroSplat Icon" height="16px"> <strong>GyroSplat</strong> stores a 5D snapshot as a sum of Gaussian and Gabor splats, 302x smaller on the adiabatic GKW set. The potential and the heat-flux spectrum are formulas of the stored splats, and the fit matches both to the simulation. See the [paper](https://openreview.net/forum?id=xZMuTU83eC) and the [README](neugk/gyrosplat/README.md).

<img src="pages/imgs/gyroflow_icon.png" alt="GyroFlow Icon" height="16px"> <strong>GyroFlow</strong> is a latent flow-matching model over the compressed representations. It samples statistically steady 5D turbulence directly, without simulating the transient. See the [paper](https://arxiv.org/abs/2607.13022).

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
├── 📁 gyrosplat                      # compression with Gaussian and Gabor splats
│   ├── 📁 model                      # splats, rendering, amplitude solves, fitting stages, storage
│   ├── 📁 backends                   # synthetic and GKW potential / flux readouts
│   ├── 📁 baselines                  # Fourier, wavelets, PCA, ZFP/SZ3, plain splats, neural field
│   ├── 📁 eval                       # scores, ablations, tables and figures of the paper
│   ├── 📁 results                    # per-snapshot result rows
│   └── 📄 recipe.py                  # the pipeline: compress / decode / score
│
├── 📁 diffusion                      # GyroFlow: latent flow matching / diffusion
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

```
@inproceedings{gutenbrunner2026gyrosplat,
    title={GyroSplat: A Linear Representation of Fusion Plasma Turbulence with Direct and Differentiable Physics},
    author={Gerald Gutenbrunner and Andreas Radler and Gianluca Galletti and Johannes Brandstetter and Fabian Paischer},
    booktitle={RPS Workshop at NeurIPS 2026},
    year={2026},
    url={https://openreview.net/forum?id=xZMuTU83eC}
}
```
