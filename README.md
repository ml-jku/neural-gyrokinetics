# Neural Gyrokinetics

Machine learning tools to accelerate high-dimensional plasma turbulence simulations.
Neural Gyrokinetics includes research code for
- <img src="docs/imgs/gyroswin_icon.png" alt="GyroSwin Icon" height="12px"> <strong>[GyroSwin](https://arxiv.org/abs/2510.07314)</strong>, a 5D neural surrogate for nonlinear gyrokinetics.
- <img src="docs/imgs/pinc_icon.png" alt="PINC Icon" height="12px"> <strong>[PINC](TODO)</strong>, physics-informed neural compression for plasma data.

## Who is this for?
For researchers at the intersection between scientific machine learning and plasma physics, or in genral high-dimensional simulations.

## Pretrained GyroSwin Models

We are working on a public release of the GyroSwin series (Small, Medium, Large) as well as the PINC-VQ-VAE, on Huggingface. As soon as they are online, we will link them here.

## Data Generation
The dataset used to train GyroSwin is too large to be easily distributed,
but we include instructions on how to generate it as well as the configuration files needed in the `data_generation` directory. 

## Running
Running is managed with Hydra configs, structured as follows.

```
📁 configs
├── 📁 dataset                     # Dataset configs (specify paths and trajectories here)
├── 📁 logging                     # Logging configs
├── 📁 model                       # Configs for GyroSwin and baselines
├── 📁 training                    # Training configs
└── 📁 validation                  # Validation configs
```

After generating and preprocessing the dataset, GyroSwin and baselines training can be started with `main_gyroswin.py`.

## <img src="docs/imgs/gyroswin_icon.png" alt="GyroSwin Icon" height="18px"> GyroSwin
<p align="center">
  <img src="docs/imgs/figure1.png" alt="Figure 1" width="66%">
</p>
GyroSwin is a 5D vision transformer trained to capture the full nonlinear dynamics of gyrokinetic plasma turbulence. It uses shifted window linear attention, as global attention is too expensive for 5-dimensional grids.
GyroSwin provides accurate predictions of turbulent transport at a fraction of the computational cost, while preserving key physical phenomena missed by tabular regression or quasilinear models.

Check out our [blogpost](https://ml-jku.github.io/blog/2025/gyroswin/)!


## <img src="docs/imgs/pinc_icon.png" alt="PINC Icon" height="18px"> Physics-Informed Neural Compression of Plasma Data
<p align="center">
  <img src="docs/imgs/pinc.png" alt="Figure 1" width="66%">
</p>

__Physics-Inspired Neural Compression (PINC)__ investigates compression of (storage intensve) gyrokinetic plasma turbulence data by up to 70,000× while preserving key physical characteristics. It also proposes a unified evaluation pipeline to assess how well different compression techniques retain spatial and temporal turbulence phenomena.

PINC is presented in our second [blogpost](https://ml-jku.github.io/blog/2025/pinc/).


## Project structure
```
📁 data_generation                 # Info for generating gyrokinetics data from GKW

📁 configs                         # Experiment configs

📁 neugk
├── 📁 dataset                     # Dataset utilities and preprocessing
│   ├── 📄 augment.py              # Data augmentation functions
│   ├── 📄 cyclone.py              # Gyrokinetics dataset class
│   └── 📄 preprocess.py           # Preprocessing utilities
│
├── 📁 models                      # Model architectures
│   ├── 📁 nd_vit                  # nD Vision Transformer modules
│   │   ├── 📄 drop.py             # Dropout and regularization
│   │   ├── 📄 patching.py         # Patching utilities
│   │   ├── 📄 positional.py       # Positional encodings
│   │   ├── 📄 swin_layers.py      # Swin Transformer layers
│   │   └── 📄 vit_layers.py       # ViT layers
│   ├── 📄 gk_unet.py              # UNet swin model
│   └── 📄 x_layers.py             # Common utility layers
│
├── 📁 gyroswin                    # Code from the GyroSwin paper
│   ├── 📁 eval                    # Evaluation and analysis
│   │   ├── 📄 compute_diagnostics.py  # Compute diagnostics for turbulence
│   │   ├── 📄 evaluate.py         # Evaluation runner
│   │   ├── 📄 inference.py        # Inference utilities
│   │   ├── 📄 plot_utils.py       # Plotting helper functions
│   │   ├── 📄 postprocess.py      # Postprocessing of outputs
│   │   └── 📄 rollout.py          # Rollout evaluation script
│   ├── 📁 models                  # Model architectures
│   │   ├── 📄 fno.py              # Fourier Neural Operator baseline
│   │   ├── 📄 transformer.py      # Transformer baseline
│   │   ├── 📄 transolver.py       # Transolver baseline
│   │   ├── 📄 vit_flat.py         # Vision Transformer baseline
│   │   ├── 📄 pointnet.py         # PointNet baseline
│   │   ├── 📄 gyroswin.py         # Multi-head GyroSwin
│   │   └── 📄 x_layers.py         # GyroSwin cross attention mixing blocks
│   ├── 📁 train                   # Training utilities
│   │   ├── 📄 integrals.py        # Gyrokinetics integrals
│   │   └── 📄 losses.py           # Physics-informed loss functions
│   ├── 📄 utils.py                # General helper functions
│   └── 📄 run.py                  # Runner function
│
├── 📁 pinc                        # Code from physics-inspired compression
│   ├── 📁 autoencoders            # 5D swin autoencoder and VQ-VAE
│   │   ├── 📄 swin_ae.py          # Dataset and dataloader
│   │   ├── 📄 peft_utils.py       # LoRA utilities for PINC training of large models
│   │   └── 📄 train.py            # Autoencoder training and evaluation utilities
│   ├── 📁 neural_fields           # Neural fields models, training and evaluation
│   │   ├── 📁 models              # MLP, SIREN and WIRE
│   │   ├── 📄 data.py             # Dataset and dataloader
│   │   ├── 📄 gk_losses.py        # Gyrokinetic physics-informed losses
│   │   └── 📄 train.py            # Neural field training and evaluation utilities
│   └── 📁 experiments             # Experiment notebooks and integration with GKW

📄 main_gyroswin.py                # GyroSwin entry point for training/experiments
📄 main_pinc.py                    # PINC autoencoder training
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
