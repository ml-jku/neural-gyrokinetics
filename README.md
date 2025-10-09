# Neural Gyrokinetics

Machine learning tools to accelerate high-dimensional plasma turbulence simulations.
Neural Gyrokinetics includes research code for <img src="pages/imgs/gyroswin_icon.png" alt="GyroSwin Icon" height="12px"> <strong>[GyroSwin](https://arxiv.org/abs/2510.07314)</strong>, a 5D neural surrogate for nonlinear gyrokinetics.

## Who is this for?
For researchers at the intersection between scientific machine learning and plasma physics, or in genral high-dimensional simulations.

## Pretrained GyroSwin Models

We are working on a public release of the GyroSwin series (Small, Medium, Large) on Huggingface. As soon as they are online, we will link them here.

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

After generating and preprocessing the dataset, GyroSwin and baselines training can be started with the `main.py` entrypoint.

## <img src="pages/imgs/gyroswin_icon.png" alt="GyroSwin Icon" height="18px"> GyroSwin
<p align="center">
  <img src="pages/imgs/figure1.png" alt="Figure 1" width="66%">
</p>
GyroSwin is a 5D vision transformers trained to capture the full nonlinear dynamics of gyrokinetic plasma turbulence. It uses shifted window linear attention, as global attention is too expensive for 5-dimensional grids.
GyroSwin provides accurate predictions of turbulent transport at a fraction of the computational cost, while preserving key physical phenomena missed by tabular regression or quasilinear models. 

Check out our [blogpost](https://ml-jku.github.io/gyroswin/)!

## Project structure
```
📁 data_generation                 # Configs for generating gyrokinetics data

📁 gyroswin
├── 📁 dataset                     # Dataset utilities and preprocessing
│   ├── 📄 augment.py              # Data augmentation functions
│   ├── 📄 cyclone.py              # Gyrokinetics dataset class
│   ├── 📄 preprocess.py           # Preprocessing utilities
├── 📁 eval                        # Evaluation and analysis
│   ├── 📄 compute_diagnostics.py  # Compute diagnostics for turbulence
│   ├── 📄 evaluate.py             # Evaluation runner
│   ├── 📄 inference.py            # Inference utilities
│   ├── 📄 plot_utils.py           # Plotting helper functions
│   ├── 📄 postprocess.py          # Postprocessing of outputs
│   ├── 📄 rollout.py              # Rollout evaluation script
├── 📁 models                      # Model architectures
│   ├── 📁 nd_vit                  # nD Vision Transformer modules
│   │   ├── 📄 drop.py             # Dropout and regularization
│   │   ├── 📄 patching.py         # Patching utilities
│   │   ├── 📄 positional.py       # Positional encodings
│   │   ├── 📄 swin_layers.py      # Swin Transformer layers
│   │   ├── 📄 vit_layers.py       # ViT layers
│   │   ├── 📄 x_layers.py         # Extra/custom transformer layers
│   ├── 📄 fno.py                  # Fourier Neural Operator baseline
│   ├── 📄 transformer.py          # Transformer baseline
│   ├── 📄 transolver.py           # Transolver baseline
│   ├── 📄 vit_flat.py             # Vision Transformer baseline
│   ├── 📄 swin_flat.py            # Basic Swin Transformer
│   ├── 📄 pointnet.py             # PointNet baseline
│   ├── 📄 gk_unet.py              # UNet swin model
│   ├── 📄 gk_multi.py             # Multi-head GyroSwin
│   ├── 📄 layers.py               # Common model layers
├── 📁 train                       # Training utilities
│   ├── 📄 integrals.py            # Gyrokinetics integrals
│   ├── 📄 losses.py               # Physics-informed loss functions
├── 📄 utils.py                    # General helper functions
└── 📄 run.py                      # Runner function

📄 utils.py                        # General utilities
📄 main.py                         # Main entry point for training/experiments
```

## Citing

```
@inproceedings{paischer2025gyroswin,
    author       = {Fabian Paischer and Gianluca Galletti
                    and William Hornsby and Paul Setinek
                    and Lorenzo Zanisi and Naomi Carey
                    and Stanislas Pamela and Johannes Brandstetter
                },
    booktitle    = {Advances in Neural Information Processing Systems 38: Annual Conference
                  on Neural Information Processing Systems 2025, NeurIPS 2025, San Diego,
                  CA, USA, December 02 - 07, 2025},
    year         = {2025}
    }
```
