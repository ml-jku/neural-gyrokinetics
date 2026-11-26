# <img src="../../pages/imgs/gyrosplat_icon.png" alt="GyroSplat Icon" height="28px"> GyroSplat

Compression of 5D gyrokinetic snapshots with Gaussian and Gabor splats. The potential and the heat flux are computed directly from the stored splats, and the fit matches both to the simulation. On the adiabatic GKW set one snapshot goes from 89 MB to 295 KB (302x) with a heat-flux error below 1e-6. [paper](https://openreview.net/forum?id=xZMuTU83eC)

All commands run from the repository root. Requirements are the repository's (`pip install -r requirements.txt`); the wavelet, ZFP and SZ3 baselines also need `pywavelets`, `zfpy` and `pysz`.

## Quick start
```bash
# a small generated snapshot, on a CPU, in seconds
python -m neugk.gyrosplat fit --dataset synthetic --config configs/gyrosplat/recipe/synthetic.yaml --device cpu

# one GKW snapshot at 302x, one GPU, a few minutes
python -m neugk.gyrosplat fit --dataset gkw --config configs/gyrosplat/recipe/gkw_302x.yaml \
    data.root=/path/to/snapshots data.trajectory=iteration_13 data.timestep=100 --out results/gyrosplat.json

# a baseline at the same budget, scored by the same functions
python -m neugk.gyrosplat baseline --dataset gkw --method fourier_gn --budget 73728 \
    --config configs/gyrosplat/recipe/gkw_302x.yaml \
    data.root=/path/to/snapshots data.trajectory=iteration_13 data.timestep=100 --out results/fourier_gn.json

# medians over result files
python -m neugk.gyrosplat table results/*.json
```
Every setting is a field of `RecipeConfig` (`config.py`). The YAML files in `configs/gyrosplat/recipe/` are the recipes of the paper (`gkw_302x`, `gkw_142x`, and `synthetic` for the CPU); `key=value` on the command line overrides them.

## Method
A snapshot is written as a sum of separable Gaussian splats with complex amplitudes. A quarter of the budget goes to groups of seven Gabor splats that share one envelope and each carry a wave along the binormal direction. The fit has three stages:

1. **Base splats.** Gaussians are placed where the field is large and fitted to the field and the potential.
2. **Gabor groups.** The groups are added and fitted together with the base splats; the flux term moves only the Gabor splats.
3. **Flux refinement.** Eight Gauss-Newton steps on the Gabor amplitudes match the heat-flux spectrum on every nonzero binormal mode.

Stored are 17 numbers per base splat, 14 per shared envelope plus 3 per Gabor splat, and 4 normalization values: 73,732 numbers at 302x and 156,673 at 142x. `recipe.py` holds the whole pipeline in one file; the stages it calls are in `model/`.

## Results and notebooks
The stored rows are enough to rebuild every table and figure of the paper; the notebooks in `notebooks/gyrosplat/` need no GPU and no data beyond this repository.

| notebook | content |
|---|---|
| `reproduce_paper.ipynb` | main table at equal bytes, ablations, rate-distortion figure, refinement of other codes |
| `01_seeds_and_storage.ipynb` | payload bytes, fitting seeds, paired comparison, turbulence regimes |
| `02_rate_distortion.ipynb` | the sweep from 1000x to 100x |
| `03_diagnostics_outside_the_objective.ipynb` | heat-flux views, flux fitted on half of the modes, solver restarts |
| `04_frozen_decoders.ipynb` | AE, VAE and VQ-VAE latents refined through frozen decoders |
| `05_settings_stability_cost.ipynb` | recipe settings, damping, time and memory |

`results/rows/` and `results/experiments/` hold the rows; their READMEs describe the fields and say which rows come from this code and which from the PINC code and models (the neural fields and the autoencoders).

## Reproducing the rows
```bash
scripts/gyrosplat/reproduce_gkw.sh <data_root> <results_dir>          # all methods, 39 snapshots, both budgets
scripts/gyrosplat/reproduce_ablation.sh <data_root> <results_dir>     # the ablation rows at 302x
scripts/gyrosplat/reproduce_experiments.sh <data_root> <results_dir>  # seeds, equal-byte codes, sweeps
```
The scripts write one JSON per method, budget and snapshot and skip finished files; `CELLS` restricts the snapshots. One snapshot takes about twenty minutes for all methods on one GPU. Fits run in float32 with seeded placement, so a re-run reproduces the medians to the printed digits. Nine snapshots exist in two preprocessed versions; the rows use the second one (listed in the header of `reproduce_gkw.sh`).

## Scores
- **PSNR**: the reference's largest modulus squared over the mean squared error, for the distribution and for the potential.
- **Flux error**: relative L1 error of the heat-flux spectrum over the binormal modes 1 to 31.
- **kappa**: gross over net transport of the flux integrand; the tables report kappa / kappa_truth, 1 is ideal.
- **Compression ratio**: stored bytes against the raw snapshot. The comparison tables count 4 bytes per stored value and, for sparse codes, an int32 position per kept coefficient.

## Layout
```
📁 neugk/gyrosplat
├── 📄 recipe.py                      # the pipeline: compress / decode / score
├── 📄 config.py, cli.py              # recipe settings, command line
├── 📁 model                          # the method
│   ├── 📄 atoms.py, synth.py         # splats and rendering
│   ├── 📄 init.py, solve.py, train.py  # placement, amplitude solves, descent stages
│   ├── 📄 carriers.py, gn.py         # Gabor groups, flux refinement
│   └── 📄 normalize.py, pack.py      # normalization, storage accounting
├── 📁 backends, datasets             # synthetic and GKW readouts and loaders
├── 📁 baselines                      # Fourier, wavelets, PCA, ZFP/SZ3, plain splats, neural field
├── 📁 eval                           # the experiments
│   ├── 📄 metrics.py, diagnostics.py # scores, flux views
│   ├── 📄 ablation.py                # the ablation rows
│   └── 📁 paper                      # rows, tables and figures
└── 📁 results                        # stored per-snapshot rows

📁 configs/gyrosplat                  # recipes and dataset files
📁 scripts/gyrosplat                  # reproduction runs
📁 notebooks/gyrosplat                # tables and figures from the stored rows
📁 tests/gyrosplat                    # python -m pytest tests/gyrosplat
```
A new dataset needs one backend: how to transform the stored array to the spectral layout, how to solve the potential, how to integrate the flux. `backends/synthetic.py` is the template.

## Citing
```
@inproceedings{gutenbrunner2026gyrosplat,
    title={GyroSplat: A Linear Representation of Fusion Plasma Turbulence with Direct and Differentiable Physics},
    author={Gerald Gutenbrunner and Andreas Radler and Gianluca Galletti and Johannes Brandstetter and Fabian Paischer},
    booktitle={RPS Workshop at NeurIPS 2026},
    year={2026},
    url={https://openreview.net/forum?id=xZMuTU83eC}
}
```
