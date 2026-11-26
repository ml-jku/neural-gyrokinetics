# The rows behind the comparison at equal bytes and the appendix experiments

One JSON object per line unless noted; the fields follow `neugk/gyrosplat/eval/paper/rows.py`
(`psnr_f`, `psnr_phi` in dB, `flux_rel_l1` over the nonzero binormal modes, `kappa_rel` =
kappa / kappa_truth, `f_rel_l2`, `phi_rel_l2`) and the method keys of its `METHODS`.
`neugk/gyrosplat/eval/paper/experiments.py` loads the files and builds every table from them; the
notebooks `01` to `05` and the main notebook show them.

| file                      | rows  | content                                                                  |
|---------------------------|------:|--------------------------------------------------------------------------|
| `seeds.jsonl`             | 1,248 | GyroSplat and the neural field + physics (seeds 0-4), the field-only     |
|                           |       | neural field and splats (seeds 0-2), both budgets, 39 snapshots          |
| `byte_matched.jsonl`      |   468 | Fourier top-K, Fourier + refinement, wavelets Haar / bior4.4 / db4 and   |
|                           |       | Haar + refinement at 294,928 and 626,692 bytes (`stored_values`, `bytes`)|
|                           |       | with the heat-flux views outside the objective (`view_*`)                |
| `rate_distortion.jsonl`   | 5,928 | the sweep at eight budgets (`budget` = nominal ratio, `bytes`); seeded   |
|                           |       | methods seeds 0-2; the Fourier codes re-run at GyroSplat's exact bytes   |
| `views.jsonl`             |   663 | Q(kx), Q(s), Q(v_par, mu) and pointwise relative L1 errors of the stored |
|                           |       | fits (GyroSplat, neural fields, splats, AE, VAE, VQ-VAE, PCA, ZFP, SZ3)  |
| `heldout_modes.jsonl`     |    18 | GyroSplat at 302x with the flux fitted on the even or the odd modes      |
|                           |       | (`fit_band`), the flux spectra, and the fit of every mode for comparison |
| `restarts/<snap>.npz`     |     8 | time, heat flux and ky spectrum of the solver restarted from the         |
|                           |       | original snapshot, the original + 1e-6, GyroSplat, NF + physics, the AE  |
| `latent_refinement.jsonl` |   576 | AE, VAE, VQ-VAE: stored latent, 400 (or 2,000) Adam steps, then          |
|                           |       | Gauss-Newton or Levenberg-Marquardt; seconds per stage                   |
| `damping.jsonl`           |   132 | damping and step count of the final refinement on 12 snapshots, with    |
|                           |       | the saved joint-fit state each point starts from (`start_*`)             |
| `timing.jsonl`            |   252 | encode / decode seconds and peak GPU memory, one job per dedicated GPU;  |
|                           |       | `excluded` marks the jobs left out (a GPU's first job, a shared GPU)     |
| `operator_cost.jsonl`     |     9 | the diagnostic operator: construction time, peak memory, held bytes      |
| `storage.json`            |       | payload bytes per method and budget, the shared model sizes, the sweep   |
|                           |       | budgets and the stored GyroSplat file sizes                              |

The rows of GyroSplat, the plain splats, the Fourier and wavelet codes, PCA, ZFP, SZ3, the
field-only neural field of the sweep (`baselines/neural_field.py`) and the damping sweep were
produced with this package's code, one job per GPU (`scripts/gyrosplat/reproduce_experiments.sh` runs
the same configurations; the damping sweep re-ran only the last stage from saved joint-fit
states).  The neural fields of Table 1 (field objective and physics stage, all seeds), the
neural field + physics of the sweep, the AE / VAE / VQ-VAE latent refinement and the solver
restarts were run with the code and the trained models of PINC and a
gyrokinetic solver, and scored with the same functions; their rows are stored here only.  The fits of the flux on half of
the modes used the recipe with the flux objective and the Gauss-Newton refinement restricted
to those modes and the Gabor groups seeded on them (even: modes 2, 6, 10, 16, 20, 26, 30;
odd: 1, 5, 11, 15, 21, 25, 31).
