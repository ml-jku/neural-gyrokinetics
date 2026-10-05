# Validation metrics

Logged under `val_traj/`. Names changed relative to `main`; old → new below.
`—` means the metric was dropped (old) or is new (new).

## AE (`AEEvaluator`)

| main | now | definition |
|---|---|---|
| `df_mse` | `df_mse` | MSE of the normalized reconstruction |
| — | `df_rel_l2` | relative L2 of the denormalized df (zf recombined) |
| `phi_int_mse` | `phi_int_mse` | MSE of the field-solved phi of reconstruction vs target; now real-space `(x, s, y)` instead of the complex spectral potential |
| — | `phi_int_rel_l2` | relative L2 of the same pair |
| `flux_int_mse` | `flux_int_mse` | squared error of the integrated heat flux, reconstruction vs target df |
| — | `flux_int_rel_err` | `|eflux_pred - eflux_tgt| / |eflux_tgt|` |
| `flux_target_mse` | — | dropped (reconstruction flux vs the metadata flux) |

## Diffusion (`DiffusionEvaluator`, `FlowMatchingRunner`)

| main | now | definition |
|---|---|---|
| `fm_loss` | `fm_loss` | flow-matching loss over the whole val set, fixed noise key |
| `df` | `df_mse` | MSE of the normalized decoded sample vs the target snapshot |
| — | `df_rel_l2` | relative L2 of the denormalized pair |
| `avg_flux_rmse` | `avg_flux_rmse` | RMSE of the per-trajectory mean sampled flux vs `avg_flux` |
| `avg_flux_rel_mae` | `avg_flux_rel_err` | mean `|mean - gt| / |gt|` over trajectories (denominator now `|gt|`) |
| `avg_flux_corr`, `avg_flux_slope`, `avg_flux_{pred,std,gt}/<traj>` | unchanged | |

## GyroSwin (`GyroSwinEvaluator`), per rollout step `_x{t}` and step mean

| main | now | definition |
|---|---|---|
| `df_x{t}`, `phi_x{t}` | `df_x{t}`, `phi_x{t}` | relative-norm MSE `‖p - y‖² / (‖y‖² + 1e-4)` of the denormalized fields (was plain MSE of normalized fields) |
| — | `df_rel_l2_x{t}`, `phi_rel_l2_x{t}` | relative L2 of the denormalized fields |
| — | `flux_x{t}`, `fluxavg_x{t}` | MSE of the flux heads (denormalized) |
| — | `phi_int_x{t}` | MSE of the phi field-solved from the predicted df vs target phi |
| — | `flux_int_rel_err_x{t}` | `|eflux - flux| / |flux|` of the integrated predicted heat flux vs the target flux |
| — | `{name}` | mean of `{name}_x{t}` over the rollout steps |

The training loss terms `phi_int` / `flux_int` (`model.extra_loss_weights`) keep their loss
form (`flux_int = pflux² + (eflux - flux)²`); only the validation metric is the relative error.

## Spectral (`validation.eval_spectra`, AE and diffusion)

`kyspec_*`, `qspec_*` (`pc`, `sc`, `l1`, `rl2`, `rl1`, `wd`), `zfphi_rl2`, `zfflow_rl2`,
`zfshear_rl2`, `zf_energy_err`: unchanged names; per-trajectory time averages, mean over
trajectories, now over the validation trajectories of every process.
