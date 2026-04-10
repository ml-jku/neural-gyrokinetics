"""Generate samples from a trained diffusion checkpoint and save them as H5 files.

Output per trajectory:
    {output_dir}/{output_prefix}_{traj_name}.h5
        /df              float32 (n_samples, C, *spatial)  — denormalized 5-D field
        /gt_flux_series  float32 (T,)                      — GT heat-flux time series
        /pred_flux       float32 (n_samples,)               — per-sample predicted flux
                                                              (only if calculate_mse_metrics)
    attrs: traj_name, n_samples, steps, gt_avg_flux, <cond_keys...>
"""

import argparse
import os

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

from neugk.diffusion import get_diffusion_runner
from neugk.integrals import FluxIntegral
from neugk.utils import recombine_zf


def _traj_basename(path: str) -> str:
    """Strip directory and known suffixes to get a clean trajectory name."""
    b = os.path.basename(path)
    for suf in ("_ifft_realpotens", "_ifft"):
        if b.endswith(suf):
            b = b[: -len(suf)]
    if b.endswith(".h5"):
        b = b[:-3]
    return b


def generate(gen_cfg_path: str) -> None:
    gen_cfg = OmegaConf.load(gen_cfg_path)

    diff_ckpt_dir = gen_cfg.diffusion_checkpoint_dir
    diff_cfg = OmegaConf.load(os.path.join(diff_ckpt_dir, "config.yaml"))

    # data path
    # data_path overrides the checkpoint config path for BOTH training and
    # generation data.  If generation targets live in a different directory,
    # set data_path to that directory AND provide inference.latent_scale so
    # that training-data access is not required.
    data_path = gen_cfg.get("data_path") or gen_cfg.get("generation_data_path")
    if data_path:
        diff_cfg.dataset.path = data_path

    # override val set with generation targets
    diff_cfg.dataset.validation_trajectories = list(gen_cfg.generation.trajectories)

    # setup runner
    diff_cfg.logging.writer = None
    diff_cfg.logging.tqdm = True
    diff_cfg.ddp.enable = False
    diff_cfg.training.num_workers = 2
    diff_cfg.training.pin_memory = True

    # runner: loads AE, precomputes/loads cached latents, builds model
    # NOTE: get_diffusion_runner calls set_seed which re-enables deterministic
    # algorithms, so we disable them the runner is created
    runner = get_diffusion_runner(rank=0, cfg=diff_cfg, world_size=1)
    torch.use_deterministic_algorithms(False)
    print(f"Model: {sum(p.numel() for p in runner.model.parameters()) / 1e6:.1f}M params")
    print(f"Targets: {[os.path.basename(f) for f in runner.valsets[0].files]}")

    # optional latent_scale override
    latent_scale_override = gen_cfg.inference.get("latent_scale")
    if latent_scale_override is not None:
        runner.latent_scale = float(latent_scale_override)
        print(f"latent_scale overridden to {runner.latent_scale:.6f}")
    else:
        print(f"latent_scale = {runner.latent_scale:.6f}  (computed from training set)")

    # load checkpoint weights
    snapshot = gen_cfg.get("model_snapshot", "best.pth")
    ckpt_path = os.path.join(diff_ckpt_dir, snapshot)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    runner.model.load_state_dict(ckpt["model_state_dict"])
    runner.model.eval()
    print(f"Loaded {snapshot} (epoch {ckpt['epoch']})")

    # generation config
    n_samples = int(gen_cfg.generation.n_samples)
    chunk_size = int(gen_cfg.generation.chunk_size)
    steps = int(gen_cfg.generation.steps)
    output_dir = gen_cfg.generation.output_dir
    output_prefix = gen_cfg.generation.get("output_prefix", "generated")
    calc_metrics = bool(gen_cfg.generation.get("calculate_mse_metrics", True))
    separate_zf = runner.cfg.dataset.separate_zf

    os.makedirs(output_dir, exist_ok=True)

    valset = runner.valsets[0]
    cond_keys = sorted(runner.cfg.model.conditioning)
    _cond_map = {"itg": "ion_temp_grad", "dg": "density_grad"}

    integrator = FluxIntegral(real_potens=False)
    integrator.cpu()

    # generate per trajectory
    for fi, fpath in enumerate(valset.files):
        traj_name = _traj_basename(fpath)
        meta = valset.metadata[fi]

        cond = torch.tensor(
            [float(np.squeeze(meta[_cond_map.get(k, k)])) for k in cond_keys],
            dtype=torch.float32,
        ).unsqueeze(0)

        cond_str = "  ".join(f"{k}={v:.3f}" for k, v in zip(cond_keys, cond[0].tolist()))
        print(f"\n[{traj_name}]  {cond_str}")

        # prefetch geometry for the full chunk size
        full_geom = valset.get_batch_geometry(
            torch.full((chunk_size,), fi, dtype=torch.long)
        )

        all_df: list[torch.Tensor] = []
        all_flux: list[float] = []

        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            bs = end - start
            c = cond.to(runner.device).expand(bs, -1)

            with torch.no_grad():
                preds = runner.sample(c, steps=steps, latent_only=False)

            pred_df = preds["df"].cpu()

            # denormalize each sample
            for b in range(bs):
                pred_df[b] = valset.denormalize(fi, df=pred_df[b])

            # recombine zonal flow into 2-channel (re, im)
            if separate_zf and pred_df.shape[1] > 2:
                pred_df = recombine_zf(pred_df, dim=1)

            all_df.append(pred_df)

            if calc_metrics:
                geom_chunk = {k: v[:bs] for k, v in full_geom.items()}
                _, (_, eflux, _) = integrator(geom_chunk, pred_df)
                all_flux.extend(eflux.cpu().numpy().flatten().tolist())

        df_array = torch.cat(all_df, dim=0).numpy().astype(np.float32)  # (N, C, ...)

        gt_flux_series = np.array(meta["flux"], dtype=np.float32)
        gt_avg_flux = float(np.mean(gt_flux_series[-80:]))

        if calc_metrics and all_flux:
            pm, ps = float(np.mean(all_flux)), float(np.std(all_flux))
            print(f"  pred_flux: {pm:.3f} ± {ps:.3f}   gt_avg: {gt_avg_flux:.3f}")

        # save H5
        out_path = os.path.join(output_dir, f"{output_prefix}_{traj_name}.h5")
        with h5py.File(out_path, "w") as hf:
            hf.create_dataset("df", data=df_array, compression="gzip", compression_opts=4)
            hf.create_dataset("gt_flux_series", data=gt_flux_series)
            if calc_metrics and all_flux:
                hf.create_dataset(
                    "pred_flux", data=np.array(all_flux, dtype=np.float32)
                )
            hf.attrs["traj_name"] = traj_name
            hf.attrs["n_samples"] = n_samples
            hf.attrs["steps"] = steps
            hf.attrs["gt_avg_flux"] = gt_avg_flux
            hf.attrs["cond_keys"] = ",".join(cond_keys)
            for k, v in zip(cond_keys, cond[0].tolist()):
                hf.attrs[k] = v

        print(f"  Saved {df_array.shape} to {out_path}")

    print(f"\nDone. {len(valset.files)} files written to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate samples from a diffusion checkpoint.")
    parser.add_argument("config", help="Path to generation yaml config")
    args = parser.parse_args()
    generate(args.config)
