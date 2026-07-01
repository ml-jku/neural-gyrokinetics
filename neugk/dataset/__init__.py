import os
import torch
from torch.utils.data.dataloader import DataLoader

from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
import resource
from omegaconf import OmegaConf

from neugk.dataset.augment import noise_transform
from neugk.dataset.cyclone import (
    CycloneDataset,
    CoordinateCycloneDataset,
    CycloneSample,
)
from neugk.dataset.cyclone_diff import (
    CycloneAEDataset,
    CycloneVAEDataset,
    CycloneVQVAEDataset,
    CycloneSimSiamDataset,
    CycloneAESample,
)
from neugk.dataset.backend import H5Backend, KvikIOBackend
from neugk.dataset.augment import mask_modes


def set_ulimit(limit: int = 65536):
    # increase file descriptor limit
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft < limit:
            new_soft = min(limit, hard)
            resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
    except (ValueError, resource.error):
        pass


def bind_worker_to_numa_node():
    # bind worker to same NUMA node as parent
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None:
        try:
            import ctypes

            libnuma = ctypes.CDLL("libnuma.so.1", use_errno=True)
            if libnuma.numa_available() != -1:
                libnuma.numa_set_preferred.argtypes = [ctypes.c_int]
                libnuma.numa_set_preferred(int(local_rank))
        except (OSError, AttributeError):
            pass


def _worker_init_fn(worker_id):
    _ = worker_id
    bind_worker_to_numa_node()
    set_ulimit()


def check_partial_holdouts(dataset_cfg):
    # ensure each trajectory in partial holdouts also appears in training
    for entry in dataset_cfg.partial_holdouts:
        file = entry.trajectory
        if file not in dataset_cfg.training_trajectories:
            raise ValueError(
                f"Trajectory '{file}' in partial_holdouts is not in training_trajectories."
            )
    return


def _ae_model_name(cfg) -> str:
    ckp_path = getattr(cfg, "ae_checkpoint", None)
    if not ckp_path or not os.path.exists(ckp_path):
        return ""
    cfg_path = os.path.join(str(ckp_path), "config.yaml")
    if not os.path.exists(cfg_path):
        return ""
    try:
        ae_cfg = OmegaConf.load(cfg_path)
    except Exception:
        return ""
    return str(getattr(ae_cfg.model, "name", "")).lower()


def _is_vqvae_checkpoint(cfg) -> bool:
    return "vqvae" in _ae_model_name(cfg)


def _is_vae_checkpoint(cfg) -> bool:
    name = _ae_model_name(cfg)
    return "vae" in name and "vqvae" not in name


def get_data(cfg, rank: int = 0):
    # increase file descriptor limit for CUDA IPC
    set_ulimit()
    assert cfg.dataset.name in ["cyclone"]
    backend = getattr(cfg.dataset, "backend", "h5")
    use_ddp = dist.is_initialized()
    partial_holdouts = {}
    if cfg.dataset.partial_holdouts:
        # validate config
        check_partial_holdouts(cfg.dataset)
        for entry in cfg.dataset.partial_holdouts:
            file = entry.trajectory
            last_n = entry.last_n
            partial_holdouts[file] = last_n

    if cfg.workflow == "gyroswin":
        use_kvikio_train = True
        input_fields = set(
            cfg.dataset.input_fields
            + [
                k
                for k in cfg.model.loss_weights.keys()
                if cfg.model.loss_weights[k] > 0.0 or cfg.model.loss_scheduler[k]
            ]
        )
        if not input_fields.issubset({"df", "phi", "flux", "fluxavg"}):
            raise ValueError(f"{input_fields} contains unknown values")
        if cfg.model.name in ["pointnet", "transolver", "transformer"]:
            input_fields.add("position")
        assert not (
            "flux" in input_fields and "fluxavg" in input_fields
        ), "Cannot predict both fluxavg and flux..."
        train_input_fields = val_input_fields = sorted(input_fields)
        # crop end of trajectory for autoregressive evaluation
        train_kwargs = {}
        val_kwargs = {"tail_offset": cfg.validation.n_eval_steps}
        if cfg.model.name in ["pointnet", "transolver", "transformer"]:
            # these models use coordinates as input
            dataset_class = CoordinateCycloneDataset
        # elif cfg.choices.model == "baselines/linear_ablation":
        #     dataset_class = LinearCycloneDataset
        else:
            dataset_class = CycloneDataset
    elif cfg.workflow == "pinc":
        use_kvikio_train = True
        train_input_fields = ["df", "phi", "flux"]
        val_input_fields = ["df", "phi", "flux"]
        # serve GT spectra (kyspec/fluxspec) when gated in dataset.input_fields
        for _sk in ("kyspec", "fluxspec"):
            if _sk in set(getattr(cfg.dataset, "input_fields", []) or []):
                train_input_fields.append(_sk)
                val_input_fields.append(_sk)

        enc_cond = getattr(cfg.model, "encoder_conditioning", [])
        dec_cond = getattr(cfg.model, "decoder_conditioning", [])
        conditioning = sorted(list(set(enc_cond) | set(dec_cond)))

        train_kwargs = {"conditions": conditioning}
        val_kwargs = {"conditions": conditioning}
        if cfg.stage == "simsiam":
            dataset_class = CycloneSimSiamDataset
        else:
            dataset_class = CycloneAEDataset
    elif cfg.workflow == "diffusion":
        # no need for gds in diffusion
        use_kvikio_train = getattr(cfg.dataset, "gds_override", False)
        train_input_fields = ["df", "phi", "flux"]  # cfg.dataset.input_fields
        val_input_fields = ["df", "phi", "flux"]

        use_vqvae_latents = _is_vqvae_checkpoint(cfg)
        use_vae_latents = _is_vae_checkpoint(cfg)

        if use_vqvae_latents:
            dataset_class = CycloneVQVAEDataset
        elif use_vae_latents:
            dataset_class = CycloneVAEDataset
        else:
            dataset_class = CycloneAEDataset
        train_kwargs = {"conditions": sorted(cfg.model.conditioning)}
        val_kwargs = {"conditions": sorted(cfg.model.conditioning)}

        if use_vae_latents:
            latent_sampling_mode = getattr(
                cfg.dataset, "latent_sampling_mode", "stochastic"
            )
            val_latent_sampling_mode = getattr(
                cfg.dataset, "val_latent_sampling_mode", latent_sampling_mode
            )
            train_kwargs["latent_sampling_mode"] = latent_sampling_mode
            val_kwargs["latent_sampling_mode"] = val_latent_sampling_mode

        # latent_scaling_mode: "global", "per_channel", "per_token"
        latent_scaling_mode = getattr(cfg.dataset, "latent_scaling_mode", "global")
        train_kwargs["latent_scaling_mode"] = latent_scaling_mode
        val_kwargs["latent_scaling_mode"] = latent_scaling_mode

        if rank == 0:
            latent_type = (
                "VQVAE" if use_vqvae_latents else "VAE" if use_vae_latents else "AE"
            )
            print(f"Diffusion latent dataset mode: {latent_type}")

        # load AE cfg for normalization stats
        ae_checkpoint = getattr(cfg, "ae_checkpoint", None)
        ae_cfg = None
        if ae_checkpoint and os.path.isdir(str(ae_checkpoint)):
            ae_cfg_path = os.path.join(str(ae_checkpoint), "config.yaml")
            if os.path.exists(ae_cfg_path):
                ae_cfg = OmegaConf.load(ae_cfg_path)
                if rank == 0:
                    print(f"Loaded AE config for normalization from {ae_cfg_path}")
        train_kwargs["ae_cfg"] = ae_cfg
    else:
        raise NotImplementedError

    if not rank:
        print(f"Loading {train_input_fields} in dataset")

    # bf16 train: prefer bf16 shards, uniform bf16 batch (fast reads converted, f32 downcast otherwise); val always f32; default unchanged f32
    _prefer_dtype = getattr(cfg.dataset, "prefer_dtype", None)
    _train_dtype = torch.bfloat16 if _prefer_dtype == "bf16" else torch.float32

    # dataloading backend
    if backend == "h5":
        train_backend = H5Backend(rank)
        val_backend = H5Backend(rank)
    elif backend == "gds":
        train_backend = KvikIOBackend(
            rank, use_kvikio=use_kvikio_train, prefer_dtype=_prefer_dtype
        )
        # NOTE: for validation load without gds, save space, slow is acceptable
        val_backend = KvikIOBackend(rank, use_kvikio=False, prefer_dtype=_prefer_dtype)

    trainset = dataset_class(
        backend=train_backend,
        active_keys=cfg.dataset.active_keys,
        fields_to_load=train_input_fields,
        probe_targets=cfg.validation.probe.targets,
        path=cfg.dataset.path,
        split="train",
        random_seed=cfg.seed,
        normalization=cfg.dataset.normalization,
        normalization_scope=cfg.dataset.normalization_scope,
        spatial_ifft=cfg.dataset.spatial_ifft,
        bundle_seq_length=cfg.model.bundle_seq_length,
        trajectories=cfg.dataset.training_trajectories,
        partial_holdouts=partial_holdouts,
        cond_filters=cfg.dataset.training_cond_filters,
        subsample=cfg.dataset.subsample,
        log_transform=cfg.dataset.log_transform,
        split_into_bands=cfg.dataset.split_into_bands,
        minmax_beta1=cfg.dataset.minmax_beta1,
        minmax_beta2=cfg.dataset.minmax_beta2,
        offset=cfg.dataset.offset,
        timestep_std_filter=cfg.dataset.timestep_std_filter,
        separate_zf=cfg.dataset.separate_zf,
        num_workers=cfg.dataset.num_workers,
        real_potens=cfg.dataset.real_potens,
        decouple_mu=cfg.dataset.norm_decouple_mu,
        dtype=_train_dtype,
        rank=rank,
        **train_kwargs,
    )

    holdout_trajectories_valset = dataset_class(
        backend=val_backend,
        active_keys=cfg.dataset.active_keys,
        fields_to_load=val_input_fields,
        probe_targets=cfg.validation.probe.targets,
        path=cfg.dataset.path,
        split="val",
        random_seed=cfg.seed,
        normalization=cfg.dataset.normalization,
        normalization_scope=cfg.dataset.normalization_scope,
        normalization_stats=getattr(trainset, "stats", None),
        spatial_ifft=cfg.dataset.spatial_ifft,
        bundle_seq_length=cfg.model.bundle_seq_length,
        trajectories=cfg.dataset.validation_trajectories,
        cond_filters=cfg.dataset.eval_cond_filters,
        subsample=getattr(cfg.dataset, "val_subsample", 1),
        log_transform=cfg.dataset.log_transform,
        split_into_bands=cfg.dataset.split_into_bands,
        minmax_beta1=cfg.dataset.minmax_beta1,
        minmax_beta2=cfg.dataset.minmax_beta2,
        offset=cfg.dataset.offset,
        timestep_std_filter=cfg.dataset.timestep_std_filter,
        timestep_std_offset=cfg.dataset.timestep_std_offset,
        separate_zf=cfg.dataset.separate_zf,
        num_workers=cfg.dataset.num_workers,
        real_potens=cfg.dataset.real_potens,
        decouple_mu=cfg.dataset.norm_decouple_mu,
        rank=rank,
        **val_kwargs,
    )

    # gpudirect storage only used if kvikio is required, otherwise raw bins
    use_gpudirect = backend == "gds" and use_kvikio_train
    # must be false when returning gpu data
    pin_memory = cfg.training.pin_memory and not use_gpudirect
    prefetch_factor = min(2, cfg.training.num_workers // 2) if backend != "gds" else 1
    dataloader_kwargs = {}
    if cfg.training.num_workers > 0:
        # increase FD limit on each subprocess for large batch sizes
        dataloader_kwargs["worker_init_fn"] = _worker_init_fn

    if use_gpudirect:
        # cannot set context to dataloader workers with gds
        if cfg.training.num_workers > 0:
            dataloader_kwargs["multiprocessing_context"] = mp.get_context("spawn")
        # keep memory low
        prefetch_factor = 1

    trainloader = DataLoader(
        trainset,
        cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        shuffle=True if not use_ddp else False,
        collate_fn=trainset.collate,
        pin_memory=pin_memory,
        sampler=DistributedSampler(trainset) if use_ddp else None,
        persistent_workers=cfg.training.num_workers > 0,
        prefetch_factor=prefetch_factor if cfg.training.num_workers > 0 else None,
        **dataloader_kwargs,
    )

    holdout_trajectories_valloader = DataLoader(
        holdout_trajectories_valset,
        cfg.validation.batch_size,
        num_workers=cfg.training.num_workers,
        shuffle=False,
        collate_fn=holdout_trajectories_valset.collate,
        pin_memory=pin_memory,
        sampler=(DistributedSampler(holdout_trajectories_valset) if use_ddp else None),
        persistent_workers=cfg.training.num_workers > 0,
        prefetch_factor=prefetch_factor if cfg.training.num_workers > 0 else None,
        **dataloader_kwargs,
    )

    if partial_holdouts:
        holdout_samples_valset = dataset_class(
            backend=val_backend,
            active_keys=cfg.dataset.active_keys,
            fields_to_load=val_input_fields,
            probe_targets=cfg.validation.probe.targets,
            path=cfg.dataset.path,
            split="val",
            random_seed=cfg.seed,
            normalization=cfg.dataset.normalization,
            normalization_scope=cfg.dataset.normalization_scope,
            normalization_stats=getattr(trainset, "norm_stats", None),
            spatial_ifft=cfg.dataset.spatial_ifft,
            bundle_seq_length=cfg.model.bundle_seq_length,
            trajectories=cfg.dataset.training_trajectories,
            partial_holdouts=partial_holdouts,
            cond_filters=cfg.dataset.eval_cond_filters,
            subsample=cfg.dataset.subsample,
            log_transform=cfg.dataset.log_transform,
            minmax_beta1=cfg.dataset.minmax_beta1,
            minmax_beta2=cfg.dataset.minmax_beta2,
            offset=cfg.dataset.offset,
            timestep_std_filter=cfg.dataset.timestep_std_filter,
            timestep_std_offset=cfg.dataset.timestep_std_offset,
            separate_zf=cfg.dataset.separate_zf,
            num_workers=cfg.dataset.num_workers,
            real_potens=cfg.dataset.real_potens,
            decouple_mu=cfg.dataset.norm_decouple_mu,
            rank=rank,
        )
        holdout_samples_valloader = DataLoader(
            holdout_samples_valset,
            cfg.validation.batch_size,
            num_workers=cfg.training.num_workers,
            shuffle=False,
            collate_fn=holdout_samples_valset.collate,
            pin_memory=pin_memory,
            sampler=(DistributedSampler(holdout_samples_valset) if use_ddp else None),
            persistent_workers=cfg.training.num_workers > 0,
            prefetch_factor=prefetch_factor if cfg.training.num_workers > 0 else None,
            **dataloader_kwargs,
        )

    augmentations = []
    for key in cfg.dataset.augment:
        if cfg.dataset.augment[key].active:
            if key == "noise":
                augmentations.append(
                    noise_transform(
                        std=cfg.dataset.augment.noise.noise_std,
                        window_size=cfg.model.bundle_seq_length,
                    )
                )
            elif key == "mask_modes":
                mix_weights = getattr(
                    cfg.dataset.augment.mask_modes, "mix_weights", None
                )
                cutoff = getattr(cfg.dataset.augment.mask_modes, "cutoff", None)
                augmentations.append(
                    mask_modes(
                        mask_ratio=cfg.dataset.augment.mask_modes.mask_ratio,
                        strategy=cfg.dataset.augment.mask_modes.strategy,
                        cutoff=cutoff,
                        mix_weights=mix_weights,
                        is_fourier=cfg.dataset.augment.mask_modes.is_fourier,
                        rescale=cfg.dataset.augment.mask_modes.rescale,
                        zf_separated=cfg.dataset.separate_zf,
                        weights=cfg.dataset.augment.mask_modes.weights,
                        mask_zero_mode=cfg.dataset.augment.mask_modes.mask_zero_mode,
                        denormalize_fn=(
                            trainset.denormalize
                            if not cfg.dataset.augment.mask_modes.is_fourier
                            else None
                        ),
                        normalize_fn=(
                            trainset.normalize
                            if not cfg.dataset.augment.mask_modes.is_fourier
                            else None
                        ),
                        per_sample=getattr(
                            cfg.dataset.augment.mask_modes, "per_sample", False
                        ),
                    )
                )
            elif key in ["vicreg_variance", "vicreg_covariance", "logdet"]:
                # compute loss on latents, no augmentation function needed
                pass
            else:
                raise ValueError(f"Unknown augmentation: {key}")

    datasets = (trainset, holdout_trajectories_valset)
    dataloaders = (trainloader, holdout_trajectories_valloader)

    val_ratio = len(holdout_trajectories_valset) / len(trainset)

    if rank == 0:
        print(f"Train: {len(trainset)}")
        print(f"Holdout trajectories (val): {len(holdout_trajectories_valset)}")

    if partial_holdouts:
        val_ratio = (
            len(holdout_samples_valset) + len(holdout_trajectories_valset)
        ) / len(trainset)
        if rank == 0:
            print(f"Holdout samples (val): {len(holdout_samples_valset)}")
        datasets = ((trainset, holdout_trajectories_valset, holdout_samples_valset),)
        dataloaders = (
            trainloader,
            holdout_trajectories_valloader,
            holdout_samples_valloader,
        )

    if rank == 0:
        print(f"Validation ratio: {val_ratio:.2f}")
    return datasets, dataloaders, augmentations


__all__ = [
    "get_data",
    "CycloneDataset",
    "CycloneSample",
    "CycloneAEDataset",
    "CycloneVAEDataset",
    "CycloneVQVAEDataset",
    "CycloneAESample",
]
