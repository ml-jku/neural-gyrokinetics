from torch.utils.data.dataloader import DataLoader

from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
import resource

from neugk.dataset.augment import noise_transform
from neugk.dataset.cyclone import (
    CycloneDataset,
    CoordinateCycloneDataset,
    CycloneSample,
)
from neugk.dataset.cyclone_diff import (
    CycloneAEDataset,
    CycloneSimSiamDataset,
    CycloneAESample,
)
from neugk.dataset.backend import H5Backend, KvikIOBackend
from neugk.dataset.augment import mask_modes


def set_ulimit(limit: int = 65536):
    # os.system(f"ulimit -n {limit}")
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft < limit:
            new_soft = min(limit, hard)
            resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
    except (ValueError, resource.error):
        pass


def _worker_init_fn(worker_id):
    _ = worker_id
    set_ulimit()


def check_partial_holdouts(dataset_cfg):
    # check that each trajectory in partial holdouts also appears in training
    for entry in dataset_cfg.partial_holdouts:
        file = entry.trajectory
        if file not in dataset_cfg.training_trajectories:
            raise ValueError(
                f"Trajectory '{file}' in partial_holdouts is not in training_trajectories."
            )
    return


def get_data(cfg, rank: int = 0):
    # increase file descriptor limit for CUDA IPC and shared memory handles
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
        # exclude fields not in dataset
        # input_fields = input_fields.intersection({"df", "phi", "flux"})
        if input_fields.union({"df", "phi", "flux"}) != {"df", "phi", "flux"}:
            raise ValueError(f"{input_fields} contains unknown values")
        if cfg.model.name in ["pointnet", "transolver", "transformer"]:
            input_fields.add("position")
        assert not (
            "flux" in input_fields and "fluxavg" in input_fields
        ), "Cannot predict both fluxavg and flux..."
        train_input_fields = val_input_fields = input_fields
        # NOTE: for autoregressive evaluation, crop end of trajectory
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
        train_kwargs = {"conditions": list(cfg.model.conditioning)}
        val_kwargs = {"conditions": list(cfg.model.conditioning)}
        if cfg.stage == "simsiam":
            dataset_class = CycloneSimSiamDataset
        else:
            dataset_class = CycloneAEDataset
    elif cfg.workflow == "diffusion":
        # no need for gds in diffusion
        use_kvikio_train = False
        train_input_fields = ["df", "phi", "flux"]  # cfg.dataset.input_fields
        val_input_fields = ["df", "phi", "flux"]
        train_kwargs = {"conditions": list(cfg.model.conditioning)}
        val_kwargs = {"conditions": list(cfg.model.conditioning)}
        dataset_class = CycloneAEDataset
    else:
        raise NotImplementedError

    if not rank:
        print(f"Loading {train_input_fields} in dataset")

    # dataloading backend
    if backend == "h5":
        train_backend = H5Backend(rank)
        val_backend = H5Backend(rank)
    elif backend == "gds":
        train_backend = KvikIOBackend(rank, use_kvikio=use_kvikio_train)
        # NOTE: for validation load without gds, save space, slow is acceptable
        val_backend = KvikIOBackend(rank, use_kvikio=False)

    trainset = dataset_class(
        backend=train_backend,
        active_keys=cfg.dataset.active_keys,
        fields_to_load=train_input_fields,
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
        **train_kwargs,
    )

    holdout_trajectories_valset = dataset_class(
        backend=val_backend,
        active_keys=cfg.dataset.active_keys,
        fields_to_load=val_input_fields,
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
        **val_kwargs,
    )

    # gpudirect storage only used if kvikio is required, oterwise raw bins
    use_gpudirect = backend == "gds" and use_kvikio_train
    # dataloaders
    prefetch_factor = min(2, cfg.training.num_workers // 2)
    # NOTE: must be false when returning gpu data
    pin_memory = cfg.training.pin_memory and not use_gpudirect
    dataloader_kwargs = {}
    if cfg.training.num_workers > 0:
        dataloader_kwargs["worker_init_fn"] = _worker_init_fn

    if use_gpudirect:
        # cannot for context to dataloader workers with gds
        if cfg.training.num_workers > 0:
            dataloader_kwargs["multiprocessing_context"] = mp.get_context("spawn")
        # keep memory requirements low
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
                augmentations.append(
                    mask_modes(
                        mask_ratio=cfg.dataset.augment.mask_modes.mask_ratio,
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
                    )
                )
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
    "CycloneAESample",
]
