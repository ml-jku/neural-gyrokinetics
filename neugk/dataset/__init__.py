from torch.utils.data.dataloader import DataLoader

from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from neugk.dataset.augment import noise_transform
from neugk.dataset.cyclone import (
    CycloneDataset,
    CycloneSample,
    CoordinateCycloneDataset,
)
from neugk.dataset.cyclone_diff import CycloneAEDataset, CycloneAESample
from neugk.dataset.cyclone_diff_simsiam import CycloneSimSiamDataset


def check_partial_holdouts(dataset_cfg):
    # check that each trajectory in partial holdouts also appears in training
    for entry in dataset_cfg.partial_holdouts:
        file = entry.trajectory
        if file not in dataset_cfg.training_trajectories:
            raise ValueError(
                f"Trajectory '{file}' in partial_holdouts is not in training_trajectories."
            )
    return


def get_data(cfg):
    assert cfg.dataset.name in ["cyclone"]

    augmentations = []
    if cfg.dataset.augment.noise is True:
        augmentations.append(
            noise_transform(
                std=cfg.dataset.augment.noise_std,
                window_size=cfg.model.bundle_seq_length,
            )
        )

    use_ddp = dist.is_initialized()
    if cfg.dataset.name == "cyclone":
        partial_holdouts = {}
        if cfg.dataset.partial_holdouts:
            # validate config
            check_partial_holdouts(cfg.dataset)
            for entry in cfg.dataset.partial_holdouts:
                file = entry.trajectory
                last_n = entry.last_n
                partial_holdouts[file] = last_n

        if cfg.workflow == "gyroswin":
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

            train_input_fields = ["df", "phi", "flux"]
            val_input_fields = ["df", "phi", "flux"]
            train_kwargs = {"conditions": list(cfg.model.conditioning)}
            val_kwargs = {"conditions": list(cfg.model.conditioning)}
            if cfg.stage == "simsiam":
                dataset_class = CycloneSimSiamDataset
            else:
                dataset_class = CycloneAEDataset
        elif cfg.workflow == "diffusion":
            train_input_fields = cfg.dataset.input_fields
            val_input_fields = ["df", "phi", "flux"]
            train_kwargs = {"conditions": list(cfg.model.conditioning)}
            val_kwargs = {"conditions": list(cfg.model.conditioning)}
            dataset_class = CycloneAEDataset
            # dataset_class = CycloneAEDatasetGaussianized
        else:
            raise NotImplementedError

        trainset = dataset_class(
            active_keys=cfg.dataset.active_keys,
            input_fields=train_input_fields,
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
            separate_zf=cfg.dataset.separate_zf,
            num_workers=cfg.dataset.num_workers,
            real_potens=cfg.dataset.real_potens,
            **train_kwargs,
        )

        holdout_trajectories_valset = dataset_class(
            active_keys=cfg.dataset.active_keys,
            input_fields=val_input_fields,
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
            subsample=cfg.dataset.subsample,
            log_transform=cfg.dataset.log_transform,
            split_into_bands=cfg.dataset.split_into_bands,
            minmax_beta1=cfg.dataset.minmax_beta1,
            minmax_beta2=cfg.dataset.minmax_beta2,
            offset=cfg.dataset.offset,
            separate_zf=cfg.dataset.separate_zf,
            num_workers=cfg.dataset.num_workers,
            real_potens=cfg.dataset.real_potens,
            **val_kwargs,
        )

        trainloader = DataLoader(
            trainset,
            cfg.training.batch_size,
            num_workers=cfg.training.num_workers,
            shuffle=True if not use_ddp else False,
            collate_fn=trainset.collate,
            pin_memory=cfg.training.pin_memory,
            sampler=DistributedSampler(trainset) if use_ddp else None,
            persistent_workers=True,
            prefetch_factor=cfg.training.num_workers // 2,
        )

        holdout_trajectories_valloader = DataLoader(
            holdout_trajectories_valset,
            cfg.validation.batch_size,
            num_workers=cfg.training.num_workers,
            shuffle=False,
            collate_fn=holdout_trajectories_valset.collate,
            pin_memory=cfg.training.pin_memory,
            sampler=(
                DistributedSampler(holdout_trajectories_valset) if use_ddp else None
            ),
            persistent_workers=True,
            prefetch_factor=cfg.training.num_workers // 2,
        )

        if partial_holdouts:
            holdout_samples_valset = dataset_class(
                active_keys=cfg.dataset.active_keys,
                input_fields=input_fields,
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
                separate_zf=cfg.dataset.separate_zf,
                num_workers=cfg.dataset.num_workers,
                real_potens=cfg.dataset.real_potens,
            )
            holdout_samples_valloader = DataLoader(
                holdout_samples_valset,
                cfg.validation.batch_size,
                num_workers=cfg.training.num_workers,
                shuffle=False,
                collate_fn=holdout_samples_valset.collate,
                pin_memory=cfg.training.pin_memory,
                sampler=(
                    DistributedSampler(holdout_samples_valset) if use_ddp else None
                ),
            )

        print(f"Train: {len(trainset)} samples")
        print(f"Holdout trajectories Val: {len(holdout_trajectories_valset)} samples")

        if partial_holdouts:
            print(f"Holdout samples Val: {len(holdout_samples_valset)} samples")
            print(
                f"Validation ratio: {(len(holdout_samples_valset) + len(holdout_trajectories_valset)) / len(trainset):.2f}"
            )
            return (
                (trainset, holdout_trajectories_valset, holdout_samples_valset),
                (
                    trainloader,
                    holdout_trajectories_valloader,
                    holdout_samples_valloader,
                ),
                augmentations,
            )

    print(f"Validation ratio: {len(holdout_trajectories_valset) / len(trainset):.2f}")
    return (
        (trainset, holdout_trajectories_valset),
        (trainloader, holdout_trajectories_valloader),
        augmentations,
    )


__all__ = [
    "get_data",
    "CycloneDataset",
    "CycloneSample",
    "CycloneAEDataset",
    "CycloneAESample",
]
