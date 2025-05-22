import numpy as np
from torch.utils.data.dataloader import DataLoader

from dataset.augment import noise_transform
from dataset.cyclone import CycloneDataset, CycloneSample
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

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

        trainset = CycloneDataset(
            active_keys=cfg.dataset.active_keys,
            input_fields=["df", "phi", "flux"],  # TODO figure out how to deal with eval
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
        )

        holdout_trajectories_valset = CycloneDataset(
            active_keys=cfg.dataset.active_keys,
            input_fields=["df", "phi", "flux"],
            path=cfg.dataset.path,
            split="val",
            random_seed=cfg.seed,
            normalization=cfg.dataset.normalization,
            normalization_scope=cfg.dataset.normalization_scope,
            normalization_stats=trainset.norm_stats,
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
            prefetch_factor=cfg.training.num_workers,
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
            prefetch_factor=cfg.training.num_workers,
        )

        if partial_holdouts:
            holdout_samples_valset = CycloneDataset(
                active_keys=cfg.dataset.active_keys,
                input_fields=["df", "phi", "flux"],
                path=cfg.dataset.path,
                split="val",
                random_seed=cfg.seed,
                normalization=cfg.dataset.normalization,
                normalization_scope=cfg.dataset.normalization_scope,
                normalization_stats=trainset.norm_stats,
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


__all__ = ["get_data", "CycloneDataset", "CycloneSample"]
