from torch.utils.data.dataloader import DataLoader

from dataset.augment import noise_transform
from dataset.cyclone import CycloneDataset, CycloneSample
from torch.utils.data.distributed import DistributedSampler


def get_data(cfg):
    assert cfg.dataset.name in ["cyclone"]

    augmentations = []
    if cfg.dataset.augment.noise is True:
        augmentations.append(
            noise_transform(
                std=cfg.dataset.augment.noise_std,
                window_size=cfg.model.input_seq_length,
            )
        )

    if cfg.dataset.name == "cyclone":
        if cfg.dataset.in_memory:
            print("Loading dataset in memory!")

        trainset = CycloneDataset(
            active_keys=cfg.dataset.active_keys,
            path=cfg.dataset.path,
            split="train",
            random_seed=cfg.seed,
            test_ratio=0.0,
            normalization=cfg.dataset.normalization,
            spatial_ifft=cfg.dataset.spatial_ifft,
            in_memory=cfg.dataset.in_memory,
            bundle_seq_length=cfg.model.bundle_seq_length,
            trajectories=cfg.dataset.training_trajectories,
        )

        valset = CycloneDataset(
            active_keys=cfg.dataset.active_keys,
            path=cfg.dataset.path,
            split="val",
            random_seed=cfg.seed,
            test_ratio=0.0,
            normalization=cfg.dataset.normalization,
            spatial_ifft=cfg.dataset.spatial_ifft,
            in_memory=cfg.dataset.in_memory,
            bundle_seq_length=cfg.model.bundle_seq_length,
            trajectories=cfg.dataset.validation_trajectories,
        )

        print(f"Train: {len(trainset)} samples")
        print(f"Val: {len(valset)} samples")

        trainloader = DataLoader(
            trainset,
            cfg.training.batch_size,
            num_workers=cfg.training.num_workers,
            shuffle=True if not cfg.use_ddp else False,
            collate_fn=trainset.collate,
            pin_memory=cfg.training.pin_memory,
            sampler=DistributedSampler(trainset) if cfg.use_ddp else None,
        )

        valloader = DataLoader(
            valset,
            cfg.validation.batch_size,
            num_workers=cfg.training.num_workers,
            shuffle=False,
            collate_fn=valset.collate,
            pin_memory=cfg.training.pin_memory,
        )

    return (trainset, valset), (trainloader, valloader), augmentations



__all__ = ["get_data", "CycloneDataset", "CycloneSample"]
