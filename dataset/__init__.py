from torch.utils.data.dataloader import DataLoader

from dataset.augment import noise_transform
from dataset.cyclone import CycloneDataset, CycloneSample


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
            input_seq_length=cfg.model.input_seq_length,
            target_seq_length=cfg.model.bundle_seq_length,
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
            n_eval_steps=cfg.validation.n_eval_steps,
            input_seq_length=cfg.model.input_seq_length,
            target_seq_length=cfg.model.bundle_seq_length,
            trajectories=cfg.dataset.validation_trajectories,
        )

        # testset = CycloneDataset(
        #     path=cfg.dataset.path,
        #     split="test",
        #     random_seed=cfg.seed,
        # )

        print(f"Train: {len(trainset)} samples")
        print(f"Val: {len(valset)} samples")
        # print(f"Test: {len(testset)} samples")

        trainloader = DataLoader(
            trainset,
            cfg.training.batch_size,
            num_workers=cfg.training.num_workers,
            shuffle=True,
            collate_fn=trainset.collate,
            pin_memory=cfg.training.pin_memory,
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
