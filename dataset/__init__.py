from torch.utils.data.dataloader import DataLoader

from dataset.augment import noise_transform
from dataset.cyclone import CycloneDataset


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
        trainset = CycloneDataset(
            path=cfg.dataset.path,
            split="train",
            random_seed=cfg.seed,
            test_ratio=0.0,
            in_memory=cfg.dataset.in_memory,
        )

        valset = CycloneDataset(
            path=cfg.dataset.path,
            split="val",
            random_seed=cfg.seed,
            test_ratio=0.0,
            in_memory=cfg.dataset.in_memory,
            n_eval_steps=cfg.validation.n_eval_steps,
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
        )
        valloader = DataLoader(
            valset,
            cfg.validation.batch_size,
            num_workers=cfg.training.num_workers,
            shuffle=False
        )

    return (trainset, valset), (trainloader, valloader), augmentations
