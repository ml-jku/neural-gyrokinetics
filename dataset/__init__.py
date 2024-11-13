from torch.utils.data.dataloader import DataLoader
from functools import partial

from dataset.mhd import MHDDataset
from dataset.augment import noise_transform


def get_data(cfg):
    trainset = MHDDataset(
        path=cfg.dataset.path,
        split="train",
        active_keys=cfg.dataset.active_keys,
        input_seq_length=cfg.model.input_seq_length,
        target_seq_length=cfg.training.target_seq_length,
        random_seed=cfg.seed,
        num_workers=cfg.dataset.num_normalization_workers,
        use_tqdm=cfg.logging.tqdm,
    )

    val_seq_len = cfg.validation.extra_eval_steps
    if cfg.validation.full_rollouts is True:
        # only return/evaluate over entire rollouts
        val_seq_len = trainset.traj_length - cfg.model.input_seq_length
        cfg.validation.extra_eval_steps = int(val_seq_len)

    valset = MHDDataset(
        path=cfg.dataset.path,
        split="val",
        active_keys=cfg.dataset.active_keys,
        input_seq_length=cfg.model.input_seq_length,
        target_seq_length=val_seq_len,
        random_seed=cfg.seed,
        num_workers=cfg.dataset.num_normalization_workers,
        use_tqdm=cfg.logging.tqdm,
    )
    # testset = MHDDataset(
    #     path=cfg.dataset.path,
    #     split="test",
    #     active_keys=cfg.dataset.active_keys,
    #     input_seq_length=cfg.model.input_seq_length,
    #     target_seq_length=val_seq_len,
    #     random_seed=cfg.seed,
    #     num_workers=cfg.dataset.num_workers
    # )

    print(f"Train: {len(trainset)} samples")
    print(f"Val: {len(valset)} samples")
    # print(f"Test: {len(testset)} samples")

    # Normalize data
    if cfg.dataset.normalize:
        trainbounds = trainset.get_bounds()
        trainset.normalize(trainbounds)
        valset.normalize(trainbounds)
        # testset.normalize(trainbounds)
        print("the bounds are:")
        print(trainbounds)

    # IO efficient samplers
    trainsampler = (
        trainset.trajectory_sampler() if cfg.training.sampler == "trajectory" else None
    )
    valsampler = (
        valset.trajectory_sampler() if cfg.training.sampler == "trajectory" else None
    )

    augmentations = []
    if cfg.dataset.augment.noise is True:
        augmentations.append(
            noise_transform(
                std=cfg.dataset.augment.noise_std, window_size=cfg.model.input_seq_length
            )
        )

    trainloader = DataLoader(
        trainset,
        cfg.training.batch_size,
        shuffle=(trainsampler is None),
        sampler=trainsampler,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        collate_fn=partial(trainset.collate, (cfg.device, augmentations)),
    )
    valloader = DataLoader(
        valset,
        cfg.validation.batch_size,
        sampler=valsampler,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        collate_fn=partial(valset.collate, (cfg.device)),
    )

    return (trainset, valset), (trainloader, valloader), augmentations
