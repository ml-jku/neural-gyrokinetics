def get_model(cfg, dataset):
    # TODO need to standardize modules everywhere (eg for different inputs)

    latent_dim = cfg.model.latent_dim
    problem_dim = len(cfg.dataset.active_keys)

    if cfg.model.name == "swin":
        from models.swin_unet import SwinUnet
        from models.utils import IntegerConditionEmbed

        space = 5
        patch_size = cfg.model.swin.patch_size
        window_size = cfg.model.swin.window_size
        base_resolution = dataset.resolution
        num_heads = cfg.model.swin.num_heads
        depth = cfg.model.swin.depth
        num_layers = cfg.model.num_layers
        gradient_checkpoint = cfg.model.swin.gradient_checkpoint
        patching_hidden_ratio = cfg.model.swin.patching_hidden_ratio

        cond_fn = None
        if cfg.model.swin.timestep_conditioning:
            cond_fn = IntegerConditionEmbed(32, 501)

        bundle_steps = cfg.model.bundle_seq_length
        if bundle_steps > 1:  # TODO investigate time dimension!
            space = space + 1
            # extend patching for time dimension
            patch_size = [1] + patch_size
            window_size = [bundle_steps] + window_size
            base_resolution = (bundle_steps,) + base_resolution

        model = SwinUnet(
            space=space,
            dim=latent_dim,
            base_resolution=base_resolution,  # TODO
            patch_size=patch_size,
            window_size=window_size,
            depth=depth,
            num_heads=num_heads,
            in_channels=problem_dim,
            out_channels=problem_dim,
            num_layers=num_layers,
            use_checkpoint=gradient_checkpoint,
            drop_path=0.1,
            abs_pe=True,
            conv_patch=False,
            hidden_mlp_ratio=6.0,
            patching_hidden_ratio=patching_hidden_ratio,
            middle_depth=8,
            conditioning=cond_fn,
        )

    try:
        model
    except NameError:
        raise ValueError(f"Unknown model name: {cfg.model.name}")

    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    return model
