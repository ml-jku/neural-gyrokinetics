def get_model(cfg):
    # TODO need to standardize modules everywhere (eg for different inputs)

    latent_dim = cfg.model.latent_dim
    problem_dim = len(cfg.dataset.active_keys)
    bundle_dim = cfg.training.bundle_seq_length
    input_seq_length = cfg.model.input_seq_length

    if cfg.model.name == "unet":
        from models import unet

        model = unet.UNet(
            n_fields=problem_dim,
            input_timesteps=input_seq_length,
            output_timesteps=bundle_dim,
            hidden_dim=latent_dim,
        )

    if cfg.model.name == "fno":
        from models import fno

        modes1 = cfg.model.modes1
        modes2 = cfg.model.modes2

        model = fno.FNO2d(
            num_channels=problem_dim,
            initial_step=input_seq_length,
            width=latent_dim,
            modes1=modes1,
            modes2=modes2,
            num_outs=bundle_dim,
        )

    if cfg.model.name == "swin":
        from models.swin_unet import SwinUnet
        
        # TODO?
        space = 5 if cfg.dataset.name == "cyclone" else 2
        patch_size = cfg.model.swin.patch_size
        window_size = cfg.model.swin.window_size
        downsample = cfg.model.swin.downsample
        num_heads = cfg.model.swin.num_heads
        depth = cfg.model.swin.depth
        num_layers = cfg.model.swin.num_layers
        gradient_checkpoint = cfg.model.swin.gradient_checkpoint
        
        model = SwinUnet(
            space=space,
            dim=latent_dim,
            in_channels=problem_dim, 
            out_channels=problem_dim,
            patch_size=patch_size,
            window_size=window_size,
            img_size=(32, 8, 16, 167, 21),  # TODO
            depth=depth,
            num_heads=num_heads,
            downsample=downsample,
            num_layers=num_layers,
            use_checkpoint=gradient_checkpoint,
            drop_path=0.1,
            abs_pe=False,
            conv_patch=False,
            hidden_mlp_ratio=6.0,
            patching_hidden_ratio=12.0,
            middle_depth=8,
        )

    try:
        model
    except NameError:
        raise ValueError(f"Unknown model name: {cfg.model.name}")

    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    return model
