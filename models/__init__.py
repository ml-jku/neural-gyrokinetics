def get_model(cfg):
    # TODO need to standardize modules everywhere (eg for different inputs)

    latent_dim = cfg.model.latent_dim
    problem_dim = len(cfg.dataset.active_keys)
    bundle_dim = cfg.training.bundle_seq_length
    window_size = cfg.model.input_seq_length

    if cfg.model.name == "unet":
        from models import unet
        model = unet.UNet(
            n_fields=problem_dim,
            input_timesteps=window_size,
            output_timesteps=bundle_dim,
            hidden_dim=latent_dim
        )

    if cfg.model.name == "fno":
        from models import fno

        modes1 = cfg.model.modes1
        modes2 = cfg.model.modes2

        model = fno.FNO2d(
            num_channels=problem_dim,
            initial_step=window_size,
            width=latent_dim,
            modes1=modes1,
            modes2=modes2,
            num_outs=bundle_dim,
        )

    if cfg.model.name == "swin":
        from models.swin import Swin

        model = Swin(problem_dim, 1, problem_dim, 64, patch_size=1).cuda()

    try:
        model
    except NameError:
        raise ValueError(f"Unknown model name: {cfg.model.name}")

    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    return model
