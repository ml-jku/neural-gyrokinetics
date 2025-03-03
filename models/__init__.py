import torch


def get_model(cfg, dataset):
    # TODO need to standardize modules everywhere (eg for different inputs)

    latent_dim = cfg.model.latent_dim
    if not cfg.dataset.separate_zf:
        problem_dim = len(cfg.dataset.active_keys)
    else:
        problem_dim = 4
        problem_dim += (cfg.dataset.split_into_bands - 1) * 2 if cfg.dataset.split_into_bands else 0

    if cfg.model.name == "swin":
        from models.swin_unet import SwinUnet
        from models.utils import ContinuousConditionEmbed

        space = 5
        patch_size = cfg.model.swin.patch_size
        window_size = cfg.model.swin.window_size
        base_resolution = dataset.resolution
        num_heads = cfg.model.swin.num_heads
        depth = cfg.model.swin.depth
        num_layers = cfg.model.num_layers
        gradient_checkpoint = cfg.model.swin.gradient_checkpoint
        patching_hidden_ratio = cfg.model.swin.merging_hidden_ratio
        unmerging_hidden_ratio = cfg.model.swin.unmerging_hidden_ratio
        c_multiplier = cfg.model.swin.c_multiplier
        norm_output = cfg.model.swin.norm_output
        abs_pe = cfg.model.swin.abs_pe
        act_fn = getattr(torch.nn, cfg.model.swin.act_fn)
        patch_skip = cfg.model.swin.patch_skip

        cond_fn = None
        n_cond = cfg.model.swin.timestep_conditioning + cfg.model.swin.itg_conditioning
        if n_cond > 0:
            cond_fn = ContinuousConditionEmbed(32, n_cond)

        bundle_steps = cfg.model.bundle_seq_length
        if bundle_steps > 1:  # TODO investigate time dimension!
            space = space + 1
            # extend patching for time dimension
            patch_size = [1] + patch_size
            window_size = [bundle_steps] + window_size
            base_resolution = (bundle_steps,) + tuple(base_resolution)

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
            abs_pe=abs_pe,
            conv_patch=False,
            hidden_mlp_ratio=2.0,
            c_multiplier=c_multiplier,
            merging_hidden_ratio=patching_hidden_ratio,
            unmerging_hidden_ratio=unmerging_hidden_ratio,
            unmerging_layer_norm=cfg.model.swin.unmerging_layer_norm,
            conditioning=cond_fn,
            norm_output=norm_output,
            act_fn=act_fn,
            patch_skip=patch_skip,
        )

    if cfg.model.name == "ae":
        from models.swin_ae import SwinAE
        from models.utils import ContinuousConditionEmbed

        space = 5
        patch_size = cfg.model.swin.patch_size
        window_size = cfg.model.swin.window_size
        # TODO currently only support one resolution for all cyclones
        # TODO should move away from needing a fixed grid size
        base_resolution = dataset.resolution
        num_heads = cfg.model.swin.num_heads
        depth = cfg.model.swin.depth
        num_layers = cfg.model.num_layers
        gradient_checkpoint = cfg.model.swin.gradient_checkpoint
        patching_hidden_ratio = cfg.model.swin.patching_hidden_ratio

        cond_fn = None
        if cfg.model.swin.timestep_conditioning:
            cond_fn = ContinuousConditionEmbed(32)

        bundle_steps = cfg.model.bundle_seq_length
        if bundle_steps > 1:  # TODO investigate time dimension!
            space = space + 1
            # extend patching for time dimension
            patch_size = [1] + patch_size
            window_size = [bundle_steps] + window_size
            base_resolution = (bundle_steps,) + base_resolution

        model = SwinAE(
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
            abs_pe=False,
            conv_patch=False,
            hidden_mlp_ratio=6.0,
            patching_hidden_ratio=patching_hidden_ratio,
            conditioning=cond_fn,
        )

    if cfg.model.name == "perc":
        from models.perceiver import CompressionPerc

        base_resolution = dataset.resolution

        model = CompressionPerc(
            space=5,
            in_channels=2,
            out_channels=2,
            dim=latent_dim,
            patch_size=cfg.model.swin.patch_size,
            base_resolution=base_resolution,
            num_latent_tokens=420,
            encoder_depth=2,
            approximator_depth=8,
        )

    try:
        model
    except NameError:
        raise ValueError(f"Unknown model name: {cfg.model.name}")

    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    if cfg.logging.model_summary:
        import torchsummary

        try:
            torchsummary.summary(
                model, input_size=[(problem_dim, *base_resolution)], device="cpu"
            )
        except Exception as e:
            print("Could not print model summary, exception occurred:", e)

    return model
