import torch


def get_model(cfg, dataset, train_method="default"):
    # TODO need to standardize modules everywhere (eg for different inputs)

    latent_dim = cfg.model.latent_dim
    problem_dim = len(dataset.active_keys)
    separate_zf = cfg.dataset.separate_zf
    if separate_zf:
        problem_dim = problem_dim + 2  # NOTE: re/im parts for zonal flow

    if cfg.model.name == "swin":
        assert cfg.dataset.input_fields == [
            "df"
        ], "No more inputs than df supported for simple 5D-Swin"
        from models.swin_unet import Swin5DUnet
        from models.utils import ContinuousConditionEmbed

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
        modulation = cfg.model.swin.modulation
        refiner = train_method == "refiner"
        swin_bottleneck = cfg.model.swin.swin_bottleneck
        decouple_mu = cfg.model.decouple_mu

        cond_fn = None
        conditioning = cfg.model.conditioning
        if refiner:
            conditioning += ["refinement_step"]
        n_cond = len(conditioning)
        if n_cond > 0:
            cond_fn = ContinuousConditionEmbed(128, n_cond)

        bundle_steps = cfg.model.bundle_seq_length
        if bundle_steps > 1:  # TODO investigate time dimension!
            space = space + 1
            # extend patching for time dimension
            patch_size = [1] + patch_size
            window_size = [bundle_steps] + window_size
            base_resolution = (bundle_steps,) + tuple(base_resolution)

        model = Swin5DUnet(
            dim=latent_dim,
            base_resolution=base_resolution,  # TODO
            patch_size=patch_size,
            window_size=window_size,
            depth=depth,
            num_heads=num_heads,
            in_channels=problem_dim * (2 if refiner else 1),
            out_channels=problem_dim,
            num_layers=num_layers,
            use_checkpoint=gradient_checkpoint,
            drop_path=cfg.model.swin.drop_path,
            abs_pe=abs_pe,
            conv_patch=False,
            hidden_mlp_ratio=2.0,
            c_multiplier=c_multiplier,
            merging_hidden_ratio=patching_hidden_ratio,
            unmerging_hidden_ratio=unmerging_hidden_ratio,
            conditioning=conditioning,
            cond_embed=cond_fn,
            norm_output=norm_output,
            act_fn=act_fn,
            patch_skip=patch_skip,
            modulation=modulation,
            swin_bottleneck=swin_bottleneck,
            separate_zf=separate_zf,
            decouple_mu=decouple_mu,
        )

    if "xnet" in cfg.model.name:
        if "multi" in cfg.model.name:
            from models.swin_xnet import SwinXNetMultitask as SwinXnet
        else:
            from models.swin_xnet import SwinXnet
        from models.utils import ContinuousConditionEmbed

        df_patch_size = cfg.model.swin.patch_size
        phi_patch_size = cfg.model.swin.phi_patch_size
        df_window_size = cfg.model.swin.window_size
        phi_window_size = cfg.model.swin.phi_window_size
        df_base_resolution = dataset.resolution
        phi_base_resolution = dataset.phi_resolution
        num_heads = cfg.model.swin.num_heads
        depth = cfg.model.swin.depth
        num_layers = cfg.model.num_layers
        gradient_checkpoint = cfg.model.swin.gradient_checkpoint
        patching_hidden_ratio = cfg.model.swin.merging_hidden_ratio
        unmerging_hidden_ratio = cfg.model.swin.unmerging_hidden_ratio
        c_multiplier = cfg.model.swin.c_multiplier
        abs_pe = cfg.model.swin.abs_pe
        patch_skip = cfg.model.swin.patch_skip
        modulation = cfg.model.swin.modulation
        act_fn = getattr(torch.nn, cfg.model.swin.act_fn)
        decouple_mu = cfg.model.decouple_mu
        refiner = train_method == "refiner"
        outputs = list(cfg.model.loss_weights.keys())
        swin_bottleneck = cfg.model.swin.swin_bottleneck

        cond_fn = None
        conditioning = cfg.model.conditioning
        if refiner:
            conditioning += ["refinement_step"]
        n_cond = len(conditioning)
        if n_cond > 0:
            cond_fn = ContinuousConditionEmbed(128, n_cond)

        if cfg.model.bundle_seq_length > 1:
            raise NotImplementedError

        model = SwinXnet(
            dim=latent_dim,
            outputs=outputs,
            df_base_resolution=df_base_resolution,
            phi_base_resolution=phi_base_resolution,
            df_patch_size=df_patch_size,
            phi_patch_size=phi_patch_size,
            df_window_size=df_window_size,
            phi_window_size=phi_window_size,
            depth=depth,
            num_heads=num_heads,
            in_channels=problem_dim,
            out_channels=problem_dim,
            num_layers=num_layers,
            use_checkpoint=gradient_checkpoint,
            drop_path=0.1,
            abs_pe=abs_pe,
            c_multiplier=c_multiplier,
            merging_hidden_ratio=patching_hidden_ratio,
            unmerging_hidden_ratio=unmerging_hidden_ratio,
            conditioning=conditioning,
            cond_embed=cond_fn,
            act_fn=act_fn,
            patch_skip=patch_skip,
            modulation=modulation,
            separate_zf=separate_zf,
            decouple_mu=decouple_mu,
            swin_bottleneck=swin_bottleneck,
        )

    if cfg.model.name == "swin_flat":
        from experimental.swin_flat import SwinFlat
        from models.utils import ContinuousConditionEmbed

        space = 5
        patch_size = cfg.model.swin.patch_size
        window_size = cfg.model.swin.window_size
        base_resolution = dataset.resolution
        num_heads = cfg.model.swin.num_heads
        depth = cfg.model.swin.depth
        gradient_checkpoint = cfg.model.swin.gradient_checkpoint
        abs_pe = cfg.model.swin.abs_pe
        act_fn = getattr(torch.nn, cfg.model.swin.act_fn)
        patch_skip = cfg.model.swin.patch_skip
        modulation = cfg.model.swin.modulation
        refiner = train_method == "refiner"

        cond_fn = None
        n_cond = cfg.model.swin.timestep_conditioning + cfg.model.swin.itg_conditioning
        n_cond = n_cond + (1 if refiner else 0)
        if n_cond > 0:
            cond_fn = ContinuousConditionEmbed(128, n_cond)

        bundle_steps = cfg.model.bundle_seq_length
        if bundle_steps > 1:  # TODO investigate time dimension!
            space = space + 1
            # extend patching for time dimension
            patch_size = [1] + patch_size
            window_size = [bundle_steps] + window_size
            base_resolution = (bundle_steps,) + tuple(base_resolution)

        model = SwinFlat(
            space=space,
            dim=latent_dim,
            base_resolution=base_resolution,
            patch_size=patch_size,
            window_size=window_size,
            depth=depth[0],
            num_heads=num_heads[0],
            in_channels=problem_dim * (2 if refiner else 1),
            out_channels=problem_dim,
            use_checkpoint=gradient_checkpoint,
            drop_path=cfg.model.swin.drop_path,
            abs_pe=abs_pe,
            conv_patch=False,
            hidden_mlp_ratio=2.0,
            conditioning=cond_fn,
            modulation=modulation,
            act_fn=act_fn,
            patch_skip=patch_skip,
        )

    if cfg.model.name == "ae":
        from models.swin_ae import SwinAE
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
        abs_pe = cfg.model.swin.abs_pe
        act_fn = getattr(torch.nn, cfg.model.swin.act_fn)
        modulation = cfg.model.swin.modulation

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
            drop_path=cfg.model.swin.drop_path,
            abs_pe=abs_pe,
            conv_patch=False,
            hidden_mlp_ratio=2.0,
            c_multiplier=c_multiplier,
            merging_hidden_ratio=patching_hidden_ratio,
            unmerging_hidden_ratio=unmerging_hidden_ratio,
            conditioning=cond_fn,
            act_fn=act_fn,
            modulation=modulation,
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
