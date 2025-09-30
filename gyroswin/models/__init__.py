import torch


def get_model(cfg, dataset):
    latent_dim = cfg.model.latent_dim
    problem_dim = len(dataset.active_keys)
    separate_zf = cfg.dataset.separate_zf
    if separate_zf:
        problem_dim = problem_dim + 2  # NOTE: re/im parts for zonal flow

    if cfg.model.name == "unet":
        assert cfg.dataset.input_fields == [
            "df"
        ], "No more inputs than df supported for simple 5D-Swin"
        from gyroswin.models.gk_unet import Swin5DUnet
        from gyroswin.models.layers import ContinuousConditionEmbed

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
        use_abs_pe = cfg.model.swin.use_abs_pe
        act_fn = getattr(torch.nn, cfg.model.swin.act_fn)
        patch_skip = cfg.model.swin.patch_skip
        modulation = cfg.model.swin.modulation
        swin_bottleneck = cfg.model.swin.swin_bottleneck
        use_rpb = cfg.model.swin.use_rpb
        use_rope = cfg.model.swin.use_rope
        decouple_mu = cfg.model.decouple_mu

        cond_fn = None
        conditioning = cfg.model.conditioning
        n_cond = len(conditioning)
        if n_cond > 0:
            cond_fn = ContinuousConditionEmbed(128, n_cond)

        bundle_steps = cfg.model.bundle_seq_length
        if bundle_steps > 1:
            # extend patching for time dimension
            patch_size = [1] + patch_size
            window_size = [bundle_steps] + window_size
            base_resolution = (bundle_steps,) + tuple(base_resolution)

        model = Swin5DUnet(
            dim=latent_dim,
            base_resolution=base_resolution,
            patch_size=patch_size,
            window_size=window_size,
            depth=depth,
            num_heads=num_heads,
            in_channels=problem_dim,
            out_channels=problem_dim,
            num_layers=num_layers,
            use_checkpoint=gradient_checkpoint,
            drop_path=cfg.model.swin.drop_path,
            use_abs_pe=use_abs_pe,
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
            decouple_mu=decouple_mu,
            use_rpb=use_rpb,
            use_rope=use_rope,
        )

    if "gyroswin" in cfg.model.name:
        if "multi" in cfg.model.name:
            from gyroswin.models.gk_multi import GyroSwinMultitask as GyroSwin
        else:
            from gyroswin.models.gk_multi import GyroSwin
        from gyroswin.models.layers import ContinuousConditionEmbed

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
        use_abs_pe = cfg.model.swin.use_abs_pe
        patch_skip = cfg.model.swin.patch_skip
        modulation = cfg.model.swin.modulation
        act_fn = getattr(torch.nn, cfg.model.swin.act_fn)
        decouple_mu = cfg.model.decouple_mu
        outputs = [
            k
            for k in cfg.model.loss_weights.keys()
            if cfg.model.loss_weights[k] > 0.0 or cfg.model.loss_scheduler[k]
        ]
        assert (
            len([k for k in outputs if k.startswith("flux")]) == 1
        ), "Cannot have multiple flux targets!"
        swin_bottleneck = cfg.model.swin.swin_bottleneck
        use_rpb = cfg.model.swin.use_rpb
        use_rope = cfg.model.swin.use_rope
        latent_cross_attn = cfg.model.swin.latent_cross_attn
        flux_reduce = cfg.model.swin.flux_reduce
        flux_num_heads = cfg.model.swin.flux_num_heads
        flux_depth = cfg.model.swin.flux_depth

        cond_fn = None
        conditioning = cfg.model.conditioning
        n_cond = len(conditioning)
        if n_cond > 0:
            cond_fn = ContinuousConditionEmbed(128, n_cond)
            if cfg.model.swin.flux_conditioning:
                flux_cond_fn = ContinuousConditionEmbed(128, n_cond)
            else:
                flux_cond_fn = None

        if cfg.model.bundle_seq_length > 1:
            raise NotImplementedError

        model = GyroSwin(
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
            use_abs_pe=use_abs_pe,
            c_multiplier=c_multiplier,
            merging_hidden_ratio=patching_hidden_ratio,
            unmerging_hidden_ratio=unmerging_hidden_ratio,
            conditioning=conditioning,
            cond_embed=cond_fn,
            act_fn=act_fn,
            patch_skip=patch_skip,
            modulation=modulation,
            decouple_mu=decouple_mu,
            swin_bottleneck=swin_bottleneck,
            use_rpb=use_rpb,
            use_rope=use_rope,
            latent_cross_attn=latent_cross_attn,
            separate_zf=separate_zf,
            detach_flux_latents=cfg.model.swin.detach_flux_latents,
            real_potens=cfg.dataset.real_potens,
            flux_reduce=flux_reduce,
            flux_num_heads=flux_num_heads,
            flux_depth=flux_depth,
            flux_cond_embed=flux_cond_fn,
        )

    if cfg.model.name == "swin_flat":
        from gyroswin.models.swin_flat import SwinFlat
        from gyroswin.models.layers import ContinuousConditionEmbed

        space = 5
        patch_size = cfg.model.swin.patch_size
        window_size = cfg.model.swin.window_size
        base_resolution = dataset.resolution
        num_heads = cfg.model.swin.num_heads
        depth = cfg.model.swin.depth
        gradient_checkpoint = cfg.model.swin.gradient_checkpoint
        use_abs_pe = cfg.model.swin.use_abs_pe
        act_fn = getattr(torch.nn, cfg.model.swin.act_fn)
        patch_skip = cfg.model.swin.patch_skip
        modulation = cfg.model.swin.modulation

        cond_fn = None
        n_cond = cfg.model.swin.timestep_conditioning + cfg.model.swin.itg_conditioning
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
            in_channels=problem_dim,
            out_channels=problem_dim,
            use_checkpoint=gradient_checkpoint,
            drop_path=cfg.model.swin.drop_path,
            use_abs_pe=use_abs_pe,
            conv_patch=False,
            hidden_mlp_ratio=2.0,
            conditioning=cond_fn,
            modulation=modulation,
            act_fn=act_fn,
            patch_skip=patch_skip,
        )

    if cfg.model.name == "vit_flat":
        from gyroswin.models.vit_flat import ViTFlat
        from gyroswin.models.layers import ContinuousConditionEmbed

        space = 5
        patch_size = cfg.model.vit.patch_size
        window_size = cfg.model.vit.window_size
        base_resolution = dataset.resolution
        num_heads = cfg.model.vit.num_heads
        depth = cfg.model.vit.depth
        gradient_checkpoint = cfg.model.vit.gradient_checkpoint
        use_abs_pe = cfg.model.vit.use_abs_pe
        act_fn = getattr(torch.nn, cfg.model.vit.act_fn)
        patch_skip = cfg.model.vit.patch_skip
        modulation = cfg.model.vit.modulation

        cond_fn = None
        conditioning = cfg.model.conditioning
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

        model = ViTFlat(
            space=space,
            dim=latent_dim,
            base_resolution=base_resolution,
            patch_size=patch_size,
            depth=depth[0],
            num_heads=num_heads[0],
            in_channels=problem_dim,
            out_channels=problem_dim,
            use_checkpoint=gradient_checkpoint,
            drop_path=cfg.model.vit.drop_path,
            conv_patch=False,
            hidden_mlp_ratio=2.0,
            conditioning=cfg.model.conditioning,
            cond_embed=cond_fn,
            modulation=modulation,
            act_fn=act_fn,
            patch_skip=patch_skip,
            abs_pe=use_abs_pe,
        )

    if "fno" in cfg.model.name:
        from gyroswin.models.fno import Df5DTFNO, DfVSpace3DTFNO, DfLocal5DTFNO

        base_resolution = dataset.resolution
        num_layers = cfg.model.num_layers

        if cfg.model.name == "fno":
            model = Df5DTFNO(
                latent_dim,
                base_resolution=base_resolution,
                in_channels=problem_dim,
                out_channels=problem_dim,
                num_layers=num_layers,
            )
        if cfg.model.name == "fno3d":
            model = DfVSpace3DTFNO(
                latent_dim,
                base_resolution=base_resolution,
                in_channels=problem_dim,
                out_channels=problem_dim,
                num_layers=num_layers,
            )
        if cfg.model.name == "local_fno":
            patch_size = cfg.model.swin.patch_size
            model = DfLocal5DTFNO(
                latent_dim,
                base_resolution=base_resolution,
                patch_size=patch_size,
                in_channels=problem_dim,
                out_channels=problem_dim,
                num_layers=num_layers,
            )

    if cfg.model.name == "pointnet":
        from gyroswin.models.pointnet import PointNet

        model = PointNet(
            dim=cfg.model.latent_dim,
            n_dims=5,
            n_channels=2 if not cfg.dataset.separate_zf else 4,
            condition_keys=cfg.model.conditioning,
        )

    if cfg.model.name == "transformer":
        from gyroswin.models.transformer import Transformer

        model = Transformer(
            condition_keys=cfg.model.conditioning,
            output_channels=2,
            space=5,
            dim=cfg.model.latent_dim,
        )

    if cfg.model.name == "transolver":
        from gyroswin.models.transolver import Transolver

        model = Transolver(
            condition_keys=cfg.model.conditioning,
            output_channels=2,
            space=5,
            dim=cfg.model.latent_dim,
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
