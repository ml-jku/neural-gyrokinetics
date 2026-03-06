from typing import Optional

import torch


def get_autoencoder(cfg, dataset, rank: Optional[int] = 0):
    model_key = "autoencoder" if hasattr(cfg, "autoencoder") else "model"
    ae_cfg = getattr(cfg, model_key)

    latent_dim = ae_cfg.latent_dim
    problem_dim = len(dataset.active_keys)
    separate_zf = cfg.dataset.separate_zf
    if separate_zf:
        problem_dim = problem_dim + 2  # NOTE: re/im parts for zonal flow

    # Get model type (ae, vae, vqvae)
    model_type = getattr(ae_cfg, "model_type", "ae")

    if ae_cfg.name in ["ae", "vae", "vqvae", "simsiam"]:
        # Import appropriate model class
        if model_type == "ae":
            from neugk.pinc.autoencoders.gk_autoencoders import Swin5DAE as AE
        elif model_type == "vae":
            from neugk.pinc.autoencoders.gk_autoencoders import Swin5DVAE as AE
        elif model_type == "vqvae":
            from neugk.pinc.autoencoders.gk_autoencoders import Swin5DVQVAE as AE
        elif model_type == "simsiam":
            from neugk.pinc.autoencoders.gk_autoencoders import Swin5DSimSiam as AE
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        from neugk.models.layers import ContinuousConditionEmbed

        bottleneck_dim = ae_cfg.bottleneck.dim
        bottleneck_num_heads = getattr(ae_cfg.bottleneck, "num_heads", None)
        bottleneck_depth = getattr(ae_cfg.bottleneck, "depth", None)
        normalized_latent = getattr(ae_cfg.bottleneck, "normalized_latent", True)
        norm_learnable = getattr(ae_cfg.bottleneck, "norm_learnable", False)

        base_resolution = dataset.resolution
        decouple_mu = ae_cfg.decouple_mu
        patch_size = ae_cfg.patch.patch_size
        window_size = ae_cfg.patch.window_size
        merging_depth = ae_cfg.patch.merging_depth
        unmerging_depth = ae_cfg.patch.unmerging_depth
        patching_hidden_ratio = ae_cfg.patch.merging_hidden_ratio
        unmerging_hidden_ratio = ae_cfg.patch.unmerging_hidden_ratio
        c_multiplier = ae_cfg.patch.c_multiplier
        act_fn = getattr(torch.nn, ae_cfg.act_fn)
        norm_fn = getattr(torch.nn, getattr(ae_cfg, "norm_fn", "LayerNorm"))

        num_heads = ae_cfg.vit.num_heads
        depth = ae_cfg.vit.depth
        use_rpb = getattr(ae_cfg.vit, "use_rpb", None)
        use_rope = getattr(ae_cfg.vit, "use_rope", None)
        gated_attention = getattr(ae_cfg.vit, "gated_attention", None)
        gradient_checkpoint = ae_cfg.vit.gradient_checkpoint
        use_abs_pe = ae_cfg.vit.use_abs_pe
        modulation = ae_cfg.vit.modulation
        drop_path = ae_cfg.vit.drop_path
        qk_norm = getattr(ae_cfg.vit, "qk_norm", True)
        num_layers = len(depth)
        assert num_layers == len(num_heads)

        cond_fn = None
        conditioning = getattr(ae_cfg, "conditioning", [])
        n_cond = len(conditioning)
        if n_cond > 0:
            cond_fn = ContinuousConditionEmbed(32, n_cond)

        # VAE/VQ-VAE configs
        model_kwargs = {}
        if model_type == "vae":
            model_kwargs["beta_vae"] = getattr(ae_cfg, "beta_vae", 1.0)
        elif model_type == "vqvae":
            vq_config = {}
            if hasattr(ae_cfg, "vq"):
                vq_config = {
                    "codebook_size": getattr(ae_cfg.vq, "codebook_size", 8192),
                    "embedding_dim": getattr(ae_cfg.vq, "embedding_dim", 256),
                    "commitment_weight": getattr(ae_cfg.vq, "commitment_weight", 0.25),
                    "codebook_type": getattr(ae_cfg.vq, "codebook_type", "euclidean"),
                    "ema_decay": getattr(ae_cfg.vq, "ema_decay", 0.99),
                    "threshold_ema_dead_code": getattr(
                        ae_cfg.vq, "threshold_ema_dead_code", 2
                    ),
                }
            else:
                vq_config = {
                    "codebook_size": 8192,
                    "embedding_dim": 256,
                    "commitment_weight": 0.25,
                    "codebook_type": "euclidean",
                    "ema_decay": 0.99,
                    "threshold_ema_dead_code": 2,
                }
            model_kwargs["vq_config"] = vq_config
        elif model_type == "simsiam":
            model_kwargs["use_simae_decoder"] = ae_cfg.bottleneck.use_simae_decoder
            model_kwargs["vit_predictor"] = ae_cfg.bottleneck.vit_predictor

        ae = AE(
            dim=latent_dim,
            bottleneck_dim=bottleneck_dim,
            base_resolution=base_resolution,
            patch_size=patch_size,
            window_size=window_size,
            depth=depth,
            num_heads=num_heads,
            bottleneck_num_heads=bottleneck_num_heads,
            bottleneck_depth=bottleneck_depth,
            in_channels=problem_dim,
            out_channels=problem_dim,
            num_layers=num_layers,
            use_checkpoint=gradient_checkpoint,
            drop_path=drop_path,
            use_abs_pe=use_abs_pe,
            conv_patch=False,
            hidden_mlp_ratio=2.0,
            c_multiplier=c_multiplier,
            merging_depth=merging_depth,
            unmerging_depth=unmerging_depth,
            merging_hidden_ratio=patching_hidden_ratio,
            unmerging_hidden_ratio=unmerging_hidden_ratio,
            cond_embed=cond_fn,
            init_weights=ae_cfg.init_weights,
            patching_init_weights=ae_cfg.patching_init_weights,
            act_fn=act_fn,
            use_rpb=use_rpb,
            use_rope=use_rope,
            gated_attention=gated_attention,
            qk_norm=qk_norm,
            norm_layer=norm_fn,
            modulation=modulation,
            decouple_mu=decouple_mu,  # make it 4D
            conditioning=True,
            normalized_latent=normalized_latent,
            mid_norm_learnable=norm_learnable,
            **model_kwargs,  # VAE/VQ-VAE configs
        )

    try:
        ae
    except NameError:
        raise ValueError(f"Unknown autoencoder name: {ae_cfg.name}")

    if rank == 0 or rank is None:
        params_m = sum(p.numel() for p in ae.parameters()) / 1e6
        print(f"AE parameters: {params_m:.1f}M")
        if hasattr(ae, "get_compression_info"):
            c_info = ae.get_compression_info()
            print(f"Compression: {c_info['rate']:.1f}x (type: {c_info['type']})")

    return ae
