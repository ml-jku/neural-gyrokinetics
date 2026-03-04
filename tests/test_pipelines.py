"""Integration tests for main pipelines with mocked components."""

import pytest
import torch
from unittest.mock import MagicMock, patch
from omegaconf import OmegaConf

from neugk.gyroswin import GyroSwinRunner
from neugk.pinc import PINCRunner
from neugk.dataset import CycloneSample


class MockDataset:
    def __init__(self, cfg, workflow="gyroswin"):
        self.cfg = cfg
        self.workflow = workflow
        # Resolution: (vpar, vmu, s, x, y)
        self.resolution = (32, 8, 16, 85, 32)
        self.phi_resolution = (16, 85, 32)
        self.decouple_mu = False
        self.vmu = 8
        self.active_keys = [0, 1]

        chan = 2
        self.df_shape = (chan, 32, 8, 16, 85, 32)
        if workflow == "gyroswin":
            self.phi_shape = (1, 85, 16, 32)  # (C, x, s, y) rearranged
        else:
            self.phi_shape = (2, 16, 85, 32)  # (C, s, x, y)

        self.geometry = {
            "krho": torch.randn(1, 32),
            "ints": torch.randn(1, 16),
            "intmu": torch.randn(1, 8),
            "intvp": torch.randn(1, 32),
            "kxrh": torch.randn(1, 85),
            "adiabatic": torch.tensor(1.0),
            "de": torch.tensor(1.0),
        }

    def denormalize(self, **kwargs):
        return kwargs

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        sample = CycloneSample(
            df=torch.randn(*self.df_shape),
            y_df=torch.randn(*self.df_shape),
            phi=torch.randn(*self.phi_shape),
            y_phi=torch.randn(*self.phi_shape),
            y_flux=torch.randn(1),
            timestep=torch.tensor([0.1]),
            file_index=torch.tensor(0),
            timestep_index=torch.tensor(idx),
            itg=torch.randn(1),
            dg=torch.randn(1),
            s_hat=torch.randn(1),
            q=torch.randn(1),
            geometry={k: v.clone() for k, v in self.geometry.items()},
        )
        n_cond = len(self.cfg.model.conditioning)
        sample.conditioning = torch.randn(n_cond)
        return sample


def get_mock_dataloader(dataset):
    def collate_fn(batch):
        elem = batch[0]
        collated = {}
        fields = [
            "df",
            "y_df",
            "phi",
            "y_phi",
            "y_flux",
            "timestep",
            "itg",
            "dg",
            "s_hat",
            "q",
            "conditioning",
        ]
        for k in fields:
            if hasattr(elem, k) and getattr(elem, k) is not None:
                collated[k] = torch.stack([getattr(b, k) for b in batch])

        mock_sample = MagicMock()
        for k, v in collated.items():
            setattr(mock_sample, k, v)
        mock_sample.file_index = torch.stack([b.file_index for b in batch])
        mock_sample.timestep_index = torch.stack([b.timestep_index for b in batch])
        mock_sample.geometry = elem.geometry
        return mock_sample

    return MagicMock(
        dataset=dataset,
        __iter__=lambda self: iter([collate_fn([dataset[0]])]),
        __len__=lambda self: 1,
    )


@pytest.fixture
def common_cfg():
    return {
        "seed": 42,
        "ddp": {"enable": False, "n_nodes": 1},
        "amp": {"enable": False, "bfloat": False},
        "logging": {
            "writer": None,
            "run_id": "test",
            "tqdm": False,
            "model_summary": False,
        },
        "training": {
            "learning_rate": 1e-4,
            "weight_decay": 1e-6,
            "exclude_from_wd": None,
            "scheduler": None,
            "n_epochs": 1,
            "batch_size": 1,
            "gradnorm_balancer": "none",
            "clip_grad": False,
            "pushforward": {"unrolls": [0], "probs": [0], "epochs": [0]},
            "predict_delta": False,
            "params_to_include": None,
        },
        "dataset": {
            "input_fields": ["df"],
            "separate_zf": False,
            "real_potens": True,
            "augment": {"mask_modes": {"active": False}},
            "num_workers": 0,
        },
        "validation": {
            "validate_every_n_epochs": 100,
            "n_eval_steps": 1,
            "eval_integrals": False,
        },
        "device": "cpu",
        "load_ckpt": False,
        "output_path": "/tmp/test_out",
        "ae_checkpoint": None,
    }


@patch("neugk.runner.get_data")
@patch("neugk.gyroswin.run.GyroSwinEvaluator")
def test_gyroswin_pipeline(mock_eval, mock_get_data, common_cfg):
    cfg = OmegaConf.create(common_cfg)
    cfg.workflow = "gyroswin"
    cfg.model = {
        "name": "gyroswin_multi",
        "bundle_seq_length": 1,
        "latent_dim": 32,
        "num_layers": 1,
        "decouple_mu": False,
        "conditioning": ["itg"],
        "extra_zf_loss": False,
        "loss_weights": {"df": 1.0, "phi": 0.1, "flux": 1.0},
        "extra_loss_weights": {},
        "swin": {
            "patch_size": [4, 4, 4, 4, 4],
            "phi_patch_size": [4, 4, 4],
            "window_size": [4, 4, 4, 4, 4],
            "phi_window_size": [4, 4, 4],
            "num_heads": [2],
            "depth": [1],
            "gradient_checkpoint": False,
            "merging_hidden_ratio": 1.0,
            "unmerging_hidden_ratio": 1.0,
            "c_multiplier": 1,
            "norm_output": False,
            "use_abs_pe": False,
            "act_fn": "GELU",
            "patch_skip": True,
            "modulation": "film",
            "drop_path": 0.0,
            "swin_bottleneck": True,
            "latent_cross_attn": True,
            "use_rpb": True,
            "use_rope": False,
            "detach_flux_latents": False,
            "flux_reduce": "max",
            "flux_num_heads": 2,
            "flux_depth": 1,
            "flux_conditioning": True,
            "init_weights": "kaiming_uniform",
            "patching_init_weights": "kaiming_uniform",
            "cond_init_weights": "normal_smallvar",
        },
    }

    ds = MockDataset(cfg, workflow="gyroswin")
    mock_get_data.return_value = (
        [ds, ds],
        [get_mock_dataloader(ds), get_mock_dataloader(ds)],
        [],
    )
    runner = GyroSwinRunner(0, cfg, 1)
    runner(skip_eval=True)


@patch("neugk.runner.get_data")
@patch("neugk.pinc.run.AutoencoderEvaluator")
def test_pinc_ae_pipeline(mock_eval, mock_get_data, common_cfg):
    cfg = OmegaConf.create(common_cfg)
    cfg.workflow = "pinc"
    cfg.stage = "autoencoder"
    cfg.model = {
        "name": "ae",
        "model_type": "ae",
        "bundle_seq_length": 1,
        "latent_dim": 32,
        "act_fn": "GELU",
        "decouple_mu": False,
        "conditioning": ["itg"],
        "extra_zf_loss": False,
        "init_weights": "kaiming_uniform",
        "patching_init_weights": "kaiming_uniform",
        "cond_init_weights": "kaiming_uniform",
        "loss_weights": {"df": 1.0},
        "extra_loss_weights": {},
        "patch": {
            "patch_size": [4, 4, 4, 4, 4],
            "window_size": [4, 4, 4, 4, 4],
            "merging_depth": 1,
            "unmerging_depth": 1,
            "merging_hidden_ratio": 1.0,
            "unmerging_hidden_ratio": 1.0,
            "c_multiplier": 1.0,
        },
        "vit": {
            "num_heads": [2],
            "depth": [1],
            "gradient_checkpoint": False,
            "use_abs_pe": False,
            "use_rpb": True,
            "use_rope": False,
            "gated_attention": True,
            "modulation": "dit",
            "drop_path": 0.0,
            "qk_norm": True,
        },
        "bottleneck": {
            "dim": 32,
            "norm_learnable": False,
            "normalized_latent": True,
            "num_heads": 2,
            "depth": 1,
        },
    }

    ds = MockDataset(cfg, workflow="pinc")
    mock_get_data.return_value = (
        [ds, ds],
        [get_mock_dataloader(ds), get_mock_dataloader(ds)],
        [],
    )
    runner = PINCRunner(0, cfg, 1)
    runner(skip_eval=True)
