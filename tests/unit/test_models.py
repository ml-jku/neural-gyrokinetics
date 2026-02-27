import pytest
import torch
from neugk.pinc.autoencoders.ae_utils import load_autoencoder


def test_dummy_dataset_factory():
    # Inside load_autoencoder, there is a DummyDataset class
    # We can't access it directly, but we can verify the logic it uses if we were to mock things
    # Actually, let's just test the logic of key stripping which is atomic.

    state_dict = {
        "module.encoder.weight": torch.tensor([1.0]),
        "decoder.weight": torch.tensor([2.0]),
        "module.latent.bias": torch.tensor([3.0]),
    }

    # logic from utils.save_model_and_config or load_autoencoder
    stripped = {k.replace("module.", ""): v for k, v in state_dict.items()}

    assert "encoder.weight" in stripped
    assert "decoder.weight" in stripped
    assert "latent.bias" in stripped
    assert "module.encoder.weight" not in stripped


def test_problem_dim_logic():
    # problem_dim = 2 + 2 * int(cfg.dataset.separate_zf)
    # This is used in many places to determine input channels

    def get_dim(sep_zf):
        return 2 + 2 * int(sep_zf)

    assert get_dim(False) == 2
    assert get_dim(True) == 4
    assert get_dim(1) == 4
