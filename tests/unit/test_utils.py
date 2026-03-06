import pytest
import torch
import numpy as np

from neugk.utils import (
    is_number,
    edit_tag,
    remainig_progress,
    recombine_zf,
    separate_zf,
    RunningMeanStd,
    get_linear_burn_in_fn,
    expand_as,
)


def test_is_number():
    assert is_number("123") == True
    assert is_number("123.45") == True
    assert is_number("-123") == True
    assert is_number("+123.45e-2") == True
    assert is_number("abc") == False
    assert is_number("123a") == False
    assert is_number("") == False


def test_edit_tag():
    test_dict = {"loss": 0.1, "acc": 0.9}
    result = edit_tag(test_dict, "train", "epoch")
    assert result == {"train/loss_epoch": 0.1, "train/acc_epoch": 0.9}

    result = edit_tag(test_dict, prefix="val")
    assert result == {"val/loss": 0.1, "val/acc": 0.9}

    tagged_dict = {"train/loss_epoch": 0.1}
    result = edit_tag(tagged_dict, prefix="train", postfix="epoch")
    assert result == {"train/loss_epoch": 0.1}


def test_remainig_progress():
    assert remainig_progress(0, 100) == 1.0
    assert remainig_progress(50, 100) == 0.5
    assert remainig_progress(100, 100) == 0.0


@pytest.mark.parametrize("dim", [0, 1])
def test_recombine_zf_torch(dim):
    B, H, W = 1, 10, 10
    zf, non_zf = torch.randn(B, 2, H, W), torch.randn(B, 2, H, W)

    if dim == 0:  # C first
        input_tensor = torch.cat([zf[0], non_zf[0]], dim=0)
        expected = zf[0] + non_zf[0]
    elif dim == 1:  # B first
        input_tensor = torch.cat([zf, non_zf], dim=dim)
        expected = zf + non_zf

    output = recombine_zf(input_tensor, dim=dim)
    assert output.shape[dim] == 2
    assert torch.allclose(output, expected)


@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("data_type", ["torch", "numpy"])
def test_separate_zf(dim, data_type):
    H, W = 5, 5
    if data_type == "torch":
        B, C = 1, 2
        input_tensor = torch.randn(B, C, H, W) if dim == 1 else torch.randn(C, B, H, W)
        output = separate_zf(input_tensor, dim=dim)

        expected_shape = list(input_tensor.shape)
        expected_shape[dim] *= 2
        assert output.shape == tuple(expected_shape)

        zf_avg = input_tensor.mean(dim=-1, keepdim=True)
        zf = torch.repeat_interleave(zf_avg, repeats=W, dim=-1)
        expected = torch.cat([zf, input_tensor - zf], dim=dim)
        assert torch.allclose(output, expected)
    else:
        B, C = 1, 2
        input_array = (
            np.random.randn(B, C, H, W) if dim == 1 else np.random.randn(C, B, H, W)
        )
        output = separate_zf(input_array, dim=dim)

        expected_shape = list(input_array.shape)
        expected_shape[dim] *= 2
        assert output.shape == tuple(expected_shape)

        zf_avg = input_array.mean(axis=-1, keepdims=True)
        zf = np.repeat(zf_avg, repeats=W, axis=-1)
        expected = np.concatenate([zf, input_array - zf], axis=dim)
        assert np.allclose(output, expected)


def test_running_mean_std():
    shape = (2, 2)
    rms = RunningMeanStd(shape=shape)

    data1 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    mean1, var1 = np.mean(data1, axis=0), np.var(data1, axis=0)
    rms.update(mean1, var1, np.min(data1, axis=0), np.max(data1, axis=0), count=2)

    assert np.allclose(rms.mean, mean1, atol=5e-3)
    assert rms.count == pytest.approx(2 + 1e-4)

    data2 = np.array([[[9, 10], [11, 12]]], dtype=np.float32)
    rms.update(
        np.mean(data2, axis=0),
        np.var(data2, axis=0),
        np.min(data2, axis=0),
        np.max(data2, axis=0),
        count=1,
    )

    data_all = np.concatenate([data1, data2], axis=0)
    assert np.allclose(rms.mean, np.mean(data_all, axis=0), atol=5e-3)


def test_get_linear_burn_in_fn():
    fn = get_linear_burn_in_fn(1.0, 0.0, 0.8, 0.2)
    assert fn(0.9) == 1.0
    assert fn(0.1) == 0.0
    assert fn(0.5) == pytest.approx(0.5)


def test_expand_as():
    src, tgt = np.zeros((4,)), np.zeros((12, 2, 4))
    assert expand_as(src, tgt).shape == (1, 1, 4)

    src, tgt = np.zeros((1, 2)), torch.zeros((2, 3, 4))
    assert expand_as(src, tgt).shape == (1, 1, 2)

    # TODO wrong behavior, should match and append
    src, tgt = np.zeros((2, 1)), np.zeros((2, 3, 4))
    assert expand_as(src, tgt).shape == (1, 2, 1)


def test_config_filters():
    from omegaconf import OmegaConf
    from neugk.utils import filter_config_subset, filter_cli_priority

    superset = OmegaConf.create({"a": 1, "b": {"c": 2, "d": 3}})
    subset = OmegaConf.create({"a": 10, "b": {"c": 20, "e": 40}, "f": 50})
    filter_config_subset(superset, subset)
    assert "f" not in subset
    assert "e" not in subset["b"]

    cli = ["model.lr=0.01", "dataset.path=/tmp"]
    source = OmegaConf.create(
        {"model": {"lr": 0.001, "layers": 5}, "dataset": {"path": "/data"}}
    )
    filter_cli_priority(cli, source)
    assert "model" not in source
    assert "dataset" not in source


def test_do_ifft_roundtrip():
    from neugk.dataset.preprocess import do_ifft

    shape = (2, 2, 2, 4, 4)
    data = np.random.randn(*shape) + 1j * np.random.randn(*shape)
    data = data.astype(np.complex64)
    transformed = do_ifft(data)
    assert transformed.shape == (2,) + shape
    assert transformed.dtype == np.float32


def test_phi_transform_shapes():
    from neugk.dataset.preprocess import phi_fft_to_real

    fft_shape = (8, 4, 8)
    data = np.random.randn(*fft_shape) + 1j * np.random.randn(*fft_shape)
    data = data.astype(np.complex64)
    out_shape = (8, 4, 8)
    real_phi = phi_fft_to_real(data, out_shape)
    assert real_phi.shape == (8, 4, 8)
