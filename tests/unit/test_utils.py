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


def test_remainig_progress():
    assert remainig_progress(0, 100) == 1.0
    assert remainig_progress(50, 100) == 0.5
    assert remainig_progress(100, 100) == 0.0
    assert remainig_progress(25, 50) == 0.5


@pytest.mark.parametrize("dim", [0, 1])
def test_recombine_zf_torch(dim):
    # Test cases for torch tensors
    B, H, W = 1, 10, 10
    zf = torch.randn(B, 2, H, W)
    non_zf = torch.randn(B, 2, H, W)

    if dim == 0:  # C is first dim
        input_tensor = torch.cat([zf[0], non_zf[0]], dim=0)
        expected_output = zf[0] + non_zf[0]
    elif dim == 1:  # C is second dim (B, C, H, W)
        input_tensor = torch.cat([zf, non_zf], dim=dim)
        expected_output = zf + non_zf

    output = recombine_zf(input_tensor, dim=dim)

    assert output.shape[dim] == 2
    assert torch.allclose(output, expected_output)

    # Test case with odd number of channels
    x_odd_channels = torch.randn(B, 3, H, W) if dim == 1 else torch.randn(3, B, H, W)
    output_odd = recombine_zf(x_odd_channels, dim=dim)
    assert torch.allclose(output_odd, x_odd_channels)  # Should return unchanged

    # Test case with 2 channels
    x_2_channels = torch.randn(B, 2, H, W) if dim == 1 else torch.randn(2, B, H, W)
    output_2_channels = recombine_zf(x_2_channels, dim=dim)
    assert torch.allclose(output_2_channels, x_2_channels)  # Should return unchanged


@pytest.mark.parametrize("dim", [0, 1])
def test_recombine_zf_numpy(dim):
    # Test cases for numpy arrays
    B, H, W = 1, 10, 10
    zf = np.random.randn(B, 2, H, W)
    non_zf = np.random.randn(B, 2, H, W)

    if dim == 0:  # C is first dim
        input_array = np.concatenate([zf[0], non_zf[0]], axis=dim)
        expected_output = zf[0] + non_zf[0]
    elif dim == 1:  # C is second dim (B, C, H, W)
        input_array = np.concatenate([zf, non_zf], axis=dim)
        expected_output = zf + non_zf

    output = recombine_zf(input_array, dim=dim)

    assert output.shape[dim] == 2
    assert np.allclose(output, expected_output)

    # Test case with odd number of channels
    x_odd_channels = (
        np.random.randn(B, 3, H, W) if dim == 1 else np.random.randn(3, B, H, W)
    )
    output_odd = recombine_zf(x_odd_channels, dim=dim)
    assert np.allclose(output_odd, x_odd_channels)

    # Test case with 2 channels
    x_2_channels = (
        np.random.randn(B, 2, H, W) if dim == 1 else np.random.randn(2, B, H, W)
    )
    output_2_channels = recombine_zf(x_2_channels, dim=dim)
    assert np.allclose(output_2_channels, x_2_channels)


@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("data_type", ["torch", "numpy"])
def test_separate_zf(dim, data_type):
    H, W = 5, 5
    if data_type == "torch":
        # Example input: (B, C, H, W)
        B, C = 1, 2
        input_tensor = torch.randn(B, C, H, W) if dim == 1 else torch.randn(C, B, H, W)

        output = separate_zf(input_tensor, dim=dim)

        expected_shape = list(input_tensor.shape)
        expected_shape[dim] *= 2
        assert output.shape == tuple(expected_shape)

        zf_avg = input_tensor.mean(dim=-1, keepdim=True)  # mean over last dim (W)
        zf = torch.repeat_interleave(zf_avg, repeats=W, dim=-1)
        expected = torch.cat([zf, input_tensor - zf], dim=dim)

        assert torch.allclose(output, expected)

    else:  # numpy
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

    # Test batch 1
    data1 = np.array(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32
    )  # (2, 2, 2)
    mean1 = np.mean(data1, axis=0)
    var1 = np.var(data1, axis=0)
    rms.update(mean1, var1, np.min(data1, axis=0), np.max(data1, axis=0), count=2)

    # Use higher atol due to epsilon initialization influence
    assert np.allclose(rms.mean, mean1, atol=5e-3)
    assert np.allclose(rms.var, var1, atol=5e-3)
    assert rms.count == pytest.approx(2 + 1e-4)

    # Test batch 2
    data2 = np.array([[[9, 10], [11, 12]]], dtype=np.float32)  # (1, 2, 2)
    mean2 = np.mean(data2, axis=0)
    var2 = np.var(data2, axis=0)
    rms.update(mean2, var2, np.min(data2, axis=0), np.max(data2, axis=0), count=1)

    # Combined expected
    data_all = np.concatenate([data1, data2], axis=0)
    assert np.allclose(rms.mean, np.mean(data_all, axis=0), atol=5e-3)
    assert np.allclose(rms.var, np.var(data_all, axis=0), atol=5e-3)
    assert np.allclose(rms.min, np.min(data_all, axis=0))
    assert np.allclose(rms.max, np.max(data_all, axis=0))


def test_rms_combine():
    shape = (1,)
    rms1 = RunningMeanStd(shape=shape)
    rms2 = RunningMeanStd(shape=shape)

    d1 = np.array([1, 2, 3], dtype=np.float32).reshape(-1, 1)
    d2 = np.array([4, 5, 6, 7], dtype=np.float32).reshape(-1, 1)

    rms1.update(
        np.mean(d1, 0), np.var(d1, 0), np.min(d1, 0), np.max(d1, 0), count=len(d1)
    )
    rms2.update(
        np.mean(d2, 0), np.var(d2, 0), np.min(d2, 0), np.max(d2, 0), count=len(d2)
    )

    rms1.combine(rms2)

    d_all = np.concatenate([d1, d2])
    assert np.allclose(rms1.mean, np.mean(d_all, 0), atol=5e-3)
    assert np.allclose(rms1.var, np.var(d_all, 0), atol=5e-3)
    assert rms1.count == pytest.approx(len(d_all) + 2e-4)


def test_get_linear_burn_in_fn():
    # start=1.0, end=0.0, end_frac=0.8, start_frac=0.2
    fn = get_linear_burn_in_fn(1.0, 0.0, 0.8, 0.2)

    # progress_remaining is (1 - cur_step/total_steps)
    # 1.0 means cur_step=0 (start of training)
    # 0.0 means cur_step=total_steps (end of training)

    # case 1: before start_frac (1-prog < 0.2 => prog > 0.8)
    assert fn(0.9) == 1.0
    # case 2: after end_frac (1-prog > 0.8 => prog < 0.2)
    assert fn(0.1) == 0.0
    # case 3: midpoint (1-prog = 0.5 => prog = 0.5)
    # start + (0.5 - 0.2) * (0.0 - 1.0) / (0.8 - 0.2) = 1.0 + 0.3 * (-1.0) / 0.6 = 1.0 - 0.5 = 0.5
    assert fn(0.5) == pytest.approx(0.5)


def test_expand_as():
    from neugk.utils import expand_as

    src = np.array([1, 2])
    tgt = np.zeros((2, 3, 4))
    # expand_as will squeeze first, then unsqueeze until dims match
    res = expand_as(src, tgt)
    assert res.shape == (2, 1, 1)

    src_torch = torch.tensor([1, 2])
    tgt_torch = torch.zeros((2, 3, 4))
    res_torch = expand_as(src_torch, tgt_torch)
    assert res_torch.shape == (2, 1, 1)


def test_config_filters():
    from omegaconf import OmegaConf
    from neugk.utils import filter_config_subset, filter_cli_priority

    # filter_config_subset
    superset = OmegaConf.create({"a": 1, "b": {"c": 2, "d": 3}})
    subset = OmegaConf.create({"a": 10, "b": {"c": 20, "e": 40}, "f": 50})
    filter_config_subset(superset, subset)
    assert "f" not in subset
    assert "e" not in subset["b"]
    assert subset["b"]["c"] == 20

    # filter_cli_priority
    cli = ["model.lr=0.01", "dataset.path=/tmp"]
    source = OmegaConf.create(
        {"model": {"lr": 0.001, "layers": 5}, "dataset": {"path": "/data"}}
    )
    filter_cli_priority(cli, source)
    # The current implementation of filter_cli_priority deletes the top-level key
    # if any subkey is present in CLI because of 'subkey in c.split("=")[0]' check
    assert "model" not in source
    assert "dataset" not in source
