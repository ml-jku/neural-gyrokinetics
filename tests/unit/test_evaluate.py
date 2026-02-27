import pytest
import torch
import numpy as np

from neugk.evaluate import ComplexMetrics


@pytest.fixture
def complex_metrics_instance():
    return ComplexMetrics()


@pytest.mark.parametrize(
    "input_tensor_data, expected_batch_size, expected_channels, H, W",
    [
        # (B, 2, H, W) -> (B, H, W) complex
        ([[1.0, 2.0], [3.0, 4.0]], 2, 2, 1, 1),
        # (B, 4, H, W) -> (B, H, W) complex (summing re/im pairs)
        ([[1.0, 2.0, 5.0, 6.0], [3.0, 4.0, 7.0, 8.0]], 2, 4, 1, 1),
        # (B, 2, H, W) with spatial dimensions (H=2, W=2)
        (
            [
                [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
                [[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]],
            ],  # B=2, C=2, H=2, W=2
            2,
            2,
            2,
            2,
        ),
        # (B, 4, H, W) with spatial dimensions (H=2, W=2)
        (
            [
                [
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[5.0, 6.0], [7.0, 8.0]],
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[5.0, 6.0], [7.0, 8.0]],
                ],
                [
                    [[9.0, 10.0], [11.0, 12.0]],
                    [[13.0, 14.0], [15.0, 16.0]],
                    [[9.0, 10.0], [11.0, 12.0]],
                    [[13.0, 14.0], [15.0, 16.0]],
                ],
            ],  # B=2, C=4, H=2, W=2
            2,
            4,
            2,
            2,
        ),
    ],
)
def test_to_complex(
    complex_metrics_instance,
    input_tensor_data,
    expected_batch_size,
    expected_channels,
    H,
    W,
):
    input_tensor = torch.tensor(input_tensor_data, dtype=torch.float32)
    # Ensure input_tensor has a consistent shape (B, C, H, W) for testing
    if input_tensor.ndim == 2:  # (B, C_total) for H=1, W=1
        input_tensor = input_tensor.view(expected_batch_size, expected_channels, 1, 1)
    elif (
        input_tensor.ndim == 3
    ):  # (B, H, C_total) for W=1. Reshape to (B, C_total, H, 1)
        input_tensor = input_tensor.permute(0, 2, 1).unsqueeze(-1)  # (B, C, H, 1)

    result = complex_metrics_instance.to_complex(input_tensor)

    if expected_channels == 2:  # Original re, im
        expected = torch.complex(input_tensor[:, 0], input_tensor[:, 1])
    else:  # Recombine even/odd channels
        re_parts = input_tensor[:, 0::2].sum(dim=1)
        im_parts = input_tensor[:, 1::2].sum(dim=1)
        expected = torch.complex(re_parts, im_parts)

    assert result.shape == (expected_batch_size, H, W)
    assert torch.allclose(result, expected)


def test_complex_ssim(complex_metrics_instance):
    # Test case 1: Identical tensors
    z1 = torch.complex(torch.randn(1, 10, 10), torch.randn(1, 10, 10))
    z2 = z1.clone()
    ssim = complex_metrics_instance.complex_ssim(z1, z2)
    assert ssim == pytest.approx(1.0)

    # Test case 2: Different tensors
    z1 = torch.complex(torch.randn(1, 10, 10), torch.randn(1, 10, 10))
    z2 = torch.complex(torch.randn(1, 10, 10), torch.randn(1, 10, 10))
    ssim = complex_metrics_instance.complex_ssim(z1, z2)
    assert 0.0 <= ssim <= 1.0


def test_evaluate_all(complex_metrics_instance):
    preds_np = np.array([[[1.0, 2.0]], [[3.0, 4.0]]], dtype=np.float32)
    gts_np = np.array([[[1.1, 2.1]], [[2.9, 3.9]]], dtype=np.float32)

    preds = (
        torch.tensor(preds_np).permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
    )  # (B, C, 1, 1)
    gts = (
        torch.tensor(gts_np).permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
    )  # (B, C, 1, 1)

    results = complex_metrics_instance.evaluate_all(preds, gts)

    assert "ssim" in results
    assert "mse" in results
    assert isinstance(results["ssim"], float)
    assert isinstance(results["mse"], float)
    assert results["ssim"] >= 0.0
    assert results["mse"] >= 0.0


def test_base_evaluator_metrics():
    from neugk.evaluate import BaseEvaluator
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({"validation": {"validate_every_n_epochs": 5}})
    # BaseEvaluator is abstract, but we can instantiate it if we don't call __call__
    # or we can mock it.
    evaluator = BaseEvaluator(cfg, [], [])

    # test _is_eval_epoch
    assert evaluator._is_eval_epoch(1) == True
    assert evaluator._is_eval_epoch(2) == False
    assert evaluator._is_eval_epoch(5) == True
    assert evaluator._is_eval_epoch(10) == True

    # test _accumulate_metrics
    metrics = {}
    n_timesteps_acc = torch.tensor(0.0)

    # batch 1: scalar metric
    metrics_i = {"mse": torch.tensor(0.5), "acc": 0.8}
    metrics, n_timesteps_acc = evaluator._accumulate_metrics(
        metrics, metrics_i, n_timesteps_acc, weight=1.0
    )
    assert metrics["mse"] == 0.5
    assert metrics["acc"] == 0.8
    assert n_timesteps_acc == 1.0

    # batch 2: sequence metric
    metrics_seq = {"rollout": torch.tensor([0.1, 0.2, 0.3])}
    metrics, n_timesteps_acc = evaluator._accumulate_metrics(
        metrics, metrics_seq, n_timesteps_acc, weight=1.0
    )
    assert metrics["rollout"].shape == (3,)
    assert n_timesteps_acc == 2.0

    # test _finalize_logs
    log_dict = {}
    valname = "val"
    final_logs = evaluator._finalize_logs(log_dict, metrics, n_timesteps_acc, valname)

    assert final_logs["val/mse"] == pytest.approx(0.5 / 2.0)
    assert final_logs["val/acc"] == pytest.approx(0.8 / 2.0)
    assert final_logs["val/rollout_x1"] == pytest.approx(0.1 / 2.0)
    assert final_logs["val/rollout_x2"] == pytest.approx(0.2 / 2.0)
    assert final_logs["val/rollout_x3"] == pytest.approx(0.3 / 2.0)

    # single step sequence should NOT have suffix
    metrics_single = {"single": torch.tensor([0.9])}
    metrics = {}
    n_timesteps_acc = torch.tensor(1.0)
    metrics, _ = evaluator._accumulate_metrics(metrics, metrics_single, n_timesteps_acc)
    final_logs = evaluator._finalize_logs({}, metrics, n_timesteps_acc, valname)
    assert "val/single" in final_logs
    assert "val/single_x1" not in final_logs
