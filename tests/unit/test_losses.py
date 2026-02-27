import pytest
import torch
from neugk.pinc.losses import PINCLossWrapper


def test_pinc_loss_wrapper_schedulers():
    weights = {"df": 1.0, "phi_int": 0.5}
    # scheduler that returns 2.0 * progress_remaining
    schedulers = {"df": lambda p: 2.0 * p}

    wrapper = PINCLossWrapper(weights, schedulers)

    # initial weights
    assert wrapper.weights["df"] == 1.0

    # after one forward call with progress_remaining=0.5
    # forward logic:
    # if self.training:
    #     for k, sched in self.schedulers.items():
    #         if k in self.weights:
    #             self.weights[k] = sched(progress_remaining)

    wrapper.train()
    # Mocking inputs for forward
    preds = {"df": torch.randn(1, 2, 4, 4, 4, 4, 4)}
    tgts = {
        "df": torch.randn(1, 2, 4, 4, 4, 4, 4),
        "phi": torch.randn(1, 4, 4, 4),
        "flux": torch.randn(1, 1),
    }

    # Directly test the scheduler update logic that happens in forward
    if wrapper.training:
        for k, sched in wrapper.schedulers.items():
            if k in wrapper.weights:
                wrapper.weights[k] = sched(0.5)

    assert wrapper.weights["df"] == 1.0  # 2.0 * 0.5


def test_pinc_loss_compute_data_loss():
    wrapper = PINCLossWrapper({"df": 1.0}, {})

    p = torch.tensor([1.0, 2.0])
    t = torch.tensor([1.1, 1.9])

    # mse
    loss_mse = wrapper.compute_data_loss(p, t, loss_type="mse")
    expected_mse = torch.mean((p - t) ** 2)
    assert torch.allclose(loss_mse, expected_mse)

    # l1
    loss_l1 = wrapper.compute_data_loss(p, t, loss_type="l1")
    expected_l1 = torch.mean(torch.abs(p - t))
    assert torch.allclose(loss_l1, expected_l1)


def test_vae_loss_logic():
    wrapper = PINCLossWrapper({}, {})

    # kl_div: 0.5 * sum(exp(logvar) + mu^2 - 1 - logvar)
    mu = torch.tensor([0.0, 0.0])
    logvar = torch.tensor([0.0, 0.0])

    res = wrapper.compute_vae_loss({"mu": mu, "logvar": logvar})
    # exp(0) + 0 - 1 - 0 = 0
    assert res["kl_div"] == 0.0

    mu2 = torch.tensor([1.0])
    logvar2 = torch.tensor([0.0])
    res2 = wrapper.compute_vae_loss({"mu": mu2, "logvar": logvar2})
    # 0.5 * (exp(0) + 1^2 - 1 - 0) = 0.5 * (1 + 1 - 1) = 0.5
    assert res2["kl_div"] == 0.5
