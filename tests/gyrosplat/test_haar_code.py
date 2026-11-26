"""Reference-layout, derivative and serialized-reconstruction checks for Haar."""
import numpy as np
import pytest
import torch

pytest.importorskip("pywt")
from neugk.gyrosplat.baselines.wavelet import _forward, _inverse, wavelet_topk
from neugk.gyrosplat.baselines.haar_code import (
    inverse_haar,
    haar_linear_code,
    code_from_saved,
)
from neugk.gyrosplat.backends.synthetic import SyntheticBackend
from neugk.gyrosplat.datasets.synthetic import make_synthetic_snapshot
from neugk.gyrosplat.baselines.fourier import fourier_topk
from neugk.gyrosplat.baselines.gn_on_code import gauss_newton_on_code
from neugk.gyrosplat.config import GaussNewtonConfig


@pytest.mark.parametrize("shape", [(4, 6), (3, 5), (4, 2, 3, 7, 4)])
def test_inverse_matches_pywavelets_and_has_correct_gradients(shape):
    array = np.random.default_rng(4).normal(size=shape)
    coefficients, slices = _forward(array, "haar", 1)
    # Arbitrary retained supports must match too, not just a full roundtrip.
    coefficients *= np.random.default_rng(8).random(coefficients.shape) > 0.6
    x = torch.tensor(coefficients, dtype=torch.float64, requires_grad=True)
    expected = _inverse(coefficients, slices, "haar", shape)
    np.testing.assert_allclose(
        inverse_haar(x, shape).detach().numpy(), expected, rtol=1e-12, atol=1e-12
    )
    # Directional finite difference checks the adjoint through every dimension.
    direction = torch.randn_like(x)
    weight = torch.randn(shape, dtype=torch.float64)
    scalar = (inverse_haar(x, shape) * weight).sum()
    (grad,) = torch.autograd.grad(scalar, x)
    eps = 1e-5
    finite = (
        (
            inverse_haar(x + eps * direction, shape)
            - inverse_haar(x - eps * direction, shape)
        )
        * weight
    ).sum() / (2 * eps)
    torch.testing.assert_close((grad * direction).sum(), finite, rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize("budget", [301, 600])
def test_topk_matches_existing_baseline_and_reloads(budget, tmp_path):
    backend = SyntheticBackend(
        make_synthetic_snapshot((4, 2, 3, 7, 8), seed=5), band=[1, 2, 3]
    )
    fields = backend.to_real(backend.gt)
    H, n, code, parts = haar_linear_code(backend, fields, budget)
    ref, counts, _ = wavelet_topk(fields, budget, "haar", 1)
    torch.testing.assert_close(H, backend.to_spectral(ref), rtol=2e-6, atol=2e-6)
    assert n == counts == [budget]
    path = tmp_path / "code.pt"
    torch.save({"parts": parts, "values": code.values}, path)
    saved = torch.load(path, weights_only=False)
    reloaded = code_from_saved(backend, **saved)
    assert torch.equal(H, reloaded.decode(reloaded.values))


def test_gn_returns_the_actual_final_code():
    backend = SyntheticBackend(
        make_synthetic_snapshot((4, 2, 3, 8, 8), seed=5), band=[1, 2, 3]
    )
    _, _, code = fourier_topk(backend.gt, 300, backend.real_axes)
    cfg = GaussNewtonConfig(steps=2)
    H, info, fitted = gauss_newton_on_code(
        backend, code, cfg, log=lambda *a: None, return_code=True
    )
    torch.testing.assert_close(fitted.decode(fitted.values), H, rtol=0, atol=0)
    # The optional export must not change the default optimization result.
    H0, info0 = gauss_newton_on_code(backend, code, cfg, log=lambda *a: None)
    torch.testing.assert_close(H0, H, rtol=0, atol=0)
    assert info == info0
