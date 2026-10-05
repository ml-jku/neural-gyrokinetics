"""gyaradax integrals run in float64 without switching the process to x64."""

from __future__ import annotations

import importlib.util

import jax
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("gyaradax") is None, reason="gyaradax not installed"
)


def test_gyaradax_import_keeps_x64_off():
    from neugk_jax.evaluate.integrals import _float64, _import_gyaradax

    assert not jax.config.jax_enable_x64
    _import_gyaradax()
    assert not jax.config.jax_enable_x64
    seen = _float64(lambda: jax.config.jax_enable_x64)()
    assert seen and not jax.config.jax_enable_x64
