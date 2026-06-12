"""Sanity tests for the fold / unfold im2col primitives.

Both should be exact inverses (modulo channel-mixing) for any patch_size and
grid that divides cleanly.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import pytest

from neugk_jax.models.patching import (
    fold_patches,
    unfold_patches,
    pad_to_blocks,
    unpad,
)


@pytest.mark.parametrize("spatial,patch", [
    ((8, 8), (4, 4)),
    ((16, 24), (4, 8)),
    ((8, 12, 4), (2, 4, 2)),
    ((8, 4, 16, 4), (2, 1, 4, 2)),  # axis with patch=1 is passthrough
])
def test_fold_unfold_inverse(spatial, patch):
    x = jr.normal(jr.PRNGKey(0), (*spatial, 6))
    folded = fold_patches(x, patch)
    # folded shape: (*grid, prod(patch)*c)
    grid = tuple(s // p for s, p in zip(spatial, patch))
    prod_p = 1
    for p in patch:
        prod_p *= p
    assert folded.shape == (*grid, prod_p * 6)
    # round trip
    restored = unfold_patches(folded, patch, out_channels=6)
    assert restored.shape == x.shape
    assert jnp.allclose(restored, x, atol=1e-6)


def test_fold_groups_by_axis_first():
    """Document the layout: grid axes precede patch axes in the inner reshape."""
    x = jnp.arange(4 * 4 * 1, dtype=jnp.float32).reshape(4, 4, 1)  # 2D, c=1
    folded = fold_patches(x, (2, 2))
    # grid = (2, 2), prod_patch*c = 4 → shape (2, 2, 4)
    assert folded.shape == (2, 2, 4)
    # cell (0, 0) should be the top-left 2x2 patch (0, 1, 4, 5) — flattened with row-major
    cell00 = folded[0, 0]
    assert jnp.allclose(cell00.reshape(2, 2), x[:2, :2, 0])


def test_pad_unpad_roundtrip():
    x = jr.normal(jr.PRNGKey(0), (7, 13, 3))
    padded, pads = pad_to_blocks(x, (4, 5))
    assert padded.shape[0] % 4 == 0
    assert padded.shape[1] % 5 == 0
    restored = unpad(padded, pads, (7, 13))
    assert jnp.allclose(restored, x)
