"""Parity tests against the torch reference; skipped unless torch and ``neugk`` import.

``NEUGK_TORCH_REPO`` points at the torch repository (default: the parent of this repo).
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

_REPO = os.environ.get("NEUGK_TORCH_REPO", str(Path(__file__).resolve().parents[3]))
if _REPO not in sys.path:
    sys.path.append(_REPO)
_HAVE = all(importlib.util.find_spec(m) is not None for m in ("torch", "neugk"))
collect_ignore_glob = [] if _HAVE else ["test_*.py"]


@pytest.fixture(scope="session")
def torch_repo_root() -> Path:
    return Path(_REPO)


@pytest.fixture(scope="module")
def single_swin_residual():
    """Torch ``SwinTransformerBlock`` with the single residual, also on a doubled torch tree."""
    from neugk.models.nd_vit import swin_layers

    orig = swin_layers.SwinTransformerBlock.forward

    def forward(self, x):
        x = self.skip(x) + self.drop_path(self.forward_part1(x))
        return x + self.forward_part2(x)

    from torch_ref import torch_doubles_swin_shortcut

    if torch_doubles_swin_shortcut():
        swin_layers.SwinTransformerBlock.forward = forward
    yield
    swin_layers.SwinTransformerBlock.forward = orig
