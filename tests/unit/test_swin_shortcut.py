import torch
import torch.nn as nn

from neugk.models.nd_vit.swin_layers import (
    DiTSwinTransformerBlock,
    SwinTransformerBlock,
    set_legacy_swin_shortcut,
)


def _block(cls=SwinTransformerBlock, **kw):
    torch.manual_seed(0)
    blk = cls(2, 16, 2, grid_size=(8, 8), window_size=(4, 4), shift_size=(2, 2), **kw)
    return blk.eval()


def _branches(blk, x):
    res1 = blk.skip(x) + blk.drop_path(blk.forward_part1(x))
    return res1, blk.forward_part2(res1)


def test_legacy_shortcut_is_the_doubled_residual():
    blk, x = _block(), torch.randn(3, 8, 8, 16)
    with torch.no_grad():
        res1, mlp = _branches(blk, x)
        assert blk.legacy_double_shortcut
        assert torch.equal(blk(x), res1 + (res1 + mlp))
        set_legacy_swin_shortcut(blk, False)
        assert torch.equal(blk(x), res1 + mlp)


def test_set_legacy_shortcut_skips_dit_blocks():
    model = nn.ModuleList([_block(), _block(DiTSwinTransformerBlock, cond_dim=4)])
    set_legacy_swin_shortcut(model, False)
    assert model[0].legacy_double_shortcut is False
    assert model[1].legacy_double_shortcut is True
