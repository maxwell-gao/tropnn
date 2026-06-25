from __future__ import annotations

import torch

from tropnn import AbsDiffLUT, PairwiseLUT


def test_pairwise_lut_shape() -> None:
    layer = PairwiseLUT(8, 3, tables=4, comparisons=3, seed=0)
    y = layer(torch.randn(5, 8))
    assert y.shape == (5, 1, 3)


def test_absdiff_lut_shape() -> None:
    layer = AbsDiffLUT(8, 2, tables=4, comparisons=3, seed=0)
    x = torch.randn(5, 8)
    y = layer(x, x)
    assert y.shape == (5, 2)
