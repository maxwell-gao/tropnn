from __future__ import annotations

import pytest
import torch
from tropnn import PairwiseLinear
from tropnn.layers.surrogate import surrogate_gradient


def test_fast_sigmoid_odd_surrogate_has_lut_direction() -> None:
    u = torch.tensor([-2.0, 0.0, 2.0])
    grad = surrogate_gradient(u, "fast_sigmoid_odd")

    assert grad[0] > 0
    assert grad[1] == 0
    assert grad[2] < 0


def test_pairwise_zig_backend_is_inference_only() -> None:
    layer = PairwiseLinear(8, 5, tables=3, comparisons=3, backend="zig", seed=1)

    with pytest.raises(RuntimeError, match="inference-only"):
        layer(torch.randn(4, 8))
