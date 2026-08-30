from __future__ import annotations

import pytest
import torch
from tropnn.backends.sum_pyramid_tilelang import (
    has_sum_pyramid_tilelang,
    sum_pyramid_pairwise_route_tilelang_full,
    sum_pyramid_pairwise_route_torch,
)
from tropnn.layers.accumulation import SumPyramid
from tropnn.tools.zipf_groupsum_pclut_capacity_law import make_pyramid_anchors


def test_torch_reference_matches_explicit_sum_pyramid_and_pair_formula() -> None:
    torch.manual_seed(7)
    x = torch.randn(9, 16)
    pyramid = SumPyramid(16, signed=True, seed=11)
    anchors = make_pyramid_anchors(16, 4, 3, policy="level_biased", seed=13)
    thresholds = torch.randn(4, 3) * 0.1
    indices, margins = sum_pyramid_pairwise_route_torch(x, pyramid.signs, anchors, thresholds)
    features = pyramid(x)
    expected_margins = features[:, anchors[..., 0]] - features[:, anchors[..., 1]] - thresholds
    powers = 2 ** torch.arange(3)
    expected_indices = ((expected_margins > 0).long() * powers.view(1, 1, -1)).sum(dim=-1)
    assert torch.equal(margins, expected_margins)
    assert torch.equal(indices, expected_indices)


def test_torch_reference_margin_backward_is_the_sum_pyramid_transpose() -> None:
    torch.manual_seed(17)
    x = torch.randn(5, 8, requires_grad=True)
    signs = SumPyramid(8, signed=True, seed=3).signs
    anchors = make_pyramid_anchors(8, 3, 2, policy="node_uniform", seed=5)
    thresholds = torch.randn(3, 2, requires_grad=True)
    _, margins = sum_pyramid_pairwise_route_torch(x, signs, anchors, thresholds)
    weight = torch.randn_like(margins)
    (margins * weight).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert thresholds.grad is not None
    assert torch.equal(thresholds.grad, -weight.sum(dim=0))


@pytest.mark.skipif(not torch.cuda.is_available() or not has_sum_pyramid_tilelang(), reason="requires CUDA TileLang")
@pytest.mark.parametrize(("n_features", "tables", "comparisons", "batch"), [(64, 8, 4, 32), (1024, 32, 6, 16)])
def test_tilelang_fused_route_matches_torch_forward_and_backward(
    n_features: int,
    tables: int,
    comparisons: int,
    batch: int,
) -> None:
    device = torch.device("cuda")
    torch.manual_seed(23)
    signs = SumPyramid(n_features, signed=True, seed=29).signs.to(device)
    anchors = make_pyramid_anchors(n_features, tables, comparisons, policy="level_biased", seed=31).to(device)
    threshold_ref = (torch.randn(tables, comparisons, device=device) * 0.1).requires_grad_(True)
    threshold_opt = threshold_ref.detach().clone().requires_grad_(True)
    x_ref = torch.randn(batch, n_features, device=device, requires_grad=True)
    x_opt = x_ref.detach().clone().requires_grad_(True)
    idx_ref, margin_ref = sum_pyramid_pairwise_route_torch(x_ref, signs, anchors, threshold_ref)
    idx_opt, margin_opt, rmin_opt = sum_pyramid_pairwise_route_tilelang_full(
        x_opt,
        signs,
        anchors,
        threshold_opt,
    )
    assert torch.equal(idx_ref, idx_opt)
    assert torch.equal(margin_ref.detach().abs().argmin(dim=-1).to(torch.uint8), rmin_opt)
    assert float((margin_ref - margin_opt).detach().abs().max()) <= 2e-5
    weight = torch.randn_like(margin_ref)
    (margin_ref * weight).sum().backward()
    (margin_opt * weight).sum().backward()
    assert float((x_ref.grad - x_opt.grad).detach().abs().max()) <= 2e-5
    assert float((threshold_ref.grad - threshold_opt.grad).detach().abs().max()) <= 2e-5
