from __future__ import annotations

import pytest
import torch
from tropnn.backends.pairwise_zig import has_pairwise_zig, pairwise_zig_forward, pairwise_zig_paged_forward, pairwise_zig_soa_forward, pairwise_zig_tree_tiled_forward
from tropnn import AbsDiffLUT, PairwiseLUT, PairwiseWalshLUT
from tropnn.examples.emnist import EmnistPairwiseWalshClassifier
from tropnn.layers.surrogate import surrogate_gradient


def test_fast_sigmoid_odd_surrogate_has_lut_direction() -> None:
    u = torch.tensor([-2.0, 0.0, 2.0])
    grad = surrogate_gradient(u, "fast_sigmoid_odd")

    assert grad[0] > 0
    assert grad[1] == 0
    assert grad[2] < 0


def test_pairwise_zig_backend_is_inference_only() -> None:
    layer = PairwiseLUT(8, 5, tables=3, comparisons=3, backend="zig", seed=1)

    with pytest.raises(RuntimeError, match="inference-only"):
        layer(torch.randn(4, 8))


@pytest.mark.skipif(not has_pairwise_zig(), reason="Zig backend is not available")
def test_pairwise_zig_paged_forward_matches_standard_zig_forward() -> None:
    torch.manual_seed(0)
    layer = PairwiseLUT(16, 9, tables=5, comparisons=3, backend="torch", seed=2, lut_init_std=0.02, use_output_scaling=False)
    x = torch.randn(4, 3, 16)
    standard = pairwise_zig_forward(x.float(), layer.anchors, layer.thresholds.detach().float(), layer.lut.detach().float(), lut_dtype="f32")
    paged = pairwise_zig_paged_forward(x.float(), layer.anchors, layer.thresholds.detach().float(), layer.lut.detach().float(), lut_dtype="f32", page_size=4)
    assert torch.allclose(paged, standard, atol=1e-6)


@pytest.mark.skipif(not has_pairwise_zig(), reason="Zig backend is not available")
def test_pairwise_zig_soa_forward_matches_standard_zig_forward() -> None:
    torch.manual_seed(0)
    layer = PairwiseLUT(16, 9, tables=5, comparisons=3, backend="torch", seed=2, lut_init_std=0.02, use_output_scaling=False)
    x = torch.randn(4, 3, 16)
    standard = pairwise_zig_forward(x.float(), layer.anchors, layer.thresholds.detach().float(), layer.lut.detach().float(), lut_dtype="f32")
    soa = pairwise_zig_soa_forward(x.float(), layer.anchors, layer.thresholds.detach().float(), layer.lut.detach().float(), lut_dtype="f32")
    assert torch.allclose(soa, standard, atol=1e-6)


@pytest.mark.skipif(not has_pairwise_zig(), reason="Zig backend is not available")
def test_pairwise_zig_tree_tiled_forward_matches_standard_zig_forward() -> None:
    torch.manual_seed(0)
    layer = PairwiseLUT(16, 9, tables=5, comparisons=3, backend="torch", seed=2, lut_init_std=0.02, use_output_scaling=False)
    x = torch.randn(4, 3, 16)
    standard = pairwise_zig_forward(x.float(), layer.anchors, layer.thresholds.detach().float(), layer.lut.detach().float(), lut_dtype="f32")
    tree = pairwise_zig_tree_tiled_forward(x.float(), layer.anchors, layer.thresholds.detach().float(), layer.lut.detach().float())
    assert torch.allclose(tree, standard, atol=1e-6)


def test_absdiff_lut_selects_rows_from_coordinate_closeness() -> None:
    layer = AbsDiffLUT(3, 2, tables=1, comparisons=2, width_init=0.5, use_output_scaling=False, seed=0, lut_init_std=0.0)
    with torch.no_grad():
        layer.coords[0] = torch.tensor([0, 1])
        layer.log_widths.fill_(AbsDiffLUT._inverse_softplus(0.5))
        layer.lut[0].copy_(
            torch.tensor(
                [
                    [0.0, 0.0],
                    [1.0, 10.0],
                    [2.0, 20.0],
                    [3.0, 30.0],
                ]
            )
        )

    query = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    key = torch.tensor([[0.1, 1.0, 0.0], [1.0, 0.1, 0.0]])

    out = layer(query, key)

    assert torch.allclose(out, torch.tensor([[1.0, 10.0], [2.0, 20.0]]))
    assert torch.equal(layer._last_indices, torch.tensor([[1], [2]]))


def test_absdiff_lut_min_margin_ste_routes_credit_to_query_key_and_width() -> None:
    layer = AbsDiffLUT(2, 1, tables=1, comparisons=2, width_init=0.5, use_output_scaling=False, seed=0, lut_init_std=0.0)
    with torch.no_grad():
        layer.coords[0] = torch.tensor([0, 1])
        layer.log_widths.fill_(AbsDiffLUT._inverse_softplus(0.5))
        layer.lut[0, :, 0] = torch.tensor([5.0, 2.0, 7.0, 11.0])

    query = torch.tensor([[0.4, 2.0]], requires_grad=True)
    key = torch.tensor([[0.0, 0.0]], requires_grad=True)
    out = layer(query, key)
    out.sum().backward()

    current = layer.lut[0, 1, 0].detach()
    neighbor = layer.lut[0, 0, 0].detach()
    corr = neighbor - current
    u = torch.tensor(0.1)
    expected_q0 = -surrogate_gradient(u) * corr
    expected_k0 = surrogate_gradient(u) * corr
    expected_width = surrogate_gradient(u) * corr * torch.sigmoid(layer.log_widths.detach()[0, 0])

    assert torch.allclose(out.detach().reshape(()), current)
    assert torch.allclose(query.grad[0, 0], expected_q0)
    assert torch.allclose(key.grad[0, 0], expected_k0)
    assert torch.allclose(layer.log_widths.grad[0, 0], expected_width)
    assert layer.log_widths.grad[0, 1] == 0


def test_pairwise_walsh_materializes_order2_rows() -> None:
    layer = PairwiseWalshLUT(4, 2, tables=1, comparisons=3, walsh_order=2, seed=1)
    assert layer.walsh_term_count == 7

    with torch.no_grad():
        layer.constant.copy_(torch.tensor([[1.0, -1.0]]))
        layer.linear_coeff.copy_(
            torch.tensor(
                [
                    [
                        [0.5, 1.0],
                        [2.0, -0.25],
                        [-1.0, 0.75],
                    ]
                ]
            )
        )
        layer.pair_coeff.copy_(
            torch.tensor(
                [
                    [
                        [0.25, 0.5],
                        [-0.5, 0.25],
                        [1.5, -1.25],
                    ]
                ]
            )
        )

    lut = layer.materialize_lut()
    signs = torch.tensor([-1.0, 1.0, -1.0])
    pairs = torch.tensor([signs[0] * signs[1], signs[0] * signs[2], signs[1] * signs[2]])
    expected = layer.constant[0] + signs @ layer.linear_coeff[0] + pairs @ layer.pair_coeff[0]
    assert torch.allclose(lut[0, 2], expected)


def test_pairwise_walsh_forward_matches_flat_lut_rows() -> None:
    torch.manual_seed(0)
    structured = PairwiseWalshLUT(5, 3, tables=4, comparisons=3, walsh_order=2, seed=3)
    flat = PairwiseLUT(5, 3, tables=4, comparisons=3, seed=3)
    with torch.no_grad():
        flat.anchors.copy_(structured.anchors)
        flat.thresholds.copy_(structured.thresholds)
        flat.lut.copy_(structured.materialize_lut())

    x = torch.randn(7, 2, 5)
    structured.eval()
    flat.eval()
    assert torch.allclose(structured(x), flat(x), atol=1e-6)


def test_pairwise_walsh_gradient_uses_selected_row_and_neighbor_delta() -> None:
    layer = PairwiseWalshLUT(4, 1, tables=1, comparisons=2, walsh_order=2, use_output_scaling=False, seed=1)
    with torch.no_grad():
        layer.anchors[0, :, 0] = torch.tensor([0, 2])
        layer.anchors[0, :, 1] = torch.tensor([1, 3])
        layer.thresholds.zero_()
        layer.constant.fill_(0.0)
        layer.linear_coeff[0, :, 0] = torch.tensor([1.0, 2.0])
        layer.pair_coeff[0, :, 0] = torch.tensor([0.5])

    x = torch.tensor([[0.1, 0.0, 2.0, 0.0]], requires_grad=True)
    out = layer(x)
    out.sum().backward()

    lut = layer.materialize_lut().detach()
    current = lut[0, 3, 0]
    neighbor = lut[0, 2, 0]
    expected_x0 = surrogate_gradient(torch.tensor(0.1)) * (neighbor - current)

    assert torch.allclose(out.detach().reshape(()), current)
    assert torch.allclose(layer.constant.grad, torch.tensor([[1.0]]))
    assert torch.allclose(layer.linear_coeff.grad[0, :, 0], torch.tensor([1.0, 1.0]))
    assert torch.allclose(layer.pair_coeff.grad[0, :, 0], torch.tensor([1.0]))
    assert torch.allclose(x.grad[0, 0], expected_x0)
    assert torch.allclose(x.grad[0, 1], -expected_x0)
    assert torch.allclose(layer.thresholds.grad[0, 0], -expected_x0)
    assert layer.thresholds.grad[0, 1] == 0


def test_pairwise_walsh_emnist_classifier_shape() -> None:
    model = EmnistPairwiseWalshClassifier(
        input_dim=28 * 28,
        hidden_dim=64,
        num_classes=10,
        depth=2,
        heads=4,
        cells=4,
        code_dim=16,
        route_terms=2,
        fan_value_mode="site",
        fan_basis_rank=8,
        comparisons=6,
        pairwise_tables=4,
        walsh_order=2,
        backend="torch",
        seed=0,
    )
    logits = model(torch.randn(4, 28 * 28))
    assert logits.shape == (4, 10)
