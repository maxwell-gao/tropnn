from __future__ import annotations

import pytest
import torch
from tropnn.backends.comparator_margin_triton import has_comparator_margin_triton
from tropnn.backends.pairwise_zig import has_pairwise_zig, pairwise_zig_forward, pairwise_zig_paged_forward, pairwise_zig_soa_forward, pairwise_zig_tree_tiled_forward
from tropnn import AbsDiffLUT, ComparatorTwoSidedMargin, PairwiseLUT, PairwiseWalshLUT
from tropnn.examples.emnist import EmnistPairwiseWalshClassifier
from tropnn.layers.surrogate import surrogate_gradient
from tropnn.tools.emnist_payload_width import ComparatorGeneratorLayer


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


def test_comparator_output_major_layout_preserves_sparse_writes() -> None:
    layer = ComparatorTwoSidedMargin(
        16,
        11,
        tables=5,
        comparisons=3,
        k_c=7,
        backend="torch",
        seed=4,
        use_output_scaling=False,
        reduction_layout="output_major",
    )
    counts = torch.bincount(layer.write_indices.reshape(-1), minlength=layer.output_dim)

    assert layer.csr_offsets[0].item() == 0
    assert layer.csr_offsets[-1].item() == layer.routes * 2 * layer.k_c
    assert torch.equal(layer.csr_offsets[1:] - layer.csr_offsets[:-1], counts)
    for dst in range(layer.output_dim):
        start = int(layer.csr_offsets[dst].item())
        end = int(layer.csr_offsets[dst + 1].item())
        for source, weight_idx in zip(layer.csr_sources[start:end].tolist(), layer.csr_weight_indices[start:end].tolist(), strict=True):
            slot = weight_idx % layer.k_c
            route = source // 2
            side = source - route * 2
            assert int(layer.write_indices[route, side, slot].item()) == dst


def test_binary_comparator_matches_scaled_signed_float_initialization_and_trains_master() -> None:
    torch.manual_seed(0)
    binary = ComparatorTwoSidedMargin(
        16,
        11,
        tables=5,
        comparisons=3,
        k_c=7,
        backend="torch",
        seed=4,
        use_output_scaling=False,
        weight_mode="binary",
    )
    float_ref = ComparatorTwoSidedMargin(
        16,
        11,
        tables=5,
        comparisons=3,
        k_c=7,
        backend="torch",
        seed=4,
        use_output_scaling=False,
        weight_mode="float",
    )
    x_binary = torch.randn(6, 16, requires_grad=True)
    x_float = x_binary.detach().clone().requires_grad_(True)

    y_binary = binary(x_binary)
    y_float = float_ref(x_float)

    assert torch.equal(binary.hard_write_codes().unique(), torch.tensor([-1, 1], dtype=torch.int8))
    assert binary.binary_code_flip_fraction() == 0.0
    assert torch.allclose(y_binary, y_float)
    y_binary.square().mean().backward()
    assert binary.write_weight.grad is not None
    assert binary.write_weight.grad.abs().sum() > 0
    assert binary.thresholds.grad is not None
    assert binary.thresholds.grad.abs().sum() > 0


def test_binary_comparator_rejects_zero_weight_initialization() -> None:
    with pytest.raises(ValueError, match="require weight_init='signed'"):
        ComparatorTwoSidedMargin(8, 5, tables=2, comparisons=2, k_c=3, weight_mode="binary", weight_init="zero")


@pytest.mark.skipif(not torch.cuda.is_available() or not has_comparator_margin_triton(), reason="CUDA Triton backend is not available")
def test_comparator_output_major_triton_matches_scatter_forward_backward() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    scatter = ComparatorTwoSidedMargin(
        32,
        19,
        tables=7,
        comparisons=4,
        k_c=9,
        backend="triton",
        seed=5,
        use_output_scaling=False,
        reduction_layout="scatter",
    ).to(device)
    output_major = ComparatorTwoSidedMargin(
        32,
        19,
        tables=7,
        comparisons=4,
        k_c=9,
        backend="triton",
        seed=5,
        use_output_scaling=False,
        reduction_layout="output_major",
    ).to(device)
    with torch.no_grad():
        output_major.thresholds.copy_(scatter.thresholds)
        output_major.write_weight.copy_(scatter.write_weight)

    x_scatter = torch.randn(6, 3, 32, device=device, requires_grad=True)
    x_output_major = x_scatter.detach().clone().requires_grad_(True)
    y_scatter = scatter(x_scatter)
    y_output_major = output_major(x_output_major)
    assert torch.allclose(y_output_major, y_scatter, atol=1e-5, rtol=1e-5)

    grad = torch.randn_like(y_scatter)
    y_scatter.backward(grad)
    y_output_major.backward(grad)

    assert torch.allclose(x_output_major.grad, x_scatter.grad, atol=1e-4, rtol=1e-4)
    assert torch.allclose(output_major.thresholds.grad, scatter.thresholds.grad, atol=1e-4, rtol=1e-4)
    assert torch.allclose(output_major.write_weight.grad, scatter.write_weight.grad, atol=1e-4, rtol=1e-4)


def test_comparator_tile_local_write_pattern_stays_inside_route_tile() -> None:
    layer = ComparatorTwoSidedMargin(
        32,
        48,
        tables=8,
        comparisons=3,
        k_c=5,
        backend="torch",
        seed=7,
        use_output_scaling=False,
        reduction_layout="tile_local",
        output_tile_size=16,
    )
    tiles = (layer.output_dim + layer.output_tile_size - 1) // layer.output_tile_size
    routes_per_tile = (layer.routes + tiles - 1) // tiles

    for route in range(layer.routes):
        tile = min(route // routes_per_tile, tiles - 1)
        start = tile * layer.output_tile_size
        end = min(start + layer.output_tile_size, layer.output_dim)
        route_indices = layer.write_indices[route].reshape(-1)
        assert int(route_indices.min().item()) >= start
        assert int(route_indices.max().item()) < end


@pytest.mark.skipif(not torch.cuda.is_available() or not has_comparator_margin_triton(), reason="CUDA Triton backend is not available")
def test_comparator_tile_local_triton_matches_torch_forward_backward() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    torch_ref = ComparatorTwoSidedMargin(
        32,
        48,
        tables=8,
        comparisons=3,
        k_c=5,
        backend="torch",
        seed=7,
        use_output_scaling=False,
        reduction_layout="tile_local",
        output_tile_size=16,
    ).to(device)
    tiled = ComparatorTwoSidedMargin(
        32,
        48,
        tables=8,
        comparisons=3,
        k_c=5,
        backend="triton",
        seed=7,
        use_output_scaling=False,
        reduction_layout="tile_local",
        output_tile_size=16,
    ).to(device)
    with torch.no_grad():
        tiled.thresholds.copy_(torch_ref.thresholds)
        tiled.write_weight.copy_(torch_ref.write_weight)

    x_ref = torch.randn(4, 2, 32, device=device, requires_grad=True)
    x_tiled = x_ref.detach().clone().requires_grad_(True)
    y_ref = torch_ref(x_ref)
    y_tiled = tiled(x_tiled)
    assert torch.allclose(y_tiled, y_ref, atol=1e-5, rtol=1e-5)

    grad = torch.randn_like(y_ref)
    y_ref.backward(grad)
    y_tiled.backward(grad)

    assert torch.allclose(x_tiled.grad, x_ref.grad, atol=1e-4, rtol=1e-4)
    assert torch.allclose(tiled.thresholds.grad, torch_ref.thresholds.grad, atol=1e-4, rtol=1e-4)
    assert torch.allclose(tiled.write_weight.grad, torch_ref.write_weight.grad, atol=1e-4, rtol=1e-4)


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
    layer = PairwiseWalshLUT(4, 2, tables=1, comparisons=3, walsh_order=2, seed=1, lut_dtype="fp32")
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
    structured = PairwiseWalshLUT(5, 3, tables=4, comparisons=3, walsh_order=2, seed=3, lut_dtype="fp32")
    flat = PairwiseLUT(5, 3, tables=4, comparisons=3, seed=3, lut_dtype="fp32")
    with torch.no_grad():
        flat.anchors.copy_(structured.anchors)
        flat.thresholds.copy_(structured.thresholds)
        flat.lut.copy_(structured.materialize_lut())

    x = torch.randn(7, 2, 5)
    structured.eval()
    flat.eval()
    assert torch.allclose(structured(x), flat(x), atol=1e-6)


def test_pairwise_walsh_gradient_uses_selected_row_and_neighbor_delta() -> None:
    layer = PairwiseWalshLUT(4, 1, tables=1, comparisons=2, walsh_order=2, use_output_scaling=False, seed=1, lut_dtype="fp32")
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


def test_pairwise_walsh_slope_coeff_materializes_order2_rows() -> None:
    layer = PairwiseWalshLUT(4, 2, tables=1, comparisons=3, walsh_order=1, slope_order=2, use_output_scaling=False, seed=1, lut_dtype="fp32")
    assert layer.walsh_term_count == 4
    assert layer.slope_term_count == 7
    assert layer.slope_constant is not None
    assert layer.slope_linear_coeff is not None
    assert layer.slope_pair_coeff is not None

    with torch.no_grad():
        layer.slope_constant.copy_(torch.tensor([[1.0, -2.0, 0.5]]))
        layer.slope_linear_coeff.copy_(
            torch.tensor(
                [
                    [
                        [0.5, 1.0, -0.25],
                        [-1.0, 0.25, 2.0],
                        [0.75, -0.5, 1.5],
                    ]
                ]
            )
        )
        layer.slope_pair_coeff.copy_(
            torch.tensor(
                [
                    [
                        [0.25, -0.5, 1.0],
                        [1.5, 0.75, -0.25],
                        [-1.0, 0.5, 0.25],
                    ]
                ]
            )
        )

    coeff = layer.materialize_slope_coeff()
    signs = torch.tensor([-1.0, 1.0, -1.0])
    pairs = torch.tensor([signs[0] * signs[1], signs[0] * signs[2], signs[1] * signs[2]])
    expected = layer.slope_constant[0] + layer.slope_linear_coeff[0] @ signs + layer.slope_pair_coeff[0] @ pairs
    assert torch.allclose(coeff[0, 2], expected)


def test_pairwise_walsh_margin_affine_forward_matches_formula() -> None:
    layer = PairwiseWalshLUT(4, 2, tables=1, comparisons=2, walsh_order=2, slope_order=1, use_output_scaling=False, seed=1, lut_dtype="fp32")
    assert layer.slope_constant is not None
    assert layer.slope_linear_coeff is not None
    assert layer.slope_pair_coeff is not None
    assert layer.slope_generator is not None
    with torch.no_grad():
        layer.anchors[0, :, 0] = torch.tensor([0, 2])
        layer.anchors[0, :, 1] = torch.tensor([1, 3])
        layer.thresholds.zero_()
        layer.constant.zero_()
        layer.linear_coeff.zero_()
        layer.pair_coeff.zero_()
        layer.slope_constant.copy_(torch.tensor([[0.5, -0.25]]))
        layer.slope_linear_coeff.copy_(torch.tensor([[[1.0, -0.5], [0.25, 2.0]]]))
        layer.slope_pair_coeff.zero_()
        layer.slope_generator.copy_(torch.tensor([[[1.0, 2.0], [-3.0, 0.5]]]))

    x = torch.tensor([[0.3, 0.1, -0.2, 0.4]])
    layer.eval()
    out = layer(x).squeeze(1)

    margins = torch.tensor([0.2, -0.6])
    signs = torch.tensor([1.0, -1.0])
    coeff = layer.slope_constant[0] + layer.slope_linear_coeff[0] @ signs
    expected = (coeff[:, None] * margins[:, None] * layer.slope_generator[0]).sum(dim=0)
    assert torch.allclose(out[0], expected)


def test_pairwise_walsh_margin_affine_ste_trains_slope_selector() -> None:
    layer = PairwiseWalshLUT(2, 1, tables=1, comparisons=1, walsh_order=1, slope_order=1, use_output_scaling=False, seed=1, lut_dtype="fp32")
    assert layer.slope_constant is not None
    assert layer.slope_linear_coeff is not None
    assert layer.slope_generator is not None
    with torch.no_grad():
        layer.anchors[0, 0] = torch.tensor([0, 1])
        layer.thresholds.zero_()
        layer.constant.zero_()
        layer.linear_coeff.zero_()
        layer.slope_constant[0, 0] = 0.5
        layer.slope_linear_coeff[0, 0, 0] = 2.0
        layer.slope_generator[0, 0, 0] = 3.0

    x = torch.tensor([[0.25, 0.0]], requires_grad=True)
    out = layer(x)
    out.sum().backward()

    margin = torch.tensor(0.25)
    current_coeff = torch.tensor(2.5)
    neighbor_coeff = torch.tensor(-1.5)
    generator = torch.tensor(3.0)
    expected_grad = current_coeff * generator + surrogate_gradient(margin) * (neighbor_coeff - current_coeff) * margin * generator

    assert torch.allclose(out.detach().reshape(()), current_coeff * margin * generator)
    assert torch.allclose(x.grad[0, 0], expected_grad)
    assert torch.allclose(x.grad[0, 1], -expected_grad)
    assert torch.allclose(layer.thresholds.grad[0, 0], -expected_grad)


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


def test_comparator_generator_sign_writes_endpoint_payload() -> None:
    layer = ComparatorGeneratorLayer(
        3,
        3,
        tables=1,
        comparisons=1,
        source="sign",
        write_policy="endpoint",
        k_c=2,
        anchor_policy="local",
        seed=0,
        use_output_scaling=False,
        use_min_margin_ste=True,
    )
    with torch.no_grad():
        layer.anchors[0, 0] = torch.tensor([0, 1])
        layer.thresholds.zero_()
        layer.write_indices[0] = torch.tensor([0, 1])
        layer.write_weight[0] = torch.tensor([2.0, -3.0])

    out, route = layer.compute(torch.tensor([[0.25, 0.0, 0.0]]))

    assert torch.equal(route, torch.tensor([[1]]))
    assert torch.allclose(out, torch.tensor([[2.0, -3.0, 0.0]]))


def test_comparator_generator_margin_uses_signed_distance() -> None:
    layer = ComparatorGeneratorLayer(
        2,
        2,
        tables=1,
        comparisons=1,
        source="margin",
        write_policy="endpoint",
        k_c=1,
        anchor_policy="local",
        seed=0,
        use_output_scaling=False,
        use_min_margin_ste=True,
    )
    with torch.no_grad():
        layer.anchors[0, 0] = torch.tensor([0, 1])
        layer.thresholds.zero_()
        layer.write_indices[0] = torch.tensor([1])
        layer.write_weight[0] = torch.tensor([4.0])

    out, route = layer.compute(torch.tensor([[0.0, 0.25]]))

    assert torch.equal(route, torch.tensor([[0]]))
    assert torch.allclose(out, torch.tensor([[0.0, -1.0]]))


def test_comparator_generator_signed_margin_uses_magnitude() -> None:
    layer = ComparatorGeneratorLayer(
        2,
        2,
        tables=1,
        comparisons=1,
        source="signed_margin",
        write_policy="endpoint",
        k_c=1,
        anchor_policy="local",
        seed=0,
        use_output_scaling=False,
        use_min_margin_ste=True,
    )
    with torch.no_grad():
        layer.anchors[0, 0] = torch.tensor([0, 1])
        layer.thresholds.zero_()
        layer.write_indices[0] = torch.tensor([1])
        layer.write_weight[0] = torch.tensor([4.0])

    out, route = layer.compute(torch.tensor([[0.0, 0.25]]))

    assert torch.equal(route, torch.tensor([[0]]))
    assert torch.allclose(out, torch.tensor([[0.0, 1.0]]))


def test_comparator_generator_sign_ste_routes_gradient_to_threshold_split() -> None:
    layer = ComparatorGeneratorLayer(
        2,
        1,
        tables=1,
        comparisons=1,
        source="sign",
        write_policy="endpoint",
        k_c=1,
        anchor_policy="local",
        seed=0,
        use_output_scaling=False,
        use_min_margin_ste=True,
    )
    with torch.no_grad():
        layer.anchors[0, 0] = torch.tensor([0, 1])
        layer.thresholds.zero_()
        layer.write_indices[0] = torch.tensor([0])
        layer.write_weight[0] = torch.tensor([3.0])

    x = torch.tensor([[0.25, 0.0]], requires_grad=True)
    out = layer(x)
    out.sum().backward()

    expected = 2.0 * surrogate_gradient(torch.tensor(0.25)) * torch.tensor(3.0)
    assert torch.allclose(out.detach().reshape(()), torch.tensor(3.0))
    assert torch.allclose(x.grad[0, 0], expected)
    assert torch.allclose(x.grad[0, 1], -expected)
    assert torch.allclose(layer.thresholds.grad[0, 0], -expected)
