from __future__ import annotations

import torch
from tropnn import HashSharedSparseHinge
from tropnn.tools.emnist_payload_width import PayloadWidthEmnistClassifier


def _small_layer(mode: str, *, seed: int = 7) -> HashSharedSparseHinge:
    return HashSharedSparseHinge(
        4,
        3,
        tables=1,
        comparisons=2,
        pool_size=4,
        candidates_per_code=2,
        margin_fan_in=2,
        write_fan_out=2,
        selection_mode=mode,  # type: ignore[arg-type]
        seed=seed,
        use_output_scaling=False,
    )


def test_shared_pool_modes_have_identical_program_initialization() -> None:
    selected = _small_layer("hash")
    fixed = _small_layer("fixed")
    all_scan = _small_layer("all")

    for name in (
        "anchors",
        "candidate_ranking",
        "read_indices",
        "write_indices",
        "read_weight",
        "margin_thresholds",
        "write_weight",
        "thresholds",
    ):
        assert torch.equal(getattr(selected, name), getattr(fixed, name))
        assert torch.equal(getattr(selected, name), getattr(all_scan, name))
    assert sum(p.numel() for p in selected.parameters()) == sum(
        p.numel() for p in all_scan.parameters()
    )
    assert selected.active_candidates == fixed.active_candidates == 2
    assert all_scan.active_candidates == 4


def test_shortlist_prefixes_are_nested_across_k() -> None:
    common = dict(
        input_dim=9,
        output_dim=7,
        tables=2,
        comparisons=3,
        pool_size=4,
        margin_fan_in=3,
        write_fan_out=2,
        selection_mode="hash",
        seed=17,
    )
    k2 = HashSharedSparseHinge(candidates_per_code=2, **common)
    k4 = HashSharedSparseHinge(candidates_per_code=4, **common)
    assert torch.equal(k2.candidate_ranking, k4.candidate_ranking)
    for name in ("read_indices", "write_indices", "read_weight", "write_weight"):
        assert torch.equal(getattr(k2, name), getattr(k4, name))
    codes = torch.arange(k2.table_size).view(-1, 1).expand(-1, k2.tables)
    assert torch.equal(
        k2._candidate_ids_for_codes(codes),
        k4._candidate_ids_for_codes(codes)[..., :2],
    )


def test_k_equals_pool_matches_all_scan_forward_and_gradients() -> None:
    common = dict(
        input_dim=5,
        output_dim=4,
        tables=2,
        comparisons=2,
        pool_size=4,
        candidates_per_code=4,
        margin_fan_in=3,
        write_fan_out=2,
        seed=23,
        fixed_zero_hash_threshold=True,
    )
    selected = HashSharedSparseHinge(selection_mode="hash", **common).double().eval()
    all_scan = HashSharedSparseHinge(selection_mode="all", **common).double().eval()
    all_scan.load_state_dict(selected.state_dict())
    x_selected = torch.randn(6, 5, dtype=torch.double, requires_grad=True)
    x_all = x_selected.detach().clone().requires_grad_(True)
    y_selected = selected(x_selected)
    y_all = all_scan(x_all)
    assert torch.allclose(y_selected, y_all, atol=1e-12, rtol=1e-12)
    y_selected.square().sum().backward()
    y_all.square().sum().backward()
    assert torch.allclose(x_selected.grad, x_all.grad, atol=1e-5, rtol=1e-5)
    for name in ("read_weight", "margin_thresholds", "write_weight"):
        assert torch.allclose(
            getattr(selected, name).grad,
            getattr(all_scan, name).grad,
            atol=1e-5,
            rtol=1e-5,
        )


def test_balanced_candidate_map_is_unique_covered_and_deterministic() -> None:
    first = HashSharedSparseHinge(
        9,
        7,
        tables=3,
        comparisons=3,
        pool_size=4,
        candidates_per_code=2,
        margin_fan_in=3,
        write_fan_out=2,
        seed=11,
    )
    second = HashSharedSparseHinge(
        9,
        7,
        tables=3,
        comparisons=3,
        pool_size=4,
        candidates_per_code=2,
        margin_fan_in=3,
        write_fan_out=2,
        seed=11,
    )

    assert torch.equal(first.candidate_ranking, second.candidate_ranking)
    assert first.candidate_ranking.shape == (3, 8, 4)
    assert first.candidate_ranking.dtype == torch.long
    assert int(first.candidate_ranking.min()) == 0
    assert int(first.candidate_ranking.max()) == 3
    assert torch.all(
        first.candidate_ranking.sort(dim=-1).values
        == torch.arange(4).view(1, 1, 4)
    )
    degrees = first.candidate_code_degrees
    assert torch.equal(degrees, torch.full_like(degrees, 4))


def test_hash_fixed_and_all_scan_execute_the_expected_candidates() -> None:
    layers = {
        mode: HashSharedSparseHinge(
            3,
            1,
            tables=1,
            comparisons=1,
            pool_size=2,
            candidates_per_code=1,
            margin_fan_in=1,
            write_fan_out=1,
            selection_mode=mode,  # type: ignore[arg-type]
            seed=3,
            use_output_scaling=False,
            fixed_zero_hash_threshold=True,
        ).eval()
        for mode in ("hash", "fixed", "all")
    }
    for layer in layers.values():
        with torch.no_grad():
            layer.anchors[0, 0] = torch.tensor([0, 1])
            layer.candidate_ranking[0, 0, 0] = 0
            layer.candidate_ranking[0, 1, 0] = 1
            layer.read_weight.zero_()
            layer.margin_thresholds.fill_(-1.0)
            layer.write_weight[0, 0, 0] = 1.0
            layer.write_weight[0, 1, 0] = 3.0

    low = torch.tensor([[0.0, 1.0, 0.0]])
    high = torch.tensor([[1.0, 0.0, 0.0]])
    assert torch.allclose(layers["hash"](low), torch.tensor([[1.0]]))
    assert torch.allclose(layers["hash"](high), torch.tensor([[3.0]]))
    assert torch.allclose(layers["fixed"](low), torch.tensor([[1.0]]))
    assert torch.allclose(layers["fixed"](high), torch.tensor([[1.0]]))
    assert torch.allclose(layers["all"](low), torch.tensor([[4.0]]))
    assert torch.allclose(layers["all"](high), torch.tensor([[4.0]]))


def test_shared_candidate_accumulates_gradients_from_multiple_hash_codes() -> None:
    layer = HashSharedSparseHinge(
        3,
        1,
        tables=1,
        comparisons=1,
        pool_size=2,
        candidates_per_code=1,
        margin_fan_in=1,
        write_fan_out=1,
        selection_mode="hash",
        seed=5,
        use_output_scaling=False,
        fixed_zero_hash_threshold=True,
    ).double()
    layer.eval()
    with torch.no_grad():
        layer.anchors[0, 0] = torch.tensor([0, 1])
        layer.candidate_ranking[0, :, 0] = 0
        layer.read_indices[0, 0, 0] = 2
        layer.write_indices[0, 0, 0] = 0
        layer.read_weight.zero_()
        layer.read_weight[0, 0, 0] = 1.0
        layer.margin_thresholds.zero_()
        layer.write_weight.zero_()
        layer.write_weight[0, 0, 0] = 2.0

    x = torch.tensor(
        [[0.0, 1.0, 1.0], [1.0, 0.0, 3.0]],
        dtype=torch.double,
        requires_grad=True,
    )
    output = layer(x)
    assert torch.allclose(output[:, 0], torch.tensor([2.0, 6.0], dtype=torch.double))
    output.sum().backward()
    assert torch.allclose(
        layer.write_weight.grad[0, 0, 0],
        torch.tensor(4.0, dtype=torch.double),
    )
    assert float(layer.read_weight.grad[0, 0, 0]) == 8.0
    assert torch.equal(
        layer.write_weight.grad[0, 1],
        torch.zeros_like(layer.write_weight.grad[0, 1]),
    )


def test_only_hash_selection_uses_coarse_route_ste() -> None:
    layers = {mode: _small_layer(mode) for mode in ("hash", "fixed", "all")}
    x = torch.randn(6, 4, requires_grad=True)
    for mode, layer in layers.items():
        layer.train()
        output = layer(x)
        output.square().sum().backward(retain_graph=True)
        hash_grad = layer.thresholds.grad
        if mode == "hash":
            assert hash_grad is not None and float(hash_grad.abs().sum()) > 0.0
        else:
            assert hash_grad is None or float(hash_grad.abs().sum()) == 0.0
        assert layer.margin_thresholds.grad is not None
        layer.zero_grad(set_to_none=True)


def test_all_scan_ignores_candidate_map_and_trains_every_pool_row() -> None:
    layer = _small_layer("all")
    layer.train()
    with torch.no_grad():
        layer.margin_thresholds.fill_(-5.0)
        layer.write_weight.fill_(1.0)
    x = torch.randn(5, 4, requires_grad=True)
    before = layer(x).detach()
    with torch.no_grad():
        layer.candidate_ranking.copy_(layer.candidate_ranking.flip(-1))
    after = layer(x)
    assert torch.allclose(before, after.detach())
    after.sum().backward()
    for gradient in (
        layer.read_weight.grad,
        layer.margin_thresholds.grad,
        layer.write_weight.grad,
    ):
        assert gradient is not None
        per_candidate = gradient.reshape(layer.tables, layer.pool_size, -1).abs().sum(-1)
        assert torch.all(per_candidate > 0)


def test_shared_pool_matched_budget_ledger() -> None:
    layers = [
        HashSharedSparseHinge(
            784,
            784 if index < 4 else 47,
            tables=64,
            comparisons=6,
            pool_size=16,
            candidates_per_code=4,
            margin_fan_in=8,
            write_fan_out=32,
            selection_mode="hash",
            seed=index,
            fixed_zero_hash_threshold=True,
        )
        for index in range(5)
    ]
    assert sum(sum(p.numel() for p in layer.parameters()) for layer in layers) == 209_920
    first = layers[0]
    assert first.candidate_bank_size == 64 * 16
    assert first.active_margin_count == 64 * 4
    assert first.semantic_route_terms == 64 * 6
    assert first.semantic_action_terms == 64 * 4 * (8 + 32)
    assert first.support_index_count == 64 * 16 * (8 + 32)
    assert torch.equal(
        first.candidate_code_degrees,
        torch.full((64, 16), 16, dtype=torch.long),
    )


def test_shared_pool_rejects_unreachable_or_duplicate_candidate_budget() -> None:
    for pool_size, candidates in ((2, 3), (9, 1)):
        try:
            HashSharedSparseHinge(
                4,
                3,
                tables=1,
                comparisons=3,
                pool_size=pool_size,
                candidates_per_code=candidates,
                margin_fan_in=2,
                write_fan_out=2,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("expected invalid shared candidate budget to fail")


def _shared_classifier(
    selection_mode: str,
    candidates: int,
) -> PayloadWidthEmnistClassifier:
    return PayloadWidthEmnistClassifier(
        input_dim=28 * 28,
        classes=47,
        depth=4,
        tables=64,
        comparisons=6,
        variant="hash_shared_sparse_hinge",
        anchor_policy="permuted",
        seed=0,
        lut_init_std=0.0,
        write_degree=16,
        walsh_lut_dtype="int2",
        walsh_order=2,
        walsh_coeff_init_std=0.02,
        walsh_slope_order=2,
        walsh_slope_coeff_init_std=0.02,
        walsh_slope_generator_init_std=0.02,
        residual_scale=1.0,
        use_output_scaling=True,
        use_min_margin_ste=True,
        comparator_kc=48,
        comparator_write_policy="expander",
        comparator_reduction_layout="scatter",
        comparator_output_tile_size=32,
        comparator_weight_mode="float",
        ternary_threshold=0.5,
        hash_candidates=candidates,
        hash_pool_size=16,
        hash_selection_mode=selection_mode,  # type: ignore[arg-type]
        hash_margin_fan_in=8,
        hash_write_fan_out=32,
        hash_fixed_zero_threshold=True,
        compare_swap_alpha_init=0.0,
        compare_swap_pair_count=0,
        correction_gate_init=0.0,
        correction_kc=48,
        correction_init_std=0.02,
        route_affine_pair_count=0,
    )


def test_emnist_shared_pool_arms_are_parameter_and_initialization_matched() -> None:
    k4 = _shared_classifier("hash", 4)
    k8 = _shared_classifier("hash", 8)
    all_scan = _shared_classifier("all", 4)
    assert sum(p.numel() for p in k4.parameters()) == 209_920
    assert sum(p.numel() for p in k8.parameters()) == 209_920
    assert sum(p.numel() for p in all_scan.parameters()) == 209_920
    for layer_index in range(5):
        layers = [model.payload_layers()[layer_index] for model in (k4, k8, all_scan)]
        for name in (
            "anchors",
            "candidate_ranking",
            "read_indices",
            "write_indices",
            "read_weight",
            "margin_thresholds",
            "write_weight",
        ):
            assert torch.equal(getattr(layers[0], name), getattr(layers[1], name))
            assert torch.equal(getattr(layers[0], name), getattr(layers[2], name))
