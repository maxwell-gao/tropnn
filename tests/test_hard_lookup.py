from __future__ import annotations

import inspect

import pytest
import torch
from tropnn.layers.hard_lookup import (
    HardLookupRouter,
    HardLookupSpec,
    ProductGridLookupRouter,
    adaptive_hard_route_lookahead,
    adaptive_leaf_probabilities,
    flat_leaf_probabilities,
    forced_neighbor_codes,
    hard_route,
    pack_branches,
    weighted_neighbor_delta,
)


def _paired_supports(tables: int, count: int, dim: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    first = torch.randint(dim, (tables, count), generator=generator)
    second = (first + torch.randint(1, dim, (tables, count), generator=generator)).remainder(dim)
    return torch.stack((first, second), dim=-1)


@pytest.mark.parametrize("predicate", ["pair", "unary"])
def test_flat_and_level_adaptive_routes_match_with_repeated_thresholds(predicate: str) -> None:
    depth, tables, dim = 4, 5, 9
    generator = torch.Generator().manual_seed(4)
    x = torch.randn(3, 7, dim, generator=generator)
    supports = _paired_supports(tables, depth, dim, seed=5)
    flat_thresholds = torch.randn(tables, depth, generator=generator)
    tree_thresholds = torch.cat(
        [flat_thresholds[:, level : level + 1].expand(-1, 2**level) for level in range(depth)],
        dim=1,
    )
    flat_spec = HardLookupSpec(dim, 3, depth, predicate, "flat")  # type: ignore[arg-type]
    tree_spec = HardLookupSpec(dim, 3, depth, predicate, "adaptive")  # type: ignore[arg-type]
    flat = hard_route(x, supports, flat_thresholds, flat_spec)
    tree = hard_route(x, supports, tree_thresholds, tree_spec)
    assert torch.equal(flat.codes, tree.codes)
    assert torch.equal(flat.branches, tree.branches)
    assert torch.equal(flat.margins, tree.margins)


def test_node_specific_adaptive_supports_follow_the_executed_path() -> None:
    depth, dim = 3, 8
    x = torch.tensor([[0.8, -0.7, 0.6, -0.5, 0.4, -0.3, 0.2, -0.1]])
    supports = _paired_supports(1, 2**depth - 1, dim, seed=8)
    thresholds = torch.linspace(-0.3, 0.3, 2**depth - 1).view(1, -1)
    spec = HardLookupSpec(dim, 2, depth, "pair", "adaptive", support_layout="node")
    route = hard_route(x, supports, thresholds, spec)

    code = 0
    branches = []
    for level in range(depth):
        node = 2**level - 1 + code
        left, right = supports[0, node]
        branch = bool(x[0, left] - x[0, right] - thresholds[0, node] >= 0)
        branches.append(branch)
        code = 2 * code + int(branch)
    assert route.codes.item() == code
    assert route.branches.flatten().tolist() == branches


def test_forced_adaptive_neighbor_reexecutes_the_suffix() -> None:
    depth, dim = 4, 7
    x = torch.randn(11, dim, generator=torch.Generator().manual_seed(9))
    supports = _paired_supports(3, depth, dim, seed=10)
    thresholds = torch.randn(3, 2**depth - 1, generator=torch.Generator().manual_seed(11))
    spec = HardLookupSpec(dim, 5, depth, "pair", "adaptive")
    route = hard_route(x, supports, thresholds, spec)
    forced_level = torch.randint(depth, route.codes.shape, generator=torch.Generator().manual_seed(12))
    neighbor = forced_neighbor_codes(x, supports, thresholds, spec, forced_level, route.branches)
    shift = depth - 1 - forced_level
    assert torch.equal((route.codes >> shift) & 1, 1 - ((neighbor >> shift) & 1))


@pytest.mark.parametrize("predicate", ["pair", "unary"])
@pytest.mark.parametrize("support_layout", ["level", "node"])
def test_adaptive_lookahead_executors_are_bit_exact(predicate: str, support_layout: str) -> None:
    depth, tables, dim = 4, 5, 11
    generator = torch.Generator().manual_seed(120)
    x = torch.randn(37, dim, generator=generator)
    count = depth if support_layout == "level" else 2**depth - 1
    supports = _paired_supports(tables, count, dim, seed=121)
    thresholds = torch.randn(tables, 2**depth - 1, generator=generator)
    spec = HardLookupSpec(
        dim,
        7,
        depth,
        predicate,  # type: ignore[arg-type]
        "adaptive",
        support_layout=support_layout,  # type: ignore[arg-type]
    )
    reference = hard_route(x, supports, thresholds, spec)
    for lookahead in (1, 2, 4):
        candidate = adaptive_hard_route_lookahead(x, supports, thresholds, spec, lookahead)
        assert torch.equal(candidate.codes, reference.codes)
        assert torch.equal(candidate.branches, reference.branches)
        assert torch.equal(candidate.margins, reference.margins)


def test_flat_leaf_probabilities_match_packed_hard_codes() -> None:
    bits = torch.tensor([[[0.0, 1.0, 1.0]], [[1.0, 0.0, 0.0]]])
    for order in ("msb", "lsb"):
        probabilities = flat_leaf_probabilities(bits, order)
        assert torch.equal(probabilities.argmax(-1), pack_branches(bits.bool(), order))
        assert torch.equal(probabilities.sum(-1), torch.ones(2, 1))


def test_adaptive_leaf_probabilities_match_path_numbering() -> None:
    bits = torch.zeros(2, 7)
    bits[0, 0] = 1
    bits[0, 2] = 1
    bits[0, 6] = 1
    probabilities = adaptive_leaf_probabilities(bits, depth=3)
    assert torch.equal(probabilities.argmax(-1), torch.tensor([7, 0]))
    assert torch.equal(probabilities.sum(-1), torch.ones(2))


@pytest.mark.parametrize("predicate", ["pair", "unary"])
@pytest.mark.parametrize("topology", ["flat", "adaptive"])
@pytest.mark.parametrize("surrogate", ["soft_product", "local_counterfactual"])
def test_all_four_router_families_have_exact_hard_forward_and_route_gradients(
    predicate: str,
    topology: str,
    surrogate: str,
) -> None:
    depth, tables, dim, output_dim = 3, 4, 8, 6
    support_count = depth
    supports = _paired_supports(tables, support_count, dim, seed=13)
    threshold_count = depth if topology == "flat" else 2**depth - 1
    thresholds = torch.zeros(tables, threshold_count)
    rows = torch.randn(tables, 2**depth, output_dim, generator=torch.Generator().manual_seed(14)) * 0.02
    layer = HardLookupRouter(
        dim,
        output_dim,
        depth=depth,
        predicate=predicate,  # type: ignore[arg-type]
        topology=topology,  # type: ignore[arg-type]
        supports=supports,
        thresholds=thresholds,
        rows=rows,
        surrogate=surrogate,  # type: ignore[arg-type]
    )
    x = torch.randn(17, dim, requires_grad=True)
    expected, codes = layer.hard_output(x)
    actual = layer(x)
    assert torch.equal(actual, expected)
    assert codes.shape == (17, tables)
    actual.square().mean().backward()
    assert x.grad is not None and float(x.grad.norm()) > 0
    assert layer.thresholds.grad is not None and float(layer.thresholds.grad.norm()) > 0
    assert layer.rows.grad is not None and float(layer.rows.grad.norm()) > 0


def test_diagonal_live_action_is_strictly_nested_at_zero_slope() -> None:
    depth, tables, dim = 3, 2, 6
    supports = _paired_supports(tables, 2**depth - 1, dim, seed=15)
    thresholds = torch.zeros(tables, 2**depth - 1)
    rows = torch.randn(tables, 2**depth, dim, generator=torch.Generator().manual_seed(16))
    common = dict(
        input_dim=dim,
        output_dim=dim,
        depth=depth,
        predicate="pair",
        topology="adaptive",
        support_layout="node",
        supports=supports,
        thresholds=thresholds,
        rows=rows,
        surrogate="soft_product",
    )
    constant = HardLookupRouter(action="constant", **common)
    live = HardLookupRouter(action="diagonal_live", slopes=torch.zeros_like(rows), **common)
    x = torch.randn(19, dim)
    assert torch.equal(constant(x), live(x))
    assert torch.equal(constant.hard_codes(x), live.hard_codes(x))


def test_product_grid_has_exact_mixed_radix_codes_and_route_gradients() -> None:
    supports = torch.tensor([[0, 1], [2, 3]])
    thresholds = torch.tensor(
        [
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0]],
            [[-0.5, 0.0, 0.5], [-1.5, 0.0, 1.5]],
        ]
    )
    rows = torch.randn(2, 16, 5, generator=torch.Generator().manual_seed(122))
    layer = ProductGridLookupRouter(4, 5, supports=supports, thresholds=thresholds, rows=rows)
    x = torch.tensor([[0.25, 2.5, -0.25, -2.0]], requires_grad=True)
    route = layer.route(x)
    assert route.digits.tolist() == [[[2, 3], [1, 0]]]
    assert route.codes.tolist() == [[11, 4]]
    hard, codes = layer.hard_output(x)
    assert torch.equal(layer(x), hard)
    assert torch.equal(codes, route.codes)
    assert torch.allclose(hard, rows[0, 11] + rows[1, 4], rtol=1e-6, atol=1e-6)
    layer(x).square().mean().backward()
    assert x.grad is not None and float(x.grad.norm()) > 0
    assert layer.thresholds.grad is not None and float(layer.thresholds.grad.norm()) > 0
    assert layer.rows.grad is not None and float(layer.rows.grad.norm()) > 0


def test_product_grid_counterfactual_always_selects_an_adjacent_digit() -> None:
    generator = torch.Generator().manual_seed(123)
    supports = torch.arange(8).view(4, 2)
    thresholds = torch.randn(4, 2, 3, generator=generator)
    rows = torch.randn(4, 16, 8, generator=generator)
    layer = ProductGridLookupRouter(8, 8, supports=supports, thresholds=thresholds, rows=rows)
    route = layer.route(torch.randn(29, 8, generator=generator))
    nearest = route.margins.flatten(start_dim=-2).abs().argmin(-1)
    neighbor = layer.neighboring_codes(route, nearest)
    delta = (neighbor - route.codes).abs()
    assert bool(((neighbor >= 0) & (neighbor < 16)).all())
    assert bool(((delta == 1) | (delta == 4)).all())


def test_weighted_neighbor_delta_is_exact_when_forced_to_chunk_tables() -> None:
    generator = torch.Generator().manual_seed(20)
    rows = torch.randn(7, 8, 5, generator=generator)
    current = torch.randint(8, (11, 7), generator=generator)
    neighbors = torch.randint(8, (11, 7, 3), generator=generator)
    weights = torch.randn(11, 7, 3, generator=generator)
    expected = (
        weights.unsqueeze(-1) * (rows[torch.arange(7)[None, :, None], neighbors] - rows[torch.arange(7)[None, :], current].unsqueeze(-2))
    ).sum(dim=(1, 2))
    actual = weighted_neighbor_delta(rows, current, neighbors, weights, target_bytes=1)
    assert torch.allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_learned_support_counterfactual_is_part_of_the_shared_core() -> None:
    depth, tables, dim, output_dim = 4, 3, 8, 5
    generator = torch.Generator().manual_seed(21)
    scores = torch.randn(tables, depth, dim, generator=generator) * 1e-4
    supports = scores.argmax(-1)
    layer = HardLookupRouter(
        dim,
        output_dim,
        depth=depth,
        predicate="unary",
        topology="adaptive",
        supports=supports,
        thresholds=torch.zeros(tables, 2**depth - 1),
        rows=torch.randn(tables, 2**depth, output_dim, generator=generator),
        surrogate="local_counterfactual",
        support_scores=scores,
        support_tau=1.0,
        trainable_supports=True,
    )
    x = torch.randn(64, dim, generator=generator, requires_grad=True)
    hard, codes, productive = layer.hard_output_with_support_counterfactual(x)
    output = layer(x)
    assert torch.equal(output, hard)
    assert codes.shape == productive.shape == (64, tables)
    output.square().mean().backward()
    assert layer.support_scores is not None
    assert layer.support_scores.grad is not None and float(layer.support_scores.grad.abs().sum()) > 0


def test_learned_support_counterfactual_has_no_credit_without_a_branch_flip() -> None:
    depth, tables, dim, output_dim = 3, 2, 6, 4
    scores = torch.arange(dim, dtype=torch.float32).view(1, 1, dim).expand(tables, depth, -1).clone()
    layer = HardLookupRouter(
        dim,
        output_dim,
        depth=depth,
        predicate="unary",
        topology="adaptive",
        supports=scores.argmax(-1),
        thresholds=torch.zeros(tables, 2**depth - 1),
        rows=torch.randn(tables, 2**depth, output_dim, generator=torch.Generator().manual_seed(22)),
        surrogate="none",
        support_scores=scores,
        trainable_supports=True,
    )
    output = layer(torch.ones(9, dim))
    output.square().mean().backward()
    assert layer.support_scores is not None and layer.support_scores.grad is not None
    assert torch.count_nonzero(layer.support_scores.grad) == 0


def test_experiment_modules_do_not_define_route_lookup_or_ste_cores() -> None:
    from tropnn.tools import (
        emnist_maddness_learned_indices,
        emnist_maddness_task_ste,
        emnist_router_dataflow_factorial,
        maddness_end_to_end_ste_factorial,
        random_linear_address_action_factorial,
        random_linear_multitable_address_action_factorial,
    )

    modules = (
        emnist_maddness_learned_indices,
        emnist_maddness_task_ste,
        emnist_router_dataflow_factorial,
        maddness_end_to_end_ste_factorial,
        random_linear_address_action_factorial,
        random_linear_multitable_address_action_factorial,
    )
    forbidden_functions = {
        "hard_codes",
        "_hard_codes",
        "forced_adaptive_codes",
        "_forced_codes",
        "_gather_sum",
        "flat_leaf_probabilities",
        "tree_leaf_probabilities",
        "straight_through_bit",
    }
    forbidden_methods = {
        "route",
        "hard_codes",
        "hard_output",
        "neighboring_codes",
        "leaf_probabilities",
        "local_counterfactual_output",
    }
    for module in modules:
        locally_defined_functions = {
            name for name, value in vars(module).items() if inspect.isfunction(value) and value.__module__ == module.__name__
        }
        assert locally_defined_functions.isdisjoint(forbidden_functions)
        for value in vars(module).values():
            if inspect.isclass(value) and value.__module__ == module.__name__:
                assert set(value.__dict__).isdisjoint(forbidden_methods)
