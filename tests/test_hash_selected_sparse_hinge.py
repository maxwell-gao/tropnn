from __future__ import annotations

import torch
from torch.nn import functional as F
from tropnn import HashSelectedSparseHinge


def test_hash_selected_sparse_hinge_supports_are_sparse_and_deterministic() -> None:
    first = HashSelectedSparseHinge(
        11,
        7,
        tables=2,
        comparisons=2,
        candidates=3,
        margin_fan_in=4,
        write_fan_out=3,
        seed=5,
    )
    second = HashSelectedSparseHinge(
        11,
        7,
        tables=2,
        comparisons=2,
        candidates=3,
        margin_fan_in=4,
        write_fan_out=3,
        seed=5,
    )

    assert torch.equal(first.anchors, second.anchors)
    assert torch.equal(first.read_indices, second.read_indices)
    assert torch.equal(first.write_indices, second.write_indices)
    assert first.read_indices.shape == (2, 4, 3, 4)
    assert first.write_indices.shape == (2, 4, 3, 3)

    read_sorted = first.read_indices.sort(dim=-1).values
    write_sorted = first.write_indices.sort(dim=-1).values
    assert torch.all(read_sorted[..., 1:] > read_sorted[..., :-1])
    assert torch.all(write_sorted[..., 1:] > write_sorted[..., :-1])


def test_hash_selected_sparse_hinge_keeps_live_amplitude_inside_one_hash_cell() -> None:
    layer = HashSelectedSparseHinge(
        6,
        5,
        tables=2,
        comparisons=2,
        candidates=2,
        margin_fan_in=3,
        write_fan_out=2,
        seed=3,
        use_output_scaling=False,
    )
    layer.eval()
    with torch.no_grad():
        layer.read_weight.fill_(1.0 / 3.0)
        layer.margin_thresholds.fill_(-0.25)
        layer.write_weight.fill_(1.0)

    x = torch.zeros(1, 6, requires_grad=True)
    shifted = torch.full((1, 6), 0.5)
    y, route = layer.compute(x)
    shifted_y, shifted_route = layer.compute(shifted)

    # The pair-difference hash is unchanged by a common shift, but the selected
    # learned margins still read and transport the current amplitude.
    assert torch.equal(route.indices, shifted_route.indices)
    assert not torch.allclose(y.detach(), shifted_y)
    y.sum().backward()
    assert x.grad is not None
    assert float(x.grad.abs().sum()) > 0.0


def test_hash_code_selects_different_sparse_hinge_programs() -> None:
    layer = HashSelectedSparseHinge(
        3,
        1,
        tables=1,
        comparisons=1,
        candidates=1,
        margin_fan_in=1,
        write_fan_out=1,
        seed=7,
        use_output_scaling=False,
    )
    layer.eval()
    with torch.no_grad():
        layer.anchors[0, 0] = torch.tensor([0, 1])
        layer.read_weight.zero_()
        layer.margin_thresholds.fill_(-1.0)
        layer.write_weight[0, 0].fill_(1.0)
        layer.write_weight[0, 1].fill_(3.0)

    low, low_route = layer.compute(torch.tensor([[0.0, 1.0, 0.0]]))
    high, high_route = layer.compute(torch.tensor([[1.0, 0.0, 0.0]]))
    assert int(low_route.indices.item()) == 0
    assert int(high_route.indices.item()) == 1
    assert torch.allclose(low, torch.tensor([[1.0]]))
    assert torch.allclose(high, torch.tensor([[3.0]]))


def test_hash_route_ste_credits_neighbor_program_difference() -> None:
    layer = HashSelectedSparseHinge(
        3,
        1,
        tables=1,
        comparisons=1,
        candidates=1,
        margin_fan_in=1,
        write_fan_out=1,
        seed=7,
        use_output_scaling=False,
    )
    layer.train()
    with torch.no_grad():
        layer.anchors[0, 0] = torch.tensor([0, 1])
        layer.read_weight.zero_()
        layer.margin_thresholds.fill_(-1.0)
        layer.write_weight[0, 0].fill_(1.0)
        layer.write_weight[0, 1].fill_(3.0)

    output = layer(torch.tensor([[0.1, 0.0, 0.0]], requires_grad=True))
    output.sum().backward()
    assert layer.thresholds.grad is not None
    assert float(layer.thresholds.grad.abs().sum()) > 0.0


def test_hash_selected_sparse_hinge_parameter_ledger() -> None:
    layer = HashSelectedSparseHinge(
        10,
        7,
        tables=3,
        comparisons=2,
        candidates=2,
        margin_fan_in=4,
        write_fan_out=3,
        seed=1,
        fixed_zero_hash_threshold=True,
    )
    bank = 3 * 4 * 2
    expected_program_params = bank * (4 + 1 + 3)
    assert layer.candidate_bank_size == bank
    assert layer.active_margin_count == 3 * 2
    assert layer.semantic_route_terms == 3 * 2
    assert layer.semantic_action_terms == 3 * 2 * (4 + 3)
    assert layer.support_index_count == bank * (4 + 3)
    assert layer.payload_params == expected_program_params
    assert sum(parameter.numel() for parameter in layer.parameters()) == expected_program_params
    assert not isinstance(layer.thresholds, torch.nn.Parameter)


def test_hash_selected_sparse_hinge_matches_explicit_program_and_gradients() -> None:
    layer = HashSelectedSparseHinge(
        4,
        3,
        tables=1,
        comparisons=1,
        candidates=2,
        margin_fan_in=2,
        write_fan_out=2,
        seed=11,
        use_output_scaling=False,
        fixed_zero_hash_threshold=True,
    ).double()
    layer.eval()
    with torch.no_grad():
        layer.anchors[0, 0] = torch.tensor([0, 1])
        layer.read_indices[0, 1] = torch.tensor([[0, 2], [1, 3]])
        layer.write_indices[0, 1] = torch.tensor([[0, 1], [1, 2]])
        layer.read_weight[0, 1] = torch.tensor(
            [[0.5, -1.0], [1.5, 0.25]], dtype=torch.double
        )
        layer.margin_thresholds[0, 1] = torch.tensor(
            [-0.2, 0.1], dtype=torch.double
        )
        layer.write_weight[0, 1] = torch.tensor(
            [[2.0, -0.5], [1.0, 3.0]], dtype=torch.double
        )

    x = torch.tensor([[1.0, 0.0, -0.5, 2.0]], dtype=torch.double, requires_grad=True)
    actual = layer(x)

    read_0 = 0.5 * x[:, 0] - x[:, 2] + 0.2
    read_1 = 1.5 * x[:, 1] + 0.25 * x[:, 3] - 0.1
    expected = torch.zeros(1, 3, dtype=torch.double)
    expected[:, 0] = 2.0 * F.relu(read_0)
    expected[:, 1] = -0.5 * F.relu(read_0) + F.relu(read_1)
    expected[:, 2] = 3.0 * F.relu(read_1)
    assert torch.allclose(actual, expected)

    actual.square().sum().backward()
    assert x.grad is not None and float(x.grad.abs().sum()) > 0.0
    assert layer.read_weight.grad is not None
    assert layer.margin_thresholds.grad is not None
    assert layer.write_weight.grad is not None
    assert float(layer.read_weight.grad[0, 1].abs().sum()) > 0.0
    assert float(layer.margin_thresholds.grad[0, 1].abs().sum()) > 0.0
    assert float(layer.write_weight.grad[0, 1].abs().sum()) > 0.0
    assert torch.equal(layer.read_weight.grad[0, 0], torch.zeros_like(layer.read_weight.grad[0, 0]))


def test_hash_selected_sparse_hinge_rejects_unsupported_full_ste() -> None:
    try:
        HashSelectedSparseHinge(
            4,
            3,
            margin_fan_in=4,
            write_fan_out=3,
            use_min_margin_ste=False,
        )
    except ValueError as error:
        assert "min-margin route STE only" in str(error)
    else:
        raise AssertionError("expected unsupported full route STE to fail at construction")
