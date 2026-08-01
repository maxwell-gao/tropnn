from __future__ import annotations

import math

import torch

from tropnn import TernaryMarginAction


def _set_codes(parameter: torch.nn.Parameter, codes: torch.Tensor) -> None:
    magnitude = math.atanh(0.75)
    with torch.no_grad():
        parameter.copy_(codes.to(parameter) * magnitude)


def test_ternary_margin_action_matches_explicit_two_sided_reference() -> None:
    layer = TernaryMarginAction(5, 3, atoms=2, fan_in=2, mode="two_sided", seed=3, use_output_scaling=False)
    with torch.no_grad():
        layer.support_indices.copy_(torch.tensor([[0, 2], [1, 4]]))
        layer.thresholds.copy_(torch.tensor([0.5, -0.25]))
    _set_codes(layer.input_master, torch.tensor([[1, -1], [-1, 1]]))
    _set_codes(
        layer.direction_master,
        torch.tensor(
            [
                [[1, 0, -1], [-1, 1, 0]],
                [[0, 1, 1], [1, 0, -1]],
            ]
        ),
    )
    x = torch.tensor([[2.0, 1.0, 0.0, 4.0, -1.0], [0.0, -2.0, 3.0, 1.0, 2.0]])

    output, indices = layer.compute(x)

    input_codes = layer.hard_input_codes().float()
    direction_codes = layer.hard_direction_codes().float()
    selected = x[:, layer.support_indices]
    margins = (selected * input_codes).sum(dim=-1) - layer.thresholds
    expected = torch.relu(margins) @ direction_codes[:, 0] + torch.relu(-margins) @ direction_codes[:, 1]
    assert torch.equal(indices, (margins > 0).long())
    assert torch.equal(output, expected)


def test_two_sided_opposite_directions_recover_linear_action() -> None:
    two_sided = TernaryMarginAction(7, 4, atoms=3, fan_in=3, mode="two_sided", seed=5, use_output_scaling=False)
    linear = TernaryMarginAction(7, 4, atoms=3, fan_in=3, mode="linear", seed=9, use_output_scaling=False)
    input_codes = torch.tensor([[1, -1, 1], [-1, 0, 1], [1, 1, -1]])
    directions = torch.tensor([[1, 0, -1, 1], [0, -1, 1, 0], [-1, 1, 0, 1]])
    with torch.no_grad():
        linear.support_indices.copy_(two_sided.support_indices)
        linear.thresholds.copy_(two_sided.thresholds)
    _set_codes(two_sided.input_master, input_codes)
    _set_codes(linear.input_master, input_codes)
    _set_codes(two_sided.direction_master, torch.stack((directions, -directions), dim=1))
    _set_codes(linear.direction_master, directions.unsqueeze(1))
    x = torch.randn(11, 7)

    assert torch.allclose(two_sided(x), linear(x))


def test_ternary_margin_action_has_hard_codes_and_trainable_route_action_alignment() -> None:
    layer = TernaryMarginAction(8, 5, atoms=4, fan_in=3, mode="two_sided", seed=7)
    x = torch.randn(6, 8, requires_grad=True)
    target = torch.randn(6, 5)

    loss = (layer(x) - target).square().mean()
    loss.backward()

    assert set(layer.hard_input_codes().unique().tolist()) <= {-1, 0, 1}
    assert set(layer.hard_direction_codes().unique().tolist()) <= {-1, 0, 1}
    assert x.grad is not None and x.grad.abs().sum() > 0
    assert layer.input_master.grad is not None and layer.input_master.grad.abs().sum() > 0
    assert layer.direction_master.grad is not None and layer.direction_master.grad.abs().sum() > 0
    assert layer.thresholds.grad is not None and layer.thresholds.grad.abs().sum() > 0


def test_support_schedule_is_balanced_and_has_unique_rows() -> None:
    layer = TernaryMarginAction(13, 4, atoms=11, fan_in=5, seed=11)
    counts = torch.bincount(layer.support_indices.reshape(-1), minlength=layer.input_dim)

    assert int(counts.max() - counts.min()) <= 1
    for row in layer.support_indices:
        assert row.unique().numel() == layer.fan_in
    assert layer.semantic_route_terms == layer.atoms * layer.fan_in
    assert layer.semantic_action_terms == layer.atoms * layer.output_dim
