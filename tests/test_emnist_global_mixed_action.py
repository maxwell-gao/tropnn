from __future__ import annotations

import math

import torch
from torch import nn

from tropnn.tools.emnist_global_mixed_action import (
    EmnistGlobalMixedActionClassifier,
    GlobalMixedActionButterfly,
    GlobalMixedActionResidualStack,
    _inverse_tanh_scaled,
)


def _set_effective(parameter: nn.Parameter, values: torch.Tensor, limit: float) -> None:
    with torch.no_grad():
        parameter.copy_(torch.atanh(values / limit))


def _manual_round(core: GlobalMixedActionButterfly, state: torch.Tensor, round_index: int) -> torch.Tensor:
    stride = 1 << round_index
    paired = state.reshape(state.shape[0], -1, 2, stride)
    u = paired[:, :, 0, :].reshape(state.shape[0], -1)
    v = paired[:, :, 1, :].reshape(state.shape[0], -1)
    a, b, d_u, d_v = core.effective_actions()
    q = (u - v - core.theta_q[round_index] >= 0).to(state.dtype)
    h = d_v[round_index] * v + (a[round_index, 0] + q * (a[round_index, 1] - a[round_index, 0])) * u
    s_input = h if core.arm == "live_reroute" else v
    s = (s_input - u - core.theta_s[round_index] >= 0).to(state.dtype)
    new_u = d_u[round_index] * u + (b[round_index, 0] + s * (b[round_index, 1] - b[round_index, 0])) * h
    blocks = core.carrier_dim // (2 * stride)
    return torch.stack(
        (new_u.reshape(state.shape[0], blocks, stride), h.reshape(state.shape[0], blocks, stride)), dim=2
    ).reshape_as(state)


def test_scalar_reference_matches_hard_forward() -> None:
    core = GlobalMixedActionButterfly(carrier_dim=8, rounds=3, arm="live_reroute", action_init=0.02)
    state = torch.tensor([[1.0, 0.9, -0.3, 0.7, 0.2, -0.8, 0.6, -0.1]])
    expected = state
    for round_index in range(3):
        expected = _manual_round(core, expected, round_index)
    actual = core(state)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_first_shear_can_change_second_route() -> None:
    live = GlobalMixedActionButterfly(carrier_dim=2, rounds=1, arm="live_reroute", action_init=0.0)
    pre = GlobalMixedActionButterfly(carrier_dim=2, rounds=1, arm="pre_shear_route", action_init=0.0)
    a = torch.tensor([[0.0, 0.2]])
    b = torch.tensor([[0.1, -0.1]])
    _set_effective(live.raw_a, a, live.action_limit)
    _set_effective(live.raw_b, b, live.action_limit)
    pre.load_state_dict(live.state_dict(), strict=True)
    state = torch.tensor([[1.0, 0.9]])
    live_output = live(state)
    pre_output = pre(state)
    assert not torch.equal(live_output, pre_output)
    trace = live.trace(state)[0]
    assert trace["live_vs_pre_reroute_fraction"] == 1.0
    assert trace["forced_q_sensitivity_fraction"] == 1.0
    assert trace["mixed_amplitude"] > 0.0


def test_hard_forward_has_finite_nonzero_ste_gradients() -> None:
    core = GlobalMixedActionButterfly(carrier_dim=4, rounds=2, arm="live_reroute", action_init=0.02)
    state = torch.tensor(
        [[0.3, 0.2, -0.1, -0.2], [0.1, 0.2, 0.4, 0.3], [-0.2, -0.1, 0.2, 0.1]], requires_grad=True
    )
    loss = core(state).square().mean()
    loss.backward()
    for parameter in (core.raw_a, core.raw_b, core.raw_d, core.theta_q, core.theta_s):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    assert state.grad is not None
    assert torch.isfinite(state.grad).all()


def test_full_butterfly_ledger_and_no_dense_matrix() -> None:
    core = GlobalMixedActionButterfly(carrier_dim=1024, rounds=10, arm="live_reroute")
    ledger = core.ledger()
    assert ledger.trainable_parameters == 10_300
    assert sum(parameter.numel() for parameter in core.parameters()) == 10_300
    assert ledger.comparisons_per_example == 10_240
    assert ledger.semantic_macs_per_example == 20_480
    assert ledger.coordinate_writes_per_example == 10_240
    assert ledger.receptive_field == 1024
    assert not any(isinstance(module, nn.Linear) for module in core.modules())
    assert not any(parameter.ndim == 2 and parameter.shape == (1024, 1024) for parameter in core.parameters())


def test_four_block_stack_is_direct_serial_residual_composition() -> None:
    stack = GlobalMixedActionResidualStack(
        depth=4,
        carrier_dim=8,
        rounds=3,
        arm="pre_shear_route",
        tau=0.1,
        action_limit=0.25,
        dissipation_span=0.25,
        action_init=0.02,
    )
    state = torch.randn(7, 8)
    expected = state
    for block in stack.blocks:
        expected = block(expected)
    actual = stack(state)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert len({block.raw_a.data_ptr() for block in stack.blocks}) == 4
    ledger = stack.ledger()
    assert ledger.depth == 4
    assert ledger.stages == 12
    assert ledger.trainable_parameters == 4 * 42
    assert ledger.comparisons_per_example == 4 * 24
    assert ledger.semantic_macs_per_example == 4 * 48
    assert ledger.coordinate_writes_per_example == 4 * 24
    assert ledger.receptive_field == 8


def test_depth_one_stack_is_bit_exact_single_block() -> None:
    stack = GlobalMixedActionResidualStack(
        depth=1,
        carrier_dim=8,
        rounds=3,
        arm="pre_shear_route",
        tau=0.1,
        action_limit=0.25,
        dissipation_span=0.25,
        action_init=0.02,
    )
    state = torch.randn(5, 8)
    torch.testing.assert_close(stack(state), stack.blocks[0](state), rtol=0.0, atol=0.0)


def test_four_block_emnist_ledger_is_exact() -> None:
    stack = GlobalMixedActionResidualStack(
        depth=4,
        carrier_dim=1024,
        rounds=10,
        arm="pre_shear_route",
        tau=0.1,
        action_limit=0.25,
        dissipation_span=0.25,
        action_init=0.02,
    )
    ledger = stack.ledger()
    assert ledger.trainable_parameters == 41_200
    assert ledger.comparisons_per_example == 40_960
    assert ledger.semantic_macs_per_example == 81_920
    assert ledger.coordinate_writes_per_example == 40_960
    assert ledger.receptive_field == 1024


def test_four_block_trace_and_gradients_are_live() -> None:
    stack = GlobalMixedActionResidualStack(
        depth=4,
        carrier_dim=8,
        rounds=3,
        arm="pre_shear_route",
        tau=0.1,
        action_limit=0.25,
        dissipation_span=0.25,
        action_init=0.02,
    )
    state = torch.randn(11, 8, requires_grad=True)
    loss = stack(state).square().mean()
    loss.backward()
    assert all(math.isfinite(value) and value > 0.0 for value in stack.block_grad_norms())
    trace = stack.trace(state.detach())
    assert len(trace) == 12
    assert [int(row["stage"]) for row in trace] == list(range(12))
    assert {int(row["block"]) for row in trace} == {0, 1, 2, 3}
    assert all(math.isfinite(float(row["block_output_over_input_rms"])) for row in trace)
    boundary_rows = [row for row in trace if int(row["block"]) > 0]
    assert all(math.isfinite(float(row["upstream_skip_route_change_fraction"])) for row in boundary_rows)


def test_three_arms_have_identical_head_initialization_and_zero_logits() -> None:
    common = dict(
        input_dim=8,
        carrier_dim=8,
        classes=3,
        seed=7,
        core_depth=1,
        rounds=3,
        tau=0.1,
        action_limit=0.25,
        dissipation_span=0.25,
        action_init=0.02,
        tables=4,
        comparisons=2,
    )
    models = [
        EmnistGlobalMixedActionClassifier(arm=arm, **common)
        for arm in ("readout_only", "pre_shear_route", "live_reroute")
    ]
    assert len({model.head_initial_hash() for model in models}) == 1
    assert len({model.core.initial_hash() for model in models}) == 1
    x = torch.randn(5, 8)
    logits = [model(x) for model in models]
    assert torch.equal(logits[0], logits[1])
    assert torch.equal(logits[1], logits[2])
    assert torch.count_nonzero(logits[0]) == 0


def test_depth_one_and_four_share_head_init_and_zero_initial_logits() -> None:
    common = dict(
        input_dim=8,
        carrier_dim=8,
        classes=3,
        arm="pre_shear_route",
        seed=17,
        rounds=3,
        tau=0.1,
        action_limit=0.25,
        dissipation_span=0.25,
        action_init=0.02,
        tables=4,
        comparisons=2,
    )
    depth_one = EmnistGlobalMixedActionClassifier(core_depth=1, **common)
    depth_four = EmnistGlobalMixedActionClassifier(core_depth=4, **common)
    assert depth_one.head_initial_hash() == depth_four.head_initial_hash()
    assert len(set(depth_four.core.block_initial_hashes())) == 1
    assert len({block.raw_a.data_ptr() for block in depth_four.core.blocks}) == 4
    x = torch.randn(5, 8)
    logits_one = depth_one(x)
    logits_four = depth_four(x)
    assert torch.count_nonzero(logits_one) == 0
    assert torch.count_nonzero(logits_four) == 0


def test_initial_action_values_are_branch_distinct_and_small() -> None:
    core = GlobalMixedActionButterfly(carrier_dim=8, rounds=3, action_init=0.02)
    a, b, d_u, d_v = core.effective_actions()
    torch.testing.assert_close(a[:, 0], torch.full((3,), -0.02))
    torch.testing.assert_close(a[:, 1], torch.full((3,), 0.02))
    torch.testing.assert_close(b[:, 0], torch.full((3,), 0.02))
    torch.testing.assert_close(b[:, 1], torch.full((3,), -0.02))
    torch.testing.assert_close(d_u, torch.ones(3))
    torch.testing.assert_close(d_v, torch.ones(3))
    assert math.isclose(_inverse_tanh_scaled(0.02, 0.25), math.atanh(0.08))
