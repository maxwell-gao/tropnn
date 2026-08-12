from __future__ import annotations

import math

import torch
from torch import nn
from tropnn.layers.quantized_comparison_affine import (
    QuantizedComparisonAffineStack,
    QuantizedComparisonAffineSweep,
    QuantizedConditionalAffineAssignment,
    _dyadic_scale_ste,
    _hard_tanh_ste,
    _ternary_ste,
)
from tropnn.tools.emnist_quantized_comparison_affine import (
    EmnistQuantizedComparisonAffineClassifier,
    parse_args,
)


def _set_codes(layer: QuantizedComparisonAffineSweep, codes: torch.Tensor) -> None:
    with torch.no_grad():
        layer.coefficient_master.copy_(codes.to(torch.float32) * math.atanh(0.75))


def _set_unit_scale(layer: QuantizedComparisonAffineSweep) -> None:
    with torch.no_grad():
        layer.log2_scale_master.zero_()


def test_continuous_sweep_matches_scalar_reference() -> None:
    layer = QuantizedComparisonAffineSweep(carrier_dim=4, rounds=2, mode="continuous")
    state = torch.tensor([[0.7, -0.2, 0.4, -0.6]])
    codes = torch.zeros(2, 2, 6)
    codes[:, 0, 3] = -1.0
    codes[:, 1, 3] = 1.0
    _set_codes(layer, codes)
    with torch.no_grad():
        layer.log2_scale_master.fill_(-2.0)

    expected = state
    for round_index in range(2):
        stride = 1 << round_index
        paired = expected.reshape(1, -1, 2, stride)
        u = paired[:, :, 0, :].reshape(1, -1)
        v = paired[:, :, 1, :].reshape(1, -1)
        margin = u - v
        hinge = torch.relu(margin)
        new_u = u - 0.25 * hinge
        new_v = v + 0.25 * hinge
        expected = torch.stack((new_u.reshape(1, -1, stride), new_v.reshape(1, -1, stride)), dim=2).reshape_as(expected)

    torch.testing.assert_close(layer(state), expected, rtol=0.0, atol=0.0)


def test_continuous_wall_is_exactly_matched_and_free_jump_is_explicit() -> None:
    continuous = QuantizedComparisonAffineSweep(carrier_dim=2, rounds=1, mode="continuous")
    free = QuantizedComparisonAffineSweep(carrier_dim=2, rounds=1, mode="free")
    _set_unit_scale(continuous)
    _set_unit_scale(free)
    codes = torch.zeros(1, 2, 6)
    codes[0, 0, 3] = -1.0
    codes[0, 1, 3] = 1.0
    codes[0, 0, 4] = 1.0
    codes[0, 0, 5] = -1.0
    codes[0, 1, 4] = -1.0
    codes[0, 1, 5] = 1.0
    _set_codes(continuous, codes)
    _set_codes(free, codes)
    with torch.no_grad():
        continuous.thresholds.fill_(0.25)
        free.thresholds.fill_(0.25)

    wall = torch.tensor([[0.75, 0.50]])
    continuous_delta = continuous.forced_branch_delta(wall, 0)
    torch.testing.assert_close(continuous_delta[0], torch.zeros_like(continuous_delta[0]), rtol=0.0, atol=0.0)
    torch.testing.assert_close(continuous_delta[1], torch.zeros_like(continuous_delta[1]), rtol=0.0, atol=0.0)

    free_delta = free.forced_branch_delta(wall, 0)
    # At the wall the hinge contribution vanishes.  Only kappa*self+zeta remains.
    torch.testing.assert_close(free_delta[0], torch.tensor([[-0.25]]), rtol=0.0, atol=0.0)
    torch.testing.assert_close(free_delta[1], torch.tensor([[0.50]]), rtol=0.0, atol=0.0)


def test_continuous_special_case_is_exact_compare_exchange() -> None:
    layer = QuantizedComparisonAffineSweep(
        carrier_dim=2,
        rounds=1,
        mode="continuous",
        initial_scale_exponent=0,
    )
    codes = torch.zeros(1, 2, 6)
    codes[0, 0, 3] = -1.0
    codes[0, 1, 3] = 1.0
    _set_codes(layer, codes)
    _set_unit_scale(layer)
    state = torch.tensor([[3.0, 1.0], [-2.0, 4.0], [1.0, 1.0]])
    expected = torch.tensor([[1.0, 3.0], [-2.0, 4.0], [1.0, 1.0]])
    torch.testing.assert_close(layer(state), expected, rtol=0.0, atol=0.0)


def test_constant_corner_is_piecewise_constant_residual_update() -> None:
    layer = QuantizedComparisonAffineSweep(
        carrier_dim=2,
        rounds=1,
        mode="constant",
        initial_scale_exponent=0,
    )
    _set_unit_scale(layer)
    state = torch.tensor([[3.0, 1.0], [-2.0, 4.0]])
    expected = torch.tensor([[2.0, 2.0], [-2.0, 4.0]])
    torch.testing.assert_close(layer(state), expected, rtol=0.0, atol=0.0)
    assert layer.ledger().full_width_payload_scalars == 0


def test_branch_independent_affine_erase_and_translation_are_representable() -> None:
    layer = QuantizedComparisonAffineSweep(
        carrier_dim=2,
        rounds=1,
        mode="free",
        initial_scale_exponent=0,
    )
    codes = torch.zeros(1, 2, 6)
    # u' = 0; v' = v + u + 1, independent of the route.
    codes[0, 0, 0] = -1.0
    codes[0, 1, 1] = 1.0
    codes[0, 1, 2] = 1.0
    _set_codes(layer, codes)
    _set_unit_scale(layer)
    state = torch.tensor([[2.0, 3.0], [-4.0, 1.0]])
    expected = torch.tensor([[0.0, 6.0], [0.0, -2.0]])
    torch.testing.assert_close(layer(state), expected, rtol=0.0, atol=0.0)


def test_continuous_and_free_initial_states_are_bit_exact() -> None:
    continuous = QuantizedComparisonAffineStack(depth=4, carrier_dim=8, rounds=3, mode="continuous")
    free = QuantizedComparisonAffineStack(depth=4, carrier_dim=8, rounds=3, mode="free")
    assert continuous.initial_hash() == free.initial_hash()
    state = torch.randn(11, 8)
    torch.testing.assert_close(continuous(state), free(state), rtol=0.0, atol=0.0)


def test_hard_codes_scales_and_gradients_are_live_and_finite() -> None:
    layer = QuantizedComparisonAffineSweep(carrier_dim=8, rounds=3, mode="free")
    codes = layer.hard_coefficient_codes()
    assert set(codes.unique().tolist()) <= {-1, 0, 1}
    coefficients, scales = layer.effective_coefficients()
    assert set(scales.detach().tolist()) == {2.0**-4}
    torch.testing.assert_close(
        coefficients.detach(),
        codes.to(coefficients.dtype) * scales[:, None, None],
        rtol=0.0,
        atol=0.0,
    )
    state = torch.randn(17, 8, requires_grad=True)
    loss = layer(state).square().mean()
    loss.backward()
    for parameter in layer.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    assert layer.thresholds.grad is not None and layer.thresholds.grad.abs().sum() > 0
    # Free jump masters start at hard zero but receive a first-order signal.
    assert layer.coefficient_master.grad is not None
    assert layer.coefficient_master.grad[:, :, 4:].abs().sum() > 0


def test_l4_emnist_ledger_and_no_dense_payload_are_exact() -> None:
    continuous = QuantizedComparisonAffineStack(depth=4, carrier_dim=1024, rounds=10, mode="continuous")
    free = QuantizedComparisonAffineStack(depth=4, carrier_dim=1024, rounds=10, mode="free")
    constant = QuantizedComparisonAffineStack(depth=4, carrier_dim=1024, rounds=10, mode="constant")
    assert continuous.ledger().stored_parameters == 21_000
    assert continuous.ledger().effective_parameters == 20_840
    assert free.ledger().stored_parameters == 21_000
    assert free.ledger().effective_parameters == 21_000
    assert constant.ledger().effective_parameters == 20_680
    assert continuous.ledger().comparisons_per_example == 20_480
    assert continuous.ledger().coordinate_writes_per_example == 40_960
    assert continuous.ledger().receptive_field == 1024
    assert continuous.ledger().full_width_payload_scalars == 0
    assert sum(parameter.numel() for parameter in continuous.parameters()) == 21_000
    assert not any(isinstance(module, nn.Linear) for module in continuous.modules())
    assert not any(parameter.ndim >= 2 and parameter.shape[-1] == 1024 for parameter in continuous.parameters())


def test_trace_reports_zero_continuous_jump_and_nonzero_free_jump() -> None:
    continuous = QuantizedComparisonAffineSweep(carrier_dim=8, rounds=3, mode="continuous")
    free = QuantizedComparisonAffineSweep(carrier_dim=8, rounds=3, mode="free")
    codes = free.hard_coefficient_codes().to(torch.float32)
    codes[:, 0, 5] = -1.0
    codes[:, 1, 5] = 1.0
    _set_codes(free, codes)
    state = torch.randn(31, 8)
    assert all(row["wall_jump_rms"] == 0.0 for row in continuous.trace(state))
    assert all(row["wall_jump_rms"] > 0.0 for row in free.trace(state))


def test_continuous_masked_jump_masters_have_zero_gradient_and_do_not_move() -> None:
    layer = QuantizedComparisonAffineSweep(carrier_dim=8, rounds=3, mode="continuous")
    initial = layer.coefficient_master.detach().clone()
    optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-2, weight_decay=0.0)
    state = torch.randn(19, 8)
    optimizer.zero_grad(set_to_none=True)
    layer(state).square().mean().backward()
    assert layer.coefficient_master.grad is not None
    assert torch.count_nonzero(layer.coefficient_master.grad[:, :, 4:]) == 0
    optimizer.step()
    torch.testing.assert_close(layer.coefficient_master[:, :, 4:], initial[:, :, 4:], rtol=0.0, atol=0.0)


def test_emnist_classifier_uses_common_head_without_hidden_vector_payload() -> None:
    kwargs = {
        "input_dim": 8,
        "carrier_dim": 8,
        "classes": 3,
        "seed": 7,
        "core_depth": 2,
        "rounds": 3,
        "tau": 0.1,
        "ternary_threshold": 0.5,
        "initial_scale_exponent": -4,
        "minimum_scale_exponent": -8,
        "maximum_scale_exponent": 0,
        "tables": 4,
        "comparisons": 2,
    }
    continuous = EmnistQuantizedComparisonAffineClassifier(mode="continuous", **kwargs)
    free = EmnistQuantizedComparisonAffineClassifier(mode="free", **kwargs)
    assert continuous.head_initial_hash() == free.head_initial_hash()
    assert continuous.core.initial_hash() == free.core.initial_hash()
    images = torch.randn(5, 8)
    torch.testing.assert_close(continuous(images), free(images), rtol=0.0, atol=0.0)
    assert continuous(images).shape == (5, 3)
    assert continuous.core.ledger().full_width_payload_scalars == 0


def test_scale_cap_preserves_initial_forward_and_bounds_all_effective_scales() -> None:
    old = QuantizedComparisonAffineStack(
        depth=2,
        carrier_dim=8,
        rounds=3,
        mode="continuous",
        initial_scale_exponent=-4,
        minimum_scale_exponent=-8,
        maximum_scale_exponent=0,
    )
    capped = QuantizedComparisonAffineStack(
        depth=2,
        carrier_dim=8,
        rounds=3,
        mode="continuous",
        initial_scale_exponent=-4,
        minimum_scale_exponent=-8,
        maximum_scale_exponent=-3,
    )
    assert old.initial_hash() == capped.initial_hash()
    state = torch.randn(37, 8)
    assert torch.equal(old(state), capped(state))

    with torch.no_grad():
        for block in capped.blocks:
            block.log2_scale_master.copy_(torch.tensor([-20.0, -2.6, 7.0]))
    for block in capped.blocks:
        _coefficients, scales = block.effective_coefficients()
        assert bool(torch.all(scales <= 2.0**-3))
        assert bool(torch.all(scales >= 2.0**-8))
        assert all(float(value).is_integer() for value in torch.log2(scales).tolist())


def test_trace_separates_centered_common_and_branch_spectral_gain() -> None:
    layer = QuantizedComparisonAffineStack(
        depth=1,
        carrier_dim=8,
        rounds=3,
        mode="continuous",
        maximum_scale_exponent=-3,
    )
    rows = layer.trace(torch.randn(31, 8))
    assert len(rows) == 3
    for row in rows:
        assert float(row["state_out_centered_rms"]) >= 0.0
        assert float(row["state_out_common_rms"]) >= 0.0
        assert float(row["hard_branch_sigma_max"]) >= float(row["hard_branch_sigma_min"]) > 0.0
        assert float(row["hard_branch_abs_det_min"]) > 0.0
        assert float(row["scale"]) <= 2.0**-3
        assert int(row["scale_at_upper_bound"]) in {0, 1}


def test_hard_branch_matrices_match_autograd_away_from_the_wall() -> None:
    layer = QuantizedComparisonAffineSweep(
        carrier_dim=2,
        rounds=1,
        mode="free",
        initial_scale_exponent=-2,
    )
    codes = torch.tensor([[[1.0, -1.0, 1.0, 1.0, -1.0, 1.0], [-1.0, 1.0, -1.0, -1.0, 1.0, -1.0]]])
    _set_codes(layer, codes)
    with torch.no_grad():
        layer.thresholds.fill_(0.125)
    matrix_zero, matrix_one = layer.hard_branch_matrices(0)

    def hard_round(state: torch.Tensor) -> torch.Tensor:
        return layer._round(state.unsqueeze(0), 0, hard_only=True)[0].squeeze(0)

    state_zero = torch.tensor([-1.0, 1.0], requires_grad=True)
    state_one = torch.tensor([1.0, -1.0], requires_grad=True)
    jacobian_zero = torch.autograd.functional.jacobian(hard_round, state_zero)
    jacobian_one = torch.autograd.functional.jacobian(hard_round, state_one)
    torch.testing.assert_close(jacobian_zero, matrix_zero, rtol=0.0, atol=0.0)
    torch.testing.assert_close(jacobian_one, matrix_one, rtol=0.0, atol=0.0)


def test_emnist_scale_cap_v2_cli_defaults_and_rejects_invalid_bounds(tmp_path) -> None:
    args = parse_args(
        [
            "--mode",
            "continuous",
            "--root",
            str(tmp_path),
            "--out-dir",
            str(tmp_path / "out"),
        ]
    )
    assert args.initial_scale_exponent == -4
    assert args.minimum_scale_exponent == -8
    assert args.maximum_scale_exponent == -3

    try:
        parse_args(
            [
                "--mode",
                "continuous",
                "--root",
                str(tmp_path),
                "--out-dir",
                str(tmp_path / "bad"),
                "--maximum-scale-exponent",
                "-5",
            ]
        )
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError("invalid scale bounds were accepted")


def test_addressable_instruction_exactly_implements_max_and_preserves_source() -> None:
    codes = torch.zeros(6)
    codes[3] = 1.0
    instruction = QuantizedConditionalAffineAssignment(
        registers=3,
        source=0,
        target=1,
        mode="continuous",
        initial_codes=codes,
        initial_scale_exponent=0,
        trainable_threshold=False,
    )
    state = torch.tensor([[3.0, 1.0, 7.0], [-2.0, 4.0, -5.0]])
    expected = torch.tensor([[3.0, 3.0, 7.0], [-2.0, 4.0, -5.0]])
    torch.testing.assert_close(instruction(state), expected, rtol=0.0, atol=0.0)


def test_addressable_instruction_supports_dyadic_affine_accumulate_and_erase() -> None:
    accumulate_codes = torch.zeros(6)
    accumulate_codes[1] = 1.0
    accumulate = QuantizedConditionalAffineAssignment(
        registers=3,
        source=0,
        target=1,
        mode="continuous",
        initial_codes=accumulate_codes,
        initial_scale_exponent=-1,
        trainable_threshold=False,
    )
    erase_codes = torch.zeros(6)
    erase_codes[0] = -1.0
    erase = QuantizedConditionalAffineAssignment(
        registers=3,
        source=1,
        target=2,
        mode="continuous",
        initial_codes=erase_codes,
        initial_scale_exponent=0,
        trainable_threshold=False,
    )
    state = torch.tensor([[2.0, 3.0, 9.0]])
    after_accumulate = accumulate(state)
    torch.testing.assert_close(after_accumulate, torch.tensor([[2.0, 4.0, 9.0]]), rtol=0.0, atol=0.0)
    torch.testing.assert_close(erase(after_accumulate), torch.tensor([[2.0, 4.0, 0.0]]), rtol=0.0, atol=0.0)


def test_all_ste_forwards_are_bit_exact_hard_values() -> None:
    generator = torch.Generator().manual_seed(0xC0FFEE)
    margins = torch.randn(1_000_003, generator=generator)
    route = _hard_tanh_ste(margins, 0.1)
    assert torch.equal(route, (margins > 0).to(margins.dtype))

    masters = torch.randn(1_000_003, generator=generator)
    ternary = _ternary_ste(masters, 0.5)
    bounded = torch.tanh(masters)
    expected_codes = torch.where(
        bounded > 0.5,
        torch.ones_like(bounded),
        torch.where(bounded < -0.5, -torch.ones_like(bounded), torch.zeros_like(bounded)),
    )
    assert torch.equal(ternary, expected_codes)

    log2_master = 16.0 * torch.rand(1_000_003, generator=generator) - 8.0
    scale = _dyadic_scale_ste(log2_master, -8, 0)
    expected_scale = torch.exp2(torch.round(log2_master.clamp(-8.0, 0.0)))
    assert torch.equal(scale, expected_scale)


def test_free_reference_forward_matches_literal_hard_route_bit_exactly() -> None:
    layer = QuantizedComparisonAffineSweep(carrier_dim=8, rounds=3, mode="free")
    codes = layer.hard_coefficient_codes().to(torch.float32)
    codes[:, 0, 4] = 1.0
    codes[:, 1, 5] = -1.0
    _set_codes(layer, codes)
    state = torch.randn(4099, 8)
    regular = state
    literal = state
    for round_index in range(layer.rounds):
        regular, _ = layer._round(regular, round_index)
        literal, _ = layer._round(literal, round_index, hard_only=True)
    assert torch.equal(regular, literal)
