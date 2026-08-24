from __future__ import annotations

import math

import pytest
import torch
import tropnn.tools.comparison_microcode_dual_oracle as probe
from tropnn.tools.comparison_microcode_dual_oracle import (
    ARMS,
    ComparisonMicrocodeProgram,
    GuardTrace,
    ProbeConfig,
    evaluate_arm,
    make_inputs,
    make_teacher,
    normalized_delta_loss,
    run_probe,
    teacher_dataset,
    train_arm,
)


def _small_config(**overrides: object) -> ProbeConfig:
    values: dict[str, object] = {
        "dim": 8,
        "sweeps": 2,
        "train_samples": 128,
        "validation_samples": 32,
        "epochs": 0,
        "batch_size": 32,
        "seed": 7,
        "jacobian_samples": 2,
    }
    values.update(overrides)
    return ProbeConfig(**values)


def test_four_factorial_arms_have_one_core_budget_and_split_oracle_side_information() -> None:
    config = _small_config()
    models = [ComparisonMicrocodeProgram(config, arm) for arm in ARMS]
    ledgers = [model.ledger() for model in models]

    assert all(ledger == ledgers[0] for ledger in ledgers[1:])
    ledger = ledgers[0]
    assert ledger.program_length_clocks == 6
    assert ledger.stored_instruction_rows == 12
    assert ledger.stored_parameter_scalars == 36
    assert ledger.active_instruction_invocations == 24
    assert ledger.comparisons == 24
    assert ledger.state_reads == 48
    assert ledger.state_writes == 48
    assert ledger.threshold_scalar_reads == 24
    assert ledger.action_scalar_reads == 48
    assert ledger.route_code_reads == 24
    assert ledger.semantic_multiply_terms == 72
    assert ledger.state_traffic_bytes == 384
    assert ledger.threshold_traffic_bytes == 96
    assert ledger.action_traffic_bytes == 192
    assert ledger.route_code_traffic_bytes == 24
    assert ledger.logical_active_bytes == 696
    assert ledger.receptive_field == 8
    assert ledger.full_width_payload_scalars == 0
    assert ledger.dense_matrix_parameters == 0

    side_ledgers = [model.oracle_side_information_ledger() for model in models]
    assert [side.enabled for side in side_ledgers] == [True, True, False, False]
    assert [side.total_side_information_bytes for side in side_ledgers] == [120, 120, 0, 0]
    assert [model.trainable_parameter_scalars for model in models] == [0, 12, 24, 36]


def test_oracle_guard_trace_is_coherent_and_never_uses_candidate_margin() -> None:
    config = _small_config()
    teacher = make_teacher(config)
    model = ComparisonMicrocodeProgram(config, "oracle_recognition_oracle_action")
    inputs = make_inputs(64, config.dim, 123, dtype=torch.float32, device=torch.device("cpu"))
    targets, teacher_trace = teacher_dataset(teacher, inputs)

    with torch.no_grad():
        model.thresholds.fill_(10.0)
        output, output_trace, candidate_margins = model.forward_with_guard_trace(
            inputs,
            guard_trace_override=teacher_trace,
            surrogate=False,
        )

    assert torch.equal(output, targets)
    assert torch.equal(output_trace.route_bits, teacher_trace.route_bits)
    assert torch.equal(output_trace.gated_margins, teacher_trace.gated_margins)
    assert not torch.equal(
        output_trace.gated_margins,
        output_trace.route_bits * candidate_margins,
    )


def test_guard_action_is_continuous_and_rank_one_compatible() -> None:
    config = _small_config()
    model = make_teacher(config)
    diagnostics = model.guard_compatibility()

    assert diagnostics["maximum_jacobian_jump_rank"] == 1
    assert float(diagnostics["maximum_guard_alignment_residual"]) < 1e-6
    assert float(diagnostics["maximum_wall_value_gap"]) < 1e-6
    assert float(diagnostics["minimum_branch_determinant"]) > 0.0

    stage = 0
    threshold = model.thresholds[stage, 0].detach()
    epsilon = 1e-5
    v = torch.tensor(0.31)
    negative = torch.stack((v + threshold - epsilon, v)).reshape(1, 2)
    positive = torch.stack((v + threshold + epsilon, v)).reshape(1, 2)
    action = model.actions[stage].detach()

    def local_forward(pair: torch.Tensor) -> torch.Tensor:
        margin = pair[:, 0] - pair[:, 1] - threshold
        route = (margin > 0.0).to(pair.dtype)
        amplitude = route * margin
        return pair + amplitude[:, None] * action[None, :]

    jump = local_forward(positive) - local_forward(negative)
    assert float(jump.norm().item()) < 4.0 * epsilon


def test_factorial_axes_expose_only_the_expected_gradients() -> None:
    config = _small_config()
    teacher = make_teacher(config)
    inputs = make_inputs(64, config.dim, 31, dtype=torch.float32, device=torch.device("cpu"))
    targets, teacher_trace = teacher_dataset(teacher, inputs)

    for arm in ARMS:
        model = ComparisonMicrocodeProgram(config, arm)
        override = teacher_trace if model.spec.oracle_guard_trace else None
        output, _guard_trace, _margins = model.forward_with_guard_trace(
            inputs,
            guard_trace_override=override,
            surrogate=True,
        )
        loss = normalized_delta_loss(output, targets, inputs)
        if model.trainable_parameter_scalars:
            loss.backward()

        if model.spec.oracle_guard_trace:
            assert not model.thresholds.requires_grad
            assert model.thresholds.grad is None
        else:
            assert model.thresholds.requires_grad
            assert model.thresholds.grad is not None
            assert torch.isfinite(model.thresholds.grad).all()
            assert float(model.thresholds.grad.abs().sum().item()) > 0.0

        if model.spec.oracle_transport:
            assert not model.actions.requires_grad
            assert model.actions.grad is None
        else:
            assert model.actions.requires_grad
            assert model.actions.grad is not None
            assert torch.isfinite(model.actions.grad).all()
            assert float(model.actions.grad.abs().sum().item()) > 0.0


def test_learned_guard_geometry_recomputes_trace_from_its_own_live_state() -> None:
    config = _small_config()
    model = ComparisonMicrocodeProgram(config, "learned_recognition_learned_action")
    inputs = make_inputs(512, config.dim, 37, dtype=torch.float32, device=torch.device("cpu"))
    with torch.no_grad():
        model.thresholds.zero_()
        model.actions.zero_()
        _zero_output, zero_trace, _zero_margins = model.forward_with_guard_trace(inputs, surrogate=False)
        model.actions[0] = torch.tensor((1.5, -1.5))
        _live_output, live_trace, _live_margins = model.forward_with_guard_trace(inputs, surrogate=False)
        _forced_output, forced_trace, _forced_margins = model.forward_with_guard_trace(
            inputs,
            guard_trace_override=zero_trace,
            surrogate=False,
        )

    assert torch.equal(live_trace.route_bits[:, 0], zero_trace.route_bits[:, 0])
    assert torch.count_nonzero(live_trace.route_bits[:, 1:] != zero_trace.route_bits[:, 1:]) > 0
    assert torch.equal(forced_trace.route_bits, zero_trace.route_bits)
    assert torch.equal(forced_trace.gated_margins, zero_trace.gated_margins)


def test_learned_transport_initialization_is_seeded_and_teacher_sign_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _small_config(seed=19)
    baseline = ComparisonMicrocodeProgram(config, "learned_recognition_learned_action")
    repeat = ComparisonMicrocodeProgram(config, "learned_recognition_learned_action")
    different_seed = ComparisonMicrocodeProgram(
        _small_config(seed=20),
        "learned_recognition_learned_action",
    )
    original_teacher_parameters = probe._teacher_parameters

    def flipped_teacher_signs(*args: object, **kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
        thresholds, actions = original_teacher_parameters(*args, **kwargs)
        return thresholds, -actions

    monkeypatch.setattr(probe, "_teacher_parameters", flipped_teacher_signs)
    flipped_teacher = ComparisonMicrocodeProgram(config, "learned_recognition_learned_action")

    assert torch.equal(baseline.initial_actions, repeat.initial_actions)
    assert torch.equal(baseline.initial_actions, flipped_teacher.initial_actions)
    assert not torch.equal(baseline.initial_actions, different_seed.initial_actions)
    assert float(baseline.initial_actions.abs().max().item()) <= 0.03
    alignment = baseline.transport_initialization_alignment()
    assert alignment["kind"] == "seeded_random_teacher_independent"
    assert -1.0 <= float(alignment["cosine_with_teacher"]) <= 1.0
    assert 0.0 <= float(alignment["sign_agreement_with_teacher"]) <= 1.0

    oracle = ComparisonMicrocodeProgram(config, "learned_recognition_oracle_action")
    oracle_alignment = oracle.transport_initialization_alignment()
    assert oracle_alignment["kind"] == "oracle_teacher_coefficients"
    assert math.isclose(float(oracle_alignment["cosine_with_teacher"]), 1.0)
    assert math.isclose(float(oracle_alignment["sign_agreement_with_teacher"]), 1.0)


def test_metrics_separate_full_and_residual_derivatives_and_per_sample_reach() -> None:
    config = _small_config(jacobian_samples=2)
    teacher = make_teacher(config)
    model = ComparisonMicrocodeProgram(config, "oracle_recognition_oracle_action")
    inputs = make_inputs(16, config.dim, 41, dtype=torch.float32, device=torch.device("cpu"))
    targets, teacher_trace = teacher_dataset(teacher, inputs)
    metrics = evaluate_arm(model, teacher, inputs, targets, teacher_trace, config)

    assert float(metrics["delta_nmse"]) == 0.0
    assert float(metrics["output_nmse"]) == 0.0
    assert math.isclose(float(metrics["delta_cosine"]), 1.0)
    assert math.isclose(float(metrics["output_cosine"]), 1.0)
    assert 0.0 <= float(metrics["route_agreement"]) <= 1.0
    assert float(metrics["signed_teacher_guard_trace_forcing_effect"]) >= 0.0
    assert metrics["true_best_route_regret_available"] is False

    assert metrics["derivative_semantics"] == "conditional_on_fixed_external_teacher_guard_trace"
    assert float(metrics["full_jacobian_relative_frobenius_error"]) > 0.0
    assert int(metrics["model_full_jacobian"]["rank_min"]) == config.dim
    assert int(metrics["model_residual_jacobian"]["rank_max"]) == 0
    assert int(metrics["teacher_residual_jacobian"]["rank_max"]) > 0
    assert len(metrics["model_full_jacobian"]["spectrum_mean"]) == config.dim
    assert len(metrics["model_full_jacobian"]["spectrum_per_sample"]) == 2
    assert len(metrics["model_full_jacobian"]["rank_per_sample"]) == 2

    for prefix in ("full_jvp", "residual_jvp", "full_vjp", "residual_vjp"):
        assert float(metrics[f"{prefix}_relative_error"]) >= 0.0
        assert -1.0 <= float(metrics[f"{prefix}_cosine"]) <= 1.0

    full_reach = metrics["model_full_map_impulse_reach"]
    residual_reach = metrics["model_residual_impulse_reach"]
    assert math.isclose(float(full_reach["reach_fraction"]), 1.0 / config.dim)
    assert float(full_reach["full_width_fraction"]) == 0.0
    assert int(full_reach["outputs_reached_min"]) == 1
    assert len(full_reach["per_sample"]) == 2
    assert int(residual_reach["outputs_reached_min"]) == 0
    assert float(residual_reach["full_width_fraction"]) == 0.0
    assert len(residual_reach["per_sample"]) == 2
    for sample in full_reach["per_sample"]:
        assert set(sample["outputs_reached_quantiles"]) == {"q0", "q25", "q50", "q75", "q100"}


def test_zero_jacobian_sample_cap_audits_complete_validation_split() -> None:
    config = _small_config(validation_samples=5, jacobian_samples=0)
    teacher = make_teacher(config)
    model = ComparisonMicrocodeProgram(config, "learned_recognition_oracle_action")
    inputs = make_inputs(5, config.dim, 47, dtype=torch.float32, device=torch.device("cpu"))
    targets, teacher_trace = teacher_dataset(teacher, inputs)
    metrics = evaluate_arm(model, teacher, inputs, targets, teacher_trace, config)

    assert metrics["audit_samples"] == 5
    assert metrics["audit_sampling"] == "complete_validation_split"
    assert len(metrics["model_full_jacobian"]["rank_per_sample"]) == 5
    assert len(metrics["model_residual_impulse_reach"]["per_sample"]) == 5


def test_run_probe_json_names_honest_axes_and_separate_ledgers() -> None:
    config = _small_config(train_samples=16, validation_samples=4, jacobian_samples=1)
    result = run_probe(config)

    assert result["factorial_axes"] == {
        "guard": ["oracle_guard_trace", "learned_guard_geometry"],
        "transport": ["oracle_transport", "learned_transport"],
    }
    assert "pure recognition oracle" in result["guard_axis_caveat"]
    assert "common_deployable_core_ledger" in result
    assert "active_work_ledger" not in result
    assert result["true_best_route_regret_available"] is False

    arms = result["arms"]
    assert [arms[name]["guard_axis"] for name in ARMS] == [
        "oracle_guard_trace",
        "oracle_guard_trace",
        "learned_guard_geometry",
        "learned_guard_geometry",
    ]
    assert [arms[name]["transport_axis"] for name in ARMS] == [
        "oracle_transport",
        "learned_transport",
        "oracle_transport",
        "learned_transport",
    ]
    assert [arms[name]["oracle_side_information_ledger"]["total_side_information_bytes"] for name in ARMS] == [
        120,
        120,
        0,
        0,
    ]


def test_oracle_guard_trace_transport_learning_reduces_training_error() -> None:
    config = _small_config(epochs=20, action_lr=0.03)
    teacher = make_teacher(config)
    inputs = make_inputs(config.train_samples, config.dim, 53, dtype=torch.float32, device=torch.device("cpu"))
    targets, teacher_trace = teacher_dataset(teacher, inputs)
    model = ComparisonMicrocodeProgram(config, "oracle_recognition_learned_action")
    result = train_arm(model, inputs, targets, teacher_trace, config)

    assert int(result["optimizer_steps"]) == config.epochs * math.ceil(config.train_samples / config.batch_size)
    assert float(result["maximum_gradient_norm"]) > 0.0
    assert float(result["final_delta_nmse"]) < float(result["initial_delta_nmse"])


def test_guard_trace_rejects_incoherent_shapes() -> None:
    with pytest.raises(ValueError, match="identical shapes"):
        GuardTrace(torch.zeros(1, 2), torch.zeros(1, 3))
