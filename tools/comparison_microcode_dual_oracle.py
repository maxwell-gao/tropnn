"""Dual-oracle probe for comparison-addressed sparse microcode.

This module is a controlled CPU reference, not a language-model layer or a
throughput kernel.  One teacher and one active-work ledger are shared by the
four causal arms in the guard-trace x transport-coefficient matrix.  The
oracle guard axis consumes a coherent stored teacher pair ``(q, q*m)`` at
every instruction.  It never combines a teacher bit with a candidate margin.
The learned guard axis recomputes both ``q`` and ``m`` at every stage from its
own recursively updated live state.  Because the continuous instruction uses
``q*m=ReLU(m)``, recognition and amplitude are mathematically inseparable;
the oracle axis is a side-information ceiling, not a pure recognition oracle.

For a pair ``(u, v)`` at stage ``s`` the hard instruction is

    m = u - v - theta[s, pair]
    q = 1[m > 0]
    (u', v') = (u, v) + q * m * (a[s], b[s]).

The hard map is continuous at ``m=0``.  Its adjacent chamber Jacobians differ
by ``[a, b]^T [1, -1]``, so recognition and action obey the rank-one Hadamard
compatibility condition by construction.  Radix-2 pairings are composed for
two sweeps, giving global receptive field without a dense matrix or a
carrier-width payload.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor, nn

ArmName = Literal[
    "oracle_recognition_oracle_action",
    "oracle_recognition_learned_action",
    "learned_recognition_oracle_action",
    "learned_recognition_learned_action",
]

ARMS: tuple[ArmName, ...] = (
    "oracle_recognition_oracle_action",
    "oracle_recognition_learned_action",
    "learned_recognition_oracle_action",
    "learned_recognition_learned_action",
)


@dataclass(frozen=True)
class ProbeConfig:
    dim: int = 8
    sweeps: int = 2
    train_samples: int = 2048
    validation_samples: int = 512
    epochs: int = 400
    batch_size: int = 256
    action_lr: float = 0.03
    threshold_lr: float = 0.02
    route_temperature: float = 0.15
    seed: int = 0
    # Zero means the complete validation split.  That is the default for the
    # D8 probe; larger exploratory dimensions should set an explicit cap.
    jacobian_samples: int = 0
    jacobian_rank_tolerance: float = 1e-6
    impulse_tolerance: float = 1e-5
    scalar_bytes: int = 4

    def validate(self) -> None:
        if self.dim < 2 or self.dim & (self.dim - 1):
            raise ValueError("dim must be a power of two at least two")
        if self.sweeps < 1:
            raise ValueError("sweeps must be positive")
        if self.train_samples < 1 or self.validation_samples < 1:
            raise ValueError("sample counts must be positive")
        if self.epochs < 0 or self.batch_size < 1:
            raise ValueError("epochs must be nonnegative and batch_size positive")
        if self.action_lr <= 0.0 or self.threshold_lr <= 0.0:
            raise ValueError("learning rates must be positive")
        if self.route_temperature <= 0.0:
            raise ValueError("route_temperature must be positive")
        if self.jacobian_samples < 0:
            raise ValueError("jacobian_samples must be nonnegative")
        if self.jacobian_rank_tolerance <= 0.0 or self.impulse_tolerance <= 0.0:
            raise ValueError("Jacobian tolerances must be positive")
        if self.scalar_bytes not in {2, 4, 8}:
            raise ValueError("scalar_bytes must be 2, 4, or 8")


@dataclass(frozen=True)
class DeployableCoreLedger:
    carrier_dim: int
    sweeps: int
    rounds_per_sweep: int
    program_length_clocks: int
    stored_instruction_rows: int
    stored_parameter_scalars: int
    active_instruction_invocations: int
    comparisons: int
    state_reads: int
    state_writes: int
    threshold_scalar_reads: int
    action_scalar_reads: int
    route_code_reads: int
    semantic_multiply_terms: int
    state_traffic_bytes: int
    threshold_traffic_bytes: int
    action_traffic_bytes: int
    route_code_traffic_bytes: int
    logical_active_bytes: int
    receptive_field: int
    full_width_payload_scalars: int
    dense_matrix_parameters: int


@dataclass(frozen=True)
class OracleSideInformationLedger:
    enabled: bool
    guard_route_bit_labels: int
    gated_margin_scalars: int
    guard_route_label_bytes: int
    gated_margin_bytes: int
    total_side_information_bytes: int


@dataclass(frozen=True)
class GuardTrace:
    """One coherent hard guard trace: route bit and its gated margin."""

    route_bits: Tensor
    gated_margins: Tensor

    def __post_init__(self) -> None:
        if self.route_bits.shape != self.gated_margins.shape:
            raise ValueError("route_bits and gated_margins must have identical shapes")

    def __getitem__(self, index: object) -> GuardTrace:
        return GuardTrace(self.route_bits[index], self.gated_margins[index])


@dataclass(frozen=True)
class ArmSpec:
    name: ArmName
    oracle_guard_trace: bool
    oracle_transport: bool


def arm_spec(name: ArmName) -> ArmSpec:
    if name not in ARMS:
        raise ValueError(f"unknown arm {name!r}")
    return ArmSpec(
        name=name,
        oracle_guard_trace=name.startswith("oracle_recognition"),
        oracle_transport=name.endswith("oracle_action"),
    )


def _hard_route_ste(margin: Tensor, temperature: float) -> tuple[Tensor, Tensor]:
    hard = (margin > 0.0).to(margin.dtype)
    soft = 0.5 * (1.0 + torch.tanh(margin / temperature))
    return hard + (soft - soft.detach()), hard


def _cosine(left: Tensor, right: Tensor, epsilon: float = 1e-12) -> float:
    left_flat = left.detach().double().reshape(-1)
    right_flat = right.detach().double().reshape(-1)
    denominator = left_flat.norm() * right_flat.norm()
    if float(denominator.item()) <= epsilon:
        return 1.0 if float((left_flat - right_flat).norm().item()) <= epsilon else 0.0
    return float(torch.dot(left_flat, right_flat).div(denominator).item())


def _relative_error(actual: Tensor, expected: Tensor, epsilon: float = 1e-12) -> float:
    numerator = (actual.detach().double() - expected.detach().double()).norm()
    denominator = expected.detach().double().norm().clamp_min(epsilon)
    return float((numerator / denominator).item())


def _nmse(actual: Tensor, expected: Tensor, reference: Tensor) -> float:
    numerator = (actual.detach().double() - expected.detach().double()).square().sum()
    denominator = reference.detach().double().square().sum().clamp_min(1e-12)
    return float((numerator / denominator).item())


def _teacher_parameters(config: ProbeConfig, *, dtype: torch.dtype, device: torch.device) -> tuple[Tensor, Tensor]:
    rounds = int(math.log2(config.dim))
    stages = config.sweeps * rounds
    pairs = config.dim // 2
    action_cycle = torch.tensor(
        (
            (0.28, -0.16),
            (-0.12, 0.26),
            (0.24, -0.14),
            (-0.16, 0.22),
            (0.20, -0.18),
            (-0.10, 0.24),
        ),
        dtype=dtype,
        device=device,
    )
    actions = action_cycle[torch.arange(stages, device=device) % action_cycle.shape[0]].clone()
    stage = torch.arange(1, stages + 1, dtype=dtype, device=device)[:, None]
    pair = torch.arange(1, pairs + 1, dtype=dtype, device=device)[None, :]
    thresholds = 0.18 * torch.sin(1.61803398875 * stage * pair)
    return thresholds, actions


def _learned_initial_parameters(config: ProbeConfig, *, dtype: torch.dtype, device: torch.device) -> tuple[Tensor, Tensor]:
    stages = config.sweeps * int(math.log2(config.dim))
    pairs = config.dim // 2
    threshold = torch.zeros(stages, pairs, dtype=dtype, device=device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed + 71_003)
    # This stream never reads teacher values or signs.  The small nonzero
    # random initialization gives the learned guard path first-order credit
    # without sign-matching or warm-starting the transport teacher.
    actions = 0.06 * torch.rand(stages, 2, generator=generator, dtype=dtype) - 0.03
    actions = actions.to(device=device)
    return threshold, actions


class ComparisonMicrocodeProgram(nn.Module):
    """Guard-aligned two-lane microcode on repeated butterfly matchings."""

    def __init__(
        self,
        config: ProbeConfig,
        arm: ArmName,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
        _teacher_program: bool = False,
    ) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.spec = arm_spec(arm)
        self.rounds_per_sweep = int(math.log2(config.dim))
        self.stages = config.sweeps * self.rounds_per_sweep
        self.pairs_per_stage = config.dim // 2
        device = torch.device(device)

        teacher_thresholds, teacher_actions = _teacher_parameters(config, dtype=dtype, device=device)
        initial_thresholds, initial_actions = _learned_initial_parameters(config, dtype=dtype, device=device)
        # Candidate oracle-guard arms receive no teacher geometry: their
        # unused candidate thresholds stay at the ordinary zero
        # initialization.  Only the private teacher role owns the thresholds
        # used to generate stored guard traces.
        thresholds = teacher_thresholds if _teacher_program else initial_thresholds
        actions = teacher_actions if (_teacher_program or self.spec.oracle_transport) else initial_actions
        self.thresholds = nn.Parameter(
            thresholds.clone(),
            requires_grad=not (_teacher_program or self.spec.oracle_guard_trace),
        )
        self.actions = nn.Parameter(
            actions.clone(),
            requires_grad=not (_teacher_program or self.spec.oracle_transport),
        )
        self.register_buffer("initial_thresholds", thresholds.clone())
        self.register_buffer("initial_actions", actions.clone())
        self.register_buffer("teacher_actions", teacher_actions.clone())

    def ledger(self) -> DeployableCoreLedger:
        invocations = self.stages * self.pairs_per_stage
        state_reads = 2 * invocations
        state_writes = 2 * invocations
        threshold_reads = invocations
        action_reads = 2 * invocations
        route_reads = invocations
        state_bytes = (state_reads + state_writes) * self.config.scalar_bytes
        threshold_bytes = threshold_reads * self.config.scalar_bytes
        action_bytes = action_reads * self.config.scalar_bytes
        route_bytes = route_reads
        return DeployableCoreLedger(
            carrier_dim=self.config.dim,
            sweeps=self.config.sweeps,
            rounds_per_sweep=self.rounds_per_sweep,
            program_length_clocks=self.stages,
            stored_instruction_rows=2 * self.stages,
            stored_parameter_scalars=self.thresholds.numel() + self.actions.numel(),
            active_instruction_invocations=invocations,
            comparisons=invocations,
            state_reads=state_reads,
            state_writes=state_writes,
            threshold_scalar_reads=threshold_reads,
            action_scalar_reads=action_reads,
            route_code_reads=route_reads,
            semantic_multiply_terms=3 * invocations,
            state_traffic_bytes=state_bytes,
            threshold_traffic_bytes=threshold_bytes,
            action_traffic_bytes=action_bytes,
            route_code_traffic_bytes=route_bytes,
            logical_active_bytes=state_bytes + threshold_bytes + action_bytes + route_bytes,
            receptive_field=self.config.dim,
            full_width_payload_scalars=0,
            dense_matrix_parameters=0,
        )

    def oracle_side_information_ledger(self) -> OracleSideInformationLedger:
        invocations = self.stages * self.pairs_per_stage
        if not self.spec.oracle_guard_trace:
            return OracleSideInformationLedger(False, 0, 0, 0, 0, 0)
        route_bytes = invocations
        margin_bytes = invocations * self.config.scalar_bytes
        return OracleSideInformationLedger(
            enabled=True,
            guard_route_bit_labels=invocations,
            gated_margin_scalars=invocations,
            guard_route_label_bytes=route_bytes,
            gated_margin_bytes=margin_bytes,
            total_side_information_bytes=route_bytes + margin_bytes,
        )

    def transport_initialization_alignment(self) -> dict[str, float | str]:
        initial = self.initial_actions.detach()
        teacher = self.teacher_actions.detach()
        return {
            "kind": "oracle_teacher_coefficients" if self.spec.oracle_transport else "seeded_random_teacher_independent",
            "cosine_with_teacher": _cosine(initial, teacher),
            "sign_agreement_with_teacher": float((torch.sign(initial) == torch.sign(teacher)).float().mean().item()),
        }

    @property
    def trainable_parameter_scalars(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def _split_round(self, state: Tensor, round_index: int) -> tuple[Tensor, Tensor, int]:
        stride = 1 << round_index
        paired = state.reshape(state.shape[0], -1, 2, stride)
        u = paired[:, :, 0, :].reshape(state.shape[0], -1)
        v = paired[:, :, 1, :].reshape(state.shape[0], -1)
        return u, v, stride

    @staticmethod
    def _join_round(u: Tensor, v: Tensor, *, stride: int, dim: int) -> Tensor:
        batch = u.shape[0]
        blocks = dim // (2 * stride)
        return torch.stack(
            (u.reshape(batch, blocks, stride), v.reshape(batch, blocks, stride)),
            dim=2,
        ).reshape(batch, dim)

    def forward_with_guard_trace(
        self,
        state: Tensor,
        *,
        guard_trace_override: GuardTrace | None = None,
        surrogate: bool = True,
    ) -> tuple[Tensor, GuardTrace, Tensor]:
        if state.ndim != 2 or state.shape[1] != self.config.dim:
            raise ValueError(f"expected [batch,{self.config.dim}], got {tuple(state.shape)}")
        expected_trace_shape = (state.shape[0], self.stages, self.pairs_per_stage)
        if guard_trace_override is not None:
            if guard_trace_override.route_bits.shape != expected_trace_shape:
                raise ValueError(
                    "guard trace must have shape "
                    f"[{state.shape[0]},{self.stages},{self.pairs_per_stage}], "
                    f"got {tuple(guard_trace_override.route_bits.shape)}"
                )

        route_rows: list[Tensor] = []
        margin_rows: list[Tensor] = []
        gated_margin_rows: list[Tensor] = []
        for stage in range(self.stages):
            round_index = stage % self.rounds_per_sweep
            u, v, stride = self._split_round(state, round_index)
            margin = u - v - self.thresholds[stage].to(device=state.device, dtype=state.dtype)
            if guard_trace_override is None:
                if surrogate:
                    route, hard_route = _hard_route_ste(margin, self.config.route_temperature)
                else:
                    hard_route = (margin > 0.0).to(state.dtype)
                    route = hard_route
                amplitude = route * margin
                hard_gated_margin = hard_route * margin
            else:
                # The oracle supplies the coherent pair (q, q*m).  Candidate
                # margins are still computed for the common core ledger but
                # never enter the oracle action amplitude.
                hard_route = guard_trace_override.route_bits[:, stage].to(device=state.device, dtype=state.dtype)
                hard_gated_margin = guard_trace_override.gated_margins[:, stage].to(
                    device=state.device,
                    dtype=state.dtype,
                )
                amplitude = hard_gated_margin

            action = self.actions[stage].to(device=state.device, dtype=state.dtype)
            new_u = u + action[0] * amplitude
            new_v = v + action[1] * amplitude
            state = self._join_round(new_u, new_v, stride=stride, dim=self.config.dim)
            route_rows.append(hard_route)
            margin_rows.append(margin)
            gated_margin_rows.append(hard_gated_margin)
        trace = GuardTrace(torch.stack(route_rows, dim=1), torch.stack(gated_margin_rows, dim=1))
        return state, trace, torch.stack(margin_rows, dim=1)

    def forward(self, state: Tensor) -> Tensor:
        return self.forward_with_guard_trace(state)[0]

    def branch_jacobians(self, stage: int) -> tuple[Tensor, Tensor]:
        if not 0 <= stage < self.stages:
            raise IndexError(f"stage must be in [0,{self.stages}), got {stage}")
        identity = torch.eye(2, dtype=self.actions.dtype, device=self.actions.device)
        normal = torch.tensor((1.0, -1.0), dtype=self.actions.dtype, device=self.actions.device)
        jump = torch.outer(self.actions[stage], normal)
        return identity, identity + jump

    def guard_compatibility(self) -> dict[str, float | int]:
        normal = torch.tensor((1.0, -1.0), dtype=self.actions.dtype, device=self.actions.device)
        ranks: list[int] = []
        residuals: list[float] = []
        determinants: list[float] = []
        wall_gaps: list[float] = []
        for stage in range(self.stages):
            branch_zero, branch_one = self.branch_jacobians(stage)
            jump = branch_one - branch_zero
            expected = torch.outer(self.actions[stage], normal)
            singular_values = torch.linalg.svdvals(jump.double())
            tolerance = max(float(singular_values.max().item()), 1.0) * 1e-6
            ranks.append(int((singular_values > tolerance).sum().item()))
            residuals.append(float((jump - expected).double().norm().item()))
            determinants.extend((float(torch.linalg.det(branch_zero).item()), float(torch.linalg.det(branch_one).item())))

            theta = self.thresholds[stage, 0]
            wall = torch.stack((theta + 0.37, torch.tensor(0.37, dtype=theta.dtype, device=theta.device)))
            negative_value = branch_zero @ wall
            positive_value = branch_one @ wall - self.actions[stage] * theta
            wall_gaps.append(float((positive_value - negative_value).double().norm().item()))
        return {
            "maximum_jacobian_jump_rank": max(ranks),
            "maximum_guard_alignment_residual": max(residuals),
            "minimum_branch_determinant": min(determinants),
            "maximum_wall_value_gap": max(wall_gaps),
        }


def make_teacher(
    config: ProbeConfig,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> ComparisonMicrocodeProgram:
    return ComparisonMicrocodeProgram(
        config,
        "oracle_recognition_oracle_action",
        dtype=dtype,
        device=device,
        _teacher_program=True,
    )


def make_inputs(count: int, dim: int, seed: int, *, dtype: torch.dtype, device: torch.device) -> Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    values = 2.5 * torch.rand(count, dim, generator=generator, dtype=dtype) - 1.25
    return values.to(device=device)


@torch.no_grad()
def teacher_dataset(teacher: ComparisonMicrocodeProgram, inputs: Tensor) -> tuple[Tensor, GuardTrace]:
    output, guard_trace, _margins = teacher.forward_with_guard_trace(inputs, surrogate=False)
    return output, guard_trace


def _primary_forward(
    model: ComparisonMicrocodeProgram,
    inputs: Tensor,
    teacher_guard_trace: GuardTrace,
    *,
    surrogate: bool,
) -> tuple[Tensor, GuardTrace]:
    override = teacher_guard_trace if model.spec.oracle_guard_trace else None
    output, guard_trace, _margins = model.forward_with_guard_trace(
        inputs,
        guard_trace_override=override,
        surrogate=surrogate,
    )
    return output, guard_trace


def normalized_delta_loss(output: Tensor, target: Tensor, inputs: Tensor) -> Tensor:
    numerator = (output - target).square().sum()
    denominator = (target - inputs).detach().square().sum().clamp_min(1e-12)
    return numerator / denominator


def train_arm(
    model: ComparisonMicrocodeProgram,
    train_inputs: Tensor,
    train_targets: Tensor,
    train_guard_trace: GuardTrace,
    config: ProbeConfig,
) -> dict[str, float | int]:
    with torch.no_grad():
        initial_output, _ = _primary_forward(model, train_inputs, train_guard_trace, surrogate=False)
        initial_nmse = float(normalized_delta_loss(initial_output, train_targets, train_inputs).item())
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters or config.epochs == 0:
        return {
            "initial_delta_nmse": initial_nmse,
            "final_delta_nmse": initial_nmse,
            "optimizer_steps": 0,
            "maximum_gradient_norm": 0.0,
        }

    groups: list[dict[str, object]] = []
    if model.thresholds.requires_grad:
        groups.append({"params": [model.thresholds], "lr": config.threshold_lr})
    if model.actions.requires_grad:
        groups.append({"params": [model.actions], "lr": config.action_lr})
    optimizer = torch.optim.Adam(groups, betas=(0.9, 0.999), eps=1e-8)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed + 100_003)
    maximum_gradient_norm = 0.0
    optimizer_steps = 0

    for _epoch in range(config.epochs):
        order = torch.randperm(train_inputs.shape[0], generator=generator, device="cpu").to(train_inputs.device)
        for start in range(0, train_inputs.shape[0], config.batch_size):
            index = order[start : start + config.batch_size]
            output, _ = _primary_forward(model, train_inputs[index], train_guard_trace[index], surrogate=True)
            loss = normalized_delta_loss(output, train_targets[index], train_inputs[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=5.0)
            maximum_gradient_norm = max(maximum_gradient_norm, float(gradient_norm.item()))
            optimizer.step()
            optimizer_steps += 1

    with torch.no_grad():
        final_output, _ = _primary_forward(model, train_inputs, train_guard_trace, surrogate=False)
        final_nmse = float(normalized_delta_loss(final_output, train_targets, train_inputs).item())
    return {
        "initial_delta_nmse": initial_nmse,
        "final_delta_nmse": final_nmse,
        "optimizer_steps": optimizer_steps,
        "maximum_gradient_norm": maximum_gradient_norm,
    }


def _single_function(
    model: ComparisonMicrocodeProgram,
    guard_trace_override: GuardTrace | None,
):
    def function(vector: Tensor) -> Tensor:
        override = None
        if guard_trace_override is not None:
            override = GuardTrace(
                guard_trace_override.route_bits.unsqueeze(0),
                guard_trace_override.gated_margins.unsqueeze(0),
            )
        output, _guard_trace, _margins = model.forward_with_guard_trace(
            vector.unsqueeze(0),
            guard_trace_override=override,
            surrogate=False,
        )
        return output.squeeze(0)

    return function


def jacobian_diagnostics(
    model: ComparisonMicrocodeProgram,
    teacher: ComparisonMicrocodeProgram,
    inputs: Tensor,
    teacher_guard_trace: GuardTrace,
    config: ProbeConfig,
) -> dict[str, object]:
    audit_count = inputs.shape[0] if config.jacobian_samples == 0 else min(config.jacobian_samples, inputs.shape[0])
    model_jacobians: list[Tensor] = []
    teacher_jacobians: list[Tensor] = []
    model_jvps: list[Tensor] = []
    teacher_jvps: list[Tensor] = []
    model_vjps: list[Tensor] = []
    teacher_vjps: list[Tensor] = []

    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed + 200_003)
    for item in range(audit_count):
        vector = inputs[item].detach().clone().requires_grad_(True)
        model_override = teacher_guard_trace[item] if model.spec.oracle_guard_trace else None
        model_function = _single_function(model, model_override)
        teacher_function = _single_function(teacher, None)
        model_jacobian = torch.autograd.functional.jacobian(model_function, vector, create_graph=False)
        teacher_jacobian = torch.autograd.functional.jacobian(teacher_function, vector, create_graph=False)
        model_jacobians.append(model_jacobian.detach())
        teacher_jacobians.append(teacher_jacobian.detach())

        tangent = torch.randn(config.dim, generator=generator, dtype=vector.dtype).to(vector.device)
        cotangent = torch.randn(config.dim, generator=generator, dtype=vector.dtype).to(vector.device)
        _model_output, model_jvp = torch.autograd.functional.jvp(model_function, vector, tangent, create_graph=False)
        _teacher_output, teacher_jvp = torch.autograd.functional.jvp(teacher_function, vector, tangent, create_graph=False)
        _model_output, model_vjp = torch.autograd.functional.vjp(model_function, vector, v=cotangent, create_graph=False)
        _teacher_output, teacher_vjp = torch.autograd.functional.vjp(teacher_function, vector, v=cotangent, create_graph=False)
        model_jvps.append(model_jvp.detach())
        teacher_jvps.append(teacher_jvp.detach())
        model_vjps.append(model_vjp.detach())
        teacher_vjps.append(teacher_vjp.detach())

    model_jacobian_stack = torch.stack(model_jacobians)
    teacher_jacobian_stack = torch.stack(teacher_jacobians)
    identity = torch.eye(config.dim, dtype=model_jacobian_stack.dtype, device=model_jacobian_stack.device)
    model_residual_jacobian_stack = model_jacobian_stack - identity
    teacher_residual_jacobian_stack = teacher_jacobian_stack - identity
    model_jvp_stack = torch.stack(model_jvps)
    teacher_jvp_stack = torch.stack(teacher_jvps)
    model_vjp_stack = torch.stack(model_vjps)
    teacher_vjp_stack = torch.stack(teacher_vjps)
    tangent_stack: list[Tensor] = []
    cotangent_stack: list[Tensor] = []
    generator.manual_seed(config.seed + 200_003)
    for _item in range(audit_count):
        tangent_stack.append(torch.randn(config.dim, generator=generator, dtype=inputs.dtype).to(inputs.device))
        cotangent_stack.append(torch.randn(config.dim, generator=generator, dtype=inputs.dtype).to(inputs.device))
    tangents = torch.stack(tangent_stack)
    cotangents = torch.stack(cotangent_stack)
    model_residual_jvp_stack = model_jvp_stack - tangents
    teacher_residual_jvp_stack = teacher_jvp_stack - tangents
    model_residual_vjp_stack = model_vjp_stack - cotangents
    teacher_residual_vjp_stack = teacher_vjp_stack - cotangents

    def spectrum_and_rank(stack: Tensor) -> dict[str, object]:
        spectra = torch.linalg.svdvals(stack.double())
        thresholds = config.jacobian_rank_tolerance * spectra.amax(dim=1, keepdim=True).clamp_min(1e-12)
        ranks = (spectra > thresholds).sum(dim=1)
        return {
            "rank_mean": float(ranks.double().mean().item()),
            "rank_min": int(ranks.min().item()),
            "rank_max": int(ranks.max().item()),
            "rank_per_sample": ranks.tolist(),
            "spectrum_mean": spectra.mean(dim=0).tolist(),
            "spectrum_per_sample": spectra.tolist(),
        }

    def impulse_summary(stack: Tensor) -> dict[str, object]:
        # One count for every (validation sample, input impulse).  Nothing is
        # unioned across samples and no mean Jacobian is thresholded.
        mask = stack.detach().abs() > config.impulse_tolerance
        outputs_per_impulse = mask.sum(dim=1)
        flat = outputs_per_impulse.float().reshape(-1)
        quantile_levels = torch.tensor((0.0, 0.25, 0.5, 0.75, 1.0), device=flat.device)
        quantiles = torch.quantile(flat, quantile_levels)
        per_sample: list[dict[str, object]] = []
        for sample_counts in outputs_per_impulse.float():
            sample_quantiles = torch.quantile(sample_counts, quantile_levels)
            per_sample.append(
                {
                    "reach_fraction": float(sample_counts.mean().div(config.dim).item()),
                    "full_width_fraction": float((sample_counts == config.dim).float().mean().item()),
                    "outputs_reached_min": int(sample_counts.min().item()),
                    "outputs_reached_mean": float(sample_counts.mean().item()),
                    "outputs_reached_quantiles": {
                        "q0": float(sample_quantiles[0].item()),
                        "q25": float(sample_quantiles[1].item()),
                        "q50": float(sample_quantiles[2].item()),
                        "q75": float(sample_quantiles[3].item()),
                        "q100": float(sample_quantiles[4].item()),
                    },
                }
            )
        return {
            "reach_fraction": float(mask.float().mean().item()),
            "full_width_fraction": float((outputs_per_impulse == config.dim).float().mean().item()),
            "outputs_reached_min": int(outputs_per_impulse.min().item()),
            "outputs_reached_mean": float(flat.mean().item()),
            "outputs_reached_quantiles": {
                "q0": float(quantiles[0].item()),
                "q25": float(quantiles[1].item()),
                "q50": float(quantiles[2].item()),
                "q75": float(quantiles[3].item()),
                "q100": float(quantiles[4].item()),
            },
            "per_sample": per_sample,
        }

    return {
        "audit_samples": audit_count,
        "audit_sampling": "complete_validation_split" if audit_count == inputs.shape[0] else "deterministic_prefix",
        "derivative_semantics": (
            "conditional_on_fixed_external_teacher_guard_trace" if model.spec.oracle_guard_trace else "end_to_end_autonomous_live_guard_geometry"
        ),
        "full_jacobian_relative_frobenius_error": _relative_error(model_jacobian_stack, teacher_jacobian_stack),
        "residual_jacobian_relative_frobenius_error": _relative_error(
            model_residual_jacobian_stack,
            teacher_residual_jacobian_stack,
        ),
        "model_full_jacobian": spectrum_and_rank(model_jacobian_stack),
        "teacher_full_jacobian": spectrum_and_rank(teacher_jacobian_stack),
        "model_residual_jacobian": spectrum_and_rank(model_residual_jacobian_stack),
        "teacher_residual_jacobian": spectrum_and_rank(teacher_residual_jacobian_stack),
        "full_jvp_relative_error": _relative_error(model_jvp_stack, teacher_jvp_stack),
        "full_jvp_cosine": _cosine(model_jvp_stack, teacher_jvp_stack),
        "residual_jvp_relative_error": _relative_error(model_residual_jvp_stack, teacher_residual_jvp_stack),
        "residual_jvp_cosine": _cosine(model_residual_jvp_stack, teacher_residual_jvp_stack),
        "full_vjp_relative_error": _relative_error(model_vjp_stack, teacher_vjp_stack),
        "full_vjp_cosine": _cosine(model_vjp_stack, teacher_vjp_stack),
        "residual_vjp_relative_error": _relative_error(model_residual_vjp_stack, teacher_residual_vjp_stack),
        "residual_vjp_cosine": _cosine(model_residual_vjp_stack, teacher_residual_vjp_stack),
        "model_full_map_impulse_reach": impulse_summary(model_jacobian_stack),
        "teacher_full_map_impulse_reach": impulse_summary(teacher_jacobian_stack),
        "model_residual_impulse_reach": impulse_summary(model_residual_jacobian_stack),
        "teacher_residual_impulse_reach": impulse_summary(teacher_residual_jacobian_stack),
    }


def evaluate_arm(
    model: ComparisonMicrocodeProgram,
    teacher: ComparisonMicrocodeProgram,
    inputs: Tensor,
    targets: Tensor,
    teacher_guard_trace: GuardTrace,
    config: ProbeConfig,
) -> dict[str, object]:
    with torch.no_grad():
        primary, _primary_guard_trace = _primary_forward(model, inputs, teacher_guard_trace, surrogate=False)
        autonomous, autonomous_guard_trace, _margins = model.forward_with_guard_trace(inputs, surrogate=False)
        forced, _forced_guard_trace, _margins = model.forward_with_guard_trace(
            inputs,
            guard_trace_override=teacher_guard_trace,
            surrogate=False,
        )
        route_match = autonomous_guard_trace.route_bits == teacher_guard_trace.route_bits
        per_stage_agreement = route_match.float().mean(dim=(0, 2))
        autonomous_error = (autonomous - targets).double().square().sum(dim=1)
        forced_error = (forced - targets).double().square().sum(dim=1)
        reference_energy = (targets - inputs).double().square().sum(dim=1).mean().clamp_min(1e-12)
        forcing_effect = (autonomous_error - forced_error).mean() / reference_energy
        forcing_improved_fraction = (autonomous_error > forced_error).double().mean()
        primary_delta = primary - inputs
        target_delta = targets - inputs
        scalar_metrics: dict[str, object] = {
            "output_nmse": _nmse(primary, targets, targets),
            "output_cosine": _cosine(primary, targets),
            "delta_nmse": _nmse(primary_delta, target_delta, target_delta),
            "delta_cosine": _cosine(primary_delta, target_delta),
            "route_agreement": float(route_match.float().mean().item()),
            "route_agreement_per_stage": per_stage_agreement.tolist(),
            "signed_teacher_guard_trace_forcing_effect": float(forcing_effect.item()),
            "teacher_guard_trace_forcing_improved_fraction": float(forcing_improved_fraction.item()),
            "true_best_route_regret_available": False,
            "autonomous_delta_nmse": _nmse(autonomous - inputs, target_delta, target_delta),
            "teacher_guard_trace_forced_delta_nmse": _nmse(forced - inputs, target_delta, target_delta),
        }
    scalar_metrics.update(jacobian_diagnostics(model, teacher, inputs, teacher_guard_trace, config))
    return scalar_metrics


def run_probe(config: ProbeConfig, arms: tuple[ArmName, ...] = ARMS) -> dict[str, object]:
    config.validate()
    device = torch.device("cpu")
    dtype = torch.float32
    teacher = make_teacher(config, dtype=dtype, device=device)
    train_inputs = make_inputs(config.train_samples, config.dim, config.seed + 11, dtype=dtype, device=device)
    validation_inputs = make_inputs(config.validation_samples, config.dim, config.seed + 29, dtype=dtype, device=device)
    train_targets, train_guard_trace = teacher_dataset(teacher, train_inputs)
    validation_targets, validation_guard_trace = teacher_dataset(teacher, validation_inputs)

    results: dict[str, object] = {}
    common_ledger: DeployableCoreLedger | None = None
    for name in arms:
        model = ComparisonMicrocodeProgram(config, name, dtype=dtype, device=device)
        ledger = model.ledger()
        if common_ledger is None:
            common_ledger = ledger
        elif ledger != common_ledger:
            raise RuntimeError("all oracle arms must have an identical active-work ledger")
        training = train_arm(model, train_inputs, train_targets, train_guard_trace, config)
        oracle_side_information = model.oracle_side_information_ledger()
        results[name] = {
            "legacy_arm_name": name,
            "guard_axis": ("oracle_guard_trace" if model.spec.oracle_guard_trace else "learned_guard_geometry"),
            "transport_axis": "oracle_transport" if model.spec.oracle_transport else "learned_transport",
            "oracle_guard_trace": model.spec.oracle_guard_trace,
            "oracle_transport": model.spec.oracle_transport,
            "guard_state_source": (
                "stored_coherent_teacher_q_and_q_times_margin_trace"
                if model.spec.oracle_guard_trace
                else "candidate_comparison_and_margin_from_its_own_recursively_updated_live_state"
            ),
            "transport_initialization": model.transport_initialization_alignment(),
            "stored_parameter_scalars": ledger.stored_parameter_scalars,
            "trainable_parameter_scalars": model.trainable_parameter_scalars,
            "oracle_side_information_ledger": asdict(oracle_side_information),
            "training": training,
            "validation": evaluate_arm(
                model,
                teacher,
                validation_inputs,
                validation_targets,
                validation_guard_trace,
                config,
            ),
            "guard_compatibility": model.guard_compatibility(),
        }

    assert common_ledger is not None
    return {
        "probe": "comparison_microcode_dual_oracle",
        "claim_scope": "controlled CPU teacher probe; not an LM or throughput result",
        "factorial_axes": {
            "guard": ["oracle_guard_trace", "learned_guard_geometry"],
            "transport": ["oracle_transport", "learned_transport"],
        },
        "guard_axis_caveat": (
            "The continuous instruction transports q*m=ReLU(m), so the route bit and gated "
            "margin/amplitude cannot be separated into a pure recognition oracle without "
            "changing the map."
        ),
        "oracle_guard_trace_definition": (
            "Per-sample stored teacher side information supplies the coherent pair (q, q*m) "
            "at every instruction; candidate margins never supply oracle amplitudes."
        ),
        "learned_guard_geometry_definition": ("Every instruction recomputes both q and m from the candidate's own recursively updated live state."),
        "oracle_guard_derivative_definition": (
            "Jacobians/JVPs/VJPs condition on the stored (q,q*m) side information held fixed; "
            "they do not differentiate through the teacher trace provider."
        ),
        "signed_teacher_guard_trace_forcing_effect_definition": (
            "(autonomous squared error - teacher-guard-trace-forced squared error) divided by "
            "mean teacher residual energy; positive means forcing the coherent teacher trace helps."
        ),
        "true_best_route_regret_available": False,
        "config": asdict(config),
        "common_deployable_core_ledger": asdict(common_ledger),
        "teacher_guard_compatibility": teacher.guard_compatibility(),
        "arms": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--sweeps", type=int, default=2)
    parser.add_argument("--train-samples", type=int, default=2048)
    parser.add_argument("--validation-samples", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--action-lr", type=float, default=0.03)
    parser.add_argument("--threshold-lr", type=float, default=0.02)
    parser.add_argument("--route-temperature", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--jacobian-samples",
        type=int,
        default=0,
        help="deterministic validation-prefix size; 0 audits the complete validation split (default)",
    )
    parser.add_argument("--arm", action="append", choices=ARMS, dest="arms")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ProbeConfig(
        dim=args.dim,
        sweeps=args.sweeps,
        train_samples=args.train_samples,
        validation_samples=args.validation_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        action_lr=args.action_lr,
        threshold_lr=args.threshold_lr,
        route_temperature=args.route_temperature,
        seed=args.seed,
        jacobian_samples=args.jacobian_samples,
    )
    selected_arms = ARMS if args.arms is None else tuple(args.arms)
    result = run_probe(config, selected_arms)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
