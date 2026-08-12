from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Iterable, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

QuantizedComparisonAffineMode = Literal["constant", "continuous", "free"]

__all__ = [
    "QuantizedConditionalAffineAssignment",
    "QuantizedComparisonAffineLedger",
    "QuantizedComparisonAffineMode",
    "QuantizedComparisonAffineStack",
    "QuantizedComparisonAffineSweep",
]


_ELL = 0
_SOURCE = 1
_BIAS = 2
_HINGE = 3
_SELF_JUMP = 4
_BIAS_JUMP = 5
_COEFFICIENTS = 6


def _hard_tanh_ste(margin: Tensor, tau: float) -> Tensor:
    """Return an exact hard bit with a bounded local route derivative."""

    if tau <= 0.0:
        raise ValueError(f"tau must be positive, got {tau}")
    hard = (margin > 0.0).to(margin.dtype)
    soft = 0.5 * (torch.tanh(margin / tau) + 1.0)
    # Parenthesize the zero-valued surrogate correction.  `(hard+soft)-soft`
    # is not bit-exact in float32 and can leak a 1-ulp soft residue.
    return hard + (soft - soft.detach())


def _ternary_ste(master: Tensor, threshold: float) -> Tensor:
    """Materialize {-1,0,+1} while differentiating through tanh(master)."""

    bounded = torch.tanh(master)
    hard = torch.where(
        bounded > threshold,
        torch.ones_like(bounded),
        torch.where(bounded < -threshold, -torch.ones_like(bounded), torch.zeros_like(bounded)),
    )
    return hard + (bounded - bounded.detach())


def _master_from_codes(codes: Tensor, magnitude: float = 0.75) -> Tensor:
    if not 0.0 < magnitude < 1.0:
        raise ValueError("magnitude must be in (0,1)")
    return codes.to(torch.float32) * math.atanh(magnitude)


def _dyadic_scale_ste(log2_master: Tensor, minimum_exponent: int, maximum_exponent: int) -> Tensor:
    """Use an exact power-of-two forward with a smooth log2 master gradient."""

    bounded = log2_master.clamp(float(minimum_exponent), float(maximum_exponent))
    soft = torch.exp2(bounded)
    hard = torch.exp2(torch.round(bounded))
    return hard + (soft - soft.detach())


def _rms(value: Tensor) -> float:
    if value.numel() == 0:
        return 0.0
    return float(value.detach().float().square().mean().sqrt().item())


def _centered_rms(value: Tensor) -> float:
    if value.numel() == 0:
        return 0.0
    centered = value.detach().float() - value.detach().float().mean(dim=-1, keepdim=True)
    return float(centered.square().mean().sqrt().item())


def _common_rms(value: Tensor) -> float:
    if value.numel() == 0:
        return 0.0
    return float(value.detach().float().mean(dim=-1).square().mean().sqrt().item())


def _tensor_hash(items: Iterable[tuple[str, Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in items:
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class QuantizedComparisonAffineLedger:
    carrier_dim: int
    depth: int
    rounds: int
    pairs_per_round: int
    stored_parameters: int
    effective_parameters: int
    comparisons_per_example: int
    coordinate_writes_per_example: int
    dyadic_scale_applications_per_example: int
    signed_or_additive_terms_per_example: int
    instruction_code_reads: int
    instruction_code_storage_bits: int
    receptive_field: int
    full_width_payload_scalars: int


class QuantizedConditionalAffineAssignment(nn.Module):
    r"""One addressable comparison-gated affine register instruction.

    For source register ``u`` and target register ``v`` this instruction writes

    .. math::

       v'=(1+\ell)v+au+b+\delta[u-v-\theta]_+
          +1[u-v-\theta>0](\kappa v+\zeta).

    ``continuous`` masks ``kappa,zeta`` and is exactly an affine--ReLU atom;
    ``free`` admits an explicit discontinuous jump; ``constant`` retains only
    branch-dependent immediate offsets.  Coefficients are ternary times one
    dyadic scale, while the addressed source and target are ordinary live
    registers.

    With protected zero scratch (or immediate bias), addressable edges, and an
    unbounded ordered program, this instruction is a sparse dyadic
    affine--ReLU-complete ISA: branch-independent instances accumulate affine
    gates, and ``delta=1`` implements ``v <- max(v,u)``.  That compiler-level
    statement is distinct from the fixed-depth, shared-round sweep below.
    """

    is_quantized_conditional_affine_assignment = True

    def __init__(
        self,
        *,
        registers: int,
        source: int,
        target: int,
        mode: QuantizedComparisonAffineMode = "continuous",
        tau: float = 0.1,
        ternary_threshold: float = 0.5,
        initial_scale_exponent: int = 0,
        initial_codes: Tensor | None = None,
        threshold: float = 0.0,
        trainable_threshold: bool = True,
    ) -> None:
        super().__init__()
        if registers < 2:
            raise ValueError("registers must be at least two")
        if not 0 <= source < registers or not 0 <= target < registers:
            raise ValueError("source and target must address live registers")
        if source == target:
            raise ValueError("source and target must be distinct")
        if mode not in {"constant", "continuous", "free"}:
            raise ValueError(f"unsupported mode {mode!r}")
        if tau <= 0.0:
            raise ValueError("tau must be positive")
        if not 0.0 < ternary_threshold < 1.0:
            raise ValueError("ternary_threshold must be in (0,1)")

        self.registers = int(registers)
        self.source = int(source)
        self.target = int(target)
        self.mode: QuantizedComparisonAffineMode = mode
        self.tau = float(tau)
        self.ternary_threshold = float(ternary_threshold)
        codes = torch.zeros(_COEFFICIENTS) if initial_codes is None else initial_codes.detach().float().clone()
        if codes.shape != (_COEFFICIENTS,):
            raise ValueError(f"initial_codes must have shape ({_COEFFICIENTS},)")
        if not bool(torch.all((codes == -1) | (codes == 0) | (codes == 1))):
            raise ValueError("initial_codes must be ternary")
        self.coefficient_master = nn.Parameter(_master_from_codes(codes))
        self.log2_scale_master = nn.Parameter(torch.tensor(float(initial_scale_exponent)))
        threshold_tensor = torch.tensor(float(threshold))
        if trainable_threshold:
            self.threshold = nn.Parameter(threshold_tensor)
        else:
            self.register_buffer("threshold", threshold_tensor)

        mask = torch.zeros(_COEFFICIENTS)
        if mode == "constant":
            mask[[_BIAS, _BIAS_JUMP]] = 1.0
        elif mode == "continuous":
            mask[:_SELF_JUMP] = 1.0
        else:
            mask[:] = 1.0
        self.register_buffer("coefficient_mask", mask)

    def effective_coefficients(self) -> Tensor:
        codes = _ternary_ste(self.coefficient_master, self.ternary_threshold)
        scale = _dyadic_scale_ste(self.log2_scale_master, -24, 24)
        return codes * self.coefficient_mask.to(codes) * scale

    def forward(self, state: Tensor) -> Tensor:
        if state.ndim != 2 or state.shape[1] != self.registers:
            raise ValueError(f"expected [batch,{self.registers}], got {tuple(state.shape)}")
        row = self.effective_coefficients().to(device=state.device, dtype=state.dtype)
        u = state[:, self.source]
        v = state[:, self.target]
        margin = u - v - self.threshold.to(device=state.device, dtype=state.dtype)
        q = _hard_tanh_ste(margin, self.tau)
        new_v = (1.0 + row[_ELL]) * v + row[_SOURCE] * u + row[_BIAS] + row[_HINGE] * F.relu(margin) + q * (row[_SELF_JUMP] * v + row[_BIAS_JUMP])
        output = state.clone()
        output[:, self.target] = new_v
        return output

    def hard_coefficient_codes(self) -> Tensor:
        bounded = torch.tanh(self.coefficient_master.detach())
        return torch.where(
            bounded > self.ternary_threshold,
            torch.ones_like(bounded),
            torch.where(
                bounded < -self.ternary_threshold,
                -torch.ones_like(bounded),
                torch.zeros_like(bounded),
            ),
        ).to(torch.int8)


class QuantizedComparisonAffineSweep(nn.Module):
    r"""One globally connected sweep of quantized conditional affine assignments.

    Round ``r`` pairs coordinates whose binary addresses differ in bit ``r``.
    For one pair ``(u,v)`` and margin ``m=u-v-theta``, both endpoints are
    updated simultaneously from the pre-round state:

    .. math::

       u'=(1+\ell_u)u+a_uv+b_u+\delta_u[m]_+
          +q(\kappa_u u+\zeta_u),

       v'=(1+\ell_v)v+a_vu+b_v+\delta_v[m]_+
          +q(\kappa_v v+\zeta_v),\qquad q=1[m>0].

    Every coefficient is a shared ternary instruction code times one learned
    dyadic scale per round.  Pair thresholds are local.  ``continuous`` masks
    ``kappa,zeta`` exactly to zero, yielding an affine--ReLU/NTG map.  ``free``
    exposes those two wall-jump fields.  ``constant`` keeps only the immediate
    branch offset and is the scalar PC-LUT/MADDNESS-style constant-row corner.

    The layer stores no carrier-width payload and performs no dense matrix
    multiplication.  The Torch implementation is a semantic reference, not a
    fused performance kernel.
    """

    is_quantized_comparison_affine = True

    def __init__(
        self,
        *,
        carrier_dim: int = 1024,
        rounds: int | None = None,
        mode: QuantizedComparisonAffineMode = "continuous",
        tau: float = 0.1,
        ternary_threshold: float = 0.5,
        initial_scale_exponent: int = -4,
        minimum_scale_exponent: int = -8,
        maximum_scale_exponent: int = 0,
    ) -> None:
        super().__init__()
        if carrier_dim < 2 or carrier_dim & (carrier_dim - 1):
            raise ValueError(f"carrier_dim must be a power of two >=2, got {carrier_dim}")
        full_rounds = int(math.log2(carrier_dim))
        rounds = full_rounds if rounds is None else int(rounds)
        if not 1 <= rounds <= full_rounds:
            raise ValueError(f"rounds must be in [1,{full_rounds}], got {rounds}")
        if mode not in {"constant", "continuous", "free"}:
            raise ValueError(f"unsupported mode {mode!r}")
        if not 0.0 < ternary_threshold < 1.0:
            raise ValueError("ternary_threshold must be in (0,1)")
        if not minimum_scale_exponent <= initial_scale_exponent <= maximum_scale_exponent:
            raise ValueError("initial scale exponent lies outside its registered bounds")
        if tau <= 0.0:
            raise ValueError("tau must be positive")

        self.carrier_dim = int(carrier_dim)
        self.rounds = rounds
        self.mode: QuantizedComparisonAffineMode = mode
        self.tau = float(tau)
        self.ternary_threshold = float(ternary_threshold)
        self.minimum_scale_exponent = int(minimum_scale_exponent)
        self.maximum_scale_exponent = int(maximum_scale_exponent)
        self.pairs_per_round = self.carrier_dim // 2

        # [round, endpoint (u/v), field].  The continuous/free initialization
        # is a small partial compare-exchange: on m>0, u moves down and v up.
        codes = torch.zeros(self.rounds, 2, _COEFFICIENTS)
        if mode == "constant":
            codes[:, 0, _BIAS_JUMP] = -1.0
            codes[:, 1, _BIAS_JUMP] = 1.0
        else:
            codes[:, 0, _HINGE] = -1.0
            codes[:, 1, _HINGE] = 1.0
        self.coefficient_master = nn.Parameter(_master_from_codes(codes))
        self.log2_scale_master = nn.Parameter(torch.full((self.rounds,), float(initial_scale_exponent)))
        self.thresholds = nn.Parameter(torch.zeros(self.rounds, self.pairs_per_round))

        self.register_buffer("initial_coefficient_master", self.coefficient_master.detach().clone())
        self.register_buffer("initial_log2_scale_master", self.log2_scale_master.detach().clone())
        self.register_buffer("initial_thresholds", self.thresholds.detach().clone())

        mask = torch.zeros(2, _COEFFICIENTS)
        if mode == "constant":
            mask[:, _BIAS] = 1.0
            mask[:, _BIAS_JUMP] = 1.0
        elif mode == "continuous":
            mask[:, :_SELF_JUMP] = 1.0
        else:
            mask[:, :] = 1.0
        self.register_buffer("coefficient_mask", mask)

    def hard_coefficient_codes(self) -> Tensor:
        bounded = torch.tanh(self.coefficient_master.detach())
        return torch.where(
            bounded > self.ternary_threshold,
            torch.ones_like(bounded),
            torch.where(bounded < -self.ternary_threshold, -torch.ones_like(bounded), torch.zeros_like(bounded)),
        ).to(torch.int8)

    def effective_coefficients(self) -> tuple[Tensor, Tensor]:
        codes = _ternary_ste(self.coefficient_master, self.ternary_threshold)
        codes = codes * self.coefficient_mask.unsqueeze(0).to(device=codes.device, dtype=codes.dtype)
        scales = _dyadic_scale_ste(
            self.log2_scale_master,
            self.minimum_scale_exponent,
            self.maximum_scale_exponent,
        )
        return codes * scales[:, None, None], scales

    def _split_round(self, state: Tensor, round_index: int) -> tuple[Tensor, Tensor, int]:
        stride = 1 << round_index
        paired = state.reshape(state.shape[0], -1, 2, stride)
        u = paired[:, :, 0, :].reshape(state.shape[0], -1)
        v = paired[:, :, 1, :].reshape(state.shape[0], -1)
        return u, v, stride

    @staticmethod
    def _join_round(u: Tensor, v: Tensor, stride: int, carrier_dim: int) -> Tensor:
        batch = u.shape[0]
        blocks = carrier_dim // (2 * stride)
        return torch.stack(
            (u.reshape(batch, blocks, stride), v.reshape(batch, blocks, stride)),
            dim=2,
        ).reshape(batch, carrier_dim)

    def _round(
        self,
        state: Tensor,
        round_index: int,
        *,
        hard_only: bool = False,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        u, v, stride = self._split_round(state, round_index)
        coefficients, scales = self.effective_coefficients()
        row_u = coefficients[round_index, 0].to(device=state.device, dtype=state.dtype)
        row_v = coefficients[round_index, 1].to(device=state.device, dtype=state.dtype)
        theta = self.thresholds[round_index].to(device=state.device, dtype=state.dtype)
        margin = u - v - theta
        q = (margin > 0.0).to(state.dtype) if hard_only else _hard_tanh_ste(margin, self.tau)
        hinge = F.relu(margin)

        new_u = (1.0 + row_u[_ELL]) * u + row_u[_SOURCE] * v + row_u[_BIAS] + row_u[_HINGE] * hinge + q * (row_u[_SELF_JUMP] * u + row_u[_BIAS_JUMP])
        new_v = (1.0 + row_v[_ELL]) * v + row_v[_SOURCE] * u + row_v[_BIAS] + row_v[_HINGE] * hinge + q * (row_v[_SELF_JUMP] * v + row_v[_BIAS_JUMP])
        output = self._join_round(new_u, new_v, stride, self.carrier_dim)
        return output, {
            "u": u,
            "v": v,
            "q": q,
            "margin": margin,
            "new_u": new_u,
            "new_v": new_v,
            "row_u": row_u,
            "row_v": row_v,
            "scale": scales[round_index],
        }

    def hard_branch_matrices(self, round_index: int) -> tuple[Tensor, Tensor]:
        """Return the q=0/q=1 hard-forward 2x2 affine Jacobians.

        These matrices exclude the tanh route-surrogate derivative used only
        by training.  In ``free`` mode they also do not summarize the finite
        output jump at the comparison wall.
        """

        if not 0 <= round_index < self.rounds:
            raise IndexError(f"round_index must be in [0,{self.rounds}), got {round_index}")
        coefficients, _scales = self.effective_coefficients()
        row_u = coefficients[round_index, 0]
        row_v = coefficients[round_index, 1]
        matrix_zero = torch.stack(
            (
                torch.stack((1.0 + row_u[_ELL], row_u[_SOURCE])),
                torch.stack((row_v[_SOURCE], 1.0 + row_v[_ELL])),
            )
        )
        matrix_one = torch.stack(
            (
                torch.stack(
                    (
                        1.0 + row_u[_ELL] + row_u[_HINGE] + row_u[_SELF_JUMP],
                        row_u[_SOURCE] - row_u[_HINGE],
                    )
                ),
                torch.stack(
                    (
                        row_v[_SOURCE] + row_v[_HINGE],
                        1.0 + row_v[_ELL] - row_v[_HINGE] + row_v[_SELF_JUMP],
                    )
                ),
            )
        )
        return matrix_zero, matrix_one

    def forward(self, state: Tensor) -> Tensor:
        if state.ndim != 2 or state.shape[1] != self.carrier_dim:
            raise ValueError(f"expected [batch,{self.carrier_dim}], got {tuple(state.shape)}")
        for round_index in range(self.rounds):
            state, _ = self._round(state, round_index)
        return state

    @torch.no_grad()
    def trace(self, state: Tensor) -> list[dict[str, float | int]]:
        rows: list[dict[str, float | int]] = []
        codes = self.hard_coefficient_codes()
        initial_codes = self._hard_codes_from_master(
            self.initial_coefficient_master,
            self.ternary_threshold,
        )
        for round_index in range(self.rounds):
            state_in = state
            state, values = self._round(state, round_index, hard_only=True)
            q = values["q"]
            margin = values["margin"]
            row_u = values["row_u"]
            row_v = values["row_v"]
            # Project each sample to the active wall while retaining v.  This
            # measures the true branch-value jump, not a forced off-wall delta.
            wall_u = values["v"] + self.thresholds[round_index].to(
                device=values["v"].device,
                dtype=values["v"].dtype,
            )
            jump_u = row_u[_SELF_JUMP] * wall_u + row_u[_BIAS_JUMP]
            jump_v = row_v[_SELF_JUMP] * values["v"] + row_v[_BIAS_JUMP]
            wall_jump = torch.stack((jump_u, jump_v), dim=-1)
            branch_coverage = ((q.amin(dim=0) == 0) & (q.amax(dim=0) == 1)).float().mean()
            forced_u, forced_v = self.forced_branch_delta(state_in, round_index)
            matrix_zero, matrix_one = self.hard_branch_matrices(round_index)
            matrix_zero = matrix_zero.float()
            matrix_one = matrix_one.float()
            singular_values = torch.cat((torch.linalg.svdvals(matrix_zero), torch.linalg.svdvals(matrix_one)))
            determinants = torch.stack((torch.linalg.det(matrix_zero), torch.linalg.det(matrix_one)))
            raw_log2_scale = float(self.log2_scale_master[round_index].detach().item())
            rows.append(
                {
                    "round": round_index,
                    "q_fraction": float(q.float().mean().item()),
                    "minority_fraction": float(min(q.float().mean().item(), 1.0 - q.float().mean().item())),
                    "pair_branch_coverage_fraction": float(branch_coverage.item()),
                    "tie_fraction": float((margin.abs() <= 1e-6).float().mean().item()),
                    "state_in_rms": _rms(state_in),
                    "state_out_rms": _rms(state),
                    "state_in_centered_rms": _centered_rms(state_in),
                    "state_out_centered_rms": _centered_rms(state),
                    "state_in_common_rms": _common_rms(state_in),
                    "state_out_common_rms": _common_rms(state),
                    "action_rms": _rms(state - state_in),
                    "wall_jump_rms": _rms(wall_jump),
                    "forced_branch_delta_rms": _rms(torch.stack((forced_u, forced_v), dim=-1)),
                    "scale": float(values["scale"].item()),
                    "raw_log2_scale_master": raw_log2_scale,
                    "scale_at_upper_bound": int(float(values["scale"].item()) >= 2.0**self.maximum_scale_exponent),
                    "scale_master_above_upper_bound": int(raw_log2_scale > self.maximum_scale_exponent),
                    "hard_branch_sigma_max": float(singular_values.max().item()),
                    "hard_branch_sigma_min": float(singular_values.min().item()),
                    "hard_branch_det_min": float(determinants.min().item()),
                    "hard_branch_det_max": float(determinants.max().item()),
                    "hard_branch_abs_det_min": float(determinants.abs().min().item()),
                    "code_zero_fraction": float((codes[round_index] == 0).float().mean().item()),
                    "code_change_fraction": float((codes[round_index] != initial_codes[round_index]).float().mean().item()),
                    "threshold_rms": _rms(self.thresholds[round_index]),
                }
            )
        return rows

    @staticmethod
    def _hard_codes_from_master(master: Tensor, threshold: float = 0.5) -> Tensor:
        bounded = torch.tanh(master.detach())
        return torch.where(
            bounded > threshold,
            torch.ones_like(bounded),
            torch.where(bounded < -threshold, -torch.ones_like(bounded), torch.zeros_like(bounded)),
        ).to(torch.int8)

    def forced_branch_delta(self, state: Tensor, round_index: int) -> tuple[Tensor, Tensor]:
        """Return the two affine branch extensions' difference.

        This includes ``hinge_coefficient * margin`` as well as the explicit
        discontinuous jump.  It is not merely a numerical flip of the hard
        route bit at the factual input.
        """

        if state.ndim != 2 or state.shape[1] != self.carrier_dim:
            raise ValueError(f"expected [batch,{self.carrier_dim}], got {tuple(state.shape)}")
        u, v, _stride = self._split_round(state, round_index)
        coefficients, _scales = self.effective_coefficients()
        row_u = coefficients[round_index, 0].to(device=state.device, dtype=state.dtype)
        row_v = coefficients[round_index, 1].to(device=state.device, dtype=state.dtype)
        theta = self.thresholds[round_index].to(device=state.device, dtype=state.dtype)
        margin = u - v - theta
        return (
            row_u[_HINGE] * margin + row_u[_SELF_JUMP] * u + row_u[_BIAS_JUMP],
            row_v[_HINGE] * margin + row_v[_SELF_JUMP] * v + row_v[_BIAS_JUMP],
        )

    def initial_hash(self) -> str:
        return _tensor_hash(
            (
                ("coefficient_master", self.initial_coefficient_master),
                ("log2_scale_master", self.initial_log2_scale_master),
                ("thresholds", self.initial_thresholds),
            )
        )

    def displacement(self) -> dict[str, float]:
        return {
            "coefficient_master_displacement_rms": _rms(self.coefficient_master - self.initial_coefficient_master),
            "log2_scale_displacement_rms": _rms(self.log2_scale_master - self.initial_log2_scale_master),
            "threshold_displacement_rms": _rms(self.thresholds - self.initial_thresholds),
        }

    def coefficient_parameters(self) -> list[nn.Parameter]:
        return [self.coefficient_master, self.log2_scale_master]

    def threshold_parameters(self) -> list[nn.Parameter]:
        return [self.thresholds]

    def ledger(self) -> QuantizedComparisonAffineLedger:
        edges = self.rounds * self.pairs_per_round
        stored = self.coefficient_master.numel() + self.log2_scale_master.numel() + self.thresholds.numel()
        active_fields = {"constant": 2, "continuous": 4, "free": 6}[self.mode]
        effective = self.thresholds.numel() + self.log2_scale_master.numel() + self.rounds * 2 * active_fields
        return QuantizedComparisonAffineLedger(
            carrier_dim=self.carrier_dim,
            depth=1,
            rounds=self.rounds,
            pairs_per_round=self.pairs_per_round,
            stored_parameters=stored,
            effective_parameters=effective,
            comparisons_per_example=edges,
            coordinate_writes_per_example=2 * edges,
            dyadic_scale_applications_per_example=2 * active_fields * edges,
            signed_or_additive_terms_per_example=2 * active_fields * edges,
            instruction_code_reads=self.rounds * 2 * active_fields,
            instruction_code_storage_bits=self.coefficient_master.numel() * 2,
            receptive_field=1 << self.rounds,
            full_width_payload_scalars=0,
        )


class QuantizedComparisonAffineStack(nn.Module):
    """Serial composition of independent quantized comparison-affine sweeps."""

    def __init__(
        self,
        *,
        depth: int,
        carrier_dim: int = 1024,
        rounds: int | None = None,
        mode: QuantizedComparisonAffineMode = "continuous",
        tau: float = 0.1,
        ternary_threshold: float = 0.5,
        initial_scale_exponent: int = -4,
        minimum_scale_exponent: int = -8,
        maximum_scale_exponent: int = 0,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be positive, got {depth}")
        self.depth = int(depth)
        self.carrier_dim = int(carrier_dim)
        self.mode: QuantizedComparisonAffineMode = mode
        self.blocks = nn.ModuleList(
            QuantizedComparisonAffineSweep(
                carrier_dim=carrier_dim,
                rounds=rounds,
                mode=mode,
                tau=tau,
                ternary_threshold=ternary_threshold,
                initial_scale_exponent=initial_scale_exponent,
                minimum_scale_exponent=minimum_scale_exponent,
                maximum_scale_exponent=maximum_scale_exponent,
            )
            for _ in range(depth)
        )

    def forward(self, state: Tensor) -> Tensor:
        for block in self.blocks:
            # Every sweep is already a state map with its identity term.
            state = block(state)
        return state

    @torch.no_grad()
    def trace(self, state: Tensor) -> list[dict[str, float | int]]:
        rows: list[dict[str, float | int]] = []
        input_rms = _rms(state)
        input_centered_rms = _centered_rms(state)
        input_common_rms = _common_rms(state)
        for block_index, block in enumerate(self.blocks):
            state_in = state
            block_rows = block.trace(state_in)
            state = block(state_in)
            block_input_rms = _rms(state_in)
            block_output_rms = _rms(state)
            block_input_centered_rms = _centered_rms(state_in)
            block_output_centered_rms = _centered_rms(state)
            block_input_common_rms = _common_rms(state_in)
            block_output_common_rms = _common_rms(state)
            for row in block_rows:
                rows.append(
                    {
                        "block": block_index,
                        "stage": block_index * block.rounds + int(row["round"]),
                        **row,
                        "block_input_rms": block_input_rms,
                        "block_output_rms": block_output_rms,
                        "block_gain_rms": block_output_rms / max(block_input_rms, 1e-12),
                        "block_output_over_input_rms": block_output_rms / max(input_rms, 1e-12),
                        "block_input_centered_rms": block_input_centered_rms,
                        "block_output_centered_rms": block_output_centered_rms,
                        "block_centered_gain_rms": block_output_centered_rms / max(block_input_centered_rms, 1e-12),
                        "block_output_over_input_centered_rms": block_output_centered_rms / max(input_centered_rms, 1e-12),
                        "block_input_common_rms": block_input_common_rms,
                        "block_output_common_rms": block_output_common_rms,
                        "block_common_gain_rms": block_output_common_rms / max(block_input_common_rms, 1e-12),
                        "block_output_over_input_common_rms": block_output_common_rms / max(input_common_rms, 1e-12),
                    }
                )
        return rows

    def initial_hash(self) -> str:
        items: list[tuple[str, Tensor]] = []
        for block_index, block in enumerate(self.blocks):
            items.extend(
                (
                    (f"blocks.{block_index}.coefficient_master", block.initial_coefficient_master),
                    (f"blocks.{block_index}.log2_scale_master", block.initial_log2_scale_master),
                    (f"blocks.{block_index}.thresholds", block.initial_thresholds),
                )
            )
        return _tensor_hash(items)

    def block_initial_hashes(self) -> list[str]:
        return [block.initial_hash() for block in self.blocks]

    def coefficient_parameters(self) -> Iterable[nn.Parameter]:
        for block in self.blocks:
            yield from block.coefficient_parameters()

    def threshold_parameters(self) -> Iterable[nn.Parameter]:
        for block in self.blocks:
            yield from block.threshold_parameters()

    def displacement(self) -> dict[str, float]:
        rows = [block.displacement() for block in self.blocks]
        return {key: math.sqrt(sum(row[key] * row[key] for row in rows) / len(rows)) for key in rows[0]}

    def block_displacements(self) -> list[dict[str, float]]:
        return [block.displacement() for block in self.blocks]

    def block_grad_norms(self) -> list[float]:
        values = []
        for block in self.blocks:
            total = 0.0
            for parameter in block.parameters():
                if parameter.grad is not None:
                    total += float(parameter.grad.detach().float().square().sum().item())
            values.append(math.sqrt(total))
        return values

    def ledger(self) -> QuantizedComparisonAffineLedger:
        one = self.blocks[0].ledger()
        return QuantizedComparisonAffineLedger(
            carrier_dim=one.carrier_dim,
            depth=self.depth,
            rounds=one.rounds,
            pairs_per_round=one.pairs_per_round,
            stored_parameters=self.depth * one.stored_parameters,
            effective_parameters=self.depth * one.effective_parameters,
            comparisons_per_example=self.depth * one.comparisons_per_example,
            coordinate_writes_per_example=self.depth * one.coordinate_writes_per_example,
            dyadic_scale_applications_per_example=self.depth * one.dyadic_scale_applications_per_example,
            signed_or_additive_terms_per_example=self.depth * one.signed_or_additive_terms_per_example,
            instruction_code_reads=self.depth * one.instruction_code_reads,
            instruction_code_storage_bits=self.depth * one.instruction_code_storage_bits,
            receptive_field=one.receptive_field,
            full_width_payload_scalars=0,
        )
