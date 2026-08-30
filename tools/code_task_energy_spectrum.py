"""Operator-order energy spectra for frozen categorical code families.

The estimator answers a deliberately operational question: how much held-out
target variance can be decoded by additive table rows, by a registered set of
pair-cell rows, and (for small table subsets) by an unrestricted joint tuple.
All fits in this module are deterministic float64 sufficient-statistic or
conjugate-gradient computations on CPU or CUDA; no SGD is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch
from torch import Tensor

SCHEMA = "code-task-energy-spectrum-v1"
EMNIST_SCHEMA = "code-task-energy-spectrum-emnist-v1"
PLANTED_SCHEMA = "code-task-energy-spectrum-planted-validation-v1"
DEFAULT_RIDGE = 1e-6
DEFAULT_CG_TOLERANCE = 1e-6
DEFAULT_CG_ITERATIONS = 2048
DEFAULT_CG_TARGET_CHUNK = 8
CG_CHUNKS_PER_SPARSE_BATCH = 8
JL_DIMENSION = 64
JL_SEEDS = (41_003, 41_009, 41_021)
FIT_GENERATOR_SEED = 90_007
VALIDATION_GENERATOR_SEED = 90_013
TEST_GENERATOR_SEED = 90_019
ZIPF_FIT_SAMPLES = 1 << 20
ZIPF_EVAL_SAMPLES = 16_384
ZIPF_FAMILIES = ("canonical", "walsh", "independent")


def _require_matrix(name: str, value: Tensor) -> Tensor:
    if value.ndim != 2:
        raise ValueError(f"{name} must be a matrix, got {tuple(value.shape)}")
    return value


def _validate_codes(codes: Tensor, rows: int) -> Tensor:
    codes = _require_matrix("codes", codes.detach().to(device="cpu", dtype=torch.int64).contiguous())
    if rows < 2:
        raise ValueError("rows must be at least two")
    if codes.numel() and (int(codes.min()) < 0 or int(codes.max()) >= rows):
        raise ValueError(f"code outside [0,{rows})")
    return codes


def _validate_targets(targets: Tensor, samples: int) -> Tensor:
    targets = _require_matrix("targets", targets.detach().to(device="cpu", dtype=torch.float64).contiguous())
    if targets.shape[0] != samples:
        raise ValueError("code and target sample counts differ")
    if not bool(torch.isfinite(targets).all()):
        raise ValueError("targets contain nonfinite values")
    return targets


def adjacent_pairing(tables: int) -> tuple[tuple[int, int], ...]:
    if tables < 2 or tables % 2:
        raise ValueError("adjacent pairing requires a positive even table count")
    return tuple((left, left + 1) for left in range(0, tables, 2))


def random_pairing(tables: int, seed: int) -> tuple[tuple[int, int], ...]:
    if tables < 2 or tables % 2:
        raise ValueError("random pairing requires a positive even table count")
    order = torch.randperm(tables, generator=torch.Generator(device="cpu").manual_seed(int(seed))).tolist()
    return tuple(sorted((min(order[i], order[i + 1]), max(order[i], order[i + 1])) for i in range(0, tables, 2)))


@dataclass(frozen=True)
class CategoricalDesign:
    """One active category in every feature group for every sample."""

    indices: Tensor
    group_sizes: tuple[int, ...]
    group_offsets: tuple[int, ...]

    def __post_init__(self) -> None:
        indices = _require_matrix("design indices", self.indices)
        if indices.dtype != torch.int64 or indices.device.type != "cpu":
            raise ValueError("design indices must be CPU int64")
        if indices.shape[1] != len(self.group_sizes) or len(self.group_sizes) != len(self.group_offsets):
            raise ValueError("design group metadata does not match its columns")
        if any(size < 1 for size in self.group_sizes):
            raise ValueError("design group sizes must be positive")
        expected = 0
        for offset, size in zip(self.group_offsets, self.group_sizes, strict=True):
            if offset != expected:
                raise ValueError("design offsets must be contiguous")
            expected += size
        if indices.numel() and (int(indices.min()) < 0 or int(indices.max()) >= expected):
            raise ValueError("design index outside the declared feature range")

    @property
    def samples(self) -> int:
        return int(self.indices.shape[0])

    @property
    def groups(self) -> int:
        return int(self.indices.shape[1])

    @property
    def features(self) -> int:
        return sum(self.group_sizes)


def categorical_design(
    codes: Tensor,
    rows: int,
    pairings: Sequence[tuple[int, int]] = (),
    *,
    include_marginals: bool = True,
) -> CategoricalDesign:
    """Construct marginal and pair-cell one-hot group indices.

    Indices are already offset into one flat coefficient array.  Retaining the
    marginal groups in the degree-2 design follows the registered 48-active-
    feature contract even though adjacent pair cells span their marginals.
    """

    codes = _validate_codes(codes, rows)
    columns: list[Tensor] = []
    sizes: list[int] = []
    offsets: list[int] = []
    offset = 0
    if include_marginals:
        for table in range(codes.shape[1]):
            columns.append(codes[:, table] + offset)
            sizes.append(rows)
            offsets.append(offset)
            offset += rows
    used: set[tuple[int, int]] = set()
    for left, right in pairings:
        left, right = int(left), int(right)
        if not (0 <= left < right < codes.shape[1]):
            raise ValueError(f"invalid table pair {(left, right)}")
        if (left, right) in used:
            raise ValueError(f"duplicate table pair {(left, right)}")
        used.add((left, right))
        columns.append(codes[:, left] * rows + codes[:, right] + offset)
        sizes.append(rows * rows)
        offsets.append(offset)
        offset += rows * rows
    if not columns:
        raise ValueError("categorical design has no feature groups")
    return CategoricalDesign(torch.stack(columns, dim=1).contiguous(), tuple(sizes), tuple(offsets))


def _design_counts(design: CategoricalDesign) -> Tensor:
    return torch.bincount(design.indices.reshape(-1), minlength=design.features).to(torch.float64)


def _design_forward(design: CategoricalDesign, coefficient: Tensor, feature_mean: Tensor) -> Tensor:
    if coefficient.ndim != 2 or coefficient.shape[0] != design.features:
        raise ValueError("coefficient shape does not match design")
    prediction = torch.zeros(design.samples, coefficient.shape[1], dtype=torch.float64)
    for group in range(design.groups):
        prediction.add_(coefficient[design.indices[:, group]])
    prediction.sub_(feature_mean @ coefficient)
    return prediction


def _design_transpose(design: CategoricalDesign, values: Tensor, feature_mean: Tensor) -> Tensor:
    values = _validate_targets(values, design.samples)
    result = torch.zeros(design.features, values.shape[1], dtype=torch.float64)
    for group in range(design.groups):
        result.index_add_(0, design.indices[:, group], values)
    result.div_(design.samples)
    result.sub_(feature_mean[:, None] * values.mean(dim=0, keepdim=True))
    return result


@dataclass(frozen=True)
class _SparseCenteredOperator:
    """CSR implementation of the centered categorical normal operator."""

    matrix: Tensor
    transpose: Tensor
    feature_mean: Tensor
    samples: int
    ridge: float

    @classmethod
    def from_design(
        cls,
        design: CategoricalDesign,
        feature_mean: Tensor,
        ridge: float,
        device: torch.device,
    ) -> "_SparseCenteredOperator":
        groups = design.groups
        crow = torch.arange(0, (design.samples + 1) * groups, groups, dtype=torch.int64, device=device)
        columns = design.indices.reshape(-1).to(device)
        values = torch.ones(columns.numel(), dtype=torch.float64, device=device)
        matrix = torch.sparse_csr_tensor(crow, columns, values, size=(design.samples, design.features), check_invariants=False)
        return cls(matrix, matrix.transpose(0, 1), feature_mean.to(device), design.samples, ridge)

    def forward(self, coefficient: Tensor) -> Tensor:
        result = torch.sparse.mm(self.matrix, coefficient)
        result.sub_(self.feature_mean @ coefficient)
        return result

    def transpose_apply(self, values: Tensor) -> Tensor:
        result = torch.sparse.mm(self.transpose, values) / self.samples
        result.sub_(self.feature_mean[:, None] * values.mean(dim=0, keepdim=True))
        return result

    def normal(self, coefficient: Tensor) -> Tensor:
        return self.transpose_apply(self.forward(coefficient)) + self.ridge * coefficient


@dataclass(frozen=True)
class _TritonCenteredOperator:
    indices: Tensor
    feature_mean: Tensor
    samples: int
    features: int
    ridge: float

    @classmethod
    def from_design(cls, design: CategoricalDesign, feature_mean: Tensor, ridge: float, device: torch.device) -> "_TritonCenteredOperator":
        return cls(
            design.indices.to(device=device, dtype=torch.int32).contiguous(),
            feature_mean.to(device),
            design.samples,
            design.features,
            ridge,
        )

    def forward(self, coefficient: Tensor) -> Tensor:
        from tropnn.backends.categorical_energy_triton import categorical_forward

        result = categorical_forward(self.indices, coefficient.contiguous())
        result.sub_(self.feature_mean @ coefficient)
        return result

    def transpose_apply(self, values: Tensor) -> Tensor:
        from tropnn.backends.categorical_energy_triton import categorical_transpose

        result = categorical_transpose(self.indices, values.contiguous(), self.features) / self.samples
        result.sub_(self.feature_mean[:, None] * values.mean(dim=0, keepdim=True))
        return result

    def normal(self, coefficient: Tensor) -> Tensor:
        return self.transpose_apply(self.forward(coefficient)) + self.ridge * coefficient


@dataclass(frozen=True)
class CategoricalFit:
    coefficient: Tensor
    target_mean: Tensor
    feature_mean: Tensor
    ridge: float
    solver: str
    iterations: int
    relative_residual: float
    converged: bool
    feature_count: int
    active_features_per_sample: int
    execution_device: str

    def predict(self, design: CategoricalDesign) -> Tensor:
        if design.features != self.feature_count or design.groups != self.active_features_per_sample:
            raise ValueError("evaluation design does not match fitted design")
        return self.target_mean + _design_forward(design, self.coefficient, self.feature_mean)

    def audit(self) -> dict[str, object]:
        result = asdict(self)
        for name in ("coefficient", "target_mean", "feature_mean"):
            value = result.pop(name)
            result[f"{name}_shape"] = list(value.shape)
        return result


def _count_gram(design: CategoricalDesign, device: torch.device = torch.device("cpu")) -> Tensor:
    """Accumulate X'X exactly from categorical co-occurrence counts."""

    indices = design.indices.to(device)
    gram = torch.zeros(design.features, design.features, dtype=torch.float64, device=device)
    for left in range(design.groups):
        left_offset = design.group_offsets[left]
        left_size = design.group_sizes[left]
        left_code = indices[:, left] - left_offset
        left_slice = slice(left_offset, left_offset + left_size)
        counts = torch.bincount(left_code, minlength=left_size).to(torch.float64)
        gram[left_slice, left_slice] = torch.diag(counts)
        for right in range(left + 1, design.groups):
            right_offset = design.group_offsets[right]
            right_size = design.group_sizes[right]
            right_code = indices[:, right] - right_offset
            joint = torch.bincount(left_code * right_size + right_code, minlength=left_size * right_size)
            block = joint.reshape(left_size, right_size).to(torch.float64)
            right_slice = slice(right_offset, right_offset + right_size)
            gram[left_slice, right_slice] = block
            gram[right_slice, left_slice] = block.T
    return gram


def fit_v1_count_ridge(
    codes: Tensor,
    targets: Tensor,
    rows: int,
    *,
    ridge: float = DEFAULT_RIDGE,
    solver_device: torch.device | str = "cpu",
) -> CategoricalFit:
    """Fit the additive V1 model from exact count sufficient statistics."""

    if ridge <= 0:
        raise ValueError("ridge must be positive for the redundant one-hot parameterization")
    codes = _validate_codes(codes, rows)
    targets = _validate_targets(targets, codes.shape[0])
    design = categorical_design(codes, rows)
    execution_device = torch.device(solver_device)
    if execution_device.type == "cuda":
        from tropnn.backends.categorical_energy_triton import categorical_transpose

        indices = design.indices.to(device=execution_device, dtype=torch.int32).contiguous()
        target_values = targets.to(execution_device)
        target_mean_device = target_values.mean(dim=0)
        centered = target_values - target_mean_device
        counts_device = torch.bincount(indices.reshape(-1).to(torch.int64), minlength=design.features).to(torch.float64)
        feature_mean_device = counts_device / design.samples
        raw_gram = _count_gram(design, execution_device) / design.samples
        gram = raw_gram - torch.outer(feature_mean_device, feature_mean_device)
        cross = categorical_transpose(indices, centered.contiguous(), design.features) / design.samples
        cross.sub_(feature_mean_device[:, None] * centered.mean(dim=0, keepdim=True))
        regularized = gram + ridge * torch.eye(design.features, dtype=torch.float64, device=execution_device)
        coefficient_device = torch.linalg.solve(regularized, cross)
        residual = regularized @ coefficient_device - cross
        relative = float(residual.norm() / cross.norm().clamp_min(1e-30))
        coefficient = coefficient_device.cpu()
        target_mean = target_mean_device.cpu()
        feature_mean = feature_mean_device.cpu()
    else:
        target_mean = targets.mean(dim=0)
        centered = targets - target_mean
        counts = _design_counts(design)
        feature_mean = counts / design.samples
        raw_gram = _count_gram(design) / design.samples
        gram = raw_gram - torch.outer(feature_mean, feature_mean)
        cross = _design_transpose(design, centered, feature_mean)
        regularized = gram + ridge * torch.eye(design.features, dtype=torch.float64)
        coefficient = torch.linalg.solve(regularized, cross)
        residual = regularized @ coefficient - cross
        relative = float(residual.norm() / cross.norm().clamp_min(1e-30))
    return CategoricalFit(
        coefficient=coefficient,
        target_mean=target_mean,
        feature_mean=feature_mean,
        ridge=float(ridge),
        solver="count_normal_equation_float64",
        iterations=1,
        relative_residual=relative,
        converged=True,
        feature_count=design.features,
        active_features_per_sample=design.groups,
        execution_device=str(execution_device),
    )


def _chunk_inner(left: Tensor, right: Tensor, slices: Sequence[slice]) -> Tensor:
    return torch.stack([(left[:, item] * right[:, item]).sum() for item in slices])


def _column_scales(values: Tensor, slices: Sequence[slice], columns: int) -> Tensor:
    result = torch.zeros(columns, dtype=torch.float64, device=values.device)
    for value, item in zip(values, slices, strict=True):
        result[item] = value
    return result.unsqueeze(0)


def _pcg_chunk_batch(
    operator: _SparseCenteredOperator | _TritonCenteredOperator,
    centered_target: Tensor,
    max_iterations: int,
    tolerance: float,
    diagonal: Tensor,
    target_chunk: int,
) -> tuple[Tensor, list[int], list[float], list[bool]]:
    slices = [slice(start, min(start + target_chunk, centered_target.shape[1])) for start in range(0, centered_target.shape[1], target_chunk)]
    right_hand_side = operator.transpose_apply(centered_target)
    solution = torch.zeros_like(right_hand_side)
    residual = right_hand_side.clone()
    preconditioned = residual / diagonal[:, None]
    direction = preconditioned.clone()
    residual_dot = _chunk_inner(residual, preconditioned, slices)
    raw_initial_norm = _chunk_inner(right_hand_side, right_hand_side, slices).sqrt()
    initial_norm = raw_initial_norm.clamp_min(1e-30)
    relative_residual = torch.ones(len(slices), dtype=torch.float64)
    converged = raw_initial_norm == 0
    completed_by_chunk = torch.zeros(len(slices), dtype=torch.int64, device=centered_target.device)
    direction.mul_(_column_scales((~converged).to(torch.float64), slices, direction.shape[1]))
    for completed in range(1, max_iterations + 1):
        active = ~converged
        if not bool(active.any()):
            break
        image = operator.normal(direction)
        denominator = _chunk_inner(direction, image, slices)
        if not bool(torch.isfinite(denominator[active]).all()) or bool((denominator[active] <= 0).any()):
            raise RuntimeError("CG encountered a nonpositive or nonfinite curvature")
        step = torch.zeros_like(denominator)
        step[active] = residual_dot[active] / denominator[active]
        step_columns = _column_scales(step, slices, direction.shape[1])
        solution.add_(direction * step_columns)
        residual.sub_(image * step_columns)
        next_norm = _chunk_inner(residual, residual, slices).sqrt()
        relative_residual = next_norm / initial_norm
        newly_converged = active & (relative_residual <= tolerance)
        completed_by_chunk[newly_converged] = completed
        converged |= newly_converged
        active = ~converged
        if not bool(active.any()):
            break
        next_preconditioned = residual / diagonal[:, None]
        next_dot = _chunk_inner(residual, next_preconditioned, slices)
        if not bool(torch.isfinite(next_dot[active]).all()) or bool((next_dot[active] < 0).any()):
            raise RuntimeError("CG preconditioned residual became invalid")
        beta = torch.zeros_like(next_dot)
        beta[active] = next_dot[active] / residual_dot[active].clamp_min(1e-300)
        direction.mul_(_column_scales(beta, slices, direction.shape[1])).add_(next_preconditioned)
        direction.mul_(_column_scales(active.to(torch.float64), slices, direction.shape[1]))
        preconditioned = next_preconditioned
        residual_dot = next_dot
    completed_by_chunk[(raw_initial_norm == 0)] = 0
    relative_residual[raw_initial_norm == 0] = 0.0
    completed_by_chunk[(~converged) & (completed_by_chunk == 0)] = max_iterations
    return solution, completed_by_chunk.tolist(), relative_residual.tolist(), converged.tolist()


def fit_v2_sparse_cg(
    codes: Tensor,
    targets: Tensor,
    rows: int,
    pairings: Sequence[tuple[int, int]],
    *,
    ridge: float = DEFAULT_RIDGE,
    max_iterations: int = DEFAULT_CG_ITERATIONS,
    tolerance: float = DEFAULT_CG_TOLERANCE,
    target_chunk: int = DEFAULT_CG_TARGET_CHUNK,
    solver_device: torch.device | str = "cpu",
    progress: Callable[[str], None] | None = None,
) -> CategoricalFit:
    """Fit marginal plus pair-cell V2 by matrix-free float64 PCG."""

    if ridge <= 0 or max_iterations < 1 or tolerance <= 0 or target_chunk < 1:
        raise ValueError("invalid V2 solver controls")
    codes = _validate_codes(codes, rows)
    targets = _validate_targets(targets, codes.shape[0])
    design = categorical_design(codes, rows, pairings)
    target_mean = targets.mean(dim=0)
    centered = targets - target_mean
    counts = _design_counts(design)
    feature_mean = counts / design.samples
    execution_device = torch.device(solver_device)
    # diag(Z'Z/N + lambda I) = p(1-p) + lambda.
    diagonal = (feature_mean * (1.0 - feature_mean) + ridge).clamp_min(ridge).to(execution_device)
    if execution_device.type == "cuda":
        operator: _SparseCenteredOperator | _TritonCenteredOperator = _TritonCenteredOperator.from_design(
            design, feature_mean, float(ridge), execution_device
        )
    else:
        operator = _SparseCenteredOperator.from_design(design, feature_mean, float(ridge), execution_device)
    pieces: list[Tensor] = []
    iterations = 0
    relative_residual = 0.0
    converged = True
    sparse_batch_width = target_chunk * CG_CHUNKS_PER_SPARSE_BATCH
    for batch_start in range(0, targets.shape[1], sparse_batch_width):
        batch_stop = min(batch_start + sparse_batch_width, targets.shape[1])
        piece, completed_values, relative_values, converged_values = _pcg_chunk_batch(
            operator,
            centered[:, batch_start:batch_stop].to(execution_device),
            int(max_iterations),
            float(tolerance),
            diagonal,
            target_chunk,
        )
        pieces.append(piece.cpu())
        for local_start, (completed, relative, piece_converged) in enumerate(
            zip(completed_values, relative_values, converged_values, strict=True)
        ):
            start = batch_start + local_start * target_chunk
            stop = min(start + target_chunk, batch_stop)
            iterations = max(iterations, completed)
            relative_residual = max(relative_residual, relative)
            converged = converged and piece_converged
            if progress is not None:
                progress(
                    f"targets={start}:{stop} iterations={completed} "
                    f"relative_residual={relative:.3e} converged={piece_converged}"
                )
    return CategoricalFit(
        coefficient=torch.cat(pieces, dim=1),
        target_mean=target_mean,
        feature_mean=feature_mean,
        ridge=float(ridge),
        solver="matrix_free_jacobi_pcg_float64",
        iterations=iterations,
        relative_residual=relative_residual,
        converged=converged,
        feature_count=design.features,
        active_features_per_sample=design.groups,
        execution_device=str(execution_device),
    )


def mixed_radix_keys(codes: Tensor, rows: int) -> Tensor:
    codes = _validate_codes(codes, rows)
    if rows**codes.shape[1] > torch.iinfo(torch.int64).max:
        raise ValueError("joint tuple does not fit in an int64 mixed-radix key")
    key = torch.zeros(codes.shape[0], dtype=torch.int64)
    for table in range(codes.shape[1]):
        key.mul_(rows).add_(codes[:, table])
    return key


@dataclass(frozen=True)
class JointMeanResult:
    prediction: Tensor
    seen: Tensor
    requested_cells: int
    observed_requested_cells: int
    matched_fit_samples: int

    @property
    def unseen_fraction(self) -> float:
        return float((~self.seen).to(torch.float64).mean()) if self.seen.numel() else 0.0


def joint_mean_predict(
    fit_codes: Tensor,
    fit_targets: Tensor,
    evaluation_codes: Tensor,
    rows: int,
    *,
    global_mean: Tensor | None = None,
) -> JointMeanResult:
    """Fit requested joint-cell means and use the fit mean for unseen cells."""

    fit_codes = _validate_codes(fit_codes, rows)
    evaluation_codes = _validate_codes(evaluation_codes, rows)
    if evaluation_codes.shape[1] != fit_codes.shape[1]:
        raise ValueError("fit and evaluation tuples have different widths")
    fit_targets = _validate_targets(fit_targets, fit_codes.shape[0])
    if global_mean is None:
        global_mean = fit_targets.mean(dim=0)
    global_mean = global_mean.detach().to(device="cpu", dtype=torch.float64)
    if global_mean.shape != (fit_targets.shape[1],):
        raise ValueError("global mean shape mismatch")
    requested = torch.unique(mixed_radix_keys(evaluation_codes, rows), sorted=True)
    fit_key = mixed_radix_keys(fit_codes, rows)
    location = torch.searchsorted(requested, fit_key)
    matched = location < requested.numel()
    if bool(matched.any()):
        matched_indices = matched.nonzero(as_tuple=False).flatten()
        matched[matched_indices] = requested[location[matched_indices]] == fit_key[matched_indices]
    counts = torch.zeros(requested.numel(), dtype=torch.int64)
    sums = torch.zeros(requested.numel(), fit_targets.shape[1], dtype=torch.float64)
    if bool(matched.any()):
        selected = location[matched]
        counts.index_add_(0, selected, torch.ones_like(selected))
        sums.index_add_(0, selected, fit_targets[matched])
    means = global_mean.expand(requested.numel(), -1).clone()
    observed = counts > 0
    means[observed] = sums[observed] / counts[observed, None]
    evaluation_key = mixed_radix_keys(evaluation_codes, rows)
    evaluation_location = torch.searchsorted(requested, evaluation_key)
    if bool((evaluation_location >= requested.numel()).any()):
        raise RuntimeError("evaluation key was absent from its own requested set")
    seen = observed[evaluation_location]
    prediction = global_mean.expand(evaluation_codes.shape[0], -1).clone()
    prediction[seen] = means[evaluation_location[seen]]
    return JointMeanResult(
        prediction=prediction,
        seen=seen,
        requested_cells=int(requested.numel()),
        observed_requested_cells=int(observed.sum()),
        matched_fit_samples=int(matched.sum()),
    )


def explained_variance(target: Tensor, prediction: Tensor, global_fit_mean: Tensor) -> dict[str, float | int]:
    target = _validate_targets(target, target.shape[0])
    prediction = _validate_targets(prediction, target.shape[0])
    global_fit_mean = global_fit_mean.detach().to(device="cpu", dtype=torch.float64)
    if prediction.shape != target.shape or global_fit_mean.shape != (target.shape[1],):
        raise ValueError("energy metric shapes do not match")
    baseline_sse = (target - global_fit_mean).square().sum()
    model_sse = (target - prediction).square().sum()
    if float(baseline_sse) <= 0:
        raise ValueError("target has no held variance relative to the fit mean")
    return {
        "samples": target.shape[0],
        "target_dimensions": target.shape[1],
        "baseline_sse": float(baseline_sse),
        "model_sse": float(model_sse),
        "explained_variance": float(1.0 - model_sse / baseline_sse),
    }


def rademacher_jl(input_dim: int, output_dim: int = JL_DIMENSION, *, seed: int) -> Tensor:
    if input_dim < 1 or output_dim < 1:
        raise ValueError("JL dimensions must be positive")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    signs = 2 * torch.randint(0, 2, (input_dim, output_dim), generator=generator, dtype=torch.int8) - 1
    return signs.to(torch.float64) / math.sqrt(output_dim)


def project_target(target: Tensor, *, dimension: int = JL_DIMENSION, seed: int) -> Tensor:
    target = _require_matrix("target", target.detach().to(device="cpu", dtype=torch.float64))
    return target @ rademacher_jl(target.shape[1], dimension, seed=seed)


@dataclass(frozen=True)
class SpectrumResult:
    v1: float
    v2: float
    vt: float | None
    first_order: float
    second_order: float
    higher_order: float | None
    v1_audit: dict[str, object]
    v2_audit: dict[str, object]
    joint_audit: dict[str, object] | None


def estimate_spectrum(
    fit_codes: Tensor,
    fit_targets: Tensor,
    evaluation_codes: Tensor,
    evaluation_targets: Tensor,
    rows: int,
    pairings: Sequence[tuple[int, int]],
    *,
    ridge: float = DEFAULT_RIDGE,
    cg_iterations: int = DEFAULT_CG_ITERATIONS,
    cg_tolerance: float = DEFAULT_CG_TOLERANCE,
    cg_target_chunk: int = DEFAULT_CG_TARGET_CHUNK,
    include_joint: bool = False,
) -> SpectrumResult:
    fit_codes = _validate_codes(fit_codes, rows)
    evaluation_codes = _validate_codes(evaluation_codes, rows)
    fit_targets = _validate_targets(fit_targets, fit_codes.shape[0])
    evaluation_targets = _validate_targets(evaluation_targets, evaluation_codes.shape[0])
    v1_fit = fit_v1_count_ridge(fit_codes, fit_targets, rows, ridge=ridge)
    v1_design = categorical_design(evaluation_codes, rows)
    v1_metric = explained_variance(evaluation_targets, v1_fit.predict(v1_design), v1_fit.target_mean)
    v2_fit = fit_v2_sparse_cg(
        fit_codes,
        fit_targets,
        rows,
        pairings,
        ridge=ridge,
        max_iterations=cg_iterations,
        tolerance=cg_tolerance,
        target_chunk=cg_target_chunk,
    )
    v2_design = categorical_design(evaluation_codes, rows, pairings)
    v2_metric = explained_variance(evaluation_targets, v2_fit.predict(v2_design), v2_fit.target_mean)
    vt: float | None = None
    joint_audit: dict[str, object] | None = None
    if include_joint:
        joint = joint_mean_predict(fit_codes, fit_targets, evaluation_codes, rows, global_mean=v1_fit.target_mean)
        joint_metric = explained_variance(evaluation_targets, joint.prediction, v1_fit.target_mean)
        vt = float(joint_metric["explained_variance"])
        joint_audit = {
            **joint_metric,
            "requested_cells": joint.requested_cells,
            "observed_requested_cells": joint.observed_requested_cells,
            "matched_fit_samples": joint.matched_fit_samples,
            "unseen_evaluation_fraction": joint.unseen_fraction,
            "fallback": "global_fit_mean",
        }
    v1 = float(v1_metric["explained_variance"])
    v2 = float(v2_metric["explained_variance"])
    return SpectrumResult(
        v1=v1,
        v2=v2,
        vt=vt,
        first_order=v1,
        second_order=v2 - v1,
        higher_order=None if vt is None else vt - v2,
        v1_audit={**v1_metric, **v1_fit.audit()},
        v2_audit={**v2_metric, **v2_fit.audit(), "pairings": [list(pair) for pair in pairings]},
        joint_audit=joint_audit,
    )


def _balanced_binary_codes(repeats: int = 64) -> Tensor:
    values = torch.arange(16, dtype=torch.int64)
    bits = ((values[:, None] >> torch.arange(4)) & 1).to(torch.int64)
    return bits.repeat(repeats, 1)


def planted_validation(
    *,
    ridge: float = DEFAULT_RIDGE,
    cg_tolerance: float = DEFAULT_CG_TOLERANCE,
    cg_iterations: int = DEFAULT_CG_ITERATIONS,
) -> dict[str, object]:
    """Recover planted first-, second-, and fourth-order spectra."""

    codes = _balanced_binary_codes()
    pairing = adjacent_pairing(4)
    signs = 2.0 * codes.to(torch.float64) - 1.0
    targets = {
        "pure_first_order": signs[:, :1],
        "pure_second_order": (signs[:, 0] * signs[:, 1]).unsqueeze(1),
        "parity_four": signs.prod(dim=1, keepdim=True),
    }
    spectra: dict[str, object] = {}
    for name, target in targets.items():
        spectra[name] = asdict(
            estimate_spectrum(
                codes,
                target,
                codes,
                target,
                2,
                pairing,
                ridge=ridge,
                cg_iterations=cg_iterations,
                cg_tolerance=cg_tolerance,
                include_joint=True,
            )
        )

    generator = torch.Generator(device="cpu").manual_seed(71_003)
    correlated = torch.randint(0, 2, (8192, 4), generator=generator)
    copy = torch.rand(8192, generator=generator) < 0.85
    correlated[copy, 1] = correlated[copy, 0]
    correlated_signs = 2.0 * correlated.to(torch.float64) - 1.0
    correlated_target = (correlated_signs[:, 0] + 0.5 * correlated_signs[:, 2] * correlated_signs[:, 3]).unsqueeze(1)
    correlated_result = estimate_spectrum(
        correlated[:4096],
        correlated_target[:4096],
        correlated[4096:],
        correlated_target[4096:],
        2,
        pairing,
        ridge=ridge,
        cg_iterations=cg_iterations,
        cg_tolerance=cg_tolerance,
        include_joint=True,
    )
    spectra["correlated_codes"] = asdict(correlated_result)

    tolerance = {
        "pure_component_absolute": 2e-4,
        "zero_leakage_absolute": 2e-4,
        "correlated_monotonic_slack": 2e-3,
        "solver_relative_residual": cg_tolerance,
    }
    first = spectra["pure_first_order"]
    second = spectra["pure_second_order"]
    parity = spectra["parity_four"]
    corr = spectra["correlated_codes"]
    checks = {
        "pure_first_v1": abs(first["v1"] - 1.0) <= tolerance["pure_component_absolute"],
        "pure_first_v2_increment": abs(first["second_order"]) <= tolerance["zero_leakage_absolute"],
        "pure_second_v1_zero": abs(second["v1"]) <= tolerance["zero_leakage_absolute"],
        "pure_second_v2": abs(second["v2"] - 1.0) <= tolerance["pure_component_absolute"],
        "parity_v1_zero": abs(parity["v1"]) <= tolerance["zero_leakage_absolute"],
        "parity_v2_zero": abs(parity["v2"]) <= tolerance["zero_leakage_absolute"],
        "parity_joint": parity["vt"] >= 0.999,
        "parity_top_order_fraction": parity["higher_order"] >= 0.95,
        "correlated_bounded": -tolerance["correlated_monotonic_slack"] <= corr["v1"] <= 1.0 + tolerance["correlated_monotonic_slack"]
        and -tolerance["correlated_monotonic_slack"] <= corr["v2"] <= 1.0 + tolerance["correlated_monotonic_slack"]
        and -tolerance["correlated_monotonic_slack"] <= corr["vt"] <= 1.0 + tolerance["correlated_monotonic_slack"],
        "correlated_monotone": corr["v1"] <= corr["v2"] + tolerance["correlated_monotonic_slack"]
        and corr["v2"] <= corr["vt"] + tolerance["correlated_monotonic_slack"],
        "all_cg_converged": all(value["v2_audit"]["converged"] for value in spectra.values()),
    }
    return {
        "schema": PLANTED_SCHEMA,
        "complete": all(checks.values()),
        "protocol": {
            "dtype": "float64",
            "optimizer": None,
            "ridge": ridge,
            "cg_tolerance": cg_tolerance,
            "cg_iterations": cg_iterations,
            "pairing": [list(pair) for pair in pairing],
            "tolerance": tolerance,
        },
        "spectra": spectra,
        "checks": checks,
    }


def _tensor_sha256(value: Tensor) -> str:
    contiguous = value.detach().to(device="cpu").contiguous()
    return hashlib.sha256(contiguous.numpy().tobytes()).hexdigest()


def _path_metadata(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _route_health(codes: Tensor, rows: int) -> dict[str, object]:
    codes = _validate_codes(codes, rows)
    entropies: list[float] = []
    observed: list[int] = []
    maximum: list[float] = []
    for table in range(codes.shape[1]):
        counts = torch.bincount(codes[:, table], minlength=rows).to(torch.float64)
        probability = counts[counts > 0] / counts.sum()
        entropies.append(float(-(probability * probability.log2()).sum()))
        observed.append(int((counts > 0).sum()))
        maximum.append(float(counts.max() / counts.sum()))
    return {
        "samples": codes.shape[0],
        "entropy_bits_mean": sum(entropies) / len(entropies),
        "entropy_bits_min": min(entropies),
        "entropy_bits_max": max(entropies),
        "observed_rows_mean": sum(observed) / len(observed),
        "observed_rows_min": min(observed),
        "observed_rows_max": max(observed),
        "maximum_cell_mass_mean": sum(maximum) / len(maximum),
    }


@dataclass
class FrozenZipfRouter:
    family: str
    seed: int
    rows: int
    tables: int
    route: Callable[[Tensor], Tensor]
    source: dict[str, object]
    model: torch.nn.Module | None = None
    sentinel: dict[str, Tensor] | None = None

    def verify_unchanged(self) -> dict[str, object]:
        if self.model is None or self.sentinel is None:
            return {"tensor_count": 0, "all_equal": True, "mismatch": []}
        after = self.model.state_dict()
        mismatch = [name for name, before in self.sentinel.items() if name not in after or not torch.equal(before, after[name].detach().cpu())]
        return {"tensor_count": len(self.sentinel), "all_equal": not mismatch, "mismatch": mismatch}


def _load_zipf_router(
    family: str,
    seed: int,
    device: torch.device,
    *,
    canonical_dir: Path,
    walsh_dir: Path,
    walsh_reference_dir: Path,
    independent_dir: Path,
) -> FrozenZipfRouter:
    if family not in ZIPF_FAMILIES or seed not in (0, 1, 2):
        raise ValueError("unregistered Zipf family or seed")
    if family == "canonical":
        run_key = f"lut-d32-t32-c6-a1p0-e1p0-wd0p0-s{seed}"
        result_path = canonical_dir / "runs" / f"{run_key}.json"
        result = json.loads(result_path.read_text())
        if result.get("schema") != "zipf-canonical-pclut-capacity-law-run-v2" or result.get("complete") is not True:
            raise RuntimeError("invalid canonical source result")
        config = result["config"]
        if any(config[key] != value for key, value in {"family": "lut", "model_dim": 32, "tables": 32, "comparisons": 6, "seed": seed}.items()):
            raise RuntimeError("canonical source is outside the registered D32/T32/C6 slice")
        anchors = torch.tensor(result["route"]["encoder_anchors"], dtype=torch.int64, device=device)
        thresholds = torch.tensor(result["route"]["encoder_thresholds"], dtype=torch.float32, device=device)
        powers = (2 ** torch.arange(6, dtype=torch.int64, device=device)).view(1, 1, -1)

        @torch.no_grad()
        def route(x: Tensor) -> Tensor:
            margins = x[:, anchors[..., 0]] - x[:, anchors[..., 1]] - thresholds.unsqueeze(0)
            return ((margins > 0).to(torch.int64) * powers).sum(dim=-1)

        return FrozenZipfRouter(
            family,
            seed,
            64,
            32,
            route,
            {
                "run_key": run_key,
                "result": _path_metadata(result_path),
                "schema": result["schema"],
                "stored_validation_encoder_entropy": result["validation"]["encoder_route_entropy_bits_mean"],
                "stored_validation_encoder_observed_rows": result["validation"]["encoder_observed_rows_mean"],
            },
        )

    if family == "walsh":
        from tropnn.tools.zipf_groupsum_walsh_pair_merge_bridge import _load_checkpointed_source

        result, model, paths = _load_checkpointed_source(walsh_dir, walsh_reference_dir, seed, device)

        @torch.no_grad()
        def route(x: Tensor) -> Tensor:
            return model.encoder.route(x).indices

        return FrozenZipfRouter(
            family,
            seed,
            64,
            32,
            route,
            {
                "run_key": result["run_key"],
                "result": paths["result"],
                "reference": paths["reference"],
                "checkpoint": result["checkpoint"],
                "schema": result["schema"],
                "stored_route_health": result["route_health"]["encoder"],
            },
            model=model,
            sentinel={name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
        )

    from tropnn.tools.zipf_groupsum_fixed_recognizer_controls import _load_source

    run_key = f"independent-group-sum-d32-t32-c6-g512-a1p0-e1p0-wd0p0-s{seed}"
    result, model = _load_source(independent_dir, run_key, device)

    @torch.no_grad()
    def route(x: Tensor) -> Tensor:
        return model.encoder.route(x).indices

    result_path = independent_dir / "runs" / f"{run_key}.json"
    checkpoint_path = independent_dir / "checkpoints" / f"{run_key}.pt"
    return FrozenZipfRouter(
        family,
        seed,
        64,
        32,
        route,
        {
            "run_key": run_key,
            "result": _path_metadata(result_path),
            "checkpoint": _path_metadata(checkpoint_path),
            "schema": result["schema"],
            "stored_route_health": result["validation"]["encoder_route_health"],
        },
        model=model,
        sentinel={name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
    )


@torch.no_grad()
def _capture_route_codes(
    router: FrozenZipfRouter,
    probabilities: Tensor,
    *,
    samples: int,
    batch_size: int,
    generator_seed: int,
) -> Tensor:
    codes = torch.empty(samples, router.tables, dtype=torch.uint8)
    generator = torch.Generator(device=probabilities.device).manual_seed(generator_seed)
    from tropnn.tools.zipf_addressing_capacity_law import sample_liu_gore_batch

    for start in range(0, samples, batch_size):
        stop = min(start + batch_size, samples)
        x = sample_liu_gore_batch(probabilities, stop - start, generator=generator)
        batch_codes = router.route(x)
        if batch_codes.shape != (stop - start, router.tables):
            raise RuntimeError("frozen router returned the wrong code shape")
        codes[start:stop] = batch_codes.to(device="cpu", dtype=torch.uint8)
    return codes


@torch.no_grad()
def _capture_zipf_split(
    router: FrozenZipfRouter,
    probabilities: Tensor,
    sketches: Tensor,
    *,
    samples: int,
    batch_size: int,
    generator_seed: int,
) -> tuple[Tensor, Tensor]:
    codes = torch.empty(samples, router.tables, dtype=torch.uint8)
    targets = torch.empty(samples, sketches.shape[1], dtype=torch.float64)
    generator = torch.Generator(device=probabilities.device).manual_seed(generator_seed)
    from tropnn.tools.zipf_addressing_capacity_law import sample_liu_gore_batch

    for start in range(0, samples, batch_size):
        stop = min(start + batch_size, samples)
        x = sample_liu_gore_batch(probabilities, stop - start, generator=generator)
        batch_codes = router.route(x)
        codes[start:stop] = batch_codes.to(device="cpu", dtype=torch.uint8)
        targets[start:stop] = (x.to(torch.float64) @ sketches).cpu()
    return codes, targets


def _zipf_route_replay(
    router: FrozenZipfRouter,
    probabilities: Tensor,
    *,
    batch_size: int,
) -> dict[str, object]:
    if router.family == "walsh":
        seed, samples = 60_013, 4096
    elif router.family == "canonical":
        seed, samples = 70_001, 4096
    else:
        # The formal evaluator records route health from its first eight
        # 512-example batches even though loss metrics use all 16,384 rows.
        seed, samples = 70_001, 4096
    # All source route-health records were generated with 512-example batches.
    # The sampler draws activity and amplitude separately, so changing batch
    # size changes the seeded stream rather than merely its chunking.
    if batch_size != 512:
        raise ValueError("formal Zipf replay requires batch_size=512")
    codes = _capture_route_codes(router, probabilities, samples=samples, batch_size=batch_size, generator_seed=seed)
    measured = _route_health(codes, router.rows)
    if router.family == "canonical":
        differences = {
            "entropy_bits_mean": abs(measured["entropy_bits_mean"] - router.source["stored_validation_encoder_entropy"]),
            "observed_rows_mean": abs(measured["observed_rows_mean"] - router.source["stored_validation_encoder_observed_rows"]),
        }
    else:
        stored = router.source["stored_route_health"]
        differences = {key: abs(measured[key] - stored[key]) for key in stored}
    return {
        "generator_seed": seed,
        "samples": samples,
        "measured": measured,
        "stored_absolute_differences": differences,
        "exact": all(value == 0 for value in differences.values()),
    }


def _sliced_metric(target: Tensor, prediction: Tensor, fit_mean: Tensor, start: int, stop: int) -> dict[str, float | int]:
    return explained_variance(target[:, start:stop], prediction[:, start:stop], fit_mean[start:stop])


def _spectrum_payload(
    v1: dict[str, float | int],
    v2: dict[str, float | int],
    joint: dict[str, float | int] | None = None,
) -> dict[str, object]:
    first = float(v1["explained_variance"])
    second_cumulative = float(v2["explained_variance"])
    result: dict[str, object] = {
        "v1": first,
        "v2": second_cumulative,
        "first_order": first,
        "second_order": second_cumulative - first,
        "v1_metric": v1,
        "v2_metric": v2,
    }
    if joint is not None:
        total = float(joint["explained_variance"])
        result.update({"vt": total, "higher_order": total - second_cumulative, "joint_metric": joint})
    return result


def _mean_sem(values: Sequence[float]) -> dict[str, object]:
    tensor = torch.tensor(values, dtype=torch.float64)
    sem = 0.0 if tensor.numel() < 2 else float(tensor.std(unbiased=True) / math.sqrt(tensor.numel()))
    return {"values": list(values), "mean": float(tensor.mean()), "sem": sem}


def _summarize_zipf_sketches(sketches: Sequence[dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for split in ("validation", "test"):
        split_summary: dict[str, object] = {}
        for scope, metrics in (
            ("t32_adjacent", ("v1", "v2", "first_order", "second_order")),
            ("t32_random", ("v1", "v2", "first_order", "second_order")),
            ("t4", ("v1", "v2", "vt", "first_order", "second_order", "higher_order")),
        ):
            split_summary[scope] = {
                metric: _mean_sem([float(sketch[split][scope][metric]) for sketch in sketches]) for metric in metrics
            }
        result[split] = split_summary
    return result


def run_zipf_spectrum(
    family: str,
    seed: int,
    output_dir: Path,
    *,
    device: torch.device,
    batch_size: int,
    fit_samples: int,
    eval_samples: int,
    ridge: float,
    cg_iterations: int,
    cg_tolerance: float,
    cg_target_chunk: int,
    canonical_dir: Path,
    walsh_dir: Path,
    walsh_reference_dir: Path,
    independent_dir: Path,
    preregistration: Path,
) -> dict[str, object]:
    output_path = output_dir / "runs" / f"{family}-s{seed}.json"
    if output_path.exists():
        existing = json.loads(output_path.read_text())
        if existing.get("schema") == SCHEMA and existing.get("complete") is True:
            return existing
        raise FileExistsError(f"incomplete or invalid result already exists: {output_path}")
    if fit_samples != ZIPF_FIT_SAMPLES or eval_samples != ZIPF_EVAL_SAMPLES or batch_size != 512:
        raise ValueError("formal Zipf run must use the preregistered sample counts")
    started = time.perf_counter()
    torch.set_grad_enabled(False)
    router = _load_zipf_router(
        family,
        seed,
        device,
        canonical_dir=canonical_dir,
        walsh_dir=walsh_dir,
        walsh_reference_dir=walsh_reference_dir,
        independent_dir=independent_dir,
    )
    from tropnn.tools.zipf_addressing_capacity_law import zipf_probabilities

    probabilities = zipf_probabilities(1024, 1.0, 1.0, device=device)
    route_replay = _zipf_route_replay(router, probabilities, batch_size=batch_size)
    if not route_replay["exact"]:
        raise RuntimeError(f"{family} seed {seed} failed frozen route-health replay")
    print(f"family={family} seed={seed} phase=route-replay status=exact", flush=True)
    sketch_matrix = torch.cat([rademacher_jl(1024, JL_DIMENSION, seed=jl_seed) for jl_seed in JL_SEEDS], dim=1).to(device)
    fit_codes, fit_targets = _capture_zipf_split(
        router,
        probabilities,
        sketch_matrix,
        samples=fit_samples,
        batch_size=batch_size,
        generator_seed=FIT_GENERATOR_SEED,
    )
    print(f"family={family} seed={seed} phase=fit-capture samples={fit_samples}", flush=True)
    validation_codes, validation_targets = _capture_zipf_split(
        router,
        probabilities,
        sketch_matrix,
        samples=eval_samples,
        batch_size=batch_size,
        generator_seed=VALIDATION_GENERATOR_SEED,
    )
    test_codes, test_targets = _capture_zipf_split(
        router,
        probabilities,
        sketch_matrix,
        samples=eval_samples,
        batch_size=batch_size,
        generator_seed=TEST_GENERATOR_SEED,
    )
    print(f"family={family} seed={seed} phase=evaluation-capture samples={2 * eval_samples}", flush=True)
    del sketch_matrix
    source_unchanged = router.verify_unchanged()
    if not source_unchanged["all_equal"]:
        raise RuntimeError("frozen source changed during code capture")

    adjacent = adjacent_pairing(router.tables)
    random = random_pairing(router.tables, 51_000 + seed)
    t4_pairing = adjacent_pairing(4)
    print(f"family={family} seed={seed} phase=v1-fit", flush=True)
    v1_fit = fit_v1_count_ridge(fit_codes, fit_targets, router.rows, ridge=ridge, solver_device=device)
    if v1_fit.relative_residual > 1e-10:
        raise RuntimeError(f"V1 solve residual is too large: {v1_fit.relative_residual}")
    print(f"family={family} seed={seed} phase=v2-adjacent-fit", flush=True)
    adjacent_fit = fit_v2_sparse_cg(
        fit_codes,
        fit_targets,
        router.rows,
        adjacent,
        ridge=ridge,
        max_iterations=cg_iterations,
        tolerance=cg_tolerance,
        target_chunk=cg_target_chunk,
        solver_device=device,
        progress=lambda message: print(f"family={family} seed={seed} solver=adjacent {message}", flush=True),
    )
    print(f"family={family} seed={seed} phase=v2-random-fit", flush=True)
    random_fit = fit_v2_sparse_cg(
        fit_codes,
        fit_targets,
        router.rows,
        random,
        ridge=ridge,
        max_iterations=cg_iterations,
        tolerance=cg_tolerance,
        target_chunk=cg_target_chunk,
        solver_device=device,
        progress=lambda message: print(f"family={family} seed={seed} solver=random {message}", flush=True),
    )
    fit_codes_t4 = fit_codes[:, :4]
    print(f"family={family} seed={seed} phase=t4-fits", flush=True)
    t4_fit = fit_v1_count_ridge(fit_codes_t4, fit_targets, router.rows, ridge=ridge, solver_device=device)
    t4_pair_fit = fit_v2_sparse_cg(
        fit_codes_t4,
        fit_targets,
        router.rows,
        t4_pairing,
        ridge=ridge,
        max_iterations=cg_iterations,
        tolerance=cg_tolerance,
        target_chunk=cg_target_chunk,
        solver_device=device,
        progress=lambda message: print(f"family={family} seed={seed} solver=t4 {message}", flush=True),
    )
    if not (adjacent_fit.converged and random_fit.converged and t4_pair_fit.converged):
        raise RuntimeError("one or more registered V2 CG solves did not converge")

    split_values: dict[str, dict[str, object]] = {}
    for split, codes, targets in (
        ("validation", validation_codes, validation_targets),
        ("test", test_codes, test_targets),
    ):
        marginal_prediction = v1_fit.predict(categorical_design(codes, router.rows))
        adjacent_prediction = adjacent_fit.predict(categorical_design(codes, router.rows, adjacent))
        random_prediction = random_fit.predict(categorical_design(codes, router.rows, random))
        codes_t4 = codes[:, :4]
        t4_prediction = t4_fit.predict(categorical_design(codes_t4, router.rows))
        t4_pair_prediction = t4_pair_fit.predict(categorical_design(codes_t4, router.rows, t4_pairing))
        joint = joint_mean_predict(fit_codes_t4, fit_targets, codes_t4, router.rows, global_mean=t4_fit.target_mean)
        split_values[split] = {
            "targets": targets,
            "marginal": marginal_prediction,
            "adjacent": adjacent_prediction,
            "random": random_prediction,
            "t4_marginal": t4_prediction,
            "t4_pair": t4_pair_prediction,
            "joint": joint,
        }

    sketches: list[dict[str, object]] = []
    numerical_checks: list[bool] = []
    for sketch_index, jl_seed in enumerate(JL_SEEDS):
        start = sketch_index * JL_DIMENSION
        stop = start + JL_DIMENSION
        sketch: dict[str, object] = {"jl_seed": jl_seed}
        for split in ("validation", "test"):
            values = split_values[split]
            targets = values["targets"]
            assert isinstance(targets, Tensor)
            v1_metric = _sliced_metric(targets, values["marginal"], v1_fit.target_mean, start, stop)
            adjacent_metric = _sliced_metric(targets, values["adjacent"], adjacent_fit.target_mean, start, stop)
            random_metric = _sliced_metric(targets, values["random"], random_fit.target_mean, start, stop)
            t4_v1_metric = _sliced_metric(targets, values["t4_marginal"], t4_fit.target_mean, start, stop)
            t4_v2_metric = _sliced_metric(targets, values["t4_pair"], t4_pair_fit.target_mean, start, stop)
            joint = values["joint"]
            assert isinstance(joint, JointMeanResult)
            joint_metric = _sliced_metric(targets, joint.prediction, t4_fit.target_mean, start, stop)
            joint_metric.update(
                {
                    "requested_cells": joint.requested_cells,
                    "observed_requested_cells": joint.observed_requested_cells,
                    "matched_fit_samples": joint.matched_fit_samples,
                    "unseen_evaluation_fraction": joint.unseen_fraction,
                    "fallback": "global_fit_mean",
                }
            )
            adjacent_spectrum = _spectrum_payload(v1_metric, adjacent_metric)
            random_spectrum = _spectrum_payload(v1_metric, random_metric)
            t4_spectrum = _spectrum_payload(t4_v1_metric, t4_v2_metric, joint_metric)
            numerical_checks.extend(
                (
                    float(adjacent_spectrum["v2"]) + 0.005 >= float(adjacent_spectrum["v1"]),
                    float(t4_spectrum["v2"]) + 0.005 >= float(t4_spectrum["v1"]),
                    float(t4_spectrum["vt"]) + 0.005 >= float(t4_spectrum["v2"]),
                )
            )
            sketch[split] = {"t32_adjacent": adjacent_spectrum, "t32_random": random_spectrum, "t4": t4_spectrum}
        sketches.append(sketch)
    summary = _summarize_zipf_sketches(sketches)
    sem_values = [
        summary[split][scope][metric]["sem"]
        for split in ("validation", "test")
        for scope, metrics in (("t32_adjacent", ("v1", "v2")), ("t4", ("v1", "v2", "vt")))
        for metric in metrics
    ]
    checks = {
        "route_health_exact_replay": route_replay["exact"],
        "source_state_unchanged": source_unchanged["all_equal"],
        "v1_relative_residual_at_most_1e_10": v1_fit.relative_residual <= 1e-10 and t4_fit.relative_residual <= 1e-10,
        "all_cg_converged": adjacent_fit.converged and random_fit.converged and t4_pair_fit.converged,
        "all_cg_relative_residual_at_most_tolerance": max(
            adjacent_fit.relative_residual, random_fit.relative_residual, t4_pair_fit.relative_residual
        )
        <= cg_tolerance,
        "nested_within_0p005": all(numerical_checks),
        "primary_sketch_sem_at_most_0p025": max(sem_values) <= 0.025,
    }
    payload = {
        "schema": SCHEMA,
        "complete": all(checks.values()),
        "family": family,
        "seed": seed,
        "protocol": {
            "fit_samples": fit_samples,
            "evaluation_samples_per_split": eval_samples,
            "fit_generator_seed": FIT_GENERATOR_SEED,
            "validation_generator_seed": VALIDATION_GENERATOR_SEED,
            "test_generator_seed": TEST_GENERATOR_SEED,
            "tables": router.tables,
            "rows": router.rows,
            "comparisons": 6,
            "target": "liu_gore_x_rademacher_jl64",
            "jl_seeds": list(JL_SEEDS),
            "ridge": ridge,
            "cg_iterations": cg_iterations,
            "cg_tolerance": cg_tolerance,
            "cg_target_chunk": cg_target_chunk,
            "adjacent_pairing": [list(pair) for pair in adjacent],
            "random_pairing": [list(pair) for pair in random],
            "t4_tables": [0, 1, 2, 3],
            "dtype": "float64_triton_cuda_v1_v2_cpu_scoring",
            "optimizer": None,
            "device_for_code_extraction_and_jl_projection": str(device),
        },
        "preregistration": {**_path_metadata(preregistration), "sha256": hashlib.sha256(preregistration.read_bytes()).hexdigest()},
        "source": router.source,
        "source_replay": route_replay,
        "source_state_verification": source_unchanged,
        "captures": {
            "fit_code_sha256": _tensor_sha256(fit_codes),
            "validation_code_sha256": _tensor_sha256(validation_codes),
            "test_code_sha256": _tensor_sha256(test_codes),
            "fit_route_health": _route_health(fit_codes, router.rows),
            "validation_route_health": _route_health(validation_codes, router.rows),
            "test_route_health": _route_health(test_codes, router.rows),
        },
        "fit_audits": {
            "v1": v1_fit.audit(),
            "v2_adjacent": adjacent_fit.audit(),
            "v2_random": random_fit.audit(),
            "t4_v1": t4_fit.audit(),
            "t4_v2": t4_pair_fit.audit(),
        },
        "sketches": sketches,
        "summary": summary,
        "checks": checks,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_exclusive(output_path, payload)
    print(f"family={family} seed={seed} phase=sealed path={output_path}", flush=True)
    if payload["complete"] is not True:
        raise RuntimeError(f"numerical gate failed after preserving result: {output_path}")
    return payload


def _fit_zipf_captures(
    router: FrozenZipfRouter,
    fit_codes: Tensor,
    fit_targets: Tensor,
    validation_codes: Tensor,
    validation_targets: Tensor,
    test_codes: Tensor,
    test_targets: Tensor,
    *,
    ridge: float,
    cg_iterations: int,
    cg_tolerance: float,
    cg_target_chunk: int,
    solver_device: torch.device,
    progress_prefix: str,
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object], dict[str, bool]]:
    adjacent = adjacent_pairing(router.tables)
    random = random_pairing(router.tables, 51_000 + router.seed)
    t4_pairing = adjacent_pairing(4)
    print(f"{progress_prefix} phase=v1-fit", flush=True)
    v1_fit = fit_v1_count_ridge(fit_codes, fit_targets, router.rows, ridge=ridge, solver_device=solver_device)
    print(f"{progress_prefix} phase=v2-adjacent-fit", flush=True)
    adjacent_fit = fit_v2_sparse_cg(
        fit_codes,
        fit_targets,
        router.rows,
        adjacent,
        ridge=ridge,
        max_iterations=cg_iterations,
        tolerance=cg_tolerance,
        target_chunk=cg_target_chunk,
        solver_device=solver_device,
        progress=lambda message: print(f"{progress_prefix} solver=adjacent {message}", flush=True),
    )
    print(f"{progress_prefix} phase=v2-random-fit", flush=True)
    random_fit = fit_v2_sparse_cg(
        fit_codes,
        fit_targets,
        router.rows,
        random,
        ridge=ridge,
        max_iterations=cg_iterations,
        tolerance=cg_tolerance,
        target_chunk=cg_target_chunk,
        solver_device=solver_device,
        progress=lambda message: print(f"{progress_prefix} solver=random {message}", flush=True),
    )
    fit_codes_t4 = fit_codes[:, :4]
    print(f"{progress_prefix} phase=t4-fits", flush=True)
    t4_fit = fit_v1_count_ridge(fit_codes_t4, fit_targets, router.rows, ridge=ridge, solver_device=solver_device)
    t4_pair_fit = fit_v2_sparse_cg(
        fit_codes_t4,
        fit_targets,
        router.rows,
        t4_pairing,
        ridge=ridge,
        max_iterations=cg_iterations,
        tolerance=cg_tolerance,
        target_chunk=cg_target_chunk,
        solver_device=solver_device,
        progress=lambda message: print(f"{progress_prefix} solver=t4 {message}", flush=True),
    )
    split_values: dict[str, dict[str, object]] = {}
    for split, codes, targets in (
        ("validation", validation_codes, validation_targets),
        ("test", test_codes, test_targets),
    ):
        codes_t4 = codes[:, :4]
        split_values[split] = {
            "targets": targets,
            "marginal": v1_fit.predict(categorical_design(codes, router.rows)),
            "adjacent": adjacent_fit.predict(categorical_design(codes, router.rows, adjacent)),
            "random": random_fit.predict(categorical_design(codes, router.rows, random)),
            "t4_marginal": t4_fit.predict(categorical_design(codes_t4, router.rows)),
            "t4_pair": t4_pair_fit.predict(categorical_design(codes_t4, router.rows, t4_pairing)),
            "joint": joint_mean_predict(fit_codes_t4, fit_targets, codes_t4, router.rows, global_mean=t4_fit.target_mean),
        }
    sketches: list[dict[str, object]] = []
    numerical_checks: list[bool] = []
    for sketch_index, jl_seed in enumerate(JL_SEEDS):
        start = sketch_index * JL_DIMENSION
        stop = start + JL_DIMENSION
        sketch: dict[str, object] = {"jl_seed": jl_seed}
        for split in ("validation", "test"):
            values = split_values[split]
            targets = values["targets"]
            assert isinstance(targets, Tensor)
            v1_metric = _sliced_metric(targets, values["marginal"], v1_fit.target_mean, start, stop)
            adjacent_metric = _sliced_metric(targets, values["adjacent"], adjacent_fit.target_mean, start, stop)
            random_metric = _sliced_metric(targets, values["random"], random_fit.target_mean, start, stop)
            t4_v1_metric = _sliced_metric(targets, values["t4_marginal"], t4_fit.target_mean, start, stop)
            t4_v2_metric = _sliced_metric(targets, values["t4_pair"], t4_pair_fit.target_mean, start, stop)
            joint = values["joint"]
            assert isinstance(joint, JointMeanResult)
            joint_metric = _sliced_metric(targets, joint.prediction, t4_fit.target_mean, start, stop)
            joint_metric.update(
                {
                    "requested_cells": joint.requested_cells,
                    "observed_requested_cells": joint.observed_requested_cells,
                    "matched_fit_samples": joint.matched_fit_samples,
                    "unseen_evaluation_fraction": joint.unseen_fraction,
                    "fallback": "global_fit_mean",
                }
            )
            adjacent_spectrum = _spectrum_payload(v1_metric, adjacent_metric)
            random_spectrum = _spectrum_payload(v1_metric, random_metric)
            t4_spectrum = _spectrum_payload(t4_v1_metric, t4_v2_metric, joint_metric)
            numerical_checks.extend(
                (
                    float(adjacent_spectrum["v2"]) + 0.005 >= float(adjacent_spectrum["v1"]),
                    float(t4_spectrum["v2"]) + 0.005 >= float(t4_spectrum["v1"]),
                    float(t4_spectrum["vt"]) + 0.005 >= float(t4_spectrum["v2"]),
                )
            )
            sketch[split] = {"t32_adjacent": adjacent_spectrum, "t32_random": random_spectrum, "t4": t4_spectrum}
        sketches.append(sketch)
    summary = _summarize_zipf_sketches(sketches)
    sem_values = [
        summary[split][scope][metric]["sem"]
        for split in ("validation", "test")
        for scope, metrics in (("t32_adjacent", ("v1", "v2")), ("t4", ("v1", "v2", "vt")))
        for metric in metrics
    ]
    checks = {
        "v1_relative_residual_at_most_1e_10": v1_fit.relative_residual <= 1e-10 and t4_fit.relative_residual <= 1e-10,
        "all_cg_converged": adjacent_fit.converged and random_fit.converged and t4_pair_fit.converged,
        "all_cg_relative_residual_at_most_tolerance": max(
            adjacent_fit.relative_residual, random_fit.relative_residual, t4_pair_fit.relative_residual
        )
        <= cg_tolerance,
        "nested_within_0p005": all(numerical_checks),
        "primary_sketch_sem_at_most_0p025": max(sem_values) <= 0.025,
    }
    audits = {
        "v1": v1_fit.audit(),
        "v2_adjacent": adjacent_fit.audit(),
        "v2_random": random_fit.audit(),
        "t4_v1": t4_fit.audit(),
        "t4_v2": t4_pair_fit.audit(),
    }
    return audits, sketches, summary, checks


def _fixed_pair_anchors(features: int, tables: int, comparisons: int, *, seed: int) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    first = torch.randint(0, features, (tables, comparisons), generator=generator)
    second = torch.randint(0, features - 1, (tables, comparisons), generator=generator)
    second += second >= first
    return torch.stack((first, second), dim=-1)


def _normalized_hd(x: Tensor, transforms: Sequence[torch.nn.Module]) -> Tensor:
    current = x
    scale = math.sqrt(x.shape[-1])
    for transform in transforms:
        current = transform(current) / scale
    return current


@torch.no_grad()
def _build_hd_router(
    rounds: int,
    seed: int,
    device: torch.device,
    probabilities: Tensor,
    *,
    compiler_samples: int,
    batch_size: int,
) -> tuple[FrozenZipfRouter, dict[str, object]]:
    if rounds not in (1, 2, 3) or compiler_samples != 65_536 or batch_size != 512:
        raise ValueError("HD construction is outside the preregistered matrix")
    from tropnn.layers.accumulation import WalshButterfly
    from tropnn.tools.zipf_addressing_capacity_law import sample_liu_gore_batch

    transforms = [WalshButterfly(1024, seed=61_000 + 100 * seed + index).to(device).eval() for index in range(1, rounds + 1)]
    anchors_cpu = _fixed_pair_anchors(1024, 32, 6, seed=62_000 + seed)
    anchors = anchors_cpu.to(device)
    margins = torch.empty(compiler_samples, 32, 6, dtype=torch.float32, device=device)
    generator = torch.Generator(device=device).manual_seed(FIT_GENERATOR_SEED)
    for start in range(0, compiler_samples, batch_size):
        stop = min(start + batch_size, compiler_samples)
        x = sample_liu_gore_batch(probabilities, stop - start, generator=generator)
        mixed = _normalized_hd(x, transforms)
        margins[start:stop] = mixed[:, anchors[..., 0]] - mixed[:, anchors[..., 1]]
    ordered = margins.reshape(compiler_samples, -1).sort(dim=0, stable=True).values
    midpoint = compiler_samples // 2
    thresholds = ((ordered[midpoint - 1] + ordered[midpoint]) * 0.5).reshape(32, 6)
    powers = (2 ** torch.arange(6, dtype=torch.int64, device=device)).view(1, 1, -1)

    @torch.no_grad()
    def route(x: Tensor) -> Tensor:
        mixed = _normalized_hd(x, transforms)
        route_margins = mixed[:, anchors[..., 0]] - mixed[:, anchors[..., 1]] - thresholds.unsqueeze(0)
        return ((route_margins > 0).to(torch.int64) * powers).sum(dim=-1)

    compiler_codes = ((margins > thresholds.unsqueeze(0)).to(torch.int64) * powers).sum(dim=-1).cpu()
    above = margins > thresholds.unsqueeze(0)
    equal = margins == thresholds.unsqueeze(0)
    one_fraction = above.to(torch.float64).mean(dim=0).cpu()
    tie_fraction = equal.to(torch.float64).mean(dim=0).cpu()
    below_count = (margins < thresholds.unsqueeze(0)).sum(dim=0)
    at_or_below_count = (margins <= thresholds.unsqueeze(0)).sum(dim=0)
    median_rank_bracket_valid = bool(((below_count <= midpoint) & (at_or_below_count >= midpoint)).all())
    module = torch.nn.ModuleList(transforms)
    router = FrozenZipfRouter(
        f"hd{rounds}",
        seed,
        64,
        32,
        route,
        {
            "kind": "constructed_normalized_repeated_hd",
            "rounds": rounds,
            "sign_seeds": [61_000 + 100 * seed + index for index in range(1, rounds + 1)],
            "anchor_seed": 62_000 + seed,
            "compiler_generator_seed": FIT_GENERATOR_SEED,
            "compiler_samples": compiler_samples,
        },
        model=module,
        sentinel={name: value.detach().cpu().clone() for name, value in module.state_dict().items()},
    )
    audit = {
        "anchors_sha256": _tensor_sha256(anchors_cpu),
        "thresholds_sha256": _tensor_sha256(thresholds),
        "thresholds": thresholds.cpu().tolist(),
        "sign_sha256_by_round": [_tensor_sha256(transform.signs) for transform in transforms],
        "compiler_code_sha256": _tensor_sha256(compiler_codes.to(torch.uint8)),
        "compiler_route_health": _route_health(compiler_codes, 64),
        "mean_bit_one_fraction": float(one_fraction.mean()),
        "minimum_bit_one_fraction": float(one_fraction.min()),
        "maximum_bit_one_fraction": float(one_fraction.max()),
        "maximum_bit_balance_error": float((one_fraction - 0.5).abs().max()),
        "maximum_bit_threshold_tie_fraction": float(tie_fraction.max()),
        "median_rank_bracket_valid": median_rank_bracket_valid,
        "median_rule": "stable_sort_midpoint_of_two_central_float32_margins",
    }
    return router, audit


def run_hd_spectrum(
    rounds: int,
    seed: int,
    output_dir: Path,
    *,
    device: torch.device,
    batch_size: int,
    fit_samples: int,
    eval_samples: int,
    ridge: float,
    cg_iterations: int,
    cg_tolerance: float,
    cg_target_chunk: int,
    preregistration: Path,
) -> dict[str, object]:
    family = f"hd{rounds}"
    output_path = output_dir / "runs" / f"{family}-s{seed}.json"
    if output_path.exists():
        existing = json.loads(output_path.read_text())
        if existing.get("schema") == SCHEMA and existing.get("complete") is True:
            return existing
        raise FileExistsError(f"incomplete or invalid result already exists: {output_path}")
    if fit_samples != ZIPF_FIT_SAMPLES or eval_samples != ZIPF_EVAL_SAMPLES or batch_size != 512:
        raise ValueError("formal HD run must use the preregistered sample and batch counts")
    started = time.perf_counter()
    from tropnn.tools.zipf_addressing_capacity_law import zipf_probabilities

    probabilities = zipf_probabilities(1024, 1.0, 1.0, device=device)
    router, construction = _build_hd_router(rounds, seed, device, probabilities, compiler_samples=65_536, batch_size=batch_size)
    print(f"family={family} seed={seed} phase=construction status=complete", flush=True)
    sketch_matrix = torch.cat([rademacher_jl(1024, JL_DIMENSION, seed=jl_seed) for jl_seed in JL_SEEDS], dim=1).to(device)
    fit_codes, fit_targets = _capture_zipf_split(
        router,
        probabilities,
        sketch_matrix,
        samples=fit_samples,
        batch_size=batch_size,
        generator_seed=FIT_GENERATOR_SEED,
    )
    print(f"family={family} seed={seed} phase=fit-capture samples={fit_samples}", flush=True)
    validation_codes, validation_targets = _capture_zipf_split(
        router,
        probabilities,
        sketch_matrix,
        samples=eval_samples,
        batch_size=batch_size,
        generator_seed=VALIDATION_GENERATOR_SEED,
    )
    test_codes, test_targets = _capture_zipf_split(
        router,
        probabilities,
        sketch_matrix,
        samples=eval_samples,
        batch_size=batch_size,
        generator_seed=TEST_GENERATOR_SEED,
    )
    del sketch_matrix
    source_unchanged = router.verify_unchanged()
    audits, sketches, summary, numerical_checks = _fit_zipf_captures(
        router,
        fit_codes,
        fit_targets,
        validation_codes,
        validation_targets,
        test_codes,
        test_targets,
        ridge=ridge,
        cg_iterations=cg_iterations,
        cg_tolerance=cg_tolerance,
        cg_target_chunk=cg_target_chunk,
        solver_device=device,
        progress_prefix=f"family={family} seed={seed}",
    )
    checks = {
        "source_state_unchanged": source_unchanged["all_equal"],
        "compiler_median_rank_bracket_valid": construction["median_rank_bracket_valid"],
        **numerical_checks,
    }
    payload = {
        "schema": SCHEMA,
        "complete": all(checks.values()),
        "family": family,
        "seed": seed,
        "protocol": {
            "fit_samples": fit_samples,
            "evaluation_samples_per_split": eval_samples,
            "fit_generator_seed": FIT_GENERATOR_SEED,
            "validation_generator_seed": VALIDATION_GENERATOR_SEED,
            "test_generator_seed": TEST_GENERATOR_SEED,
            "batch_size": batch_size,
            "tables": 32,
            "rows": 64,
            "comparisons": 6,
            "target": "liu_gore_x_rademacher_jl64",
            "jl_seeds": list(JL_SEEDS),
            "ridge": ridge,
            "cg_iterations": cg_iterations,
            "cg_tolerance": cg_tolerance,
            "cg_target_chunk": cg_target_chunk,
            "adjacent_pairing": [list(pair) for pair in adjacent_pairing(32)],
            "random_pairing": [list(pair) for pair in random_pairing(32, 51_000 + seed)],
            "t4_tables": [0, 1, 2, 3],
            "dtype": "float64_triton_cuda_v1_v2_cpu_scoring",
            "optimizer": None,
            "device_for_code_extraction_and_jl_projection": str(device),
        },
        "preregistration": {**_path_metadata(preregistration), "sha256": hashlib.sha256(preregistration.read_bytes()).hexdigest()},
        "source": router.source,
        "construction": construction,
        "source_state_verification": source_unchanged,
        "captures": {
            "fit_code_sha256": _tensor_sha256(fit_codes),
            "validation_code_sha256": _tensor_sha256(validation_codes),
            "test_code_sha256": _tensor_sha256(test_codes),
            "fit_route_health": _route_health(fit_codes, 64),
            "validation_route_health": _route_health(validation_codes, 64),
            "test_route_health": _route_health(test_codes, 64),
        },
        "fit_audits": audits,
        "sketches": sketches,
        "summary": summary,
        "checks": checks,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_exclusive(output_path, payload)
    print(f"family={family} seed={seed} phase=sealed path={output_path}", flush=True)
    if payload["complete"] is not True:
        raise RuntimeError(f"numerical gate failed after preserving result: {output_path}")
    return payload


@torch.no_grad()
def _capture_emnist_codes(
    x: Tensor,
    stem: torch.nn.Module,
    pq_router: torch.nn.Module,
    grid_router: torch.nn.Module,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    import torch.nn.functional as functional

    pq_codes = torch.empty(x.shape[0], 32, dtype=torch.uint8)
    grid_codes = torch.empty_like(pq_codes)
    for start in range(0, x.shape[0], batch_size):
        stop = min(start + batch_size, x.shape[0])
        features = functional.gelu(stem(x[start:stop].to(device)))
        pq_codes[start:stop] = pq_router.hard_codes(features).to(device="cpu", dtype=torch.uint8)  # type: ignore[attr-defined]
        grid_codes[start:stop] = grid_router.hard_codes(features).to(device="cpu", dtype=torch.uint8)  # type: ignore[attr-defined]
    return pq_codes, grid_codes


def _fit_emnist_codes(
    fit_codes: Tensor,
    fit_targets: Tensor,
    held_codes: Tensor,
    held_targets: Tensor,
    *,
    seed: int,
    ridge: float,
    cg_iterations: int,
    cg_tolerance: float,
    cg_target_chunk: int,
    solver_device: torch.device,
    progress_prefix: str,
) -> tuple[dict[str, object], dict[str, object], dict[str, bool]]:
    rows = 16
    adjacent = adjacent_pairing(32)
    random = random_pairing(32, 51_000 + seed)
    t4_pairing = adjacent_pairing(4)
    print(f"{progress_prefix} phase=v1-fit", flush=True)
    v1_fit = fit_v1_count_ridge(fit_codes, fit_targets, rows, ridge=ridge, solver_device=solver_device)
    print(f"{progress_prefix} phase=v2-adjacent-fit", flush=True)
    adjacent_fit = fit_v2_sparse_cg(
        fit_codes,
        fit_targets,
        rows,
        adjacent,
        ridge=ridge,
        max_iterations=cg_iterations,
        tolerance=cg_tolerance,
        target_chunk=cg_target_chunk,
        solver_device=solver_device,
        progress=lambda message: print(f"{progress_prefix} solver=adjacent {message}", flush=True),
    )
    print(f"{progress_prefix} phase=v2-random-fit", flush=True)
    random_fit = fit_v2_sparse_cg(
        fit_codes,
        fit_targets,
        rows,
        random,
        ridge=ridge,
        max_iterations=cg_iterations,
        tolerance=cg_tolerance,
        target_chunk=cg_target_chunk,
        solver_device=solver_device,
        progress=lambda message: print(f"{progress_prefix} solver=random {message}", flush=True),
    )
    fit_t4 = fit_codes[:, :4]
    held_t4 = held_codes[:, :4]
    print(f"{progress_prefix} phase=t4-fits", flush=True)
    t4_fit = fit_v1_count_ridge(fit_t4, fit_targets, rows, ridge=ridge, solver_device=solver_device)
    t4_pair_fit = fit_v2_sparse_cg(
        fit_t4,
        fit_targets,
        rows,
        t4_pairing,
        ridge=ridge,
        max_iterations=cg_iterations,
        tolerance=cg_tolerance,
        target_chunk=cg_target_chunk,
        solver_device=solver_device,
        progress=lambda message: print(f"{progress_prefix} solver=t4 {message}", flush=True),
    )
    v1_metric = explained_variance(
        held_targets,
        v1_fit.predict(categorical_design(held_codes, rows)),
        v1_fit.target_mean,
    )
    adjacent_metric = explained_variance(
        held_targets,
        adjacent_fit.predict(categorical_design(held_codes, rows, adjacent)),
        adjacent_fit.target_mean,
    )
    random_metric = explained_variance(
        held_targets,
        random_fit.predict(categorical_design(held_codes, rows, random)),
        random_fit.target_mean,
    )
    t4_v1_metric = explained_variance(
        held_targets,
        t4_fit.predict(categorical_design(held_t4, rows)),
        t4_fit.target_mean,
    )
    t4_v2_metric = explained_variance(
        held_targets,
        t4_pair_fit.predict(categorical_design(held_t4, rows, t4_pairing)),
        t4_pair_fit.target_mean,
    )
    joint = joint_mean_predict(fit_t4, fit_targets, held_t4, rows, global_mean=t4_fit.target_mean)
    joint_metric = explained_variance(held_targets, joint.prediction, t4_fit.target_mean)
    joint_metric.update(
        {
            "requested_cells": joint.requested_cells,
            "observed_requested_cells": joint.observed_requested_cells,
            "matched_fit_samples": joint.matched_fit_samples,
            "unseen_evaluation_fraction": joint.unseen_fraction,
            "fallback": "global_fit_mean",
        }
    )
    t32_adjacent = _spectrum_payload(v1_metric, adjacent_metric)
    t32_random = _spectrum_payload(v1_metric, random_metric)
    t4 = _spectrum_payload(t4_v1_metric, t4_v2_metric, joint_metric)
    checks = {
        "v1_relative_residual_at_most_1e_10": v1_fit.relative_residual <= 1e-10 and t4_fit.relative_residual <= 1e-10,
        "all_cg_converged": adjacent_fit.converged and random_fit.converged and t4_pair_fit.converged,
        "all_cg_relative_residual_at_most_tolerance": max(
            adjacent_fit.relative_residual, random_fit.relative_residual, t4_pair_fit.relative_residual
        )
        <= cg_tolerance,
        "t32_nested_within_0p005": float(t32_adjacent["v2"]) + 0.005 >= float(t32_adjacent["v1"]),
        "t4_v2_nested_within_0p005": float(t4["v2"]) + 0.005 >= float(t4["v1"]),
        "t4_joint_nested_within_0p005": float(t4["vt"]) + 0.005 >= float(t4["v2"]),
    }
    audits = {
        "v1": v1_fit.audit(),
        "v2_adjacent": adjacent_fit.audit(),
        "v2_random": random_fit.audit(),
        "t4_v1": t4_fit.audit(),
        "t4_v2": t4_pair_fit.audit(),
    }
    spectra = {"t32_adjacent": t32_adjacent, "t32_random": t32_random, "t4": t4}
    return audits, spectra, checks


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run_emnist_spectrum(
    seed: int,
    output_dir: Path,
    *,
    device: torch.device,
    batch_size: int,
    emnist_root: Path,
    artifact_path: Path,
    ridge: float,
    cg_iterations: int,
    cg_tolerance: float,
    cg_target_chunk: int,
    preregistration: Path,
) -> dict[str, object]:
    output_path = output_dir / "runs" / f"emnist-s{seed}.json"
    if output_path.exists():
        existing = json.loads(output_path.read_text())
        if existing.get("schema") == EMNIST_SCHEMA and existing.get("complete") is True:
            return existing
        raise FileExistsError(f"incomplete or invalid result already exists: {output_path}")
    if seed not in (0, 1, 2) or batch_size != 512:
        raise ValueError("formal EMNIST run is outside the preregistered seed/batch matrix")
    started = time.perf_counter()
    from tropnn.tools.emnist_payload_dtype_sweep import _load_emnist_split
    from tropnn.tools.emnist_pq_product_grid_factorial import (
        AdditiveProductGridHead,
        NearestCentroidProductHead,
        _source_linear,
    )

    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    if artifact.get("schema") != "emnist-pq-product-grid-factorial-v1":
        raise ValueError("unexpected EMNIST PQ/grid artifact schema")
    artifact_protocol = artifact["protocol"]
    required_protocol = {
        "dataset": "EMNIST Balanced",
        "train_examples": 112_800,
        "held_examples": 18_800,
        "hidden_dim": 64,
        "tables": 32,
        "codes_per_table": 16,
        "classes": 47,
        "seeds": [0, 1, 2],
    }
    if any(artifact_protocol.get(key) != value for key, value in required_protocol.items()):
        raise RuntimeError("frozen EMNIST artifact protocol does not match preregistration")
    source_path = Path(artifact_protocol["source_artifact"])
    if not source_path.is_file():
        raise FileNotFoundError(f"frozen EMNIST source artifact is missing: {source_path}")
    source_stat = source_path.stat()
    if source_stat.st_size != artifact_protocol["source_artifact_size"] or source_stat.st_mtime_ns != artifact_protocol["source_artifact_mtime_ns"]:
        raise RuntimeError("frozen EMNIST source artifact metadata changed")
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    if source.get("schema") != "emnist-maddness-task-ste-v1":
        raise ValueError("unexpected frozen EMNIST stem artifact schema")
    train_x, train_y = _load_emnist_split(emnist_root, "balanced", train=True, limit=0, seed=0)
    held_x, held_y = _load_emnist_split(emnist_root, "balanced", train=False, limit=0, seed=0)
    if train_x.shape != (112_800, 784) or held_x.shape != (18_800, 784):
        raise RuntimeError("EMNIST Balanced split shape does not match preregistration")
    classes = 47
    stem = _source_linear(source["state"], seed, "stem", 784, 64).to(device).eval()
    state = artifact["state"]
    pq_router = NearestCentroidProductHead(
        state[f"seed{seed}.pq.centroids"],
        state[f"seed{seed}.pq.tied_rows"],
        trainable_rows=False,
    ).to(device).eval()
    grid_router = AdditiveProductGridHead(
        64,
        classes,
        supports=state[f"seed{seed}.grid.supports"],
        thresholds=state[f"seed{seed}.grid.thresholds"],
        rows=state[f"seed{seed}.grid.tied_rows"],
        bins=4,
        surrogate="none",
        trainable_thresholds=False,
        trainable_rows=False,
    ).to(device).eval()
    modules = torch.nn.ModuleList((stem, pq_router, grid_router))
    sentinel = {name: value.detach().cpu().clone() for name, value in modules.state_dict().items()}
    train_pq, train_grid = _capture_emnist_codes(
        train_x,
        stem,
        pq_router,
        grid_router,
        batch_size=batch_size,
        device=device,
    )
    held_pq, held_grid = _capture_emnist_codes(
        held_x,
        stem,
        pq_router,
        grid_router,
        batch_size=batch_size,
        device=device,
    )
    after = modules.state_dict()
    mismatches = [name for name, value in sentinel.items() if name not in after or not torch.equal(value, after[name].detach().cpu())]
    print(f"family=emnist seed={seed} phase=code-capture samples={len(train_x) + len(held_x)}", flush=True)
    fit_targets = torch.nn.functional.one_hot(train_y, num_classes=classes).to(torch.float64)
    held_targets = torch.nn.functional.one_hot(held_y, num_classes=classes).to(torch.float64)
    arm_payloads: dict[str, object] = {}
    arm_checks: dict[str, bool] = {}
    for arm, fit_codes, evaluation_codes in (
        ("pq", train_pq, held_pq),
        ("grid", train_grid, held_grid),
    ):
        audits, spectra, checks = _fit_emnist_codes(
            fit_codes,
            fit_targets,
            evaluation_codes,
            held_targets,
            seed=seed,
            ridge=ridge,
            cg_iterations=cg_iterations,
            cg_tolerance=cg_tolerance,
            cg_target_chunk=cg_target_chunk,
            solver_device=device,
            progress_prefix=f"family=emnist arm={arm} seed={seed}",
        )
        arm_payloads[arm] = {
            "fit_code_sha256": _tensor_sha256(fit_codes),
            "held_code_sha256": _tensor_sha256(evaluation_codes),
            "fit_route_health": _route_health(fit_codes, 16),
            "held_route_health": _route_health(evaluation_codes, 16),
            "fit_audits": audits,
            "spectra": spectra,
            "checks": checks,
        }
        arm_checks.update({f"{arm}_{name}": value for name, value in checks.items()})
    checks = {"source_state_unchanged": not mismatches, **arm_checks}
    required_checks = {name: value for name, value in checks.items() if "t4_joint_nested" not in name}
    payload = {
        "schema": EMNIST_SCHEMA,
        "complete": all(required_checks.values()),
        "family": "emnist",
        "seed": seed,
        "protocol": {
            "dataset": "EMNIST Balanced",
            "fit_samples": len(train_x),
            "evaluation_samples": len(held_x),
            "batch_size": batch_size,
            "tables": 32,
            "rows": 16,
            "comparisons_per_table": 6,
            "target": "exact_class_one_hot",
            "target_dimensions": classes,
            "jl_seeds": [],
            "ridge": ridge,
            "cg_iterations": cg_iterations,
            "cg_tolerance": cg_tolerance,
            "cg_target_chunk": cg_target_chunk,
            "adjacent_pairing": [list(pair) for pair in adjacent_pairing(32)],
            "random_pairing": [list(pair) for pair in random_pairing(32, 51_000 + seed)],
            "t4_tables": [0, 1, 2, 3],
            "dtype": "float64_triton_cuda_v1_v2_cpu_scoring",
            "optimizer": None,
            "device_for_stem_and_code_extraction": str(device),
        },
        "preregistration": {**_path_metadata(preregistration), "sha256": hashlib.sha256(preregistration.read_bytes()).hexdigest()},
        "source": {
            "pq_grid_artifact": {**_path_metadata(artifact_path), "sha256": _file_sha256(artifact_path)},
            "stem_artifact": {**_path_metadata(source_path), "sha256": _file_sha256(source_path)},
            "pq_grid_schema": artifact["schema"],
            "stem_schema": source["schema"],
            "recognition_state_sha256": {
                key: _tensor_sha256(state[f"seed{seed}.{key}"])
                for key in ("pq.centroids", "grid.supports", "grid.thresholds")
            },
        },
        "source_state_verification": {"tensor_count": len(sentinel), "all_equal": not mismatches, "mismatch": mismatches},
        "targets": {
            "fit_label_sha256": _tensor_sha256(train_y),
            "held_label_sha256": _tensor_sha256(held_y),
            "fit_class_counts": torch.bincount(train_y, minlength=classes).tolist(),
            "held_class_counts": torch.bincount(held_y, minlength=classes).tolist(),
        },
        "arms": arm_payloads,
        "checks": checks,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_exclusive(output_path, payload)
    print(f"family=emnist seed={seed} phase=sealed path={output_path}", flush=True)
    if payload["complete"] is not True:
        raise RuntimeError(f"numerical gate failed after preserving result: {output_path}")
    return payload


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _write_exclusive(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="run the planted-spectrum estimator gate")
    validate.add_argument("--output", type=Path)
    validate.add_argument("--ridge", type=float, default=DEFAULT_RIDGE)
    validate.add_argument("--cg-tolerance", type=float, default=DEFAULT_CG_TOLERANCE)
    validate.add_argument("--cg-iterations", type=int, default=DEFAULT_CG_ITERATIONS)
    zipf = commands.add_parser("zipf-run", help="measure one frozen Zipf family/seed")
    zipf.add_argument("--family", choices=ZIPF_FAMILIES, required=True)
    zipf.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    zipf.add_argument("--output-dir", type=Path, default=Path("python/results/zipf_code_energy_spectrum_v1"))
    zipf.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    zipf.add_argument("--batch-size", type=int, default=512)
    zipf.add_argument("--fit-samples", type=int, default=ZIPF_FIT_SAMPLES)
    zipf.add_argument("--eval-samples", type=int, default=ZIPF_EVAL_SAMPLES)
    zipf.add_argument("--ridge", type=float, default=DEFAULT_RIDGE)
    zipf.add_argument("--cg-tolerance", type=float, default=DEFAULT_CG_TOLERANCE)
    zipf.add_argument("--cg-iterations", type=int, default=DEFAULT_CG_ITERATIONS)
    zipf.add_argument("--cg-target-chunk", type=int, default=DEFAULT_CG_TARGET_CHUNK)
    zipf.add_argument("--cpu-threads", type=int, default=16)
    zipf.add_argument("--canonical-dir", type=Path, default=Path("python/results/zipf_canonical_pclut_capacity_law_v2"))
    zipf.add_argument("--walsh-dir", type=Path, default=Path("python/results/zipf_groupsum_walsh_stage1_checkpointed_v2"))
    zipf.add_argument("--walsh-reference-dir", type=Path, default=Path("python/results/zipf_groupsum_walsh_stage1_v1"))
    zipf.add_argument("--independent-dir", type=Path, default=Path("python/results/zipf_groupsum_pclut_capacity_law_v3"))
    zipf.add_argument(
        "--preregistration",
        type=Path,
        default=Path("python/report/code_task_energy_spectrum_preregistration.md"),
    )
    hd = commands.add_parser("hd-run", help="measure one preregistered repeated-HD construction arm")
    hd.add_argument("--rounds", type=int, choices=(1, 2, 3), required=True)
    hd.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    hd.add_argument("--output-dir", type=Path, default=Path("python/results/zipf_code_energy_spectrum_v1"))
    hd.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    hd.add_argument("--batch-size", type=int, default=512)
    hd.add_argument("--fit-samples", type=int, default=ZIPF_FIT_SAMPLES)
    hd.add_argument("--eval-samples", type=int, default=ZIPF_EVAL_SAMPLES)
    hd.add_argument("--ridge", type=float, default=DEFAULT_RIDGE)
    hd.add_argument("--cg-tolerance", type=float, default=DEFAULT_CG_TOLERANCE)
    hd.add_argument("--cg-iterations", type=int, default=DEFAULT_CG_ITERATIONS)
    hd.add_argument("--cg-target-chunk", type=int, default=DEFAULT_CG_TARGET_CHUNK)
    hd.add_argument("--cpu-threads", type=int, default=16)
    hd.add_argument(
        "--preregistration",
        type=Path,
        default=Path("python/report/code_task_energy_spectrum_preregistration.md"),
    )
    emnist = commands.add_parser("emnist-run", help="measure one frozen EMNIST PQ/grid seed")
    emnist.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    emnist.add_argument("--output-dir", type=Path, default=Path("python/results/zipf_code_energy_spectrum_v1"))
    emnist.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    emnist.add_argument("--batch-size", type=int, default=512)
    emnist.add_argument("--cpu-threads", type=int, default=16)
    emnist.add_argument("--emnist-root", type=Path, required=True)
    emnist.add_argument("--artifact", type=Path, required=True)
    emnist.add_argument("--ridge", type=float, default=DEFAULT_RIDGE)
    emnist.add_argument("--cg-tolerance", type=float, default=DEFAULT_CG_TOLERANCE)
    emnist.add_argument("--cg-iterations", type=int, default=DEFAULT_CG_ITERATIONS)
    emnist.add_argument("--cg-target-chunk", type=int, default=DEFAULT_CG_TARGET_CHUNK)
    emnist.add_argument(
        "--preregistration",
        type=Path,
        default=Path("python/report/code_task_energy_spectrum_preregistration.md"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "validate":
        payload = planted_validation(ridge=args.ridge, cg_tolerance=args.cg_tolerance, cg_iterations=args.cg_iterations)
        if args.output is not None:
            _write_exclusive(args.output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), flush=True)
        if payload["complete"] is not True:
            raise SystemExit(1)
        return
    if args.command == "zipf-run":
        if args.cpu_threads < 1 or args.batch_size < 1:
            raise SystemExit("cpu threads and batch size must be positive")
        torch.set_num_threads(args.cpu_threads)
        result = run_zipf_spectrum(
            args.family,
            args.seed,
            args.output_dir,
            device=torch.device(args.device),
            batch_size=args.batch_size,
            fit_samples=args.fit_samples,
            eval_samples=args.eval_samples,
            ridge=args.ridge,
            cg_iterations=args.cg_iterations,
            cg_tolerance=args.cg_tolerance,
            cg_target_chunk=args.cg_target_chunk,
            canonical_dir=args.canonical_dir,
            walsh_dir=args.walsh_dir,
            walsh_reference_dir=args.walsh_reference_dir,
            independent_dir=args.independent_dir,
            preregistration=args.preregistration,
        )
        print(json.dumps({"family": result["family"], "seed": result["seed"], "summary": result["summary"]}, indent=2), flush=True)
        return
    if args.command == "hd-run":
        if args.cpu_threads < 1 or args.batch_size < 1:
            raise SystemExit("cpu threads and batch size must be positive")
        torch.set_num_threads(args.cpu_threads)
        result = run_hd_spectrum(
            args.rounds,
            args.seed,
            args.output_dir,
            device=torch.device(args.device),
            batch_size=args.batch_size,
            fit_samples=args.fit_samples,
            eval_samples=args.eval_samples,
            ridge=args.ridge,
            cg_iterations=args.cg_iterations,
            cg_tolerance=args.cg_tolerance,
            cg_target_chunk=args.cg_target_chunk,
            preregistration=args.preregistration,
        )
        print(json.dumps({"family": result["family"], "seed": result["seed"], "summary": result["summary"]}, indent=2), flush=True)
        return
    if args.command == "emnist-run":
        if args.cpu_threads < 1 or args.batch_size < 1:
            raise SystemExit("cpu threads and batch size must be positive")
        torch.set_num_threads(args.cpu_threads)
        result = run_emnist_spectrum(
            args.seed,
            args.output_dir,
            device=torch.device(args.device),
            batch_size=args.batch_size,
            emnist_root=args.emnist_root,
            artifact_path=args.artifact,
            ridge=args.ridge,
            cg_iterations=args.cg_iterations,
            cg_tolerance=args.cg_tolerance,
            cg_target_chunk=args.cg_target_chunk,
            preregistration=args.preregistration,
        )
        print(
            json.dumps(
                {
                    "family": result["family"],
                    "seed": result["seed"],
                    "pq_v1": result["arms"]["pq"]["spectra"]["t32_adjacent"]["v1"],
                    "grid_v1": result["arms"]["grid"]["spectra"]["t32_adjacent"]["v1"],
                },
                indent=2,
            ),
            flush=True,
        )
        return
    raise AssertionError(args.command)


if __name__ == "__main__":
    main()
