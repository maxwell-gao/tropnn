from __future__ import annotations

import math

import torch
from tropnn.tools.ternary_margin_projection_fit import (
    DenseTernaryProjection,
    FixedTernaryProjection,
    SparseFloatMarginAction,
    build_student,
    materialize_affine,
    matrix_rank_ceiling_nmse,
    orthogonal_teacher,
    rank_ceiling_nmse,
    ternary_teacher,
    truncated_svd_teacher,
)


def test_orthogonal_rank_ceiling_matches_truncated_svd_error() -> None:
    dim = 12
    rank = 5
    teacher = orthogonal_teacher(dim, seed=4, device=torch.device("cpu"))
    approximation = truncated_svd_teacher(teacher, rank)
    measured = ((approximation - teacher).square().sum() / teacher.square().sum()).item()

    assert torch.allclose(teacher @ teacher.T, torch.eye(dim), atol=1e-5)
    assert abs(measured - rank_ceiling_nmse(dim, rank)) < 1e-5
    assert abs(measured - matrix_rank_ceiling_nmse(teacher, rank)) < 1e-5


def test_ternary_teacher_has_expected_codes_and_scale() -> None:
    dim = 16
    teacher = ternary_teacher(dim, seed=4, device=torch.device("cpu"))
    codes = teacher * math.sqrt(dim)

    assert set(codes.unique().tolist()) <= {-1.0, 0.0, 1.0}
    assert torch.linalg.matrix_rank(teacher) == dim


def test_materialized_sparse_float_affine_matches_forward() -> None:
    dim = 9
    model = SparseFloatMarginAction(dim, atoms=7, fan_in=3, seed=2)
    weight, bias = materialize_affine(model, dim, device=torch.device("cpu"))
    x = torch.randn(13, dim)

    assert torch.equal(bias, torch.zeros_like(bias))
    assert torch.allclose(model(x), x @ weight.T, atol=1e-6)


def test_materialized_dense_ternary_projection_matches_forward() -> None:
    dim = 9
    model = DenseTernaryProjection(dim, seed=2)
    weight, bias = materialize_affine(model, dim, device=torch.device("cpu"))
    x = torch.randn(13, dim)

    assert torch.equal(bias, torch.zeros_like(bias))
    assert set(model.hard_codes().unique().tolist()) <= {-1, 0, 1}
    assert torch.allclose(model(x), x @ weight.T, atol=1e-6)


def test_scaled_ternary_oracle_improves_fixed_scale() -> None:
    dim = 32
    teacher = orthogonal_teacher(dim, seed=7, device=torch.device("cpu"))
    fixed = FixedTernaryProjection(teacher, optimize_scale=False)
    scaled = FixedTernaryProjection(teacher, optimize_scale=True)
    fixed_weight, _ = materialize_affine(fixed, dim, device=torch.device("cpu"))
    scaled_weight, _ = materialize_affine(scaled, dim, device=torch.device("cpu"))
    fixed_error = (fixed_weight - teacher).square().sum()
    scaled_error = (scaled_weight - teacher).square().sum()

    assert scaled_error <= fixed_error
    assert set(scaled.hard_codes().unique().tolist()) <= {-1, 0, 1}


def test_materialized_ternary_margin_linear_action_matches_forward() -> None:
    dim = 8
    teacher = orthogonal_teacher(dim, seed=6, device=torch.device("cpu"))
    model = build_student(
        "ternary_margin",
        teacher=teacher,
        dim=dim,
        atoms=11,
        fan_in=4,
        seed=3,
    )
    weight, bias = materialize_affine(model, dim, device=torch.device("cpu"))
    x = torch.randn(17, dim)

    assert torch.equal(bias, torch.zeros_like(bias))
    assert torch.allclose(model(x), x @ weight.T, atol=1e-6)
