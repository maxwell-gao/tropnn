from __future__ import annotations

import torch
from tropnn.tools.ternary_margin_projection_fit import (
    SparseFloatMarginAction,
    build_student,
    materialize_affine,
    orthogonal_teacher,
    rank_ceiling_nmse,
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


def test_materialized_sparse_float_affine_matches_forward() -> None:
    dim = 9
    model = SparseFloatMarginAction(dim, atoms=7, fan_in=3, seed=2)
    weight, bias = materialize_affine(model, dim, device=torch.device("cpu"))
    x = torch.randn(13, dim)

    assert torch.equal(bias, torch.zeros_like(bias))
    assert torch.allclose(model(x), x @ weight.T, atol=1e-6)


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

