from __future__ import annotations

import pytest
import torch
from tropnn.tools.code_task_energy_spectrum import (
    adjacent_pairing,
    categorical_design,
    fit_v1_count_ridge,
    fit_v2_sparse_cg,
    joint_mean_predict,
    planted_validation,
    random_pairing,
)


def _explicit_centered_ridge(design, target: torch.Tensor, ridge: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.zeros(design.samples, design.features, dtype=torch.float64)
    x.scatter_(1, design.indices, 1.0)
    feature_mean = x.mean(0)
    target_mean = target.mean(0)
    centered_x = x - feature_mean
    centered_y = target - target_mean
    coefficient = torch.linalg.solve(
        centered_x.T @ centered_x / len(x) + ridge * torch.eye(design.features, dtype=torch.float64),
        centered_x.T @ centered_y / len(x),
    )
    return target_mean + centered_x @ coefficient, coefficient, feature_mean


def test_v1_count_statistics_match_explicit_dense_ridge() -> None:
    generator = torch.Generator().manual_seed(13)
    codes = torch.randint(0, 4, (256, 3), generator=generator)
    target = torch.randn(256, 2, generator=generator, dtype=torch.float64)
    ridge = 3e-3
    fitted = fit_v1_count_ridge(codes, target, 4, ridge=ridge)
    design = categorical_design(codes, 4)
    expected, coefficient, feature_mean = _explicit_centered_ridge(design, target, ridge)
    assert torch.allclose(fitted.feature_mean, feature_mean, atol=0, rtol=0)
    assert torch.allclose(fitted.coefficient, coefficient, atol=1e-11, rtol=1e-10)
    assert torch.allclose(fitted.predict(design), expected, atol=1e-11, rtol=1e-10)


def test_v2_pcg_matches_explicit_dense_ridge_and_converges() -> None:
    generator = torch.Generator().manual_seed(17)
    codes = torch.randint(0, 3, (512, 4), generator=generator)
    target = torch.randn(512, 3, generator=generator, dtype=torch.float64)
    pairings = ((0, 1), (2, 3))
    ridge = 1e-2
    fitted = fit_v2_sparse_cg(
        codes,
        target,
        3,
        pairings,
        ridge=ridge,
        max_iterations=256,
        tolerance=1e-10,
        target_chunk=2,
    )
    design = categorical_design(codes, 3, pairings)
    expected, _, _ = _explicit_centered_ridge(design, target, ridge)
    assert fitted.converged
    assert fitted.relative_residual <= 1e-10
    assert torch.allclose(fitted.predict(design), expected, atol=2e-9, rtol=2e-9)


def test_batched_sparse_matvec_preserves_independent_cg_chunks() -> None:
    generator = torch.Generator().manual_seed(19)
    codes = torch.randint(0, 3, (384, 4), generator=generator)
    target = torch.randn(384, 17, generator=generator, dtype=torch.float64)
    pairings = ((0, 1), (2, 3))
    common = dict(ridge=1e-2, max_iterations=256, tolerance=1e-10, target_chunk=2)
    together = fit_v2_sparse_cg(codes, target, 3, pairings, **common)
    separate = [fit_v2_sparse_cg(codes, target[:, start : start + 2], 3, pairings, **common) for start in range(0, 17, 2)]
    assert together.converged
    assert all(item.converged for item in separate)
    assert torch.allclose(together.coefficient, torch.cat([item.coefficient for item in separate], dim=1), atol=2e-11, rtol=2e-10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/Triton")
def test_fused_cuda_estimators_match_cpu_float64_reference() -> None:
    generator = torch.Generator().manual_seed(23)
    codes = torch.randint(0, 4, (1024, 4), generator=generator)
    target = torch.randn(1024, 3, generator=generator, dtype=torch.float64)
    pairings = adjacent_pairing(4)
    cpu_v1 = fit_v1_count_ridge(codes, target, 4, ridge=1e-2)
    gpu_v1 = fit_v1_count_ridge(codes, target, 4, ridge=1e-2, solver_device="cuda:0")
    cpu_v2 = fit_v2_sparse_cg(codes, target, 4, pairings, ridge=1e-2, tolerance=1e-10)
    gpu_v2 = fit_v2_sparse_cg(codes, target, 4, pairings, ridge=1e-2, tolerance=1e-10, solver_device="cuda:0")
    design_v1 = categorical_design(codes, 4)
    design_v2 = categorical_design(codes, 4, pairings)
    assert torch.allclose(gpu_v1.predict(design_v1), cpu_v1.predict(design_v1), atol=2e-10, rtol=2e-10)
    assert torch.allclose(gpu_v2.predict(design_v2), cpu_v2.predict(design_v2), atol=2e-9, rtol=2e-9)


def test_joint_mean_uses_global_fit_mean_for_unseen_tuple() -> None:
    fit_codes = torch.tensor([[0, 0], [0, 0], [1, 0]], dtype=torch.int64)
    fit_target = torch.tensor([[1.0], [3.0], [8.0]], dtype=torch.float64)
    evaluation_codes = torch.tensor([[0, 0], [1, 1]], dtype=torch.int64)
    result = joint_mean_predict(fit_codes, fit_target, evaluation_codes, 2)
    assert result.seen.tolist() == [True, False]
    assert result.prediction[0, 0] == 2.0
    assert result.prediction[1, 0] == 4.0
    assert result.unseen_fraction == 0.5


def test_pairing_helpers_are_complete_and_deterministic() -> None:
    assert adjacent_pairing(6) == ((0, 1), (2, 3), (4, 5))
    first = random_pairing(8, 91003)
    assert first == random_pairing(8, 91003)
    assert sorted(table for pair in first for table in pair) == list(range(8))
    assert first != adjacent_pairing(8)


def test_planted_spectrum_gate_passes() -> None:
    result = planted_validation()
    assert result["complete"] is True
    assert all(result["checks"].values())
