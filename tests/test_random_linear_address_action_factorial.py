from __future__ import annotations

import torch
from tropnn.tools.random_linear_address_action_factorial import (
    ArmResult,
    _make_pair_page_regressor,
    orthogonal_teacher,
    summarize,
)


def test_experiment_models_use_hard_one_hot_shared_routes() -> None:
    for address in ("flat", "tree"):
        model = _make_pair_page_regressor(8, 3, address, "constant", anchor_seed=3, row_seed=4, tau=1.0)
        x = torch.randn(64, 8)
        probability_code = model.leaf_probabilities(x).argmax(dim=-1)
        assert torch.equal(probability_code, model.hard_codes(x))


def test_zero_slope_live_is_exact_constant() -> None:
    constant = _make_pair_page_regressor(8, 3, "tree", "constant", anchor_seed=5, row_seed=6, tau=1.0)
    live = _make_pair_page_regressor(8, 3, "tree", "live", anchor_seed=5, row_seed=6, tau=1.0)
    x = torch.randn(32, 8)
    assert torch.equal(constant.supports, live.supports)
    assert torch.equal(constant.rows, live.rows)
    assert torch.equal(constant(x), live(x))


def test_orthogonal_teacher_preserves_norm() -> None:
    teacher = orthogonal_teacher(12, 7, torch.device("cpu"))
    identity = teacher @ teacher.T
    assert torch.allclose(identity, torch.eye(12), atol=1e-5, rtol=1e-5)


def _row(arm: str, seed: int, r2: float) -> ArmResult:
    return ArmResult(arm, seed, 1, 0.0, 1.0 - r2, r2, 0.0, 1, 0.0, 1.0, 0.0, 0.0, [])


def test_factorial_summary_signs() -> None:
    rows = []
    for seed in (0, 1, 2):
        rows.extend(
            (
                _row("flat_constant", seed, 0.10),
                _row("tree_constant", seed, 0.20),
                _row("flat_live", seed, 0.30),
                _row("tree_live", seed, 0.50),
            )
        )
    summary = summarize(rows)
    assert summary["decisions"]["address_factor"]["pass"]
    assert summary["decisions"]["live_action_factor"]["pass"]
    assert summary["decisions"]["synergy"]["pass"]
