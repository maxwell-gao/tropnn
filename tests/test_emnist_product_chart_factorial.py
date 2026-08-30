import pytest
import torch
from tropnn.tools.emnist_product_chart_factorial import Evaluation, factorize_additive_rows, make_factorial_models, summarize


def test_factorized_initialization_is_best_svd_reconstruction() -> None:
    generator = torch.Generator().manual_seed(920)
    left = torch.randn(12, 3, generator=generator)
    right = torch.randn(3, 5, generator=generator)
    rows = (left @ right).reshape(3, 4, 5)
    offsets, basis, error = factorize_additive_rows(rows, 3)
    torch.testing.assert_close((offsets.reshape(-1, 3) @ basis).reshape_as(rows), rows, rtol=1e-5, atol=1e-5)
    assert error < 1e-12


def test_all_arms_share_initial_hard_output_with_zero_maps() -> None:
    generator = torch.Generator().manual_seed(921)
    centroids = torch.randn(2, 4, 2, generator=generator)
    rows = torch.randn(2, 4, 5, generator=generator)
    models, _error = make_factorial_models(centroids, rows, rank=3, temperature=1.0, seed=0)
    x = torch.randn(64, 4, generator=generator)
    reference = models["frozen_constant"](x)
    for model in models.values():
        assert torch.equal(model.hard_codes(x), models["frozen_constant"].hard_codes(x))
        assert torch.equal(model(x), reference)


def _row(seed: int, arm: str, ce: float) -> Evaluation:
    return Evaluation(seed, arm, ce, 0.5, 0.8, 0.0, 4.0, 3.0, 16.0, 0.2, 0.1, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0, 1.0)


def test_summary_keeps_address_and_action_effects_separate() -> None:
    rows = [
        _row(0, "frozen_constant", 1.0),
        _row(0, "trained_constant", 0.9),
        _row(0, "frozen_shared", 0.85),
        _row(0, "trained_shared", 0.7),
        _row(0, "frozen_local", 0.8),
        _row(0, "trained_local", 0.6),
    ]
    summary = summarize(rows)
    expected = {
        "trained_address_gain_under_constant_by_seed": [0.1],
        "trained_address_gain_under_shared_by_seed": [0.15],
        "trained_address_gain_under_local_by_seed": [0.2],
        "shared_linear_gain_under_frozen_address_by_seed": [0.15],
        "shared_linear_gain_under_trained_address_by_seed": [0.2],
        "code_conditioned_slope_gain_under_frozen_address_by_seed": [0.05],
        "code_conditioned_slope_gain_under_trained_address_by_seed": [0.1],
        "local_field_total_gain_under_frozen_address_by_seed": [0.2],
        "local_field_total_gain_under_trained_address_by_seed": [0.3],
        "difference_in_differences_by_seed": [0.1],
    }
    for key, value in expected.items():
        assert summary["effects"][key] == pytest.approx(value)
    assert all(summary["signals"].values())
