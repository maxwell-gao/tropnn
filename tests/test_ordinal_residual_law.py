from __future__ import annotations

import json

import torch
from tropnn.layers.ordinal_residual_law import OrdinalResidualLawBlock
from tropnn.tools.emnist_ordinal_residual_law import OrdinalResidualLawClassifier
from tropnn.tools.summarize_emnist_ordinal_residual_law import summarize


def test_laws_match_parameter_budget_and_zero_identity() -> None:
    x = torch.randn(5, 784)
    expected = 196 * 8 * 4
    for law in ("euclidean_euler", "intrinsic_exp"):
        block = OrdinalResidualLawBlock(784, law=law, seed=3)
        assert block.operator_parameters == expected
        output, before, after = block.forward_with_codes(x)
        torch.testing.assert_close(output, x, rtol=0, atol=2e-7)
        assert torch.equal(before, after)


def test_retractions_share_first_order_vector_field() -> None:
    x = torch.randn(4, 8)
    euler = OrdinalResidualLawBlock(8, law="euclidean_euler", seed=0, residual_scale=1e-4)
    intrinsic = OrdinalResidualLawBlock(8, law="intrinsic_exp", seed=0, residual_scale=1e-4)
    with torch.no_grad():
        euler.feature_weight.normal_(std=0.2)
        intrinsic.feature_weight.copy_(euler.feature_weight)
    euler_delta = euler(x) - x
    intrinsic_delta = intrinsic(x) - x
    torch.testing.assert_close(euler_delta, intrinsic_delta, rtol=2e-3, atol=2e-7)


def test_both_retractions_preserve_the_selected_chamber() -> None:
    x = torch.randn(128, 16)
    for law in ("euclidean_euler", "intrinsic_exp"):
        block = OrdinalResidualLawBlock(16, law=law, seed=9, residual_scale=0.25)
        with torch.no_grad():
            block.feature_weight.normal_(std=5.0)
        _, before, after = block.forward_with_codes(x)
        assert torch.equal(before, after)


def test_training_gradient_reaches_both_residual_laws() -> None:
    x = torch.randn(3, 1, 28, 28)
    target = torch.randint(0, 47, (3,))
    for law in ("euclidean_euler", "intrinsic_exp"):
        model = OrdinalResidualLawClassifier(dim=784, classes=47, depth=2, law=law, seed=2, residual_scale=0.25)
        torch.nn.functional.cross_entropy(model(x), target).backward()
        gradients = [parameter.grad for block in model.blocks for parameter in block.parameters()]
        assert gradients and all(gradient is not None for gradient in gradients)
        assert sum(float(gradient.abs().sum()) for gradient in gradients if gradient is not None) > 0.0


def test_summary_replays_frozen_residual_law_gate(tmp_path) -> None:
    for law, ce, accuracy in (("noop", 1.2, 0.6), ("euclidean_euler", 0.9, 0.7), ("intrinsic_exp", 0.88, 0.72)):
        for seed in range(3):
            path = tmp_path / law / f"seed{seed}" / "result.json"
            path.parent.mkdir(parents=True)
            path.write_text(
                json.dumps(
                    {
                        "final_held_ce": ce,
                        "final_held_accuracy": accuracy,
                        "finite": True,
                        "held_transition_fraction_mean": 0.0,
                    }
                )
            )
    assert summarize(tmp_path)["scientific_gate"]["pass"] is True
