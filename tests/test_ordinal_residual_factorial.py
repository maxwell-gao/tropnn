from __future__ import annotations

import json

import torch
from tropnn.layers.ordinal_residual import FactorialOrdinalResidualBlock
from tropnn.tools.emnist_ordinal_residual_factorial import FactorialOrdinalEmnistClassifier
from tropnn.tools.summarize_emnist_ordinal_residual_factorial import summarize


def test_factorial_arms_match_exact_parameter_budget_and_identity() -> None:
    x = torch.randn(3, 784)
    expected = 196 * 8 * 4
    for family in ("constant_canonical", "constant_relabel", "live_canonical", "live_relabel", "dense"):
        block = FactorialOrdinalResidualBlock(784, kind=family, seed=3)
        assert block.operator_parameters == expected
        output, before, after = block.forward_with_codes(x)
        torch.testing.assert_close(output, x, rtol=0, atol=0)
        assert torch.equal(before, after)


def test_factorial_changes_one_axis_at_a_time() -> None:
    constant = FactorialOrdinalResidualBlock(784, kind="constant_canonical", seed=7)
    live = FactorialOrdinalResidualBlock(784, kind="live_canonical", seed=7)
    relabel = FactorialOrdinalResidualBlock(784, kind="live_relabel", seed=7)
    assert torch.equal(constant.permutation, live.permutation)
    assert torch.equal(constant.chamber_features, live.chamber_features)
    assert torch.equal(live.chamber_features, relabel.chamber_features)
    assert torch.equal(live.chamber_relabel, torch.arange(24))
    assert not torch.equal(live.chamber_relabel, relabel.chamber_relabel)
    with torch.no_grad():
        constant.feature_weight.normal_()
        live.feature_weight.copy_(constant.feature_weight)
        relabel.feature_weight.copy_(constant.feature_weight)
    x = torch.randn(4, 784)
    assert not torch.allclose(constant(x), live(x))
    assert not torch.allclose(live(x), relabel(x))


def test_factorial_all_trainable_arms_receive_gradient() -> None:
    x = torch.randn(3, 784)
    target = torch.randint(0, 47, (3,))
    for family in ("constant_canonical", "constant_relabel", "live_canonical", "live_relabel", "dense"):
        model = FactorialOrdinalEmnistClassifier(dim=784, classes=47, depth=2, family=family, seed=2, residual_scale=0.25)
        torch.nn.functional.cross_entropy(model(x), target).backward()
        grads = [parameter.grad for block in model.blocks for parameter in block.parameters()]
        assert grads and all(gradient is not None for gradient in grads)
        assert sum(float(gradient.abs().sum()) for gradient in grads if gradient is not None) > 0


def test_factorial_summary_replays_interaction_gate(tmp_path) -> None:
    ce = {
        "noop": 1.2,
        "constant_canonical": 1.0,
        "constant_relabel": 1.01,
        "live_canonical": 0.9,
        "live_relabel": 0.95,
        "dense": 0.8,
    }
    for family, value in ce.items():
        for seed in range(3):
            path = tmp_path / family / f"seed{seed}" / "result.json"
            path.parent.mkdir(parents=True)
            path.write_text(
                json.dumps(
                    {
                        "schema": "emnist-ordinal-residual-factorial-v1",
                        "family": family,
                        "seed": seed,
                        "finite": True,
                        "final_held_ce": value,
                        "final_held_accuracy": 0.5,
                        "held_effective_chambers_mean": 12.0,
                        "held_transition_fraction_mean": 0.2,
                        "held_transition_distance_mean": 0.3,
                        "core_parameters_per_layer": 0 if family == "noop" else 6272,
                    }
                )
            )
    assert summarize(tmp_path)["scientific_gate"]["pass"] is True
