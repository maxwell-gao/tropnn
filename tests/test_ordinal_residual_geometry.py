from __future__ import annotations

import torch
from tropnn.layers.ordinal_residual import MatchedOrdinalResidualBlock, s4_diffusion_features
from tropnn.tools.emnist_ordinal_residual_geometry import OrdinalResidualEmnistClassifier, _s4_distances
from tropnn.tools.summarize_emnist_ordinal_residual_geometry import summarize


def test_s4_features_and_distances_follow_cayley_geometry() -> None:
    features = s4_diffusion_features()
    distances = _s4_distances()
    assert features.shape == (24, 8)
    assert torch.isfinite(features).all()
    assert torch.equal(distances, distances.T)
    assert int(distances.max()) == 6
    assert int((distances == 1).sum()) == 24 * 3


def test_non_noop_blocks_have_exact_matched_parameter_count() -> None:
    expected = 196 * 24 * 4
    for family in ("row", "coxeter", "coxeter_relabel", "dense"):
        block = MatchedOrdinalResidualBlock(784, kind=family, seed=3)
        assert block.operator_parameters == expected
        assert block.expected_operator_parameters == expected


def test_all_zero_initialized_residuals_are_exact_identity() -> None:
    x = torch.randn(5, 784)
    for family in ("row", "coxeter", "coxeter_relabel", "dense", "noop"):
        block = MatchedOrdinalResidualBlock(784, kind=family, seed=5)
        output, before, after = block.forward_with_codes(x)
        torch.testing.assert_close(output, x, rtol=0, atol=0)
        assert torch.equal(before, after)


def test_canonical_and_relabel_use_same_work_but_different_feature_assignment() -> None:
    canonical = MatchedOrdinalResidualBlock(784, kind="coxeter", seed=7)
    relabeled = MatchedOrdinalResidualBlock(784, kind="coxeter_relabel", seed=7)
    assert canonical.operator_parameters == relabeled.operator_parameters
    assert torch.equal(canonical.permutation, relabeled.permutation)
    assert torch.equal(canonical.chamber_features, relabeled.chamber_features)
    assert not torch.equal(canonical.chamber_relabel, relabeled.chamber_relabel)
    with torch.no_grad():
        canonical.feature_weight.normal_()
        relabeled.feature_weight.copy_(canonical.feature_weight)
    x = torch.randn(3, 784)
    assert not torch.allclose(canonical(x), relabeled(x))


def test_classifier_backward_reaches_each_residual_family() -> None:
    x = torch.randn(4, 784)
    target = torch.randint(0, 47, (4,))
    for family in ("row", "coxeter", "coxeter_relabel", "dense"):
        model = OrdinalResidualEmnistClassifier(dim=784, classes=47, depth=2, family=family, seed=11, residual_scale=0.25)
        torch.nn.functional.cross_entropy(model(x), target).backward()
        gradients = [parameter.grad for block in model.blocks for parameter in block.parameters()]
        assert gradients and all(gradient is not None for gradient in gradients)
        assert sum(float(gradient.abs().sum()) for gradient in gradients if gradient is not None) > 0


def test_summary_replays_frozen_gate(tmp_path) -> None:
    families = ("noop", "row", "coxeter", "coxeter_relabel", "dense")
    for family in families:
        for seed in range(3):
            path = tmp_path / family / f"seed{seed}" / "result.json"
            path.parent.mkdir(parents=True)
            ce = {"noop": 1.2, "row": 1.0, "coxeter": 0.95, "coxeter_relabel": 0.98, "dense": 0.8}[family]
            path.write_text(
                __import__("json").dumps(
                    {
                        "schema": "emnist-ordinal-residual-geometry-v1",
                        "family": family,
                        "seed": seed,
                        "finite": True,
                        "final_held_ce": ce,
                        "final_held_accuracy": 0.5,
                        "held_effective_chambers_mean": 12.0,
                        "held_transition_fraction_mean": 0.2,
                        "held_transition_distance_mean": 0.3,
                        "core_parameters_per_layer": 0 if family == "noop" else 18816,
                    }
                )
            )
    result = summarize(tmp_path)
    assert result["scientific_gate"]["pass"] is True
