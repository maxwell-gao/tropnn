from __future__ import annotations

import math

import torch
from tropnn.tools.emnist_payload_feature_probe import (
    FeatureMoments,
    ModelConfig,
    _build_model,
    _project_hidden_payloads,
    _projector_overlap,
    _spectral_summary,
    _weighted_covariance,
)


def test_spectral_summary_recovers_rank_one() -> None:
    summary = _spectral_summary(torch.tensor([0.0, 0.0, 5.0]))

    assert summary["effective_rank"] == 1.0
    assert summary["stable_rank"] == 1.0
    assert summary["rank99"] == 1


def test_spectral_summary_keeps_scalar_threshold_on_cuda() -> None:
    if not torch.cuda.is_available():
        return
    summary = _spectral_summary(torch.tensor([0.0, 1.0], device="cuda"))
    assert summary["rank90"] == 1


def test_weighted_covariance_uses_route_frequency() -> None:
    rows = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
    usage = torch.tensor([[4, 0]])

    covariance = _weighted_covariance(rows, usage, centered=False)

    torch.testing.assert_close(covariance, torch.tensor([[4.0, 0.0], [0.0, 0.0]]))


def test_feature_moments_measure_perfect_class_separation() -> None:
    moments = FeatureMoments.zeros(classes=2, features=2)
    moments.update(
        torch.tensor([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]]),
        torch.tensor([0, 0, 1, 1]),
    )

    assert math.isclose(moments.between_total_ratio(), 1.0)
    torch.testing.assert_close(moments.centroids(), torch.tensor([[1.0, 0.0], [-1.0, 0.0]]))


def test_projector_overlap_has_random_baseline_not_built_in() -> None:
    identity = torch.eye(4)

    assert _projector_overlap(identity[:, :2], identity[:, :2]) == 1.0
    assert _projector_overlap(identity[:, :2], identity[:, 2:]) == 0.0


def test_hidden_projection_is_effective_and_restored() -> None:
    model = _build_model(
        ModelConfig(depth=1, tables=2, comparisons=2),
        classes=3,
        seed=0,
    )
    block = model.blocks[0]
    with torch.no_grad():
        block.lut.normal_()
    original = block.lut.detach().clone()
    basis = torch.eye(28 * 28)[:, :2]

    with _project_hidden_payloads(model, [basis]):
        torch.testing.assert_close(block.lut[..., :2], original[..., :2])
        torch.testing.assert_close(block.lut[..., 2:], torch.zeros_like(block.lut[..., 2:]))

    torch.testing.assert_close(block.lut, original)


def test_binary_payload_materializes_zero_one_with_ste_gradient() -> None:
    model = _build_model(
        ModelConfig(depth=1, tables=2, comparisons=2, payload_mode="binary01"),
        classes=3,
        seed=0,
    )
    block = model.blocks[0]
    with torch.no_grad():
        block.lut.flatten()[:4].copy_(torch.tensor([-0.2, 0.49, 0.51, 1.2]))

    values = block.materialized_lut().flatten()[:4]
    torch.testing.assert_close(values, torch.tensor([0.0, 0.0, 1.0, 1.0]))
    values.sum().backward()
    torch.testing.assert_close(block.lut.grad.flatten()[:4], torch.ones(4))


def test_binary_projection_rethresholds_and_restores_master() -> None:
    model = _build_model(
        ModelConfig(depth=1, tables=2, comparisons=2, payload_mode="binary01"),
        classes=3,
        seed=0,
    )
    block = model.blocks[0]
    with torch.no_grad():
        block.lut.bernoulli_(0.5)
    original = block.lut.detach().clone()
    basis = torch.eye(28 * 28)[:, :2]

    with _project_hidden_payloads(model, [basis]):
        assert set(block.materialized_lut().unique().tolist()) <= {0.0, 1.0}
        torch.testing.assert_close(block.materialized_lut()[..., 2:], torch.zeros_like(block.lut[..., 2:]))

    torch.testing.assert_close(block.lut, original)


def test_binary_projection_can_expose_continuous_diagnostic_and_restore_mode() -> None:
    model = _build_model(
        ModelConfig(depth=1, tables=2, comparisons=2, payload_mode="binary01"),
        classes=3,
        seed=0,
    )
    block = model.blocks[0]
    with torch.no_grad():
        block.lut.zero_()
        block.lut[..., 0] = 1.0
    original = block.lut.detach().clone()
    basis = torch.zeros(28 * 28, 1)
    basis[0, 0] = 2**-0.5
    basis[1, 0] = 2**-0.5

    with _project_hidden_payloads(model, [basis], force_continuous=True):
        assert block.payload_mode == "float"
        torch.testing.assert_close(block.materialized_lut()[..., :2], torch.full_like(block.lut[..., :2], 0.5))

    assert block.payload_mode == "binary01"
    torch.testing.assert_close(block.lut, original)
