from __future__ import annotations

import torch
from tropnn.tools.s4_superposition_scaling_probe import (
    ChamberSuperpositionModel,
    apply_superposition_regularizer,
    fit_loss_floor,
    make_ordinal_data,
    payload_metrics,
    routes_to_target,
)


def test_ordinal_data_has_fixed_legal_s4_feature_system() -> None:
    _, first = make_ordinal_data(
        input_dim=12,
        tables=5,
        train_samples=512,
        held_samples=128,
        seed=7,
        device=torch.device("cpu"),
    )
    _, second = make_ordinal_data(
        input_dim=12,
        tables=5,
        train_samples=512,
        held_samples=128,
        seed=7,
        device=torch.device("cpu"),
    )
    assert first.route_fingerprint == second.route_fingerprint
    assert first.feature_counts.shape == (5 * 24,)
    assert first.train_routes.min() >= 0
    assert first.train_routes.max() < 24
    assert torch.equal(first.train_routes, second.train_routes)


def test_linear_and_pclut_cleanup_share_message_and_output_shapes() -> None:
    routes = torch.tensor([[0, 5, 23], [4, 1, 7]])
    target = routes_to_target(routes, 3 * 24)
    assert target.shape == (2, 72)
    assert torch.equal(target.sum(dim=1), torch.tensor([3.0, 3.0]))
    for readout in ("linear", "pclut_cleanup"):
        model = ChamberSuperpositionModel(
            tables=3,
            message_width=4,
            readout=readout,
            cleanup_tables=2,
            cleanup_comparisons=3,
            seed=11,
        )
        prediction = model(routes)
        assert model.message(routes).shape == (2, 4)
        assert prediction.shape == target.shape
        (prediction - target).square().mean().backward()
        assert model.payload.grad is not None
        if model.cleanup is not None:
            assert model.cleanup.lut.grad is not None


def test_growth_moves_row_norms_toward_one_and_decay_toward_zero() -> None:
    payload = torch.tensor([[0.2, 0.0], [2.0, 0.0]])
    grown = payload.clone()
    apply_superposition_regularizer(grown, learning_rate=0.1, coefficient=-1.0)
    assert grown[0].norm() > payload[0].norm()
    assert grown[1].norm() < payload[1].norm()
    decayed = payload.clone()
    apply_superposition_regularizer(decayed, learning_rate=0.1, coefficient=0.5)
    assert torch.all(decayed.norm(dim=1) < payload.norm(dim=1))


def test_overlap_metric_recovers_one_over_width_scale_for_random_rows() -> None:
    generator = torch.Generator().manual_seed(19)
    payload = torch.randn(24 * 170, 16, generator=generator)
    metrics = payload_metrics(payload, tables=170)
    assert abs(float(metrics["mean_squared_overlap"]) * 16 - 1.0) < 0.08


def test_loss_floor_fit_recovers_synthetic_curve() -> None:
    rows = [
        {"message_width": width, "held_loss": 0.03 + 1.7 / width**0.8}
        for width in (4, 8, 16, 32, 64, 128)
    ]
    fit = fit_loss_floor(rows)
    assert abs(fit["loss_floor"] - 0.03) < 0.01
    assert abs(fit["alpha"] - 0.8) < 0.08
    assert fit["r2"] > 0.999
