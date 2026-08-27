from __future__ import annotations

from argparse import Namespace

import torch
import torch.nn.functional as F
from tropnn.tools.emnist_pair_node_specific_tree import ARMS, paired_models


def _args() -> Namespace:
    return Namespace(
        state_dim=64,
        tables=32,
        depth=4,
        residual_scale=0.25,
        tau=1.0,
        prototype_std=0.02,
    )


def test_node_specific_arm_is_payload_matched_and_has_per_node_supports() -> None:
    models, checks = paired_models(_args(), classes=47, seed=0)
    assert all(checks.values())
    assert set(models) == set(ARMS)
    flat = models["flat_pair"]
    level = models["adaptive_level_pair"]
    node = models["adaptive_node_pair"]
    for flat_layer, level_layer, node_layer in zip(
        flat.router_layers(),
        level.router_layers(),
        node.router_layers(),
        strict=True,
    ):
        assert torch.equal(flat_layer.rows, level_layer.rows)
        assert torch.equal(flat_layer.rows, node_layer.rows)
        assert level_layer.spec.support_layout == "level"
        assert node_layer.spec.support_layout == "node"
        assert node_layer.supports.shape == (32, 15, 2)


def test_node_specific_arm_gets_stem_threshold_and_row_gradients() -> None:
    models, _checks = paired_models(_args(), classes=47, seed=1)
    model = models["adaptive_node_pair"]
    x = torch.randn(16, 1, 28, 28)
    target = torch.randint(47, (16,))
    F.cross_entropy(model(x), target).backward()
    assert model.stem.weight.grad is not None and float(model.stem.weight.grad.norm()) > 0
    for layer in model.router_layers():
        assert layer.thresholds.grad is not None and float(layer.thresholds.grad.norm()) > 0
        assert layer.rows.grad is not None and float(layer.rows.grad.norm()) > 0
