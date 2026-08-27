from __future__ import annotations

from argparse import Namespace

import torch
import torch.nn.functional as F
from tropnn.tools.emnist_router_dataflow_factorial import (
    HARD_ARMS,
    RouterStackClassifier,
    paired_models,
    summarize,
)


def test_paired_models_start_flat_tree_exact_within_predicate() -> None:
    args = Namespace(
        state_dim=64,
        tables=32,
        depth=4,
        residual_scale=0.25,
        tau=1.0,
        prototype_std=0.02,
    )
    models, checks = paired_models(args, classes=47, seed=0)
    assert all(checks.values())
    assert set(HARD_ARMS).issubset(models)


def test_five_hard_layers_receive_final_task_gradients() -> None:
    model = RouterStackClassifier(
        16,
        8,
        5,
        hidden_layers=4,
        tables=4,
        depth=4,
        predicate="pair",
        topology="adaptive",
        seed=17,
        residual_scale=0.25,
        tau=1.0,
        prototype_std=0.02,
    )
    x = torch.randn(32, 1, 4, 4)
    target = torch.randint(5, (32,))
    F.cross_entropy(model(x), target).backward()
    assert model.stem.weight.grad is not None and float(model.stem.weight.grad.norm()) > 0
    for layer in model.router_layers():
        assert layer.thresholds.grad is not None and float(layer.thresholds.grad.norm()) > 0
        assert layer.rows.grad is not None and float(layer.rows.grad.norm()) > 0


def test_summarize_uses_positive_as_improvement() -> None:
    from tropnn.tools.emnist_router_dataflow_factorial import ArmEvaluation, SeedResult

    evaluations = [
        ArmEvaluation("flat_pair", 0.8, 0.5, 1, True, 1.0, 1.0, 0.5, 2.0, 0.1),
        ArmEvaluation("flat_unary", 1.0, 0.5, 1, True, 1.0, 1.0, 0.5, 2.0, 0.1),
        ArmEvaluation("adaptive_pair", 0.7, 0.5, 1, True, 1.0, 1.0, 0.5, 2.0, 0.1),
        ArmEvaluation("adaptive_unary", 0.95, 0.5, 1, True, 1.0, 1.0, 0.5, 2.0, 0.1),
        ArmEvaluation("dense_l4", 0.6, 0.5, 1, None, None, None, None, None, None),
    ]
    gradients = {arm: {"stem": 1.0, "thresholds": 1.0, "rows": 1.0} for arm in HARD_ARMS}
    row = SeedResult(0, {"ok": True}, {}, gradients, evaluations, 1.0)
    summary = summarize([row])
    assert summary["effects"]["pair_predicate_grand_mean_ce_gain"] > 0
    assert summary["effects"]["adaptive_topology_grand_mean_ce_gain"] > 0
