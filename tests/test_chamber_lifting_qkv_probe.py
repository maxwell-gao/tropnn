from __future__ import annotations

import itertools

import torch
from tropnn.layers import ChamberLiftingStage, ChamberLiftingTower, permutation_rank4
from tropnn.tools.chamber_lifting_qkv_probe import (
    LiftingPairwiseCandidateScorer,
    LiftingValue,
    ProbeConfig,
    make_datasets,
    train_aggregation,
    train_value,
)


def test_permutation_rank4_enumerates_all_s4_chambers() -> None:
    permutations = torch.tensor(tuple(itertools.permutations(range(4))))
    ranks = permutation_rank4(permutations)

    assert torch.equal(ranks, torch.arange(24))


def test_zero_float_lifting_stage_is_identity() -> None:
    stage = ChamberLiftingStage(
        8,
        permutation=torch.tensor([4, 1, 7, 0, 6, 2, 5, 3]),
        coefficient_mode="float",
        seed=3,
    )
    with torch.no_grad():
        stage.base_coefficient_master.zero_()
        stage.coefficient_master.zero_()
    x = torch.randn(11, 8)
    output, chamber = stage.forward_with_chambers(x)

    assert torch.equal(output, x)
    assert chamber.shape == (11, 2)
    assert int(chamber.min()) >= 0
    assert int(chamber.max()) < 24


def test_ternary_lifting_tower_has_hard_codes_and_recursive_gradients() -> None:
    tower = ChamberLiftingTower(32, depth=6, coefficient_mode="ternary", seed=5)
    x = torch.randn(7, 32, requires_grad=True)
    output, chambers = tower.forward_with_chambers(x)
    output.square().mean().backward()

    assert output.shape == x.shape
    assert len(chambers) == 6
    assert all(chamber.shape == (7, 8) for chamber in chambers)
    assert torch.isfinite(output).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(stage.base_coefficient_master.grad is not None for stage in tower.stages)
    assert all(stage.coefficient_master.grad is not None for stage in tower.stages)
    assert all(set(stage.hard_ternary_codes().unique().tolist()) <= {-1, 0, 1} for stage in tower.stages)
    receptive_min, receptive_max = tower.receptive_field_sizes()
    assert 4 <= receptive_min <= receptive_max <= 32
    assert tower.active_operator_reads_per_item == 48
    assert tower.integer_adds_per_item == 576


def test_small_qkv_probe_trains_both_lifting_roles() -> None:
    config = ProbeConfig(
        dim=8,
        train_classes=8,
        ood_classes=4,
        train_pairs=16,
        test_pairs=8,
        seq_train=32,
        seq_test=16,
        value_train=32,
        value_test=16,
        candidates=4,
        value_classes=4,
        out_dim=4,
        tables=2,
        comparisons=2,
        rank=4,
        steps=2,
        batch_size=8,
        seed=7,
        device="cpu",
    )
    datasets = make_datasets(config, torch.device("cpu"))
    scorer = LiftingPairwiseCandidateScorer(
        config,
        depth=2,
        coefficient_mode="float",
        tower_sharing="shared",
    )
    value = LiftingValue(config, depth=2, coefficient_mode="float")

    aggregation_metrics = train_aggregation(
        scorer,
        config,
        datasets.aggregation_train,
        datasets.aggregation_test,
        datasets.aggregation_ood,
    )
    value_metrics = train_value(value, config, datasets.value_train, datasets.value_test, datasets.value_ood)

    assert 0.0 <= aggregation_metrics["ood_value_acc"] <= 1.0
    assert -1.0 <= value_metrics["ood_cosine"] <= 1.0
    assert aggregation_metrics["params"] > 0
    assert value_metrics["params"] > 0
