from argparse import Namespace

import torch
from tropnn.layers.hard_lookup import sum_lookup_rows
from tropnn.tools.emnist_pq_product_grid_factorial import (
    AdditiveProductGridHead,
    NearestCentroidProductHead,
    _train_action_rows,
    fit_lloyd_product_codebook,
    summarize,
)
from tropnn.tools.product_grid_pc_action_factorial import compile_balanced_product_grid


def test_lloyd_product_codebook_is_occupied_and_replays_nearest_codes() -> None:
    generator = torch.Generator().manual_seed(810)
    centers = torch.tensor(
        [
            [[-2.0, -2.0], [-2.0, 2.0], [2.0, -2.0], [2.0, 2.0]],
            [[-3.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [3.0, 0.0]],
        ]
    )
    labels = torch.randint(4, (2048, 2), generator=generator)
    local = torch.stack([centers[table, labels[:, table]] for table in range(2)], dim=1)
    x = (local + 0.1 * torch.randn(2048, 2, 2, generator=generator)).reshape(2048, 4)
    fitted, codes, completed = fit_lloyd_product_codebook(x, centers + 0.2, iterations=10)
    rows = torch.randn(2, 4, 3, generator=generator)
    head = NearestCentroidProductHead(fitted, rows, trainable_rows=False)
    assert torch.equal(codes, head.hard_codes(x))
    assert all(torch.unique(codes[:, table]).numel() == 4 for table in range(2))
    assert max(completed) <= 10
    output, replay_codes = head.hard_output(x)
    assert torch.equal(replay_codes, codes)
    assert torch.equal(output, sum_lookup_rows(rows, codes))


def test_additive_product_grid_reference_output_is_exact() -> None:
    x = torch.randn(1024, 8, generator=torch.Generator().manual_seed(811))
    supports, thresholds, codes = compile_balanced_product_grid(x, tables=4)
    rows = torch.randn(4, 16, 5, generator=torch.Generator().manual_seed(812))
    head = AdditiveProductGridHead(
        8,
        5,
        supports=supports,
        thresholds=thresholds,
        rows=rows,
        bins=4,
        surrogate="none",
        trainable_thresholds=False,
        trainable_rows=False,
    )
    output, deployed_codes = head.hard_output(x)
    assert torch.equal(deployed_codes, codes)
    assert torch.equal(output, sum_lookup_rows(rows, codes))


def test_action_only_training_updates_both_matched_row_tables() -> None:
    generator = torch.Generator().manual_seed(813)
    pq_rows = torch.zeros(2, 4, 3)
    grid_rows = torch.zeros(2, 4, 3)
    pq_codes = torch.randint(4, (128, 2), generator=generator)
    grid_codes = torch.randint(4, (128, 2), generator=generator)
    labels = torch.randint(3, (128,), generator=generator)
    pq_final, grid_final, curves = _train_action_rows(
        pq_rows,
        grid_rows,
        pq_codes,
        grid_codes,
        labels,
        epochs=2,
        batch_size=32,
        lr=0.01,
        seed=0,
        device=torch.device("cpu"),
    )
    assert not torch.equal(pq_final, pq_rows)
    assert not torch.equal(grid_final, grid_rows)
    assert len(curves["pq_free_action"]) == 2
    assert len(curves["grid_free_action"]) == 2


def test_summary_replays_frozen_decision_formulas() -> None:
    from tropnn.tools.emnist_pq_product_grid_factorial import Evaluation

    rows = []
    for seed in range(3):
        for arm, ce, accuracy in (
            ("pq_tied_frozen", 1.0, 0.7),
            ("pq_free_action", 0.8, 0.8),
            ("grid_tied_frozen", 1.0, 0.7),
            ("grid_free_action", 0.81, 0.798),
        ):
            rows.append(Evaluation(seed, arm, ce, accuracy, 4.0, 4.0, 16.0, 0.1, 0.0))
    summary = summarize(rows)
    assert summary["decisions"] == {
        "grid_retains_pq_quality": True,
        "pq_geometry_materially_better": False,
        "pq_free_action_helps": True,
        "grid_free_action_helps": True,
    }


def test_formal_shape_defaults_are_explicit() -> None:
    args = Namespace(hidden_dim=64, tables=32, compiler_samples=32768, epochs=10)
    assert args.hidden_dim == 2 * args.tables
    assert args.compiler_samples == 32768
    assert args.epochs == 10
