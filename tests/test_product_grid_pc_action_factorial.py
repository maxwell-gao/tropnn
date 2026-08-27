from argparse import Namespace

import torch
from tropnn.layers.hard_lookup import ProductGridLookupRouter
from tropnn.tools.product_grid_pc_action_factorial import (
    ARMS,
    compile_balanced_product_grid,
    fit_seed,
    summarize,
)


def test_balanced_product_grid_uses_all_sixteen_cells() -> None:
    generator = torch.Generator().manual_seed(201)
    x = torch.randn(4096, 8, generator=generator)
    supports, thresholds, codes = compile_balanced_product_grid(x, tables=4)
    assert supports.tolist() == [[0, 1], [2, 3], [4, 5], [6, 7]]
    assert thresholds.shape == (4, 2, 3)
    for table in range(4):
        assert torch.unique(codes[:, table]).numel() == 16
        counts = torch.bincount(codes[:, table], minlength=16)
        assert int(counts.min()) > 0


def test_product_grid_compiler_and_deployment_codes_match() -> None:
    x = torch.randn(1024, 8, generator=torch.Generator().manual_seed(202))
    supports, thresholds, codes = compile_balanced_product_grid(x, tables=4)
    rows = torch.randn(4, 16, 8, generator=torch.Generator().manual_seed(203))
    model = ProductGridLookupRouter(
        8,
        8,
        supports=supports,
        thresholds=thresholds,
        rows=rows,
        surrogate="none",
        trainable_thresholds=False,
        trainable_rows=False,
    )
    assert torch.equal(codes, model.hard_codes(x))


def test_tiny_product_grid_factorial_has_exact_hard_forwards() -> None:
    args = Namespace(
        device="cpu",
        dim=8,
        tables=4,
        depth=4,
        compiler_samples=512,
        held_samples=256,
        steps=2,
        batch_size=64,
        lr=0.0001,
        ridge=1.0,
        tau=1.0,
        seeds=(0,),
        log_every=1,
    )
    rows, audit, state = fit_seed(0, args)
    assert [row.arm for row in rows] == list(ARMS)
    assert audit["all_hard_replays_exact"] is True
    assert audit["all_finite"] is True
    assert float(audit["tree_cached_vs_direct_action_rows_max_abs_difference"]) < 5e-5
    assert float(audit["grid_cached_vs_direct_action_rows_max_abs_difference"]) < 5e-5
    assert state["grid.initial_thresholds"].shape == (4, 2, 3)
    summary = summarize(rows)
    assert set(summary["arms"]) == set(ARMS)
    assert set(summary["decisions"]) == {
        "grid_is_strong_product_atlas",
        "four_round_dependency_not_required",
        "adaptive_tree_materially_better",
        "tree_free_action_adaptation",
        "grid_free_action_adaptation",
    }
