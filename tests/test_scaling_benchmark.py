from __future__ import annotations

import json

import torch
from tropnn.tools.scaling_benchmark import (
    FAMILIES,
    RunConfig,
    _configs_from_args,
    feature_probabilities,
    main,
    run_config,
)


def test_feature_probabilities_match_activation_density() -> None:
    probs = feature_probabilities(64, 1.0, 2.0, device=torch.device("cpu"))

    assert probs.shape == (64,)
    assert torch.all(probs >= 0)
    assert torch.isclose(probs.sum(), torch.tensor(2.0), atol=1e-6)
    assert float(probs.max()) <= 1.0


def test_all_scaling_families_run_one_train_step() -> None:
    for family in FAMILIES:
        row = run_config(
            RunConfig(
                family=family,
                n_features=16,
                model_dim=4,
                alpha=0.0,
                activation_density=1.0,
                batch_size=8,
                steps=1,
                lr=1e-3,
                paper_lr=1e-2,
                weight_decay=-1.0,
                heads=4,
                cells=3,
                code_scale_mode="sqrt",
                pairwise_tables=4,
                comparisons=2,
                seed=0,
                device="cpu",
                backend="torch",
            )
        )

        assert row["family"] == family
        assert float(row["final_loss"]) >= 0
        assert float(row["params"]) > 0
        assert 0.0 <= float(row["represented_fraction"]) <= 1.0
        assert "overlap_times_dim" in row
        assert "loss_per_activation" in row
        assert "route_entropy_norm" in row


def test_quick_scaling_benchmark_writes_outputs(tmp_path) -> None:
    main(["--quick", "--device", "cpu", "--tag", "unit", "--output-dir", str(tmp_path)])

    csv_files = list(tmp_path.glob("runs-unit-*.csv"))
    json_files = list(tmp_path.glob("summary-unit-*.json"))
    assert len(csv_files) == 1
    assert len(json_files) == 1

    summary = json.loads(json_files[0].read_text())
    assert len(summary["runs"]) == 8
    assert summary["exponents"]
    assert summary["group_metrics"]
    assert {"family", "model_dim", "final_loss", "mean_squared_overlap"}.issubset(summary["runs"][0])
    assert {"mean_overlap_times_dim", "mean_represented_fraction"}.issubset(summary["group_metrics"][0])


def test_sweep_lists_expand_tropical_only() -> None:
    import argparse

    args = argparse.Namespace(
        families="paper,tropical,pairwise",
        n_features=16,
        model_dims="4",
        alphas="1.0",
        activation_density=1.0,
        batch_size=8,
        steps=1,
        lr=1e-3,
        paper_lr=1e-2,
        weight_decay=-1.0,
        heads=4,
        cells=3,
        heads_list="2,4",
        cells_list="2,3",
        code_scale_mode="sqrt",
        code_scale_modes="sqrt,linear",
        pairwise_tables=0,
        comparisons=2,
        seeds="0",
        device="cpu",
        backend="torch",
    )

    configs = _configs_from_args(args)

    assert len([config for config in configs if config.family == "tropical"]) == 8
    assert len([config for config in configs if config.family == "paper"]) == 1
    assert len([config for config in configs if config.family == "pairwise"]) == 1
