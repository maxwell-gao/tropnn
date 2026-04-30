from __future__ import annotations

import json

import torch
from tropnn.tools.scaling_benchmark import FAMILIES, RunConfig, feature_probabilities, main, run_config


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
                route_terms=2,
                seed=0,
                device="cpu",
                backend="torch",
            )
        )

        assert row["family"] == family
        assert float(row["final_loss"]) >= 0.0
        assert float(row["params"]) > 0.0
        assert 0.0 <= float(row["represented_fraction"]) <= 1.0
        assert "overlap_times_dim" in row
        assert "frequency_weighted_overlap_times_dim" in row
        assert "self_gain_weighted_mse" in row
        assert "offdiag_weighted_energy" in row
        assert "route_entropy_norm" in row


def test_quick_scaling_benchmark_writes_outputs(tmp_path) -> None:
    main(["--quick", "--tag", "unit", "--output-dir", str(tmp_path)])

    csv_files = list(tmp_path.glob("runs-unit-*.csv"))
    json_files = list(tmp_path.glob("summary-unit-*.json"))
    assert len(csv_files) == 1
    assert len(json_files) == 1

    summary = json.loads(json_files[0].read_text())
    assert len(summary["runs"]) == 14
    assert summary["exponents"]
    assert summary["group_metrics"]
    assert {"family", "model_dim", "final_loss", "overlap_times_dim"}.issubset(summary["runs"][0])
    assert {
        "mean_overlap_times_dim",
        "mean_frequency_weighted_overlap_times_dim",
        "mean_self_gain_weighted_mse",
        "mean_offdiag_weighted_energy",
    }.issubset(summary["group_metrics"][0])
