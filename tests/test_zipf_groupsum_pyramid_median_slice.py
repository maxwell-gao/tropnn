from __future__ import annotations

import json

from tropnn.tools.zipf_groupsum_pyramid_median_slice import configs, summarize


def test_registered_pyramid_median_slice_is_exactly_t8_by_three_seeds() -> None:
    rows = configs("cuda:0")
    assert len(rows) == 24
    assert {row.tables for row in rows} == {1, 2, 4, 8, 16, 32, 64, 128}
    assert {row.seed for row in rows} == {0, 1, 2}
    assert all(row.arm == "pyramid_signed_median" and row.model_dim == 32 and row.comparisons == 6 and row.steps == 10_000 for row in rows)


def test_summary_pairs_every_median_with_same_t_seed_sum(tmp_path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "median"
    (source / "runs").mkdir(parents=True)
    (output / "runs").mkdir(parents=True)
    for index, config in enumerate(configs("cuda:0")):
        median = {
            "schema": "zipf-groupsum-pclut-capacity-law-run-v3",
            "complete": True,
            "run_key": config.run_key,
            "config": {"arm": config.arm, "tables": config.tables, "seed": config.seed},
            "validation": {"total_loss": 2.0 + index},
            "test": {"total_loss": 3.0 + index},
        }
        (output / "runs" / f"{config.run_key}.json").write_text(json.dumps(median))
        sum_key = config.run_key.replace("pyramid-signed-median", "pyramid-signed-sum")
        summed = {
            "complete": True,
            "config": {"arm": "pyramid_signed_sum"},
            "validation": {"total_loss": 4.0 + index},
            "test": {"total_loss": 5.0 + index},
        }
        (source / "runs" / f"{sum_key}.json").write_text(json.dumps(summed))
    result = summarize(source, output)
    assert result["complete"] is True and result["paired_count"] == 24
    assert all(row["test_sum_minus_median"] == 2.0 for row in result["paired"])
