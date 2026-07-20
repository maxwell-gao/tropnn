from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def flatten_summary(path: Path, root: Path) -> dict[str, object]:
    summary = json.loads(path.read_text())
    config = summary.get("config", {})
    initial = summary.get("initial", {})
    final = summary.get("final", {})
    row: dict[str, object] = {
        "run": str(path.parent.relative_to(root)),
        "parameters": summary.get("parameters", 0),
        "trainable_parameters": summary.get("trainable_parameters", 0),
    }
    row.update({f"config_{key}": value for key, value in config.items() if not isinstance(value, (dict, list))})
    row.update({f"initial_{key}": value for key, value in initial.items()})
    row.update({f"final_{key}": value for key, value in final.items()})
    return row


def numeric(rows: Iterable[dict[str, object]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def mean(rows: Iterable[dict[str, object]], key: str) -> float:
    values = numeric(rows, key)
    return statistics.fmean(values) if values else math.nan


def paired_delta(
    by_variant: dict[str, list[dict[str, object]]],
    first: str,
    second: str,
    metric: str,
) -> tuple[float, float, float]:
    first_by_seed = {int(row["config_seed"]): float(row[metric]) for row in by_variant.get(first, []) if metric in row}
    second_by_seed = {int(row["config_seed"]): float(row[metric]) for row in by_variant.get(second, []) if metric in row}
    seeds = sorted(first_by_seed.keys() & second_by_seed.keys())
    deltas = [first_by_seed[seed] - second_by_seed[seed] for seed in seeds]
    if not deltas:
        return math.nan, math.nan, math.nan
    generator = random.Random(20260719)
    bootstrap = [
        statistics.fmean(generator.choice(deltas) for _ in deltas)
        for _ in range(10000)
    ]
    bootstrap.sort()
    return statistics.fmean(deltas), bootstrap[int(0.025 * len(bootstrap))], bootstrap[int(0.975 * len(bootstrap))]


def select_hyperparameters(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    candidates = [row for row in rows if str(row["run"]).startswith("pair/lr_")]
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in candidates:
        grouped[str(row.get("config_variant"))].append(row)
    selected: dict[str, dict[str, object]] = {}
    metric = "final_held_pair_normalized_mse"
    for variant, group in grouped.items():
        valid = [row for row in group if isinstance(row.get(metric), (int, float))]
        if not valid:
            continue
        best = min(valid, key=lambda row: float(row[metric]))
        selected[variant] = {
            "learning_rate": best.get("config_learning_rate"),
            "threshold_learning_rate": best.get("config_threshold_learning_rate"),
            "weight_decay": best.get("config_weight_decay"),
            "validation_normalized_mse": best.get(metric),
            "run": best.get("run"),
        }
    return selected


def core_report(rows: list[dict[str, object]]) -> list[str]:
    core = [row for row in rows if str(row["run"]).startswith("pair/core_")]
    by_variant: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in core:
        by_variant[str(row.get("config_variant"))].append(row)
    lines = ["# Comparison-Routed Relation LUT: Experimental Report", "", "## Core results", ""]
    lines.append("| Variant | Seeds | Held-object R2 | Top-16 | NDCG@16 | Attention cosine |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for variant in sorted(by_variant):
        group = by_variant[variant]
        lines.append(
            f"| {variant} | {len(group)} | "
            f"{mean(group, 'final_held_object_r2'):.4f} | "
            f"{mean(group, 'final_held_object_top16_recall'):.4f} | "
            f"{mean(group, 'final_held_object_ndcg16'):.4f} | "
            f"{mean(group, 'final_held_object_attention_output_cosine'):.4f} |"
        )

    lines.extend(["", "## Five questions", ""])
    exact_delta, exact_low, exact_high = paired_delta(
        by_variant,
        "gram_free_float",
        "constrained_float",
        "initial_held_object_mse",
    )
    lines.append(
        "1. **Constrained Gram recovery.** Initial materialization MSE delta "
        f"is {exact_delta:.3e} with bootstrap CI [{exact_low:.3e}, {exact_high:.3e}]."
    )
    free_delta, free_low, free_high = paired_delta(
        by_variant,
        "gram_free_float",
        "constrained_float",
        "final_held_object_top16_recall",
    )
    lines.append(
        "2. **Gram-init free versus constrained.** Top-16 delta "
        f"is {free_delta:+.4f}, CI [{free_low:+.4f}, {free_high:+.4f}]."
    )
    random_delta, random_low, random_high = paired_delta(
        by_variant,
        "random_free_float",
        "gram_free_float",
        "final_held_object_top16_recall",
    )
    lines.append(
        "3. **Random free training.** Random-minus-Gram Top-16 delta "
        f"is {random_delta:+.4f}, CI [{random_low:+.4f}, {random_high:+.4f}]."
    )
    ternary_delta, ternary_low, ternary_high = paired_delta(
        by_variant,
        "gram_free_ternary",
        "gram_free_float",
        "final_held_object_top16_recall",
    )
    binary_delta, binary_low, binary_high = paired_delta(
        by_variant,
        "gram_free_binary",
        "gram_free_float",
        "final_held_object_top16_recall",
    )
    lines.append(
        "4. **Low-bit relation.** Ternary-minus-float Top-16 delta "
        f"is {ternary_delta:+.4f}, CI [{ternary_low:+.4f}, {ternary_high:+.4f}]; "
        f"binary delta is {binary_delta:+.4f}, CI [{binary_low:+.4f}, {binary_high:+.4f}]."
    )
    lines.append(
        "5. **B/K/depth scaling.** See the scaling tables below; a claim requires "
        "positive held-object trends and improvement over lookup-matched width."
    )

    scaling = [
        row
        for row in rows
        if str(row["run"]).startswith(("pair/scaleB_", "pair/scaleK_", "pair/budget_", "pair/depth_"))
    ]
    lines.extend(["", "## Scaling runs", ""])
    lines.append("| Run | Parameters | Held-object Top-16 | Held-object R2 |")
    lines.append("|---|---:|---:|---:|")
    for row in sorted(scaling, key=lambda item: str(item["run"])):
        lines.append(
            f"| {row['run']} | {int(row.get('parameters', 0))} | "
            f"{float(row.get('final_held_object_top16_recall', math.nan)):.4f} | "
            f"{float(row.get('final_held_object_r2', math.nan)):.4f} |"
        )
    return lines


def run(result_dir: Path, out_dir: Path) -> None:
    rows = [flatten_summary(path, result_dir) for path in sorted(result_dir.rglob("summary.json"))]
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with (out_dir / "aggregate.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    selected = select_hyperparameters(rows)
    (out_dir / "best_hyperparameters.json").write_text(json.dumps(selected, indent=2) + "\n")
    report = core_report(rows)
    report.insert(2, f"Completed runs found: {len(rows)}.")
    (out_dir / "report.md").write_text("\n".join(report) + "\n")
    print(json.dumps({"runs": len(rows), "best_hyperparameters": selected}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate the comparison relation LUT experiment suite.")
    parser.add_argument("--result-dir", type=Path, default=Path("results/comparison_relation_lut"))
    parser.add_argument("--out-dir", type=Path, default=Path("python/report/comparison_relation_lut_results"))
    args = parser.parse_args()
    run(args.result_dir, args.out_dir)


if __name__ == "__main__":
    main()
