"""Summarize the learned-unary budget sweep and apply the Wiki103 gate."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

VARIANTS = (
    "oracle",
    "random_fixed",
    "supervised_unary",
    "supervised_finetune",
    "learned_end_to_end",
    "binary_votes",
    "ternary_votes",
)
METRICS = ("r2", "top16", "top1", "spearman", "retrieval_cosine")


def mean(rows: list[dict], path: tuple[str, ...]) -> float:
    values = []
    for row in rows:
        value = row
        for key in path:
            value = value[key]
        values.append(float(value))
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for path in sorted(args.result_dir.glob("r*_u*_seed*.json")):
        row = json.loads(path.read_text())
        grouped[(int(row["relation_tables"]), int(row["unary_tables"]))].append(row)
    if not grouped:
        raise RuntimeError(f"no result JSON files found in {args.result_dir}")

    metric_rows = []
    decisions = []
    for (relation_tables, unary_tables), rows in sorted(grouped.items()):
        for variant in VARIANTS:
            metric_rows.append(
                {
                    "relation_tables": relation_tables,
                    "unary_tables": unary_tables,
                    "variant": variant,
                    "seeds": len(rows),
                    **{metric: mean(rows, (variant, metric)) for metric in METRICS},
                }
            )
        by_variant = {row["variant"]: row for row in metric_rows if row["relation_tables"] == relation_tables and row["unary_tables"] == unary_tables}
        learned_recovery = mean(rows, ("learned_recovery",))
        supervised_recovery = mean(rows, ("supervised_recovery",))
        binary_retention = by_variant["binary_votes"]["top16"] / max(by_variant["learned_end_to_end"]["top16"], 1e-8)
        learned_gate = learned_recovery >= 0.8
        lowbit_gate = learned_gate and binary_retention >= 0.9
        retrieval_gate = learned_gate and by_variant["learned_end_to_end"]["top16"] > 0.5
        decisions.append(
            {
                "relation_tables": relation_tables,
                "unary_tables": unary_tables,
                "seeds": len(rows),
                "supervised_recovery": supervised_recovery,
                "learned_recovery": learned_recovery,
                "binary_top16_retention": binary_retention,
                "learned_top16": by_variant["learned_end_to_end"]["top16"],
                "learned_gate_passed": learned_gate,
                "lowbit_gate_passed": lowbit_gate,
                "associative_retrieval_gate_passed": retrieval_gate,
                "all_gates_passed": learned_gate and lowbit_gate and retrieval_gate,
            }
        )

    best = max(decisions, key=lambda row: row["learned_recovery"])
    enter_wiki103 = any(row["all_gates_passed"] for row in decisions)
    decision = {
        "enter_wiki103": enter_wiki103,
        "next_stage": "wiki103" if enter_wiki103 else "stop",
        "gate": "mean learned recovery >= 0.8, binary retention >= 0.9, and learned Top-16 > 0.5",
        "best_configuration": best,
        "configurations": decisions,
    }
    args.result_dir.mkdir(parents=True, exist_ok=True)
    with (args.result_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    (args.result_dir / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")

    lines = [
        "# Learned Unary Coxeter Attention Scaling Gate",
        "",
        "## Decision",
        "",
        "**ENTER Wiki103.**" if enter_wiki103 else "**STOP before Wiki103.**",
        "",
        "The required gate is mean end-to-end recovery `>= 0.8`, followed "
        "conditionally by binary-vote retention `>= 0.9` and learned Top-16 "
        "`> 0.5`.",
        "",
        "## Budget sweep",
        "",
        "| Relation T | Unary T | Supervised recovery | End-to-end recovery | Learned Top-16 | Binary retention | Pass |",
        "|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in decisions:
        lines.append(
            f"| {row['relation_tables']} | {row['unary_tables']} | {row['supervised_recovery']:.4f} | "
            f"{row['learned_recovery']:.4f} | {row['learned_top16']:.4f} | "
            f"{row['binary_top16_retention']:.4f} | {'yes' if row['all_gates_passed'] else 'no'} |"
        )
    lines += [
        "",
        "## Held-object metrics",
        "",
        "| Relation T | Unary T | Variant | R2 | Top-16 | Top-1 | Spearman | Retrieval cosine |",
        "|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in metric_rows:
        lines.append(
            f"| {row['relation_tables']} | {row['unary_tables']} | {row['variant']} | {row['r2']:.4f} | "
            f"{row['top16']:.4f} | {row['top1']:.4f} | {row['spearman']:.4f} | {row['retrieval_cosine']:.4f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "The oracle gate asks whether local ordinal coordinates retain a "
        "relation. The learned-unary gate asks the harder and operationally "
        "relevant question: whether fixed input comparisons plus free unary LUT "
        "payloads can discover those coordinates from relation loss. Scaling "
        "relation width and unary width separates a budget failure from a "
        "trainability failure before any language-model run.",
        "",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines))
    print(json.dumps(decision))


if __name__ == "__main__":
    main()
