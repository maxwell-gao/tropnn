from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

FAMILIES = ("noop", "row", "coxeter", "coxeter_relabel", "dense")
SEEDS = (0, 1, 2)


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def summarize(root: Path) -> dict[str, object]:
    rows: dict[str, list[dict[str, object]]] = {}
    for family in FAMILIES:
        family_rows = []
        for seed in SEEDS:
            path = root / family / f"seed{seed}" / "result.json"
            value = json.loads(path.read_text())
            if value.get("schema") != "emnist-ordinal-residual-geometry-v1":
                raise ValueError(f"invalid schema in {path}")
            if value.get("family") != family or int(value.get("seed", -1)) != seed:
                raise ValueError(f"identity mismatch in {path}")
            if not bool(value.get("finite")):
                raise ValueError(f"non-finite result in {path}")
            family_rows.append(value)
        rows[family] = family_rows

    def values(family: str, field: str) -> list[float]:
        return [float(row[field]) for row in rows[family]]

    def mean(items: list[float]) -> float:
        return sum(items) / len(items)

    row_ce = values("row", "final_held_ce")
    coxeter_ce = values("coxeter", "final_held_ce")
    relabel_ce = values("coxeter_relabel", "final_held_ce")
    row_gain = [baseline - candidate for baseline, candidate in zip(row_ce, coxeter_ce)]
    relabel_gain = [baseline - candidate for baseline, candidate in zip(relabel_ce, coxeter_ce)]
    effective = values("coxeter", "held_effective_chambers_mean")
    gate = {
        "coxeter_beats_row_all_seeds": all(value > 0 for value in row_gain),
        "coxeter_vs_row_mean_ce_gain": mean(row_gain),
        "coxeter_vs_row_mean_ce_gain_at_least_0p02": mean(row_gain) >= 0.02,
        "coxeter_beats_relabel_all_seeds": all(value > 0 for value in relabel_gain),
        "coxeter_vs_relabel_mean_ce_gain": mean(relabel_gain),
        "coxeter_vs_relabel_mean_ce_gain_at_least_0p01": mean(relabel_gain) >= 0.01,
        "coxeter_effective_chambers_at_least_8_all_seeds": all(value >= 8 for value in effective),
    }
    gate["pass"] = all(
        bool(gate[key])
        for key in (
            "coxeter_beats_row_all_seeds",
            "coxeter_vs_row_mean_ce_gain_at_least_0p02",
            "coxeter_beats_relabel_all_seeds",
            "coxeter_vs_relabel_mean_ce_gain_at_least_0p01",
            "coxeter_effective_chambers_at_least_8_all_seeds",
        )
    )
    aggregates = {
        family: {
            "held_ce": values(family, "final_held_ce"),
            "held_ce_mean": mean(values(family, "final_held_ce")),
            "held_accuracy": values(family, "final_held_accuracy"),
            "held_accuracy_mean": mean(values(family, "final_held_accuracy")),
            "effective_chambers_mean": mean(values(family, "held_effective_chambers_mean")),
            "transition_fraction_mean": mean(values(family, "held_transition_fraction_mean")),
            "transition_distance_mean": mean(values(family, "held_transition_distance_mean")),
            "core_parameters_per_layer": int(rows[family][0]["core_parameters_per_layer"]),
        }
        for family in FAMILIES
    }
    for family in FAMILIES:
        if not all(math.isfinite(float(item)) for item in aggregates[family]["held_ce"]):
            raise ValueError(f"non-finite aggregate for {family}")
    return {
        "schema": "emnist-ordinal-residual-geometry-summary-v1",
        "complete_runs": len(FAMILIES) * len(SEEDS),
        "aggregates": aggregates,
        "scientific_gate": gate,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = summarize(args.root)
    _atomic_json(args.output, result)
    print(json.dumps(result["scientific_gate"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
