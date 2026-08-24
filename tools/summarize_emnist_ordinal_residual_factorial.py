from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

FAMILIES = ("noop", "constant_canonical", "constant_relabel", "live_canonical", "live_relabel", "dense")
SEEDS = (0, 1, 2)


def summarize(root: Path) -> dict[str, object]:
    rows = {family: [json.loads((root / family / f"seed{seed}" / "result.json").read_text()) for seed in SEEDS] for family in FAMILIES}
    for family, family_rows in rows.items():
        for seed, row in zip(SEEDS, family_rows):
            if (
                row.get("schema") != "emnist-ordinal-residual-factorial-v1"
                or row.get("family") != family
                or int(row.get("seed", -1)) != seed
                or not row.get("finite")
            ):
                raise ValueError(f"invalid result {family}/seed{seed}")

    def metric(family: str, field: str) -> list[float]:
        return [float(row[field]) for row in rows[family]]

    def mean(values: list[float]) -> float:
        return sum(values) / len(values)

    cc = metric("constant_canonical", "final_held_ce")
    cr = metric("constant_relabel", "final_held_ce")
    lc = metric("live_canonical", "final_held_ce")
    lr = metric("live_relabel", "final_held_ce")
    live_gain = [a - b for a, b in zip(cc, lc)]
    live_geometry_gain = [a - b for a, b in zip(lr, lc)]
    constant_geometry_gain = [a - b for a, b in zip(cr, cc)]
    interaction = [live - constant for live, constant in zip(live_geometry_gain, constant_geometry_gain)]
    effective = metric("live_canonical", "held_effective_chambers_mean")
    gate = {
        "live_beats_constant_all_seeds": all(value > 0 for value in live_gain),
        "live_minus_constant_mean_ce_gain": mean(live_gain),
        "live_minus_constant_mean_ce_gain_at_least_0p02": mean(live_gain) >= 0.02,
        "live_canonical_beats_live_relabel_all_seeds": all(value > 0 for value in live_geometry_gain),
        "live_canonical_vs_relabel_mean_ce_gain": mean(live_geometry_gain),
        "live_canonical_vs_relabel_mean_ce_gain_at_least_0p01": mean(live_geometry_gain) >= 0.01,
        "constant_canonical_vs_relabel_mean_ce_gain": mean(constant_geometry_gain),
        "factorial_interaction_mean": mean(interaction),
        "factorial_interaction_positive": mean(interaction) > 0,
        "effective_chambers_at_least_8_all_seeds": all(value >= 8 for value in effective),
    }
    gate["pass"] = all(
        bool(gate[key])
        for key in (
            "live_beats_constant_all_seeds",
            "live_minus_constant_mean_ce_gain_at_least_0p02",
            "live_canonical_beats_live_relabel_all_seeds",
            "live_canonical_vs_relabel_mean_ce_gain_at_least_0p01",
            "factorial_interaction_positive",
            "effective_chambers_at_least_8_all_seeds",
        )
    )
    aggregates = {
        family: {
            "held_ce": metric(family, "final_held_ce"),
            "held_ce_mean": mean(metric(family, "final_held_ce")),
            "held_accuracy_mean": mean(metric(family, "final_held_accuracy")),
            "effective_chambers_mean": mean(metric(family, "held_effective_chambers_mean")),
            "transition_fraction_mean": mean(metric(family, "held_transition_fraction_mean")),
            "transition_distance_mean": mean(metric(family, "held_transition_distance_mean")),
            "core_parameters_per_layer": int(rows[family][0]["core_parameters_per_layer"]),
        }
        for family in FAMILIES
    }
    return {
        "schema": "emnist-ordinal-residual-factorial-summary-v1",
        "complete_runs": 18,
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
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(result["scientific_gate"], sort_keys=True))


if __name__ == "__main__":
    main()
