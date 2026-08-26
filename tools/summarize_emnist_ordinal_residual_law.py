from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize(root: Path) -> dict[str, object]:
    laws = ("noop", "euclidean_euler", "intrinsic_exp")
    rows = {law: [json.loads((root / law / f"seed{seed}" / "result.json").read_text()) for seed in range(3)] for law in laws}
    ce_gain = [float(rows["euclidean_euler"][seed]["final_held_ce"]) - float(rows["intrinsic_exp"][seed]["final_held_ce"]) for seed in range(3)]
    accuracy_gain = [
        float(rows["intrinsic_exp"][seed]["final_held_accuracy"]) - float(rows["euclidean_euler"][seed]["final_held_accuracy"]) for seed in range(3)
    ]
    mean_ce_gain = sum(ce_gain) / len(ce_gain)
    gate = {
        "intrinsic_ce_better_all_seeds": all(value > 0.0 for value in ce_gain),
        "mean_intrinsic_ce_gain_ge_0p01": mean_ce_gain >= 0.01,
        "intrinsic_accuracy_better_all_seeds": all(value > 0.0 for value in accuracy_gain),
        "finite_all_runs": all(bool(row["finite"]) for law in laws for row in rows[law]),
        "learned_laws_preserve_chamber": all(
            float(row["held_transition_fraction_mean"]) == 0.0 for law in ("euclidean_euler", "intrinsic_exp") for row in rows[law]
        ),
    }
    return {
        "schema": "emnist-ordinal-residual-law-summary-v1",
        "records": rows,
        "means": {
            law: {
                "held_ce": sum(float(row["final_held_ce"]) for row in rows[law]) / 3.0,
                "held_accuracy": sum(float(row["final_held_accuracy"]) for row in rows[law]) / 3.0,
            }
            for law in laws
        },
        "paired_intrinsic_minus_euler": {
            "ce_gain_by_seed": ce_gain,
            "mean_ce_gain": mean_ce_gain,
            "accuracy_gain_by_seed": accuracy_gain,
            "mean_accuracy_gain": sum(accuracy_gain) / len(accuracy_gain),
        },
        "scientific_gate": {**gate, "pass": all(gate.values())},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(result["scientific_gate"], indent=2))


if __name__ == "__main__":
    main()
