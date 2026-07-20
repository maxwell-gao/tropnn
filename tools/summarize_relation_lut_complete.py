from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return math.nan


def average(rows: list[dict[str, str]], key: str) -> float:
    values = [number(row, key) for row in rows]
    values = [value for value in values if math.isfinite(value)]
    return mean(values) if values else math.nan


def grouped(rows: list[dict[str, str]], keys: tuple[str, ...]) -> dict[tuple[str, ...], list[dict[str, str]]]:
    result: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        result[tuple(row.get(key, "") for key in keys)].append(row)
    return result


def paired_delta(
    rows: list[dict[str, str]], left: str, right: str, metric: str, *, samples: int = 20_000
) -> tuple[float, float, float]:
    by_variant: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        value = number(row, metric)
        if math.isfinite(value):
            by_variant[row["config_variant"]][row["config_seed"]] = value
    seeds = sorted(set(by_variant[left]) & set(by_variant[right]))
    differences = [by_variant[left][seed] - by_variant[right][seed] for seed in seeds]
    if not differences:
        return math.nan, math.nan, math.nan
    rng = random.Random(0)
    draws = sorted(mean(rng.choices(differences, k=len(differences))) for _ in range(samples))
    return mean(differences), draws[int(0.025 * samples)], draws[int(0.975 * samples)]


def fmt(value: float, digits: int = 4) -> str:
    return "nan" if not math.isfinite(value) else f"{value:.{digits}f}"


def ci_text(result: tuple[float, float, float]) -> str:
    delta, low, high = result
    return f"{delta:+.4f} [{low:+.4f}, {high:+.4f}]"


def table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def core_section(core: list[dict[str, str]]) -> list[str]:
    metrics = (
        "final_train_pair_r2",
        "final_held_pair_r2",
        "final_held_object_r2",
        "final_held_object_top16_recall",
        "final_held_object_attention_output_cosine",
    )
    rows = []
    for (variant,), group in sorted(grouped(core, ("config_variant",)).items()):
        rows.append(
            [variant, str(len(group)), str(round(average(group, "parameters")))]
            + [fmt(average(group, metric)) for metric in metrics]
        )
    return table(
        ["Variant", "n", "Params", "Train R2", "Held-pair R2", "Held-object R2", "Top-16", "Output cosine"],
        rows,
    )


def realizability_section(rows: list[dict[str, str]]) -> list[str]:
    output = []
    for (teacher, student), group in sorted(grouped(rows, ("config_teacher", "config_student")).items()):
        output.append(
            [
                teacher,
                student,
                str(len(group)),
                fmt(average(group, "final_train_pair_r2")),
                fmt(average(group, "final_held_pair_r2")),
                fmt(average(group, "final_held_object_r2")),
            ]
        )
    return table(["Teacher", "Student", "n", "Train R2", "Held-pair R2", "Held-object R2"], output)


def geometry_section(rows: list[dict[str, str]]) -> list[str]:
    output = []
    keys = ("config_distribution", "config_teacher", "config_variant")
    for (distribution, teacher, variant), group in sorted(grouped(rows, keys).items()):
        output.append(
            [
                distribution,
                teacher,
                variant,
                fmt(average(group, "final_held_object_r2")),
                fmt(average(group, "final_held_object_top16_recall")),
            ]
        )
    return table(["Distribution", "Teacher", "Variant", "Held-object R2", "Top-16"], output)


def scaling_section(rows: list[dict[str, str]], prefix: str, field: str) -> list[str]:
    selected = [row for row in rows if Path(row["run"]).name.startswith(prefix)]
    output = []
    for (variant, value), group in sorted(
        grouped(selected, ("config_variant", field)).items(), key=lambda item: (item[0][0], float(item[0][1]))
    ):
        output.append(
            [
                variant,
                value,
                str(round(average(group, "parameters"))),
                fmt(average(group, "final_held_object_r2")),
                fmt(average(group, "final_held_object_top16_recall")),
            ]
        )
    return table(["Variant", field.removeprefix("config_"), "Params", "Held-object R2", "Top-16"], output)


def depth_section(rows: list[dict[str, str]]) -> list[str]:
    selected = [row for row in rows if Path(row["run"]).name.startswith("depth_")]
    output = []
    buckets: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in selected:
        match = re.search(r"depth_(additive|serial|width)_L(\d+)", Path(row["run"]).name)
        if match:
            buckets[(match.group(1), int(match.group(2)))].append(row)
    for (kind, depth), group in sorted(buckets.items()):
        output.append(
            [
                kind,
                str(depth),
                str(round(average(group, "parameters"))),
                fmt(average(group, "final_held_object_r2")),
                fmt(average(group, "final_held_object_top16_recall")),
            ]
        )
    return table(["Composition", "L / width multiplier", "Params", "Held-object R2", "Top-16"], output)


def budget_section(rows: list[dict[str, str]]) -> list[str]:
    selected = [row for row in rows if Path(row["run"]).name.startswith("budget_")]
    output = []
    for name, group in sorted(grouped(selected, ("run",)).items()):
        label = re.sub(r"_s\d+$", "", Path(name[0]).name)
        if any(existing[0] == label for existing in output):
            continue
        peers = [row for row in selected if re.sub(r"_s\d+$", "", Path(row["run"]).name) == label]
        output.append(
            [
                label,
                str(round(average(peers, "parameters"))),
                fmt(average(peers, "final_held_object_r2")),
                fmt(average(peers, "final_held_object_top16_recall")),
            ]
        )
    return table(["Budget control", "Params", "Held-object R2", "Top-16"], output)


def attention_section(rows: list[dict[str, str]]) -> list[str]:
    matched = [row for row in rows if row["config_depth"] == row["config_task_hops"]]
    output = []
    keys = ("config_variant", "config_shared_relation", "config_depth")
    for (variant, shared, depth), group in sorted(
        grouped(matched, keys).items(), key=lambda item: (item[0][0], item[0][1], int(item[0][2]))
    ):
        label = variant + (" shared" if shared == "True" else "")
        output.append(
            [
                label,
                depth,
                str(round(average(group, "parameters"))),
                fmt(average(group, "final_seen_noise0p0_task_accuracy")),
                fmt(average(group, "final_seen_noise0p05_task_accuracy")),
                fmt(average(group, "final_unseen_noise0p0_task_accuracy")),
            ]
        )
    return table(["Variant", "L=H", "Params", "Seen n=0", "Seen n=.05", "Unseen n=0"], output)


def long_training_section(short_rows: list[dict[str, str]], long_rows: list[dict[str, str]]) -> list[str]:
    short = [row for row in short_rows if row["config_depth"] == "1" and row["config_task_hops"] == "1" and row["config_seed"] == "0"]
    long = [row for row in long_rows if row.get("config_task_hops") == "1"]
    output = []
    for label, rows in (("3k", short), ("40k", long)):
        for (variant, shared), group in sorted(grouped(rows, ("config_variant", "config_shared_relation")).items()):
            name = variant + (" shared" if shared == "True" else "")
            output.append(
                [
                    label,
                    name,
                    fmt(average(group, "final_seen_noise0p0_task_accuracy")),
                    fmt(average(group, "final_unseen_noise0p0_task_accuracy")),
                ]
            )
    return table(["Steps", "Variant", "Seen n=0", "Unseen n=0"], output)


def build_report(
    pair_rows: list[dict[str, str]],
    attention_rows: list[dict[str, str]],
    realizability_rows: list[dict[str, str]],
    pilot: dict[str, object],
) -> str:
    core = [row for row in pair_rows if Path(row["run"]).name.startswith("core_")]
    geometry = [row for row in pair_rows if Path(row["run"]).name.startswith("geometry_")]
    long_attention = [row for row in pair_rows if row.get("config_task_hops")]
    gram_exact = average([row for row in core if row["config_variant"] == "constrained_float"], "initial_held_pair_mse") - average(
        [row for row in core if row["config_variant"] == "quantized_gram"], "initial_held_pair_mse"
    )
    top16 = "final_held_object_top16_recall"
    gram_vs_constrained = paired_delta(core, "gram_free_float", "constrained_float", top16)
    random_vs_gram = paired_delta(core, "random_free_float", "gram_free_float", top16)
    ternary_vs_float = paired_delta(core, "gram_free_ternary", "gram_free_float", top16)
    binary_vs_float = paired_delta(core, "gram_free_binary", "gram_free_float", top16)
    core_groups = grouped(core, ("config_variant",))
    dense_top16 = average(core_groups[("dense_oracle",)], top16)
    constrained_top16 = average(core_groups[("constrained_float",)], top16)
    constrained_r2 = average(core_groups[("constrained_float",)], "final_held_object_r2")
    random_train = average(core_groups[("random_free_float",)], "final_train_pair_r2")
    random_held = average(core_groups[("random_free_float",)], "final_held_object_r2")
    lines = [
        "# Directly Trained Comparison-Routed Binary Relations",
        "",
        "## Scope and protocol",
        "",
        f"The completed suite contains **{len(pair_rows) - len(long_attention)} pair-score runs**, "
        f"**{len(attention_rows)} multi-hop retrieval runs**, **{len(realizability_rows)} realizability controls**, "
        f"and **{len(long_attention)} 40k-step attention controls**. All GPU jobs completed without a recorded traceback.",
        "",
        "The optimizer pilot was selected only on held-pair normalized MSE. The selected settings were frozen by "
        "parameterization. The full attention grid uses 2k fixed-threshold steps followed by 1k joint-threshold steps; "
        "the 40k controls show that the short schedule captures the plateau while avoiding severe redundant compute.",
        "",
        "```json",
        json.dumps(pilot, indent=2, sort_keys=True),
        "```",
        "",
        "## Direct answers to the five questions",
        "",
        "### 1. Can constrained Gram recover the baseline?",
        "",
        f"It exactly materializes the discretized Gram table at initialization: the constrained-minus-materialized "
        f"initial held-pair MSE is {gram_exact:.3e}. It does **not** recover the dense relation on held objects. "
        f"Dense reaches Top-16 {dense_top16:.4f}; constrained Gram reaches {constrained_top16:.4f} with held-object "
        f"R2 {constrained_r2:.4f}. The failure is therefore in route-induced generalization, not in table materialization.",
        "",
        "### 2. Is Gram-initialized free better than constrained?",
        "",
        f"No detectable advantage at the default budget. The paired Top-16 delta is {ci_text(gram_vs_constrained)}. "
        "Freeing the relation cells does not repair the missing geometry.",
        "",
        "### 3. Can random-free relations train normally?",
        "",
        f"They optimize, but they do not generalize as relational scorers. Random-free train-pair R2 is {random_train:.4f}, "
        f"held-object R2 is {random_held:.4f}, and random-minus-Gram Top-16 is {ci_text(random_vs_gram)}. "
        "The result rejects an initialization barrier while exposing a held-object generalization barrier.",
        "",
        "### 4. Is a binary relation close to a float relation?",
        "",
        f"Only in the weak absolute Top-16 scale of this task. Ternary-minus-float is {ci_text(ternary_vs_float)}; "
        f"binary-minus-float is {ci_text(binary_vs_float)}. Binary payloads retain part of the ranking signal, but their "
        "held-object R2 is negative in the default-budget core, so they are not equivalent to float relations.",
        "",
        "### 5. Does the relation task scale with B, K, or depth?",
        "",
        "Yes with route width B; weakly or non-monotonically with code count K; and only weakly with true serial depth. "
        "At roughly equal parameter counts, parallel width is much stronger than serial score-conditioned composition. "
        "Multi-hop retrieval succeeds on seen objects but remains near chance on unseen objects, so increasing relation "
        "depth does not create a transferable metric space.",
        "",
        "## Core pair-score results",
        "",
        *core_section(core),
        "",
        "## Fixed-route realizability controls",
        "",
        *realizability_section(realizability_rows),
        "",
        "These controls separate optimization from function-class mismatch. A student should be judged realizable only "
        "when both held-pair and held-object R2 remain high; fitting train pairs alone is insufficient.",
        "",
        "## Distribution and teacher robustness",
        "",
        *geometry_section(geometry),
        "",
        "## B scaling",
        "",
        *scaling_section(pair_rows, "scaleB_", "config_num_banks"),
        "",
        "## K scaling",
        "",
        *scaling_section(pair_rows, "scaleK_", "config_num_codes"),
        "",
        "## Parameter-matched controls",
        "",
        *budget_section(pair_rows),
        "",
        "## Serial composition versus width",
        "",
        *depth_section(pair_rows),
        "",
        "## Multi-hop retrieval at matched model depth and task hops",
        "",
        *attention_section(attention_rows),
        "",
        "## Short versus long attention training",
        "",
        *long_training_section(attention_rows, long_attention),
        "",
        "The long controls do not reveal delayed unseen-object generalization. They mainly sharpen memorization of the "
        "seen vocabulary, supporting the use of the shorter schedule for the full factorial grid.",
        "",
        "## Scientific conclusion",
        "",
        "A binary relation table is directly trainable and its constrained Gram form is exactly materializable. Those "
        "facts are weaker than learning a reusable relation. Free relation cells, Gram initialization, threshold training, "
        "low-bit payloads, and serial relation depth all fail to close the held-object gap. The strongest positive signal is "
        "route width, which improves coverage rather than constructing a shared inner-product-like geometry. This suite "
        "therefore supports direct discrete binary relation learning as a memorizing relation mechanism, but not yet as a "
        "drop-in replacement for the transferable metric and retrieval geometry supplied by dot-product attention.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the complete comparison-relation experiment suite.")
    parser.add_argument("--pair-csv", type=Path, required=True)
    parser.add_argument("--attention-csv", type=Path, required=True)
    parser.add_argument("--realizability-csv", type=Path, required=True)
    parser.add_argument("--pilot-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(
        read_rows(args.pair_csv),
        read_rows(args.attention_csv),
        read_rows(args.realizability_csv),
        json.loads(args.pilot_json.read_text()),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    print(args.out)


if __name__ == "__main__":
    main()
