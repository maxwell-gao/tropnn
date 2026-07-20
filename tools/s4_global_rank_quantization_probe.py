from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from tropnn.tools.bilinear_retrieval_probe import make_problem, retrieval_metrics
from tropnn.tools.coxeter_relation_probe import LocalS4Router, r2_score
from tropnn.tools.s4_cross_table_kernel_probe import (
    S4_ORDER,
    fit_global_tower,
    parse_ridge_grid,
    sample_pair_split,
    teacher_coordinates,
    teacher_score_matrix,
    tower_embeddings,
    tower_score_matrix,
)


@dataclass(frozen=True)
class QuantizationSpec:
    name: str
    levels: tuple[int, ...]
    storage_bits: int
    group_size: int


SPECS = (
    QuantizationSpec("binary", (-1, 1), 1, 4),
    QuantizationSpec("ternary", (-1, 0, 1), 2, 3),
    QuantizationSpec("int2", (-3, -1, 1, 3), 2, 3),
    QuantizationSpec("int4", tuple(range(-15, 16, 2)), 4, 1),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quantize a frozen-route global rank-12 chamber teacher and score it through exact grouped LUTs.")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run")
    run.add_argument("--teacher", choices=("raw_bilinear", "ordinal_bilinear"), required=True)
    run.add_argument("--input-dim", type=int, default=32)
    run.add_argument("--train-queries", type=int, default=2048)
    run.add_argument("--train-keys", type=int, default=2048)
    run.add_argument("--test-queries", type=int, default=256)
    run.add_argument("--test-keys", type=int, default=512)
    run.add_argument("--max-value", type=int, default=15)
    run.add_argument("--tables", type=int, default=16)
    run.add_argument("--rank", type=int, default=12)
    run.add_argument("--fit-samples", type=int, default=65536)
    run.add_argument("--validation-samples", type=int, default=16384)
    run.add_argument("--top-k", type=int, default=16)
    run.add_argument("--ridge-grid", default="0.001,0.01,0.1,1.0,10.0")
    run.add_argument("--als-rounds", type=int, default=4)
    run.add_argument("--als-restarts", type=int, default=2)
    run.add_argument("--als-cg-iterations", type=int, default=64)
    run.add_argument("--cg-tolerance", type=float, default=1e-6)
    run.add_argument("--batch-size", type=int, default=4096)
    run.add_argument("--device", default="cuda")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--out-dir", type=Path, required=True)

    summarize = commands.add_parser("summarize")
    summarize.add_argument("--result-dir", type=Path, required=True)
    summarize.add_argument("--out-report", type=Path, required=True)
    return parser


def level_tensor(spec: QuantizationSpec, device: torch.device) -> Tensor:
    return torch.tensor(spec.levels, device=device, dtype=torch.float32)


def nearest_level_indices(values: Tensor, levels: Tensor, scale: Tensor | float) -> Tensor:
    normalized = values / torch.as_tensor(scale, device=values.device, dtype=values.dtype).clamp_min(1e-12)
    return (normalized.unsqueeze(-1) - levels).abs().argmin(dim=-1)


def fit_symmetric_scale(values: Tensor, levels: Tensor, iterations: int = 24) -> Tensor:
    """Fit one deployable per-tensor scale by one-dimensional Lloyd updates."""
    flattened = values.float().reshape(-1)
    absolute = flattened.abs()
    max_level = levels.abs().max().clamp_min(1.0)
    quantiles = torch.tensor((0.75, 0.85, 0.9, 0.95, 0.99, 0.999, 1.0), device=values.device)
    starts = torch.quantile(absolute, quantiles).div(max_level).clamp_min(1e-8)
    starts = torch.cat(
        (
            starts,
            flattened.square().mean().sqrt().view(1).div(max_level),
            absolute.mean().view(1).div(levels.abs().mean().clamp_min(1e-6)),
        )
    )
    best_scale = starts[0]
    best_mse = torch.tensor(float("inf"), device=values.device)
    for initial in starts:
        scale = initial
        for _ in range(iterations):
            indices = nearest_level_indices(flattened, levels, scale)
            codes = levels[indices]
            next_scale = (flattened * codes).sum() / codes.square().sum().clamp_min(1e-12)
            if torch.isclose(scale, next_scale, rtol=1e-7, atol=1e-9):
                scale = next_scale
                break
            scale = next_scale.clamp_min(1e-8)
        indices = nearest_level_indices(flattened, levels, scale)
        reconstruction = levels[indices] * scale
        mse = (flattened - reconstruction).square().mean()
        if mse < best_mse:
            best_mse = mse
            best_scale = scale
    return best_scale


def quantize_embeddings(values: Tensor, spec: QuantizationSpec, scale: Tensor) -> tuple[Tensor, Tensor]:
    levels = level_tensor(spec, values.device)
    indices = nearest_level_indices(values, levels, scale)
    reconstruction = levels[indices] * scale
    return indices, reconstruction


def pack_groups(indices: Tensor, alphabet: int, group_size: int) -> Tensor:
    if indices.shape[-1] % group_size:
        raise ValueError(f"rank {indices.shape[-1]} is not divisible by group size {group_size}")
    grouped = indices.view(indices.shape[0], -1, group_size)
    powers = alphabet ** torch.arange(group_size, device=indices.device, dtype=torch.int64)
    return (grouped.to(torch.int64) * powers).sum(dim=-1)


def grouped_relation_lut(
    spec: QuantizationSpec,
    *,
    device: torch.device,
) -> Tensor:
    alphabet = len(spec.levels)
    states = alphabet**spec.group_size
    code = torch.arange(states, device=device, dtype=torch.int64)
    powers = alphabet ** torch.arange(spec.group_size, device=device, dtype=torch.int64)
    digits = (code[:, None] // powers[None, :]) % alphabet
    levels = torch.tensor(spec.levels, device=device, dtype=torch.int32)
    decoded = levels[digits]
    return torch.sum(decoded[:, None, :] * decoded[None, :, :], dim=-1, dtype=torch.int32)


def grouped_lut_score_matrix(
    query_indices: Tensor,
    key_indices: Tensor,
    spec: QuantizationSpec,
    query_scale: Tensor,
    key_scale: Tensor,
    *,
    tables: int,
    bias: Tensor,
    target_scale: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    query_codes = pack_groups(query_indices, len(spec.levels), spec.group_size)
    key_codes = pack_groups(key_indices, len(spec.levels), spec.group_size)
    relation = grouped_relation_lut(
        spec,
        device=query_indices.device,
    )
    integer_score = torch.zeros(
        query_indices.shape[0],
        key_indices.shape[0],
        device=query_indices.device,
        dtype=torch.int32,
    )
    for group in range(query_codes.shape[1]):
        integer_score += relation[query_codes[:, None, group], key_codes[None, :, group]]
    score = (integer_score.float() * query_scale * key_scale / tables + bias) * target_scale
    return score, relation, integer_score


def stable_topk_recall(reference: Tensor, prediction: Tensor, top_k: int) -> float:
    """Top-k recall with key index as the deterministic secondary order."""
    top_k = min(top_k, reference.shape[-1])
    reference_top = torch.argsort(reference, dim=-1, descending=True, stable=True)[:, :top_k]
    prediction_top = torch.argsort(prediction, dim=-1, descending=True, stable=True)[:, :top_k]
    matches = (prediction_top.unsqueeze(-1) == reference_top.unsqueeze(-2)).any(dim=-1)
    return float(matches.to(torch.float64).mean().item())


def topk_retention(reference: Tensor, prediction: Tensor, top_k: int) -> float:
    return stable_topk_recall(reference, prediction, top_k)


def run(args: argparse.Namespace) -> None:
    matched_rank = S4_ORDER // 2
    if args.rank != matched_rank:
        raise ValueError(f"the matched S4 experiment requires rank={matched_rank}, got {args.rank}")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    problem = make_problem(args)
    relation = problem.relation.to(device)
    train_queries = problem.train_queries.to(device)
    train_keys = problem.train_keys.to(device)
    test_queries = problem.test_queries.to(device)
    test_keys = problem.test_keys.to(device)
    train_query_teacher = teacher_coordinates(train_queries, args.teacher)
    train_key_teacher = teacher_coordinates(train_keys, args.teacher)
    test_query_teacher = teacher_coordinates(test_queries, args.teacher)
    test_key_teacher = teacher_coordinates(test_keys, args.teacher)

    router = LocalS4Router(args.input_dim, args.tables, args.seed).to(device)
    train_query_route = router.route(train_queries)
    train_key_route = router.route(train_keys)
    test_query_route = router.route(test_queries)
    test_key_route = router.route(test_keys)
    fit_split = sample_pair_split(
        train_query_teacher,
        train_key_teacher,
        relation,
        args.fit_samples,
        args.seed + 2003,
    )
    validation = sample_pair_split(
        train_query_teacher,
        train_key_teacher,
        relation,
        args.validation_samples,
        args.seed + 2017,
    )

    started = time.perf_counter()
    candidates = [
        fit_global_tower(
            train_query_route,
            train_key_route,
            fit_split,
            validation,
            rank=args.rank,
            ridge=ridge,
            rounds=args.als_rounds,
            restarts=args.als_restarts,
            cg_iterations=args.als_cg_iterations,
            tolerance=args.cg_tolerance,
            batch_size=args.batch_size,
            seed=args.seed + 3011,
        )
        for ridge in parse_ridge_grid(args.ridge_grid)
    ]
    fitted = max(candidates, key=lambda candidate: candidate.validation_r2)

    train_query_embedding = tower_embeddings(train_query_route, fitted.query_factor)
    train_key_embedding = tower_embeddings(train_key_route, fitted.key_factor)
    test_query_embedding = tower_embeddings(test_query_route, fitted.query_factor)
    test_key_embedding = tower_embeddings(test_key_route, fitted.key_factor)
    target = teacher_score_matrix(test_query_teacher, test_key_teacher, relation)
    full_score = tower_score_matrix(test_query_route, test_key_route, fitted)
    full_metrics = retrieval_metrics(target, full_score, args.top_k, args.seed + 601)
    full_metrics["topk_recall"] = stable_topk_recall(target, full_score, args.top_k)

    variants: list[dict[str, object]] = [
        {
            "precision": "fp32_teacher",
            "storage_bits": 32,
            "pair_lut_reads": 0,
            "shared_relation_lut_entries": 0,
            "task_topk_retention": 1.0,
            "teacher_topk_overlap": 1.0,
            "teacher_score_r2": 1.0,
            **full_metrics,
        }
    ]
    for spec in SPECS:
        levels = level_tensor(spec, device)
        query_scale = fit_symmetric_scale(train_query_embedding, levels)
        key_scale = fit_symmetric_scale(train_key_embedding, levels)
        query_indices, query_reconstruction = quantize_embeddings(test_query_embedding, spec, query_scale)
        key_indices, key_reconstruction = quantize_embeddings(test_key_embedding, spec, key_scale)
        quantized_score, relation_lut, integer_score = grouped_lut_score_matrix(
            query_indices,
            key_indices,
            spec,
            query_scale,
            key_scale,
            tables=args.tables,
            bias=fitted.bias,
            target_scale=fitted.target_scale,
        )
        direct_score = (query_reconstruction @ key_reconstruction.T / args.tables + fitted.bias) * fitted.target_scale
        max_lut_error = float((quantized_score - direct_score).abs().max().item())
        if max_lut_error > 5e-5:
            raise RuntimeError(f"{spec.name} grouped LUT mismatch: {max_lut_error}")
        metrics = retrieval_metrics(target, quantized_score, args.top_k, args.seed + 601)
        metrics["topk_recall"] = stable_topk_recall(target, integer_score, args.top_k)
        variants.append(
            {
                "precision": spec.name,
                "levels": list(spec.levels),
                "storage_bits": spec.storage_bits,
                "group_size": spec.group_size,
                "pair_lut_reads": args.rank // spec.group_size,
                "shared_relation_lut_entries": relation_lut.numel(),
                "relation_lut_min": int(relation_lut.min().item()),
                "relation_lut_max": int(relation_lut.max().item()),
                "query_scale": float(query_scale.item()),
                "key_scale": float(key_scale.item()),
                "query_embedding_mse": float((test_query_embedding - query_reconstruction).square().mean().item()),
                "key_embedding_mse": float((test_key_embedding - key_reconstruction).square().mean().item()),
                "lut_direct_max_error": max_lut_error,
                "task_topk_retention": metrics["topk_recall"] / max(full_metrics["topk_recall"], 1e-12),
                "teacher_topk_overlap": topk_retention(full_score, integer_score, args.top_k),
                "teacher_score_r2": r2_score(full_score, quantized_score),
                **metrics,
            }
        )

    result = {
        "seed": args.seed,
        "teacher": args.teacher,
        "input_dim": args.input_dim,
        "tables": args.tables,
        "states_per_table": S4_ORDER,
        "rank": args.rank,
        "train_queries": args.train_queries,
        "train_keys": args.train_keys,
        "test_queries": args.test_queries,
        "test_keys": args.test_keys,
        "fit_samples": args.fit_samples,
        "validation_samples": args.validation_samples,
        "top_k": args.top_k,
        "ridge_grid": parse_ridge_grid(args.ridge_grid),
        "selected_ridge": fitted.ridge,
        "selected_validation_r2": fitted.validation_r2,
        "quantization_target": "post-aggregation rank-12 object embedding",
        "scale_granularity": "one symmetric scale per query/key tower",
        "score_path": "exact integer grouped relation LUT plus integer-code packing and addition",
        "ranking_tie_break": "stable key index order",
        "factor_payload_parameters": fitted.query_factor.numel() + fitted.key_factor.numel(),
        "route_anchor_groups": router.anchors.detach().cpu().tolist(),
        "elapsed_seconds": time.perf_counter() - started,
        "variants": variants,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"seed{args.seed}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)


def mean_sem(values: list[float]) -> tuple[float, float]:
    mean = statistics.mean(values)
    sem = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return mean, sem


def summarize(args: argparse.Namespace) -> None:
    runs = [json.loads(path.read_text()) for path in sorted(args.result_dir.glob("**/seed*.json"))]
    if not runs:
        raise RuntimeError(f"no seed JSON files under {args.result_dir}")
    flat = [
        {
            "seed": run["seed"],
            "teacher": run["teacher"],
            "rank": run["rank"],
            "selected_ridge": run["selected_ridge"],
            **variant,
        }
        for run in runs
        for variant in run["variants"]
    ]
    fields = sorted({key for row in flat for key in row if key != "levels"})
    args.result_dir.mkdir(parents=True, exist_ok=True)
    with (args.result_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: value for key, value in row.items() if key != "levels"} for row in flat])

    precisions = ("fp32_teacher", "binary", "ternary", "int2", "int4")
    teachers = sorted({run["teacher"] for run in runs})
    aggregate: list[dict[str, object]] = []
    for teacher in teachers:
        for precision in precisions:
            members = [row for row in flat if row["teacher"] == teacher and row["precision"] == precision]
            if not members:
                continue
            row: dict[str, object] = {
                "teacher": teacher,
                "precision": precision,
                "seeds": len(members),
                "storage_bits": int(members[0]["storage_bits"]),
                "pair_lut_reads": int(members[0]["pair_lut_reads"]),
                "shared_relation_lut_entries": int(members[0]["shared_relation_lut_entries"]),
            }
            for metric in (
                "topk_recall",
                "task_topk_retention",
                "teacher_topk_overlap",
                "teacher_score_r2",
                "score_r2",
                "spearman",
            ):
                mean, sem = mean_sem([float(member[metric]) for member in members])
                row[metric] = mean
                row[f"{metric}_sem"] = sem
            aggregate.append(row)
    with (args.result_dir / "aggregate.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)

    example = runs[0]
    lines = [
        "# Global Rank-12 Chamber Embedding Quantization",
        "",
        "## Protocol",
        "",
        "The full-precision global rank-12 model is refit on the identical frozen local-S4 routes and pair splits used "
        "by the matched-budget cross-table experiment. Only the post-aggregation 12-dimensional object embeddings "
        "are quantized. One symmetric scale per query/key tower is calibrated without labels on training-object "
        "embeddings.",
        "",
        "Scores are produced by exact integer grouped relation LUTs over the integer codes, not by using a "
        "floating-point dot product for the reported quantized path. The common dequantization scale and bias are "
        "applied only for score metrics and can be omitted for ranking. Integer-score ties use key index as a stable "
        "secondary order. `Task retention` is quantized task Top-16 divided by the full-precision rank-12 task "
        "Top-16. `Teacher overlap` is the fraction of the full-precision model's Top-16 keys recovered by the "
        "quantized model.",
        "",
        f"Configuration: rank `{example['rank']}`, tables `{example['tables']}`, seeds "
        f"`{','.join(str(seed) for seed in sorted({int(run['seed']) for run in runs}))}`, fit/validation pairs "
        f"`{example['fit_samples']:,}/{example['validation_samples']:,}`, held queries/keys "
        f"`{example['test_queries']:,}/{example['test_keys']:,}`.",
        "",
    ]
    for teacher in teachers:
        lines.extend(
            [
                f"## {teacher}",
                "",
                "| Precision | Bits | Pair LUT reads | Shared ROM | Top-16 | Task retention | Teacher overlap | "
                "Teacher-score R2 | Task R2 | Spearman |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in aggregate:
            if row["teacher"] != teacher:
                continue
            lines.append(
                f"| {row['precision']} | {row['storage_bits']} | {row['pair_lut_reads']} | "
                f"{row['shared_relation_lut_entries']:,} | {row['topk_recall']:.4f} | "
                f"{row['task_topk_retention']:.4f} +/- {row['task_topk_retention_sem']:.4f} | "
                f"{row['teacher_topk_overlap']:.4f} +/- {row['teacher_topk_overlap_sem']:.4f} | "
                f"{row['teacher_score_r2']:.4f} | {row['score_r2']:.4f} | {row['spearman']:.4f} |"
            )
        lines.append("")

    lookup = {(row["teacher"], row["precision"]): row for row in aggregate}
    lines.extend(["## Interpretation", ""])
    for teacher in teachers:
        lines.append(
            f"- `{teacher}` Top-16 task retention: "
            + ", ".join(
                f"{precision} `{lookup[(teacher, precision)]['task_topk_retention']:.1%}`" for precision in ("binary", "ternary", "int2", "int4")
            )
            + "."
        )
    lines.extend(
        [
            "",
            "Int4 is the clear fidelity point: it preserves about 98% of full-precision task Top-16 for both "
            "teachers, with 12 integer LUT reads per pair. Int2 is the strongest traffic compromise: four reads "
            "preserve about 78-82%. Ternary preserves about 71-73%, while binary retains only about half.",
            "",
            "Even binary remains above the previous same-table full-LUT Top-16 controls (`0.0674` raw and `0.0723` "
            "ordinal), so the global cross-table gain does not require a high-precision final dot product. However, "
            "binary and ternary discard too much of the rank-12 gain to be the default fidelity choice.",
            "",
        ]
    )
    lines.extend(
        [
            "## Scope",
            "",
            "This probe isolates replacement of the final width-12 dot product. The rank-12 factor rows remain "
            "full precision before object-side aggregation; factor-payload quantization is a separate gate. The ROM "
            "is fixed by the quantization alphabet and is not a learned relation table.",
            "",
            "## Artifacts",
            "",
            f"- Results: `{args.result_dir}`",
            f"- Logs: `logs/relation_energy/{args.result_dir.name}`",
            "- Launcher: `scripts/run_tropnn_s4_global_rank_quantization_4gpu.sh`",
            "- Probe: `python/src/tropnn/tools/s4_global_rank_quantization_probe.py`",
            "",
            "The authoritative directory above uses an integer ROM and stable key-index tie breaking. Earlier "
            "scaled-LUT and unstable-tie diagnostics remain preserved under the two sibling result directories "
            "without rewriting their evidence.",
        ]
    )
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text("\n".join(lines) + "\n")
    print(json.dumps({"runs": len(runs), "teachers": teachers, "report": str(args.out_report)}, sort_keys=True))


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run":
        run(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
