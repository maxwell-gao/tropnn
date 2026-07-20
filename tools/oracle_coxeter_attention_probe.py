from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from torch import Tensor, nn

from tropnn.tools.bilinear_retrieval_probe import (
    make_problem,
    predict_score_matrix,
    retrieval_metrics,
    teacher_scores,
)
from tropnn.tools.coxeter_relation_probe import (
    LocalS4Router,
    UniformPairs,
    active_state_coverage,
    categorical_prediction,
    make_relation_pairs,
    make_uniform_pairs,
    positive_negative_accuracy,
    ridge_cg,
)

DECODERS = ("relative", "absolute", "k4_full_joint", "exact")
K4_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


@dataclass(frozen=True)
class OracleCoordinates:
    query: Tensor
    key: Tensor


def orthonormal_columns(dim: int, rank: int, generator: torch.Generator) -> Tensor:
    matrix = torch.randn(dim, rank, generator=generator)
    return torch.linalg.qr(matrix, mode="reduced").Q


def make_oracle_coordinates(args: argparse.Namespace) -> tuple[object, OracleCoordinates]:
    problem = make_problem(args)
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 3001)
    query = orthonormal_columns(args.input_dim, args.relation_rank, generator)
    key = query.clone() if args.teacher == "shared_gram" else orthonormal_columns(
        args.input_dim, args.relation_rank, generator
    )
    relation = query @ key.T / math.sqrt(args.relation_rank)
    return replace(problem, relation=relation), OracleCoordinates(query, key)


def permutation_to_k4_code(router: LocalS4Router) -> Tensor:
    code = torch.empty(24, dtype=torch.long)
    for permutation_index in range(24):
        order = router_order(permutation_index)
        rank = [0, 0, 0, 0]
        for position, coordinate in enumerate(order):
            rank[coordinate] = position
        value = 0
        for bit, (left, right) in enumerate(K4_EDGES):
            value |= int(rank[left] > rank[right]) << bit
        code[permutation_index] = value
    return code


def router_order(index: int) -> tuple[int, int, int, int]:
    elements = [0, 1, 2, 3]
    order: list[int] = []
    remainder = index
    for factorial in (6, 2, 1, 1):
        position, remainder = divmod(remainder, factorial)
        order.append(elements.pop(position))
    return tuple(order)  # type: ignore[return-value]


def route_pair_codes(
    router: LocalS4Router,
    query_route: Tensor,
    key_route: Tensor,
    pairs: UniformPairs,
    decoder: str,
) -> Tensor:
    query = query_route[pairs.query_index]
    key = key_route[pairs.key_index]
    if decoder == "k4_full_joint":
        route_to_bits = permutation_to_k4_code(router).to(query.device)
        return route_to_bits[query] * 64 + route_to_bits[key]
    return router.pair_codes(query, key, decoder)


def decoder_rows(decoder: str) -> int:
    if decoder == "relative":
        return 24
    if decoder == "absolute":
        return 24 * 24
    if decoder == "k4_full_joint":
        return 64 * 64
    raise ValueError(f"decoder {decoder!r} has no categorical rows")


class OracleCategoricalScorer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        coordinates: OracleCoordinates,
        router: LocalS4Router,
        decoder: str,
        coefficient: Tensor,
        bias: Tensor,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.router = router
        self.decoder = decoder
        self.rows = decoder_rows(decoder)
        self.register_buffer("query_projection", coordinates.query)
        self.register_buffer("key_projection", coordinates.key)
        self.register_buffer("coefficient", coefficient.view(router.tables, self.rows))
        self.register_buffer("bias", bias.reshape(()))
        self.register_buffer("table_index", torch.arange(router.tables).view(1, -1))
        self.register_buffer("route_to_bits", permutation_to_k4_code(router))

    def forward(self, pair: Tensor) -> Tensor:
        query, key = pair.split(self.input_dim, dim=-1)
        query_route = self.router.route(query @ self.query_projection)
        key_route = self.router.route(key @ self.key_projection)
        if self.decoder == "k4_full_joint":
            route_to_bits = self.route_to_bits.to(query_route.device)
            codes = route_to_bits[query_route] * 64 + route_to_bits[key_route]
        else:
            codes = self.router.pair_codes(query_route, key_route, self.decoder)
        score = self.coefficient[self.table_index.to(pair.device), codes].sum(dim=-1)
        return (score / math.sqrt(self.router.tables) + self.bias).unsqueeze(-1)


class ExactOracleScorer(nn.Module):
    def __init__(self, relation: Tensor) -> None:
        super().__init__()
        self.input_dim = relation.shape[0]
        self.register_buffer("relation", relation)

    def forward(self, pair: Tensor) -> Tensor:
        query, key = pair.split(self.input_dim, dim=-1)
        return ((query @ self.relation) * key).sum(dim=-1, keepdim=True)


def expand_absolute_to_full(router: LocalS4Router, coefficient: Tensor) -> Tensor:
    absolute = coefficient.view(router.tables, 24, 24)
    full = coefficient.new_zeros(router.tables, 64, 64)
    code = permutation_to_k4_code(router).to(coefficient.device)
    full[:, code[:, None], code[None, :]] = absolute
    return full.reshape(-1)


def evaluate_scorer(
    args: argparse.Namespace,
    scorer: nn.Module,
    relation_pairs: object,
    relation: Tensor,
) -> dict[str, float]:
    prediction = predict_score_matrix(
        scorer,
        relation_pairs.test_queries,
        relation_pairs.test_keys,
        args.eval_batch_size,
    )
    target = teacher_scores(relation_pairs.test_queries, relation_pairs.test_keys, relation)
    return {
        "train_pair_accuracy": positive_negative_accuracy(
            scorer,
            relation_pairs.positive,
            relation_pairs.negative,
            args.eval_batch_size,
        ),
        **{f"test_{key}": value for key, value in retrieval_metrics(target, prediction, args.top_k, args.seed + 601).items()},
    }


def fit_categorical(
    args: argparse.Namespace,
    router: LocalS4Router,
    coordinates: OracleCoordinates,
    train_query_route: Tensor,
    train_key_route: Tensor,
    fit: UniformPairs,
    validation: UniformPairs,
    decoder: str,
) -> tuple[OracleCategoricalScorer, Tensor, Tensor, dict[str, float | int]]:
    rows = decoder_rows(decoder)
    fit_codes = route_pair_codes(router, train_query_route, train_key_route, fit, decoder)
    validation_codes = route_pair_codes(router, train_query_route, train_key_route, validation, decoder)
    coefficient, bias, iterations, relative_residual = ridge_cg(
        fit_codes,
        fit.target,
        rows,
        args.ridge,
        args.cg_iterations,
        args.cg_tolerance,
        args.batch_size,
    )
    fit_prediction = categorical_prediction(fit_codes, coefficient, bias, args.tables)
    validation_prediction = categorical_prediction(validation_codes, coefficient, bias, args.tables)
    scorer = OracleCategoricalScorer(
        args.input_dim,
        coordinates,
        router,
        decoder,
        coefficient,
        bias,
    ).eval()
    metadata: dict[str, float | int] = {
        "parameters": coefficient.numel() + 1,
        "uniform_fit_r2": score_r2(fit.target, fit_prediction),
        "uniform_validation_r2": score_r2(validation.target, validation_prediction),
        "validation_active_state_seen_fraction": active_state_coverage(fit_codes, validation_codes, rows),
        "solver_iterations": iterations,
        "solver_relative_residual": relative_residual,
    }
    return scorer, coefficient, bias, metadata


def score_r2(target: Tensor, prediction: Tensor) -> float:
    target = target.to(torch.float64)
    prediction = prediction.to(torch.float64)
    residual = (target - prediction).square().sum()
    total = (target - target.mean()).square().sum().clamp_min(1e-12)
    return float((1.0 - residual / total).item())


def run(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    problem, cpu_coordinates = make_oracle_coordinates(args)
    coordinates = OracleCoordinates(cpu_coordinates.query.to(device), cpu_coordinates.key.to(device))
    relation = problem.relation.to(device)
    train_queries = problem.train_queries.to(device)
    train_keys = problem.train_keys.to(device)
    query_features = train_queries @ coordinates.query
    key_features = train_keys @ coordinates.key
    router = LocalS4Router(args.relation_rank, args.tables, args.seed).to(device)
    train_query_route = router.route(query_features)
    train_key_route = router.route(key_features)
    fit = make_uniform_pairs(train_queries, train_keys, relation, args.fit_samples, args.seed + 2003)
    validation = make_uniform_pairs(train_queries, train_keys, relation, args.validation_samples, args.seed + 2017)
    relation_pairs = make_relation_pairs(args, problem, relation, device)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fitted: dict[str, tuple[OracleCategoricalScorer, Tensor, Tensor, dict[str, float | int]]] = {}
    for decoder in ("relative", "absolute"):
        fitted[decoder] = fit_categorical(
            args,
            router,
            coordinates,
            train_query_route,
            train_key_route,
            fit,
            validation,
            decoder,
        )

    absolute_scorer, absolute_coefficient, absolute_bias, absolute_metadata = fitted["absolute"]
    del absolute_scorer
    full_coefficient = expand_absolute_to_full(router, absolute_coefficient)
    full_scorer = OracleCategoricalScorer(
        args.input_dim,
        coordinates,
        router,
        "k4_full_joint",
        full_coefficient,
        absolute_bias,
    ).eval()
    fitted["k4_full_joint"] = (
        full_scorer,
        full_coefficient,
        absolute_bias,
        {
            **absolute_metadata,
            "parameters": full_coefficient.numel() + 1,
            "embedded_from_absolute": 1,
        },
    )

    for decoder in ("relative", "absolute", "k4_full_joint"):
        scorer, _coefficient, _bias, metadata = fitted[decoder]
        result: dict[str, object] = {
            "decoder": decoder,
            "teacher": args.teacher,
            "seed": args.seed,
            "input_dim": args.input_dim,
            "relation_rank": args.relation_rank,
            "tables": args.tables,
            "comparisons_per_table": 6,
            "fit_samples": args.fit_samples,
            "validation_samples": args.validation_samples,
            "oracle_query_key_coordinates": True,
            "fixed_s4_routes": True,
            "ridge": args.ridge,
            **metadata,
            **evaluate_scorer(args, scorer, relation_pairs, relation),
        }
        (args.out_dir / f"{decoder}.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, sort_keys=True), flush=True)

    exact = ExactOracleScorer(relation).eval()
    exact_result: dict[str, object] = {
        "decoder": "exact",
        "teacher": args.teacher,
        "seed": args.seed,
        "input_dim": args.input_dim,
        "relation_rank": args.relation_rank,
        "tables": args.tables,
        "comparisons_per_table": 0,
        "fit_samples": args.fit_samples,
        "validation_samples": args.validation_samples,
        "oracle_query_key_coordinates": True,
        "fixed_s4_routes": False,
        "ridge": 0.0,
        "parameters": 0,
        "uniform_fit_r2": 1.0,
        "uniform_validation_r2": 1.0,
        "validation_active_state_seen_fraction": 1.0,
        "solver_iterations": 0,
        "solver_relative_residual": 0.0,
        **evaluate_scorer(args, exact, relation_pairs, relation),
    }
    (args.out_dir / "exact.json").write_text(json.dumps(exact_result, indent=2) + "\n")
    print(json.dumps(exact_result, sort_keys=True), flush=True)


def mean(rows: list[dict[str, object]], key: str) -> float:
    return statistics.fmean(float(row[key]) for row in rows)


def summarize(args: argparse.Namespace) -> None:
    rows = [json.loads(path.read_text()) for path in args.result_dir.glob("**/*.json")]
    rows = [row for row in rows if row.get("decoder") in DECODERS]
    if not rows:
        raise RuntimeError(f"no oracle Coxeter results in {args.result_dir}")
    rows.sort(key=lambda row: (row["teacher"], row["relation_rank"], row["tables"], row["decoder"], row["seed"]))
    summary_path = args.result_dir / "summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)

    groups: dict[tuple[str, int, int, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["teacher"]), int(row["relation_rank"]), int(row["tables"]), str(row["decoder"]))
        groups.setdefault(key, []).append(row)
    aggregates = []
    for (teacher, rank, tables, decoder), members in sorted(groups.items()):
        aggregates.append(
            {
                "teacher": teacher,
                "rank": rank,
                "tables": tables,
                "decoder": decoder,
                "seeds": len(members),
                "parameters": int(members[0]["parameters"]),
                "validation_r2": mean(members, "uniform_validation_r2"),
                "test_r2": mean(members, "test_score_r2"),
                "top16": mean(members, "test_topk_recall"),
                "top1": mean(members, "test_top1_accuracy"),
                "spearman": mean(members, "test_spearman"),
                "hard_negative": mean(members, "test_hard_negative_preference_accuracy"),
                "active_state_seen": mean(members, "validation_active_state_seen_fraction"),
            }
        )
    aggregate_path = args.result_dir / "aggregate.csv"
    with aggregate_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregates[0]))
        writer.writeheader()
        writer.writerows(aggregates)

    primary = [row for row in aggregates if row["teacher"] == "random_bilinear" and row["decoder"] in {"relative", "absolute"}]
    best = max(primary, key=lambda row: row["top16"])
    passed = float(best["top16"]) > args.top16_gate
    control = [row for row in aggregates if row["teacher"] == "shared_gram" and row["decoder"] in {"relative", "absolute"}]
    best_control = max(control, key=lambda row: row["top16"]) if control else None

    lines = [
        "# Oracle Q/K Coxeter Attention Gate",
        "",
        "## Decision",
        "",
        (
            "**PASS the oracle gate and continue to learned unary Q/K experiments.**"
            if passed
            else "**STOP: do not enter Wiki103 with this architecture.**"
        ),
        "",
        f"The prespecified gate is mean held-object Top-16 `> {args.top16_gate:.2f}` "
        "for at least one float S4 decoder with teacher-optimal Q/K coordinates. "
        f"The best random-bilinear result is `{float(best['top16']):.4f}` from "
        f"`{best['decoder']}`, rank `{best['rank']}`, T `{best['tables']}`.",
        "",
        "## Setup",
        "",
        "- Inputs: normalized finite integer vectors, D=32.",
        "- Teachers: low-rank random bilinear and shared-Gram positive control.",
        "- Q/K coordinates: exact teacher factor coordinates; no STE or learned unary layer.",
        "- Routing: fixed local-S4 order routes over random K4 coordinate groups.",
        "- Decoder payloads: float scalar tables fitted by ridge-CG.",
        "- K4-full joint is the exact 24x24 solution embedded into its reachable 64x64 rows.",
        "",
        "## Seed-mean results",
        "",
        "| Teacher | Rank | T | Decoder | Params | Valid R2 | Test R2 | Top-16 | Top-1 | Spearman | Hard-neg | Seen state |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        lines.append(
            "| {teacher} | {rank} | {tables} | {decoder} | {parameters} | "
            "{validation_r2:.4f} | {test_r2:.4f} | {top16:.4f} | {top1:.4f} | "
            "{spearman:.4f} | {hard_negative:.4f} | "
            "{active_state_seen:.4f} |".format(**row)
        )
    lines.extend(["", "## Interpretation", ""])
    if passed:
        lines.extend(
            [
                "The S4 decoder has sufficient oracle capacity. The next "
                "required gate is whether a residual zero-initialized unary "
                "PC-LUT can learn at least 80% of the oracle Top-16 improvement, "
                "first with supervised Q/K coordinate targets and then "
                "end-to-end from relation loss.",
                "",
                "No Wiki103 run is justified yet; unary learning, low-bit votes, and associative retrieval remain required.",
            ]
        )
    else:
        lines.extend(
            [
                "Providing the exact teacher Q/K factor coordinates removes the "
                "coordinate-learning and credit-assignment problems. Failure "
                "below the gate therefore identifies the local ordinal relation "
                "decoder itself as the limiting function class.",
                "",
                "Training unary PC-LUT Q/K payloads cannot exceed this oracle "
                "ceiling. Ternary/binary votes and associative retrieval can "
                "only add constraints, so they are not run after a failed "
                "oracle gate.",
            ]
        )
    if best_control is not None:
        lines.extend(
            [
                "",
                "The best shared-Gram positive control reaches Top-16 "
                f"`{float(best_control['top16']):.4f}` with "
                f"`{best_control['decoder']}`. This distinguishes an "
                "implementation failure from a mismatch between local ordinal "
                "relations and general bilinear retrieval.",
            ]
        )
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text("\n".join(lines) + "\n")
    decision = {
        "oracle_gate_passed": passed,
        "top16_gate": args.top16_gate,
        "best_random_bilinear": best,
        "best_shared_gram": best_control,
        "enter_wiki103": False,
        "next_stage": "learned_unary_qk" if passed else "stop",
    }
    (args.result_dir / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    print(json.dumps(decision, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Oracle-coordinate Coxeter relation gate before learned attention.")
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--teacher", choices=("random_bilinear", "shared_gram"), default="random_bilinear")
    run_parser.add_argument("--input-dim", type=int, default=32)
    run_parser.add_argument("--relation-rank", type=int, choices=(4, 8, 16, 32), default=8)
    run_parser.add_argument("--train-queries", type=int, default=2048)
    run_parser.add_argument("--train-keys", type=int, default=2048)
    run_parser.add_argument("--test-queries", type=int, default=256)
    run_parser.add_argument("--test-keys", type=int, default=512)
    run_parser.add_argument("--max-value", type=int, default=15)
    run_parser.add_argument("--tables", type=int, choices=(16, 64, 256), default=64)
    run_parser.add_argument("--fit-samples", type=int, default=131072)
    run_parser.add_argument("--validation-samples", type=int, default=32768)
    run_parser.add_argument("--positive-per-query", type=int, default=16)
    run_parser.add_argument("--hard-negative-per-query", type=int, default=8)
    run_parser.add_argument("--top-k", type=int, default=16)
    run_parser.add_argument("--ridge", type=float, default=0.001)
    run_parser.add_argument("--cg-iterations", type=int, default=128)
    run_parser.add_argument("--cg-tolerance", type=float, default=1e-7)
    run_parser.add_argument("--batch-size", type=int, default=8192)
    run_parser.add_argument("--eval-batch-size", type=int, default=8192)
    run_parser.add_argument("--device", default="cuda")
    run_parser.add_argument("--seed", type=int, default=0)
    run_parser.add_argument("--out-dir", type=Path, required=True)

    summarize_parser = commands.add_parser("summarize")
    summarize_parser.add_argument("--result-dir", type=Path, required=True)
    summarize_parser.add_argument("--out-report", type=Path, required=True)
    summarize_parser.add_argument("--top16-gate", type=float, default=0.5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run":
        run(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
