from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from tropnn.tools.bilinear_retrieval_probe import (
    make_problem,
    predict_score_matrix,
    retrieval_metrics,
    teacher_scores,
)
from tropnn.tools.fixed_route_relation_energy_probe import RelationPairs


DECODERS = ("kendall", "relative", "relative_binary", "absolute")
PERMUTATIONS = tuple(itertools.permutations(range(4)))


def permutation_rank(order: Tensor) -> Tensor:
    rank = torch.zeros(order.shape[:-1], device=order.device, dtype=torch.long)
    factorials = (6, 2, 1)
    for position, factorial in enumerate(factorials):
        smaller = (order[..., position + 1 :] < order[..., position : position + 1]).sum(dim=-1)
        rank += smaller * factorial
    return rank


def permutation_tables() -> tuple[Tensor, Tensor, Tensor]:
    lookup = {permutation: index for index, permutation in enumerate(PERMUTATIONS)}
    inverse = torch.empty(24, dtype=torch.long)
    composition = torch.empty(24, 24, dtype=torch.long)
    lengths = torch.empty(24, dtype=torch.long)
    for left_index, left in enumerate(PERMUTATIONS):
        inverse_permutation = tuple(left.index(position) for position in range(4))
        inverse[left_index] = lookup[inverse_permutation]
        lengths[left_index] = sum(
            left[first] > left[second]
            for first in range(4)
            for second in range(first + 1, 4)
        )
        for right_index, right in enumerate(PERMUTATIONS):
            composed = tuple(left[right[position]] for position in range(4))
            composition[left_index, right_index] = lookup[composed]
    return inverse, composition, lengths


def make_clique_anchors(input_dim: int, tables: int, seed: int) -> Tensor:
    if input_dim < 4:
        raise ValueError(f"input_dim must be at least four, got {input_dim}")
    generator = torch.Generator(device="cpu").manual_seed(seed + 1709)
    return torch.stack([torch.randperm(input_dim, generator=generator)[:4] for _ in range(tables)])


class LocalS4Router(nn.Module):
    def __init__(self, input_dim: int, tables: int, seed: int) -> None:
        super().__init__()
        inverse, composition, lengths = permutation_tables()
        self.input_dim = input_dim
        self.tables = tables
        self.register_buffer("anchors", make_clique_anchors(input_dim, tables, seed))
        self.register_buffer("inverse", inverse)
        self.register_buffer("composition", composition)
        self.register_buffer("lengths", lengths)

    def route(self, values: Tensor) -> Tensor:
        selected = values[:, self.anchors.flatten()].view(values.shape[0], self.tables, 4)
        # Integer-valued probe inputs contain ties. Stable sorting implements a
        # fixed anchor-position tie break and always returns a legal S4 chamber.
        order = torch.argsort(selected, dim=-1, stable=True)
        return permutation_rank(order)

    def pair_codes(self, query_route: Tensor, key_route: Tensor, decoder: str) -> Tensor:
        if decoder == "absolute":
            return query_route * 24 + key_route
        inverse_query = self.inverse[query_route]
        relative = self.composition[inverse_query, key_route]
        if decoder == "kendall":
            return self.lengths[relative]
        return relative


def decoder_rows(decoder: str) -> int:
    if decoder == "kendall":
        return 7
    if decoder == "absolute":
        return 24 * 24
    return 24


class CategoricalRelationScorer(nn.Module):
    def __init__(
        self,
        router: LocalS4Router,
        decoder: str,
        coefficient: Tensor,
        bias: Tensor,
    ) -> None:
        super().__init__()
        self.router = router
        self.decoder = decoder
        self.rows = decoder_rows(decoder)
        self.register_buffer("coefficient", coefficient.view(router.tables, self.rows))
        self.register_buffer("bias", bias.reshape(()))
        self.register_buffer("table_index", torch.arange(router.tables).view(1, -1))

    def forward(self, pair: Tensor) -> Tensor:
        query, key = pair.split(self.router.input_dim, dim=-1)
        codes = self.router.pair_codes(self.router.route(query), self.router.route(key), self.decoder)
        table_index = self.table_index.to(pair.device)
        score = self.coefficient[table_index, codes].sum(dim=-1) / math.sqrt(self.router.tables)
        return (score + self.bias).unsqueeze(-1)


@dataclass(frozen=True)
class UniformPairs:
    query_index: Tensor
    key_index: Tensor
    target: Tensor


def make_uniform_pairs(
    queries: Tensor,
    keys: Tensor,
    relation: Tensor,
    samples: int,
    seed: int,
) -> UniformPairs:
    generator = torch.Generator(device=queries.device).manual_seed(seed)
    query_index = torch.randint(0, queries.shape[0], (samples,), generator=generator, device=queries.device)
    key_index = torch.randint(0, keys.shape[0], (samples,), generator=generator, device=keys.device)
    query = queries[query_index]
    key = keys[key_index]
    target = ((query @ relation) * key).sum(dim=-1)
    return UniformPairs(query_index, key_index, target)


def make_relation_pairs(
    args: argparse.Namespace,
    problem: object,
    relation: Tensor,
    device: torch.device,
) -> RelationPairs:
    queries = problem.train_queries.to(device)
    keys = problem.train_keys.to(device)
    scores = teacher_scores(queries, keys, relation)
    positive_count = args.positive_per_query
    hard_count = args.hard_negative_per_query
    if hard_count > positive_count:
        raise ValueError("hard-negative-per-query cannot exceed positive-per-query")
    excluded_count = max(64, positive_count + hard_count)
    top_indices = torch.topk(scores, k=excluded_count, dim=-1).indices
    positive_indices = top_indices[:, :positive_count]
    hard_indices = top_indices[:, positive_count : positive_count + hard_count]

    random_count = positive_count - hard_count
    generator = torch.Generator(device=device).manual_seed(args.seed + 701)
    random_indices = torch.randint(
        0,
        keys.shape[0],
        (queries.shape[0], random_count),
        generator=generator,
        device=device,
    )
    invalid = (random_indices.unsqueeze(-1) == top_indices.unsqueeze(1)).any(dim=-1)
    while bool(invalid.any()):
        random_indices[invalid] = torch.randint(
            0,
            keys.shape[0],
            (int(invalid.sum().item()),),
            generator=generator,
            device=device,
        )
        invalid = (random_indices.unsqueeze(-1) == top_indices.unsqueeze(1)).any(dim=-1)
    negative_indices = torch.cat([hard_indices, random_indices], dim=-1)
    query_indices = torch.arange(queries.shape[0], device=device).unsqueeze(1).expand(-1, positive_count)
    positive = torch.cat([queries[query_indices], keys[positive_indices]], dim=-1).reshape(-1, 2 * args.input_dim)
    negative = torch.cat([queries[query_indices], keys[negative_indices]], dim=-1).reshape(-1, 2 * args.input_dim)
    return RelationPairs(
        positive=positive,
        negative=negative,
        positive_target=scores.gather(1, positive_indices).reshape(-1, 1),
        negative_target=scores.gather(1, negative_indices).reshape(-1, 1),
        test_queries=problem.test_queries.to(device),
        test_keys=problem.test_keys.to(device),
        relation=relation,
    )


def pair_route_codes(
    router: LocalS4Router,
    query_routes: Tensor,
    key_routes: Tensor,
    pairs: UniformPairs,
    decoder: str,
) -> Tensor:
    return router.pair_codes(query_routes[pairs.query_index], key_routes[pairs.key_index], decoder)


def categorical_prediction(codes: Tensor, coefficient: Tensor, bias: Tensor, tables: int) -> Tensor:
    rows = coefficient.numel() // tables
    table_offsets = torch.arange(tables, device=codes.device).view(1, -1) * rows
    flat_codes = codes + table_offsets
    return coefficient[flat_codes].sum(dim=-1) / math.sqrt(tables) + bias


def categorical_rhs(codes: Tensor, target: Tensor, parameters: int, batch_size: int) -> Tensor:
    rows = parameters // codes.shape[1]
    scale = 1.0 / math.sqrt(codes.shape[1])
    result = torch.zeros(parameters + 1, device=codes.device, dtype=torch.float32)
    table_offsets = torch.arange(codes.shape[1], device=codes.device).view(1, -1) * rows
    for start in range(0, codes.shape[0], batch_size):
        batch_codes = codes[start : start + batch_size] + table_offsets
        batch_target = target[start : start + batch_size].float()
        result[:-1].scatter_add_(
            0,
            batch_codes.reshape(-1),
            (batch_target[:, None] * scale).expand_as(batch_codes).reshape(-1),
        )
        result[-1] += batch_target.sum()
    return result / codes.shape[0]


def categorical_normal_matvec(
    codes: Tensor,
    vector: Tensor,
    ridge: float,
    batch_size: int,
) -> Tensor:
    tables = codes.shape[1]
    parameters = vector.numel() - 1
    rows = parameters // tables
    scale = 1.0 / math.sqrt(tables)
    result = ridge * vector
    table_offsets = torch.arange(tables, device=codes.device).view(1, -1) * rows
    for start in range(0, codes.shape[0], batch_size):
        batch_codes = codes[start : start + batch_size] + table_offsets
        prediction = vector[:-1][batch_codes].sum(dim=-1) * scale + vector[-1]
        result[:-1].scatter_add_(
            0,
            batch_codes.reshape(-1),
            (prediction[:, None] * scale).expand_as(batch_codes).reshape(-1),
        )
        result[-1] += prediction.sum()
    return result / codes.shape[0]


def ridge_cg(
    codes: Tensor,
    target: Tensor,
    rows: int,
    ridge: float,
    iterations: int,
    tolerance: float,
    batch_size: int,
) -> tuple[Tensor, Tensor, int, float]:
    parameters = codes.shape[1] * rows
    target_mean = target.mean()
    target_std = target.std(unbiased=False).clamp_min(1e-6)
    normalized_target = ((target - target_mean) / target_std).float()
    right_hand_side = categorical_rhs(codes, normalized_target, parameters, batch_size)
    solution = torch.zeros_like(right_hand_side)
    residual = right_hand_side.clone()
    direction = residual.clone()
    residual_squared = torch.dot(residual, residual)
    initial_norm = residual_squared.sqrt().clamp_min(1e-12)
    relative_residual = 1.0
    completed = 0
    for completed in range(1, iterations + 1):
        image = categorical_normal_matvec(codes, direction, ridge, batch_size)
        denominator = torch.dot(direction, image).clamp_min(1e-20)
        step = residual_squared / denominator
        solution += step * direction
        residual -= step * image
        next_squared = torch.dot(residual, residual)
        relative_residual = float((next_squared.sqrt() / initial_norm).item())
        if relative_residual <= tolerance:
            break
        direction = residual + (next_squared / residual_squared) * direction
        residual_squared = next_squared
    coefficient = solution[:-1] * target_std
    bias = solution[-1] * target_std + target_mean
    return coefficient, bias, completed, relative_residual


def binary_sign_fit(codes: Tensor, coefficient: Tensor, target: Tensor, tables: int) -> tuple[Tensor, Tensor]:
    signs = torch.where(coefficient >= 0, torch.ones_like(coefficient), -torch.ones_like(coefficient))
    raw = categorical_prediction(codes, signs, torch.zeros((), device=codes.device), tables)
    centered_raw = raw - raw.mean()
    centered_target = target - target.mean()
    scale = torch.dot(centered_raw, centered_target) / torch.dot(centered_raw, centered_raw).clamp_min(1e-12)
    if scale < 0:
        signs = -signs
        raw = -raw
        scale = -scale
    bias = target.mean() - scale * raw.mean()
    return signs * scale, bias


def r2_score(target: Tensor, prediction: Tensor) -> float:
    target = target.to(torch.float64)
    prediction = prediction.to(torch.float64)
    residual = (target - prediction).square().sum()
    total = (target - target.mean()).square().sum().clamp_min(1e-12)
    return float((1.0 - residual / total).item())


@torch.no_grad()
def positive_negative_accuracy(model: nn.Module, positive: Tensor, negative: Tensor, batch_size: int) -> float:
    correct = 0
    for start in range(0, positive.shape[0], batch_size):
        positive_score = model(positive[start : start + batch_size])
        negative_score = model(negative[start : start + batch_size])
        correct += int((positive_score > negative_score).sum().item())
    return correct / positive.shape[0]


def active_state_coverage(fit_codes: Tensor, validation_codes: Tensor, rows: int) -> float:
    seen = torch.zeros((fit_codes.shape[1], rows), device=fit_codes.device, dtype=torch.bool)
    table_index = torch.arange(fit_codes.shape[1], device=fit_codes.device).view(1, -1)
    seen[table_index, fit_codes] = True
    return float(seen[table_index, validation_codes].to(torch.float64).mean().item())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local-S4 Coxeter relation decoder probe.")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run")
    run.add_argument("--input-dim", type=int, default=32)
    run.add_argument("--train-queries", type=int, default=2048)
    run.add_argument("--train-keys", type=int, default=2048)
    run.add_argument("--test-queries", type=int, default=256)
    run.add_argument("--test-keys", type=int, default=512)
    run.add_argument("--max-value", type=int, default=15)
    run.add_argument(
        "--teacher",
        choices=("random_bilinear", "permutation_invariant"),
        default="random_bilinear",
    )
    run.add_argument("--tables", type=int, default=64)
    run.add_argument("--fit-samples", type=int, default=131072)
    run.add_argument("--validation-samples", type=int, default=32768)
    run.add_argument("--positive-per-query", type=int, default=16)
    run.add_argument("--hard-negative-per-query", type=int, default=8)
    run.add_argument("--top-k", type=int, default=16)
    run.add_argument("--ridge", type=float, default=0.001)
    run.add_argument("--cg-iterations", type=int, default=128)
    run.add_argument("--cg-tolerance", type=float, default=1e-7)
    run.add_argument("--batch-size", type=int, default=8192)
    run.add_argument("--eval-batch-size", type=int, default=8192)
    run.add_argument("--device", default="cuda")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--out-dir", type=Path, required=True)

    summarize = commands.add_parser("summarize")
    summarize.add_argument("--result-dir", type=Path, required=True)
    summarize.add_argument("--out-report", type=Path, required=True)
    return parser


def fit_decoder(
    args: argparse.Namespace,
    router: LocalS4Router,
    train_query_routes: Tensor,
    train_key_routes: Tensor,
    fit: UniformPairs,
    validation: UniformPairs,
    decoder: str,
) -> tuple[CategoricalRelationScorer, dict[str, object]]:
    base_decoder = "relative" if decoder == "relative_binary" else decoder
    rows = decoder_rows(base_decoder)
    fit_codes = pair_route_codes(router, train_query_routes, train_key_routes, fit, base_decoder)
    validation_codes = pair_route_codes(router, train_query_routes, train_key_routes, validation, base_decoder)
    started = time.perf_counter()
    coefficient, bias, iterations, relative_residual = ridge_cg(
        fit_codes,
        fit.target,
        rows,
        args.ridge,
        args.cg_iterations,
        args.cg_tolerance,
        args.batch_size,
    )
    if decoder == "relative_binary":
        coefficient, bias = binary_sign_fit(fit_codes, coefficient, fit.target, args.tables)
    elapsed = time.perf_counter() - started
    fit_prediction = categorical_prediction(fit_codes, coefficient, bias, args.tables)
    validation_prediction = categorical_prediction(validation_codes, coefficient, bias, args.tables)
    model = CategoricalRelationScorer(router, base_decoder, coefficient, bias).eval()
    metadata: dict[str, object] = {
        "decoder": decoder,
        "parameters": coefficient.numel() + 1,
        "logical_payload_bits": coefficient.numel() if decoder == "relative_binary" else None,
        "solver": "ridge_cg_then_sign" if decoder == "relative_binary" else "ridge_cg",
        "solver_iterations": iterations,
        "solver_relative_residual": relative_residual,
        "elapsed_seconds": elapsed,
        "uniform_fit_r2": r2_score(fit.target, fit_prediction),
        "uniform_validation_r2": r2_score(validation.target, validation_prediction),
        "validation_active_state_seen_fraction": active_state_coverage(fit_codes, validation_codes, rows),
    }
    return model, metadata


def run(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    problem = make_problem(args)
    train_queries = problem.train_queries.to(device)
    train_keys = problem.train_keys.to(device)
    relation = problem.relation.to(device)
    if args.teacher == "permutation_invariant":
        relation = torch.eye(args.input_dim, device=device) / math.sqrt(args.input_dim)
    router = LocalS4Router(args.input_dim, args.tables, args.seed).to(device)
    train_query_routes = router.route(train_queries)
    train_key_routes = router.route(train_keys)
    fit = make_uniform_pairs(train_queries, train_keys, relation, args.fit_samples, args.seed + 2003)
    validation = make_uniform_pairs(
        train_queries,
        train_keys,
        relation,
        args.validation_samples,
        args.seed + 2017,
    )
    relation_pairs = make_relation_pairs(args, problem, relation, device)
    test_target = teacher_scores(relation_pairs.test_queries, relation_pairs.test_keys, relation)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for decoder in DECODERS:
        model, metadata = fit_decoder(
            args,
            router,
            train_query_routes,
            train_key_routes,
            fit,
            validation,
            decoder,
        )
        prediction = predict_score_matrix(
            model,
            relation_pairs.test_queries,
            relation_pairs.test_keys,
            args.eval_batch_size,
        )
        metrics = retrieval_metrics(test_target, prediction, args.top_k, args.seed + 601)
        result: dict[str, object] = {
            "seed": args.seed,
            "teacher": args.teacher,
            "input_dim": args.input_dim,
            "tables": args.tables,
            "comparisons_per_table": 6,
            "permutation_order": 4,
            "fit_samples": args.fit_samples,
            "validation_samples": args.validation_samples,
            "ridge": args.ridge,
            "fixed_clique_anchors": True,
            "fixed_stable_tie_break": True,
            "train_pair_accuracy": positive_negative_accuracy(
                model,
                relation_pairs.positive,
                relation_pairs.negative,
                args.eval_batch_size,
            ),
            **metadata,
            **{f"test_{key}": value for key, value in metrics.items()},
        }
        output = args.out_dir / f"seed{args.seed}_{decoder}.json"
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, sort_keys=True), flush=True)


def summarize(args: argparse.Namespace) -> None:
    rows = [json.loads(path.read_text()) for path in args.result_dir.glob("seed*_*.json")]
    if not rows:
        raise RuntimeError(f"no Coxeter relation results in {args.result_dir}")
    rows.sort(key=lambda row: (DECODERS.index(row["decoder"]), row["seed"]))
    fieldnames = sorted({key for row in rows for key in row})
    summary_path = args.result_dir / "summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    grouped = {decoder: [row for row in rows if row["decoder"] == decoder] for decoder in DECODERS}

    def mean(decoder: str, metric: str) -> float:
        return statistics.mean(float(row[metric]) for row in grouped[decoder])

    setup = rows[0]
    lines = [
        "# Local-S4 Coxeter Relation Decoder",
        "",
        "This probe replaces arbitrary C6 routes by complete comparisons on four anchors. Each object receives a legal permutation in S4. Pair relations are decoded either from Coxeter length, the full relative permutation, or the absolute permutation pair. Routes and anchors remain fixed; scalar payloads are fit by convex ridge-CG on uniform teacher pairs.",
        "",
        "## Setup",
        "",
        f"- Input dimension: {setup['input_dim']}",
        f"- Teacher: {setup['teacher']}",
        f"- Local S4 tables: {setup['tables']}; comparisons per table: 6",
        f"- Uniform fit/validation pairs per seed: {setup['fit_samples']:,}/{setup['validation_samples']:,}",
        f"- Seeds: {', '.join(str(seed) for seed in sorted({int(row['seed']) for row in rows}))}",
        "- Teacher and held-object retrieval protocol match the existing fixed-route relation-energy probe.",
        "",
        "## Per-seed results",
        "",
        "| Decoder | Seed | Params | Fit R2 | Valid R2 | Train pair | Held pair | Hard-neg | Top-16 | Top-1 | Spearman | Seen state |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['decoder']} | {row['seed']} | {row['parameters']:,} | "
            f"{row['uniform_fit_r2']:.4f} | {row['uniform_validation_r2']:.4f} | "
            f"{row['train_pair_accuracy']:.4f} | {row['test_random_pair_order_accuracy']:.4f} | "
            f"{row['test_hard_negative_preference_accuracy']:.4f} | {row['test_topk_recall']:.4f} | "
            f"{row['test_top1_accuracy']:.4f} | {row['test_spearman']:.4f} | "
            f"{row['validation_active_state_seen_fraction']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Seed means",
            "",
            "| Decoder | Params | Valid R2 | Held pair | Hard-neg | Top-16 | Top-1 | Spearman |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for decoder in DECODERS:
        lines.append(
            f"| {decoder} | {int(grouped[decoder][0]['parameters']):,} | "
            f"{mean(decoder, 'uniform_validation_r2'):.4f} | "
            f"{mean(decoder, 'test_random_pair_order_accuracy'):.4f} | "
            f"{mean(decoder, 'test_hard_negative_preference_accuracy'):.4f} | "
            f"{mean(decoder, 'test_topk_recall'):.4f} | "
            f"{mean(decoder, 'test_top1_accuracy'):.4f} | "
            f"{mean(decoder, 'test_spearman'):.4f} |"
        )
    relative_gap = mean("absolute", "test_topk_recall") - mean("relative", "test_topk_recall")
    length_gain = mean("relative", "test_topk_recall") - mean("kendall", "test_topk_recall")
    binary_gap = mean("relative", "test_topk_recall") - mean("relative_binary", "test_topk_recall")
    lines.extend(
        [
            "",
            "## Decision diagnostics",
            "",
            f"- Full relative permutation minus Kendall length changes Top-16 by `{length_gain:+.4f}`. This measures information lost by compressing the Coxeter element to inversion count.",
            f"- Absolute joint permutation minus relative permutation changes Top-16 by `{relative_gap:+.4f}`. A large positive gap rejects the left-invariant Coxeter quotient for this teacher.",
            f"- Float relative minus binary-sign relative changes Top-16 by `{binary_gap:+.4f}`. This tests whether the unary binary-payload result transfers to relation scoring.",
            "- If both relative and absolute remain weak, local ordinal routes themselves are insufficient. If absolute succeeds while relative fails, coordinate-frame information rather than optimization is essential.",
        ]
    )
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text("\n".join(lines) + "\n")
    print(summary_path)
    print(args.out_report)


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run":
        run(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
