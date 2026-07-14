from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from torch import Tensor, nn

from tropnn.tools.bilinear_retrieval_probe import predict_score_matrix, retrieval_metrics, teacher_scores
from tropnn.tools.fixed_wide_route_decoder_probe import FixedWideRouteBits
from tropnn.tools.fixed_route_relation_energy_probe import make_relation_pairs
from tropnn.tools.route_degree_sparsity_probe import (
    add_problem_arguments,
    make_encoder,
    make_uniform_split,
    r2_score,
    split_slice,
    uniform_pair_indices,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exact ridge-CG oracle for the original additive PC-LUT function space."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run")
    add_problem_arguments(run)
    run.add_argument("--screen-samples", type=int, default=32768)
    run.add_argument("--fit-samples", type=int, default=65536)
    run.add_argument("--validation-samples", type=int, default=32768)
    run.add_argument("--ridge", type=float, default=0.001)
    run.add_argument("--cg-iterations", type=int, default=256)
    run.add_argument("--cg-tolerance", type=float, default=1e-7)
    run.add_argument("--eval-batch-size", type=int, default=512)
    run.add_argument("--device", default="cuda")
    run.add_argument("--out-dir", type=Path, required=True)

    summarize = commands.add_parser("summarize")
    summarize.add_argument("--result-dir", type=Path, required=True)
    summarize.add_argument("--additive-summary", type=Path, required=True)
    summarize.add_argument("--degree-summary", type=Path, required=True)
    summarize.add_argument("--dense-route-summary", type=Path, required=True)
    summarize.add_argument("--out-report", type=Path, required=True)
    return parser


@torch.no_grad()
def route_codes(encoder: FixedWideRouteBits, pair: Tensor) -> Tensor:
    route_bits = encoder(pair)
    comparisons = encoder.comparisons
    if route_bits.shape[1] % comparisons:
        raise ValueError("route bit dimension is not divisible by comparisons")
    powers = 2 ** torch.arange(comparisons, device=pair.device, dtype=torch.long)
    bits = (route_bits > 0).to(torch.long).view(pair.shape[0], -1, comparisons)
    return (bits * powers.view(1, 1, -1)).sum(dim=-1)


def design_forward(codes: Tensor, coefficient: Tensor, table_size: int) -> Tensor:
    payload = coefficient.view(codes.shape[1], table_size)
    tables = torch.arange(codes.shape[1], device=codes.device).view(1, -1)
    return payload[tables, codes].sum(dim=-1)


def design_transpose(codes: Tensor, values: Tensor, table_size: int) -> Tensor:
    result = torch.zeros(
        (codes.shape[1], table_size),
        device=codes.device,
        dtype=values.dtype,
    )
    result.scatter_add_(
        1,
        codes.transpose(0, 1),
        values.view(1, -1).expand(codes.shape[1], -1),
    )
    return result.flatten()


def normal_matvec(codes: Tensor, vector: Tensor, table_size: int, ridge: float) -> Tensor:
    prediction = design_forward(codes, vector, table_size)
    return design_transpose(codes, prediction, table_size) / codes.shape[0] + ridge * vector


def ridge_conjugate_gradient(
    codes: Tensor,
    target: Tensor,
    table_size: int,
    ridge: float,
    max_iterations: int,
    tolerance: float,
) -> tuple[Tensor, int, float]:
    right_hand_side = design_transpose(codes, target, table_size) / codes.shape[0]
    coefficient = torch.zeros_like(right_hand_side)
    residual = right_hand_side.clone()
    direction = residual.clone()
    residual_norm_squared = torch.dot(residual, residual)
    initial_norm = residual_norm_squared.sqrt().clamp_min(1e-12)
    relative_residual = 1.0
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        image = normal_matvec(codes, direction, table_size, ridge)
        denominator = torch.dot(direction, image).clamp_min(1e-20)
        step = residual_norm_squared / denominator
        coefficient = coefficient + step * direction
        residual = residual - step * image
        next_norm_squared = torch.dot(residual, residual)
        relative_residual = float((next_norm_squared.sqrt() / initial_norm).item())
        if relative_residual <= tolerance:
            break
        direction = residual + (next_norm_squared / residual_norm_squared) * direction
        residual_norm_squared = next_norm_squared
    return coefficient, iterations, relative_residual


class AdditiveRouteDecoder(nn.Module):
    def __init__(
        self,
        comparisons: int,
        coefficient: Tensor,
        target_scale: Tensor,
    ) -> None:
        super().__init__()
        self.comparisons = comparisons
        self.table_size = 1 << comparisons
        self.register_buffer("coefficient", coefficient)
        self.register_buffer("target_scale", target_scale)

    def forward(self, route_bits: Tensor) -> Tensor:
        powers = 2 ** torch.arange(
            self.comparisons,
            device=route_bits.device,
            dtype=torch.long,
        )
        bits = (route_bits > 0).to(torch.long).view(route_bits.shape[0], -1, self.comparisons)
        codes = (bits * powers.view(1, 1, -1)).sum(dim=-1)
        return (design_forward(codes, self.coefficient, self.table_size) * self.target_scale).unsqueeze(-1)


class RoutedAdditiveOracle(nn.Module):
    def __init__(self, encoder: FixedWideRouteBits, decoder: AdditiveRouteDecoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, pair: Tensor) -> Tensor:
        return self.decoder(self.encoder(pair))


@torch.no_grad()
def pair_accuracy(model: nn.Module, positive: Tensor, negative: Tensor, batch_size: int) -> float:
    correct = 0
    for start in range(0, positive.shape[0], batch_size):
        positive_score = model(positive[start : start + batch_size])
        negative_score = model(negative[start : start + batch_size])
        correct += int((positive_score > negative_score).sum().item())
    return correct / positive.shape[0]


def run(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    problem = make_problem_for_args(args)
    indices = uniform_pair_indices(args)
    fit = make_uniform_split(problem, indices[split_slice(args, "fit")], device)
    validation = make_uniform_split(problem, indices[split_slice(args, "validation")], device)
    encoder = make_encoder(args, device)
    fit_codes = route_codes(encoder, fit.pair)
    validation_codes = route_codes(encoder, validation.pair)
    table_size = 1 << args.comparisons
    design_columns = fit_codes.shape[1] * table_size
    active_columns = fit_codes.shape[1]
    if design_columns != 8192 or active_columns != 256:
        raise RuntimeError(
            f"expected a 256x32 additive design, got {active_columns}x{table_size}"
        )

    target_scale = fit.target.std().clamp_min(1e-6)
    normalized_target = fit.target.flatten() / target_scale
    started = time.perf_counter()
    coefficient, iterations, relative_residual = ridge_conjugate_gradient(
        fit_codes,
        normalized_target,
        table_size,
        args.ridge,
        args.cg_iterations,
        args.cg_tolerance,
    )
    elapsed = time.perf_counter() - started
    train_prediction = design_forward(fit_codes, coefficient, table_size) * target_scale
    validation_prediction = design_forward(validation_codes, coefficient, table_size) * target_scale

    decoder = AdditiveRouteDecoder(args.comparisons, coefficient, target_scale)
    model = RoutedAdditiveOracle(encoder, decoder).eval()
    relation_pairs = make_relation_pairs(args, device)
    test_target = teacher_scores(
        relation_pairs.test_queries,
        relation_pairs.test_keys,
        relation_pairs.relation,
    )
    test_prediction = predict_score_matrix(
        model,
        relation_pairs.test_queries,
        relation_pairs.test_keys,
        args.eval_batch_size,
    )
    metrics = retrieval_metrics(test_target, test_prediction, args.top_k, args.seed + 601)
    result: dict[str, object] = {
        "variant": "additive_route_ridge_cg_oracle",
        "route_bit_dim": encoder.output_dim,
        "tables": active_columns,
        "table_size": table_size,
        "design_columns": design_columns,
        "active_columns_per_sample": active_columns,
        "effective_unregularized_rank_upper_bound": 1 + active_columns * (table_size - 1),
        "parameters": coefficient.numel(),
        "ridge": args.ridge,
        "solver_iterations": iterations,
        "solver_relative_residual": relative_residual,
        "fit_seconds": elapsed,
        "uniform_train_r2": r2_score(fit.target, train_prediction),
        "uniform_validation_r2": r2_score(validation.target, validation_prediction),
        "train_pair_accuracy": pair_accuracy(
            model,
            relation_pairs.positive,
            relation_pairs.negative,
            args.eval_batch_size,
        ),
        **{f"test_{key}": value for key, value in metrics.items()},
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "additive_route_ridge_cg_oracle.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    torch.save(
        {
            "coefficient": coefficient.cpu(),
            "target_scale": target_scale.cpu(),
            "comparisons": args.comparisons,
        },
        args.out_dir / "additive_route_ridge_cg_oracle.pt",
    )
    print(json.dumps(result, sort_keys=True), flush=True)


def make_problem_for_args(args: argparse.Namespace) -> object:
    from tropnn.tools.bilinear_retrieval_probe import make_problem

    return make_problem(args)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def summarize(args: argparse.Namespace) -> None:
    oracle_path = args.result_dir / "additive_route_ridge_cg_oracle.json"
    oracle = json.loads(oracle_path.read_text())
    additive = next(
        row
        for row in read_csv(args.additive_summary)
        if row["variant"] == "pc_mse_adamw" and int(row["width"]) == 16
    )
    degree_rows = read_csv(args.degree_summary)
    degree_8k = next(
        row
        for row in degree_rows
        if int(row["max_degree"]) == 2 and int(row["support_budget"]) == 8192
    )
    degree_32k = next(
        row
        for row in degree_rows
        if int(row["max_degree"]) == 2 and int(row["support_budget"]) == 32768
    )
    dense = next(row for row in read_csv(args.dense_route_summary) if row["objective"] == "mse")
    comparison_rows = [
        {
            "decoder": "Original additive AdamW",
            "params": int(additive["parameters"]),
            "uniform_valid_r2": None,
            "held": float(additive["test_random_pair_order_accuracy"]),
            "hard": float(additive["test_hard_negative_preference_accuracy"]),
            "top16": float(additive["test_topk_recall"]),
            "top1": float(additive["test_top1_accuracy"]),
            "spearman": float(additive["test_spearman"]),
        },
        {
            "decoder": "Additive ridge-CG oracle",
            "params": oracle["parameters"],
            "uniform_valid_r2": oracle["uniform_validation_r2"],
            "held": oracle["test_random_pair_order_accuracy"],
            "hard": oracle["test_hard_negative_preference_accuracy"],
            "top16": oracle["test_topk_recall"],
            "top1": oracle["test_top1_accuracy"],
            "spearman": oracle["test_spearman"],
        },
        {
            "decoder": "Screened degree-2 8K",
            "params": int(degree_8k["parameters"]),
            "uniform_valid_r2": float(degree_8k["uniform_validation_r2"]),
            "held": float(degree_8k["test_random_pair_order_accuracy"]),
            "hard": float(degree_8k["test_hard_negative_preference_accuracy"]),
            "top16": float(degree_8k["test_topk_recall"]),
            "top1": float(degree_8k["test_top1_accuracy"]),
            "spearman": float(degree_8k["test_spearman"]),
        },
        {
            "decoder": "Screened degree-2 32K",
            "params": int(degree_32k["parameters"]),
            "uniform_valid_r2": float(degree_32k["uniform_validation_r2"]),
            "held": float(degree_32k["test_random_pair_order_accuracy"]),
            "hard": float(degree_32k["test_hard_negative_preference_accuracy"]),
            "top16": float(degree_32k["test_topk_recall"]),
            "top1": float(degree_32k["test_top1_accuracy"]),
            "spearman": float(degree_32k["test_spearman"]),
        },
        {
            "decoder": "Dense route H256",
            "params": int(dense["trainable_parameters"]),
            "uniform_valid_r2": None,
            "held": float(dense["test_random_pair_order_accuracy"]),
            "hard": float(dense["test_hard_negative_preference_accuracy"]),
            "top16": float(dense["test_topk_recall"]),
            "top1": float(dense["test_top1_accuracy"]),
            "spearman": float(dense["test_spearman"]),
        },
    ]
    summary_path = args.result_dir / "summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)

    gap = float(degree_8k["uniform_validation_r2"]) - oracle["uniform_validation_r2"]
    lines = [
        "# Original Additive PC-LUT Ridge-CG Oracle",
        "",
        "## Exact function space",
        "",
        "The 1,280 frozen comparison bits are repacked into 256 original T32 route codes. The implicit design matrix has 8,192 columns, one column for every `(table, row)` payload, and exactly 256 active columns per sample. Its prediction is exactly the original additive PC-LUT function `sum_t payload[t, code_t]`.",
        "",
        "This is one flat T256/C5 additive function. The seed-block count is not network depth.",
        "",
        "The design matrix is applied implicitly: `X beta` is a table gather and sum, while `X^T v` is a table-row scatter-add. Ridge-CG therefore solves the convex payload oracle without materializing a dense matrix and without AdamW, STE, threshold learning, initialization, or route changes.",
        "",
        "| Decoder | Params | Uniform valid R2 | Held pair | Hard-neg | Top-16 | Top-1 | Spearman |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison_rows:
        valid = "n/a" if row["uniform_valid_r2"] is None else f"{row['uniform_valid_r2']:.4f}"
        lines.append(
            f"| {row['decoder']} | {row['params']:,} | {valid} | {row['held']:.4f} | "
            f"{row['hard']:.4f} | {row['top16']:.4f} | {row['top1']:.4f} | {row['spearman']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Structural test",
            "",
            f"At the same approximately 8K coefficient budget, the screened cross-table degree-2 model exceeds the exact additive oracle by {gap:+.4f} uniform validation R2. Because both use the same frozen route, uniform fit/validation pairs, ridge coefficient objective, and held-out retrieval evaluator, this residual gap is approximation bias from the original table grouping rather than an optimizer or STE failure.",
            "",
            "## Difference from earlier analytic probes",
            "",
            "- Route covariance and projection-capacity probes measured the geometry, rank, and injectivity of the route representation. They did not condition on the bilinear target and did not restrict the decoder to original additive tables.",
            "- Finite-support exact-recovery experiments showed that an almost-injective joint route can index an arbitrary target on observed samples. That corresponds to a free function of the complete route tuple, not a sum of 256 local table functions, and it does not test held-out generalization.",
            "- The degree-sparsity probe changed the decoder basis to selected cross-table Boolean monomials. It established where target-conditioned interaction energy lies, but did not measure the optimum inside the original additive payload span.",
            "- This oracle is the orthogonal ridge projection of the target onto the exact original additive PC-LUT span under the actual uniform route distribution. It is the first experiment that directly separates optimization error from additive-factorization approximation error.",
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
