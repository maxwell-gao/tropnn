from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch
from torch import Tensor, nn

from tropnn.tools.bilinear_retrieval_probe import (
    evaluate_fixed_pairs,
    make_fixed_pairs,
    make_problem,
    predict_score_matrix,
    retrieval_metrics,
    sampled_pair_batch,
    teacher_scores,
    write_history,
)
from tropnn.tools.bitlinear_backprop_probe import PCLUTStudent


DENSE_VARIANTS = {
    "D4": ("mlp", 4),
    "D8": ("mlp", 8),
    "Bilinear": ("bilinear", 1),
}


class DenseResidualBlock(nn.Module):
    def __init__(self, width: int, depth: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.projection = nn.Linear(width, width)
        self.residual_scale = 1.0 / math.sqrt(depth)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.residual_scale * torch.nn.functional.gelu(self.projection(self.norm(x)))


class DenseResidualScorer(nn.Module):
    def __init__(self, input_dim: int, width: int, depth: int) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, width)
        self.blocks = nn.ModuleList(DenseResidualBlock(width, depth) for _ in range(depth))
        self.readout = nn.Linear(width, 1)

    def forward(self, pair: Tensor) -> Tensor:
        hidden = torch.nn.functional.gelu(self.input_projection(pair))
        for block in self.blocks:
            hidden = block(hidden)
        return self.readout(hidden)


class DenseBilinearScorer(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.weight = nn.Parameter(torch.zeros(input_dim, input_dim))
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, pair: Tensor) -> Tensor:
        query, key = pair.split(self.input_dim, dim=-1)
        return ((query @ self.weight) * key).sum(dim=-1, keepdim=True) + self.bias


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def matched_pc_lut_parameters(input_dim: int, depth: int, tables: int, comparisons: int, seed: int) -> int:
    model = PCLUTStudent(
        input_dim=2 * input_dim,
        output_dim=1,
        depth=depth,
        tables=tables,
        comparisons=comparisons,
        trainable_thresholds=False,
        seed=seed,
    )
    return parameter_count(model)


def dense_parameter_formula(pair_dim: int, width: int, depth: int) -> int:
    return (pair_dim + 1) * width + depth * (width * width + 3 * width) + width + 1


def closest_width(pair_dim: int, depth: int, target_parameters: int) -> int:
    return min(
        range(1, 4097),
        key=lambda width: abs(dense_parameter_formula(pair_dim, width, depth) - target_parameters),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dense controls for the PC-LUT pair-level bilinear retrieval probe.")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run")
    run.add_argument("--variant", choices=list(DENSE_VARIANTS), required=True)
    run.add_argument("--input-dim", type=int, default=32)
    run.add_argument("--train-queries", type=int, default=2048)
    run.add_argument("--train-keys", type=int, default=2048)
    run.add_argument("--test-queries", type=int, default=256)
    run.add_argument("--test-keys", type=int, default=512)
    run.add_argument("--max-value", type=int, default=15)
    run.add_argument("--tables", type=int, default=16)
    run.add_argument("--comparisons", type=int, default=5)
    run.add_argument("--steps", type=int, default=10000)
    run.add_argument("--batch-size", type=int, default=512)
    run.add_argument("--eval-pairs", type=int, default=8192)
    run.add_argument("--eval-every", type=int, default=1000)
    run.add_argument("--top-k", type=int, default=16)
    run.add_argument("--lr", type=float, default=0.005)
    run.add_argument("--device", default="cuda")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--out-dir", type=Path, required=True)

    summarize = commands.add_parser("summarize")
    summarize.add_argument("--result-dir", type=Path, required=True)
    summarize.add_argument("--pc-lut-summary", type=Path, required=True)
    summarize.add_argument("--out-report", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    problem = make_problem(args)
    family, depth = DENSE_VARIANTS[args.variant]
    target_parameters = None
    hidden_width = None
    if family == "mlp":
        target_parameters = matched_pc_lut_parameters(
            args.input_dim, depth, args.tables, args.comparisons, args.seed
        )
        hidden_width = closest_width(2 * args.input_dim, depth, target_parameters)
        model: nn.Module = DenseResidualScorer(2 * args.input_dim, hidden_width, depth)
    else:
        model = DenseBilinearScorer(args.input_dim)
    model = model.to(device)

    train_queries = problem.train_queries.to(device)
    train_keys = problem.train_keys.to(device)
    relation = problem.relation.to(device)
    fixed_pairs = make_fixed_pairs(problem, args, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    sample_generator = torch.Generator(device=device).manual_seed(args.seed + 307)
    history = [evaluate_fixed_pairs(model, fixed_pairs, args.batch_size, 0)]

    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        query_indices = torch.randint(
            0, train_queries.shape[0], (args.batch_size,), generator=sample_generator, device=device
        )
        key_indices = torch.randint(
            0, train_keys.shape[0], (args.batch_size,), generator=sample_generator, device=device
        )
        pair, target = sampled_pair_batch(
            train_queries, train_keys, relation, query_indices, key_indices
        )
        prediction = model(pair)
        loss = (prediction - target).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % args.eval_every == 0 or step == args.steps:
            row = evaluate_fixed_pairs(model, fixed_pairs, args.batch_size, step)
            history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    elapsed = time.perf_counter() - started

    model.eval()
    train_target = teacher_scores(
        problem.train_queries[: args.test_queries].to(device),
        problem.train_keys[: args.test_keys].to(device),
        relation,
    )
    train_prediction = predict_score_matrix(
        model,
        problem.train_queries[: args.test_queries].to(device),
        problem.train_keys[: args.test_keys].to(device),
        args.batch_size,
    )
    test_target = teacher_scores(problem.test_queries.to(device), problem.test_keys.to(device), relation)
    test_prediction = predict_score_matrix(
        model, problem.test_queries.to(device), problem.test_keys.to(device), args.batch_size
    )
    result: dict[str, object] = {
        "variant": args.variant,
        "family": family,
        "depth": depth,
        "hidden_width": hidden_width,
        "target_pc_lut_parameters": target_parameters,
        "parameter_count": parameter_count(model),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "elapsed_seconds": elapsed,
        "steps_per_second": args.steps / elapsed,
        "train": retrieval_metrics(train_target, train_prediction, args.top_k, args.seed + 503),
        "test": retrieval_metrics(test_target, test_prediction, args.top_k, args.seed + 601),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"{args.variant}.json").write_text(json.dumps(result, indent=2) + "\n")
    write_history(args.out_dir / f"{args.variant}_history.csv", history)
    print(json.dumps(result, sort_keys=True), flush=True)


def flatten_result(result: dict[str, object]) -> dict[str, object]:
    row = {key: value for key, value in result.items() if key not in {"train", "test"}}
    for split in ("train", "test"):
        metrics = result[split]
        assert isinstance(metrics, dict)
        row.update({f"{split}_{key}": value for key, value in metrics.items()})
    return row


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def metric(row: dict[str, object] | dict[str, str], name: str) -> float:
    return float(row[name])


def summarize(args: argparse.Namespace) -> None:
    dense_rows = [flatten_result(json.loads(path.read_text())) for path in sorted(args.result_dir.glob("*.json"))]
    if not dense_rows:
        raise RuntimeError(f"No dense result JSON files found in {args.result_dir}")
    summary_path = args.result_dir / "summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dense_rows[0]))
        writer.writeheader()
        writer.writerows(dense_rows)

    pc_rows = read_csv(args.pc_lut_summary)
    selected_pc = [row for row in pc_rows if row["variant"] in {"E4", "E8", "F4", "F8"}]
    combined: list[dict[str, object] | dict[str, str]] = [*selected_pc, *dense_rows]
    lines = [
        "# Dense Controls for Pair-Level Bilinear Retrieval",
        "",
        "The target is the same fixed binary bilinear relation `s(q,k) = q^T A k` used by the PC-LUT probe. All models use the same sampled pairs, 10k-step score-MSE objective, held-out queries and keys, and retrieval metrics.",
        "",
        "`D4` and `D8` are residual GELU MLPs whose parameter counts are matched to fixed-threshold `F4` and `F8`. `Bilinear` directly learns the matrix in the teacher function and is therefore an architectural calibration, not a parameter-matched generic baseline.",
        "",
        "| Variant | Parameters | Test R2 | Spearman | Top-16 recall | Top-1 | Hard-negative | Pair order | Steps/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in combined:
        steps_per_second = row.get("steps_per_second", "nan")
        parameters = row.get("parameter_count", row.get("parameters"))
        lines.append(
            f"| {row['variant']} | {int(float(parameters)):,} | "
            f"{metric(row, 'test_score_r2'):.4f} | {metric(row, 'test_spearman'):.4f} | "
            f"{metric(row, 'test_topk_recall'):.4f} | {metric(row, 'test_top1_accuracy'):.4f} | "
            f"{metric(row, 'test_hard_negative_preference_accuracy'):.4f} | "
            f"{metric(row, 'test_random_pair_order_accuracy'):.4f} | {float(steps_per_second):.1f} |"
        )
    lines.extend(
        [
            "",
            "Random baselines are 0.03125 for Top-16 recall, 0.001953 for exact Top-1, and 0.5 for pair ordering.",
            "",
            "## Interpretation boundary",
            "",
            "The bilinear scorer measures the advantage of using the correct inner-product function class. The parameter-matched MLP measures generic dense nonlinear learning. PC-LUT should therefore be compared primarily with D4/D8 for parameter efficiency and with Bilinear only to quantify the remaining gap to an exact relation-selection inductive bias.",
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
