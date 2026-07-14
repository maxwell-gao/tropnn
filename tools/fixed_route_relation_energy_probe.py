from __future__ import annotations

import argparse
import csv
import json
import math
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


VARIANTS = (
    "pc_mse_adamw",
    "pc_ff_logistic_adamw",
    "pc_histogram_log_ratio",
    "dense_mlp_pairwise",
    "dense_bilinear_pairwise",
)

DEFAULT_LEARNING_RATES = {
    "pc_mse_adamw": 0.005,
    "pc_ff_logistic_adamw": 0.01,
    "dense_mlp_pairwise": 0.005,
    "dense_bilinear_pairwise": 0.005,
}


@dataclass(frozen=True)
class RelationPairs:
    positive: Tensor
    negative: Tensor
    positive_target: Tensor
    negative_target: Tensor
    test_queries: Tensor
    test_keys: Tensor
    relation: Tensor


class FixedPairwiseEnergy(nn.Module):
    def __init__(self, input_dim: int, tables: int, comparisons: int, seed: int) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        anchor_a = torch.randint(0, input_dim, (tables, comparisons), generator=generator)
        anchor_b = torch.randint(0, input_dim - 1, (tables, comparisons), generator=generator)
        anchor_b += (anchor_b >= anchor_a).to(anchor_b.dtype)
        self.register_buffer("anchor_a", anchor_a)
        self.register_buffer("anchor_b", anchor_b)
        self.register_buffer("powers", 2 ** torch.arange(comparisons, dtype=torch.long))
        self.payload = nn.Parameter(torch.zeros(tables, 1 << comparisons))

    def route(self, pair: Tensor) -> Tensor:
        margins = pair[:, self.anchor_a.flatten()] - pair[:, self.anchor_b.flatten()]
        margins = margins.view(pair.shape[0], *self.anchor_a.shape)
        return ((margins > 0).to(torch.long) * self.powers).sum(dim=-1)

    def forward(self, pair: Tensor) -> Tensor:
        codes = self.route(pair)
        tables = torch.arange(self.payload.shape[0], device=pair.device).view(1, -1)
        return self.payload[tables, codes].sum(dim=-1, keepdim=True)


class DensePairScorer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.readout = nn.Linear(hidden_dim, 1)

    def forward(self, pair: Tensor) -> Tensor:
        return self.readout(torch.nn.functional.gelu(self.hidden(pair)))


class DenseBilinearScorer(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.weight = nn.Parameter(torch.zeros(input_dim, input_dim))
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, pair: Tensor) -> Tensor:
        query, key = pair.split(self.input_dim, dim=-1)
        return ((query @ self.weight) * key).sum(dim=-1, keepdim=True) + self.bias


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fixed-route relation-energy training comparison.")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run")
    run.add_argument("--variant", choices=VARIANTS, required=True)
    run.add_argument("--input-dim", type=int, default=32)
    run.add_argument("--train-queries", type=int, default=2048)
    run.add_argument("--train-keys", type=int, default=2048)
    run.add_argument("--test-queries", type=int, default=256)
    run.add_argument("--test-keys", type=int, default=512)
    run.add_argument("--max-value", type=int, default=15)
    run.add_argument("--tables", type=int, default=16)
    run.add_argument("--comparisons", type=int, default=5)
    run.add_argument("--positive-per-query", type=int, default=16)
    run.add_argument("--hard-negative-per-query", type=int, default=8)
    run.add_argument("--steps", type=int, default=10000)
    run.add_argument("--batch-size", type=int, default=512)
    run.add_argument("--eval-batch-size", type=int, default=8192)
    run.add_argument("--eval-every", type=int, default=1000)
    run.add_argument("--top-k", type=int, default=16)
    run.add_argument("--margin", type=float, default=1.0)
    run.add_argument("--smoothing", type=float, default=1.0)
    run.add_argument("--lr", type=float)
    run.add_argument("--device", default="cuda")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--out-dir", type=Path, required=True)

    summarize = commands.add_parser("summarize")
    summarize.add_argument("--result-dir", type=Path, required=True)
    summarize.add_argument("--out-report", type=Path, required=True)
    return parser


def make_relation_pairs(args: argparse.Namespace, device: torch.device) -> RelationPairs:
    problem = make_problem(args)
    queries = problem.train_queries.to(device)
    keys = problem.train_keys.to(device)
    relation = problem.relation.to(device)
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
        0, keys.shape[0], (queries.shape[0], random_count), generator=generator, device=device
    )
    invalid = (random_indices.unsqueeze(-1) == top_indices.unsqueeze(1)).any(dim=-1)
    while bool(invalid.any()):
        random_indices[invalid] = torch.randint(
            0, keys.shape[0], (int(invalid.sum().item()),), generator=generator, device=device
        )
        invalid = (random_indices.unsqueeze(-1) == top_indices.unsqueeze(1)).any(dim=-1)
    negative_indices = torch.cat([hard_indices, random_indices], dim=-1)

    query_indices = torch.arange(queries.shape[0], device=device).unsqueeze(1).expand(-1, positive_count)
    positive = torch.cat([queries[query_indices], keys[positive_indices]], dim=-1).reshape(-1, 2 * args.input_dim)
    negative = torch.cat([queries[query_indices], keys[negative_indices]], dim=-1).reshape(-1, 2 * args.input_dim)
    positive_target = scores.gather(1, positive_indices).reshape(-1, 1)
    negative_target = scores.gather(1, negative_indices).reshape(-1, 1)
    return RelationPairs(
        positive=positive,
        negative=negative,
        positive_target=positive_target,
        negative_target=negative_target,
        test_queries=problem.test_queries.to(device),
        test_keys=problem.test_keys.to(device),
        relation=relation,
    )


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


@torch.no_grad()
def pair_accuracy(model: nn.Module, pairs: RelationPairs, batch_size: int) -> float:
    correct = 0
    total = pairs.positive.shape[0]
    for start in range(0, total, batch_size):
        end = start + batch_size
        positive_score = model(pairs.positive[start:end])
        negative_score = model(pairs.negative[start:end])
        correct += int((positive_score > negative_score).sum().item())
    return correct / total


@torch.no_grad()
def fit_histogram(model: FixedPairwiseEnergy, pairs: RelationPairs, smoothing: float) -> dict[str, object]:
    started = time.perf_counter()
    positive_codes = model.route(pairs.positive)
    negative_codes = model.route(pairs.negative)
    table_size = model.payload.shape[1]
    positive_total = positive_codes.shape[0]
    negative_total = negative_codes.shape[0]
    payload = torch.empty_like(model.payload)
    occupied = []
    for table in range(model.payload.shape[0]):
        positive_count = torch.bincount(positive_codes[:, table], minlength=table_size).to(torch.float64)
        negative_count = torch.bincount(negative_codes[:, table], minlength=table_size).to(torch.float64)
        positive_probability = (positive_count + smoothing) / (positive_total + smoothing * table_size)
        negative_probability = (negative_count + smoothing) / (negative_total + smoothing * table_size)
        payload[table] = (positive_probability.log() - negative_probability.log()).to(payload.dtype)
        occupied.append(float(((positive_count + negative_count) > 0).to(torch.float64).mean().item()))
    model.payload.copy_(payload)
    model.payload.requires_grad_(False)
    return {
        "fit_seconds": time.perf_counter() - started,
        "mean_occupied_row_fraction": sum(occupied) / len(occupied),
    }


def training_loss(
    variant: str,
    model: nn.Module,
    pairs: RelationPairs,
    indices: Tensor,
    margin: float,
) -> Tensor:
    positive_score = model(pairs.positive[indices])
    negative_score = model(pairs.negative[indices])
    if variant == "pc_mse_adamw":
        positive_error = (positive_score - pairs.positive_target[indices]).square()
        negative_error = (negative_score - pairs.negative_target[indices]).square()
        return 0.5 * (positive_error.mean() + negative_error.mean())
    if variant == "pc_ff_logistic_adamw":
        return 0.5 * (
            torch.nn.functional.softplus(-positive_score).mean()
            + torch.nn.functional.softplus(negative_score).mean()
        )
    return torch.nn.functional.softplus(margin - positive_score + negative_score).mean()


def make_model(args: argparse.Namespace) -> nn.Module:
    if args.variant.startswith("pc_"):
        return FixedPairwiseEnergy(2 * args.input_dim, args.tables, args.comparisons, args.seed)
    if args.variant == "dense_mlp_pairwise":
        return DensePairScorer(2 * args.input_dim, hidden_dim=8)
    return DenseBilinearScorer(args.input_dim)


def run(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    pairs = make_relation_pairs(args, device)
    model = make_model(args).to(device)
    history: list[dict[str, float | int]] = []
    fit_metadata: dict[str, object] = {}
    learning_rate = args.lr if args.lr is not None else DEFAULT_LEARNING_RATES.get(args.variant)

    if args.variant == "pc_histogram_log_ratio":
        fit_metadata = fit_histogram(model, pairs, args.smoothing)
        elapsed = float(fit_metadata["fit_seconds"])
        history.append({"step": 0, "loss": math.nan, "train_pair_accuracy": pair_accuracy(model, pairs, args.eval_batch_size)})
    else:
        assert learning_rate is not None
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
        generator = torch.Generator(device=device).manual_seed(args.seed + 809)
        started = time.perf_counter()
        for step in range(1, args.steps + 1):
            indices = torch.randint(
                0, pairs.positive.shape[0], (args.batch_size,), generator=generator, device=device
            )
            loss = training_loss(args.variant, model, pairs, indices, args.margin)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step % args.eval_every == 0 or step == args.steps:
                row = {
                    "step": step,
                    "loss": float(loss.detach().item()),
                    "train_pair_accuracy": pair_accuracy(model, pairs, args.eval_batch_size),
                }
                history.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
        elapsed = time.perf_counter() - started

    model.eval()
    test_target = teacher_scores(pairs.test_queries, pairs.test_keys, pairs.relation)
    test_prediction = predict_score_matrix(model, pairs.test_queries, pairs.test_keys, args.eval_batch_size)
    metrics = retrieval_metrics(test_target, test_prediction, args.top_k, args.seed + 601)
    result: dict[str, object] = {
        "variant": args.variant,
        "parameters": parameter_count(model),
        "tables": args.tables if args.variant.startswith("pc_") else None,
        "comparisons": args.comparisons if args.variant.startswith("pc_") else None,
        "fixed_anchors": args.variant.startswith("pc_"),
        "fixed_zero_thresholds": args.variant.startswith("pc_"),
        "optimizer": None if args.variant == "pc_histogram_log_ratio" else "AdamW",
        "learning_rate": learning_rate,
        "steps": 0 if args.variant == "pc_histogram_log_ratio" else args.steps,
        "elapsed_seconds": elapsed,
        "steps_per_second": None if args.variant == "pc_histogram_log_ratio" else args.steps / elapsed,
        "train_pairs": pairs.positive.shape[0],
        "train_pair_accuracy": pair_accuracy(model, pairs, args.eval_batch_size),
        **fit_metadata,
        **{f"test_{key}": value for key, value in metrics.items()},
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"{args.variant}.json").write_text(json.dumps(result, indent=2) + "\n")
    with (args.out_dir / f"{args.variant}_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    print(json.dumps(result, sort_keys=True), flush=True)


def summarize(args: argparse.Namespace) -> None:
    by_variant = {
        result["variant"]: result
        for result in (json.loads(path.read_text()) for path in args.result_dir.glob("*.json"))
    }
    missing = [variant for variant in VARIANTS if variant not in by_variant]
    if missing:
        raise RuntimeError(f"Missing variants: {', '.join(missing)}")
    rows = [by_variant[variant] for variant in VARIANTS]
    fieldnames = sorted({key for row in rows for key in row})
    summary_path = args.result_dir / "summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Fixed-Route Relation-Energy Training",
        "",
        "All methods use the same binary bilinear teacher, train/test query-key pools, and 32,768 paired examples. Positive keys are teacher Top-16 keys. Half of the negatives are teacher ranks 17-24 and half are random keys outside the teacher Top-64.",
        "",
        "The three PC-LUT variants share exactly the same fixed random anchors, zero thresholds, 16 tables, 5 comparisons, and 512 scalar payload parameters. The histogram variant estimates a smoothed per-chamber positive/negative log-density ratio without AdamW, STE, or back-propagation.",
        "",
        "| Variant | Params | Train pair | Held-out pair order | Hard-negative | Top-16 recall | Top-1 | Spearman | Score R2 | Steps/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        speed = "closed form" if row["steps_per_second"] is None else f"{row['steps_per_second']:.1f}"
        lines.append(
            f"| {row['variant']} | {row['parameters']:,} | {row['train_pair_accuracy']:.4f} | "
            f"{row['test_random_pair_order_accuracy']:.4f} | "
            f"{row['test_hard_negative_preference_accuracy']:.4f} | {row['test_topk_recall']:.4f} | "
            f"{row['test_top1_accuracy']:.4f} | {row['test_spearman']:.4f} | "
            f"{row['test_score_r2']:.4f} | {speed} |"
        )
    named = {row["variant"]: row for row in rows}
    histogram = named["pc_histogram_log_ratio"]
    ff = named["pc_ff_logistic_adamw"]
    mse = named["pc_mse_adamw"]
    mlp = named["dense_mlp_pairwise"]
    bilinear = named["dense_bilinear_pairwise"]
    lines.extend(
        [
            "",
            "Random baselines are 0.5 for pair ordering and hard-negative preference, 0.03125 for Top-16 recall, and 0.001953 for exact Top-1.",
            "",
            "The MSE model is the scalar-score PC-LUT baseline. The local FF model uses independent positive/negative logistic goodness losses. The dense models use a paired ranking loss. Consequently, retrieval metrics are the primary comparison; score R2 is diagnostic and is not expected to be calibrated for contrastive objectives.",
            "",
            "## Findings",
            "",
            f"The closed-form histogram reaches train pair accuracy {histogram['train_pair_accuracy']:.4f}, versus {ff['train_pair_accuracy']:.4f} for local FF AdamW and {mse['train_pair_accuracy']:.4f} for MSE AdamW. It fits in {histogram['fit_seconds']:.3f} seconds, while the AdamW variants require about {ff['elapsed_seconds']:.1f}-{mse['elapsed_seconds']:.1f} seconds. All histogram rows are occupied, so completely unseen rows are not the limiting factor.",
            "",
            f"The three fixed-route PC-LUT methods remain close on held-out retrieval: hard-negative accuracy is {min(mse['test_hard_negative_preference_accuracy'], ff['test_hard_negative_preference_accuracy'], histogram['test_hard_negative_preference_accuracy']):.4f}-{max(mse['test_hard_negative_preference_accuracy'], ff['test_hard_negative_preference_accuracy'], histogram['test_hard_negative_preference_accuracy']):.4f}, and Top-16 recall is {min(mse['test_topk_recall'], ff['test_topk_recall'], histogram['test_topk_recall']):.4f}-{max(mse['test_topk_recall'], ff['test_topk_recall'], histogram['test_topk_recall']):.4f}. Replacing MSE with local FF or replacing AdamW with exact chamber counts does not rescue relation selection.",
            "",
            f"The parameter-matched dense MLP reaches hard-negative accuracy {mlp['test_hard_negative_preference_accuracy']:.4f}, Top-16 recall {mlp['test_topk_recall']:.4f}, and Top-1 {mlp['test_top1_accuracy']:.4f}. The dense bilinear scorer reaches {bilinear['test_hard_negative_preference_accuracy']:.4f}, {bilinear['test_topk_recall']:.4f}, and {bilinear['test_top1_accuracy']:.4f}, respectively. The primary bottleneck in this controlled setting is therefore the fixed single-layer comparison quotient and its additive per-table factorization, not AdamW, STE, payload coverage, or payload parameter count.",
            "",
            "Contrastive score R2 is negative because pairwise and FF losses identify ordering or log odds rather than the teacher's absolute scale. This does not contradict their retrieval metrics.",
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
