from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from torch import Tensor, nn

from tropnn.tools.bilinear_retrieval_probe import predict_score_matrix, retrieval_metrics, teacher_scores
from tropnn.tools.fixed_route_relation_energy_probe import (
    DEFAULT_LEARNING_RATES,
    VARIANTS,
    DenseBilinearScorer,
    DensePairScorer,
    FixedPairwiseEnergy,
    fit_histogram,
    make_relation_pairs,
    pair_accuracy,
    parameter_count,
    training_loss,
)


DEPTHS = (1, 2, 4, 8, 16)


class AdditiveEnergy(nn.Module):
    def __init__(self, blocks: list[nn.Module]) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, pair: Tensor) -> Tensor:
        score = self.blocks[0](pair)
        for block in self.blocks[1:]:
            score = score + block(pair)
        return score


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Additive fixed-route relation-energy depth sweep.")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run")
    run.add_argument("--variant", choices=VARIANTS, required=True)
    run.add_argument("--depth", choices=DEPTHS, type=int, required=True)
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


def make_additive_model(args: argparse.Namespace) -> AdditiveEnergy:
    blocks: list[nn.Module] = []
    for layer in range(args.depth):
        if args.variant.startswith("pc_"):
            blocks.append(
                FixedPairwiseEnergy(
                    2 * args.input_dim,
                    args.tables,
                    args.comparisons,
                    args.seed + 1009 * layer,
                )
            )
        elif args.variant == "dense_mlp_pairwise":
            blocks.append(DensePairScorer(2 * args.input_dim, hidden_dim=8))
        else:
            blocks.append(DenseBilinearScorer(args.input_dim))
    return AdditiveEnergy(blocks)


def effective_learning_rate(args: argparse.Namespace) -> float | None:
    if args.variant == "pc_histogram_log_ratio":
        return None
    learning_rate = args.lr if args.lr is not None else DEFAULT_LEARNING_RATES[args.variant]
    if args.variant == "dense_bilinear_pairwise":
        learning_rate /= args.depth
    return learning_rate


@torch.no_grad()
def fit_additive_histogram(
    model: AdditiveEnergy,
    pairs: object,
    smoothing: float,
) -> dict[str, float]:
    started = time.perf_counter()
    occupied = []
    for block in model.blocks:
        assert isinstance(block, FixedPairwiseEnergy)
        metadata = fit_histogram(block, pairs, smoothing)
        occupied.append(float(metadata["mean_occupied_row_fraction"]))
    return {
        "fit_seconds": time.perf_counter() - started,
        "mean_occupied_row_fraction": sum(occupied) / len(occupied),
    }


def run(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    pairs = make_relation_pairs(args, device)
    model = make_additive_model(args).to(device)
    learning_rate = effective_learning_rate(args)
    history: list[dict[str, float | int]] = []
    fit_metadata: dict[str, float] = {}

    if args.variant == "pc_histogram_log_ratio":
        fit_metadata = fit_additive_histogram(model, pairs, args.smoothing)
        elapsed = fit_metadata["fit_seconds"]
        history.append(
            {
                "step": 0,
                "loss": float("nan"),
                "train_pair_accuracy": pair_accuracy(model, pairs, args.eval_batch_size),
            }
        )
    else:
        assert learning_rate is not None
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
        generator = torch.Generator(device=device).manual_seed(args.seed + 809)
        started = time.perf_counter()
        for step in range(1, args.steps + 1):
            indices = torch.randint(
                0,
                pairs.positive.shape[0],
                (args.batch_size,),
                generator=generator,
                device=device,
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
        "depth": args.depth,
        "parameters": parameter_count(model),
        "tables_per_layer": args.tables if args.variant.startswith("pc_") else None,
        "comparisons": args.comparisons if args.variant.startswith("pc_") else None,
        "fixed_zero_thresholds": args.variant.startswith("pc_"),
        "optimizer": None if args.variant == "pc_histogram_log_ratio" else "AdamW",
        "learning_rate": learning_rate,
        "steps": 0 if args.variant == "pc_histogram_log_ratio" else args.steps,
        "elapsed_seconds": elapsed,
        "steps_per_second": None if args.variant == "pc_histogram_log_ratio" else args.steps / elapsed,
        "train_pair_accuracy": pair_accuracy(model, pairs, args.eval_batch_size),
        **fit_metadata,
        **{f"test_{key}": value for key, value in metrics.items()},
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.variant}_L{args.depth}"
    (args.out_dir / f"{stem}.json").write_text(json.dumps(result, indent=2) + "\n")
    with (args.out_dir / f"{stem}_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    print(json.dumps(result, sort_keys=True), flush=True)


def summarize(args: argparse.Namespace) -> None:
    results = [json.loads(path.read_text()) for path in args.result_dir.glob("*.json")]
    by_key = {(row["variant"], row["depth"]): row for row in results}
    missing = [(variant, depth) for variant in VARIANTS for depth in DEPTHS if (variant, depth) not in by_key]
    if missing:
        raise RuntimeError(f"Missing runs: {missing}")
    rows = [by_key[(variant, depth)] for variant in VARIANTS for depth in DEPTHS]
    fieldnames = sorted({key for row in rows for key in row})
    summary_path = args.result_dir / "summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Additive Relation-Energy Depth Sweep",
        "",
        "Each additive layer introduces an independently seeded fixed route. PC-LUT layers use 16 tables, 5 comparisons, zero thresholds, and 512 scalar payload parameters per layer. Layers do not exchange hidden states: their scalar energies are summed directly.",
        "",
        "| Variant | L | Params | Train pair | Held-out pair | Hard-neg | Top-16 | Top-1 | Spearman | Steps/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        speed = "closed form" if row["steps_per_second"] is None else f"{row['steps_per_second']:.1f}"
        lines.append(
            f"| {row['variant']} | {row['depth']} | {row['parameters']:,} | "
            f"{row['train_pair_accuracy']:.4f} | {row['test_random_pair_order_accuracy']:.4f} | "
            f"{row['test_hard_negative_preference_accuracy']:.4f} | {row['test_topk_recall']:.4f} | "
            f"{row['test_top1_accuracy']:.4f} | {row['test_spearman']:.4f} | {speed} |"
        )

    lines.extend(["", "## Scaling deltas", ""])
    for variant in VARIANTS:
        first = by_key[(variant, 1)]
        last = by_key[(variant, 16)]
        lines.append(
            f"- `{variant}` L1->L16: held-out pair {first['test_random_pair_order_accuracy']:.4f}->{last['test_random_pair_order_accuracy']:.4f}, "
            f"hard-negative {first['test_hard_negative_preference_accuracy']:.4f}->{last['test_hard_negative_preference_accuracy']:.4f}, "
            f"Top-16 {first['test_topk_recall']:.4f}->{last['test_topk_recall']:.4f}, "
            f"Top-1 {first['test_top1_accuracy']:.4f}->{last['test_top1_accuracy']:.4f}."
        )
    lines.extend(
        [
            "",
            "Random baselines are 0.5 for pair ordering and hard-negative preference, 0.03125 for Top-16 recall, and 0.001953 for exact Top-1.",
            "",
            "This sweep tests accumulation of independent comparison partitions. It does not test recursive geometry because every layer routes the original pair rather than the previous layer's representation.",
            "",
            "## Findings",
            "",
            "All PC-LUT objectives improve monotonically in the main relation metrics as independent partitions are added. The single-route result was therefore not an upper bound on comparison information: multiple independently anchored quotients recover complementary relation evidence.",
            "",
            f"MSE is the strongest PC-LUT objective at L16: held-out pair order {by_key[('pc_mse_adamw', 16)]['test_random_pair_order_accuracy']:.4f}, hard-negative {by_key[('pc_mse_adamw', 16)]['test_hard_negative_preference_accuracy']:.4f}, Top-16 {by_key[('pc_mse_adamw', 16)]['test_topk_recall']:.4f}, and Top-1 {by_key[('pc_mse_adamw', 16)]['test_top1_accuracy']:.4f}. Local FF reaches higher training pair accuracy but weaker held-out ordering, indicating that binary positive/negative goodness discards useful within-group score structure and overfits the sampled relation boundary more readily than score regression.",
            "",
            "Histogram log-density ratios track local FF closely without gradient optimization, confirming that AdamW is not the main limitation of the logistic goodness model. Their gap from MSE is primarily an objective-information gap, not an optimizer gap.",
            "",
            f"At nearly equal L16 parameter counts, PC-MSE uses {by_key[('pc_mse_adamw', 16)]['parameters']:,} parameters versus {by_key[('dense_mlp_pairwise', 16)]['parameters']:,} for the dense MLP, but reaches Top-16 {by_key[('pc_mse_adamw', 16)]['test_topk_recall']:.4f} versus {by_key[('dense_mlp_pairwise', 16)]['test_topk_recall']:.4f}. PC-MSE L16 is instead only roughly competitive with the 529-parameter dense MLP at L1, quantifying a roughly 15.5x parameter-efficiency deficit on this relation task.",
            "",
            "Because all additive layers route the same input, `L layers x T16` is algebraically identical to one layer with `T = 16L`. The positive scaling result is table-width scaling, not recursive depth scaling. A true depth experiment must let the next route depend on the previous LUT output.",
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
