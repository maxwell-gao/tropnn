from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch
from torch import Tensor, nn

from tropnn.tools.bilinear_retrieval_probe import predict_score_matrix, retrieval_metrics, teacher_scores
from tropnn.tools.fixed_l16_route_decoder_probe import FixedL16RouteBits
from tropnn.tools.fixed_route_relation_energy_probe import make_relation_pairs


GROUP_SIZES = (1, 2, 3, 5, 8)


class CrossTableLUT(nn.Module):
    def __init__(self, input_bits: int, group_size: int, comparisons: int, seed: int) -> None:
        super().__init__()
        if input_bits % comparisons:
            raise ValueError("input_bits must contain complete first-stage route codes")
        first_stage_tables = input_bits // comparisons
        generator = torch.Generator(device="cpu").manual_seed(seed)
        ordered_bits = []
        for bit in range(comparisons):
            table_order = torch.randperm(first_stage_tables, generator=generator)
            ordered_bits.append(table_order * comparisons + bit)
        feature_order = torch.cat(ordered_bits)
        groups = math.ceil(input_bits / group_size)
        padding = groups * group_size - input_bits
        if padding:
            feature_order = torch.cat([feature_order, feature_order[:padding]])
        self.register_buffer("feature_groups", feature_order.view(groups, group_size))
        self.register_buffer("powers", 2 ** torch.arange(group_size, dtype=torch.long))
        self.payload = nn.Parameter(torch.zeros(groups, 1 << group_size))

    def forward(self, route_bits: Tensor) -> Tensor:
        selected = route_bits[:, self.feature_groups.reshape(-1)].view(
            route_bits.shape[0], *self.feature_groups.shape
        )
        codes = ((selected > 0).to(torch.long) * self.powers).sum(dim=-1)
        groups = torch.arange(self.payload.shape[0], device=route_bits.device).view(1, -1)
        return self.payload[groups, codes].sum(dim=-1, keepdim=True)


class RoutedInteractionModel(nn.Module):
    def __init__(self, encoder: FixedL16RouteBits, decoder: CrossTableLUT) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, pair: Tensor) -> Tensor:
        return self.decoder(self.encoder(pair))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cross-table route-bit interaction-order sweep.")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run")
    run.add_argument("--group-size", choices=GROUP_SIZES, type=int, required=True)
    run.add_argument("--input-dim", type=int, default=32)
    run.add_argument("--train-queries", type=int, default=2048)
    run.add_argument("--train-keys", type=int, default=2048)
    run.add_argument("--test-queries", type=int, default=256)
    run.add_argument("--test-keys", type=int, default=512)
    run.add_argument("--max-value", type=int, default=15)
    run.add_argument("--tables", type=int, default=16)
    run.add_argument("--comparisons", type=int, default=5)
    run.add_argument("--depth", type=int, default=16)
    run.add_argument("--positive-per-query", type=int, default=16)
    run.add_argument("--hard-negative-per-query", type=int, default=8)
    run.add_argument("--steps", type=int, default=10000)
    run.add_argument("--batch-size", type=int, default=512)
    run.add_argument("--eval-batch-size", type=int, default=4096)
    run.add_argument("--eval-every", type=int, default=1000)
    run.add_argument("--top-k", type=int, default=16)
    run.add_argument("--lr", type=float, default=0.005)
    run.add_argument("--device", default="cuda")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--out-dir", type=Path, required=True)

    summarize = commands.add_parser("summarize")
    summarize.add_argument("--result-dir", type=Path, required=True)
    summarize.add_argument("--additive-summary", type=Path, required=True)
    summarize.add_argument("--dense-route-summary", type=Path, required=True)
    summarize.add_argument("--out-report", type=Path, required=True)
    return parser


@torch.no_grad()
def score_features(decoder: nn.Module, features: Tensor, batch_size: int) -> Tensor:
    return torch.cat(
        [decoder(features[start : start + batch_size]) for start in range(0, features.shape[0], batch_size)]
    )


@torch.no_grad()
def pair_accuracy(decoder: nn.Module, positive: Tensor, negative: Tensor, batch_size: int) -> float:
    positive_score = score_features(decoder, positive, batch_size)
    negative_score = score_features(decoder, negative, batch_size)
    return float((positive_score > negative_score).to(torch.float64).mean().item())


def run(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    pairs = make_relation_pairs(args, device)
    encoder = FixedL16RouteBits(
        input_dim=2 * args.input_dim,
        tables=args.tables,
        comparisons=args.comparisons,
        depth=args.depth,
        seed=args.seed,
    ).to(device)
    decoder = CrossTableLUT(encoder.output_dim, args.group_size, args.comparisons, args.seed + 1601).to(device)

    feature_started = time.perf_counter()
    positive_features = encoder(pairs.positive)
    negative_features = encoder(pairs.negative)
    feature_seconds = time.perf_counter() - feature_started

    optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=0.0)
    generator = torch.Generator(device=device).manual_seed(args.seed + 809)
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        indices = torch.randint(
            0,
            positive_features.shape[0],
            (args.batch_size,),
            generator=generator,
            device=device,
        )
        positive_score = decoder(positive_features[indices])
        negative_score = decoder(negative_features[indices])
        loss = 0.5 * (
            (positive_score - pairs.positive_target[indices]).square().mean()
            + (negative_score - pairs.negative_target[indices]).square().mean()
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % args.eval_every == 0 or step == args.steps:
            row = {
                "step": step,
                "loss": float(loss.detach().item()),
                "train_pair_accuracy": pair_accuracy(
                    decoder, positive_features, negative_features, args.eval_batch_size
                ),
            }
            history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    elapsed = time.perf_counter() - started

    model = RoutedInteractionModel(encoder, decoder).eval()
    test_target = teacher_scores(pairs.test_queries, pairs.test_keys, pairs.relation)
    test_prediction = predict_score_matrix(model, pairs.test_queries, pairs.test_keys, args.eval_batch_size)
    metrics = retrieval_metrics(test_target, test_prediction, args.top_k, args.seed + 601)
    result: dict[str, object] = {
        "group_size": args.group_size,
        "groups": decoder.payload.shape[0],
        "parameters": decoder.payload.numel(),
        "route_bit_dim": encoder.output_dim,
        "feature_seconds": feature_seconds,
        "steps": args.steps,
        "elapsed_seconds": elapsed,
        "steps_per_second": args.steps / elapsed,
        "train_pair_accuracy": pair_accuracy(
            decoder, positive_features, negative_features, args.eval_batch_size
        ),
        **{f"test_{key}": value for key, value in metrics.items()},
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"cross_table_g{args.group_size}"
    (args.out_dir / f"{stem}.json").write_text(json.dumps(result, indent=2) + "\n")
    with (args.out_dir / f"{stem}_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    print(json.dumps(result, sort_keys=True), flush=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def summarize(args: argparse.Namespace) -> None:
    results = [json.loads(path.read_text()) for path in sorted(args.result_dir.glob("*.json"))]
    by_group = {row["group_size"]: row for row in results}
    missing = [group for group in GROUP_SIZES if group not in by_group]
    if missing:
        raise RuntimeError(f"Missing group sizes: {missing}")
    rows = [by_group[group] for group in GROUP_SIZES]
    summary_path = args.result_dir / "summary.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    additive_rows = read_csv(args.additive_summary)
    additive = next(
        row for row in additive_rows if row["variant"] == "pc_mse_adamw" and int(row["depth"]) == 16
    )
    dense_rows = read_csv(args.dense_route_summary)
    dense = next(row for row in dense_rows if row["objective"] == "mse")
    lines = [
        "# Cross-Table Route-Bit Interaction Sweep",
        "",
        "The fixed L16 route encoder is unchanged. Its 1,280 bits are permuted and partitioned into second-stage LUT addresses containing bits from different first-stage tables. Every route bit is used once before minimal padding. Only second-stage scalar payloads are trained with score MSE.",
        "",
        "| Decoder | Interaction g | Groups | Params | Train pair | Held-out pair | Hard-neg | Top-16 | Top-1 | Spearman | Steps/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| Original additive tables | within-table 5 | 256 | {int(additive['parameters']):,} | {float(additive['train_pair_accuracy']):.4f} | {float(additive['test_random_pair_order_accuracy']):.4f} | {float(additive['test_hard_negative_preference_accuracy']):.4f} | {float(additive['test_topk_recall']):.4f} | {float(additive['test_top1_accuracy']):.4f} | {float(additive['test_spearman']):.4f} | {float(additive['steps_per_second']):.1f} |",
    ]
    for row in rows:
        lines.append(
            f"| Cross-table LUT | {row['group_size']} | {row['groups']} | {row['parameters']:,} | "
            f"{row['train_pair_accuracy']:.4f} | {row['test_random_pair_order_accuracy']:.4f} | "
            f"{row['test_hard_negative_preference_accuracy']:.4f} | {row['test_topk_recall']:.4f} | "
            f"{row['test_top1_accuracy']:.4f} | {row['test_spearman']:.4f} | {row['steps_per_second']:.1f} |"
        )
    lines.extend(
        [
            f"| Dense H256 oracle | all bits | 256 hidden | {int(dense['trainable_parameters']):,} | {float(dense['train_pair_accuracy']):.4f} | {float(dense['test_random_pair_order_accuracy']):.4f} | {float(dense['test_hard_negative_preference_accuracy']):.4f} | {float(dense['test_topk_recall']):.4f} | {float(dense['test_top1_accuracy']):.4f} | {float(dense['test_spearman']):.4f} | {float(dense['steps_per_second']):.1f} |",
            "",
            "This experiment isolates interaction structure from routing: all decoders receive exactly the same frozen route bits. Improvement from cross-table LUTs would show that sparse joint lookups can recover relation information without GEMM.",
            "",
            "## Findings",
            "",
            f"Unary decoding is ineffective: g=1 yields Top-16 {by_group[1]['test_topk_recall']:.4f} and Spearman {by_group[1]['test_spearman']:.4f}. Relation information first becomes usable through interactions, with g=2 and g=3 reaching Top-16 {by_group[2]['test_topk_recall']:.4f} and {by_group[3]['test_topk_recall']:.4f}.",
            "",
            f"At equal 8,192-parameter budgets, cross-table g=5 changes Top-16 from {float(additive['test_topk_recall']):.4f} to {by_group[5]['test_topk_recall']:.4f}, but lowers Top-1 from {float(additive['test_top1_accuracy']):.4f} to {by_group[5]['test_top1_accuracy']:.4f}. Random cross-table regrouping is therefore only a modest redistribution of capacity, not a recovery of the dense-decoder gain.",
            "",
            f"Increasing to g=8 raises training pair accuracy to {by_group[8]['train_pair_accuracy']:.4f} while held-out Top-16 falls to {by_group[8]['test_topk_recall']:.4f}. The exponential table fragments data into sparse joint cells and overfits despite using {by_group[8]['parameters']:,} parameters.",
            "",
            "The dense oracle reuses every route bit across many learned hidden features. Each decoder in this sweep assigns every bit to only one disjoint group. The remaining hypothesis is therefore overlapping multi-hash grouping: reuse each bit in several independently formed low-order LUT interactions without introducing GEMM.",
            "",
            "Reported steps/s for cross-table decoders excludes one-time frozen route-bit extraction and is not directly comparable to the original additive model's end-to-end training throughput.",
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
