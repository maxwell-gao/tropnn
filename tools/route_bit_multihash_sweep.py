from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from torch import Tensor, nn

from tropnn.tools.bilinear_retrieval_probe import predict_score_matrix, retrieval_metrics, teacher_scores
from tropnn.tools.fixed_l16_route_decoder_probe import FixedL16RouteBits
from tropnn.tools.fixed_route_relation_energy_probe import make_relation_pairs
from tropnn.tools.route_bit_interaction_sweep import CrossTableLUT


GROUP_SIZES = (3, 5)
HASH_COUNTS = (2, 4, 8)


class MultiHashCrossTableLUT(nn.Module):
    def __init__(self, input_bits: int, group_size: int, comparisons: int, hashes: int, seed: int) -> None:
        super().__init__()
        self.decoders = nn.ModuleList(
            CrossTableLUT(input_bits, group_size, comparisons, seed + 7919 * hash_index)
            for hash_index in range(hashes)
        )

    def forward(self, route_bits: Tensor) -> Tensor:
        score = self.decoders[0](route_bits)
        for decoder in self.decoders[1:]:
            score = score + decoder(route_bits)
        return score


class RoutedMultiHashModel(nn.Module):
    def __init__(self, encoder: FixedL16RouteBits, decoder: MultiHashCrossTableLUT) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, pair: Tensor) -> Tensor:
        return self.decoder(self.encoder(pair))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Overlapping multi-hash route-bit LUT sweep.")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run")
    run.add_argument("--group-size", choices=GROUP_SIZES, type=int, required=True)
    run.add_argument("--hashes", choices=HASH_COUNTS, type=int, required=True)
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
    summarize.add_argument("--interaction-summary", type=Path, required=True)
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
    return float(
        (score_features(decoder, positive, batch_size) > score_features(decoder, negative, batch_size))
        .to(torch.float64)
        .mean()
        .item()
    )


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
    decoder = MultiHashCrossTableLUT(
        encoder.output_dim,
        args.group_size,
        args.comparisons,
        args.hashes,
        args.seed + 1601,
    ).to(device)

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

    model = RoutedMultiHashModel(encoder, decoder).eval()
    test_target = teacher_scores(pairs.test_queries, pairs.test_keys, pairs.relation)
    test_prediction = predict_score_matrix(model, pairs.test_queries, pairs.test_keys, args.eval_batch_size)
    metrics = retrieval_metrics(test_target, test_prediction, args.top_k, args.seed + 601)
    parameters = sum(parameter.numel() for parameter in decoder.parameters())
    result: dict[str, object] = {
        "group_size": args.group_size,
        "hashes": args.hashes,
        "parameters": parameters,
        "route_bit_uses": args.hashes,
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
    stem = f"cross_table_g{args.group_size}_h{args.hashes}"
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
    by_key = {(row["group_size"], row["hashes"]): row for row in results}
    missing = [(group, hashes) for group in GROUP_SIZES for hashes in HASH_COUNTS if (group, hashes) not in by_key]
    if missing:
        raise RuntimeError(f"Missing runs: {missing}")
    rows = [by_key[(group, hashes)] for group in GROUP_SIZES for hashes in HASH_COUNTS]
    summary_path = args.result_dir / "summary.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    interaction_rows = read_csv(args.interaction_summary)
    single = {int(row["group_size"]): row for row in interaction_rows if int(row["group_size"]) in GROUP_SIZES}
    dense_rows = read_csv(args.dense_route_summary)
    dense = next(row for row in dense_rows if row["objective"] == "mse")
    lines = [
        "# Overlapping Multi-Hash Route-Bit LUTs",
        "",
        "Every hash independently repartitions all 1,280 frozen route bits into cross-table LUT addresses. Increasing the hash count reuses each bit with new partners while preserving lookup-and-accumulate inference.",
        "",
        "| g | Hashes | Params | Train pair | Held-out pair | Hard-neg | Top-16 | Top-1 | Spearman | Steps/s |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in GROUP_SIZES:
        baseline = single[group]
        lines.append(
            f"| {group} | 1 | {int(baseline['parameters']):,} | {float(baseline['train_pair_accuracy']):.4f} | "
            f"{float(baseline['test_random_pair_order_accuracy']):.4f} | {float(baseline['test_hard_negative_preference_accuracy']):.4f} | "
            f"{float(baseline['test_topk_recall']):.4f} | {float(baseline['test_top1_accuracy']):.4f} | "
            f"{float(baseline['test_spearman']):.4f} | {float(baseline['steps_per_second']):.1f} |"
        )
        for hashes in HASH_COUNTS:
            row = by_key[(group, hashes)]
            lines.append(
                f"| {group} | {hashes} | {row['parameters']:,} | {row['train_pair_accuracy']:.4f} | "
                f"{row['test_random_pair_order_accuracy']:.4f} | {row['test_hard_negative_preference_accuracy']:.4f} | "
                f"{row['test_topk_recall']:.4f} | {row['test_top1_accuracy']:.4f} | "
                f"{row['test_spearman']:.4f} | {row['steps_per_second']:.1f} |"
            )
    lines.extend(
        [
            f"| dense | H256 | {int(dense['trainable_parameters']):,} | {float(dense['train_pair_accuracy']):.4f} | {float(dense['test_random_pair_order_accuracy']):.4f} | {float(dense['test_hard_negative_preference_accuracy']):.4f} | {float(dense['test_topk_recall']):.4f} | {float(dense['test_top1_accuracy']):.4f} | {float(dense['test_spearman']):.4f} | {float(dense['steps_per_second']):.1f} |",
            "",
            "This is a decoder-only speed measurement after frozen route-bit extraction. It does not include first-stage comparison cost.",
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
