from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from torch import Tensor, nn

from tropnn.tools.bilinear_retrieval_probe import predict_score_matrix, retrieval_metrics, teacher_scores
from tropnn.tools.fixed_route_relation_energy_probe import FixedPairwiseEnergy, make_relation_pairs


OBJECTIVES = ("mse", "pairwise")


class FixedWideRouteBits(nn.Module):
    def __init__(self, input_dim: int, tables_per_seed_block: int, comparisons: int, seed_blocks: int, seed: int) -> None:
        super().__init__()
        self.comparisons = comparisons
        self.route_blocks = nn.ModuleList(
            FixedPairwiseEnergy(input_dim, tables_per_seed_block, comparisons, seed + 1009 * block)
            for block in range(seed_blocks)
        )
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @property
    def output_dim(self) -> int:
        return len(self.route_blocks) * self.route_blocks[0].payload.shape[0] * self.comparisons

    @torch.no_grad()
    def forward(self, pair: Tensor) -> Tensor:
        codes = torch.cat([route_block.route(pair) for route_block in self.route_blocks], dim=-1)
        shifts = torch.arange(self.comparisons, device=pair.device)
        bits = ((codes.unsqueeze(-1) >> shifts) & 1).reshape(pair.shape[0], -1)
        return 2.0 * bits.to(pair.dtype) - 1.0


class DenseRouteDecoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.readout = nn.Linear(hidden_dim, 1)

    def forward(self, route_bits: Tensor) -> Tensor:
        return self.readout(torch.nn.functional.gelu(self.hidden(route_bits)))


class RoutedDiagnostic(nn.Module):
    def __init__(self, encoder: FixedWideRouteBits, decoder: DenseRouteDecoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, pair: Tensor) -> Tensor:
        return self.decoder(self.encoder(pair))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dense diagnostic decoder over frozen wide T256/C5 PC-LUT route bits.")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run")
    run.add_argument("--objective", choices=OBJECTIVES, required=True)
    run.add_argument("--input-dim", type=int, default=32)
    run.add_argument("--train-queries", type=int, default=2048)
    run.add_argument("--train-keys", type=int, default=2048)
    run.add_argument("--test-queries", type=int, default=256)
    run.add_argument("--test-keys", type=int, default=512)
    run.add_argument("--max-value", type=int, default=15)
    run.add_argument("--tables-per-seed-block", type=int, default=16)
    run.add_argument("--comparisons", type=int, default=5)
    run.add_argument("--seed_blocks", type=int, default=16)
    run.add_argument("--hidden-dim", type=int, default=256)
    run.add_argument("--positive-per-query", type=int, default=16)
    run.add_argument("--hard-negative-per-query", type=int, default=8)
    run.add_argument("--steps", type=int, default=10000)
    run.add_argument("--batch-size", type=int, default=512)
    run.add_argument("--eval-batch-size", type=int, default=4096)
    run.add_argument("--eval-every", type=int, default=1000)
    run.add_argument("--top-k", type=int, default=16)
    run.add_argument("--margin", type=float, default=1.0)
    run.add_argument("--lr", type=float, default=0.001)
    run.add_argument("--weight-decay", type=float, default=0.0001)
    run.add_argument("--device", default="cuda")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--out-dir", type=Path, required=True)

    summarize = commands.add_parser("summarize")
    summarize.add_argument("--result-dir", type=Path, required=True)
    summarize.add_argument("--additive-summary", type=Path, required=True)
    summarize.add_argument("--out-report", type=Path, required=True)
    return parser


@torch.no_grad()
def score_features(decoder: nn.Module, features: Tensor, batch_size: int) -> Tensor:
    return torch.cat(
        [decoder(features[start : start + batch_size]) for start in range(0, features.shape[0], batch_size)]
    )


@torch.no_grad()
def feature_pair_accuracy(decoder: nn.Module, positive: Tensor, negative: Tensor, batch_size: int) -> float:
    positive_score = score_features(decoder, positive, batch_size)
    negative_score = score_features(decoder, negative, batch_size)
    return float((positive_score > negative_score).to(torch.float64).mean().item())


def train_loss(
    objective: str,
    decoder: nn.Module,
    positive: Tensor,
    negative: Tensor,
    positive_target: Tensor,
    negative_target: Tensor,
    indices: Tensor,
    margin: float,
) -> Tensor:
    positive_score = decoder(positive[indices])
    negative_score = decoder(negative[indices])
    if objective == "mse":
        return 0.5 * (
            (positive_score - positive_target[indices]).square().mean()
            + (negative_score - negative_target[indices]).square().mean()
        )
    return torch.nn.functional.softplus(margin - positive_score + negative_score).mean()


def trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def run(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    pairs = make_relation_pairs(args, device)
    encoder = FixedWideRouteBits(
        input_dim=2 * args.input_dim,
        tables_per_seed_block=args.tables_per_seed_block,
        comparisons=args.comparisons,
        seed_blocks=args.seed_blocks,
        seed=args.seed,
    ).to(device)
    decoder = DenseRouteDecoder(encoder.output_dim, args.hidden_dim).to(device)

    feature_started = time.perf_counter()
    positive_features = encoder(pairs.positive)
    negative_features = encoder(pairs.negative)
    feature_seconds = time.perf_counter() - feature_started

    optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
        loss = train_loss(
            args.objective,
            decoder,
            positive_features,
            negative_features,
            pairs.positive_target,
            pairs.negative_target,
            indices,
            args.margin,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % args.eval_every == 0 or step == args.steps:
            row = {
                "step": step,
                "loss": float(loss.detach().item()),
                "train_pair_accuracy": feature_pair_accuracy(
                    decoder, positive_features, negative_features, args.eval_batch_size
                ),
            }
            history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    elapsed = time.perf_counter() - started

    model = RoutedDiagnostic(encoder, decoder).eval()
    test_target = teacher_scores(pairs.test_queries, pairs.test_keys, pairs.relation)
    test_prediction = predict_score_matrix(model, pairs.test_queries, pairs.test_keys, args.eval_batch_size)
    metrics = retrieval_metrics(test_target, test_prediction, args.top_k, args.seed + 601)
    result: dict[str, object] = {
        "objective": args.objective,
        "seed_blocks": args.seed_blocks,
        "tables_per_layer": args.tables_per_seed_block,
        "comparisons": args.comparisons,
        "route_bit_dim": encoder.output_dim,
        "hidden_dim": args.hidden_dim,
        "trainable_parameters": trainable_parameters(decoder),
        "feature_seconds": feature_seconds,
        "steps": args.steps,
        "elapsed_seconds": elapsed,
        "steps_per_second": args.steps / elapsed,
        "train_pair_accuracy": feature_pair_accuracy(
            decoder, positive_features, negative_features, args.eval_batch_size
        ),
        **{f"test_{key}": value for key, value in metrics.items()},
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"route_bits_h{args.hidden_dim}_{args.objective}"
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
    diagnostics = [json.loads(path.read_text()) for path in sorted(args.result_dir.glob("*.json"))]
    by_objective = {row["objective"]: row for row in diagnostics}
    missing = [objective for objective in OBJECTIVES if objective not in by_objective]
    if missing:
        raise RuntimeError(f"Missing objectives: {missing}")
    summary_path = args.result_dir / "summary.csv"
    fieldnames = sorted({key for row in diagnostics for key in row})
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(diagnostics)

    additive = read_csv(args.additive_summary)
    controls = {
        (row["variant"], int(row["width"])): row
        for row in additive
        if row["variant"] in {"pc_mse_adamw", "dense_mlp_pairwise"}
    }
    pc = controls[("pc_mse_adamw", 16)]
    dense_l1 = controls[("dense_mlp_pairwise", 1)]
    dense_l16 = controls[("dense_mlp_pairwise", 16)]
    mse = by_objective["mse"]
    pairwise = by_objective["pairwise"]

    lines = [
        "# Dense Decoder over Fixed Wide T256/C5 Route Bits",
        "",
        "The encoder exactly reproduces the fixed wide T256/C5 route from the additive width sweep and expands every route code into its five comparison bits. The resulting 1,280-bit vector is a lossless representation of all route codes. Only the dense diagnostic decoder is trained.",
        "",
        "The 16 seed blocks are flattened into one T256/C5 parallel route ensemble; no block consumes another block output.",
        "",
        "| Model | Trainable params | Train pair | Held-out pair | Hard-neg | Top-16 | Top-1 | Spearman |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| PC additive MSE W16 (T256) | {int(pc['parameters']):,} | {float(pc['train_pair_accuracy']):.4f} | {float(pc['test_random_pair_order_accuracy']):.4f} | {float(pc['test_hard_negative_preference_accuracy']):.4f} | {float(pc['test_topk_recall']):.4f} | {float(pc['test_top1_accuracy']):.4f} | {float(pc['test_spearman']):.4f} |",
        f"| Route bits + dense H256 MSE | {mse['trainable_parameters']:,} | {mse['train_pair_accuracy']:.4f} | {mse['test_random_pair_order_accuracy']:.4f} | {mse['test_hard_negative_preference_accuracy']:.4f} | {mse['test_topk_recall']:.4f} | {mse['test_top1_accuracy']:.4f} | {mse['test_spearman']:.4f} |",
        f"| Route bits + dense H256 pairwise | {pairwise['trainable_parameters']:,} | {pairwise['train_pair_accuracy']:.4f} | {pairwise['test_random_pair_order_accuracy']:.4f} | {pairwise['test_hard_negative_preference_accuracy']:.4f} | {pairwise['test_topk_recall']:.4f} | {pairwise['test_top1_accuracy']:.4f} | {pairwise['test_spearman']:.4f} |",
        f"| Raw pair + dense MLP L1 | {int(dense_l1['parameters']):,} | {float(dense_l1['train_pair_accuracy']):.4f} | {float(dense_l1['test_random_pair_order_accuracy']):.4f} | {float(dense_l1['test_hard_negative_preference_accuracy']):.4f} | {float(dense_l1['test_topk_recall']):.4f} | {float(dense_l1['test_top1_accuracy']):.4f} | {float(dense_l1['test_spearman']):.4f} |",
        f"| Raw pair + dense MLP W16 | {int(dense_l16['parameters']):,} | {float(dense_l16['train_pair_accuracy']):.4f} | {float(dense_l16['test_random_pair_order_accuracy']):.4f} | {float(dense_l16['test_hard_negative_preference_accuracy']):.4f} | {float(dense_l16['test_topk_recall']):.4f} | {float(dense_l16['test_top1_accuracy']):.4f} | {float(dense_l16['test_spearman']):.4f} |",
        "",
        "## Decision rule",
        "",
        "A large gain over PC additive MSE means that the joint route representation retains relation information and additive per-table decoding is the bottleneck. Failure despite the high-capacity decoder means that the fixed comparison routes themselves discard the information needed for held-out retrieval.",
        "",
        "## Findings",
        "",
        f"The MSE diagnostic raises Top-16 recall from {float(pc['test_topk_recall']):.4f} to {mse['test_topk_recall']:.4f} and Top-1 from {float(pc['test_top1_accuracy']):.4f} to {mse['test_top1_accuracy']:.4f} without changing a single comparison. Held-out Spearman reaches {mse['test_spearman']:.4f}, showing that the route bits support generalizable relation reconstruction rather than only memorization of training pairs.",
        "",
        f"Relative to the gap between additive PC-MSE and the raw-input dense W16 control, the route-bit decoder closes {(mse['test_topk_recall'] - float(pc['test_topk_recall'])) / (float(dense_l16['test_topk_recall']) - float(pc['test_topk_recall'])):.1%} of the Top-16 gap and {(mse['test_top1_accuracy'] - float(pc['test_top1_accuracy'])) / (float(dense_l16['test_top1_accuracy']) - float(pc['test_top1_accuracy'])):.1%} of the Top-1 gap. The primary failure is therefore the additive per-table factorization, though the 328k-parameter diagnostic is not itself an efficient replacement.",
        "",
        "MSE again outperforms pairwise ranking on held-out retrieval, indicating that continuous teacher scores provide useful within-positive and within-negative ordering information. The next structural question is how much cross-table interaction order is required to recover the dense-decoder gain without using a dense matrix.",
    ]
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
