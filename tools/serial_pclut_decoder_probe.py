from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import torch
from torch import Tensor, nn

from tropnn.layers import PairwiseLUT
from tropnn.tools.bilinear_retrieval_probe import predict_score_matrix, retrieval_metrics, teacher_scores
from tropnn.tools.fixed_route_relation_energy_probe import make_relation_pairs, pair_accuracy


DEPTHS = (2, 4, 8, 16)


class SerialPCLUTDecoder(nn.Module):
    """Residual PC-LUT composition followed by a scalar PC-LUT readout."""

    def __init__(
        self,
        input_dim: int,
        *,
        depth: int,
        tables: int,
        comparisons: int,
        seed: int,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be positive, got {depth}")
        layer = dict(
            tables=tables,
            comparisons=comparisons,
            backend="torch",
            lut_init_std=0.0,
            use_min_margin_ste=True,
            use_output_scaling=True,
            fixed_zero_threshold=True,
            anchor_policy="random",
        )
        self.blocks = nn.ModuleList(
            PairwiseLUT(
                input_dim=input_dim,
                output_dim=input_dim,
                seed=seed + 1009 * index,
                anchor_seed=seed + 1009 * index,
                **layer,
            )
            for index in range(depth - 1)
        )
        self.readout = PairwiseLUT(
            input_dim=input_dim,
            output_dim=1,
            seed=seed + 1009 * (depth - 1),
            anchor_seed=seed + 1009 * (depth - 1),
            **layer,
        )

    def forward(self, pair: Tensor) -> Tensor:
        hidden = pair.unsqueeze(1)
        for block in self.blocks:
            hidden = hidden + block(hidden)
        return self.readout(hidden)[:, 0, :]

    @torch.no_grad()
    def payload_rms(self) -> list[float]:
        layers = [*self.blocks, self.readout]
        return [float(layer.lut.square().mean().sqrt().item()) for layer in layers]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="True serial residual PC-LUT relation decoder sweep.")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run")
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
    run.add_argument("--eval-batch-size", type=int, default=4096)
    run.add_argument("--eval-every", type=int, default=1000)
    run.add_argument("--top-k", type=int, default=16)
    run.add_argument("--lr", type=float, default=0.001)
    run.add_argument("--weight-decay", type=float, default=0.0)
    run.add_argument("--device", default="cuda")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--out-dir", type=Path, required=True)

    summarize = commands.add_parser("summarize")
    summarize.add_argument("--result-dir", type=Path, required=True)
    summarize.add_argument("--width-summary", type=Path, required=True)
    summarize.add_argument("--wide-decoder-summary", type=Path, required=True)
    summarize.add_argument("--out-report", type=Path, required=True)
    return parser


def train_loss(model: nn.Module, pairs: object, indices: Tensor) -> Tensor:
    positive = model(pairs.positive[indices])
    negative = model(pairs.negative[indices])
    return 0.5 * (
        (positive - pairs.positive_target[indices]).square().mean()
        + (negative - pairs.negative_target[indices]).square().mean()
    )


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def run(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    pairs = make_relation_pairs(args, device)
    model = SerialPCLUTDecoder(
        2 * args.input_dim,
        depth=args.depth,
        tables=args.tables,
        comparisons=args.comparisons,
        seed=args.seed,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    generator = torch.Generator(device=device).manual_seed(args.seed + 809)
    history: list[dict[str, float | int]] = []

    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        indices = torch.randint(
            0,
            pairs.positive.shape[0],
            (args.batch_size,),
            generator=generator,
            device=device,
        )
        loss = train_loss(model, pairs, indices)
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
        "variant": "serial_residual_pclut",
        "depth": args.depth,
        "residual_blocks": args.depth - 1,
        "hidden_dim": 2 * args.input_dim,
        "tables_per_layer": args.tables,
        "comparisons": args.comparisons,
        "fixed_zero_thresholds": True,
        "zero_payload_initialization": True,
        "parameters": parameter_count(model),
        "steps": args.steps,
        "elapsed_seconds": elapsed,
        "steps_per_second": args.steps / elapsed,
        "seed": args.seed,
        "train_pair_accuracy": pair_accuracy(model, pairs, args.eval_batch_size),
        "payload_rms_by_layer": model.payload_rms(),
        **{f"test_{key}": value for key, value in metrics.items()},
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"serial_L{args.depth}_seed{args.seed}"
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
    results = [json.loads(path.read_text()) for path in sorted(args.result_dir.glob("serial_L*_seed*.json"))]
    missing = [
        (depth, seed)
        for depth in DEPTHS
        for seed in (0, 1)
        if not any(row["depth"] == depth and row["seed"] == seed for row in results)
    ]
    if missing:
        raise RuntimeError(f"Missing serial runs: {missing}")

    summary_path = args.result_dir / "summary.csv"
    scalar_rows = [{key: value for key, value in row.items() if key != "payload_rms_by_layer"} for row in results]
    fieldnames = sorted({key for row in scalar_rows for key in row})
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scalar_rows)

    width_rows = read_csv(args.width_summary)
    pc_wide = next(row for row in width_rows if row["variant"] == "pc_mse_adamw" and int(row["width"]) == 16)
    dense_wide = next(row for row in width_rows if row["variant"] == "dense_mlp_pairwise" and int(row["width"]) == 16)
    dense_route = next(row for row in read_csv(args.wide_decoder_summary) if row["objective"] == "mse")
    metric_names = (
        "train_pair_accuracy",
        "test_random_pair_order_accuracy",
        "test_hard_negative_preference_accuracy",
        "test_topk_recall",
        "test_top1_accuracy",
        "test_spearman",
        "test_score_r2",
        "steps_per_second",
    )

    lines = [
        "# True Serial Residual PC-LUT Relation Decoder",
        "",
        "## Architecture",
        "",
        "Depth counts all PC-LUT layers. L2 contains one 64-dimensional residual PC-LUT block and one scalar PC-LUT readout; L16 contains fifteen residual blocks and one readout. Every hidden route reads the state produced by the preceding residual block:",
        "",
        "```text",
        "h0 = concat(q, k)",
        "h[l+1] = h[l] + PC_LUT[l](h[l])",
        "score = PC_LUT_readout(h[L-1])",
        "```",
        "",
        "All layers use T16/C5, fixed random anchors, fixed zero thresholds, zero payload initialization, min-margin STE, and AdamW score MSE. This is recursive composition, unlike the renamed W16/T256 parallel route ensemble.",
        "",
        "## Runs",
        "",
        "| L | Seed | Params | Train pair | Held pair | Hard-neg | Top-16 | Top-1 | Spearman | Score R2 | Steps/s |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(results, key=lambda item: (item["depth"], item["seed"])):
        lines.append(
            f"| {row['depth']} | {row['seed']} | {row['parameters']:,} | {row['train_pair_accuracy']:.4f} | "
            f"{row['test_random_pair_order_accuracy']:.4f} | {row['test_hard_negative_preference_accuracy']:.4f} | "
            f"{row['test_topk_recall']:.4f} | {row['test_top1_accuracy']:.4f} | {row['test_spearman']:.4f} | "
            f"{row['test_score_r2']:.4f} | {row['steps_per_second']:.1f} |"
        )

    lines.extend(["", "## Two-seed means", "", "| L | Params | Train pair | Held pair | Hard-neg | Top-16 | Top-1 | Spearman | Score R2 | Steps/s |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for depth in DEPTHS:
        selected = [row for row in results if row["depth"] == depth]
        means = {name: statistics.mean(float(row[name]) for row in selected) for name in metric_names}
        lines.append(
            f"| {depth} | {selected[0]['parameters']:,} | {means['train_pair_accuracy']:.4f} | "
            f"{means['test_random_pair_order_accuracy']:.4f} | {means['test_hard_negative_preference_accuracy']:.4f} | "
            f"{means['test_topk_recall']:.4f} | {means['test_top1_accuracy']:.4f} | {means['test_spearman']:.4f} | "
            f"{means['test_score_r2']:.4f} | {means['steps_per_second']:.1f} |"
        )

    lines.extend(
        [
            "",
            "## Existing width and diagnostic controls",
            "",
            "| Control | Params | Train pair | Held pair | Hard-neg | Top-16 | Top-1 | Spearman |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            f"| Parallel PC-LUT W16/T256 | {int(pc_wide['parameters']):,} | {float(pc_wide['train_pair_accuracy']):.4f} | {float(pc_wide['test_random_pair_order_accuracy']):.4f} | {float(pc_wide['test_hard_negative_preference_accuracy']):.4f} | {float(pc_wide['test_topk_recall']):.4f} | {float(pc_wide['test_top1_accuracy']):.4f} | {float(pc_wide['test_spearman']):.4f} |",
            f"| Parallel dense MLP W16 | {int(dense_wide['parameters']):,} | {float(dense_wide['train_pair_accuracy']):.4f} | {float(dense_wide['test_random_pair_order_accuracy']):.4f} | {float(dense_wide['test_hard_negative_preference_accuracy']):.4f} | {float(dense_wide['test_topk_recall']):.4f} | {float(dense_wide['test_top1_accuracy']):.4f} | {float(dense_wide['test_spearman']):.4f} |",
            f"| Fixed wide route + dense H256 | {int(dense_route['trainable_parameters']):,} | {float(dense_route['train_pair_accuracy']):.4f} | {float(dense_route['test_random_pair_order_accuracy']):.4f} | {float(dense_route['test_hard_negative_preference_accuracy']):.4f} | {float(dense_route['test_topk_recall']):.4f} | {float(dense_route['test_top1_accuracy']):.4f} | {float(dense_route['test_spearman']):.4f} |",
            "",
            "The serial model is not parameter-matched to the scalar parallel ensemble: hidden payloads are 64-dimensional because they must transform the state consumed by later layers. Results therefore test whether recursive composition is useful, while parameter efficiency must be interpreted from the reported counts.",
        ]
    )

    def mean_at(depth: int, metric: str) -> float:
        return statistics.mean(float(row[metric]) for row in results if row["depth"] == depth)

    l8_parameters = next(int(row["parameters"]) for row in results if row["depth"] == 8)
    l16_parameters = next(int(row["parameters"]) for row in results if row["depth"] == 16)
    parallel_parameters = int(pc_wide["parameters"])
    lines.extend(
        [
            "",
            "## Findings",
            "",
            f"True recursive depth has a reproducible positive signal. From L2 to L16, mean held-pair accuracy rises from {mean_at(2, 'test_random_pair_order_accuracy'):.4f} to {mean_at(16, 'test_random_pair_order_accuracy'):.4f}, Top-16 from {mean_at(2, 'test_topk_recall'):.4f} to {mean_at(16, 'test_topk_recall'):.4f}, and Spearman from {mean_at(2, 'test_spearman'):.4f} to {mean_at(16, 'test_spearman'):.4f}. The mean train-minus-held pair gap shrinks from {mean_at(2, 'train_pair_accuracy') - mean_at(2, 'test_random_pair_order_accuracy'):.4f} to {mean_at(16, 'train_pair_accuracy') - mean_at(16, 'test_random_pair_order_accuracy'):.4f}, so the depth gain is not explained by a growing train-only fit.",
            "",
            f"The gain is parameter-inefficient. L8 uses {l8_parameters:,} parameters ({l8_parameters / parallel_parameters:.1f}x the parallel W16/T256 control) to approximately match its Top-16 result. L16 uses {l16_parameters:,} parameters ({l16_parameters / parallel_parameters:.1f}x) and improves Top-16 by only {mean_at(16, 'test_topk_recall') - float(pc_wide['test_topk_recall']):+.4f}; its mean Top-1 remains {mean_at(16, 'test_top1_accuracy'):.4f} versus {float(pc_wide['test_top1_accuracy']):.4f} for the parallel control.",
            "",
            f"Sequential cost scales almost linearly with depth: throughput falls from {mean_at(2, 'steps_per_second'):.1f} steps/s at L2 to {mean_at(16, 'steps_per_second'):.1f} at L16. The dense H256 diagnostic remains much stronger despite fewer parameters than serial L16, so recursive PC-LUT composition does not close the relation-decoding gap.",
            "",
            f"All mean score R2 values remain negative, including {mean_at(16, 'test_score_r2'):.4f} at L16. Depth improves ranking structure without learning a well-calibrated global bilinear score under this 10k-step setup.",
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
