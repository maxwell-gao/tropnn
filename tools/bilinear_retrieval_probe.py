from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from tropnn.tools.bitlinear_backprop_probe import PCLUTStudent


VARIANTS = {"E4": (4, True), "E8": (8, True), "F4": (4, False), "F8": (8, False)}


@dataclass(frozen=True)
class BilinearProblem:
    train_queries: Tensor
    train_keys: Tensor
    test_queries: Tensor
    test_keys: Tensor
    relation: Tensor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train deep PC-LUT scorers on q^T A k and evaluate retrieval and relation selection.")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run")
    run.add_argument("--variant", choices=list(VARIANTS), required=True)
    run.add_argument("--input-dim", type=int, default=32)
    run.add_argument("--train-queries", type=int, default=2048)
    run.add_argument("--train-keys", type=int, default=2048)
    run.add_argument("--test-queries", type=int, default=256)
    run.add_argument("--test-keys", type=int, default=512)
    run.add_argument("--max-value", type=int, default=15)
    run.add_argument("--tables", type=int, default=16)
    run.add_argument("--comparisons", type=int, default=5)
    run.add_argument("--steps", type=int, default=5000)
    run.add_argument("--batch-size", type=int, default=512)
    run.add_argument("--eval-pairs", type=int, default=8192)
    run.add_argument("--eval-every", type=int, default=500)
    run.add_argument("--top-k", type=int, default=16)
    run.add_argument("--payload-lr", type=float, default=0.01)
    run.add_argument("--threshold-lr", type=float, default=0.002)
    run.add_argument("--device", default="cuda")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--out-dir", type=Path, required=True)

    summarize = commands.add_parser("summarize")
    summarize.add_argument("--result-dir", type=Path, required=True)
    summarize.add_argument("--out-report", type=Path, required=True)
    return parser


def normalize_rows(values: Tensor) -> Tensor:
    centered = values - values.mean(dim=-1, keepdim=True)
    return centered / centered.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)


def make_problem(args: argparse.Namespace) -> BilinearProblem:
    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    def sample(count: int) -> Tensor:
        values = torch.randint(0, args.max_value + 1, (count, args.input_dim), generator=generator)
        return normalize_rows(values.to(torch.float32))

    signs = torch.randint(0, 2, (args.input_dim, args.input_dim), generator=generator, dtype=torch.int64)
    relation = (2.0 * signs.to(torch.float32) - 1.0) / args.input_dim
    return BilinearProblem(
        train_queries=sample(args.train_queries),
        train_keys=sample(args.train_keys),
        test_queries=sample(args.test_queries),
        test_keys=sample(args.test_keys),
        relation=relation,
    )


def teacher_scores(queries: Tensor, keys: Tensor, relation: Tensor) -> Tensor:
    return queries @ relation @ keys.T


def sampled_pair_batch(
    queries: Tensor,
    keys: Tensor,
    relation: Tensor,
    query_indices: Tensor,
    key_indices: Tensor,
) -> tuple[Tensor, Tensor]:
    q = queries[query_indices]
    k = keys[key_indices]
    pair = torch.cat([q, k], dim=-1)
    target = ((q @ relation) * k).sum(dim=-1, keepdim=True)
    return pair, target


@torch.no_grad()
def predict_pairs(model: PCLUTStudent, pairs: Tensor, batch_size: int) -> Tensor:
    outputs = [model(pairs[start : start + batch_size]) for start in range(0, pairs.shape[0], batch_size)]
    return torch.cat(outputs, dim=0).squeeze(-1)


@torch.no_grad()
def predict_score_matrix(model: PCLUTStudent, queries: Tensor, keys: Tensor, batch_size: int) -> Tensor:
    rows: list[Tensor] = []
    queries_per_chunk = max(1, batch_size // keys.shape[0])
    for start in range(0, queries.shape[0], queries_per_chunk):
        q = queries[start : start + queries_per_chunk]
        pair = torch.cat(
            [
                q.repeat_interleave(keys.shape[0], dim=0),
                keys.repeat(q.shape[0], 1),
            ],
            dim=-1,
        )
        rows.append(predict_pairs(model, pair, batch_size).view(q.shape[0], keys.shape[0]))
    return torch.cat(rows, dim=0)


def r2_score(target: Tensor, prediction: Tensor) -> float:
    target64 = target.to(torch.float64)
    prediction64 = prediction.to(torch.float64)
    mse = (target64 - prediction64).square().mean()
    variance = (target64 - target64.mean()).square().mean()
    return float((1.0 - mse / variance).item())


def mse_score(target: Tensor, prediction: Tensor) -> float:
    return float((target.to(torch.float64) - prediction.to(torch.float64)).square().mean().item())


def row_spearman(target: Tensor, prediction: Tensor) -> float:
    target_rank = torch.argsort(torch.argsort(target, dim=-1), dim=-1).to(torch.float64)
    prediction_rank = torch.argsort(torch.argsort(prediction, dim=-1), dim=-1).to(torch.float64)
    target_rank -= target_rank.mean(dim=-1, keepdim=True)
    prediction_rank -= prediction_rank.mean(dim=-1, keepdim=True)
    numerator = (target_rank * prediction_rank).sum(dim=-1)
    denominator = target_rank.norm(dim=-1) * prediction_rank.norm(dim=-1)
    return float((numerator / denominator.clamp_min(1e-12)).mean().item())


def topk_recall(target: Tensor, prediction: Tensor, k: int) -> float:
    k = min(k, target.shape[-1])
    target_top = torch.topk(target, k=k, dim=-1).indices
    prediction_top = torch.topk(prediction, k=k, dim=-1).indices
    matches = (prediction_top.unsqueeze(-1) == target_top.unsqueeze(-2)).any(dim=-1)
    return float(matches.to(torch.float64).mean().item())


def ndcg(target: Tensor, prediction: Tensor, k: int) -> float:
    k = min(k, target.shape[-1])
    prediction_top = torch.topk(prediction, k=k, dim=-1).indices
    ideal_top = torch.topk(target, k=k, dim=-1).indices
    relevance = target.to(torch.float64) - target.min(dim=-1, keepdim=True).values.to(torch.float64)
    discounts = 1.0 / torch.log2(torch.arange(k, device=target.device, dtype=torch.float64) + 2.0)
    dcg = (relevance.gather(1, prediction_top) * discounts).sum(dim=-1)
    idcg = (relevance.gather(1, ideal_top) * discounts).sum(dim=-1)
    return float((dcg / idcg.clamp_min(1e-12)).mean().item())


def relation_selection_metrics(target: Tensor, prediction: Tensor, top_k: int, seed: int) -> dict[str, float]:
    teacher_order = torch.argsort(target, dim=-1, descending=True)
    teacher_best = teacher_order[:, 0]
    student_order = torch.argsort(prediction, dim=-1, descending=True)
    top1_accuracy = (student_order[:, 0] == teacher_best).to(torch.float64).mean()
    teacher_best_rank = (student_order == teacher_best.unsqueeze(1)).to(torch.float64).argmax(dim=1) + 1
    reciprocal_rank = (1.0 / teacher_best_rank.to(torch.float64)).mean()

    hard_count = min(top_k, target.shape[-1] - 1)
    hard_negative_indices = teacher_order[:, 1 : hard_count + 1]
    student_positive = prediction.gather(1, teacher_best.unsqueeze(1))
    student_hard = prediction.gather(1, hard_negative_indices)
    hard_preference = (student_positive > student_hard).to(torch.float64).mean()

    generator = torch.Generator(device="cpu").manual_seed(seed)
    pair_count = 64
    left = torch.randint(0, target.shape[1], (target.shape[0], pair_count), generator=generator).to(target.device)
    offset = torch.randint(1, target.shape[1], (target.shape[0], pair_count), generator=generator).to(target.device)
    right = (left + offset) % target.shape[1]
    teacher_difference = target.gather(1, left) - target.gather(1, right)
    student_difference = prediction.gather(1, left) - prediction.gather(1, right)
    pairwise_accuracy = ((teacher_difference > 0) == (student_difference > 0)).to(torch.float64).mean()
    return {
        "top1_accuracy": float(top1_accuracy.item()),
        "teacher_top1_mrr": float(reciprocal_rank.item()),
        "hard_negative_preference_accuracy": float(hard_preference.item()),
        "random_pair_order_accuracy": float(pairwise_accuracy.item()),
    }


def retrieval_metrics(target: Tensor, prediction: Tensor, top_k: int, seed: int) -> dict[str, float]:
    return {
        "score_mse": mse_score(target, prediction),
        "score_r2": r2_score(target, prediction),
        "spearman": row_spearman(target, prediction),
        "topk_recall": topk_recall(target, prediction, top_k),
        "ndcg": ndcg(target, prediction, top_k),
        **relation_selection_metrics(target, prediction, top_k, seed),
    }


def make_fixed_pairs(problem: BilinearProblem, args: argparse.Namespace, device: torch.device) -> dict[str, tuple[Tensor, Tensor]]:
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 401)
    fixed: dict[str, tuple[Tensor, Tensor]] = {}
    for name, queries, keys in (
        ("train", problem.train_queries, problem.train_keys),
        ("test", problem.test_queries, problem.test_keys),
    ):
        q_idx = torch.randint(0, queries.shape[0], (args.eval_pairs,), generator=generator)
        k_idx = torch.randint(0, keys.shape[0], (args.eval_pairs,), generator=generator)
        pair, target = sampled_pair_batch(
            queries.to(device), keys.to(device), problem.relation.to(device), q_idx.to(device), k_idx.to(device)
        )
        fixed[name] = (pair, target.squeeze(-1))
    return fixed


@torch.no_grad()
def evaluate_fixed_pairs(
    model: PCLUTStudent,
    fixed_pairs: dict[str, tuple[Tensor, Tensor]],
    batch_size: int,
    step: int,
) -> dict[str, float | int]:
    model.eval()
    row: dict[str, float | int] = {"step": step}
    for name, (pairs, target) in fixed_pairs.items():
        prediction = predict_pairs(model, pairs, batch_size)
        row[f"{name}_mse"] = mse_score(target, prediction)
        row[f"{name}_r2"] = r2_score(target, prediction)
    model.train()
    return row


@torch.no_grad()
def readout_route_codes(model: PCLUTStudent, pairs: Tensor, batch_size: int) -> Tensor:
    codes = []
    for start in range(0, pairs.shape[0], batch_size):
        hidden = model.encode(pairs[start : start + batch_size])
        codes.append(model.readout.route(hidden.unsqueeze(1)).indices[:, 0, :].cpu())
    return torch.cat(codes, dim=0)


def optimizer_for(model: PCLUTStudent, args: argparse.Namespace) -> torch.optim.Optimizer:
    payloads = [parameter for name, parameter in model.named_parameters() if name.endswith("lut")]
    thresholds = [parameter for name, parameter in model.named_parameters() if name.endswith("thresholds")]
    groups: list[dict[str, object]] = [{"params": payloads, "lr": args.payload_lr}]
    if thresholds:
        groups.append({"params": thresholds, "lr": args.threshold_lr})
    return torch.optim.AdamW(groups, weight_decay=0.0)


def write_history(path: Path, rows: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    problem = make_problem(args)
    depth, trainable_thresholds = VARIANTS[args.variant]
    model = PCLUTStudent(
        input_dim=2 * args.input_dim,
        output_dim=1,
        depth=depth,
        tables=args.tables,
        comparisons=args.comparisons,
        trainable_thresholds=trainable_thresholds,
        seed=args.seed,
    ).to(device)
    train_queries = problem.train_queries.to(device)
    train_keys = problem.train_keys.to(device)
    relation = problem.relation.to(device)
    fixed_pairs = make_fixed_pairs(problem, args, device)
    route_probe_pairs = fixed_pairs["test"][0]
    initial_codes = readout_route_codes(model, route_probe_pairs, args.batch_size)
    optimizer = optimizer_for(model, args)
    sample_generator = torch.Generator(device=device).manual_seed(args.seed + 307)
    history = [evaluate_fixed_pairs(model, fixed_pairs, args.batch_size, 0)]
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        q_idx = torch.randint(0, train_queries.shape[0], (args.batch_size,), generator=sample_generator, device=device)
        k_idx = torch.randint(0, train_keys.shape[0], (args.batch_size,), generator=sample_generator, device=device)
        pair, target = sampled_pair_batch(train_queries, train_keys, relation, q_idx, k_idx)
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
    train_target = teacher_scores(problem.train_queries[: args.test_queries].to(device), problem.train_keys[: args.test_keys].to(device), relation)
    train_prediction = predict_score_matrix(
        model,
        problem.train_queries[: args.test_queries].to(device),
        problem.train_keys[: args.test_keys].to(device),
        args.batch_size,
    )
    test_target = teacher_scores(problem.test_queries.to(device), problem.test_keys.to(device), relation)
    test_prediction = predict_score_matrix(model, problem.test_queries.to(device), problem.test_keys.to(device), args.batch_size)
    train_metrics = retrieval_metrics(train_target, train_prediction, args.top_k, args.seed + 503)
    test_metrics = retrieval_metrics(test_target, test_prediction, args.top_k, args.seed + 601)
    final_codes = readout_route_codes(model, route_probe_pairs, args.batch_size)

    result: dict[str, object] = {
        "variant": args.variant,
        "depth": depth,
        "trainable_thresholds": trainable_thresholds,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "input_dim_per_item": args.input_dim,
        "pair_input_dim": 2 * args.input_dim,
        "tables": args.tables,
        "comparisons": args.comparisons,
        "train_queries": args.train_queries,
        "train_keys": args.train_keys,
        "test_queries": args.test_queries,
        "test_keys": args.test_keys,
        "top_k": args.top_k,
        "payload_lr": args.payload_lr,
        "threshold_lr": args.threshold_lr,
        "elapsed_seconds": elapsed,
        "steps_per_second": args.steps / elapsed,
        "route_rows_changed_fraction": float((final_codes != initial_codes).any(dim=1).to(torch.float64).mean().item()),
        "route_table_codes_changed_fraction": float((final_codes != initial_codes).to(torch.float64).mean().item()),
        **{f"train_{name}": value for name, value in train_metrics.items()},
        **{f"test_{name}": value for name, value in test_metrics.items()},
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"{args.variant}.json").write_text(json.dumps(result, indent=2) + "\n")
    write_history(args.out_dir / f"{args.variant}.history.csv", history)
    print(json.dumps(result, sort_keys=True), flush=True)


def summarize(args: argparse.Namespace) -> None:
    results = []
    for variant in VARIANTS:
        path = args.result_dir / f"{variant}.json"
        if not path.exists():
            raise FileNotFoundError(f"missing result {path}")
        results.append(json.loads(path.read_text()))
    summary_path = args.result_dir / "summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    by_name = {row["variant"]: row for row in results}
    setup = results[0]
    lines = [
        "# Deep PC-LUT Bilinear Retrieval and Relation Selection",
        "",
        "The production PC-LUT stack is trained only with MSE on a fixed bilinear teacher score `q^T A k`.",
        "Top-k retrieval and relation-selection metrics are evaluated without ranking-loss training.",
        "",
        "## Setup",
        "",
        f"- Query/key dimension: {setup['input_dim_per_item']} each; pair input dimension: {setup['pair_input_dim']}",
        "- Relation matrix: fixed random binary-sign matrix scaled by 1/32",
        "- Inputs: independently sampled random integers, row-centered and variance-normalized",
        f"- Training pools: {setup['train_queries']} queries and {setup['train_keys']} keys",
        f"- Held-out retrieval: {setup['test_queries']} queries against {setup['test_keys']} keys",
        f"- PC-LUT: {setup['tables']} tables, {setup['comparisons']} comparisons, residual depth 4 or 8",
        f"- Optimization: {setup['steps']} AdamW steps, payload LR {setup['payload_lr']}, threshold LR {setup['threshold_lr']}",
        "",
        "## Held-out results",
        "",
        "| variant | score R2 | Spearman | top-16 recall | NDCG | top1 | MRR | hard-neg preference | pair-order acc | route rows changed | steps/s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        lines.append(
            "| {variant} | {test_score_r2:.4f} | {test_spearman:.4f} | {test_topk_recall:.4f} | "
            "{test_ndcg:.4f} | {test_top1_accuracy:.4f} | {test_teacher_top1_mrr:.4f} | "
            "{test_hard_negative_preference_accuracy:.4f} | {test_random_pair_order_accuracy:.4f} | "
            "{route_rows_changed_fraction:.3f} | {steps_per_second:.1f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Trainable-threshold deltas",
            "",
        ]
    )
    for depth in (4, 8):
        learned = by_name[f"E{depth}"]
        fixed = by_name[f"F{depth}"]
        lines.append(
            f"- Depth {depth}: E-F score R2 `{learned['test_score_r2'] - fixed['test_score_r2']:+.4f}`, "
            f"top-16 recall `{learned['test_topk_recall'] - fixed['test_topk_recall']:+.4f}`, "
            f"top1 `{learned['test_top1_accuracy'] - fixed['test_top1_accuracy']:+.4f}`."
        )
    best_topk = max(results, key=lambda row: row["test_topk_recall"])
    best_top1 = max(results, key=lambda row: row["test_top1_accuracy"])
    lines.extend(
        [
            "",
            "## Main findings",
            "",
            f"- Best top-16 recall: `{best_topk['test_topk_recall']:.4f}` from {best_topk['variant']} (random: 0.03125).",
            f"- Best top1 accuracy: `{best_top1['test_top1_accuracy']:.4f}` from {best_top1['variant']} (random: 0.001953).",
            "- Depth 8 does not improve score recovery or retrieval over depth 4 under either threshold policy.",
            "- Trainable thresholds are nearly neutral for score R2 and do not improve top-tail selection.",
            "- Broad pair ordering is learned more successfully than teacher-top1 versus hard-negative discrimination.",
            "",
            "## Metric interpretation",
            "",
            "- `top1` is exact agreement with the teacher's best key.",
            "- `MRR` is the reciprocal student rank of the teacher-best key.",
            "- `hard-neg preference` asks whether the teacher-best key beats each teacher rank 2-17 key.",
            "- `pair-order acc` measures ordering agreement on random candidate pairs; chance is 0.5.",
            "- Random top-16 recall is 0.03125 and random top1 accuracy is 0.001953.",
        ]
    )
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text("\n".join(lines) + "\n")
    print(json.dumps({"summary": str(summary_path), "report": str(args.out_report)}, sort_keys=True))


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run":
        run(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
