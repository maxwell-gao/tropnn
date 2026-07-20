"""Compare relation objectives without changing the unary Coxeter model."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

from tropnn.tools.learned_unary_coxeter_attention_probe import (
    UnaryRelationModel,
    fit_relation_for_coordinates,
    fit_unary,
    make_groups,
    make_problem,
    make_unary_anchors,
    metrics,
    permutation_tensors,
    sample_pairs,
    score_matrix,
    unary_codes,
    unary_design,
)

LOSSES = ("mse", "ranking", "listwise", "ranking_listwise")
INITS = ("supervised", "random")


def relation_objective(
    prediction: Tensor,
    target: Tensor,
    hard_negative: Tensor,
    *,
    name: str,
    temperature: float,
) -> Tensor:
    if name == "mse":
        return F.mse_loss(prediction, target)

    positive_index = hard_negative[:, :16]
    negative_index = hard_negative[:, 16:64]
    positive = prediction.gather(1, positive_index)
    negative = prediction.gather(1, negative_index)
    ranking = F.softplus(-(positive.unsqueeze(-1) - negative.unsqueeze(-2)) / temperature).mean()
    if name == "ranking":
        return ranking

    teacher_probability = F.softmax(target / temperature, dim=-1)
    listwise = F.kl_div(
        F.log_softmax(prediction / temperature, dim=-1),
        teacher_probability,
        reduction="batchmean",
    )
    if name == "listwise":
        return listwise
    if name == "ranking_listwise":
        return ranking + listwise
    raise ValueError(f"unknown loss: {name}")


def train(
    model: UnaryRelationModel,
    train_code: Tensor,
    relation: Tensor,
    hard_negative: Tensor,
    *,
    loss_name: str,
    steps: int,
    batch_queries: int,
    temperature: float,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> float:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    objects = train_code.shape[0]
    all_keys = torch.arange(objects, device=train_code.device)
    final_loss = math.nan
    for _ in range(steps):
        query = torch.randint(objects, (batch_queries,), generator=generator).to(train_code.device)
        query_index = query.view(-1, 1).expand(-1, objects).reshape(-1)
        key_index = all_keys.view(1, -1).expand(batch_queries, -1).reshape(-1)
        prediction = model(train_code, query_index, key_index).view(batch_queries, objects)
        target = relation[query]
        loss = relation_objective(
            prediction,
            target,
            hard_negative[query],
            name=loss_name,
            temperature=temperature,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        final_loss = float(loss.detach().item())
    return final_loss


def evaluate_model(
    model: UnaryRelationModel,
    test_code: Tensor,
    test_relation: Tensor,
    values: Tensor,
) -> dict[str, float]:
    with torch.no_grad():
        query, key = model.coordinates(test_code)
        prediction = score_matrix(
            query,
            key,
            model.groups,
            model.orders,
            model.relative,
            model.weight,
            model.bias,
        )
    return metrics(prediction, test_relation, values)


def run(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    problem = make_problem(
        args.seed,
        device,
        dimension=args.dimension,
        rank=args.rank,
        train_objects=args.train_objects,
        test_objects=args.test_objects,
    )
    orders, relative, adjacent = permutation_tensors(device)
    groups = make_groups(args.rank, args.relation_tables, args.seed + 101, device)
    anchors = make_unary_anchors(
        args.dimension,
        args.unary_tables,
        args.comparisons,
        args.seed + 211,
        device,
    )
    train_code = unary_codes(problem.train_x, anchors)
    test_code = unary_codes(problem.test_x, anchors)
    rows = 1 << args.comparisons
    train_design = unary_design(train_code, rows)
    test_design = unary_design(test_code, rows)

    pair_generator = torch.Generator().manual_seed(args.seed + 307)
    fit_query, fit_key = sample_pairs(args.fit_pairs, args.train_objects, pair_generator, device)
    values = torch.randn(
        args.test_objects,
        args.value_dimension,
        generator=pair_generator,
    ).to(device)

    oracle_weight, oracle_bias = fit_relation_for_coordinates(
        problem.train_q,
        problem.train_k,
        problem.train_relation,
        groups,
        orders,
        relative,
        fit_query,
        fit_key,
    )
    oracle_prediction = score_matrix(
        problem.test_q,
        problem.test_k,
        groups,
        orders,
        relative,
        oracle_weight,
        oracle_bias,
    )
    oracle_metrics = metrics(oracle_prediction, problem.test_relation, values)

    supervised_query = fit_unary(train_design, problem.train_q, args.unary_ridge)
    supervised_key = fit_unary(train_design, problem.train_k, args.unary_ridge)
    supervised_train_q = train_design @ supervised_query
    supervised_train_k = train_design @ supervised_key
    supervised_weight, supervised_bias = fit_relation_for_coordinates(
        supervised_train_q,
        supervised_train_k,
        problem.train_relation,
        groups,
        orders,
        relative,
        fit_query,
        fit_key,
    )

    random_query = torch.randn(args.unary_tables, rows, args.rank, device=device) * args.init_std
    random_key = torch.randn(args.unary_tables, rows, args.rank, device=device) * args.init_std
    random_train_q = train_design @ random_query.view(-1, args.rank)
    random_train_k = train_design @ random_key.view(-1, args.rank)
    random_weight, random_bias = fit_relation_for_coordinates(
        random_train_q,
        random_train_k,
        problem.train_relation,
        groups,
        orders,
        relative,
        fit_query,
        fit_key,
    )

    if args.initialization == "supervised":
        query_payload = supervised_query.view(args.unary_tables, rows, args.rank)
        key_payload = supervised_key.view(args.unary_tables, rows, args.rank)
        weight, bias = supervised_weight, supervised_bias
    else:
        query_payload, key_payload = random_query, random_key
        weight, bias = random_weight, random_bias

    model = UnaryRelationModel(
        query_payload.clone(),
        key_payload.clone(),
        groups,
        orders,
        relative,
        adjacent,
        weight.clone(),
        bias.clone(),
        unary_tables=args.unary_tables,
        temperature=args.ste_temperature,
    ).to(device)
    initial_metrics = evaluate_model(model, test_code, problem.test_relation, values)
    hard_negative = problem.train_relation.topk(64, dim=-1).indices
    final_loss = train(
        model,
        train_code,
        problem.train_relation,
        hard_negative,
        loss_name=args.loss,
        steps=args.steps,
        batch_queries=args.batch_queries,
        temperature=args.loss_temperature,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed + 401,
    )
    final_metrics = evaluate_model(model, test_code, problem.test_relation, values)

    random_test_q = test_design @ random_query.view(-1, args.rank)
    random_test_k = test_design @ random_key.view(-1, args.rank)
    random_prediction = score_matrix(
        random_test_q,
        random_test_k,
        groups,
        orders,
        relative,
        random_weight,
        random_bias,
    )
    random_metrics = metrics(random_prediction, problem.test_relation, values)
    denominator = max(oracle_metrics["top16"] - random_metrics["top16"], 1e-8)
    result: dict[str, object] = {
        "seed": args.seed,
        "loss": args.loss,
        "initialization": args.initialization,
        "rank": args.rank,
        "relation_tables": args.relation_tables,
        "unary_tables": args.unary_tables,
        "comparisons": args.comparisons,
        "steps": args.steps,
        "batch_queries": args.batch_queries,
        "loss_temperature": args.loss_temperature,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "final_train_loss": final_loss,
        "oracle": oracle_metrics,
        "random_fixed": random_metrics,
        "initial": initial_metrics,
        "final": final_metrics,
        "retention": final_metrics["top16"] / max(initial_metrics["top16"], 1e-8),
        "oracle_recovery": (final_metrics["top16"] - random_metrics["top16"]) / denominator,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    return result


def summarize(result_dir: Path, report: Path) -> dict[str, object]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for path in sorted(result_dir.glob("*_seed*.json")):
        row = json.loads(path.read_text())
        grouped[(row["loss"], row["initialization"])].append(row)
    if not grouped:
        raise RuntimeError(f"no result JSON files found in {result_dir}")

    summary = []
    for loss in LOSSES:
        for initialization in INITS:
            rows = grouped[(loss, initialization)]
            summary.append(
                {
                    "loss": loss,
                    "initialization": initialization,
                    "seeds": len(rows),
                    "initial_top16": sum(row["initial"]["top16"] for row in rows) / len(rows),
                    "final_top16": sum(row["final"]["top16"] for row in rows) / len(rows),
                    "final_r2": sum(row["final"]["r2"] for row in rows) / len(rows),
                    "final_top1": sum(row["final"]["top1"] for row in rows) / len(rows),
                    "final_spearman": sum(row["final"]["spearman"] for row in rows) / len(rows),
                    "retrieval_cosine": sum(row["final"]["retrieval_cosine"] for row in rows) / len(rows),
                    "retention": sum(row["retention"] for row in rows) / len(rows),
                    "oracle_recovery": sum(row["oracle_recovery"] for row in rows) / len(rows),
                }
            )

    supervised = {row["loss"]: row for row in summary if row["initialization"] == "supervised"}
    random = {row["loss"]: row for row in summary if row["initialization"] == "random"}
    preserving = [loss for loss in LOSSES if supervised[loss]["retention"] >= 0.9]
    recovering = [loss for loss in preserving if random[loss]["oracle_recovery"] >= 0.8]
    decision = {
        "supervised_preservation_gate": "Top-16 retention >= 0.9",
        "random_recovery_gate": "oracle improvement recovery >= 0.8",
        "losses_preserving_supervised_geometry": preserving,
        "losses_recovering_from_random": recovering,
        "relation_loss_gate_passed": bool(recovering),
        "enter_wiki103": bool(recovering),
    }
    with (result_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (result_dir / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")

    lines = [
        "# Coxeter Relation Loss Ablation",
        "",
        "## Decision",
        "",
        "**Relation-loss gate passed.**" if recovering else "**Relation-loss gate failed; do not enter Wiki103.**",
        "",
        "All runs use the same teacher, objects, fixed unary comparisons, "
        "local-S4 groups, model budget, query batches, and complete 512-key "
        "candidate rows. Only the scalar objective changes.",
        "",
        "## Four-seed means",
        "",
        "| Loss | Init | Initial Top-16 | Final Top-16 | Retention | Oracle recovery | R2 | Spearman | Retrieval cosine |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['loss']} | {row['initialization']} | {row['initial_top16']:.4f} | "
            f"{row['final_top16']:.4f} | {row['retention']:.4f} | {row['oracle_recovery']:.4f} | "
            f"{row['final_r2']:.4f} | {row['final_spearman']:.4f} | {row['retrieval_cosine']:.4f} |"
        )
    lines += [
        "",
        "## Losses",
        "",
        "- `mse`: pointwise score regression over every key.",
        "- `ranking`: logistic ordering of teacher Top-16 positives against ranks 17-64.",
        "- `listwise`: KL between teacher and model softmax distributions over all 512 keys.",
        "- `ranking_listwise`: sum of ranking and listwise objectives.",
        "",
        "The supervised-init control asks whether an objective preserves an "
        "already useful unary coordinate system. Only objectives that pass "
        "this control are eligible for the random-init recovery test.",
        "",
    ]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines))
    print(json.dumps(decision))
    return decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--loss", choices=LOSSES, default="mse")
    parser.add_argument("--initialization", choices=INITS, default="supervised")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dimension", type=int, default=32)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--train-objects", type=int, default=512)
    parser.add_argument("--test-objects", type=int, default=256)
    parser.add_argument("--relation-tables", type=int, default=64)
    parser.add_argument("--unary-tables", type=int, default=256)
    parser.add_argument("--comparisons", type=int, default=5)
    parser.add_argument("--fit-pairs", type=int, default=32768)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-queries", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--unary-ridge", type=float, default=1e-3)
    parser.add_argument("--ste-temperature", type=float, default=0.05)
    parser.add_argument("--loss-temperature", type=float, default=0.05)
    parser.add_argument("--init-std", type=float, default=0.02)
    parser.add_argument("--value-dimension", type=int, default=16)
    parser.add_argument("--output", default="result.json")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.summarize and (args.result_dir is None or args.report is None):
        parser.error("--summarize requires --result-dir and --report")
    return args


def main() -> None:
    args = parse_args()
    if args.summarize:
        summarize(args.result_dir, args.report)
    else:
        run(args)


if __name__ == "__main__":
    main()
