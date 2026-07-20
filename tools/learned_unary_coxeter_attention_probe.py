"""Can unary PC-LUTs learn coordinates that make relative-S4 retrieval work?"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

PERMUTATIONS = list(itertools.permutations(range(4)))
PERMUTATION_INDEX = {permutation: index for index, permutation in enumerate(PERMUTATIONS)}


@dataclass(frozen=True)
class Problem:
    train_x: Tensor
    test_x: Tensor
    train_q: Tensor
    train_k: Tensor
    test_q: Tensor
    test_k: Tensor
    train_relation: Tensor
    test_relation: Tensor


def make_problem(seed: int, device: torch.device, *, dimension: int, rank: int, train_objects: int, test_objects: int) -> Problem:
    generator = torch.Generator().manual_seed(seed)
    count = train_objects + test_objects
    x = torch.randint(-4, 5, (count, dimension), generator=generator, dtype=torch.float32)
    x = F.normalize(x, dim=-1)
    query_factor = torch.linalg.qr(torch.randn(dimension, rank, generator=generator), mode="reduced").Q
    key_factor = torch.linalg.qr(torch.randn(dimension, rank, generator=generator), mode="reduced").Q
    query = x @ query_factor
    key = x @ key_factor
    relation = query @ key.T / math.sqrt(rank)
    split = train_objects
    return Problem(
        train_x=x[:split].to(device),
        test_x=x[split:].to(device),
        train_q=query[:split].to(device),
        train_k=key[:split].to(device),
        test_q=query[split:].to(device),
        test_k=key[split:].to(device),
        train_relation=relation[:split, :split].to(device),
        test_relation=relation[split:, split:].to(device),
    )


def permutation_tensors(device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    orders = torch.tensor(PERMUTATIONS, device=device, dtype=torch.long)
    relative = torch.empty(24, 24, device=device, dtype=torch.long)
    adjacent = torch.empty(24, 3, device=device, dtype=torch.long)
    for left_index, left in enumerate(PERMUTATIONS):
        inverse = [0, 0, 0, 0]
        for position, coordinate in enumerate(left):
            inverse[coordinate] = position
        for right_index, right in enumerate(PERMUTATIONS):
            relative[left_index, right_index] = PERMUTATION_INDEX[tuple(inverse[coordinate] for coordinate in right)]
        for position in range(3):
            neighbor = list(left)
            neighbor[position], neighbor[position + 1] = neighbor[position + 1], neighbor[position]
            adjacent[left_index, position] = PERMUTATION_INDEX[tuple(neighbor)]
    return orders, relative, adjacent


def make_groups(rank: int, tables: int, seed: int, device: torch.device) -> Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.stack([torch.randperm(rank, generator=generator)[:4] for _ in range(tables)]).to(device)


def make_unary_anchors(dimension: int, tables: int, comparisons: int, seed: int, device: torch.device) -> Tensor:
    generator = torch.Generator().manual_seed(seed)
    anchors = torch.empty(tables, comparisons, 2, dtype=torch.long)
    for table in range(tables):
        for comparison in range(comparisons):
            pair = torch.randperm(dimension, generator=generator)[:2]
            anchors[table, comparison] = pair
    return anchors.to(device)


def unary_codes(x: Tensor, anchors: Tensor) -> Tensor:
    margins = x[:, anchors[..., 0]] - x[:, anchors[..., 1]]
    powers = (2 ** torch.arange(anchors.shape[1], device=x.device)).view(1, 1, -1)
    return ((margins > 0).long() * powers).sum(dim=-1)


def unary_design(codes: Tensor, rows: int) -> Tensor:
    objects, tables = codes.shape
    design = torch.zeros(objects, tables * rows, device=codes.device)
    offsets = torch.arange(tables, device=codes.device).view(1, -1) * rows
    design.scatter_(1, codes + offsets, 1.0 / math.sqrt(tables))
    return design


def fit_unary(design: Tensor, target: Tensor, ridge: float) -> Tensor:
    kernel = design @ design.T
    kernel.diagonal().add_(ridge)
    return design.T @ torch.linalg.solve(kernel, target)


def coordinate_r2(prediction: Tensor, target: Tensor) -> float:
    residual = (prediction - target).square().sum()
    total = (target - target.mean(dim=0, keepdim=True)).square().sum().clamp_min(1e-12)
    return float((1.0 - residual / total).item())


def route_coordinates(coordinates: Tensor, groups: Tensor, orders: Tensor) -> tuple[Tensor, Tensor]:
    local = coordinates[:, groups]
    sorted_coordinates = local.argsort(dim=-1)
    matches = (sorted_coordinates.unsqueeze(-2) == orders.view(1, 1, 24, 4)).all(dim=-1)
    return matches.long().argmax(dim=-1), local


def relation_codes(query_route: Tensor, key_route: Tensor, query_index: Tensor, key_index: Tensor, relative: Tensor) -> Tensor:
    return relative[query_route[query_index], key_route[key_index]]


def categorical_product(codes: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    tables = codes.shape[-1]
    table = torch.arange(tables, device=codes.device)
    return weight[table, codes].sum(dim=-1) + bias


def categorical_transpose(codes: Tensor, values: Tensor, states: int) -> Tensor:
    samples, tables = codes.shape
    offsets = torch.arange(tables, device=codes.device).view(1, -1) * states
    flat = codes + offsets
    result = torch.zeros(tables * states, device=codes.device)
    result.scatter_add_(0, flat.reshape(-1), values.view(-1, 1).expand(samples, tables).reshape(-1))
    return result.view(tables, states)


def fit_decoder(codes: Tensor, target: Tensor, *, states: int = 24, ridge: float = 1e-3, iterations: int = 80) -> tuple[Tensor, Tensor]:
    tables = codes.shape[1]
    bias = target.mean()
    centered = target - bias
    rhs = categorical_transpose(codes, centered, states) / target.numel()
    weight = torch.zeros(tables, states, device=codes.device)
    residual = rhs.clone()
    direction = residual.clone()
    residual_norm = (residual * residual).sum()
    for _ in range(iterations):
        prediction = categorical_product(codes, direction, torch.zeros_like(bias))
        normal = categorical_transpose(codes, prediction, states) / target.numel() + ridge * direction
        step = residual_norm / (direction.mul(normal).sum().clamp_min(1e-20))
        weight.add_(direction, alpha=step)
        residual.sub_(normal, alpha=step)
        next_norm = (residual * residual).sum()
        if next_norm.sqrt() < 1e-6:
            break
        direction.mul_(next_norm / residual_norm).add_(residual)
        residual_norm = next_norm
    return weight, bias


def sample_pairs(count: int, objects: int, generator: torch.Generator, device: torch.device) -> tuple[Tensor, Tensor]:
    query = torch.randint(objects, (count,), generator=generator)
    key = torch.randint(objects, (count,), generator=generator)
    return query.to(device), key.to(device)


class UnaryRelationModel(nn.Module):
    def __init__(
        self,
        query_payload: Tensor,
        key_payload: Tensor,
        groups: Tensor,
        orders: Tensor,
        relative: Tensor,
        adjacent: Tensor,
        weight: Tensor,
        bias: Tensor,
        *,
        unary_tables: int,
        temperature: float,
    ) -> None:
        super().__init__()
        self.query_payload = nn.Parameter(query_payload)
        self.key_payload = nn.Parameter(key_payload)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias.reshape(()))
        self.register_buffer("groups", groups)
        self.register_buffer("orders", orders)
        self.register_buffer("relative", relative)
        self.register_buffer("adjacent", adjacent)
        self.unary_tables = unary_tables
        self.temperature = temperature

    def coordinates(self, codes: Tensor) -> tuple[Tensor, Tensor]:
        tables = torch.arange(self.unary_tables, device=codes.device)
        scale = 1.0 / math.sqrt(self.unary_tables)
        query = self.query_payload[tables, codes].sum(dim=1) * scale
        key = self.key_payload[tables, codes].sum(dim=1) * scale
        return query, key

    def score(self, query_local: Tensor, key_local: Tensor, query_route: Tensor, key_route: Tensor) -> Tensor:
        tables = torch.arange(query_route.shape[1], device=query_route.device)
        current_code = self.relative[query_route, key_route]
        current = self.weight[tables, current_code]
        correction = torch.zeros_like(current)
        query_order = self.orders[query_route]
        key_order = self.orders[key_route]
        for position in range(3):
            q_low = query_order[..., position : position + 1]
            q_high = query_order[..., position + 1 : position + 2]
            q_margin = query_local.gather(-1, q_high).squeeze(-1) - query_local.gather(-1, q_low).squeeze(-1)
            q_neighbor = self.adjacent[query_route, position]
            q_value = self.weight[tables, self.relative[q_neighbor, key_route]]
            q_gate = torch.sigmoid(q_margin / self.temperature)
            correction = correction + (q_gate - q_gate.detach()) * (q_value - current)

            k_low = key_order[..., position : position + 1]
            k_high = key_order[..., position + 1 : position + 2]
            k_margin = key_local.gather(-1, k_high).squeeze(-1) - key_local.gather(-1, k_low).squeeze(-1)
            k_neighbor = self.adjacent[key_route, position]
            k_value = self.weight[tables, self.relative[query_route, k_neighbor]]
            k_gate = torch.sigmoid(k_margin / self.temperature)
            correction = correction + (k_gate - k_gate.detach()) * (k_value - current)
        return (current + correction).sum(dim=-1) + self.bias

    def forward(self, unary_code: Tensor, query_index: Tensor, key_index: Tensor) -> Tensor:
        query, key = self.coordinates(unary_code)
        query_route, query_local = route_coordinates(query, self.groups, self.orders)
        key_route, key_local = route_coordinates(key, self.groups, self.orders)
        return self.score(
            query_local[query_index],
            key_local[key_index],
            query_route[query_index],
            key_route[key_index],
        )


@torch.no_grad()
def score_matrix(
    query: Tensor,
    key: Tensor,
    groups: Tensor,
    orders: Tensor,
    relative: Tensor,
    weight: Tensor,
    bias: Tensor,
    *,
    chunk: int = 32,
) -> Tensor:
    query_route, _ = route_coordinates(query, groups, orders)
    key_route, _ = route_coordinates(key, groups, orders)
    tables = torch.arange(groups.shape[0], device=query.device).view(1, 1, -1)
    rows = []
    for start in range(0, query.shape[0], chunk):
        codes = relative[query_route[start : start + chunk, None, :], key_route[None, :, :]]
        rows.append(weight[tables, codes].sum(dim=-1) + bias)
    return torch.cat(rows, dim=0)


def rankdata(values: Tensor) -> Tensor:
    order = values.argsort()
    rank = torch.empty_like(order, dtype=torch.float32)
    rank[order] = torch.arange(values.numel(), device=values.device, dtype=torch.float32)
    return rank


@torch.no_grad()
def metrics(prediction: Tensor, target: Tensor, values: Tensor) -> dict[str, float]:
    residual = (prediction - target).square().sum()
    total = (target - target.mean()).square().sum().clamp_min(1e-12)
    k = min(16, target.shape[1])
    teacher_top = target.topk(k, dim=-1).indices
    predicted_top = prediction.topk(k, dim=-1).indices
    overlap = (predicted_top.unsqueeze(-1) == teacher_top.unsqueeze(-2)).any(dim=-1).float().mean()
    top1 = (prediction.argmax(dim=-1) == target.argmax(dim=-1)).float().mean()
    pred_rank = rankdata(prediction.flatten())
    target_rank = rankdata(target.flatten())
    spearman = F.cosine_similarity(
        pred_rank - pred_rank.mean(), target_rank - target_rank.mean(), dim=0
    )
    teacher_value = values[teacher_top].mean(dim=1)
    predicted_value = values[predicted_top].mean(dim=1)
    retrieval_cosine = F.cosine_similarity(teacher_value, predicted_value, dim=-1).mean()
    return {
        "r2": float((1.0 - residual / total).item()),
        "top16": float(overlap.item()),
        "top1": float(top1.item()),
        "spearman": float(spearman.item()),
        "retrieval_cosine": float(retrieval_cosine.item()),
    }


def quantize_votes(weight: Tensor, mode: str) -> Tensor:
    scale = weight.abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
    if mode == "binary":
        return weight.sign() * scale
    if mode == "ternary":
        return (weight / scale).round().clamp(-1, 1) * scale
    raise ValueError(mode)


def fit_relation_for_coordinates(
    query: Tensor,
    key: Tensor,
    relation: Tensor,
    groups: Tensor,
    orders: Tensor,
    relative: Tensor,
    pair_query: Tensor,
    pair_key: Tensor,
) -> tuple[Tensor, Tensor]:
    query_route, _ = route_coordinates(query, groups, orders)
    key_route, _ = route_coordinates(key, groups, orders)
    codes = relation_codes(query_route, key_route, pair_query, pair_key, relative)
    target = relation[pair_query, pair_key]
    return fit_decoder(codes, target)


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
    anchors = make_unary_anchors(args.dimension, args.unary_tables, args.comparisons, args.seed + 211, device)
    train_codes = unary_codes(problem.train_x, anchors)
    test_codes = unary_codes(problem.test_x, anchors)
    rows = 1 << args.comparisons
    train_design = unary_design(train_codes, rows)
    test_design = unary_design(test_codes, rows)
    generator = torch.Generator().manual_seed(args.seed + 307)
    fit_query, fit_key = sample_pairs(args.fit_pairs, args.train_objects, generator, device)
    values = torch.randn(args.test_objects, args.value_dimension, generator=generator).to(device)

    oracle_weight, oracle_bias = fit_relation_for_coordinates(
        problem.train_q, problem.train_k, problem.train_relation, groups, orders, relative, fit_query, fit_key
    )
    oracle_prediction = score_matrix(
        problem.test_q, problem.test_k, groups, orders, relative, oracle_weight, oracle_bias
    )
    oracle_metrics = metrics(oracle_prediction, problem.test_relation, values)

    random_query_payload = torch.randn(args.unary_tables, rows, args.rank, device=device) * args.init_std
    random_key_payload = torch.randn(args.unary_tables, rows, args.rank, device=device) * args.init_std
    random_query = train_design @ random_query_payload.view(-1, args.rank)
    random_key = train_design @ random_key_payload.view(-1, args.rank)
    random_weight, random_bias = fit_relation_for_coordinates(
        random_query, random_key, problem.train_relation, groups, orders, relative, fit_query, fit_key
    )
    random_test_query = test_design @ random_query_payload.view(-1, args.rank)
    random_test_key = test_design @ random_key_payload.view(-1, args.rank)
    random_prediction = score_matrix(
        random_test_query, random_test_key, groups, orders, relative, random_weight, random_bias
    )
    random_metrics = metrics(random_prediction, problem.test_relation, values)

    supervised_query_flat = fit_unary(train_design, problem.train_q, args.unary_ridge)
    supervised_key_flat = fit_unary(train_design, problem.train_k, args.unary_ridge)
    supervised_query = train_design @ supervised_query_flat
    supervised_key = train_design @ supervised_key_flat
    supervised_weight, supervised_bias = fit_relation_for_coordinates(
        supervised_query, supervised_key, problem.train_relation, groups, orders, relative, fit_query, fit_key
    )
    supervised_test_query = test_design @ supervised_query_flat
    supervised_test_key = test_design @ supervised_key_flat
    supervised_prediction = score_matrix(
        supervised_test_query,
        supervised_test_key,
        groups,
        orders,
        relative,
        supervised_weight,
        supervised_bias,
    )
    supervised_metrics = metrics(supervised_prediction, problem.test_relation, values)

    model = UnaryRelationModel(
        random_query_payload.clone(),
        random_key_payload.clone(),
        groups,
        orders,
        relative,
        adjacent,
        random_weight.clone(),
        random_bias.clone(),
        unary_tables=args.unary_tables,
        temperature=args.temperature,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    train_generator = torch.Generator().manual_seed(args.seed + 401)
    for step in range(args.steps):
        query_index, key_index = sample_pairs(args.batch_size, args.train_objects, train_generator, device)
        prediction = model(train_codes, query_index, key_index)
        target = problem.train_relation[query_index, key_index]
        loss = F.mse_loss(prediction, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    supervised_finetune = UnaryRelationModel(
        supervised_query_flat.view(args.unary_tables, rows, args.rank).clone(),
        supervised_key_flat.view(args.unary_tables, rows, args.rank).clone(),
        groups,
        orders,
        relative,
        adjacent,
        supervised_weight.clone(),
        supervised_bias.clone(),
        unary_tables=args.unary_tables,
        temperature=args.temperature,
    ).to(device)
    finetune_optimizer = torch.optim.AdamW(
        supervised_finetune.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    finetune_generator = torch.Generator().manual_seed(args.seed + 503)
    for step in range(args.steps):
        query_index, key_index = sample_pairs(args.batch_size, args.train_objects, finetune_generator, device)
        prediction = supervised_finetune(train_codes, query_index, key_index)
        target = problem.train_relation[query_index, key_index]
        loss = F.mse_loss(prediction, target)
        finetune_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(supervised_finetune.parameters(), 1.0)
        finetune_optimizer.step()

    with torch.no_grad():
        learned_test_query, learned_test_key = model.coordinates(test_codes)
        learned_prediction = score_matrix(
            learned_test_query,
            learned_test_key,
            groups,
            orders,
            relative,
            model.weight,
            model.bias,
        )
        learned_metrics = metrics(learned_prediction, problem.test_relation, values)
        finetuned_test_query, finetuned_test_key = supervised_finetune.coordinates(test_codes)
        finetuned_prediction = score_matrix(
            finetuned_test_query,
            finetuned_test_key,
            groups,
            orders,
            relative,
            supervised_finetune.weight,
            supervised_finetune.bias,
        )
        finetuned_metrics = metrics(finetuned_prediction, problem.test_relation, values)
        binary_prediction = score_matrix(
            learned_test_query,
            learned_test_key,
            groups,
            orders,
            relative,
            quantize_votes(model.weight, "binary"),
            model.bias,
        )
        ternary_prediction = score_matrix(
            learned_test_query,
            learned_test_key,
            groups,
            orders,
            relative,
            quantize_votes(model.weight, "ternary"),
            model.bias,
        )
        binary_metrics = metrics(binary_prediction, problem.test_relation, values)
        ternary_metrics = metrics(ternary_prediction, problem.test_relation, values)

    denominator = max(oracle_metrics["top16"] - random_metrics["top16"], 1e-8)
    learned_recovery = (learned_metrics["top16"] - random_metrics["top16"]) / denominator
    supervised_recovery = (supervised_metrics["top16"] - random_metrics["top16"]) / denominator
    result: dict[str, object] = {
        "seed": args.seed,
        "rank": args.rank,
        "relation_tables": args.relation_tables,
        "unary_tables": args.unary_tables,
        "comparisons": args.comparisons,
        "steps": args.steps,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "coordinate_r2": {
            "query_train": coordinate_r2(supervised_query, problem.train_q),
            "query_test": coordinate_r2(supervised_test_query, problem.test_q),
            "key_train": coordinate_r2(supervised_key, problem.train_k),
            "key_test": coordinate_r2(supervised_test_key, problem.test_k),
        },
        "oracle": oracle_metrics,
        "random_fixed": random_metrics,
        "supervised_unary": supervised_metrics,
        "supervised_finetune": finetuned_metrics,
        "learned_end_to_end": learned_metrics,
        "binary_votes": binary_metrics,
        "ternary_votes": ternary_metrics,
        "supervised_recovery": supervised_recovery,
        "learned_recovery": learned_recovery,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    return result


def summarize(result_dir: Path, report: Path) -> dict[str, object]:
    rows = [json.loads(path.read_text()) for path in sorted(result_dir.glob("seed*.json"))]
    variants = ["oracle", "random_fixed", "supervised_unary", "learned_end_to_end", "binary_votes", "ternary_votes"]
    summary_rows = []
    for variant in variants:
        summary_rows.append(
            {
                "variant": variant,
                **{
                    metric: sum(float(row[variant][metric]) for row in rows) / len(rows)
                    for metric in ("r2", "top16", "top1", "spearman", "retrieval_cosine")
                },
            }
        )
    mean_supervised_recovery = sum(float(row["supervised_recovery"]) for row in rows) / len(rows)
    mean_learned_recovery = sum(float(row["learned_recovery"]) for row in rows) / len(rows)
    lookup = {row["variant"]: row for row in summary_rows}
    binary_retention = lookup["binary_votes"]["top16"] / max(lookup["learned_end_to_end"]["top16"], 1e-8)
    learned_gate = mean_learned_recovery >= 0.8
    lowbit_gate = binary_retention >= 0.9
    retrieval_gate = lookup["learned_end_to_end"]["top16"] > 0.5
    enter_wiki103 = learned_gate and lowbit_gate and retrieval_gate
    decision = {
        "seeds": len(rows),
        "mean_supervised_recovery": mean_supervised_recovery,
        "mean_learned_recovery": mean_learned_recovery,
        "binary_top16_retention": binary_retention,
        "learned_gate_passed": learned_gate,
        "lowbit_gate_passed": lowbit_gate,
        "associative_retrieval_gate_passed": retrieval_gate,
        "enter_wiki103": enter_wiki103,
        "next_stage": "wiki103" if enter_wiki103 else "stop",
    }
    with (result_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    (result_dir / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    lines = [
        "# Learned Unary Coxeter Attention Gate",
        "",
        "## Decision",
        "",
        "**ENTER Wiki103.**" if enter_wiki103 else "**STOP before Wiki103.**",
        "",
        f"- End-to-end recovery of oracle Top-16 improvement: `{mean_learned_recovery:.4f}` (gate: >= 0.8).",
        f"- Supervised unary recovery: `{mean_supervised_recovery:.4f}`.",
        f"- Binary vote Top-16 retention: `{binary_retention:.4f}` (gate: >= 0.9).",
        f"- Learned associative Top-16: `{lookup['learned_end_to_end']['top16']:.4f}` (gate: > 0.5).",
        "",
        "## Mean held-object metrics",
        "",
        "| Variant | R2 | Top-16 | Top-1 | Spearman | Retrieval cosine |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['variant']} | {row['r2']:.4f} | {row['top16']:.4f} | {row['top1']:.4f} | "
            f"{row['spearman']:.4f} | {row['retrieval_cosine']:.4f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "The supervised path separates unary PC-LUT representational capacity "
        "from relation-loss trainability. The end-to-end path starts from "
        "random unary payloads and receives only relation loss through "
        "adjacent-swap finite-difference STE. Wiki103 is permitted only when "
        "this trainable path recovers the prespecified fraction of oracle "
        "improvement and the learned scalar votes survive binary quantization.",
        "",
    ]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines))
    print(json.dumps(decision))
    return decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dimension", type=int, default=32)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--train-objects", type=int, default=512)
    parser.add_argument("--test-objects", type=int, default=256)
    parser.add_argument("--relation-tables", type=int, default=64)
    parser.add_argument("--unary-tables", type=int, default=64)
    parser.add_argument("--comparisons", type=int, default=5)
    parser.add_argument("--fit-pairs", type=int, default=32768)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--unary-ridge", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.05)
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
