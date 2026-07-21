from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from tropnn.tools.bilinear_retrieval_probe import make_problem, retrieval_metrics
from tropnn.tools.coxeter_relation_probe import PERMUTATIONS, LocalS4Router
from tropnn.tools.s4_cross_table_kernel_probe import ordinal_coordinates

TEACHERS = (
    "raw_bilinear",
    "ordinal_bilinear",
    "full_centered_rank",
    "full_comparison_xor",
    "full_a2_chamber",
)
VARIANTS: dict[str, tuple[str, tuple[int, ...]]] = {
    "global_centered_rank": ("local", (0,)),
    "global_comparison_xor": ("local", (1,)),
    "global_a2_chamber": ("local", (2,)),
    "global_comparison_plus_a2": ("local", (1, 2)),
    "global_all_native": ("local", (0, 1, 2)),
    "full_centered_rank_oracle": ("full", (0,)),
    "full_comparison_xor_oracle": ("full", (1,)),
    "full_a2_chamber_oracle": ("full", (2,)),
}


@dataclass(frozen=True)
class PairIndices:
    query: Tensor
    key: Tensor


@dataclass(frozen=True)
class NativeLayout:
    input_dim: int
    coordinate_count: Tensor
    edge_keys: tuple[tuple[int, int], ...]
    edge_occurrences: tuple[tuple[tuple[int, int, int], ...], ...]
    triple_keys: tuple[tuple[int, int, int], ...]
    triple_occurrences: tuple[tuple[tuple[int, int, int, int], ...], ...]


@dataclass(frozen=True)
class NativeRepresentation:
    centered_rank: Tensor
    comparison_root: Tensor
    a2_code: Tensor


@dataclass(frozen=True)
class KernelFit:
    weight: Tensor
    bias: Tensor
    ridge: float
    validation_r2: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test coordinate-native global rank, comparison-root/XOR, and A2 chamber kernels on frozen local-S4 routes."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run")
    run.add_argument("--teacher", choices=TEACHERS, required=True)
    run.add_argument("--input-dim", type=int, default=32)
    run.add_argument("--train-queries", type=int, default=2048)
    run.add_argument("--train-keys", type=int, default=2048)
    run.add_argument("--test-queries", type=int, default=256)
    run.add_argument("--test-keys", type=int, default=512)
    run.add_argument("--max-value", type=int, default=15)
    run.add_argument("--tables", type=int, default=16)
    run.add_argument("--fit-samples", type=int, default=65536)
    run.add_argument("--validation-samples", type=int, default=16384)
    run.add_argument("--pair-batch-size", type=int, default=4096)
    run.add_argument("--eval-query-batch", type=int, default=16)
    run.add_argument("--top-k", type=int, default=16)
    run.add_argument("--ridge-grid", default="0.0,0.0001,0.001,0.01,0.1")
    run.add_argument("--device", default="cuda")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--out-dir", type=Path, required=True)

    summarize = commands.add_parser("summarize")
    summarize.add_argument("--result-dir", type=Path, required=True)
    summarize.add_argument("--out-report", type=Path, required=True)
    return parser


def parse_ridge_grid(text: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    if not values or any(value < 0.0 for value in values):
        raise ValueError("ridge grid must contain nonnegative comma-separated values")
    return values


def stable_ranks(values: Tensor) -> Tensor:
    """Ranks in coordinate order, using coordinate index as the stable tie break."""
    order = torch.argsort(values, dim=-1, stable=True)
    ranks = torch.empty_like(values)
    positions = torch.arange(values.shape[-1], device=values.device, dtype=values.dtype)
    ranks.scatter_(-1, order, positions.expand_as(order))
    return ranks


def decode_local_ranks(routes: Tensor) -> Tensor:
    """Decode each S4 chamber into the ranks of its four anchor slots."""
    permutations = torch.tensor(PERMUTATIONS, device=routes.device, dtype=torch.long)
    order = permutations[routes]
    ranks = torch.empty_like(order)
    positions = torch.arange(4, device=routes.device).view(1, 1, 4)
    ranks.scatter_(-1, order, positions.expand_as(order))
    return ranks


def permutation_rank_s3(order: Tensor) -> Tensor:
    first = (order[..., 1:] < order[..., :1]).sum(dim=-1)
    second = (order[..., 2:] < order[..., 1:2]).sum(dim=-1)
    return 2 * first + second


def build_native_layout(anchors: Tensor, input_dim: int | None = None) -> NativeLayout:
    anchors_cpu = anchors.detach().cpu().to(torch.long)
    inferred_dim = int(anchors_cpu.max().item()) + 1
    input_dim = inferred_dim if input_dim is None else input_dim
    coordinate_count = torch.zeros(input_dim, dtype=torch.long)
    edges: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    triples: dict[tuple[int, int, int], list[tuple[int, int, int, int]]] = {}
    for table, row in enumerate(anchors_cpu.tolist()):
        for coordinate in row:
            coordinate_count[coordinate] += 1
        for left, right in itertools.combinations(range(4), 2):
            key = tuple(sorted((row[left], row[right])))
            edges.setdefault(key, []).append((table, left, right))
        for slots in itertools.combinations(range(4), 3):
            keyed_slots = sorted(((row[slot], slot) for slot in slots))
            key = tuple(coordinate for coordinate, _ in keyed_slots)
            canonical_slots = tuple(slot for _, slot in keyed_slots)
            triples.setdefault(key, []).append((table, *canonical_slots))
    edge_keys = tuple(sorted(edges))
    triple_keys = tuple(sorted(triples))
    return NativeLayout(
        input_dim=input_dim,
        coordinate_count=coordinate_count,
        edge_keys=edge_keys,
        edge_occurrences=tuple(tuple(edges[key]) for key in edge_keys),
        triple_keys=triple_keys,
        triple_occurrences=tuple(tuple(triples[key]) for key in triple_keys),
    )


def local_root_vertex_scatter(local_ranks: Tensor) -> Tensor:
    """Sum signed K4 roots at vertices; this is exactly 2 * centered rank."""
    result = torch.zeros_like(local_ranks, dtype=torch.float32)
    ranks = local_ranks.to(torch.float32)
    for left, right in itertools.combinations(range(4), 2):
        sign = torch.where(ranks[..., left] > ranks[..., right], 1.0, -1.0)
        result[..., left] += sign
        result[..., right] -= sign
    return result


def _normalise_nonzero(values: Tensor) -> Tensor:
    return values / values.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def build_local_representation(
    routes: Tensor,
    anchors: Tensor,
    input_dim: int | None = None,
) -> tuple[NativeRepresentation, dict[str, float | int]]:
    layout = build_native_layout(anchors, input_dim)
    ranks = decode_local_ranks(routes)
    objects = routes.shape[0]
    device = routes.device

    rank_sum = torch.zeros(objects, layout.input_dim, device=device)
    for table in range(anchors.shape[0]):
        for slot in range(4):
            rank_sum[:, int(anchors[table, slot])] += ranks[:, table, slot].to(torch.float32) - 1.5
    counts = layout.coordinate_count.to(device=device, dtype=torch.float32)
    covered = counts > 0
    centered_rank = rank_sum / counts.clamp_min(1.0)
    centered_rank[:, covered] -= centered_rank[:, covered].mean(dim=-1, keepdim=True)
    centered_rank[:, ~covered] = 0.0
    centered_rank = _normalise_nonzero(centered_rank)

    edge_votes = torch.zeros(objects, len(layout.edge_keys), device=device)
    edge_disagreements = torch.zeros((), device=device)
    edge_observations = 0
    anchors_cpu = anchors.detach().cpu()
    for edge_index, occurrences in enumerate(layout.edge_occurrences):
        low_coordinate, _ = layout.edge_keys[edge_index]
        for table, left, right in occurrences:
            if int(anchors_cpu[table, left]) == low_coordinate:
                low_slot, high_slot = left, right
            else:
                low_slot, high_slot = right, left
            sign = torch.where(ranks[:, table, low_slot] > ranks[:, table, high_slot], 1.0, -1.0)
            edge_votes[:, edge_index] += sign
        count = len(occurrences)
        edge_disagreements += ((count - edge_votes[:, edge_index].abs()) * 0.5).sum()
        edge_observations += objects * count
    comparison_root = torch.where(edge_votes >= 0.0, 1.0, -1.0)
    comparison_root /= math.sqrt(max(1, comparison_root.shape[1]))

    a2_votes = torch.zeros(objects, len(layout.triple_keys), 6, device=device)
    a2_disagreements = torch.zeros((), device=device)
    a2_observations = 0
    for triple_index, occurrences in enumerate(layout.triple_occurrences):
        for table, first, second, third in occurrences:
            triple_ranks = ranks[:, table, [first, second, third]]
            code = permutation_rank_s3(torch.argsort(triple_ranks, dim=-1, stable=True))
            a2_votes[:, triple_index].scatter_add_(
                1,
                code.unsqueeze(1),
                torch.ones(objects, 1, device=device),
            )
        count = len(occurrences)
        a2_disagreements += (count - a2_votes[:, triple_index].amax(dim=-1)).sum()
        a2_observations += objects * count
    a2_code = a2_votes.argmax(dim=-1).to(torch.uint8)

    metadata: dict[str, float | int] = {
        "covered_coordinates": int(covered.sum().item()),
        "coordinate_dimension": layout.input_dim,
        "unique_edges": len(layout.edge_keys),
        "possible_edges": math.comb(layout.input_dim, 2),
        "unique_triples": len(layout.triple_keys),
        "possible_triples": math.comb(layout.input_dim, 3),
        "edge_duplicate_disagreement_rate": float(edge_disagreements.item() / max(1, edge_observations)),
        "a2_duplicate_disagreement_rate": float(a2_disagreements.item() / max(1, a2_observations)),
    }
    return NativeRepresentation(centered_rank, comparison_root, a2_code), metadata


def full_comparison_features(ranks: Tensor) -> Tensor:
    pairs = torch.combinations(torch.arange(ranks.shape[1], device=ranks.device), r=2)
    result = torch.where(ranks[:, pairs[:, 0]] > ranks[:, pairs[:, 1]], 1.0, -1.0)
    return result / math.sqrt(result.shape[1])


def full_a2_codes(ranks: Tensor, object_batch: int = 128) -> Tensor:
    triples = torch.combinations(torch.arange(ranks.shape[1], device=ranks.device), r=3)
    chunks: list[Tensor] = []
    for start in range(0, ranks.shape[0], object_batch):
        selected = ranks[start : start + object_batch, triples]
        order = torch.argsort(selected, dim=-1, stable=True)
        chunks.append(permutation_rank_s3(order).to(torch.uint8))
    return torch.cat(chunks, dim=0)


def build_full_representation(values: Tensor) -> NativeRepresentation:
    ranks = stable_ranks(values)
    centered = ranks - 0.5 * (values.shape[-1] - 1)
    return NativeRepresentation(
        _normalise_nonzero(centered),
        full_comparison_features(ranks),
        full_a2_codes(ranks),
    )


def sample_pair_indices(query_count: int, key_count: int, samples: int, seed: int, device: torch.device) -> PairIndices:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    query = torch.randint(query_count, (samples,), generator=generator).to(device)
    key = torch.randint(key_count, (samples,), generator=generator).to(device)
    return PairIndices(query, key)


def pair_channels(
    query: NativeRepresentation,
    key: NativeRepresentation,
    indices: PairIndices,
    batch_size: int,
) -> Tensor:
    chunks: list[Tensor] = []
    for start in range(0, indices.query.shape[0], batch_size):
        qi = indices.query[start : start + batch_size]
        ki = indices.key[start : start + batch_size]
        rank = (query.centered_rank[qi] * key.centered_rank[ki]).sum(dim=-1)
        root = (query.comparison_root[qi] * key.comparison_root[ki]).sum(dim=-1)
        equal = (query.a2_code[qi] == key.a2_code[ki]).to(torch.float32).mean(dim=-1)
        a2 = (6.0 * equal - 1.0) / 5.0
        chunks.append(torch.stack((rank, root, a2), dim=-1))
    return torch.cat(chunks, dim=0)


def channel_matrices(
    query: NativeRepresentation,
    key: NativeRepresentation,
    query_batch: int,
) -> tuple[Tensor, Tensor, Tensor]:
    centered = query.centered_rank @ key.centered_rank.T
    root = query.comparison_root @ key.comparison_root.T
    a2_rows: list[Tensor] = []
    for start in range(0, query.a2_code.shape[0], query_batch):
        equal = (query.a2_code[start : start + query_batch, None, :] == key.a2_code[None, :, :]).to(torch.float32).mean(dim=-1)
        a2_rows.append((6.0 * equal - 1.0) / 5.0)
    return centered, root, torch.cat(a2_rows, dim=0)


def teacher_pair_target(
    teacher: str,
    raw_query: Tensor,
    raw_key: Tensor,
    full_query: NativeRepresentation,
    full_key: NativeRepresentation,
    relation: Tensor,
    indices: PairIndices,
    batch_size: int,
) -> Tensor:
    if teacher in {"raw_bilinear", "ordinal_bilinear"}:
        query = raw_query if teacher == "raw_bilinear" else ordinal_coordinates(raw_query)
        key = raw_key if teacher == "raw_bilinear" else ordinal_coordinates(raw_key)
        return ((query[indices.query] @ relation) * key[indices.key]).sum(dim=-1)
    channel = {
        "full_centered_rank": 0,
        "full_comparison_xor": 1,
        "full_a2_chamber": 2,
    }[teacher]
    return pair_channels(full_query, full_key, indices, batch_size)[:, channel]


def teacher_score_matrix(
    teacher: str,
    raw_query: Tensor,
    raw_key: Tensor,
    full_matrices: tuple[Tensor, Tensor, Tensor],
    relation: Tensor,
) -> Tensor:
    if teacher == "raw_bilinear":
        return raw_query @ relation @ raw_key.T
    if teacher == "ordinal_bilinear":
        return ordinal_coordinates(raw_query) @ relation @ ordinal_coordinates(raw_key).T
    channel = {
        "full_centered_rank": 0,
        "full_comparison_xor": 1,
        "full_a2_chamber": 2,
    }[teacher]
    return full_matrices[channel]


def r2_score(target: Tensor, prediction: Tensor) -> float:
    target64 = target.to(torch.float64)
    prediction64 = prediction.to(torch.float64)
    variance = (target64 - target64.mean()).square().mean()
    mse = (target64 - prediction64).square().mean()
    return float((1.0 - mse / variance.clamp_min(1e-30)).item())


def fit_kernel(
    fit_channels: Tensor,
    fit_target: Tensor,
    validation_channels: Tensor,
    validation_target: Tensor,
    columns: tuple[int, ...],
    ridge_grid: tuple[float, ...],
) -> KernelFit:
    fit_x = fit_channels[:, columns].to(torch.float64)
    valid_x = validation_channels[:, columns].to(torch.float64)
    fit_y = fit_target.to(torch.float64)
    candidates: list[KernelFit] = []
    for ridge in ridge_grid:
        design = torch.cat((fit_x, torch.ones_like(fit_x[:, :1])), dim=1)
        normal = design.T @ design / design.shape[0]
        penalty = torch.eye(design.shape[1], device=design.device, dtype=design.dtype) * ridge
        penalty[-1, -1] = 0.0
        rhs = design.T @ fit_y / design.shape[0]
        try:
            solution = torch.linalg.solve(normal + penalty, rhs)
        except torch.linalg.LinAlgError:
            solution = torch.linalg.lstsq(normal + penalty, rhs.unsqueeze(1)).solution.squeeze(1)
        prediction = valid_x @ solution[:-1] + solution[-1]
        candidates.append(KernelFit(solution[:-1], solution[-1], ridge, r2_score(validation_target, prediction)))
    return max(candidates, key=lambda candidate: candidate.validation_r2)


def predict_kernel(matrices: tuple[Tensor, Tensor, Tensor], columns: tuple[int, ...], fit: KernelFit) -> Tensor:
    prediction = torch.zeros_like(matrices[0]) + fit.bias.to(matrices[0].dtype)
    for weight, column in zip(fit.weight, columns, strict=True):
        prediction += weight.to(prediction.dtype) * matrices[column]
    return prediction


def kernel_metadata(
    name: str,
    columns: tuple[int, ...],
    fit: KernelFit,
    fit_channels: Tensor,
    fit_target: Tensor,
    validation_channels: Tensor,
    validation_target: Tensor,
    matrices: tuple[Tensor, Tensor, Tensor],
    target: Tensor,
    dimensions: tuple[int, int, int],
    top_k: int,
    seed: int,
) -> dict[str, object]:
    fit_prediction = fit_channels[:, columns] @ fit.weight.to(fit_channels.dtype) + fit.bias.to(fit_channels.dtype)
    validation_prediction = validation_channels[:, columns] @ fit.weight.to(validation_channels.dtype) + fit.bias.to(validation_channels.dtype)
    prediction = predict_kernel(matrices, columns, fit)
    return {
        "variant": name,
        "learned_parameters": len(columns) + 1,
        "learned_cross_table_embedding_parameters": 0,
        "kernel_channels": list(columns),
        "fixed_feature_dimension": sum(dimensions[column] for column in columns),
        "selected_ridge": fit.ridge,
        "weights": [float(value) for value in fit.weight.detach().cpu()],
        "bias": float(fit.bias.item()),
        "fit_r2": r2_score(fit_target, fit_prediction),
        "validation_r2": r2_score(validation_target, validation_prediction),
        **retrieval_metrics(target, prediction, top_k, seed + 601),
    }


def run(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    problem = make_problem(args)
    relation = problem.relation.to(device)
    train_query = problem.train_queries.to(device)
    train_key = problem.train_keys.to(device)
    test_query = problem.test_queries.to(device)
    test_key = problem.test_keys.to(device)
    router = LocalS4Router(args.input_dim, args.tables, args.seed).to(device)

    started = time.perf_counter()
    train_query_full = build_full_representation(train_query)
    train_key_full = build_full_representation(train_key)
    test_query_full = build_full_representation(test_query)
    test_key_full = build_full_representation(test_key)

    train_query_local, query_diagnostics = build_local_representation(router.route(train_query), router.anchors, args.input_dim)
    train_key_local, key_diagnostics = build_local_representation(router.route(train_key), router.anchors, args.input_dim)
    test_query_local, _ = build_local_representation(router.route(test_query), router.anchors, args.input_dim)
    test_key_local, _ = build_local_representation(router.route(test_key), router.anchors, args.input_dim)

    fit_indices = sample_pair_indices(args.train_queries, args.train_keys, args.fit_samples, args.seed + 2003, device)
    validation_indices = sample_pair_indices(args.train_queries, args.train_keys, args.validation_samples, args.seed + 2017, device)
    fit_target = teacher_pair_target(
        args.teacher,
        train_query,
        train_key,
        train_query_full,
        train_key_full,
        relation,
        fit_indices,
        args.pair_batch_size,
    )
    validation_target = teacher_pair_target(
        args.teacher,
        train_query,
        train_key,
        train_query_full,
        train_key_full,
        relation,
        validation_indices,
        args.pair_batch_size,
    )
    local_fit_channels = pair_channels(train_query_local, train_key_local, fit_indices, args.pair_batch_size)
    local_validation_channels = pair_channels(train_query_local, train_key_local, validation_indices, args.pair_batch_size)
    full_fit_channels = pair_channels(train_query_full, train_key_full, fit_indices, args.pair_batch_size)
    full_validation_channels = pair_channels(train_query_full, train_key_full, validation_indices, args.pair_batch_size)

    local_matrices = channel_matrices(test_query_local, test_key_local, args.eval_query_batch)
    full_matrices = channel_matrices(test_query_full, test_key_full, args.eval_query_batch)
    target = teacher_score_matrix(args.teacher, test_query, test_key, full_matrices, relation)
    ridge_grid = parse_ridge_grid(args.ridge_grid)
    local_layout = build_native_layout(router.anchors, args.input_dim)
    local_dimensions = (args.input_dim, len(local_layout.edge_keys), 6 * len(local_layout.triple_keys))
    full_dimensions = (args.input_dim, math.comb(args.input_dim, 2), 6 * math.comb(args.input_dim, 3))

    variants: list[dict[str, object]] = []
    for name, (source, columns) in VARIANTS.items():
        if source == "local":
            fit_channels = local_fit_channels
            validation_channels = local_validation_channels
            matrices = local_matrices
            dimensions = local_dimensions
        else:
            fit_channels = full_fit_channels
            validation_channels = full_validation_channels
            matrices = full_matrices
            dimensions = full_dimensions
        fit = fit_kernel(
            fit_channels,
            fit_target,
            validation_channels,
            validation_target,
            columns,
            ridge_grid,
        )
        variants.append(
            kernel_metadata(
                name,
                columns,
                fit,
                fit_channels,
                fit_target,
                validation_channels,
                validation_target,
                matrices,
                target,
                dimensions,
                args.top_k,
                args.seed,
            )
        )

    result = {
        "seed": args.seed,
        "teacher": args.teacher,
        "input_dim": args.input_dim,
        "tables": args.tables,
        "fit_samples": args.fit_samples,
        "validation_samples": args.validation_samples,
        "ridge_grid": ridge_grid,
        "same_pair_splits_across_variants": True,
        "route_anchor_groups": router.anchors.detach().cpu().tolist(),
        "semantics": {
            "global_centered_rank": "average local centered S4 ranks at their original coordinate labels, recenter, then cosine",
            "global_comparison_xor": "majority-merge canonical coordinate-edge signs; dot = 1 - 2 * Hamming / unique_edges",
            "global_a2_chamber": "mode-merge canonical coordinate triples into one of six S3 chambers; centered one-hot equality kernel",
            "braid_relation": "s1 s2 s1 = s2 s1 s2 holds in every legal A2 chamber and is therefore a constant, not a feature",
            "calibration": "ridge-selected affine scalar weights only; no learned table or chamber embedding",
        },
        "local_feature_dimensions": local_dimensions,
        "full_feature_dimensions": full_dimensions,
        "train_query_diagnostics": query_diagnostics,
        "train_key_diagnostics": key_diagnostics,
        "elapsed_seconds": time.perf_counter() - started,
        "variants": variants,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"seed{args.seed}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)


def mean_sem(values: list[float]) -> tuple[float, float]:
    mean = statistics.mean(values)
    sem = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return mean, sem


def summarize(args: argparse.Namespace) -> None:
    runs = [json.loads(path.read_text()) for path in sorted(args.result_dir.glob("**/seed*.json"))]
    if not runs:
        raise RuntimeError(f"no seed JSON files under {args.result_dir}")
    flat = [{"seed": run["seed"], "teacher": run["teacher"], **variant} for run in runs for variant in run["variants"]]
    fields = sorted({key for row in flat for key in row})
    with (args.result_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)

    aggregate: list[dict[str, object]] = []
    for teacher in TEACHERS:
        for variant in VARIANTS:
            members = [row for row in flat if row["teacher"] == teacher and row["variant"] == variant]
            if not members:
                continue
            row: dict[str, object] = {
                "teacher": teacher,
                "variant": variant,
                "seeds": len(members),
                "learned_parameters": members[0]["learned_parameters"],
                "fixed_feature_dimension": members[0]["fixed_feature_dimension"],
            }
            for metric in ("fit_r2", "validation_r2", "score_r2", "spearman", "topk_recall", "top1_accuracy"):
                mean, sem = mean_sem([float(member[metric]) for member in members])
                row[metric] = mean
                row[f"{metric}_sem"] = sem
            aggregate.append(row)
    with (args.result_dir / "aggregate.json").open("w") as handle:
        json.dump(aggregate, handle, indent=2)
        handle.write("\n")

    indexed = {(str(row["teacher"]), str(row["variant"])): row for row in aggregate}
    raw_all = indexed[("raw_bilinear", "global_all_native")]
    ordinal_all = indexed[("ordinal_bilinear", "global_all_native")]
    rank_all = indexed[("full_centered_rank", "global_all_native")]
    root_only = indexed[("full_comparison_xor", "global_comparison_xor")]
    root_plus_a2 = indexed[("full_comparison_xor", "global_comparison_plus_a2")]
    a2_all = indexed[("full_a2_chamber", "global_all_native")]
    diagnostic_by_seed = {int(run["seed"]): run["train_query_diagnostics"] for run in runs if run["teacher"] == "raw_bilinear"}
    diagnostics = list(diagnostic_by_seed.values())
    covered_mean = statistics.mean(float(item["covered_coordinates"]) for item in diagnostics)
    edge_mean = statistics.mean(float(item["unique_edges"]) for item in diagnostics)
    triple_mean = statistics.mean(float(item["unique_triples"]) for item in diagnostics)
    possible_edges = int(diagnostics[0]["possible_edges"])
    possible_triples = int(diagnostics[0]["possible_triples"])

    lines = [
        "# Native global ordinal kernels on frozen local-S4 routes",
        "",
        "This probe tests three coordinate-native kernels without a learned cross-table embedding: "
        "global centered ranks, canonical comparison roots (exact XOR/popcount similarity), and "
        "canonical A2/S3 triple chambers. Each kernel learns only an affine calibration on sampled pairs.",
        "",
        "The literal A2 braid equality is not used as a bit: it is true in every legal chamber. "
        "The six-way S3 chamber is the smallest nonconstant higher-order feature.",
        "",
        "## Main result",
        "",
        "These fixed isotropic kernels do not recover either random bilinear teacher. The strongest "
        f"three-channel native combination reaches held R2 {raw_all['score_r2']:.4f} raw and "
        f"{ordinal_all['score_r2']:.4f} ordinal, with Top-16 recall "
        f"{raw_all['topk_recall']:.4f}/{ordinal_all['topk_recall']:.4f}; random Top-16 expectation is "
        "0.03125. Even the full-coordinate rank, root, and A2 kernels remain near zero on these teachers. "
        "Therefore sparse chart coverage is not the explanation: an invariant similarity kernel cannot "
        "express the learned query/key-specific geometry of a random asymmetric relation.",
        "",
        "This contrasts with the matched learned rank-12 controls, whose held R2 is 0.3006/0.3097 and "
        "Top-16 is 0.2196/0.2314 raw/ordinal. Low rank was useful there as a separable parameterization of "
        "a learned relation, not as an intrinsic coordinate system of ordinal space.",
        "",
        "The matched ordinal teachers provide the positive control. The three-channel local construction "
        f"recovers held R2 {rank_all['score_r2']:.4f} for centered-rank similarity and "
        f"{a2_all['score_r2']:.4f} for full-A2 similarity. On the full-comparison teacher, roots alone give "
        f"{root_only['score_r2']:.4f}; adding A2 changes this to {root_plus_a2['score_r2']:.4f}. "
        "Thus the native global merge is valid, but the A2 lift adds little beyond its constituent pair "
        "comparisons at this chart coverage.",
        "",
        "## Protocol",
        "",
        f"Frozen local-S4 routing uses {runs[0]['tables']} K4 charts over D={runs[0]['input_dim']}. "
        f"There are {runs[0]['fit_samples']:,} fit pairs, {runs[0]['validation_samples']:,} validation "
        "pairs, 256 held queries, 512 held keys, and three seeds. All variants within a seed share objects, "
        "anchors, pair splits, and teacher. A one-channel kernel learns only scale plus bias; combined kernels "
        "learn one scalar per channel plus bias. No variant learns a table, chamber, Q, K, or cross-table "
        "embedding.",
        "",
    ]
    for teacher in TEACHERS:
        rows = [row for row in aggregate if row["teacher"] == teacher]
        if not rows:
            continue
        lines.extend(
            [
                f"## {teacher}",
                "",
                "| Variant | Params | Fixed dim | Validation R2 | Test R2 | Top-16 recall | Spearman |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            lines.append(
                f"| {row['variant']} | {row['learned_parameters']} | {row['fixed_feature_dimension']} | "
                f"{row['validation_r2']:.4f} +/- {row['validation_r2_sem']:.4f} | "
                f"{row['score_r2']:.4f} +/- {row['score_r2_sem']:.4f} | "
                f"{row['topk_recall']:.4f} +/- {row['topk_recall_sem']:.4f} | "
                f"{row['spearman']:.4f} +/- {row['spearman_sem']:.4f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Structural audit",
            "",
            f"Across seeds, the local charts cover on average {covered_mean:.1f}/32 coordinates, "
            f"{edge_mean:.1f}/{possible_edges} canonical edges ({100.0 * edge_mean / possible_edges:.1f}%), "
            f"and {triple_mean:.1f}/{possible_triples} canonical triples "
            f"({100.0 * triple_mean / possible_triples:.2f}%). Duplicate-chart disagreement caused by "
            "local stable tie breaking is below 0.34% for edges and below 0.10% for A2 triples.",
            "",
            "Comparison-root bits remain separate. Summing the six signed K4 roots at their four "
            "vertices would equal two times the local centered-rank vector, so that construction "
            "would not be an independent kernel.",
            "",
            "Raw JSON, per-run CSV rows, exact anchors, fitted scalar weights, duplicate-chart "
            "disagreement diagnostics, and timing are stored beside this report.",
            "",
            "## Artifacts",
            "",
            f"- Results: `{args.result_dir}`",
            "- Launcher: `scripts/run_tropnn_s4_native_global_kernel_4gpu.sh`",
            "- Probe: `python/src/tropnn/tools/s4_native_global_kernel_probe.py`",
            "- Tests: `python/src/tropnn/tests/test_s4_native_global_kernel_probe.py`",
            "- Learned rank-12 comparison: `python/report/s4_ordinal_structure_ablation.md`",
            "",
        ]
    )
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text("\n".join(lines))


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run":
        run(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
