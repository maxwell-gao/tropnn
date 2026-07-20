from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from tropnn.tools.bilinear_retrieval_probe import make_problem, retrieval_metrics
from tropnn.tools.coxeter_relation_probe import LocalS4Router, categorical_prediction, r2_score, ridge_cg

S4_ORDER = 24
DECODERS = ("same_table_full", "global_rank", "sparse_cross_table", "dense_w_diagnostic")
TEACHERS = ("raw_bilinear", "ordinal_bilinear")


@dataclass(frozen=True)
class PairSplit:
    query_index: Tensor
    key_index: Tensor
    target: Tensor


@dataclass(frozen=True)
class CategoricalFit:
    edges: Tensor
    coefficient: Tensor
    bias: Tensor
    iterations: int
    relative_residual: float
    ridge: float


@dataclass(frozen=True)
class TowerFit:
    query_factor: Tensor
    key_factor: Tensor
    bias: Tensor
    target_scale: Tensor
    validation_r2: float
    restart: int
    rounds: int
    cg_iterations: int
    cg_relative_residual: float
    ridge: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Matched-budget same-table, global-rank, sparse-cross-table, and dense-W decoders on frozen local-S4 routes."
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
    run.add_argument("--rank", type=int, default=12)
    run.add_argument("--screen-samples", type=int, default=32768)
    run.add_argument("--fit-samples", type=int, default=65536)
    run.add_argument("--validation-samples", type=int, default=16384)
    run.add_argument("--top-k", type=int, default=16)
    run.add_argument("--ridge-grid", default="0.001,0.01,0.1,1.0,10.0")
    run.add_argument("--cg-iterations", type=int, default=96)
    run.add_argument("--cg-tolerance", type=float, default=1e-6)
    run.add_argument("--als-rounds", type=int, default=4)
    run.add_argument("--als-restarts", type=int, default=2)
    run.add_argument("--als-cg-iterations", type=int, default=64)
    run.add_argument("--batch-size", type=int, default=4096)
    run.add_argument("--eval-query-batch", type=int, default=32)
    run.add_argument("--device", default="cuda")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--out-dir", type=Path, required=True)

    summarize = commands.add_parser("summarize")
    summarize.add_argument("--result-dir", type=Path, required=True)
    summarize.add_argument("--out-report", type=Path, required=True)
    return parser


def ordinal_coordinates(values: Tensor) -> Tensor:
    """Return centered, unit-norm coordinate ranks with a stable tie break."""
    order = torch.argsort(values, dim=-1, stable=True)
    ranks = torch.empty_like(values)
    positions = torch.arange(values.shape[-1], device=values.device, dtype=values.dtype)
    ranks.scatter_(-1, order, positions.expand_as(order))
    ranks = ranks - 0.5 * (values.shape[-1] - 1)
    return ranks / ranks.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def parse_ridge_grid(text: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    if not values or any(value < 0.0 for value in values):
        raise ValueError("ridge grid must contain nonnegative comma-separated values")
    return values


def teacher_coordinates(values: Tensor, teacher: str) -> Tensor:
    if teacher == "raw_bilinear":
        return values
    if teacher == "ordinal_bilinear":
        return ordinal_coordinates(values)
    raise ValueError(f"unsupported teacher {teacher!r}")


def teacher_score_matrix(query: Tensor, key: Tensor, relation: Tensor) -> Tensor:
    return query @ relation @ key.T


def sample_pair_split(
    query: Tensor,
    key: Tensor,
    relation: Tensor,
    samples: int,
    seed: int,
) -> PairSplit:
    generator = torch.Generator(device=query.device).manual_seed(seed)
    query_index = torch.randint(query.shape[0], (samples,), generator=generator, device=query.device)
    key_index = torch.randint(key.shape[0], (samples,), generator=generator, device=key.device)
    target = ((query[query_index] @ relation) * key[key_index]).sum(dim=-1)
    return PairSplit(query_index, key_index, target)


def all_table_edges(tables: int, *, diagonal: bool | None = None, device: torch.device | None = None) -> Tensor:
    edges = torch.cartesian_prod(torch.arange(tables), torch.arange(tables))
    if diagonal is True:
        edges = edges[edges[:, 0] == edges[:, 1]]
    elif diagonal is False:
        edges = edges[edges[:, 0] != edges[:, 1]]
    return edges.to(device=device)


def edge_codes(
    query_route: Tensor,
    key_route: Tensor,
    query_index: Tensor,
    key_index: Tensor,
    edges: Tensor,
) -> Tensor:
    query_code = query_route[query_index][:, edges[:, 0]]
    key_code = key_route[key_index][:, edges[:, 1]]
    return query_code * S4_ORDER + key_code


def categorical_edge_prediction(codes: Tensor, coefficient: Tensor, bias: Tensor) -> Tensor:
    return categorical_prediction(codes, coefficient.reshape(-1), bias, codes.shape[1])


def fit_categorical_edges(
    query_route: Tensor,
    key_route: Tensor,
    split: PairSplit,
    edges: Tensor,
    *,
    ridge: float,
    iterations: int,
    tolerance: float,
    batch_size: int,
) -> CategoricalFit:
    codes = edge_codes(query_route, key_route, split.query_index, split.key_index, edges)
    coefficient, bias, completed, relative_residual = ridge_cg(
        codes,
        split.target,
        S4_ORDER * S4_ORDER,
        ridge,
        iterations,
        tolerance,
        batch_size,
    )
    return CategoricalFit(
        edges=edges,
        coefficient=coefficient.view(edges.shape[0], S4_ORDER * S4_ORDER),
        bias=bias,
        iterations=completed,
        relative_residual=relative_residual,
        ridge=ridge,
    )


def select_categorical_fit(
    query_route: Tensor,
    key_route: Tensor,
    fit_split: PairSplit,
    validation: PairSplit,
    edges: Tensor,
    *,
    ridges: tuple[float, ...],
    iterations: int,
    tolerance: float,
    batch_size: int,
) -> CategoricalFit:
    validation_codes = edge_codes(
        query_route,
        key_route,
        validation.query_index,
        validation.key_index,
        edges,
    )
    best: CategoricalFit | None = None
    best_r2 = float("-inf")
    for ridge in ridges:
        candidate = fit_categorical_edges(
            query_route,
            key_route,
            fit_split,
            edges,
            ridge=ridge,
            iterations=iterations,
            tolerance=tolerance,
            batch_size=batch_size,
        )
        prediction = categorical_edge_prediction(validation_codes, candidate.coefficient, candidate.bias)
        validation_r2 = r2_score(validation.target, prediction)
        if validation_r2 > best_r2:
            best = candidate
            best_r2 = validation_r2
    assert best is not None
    return best


def screen_sparse_edges(
    query_route: Tensor,
    key_route: Tensor,
    split: PairSplit,
    budget: int,
) -> tuple[Tensor, Tensor]:
    """Select target-conditioned off-diagonal blocks on an isolated screen split."""
    candidates = all_table_edges(query_route.shape[1], diagonal=False, device=query_route.device)
    target = split.target.float()
    centered = target - target.mean()
    total = centered.square().sum().clamp_min(1e-12)
    scores = torch.empty(candidates.shape[0], device=query_route.device)
    for index, edge in enumerate(candidates):
        code = edge_codes(query_route, key_route, split.query_index, split.key_index, edge.view(1, 2)).squeeze(-1)
        count = torch.bincount(code, minlength=S4_ORDER * S4_ORDER).float()
        sums = torch.zeros(S4_ORDER * S4_ORDER, device=query_route.device)
        sums.scatter_add_(0, code, target)
        mean = sums / count.clamp_min(1.0)
        prediction = mean[code]
        scores[index] = 1.0 - (target - prediction).square().sum() / total
    selected = torch.topk(scores, k=budget).indices
    return candidates[selected], scores[selected]


def tower_embeddings(routes: Tensor, factor: Tensor) -> Tensor:
    tables = routes.shape[1]
    table = torch.arange(tables, device=routes.device).view(1, tables)
    return factor[table, routes].sum(dim=1)


def tower_pair_prediction(
    query_route: Tensor,
    key_route: Tensor,
    split: PairSplit,
    query_factor: Tensor,
    key_factor: Tensor,
    bias: Tensor,
    target_scale: Tensor,
) -> Tensor:
    query = tower_embeddings(query_route, query_factor)[split.query_index]
    key = tower_embeddings(key_route, key_factor)[split.key_index]
    normalized = (query * key).sum(dim=-1) / query_route.shape[1] + bias
    return normalized * target_scale


def tower_score_matrix(
    query_route: Tensor,
    key_route: Tensor,
    fit: TowerFit,
) -> Tensor:
    query = tower_embeddings(query_route, fit.query_factor)
    key = tower_embeddings(key_route, fit.key_factor)
    return (query @ key.T / query_route.shape[1] + fit.bias) * fit.target_scale


def tower_rhs(
    variable_route: Tensor,
    variable_index: Tensor,
    fixed_embedding: Tensor,
    target: Tensor,
    *,
    states: int,
    tables: int,
    rank: int,
    batch_size: int,
) -> Tensor:
    result = torch.zeros(tables * states, rank, device=variable_route.device)
    table_offset = torch.arange(tables, device=variable_route.device).view(1, -1) * states
    scale = 1.0 / tables
    for start in range(0, target.shape[0], batch_size):
        stop = min(start + batch_size, target.shape[0])
        code = variable_route[variable_index[start:stop]] + table_offset
        value = target[start:stop, None] * fixed_embedding[start:stop] * scale
        result.index_add_(0, code.reshape(-1), value[:, None, :].expand(-1, tables, -1).reshape(-1, rank))
    return result / target.shape[0]


def tower_normal_matvec(
    variable_route: Tensor,
    variable_index: Tensor,
    fixed_embedding: Tensor,
    vector: Tensor,
    *,
    ridge: float,
    states: int,
    tables: int,
    batch_size: int,
) -> Tensor:
    rank = vector.shape[-1]
    flat = vector.reshape(tables * states, rank)
    result = ridge * flat
    table = torch.arange(tables, device=variable_route.device).view(1, tables)
    table_offset = table * states
    scale = 1.0 / tables
    for start in range(0, variable_index.shape[0], batch_size):
        stop = min(start + batch_size, variable_index.shape[0])
        route = variable_route[variable_index[start:stop]]
        variable_embedding = vector[table, route].sum(dim=1)
        prediction = (variable_embedding * fixed_embedding[start:stop]).sum(dim=-1) * scale
        code = route + table_offset
        value = prediction[:, None] * fixed_embedding[start:stop] * scale
        result.index_add_(0, code.reshape(-1), value[:, None, :].expand(-1, tables, -1).reshape(-1, rank))
    return (result / variable_index.shape[0]).reshape_as(vector)


def solve_tower_side(
    variable_route: Tensor,
    variable_index: Tensor,
    fixed_embedding: Tensor,
    target: Tensor,
    initial: Tensor,
    *,
    ridge: float,
    iterations: int,
    tolerance: float,
    batch_size: int,
) -> tuple[Tensor, int, float]:
    tables, states, rank = initial.shape
    rhs = tower_rhs(
        variable_route,
        variable_index,
        fixed_embedding,
        target,
        states=states,
        tables=tables,
        rank=rank,
        batch_size=batch_size,
    ).reshape_as(initial)
    solution = initial.clone()
    image = tower_normal_matvec(
        variable_route,
        variable_index,
        fixed_embedding,
        solution,
        ridge=ridge,
        states=states,
        tables=tables,
        batch_size=batch_size,
    )
    residual = rhs - image
    direction = residual.clone()
    residual_squared = torch.dot(residual.reshape(-1), residual.reshape(-1))
    initial_norm = residual_squared.sqrt().clamp_min(1e-12)
    relative_residual = 1.0
    completed = 0
    for completed in range(1, iterations + 1):
        image = tower_normal_matvec(
            variable_route,
            variable_index,
            fixed_embedding,
            direction,
            ridge=ridge,
            states=states,
            tables=tables,
            batch_size=batch_size,
        )
        denominator = torch.dot(direction.reshape(-1), image.reshape(-1)).clamp_min(1e-20)
        step = residual_squared / denominator
        solution += step * direction
        residual -= step * image
        next_squared = torch.dot(residual.reshape(-1), residual.reshape(-1))
        relative_residual = float((next_squared.sqrt() / initial_norm).item())
        if relative_residual <= tolerance:
            break
        direction = residual + (next_squared / residual_squared.clamp_min(1e-30)) * direction
        residual_squared = next_squared
    return solution, completed, relative_residual


def balance_tower_factors(query_factor: Tensor, key_factor: Tensor) -> tuple[Tensor, Tensor]:
    query_norm = query_factor.square().sum(dim=(0, 1)).sqrt().clamp_min(1e-12)
    key_norm = key_factor.square().sum(dim=(0, 1)).sqrt().clamp_min(1e-12)
    scale = (key_norm / query_norm).sqrt()
    return query_factor * scale, key_factor / scale


def fit_global_tower(
    query_route: Tensor,
    key_route: Tensor,
    fit: PairSplit,
    validation: PairSplit,
    *,
    rank: int,
    ridge: float,
    rounds: int,
    restarts: int,
    cg_iterations: int,
    tolerance: float,
    batch_size: int,
    seed: int,
) -> TowerFit:
    target_mean = fit.target.mean()
    target_scale = fit.target.std(unbiased=False).clamp_min(1e-6)
    normalized_target = (fit.target - target_mean) / target_scale
    best: TowerFit | None = None
    tables = query_route.shape[1]
    generator = torch.Generator(device=query_route.device).manual_seed(seed)
    for restart in range(restarts):
        query_factor = torch.zeros(tables, S4_ORDER, rank, device=query_route.device)
        key_factor = torch.randn(
            tables,
            S4_ORDER,
            rank,
            generator=generator,
            device=query_route.device,
        ) / math.sqrt(tables * rank)
        bias = target_mean / target_scale
        completed = 0
        relative_residual = 1.0
        for _ in range(rounds):
            key_embedding = tower_embeddings(key_route, key_factor)[fit.key_index]
            query_factor, completed, relative_residual = solve_tower_side(
                query_route,
                fit.query_index,
                key_embedding,
                normalized_target - bias,
                query_factor,
                ridge=ridge,
                iterations=cg_iterations,
                tolerance=tolerance,
                batch_size=batch_size,
            )
            query_embedding = tower_embeddings(query_route, query_factor)[fit.query_index]
            key_factor, completed, relative_residual = solve_tower_side(
                key_route,
                fit.key_index,
                query_embedding,
                normalized_target - bias,
                key_factor,
                ridge=ridge,
                iterations=cg_iterations,
                tolerance=tolerance,
                batch_size=batch_size,
            )
            query_factor, key_factor = balance_tower_factors(query_factor, key_factor)
            prediction = (tower_embeddings(query_route, query_factor)[fit.query_index] * tower_embeddings(key_route, key_factor)[fit.key_index]).sum(
                dim=-1
            ) / tables
            bias = (normalized_target - prediction).mean()
        validation_prediction = tower_pair_prediction(
            query_route,
            key_route,
            validation,
            query_factor,
            key_factor,
            bias,
            target_scale,
        )
        candidate = TowerFit(
            query_factor=query_factor,
            key_factor=key_factor,
            bias=bias,
            target_scale=target_scale,
            validation_r2=r2_score(validation.target, validation_prediction),
            restart=restart,
            rounds=rounds,
            cg_iterations=completed,
            cg_relative_residual=relative_residual,
            ridge=ridge,
        )
        if best is None or candidate.validation_r2 > best.validation_r2:
            best = candidate
    assert best is not None
    return best


def categorical_score_matrix(
    query_route: Tensor,
    key_route: Tensor,
    fit: CategoricalFit,
    *,
    query_batch: int,
) -> Tensor:
    edge = torch.arange(fit.edges.shape[0], device=query_route.device).view(1, 1, -1)
    rows: list[Tensor] = []
    for start in range(0, query_route.shape[0], query_batch):
        query = query_route[start : start + query_batch]
        query_code = query[:, None, fit.edges[:, 0]]
        key_code = key_route[None, :, fit.edges[:, 1]]
        code = query_code * S4_ORDER + key_code
        score = fit.coefficient[edge, code].sum(dim=-1) / math.sqrt(fit.edges.shape[0]) + fit.bias
        rows.append(score)
    return torch.cat(rows, dim=0)


def dense_w_spectrum(fit: CategoricalFit, tables: int, rank: int) -> dict[str, object]:
    expected = all_table_edges(tables, device=fit.edges.device)
    if not torch.equal(fit.edges, expected):
        raise ValueError("dense W spectrum requires lexicographically ordered all-table edges")
    matrix = fit.coefficient.view(tables, tables, S4_ORDER, S4_ORDER).permute(0, 2, 1, 3).reshape(tables * S4_ORDER, tables * S4_ORDER)
    singular = torch.linalg.svdvals(matrix.float())
    energy = singular.square()
    total = energy.sum().clamp_min(1e-20)
    effective_rank = float(torch.exp(-(energy / total * (energy / total).clamp_min(1e-20).log()).sum()).item())
    return {
        "top_rank_energy": float((energy[:rank].sum() / total).item()),
        "top_4_energy": float((energy[:4].sum() / total).item()),
        "top_8_energy": float((energy[:8].sum() / total).item()),
        "top_16_energy": float((energy[:16].sum() / total).item()),
        "effective_rank": effective_rank,
        "leading_singular_values": [float(value) for value in singular[: min(32, singular.numel())].tolist()],
    }


def categorical_metadata(
    name: str,
    fit: CategoricalFit,
    train_query_route: Tensor,
    train_key_route: Tensor,
    fit_split: PairSplit,
    validation: PairSplit,
) -> dict[str, object]:
    fit_codes = edge_codes(
        train_query_route,
        train_key_route,
        fit_split.query_index,
        fit_split.key_index,
        fit.edges,
    )
    validation_codes = edge_codes(
        train_query_route,
        train_key_route,
        validation.query_index,
        validation.key_index,
        fit.edges,
    )
    fit_prediction = categorical_edge_prediction(fit_codes, fit.coefficient, fit.bias)
    validation_prediction = categorical_edge_prediction(validation_codes, fit.coefficient, fit.bias)
    return {
        "decoder": name,
        "parameters": fit.coefficient.numel() + 1,
        "payload_parameters": fit.coefficient.numel(),
        "object_lut_reads_per_object": 0,
        "pair_payload_reads": int(fit.edges.shape[0]),
        "pair_dot_width": 0,
        "solver": "ridge_cg",
        "solver_iterations": fit.iterations,
        "solver_relative_residual": fit.relative_residual,
        "selected_ridge": fit.ridge,
        "uniform_fit_r2": r2_score(fit_split.target, fit_prediction),
        "uniform_validation_r2": r2_score(validation.target, validation_prediction),
        "edges": fit.edges.detach().cpu().tolist(),
    }


def run(args: argparse.Namespace) -> None:
    expected_rank_numerator = args.tables * S4_ORDER * S4_ORDER
    expected_rank_denominator = 2 * args.tables * S4_ORDER
    matched_rank = expected_rank_numerator // expected_rank_denominator
    if expected_rank_numerator % expected_rank_denominator or args.rank != matched_rank:
        raise ValueError(f"strict parameter matching requires rank={matched_rank} for {args.tables} tables and {S4_ORDER} states, got {args.rank}")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    ridge_grid = parse_ridge_grid(args.ridge_grid)
    problem = make_problem(args)
    relation = problem.relation.to(device)
    train_queries = problem.train_queries.to(device)
    train_keys = problem.train_keys.to(device)
    test_queries = problem.test_queries.to(device)
    test_keys = problem.test_keys.to(device)
    train_query_teacher = teacher_coordinates(train_queries, args.teacher)
    train_key_teacher = teacher_coordinates(train_keys, args.teacher)
    test_query_teacher = teacher_coordinates(test_queries, args.teacher)
    test_key_teacher = teacher_coordinates(test_keys, args.teacher)

    router = LocalS4Router(args.input_dim, args.tables, args.seed).to(device)
    train_query_route = router.route(train_queries)
    train_key_route = router.route(train_keys)
    test_query_route = router.route(test_queries)
    test_key_route = router.route(test_keys)
    screen = sample_pair_split(
        train_query_teacher,
        train_key_teacher,
        relation,
        args.screen_samples,
        args.seed + 1999,
    )
    fit_split = sample_pair_split(
        train_query_teacher,
        train_key_teacher,
        relation,
        args.fit_samples,
        args.seed + 2003,
    )
    validation = sample_pair_split(
        train_query_teacher,
        train_key_teacher,
        relation,
        args.validation_samples,
        args.seed + 2017,
    )
    test_target = teacher_score_matrix(test_query_teacher, test_key_teacher, relation)

    started = time.perf_counter()
    same_edges = all_table_edges(args.tables, diagonal=True, device=device)
    same_fit = select_categorical_fit(
        train_query_route,
        train_key_route,
        fit_split,
        validation,
        same_edges,
        ridges=ridge_grid,
        iterations=args.cg_iterations,
        tolerance=args.cg_tolerance,
        batch_size=args.batch_size,
    )
    sparse_edges, sparse_screen_scores = screen_sparse_edges(
        train_query_route,
        train_key_route,
        screen,
        args.tables,
    )
    sparse_fit = select_categorical_fit(
        train_query_route,
        train_key_route,
        fit_split,
        validation,
        sparse_edges,
        ridges=ridge_grid,
        iterations=args.cg_iterations,
        tolerance=args.cg_tolerance,
        batch_size=args.batch_size,
    )
    global_candidates = [
        fit_global_tower(
            train_query_route,
            train_key_route,
            fit_split,
            validation,
            rank=args.rank,
            ridge=ridge,
            rounds=args.als_rounds,
            restarts=args.als_restarts,
            cg_iterations=args.als_cg_iterations,
            tolerance=args.cg_tolerance,
            batch_size=args.batch_size,
            seed=args.seed + 3011,
        )
        for ridge in ridge_grid
    ]
    global_fit = max(
        global_candidates,
        key=lambda candidate: candidate.validation_r2,
    )
    dense_edges = all_table_edges(args.tables, device=device)
    dense_fit = select_categorical_fit(
        train_query_route,
        train_key_route,
        fit_split,
        validation,
        dense_edges,
        ridges=ridge_grid,
        iterations=args.cg_iterations,
        tolerance=args.cg_tolerance,
        batch_size=args.batch_size,
    )

    variants: list[dict[str, object]] = []
    for name, fitted in (("same_table_full", same_fit), ("sparse_cross_table", sparse_fit), ("dense_w_diagnostic", dense_fit)):
        metadata = categorical_metadata(
            name,
            fitted,
            train_query_route,
            train_key_route,
            fit_split,
            validation,
        )
        prediction = categorical_score_matrix(
            test_query_route,
            test_key_route,
            fitted,
            query_batch=args.eval_query_batch,
        )
        metadata.update(retrieval_metrics(test_target, prediction, args.top_k, args.seed + 601))
        if name == "sparse_cross_table":
            metadata["edge_selection"] = "offline target-conditioned cell-mean screen on disjoint screen split"
            metadata["screen_edge_r2_mean"] = float(sparse_screen_scores.mean().item())
            metadata["screen_edge_r2_min"] = float(sparse_screen_scores.min().item())
            metadata["screen_edge_r2_max"] = float(sparse_screen_scores.max().item())
        if name == "dense_w_diagnostic":
            metadata["diagnostic_only"] = True
            metadata["spectrum"] = dense_w_spectrum(fitted, args.tables, args.rank)
        variants.append(metadata)

    global_validation_prediction = tower_pair_prediction(
        train_query_route,
        train_key_route,
        validation,
        global_fit.query_factor,
        global_fit.key_factor,
        global_fit.bias,
        global_fit.target_scale,
    )
    global_fit_prediction = tower_pair_prediction(
        train_query_route,
        train_key_route,
        fit_split,
        global_fit.query_factor,
        global_fit.key_factor,
        global_fit.bias,
        global_fit.target_scale,
    )
    global_prediction = tower_score_matrix(test_query_route, test_key_route, global_fit)
    global_metadata: dict[str, object] = {
        "decoder": "global_rank",
        "parameters": global_fit.query_factor.numel() + global_fit.key_factor.numel() + 1,
        "payload_parameters": global_fit.query_factor.numel() + global_fit.key_factor.numel(),
        "object_lut_reads_per_object": args.tables,
        "pair_payload_reads": 0,
        "pair_dot_width": args.rank,
        "solver": "alternating_ridge_cg",
        "solver_iterations": global_fit.cg_iterations,
        "solver_relative_residual": global_fit.cg_relative_residual,
        "solver_rounds": global_fit.rounds,
        "selected_restart": global_fit.restart,
        "selected_ridge": global_fit.ridge,
        "uniform_fit_r2": r2_score(fit_split.target, global_fit_prediction),
        "uniform_validation_r2": r2_score(validation.target, global_validation_prediction),
        **retrieval_metrics(test_target, global_prediction, args.top_k, args.seed + 601),
    }
    variants.insert(1, global_metadata)

    matched_payloads = {row["decoder"]: row["payload_parameters"] for row in variants if row["decoder"] != "dense_w_diagnostic"}
    if len(set(matched_payloads.values())) != 1:
        raise RuntimeError(f"matched-budget invariant failed: {matched_payloads}")
    result = {
        "seed": args.seed,
        "teacher": args.teacher,
        "input_dim": args.input_dim,
        "train_queries": args.train_queries,
        "train_keys": args.train_keys,
        "test_queries": args.test_queries,
        "test_keys": args.test_keys,
        "max_value": args.max_value,
        "top_k": args.top_k,
        "tables": args.tables,
        "states_per_table": S4_ORDER,
        "rank": args.rank,
        "screen_samples": args.screen_samples,
        "fit_samples": args.fit_samples,
        "validation_samples": args.validation_samples,
        "ridge_grid": ridge_grid,
        "fixed_routes": True,
        "same_router_for_all_decoders": True,
        "route_anchor_groups": router.anchors.detach().cpu().tolist(),
        "matched_payload_parameters": next(iter(matched_payloads.values())),
        "dense_payload_parameters": dense_fit.coefficient.numel(),
        "elapsed_seconds": time.perf_counter() - started,
        "variants": variants,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"seed{args.seed}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)


def summarize(args: argparse.Namespace) -> None:
    runs = [json.loads(path.read_text()) for path in sorted(args.result_dir.glob("**/seed*.json"))]
    if not runs:
        raise RuntimeError(f"no seed JSON files under {args.result_dir}")
    flat = [
        {
            "seed": run["seed"],
            "teacher": run["teacher"],
            "tables": run["tables"],
            "rank": run["rank"],
            "matched_payload_parameters": run["matched_payload_parameters"],
            "dense_payload_parameters": run["dense_payload_parameters"],
            **{key: value for key, value in variant.items() if key not in {"edges", "spectrum"}},
            "spectrum": variant.get("spectrum"),
        }
        for run in runs
        for variant in run["variants"]
    ]
    csv_fields = sorted({key for row in flat for key in row if key != "spectrum"})
    args.result_dir.mkdir(parents=True, exist_ok=True)
    with (args.result_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows([{key: value for key, value in row.items() if key != "spectrum"} for row in flat])

    teachers = sorted({row["teacher"] for row in flat})
    seeds = sorted({int(run["seed"]) for run in runs})
    example = runs[0]
    aggregate: list[dict[str, object]] = []
    for teacher in teachers:
        for decoder in DECODERS:
            members = [row for row in flat if row["teacher"] == teacher and row["decoder"] == decoder]
            if not members:
                continue
            aggregate.append(
                {
                    "teacher": teacher,
                    "decoder": decoder,
                    "seeds": len(members),
                    "parameters": int(members[0]["parameters"]),
                    "payload_parameters": int(members[0]["payload_parameters"]),
                    "object_lut_reads_per_object": int(members[0]["object_lut_reads_per_object"]),
                    "pair_payload_reads": int(members[0]["pair_payload_reads"]),
                    "pair_dot_width": int(members[0]["pair_dot_width"]),
                    "selected_ridge": statistics.mean(float(member["selected_ridge"]) for member in members),
                    **{
                        metric: statistics.mean(float(member[metric]) for member in members)
                        for metric in (
                            "uniform_fit_r2",
                            "uniform_validation_r2",
                            "score_r2",
                            "topk_recall",
                            "top1_accuracy",
                            "spearman",
                            "hard_negative_preference_accuracy",
                        )
                    },
                }
            )
    with (args.result_dir / "aggregate.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)

    spectrum_by_teacher: dict[str, dict[str, float]] = {}
    for teacher in teachers:
        spectra = [row["spectrum"] for row in flat if row["teacher"] == teacher and row["spectrum"] is not None]
        spectrum_by_teacher[teacher] = {
            key: statistics.mean(float(spectrum[key]) for spectrum in spectra)
            for key in ("top_rank_energy", "top_4_energy", "top_8_energy", "top_16_energy", "effective_rank")
        }

    def paired_delta(
        teacher: str,
        decoder: str,
        baseline: str,
        metric: str,
    ) -> tuple[float, float, int]:
        by_decoder = {
            name: {int(row["seed"]): float(row[metric]) for row in flat if row["teacher"] == teacher and row["decoder"] == name}
            for name in (decoder, baseline)
        }
        seeds = sorted(set(by_decoder[decoder]) & set(by_decoder[baseline]))
        deltas = [by_decoder[decoder][seed] - by_decoder[baseline][seed] for seed in seeds]
        mean = statistics.mean(deltas)
        sem = statistics.stdev(deltas) / math.sqrt(len(deltas)) if len(deltas) > 1 else 0.0
        return mean, sem, len(deltas)

    lines = [
        "# Frozen S4 Cross-Table Kernel Comparison",
        "",
        "## Protocol",
        "",
        "All decoders use identical frozen local-S4 routes, train/test objects, pair splits, and teacher matrix "
        "within each seed. `same_table_full`, `global_rank`, and `sparse_cross_table` have exactly matched "
        "payload parameter counts. `dense_w_diagnostic` is an explicitly higher-budget, validation-regularized "
        "diagnostic rather than a guaranteed finite-sample ceiling.",
        "",
        "The sparse decoder selects off-diagonal table blocks from a disjoint screen split using target-conditioned "
        "cell-mean R2, then fits only the selected blocks on the fit split. The global decoder uses alternating "
        "ridge-CG. Categorical decoders use ridge-CG. Tables report means over three seeds; paired deltas below "
        "report mean plus or minus one standard error across the same seeds.",
        "",
        f"Configuration: input dimension `{example['input_dim']}`, tables `{example['tables']}`, local states "
        f"`{example['states_per_table']}`, matched rank `{example['rank']}`, seeds "
        f"`{','.join(str(seed) for seed in seeds)}`, screen/fit/validation pairs "
        f"`{example['screen_samples']:,}/{example['fit_samples']:,}/{example['validation_samples']:,}`, and ridge "
        f"grid `{','.join(str(value) for value in example['ridge_grid'])}`. The formal launcher uses "
        f"{example.get('train_queries', 2048):,}/{example.get('train_keys', 2048):,} training objects and "
        f"{example.get('test_queries', 256):,}/{example.get('test_keys', 512):,} held query/key objects on "
        "normalized random-integer inputs.",
        "",
    ]
    for teacher in teachers:
        lines.extend(
            [
                f"## {teacher}",
                "",
                "| Decoder | Params | Obj LUT reads | Pair reads | Dot width | Ridge | Fit R2 | Valid R2 | Held R2 | "
                "Top-16 | Top-1 | Spearman | Hard-neg |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in aggregate:
            if row["teacher"] != teacher:
                continue
            lines.append(
                f"| {row['decoder']} | {row['parameters']:,} | {row['object_lut_reads_per_object']} | "
                f"{row['pair_payload_reads']} | {row['pair_dot_width']} | {row['selected_ridge']:.4g} | "
                f"{row['uniform_fit_r2']:.4f} | {row['uniform_validation_r2']:.4f} | {row['score_r2']:.4f} | "
                f"{row['topk_recall']:.4f} | {row['top1_accuracy']:.4f} | {row['spearman']:.4f} | "
                f"{row['hard_negative_preference_accuracy']:.4f} |"
            )
        spectrum = spectrum_by_teacher[teacher]
        lines.extend(
            [
                "",
                f"Dense-W spectrum: top-rank energy `{spectrum['top_rank_energy']:.4f}`, "
                f"top-4 `{spectrum['top_4_energy']:.4f}`, top-8 `{spectrum['top_8_energy']:.4f}`, "
                f"top-16 `{spectrum['top_16_energy']:.4f}`, entropy effective rank "
                f"`{spectrum['effective_rank']:.2f}`.",
                "",
            ]
        )

    lines.extend(["## Interpretation", ""])
    lookup = {(row["teacher"], row["decoder"]): row for row in aggregate}
    for teacher in teachers:
        global_rank = lookup[(teacher, "global_rank")]
        dense = lookup[(teacher, "dense_w_diagnostic")]
        global_topk = paired_delta(teacher, "global_rank", "same_table_full", "topk_recall")
        global_r2 = paired_delta(teacher, "global_rank", "same_table_full", "score_r2")
        sparse_topk = paired_delta(teacher, "sparse_cross_table", "same_table_full", "topk_recall")
        sparse_r2 = paired_delta(teacher, "sparse_cross_table", "same_table_full", "score_r2")
        lines.extend(
            [
                f"- `{teacher}` global-rank minus same-table: Top-16 "
                f"`{global_topk[0]:+.4f} +/- {global_topk[1]:.4f}` SEM and held-object R2 "
                f"`{global_r2[0]:+.4f} +/- {global_r2[1]:.4f}` SEM (`n={global_topk[2]}`).",
                f"- `{teacher}` sparse-cross minus same-table: Top-16 "
                f"`{sparse_topk[0]:+.4f} +/- {sparse_topk[1]:.4f}` SEM and held-object R2 "
                f"`{sparse_r2[0]:+.4f} +/- {sparse_r2[1]:.4f}` SEM (`n={sparse_topk[2]}`).",
                f"- `{teacher}` dense-W fits much more of the sampled training target "
                f"(`{dense['uniform_fit_r2']:.4f}` versus `{global_rank['uniform_fit_r2']:.4f}`) but generalizes "
                f"worse than global-rank (held R2 `{dense['score_r2']:.4f}` versus `{global_rank['score_r2']:.4f}`).",
            ]
        )
    lines.extend(
        [
            "",
            "The matched-budget result supports a diffuse, globally separable cross-table kernel, not a small set of "
            "isolated full table-pair blocks. The dense decoder contains the rank-12 decoder in function class, so its "
            "lower held-object score is not a capacity contradiction: with this sample budget, factorization supplies "
            "a substantially better inductive bias than ridge on 147,456 independent cell parameters.",
            "",
            "Raw and ordinal teachers give nearly the same ordering of methods and similar absolute results. On this "
            "normalized random-integer support, the raw bilinear teacher is therefore not the main explanation for the "
            "weak same-table relation LUT; the dominant tested restriction is block-diagonal table pairing. This does "
            "not establish the same conclusion for natural data or other teacher families.",
            "",
            "The validation-selected dense-W spectrum describes a finite-sample ridge solution only. Its broad "
            "spectrum is not a necessary condition against a useful factorized predictor because direct W ridge and "
            "factor ridge impose different regularization.",
            "",
            "Execution differs qualitatively: global-rank reads 16 rank-12 rows once per object and then uses a "
            "width-12 query-key dot product; same-table and sparse-cross read 16 scalar payloads per pair, while dense-W "
            "reads 256. The global decoder is separable and amortizes its LUT traffic across all candidate pairs, but it "
            "is a small arithmetic kernel rather than a strictly lookup-only relation operator.",
            "",
            "This frozen-route, closed-form probe establishes representational and sample-efficiency evidence, not "
            "end-to-end trainability. The next gate is to train the global factors online with an alive initialization "
            "or supervised warm start, then test whether Gauge/Coxeter sharing can compress the factor tables without "
            "destroying the cross-table gain.",
            "",
            "## Artifacts",
            "",
            f"- Authoritative results and CSV summaries: `{args.result_dir}`",
            f"- Per-run logs: `logs/relation_energy/{args.result_dir.name}`",
            "- Launcher: `scripts/run_tropnn_s4_cross_table_kernel_4gpu.sh`",
            "- Probe and tests: `python/src/tropnn/tools/s4_cross_table_kernel_probe.py` and "
            "`python/src/tropnn/tests/test_s4_cross_table_kernel_probe.py`",
            "",
            f"Formal invocation: `RUN_ID={args.result_dir.name} "
            "scripts/run_tropnn_s4_cross_table_kernel_4gpu.sh` with GPUs `0,2,3,4`. The earlier fixed-ridge diagnostic "
            "is preserved under `results/relation_energy/s4_cross_table_kernel_t16_20260720`; the authoritative run "
            "uses validation selection over the recorded ridge grid rather than rewriting that evidence.",
        ]
    )
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text("\n".join(lines) + "\n")
    print(json.dumps({"runs": len(runs), "teachers": teachers, "report": str(args.out_report)}, sort_keys=True))


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run":
        run(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
