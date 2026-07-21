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
from tropnn.tools.coxeter_relation_probe import PERMUTATIONS, LocalS4Router, permutation_tables
from tropnn.tools.route_degree_sparsity_probe import ridge_conjugate_gradient
from tropnn.tools.s4_native_global_kernel_probe import (
    NativeLayout,
    PairIndices,
    build_local_representation,
    build_native_layout,
    sample_pair_indices,
)

TEACHERS = (
    "intrinsic_kendall",
    "intrinsic_mallows",
    "intrinsic_cayley_diffusion",
    "root_incidence_tied",
    "root_hodge_spectral",
    "root_incidence_anisotropic",
)
VARIANTS = (
    "intrinsic_kendall",
    "intrinsic_mallows",
    "intrinsic_cayley_diffusion",
    "global_root_identity",
    "root_incidence_tied",
    "root_hodge_spectral",
    "root_diagonal",
    "root_incidence_sparse",
)


@dataclass(frozen=True)
class IntrinsicTables:
    kendall: Tensor
    mallows: Tensor
    cayley_diffusion: Tensor


@dataclass(frozen=True)
class RootGeometry:
    incidence: Tensor
    identity: Tensor
    tied_off_diagonal: Tensor
    hodge_gradient: Tensor
    hodge_residual: Tensor
    support_rows: Tensor
    support_columns: Tensor


@dataclass(frozen=True)
class LinearFit:
    weight: Tensor
    bias: Tensor
    ridge: float
    validation_r2: float
    iterations: int
    relative_residual: float


@dataclass(frozen=True)
class VariantDesign:
    name: str
    fit_features: Tensor
    validation_features: Tensor
    prediction_kind: str
    operators: tuple[Tensor, ...] = ()
    intrinsic_columns: tuple[int, ...] = ()
    entry_rows: Tensor | None = None
    entry_columns: Tensor | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tiered S4-intrinsic and structured comparison-root relation-kernel probe.")
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
    run.add_argument("--fit-samples", type=int, default=32768)
    run.add_argument("--validation-samples", type=int, default=8192)
    run.add_argument("--feature-batch-size", type=int, default=4096)
    run.add_argument("--solver-batch-size", type=int, default=8192)
    run.add_argument("--cg-iterations", type=int, default=64)
    run.add_argument("--cg-tolerance", type=float, default=1e-5)
    run.add_argument("--ridge-grid", default="0.00001,0.0001,0.001,0.01")
    run.add_argument("--mallows-beta", type=float, default=0.75)
    run.add_argument("--diffusion-time", type=float, default=0.75)
    run.add_argument("--top-k", type=int, default=16)
    run.add_argument("--device", default="cuda")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--out-dir", type=Path, required=True)

    summarize = commands.add_parser("summarize")
    summarize.add_argument("--result-dir", type=Path, required=True)
    summarize.add_argument("--out-report", type=Path, required=True)
    return parser


def parse_ridge_grid(text: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    if not values or any(value <= 0.0 for value in values):
        raise ValueError("ridge grid must contain positive comma-separated values")
    return values


def _standardize_kernel_table(table: Tensor) -> Tensor:
    centered = table - table.mean()
    return centered / centered.std(unbiased=False).clamp_min(1e-12)


def intrinsic_kernel_tables(
    device: torch.device | str = "cpu",
    *,
    mallows_beta: float = 0.75,
    diffusion_time: float = 0.75,
) -> IntrinsicTables:
    if mallows_beta <= 0.0 or diffusion_time <= 0.0:
        raise ValueError("Mallows beta and diffusion time must be positive")
    device = torch.device(device)
    inverse, composition, lengths = permutation_tables()
    inverse = inverse.to(device)
    composition = composition.to(device)
    lengths = lengths.to(device=device, dtype=torch.float64)
    states = torch.arange(24, device=device)
    relative = composition[inverse[:, None], states[None, :]]
    relative_length = lengths[relative]
    kendall = 1.0 - 2.0 * relative_length / 6.0
    mallows = torch.exp(-mallows_beta * relative_length)

    lookup = {permutation: index for index, permutation in enumerate(PERMUTATIONS)}
    adjacency = torch.zeros(24, 24, dtype=torch.float64, device=device)
    for state, permutation in enumerate(PERMUTATIONS):
        for generator in range(3):
            neighbour = list(permutation)
            neighbour[generator], neighbour[generator + 1] = neighbour[generator + 1], neighbour[generator]
            adjacency[state, lookup[tuple(neighbour)]] = 1.0
    laplacian = 3.0 * torch.eye(24, dtype=torch.float64, device=device) - adjacency
    eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
    heat = (eigenvectors * torch.exp(-diffusion_time * eigenvalues)) @ eigenvectors.T
    return IntrinsicTables(
        _standardize_kernel_table(kendall).to(torch.float32),
        _standardize_kernel_table(mallows).to(torch.float32),
        _standardize_kernel_table(heat).to(torch.float32),
    )


def stack_intrinsic_tables(tables: IntrinsicTables) -> Tensor:
    return torch.stack((tables.kendall, tables.mallows, tables.cayley_diffusion), dim=0)


def intrinsic_pair_features(
    query_routes: Tensor,
    key_routes: Tensor,
    indices: PairIndices,
    tables: IntrinsicTables,
    batch_size: int,
) -> Tensor:
    kernel_tables = stack_intrinsic_tables(tables)
    chunks: list[Tensor] = []
    for start in range(0, indices.query.shape[0], batch_size):
        query = query_routes[indices.query[start : start + batch_size]]
        key = key_routes[indices.key[start : start + batch_size]]
        scores = [table[query, key].mean(dim=-1) for table in kernel_tables]
        chunks.append(torch.stack(scores, dim=-1))
    return torch.cat(chunks, dim=0)


def intrinsic_score_matrices(
    query_routes: Tensor,
    key_routes: Tensor,
    tables: IntrinsicTables,
) -> tuple[Tensor, Tensor, Tensor]:
    result: list[Tensor] = []
    for kernel_table in stack_intrinsic_tables(tables):
        score = torch.zeros(
            query_routes.shape[0],
            key_routes.shape[0],
            device=query_routes.device,
        )
        for table in range(query_routes.shape[1]):
            score += kernel_table[query_routes[:, table, None], key_routes[None, :, table]]
        result.append(score / query_routes.shape[1])
    return result[0], result[1], result[2]


def root_geometry(layout: NativeLayout, device: torch.device | str = "cpu") -> RootGeometry:
    device = torch.device(device)
    edges = len(layout.edge_keys)
    incidence = torch.zeros(layout.input_dim, edges, device=device)
    for edge, (low, high) in enumerate(layout.edge_keys):
        incidence[low, edge] = -1.0
        incidence[high, edge] = 1.0
    gram = incidence.T @ incidence
    identity = torch.eye(edges, device=device)
    off_diagonal = gram - 2.0 * identity
    tied_norm = torch.linalg.matrix_norm(off_diagonal, ord=2).clamp_min(1e-12)
    tied_off_diagonal = off_diagonal / tied_norm

    vertex_laplacian = incidence @ incidence.T
    hodge_gradient = incidence.T @ torch.linalg.pinv(vertex_laplacian) @ incidence
    hodge_gradient = 0.5 * (hodge_gradient + hodge_gradient.T)
    hodge_residual = identity - hodge_gradient
    support = gram != 0.0
    support_rows, support_columns = torch.nonzero(support, as_tuple=True)
    return RootGeometry(
        incidence,
        identity,
        tied_off_diagonal,
        hodge_gradient,
        hodge_residual,
        support_rows,
        support_columns,
    )


def structured_teacher_operator(teacher: str, geometry: RootGeometry, seed: int) -> Tensor:
    if teacher == "root_incidence_tied":
        operator = 0.65 * geometry.identity + 0.35 * geometry.tied_off_diagonal
    elif teacher == "root_hodge_spectral":
        operator = 0.80 * geometry.hodge_gradient - 0.35 * geometry.hodge_residual
    elif teacher == "root_incidence_anisotropic":
        generator = torch.Generator(device="cpu").manual_seed(seed + 9091)
        edges = geometry.identity.shape[0]
        left = torch.randn(edges, generator=generator).to(geometry.identity.device)
        right = torch.randn(edges, generator=generator).to(geometry.identity.device)
        diagonal = torch.randn(edges, generator=generator).to(geometry.identity.device)
        left = left / left.std(unbiased=False).clamp_min(1e-12)
        right = right / right.std(unbiased=False).clamp_min(1e-12)
        support_values = geometry.incidence.T @ geometry.incidence
        support_values = support_values / torch.linalg.matrix_norm(support_values, ord=2).clamp_min(1e-12)
        operator = left[:, None] * support_values * right[None, :] + 0.35 * torch.diag(diagonal)
    else:
        raise ValueError(f"teacher {teacher!r} is not a structured root operator")
    return operator * (math.sqrt(operator.shape[0]) / torch.linalg.matrix_norm(operator).clamp_min(1e-12))


def bilinear_pair_features(
    query_root: Tensor,
    key_root: Tensor,
    indices: PairIndices,
    operators: tuple[Tensor, ...],
    batch_size: int,
) -> Tensor:
    chunks: list[Tensor] = []
    for start in range(0, indices.query.shape[0], batch_size):
        query = query_root[indices.query[start : start + batch_size]]
        key = key_root[indices.key[start : start + batch_size]]
        features = [((query @ operator) * key).sum(dim=-1) for operator in operators]
        chunks.append(torch.stack(features, dim=-1))
    return torch.cat(chunks, dim=0)


def entry_pair_features(
    query_root: Tensor,
    key_root: Tensor,
    indices: PairIndices,
    rows: Tensor,
    columns: Tensor,
    batch_size: int,
) -> Tensor:
    chunks: list[Tensor] = []
    for start in range(0, indices.query.shape[0], batch_size):
        query = query_root[indices.query[start : start + batch_size]][:, rows]
        key = key_root[indices.key[start : start + batch_size]][:, columns]
        chunks.append((query * key).to(torch.float16))
    return torch.cat(chunks, dim=0)


def r2_score(target: Tensor, prediction: Tensor) -> float:
    target64 = target.to(torch.float64)
    prediction64 = prediction.to(torch.float64)
    variance = (target64 - target64.mean()).square().mean().clamp_min(1e-30)
    return float((1.0 - (target64 - prediction64).square().mean() / variance).item())


def fit_linear_features(
    fit_features: Tensor,
    fit_target: Tensor,
    validation_features: Tensor,
    validation_target: Tensor,
    *,
    ridge_grid: tuple[float, ...],
    iterations: int,
    tolerance: float,
    batch_size: int,
) -> LinearFit:
    feature_mean = fit_features.float().mean(dim=0)
    feature_std = fit_features.float().std(dim=0, unbiased=False)
    feature_inv_std = torch.where(feature_std > 1e-7, feature_std.reciprocal(), 0.0)
    target_mean = fit_target.mean()
    centered_target = fit_target - target_mean
    candidates: list[LinearFit] = []
    for ridge in ridge_grid:
        standardized_weight, completed, relative_residual = ridge_conjugate_gradient(
            fit_features,
            feature_mean,
            feature_inv_std,
            centered_target,
            ridge,
            iterations,
            tolerance,
            batch_size,
        )
        raw_weight = standardized_weight * feature_inv_std
        bias = target_mean - torch.dot(feature_mean, raw_weight)
        validation_prediction = validation_features.float() @ raw_weight + bias
        candidates.append(
            LinearFit(
                raw_weight,
                bias,
                ridge,
                r2_score(validation_target, validation_prediction),
                completed,
                relative_residual,
            )
        )
    return max(candidates, key=lambda candidate: candidate.validation_r2)


def root_score_matrix(query_root: Tensor, key_root: Tensor, operator: Tensor) -> Tensor:
    return query_root @ operator @ key_root.T


def variant_prediction(
    design: VariantDesign,
    fit: LinearFit,
    intrinsic_matrices: tuple[Tensor, Tensor, Tensor],
    query_root: Tensor,
    key_root: Tensor,
) -> Tensor:
    if design.prediction_kind == "intrinsic":
        prediction = torch.zeros_like(intrinsic_matrices[0]) + fit.bias
        for weight, column in zip(fit.weight, design.intrinsic_columns, strict=True):
            prediction += weight * intrinsic_matrices[column]
        return prediction
    if design.prediction_kind == "operator_basis":
        operator = torch.zeros_like(design.operators[0])
        for weight, basis in zip(fit.weight, design.operators, strict=True):
            operator += weight * basis
    elif design.prediction_kind == "operator_entries":
        if design.entry_rows is None or design.entry_columns is None:
            raise RuntimeError("operator-entry design is missing its indices")
        operator = torch.zeros(query_root.shape[1], key_root.shape[1], device=query_root.device)
        operator[design.entry_rows, design.entry_columns] = fit.weight
    else:
        raise ValueError(f"unsupported prediction kind {design.prediction_kind!r}")
    return root_score_matrix(query_root, key_root, operator) + fit.bias


def teacher_pair_target(
    teacher: str,
    intrinsic_features: Tensor,
    query_root: Tensor,
    key_root: Tensor,
    indices: PairIndices,
    teacher_operator: Tensor | None,
) -> Tensor:
    if teacher.startswith("intrinsic_"):
        return intrinsic_features[:, TEACHERS.index(teacher)]
    if teacher_operator is None:
        raise RuntimeError("root teacher requires an operator")
    query = query_root[indices.query]
    key = key_root[indices.key]
    return ((query @ teacher_operator) * key).sum(dim=-1)


def run(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    problem = make_problem(args)
    train_query = problem.train_queries.to(device)
    train_key = problem.train_keys.to(device)
    test_query = problem.test_queries.to(device)
    test_key = problem.test_keys.to(device)
    router = LocalS4Router(args.input_dim, args.tables, args.seed).to(device)
    started = time.perf_counter()

    train_query_routes = router.route(train_query)
    train_key_routes = router.route(train_key)
    test_query_routes = router.route(test_query)
    test_key_routes = router.route(test_key)
    train_query_representation, query_diagnostics = build_local_representation(train_query_routes, router.anchors, args.input_dim)
    train_key_representation, key_diagnostics = build_local_representation(train_key_routes, router.anchors, args.input_dim)
    test_query_representation, _ = build_local_representation(test_query_routes, router.anchors, args.input_dim)
    test_key_representation, _ = build_local_representation(test_key_routes, router.anchors, args.input_dim)
    train_query_root = train_query_representation.comparison_root
    train_key_root = train_key_representation.comparison_root
    test_query_root = test_query_representation.comparison_root
    test_key_root = test_key_representation.comparison_root

    layout = build_native_layout(router.anchors, args.input_dim)
    geometry = root_geometry(layout, device)
    intrinsic_tables = intrinsic_kernel_tables(
        device,
        mallows_beta=args.mallows_beta,
        diffusion_time=args.diffusion_time,
    )
    fit_indices = sample_pair_indices(args.train_queries, args.train_keys, args.fit_samples, args.seed + 2003, device)
    validation_indices = sample_pair_indices(args.train_queries, args.train_keys, args.validation_samples, args.seed + 2017, device)
    intrinsic_fit = intrinsic_pair_features(
        train_query_routes,
        train_key_routes,
        fit_indices,
        intrinsic_tables,
        args.feature_batch_size,
    )
    intrinsic_validation = intrinsic_pair_features(
        train_query_routes,
        train_key_routes,
        validation_indices,
        intrinsic_tables,
        args.feature_batch_size,
    )
    intrinsic_matrices = intrinsic_score_matrices(test_query_routes, test_key_routes, intrinsic_tables)

    identity_operators = (geometry.identity,)
    tied_operators = (geometry.identity, geometry.tied_off_diagonal)
    hodge_operators = (geometry.hodge_gradient, geometry.hodge_residual)
    identity_fit = bilinear_pair_features(train_query_root, train_key_root, fit_indices, identity_operators, args.feature_batch_size)
    identity_validation = bilinear_pair_features(train_query_root, train_key_root, validation_indices, identity_operators, args.feature_batch_size)
    tied_fit = bilinear_pair_features(train_query_root, train_key_root, fit_indices, tied_operators, args.feature_batch_size)
    tied_validation = bilinear_pair_features(train_query_root, train_key_root, validation_indices, tied_operators, args.feature_batch_size)
    hodge_fit = bilinear_pair_features(train_query_root, train_key_root, fit_indices, hodge_operators, args.feature_batch_size)
    hodge_validation = bilinear_pair_features(train_query_root, train_key_root, validation_indices, hodge_operators, args.feature_batch_size)
    diagonal_indices = torch.arange(train_query_root.shape[1], device=device)
    diagonal_fit = entry_pair_features(
        train_query_root,
        train_key_root,
        fit_indices,
        diagonal_indices,
        diagonal_indices,
        args.feature_batch_size,
    )
    diagonal_validation = entry_pair_features(
        train_query_root,
        train_key_root,
        validation_indices,
        diagonal_indices,
        diagonal_indices,
        args.feature_batch_size,
    )
    incidence_fit = entry_pair_features(
        train_query_root,
        train_key_root,
        fit_indices,
        geometry.support_rows,
        geometry.support_columns,
        args.feature_batch_size,
    )
    incidence_validation = entry_pair_features(
        train_query_root,
        train_key_root,
        validation_indices,
        geometry.support_rows,
        geometry.support_columns,
        args.feature_batch_size,
    )

    designs = [
        VariantDesign("intrinsic_kendall", intrinsic_fit[:, 0:1], intrinsic_validation[:, 0:1], "intrinsic", intrinsic_columns=(0,)),
        VariantDesign("intrinsic_mallows", intrinsic_fit[:, 1:2], intrinsic_validation[:, 1:2], "intrinsic", intrinsic_columns=(1,)),
        VariantDesign(
            "intrinsic_cayley_diffusion",
            intrinsic_fit[:, 2:3],
            intrinsic_validation[:, 2:3],
            "intrinsic",
            intrinsic_columns=(2,),
        ),
        VariantDesign("global_root_identity", identity_fit, identity_validation, "operator_basis", identity_operators),
        VariantDesign("root_incidence_tied", tied_fit, tied_validation, "operator_basis", tied_operators),
        VariantDesign("root_hodge_spectral", hodge_fit, hodge_validation, "operator_basis", hodge_operators),
        VariantDesign(
            "root_diagonal",
            diagonal_fit,
            diagonal_validation,
            "operator_entries",
            entry_rows=diagonal_indices,
            entry_columns=diagonal_indices,
        ),
        VariantDesign(
            "root_incidence_sparse",
            incidence_fit,
            incidence_validation,
            "operator_entries",
            entry_rows=geometry.support_rows,
            entry_columns=geometry.support_columns,
        ),
    ]

    teacher_operator = None
    if args.teacher.startswith("root_"):
        teacher_operator = structured_teacher_operator(args.teacher, geometry, args.seed)
    fit_target = teacher_pair_target(
        args.teacher,
        intrinsic_fit,
        train_query_root,
        train_key_root,
        fit_indices,
        teacher_operator,
    )
    validation_target = teacher_pair_target(
        args.teacher,
        intrinsic_validation,
        train_query_root,
        train_key_root,
        validation_indices,
        teacher_operator,
    )
    if args.teacher.startswith("intrinsic_"):
        target = intrinsic_matrices[TEACHERS.index(args.teacher)]
    else:
        if teacher_operator is None:
            raise RuntimeError("root teacher requires an operator")
        target = root_score_matrix(test_query_root, test_key_root, teacher_operator)

    ridge_grid = parse_ridge_grid(args.ridge_grid)
    variants: list[dict[str, object]] = []
    for design in designs:
        fit = fit_linear_features(
            design.fit_features,
            fit_target,
            design.validation_features,
            validation_target,
            ridge_grid=ridge_grid,
            iterations=args.cg_iterations,
            tolerance=args.cg_tolerance,
            batch_size=args.solver_batch_size,
        )
        fit_prediction = design.fit_features.float() @ fit.weight + fit.bias
        validation_prediction = design.validation_features.float() @ fit.weight + fit.bias
        prediction = variant_prediction(
            design,
            fit,
            intrinsic_matrices,
            test_query_root,
            test_key_root,
        )
        variants.append(
            {
                "variant": design.name,
                "learned_parameters": fit.weight.numel() + 1,
                "relation_coefficients": fit.weight.numel(),
                "selected_ridge": fit.ridge,
                "fit_r2": r2_score(fit_target, fit_prediction),
                "validation_r2": r2_score(validation_target, validation_prediction),
                "solver_iterations": fit.iterations,
                "solver_relative_residual": fit.relative_residual,
                **retrieval_metrics(target, prediction, args.top_k, args.seed + 601),
            }
        )

    teacher_asymmetry = 0.0
    if teacher_operator is not None:
        teacher_asymmetry = float(
            (torch.linalg.matrix_norm(teacher_operator - teacher_operator.T) / torch.linalg.matrix_norm(teacher_operator)).item()
        )
    result = {
        "seed": args.seed,
        "teacher": args.teacher,
        "input_dim": args.input_dim,
        "tables": args.tables,
        "fit_samples": args.fit_samples,
        "validation_samples": args.validation_samples,
        "mallows_beta": args.mallows_beta,
        "diffusion_time": args.diffusion_time,
        "ridge_grid": ridge_grid,
        "route_anchor_groups": router.anchors.detach().cpu().tolist(),
        "unique_root_edges": len(layout.edge_keys),
        "incidence_operator_entries": int(geometry.support_rows.numel()),
        "teacher_operator_asymmetry": teacher_asymmetry,
        "train_query_diagnostics": query_diagnostics,
        "train_key_diagnostics": key_diagnostics,
        "semantics": {
            "intrinsic": "shared same-chart S4 kernels, averaged over tables; no coordinate-valued teacher",
            "kendall": "centered normalized comparison-root dot, equivalently affine Coxeter length",
            "mallows": "exp(-beta * Coxeter length), centered and standardized",
            "cayley_diffusion": "exact heat kernel exp(-tau L) on the S4 Cayley graph of adjacent transpositions",
            "root_incidence_tied": "two learned scalars on identity and signed root-incidence adjacency",
            "root_hodge_spectral": "two learned scalars on gradient and residual Hodge projectors",
            "root_diagonal": "one learned weight per globally labeled comparison root",
            "root_incidence_sparse": "one learned coefficient for each directed pair of identical or coordinate-incident roots",
        },
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
            }
            for metric in ("fit_r2", "validation_r2", "score_r2", "spearman", "topk_recall", "top1_accuracy"):
                mean, sem = mean_sem([float(member[metric]) for member in members])
                row[metric] = mean
                row[f"{metric}_sem"] = sem
            aggregate.append(row)
    (args.result_dir / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n")

    lines = [
        "# Tiered intrinsic and structured-root kernel test",
        "",
        "This experiment removes raw and rank-bilinear teachers. Tier 1 uses only shared intrinsic S4 "
        "relations. Tier 2 generates scores only from globally labeled comparison roots and structured "
        "root-space operators.",
        "",
        "## Protocol",
        "",
        f"Configuration: D={runs[0]['input_dim']}, T={runs[0]['tables']} local S4 charts, "
        f"{runs[0]['fit_samples']:,}/{runs[0]['validation_samples']:,} fit/validation pairs, three seeds. "
        "Kendall and Mallows are functions of relative Coxeter length. Cayley diffusion is the exact "
        "heat kernel of the adjacent-transposition Cayley graph, rather than another distance exponential.",
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
                "| Variant | Learned params | Validation R2 | Held R2 | Top-16 | Spearman |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            lines.append(
                f"| {row['variant']} | {row['learned_parameters']} | "
                f"{row['validation_r2']:.4f} +/- {row['validation_r2_sem']:.4f} | "
                f"{row['score_r2']:.4f} +/- {row['score_r2_sem']:.4f} | "
                f"{row['topk_recall']:.4f} +/- {row['topk_recall_sem']:.4f} | "
                f"{row['spearman']:.4f} +/- {row['spearman_sem']:.4f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Structural boundary",
            "",
            "The incidence-sparse operator is not a random matrix on raw coordinates. Its domain and "
            "codomain are comparison roots, and a coefficient exists only when two roots are identical "
            "or share a coordinate. The anisotropic teacher uses this same support but different query/key "
            "root weights, so it is a native asymmetric relation stress test.",
            "",
            "## Artifacts",
            "",
            f"- Results: `{args.result_dir}`",
            "- Launcher: `scripts/run_tropnn_s4_native_kernel_hierarchy_4gpu.sh`",
            "- Probe: `python/src/tropnn/tools/s4_native_kernel_hierarchy_probe.py`",
            "- Tests: `python/src/tropnn/tests/test_s4_native_kernel_hierarchy_probe.py`",
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
