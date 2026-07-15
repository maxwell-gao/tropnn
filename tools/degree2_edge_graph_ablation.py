from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from pathlib import Path

import torch
from torch import Tensor

from tropnn.tools.bilinear_retrieval_probe import (
    make_problem,
    predict_score_matrix,
    retrieval_metrics,
    teacher_scores,
)
from tropnn.tools.fixed_route_relation_energy_probe import make_relation_pairs
from tropnn.tools.route_degree_sparsity_probe import (
    RoutedPolynomialModel,
    SparsePolynomialDecoder,
    add_problem_arguments,
    make_encoder,
    make_uniform_split,
    materialize_features,
    normalized_support_scores,
    pair_accuracy,
    predict_from_materialized,
    r2_score,
    ridge_conjugate_gradient,
    split_slice,
    uniform_pair_indices,
)


GRAPHS = ("random", "shared_anchor", "cross_qk", "offline_screened", "online_grown")
STOCHASTIC_GRAPH_SEEDS = (0, 1, 2)
EXPECTED_RUNS = tuple(
    (graph, graph_seed)
    for graph in GRAPHS
    for graph_seed in ((0,) if graph == "offline_screened" else STOCHASTIC_GRAPH_SEEDS)
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fixed-budget degree-2 edge-graph ablation on frozen PC-LUT route bits."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run")
    add_problem_arguments(run)
    run.add_argument("--graph", choices=GRAPHS, required=True)
    run.add_argument("--graph-seed", type=int, default=0)
    run.add_argument("--screen-samples", type=int, default=32768)
    run.add_argument("--fit-samples", type=int, default=65536)
    run.add_argument("--validation-samples", type=int, default=32768)
    run.add_argument("--edge-budget", type=int, default=8192)
    run.add_argument("--batch-size", type=int, default=2048)
    run.add_argument("--eval-batch-size", type=int, default=512)
    run.add_argument("--feature-sample-chunk", type=int, default=2048)
    run.add_argument("--feature-support-chunk", type=int, default=1024)
    run.add_argument("--ridge", type=float, default=0.001)
    run.add_argument("--cg-iterations", type=int, default=96)
    run.add_argument("--cg-tolerance", type=float, default=1e-5)
    run.add_argument("--online-rounds", type=int, default=8)
    run.add_argument("--online-proposals", type=int, default=32768)
    run.add_argument("--online-screen-batch", type=int, default=8192)
    run.add_argument("--online-cg-iterations", type=int, default=48)
    run.add_argument("--online-cg-tolerance", type=float, default=1e-4)
    run.add_argument("--device", default="cuda")
    run.add_argument("--out-dir", type=Path, required=True)

    summarize = commands.add_parser("summarize")
    summarize.add_argument("--result-dir", type=Path, required=True)
    summarize.add_argument("--additive-summary", type=Path, required=True)
    summarize.add_argument("--dense-route-summary", type=Path, required=True)
    summarize.add_argument("--out-report", type=Path, required=True)
    return parser


def all_degree2_edges(route_bits: int) -> Tensor:
    return torch.triu_indices(route_bits, route_bits, offset=1).transpose(0, 1).contiguous()


def route_anchor_metadata(encoder: object, input_dim: int) -> dict[str, Tensor]:
    anchor_a = torch.cat([block.anchor_a.reshape(-1).cpu() for block in encoder.route_blocks])
    anchor_b = torch.cat([block.anchor_b.reshape(-1).cpu() for block in encoder.route_blocks])
    tables = sum(block.anchor_a.shape[0] for block in encoder.route_blocks)
    table = torch.arange(tables).repeat_interleave(encoder.comparisons)
    q_only = (anchor_a < input_dim) & (anchor_b < input_dim)
    k_only = (anchor_a >= input_dim) & (anchor_b >= input_dim)
    return {
        "anchor_a": anchor_a,
        "anchor_b": anchor_b,
        "table": table,
        "q_only": q_only,
        "k_only": k_only,
    }


def edge_masks(edges: Tensor, metadata: dict[str, Tensor]) -> dict[str, Tensor]:
    left, right = edges[:, 0], edges[:, 1]
    anchor_a, anchor_b = metadata["anchor_a"], metadata["anchor_b"]
    shared = (
        (anchor_a[left] == anchor_a[right])
        | (anchor_a[left] == anchor_b[right])
        | (anchor_b[left] == anchor_a[right])
        | (anchor_b[left] == anchor_b[right])
    )
    cross_qk = (metadata["q_only"][left] & metadata["k_only"][right]) | (
        metadata["k_only"][left] & metadata["q_only"][right]
    )
    same_table = metadata["table"][left] == metadata["table"][right]
    return {"shared_anchor": shared, "cross_qk": cross_qk, "same_table": same_table}


def sample_edges(candidates: Tensor, budget: int, seed: int) -> Tensor:
    if candidates.shape[0] < budget:
        raise ValueError(f"edge graph has only {candidates.shape[0]} candidates for budget {budget}")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    selected = torch.randperm(candidates.shape[0], generator=generator)[:budget]
    return candidates[selected]


def fit_standardized(
    features: Tensor,
    target: Tensor,
    *,
    ridge: float,
    iterations: int,
    tolerance: float,
    batch_size: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, int, float]:
    feature_mean = features.float().mean(dim=0)
    feature_inv_std = (1.0 - feature_mean.square()).clamp_min(1e-5).rsqrt()
    coefficient, solver_iterations, relative_residual = ridge_conjugate_gradient(
        features,
        feature_mean,
        feature_inv_std,
        target,
        ridge,
        iterations,
        tolerance,
        batch_size,
    )
    prediction = predict_from_materialized(
        features,
        feature_mean,
        feature_inv_std,
        coefficient,
        torch.zeros((), device=features.device),
        batch_size,
    )
    return feature_mean, feature_inv_std, coefficient, prediction, solver_iterations, relative_residual


def degree1_residual(
    route_bits: Tensor,
    normalized_target: Tensor,
    args: argparse.Namespace,
) -> Tensor:
    features = route_bits.to(torch.float16)
    _, _, _, prediction, _, _ = fit_standardized(
        features,
        normalized_target,
        ridge=args.ridge,
        iterations=args.cg_iterations,
        tolerance=args.cg_tolerance,
        batch_size=args.batch_size,
    )
    return normalized_target - prediction


def offline_screened_edges(
    route_bits: Tensor,
    normalized_target: Tensor,
    budget: int,
    args: argparse.Namespace,
) -> tuple[Tensor, dict[str, object]]:
    residual = degree1_residual(route_bits, normalized_target, args)
    sample_count = route_bits.shape[0]
    mean = route_bits.transpose(0, 1) @ route_bits / sample_count
    covariance = (route_bits * residual.unsqueeze(-1)).transpose(0, 1) @ route_bits / sample_count
    variance = (1.0 - mean.square()).clamp_min(1e-6)
    score = covariance.abs() / variance.sqrt()
    score.masked_fill_(torch.tril(torch.ones_like(score, dtype=torch.bool)), -torch.inf)
    selected_score, selected = torch.topk(score.flatten(), k=budget)
    width = route_bits.shape[1]
    edges = torch.stack(
        [torch.div(selected, width, rounding_mode="floor"), selected.remainder(width)], dim=-1
    )
    return edges.cpu(), {
        "selection": "exhaustive residualized degree-2 normalized covariance",
        "mean_selected_score": float(selected_score.mean().item()),
        "best_selected_score": float(selected_score.max().item()),
    }


def online_residual_grown_edges(
    route_bits: Tensor,
    normalized_target: Tensor,
    all_edges: Tensor,
    budget: int,
    args: argparse.Namespace,
) -> tuple[Tensor, dict[str, object]]:
    if args.online_rounds < 1:
        raise ValueError("online-rounds must be positive")
    if args.online_proposals < math.ceil(budget / args.online_rounds):
        raise ValueError("online-proposals must cover the per-round growth")
    graph_generator = torch.Generator(device="cpu").manual_seed(args.graph_seed + 4079)
    sample_generator = torch.Generator(device="cpu").manual_seed(args.graph_seed + 6113)
    active_mask = torch.zeros(all_edges.shape[0], dtype=torch.bool)
    active_edges: list[Tensor] = []
    active_features: list[Tensor] = []
    residual = normalized_target
    history: list[dict[str, float | int]] = []
    active_count = 0

    for round_index in range(args.online_rounds):
        remaining = budget - active_count
        rounds_left = args.online_rounds - round_index
        grow = min(remaining, math.ceil(remaining / rounds_left))
        permutation = torch.randperm(all_edges.shape[0], generator=graph_generator)
        proposal_index = permutation[~active_mask[permutation]][: args.online_proposals]
        proposals = all_edges[proposal_index].to(route_bits.device)
        batch_count = min(args.online_screen_batch, route_bits.shape[0])
        sample_index = torch.randperm(route_bits.shape[0], generator=sample_generator)[:batch_count].to(
            route_bits.device
        )
        scores = normalized_support_scores(
            route_bits[sample_index],
            residual[sample_index].unsqueeze(-1),
            proposals,
            256,
        )
        selected_score, selected = torch.topk(scores, k=grow)
        selected_index = proposal_index[selected.cpu()]
        selected_edges = all_edges[selected_index]
        active_mask[selected_index] = True
        active_edges.append(selected_edges)
        new_features = materialize_features(
            route_bits,
            selected_edges.to(route_bits.device),
            args.feature_sample_chunk,
            args.feature_support_chunk,
        )
        active_features.append(new_features)
        active_count += selected_edges.shape[0]
        features = torch.cat(active_features, dim=1)
        _, _, _, prediction, solver_iterations, solver_residual = fit_standardized(
            features,
            normalized_target,
            ridge=args.ridge,
            iterations=args.online_cg_iterations,
            tolerance=args.online_cg_tolerance,
            batch_size=args.batch_size,
        )
        residual = normalized_target - prediction
        history.append(
            {
                "round": round_index + 1,
                "active_edges": active_count,
                "proposals": proposal_index.shape[0],
                "mean_selected_score": float(selected_score.mean().item()),
                "best_selected_score": float(selected_score.max().item()),
                "screen_r2": r2_score(normalized_target, prediction),
                "solver_iterations": solver_iterations,
                "solver_relative_residual": solver_residual,
            }
        )
        print(json.dumps(history[-1], sort_keys=True), flush=True)

    return torch.cat(active_edges), {
        "selection": "mini-batch residual growth from random proposals",
        "growth_history": history,
        "online_rounds": args.online_rounds,
        "online_proposals_per_round": args.online_proposals,
        "online_screen_batch": args.online_screen_batch,
    }


def select_graph(
    graph: str,
    encoder: object,
    screen_route: Tensor | None,
    normalized_screen_target: Tensor,
    args: argparse.Namespace,
) -> tuple[Tensor, dict[str, object], dict[str, Tensor], Tensor]:
    all_edges = all_degree2_edges(encoder.output_dim)
    metadata = route_anchor_metadata(encoder, args.input_dim)
    masks = edge_masks(all_edges, metadata)
    seed = args.seed + 10007 * args.graph_seed + 1709
    if graph == "random":
        return sample_edges(all_edges, args.edge_budget, seed), {"selection": "uniform random"}, metadata, all_edges
    if graph == "shared_anchor":
        candidates = all_edges[masks["shared_anchor"]]
        return sample_edges(candidates, args.edge_budget, seed), {
            "selection": "uniform among comparison pairs sharing at least one input coordinate",
            "candidate_edges": candidates.shape[0],
        }, metadata, all_edges
    if graph == "cross_qk":
        candidates = all_edges[masks["cross_qk"]]
        return sample_edges(candidates, args.edge_budget, seed), {
            "selection": "uniform q-only comparator x k-only comparator",
            "candidate_edges": candidates.shape[0],
        }, metadata, all_edges
    if screen_route is None:
        raise ValueError(f"{graph} requires screen route features")
    if graph == "offline_screened":
        edges, selection = offline_screened_edges(
            screen_route, normalized_screen_target, args.edge_budget, args
        )
    else:
        edges, selection = online_residual_grown_edges(
            screen_route,
            normalized_screen_target,
            all_edges,
            args.edge_budget,
            args,
        )
    return edges, selection, metadata, all_edges


def graph_statistics(supports: Tensor, metadata: dict[str, Tensor]) -> dict[str, float | int]:
    masks = edge_masks(supports, metadata)
    node_count = metadata["anchor_a"].shape[0]
    degree = torch.bincount(supports.flatten(), minlength=node_count)
    return {
        "node_coverage": float((degree > 0).float().mean().item()),
        "mean_node_degree": float(degree.float().mean().item()),
        "max_node_degree": int(degree.max().item()),
        "shared_anchor_edge_fraction": float(masks["shared_anchor"].float().mean().item()),
        "cross_qk_edge_fraction": float(masks["cross_qk"].float().mean().item()),
        "same_table_edge_fraction": float(masks["same_table"].float().mean().item()),
    }


def run(args: argparse.Namespace) -> None:
    if args.edge_budget < 1:
        raise ValueError("edge-budget must be positive")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    problem = make_problem(args)
    all_indices = uniform_pair_indices(args)
    screen = make_uniform_split(problem, all_indices[split_slice(args, "screen")], device)
    fit = make_uniform_split(problem, all_indices[split_slice(args, "fit")], device)
    validation = make_uniform_split(problem, all_indices[split_slice(args, "validation")], device)
    encoder = make_encoder(args, device)

    screen_target_mean = screen.target.mean()
    screen_target_std = screen.target.std().clamp_min(1e-6)
    normalized_screen_target = ((screen.target.flatten() - screen_target_mean) / screen_target_std).float()
    screen_route = encoder(screen.pair) if args.graph in {"offline_screened", "online_grown"} else None
    selection_started = time.perf_counter()
    supports, selection, metadata, _ = select_graph(
        args.graph,
        encoder,
        screen_route,
        normalized_screen_target,
        args,
    )
    selection_seconds = time.perf_counter() - selection_started
    if supports.shape != (args.edge_budget, 2) or torch.unique(supports, dim=0).shape[0] != args.edge_budget:
        raise RuntimeError("selected graph does not contain exactly edge-budget unique degree-2 edges")

    feature_started = time.perf_counter()
    fit_route = encoder(fit.pair)
    validation_route = encoder(validation.pair)
    fit_features = materialize_features(
        fit_route,
        supports.to(device),
        args.feature_sample_chunk,
        args.feature_support_chunk,
    )
    validation_features = materialize_features(
        validation_route,
        supports.to(device),
        args.feature_sample_chunk,
        args.feature_support_chunk,
    )
    feature_seconds = time.perf_counter() - feature_started

    target_mean = fit.target.mean()
    target_std = fit.target.std().clamp_min(1e-6)
    normalized_target = ((fit.target.flatten() - target_mean) / target_std).float()
    solver_started = time.perf_counter()
    feature_mean, feature_inv_std, coefficient, train_prediction, iterations, relative_residual = fit_standardized(
        fit_features,
        normalized_target,
        ridge=args.ridge,
        iterations=args.cg_iterations,
        tolerance=args.cg_tolerance,
        batch_size=args.batch_size,
    )
    solver_seconds = time.perf_counter() - solver_started
    validation_prediction = predict_from_materialized(
        validation_features,
        feature_mean,
        feature_inv_std,
        coefficient,
        torch.zeros((), device=device),
        args.batch_size,
    )
    train_prediction = train_prediction * target_std + target_mean
    validation_prediction = validation_prediction * target_std + target_mean

    decoder = SparsePolynomialDecoder(
        supports.to(device),
        feature_mean,
        feature_inv_std,
        coefficient,
        torch.zeros((), device=device),
        target_mean,
        target_std,
        args.feature_support_chunk,
    )
    model = RoutedPolynomialModel(encoder, decoder).eval()
    relation_pairs = make_relation_pairs(args, device)
    test_target = teacher_scores(
        relation_pairs.test_queries, relation_pairs.test_keys, relation_pairs.relation
    )
    test_prediction = predict_score_matrix(
        model, relation_pairs.test_queries, relation_pairs.test_keys, args.eval_batch_size
    )
    metrics = retrieval_metrics(test_target, test_prediction, args.top_k, args.seed + 601)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"edge_{args.graph}_gseed{args.graph_seed}"
    checkpoint = args.out_dir / f"{stem}.pt"
    torch.save(
        {
            "supports": supports,
            "coefficient": coefficient.cpu(),
            "feature_mean": feature_mean.cpu(),
            "feature_inv_std": feature_inv_std.cpu(),
            "target_mean": target_mean.cpu(),
            "target_std": target_std.cpu(),
            "route_seed": args.seed,
            "graph_seed": args.graph_seed,
            "graph": args.graph,
        },
        checkpoint,
    )
    result: dict[str, object] = {
        "graph": args.graph,
        "graph_seed": args.graph_seed,
        "route_seed": args.seed,
        "route_bit_dim": encoder.output_dim,
        "edge_budget": args.edge_budget,
        "actual_edges": supports.shape[0],
        "parameters": supports.shape[0] + 1,
        "selection_seconds": selection_seconds,
        "feature_seconds": feature_seconds,
        "solver_seconds": solver_seconds,
        "solver_iterations": iterations,
        "solver_relative_residual": relative_residual,
        "ridge": args.ridge,
        "uniform_train_r2": r2_score(fit.target, train_prediction),
        "uniform_validation_r2": r2_score(validation.target, validation_prediction),
        "train_pair_accuracy": pair_accuracy(
            model, relation_pairs.positive, relation_pairs.negative, args.eval_batch_size
        ),
        "checkpoint": str(checkpoint),
        **graph_statistics(supports, metadata),
        **selection,
        **{f"test_{key}": value for key, value in metrics.items()},
    }
    (args.out_dir / f"{stem}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def support_keys(path: str, width: int) -> set[int]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return {int(left) * width + int(right) for left, right in payload["supports"].tolist()}


def summarize(args: argparse.Namespace) -> None:
    rows = [json.loads(path.read_text()) for path in sorted(args.result_dir.glob("edge_*_gseed*.json"))]
    missing = [
        (graph, graph_seed)
        for graph, graph_seed in EXPECTED_RUNS
        if not any(row["graph"] == graph and row["graph_seed"] == graph_seed for row in rows)
    ]
    if missing:
        raise RuntimeError(f"missing edge-graph runs: {missing}")
    offline = next(row for row in rows if row["graph"] == "offline_screened")
    offline_keys = support_keys(offline["checkpoint"], int(offline["route_bit_dim"]))
    for row in rows:
        keys = support_keys(row["checkpoint"], int(row["route_bit_dim"]))
        row["offline_edge_overlap_fraction"] = len(keys & offline_keys) / int(row["edge_budget"])

    nested = {"growth_history"}
    scalar_rows = [{key: value for key, value in row.items() if key not in nested} for row in rows]
    summary_path = args.result_dir / "summary.csv"
    fieldnames = sorted({key for row in scalar_rows for key in row})
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scalar_rows)

    additive = next(
        row
        for row in read_csv(args.additive_summary)
        if row["variant"] == "pc_mse_adamw" and int(row["width"]) == 16
    )
    dense = next(row for row in read_csv(args.dense_route_summary) if row["objective"] == "mse")
    metric_names = (
        "uniform_train_r2",
        "uniform_validation_r2",
        "train_pair_accuracy",
        "test_random_pair_order_accuracy",
        "test_hard_negative_preference_accuracy",
        "test_topk_recall",
        "test_top1_accuracy",
        "test_spearman",
        "test_score_r2",
        "selection_seconds",
        "feature_seconds",
        "solver_seconds",
        "offline_edge_overlap_fraction",
        "node_coverage",
        "shared_anchor_edge_fraction",
        "cross_qk_edge_fraction",
    )

    grouped: dict[str, dict[str, float | int]] = {}
    for graph in GRAPHS:
        selected = [row for row in rows if row["graph"] == graph]
        values: dict[str, float | int] = {
            "runs": len(selected),
            "parameters": int(selected[0]["parameters"]),
        }
        values.update(
            {
                metric: statistics.mean(float(row[metric]) for row in selected)
                for metric in metric_names
            }
        )
        grouped[graph] = values

    lines = [
        "# Fixed-8K Degree-2 Edge-Graph Ablation",
        "",
        "## Controlled question",
        "",
        "All runs use the identical frozen W16/T256/C5 route encoder, uniform screen/fit/validation split, 8,192 unique degree-2 Walsh supports, final ridge-CG coefficient fit, and held-out retrieval evaluator. Only the edge graph changes.",
        "",
        "`random`, `shared_anchor`, and `cross_qk` never inspect the target when choosing edges. `offline_screened` exhaustively ranks all 818,560 edges against the degree-1 residual on the screen split. `online_grown` adds 1,024 edges for eight rounds from 32,768 random proposals scored on a screen mini-batch, refitting current coefficients between rounds; it never performs a global Walsh scan.",
        "",
        "## Runs",
        "",
        "| Graph | Seed | Valid R2 | Held pair | Hard-neg | Top-16 | Top-1 | Spearman | Oracle overlap | Shared-anchor | Cross-qk | Select s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: (item["graph"], item["graph_seed"])):
        lines.append(
            f"| {row['graph']} | {row['graph_seed']} | {row['uniform_validation_r2']:.4f} | "
            f"{row['test_random_pair_order_accuracy']:.4f} | {row['test_hard_negative_preference_accuracy']:.4f} | "
            f"{row['test_topk_recall']:.4f} | {row['test_top1_accuracy']:.4f} | {row['test_spearman']:.4f} | "
            f"{row['offline_edge_overlap_fraction']:.4f} | {row['shared_anchor_edge_fraction']:.4f} | "
            f"{row['cross_qk_edge_fraction']:.4f} | {row['selection_seconds']:.1f} |"
        )

    lines.extend(
        [
            "",
            "## Graph-seed means",
            "",
            "| Graph | Runs | Params | Valid R2 | Held pair | Hard-neg | Top-16 | Top-1 | Spearman | Oracle overlap | Node coverage | Select s |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for graph in GRAPHS:
        row = grouped[graph]
        lines.append(
            f"| {graph} | {row['runs']} | {row['parameters']:,} | {row['uniform_validation_r2']:.4f} | "
            f"{row['test_random_pair_order_accuracy']:.4f} | {row['test_hard_negative_preference_accuracy']:.4f} | "
            f"{row['test_topk_recall']:.4f} | {row['test_top1_accuracy']:.4f} | {row['test_spearman']:.4f} | "
            f"{row['offline_edge_overlap_fraction']:.4f} | {row['node_coverage']:.4f} | {row['selection_seconds']:.1f} |"
        )

    lines.extend(
        [
            "",
            "## Existing controls",
            "",
            "| Decoder | Params | Held pair | Hard-neg | Top-16 | Top-1 | Spearman |",
            "|---|---:|---:|---:|---:|---:|---:|",
            f"| Original additive PC-LUT | {int(additive['parameters']):,} | {float(additive['test_random_pair_order_accuracy']):.4f} | {float(additive['test_hard_negative_preference_accuracy']):.4f} | {float(additive['test_topk_recall']):.4f} | {float(additive['test_top1_accuracy']):.4f} | {float(additive['test_spearman']):.4f} |",
            f"| Dense route H256 | {int(dense['trainable_parameters']):,} | {float(dense['test_random_pair_order_accuracy']):.4f} | {float(dense['test_hard_negative_preference_accuracy']):.4f} | {float(dense['test_topk_recall']):.4f} | {float(dense['test_top1_accuracy']):.4f} | {float(dense['test_spearman']):.4f} |",
        ]
    )

    random_topk = float(grouped["random"]["test_topk_recall"])
    oracle_topk = float(grouped["offline_screened"]["test_topk_recall"])
    online_topk = float(grouped["online_grown"]["test_topk_recall"])
    geometry_best = max(("shared_anchor", "cross_qk"), key=lambda name: grouped[name]["test_topk_recall"])
    geometry_topk = float(grouped[geometry_best]["test_topk_recall"])
    denominator = max(oracle_topk - random_topk, 1e-12)
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"The best target-free geometry graph is `{geometry_best}` with mean Top-16 {geometry_topk:.4f}. It recovers {(geometry_topk - random_topk) / denominator:.1%} of the oracle-minus-random Top-16 gap.",
            "",
            f"Online residual growth reaches mean Top-16 {online_topk:.4f}, recovering {(online_topk - random_topk) / denominator:.1%} of the same gap without globally screening all Walsh pairs. Its mean exact edge overlap with the offline oracle is {grouped['online_grown']['offline_edge_overlap_fraction']:.2%}.",
            "",
            f"The offline oracle reaches Top-16 {oracle_topk:.4f}. Geometry-derived fixed graphs are sufficient only if they approach this value; otherwise the experiment identifies target-conditioned graph discovery, rather than coefficient optimization, as the unresolved mechanism.",
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
