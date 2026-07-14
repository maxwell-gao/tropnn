from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from tropnn.tools.bilinear_retrieval_probe import (
    make_problem,
    predict_score_matrix,
    retrieval_metrics,
    teacher_scores,
)
from tropnn.tools.fixed_wide_route_decoder_probe import FixedWideRouteBits
from tropnn.tools.fixed_route_relation_energy_probe import make_relation_pairs


CURVE_DEGREES = (1, 2, 3, 5, 8)
CURVE_BUDGETS = (128, 512, 2048, 8192, 32768)


@dataclass(frozen=True)
class UniformPairSplit:
    pair: Tensor
    target: Tensor


def add_problem_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-dim", type=int, default=32)
    parser.add_argument("--train-queries", type=int, default=2048)
    parser.add_argument("--train-keys", type=int, default=2048)
    parser.add_argument("--test-queries", type=int, default=256)
    parser.add_argument("--test-keys", type=int, default=512)
    parser.add_argument("--max-value", type=int, default=15)
    parser.add_argument("--tables-per-seed-block", type=int, default=16)
    parser.add_argument("--comparisons", type=int, default=5)
    parser.add_argument("--seed-blocks", type=int, default=16)
    parser.add_argument("--positive-per-query", type=int, default=16)
    parser.add_argument("--hard-negative-per-query", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Degree-sparsity approximation curve for a bilinear score on frozen PC-LUT route bits."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare")
    add_problem_arguments(prepare)
    prepare.add_argument("--screen-samples", type=int, default=32768)
    prepare.add_argument("--fit-samples", type=int, default=65536)
    prepare.add_argument("--validation-samples", type=int, default=32768)
    prepare.add_argument("--max-degree", type=int, default=8)
    prepare.add_argument("--beam-width", type=int, default=1024)
    prepare.add_argument("--supports-per-degree", type=int, default=65536)
    prepare.add_argument("--random-candidates", type=int, default=8192)
    prepare.add_argument("--score-chunk-size", type=int, default=256)
    prepare.add_argument("--residual-support-budget", type=int, default=8192)
    prepare.add_argument("--residual-exact-degree-fraction", type=float, default=0.25)
    prepare.add_argument("--residual-ridge", type=float, default=0.001)
    prepare.add_argument("--residual-cg-iterations", type=int, default=64)
    prepare.add_argument("--residual-cg-tolerance", type=float, default=1e-4)
    prepare.add_argument("--device", default="cuda")
    prepare.add_argument("--support-file", type=Path, required=True)

    run = commands.add_parser("run")
    add_problem_arguments(run)
    run.add_argument("--screen-samples", type=int, default=32768)
    run.add_argument("--fit-samples", type=int, default=65536)
    run.add_argument("--validation-samples", type=int, default=32768)
    run.add_argument("--degree", choices=CURVE_DEGREES, type=int, required=True)
    run.add_argument("--support-budget", choices=CURVE_BUDGETS + (1280,), type=int, required=True)
    run.add_argument("--steps", type=int, default=5000)
    run.add_argument("--batch-size", type=int, default=2048)
    run.add_argument("--eval-batch-size", type=int, default=512)
    run.add_argument("--eval-every", type=int, default=500)
    run.add_argument("--feature-sample-chunk", type=int, default=2048)
    run.add_argument("--feature-support-chunk", type=int, default=1024)
    run.add_argument("--exact-degree-fraction", type=float, default=0.25)
    run.add_argument("--ridge", type=float, default=0.001)
    run.add_argument("--cg-iterations", type=int, default=96)
    run.add_argument("--cg-tolerance", type=float, default=1e-5)
    run.add_argument("--device", default="cuda")
    run.add_argument("--support-file", type=Path, required=True)
    run.add_argument("--out-dir", type=Path, required=True)

    summarize = commands.add_parser("summarize")
    summarize.add_argument("--result-dir", type=Path, required=True)
    summarize.add_argument("--additive-summary", type=Path, required=True)
    summarize.add_argument("--multihash-summary", type=Path, required=True)
    summarize.add_argument("--dense-route-summary", type=Path, required=True)
    summarize.add_argument("--out-report", type=Path, required=True)
    return parser


def uniform_pair_indices(args: argparse.Namespace) -> Tensor:
    total = args.screen_samples + args.fit_samples + args.validation_samples
    population = args.train_queries * args.train_keys
    if total > population:
        raise ValueError(f"requested {total} unique pairs from a population of {population}")
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 2203)
    return torch.randperm(population, generator=generator)[:total]


def make_uniform_split(problem: object, flat_indices: Tensor, device: torch.device) -> UniformPairSplit:
    train_keys = problem.train_keys.shape[0]
    query_index = torch.div(flat_indices, train_keys, rounding_mode="floor")
    key_index = flat_indices.remainder(train_keys)
    query = problem.train_queries[query_index].to(device)
    key = problem.train_keys[key_index].to(device)
    relation = problem.relation.to(device)
    target = ((query @ relation) * key).sum(dim=-1, keepdim=True)
    return UniformPairSplit(torch.cat([query, key], dim=-1), target)


def split_slice(args: argparse.Namespace, name: str) -> slice:
    screen_end = args.screen_samples
    fit_end = screen_end + args.fit_samples
    if name == "screen":
        return slice(0, screen_end)
    if name == "fit":
        return slice(screen_end, fit_end)
    if name == "validation":
        return slice(fit_end, fit_end + args.validation_samples)
    raise ValueError(name)


def make_encoder(args: argparse.Namespace, device: torch.device) -> FixedWideRouteBits:
    return FixedWideRouteBits(
        input_dim=2 * args.input_dim,
        tables_per_seed_block=args.tables_per_seed_block,
        comparisons=args.comparisons,
        seed_blocks=args.seed_blocks,
        seed=args.seed,
    ).to(device)


def monomial_values(route_bits: Tensor, supports: Tensor, chunk_size: int) -> Tensor:
    if supports.shape[0] == 0:
        return route_bits.new_empty((route_bits.shape[0], 0))
    chunks = []
    for start in range(0, supports.shape[0], chunk_size):
        support = supports[start : start + chunk_size]
        chunks.append(route_bits[:, support].prod(dim=-1))
    return torch.cat(chunks, dim=1)


def normalized_support_scores(
    route_bits: Tensor,
    centered_target: Tensor,
    supports: Tensor,
    chunk_size: int,
) -> Tensor:
    scores = []
    for start in range(0, supports.shape[0], chunk_size):
        values = monomial_values(route_bits, supports[start : start + chunk_size], chunk_size)
        mean = values.mean(dim=0)
        covariance = (values * centered_target).mean(dim=0)
        variance = (1.0 - mean.square()).clamp_min(1e-6)
        score = covariance.abs() / variance.sqrt()
        score = torch.where(variance > 1e-5, score, torch.full_like(score, -torch.inf))
        scores.append(score)
    return torch.cat(scores)


def random_supports(
    input_bits: int,
    degree: int,
    count: int,
    seed: int,
    device: torch.device,
) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    collected = []
    remaining = count
    while remaining > 0:
        proposal = torch.randint(
            0,
            input_bits,
            (max(remaining * 2, 1024), degree),
            generator=generator,
        ).sort(dim=-1).values
        valid = (proposal[:, 1:] != proposal[:, :-1]).all(dim=-1)
        proposal = proposal[valid]
        if proposal.shape[0]:
            collected.append(proposal[:remaining])
            remaining -= min(remaining, proposal.shape[0])
    return torch.unique(torch.cat(collected), dim=0).to(device)


def extension_candidates(
    route_bits: Tensor,
    centered_target: Tensor,
    prefixes: Tensor,
    keep: int,
) -> tuple[Tensor, Tensor]:
    prefix_values = monomial_values(route_bits, prefixes, 256)
    sample_count = route_bits.shape[0]
    mean = prefix_values.transpose(0, 1) @ route_bits / sample_count
    covariance = (prefix_values * centered_target).transpose(0, 1) @ route_bits / sample_count
    variance = (1.0 - mean.square()).clamp_min(1e-6)
    score = covariance.abs() / variance.sqrt()
    bit_index = torch.arange(route_bits.shape[1], device=route_bits.device)
    score.masked_fill_(bit_index.view(1, -1) <= prefixes[:, -1:].to(bit_index.dtype), -torch.inf)
    flat_score, flat_index = torch.topk(score.flatten(), k=min(keep, int(torch.isfinite(score).sum().item())))
    prefix_index = torch.div(flat_index, route_bits.shape[1], rounding_mode="floor")
    extension = flat_index.remainder(route_bits.shape[1]).unsqueeze(-1)
    return torch.cat([prefixes[prefix_index], extension], dim=-1), flat_score


def merge_candidates(
    supports: list[Tensor],
    scores: list[Tensor],
    keep: int,
) -> tuple[Tensor, Tensor]:
    joined_supports = torch.cat(supports)
    joined_scores = torch.cat(scores)
    unique_supports, inverse = torch.unique(joined_supports, dim=0, return_inverse=True)
    unique_scores = torch.full(
        (unique_supports.shape[0],),
        -torch.inf,
        device=joined_scores.device,
        dtype=joined_scores.dtype,
    )
    unique_scores.scatter_reduce_(0, inverse, joined_scores, reduce="amax", include_self=True)
    selected_scores, selected = torch.topk(unique_scores, k=min(keep, unique_scores.shape[0]))
    return unique_supports[selected], selected_scores


def residual_after_sparse_fit(
    route_bits: Tensor,
    target: Tensor,
    support_payload: dict[str, object],
    degree: int,
    args: argparse.Namespace,
) -> tuple[Tensor, float, int]:
    supports, _, _ = select_supports(
        support_payload,
        degree,
        args.residual_support_budget,
        args.residual_exact_degree_fraction,
    )
    supports = supports.to(route_bits.device)
    features = materialize_features(route_bits, supports, 2048, 1024)
    feature_mean = features.float().mean(dim=0)
    feature_inv_std = (1.0 - feature_mean.square()).clamp_min(1e-5).rsqrt()
    coefficient, _, _ = ridge_conjugate_gradient(
        features,
        feature_mean,
        feature_inv_std,
        target,
        args.residual_ridge,
        args.residual_cg_iterations,
        args.residual_cg_tolerance,
        2048,
    )
    prediction = predict_from_materialized(
        features,
        feature_mean,
        feature_inv_std,
        coefficient,
        torch.zeros((), device=route_bits.device),
        2048,
    )
    residual = target - prediction
    explained = 1.0 - float(residual.square().sum().item() / target.square().sum().clamp_min(1e-12).item())
    return residual, explained, supports.shape[0]


@torch.no_grad()
def prepare(args: argparse.Namespace) -> None:
    if args.max_degree < 1:
        raise ValueError("max-degree must be positive")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    problem = make_problem(args)
    all_indices = uniform_pair_indices(args)
    screen = make_uniform_split(problem, all_indices[split_slice(args, "screen")], device)
    encoder = make_encoder(args, device)
    route_bits = encoder(screen.pair)
    centered_target = screen.target.flatten()
    centered_target = (centered_target - centered_target.mean()) / centered_target.std().clamp_min(1e-6)

    supports_by_degree: dict[int, Tensor] = {}
    scores_by_degree: dict[int, Tensor] = {}
    singleton = torch.arange(route_bits.shape[1], device=device).unsqueeze(-1)
    singleton_score = normalized_support_scores(
        route_bits, centered_target.unsqueeze(-1), singleton, args.score_chunk_size
    )
    order = singleton_score.argsort(descending=True)
    supports_by_degree[1] = singleton[order].cpu()
    scores_by_degree[1] = singleton_score[order].cpu()
    beam = singleton[order[: args.beam_width]]
    selection_payload: dict[str, object] = {
        "supports_by_degree": supports_by_degree,
        "scores_by_degree": scores_by_degree,
    }
    residual, explained, residual_supports = residual_after_sparse_fit(
        route_bits,
        centered_target,
        selection_payload,
        1,
        args,
    )
    residual_curve = {1: explained}
    print(
        json.dumps(
            {
                "degree": 1,
                "candidates": singleton.shape[0],
                "residual_fit_supports": residual_supports,
                "cumulative_screen_r2": explained,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    for degree in range(2, args.max_degree + 1):
        prefixes = singleton if degree == 2 else beam
        extended, extended_score = extension_candidates(
            route_bits,
            residual.unsqueeze(-1),
            prefixes,
            args.supports_per_degree * 2,
        )
        random = random_supports(
            route_bits.shape[1],
            degree,
            args.random_candidates,
            args.seed + 3571 * degree,
            device,
        )
        random_score = normalized_support_scores(
            route_bits,
            residual.unsqueeze(-1),
            random,
            args.score_chunk_size,
        )
        selected, selected_score = merge_candidates(
            [extended, random],
            [extended_score, random_score],
            args.supports_per_degree,
        )
        supports_by_degree[degree] = selected.cpu()
        scores_by_degree[degree] = selected_score.cpu()
        beam = selected[: args.beam_width]
        residual, explained, residual_supports = residual_after_sparse_fit(
            route_bits,
            centered_target,
            selection_payload,
            degree,
            args,
        )
        residual_curve[degree] = explained
        print(
            json.dumps(
                {
                    "degree": degree,
                    "candidates": selected.shape[0],
                    "best_score": float(selected_score[0].item()),
                    "median_score": float(selected_score.median().item()),
                    "residual_fit_supports": residual_supports,
                    "cumulative_screen_r2": explained,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    metadata = {
        "input_dim": args.input_dim,
        "train_queries": args.train_queries,
        "train_keys": args.train_keys,
        "test_queries": args.test_queries,
        "test_keys": args.test_keys,
        "max_value": args.max_value,
        "tables_per_seed_block": args.tables_per_seed_block,
        "comparisons": args.comparisons,
        "seed_blocks": args.seed_blocks,
        "seed": args.seed,
        "screen_samples": args.screen_samples,
        "fit_samples": args.fit_samples,
        "validation_samples": args.validation_samples,
        "route_bit_dim": route_bits.shape[1],
        "selection": "residualized normalized empirical covariance; degree 2 exhaustive; degree >=3 beam plus random",
        "residual_support_budget": args.residual_support_budget,
        "residual_exact_degree_fraction": args.residual_exact_degree_fraction,
        "residual_ridge": args.residual_ridge,
        "residual_curve": residual_curve,
    }
    args.support_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "metadata": metadata,
            "supports_by_degree": supports_by_degree,
            "scores_by_degree": scores_by_degree,
        },
        args.support_file,
    )
    print(args.support_file)


def padded_supports(supports: Tensor, width: int) -> Tensor:
    if supports.shape[1] == width:
        return supports
    padding = torch.full((supports.shape[0], width - supports.shape[1]), -1, dtype=supports.dtype)
    return torch.cat([supports, padding], dim=-1)


def select_supports(
    payload: dict[str, object],
    degree: int,
    budget: int,
    exact_degree_fraction: float,
) -> tuple[Tensor, Tensor, Tensor]:
    supports_by_degree = payload["supports_by_degree"]
    scores_by_degree = payload["scores_by_degree"]
    supports = []
    scores = []
    orders = []
    for order in range(1, degree + 1):
        current = supports_by_degree[order]
        supports.append(padded_supports(current, degree))
        scores.append(scores_by_degree[order])
        orders.append(torch.full((current.shape[0],), order, dtype=torch.long))
    all_supports = torch.cat(supports)
    all_scores = torch.cat(scores)
    all_orders = torch.cat(orders)
    actual_budget = min(budget, all_scores.shape[0])
    if degree == 1 or exact_degree_fraction <= 0:
        selected_score, selected = torch.topk(all_scores, k=actual_budget)
        return all_supports[selected], all_orders[selected], selected_score

    exact_index = torch.nonzero(all_orders == degree, as_tuple=False).flatten()
    reserved = min(
        exact_index.shape[0],
        max(1, int(round(actual_budget * exact_degree_fraction))),
    )
    exact_score, exact_order = torch.topk(all_scores[exact_index], k=reserved)
    selected_exact = exact_index[exact_order]
    available = torch.ones(all_scores.shape[0], dtype=torch.bool)
    available[selected_exact] = False
    remainder = actual_budget - reserved
    if remainder:
        remaining_index = torch.nonzero(available, as_tuple=False).flatten()
        remaining_score, remaining_order = torch.topk(all_scores[remaining_index], k=remainder)
        selected = torch.cat([selected_exact, remaining_index[remaining_order]])
        selected_score = torch.cat([exact_score, remaining_score])
    else:
        selected = selected_exact
        selected_score = exact_score
    return all_supports[selected], all_orders[selected], selected_score


def materialize_features(
    route_bits: Tensor,
    supports: Tensor,
    sample_chunk: int,
    support_chunk: int,
) -> Tensor:
    output = torch.empty(
        (route_bits.shape[0], supports.shape[0]),
        device=route_bits.device,
        dtype=torch.float16,
    )
    support_mask = supports >= 0
    support_indices = supports.clamp_min(0)
    for sample_start in range(0, route_bits.shape[0], sample_chunk):
        sample = route_bits[sample_start : sample_start + sample_chunk]
        for support_start in range(0, supports.shape[0], support_chunk):
            index = support_indices[support_start : support_start + support_chunk]
            mask = support_mask[support_start : support_start + support_chunk]
            selected = sample[:, index]
            selected = torch.where(mask.view(1, *mask.shape), selected, torch.ones_like(selected))
            output[
                sample_start : sample_start + sample.shape[0],
                support_start : support_start + index.shape[0],
            ] = selected.prod(dim=-1).to(torch.float16)
    return output


class SparsePolynomialDecoder(nn.Module):
    def __init__(
        self,
        supports: Tensor,
        feature_mean: Tensor,
        feature_inv_std: Tensor,
        coefficient: Tensor,
        bias: Tensor,
        target_mean: Tensor,
        target_std: Tensor,
        support_chunk: int,
    ) -> None:
        super().__init__()
        self.register_buffer("supports", supports)
        self.register_buffer("feature_mean", feature_mean)
        self.register_buffer("feature_inv_std", feature_inv_std)
        self.register_buffer("coefficient", coefficient)
        self.register_buffer("bias", bias)
        self.register_buffer("target_mean", target_mean)
        self.register_buffer("target_std", target_std)
        self.support_chunk = support_chunk

    def forward(self, route_bits: Tensor) -> Tensor:
        score = self.bias.expand(route_bits.shape[0]).clone()
        for start in range(0, self.supports.shape[0], self.support_chunk):
            support = self.supports[start : start + self.support_chunk]
            mask = support >= 0
            selected = route_bits[:, support.clamp_min(0)]
            selected = torch.where(mask.view(1, *mask.shape), selected, torch.ones_like(selected))
            feature = selected.prod(dim=-1)
            feature = (feature - self.feature_mean[start : start + support.shape[0]]) * self.feature_inv_std[
                start : start + support.shape[0]
            ]
            score = score + feature @ self.coefficient[start : start + support.shape[0]]
        return (score * self.target_std + self.target_mean).unsqueeze(-1)


class RoutedPolynomialModel(nn.Module):
    def __init__(self, encoder: FixedWideRouteBits, decoder: SparsePolynomialDecoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, pair: Tensor) -> Tensor:
        return self.decoder(self.encoder(pair))


def predict_from_materialized(
    features: Tensor,
    feature_mean: Tensor,
    feature_inv_std: Tensor,
    coefficient: Tensor,
    bias: Tensor,
    batch_size: int,
) -> Tensor:
    predictions = []
    for start in range(0, features.shape[0], batch_size):
        feature = features[start : start + batch_size].float()
        feature = (feature - feature_mean) * feature_inv_std
        predictions.append(feature @ coefficient + bias)
    return torch.cat(predictions)


def r2_score(target: Tensor, prediction: Tensor) -> float:
    target = target.flatten().to(torch.float64)
    prediction = prediction.flatten().to(torch.float64)
    residual = (target - prediction).square().sum()
    total = (target - target.mean()).square().sum().clamp_min(1e-12)
    return float((1.0 - residual / total).item())


def standardized_normal_matvec(
    features: Tensor,
    feature_mean: Tensor,
    feature_inv_std: Tensor,
    vector: Tensor,
    ridge: float,
    batch_size: int,
) -> Tensor:
    result = ridge * vector
    scale = 1.0 / features.shape[0]
    for start in range(0, features.shape[0], batch_size):
        feature = features[start : start + batch_size].float()
        feature = (feature - feature_mean) * feature_inv_std
        result = result + scale * (feature.transpose(0, 1) @ (feature @ vector))
    return result


def standardized_rhs(
    features: Tensor,
    feature_mean: Tensor,
    feature_inv_std: Tensor,
    target: Tensor,
    batch_size: int,
) -> Tensor:
    result = torch.zeros(features.shape[1], device=features.device, dtype=torch.float32)
    scale = 1.0 / features.shape[0]
    for start in range(0, features.shape[0], batch_size):
        feature = features[start : start + batch_size].float()
        feature = (feature - feature_mean) * feature_inv_std
        result = result + scale * (feature.transpose(0, 1) @ target[start : start + batch_size])
    return result


def ridge_conjugate_gradient(
    features: Tensor,
    feature_mean: Tensor,
    feature_inv_std: Tensor,
    target: Tensor,
    ridge: float,
    max_iterations: int,
    tolerance: float,
    batch_size: int,
) -> tuple[Tensor, int, float]:
    right_hand_side = standardized_rhs(
        features,
        feature_mean,
        feature_inv_std,
        target,
        batch_size,
    )
    coefficient = torch.zeros_like(right_hand_side)
    residual = right_hand_side.clone()
    direction = residual.clone()
    residual_norm_squared = torch.dot(residual, residual)
    initial_norm = residual_norm_squared.sqrt().clamp_min(1e-12)
    relative_residual = 1.0
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        image = standardized_normal_matvec(
            features,
            feature_mean,
            feature_inv_std,
            direction,
            ridge,
            batch_size,
        )
        denominator = torch.dot(direction, image).clamp_min(1e-20)
        step = residual_norm_squared / denominator
        coefficient = coefficient + step * direction
        residual = residual - step * image
        next_norm_squared = torch.dot(residual, residual)
        relative_residual = float((next_norm_squared.sqrt() / initial_norm).item())
        if relative_residual <= tolerance:
            break
        direction = residual + (next_norm_squared / residual_norm_squared) * direction
        residual_norm_squared = next_norm_squared
    return coefficient, iterations, relative_residual


@torch.no_grad()
def pair_accuracy(model: nn.Module, positive: Tensor, negative: Tensor, batch_size: int) -> float:
    correct = 0
    for start in range(0, positive.shape[0], batch_size):
        positive_score = model(positive[start : start + batch_size])
        negative_score = model(negative[start : start + batch_size])
        correct += int((positive_score > negative_score).sum().item())
    return correct / positive.shape[0]


def verify_support_metadata(args: argparse.Namespace, metadata: dict[str, object]) -> None:
    keys = (
        "input_dim",
        "train_queries",
        "train_keys",
        "test_queries",
        "test_keys",
        "max_value",
        "tables_per_seed_block",
        "comparisons",
        "seed_blocks",
        "seed",
        "screen_samples",
        "fit_samples",
        "validation_samples",
    )
    mismatches = [key for key in keys if getattr(args, key) != metadata[key]]
    if mismatches:
        raise ValueError(f"support metadata mismatch: {', '.join(mismatches)}")


def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    support_payload = torch.load(args.support_file, map_location="cpu", weights_only=True)
    verify_support_metadata(args, support_payload["metadata"])
    supports, support_orders, support_scores = select_supports(
        support_payload,
        args.degree,
        args.support_budget,
        args.exact_degree_fraction,
    )
    supports = supports.to(device)

    problem = make_problem(args)
    all_indices = uniform_pair_indices(args)
    fit = make_uniform_split(problem, all_indices[split_slice(args, "fit")], device)
    validation = make_uniform_split(problem, all_indices[split_slice(args, "validation")], device)
    encoder = make_encoder(args, device)
    feature_started = time.perf_counter()
    fit_route = encoder(fit.pair)
    validation_route = encoder(validation.pair)
    fit_features = materialize_features(
        fit_route,
        supports,
        args.feature_sample_chunk,
        args.feature_support_chunk,
    )
    validation_features = materialize_features(
        validation_route,
        supports,
        args.feature_sample_chunk,
        args.feature_support_chunk,
    )
    feature_seconds = time.perf_counter() - feature_started

    feature_mean = fit_features.float().mean(dim=0)
    feature_inv_std = (1.0 - feature_mean.square()).clamp_min(1e-5).rsqrt()
    target_mean = fit.target.mean()
    target_std = fit.target.std().clamp_min(1e-6)
    normalized_target = ((fit.target.flatten() - target_mean) / target_std).float()
    started = time.perf_counter()
    coefficient, solver_iterations, solver_relative_residual = ridge_conjugate_gradient(
        fit_features,
        feature_mean,
        feature_inv_std,
        normalized_target,
        args.ridge,
        args.cg_iterations,
        args.cg_tolerance,
        args.batch_size,
    )
    elapsed = time.perf_counter() - started
    bias = torch.zeros((), device=device)
    history: list[dict[str, float | int]] = [
        {
            "iteration": solver_iterations,
            "relative_residual": solver_relative_residual,
        }
    ]
    print(json.dumps(history[-1], sort_keys=True), flush=True)

    train_prediction = predict_from_materialized(
        fit_features,
        feature_mean,
        feature_inv_std,
        coefficient,
        bias,
        args.batch_size,
    )
    validation_prediction = predict_from_materialized(
        validation_features,
        feature_mean,
        feature_inv_std,
        coefficient,
        bias,
        args.batch_size,
    )
    train_prediction = train_prediction * target_std + target_mean
    validation_prediction = validation_prediction * target_std + target_mean

    decoder = SparsePolynomialDecoder(
        supports,
        feature_mean,
        feature_inv_std,
        coefficient,
        bias,
        target_mean,
        target_std,
        args.feature_support_chunk,
    )
    model = RoutedPolynomialModel(encoder, decoder).eval()
    relation_pairs = make_relation_pairs(args, device)
    test_target = teacher_scores(
        relation_pairs.test_queries,
        relation_pairs.test_keys,
        relation_pairs.relation,
    )
    test_prediction = predict_score_matrix(
        model,
        relation_pairs.test_queries,
        relation_pairs.test_keys,
        args.eval_batch_size,
    )
    metrics = retrieval_metrics(test_target, test_prediction, args.top_k, args.seed + 601)
    degree_counts = {
        f"degree_{degree}_supports": int((support_orders == degree).sum().item())
        for degree in range(1, args.degree + 1)
    }
    result: dict[str, object] = {
        "max_degree": args.degree,
        "support_budget": args.support_budget,
        "actual_supports": supports.shape[0],
        "parameters": supports.shape[0] + 1,
        "solver": "ridge_cg",
        "ridge": args.ridge,
        "solver_iterations": solver_iterations,
        "solver_relative_residual": solver_relative_residual,
        "exact_degree_fraction": args.exact_degree_fraction,
        "screen_score_mean": float(support_scores.mean().item()),
        "feature_seconds": feature_seconds,
        "steps": solver_iterations,
        "elapsed_seconds": elapsed,
        "steps_per_second": solver_iterations / elapsed,
        "uniform_train_r2": r2_score(fit.target, train_prediction),
        "uniform_validation_r2": r2_score(validation.target, validation_prediction),
        "train_pair_accuracy": pair_accuracy(
            model,
            relation_pairs.positive,
            relation_pairs.negative,
            args.eval_batch_size,
        ),
        **degree_counts,
        **{f"test_{key}": value for key, value in metrics.items()},
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"degree_{args.degree}_budget_{args.support_budget}"
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
    rows = [json.loads(path.read_text()) for path in args.result_dir.glob("degree_*_budget_*.json")]
    rows.sort(key=lambda row: (row["max_degree"], row["support_budget"]))
    if not rows:
        raise RuntimeError(f"no degree-sparsity results in {args.result_dir}")
    fieldnames = sorted({key for row in rows for key in row})
    summary_path = args.result_dir / "summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    additive = next(
        row
        for row in read_csv(args.additive_summary)
        if row["variant"] == "pc_mse_adamw" and int(row["width"]) == 16
    )
    multihash = next(
        row
        for row in read_csv(args.multihash_summary)
        if int(row["group_size"]) == 3 and int(row["hashes"]) == 8
    )
    dense = next(row for row in read_csv(args.dense_route_summary) if row["objective"] == "mse")
    best = max(rows, key=lambda row: row["test_topk_recall"])
    lines = [
        "# Degree-Sparsity Approximation on Fixed PC-LUT Routes",
        "",
        "This probe holds the 1,280-bit fixed wide T256/C5 route encoder fixed and approximates the bilinear teacher score with a sparse linear combination of Boolean monomials. Supports are screened on uniformly sampled training query-key pairs, coefficients are fit on a disjoint uniform split, and all retrieval metrics use held-out queries and keys.",
        "",
        "Screening is residualized by degree: each order is scored against the residual left by lower-order sparse ridge fits. Degree 1 and degree 2 screening are exhaustive. Degree 3-8 screening is a heavy-correlation beam search augmented with uniformly random supports. Each fitted model reserves one quarter of its budget for its declared maximum degree. The high-degree curves are therefore measured lower bounds, not a complete Walsh spectrum on the unreachable uniform hypercube.",
        "",
        "| Max degree | Budget | Actual | Uniform train R2 | Uniform valid R2 | Held pair | Hard-neg | Top-16 | Top-1 | Spearman | CG it/s |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['max_degree']} | {row['support_budget']:,} | {row['actual_supports']:,} | "
            f"{row['uniform_train_r2']:.4f} | {row['uniform_validation_r2']:.4f} | "
            f"{row['test_random_pair_order_accuracy']:.4f} | {row['test_hard_negative_preference_accuracy']:.4f} | "
            f"{row['test_topk_recall']:.4f} | {row['test_top1_accuracy']:.4f} | "
            f"{row['test_spearman']:.4f} | {row['steps_per_second']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Existing decoder controls",
            "",
            "| Decoder | Params | Held pair | Hard-neg | Top-16 | Top-1 | Spearman |",
            "|---|---:|---:|---:|---:|---:|---:|",
            f"| Original additive PC-LUT | {int(additive['parameters']):,} | {float(additive['test_random_pair_order_accuracy']):.4f} | {float(additive['test_hard_negative_preference_accuracy']):.4f} | {float(additive['test_topk_recall']):.4f} | {float(additive['test_top1_accuracy']):.4f} | {float(additive['test_spearman']):.4f} |",
            f"| Random interaction g3-h8 | {int(multihash['parameters']):,} | {float(multihash['test_random_pair_order_accuracy']):.4f} | {float(multihash['test_hard_negative_preference_accuracy']):.4f} | {float(multihash['test_topk_recall']):.4f} | {float(multihash['test_top1_accuracy']):.4f} | {float(multihash['test_spearman']):.4f} |",
            f"| Dense route H256 | {int(dense['trainable_parameters']):,} | {float(dense['test_random_pair_order_accuracy']):.4f} | {float(dense['test_hard_negative_preference_accuracy']):.4f} | {float(dense['test_topk_recall']):.4f} | {float(dense['test_top1_accuracy']):.4f} | {float(dense['test_spearman']):.4f} |",
            "",
            "## Decision rule",
            "",
            "Rapid gains with degree at a fixed support budget indicate an interaction-order bottleneck. Rapid gains with support budget at a fixed degree indicate a dense-within-degree target. Saturation at low degree and modest budget supports a sparse LUT decoder. Failure of every sparse polynomial despite the dense route decoder succeeding means that the useful route function is compositional or spectrally broad rather than a sparse bounded-degree expansion.",
            "",
            "## Best screened polynomial",
            "",
            f"The best Top-16 point is degree {best['max_degree']} with budget {best['support_budget']:,}: uniform validation R2 {best['uniform_validation_r2']:.4f}, Held pair {best['test_random_pair_order_accuracy']:.4f}, Hard-neg {best['test_hard_negative_preference_accuracy']:.4f}, Top-16 {best['test_topk_recall']:.4f}, Top-1 {best['test_top1_accuracy']:.4f}, and Spearman {best['test_spearman']:.4f}.",
        ]
    )
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text("\n".join(lines) + "\n")
    print(summary_path)
    print(args.out_report)


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "run":
        run(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
