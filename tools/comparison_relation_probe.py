from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from tropnn.layers.relation import ComparisonRelationLUT, RelationQuantization

VARIANTS = (
    "dense_oracle",
    "quantized_gram",
    "constrained_float",
    "gram_free_float",
    "random_free_float",
    "gram_free_ternary",
    "random_free_ternary",
    "gram_free_binary",
    "random_free_binary",
    "additive_float",
)


@dataclass(frozen=True)
class ProbeConfig:
    out_dir: Path
    variant: str = "gram_free_float"
    distribution: str = "random_int"
    teacher: str = "rank16"
    device: str = "cuda"
    input_dim: int = 64
    num_banks: int = 16
    num_codes: int = 32
    relation_rank: int = 16
    depth: int = 1
    serial_conditioning: bool = False
    train_queries: int = 4096
    train_keys: int = 4096
    test_queries: int = 256
    test_keys: int = 512
    batch_size: int = 4096
    fixed_steps: int = 10000
    joint_steps: int = 10000
    eval_every: int = 1000
    learning_rate: float = 0.003
    threshold_learning_rate: float = 0.0003
    seed: int = 0
    top_k: int = 16
    eval_pairs: int = 65536
    weight_decay: float = 0.0


@dataclass(frozen=True)
class RelationProblem:
    train_queries: Tensor
    train_keys: Tensor
    test_queries: Tensor
    test_keys: Tensor
    query_projection: Tensor
    key_projection: Tensor
    teacher_matrix: Tensor
    target_scale: Tensor

    def scores(self, query: Tensor, key: Tensor) -> Tensor:
        return (query @ self.teacher_matrix @ key.transpose(-1, -2)) / self.target_scale

    def aligned_scores(self, query: Tensor, key: Tensor) -> Tensor:
        return ((query @ self.teacher_matrix) * key).sum(dim=-1) / self.target_scale

    def query_features(self, query: Tensor) -> Tensor:
        return query @ self.query_projection

    def key_features(self, key: Tensor) -> Tensor:
        return key @ self.key_projection


class DenseOracle(nn.Module):
    def __init__(self, problem: RelationProblem) -> None:
        super().__init__()
        self.problem = problem

    def forward(self, query: Tensor, key: Tensor) -> Tensor:
        return self.problem.scores(query, key)

    def score_aligned(self, query: Tensor, key: Tensor) -> Tensor:
        return self.problem.aligned_scores(query, key)


class SerialRelationScorer(nn.Module):
    """Accumulated score controls subsequent thresholds; disabled gamma is the additive control."""

    def __init__(self, layers: list[ComparisonRelationLUT], *, conditioning: bool) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)
        shape = (len(layers), layers[0].num_banks, layers[0].comparisons)
        self.query_gamma = nn.Parameter(torch.zeros(shape), requires_grad=conditioning)
        self.key_gamma = nn.Parameter(torch.zeros(shape), requires_grad=conditioning)

    def set_threshold_training(self, enabled: bool) -> None:
        for layer in self.layers:
            layer.set_threshold_training(enabled)

    def score_aligned(self, query: Tensor, key: Tensor) -> Tensor:
        state = torch.zeros(query.shape[:-1], device=query.device, dtype=query.dtype)
        for depth, layer in enumerate(self.layers):
            control = torch.tanh(state / math.sqrt(query.shape[-1])).unsqueeze(-1).unsqueeze(-1)
            state = state + layer.score_aligned(
                query,
                key,
                query_threshold_offset=control * self.query_gamma[depth],
                key_threshold_offset=control * self.key_gamma[depth],
            )
        return state

    def forward(self, query: Tensor, key: Tensor) -> Tensor:
        prefix = query.shape[:-2]
        queries = query.unsqueeze(-2).expand(*prefix, query.shape[-2], key.shape[-2], query.shape[-1])
        keys = key.unsqueeze(-3).expand(*prefix, query.shape[-2], key.shape[-2], key.shape[-1])
        scores = self.score_aligned(queries.reshape(-1, query.shape[-1]), keys.reshape(-1, key.shape[-1]))
        return scores.reshape(*prefix, query.shape[-2], key.shape[-2])


def make_problem(config: ProbeConfig, device: torch.device) -> RelationProblem:
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    sizes = (config.train_queries, config.train_keys, config.test_queries, config.test_keys)
    samples = [_sample_vectors(size, config.input_dim, config.distribution, generator) for size in sizes]
    teacher = _teacher_matrix(config.input_dim, config.teacher, config.relation_rank, generator)
    u, singular, vh = torch.linalg.svd(teacher, full_matrices=False)
    rank = min(config.relation_rank, singular.numel())
    root = singular[:rank].sqrt()
    query_projection = u[:, :rank] * root
    key_projection = vh[:rank].transpose(0, 1) * root
    if rank < config.relation_rank:
        query_projection = torch.nn.functional.pad(query_projection, (0, config.relation_rank - rank))
        key_projection = torch.nn.functional.pad(key_projection, (0, config.relation_rank - rank))
    raw = samples[0] @ teacher @ samples[1][: min(1024, samples[1].shape[0])].T
    scale = raw.std(unbiased=False).clamp_min(1e-6)
    return RelationProblem(
        *(sample.to(device) for sample in samples),
        query_projection.to(device),
        key_projection.to(device),
        teacher.to(device),
        scale.to(device),
    )


def _sample_vectors(size: int, dim: int, distribution: str, generator: torch.Generator) -> Tensor:
    if distribution == "random_int":
        samples = torch.randint(-15, 16, (size, dim), generator=generator).float()
    elif distribution == "gaussian":
        samples = torch.randn(size, dim, generator=generator)
    else:
        raise ValueError(f"unsupported distribution {distribution!r}")
    return torch.nn.functional.normalize(samples, dim=-1)


def _teacher_matrix(dim: int, teacher: str, rank: int, generator: torch.Generator) -> Tensor:
    if teacher == "identity":
        return torch.eye(dim)
    if teacher == "rank16":
        actual_rank = min(rank, dim)
        left = torch.linalg.qr(torch.randn(dim, actual_rank, generator=generator), mode="reduced").Q
        right = torch.linalg.qr(torch.randn(dim, actual_rank, generator=generator), mode="reduced").Q
        return (left * torch.linspace(1.0, 0.25, actual_rank)) @ right.T
    if teacher == "full":
        left = torch.linalg.qr(torch.randn(dim, dim, generator=generator)).Q
        right = torch.linalg.qr(torch.randn(dim, dim, generator=generator)).Q
        return (left * torch.linspace(1.0, 0.1, dim)) @ right.T
    raise ValueError(f"unsupported teacher {teacher!r}")


def initialized_layer(
    config: ProbeConfig,
    problem: RelationProblem,
    *,
    quantization: RelationQuantization,
    random_free: bool,
    seed_offset: int = 0,
) -> ComparisonRelationLUT:
    constrained = ComparisonRelationLUT(
        config.input_dim,
        num_banks=config.num_banks,
        num_codes=config.num_codes,
        relation_rank=config.relation_rank,
        relation_mode="constrained_gram",
        relation_init="zeros",
        seed=config.seed + seed_offset,
    ).to(problem.train_queries.device)
    constrained.calibrate_routes(problem.train_queries, problem.train_keys)
    query_factors, key_factors = constrained.initialize_from_samples(
        problem.train_queries,
        problem.train_keys,
        problem.query_features(problem.train_queries),
        problem.key_features(problem.train_keys),
    )
    if config.variant in {"quantized_gram", "constrained_float"}:
        return constrained
    if config.variant == "additive_float":
        additive = ComparisonRelationLUT(
            config.input_dim,
            num_banks=config.num_banks,
            num_codes=config.num_codes,
            relation_rank=config.relation_rank,
            relation_mode="additive",
            relation_init="zeros",
            quantization="float",
            seed=config.seed + seed_offset,
        ).to(problem.train_queries.device)
        additive.query_router.load_state_dict(constrained.query_router.state_dict())
        additive.key_router.load_state_dict(constrained.key_router.state_dict())
        additive.initialize_cross_gram(query_factors, key_factors)
        return additive
    return ComparisonRelationLUT.free_from_constrained(
        constrained,
        quantization=quantization,
        random_init=random_free,
        seed=config.seed + seed_offset,
    ).to(problem.train_queries.device)


def make_model(config: ProbeConfig, problem: RelationProblem) -> nn.Module:
    if config.variant == "dense_oracle":
        return DenseOracle(problem)
    quantization: RelationQuantization = "float"
    if config.variant.endswith("ternary"):
        quantization = "ternary"
    elif config.variant.endswith("binary"):
        quantization = "binary"
    layers = [
        initialized_layer(
            config,
            problem,
            quantization=quantization,
            random_free=config.variant.startswith("random_free"),
            seed_offset=depth * 1000003,
        )
        for depth in range(config.depth)
    ]
    if config.depth == 1:
        return layers[0]
    with torch.no_grad():
        for layer in layers:
            if layer.spec.relation_mode == "free":
                layer.relation.div_(config.depth)
            else:
                layer.query_factors.div_(math.sqrt(config.depth))
                layer.key_factors.div_(math.sqrt(config.depth))
    return SerialRelationScorer(layers, conditioning=config.serial_conditioning)


@torch.no_grad()
def evaluate(model: nn.Module, problem: RelationProblem, top_k: int) -> dict[str, float]:
    prediction = model(problem.test_queries.unsqueeze(0), problem.test_keys.unsqueeze(0)).squeeze(0)
    target = problem.scores(problem.test_queries, problem.test_keys)
    error = prediction - target
    variance = target.var(unbiased=False).clamp_min(1e-12)
    pred_rank = prediction.argsort(dim=-1).argsort(dim=-1).float()
    true_rank = target.argsort(dim=-1).argsort(dim=-1).float()
    pred_rank -= pred_rank.mean(dim=-1, keepdim=True)
    true_rank -= true_rank.mean(dim=-1, keepdim=True)
    spearman = (pred_rank * true_rank).sum(dim=-1)
    spearman /= pred_rank.square().sum(dim=-1).sqrt() * true_rank.square().sum(dim=-1).sqrt() + 1e-12
    k = min(top_k, target.shape[-1])
    true_top = target.topk(k, dim=-1).indices
    pred_top = prediction.topk(k, dim=-1).indices
    overlap = (pred_top.unsqueeze(-1) == true_top.unsqueeze(-2)).any(dim=-1).float()
    discounts = 1.0 / torch.log2(torch.arange(k, device=target.device, dtype=torch.float32) + 2.0)
    teacher_best = target.argmax(dim=-1)
    order = prediction.argsort(dim=-1, descending=True)
    reciprocal_rank = 1.0 / ((order == teacher_best.unsqueeze(-1)).long().argmax(dim=-1).float() + 1.0)
    hard = target.topk(min(17, target.shape[-1]), dim=-1).indices
    hard_accuracy = (
        prediction.gather(1, hard[:, :1]) > prediction.gather(1, hard[:, 1:])
    ).float().mean()
    result = {
        "mse": float(error.square().mean().item()),
        "normalized_mse": float((error.square().mean() / variance).item()),
        "r2": float((1.0 - error.square().mean() / variance).item()),
        "spearman": float(spearman.mean().item()),
        "top1": float((prediction.argmax(dim=-1) == teacher_best).float().mean().item()),
        "top16_recall": float(overlap.mean().item()),
        "ndcg16": float(((overlap * discounts).sum(dim=-1) / discounts.sum()).mean().item()),
        "mrr": float(reciprocal_rank.mean().item()),
        "hard_negative_accuracy": float(hard_accuracy.item()),
    }
    result.update(_attention_output_metrics(prediction, target, seed=1729))
    return result


@torch.no_grad()
def pair_split_metrics(
    model: nn.Module,
    problem: RelationProblem,
    *,
    count: int,
    held: bool,
    seed: int,
) -> dict[str, float]:
    query_index, key_index = sample_pair_indices(
        problem.train_queries.shape[0],
        problem.train_keys.shape[0],
        count,
        held=held,
        seed=seed,
        device=problem.train_queries.device,
    )
    query = problem.train_queries[query_index]
    key = problem.train_keys[key_index]
    prediction = model.score_aligned(query, key)
    target = problem.aligned_scores(query, key)
    error = prediction - target
    variance = target.var(unbiased=False).clamp_min(1e-12)
    correlation = torch.corrcoef(torch.stack([prediction, target]))[0, 1]
    return {
        "mse": float(error.square().mean().item()),
        "normalized_mse": float((error.square().mean() / variance).item()),
        "r2": float((1.0 - error.square().mean() / variance).item()),
        "pearson": float(correlation.item()),
    }


def sample_pair_indices(
    query_count: int,
    key_count: int,
    count: int,
    *,
    held: bool,
    seed: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    generator = torch.Generator(device=device).manual_seed(seed)
    queries: list[Tensor] = []
    keys: list[Tensor] = []
    remaining = count
    while remaining:
        draw = max(remaining * (12 if held else 2), 1024)
        query = torch.randint(query_count, (draw,), generator=generator, device=device)
        key = torch.randint(key_count, (draw,), generator=generator, device=device)
        is_held = ((query * 1000003 + key * 9176) % 10) == 0
        keep = is_held if held else ~is_held
        take = min(remaining, int(keep.sum().item()))
        if take:
            queries.append(query[keep][:take])
            keys.append(key[keep][:take])
            remaining -= take
    return torch.cat(queries), torch.cat(keys)


@torch.no_grad()
def _attention_output_metrics(prediction: Tensor, target: Tensor, *, seed: int) -> dict[str, float]:
    generator = torch.Generator(device=target.device).manual_seed(seed)
    value = torch.randn(target.shape[-1], 32, generator=generator, device=target.device)
    value = torch.nn.functional.normalize(value, dim=-1)
    target_probability = torch.softmax(target, dim=-1)
    prediction_probability = torch.softmax(prediction, dim=-1)
    target_output = target_probability @ value
    prediction_output = prediction_probability @ value
    kl = (
        target_probability
        * (target_probability.clamp_min(1e-12).log() - prediction_probability.clamp_min(1e-12).log())
    ).sum(dim=-1)
    pred_top = prediction.topk(min(16, prediction.shape[-1]), dim=-1).indices
    retained_mass = target_probability.gather(1, pred_top).sum(dim=-1)
    return {
        "attention_kl": float(kl.mean().item()),
        "attention_output_mse": float((prediction_output - target_output).square().mean().item()),
        "attention_output_cosine": float(
            torch.nn.functional.cosine_similarity(prediction_output, target_output, dim=-1).mean().item()
        ),
        "teacher_mass_at_pred_top16": float(retained_mass.mean().item()),
    }


@torch.no_grad()
def route_statistics(model: nn.Module, problem: RelationProblem, *, pairs: int, seed: int) -> dict[str, float]:
    layers = relation_layers(model)
    if not layers:
        return {}
    layer = layers[0]
    query_route, key_route = layer.routes(problem.train_queries, problem.train_keys)
    entropies: list[float] = []
    occupancies: list[float] = []
    for route in (query_route.indices, key_route.indices):
        for bank in range(layer.num_banks):
            count = torch.bincount(route[:, bank], minlength=layer.num_codes).float()
            probability = count / count.sum()
            entropies.append(float((-(probability * probability.clamp_min(1e-12).log2()).sum()).item()))
            occupancies.append(float((count > 0).float().mean().item()))
    query_index, key_index = sample_pair_indices(
        problem.train_queries.shape[0],
        problem.train_keys.shape[0],
        pairs,
        held=False,
        seed=seed,
        device=problem.train_queries.device,
    )
    visited = []
    for bank in range(layer.num_banks):
        cell = query_route.indices[query_index, bank] * layer.num_codes + key_route.indices[key_index, bank]
        visited.append(torch.unique(cell).numel() / (layer.num_codes * layer.num_codes))
    query_collision = 1.0 - torch.unique(query_route.indices, dim=0).shape[0] / query_route.indices.shape[0]
    key_collision = 1.0 - torch.unique(key_route.indices, dim=0).shape[0] / key_route.indices.shape[0]
    return {
        "route_entropy_bits": sum(entropies) / len(entropies),
        "route_code_occupancy": sum(occupancies) / len(occupancies),
        "relation_cell_visitation": sum(visited) / len(visited),
        "query_route_collision": float(query_collision),
        "key_route_collision": float(key_collision),
    }


def relation_layers(model: nn.Module) -> list[ComparisonRelationLUT]:
    if isinstance(model, ComparisonRelationLUT):
        return [model]
    if isinstance(model, SerialRelationScorer):
        return list(model.layers)
    return []


def set_threshold_training(model: nn.Module, enabled: bool) -> None:
    if isinstance(model, ComparisonRelationLUT):
        model.set_threshold_training(enabled)
    elif isinstance(model, SerialRelationScorer):
        model.set_threshold_training(enabled)


def optimizer_for(model: nn.Module, config: ProbeConfig, *, joint: bool) -> torch.optim.Optimizer:
    threshold_ids = {
        id(router.thresholds)
        for layer in relation_layers(model)
        for router in (layer.query_router, layer.key_router)
    }
    payload = [p for p in model.parameters() if p.requires_grad and id(p) not in threshold_ids]
    thresholds = [p for p in model.parameters() if p.requires_grad and id(p) in threshold_ids]
    groups: list[dict[str, object]] = [{"params": payload, "lr": config.learning_rate}]
    if joint and thresholds:
        groups.append({"params": thresholds, "lr": config.threshold_learning_rate})
    return torch.optim.AdamW(groups, weight_decay=config.weight_decay, betas=(0.9, 0.95))


def run(config: ProbeConfig) -> dict[str, object]:
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    problem = make_problem(config, device)
    model = make_model(config, problem).to(device)
    config.out_dir.mkdir(parents=True, exist_ok=True)
    def measured(phase: str, step: int) -> dict[str, float | int | str]:
        metrics: dict[str, float | int | str] = {"phase": phase, "step": step}
        metrics.update({f"held_object_{key}": value for key, value in evaluate(model, problem, config.top_k).items()})
        metrics.update(
            {
                f"train_pair_{key}": value
                for key, value in pair_split_metrics(
                    model, problem, count=config.eval_pairs, held=False, seed=config.seed + 1201
                ).items()
            }
        )
        metrics.update(
            {
                f"held_pair_{key}": value
                for key, value in pair_split_metrics(
                    model, problem, count=config.eval_pairs, held=True, seed=config.seed + 1201
                ).items()
            }
        )
        metrics.update(route_statistics(model, problem, pairs=config.eval_pairs, seed=config.seed + 1301))
        return metrics

    history: list[dict[str, float | int | str]] = [measured("initial", 0)]
    if config.variant not in {"dense_oracle", "quantized_gram"}:
        generator = torch.Generator(device=device).manual_seed(config.seed + 919)
        started = time.perf_counter()
        global_step = 0
        for phase, steps, joint in (("fixed", config.fixed_steps, False), ("joint", config.joint_steps, True)):
            set_threshold_training(model, joint)
            optimizer = optimizer_for(model, config, joint=joint)
            model.train()
            for phase_step in range(1, steps + 1):
                qi, ki = sample_pair_indices(
                    problem.train_queries.shape[0],
                    problem.train_keys.shape[0],
                    config.batch_size,
                    held=False,
                    seed=config.seed + global_step * 7919 + 1409,
                    device=device,
                )
                query, key = problem.train_queries[qi], problem.train_keys[ki]
                target = problem.aligned_scores(query, key)
                prediction = model.score_aligned(query, key)
                loss = (prediction - target).square().mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                global_step += 1
                if phase_step % config.eval_every == 0 or phase_step == steps:
                    model.eval()
                    row = measured(phase, global_step)
                    row["train_loss"] = float(loss.item())
                    row["steps_per_second"] = global_step / max(time.perf_counter() - started, 1e-9)
                    history.append(row)
                    model.train()
            if phase == "fixed":
                torch.save(model.state_dict(), config.out_dir / "checkpoint_fixed.pt")
    model.eval()
    summary = {
        "config": {**asdict(config), "out_dir": str(config.out_dir)},
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "active_comparisons_per_pair": config.depth * 2 * config.num_banks * int(math.log2(config.num_codes)),
        "active_relation_lookups_per_pair": config.depth * config.num_banks,
        "float_master_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
        "inference_relation_bits": (
            config.depth
            * config.num_banks
            * config.num_codes
            * config.num_codes
            * (2 if config.variant.endswith("ternary") else 1 if config.variant.endswith("binary") else 32)
        ),
        "initial": history[0],
        "final": history[-1],
    }
    with (config.out_dir / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in history for key in row}))
        writer.writeheader()
        writer.writerows(history)
    (config.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    torch.save(model.state_dict(), config.out_dir / "checkpoint.pt")
    print(json.dumps(summary["final"], sort_keys=True))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a comparison-routed discrete binary relation.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, default="gram_free_float")
    parser.add_argument("--distribution", choices=("random_int", "gaussian"), default="random_int")
    parser.add_argument("--teacher", choices=("identity", "rank16", "full"), default="rank16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--input-dim", type=int, default=64)
    parser.add_argument("--num-banks", type=int, default=16)
    parser.add_argument("--num-codes", type=int, default=32)
    parser.add_argument("--relation-rank", type=int, default=16)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--serial-conditioning", action="store_true")
    parser.add_argument("--train-queries", type=int, default=4096)
    parser.add_argument("--train-keys", type=int, default=4096)
    parser.add_argument("--test-queries", type=int, default=256)
    parser.add_argument("--test-keys", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--fixed-steps", type=int, default=10000)
    parser.add_argument("--joint-steps", type=int, default=10000)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--threshold-learning-rate", type=float, default=0.0003)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--eval-pairs", type=int, default=65536)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    return parser


def main() -> None:
    run(ProbeConfig(**vars(build_parser().parse_args())))


if __name__ == "__main__":
    main()
