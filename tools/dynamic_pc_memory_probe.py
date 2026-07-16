from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
from torch import Tensor


@dataclass(frozen=True)
class RouteConfig:
    tables: int
    comparisons: int

    def __post_init__(self) -> None:
        if self.tables < 1 or self.comparisons < 1:
            raise ValueError("tables and comparisons must be positive")

    @property
    def table_size(self) -> int:
        return 1 << self.comparisons

    @property
    def name(self) -> str:
        return f"pc_t{self.tables}_c{self.comparisons}"


@dataclass(frozen=True)
class ProbeConfig:
    out_dir: Path
    input_dim: int = 32
    value_dim: int = 32
    routes: tuple[RouteConfig, ...] = (RouteConfig(4, 3), RouteConfig(16, 5))
    facts: tuple[int, ...] = (1, 4, 8, 16, 32, 64)
    noise: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5)
    seeds: tuple[int, ...] = (0, 1, 2, 3)
    eta: float = 1.0
    concept_facts: int = 32

    def __post_init__(self) -> None:
        if self.input_dim < 2 or self.value_dim < 1:
            raise ValueError("input_dim must be at least 2 and value_dim must be positive")
        if not self.routes or not self.facts or not self.noise or not self.seeds:
            raise ValueError("routes, facts, noise, and seeds must be non-empty")
        if min(self.facts) < 1 or self.concept_facts < 1:
            raise ValueError("fact counts must be positive")
        if min(self.noise) < 0.0:
            raise ValueError("noise must be non-negative")
        if not 0.0 < self.eta <= 1.0:
            raise ValueError("eta must be in (0, 1]")


class OnlineMemory(Protocol):
    name: str
    state_scalars: int

    def read(self, keys: Tensor) -> Tensor: ...
    def write(self, key: Tensor, value: Tensor, eta: float) -> tuple[Tensor, Tensor]: ...
    def paired_similarity(self, left: Tensor, right: Tensor) -> Tensor: ...


class PairwiseRouter:
    """Fixed zero-threshold PC-LUT routes used as sparse memory addresses."""

    def __init__(self, input_dim: int, config: RouteConfig, seed: int) -> None:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        left = torch.randint(input_dim, (config.tables, config.comparisons), generator=generator)
        offset = torch.randint(1, input_dim, (config.tables, config.comparisons), generator=generator)
        self.anchors = torch.stack((left, (left + offset) % input_dim), dim=-1)
        self.powers = 2 ** torch.arange(config.comparisons, dtype=torch.long)
        self.config = config

    def route(self, inputs: Tensor) -> Tensor:
        if inputs.ndim == 1:
            inputs = inputs.unsqueeze(0)
        if inputs.ndim != 2:
            raise ValueError(f"expected [batch, input_dim], got {tuple(inputs.shape)}")
        margins = inputs[:, self.anchors[..., 0]] - inputs[:, self.anchors[..., 1]]
        return ((margins > 0).to(torch.long) * self.powers.view(1, 1, -1)).sum(dim=-1)

    def paired_similarity(self, left: Tensor, right: Tensor) -> Tensor:
        left_codes = self.route(left)
        right_codes = self.route(right)
        if left_codes.shape != right_codes.shape:
            raise ValueError("paired similarity requires equal batch shapes")
        return (left_codes == right_codes).to(torch.float64).mean(dim=-1)


class DynamicPCMemory:
    """A recurrent payload table addressed by fixed pairwise-comparison routes."""

    def __init__(self, router: PairwiseRouter, value_dim: int) -> None:
        self.router = router
        self.name = router.config.name
        self.state = torch.zeros(router.config.tables, router.config.table_size, value_dim, dtype=torch.float64)
        self.state_scalars = self.state.numel()

    def read(self, keys: Tensor) -> Tensor:
        codes = self.router.route(keys)
        tables = torch.arange(self.router.config.tables).view(1, -1).expand_as(codes)
        return self.state[tables, codes].mean(dim=1)

    def write(self, key: Tensor, value: Tensor, eta: float) -> tuple[Tensor, Tensor]:
        codes = self.router.route(key).squeeze(0)
        before = self.read(key).squeeze(0)
        error = value - before
        tables = torch.arange(self.router.config.tables)
        self.state[tables, codes] += float(eta) * error.unsqueeze(0)
        return before, self.read(key).squeeze(0)

    def paired_similarity(self, left: Tensor, right: Tensor) -> Tensor:
        return self.router.paired_similarity(left, right)


class DotProductLMS:
    """Normalized dot-product fast-weight memory used as the continuous control."""

    name = "dot_lms"

    def __init__(self, input_dim: int, value_dim: int) -> None:
        self.state = torch.zeros(value_dim, input_dim, dtype=torch.float64)
        self.state_scalars = self.state.numel()

    def read(self, keys: Tensor) -> Tensor:
        if keys.ndim == 1:
            keys = keys.unsqueeze(0)
        return keys.to(torch.float64) @ self.state.T

    def write(self, key: Tensor, value: Tensor, eta: float) -> tuple[Tensor, Tensor]:
        key = key.to(torch.float64)
        before = self.read(key).squeeze(0)
        error = value - before
        denominator = key.square().sum().clamp_min(1e-12)
        self.state += float(eta) * error.unsqueeze(1) * key.unsqueeze(0) / denominator
        return before, self.read(key).squeeze(0)

    def paired_similarity(self, left: Tensor, right: Tensor) -> Tensor:
        return (left.to(torch.float64) * right.to(torch.float64)).sum(dim=-1)


def normalized(values: Tensor) -> Tensor:
    return values / values.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def episode(seed: int, facts: int, input_dim: int, value_dim: int) -> tuple[Tensor, Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    keys = normalized(torch.randn(facts, input_dim, generator=generator, dtype=torch.float64))
    values = torch.randint(0, 2, (facts, value_dim), generator=generator, dtype=torch.long).to(torch.float64)
    values = (2.0 * values - 1.0) / math.sqrt(value_dim)
    return keys, values


def memories(config: ProbeConfig, seed: int) -> list[OnlineMemory]:
    result: list[OnlineMemory] = [DotProductLMS(config.input_dim, config.value_dim)]
    for index, route in enumerate(config.routes):
        router = PairwiseRouter(config.input_dim, route, seed=100_003 + 997 * seed + 53 * index)
        result.append(DynamicPCMemory(router, config.value_dim))
    return result


def write_facts(memory: OnlineMemory, keys: Tensor, values: Tensor, eta: float) -> None:
    for key, value in zip(keys, values, strict=True):
        memory.write(key, value, eta)


def retrieval_metrics(reads: Tensor, targets: Tensor) -> dict[str, float]:
    scores = reads @ targets.T
    expected = torch.arange(targets.shape[0])
    prediction = scores.argmax(dim=-1)
    predicted_bits = torch.where(reads >= 0, 1.0, -1.0)
    target_bits = torch.where(targets >= 0, 1.0, -1.0)
    cosine = torch.nn.functional.cosine_similarity(reads, targets, dim=-1, eps=1e-12)
    return {
        "mse": float((reads - targets).square().mean().item()),
        "bit_accuracy": float((predicted_bits == target_bits).to(torch.float64).mean().item()),
        "cosine": float(cosine.mean().item()),
        "top1": float((prediction == expected).to(torch.float64).mean().item()),
    }


def identity_experiment(config: ProbeConfig) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed in config.seeds:
        keys, values = episode(seed + 10_000, 2, config.input_dim, config.value_dim)
        for memory in memories(config, seed):
            before, after = memory.write(keys[0], values[0], eta=1.0)
            similarity = float(memory.paired_similarity(keys[1:2], keys[0:1]).item())
            cross_read = memory.read(keys[1:2]).squeeze(0)
            expected_cross = similarity * values[0]
            rows.append(
                {
                    "seed": seed,
                    "model": memory.name,
                    "state_scalars": memory.state_scalars,
                    "before_norm": float(before.norm().item()),
                    "read_after_write_max_error": float((after - values[0]).abs().max().item()),
                    "cross_kernel": similarity,
                    "cross_kernel_max_error": float((cross_read - expected_cross).abs().max().item()),
                }
            )
    return rows


def noisy_queries(keys: Tensor, noise: float, seed: int) -> Tensor:
    if noise == 0.0:
        return keys.clone()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    perturbation = normalized(torch.randn(keys.shape, generator=generator, dtype=keys.dtype))
    return normalized(keys + float(noise) * perturbation)


def capacity_experiment(config: ProbeConfig) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    maximum_facts = max(config.facts)
    for seed in config.seeds:
        all_keys, all_values = episode(seed + 20_000, maximum_facts, config.input_dim, config.value_dim)
        for fact_count in config.facts:
            keys = all_keys[:fact_count]
            values = all_values[:fact_count]
            for memory in memories(config, seed):
                write_facts(memory, keys, values, config.eta)
                for noise in config.noise:
                    queries = noisy_queries(keys, noise, seed=30_001 + seed * 101 + fact_count * 17 + round(noise * 1000))
                    reads = memory.read(queries)
                    similarity = memory.paired_similarity(queries, keys)
                    rows.append(
                        {
                            "seed": seed,
                            "model": memory.name,
                            "state_scalars": memory.state_scalars,
                            "facts": fact_count,
                            "noise": noise,
                            "eta": config.eta,
                            "kernel_similarity": float(similarity.mean().item()),
                            **retrieval_metrics(reads, values),
                        }
                    )
    return rows


def subset_bit_accuracy(reads: Tensor, targets: Tensor, indices: Tensor) -> float | None:
    if indices.numel() == 0:
        return None
    predicted = torch.where(reads[indices] >= 0, 1.0, -1.0)
    expected = torch.where(targets[indices] >= 0, 1.0, -1.0)
    return float((predicted == expected).to(torch.float64).mean().item())


def concept_drift_experiment(config: ProbeConfig) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    facts = config.concept_facts
    stages = sorted({0, facts // 4, facts // 2, 3 * facts // 4, facts})
    for seed in config.seeds:
        keys, old_values = episode(seed + 40_000, facts, config.input_dim, config.value_dim)
        _, new_values = episode(seed + 50_000, facts, config.input_dim, config.value_dim)
        order = torch.randperm(facts, generator=torch.Generator(device="cpu").manual_seed(seed + 60_000))
        for memory in memories(config, seed):
            write_facts(memory, keys, old_values, config.eta)
            previous = 0
            for overwritten in stages:
                for index in order[previous:overwritten]:
                    memory.write(keys[index], new_values[index], config.eta)
                current = old_values.clone()
                current[order[:overwritten]] = new_values[order[:overwritten]]
                reads = memory.read(keys)
                changed = order[:overwritten]
                unchanged = order[overwritten:]
                rows.append(
                    {
                        "seed": seed,
                        "model": memory.name,
                        "state_scalars": memory.state_scalars,
                        "facts": facts,
                        "overwritten": overwritten,
                        "overwritten_fraction": overwritten / facts,
                        "eta": config.eta,
                        "new_value_bit_accuracy": subset_bit_accuracy(reads, new_values, changed),
                        "retained_old_bit_accuracy": subset_bit_accuracy(reads, old_values, unchanged),
                        **retrieval_metrics(reads, current),
                    }
                )
                previous = overwritten
    return rows


def aggregate(rows: list[dict[str, object]], keys: tuple[str, ...], metrics: tuple[str, ...]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    result: list[dict[str, object]] = []
    for group_key, members in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        summary = dict(zip(keys, group_key, strict=True))
        summary["runs"] = len(members)
        summary["state_scalars"] = members[0]["state_scalars"]
        for metric in metrics:
            values = [float(row[metric]) for row in members if row.get(metric) is not None]
            summary[metric] = sum(values) / len(values) if values else None
        result.append(summary)
    return result


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, object]], columns: tuple[str, ...]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column)
            values.append("" if value is None else (f"{value:.6f}" if isinstance(value, float) else str(value)))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def build_report(config: ProbeConfig, identity: list[dict[str, object]], capacity: list[dict[str, object]], drift: list[dict[str, object]]) -> str:
    identity_summary = aggregate(identity, ("model",), ("read_after_write_max_error", "cross_kernel_max_error"))
    capacity_summary = aggregate(capacity, ("model", "facts", "noise"), ("kernel_similarity", "mse", "bit_accuracy", "cosine", "top1"))
    drift_summary = aggregate(
        drift,
        ("model", "overwritten", "overwritten_fraction"),
        ("mse", "bit_accuracy", "cosine", "top1", "new_value_bit_accuracy", "retained_old_bit_accuracy"),
    )
    max_facts = max(config.facts)
    exact_capacity = [row for row in capacity_summary if row["noise"] == 0.0]
    noisy_capacity = [row for row in capacity_summary if row["facts"] == max_facts]
    lines = [
        "# Dynamic PC-LUT Memory CPU Probe",
        "",
        "The PC memory reads the average of dynamically updated rows selected by fixed pairwise-comparison routes. Dot LMS is a normalized fast-weight control.",
        "",
        f"Seeds: `{list(config.seeds)}`. Input/value dimensions: `{config.input_dim}/{config.value_dim}`. Update rate: `{config.eta}`.",
        "",
        "## A. Read-after-write identities",
        "",
        *markdown_table(identity_summary, ("model", "state_scalars", "read_after_write_max_error", "cross_kernel_max_error")),
        "",
        "## B/D. Exact-query capacity and dot-product LMS control",
        "",
        *markdown_table(exact_capacity, ("model", "state_scalars", "facts", "top1", "bit_accuracy", "cosine", "mse")),
        "",
        f"## B/D. Noise sweep at {max_facts} facts",
        "",
        *markdown_table(noisy_capacity, ("model", "state_scalars", "noise", "kernel_similarity", "top1", "bit_accuracy", "cosine")),
        "",
        "## C. Overwrite/concept drift",
        "",
        *markdown_table(
            drift_summary,
            ("model", "state_scalars", "overwritten_fraction", "top1", "bit_accuracy", "new_value_bit_accuracy", "retained_old_bit_accuracy"),
        ),
        "",
    ]
    return "\n".join(lines)


def run_probe(config: ProbeConfig) -> dict[str, object]:
    identity = identity_experiment(config)
    capacity = capacity_experiment(config)
    drift = concept_drift_experiment(config)
    identity_summary = aggregate(identity, ("model",), ("read_after_write_max_error", "cross_kernel_max_error"))
    capacity_summary = aggregate(capacity, ("model", "facts", "noise"), ("kernel_similarity", "mse", "bit_accuracy", "cosine", "top1"))
    drift_summary = aggregate(
        drift,
        ("model", "overwritten", "overwritten_fraction"),
        ("mse", "bit_accuracy", "cosine", "top1", "new_value_bit_accuracy", "retained_old_bit_accuracy"),
    )
    config.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(config.out_dir / "identity.csv", identity)
    write_csv(config.out_dir / "capacity_noise.csv", capacity)
    write_csv(config.out_dir / "capacity_noise_summary.csv", capacity_summary)
    write_csv(config.out_dir / "concept_drift.csv", drift)
    write_csv(config.out_dir / "concept_drift_summary.csv", drift_summary)
    report = build_report(config, identity, capacity, drift)
    (config.out_dir / "report.md").write_text(report + "\n")
    summary = {
        "config": {
            "input_dim": config.input_dim,
            "value_dim": config.value_dim,
            "routes": [{"tables": route.tables, "comparisons": route.comparisons} for route in config.routes],
            "facts": list(config.facts),
            "noise": list(config.noise),
            "seeds": list(config.seeds),
            "eta": config.eta,
            "concept_facts": config.concept_facts,
        },
        "identity": identity_summary,
        "capacity": capacity_summary,
        "concept_drift": drift_summary,
    }
    (config.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(report, flush=True)
    return summary


def parse_route(value: str) -> RouteConfig:
    try:
        tables, comparisons = value.lower().split("x", maxsplit=1)
        return RouteConfig(int(tables), int(comparisons))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("route must use TABLESxCOMPARISONS, for example 16x5") from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CPU probe for dynamic comparison-addressed PC-LUT memory.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--input-dim", type=int, default=32)
    parser.add_argument("--value-dim", type=int, default=32)
    parser.add_argument("--routes", type=parse_route, nargs="+", default=(RouteConfig(4, 3), RouteConfig(16, 5)))
    parser.add_argument("--facts", type=int, nargs="+", default=(1, 4, 8, 16, 32, 64))
    parser.add_argument("--noise", type=float, nargs="+", default=(0.0, 0.1, 0.25, 0.5))
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2, 3))
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--concept-facts", type=int, default=32)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_probe(
        ProbeConfig(
            out_dir=args.out_dir,
            input_dim=args.input_dim,
            value_dim=args.value_dim,
            routes=tuple(args.routes),
            facts=tuple(args.facts),
            noise=tuple(args.noise),
            seeds=tuple(args.seeds),
            eta=args.eta,
            concept_facts=args.concept_facts,
        )
    )


if __name__ == "__main__":
    main()
