"""Go/no-go probe for chamber-conditioned lifting in the historical QKV toys.

The archived QKV experiment established two reference gaps at T16/C4/200 steps:

* selective-aggregation OOD value accuracy: PairwiseScore 0.2959 versus Dense
  QK 0.7285;
* value-transform OOD cosine: ordinary value LUT 0.7638 versus Dense Wv
  0.9888.

An ordinary value LUT is not used by the default selective-aggregation arm:
that toy transports random one-hot values directly.  Consequently this probe
uses the same chamber-lifting primitive in the two causally relevant roles:

1. recursive comparison-only Q/K towers before the unchanged PairwiseScore
   relation LUT;
2. a recursive value map in place of the ordinary full-vector value LUT.

The preregistered joint gate is OOD selective-aggregation accuracy >= 0.60 and
OOD value-transform cosine >= 0.90 in the same architecture variant.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from tropnn.layers import ChamberLiftingTower, PairwiseLUT
from tropnn.layers.surrogate import ste_heaviside

CoefficientMode = Literal["float", "ternary"]
TowerSharing = Literal["shared", "separate"]

HISTORICAL_AGGREGATION_OOD = 0.2958984375
HISTORICAL_VALUE_OOD_COSINE = 0.7638487815856934
AGGREGATION_GO_GATE = 0.60
VALUE_GO_GATE = 0.90


@dataclass
class ProbeConfig:
    dim: int = 32
    train_classes: int = 64
    ood_classes: int = 32
    train_pairs: int = 8192
    test_pairs: int = 2048
    seq_train: int = 4096
    seq_test: int = 1024
    value_train: int = 8192
    value_test: int = 2048
    candidates: int = 8
    value_classes: int = 16
    out_dim: int = 16
    noise: float = 0.15
    tables: int = 16
    comparisons: int = 4
    rank: int = 16
    steps: int = 200
    batch_size: int = 256
    lr: float = 3e-3
    lift_scale: float = 0.25
    ternary_threshold: float = 0.5
    seed: int = 0
    device: str = "auto"
    output_dir: str = "python/results/qkv_toys/chamber_lifting_go_no_go"

    def __post_init__(self) -> None:
        if self.dim < 4 or self.dim % 4 != 0:
            raise ValueError("dim must be divisible by four")
        if self.dim != 2 * self.out_dim:
            raise ValueError("the fixed value readout requires dim == 2 * out_dim")
        if min(self.train_classes, self.ood_classes, self.tables, self.comparisons, self.steps, self.batch_size) <= 0:
            raise ValueError("class counts, table shape, steps, and batch size must be positive")


@dataclass(frozen=True)
class ProbeDatasets:
    aggregation_train: tuple[Tensor, Tensor, Tensor, Tensor]
    aggregation_test: tuple[Tensor, Tensor, Tensor, Tensor]
    aggregation_ood: tuple[Tensor, Tensor, Tensor, Tensor]
    value_train: tuple[Tensor, Tensor]
    value_test: tuple[Tensor, Tensor]
    value_ood: tuple[Tensor, Tensor]


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sample_class_vectors(prototypes: Tensor, labels: Tensor, noise: float) -> Tensor:
    return prototypes.index_select(0, labels) + torch.randn(*labels.shape, prototypes.shape[-1], device=prototypes.device) * noise


def _make_pair_dataset(prototypes: Tensor, n_pairs: int, noise: float) -> tuple[Tensor, Tensor, Tensor]:
    """Generate and discard the historical compatibility data to preserve RNG order."""

    n_classes = prototypes.shape[0]
    labels = torch.arange(n_pairs, device=prototypes.device) % 2
    class_a = torch.randint(0, n_classes, (n_pairs,), device=prototypes.device)
    class_b = torch.randint(0, n_classes, (n_pairs,), device=prototypes.device)
    class_b = torch.where(labels.bool(), class_a, (class_a + 1 + class_b % max(1, n_classes - 1)) % n_classes)
    return (
        _sample_class_vectors(prototypes, class_a, noise),
        _sample_class_vectors(prototypes, class_b, noise),
        labels.float(),
    )


def _make_aggregation_dataset(
    prototypes: Tensor,
    n_items: int,
    *,
    candidates: int,
    value_classes: int,
    noise: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    n_classes = prototypes.shape[0]
    query_class = torch.randint(0, n_classes, (n_items,), device=prototypes.device)
    target_position = torch.randint(0, candidates, (n_items,), device=prototypes.device)
    candidate_class = torch.randint(0, n_classes, (n_items, candidates), device=prototypes.device)
    candidate_class = torch.where(
        torch.arange(candidates, device=prototypes.device).view(1, -1) == target_position.view(-1, 1),
        query_class.view(-1, 1),
        candidate_class,
    )
    candidate_class = torch.where(candidate_class == query_class.view(-1, 1), (candidate_class + 1) % n_classes, candidate_class)
    candidate_class.scatter_(1, target_position.view(-1, 1), query_class.view(-1, 1))

    query = _sample_class_vectors(prototypes, query_class, noise)
    candidate = _sample_class_vectors(prototypes, candidate_class.reshape(-1), noise).view(n_items, candidates, -1)
    value_ids = torch.randint(0, value_classes, (n_items, candidates), device=prototypes.device)
    target_value = value_ids.gather(1, target_position.view(-1, 1)).squeeze(1)
    values = F.one_hot(value_ids, num_classes=value_classes).float()
    return query, candidate, values, target_value


def _make_value_dataset(prototypes: Tensor, teacher: Tensor, n_items: int, noise: float) -> tuple[Tensor, Tensor]:
    labels = torch.randint(0, prototypes.shape[0], (n_items,), device=prototypes.device)
    x = _sample_class_vectors(prototypes, labels, noise)
    return x, torch.tanh(x @ teacher)


def make_datasets(config: ProbeConfig, device: torch.device) -> ProbeDatasets:
    """Reproduce the data-generation order of the archived QKV toy."""

    _set_seed(config.seed)
    train_prototypes = F.normalize(torch.randn(config.train_classes, config.dim, device=device), dim=-1)
    ood_prototypes = F.normalize(torch.randn(config.ood_classes, config.dim, device=device), dim=-1)
    teacher = torch.randn(config.dim, config.out_dim, device=device) / math.sqrt(config.dim)

    # The archived run generated compatibility pairs before the two target
    # datasets. Keep those RNG draws so the matched baseline remains comparable.
    _make_pair_dataset(train_prototypes, config.train_pairs, config.noise)
    _make_pair_dataset(train_prototypes, config.test_pairs, config.noise)
    _make_pair_dataset(ood_prototypes, config.test_pairs, config.noise)

    return ProbeDatasets(
        aggregation_train=_make_aggregation_dataset(
            train_prototypes,
            config.seq_train,
            candidates=config.candidates,
            value_classes=config.value_classes,
            noise=config.noise,
        ),
        aggregation_test=_make_aggregation_dataset(
            train_prototypes,
            config.seq_test,
            candidates=config.candidates,
            value_classes=config.value_classes,
            noise=config.noise,
        ),
        aggregation_ood=_make_aggregation_dataset(
            ood_prototypes,
            config.seq_test,
            candidates=config.candidates,
            value_classes=config.value_classes,
            noise=config.noise,
        ),
        value_train=_make_value_dataset(train_prototypes, teacher, config.value_train, config.noise),
        value_test=_make_value_dataset(train_prototypes, teacher, config.value_test, config.noise),
        value_ood=_make_value_dataset(ood_prototypes, teacher, config.value_test, config.noise),
    )


class PairwiseHash(nn.Module):
    """The exact independent query/key hash used by the historical toy."""

    def __init__(self, dim: int, tables: int, comparisons: int, *, seed: int) -> None:
        super().__init__()
        self.tables = int(tables)
        self.comparisons = int(comparisons)
        self.cells = 1 << comparisons
        generator = torch.Generator(device="cpu").manual_seed(seed)
        anchors = torch.zeros(tables, comparisons, 2, dtype=torch.long)
        for table in range(tables):
            for comparison in range(comparisons):
                a = int(torch.randint(0, dim, (1,), generator=generator).item())
                b = int(torch.randint(0, dim, (1,), generator=generator).item())
                while a == b:
                    b = int(torch.randint(0, dim, (1,), generator=generator).item())
                anchors[table, comparison] = torch.tensor((a, b))
        self.register_buffer("anchors", anchors)
        self.register_buffer("powers", 2 ** torch.arange(comparisons, dtype=torch.long))
        self.thresholds = nn.Parameter(torch.zeros(tables, comparisons))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        flat = x.reshape(-1, x.shape[-1])
        a = flat[:, self.anchors[..., 0].reshape(-1)].view(flat.shape[0], self.tables, self.comparisons)
        b = flat[:, self.anchors[..., 1].reshape(-1)].view(flat.shape[0], self.tables, self.comparisons)
        margins = a - b - self.thresholds.to(device=flat.device, dtype=flat.dtype)
        indices = ((margins > 0).long() * self.powers.to(flat.device).view(1, 1, -1)).sum(dim=-1)
        min_bit = margins.abs().argmin(dim=-1)
        min_margin = margins.gather(-1, min_bit.unsqueeze(-1)).squeeze(-1)
        shape = (*x.shape[:-1], self.tables)
        return indices.view(shape), min_bit.view(shape), min_margin.view(shape)


class PairwiseScoreCore(nn.Module):
    """Free same-table relation score over independent comparison hashes."""

    def __init__(self, dim: int, tables: int, comparisons: int, *, seed: int) -> None:
        super().__init__()
        self.tables = int(tables)
        self.cells = 1 << comparisons
        self.q_hash = PairwiseHash(dim, tables, comparisons, seed=seed)
        self.k_hash = PairwiseHash(dim, tables, comparisons, seed=seed + 1009)
        generator = torch.Generator(device="cpu").manual_seed(seed + 2027)
        self.score = nn.Parameter(torch.randn(tables, self.cells, self.cells, generator=generator) * 0.02)

    def _lookup(self, q_index: Tensor, k_index: Tensor) -> Tensor:
        flat_q = q_index.reshape(-1, self.tables)
        flat_k = k_index.reshape(-1, self.tables)
        table_index = torch.arange(self.tables, device=q_index.device).view(1, self.tables)
        values = self.score[table_index, flat_q, flat_k]
        return (values.sum(dim=-1) / math.sqrt(self.tables)).view(q_index.shape[:-1])

    def forward(self, query: Tensor, key: Tensor) -> Tensor:
        q_index, q_bit, q_margin = self.q_hash(query)
        k_index, k_bit, k_margin = self.k_hash(key)
        output = self._lookup(q_index, k_index)
        if self.training:
            q_neighbor = q_index ^ (2**q_bit).long()
            k_neighbor = k_index ^ (2**k_bit).long()
            q_delta = self._lookup(q_neighbor, k_index) - output
            k_delta = self._lookup(q_index, k_neighbor) - output
            q_ste = (
                ste_heaviside(q_margin, "fast_sigmoid_odd") - (q_margin > 0).to(q_margin.dtype)
            ).sum(dim=-1) / math.sqrt(self.tables)
            k_ste = (
                ste_heaviside(k_margin, "fast_sigmoid_odd") - (k_margin > 0).to(k_margin.dtype)
            ).sum(dim=-1) / math.sqrt(self.tables)
            output = output + q_ste * q_delta.detach() + k_ste * k_delta.detach()
        return output


class RawPairwiseCandidateScorer(nn.Module):
    def __init__(self, config: ProbeConfig) -> None:
        super().__init__()
        self.core = PairwiseScoreCore(config.dim, config.tables, config.comparisons, seed=config.seed)

    def forward(self, query: Tensor, candidates: Tensor) -> Tensor:
        batch, count, dim = candidates.shape
        q = query[:, None, :].expand(-1, count, -1).reshape(batch * count, dim)
        k = candidates.reshape(batch * count, dim)
        return self.core(q, k).view(batch, count)


class DenseCandidateScorer(nn.Module):
    def __init__(self, config: ProbeConfig) -> None:
        super().__init__()
        self.q = nn.Linear(config.dim, config.rank, bias=False)
        self.k = nn.Linear(config.dim, config.rank, bias=False)

    def forward(self, query: Tensor, candidates: Tensor) -> Tensor:
        q = self.q(query)
        k = self.k(candidates)
        return (q[:, None, :] * k).sum(dim=-1) / math.sqrt(q.shape[-1])


class LiftingPairwiseCandidateScorer(nn.Module):
    def __init__(
        self,
        config: ProbeConfig,
        *,
        depth: int,
        coefficient_mode: CoefficientMode,
        tower_sharing: TowerSharing,
    ) -> None:
        super().__init__()
        self.tower_sharing = tower_sharing
        self.q_tower = ChamberLiftingTower(
            config.dim,
            depth=depth,
            coefficient_mode=coefficient_mode,
            lift_scale=config.lift_scale,
            ternary_threshold=config.ternary_threshold,
            seed=config.seed + 3001,
        )
        self.k_tower = (
            self.q_tower
            if tower_sharing == "shared"
            else ChamberLiftingTower(
                config.dim,
                depth=depth,
                coefficient_mode=coefficient_mode,
                lift_scale=config.lift_scale,
                ternary_threshold=config.ternary_threshold,
                seed=config.seed + 4001,
            )
        )
        self.core = PairwiseScoreCore(config.dim, config.tables, config.comparisons, seed=config.seed)

    def forward(self, query: Tensor, candidates: Tensor) -> Tensor:
        batch, count, dim = candidates.shape
        q = self.q_tower(query)
        k = self.k_tower(candidates.reshape(batch * count, dim))
        q = q[:, None, :].expand(-1, count, -1).reshape(batch * count, dim)
        return self.core(q, k).view(batch, count)

    def lifting_stats(self) -> dict[str, float]:
        q_min, q_max = self.q_tower.receptive_field_sizes()
        k_min, k_max = self.k_tower.receptive_field_sizes()
        nonzero = self.q_tower.ternary_nonzero_fraction()
        if self.k_tower is not self.q_tower and self.q_tower.coefficient_mode == "ternary":
            nonzero = 0.5 * (nonzero + self.k_tower.ternary_nonzero_fraction())
        return {
            "receptive_field_min": float(min(q_min, k_min)),
            "receptive_field_max": float(max(q_max, k_max)),
            "ternary_nonzero_fraction": nonzero,
            "active_operator_reads_per_qk": float(
                self.q_tower.active_operator_reads_per_item + self.k_tower.active_operator_reads_per_item
            ),
            "integer_adds_per_qk": float(self.q_tower.integer_adds_per_item + self.k_tower.integer_adds_per_item),
        }


class DenseValue(nn.Module):
    def __init__(self, config: ProbeConfig) -> None:
        super().__init__()
        self.linear = nn.Linear(config.dim, config.out_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


class OrdinaryLutValue(nn.Module):
    def __init__(self, config: ProbeConfig) -> None:
        super().__init__()
        self.layer = PairwiseLUT(
            config.dim,
            config.out_dim,
            tables=config.tables,
            comparisons=config.comparisons,
            backend="torch",
            seed=config.seed,
            lut_init_std=0.0,
            lut_dtype="fp32",
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x.unsqueeze(1)).squeeze(1)


class LiftingValue(nn.Module):
    def __init__(self, config: ProbeConfig, *, depth: int, coefficient_mode: CoefficientMode) -> None:
        super().__init__()
        self.tower = ChamberLiftingTower(
            config.dim,
            depth=depth,
            coefficient_mode=coefficient_mode,
            lift_scale=config.lift_scale,
            ternary_threshold=config.ternary_threshold,
            seed=config.seed + 5003,
        )
        self.out_dim = config.out_dim

    def forward(self, x: Tensor) -> Tensor:
        mixed = self.tower(x)
        # Fixed two-way fold: all learned cross-channel interaction remains in
        # the chamber lifting tower.
        return (mixed[..., : self.out_dim] + mixed[..., self.out_dim :]) / math.sqrt(2.0)

    def lifting_stats(self) -> dict[str, float]:
        receptive_min, receptive_max = self.tower.receptive_field_sizes()
        return {
            "receptive_field_min": float(receptive_min),
            "receptive_field_max": float(receptive_max),
            "ternary_nonzero_fraction": self.tower.ternary_nonzero_fraction(),
            "active_operator_reads": float(self.tower.active_operator_reads_per_item),
            "integer_adds": float(self.tower.integer_adds_per_item),
        }


def _batch_indices(items: int, batch_size: int, device: torch.device) -> Tensor:
    return torch.randint(0, items, (batch_size,), device=device)


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def train_aggregation(
    model: nn.Module,
    config: ProbeConfig,
    train_data: tuple[Tensor, Tensor, Tensor, Tensor],
    test_data: tuple[Tensor, Tensor, Tensor, Tensor],
    ood_data: tuple[Tensor, Tensor, Tensor, Tensor],
) -> dict[str, float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    query, candidates, values, target = train_data
    start = time.time()
    model.train()
    for _ in range(config.steps):
        index = _batch_indices(query.shape[0], config.batch_size, query.device)
        scores = model(query[index], candidates[index])
        weights = (scores / 0.5).softmax(dim=-1)
        logits = (weights.unsqueeze(-1) * values[index]).sum(dim=1).clamp_min(1e-8).log()
        loss = F.cross_entropy(logits, target[index])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    train_seconds = time.time() - start

    def evaluate(data: tuple[Tensor, Tensor, Tensor, Tensor], prefix: str) -> dict[str, float]:
        model.eval()
        with torch.no_grad():
            scores = model(data[0], data[1])
            weights = (scores / 0.5).softmax(dim=-1)
            logits = (weights.unsqueeze(-1) * data[2]).sum(dim=1).clamp_min(1e-8).log()
            prediction = logits.argmax(dim=-1)
            score_value = data[2].argmax(dim=-1).gather(1, scores.argmax(dim=-1, keepdim=True)).squeeze(1)
            top2 = scores.topk(2, dim=-1).values
            return {
                f"{prefix}_loss": float(F.cross_entropy(logits, data[3]).item()),
                f"{prefix}_value_acc": float((prediction == data[3]).float().mean().item()),
                f"{prefix}_score_top1_value_acc": float((score_value == data[3]).float().mean().item()),
                f"{prefix}_score_margin": float((top2[:, 0] - top2[:, 1]).mean().item()),
            }

    metrics = {"train_seconds": train_seconds, "params": float(_parameter_count(model))}
    metrics.update(evaluate(test_data, "test"))
    metrics.update(evaluate(ood_data, "ood"))
    return metrics


def train_value(
    model: nn.Module,
    config: ProbeConfig,
    train_data: tuple[Tensor, Tensor],
    test_data: tuple[Tensor, Tensor],
    ood_data: tuple[Tensor, Tensor],
) -> dict[str, float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    x, target = train_data
    start = time.time()
    model.train()
    for _ in range(config.steps):
        index = _batch_indices(x.shape[0], config.batch_size, x.device)
        prediction = model(x[index])
        loss = F.mse_loss(prediction, target[index])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    train_seconds = time.time() - start

    def evaluate(data: tuple[Tensor, Tensor], prefix: str) -> dict[str, float]:
        model.eval()
        with torch.no_grad():
            prediction = model(data[0])
            mse = F.mse_loss(prediction, data[1])
            cosine = F.cosine_similarity(prediction, data[1], dim=-1).mean()
            relative = (prediction - data[1]).square().sum().sqrt() / data[1].square().sum().sqrt().clamp_min(1e-8)
            return {
                f"{prefix}_mse": float(mse.item()),
                f"{prefix}_cosine": float(cosine.item()),
                f"{prefix}_rel_error": float(relative.item()),
            }

    metrics = {"train_seconds": train_seconds, "params": float(_parameter_count(model))}
    metrics.update(evaluate(test_data, "test"))
    metrics.update(evaluate(ood_data, "ood"))
    return metrics


def _model_seed(config: ProbeConfig, offset: int) -> None:
    _set_seed(config.seed + offset)


def run_seed(
    config: ProbeConfig,
    *,
    depths: list[int],
    coefficient_modes: list[CoefficientMode],
    tower_sharings: list[TowerSharing],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    device = _device(config.device)
    datasets = make_datasets(config, device)
    rows: list[dict[str, object]] = []
    decisions: list[dict[str, object]] = []

    _model_seed(config, 11)
    dense_score = DenseCandidateScorer(config).to(device)
    dense_aggregation = train_aggregation(
        dense_score,
        config,
        datasets.aggregation_train,
        datasets.aggregation_test,
        datasets.aggregation_ood,
    )
    rows.append({"toy": "selective_aggregation", "family": "dense_qk", "seed": config.seed, **dense_aggregation})

    _model_seed(config, 13)
    raw_score = RawPairwiseCandidateScorer(config).to(device)
    raw_aggregation = train_aggregation(
        raw_score,
        config,
        datasets.aggregation_train,
        datasets.aggregation_test,
        datasets.aggregation_ood,
    )
    rows.append({"toy": "selective_aggregation", "family": "pairwise_score", "seed": config.seed, **raw_aggregation})

    _model_seed(config, 17)
    dense_value = DenseValue(config).to(device)
    dense_value_metrics = train_value(dense_value, config, datasets.value_train, datasets.value_test, datasets.value_ood)
    rows.append({"toy": "value_transform", "family": "dense_wv", "seed": config.seed, **dense_value_metrics})

    _model_seed(config, 19)
    ordinary_value = OrdinaryLutValue(config).to(device)
    ordinary_value_metrics = train_value(
        ordinary_value,
        config,
        datasets.value_train,
        datasets.value_test,
        datasets.value_ood,
    )
    rows.append({"toy": "value_transform", "family": "lut_value", "seed": config.seed, **ordinary_value_metrics})

    for depth in depths:
        for coefficient_mode in coefficient_modes:
            variant = f"lifting_d{depth}_{coefficient_mode}"

            _model_seed(config, 101 + 7 * depth + (0 if coefficient_mode == "float" else 1))
            lifting_value = LiftingValue(config, depth=depth, coefficient_mode=coefficient_mode).to(device)
            value_metrics = train_value(
                lifting_value,
                config,
                datasets.value_train,
                datasets.value_test,
                datasets.value_ood,
            )
            value_stats = lifting_value.lifting_stats()
            rows.append(
                {
                    "toy": "value_transform",
                    "family": variant,
                    "seed": config.seed,
                    "depth": depth,
                    "coefficient_mode": coefficient_mode,
                    **value_metrics,
                    **value_stats,
                }
            )

            for tower_sharing in tower_sharings:
                family = f"{variant}_{tower_sharing}"
                _model_seed(config, 211 + 11 * depth + (0 if coefficient_mode == "float" else 1))
                lifting_score = LiftingPairwiseCandidateScorer(
                    config,
                    depth=depth,
                    coefficient_mode=coefficient_mode,
                    tower_sharing=tower_sharing,
                ).to(device)
                aggregation_metrics = train_aggregation(
                    lifting_score,
                    config,
                    datasets.aggregation_train,
                    datasets.aggregation_test,
                    datasets.aggregation_ood,
                )
                score_stats = lifting_score.lifting_stats()
                rows.append(
                    {
                        "toy": "selective_aggregation",
                        "family": family,
                        "seed": config.seed,
                        "depth": depth,
                        "coefficient_mode": coefficient_mode,
                        "tower_sharing": tower_sharing,
                        **aggregation_metrics,
                        **score_stats,
                    }
                )
                aggregation_pass = float(aggregation_metrics["ood_value_acc"]) >= AGGREGATION_GO_GATE
                value_pass = float(value_metrics["ood_cosine"]) >= VALUE_GO_GATE
                decisions.append(
                    {
                        "variant": family,
                        "value_variant": variant,
                        "seed": config.seed,
                        "aggregation_ood_value_acc": aggregation_metrics["ood_value_acc"],
                        "value_ood_cosine": value_metrics["ood_cosine"],
                        "aggregation_pass": aggregation_pass,
                        "value_pass": value_pass,
                        "joint_pass": aggregation_pass and value_pass,
                    }
                )

    return rows, decisions


def summarize_decisions(decisions: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for decision in decisions:
        grouped.setdefault(str(decision["variant"]), []).append(decision)
    summary: list[dict[str, object]] = []
    for variant, items in sorted(grouped.items()):
        aggregation = torch.tensor([float(item["aggregation_ood_value_acc"]) for item in items])
        value = torch.tensor([float(item["value_ood_cosine"]) for item in items])
        summary.append(
            {
                "variant": variant,
                "seeds": len(items),
                "aggregation_ood_mean": float(aggregation.mean().item()),
                "aggregation_ood_min": float(aggregation.min().item()),
                "value_ood_cosine_mean": float(value.mean().item()),
                "value_ood_cosine_min": float(value.min().item()),
                "mean_gate_pass": bool(aggregation.mean() >= AGGREGATION_GO_GATE and value.mean() >= VALUE_GO_GATE),
                "all_seed_gate_pass": bool(aggregation.min() >= AGGREGATION_GO_GATE and value.min() >= VALUE_GO_GATE),
            }
        )
    return summary


def _write_outputs(
    output_dir: Path,
    config: ProbeConfig,
    rows: list[dict[str, object]],
    decisions: list[dict[str, object]],
    decision_summary: list[dict[str, object]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with (output_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(
            {
                "config": asdict(config),
                "historical_reference": {
                    "aggregation_ood_value_acc": HISTORICAL_AGGREGATION_OOD,
                    "value_ood_cosine": HISTORICAL_VALUE_OOD_COSINE,
                },
                "gates": {
                    "aggregation_ood_value_acc": AGGREGATION_GO_GATE,
                    "value_ood_cosine": VALUE_GO_GATE,
                },
                "rows": rows,
                "decisions": decisions,
                "decision_summary": decision_summary,
            },
            handle,
            indent=2,
        )


def _parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def _parse_modes(value: str) -> list[CoefficientMode]:
    modes = [item.strip() for item in value.split(",") if item.strip()]
    if any(mode not in {"float", "ternary"} for mode in modes):
        raise ValueError("coefficient modes must be float or ternary")
    return modes  # type: ignore[return-value]


def _parse_sharing(value: str) -> list[TowerSharing]:
    modes = [item.strip() for item in value.split(",") if item.strip()]
    if any(mode not in {"shared", "separate"} for mode in modes):
        raise ValueError("tower sharing modes must be shared or separate")
    return modes  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depths", default="2,4,6")
    parser.add_argument("--coefficient-modes", default="float,ternary")
    parser.add_argument("--tower-sharings", default="shared,separate")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--steps", type=int, default=ProbeConfig.steps)
    parser.add_argument("--batch-size", type=int, default=ProbeConfig.batch_size)
    parser.add_argument("--lr", type=float, default=ProbeConfig.lr)
    parser.add_argument("--lift-scale", type=float, default=ProbeConfig.lift_scale)
    parser.add_argument("--ternary-threshold", type=float, default=ProbeConfig.ternary_threshold)
    parser.add_argument("--device", default=ProbeConfig.device)
    parser.add_argument("--output-dir", default=ProbeConfig.output_dir)
    args = parser.parse_args()

    depths = _parse_ints(args.depths)
    coefficient_modes = _parse_modes(args.coefficient_modes)
    tower_sharings = _parse_sharing(args.tower_sharings)
    seeds = _parse_ints(args.seeds)
    rows: list[dict[str, object]] = []
    decisions: list[dict[str, object]] = []
    output_dir = Path(args.output_dir)
    final_config: ProbeConfig | None = None
    for seed in seeds:
        config = ProbeConfig(
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            lift_scale=args.lift_scale,
            ternary_threshold=args.ternary_threshold,
            seed=seed,
            device=args.device,
            output_dir=str(output_dir),
        )
        final_config = config
        seed_rows, seed_decisions = run_seed(
            config,
            depths=depths,
            coefficient_modes=coefficient_modes,
            tower_sharings=tower_sharings,
        )
        rows.extend(seed_rows)
        decisions.extend(seed_decisions)
        for row in seed_rows:
            print(json.dumps(row, sort_keys=True))

    assert final_config is not None
    decision_summary = summarize_decisions(decisions)
    _write_outputs(output_dir, final_config, rows, decisions, decision_summary)
    print(json.dumps({"decision_summary": decision_summary}, sort_keys=True))
    print(f"wrote {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
