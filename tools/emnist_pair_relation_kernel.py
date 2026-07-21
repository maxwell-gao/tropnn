"""End-to-end real-pair tasks for Global Coxeter and root-incidence kernels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tropnn.layers import (
    BalancedS4Router,
    CoxeterPairScorer,
    GlobalChamberKernel,
    IntrinsicS4Kernel,
    PairwiseLUT,
    RootIncidenceKernel,
    SameTableFullKernel,
)
from tropnn.tools.emnist_payload_dtype_sweep import _load_emnist_split
from tropnn.tools.emnist_payload_feature_probe import FeatureProbeLUTLayer

TaskName = Literal["same_class", "digit_greater"]
SplitMode = Literal["object", "class"]
PayloadMode = Literal["float", "binary01"]
ObjectiveName = Literal["relation_only", "relation_aux"]

CLASS_TRAIN = (29, 15, 34, 16, 3, 43, 4, 28, 36, 44, 46, 12, 39, 25, 33, 24, 8, 26, 19, 22, 17, 37, 6, 2, 11, 5, 40, 13, 38, 23)
CLASS_VALID = (31, 30, 9, 14, 10, 21, 1, 42)
CLASS_TEST = (45, 32, 0, 41, 35, 20, 7, 18, 27)


@dataclass(frozen=True)
class PairExperimentConfig:
    task: TaskName
    split_mode: SplitMode
    decoder: str
    payload_mode: PayloadMode
    objective: ObjectiveName
    seed: int
    epochs: int
    batch_size: int
    encoder_lr: float
    relation_lr: float
    auxiliary_weight: float
    relation_dim: int
    relation_tables: int
    relation_coverage: int
    encoder_depth: int
    encoder_tables: int
    encoder_comparisons: int
    train_pairs_per_epoch: int
    eval_pairs: int
    retrieval_queries: int
    retrieval_candidates: int
    retrieval_positives: int
    hard_reservoir: int
    data_split_seed: int
    max_train_examples: int
    max_eval_examples: int


@dataclass(frozen=True)
class TensorSplit:
    images: Tensor
    labels: Tensor


@dataclass(frozen=True)
class TaskSplits:
    train: TensorSplit
    validation: TensorSplit
    test: TensorSplit
    auxiliary_classes: tuple[int, ...]


@dataclass(frozen=True)
class PairIndices:
    query: Tensor
    key: Tensor
    target: Tensor


@dataclass(frozen=True)
class RetrievalSet:
    query: Tensor
    candidates: Tensor
    relevant: Tensor


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _filter_classes(images: Tensor, labels: Tensor, classes: tuple[int, ...]) -> TensorSplit:
    class_tensor = torch.tensor(classes, dtype=labels.dtype)
    mask = (labels[:, None] == class_tensor[None, :]).any(dim=1)
    return TensorSplit(images[mask], labels[mask])


def stratified_half_split(images: Tensor, labels: Tensor, seed: int) -> tuple[TensorSplit, TensorSplit]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    validation: list[Tensor] = []
    test: list[Tensor] = []
    for label in torch.unique(labels, sorted=True):
        indices = torch.nonzero(labels == label, as_tuple=False).flatten()
        indices = indices[torch.randperm(indices.numel(), generator=generator)]
        midpoint = indices.numel() // 2
        validation.append(indices[:midpoint])
        test.append(indices[midpoint:])
    validation_index = torch.cat(validation)
    test_index = torch.cat(test)
    return TensorSplit(images[validation_index], labels[validation_index]), TensorSplit(images[test_index], labels[test_index])


def load_task_splits(data_root: Path, task: TaskName, split_mode: SplitMode, seed: int = 1729) -> TaskSplits:
    if task == "digit_greater" and split_mode != "object":
        raise ValueError("digit_greater supports object holdout only")
    train_images, train_labels = _load_emnist_split(data_root, "balanced", train=True, limit=0, seed=seed)
    test_images, test_labels = _load_emnist_split(data_root, "balanced", train=False, limit=0, seed=seed)
    if task == "digit_greater":
        digit_classes = tuple(range(10))
        train = _filter_classes(train_images, train_labels, digit_classes)
        held = _filter_classes(test_images, test_labels, digit_classes)
        validation, test = stratified_half_split(held.images, held.labels, seed)
        return TaskSplits(train, validation, test, digit_classes)
    if split_mode == "class":
        return TaskSplits(
            _filter_classes(train_images, train_labels, CLASS_TRAIN),
            _filter_classes(test_images, test_labels, CLASS_VALID),
            _filter_classes(test_images, test_labels, CLASS_TEST),
            CLASS_TRAIN,
        )
    validation, test = stratified_half_split(test_images, test_labels, seed)
    classes = tuple(int(value) for value in torch.unique(train_labels, sorted=True).tolist())
    return TaskSplits(TensorSplit(train_images, train_labels), validation, test, classes)


def split_fingerprint(split: TensorSplit) -> str:
    """Stable lightweight identity for labels and selected image contents."""

    digest = hashlib.sha256()
    digest.update(str(tuple(split.images.shape)).encode())
    digest.update(split.labels.contiguous().numpy().tobytes())
    pixel_sum = ((split.images + 1.0) * 127.5).round().to(torch.int64).sum(dim=1)
    digest.update(pixel_sum.contiguous().numpy().tobytes())
    return digest.hexdigest()


def index_fingerprint(*tensors: Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        value = tensor.detach().cpu().contiguous()
        digest.update(str((str(value.dtype), tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def remap_auxiliary_labels(labels: Tensor, classes: tuple[int, ...]) -> Tensor:
    result = torch.full_like(labels, -1)
    for index, label in enumerate(classes):
        result[labels == label] = index
    if bool((result < 0).any()):
        raise ValueError("auxiliary labels contain a class outside the training class map")
    return result


def _class_indices(labels: Tensor) -> tuple[Tensor, tuple[Tensor, ...]]:
    classes = torch.unique(labels, sorted=True)
    groups = tuple(torch.nonzero(labels == label, as_tuple=False).flatten() for label in classes)
    if any(group.numel() < 2 for group in groups):
        raise ValueError("each task class needs at least two objects")
    return classes, groups


def _sample_group_members(groups: tuple[Tensor, ...], class_codes: Tensor, generator: torch.Generator) -> Tensor:
    result = torch.empty(class_codes.shape, dtype=torch.long)
    for class_index, group in enumerate(groups):
        locations = torch.nonzero(class_codes == class_index, as_tuple=False).flatten()
        if locations.numel() == 0:
            continue
        positions = torch.randint(group.numel(), (locations.numel(),), generator=generator)
        result[locations] = group[positions]
    return result


def sample_pair_indices(labels: Tensor, task: TaskName, count: int, seed: int) -> PairIndices:
    if count < 2:
        raise ValueError("pair count must be at least two")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    _classes, groups = _class_indices(labels)
    query_parts: list[Tensor] = []
    key_parts: list[Tensor] = []
    target_parts: list[Tensor] = []
    if task == "same_class":
        positive_count = count // 2
        negative_count = count - positive_count
        positive_class = torch.randint(len(groups), (positive_count,), generator=generator)
        for class_index in range(len(groups)):
            take = torch.nonzero(positive_class == class_index, as_tuple=False).flatten()
            if take.numel() == 0:
                continue
            group = groups[class_index]
            first = torch.randint(group.numel(), (take.numel(),), generator=generator)
            offset = torch.randint(1, group.numel(), (take.numel(),), generator=generator)
            query_parts.append(group[first])
            key_parts.append(group[(first + offset) % group.numel()])
            target_parts.append(torch.ones(take.numel()))
        left_class = torch.randint(len(groups), (negative_count,), generator=generator)
        right_offset = torch.randint(1, len(groups), (negative_count,), generator=generator)
        right_class = (left_class + right_offset) % len(groups)
        query_parts.append(_sample_group_members(groups, left_class, generator))
        key_parts.append(_sample_group_members(groups, right_class, generator))
        target_parts.append(torch.zeros(negative_count))
    else:
        first_class = torch.randint(len(groups), (count,), generator=generator)
        second_offset = torch.randint(1, len(groups), (count,), generator=generator)
        second_class = (first_class + second_offset) % len(groups)
        lower_class = torch.minimum(first_class, second_class)
        upper_class = torch.maximum(first_class, second_class)
        target = torch.cat((torch.ones(count // 2), torch.zeros(count - count // 2)))
        target = target[torch.randperm(count, generator=generator)]
        query_class = torch.where(target.bool(), upper_class, lower_class)
        key_class = torch.where(target.bool(), lower_class, upper_class)
        query_parts.append(_sample_group_members(groups, query_class, generator))
        key_parts.append(_sample_group_members(groups, key_class, generator))
        target_parts.append(target)
    query = torch.cat(query_parts)
    key = torch.cat(key_parts)
    target = torch.cat(target_parts)
    order = torch.randperm(query.numel(), generator=generator)
    return PairIndices(query[order], key[order], target[order])


def sample_training_pair_indices(labels: Tensor, task: TaskName, seed: int) -> PairIndices:
    """Build the preregistered default epoch: two query roles per object.

    For ``same_class`` every training object is used exactly once as the query
    of a positive pair and once as the query of a negative pair.  The digit
    task has no per-object query requirement, so it draws the same total
    number of uniformly sampled, distinct-class pairs.
    """

    if task == "digit_greater":
        return sample_pair_indices(labels, task, 2 * labels.numel(), seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    classes, groups = _class_indices(labels)
    query = torch.arange(labels.numel())
    class_codes = torch.searchsorted(classes, labels)
    positive_key = torch.empty_like(query)
    for group in groups:
        offsets = torch.randint(1, group.numel(), (group.numel(),), generator=generator)
        positive_key[group] = group[(torch.arange(group.numel()) + offsets) % group.numel()]
    negative_offset = torch.randint(1, len(groups), (labels.numel(),), generator=generator)
    negative_class = (class_codes + negative_offset) % len(groups)
    negative_key = _sample_group_members(groups, negative_class, generator)
    query = torch.cat((query, query))
    key = torch.cat((positive_key, negative_key))
    target = torch.cat((torch.ones(labels.numel()), torch.zeros(labels.numel())))
    order = torch.randperm(query.numel(), generator=generator)
    return PairIndices(query[order], key[order], target[order])


def build_retrieval_set(
    split: TensorSplit,
    *,
    queries: int,
    candidates: int,
    positives: int,
    seed: int,
    hard: bool = False,
    hard_reservoir: int = 4096,
) -> RetrievalSet:
    if positives >= candidates:
        raise ValueError("positives must be smaller than candidates")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    query_count = min(queries, split.labels.numel())
    query_index = torch.randperm(split.labels.numel(), generator=generator)[:query_count]
    candidate_rows: list[Tensor] = []
    relevant_rows: list[Tensor] = []
    negative_count = candidates - positives
    normalized_pixels = None
    if hard:
        centered = split.images - split.images.mean(dim=1, keepdim=True)
        normalized_pixels = F.normalize(centered, dim=1)
    all_indices = torch.arange(split.labels.numel())
    for query in query_index:
        same = all_indices[(split.labels == split.labels[query]) & (all_indices != query)]
        different = all_indices[split.labels != split.labels[query]]
        if same.numel() < positives or different.numel() < negative_count:
            raise ValueError("evaluation split is too small for the requested retrieval pool")
        positive_index = same[torch.randperm(same.numel(), generator=generator)[:positives]]
        if hard:
            assert normalized_pixels is not None
            reservoir_size = min(hard_reservoir, different.numel())
            reservoir = different[torch.randperm(different.numel(), generator=generator)[:reservoir_size]]
            similarity = normalized_pixels[reservoir] @ normalized_pixels[query]
            negative_index = reservoir[torch.topk(similarity, k=negative_count).indices]
        else:
            negative_index = different[torch.randperm(different.numel(), generator=generator)[:negative_count]]
        row = torch.cat((positive_index, negative_index))
        order = torch.randperm(row.numel(), generator=generator)
        candidate_rows.append(row[order])
        relevant_rows.append((split.labels[row[order]] == split.labels[query]).to(torch.bool))
    return RetrievalSet(query_index, torch.stack(candidate_rows), torch.stack(relevant_rows))


class PairUnaryEncoder(nn.Module):
    def __init__(
        self,
        *,
        payload_mode: PayloadMode,
        auxiliary_classes: int,
        use_auxiliary: bool,
        depth: int,
        tables: int,
        comparisons: int,
        relation_dim: int,
        seed: int,
    ) -> None:
        super().__init__()

        def layer(output_dim: int, layer_seed: int, init_std: float) -> FeatureProbeLUTLayer:
            return FeatureProbeLUTLayer(
                28 * 28,
                output_dim,
                tables=tables,
                comparisons=comparisons,
                payload_mode=payload_mode,
                anchor_policy="permuted",
                seed=layer_seed,
                lut_init_std=init_std,
                use_output_scaling=True,
                use_min_margin_ste=True,
            )

        self.blocks = nn.ModuleList(layer(28 * 28, seed + 101 * index, 0.0) for index in range(depth))
        coordinate_init = 0.6 if payload_mode == "binary01" else 0.02
        self.coordinate_head = layer(relation_dim, seed + 9001, coordinate_init)
        self.auxiliary_head = layer(auxiliary_classes, seed + 10007, 0.0) if use_auxiliary else None
        self.payload_mode = payload_mode

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor | None]:
        hidden = images.flatten(start_dim=1).float()
        for block in self.blocks:
            write, _ = block.compute(hidden)
            hidden = hidden + write
        coordinates, _ = self.coordinate_head.compute(hidden)
        coordinates = coordinates - coordinates.mean(dim=-1, keepdim=True)
        coordinates = coordinates / coordinates.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-4)
        auxiliary = self.auxiliary_head.compute(hidden)[0] if self.auxiliary_head is not None else None
        return coordinates, auxiliary


class DenseQK(nn.Module):
    def __init__(self, dimension: int, rank: int, seed: int) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(seed + 503)
        self.query = nn.Parameter(torch.randn(dimension, rank, generator=generator) / math.sqrt(dimension))
        self.key = nn.Parameter(torch.randn(dimension, rank, generator=generator) / math.sqrt(dimension))
        self.bias = nn.Parameter(torch.zeros(()))
        self.rank = rank

    def forward(self, query: Tensor, key: Tensor) -> Tensor:
        return ((query @ self.query) * (key @ self.key)).sum(dim=-1) / math.sqrt(self.rank) + self.bias


class ConcatMLP(nn.Module):
    def __init__(self, dimension: int, hidden: int, seed: int) -> None:
        super().__init__()
        with torch.random.fork_rng():
            torch.manual_seed(seed + 601)
            self.first = nn.Linear(2 * dimension, hidden)
            self.second = nn.Linear(hidden, 1)

    def forward(self, query: Tensor, key: Tensor) -> Tensor:
        return self.second(F.gelu(self.first(torch.cat((query, key), dim=-1)))).squeeze(-1)


class JointPairLUT(nn.Module):
    def __init__(self, dimension: int, tables: int, seed: int) -> None:
        super().__init__()
        self.layer = PairwiseLUT(
            2 * dimension,
            1,
            tables=tables,
            comparisons=6,
            backend="torch",
            anchor_policy="random_no_replace",
            seed=seed + 701,
            lut_init_std=0.02,
            lut_dtype="fp32",
        )

    def forward(self, query: Tensor, key: Tensor) -> Tensor:
        return self.layer(torch.cat((query, key), dim=-1)).squeeze(1).squeeze(-1)


def parse_decoder_name(name: str) -> tuple[str, int | None]:
    prefixes = {
        "global_free_r": "global_free",
        "global_coxeter_r": "global_coxeter",
        "dense_qk_r": "dense_qk",
        "jointpair_t": "jointpair_t",
    }
    for prefix, kind in prefixes.items():
        if name.startswith(prefix):
            return kind, int(name[len(prefix) :])
    if name == "concat_mlp":
        return name, 128
    return name, None


class PairRelationModel(nn.Module):
    def __init__(self, config: PairExperimentConfig, auxiliary_classes: int) -> None:
        super().__init__()
        self.config = config
        self.encoder = PairUnaryEncoder(
            payload_mode=config.payload_mode,
            auxiliary_classes=auxiliary_classes,
            use_auxiliary=config.objective == "relation_aux",
            depth=config.encoder_depth,
            tables=config.encoder_tables,
            comparisons=config.encoder_comparisons,
            relation_dim=config.relation_dim,
            seed=config.seed,
        )
        kind, value = parse_decoder_name(config.decoder)
        symmetry = "symmetric" if config.task == "same_class" else "antisymmetric"
        self.router: BalancedS4Router | None = None
        if kind in {"kendall", "mallows", "root_diagonal", "root_incidence", "same_table_full", "global_free", "global_coxeter"}:
            self.router = BalancedS4Router(
                config.relation_dim,
                config.relation_tables,
                coverage=config.relation_coverage,
                seed=config.seed,
            )
            if kind in {"kendall", "mallows"}:
                kernel = IntrinsicS4Kernel(config.relation_tables, kind)
            elif kind == "root_diagonal":
                kernel = RootIncidenceKernel(self.router, diagonal=True, seed=config.seed)
            elif kind == "root_incidence":
                kernel = RootIncidenceKernel(self.router, seed=config.seed)
            elif kind == "same_table_full":
                kernel = SameTableFullKernel(config.relation_tables, seed=config.seed)
            elif kind == "global_free":
                assert value is not None
                kernel = GlobalChamberKernel(config.relation_tables, value, seed=config.seed)
            else:
                assert value is not None
                kernel = GlobalChamberKernel(config.relation_tables, value, shared_coxeter=True, seed=config.seed)
            self.relation = CoxeterPairScorer(self.router, kernel, symmetry=symmetry)
        elif kind == "dense_qk":
            assert value is not None
            self.relation = DenseQK(config.relation_dim, value, config.seed)
        elif kind == "jointpair_t":
            assert value is not None
            self.relation = JointPairLUT(config.relation_dim, value, config.seed)
        elif kind == "concat_mlp":
            assert value is not None
            self.relation = ConcatMLP(config.relation_dim, value, config.seed)
        else:
            raise ValueError(f"unsupported decoder {config.decoder!r}")
        self.symmetry = symmetry

    def directed_score(self, query: Tensor, key: Tensor) -> Tensor:
        return self.relation(query, key)

    def score(self, query: Tensor, key: Tensor) -> Tensor:
        if isinstance(self.relation, CoxeterPairScorer):
            return self.relation(query, key)
        forward = self.directed_score(query, key)
        reverse = self.directed_score(key, query)
        if self.symmetry == "symmetric":
            return 0.5 * (forward + reverse)
        return 0.5 * (forward - reverse)

    @property
    def relation_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.relation.parameters())

    @property
    def encoder_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.encoder.parameters())


def binary_payload_is_valid(model: PairRelationModel) -> bool:
    if model.config.payload_mode != "binary01":
        return True
    for module in model.encoder.modules():
        if isinstance(module, FeatureProbeLUTLayer):
            values = module.materialized_lut().detach()
            if not bool(((values == 0.0) | (values == 1.0)).all()):
                return False
    return True


def encode_unique_pair_batch(
    model: PairRelationModel,
    images: Tensor,
    pairs: PairIndices,
    start: int,
    stop: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None, Tensor]:
    query = pairs.query[start:stop]
    key = pairs.key[start:stop]
    unique, inverse = torch.unique(torch.cat((query, key)), sorted=True, return_inverse=True)
    coordinates, auxiliary = model.encoder(images[unique].to(device, non_blocking=True))
    size = query.numel()
    query_aux = auxiliary[inverse[:size]] if auxiliary is not None else None
    key_aux = auxiliary[inverse[size:]] if auxiliary is not None else None
    return coordinates[inverse[:size]], coordinates[inverse[size:]], query_aux, key_aux, unique


def roc_auc(target: Tensor, prediction: Tensor) -> float:
    positive = prediction[target > 0.5]
    negative = prediction[target <= 0.5]
    if positive.numel() == 0 or negative.numel() == 0:
        return math.nan
    total = 0.0
    comparisons = 0
    for start in range(0, positive.numel(), 1024):
        difference = positive[start : start + 1024, None] - negative[None, :]
        total += float((difference > 0).sum().item()) + 0.5 * float((difference == 0).sum().item())
        comparisons += difference.numel()
    return total / comparisons


def average_precision(target: Tensor, prediction: Tensor) -> float:
    order = torch.argsort(prediction, descending=True)
    sorted_target = target[order]
    precision = sorted_target.cumsum(0) / torch.arange(1, target.numel() + 1, device=target.device)
    return float((precision * sorted_target).sum().item() / sorted_target.sum().clamp_min(1.0).item())


def pair_metrics(target: Tensor, prediction: Tensor) -> dict[str, float]:
    target = target.float().cpu()
    prediction = prediction.float().cpu()
    return {
        "bce": float(F.binary_cross_entropy_with_logits(prediction, target).item()),
        "roc_auc": roc_auc(target, prediction),
        "pr_auc": average_precision(target, prediction),
        "accuracy": float(((prediction > 0) == (target > 0.5)).float().mean().item()),
    }


def digit_relation_metrics(labels: Tensor, pairs: PairIndices, prediction: Tensor) -> dict[str, float]:
    query_label = labels[pairs.query].cpu()
    key_label = labels[pairs.key].cpu()
    target = pairs.target.cpu()
    prediction = prediction.cpu()
    per_query_auc: list[float] = []
    for label in torch.unique(query_label, sorted=True):
        mask = query_label == label
        values = target[mask]
        if bool((values == 0).any()) and bool((values == 1).any()):
            per_query_auc.append(roc_auc(values, prediction[mask]))
    adjacent = (query_label - key_label).abs() == 1
    return {
        "macro_roc_auc": statistics.mean(per_query_auc) if per_query_auc else math.nan,
        "adjacent_accuracy": float(((prediction[adjacent] > 0) == (target[adjacent] > 0.5)).float().mean().item()),
    }


def retrieval_metrics(relevant: Tensor, prediction: Tensor, top_k: int = 16) -> dict[str, float]:
    relevant = relevant.bool().cpu()
    prediction = prediction.float().cpu()
    k = min(top_k, prediction.shape[1])
    order = torch.argsort(prediction, dim=1, descending=True)
    top = relevant.gather(1, order[:, :k])
    recall = top.sum(dim=1).float() / relevant.sum(dim=1).clamp_min(1)
    ranked = relevant.gather(1, order)
    first = ranked.float().argmax(dim=1) + 1
    hit = top[:, 0].float()
    discounts = 1.0 / torch.log2(torch.arange(k, dtype=torch.float32) + 2.0)
    dcg = (top.float() * discounts).sum(dim=1)
    ideal_count = relevant.sum(dim=1).clamp_max(k)
    idcg = torch.stack([discounts[: int(count)].sum() for count in ideal_count])
    return {
        "recall_at_16": float(recall.mean().item()),
        "hit_at_1": float(hit.mean().item()),
        "mrr": float((1.0 / first.float()).mean().item()),
        "ndcg_at_16": float((dcg / idcg.clamp_min(1e-12)).mean().item()),
    }


@torch.no_grad()
def encode_split(model: PairRelationModel, split: TensorSplit, device: torch.device, batch_size: int) -> tuple[Tensor, Tensor | None]:
    model.eval()
    coordinates: list[Tensor] = []
    auxiliary: list[Tensor] = []
    for start in range(0, split.labels.numel(), batch_size):
        coordinate, logits = model.encoder(split.images[start : start + batch_size].to(device))
        coordinates.append(coordinate.cpu())
        if logits is not None:
            auxiliary.append(logits.cpu())
    return torch.cat(coordinates), torch.cat(auxiliary) if auxiliary else None


@torch.no_grad()
def score_coordinate_pairs(model: PairRelationModel, query: Tensor, key: Tensor, device: torch.device, batch_size: int) -> Tensor:
    model.eval()
    rows: list[Tensor] = []
    for start in range(0, query.shape[0], batch_size):
        rows.append(model.score(query[start : start + batch_size].to(device), key[start : start + batch_size].to(device)).cpu())
    return torch.cat(rows)


@torch.no_grad()
def evaluate_model(
    model: PairRelationModel,
    split: TensorSplit,
    task: TaskName,
    device: torch.device,
    *,
    seed: int,
    batch_size: int,
    eval_pairs: int,
    retrieval_queries: int,
    retrieval_candidates: int,
    retrieval_positives: int,
    hard_reservoir: int,
    include_hard: bool = True,
    fingerprint_sink: dict[str, str] | None = None,
    fingerprint_prefix: str = "",
) -> dict[str, float]:
    coordinates, _ = encode_split(model, split, device, batch_size)
    pairs = sample_pair_indices(split.labels, task, eval_pairs, seed)
    if fingerprint_sink is not None:
        fingerprint_sink[f"{fingerprint_prefix}pairs"] = index_fingerprint(
            pairs.query,
            pairs.key,
            pairs.target,
        )
    prediction = score_coordinate_pairs(model, coordinates[pairs.query], coordinates[pairs.key], device, batch_size)
    metrics = {f"pair_{key}": value for key, value in pair_metrics(pairs.target, prediction).items()}
    if task == "digit_greater":
        metrics.update({f"pair_{key}": value for key, value in digit_relation_metrics(split.labels, pairs, prediction).items()})
    if task == "same_class":
        retrieval_modes = (("random", False), ("hard", True)) if include_hard else (("random", False),)
        for name, hard in retrieval_modes:
            retrieval = build_retrieval_set(
                split,
                queries=retrieval_queries,
                candidates=retrieval_candidates,
                positives=retrieval_positives,
                seed=seed + (101 if hard else 0),
                hard=hard,
                hard_reservoir=hard_reservoir,
            )
            if fingerprint_sink is not None:
                fingerprint_sink[f"{fingerprint_prefix}{name}_retrieval"] = index_fingerprint(
                    retrieval.query,
                    retrieval.candidates,
                    retrieval.relevant,
                )
            query = coordinates[retrieval.query][:, None, :].expand(-1, retrieval.candidates.shape[1], -1).reshape(-1, coordinates.shape[1])
            key = coordinates[retrieval.candidates.reshape(-1)]
            scores = score_coordinate_pairs(model, query, key, device, batch_size).view(retrieval.candidates.shape)
            metrics.update({f"{name}_{key}": value for key, value in retrieval_metrics(retrieval.relevant, scores).items()})
    return metrics


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def relation_execution_metadata(
    model: PairRelationModel,
    device: torch.device,
    *,
    pairs: int = 8192,
    warmups: int = 3,
    iterations: int = 10,
) -> dict[str, float | int | str]:
    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(model.config.seed + 8093)
    query = torch.randn(pairs, model.config.relation_dim, generator=generator).to(device)
    key = torch.randn(pairs, model.config.relation_dim, generator=generator).to(device)

    def time_call(function) -> float:
        for _ in range(warmups):
            function()
        _sync(device)
        started = time.perf_counter()
        for _ in range(iterations):
            function()
        _sync(device)
        return (time.perf_counter() - started) / (iterations * pairs)

    seconds_per_pair = time_call(lambda: model.score(query, key))
    result: dict[str, float | int | str] = {
        "torch_direct_pairs_per_second": 1.0 / max(seconds_per_pair, 1e-30),
        "benchmark_pairs": pairs,
        "benchmark_warmups": warmups,
        "benchmark_iterations": iterations,
    }
    kind, value = parse_decoder_name(model.config.decoder)
    if kind in {"kendall", "mallows", "same_table_full"}:
        result.update({"execution_class": "same-chart relation LUT", "active_pair_reads": model.config.relation_tables})
    elif kind == "global_free":
        assert value is not None
        result.update(
            {
                "execution_class": "separable free chamber tower",
                "object_factor_reads": model.config.relation_tables * value,
                "active_pair_products": value,
            }
        )
    elif kind == "global_coxeter":
        assert value is not None
        result.update(
            {
                "execution_class": "separable shared Coxeter tower",
                "object_factor_reads": model.config.relation_tables * 12 * value,
                "active_pair_products": value,
            }
        )
    elif kind == "jointpair_t":
        assert value is not None
        result.update({"execution_class": "nonseparable pair LUT", "active_pair_reads": value})
    elif kind == "dense_qk":
        assert value is not None
        result.update({"execution_class": "dense QK diagnostic", "dense_projection_products_per_object": 2 * model.config.relation_dim * value})
    elif kind == "concat_mlp":
        result.update({"execution_class": "dense concat-MLP diagnostic", "dense_pair_products": 2 * model.config.relation_dim * 128 + 128})

    if isinstance(model.relation, CoxeterPairScorer) and isinstance(model.relation.kernel, RootIncidenceKernel):
        router = model.relation.router
        kernel = model.relation.kernel
        query_features = router.route(query)
        key_features = router.route(key)
        direct_seconds = time_call(lambda: model.relation._score_features(query_features, key_features))
        cached_seconds = time_call(
            lambda: kernel.cached_score(query_features.roots, key_features.roots, symmetry=model.relation.symmetry)
        )
        transform_seconds = time_call(lambda: kernel.transform_roots(key_features.roots))
        torch.testing.assert_close(
            model.relation._score_features(query_features, key_features),
            kernel.cached_score(query_features.roots, key_features.roots, symmetry=model.relation.symmetry),
            rtol=1e-5,
            atol=1e-5,
        )
        result.update(
            {
                "execution_class": "comparison-root sparse operator",
                "direct_sparse_relation_reads_per_pair": int(kernel.weight.numel()),
                "cache_relation_reads_per_object": int(kernel.weight.numel()),
                "cached_pair_reads": router.roots,
                "cached_pair_add_sub": router.roots,
                "torch_precomputed_root_direct_pairs_per_second": 1.0 / max(direct_seconds, 1e-30),
                "torch_cached_pairs_per_second": 1.0 / max(cached_seconds, 1e-30),
                "torch_cache_transforms_per_second": 1.0 / max(transform_seconds, 1e-30),
            }
        )
    return result


def validate_completed_config(existing: dict[str, object], config: PairExperimentConfig, path: Path) -> None:
    if existing.get("config") != asdict(config):
        raise ValueError(f"completed result config mismatch at {path}")


def _atomic_torch_save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(args: argparse.Namespace) -> dict[str, object]:
    config = PairExperimentConfig(
        task=args.task,
        split_mode=args.split_mode,
        decoder=args.decoder,
        payload_mode=args.payload_mode,
        objective=args.objective,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        encoder_lr=args.encoder_lr,
        relation_lr=args.relation_lr,
        auxiliary_weight=args.auxiliary_weight,
        relation_dim=args.relation_dim,
        relation_tables=args.relation_tables,
        relation_coverage=args.relation_coverage,
        encoder_depth=args.encoder_depth,
        encoder_tables=args.encoder_tables,
        encoder_comparisons=args.encoder_comparisons,
        train_pairs_per_epoch=args.train_pairs_per_epoch,
        eval_pairs=args.eval_pairs,
        retrieval_queries=args.retrieval_queries,
        retrieval_candidates=args.retrieval_candidates,
        retrieval_positives=args.retrieval_positives,
        hard_reservoir=args.hard_reservoir,
        data_split_seed=args.data_split_seed,
        max_train_examples=args.max_train_examples,
        max_eval_examples=args.max_eval_examples,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.out_dir / "result.json"
    if result_path.exists():
        existing = json.loads(result_path.read_text())
        if existing.get("complete"):
            validate_completed_config(existing, config, result_path)
            print(json.dumps({"status": "skipped_complete", "result": str(result_path)}), flush=True)
            return existing

    seed_everything(config.seed)
    splits = load_task_splits(args.data_root, config.task, config.split_mode, config.data_split_seed)
    if args.max_train_examples > 0:
        splits = TaskSplits(
            TensorSplit(splits.train.images[: args.max_train_examples], splits.train.labels[: args.max_train_examples]),
            splits.validation,
            splits.test,
            splits.auxiliary_classes,
        )
    if args.max_eval_examples > 0:
        splits = TaskSplits(
            splits.train,
            TensorSplit(splits.validation.images[: args.max_eval_examples], splits.validation.labels[: args.max_eval_examples]),
            TensorSplit(splits.test.images[: args.max_eval_examples], splits.test.labels[: args.max_eval_examples]),
            splits.auxiliary_classes,
        )
    device = torch.device(args.device)
    model = PairRelationModel(config, len(splits.auxiliary_classes)).to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": config.encoder_lr},
            {"params": model.relation.parameters(), "lr": config.relation_lr},
        ],
        weight_decay=0.0,
    )
    train_aux_labels = remap_auxiliary_labels(splits.train.labels, splits.auxiliary_classes)
    pair_count = config.train_pairs_per_epoch or 2 * splits.train.labels.numel()
    history: list[dict[str, object]] = []
    best_metric = float("-inf")
    best_epoch = 0
    checkpoint_path = args.out_dir / "best.pt"
    for incomplete in (args.out_dir / "history.csv", checkpoint_path):
        if incomplete.exists():
            incomplete.rename(incomplete.with_name(f"{incomplete.stem}.incomplete-{int(time.time())}{incomplete.suffix}"))
    started = time.perf_counter()
    training_seconds = 0.0
    optimizer_steps = 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        if config.train_pairs_per_epoch:
            pairs = sample_pair_indices(splits.train.labels, config.task, pair_count, config.seed + 1009 * epoch)
        else:
            pairs = sample_training_pair_indices(splits.train.labels, config.task, config.seed + 1009 * epoch)
        loss_sum = 0.0
        relation_loss_sum = 0.0
        auxiliary_loss_sum = 0.0
        seen = 0
        training_started = time.perf_counter()
        for start in range(0, pairs.target.numel(), config.batch_size):
            stop = min(start + config.batch_size, pairs.target.numel())
            query, key, query_aux, key_aux, _ = encode_unique_pair_batch(model, splits.train.images, pairs, start, stop, device)
            target = pairs.target[start:stop].to(device)
            prediction = model.score(query, key)
            relation_loss = F.binary_cross_entropy_with_logits(prediction, target)
            auxiliary_loss = torch.zeros((), device=device)
            if query_aux is not None and key_aux is not None:
                query_label = train_aux_labels[pairs.query[start:stop]].to(device)
                key_label = train_aux_labels[pairs.key[start:stop]].to(device)
                auxiliary_loss = 0.5 * (F.cross_entropy(query_aux, query_label) + F.cross_entropy(key_aux, key_label))
            loss = relation_loss + config.auxiliary_weight * auxiliary_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer_steps += 1
            batch = stop - start
            loss_sum += float(loss.item()) * batch
            relation_loss_sum += float(relation_loss.item()) * batch
            auxiliary_loss_sum += float(auxiliary_loss.item()) * batch
            seen += batch
        _sync(device)
        training_seconds += time.perf_counter() - training_started
        validation = evaluate_model(
            model,
            splits.validation,
            config.task,
            device,
            seed=config.seed + 3001,
            batch_size=args.eval_batch_size,
            eval_pairs=args.eval_pairs,
            retrieval_queries=args.retrieval_queries,
            retrieval_candidates=args.retrieval_candidates,
            retrieval_positives=args.retrieval_positives,
            hard_reservoir=args.hard_reservoir,
            include_hard=False,
        )
        selection = validation["random_recall_at_16"] if config.task == "same_class" else validation["pair_macro_roc_auc"]
        row: dict[str, object] = {
            "epoch": epoch,
            "train_loss": loss_sum / seen,
            "train_relation_loss": relation_loss_sum / seen,
            "train_auxiliary_loss": auxiliary_loss_sum / seen,
            **{f"validation_{key}": value for key, value in validation.items()},
        }
        history.append(row)
        _write_csv(args.out_dir / "history.csv", history)
        if math.isfinite(float(selection)) and float(selection) > best_metric:
            best_metric = float(selection)
            best_epoch = epoch
            _atomic_torch_save(
                {
                    "config": asdict(config),
                    "epoch": epoch,
                    "selection_metric": best_metric,
                    "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                },
                checkpoint_path,
            )
        print(json.dumps(row, sort_keys=True), flush=True)
    if best_epoch == 0:
        raise RuntimeError("no finite validation checkpoint was produced")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    evaluation_set_fingerprints: dict[str, str] = {}
    validation = evaluate_model(
        model,
        splits.validation,
        config.task,
        device,
        seed=config.seed + 3001,
        batch_size=args.eval_batch_size,
        eval_pairs=args.eval_pairs,
        retrieval_queries=args.retrieval_queries,
        retrieval_candidates=args.retrieval_candidates,
        retrieval_positives=args.retrieval_positives,
        hard_reservoir=args.hard_reservoir,
        fingerprint_sink=evaluation_set_fingerprints,
        fingerprint_prefix="validation_",
    )
    test = evaluate_model(
        model,
        splits.test,
        config.task,
        device,
        seed=config.seed + 4001,
        batch_size=args.eval_batch_size,
        eval_pairs=args.eval_pairs,
        retrieval_queries=args.retrieval_queries,
        retrieval_candidates=args.retrieval_candidates,
        retrieval_positives=args.retrieval_positives,
        hard_reservoir=args.hard_reservoir,
        fingerprint_sink=evaluation_set_fingerprints,
        fingerprint_prefix="test_",
    )
    router_metadata = {}
    if model.router is not None:
        router_metadata = {
            "root_edges": model.router.roots,
            "root_incidence_entries": model.router.incidence_entries,
            "chart_anchors": model.router.anchors.cpu().tolist(),
        }
    result: dict[str, object] = {
        "complete": True,
        "config": asdict(config),
        "best_epoch": best_epoch,
        "best_validation_selection_metric": best_metric,
        "encoder_parameters": model.encoder_parameters,
        "relation_parameters": model.relation_parameters,
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "binary_payload_valid": binary_payload_is_valid(model),
        "train_examples": splits.train.labels.numel(),
        "validation_examples": splits.validation.labels.numel(),
        "test_examples": splits.test.labels.numel(),
        "split_fingerprints": {
            "train": split_fingerprint(splits.train),
            "validation": split_fingerprint(splits.validation),
            "test": split_fingerprint(splits.test),
        },
        "class_train": list(CLASS_TRAIN) if config.split_mode == "class" else None,
        "class_validation": list(CLASS_VALID) if config.split_mode == "class" else None,
        "class_test": list(CLASS_TEST) if config.split_mode == "class" else None,
        "validation": validation,
        "test": test,
        "router": router_metadata,
        "evaluation_protocol": {
            "training_pairs": "one positive and one negative query role per object per epoch"
            if not config.train_pairs_per_epoch and config.task == "same_class"
            else "uniform distinct-class sampling",
            "pair_seed_validation": config.seed + 3001,
            "pair_seed_test": config.seed + 4001,
            "hard_seed_offset": 101,
            "eval_pairs": config.eval_pairs,
            "retrieval_queries": config.retrieval_queries,
            "retrieval_candidates": config.retrieval_candidates,
            "retrieval_positives": config.retrieval_positives,
            "hard_reservoir": config.hard_reservoir,
            "set_fingerprints": evaluation_set_fingerprints,
        },
        "optimizer_steps": optimizer_steps,
        "training_seconds": training_seconds,
        "train_pairs_per_second": config.epochs * pair_count / max(training_seconds, 1e-30),
        "relation_execution": relation_execution_metadata(model, device),
        "elapsed_seconds": time.perf_counter() - started,
    }
    temporary = result_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, result_path)
    print(json.dumps({"status": "complete", "result": str(result_path), "best_epoch": best_epoch}), flush=True)
    return result


def mean_sem(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0


def paired_metric_gate(
    indexed: dict[tuple[str, str, str, str, str], dict[int, dict[str, object]]],
    left: tuple[str, str, str, str, str],
    right: tuple[str, str, str, str, str],
    metric: str,
) -> dict[str, object]:
    if left not in indexed or right not in indexed:
        return {"complete": False, "passed": False, "reason": "missing group"}
    seeds = sorted(set(indexed[left]) & set(indexed[right]))
    if seeds != [0, 1, 2]:
        return {"complete": False, "passed": False, "reason": f"expected seeds 0,1,2; found {seeds}"}
    deltas = [float(indexed[left][seed][metric]) - float(indexed[right][seed][metric]) for seed in seeds]
    mean, sem = mean_sem(deltas)
    threshold = max(0.02, 2.0 * sem)
    return {
        "complete": True,
        "passed": mean > threshold and all(delta > 0.0 for delta in deltas),
        "left": left[-1],
        "right": right[-1],
        "metric": metric,
        "paired_deltas": deltas,
        "mean_delta": mean,
        "sem": sem,
        "threshold": threshold,
        "all_seed_same_sign": all(delta > 0.0 for delta in deltas),
    }


def improvement_retention(
    indexed: dict[tuple[str, str, str, str, str], dict[int, dict[str, object]]],
    numerator: tuple[str, str, str, str, str],
    denominator: tuple[str, str, str, str, str],
    metric: str,
    random_value: float,
    threshold: float,
) -> dict[str, object]:
    if numerator not in indexed or denominator not in indexed:
        return {"complete": False, "passed": False, "reason": "missing group"}
    seeds = sorted(set(indexed[numerator]) & set(indexed[denominator]))
    if seeds != [0, 1, 2]:
        return {"complete": False, "passed": False, "reason": f"expected seeds 0,1,2; found {seeds}"}
    numerator_mean = statistics.mean(float(indexed[numerator][seed][metric]) for seed in seeds)
    denominator_mean = statistics.mean(float(indexed[denominator][seed][metric]) for seed in seeds)
    retention = (numerator_mean - random_value) / max(denominator_mean - random_value, 1e-12)
    return {
        "complete": True,
        "passed": retention >= threshold,
        "retention": retention,
        "threshold": threshold,
        "numerator_mean": numerator_mean,
        "denominator_mean": denominator_mean,
        "random_value": random_value,
    }


def absolute_group_gate(
    indexed: dict[tuple[str, str, str, str, str], dict[int, dict[str, object]]],
    key: tuple[str, str, str, str, str],
    metric: str,
    threshold: float,
    direction: str = "above",
) -> dict[str, object]:
    if key not in indexed or sorted(indexed[key]) != [0, 1, 2]:
        return {"complete": False, "passed": False, "reason": "missing three-seed group"}
    values = [float(indexed[key][seed][metric]) for seed in (0, 1, 2)]
    mean, sem = mean_sem(values)
    passed = mean >= threshold if direction == "above" else mean <= threshold
    return {"complete": True, "passed": passed, "mean": mean, "sem": sem, "threshold": threshold, "direction": direction}


def architecture_svg() -> str:
    boxes = (
        (20, 65, 140, 54, "EMNIST image", "784 pixels"),
        (205, 65, 180, 54, "Shared unary PC-LUT", "L4, T64/C6"),
        (430, 65, 130, 54, "Coordinates", "h in R^32"),
        (605, 20, 170, 54, "16 balanced K4 charts", "p_t in S4"),
        (605, 110, 170, 54, "Global roots", "c(h), E about 96"),
        (820, 20, 190, 54, "Global chamber kernel", "free / Coxeter shared"),
        (820, 110, 190, 54, "Root-incidence kernel", "per-object cache: M c(h)"),
        (1055, 65, 120, 54, "Pair logit", "per-pair root +/- reads"),
    )
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="190" viewBox="0 0 1200 190">',
        '<defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">'
        '<path d="M0,0 L8,4 L0,8 z" fill="#334155"/></marker></defs>',
        '<rect width="1200" height="190" fill="#ffffff"/>',
    ]
    for x, y, width, height, title, subtitle in boxes:
        parts.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="8" fill="#eef2ff" stroke="#4f46e5"/>')
        parts.append(
            f'<text x="{x + width / 2}" y="{y + 22}" text-anchor="middle" font-family="sans-serif" '
            f'font-size="13" font-weight="bold">{title}</text>'
        )
        parts.append(f'<text x="{x + width / 2}" y="{y + 41}" text-anchor="middle" font-family="sans-serif" font-size="11">{subtitle}</text>')
    lines = (
        (160, 92, 205, 92),
        (385, 92, 430, 92),
        (560, 92, 605, 47),
        (560, 92, 605, 137),
        (775, 47, 820, 47),
        (775, 137, 820, 137),
        (1010, 47, 1055, 92),
        (1010, 137, 1055, 92),
    )
    for x1, y1, x2, y2 in lines:
        parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#334155" stroke-width="2" marker-end="url(#a)"/>')
    parts.append("</svg>")
    return "".join(parts)


def summarize_results(args: argparse.Namespace) -> dict[str, object]:
    results = [json.loads(path.read_text()) for path in sorted(args.result_dir.glob("**/result.json"))]
    results = [result for result in results if result.get("complete")]
    if not results:
        raise RuntimeError(f"no complete results under {args.result_dir}")
    rows: list[dict[str, object]] = []
    for result in results:
        config = result["config"]
        rows.append(
            {
                **config,
                "relation_parameters": result["relation_parameters"],
                "best_epoch": result["best_epoch"],
                "train_pairs_per_second": result.get("train_pairs_per_second", math.nan),
                "relation_execution_class": result.get("relation_execution", {}).get("execution_class", "unrecorded"),
                "torch_direct_pairs_per_second": result.get("relation_execution", {}).get("torch_direct_pairs_per_second", math.nan),
                **{f"test_{key}": value for key, value in result["test"].items()},
            }
        )
    fields = sorted({key for row in rows for key in row})
    with (args.result_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    groups: dict[tuple[str, str, str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["task"]), str(row["split_mode"]), str(row["payload_mode"]), str(row["objective"]), str(row["decoder"]))
        groups.setdefault(key, []).append(row)
    aggregate: list[dict[str, object]] = []
    for key, group in sorted(groups.items()):
        record: dict[str, object] = dict(zip(("task", "split_mode", "payload_mode", "objective", "decoder"), key))
        for metric in (
            "test_pair_roc_auc",
            "test_pair_macro_roc_auc",
            "test_pair_adjacent_accuracy",
            "test_pair_pr_auc",
            "test_pair_accuracy",
            "test_random_recall_at_16",
            "test_random_hit_at_1",
            "train_pairs_per_second",
            "torch_direct_pairs_per_second",
        ):
            values = [float(row[metric]) for row in group if metric in row]
            if values:
                record[f"{metric}_mean"], record[f"{metric}_sem"] = mean_sem(values)
        aggregate.append(record)
    indexed: dict[tuple[str, str, str, str, str], dict[int, dict[str, object]]] = {}
    for result in results:
        config = result["config"]
        key = (
            str(config["task"]),
            str(config["split_mode"]),
            str(config["payload_mode"]),
            str(config["objective"]),
            str(config["decoder"]),
        )
        indexed.setdefault(key, {})[int(config["seed"])] = result["test"]
    class_root = ("same_class", "class", "float", "relation_only", "root_incidence")
    class_same = ("same_class", "class", "float", "relation_only", "same_table_full")
    class_joint = ("same_class", "class", "float", "relation_only", "jointpair_t16")
    class_coxeter = ("same_class", "class", "float", "relation_only", "global_coxeter_r12")
    class_free = ("same_class", "class", "float", "relation_only", "global_free_r12")
    object_root = ("same_class", "object", "float", "relation_only", "root_incidence")
    class_binary_root = ("same_class", "class", "binary01", "relation_only", "root_incidence")
    class_aux_root = ("same_class", "class", "float", "relation_aux", "root_incidence")
    digit_root = ("digit_greater", "object", "float", "relation_only", "root_incidence")
    digit_dense = ("digit_greater", "object", "float", "relation_only", "dense_qk_r16")
    digit_same = ("digit_greater", "object", "float", "relation_only", "same_table_full")
    gates = {
        "root_vs_same_table": paired_metric_gate(indexed, class_root, class_same, "random_recall_at_16"),
        "root_vs_budget_matched_jointpair": paired_metric_gate(indexed, class_root, class_joint, "random_recall_at_16"),
        "coxeter_sharing_vs_free_rank12": paired_metric_gate(indexed, class_coxeter, class_free, "random_recall_at_16"),
        "root_class_transfer": improvement_retention(indexed, class_root, object_root, "random_recall_at_16", 16.0 / 512.0, 0.70),
        "root_binary_retention": improvement_retention(indexed, class_binary_root, class_root, "random_recall_at_16", 16.0 / 512.0, 0.90),
        "root_relation_only_vs_aux": improvement_retention(indexed, class_root, class_aux_root, "random_recall_at_16", 16.0 / 512.0, 0.80),
        "digit_root_auc": absolute_group_gate(indexed, digit_root, "macro_roc_auc", 0.85),
        "digit_root_vs_same_table": paired_metric_gate(indexed, digit_root, digit_same, "macro_roc_auc"),
    }
    digit_dense_gap = {"complete": False, "passed": False, "reason": "missing groups"}
    if digit_root in indexed and digit_dense in indexed and sorted(indexed[digit_root]) == sorted(indexed[digit_dense]) == [0, 1, 2]:
        root_mean = statistics.mean(float(indexed[digit_root][seed]["macro_roc_auc"]) for seed in (0, 1, 2))
        dense_mean = statistics.mean(float(indexed[digit_dense][seed]["macro_roc_auc"]) for seed in (0, 1, 2))
        digit_dense_gap = {
            "complete": True,
            "passed": root_mean >= dense_mean - 0.02,
            "root_mean": root_mean,
            "dense_mean": dense_mean,
            "allowed_gap": 0.02,
        }
    gates["digit_root_within_dense"] = digit_dense_gap
    complete_gates = all(bool(gate.get("complete")) for gate in gates.values())
    native_kernel_pass = bool(gates["root_vs_same_table"].get("passed")) and bool(
        gates["root_vs_budget_matched_jointpair"].get("passed")
    )
    coxeter_sharing_pass = bool(gates["coxeter_sharing_vs_free_rank12"].get("passed"))
    digit_pass = all(
        bool(gates[name].get("passed"))
        for name in ("digit_root_auc", "digit_root_vs_same_table", "digit_root_within_dense")
    )
    semantic_pass = complete_gates and (native_kernel_pass or coxeter_sharing_pass or digit_pass)
    decision = {
        "complete_runs": len(results),
        "groups": len(aggregate),
        "all_preregistered_gates_complete": complete_gates,
        "native_root_kernel_passed": native_kernel_pass,
        "coxeter_sharing_passed": coxeter_sharing_pass,
        "digit_anisotropic_kernel_passed": digit_pass,
        "semantic_gate_passed": semantic_pass,
        "next_stage": "relation_quantization" if semantic_pass else "stop",
        "gates": gates,
    }
    (args.result_dir / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    figure = args.out_report.parent / "figures" / "emnist_pair_global_coxeter_kernel.svg"
    figure.parent.mkdir(parents=True, exist_ok=True)
    figure.write_text(architecture_svg())
    lines = [
        "# Global Coxeter and Root-Incidence Kernels on Real EMNIST Pairs",
        "",
        "## Architecture",
        "",
        f"![architecture]({figure.relative_to(args.out_report.parent)})",
        "",
        "## Aggregate results",
        "",
        "| Task | Split | Payload | Objective | Decoder | ROC-AUC | Recall@16 |",
        "|---|---|---|---|---|---:|---:|",
    ]
    for row in aggregate:
        auc = row.get("test_pair_roc_auc_mean", math.nan)
        recall = row.get("test_random_recall_at_16_mean", math.nan)
        lines.append(
            f"| {row['task']} | {row['split_mode']} | {row['payload_mode']} | {row['objective']} | {row['decoder']} | "
            f"{float(auc):.4f} | {float(recall):.4f} |"
        )
    lines += [
        "",
        "## Preregistered decision",
        "",
        f"- Native root kernel gate: `{'PASS' if native_kernel_pass else 'FAIL'}`.",
        f"- Coxeter sharing gate: `{'PASS' if coxeter_sharing_pass else 'FAIL'}`.",
        f"- Digit anisotropic gate: `{'PASS' if digit_pass else 'FAIL'}`.",
        f"- Next stage: `{decision['next_stage']}`.",
        "",
        "## Interpretation boundary",
        "",
        "This report is generated from complete result JSON files without rewriting raw logs. Dense QK and concat MLP are diagnostics; "
        "their presence does not make the comparison-native kernels GEMM-free. Root-incidence execution is a Torch reference until a "
        "separate matched systems gate is passed.",
        "",
    ]
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text("\n".join(lines))
    print(json.dumps(decision), flush=True)
    return decision


def select_learning_rates(args: argparse.Namespace) -> dict[str, object]:
    results = [json.loads(path.read_text()) for path in sorted(args.result_dir.glob("**/result.json"))]
    results = [result for result in results if result.get("complete")]
    selected: dict[str, object] = {}
    env_lines: list[str] = []
    for payload in ("float", "binary01"):
        candidates: dict[tuple[float, float], list[float]] = {}
        decoder_sets: dict[tuple[float, float], set[str]] = {}
        for result in results:
            config = result["config"]
            if config["payload_mode"] != payload:
                continue
            key = (float(config["encoder_lr"]), float(config["relation_lr"]))
            candidates.setdefault(key, []).append(float(result["best_validation_selection_metric"]))
            decoder_sets.setdefault(key, set()).add(str(config["decoder"]))
        complete = [key for key in candidates if {"root_incidence", "dense_qk_r16"} <= decoder_sets[key]]
        if not complete:
            raise RuntimeError(f"no complete two-decoder learning-rate candidates for {payload}")
        choice = min(complete, key=lambda key: (-statistics.mean(candidates[key]), key[0], key[1]))
        prefix = "FLOAT" if payload == "float" else "BINARY"
        selected[payload] = {
            "encoder_lr": choice[0],
            "relation_lr": choice[1],
            "mean_validation_recall_at_16": statistics.mean(candidates[choice]),
        }
        env_lines.extend((f"{prefix}_ENCODER_LR={choice[0]}", f"{prefix}_RELATION_LR={choice[1]}"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selected, indent=2) + "\n")
    args.output.with_suffix(".env").write_text("\n".join(env_lines) + "\n")
    print(json.dumps(selected), flush=True)
    return selected


def check_frontier_gate(args: argparse.Namespace) -> dict[str, object]:
    results = [json.loads(path.read_text()) for path in sorted(args.result_dir.glob("**/result.json"))]
    metrics: dict[str, list[tuple[float, float]]] = {"dense_qk_r16": [], "concat_mlp": []}
    for result in results:
        config = result.get("config", {})
        decoder = config.get("decoder")
        if (
            result.get("complete")
            and decoder in metrics
            and config.get("task") == "same_class"
            and config.get("split_mode") == "object"
            and config.get("payload_mode") == "float"
            and config.get("objective") == "relation_only"
        ):
            metrics[str(decoder)].append(
                (float(result["validation"]["pair_roc_auc"]), float(result["validation"]["random_recall_at_16"]))
            )
    decoder_rows: dict[str, object] = {}
    passed = False
    for decoder, values in metrics.items():
        if len(values) != 3:
            raise RuntimeError(f"frontier gate requires three complete seeds for {decoder}, found {len(values)}")
        mean_auc = statistics.mean(value[0] for value in values)
        mean_recall = statistics.mean(value[1] for value in values)
        decoder_passed = mean_auc >= 0.90 and mean_recall >= 0.50
        decoder_rows[decoder] = {"mean_roc_auc": mean_auc, "mean_recall_at_16": mean_recall, "passed": decoder_passed}
        passed = passed or decoder_passed
    decision = {"passed": passed, "criterion": "at least one dense diagnostic has AUC >= 0.90 and Recall@16 >= 0.50", "decoders": decoder_rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, indent=2) + "\n")
    print(json.dumps(decision), flush=True)
    return decision


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--task", choices=("same_class", "digit_greater"), required=True)
    run.add_argument("--split-mode", choices=("object", "class"), required=True)
    run.add_argument("--decoder", required=True)
    run.add_argument("--payload-mode", choices=("float", "binary01"), required=True)
    run.add_argument("--objective", choices=("relation_only", "relation_aux"), required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--epochs", type=int, default=20)
    run.add_argument("--batch-size", type=int, default=512)
    run.add_argument("--eval-batch-size", type=int, default=4096)
    run.add_argument("--encoder-lr", type=float, required=True)
    run.add_argument("--relation-lr", type=float, required=True)
    run.add_argument("--auxiliary-weight", type=float, default=0.25)
    run.add_argument("--relation-dim", type=int, default=32)
    run.add_argument("--relation-tables", type=int, default=16)
    run.add_argument("--relation-coverage", type=int, default=2)
    run.add_argument("--encoder-depth", type=int, default=4)
    run.add_argument("--encoder-tables", type=int, default=64)
    run.add_argument("--encoder-comparisons", type=int, default=6)
    run.add_argument("--train-pairs-per-epoch", type=int, default=0)
    run.add_argument("--eval-pairs", type=int, default=8192)
    run.add_argument("--retrieval-queries", type=int, default=256)
    run.add_argument("--retrieval-candidates", type=int, default=512)
    run.add_argument("--retrieval-positives", type=int, default=16)
    run.add_argument("--hard-reservoir", type=int, default=4096)
    run.add_argument("--data-split-seed", type=int, default=1729)
    run.add_argument("--max-train-examples", type=int, default=0)
    run.add_argument("--max-eval-examples", type=int, default=0)
    run.add_argument("--data-root", type=Path, default=Path("data"))
    run.add_argument("--device", default="cuda")
    run.add_argument("--out-dir", type=Path, required=True)
    summarize = commands.add_parser("summarize")
    summarize.add_argument("--result-dir", type=Path, required=True)
    summarize.add_argument("--out-report", type=Path, required=True)
    select_lr = commands.add_parser("select-lr")
    select_lr.add_argument("--result-dir", type=Path, required=True)
    select_lr.add_argument("--output", type=Path, required=True)
    gate = commands.add_parser("frontier-gate")
    gate.add_argument("--result-dir", type=Path, required=True)
    gate.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run":
        run_experiment(args)
    elif args.command == "summarize":
        summarize_results(args)
    elif args.command == "select-lr":
        select_learning_rates(args)
    else:
        decision = check_frontier_gate(args)
        if not decision["passed"]:
            raise SystemExit(3)


if __name__ == "__main__":
    main()
