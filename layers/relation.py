from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

from .pairwise import _make_pairwise_anchors
from .surrogate import ste_heaviside

RelationMode = Literal["constrained_gram", "free", "additive"]
RelationInit = Literal["random", "zeros"]
RelationQuantization = Literal["float", "ternary", "binary"]

__all__ = [
    "ComparisonRelationLUT",
    "ComparisonRelationSpec",
    "ComparisonRoute",
    "RelationInit",
    "RelationMode",
    "RelationQuantization",
]


@dataclass(frozen=True)
class ComparisonRelationSpec:
    input_dim: int
    num_banks: int
    num_codes: int
    relation_rank: int
    relation_mode: RelationMode
    relation_init: RelationInit
    quantization: RelationQuantization
    anchor_policy: str
    seed: int
    surrogate: str

    def __post_init__(self) -> None:
        if self.input_dim < 2:
            raise ValueError("input_dim must be at least 2")
        if self.num_banks < 1:
            raise ValueError("num_banks must be positive")
        if self.num_codes < 2 or self.num_codes & (self.num_codes - 1):
            raise ValueError("num_codes must be a power of two greater than one")
        if self.relation_rank < 1:
            raise ValueError("relation_rank must be positive")
        if self.relation_mode not in {"constrained_gram", "free", "additive"}:
            raise ValueError(f"unsupported relation_mode {self.relation_mode!r}")
        if self.relation_init not in {"random", "zeros"}:
            raise ValueError(f"unsupported relation_init {self.relation_init!r}")
        if self.quantization not in {"float", "ternary", "binary"}:
            raise ValueError(f"unsupported quantization {self.quantization!r}")

    @property
    def comparisons(self) -> int:
        return self.num_codes.bit_length() - 1


@dataclass(frozen=True)
class ComparisonRoute:
    indices: Tensor
    margins: Tensor

    def detach(self) -> "ComparisonRoute":
        return ComparisonRoute(self.indices.detach(), self.margins.detach())


class _ComparisonRouter(nn.Module):
    """Fixed anchor graph with calibratable, optionally trainable thresholds."""

    def __init__(self, spec: ComparisonRelationSpec, *, seed_offset: int, train_thresholds: bool) -> None:
        super().__init__()
        anchors = _make_pairwise_anchors(
            spec.input_dim,
            spec.num_banks,
            spec.comparisons,
            policy=spec.anchor_policy,
            seed=spec.seed + seed_offset,
        )
        self.register_buffer("anchors", anchors)
        self.register_buffer("powers", 2 ** torch.arange(spec.comparisons, dtype=torch.long))
        self.thresholds = nn.Parameter(
            torch.zeros(spec.num_banks, spec.comparisons),
            requires_grad=train_thresholds,
        )

    def forward(self, x: Tensor, threshold_offset: Tensor | None = None) -> ComparisonRoute:
        if x.shape[-1] <= int(self.anchors.max().item()):
            raise ValueError(f"input width {x.shape[-1]} does not cover all anchors")
        anchor_a = self.anchors[..., 0].reshape(-1)
        anchor_b = self.anchors[..., 1].reshape(-1)
        route_shape = (*x.shape[:-1], *self.anchors.shape[:2])
        margins = x[..., anchor_a].reshape(route_shape) - x[..., anchor_b].reshape(route_shape)
        margins = margins - self.thresholds.to(device=x.device, dtype=x.dtype)
        if threshold_offset is not None:
            margins = margins - threshold_offset.to(device=x.device, dtype=x.dtype)
        indices = ((margins > 0).to(torch.long) * self.powers).sum(dim=-1)
        return ComparisonRoute(indices, margins)

    @torch.no_grad()
    def calibrate(self, samples: Tensor) -> None:
        if samples.ndim != 2:
            raise ValueError(f"calibration samples must be [samples, input_dim], got {tuple(samples.shape)}")
        anchor_a = self.anchors[..., 0].reshape(-1)
        anchor_b = self.anchors[..., 1].reshape(-1)
        margins = samples[:, anchor_a] - samples[:, anchor_b]
        margins = margins.reshape(samples.shape[0], *self.anchors.shape[:2])
        self.thresholds.copy_(margins.median(dim=0).values.to(self.thresholds))

    def set_threshold_training(self, enabled: bool) -> None:
        self.thresholds.requires_grad_(enabled)


class ComparisonRelationLUT(nn.Module):
    """Direct relation over independently comparison-routed query and key codes."""

    def __init__(
        self,
        input_dim: int,
        *,
        num_banks: int = 16,
        num_codes: int = 32,
        relation_rank: int = 16,
        relation_mode: RelationMode = "free",
        relation_init: RelationInit = "random",
        quantization: RelationQuantization = "float",
        train_thresholds: bool = False,
        anchor_policy: str = "expander",
        seed: int = 0,
        surrogate: str = "fast_sigmoid_odd",
    ) -> None:
        super().__init__()
        self.spec = ComparisonRelationSpec(
            int(input_dim),
            int(num_banks),
            int(num_codes),
            int(relation_rank),
            relation_mode,
            relation_init,
            quantization,
            anchor_policy,
            int(seed),
            surrogate,
        )
        self.query_router = _ComparisonRouter(self.spec, seed_offset=0, train_thresholds=train_thresholds)
        self.key_router = _ComparisonRouter(self.spec, seed_offset=104729, train_thresholds=train_thresholds)
        generator = torch.Generator(device="cpu").manual_seed(self.spec.seed + 17)
        if self.spec.relation_mode == "constrained_gram":
            scale = 1.0 / math.sqrt(self.spec.relation_rank)
            query = torch.randn(
                self.spec.num_banks,
                self.spec.num_codes,
                self.spec.relation_rank,
                generator=generator,
            ) * scale
            key = torch.randn(
                self.spec.num_banks,
                self.spec.num_codes,
                self.spec.relation_rank,
                generator=generator,
            ) * scale
            if self.spec.relation_init == "zeros":
                query.zero_()
                key.zero_()
            self.query_factors = nn.Parameter(query)
            self.key_factors = nn.Parameter(key)
        elif self.spec.relation_mode == "free":
            relation = torch.randn(
                self.spec.num_banks,
                self.spec.num_codes,
                self.spec.num_codes,
                generator=generator,
            ) / math.sqrt(self.spec.num_banks)
            if self.spec.relation_init == "zeros":
                relation.zero_()
            self.relation = nn.Parameter(relation)
        else:
            scale = 1.0 / math.sqrt(self.spec.num_banks)
            query = torch.randn(self.spec.num_banks, self.spec.num_codes, generator=generator) * scale
            key = torch.randn(self.spec.num_banks, self.spec.num_codes, generator=generator) * scale
            if self.spec.relation_init == "zeros":
                query.zero_()
                key.zero_()
            self.query_values = nn.Parameter(query)
            self.key_values = nn.Parameter(key)

    @property
    def input_dim(self) -> int:
        return self.spec.input_dim

    @property
    def num_banks(self) -> int:
        return self.spec.num_banks

    @property
    def num_codes(self) -> int:
        return self.spec.num_codes

    @property
    def comparisons(self) -> int:
        return self.spec.comparisons

    @property
    def relation_rank(self) -> int:
        return self.spec.relation_rank

    @property
    def quantization(self) -> RelationQuantization:
        return self.spec.quantization

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, num_banks={self.num_banks}, num_codes={self.num_codes}, "
            f"relation_rank={self.relation_rank}, relation_mode={self.spec.relation_mode!r}, "
            f"quantization={self.quantization!r}"
        )

    def calibrate_routes(self, query_samples: Tensor, key_samples: Tensor) -> None:
        self.query_router.calibrate(query_samples)
        self.key_router.calibrate(key_samples)

    def set_threshold_training(self, enabled: bool) -> None:
        self.query_router.set_threshold_training(enabled)
        self.key_router.set_threshold_training(enabled)

    def routes(
        self,
        query: Tensor,
        key: Tensor,
        *,
        query_threshold_offset: Tensor | None = None,
        key_threshold_offset: Tensor | None = None,
    ) -> tuple[ComparisonRoute, ComparisonRoute]:
        return (
            self.query_router(query, query_threshold_offset),
            self.key_router(key, key_threshold_offset),
        )

    def materialized_relation(self, *, quantized: bool = False) -> Tensor:
        if self.spec.relation_mode == "constrained_gram":
            relation = torch.matmul(self.query_factors, self.key_factors.transpose(-1, -2))
            relation = relation / math.sqrt(self.relation_rank)
        elif self.spec.relation_mode == "free":
            relation = self.relation
        else:
            relation = self.query_values.unsqueeze(-1) + self.key_values.unsqueeze(-2)
        return self._quantize(relation) if quantized else relation

    def _quantize(self, relation: Tensor) -> Tensor:
        if self.quantization == "float":
            return relation
        scale = relation.detach().abs().mean(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        if self.quantization == "ternary":
            discrete = torch.round(relation / scale).clamp(-1, 1)
        else:
            discrete = torch.where(relation >= 0, torch.ones_like(relation), -torch.ones_like(relation))
        quantized = scale * discrete
        return quantized.detach() + (relation - relation.detach())

    @torch.no_grad()
    def initialize_cross_gram(self, query_factors: Tensor, key_factors: Tensor) -> None:
        expected = (self.num_banks, self.num_codes, self.relation_rank)
        if tuple(query_factors.shape) != expected or tuple(key_factors.shape) != expected:
            raise ValueError(f"cross-Gram factors must both have shape {expected}")
        if self.spec.relation_mode == "constrained_gram":
            self.query_factors.copy_(query_factors.to(self.query_factors))
            self.key_factors.copy_(key_factors.to(self.key_factors))
        elif self.spec.relation_mode == "free":
            relation = torch.matmul(query_factors, key_factors.transpose(-1, -2))
            self.relation.copy_((relation / math.sqrt(self.relation_rank)).to(self.relation))
        else:
            relation = torch.matmul(query_factors, key_factors.transpose(-1, -2))
            relation = relation / math.sqrt(self.relation_rank)
            grand = relation.mean(dim=(-2, -1), keepdim=True)
            self.query_values.copy_((relation.mean(dim=-1) - 0.5 * grand.squeeze(-1)).to(self.query_values))
            self.key_values.copy_((relation.mean(dim=-2) - 0.5 * grand.squeeze(-2)).to(self.key_values))

    @torch.no_grad()
    def initialize_from_samples(
        self,
        query_samples: Tensor,
        key_samples: Tensor,
        query_features: Tensor,
        key_features: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if query_features.shape != (query_samples.shape[0], self.relation_rank):
            raise ValueError("query_features shape does not match samples and relation_rank")
        if key_features.shape != (key_samples.shape[0], self.relation_rank):
            raise ValueError("key_features shape does not match samples and relation_rank")
        query_codes = self.query_router(query_samples).indices
        key_codes = self.key_router(key_samples).indices
        query_factors = _cell_means(query_codes, query_features, self.num_codes)
        key_factors = _cell_means(key_codes, key_features, self.num_codes)
        bank_scale = self.num_banks ** -0.25
        query_factors.mul_(bank_scale)
        key_factors.mul_(bank_scale)
        self.initialize_cross_gram(query_factors, key_factors)
        return query_factors, key_factors

    @classmethod
    @torch.no_grad()
    def free_from_constrained(
        cls,
        source: "ComparisonRelationLUT",
        *,
        quantization: RelationQuantization = "float",
        random_init: bool = False,
        seed: int | None = None,
    ) -> "ComparisonRelationLUT":
        if source.spec.relation_mode != "constrained_gram":
            raise ValueError("source must use constrained_gram relation_mode")
        result = cls(
            source.input_dim,
            num_banks=source.num_banks,
            num_codes=source.num_codes,
            relation_rank=source.relation_rank,
            relation_mode="free",
            relation_init="zeros",
            quantization=quantization,
            train_thresholds=source.query_router.thresholds.requires_grad,
            anchor_policy=source.spec.anchor_policy,
            seed=source.spec.seed if seed is None else seed,
            surrogate=source.spec.surrogate,
        )
        result.query_router.load_state_dict(source.query_router.state_dict())
        result.key_router.load_state_dict(source.key_router.state_dict())
        relation = source.materialized_relation(quantized=False).detach()
        if random_init:
            generator = torch.Generator(device="cpu").manual_seed((source.spec.seed if seed is None else seed) + 7919)
            source_flat = relation.detach().cpu().flatten(1)
            shuffled = torch.empty_like(source_flat)
            for bank in range(source_flat.shape[0]):
                permutation = torch.randperm(source_flat.shape[1], generator=generator)
                shuffled[bank] = source_flat[bank, permutation]
            relation = shuffled.reshape_as(relation).to(relation.device)
        result.relation.copy_(relation.to(result.relation))
        return result

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        *,
        query_threshold_offset: Tensor | None = None,
        key_threshold_offset: Tensor | None = None,
        mask: Tensor | None = None,
        return_routes: bool = False,
    ) -> Tensor | tuple[Tensor, ComparisonRoute, ComparisonRoute]:
        if query.shape[-1] != self.input_dim or key.shape[-1] != self.input_dim:
            raise ValueError(f"query and key must end in input_dim={self.input_dim}")
        if query.ndim < 2 or key.ndim < 2 or query.shape[:-2] != key.shape[:-2]:
            raise ValueError("query and key must have matching prefixes and explicit item dimensions")
        query_route, key_route = self.routes(
            query,
            key,
            query_threshold_offset=query_threshold_offset,
            key_threshold_offset=key_threshold_offset,
        )
        scores = self._score_matrix(query_route, key_route)
        if mask is not None:
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
            else:
                scores = scores + mask.to(device=scores.device, dtype=scores.dtype)
        return (scores, query_route, key_route) if return_routes else scores

    def score_aligned(
        self,
        query: Tensor,
        key: Tensor,
        *,
        query_threshold_offset: Tensor | None = None,
        key_threshold_offset: Tensor | None = None,
    ) -> Tensor:
        if query.shape != key.shape or query.shape[-1] != self.input_dim:
            raise ValueError(f"aligned query and key must have identical shape [..., {self.input_dim}]")
        query_route, key_route = self.routes(
            query,
            key,
            query_threshold_offset=query_threshold_offset,
            key_threshold_offset=key_threshold_offset,
        )
        return self._score_aligned(query_route, key_route)

    def _score_matrix(self, query_route: ComparisonRoute, key_route: ComparisonRoute) -> Tensor:
        relation = self.materialized_relation(quantized=True)
        query_index = query_route.indices.unsqueeze(-2)
        key_index = key_route.indices.unsqueeze(-3)
        bank_shape = [1] * query_index.ndim
        bank_shape[-1] = self.num_banks
        bank = torch.arange(self.num_banks, device=relation.device).view(bank_shape)
        bank_scores = relation[bank, query_index, key_index]
        scores = bank_scores.sum(dim=-1)
        if self.training and _route_requires_grad(query_route, key_route, self):
            scores = scores + self._matrix_route_surrogate(relation, bank_scores, query_route, key_route)
        return scores / math.sqrt(self.num_banks)

    def _score_aligned(self, query_route: ComparisonRoute, key_route: ComparisonRoute) -> Tensor:
        relation = self.materialized_relation(quantized=True)
        bank_shape = [1] * query_route.indices.ndim
        bank_shape[-1] = self.num_banks
        bank = torch.arange(self.num_banks, device=relation.device).view(bank_shape)
        bank_scores = relation[bank, query_route.indices, key_route.indices]
        scores = bank_scores.sum(dim=-1)
        if self.training and _route_requires_grad(query_route, key_route, self):
            scores = scores + self._aligned_route_surrogate(relation, bank_scores, query_route, key_route)
        return scores / math.sqrt(self.num_banks)

    def _matrix_route_surrogate(
        self,
        relation: Tensor,
        bank_scores: Tensor,
        query_route: ComparisonRoute,
        key_route: ComparisonRoute,
    ) -> Tensor:
        powers = self.query_router.powers
        query_neighbor = query_route.indices.unsqueeze(-1) ^ powers
        key_neighbor = key_route.indices.unsqueeze(-1) ^ powers
        query_neighbor = query_neighbor.unsqueeze(-3)
        key_for_query = key_route.indices.unsqueeze(-3).unsqueeze(-1)
        query_for_key = query_route.indices.unsqueeze(-2).unsqueeze(-1)
        key_neighbor = key_neighbor.unsqueeze(-4)
        bank_shape = [1] * query_neighbor.ndim
        bank_shape[-2] = self.num_banks
        bank = torch.arange(self.num_banks, device=relation.device).view(bank_shape)
        query_value = relation[bank, query_neighbor, key_for_query]
        key_value = relation[bank, query_for_key, key_neighbor]
        current = bank_scores.unsqueeze(-1)
        query_delta = _ste_delta(query_route.margins, self.spec.surrogate).unsqueeze(-3)
        key_delta = _ste_delta(key_route.margins, self.spec.surrogate).unsqueeze(-4)
        return (query_delta * (query_value - current) + key_delta * (key_value - current)).sum(dim=(-1, -2))

    def _aligned_route_surrogate(
        self,
        relation: Tensor,
        bank_scores: Tensor,
        query_route: ComparisonRoute,
        key_route: ComparisonRoute,
    ) -> Tensor:
        powers = self.query_router.powers
        query_neighbor = query_route.indices.unsqueeze(-1) ^ powers
        key_neighbor = key_route.indices.unsqueeze(-1) ^ powers
        bank_shape = [1] * query_neighbor.ndim
        bank_shape[-2] = self.num_banks
        bank = torch.arange(self.num_banks, device=relation.device).view(bank_shape)
        query_value = relation[bank, query_neighbor, key_route.indices.unsqueeze(-1)]
        key_value = relation[bank, query_route.indices.unsqueeze(-1), key_neighbor]
        current = bank_scores.unsqueeze(-1)
        query_delta = _ste_delta(query_route.margins, self.spec.surrogate)
        key_delta = _ste_delta(key_route.margins, self.spec.surrogate)
        return (query_delta * (query_value - current) + key_delta * (key_value - current)).sum(dim=(-1, -2))


def _cell_means(codes: Tensor, features: Tensor, num_codes: int) -> Tensor:
    banks = codes.shape[-1]
    result = torch.zeros(banks, num_codes, features.shape[-1], device=features.device, dtype=features.dtype)
    global_mean = features.mean(dim=0)
    for bank in range(banks):
        count = torch.bincount(codes[:, bank], minlength=num_codes)
        result[bank].index_add_(0, codes[:, bank], features)
        occupied = count > 0
        result[bank, occupied] /= count[occupied].to(features.dtype).unsqueeze(-1)
        result[bank, ~occupied] = global_mean
    return result


def _ste_delta(margin: Tensor, surrogate: str) -> Tensor:
    return ste_heaviside(margin, surrogate) - (margin > 0).to(margin.dtype)


def _route_requires_grad(
    query_route: ComparisonRoute,
    key_route: ComparisonRoute,
    layer: ComparisonRelationLUT,
) -> bool:
    return (
        query_route.margins.requires_grad
        or key_route.margins.requires_grad
        or layer.query_router.thresholds.requires_grad
        or layer.key_router.thresholds.requires_grad
    )
