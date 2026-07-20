from __future__ import annotations

import itertools
import math
from functools import lru_cache

import torch
from torch import Tensor, nn

__all__ = [
    "GaugeAlignedS4Relation",
    "circulant_relation_edges",
    "s4_fourier_energy",
    "s4_gauge_maps",
    "s4_tables",
]

S4_ORDER = 24
_PERMUTATIONS = tuple(itertools.permutations(range(4)))


@lru_cache(maxsize=1)
def s4_tables() -> tuple[Tensor, Tensor, Tensor]:
    """Return inverse, composition, and Coxeter-length tables for S4."""
    lookup = {permutation: index for index, permutation in enumerate(_PERMUTATIONS)}
    inverse = torch.empty(S4_ORDER, dtype=torch.long)
    composition = torch.empty(S4_ORDER, S4_ORDER, dtype=torch.long)
    length = torch.empty(S4_ORDER, dtype=torch.long)
    for left_index, left in enumerate(_PERMUTATIONS):
        inverse[left_index] = lookup[tuple(left.index(position) for position in range(4))]
        length[left_index] = sum(
            left[first] > left[second]
            for first in range(4)
            for second in range(first + 1, 4)
        )
        for right_index, right in enumerate(_PERMUTATIONS):
            composition[left_index, right_index] = lookup[
                tuple(left[right[position]] for position in range(4))
            ]
    return inverse, composition, length


@lru_cache(maxsize=1)
def s4_gauge_maps() -> Tensor:
    """Precompute u (p^-1 tau q) u^-1 for every u, tau, p, and q."""
    inverse, composition, _ = s4_tables()
    maps = torch.empty(S4_ORDER * S4_ORDER, S4_ORDER, S4_ORDER, dtype=torch.long)
    for u in range(S4_ORDER):
        for tau in range(S4_ORDER):
            candidate = u * S4_ORDER + tau
            for query in range(S4_ORDER):
                left = composition[inverse[query], tau]
                for key in range(S4_ORDER):
                    relative = composition[left, key]
                    maps[candidate, query, key] = composition[u, composition[relative, inverse[u]]]
    return maps


def circulant_relation_edges(tables: int, offsets: tuple[int, ...] = (1, 5)) -> Tensor:
    if tables < 2:
        raise ValueError("tables must be at least two")
    normalized = tuple(dict.fromkeys(offset % tables for offset in offsets if offset % tables))
    if not normalized:
        raise ValueError("offsets must contain a nonzero offset modulo tables")
    edges = [(table, (table + offset) % tables) for offset in normalized for table in range(tables)]
    if len(set(edges)) != len(edges):
        raise ValueError("offsets produce duplicate directed edges")
    return torch.tensor(edges, dtype=torch.long)


class GaugeAlignedS4Relation(nn.Module):
    """Shared relation tables over gauge-aligned local S4 comparison routes."""

    def __init__(
        self,
        tables: int,
        *,
        templates: int = 1,
        second_order: bool = False,
        edge_offsets: tuple[int, ...] = (1, 5),
    ) -> None:
        super().__init__()
        if tables < 1:
            raise ValueError("tables must be positive")
        if templates < 1 or templates > tables:
            raise ValueError("templates must be in [1, tables]")
        self.tables = int(tables)
        self.templates = int(templates)
        self.first_order = nn.Parameter(torch.zeros(templates, S4_ORDER))
        self.bias = nn.Parameter(torch.zeros(()))
        self.second_order = nn.Parameter(torch.zeros(S4_ORDER, S4_ORDER)) if second_order else None
        self.register_buffer("candidate_maps", s4_gauge_maps().clone())
        self.register_buffer("gauge_ids", torch.zeros(tables, dtype=torch.long))
        self.register_buffer("template_ids", torch.arange(tables, dtype=torch.long).remainder(templates))
        edges = circulant_relation_edges(tables, edge_offsets) if second_order else torch.empty(0, 2, dtype=torch.long)
        self.register_buffer("edges", edges)

    def set_structure(self, gauge_ids: Tensor, template_ids: Tensor | None = None) -> None:
        if tuple(gauge_ids.shape) != (self.tables,):
            raise ValueError(f"gauge_ids must have shape ({self.tables},)")
        if bool(((gauge_ids < 0) | (gauge_ids >= S4_ORDER * S4_ORDER)).any()):
            raise ValueError("gauge_ids must index an S4 x S4 gauge pair")
        self.gauge_ids.copy_(gauge_ids.to(self.gauge_ids))
        if template_ids is not None:
            if tuple(template_ids.shape) != (self.tables,):
                raise ValueError(f"template_ids must have shape ({self.tables},)")
            if bool(((template_ids < 0) | (template_ids >= self.templates)).any()):
                raise ValueError("template_ids exceed the template count")
            self.template_ids.copy_(template_ids.to(self.template_ids))

    def relation_codes(self, query_route: Tensor, key_route: Tensor) -> Tensor:
        if query_route.shape != key_route.shape or query_route.shape[-1] != self.tables:
            raise ValueError(f"routes must have identical shape [..., {self.tables}]")
        table = torch.arange(self.tables, device=query_route.device)
        maps = self.candidate_maps[self.gauge_ids].to(query_route.device)
        return maps[table, query_route.long(), key_route.long()]

    def score_aligned_routes(self, query_route: Tensor, key_route: Tensor) -> Tensor:
        codes = self.relation_codes(query_route, key_route)
        score = self.first_order[self.template_ids, codes].sum(dim=-1) / math.sqrt(self.tables)
        if self.second_order is not None:
            left = codes[..., self.edges[:, 0]]
            right = codes[..., self.edges[:, 1]]
            score = score + self.second_order[left, right].sum(dim=-1) / math.sqrt(self.edges.shape[0])
        return score + self.bias

    def score_matrix_routes(self, query_route: Tensor, key_route: Tensor) -> Tensor:
        if query_route.shape[:-2] != key_route.shape[:-2] or query_route.shape[-1] != self.tables:
            raise ValueError("query/key routes need matching prefixes and table count")
        query = query_route.unsqueeze(-2).expand(*query_route.shape[:-2], query_route.shape[-2], key_route.shape[-2], self.tables)
        key = key_route.unsqueeze(-3).expand_as(query)
        return self.score_aligned_routes(query, key)


def _cycle_type(permutation: tuple[int, ...]) -> tuple[int, ...]:
    seen: set[int] = set()
    cycles: list[int] = []
    for start in range(4):
        if start in seen:
            continue
        node = start
        size = 0
        while node not in seen:
            seen.add(node)
            node = permutation[node]
            size += 1
        cycles.append(size)
    return tuple(sorted(cycles, reverse=True))


def s4_fourier_energy(function: Tensor) -> dict[str, float]:
    """Exact central-isotypic energy of scalar functions on S4."""
    if function.shape[-1] != S4_ORDER:
        raise ValueError("S4 functions must have 24 entries")
    inverse, composition, length = s4_tables()
    class_order = ((1, 1, 1, 1), (2, 1, 1), (2, 2), (3, 1), (4,))
    class_index = {cycle: index for index, cycle in enumerate(class_order)}
    element_class = torch.tensor([class_index[_cycle_type(p)] for p in _PERMUTATIONS], dtype=torch.long)
    irreps = {
        "trivial": (1, (1, 1, 1, 1, 1)),
        "standard": (3, (3, 1, -1, 0, -1)),
        "two_two": (2, (2, 0, 2, -1, 0)),
        "standard_sign": (3, (3, -1, -1, 0, 1)),
        "sign": (1, (1, -1, 1, 1, -1)),
    }
    flat = function.detach().to(dtype=torch.float64, device="cpu").reshape(-1, S4_ORDER)
    total = flat.square().sum(dim=-1).clamp_min(1e-20)
    result: dict[str, float] = {}
    group = torch.arange(S4_ORDER)
    shifted = composition[inverse[:, None], group[None, :]]
    for name, (dimension, character_values) in irreps.items():
        character = torch.tensor(character_values, dtype=torch.float64)[element_class]
        projected = dimension / S4_ORDER * (character[:, None] * flat[:, shifted]).sum(dim=1)
        result[name] = float((projected.square().sum(dim=-1) / total).mean().item())
    length_projection = torch.empty_like(flat)
    for value in range(7):
        mask = length == value
        length_projection[:, mask] = flat[:, mask].mean(dim=-1, keepdim=True)
    result["coxeter_length"] = float((length_projection.square().sum(dim=-1) / total).mean().item())
    return result
