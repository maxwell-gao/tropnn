from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .surrogate import ste_heaviside

__all__ = ["CoxeterLUT", "CoxeterRoute", "K4FullLUT"]


_K4_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


@dataclass(frozen=True)
class CoxeterRoute:
    indices: Tensor
    adjacent_gaps: Tensor

    def detach(self) -> "CoxeterRoute":
        return CoxeterRoute(self.indices.detach(), self.adjacent_gaps.detach())


def _permutation_rank(order: Tensor) -> Tensor:
    rank = torch.zeros(order.shape[:-1], device=order.device, dtype=torch.long)
    for position, factorial in enumerate((6, 2, 1)):
        smaller = (order[..., position + 1 :] < order[..., position : position + 1]).sum(dim=-1)
        rank += smaller * factorial
    return rank


def _neighbor_table() -> Tensor:
    permutations = tuple(itertools.permutations(range(4)))
    lookup = {permutation: index for index, permutation in enumerate(permutations)}
    neighbors = torch.empty(24, 3, dtype=torch.long)
    for index, permutation in enumerate(permutations):
        for generator in range(3):
            neighbor = list(permutation)
            neighbor[generator], neighbor[generator + 1] = neighbor[generator + 1], neighbor[generator]
            neighbors[index, generator] = lookup[tuple(neighbor)]
    return neighbors


def _clique_anchors(input_dim: int, tables: int, policy: str, seed: int) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    offsets = torch.arange(4, dtype=torch.long)
    if policy in {"local", "cyclic"}:
        stride = 4 if policy == "local" else 1
        return (torch.arange(tables).unsqueeze(1) * stride + offsets) % input_dim
    if policy == "block":
        block = max(4, input_dim // max(1, tables))
        return (torch.arange(tables).unsqueeze(1) * block + offsets) % input_dim
    if policy == "expander":
        strides = torch.tensor((1, 17, 43, 97), dtype=torch.long)
        return (torch.arange(tables).unsqueeze(1) * 131 + strides) % input_dim
    if policy == "permuted":
        groups: list[Tensor] = []
        permutation = torch.randperm(input_dim, generator=generator)
        cursor = 0
        while len(groups) < tables:
            if cursor + 4 > input_dim:
                permutation = torch.randperm(input_dim, generator=generator)
                cursor = 0
            groups.append(permutation[cursor : cursor + 4])
            cursor += 4
        return torch.stack(groups)
    return torch.stack([torch.randperm(input_dim, generator=generator)[:4] for _ in range(tables)])


def _clique_edges(anchors: Tensor) -> Tensor:
    edges = torch.tensor(_K4_EDGES, dtype=torch.long)
    return anchors[:, edges]


class K4FullLUT(nn.Module):
    """Free 64-row LUT routed by the six edges of the same local K4 used by CoxeterLUT."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        tables: int = 16,
        comparisons: int = 6,
        anchor_policy: str = "permuted",
        seed: int = 0,
        lut_init_std: float = 0.0,
        use_output_scaling: bool = True,
        use_min_margin_ste: bool = True,
    ) -> None:
        super().__init__()
        if input_dim < 4:
            raise ValueError(f"K4FullLUT requires input_dim >= 4, got {input_dim}")
        if tables < 1:
            raise ValueError(f"tables must be positive, got {tables}")
        if comparisons != 6:
            raise ValueError("local K4 routing has exactly six pairwise comparisons")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.tables = int(tables)
        self.comparisons = 6
        self.table_size = 64
        self.payload_width = self.output_dim
        self.write_degree = self.output_dim
        self.output_scale = 1.0 / math.sqrt(self.tables) if use_output_scaling else 1.0
        self.use_min_margin_ste = bool(use_min_margin_ste)
        self.anchor_policy = anchor_policy

        clique = _clique_anchors(self.input_dim, self.tables, anchor_policy, seed + 101)
        self.register_buffer("clique_anchors", clique)
        self.register_buffer("anchors", _clique_edges(clique))
        self.register_buffer("powers", 2 ** torch.arange(self.comparisons, dtype=torch.long))
        self.thresholds = nn.Parameter(torch.zeros(self.tables, self.comparisons))
        generator = torch.Generator(device="cpu").manual_seed(seed + 211)
        self.lut = nn.Parameter(
            torch.randn(self.tables, self.table_size, self.output_dim, generator=generator) * lut_init_std
        )

    @property
    def payload_params(self) -> int:
        return self.lut.numel()

    @property
    def bias_generator_params(self) -> int:
        return self.payload_params

    @property
    def slope_coeff_params(self) -> int:
        return 0

    @property
    def slope_generator_params(self) -> int:
        return 0

    def payload_parameters(self) -> list[Tensor]:
        return [self.lut]

    def threshold_parameters(self) -> list[Tensor]:
        return [self.thresholds]

    def clear_packed_payload_cache(self) -> None:
        return None

    def route(self, x: Tensor) -> CoxeterRoute:
        a = self.anchors[:, :, 0].flatten()
        b = self.anchors[:, :, 1].flatten()
        margins = x[:, a].view(x.shape[0], self.tables, self.comparisons)
        margins = margins - x[:, b].view(x.shape[0], self.tables, self.comparisons)
        margins = margins - self.thresholds.to(device=x.device, dtype=x.dtype).unsqueeze(0)
        powers = self.powers.to(device=x.device).view(1, 1, self.comparisons)
        indices = ((margins > 0).to(torch.long) * powers).sum(dim=-1)
        return CoxeterRoute(indices, margins)

    def _lookup(self, indices: Tensor) -> Tensor:
        table_offsets = torch.arange(self.tables, device=indices.device).view(1, self.tables) * self.table_size
        flat_indices = (indices + table_offsets).reshape(-1)
        rows = self.lut.reshape(self.tables * self.table_size, self.output_dim).index_select(0, flat_indices)
        return rows.view(indices.shape[0], self.tables, self.output_dim)

    def _sum(self, payload: Tensor) -> Tensor:
        return payload.sum(dim=1) * self.output_scale

    def _ste_correction(self, route: CoxeterRoute, payload: Tensor) -> Tensor:
        if self.use_min_margin_ste:
            bit = route.adjacent_gaps.abs().argmin(dim=-1)
            margin = route.adjacent_gaps.gather(-1, bit.unsqueeze(-1)).squeeze(-1)
            neighbor = route.indices ^ (2 ** bit).long()
            ste = ste_heaviside(margin) - (margin > 0).to(margin.dtype)
            return self._sum((self._lookup(neighbor) - payload) * ste.unsqueeze(-1))

        correction = torch.zeros(payload.shape[0], self.output_dim, device=payload.device, dtype=payload.dtype)
        for bit in range(self.comparisons):
            margin = route.adjacent_gaps[..., bit]
            ste = ste_heaviside(margin) - (margin > 0).to(margin.dtype)
            correction += self._sum(
                (self._lookup(route.indices ^ int(self.powers[bit].item())) - payload) * ste.unsqueeze(-1)
            )
        return correction

    def compute(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if x.ndim != 2 or x.shape[-1] != self.input_dim:
            raise ValueError(f"K4FullLUT expected [batch, {self.input_dim}], got {tuple(x.shape)}")
        input_dtype = x.dtype
        route = self.route(x.float())
        payload = self._lookup(route.indices)
        output = self._sum(payload)
        if self.training and (x.requires_grad or self.thresholds.requires_grad):
            output = output + self._ste_correction(route, payload)
        return output.to(input_dtype), route.indices

    def forward(self, x: Tensor) -> Tensor:
        output, _ = self.compute(x)
        return output


class CoxeterLUT(nn.Module):
    """Unary full-vector LUT routed by legal chambers of the local S4 braid arrangement."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        tables: int = 16,
        comparisons: int = 6,
        anchor_policy: str = "permuted",
        seed: int = 0,
        lut_init_std: float = 0.0,
        use_output_scaling: bool = True,
        use_min_margin_ste: bool = True,
    ) -> None:
        super().__init__()
        if input_dim < 4:
            raise ValueError(f"CoxeterLUT requires input_dim >= 4, got {input_dim}")
        if tables < 1:
            raise ValueError(f"tables must be positive, got {tables}")
        if comparisons != 6:
            raise ValueError("local S4 routing has exactly six pairwise comparisons")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.tables = int(tables)
        self.comparisons = 6
        self.table_size = 24
        self.payload_width = self.output_dim
        self.write_degree = self.output_dim
        self.output_scale = 1.0 / math.sqrt(self.tables) if use_output_scaling else 1.0
        self.use_min_margin_ste = bool(use_min_margin_ste)
        self.anchor_policy = anchor_policy
        self.register_buffer("anchors", _clique_anchors(self.input_dim, self.tables, anchor_policy, seed + 101))
        self.register_buffer("neighbors", _neighbor_table())
        self.thresholds = nn.Parameter(torch.zeros(self.tables, 4))
        generator = torch.Generator(device="cpu").manual_seed(seed + 211)
        self.lut = nn.Parameter(
            torch.randn(self.tables, self.table_size, self.output_dim, generator=generator) * lut_init_std
        )

    @property
    def payload_params(self) -> int:
        return self.lut.numel()

    @property
    def bias_generator_params(self) -> int:
        return self.payload_params

    @property
    def slope_coeff_params(self) -> int:
        return 0

    @property
    def slope_generator_params(self) -> int:
        return 0

    def payload_parameters(self) -> list[Tensor]:
        return [self.lut]

    def threshold_parameters(self) -> list[Tensor]:
        return [self.thresholds]

    def clear_packed_payload_cache(self) -> None:
        return None

    def route(self, x: Tensor) -> CoxeterRoute:
        selected = x[:, self.anchors.flatten()].view(x.shape[0], self.tables, 4)
        shifted = selected - self.thresholds.to(device=x.device, dtype=x.dtype).unsqueeze(0)
        order = torch.argsort(shifted, dim=-1, stable=True)
        sorted_values = shifted.gather(-1, order)
        return CoxeterRoute(_permutation_rank(order), sorted_values[..., 1:] - sorted_values[..., :-1])

    def _lookup(self, indices: Tensor) -> Tensor:
        table_offsets = torch.arange(self.tables, device=indices.device).view(1, self.tables) * self.table_size
        flat_indices = (indices + table_offsets).reshape(-1)
        rows = self.lut.reshape(self.tables * self.table_size, self.output_dim).index_select(0, flat_indices)
        return rows.view(indices.shape[0], self.tables, self.output_dim)

    def _sum(self, payload: Tensor) -> Tensor:
        return payload.sum(dim=1) * self.output_scale

    def _ste_correction(self, route: CoxeterRoute, payload: Tensor) -> Tensor:
        neighbor_indices = self.neighbors[route.indices]
        if self.use_min_margin_ste:
            generator = route.adjacent_gaps.argmin(dim=-1)
            margin = route.adjacent_gaps.gather(-1, generator.unsqueeze(-1)).squeeze(-1)
            neighbor = neighbor_indices.gather(-1, generator.unsqueeze(-1)).squeeze(-1)
            ste = ste_heaviside(margin) - (margin > 0).to(margin.dtype)
            return self._sum((self._lookup(neighbor) - payload) * ste.unsqueeze(-1))

        correction = torch.zeros(payload.shape[0], self.output_dim, device=payload.device, dtype=payload.dtype)
        for generator in range(3):
            margin = route.adjacent_gaps[..., generator]
            ste = ste_heaviside(margin) - (margin > 0).to(margin.dtype)
            correction += self._sum(
                (self._lookup(neighbor_indices[..., generator]) - payload) * ste.unsqueeze(-1)
            )
        return correction

    def compute(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if x.ndim != 2 or x.shape[-1] != self.input_dim:
            raise ValueError(f"CoxeterLUT expected [batch, {self.input_dim}], got {tuple(x.shape)}")
        input_dtype = x.dtype
        route = self.route(x.float())
        payload = self._lookup(route.indices)
        output = self._sum(payload)
        if self.training and (x.requires_grad or self.thresholds.requires_grad):
            output = output + self._ste_correction(route, payload)
        return output.to(input_dtype), route.indices

    def forward(self, x: Tensor) -> Tensor:
        output, _ = self.compute(x)
        return output
