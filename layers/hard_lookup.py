"""Shared hard routing, lookup, and surrogate-gradient semantics.

This module is the implementation boundary for trainable lookup routers used by
the controlled TropNN experiments.  It deliberately knows nothing about
datasets, teachers, training loops, metrics, reports, or artifact schemas.

The same implementation covers the Cartesian product

``pair | unary`` predicate x ``flat | adaptive`` topology,

plus the two surrogate families used in the repository:

* ``soft_product``: hard one-hot forward with the product-of-branches soft
  backward used by the early address/action factorials;
* ``local_counterfactual``: hard forward with credit from the nearest executed
  wall and the exact counterfactual row reached by crossing that wall.

Adaptive trees may share one predicate support per level (MADDNESS-style) or
store a distinct support at every node.  These are genuine model differences,
so they are explicit configuration rather than separate implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

Predicate = Literal["pair", "unary"]
Topology = Literal["flat", "adaptive"]
SupportLayout = Literal["level", "node"]
BitOrder = Literal["msb", "lsb"]
TieBreak = Literal["positive", "nonnegative"]
TrainingSurrogate = Literal["none", "soft_product", "local_counterfactual"]
Action = Literal["constant", "diagonal_live"]

__all__ = [
    "Action",
    "BitOrder",
    "HardLookupRoute",
    "HardLookupRouter",
    "HardLookupSpec",
    "ProductGridLookupRouter",
    "ProductGridRoute",
    "Predicate",
    "SupportLayout",
    "TieBreak",
    "Topology",
    "TrainingSurrogate",
    "adaptive_leaf_probabilities",
    "adaptive_hard_route_lookahead",
    "flat_leaf_probabilities",
    "forced_neighbor_codes",
    "gather_lookup_rows",
    "hard_forward_soft_backward",
    "hard_route",
    "pack_branches",
    "predicate_values",
    "straight_through_bits",
    "sum_lookup_rows",
    "weighted_neighbor_delta",
]


@dataclass(frozen=True)
class HardLookupSpec:
    input_dim: int
    output_dim: int
    depth: int
    predicate: Predicate
    topology: Topology
    support_layout: SupportLayout = "level"
    bit_order: BitOrder = "msb"
    tie_break: TieBreak = "nonnegative"
    surrogate: TrainingSurrogate = "local_counterfactual"
    action: Action = "constant"
    tau: float = 1.0

    def __post_init__(self) -> None:
        if self.input_dim < 1 or self.output_dim < 1 or self.depth < 1:
            raise ValueError("input_dim, output_dim, and depth must be positive")
        if self.predicate not in {"pair", "unary"}:
            raise ValueError(f"unsupported predicate {self.predicate!r}")
        if self.topology not in {"flat", "adaptive"}:
            raise ValueError(f"unsupported topology {self.topology!r}")
        if self.support_layout not in {"level", "node"}:
            raise ValueError(f"unsupported support layout {self.support_layout!r}")
        if self.topology == "flat" and self.support_layout != "level":
            raise ValueError("flat routing requires support_layout='level'")
        if self.topology == "adaptive" and self.bit_order != "msb":
            raise ValueError("adaptive tree leaf numbering requires bit_order='msb'")
        if self.bit_order not in {"msb", "lsb"}:
            raise ValueError(f"unsupported bit order {self.bit_order!r}")
        if self.tie_break not in {"positive", "nonnegative"}:
            raise ValueError(f"unsupported tie break {self.tie_break!r}")
        if self.surrogate not in {"none", "soft_product", "local_counterfactual"}:
            raise ValueError(f"unsupported surrogate {self.surrogate!r}")
        if self.action not in {"constant", "diagonal_live"}:
            raise ValueError(f"unsupported action {self.action!r}")
        if self.action == "diagonal_live" and self.input_dim != self.output_dim:
            raise ValueError("diagonal_live action requires input_dim == output_dim")
        if self.tau <= 0:
            raise ValueError("tau must be positive")

    @property
    def rows(self) -> int:
        return 1 << self.depth

    @property
    def nodes(self) -> int:
        return self.rows - 1

    @property
    def support_count(self) -> int:
        return self.nodes if self.topology == "adaptive" and self.support_layout == "node" else self.depth

    @property
    def threshold_count(self) -> int:
        return self.nodes if self.topology == "adaptive" else self.depth


@dataclass(frozen=True)
class HardLookupRoute:
    """The exact branches executed by a hard lookup route."""

    codes: Tensor
    margins: Tensor
    branches: Tensor

    def detach(self) -> "HardLookupRoute":
        return HardLookupRoute(self.codes.detach(), self.margins.detach(), self.branches.detach())


@dataclass(frozen=True)
class ProductGridRoute:
    """Exact mixed-radix route for a product of ordered scalar bins."""

    codes: Tensor
    digits: Tensor
    margins: Tensor
    branches: Tensor

    def detach(self) -> "ProductGridRoute":
        return ProductGridRoute(
            self.codes.detach(),
            self.digits.detach(),
            self.margins.detach(),
            self.branches.detach(),
        )


def _flatten_input(x: Tensor) -> tuple[Tensor, torch.Size]:
    if x.ndim < 2:
        raise ValueError("input must have at least a batch and feature dimension")
    return x.reshape(-1, x.shape[-1]), x.shape[:-1]


def _normalize_supports(supports: Tensor, predicate: Predicate) -> Tensor:
    supports = supports.to(dtype=torch.long)
    if predicate == "unary":
        if supports.ndim == 2:
            return supports.unsqueeze(-1)
        if supports.ndim == 3 and supports.shape[-1] in {1, 2}:
            return supports
        raise ValueError("unary supports must be [tables,count] or [tables,count,1|2]")
    if supports.ndim != 3 or supports.shape[-1] != 2:
        raise ValueError("pair supports must be [tables,count,2]")
    return supports


def _take_positive(margins: Tensor, tie_break: TieBreak) -> Tensor:
    return margins > 0 if tie_break == "positive" else margins >= 0


def predicate_values(x: Tensor, supports: Tensor, predicate: Predicate) -> Tensor:
    """Evaluate every stored predicate support without applying thresholds."""

    flat, prefix = _flatten_input(x)
    normalized = _normalize_supports(supports, predicate).to(device=x.device)
    first = flat[:, normalized[..., 0]]
    values = first if predicate == "unary" else first - flat[:, normalized[..., 1]]
    return values.reshape(*prefix, *values.shape[1:])


def pack_branches(branches: Tensor, bit_order: BitOrder = "msb") -> Tensor:
    """Pack the final branch dimension into an integer leaf code."""

    if branches.ndim < 1:
        raise ValueError("branches must have a branch dimension")
    depth = branches.shape[-1]
    if bit_order == "msb":
        powers = 2 ** torch.arange(depth - 1, -1, -1, device=branches.device)
    elif bit_order == "lsb":
        powers = 2 ** torch.arange(depth, device=branches.device)
    else:
        raise ValueError(f"unsupported bit order {bit_order!r}")
    return (branches.to(torch.int64) * powers).sum(dim=-1)


def _validated_route_inputs(
    x: Tensor,
    supports: Tensor,
    thresholds: Tensor,
    spec: HardLookupSpec,
) -> tuple[Tensor, torch.Size, Tensor, Tensor]:
    flat, prefix = _flatten_input(x)
    if flat.shape[-1] != spec.input_dim:
        raise ValueError(f"expected input_dim={spec.input_dim}, got {flat.shape[-1]}")
    normalized = _normalize_supports(supports, spec.predicate).to(device=x.device)
    if normalized.shape[1] != spec.support_count:
        raise ValueError(f"expected {spec.support_count} supports per table, got {normalized.shape[1]}")
    if thresholds.ndim != 2 or thresholds.shape != (normalized.shape[0], spec.threshold_count):
        raise ValueError(f"thresholds must be [{normalized.shape[0]},{spec.threshold_count}], got {tuple(thresholds.shape)}")
    values = predicate_values(flat, normalized, spec.predicate).reshape(flat.shape[0], normalized.shape[0], normalized.shape[1])
    return flat, prefix, values, thresholds.to(device=x.device, dtype=values.dtype)


def hard_route(x: Tensor, supports: Tensor, thresholds: Tensor, spec: HardLookupSpec) -> HardLookupRoute:
    """Execute one flat fern or one adaptive tree per table."""

    flat, prefix, values, threshold_values = _validated_route_inputs(x, supports, thresholds, spec)
    items, tables = flat.shape[0], values.shape[1]
    if spec.topology == "flat":
        margins = values - threshold_values.unsqueeze(0)
        branches = _take_positive(margins, spec.tie_break)
        return HardLookupRoute(
            pack_branches(branches, spec.bit_order).reshape(*prefix, tables),
            margins.reshape(*prefix, tables, spec.depth),
            branches.reshape(*prefix, tables, spec.depth),
        )

    path_code = torch.zeros((items, tables), dtype=torch.int64, device=x.device)
    table_ids = torch.arange(tables, device=x.device)[None, :]
    path_margins: list[Tensor] = []
    path_branches: list[Tensor] = []
    for level in range(spec.depth):
        node = (2**level - 1) + path_code
        if spec.support_layout == "level":
            value = values[:, :, level]
        else:
            value = values.gather(2, node.unsqueeze(-1)).squeeze(-1)
        threshold = threshold_values[table_ids, node]
        margin = value - threshold
        branch = _take_positive(margin, spec.tie_break)
        path_margins.append(margin)
        path_branches.append(branch)
        path_code = 2 * path_code + branch.to(torch.int64)
    branches = torch.stack(path_branches, dim=-1)
    codes = pack_branches(branches, spec.bit_order)
    return HardLookupRoute(
        codes.reshape(*prefix, tables),
        torch.stack(path_margins, dim=-1).reshape(*prefix, tables, spec.depth),
        branches.reshape(*prefix, tables, spec.depth),
    )


def adaptive_hard_route_lookahead(
    x: Tensor,
    supports: Tensor,
    thresholds: Tensor,
    spec: HardLookupSpec,
    lookahead: int,
) -> HardLookupRoute:
    """Execute an adaptive tree in groups of speculative comparison levels.

    ``lookahead=1`` is the ordinary depth-round traversal.  A group of ``g``
    levels evaluates all ``2**g - 1`` predicates in the currently selected
    subtree before decoding its ``g`` path bits.  The route is bit-exact with
    :func:`hard_route`; only the amount of speculative work and the number of
    threshold-dependent comparison rounds change.
    """

    if spec.topology != "adaptive":
        raise ValueError("lookahead execution requires an adaptive route")
    if lookahead < 1 or lookahead > spec.depth:
        raise ValueError("lookahead must be in [1, depth]")
    flat, prefix, values, threshold_values = _validated_route_inputs(x, supports, thresholds, spec)
    items, tables = flat.shape[0], values.shape[1]
    path_code = torch.zeros((items, tables), dtype=torch.int64, device=x.device)
    path_margins: list[Tensor] = []
    path_branches: list[Tensor] = []
    expanded_thresholds = threshold_values.unsqueeze(0).expand(items, -1, -1)

    for block_start in range(0, spec.depth, lookahead):
        width = min(lookahead, spec.depth - block_start)
        local_nodes = torch.arange(2**width - 1, device=x.device)
        relative_levels = torch.floor(torch.log2((local_nodes + 1).to(torch.float32))).to(torch.int64)
        relative_prefixes = local_nodes - (2**relative_levels - 1)
        absolute_levels = block_start + relative_levels
        absolute_nodes = (
            (2**absolute_levels - 1).view(1, 1, -1)
            + torch.bitwise_left_shift(path_code.unsqueeze(-1), relative_levels.view(1, 1, -1))
            + relative_prefixes.view(1, 1, -1)
        )
        if spec.support_layout == "level":
            candidate_values = values[:, :, absolute_levels]
        else:
            candidate_values = values.gather(2, absolute_nodes)
        candidate_thresholds = expanded_thresholds.gather(2, absolute_nodes)
        candidate_margins = candidate_values - candidate_thresholds
        candidate_branches = _take_positive(candidate_margins, spec.tie_break)

        local_code = torch.zeros_like(path_code)
        for relative_level in range(width):
            local_node = (2**relative_level - 1) + local_code
            margin = candidate_margins.gather(2, local_node.unsqueeze(-1)).squeeze(-1)
            branch = candidate_branches.gather(2, local_node.unsqueeze(-1)).squeeze(-1)
            path_margins.append(margin)
            path_branches.append(branch)
            local_code = 2 * local_code + branch.to(torch.int64)
        path_code = torch.bitwise_left_shift(path_code, width) + local_code

    branches = torch.stack(path_branches, dim=-1)
    return HardLookupRoute(
        path_code.reshape(*prefix, tables),
        torch.stack(path_margins, dim=-1).reshape(*prefix, tables, spec.depth),
        branches.reshape(*prefix, tables, spec.depth),
    )


def forced_neighbor_codes(
    x: Tensor,
    supports: Tensor,
    thresholds: Tensor,
    spec: HardLookupSpec,
    forced_level: Tensor,
    original_branches: Tensor,
) -> Tensor:
    """Cross one executed wall and, for a tree, execute the new suffix."""

    if forced_level.shape != original_branches.shape[:-1]:
        raise ValueError("forced_level shape must match branches without depth")
    if original_branches.shape[-1] != spec.depth:
        raise ValueError("original branch depth mismatch")
    if spec.topology == "flat":
        current = pack_branches(original_branches, spec.bit_order)
        shift = spec.depth - 1 - forced_level if spec.bit_order == "msb" else forced_level
        return current ^ torch.bitwise_left_shift(torch.ones_like(current), shift)

    flat, prefix, values, threshold_values = _validated_route_inputs(x, supports, thresholds, spec)
    items, tables = flat.shape[0], values.shape[1]
    forced = forced_level.reshape(items, tables)
    original = original_branches.reshape(items, tables, spec.depth)
    path_code = torch.zeros((items, tables), dtype=torch.int64, device=x.device)
    table_ids = torch.arange(tables, device=x.device)[None, :]
    new_branches: list[Tensor] = []
    for level in range(spec.depth):
        node = (2**level - 1) + path_code
        if spec.support_layout == "level":
            value = values[:, :, level]
        else:
            value = values.gather(2, node.unsqueeze(-1)).squeeze(-1)
        natural = _take_positive(value - threshold_values[table_ids, node], spec.tie_break)
        branch = torch.where(forced == level, ~original[:, :, level], natural)
        new_branches.append(branch)
        path_code = 2 * path_code + branch.to(torch.int64)
    return pack_branches(torch.stack(new_branches, dim=-1), spec.bit_order).reshape(*prefix, tables)


def gather_lookup_rows(rows: Tensor, codes: Tensor) -> Tensor:
    """Gather one payload row per table without reducing the table axis."""

    if rows.ndim != 3:
        raise ValueError("rows must be [tables,row_count,output_dim]")
    if codes.ndim < 1 or codes.shape[-1] != rows.shape[0]:
        raise ValueError("codes must end in the rows table dimension")
    flat = codes.reshape(-1, rows.shape[0])
    table_ids = torch.arange(rows.shape[0], device=codes.device)[None, :]
    selected = rows.to(device=codes.device)[table_ids, flat]
    return selected.reshape(*codes.shape[:-1], rows.shape[0], rows.shape[-1])


def sum_lookup_rows(
    rows: Tensor,
    codes: Tensor,
    *,
    accumulation_dtype: torch.dtype | None = None,
    target_bytes: int = 16 * 1024 * 1024,
) -> Tensor:
    """Sum one row per table while bounding the temporary gather size."""

    if rows.ndim != 3 or codes.shape[-1] != rows.shape[0]:
        raise ValueError("rows/codes table dimensions do not match")
    prefix, tables = codes.shape[:-1], rows.shape[0]
    items = max(1, codes.numel() // tables)
    flat_codes = codes.reshape(items, tables)
    dtype = rows.dtype if accumulation_dtype is None else accumulation_dtype
    payload = rows.to(device=codes.device)
    output = torch.zeros(items, rows.shape[-1], device=codes.device, dtype=dtype)
    element_size = payload.element_size()
    bytes_per_table = max(1, items * rows.shape[-1] * element_size)
    chunk = max(1, min(tables, target_bytes // bytes_per_table))
    for start in range(0, tables, chunk):
        stop = min(start + chunk, tables)
        table_ids = torch.arange(start, stop, device=codes.device)[None, :]
        selected = payload[table_ids, flat_codes[:, start:stop]]
        output = output + selected.sum(dim=1, dtype=dtype)
    return output.reshape(*prefix, rows.shape[-1])


def weighted_neighbor_delta(
    rows: Tensor,
    current_codes: Tensor,
    neighbor_codes: Tensor,
    weights: Tensor,
    *,
    target_bytes: int = 16 * 1024 * 1024,
) -> Tensor:
    """Sum weighted counterfactual row differences over tables and neighbors."""

    if rows.ndim != 3 or current_codes.shape[-1] != rows.shape[0]:
        raise ValueError("rows/current_codes table dimensions do not match")
    prefix, tables = current_codes.shape[:-1], rows.shape[0]
    items = max(1, current_codes.numel() // tables)
    current = current_codes.reshape(items, tables)
    if neighbor_codes.shape == current_codes.shape:
        neighbors = 1
        neighbor = neighbor_codes.reshape(items, tables, 1)
        weight = weights.reshape(items, tables, 1, 1)
    else:
        if neighbor_codes.shape[:-1] != (*current_codes.shape,):
            raise ValueError("neighbor_codes must be current_codes or current_codes plus a neighbor axis")
        neighbors = neighbor_codes.shape[-1]
        neighbor = neighbor_codes.reshape(items, tables, neighbors)
        weight = weights.reshape(items, tables, neighbors, 1)
    payload = rows.to(device=current_codes.device, dtype=torch.float32)
    output = torch.zeros(items, rows.shape[-1], device=current_codes.device, dtype=torch.float32)
    bytes_per_table = max(1, items * (neighbors + 1) * rows.shape[-1] * payload.element_size())
    chunk = max(1, min(tables, target_bytes // bytes_per_table))
    for start in range(0, tables, chunk):
        stop = min(start + chunk, tables)
        table_ids = torch.arange(start, stop, device=current_codes.device)[None, :]
        current_rows = payload[table_ids, current[:, start:stop]].unsqueeze(-2)
        neighbor_rows = payload[table_ids.unsqueeze(-1), neighbor[:, start:stop]]
        output = output + (weight[:, start:stop].float() * (neighbor_rows - current_rows)).sum(dim=(1, 2))
    return output.reshape(*prefix, rows.shape[-1])


def straight_through_bits(margins: Tensor, tau: float, tie_break: TieBreak = "nonnegative") -> Tensor:
    if tau <= 0:
        raise ValueError("tau must be positive")
    hard = _take_positive(margins, tie_break).to(margins.dtype)
    soft = torch.sigmoid(margins / tau)
    return hard + soft - soft.detach()


def flat_leaf_probabilities(bits: Tensor, bit_order: BitOrder = "msb") -> Tensor:
    """Product-of-bits leaf weights for a flat code."""

    if bits.ndim < 1:
        raise ValueError("bits must have a depth dimension")
    depth = bits.shape[-1]
    codes = torch.arange(2**depth, device=bits.device)
    shifts = torch.arange(depth - 1, -1, -1, device=bits.device) if bit_order == "msb" else torch.arange(depth, device=bits.device)
    patterns = ((codes[:, None] >> shifts[None, :]) & 1).to(bits.dtype)
    factors = bits.unsqueeze(-2) * patterns + (1.0 - bits.unsqueeze(-2)) * (1.0 - patterns)
    return factors.prod(dim=-1)


def adaptive_leaf_probabilities(bits: Tensor, depth: int) -> Tensor:
    """Propagate soft mass through a complete binary tree."""

    if bits.shape[-1] != 2**depth - 1:
        raise ValueError("tree bits do not match a complete binary tree")
    mass = torch.ones((*bits.shape[:-1], 1), device=bits.device, dtype=bits.dtype)
    offset = 0
    for level in range(depth):
        width = 2**level
        level_bits = bits[..., offset : offset + width]
        mass = torch.stack((mass * (1.0 - level_bits), mass * level_bits), dim=-1).reshape(*bits.shape[:-1], -1)
        offset += width
    return mass


class _HardForwardSoftBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, hard: Tensor, soft: Tensor) -> Tensor:
        del ctx, soft
        return hard.clone()

    @staticmethod
    def backward(ctx: object, gradient: Tensor) -> tuple[None, Tensor]:
        del ctx
        return None, gradient


def hard_forward_soft_backward(hard: Tensor, soft: Tensor) -> Tensor:
    """Use ``hard`` exactly in forward and route all output credit to ``soft``."""

    if hard.shape != soft.shape:
        raise ValueError("hard and soft outputs must have identical shapes")
    return _HardForwardSoftBackward.apply(hard, soft)


class HardLookupRouter(nn.Module):
    """A data-agnostic hard route -> additive row lookup module.

    All tensors are supplied explicitly.  Random initialization and scientific
    protocol choices belong to experiment code, not this implementation.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        depth: int,
        predicate: Predicate,
        topology: Topology,
        supports: Tensor,
        thresholds: Tensor,
        rows: Tensor,
        support_layout: SupportLayout = "level",
        bit_order: BitOrder = "msb",
        tie_break: TieBreak = "nonnegative",
        surrogate: TrainingSurrogate = "local_counterfactual",
        action: Action = "constant",
        slopes: Tensor | None = None,
        tau: float = 1.0,
        trainable_thresholds: bool = True,
        trainable_rows: bool = True,
        trainable_slopes: bool = True,
        support_scores: Tensor | None = None,
        support_tau: float = 1.0,
        trainable_supports: bool = False,
    ) -> None:
        super().__init__()
        self.spec = HardLookupSpec(
            int(input_dim),
            int(output_dim),
            int(depth),
            predicate,
            topology,
            support_layout,
            bit_order,
            tie_break,
            surrogate,
            action,
            float(tau),
        )
        normalized = _normalize_supports(supports.detach().clone(), predicate)
        if normalized.shape[1] != self.spec.support_count:
            raise ValueError(f"expected {self.spec.support_count} supports, got {normalized.shape[1]}")
        tables = normalized.shape[0]
        if normalized.numel() and (int(normalized.min()) < 0 or int(normalized.max()) >= self.spec.input_dim):
            raise ValueError("support coordinate out of range")
        if thresholds.shape != (tables, self.spec.threshold_count):
            raise ValueError(f"thresholds must be [{tables},{self.spec.threshold_count}]")
        if rows.shape != (tables, self.spec.rows, self.spec.output_dim):
            raise ValueError(f"rows must be [{tables},{self.spec.rows},{self.spec.output_dim}]")
        self.register_buffer("supports", normalized)
        self.register_buffer("initial_thresholds", thresholds.detach().clone())
        self._register_state("thresholds", thresholds.detach().clone(), trainable_thresholds)
        self._register_state("rows", rows.detach().clone(), trainable_rows)
        if action == "diagonal_live":
            initial_slopes = torch.zeros_like(rows) if slopes is None else slopes.detach().clone()
            if initial_slopes.shape != rows.shape:
                raise ValueError("slopes must have the same shape as rows")
            self._register_state("slopes", initial_slopes, trainable_slopes)
        elif slopes is not None:
            raise ValueError("slopes are only valid for action='diagonal_live'")
        else:
            self.register_parameter("slopes", None)
        if support_scores is None:
            self.register_parameter("support_scores", None)
        else:
            if predicate != "unary" or topology != "adaptive" or support_layout != "level":
                raise ValueError("learned supports currently require unary adaptive level-shared routing")
            expected = (tables, self.spec.depth, self.spec.input_dim)
            if support_scores.shape != expected:
                raise ValueError(f"support_scores must be {expected}")
            if self.spec.input_dim < 2:
                raise ValueError("learned supports require input_dim >= 2")
            selected = support_scores.argmax(dim=-1)
            if not torch.equal(selected.cpu(), normalized[..., 0].cpu()):
                raise ValueError("initial supports must equal support_scores.argmax(-1)")
            self.support_scores = nn.Parameter(support_scores.detach().clone(), requires_grad=trainable_supports)
        if support_tau <= 0:
            raise ValueError("support_tau must be positive")
        self.support_tau = float(support_tau)
        self.support_learning = bool(trainable_supports)

    def _register_state(self, name: str, value: Tensor, trainable: bool) -> None:
        if trainable:
            setattr(self, name, nn.Parameter(value))
        else:
            self.register_buffer(name, value)

    @property
    def input_dim(self) -> int:
        return self.spec.input_dim

    @property
    def output_dim(self) -> int:
        return self.spec.output_dim

    @property
    def depth(self) -> int:
        return self.spec.depth

    @property
    def tables(self) -> int:
        return self.rows.shape[0]

    @property
    def predicate(self) -> Predicate:
        return self.spec.predicate

    @property
    def topology(self) -> Topology:
        return self.spec.topology

    @property
    def tau(self) -> float:
        return self.spec.tau

    def selected_supports(self) -> Tensor:
        if self.support_scores is None:
            return self.supports
        return self.support_scores.argmax(dim=-1)

    def set_support_learning(self, enabled: bool) -> None:
        if self.support_scores is None and enabled:
            raise ValueError("this router has no learnable support scores")
        self.support_learning = bool(enabled)
        if self.support_scores is not None:
            self.support_scores.requires_grad_(enabled)

    def _active_supports(self, supports: Tensor | None) -> Tensor:
        return self.selected_supports() if supports is None else supports

    def route(self, x: Tensor, *, supports: Tensor | None = None) -> HardLookupRoute:
        return hard_route(x, self._active_supports(supports), self.thresholds, self.spec)

    def hard_codes(self, x: Tensor, *, supports: Tensor | None = None) -> Tensor:
        return self.route(x, supports=supports).codes

    def selected_actions(self, x: Tensor, codes: Tensor) -> Tensor:
        actions = gather_lookup_rows(self.rows, codes)
        if self.slopes is not None:
            actions = actions + gather_lookup_rows(self.slopes, codes) * x.unsqueeze(-2)
        return actions

    def hard_output(self, x: Tensor, *, supports: Tensor | None = None) -> tuple[Tensor, Tensor]:
        route = self.route(x, supports=supports)
        return self.selected_actions(x, route.codes).sum(dim=-2), route.codes

    def neighboring_codes(
        self,
        x: Tensor,
        route: HardLookupRoute,
        nearest: Tensor,
        *,
        supports: Tensor | None = None,
    ) -> Tensor:
        return forced_neighbor_codes(
            x,
            self._active_supports(supports),
            self.thresholds,
            self.spec,
            nearest,
            route.branches,
        )

    def leaf_probabilities(self, x: Tensor, *, supports: Tensor | None = None) -> Tensor:
        active_supports = self._active_supports(supports)
        if self.topology == "flat":
            route = hard_route(x, active_supports, self.thresholds, self.spec)
            bits = straight_through_bits(route.margins, self.tau, self.spec.tie_break)
            return flat_leaf_probabilities(bits, self.spec.bit_order)

        _flat, prefix, values, threshold_values = _validated_route_inputs(x, active_supports, self.thresholds, self.spec)
        node_margins: list[Tensor] = []
        for level in range(self.depth):
            start, stop = 2**level - 1, 2 ** (level + 1) - 1
            if self.spec.support_layout == "level":
                value = values[:, :, level : level + 1].expand(-1, -1, 2**level)
            else:
                value = values[:, :, start:stop]
            node_margins.append(value - threshold_values[None, :, start:stop])
        margins = torch.cat(node_margins, dim=-1)
        bits = straight_through_bits(margins, self.tau, self.spec.tie_break)
        probabilities = adaptive_leaf_probabilities(bits, self.depth)
        return probabilities.reshape(*prefix, self.tables, self.spec.rows)

    def soft_product_output(self, x: Tensor, *, supports: Tensor | None = None) -> Tensor:
        leaf = self.leaf_probabilities(x, supports=supports)
        soft = torch.einsum("...tr,tro->...o", leaf, self.rows)
        if self.slopes is not None:
            soft = soft + torch.einsum("...tr,tro->...o", leaf, self.slopes) * x
        hard = self.hard_output(x, supports=supports)[0]
        return hard_forward_soft_backward(hard, soft)

    def local_counterfactual_output(self, x: Tensor, *, supports: Tensor | None = None) -> Tensor:
        route = self.route(x, supports=supports)
        current = self.selected_actions(x, route.codes)
        hard = current.sum(dim=-2)
        nearest = route.margins.abs().argmin(dim=-1)
        neighbor_codes = self.neighboring_codes(x, route, nearest, supports=supports)
        neighbor = self.selected_actions(x, neighbor_codes)
        chosen_margin = route.margins.gather(-1, nearest.unsqueeze(-1)).squeeze(-1)
        chosen_branch = route.branches.gather(-1, nearest.unsqueeze(-1)).squeeze(-1)
        direction = (2 * chosen_branch.to(x.dtype) - 1).unsqueeze(-1) * (current - neighbor)
        gate = torch.sigmoid(chosen_margin / self.tau)
        correction = ((gate - gate.detach()).unsqueeze(-1) * direction.detach()).sum(dim=-2)
        return hard + correction

    def _alternate_support(
        self,
        x: Tensor,
        supports: Tensor,
        route: HardLookupRoute,
        nearest: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if self.support_scores is None:
            raise RuntimeError("support counterfactual requires support_scores")
        flat, prefix = _flatten_input(x)
        items, tables = flat.shape[0], self.tables
        nearest_flat = nearest.reshape(items, tables)
        margins = route.margins.reshape(items, tables, self.depth)
        branches = route.branches.reshape(items, tables, self.depth)
        selected = _normalize_supports(supports, "unary")[..., 0]
        table_ids = torch.arange(tables, device=x.device)[None, :]
        ranked = self.support_scores.argsort(dim=-1, descending=True)
        candidates = ranked[table_ids, nearest_flat]
        current_indices = selected.to(device=x.device)[table_ids, nearest_flat]
        current_value = flat.gather(1, current_indices)
        chosen_margin = margins.gather(-1, nearest_flat.unsqueeze(-1)).squeeze(-1)
        chosen_branch = branches.gather(-1, nearest_flat.unsqueeze(-1)).squeeze(-1)
        chosen_threshold = current_value - chosen_margin
        values = flat[:, None, :].expand(items, tables, self.input_dim).gather(2, candidates)
        flips = _take_positive(values - chosen_threshold.unsqueeze(-1), self.spec.tie_break) != chosen_branch.unsqueeze(-1)
        flips = flips & (candidates != current_indices.unsqueeze(-1))
        has_flip = flips.any(-1)
        first_flip_rank = flips.to(torch.int64).argmax(-1)
        rank = torch.where(has_flip, first_flip_rank, torch.ones_like(first_flip_rank))
        alternate_indices = candidates.gather(-1, rank.unsqueeze(-1)).squeeze(-1)
        current_scores = self.support_scores[table_ids, nearest_flat, current_indices]
        alternate_scores = self.support_scores[table_ids, nearest_flat, alternate_indices]
        return (
            alternate_indices.reshape(*prefix, tables),
            (alternate_scores - current_scores).reshape(*prefix, tables),
            has_flip.reshape(*prefix, tables),
        )

    def _codes_with_support_swap(
        self,
        x: Tensor,
        supports: Tensor,
        swapped_level: Tensor,
        alternate_supports: Tensor,
    ) -> Tensor:
        flat, prefix = _flatten_input(x)
        items, tables = flat.shape[0], self.tables
        regular_supports = _normalize_supports(supports, "unary")[..., 0].to(device=x.device)
        swapped = swapped_level.reshape(items, tables)
        alternate = alternate_supports.reshape(items, tables)
        code = torch.zeros((items, tables), dtype=torch.int64, device=x.device)
        table_ids = torch.arange(tables, device=x.device)[None, :]
        thresholds = self.thresholds.to(device=x.device, dtype=x.dtype)
        for level in range(self.depth):
            regular = regular_supports[:, level][None, :].expand(items, -1)
            coordinate = torch.where(swapped == level, alternate, regular)
            node = (2**level - 1) + code
            value = flat.gather(1, coordinate)
            branch = _take_positive(value - thresholds[table_ids, node], self.spec.tie_break)
            code = 2 * code + branch.to(torch.int64)
        return code.reshape(*prefix, tables)

    def hard_output_with_support_counterfactual(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        supports = self.selected_supports()
        output, codes = self.hard_output(x, supports=supports)
        route = self.route(x, supports=supports)
        nearest = route.margins.abs().argmin(-1)
        alternate, _score_gap, has_flip = self._alternate_support(x, supports, route, nearest)
        alternate_codes = self._codes_with_support_swap(x, supports, nearest, alternate)
        return output, codes, (alternate_codes != codes) & has_flip

    def _support_counterfactual_correction(self, x: Tensor) -> Tensor:
        if self.support_scores is None or not self.support_learning:
            return torch.zeros(*x.shape[:-1], self.output_dim, dtype=x.dtype, device=x.device)
        supports = self.selected_supports()
        route = self.route(x, supports=supports)
        nearest = route.margins.abs().argmin(-1)
        alternate, score_gap, has_flip = self._alternate_support(x, supports, route, nearest)
        alternate_codes = self._codes_with_support_swap(x, supports, nearest, alternate)
        current = self.selected_actions(x, route.codes)
        alternative = self.selected_actions(x, alternate_codes)
        productive = has_flip & (alternate_codes != route.codes)
        probability = torch.sigmoid(score_gap / self.support_tau)
        gate = (probability - probability.detach()) * productive.to(probability.dtype)
        return (gate.unsqueeze(-1) * (alternative - current).detach()).sum(dim=-2)

    def forward(self, x: Tensor, *, supports: Tensor | None = None) -> Tensor:
        if self.spec.surrogate == "soft_product":
            output = self.soft_product_output(x, supports=supports)
        elif self.spec.surrogate == "local_counterfactual":
            output = self.local_counterfactual_output(x, supports=supports)
        else:
            output = self.hard_output(x, supports=supports)[0]
        if supports is None and self.support_scores is not None:
            output = output + self._support_counterfactual_correction(x)
        return output


class ProductGridLookupRouter(nn.Module):
    """Parallel scalar-bin product address followed by additive vector rows.

    Each table reads ``axes`` coordinates.  Every coordinate is compared with
    ``bins - 1`` thresholds in parallel, and the number of passed thresholds
    is one mixed-radix digit.  This gives ``bins**axes`` rows without treating
    the thermometer predicates as independent address bits.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        supports: Tensor,
        thresholds: Tensor,
        rows: Tensor,
        bins: int = 4,
        tie_break: TieBreak = "nonnegative",
        surrogate: Literal["none", "local_counterfactual"] = "local_counterfactual",
        tau: float = 1.0,
        trainable_thresholds: bool = True,
        trainable_rows: bool = True,
    ) -> None:
        super().__init__()
        if input_dim < 1 or output_dim < 1 or bins < 2:
            raise ValueError("input_dim, output_dim, and bins must be valid")
        if supports.ndim != 2 or supports.shape[1] < 1:
            raise ValueError("supports must be [tables,axes]")
        supports = supports.detach().clone().to(torch.int64)
        if int(supports.min()) < 0 or int(supports.max()) >= input_dim:
            raise ValueError("support coordinate out of range")
        tables, axes = supports.shape
        if thresholds.shape != (tables, axes, bins - 1):
            raise ValueError(f"thresholds must be [{tables},{axes},{bins - 1}]")
        if rows.shape != (tables, bins**axes, output_dim):
            raise ValueError(f"rows must be [{tables},{bins**axes},{output_dim}]")
        if tie_break not in {"positive", "nonnegative"}:
            raise ValueError("unsupported tie break")
        if surrogate not in {"none", "local_counterfactual"}:
            raise ValueError("unsupported surrogate")
        if tau <= 0:
            raise ValueError("tau must be positive")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.bins = int(bins)
        self.tie_break = tie_break
        self.surrogate = surrogate
        self.tau = float(tau)
        self.register_buffer("supports", supports)
        self.register_buffer("initial_thresholds", thresholds.detach().clone())
        self._register_state("thresholds", thresholds.detach().clone(), trainable_thresholds)
        self._register_state("rows", rows.detach().clone(), trainable_rows)

    def _register_state(self, name: str, value: Tensor, trainable: bool) -> None:
        if trainable:
            setattr(self, name, nn.Parameter(value))
        else:
            self.register_buffer(name, value)

    @property
    def tables(self) -> int:
        return int(self.supports.shape[0])

    @property
    def axes(self) -> int:
        return int(self.supports.shape[1])

    @property
    def comparisons(self) -> int:
        return self.axes * (self.bins - 1)

    @property
    def row_count(self) -> int:
        return self.bins**self.axes

    def route(self, x: Tensor) -> ProductGridRoute:
        flat, prefix = _flatten_input(x)
        if flat.shape[-1] != self.input_dim:
            raise ValueError(f"expected input_dim={self.input_dim}, got {flat.shape[-1]}")
        values = flat[:, self.supports.to(device=x.device)]
        margins = values.unsqueeze(-1) - self.thresholds.to(device=x.device, dtype=x.dtype).unsqueeze(0)
        branches = _take_positive(margins, self.tie_break)
        digits = branches.to(torch.int64).sum(dim=-1)
        powers = self.bins ** torch.arange(self.axes - 1, -1, -1, device=x.device)
        codes = (digits * powers.view(1, 1, -1)).sum(dim=-1)
        return ProductGridRoute(
            codes.reshape(*prefix, self.tables),
            digits.reshape(*prefix, self.tables, self.axes),
            margins.reshape(*prefix, self.tables, self.axes, self.bins - 1),
            branches.reshape(*prefix, self.tables, self.axes, self.bins - 1),
        )

    def hard_codes(self, x: Tensor) -> Tensor:
        return self.route(x).codes

    def selected_actions(self, codes: Tensor) -> Tensor:
        return gather_lookup_rows(self.rows, codes)

    def hard_output(self, x: Tensor) -> tuple[Tensor, Tensor]:
        route = self.route(x)
        return self.selected_actions(route.codes).sum(dim=-2), route.codes

    def neighboring_codes(self, route: ProductGridRoute, nearest: Tensor) -> Tensor:
        flat_branches = route.branches.flatten(start_dim=-2)
        chosen_branch = flat_branches.gather(-1, nearest.unsqueeze(-1)).squeeze(-1)
        axis = torch.div(nearest, self.bins - 1, rounding_mode="floor")
        powers = self.bins ** (self.axes - 1 - axis)
        delta = torch.where(chosen_branch, -powers, powers)
        return route.codes + delta

    def local_counterfactual_output(self, x: Tensor) -> Tensor:
        route = self.route(x)
        current = self.selected_actions(route.codes)
        hard = current.sum(dim=-2)
        flat_margins = route.margins.flatten(start_dim=-2)
        flat_branches = route.branches.flatten(start_dim=-2)
        nearest = flat_margins.abs().argmin(dim=-1)
        neighbor_codes = self.neighboring_codes(route, nearest)
        neighbor = self.selected_actions(neighbor_codes)
        chosen_margin = flat_margins.gather(-1, nearest.unsqueeze(-1)).squeeze(-1)
        chosen_branch = flat_branches.gather(-1, nearest.unsqueeze(-1)).squeeze(-1)
        direction = (2 * chosen_branch.to(x.dtype) - 1).unsqueeze(-1) * (current - neighbor)
        gate = torch.sigmoid(chosen_margin / self.tau)
        correction = ((gate - gate.detach()).unsqueeze(-1) * direction.detach()).sum(dim=-2)
        return hard + correction

    def forward(self, x: Tensor) -> Tensor:
        if self.surrogate == "local_counterfactual":
            return self.local_counterfactual_output(x)
        return self.hard_output(x)[0]
