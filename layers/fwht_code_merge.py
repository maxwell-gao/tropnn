from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from .accumulation import WalshButterfly
from .hard_lookup import sum_lookup_rows, weighted_neighbor_delta

__all__ = [
    "CodeMergeRoute",
    "DirectCodeMergeLUT",
    "DirectPairCodeEncoder",
    "FWHTCodeMergeLUT",
    "FWHTFlatPairLUT",
    "FWHTPairCodeEncoder",
    "PairCodeMergeLUT",
    "PairCodeRoute",
    "make_disjoint_pair_supports",
]


def _pack_lsb(bits: Tensor) -> Tensor:
    powers = 2 ** torch.arange(bits.shape[-1], device=bits.device, dtype=torch.int64)
    return (bits.to(torch.int64) * powers).sum(dim=-1)


def _zero_forward_sigmoid(margins: Tensor, tau: float) -> Tensor:
    soft = torch.sigmoid(margins / tau)
    return soft - soft.detach()


def make_disjoint_pair_supports(input_dim: int, tables: int, comparisons: int, *, seed: int) -> Tensor:
    """Sample fixed pair predicates without reusing a coordinate.

    This is deliberately a support generator, not a learned projection.  It is
    useful when the experiment should attribute all dense mixing to the fixed
    Walsh butterfly rather than to pair selection.
    """

    input_dim = int(input_dim)
    tables = int(tables)
    comparisons = int(comparisons)
    required = 2 * tables * comparisons
    if min(input_dim, tables, comparisons) < 1 or required > input_dim:
        raise ValueError("disjoint supports require 2 * tables * comparisons <= input_dim")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    selected = torch.randperm(input_dim, generator=generator)[:required]
    return selected.reshape(comparisons, tables, 2).permute(1, 0, 2).contiguous()


def _initial_merger_maps(mergers: int, rows: int, *, seed: int, initialization: str) -> Tensor:
    input_rows = rows * rows
    if initialization == "xor":
        inputs = torch.arange(input_rows, dtype=torch.int64)
        mapping = (inputs // rows) ^ (inputs % rows)
        return mapping.unsqueeze(0).expand(mergers, -1).clone()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    balanced_labels = torch.arange(input_rows, dtype=torch.int64) % rows
    maps = torch.empty(mergers, input_rows, dtype=torch.int64)
    for merger in range(mergers):
        permutation = torch.randperm(input_rows, generator=generator)
        maps[merger, permutation] = balanced_labels
    return maps


@dataclass(frozen=True)
class PairCodeRoute:
    codes: Tensor
    margins: Tensor


@dataclass(frozen=True)
class CodeMergeRoute:
    leaf_codes: Tensor
    leaf_margins: Tensor
    merger_input_codes: Tensor
    merger_logits: Tensor
    merged_codes: Tensor


class DirectPairCodeEncoder(nn.Module):
    """Trainable-threshold pair codes in the original input coordinates."""

    def __init__(self, input_dim: int, supports: Tensor) -> None:
        super().__init__()
        input_dim = int(input_dim)
        if input_dim < 1 or supports.ndim != 3 or supports.shape[-1] != 2:
            raise ValueError("supports must be [tables,comparisons,2]")
        supports = supports.to(dtype=torch.int64)
        if supports.numel() and (int(supports.min()) < 0 or int(supports.max()) >= input_dim):
            raise ValueError("support coordinate outside input width")
        self.input_dim = input_dim
        self.tables = int(supports.shape[0])
        self.comparisons = int(supports.shape[1])
        self.register_buffer("supports", supports.contiguous(), persistent=True)
        self.thresholds = nn.Parameter(torch.zeros(self.tables, self.comparisons))

    @property
    def rows(self) -> int:
        return 1 << self.comparisons

    def route(self, x: Tensor) -> PairCodeRoute:
        if x.ndim < 2 or x.shape[-1] != self.input_dim:
            raise ValueError(f"expected input [...,{self.input_dim}], got {tuple(x.shape)}")
        supports = self.supports.to(device=x.device)
        values = x[..., supports[..., 0]] - x[..., supports[..., 1]]
        margins = values - self.thresholds.to(device=x.device, dtype=values.dtype)
        return PairCodeRoute(_pack_lsb(margins > 0), margins)

    def forward(self, x: Tensor) -> Tensor:
        return self.route(x).codes


class FWHTPairCodeEncoder(nn.Module):
    """Fixed randomized FWHT followed by trainable-threshold pair codes."""

    def __init__(
        self,
        input_dim: int,
        transform_dim: int,
        supports: Tensor,
        *,
        seed: int,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        input_dim = int(input_dim)
        transform_dim = int(transform_dim)
        if input_dim < 1 or transform_dim < input_dim or transform_dim & (transform_dim - 1):
            raise ValueError("transform_dim must be a power of two no smaller than input_dim")
        if supports.ndim != 3 or supports.shape[-1] != 2:
            raise ValueError("supports must be [tables,comparisons,2]")
        supports = supports.to(dtype=torch.int64)
        if supports.numel() and (int(supports.min()) < 0 or int(supports.max()) >= transform_dim):
            raise ValueError("support coordinate outside transform width")
        self.input_dim = input_dim
        self.transform_dim = transform_dim
        self.tables = int(supports.shape[0])
        self.comparisons = int(supports.shape[1])
        self.normalize = bool(normalize)
        self.transform = WalshButterfly(transform_dim, seed=int(seed))
        self.register_buffer("supports", supports.contiguous(), persistent=True)
        self.thresholds = nn.Parameter(torch.zeros(self.tables, self.comparisons))

    @property
    def rows(self) -> int:
        return 1 << self.comparisons

    def mixed_coordinates(self, x: Tensor) -> Tensor:
        if x.ndim < 2 or x.shape[-1] != self.input_dim:
            raise ValueError(f"expected input [...,{self.input_dim}], got {tuple(x.shape)}")
        if self.transform_dim == self.input_dim:
            padded = x
        else:
            padded = torch.nn.functional.pad(x, (0, self.transform_dim - self.input_dim))
        mixed = self.transform(padded)
        return mixed / math.sqrt(self.transform_dim) if self.normalize else mixed

    def route(self, x: Tensor) -> PairCodeRoute:
        mixed = self.mixed_coordinates(x)
        supports = self.supports.to(device=x.device)
        values = mixed[..., supports[..., 0]] - mixed[..., supports[..., 1]]
        margins = values - self.thresholds.to(device=x.device, dtype=values.dtype)
        return PairCodeRoute(_pack_lsb(margins > 0), margins)

    def forward(self, x: Tensor) -> Tensor:
        return self.route(x).codes


class FWHTFlatPairLUT(nn.Module):
    """FWHT pair recognizer with the classic independent additive LUT action."""

    def __init__(
        self,
        input_dim: int,
        transform_dim: int,
        output_dim: int,
        supports: Tensor,
        *,
        seed: int,
        row_init_std: float,
        normalize: bool = True,
        tau: float = 1.0,
    ) -> None:
        super().__init__()
        self.encoder = FWHTPairCodeEncoder(input_dim, transform_dim, supports, seed=seed, normalize=normalize)
        self.output_dim = int(output_dim)
        if tau <= 0:
            raise ValueError("tau must be positive")
        self.tau = float(tau)
        generator = torch.Generator(device="cpu").manual_seed(int(seed) + 1)
        self.action_rows = nn.Parameter(
            torch.randn(self.encoder.tables, self.encoder.rows, self.output_dim, generator=generator) * float(row_init_std)
        )
        self.bias = nn.Parameter(torch.zeros(self.output_dim))
        self.register_buffer("powers", 2 ** torch.arange(self.encoder.comparisons, dtype=torch.int64), persistent=False)

    @property
    def thresholds(self) -> nn.Parameter:
        return self.encoder.thresholds

    def hard_codes(self, x: Tensor) -> Tensor:
        return self.encoder.route(x).codes

    def hard_output(self, x: Tensor) -> tuple[Tensor, Tensor]:
        route = self.encoder.route(x)
        return sum_lookup_rows(self.action_rows, route.codes, accumulation_dtype=torch.float32) + self.bias, route.codes

    def forward(self, x: Tensor) -> Tensor:
        route = self.encoder.route(x)
        hard = sum_lookup_rows(self.action_rows, route.codes, accumulation_dtype=torch.float32) + self.bias
        if not self.training:
            return hard
        neighbors = route.codes.unsqueeze(-1) ^ self.powers.to(device=x.device)
        weights = _zero_forward_sigmoid(route.margins, self.tau)
        return hard + weighted_neighbor_delta(self.action_rows, route.codes, neighbors, weights).to(hard.dtype)


class PairCodeMergeLUT(nn.Module):
    """Pair-code encoder, pairwise code mergers, and additive action LUTs.

    Two adjacent C-bit leaf codes address one ``2**(2C)`` merger row.  That row
    emits a C-bit code, which addresses one output-vector action row.  During
    training the forward path remains exactly hard.  Zero-valued local
    counterfactual corrections carry credit to selected merger logits and leaf
    thresholds.  Deployment compiles every merger into an integer map and
    performs no soft evaluation.
    """

    def __init__(
        self,
        encoder: DirectPairCodeEncoder | FWHTPairCodeEncoder,
        output_dim: int,
        *,
        seed: int,
        row_init_std: float,
        merger_init_logit: float = 0.005,
        merger_initialization: str = "balanced_random",
        tau: float = 1.0,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        if self.encoder.tables % 2:
            raise ValueError("code merging requires an even number of leaf tables")
        if merger_init_logit <= 0:
            raise ValueError("merger_init_logit must be positive")
        self.output_dim = int(output_dim)
        self.mergers = self.encoder.tables // 2
        if tau <= 0:
            raise ValueError("tau must be positive")
        self.tau = float(tau)
        rows = self.encoder.rows
        if merger_initialization not in {"balanced_random", "xor"}:
            raise ValueError("merger_initialization must be 'balanced_random' or 'xor'")
        self.merger_initialization = merger_initialization
        initial_maps = _initial_merger_maps(
            self.mergers,
            rows,
            seed=int(seed) + 2,
            initialization=merger_initialization,
        )
        initial_bits = (initial_maps.unsqueeze(-1) >> torch.arange(self.encoder.comparisons)) & 1
        initial_logits = (2 * initial_bits.to(torch.float32) - 1) * float(merger_init_logit)
        self.register_buffer("initial_merger_map", initial_maps, persistent=True)
        self.merger_logits = nn.Parameter(initial_logits)
        generator = torch.Generator(device="cpu").manual_seed(int(seed) + 1)
        self.action_rows = nn.Parameter(torch.randn(self.mergers, rows, self.output_dim, generator=generator) * float(row_init_std))
        self.bias = nn.Parameter(torch.zeros(self.output_dim))
        self.register_buffer("powers", 2 ** torch.arange(self.encoder.comparisons, dtype=torch.int64), persistent=False)

    @property
    def thresholds(self) -> nn.Parameter:
        return self.encoder.thresholds

    @property
    def rows(self) -> int:
        return self.encoder.rows

    def compiled_merger_map(self) -> Tensor:
        return _pack_lsb(self.merger_logits > 0)

    def _merger_input_codes(self, leaf_codes: Tensor) -> Tensor:
        paired = leaf_codes.reshape(*leaf_codes.shape[:-1], self.mergers, 2)
        return paired[..., 0] * self.rows + paired[..., 1]

    def _selected_merger_logits(self, merger_input_codes: Tensor) -> Tensor:
        flat = merger_input_codes.reshape(-1, self.mergers)
        table_ids = torch.arange(self.mergers, device=flat.device)[None, :]
        selected = self.merger_logits.to(device=flat.device)[table_ids, flat]
        return selected.reshape(*merger_input_codes.shape, self.encoder.comparisons)

    def _compiled_codes(self, merger_input_codes: Tensor, compiled_map: Tensor | None = None) -> Tensor:
        mapping = self.compiled_merger_map() if compiled_map is None else compiled_map
        flat = merger_input_codes.reshape(-1, self.mergers)
        table_ids = torch.arange(self.mergers, device=flat.device)[None, :]
        selected = mapping.to(device=flat.device)[table_ids, flat]
        return selected.reshape(*merger_input_codes.shape)

    def route(self, x: Tensor) -> CodeMergeRoute:
        leaf = self.encoder.route(x)
        merger_input = self._merger_input_codes(leaf.codes)
        logits = self._selected_merger_logits(merger_input)
        return CodeMergeRoute(leaf.codes, leaf.margins, merger_input, logits, _pack_lsb(logits > 0))

    def hard_codes(self, x: Tensor) -> Tensor:
        leaf = self.encoder.route(x)
        return self._compiled_codes(self._merger_input_codes(leaf.codes))

    def hard_output(self, x: Tensor) -> tuple[Tensor, Tensor]:
        codes = self.hard_codes(x)
        return sum_lookup_rows(self.action_rows, codes, accumulation_dtype=torch.float32) + self.bias, codes

    def _neighbor_merger_codes(self, leaf_codes: Tensor) -> Tensor:
        prefix = leaf_codes.shape[:-1]
        paired = leaf_codes.reshape(*prefix, self.mergers, 2)
        powers = self.powers.to(device=leaf_codes.device)
        left_neighbors = ((paired[..., 0].unsqueeze(-1) ^ powers) * self.rows) + paired[..., 1].unsqueeze(-1)
        right_neighbors = (paired[..., 0].unsqueeze(-1) * self.rows) + (paired[..., 1].unsqueeze(-1) ^ powers)
        neighbor_inputs = torch.cat((left_neighbors, right_neighbors), dim=-1)
        flat = neighbor_inputs.reshape(-1, self.mergers, 2 * self.encoder.comparisons)
        logits = self.merger_logits.to(device=leaf_codes.device).unsqueeze(0).expand(flat.shape[0], -1, -1, -1)
        gather_index = flat.unsqueeze(-1).expand(-1, -1, -1, self.encoder.comparisons)
        neighbor_logits = logits.gather(2, gather_index)
        return _pack_lsb(neighbor_logits > 0).reshape(*prefix, self.mergers, 2 * self.encoder.comparisons)

    def forward(self, x: Tensor) -> Tensor:
        route = self.route(x)
        hard = sum_lookup_rows(self.action_rows, route.merged_codes, accumulation_dtype=torch.float32) + self.bias
        if not self.training:
            return hard

        output_neighbors = route.merged_codes.unsqueeze(-1) ^ self.powers.to(device=x.device)
        output_weights = _zero_forward_sigmoid(route.merger_logits, self.tau)
        merger_correction = weighted_neighbor_delta(
            self.action_rows,
            route.merged_codes,
            output_neighbors,
            output_weights,
        )

        leaf_neighbors = self._neighbor_merger_codes(route.leaf_codes)
        leaf_margins = route.leaf_margins.reshape(*route.leaf_margins.shape[:-2], self.mergers, 2 * self.encoder.comparisons)
        leaf_weights = _zero_forward_sigmoid(leaf_margins, self.tau)
        leaf_correction = weighted_neighbor_delta(
            self.action_rows,
            route.merged_codes,
            leaf_neighbors,
            leaf_weights,
        )
        return hard + merger_correction.to(hard.dtype) + leaf_correction.to(hard.dtype)


class DirectCodeMergeLUT(PairCodeMergeLUT):
    """Original-coordinate pair codes followed by learned code mergers."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        supports: Tensor,
        *,
        seed: int,
        row_init_std: float,
        merger_init_logit: float = 0.005,
        merger_initialization: str = "balanced_random",
        tau: float = 1.0,
    ) -> None:
        super().__init__(
            DirectPairCodeEncoder(input_dim, supports),
            output_dim,
            seed=seed,
            row_init_std=row_init_std,
            merger_init_logit=merger_init_logit,
            merger_initialization=merger_initialization,
            tau=tau,
        )


class FWHTCodeMergeLUT(PairCodeMergeLUT):
    """Fixed randomized FWHT pair codes followed by learned code mergers."""

    def __init__(
        self,
        input_dim: int,
        transform_dim: int,
        output_dim: int,
        supports: Tensor,
        *,
        seed: int,
        row_init_std: float,
        merger_init_logit: float = 0.005,
        merger_initialization: str = "balanced_random",
        normalize: bool = True,
        tau: float = 1.0,
    ) -> None:
        super().__init__(
            FWHTPairCodeEncoder(input_dim, transform_dim, supports, seed=seed, normalize=normalize),
            output_dim,
            seed=seed,
            row_init_std=row_init_std,
            merger_init_logit=merger_init_logit,
            merger_initialization=merger_initialization,
            tau=tau,
        )
