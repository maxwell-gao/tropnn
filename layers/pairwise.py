from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..backend import Backend
from .base import LUTLayerSpec, LUTModuleBase, finish_lut_output
from .surrogate import ste_heaviside, surrogate_gradient

PAIRWISE_ANCHOR_POLICIES = ("random", "random_no_replace", "local", "cyclic", "block", "expander", "permuted")

__all__ = ["AbsDiffLUT", "AbsDiffSpec", "PAIRWISE_ANCHOR_POLICIES", "PairwiseLUT", "PairwiseRoute", "PairwiseSpec", "PairwiseWalshLUT"]


@dataclass(frozen=True)
class PairwiseSpec:
    input_dim: int
    output_dim: int
    tables: int
    comparisons: int
    backend: Backend
    use_min_margin_ste: bool
    surrogate: str
    cpu_lut_dtype: Literal["f32", "f16"]
    anchor_policy: str
    anchor_seed: int
    use_output_scaling: bool

    def __post_init__(self) -> None:
        if self.tables < 1 or self.comparisons < 1:
            raise ValueError("tables and comparisons must be positive")
        if self.backend not in {"torch", "tilelang", "zig"}:
            raise ValueError(f"unsupported backend {self.backend!r}")
        if self.cpu_lut_dtype not in {"f32", "f16"}:
            raise ValueError("cpu_lut_dtype must be 'f32' or 'f16'")
        if self.anchor_policy not in PAIRWISE_ANCHOR_POLICIES:
            raise ValueError(f"unsupported anchor policy {self.anchor_policy!r}")

    @property
    def table_size(self) -> int:
        return 1 << self.comparisons

    @property
    def output_scale(self) -> float:
        return 1.0 / math.sqrt(self.tables) if self.use_output_scaling else 1.0


@dataclass(frozen=True)
class AbsDiffSpec:
    features: int
    output_dim: int
    tables: int
    comparisons: int
    use_min_margin_ste: bool
    surrogate: str
    use_output_scaling: bool

    def __post_init__(self) -> None:
        if self.features < 1 or self.output_dim < 1 or self.tables < 1 or self.comparisons < 1:
            raise ValueError("features, output_dim, tables, and comparisons must be positive")

    @property
    def table_size(self) -> int:
        return 1 << self.comparisons

    @property
    def output_scale(self) -> float:
        return 1.0 / math.sqrt(self.tables) if self.use_output_scaling else 1.0


@dataclass(frozen=True)
class PairwiseRoute:
    indices: Tensor
    margins: Tensor

    def detach(self) -> "PairwiseRoute":
        return PairwiseRoute(self.indices.detach(), self.margins.detach())


@dataclass
class _ZigLutCache:
    dtype: Literal["f32", "f16"]
    tensor: Tensor | None = None
    version: int = -1

    def materialize(self, lut: Tensor) -> Tensor:
        if self.dtype == "f32":
            return lut.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if self.tensor is None or self.version != lut._version or self.tensor.shape != lut.shape:
            self.tensor = lut.detach().to(device="cpu", dtype=torch.float16).contiguous()
            self.version = lut._version
        return self.tensor


class PairwiseLUT(LUTModuleBase):
    """Pairwise comparison LUT: anchors compare coordinates, thresholds define boundaries, lut stores payload rows."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        tables: int = 16,
        comparisons: int = 4,
        backend: Backend = "torch",
        seed: int = 0,
        lut_init_std: float = 0.0,
        use_min_margin_ste: bool = True,
        use_output_scaling: bool = True,
        fixed_zero_threshold: bool = False,
        surrogate: str = "fast_sigmoid_odd",
        cpu_lut_dtype: Literal["f32", "f16"] = "f32",
        anchor_policy: str = "random",
        anchor_seed: int | None = None,
    ) -> None:
        spec = PairwiseSpec(
            int(input_dim),
            int(output_dim),
            int(tables),
            int(comparisons),
            backend,
            bool(use_min_margin_ste),
            surrogate,
            cpu_lut_dtype,
            anchor_policy,
            seed if anchor_seed is None else int(anchor_seed),
            bool(use_output_scaling),
        )
        surrogate_gradient(torch.zeros((), dtype=torch.float32), spec.surrogate)
        super().__init__(LUTLayerSpec.build(spec.input_dim, spec.output_dim, backend=spec.backend, output_scale=spec.output_scale))
        self.spec = spec
        self._zig_cache = _ZigLutCache(spec.cpu_lut_dtype)
        self.register_buffer("anchors", _make_pairwise_anchors(spec.input_dim, spec.tables, spec.comparisons, policy=spec.anchor_policy, seed=spec.anchor_seed))
        self.register_buffer("powers", 2 ** torch.arange(spec.comparisons, dtype=torch.long))
        thresholds = torch.zeros(spec.tables, spec.comparisons)
        self.register_buffer("thresholds", thresholds) if fixed_zero_threshold else setattr(self, "thresholds", nn.Parameter(thresholds))
        self.lut = nn.Parameter(_init_lut(spec, seed=seed, init_std=lut_init_std))

    @property
    def tables(self) -> int: return self.spec.tables
    @property
    def comparisons(self) -> int: return self.spec.comparisons
    @property
    def table_size(self) -> int: return self.spec.table_size
    @property
    def use_min_margin_ste(self) -> bool: return self.spec.use_min_margin_ste
    @property
    def surrogate(self) -> str: return self.spec.surrogate
    @property
    def cpu_lut_dtype(self) -> Literal["f32", "f16"]: return self.spec.cpu_lut_dtype
    @property
    def anchor_policy(self) -> str: return self.spec.anchor_policy

    def extra_repr(self) -> str:
        return f"input_dim={self.input_dim}, output_dim={self.output_dim}, tables={self.tables}, comparisons={self.comparisons}, backend={self.backend!r}, anchor_policy={self.anchor_policy!r}"

    def forward(self, x: Tensor) -> Tensor:
        input_dtype = x.dtype
        self._check_input_shape(x)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        output, route = self.compute(x, compute_dtype=_compute_dtype_for_lut(x), training=self.training)
        return self._finish_output(output, route, input_dtype)

    def compute(self, x: Tensor, *, compute_dtype: torch.dtype, training: bool) -> tuple[Tensor, PairwiseRoute]:
        """Route rows, read LUT payloads, and attach the finite-difference STE path."""

        x = x.to(torch.float32 if self.backend in {"tilelang", "zig"} else compute_dtype)
        if self.backend == "tilelang":
            return self._tilelang_compute(x, compute_dtype=compute_dtype)

        route = self.cache_index(x)
        if self.backend == "zig":
            return self._zig_compute(x, route, compute_dtype=compute_dtype, training=training)

        lut = self.lut_payload(dtype=compute_dtype, device=route.indices.device)
        output = self.lut_forward(route, lut, compute_dtype=compute_dtype)
        if training and (x.requires_grad or bool(getattr(self.thresholds, "requires_grad", False))):
            output = output + self.lut_backward_surrogate(route, lut).to(output.dtype)
        return output, route

    def cache_index(self, x: Tensor) -> PairwiseRoute:
        return _cache_pairwise_index(x, self.anchors, self.thresholds, self.powers)

    def lut_payload(self, *, dtype: torch.dtype, device: torch.device) -> Tensor:
        return self.lut.to(dtype=dtype, device=device)

    def lut_forward(self, route: PairwiseRoute, lut: Tensor, *, compute_dtype: torch.dtype) -> Tensor:
        return _sum_lut_rows(route.indices, lut, table_size=self.table_size, output_dim=self.output_dim, compute_dtype=compute_dtype)

    def lut_backward_surrogate(self, route: PairwiseRoute, lut: Tensor) -> Tensor:
        if self.use_min_margin_ste:
            bit = route.margins.abs().argmin(dim=-1)
            margin = route.margins.gather(dim=-1, index=bit.unsqueeze(-1)).squeeze(-1)
            neighbor = route.indices ^ (2**bit).long()
            ste = ste_heaviside(margin, self.surrogate) - (margin > 0).to(margin.dtype)
            return _single_bit_ste_delta(route.indices, neighbor, ste, lut, table_size=self.table_size, output_dim=self.output_dim)
        return _all_bits_ste_delta(route, self.powers, lut, table_size=self.table_size, output_dim=self.output_dim, surrogate=self.surrogate)

    def route(self, x: Tensor) -> PairwiseRoute: return self.cache_index(x)
    def payload_table(self, *, dtype: torch.dtype, device: torch.device) -> Tensor: return self.lut_payload(dtype=dtype, device=device)
    def lookup(self, route: PairwiseRoute, lut: Tensor, *, compute_dtype: torch.dtype) -> Tensor: return self.lut_forward(route, lut, compute_dtype=compute_dtype)
    def ste_correction(self, route: PairwiseRoute, lut: Tensor) -> Tensor: return self.lut_backward_surrogate(route, lut)

    def _tilelang_compute(self, x: Tensor, *, compute_dtype: torch.dtype) -> tuple[Tensor, PairwiseRoute]:
        from ..backends import pairwise_tilelang

        output, indices, margins = pairwise_tilelang(
            x.to(torch.float32),
            self.anchors.to(device=x.device),
            self.thresholds.to(dtype=torch.float32, device=x.device),
            self.lut.to(dtype=torch.float32, device=x.device),
            use_min_margin_ste=self.use_min_margin_ste,
            surrogate=self.surrogate,
        )
        return output.to(compute_dtype), PairwiseRoute(indices, margins)

    def _zig_compute(self, x: Tensor, route: PairwiseRoute, *, compute_dtype: torch.dtype, training: bool) -> tuple[Tensor, PairwiseRoute]:
        if training:
            raise RuntimeError("PairwiseLUT backend='zig' is inference-only; call .eval() or use backend='torch' for training")
        from ..backends import pairwise_zig_forward

        output = pairwise_zig_forward(x, self.anchors, self.thresholds.detach().to(dtype=torch.float32, device="cpu"), self._zig_cache.materialize(self.lut), lut_dtype=self.cpu_lut_dtype)
        return output.to(device=x.device, dtype=compute_dtype), route


class PairwiseWalshLUT(PairwiseLUT):
    """PairwiseLUT with a structured Walsh payload table."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        tables: int = 16,
        comparisons: int = 4,
        walsh_order: Literal[1, 2] = 2,
        backend: Backend = "torch",
        seed: int = 0,
        coeff_init_std: float = 0.02,
        use_min_margin_ste: bool = True,
        use_output_scaling: bool = True,
        surrogate: str = "fast_sigmoid_odd",
        fixed_zero_threshold: bool = False,
        anchor_policy: str = "random",
        anchor_seed: int | None = None,
    ) -> None:
        if walsh_order not in {1, 2}:
            raise ValueError("walsh_order must be 1 or 2")
        if backend != "torch":
            raise ValueError("PairwiseWalshLUT only supports backend='torch'")
        super().__init__(input_dim, output_dim, tables=tables, comparisons=comparisons, backend=backend, seed=seed, lut_init_std=0.0, use_min_margin_ste=use_min_margin_ste, use_output_scaling=use_output_scaling, fixed_zero_threshold=fixed_zero_threshold, surrogate=surrogate, anchor_policy=anchor_policy, anchor_seed=anchor_seed)
        del self.lut
        self.walsh_order = int(walsh_order)
        pair_indices = torch.combinations(torch.arange(comparisons, dtype=torch.long), r=2)
        if self.walsh_order == 1:
            pair_indices = pair_indices[:0]
        self.register_buffer("pair_indices", pair_indices)
        term_count = 1 + comparisons + int(pair_indices.shape[0])
        init_std = coeff_init_std / math.sqrt(term_count)
        gen = torch.Generator(device="cpu").manual_seed(seed + 1)
        self.constant = nn.Parameter(torch.randn(tables, output_dim, generator=gen) * init_std)
        self.linear_coeff = nn.Parameter(torch.randn(tables, comparisons, output_dim, generator=gen) * init_std)
        self.pair_coeff = nn.Parameter(torch.randn(tables, pair_indices.shape[0], output_dim, generator=gen) * init_std)
        bit_values = torch.arange(self.table_size, dtype=torch.long).unsqueeze(-1).bitwise_and(self.powers.view(1, -1))
        self.register_buffer("walsh_bits", torch.where(bit_values > 0, torch.ones_like(bit_values), -torch.ones_like(bit_values)).float())

    @property
    def walsh_term_count(self) -> int:
        return 1 + self.comparisons + int(self.pair_indices.shape[0])

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, walsh_order={self.walsh_order}"

    def lut_payload(self, *, dtype: torch.dtype, device: torch.device) -> Tensor:
        return self.materialize_lut(dtype=dtype, device=device)

    def materialize_lut(self, *, dtype: torch.dtype | None = None, device: torch.device | None = None) -> Tensor:
        dtype = dtype if dtype is not None else self.constant.dtype
        device = device if device is not None else self.constant.device
        bits = self.walsh_bits.to(dtype=dtype, device=device)
        output = self.constant.to(dtype=dtype, device=device).unsqueeze(1)
        output = output + (bits.view(1, self.table_size, self.comparisons, 1) * self.linear_coeff.to(dtype=dtype, device=device).view(self.tables, 1, self.comparisons, self.output_dim)).sum(dim=2)
        if self.pair_indices.numel() > 0:
            pairs = self.pair_indices.to(device=device)
            pair_bits = bits[:, pairs[:, 0]] * bits[:, pairs[:, 1]]
            output = output + (pair_bits.view(1, self.table_size, -1, 1) * self.pair_coeff.to(dtype=dtype, device=device).view(self.tables, 1, -1, self.output_dim)).sum(dim=2)
        return output


class AbsDiffLUT(nn.Module):
    """Two-input relation LUT based on coordinate closeness."""

    def __init__(
        self,
        features: int,
        output_dim: int,
        *,
        tables: int = 16,
        comparisons: int = 4,
        seed: int = 0,
        lut_init_std: float = 0.02,
        width_init: float = 0.2,
        use_min_margin_ste: bool = True,
        use_output_scaling: bool = True,
        surrogate: str = "fast_sigmoid_odd",
    ) -> None:
        super().__init__()
        if width_init <= 0:
            raise ValueError("width_init must be positive")
        self.spec = AbsDiffSpec(int(features), int(output_dim), int(tables), int(comparisons), bool(use_min_margin_ste), surrogate, bool(use_output_scaling))
        surrogate_gradient(torch.zeros((), dtype=torch.float32), self.surrogate)
        self._last_route: PairwiseRoute | None = None
        self._last_indices: Tensor | None = None
        self._last_margins: Tensor | None = None
        gen = torch.Generator(device="cpu").manual_seed(seed)
        self.register_buffer("coords", torch.randint(0, self.features, (self.tables, self.comparisons), generator=gen, dtype=torch.long))
        self.register_buffer("powers", 2 ** torch.arange(self.comparisons, dtype=torch.long))
        self.log_widths = nn.Parameter(torch.full((self.tables, self.comparisons), self._inverse_softplus(width_init)))
        self.lut = nn.Parameter(torch.randn(self.tables, self.table_size, self.output_dim, generator=gen) * lut_init_std)

    @property
    def features(self) -> int: return self.spec.features
    @property
    def output_dim(self) -> int: return self.spec.output_dim
    @property
    def tables(self) -> int: return self.spec.tables
    @property
    def comparisons(self) -> int: return self.spec.comparisons
    @property
    def table_size(self) -> int: return self.spec.table_size
    @property
    def use_min_margin_ste(self) -> bool: return self.spec.use_min_margin_ste
    @property
    def surrogate(self) -> str: return self.spec.surrogate
    @property
    def output_scale(self) -> float: return self.spec.output_scale
    @staticmethod
    def _inverse_softplus(value: float) -> float: return math.log(math.expm1(value))

    def route(self, query: Tensor, key: Tensor) -> PairwiseRoute:
        prefix = query.shape[:-1]
        coord_flat = self.coords.reshape(-1)
        q = query[..., coord_flat].view(*prefix, self.tables, self.comparisons)
        k = key[..., coord_flat].view(*prefix, self.tables, self.comparisons)
        widths = F.softplus(self.log_widths).to(dtype=query.dtype, device=query.device)
        margins = widths - (q - k).abs()
        powers = self.powers.to(device=query.device).view(*([1] * len(prefix)), 1, -1)
        return PairwiseRoute(((margins > 0).to(torch.long) * powers).sum(dim=-1), margins)

    def lookup(self, route: PairwiseRoute, *, compute_dtype: torch.dtype) -> Tensor:
        return _sum_lut_rows(route.indices, self.lut.to(dtype=compute_dtype, device=route.indices.device), table_size=self.table_size, output_dim=self.output_dim, compute_dtype=compute_dtype)

    def ste_correction(self, route: PairwiseRoute) -> Tensor:
        lut = self.lut.to(dtype=torch.float32, device=route.indices.device)
        if self.use_min_margin_ste:
            bit = route.margins.abs().argmin(dim=-1)
            margin = route.margins.gather(dim=-1, index=bit.unsqueeze(-1)).squeeze(-1)
            neighbor = route.indices ^ (2**bit).long()
            ste = ste_heaviside(margin, self.surrogate) - (margin > 0).to(margin.dtype)
            return _single_bit_ste_delta(route.indices, neighbor, ste, lut, table_size=self.table_size, output_dim=self.output_dim)
        return _all_bits_ste_delta(route, self.powers, lut, table_size=self.table_size, output_dim=self.output_dim, surrogate=self.surrogate)

    def forward(self, query: Tensor, key: Tensor) -> Tensor:
        output_dtype = query.dtype
        compute_dtype = torch.promote_types(query.dtype, key.dtype)
        compute_dtype = torch.float32 if compute_dtype in {torch.float16, torch.bfloat16} else compute_dtype
        route = self.route(query.to(compute_dtype), key.to(compute_dtype))
        output = self.lookup(route, compute_dtype=compute_dtype)
        if self.training and (query.requires_grad or key.requires_grad or self.log_widths.requires_grad):
            output = output + self.ste_correction(route).to(output.dtype)
        return finish_lut_output(self, output, route, output_dtype, self.output_scale)


def _init_lut(spec: PairwiseSpec, *, seed: int, init_std: float) -> Tensor:
    if init_std == 0.0:
        return torch.zeros(spec.tables, spec.table_size, spec.output_dim)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(spec.tables, spec.table_size, spec.output_dim, generator=gen) * init_std


def _compute_dtype_for_lut(x: Tensor) -> torch.dtype:
    return torch.float32 if x.dtype in {torch.float16, torch.bfloat16} else x.dtype


def _make_pairwise_anchors(input_dim: int, tables: int, comparisons: int, *, policy: str, seed: int | None) -> Tensor:
    if input_dim < 2:
        raise ValueError(f"Pairwise anchors require input_dim >= 2, got {input_dim}")
    if policy not in PAIRWISE_ANCHOR_POLICIES:
        raise ValueError(f"unsupported anchor policy {policy!r}")

    route_count = tables * comparisons
    flat = torch.arange(route_count, dtype=torch.long)
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(int(seed))

    if policy == "random":
        a = torch.randint(0, input_dim, (route_count,), generator=gen)
        b = torch.randint(0, input_dim - 1, (route_count,), generator=gen)
        b = b + (b >= a).long()
    elif policy == "random_no_replace":
        pair_count = input_dim * (input_dim - 1)
        if route_count <= pair_count and pair_count <= 2_000_000:
            raw = torch.randperm(pair_count, generator=gen)[:route_count]
            a = raw // (input_dim - 1)
            b = raw % (input_dim - 1)
            b = b + (b >= a).long()
        else:
            a = torch.randint(0, input_dim, (route_count,), generator=gen)
            b = torch.randint(0, input_dim - 1, (route_count,), generator=gen)
            b = b + (b >= a).long()
    elif policy == "local":
        a = flat.remainder(input_dim)
        b = (a + 1).remainder(input_dim)
    elif policy == "cyclic":
        a = flat.remainder(input_dim)
        stride = ((flat // input_dim) + (flat % comparisons) + 1).remainder(input_dim - 1) + 1
        b = (a + stride).remainder(input_dim)
    elif policy == "block":
        block = min(input_dim, max(2, int(math.sqrt(input_dim))))
        block_count = (input_dim + block - 1) // block
        block_id = (flat // block).remainder(block_count)
        within = flat.remainder(block)
        a = (block_id * block + within).remainder(input_dim)
        offset = ((flat // max(1, block_count)) % (block - 1)) + 1
        b = (block_id * block + within + offset).remainder(input_dim)
    elif policy == "expander":
        a = (flat * 37 + 17).remainder(input_dim)
        stride = ((flat * 53 + 19) % (input_dim - 1)) + 1
        b = (a + stride).remainder(input_dim)
    elif policy == "permuted":
        perm_a = torch.randperm(input_dim, generator=gen)
        perm_b = torch.randperm(input_dim, generator=gen)
        a = perm_a[flat.remainder(input_dim)]
        b = perm_b[(flat * 7 + flat // max(1, input_dim)).remainder(input_dim)]
        b = torch.where(a == b, (b + 1).remainder(input_dim), b)
    else:
        raise AssertionError(f"unreachable anchor policy {policy!r}")

    b = torch.where(a == b, (b + 1).remainder(input_dim), b)
    return torch.stack((a, b), dim=-1).view(tables, comparisons, 2)


def _cache_pairwise_index(x: Tensor, anchors: Tensor, thresholds: Tensor, powers: Tensor) -> PairwiseRoute:
    prefix = x.shape[:-1]
    tables, comparisons, _ = anchors.shape
    a = anchors[:, :, 0].flatten()
    b = anchors[:, :, 1].flatten()
    margins = x[..., a].view(*prefix, tables, comparisons) - x[..., b].view(*prefix, tables, comparisons)
    margins = margins - thresholds.to(dtype=x.dtype, device=x.device)
    powers = powers.to(device=x.device).view(*([1] * len(prefix)), 1, -1)
    return PairwiseRoute(((margins > 0).to(torch.long) * powers).sum(dim=-1), margins)


def _sum_lut_rows(indices: Tensor, lut: Tensor, *, table_size: int, output_dim: int, compute_dtype: torch.dtype) -> Tensor:
    prefix, routes = indices.shape[:-1], indices.shape[-1]
    items = max(1, indices.numel() // routes)
    flat_indices = indices.reshape(items, routes)
    flat_lut = lut.reshape(routes * table_size, output_dim)
    output = torch.zeros(items, output_dim, device=indices.device, dtype=compute_dtype)
    chunk = _route_chunk_size(item_count=items, payload_width=output_dim, compute_dtype=compute_dtype, route_count=routes)
    for start in range(0, routes, chunk):
        stop = min(start + chunk, routes)
        offsets = (torch.arange(start, stop, device=indices.device) * table_size).view(1, -1)
        rows = (flat_indices[:, start:stop] + offsets).reshape(-1)
        values = flat_lut.index_select(0, rows).view(items, stop - start, output_dim)
        output = output + values.sum(dim=1)
    return output.view(*prefix, output_dim)


def _single_bit_ste_delta(current_indices: Tensor, neighbor_indices: Tensor, ste_weight: Tensor, lut: Tensor, *, table_size: int, output_dim: int) -> Tensor:
    prefix, routes = current_indices.shape[:-1], current_indices.shape[-1]
    items = max(1, current_indices.numel() // routes)
    current = current_indices.reshape(items, routes)
    neighbor = neighbor_indices.reshape(items, routes)
    weight = ste_weight.reshape(items, routes, 1).float()
    flat_lut = lut.to(dtype=torch.float32, device=current_indices.device).reshape(routes * table_size, output_dim)
    output = torch.zeros(items, output_dim, device=current_indices.device, dtype=torch.float32)
    chunk = _route_chunk_size(item_count=items, payload_width=output_dim, compute_dtype=torch.float32, route_count=routes)
    for start in range(0, routes, chunk):
        stop = min(start + chunk, routes)
        offsets = (torch.arange(start, stop, device=current_indices.device) * table_size).view(1, -1)
        rows = (current[:, start:stop] + offsets).reshape(-1)
        neighbor_rows = (neighbor[:, start:stop] + offsets).reshape(-1)
        cur = flat_lut.index_select(0, rows).view(items, stop - start, output_dim)
        nbr = flat_lut.index_select(0, neighbor_rows).view(items, stop - start, output_dim)
        output = output + (weight[:, start:stop] * (nbr - cur)).sum(dim=1)
    return output.view(*prefix, output_dim)


def _all_bits_ste_delta(route: PairwiseRoute, powers: Tensor, lut: Tensor, *, table_size: int, output_dim: int, surrogate: str) -> Tensor:
    prefix, routes = route.indices.shape[:-1], route.indices.shape[-1]
    comparisons = route.margins.shape[-1]
    items = max(1, route.indices.numel() // routes)
    current = route.indices.reshape(items, routes)
    neighbor = current.unsqueeze(-1) ^ powers.to(device=route.indices.device).view(1, 1, -1)
    ste = ste_heaviside(route.margins, surrogate) - (route.margins > 0).to(route.margins.dtype)
    ste = ste.reshape(items, routes, comparisons, 1).float()
    flat_lut = lut.to(dtype=torch.float32, device=route.indices.device).reshape(routes * table_size, output_dim)
    output = torch.zeros(items, output_dim, device=route.indices.device, dtype=torch.float32)
    chunk = _route_chunk_size(item_count=items, payload_width=output_dim * (comparisons + 1), compute_dtype=torch.float32, route_count=routes, target_bytes=8 * 1024 * 1024)
    for start in range(0, routes, chunk):
        stop = min(start + chunk, routes)
        offsets = (torch.arange(start, stop, device=route.indices.device) * table_size).view(1, -1)
        rows = (current[:, start:stop] + offsets).reshape(-1)
        cur = flat_lut.index_select(0, rows).view(items, stop - start, 1, output_dim)
        neighbor_rows = (neighbor[:, start:stop] + offsets.unsqueeze(-1)).reshape(-1)
        nbr = flat_lut.index_select(0, neighbor_rows).view(items, stop - start, comparisons, output_dim)
        output = output + (ste[:, start:stop] * (nbr - cur)).sum(dim=(1, 2))
    return output.view(*prefix, output_dim)


def _route_chunk_size(*, item_count: int, payload_width: int, compute_dtype: torch.dtype, route_count: int, target_bytes: int = 16 * 1024 * 1024) -> int:
    bytes_per_route = item_count * payload_width * torch.finfo(compute_dtype).bits // 8
    return max(1, min(route_count, target_bytes // max(1, bytes_per_route)))
