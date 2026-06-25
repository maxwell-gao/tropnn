from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..backend import Backend
from .base import LUTModuleBase
from .surrogate import ste_heaviside, surrogate_gradient

PAIRWISE_ANCHOR_POLICIES = (
    "random",
    "random_no_replace",
    "local",
    "cyclic",
    "block",
    "expander",
    "permuted",
)


@dataclass(frozen=True)
class PairwiseRoute:
    """Discrete table address and its continuous comparison margins."""

    indices: Tensor
    margins: Tensor


def _route_chunk_size(
    *,
    item_count: int,
    payload_width: int,
    compute_dtype: torch.dtype,
    route_count: int,
    target_bytes: int = 16 * 1024 * 1024,
) -> int:
    bytes_per_route = item_count * payload_width * torch.finfo(compute_dtype).bits // 8
    return max(1, min(route_count, target_bytes // max(1, bytes_per_route)))


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


def _route_pairwise(x: Tensor, anchors: Tensor, thresholds: Tensor, powers: Tensor) -> PairwiseRoute:
    batch, seq, _ = x.shape
    tables, comparisons, _ = anchors.shape
    a = anchors[:, :, 0].flatten()
    b = anchors[:, :, 1].flatten()
    left = x[..., a].view(batch, seq, tables, comparisons)
    right = x[..., b].view(batch, seq, tables, comparisons)
    margins = left - right - thresholds.to(dtype=x.dtype, device=x.device)
    indices = ((margins > 0).to(torch.long) * powers.to(device=x.device).view(1, 1, 1, -1)).sum(dim=-1)
    return PairwiseRoute(indices=indices, margins=margins)


def _lookup_lut_rows(indices: Tensor, lut: Tensor, *, table_size: int, output_dim: int, compute_dtype: torch.dtype) -> Tensor:
    prefix_shape = indices.shape[:-1]
    route_count = indices.shape[-1]
    item_count = max(1, indices.numel() // route_count)
    flat_indices = indices.reshape(item_count, route_count)
    flat_lut = lut.reshape(route_count * table_size, output_dim)
    output = torch.zeros(item_count, output_dim, device=indices.device, dtype=compute_dtype)
    chunk = _route_chunk_size(item_count=item_count, payload_width=output_dim, compute_dtype=compute_dtype, route_count=route_count)

    for route_start in range(0, route_count, chunk):
        route_stop = min(route_start + chunk, route_count)
        offsets = (torch.arange(route_start, route_stop, device=indices.device) * table_size).view(1, -1)
        rows = (flat_indices[:, route_start:route_stop] + offsets).reshape(-1)
        values = flat_lut.index_select(0, rows).view(item_count, route_stop - route_start, output_dim)
        output = output + values.sum(dim=1)

    return output.view(*prefix_shape, output_dim)


def _min_margin_neighbors(route: PairwiseRoute, surrogate: str) -> tuple[Tensor, Tensor, Tensor]:
    bit = route.margins.abs().argmin(dim=-1)
    margin = route.margins.gather(dim=-1, index=bit.unsqueeze(-1)).squeeze(-1)
    neighbor = route.indices ^ (2**bit).long()
    ste = ste_heaviside(margin, surrogate) - (margin > 0).to(margin.dtype)
    return neighbor, ste, margin


def _ste_lut_delta(
    *,
    current_indices: Tensor,
    neighbor_indices: Tensor,
    ste_weight: Tensor,
    lut: Tensor,
    table_size: int,
    output_dim: int,
) -> Tensor:
    prefix_shape = current_indices.shape[:-1]
    route_count = current_indices.shape[-1]
    item_count = max(1, current_indices.numel() // route_count)
    current_flat = current_indices.reshape(item_count, route_count)
    neighbor_flat = neighbor_indices.reshape(item_count, route_count)
    weight_flat = ste_weight.reshape(item_count, route_count, 1).float()
    flat_lut = lut.to(dtype=torch.float32, device=current_indices.device).reshape(route_count * table_size, output_dim)
    output = torch.zeros(item_count, output_dim, device=current_indices.device, dtype=torch.float32)
    chunk = _route_chunk_size(item_count=item_count, payload_width=output_dim, compute_dtype=torch.float32, route_count=route_count)

    for route_start in range(0, route_count, chunk):
        route_stop = min(route_start + chunk, route_count)
        offsets = (torch.arange(route_start, route_stop, device=current_indices.device) * table_size).view(1, -1)
        current_rows = (current_flat[:, route_start:route_stop] + offsets).reshape(-1)
        neighbor_rows = (neighbor_flat[:, route_start:route_stop] + offsets).reshape(-1)
        current = flat_lut.index_select(0, current_rows).view(item_count, route_stop - route_start, output_dim)
        neighbor = flat_lut.index_select(0, neighbor_rows).view(item_count, route_stop - route_start, output_dim)
        output = output + (weight_flat[:, route_start:route_stop] * (neighbor - current)).sum(dim=1)

    return output.view(*prefix_shape, output_dim)


def _full_ste_lut_delta(route: PairwiseRoute, powers: Tensor, lut: Tensor, *, table_size: int, output_dim: int, surrogate: str) -> Tensor:
    prefix_shape = route.indices.shape[:-1]
    route_count = route.indices.shape[-1]
    comparisons = route.margins.shape[-1]
    item_count = max(1, route.indices.numel() // route_count)
    current_flat = route.indices.reshape(item_count, route_count)
    neighbor_flat = current_flat.unsqueeze(-1) ^ powers.to(device=route.indices.device).view(1, 1, -1)
    ste = ste_heaviside(route.margins, surrogate) - (route.margins > 0).to(route.margins.dtype)
    ste_flat = ste.reshape(item_count, route_count, comparisons, 1).float()
    flat_lut = lut.to(dtype=torch.float32, device=route.indices.device).reshape(route_count * table_size, output_dim)
    output = torch.zeros(item_count, output_dim, device=route.indices.device, dtype=torch.float32)
    chunk = _route_chunk_size(
        item_count=item_count,
        payload_width=output_dim * (comparisons + 1),
        compute_dtype=torch.float32,
        route_count=route_count,
        target_bytes=8 * 1024 * 1024,
    )

    for route_start in range(0, route_count, chunk):
        route_stop = min(route_start + chunk, route_count)
        offsets = (torch.arange(route_start, route_stop, device=route.indices.device) * table_size).view(1, -1)
        current_rows = (current_flat[:, route_start:route_stop] + offsets).reshape(-1)
        current = flat_lut.index_select(0, current_rows).view(item_count, route_stop - route_start, 1, output_dim)
        neighbor_rows = (neighbor_flat[:, route_start:route_stop] + offsets.unsqueeze(-1)).reshape(-1)
        neighbor = flat_lut.index_select(0, neighbor_rows).view(item_count, route_stop - route_start, comparisons, output_dim)
        output = output + (ste_flat[:, route_start:route_stop] * (neighbor - current)).sum(dim=(1, 2))

    return output.view(*prefix_shape, output_dim)


class PairwiseLUT(LUTModuleBase):
    """Pairwise comparison lookup table.

    The computation is deliberately small enough to read as the algorithm:
    route comparisons, obtain a payload table, gather selected rows, then add a
    finite-difference STE correction during training.
    """

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
        if tables < 1 or comparisons < 1:
            raise ValueError("tables and comparisons must be positive")
        if backend not in {"torch", "tilelang", "zig"}:
            raise ValueError(f"unsupported backend {backend!r}")
        if cpu_lut_dtype not in {"f32", "f16"}:
            raise ValueError("cpu_lut_dtype must be 'f32' or 'f16'")
        surrogate_gradient(torch.zeros((), dtype=torch.float32), surrogate)

        output_scale = 1.0 / math.sqrt(tables) if use_output_scaling else 1.0
        super().__init__(input_dim, output_dim, backend=backend, output_scale=output_scale)
        self.tables = int(tables)
        self.comparisons = int(comparisons)
        self.table_size = 1 << self.comparisons
        self.use_min_margin_ste = bool(use_min_margin_ste)
        self.surrogate = surrogate
        self.cpu_lut_dtype = cpu_lut_dtype
        self.anchor_policy = anchor_policy
        self.anchor_seed = seed if anchor_seed is None else int(anchor_seed)
        self._zig_lut_f16_cache: Tensor | None = None
        self._zig_lut_f16_cache_version = -1

        self.register_buffer(
            "anchors",
            _make_pairwise_anchors(input_dim, tables, comparisons, policy=anchor_policy, seed=self.anchor_seed),
        )
        thresholds = torch.zeros(tables, comparisons)
        if fixed_zero_threshold:
            self.register_buffer("thresholds", thresholds)
        else:
            self.thresholds = nn.Parameter(thresholds)
        if lut_init_std == 0.0:
            self.lut = nn.Parameter(torch.zeros(tables, self.table_size, output_dim))
        else:
            gen = torch.Generator(device="cpu").manual_seed(seed)
            self.lut = nn.Parameter(torch.randn(tables, self.table_size, output_dim, generator=gen) * lut_init_std)
        self.register_buffer("powers", 2 ** torch.arange(comparisons, dtype=torch.long))

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, output_dim={self.output_dim}, tables={self.tables}, "
            f"comparisons={self.comparisons}, backend={self.backend!r}, anchor_policy={self.anchor_policy!r}"
        )

    def payload_table(self, *, dtype: torch.dtype, device: torch.device) -> Tensor:
        return self.lut.to(dtype=dtype, device=device)

    def route(self, x: Tensor) -> PairwiseRoute:
        return _route_pairwise(x, self.anchors, self.thresholds, self.powers)

    def lookup(self, route: PairwiseRoute, lut: Tensor, *, compute_dtype: torch.dtype) -> Tensor:
        return _lookup_lut_rows(route.indices, lut, table_size=self.table_size, output_dim=self.output_dim, compute_dtype=compute_dtype)

    def ste_correction(self, route: PairwiseRoute, lut: Tensor) -> Tensor:
        if self.use_min_margin_ste:
            neighbor, ste, _ = _min_margin_neighbors(route, self.surrogate)
            return _ste_lut_delta(
                current_indices=route.indices,
                neighbor_indices=neighbor,
                ste_weight=ste,
                lut=lut,
                table_size=self.table_size,
                output_dim=self.output_dim,
            )
        return _full_ste_lut_delta(route, self.powers, lut, table_size=self.table_size, output_dim=self.output_dim, surrogate=self.surrogate)

    def _zig_lut_for_inference(self) -> Tensor:
        if self.cpu_lut_dtype == "f32":
            return self.lut.detach().to(device="cpu", dtype=torch.float32).contiguous()
        version = self.lut._version
        cache = self._zig_lut_f16_cache
        if cache is None or self._zig_lut_f16_cache_version != version or cache.shape != self.lut.shape:
            cache = self.lut.detach().to(device="cpu", dtype=torch.float16).contiguous()
            self._zig_lut_f16_cache = cache
            self._zig_lut_f16_cache_version = version
        return cache

    def _tilelang_compute(self, x: Tensor, *, compute_dtype: torch.dtype) -> tuple[Tensor, Tensor, Tensor]:
        from ..backends import pairwise_tilelang

        output, indices, margins = pairwise_tilelang(
            x.to(torch.float32),
            self.anchors.to(device=x.device),
            self.thresholds.to(dtype=torch.float32, device=x.device),
            self.lut.to(dtype=torch.float32, device=x.device),
            use_min_margin_ste=self.use_min_margin_ste,
            surrogate=self.surrogate,
        )
        return output.to(compute_dtype), indices, margins

    def _zig_compute(self, x: Tensor, route: PairwiseRoute, *, compute_dtype: torch.dtype, training: bool) -> tuple[Tensor, Tensor, Tensor]:
        if training:
            raise RuntimeError("PairwiseLUT backend='zig' is inference-only; call .eval() or use backend='torch' for training")
        from ..backends import pairwise_zig_forward

        output = pairwise_zig_forward(
            x,
            self.anchors,
            self.thresholds.detach().to(dtype=torch.float32, device="cpu"),
            self._zig_lut_for_inference(),
            lut_dtype=self.cpu_lut_dtype,
        )
        return output.to(device=x.device, dtype=compute_dtype), route.indices, route.margins

    def compute(self, x: Tensor, *, compute_dtype: torch.dtype, training: bool) -> tuple[Tensor, Tensor, Tensor]:
        x = x.to(torch.float32 if self.backend in {"tilelang", "zig"} else compute_dtype)
        if self.backend == "tilelang":
            return self._tilelang_compute(x, compute_dtype=compute_dtype)

        route = self.route(x)
        if self.backend == "zig":
            return self._zig_compute(x, route, compute_dtype=compute_dtype, training=training)

        lut = self.payload_table(dtype=compute_dtype, device=route.indices.device)
        output = self.lookup(route, lut, compute_dtype=compute_dtype)
        if training and (x.requires_grad or bool(getattr(self.thresholds, "requires_grad", False))):
            output = output + self.ste_correction(route, lut).to(output.dtype)
        return output, route.indices, route.margins


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
        super().__init__(
            input_dim,
            output_dim,
            tables=tables,
            comparisons=comparisons,
            backend=backend,
            seed=seed,
            lut_init_std=0.0,
            use_min_margin_ste=use_min_margin_ste,
            use_output_scaling=use_output_scaling,
            fixed_zero_threshold=fixed_zero_threshold,
            surrogate=surrogate,
            anchor_policy=anchor_policy,
            anchor_seed=anchor_seed,
        )
        del self.lut
        self.walsh_order = int(walsh_order)
        pair_indices = torch.combinations(torch.arange(comparisons, dtype=torch.long), r=2)
        if walsh_order == 1:
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

    def materialize_lut(self, *, dtype: torch.dtype | None = None, device: torch.device | None = None) -> Tensor:
        compute_dtype = dtype if dtype is not None else self.constant.dtype
        compute_device = device if device is not None else self.constant.device
        bits = self.walsh_bits.to(dtype=compute_dtype, device=compute_device)
        output = self.constant.to(dtype=compute_dtype, device=compute_device).unsqueeze(1)
        linear_terms = bits.view(1, self.table_size, self.comparisons, 1) * self.linear_coeff.to(
            dtype=compute_dtype,
            device=compute_device,
        ).view(self.tables, 1, self.comparisons, self.output_dim)
        output = output + linear_terms.sum(dim=2)
        if self.pair_indices.numel() > 0:
            pairs = self.pair_indices.to(device=compute_device)
            pair_bits = bits[:, pairs[:, 0]] * bits[:, pairs[:, 1]]
            pair_terms = pair_bits.view(1, self.table_size, -1, 1) * self.pair_coeff.to(
                dtype=compute_dtype,
                device=compute_device,
            ).view(self.tables, 1, -1, self.output_dim)
            output = output + pair_terms.sum(dim=2)
        return output

    def payload_table(self, *, dtype: torch.dtype, device: torch.device) -> Tensor:
        return self.materialize_lut(dtype=dtype, device=device)


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
        if features < 1 or output_dim < 1 or tables < 1 or comparisons < 1:
            raise ValueError("features, output_dim, tables, and comparisons must be positive")
        if width_init <= 0:
            raise ValueError("width_init must be positive")
        surrogate_gradient(torch.zeros((), dtype=torch.float32), surrogate)

        self.features = int(features)
        self.output_dim = int(output_dim)
        self.tables = int(tables)
        self.comparisons = int(comparisons)
        self.table_size = 1 << int(comparisons)
        self.use_min_margin_ste = bool(use_min_margin_ste)
        self.surrogate = surrogate
        self.output_scale = 1.0 / math.sqrt(tables) if use_output_scaling else 1.0
        self.cache_route_debug = True
        self._last_indices: Tensor | None = None
        self._last_margins: Tensor | None = None

        gen = torch.Generator(device="cpu").manual_seed(seed)
        self.register_buffer("coords", torch.randint(0, features, (tables, comparisons), generator=gen, dtype=torch.long))
        self.log_widths = nn.Parameter(torch.full((tables, comparisons), self._inverse_softplus(width_init)))
        self.lut = nn.Parameter(torch.randn(tables, self.table_size, output_dim, generator=gen) * lut_init_std)
        self.register_buffer("powers", 2 ** torch.arange(comparisons, dtype=torch.long))

    @staticmethod
    def _inverse_softplus(value: float) -> float:
        return math.log(math.expm1(value))

    @staticmethod
    def _compute_dtype(query: Tensor, key: Tensor) -> torch.dtype:
        dtype = torch.promote_types(query.dtype, key.dtype)
        return torch.float32 if dtype in {torch.float16, torch.bfloat16} else dtype

    def route(self, query: Tensor, key: Tensor) -> PairwiseRoute:
        coord_flat = self.coords.reshape(-1)
        q = query[..., coord_flat].view(*query.shape[:-1], self.tables, self.comparisons)
        k = key[..., coord_flat].view(*key.shape[:-1], self.tables, self.comparisons)
        widths = F.softplus(self.log_widths).to(dtype=query.dtype, device=query.device)
        margins = widths - (q - k).abs()
        indices = ((margins > 0).to(torch.long) * self.powers.to(device=query.device).view(1, 1, -1)).sum(dim=-1)
        return PairwiseRoute(indices=indices, margins=margins)

    def lookup(self, route: PairwiseRoute, *, compute_dtype: torch.dtype) -> Tensor:
        return _lookup_lut_rows(
            route.indices,
            self.lut.to(dtype=compute_dtype, device=route.indices.device),
            table_size=self.table_size,
            output_dim=self.output_dim,
            compute_dtype=compute_dtype,
        )

    def ste_correction(self, route: PairwiseRoute) -> Tensor:
        lut = self.lut.to(dtype=torch.float32, device=route.indices.device)
        if self.use_min_margin_ste:
            neighbor, ste, _ = _min_margin_neighbors(route, self.surrogate)
            return _ste_lut_delta(
                current_indices=route.indices,
                neighbor_indices=neighbor,
                ste_weight=ste,
                lut=lut,
                table_size=self.table_size,
                output_dim=self.output_dim,
            )
        return _full_ste_lut_delta(route, self.powers, lut, table_size=self.table_size, output_dim=self.output_dim, surrogate=self.surrogate)

    def forward(self, query: Tensor, key: Tensor) -> Tensor:
        output_dtype = query.dtype
        compute_dtype = self._compute_dtype(query, key)
        route = self.route(query.to(compute_dtype), key.to(compute_dtype))
        output = self.lookup(route, compute_dtype=compute_dtype)
        if self.training and (query.requires_grad or key.requires_grad or self.log_widths.requires_grad):
            output = output + self.ste_correction(route).to(output.dtype)
        if self.output_scale != 1.0:
            output = output * self.output_scale
        if self.cache_route_debug:
            self._last_indices = route.indices.detach()
            self._last_margins = route.margins.detach()
        else:
            self._last_indices = None
            self._last_margins = None
        return output.to(dtype=output_dtype)


__all__ = [
    "AbsDiffLUT",
    "PAIRWISE_ANCHOR_POLICIES",
    "PairwiseLUT",
    "PairwiseRoute",
    "PairwiseWalshLUT",
    "_make_pairwise_anchors",
]
