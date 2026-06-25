from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..backend import Backend
from .base import RoutedLinearBase
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


def _make_pairwise_anchors(
    in_features: int,
    tables: int,
    comparisons: int,
    *,
    policy: str,
    seed: int | None,
) -> Tensor:
    if in_features < 2:
        raise ValueError(f"Pairwise anchors require in_features >= 2, got {in_features}")
    if policy not in PAIRWISE_ANCHOR_POLICIES:
        raise ValueError(f"unsupported pairwise anchor policy {policy!r}; expected one of {PAIRWISE_ANCHOR_POLICIES}")

    route_count = tables * comparisons
    flat = torch.arange(route_count, dtype=torch.long)
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(int(seed))

    if policy == "random":
        a = torch.randint(0, in_features, (route_count,), generator=gen)
        b = torch.randint(0, in_features - 1, (route_count,), generator=gen)
        b = b + (b >= a).long()
    elif policy == "random_no_replace":
        pair_count = in_features * (in_features - 1)
        if route_count <= pair_count and pair_count <= 2_000_000:
            raw = torch.randperm(pair_count, generator=gen)[:route_count]
            a = raw // (in_features - 1)
            b = raw % (in_features - 1)
            b = b + (b >= a).long()
        else:
            a = torch.randint(0, in_features, (route_count,), generator=gen)
            b = torch.randint(0, in_features - 1, (route_count,), generator=gen)
            b = b + (b >= a).long()
    elif policy == "local":
        a = flat.remainder(in_features)
        b = (a + 1).remainder(in_features)
    elif policy == "cyclic":
        a = flat.remainder(in_features)
        stride = ((flat // in_features) + (flat % comparisons) + 1).remainder(in_features - 1) + 1
        b = (a + stride).remainder(in_features)
    elif policy == "block":
        block = min(in_features, max(2, int(math.sqrt(in_features))))
        block_count = (in_features + block - 1) // block
        block_id = (flat // block).remainder(block_count)
        within = flat.remainder(block)
        a = (block_id * block + within).remainder(in_features)
        offset = ((flat // max(1, block_count)) % (block - 1)) + 1
        b = (block_id * block + within + offset).remainder(in_features)
    elif policy == "expander":
        a = (flat * 37 + 17).remainder(in_features)
        stride = ((flat * 53 + 19) % (in_features - 1)) + 1
        b = (a + stride).remainder(in_features)
    elif policy == "permuted":
        perm_a = torch.randperm(in_features, generator=gen)
        perm_b = torch.randperm(in_features, generator=gen)
        a = perm_a[flat.remainder(in_features)]
        b = perm_b[(flat * 7 + flat // max(1, in_features)).remainder(in_features)]
        b = torch.where(a == b, (b + 1).remainder(in_features), b)
    else:
        raise AssertionError(f"unreachable anchor policy {policy!r}")

    b = torch.where(a == b, (b + 1).remainder(in_features), b)
    return torch.stack((a, b), dim=-1).view(tables, comparisons, 2)


class PairwiseLinear(RoutedLinearBase):
    """Compare -> lookup -> accumulate layer with no GEMM in the forward path.

    Each table owns a small set of coordinate comparisons. The comparison bits
    index one row in that table's payload LUT, and the selected rows are summed.
    During training, finite-difference STE sends gradients through the nearest
    comparator boundary while LUT rows receive ordinary selected-row gradients.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
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
        if tables < 1:
            raise ValueError(f"tables must be >= 1, got {tables}")
        if comparisons < 1:
            raise ValueError(f"comparisons must be >= 1, got {comparisons}")
        if backend not in {"torch", "tilelang", "zig"}:
            raise ValueError(f"PairwiseLinear supports backend='torch', 'tilelang', or 'zig', got {backend!r}")
        if cpu_lut_dtype not in {"f32", "f16"}:
            raise ValueError(f"cpu_lut_dtype must be 'f32' or 'f16', got {cpu_lut_dtype!r}")
        surrogate_gradient(torch.zeros((), dtype=torch.float32), surrogate)

        output_scale = 1.0 / math.sqrt(tables) if use_output_scaling else 1.0
        super().__init__(in_features, out_features, backend=backend, output_scale=output_scale)

        self.tables = int(tables)
        self.comparisons = int(comparisons)
        self.table_size = 1 << int(comparisons)
        self.use_min_margin_ste = bool(use_min_margin_ste)
        self.fixed_zero_threshold = bool(fixed_zero_threshold)
        self.surrogate = surrogate
        self.cpu_lut_dtype = cpu_lut_dtype
        self.anchor_policy = anchor_policy
        self.anchor_seed = seed if anchor_seed is None else int(anchor_seed)
        self._zig_lut_f16_cache: Tensor | None = None
        self._zig_lut_f16_cache_version = -1

        anchors = _make_pairwise_anchors(
            in_features,
            tables,
            comparisons,
            policy=anchor_policy,
            seed=self.anchor_seed,
        )
        self.register_buffer("anchors", anchors)
        thresholds = torch.zeros(tables, comparisons)
        if fixed_zero_threshold:
            self.register_buffer("thresholds", thresholds)
        else:
            self.thresholds = nn.Parameter(thresholds)
        if lut_init_std == 0.0:
            self.lut = nn.Parameter(torch.zeros(tables, self.table_size, out_features))
        else:
            gen = torch.Generator(device="cpu").manual_seed(seed)
            self.lut = nn.Parameter(torch.randn(tables, self.table_size, out_features, generator=gen) * lut_init_std)
        self.register_buffer("powers", 2 ** torch.arange(comparisons, dtype=torch.long))

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, tables={self.tables}, "
            f"comparisons={self.comparisons}, backend={self.backend!r}, use_min_margin_ste={self.use_min_margin_ste}, "
            f"fixed_zero_threshold={self.fixed_zero_threshold}, surrogate={self.surrogate!r}, "
            f"cpu_lut_dtype={self.cpu_lut_dtype!r}, anchor_policy={self.anchor_policy!r}, anchor_seed={self.anchor_seed}"
        )

    def _project_input(self, x: Tensor, compute_dtype: torch.dtype) -> Tensor:
        if self.backend in {"tilelang", "zig"}:
            return x.to(torch.float32)
        return x.to(compute_dtype)

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

    def _lookup_chunked(self, indices: Tensor, *, compute_dtype: torch.dtype) -> Tensor:
        batch, seq, route_count = indices.shape
        item_count = batch * seq
        indices_flat = indices.reshape(item_count, route_count)
        lut_table = self.lut.to(dtype=compute_dtype, device=indices.device).reshape(
            route_count * self.table_size,
            self.out_features,
        )
        route_chunk = self._route_chunk_size(
            item_count=item_count,
            payload_width=self.out_features,
            compute_dtype=compute_dtype,
            route_count=route_count,
        )
        output = torch.zeros(item_count, self.out_features, device=indices.device, dtype=compute_dtype)

        for route_start in range(0, route_count, route_chunk):
            route_stop = min(route_start + route_chunk, route_count)
            route_offsets = (torch.arange(route_start, route_stop, device=indices.device) * self.table_size).view(1, -1)
            linear_idx = (indices_flat[:, route_start:route_stop] + route_offsets).reshape(-1)
            selected = lut_table.index_select(0, linear_idx).view(item_count, route_stop - route_start, self.out_features)
            output = output + selected.sum(dim=1)

        return output.view(batch, seq, self.out_features)

    def _compute_indices(self, latent: Tensor) -> tuple[Tensor, Tensor]:
        batch, seq, _ = latent.shape
        anchor_a = self.anchors[:, :, 0].flatten()
        anchor_b = self.anchors[:, :, 1].flatten()
        x_a = latent[..., anchor_a].view(batch, seq, self.tables, self.comparisons)
        x_b = latent[..., anchor_b].view(batch, seq, self.tables, self.comparisons)
        thresholds = self.thresholds.to(dtype=latent.dtype, device=latent.device)
        margins = x_a - x_b - thresholds
        indices = ((margins > 0).to(torch.long) * self.powers.to(device=latent.device).view(1, 1, 1, -1)).sum(dim=-1)
        return indices, margins

    def _min_margin_ste(self, indices: Tensor, margins: Tensor) -> Tensor:
        r_mins = margins.abs().argmin(dim=-1)
        u_mins = margins.gather(dim=-1, index=r_mins.unsqueeze(-1)).squeeze(-1)
        neighbor_indices = indices ^ (2**r_mins).long()
        ste_delta = ste_heaviside(u_mins, self.surrogate) - (u_mins > 0).to(u_mins.dtype)

        batch, seq, route_count = indices.shape
        item_count = batch * seq
        current_flat = indices.reshape(item_count, route_count)
        neighbor_flat = neighbor_indices.reshape(item_count, route_count)
        ste_flat = ste_delta.reshape(item_count, route_count, 1).float()
        lut_table = self.lut.to(dtype=torch.float32, device=indices.device).reshape(route_count * self.table_size, self.out_features)
        route_chunk = self._route_chunk_size(
            item_count=item_count,
            payload_width=self.out_features,
            compute_dtype=torch.float32,
            route_count=route_count,
        )
        corr = torch.zeros(item_count, self.out_features, device=indices.device, dtype=torch.float32)

        for route_start in range(0, route_count, route_chunk):
            route_stop = min(route_start + route_chunk, route_count)
            route_offsets = (torch.arange(route_start, route_stop, device=indices.device) * self.table_size).view(1, -1)
            current_idx = (current_flat[:, route_start:route_stop] + route_offsets).reshape(-1)
            neighbor_idx = (neighbor_flat[:, route_start:route_stop] + route_offsets).reshape(-1)
            current = lut_table.index_select(0, current_idx).view(item_count, route_stop - route_start, self.out_features)
            neighbor = lut_table.index_select(0, neighbor_idx).view(item_count, route_stop - route_start, self.out_features)
            corr = corr + (ste_flat[:, route_start:route_stop] * (neighbor - current)).sum(dim=1)

        return corr.view(batch, seq, self.out_features)

    def _full_ste(self, indices: Tensor, margins: Tensor) -> Tensor:
        batch, seq, route_count = indices.shape
        item_count = batch * seq
        current_flat = indices.reshape(item_count, route_count)
        powers = self.powers.to(device=indices.device)
        neighbor_flat = current_flat.unsqueeze(-1) ^ powers.view(1, 1, -1)
        ste_delta = ste_heaviside(margins, self.surrogate) - (margins > 0).to(margins.dtype)
        ste_flat = ste_delta.reshape(item_count, route_count, self.comparisons, 1).float()
        lut_table = self.lut.to(dtype=torch.float32, device=indices.device).reshape(route_count * self.table_size, self.out_features)
        route_chunk = self._route_chunk_size(
            item_count=item_count,
            payload_width=self.out_features * (self.comparisons + 1),
            compute_dtype=torch.float32,
            route_count=route_count,
            target_bytes=8 * 1024 * 1024,
        )
        corr = torch.zeros(item_count, self.out_features, device=indices.device, dtype=torch.float32)

        for route_start in range(0, route_count, route_chunk):
            route_stop = min(route_start + route_chunk, route_count)
            route_offsets = (torch.arange(route_start, route_stop, device=indices.device) * self.table_size).view(1, -1)
            current_idx = (current_flat[:, route_start:route_stop] + route_offsets).reshape(-1)
            current = lut_table.index_select(0, current_idx).view(item_count, route_stop - route_start, 1, self.out_features)
            neighbor_idx = (neighbor_flat[:, route_start:route_stop] + route_offsets.unsqueeze(-1)).reshape(-1)
            neighbor = lut_table.index_select(0, neighbor_idx).view(
                item_count,
                route_stop - route_start,
                self.comparisons,
                self.out_features,
            )
            corr = corr + (ste_flat[:, route_start:route_stop] * (neighbor - current)).sum(dim=(1, 2))

        return corr.view(batch, seq, self.out_features)

    def _route_output(
        self,
        latent: Tensor,
        *,
        input_device: torch.device,
        compute_dtype: torch.dtype,
        training: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        del input_device
        if self.backend == "tilelang":
            from ..backends import pairwise_tilelang

            thresholds = self.thresholds.to(dtype=torch.float32, device=latent.device)
            output, indices, margins = pairwise_tilelang(
                latent.to(torch.float32),
                self.anchors.to(device=latent.device),
                thresholds,
                self.lut.to(dtype=torch.float32, device=latent.device),
                use_min_margin_ste=self.use_min_margin_ste,
                surrogate=self.surrogate,
            )
            return output.to(compute_dtype), indices, margins

        indices, margins = self._compute_indices(latent)
        if self.backend == "zig":
            if training:
                raise RuntimeError("PairwiseLinear backend='zig' is inference-only; call .eval() or use backend='torch' for training")
            from ..backends import pairwise_zig_forward

            output = pairwise_zig_forward(
                latent,
                self.anchors,
                self.thresholds.detach().to(dtype=torch.float32, device="cpu"),
                self._zig_lut_for_inference(),
                lut_dtype=self.cpu_lut_dtype,
            )
            return output.to(device=latent.device, dtype=compute_dtype), indices, margins

        output = self._lookup_chunked(indices, compute_dtype=compute_dtype)
        threshold_has_grad = bool(getattr(self.thresholds, "requires_grad", False))
        if training and (latent.requires_grad or threshold_has_grad):
            ste_corr = self._min_margin_ste(indices, margins) if self.use_min_margin_ste else self._full_ste(indices, margins)
            output = output + ste_corr.to(output.dtype)
        return output, indices, margins


class AbsDiffLUT(nn.Module):
    """Two-input compare/LUT relation layer for query-key coordinate agreement."""

    def __init__(
        self,
        features: int,
        out_features: int,
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
        if features < 1:
            raise ValueError(f"features must be >= 1, got {features}")
        if out_features < 1:
            raise ValueError(f"out_features must be >= 1, got {out_features}")
        if tables < 1:
            raise ValueError(f"tables must be >= 1, got {tables}")
        if comparisons < 1:
            raise ValueError(f"comparisons must be >= 1, got {comparisons}")
        if width_init <= 0:
            raise ValueError(f"width_init must be > 0, got {width_init}")
        surrogate_gradient(torch.zeros((), dtype=torch.float32), surrogate)

        self.features = int(features)
        self.out_features = int(out_features)
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
        coords = torch.randint(0, features, (tables, comparisons), generator=gen, dtype=torch.long)
        self.register_buffer("coords", coords)
        self.log_widths = nn.Parameter(torch.full((tables, comparisons), self._inverse_softplus(width_init)))
        self.lut = nn.Parameter(torch.randn(tables, self.table_size, out_features, generator=gen) * lut_init_std)
        self.register_buffer("powers", 2 ** torch.arange(comparisons, dtype=torch.long))

    @staticmethod
    def _inverse_softplus(value: float) -> float:
        return math.log(math.expm1(value))

    def extra_repr(self) -> str:
        widths = F.softplus(self.log_widths.detach())
        return (
            f"features={self.features}, out_features={self.out_features}, tables={self.tables}, "
            f"comparisons={self.comparisons}, width_mean={float(widths.mean()):.4f}, "
            f"use_min_margin_ste={self.use_min_margin_ste}, surrogate={self.surrogate!r}"
        )

    def _compute_dtype(self, query: Tensor, key: Tensor) -> torch.dtype:
        dtype = torch.promote_types(query.dtype, key.dtype)
        return torch.float32 if dtype in {torch.float16, torch.bfloat16} else dtype

    def _route_chunk_size(
        self,
        *,
        item_count: int,
        payload_width: int,
        compute_dtype: torch.dtype,
        route_count: int,
        target_bytes: int = 16 * 1024 * 1024,
    ) -> int:
        bytes_per_route = item_count * payload_width * torch.finfo(compute_dtype).bits // 8
        return max(1, min(route_count, target_bytes // max(1, bytes_per_route)))

    def _lookup_chunked(self, indices: Tensor, *, compute_dtype: torch.dtype) -> Tensor:
        prefix_shape = indices.shape[:-1]
        route_count = indices.shape[-1]
        item_count = max(1, indices.numel() // route_count)
        indices_flat = indices.reshape(item_count, route_count)
        lut_table = self.lut.to(dtype=compute_dtype, device=indices.device).reshape(
            route_count * self.table_size,
            self.out_features,
        )
        route_chunk = self._route_chunk_size(
            item_count=item_count,
            payload_width=self.out_features,
            compute_dtype=compute_dtype,
            route_count=route_count,
        )
        output = torch.zeros(item_count, self.out_features, device=indices.device, dtype=compute_dtype)

        for route_start in range(0, route_count, route_chunk):
            route_stop = min(route_start + route_chunk, route_count)
            route_offsets = (torch.arange(route_start, route_stop, device=indices.device) * self.table_size).view(1, -1)
            linear_idx = (indices_flat[:, route_start:route_stop] + route_offsets).reshape(-1)
            selected = lut_table.index_select(0, linear_idx).view(item_count, route_stop - route_start, self.out_features)
            output = output + selected.sum(dim=1)

        return output.view(*prefix_shape, self.out_features)

    def _compute_indices(self, query: Tensor, key: Tensor) -> tuple[Tensor, Tensor]:
        coord_flat = self.coords.reshape(-1)
        q = query[..., coord_flat].view(*query.shape[:-1], self.tables, self.comparisons)
        k = key[..., coord_flat].view(*key.shape[:-1], self.tables, self.comparisons)
        widths = F.softplus(self.log_widths).to(dtype=query.dtype, device=query.device)
        margins = widths - (q - k).abs()
        indices = ((margins > 0).to(torch.long) * self.powers.to(device=query.device).view(1, 1, -1)).sum(dim=-1)
        return indices, margins

    def _min_margin_ste(self, indices: Tensor, margins: Tensor) -> Tensor:
        r_mins = margins.abs().argmin(dim=-1)
        u_mins = margins.gather(dim=-1, index=r_mins.unsqueeze(-1)).squeeze(-1)
        neighbor_indices = indices ^ (2**r_mins).long()
        ste_delta = ste_heaviside(u_mins, self.surrogate) - (u_mins > 0).to(u_mins.dtype)

        prefix_shape = indices.shape[:-1]
        route_count = indices.shape[-1]
        item_count = max(1, indices.numel() // route_count)
        current_flat = indices.reshape(item_count, route_count)
        neighbor_flat = neighbor_indices.reshape(item_count, route_count)
        ste_flat = ste_delta.reshape(item_count, route_count, 1).float()
        lut_table = self.lut.to(dtype=torch.float32, device=indices.device).reshape(route_count * self.table_size, self.out_features)
        route_chunk = self._route_chunk_size(
            item_count=item_count,
            payload_width=self.out_features,
            compute_dtype=torch.float32,
            route_count=route_count,
        )
        corr = torch.zeros(item_count, self.out_features, device=indices.device, dtype=torch.float32)

        for route_start in range(0, route_count, route_chunk):
            route_stop = min(route_start + route_chunk, route_count)
            route_offsets = (torch.arange(route_start, route_stop, device=indices.device) * self.table_size).view(1, -1)
            current_idx = (current_flat[:, route_start:route_stop] + route_offsets).reshape(-1)
            neighbor_idx = (neighbor_flat[:, route_start:route_stop] + route_offsets).reshape(-1)
            current = lut_table.index_select(0, current_idx).view(item_count, route_stop - route_start, self.out_features)
            neighbor = lut_table.index_select(0, neighbor_idx).view(item_count, route_stop - route_start, self.out_features)
            corr = corr + (ste_flat[:, route_start:route_stop] * (neighbor - current)).sum(dim=1)

        return corr.view(*prefix_shape, self.out_features)

    def _full_ste(self, indices: Tensor, margins: Tensor) -> Tensor:
        prefix_shape = indices.shape[:-1]
        route_count = indices.shape[-1]
        item_count = max(1, indices.numel() // route_count)
        current_flat = indices.reshape(item_count, route_count)
        neighbor_flat = current_flat.unsqueeze(-1) ^ self.powers.to(device=indices.device).view(1, 1, -1)
        ste_delta = ste_heaviside(margins, self.surrogate) - (margins > 0).to(margins.dtype)
        ste_flat = ste_delta.reshape(item_count, route_count, self.comparisons, 1).float()
        lut_table = self.lut.to(dtype=torch.float32, device=indices.device).reshape(route_count * self.table_size, self.out_features)
        route_chunk = self._route_chunk_size(
            item_count=item_count,
            payload_width=self.out_features * (self.comparisons + 1),
            compute_dtype=torch.float32,
            route_count=route_count,
            target_bytes=8 * 1024 * 1024,
        )
        corr = torch.zeros(item_count, self.out_features, device=indices.device, dtype=torch.float32)

        for route_start in range(0, route_count, route_chunk):
            route_stop = min(route_start + route_chunk, route_count)
            route_offsets = (torch.arange(route_start, route_stop, device=indices.device) * self.table_size).view(1, -1)
            current_idx = (current_flat[:, route_start:route_stop] + route_offsets).reshape(-1)
            current = lut_table.index_select(0, current_idx).view(item_count, route_stop - route_start, 1, self.out_features)
            neighbor_idx = (neighbor_flat[:, route_start:route_stop] + route_offsets.unsqueeze(-1)).reshape(-1)
            neighbor = lut_table.index_select(0, neighbor_idx).view(
                item_count,
                route_stop - route_start,
                self.comparisons,
                self.out_features,
            )
            corr = corr + (ste_flat[:, route_start:route_stop] * (neighbor - current)).sum(dim=(1, 2))

        return corr.view(*prefix_shape, self.out_features)

    def forward(self, query: Tensor, key: Tensor) -> Tensor:
        if query.shape != key.shape:
            raise ValueError(f"query and key must have the same shape, got {tuple(query.shape)} and {tuple(key.shape)}")
        if query.ndim < 2:
            raise ValueError(f"query/key must have at least 2 dims [..., features], got {tuple(query.shape)}")
        if query.shape[-1] != self.features:
            raise ValueError(f"expected last dimension {self.features}, got {query.shape[-1]}")

        output_dtype = query.dtype
        compute_dtype = self._compute_dtype(query, key)
        q = query.to(compute_dtype)
        k = key.to(compute_dtype)
        indices, margins = self._compute_indices(q, k)
        output = self._lookup_chunked(indices, compute_dtype=compute_dtype)
        if self.training and (query.requires_grad or key.requires_grad or self.log_widths.requires_grad):
            ste_corr = self._min_margin_ste(indices, margins) if self.use_min_margin_ste else self._full_ste(indices, margins)
            output = output + ste_corr.to(output.dtype)
        if self.output_scale != 1.0:
            output = output * self.output_scale
        if self.cache_route_debug:
            self._last_indices = indices.detach()
            self._last_margins = margins.detach()
        else:
            self._last_indices = None
            self._last_margins = None
        return output.to(dtype=output_dtype)


class PairwiseWalshLinear(PairwiseLinear):
    """Pairwise LUT whose payload rows are generated from low-order Walsh terms.

    This is still a compare -> lookup -> accumulate operator at inference time:
    the generated table is materialized from trainable coefficients, then the
    normal PairwiseLinear row-selection path is used. Materialization uses only
    broadcasted elementwise products and reductions, not GEMM.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
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
            raise ValueError(f"walsh_order must be 1 or 2, got {walsh_order}")
        if backend != "torch":
            raise ValueError("PairwiseWalshLinear only supports backend='torch'")
        super().__init__(
            in_features,
            out_features,
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
        generator = torch.Generator(device="cpu").manual_seed(seed + 1)
        self.constant = nn.Parameter(torch.randn(tables, out_features, generator=generator) * init_std)
        self.linear_coeff = nn.Parameter(torch.randn(tables, comparisons, out_features, generator=generator) * init_std)
        self.pair_coeff = nn.Parameter(torch.randn(tables, pair_indices.shape[0], out_features, generator=generator) * init_std)

        bit_values = torch.arange(self.table_size, dtype=torch.long).unsqueeze(-1).bitwise_and(self.powers.view(1, -1))
        self.register_buffer("walsh_bits", torch.where(bit_values > 0, torch.ones_like(bit_values), -torch.ones_like(bit_values)).float())

    @property
    def walsh_term_count(self) -> int:
        return 1 + self.comparisons + int(self.pair_indices.shape[0])

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, tables={self.tables}, "
            f"comparisons={self.comparisons}, walsh_order={self.walsh_order}, backend={self.backend!r}, "
            f"use_min_margin_ste={self.use_min_margin_ste}, surrogate={self.surrogate!r}, "
            f"anchor_policy={self.anchor_policy!r}, anchor_seed={self.anchor_seed}"
        )

    def materialize_lut(self, *, dtype: torch.dtype | None = None, device: torch.device | None = None) -> Tensor:
        compute_dtype = dtype if dtype is not None else self.constant.dtype
        compute_device = device if device is not None else self.constant.device
        bits = self.walsh_bits.to(dtype=compute_dtype, device=compute_device)
        output = self.constant.to(dtype=compute_dtype, device=compute_device).unsqueeze(1)
        linear = bits.view(1, self.table_size, self.comparisons, 1) * self.linear_coeff.to(
            dtype=compute_dtype,
            device=compute_device,
        ).view(self.tables, 1, self.comparisons, self.out_features)
        output = output + linear.sum(dim=2)
        if self.pair_indices.numel() > 0:
            pairs = self.pair_indices.to(device=compute_device)
            pair_bits = bits[:, pairs[:, 0]] * bits[:, pairs[:, 1]]
            pairwise = pair_bits.view(1, self.table_size, -1, 1) * self.pair_coeff.to(
                dtype=compute_dtype,
                device=compute_device,
            ).view(self.tables, 1, -1, self.out_features)
            output = output + pairwise.sum(dim=2)
        return output

    def _lookup_chunked_with_lut(self, indices: Tensor, lut: Tensor, *, compute_dtype: torch.dtype) -> Tensor:
        batch, seq, route_count = indices.shape
        item_count = batch * seq
        indices_flat = indices.reshape(item_count, route_count)
        lut_table = lut.reshape(route_count * self.table_size, self.out_features)
        route_chunk = self._route_chunk_size(
            item_count=item_count,
            payload_width=self.out_features,
            compute_dtype=compute_dtype,
            route_count=route_count,
        )
        output = torch.zeros(item_count, self.out_features, device=indices.device, dtype=compute_dtype)
        for route_start in range(0, route_count, route_chunk):
            route_stop = min(route_start + route_chunk, route_count)
            route_offsets = (torch.arange(route_start, route_stop, device=indices.device) * self.table_size).view(1, -1)
            linear_idx = (indices_flat[:, route_start:route_stop] + route_offsets).reshape(-1)
            selected = lut_table.index_select(0, linear_idx).view(item_count, route_stop - route_start, self.out_features)
            output = output + selected.sum(dim=1)
        return output.view(batch, seq, self.out_features)

    def _min_margin_ste_with_lut(self, indices: Tensor, margins: Tensor, lut: Tensor) -> Tensor:
        r_mins = margins.abs().argmin(dim=-1)
        u_mins = margins.gather(dim=-1, index=r_mins.unsqueeze(-1)).squeeze(-1)
        neighbor_indices = indices ^ (2**r_mins).long()
        ste_delta = ste_heaviside(u_mins, self.surrogate) - (u_mins > 0).to(u_mins.dtype)

        batch, seq, route_count = indices.shape
        item_count = batch * seq
        current_flat = indices.reshape(item_count, route_count)
        neighbor_flat = neighbor_indices.reshape(item_count, route_count)
        ste_flat = ste_delta.reshape(item_count, route_count, 1).float()
        lut_table = lut.to(dtype=torch.float32, device=indices.device).reshape(route_count * self.table_size, self.out_features)
        route_chunk = self._route_chunk_size(
            item_count=item_count,
            payload_width=self.out_features,
            compute_dtype=torch.float32,
            route_count=route_count,
        )
        corr = torch.zeros(item_count, self.out_features, device=indices.device, dtype=torch.float32)
        for route_start in range(0, route_count, route_chunk):
            route_stop = min(route_start + route_chunk, route_count)
            route_offsets = (torch.arange(route_start, route_stop, device=indices.device) * self.table_size).view(1, -1)
            current_idx = (current_flat[:, route_start:route_stop] + route_offsets).reshape(-1)
            neighbor_idx = (neighbor_flat[:, route_start:route_stop] + route_offsets).reshape(-1)
            current = lut_table.index_select(0, current_idx).view(item_count, route_stop - route_start, self.out_features)
            neighbor = lut_table.index_select(0, neighbor_idx).view(item_count, route_stop - route_start, self.out_features)
            corr = corr + (ste_flat[:, route_start:route_stop] * (neighbor - current)).sum(dim=1)
        return corr.view(batch, seq, self.out_features)

    def _full_ste_with_lut(self, indices: Tensor, margins: Tensor, lut: Tensor) -> Tensor:
        batch, seq, route_count = indices.shape
        item_count = batch * seq
        current_flat = indices.reshape(item_count, route_count)
        powers = self.powers.to(device=indices.device)
        neighbor_flat = current_flat.unsqueeze(-1) ^ powers.view(1, 1, -1)
        ste_delta = ste_heaviside(margins, self.surrogate) - (margins > 0).to(margins.dtype)
        ste_flat = ste_delta.reshape(item_count, route_count, self.comparisons, 1).float()
        lut_table = lut.to(dtype=torch.float32, device=indices.device).reshape(route_count * self.table_size, self.out_features)
        route_chunk = self._route_chunk_size(
            item_count=item_count,
            payload_width=self.out_features * (self.comparisons + 1),
            compute_dtype=torch.float32,
            route_count=route_count,
            target_bytes=8 * 1024 * 1024,
        )
        corr = torch.zeros(item_count, self.out_features, device=indices.device, dtype=torch.float32)
        for route_start in range(0, route_count, route_chunk):
            route_stop = min(route_start + route_chunk, route_count)
            route_offsets = (torch.arange(route_start, route_stop, device=indices.device) * self.table_size).view(1, -1)
            current_idx = (current_flat[:, route_start:route_stop] + route_offsets).reshape(-1)
            current = lut_table.index_select(0, current_idx).view(item_count, route_stop - route_start, 1, self.out_features)
            neighbor_idx = (neighbor_flat[:, route_start:route_stop] + route_offsets.unsqueeze(-1)).reshape(-1)
            neighbor = lut_table.index_select(0, neighbor_idx).view(
                item_count,
                route_stop - route_start,
                self.comparisons,
                self.out_features,
            )
            corr = corr + (ste_flat[:, route_start:route_stop] * (neighbor - current)).sum(dim=(1, 2))
        return corr.view(batch, seq, self.out_features)

    def _route_output(
        self,
        latent: Tensor,
        *,
        input_device: torch.device,
        compute_dtype: torch.dtype,
        training: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        del input_device
        indices, margins = self._compute_indices(latent)
        lut = self.materialize_lut(dtype=compute_dtype, device=indices.device)
        output = self._lookup_chunked_with_lut(indices, lut, compute_dtype=compute_dtype)
        threshold_has_grad = bool(getattr(self.thresholds, "requires_grad", False))
        if training and (latent.requires_grad or threshold_has_grad):
            ste_corr = self._min_margin_ste_with_lut(indices, margins, lut) if self.use_min_margin_ste else self._full_ste_with_lut(indices, margins, lut)
            output = output + ste_corr.to(output.dtype)
        return output, indices, margins


__all__ = [
    "AbsDiffLUT",
    "PAIRWISE_ANCHOR_POLICIES",
    "PairwiseLinear",
    "PairwiseWalshLinear",
    "_make_pairwise_anchors",
]
