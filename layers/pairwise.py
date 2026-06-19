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


def _next_power_of_two(n: int) -> int:
    return 1 << (max(1, n) - 1).bit_length()


def _fwht_last_dim(x: Tensor) -> Tensor:
    original_shape = x.shape
    batch_shape = x.shape[:-1]
    width = x.shape[-1]
    h = 1
    y = x
    while h < width:
        y = y.reshape(*batch_shape, -1, h * 2)
        a = y[..., :h]
        b = y[..., h : h * 2]
        y = torch.cat((a + b, a - b), dim=-1)
        h *= 2
    return y.reshape(*original_shape)


class PairwiseLinear(RoutedLinearBase):
    """Classic pairwise-comparator LUT layer with optional min-margin STE."""

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
        accumulation: Literal["sum", "two_bank_max"] = "sum",
        max_group_size: int = 4,
        slope_bank_rank: int = 0,
        slope_bank_atom_init_std: float = 0.02,
        slope_bank_coeff_init_std: float = 0.0,
    ) -> None:
        if tables < 1:
            raise ValueError(f"tables must be >= 1, got {tables}")
        if comparisons < 1:
            raise ValueError(f"comparisons must be >= 1, got {comparisons}")
        if backend not in {"torch", "tilelang", "zig"}:
            raise ValueError(f"PairwiseLinear currently supports backend='torch', 'tilelang', or 'zig', got {backend!r}")
        if cpu_lut_dtype not in {"f32", "f16"}:
            raise ValueError(f"cpu_lut_dtype must be 'f32' or 'f16', got {cpu_lut_dtype!r}")
        if accumulation not in {"sum", "two_bank_max"}:
            raise ValueError(f"accumulation must be 'sum' or 'two_bank_max', got {accumulation!r}")
        if max_group_size < 1:
            raise ValueError(f"max_group_size must be >= 1, got {max_group_size}")
        if accumulation == "two_bank_max" and backend != "torch":
            raise ValueError("PairwiseLinear accumulation='two_bank_max' currently supports backend='torch' only")
        if accumulation == "two_bank_max" and not use_min_margin_ste:
            raise ValueError("PairwiseLinear accumulation='two_bank_max' currently supports min-margin STE only")
        if slope_bank_rank < 0:
            raise ValueError(f"slope_bank_rank must be >= 0, got {slope_bank_rank}")
        if slope_bank_rank > 0 and backend != "torch":
            raise ValueError("PairwiseLinear slope bank currently supports backend='torch' only")
        if slope_bank_rank > 0 and not use_min_margin_ste:
            raise ValueError("PairwiseLinear slope bank currently supports min-margin STE only")
        if slope_bank_atom_init_std < 0:
            raise ValueError(f"slope_bank_atom_init_std must be >= 0, got {slope_bank_atom_init_std}")
        if slope_bank_coeff_init_std < 0:
            raise ValueError(f"slope_bank_coeff_init_std must be >= 0, got {slope_bank_coeff_init_std}")
        surrogate_gradient(torch.zeros((), dtype=torch.float32), surrogate)

        max_groups = (tables + max_group_size - 1) // max_group_size
        scale_terms = max_groups if accumulation == "two_bank_max" else tables
        output_scale = 1.0 / math.sqrt(scale_terms) if use_output_scaling else 1.0
        super().__init__(in_features, out_features, backend=backend, output_scale=output_scale)

        self.tables = tables
        self.comparisons = comparisons
        self.table_size = 1 << comparisons
        self.use_min_margin_ste = use_min_margin_ste
        self.fixed_zero_threshold = fixed_zero_threshold
        self.surrogate = surrogate
        self.cpu_lut_dtype = cpu_lut_dtype
        self.accumulation = accumulation
        self.max_group_size = max_group_size
        self.max_groups = max_groups
        self.slope_bank_rank = slope_bank_rank
        self.slope_bank_atom_init_std = slope_bank_atom_init_std
        self.slope_bank_coeff_init_std = slope_bank_coeff_init_std
        self._zig_lut_f16_cache: Tensor | None = None
        self._zig_lut_f16_cache_version = -1

        torch.manual_seed(seed)
        anchors = torch.zeros(tables, comparisons, 2, dtype=torch.long)
        for table_idx in range(tables):
            for comp_idx in range(comparisons):
                a = torch.randint(0, in_features, (1,)).item()
                b = torch.randint(0, in_features, (1,)).item()
                while a == b:
                    b = torch.randint(0, in_features, (1,)).item()
                anchors[table_idx, comp_idx, 0] = a
                anchors[table_idx, comp_idx, 1] = b
        self.register_buffer("anchors", anchors)
        thresholds = torch.zeros(tables, comparisons)
        if fixed_zero_threshold:
            self.register_buffer("thresholds", thresholds)
        else:
            self.thresholds = nn.Parameter(thresholds)
        if accumulation == "two_bank_max":
            if lut_init_std == 0.0:
                self.lut_pos = nn.Parameter(torch.zeros(tables, self.table_size, out_features))
                self.lut_neg = nn.Parameter(torch.zeros(tables, self.table_size, out_features))
            else:
                self.lut_pos = nn.Parameter(torch.randn(tables, self.table_size, out_features) * lut_init_std)
                self.lut_neg = nn.Parameter(torch.randn(tables, self.table_size, out_features) * lut_init_std)
        elif lut_init_std == 0.0:
            self.lut = nn.Parameter(torch.zeros(tables, self.table_size, out_features))
        else:
            self.lut = nn.Parameter(torch.randn(tables, self.table_size, out_features) * lut_init_std)
        if slope_bank_rank > 0:
            self.slope_u = nn.Parameter(torch.randn(slope_bank_rank, in_features) * slope_bank_atom_init_std)
            self.slope_v = nn.Parameter(torch.randn(slope_bank_rank, out_features) * slope_bank_atom_init_std)
            if slope_bank_coeff_init_std == 0.0:
                self.slope_coeff = nn.Parameter(torch.zeros(tables, self.table_size, slope_bank_rank))
            else:
                self.slope_coeff = nn.Parameter(torch.randn(tables, self.table_size, slope_bank_rank) * slope_bank_coeff_init_std)
        self.register_buffer("powers", 2 ** torch.arange(comparisons, dtype=torch.long))

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, tables={self.tables}, "
            f"comparisons={self.comparisons}, backend={self.backend!r}, use_min_margin_ste={self.use_min_margin_ste}, "
            f"fixed_zero_threshold={self.fixed_zero_threshold}, surrogate={self.surrogate!r}, cpu_lut_dtype={self.cpu_lut_dtype!r}, "
            f"accumulation={self.accumulation!r}, max_group_size={self.max_group_size}, slope_bank_rank={self.slope_bank_rank}"
        )

    def _project_input(self, x: Tensor, compute_dtype: torch.dtype) -> Tensor:
        if self.backend == "zig":
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

    def _select_rows(self, table_idx: int, indices: Tensor, compute_dtype: torch.dtype) -> Tensor:
        batch, seq = indices.shape
        values = self.lut[table_idx].to(compute_dtype).index_select(0, indices.reshape(-1))
        return values.view(batch, seq, self.out_features)

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

    def _lookup_sum(self, indices: Tensor, compute_dtype: torch.dtype) -> Tensor:
        return self._lookup_chunked(indices, compute_dtype=compute_dtype)

    def _lookup_slope_bank(self, latent: Tensor, indices: Tensor, compute_dtype: torch.dtype) -> Tensor:
        batch, seq, route_count = indices.shape
        item_count = batch * seq
        indices_flat = indices.reshape(item_count, route_count)
        coeff_table = self.slope_coeff.to(dtype=compute_dtype, device=indices.device).reshape(
            route_count * self.table_size,
            self.slope_bank_rank,
        )
        route_chunk = self._route_chunk_size(
            item_count=item_count,
            payload_width=self.slope_bank_rank,
            compute_dtype=compute_dtype,
            route_count=route_count,
        )
        coeff_sum = torch.zeros(item_count, self.slope_bank_rank, device=indices.device, dtype=compute_dtype)

        for route_start in range(0, route_count, route_chunk):
            route_stop = min(route_start + route_chunk, route_count)
            route_offsets = (torch.arange(route_start, route_stop, device=indices.device) * self.table_size).view(1, -1)
            linear_idx = (indices_flat[:, route_start:route_stop] + route_offsets).reshape(-1)
            selected = coeff_table.index_select(0, linear_idx).view(item_count, route_stop - route_start, self.slope_bank_rank)
            coeff_sum = coeff_sum + selected.sum(dim=1)

        latent_flat = latent.reshape(item_count, self.in_features).to(dtype=compute_dtype)
        atom_scalar = latent_flat.matmul(self.slope_u.to(dtype=compute_dtype, device=indices.device).t())
        weighted = coeff_sum * atom_scalar
        out = weighted.matmul(self.slope_v.to(dtype=compute_dtype, device=indices.device))
        return out.view(batch, seq, self.out_features)

    def _lookup_two_bank_max(self, indices: Tensor, compute_dtype: torch.dtype) -> Tensor:
        batch, seq, route_count = indices.shape
        item_count = batch * seq
        indices_flat = indices.reshape(item_count, route_count)
        pos_table = self.lut_pos.to(dtype=compute_dtype, device=indices.device).reshape(
            route_count * self.table_size,
            self.out_features,
        )
        neg_table = self.lut_neg.to(dtype=compute_dtype, device=indices.device).reshape(
            route_count * self.table_size,
            self.out_features,
        )
        output = torch.zeros(item_count, self.out_features, device=indices.device, dtype=compute_dtype)

        for route_start in range(0, route_count, self.max_group_size):
            route_stop = min(route_start + self.max_group_size, route_count)
            route_offsets = (torch.arange(route_start, route_stop, device=indices.device) * self.table_size).view(1, -1)
            linear_idx = (indices_flat[:, route_start:route_stop] + route_offsets).reshape(-1)
            group_width = route_stop - route_start
            pos = pos_table.index_select(0, linear_idx).view(item_count, group_width, self.out_features)
            neg = neg_table.index_select(0, linear_idx).view(item_count, group_width, self.out_features)
            output = output + pos.max(dim=1).values - neg.max(dim=1).values

        return output.view(batch, seq, self.out_features)

    def _compute_indices(self, latent: Tensor) -> tuple[Tensor, Tensor]:
        batch, seq, _ = latent.shape
        anchor_a = self.anchors[:, :, 0].flatten()
        anchor_b = self.anchors[:, :, 1].flatten()
        x_a = latent[..., anchor_a].view(batch, seq, self.tables, self.comparisons)
        x_b = latent[..., anchor_b].view(batch, seq, self.tables, self.comparisons)
        margins = x_a - x_b - self.thresholds.to(dtype=latent.dtype, device=latent.device)
        indices = (((margins > 0).to(torch.long)) * self.powers.view(1, 1, 1, -1)).sum(dim=-1)
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

    def _slope_bank_min_margin_ste(self, latent: Tensor, indices: Tensor, margins: Tensor) -> Tensor:
        r_mins = margins.abs().argmin(dim=-1)
        u_mins = margins.gather(dim=-1, index=r_mins.unsqueeze(-1)).squeeze(-1)
        neighbor_indices = indices ^ (2**r_mins).long()
        ste_delta = ste_heaviside(u_mins, self.surrogate) - (u_mins > 0).to(u_mins.dtype)

        batch, seq, route_count = indices.shape
        item_count = batch * seq
        current_flat = indices.reshape(item_count, route_count)
        neighbor_flat = neighbor_indices.reshape(item_count, route_count)
        ste_flat = ste_delta.reshape(item_count, route_count, 1).float()
        coeff_table = self.slope_coeff.to(dtype=torch.float32, device=indices.device).reshape(
            route_count * self.table_size,
            self.slope_bank_rank,
        )
        atom_scalar = latent.reshape(item_count, self.in_features).float().matmul(
            self.slope_u.to(dtype=torch.float32, device=indices.device).t()
        )
        route_chunk = self._route_chunk_size(
            item_count=item_count,
            payload_width=self.slope_bank_rank,
            compute_dtype=torch.float32,
            route_count=route_count,
        )
        coeff_delta = torch.zeros(item_count, self.slope_bank_rank, device=indices.device, dtype=torch.float32)

        for route_start in range(0, route_count, route_chunk):
            route_stop = min(route_start + route_chunk, route_count)
            route_offsets = (torch.arange(route_start, route_stop, device=indices.device) * self.table_size).view(1, -1)
            current_idx = (current_flat[:, route_start:route_stop] + route_offsets).reshape(-1)
            neighbor_idx = (neighbor_flat[:, route_start:route_stop] + route_offsets).reshape(-1)
            current = coeff_table.index_select(0, current_idx).view(item_count, route_stop - route_start, self.slope_bank_rank)
            neighbor = coeff_table.index_select(0, neighbor_idx).view(item_count, route_stop - route_start, self.slope_bank_rank)
            coeff_delta = coeff_delta + (ste_flat[:, route_start:route_stop] * (neighbor - current)).sum(dim=1)

        out = (coeff_delta * atom_scalar).matmul(self.slope_v.to(dtype=torch.float32, device=indices.device))
        return out.view(batch, seq, self.out_features)

    def _full_ste(self, indices: Tensor, margins: Tensor) -> Tensor:
        batch, seq, route_count = indices.shape
        item_count = batch * seq
        current_flat = indices.reshape(item_count, route_count)
        neighbor_flat = current_flat.unsqueeze(-1) ^ self.powers.view(1, 1, -1)
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
            neighbor_offsets = route_offsets.unsqueeze(-1)
            neighbor_idx = (neighbor_flat[:, route_start:route_stop] + neighbor_offsets).reshape(-1)
            neighbor = lut_table.index_select(0, neighbor_idx).view(
                item_count,
                route_stop - route_start,
                self.comparisons,
                self.out_features,
            )
            corr = corr + (ste_flat[:, route_start:route_stop] * (neighbor - current)).sum(dim=(1, 2))

        return corr.view(batch, seq, self.out_features)

    def _two_bank_min_margin_ste(self, indices: Tensor, margins: Tensor) -> Tensor:
        r_mins = margins.abs().argmin(dim=-1)
        u_mins = margins.gather(dim=-1, index=r_mins.unsqueeze(-1)).squeeze(-1)
        neighbor_indices = indices ^ (2**r_mins).long()
        ste_delta = ste_heaviside(u_mins, self.surrogate) - (u_mins > 0).to(u_mins.dtype)

        batch, seq, route_count = indices.shape
        item_count = batch * seq
        current_flat = indices.reshape(item_count, route_count)
        neighbor_flat = neighbor_indices.reshape(item_count, route_count)
        ste_flat = ste_delta.reshape(item_count, route_count, 1).float()
        pos_table = self.lut_pos.to(dtype=torch.float32, device=indices.device).reshape(route_count * self.table_size, self.out_features)
        neg_table = self.lut_neg.to(dtype=torch.float32, device=indices.device).reshape(route_count * self.table_size, self.out_features)
        corr = torch.zeros(item_count, self.out_features, device=indices.device, dtype=torch.float32)

        for route_start in range(0, route_count, self.max_group_size):
            route_stop = min(route_start + self.max_group_size, route_count)
            group_width = route_stop - route_start
            route_offsets = (torch.arange(route_start, route_stop, device=indices.device) * self.table_size).view(1, -1)
            current_idx = (current_flat[:, route_start:route_stop] + route_offsets).reshape(-1)
            neighbor_idx = (neighbor_flat[:, route_start:route_stop] + route_offsets).reshape(-1)
            pos_current = pos_table.index_select(0, current_idx).view(item_count, group_width, self.out_features)
            pos_neighbor = pos_table.index_select(0, neighbor_idx).view(item_count, group_width, self.out_features)
            neg_current = neg_table.index_select(0, current_idx).view(item_count, group_width, self.out_features)
            neg_neighbor = neg_table.index_select(0, neighbor_idx).view(item_count, group_width, self.out_features)

            def bank_delta(current: Tensor, neighbor: Tensor) -> Tensor:
                base = current.max(dim=1).values
                if group_width == 1:
                    alt = neighbor
                else:
                    mask = torch.eye(group_width, device=indices.device, dtype=torch.bool).view(1, group_width, group_width, 1)
                    expanded = current.unsqueeze(1).expand(-1, group_width, -1, -1)
                    other = expanded.masked_fill(mask, -torch.inf).max(dim=2).values
                    alt = torch.maximum(other, neighbor)
                return alt - base.unsqueeze(1)

            pos_delta = bank_delta(pos_current, pos_neighbor)
            neg_delta = bank_delta(neg_current, neg_neighbor)
            corr = corr + (ste_flat[:, route_start:route_stop] * (pos_delta - neg_delta)).sum(dim=1)

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
        if self.backend == "zig":
            if training:
                raise RuntimeError("PairwiseLinear backend='zig' is inference-only; call .eval() or use backend='torch' for training")
            from ..backends import pairwise_zig_forward

            output = pairwise_zig_forward(
                latent.contiguous(),
                self.anchors.to(device="cpu", dtype=torch.long),
                self.thresholds.detach().to(device="cpu", dtype=torch.float32),
                self._zig_lut_for_inference(),
                lut_dtype=self.cpu_lut_dtype,
            )
            empty_indices = torch.empty((*latent.shape[:2], 0), device=latent.device, dtype=torch.long)
            empty_margins = torch.empty((*latent.shape[:2], 0), device=latent.device, dtype=latent.dtype)
            return output, empty_indices, empty_margins

        if self.backend == "tilelang":
            if not latent.is_cuda:
                raise ValueError("PairwiseLinear backend='tilelang' requires CUDA input tensors")
            if compute_dtype != torch.float32:
                raise TypeError(f"PairwiseLinear backend='tilelang' requires float32 compute dtype, got {compute_dtype}")
            from ..backends import pairwise_tilelang

            return pairwise_tilelang(
                latent,
                self.anchors.to(device=latent.device),
                self.thresholds.to(dtype=compute_dtype, device=latent.device),
                self.lut.to(dtype=compute_dtype, device=latent.device),
                use_min_margin_ste=self.use_min_margin_ste,
                surrogate=self.surrogate,
            )

        indices, margins = self._compute_indices(latent)
        if self.accumulation == "two_bank_max":
            output = self._lookup_two_bank_max(indices, compute_dtype)
            if self.slope_bank_rank > 0:
                output = output + self._lookup_slope_bank(latent, indices, compute_dtype)
            if training and (latent.requires_grad or self.thresholds.requires_grad):
                output = output + self._two_bank_min_margin_ste(indices, margins).to(output.dtype)
                if self.slope_bank_rank > 0:
                    output = output + self._slope_bank_min_margin_ste(latent, indices, margins).to(output.dtype)
            return output, indices, margins

        output = self._lookup_sum(indices, compute_dtype)
        if self.slope_bank_rank > 0:
            output = output + self._lookup_slope_bank(latent, indices, compute_dtype)
        if training and (latent.requires_grad or self.thresholds.requires_grad):
            ste_corr = self._min_margin_ste(indices, margins) if self.use_min_margin_ste else self._full_ste(indices, margins)
            output = output + ste_corr.to(output.dtype)
            if self.slope_bank_rank > 0:
                output = output + self._slope_bank_min_margin_ste(latent, indices, margins).to(output.dtype)
        return output, indices, margins


class PairwiseTableMixLinear(PairwiseLinear):
    """Plain full-output Pairwise LUT with learned table-space aggregation.

    Each route still emits a full output vector:

        Y_t = LUT_t[j_t] in R^{d_out}.

    The requested table-space mixer is linear over the table dimension and is
    followed by the usual table sum.  Algebraically,

        sum_s (C Y)_s = sum_t column_sum(C)_t Y_t,

    so these variants are function-class equivalent to learned table gates.
    The implementation computes that equivalent weighted sum directly.  This
    keeps the experiment faithful while avoiding a large and pointless
    materialized [batch, tables, d_out] butterfly in every layer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        tables: int = 16,
        comparisons: int = 4,
        table_mix: Literal["none", "random_scatter", "diag", "butterfly", "lowrank", "dense"] = "diag",
        table_mix_rank: int = 4,
        table_mix_init_std: float = 0.02,
        backend: Backend = "torch",
        seed: int = 0,
        lut_init_std: float = 0.0,
        use_min_margin_ste: bool = True,
        use_output_scaling: bool = True,
        fixed_zero_threshold: bool = False,
        surrogate: str = "fast_sigmoid_odd",
        cpu_lut_dtype: Literal["f32", "f16"] = "f32",
        accumulation: Literal["sum", "two_bank_max"] = "sum",
        max_group_size: int = 4,
        slope_bank_rank: int = 0,
        slope_bank_atom_init_std: float = 0.02,
        slope_bank_coeff_init_std: float = 0.0,
    ) -> None:
        if table_mix == "random_scatter":
            table_mix = "none"
        if table_mix not in {"none", "diag", "butterfly", "lowrank", "dense"}:
            raise ValueError(f"unsupported table_mix {table_mix!r}")
        if backend != "torch":
            raise ValueError(f"PairwiseTableMixLinear currently supports backend='torch' only, got {backend!r}")
        if accumulation != "sum":
            raise ValueError("PairwiseTableMixLinear only supports accumulation='sum'")
        if slope_bank_rank != 0:
            raise ValueError("PairwiseTableMixLinear does not combine table mixing with slope_bank_rank > 0")
        if not use_min_margin_ste:
            raise ValueError("PairwiseTableMixLinear currently supports min-margin STE only")
        if table_mix_rank < 1:
            raise ValueError(f"table_mix_rank must be >= 1, got {table_mix_rank}")
        if table_mix_init_std < 0:
            raise ValueError(f"table_mix_init_std must be >= 0, got {table_mix_init_std}")
        super().__init__(
            in_features,
            out_features,
            tables=tables,
            comparisons=comparisons,
            backend=backend,
            seed=seed,
            lut_init_std=lut_init_std,
            use_min_margin_ste=use_min_margin_ste,
            use_output_scaling=use_output_scaling,
            fixed_zero_threshold=fixed_zero_threshold,
            surrogate=surrogate,
            cpu_lut_dtype=cpu_lut_dtype,
            accumulation=accumulation,
            max_group_size=max_group_size,
            slope_bank_rank=slope_bank_rank,
            slope_bank_atom_init_std=slope_bank_atom_init_std,
            slope_bank_coeff_init_std=slope_bank_coeff_init_std,
        )
        self.table_mix = table_mix
        self.table_mix_rank = int(table_mix_rank)
        self.table_mix_init_std = float(table_mix_init_std)

        gen = torch.Generator(device="cpu").manual_seed(seed + 70001)
        if table_mix == "diag":
            self.table_mix_diag = nn.Parameter(torch.zeros(self.tables))
        elif table_mix == "butterfly":
            self.padded_tables = _next_power_of_two(self.tables)
            self.table_mix_stages = int(math.log2(self.padded_tables))
            delta = torch.randn(
                self.table_mix_stages,
                self.padded_tables // 2,
                2,
                2,
                generator=gen,
            ) * table_mix_init_std
            self.table_mix_delta = nn.Parameter(delta)
            eye = torch.eye(2).view(1, 1, 2, 2).expand(
                self.table_mix_stages,
                self.padded_tables // 2,
                2,
                2,
            ).clone()
            self.register_buffer("table_mix_eye", eye)
            self.register_buffer("table_mix_basis_eye", torch.eye(self.tables).unsqueeze(-1))
        elif table_mix == "lowrank":
            self.table_mix_down = nn.Parameter(torch.randn(self.tables, table_mix_rank, generator=gen) / math.sqrt(self.tables))
            self.table_mix_up = nn.Parameter(torch.zeros(table_mix_rank, self.tables))
            self.table_mix_scale = 1.0 / math.sqrt(table_mix_rank)
        elif table_mix == "dense":
            self.table_mix_delta = nn.Parameter(torch.zeros(self.tables, self.tables))
            self.register_buffer("table_mix_eye", torch.eye(self.tables))

    def extra_repr(self) -> str:
        return (
            f"{super().extra_repr()}, table_mix={self.table_mix!r}, table_mix_rank={self.table_mix_rank}, "
            f"table_mix_init_std={self.table_mix_init_std}"
        )

    def _butterfly_mix_payload(self, payload: Tensor) -> Tensor:
        y = payload.transpose(1, 2)
        pad = self.padded_tables - self.tables
        y = F.pad(y, (0, pad)) if pad > 0 else y
        batch_shape = y.shape[:-1]
        for stage in range(self.table_mix_stages):
            stride = 1 << stage
            pairs = y.reshape(*batch_shape, -1, 2, stride).transpose(-2, -1)
            flat_pairs = pairs.reshape(*batch_shape, self.padded_tables // 2, 2)
            matrix = (self.table_mix_eye[stage] + self.table_mix_delta[stage]).to(device=payload.device, dtype=payload.dtype)
            mixed = torch.einsum("...pi,pio->...po", flat_pairs, matrix)
            y = mixed.reshape(*batch_shape, -1, stride, 2).transpose(-2, -1).reshape(*batch_shape, self.padded_tables)
        return y[..., : self.tables].transpose(1, 2)

    def _table_weights(self, dtype: torch.dtype, device: torch.device) -> Tensor:
        if self.table_mix == "none":
            return torch.ones(self.tables, device=device, dtype=dtype)
        if self.table_mix == "diag":
            return 1.0 + self.table_mix_diag.to(device=device, dtype=dtype)
        if self.table_mix == "lowrank":
            down = self.table_mix_down.to(device=device, dtype=dtype)
            up = self.table_mix_up.to(device=device, dtype=dtype)
            return 1.0 + self.table_mix_scale * down.matmul(up.sum(dim=1))
        if self.table_mix == "dense":
            matrix = (self.table_mix_eye + self.table_mix_delta).to(device=device, dtype=dtype)
            return matrix.sum(dim=0)
        if self.table_mix == "butterfly":
            basis = self.table_mix_basis_eye.to(device=device, dtype=dtype)
            mixed = self._butterfly_mix_payload(basis)
            return mixed.sum(dim=1).squeeze(-1)
        raise AssertionError(f"unreachable table_mix {self.table_mix!r}")

    def _lookup_weighted_sum(self, indices: Tensor, compute_dtype: torch.dtype, weights: Tensor) -> Tensor:
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
            selected = selected * weights[route_start:route_stop].view(1, -1, 1)
            output = output + selected.sum(dim=1)

        return output.view(batch, seq, self.out_features)

    def _weighted_min_margin_ste(self, indices: Tensor, margins: Tensor, weights: Tensor) -> Tensor:
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
        weights_f32 = weights.to(dtype=torch.float32, device=indices.device)

        for route_start in range(0, route_count, route_chunk):
            route_stop = min(route_start + route_chunk, route_count)
            route_offsets = (torch.arange(route_start, route_stop, device=indices.device) * self.table_size).view(1, -1)
            current_idx = (current_flat[:, route_start:route_stop] + route_offsets).reshape(-1)
            neighbor_idx = (neighbor_flat[:, route_start:route_stop] + route_offsets).reshape(-1)
            current = lut_table.index_select(0, current_idx).view(item_count, route_stop - route_start, self.out_features)
            neighbor = lut_table.index_select(0, neighbor_idx).view(item_count, route_stop - route_start, self.out_features)
            route_weights = weights_f32[route_start:route_stop].view(1, -1, 1)
            corr = corr + (ste_flat[:, route_start:route_stop] * route_weights * (neighbor - current)).sum(dim=1)

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
        weights = self._table_weights(compute_dtype, indices.device)
        output = self._lookup_weighted_sum(indices, compute_dtype, weights)
        if training and (latent.requires_grad or self.thresholds.requires_grad):
            output = output + self._weighted_min_margin_ste(indices, margins, weights).to(output.dtype)
        return output, indices, margins


class PairwiseFoldingLinear(PairwiseLinear):
    """Pairwise LUT with chamber-conditioned diagonal/block-sign affine folding.

    Plain PairwiseLinear is piecewise constant before the residual path: each
    chamber selects a translation vector. This variant adds a cheap
    chamber-affine term,

        y += alpha * sum_t D[t, route_t(x)] F x,

    where F is a fixed fold/repeat map from input features to output features
    and D is a route-conditioned block-diagonal sign map. The forward path is
    still comparator routing plus table lookup plus elementwise multiply/add;
    it does not introduce dense matrix multiplication.
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
        accumulation: Literal["sum", "two_bank_max"] = "sum",
        max_group_size: int = 4,
        slope_bank_rank: int = 0,
        slope_bank_atom_init_std: float = 0.02,
        slope_bank_coeff_init_std: float = 0.0,
        fold_alpha: float = 0.1,
        fold_block_size: int = 8,
        fold_sign_init_std: float = 0.02,
        fold_mode: Literal["sign", "perm_bank"] = "sign",
        fold_perm_banks: int = 8,
        hard_fold_sign: bool = True,
    ) -> None:
        if backend != "torch":
            raise ValueError(f"PairwiseFoldingLinear currently supports backend='torch' only, got {backend!r}")
        if fold_block_size < 1:
            raise ValueError(f"fold_block_size must be >= 1, got {fold_block_size}")
        if fold_sign_init_std < 0:
            raise ValueError(f"fold_sign_init_std must be >= 0, got {fold_sign_init_std}")
        if fold_mode not in {"sign", "perm_bank"}:
            raise ValueError(f"fold_mode must be 'sign' or 'perm_bank', got {fold_mode!r}")
        if fold_perm_banks < 1:
            raise ValueError(f"fold_perm_banks must be >= 1, got {fold_perm_banks}")
        super().__init__(
            in_features,
            out_features,
            tables=tables,
            comparisons=comparisons,
            backend=backend,
            seed=seed,
            lut_init_std=lut_init_std,
            use_min_margin_ste=use_min_margin_ste,
            use_output_scaling=use_output_scaling,
            fixed_zero_threshold=fixed_zero_threshold,
            surrogate=surrogate,
            cpu_lut_dtype=cpu_lut_dtype,
            accumulation=accumulation,
            max_group_size=max_group_size,
            slope_bank_rank=slope_bank_rank,
            slope_bank_atom_init_std=slope_bank_atom_init_std,
            slope_bank_coeff_init_std=slope_bank_coeff_init_std,
        )
        self.fold_block_size = int(fold_block_size)
        self.fold_blocks = (out_features + fold_block_size - 1) // fold_block_size
        self.fold_sign_init_std = float(fold_sign_init_std)
        self.fold_mode = fold_mode
        self.fold_perm_banks = int(fold_perm_banks)
        self.hard_fold_sign = bool(hard_fold_sign)
        self.fold_alpha = nn.Parameter(torch.tensor(float(fold_alpha), dtype=torch.float32))
        gen = torch.Generator(device="cpu").manual_seed(seed + 7919)
        self.fold_sign_logits = nn.Parameter(
            torch.randn(tables, self.table_size, self.fold_blocks, generator=gen) * fold_sign_init_std
        )
        if fold_mode == "perm_bank":
            perm_bank = torch.stack([torch.randperm(out_features, generator=gen) for _ in range(fold_perm_banks)], dim=0)
            perm_ids = torch.randint(0, fold_perm_banks, (tables, self.table_size), generator=gen, dtype=torch.long)
        else:
            perm_bank = torch.empty(0, out_features, dtype=torch.long)
            perm_ids = torch.empty(0, dtype=torch.long)
        self.register_buffer("fold_perm_bank", perm_bank)
        self.register_buffer("fold_perm_ids", perm_ids)
        if in_features < out_features:
            fold_input_index = torch.arange(out_features, dtype=torch.long).remainder(in_features)
        else:
            fold_input_index = torch.empty(0, dtype=torch.long)
        self.register_buffer("fold_input_index", fold_input_index)

    def extra_repr(self) -> str:
        return (
            f"{super().extra_repr()}, fold_alpha={float(self.fold_alpha.detach()):.4f}, "
            f"fold_block_size={self.fold_block_size}, fold_mode={self.fold_mode!r}, "
            f"fold_perm_banks={self.fold_perm_banks}, hard_fold_sign={self.hard_fold_sign}"
        )

    def _fold_sign_values(self, dtype: torch.dtype, device: torch.device) -> Tensor:
        logits = self.fold_sign_logits.to(dtype=dtype, device=device)
        if not self.hard_fold_sign:
            return torch.tanh(logits)
        soft = torch.tanh(logits)
        hard = torch.where(logits >= 0, torch.ones_like(logits), -torch.ones_like(logits))
        return hard.detach() - soft.detach() + soft

    def _expand_fold_blocks(self, blocks: Tensor) -> Tensor:
        if self.fold_block_size == 1:
            return blocks[..., : self.out_features]
        return blocks.repeat_interleave(self.fold_block_size, dim=-1)[..., : self.out_features]

    def _fold_latent_to_output(self, latent: Tensor, dtype: torch.dtype) -> Tensor:
        x = latent.to(dtype=dtype)
        if self.in_features == self.out_features:
            return x
        if self.in_features > self.out_features:
            groups = (self.in_features + self.out_features - 1) // self.out_features
            padded = groups * self.out_features
            if padded != self.in_features:
                x = F.pad(x, (0, padded - self.in_features))
            y = x.view(*x.shape[:-1], groups, self.out_features).sum(dim=-2)
            return y / math.sqrt(groups)
        idx = self.fold_input_index.to(device=latent.device)
        repeats = (self.out_features + self.in_features - 1) // self.in_features
        return x.index_select(-1, idx) / math.sqrt(repeats)

    def _lookup_fold_signs(self, indices: Tensor, compute_dtype: torch.dtype) -> Tensor:
        batch, seq, route_count = indices.shape
        item_count = batch * seq
        indices_flat = indices.reshape(item_count, route_count)
        sign_table = self._fold_sign_values(compute_dtype, indices.device).reshape(route_count * self.table_size, self.fold_blocks)
        route_chunk = self._route_chunk_size(
            item_count=item_count,
            payload_width=self.fold_blocks,
            compute_dtype=compute_dtype,
            route_count=route_count,
        )
        sign_sum = torch.zeros(item_count, self.fold_blocks, device=indices.device, dtype=compute_dtype)

        for route_start in range(0, route_count, route_chunk):
            route_stop = min(route_start + route_chunk, route_count)
            route_offsets = (torch.arange(route_start, route_stop, device=indices.device) * self.table_size).view(1, -1)
            linear_idx = (indices_flat[:, route_start:route_stop] + route_offsets).reshape(-1)
            selected = sign_table.index_select(0, linear_idx).view(item_count, route_stop - route_start, self.fold_blocks)
            sign_sum = sign_sum + selected.sum(dim=1)

        return self._expand_fold_blocks(sign_sum).view(batch, seq, self.out_features)

    def _lookup_folding(self, latent: Tensor, indices: Tensor, compute_dtype: torch.dtype) -> Tensor:
        if self.fold_mode == "perm_bank":
            return self._lookup_permutation_folding(latent, indices, compute_dtype)
        folded = self._fold_latent_to_output(latent, compute_dtype)
        signs = self._lookup_fold_signs(indices, compute_dtype)
        alpha = self.fold_alpha.to(dtype=compute_dtype, device=indices.device)
        return alpha * folded * signs

    def _lookup_permutation_folding(self, latent: Tensor, indices: Tensor, compute_dtype: torch.dtype) -> Tensor:
        batch, seq, route_count = indices.shape
        item_count = batch * seq
        indices_flat = indices.reshape(item_count, route_count)
        folded_flat = self._fold_latent_to_output(latent, compute_dtype).reshape(item_count, self.out_features)
        sign_table = self._fold_sign_values(compute_dtype, indices.device).reshape(route_count * self.table_size, self.fold_blocks)
        perm_id_table = self.fold_perm_ids.to(device=indices.device).reshape(route_count * self.table_size)
        perm_bank = self.fold_perm_bank.to(device=indices.device)
        route_chunk = self._route_chunk_size(
            item_count=item_count,
            payload_width=self.fold_blocks,
            compute_dtype=compute_dtype,
            route_count=route_count,
        )
        output = torch.zeros(item_count, self.out_features, device=indices.device, dtype=compute_dtype)

        for route_start in range(0, route_count, route_chunk):
            route_stop = min(route_start + route_chunk, route_count)
            route_offsets = (torch.arange(route_start, route_stop, device=indices.device) * self.table_size).view(1, -1)
            linear_idx = (indices_flat[:, route_start:route_stop] + route_offsets).reshape(-1)
            width = route_stop - route_start
            signs = sign_table.index_select(0, linear_idx).view(item_count, width, self.fold_blocks)
            signs = self._expand_fold_blocks(signs)
            perm_ids = perm_id_table.index_select(0, linear_idx).view(item_count, width)
            chunk = torch.zeros(item_count, width, self.out_features, device=indices.device, dtype=compute_dtype)
            for bank_idx in range(self.fold_perm_banks):
                mask = (perm_ids == bank_idx).to(dtype=compute_dtype).unsqueeze(-1)
                permuted = folded_flat.index_select(-1, perm_bank[bank_idx]).unsqueeze(1)
                chunk = chunk + mask * signs * permuted
            output = output + chunk.sum(dim=1)

        alpha = self.fold_alpha.to(dtype=compute_dtype, device=indices.device)
        return (alpha * output).view(batch, seq, self.out_features)

    def _folding_min_margin_ste(self, latent: Tensor, indices: Tensor, margins: Tensor) -> Tensor:
        r_mins = margins.abs().argmin(dim=-1)
        u_mins = margins.gather(dim=-1, index=r_mins.unsqueeze(-1)).squeeze(-1)
        neighbor_indices = indices ^ (2**r_mins).long()
        ste_delta = ste_heaviside(u_mins, self.surrogate) - (u_mins > 0).to(u_mins.dtype)

        batch, seq, route_count = indices.shape
        item_count = batch * seq
        current_flat = indices.reshape(item_count, route_count)
        neighbor_flat = neighbor_indices.reshape(item_count, route_count)
        ste_flat = ste_delta.reshape(item_count, route_count, 1).float()
        sign_table = self._fold_sign_values(torch.float32, indices.device).reshape(route_count * self.table_size, self.fold_blocks)
        route_chunk = self._route_chunk_size(
            item_count=item_count,
            payload_width=self.fold_blocks,
            compute_dtype=torch.float32,
            route_count=route_count,
        )
        sign_delta = torch.zeros(item_count, self.fold_blocks, device=indices.device, dtype=torch.float32)

        for route_start in range(0, route_count, route_chunk):
            route_stop = min(route_start + route_chunk, route_count)
            route_offsets = (torch.arange(route_start, route_stop, device=indices.device) * self.table_size).view(1, -1)
            current_idx = (current_flat[:, route_start:route_stop] + route_offsets).reshape(-1)
            neighbor_idx = (neighbor_flat[:, route_start:route_stop] + route_offsets).reshape(-1)
            current = sign_table.index_select(0, current_idx).view(item_count, route_stop - route_start, self.fold_blocks)
            neighbor = sign_table.index_select(0, neighbor_idx).view(item_count, route_stop - route_start, self.fold_blocks)
            sign_delta = sign_delta + (ste_flat[:, route_start:route_stop] * (neighbor - current)).sum(dim=1)

        expanded_delta = self._expand_fold_blocks(sign_delta)
        folded = self._fold_latent_to_output(latent, torch.float32).reshape(item_count, self.out_features)
        alpha = self.fold_alpha.to(dtype=torch.float32, device=indices.device)
        return (alpha * folded * expanded_delta).view(batch, seq, self.out_features)

    def _route_output(
        self,
        latent: Tensor,
        *,
        input_device: torch.device,
        compute_dtype: torch.dtype,
        training: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        output, indices, margins = super()._route_output(
            latent,
            input_device=input_device,
            compute_dtype=compute_dtype,
            training=training,
        )
        output = output + self._lookup_folding(latent, indices, compute_dtype)
        if self.fold_mode == "sign" and training and self.use_min_margin_ste and (latent.requires_grad or self.thresholds.requires_grad):
            output = output + self._folding_min_margin_ste(latent, indices, margins).to(output.dtype)
        return output, indices, margins


class PairwiseAffineTwoBankLinear(PairwiseFoldingLinear):
    """Two-bank max/min Pairwise LUT whose bank entries are cheap affine atoms.

    Unlike PairwiseLinear(accumulation='two_bank_max'), which takes max over
    constants, this class computes max over route-conditioned diagonal affine
    atoms:

        sum_g max_{t in g} (b^+_{t,j_t} + alpha s^+_{t,j_t} * F(x))
            - max_{t in g} (b^-_{t,j_t} + alpha s^-_{t,j_t} * F(x)).

    This is a hardware-cheap tropical rational / DC CPWL map: the added
    affine part is only lookup + elementwise multiply/add.
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
        max_group_size: int = 4,
        fold_alpha: float = 0.1,
        fold_block_size: int = 8,
        fold_sign_init_std: float = 0.02,
        hard_fold_sign: bool = True,
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            tables=tables,
            comparisons=comparisons,
            backend=backend,
            seed=seed,
            lut_init_std=lut_init_std,
            use_min_margin_ste=use_min_margin_ste,
            use_output_scaling=use_output_scaling,
            fixed_zero_threshold=fixed_zero_threshold,
            surrogate=surrogate,
            cpu_lut_dtype=cpu_lut_dtype,
            accumulation="two_bank_max",
            max_group_size=max_group_size,
            slope_bank_rank=0,
            fold_alpha=fold_alpha,
            fold_block_size=fold_block_size,
            fold_sign_init_std=fold_sign_init_std,
            fold_mode="sign",
            hard_fold_sign=hard_fold_sign,
        )
        gen = torch.Generator(device="cpu").manual_seed(seed + 104729)
        self.fold_sign_logits_neg = nn.Parameter(
            torch.randn(tables, self.table_size, self.fold_blocks, generator=gen) * fold_sign_init_std
        )

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, affine_two_bank=True"

    def _fold_sign_values_neg(self, dtype: torch.dtype, device: torch.device) -> Tensor:
        logits = self.fold_sign_logits_neg.to(dtype=dtype, device=device)
        if not self.hard_fold_sign:
            return torch.tanh(logits)
        soft = torch.tanh(logits)
        hard = torch.where(logits >= 0, torch.ones_like(logits), -torch.ones_like(logits))
        return hard.detach() - soft.detach() + soft

    def _lookup_two_bank_affine(self, latent: Tensor, indices: Tensor, compute_dtype: torch.dtype) -> Tensor:
        batch, seq, route_count = indices.shape
        item_count = batch * seq
        indices_flat = indices.reshape(item_count, route_count)
        folded_flat = self._fold_latent_to_output(latent, compute_dtype).reshape(item_count, self.out_features)
        pos_table = self.lut_pos.to(dtype=compute_dtype, device=indices.device).reshape(
            route_count * self.table_size,
            self.out_features,
        )
        neg_table = self.lut_neg.to(dtype=compute_dtype, device=indices.device).reshape(
            route_count * self.table_size,
            self.out_features,
        )
        pos_sign_table = self._fold_sign_values(compute_dtype, indices.device).reshape(route_count * self.table_size, self.fold_blocks)
        neg_sign_table = self._fold_sign_values_neg(compute_dtype, indices.device).reshape(
            route_count * self.table_size,
            self.fold_blocks,
        )
        alpha = self.fold_alpha.to(dtype=compute_dtype, device=indices.device)
        folded = folded_flat.unsqueeze(1)
        output = torch.zeros(item_count, self.out_features, device=indices.device, dtype=compute_dtype)

        for route_start in range(0, route_count, self.max_group_size):
            route_stop = min(route_start + self.max_group_size, route_count)
            group_width = route_stop - route_start
            route_offsets = (torch.arange(route_start, route_stop, device=indices.device) * self.table_size).view(1, -1)
            linear_idx = (indices_flat[:, route_start:route_stop] + route_offsets).reshape(-1)
            pos = pos_table.index_select(0, linear_idx).view(item_count, group_width, self.out_features)
            neg = neg_table.index_select(0, linear_idx).view(item_count, group_width, self.out_features)
            pos_sign = pos_sign_table.index_select(0, linear_idx).view(item_count, group_width, self.fold_blocks)
            neg_sign = neg_sign_table.index_select(0, linear_idx).view(item_count, group_width, self.fold_blocks)
            pos = pos + alpha * self._expand_fold_blocks(pos_sign) * folded
            neg = neg + alpha * self._expand_fold_blocks(neg_sign) * folded
            output = output + pos.max(dim=1).values - neg.max(dim=1).values

        return output.view(batch, seq, self.out_features)

    def _route_output(
        self,
        latent: Tensor,
        *,
        input_device: torch.device,
        compute_dtype: torch.dtype,
        training: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        del input_device, training
        indices, margins = self._compute_indices(latent)
        return self._lookup_two_bank_affine(latent, indices, compute_dtype), indices, margins


class PairwiseDelayedHeadLinear(RoutedLinearBase):
    """Pairwise LUT that preserves per-table head lanes before accumulation.

    Instead of each table directly emitting a full output vector and summing
    immediately, each route emits a small head payload. The head lanes are
    route-conditioned affine atoms and are scattered into output coordinates
    only after the per-table identity has been preserved:

        h_{t,r} = b_{t,j_t,r} + alpha s_{t,j_t,r} * F(x)_{slot(t,r)}
        y = scatter_sum_{t,r -> slot(t,r)} h_{t,r}.

    This keeps the forward path in compare / lookup / elementwise / scatter-add
    form and avoids dense matrix multiplication.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        tables: int = 16,
        comparisons: int = 4,
        head_dim: int = 8,
        backend: Backend = "torch",
        seed: int = 0,
        lut_init_std: float = 0.0,
        use_output_scaling: bool = True,
        fixed_zero_threshold: bool = False,
        fold_alpha: float = 0.1,
        sign_init_std: float = 0.02,
    ) -> None:
        if tables < 1:
            raise ValueError(f"tables must be >= 1, got {tables}")
        if comparisons < 1:
            raise ValueError(f"comparisons must be >= 1, got {comparisons}")
        if head_dim < 1:
            raise ValueError(f"head_dim must be >= 1, got {head_dim}")
        if backend != "torch":
            raise ValueError(f"PairwiseDelayedHeadLinear currently supports backend='torch' only, got {backend!r}")
        if sign_init_std < 0:
            raise ValueError(f"sign_init_std must be >= 0, got {sign_init_std}")
        output_scale = 1.0 / math.sqrt(tables) if use_output_scaling else 1.0
        super().__init__(in_features, out_features, backend=backend, output_scale=output_scale)
        self.tables = int(tables)
        self.comparisons = int(comparisons)
        self.table_size = 1 << comparisons
        self.head_dim = int(head_dim)
        self.fixed_zero_threshold = bool(fixed_zero_threshold)
        self.fold_alpha = nn.Parameter(torch.tensor(float(fold_alpha), dtype=torch.float32))

        gen = torch.Generator(device="cpu").manual_seed(seed)
        anchors = torch.zeros(tables, comparisons, 2, dtype=torch.long)
        for table_idx in range(tables):
            for comp_idx in range(comparisons):
                a = torch.randint(0, in_features, (1,), generator=gen).item()
                b = torch.randint(0, in_features, (1,), generator=gen).item()
                while a == b:
                    b = torch.randint(0, in_features, (1,), generator=gen).item()
                anchors[table_idx, comp_idx, 0] = a
                anchors[table_idx, comp_idx, 1] = b
        self.register_buffer("anchors", anchors)
        thresholds = torch.zeros(tables, comparisons)
        if fixed_zero_threshold:
            self.register_buffer("thresholds", thresholds)
        else:
            self.thresholds = nn.Parameter(thresholds)
        if lut_init_std == 0.0:
            self.lut = nn.Parameter(torch.zeros(tables, self.table_size, head_dim))
        else:
            self.lut = nn.Parameter(torch.randn(tables, self.table_size, head_dim, generator=gen) * lut_init_std)
        self.sign_logits = nn.Parameter(torch.randn(tables, self.table_size, head_dim, generator=gen) * sign_init_std)
        slots = torch.randint(0, out_features, (tables, head_dim), generator=gen, dtype=torch.long)
        self.register_buffer("head_slots", slots)
        self.register_buffer("powers", 2 ** torch.arange(comparisons, dtype=torch.long))
        if in_features < out_features:
            resize_index = torch.arange(out_features, dtype=torch.long).remainder(in_features)
        else:
            resize_index = torch.empty(0, dtype=torch.long)
        self.register_buffer("resize_index", resize_index)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, tables={self.tables}, "
            f"comparisons={self.comparisons}, head_dim={self.head_dim}, fixed_zero_threshold={self.fixed_zero_threshold}"
        )

    def _project_input(self, x: Tensor, compute_dtype: torch.dtype) -> Tensor:
        return x.to(compute_dtype)

    def _resize_to_output(self, latent: Tensor) -> Tensor:
        if self.in_features == self.out_features:
            return latent
        if self.in_features > self.out_features:
            groups = (self.in_features + self.out_features - 1) // self.out_features
            padded = groups * self.out_features
            x = F.pad(latent, (0, padded - self.in_features)) if padded != self.in_features else latent
            return x.view(*x.shape[:-1], groups, self.out_features).sum(dim=-2) / math.sqrt(groups)
        repeats = (self.out_features + self.in_features - 1) // self.in_features
        return latent.index_select(-1, self.resize_index.to(device=latent.device)) / math.sqrt(repeats)

    def _compute_indices(self, latent: Tensor) -> tuple[Tensor, Tensor]:
        batch, seq, _ = latent.shape
        anchor_a = self.anchors[:, :, 0].flatten()
        anchor_b = self.anchors[:, :, 1].flatten()
        x_a = latent[..., anchor_a].view(batch, seq, self.tables, self.comparisons)
        x_b = latent[..., anchor_b].view(batch, seq, self.tables, self.comparisons)
        margins = x_a - x_b - self.thresholds.to(dtype=latent.dtype, device=latent.device)
        indices = (((margins > 0).to(torch.long)) * self.powers.view(1, 1, 1, -1)).sum(dim=-1)
        return indices, margins

    def _sign_values(self, dtype: torch.dtype, device: torch.device) -> Tensor:
        logits = self.sign_logits.to(dtype=dtype, device=device)
        soft = torch.tanh(logits)
        hard = torch.where(logits >= 0, torch.ones_like(logits), -torch.ones_like(logits))
        return hard.detach() - soft.detach() + soft

    def _route_output(
        self,
        latent: Tensor,
        *,
        input_device: torch.device,
        compute_dtype: torch.dtype,
        training: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        del input_device, training
        indices, margins = self._compute_indices(latent)
        batch, seq, route_count = indices.shape
        item_count = batch * seq
        indices_flat = indices.reshape(item_count, route_count)
        route_offsets = (torch.arange(route_count, device=indices.device) * self.table_size).view(1, route_count)
        linear_idx = (indices_flat + route_offsets).reshape(-1)
        lut_table = self.lut.to(dtype=compute_dtype, device=indices.device).reshape(route_count * self.table_size, self.head_dim)
        sign_table = self._sign_values(compute_dtype, indices.device).reshape(route_count * self.table_size, self.head_dim)
        payload = lut_table.index_select(0, linear_idx).view(item_count, route_count, self.head_dim)
        signs = sign_table.index_select(0, linear_idx).view(item_count, route_count, self.head_dim)
        slots = self.head_slots.to(device=indices.device).reshape(-1)
        folded = self._resize_to_output(latent.to(compute_dtype)).reshape(item_count, self.out_features)
        slot_values = folded.index_select(-1, slots).view(item_count, route_count, self.head_dim)
        alpha = self.fold_alpha.to(dtype=compute_dtype, device=indices.device)
        payload = payload + alpha * signs * slot_values
        output = torch.zeros(item_count, self.out_features, device=indices.device, dtype=compute_dtype)
        output.scatter_add_(1, slots.view(1, -1).expand(item_count, -1), payload.reshape(item_count, -1))
        return output.view(batch, seq, self.out_features), indices, margins


class PairwiseDelayedTableLinear(PairwiseDelayedHeadLinear):
    """Delayed-table Pairwise LUT with learned aggregate mixing over tables.

    This keeps each table's small payload alive until after lookup, then applies
    a cheap correspondence map across the table dimension before the final
    scatter-add:

        H[t, r] = LUT_t[j_t, r] + alpha * sign_t[j_t, r] * F(x)_{slot(t,r)}
        H' = C_table(H)
        y = scatter_sum_{t,r -> slot(t,r)} H'[t, r].

    The table mixer never performs a dense feature/output projection.  The
    dense option is only a small table-space upper bound, not a full output
    matrix multiply.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        tables: int = 16,
        comparisons: int = 4,
        head_dim: int = 8,
        table_mix: Literal["none", "random_scatter", "diag", "butterfly", "lowrank", "dense"] = "butterfly",
        table_mix_rank: int = 4,
        table_mix_init_std: float = 0.02,
        backend: Backend = "torch",
        seed: int = 0,
        lut_init_std: float = 0.0,
        use_output_scaling: bool = True,
        fixed_zero_threshold: bool = False,
        fold_alpha: float = 0.1,
        sign_init_std: float = 0.02,
    ) -> None:
        if table_mix == "random_scatter":
            table_mix = "none"
        if table_mix not in {"none", "diag", "butterfly", "lowrank", "dense"}:
            raise ValueError(f"unsupported table_mix {table_mix!r}")
        if table_mix_rank < 1:
            raise ValueError(f"table_mix_rank must be >= 1, got {table_mix_rank}")
        if table_mix_init_std < 0:
            raise ValueError(f"table_mix_init_std must be >= 0, got {table_mix_init_std}")
        super().__init__(
            in_features,
            out_features,
            tables=tables,
            comparisons=comparisons,
            head_dim=head_dim,
            backend=backend,
            seed=seed,
            lut_init_std=lut_init_std,
            use_output_scaling=use_output_scaling,
            fixed_zero_threshold=fixed_zero_threshold,
            fold_alpha=fold_alpha,
            sign_init_std=sign_init_std,
        )
        self.table_mix = table_mix
        self.table_mix_rank = int(table_mix_rank)
        self.table_mix_init_std = float(table_mix_init_std)

        gen = torch.Generator(device="cpu").manual_seed(seed + 30011)
        if table_mix == "diag":
            self.table_mix_diag = nn.Parameter(torch.zeros(self.tables))
        elif table_mix == "butterfly":
            self.padded_tables = _next_power_of_two(self.tables)
            self.table_mix_stages = int(math.log2(self.padded_tables))
            delta = torch.randn(
                self.table_mix_stages,
                self.padded_tables // 2,
                2,
                2,
                generator=gen,
            ) * table_mix_init_std
            self.table_mix_delta = nn.Parameter(delta)
            eye = torch.eye(2).view(1, 1, 2, 2).expand(
                self.table_mix_stages,
                self.padded_tables // 2,
                2,
                2,
            ).clone()
            self.register_buffer("table_mix_eye", eye)
        elif table_mix == "lowrank":
            self.table_mix_down = nn.Parameter(torch.randn(self.tables, table_mix_rank, generator=gen) / math.sqrt(self.tables))
            self.table_mix_up = nn.Parameter(torch.zeros(table_mix_rank, self.tables))
            self.table_mix_scale = 1.0 / math.sqrt(table_mix_rank)
        elif table_mix == "dense":
            self.table_mix_delta = nn.Parameter(torch.zeros(self.tables, self.tables))
            self.register_buffer("table_mix_eye", torch.eye(self.tables))

    def extra_repr(self) -> str:
        return (
            f"{super().extra_repr()}, table_mix={self.table_mix!r}, table_mix_rank={self.table_mix_rank}, "
            f"table_mix_init_std={self.table_mix_init_std}"
        )

    def _butterfly_table_mix(self, payload: Tensor) -> Tensor:
        if self.table_mix_stages == 0:
            return payload
        y = payload.transpose(1, 2)
        pad = self.padded_tables - self.tables
        y = F.pad(y, (0, pad)) if pad > 0 else y
        batch_shape = y.shape[:-1]
        for stage in range(self.table_mix_stages):
            stride = 1 << stage
            pairs = y.reshape(*batch_shape, -1, 2, stride).transpose(-2, -1)
            flat_pairs = pairs.reshape(*batch_shape, self.padded_tables // 2, 2)
            matrix = (self.table_mix_eye[stage] + self.table_mix_delta[stage]).to(device=payload.device, dtype=payload.dtype)
            mixed = torch.einsum("...pi,pio->...po", flat_pairs, matrix)
            y = mixed.reshape(*batch_shape, -1, stride, 2).transpose(-2, -1).reshape(*batch_shape, self.padded_tables)
        return y[..., : self.tables].transpose(1, 2)

    def _mix_payload(self, payload: Tensor) -> Tensor:
        if self.table_mix == "none":
            return payload
        if self.table_mix == "diag":
            gate = 1.0 + self.table_mix_diag.to(device=payload.device, dtype=payload.dtype)
            return payload * gate.view(1, self.tables, 1)
        if self.table_mix == "butterfly":
            return self._butterfly_table_mix(payload)
        if self.table_mix == "lowrank":
            down = self.table_mix_down.to(device=payload.device, dtype=payload.dtype)
            up = self.table_mix_up.to(device=payload.device, dtype=payload.dtype)
            hidden = torch.einsum("nth,tr->nrh", payload, down)
            residual = torch.einsum("nrh,rs->nsh", hidden, up)
            return payload + residual * self.table_mix_scale
        if self.table_mix == "dense":
            matrix = (self.table_mix_eye + self.table_mix_delta).to(device=payload.device, dtype=payload.dtype)
            return torch.einsum("st,nth->nsh", matrix, payload)
        raise AssertionError(f"unreachable table_mix {self.table_mix!r}")

    def _route_output(
        self,
        latent: Tensor,
        *,
        input_device: torch.device,
        compute_dtype: torch.dtype,
        training: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        del input_device, training
        indices, margins = self._compute_indices(latent)
        batch, seq, route_count = indices.shape
        item_count = batch * seq
        indices_flat = indices.reshape(item_count, route_count)
        route_offsets = (torch.arange(route_count, device=indices.device) * self.table_size).view(1, route_count)
        linear_idx = (indices_flat + route_offsets).reshape(-1)
        lut_table = self.lut.to(dtype=compute_dtype, device=indices.device).reshape(route_count * self.table_size, self.head_dim)
        sign_table = self._sign_values(compute_dtype, indices.device).reshape(route_count * self.table_size, self.head_dim)
        payload = lut_table.index_select(0, linear_idx).view(item_count, route_count, self.head_dim)
        signs = sign_table.index_select(0, linear_idx).view(item_count, route_count, self.head_dim)
        slots = self.head_slots.to(device=indices.device).reshape(-1)
        folded = self._resize_to_output(latent.to(compute_dtype)).reshape(item_count, self.out_features)
        slot_values = folded.index_select(-1, slots).view(item_count, route_count, self.head_dim)
        alpha = self.fold_alpha.to(dtype=compute_dtype, device=indices.device)
        payload = payload + alpha * signs * slot_values
        payload = self._mix_payload(payload)
        output = torch.zeros(item_count, self.out_features, device=indices.device, dtype=compute_dtype)
        output.scatter_add_(1, slots.view(1, -1).expand(item_count, -1), payload.reshape(item_count, -1))
        return output.view(batch, seq, self.out_features), indices, margins


class TropicalSawtoothLinear(RoutedLinearBase):
    """Coordinate-wise tropical sawtooth layer with fixed structured mix/unmix.

    The layer implements a cheap CPWL folding map:

        u = H resize(x)
        z_k = slope[k, bin(u_k)] * u_k + offset[k, bin(u_k)]
        y = H z

    where H is a fixed normalized Hadamard transform padded to a power of two.
    The bin slopes are alternating signs, so adjacent intervals are folded back
    over the same output range. No dense matrix multiplication is introduced.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bins: int = 8,
        bound: float = 2.0,
        slope_init: float = 1.0,
        backend: Backend = "torch",
        seed: int = 0,
        use_hadamard_mix: bool = True,
    ) -> None:
        if backend != "torch":
            raise ValueError(f"TropicalSawtoothLinear currently supports backend='torch' only, got {backend!r}")
        if bins < 2:
            raise ValueError(f"bins must be >= 2, got {bins}")
        if bound <= 0:
            raise ValueError(f"bound must be > 0, got {bound}")
        if slope_init <= 0:
            raise ValueError(f"slope_init must be > 0, got {slope_init}")
        super().__init__(in_features, out_features, backend=backend, output_scale=1.0)
        self.bins = int(bins)
        self.bound = float(bound)
        self.bin_width = 2.0 * self.bound / self.bins
        self.use_hadamard_mix = bool(use_hadamard_mix)
        self.padded_features = _next_power_of_two(out_features)
        gen = torch.Generator(device="cpu").manual_seed(seed + 1543)
        phases = torch.randint(0, 2, (out_features,), generator=gen, dtype=torch.long)
        bin_ids = torch.arange(bins, dtype=torch.long).view(1, bins)
        signs = torch.where((bin_ids + phases.view(out_features, 1)).remainder(2) == 0, 1.0, -1.0)
        left = torch.linspace(-self.bound, self.bound - self.bin_width, bins).view(1, bins)
        right = left + self.bin_width
        init_offsets = torch.where(signs > 0, -left - self.bin_width / 2.0, right - self.bin_width / 2.0)
        self.offset = nn.Parameter(init_offsets)
        self.log_slope = nn.Parameter(torch.full((out_features,), self._inverse_softplus(float(slope_init))))
        self.register_buffer("slope_signs", signs)
        self.register_buffer("coord_offsets", torch.arange(out_features, dtype=torch.long) * bins)
        if in_features < out_features:
            resize_index = torch.arange(out_features, dtype=torch.long).remainder(in_features)
        else:
            resize_index = torch.empty(0, dtype=torch.long)
        self.register_buffer("resize_index", resize_index)

    @staticmethod
    def _inverse_softplus(value: float) -> float:
        return math.log(math.expm1(value))

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, bins={self.bins}, "
            f"bound={self.bound}, use_hadamard_mix={self.use_hadamard_mix}"
        )

    def _project_input(self, x: Tensor, compute_dtype: torch.dtype) -> Tensor:
        return x.to(compute_dtype)

    def _resize_to_output(self, latent: Tensor) -> Tensor:
        if self.in_features == self.out_features:
            return latent
        if self.in_features > self.out_features:
            groups = (self.in_features + self.out_features - 1) // self.out_features
            padded = groups * self.out_features
            x = F.pad(latent, (0, padded - self.in_features)) if padded != self.in_features else latent
            return x.view(*x.shape[:-1], groups, self.out_features).sum(dim=-2) / math.sqrt(groups)
        repeats = (self.out_features + self.in_features - 1) // self.in_features
        return latent.index_select(-1, self.resize_index.to(device=latent.device)) / math.sqrt(repeats)

    def _hadamard_mix(self, x: Tensor) -> Tensor:
        if not self.use_hadamard_mix:
            return x
        pad = self.padded_features - self.out_features
        y = F.pad(x, (0, pad)) if pad > 0 else x
        y = _fwht_last_dim(y) / math.sqrt(self.padded_features)
        return y[..., : self.out_features]

    def _route_output(
        self,
        latent: Tensor,
        *,
        input_device: torch.device,
        compute_dtype: torch.dtype,
        training: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        del input_device, training
        x = self._resize_to_output(latent.to(compute_dtype))
        u = self._hadamard_mix(x)
        scaled = torch.floor((u + self.bound) / self.bin_width).clamp(0, self.bins - 1).long()
        flat_idx = (scaled + self.coord_offsets.to(device=scaled.device).view(1, 1, -1)).reshape(-1)
        signs = self.slope_signs.to(dtype=compute_dtype, device=u.device).reshape(-1).index_select(0, flat_idx).view_as(u)
        offsets = self.offset.to(dtype=compute_dtype, device=u.device).reshape(-1).index_select(0, flat_idx).view_as(u)
        slope_abs = F.softplus(self.log_slope).to(dtype=compute_dtype, device=u.device).view(1, 1, -1)
        z = signs * slope_abs * u + offsets
        output = self._hadamard_mix(z)
        left = -self.bound + scaled.to(dtype=compute_dtype) * self.bin_width
        right = left + self.bin_width
        margins = torch.minimum(u - left, right - u)
        return output, scaled, margins


class AbsDiffLUT(nn.Module):
    """Relation LUT with bits H(width - |q_a - k_a|).

    This is the two-input counterpart to :class:`PairwiseLinear`.  It is meant
    for score/routing functions where the primitive relation is query-key
    coordinate agreement rather than ordering inside a single vector.
    """

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

        self.features = features
        self.out_features = out_features
        self.tables = tables
        self.comparisons = comparisons
        self.table_size = 1 << comparisons
        self.use_min_margin_ste = use_min_margin_ste
        self.surrogate = surrogate
        self.output_scale = 1.0 / math.sqrt(tables) if use_output_scaling else 1.0
        self.cache_route_debug = True
        self._last_indices: Tensor | None = None
        self._last_margins: Tensor | None = None

        gen = torch.Generator(device="cpu").manual_seed(seed)
        coords = torch.randint(0, features, (tables, comparisons), generator=gen, dtype=torch.long)
        self.register_buffer("coords", coords)
        self.log_widths = nn.Parameter(torch.full((tables, comparisons), self._inverse_softplus(width_init)))
        self.lut = nn.Parameter(torch.randn(tables, self.table_size, out_features) * lut_init_std)
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
            neighbor_offsets = route_offsets.unsqueeze(-1)
            neighbor_idx = (neighbor_flat[:, route_start:route_stop] + neighbor_offsets).reshape(-1)
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
    """Pairwise-comparator LUT layer whose rows are generated by low-order Walsh features."""

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
    ) -> None:
        if walsh_order not in {1, 2}:
            raise ValueError(f"walsh_order must be 1 or 2, got {walsh_order}")
        if backend != "torch":
            raise ValueError(f"PairwiseWalshLinear currently supports backend='torch' only, got {backend!r}")

        super().__init__(
            in_features,
            out_features,
            tables=tables,
            comparisons=comparisons,
            backend=backend,
            seed=seed,
            lut_init_std=coeff_init_std,
            use_min_margin_ste=use_min_margin_ste,
            use_output_scaling=use_output_scaling,
            surrogate=surrogate,
        )
        del self.lut

        self.walsh_order = walsh_order
        pair_indices = torch.combinations(torch.arange(comparisons, dtype=torch.long), r=2)
        if walsh_order == 1:
            pair_indices = pair_indices[:0]
        self.register_buffer("pair_indices", pair_indices)

        term_count = 1 + comparisons + pair_indices.shape[0]
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
            f"use_min_margin_ste={self.use_min_margin_ste}, surrogate={self.surrogate!r}"
        )

    def materialize_lut(self, *, dtype: torch.dtype | None = None, device: torch.device | None = None) -> Tensor:
        compute_dtype = dtype if dtype is not None else self.constant.dtype
        compute_device = device if device is not None else self.constant.device
        bits = self.walsh_bits.to(dtype=compute_dtype, device=compute_device)
        output = self.constant.to(dtype=compute_dtype, device=compute_device).unsqueeze(1)
        output = output + torch.einsum(
            "jc,tco->tjo",
            bits,
            self.linear_coeff.to(dtype=compute_dtype, device=compute_device),
        )
        if self.pair_indices.numel() > 0:
            pairs = self.pair_indices.to(device=compute_device)
            pair_bits = bits[:, pairs[:, 0]] * bits[:, pairs[:, 1]]
            output = output + torch.einsum(
                "jp,tpo->tjo",
                pair_bits,
                self.pair_coeff.to(dtype=compute_dtype, device=compute_device),
            )
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
        neighbor_flat = current_flat.unsqueeze(-1) ^ self.powers.view(1, 1, -1)
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
            neighbor_offsets = route_offsets.unsqueeze(-1)
            neighbor_idx = (neighbor_flat[:, route_start:route_stop] + neighbor_offsets).reshape(-1)
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
        if training and (latent.requires_grad or self.thresholds.requires_grad):
            if self.use_min_margin_ste:
                ste_corr = self._min_margin_ste_with_lut(indices, margins, lut)
            else:
                ste_corr = self._full_ste_with_lut(indices, margins, lut)
            output = output + ste_corr.to(output.dtype)
        return output, indices, margins
