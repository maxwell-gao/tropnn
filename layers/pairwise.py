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
