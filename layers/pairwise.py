from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
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
        lut_init_std: float = 0.02,
        use_min_margin_ste: bool = True,
        use_output_scaling: bool = True,
        surrogate: str = "fast_sigmoid_odd",
        cpu_lut_dtype: Literal["f32", "f16"] = "f32",
    ) -> None:
        if tables < 1:
            raise ValueError(f"tables must be >= 1, got {tables}")
        if comparisons < 1:
            raise ValueError(f"comparisons must be >= 1, got {comparisons}")
        if backend not in {"torch", "tilelang", "zig"}:
            raise ValueError(f"PairwiseLinear currently supports backend='torch', 'tilelang', or 'zig', got {backend!r}")
        if cpu_lut_dtype not in {"f32", "f16"}:
            raise ValueError(f"cpu_lut_dtype must be 'f32' or 'f16', got {cpu_lut_dtype!r}")
        surrogate_gradient(torch.zeros((), dtype=torch.float32), surrogate)

        output_scale = 1.0 / math.sqrt(tables) if use_output_scaling else 1.0
        super().__init__(in_features, out_features, backend=backend, output_scale=output_scale)

        self.tables = tables
        self.comparisons = comparisons
        self.table_size = 1 << comparisons
        self.use_min_margin_ste = use_min_margin_ste
        self.surrogate = surrogate
        self.cpu_lut_dtype = cpu_lut_dtype
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
        self.thresholds = nn.Parameter(torch.zeros(tables, comparisons))
        self.lut = nn.Parameter(torch.randn(tables, self.table_size, out_features) * lut_init_std)
        self.register_buffer("powers", 2 ** torch.arange(comparisons, dtype=torch.long))

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, tables={self.tables}, "
            f"comparisons={self.comparisons}, backend={self.backend!r}, use_min_margin_ste={self.use_min_margin_ste}, "
            f"surrogate={self.surrogate!r}, cpu_lut_dtype={self.cpu_lut_dtype!r}"
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
        output = self._lookup_sum(indices, compute_dtype)
        if training and (latent.requires_grad or self.thresholds.requires_grad):
            ste_corr = self._min_margin_ste(indices, margins) if self.use_min_margin_ste else self._full_ste(indices, margins)
            output = output + ste_corr.to(output.dtype)
        return output, indices, margins
