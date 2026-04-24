from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from ..backend import Backend
from .base import RoutedLinearBase
from .surrogate import ste_heaviside


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
    ) -> None:
        if tables < 1:
            raise ValueError(f"tables must be >= 1, got {tables}")
        if comparisons < 1:
            raise ValueError(f"comparisons must be >= 1, got {comparisons}")
        if backend != "torch":
            raise ValueError(f"PairwiseLinear currently supports backend='torch' only, got {backend!r}")

        output_scale = 1.0 / math.sqrt(tables) if use_output_scaling else 1.0
        super().__init__(in_features, out_features, backend=backend, output_scale=output_scale)

        self.tables = tables
        self.comparisons = comparisons
        self.table_size = 1 << comparisons
        self.use_min_margin_ste = use_min_margin_ste

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
            f"comparisons={self.comparisons}, backend={self.backend!r}, use_min_margin_ste={self.use_min_margin_ste}"
        )

    def _project_input(self, x: Tensor, compute_dtype: torch.dtype) -> Tensor:
        return x.to(compute_dtype)

    def _select_rows(self, table_idx: int, indices: Tensor, compute_dtype: torch.dtype) -> Tensor:
        batch, seq = indices.shape
        values = self.lut[table_idx].to(compute_dtype).index_select(0, indices.reshape(-1))
        return values.view(batch, seq, self.out_features)

    def _lookup_sum(self, indices: Tensor, compute_dtype: torch.dtype) -> Tensor:
        output = torch.zeros(*indices.shape[:2], self.out_features, device=indices.device, dtype=compute_dtype)
        for table_idx in range(indices.shape[-1]):
            output = output + self._select_rows(table_idx, indices[:, :, table_idx], compute_dtype)
        return output

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
        ste_delta = ste_heaviside(u_mins) - (u_mins > 0).to(u_mins.dtype)

        corr = torch.zeros(*indices.shape[:2], self.out_features, device=indices.device, dtype=torch.float32)
        for table_idx in range(indices.shape[-1]):
            current = self._select_rows(table_idx, indices[:, :, table_idx], torch.float32)
            neighbor = self._select_rows(table_idx, neighbor_indices[:, :, table_idx], torch.float32)
            corr = corr + ste_delta[:, :, table_idx].unsqueeze(-1).float() * (neighbor - current)
        return corr

    def _full_ste(self, indices: Tensor, margins: Tensor) -> Tensor:
        corr = torch.zeros(*indices.shape[:2], self.out_features, device=indices.device, dtype=torch.float32)
        for table_idx in range(indices.shape[-1]):
            current = self._select_rows(table_idx, indices[:, :, table_idx], torch.float32)
            for comp_idx in range(self.comparisons):
                neighbor_idx = indices[:, :, table_idx] ^ self.powers[comp_idx]
                neighbor = self._select_rows(table_idx, neighbor_idx, torch.float32)
                margin = margins[:, :, table_idx, comp_idx]
                ste_delta = ste_heaviside(margin) - (margin > 0).to(margin.dtype)
                corr = corr + ste_delta.unsqueeze(-1).float() * (neighbor - current)
        return corr

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
        output = self._lookup_sum(indices, compute_dtype)
        if training and (latent.requires_grad or self.thresholds.requires_grad):
            ste_corr = self._min_margin_ste(indices, margins) if self.use_min_margin_ste else self._full_ste(indices, margins)
            output = output + ste_corr.to(output.dtype)
        return output, indices, margins
