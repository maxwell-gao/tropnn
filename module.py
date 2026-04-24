from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .functional import Backend, trop_minface_eval_output, trop_minface_training_output, trop_scores


class TropLinear(nn.Module):
    """Grouped tropical affine layer with hard inference and min-face training."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        tables: int = 16,
        groups: int = 2,
        cells: int = 4,
        rank: int = 32,
        backend: Backend = "torch",
        seed: int = 0,
        use_output_scaling: bool = True,
    ) -> None:
        super().__init__()
        if in_features < 1:
            raise ValueError(f"in_features must be >= 1, got {in_features}")
        if out_features < 1:
            raise ValueError(f"out_features must be >= 1, got {out_features}")
        if tables < 1:
            raise ValueError(f"tables must be >= 1, got {tables}")
        if groups < 1:
            raise ValueError(f"groups must be >= 1, got {groups}")
        if cells < 2:
            raise ValueError(f"cells must be >= 2, got {cells}")
        if rank < 1:
            raise ValueError(f"rank must be >= 1, got {rank}")

        self.in_features = in_features
        self.out_features = out_features
        self.tables = tables
        self.groups = groups
        self.cells = cells
        self.rank = rank
        self.backend = backend
        self.output_scale = 1.0 / math.sqrt(tables * groups) if use_output_scaling else 1.0

        torch.manual_seed(seed)
        self.proj = nn.Linear(in_features, rank, bias=False)
        nn.init.normal_(self.proj.weight, mean=0.0, std=1.0 / math.sqrt(max(1, in_features)))

        router_std = 1.0 / math.sqrt(max(1, rank))
        self.router_weight = nn.Parameter(torch.randn(tables, groups, cells, rank) * router_std)
        self.router_bias = nn.Parameter(torch.zeros(tables, groups, cells))
        self.affine_weight = nn.Parameter(torch.randn(tables, groups, cells, rank, out_features) * 0.02)
        self.affine_bias = nn.Parameter(torch.zeros(tables, groups, cells, out_features))

        self.register_buffer("_radix", cells ** torch.arange(groups, dtype=torch.long))
        self._last_indices: Optional[Tensor] = None
        self._last_margins: Optional[Tensor] = None

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, tables={self.tables}, "
            f"groups={self.groups}, cells={self.cells}, rank={self.rank}, backend={self.backend!r}"
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        compute_dtype = torch.float32 if x.dtype in {torch.float16, torch.bfloat16} else x.dtype
        z = self.proj(x).to(compute_dtype)
        scores = trop_scores(
            z,
            self.router_weight.to(dtype=compute_dtype, device=x.device),
            self.router_bias.to(dtype=compute_dtype, device=x.device),
            backend=self.backend,
        )
        cell_values = torch.einsum(
            "bsr,tgkro->bstgko",
            z,
            self.affine_weight.to(dtype=compute_dtype, device=x.device),
        ) + self.affine_bias.to(dtype=compute_dtype, device=x.device).unsqueeze(0).unsqueeze(0)
        if self.training:
            group_output, winner_idx, margins = trop_minface_training_output(scores, cell_values)
        else:
            group_output, winner_idx, margins = trop_minface_eval_output(scores, cell_values)
        output = group_output.sum(dim=3).sum(dim=2)
        if self.output_scale != 1.0:
            output = output * self.output_scale
        self._last_indices = (winner_idx.long() * self._radix.view(1, 1, 1, -1)).sum(dim=-1).detach()
        self._last_margins = margins.detach()
        return output.to(dtype=x.dtype)