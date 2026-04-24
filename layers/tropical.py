from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from ..backend import Backend, trop_scores
from .base import RoutedLinearBase


def _winner_values(cell_values: Tensor, winner_idx: Tensor) -> Tensor:
    out_dim = cell_values.shape[-1]
    gather_idx = winner_idx.unsqueeze(-1).unsqueeze(-1).expand(*winner_idx.shape, 1, out_dim)
    return cell_values.gather(-2, gather_idx).squeeze(-2)


def _training_output(scores: Tensor, cell_values: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    top2_vals, top2_idx = scores.topk(k=2, dim=-1)
    winner_idx = top2_idx[..., 0]
    winner_values = _winner_values(cell_values, winner_idx)
    runner_values = _winner_values(cell_values, top2_idx[..., 1])
    margins = top2_vals[..., 0] - top2_vals[..., 1]
    group_output = winner_values + (0.5 / (1.0 + margins.abs())).unsqueeze(-1) * (runner_values - winner_values)
    return group_output, winner_idx, margins


def _eval_output(scores: Tensor, cell_values: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    winner_idx = scores.argmax(dim=-1)
    top2_vals = scores.topk(k=2, dim=-1).values
    margins = top2_vals[..., 0] - top2_vals[..., 1]
    return _winner_values(cell_values, winner_idx), winner_idx, margins


class TropLinear(RoutedLinearBase):
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
        if tables < 1:
            raise ValueError(f"tables must be >= 1, got {tables}")
        if groups < 1:
            raise ValueError(f"groups must be >= 1, got {groups}")
        if cells < 2:
            raise ValueError(f"cells must be >= 2, got {cells}")
        if rank < 1:
            raise ValueError(f"rank must be >= 1, got {rank}")

        output_scale = 1.0 / math.sqrt(tables * groups) if use_output_scaling else 1.0
        super().__init__(in_features, out_features, backend=backend, output_scale=output_scale)

        self.tables = tables
        self.groups = groups
        self.cells = cells
        self.rank = rank

        torch.manual_seed(seed)
        self.proj = nn.Linear(in_features, rank, bias=False)
        nn.init.normal_(self.proj.weight, mean=0.0, std=1.0 / math.sqrt(max(1, in_features)))

        router_std = 1.0 / math.sqrt(max(1, rank))
        self.router_weight = nn.Parameter(torch.randn(tables, groups, cells, rank) * router_std)
        self.router_bias = nn.Parameter(torch.zeros(tables, groups, cells))
        self.affine_weight = nn.Parameter(torch.randn(tables, groups, cells, rank, out_features) * 0.02)
        self.affine_bias = nn.Parameter(torch.zeros(tables, groups, cells, out_features))

        self.register_buffer("_radix", cells ** torch.arange(groups, dtype=torch.long))

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, tables={self.tables}, "
            f"groups={self.groups}, cells={self.cells}, rank={self.rank}, backend={self.backend!r}"
        )

    def _project_input(self, x: Tensor, compute_dtype: torch.dtype) -> Tensor:
        return self.proj(x).to(compute_dtype)

    def _route_output(
        self,
        latent: Tensor,
        *,
        input_device: torch.device,
        compute_dtype: torch.dtype,
        training: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        scores = trop_scores(
            latent,
            self.router_weight.to(dtype=compute_dtype, device=input_device),
            self.router_bias.to(dtype=compute_dtype, device=input_device),
            backend=self.backend,
        )
        cell_values = torch.einsum(
            "bsr,tgkro->bstgko",
            latent,
            self.affine_weight.to(dtype=compute_dtype, device=input_device),
        ) + self.affine_bias.to(dtype=compute_dtype, device=input_device).unsqueeze(0).unsqueeze(0)
        if training:
            group_output, winner_idx, margins = _training_output(scores, cell_values)
        else:
            group_output, winner_idx, margins = _eval_output(scores, cell_values)
        output = group_output.sum(dim=3).sum(dim=2)
        route_indices = (winner_idx.long() * self._radix.view(1, 1, 1, -1)).sum(dim=-1)
        return output, route_indices, margins