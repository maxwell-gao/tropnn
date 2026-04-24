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


def _top2_indices(scores: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    top2_vals, top2_idx = scores.topk(k=2, dim=-1)
    return top2_idx[..., 0], top2_idx[..., 1], top2_vals[..., 0] - top2_vals[..., 1]


def _minface_mix(winner_values: Tensor, runner_values: Tensor, margins: Tensor) -> Tensor:
    return winner_values + (0.5 / (1.0 + margins.abs())).unsqueeze(-1) * (runner_values - winner_values)


class _GroupedTropicalPayloadBase(RoutedLinearBase):
    """Shared grouped tropical router for sparse/shared payload variants."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        tables: int,
        groups: int,
        cells: int,
        rank: int,
        backend: Backend,
        seed: int,
        use_output_scaling: bool,
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

        self.register_buffer("_radix", cells ** torch.arange(groups, dtype=torch.long))

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, tables={self.tables}, "
            f"groups={self.groups}, cells={self.cells}, rank={self.rank}, backend={self.backend!r}"
        )

    def _project_input(self, x: Tensor, compute_dtype: torch.dtype) -> Tensor:
        return self.proj(x).to(compute_dtype)

    def _scores(self, latent: Tensor, *, input_device: torch.device, compute_dtype: torch.dtype) -> Tensor:
        return trop_scores(
            latent,
            self.router_weight.to(dtype=compute_dtype, device=input_device),
            self.router_bias.to(dtype=compute_dtype, device=input_device),
            backend=self.backend,
        )

    def _route_indices(self, winner_idx: Tensor) -> Tensor:
        return (winner_idx.long() * self._radix.view(1, 1, 1, -1)).sum(dim=-1)


def _select_static_payload(payload: Tensor, winner_idx: Tensor, *, compute_dtype: torch.dtype, input_device: torch.device) -> Tensor:
    values = payload.to(dtype=compute_dtype, device=input_device).unsqueeze(0).unsqueeze(0)
    gather_idx = winner_idx.unsqueeze(-1).unsqueeze(-1).expand(*winner_idx.shape, 1, values.shape[-1])
    return values.expand(*winner_idx.shape[:2], -1, -1, -1, -1).gather(-2, gather_idx).squeeze(-2)


class TropLUTLinear(_GroupedTropicalPayloadBase):
    """Grouped tropical router with pure selected-vector LUT payload."""

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
        lut_init_std: float = 0.02,
        use_output_scaling: bool = True,
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            tables=tables,
            groups=groups,
            cells=cells,
            rank=rank,
            backend=backend,
            seed=seed,
            use_output_scaling=use_output_scaling,
        )
        self.lut = nn.Parameter(torch.randn(tables, groups, cells, out_features) * lut_init_std)

    def _payload_values(self, latent: Tensor, *, input_device: torch.device, compute_dtype: torch.dtype) -> Tensor:
        del latent
        return self.lut.to(dtype=compute_dtype, device=input_device).unsqueeze(0).unsqueeze(0)

    def _hard_output(self, winner_idx: Tensor, *, input_device: torch.device, compute_dtype: torch.dtype) -> Tensor:
        return _select_static_payload(self.lut, winner_idx, compute_dtype=compute_dtype, input_device=input_device).sum(dim=3).sum(dim=2)

    def _route_output(
        self,
        latent: Tensor,
        *,
        input_device: torch.device,
        compute_dtype: torch.dtype,
        training: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        scores = self._scores(latent, input_device=input_device, compute_dtype=compute_dtype)
        winner_idx, runner_idx, margins = _top2_indices(scores)
        if training:
            cell_values = self._payload_values(latent, input_device=input_device, compute_dtype=compute_dtype)
            winner_values = _winner_values(cell_values.expand(*scores.shape[:2], -1, -1, -1, -1), winner_idx)
            runner_values = _winner_values(cell_values.expand(*scores.shape[:2], -1, -1, -1, -1), runner_idx)
            output = _minface_mix(winner_values, runner_values, margins).sum(dim=3).sum(dim=2)
        else:
            output = self._hard_output(winner_idx, input_device=input_device, compute_dtype=compute_dtype)
        return output, self._route_indices(winner_idx), margins


class TropBinaryAdditiveLUT(TropLUTLinear):
    """Binary tropical comparisons with group-additive LUT payloads."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        tables: int = 16,
        groups: int = 6,
        rank: int = 32,
        backend: Backend = "torch",
        seed: int = 0,
        lut_init_std: float = 0.02,
        use_output_scaling: bool = True,
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            tables=tables,
            groups=groups,
            cells=2,
            rank=rank,
            backend=backend,
            seed=seed,
            lut_init_std=lut_init_std,
            use_output_scaling=use_output_scaling,
        )


class TropCodeLinear(RoutedLinearBase):
    """Tropical heads with compact selected codes and a shared output map."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        heads: int = 32,
        cells: int = 4,
        code_dim: int = 32,
        backend: Backend = "torch",
        seed: int = 0,
        code_init_std: float = 0.02,
        use_output_scaling: bool = True,
    ) -> None:
        if heads < 1:
            raise ValueError(f"heads must be >= 1, got {heads}")
        if cells < 2:
            raise ValueError(f"cells must be >= 2, got {cells}")
        if code_dim < 1:
            raise ValueError(f"code_dim must be >= 1, got {code_dim}")

        super().__init__(in_features, out_features, backend=backend, output_scale=1.0)
        self.heads = heads
        self.cells = cells
        self.code_dim = code_dim
        self.code_scale = 1.0 / math.sqrt(heads) if use_output_scaling else 1.0

        torch.manual_seed(seed)
        self.proj = nn.Linear(in_features, code_dim, bias=False)
        nn.init.normal_(self.proj.weight, mean=0.0, std=1.0 / math.sqrt(max(1, in_features)))

        router_std = 1.0 / math.sqrt(max(1, code_dim))
        self.router_weight = nn.Parameter(torch.randn(heads, cells, code_dim) * router_std)
        self.router_bias = nn.Parameter(torch.zeros(heads, cells))
        self.code = nn.Parameter(torch.randn(heads, cells, code_dim) * code_init_std)
        self.output_proj = nn.Linear(code_dim, out_features)
        nn.init.kaiming_uniform_(self.output_proj.weight, a=math.sqrt(5))
        if self.output_proj.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.output_proj.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.output_proj.bias, -bound, bound)

    @property
    def tables(self) -> int:
        return self.heads

    @property
    def groups(self) -> int:
        return 1

    @property
    def rank(self) -> int:
        return self.code_dim

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, heads={self.heads}, "
            f"cells={self.cells}, code_dim={self.code_dim}, backend={self.backend!r}"
        )

    def _project_input(self, x: Tensor, compute_dtype: torch.dtype) -> Tensor:
        return self.proj(x).to(compute_dtype)

    def _scores(self, latent: Tensor, *, input_device: torch.device, compute_dtype: torch.dtype) -> Tensor:
        weight = self.router_weight.to(dtype=compute_dtype, device=input_device).unsqueeze(1)
        bias = self.router_bias.to(dtype=compute_dtype, device=input_device).unsqueeze(1)
        return trop_scores(latent, weight, bias, backend=self.backend).squeeze(3)

    def _selected_codes(self, winner_idx: Tensor, *, input_device: torch.device, compute_dtype: torch.dtype) -> Tensor:
        code = self.code.to(dtype=compute_dtype, device=input_device).unsqueeze(0).unsqueeze(0)
        gather_idx = winner_idx.unsqueeze(-1).unsqueeze(-1).expand(*winner_idx.shape, 1, self.code_dim)
        return code.expand(*winner_idx.shape[:2], -1, -1, -1).gather(-2, gather_idx).squeeze(-2)

    def _output_from_codes(self, latent: Tensor, codes: Tensor, *, input_device: torch.device, compute_dtype: torch.dtype) -> Tensor:
        hidden = latent + codes.sum(dim=2) * self.code_scale
        weight = self.output_proj.weight.to(dtype=compute_dtype, device=input_device)
        bias = self.output_proj.bias
        output = torch.matmul(hidden, weight.t())
        if bias is not None:
            output = output + bias.to(dtype=compute_dtype, device=input_device)
        return output

    def _route_output(
        self,
        latent: Tensor,
        *,
        input_device: torch.device,
        compute_dtype: torch.dtype,
        training: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        scores = self._scores(latent, input_device=input_device, compute_dtype=compute_dtype)
        winner_idx, runner_idx, margins = _top2_indices(scores)
        if training:
            winner_codes = self._selected_codes(winner_idx, input_device=input_device, compute_dtype=compute_dtype)
            runner_codes = self._selected_codes(runner_idx, input_device=input_device, compute_dtype=compute_dtype)
            codes = _minface_mix(winner_codes, runner_codes, margins)
        else:
            codes = self._selected_codes(winner_idx, input_device=input_device, compute_dtype=compute_dtype)
        return self._output_from_codes(latent, codes, input_device=input_device, compute_dtype=compute_dtype), winner_idx, margins


class TropGatedLinear(_GroupedTropicalPayloadBase):
    """Grouped tropical router with cell-local scalar-gated vector payloads."""

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
        super().__init__(
            in_features,
            out_features,
            tables=tables,
            groups=groups,
            cells=cells,
            rank=rank,
            backend=backend,
            seed=seed,
            use_output_scaling=use_output_scaling,
        )
        self.gate_weight = nn.Parameter(torch.randn(tables, groups, cells, rank) * (1.0 / math.sqrt(max(1, rank))))
        self.gate_bias = nn.Parameter(torch.zeros(tables, groups, cells))
        self.payload_vector = nn.Parameter(torch.randn(tables, groups, cells, out_features) * 0.02)
        self.payload_bias = nn.Parameter(torch.zeros(tables, groups, cells, out_features))

    def _cell_values(self, latent: Tensor, *, input_device: torch.device, compute_dtype: torch.dtype) -> Tensor:
        gate = torch.einsum(
            "bsr,tgkr->bstgk",
            latent,
            self.gate_weight.to(dtype=compute_dtype, device=input_device),
        ) + self.gate_bias.to(dtype=compute_dtype, device=input_device).unsqueeze(0).unsqueeze(0)
        return (
            gate.unsqueeze(-1) * self.payload_vector.to(dtype=compute_dtype, device=input_device).unsqueeze(0).unsqueeze(0)
            + self.payload_bias.to(dtype=compute_dtype, device=input_device).unsqueeze(0).unsqueeze(0)
        )

    def _hard_output(self, latent: Tensor, winner_idx: Tensor, *, input_device: torch.device, compute_dtype: torch.dtype) -> Tensor:
        batch, steps, rank = latent.shape
        item_count = batch * steps
        route_count = self.tables * self.groups
        latent_flat = latent.reshape(item_count, rank)
        winner_flat = winner_idx.reshape(item_count, route_count)
        route_offsets = (torch.arange(route_count, device=input_device) * self.cells).view(1, route_count)
        linear_idx = (winner_flat + route_offsets).reshape(-1)

        gate_table = self.gate_weight.to(dtype=compute_dtype, device=input_device).reshape(route_count * self.cells, rank)
        gate_bias = self.gate_bias.to(dtype=compute_dtype, device=input_device).reshape(route_count * self.cells)
        vector_table = self.payload_vector.to(dtype=compute_dtype, device=input_device).reshape(route_count * self.cells, self.out_features)
        bias_table = self.payload_bias.to(dtype=compute_dtype, device=input_device).reshape(route_count * self.cells, self.out_features)

        selected_gate = gate_table.index_select(0, linear_idx).view(item_count, route_count, rank)
        selected_gate_bias = gate_bias.index_select(0, linear_idx).view(item_count, route_count)
        selected_vector = vector_table.index_select(0, linear_idx).view(item_count, route_count, self.out_features)
        selected_bias = bias_table.index_select(0, linear_idx).view(item_count, route_count, self.out_features)
        gate = torch.einsum("nr,ncr->nc", latent_flat, selected_gate) + selected_gate_bias
        output = (gate.unsqueeze(-1) * selected_vector + selected_bias).sum(dim=1)
        return output.view(batch, steps, self.out_features)

    def _route_output(
        self,
        latent: Tensor,
        *,
        input_device: torch.device,
        compute_dtype: torch.dtype,
        training: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        scores = self._scores(latent, input_device=input_device, compute_dtype=compute_dtype)
        winner_idx, runner_idx, margins = _top2_indices(scores)
        if training:
            cell_values = self._cell_values(latent, input_device=input_device, compute_dtype=compute_dtype)
            winner_values = _winner_values(cell_values, winner_idx)
            runner_values = _winner_values(cell_values, runner_idx)
            output = _minface_mix(winner_values, runner_values, margins).sum(dim=3).sum(dim=2)
        else:
            output = self._hard_output(latent, winner_idx, input_device=input_device, compute_dtype=compute_dtype)
        return output, self._route_indices(winner_idx), margins


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

    def _hard_eval_output(self, latent: Tensor, winner_idx: Tensor, *, input_device: torch.device, compute_dtype: torch.dtype) -> Tensor:
        batch, steps, rank = latent.shape
        item_count = batch * steps
        route_count = self.tables * self.groups
        latent_flat = latent.reshape(item_count, rank)
        output = torch.zeros(item_count, self.out_features, device=input_device, dtype=compute_dtype)
        affine_weight = self.affine_weight.to(dtype=compute_dtype, device=input_device)
        affine_bias = self.affine_bias.to(dtype=compute_dtype, device=input_device)
        winner_flat = winner_idx.reshape(item_count, route_count)
        weight_table = affine_weight.reshape(route_count * self.cells, rank, self.out_features)
        bias_table = affine_bias.reshape(route_count * self.cells, self.out_features)
        route_chunk = self._route_chunk_size(
            item_count=item_count,
            payload_width=rank * self.out_features,
            compute_dtype=compute_dtype,
            route_count=route_count,
            target_bytes=16 * 1024 * 1024,
        )

        for route_start in range(0, route_count, route_chunk):
            route_stop = min(route_start + route_chunk, route_count)
            route_offsets = (torch.arange(route_start, route_stop, device=input_device) * self.cells).view(1, -1)
            linear_idx = (winner_flat[:, route_start:route_stop] + route_offsets).reshape(-1)
            selected_weight = weight_table.index_select(0, linear_idx).view(item_count, route_stop - route_start, rank, self.out_features)
            selected_bias = bias_table.index_select(0, linear_idx).view(item_count, route_stop - route_start, self.out_features)
            output = output + torch.einsum("nk,ncko->no", latent_flat, selected_weight) + selected_bias.sum(dim=1)

        return output.view(batch, steps, self.out_features)

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
        if training:
            cell_values = torch.einsum(
                "bsr,tgkro->bstgko",
                latent,
                self.affine_weight.to(dtype=compute_dtype, device=input_device),
            ) + self.affine_bias.to(dtype=compute_dtype, device=input_device).unsqueeze(0).unsqueeze(0)
            group_output, winner_idx, margins = _training_output(scores, cell_values)
            output = group_output.sum(dim=3).sum(dim=2)
        else:
            winner_idx = scores.argmax(dim=-1)
            top2_vals = scores.topk(k=2, dim=-1).values
            margins = top2_vals[..., 0] - top2_vals[..., 1]
            output = self._hard_eval_output(latent, winner_idx, input_device=input_device, compute_dtype=compute_dtype)
        route_indices = (winner_idx.long() * self._radix.view(1, 1, 1, -1)).sum(dim=-1)
        return output, route_indices, margins
