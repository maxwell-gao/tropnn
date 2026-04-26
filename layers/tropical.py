from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from ..backend import Backend, trop_scores
from .base import RoutedLinearBase


def _top2_indices(scores: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    top2_vals, top2_idx = scores.topk(k=2, dim=-1)
    return top2_idx[..., 0], top2_idx[..., 1], top2_vals[..., 0] - top2_vals[..., 1]


def _minface_mix(winner_values: Tensor, runner_values: Tensor, margins: Tensor) -> Tensor:
    return winner_values + (0.5 / (1.0 + margins.abs())).unsqueeze(-1) * (runner_values - winner_values)


class TropLinear(RoutedLinearBase):
    """Tropical code layer with hard head routing and a shared output map."""

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

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, heads={self.heads}, "
            f"cells={self.cells}, code_dim={self.code_dim}, backend={self.backend!r}"
        )

    def _project_input(self, x: Tensor, compute_dtype: torch.dtype) -> Tensor:
        return self.proj(x).to(compute_dtype)

    def _scores(self, latent: Tensor, *, input_device: torch.device, compute_dtype: torch.dtype) -> Tensor:
        weight = self.router_weight.to(dtype=compute_dtype, device=input_device)
        bias = self.router_bias.to(dtype=compute_dtype, device=input_device)
        return trop_scores(latent, weight, bias, backend=self.backend)

    def _selected_codes(self, winner_idx: Tensor, *, input_device: torch.device, compute_dtype: torch.dtype) -> Tensor:
        code = self.code.to(dtype=compute_dtype, device=input_device).unsqueeze(0).unsqueeze(0)
        gather_idx = winner_idx.unsqueeze(-1).unsqueeze(-1).expand(*winner_idx.shape, 1, self.code_dim)
        return code.expand(*winner_idx.shape[:2], -1, -1, -1).gather(-2, gather_idx).squeeze(-2)

    def _output_from_codes(self, latent: Tensor, codes: Tensor, *, input_device: torch.device, compute_dtype: torch.dtype) -> Tensor:
        hidden = latent + codes.sum(dim=2) * self.code_scale
        return self._output_from_hidden(hidden, input_device=input_device, compute_dtype=compute_dtype)

    def _output_from_hidden(self, hidden: Tensor, *, input_device: torch.device, compute_dtype: torch.dtype) -> Tensor:
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
        if self.backend == "tilelang" and not training:
            from ..backends import trop_route_hidden_tilelang

            weight = self.router_weight.to(dtype=compute_dtype, device=input_device)
            bias = self.router_bias.to(dtype=compute_dtype, device=input_device)
            code = self.code.to(dtype=compute_dtype, device=input_device)
            hidden, winner_idx, margins = trop_route_hidden_tilelang(
                latent,
                weight,
                bias,
                code,
                code_scale=self.code_scale,
            )
            return self._output_from_hidden(hidden, input_device=input_device, compute_dtype=compute_dtype), winner_idx, margins

        score_backend = "torch" if self.backend == "tilelang" else self.backend
        scores = trop_scores(
            latent,
            self.router_weight.to(dtype=compute_dtype, device=input_device),
            self.router_bias.to(dtype=compute_dtype, device=input_device),
            backend=score_backend,
        )
        winner_idx, runner_idx, margins = _top2_indices(scores)
        if training:
            winner_codes = self._selected_codes(winner_idx, input_device=input_device, compute_dtype=compute_dtype)
            runner_codes = self._selected_codes(runner_idx, input_device=input_device, compute_dtype=compute_dtype)
            codes = _minface_mix(winner_codes, runner_codes, margins)
        else:
            codes = self._selected_codes(winner_idx, input_device=input_device, compute_dtype=compute_dtype)
        return self._output_from_codes(latent, codes, input_device=input_device, compute_dtype=compute_dtype), winner_idx, margins


class TropFiLMLinear(TropLinear):
    """Tropical code layer with selected latent scale and offset.

    This keeps the shared low-rank factors from TropLinear:

        z = P x
        y = W h

    but each selected tropical cell contributes both an offset code and a
    feature-wise latent scale:

        h = z * (1 + sum_h gamma[h, i_h]) + sum_h code[h, i_h].

    The scale parameters are zero-initialized so the layer starts as an ordinary
    TropLinear with identical selected offsets.
    """

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
        scale_init_std: float = 0.0,
        use_output_scaling: bool = True,
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            heads=heads,
            cells=cells,
            code_dim=code_dim,
            backend=backend,
            seed=seed,
            code_init_std=code_init_std,
            use_output_scaling=use_output_scaling,
        )
        if scale_init_std == 0.0:
            scale = torch.zeros(heads, cells, code_dim)
        else:
            scale = torch.randn(heads, cells, code_dim) * scale_init_std
        self.scale = nn.Parameter(scale)

    def _selected_scales(self, winner_idx: Tensor, *, input_device: torch.device, compute_dtype: torch.dtype) -> Tensor:
        scale = self.scale.to(dtype=compute_dtype, device=input_device).unsqueeze(0).unsqueeze(0)
        gather_idx = winner_idx.unsqueeze(-1).unsqueeze(-1).expand(*winner_idx.shape, 1, self.code_dim)
        return scale.expand(*winner_idx.shape[:2], -1, -1, -1).gather(-2, gather_idx).squeeze(-2)

    def _output_from_codes_and_scales(
        self,
        latent: Tensor,
        codes: Tensor,
        scales: Tensor,
        *,
        input_device: torch.device,
        compute_dtype: torch.dtype,
    ) -> Tensor:
        scale_delta = scales.sum(dim=2) * self.code_scale
        offset = codes.sum(dim=2) * self.code_scale
        hidden = latent * (1.0 + scale_delta) + offset
        return self._output_from_hidden(hidden, input_device=input_device, compute_dtype=compute_dtype)

    def _route_output(
        self,
        latent: Tensor,
        *,
        input_device: torch.device,
        compute_dtype: torch.dtype,
        training: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        score_backend = "torch" if self.backend == "tilelang" else self.backend
        scores = trop_scores(
            latent,
            self.router_weight.to(dtype=compute_dtype, device=input_device),
            self.router_bias.to(dtype=compute_dtype, device=input_device),
            backend=score_backend,
        )
        winner_idx, runner_idx, margins = _top2_indices(scores)
        if training:
            winner_codes = self._selected_codes(winner_idx, input_device=input_device, compute_dtype=compute_dtype)
            runner_codes = self._selected_codes(runner_idx, input_device=input_device, compute_dtype=compute_dtype)
            winner_scales = self._selected_scales(winner_idx, input_device=input_device, compute_dtype=compute_dtype)
            runner_scales = self._selected_scales(runner_idx, input_device=input_device, compute_dtype=compute_dtype)
            codes = _minface_mix(winner_codes, runner_codes, margins)
            scales = _minface_mix(winner_scales, runner_scales, margins)
        else:
            codes = self._selected_codes(winner_idx, input_device=input_device, compute_dtype=compute_dtype)
            scales = self._selected_scales(winner_idx, input_device=input_device, compute_dtype=compute_dtype)
        output = self._output_from_codes_and_scales(
            latent,
            codes,
            scales,
            input_device=input_device,
            compute_dtype=compute_dtype,
        )
        return output, winner_idx, margins


class TropZeroDenseLinear(RoutedLinearBase):
    """Tropical layer with sparse coordinate routing and direct output codes.

    This variant deliberately avoids dense projections and dense readout:

        score[h, k] = sum_r w[h, k, r] * x[anchor[h, k, r]] + b[h, k]
        y = sum_h code[h, argmax_k score[h, k]]

    It is an intentionally direct zero-dense baseline rather than a drop-in
    replacement for the low-rank TropLinear geometry.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        heads: int = 32,
        cells: int = 4,
        route_terms: int = 2,
        backend: Backend = "torch",
        seed: int = 0,
        code_init_std: float = 0.02,
        use_output_scaling: bool = True,
    ) -> None:
        if heads < 1:
            raise ValueError(f"heads must be >= 1, got {heads}")
        if cells < 2:
            raise ValueError(f"cells must be >= 2, got {cells}")
        if route_terms < 1:
            raise ValueError(f"route_terms must be >= 1, got {route_terms}")
        if backend != "torch":
            raise ValueError(f"TropZeroDenseLinear currently supports backend='torch' only, got {backend!r}")

        super().__init__(in_features, out_features, backend=backend, output_scale=1.0)
        self.heads = heads
        self.cells = cells
        self.route_terms = route_terms
        self.code_scale = 1.0 / math.sqrt(heads) if use_output_scaling else 1.0

        torch.manual_seed(seed)
        anchors = torch.randint(0, in_features, (heads, cells, route_terms), dtype=torch.long)
        self.register_buffer("anchors", anchors)
        router_std = 1.0 / math.sqrt(route_terms)
        self.router_weight = nn.Parameter(torch.randn(heads, cells, route_terms) * router_std)
        self.router_bias = nn.Parameter(torch.zeros(heads, cells))
        self.code = nn.Parameter(torch.randn(heads, cells, out_features) * code_init_std)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, heads={self.heads}, "
            f"cells={self.cells}, route_terms={self.route_terms}, backend={self.backend!r}"
        )

    def _project_input(self, x: Tensor, compute_dtype: torch.dtype) -> Tensor:
        return x.to(compute_dtype)

    def _scores(self, latent: Tensor, *, input_device: torch.device, compute_dtype: torch.dtype) -> Tensor:
        del input_device
        batch, seq, _ = latent.shape
        selected = latent.index_select(-1, self.anchors.flatten()).view(batch, seq, self.heads, self.cells, self.route_terms)
        weight = self.router_weight.to(dtype=compute_dtype, device=latent.device)
        bias = self.router_bias.to(dtype=compute_dtype, device=latent.device)
        return (selected * weight.view(1, 1, self.heads, self.cells, self.route_terms)).sum(dim=-1) + bias.view(1, 1, self.heads, self.cells)

    def _selected_codes(self, winner_idx: Tensor, *, input_device: torch.device, compute_dtype: torch.dtype) -> Tensor:
        code = self.code.to(dtype=compute_dtype, device=input_device).unsqueeze(0).unsqueeze(0)
        gather_idx = winner_idx.unsqueeze(-1).unsqueeze(-1).expand(*winner_idx.shape, 1, self.out_features)
        return code.expand(*winner_idx.shape[:2], -1, -1, -1).gather(-2, gather_idx).squeeze(-2)

    def _output_from_codes(self, codes: Tensor, *, input_device: torch.device, compute_dtype: torch.dtype) -> Tensor:
        output = codes.sum(dim=2) * self.code_scale
        return output + self.bias.to(dtype=compute_dtype, device=input_device)

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
        return self._output_from_codes(codes, input_device=input_device, compute_dtype=compute_dtype), winner_idx, margins
