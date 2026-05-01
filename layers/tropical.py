from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

from ..backend import Backend, trop_scores
from .base import RoutedLinearBase
from .routing import _minface_mix, _pack_route_indices, _top2_indices
from .tropical_exact import TropLinearExactMixin

__all__ = ["TropLinear", "TropZeroDenseLinear", "_pack_route_indices"]


class TropLinear(TropLinearExactMixin, RoutedLinearBase):
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
        cpu_param_dtype: Literal["f32", "f16"] = "f32",
        exact_fused: Literal["off", "eval", "train"] = "off",
        score_route_max_bytes: int | None = None,
        cache_route_debug: bool = True,
    ) -> None:
        if heads < 1:
            raise ValueError(f"heads must be >= 1, got {heads}")
        if cells < 2:
            raise ValueError(f"cells must be >= 2, got {cells}")
        if code_dim < 1:
            raise ValueError(f"code_dim must be >= 1, got {code_dim}")
        if cpu_param_dtype not in {"f32", "f16"}:
            raise ValueError(f"cpu_param_dtype must be 'f32' or 'f16', got {cpu_param_dtype!r}")
        if exact_fused not in {"off", "eval", "train"}:
            raise ValueError(f"exact_fused must be 'off', 'eval', or 'train', got {exact_fused!r}")
        if score_route_max_bytes is not None and score_route_max_bytes < 0:
            raise ValueError(f"score_route_max_bytes must be >= 0 when set, got {score_route_max_bytes}")

        super().__init__(in_features, out_features, backend=backend, output_scale=1.0)
        self.heads = heads
        self.cells = cells
        self.code_dim = code_dim
        self.code_scale = 1.0 / math.sqrt(heads) if use_output_scaling else 1.0
        self.cpu_param_dtype = cpu_param_dtype
        self.exact_fused = exact_fused
        self.score_route_max_bytes = score_route_max_bytes
        self.cache_route_debug = cache_route_debug
        self._zig_router_weight_f16_cache: Tensor | None = None
        self._zig_router_bias_f16_cache: Tensor | None = None
        self._zig_code_f16_cache: Tensor | None = None
        self._zig_f16_cache_versions: tuple[int, int, int] | None = None
        self._exact_eval_cache: tuple[tuple[object, ...], Tensor, Tensor, Tensor, Tensor] | None = None

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
            f"cells={self.cells}, code_dim={self.code_dim}, backend={self.backend!r}, cpu_param_dtype={self.cpu_param_dtype!r}, "
            f"exact_fused={self.exact_fused!r}, score_route_max_bytes={self.score_route_max_bytes!r}, "
            f"cache_route_debug={self.cache_route_debug!r}"
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.exact_fused == "train" and self.training:
            return self._forward_exact_route(x)
        if self.exact_fused == "eval" and not self.training and not torch.is_grad_enabled():
            return self._forward_exact_fused(x, use_cache=(self.exact_fused == "eval" and not torch.is_grad_enabled()))
        return super().forward(x)

    def _project_input(self, x: Tensor, compute_dtype: torch.dtype) -> Tensor:
        if self.backend == "zig":
            return self.proj(x.to(dtype=self.proj.weight.dtype)).to(torch.float32)
        return self.proj(x.to(dtype=self.proj.weight.dtype)).to(compute_dtype)

    def _zig_params_for_inference(self) -> tuple[Tensor, Tensor, Tensor]:
        if self.cpu_param_dtype == "f32":
            return (
                self.router_weight.detach().to(device="cpu", dtype=torch.float32).contiguous(),
                self.router_bias.detach().to(device="cpu", dtype=torch.float32).contiguous(),
                self.code.detach().to(device="cpu", dtype=torch.float32).contiguous(),
            )

        versions = (self.router_weight._version, self.router_bias._version, self.code._version)
        cache_missing = (
            self._zig_router_weight_f16_cache is None
            or self._zig_router_bias_f16_cache is None
            or self._zig_code_f16_cache is None
            or self._zig_f16_cache_versions != versions
            or self._zig_router_weight_f16_cache.shape != self.router_weight.shape
            or self._zig_router_bias_f16_cache.shape != self.router_bias.shape
            or self._zig_code_f16_cache.shape != self.code.shape
        )
        if cache_missing:
            self._zig_router_weight_f16_cache = self.router_weight.detach().to(device="cpu", dtype=torch.float16).contiguous()
            self._zig_router_bias_f16_cache = self.router_bias.detach().to(device="cpu", dtype=torch.float16).contiguous()
            self._zig_code_f16_cache = self.code.detach().to(device="cpu", dtype=torch.float16).contiguous()
            self._zig_f16_cache_versions = versions
        return self._zig_router_weight_f16_cache, self._zig_router_bias_f16_cache, self._zig_code_f16_cache

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
        if self.backend == "zig":
            if training:
                raise RuntimeError("TropLinear backend='zig' is inference-only; call .eval() or use backend='torch' for training")
            if latent.device.type != "cpu":
                raise ValueError("TropLinear backend='zig' requires CPU input tensors")
            from ..backends import trop_route_hidden_zig

            weight, bias, code = self._zig_params_for_inference()
            hidden = trop_route_hidden_zig(
                latent.contiguous(),
                weight,
                bias,
                code,
                code_scale=self.code_scale,
                param_dtype=self.cpu_param_dtype,
            )
            empty_indices = torch.empty((*latent.shape[:2], 0), device=latent.device, dtype=torch.long)
            empty_margins = torch.empty((*latent.shape[:2], 0), device=latent.device, dtype=latent.dtype)
            return self._output_from_hidden(hidden, input_device=input_device, compute_dtype=torch.float32), empty_indices, empty_margins

        if self.backend == "tilelang" and latent.is_cuda and compute_dtype == torch.float32:
            weight = self.router_weight.to(dtype=compute_dtype, device=input_device)
            bias = self.router_bias.to(dtype=compute_dtype, device=input_device)
            code = self.code.to(dtype=compute_dtype, device=input_device)
            score_bytes = latent.shape[0] * latent.shape[1] * self.heads * self.cells * 4
            if not training and not torch.is_grad_enabled() and self.code_dim >= 128 and score_bytes <= 128 * 1024 * 1024:
                from ..backends import has_triton, trop_route_hidden_triton_eval

                if has_triton():
                    hidden, winner_idx, margins = trop_route_hidden_triton_eval(
                        latent,
                        weight,
                        bias,
                        code,
                        code_scale=self.code_scale,
                    )
                    return self._output_from_hidden(hidden, input_device=input_device, compute_dtype=compute_dtype), winner_idx, margins

            from ..backends import trop_route_hidden_tilelang

            hidden, winner_idx, margins = trop_route_hidden_tilelang(
                latent,
                weight,
                bias,
                code,
                code_scale=self.code_scale,
                training=training,
                score_route_max_bytes=self.score_route_max_bytes,
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


class TropZeroDenseLinear(RoutedLinearBase):
    """Tropical layer with sparse-coordinate routing and direct output codes.

    This experimental variant removes the dense projection and dense readout
    used by ``TropLinear``:

        score[h, k] = sum_r w[h, k, r] * x[anchor[h, k, r]] + b[h, k]
        y = bias + code_scale * sum_h code[h, argmax_k score[h, k]]

    It is meant as a zero-dense control for EMNIST and scaling experiments,
    not as a drop-in replacement for the optimized TropLinear TileLang path.
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
