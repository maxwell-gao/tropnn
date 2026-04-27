from __future__ import annotations

import math
from typing import Literal

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


class _TropExactRouteFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        latent: Tensor,
        router_weight: Tensor,
        router_bias: Tensor,
        code: Tensor,
        code_scale: float,
        training: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch, steps, code_dim = latent.shape
        heads, cells, _ = code.shape
        item_count = batch * steps
        latent_flat = latent.reshape(item_count, code_dim).contiguous()
        weight = router_weight.contiguous()
        bias = router_bias.contiguous()
        code_table = code.contiguous()

        if latent.is_cuda:
            try:
                from ..backends.tilelang_route import _run_trop_forward
            except (ImportError, RuntimeError):
                hidden, winner_idx, runner_idx, margins = _TropExactRouteFunction._forward_torch(
                    latent_flat,
                    weight,
                    bias,
                    code_table,
                    code_scale=float(code_scale),
                    training=bool(training),
                    shape=(batch, steps, code_dim, heads, cells),
                )
            else:
                try:
                    hidden_view, winner_view, runner_view, margins_view = _run_trop_forward(
                        latent,
                        weight,
                        bias,
                        code_table,
                        code_scale=float(code_scale),
                        training=bool(training),
                        target="cuda",
                    )
                except RuntimeError:
                    hidden, winner_idx, runner_idx, margins = _TropExactRouteFunction._forward_torch(
                        latent_flat,
                        weight,
                        bias,
                        code_table,
                        code_scale=float(code_scale),
                        training=bool(training),
                        shape=(batch, steps, code_dim, heads, cells),
                    )
                else:
                    hidden = hidden_view.reshape(item_count, code_dim)
                    winner_idx = winner_view.reshape(item_count, heads).contiguous()
                    runner_idx = runner_view.reshape(item_count, heads).contiguous()
                    margins = margins_view.reshape(item_count, heads).contiguous()
        else:
            hidden, winner_idx, runner_idx, margins = _TropExactRouteFunction._forward_torch(
                latent_flat,
                weight,
                bias,
                code_table,
                code_scale=float(code_scale),
                training=bool(training),
                shape=(batch, steps, code_dim, heads, cells),
            )

        ctx.save_for_backward(latent_flat, weight, code_table, winner_idx, runner_idx, margins)
        ctx.code_scale = float(code_scale)
        ctx.training = bool(training)
        ctx.shape = (batch, steps, code_dim, heads, cells)
        winner_out = winner_idx.view(batch, steps, heads)
        margins_out = margins.view(batch, steps, heads)
        ctx.mark_non_differentiable(winner_out, margins_out)
        return hidden.view(batch, steps, code_dim), winner_out, margins_out

    @staticmethod
    def _forward_torch(
        latent_flat: Tensor,
        weight: Tensor,
        router_bias: Tensor,
        code_table: Tensor,
        *,
        code_scale: float,
        training: bool,
        shape: tuple[int, int, int, int, int],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        _batch, _steps, code_dim, heads, cells = shape
        item_count = latent_flat.shape[0]
        scores = torch.matmul(latent_flat, weight.reshape(heads * cells, code_dim).t()).view(item_count, heads, cells)
        scores = scores + router_bias.view(1, heads, cells)
        winner_idx, runner_idx, margins = _top2_indices(scores)

        route_offsets = torch.arange(heads, device=latent_flat.device, dtype=torch.long).view(1, heads) * cells
        flat_code = code_table.reshape(heads * cells, code_dim)
        winner_values = flat_code.index_select(0, (winner_idx + route_offsets).reshape(-1)).view(item_count, heads, code_dim)
        if training:
            runner_values = flat_code.index_select(0, (runner_idx + route_offsets).reshape(-1)).view(item_count, heads, code_dim)
            values = _minface_mix(winner_values, runner_values, margins)
        else:
            values = winner_values
        hidden = latent_flat + values.sum(dim=1) * code_scale
        return hidden, winner_idx, runner_idx, margins

    @staticmethod
    def backward(
        ctx,
        grad_hidden: Tensor,
        grad_winner: Tensor | None,
        grad_margins_out: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, None, None]:
        del grad_winner, grad_margins_out
        latent_flat, router_weight, code, winner_idx, runner_idx, margins = ctx.saved_tensors
        batch, steps, code_dim, heads, cells = ctx.shape
        item_count = batch * steps
        grad_flat = grad_hidden.reshape(item_count, code_dim).contiguous().to(torch.float32)

        if ctx.training and grad_flat.is_cuda:
            try:
                from ..backends.triton_backward import trop_exact_route_backward_triton
            except (ImportError, RuntimeError):
                pass
            else:
                grad_latent, grad_router_weight, grad_router_bias, grad_code = trop_exact_route_backward_triton(
                    grad_flat,
                    latent_flat,
                    router_weight,
                    code,
                    winner_idx,
                    runner_idx,
                    margins,
                    code_scale=ctx.code_scale,
                )
                return (
                    grad_latent.view(batch, steps, code_dim),
                    grad_router_weight,
                    grad_router_bias,
                    grad_code,
                    None,
                    None,
                )

        code_coeff = torch.zeros(item_count, heads, cells, device=grad_flat.device, dtype=grad_flat.dtype)
        if ctx.training:
            alpha = 0.5 / (1.0 + margins.abs())
            code_coeff.scatter_add_(2, winner_idx.unsqueeze(-1), (1.0 - alpha).unsqueeze(-1))
            code_coeff.scatter_add_(2, runner_idx.unsqueeze(-1), alpha.unsqueeze(-1))
        else:
            code_coeff.scatter_add_(2, winner_idx.unsqueeze(-1), torch.ones_like(winner_idx, dtype=grad_flat.dtype).unsqueeze(-1))

        grad_code = torch.einsum("nhc,nr->hcr", code_coeff, grad_flat) * ctx.code_scale
        grad_latent = grad_flat.clone()
        grad_router_weight = torch.zeros_like(router_weight)
        grad_router_bias = torch.zeros(heads, cells, device=grad_flat.device, dtype=grad_flat.dtype)

        if ctx.training:
            route_offsets = torch.arange(heads, device=grad_flat.device, dtype=torch.long).view(1, heads) * cells
            flat_code = code.reshape(heads * cells, code_dim)
            winner_values = flat_code.index_select(0, (winner_idx + route_offsets).reshape(-1)).view(item_count, heads, code_dim)
            runner_values = flat_code.index_select(0, (runner_idx + route_offsets).reshape(-1)).view(item_count, heads, code_dim)
            delta_code = runner_values - winner_values
            dalpha = -0.5 * margins.sign() / (1.0 + margins.abs()).square()
            grad_margin = (grad_flat.unsqueeze(1) * delta_code).sum(dim=-1) * ctx.code_scale * dalpha

            route_coeff = torch.zeros(item_count, heads, cells, device=grad_flat.device, dtype=grad_flat.dtype)
            route_coeff.scatter_add_(2, winner_idx.unsqueeze(-1), grad_margin.unsqueeze(-1))
            route_coeff.scatter_add_(2, runner_idx.unsqueeze(-1), (-grad_margin).unsqueeze(-1))
            grad_router_weight = torch.einsum("nhc,nr->hcr", route_coeff, latent_flat)
            grad_router_bias = route_coeff.sum(dim=0)
            grad_latent = grad_latent + torch.einsum("nhc,hcr->nr", route_coeff, router_weight)

        return grad_latent.view(batch, steps, code_dim), grad_router_weight, grad_router_bias, grad_code, None, None


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
        cpu_param_dtype: Literal["f32", "f16"] = "f32",
        exact_fused: Literal["off", "eval", "train"] = "off",
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

        super().__init__(in_features, out_features, backend=backend, output_scale=1.0)
        self.heads = heads
        self.cells = cells
        self.code_dim = code_dim
        self.code_scale = 1.0 / math.sqrt(heads) if use_output_scaling else 1.0
        self.cpu_param_dtype = cpu_param_dtype
        self.exact_fused = exact_fused
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
            f"exact_fused={self.exact_fused!r}"
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

    def _exact_param_cache_key(self, *, input_device: torch.device, compute_dtype: torch.dtype) -> tuple[object, ...]:
        bias = self.output_proj.bias
        return (
            input_device.type,
            input_device.index,
            compute_dtype,
            self.proj.weight._version,
            self.output_proj.weight._version,
            -1 if bias is None else bias._version,
            self.router_weight._version,
            self.router_bias._version,
            self.code._version,
        )

    def _effective_exact_params(
        self,
        *,
        input_device: torch.device,
        compute_dtype: torch.dtype,
        use_cache: bool,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if use_cache:
            key = self._exact_param_cache_key(input_device=input_device, compute_dtype=compute_dtype)
            cached = self._exact_eval_cache
            if cached is not None and cached[0] == key:
                return cached[1], cached[2], cached[3], cached[4]

        proj_weight = self.proj.weight.to(dtype=compute_dtype, device=input_device)
        output_weight = self.output_proj.weight.to(dtype=compute_dtype, device=input_device)
        router_weight = self.router_weight.to(dtype=compute_dtype, device=input_device)
        router_bias = self.router_bias.to(dtype=compute_dtype, device=input_device)
        code = self.code.to(dtype=compute_dtype, device=input_device)

        router_eff = torch.matmul(router_weight.reshape(self.heads * self.cells, self.code_dim), proj_weight).view(
            self.heads,
            self.cells,
            self.in_features,
        )
        value_eff = torch.matmul(code.reshape(self.heads * self.cells, self.code_dim), output_weight.t()).view(
            self.heads,
            self.cells,
            self.out_features,
        )
        base_weight = torch.matmul(output_weight, proj_weight)

        if use_cache:
            key = self._exact_param_cache_key(input_device=input_device, compute_dtype=compute_dtype)
            self._exact_eval_cache = (key, router_eff, router_bias, value_eff, base_weight)

        return router_eff, router_bias, value_eff, base_weight

    def _exact_head_chunk_size(self, item_count: int, compute_dtype: torch.dtype, *, target_bytes: int = 256 * 1024 * 1024) -> int:
        bytes_per_value = torch.finfo(compute_dtype).bits // 8
        bytes_per_head = item_count * self.out_features * bytes_per_value * (2 if self.training else 1)
        return max(1, min(self.heads, target_bytes // max(1, bytes_per_head)))

    def _forward_exact_fused(self, x: Tensor, *, use_cache: bool) -> Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        input_dtype = x.dtype
        compute_dtype = self._compute_dtype(x)
        input_device = x.device
        x_compute = x.to(dtype=compute_dtype)
        batch, steps, _ = x_compute.shape
        item_count = batch * steps
        x_flat = x_compute.reshape(item_count, self.in_features)

        router_eff, router_bias, value_eff, base_weight = self._effective_exact_params(
            input_device=input_device,
            compute_dtype=compute_dtype,
            use_cache=use_cache,
        )
        scores = torch.matmul(x_flat, router_eff.reshape(self.heads * self.cells, self.in_features).t()).view(
            item_count,
            self.heads,
            self.cells,
        )
        scores = scores + router_bias.view(1, self.heads, self.cells)
        winner_idx, runner_idx, margins = _top2_indices(scores)

        output_flat = torch.matmul(x_flat, base_weight.t())
        bias = self.output_proj.bias
        if bias is not None:
            output_flat = output_flat + bias.to(dtype=compute_dtype, device=input_device)

        value_sum = torch.zeros(item_count, self.out_features, device=input_device, dtype=compute_dtype)
        flat_value = value_eff.reshape(self.heads * self.cells, self.out_features)
        head_chunk = self._exact_head_chunk_size(item_count, compute_dtype)
        route_offsets = torch.arange(self.heads, device=input_device, dtype=torch.long).view(1, self.heads) * self.cells
        linear_winner = winner_idx + route_offsets
        linear_runner = runner_idx + route_offsets

        for head_start in range(0, self.heads, head_chunk):
            head_stop = min(head_start + head_chunk, self.heads)
            winner_values = flat_value.index_select(0, linear_winner[:, head_start:head_stop].reshape(-1)).view(
                item_count,
                head_stop - head_start,
                self.out_features,
            )
            if self.training:
                runner_values = flat_value.index_select(0, linear_runner[:, head_start:head_stop].reshape(-1)).view(
                    item_count,
                    head_stop - head_start,
                    self.out_features,
                )
                values = _minface_mix(winner_values, runner_values, margins[:, head_start:head_stop])
            else:
                values = winner_values
            value_sum = value_sum + values.sum(dim=1)

        output_flat = output_flat + value_sum * self.code_scale
        route_shape = (batch, steps, self.heads)
        self._last_indices = winner_idx.view(route_shape).detach()
        self._last_margins = margins.view(route_shape).detach()
        return output_flat.view(batch, steps, self.out_features).to(dtype=input_dtype)

    def _forward_exact_route(self, x: Tensor) -> Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        input_dtype = x.dtype
        compute_dtype = self._compute_dtype(x)
        latent = self._project_input(x, compute_dtype)
        hidden, winner_idx, margins = _TropExactRouteFunction.apply(
            latent,
            self.router_weight.to(dtype=compute_dtype, device=x.device),
            self.router_bias.to(dtype=compute_dtype, device=x.device),
            self.code.to(dtype=compute_dtype, device=x.device),
            self.code_scale,
            self.training,
        )
        output = self._output_from_hidden(hidden, input_device=x.device, compute_dtype=compute_dtype)
        self._last_indices = winner_idx.detach()
        self._last_margins = margins.detach()
        return output.to(dtype=input_dtype)

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
