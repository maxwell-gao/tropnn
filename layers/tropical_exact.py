from __future__ import annotations

import torch
from torch import Tensor

from .routing import (
    minface_mix,
    pack_route_indices,
    recompute_route_margins_torch,
    top2_indices,
    unpack_route_indices,
)

_CUDA_TORCH_FALLBACK_SCORE_BYTES = 128 * 1024 * 1024


def _route_score_bytes(item_count: int, heads: int, cells: int) -> int:
    return item_count * heads * cells * 4


def _can_cuda_torch_route_fallback(item_count: int, heads: int, cells: int) -> bool:
    return _route_score_bytes(item_count, heads, cells) <= _CUDA_TORCH_FALLBACK_SCORE_BYTES


def _is_hard_cuda_failure(exc: BaseException) -> bool:
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "out of memory",
            "cuda error",
            "device-side assert",
            "illegal memory",
            "misaligned address",
            "unspecified launch failure",
            "invalid configuration argument",
            "too many resources requested",
        )
    )


def _raise_no_cuda_torch_fallback(exc: BaseException, item_count: int, heads: int, cells: int) -> None:
    score_bytes = _route_score_bytes(item_count, heads, cells)
    raise RuntimeError(
        "TropLinear exact_fused='train' CUDA route failed and torch fallback was disabled because it would "
        f"materialize a full score tensor ({score_bytes} bytes for items={item_count}, heads={heads}, cells={cells})."
    ) from exc


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
        score_route_max_bytes: int | None,
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
            except (ImportError, RuntimeError, OSError) as exc:
                if not _can_cuda_torch_route_fallback(item_count, heads, cells):
                    _raise_no_cuda_torch_fallback(exc, item_count, heads, cells)
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
                        score_route_max_bytes=score_route_max_bytes,
                    )
                except RuntimeError as exc:
                    if _is_hard_cuda_failure(exc) or not _can_cuda_torch_route_fallback(item_count, heads, cells):
                        _raise_no_cuda_torch_fallback(exc, item_count, heads, cells)
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

        winner_saved = pack_route_indices(winner_idx, cells)
        runner_saved = pack_route_indices(runner_idx, cells)
        ctx.save_for_backward(latent_flat, weight, bias, code_table, winner_saved, runner_saved)
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
        winner_idx, runner_idx, margins = top2_indices(scores)

        route_offsets = torch.arange(heads, device=latent_flat.device, dtype=torch.long).view(1, heads) * cells
        flat_code = code_table.reshape(heads * cells, code_dim)
        winner_values = flat_code.index_select(0, (winner_idx + route_offsets).reshape(-1)).view(item_count, heads, code_dim)
        if training:
            runner_values = flat_code.index_select(0, (runner_idx + route_offsets).reshape(-1)).view(item_count, heads, code_dim)
            values = minface_mix(winner_values, runner_values, margins)
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
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, None, None, None]:
        del grad_winner, grad_margins_out
        latent_flat, router_weight, router_bias, code, winner_saved, runner_saved = ctx.saved_tensors
        batch, steps, code_dim, heads, cells = ctx.shape
        item_count = batch * steps
        grad_flat = grad_hidden.reshape(item_count, code_dim).contiguous().to(torch.float32)

        if ctx.training and grad_flat.is_cuda:
            try:
                from ..backends.triton_backward import trop_exact_route_backward_triton
            except (ImportError, RuntimeError, OSError) as exc:
                if not _can_cuda_torch_route_fallback(item_count, heads, cells):
                    _raise_no_cuda_torch_fallback(exc, item_count, heads, cells)
            else:
                grad_latent, grad_router_weight, grad_router_bias, grad_code = trop_exact_route_backward_triton(
                    grad_flat,
                    latent_flat,
                    router_weight,
                    router_bias,
                    code,
                    winner_saved,
                    runner_saved,
                    code_scale=ctx.code_scale,
                )
                return (
                    grad_latent.view(batch, steps, code_dim),
                    grad_router_weight,
                    grad_router_bias,
                    grad_code,
                    None,
                    None,
                    None,
                )

        winner_idx = unpack_route_indices(winner_saved)
        runner_idx = unpack_route_indices(runner_saved)
        margins = recompute_route_margins_torch(latent_flat, router_weight, router_bias, winner_idx, runner_idx)

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

        return grad_latent.view(batch, steps, code_dim), grad_router_weight, grad_router_bias, grad_code, None, None, None


class TropLinearExactMixin:
    """Exact fused train/eval paths for ``TropLinear``."""

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
        winner_idx, runner_idx, margins = top2_indices(scores)

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
                values = minface_mix(winner_values, runner_values, margins[:, head_start:head_stop])
            else:
                values = winner_values
            value_sum = value_sum + values.sum(dim=1)

        output_flat = output_flat + value_sum * self.code_scale
        route_shape = (batch, steps, self.heads)
        if self.cache_route_debug:
            self._last_indices = winner_idx.view(route_shape).detach()
            self._last_margins = margins.view(route_shape).detach()
        else:
            self._last_indices = None
            self._last_margins = None
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
            self.score_route_max_bytes,
        )
        output = self._output_from_hidden(hidden, input_device=x.device, compute_dtype=compute_dtype)
        if self.cache_route_debug:
            self._last_indices = winner_idx.detach()
            self._last_margins = margins.detach()
        else:
            self._last_indices = None
            self._last_margins = None
        return output.to(dtype=input_dtype)
