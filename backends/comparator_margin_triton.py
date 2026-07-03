from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def has_comparator_margin_triton() -> bool:
    try:
        import triton  # noqa: F401
        import triton.language as tl  # noqa: F401
    except ImportError:
        return False
    return True


try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None


def _block_k(k_c: int) -> int:
    if k_c <= 16:
        return 16
    if k_c <= 32:
        return 32
    if k_c <= 64:
        return 64
    return 128


def _block_d(output_dim: int) -> int:
    if output_dim <= 128:
        return 16
    return 32


if triton is not None:

    @triton.jit
    def _twosided_margin_route_kernel(
        x_ptr,
        anchors_ptr,
        thresholds_ptr,
        margins_ptr,
        ITEMS: tl.constexpr,
        IN_FEATURES: tl.constexpr,
        ROUTES: tl.constexpr,
    ):
        row = tl.program_id(0)
        route = tl.program_id(1)

        a = tl.load(anchors_ptr + route * 2).to(tl.int32)
        b = tl.load(anchors_ptr + route * 2 + 1).to(tl.int32)
        margin = (
            tl.load(x_ptr + row * IN_FEATURES + a).to(tl.float32)
            - tl.load(x_ptr + row * IN_FEATURES + b).to(tl.float32)
            - tl.load(thresholds_ptr + route).to(tl.float32)
        )
        tl.store(margins_ptr + row * ROUTES + route, margin)


    @triton.jit
    def _twosided_margin_fwd_kernel(
        x_ptr,
        anchors_ptr,
        thresholds_ptr,
        write_indices_ptr,
        write_weight_ptr,
        output_ptr,
        margins_ptr,
        ITEMS: tl.constexpr,
        IN_FEATURES: tl.constexpr,
        OUT_FEATURES: tl.constexpr,
        ROUTES: tl.constexpr,
        K_C: tl.constexpr,
        BLOCK_K: tl.constexpr,
        SCALE: tl.constexpr,
    ):
        row = tl.program_id(0)
        route = tl.program_id(1)
        offs = tl.arange(0, BLOCK_K)
        mask = offs < K_C

        a = tl.load(anchors_ptr + route * 2).to(tl.int32)
        b = tl.load(anchors_ptr + route * 2 + 1).to(tl.int32)
        margin = (
            tl.load(x_ptr + row * IN_FEATURES + a).to(tl.float32)
            - tl.load(x_ptr + row * IN_FEATURES + b).to(tl.float32)
            - tl.load(thresholds_ptr + route).to(tl.float32)
        )
        tl.store(margins_ptr + row * ROUTES + route, margin)

        pos = tl.maximum(margin, 0.0)
        neg = tl.maximum(-margin, 0.0)
        base = route * 2 * K_C

        idx_pos = tl.load(write_indices_ptr + base + offs, mask=mask, other=0).to(tl.int32)
        w_pos = tl.load(write_weight_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        tl.atomic_add(output_ptr + row * OUT_FEATURES + idx_pos, pos * w_pos * SCALE, mask=mask)

        idx_neg = tl.load(write_indices_ptr + base + K_C + offs, mask=mask, other=0).to(tl.int32)
        w_neg = tl.load(write_weight_ptr + base + K_C + offs, mask=mask, other=0.0).to(tl.float32)
        tl.atomic_add(output_ptr + row * OUT_FEATURES + idx_neg, neg * w_neg * SCALE, mask=mask)


    @triton.jit
    def _twosided_margin_bwd_kernel(
        grad_out_ptr,
        margins_ptr,
        anchors_ptr,
        write_indices_ptr,
        write_weight_ptr,
        grad_x_ptr,
        grad_thresholds_ptr,
        grad_weight_ptr,
        ITEMS: tl.constexpr,
        IN_FEATURES: tl.constexpr,
        OUT_FEATURES: tl.constexpr,
        ROUTES: tl.constexpr,
        K_C: tl.constexpr,
        BLOCK_K: tl.constexpr,
        SCALE: tl.constexpr,
    ):
        row = tl.program_id(0)
        route = tl.program_id(1)
        offs = tl.arange(0, BLOCK_K)
        mask = offs < K_C

        margin = tl.load(margins_ptr + row * ROUTES + route).to(tl.float32)
        pos = tl.maximum(margin, 0.0)
        neg = tl.maximum(-margin, 0.0)
        base = route * 2 * K_C

        idx_pos = tl.load(write_indices_ptr + base + offs, mask=mask, other=0).to(tl.int32)
        w_pos = tl.load(write_weight_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        go_pos = tl.load(grad_out_ptr + row * OUT_FEATURES + idx_pos, mask=mask, other=0.0).to(tl.float32)
        tl.atomic_add(grad_weight_ptr + base + offs, go_pos * pos * SCALE, mask=mask)

        idx_neg = tl.load(write_indices_ptr + base + K_C + offs, mask=mask, other=0).to(tl.int32)
        w_neg = tl.load(write_weight_ptr + base + K_C + offs, mask=mask, other=0.0).to(tl.float32)
        go_neg = tl.load(grad_out_ptr + row * OUT_FEATURES + idx_neg, mask=mask, other=0.0).to(tl.float32)
        tl.atomic_add(grad_weight_ptr + base + K_C + offs, go_neg * neg * SCALE, mask=mask)

        grad_pos = tl.sum(go_pos * w_pos, axis=0) * SCALE
        grad_neg = tl.sum(go_neg * w_neg, axis=0) * SCALE
        grad_margin = tl.where(margin > 0.0, grad_pos, tl.where(margin < 0.0, -grad_neg, 0.0))

        a = tl.load(anchors_ptr + route * 2).to(tl.int32)
        b = tl.load(anchors_ptr + route * 2 + 1).to(tl.int32)
        tl.atomic_add(grad_x_ptr + row * IN_FEATURES + a, grad_margin)
        tl.atomic_add(grad_x_ptr + row * IN_FEATURES + b, -grad_margin)
        tl.atomic_add(grad_thresholds_ptr + route, -grad_margin)


    @triton.jit
    def _twosided_margin_output_major_fwd_kernel(
        margins_ptr,
        csr_offsets_ptr,
        csr_sources_ptr,
        csr_weight_indices_ptr,
        write_weight_ptr,
        output_ptr,
        ITEMS: tl.constexpr,
        OUT_FEATURES: tl.constexpr,
        ROUTES: tl.constexpr,
        K_C: tl.constexpr,
        BLOCK_D: tl.constexpr,
        MAX_DEGREE: tl.constexpr,
        SCALE: tl.constexpr,
    ):
        row = tl.program_id(0)
        output_block = tl.program_id(1)
        d = output_block * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = d < OUT_FEATURES
        start = tl.load(csr_offsets_ptr + d, mask=mask, other=0).to(tl.int32)
        end = tl.load(csr_offsets_ptr + d + 1, mask=mask, other=0).to(tl.int32)
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

        for j in range(0, MAX_DEGREE):
            entry = start + j
            active = mask & (entry < end)
            source = tl.load(csr_sources_ptr + entry, mask=active, other=0).to(tl.int32)
            weight_idx = tl.load(csr_weight_indices_ptr + entry, mask=active, other=0).to(tl.int32)
            route = source // 2
            side = source - route * 2
            margin = tl.load(margins_ptr + row * ROUTES + route, mask=active, other=0.0).to(tl.float32)
            pos = tl.maximum(margin, 0.0)
            neg = tl.maximum(-margin, 0.0)
            value = tl.where(side == 0, pos, neg)
            weight = tl.load(write_weight_ptr + weight_idx, mask=active, other=0.0).to(tl.float32)
            acc += tl.where(active, value * weight, 0.0)

        tl.store(output_ptr + row * OUT_FEATURES + d, acc * SCALE, mask=mask)


    @triton.jit
    def _twosided_margin_tile_local_fwd_kernel(
        x_ptr,
        anchors_ptr,
        thresholds_ptr,
        write_indices_ptr,
        write_weight_ptr,
        output_ptr,
        margins_ptr,
        ITEMS: tl.constexpr,
        IN_FEATURES: tl.constexpr,
        OUT_FEATURES: tl.constexpr,
        ROUTES: tl.constexpr,
        K_C: tl.constexpr,
        TILE_SIZE: tl.constexpr,
        ROUTES_PER_TILE: tl.constexpr,
        BLOCK_D: tl.constexpr,
        SCALE: tl.constexpr,
    ):
        row = tl.program_id(0)
        tile = tl.program_id(1)
        d_offs = tl.arange(0, BLOCK_D)
        d = tile * TILE_SIZE + d_offs
        d_mask = d < OUT_FEATURES
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

        for local_route in range(0, ROUTES_PER_TILE):
            route = tile * ROUTES_PER_TILE + local_route
            route_active = route < ROUTES
            a = tl.load(anchors_ptr + route * 2, mask=route_active, other=0).to(tl.int32)
            b = tl.load(anchors_ptr + route * 2 + 1, mask=route_active, other=0).to(tl.int32)
            margin = (
                tl.load(x_ptr + row * IN_FEATURES + a, mask=route_active, other=0.0).to(tl.float32)
                - tl.load(x_ptr + row * IN_FEATURES + b, mask=route_active, other=0.0).to(tl.float32)
                - tl.load(thresholds_ptr + route, mask=route_active, other=0.0).to(tl.float32)
            )
            tl.store(margins_ptr + row * ROUTES + route, margin, mask=route_active)
            pos = tl.maximum(margin, 0.0)
            neg = tl.maximum(-margin, 0.0)

            for slot in range(0, K_C):
                pos_base = route * 2 * K_C + slot
                pos_idx = tl.load(write_indices_ptr + pos_base, mask=route_active, other=-1).to(tl.int32)
                pos_w = tl.load(write_weight_ptr + pos_base, mask=route_active, other=0.0).to(tl.float32)
                acc += tl.where(d_mask & route_active & (d == pos_idx), pos * pos_w, 0.0)

                neg_base = route * 2 * K_C + K_C + slot
                neg_idx = tl.load(write_indices_ptr + neg_base, mask=route_active, other=-1).to(tl.int32)
                neg_w = tl.load(write_weight_ptr + neg_base, mask=route_active, other=0.0).to(tl.float32)
                acc += tl.where(d_mask & route_active & (d == neg_idx), neg * neg_w, 0.0)

        tl.store(output_ptr + row * OUT_FEATURES + d, acc * SCALE, mask=d_mask)


class _TwoSidedMarginTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        latent: Tensor,
        anchors: Tensor,
        thresholds: Tensor,
        write_indices: Tensor,
        write_weight: Tensor,
        output_dim: int,
        output_scale: float,
    ) -> tuple[Tensor, Tensor]:
        if triton is None:
            raise RuntimeError("Triton is not installed; use backend='torch'")
        if not latent.is_cuda:
            raise ValueError("two-sided margin Triton backend requires CUDA tensors")
        if latent.dtype != torch.float32 or thresholds.dtype != torch.float32:
            raise TypeError("two-sided margin Triton backend expects float32 compute tensors")

        items, in_features = latent.shape
        tables, comparisons, pair_width = anchors.shape
        if pair_width != 2:
            raise ValueError(f"anchors must have shape [tables, comparisons, 2], got {tuple(anchors.shape)}")
        routes = tables * comparisons
        if write_indices.shape[:2] != (routes, 2) or write_weight.shape[:2] != (routes, 2):
            raise ValueError("write_indices and write_weight must have shape [tables*comparisons, 2, k_c]")
        k_c = int(write_indices.shape[-1])
        if write_weight.shape[-1] != k_c:
            raise ValueError("write_indices and write_weight must use the same k_c")

        anchors_flat = anchors.contiguous().reshape(-1).to(device=latent.device, dtype=torch.int64)
        thresholds_flat = thresholds.contiguous().reshape(-1).to(device=latent.device, dtype=torch.float32)
        write_indices_flat = write_indices.contiguous().reshape(-1).to(device=latent.device, dtype=torch.int64)
        write_weight_flat = write_weight.contiguous().reshape(-1).to(device=latent.device, dtype=torch.float32)

        output = torch.zeros(items, int(output_dim), device=latent.device, dtype=torch.float32)
        margins = torch.empty(items, routes, device=latent.device, dtype=torch.float32)
        block_k = _block_k(k_c)

        _twosided_margin_fwd_kernel[(items, routes)](
            latent.contiguous(),
            anchors_flat,
            thresholds_flat,
            write_indices_flat,
            write_weight_flat,
            output,
            margins,
            items,
            in_features,
            int(output_dim),
            routes,
            k_c,
            BLOCK_K=block_k,
            SCALE=float(output_scale),
        )

        ctx.save_for_backward(margins, anchors_flat, write_indices_flat, write_weight_flat)
        ctx.input_shape = tuple(latent.shape)
        ctx.threshold_shape = tuple(thresholds.shape)
        ctx.weight_shape = tuple(write_weight.shape)
        ctx.output_dim = int(output_dim)
        ctx.output_scale = float(output_scale)
        ctx.k_c = k_c
        ctx.mark_non_differentiable(margins)
        return output, margins

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor, grad_margins: Tensor | None) -> tuple[Any, ...]:
        del grad_margins
        margins, anchors_flat, write_indices_flat, write_weight_flat = ctx.saved_tensors
        items, in_features = ctx.input_shape
        routes = int(margins.shape[-1])
        k_c = int(ctx.k_c)
        grad_flat = grad_output.reshape(items, int(ctx.output_dim)).contiguous().to(torch.float32)
        grad_x = torch.zeros(items, in_features, device=grad_output.device, dtype=torch.float32)
        grad_thresholds = torch.zeros(routes, device=grad_output.device, dtype=torch.float32)
        grad_weight = torch.zeros(int(write_weight_flat.numel()), device=grad_output.device, dtype=torch.float32)
        block_k = _block_k(k_c)

        _twosided_margin_bwd_kernel[(items, routes)](
            grad_flat,
            margins.contiguous(),
            anchors_flat,
            write_indices_flat,
            write_weight_flat,
            grad_x,
            grad_thresholds,
            grad_weight,
            items,
            in_features,
            int(ctx.output_dim),
            routes,
            k_c,
            BLOCK_K=block_k,
            SCALE=float(ctx.output_scale),
        )

        return grad_x, None, grad_thresholds.view(ctx.threshold_shape), None, grad_weight.view(ctx.weight_shape), None, None


class _TwoSidedMarginOutputMajorTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        latent: Tensor,
        anchors: Tensor,
        thresholds: Tensor,
        write_indices: Tensor,
        write_weight: Tensor,
        csr_offsets: Tensor,
        csr_sources: Tensor,
        csr_weight_indices: Tensor,
        csr_max_degree: int,
        output_dim: int,
        output_scale: float,
    ) -> tuple[Tensor, Tensor]:
        if triton is None:
            raise RuntimeError("Triton is not installed; use backend='torch'")
        if not latent.is_cuda:
            raise ValueError("two-sided margin Triton backend requires CUDA tensors")
        if latent.dtype != torch.float32 or thresholds.dtype != torch.float32:
            raise TypeError("two-sided margin Triton backend expects float32 compute tensors")

        items, in_features = latent.shape
        tables, comparisons, pair_width = anchors.shape
        if pair_width != 2:
            raise ValueError(f"anchors must have shape [tables, comparisons, 2], got {tuple(anchors.shape)}")
        routes = tables * comparisons
        if write_indices.shape[:2] != (routes, 2) or write_weight.shape[:2] != (routes, 2):
            raise ValueError("write_indices and write_weight must have shape [tables*comparisons, 2, k_c]")
        k_c = int(write_indices.shape[-1])
        if write_weight.shape[-1] != k_c:
            raise ValueError("write_indices and write_weight must use the same k_c")
        if int(csr_offsets.numel()) != int(output_dim) + 1:
            raise ValueError("csr_offsets must have shape [output_dim + 1]")
        if int(csr_sources.numel()) != int(csr_weight_indices.numel()):
            raise ValueError("csr_sources and csr_weight_indices must have the same length")

        anchors_flat = anchors.contiguous().reshape(-1).to(device=latent.device, dtype=torch.int64)
        thresholds_flat = thresholds.contiguous().reshape(-1).to(device=latent.device, dtype=torch.float32)
        write_indices_flat = write_indices.contiguous().reshape(-1).to(device=latent.device, dtype=torch.int64)
        write_weight_flat = write_weight.contiguous().reshape(-1).to(device=latent.device, dtype=torch.float32)
        csr_offsets_flat = csr_offsets.contiguous().to(device=latent.device, dtype=torch.int64)
        csr_sources_flat = csr_sources.contiguous().to(device=latent.device, dtype=torch.int64)
        csr_weight_indices_flat = csr_weight_indices.contiguous().to(device=latent.device, dtype=torch.int64)

        output = torch.empty(items, int(output_dim), device=latent.device, dtype=torch.float32)
        margins = torch.empty(items, routes, device=latent.device, dtype=torch.float32)
        block_d = _block_d(int(output_dim))

        _twosided_margin_route_kernel[(items, routes)](
            latent.contiguous(),
            anchors_flat,
            thresholds_flat,
            margins,
            items,
            in_features,
            routes,
        )
        _twosided_margin_output_major_fwd_kernel[(items, triton.cdiv(int(output_dim), block_d))](
            margins,
            csr_offsets_flat,
            csr_sources_flat,
            csr_weight_indices_flat,
            write_weight_flat,
            output,
            items,
            int(output_dim),
            routes,
            k_c,
            BLOCK_D=block_d,
            MAX_DEGREE=int(csr_max_degree),
            SCALE=float(output_scale),
        )

        ctx.save_for_backward(margins, anchors_flat, write_indices_flat, write_weight_flat)
        ctx.input_shape = tuple(latent.shape)
        ctx.threshold_shape = tuple(thresholds.shape)
        ctx.weight_shape = tuple(write_weight.shape)
        ctx.output_dim = int(output_dim)
        ctx.output_scale = float(output_scale)
        ctx.k_c = k_c
        ctx.mark_non_differentiable(margins)
        return output, margins

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor, grad_margins: Tensor | None) -> tuple[Any, ...]:
        del grad_margins
        margins, anchors_flat, write_indices_flat, write_weight_flat = ctx.saved_tensors
        items, in_features = ctx.input_shape
        routes = int(margins.shape[-1])
        k_c = int(ctx.k_c)
        grad_flat = grad_output.reshape(items, int(ctx.output_dim)).contiguous().to(torch.float32)
        grad_x = torch.zeros(items, in_features, device=grad_output.device, dtype=torch.float32)
        grad_thresholds = torch.zeros(routes, device=grad_output.device, dtype=torch.float32)
        grad_weight = torch.zeros(int(write_weight_flat.numel()), device=grad_output.device, dtype=torch.float32)
        block_k = _block_k(k_c)

        _twosided_margin_bwd_kernel[(items, routes)](
            grad_flat,
            margins.contiguous(),
            anchors_flat,
            write_indices_flat,
            write_weight_flat,
            grad_x,
            grad_thresholds,
            grad_weight,
            items,
            in_features,
            int(ctx.output_dim),
            routes,
            k_c,
            BLOCK_K=block_k,
            SCALE=float(ctx.output_scale),
        )

        return (
            grad_x,
            None,
            grad_thresholds.view(ctx.threshold_shape),
            None,
            grad_weight.view(ctx.weight_shape),
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _TwoSidedMarginTileLocalTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        latent: Tensor,
        anchors: Tensor,
        thresholds: Tensor,
        write_indices: Tensor,
        write_weight: Tensor,
        output_tile_size: int,
        output_dim: int,
        output_scale: float,
    ) -> tuple[Tensor, Tensor]:
        if triton is None:
            raise RuntimeError("Triton is not installed; use backend='torch'")
        if not latent.is_cuda:
            raise ValueError("two-sided margin Triton backend requires CUDA tensors")
        if latent.dtype != torch.float32 or thresholds.dtype != torch.float32:
            raise TypeError("two-sided margin Triton backend expects float32 compute tensors")

        items, in_features = latent.shape
        tables, comparisons, pair_width = anchors.shape
        if pair_width != 2:
            raise ValueError(f"anchors must have shape [tables, comparisons, 2], got {tuple(anchors.shape)}")
        routes = tables * comparisons
        if write_indices.shape[:2] != (routes, 2) or write_weight.shape[:2] != (routes, 2):
            raise ValueError("write_indices and write_weight must have shape [tables*comparisons, 2, k_c]")
        k_c = int(write_indices.shape[-1])
        if write_weight.shape[-1] != k_c:
            raise ValueError("write_indices and write_weight must use the same k_c")

        anchors_flat = anchors.contiguous().reshape(-1).to(device=latent.device, dtype=torch.int64)
        thresholds_flat = thresholds.contiguous().reshape(-1).to(device=latent.device, dtype=torch.float32)
        write_indices_flat = write_indices.contiguous().reshape(-1).to(device=latent.device, dtype=torch.int64)
        write_weight_flat = write_weight.contiguous().reshape(-1).to(device=latent.device, dtype=torch.float32)

        output_tile_size = int(output_tile_size)
        output_dim = int(output_dim)
        tiles = triton.cdiv(output_dim, output_tile_size)
        routes_per_tile = triton.cdiv(routes, tiles)
        output = torch.empty(items, output_dim, device=latent.device, dtype=torch.float32)
        margins = torch.empty(items, routes, device=latent.device, dtype=torch.float32)

        _twosided_margin_tile_local_fwd_kernel[(items, tiles)](
            latent.contiguous(),
            anchors_flat,
            thresholds_flat,
            write_indices_flat,
            write_weight_flat,
            output,
            margins,
            items,
            in_features,
            output_dim,
            routes,
            k_c,
            TILE_SIZE=output_tile_size,
            ROUTES_PER_TILE=routes_per_tile,
            BLOCK_D=output_tile_size,
            SCALE=float(output_scale),
        )

        ctx.save_for_backward(margins, anchors_flat, write_indices_flat, write_weight_flat)
        ctx.input_shape = tuple(latent.shape)
        ctx.threshold_shape = tuple(thresholds.shape)
        ctx.weight_shape = tuple(write_weight.shape)
        ctx.output_dim = output_dim
        ctx.output_scale = float(output_scale)
        ctx.k_c = k_c
        ctx.mark_non_differentiable(margins)
        return output, margins

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor, grad_margins: Tensor | None) -> tuple[Any, ...]:
        del grad_margins
        margins, anchors_flat, write_indices_flat, write_weight_flat = ctx.saved_tensors
        items, in_features = ctx.input_shape
        routes = int(margins.shape[-1])
        k_c = int(ctx.k_c)
        grad_flat = grad_output.reshape(items, int(ctx.output_dim)).contiguous().to(torch.float32)
        grad_x = torch.zeros(items, in_features, device=grad_output.device, dtype=torch.float32)
        grad_thresholds = torch.zeros(routes, device=grad_output.device, dtype=torch.float32)
        grad_weight = torch.zeros(int(write_weight_flat.numel()), device=grad_output.device, dtype=torch.float32)
        block_k = _block_k(k_c)

        _twosided_margin_bwd_kernel[(items, routes)](
            grad_flat,
            margins.contiguous(),
            anchors_flat,
            write_indices_flat,
            write_weight_flat,
            grad_x,
            grad_thresholds,
            grad_weight,
            items,
            in_features,
            int(ctx.output_dim),
            routes,
            k_c,
            BLOCK_K=block_k,
            SCALE=float(ctx.output_scale),
        )

        return grad_x, None, grad_thresholds.view(ctx.threshold_shape), None, grad_weight.view(ctx.weight_shape), None, None, None


def comparator_two_sided_margin_triton(
    latent: Tensor,
    anchors: Tensor,
    thresholds: Tensor,
    write_indices: Tensor,
    write_weight: Tensor,
    *,
    output_dim: int,
    output_scale: float,
) -> tuple[Tensor, Tensor]:
    return _TwoSidedMarginTritonFunction.apply(
        latent,
        anchors,
        thresholds,
        write_indices,
        write_weight,
        int(output_dim),
        float(output_scale),
    )


def comparator_two_sided_margin_output_major_triton(
    latent: Tensor,
    anchors: Tensor,
    thresholds: Tensor,
    write_indices: Tensor,
    write_weight: Tensor,
    csr_offsets: Tensor,
    csr_sources: Tensor,
    csr_weight_indices: Tensor,
    *,
    csr_max_degree: int,
    output_dim: int,
    output_scale: float,
) -> tuple[Tensor, Tensor]:
    return _TwoSidedMarginOutputMajorTritonFunction.apply(
        latent,
        anchors,
        thresholds,
        write_indices,
        write_weight,
        csr_offsets,
        csr_sources,
        csr_weight_indices,
        int(csr_max_degree),
        int(output_dim),
        float(output_scale),
    )


def comparator_two_sided_margin_tile_local_triton(
    latent: Tensor,
    anchors: Tensor,
    thresholds: Tensor,
    write_indices: Tensor,
    write_weight: Tensor,
    *,
    output_tile_size: int,
    output_dim: int,
    output_scale: float,
) -> tuple[Tensor, Tensor]:
    return _TwoSidedMarginTileLocalTritonFunction.apply(
        latent,
        anchors,
        thresholds,
        write_indices,
        write_weight,
        int(output_tile_size),
        int(output_dim),
        float(output_scale),
    )
