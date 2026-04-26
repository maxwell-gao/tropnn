from functools import lru_cache
from typing import Any

import torch
from torch import Tensor

from .tilelang_route import has_tilelang


def _next_power_of_2(value: int) -> int:
    if value < 1:
        return 1
    return 1 << (value - 1).bit_length()


def _select_block_size(value: int, *, min_block: int = 32, max_block: int = 256) -> int:
    return max(min_block, min(max_block, _next_power_of_2(value)))


def _require_float32(*tensors: Tensor) -> None:
    for tensor in tensors:
        if tensor.dtype != torch.float32:
            raise TypeError(f"Pairwise TileLang backend currently expects float32 compute tensors, got {tensor.dtype}")


@lru_cache(maxsize=64)
def _pairwise_route_kernel(
    item_count: int,
    in_features: int,
    tables: int,
    comparisons: int,
    route_block: int,
    table_blocks: int,
    target: str,
) -> Any:
    try:
        import tilelang
        import tilelang.language as T
    except ImportError as exc:
        raise RuntimeError("TileLang is not installed; install tilelang or use backend='torch'") from exc

    @tilelang.jit(target=target, compile_flags=["-allow-unsupported-compiler", "-ccbin=/usr/bin/g++"])
    def route_kernel() -> Any:
        input_dim = in_features
        route_count = tables
        comp_count = comparisons
        route_width = route_block
        route_tiles = table_blocks

        @T.prim_func
        def kernel(
            latent: T.Tensor((item_count, input_dim), "float32"),
            anchors: T.Tensor((route_count, comp_count, 2), "int64"),
            thresholds: T.Tensor((route_count, comp_count), "float32"),
            indices: T.Tensor((item_count, route_count), "int64"),
            margins: T.Tensor((item_count, route_count, comp_count), "float32"),
        ):
            with T.Kernel(item_count, threads=route_width) as row:
                tx = T.get_thread_bindings()[0]
                idx = T.alloc_fragment((1,), "int32")
                power = T.alloc_fragment((1,), "int32")
                for table_tile in T.serial(route_tiles):
                    table = table_tile * route_width + tx
                    if table < route_count:
                        idx[0] = 0
                        power[0] = 1
                        for comp in T.serial(comp_count):
                            a = anchors[table, comp, 0]
                            b = anchors[table, comp, 1]
                            margin = latent[row, a] - latent[row, b] - thresholds[table, comp]
                            margins[row, table, comp] = margin
                            if margin > 0.0:
                                idx[0] = idx[0] + power[0]
                            power[0] = power[0] * 2
                        indices[row, table] = idx[0]

        return kernel

    return route_kernel()


@lru_cache(maxsize=64)
def _pairwise_forward_block_kernel(
    item_count: int,
    out_features: int,
    tables: int,
    table_size: int,
    block_d: int,
    out_blocks: int,
    target: str,
) -> Any:
    try:
        import tilelang
        import tilelang.language as T
    except ImportError as exc:
        raise RuntimeError("TileLang is not installed; install tilelang or use backend='torch'") from exc

    @tilelang.jit(target=target, compile_flags=["-allow-unsupported-compiler", "-ccbin=/usr/bin/g++"])
    def forward_kernel() -> Any:
        route_count = tables
        bucket_count = table_size
        output_dim = out_features
        block_width = block_d

        @T.prim_func
        def kernel(
            indices: T.Tensor((item_count, route_count), "int64"),
            lut: T.Tensor((route_count, bucket_count, output_dim), "float32"),
            output: T.Tensor((item_count, output_dim), "float32"),
        ):
            with T.Kernel(item_count, out_blocks, threads=block_width) as (row, out_tile):
                tx = T.get_thread_bindings()[0]
                out_col = out_tile * block_width + tx
                acc = T.alloc_fragment((1,), "float32")
                if out_col < output_dim:
                    acc[0] = 0.0
                    for table in T.serial(route_count):
                        idx = T.cast(indices[row, table], "int32")
                        acc[0] = acc[0] + lut[table, idx, out_col]
                    output[row, out_col] = acc[0]

        return kernel

    return forward_kernel()


@lru_cache(maxsize=64)
def _pairwise_lut_backward_block_kernel(
    item_count: int,
    out_features: int,
    tables: int,
    table_size: int,
    block_d: int,
    out_blocks: int,
    target: str,
) -> Any:
    try:
        import tilelang
        import tilelang.language as T
    except ImportError as exc:
        raise RuntimeError("TileLang is not installed; install tilelang or use backend='torch'") from exc

    @tilelang.jit(target=target, compile_flags=["-allow-unsupported-compiler", "-ccbin=/usr/bin/g++"])
    def lut_backward_kernel() -> Any:
        route_count = tables
        bucket_count = table_size
        output_dim = out_features
        block_width = block_d

        @T.prim_func
        def kernel(
            grad_output: T.Tensor((item_count, output_dim), "float32"),
            indices: T.Tensor((item_count, route_count), "int64"),
            grad_lut: T.Tensor((route_count, bucket_count, output_dim), "float32"),
        ):
            with T.Kernel(item_count, route_count, out_blocks, threads=block_width) as (row, table, out_tile):
                tx = T.get_thread_bindings()[0]
                out_col = out_tile * block_width + tx
                if out_col < output_dim:
                    idx = T.cast(indices[row, table], "int32")
                    T.atomic_add(grad_lut[table, idx, out_col], grad_output[row, out_col])

        return kernel

    return lut_backward_kernel()


@lru_cache(maxsize=64)
def _pairwise_min_backward_kernel(
    item_count: int,
    in_features: int,
    out_features: int,
    tables: int,
    comparisons: int,
    table_size: int,
    block_d: int,
    out_blocks: int,
    target: str,
) -> Any:
    try:
        import tilelang
        import tilelang.language as T
    except ImportError as exc:
        raise RuntimeError("TileLang is not installed; install tilelang or use backend='torch'") from exc

    @tilelang.jit(target=target, compile_flags=["-allow-unsupported-compiler", "-ccbin=/usr/bin/g++"])
    def min_backward_kernel() -> Any:
        input_dim = in_features
        output_dim = out_features
        route_count = tables
        comp_count = comparisons
        bucket_count = table_size
        block_width = block_d
        output_tiles = out_blocks

        @T.prim_func
        def kernel(
            grad_output: T.Tensor((item_count, output_dim), "float32"),
            indices: T.Tensor((item_count, route_count), "int64"),
            margins: T.Tensor((item_count, route_count, comp_count), "float32"),
            anchors: T.Tensor((route_count, comp_count, 2), "int64"),
            lut: T.Tensor((route_count, bucket_count, output_dim), "float32"),
            grad_latent: T.Tensor((item_count, input_dim), "float32"),
            grad_thresholds: T.Tensor((route_count, comp_count), "float32"),
        ):
            with T.Kernel(item_count, route_count, threads=block_width) as (row, table):
                tx = T.get_thread_bindings()[0]
                partial = T.alloc_shared((block_width,), "float32")
                r_min = T.alloc_fragment((1,), "int32")
                min_abs = T.alloc_fragment((1,), "float32")
                r_min[0] = 0
                min_abs[0] = T.abs(margins[row, table, 0])
                for comp in T.serial(1, comp_count):
                    abs_margin = T.abs(margins[row, table, comp])
                    if abs_margin < min_abs[0]:
                        min_abs[0] = abs_margin
                        r_min[0] = comp

                current_idx = T.cast(indices[row, table], "int32")
                power = T.alloc_fragment((1,), "int32")
                power[0] = 1
                for comp in T.serial(comp_count):
                    if comp < r_min[0]:
                        power[0] = power[0] * 2
                neighbor_idx = T.bitwise_xor(current_idx, power[0])

                dot = T.alloc_fragment((1,), "float32")
                dot[0] = 0.0
                for out_tile in T.serial(output_tiles):
                    out_col = out_tile * block_width + tx
                    if out_col < output_dim:
                        delta = lut[table, neighbor_idx, out_col] - lut[table, current_idx, out_col]
                        dot[0] = dot[0] + grad_output[row, out_col] * delta
                partial[tx] = dot[0]
                T.sync_threads()
                if block_width >= 256:
                    if tx < 128:
                        partial[tx] = partial[tx] + partial[tx + 128]
                    T.sync_threads()
                if block_width >= 128:
                    if tx < 64:
                        partial[tx] = partial[tx] + partial[tx + 64]
                    T.sync_threads()
                if block_width >= 64:
                    if tx < 32:
                        partial[tx] = partial[tx] + partial[tx + 32]
                    T.sync_threads()
                if tx < 16:
                    partial[tx] = partial[tx] + partial[tx + 16]
                T.sync_threads()
                if tx < 8:
                    partial[tx] = partial[tx] + partial[tx + 8]
                T.sync_threads()
                if tx < 4:
                    partial[tx] = partial[tx] + partial[tx + 4]
                T.sync_threads()
                if tx < 2:
                    partial[tx] = partial[tx] + partial[tx + 2]
                T.sync_threads()
                if tx < 1:
                    partial[tx] = partial[tx] + partial[tx + 1]
                T.sync_threads()

                if tx == 0:
                    u = margins[row, table, r_min[0]]
                    denom = (1.0 + T.abs(u)) * (1.0 + T.abs(u))
                    surr = T.alloc_fragment((1,), "float32")
                    surr[0] = 0.0
                    if u > 0.0:
                        surr[0] = -0.5 / denom
                    else:
                        if u < 0.0:
                            surr[0] = 0.5 / denom

                    grad_margin = partial[0] * surr[0]
                    a = anchors[table, r_min[0], 0]
                    b = anchors[table, r_min[0], 1]
                    T.atomic_add(grad_latent[row, a], grad_margin)
                    T.atomic_add(grad_latent[row, b], -grad_margin)
                    T.atomic_add(grad_thresholds[table, r_min[0]], -grad_margin)

        return kernel

    return min_backward_kernel()


@lru_cache(maxsize=64)
def _pairwise_full_backward_kernel(
    item_count: int,
    in_features: int,
    out_features: int,
    tables: int,
    comparisons: int,
    table_size: int,
    block_d: int,
    out_blocks: int,
    target: str,
) -> Any:
    try:
        import tilelang
        import tilelang.language as T
    except ImportError as exc:
        raise RuntimeError("TileLang is not installed; install tilelang or use backend='torch'") from exc

    @tilelang.jit(target=target, compile_flags=["-allow-unsupported-compiler", "-ccbin=/usr/bin/g++"])
    def full_backward_kernel() -> Any:
        input_dim = in_features
        output_dim = out_features
        route_count = tables
        comp_count = comparisons
        bucket_count = table_size
        block_width = block_d
        output_tiles = out_blocks

        @T.prim_func
        def kernel(
            grad_output: T.Tensor((item_count, output_dim), "float32"),
            indices: T.Tensor((item_count, route_count), "int64"),
            margins: T.Tensor((item_count, route_count, comp_count), "float32"),
            anchors: T.Tensor((route_count, comp_count, 2), "int64"),
            lut: T.Tensor((route_count, bucket_count, output_dim), "float32"),
            grad_latent: T.Tensor((item_count, input_dim), "float32"),
            grad_thresholds: T.Tensor((route_count, comp_count), "float32"),
        ):
            with T.Kernel(item_count, route_count, comp_count, threads=block_width) as (row, table, comp):
                tx = T.get_thread_bindings()[0]
                partial = T.alloc_shared((block_width,), "float32")
                current_idx = T.cast(indices[row, table], "int32")
                power = T.alloc_fragment((1,), "int32")
                power[0] = 1
                for c in T.serial(comp_count):
                    if c < comp:
                        power[0] = power[0] * 2
                neighbor_idx = T.bitwise_xor(current_idx, power[0])

                dot = T.alloc_fragment((1,), "float32")
                dot[0] = 0.0
                for out_tile in T.serial(output_tiles):
                    out_col = out_tile * block_width + tx
                    if out_col < output_dim:
                        delta = lut[table, neighbor_idx, out_col] - lut[table, current_idx, out_col]
                        dot[0] = dot[0] + grad_output[row, out_col] * delta
                partial[tx] = dot[0]
                T.sync_threads()
                if block_width >= 256:
                    if tx < 128:
                        partial[tx] = partial[tx] + partial[tx + 128]
                    T.sync_threads()
                if block_width >= 128:
                    if tx < 64:
                        partial[tx] = partial[tx] + partial[tx + 64]
                    T.sync_threads()
                if block_width >= 64:
                    if tx < 32:
                        partial[tx] = partial[tx] + partial[tx + 32]
                    T.sync_threads()
                if tx < 16:
                    partial[tx] = partial[tx] + partial[tx + 16]
                T.sync_threads()
                if tx < 8:
                    partial[tx] = partial[tx] + partial[tx + 8]
                T.sync_threads()
                if tx < 4:
                    partial[tx] = partial[tx] + partial[tx + 4]
                T.sync_threads()
                if tx < 2:
                    partial[tx] = partial[tx] + partial[tx + 2]
                T.sync_threads()
                if tx < 1:
                    partial[tx] = partial[tx] + partial[tx + 1]
                T.sync_threads()

                if tx == 0:
                    u = margins[row, table, comp]
                    denom = (1.0 + T.abs(u)) * (1.0 + T.abs(u))
                    surr = T.alloc_fragment((1,), "float32")
                    surr[0] = 0.0
                    if u > 0.0:
                        surr[0] = -0.5 / denom
                    else:
                        if u < 0.0:
                            surr[0] = 0.5 / denom

                    grad_margin = partial[0] * surr[0]
                    a = anchors[table, comp, 0]
                    b = anchors[table, comp, 1]
                    T.atomic_add(grad_latent[row, a], grad_margin)
                    T.atomic_add(grad_latent[row, b], -grad_margin)
                    T.atomic_add(grad_thresholds[table, comp], -grad_margin)

        return kernel

    return full_backward_kernel()


def _run_forward(
    latent: Tensor,
    anchors: Tensor,
    thresholds: Tensor,
    lut: Tensor,
    *,
    target: str,
) -> tuple[Tensor, Tensor, Tensor]:
    if not has_tilelang():
        raise RuntimeError("TileLang is not installed; install tilelang or use backend='torch'")
    if not latent.is_cuda:
        raise ValueError("Pairwise TileLang backend requires CUDA tensors")
    if latent.ndim != 3:
        raise ValueError(f"latent must have shape [batch, steps, in_features], got {tuple(latent.shape)}")
    _require_float32(latent, thresholds, lut)

    batch, steps, in_features = latent.shape
    tables, comparisons, pair_width = anchors.shape
    if pair_width != 2:
        raise ValueError(f"anchors must have shape [tables, comparisons, 2], got {tuple(anchors.shape)}")
    if thresholds.shape != (tables, comparisons):
        raise ValueError(f"thresholds must have shape {(tables, comparisons)}, got {tuple(thresholds.shape)}")
    if lut.ndim != 3 or lut.shape[0] != tables:
        raise ValueError(f"lut must have shape [tables, table_size, out_features], got {tuple(lut.shape)}")

    item_count = batch * steps
    table_size = lut.shape[1]
    out_features = lut.shape[2]
    latent_flat = latent.reshape(item_count, in_features).contiguous()
    anchors_contig = anchors.contiguous()
    thresholds_contig = thresholds.contiguous()
    lut_contig = lut.contiguous()
    indices = torch.empty((item_count, tables), device=latent.device, dtype=torch.int64)
    margins = torch.empty((item_count, tables, comparisons), device=latent.device, dtype=torch.float32)
    output = torch.empty((item_count, out_features), device=latent.device, dtype=torch.float32)

    route_block = _select_block_size(tables)
    table_blocks = (tables + route_block - 1) // route_block
    block_d = _select_block_size(out_features)
    out_blocks = (out_features + block_d - 1) // block_d
    route_kernel = _pairwise_route_kernel(item_count, in_features, tables, comparisons, route_block, table_blocks, target)
    forward_kernel = _pairwise_forward_block_kernel(item_count, out_features, tables, table_size, block_d, out_blocks, target)
    route_kernel(latent_flat, anchors_contig, thresholds_contig, indices, margins)
    forward_kernel(indices, lut_contig, output)
    return output.view(batch, steps, out_features), indices.view(batch, steps, tables), margins.view(batch, steps, tables, comparisons)


class _PairwiseTileLangFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        latent: Tensor,
        anchors: Tensor,
        thresholds: Tensor,
        lut: Tensor,
        use_min_margin_ste: bool,
        target: str,
    ) -> tuple[Tensor, Tensor, Tensor]:
        output, indices, margins = _run_forward(latent, anchors, thresholds, lut, target=target)
        ctx.save_for_backward(indices, margins, anchors, lut)
        ctx.latent_shape = tuple(latent.shape)
        ctx.use_min_margin_ste = bool(use_min_margin_ste)
        ctx.target = target
        ctx.mark_non_differentiable(indices, margins)
        return output, indices, margins

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor, grad_indices: Tensor | None, grad_margins: Tensor | None) -> tuple[Any, ...]:
        del grad_indices, grad_margins
        indices, margins, anchors, lut = ctx.saved_tensors
        batch, steps, in_features = ctx.latent_shape
        item_count = batch * steps
        tables = indices.shape[-1]
        comparisons = margins.shape[-1]
        table_size = lut.shape[1]
        out_features = lut.shape[2]

        grad_flat = grad_output.reshape(item_count, out_features).contiguous().to(torch.float32)
        indices_flat = indices.reshape(item_count, tables).contiguous()
        margins_flat = margins.reshape(item_count, tables, comparisons).contiguous()
        lut_contig = lut.contiguous()
        anchors_contig = anchors.contiguous()
        grad_latent = torch.zeros((item_count, in_features), device=grad_output.device, dtype=torch.float32)
        grad_thresholds = torch.zeros((tables, comparisons), device=grad_output.device, dtype=torch.float32)
        grad_lut = torch.zeros_like(lut_contig)

        block_d = _select_block_size(out_features)
        out_blocks = (out_features + block_d - 1) // block_d
        lut_kernel = _pairwise_lut_backward_block_kernel(item_count, out_features, tables, table_size, block_d, out_blocks, ctx.target)
        lut_kernel(grad_flat, indices_flat, grad_lut)

        if ctx.use_min_margin_ste:
            ste_kernel = _pairwise_min_backward_kernel(
                item_count,
                in_features,
                out_features,
                tables,
                comparisons,
                table_size,
                block_d,
                out_blocks,
                ctx.target,
            )
        else:
            ste_kernel = _pairwise_full_backward_kernel(
                item_count,
                in_features,
                out_features,
                tables,
                comparisons,
                table_size,
                block_d,
                out_blocks,
                ctx.target,
            )
        ste_kernel(grad_flat, indices_flat, margins_flat, anchors_contig, lut_contig, grad_latent, grad_thresholds)

        return grad_latent.view(batch, steps, in_features), None, grad_thresholds, grad_lut, None, None


def pairwise_tilelang(
    latent: Tensor,
    anchors: Tensor,
    thresholds: Tensor,
    lut: Tensor,
    *,
    use_min_margin_ste: bool,
    target: str = "cuda",
) -> tuple[Tensor, Tensor, Tensor]:
    return _PairwiseTileLangFunction.apply(latent, anchors, thresholds, lut, use_min_margin_ste, target)
