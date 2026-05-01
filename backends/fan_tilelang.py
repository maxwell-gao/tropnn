from functools import lru_cache
from typing import Any

import torch
from torch import Tensor

from ._utils import (
    can_materialize_scores as _can_materialize_scores,
)
from ._utils import (
    float32_tilelang_dtype_name as _dtype_name,
)
from ._utils import (
    select_block_size as _select_block_size,
)
from .tilelang_route import (
    _trop_route_kernel,
    _trop_route_parallel_kernel,
    _trop_top2_scores_kernel,
    has_tilelang,
)


@lru_cache(maxsize=64)
def _fan_basis_coeff_sum_kernel(
    item_count: int,
    heads: int,
    cells: int,
    basis_rank: int,
    code_scale: float,
    training: bool,
    block_r: int,
    rank_blocks: int,
    target: str,
) -> Any:
    try:
        import tilelang
        import tilelang.language as T
    except ImportError as exc:
        raise RuntimeError("TileLang is not installed; install tilelang or use backend='torch'/'auto'") from exc

    @tilelang.jit(target=target, compile_flags=["-allow-unsupported-compiler", "-ccbin=/usr/bin/g++"])
    def coeff_sum_kernel() -> Any:
        head_count = heads
        rank_count = basis_rank
        scale = code_scale
        train_mode = training
        block_width = block_r

        @T.prim_func
        def kernel(
            value_coeff: T.Tensor((head_count, cells, rank_count), "float32"),
            winner_idx: T.Tensor((item_count, head_count), "int64"),
            runner_idx: T.Tensor((item_count, head_count), "int64"),
            margins: T.Tensor((item_count, head_count), "float32"),
            coeff_sum: T.Tensor((item_count, rank_count), "float32"),
        ):
            with T.Kernel(item_count, rank_blocks, threads=block_width) as (row, rank_tile):
                tx = T.get_thread_bindings()[0]
                rank = rank_tile * block_width + tx
                acc = T.alloc_fragment((1,), "float32")
                value = T.alloc_fragment((1,), "float32")
                if rank < rank_count:
                    acc[0] = 0.0
                    for head in T.serial(head_count):
                        winner = T.cast(winner_idx[row, head], "int32")
                        value[0] = value_coeff[head, winner, rank]
                        if train_mode:
                            runner = T.cast(runner_idx[row, head], "int32")
                            alpha = 0.5 / (1.0 + T.abs(margins[row, head]))
                            value[0] = value[0] + alpha * (value_coeff[head, runner, rank] - value[0])
                        acc[0] = acc[0] + value[0]
                    coeff_sum[row, rank] = acc[0] * scale

        return kernel

    return coeff_sum_kernel()


@lru_cache(maxsize=64)
def _fan_basis_hidden_from_coeff_kernel(
    item_count: int,
    code_dim: int,
    basis_rank: int,
    block_d: int,
    code_blocks: int,
    target: str,
) -> Any:
    try:
        import tilelang
        import tilelang.language as T
    except ImportError as exc:
        raise RuntimeError("TileLang is not installed; install tilelang or use backend='torch'/'auto'") from exc

    @tilelang.jit(target=target, compile_flags=["-allow-unsupported-compiler", "-ccbin=/usr/bin/g++"])
    def hidden_kernel() -> Any:
        latent_dim = code_dim
        rank_count = basis_rank
        block_width = block_d

        @T.prim_func
        def kernel(
            latent: T.Tensor((item_count, latent_dim), "float32"),
            coeff_sum: T.Tensor((item_count, rank_count), "float32"),
            value_basis: T.Tensor((rank_count, latent_dim), "float32"),
            hidden: T.Tensor((item_count, latent_dim), "float32"),
        ):
            with T.Kernel(item_count, code_blocks, threads=block_width) as (row, dim_tile):
                tx = T.get_thread_bindings()[0]
                dim = dim_tile * block_width + tx
                acc = T.alloc_fragment((1,), "float32")
                if dim < latent_dim:
                    acc[0] = latent[row, dim]
                    for rank in T.serial(rank_count):
                        acc[0] = acc[0] + coeff_sum[row, rank] * value_basis[rank, dim]
                    hidden[row, dim] = acc[0]

        return kernel

    return hidden_kernel()


@lru_cache(maxsize=64)
def _fan_basis_hidden_kernel(
    item_count: int,
    heads: int,
    cells: int,
    code_dim: int,
    basis_rank: int,
    code_scale: float,
    training: bool,
    block_d: int,
    code_blocks: int,
    target: str,
) -> Any:
    try:
        import tilelang
        import tilelang.language as T
    except ImportError as exc:
        raise RuntimeError("TileLang is not installed; install tilelang or use backend='torch'/'auto'") from exc

    @tilelang.jit(target=target, compile_flags=["-allow-unsupported-compiler", "-ccbin=/usr/bin/g++"])
    def hidden_kernel() -> Any:
        head_count = heads
        latent_dim = code_dim
        rank_count = basis_rank
        train_mode = training
        block_width = block_d
        latent_tiles = code_blocks

        @T.prim_func
        def kernel(
            latent: T.Tensor((item_count, latent_dim), "float32"),
            value_coeff: T.Tensor((head_count, cells, rank_count), "float32"),
            value_basis: T.Tensor((rank_count, latent_dim), "float32"),
            winner_idx: T.Tensor((item_count, head_count), "int64"),
            runner_idx: T.Tensor((item_count, head_count), "int64"),
            margins: T.Tensor((item_count, head_count), "float32"),
            hidden: T.Tensor((item_count, latent_dim), "float32"),
        ):
            with T.Kernel(item_count, latent_tiles, threads=block_width) as (row, dim_tile):
                tx = T.get_thread_bindings()[0]
                dim = dim_tile * block_width + tx
                acc = T.alloc_fragment((1,), "float32")
                value = T.alloc_fragment((1,), "float32")
                runner_value = T.alloc_fragment((1,), "float32")
                if dim < latent_dim:
                    acc[0] = latent[row, dim]
                    for head in T.serial(head_count):
                        winner = T.cast(winner_idx[row, head], "int32")
                        value[0] = 0.0
                        for rank in T.serial(rank_count):
                            value[0] = value[0] + value_coeff[head, winner, rank] * value_basis[rank, dim]
                        if train_mode:
                            runner = T.cast(runner_idx[row, head], "int32")
                            runner_value[0] = 0.0
                            for rank in T.serial(rank_count):
                                runner_value[0] = runner_value[0] + value_coeff[head, runner, rank] * value_basis[rank, dim]
                            alpha = 0.5 / (1.0 + T.abs(margins[row, head]))
                            value[0] = value[0] + alpha * (runner_value[0] - value[0])
                        acc[0] = acc[0] + value[0] * code_scale
                    hidden[row, dim] = acc[0]

        return kernel

    return hidden_kernel()


@lru_cache(maxsize=64)
def _fan_basis_value_backward_kernel(
    item_count: int,
    heads: int,
    cells: int,
    code_dim: int,
    basis_rank: int,
    code_scale: float,
    training: bool,
    block_d: int,
    code_blocks: int,
    target: str,
) -> Any:
    try:
        import tilelang
        import tilelang.language as T
    except ImportError as exc:
        raise RuntimeError("TileLang is not installed; install tilelang or use backend='torch'/'auto'") from exc

    @tilelang.jit(target=target, compile_flags=["-allow-unsupported-compiler", "-ccbin=/usr/bin/g++"])
    def value_backward_kernel() -> Any:
        head_count = heads
        latent_dim = code_dim
        rank_count = basis_rank
        train_mode = training
        block_width = block_d
        latent_tiles = code_blocks

        @T.prim_func
        def kernel(
            grad_hidden: T.Tensor((item_count, latent_dim), "float32"),
            value_coeff: T.Tensor((head_count, cells, rank_count), "float32"),
            value_basis: T.Tensor((rank_count, latent_dim), "float32"),
            winner_idx: T.Tensor((item_count, head_count), "int64"),
            runner_idx: T.Tensor((item_count, head_count), "int64"),
            margins: T.Tensor((item_count, head_count), "float32"),
            grad_latent: T.Tensor((item_count, latent_dim), "float32"),
            grad_value_coeff: T.Tensor((head_count, cells, rank_count), "float32"),
            grad_value_basis: T.Tensor((rank_count, latent_dim), "float32"),
        ):
            with T.Kernel(item_count, latent_tiles, threads=block_width) as (row, dim_tile):
                tx = T.get_thread_bindings()[0]
                dim = dim_tile * block_width + tx
                if dim < latent_dim:
                    grad_latent[row, dim] = grad_hidden[row, dim]

            with T.Kernel(item_count, head_count, rank_count * latent_tiles, threads=block_width) as (row, head, basis_dim_tile):
                tx = T.get_thread_bindings()[0]
                basis_idx = basis_dim_tile // latent_tiles
                dim_tile = basis_dim_tile - basis_idx * latent_tiles
                dim = dim_tile * block_width + tx
                coeff_mix = T.alloc_fragment((1,), "float32")
                if dim < latent_dim:
                    winner = T.cast(winner_idx[row, head], "int32")
                    coeff_mix[0] = value_coeff[head, winner, basis_idx]
                    if train_mode:
                        runner = T.cast(runner_idx[row, head], "int32")
                        alpha = 0.5 / (1.0 + T.abs(margins[row, head]))
                        coeff_mix[0] = coeff_mix[0] + alpha * (value_coeff[head, runner, basis_idx] - coeff_mix[0])
                    T.atomic_add(grad_value_basis[basis_idx, dim], grad_hidden[row, dim] * code_scale * coeff_mix[0])

            with T.Kernel(item_count, head_count, rank_count, threads=block_width) as (row, head, coeff_basis_idx):
                tx = T.get_thread_bindings()[0]
                partial = T.alloc_shared((block_width,), "float32")
                dot = T.alloc_fragment((1,), "float32")
                winner = T.cast(winner_idx[row, head], "int32")
                dot[0] = 0.0
                for dim_tile in T.serial(latent_tiles):
                    dim = dim_tile * block_width + tx
                    if dim < latent_dim:
                        dot[0] = dot[0] + grad_hidden[row, dim] * value_basis[coeff_basis_idx, dim]
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
                    if train_mode:
                        runner = T.cast(runner_idx[row, head], "int32")
                        alpha = 0.5 / (1.0 + T.abs(margins[row, head]))
                        T.atomic_add(grad_value_coeff[head, winner, coeff_basis_idx], partial[0] * code_scale * (1.0 - alpha))
                        T.atomic_add(grad_value_coeff[head, runner, coeff_basis_idx], partial[0] * code_scale * alpha)
                    else:
                        T.atomic_add(grad_value_coeff[head, winner, coeff_basis_idx], partial[0] * code_scale)

        return kernel

    return value_backward_kernel()


@lru_cache(maxsize=64)
def _fan_basis_value_backward_fast_kernel(
    item_count: int,
    heads: int,
    cells: int,
    code_dim: int,
    basis_rank: int,
    code_scale: float,
    training: bool,
    block_d: int,
    code_blocks: int,
    block_r: int,
    rank_blocks: int,
    target: str,
) -> Any:
    try:
        import tilelang
        import tilelang.language as T
    except ImportError as exc:
        raise RuntimeError("TileLang is not installed; install tilelang or use backend='torch'/'auto'") from exc

    @tilelang.jit(target=target, compile_flags=["-allow-unsupported-compiler", "-ccbin=/usr/bin/g++"])
    def value_backward_kernel() -> Any:
        head_count = heads
        latent_dim = code_dim
        rank_count = basis_rank
        scale = code_scale
        train_mode = training
        dim_block_width = block_d
        rank_block_width = block_r
        latent_tiles = code_blocks

        @T.prim_func
        def kernel(
            grad_hidden: T.Tensor((item_count, latent_dim), "float32"),
            coeff_sum: T.Tensor((item_count, rank_count), "float32"),
            value_coeff: T.Tensor((head_count, cells, rank_count), "float32"),
            value_basis: T.Tensor((rank_count, latent_dim), "float32"),
            winner_idx: T.Tensor((item_count, head_count), "int64"),
            runner_idx: T.Tensor((item_count, head_count), "int64"),
            margins: T.Tensor((item_count, head_count), "float32"),
            grad_latent: T.Tensor((item_count, latent_dim), "float32"),
            grad_value_coeff: T.Tensor((head_count, cells, rank_count), "float32"),
            grad_value_basis: T.Tensor((rank_count, latent_dim), "float32"),
            rank_grad: T.Tensor((item_count, rank_count), "float32"),
        ):
            with T.Kernel(item_count, latent_tiles, threads=dim_block_width) as (row_copy, dim_tile_copy):
                tx = T.get_thread_bindings()[0]
                dim = dim_tile_copy * dim_block_width + tx
                if dim < latent_dim:
                    grad_latent[row_copy, dim] = grad_hidden[row_copy, dim]

            with T.Kernel(item_count, rank_blocks, threads=rank_block_width) as (row_rank, rank_tile_dot):
                tx = T.get_thread_bindings()[0]
                rank = rank_tile_dot * rank_block_width + tx
                dot = T.alloc_fragment((1,), "float32")
                if rank < rank_count:
                    dot[0] = 0.0
                    for dim_tile_dot in T.serial(latent_tiles):
                        for lane_dot in T.serial(dim_block_width):
                            dim = dim_tile_dot * dim_block_width + lane_dot
                            if dim < latent_dim:
                                dot[0] = dot[0] + grad_hidden[row_rank, dim] * value_basis[rank, dim]
                    rank_grad[row_rank, rank] = dot[0]

            with T.Kernel(rank_count, latent_tiles, threads=dim_block_width) as (rank_basis, dim_tile_basis):
                tx = T.get_thread_bindings()[0]
                dim = dim_tile_basis * dim_block_width + tx
                acc = T.alloc_fragment((1,), "float32")
                if dim < latent_dim:
                    acc[0] = 0.0
                    for item in T.serial(item_count):
                        acc[0] = acc[0] + coeff_sum[item, rank_basis] * grad_hidden[item, dim]
                    grad_value_basis[rank_basis, dim] = acc[0]

            with T.Kernel(item_count, head_count, rank_blocks, threads=rank_block_width) as (
                row_coeff,
                head_coeff,
                rank_tile_coeff,
            ):
                tx = T.get_thread_bindings()[0]
                rank = rank_tile_coeff * rank_block_width + tx
                if rank < rank_count:
                    winner = T.cast(winner_idx[row_coeff, head_coeff], "int32")
                    grad_value = rank_grad[row_coeff, rank] * scale
                    if train_mode:
                        runner = T.cast(runner_idx[row_coeff, head_coeff], "int32")
                        alpha = 0.5 / (1.0 + T.abs(margins[row_coeff, head_coeff]))
                        T.atomic_add(grad_value_coeff[head_coeff, winner, rank], grad_value * (1.0 - alpha))
                        T.atomic_add(grad_value_coeff[head_coeff, runner, rank], grad_value * alpha)
                    else:
                        T.atomic_add(grad_value_coeff[head_coeff, winner, rank], grad_value)

        return kernel

    return value_backward_kernel()


@lru_cache(maxsize=64)
def _fan_basis_coeff_backward_kernel(
    item_count: int,
    heads: int,
    cells: int,
    basis_rank: int,
    code_scale: float,
    training: bool,
    block_r: int,
    rank_blocks: int,
    target: str,
) -> Any:
    try:
        import tilelang
        import tilelang.language as T
    except ImportError as exc:
        raise RuntimeError("TileLang is not installed; install tilelang or use backend='torch'/'auto'") from exc

    @tilelang.jit(target=target, compile_flags=["-allow-unsupported-compiler", "-ccbin=/usr/bin/g++"])
    def coeff_backward_kernel() -> Any:
        head_count = heads
        rank_count = basis_rank
        scale = code_scale
        train_mode = training
        block_width = block_r

        @T.prim_func
        def kernel(
            rank_grad: T.Tensor((item_count, rank_count), "float32"),
            winner_idx: T.Tensor((item_count, head_count), "int64"),
            runner_idx: T.Tensor((item_count, head_count), "int64"),
            margins: T.Tensor((item_count, head_count), "float32"),
            grad_value_coeff: T.Tensor((head_count, cells, rank_count), "float32"),
        ):
            with T.Kernel(item_count, head_count, rank_blocks, threads=block_width) as (
                row_coeff,
                head_coeff,
                rank_tile_coeff,
            ):
                tx = T.get_thread_bindings()[0]
                rank = rank_tile_coeff * block_width + tx
                if rank < rank_count:
                    winner = T.cast(winner_idx[row_coeff, head_coeff], "int32")
                    grad_value = rank_grad[row_coeff, rank] * scale
                    if train_mode:
                        runner = T.cast(runner_idx[row_coeff, head_coeff], "int32")
                        alpha = 0.5 / (1.0 + T.abs(margins[row_coeff, head_coeff]))
                        T.atomic_add(grad_value_coeff[head_coeff, winner, rank], grad_value * (1.0 - alpha))
                        T.atomic_add(grad_value_coeff[head_coeff, runner, rank], grad_value * alpha)
                    else:
                        T.atomic_add(grad_value_coeff[head_coeff, winner, rank], grad_value)

        return kernel

    return coeff_backward_kernel()


@lru_cache(maxsize=64)
def _fan_basis_router_backward_fast_kernel(
    item_count: int,
    heads: int,
    cells: int,
    code_dim: int,
    basis_rank: int,
    code_scale: float,
    block_d: int,
    code_blocks: int,
    block_r: int,
    rank_blocks: int,
    target: str,
) -> Any:
    try:
        import tilelang
        import tilelang.language as T
    except ImportError as exc:
        raise RuntimeError("TileLang is not installed; install tilelang or use backend='torch'/'auto'") from exc

    @tilelang.jit(target=target, compile_flags=["-allow-unsupported-compiler", "-ccbin=/usr/bin/g++"])
    def router_backward_kernel() -> Any:
        head_count = heads
        latent_dim = code_dim
        rank_count = basis_rank
        scale = code_scale
        dim_block_width = block_d
        rank_block_width = block_r
        latent_tiles = code_blocks

        @T.prim_func
        def kernel(
            latent: T.Tensor((item_count, latent_dim), "float32"),
            sites: T.Tensor((head_count, cells, latent_dim), "float32"),
            value_coeff: T.Tensor((head_count, cells, rank_count), "float32"),
            rank_grad: T.Tensor((item_count, rank_count), "float32"),
            winner_idx: T.Tensor((item_count, head_count), "int64"),
            runner_idx: T.Tensor((item_count, head_count), "int64"),
            margins: T.Tensor((item_count, head_count), "float32"),
            grad_latent: T.Tensor((item_count, latent_dim), "float32"),
            grad_sites: T.Tensor((head_count, cells, latent_dim), "float32"),
            grad_lifting: T.Tensor((head_count, cells), "float32"),
            grad_route_margin: T.Tensor((item_count, head_count), "float32"),
        ):
            with T.Kernel(item_count, head_count, threads=rank_block_width) as (row_route, head_route):
                tx = T.get_thread_bindings()[0]
                partial = T.alloc_shared((rank_block_width,), "float32")
                winner = T.cast(winner_idx[row_route, head_route], "int32")
                runner = T.cast(runner_idx[row_route, head_route], "int32")
                dot = T.alloc_fragment((1,), "float32")
                dot[0] = 0.0
                for rank_tile_route in T.serial(rank_blocks):
                    rank = rank_tile_route * rank_block_width + tx
                    if rank < rank_count:
                        delta_coeff = value_coeff[head_route, runner, rank] - value_coeff[head_route, winner, rank]
                        dot[0] = dot[0] + rank_grad[row_route, rank] * delta_coeff
                partial[tx] = dot[0]
                T.sync_threads()
                if rank_block_width >= 256:
                    if tx < 128:
                        partial[tx] = partial[tx] + partial[tx + 128]
                    T.sync_threads()
                if rank_block_width >= 128:
                    if tx < 64:
                        partial[tx] = partial[tx] + partial[tx + 64]
                    T.sync_threads()
                if rank_block_width >= 64:
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
                    margin = margins[row_route, head_route]
                    denom = (1.0 + T.abs(margin)) * (1.0 + T.abs(margin))
                    dalpha = T.alloc_fragment((1,), "float32")
                    dalpha[0] = 0.0
                    if margin > 0.0:
                        dalpha[0] = -0.5 / denom
                    else:
                        if margin < 0.0:
                            dalpha[0] = 0.5 / denom
                    grad_margin = partial[0] * scale * dalpha[0]
                    grad_route_margin[row_route, head_route] = grad_margin
                    T.atomic_add(grad_lifting[head_route, winner], grad_margin)
                    T.atomic_add(grad_lifting[head_route, runner], -grad_margin)

            with T.Kernel(item_count, head_count, latent_tiles, threads=dim_block_width) as (
                row_apply,
                head_apply,
                dim_tile_apply,
            ):
                tx = T.get_thread_bindings()[0]
                dim = dim_tile_apply * dim_block_width + tx
                if dim < latent_dim:
                    winner = T.cast(winner_idx[row_apply, head_apply], "int32")
                    runner = T.cast(runner_idx[row_apply, head_apply], "int32")
                    grad_margin = grad_route_margin[row_apply, head_apply]
                    latent_value = latent[row_apply, dim]
                    T.atomic_add(grad_sites[head_apply, winner, dim], grad_margin * latent_value)
                    T.atomic_add(grad_sites[head_apply, runner, dim], -grad_margin * latent_value)
                    site_delta = sites[head_apply, winner, dim] - sites[head_apply, runner, dim]
                    T.atomic_add(grad_latent[row_apply, dim], grad_margin * site_delta)

        return kernel

    return router_backward_kernel()


def _run_fan_basis_forward(
    latent: Tensor,
    sites: Tensor,
    lifting: Tensor,
    value_coeff: Tensor,
    value_basis: Tensor,
    *,
    code_scale: float,
    training: bool,
    target: str,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    batch, steps, code_dim = latent.shape
    heads, cells, _ = sites.shape
    basis_rank = value_basis.shape[0]
    item_count = batch * steps
    latent_flat = latent.reshape(item_count, code_dim).contiguous()
    sites_table = sites.contiguous()
    lifting_table = lifting.contiguous()
    coeff = value_coeff.contiguous()
    basis = value_basis.contiguous()
    winner_idx = torch.empty((item_count, heads), device=latent.device, dtype=torch.int64)
    runner_idx = torch.empty((item_count, heads), device=latent.device, dtype=torch.int64)
    margins = torch.empty((item_count, heads), device=latent.device, dtype=latent.dtype)
    coeff_sum = torch.empty((item_count, basis_rank), device=latent.device, dtype=latent.dtype)
    hidden = torch.empty((item_count, code_dim), device=latent.device, dtype=latent.dtype)
    block_d = _select_block_size(code_dim)
    code_blocks = (code_dim + block_d - 1) // block_d
    block_r = _select_block_size(basis_rank)
    rank_blocks = (basis_rank + block_r - 1) // block_r

    if code_dim >= 128 and _can_materialize_scores(item_count, heads, cells):
        try:
            from .triton_scores import has_triton, trop_scores_triton
        except ImportError:
            use_score_route = False
        else:
            use_score_route = has_triton()
        if use_score_route:
            scores = trop_scores_triton(latent, sites_table, lifting_table).reshape(item_count, heads, cells).contiguous()
            top2_kernel = _trop_top2_scores_kernel(item_count, heads, cells, target)
            top2_kernel(scores, winner_idx, runner_idx, margins)
        else:
            route_kernel = _trop_route_parallel_kernel(item_count, heads, cells, code_dim, block_d, code_blocks, target)
            route_kernel(latent_flat, sites_table, lifting_table, winner_idx, runner_idx, margins)
    elif code_dim >= 32:
        route_kernel = _trop_route_parallel_kernel(item_count, heads, cells, code_dim, block_d, code_blocks, target)
        route_kernel(latent_flat, sites_table, lifting_table, winner_idx, runner_idx, margins)
    else:
        route_kernel = _trop_route_kernel(item_count, heads, cells, code_dim, target)
        route_kernel(latent_flat, sites_table, lifting_table, winner_idx, runner_idx, margins)

    coeff_kernel = _fan_basis_coeff_sum_kernel(
        item_count,
        heads,
        cells,
        basis_rank,
        float(code_scale),
        bool(training),
        block_r,
        rank_blocks,
        target,
    )
    hidden_kernel = _fan_basis_hidden_from_coeff_kernel(
        item_count,
        code_dim,
        basis_rank,
        block_d,
        code_blocks,
        target,
    )
    coeff_kernel(coeff, winner_idx, runner_idx, margins, coeff_sum)
    hidden_kernel(latent_flat, coeff_sum, basis, hidden)
    return (
        hidden.view(batch, steps, code_dim),
        winner_idx.view(batch, steps, heads),
        runner_idx.view(batch, steps, heads),
        margins.view(batch, steps, heads),
        coeff_sum,
    )


class _TropFanBasisRouteHiddenTileLangFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        latent: Tensor,
        sites: Tensor,
        lifting: Tensor,
        value_coeff: Tensor,
        value_basis: Tensor,
        code_scale: float,
        training: bool,
        target: str,
    ) -> tuple[Tensor, Tensor, Tensor]:
        hidden, winner_idx, runner_idx, margins, coeff_sum = _run_fan_basis_forward(
            latent,
            sites,
            lifting,
            value_coeff,
            value_basis,
            code_scale=code_scale,
            training=training,
            target=target,
        )
        ctx.save_for_backward(latent, sites, value_coeff, value_basis, winner_idx, runner_idx, margins, coeff_sum)
        ctx.code_scale = float(code_scale)
        ctx.training = bool(training)
        ctx.target = target
        ctx.mark_non_differentiable(winner_idx, margins)
        return hidden, winner_idx, margins

    @staticmethod
    def backward(ctx: Any, grad_hidden: Tensor, grad_winner: Tensor | None, grad_margins: Tensor | None) -> tuple[Any, ...]:
        del grad_winner, grad_margins
        latent, sites, value_coeff, value_basis, winner_idx, runner_idx, margins, coeff_sum = ctx.saved_tensors
        batch, steps, code_dim = latent.shape
        heads, cells, _ = sites.shape
        basis_rank = value_basis.shape[0]
        item_count = batch * steps
        grad_flat = grad_hidden.reshape(item_count, code_dim).contiguous().to(torch.float32)
        latent_flat = latent.reshape(item_count, code_dim).contiguous()
        winner_flat = winner_idx.reshape(item_count, heads).contiguous()
        runner_flat = runner_idx.reshape(item_count, heads).contiguous()
        margins_flat = margins.reshape(item_count, heads).contiguous()
        sites_table = sites.contiguous()
        coeff = value_coeff.contiguous()
        basis = value_basis.contiguous()

        grad_latent = grad_flat.clone()
        grad_sites = torch.zeros_like(sites_table)
        grad_lifting = torch.zeros((heads, cells), device=grad_hidden.device, dtype=torch.float32)
        grad_value_coeff = torch.zeros_like(coeff)
        # The low-rank value field has exact dense algebra; keep that on cuBLAS
        # and reserve TileLang for the sparse winner/runner fan reductions.
        rank_grad = grad_flat.matmul(basis.t()).contiguous()
        grad_value_basis = coeff_sum.contiguous().t().matmul(grad_flat).contiguous()
        block_d = _select_block_size(code_dim)
        code_blocks = (code_dim + block_d - 1) // block_d
        block_r = _select_block_size(basis_rank)
        rank_blocks = (basis_rank + block_r - 1) // block_r

        coeff_kernel = _fan_basis_coeff_backward_kernel(
            item_count,
            heads,
            cells,
            basis_rank,
            ctx.code_scale,
            ctx.training,
            block_r,
            rank_blocks,
            ctx.target,
        )
        coeff_kernel(
            rank_grad,
            winner_flat,
            runner_flat,
            margins_flat,
            grad_value_coeff,
        )

        if ctx.training:
            grad_route_margin = torch.empty((item_count, heads), device=grad_hidden.device, dtype=torch.float32)
            router_kernel = _fan_basis_router_backward_fast_kernel(
                item_count,
                heads,
                cells,
                code_dim,
                basis_rank,
                ctx.code_scale,
                block_d,
                code_blocks,
                block_r,
                rank_blocks,
                ctx.target,
            )
            router_kernel(
                latent_flat,
                sites_table,
                coeff,
                rank_grad,
                winner_flat,
                runner_flat,
                margins_flat,
                grad_latent,
                grad_sites,
                grad_lifting,
                grad_route_margin,
            )

        return (
            grad_latent.view(batch, steps, code_dim),
            grad_sites,
            grad_lifting,
            grad_value_coeff,
            grad_value_basis,
            None,
            None,
            None,
        )


def trop_fan_basis_route_hidden_tilelang(
    latent: Tensor,
    sites: Tensor,
    lifting: Tensor,
    value_coeff: Tensor,
    value_basis: Tensor,
    *,
    code_scale: float,
    training: bool = False,
    target: str = "cuda",
) -> tuple[Tensor, Tensor, Tensor]:
    if not has_tilelang():
        raise RuntimeError("TileLang is not installed; install tilelang or use backend='torch'/'auto'")
    if not latent.is_cuda:
        raise ValueError("TileLang TropFanBasis backend requires CUDA tensors")
    if latent.ndim != 3:
        raise ValueError(f"latent must have shape [batch, steps, code_dim], got {tuple(latent.shape)}")
    if sites.ndim != 3:
        raise ValueError(f"sites must have shape [heads, cells, code_dim], got {tuple(sites.shape)}")
    if lifting.shape != sites.shape[:2]:
        raise ValueError(f"lifting must have shape [heads, cells], got {tuple(lifting.shape)}")
    if value_coeff.ndim != 3:
        raise ValueError(f"value_coeff must have shape [heads, cells, rank], got {tuple(value_coeff.shape)}")
    if value_coeff.shape[:2] != sites.shape[:2]:
        raise ValueError(f"value_coeff heads/cells must match sites, got {tuple(value_coeff.shape)} and {tuple(sites.shape)}")
    if value_basis.ndim != 2:
        raise ValueError(f"value_basis must have shape [rank, code_dim], got {tuple(value_basis.shape)}")

    _, _, code_dim = latent.shape
    heads, cells, site_dim = sites.shape
    rank = value_coeff.shape[2]
    if site_dim != code_dim:
        raise ValueError(f"latent code_dim {code_dim} does not match sites code_dim {site_dim}")
    if value_basis.shape != (rank, code_dim):
        raise ValueError(f"value_basis must have shape {(rank, code_dim)}, got {tuple(value_basis.shape)}")
    if not all(tensor.is_cuda for tensor in (sites, lifting, value_coeff, value_basis)):
        raise ValueError("TileLang TropFanBasis backend requires all tensors on CUDA")
    del heads, cells

    for tensor in (latent, sites, lifting, value_coeff, value_basis):
        _dtype_name(tensor.dtype)

    try:
        hidden, winner_idx, margins = _TropFanBasisRouteHiddenTileLangFunction.apply(
            latent,
            sites,
            lifting,
            value_coeff,
            value_basis,
            float(code_scale),
            bool(training),
            target,
        )
    except Exception as exc:
        raise RuntimeError(
            "TileLang TropFanBasis backend failed to compile or launch. Ensure a CUDA toolkit compatible with the GPU is "
            "first on PATH and export CC=/usr/bin/gcc CXX=/usr/bin/g++ before running."
        ) from exc

    return hidden, winner_idx, margins
