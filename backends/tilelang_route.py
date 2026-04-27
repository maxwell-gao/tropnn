from functools import lru_cache
from typing import Any

import torch
from torch import Tensor

_MAX_SCORE_ROUTE_BYTES = 128 * 1024 * 1024


def _next_power_of_2(value: int) -> int:
    if value < 1:
        return 1
    return 1 << (value - 1).bit_length()


def _select_block_size(value: int, *, min_block: int = 32, max_block: int = 256) -> int:
    return max(min_block, min(max_block, _next_power_of_2(value)))


def has_tilelang() -> bool:
    try:
        import tilelang  # noqa: F401
        import tilelang.language  # noqa: F401
    except ImportError:
        return False
    return True


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.float32:
        return "float32"
    raise TypeError(f"TileLang route backend currently supports float32 tensors only, got {dtype}")


def _can_materialize_scores(item_count: int, heads: int, cells: int) -> bool:
    return item_count * heads * cells * 4 <= _MAX_SCORE_ROUTE_BYTES


@lru_cache(maxsize=64)
def _trop_route_kernel(
    item_count: int,
    heads: int,
    cells: int,
    code_dim: int,
    target: str,
) -> Any:
    try:
        import tilelang
        import tilelang.language as T
    except ImportError as exc:
        raise RuntimeError("TileLang is not installed; install tilelang or use backend='torch'/'auto'") from exc

    @tilelang.jit(target=target, compile_flags=["-allow-unsupported-compiler", "-ccbin=/usr/bin/g++"])
    def route_kernel() -> Any:
        head_count = heads
        cell_count = cells
        latent_dim = code_dim

        @T.prim_func
        def kernel(
            latent: T.Tensor((item_count, latent_dim), "float32"),
            router_weight: T.Tensor((head_count, cell_count, latent_dim), "float32"),
            router_bias: T.Tensor((head_count, cell_count), "float32"),
            winner_idx: T.Tensor((item_count, head_count), "int64"),
            runner_idx: T.Tensor((item_count, head_count), "int64"),
            margins: T.Tensor((item_count, head_count), "float32"),
        ):
            with T.Kernel(item_count, head_count, threads=1) as (row, head):
                best = T.alloc_fragment((1,), "float32")
                second = T.alloc_fragment((1,), "float32")
                score = T.alloc_fragment((1,), "float32")
                best_cell = T.alloc_fragment((1,), "int32")
                second_cell = T.alloc_fragment((1,), "int32")
                best[0] = -3.4028234663852886e38
                second[0] = -3.4028234663852886e38
                best_cell[0] = 0
                second_cell[0] = 0

                for cell in T.serial(cell_count):
                    score[0] = 0.0
                    for dim in T.serial(latent_dim):
                        score[0] = score[0] + latent[row, dim] * router_weight[head, cell, dim]
                    score[0] = score[0] + router_bias[head, cell]
                    if score[0] > best[0]:
                        second[0] = best[0]
                        second_cell[0] = best_cell[0]
                        best[0] = score[0]
                        best_cell[0] = cell
                    else:
                        if score[0] > second[0]:
                            second[0] = score[0]
                            second_cell[0] = cell

                winner_idx[row, head] = best_cell[0]
                runner_idx[row, head] = second_cell[0]
                margins[row, head] = best[0] - second[0]

        return kernel

    return route_kernel()


@lru_cache(maxsize=64)
def _trop_route_parallel_kernel(
    item_count: int,
    heads: int,
    cells: int,
    code_dim: int,
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
    def route_kernel() -> Any:
        head_count = heads
        cell_count = cells
        latent_dim = code_dim
        block_width = block_d
        latent_tiles = code_blocks

        @T.prim_func
        def kernel(
            latent: T.Tensor((item_count, latent_dim), "float32"),
            router_weight: T.Tensor((head_count, cell_count, latent_dim), "float32"),
            router_bias: T.Tensor((head_count, cell_count), "float32"),
            winner_idx: T.Tensor((item_count, head_count), "int64"),
            runner_idx: T.Tensor((item_count, head_count), "int64"),
            margins: T.Tensor((item_count, head_count), "float32"),
        ):
            with T.Kernel(item_count, head_count, threads=block_width) as (row, head):
                tx = T.get_thread_bindings()[0]
                partial = T.alloc_shared((block_width,), "float32")
                best = T.alloc_fragment((1,), "float32")
                second = T.alloc_fragment((1,), "float32")
                dot = T.alloc_fragment((1,), "float32")
                best_cell = T.alloc_fragment((1,), "int32")
                second_cell = T.alloc_fragment((1,), "int32")
                if tx == 0:
                    best[0] = -3.4028234663852886e38
                    second[0] = -3.4028234663852886e38
                    best_cell[0] = 0
                    second_cell[0] = 0

                for cell in T.serial(cell_count):
                    dot[0] = 0.0
                    for dim_tile in T.serial(latent_tiles):
                        dim = dim_tile * block_width + tx
                        if dim < latent_dim:
                            dot[0] = dot[0] + latent[row, dim] * router_weight[head, cell, dim]
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
                        score = partial[0] + router_bias[head, cell]
                        if score > best[0]:
                            second[0] = best[0]
                            second_cell[0] = best_cell[0]
                            best[0] = score
                            best_cell[0] = cell
                        else:
                            if score > second[0]:
                                second[0] = score
                                second_cell[0] = cell
                    T.sync_threads()

                if tx == 0:
                    winner_idx[row, head] = best_cell[0]
                    runner_idx[row, head] = second_cell[0]
                    margins[row, head] = best[0] - second[0]

        return kernel

    return route_kernel()


@lru_cache(maxsize=64)
def _trop_top2_scores_kernel(
    item_count: int,
    heads: int,
    cells: int,
    target: str,
) -> Any:
    try:
        import tilelang
        import tilelang.language as T
    except ImportError as exc:
        raise RuntimeError("TileLang is not installed; install tilelang or use backend='torch'/'auto'") from exc

    @tilelang.jit(target=target, compile_flags=["-allow-unsupported-compiler", "-ccbin=/usr/bin/g++"])
    def top2_kernel() -> Any:
        head_count = heads
        cell_count = cells

        @T.prim_func
        def kernel(
            scores: T.Tensor((item_count, head_count, cell_count), "float32"),
            winner_idx: T.Tensor((item_count, head_count), "int64"),
            runner_idx: T.Tensor((item_count, head_count), "int64"),
            margins: T.Tensor((item_count, head_count), "float32"),
        ):
            with T.Kernel(item_count, head_count, threads=1) as (row, head):
                best = T.alloc_fragment((1,), "float32")
                second = T.alloc_fragment((1,), "float32")
                best_cell = T.alloc_fragment((1,), "int32")
                second_cell = T.alloc_fragment((1,), "int32")
                best[0] = -3.4028234663852886e38
                second[0] = -3.4028234663852886e38
                best_cell[0] = 0
                second_cell[0] = 0

                for cell in T.serial(cell_count):
                    score = scores[row, head, cell]
                    if score > best[0]:
                        second[0] = best[0]
                        second_cell[0] = best_cell[0]
                        best[0] = score
                        best_cell[0] = cell
                    else:
                        if score > second[0]:
                            second[0] = score
                            second_cell[0] = cell

                winner_idx[row, head] = best_cell[0]
                runner_idx[row, head] = second_cell[0]
                margins[row, head] = best[0] - second[0]

        return kernel

    return top2_kernel()


@lru_cache(maxsize=64)
def _trop_hidden_kernel(
    item_count: int,
    heads: int,
    cells: int,
    code_dim: int,
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
        cell_count = cells
        latent_dim = code_dim
        train_mode = training
        block_width = block_d
        latent_tiles = code_blocks

        @T.prim_func
        def kernel(
            latent: T.Tensor((item_count, latent_dim), "float32"),
            code: T.Tensor((head_count, cell_count, latent_dim), "float32"),
            winner_idx: T.Tensor((item_count, head_count), "int64"),
            runner_idx: T.Tensor((item_count, head_count), "int64"),
            margins: T.Tensor((item_count, head_count), "float32"),
            hidden: T.Tensor((item_count, latent_dim), "float32"),
        ):
            with T.Kernel(item_count, latent_tiles, threads=block_width) as (row, dim_tile):
                tx = T.get_thread_bindings()[0]
                dim = dim_tile * block_width + tx
                acc = T.alloc_fragment((1,), "float32")
                if dim < latent_dim:
                    acc[0] = latent[row, dim]
                    for head in T.serial(head_count):
                        winner = T.cast(winner_idx[row, head], "int32")
                        if train_mode:
                            runner = T.cast(runner_idx[row, head], "int32")
                            margin = margins[row, head]
                            alpha = 0.5 / (1.0 + T.abs(margin))
                            mixed = code[head, winner, dim] + alpha * (code[head, runner, dim] - code[head, winner, dim])
                            acc[0] = acc[0] + mixed * code_scale
                        else:
                            acc[0] = acc[0] + code[head, winner, dim] * code_scale
                    hidden[row, dim] = acc[0]

        return kernel

    return hidden_kernel()


@lru_cache(maxsize=64)
def _trop_code_backward_kernel(
    item_count: int,
    heads: int,
    cells: int,
    code_dim: int,
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
    def code_backward_kernel() -> Any:
        head_count = heads
        cell_count = cells
        latent_dim = code_dim
        train_mode = training
        block_width = block_d
        latent_tiles = code_blocks

        @T.prim_func
        def kernel(
            grad_hidden: T.Tensor((item_count, latent_dim), "float32"),
            winner_idx: T.Tensor((item_count, head_count), "int64"),
            runner_idx: T.Tensor((item_count, head_count), "int64"),
            margins: T.Tensor((item_count, head_count), "float32"),
            grad_latent: T.Tensor((item_count, latent_dim), "float32"),
            grad_code: T.Tensor((head_count, cell_count, latent_dim), "float32"),
        ):
            with T.Kernel(item_count, latent_tiles, threads=block_width) as (row, dim_tile):
                tx = T.get_thread_bindings()[0]
                dim = dim_tile * block_width + tx
                if dim < latent_dim:
                    grad_latent[row, dim] = grad_hidden[row, dim]

            with T.Kernel(item_count, head_count, latent_tiles, threads=block_width) as (row, head, dim_tile):
                tx = T.get_thread_bindings()[0]
                dim = dim_tile * block_width + tx
                if dim < latent_dim:
                    winner = T.cast(winner_idx[row, head], "int32")
                    grad_value = grad_hidden[row, dim] * code_scale
                    if train_mode:
                        runner = T.cast(runner_idx[row, head], "int32")
                        alpha = 0.5 / (1.0 + T.abs(margins[row, head]))
                        T.atomic_add(grad_code[head, winner, dim], grad_value * (1.0 - alpha))
                        T.atomic_add(grad_code[head, runner, dim], grad_value * alpha)
                    else:
                        T.atomic_add(grad_code[head, winner, dim], grad_value)

        return kernel

    return code_backward_kernel()


@lru_cache(maxsize=64)
def _trop_router_backward_kernel(
    item_count: int,
    heads: int,
    cells: int,
    code_dim: int,
    code_scale: float,
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
    def router_backward_kernel() -> Any:
        head_count = heads
        cell_count = cells
        latent_dim = code_dim
        block_width = block_d
        latent_tiles = code_blocks

        @T.prim_func
        def kernel(
            grad_hidden: T.Tensor((item_count, latent_dim), "float32"),
            latent: T.Tensor((item_count, latent_dim), "float32"),
            router_weight: T.Tensor((head_count, cell_count, latent_dim), "float32"),
            code: T.Tensor((head_count, cell_count, latent_dim), "float32"),
            winner_idx: T.Tensor((item_count, head_count), "int64"),
            runner_idx: T.Tensor((item_count, head_count), "int64"),
            margins: T.Tensor((item_count, head_count), "float32"),
            grad_latent: T.Tensor((item_count, latent_dim), "float32"),
            grad_router_weight: T.Tensor((head_count, cell_count, latent_dim), "float32"),
            grad_router_bias: T.Tensor((head_count, cell_count), "float32"),
        ):
            with T.Kernel(item_count, head_count, threads=block_width) as (row, head):
                tx = T.get_thread_bindings()[0]
                partial = T.alloc_shared((block_width,), "float32")
                winner = T.cast(winner_idx[row, head], "int32")
                runner = T.cast(runner_idx[row, head], "int32")
                margin = margins[row, head]

                dot = T.alloc_fragment((1,), "float32")
                dot[0] = 0.0
                for dim_tile in T.serial(latent_tiles):
                    dim = dim_tile * block_width + tx
                    if dim < latent_dim:
                        delta_code = code[head, runner, dim] - code[head, winner, dim]
                        dot[0] = dot[0] + grad_hidden[row, dim] * delta_code
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

                denom = (1.0 + T.abs(margin)) * (1.0 + T.abs(margin))
                dalpha = T.alloc_fragment((1,), "float32")
                dalpha[0] = 0.0
                if margin > 0.0:
                    dalpha[0] = -0.5 / denom
                else:
                    if margin < 0.0:
                        dalpha[0] = 0.5 / denom
                grad_margin = partial[0] * code_scale * dalpha[0]
                if tx == 0:
                    T.atomic_add(grad_router_bias[head, winner], grad_margin)
                    T.atomic_add(grad_router_bias[head, runner], -grad_margin)

                for dim_tile in T.serial(latent_tiles):
                    dim = dim_tile * block_width + tx
                    if dim < latent_dim:
                        latent_value = latent[row, dim]
                        T.atomic_add(grad_router_weight[head, winner, dim], grad_margin * latent_value)
                        T.atomic_add(grad_router_weight[head, runner, dim], -grad_margin * latent_value)
                        router_delta = router_weight[head, winner, dim] - router_weight[head, runner, dim]
                        T.atomic_add(grad_latent[row, dim], grad_margin * router_delta)

        return kernel

    return router_backward_kernel()


def _run_trop_forward(
    latent: Tensor,
    router_weight: Tensor,
    router_bias: Tensor,
    code: Tensor,
    *,
    code_scale: float,
    training: bool,
    target: str,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    batch, steps, code_dim = latent.shape
    heads, cells, _ = router_weight.shape
    item_count = batch * steps
    latent_flat = latent.reshape(item_count, code_dim).contiguous()
    weight = router_weight.contiguous()
    bias = router_bias.contiguous()
    code_table = code.contiguous()
    winner_idx = torch.empty((item_count, heads), device=latent.device, dtype=torch.int64)
    runner_idx = torch.empty((item_count, heads), device=latent.device, dtype=torch.int64)
    margins = torch.empty((item_count, heads), device=latent.device, dtype=latent.dtype)
    hidden = torch.empty((item_count, code_dim), device=latent.device, dtype=latent.dtype)
    block_d = _select_block_size(code_dim)
    code_blocks = (code_dim + block_d - 1) // block_d

    hidden_kernel = _trop_hidden_kernel(
        item_count,
        heads,
        cells,
        code_dim,
        float(code_scale),
        bool(training),
        block_d,
        code_blocks,
        target,
    )
    if code_dim >= 128 and _can_materialize_scores(item_count, heads, cells):
        try:
            from .triton_scores import has_triton, trop_scores_triton
        except ImportError:
            use_score_route = False
        else:
            use_score_route = has_triton()
        if use_score_route:
            scores = trop_scores_triton(latent, weight, bias).reshape(item_count, heads, cells).contiguous()
            top2_kernel = _trop_top2_scores_kernel(item_count, heads, cells, target)
            top2_kernel(scores, winner_idx, runner_idx, margins)
        else:
            route_kernel = _trop_route_parallel_kernel(item_count, heads, cells, code_dim, block_d, code_blocks, target)
            route_kernel(latent_flat, weight, bias, winner_idx, runner_idx, margins)
    elif code_dim >= 32:
        route_kernel = _trop_route_parallel_kernel(item_count, heads, cells, code_dim, block_d, code_blocks, target)
        route_kernel(latent_flat, weight, bias, winner_idx, runner_idx, margins)
    else:
        route_kernel = _trop_route_kernel(item_count, heads, cells, code_dim, target)
        route_kernel(latent_flat, weight, bias, winner_idx, runner_idx, margins)
    hidden_kernel(latent_flat, code_table, winner_idx, runner_idx, margins, hidden)
    return (
        hidden.view(batch, steps, code_dim),
        winner_idx.view(batch, steps, heads),
        runner_idx.view(batch, steps, heads),
        margins.view(batch, steps, heads),
    )


class _TropRouteHiddenTileLangFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        latent: Tensor,
        router_weight: Tensor,
        router_bias: Tensor,
        code: Tensor,
        code_scale: float,
        training: bool,
        target: str,
    ) -> tuple[Tensor, Tensor, Tensor]:
        hidden, winner_idx, runner_idx, margins = _run_trop_forward(
            latent,
            router_weight,
            router_bias,
            code,
            code_scale=code_scale,
            training=training,
            target=target,
        )
        ctx.save_for_backward(latent, router_weight, code, winner_idx, runner_idx, margins)
        ctx.code_scale = float(code_scale)
        ctx.training = bool(training)
        ctx.target = target
        ctx.mark_non_differentiable(winner_idx, margins)
        return hidden, winner_idx, margins

    @staticmethod
    def backward(ctx: Any, grad_hidden: Tensor, grad_winner: Tensor | None, grad_margins: Tensor | None) -> tuple[Any, ...]:
        del grad_winner, grad_margins
        latent, router_weight, code, winner_idx, runner_idx, margins = ctx.saved_tensors
        batch, steps, code_dim = latent.shape
        heads, cells, _ = code.shape
        item_count = batch * steps
        grad_flat = grad_hidden.reshape(item_count, code_dim).contiguous().to(torch.float32)
        latent_flat = latent.reshape(item_count, code_dim).contiguous()
        winner_flat = winner_idx.reshape(item_count, heads).contiguous()
        runner_flat = runner_idx.reshape(item_count, heads).contiguous()
        margins_flat = margins.reshape(item_count, heads).contiguous()
        weight = router_weight.contiguous()
        code_table = code.contiguous()

        grad_latent = torch.empty((item_count, code_dim), device=grad_hidden.device, dtype=torch.float32)
        grad_code = torch.zeros_like(code_table)
        grad_router_weight = torch.zeros_like(weight)
        grad_router_bias = torch.zeros((heads, cells), device=grad_hidden.device, dtype=torch.float32)
        block_d = _select_block_size(code_dim)
        code_blocks = (code_dim + block_d - 1) // block_d

        code_kernel = _trop_code_backward_kernel(
            item_count,
            heads,
            cells,
            code_dim,
            ctx.code_scale,
            ctx.training,
            block_d,
            code_blocks,
            ctx.target,
        )
        code_kernel(grad_flat, winner_flat, runner_flat, margins_flat, grad_latent, grad_code)

        if ctx.training:
            router_kernel = _trop_router_backward_kernel(
                item_count,
                heads,
                cells,
                code_dim,
                ctx.code_scale,
                block_d,
                code_blocks,
                ctx.target,
            )
            router_kernel(
                grad_flat,
                latent_flat,
                weight,
                code_table,
                winner_flat,
                runner_flat,
                margins_flat,
                grad_latent,
                grad_router_weight,
                grad_router_bias,
            )

        return grad_latent.view(batch, steps, code_dim), grad_router_weight, grad_router_bias, grad_code, None, None, None


def trop_route_hidden_tilelang(
    latent: Tensor,
    router_weight: Tensor,
    router_bias: Tensor,
    code: Tensor,
    *,
    code_scale: float,
    training: bool = False,
    target: str = "cuda",
) -> tuple[Tensor, Tensor, Tensor]:
    if not has_tilelang():
        raise RuntimeError("TileLang is not installed; install tilelang or use backend='torch'/'auto'")
    if not latent.is_cuda:
        raise ValueError("TileLang route backend requires CUDA tensors")
    if latent.ndim != 3:
        raise ValueError(f"latent must have shape [batch, steps, code_dim], got {tuple(latent.shape)}")
    if router_weight.ndim != 3:
        raise ValueError(f"router_weight must have shape [heads, cells, code_dim], got {tuple(router_weight.shape)}")
    if router_bias.shape != router_weight.shape[:2]:
        raise ValueError(f"router_bias must have shape [heads, cells], got {tuple(router_bias.shape)}")
    if code.shape != router_weight.shape:
        raise ValueError(f"code must match router_weight shape, got {tuple(code.shape)} and {tuple(router_weight.shape)}")

    batch, steps, code_dim = latent.shape
    heads, cells, weight_dim = router_weight.shape
    if weight_dim != code_dim:
        raise ValueError(f"latent code_dim {code_dim} does not match router_weight code_dim {weight_dim}")

    _dtype_name(latent.dtype)
    try:
        hidden, winner_idx, margins = _TropRouteHiddenTileLangFunction.apply(
            latent,
            router_weight,
            router_bias,
            code,
            float(code_scale),
            bool(training),
            target,
        )
    except Exception as exc:
        raise RuntimeError(
            "TileLang backend failed to compile or launch. Ensure a CUDA toolkit compatible with the GPU is first on PATH "
            "and export CC=/usr/bin/gcc CXX=/usr/bin/g++ before running."
        ) from exc

    return hidden, winner_idx, margins
