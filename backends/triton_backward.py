from __future__ import annotations

from typing import Final

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl

    _HAS_TRITON: Final[bool] = True
except ImportError:  # pragma: no cover - optional dependency at runtime
    triton = None
    tl = None
    _HAS_TRITON = False


def has_triton_backward() -> bool:
    return _HAS_TRITON


def _next_power_of_2(value: int) -> int:
    if value < 1:
        return 1
    return 1 << (value - 1).bit_length()


if _HAS_TRITON:

    @triton.jit
    def _recompute_route_margins_kernel(
        latent_ptr,
        router_weight_ptr,
        router_bias_ptr,
        winner_ptr,
        runner_ptr,
        margins_ptr,
        item_count: tl.constexpr,
        heads: tl.constexpr,
        cells: tl.constexpr,
        code_dim: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0)
        head = tl.program_id(axis=1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        row_mask = offs_m < item_count

        winner = tl.load(winner_ptr + offs_m * heads + head, mask=row_mask, other=0).to(tl.int32)
        runner = tl.load(runner_ptr + offs_m * heads + head, mask=row_mask, other=0).to(tl.int32)
        acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

        for d0 in range(0, code_dim, BLOCK_D):
            dim = d0 + offs_d
            dim_mask = dim < code_dim
            latent = tl.load(
                latent_ptr + offs_m[:, None] * code_dim + dim[None, :],
                mask=row_mask[:, None] & dim_mask[None, :],
                other=0.0,
            )
            winner_weight = tl.load(
                router_weight_ptr + (head * cells + winner[:, None]) * code_dim + dim[None, :],
                mask=row_mask[:, None] & dim_mask[None, :],
                other=0.0,
            )
            runner_weight = tl.load(
                router_weight_ptr + (head * cells + runner[:, None]) * code_dim + dim[None, :],
                mask=row_mask[:, None] & dim_mask[None, :],
                other=0.0,
            )
            acc += tl.sum(latent * (winner_weight - runner_weight), axis=1)

        winner_bias = tl.load(router_bias_ptr + head * cells + winner, mask=row_mask, other=0.0)
        runner_bias = tl.load(router_bias_ptr + head * cells + runner, mask=row_mask, other=0.0)
        tl.store(margins_ptr + offs_m * heads + head, acc + winner_bias - runner_bias, mask=row_mask)

    @triton.jit
    def _grad_margin_kernel(
        grad_hidden_ptr,
        code_ptr,
        winner_ptr,
        runner_ptr,
        margins_ptr,
        grad_margin_ptr,
        item_count: tl.constexpr,
        heads: tl.constexpr,
        cells: tl.constexpr,
        code_dim: tl.constexpr,
        code_scale: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0)
        head = tl.program_id(axis=1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        row_mask = offs_m < item_count

        winner = tl.load(winner_ptr + offs_m * heads + head, mask=row_mask, other=0).to(tl.int32)
        runner = tl.load(runner_ptr + offs_m * heads + head, mask=row_mask, other=0).to(tl.int32)
        acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

        for d0 in range(0, code_dim, BLOCK_D):
            dim = d0 + offs_d
            dim_mask = dim < code_dim
            grad = tl.load(
                grad_hidden_ptr + offs_m[:, None] * code_dim + dim[None, :],
                mask=row_mask[:, None] & dim_mask[None, :],
                other=0.0,
            )
            winner_code = tl.load(
                code_ptr + (head * cells + winner[:, None]) * code_dim + dim[None, :],
                mask=row_mask[:, None] & dim_mask[None, :],
                other=0.0,
            )
            runner_code = tl.load(
                code_ptr + (head * cells + runner[:, None]) * code_dim + dim[None, :],
                mask=row_mask[:, None] & dim_mask[None, :],
                other=0.0,
            )
            acc += tl.sum(grad * (runner_code - winner_code), axis=1)

        margin = tl.load(margins_ptr + offs_m * heads + head, mask=row_mask, other=0.0)
        abs_margin = tl.abs(margin)
        denom = (1.0 + abs_margin) * (1.0 + abs_margin)
        dalpha = tl.where(margin > 0.0, -0.5 / denom, tl.where(margin < 0.0, 0.5 / denom, 0.0))
        tl.store(grad_margin_ptr + offs_m * heads + head, acc * code_scale * dalpha, mask=row_mask)

    @triton.jit
    def _grad_code_kernel(
        grad_hidden_ptr,
        winner_ptr,
        runner_ptr,
        margins_ptr,
        grad_code_ptr,
        item_count: tl.constexpr,
        heads: tl.constexpr,
        cells: tl.constexpr,
        code_dim: tl.constexpr,
        code_scale: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        head = tl.program_id(axis=0)
        pid_d = tl.program_id(axis=1)
        offs_c = tl.arange(0, BLOCK_C)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        cell_mask = offs_c < cells
        dim_mask = offs_d < code_dim
        acc = tl.zeros((BLOCK_C, BLOCK_D), dtype=tl.float32)

        for m0 in range(0, item_count, BLOCK_M):
            offs_m = m0 + tl.arange(0, BLOCK_M)
            row_mask = offs_m < item_count
            winner = tl.load(winner_ptr + offs_m * heads + head, mask=row_mask, other=-1).to(tl.int32)
            runner = tl.load(runner_ptr + offs_m * heads + head, mask=row_mask, other=-1).to(tl.int32)
            margin = tl.load(margins_ptr + offs_m * heads + head, mask=row_mask, other=0.0)
            alpha = 0.5 / (1.0 + tl.abs(margin))
            winner_coeff = tl.where(winner[:, None] == offs_c[None, :], 1.0 - alpha[:, None], 0.0)
            runner_coeff = tl.where(runner[:, None] == offs_c[None, :], alpha[:, None], 0.0)
            coeff = tl.where(row_mask[:, None] & cell_mask[None, :], winner_coeff + runner_coeff, 0.0)
            grad = tl.load(
                grad_hidden_ptr + offs_m[:, None] * code_dim + offs_d[None, :],
                mask=row_mask[:, None] & dim_mask[None, :],
                other=0.0,
            )
            acc += tl.dot(tl.trans(coeff), grad, input_precision="ieee")

        tl.store(
            grad_code_ptr + (head * cells + offs_c[:, None]) * code_dim + offs_d[None, :],
            acc * code_scale,
            mask=cell_mask[:, None] & dim_mask[None, :],
        )

    @triton.jit
    def _grad_router_kernel(
        grad_margin_ptr,
        latent_ptr,
        winner_ptr,
        runner_ptr,
        grad_router_weight_ptr,
        grad_router_bias_ptr,
        item_count: tl.constexpr,
        heads: tl.constexpr,
        cells: tl.constexpr,
        code_dim: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        head = tl.program_id(axis=0)
        pid_d = tl.program_id(axis=1)
        offs_c = tl.arange(0, BLOCK_C)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        cell_mask = offs_c < cells
        dim_mask = offs_d < code_dim
        acc_weight = tl.zeros((BLOCK_C, BLOCK_D), dtype=tl.float32)
        acc_bias = tl.zeros((BLOCK_C,), dtype=tl.float32)

        for m0 in range(0, item_count, BLOCK_M):
            offs_m = m0 + tl.arange(0, BLOCK_M)
            row_mask = offs_m < item_count
            winner = tl.load(winner_ptr + offs_m * heads + head, mask=row_mask, other=-1).to(tl.int32)
            runner = tl.load(runner_ptr + offs_m * heads + head, mask=row_mask, other=-1).to(tl.int32)
            grad_margin = tl.load(grad_margin_ptr + offs_m * heads + head, mask=row_mask, other=0.0)
            winner_coeff = tl.where(winner[:, None] == offs_c[None, :], grad_margin[:, None], 0.0)
            runner_coeff = tl.where(runner[:, None] == offs_c[None, :], -grad_margin[:, None], 0.0)
            coeff = tl.where(row_mask[:, None] & cell_mask[None, :], winner_coeff + runner_coeff, 0.0)
            latent = tl.load(
                latent_ptr + offs_m[:, None] * code_dim + offs_d[None, :],
                mask=row_mask[:, None] & dim_mask[None, :],
                other=0.0,
            )
            acc_weight += tl.dot(tl.trans(coeff), latent, input_precision="ieee")
            acc_bias += tl.sum(coeff, axis=0)

        tl.store(
            grad_router_weight_ptr + (head * cells + offs_c[:, None]) * code_dim + offs_d[None, :],
            acc_weight,
            mask=cell_mask[:, None] & dim_mask[None, :],
        )
        tl.store(grad_router_bias_ptr + head * cells + offs_c, acc_bias, mask=cell_mask & (pid_d == 0))

    @triton.jit
    def _grad_code_sparse_kernel(
        grad_hidden_ptr,
        winner_ptr,
        runner_ptr,
        margins_ptr,
        grad_code_ptr,
        item_count: tl.constexpr,
        heads: tl.constexpr,
        cells: tl.constexpr,
        code_dim: tl.constexpr,
        code_scale: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0)
        head = tl.program_id(axis=1)
        pid_d = tl.program_id(axis=2)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        row_mask = offs_m < item_count
        dim_mask = offs_d < code_dim

        winner = tl.load(winner_ptr + offs_m * heads + head, mask=row_mask, other=0).to(tl.int32)
        runner = tl.load(runner_ptr + offs_m * heads + head, mask=row_mask, other=0).to(tl.int32)
        margin = tl.load(margins_ptr + offs_m * heads + head, mask=row_mask, other=0.0)
        alpha = 0.5 / (1.0 + tl.abs(margin))
        grad = tl.load(
            grad_hidden_ptr + offs_m[:, None] * code_dim + offs_d[None, :],
            mask=row_mask[:, None] & dim_mask[None, :],
            other=0.0,
        ) * code_scale

        winner_offsets = (head * cells + winner[:, None]) * code_dim + offs_d[None, :]
        runner_offsets = (head * cells + runner[:, None]) * code_dim + offs_d[None, :]
        mask = row_mask[:, None] & dim_mask[None, :]
        tl.atomic_add(grad_code_ptr + winner_offsets, grad * (1.0 - alpha[:, None]), sem="relaxed", mask=mask)
        tl.atomic_add(grad_code_ptr + runner_offsets, grad * alpha[:, None], sem="relaxed", mask=mask)

    @triton.jit
    def _grad_router_sparse_kernel(
        grad_margin_ptr,
        latent_ptr,
        winner_ptr,
        runner_ptr,
        grad_router_weight_ptr,
        grad_router_bias_ptr,
        item_count: tl.constexpr,
        heads: tl.constexpr,
        cells: tl.constexpr,
        code_dim: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0)
        head = tl.program_id(axis=1)
        pid_d = tl.program_id(axis=2)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        row_mask = offs_m < item_count
        dim_mask = offs_d < code_dim

        winner = tl.load(winner_ptr + offs_m * heads + head, mask=row_mask, other=0).to(tl.int32)
        runner = tl.load(runner_ptr + offs_m * heads + head, mask=row_mask, other=0).to(tl.int32)
        grad_margin = tl.load(grad_margin_ptr + offs_m * heads + head, mask=row_mask, other=0.0)
        latent = tl.load(
            latent_ptr + offs_m[:, None] * code_dim + offs_d[None, :],
            mask=row_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )

        winner_offsets = (head * cells + winner[:, None]) * code_dim + offs_d[None, :]
        runner_offsets = (head * cells + runner[:, None]) * code_dim + offs_d[None, :]
        mask = row_mask[:, None] & dim_mask[None, :]
        update = grad_margin[:, None] * latent
        tl.atomic_add(grad_router_weight_ptr + winner_offsets, update, sem="relaxed", mask=mask)
        tl.atomic_add(grad_router_weight_ptr + runner_offsets, -update, sem="relaxed", mask=mask)
        tl.atomic_add(grad_router_bias_ptr + head * cells + winner, grad_margin, sem="relaxed", mask=row_mask & (pid_d == 0))
        tl.atomic_add(grad_router_bias_ptr + head * cells + runner, -grad_margin, sem="relaxed", mask=row_mask & (pid_d == 0))

    @triton.jit
    def _grad_latent_kernel(
        grad_hidden_ptr,
        grad_margin_ptr,
        router_weight_ptr,
        winner_ptr,
        runner_ptr,
        grad_latent_ptr,
        item_count: tl.constexpr,
        heads: tl.constexpr,
        cells: tl.constexpr,
        code_dim: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0)
        pid_d = tl.program_id(axis=1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        row_mask = offs_m < item_count
        dim_mask = offs_d < code_dim

        acc = tl.load(
            grad_hidden_ptr + offs_m[:, None] * code_dim + offs_d[None, :],
            mask=row_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )
        for head in range(0, heads):
            winner = tl.load(winner_ptr + offs_m * heads + head, mask=row_mask, other=0).to(tl.int32)
            runner = tl.load(runner_ptr + offs_m * heads + head, mask=row_mask, other=0).to(tl.int32)
            grad_margin = tl.load(grad_margin_ptr + offs_m * heads + head, mask=row_mask, other=0.0)
            winner_weight = tl.load(
                router_weight_ptr + (head * cells + winner[:, None]) * code_dim + offs_d[None, :],
                mask=row_mask[:, None] & dim_mask[None, :],
                other=0.0,
            )
            runner_weight = tl.load(
                router_weight_ptr + (head * cells + runner[:, None]) * code_dim + offs_d[None, :],
                mask=row_mask[:, None] & dim_mask[None, :],
                other=0.0,
            )
            acc += grad_margin[:, None] * (winner_weight - runner_weight)

        tl.store(
            grad_latent_ptr + offs_m[:, None] * code_dim + offs_d[None, :],
            acc,
            mask=row_mask[:, None] & dim_mask[None, :],
        )


def trop_exact_route_backward_triton(
    grad_hidden: Tensor,
    latent: Tensor,
    router_weight: Tensor,
    router_bias: Tensor,
    code: Tensor,
    winner_idx: Tensor,
    runner_idx: Tensor,
    *,
    code_scale: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if not _HAS_TRITON:
        raise RuntimeError("Triton is not available")
    if not grad_hidden.is_cuda:
        raise ValueError("trop_exact_route_backward_triton requires CUDA tensors")

    item_count, code_dim = grad_hidden.shape
    heads, cells, weight_dim = router_weight.shape
    if weight_dim != code_dim or code.shape != router_weight.shape:
        raise ValueError("router_weight/code shapes must be [heads, cells, code_dim]")
    if router_bias.shape != (heads, cells):
        raise ValueError(f"router_bias must have shape {(heads, cells)}, got {tuple(router_bias.shape)}")
    if latent.shape != (item_count, code_dim):
        raise ValueError(f"latent must have shape {(item_count, code_dim)}, got {tuple(latent.shape)}")
    if winner_idx.shape != (item_count, heads) or runner_idx.shape != (item_count, heads):
        raise ValueError("winner_idx and runner_idx must have shape [items, heads]")

    grad = grad_hidden.contiguous().to(torch.float32)
    z = latent.contiguous().to(torch.float32)
    weight = router_weight.contiguous().to(torch.float32)
    bias = router_bias.contiguous().to(torch.float32)
    code_table = code.contiguous().to(torch.float32)
    winner = winner_idx.contiguous()
    runner = runner_idx.contiguous()

    use_sparse_selected = cells >= 16
    margins = torch.empty((item_count, heads), device=grad.device, dtype=torch.float32)
    grad_margin = torch.empty((item_count, heads), device=grad.device, dtype=torch.float32)
    grad_latent = torch.empty((item_count, code_dim), device=grad.device, dtype=torch.float32)
    grad_router_weight = torch.zeros_like(weight) if use_sparse_selected else torch.empty_like(weight)
    grad_router_bias = (
        torch.zeros((heads, cells), device=grad.device, dtype=torch.float32)
        if use_sparse_selected
        else torch.empty((heads, cells), device=grad.device, dtype=torch.float32)
    )
    grad_code = torch.zeros_like(code_table) if use_sparse_selected else torch.empty_like(code_table)

    block_m = 64 if item_count >= 8192 else 32
    block_d = min(64, _next_power_of_2(code_dim))
    block_c = _next_power_of_2(cells)

    _recompute_route_margins_kernel[(triton.cdiv(item_count, block_m), heads)](
        z,
        weight,
        bias,
        winner,
        runner,
        margins,
        item_count,
        heads,
        cells,
        code_dim,
        BLOCK_M=block_m,
        BLOCK_D=block_d,
    )

    _grad_margin_kernel[(triton.cdiv(item_count, block_m), heads)](
        grad,
        code_table,
        winner,
        runner,
        margins,
        grad_margin,
        item_count,
        heads,
        cells,
        code_dim,
        float(code_scale),
        BLOCK_M=block_m,
        BLOCK_D=block_d,
    )
    if use_sparse_selected:
        _grad_code_sparse_kernel[(triton.cdiv(item_count, block_m), heads, triton.cdiv(code_dim, block_d))](
            grad,
            winner,
            runner,
            margins,
            grad_code,
            item_count,
            heads,
            cells,
            code_dim,
            float(code_scale),
            BLOCK_M=block_m,
            BLOCK_D=block_d,
        )
    else:
        _grad_code_kernel[(heads, triton.cdiv(code_dim, block_d))](
            grad,
            winner,
            runner,
            margins,
            grad_code,
            item_count,
            heads,
            cells,
            code_dim,
            float(code_scale),
            BLOCK_M=block_m,
            BLOCK_C=block_c,
            BLOCK_D=block_d,
        )
    del margins
    if use_sparse_selected:
        _grad_router_sparse_kernel[(triton.cdiv(item_count, block_m), heads, triton.cdiv(code_dim, block_d))](
            grad_margin,
            z,
            winner,
            runner,
            grad_router_weight,
            grad_router_bias,
            item_count,
            heads,
            cells,
            code_dim,
            BLOCK_M=block_m,
            BLOCK_D=block_d,
        )
    else:
        _grad_router_kernel[(heads, triton.cdiv(code_dim, block_d))](
            grad_margin,
            z,
            winner,
            runner,
            grad_router_weight,
            grad_router_bias,
            item_count,
            heads,
            cells,
            code_dim,
            BLOCK_M=block_m,
            BLOCK_C=block_c,
            BLOCK_D=block_d,
        )
    _grad_latent_kernel[(triton.cdiv(item_count, block_m), triton.cdiv(code_dim, block_d))](
        grad,
        grad_margin,
        weight,
        winner,
        runner,
        grad_latent,
        item_count,
        heads,
        cells,
        code_dim,
        BLOCK_M=block_m,
        BLOCK_D=block_d,
    )
    return grad_latent, grad_router_weight, grad_router_bias, grad_code
