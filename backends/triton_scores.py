from __future__ import annotations

from typing import Final

import torch
from torch import Tensor

from ._utils import next_power_of_2 as _next_power_of_2

try:
    import triton
    import triton.language as tl

    _HAS_TRITON: Final[bool] = True
except ImportError:  # pragma: no cover - optional dependency at runtime
    triton = None
    tl = None
    _HAS_TRITON = False


def has_triton() -> bool:
    return _HAS_TRITON


if _HAS_TRITON:

    @triton.jit
    def _scores_kernel(
        z_ptr,
        w_ptr,
        b_ptr,
        out_ptr,
        num_rows,
        num_cols,
        rank,
        stride_zm,
        stride_zk,
        stride_wn,
        stride_wk,
        stride_om,
        stride_on,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k0 in range(0, rank, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            z = tl.load(
                z_ptr + offs_m[:, None] * stride_zm + offs_k[None, :] * stride_zk,
                mask=(offs_m[:, None] < num_rows) & (offs_k[None, :] < rank),
                other=0.0,
            )
            w = tl.load(
                w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk,
                mask=(offs_n[None, :] < num_cols) & (offs_k[:, None] < rank),
                other=0.0,
            )
            acc += tl.dot(z, w, input_precision="ieee")

        bias = tl.load(b_ptr + offs_n, mask=offs_n < num_cols, other=0.0)
        acc += bias[None, :]

        tl.store(
            out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            acc,
            mask=(offs_m[:, None] < num_rows) & (offs_n[None, :] < num_cols),
        )

    @triton.jit
    def _top2_stream_kernel(
        z_ptr,
        w_ptr,
        b_ptr,
        winner_ptr,
        runner_ptr,
        margins_ptr,
        item_count,
        heads: tl.constexpr,
        cells: tl.constexpr,
        code_dim: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0)
        pid_h = tl.program_id(axis=1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
        row_mask = offs_m < item_count
        head_mask = offs_h < heads

        best = tl.full((BLOCK_M, BLOCK_H), -3.4028234663852886e38, dtype=tl.float32)
        second = tl.full((BLOCK_M, BLOCK_H), -3.4028234663852886e38, dtype=tl.float32)
        best_cell = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.int32)
        second_cell = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.int32)

        for cell in range(0, cells):
            score = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)
            for d0 in range(0, code_dim, BLOCK_D):
                offs_d = d0 + tl.arange(0, BLOCK_D)
                dim_mask = offs_d < code_dim
                z = tl.load(
                    z_ptr + offs_m[:, None] * code_dim + offs_d[None, :],
                    mask=row_mask[:, None] & dim_mask[None, :],
                    other=0.0,
                )
                w = tl.load(
                    w_ptr + (offs_h[None, :] * cells + cell) * code_dim + offs_d[:, None],
                    mask=head_mask[None, :] & dim_mask[:, None],
                    other=0.0,
                )
                score += tl.dot(z, w, input_precision="ieee")

            bias = tl.load(b_ptr + offs_h * cells + cell, mask=head_mask, other=0.0)
            score += bias[None, :]
            is_best = score > best
            is_second = (score > second) & ~is_best
            second_cell = tl.where(is_best, best_cell, tl.where(is_second, cell, second_cell))
            second = tl.where(is_best, best, tl.where(is_second, score, second))
            best_cell = tl.where(is_best, cell, best_cell)
            best = tl.where(is_best, score, best)

        out_offsets = offs_m[:, None] * heads + offs_h[None, :]
        out_mask = row_mask[:, None] & head_mask[None, :]
        tl.store(winner_ptr + out_offsets, best_cell.to(tl.int64), mask=out_mask)
        tl.store(runner_ptr + out_offsets, second_cell.to(tl.int64), mask=out_mask)
        tl.store(margins_ptr + out_offsets, best - second, mask=out_mask)

    @triton.jit
    def _hidden_from_scores_kernel(
        scores_ptr,
        latent_ptr,
        code_ptr,
        hidden_ptr,
        winner_ptr,
        margins_ptr,
        item_count,
        heads: tl.constexpr,
        cells: tl.constexpr,
        code_dim: tl.constexpr,
        code_scale: tl.constexpr,
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
            latent_ptr + offs_m[:, None] * code_dim + offs_d[None, :],
            mask=row_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )

        store_route_mask = row_mask & (pid_d == 0)
        for head in range(0, heads):
            best = tl.full((BLOCK_M,), -3.4028234663852886e38, dtype=tl.float32)
            second = tl.full((BLOCK_M,), -3.4028234663852886e38, dtype=tl.float32)
            best_cell = tl.zeros((BLOCK_M,), dtype=tl.int32)

            for cell in range(0, cells):
                score = tl.load(
                    scores_ptr + (offs_m * heads + head) * cells + cell,
                    mask=row_mask,
                    other=-3.4028234663852886e38,
                )
                is_best = score > best
                is_second = (score > second) & ~is_best
                second = tl.where(is_best, best, tl.where(is_second, score, second))
                best = tl.where(is_best, score, best)
                best_cell = tl.where(is_best, cell, best_cell)

            code_offsets = head * cells * code_dim + best_cell[:, None] * code_dim + offs_d[None, :]
            selected = tl.load(code_ptr + code_offsets, mask=row_mask[:, None] & dim_mask[None, :], other=0.0)
            acc += selected * code_scale

            tl.store(
                winner_ptr + offs_m * heads + head,
                best_cell.to(tl.int64),
                mask=store_route_mask,
            )
            tl.store(
                margins_ptr + offs_m * heads + head,
                best - second,
                mask=store_route_mask,
            )

        tl.store(
            hidden_ptr + offs_m[:, None] * code_dim + offs_d[None, :],
            acc,
            mask=row_mask[:, None] & dim_mask[None, :],
        )

    @triton.jit
    def _hidden_from_scores_train_kernel(
        scores_ptr,
        latent_ptr,
        code_ptr,
        hidden_ptr,
        winner_ptr,
        runner_ptr,
        margins_ptr,
        item_count,
        heads: tl.constexpr,
        cells: tl.constexpr,
        code_dim: tl.constexpr,
        code_scale: tl.constexpr,
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
            latent_ptr + offs_m[:, None] * code_dim + offs_d[None, :],
            mask=row_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )

        store_route_mask = row_mask & (pid_d == 0)
        for head in range(0, heads):
            best = tl.full((BLOCK_M,), -3.4028234663852886e38, dtype=tl.float32)
            second = tl.full((BLOCK_M,), -3.4028234663852886e38, dtype=tl.float32)
            best_cell = tl.zeros((BLOCK_M,), dtype=tl.int32)
            second_cell = tl.zeros((BLOCK_M,), dtype=tl.int32)

            for cell in range(0, cells):
                score = tl.load(
                    scores_ptr + (offs_m * heads + head) * cells + cell,
                    mask=row_mask,
                    other=-3.4028234663852886e38,
                )
                is_best = score > best
                is_second = (score > second) & ~is_best
                second_cell = tl.where(is_best, best_cell, tl.where(is_second, cell, second_cell))
                second = tl.where(is_best, best, tl.where(is_second, score, second))
                best_cell = tl.where(is_best, cell, best_cell)
                best = tl.where(is_best, score, best)

            margin = best - second
            alpha = 0.5 / (1.0 + tl.abs(margin))
            winner_offsets = head * cells * code_dim + best_cell[:, None] * code_dim + offs_d[None, :]
            runner_offsets = head * cells * code_dim + second_cell[:, None] * code_dim + offs_d[None, :]
            winner_code = tl.load(code_ptr + winner_offsets, mask=row_mask[:, None] & dim_mask[None, :], other=0.0)
            runner_code = tl.load(code_ptr + runner_offsets, mask=row_mask[:, None] & dim_mask[None, :], other=0.0)
            acc += (winner_code + alpha[:, None] * (runner_code - winner_code)) * code_scale

            tl.store(
                winner_ptr + offs_m * heads + head,
                best_cell.to(tl.int64),
                mask=store_route_mask,
            )
            tl.store(
                runner_ptr + offs_m * heads + head,
                second_cell.to(tl.int64),
                mask=store_route_mask,
            )
            tl.store(
                margins_ptr + offs_m * heads + head,
                margin,
                mask=store_route_mask,
            )

        tl.store(
            hidden_ptr + offs_m[:, None] * code_dim + offs_d[None, :],
            acc,
            mask=row_mask[:, None] & dim_mask[None, :],
        )

    @triton.jit
    def _fan_basis_coeff_from_scores_kernel(
        scores_ptr,
        coeff_ptr,
        coeff_sum_ptr,
        winner_ptr,
        margins_ptr,
        item_count,
        heads: tl.constexpr,
        cells: tl.constexpr,
        basis_rank: tl.constexpr,
        code_scale: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0)
        pid_r = tl.program_id(axis=1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        row_mask = offs_m < item_count
        rank_mask = offs_r < basis_rank
        coeff_acc = tl.zeros((BLOCK_M, BLOCK_R), dtype=tl.float32)
        store_route_mask = row_mask & (pid_r == 0)

        for head in range(0, heads):
            best = tl.full((BLOCK_M,), -3.4028234663852886e38, dtype=tl.float32)
            second = tl.full((BLOCK_M,), -3.4028234663852886e38, dtype=tl.float32)
            best_cell = tl.zeros((BLOCK_M,), dtype=tl.int32)

            for cell in range(0, cells):
                score = tl.load(
                    scores_ptr + (offs_m * heads + head) * cells + cell,
                    mask=row_mask,
                    other=-3.4028234663852886e38,
                )
                is_best = score > best
                is_second = (score > second) & ~is_best
                second = tl.where(is_best, best, tl.where(is_second, score, second))
                best = tl.where(is_best, score, best)
                best_cell = tl.where(is_best, cell, best_cell)

            coeff_offsets = head * cells * basis_rank + best_cell[:, None] * basis_rank + offs_r[None, :]
            selected = tl.load(coeff_ptr + coeff_offsets, mask=row_mask[:, None] & rank_mask[None, :], other=0.0)
            coeff_acc += selected

            tl.store(
                winner_ptr + offs_m * heads + head,
                best_cell.to(tl.int64),
                mask=store_route_mask,
            )
            tl.store(
                margins_ptr + offs_m * heads + head,
                best - second,
                mask=store_route_mask,
            )

        tl.store(
            coeff_sum_ptr + offs_m[:, None] * basis_rank + offs_r[None, :],
            coeff_acc * code_scale,
            mask=row_mask[:, None] & rank_mask[None, :],
        )


def trop_scores_triton(z: Tensor, router_weight: Tensor, router_bias: Tensor) -> Tensor:
    if not _HAS_TRITON:
        raise RuntimeError("Triton is not available")
    if not z.is_cuda:
        raise ValueError("trop_scores_triton requires CUDA tensors")
    if router_weight.ndim != 3:
        raise ValueError(f"router_weight must have shape [heads, cells, rank], got {tuple(router_weight.shape)}")
    if router_bias.shape != router_weight.shape[:2]:
        raise ValueError(f"router_bias must have shape [heads, cells] matching router_weight, got {tuple(router_bias.shape)}")

    batch, steps, rank = z.shape
    flat_cols = router_bias.numel()
    z_flat = z.reshape(batch * steps, rank).contiguous().to(torch.float32)
    w_flat = router_weight.reshape(flat_cols, rank).contiguous().to(torch.float32)
    b_flat = router_bias.reshape(flat_cols).contiguous().to(torch.float32)
    out = torch.empty((batch * steps, flat_cols), device=z.device, dtype=torch.float32)

    def grid(meta: dict[str, int]) -> tuple[int, int]:
        return triton.cdiv(batch * steps, meta["BLOCK_M"]), triton.cdiv(flat_cols, meta["BLOCK_N"])

    _scores_kernel[grid](
        z_flat,
        w_flat,
        b_flat,
        out,
        batch * steps,
        flat_cols,
        rank,
        z_flat.stride(0),
        z_flat.stride(1),
        w_flat.stride(0),
        w_flat.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=32,
        BLOCK_N=32,
        BLOCK_K=32,
    )
    return out.reshape(batch, steps, *router_bias.shape)


def _requires_grad_path(*tensors: Tensor) -> bool:
    return torch.is_grad_enabled() and any(tensor.requires_grad for tensor in tensors)


def trop_top2_stream_triton(z: Tensor, router_weight: Tensor, router_bias: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    if not _HAS_TRITON:
        raise RuntimeError("Triton is not available")
    if not z.is_cuda:
        raise ValueError("trop_top2_stream_triton requires CUDA tensors")
    if z.ndim != 3:
        raise ValueError(f"z must have shape [batch, steps, code_dim], got {tuple(z.shape)}")
    if router_weight.ndim != 3:
        raise ValueError(f"router_weight must have shape [heads, cells, code_dim], got {tuple(router_weight.shape)}")
    if router_bias.shape != router_weight.shape[:2]:
        raise ValueError(f"router_bias must have shape [heads, cells] matching router_weight, got {tuple(router_bias.shape)}")

    batch, steps, code_dim = z.shape
    heads, cells, weight_dim = router_weight.shape
    if weight_dim != code_dim:
        raise ValueError(f"z code_dim {code_dim} does not match router_weight code_dim {weight_dim}")

    item_count = batch * steps
    z_flat = z.reshape(item_count, code_dim).contiguous().to(torch.float32)
    weight = router_weight.reshape(heads * cells, code_dim).contiguous().to(torch.float32)
    bias = router_bias.reshape(heads * cells).contiguous().to(torch.float32)
    winner_idx = torch.empty((item_count, heads), device=z.device, dtype=torch.int64)
    runner_idx = torch.empty((item_count, heads), device=z.device, dtype=torch.int64)
    margins = torch.empty((item_count, heads), device=z.device, dtype=torch.float32)

    block_m = 16 if code_dim >= 128 else 32
    block_h = 16
    block_d = min(64, max(32, _next_power_of_2(code_dim)))

    _top2_stream_kernel[(triton.cdiv(item_count, block_m), triton.cdiv(heads, block_h))](
        z_flat,
        weight,
        bias,
        winner_idx,
        runner_idx,
        margins,
        item_count,
        heads,
        cells,
        code_dim,
        BLOCK_M=block_m,
        BLOCK_H=block_h,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return (
        winner_idx.view(batch, steps, heads),
        runner_idx.view(batch, steps, heads),
        margins.view(batch, steps, heads),
    )


def trop_route_hidden_triton_eval(
    z: Tensor,
    router_weight: Tensor,
    router_bias: Tensor,
    code: Tensor,
    *,
    code_scale: float,
) -> tuple[Tensor, Tensor, Tensor]:
    if not _HAS_TRITON:
        raise RuntimeError("Triton is not available")
    if _requires_grad_path(z, router_weight, router_bias, code):
        raise RuntimeError("trop_route_hidden_triton_eval is inference-only; use the TileLang backend for autograd")
    if not z.is_cuda:
        raise ValueError("trop_route_hidden_triton_eval requires CUDA tensors")
    if z.ndim != 3:
        raise ValueError(f"z must have shape [batch, steps, code_dim], got {tuple(z.shape)}")
    if router_weight.ndim != 3:
        raise ValueError(f"router_weight must have shape [heads, cells, code_dim], got {tuple(router_weight.shape)}")
    if router_bias.shape != router_weight.shape[:2]:
        raise ValueError(f"router_bias must have shape [heads, cells] matching router_weight, got {tuple(router_bias.shape)}")
    if code.shape != router_weight.shape:
        raise ValueError(f"code must match router_weight shape, got {tuple(code.shape)} and {tuple(router_weight.shape)}")

    batch, steps, code_dim = z.shape
    heads, cells, weight_dim = router_weight.shape
    if weight_dim != code_dim:
        raise ValueError(f"z code_dim {code_dim} does not match router_weight code_dim {weight_dim}")

    item_count = batch * steps
    z_flat = z.reshape(item_count, code_dim).contiguous().to(torch.float32)
    weight = router_weight.contiguous().to(torch.float32)
    bias = router_bias.contiguous().to(torch.float32)
    code_table = code.contiguous().to(torch.float32)
    scores = trop_scores_triton(z, weight, bias).reshape(item_count, heads, cells).contiguous()
    hidden = torch.empty((item_count, code_dim), device=z.device, dtype=torch.float32)
    winner_idx = torch.empty((item_count, heads), device=z.device, dtype=torch.int64)
    margins = torch.empty((item_count, heads), device=z.device, dtype=torch.float32)

    block_d = min(128, _next_power_of_2(code_dim))
    block_m = 4 if block_d >= 128 else 8

    def grid(meta: dict[str, int]) -> tuple[int, int]:
        return triton.cdiv(item_count, meta["BLOCK_M"]), triton.cdiv(code_dim, meta["BLOCK_D"])

    _hidden_from_scores_kernel[grid](
        scores,
        z_flat,
        code_table,
        hidden,
        winner_idx,
        margins,
        item_count,
        heads,
        cells,
        code_dim,
        float(code_scale),
        BLOCK_M=block_m,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return hidden.view(batch, steps, code_dim), winner_idx.view(batch, steps, heads), margins.view(batch, steps, heads)


def trop_route_hidden_triton_train(
    z: Tensor,
    router_weight: Tensor,
    router_bias: Tensor,
    code: Tensor,
    *,
    code_scale: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if not _HAS_TRITON:
        raise RuntimeError("Triton is not available")
    if not z.is_cuda:
        raise ValueError("trop_route_hidden_triton_train requires CUDA tensors")
    if z.ndim != 3:
        raise ValueError(f"z must have shape [batch, steps, code_dim], got {tuple(z.shape)}")
    if router_weight.ndim != 3:
        raise ValueError(f"router_weight must have shape [heads, cells, code_dim], got {tuple(router_weight.shape)}")
    if router_bias.shape != router_weight.shape[:2]:
        raise ValueError(f"router_bias must have shape [heads, cells] matching router_weight, got {tuple(router_bias.shape)}")
    if code.shape != router_weight.shape:
        raise ValueError(f"code must match router_weight shape, got {tuple(code.shape)} and {tuple(router_weight.shape)}")

    batch, steps, code_dim = z.shape
    heads, cells, weight_dim = router_weight.shape
    if weight_dim != code_dim:
        raise ValueError(f"z code_dim {code_dim} does not match router_weight code_dim {weight_dim}")

    item_count = batch * steps
    z_flat = z.reshape(item_count, code_dim).contiguous().to(torch.float32)
    weight = router_weight.contiguous().to(torch.float32)
    bias = router_bias.contiguous().to(torch.float32)
    code_table = code.contiguous().to(torch.float32)
    scores = trop_scores_triton(z, weight, bias).reshape(item_count, heads, cells).contiguous()
    hidden = torch.empty((item_count, code_dim), device=z.device, dtype=torch.float32)
    winner_idx = torch.empty((item_count, heads), device=z.device, dtype=torch.int64)
    runner_idx = torch.empty((item_count, heads), device=z.device, dtype=torch.int64)
    margins = torch.empty((item_count, heads), device=z.device, dtype=torch.float32)

    block_d = min(128, _next_power_of_2(code_dim))
    block_m = 4 if block_d >= 128 else 8

    def grid(meta: dict[str, int]) -> tuple[int, int]:
        return triton.cdiv(item_count, meta["BLOCK_M"]), triton.cdiv(code_dim, meta["BLOCK_D"])

    _hidden_from_scores_train_kernel[grid](
        scores,
        z_flat,
        code_table,
        hidden,
        winner_idx,
        runner_idx,
        margins,
        item_count,
        heads,
        cells,
        code_dim,
        float(code_scale),
        BLOCK_M=block_m,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return (
        hidden.view(batch, steps, code_dim),
        winner_idx.view(batch, steps, heads),
        runner_idx.view(batch, steps, heads),
        margins.view(batch, steps, heads),
    )


def trop_fan_basis_hidden_triton_eval(
    z: Tensor,
    sites: Tensor,
    lifting: Tensor,
    value_coeff: Tensor,
    value_basis: Tensor,
    *,
    code_scale: float,
) -> tuple[Tensor, Tensor, Tensor]:
    if not _HAS_TRITON:
        raise RuntimeError("Triton is not available")
    if _requires_grad_path(z, sites, lifting, value_coeff, value_basis):
        raise RuntimeError("trop_fan_basis_hidden_triton_eval is inference-only; use the TileLang backend for autograd")
    if not z.is_cuda:
        raise ValueError("trop_fan_basis_hidden_triton_eval requires CUDA tensors")
    if z.ndim != 3:
        raise ValueError(f"z must have shape [batch, steps, code_dim], got {tuple(z.shape)}")
    if sites.ndim != 3:
        raise ValueError(f"sites must have shape [heads, cells, code_dim], got {tuple(sites.shape)}")
    if lifting.shape != sites.shape[:2]:
        raise ValueError(f"lifting must have shape [heads, cells] matching sites, got {tuple(lifting.shape)}")
    if value_coeff.ndim != 3 or value_coeff.shape[:2] != sites.shape[:2]:
        raise ValueError(f"value_coeff must have shape [heads, cells, rank], got {tuple(value_coeff.shape)}")
    if value_basis.ndim != 2:
        raise ValueError(f"value_basis must have shape [rank, code_dim], got {tuple(value_basis.shape)}")

    batch, steps, code_dim = z.shape
    heads, cells, site_dim = sites.shape
    basis_rank = value_coeff.shape[2]
    if site_dim != code_dim:
        raise ValueError(f"z code_dim {code_dim} does not match sites code_dim {site_dim}")
    if value_basis.shape != (basis_rank, code_dim):
        raise ValueError(f"value_basis must have shape {(basis_rank, code_dim)}, got {tuple(value_basis.shape)}")

    item_count = batch * steps
    z_flat = z.reshape(item_count, code_dim).contiguous().to(torch.float32)
    site_table = sites.contiguous().to(torch.float32)
    lift = lifting.contiguous().to(torch.float32)
    coeff = value_coeff.contiguous().to(torch.float32)
    basis = value_basis.contiguous().to(torch.float32)
    scores = trop_scores_triton(z, site_table, lift).reshape(item_count, heads, cells).contiguous()
    coeff_sum = torch.empty((item_count, basis_rank), device=z.device, dtype=torch.float32)
    winner_idx = torch.empty((item_count, heads), device=z.device, dtype=torch.int64)
    margins = torch.empty((item_count, heads), device=z.device, dtype=torch.float32)

    block_r = min(64, _next_power_of_2(basis_rank))
    block_m = 4 if block_r >= 64 else 8

    def grid(meta: dict[str, int]) -> tuple[int, int]:
        return triton.cdiv(item_count, meta["BLOCK_M"]), triton.cdiv(basis_rank, meta["BLOCK_R"])

    _fan_basis_coeff_from_scores_kernel[grid](
        scores,
        coeff,
        coeff_sum,
        winner_idx,
        margins,
        item_count,
        heads,
        cells,
        basis_rank,
        float(code_scale),
        BLOCK_M=block_m,
        BLOCK_R=block_r,
        num_warps=4,
    )
    hidden = torch.addmm(z_flat, coeff_sum, basis)
    return hidden.view(batch, steps, code_dim), winner_idx.view(batch, steps, heads), margins.view(batch, steps, heads)
