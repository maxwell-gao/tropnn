"""Fused CUDA operators for categorical energy-spectrum normal equations."""

from __future__ import annotations

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _categorical_forward_kernel(
        indices_ptr,
        coefficient_ptr,
        output_ptr,
        SAMPLES: tl.constexpr,
        GROUPS: tl.constexpr,
        TARGETS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        targets = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
        row_mask = rows < SAMPLES
        target_mask = targets < TARGETS
        mask = row_mask[:, None] & target_mask[None, :]
        total = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float64)
        for group in range(GROUPS):
            index = tl.load(indices_ptr + rows * GROUPS + group, mask=row_mask, other=0).to(tl.int64)
            total += tl.load(
                coefficient_ptr + index[:, None] * TARGETS + targets[None, :],
                mask=mask,
                other=0.0,
            )
        tl.store(output_ptr + rows[:, None] * TARGETS + targets[None, :], total, mask=mask)


    @triton.jit
    def _categorical_transpose_kernel(
        indices_ptr,
        values_ptr,
        output_ptr,
        SAMPLES: tl.constexpr,
        GROUPS: tl.constexpr,
        TARGETS: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        targets = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
        row_mask = rows < SAMPLES
        target_mask = targets < TARGETS
        mask = row_mask[:, None] & target_mask[None, :]
        values = tl.load(values_ptr + rows[:, None] * TARGETS + targets[None, :], mask=mask, other=0.0)
        for group in range(GROUPS):
            index = tl.load(indices_ptr + rows * GROUPS + group, mask=row_mask, other=0).to(tl.int64)
            tl.atomic_add(
                output_ptr + index[:, None] * TARGETS + targets[None, :],
                values,
                mask=mask,
            )


def _validate(indices: Tensor, values: Tensor) -> None:
    if triton is None:
        raise RuntimeError("Triton is unavailable")
    if not indices.is_cuda or not values.is_cuda or indices.device != values.device:
        raise ValueError("categorical Triton operands must share a CUDA device")
    if indices.dtype != torch.int32 or values.dtype != torch.float64:
        raise ValueError("categorical Triton expects int32 indices and float64 values")
    if indices.ndim != 2 or values.ndim != 2 or not indices.is_contiguous() or not values.is_contiguous():
        raise ValueError("categorical Triton operands must be contiguous matrices")


def categorical_forward(indices: Tensor, coefficient: Tensor) -> Tensor:
    """Compute the raw categorical gather sum on CUDA."""

    _validate(indices, coefficient)
    samples, groups = indices.shape
    targets = coefficient.shape[1]
    output = torch.empty(samples, targets, dtype=torch.float64, device=indices.device)
    block_m = 8
    block_k = min(32, triton.next_power_of_2(targets))
    _categorical_forward_kernel[(triton.cdiv(samples, block_m), triton.cdiv(targets, block_k))](
        indices,
        coefficient,
        output,
        SAMPLES=samples,
        GROUPS=groups,
        TARGETS=targets,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        num_warps=4,
    )
    return output


def categorical_transpose(indices: Tensor, values: Tensor, features: int) -> Tensor:
    """Compute the raw categorical scatter sum on CUDA."""

    _validate(indices, values)
    samples, groups = indices.shape
    targets = values.shape[1]
    output = torch.zeros(features, targets, dtype=torch.float64, device=indices.device)
    block_n = 32
    block_k = min(8, triton.next_power_of_2(targets))
    _categorical_transpose_kernel[(triton.cdiv(samples, block_n), triton.cdiv(targets, block_k))](
        indices,
        values,
        output,
        SAMPLES=samples,
        GROUPS=groups,
        TARGETS=targets,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
    )
    return output
