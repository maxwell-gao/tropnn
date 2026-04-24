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


def trop_scores_triton(z: Tensor, router_weight: Tensor, router_bias: Tensor) -> Tensor:
    if not _HAS_TRITON:
        raise RuntimeError("Triton is not available")
    if not z.is_cuda:
        raise ValueError("trop_scores_triton requires CUDA tensors")
    if router_weight.ndim != 3:
        raise ValueError(f"router_weight must have shape [heads, cells, rank], got {tuple(router_weight.shape)}")
    if router_bias.shape != router_weight.shape[:2]:
        raise ValueError(
            f"router_bias must have shape [heads, cells] matching router_weight, got {tuple(router_bias.shape)}"
        )

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
