from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from .pairwise_payload import PackedLutDType, _pack_lut_payload


def has_triton() -> bool:
    try:
        import triton  # noqa: F401
        import triton.language as tl  # noqa: F401
    except ImportError:
        return False
    return True


def _block_d(out_features: int) -> int:
    if out_features <= 32:
        return 32
    if out_features <= 64:
        return 64
    return 128


try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - imported lazily by backend users.
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _surrogate_grad(u, USE_IZHIKEVICH: tl.constexpr):
        if USE_IZHIKEVICH:
            abs_u = tl.minimum(tl.abs(u), 10.0)
            sig = 1.0 / (1.0 + tl.exp(-abs_u))
            return tl.where(u > 0.0, -2.0 * sig * (1.0 - sig), 2.0 * sig * (1.0 - sig))
        denom = (1.0 + tl.abs(u)) * (1.0 + tl.abs(u))
        return tl.where(u > 0.0, -0.5 / denom, tl.where(u < 0.0, 0.5 / denom, 0.0))


    @triton.jit
    def _decode_4bit(packed_ptr, scales_ptr, codebook_ptr, table, idx, offs, mask, TABLE_SIZE: tl.constexpr, PACKED_WIDTH: tl.constexpr):
        pack = offs // 2
        byte = tl.load(packed_ptr + (table * TABLE_SIZE + idx) * PACKED_WIDTH + pack, mask=mask, other=0).to(tl.int32)
        high = byte // 16
        low = byte - high * 16
        code = tl.where(offs - pack * 2 == 0, low, high)
        scale = tl.load(scales_ptr + table * TABLE_SIZE + idx).to(tl.float32)
        return tl.load(codebook_ptr + code, mask=mask, other=0.0).to(tl.float32) * scale


    @triton.jit
    def _route_kernel(
        x_ptr,
        anchors_ptr,
        thresholds_ptr,
        indices_ptr,
        margins_ptr,
        ITEMS: tl.constexpr,
        IN_FEATURES: tl.constexpr,
        TABLES: tl.constexpr,
        COMPARISONS: tl.constexpr,
    ):
        row = tl.program_id(0)
        if row >= ITEMS:
            return
        x_base = x_ptr + row * IN_FEATURES
        for table in range(TABLES):
            idx = 0
            for comp in range(COMPARISONS):
                anchor_base = (table * COMPARISONS + comp) * 2
                a = tl.load(anchors_ptr + anchor_base).to(tl.int32)
                b = tl.load(anchors_ptr + anchor_base + 1).to(tl.int32)
                threshold = tl.load(thresholds_ptr + table * COMPARISONS + comp).to(tl.float32)
                margin = tl.load(x_base + a).to(tl.float32) - tl.load(x_base + b).to(tl.float32) - threshold
                tl.store(margins_ptr + (row * TABLES + table) * COMPARISONS + comp, margin)
                idx = idx | (tl.where(margin > 0.0, 1, 0) << comp)
            tl.store(indices_ptr + row * TABLES + table, idx)


    @triton.jit
    def _gather_float_kernel(
        indices_ptr,
        lut_ptr,
        out_ptr,
        ITEMS: tl.constexpr,
        OUT_FEATURES: tl.constexpr,
        TABLES: tl.constexpr,
        TABLE_SIZE: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0)
        out_block = tl.program_id(1)
        offs = out_block * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = offs < OUT_FEATURES
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for table in range(TABLES):
            idx = tl.load(indices_ptr + row * TABLES + table).to(tl.int32)
            acc += tl.load(lut_ptr + (table * TABLE_SIZE + idx) * OUT_FEATURES + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + row * OUT_FEATURES + offs, acc, mask=mask)


    @triton.jit
    def _gather_int8_kernel(
        indices_ptr,
        codes_ptr,
        scales_ptr,
        out_ptr,
        ITEMS: tl.constexpr,
        OUT_FEATURES: tl.constexpr,
        TABLES: tl.constexpr,
        TABLE_SIZE: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0)
        out_block = tl.program_id(1)
        offs = out_block * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = offs < OUT_FEATURES
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for table in range(TABLES):
            idx = tl.load(indices_ptr + row * TABLES + table).to(tl.int32)
            scale = tl.load(scales_ptr + table * TABLE_SIZE + idx).to(tl.float32)
            code = tl.load(codes_ptr + (table * TABLE_SIZE + idx) * OUT_FEATURES + offs, mask=mask, other=0).to(tl.float32)
            acc += code * scale
        tl.store(out_ptr + row * OUT_FEATURES + offs, acc, mask=mask)


    @triton.jit
    def _gather_4bit_kernel(
        indices_ptr,
        packed_ptr,
        scales_ptr,
        codebook_ptr,
        out_ptr,
        ITEMS: tl.constexpr,
        OUT_FEATURES: tl.constexpr,
        TABLES: tl.constexpr,
        TABLE_SIZE: tl.constexpr,
        PACKED_WIDTH: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0)
        out_block = tl.program_id(1)
        offs = out_block * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = offs < OUT_FEATURES
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for table in range(TABLES):
            idx = tl.load(indices_ptr + row * TABLES + table).to(tl.int32)
            acc += _decode_4bit(packed_ptr, scales_ptr, codebook_ptr, table, idx, offs, mask, TABLE_SIZE, PACKED_WIDTH)
        tl.store(out_ptr + row * OUT_FEATURES + offs, acc, mask=mask)


    @triton.jit
    def _lut_grad_kernel(
        grad_out_ptr,
        indices_ptr,
        grad_lut_ptr,
        ITEMS: tl.constexpr,
        OUT_FEATURES: tl.constexpr,
        TABLES: tl.constexpr,
        TABLE_SIZE: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0)
        table = tl.program_id(1)
        out_block = tl.program_id(2)
        offs = out_block * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = offs < OUT_FEATURES
        idx = tl.load(indices_ptr + row * TABLES + table).to(tl.int32)
        grad = tl.load(grad_out_ptr + row * OUT_FEATURES + offs, mask=mask, other=0.0).to(tl.float32)
        tl.atomic_add(grad_lut_ptr + (table * TABLE_SIZE + idx) * OUT_FEATURES + offs, grad, mask=mask)


    @triton.jit
    def _ste_min_float_kernel(
        grad_out_ptr,
        indices_ptr,
        margins_ptr,
        anchors_ptr,
        lut_ptr,
        grad_x_ptr,
        grad_thresholds_ptr,
        ITEMS: tl.constexpr,
        IN_FEATURES: tl.constexpr,
        OUT_FEATURES: tl.constexpr,
        TABLES: tl.constexpr,
        COMPARISONS: tl.constexpr,
        TABLE_SIZE: tl.constexpr,
        BLOCK_D: tl.constexpr,
        USE_IZHIKEVICH: tl.constexpr,
    ):
        row = tl.program_id(0)
        table = tl.program_id(1)
        r_min = 0
        min_abs = tl.abs(tl.load(margins_ptr + (row * TABLES + table) * COMPARISONS))
        for comp in range(1, COMPARISONS):
            u = tl.load(margins_ptr + (row * TABLES + table) * COMPARISONS + comp)
            abs_u = tl.abs(u)
            if abs_u < min_abs:
                min_abs = abs_u
                r_min = comp
        current_idx = tl.load(indices_ptr + row * TABLES + table).to(tl.int32)
        neighbor_idx = current_idx ^ (1 << r_min)
        offs = tl.arange(0, BLOCK_D)
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for start in range(0, OUT_FEATURES, BLOCK_D):
            d = start + offs
            mask = d < OUT_FEATURES
            grad = tl.load(grad_out_ptr + row * OUT_FEATURES + d, mask=mask, other=0.0).to(tl.float32)
            current = tl.load(lut_ptr + (table * TABLE_SIZE + current_idx) * OUT_FEATURES + d, mask=mask, other=0.0).to(tl.float32)
            neighbor = tl.load(lut_ptr + (table * TABLE_SIZE + neighbor_idx) * OUT_FEATURES + d, mask=mask, other=0.0).to(tl.float32)
            acc += grad * (neighbor - current)
        grad_margin = tl.sum(acc, axis=0) * _surrogate_grad(tl.load(margins_ptr + (row * TABLES + table) * COMPARISONS + r_min), USE_IZHIKEVICH)
        anchor_base = (table * COMPARISONS + r_min) * 2
        a = tl.load(anchors_ptr + anchor_base).to(tl.int32)
        b = tl.load(anchors_ptr + anchor_base + 1).to(tl.int32)
        tl.atomic_add(grad_x_ptr + row * IN_FEATURES + a, grad_margin)
        tl.atomic_add(grad_x_ptr + row * IN_FEATURES + b, -grad_margin)
        tl.atomic_add(grad_thresholds_ptr + table * COMPARISONS + r_min, -grad_margin)


    @triton.jit
    def _ste_full_float_kernel(
        grad_out_ptr,
        indices_ptr,
        margins_ptr,
        anchors_ptr,
        lut_ptr,
        grad_x_ptr,
        grad_thresholds_ptr,
        ITEMS: tl.constexpr,
        IN_FEATURES: tl.constexpr,
        OUT_FEATURES: tl.constexpr,
        TABLES: tl.constexpr,
        COMPARISONS: tl.constexpr,
        TABLE_SIZE: tl.constexpr,
        BLOCK_D: tl.constexpr,
        USE_IZHIKEVICH: tl.constexpr,
    ):
        row = tl.program_id(0)
        table = tl.program_id(1)
        comp = tl.program_id(2)
        current_idx = tl.load(indices_ptr + row * TABLES + table).to(tl.int32)
        neighbor_idx = current_idx ^ (1 << comp)
        offs = tl.arange(0, BLOCK_D)
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for start in range(0, OUT_FEATURES, BLOCK_D):
            d = start + offs
            mask = d < OUT_FEATURES
            grad = tl.load(grad_out_ptr + row * OUT_FEATURES + d, mask=mask, other=0.0).to(tl.float32)
            current = tl.load(lut_ptr + (table * TABLE_SIZE + current_idx) * OUT_FEATURES + d, mask=mask, other=0.0).to(tl.float32)
            neighbor = tl.load(lut_ptr + (table * TABLE_SIZE + neighbor_idx) * OUT_FEATURES + d, mask=mask, other=0.0).to(tl.float32)
            acc += grad * (neighbor - current)
        u = tl.load(margins_ptr + (row * TABLES + table) * COMPARISONS + comp)
        grad_margin = tl.sum(acc, axis=0) * _surrogate_grad(u, USE_IZHIKEVICH)
        anchor_base = (table * COMPARISONS + comp) * 2
        a = tl.load(anchors_ptr + anchor_base).to(tl.int32)
        b = tl.load(anchors_ptr + anchor_base + 1).to(tl.int32)
        tl.atomic_add(grad_x_ptr + row * IN_FEATURES + a, grad_margin)
        tl.atomic_add(grad_x_ptr + row * IN_FEATURES + b, -grad_margin)
        tl.atomic_add(grad_thresholds_ptr + table * COMPARISONS + comp, -grad_margin)


    @triton.jit
    def _ste_min_int8_kernel(
        grad_out_ptr,
        indices_ptr,
        margins_ptr,
        anchors_ptr,
        codes_ptr,
        scales_ptr,
        grad_x_ptr,
        grad_thresholds_ptr,
        ITEMS: tl.constexpr,
        IN_FEATURES: tl.constexpr,
        OUT_FEATURES: tl.constexpr,
        TABLES: tl.constexpr,
        COMPARISONS: tl.constexpr,
        TABLE_SIZE: tl.constexpr,
        BLOCK_D: tl.constexpr,
        USE_IZHIKEVICH: tl.constexpr,
    ):
        row = tl.program_id(0)
        table = tl.program_id(1)
        r_min = 0
        min_abs = tl.abs(tl.load(margins_ptr + (row * TABLES + table) * COMPARISONS))
        for comp in range(1, COMPARISONS):
            u = tl.load(margins_ptr + (row * TABLES + table) * COMPARISONS + comp)
            abs_u = tl.abs(u)
            if abs_u < min_abs:
                min_abs = abs_u
                r_min = comp
        current_idx = tl.load(indices_ptr + row * TABLES + table).to(tl.int32)
        neighbor_idx = current_idx ^ (1 << r_min)
        current_scale = tl.load(scales_ptr + table * TABLE_SIZE + current_idx).to(tl.float32)
        neighbor_scale = tl.load(scales_ptr + table * TABLE_SIZE + neighbor_idx).to(tl.float32)
        offs = tl.arange(0, BLOCK_D)
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for start in range(0, OUT_FEATURES, BLOCK_D):
            d = start + offs
            mask = d < OUT_FEATURES
            grad = tl.load(grad_out_ptr + row * OUT_FEATURES + d, mask=mask, other=0.0).to(tl.float32)
            current = tl.load(codes_ptr + (table * TABLE_SIZE + current_idx) * OUT_FEATURES + d, mask=mask, other=0).to(tl.float32) * current_scale
            neighbor = tl.load(codes_ptr + (table * TABLE_SIZE + neighbor_idx) * OUT_FEATURES + d, mask=mask, other=0).to(tl.float32) * neighbor_scale
            acc += grad * (neighbor - current)
        grad_margin = tl.sum(acc, axis=0) * _surrogate_grad(tl.load(margins_ptr + (row * TABLES + table) * COMPARISONS + r_min), USE_IZHIKEVICH)
        anchor_base = (table * COMPARISONS + r_min) * 2
        a = tl.load(anchors_ptr + anchor_base).to(tl.int32)
        b = tl.load(anchors_ptr + anchor_base + 1).to(tl.int32)
        tl.atomic_add(grad_x_ptr + row * IN_FEATURES + a, grad_margin)
        tl.atomic_add(grad_x_ptr + row * IN_FEATURES + b, -grad_margin)
        tl.atomic_add(grad_thresholds_ptr + table * COMPARISONS + r_min, -grad_margin)


    @triton.jit
    def _ste_full_int8_kernel(
        grad_out_ptr,
        indices_ptr,
        margins_ptr,
        anchors_ptr,
        codes_ptr,
        scales_ptr,
        grad_x_ptr,
        grad_thresholds_ptr,
        ITEMS: tl.constexpr,
        IN_FEATURES: tl.constexpr,
        OUT_FEATURES: tl.constexpr,
        TABLES: tl.constexpr,
        COMPARISONS: tl.constexpr,
        TABLE_SIZE: tl.constexpr,
        BLOCK_D: tl.constexpr,
        USE_IZHIKEVICH: tl.constexpr,
    ):
        row = tl.program_id(0)
        table = tl.program_id(1)
        comp = tl.program_id(2)
        current_idx = tl.load(indices_ptr + row * TABLES + table).to(tl.int32)
        neighbor_idx = current_idx ^ (1 << comp)
        current_scale = tl.load(scales_ptr + table * TABLE_SIZE + current_idx).to(tl.float32)
        neighbor_scale = tl.load(scales_ptr + table * TABLE_SIZE + neighbor_idx).to(tl.float32)
        offs = tl.arange(0, BLOCK_D)
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for start in range(0, OUT_FEATURES, BLOCK_D):
            d = start + offs
            mask = d < OUT_FEATURES
            grad = tl.load(grad_out_ptr + row * OUT_FEATURES + d, mask=mask, other=0.0).to(tl.float32)
            current = tl.load(codes_ptr + (table * TABLE_SIZE + current_idx) * OUT_FEATURES + d, mask=mask, other=0).to(tl.float32) * current_scale
            neighbor = tl.load(codes_ptr + (table * TABLE_SIZE + neighbor_idx) * OUT_FEATURES + d, mask=mask, other=0).to(tl.float32) * neighbor_scale
            acc += grad * (neighbor - current)
        u = tl.load(margins_ptr + (row * TABLES + table) * COMPARISONS + comp)
        grad_margin = tl.sum(acc, axis=0) * _surrogate_grad(u, USE_IZHIKEVICH)
        anchor_base = (table * COMPARISONS + comp) * 2
        a = tl.load(anchors_ptr + anchor_base).to(tl.int32)
        b = tl.load(anchors_ptr + anchor_base + 1).to(tl.int32)
        tl.atomic_add(grad_x_ptr + row * IN_FEATURES + a, grad_margin)
        tl.atomic_add(grad_x_ptr + row * IN_FEATURES + b, -grad_margin)
        tl.atomic_add(grad_thresholds_ptr + table * COMPARISONS + comp, -grad_margin)


    @triton.jit
    def _ste_min_4bit_kernel(
        grad_out_ptr,
        indices_ptr,
        margins_ptr,
        anchors_ptr,
        packed_ptr,
        scales_ptr,
        codebook_ptr,
        grad_x_ptr,
        grad_thresholds_ptr,
        ITEMS: tl.constexpr,
        IN_FEATURES: tl.constexpr,
        OUT_FEATURES: tl.constexpr,
        TABLES: tl.constexpr,
        COMPARISONS: tl.constexpr,
        TABLE_SIZE: tl.constexpr,
        PACKED_WIDTH: tl.constexpr,
        BLOCK_D: tl.constexpr,
        USE_IZHIKEVICH: tl.constexpr,
    ):
        row = tl.program_id(0)
        table = tl.program_id(1)
        r_min = 0
        min_abs = tl.abs(tl.load(margins_ptr + (row * TABLES + table) * COMPARISONS))
        for comp in range(1, COMPARISONS):
            u = tl.load(margins_ptr + (row * TABLES + table) * COMPARISONS + comp)
            abs_u = tl.abs(u)
            if abs_u < min_abs:
                min_abs = abs_u
                r_min = comp
        current_idx = tl.load(indices_ptr + row * TABLES + table).to(tl.int32)
        neighbor_idx = current_idx ^ (1 << r_min)
        offs = tl.arange(0, BLOCK_D)
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for start in range(0, OUT_FEATURES, BLOCK_D):
            d = start + offs
            mask = d < OUT_FEATURES
            grad = tl.load(grad_out_ptr + row * OUT_FEATURES + d, mask=mask, other=0.0).to(tl.float32)
            current = _decode_4bit(packed_ptr, scales_ptr, codebook_ptr, table, current_idx, d, mask, TABLE_SIZE, PACKED_WIDTH)
            neighbor = _decode_4bit(packed_ptr, scales_ptr, codebook_ptr, table, neighbor_idx, d, mask, TABLE_SIZE, PACKED_WIDTH)
            acc += grad * (neighbor - current)
        grad_margin = tl.sum(acc, axis=0) * _surrogate_grad(tl.load(margins_ptr + (row * TABLES + table) * COMPARISONS + r_min), USE_IZHIKEVICH)
        anchor_base = (table * COMPARISONS + r_min) * 2
        a = tl.load(anchors_ptr + anchor_base).to(tl.int32)
        b = tl.load(anchors_ptr + anchor_base + 1).to(tl.int32)
        tl.atomic_add(grad_x_ptr + row * IN_FEATURES + a, grad_margin)
        tl.atomic_add(grad_x_ptr + row * IN_FEATURES + b, -grad_margin)
        tl.atomic_add(grad_thresholds_ptr + table * COMPARISONS + r_min, -grad_margin)


    @triton.jit
    def _ste_full_4bit_kernel(
        grad_out_ptr,
        indices_ptr,
        margins_ptr,
        anchors_ptr,
        packed_ptr,
        scales_ptr,
        codebook_ptr,
        grad_x_ptr,
        grad_thresholds_ptr,
        ITEMS: tl.constexpr,
        IN_FEATURES: tl.constexpr,
        OUT_FEATURES: tl.constexpr,
        TABLES: tl.constexpr,
        COMPARISONS: tl.constexpr,
        TABLE_SIZE: tl.constexpr,
        PACKED_WIDTH: tl.constexpr,
        BLOCK_D: tl.constexpr,
        USE_IZHIKEVICH: tl.constexpr,
    ):
        row = tl.program_id(0)
        table = tl.program_id(1)
        comp = tl.program_id(2)
        current_idx = tl.load(indices_ptr + row * TABLES + table).to(tl.int32)
        neighbor_idx = current_idx ^ (1 << comp)
        offs = tl.arange(0, BLOCK_D)
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for start in range(0, OUT_FEATURES, BLOCK_D):
            d = start + offs
            mask = d < OUT_FEATURES
            grad = tl.load(grad_out_ptr + row * OUT_FEATURES + d, mask=mask, other=0.0).to(tl.float32)
            current = _decode_4bit(packed_ptr, scales_ptr, codebook_ptr, table, current_idx, d, mask, TABLE_SIZE, PACKED_WIDTH)
            neighbor = _decode_4bit(packed_ptr, scales_ptr, codebook_ptr, table, neighbor_idx, d, mask, TABLE_SIZE, PACKED_WIDTH)
            acc += grad * (neighbor - current)
        u = tl.load(margins_ptr + (row * TABLES + table) * COMPARISONS + comp)
        grad_margin = tl.sum(acc, axis=0) * _surrogate_grad(u, USE_IZHIKEVICH)
        anchor_base = (table * COMPARISONS + comp) * 2
        a = tl.load(anchors_ptr + anchor_base).to(tl.int32)
        b = tl.load(anchors_ptr + anchor_base + 1).to(tl.int32)
        tl.atomic_add(grad_x_ptr + row * IN_FEATURES + a, grad_margin)
        tl.atomic_add(grad_x_ptr + row * IN_FEATURES + b, -grad_margin)
        tl.atomic_add(grad_thresholds_ptr + table * COMPARISONS + comp, -grad_margin)


def _run_forward(
    latent: Tensor,
    anchors: Tensor,
    thresholds: Tensor,
    lut: Tensor,
    *,
    lut_dtype: PackedLutDType,
    packed_payload: Any | None = None,
) -> tuple[Tensor, Tensor, Tensor, Any]:
    if triton is None:
        raise RuntimeError("Triton is not installed; install triton or use backend='torch'")
    if not latent.is_cuda:
        raise ValueError("Pairwise Triton backend requires CUDA tensors")
    if latent.dtype not in {torch.float32, torch.bfloat16, torch.float16} or thresholds.dtype != torch.float32:
        raise TypeError("Pairwise Triton backend expects fp32/bf16/fp16 latent and fp32 thresholds")

    batch, steps, in_features = latent.shape
    tables, comparisons, pair_width = anchors.shape
    if pair_width != 2:
        raise ValueError(f"anchors must have shape [tables, comparisons, 2], got {tuple(anchors.shape)}")
    payload = packed_payload if packed_payload is not None else _pack_lut_payload(lut, lut_dtype)
    if payload.mode != lut_dtype:
        raise ValueError("packed_payload mode does not match lut_dtype")
    item_count = batch * steps
    out_features = payload.out_features
    block_d = _block_d(out_features)
    out_blocks = triton.cdiv(out_features, block_d)

    latent_flat = latent.reshape(item_count, in_features).contiguous()
    anchors_flat = anchors.contiguous().reshape(-1).to(torch.int64)
    thresholds_flat = thresholds.contiguous().reshape(-1)
    indices = torch.empty((item_count, tables), device=latent.device, dtype=torch.int64)
    margins = torch.empty((item_count, tables, comparisons), device=latent.device, dtype=torch.float32)
    output = torch.empty((item_count, out_features), device=latent.device, dtype=torch.float32)

    _route_kernel[(item_count,)](
        latent_flat,
        anchors_flat,
        thresholds_flat,
        indices,
        margins,
        item_count,
        in_features,
        tables,
        comparisons,
    )
    grid = (item_count, out_blocks)
    if lut_dtype in {"int8", "fp8"}:
        _gather_int8_kernel[grid](indices, payload.data, payload.scales, output, item_count, out_features, tables, payload.table_size, BLOCK_D=block_d)
    elif lut_dtype in {"fp4", "nf4"}:
        _gather_4bit_kernel[grid](
            indices,
            payload.data,
            payload.scales,
            payload.codebook,
            output,
            item_count,
            out_features,
            tables,
            payload.table_size,
            payload.data.shape[-1],
            BLOCK_D=block_d,
        )
    else:
        _gather_float_kernel[grid](indices, payload.data, output, item_count, out_features, tables, payload.table_size, BLOCK_D=block_d)
    return output.view(batch, steps, out_features), indices.view(batch, steps, tables), margins.view(batch, steps, tables, comparisons), payload


class _PairwiseTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        latent: Tensor,
        anchors: Tensor,
        thresholds: Tensor,
        lut: Tensor,
        use_min_margin_ste: bool,
        surrogate: str,
        lut_dtype: str,
        packed_payload: Any | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        output, indices, margins, payload = _run_forward(latent, anchors, thresholds, lut, lut_dtype=lut_dtype, packed_payload=packed_payload)
        ctx.save_for_backward(indices, margins, anchors.contiguous().reshape(-1).to(torch.int64), payload.data, payload.scales, payload.codebook)
        ctx.latent_shape = tuple(latent.shape)
        ctx.lut_shape = tuple(lut.shape)
        ctx.table_size = payload.table_size
        ctx.out_features = payload.out_features
        ctx.payload_width = int(payload.data.shape[-1]) if payload.data.ndim == 3 else 0
        ctx.use_min_margin_ste = bool(use_min_margin_ste)
        ctx.use_izhikevich_surrogate = surrogate == "izhikevich"
        ctx.lut_dtype = lut_dtype
        ctx.latent_input_dtype = latent.dtype
        ctx.lut_input_dtype = lut.dtype
        ctx.mark_non_differentiable(indices, margins)
        return output, indices, margins

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor, grad_indices: Tensor | None, grad_margins: Tensor | None) -> tuple[Any, ...]:
        del grad_indices, grad_margins
        indices, margins, anchors_flat, payload_data, payload_scales, payload_codebook = ctx.saved_tensors
        batch, steps, in_features = ctx.latent_shape
        item_count = batch * steps
        tables = int(indices.shape[-1])
        comparisons = int(margins.shape[-1])
        table_size = int(ctx.table_size)
        out_features = int(ctx.out_features)
        block_d = _block_d(out_features)
        out_blocks = triton.cdiv(out_features, block_d)

        grad_flat = grad_output.reshape(item_count, out_features).contiguous().to(torch.float32)
        indices_flat = indices.reshape(item_count, tables).contiguous()
        margins_flat = margins.reshape(item_count, tables, comparisons).contiguous()
        grad_x = torch.zeros((item_count, in_features), device=grad_output.device, dtype=torch.float32)
        grad_thresholds = torch.zeros((tables, comparisons), device=grad_output.device, dtype=torch.float32)
        grad_lut = torch.zeros(ctx.lut_shape, device=grad_output.device, dtype=torch.float32)

        _lut_grad_kernel[(item_count, tables, out_blocks)](
            grad_flat,
            indices_flat,
            grad_lut,
            item_count,
            out_features,
            tables,
            table_size,
            BLOCK_D=block_d,
        )
        if ctx.lut_dtype in {"fp32", "bf16", "fp16"}:
            kernel = _ste_min_float_kernel if ctx.use_min_margin_ste else _ste_full_float_kernel
            grid = (item_count, tables) if ctx.use_min_margin_ste else (item_count, tables, comparisons)
            kernel[grid](
                grad_flat,
                indices_flat,
                margins_flat,
                anchors_flat,
                payload_data,
                grad_x,
                grad_thresholds,
                item_count,
                in_features,
                out_features,
                tables,
                comparisons,
                table_size,
                BLOCK_D=block_d,
                USE_IZHIKEVICH=ctx.use_izhikevich_surrogate,
            )
        elif ctx.lut_dtype in {"int8", "fp8"}:
            kernel = _ste_min_int8_kernel if ctx.use_min_margin_ste else _ste_full_int8_kernel
            grid = (item_count, tables) if ctx.use_min_margin_ste else (item_count, tables, comparisons)
            kernel[grid](
                grad_flat,
                indices_flat,
                margins_flat,
                anchors_flat,
                payload_data,
                payload_scales,
                grad_x,
                grad_thresholds,
                item_count,
                in_features,
                out_features,
                tables,
                comparisons,
                table_size,
                BLOCK_D=block_d,
                USE_IZHIKEVICH=ctx.use_izhikevich_surrogate,
            )
        else:
            kernel = _ste_min_4bit_kernel if ctx.use_min_margin_ste else _ste_full_4bit_kernel
            grid = (item_count, tables) if ctx.use_min_margin_ste else (item_count, tables, comparisons)
            kernel[grid](
                grad_flat,
                indices_flat,
                margins_flat,
                anchors_flat,
                payload_data,
                payload_scales,
                payload_codebook,
                grad_x,
                grad_thresholds,
                item_count,
                in_features,
                out_features,
                tables,
                comparisons,
                table_size,
                ctx.payload_width,
                BLOCK_D=block_d,
                USE_IZHIKEVICH=ctx.use_izhikevich_surrogate,
            )
        return grad_x.view(batch, steps, in_features).to(ctx.latent_input_dtype), None, grad_thresholds, grad_lut.to(dtype=ctx.lut_input_dtype), None, None, None, None


def pairwise_triton(
    latent: Tensor,
    anchors: Tensor,
    thresholds: Tensor,
    lut: Tensor,
    *,
    use_min_margin_ste: bool,
    surrogate: str = "fast_sigmoid_odd",
    lut_dtype: PackedLutDType = "bf16",
    packed_payload: Any | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    if surrogate not in {"fast_sigmoid_odd", "izhikevich"}:
        raise ValueError(f"Pairwise Triton backend does not support surrogate={surrogate!r}")
    if lut_dtype not in {"fp32", "bf16", "fp16", "int8", "fp8", "fp4", "nf4"}:
        raise ValueError(f"Pairwise Triton backend does not support lut_dtype={lut_dtype!r}")
    return _PairwiseTritonFunction.apply(latent, anchors, thresholds, lut, use_min_margin_ste, surrogate, lut_dtype, packed_payload)
