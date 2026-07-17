from functools import lru_cache
import os
from typing import Any, Literal

import torch
from torch import Tensor

from ._utils import select_block_size as _select_block_size
from .pairwise_payload import PackedLutDType, _PackedPayload, _pack_lut_payload

RouteIndexDType = Literal["int64", "uint8"]


def _float_payload_dtype_name(lut_dtype: str) -> str:
    if lut_dtype in {"bf16", "binary01_bf16"}:
        return "bfloat16"
    if lut_dtype == "fp16":
        return "float16"
    return "float32"


def _route_input_dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.float32:
        return "float32"
    raise TypeError(f"Pairwise TileLang route expects fp32/bf16/fp16 latent, got {dtype}")


def has_tilelang() -> bool:
    try:
        import tilelang  # noqa: F401
    except ImportError:
        return False
    return True


def _enabled_by_env(name: str, *, default: bool = True) -> bool:
    fallback = "1" if default else "0"
    return os.environ.get(name, fallback).strip().lower() not in {"0", "false", "no", "off"}


def _positive_env_int(name: str, default: int = 1) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    value = int(raw)
    if value < 1:
        raise ValueError(f"{name} must be >= 1")
    return value


def _ste_dot_block_size(out_features: int) -> int:
    value = os.environ.get("TROPNN_TILELANG_STE_BLOCK_D")
    if value is None or value.strip() == "":
        return _select_block_size(out_features)
    block = int(value)
    if block not in {32, 64, 128, 256}:
        raise ValueError("TROPNN_TILELANG_STE_BLOCK_D must be one of 32, 64, 128, or 256")
    return block


def _lut_grad_row_block(item_count: int = 0) -> int:
    value = os.environ.get("TROPNN_TILELANG_LUT_GRAD_ROW_BLOCK")
    if value is None or value.strip() == "":
        return 512 if item_count >= 4096 else 128
    block = int(value)
    if block not in {32, 64, 128, 256, 512}:
        raise ValueError("TROPNN_TILELANG_LUT_GRAD_ROW_BLOCK must be one of 32, 64, 128, 256, or 512")
    return block


def _default_binary_ste_table_group(bits_per_code: int, packed_width: int, tables: int) -> int:
    if bits_per_code == 1:
        group = 16
    elif packed_width <= 128:
        group = 16
    elif packed_width <= 256:
        group = 8
    else:
        group = 4
    return min(group, tables)


@lru_cache(maxsize=64)
def _pairwise_route_kernel(
    item_count: int,
    in_features: int,
    tables: int,
    comparisons: int,
    route_block: int,
    table_blocks: int,
    latent_dtype_name: str,
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
            latent: T.Tensor((item_count, input_dim), latent_dtype_name),
            anchors: T.Tensor((route_count, comp_count, 2), "int64"),
            thresholds: T.Tensor((route_count, comp_count), "float32"),
            indices: T.Tensor((item_count, route_count), "int64"),
            margins: T.Tensor((item_count, route_count, comp_count), "float32"),
            rmins: T.Tensor((item_count, route_count), "uint8"),
        ):
            with T.Kernel(item_count, threads=route_width) as row:
                tx = T.get_thread_bindings()[0]
                idx = T.alloc_fragment((1,), "int32")
                power = T.alloc_fragment((1,), "int32")
                best_r = T.alloc_fragment((1,), "int32")
                best_abs = T.alloc_fragment((1,), "float32")
                for table_tile in T.serial(route_tiles):
                    table = table_tile * route_width + tx
                    if table < route_count:
                        idx[0] = 0
                        power[0] = 1
                        best_r[0] = 0
                        best_abs[0] = 1.0e30
                        for comp in T.serial(comp_count):
                            a = anchors[table, comp, 0]
                            b = anchors[table, comp, 1]
                            margin = T.cast(latent[row, a], "float32") - T.cast(latent[row, b], "float32") - thresholds[table, comp]
                            margins[row, table, comp] = margin
                            abs_margin = T.abs(margin)
                            if abs_margin < best_abs[0]:
                                best_abs[0] = abs_margin
                                best_r[0] = comp
                            if margin > 0.0:
                                idx[0] = idx[0] + power[0]
                            power[0] = power[0] * 2
                        indices[row, table] = idx[0]
                        rmins[row, table] = T.cast(best_r[0], "uint8")

        return kernel

    return route_kernel()


@lru_cache(maxsize=64)
def _pairwise_route_u8_kernel(
    item_count: int,
    in_features: int,
    tables: int,
    comparisons: int,
    route_block: int,
    table_blocks: int,
    latent_dtype_name: str,
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
            latent: T.Tensor((item_count, input_dim), latent_dtype_name),
            anchors: T.Tensor((route_count, comp_count, 2), "int64"),
            thresholds: T.Tensor((route_count, comp_count), "float32"),
            indices: T.Tensor((item_count, route_count), "uint8"),
            margins: T.Tensor((item_count, route_count, comp_count), "float32"),
            rmins: T.Tensor((item_count, route_count), "uint8"),
        ):
            with T.Kernel(item_count, threads=route_width) as row:
                tx = T.get_thread_bindings()[0]
                idx = T.alloc_fragment((1,), "int32")
                power = T.alloc_fragment((1,), "int32")
                best_r = T.alloc_fragment((1,), "int32")
                best_abs = T.alloc_fragment((1,), "float32")
                for table_tile in T.serial(route_tiles):
                    table = table_tile * route_width + tx
                    if table < route_count:
                        idx[0] = 0
                        power[0] = 1
                        best_r[0] = 0
                        best_abs[0] = 1.0e30
                        for comp in T.serial(comp_count):
                            a = anchors[table, comp, 0]
                            b = anchors[table, comp, 1]
                            margin = T.cast(latent[row, a], "float32") - T.cast(latent[row, b], "float32") - thresholds[table, comp]
                            margins[row, table, comp] = margin
                            abs_margin = T.abs(margin)
                            if abs_margin < best_abs[0]:
                                best_abs[0] = abs_margin
                                best_r[0] = comp
                            if margin > 0.0:
                                idx[0] = idx[0] + power[0]
                            power[0] = power[0] * 2
                        indices[row, table] = T.cast(idx[0], "uint8")
                        rmins[row, table] = T.cast(best_r[0], "uint8")

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
    lut_dtype_name: str,
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
        lut_type = lut_dtype_name

        @T.prim_func
        def kernel(
            indices: T.Tensor((item_count, route_count), "int64"),
            lut: T.Tensor((route_count, bucket_count, output_dim), lut_type),
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
                        acc[0] = acc[0] + T.cast(lut[table, idx, out_col], "float32")
                    output[row, out_col] = acc[0]

        return kernel

    return forward_kernel()


@lru_cache(maxsize=64)
def _pairwise_forward_u8_block_kernel(
    item_count: int,
    out_features: int,
    tables: int,
    table_size: int,
    block_d: int,
    out_blocks: int,
    lut_dtype_name: str,
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
        lut_type = lut_dtype_name

        @T.prim_func
        def kernel(
            indices: T.Tensor((item_count, route_count), "uint8"),
            lut: T.Tensor((route_count, bucket_count, output_dim), lut_type),
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
                        acc[0] = acc[0] + T.cast(lut[table, idx, out_col], "float32")
                    output[row, out_col] = acc[0]

        return kernel

    return forward_kernel()


@lru_cache(maxsize=64)
def _pairwise_forward_int8_block_kernel(
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
            codes: T.Tensor((route_count, bucket_count, output_dim), "int8"),
            scales: T.Tensor((route_count, bucket_count), "float32"),
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
                        acc[0] = acc[0] + T.cast(codes[table, idx, out_col], "float32") * scales[table, idx]
                    output[row, out_col] = acc[0]

        return kernel

    return forward_kernel()


@lru_cache(maxsize=64)
def _pairwise_forward_4bit_block_kernel(
    item_count: int,
    out_features: int,
    tables: int,
    table_size: int,
    packed_width: int,
    bits_per_code: int,
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
        pack_dim = packed_width
        bits = bits_per_code
        block_width = block_d

        @T.prim_func
        def kernel(
            indices: T.Tensor((item_count, route_count), "int64"),
            packed: T.Tensor((route_count, bucket_count, pack_dim), "uint8"),
            scales: T.Tensor((route_count, bucket_count), "float32"),
            codebook: T.Tensor((16,), "float32"),
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
                        pack_col = out_col // 8 if bits == 1 else out_col // 4 if bits == 2 else out_col // 2
                        byte = T.cast(packed[table, idx, pack_col], "int32")
                        code = T.alloc_fragment((1,), "int32")
                        if bits == 1:
                            pos = out_col - pack_col * 8
                            divisor = T.alloc_fragment((1,), "int32")
                            divisor[0] = 1
                            for part in T.serial(8):
                                if part < pos:
                                    divisor[0] = divisor[0] * 2
                            code[0] = byte // divisor[0] - (byte // (divisor[0] * 2)) * 2
                        elif bits == 2:
                            pos = out_col - pack_col * 4
                            divisor = T.alloc_fragment((1,), "int32")
                            divisor[0] = 1
                            for part in T.serial(4):
                                if part < pos:
                                    divisor[0] = divisor[0] * 4
                            code[0] = byte // divisor[0] - (byte // (divisor[0] * 4)) * 4
                        else:
                            code[0] = byte // 16
                            if out_col - pack_col * 2 == 0:
                                code[0] = byte - (byte // 16) * 16
                        acc[0] = acc[0] + codebook[code[0]] * scales[table, idx]
                    output[row, out_col] = acc[0]

        return kernel

    return forward_kernel()


@lru_cache(maxsize=64)
def _pairwise_forward_binary01_block_kernel(
    item_count: int,
    out_features: int,
    tables: int,
    table_size: int,
    packed_width: int,
    bits_per_code: int,
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
        pack_dim = packed_width
        bits = bits_per_code
        block_width = block_d

        @T.prim_func
        def kernel(
            indices: T.Tensor((item_count, route_count), "int64"),
            packed: T.Tensor((route_count, bucket_count, pack_dim), "uint8"),
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
                        pack_col = out_col // 8 if bits == 1 else out_col // 4
                        byte = T.cast(packed[table, idx, pack_col], "int32")
                        code = T.alloc_fragment((1,), "int32")
                        if bits == 1:
                            pos = out_col - pack_col * 8
                            code[0] = T.bitwise_and(T.shift_right(byte, pos), 1)
                        else:
                            pos = (out_col - pack_col * 4) * 2
                            code[0] = T.bitwise_and(T.shift_right(byte, pos), 3)
                        acc[0] = acc[0] + T.cast(code[0], "float32")
                    output[row, out_col] = acc[0]

        return kernel

    return forward_kernel()


@lru_cache(maxsize=64)
def _pairwise_forward_binary01_byte_kernel(
    item_count: int,
    out_features: int,
    tables: int,
    table_size: int,
    packed_width: int,
    bits_per_code: int,
    block_p: int,
    pack_blocks: int,
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
        pack_dim = packed_width
        bits = bits_per_code
        block_width = block_p

        @T.prim_func
        def kernel(
            indices: T.Tensor((item_count, route_count), "int64"),
            packed: T.Tensor((route_count, bucket_count, pack_dim), "uint8"),
            output: T.Tensor((item_count, output_dim), "float32"),
        ):
            with T.Kernel(item_count, pack_blocks, threads=block_width) as (row, pack_tile):
                tx = T.get_thread_bindings()[0]
                pack_col = pack_tile * block_width + tx
                acc = T.alloc_fragment((8,), "float32")
                for slot in T.serial(8):
                    acc[slot] = 0.0

                if pack_col < pack_dim:
                    for table in T.serial(route_count):
                        idx = T.cast(indices[row, table], "int32")
                        byte = T.cast(packed[table, idx, pack_col], "int32")
                        if bits == 1:
                            for slot in T.serial(8):
                                d = pack_col * 8 + slot
                                if d < output_dim:
                                    code = T.bitwise_and(T.shift_right(byte, slot), 1)
                                    acc[slot] = acc[slot] + T.cast(code, "float32")
                        else:
                            for slot in T.serial(4):
                                d = pack_col * 4 + slot
                                if d < output_dim:
                                    shift = slot * 2
                                    code = T.bitwise_and(T.shift_right(byte, shift), 3)
                                    acc[slot] = acc[slot] + T.cast(code, "float32")

                    if bits == 1:
                        for slot in T.serial(8):
                            d = pack_col * 8 + slot
                            if d < output_dim:
                                output[row, d] = acc[slot]
                    else:
                        for slot in T.serial(4):
                            d = pack_col * 4 + slot
                            if d < output_dim:
                                output[row, d] = acc[slot]

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
def _pairwise_lut_backward_rowtile_kernel(
    item_count: int,
    out_features: int,
    tables: int,
    table_size: int,
    block_d: int,
    out_blocks: int,
    row_block: int,
    row_blocks: int,
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
        rows_per_block = row_block

        @T.prim_func
        def kernel(
            grad_output: T.Tensor((item_count, output_dim), "float32"),
            indices: T.Tensor((item_count, route_count), "int64"),
            grad_lut: T.Tensor((route_count, bucket_count, output_dim), "float32"),
        ):
            with T.Kernel(row_blocks, route_count, out_blocks, threads=block_width) as (row_tile, table, out_tile):
                tx = T.get_thread_bindings()[0]
                out_col = out_tile * block_width + tx
                bucket_acc = T.alloc_shared((bucket_count, block_width), "float32")
                for code in T.serial(bucket_count):
                    if out_col < output_dim:
                        bucket_acc[code, tx] = 0.0
                T.sync_threads()

                row_start = row_tile * rows_per_block
                for row_offset in T.serial(rows_per_block):
                    row = row_start + row_offset
                    if row < item_count and out_col < output_dim:
                        idx = T.cast(indices[row, table], "int32")
                        bucket_acc[idx, tx] = bucket_acc[idx, tx] + grad_output[row, out_col]
                T.sync_threads()

                for code in T.serial(bucket_count):
                    if out_col < output_dim:
                        value = bucket_acc[code, tx]
                        if value != 0.0:
                            T.atomic_add(grad_lut[table, code, out_col], value)

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
    lut_dtype_name: str,
    use_izhikevich_surrogate: bool,
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
        lut_type = lut_dtype_name
        use_izhikevich = use_izhikevich_surrogate

        @T.prim_func
        def kernel(
            grad_output: T.Tensor((item_count, output_dim), "float32"),
            indices: T.Tensor((item_count, route_count), "int64"),
            margins: T.Tensor((item_count, route_count, comp_count), "float32"),
            rmins: T.Tensor((item_count, route_count), "uint8"),
            anchors: T.Tensor((route_count, comp_count, 2), "int64"),
            lut: T.Tensor((route_count, bucket_count, output_dim), lut_type),
            grad_latent: T.Tensor((item_count, input_dim), "float32"),
            grad_thresholds: T.Tensor((route_count, comp_count), "float32"),
        ):
            with T.Kernel(item_count, route_count, threads=block_width) as (row, table):
                tx = T.get_thread_bindings()[0]
                partial = T.alloc_shared((block_width,), "float32")
                r_min = T.alloc_fragment((1,), "int32")
                r_min[0] = T.cast(rmins[row, table], "int32")

                current_idx = T.cast(indices[row, table], "int32")
                power = T.alloc_fragment((1,), "int32")
                power[0] = 1
                if comp_count >= 2:
                    if r_min[0] == 1:
                        power[0] = 2
                if comp_count >= 3:
                    if r_min[0] == 2:
                        power[0] = 4
                if comp_count >= 4:
                    if r_min[0] == 3:
                        power[0] = 8
                if comp_count >= 5:
                    if r_min[0] == 4:
                        power[0] = 16
                if comp_count >= 6:
                    if r_min[0] == 5:
                        power[0] = 32
                if comp_count >= 7:
                    if r_min[0] == 6:
                        power[0] = 64
                if comp_count >= 8:
                    if r_min[0] == 7:
                        power[0] = 128
                neighbor_idx = T.bitwise_xor(current_idx, power[0])

                dot = T.alloc_fragment((1,), "float32")
                dot[0] = 0.0
                for out_tile in T.serial(output_tiles):
                    out_col = out_tile * block_width + tx
                    if out_col < output_dim:
                        delta = T.cast(lut[table, neighbor_idx, out_col], "float32") - T.cast(lut[table, current_idx, out_col], "float32")
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
                    surr = T.alloc_fragment((1,), "float32")
                    surr[0] = 0.0
                    if use_izhikevich:
                        abs_u = T.alloc_fragment((1,), "float32")
                        abs_u[0] = T.abs(u)
                        if abs_u[0] > 10.0:
                            abs_u[0] = 10.0
                        sig = 1.0 / (1.0 + T.exp(0.0 - abs_u[0]))
                        if u > 0.0:
                            surr[0] = -2.0 * sig * (1.0 - sig)
                        else:
                            surr[0] = 2.0 * sig * (1.0 - sig)
                    else:
                        denom = (1.0 + T.abs(u)) * (1.0 + T.abs(u))
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
    lut_dtype_name: str,
    use_izhikevich_surrogate: bool,
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
        lut_type = lut_dtype_name
        use_izhikevich = use_izhikevich_surrogate

        @T.prim_func
        def kernel(
            grad_output: T.Tensor((item_count, output_dim), "float32"),
            indices: T.Tensor((item_count, route_count), "int64"),
            margins: T.Tensor((item_count, route_count, comp_count), "float32"),
            anchors: T.Tensor((route_count, comp_count, 2), "int64"),
            lut: T.Tensor((route_count, bucket_count, output_dim), lut_type),
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
                        delta = T.cast(lut[table, neighbor_idx, out_col], "float32") - T.cast(lut[table, current_idx, out_col], "float32")
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
                    surr = T.alloc_fragment((1,), "float32")
                    surr[0] = 0.0
                    if use_izhikevich:
                        abs_u = T.alloc_fragment((1,), "float32")
                        abs_u[0] = T.abs(u)
                        if abs_u[0] > 10.0:
                            abs_u[0] = 10.0
                        sig = 1.0 / (1.0 + T.exp(0.0 - abs_u[0]))
                        if u > 0.0:
                            surr[0] = -2.0 * sig * (1.0 - sig)
                        else:
                            surr[0] = 2.0 * sig * (1.0 - sig)
                    else:
                        denom = (1.0 + T.abs(u)) * (1.0 + T.abs(u))
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


@lru_cache(maxsize=64)
def _pairwise_min_backward_binary01_kernel(
    item_count: int,
    in_features: int,
    out_features: int,
    tables: int,
    comparisons: int,
    table_size: int,
    packed_width: int,
    bits_per_code: int,
    block_d: int,
    out_blocks: int,
    use_izhikevich_surrogate: bool,
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
        pack_dim = packed_width
        bits = bits_per_code
        block_width = block_d
        output_tiles = out_blocks
        use_izhikevich = use_izhikevich_surrogate

        @T.prim_func
        def kernel(
            grad_output: T.Tensor((item_count, output_dim), "float32"),
            indices: T.Tensor((item_count, route_count), "int64"),
            margins: T.Tensor((item_count, route_count, comp_count), "float32"),
            rmins: T.Tensor((item_count, route_count), "uint8"),
            anchors: T.Tensor((route_count, comp_count, 2), "int64"),
            packed: T.Tensor((route_count, bucket_count, pack_dim), "uint8"),
            grad_latent: T.Tensor((item_count, input_dim), "float32"),
            grad_thresholds: T.Tensor((route_count, comp_count), "float32"),
        ):
            with T.Kernel(item_count, route_count, threads=block_width) as (row, table):
                tx = T.get_thread_bindings()[0]
                partial = T.alloc_shared((block_width,), "float32")
                r_min = T.alloc_fragment((1,), "int32")
                r_min[0] = T.cast(rmins[row, table], "int32")

                current_idx = T.cast(indices[row, table], "int32")
                power = T.alloc_fragment((1,), "int32")
                power[0] = 1
                if comp_count >= 2:
                    if r_min[0] == 1:
                        power[0] = 2
                if comp_count >= 3:
                    if r_min[0] == 2:
                        power[0] = 4
                if comp_count >= 4:
                    if r_min[0] == 3:
                        power[0] = 8
                if comp_count >= 5:
                    if r_min[0] == 4:
                        power[0] = 16
                if comp_count >= 6:
                    if r_min[0] == 5:
                        power[0] = 32
                if comp_count >= 7:
                    if r_min[0] == 6:
                        power[0] = 64
                if comp_count >= 8:
                    if r_min[0] == 7:
                        power[0] = 128
                neighbor_idx = T.bitwise_xor(current_idx, power[0])

                dot = T.alloc_fragment((1,), "float32")
                dot[0] = 0.0
                for out_tile in T.serial(output_tiles):
                    out_col = out_tile * block_width + tx
                    if out_col < output_dim:
                        pack_col = out_col // 8 if bits == 1 else out_col // 4
                        current_byte = T.cast(packed[table, current_idx, pack_col], "int32")
                        neighbor_byte = T.cast(packed[table, neighbor_idx, pack_col], "int32")
                        current_code = T.alloc_fragment((1,), "int32")
                        neighbor_code = T.alloc_fragment((1,), "int32")
                        if bits == 1:
                            pos = out_col - pack_col * 8
                            current_code[0] = T.bitwise_and(T.shift_right(current_byte, pos), 1)
                            neighbor_code[0] = T.bitwise_and(T.shift_right(neighbor_byte, pos), 1)
                        else:
                            pos = (out_col - pack_col * 4) * 2
                            current_code[0] = T.bitwise_and(T.shift_right(current_byte, pos), 3)
                            neighbor_code[0] = T.bitwise_and(T.shift_right(neighbor_byte, pos), 3)
                        dot[0] = dot[0] + grad_output[row, out_col] * T.cast(neighbor_code[0] - current_code[0], "float32")
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
                    surr = T.alloc_fragment((1,), "float32")
                    surr[0] = 0.0
                    if use_izhikevich:
                        abs_u = T.alloc_fragment((1,), "float32")
                        abs_u[0] = T.abs(u)
                        if abs_u[0] > 10.0:
                            abs_u[0] = 10.0
                        sig = 1.0 / (1.0 + T.exp(0.0 - abs_u[0]))
                        if u > 0.0:
                            surr[0] = -2.0 * sig * (1.0 - sig)
                        else:
                            surr[0] = 2.0 * sig * (1.0 - sig)
                    else:
                        denom = (1.0 + T.abs(u)) * (1.0 + T.abs(u))
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
def _pairwise_min_backward_binary01_byte_kernel(
    item_count: int,
    in_features: int,
    out_features: int,
    tables: int,
    comparisons: int,
    table_size: int,
    packed_width: int,
    bits_per_code: int,
    block_d: int,
    use_izhikevich_surrogate: bool,
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
        pack_dim = packed_width
        bits = bits_per_code
        block_width = block_d
        pack_tiles = (packed_width + block_d - 1) // block_d
        use_izhikevich = use_izhikevich_surrogate

        @T.prim_func
        def kernel(
            grad_output: T.Tensor((item_count, output_dim), "float32"),
            indices: T.Tensor((item_count, route_count), "int64"),
            margins: T.Tensor((item_count, route_count, comp_count), "float32"),
            rmins: T.Tensor((item_count, route_count), "uint8"),
            anchors: T.Tensor((route_count, comp_count, 2), "int64"),
            packed: T.Tensor((route_count, bucket_count, pack_dim), "uint8"),
            grad_latent: T.Tensor((item_count, input_dim), "float32"),
            grad_thresholds: T.Tensor((route_count, comp_count), "float32"),
        ):
            with T.Kernel(item_count, route_count, threads=block_width) as (row, table):
                tx = T.get_thread_bindings()[0]
                partial = T.alloc_shared((block_width,), "float32")
                r_min = T.alloc_fragment((1,), "int32")
                r_min[0] = T.cast(rmins[row, table], "int32")

                current_idx = T.cast(indices[row, table], "int32")
                power = T.alloc_fragment((1,), "int32")
                power[0] = 1
                if comp_count >= 2:
                    if r_min[0] == 1:
                        power[0] = 2
                if comp_count >= 3:
                    if r_min[0] == 2:
                        power[0] = 4
                if comp_count >= 4:
                    if r_min[0] == 3:
                        power[0] = 8
                if comp_count >= 5:
                    if r_min[0] == 4:
                        power[0] = 16
                if comp_count >= 6:
                    if r_min[0] == 5:
                        power[0] = 32
                if comp_count >= 7:
                    if r_min[0] == 6:
                        power[0] = 64
                if comp_count >= 8:
                    if r_min[0] == 7:
                        power[0] = 128
                neighbor_idx = T.bitwise_xor(current_idx, power[0])

                dot = T.alloc_fragment((1,), "float32")
                dot[0] = 0.0
                for pack_tile in T.serial(pack_tiles):
                    pack_col = pack_tile * block_width + tx
                    if pack_col < pack_dim:
                        current_byte = T.cast(packed[table, current_idx, pack_col], "int32")
                        neighbor_byte = T.cast(packed[table, neighbor_idx, pack_col], "int32")
                        changed = T.bitwise_xor(current_byte, neighbor_byte)
                        if changed != 0:
                            if bits == 1:
                                for bit in T.serial(8):
                                    d = pack_col * 8 + bit
                                    if d < output_dim:
                                        current_code = T.bitwise_and(T.shift_right(current_byte, bit), 1)
                                        neighbor_code = T.bitwise_and(T.shift_right(neighbor_byte, bit), 1)
                                        if current_code != neighbor_code:
                                            dot[0] = dot[0] + grad_output[row, d] * T.cast(neighbor_code - current_code, "float32")
                            else:
                                for slot in T.serial(4):
                                    d = pack_col * 4 + slot
                                    if d < output_dim:
                                        shift = slot * 2
                                        current_code = T.bitwise_and(T.shift_right(current_byte, shift), 3)
                                        neighbor_code = T.bitwise_and(T.shift_right(neighbor_byte, shift), 3)
                                        if current_code != neighbor_code:
                                            dot[0] = dot[0] + grad_output[row, d] * T.cast(neighbor_code - current_code, "float32")
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
                    surr = T.alloc_fragment((1,), "float32")
                    surr[0] = 0.0
                    if use_izhikevich:
                        abs_u = T.alloc_fragment((1,), "float32")
                        abs_u[0] = T.abs(u)
                        if abs_u[0] > 10.0:
                            abs_u[0] = 10.0
                        sig = 1.0 / (1.0 + T.exp(0.0 - abs_u[0]))
                        if u > 0.0:
                            surr[0] = -2.0 * sig * (1.0 - sig)
                        else:
                            surr[0] = 2.0 * sig * (1.0 - sig)
                    else:
                        denom = (1.0 + T.abs(u)) * (1.0 + T.abs(u))
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
def _pairwise_min_backward_binary01_byte_strided_kernel(
    item_count: int,
    in_features: int,
    out_features: int,
    tables: int,
    comparisons: int,
    table_size: int,
    packed_width: int,
    bits_per_code: int,
    block_d: int,
    table_stride: int,
    pack_stride: int,
    use_izhikevich_surrogate: bool,
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
        sampled_route_count = (tables + table_stride - 1) // table_stride
        comp_count = comparisons
        bucket_count = table_size
        pack_dim = packed_width
        bits = bits_per_code
        block_width = block_d
        table_step = table_stride
        byte_step = pack_stride
        pack_sample_count = (packed_width + pack_stride - 1) // pack_stride
        pack_tiles = (pack_sample_count + block_d - 1) // block_d
        sample_scale = float(table_stride * pack_stride)
        use_izhikevich = use_izhikevich_surrogate

        @T.prim_func
        def kernel(
            grad_output: T.Tensor((item_count, output_dim), "float32"),
            indices: T.Tensor((item_count, route_count), "int64"),
            margins: T.Tensor((item_count, route_count, comp_count), "float32"),
            rmins: T.Tensor((item_count, route_count), "uint8"),
            anchors: T.Tensor((route_count, comp_count, 2), "int64"),
            packed: T.Tensor((route_count, bucket_count, pack_dim), "uint8"),
            grad_latent: T.Tensor((item_count, input_dim), "float32"),
            grad_thresholds: T.Tensor((route_count, comp_count), "float32"),
        ):
            with T.Kernel(item_count, sampled_route_count, threads=block_width) as (row, table_slot):
                tx = T.get_thread_bindings()[0]
                table = table_slot * table_step
                partial = T.alloc_shared((block_width,), "float32")
                r_min = T.alloc_fragment((1,), "int32")
                r_min[0] = T.cast(rmins[row, table], "int32")

                current_idx = T.cast(indices[row, table], "int32")
                power = T.alloc_fragment((1,), "int32")
                power[0] = 1
                if comp_count >= 2:
                    if r_min[0] == 1:
                        power[0] = 2
                if comp_count >= 3:
                    if r_min[0] == 2:
                        power[0] = 4
                if comp_count >= 4:
                    if r_min[0] == 3:
                        power[0] = 8
                if comp_count >= 5:
                    if r_min[0] == 4:
                        power[0] = 16
                if comp_count >= 6:
                    if r_min[0] == 5:
                        power[0] = 32
                if comp_count >= 7:
                    if r_min[0] == 6:
                        power[0] = 64
                if comp_count >= 8:
                    if r_min[0] == 7:
                        power[0] = 128
                neighbor_idx = T.bitwise_xor(current_idx, power[0])

                dot = T.alloc_fragment((1,), "float32")
                dot[0] = 0.0
                for pack_tile in T.serial(pack_tiles):
                    sample_idx = pack_tile * block_width + tx
                    if sample_idx < pack_sample_count:
                        pack_col = sample_idx * byte_step
                        if pack_col < pack_dim:
                            current_byte = T.cast(packed[table, current_idx, pack_col], "int32")
                            neighbor_byte = T.cast(packed[table, neighbor_idx, pack_col], "int32")
                            changed = T.bitwise_xor(current_byte, neighbor_byte)
                            if changed != 0:
                                if bits == 1:
                                    for bit in T.serial(8):
                                        d = pack_col * 8 + bit
                                        if d < output_dim:
                                            current_code = T.bitwise_and(T.shift_right(current_byte, bit), 1)
                                            neighbor_code = T.bitwise_and(T.shift_right(neighbor_byte, bit), 1)
                                            if current_code != neighbor_code:
                                                dot[0] = dot[0] + grad_output[row, d] * T.cast(neighbor_code - current_code, "float32")
                                else:
                                    for slot in T.serial(4):
                                        d = pack_col * 4 + slot
                                        if d < output_dim:
                                            shift = slot * 2
                                            current_code = T.bitwise_and(T.shift_right(current_byte, shift), 3)
                                            neighbor_code = T.bitwise_and(T.shift_right(neighbor_byte, shift), 3)
                                            if current_code != neighbor_code:
                                                dot[0] = dot[0] + grad_output[row, d] * T.cast(neighbor_code - current_code, "float32")
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
                    surr = T.alloc_fragment((1,), "float32")
                    surr[0] = 0.0
                    if use_izhikevich:
                        abs_u = T.alloc_fragment((1,), "float32")
                        abs_u[0] = T.abs(u)
                        if abs_u[0] > 10.0:
                            abs_u[0] = 10.0
                        sig = 1.0 / (1.0 + T.exp(0.0 - abs_u[0]))
                        if u > 0.0:
                            surr[0] = -2.0 * sig * (1.0 - sig)
                        else:
                            surr[0] = 2.0 * sig * (1.0 - sig)
                    else:
                        denom = (1.0 + T.abs(u)) * (1.0 + T.abs(u))
                        if u > 0.0:
                            surr[0] = -0.5 / denom
                        else:
                            if u < 0.0:
                                surr[0] = 0.5 / denom

                    grad_margin = partial[0] * surr[0] * sample_scale
                    a = anchors[table, r_min[0], 0]
                    b = anchors[table, r_min[0], 1]
                    T.atomic_add(grad_latent[row, a], grad_margin)
                    T.atomic_add(grad_latent[row, b], -grad_margin)
                    T.atomic_add(grad_thresholds[table, r_min[0]], -grad_margin)

        return kernel

    return min_backward_kernel()


@lru_cache(maxsize=64)
def _pairwise_min_backward_binary01_byte_grouped_kernel(
    item_count: int,
    in_features: int,
    out_features: int,
    tables: int,
    comparisons: int,
    table_size: int,
    packed_width: int,
    bits_per_code: int,
    block_d: int,
    table_group_size: int,
    use_izhikevich_surrogate: bool,
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
        table_group = table_group_size
        table_groups = (tables + table_group_size - 1) // table_group_size
        comp_count = comparisons
        bucket_count = table_size
        pack_dim = packed_width
        bits = bits_per_code
        block_width = block_d
        pack_tiles = (packed_width + block_d - 1) // block_d
        use_izhikevich = use_izhikevich_surrogate

        @T.prim_func
        def kernel(
            grad_output: T.Tensor((item_count, output_dim), "float32"),
            indices: T.Tensor((item_count, route_count), "int64"),
            margins: T.Tensor((item_count, route_count, comp_count), "float32"),
            rmins: T.Tensor((item_count, route_count), "uint8"),
            anchors: T.Tensor((route_count, comp_count, 2), "int64"),
            packed: T.Tensor((route_count, bucket_count, pack_dim), "uint8"),
            grad_latent: T.Tensor((item_count, input_dim), "float32"),
            grad_thresholds: T.Tensor((route_count, comp_count), "float32"),
        ):
            with T.Kernel(item_count, table_groups, threads=block_width) as (row, table_group_idx):
                tx = T.get_thread_bindings()[0]
                partial = T.alloc_shared((table_group, block_width), "float32")
                dot = T.alloc_fragment((table_group,), "float32")
                r_min = T.alloc_fragment((table_group,), "int32")
                current_idx = T.alloc_fragment((table_group,), "int32")
                neighbor_idx = T.alloc_fragment((table_group,), "int32")
                grad_slot = T.alloc_fragment((8,), "float32")

                for group_offset in T.serial(table_group):
                    dot[group_offset] = 0.0
                    table = table_group_idx * table_group + group_offset
                    if table < route_count:
                        r_min[group_offset] = T.cast(rmins[row, table], "int32")
                        current_idx[group_offset] = T.cast(indices[row, table], "int32")
                        power = T.alloc_fragment((1,), "int32")
                        power[0] = 1
                        if comp_count >= 2:
                            if r_min[group_offset] == 1:
                                power[0] = 2
                        if comp_count >= 3:
                            if r_min[group_offset] == 2:
                                power[0] = 4
                        if comp_count >= 4:
                            if r_min[group_offset] == 3:
                                power[0] = 8
                        if comp_count >= 5:
                            if r_min[group_offset] == 4:
                                power[0] = 16
                        if comp_count >= 6:
                            if r_min[group_offset] == 5:
                                power[0] = 32
                        if comp_count >= 7:
                            if r_min[group_offset] == 6:
                                power[0] = 64
                        if comp_count >= 8:
                            if r_min[group_offset] == 7:
                                power[0] = 128
                        neighbor_idx[group_offset] = T.bitwise_xor(current_idx[group_offset], power[0])

                for pack_tile in T.serial(pack_tiles):
                    pack_col = pack_tile * block_width + tx
                    if pack_col < pack_dim:
                        if bits == 1:
                            for slot in T.serial(8):
                                d = pack_col * 8 + slot
                                grad_slot[slot] = 0.0
                                if d < output_dim:
                                    grad_slot[slot] = grad_output[row, d]
                            for group_offset in T.serial(table_group):
                                table = table_group_idx * table_group + group_offset
                                if table < route_count:
                                    current_byte = T.cast(packed[table, current_idx[group_offset], pack_col], "int32")
                                    neighbor_byte = T.cast(packed[table, neighbor_idx[group_offset], pack_col], "int32")
                                    changed = T.bitwise_xor(current_byte, neighbor_byte)
                                    if changed != 0:
                                        for slot in T.serial(8):
                                            current_code = T.bitwise_and(T.shift_right(current_byte, slot), 1)
                                            neighbor_code = T.bitwise_and(T.shift_right(neighbor_byte, slot), 1)
                                            if current_code != neighbor_code:
                                                dot[group_offset] = dot[group_offset] + grad_slot[slot] * T.cast(neighbor_code - current_code, "float32")
                        else:
                            for slot in T.serial(4):
                                d = pack_col * 4 + slot
                                grad_slot[slot] = 0.0
                                if d < output_dim:
                                    grad_slot[slot] = grad_output[row, d]
                            for group_offset in T.serial(table_group):
                                table = table_group_idx * table_group + group_offset
                                if table < route_count:
                                    current_byte = T.cast(packed[table, current_idx[group_offset], pack_col], "int32")
                                    neighbor_byte = T.cast(packed[table, neighbor_idx[group_offset], pack_col], "int32")
                                    changed = T.bitwise_xor(current_byte, neighbor_byte)
                                    if changed != 0:
                                        for slot in T.serial(4):
                                            shift = slot * 2
                                            current_code = T.bitwise_and(T.shift_right(current_byte, shift), 3)
                                            neighbor_code = T.bitwise_and(T.shift_right(neighbor_byte, shift), 3)
                                            if current_code != neighbor_code:
                                                dot[group_offset] = dot[group_offset] + grad_slot[slot] * T.cast(neighbor_code - current_code, "float32")

                for group_offset in T.serial(table_group):
                    partial[group_offset, tx] = dot[group_offset]
                T.sync_threads()
                for group_offset in T.serial(table_group):
                    if block_width >= 256:
                        if tx < 128:
                            partial[group_offset, tx] = partial[group_offset, tx] + partial[group_offset, tx + 128]
                        T.sync_threads()
                    if block_width >= 128:
                        if tx < 64:
                            partial[group_offset, tx] = partial[group_offset, tx] + partial[group_offset, tx + 64]
                        T.sync_threads()
                    if block_width >= 64:
                        if tx < 32:
                            partial[group_offset, tx] = partial[group_offset, tx] + partial[group_offset, tx + 32]
                        T.sync_threads()
                    if tx < 16:
                        partial[group_offset, tx] = partial[group_offset, tx] + partial[group_offset, tx + 16]
                    T.sync_threads()
                    if tx < 8:
                        partial[group_offset, tx] = partial[group_offset, tx] + partial[group_offset, tx + 8]
                    T.sync_threads()
                    if tx < 4:
                        partial[group_offset, tx] = partial[group_offset, tx] + partial[group_offset, tx + 4]
                    T.sync_threads()
                    if tx < 2:
                        partial[group_offset, tx] = partial[group_offset, tx] + partial[group_offset, tx + 2]
                    T.sync_threads()
                    if tx < 1:
                        partial[group_offset, tx] = partial[group_offset, tx] + partial[group_offset, tx + 1]
                    T.sync_threads()

                if tx == 0:
                    for group_offset in T.serial(table_group):
                        table = table_group_idx * table_group + group_offset
                        if table < route_count:
                            u = margins[row, table, r_min[group_offset]]
                            surr = T.alloc_fragment((1,), "float32")
                            surr[0] = 0.0
                            if use_izhikevich:
                                abs_u = T.alloc_fragment((1,), "float32")
                                abs_u[0] = T.abs(u)
                                if abs_u[0] > 10.0:
                                    abs_u[0] = 10.0
                                sig = 1.0 / (1.0 + T.exp(0.0 - abs_u[0]))
                                if u > 0.0:
                                    surr[0] = -2.0 * sig * (1.0 - sig)
                                else:
                                    surr[0] = 2.0 * sig * (1.0 - sig)
                            else:
                                denom = (1.0 + T.abs(u)) * (1.0 + T.abs(u))
                                if u > 0.0:
                                    surr[0] = -0.5 / denom
                                else:
                                    if u < 0.0:
                                        surr[0] = 0.5 / denom

                            grad_margin = partial[group_offset, 0] * surr[0]
                            a = anchors[table, r_min[group_offset], 0]
                            b = anchors[table, r_min[group_offset], 1]
                            T.atomic_add(grad_latent[row, a], grad_margin)
                            T.atomic_add(grad_latent[row, b], -grad_margin)
                            T.atomic_add(grad_thresholds[table, r_min[group_offset]], -grad_margin)

        return kernel

    return min_backward_kernel()


@lru_cache(maxsize=64)
def _pairwise_min_backward_int8_kernel(
    item_count: int,
    in_features: int,
    out_features: int,
    tables: int,
    comparisons: int,
    table_size: int,
    block_d: int,
    out_blocks: int,
    use_izhikevich_surrogate: bool,
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
        use_izhikevich = use_izhikevich_surrogate

        @T.prim_func
        def kernel(
            grad_output: T.Tensor((item_count, output_dim), "float32"),
            indices: T.Tensor((item_count, route_count), "int64"),
            margins: T.Tensor((item_count, route_count, comp_count), "float32"),
            anchors: T.Tensor((route_count, comp_count, 2), "int64"),
            codes: T.Tensor((route_count, bucket_count, output_dim), "int8"),
            scales: T.Tensor((route_count, bucket_count), "float32"),
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
                        current = T.cast(codes[table, current_idx, out_col], "float32") * scales[table, current_idx]
                        neighbor = T.cast(codes[table, neighbor_idx, out_col], "float32") * scales[table, neighbor_idx]
                        dot[0] = dot[0] + grad_output[row, out_col] * (neighbor - current)
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
                    surr = T.alloc_fragment((1,), "float32")
                    surr[0] = 0.0
                    if use_izhikevich:
                        abs_u = T.alloc_fragment((1,), "float32")
                        abs_u[0] = T.abs(u)
                        if abs_u[0] > 10.0:
                            abs_u[0] = 10.0
                        sig = 1.0 / (1.0 + T.exp(0.0 - abs_u[0]))
                        if u > 0.0:
                            surr[0] = -2.0 * sig * (1.0 - sig)
                        else:
                            surr[0] = 2.0 * sig * (1.0 - sig)
                    else:
                        denom = (1.0 + T.abs(u)) * (1.0 + T.abs(u))
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
def _pairwise_full_backward_int8_kernel(
    item_count: int,
    in_features: int,
    out_features: int,
    tables: int,
    comparisons: int,
    table_size: int,
    block_d: int,
    out_blocks: int,
    use_izhikevich_surrogate: bool,
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
        use_izhikevich = use_izhikevich_surrogate

        @T.prim_func
        def kernel(
            grad_output: T.Tensor((item_count, output_dim), "float32"),
            indices: T.Tensor((item_count, route_count), "int64"),
            margins: T.Tensor((item_count, route_count, comp_count), "float32"),
            anchors: T.Tensor((route_count, comp_count, 2), "int64"),
            codes: T.Tensor((route_count, bucket_count, output_dim), "int8"),
            scales: T.Tensor((route_count, bucket_count), "float32"),
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
                        current = T.cast(codes[table, current_idx, out_col], "float32") * scales[table, current_idx]
                        neighbor = T.cast(codes[table, neighbor_idx, out_col], "float32") * scales[table, neighbor_idx]
                        dot[0] = dot[0] + grad_output[row, out_col] * (neighbor - current)
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
                    surr = T.alloc_fragment((1,), "float32")
                    surr[0] = 0.0
                    if use_izhikevich:
                        abs_u = T.alloc_fragment((1,), "float32")
                        abs_u[0] = T.abs(u)
                        if abs_u[0] > 10.0:
                            abs_u[0] = 10.0
                        sig = 1.0 / (1.0 + T.exp(0.0 - abs_u[0]))
                        if u > 0.0:
                            surr[0] = -2.0 * sig * (1.0 - sig)
                        else:
                            surr[0] = 2.0 * sig * (1.0 - sig)
                    else:
                        denom = (1.0 + T.abs(u)) * (1.0 + T.abs(u))
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


@lru_cache(maxsize=64)
def _pairwise_min_backward_4bit_kernel(
    item_count: int,
    in_features: int,
    out_features: int,
    tables: int,
    comparisons: int,
    table_size: int,
    packed_width: int,
    bits_per_code: int,
    block_d: int,
    out_blocks: int,
    use_izhikevich_surrogate: bool,
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
        pack_dim = packed_width
        bits = bits_per_code
        block_width = block_d
        output_tiles = out_blocks
        use_izhikevich = use_izhikevich_surrogate

        @T.prim_func
        def kernel(
            grad_output: T.Tensor((item_count, output_dim), "float32"),
            indices: T.Tensor((item_count, route_count), "int64"),
            margins: T.Tensor((item_count, route_count, comp_count), "float32"),
            anchors: T.Tensor((route_count, comp_count, 2), "int64"),
            packed: T.Tensor((route_count, bucket_count, pack_dim), "uint8"),
            scales: T.Tensor((route_count, bucket_count), "float32"),
            codebook: T.Tensor((16,), "float32"),
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
                        pack_col = out_col // 8 if bits == 1 else out_col // 4 if bits == 2 else out_col // 2
                        current_byte = T.cast(packed[table, current_idx, pack_col], "int32")
                        neighbor_byte = T.cast(packed[table, neighbor_idx, pack_col], "int32")
                        current_code = T.alloc_fragment((1,), "int32")
                        neighbor_code = T.alloc_fragment((1,), "int32")
                        if bits == 1:
                            pos = out_col - pack_col * 8
                            divisor = T.alloc_fragment((1,), "int32")
                            divisor[0] = 1
                            for part in T.serial(8):
                                if part < pos:
                                    divisor[0] = divisor[0] * 2
                            current_code[0] = current_byte // divisor[0] - (current_byte // (divisor[0] * 2)) * 2
                            neighbor_code[0] = neighbor_byte // divisor[0] - (neighbor_byte // (divisor[0] * 2)) * 2
                        elif bits == 2:
                            pos = out_col - pack_col * 4
                            divisor = T.alloc_fragment((1,), "int32")
                            divisor[0] = 1
                            for part in T.serial(4):
                                if part < pos:
                                    divisor[0] = divisor[0] * 4
                            current_code[0] = current_byte // divisor[0] - (current_byte // (divisor[0] * 4)) * 4
                            neighbor_code[0] = neighbor_byte // divisor[0] - (neighbor_byte // (divisor[0] * 4)) * 4
                        else:
                            current_code[0] = current_byte // 16
                            neighbor_code[0] = neighbor_byte // 16
                            if out_col - pack_col * 2 == 0:
                                current_code[0] = current_byte - (current_byte // 16) * 16
                                neighbor_code[0] = neighbor_byte - (neighbor_byte // 16) * 16
                        current = codebook[current_code[0]] * scales[table, current_idx]
                        neighbor = codebook[neighbor_code[0]] * scales[table, neighbor_idx]
                        dot[0] = dot[0] + grad_output[row, out_col] * (neighbor - current)
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
                    surr = T.alloc_fragment((1,), "float32")
                    surr[0] = 0.0
                    if use_izhikevich:
                        abs_u = T.alloc_fragment((1,), "float32")
                        abs_u[0] = T.abs(u)
                        if abs_u[0] > 10.0:
                            abs_u[0] = 10.0
                        sig = 1.0 / (1.0 + T.exp(0.0 - abs_u[0]))
                        if u > 0.0:
                            surr[0] = -2.0 * sig * (1.0 - sig)
                        else:
                            surr[0] = 2.0 * sig * (1.0 - sig)
                    else:
                        denom = (1.0 + T.abs(u)) * (1.0 + T.abs(u))
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
def _pairwise_full_backward_4bit_kernel(
    item_count: int,
    in_features: int,
    out_features: int,
    tables: int,
    comparisons: int,
    table_size: int,
    packed_width: int,
    bits_per_code: int,
    block_d: int,
    out_blocks: int,
    use_izhikevich_surrogate: bool,
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
        pack_dim = packed_width
        bits = bits_per_code
        block_width = block_d
        output_tiles = out_blocks
        use_izhikevich = use_izhikevich_surrogate

        @T.prim_func
        def kernel(
            grad_output: T.Tensor((item_count, output_dim), "float32"),
            indices: T.Tensor((item_count, route_count), "int64"),
            margins: T.Tensor((item_count, route_count, comp_count), "float32"),
            anchors: T.Tensor((route_count, comp_count, 2), "int64"),
            packed: T.Tensor((route_count, bucket_count, pack_dim), "uint8"),
            scales: T.Tensor((route_count, bucket_count), "float32"),
            codebook: T.Tensor((16,), "float32"),
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
                        pack_col = out_col // 8 if bits == 1 else out_col // 4 if bits == 2 else out_col // 2
                        current_byte = T.cast(packed[table, current_idx, pack_col], "int32")
                        neighbor_byte = T.cast(packed[table, neighbor_idx, pack_col], "int32")
                        current_code = T.alloc_fragment((1,), "int32")
                        neighbor_code = T.alloc_fragment((1,), "int32")
                        if bits == 1:
                            pos = out_col - pack_col * 8
                            divisor = T.alloc_fragment((1,), "int32")
                            divisor[0] = 1
                            for part in T.serial(8):
                                if part < pos:
                                    divisor[0] = divisor[0] * 2
                            current_code[0] = current_byte // divisor[0] - (current_byte // (divisor[0] * 2)) * 2
                            neighbor_code[0] = neighbor_byte // divisor[0] - (neighbor_byte // (divisor[0] * 2)) * 2
                        elif bits == 2:
                            pos = out_col - pack_col * 4
                            divisor = T.alloc_fragment((1,), "int32")
                            divisor[0] = 1
                            for part in T.serial(4):
                                if part < pos:
                                    divisor[0] = divisor[0] * 4
                            current_code[0] = current_byte // divisor[0] - (current_byte // (divisor[0] * 4)) * 4
                            neighbor_code[0] = neighbor_byte // divisor[0] - (neighbor_byte // (divisor[0] * 4)) * 4
                        else:
                            current_code[0] = current_byte // 16
                            neighbor_code[0] = neighbor_byte // 16
                            if out_col - pack_col * 2 == 0:
                                current_code[0] = current_byte - (current_byte // 16) * 16
                                neighbor_code[0] = neighbor_byte - (neighbor_byte // 16) * 16
                        current = codebook[current_code[0]] * scales[table, current_idx]
                        neighbor = codebook[neighbor_code[0]] * scales[table, neighbor_idx]
                        dot[0] = dot[0] + grad_output[row, out_col] * (neighbor - current)
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
                    surr = T.alloc_fragment((1,), "float32")
                    surr[0] = 0.0
                    if use_izhikevich:
                        abs_u = T.alloc_fragment((1,), "float32")
                        abs_u[0] = T.abs(u)
                        if abs_u[0] > 10.0:
                            abs_u[0] = 10.0
                        sig = 1.0 / (1.0 + T.exp(0.0 - abs_u[0]))
                        if u > 0.0:
                            surr[0] = -2.0 * sig * (1.0 - sig)
                        else:
                            surr[0] = 2.0 * sig * (1.0 - sig)
                    else:
                        denom = (1.0 + T.abs(u)) * (1.0 + T.abs(u))
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
    lut_dtype: PackedLutDType = "bf16",
    packed_payload: _PackedPayload | None = None,
    route_index_dtype: RouteIndexDType = "int64",
) -> tuple[Tensor, Tensor, Tensor, Tensor, _PackedPayload]:
    if not has_tilelang():
        raise RuntimeError("TileLang is not installed; install tilelang or use backend='torch'")
    if not latent.is_cuda:
        raise ValueError("Pairwise TileLang backend requires CUDA tensors")
    if latent.ndim != 3:
        raise ValueError(f"latent must have shape [batch, steps, in_features], got {tuple(latent.shape)}")
    latent_dtype_name = _route_input_dtype_name(latent.dtype)
    if thresholds.dtype != torch.float32:
        raise TypeError(f"Pairwise TileLang route expects fp32 thresholds, got {thresholds.dtype}")

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
    payload = packed_payload if packed_payload is not None else _pack_lut_payload(lut, lut_dtype)
    if payload.mode != lut_dtype or payload.table_size != table_size or payload.out_features != out_features:
        raise ValueError("packed_payload does not match the requested LUT payload layout")
    if route_index_dtype == "uint8" and comparisons > 8:
        raise ValueError("route_index_dtype='uint8' requires comparisons <= 8")
    indices = torch.empty((item_count, tables), device=latent.device, dtype=torch.uint8 if route_index_dtype == "uint8" else torch.int64)
    margins = torch.empty((item_count, tables, comparisons), device=latent.device, dtype=torch.float32)
    rmins = torch.empty((item_count, tables), device=latent.device, dtype=torch.uint8)
    output = torch.empty((item_count, out_features), device=latent.device, dtype=torch.float32)

    route_block = _select_block_size(tables)
    table_blocks = (tables + route_block - 1) // route_block
    block_d = _select_block_size(out_features)
    out_blocks = (out_features + block_d - 1) // block_d
    route_kernel = _pairwise_route_u8_kernel(item_count, in_features, tables, comparisons, route_block, table_blocks, latent_dtype_name, target) if route_index_dtype == "uint8" else _pairwise_route_kernel(item_count, in_features, tables, comparisons, route_block, table_blocks, latent_dtype_name, target)
    route_kernel(latent_flat, anchors_contig, thresholds_contig, indices, margins, rmins)
    if route_index_dtype == "uint8":
        if lut_dtype not in {"fp32", "bf16", "fp16", "binary01_bf16"}:
            raise ValueError("route_index_dtype='uint8' currently supports only fp32/bf16/fp16 payloads")
        payload_dtype_name = _float_payload_dtype_name(lut_dtype)
        forward_kernel = _pairwise_forward_u8_block_kernel(item_count, out_features, tables, table_size, block_d, out_blocks, payload_dtype_name, target)
        forward_kernel(indices, payload.data, output)
    elif lut_dtype in {"int8", "fp8"}:
        forward_kernel = _pairwise_forward_int8_block_kernel(item_count, out_features, tables, table_size, block_d, out_blocks, target)
        forward_kernel(indices, payload.data, payload.scales, output)
    elif lut_dtype in {"binary01_fixed", "binary01_1bit"}:
        if _enabled_by_env("TROPNN_TILELANG_BINARY_FORWARD_BYTE", default=True):
            pack_block = _select_block_size(payload.data.shape[-1])
            pack_blocks = (payload.data.shape[-1] + pack_block - 1) // pack_block
            forward_kernel = _pairwise_forward_binary01_byte_kernel(
                item_count,
                out_features,
                tables,
                table_size,
                payload.data.shape[-1],
                1 if lut_dtype == "binary01_1bit" else 2,
                pack_block,
                pack_blocks,
                target,
            )
        else:
            forward_kernel = _pairwise_forward_binary01_block_kernel(
                item_count,
                out_features,
                tables,
                table_size,
                payload.data.shape[-1],
                1 if lut_dtype == "binary01_1bit" else 2,
                block_d,
                out_blocks,
                target,
            )
        forward_kernel(indices, payload.data, output)
    elif lut_dtype in {"int4", "int2", "ternary_int2", "ternary_fixed", "fp4", "nf4"}:
        forward_kernel = _pairwise_forward_4bit_block_kernel(item_count, out_features, tables, table_size, payload.data.shape[-1], 1 if lut_dtype == "binary01_1bit" else 2 if lut_dtype in {"int2", "ternary_int2", "ternary_fixed", "binary01_fixed"} else 4, block_d, out_blocks, target)
        forward_kernel(indices, payload.data, payload.scales, payload.codebook, output)
    else:
        payload_dtype_name = _float_payload_dtype_name(lut_dtype)
        forward_kernel = _pairwise_forward_block_kernel(item_count, out_features, tables, table_size, block_d, out_blocks, payload_dtype_name, target)
        forward_kernel(indices, payload.data, output)
    return output.view(batch, steps, out_features), indices.view(batch, steps, tables), margins.view(batch, steps, tables, comparisons), rmins.view(batch, steps, tables), payload


class _PairwiseTileLangFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        latent: Tensor,
        anchors: Tensor,
        thresholds: Tensor,
        lut: Tensor,
        use_min_margin_ste: bool,
        surrogate: str,
        target: str,
        lut_dtype: str,
        packed_payload: _PackedPayload | None,
        route_index_dtype: str,
    ) -> tuple[Tensor, Tensor, Tensor]:
        output, indices, margins, rmins, payload = _run_forward(latent, anchors, thresholds, lut, target=target, lut_dtype=lut_dtype, packed_payload=packed_payload, route_index_dtype=route_index_dtype)  # type: ignore[arg-type]
        ctx.save_for_backward(indices, margins, rmins, anchors, payload.data, payload.scales, payload.codebook)
        ctx.latent_shape = tuple(latent.shape)
        ctx.lut_shape = tuple(lut.shape)
        ctx.table_size = payload.table_size
        ctx.out_features = payload.out_features
        ctx.payload_width = int(payload.data.shape[-1]) if payload.data.ndim == 3 else 0
        ctx.lut_dtype = lut_dtype
        ctx.route_index_dtype = route_index_dtype
        ctx.use_min_margin_ste = bool(use_min_margin_ste)
        ctx.use_izhikevich_surrogate = surrogate == "izhikevich"
        ctx.target = target
        ctx.latent_input_dtype = latent.dtype
        ctx.lut_input_dtype = lut.dtype
        ctx.mark_non_differentiable(indices, margins)
        return output, indices, margins

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor, grad_indices: Tensor | None, grad_margins: Tensor | None) -> tuple[Any, ...]:
        del grad_indices, grad_margins
        indices, margins, rmins, anchors, payload_data, payload_scales, payload_codebook = ctx.saved_tensors
        batch, steps, in_features = ctx.latent_shape
        item_count = batch * steps
        tables = indices.shape[-1]
        comparisons = margins.shape[-1]
        table_size = ctx.table_size
        out_features = ctx.out_features

        grad_flat = grad_output.reshape(item_count, out_features).contiguous().to(torch.float32)
        indices_flat = indices.reshape(item_count, tables).contiguous()
        indices_kernel = indices_flat.to(torch.int64) if indices_flat.dtype != torch.int64 else indices_flat
        margins_flat = margins.reshape(item_count, tables, comparisons).contiguous()
        rmins_flat = rmins.reshape(item_count, tables).contiguous()
        payload_data = payload_data.contiguous()
        payload_scales = payload_scales.contiguous()
        payload_codebook = payload_codebook.contiguous()
        anchors_contig = anchors.contiguous()
        grad_latent = torch.zeros((item_count, in_features), device=grad_output.device, dtype=torch.float32)
        grad_thresholds = torch.zeros((tables, comparisons), device=grad_output.device, dtype=torch.float32)
        grad_lut = torch.zeros(ctx.lut_shape, device=grad_output.device, dtype=torch.float32)

        block_d = _select_block_size(out_features)
        out_blocks = (out_features + block_d - 1) // block_d
        if _enabled_by_env("TROPNN_TILELANG_TILED_LUT_GRAD") and item_count >= table_size * 2 and table_size <= 128:
            lut_block_d = min(block_d, 128)
            lut_out_blocks = (out_features + lut_block_d - 1) // lut_block_d
            row_block = _lut_grad_row_block(item_count)
            row_blocks = (item_count + row_block - 1) // row_block
            lut_kernel = _pairwise_lut_backward_rowtile_kernel(
                item_count,
                out_features,
                tables,
                table_size,
                lut_block_d,
                lut_out_blocks,
                row_block,
                row_blocks,
                ctx.target,
            )
        else:
            lut_kernel = _pairwise_lut_backward_block_kernel(
                item_count,
                out_features,
                tables,
                table_size,
                block_d,
                out_blocks,
                ctx.target,
            )
        lut_kernel(grad_flat, indices_kernel, grad_lut)

        ste_dot_block_d = _ste_dot_block_size(out_features)
        ste_dot_out_blocks = (out_features + ste_dot_block_d - 1) // ste_dot_block_d

        if ctx.lut_dtype in {"fp32", "bf16", "fp16", "binary01_bf16"}:
            payload_dtype_name = _float_payload_dtype_name(ctx.lut_dtype)
            if ctx.use_min_margin_ste:
                ste_kernel = _pairwise_min_backward_kernel(
                    item_count,
                    in_features,
                    out_features,
                    tables,
                    comparisons,
                    table_size,
                    ste_dot_block_d,
                    ste_dot_out_blocks,
                    payload_dtype_name,
                    ctx.use_izhikevich_surrogate,
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
                    ste_dot_block_d,
                    ste_dot_out_blocks,
                    payload_dtype_name,
                    ctx.use_izhikevich_surrogate,
                    ctx.target,
                )
            if ctx.use_min_margin_ste:
                ste_kernel(grad_flat, indices_kernel, margins_flat, rmins_flat, anchors_contig, payload_data, grad_latent, grad_thresholds)
            else:
                ste_kernel(grad_flat, indices_kernel, margins_flat, anchors_contig, payload_data, grad_latent, grad_thresholds)
        elif ctx.lut_dtype in {"int8", "fp8"}:
            if ctx.use_min_margin_ste:
                ste_kernel = _pairwise_min_backward_int8_kernel(
                    item_count,
                    in_features,
                    out_features,
                    tables,
                    comparisons,
                    table_size,
                    ste_dot_block_d,
                    ste_dot_out_blocks,
                    ctx.use_izhikevich_surrogate,
                    ctx.target,
                )
            else:
                ste_kernel = _pairwise_full_backward_int8_kernel(
                    item_count,
                    in_features,
                    out_features,
                    tables,
                    comparisons,
                    table_size,
                    ste_dot_block_d,
                    ste_dot_out_blocks,
                    ctx.use_izhikevich_surrogate,
                    ctx.target,
                )
            ste_kernel(grad_flat, indices_kernel, margins_flat, anchors_contig, payload_data, payload_scales, grad_latent, grad_thresholds)
        else:
            if ctx.use_min_margin_ste and ctx.lut_dtype in {"binary01_fixed", "binary01_1bit"}:
                binary_table_stride = _positive_env_int("TROPNN_TILELANG_BINARY_STE_TABLE_STRIDE", 1)
                binary_pack_stride = _positive_env_int("TROPNN_TILELANG_BINARY_STE_PACK_STRIDE", 1)
                bits_per_code = 1 if ctx.lut_dtype == "binary01_1bit" else 2
                binary_table_group = min(
                    _positive_env_int(
                        "TROPNN_TILELANG_BINARY_STE_TABLE_GROUP",
                        _default_binary_ste_table_group(bits_per_code, ctx.payload_width, tables),
                    ),
                    tables,
                )
                binary_ste_args = (
                    item_count,
                    in_features,
                    out_features,
                    tables,
                    comparisons,
                    table_size,
                    ctx.payload_width,
                    bits_per_code,
                    _select_block_size(ctx.payload_width),
                )
                if binary_table_stride > 1 or binary_pack_stride > 1:
                    ste_kernel = _pairwise_min_backward_binary01_byte_strided_kernel(
                        *binary_ste_args,
                        binary_table_stride,
                        binary_pack_stride,
                        ctx.use_izhikevich_surrogate,
                        ctx.target,
                    )
                elif binary_table_group > 1:
                    ste_kernel = _pairwise_min_backward_binary01_byte_grouped_kernel(
                        *binary_ste_args,
                        binary_table_group,
                        ctx.use_izhikevich_surrogate,
                        ctx.target,
                    )
                else:
                    ste_kernel = _pairwise_min_backward_binary01_byte_kernel(
                        *binary_ste_args,
                        ctx.use_izhikevich_surrogate,
                        ctx.target,
                    )
                ste_kernel(grad_flat, indices_kernel, margins_flat, rmins_flat, anchors_contig, payload_data, grad_latent, grad_thresholds)
            elif ctx.use_min_margin_ste:
                ste_kernel = _pairwise_min_backward_4bit_kernel(
                    item_count,
                    in_features,
                    out_features,
                    tables,
                    comparisons,
                    table_size,
                    ctx.payload_width,
                    1 if ctx.lut_dtype == "binary01_1bit" else 2 if ctx.lut_dtype in {"int2", "ternary_int2", "ternary_fixed", "binary01_fixed"} else 4,
                    ste_dot_block_d,
                    ste_dot_out_blocks,
                    ctx.use_izhikevich_surrogate,
                    ctx.target,
                )
                ste_kernel(grad_flat, indices_kernel, margins_flat, rmins_flat, anchors_contig, payload_data, payload_scales, payload_codebook, grad_latent, grad_thresholds)
            else:
                ste_kernel = _pairwise_full_backward_4bit_kernel(
                    item_count,
                    in_features,
                    out_features,
                    tables,
                    comparisons,
                    table_size,
                    ctx.payload_width,
                    1 if ctx.lut_dtype == "binary01_1bit" else 2 if ctx.lut_dtype in {"int2", "ternary_int2", "ternary_fixed", "binary01_fixed"} else 4,
                    ste_dot_block_d,
                    ste_dot_out_blocks,
                    ctx.use_izhikevich_surrogate,
                    ctx.target,
                )
                ste_kernel(grad_flat, indices_kernel, margins_flat, anchors_contig, payload_data, payload_scales, payload_codebook, grad_latent, grad_thresholds)

        return grad_latent.view(batch, steps, in_features).to(ctx.latent_input_dtype), None, grad_thresholds, grad_lut.to(dtype=ctx.lut_input_dtype), None, None, None, None, None, None


def pairwise_tilelang(
    latent: Tensor,
    anchors: Tensor,
    thresholds: Tensor,
    lut: Tensor,
    *,
    use_min_margin_ste: bool,
    surrogate: str = "fast_sigmoid_odd",
    target: str = "cuda",
    lut_dtype: PackedLutDType = "bf16",
    packed_payload: _PackedPayload | None = None,
    route_index_dtype: RouteIndexDType = "int64",
) -> tuple[Tensor, Tensor, Tensor]:
    if surrogate not in {"fast_sigmoid_odd", "izhikevich"}:
        raise ValueError(f"Pairwise TileLang backend does not support surrogate={surrogate!r}")
    if lut_dtype not in {"fp32", "bf16", "fp16", "binary01_bf16", "int8", "fp8", "int4", "int2", "ternary_int2", "ternary_fixed", "binary01_fixed", "binary01_1bit", "fp4", "nf4"}:
        raise ValueError(f"Pairwise TileLang backend does not support lut_dtype={lut_dtype!r}")
    if route_index_dtype not in {"int64", "uint8"}:
        raise ValueError(f"Pairwise TileLang backend does not support route_index_dtype={route_index_dtype!r}")
    if route_index_dtype == "uint8" and lut_dtype not in {"fp32", "bf16", "fp16", "binary01_bf16"}:
        raise ValueError("route_index_dtype='uint8' currently supports only fp32/bf16/fp16 payloads")
    return _PairwiseTileLangFunction.apply(latent, anchors, thresholds, lut, use_min_margin_ste, surrogate, target, lut_dtype, packed_payload, route_index_dtype)
