from functools import lru_cache
from typing import Any, Literal

import torch
from torch import Tensor

from ._utils import select_block_size as _select_block_size
from .pairwise_payload import PackedLutDType, _PackedPayload, _pack_lut_payload

RouteIndexDType = Literal["int64", "uint8"]


def _float_payload_dtype_name(lut_dtype: str) -> str:
    if lut_dtype == "bf16":
        return "bfloat16"
    if lut_dtype == "fp16":
        return "float16"
    return "float32"


def has_tilelang() -> bool:
    try:
        import tilelang  # noqa: F401
    except ImportError:
        return False
    return True


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
def _pairwise_route_u8_kernel(
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
            indices: T.Tensor((item_count, route_count), "uint8"),
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
                        indices[row, table] = T.cast(idx[0], "uint8")

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
                        pack_col = out_col // 4 if bits == 2 else out_col // 2
                        byte = T.cast(packed[table, idx, pack_col], "int32")
                        code = T.alloc_fragment((1,), "int32")
                        if bits == 2:
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
            anchors: T.Tensor((route_count, comp_count, 2), "int64"),
            lut: T.Tensor((route_count, bucket_count, output_dim), lut_type),
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
                        pack_col = out_col // 4 if bits == 2 else out_col // 2
                        current_byte = T.cast(packed[table, current_idx, pack_col], "int32")
                        neighbor_byte = T.cast(packed[table, neighbor_idx, pack_col], "int32")
                        current_code = T.alloc_fragment((1,), "int32")
                        neighbor_code = T.alloc_fragment((1,), "int32")
                        if bits == 2:
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
                        pack_col = out_col // 4 if bits == 2 else out_col // 2
                        current_byte = T.cast(packed[table, current_idx, pack_col], "int32")
                        neighbor_byte = T.cast(packed[table, neighbor_idx, pack_col], "int32")
                        current_code = T.alloc_fragment((1,), "int32")
                        neighbor_code = T.alloc_fragment((1,), "int32")
                        if bits == 2:
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
    lut_dtype: PackedLutDType = "fp32",
    packed_payload: _PackedPayload | None = None,
    route_index_dtype: RouteIndexDType = "int64",
) -> tuple[Tensor, Tensor, Tensor, _PackedPayload]:
    if not has_tilelang():
        raise RuntimeError("TileLang is not installed; install tilelang or use backend='torch'")
    if not latent.is_cuda:
        raise ValueError("Pairwise TileLang backend requires CUDA tensors")
    if latent.ndim != 3:
        raise ValueError(f"latent must have shape [batch, steps, in_features], got {tuple(latent.shape)}")
    _require_float32(latent, thresholds)

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
    output = torch.empty((item_count, out_features), device=latent.device, dtype=torch.float32)

    route_block = _select_block_size(tables)
    table_blocks = (tables + route_block - 1) // route_block
    block_d = _select_block_size(out_features)
    out_blocks = (out_features + block_d - 1) // block_d
    route_kernel = _pairwise_route_u8_kernel(item_count, in_features, tables, comparisons, route_block, table_blocks, target) if route_index_dtype == "uint8" else _pairwise_route_kernel(item_count, in_features, tables, comparisons, route_block, table_blocks, target)
    route_kernel(latent_flat, anchors_contig, thresholds_contig, indices, margins)
    if route_index_dtype == "uint8":
        if lut_dtype not in {"fp32", "bf16", "fp16"}:
            raise ValueError("route_index_dtype='uint8' currently supports only fp32/bf16/fp16 payloads")
        payload_dtype_name = _float_payload_dtype_name(lut_dtype)
        forward_kernel = _pairwise_forward_u8_block_kernel(item_count, out_features, tables, table_size, block_d, out_blocks, payload_dtype_name, target)
        forward_kernel(indices, payload.data, output)
    elif lut_dtype in {"int8", "fp8"}:
        forward_kernel = _pairwise_forward_int8_block_kernel(item_count, out_features, tables, table_size, block_d, out_blocks, target)
        forward_kernel(indices, payload.data, payload.scales, output)
    elif lut_dtype in {"int4", "int2", "fp4", "nf4"}:
        forward_kernel = _pairwise_forward_4bit_block_kernel(item_count, out_features, tables, table_size, payload.data.shape[-1], 2 if lut_dtype == "int2" else 4, block_d, out_blocks, target)
        forward_kernel(indices, payload.data, payload.scales, payload.codebook, output)
    else:
        payload_dtype_name = _float_payload_dtype_name(lut_dtype)
        forward_kernel = _pairwise_forward_block_kernel(item_count, out_features, tables, table_size, block_d, out_blocks, payload_dtype_name, target)
        forward_kernel(indices, payload.data, output)
    return output.view(batch, steps, out_features), indices.view(batch, steps, tables), margins.view(batch, steps, tables, comparisons), payload


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
        output, indices, margins, payload = _run_forward(latent, anchors, thresholds, lut, target=target, lut_dtype=lut_dtype, packed_payload=packed_payload, route_index_dtype=route_index_dtype)  # type: ignore[arg-type]
        ctx.save_for_backward(indices, margins, anchors, payload.data, payload.scales, payload.codebook)
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
        ctx.lut_input_dtype = lut.dtype
        ctx.mark_non_differentiable(indices, margins)
        return output, indices, margins

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor, grad_indices: Tensor | None, grad_margins: Tensor | None) -> tuple[Any, ...]:
        del grad_indices, grad_margins
        indices, margins, anchors, payload_data, payload_scales, payload_codebook = ctx.saved_tensors
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
        payload_data = payload_data.contiguous()
        payload_scales = payload_scales.contiguous()
        payload_codebook = payload_codebook.contiguous()
        anchors_contig = anchors.contiguous()
        grad_latent = torch.zeros((item_count, in_features), device=grad_output.device, dtype=torch.float32)
        grad_thresholds = torch.zeros((tables, comparisons), device=grad_output.device, dtype=torch.float32)
        grad_lut = torch.zeros(ctx.lut_shape, device=grad_output.device, dtype=torch.float32)

        block_d = _select_block_size(out_features)
        out_blocks = (out_features + block_d - 1) // block_d
        lut_kernel = _pairwise_lut_backward_block_kernel(item_count, out_features, tables, table_size, block_d, out_blocks, ctx.target)
        lut_kernel(grad_flat, indices_kernel, grad_lut)

        if ctx.lut_dtype in {"fp32", "bf16", "fp16"}:
            payload_dtype_name = _float_payload_dtype_name(ctx.lut_dtype)
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
                    block_d,
                    out_blocks,
                    payload_dtype_name,
                    ctx.use_izhikevich_surrogate,
                    ctx.target,
                )
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
                    block_d,
                    out_blocks,
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
                    block_d,
                    out_blocks,
                    ctx.use_izhikevich_surrogate,
                    ctx.target,
                )
            ste_kernel(grad_flat, indices_kernel, margins_flat, anchors_contig, payload_data, payload_scales, grad_latent, grad_thresholds)
        else:
            if ctx.use_min_margin_ste:
                ste_kernel = _pairwise_min_backward_4bit_kernel(
                    item_count,
                    in_features,
                    out_features,
                    tables,
                    comparisons,
                    table_size,
                    ctx.payload_width,
                    2 if ctx.lut_dtype == "int2" else 4,
                    block_d,
                    out_blocks,
                    ctx.use_izhikevich_surrogate,
                    ctx.target,
                )
            else:
                ste_kernel = _pairwise_full_backward_4bit_kernel(
                    item_count,
                    in_features,
                    out_features,
                    tables,
                    comparisons,
                    table_size,
                    ctx.payload_width,
                    2 if ctx.lut_dtype == "int2" else 4,
                    block_d,
                    out_blocks,
                    ctx.use_izhikevich_surrogate,
                    ctx.target,
                )
            ste_kernel(grad_flat, indices_kernel, margins_flat, anchors_contig, payload_data, payload_scales, payload_codebook, grad_latent, grad_thresholds)

        return grad_latent.view(batch, steps, in_features), None, grad_thresholds, grad_lut.to(dtype=ctx.lut_input_dtype), None, None, None, None, None, None


def pairwise_tilelang(
    latent: Tensor,
    anchors: Tensor,
    thresholds: Tensor,
    lut: Tensor,
    *,
    use_min_margin_ste: bool,
    surrogate: str = "fast_sigmoid_odd",
    target: str = "cuda",
    lut_dtype: PackedLutDType = "fp32",
    packed_payload: _PackedPayload | None = None,
    route_index_dtype: RouteIndexDType = "int64",
) -> tuple[Tensor, Tensor, Tensor]:
    if surrogate not in {"fast_sigmoid_odd", "izhikevich"}:
        raise ValueError(f"Pairwise TileLang backend does not support surrogate={surrogate!r}")
    if lut_dtype not in {"fp32", "bf16", "fp16", "int8", "fp8", "int4", "int2", "fp4", "nf4"}:
        raise ValueError(f"Pairwise TileLang backend does not support lut_dtype={lut_dtype!r}")
    if route_index_dtype not in {"int64", "uint8"}:
        raise ValueError(f"Pairwise TileLang backend does not support route_index_dtype={route_index_dtype!r}")
    if route_index_dtype == "uint8" and lut_dtype not in {"fp32", "bf16", "fp16"}:
        raise ValueError("route_index_dtype='uint8' currently supports only fp32/bf16/fp16 payloads")
    return _PairwiseTileLangFunction.apply(latent, anchors, thresholds, lut, use_min_margin_ste, surrogate, target, lut_dtype, packed_payload, route_index_dtype)
