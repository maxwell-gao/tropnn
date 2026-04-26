from __future__ import annotations

import ctypes
from typing import Literal

import torch
from torch import Tensor

from .zig_runtime import has_zig_backend, load_zig_library, tensor_ptr


def has_pairwise_zig() -> bool:
    return has_zig_backend()


_ARGS_REGISTERED = False


def _load_pairwise_library() -> ctypes.CDLL:
    global _ARGS_REGISTERED
    lib = load_zig_library()
    if not _ARGS_REGISTERED:
        size = ctypes.c_size_t
        ptr = ctypes.c_void_p
        common_args = [size, size, size, size, size, ptr, ptr, ptr, ptr, ptr]
        lib.lut_forward_batch_with_offsets_no_cache.argtypes = common_args
        lib.lut_forward_batch_with_offsets_no_cache.restype = None
        lib.lut_forward_batch_f16_no_cache.argtypes = common_args
        lib.lut_forward_batch_f16_no_cache.restype = None
        _ARGS_REGISTERED = True
    return lib


def pairwise_zig_forward(
    latent: Tensor,
    anchors: Tensor,
    thresholds: Tensor,
    lut: Tensor,
    *,
    lut_dtype: Literal["f32", "f16"] = "f32",
) -> Tensor:
    if latent.device.type != "cpu":
        raise ValueError("PairwiseLinear backend='zig' requires CPU input tensors")
    if latent.dtype != torch.float32:
        raise TypeError(f"PairwiseLinear backend='zig' requires float32 compute tensors, got {latent.dtype}")
    if anchors.dtype != torch.long:
        raise TypeError(f"PairwiseLinear backend='zig' requires int64 anchors, got {anchors.dtype}")
    if thresholds.dtype != torch.float32:
        thresholds = thresholds.to(torch.float32)
    if lut_dtype not in {"f32", "f16"}:
        raise ValueError(f"lut_dtype must be 'f32' or 'f16', got {lut_dtype!r}")

    batch, steps, input_dim = latent.shape
    tables, comparisons, pair_width = anchors.shape
    if pair_width != 2:
        raise ValueError(f"anchors must have shape [tables, comparisons, 2], got {tuple(anchors.shape)}")
    if thresholds.shape != (tables, comparisons):
        raise ValueError(f"thresholds must have shape {(tables, comparisons)}, got {tuple(thresholds.shape)}")
    if lut.shape[:2] != (tables, 1 << comparisons):
        raise ValueError(f"lut has incompatible shape {tuple(lut.shape)} for tables={tables}, comparisons={comparisons}")

    output_dim = lut.shape[-1]
    item_count = batch * steps
    latent_flat = latent.reshape(item_count, input_dim).contiguous()
    anchors_flat = anchors.contiguous()
    thresholds_flat = thresholds.contiguous()
    output = torch.empty((item_count, output_dim), device="cpu", dtype=torch.float32)
    lib = _load_pairwise_library()

    weights = lut.contiguous()
    if lut_dtype == "f16":
        if weights.dtype != torch.float16:
            weights = weights.to(torch.float16)
        lib.lut_forward_batch_f16_no_cache(
            item_count,
            tables,
            comparisons,
            input_dim,
            output_dim,
            tensor_ptr(weights),
            tensor_ptr(anchors_flat),
            tensor_ptr(thresholds_flat),
            tensor_ptr(latent_flat),
            tensor_ptr(output),
        )
    else:
        if weights.dtype != torch.float32:
            weights = weights.to(torch.float32)
        lib.lut_forward_batch_with_offsets_no_cache(
            item_count,
            tables,
            comparisons,
            input_dim,
            output_dim,
            tensor_ptr(weights),
            tensor_ptr(anchors_flat),
            tensor_ptr(thresholds_flat),
            tensor_ptr(latent_flat),
            tensor_ptr(output),
        )

    return output.view(batch, steps, output_dim)
