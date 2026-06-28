from __future__ import annotations

import ctypes
from typing import Literal

import torch
from torch import Tensor

from .zig_runtime import configure_zig_threads, has_zig_backend, load_zig_library, tensor_ptr


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
        soa_args = [size, size, size, size, size, ptr, ptr, ptr, ptr, ptr, ptr]
        paged_args = [size, size, size, size, size, size, ptr, ptr, ptr, ptr, ptr]
        lib.lut_forward_batch_with_offsets_no_cache.argtypes = common_args
        lib.lut_forward_batch_with_offsets_no_cache.restype = None
        lib.lut_forward_batch_tree_tiled_with_offsets_no_cache.argtypes = common_args
        lib.lut_forward_batch_tree_tiled_with_offsets_no_cache.restype = None
        lib.lut_forward_batch_f16_no_cache.argtypes = common_args
        lib.lut_forward_batch_f16_no_cache.restype = None
        lib.lut_forward_batch_soa_with_offsets_no_cache.argtypes = soa_args
        lib.lut_forward_batch_soa_with_offsets_no_cache.restype = None
        lib.lut_forward_batch_soa_f16_no_cache.argtypes = soa_args
        lib.lut_forward_batch_soa_f16_no_cache.restype = None
        lib.lut_forward_batch_paged_with_offsets_no_cache.argtypes = paged_args
        lib.lut_forward_batch_paged_with_offsets_no_cache.restype = None
        lib.lut_forward_batch_paged_f16_no_cache.argtypes = paged_args
        lib.lut_forward_batch_paged_f16_no_cache.restype = None
        _ARGS_REGISTERED = True
    configure_zig_threads(lib)
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
        raise ValueError("PairwiseLUT backend='zig' requires CPU input tensors")
    if latent.dtype != torch.float32:
        raise TypeError(f"PairwiseLUT backend='zig' requires float32 compute tensors, got {latent.dtype}")
    if anchors.dtype != torch.long:
        raise TypeError(f"PairwiseLUT backend='zig' requires int64 anchors, got {anchors.dtype}")
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


def pairwise_zig_tree_tiled_forward(
    latent: Tensor,
    anchors: Tensor,
    thresholds: Tensor,
    lut: Tensor,
) -> Tensor:
    if latent.device.type != "cpu":
        raise ValueError("PairwiseLUT tree-tiled Zig forward requires CPU input tensors")
    if latent.dtype != torch.float32:
        raise TypeError(f"PairwiseLUT tree-tiled Zig forward requires float32 compute tensors, got {latent.dtype}")
    if anchors.dtype != torch.long:
        raise TypeError(f"PairwiseLUT tree-tiled Zig forward requires int64 anchors, got {anchors.dtype}")
    if thresholds.dtype != torch.float32:
        thresholds = thresholds.to(torch.float32)

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
    weights = lut.contiguous()
    if weights.dtype != torch.float32:
        weights = weights.to(torch.float32)
    lib = _load_pairwise_library()
    lib.lut_forward_batch_tree_tiled_with_offsets_no_cache(
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


def pairwise_zig_soa_forward(
    latent: Tensor,
    anchors: Tensor,
    thresholds: Tensor,
    lut: Tensor,
    *,
    lut_dtype: Literal["f32", "f16"] = "f32",
) -> Tensor:
    if latent.device.type != "cpu":
        raise ValueError("PairwiseLUT SoA Zig forward requires CPU input tensors")
    if latent.dtype != torch.float32:
        raise TypeError(f"PairwiseLUT SoA Zig forward requires float32 compute tensors, got {latent.dtype}")
    if anchors.dtype != torch.long:
        raise TypeError(f"PairwiseLUT SoA Zig forward requires int64 anchors, got {anchors.dtype}")
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
    anchor_a = anchors[..., 0].contiguous()
    anchor_b = anchors[..., 1].contiguous()
    thresholds_flat = thresholds.contiguous()
    output = torch.empty((item_count, output_dim), device="cpu", dtype=torch.float32)
    lib = _load_pairwise_library()

    weights = lut.contiguous()
    if lut_dtype == "f16":
        if weights.dtype != torch.float16:
            weights = weights.to(torch.float16)
        lib.lut_forward_batch_soa_f16_no_cache(
            item_count,
            tables,
            comparisons,
            input_dim,
            output_dim,
            tensor_ptr(weights),
            tensor_ptr(anchor_a),
            tensor_ptr(anchor_b),
            tensor_ptr(thresholds_flat),
            tensor_ptr(latent_flat),
            tensor_ptr(output),
        )
    else:
        if weights.dtype != torch.float32:
            weights = weights.to(torch.float32)
        lib.lut_forward_batch_soa_with_offsets_no_cache(
            item_count,
            tables,
            comparisons,
            input_dim,
            output_dim,
            tensor_ptr(weights),
            tensor_ptr(anchor_a),
            tensor_ptr(anchor_b),
            tensor_ptr(thresholds_flat),
            tensor_ptr(latent_flat),
            tensor_ptr(output),
        )

    return output.view(batch, steps, output_dim)


def pairwise_zig_paged_forward(
    latent: Tensor,
    anchors: Tensor,
    thresholds: Tensor,
    lut: Tensor,
    *,
    lut_dtype: Literal["f32", "f16"] = "f32",
    page_size: int = 1024,
) -> Tensor:
    if page_size < 1:
        raise ValueError(f"page_size must be positive, got {page_size}")
    if latent.device.type != "cpu":
        raise ValueError("PairwiseLUT paged Zig forward requires CPU input tensors")
    if latent.dtype != torch.float32:
        raise TypeError(f"PairwiseLUT paged Zig forward requires float32 compute tensors, got {latent.dtype}")
    if anchors.dtype != torch.long:
        raise TypeError(f"PairwiseLUT paged Zig forward requires int64 anchors, got {anchors.dtype}")
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
        lib.lut_forward_batch_paged_f16_no_cache(
            item_count,
            tables,
            comparisons,
            input_dim,
            output_dim,
            int(page_size),
            tensor_ptr(weights),
            tensor_ptr(anchors_flat),
            tensor_ptr(thresholds_flat),
            tensor_ptr(latent_flat),
            tensor_ptr(output),
        )
    else:
        if weights.dtype != torch.float32:
            weights = weights.to(torch.float32)
        lib.lut_forward_batch_paged_with_offsets_no_cache(
            item_count,
            tables,
            comparisons,
            input_dim,
            output_dim,
            int(page_size),
            tensor_ptr(weights),
            tensor_ptr(anchors_flat),
            tensor_ptr(thresholds_flat),
            tensor_ptr(latent_flat),
            tensor_ptr(output),
        )

    return output.view(batch, steps, output_dim)
