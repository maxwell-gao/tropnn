from __future__ import annotations

import ctypes
from typing import Literal

import torch
from torch import Tensor

from .zig_runtime import has_zig_backend, load_zig_library, tensor_ptr


def has_tropical_zig() -> bool:
    return has_zig_backend()


_ARGS_REGISTERED = False


def _load_tropical_library() -> ctypes.CDLL:
    global _ARGS_REGISTERED
    lib = load_zig_library()
    if not _ARGS_REGISTERED:
        size = ctypes.c_size_t
        ptr = ctypes.c_void_p
        common_args = [size, size, size, size, ctypes.c_float, ptr, ptr, ptr, ptr, ptr]
        lib.trop_route_hidden_batch_f32.argtypes = common_args
        lib.trop_route_hidden_batch_f32.restype = None
        lib.trop_route_hidden_batch_f16.argtypes = common_args
        lib.trop_route_hidden_batch_f16.restype = None
        _ARGS_REGISTERED = True
    return lib


def trop_route_hidden_zig(
    latent: Tensor,
    router_weight: Tensor,
    router_bias: Tensor,
    code: Tensor,
    *,
    code_scale: float,
    param_dtype: Literal["f32", "f16"] = "f32",
) -> Tensor:
    if latent.device.type != "cpu":
        raise ValueError("TropLinear backend='zig' requires CPU input tensors")
    if latent.dtype != torch.float32:
        raise TypeError(f"TropLinear backend='zig' requires float32 compute tensors, got {latent.dtype}")
    if param_dtype not in {"f32", "f16"}:
        raise ValueError(f"param_dtype must be 'f32' or 'f16', got {param_dtype!r}")

    batch, steps, code_dim = latent.shape
    heads, cells, weight_code_dim = router_weight.shape
    if weight_code_dim != code_dim:
        raise ValueError(f"router_weight code dim {weight_code_dim} does not match latent code dim {code_dim}")
    if router_bias.shape != (heads, cells):
        raise ValueError(f"router_bias must have shape {(heads, cells)}, got {tuple(router_bias.shape)}")
    if code.shape != (heads, cells, code_dim):
        raise ValueError(f"code must have shape {(heads, cells, code_dim)}, got {tuple(code.shape)}")

    item_count = batch * steps
    latent_flat = latent.reshape(item_count, code_dim).contiguous()
    hidden = torch.empty((item_count, code_dim), device="cpu", dtype=torch.float32)
    lib = _load_tropical_library()

    weight = router_weight.contiguous()
    bias = router_bias.contiguous()
    code_values = code.contiguous()
    if param_dtype == "f16":
        if weight.dtype != torch.float16:
            weight = weight.to(torch.float16)
        if bias.dtype != torch.float16:
            bias = bias.to(torch.float16)
        if code_values.dtype != torch.float16:
            code_values = code_values.to(torch.float16)
        lib.trop_route_hidden_batch_f16(
            item_count,
            heads,
            cells,
            code_dim,
            float(code_scale),
            tensor_ptr(latent_flat),
            tensor_ptr(weight),
            tensor_ptr(bias),
            tensor_ptr(code_values),
            tensor_ptr(hidden),
        )
    else:
        if weight.dtype != torch.float32:
            weight = weight.to(torch.float32)
        if bias.dtype != torch.float32:
            bias = bias.to(torch.float32)
        if code_values.dtype != torch.float32:
            code_values = code_values.to(torch.float32)
        lib.trop_route_hidden_batch_f32(
            item_count,
            heads,
            cells,
            code_dim,
            float(code_scale),
            tensor_ptr(latent_flat),
            tensor_ptr(weight),
            tensor_ptr(bias),
            tensor_ptr(code_values),
            tensor_ptr(hidden),
        )

    return hidden.view(batch, steps, code_dim)
