from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

Backend = Literal["auto", "torch", "triton", "tilelang", "zig"]


def _requires_grad_path(*tensors: Tensor) -> bool:
    return torch.is_grad_enabled() and any(tensor.requires_grad for tensor in tensors)


def has_triton() -> bool:
    from .backends import has_triton as _has_triton

    return _has_triton()


def has_tilelang() -> bool:
    from .backends import has_tilelang as _has_tilelang

    return _has_tilelang()


def has_pairwise_zig() -> bool:
    from .backends import has_pairwise_zig as _has_pairwise_zig

    return _has_pairwise_zig()


def has_tropical_zig() -> bool:
    from .backends import has_tropical_zig as _has_tropical_zig

    return _has_tropical_zig()


def trop_scores_reference(z: Tensor, router_weight: Tensor, router_bias: Tensor) -> Tensor:
    return torch.einsum("bsr,hkr->bshk", z, router_weight) + router_bias.unsqueeze(0).unsqueeze(0)


def trop_scores(z: Tensor, router_weight: Tensor, router_bias: Tensor, backend: Backend = "torch") -> Tensor:
    if backend in {"tilelang", "zig"}:
        raise RuntimeError(f"backend={backend!r} is a fused TropLinear inference backend, not a standalone score backend")
    if backend == "triton" and _requires_grad_path(z, router_weight, router_bias):
        raise RuntimeError("backend='triton' does not support autograd; use backend='auto' or 'torch' for training")
    if backend == "triton":
        from .backends import trop_scores_triton

        return trop_scores_triton(z, router_weight, router_bias)
    if backend == "auto" and z.is_cuda and has_triton() and not _requires_grad_path(z, router_weight, router_bias):
        from .backends import trop_scores_triton

        return trop_scores_triton(z, router_weight, router_bias)
    return trop_scores_reference(z, router_weight, router_bias)
