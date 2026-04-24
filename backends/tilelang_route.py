from __future__ import annotations

from functools import lru_cache
from typing import Any

import torch
from torch import Tensor


def has_tilelang() -> bool:
    try:
        import tilelang  # noqa: F401
        import tilelang.language  # noqa: F401
    except ImportError:
        return False
    return True


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.float32:
        return "float32"
    raise TypeError(f"TileLang route backend currently supports float32 tensors only, got {dtype}")


@lru_cache(maxsize=64)
def _route_hidden_kernel(
    item_count: int,
    heads: int,
    cells: int,
    code_dim: int,
    code_scale: float,
    dtype: str,
    target: str,
) -> Any:
    try:
        import tilelang
        import tilelang.language as T
    except ImportError as exc:
        raise RuntimeError("TileLang is not installed; install tilelang or use backend='torch'/'auto'") from exc

    @tilelang.jit(target=target, compile_flags=["-allow-unsupported-compiler", "-ccbin=/usr/bin/g++"])
    def route_hidden_kernel() -> Any:
        @T.prim_func
        def kernel(
            latent: T.Tensor((item_count, code_dim), "float32"),
            router_weight: T.Tensor((heads, cells, code_dim), "float32"),
            router_bias: T.Tensor((heads, cells), "float32"),
            code: T.Tensor((heads, cells, code_dim), "float32"),
            hidden: T.Tensor((item_count, code_dim), "float32"),
            winner_idx: T.Tensor((item_count, heads), "int64"),
            margins: T.Tensor((item_count, heads), "float32"),
        ):
            with T.Kernel(item_count, threads=1) as row:
                for dim in T.serial(code_dim):
                    hidden[row, dim] = latent[row, dim]

                for head in T.serial(heads):
                    best = T.alloc_fragment((1,), "float32")
                    second = T.alloc_fragment((1,), "float32")
                    score = T.alloc_fragment((1,), "float32")
                    best_cell = T.alloc_fragment((1,), "int32")
                    best[0] = -3.4028234663852886e38
                    second[0] = -3.4028234663852886e38
                    best_cell[0] = 0

                    for cell in T.serial(cells):
                        score[0] = 0.0
                        for dim in T.serial(code_dim):
                            score[0] = score[0] + latent[row, dim] * router_weight[head, cell, dim]
                        score[0] = score[0] + router_bias[head, cell]
                        if score[0] > best[0]:
                            second[0] = best[0]
                            best[0] = score[0]
                            best_cell[0] = cell
                        else:
                            if score[0] > second[0]:
                                second[0] = score[0]

                    winner_idx[row, head] = best_cell[0]
                    margins[row, head] = best[0] - second[0]
                    for dim in T.serial(code_dim):
                        hidden[row, dim] = hidden[row, dim] + code[head, best_cell[0], dim] * code_scale

        return kernel

    return route_hidden_kernel()


def trop_route_hidden_tilelang(
    latent: Tensor,
    router_weight: Tensor,
    router_bias: Tensor,
    code: Tensor,
    *,
    code_scale: float,
    target: str = "cuda",
) -> tuple[Tensor, Tensor, Tensor]:
    if not has_tilelang():
        raise RuntimeError("TileLang is not installed; install tilelang or use backend='torch'/'auto'")
    if not latent.is_cuda:
        raise ValueError("TileLang route backend requires CUDA tensors")
    if latent.ndim != 3:
        raise ValueError(f"latent must have shape [batch, steps, code_dim], got {tuple(latent.shape)}")
    if router_weight.ndim != 3:
        raise ValueError(f"router_weight must have shape [heads, cells, code_dim], got {tuple(router_weight.shape)}")
    if router_bias.shape != router_weight.shape[:2]:
        raise ValueError(f"router_bias must have shape [heads, cells], got {tuple(router_bias.shape)}")
    if code.shape != router_weight.shape:
        raise ValueError(f"code must match router_weight shape, got {tuple(code.shape)} and {tuple(router_weight.shape)}")

    batch, steps, code_dim = latent.shape
    heads, cells, weight_dim = router_weight.shape
    if weight_dim != code_dim:
        raise ValueError(f"latent code_dim {code_dim} does not match router_weight code_dim {weight_dim}")

    dtype = _dtype_name(latent.dtype)
    item_count = batch * steps
    latent_flat = latent.reshape(item_count, code_dim).contiguous()
    weight = router_weight.contiguous()
    bias = router_bias.contiguous()
    code_table = code.contiguous()
    hidden = torch.empty((item_count, code_dim), device=latent.device, dtype=latent.dtype)
    winner_idx = torch.empty((item_count, heads), device=latent.device, dtype=torch.int64)
    margins = torch.empty((item_count, heads), device=latent.device, dtype=latent.dtype)
    try:
        kernel = _route_hidden_kernel(item_count, heads, cells, code_dim, float(code_scale), dtype, target)
        kernel(latent_flat, weight, bias, code_table, hidden, winner_idx, margins)
    except Exception as exc:
        raise RuntimeError(
            "TileLang backend failed to compile or launch. Ensure a CUDA toolkit compatible with the GPU is first on PATH "
            "and export CC=/usr/bin/gcc CXX=/usr/bin/g++ before running."
        ) from exc

    return hidden.view(batch, steps, code_dim), winner_idx.view(batch, steps, heads), margins.view(batch, steps, heads)
