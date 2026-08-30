import math
from functools import lru_cache
from typing import Any

import torch
from torch import Tensor


def _latent_dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.float32:
        return "float32"
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.bfloat16:
        return "bfloat16"
    raise TypeError(f"sum-pyramid route expects fp32/fp16/bf16 input, got {dtype}")


def has_sum_pyramid_tilelang() -> bool:
    try:
        import tilelang  # noqa: F401
    except ImportError:
        return False
    return True


@lru_cache(maxsize=32)
def _sum_pyramid_route_kernel(
    item_count: int,
    n_features: int,
    tables: int,
    comparisons: int,
    threads: int,
    latent_dtype_name: str,
    target: str,
) -> Any:
    try:
        import tilelang
        import tilelang.language as T
    except ImportError as exc:
        raise RuntimeError("TileLang is not installed") from exc

    depth = int(math.log2(n_features))
    pyramid_width = 2 * n_features - 1

    @tilelang.jit(target=target, compile_flags=["-allow-unsupported-compiler", "-ccbin=/usr/bin/g++"])
    def route_kernel() -> Any:
        input_dim = n_features
        route_count = tables
        comp_count = comparisons
        block_width = threads
        input_dtype = latent_dtype_name
        route_tiles = (tables + threads - 1) // threads

        @T.prim_func
        def kernel(
            latent: T.Tensor((item_count, input_dim), input_dtype),
            signs: T.Tensor((input_dim,), "int8"),
            anchors: T.Tensor((route_count, comp_count, 2), "int64"),
            thresholds: T.Tensor((route_count, comp_count), "float32"),
            indices: T.Tensor((item_count, route_count), "int64"),
            margins: T.Tensor((item_count, route_count, comp_count), "float32"),
            rmins: T.Tensor((item_count, route_count), "uint8"),
        ):
            with T.Kernel(item_count, threads=block_width) as row:
                tx = T.get_thread_bindings()[0]
                pyramid = T.alloc_shared((pyramid_width,), "float32")
                for tile in T.serial((input_dim + block_width - 1) // block_width):
                    leaf = tile * block_width + tx
                    if leaf < input_dim:
                        pyramid[leaf] = T.cast(latent[row, leaf], "float32") * T.cast(signs[leaf], "float32")
                T.sync_threads()

                # Compute the leaves-first offsets from the level rather than
                # indexing a host list.  The closed forms are
                # parent_offset(l)=2N-(N>>l) and parent_count(l)=N>>(l+1).
                child_offset = T.alloc_fragment((1,), "int32")
                parent_offset = T.alloc_fragment((1,), "int32")
                parent_count = T.alloc_fragment((1,), "int32")
                for level in T.serial(depth):
                    child_offset[0] = 0
                    if level > 0:
                        child_offset[0] = 2 * input_dim - T.shift_right(input_dim, level - 1)
                    parent_offset[0] = 2 * input_dim - T.shift_right(input_dim, level)
                    parent_count[0] = T.shift_right(input_dim, level + 1)
                    parent = tx
                    if parent < parent_count[0]:
                        pyramid[parent_offset[0] + parent] = pyramid[child_offset[0] + 2 * parent] + pyramid[child_offset[0] + 2 * parent + 1]
                    if input_dim > 2 * block_width:
                        if level == 0:
                            pyramid[parent_offset[0] + block_width + tx] = (
                                pyramid[child_offset[0] + 2 * (block_width + tx)] + pyramid[child_offset[0] + 2 * (block_width + tx) + 1]
                            )
                    T.sync_threads()

                idx = T.alloc_fragment((1,), "int32")
                power = T.alloc_fragment((1,), "int32")
                best_r = T.alloc_fragment((1,), "int32")
                best_abs = T.alloc_fragment((1,), "float32")
                for table_tile in T.serial(route_tiles):
                    table = table_tile * block_width + tx
                    if table < route_count:
                        idx[0] = 0
                        power[0] = 1
                        best_r[0] = 0
                        best_abs[0] = 1.0e30
                        for comp in T.serial(comp_count):
                            left = anchors[table, comp, 0]
                            right = anchors[table, comp, 1]
                            margin = pyramid[left] - pyramid[right] - thresholds[table, comp]
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


def _validate_route_inputs(latent: Tensor, signs: Tensor, anchors: Tensor, thresholds: Tensor) -> tuple[int, int, int, int]:
    if latent.ndim != 2:
        raise ValueError(f"expected latent [items,N], got {tuple(latent.shape)}")
    item_count, n_features = latent.shape
    if n_features < 2 or n_features & (n_features - 1):
        raise ValueError("sum-pyramid route requires a power-of-two input width >=2")
    if not latent.is_cuda:
        raise ValueError("sum-pyramid TileLang route requires a CUDA tensor")
    if signs.shape != (n_features,) or signs.dtype != torch.int8:
        raise ValueError(f"expected int8 signs [{n_features}], got {tuple(signs.shape)} {signs.dtype}")
    if anchors.ndim != 3 or anchors.shape[-1] != 2:
        raise ValueError("expected anchors [T,C,2]")
    tables, comparisons, _ = anchors.shape
    if thresholds.shape != (tables, comparisons):
        raise ValueError("threshold shape does not match anchors")
    if comparisons < 1 or comparisons > 16:
        raise ValueError("comparisons must lie in [1,16]")
    if n_features > 1024:
        raise ValueError("sum-pyramid TileLang route currently supports input widths through 1024")
    if anchors.numel() and (int(anchors.min()) < 0 or int(anchors.max()) >= 2 * n_features - 1):
        raise ValueError("anchor index lies outside the sum pyramid")
    return item_count, n_features, tables, comparisons


class _SumPyramidRouteFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        latent: Tensor,
        signs: Tensor,
        anchors: Tensor,
        thresholds: Tensor,
        threads: int,
        target: str,
    ) -> tuple[Tensor, Tensor, Tensor]:
        item_count, n_features, tables, comparisons = _validate_route_inputs(latent, signs, anchors, thresholds)
        latent = latent.contiguous()
        signs = signs.to(device=latent.device, dtype=torch.int8).contiguous()
        anchors = anchors.to(device=latent.device, dtype=torch.int64).contiguous()
        thresholds = thresholds.to(device=latent.device, dtype=torch.float32).contiguous()
        indices = torch.empty(item_count, tables, device=latent.device, dtype=torch.int64)
        margins = torch.empty(item_count, tables, comparisons, device=latent.device, dtype=torch.float32)
        rmins = torch.empty(item_count, tables, device=latent.device, dtype=torch.uint8)
        kernel = _sum_pyramid_route_kernel(
            item_count,
            n_features,
            tables,
            comparisons,
            threads,
            _latent_dtype_name(latent.dtype),
            target,
        )
        kernel(latent, signs, anchors, thresholds, indices, margins, rmins)
        ctx.save_for_backward(signs, anchors)
        ctx.n_features = n_features
        ctx.latent_dtype = latent.dtype
        return indices, margins, rmins

    @staticmethod
    def backward(
        ctx: Any,
        grad_indices: Tensor | None,
        grad_margins: Tensor | None,
        grad_rmins: Tensor | None,
    ) -> tuple[object, ...]:
        del grad_indices, grad_rmins
        signs, anchors = ctx.saved_tensors
        if grad_margins is None:
            return None, None, None, None, None, None
        grad_margins = grad_margins.to(torch.float32)
        batch = grad_margins.shape[0]
        n_features = int(ctx.n_features)
        pyramid_width = 2 * n_features - 1
        flat_anchors = anchors.reshape(-1, 2)
        flat_grad = grad_margins.reshape(batch, -1)
        grad_pyramid = torch.zeros(batch, pyramid_width, device=grad_margins.device, dtype=torch.float32)
        grad_pyramid.scatter_add_(1, flat_anchors[:, 0].expand(batch, -1), flat_grad)
        grad_pyramid.scatter_add_(1, flat_anchors[:, 1].expand(batch, -1), -flat_grad)

        depth = int(math.log2(n_features))
        sizes = [n_features >> level for level in range(depth + 1)]
        offsets: list[int] = []
        offset = 0
        for size in sizes:
            offsets.append(offset)
            offset += size
        for level in range(depth, 0, -1):
            parent = grad_pyramid[:, offsets[level] : offsets[level] + sizes[level]]
            child = grad_pyramid[:, offsets[level - 1] : offsets[level - 1] + sizes[level - 1]]
            child.add_(parent.repeat_interleave(2, dim=1))
        grad_latent = (grad_pyramid[:, :n_features] * signs.to(torch.float32)).to(ctx.latent_dtype)
        grad_thresholds = -grad_margins.sum(dim=0)
        return grad_latent, None, None, grad_thresholds, None, None


def sum_pyramid_pairwise_route_tilelang(
    latent: Tensor,
    signs: Tensor,
    anchors: Tensor,
    thresholds: Tensor,
    *,
    threads: int | None = None,
    target: str = "cuda",
) -> tuple[Tensor, Tensor]:
    """Fused shared-memory SumPyramid construction and canonical pair route.

    The forward path writes route indices, margins, and nearest-bit metadata
    to HBM.  Its
    differentiable margin path uses the exact transpose pyramid in backward,
    so existing hard lookup and local-counterfactual code can consume the
    returned tensors without redefining the PC-LUT surrogate.
    """

    indices, margins, _ = sum_pyramid_pairwise_route_tilelang_full(
        latent,
        signs,
        anchors,
        thresholds,
        threads=threads,
        target=target,
    )
    return indices, margins


def sum_pyramid_pairwise_route_tilelang_full(
    latent: Tensor,
    signs: Tensor,
    anchors: Tensor,
    thresholds: Tensor,
    *,
    threads: int | None = None,
    target: str = "cuda",
) -> tuple[Tensor, Tensor, Tensor]:
    """Return packed indices, all margins, and the nearest-boundary bit."""

    if threads is None:
        needs_backward = torch.is_grad_enabled() and (latent.requires_grad or thresholds.requires_grad)
        divisor = 4 if needs_backward else 2
        threads = min(512, max(64, latent.shape[1] // divisor))
    if threads not in {64, 128, 256, 512}:
        raise ValueError("threads must be one of 64, 128, 256, or 512")
    if latent.shape[1] > 4 * threads:
        raise ValueError("threads must cover at least one quarter of the input width")
    return _SumPyramidRouteFunction.apply(latent, signs, anchors, thresholds, threads, target)


def sum_pyramid_pairwise_route_torch(
    latent: Tensor,
    signs: Tensor,
    anchors: Tensor,
    thresholds: Tensor,
) -> tuple[Tensor, Tensor]:
    """Reference route with the same leaves-first pyramid and bit order."""

    if latent.ndim != 2:
        raise ValueError("expected latent [items,N]")
    current = latent * signs.to(device=latent.device, dtype=latent.dtype)
    levels = [current]
    while current.shape[1] > 1:
        current = current.reshape(current.shape[0], current.shape[1] // 2, 2).sum(dim=-1)
        levels.append(current)
    pyramid = torch.cat(levels, dim=1)
    anchors = anchors.to(device=latent.device, dtype=torch.int64)
    thresholds = thresholds.to(device=latent.device, dtype=torch.float32)
    margins = (pyramid[:, anchors[..., 0]] - pyramid[:, anchors[..., 1]] - thresholds.unsqueeze(0)).to(torch.float32)
    powers = 2 ** torch.arange(anchors.shape[1], device=latent.device, dtype=torch.int64)
    indices = ((margins > 0).to(torch.int64) * powers.view(1, 1, -1)).sum(dim=-1)
    return indices, margins
