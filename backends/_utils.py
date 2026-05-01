from __future__ import annotations

import torch

DEFAULT_SCORE_ROUTE_BYTES = 128 * 1024 * 1024


def next_power_of_2(value: int) -> int:
    if value < 1:
        return 1
    return 1 << (value - 1).bit_length()


def select_block_size(value: int, *, min_block: int = 32, max_block: int = 256) -> int:
    return max(min_block, min(max_block, next_power_of_2(value)))


def float32_tilelang_dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.float32:
        return "float32"
    raise TypeError(f"TileLang route backend currently supports float32 tensors only, got {dtype}")


def can_materialize_scores(item_count: int, heads: int, cells: int, *, max_bytes: int | None = None) -> bool:
    limit = DEFAULT_SCORE_ROUTE_BYTES if max_bytes is None else max_bytes
    return item_count * heads * cells * 4 <= limit
