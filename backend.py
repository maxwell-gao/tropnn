from __future__ import annotations

from typing import Literal

Backend = Literal["auto", "torch", "tilelang", "triton", "zig"]


def has_tilelang() -> bool:
    try:
        import tilelang  # noqa: F401
    except ImportError:
        return False
    return True


def has_pairwise_zig() -> bool:
    from .backends import has_pairwise_zig as _has_pairwise_zig

    return _has_pairwise_zig()


def has_triton() -> bool:
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


__all__ = ["Backend", "has_pairwise_zig", "has_tilelang", "has_triton"]
