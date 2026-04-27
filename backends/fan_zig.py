from __future__ import annotations

from typing import Literal

from torch import Tensor

from .tropical_zig import has_tropical_zig, trop_route_hidden_zig


def has_trop_fan_zig() -> bool:
    return has_tropical_zig()


def trop_fan_route_hidden_zig(
    latent: Tensor,
    sites: Tensor,
    lifting: Tensor,
    values: Tensor,
    *,
    code_scale: float,
    param_dtype: Literal["f32", "f16"] = "f32",
) -> Tensor:
    return trop_route_hidden_zig(
        latent,
        sites,
        lifting,
        values,
        code_scale=code_scale,
        param_dtype=param_dtype,
    )
