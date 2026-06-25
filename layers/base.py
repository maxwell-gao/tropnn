"""Small PyTorch utilities shared by LUT layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor, nn

__all__ = ["LUTLayerSpec", "LUTModuleBase", "RouteRecord", "finish_lut_output"]


class RouteRecord(Protocol):
    """Route/cache object saved by a LUT layer after it chooses table rows."""

    indices: Tensor
    margins: Tensor


@dataclass(frozen=True)
class LUTLayerSpec:
    input_dim: int
    output_dim: int
    backend: str = "torch"
    output_scale: float = 1.0
    cache_route_debug: bool = False

    @classmethod
    def build(
        cls,
        input_dim: int,
        output_dim: int,
        *,
        backend: str = "torch",
        output_scale: float = 1.0,
        cache_route_debug: bool = False,
    ) -> "LUTLayerSpec":
        return cls(
            input_dim=int(input_dim),
            output_dim=int(output_dim),
            backend=backend,
            output_scale=float(output_scale),
            cache_route_debug=bool(cache_route_debug),
        )


class LUTModuleBase(nn.Module):
    """Thin base class for single-input LUT layers.

    The actual algorithm should stay in the concrete layer.  This class only
    stores common dimensions, checks the last input dimension, and applies the
    final output scale/debug-cache policy.
    """

    def __init__(
        self,
        layer: LUTLayerSpec,
    ) -> None:
        super().__init__()
        self._layer = layer
        self._last_route: RouteRecord | None = None
        self._last_indices: Tensor | None = None
        self._last_margins: Tensor | None = None

    @property
    def input_dim(self) -> int:
        return self._layer.input_dim

    @property
    def output_dim(self) -> int:
        return self._layer.output_dim

    @property
    def backend(self) -> str:
        return self._layer.backend

    @property
    def output_scale(self) -> float:
        return self._layer.output_scale

    @property
    def cache_route_debug(self) -> bool:
        return self._layer.cache_route_debug

    def _check_input_shape(self, x: Tensor) -> None:
        if x.ndim == 0 or x.shape[-1] != self.input_dim:
            raise ValueError(
                f"{type(self).__name__} expected last dimension {self.input_dim}, "
                f"got shape {tuple(x.shape)}"
            )

    def _finish_output(self, output: Tensor, route: RouteRecord | None, dtype: torch.dtype) -> Tensor:
        if self.output_scale != 1.0:
            output = output * self.output_scale
        self._remember_route(route)
        return output.to(dtype=dtype)

    def _remember_route(self, route: RouteRecord | None) -> None:
        if self.cache_route_debug and route is not None:
            self._last_route = route.detach() if hasattr(route, "detach") else route
            self._last_indices = route.indices.detach()
            self._last_margins = route.margins.detach()
        else:
            self._last_route = None
            self._last_indices = None
            self._last_margins = None

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, output_dim={self.output_dim}, "
            f"backend={self.backend!r}"
        )


def finish_lut_output(
    module: nn.Module,
    output: Tensor,
    route: RouteRecord | None,
    dtype: torch.dtype,
    scale: float,
) -> Tensor:
    """Apply the common output scale/dtype step for non-base LUT modules."""

    if scale != 1.0:
        output = output * scale

    if route is not None and hasattr(module, "_last_route"):
        module._last_route = route.detach() if hasattr(route, "detach") else route
        module._last_indices = route.indices.detach()
        module._last_margins = route.margins.detach()

    return output.to(dtype=dtype)
