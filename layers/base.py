from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor

from ..backend import Backend


class LUTModuleBase(nn.Module, ABC):
    """Shared shell for lookup-table modules.

    The base class only handles boring module mechanics: optional single-token
    sequence wrapping, compute dtype selection, output scaling, and debug route
    caching. Subclasses own the actual algorithm.
    """

    def __init__(self, input_dim: int, output_dim: int, *, backend: Backend = "torch", output_scale: float = 1.0) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.backend = backend
        self.output_scale = float(output_scale)
        self.cache_route_debug = True
        self._last_indices: Tensor | None = None
        self._last_margins: Tensor | None = None

    @staticmethod
    def compute_dtype(x: Tensor) -> torch.dtype:
        return torch.float32 if x.dtype in {torch.float16, torch.bfloat16} else x.dtype

    @abstractmethod
    def compute(self, x: Tensor, *, compute_dtype: torch.dtype, training: bool) -> tuple[Tensor, Tensor, Tensor]:
        """Return output, route indices, and route margins."""

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        input_dtype = x.dtype
        output, indices, margins = self.compute(x, compute_dtype=self.compute_dtype(x), training=self.training)
        if self.output_scale != 1.0:
            output = output * self.output_scale
        if self.cache_route_debug:
            self._last_indices = indices.detach()
            self._last_margins = margins.detach()
        else:
            self._last_indices = None
            self._last_margins = None
        return output.to(dtype=input_dtype)
