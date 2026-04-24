from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from ..backend import Backend


class RoutedLinearBase(nn.Module, ABC):
    """Shared shell for discrete-routing linear layers.

    Subclasses own the family-specific routing and payload math. The base class
    only standardizes input shaping, compute dtype selection, output scaling,
    and debug-state caching.
    """

    def __init__(self, in_features: int, out_features: int, *, backend: Backend = "torch", output_scale: float = 1.0) -> None:
        super().__init__()
        if in_features < 1:
            raise ValueError(f"in_features must be >= 1, got {in_features}")
        if out_features < 1:
            raise ValueError(f"out_features must be >= 1, got {out_features}")
        self.in_features = in_features
        self.out_features = out_features
        self.backend = backend
        self.output_scale = output_scale
        self._last_indices: Optional[Tensor] = None
        self._last_margins: Optional[Tensor] = None

    def _compute_dtype(self, x: Tensor) -> torch.dtype:
        return torch.float32 if x.dtype in {torch.float16, torch.bfloat16} else x.dtype

    @abstractmethod
    def _project_input(self, x: Tensor, compute_dtype: torch.dtype) -> Tensor:
        """Map input to the latent representation used by the routing family."""

    @abstractmethod
    def _route_output(
        self,
        latent: Tensor,
        *,
        input_device: torch.device,
        compute_dtype: torch.dtype,
        training: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return output, discrete route indices, and route margins."""

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        input_dtype = x.dtype
        compute_dtype = self._compute_dtype(x)
        latent = self._project_input(x, compute_dtype)
        output, route_indices, margins = self._route_output(
            latent,
            input_device=x.device,
            compute_dtype=compute_dtype,
            training=self.training,
        )
        if self.output_scale != 1.0:
            output = output * self.output_scale
        self._last_indices = route_indices.detach()
        self._last_margins = margins.detach()
        return output.to(dtype=input_dtype)
