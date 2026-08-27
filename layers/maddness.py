"""MADDNESS adapters built on the shared :mod:`hard_lookup` core.

The adaptive-unary route, hard lookup, and local counterfactual STE are not
implemented here; they are configured instances of :class:`HardLookupRouter`.
Only the MADDNESS-specific soft-PQ backward path is added by this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from .hard_lookup import HardLookupRouter, hard_forward_soft_backward

__all__ = [
    "CompiledMaddness",
    "FrozenMaddness",
    "LocalCounterfactualMaddness",
    "SoftPQMaddness",
]


@dataclass(frozen=True)
class CompiledMaddness:
    split_indices: Tensor
    thresholds: Tensor
    encoder_centroids: Tensor
    prototypes: Tensor


class _MaddnessLookup(HardLookupRouter):
    def __init__(
        self,
        compiled: CompiledMaddness,
        *,
        surrogate: str,
        tau: float = 1.0,
        trainable_thresholds: bool,
        trainable_rows: bool,
    ) -> None:
        depth = int(compiled.split_indices.shape[1])
        input_dim = int(compiled.encoder_centroids.shape[0] * compiled.encoder_centroids.shape[-1])
        output_dim = int(compiled.prototypes.shape[-1])
        super().__init__(
            input_dim,
            output_dim,
            depth=depth,
            predicate="unary",
            topology="adaptive",
            support_layout="level",
            supports=compiled.split_indices,
            thresholds=compiled.thresholds,
            rows=compiled.prototypes,
            surrogate=surrogate,  # type: ignore[arg-type]
            tau=tau,
            trainable_thresholds=trainable_thresholds,
            trainable_rows=trainable_rows,
        )

    @property
    def split_indices(self) -> Tensor:
        return self.supports[..., 0]

    @property
    def prototypes(self) -> Tensor:
        return self.rows


class FrozenMaddness(_MaddnessLookup):
    def __init__(self, compiled: CompiledMaddness) -> None:
        super().__init__(compiled, surrogate="none", trainable_thresholds=False, trainable_rows=False)


class LocalCounterfactualMaddness(_MaddnessLookup):
    def __init__(self, compiled: CompiledMaddness, tau: float) -> None:
        super().__init__(
            compiled,
            surrogate="local_counterfactual",
            tau=tau,
            trainable_thresholds=True,
            trainable_rows=True,
        )


class SoftPQMaddness(_MaddnessLookup):
    """MADDNESS hard inference with the LUT-NN soft-PQ backward path."""

    def __init__(self, compiled: CompiledMaddness, initial_temperature: float = 0.03) -> None:
        if initial_temperature <= 0:
            raise ValueError("initial_temperature must be positive")
        super().__init__(compiled, surrogate="none", trainable_thresholds=False, trainable_rows=True)
        self.encoder_centroids = torch.nn.Parameter(compiled.encoder_centroids.clone())
        self.log_temperature = torch.nn.Parameter(torch.full((self.tables,), math.log(initial_temperature)))

    def outputs(self, x: Tensor) -> tuple[Tensor, Tensor]:
        hard = self.hard_output(x)[0]
        width = x.shape[-1] // self.tables
        if width * self.tables != x.shape[-1]:
            raise ValueError("soft-PQ input dimension must be divisible by table count")
        x_local = x.reshape(*x.shape[:-1], self.tables, width)
        distances = (x_local.unsqueeze(-2) - self.encoder_centroids).square().sum(-1)
        temperature = self.log_temperature.exp().clamp(0.03, 30.0)
        temperature_shape = (1,) * (distances.ndim - 2) + (self.tables, 1)
        probabilities = torch.softmax(-distances / temperature.view(temperature_shape), dim=-1)
        soft = torch.einsum("...tk,tko->...o", probabilities, self.rows)
        return hard, soft

    def forward(self, x: Tensor) -> Tensor:
        hard, soft = self.outputs(x)
        return hard_forward_soft_backward(hard, soft)
