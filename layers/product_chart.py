from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

ProductChartAction = Literal["constant", "shared_linear", "local_linear"]
ProductChartSurrogate = Literal["none", "soft_pq"]

__all__ = [
    "ProductChartAction",
    "ProductChartCoordinates",
    "ProductChartField",
    "ProductChartSurrogate",
]


@dataclass(frozen=True)
class ProductChartCoordinates:
    """Explicit hybrid coordinate emitted by a hard product partition.

    ``codes`` are the finite recognition output.  ``residuals`` are a separate
    analog channel and must be counted as recognition bandwidth whenever this
    object, rather than ``codes`` alone, is claimed to be reconstructive.
    """

    codes: Tensor
    residuals: Tensor


class ProductChartField(nn.Module):
    """Hard product codes followed by a shared-rank local action field.

    The deployed route is exact nearest-centroid product quantization.  With
    table ``t`` and selected code ``q_t``, the active latent action is

    ``e[t,q_t] + B[t,q_t]^T (x_t - c[t,q_t])``.

    Active table latents are summed and projected by one shared output basis.
    Setting ``action='constant'`` removes ``B`` and is strictly nested in the
    local-linear model at zero local maps.  ``soft_pq`` leaves the numerical
    forward exactly hard and adds the soft-mixture gradient to the exact hard
    action gradient.  It is therefore an additive training surrogate, not a
    claim of sparse or route-only backward execution.
    """

    def __init__(
        self,
        centroids: Tensor,
        output_dim: int,
        rank: int,
        *,
        action: ProductChartAction = "local_linear",
        surrogate: ProductChartSurrogate = "soft_pq",
        temperature: float = 1.0,
        trainable_centroids: bool = True,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if centroids.ndim != 3:
            raise ValueError("centroids must have shape [tables,codes,block_width]")
        if not centroids.is_floating_point():
            raise TypeError("centroids must be floating point")
        if output_dim < 1 or rank < 1:
            raise ValueError("output_dim and rank must be positive")
        if action not in {"constant", "shared_linear", "local_linear"}:
            raise ValueError(f"unsupported action {action!r}")
        if surrogate not in {"none", "soft_pq"}:
            raise ValueError(f"unsupported surrogate {surrogate!r}")
        if temperature <= 0 or not math.isfinite(temperature):
            raise ValueError("temperature must be finite and positive")

        self.tables = int(centroids.shape[0])
        self.codes = int(centroids.shape[1])
        self.block_width = int(centroids.shape[2])
        if min(self.tables, self.codes, self.block_width) < 1:
            raise ValueError("centroid geometry must be nonempty")
        self.input_dim = self.tables * self.block_width
        self.output_dim = int(output_dim)
        self.rank = int(rank)
        self.action = action
        self.surrogate = surrogate
        self.temperature = float(temperature)

        centroid_value = centroids.detach().clone()
        if trainable_centroids:
            self.centroids = nn.Parameter(centroid_value)
        else:
            self.register_buffer("centroids", centroid_value)

        generator = torch.Generator(device="cpu").manual_seed(seed)
        offset_scale = 1.0 / math.sqrt(self.tables * self.rank)
        basis_scale = 1.0 / math.sqrt(self.rank)
        self.offsets = nn.Parameter(torch.randn(self.tables, self.codes, self.rank, generator=generator, dtype=centroids.dtype) * offset_scale)
        self.output_basis = nn.Parameter(torch.randn(self.rank, self.output_dim, generator=generator, dtype=centroids.dtype) * basis_scale)
        map_codes = self.codes if self.action == "local_linear" else 1
        local_maps = torch.zeros(self.tables, map_codes, self.block_width, self.rank, dtype=centroids.dtype)
        if self.action in {"shared_linear", "local_linear"}:
            self.local_maps = nn.Parameter(local_maps)
        else:
            self.register_buffer("local_maps", local_maps)

    @property
    def trainable_centroids(self) -> bool:
        return isinstance(self.centroids, nn.Parameter)

    def _local_input(self, x: Tensor) -> tuple[Tensor, torch.Size]:
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"expected final input dimension {self.input_dim}, got {x.shape[-1]}")
        leading = x.shape[:-1]
        return x.reshape(-1, self.tables, self.block_width), leading

    def _distances(self, local: Tensor) -> Tensor:
        centroids = self.centroids.to(device=local.device, dtype=local.dtype)
        return (local.unsqueeze(-2) - centroids.unsqueeze(0)).square().sum(dim=-1)

    def hard_codes(self, x: Tensor) -> Tensor:
        local, leading = self._local_input(x)
        codes = self._distances(local).argmin(dim=-1)
        return codes.reshape(*leading, self.tables)

    def _selected_centroids(self, flat_codes: Tensor, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        table = torch.arange(self.tables, device=device).unsqueeze(0).expand(flat_codes.shape[0], -1)
        centroids = self.centroids.to(device=device, dtype=dtype)
        return centroids[table, flat_codes]

    def chart_coordinates(self, x: Tensor) -> ProductChartCoordinates:
        local, leading = self._local_input(x)
        flat_codes = self._distances(local).argmin(dim=-1)
        selected = self._selected_centroids(flat_codes, device=local.device, dtype=local.dtype)
        residuals = local - selected
        return ProductChartCoordinates(
            codes=flat_codes.reshape(*leading, self.tables),
            residuals=residuals.reshape(*leading, self.tables, self.block_width),
        )

    def reconstruct(self, coordinates: ProductChartCoordinates) -> Tensor:
        if coordinates.codes.shape[-1] != self.tables:
            raise ValueError("coordinate code geometry does not match this product chart")
        if coordinates.residuals.shape[-2:] != (self.tables, self.block_width):
            raise ValueError("coordinate residual geometry does not match this product chart")
        leading = coordinates.codes.shape[:-1]
        if coordinates.residuals.shape[:-2] != leading:
            raise ValueError("coordinate leading dimensions must agree")
        flat_codes = coordinates.codes.reshape(-1, self.tables)
        residuals = coordinates.residuals.reshape(-1, self.tables, self.block_width)
        selected = self._selected_centroids(flat_codes, device=residuals.device, dtype=residuals.dtype)
        return (selected + residuals).reshape(*leading, self.input_dim)

    def _hard_latent_from_local(self, local: Tensor, flat_codes: Tensor) -> Tensor:
        table = torch.arange(self.tables, device=local.device).unsqueeze(0).expand(flat_codes.shape[0], -1)
        offsets = self.offsets.to(device=local.device, dtype=local.dtype)[table, flat_codes]
        if self.action == "constant":
            return offsets.sum(dim=1)
        selected_centroids = self._selected_centroids(flat_codes, device=local.device, dtype=local.dtype)
        all_maps = self.local_maps.to(device=local.device, dtype=local.dtype)
        maps = all_maps[:, 0].unsqueeze(0).expand(local.shape[0], -1, -1, -1) if self.action == "shared_linear" else all_maps[table, flat_codes]
        residuals = local - selected_centroids
        return (offsets + torch.einsum("nts,ntsr->ntr", residuals, maps)).sum(dim=1)

    def hard_output(self, x: Tensor) -> tuple[Tensor, Tensor]:
        local, leading = self._local_input(x)
        flat_codes = self._distances(local).argmin(dim=-1)
        latent = self._hard_latent_from_local(local, flat_codes)
        output = latent @ self.output_basis.to(device=x.device, dtype=x.dtype)
        return output.reshape(*leading, self.output_dim), flat_codes.reshape(*leading, self.tables)

    def soft_output(self, x: Tensor) -> Tensor:
        local, leading = self._local_input(x)
        distances = self._distances(local)
        probabilities = torch.softmax(-distances / self.temperature, dim=-1)
        offsets = self.offsets.to(device=x.device, dtype=x.dtype).unsqueeze(0)
        if self.action == "constant":
            all_latents = offsets.expand(local.shape[0], -1, -1, -1)
        else:
            centroids = self.centroids.to(device=x.device, dtype=x.dtype)
            residuals = local.unsqueeze(-2) - centroids.unsqueeze(0)
            maps = self.local_maps.to(device=x.device, dtype=x.dtype)
            if self.action == "shared_linear":
                maps = maps.expand(-1, self.codes, -1, -1)
            all_latents = offsets + torch.einsum("ntks,tksr->ntkr", residuals, maps)
        latent = (probabilities.unsqueeze(-1) * all_latents).sum(dim=(1, 2))
        output = latent @ self.output_basis.to(device=x.device, dtype=x.dtype)
        return output.reshape(*leading, self.output_dim)

    def forward(self, x: Tensor) -> Tensor:
        hard, _codes = self.hard_output(x)
        if self.surrogate == "none" or not torch.is_grad_enabled():
            return hard
        soft = self.soft_output(x)
        return hard + (soft - soft.detach())
