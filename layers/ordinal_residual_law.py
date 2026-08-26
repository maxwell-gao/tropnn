from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

from .chamber_lifting import permutation_rank4
from .ordinal_residual import s4_diffusion_features

OrdinalResidualLaw = Literal["noop", "euclidean_euler", "intrinsic_exp"]

__all__ = ["OrdinalResidualLaw", "OrdinalResidualLawBlock"]


class OrdinalResidualLawBlock(nn.Module):
    """Matched Euler versus intrinsic retraction on an S4 chamber.

    Each sorted quartet is represented by its mean and three positive adjacent
    gaps.  Both learned arms define the same tangent field

        mean_dot = a0, gap_dot_i = gap_i * a_i.

    ``euclidean_euler`` takes an ambient Euler step. ``intrinsic_exp`` follows
    the exact multiplicative flow on the positive gap cone.  Thus only the
    residual composition law changes; address, coefficients, parameters, and
    first-order vector field are shared.
    """

    chamber_count = 24
    feature_rank = 8
    action_rank = 4

    def __init__(self, dim: int, *, law: OrdinalResidualLaw, seed: int, residual_scale: float = 0.25) -> None:
        super().__init__()
        if dim < 4 or dim % 4:
            raise ValueError("dim must be positive and divisible by four")
        if law not in {"noop", "euclidean_euler", "intrinsic_exp"}:
            raise ValueError(f"unsupported residual law {law!r}")
        if not 0.0 < residual_scale <= 0.5:
            raise ValueError("residual_scale must be in (0, 0.5]")
        self.dim = int(dim)
        self.groups = self.dim // 4
        self.law = law
        self.residual_scale = float(residual_scale)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        permutation = torch.arange(dim) if seed == 0 else torch.randperm(dim, generator=generator)
        self.register_buffer("permutation", permutation)
        self.register_buffer("inverse_permutation", torch.argsort(permutation))
        self.register_buffer("chamber_features", s4_diffusion_features(rank=self.feature_rank))
        if law != "noop":
            self.feature_weight = nn.Parameter(torch.zeros(self.groups, self.feature_rank, self.action_rank))

    @property
    def operator_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def expected_operator_parameters(self) -> int:
        return 0 if self.law == "noop" else self.groups * self.feature_rank * self.action_rank

    def _group_order_chamber(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        grouped = x.index_select(-1, self.permutation).reshape(-1, self.groups, 4)
        order = torch.argsort(grouped, dim=-1, stable=True)
        return grouped, order, permutation_rank4(order)

    def chamber_codes(self, x: Tensor) -> Tensor:
        chamber = self._group_order_chamber(x)[2]
        return chamber.reshape(*x.shape[:-1], self.groups)

    def _coefficients(self, chamber: Tensor) -> Tensor:
        features = self.chamber_features.index_select(0, chamber.reshape(-1)).reshape(-1, self.groups, self.feature_rank)
        return torch.tanh(torch.einsum("bgr,gra->bga", features, self.feature_weight))

    @staticmethod
    def _from_mean_gaps(mean: Tensor, gaps: Tensor) -> Tensor:
        first = mean - (3.0 * gaps[..., 0] + 2.0 * gaps[..., 1] + gaps[..., 2]) / 4.0
        second = first + gaps[..., 0]
        third = second + gaps[..., 1]
        fourth = third + gaps[..., 2]
        return torch.stack((first, second, third, fourth), dim=-1)

    def forward_with_codes(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if x.shape[-1] != self.dim:
            raise ValueError(f"expected final dimension {self.dim}, got {tuple(x.shape)}")
        prefix = x.shape[:-1]
        grouped, order, chamber = self._group_order_chamber(x)
        before = chamber.reshape(*prefix, self.groups)
        if self.law == "noop":
            output = x
        else:
            sorted_values = torch.gather(grouped, -1, order)
            mean = sorted_values.mean(dim=-1)
            gaps = sorted_values[..., 1:] - sorted_values[..., :-1]
            coefficients = self._coefficients(chamber)
            updated_mean = mean + self.residual_scale * coefficients[..., 0]
            rates = coefficients[..., 1:]
            if self.law == "euclidean_euler":
                updated_gaps = gaps * (1.0 + self.residual_scale * rates)
            else:
                updated_gaps = gaps * torch.exp(self.residual_scale * rates)
            # Reconstruct only the chart displacement.  This is algebraically
            # the same retraction as rebuilding the absolute sorted values,
            # while making zero initialization bit-exact identity.
            sorted_delta = self._from_mean_gaps(updated_mean - mean, updated_gaps - gaps)
            grouped_delta = torch.gather(sorted_delta, -1, torch.argsort(order, dim=-1))
            delta = grouped_delta.reshape(*prefix, self.dim).index_select(-1, self.inverse_permutation)
            output = x + delta
        after = self.chamber_codes(output)
        return output, before, after

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_with_codes(x)[0]
