from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

__all__ = ["ChamberLiftingStage", "ChamberLiftingTower", "permutation_rank4"]


CoefficientMode = Literal["float", "ternary"]

# A lower triangular sweep followed by an upper triangular sweep. Every
# operation is a lifting update y[target] += coefficient * y[source].
_LIFTING_PAIRS = (
    (1, 0),
    (2, 0),
    (2, 1),
    (3, 0),
    (3, 1),
    (3, 2),
    (2, 3),
    (1, 3),
    (1, 2),
    (0, 3),
    (0, 2),
    (0, 1),
)


def permutation_rank4(order: Tensor) -> Tensor:
    """Return the lexicographic rank in S4 for a final-axis permutation."""

    if order.shape[-1] != 4:
        raise ValueError(f"permutation_rank4 expects a final dimension of 4, got {tuple(order.shape)}")
    rank = torch.zeros(order.shape[:-1], device=order.device, dtype=torch.long)
    for position, factorial in enumerate((6, 2, 1)):
        smaller = (order[..., position + 1 :] < order[..., position : position + 1]).sum(dim=-1)
        rank += smaller * factorial
    return rank


def _ternary_ste(master: Tensor, threshold: float) -> Tensor:
    hard = torch.where(
        master > threshold,
        torch.ones_like(master),
        torch.where(master < -threshold, -torch.ones_like(master), torch.zeros_like(master)),
    )
    return master + (hard - master).detach()


class ChamberLiftingStage(nn.Module):
    """One S4-routed local lifting stage.

    Coordinates are put into fixed groups of four. Each quartet's legal S4
    chamber selects twelve lifting coefficients, and a lower/upper sequence of
    local updates acts on the quartet in its original coordinate order. In
    ternary mode the hard forward uses only
    ``{-lift_scale, 0, +lift_scale}`` coefficients.

    This differs from a chamber payload LUT: the selected row acts on live
    input values instead of replacing them with a memorized vector.
    """

    chamber_count = 24
    lifting_steps = len(_LIFTING_PAIRS)

    def __init__(
        self,
        dim: int,
        *,
        permutation: Tensor,
        coefficient_mode: CoefficientMode = "ternary",
        lift_scale: float = 0.25,
        ternary_threshold: float = 0.5,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if dim < 4 or dim % 4 != 0:
            raise ValueError(f"ChamberLiftingStage requires dim divisible by 4, got {dim}")
        if coefficient_mode not in {"float", "ternary"}:
            raise ValueError(f"coefficient_mode must be 'float' or 'ternary', got {coefficient_mode!r}")
        if lift_scale <= 0:
            raise ValueError(f"lift_scale must be positive, got {lift_scale}")
        if ternary_threshold <= 0:
            raise ValueError(f"ternary_threshold must be positive, got {ternary_threshold}")
        permutation = torch.as_tensor(permutation, dtype=torch.long)
        if tuple(permutation.shape) != (dim,) or not torch.equal(torch.sort(permutation).values, torch.arange(dim)):
            raise ValueError("permutation must contain every coordinate exactly once")

        self.dim = int(dim)
        self.groups = self.dim // 4
        self.coefficient_mode = coefficient_mode
        self.lift_scale = float(lift_scale)
        self.ternary_threshold = float(ternary_threshold)
        self.register_buffer("permutation", permutation)
        self.register_buffer("inverse_permutation", torch.argsort(permutation))

        generator = torch.Generator(device="cpu").manual_seed(seed)
        base = torch.empty(self.groups, self.lifting_steps)
        if coefficient_mode == "ternary":
            base.uniform_(-0.75, 0.75, generator=generator)
        else:
            base.uniform_(-0.1, 0.1, generator=generator)
        self.base_coefficient_master = nn.Parameter(base)
        self.coefficient_master = nn.Parameter(torch.zeros(self.groups, self.chamber_count, self.lifting_steps))

    def materialized_coefficients(self) -> Tensor:
        master = self.base_coefficient_master[:, None, :] + self.coefficient_master
        if self.coefficient_mode == "ternary":
            base = _ternary_ste(master, self.ternary_threshold)
        else:
            base = torch.tanh(master)
        return base * self.lift_scale

    def hard_ternary_codes(self) -> Tensor:
        """Return integer coefficient codes for inspection or compilation."""

        master = (self.base_coefficient_master[:, None, :] + self.coefficient_master).detach()
        return torch.where(
            master > self.ternary_threshold,
            torch.ones_like(master),
            torch.where(master < -self.ternary_threshold, -torch.ones_like(master), torch.zeros_like(master)),
        ).to(torch.int8)

    def _selected_coefficients(self, chamber: Tensor) -> Tensor:
        coefficients = self.materialized_coefficients()
        offsets = torch.arange(self.groups, device=chamber.device).view(1, self.groups) * self.chamber_count
        flat_indices = (chamber + offsets).reshape(-1)
        selected = coefficients.reshape(self.groups * self.chamber_count, self.lifting_steps).index_select(0, flat_indices)
        return selected.view(chamber.shape[0], self.groups, self.lifting_steps)

    def forward_with_chambers(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if x.shape[-1] != self.dim:
            raise ValueError(f"ChamberLiftingStage expected final dimension {self.dim}, got {tuple(x.shape)}")
        original_shape = x.shape
        flat = x.reshape(-1, self.dim)
        grouped = flat.index_select(-1, self.permutation).view(flat.shape[0], self.groups, 4)

        order = torch.argsort(grouped, dim=-1, stable=True)
        chamber = permutation_rank4(order)
        coefficients = self._selected_coefficients(chamber).to(dtype=grouped.dtype)

        values = [grouped[..., index] for index in range(4)]
        for step, (target, source) in enumerate(_LIFTING_PAIRS):
            values[target] = values[target] + coefficients[..., step] * values[source]
        lifted_grouped = torch.stack(values, dim=-1)
        lifted_permuted = lifted_grouped.reshape(flat.shape[0], self.dim)
        output = lifted_permuted.index_select(-1, self.inverse_permutation)
        return output.view(original_shape), chamber.view(*original_shape[:-1], self.groups)

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_with_chambers(x)[0]


class ChamberLiftingTower(nn.Module):
    """Recursive global mixer made from chamber-conditioned local lifting.

    Each stage uses a different fixed grouping permutation. The selected
    lifting operator changes values that determine all later chambers, so
    ``depth`` is recursive comparison depth rather than additive table width.
    """

    def __init__(
        self,
        dim: int,
        *,
        depth: int,
        coefficient_mode: CoefficientMode = "ternary",
        lift_scale: float = 0.25,
        ternary_threshold: float = 0.5,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be positive, got {depth}")
        if dim < 4 or dim % 4 != 0:
            raise ValueError(f"ChamberLiftingTower requires dim divisible by 4, got {dim}")
        self.dim = int(dim)
        self.depth = int(depth)
        self.coefficient_mode = coefficient_mode
        self.lift_scale = float(lift_scale)

        generator = torch.Generator(device="cpu").manual_seed(seed + 17)
        stages = []
        for stage_index in range(self.depth):
            permutation = torch.arange(self.dim) if stage_index == 0 else torch.randperm(self.dim, generator=generator)
            stages.append(
                ChamberLiftingStage(
                    self.dim,
                    permutation=permutation,
                    coefficient_mode=coefficient_mode,
                    lift_scale=lift_scale,
                    ternary_threshold=ternary_threshold,
                    seed=seed + 1009 * (stage_index + 1),
                )
            )
        self.stages = nn.ModuleList(stages)

    def forward_with_chambers(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        chambers: list[Tensor] = []
        for stage in self.stages:
            x, stage_chambers = stage.forward_with_chambers(x)
            chambers.append(stage_chambers)
        return x, chambers

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_with_chambers(x)[0]

    def receptive_field_sizes(self) -> tuple[int, int]:
        """Return min/max structural input support after all groupings."""

        supports = [{index} for index in range(self.dim)]
        for stage in self.stages:
            permutation = stage.permutation.tolist()
            next_supports: list[set[int]] = [set() for _ in range(self.dim)]
            for group_index in range(stage.groups):
                group = permutation[4 * group_index : 4 * (group_index + 1)]
                union: set[int] = set()
                for coordinate in group:
                    union.update(supports[coordinate])
                for coordinate in group:
                    next_supports[coordinate] = set(union)
            supports = next_supports
        sizes = [len(support) for support in supports]
        return min(sizes), max(sizes)

    def ternary_nonzero_fraction(self) -> float:
        if self.coefficient_mode != "ternary":
            return float("nan")
        codes = torch.cat([stage.hard_ternary_codes().reshape(-1) for stage in self.stages])
        return float((codes != 0).float().mean().item())

    @property
    def operator_parameters(self) -> int:
        return sum(stage.base_coefficient_master.numel() + stage.coefficient_master.numel() for stage in self.stages)

    @property
    def active_operator_reads_per_item(self) -> int:
        return self.depth * (self.dim // 4)

    @property
    def integer_adds_per_item(self) -> int:
        return self.active_operator_reads_per_item * len(_LIFTING_PAIRS)

    def extra_repr(self) -> str:
        receptive_min, receptive_max = self.receptive_field_sizes()
        return (
            f"dim={self.dim}, depth={self.depth}, coefficient_mode={self.coefficient_mode}, "
            f"lift_scale={self.lift_scale}, receptive_field={receptive_min}-{receptive_max}, "
            f"operator_params={self.operator_parameters}, adds={self.integer_adds_per_item}"
        )
