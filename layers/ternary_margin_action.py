from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
import torch.nn.functional as F

TernaryMarginActionMode = Literal["linear", "two_sided"]

__all__ = ["TernaryMarginAction", "TernaryMarginActionMode"]


def _ternary_ste(master: Tensor, threshold: float) -> Tensor:
    """Materialize {-1, 0, +1} codes while passing the master gradient."""

    bounded = torch.tanh(master)
    hard = torch.where(
        bounded > threshold,
        torch.ones_like(bounded),
        torch.where(bounded < -threshold, -torch.ones_like(bounded), torch.zeros_like(bounded)),
    )
    return bounded + (hard - bounded).detach()


def _ternary_master_from_codes(codes: Tensor, *, magnitude: float = 0.75) -> Tensor:
    if not 0.0 < magnitude < 1.0:
        raise ValueError("magnitude must be in (0, 1)")
    return codes.to(torch.float32) * math.atanh(magnitude)


@dataclass(frozen=True)
class TernaryMarginActionSpec:
    input_dim: int
    output_dim: int
    atoms: int
    fan_in: int
    mode: TernaryMarginActionMode
    ternary_threshold: float
    use_output_scaling: bool

    def __post_init__(self) -> None:
        if self.input_dim < 1 or self.output_dim < 1:
            raise ValueError("input_dim and output_dim must be positive")
        if self.atoms < 1:
            raise ValueError("atoms must be positive")
        if not 1 <= self.fan_in <= self.input_dim:
            raise ValueError("fan_in must be in [1, input_dim]")
        if self.mode not in {"linear", "two_sided"}:
            raise ValueError(f"unsupported mode {self.mode!r}")
        if not 0.0 < self.ternary_threshold < 1.0:
            raise ValueError("ternary_threshold must be in (0, 1)")

    @property
    def sides(self) -> int:
        return 1 if self.mode == "linear" else 2

    @property
    def output_scale(self) -> float:
        # A random atom has margin variance proportional to fan_in, and every
        # output coordinate receives atoms contributions.  This fixed factor
        # is a layer scale, not a learned multiply in the deployed action.
        return 1.0 / math.sqrt(float(self.atoms * self.fan_in)) if self.use_output_scaling else 1.0


class TernaryMarginAction(nn.Module):
    r"""Sparse ternary recognition followed by a dense ternary live action.

    Each atom computes a live sparse margin

    .. math::

        m_t(x) = \sum_{i \in S_t} a_{ti} x_i - \theta_t,
        \qquad a_{ti} \in \{-1, 0, +1\}.

    ``mode="two_sided"`` returns

    .. math::

        \sum_t q_t^+ [m_t]_+ + q_t^- [-m_t]_+,
        \qquad q_t^\pm \in \{-1, 0, +1\}^{D_{out}},

    while ``mode="linear"`` returns :math:`\sum_t q_t m_t`.  The hard
    forward therefore uses only gather, add/subtract, threshold/ReLU, and
    ternary select-add semantics.  The Torch reference intentionally uses
    dense tensor contractions for trainability; it is not the packed kernel.
    """

    is_ternary_margin_action = True

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        atoms: int = 64,
        fan_in: int = 6,
        mode: TernaryMarginActionMode = "two_sided",
        seed: int = 0,
        ternary_threshold: float = 0.5,
        use_output_scaling: bool = True,
        fixed_zero_threshold: bool = False,
    ) -> None:
        super().__init__()
        spec = TernaryMarginActionSpec(
            input_dim=int(input_dim),
            output_dim=int(output_dim),
            atoms=int(atoms),
            fan_in=int(fan_in),
            mode=mode,
            ternary_threshold=float(ternary_threshold),
            use_output_scaling=bool(use_output_scaling),
        )
        self.spec = spec
        self.input_dim = spec.input_dim
        self.output_dim = spec.output_dim
        self.tables = spec.atoms
        self.atoms = spec.atoms
        self.comparisons = spec.fan_in
        self.fan_in = spec.fan_in
        self.mode = spec.mode
        self.sides = spec.sides
        self.ternary_threshold = spec.ternary_threshold
        self.output_scale = spec.output_scale

        # Compatibility with the controlled EMNIST payload harness.
        self.table_size = 2
        self.payload_width = self.output_dim
        self.write_degree = self.output_dim

        generator = torch.Generator(device="cpu").manual_seed(seed)
        supports = self._make_balanced_supports(generator)
        self.register_buffer("support_indices", supports)

        input_codes = torch.randint(0, 2, (self.atoms, self.fan_in), generator=generator, dtype=torch.int8)
        input_codes = input_codes.to(torch.float32).mul_(2.0).sub_(1.0)
        self.input_master = nn.Parameter(_ternary_master_from_codes(input_codes))

        direction_codes = torch.randint(
            -1,
            2,
            (self.atoms, self.sides, self.output_dim),
            generator=generator,
            dtype=torch.int8,
        )
        self.direction_master = nn.Parameter(_ternary_master_from_codes(direction_codes))

        thresholds = torch.zeros(self.atoms)
        if fixed_zero_threshold:
            self.register_buffer("thresholds", thresholds)
        else:
            self.thresholds = nn.Parameter(thresholds)

    def _make_balanced_supports(self, generator: torch.Generator) -> Tensor:
        count = self.atoms * self.fan_in
        permutation = torch.randperm(self.input_dim, generator=generator)
        positions = torch.arange(count, dtype=torch.long).remainder(self.input_dim)
        return permutation.index_select(0, positions).view(self.atoms, self.fan_in)

    @property
    def semantic_route_terms(self) -> int:
        return self.atoms * self.fan_in

    @property
    def semantic_action_terms(self) -> int:
        # Exactly one side is active for each atom in two-sided mode.
        return self.atoms * self.output_dim

    @property
    def payload_params(self) -> int:
        return self.input_master.numel() + self.direction_master.numel()

    @property
    def bias_generator_params(self) -> int:
        return self.direction_master.numel()

    @property
    def slope_coeff_params(self) -> int:
        return self.input_master.numel()

    @property
    def slope_generator_params(self) -> int:
        return 0

    def payload_parameters(self) -> list[Tensor]:
        return [self.input_master, self.direction_master]

    def threshold_parameters(self) -> list[Tensor]:
        return [self.thresholds] if isinstance(self.thresholds, nn.Parameter) else []

    def clear_packed_payload_cache(self) -> None:
        return None

    def hard_input_codes(self) -> Tensor:
        bounded = torch.tanh(self.input_master.detach())
        return torch.where(
            bounded > self.ternary_threshold,
            torch.ones_like(bounded),
            torch.where(bounded < -self.ternary_threshold, -torch.ones_like(bounded), torch.zeros_like(bounded)),
        ).to(torch.int8)

    def hard_direction_codes(self) -> Tensor:
        bounded = torch.tanh(self.direction_master.detach())
        return torch.where(
            bounded > self.ternary_threshold,
            torch.ones_like(bounded),
            torch.where(bounded < -self.ternary_threshold, -torch.ones_like(bounded), torch.zeros_like(bounded)),
        ).to(torch.int8)

    def _margins(self, x_flat: Tensor) -> Tensor:
        support = self.support_indices.to(device=x_flat.device).reshape(-1)
        selected = x_flat.index_select(-1, support).view(x_flat.shape[0], self.atoms, self.fan_in)
        coefficients = _ternary_ste(self.input_master, self.ternary_threshold).to(
            device=x_flat.device, dtype=x_flat.dtype
        )
        return (selected * coefficients.unsqueeze(0)).sum(dim=-1) - self.thresholds.to(
            device=x_flat.device, dtype=x_flat.dtype
        ).view(1, self.atoms)

    def compute(self, x: Tensor) -> tuple[Tensor, Tensor]:
        input_dtype = x.dtype
        if x.ndim == 0 or x.shape[-1] != self.input_dim:
            raise ValueError(f"TernaryMarginAction expected last dimension {self.input_dim}, got shape {tuple(x.shape)}")
        prefix = x.shape[:-1]
        x_flat = x.reshape(-1, self.input_dim).float()
        margins = self._margins(x_flat)
        directions = _ternary_ste(self.direction_master, self.ternary_threshold).to(
            device=x_flat.device, dtype=x_flat.dtype
        )

        if self.mode == "linear":
            output = margins @ directions[:, 0, :]
        else:
            positive = F.relu(margins)
            negative = F.relu(-margins)
            output = positive @ directions[:, 0, :] + negative @ directions[:, 1, :]

        indices = (margins > 0).to(torch.long)
        return (
            (output * self.output_scale).view(*prefix, self.output_dim).to(dtype=input_dtype),
            indices.view(*prefix, self.atoms),
        )

    def forward(self, x: Tensor) -> Tensor:
        output, _indices = self.compute(x)
        return output

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, output_dim={self.output_dim}, atoms={self.atoms}, "
            f"fan_in={self.fan_in}, mode={self.mode!r}, ternary_threshold={self.ternary_threshold}, "
            f"output_scale={self.output_scale:.6g}"
        )
