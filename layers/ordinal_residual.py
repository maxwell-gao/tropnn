from __future__ import annotations

import itertools
import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .chamber_lifting import permutation_rank4

OrdinalResidualKind = Literal["row", "coxeter", "coxeter_relabel", "dense", "noop"]
FactorialOrdinalResidualKind = Literal[
    "constant_canonical",
    "constant_relabel",
    "live_canonical",
    "live_relabel",
    "dense",
    "noop",
]

__all__ = [
    "FactorialOrdinalResidualBlock",
    "FactorialOrdinalResidualKind",
    "MatchedOrdinalResidualBlock",
    "OrdinalResidualKind",
    "s4_diffusion_features",
]


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


def _s4_permutations() -> Tensor:
    return torch.tensor(list(itertools.permutations(range(4))), dtype=torch.long)


def _s4_adjacency() -> Tensor:
    permutations = _s4_permutations()
    lookup = {tuple(row.tolist()): index for index, row in enumerate(permutations)}
    adjacency = torch.zeros(24, 24, dtype=torch.float64)
    for index, row in enumerate(permutations.tolist()):
        for generator in range(3):
            neighbor = list(row)
            neighbor[generator], neighbor[generator + 1] = neighbor[generator + 1], neighbor[generator]
            adjacency[index, lookup[tuple(neighbor)]] = 1.0
    return adjacency


def s4_diffusion_features(*, rank: int = 8, steps: int = 4) -> Tensor:
    """Return fixed low-frequency features tied to the S4 Cayley graph."""

    if not 1 <= rank <= 24:
        raise ValueError("rank must be in [1, 24]")
    if steps < 1:
        raise ValueError("steps must be positive")
    adjacency = _s4_adjacency()
    transition = adjacency / adjacency.sum(dim=-1, keepdim=True)
    diffusion = torch.linalg.matrix_power(0.5 * torch.eye(24, dtype=torch.float64) + 0.5 * transition, steps)
    anchors = torch.linspace(0, 23, rank, dtype=torch.float64).round().to(torch.long)
    features = diffusion.index_select(1, anchors)
    features = features - features.mean(dim=0, keepdim=True)
    rms = features.square().mean(dim=0, keepdim=True).sqrt().clamp_min(1e-12)
    return (features / rms).to(torch.float32)


class MatchedOrdinalResidualBlock(nn.Module):
    """Parameter-matched Euclidean, Coxeter, and dense residual updates.

    For ``dim=784`` every non-noop arm owns exactly 18,816 parameters:

    * row: ``196 * 24 * 4`` chamber-vector entries;
    * coxeter: ``196 * 8 * 12`` graph-feature/lifting coefficients;
    * dense: bias-free ``784 -> 12 -> 784`` ridge weights.

    All arms use the same fixed regrouping and pre-RMS normalization.  The
    relabel control permutes Cayley diffusion features relative to the exact
    chamber labels while leaving parameter count and live lifting work fixed.
    """

    chamber_count = 24
    feature_rank = 8
    lifting_steps = 12

    def __init__(
        self,
        dim: int,
        *,
        kind: OrdinalResidualKind,
        seed: int,
        residual_scale: float = 0.25,
        rms_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if dim < 4 or dim % 4:
            raise ValueError("dim must be positive and divisible by four")
        if kind not in {"row", "coxeter", "coxeter_relabel", "dense", "noop"}:
            raise ValueError(f"unsupported residual kind {kind!r}")
        if residual_scale <= 0:
            raise ValueError("residual_scale must be positive")
        self.dim = int(dim)
        self.groups = self.dim // 4
        self.kind = kind
        self.residual_scale = float(residual_scale)
        self.rms_eps = float(rms_eps)

        generator = torch.Generator(device="cpu").manual_seed(seed)
        permutation = torch.arange(dim) if seed == 0 else torch.randperm(dim, generator=generator)
        self.register_buffer("permutation", permutation)
        self.register_buffer("inverse_permutation", torch.argsort(permutation))
        features = s4_diffusion_features(rank=self.feature_rank)
        if kind == "coxeter_relabel":
            relabel_generator = torch.Generator(device="cpu").manual_seed(seed + 0xC0FFEE)
            relabel = torch.randperm(self.chamber_count, generator=relabel_generator)
        else:
            relabel = torch.arange(self.chamber_count)
        self.register_buffer("chamber_relabel", relabel)
        self.register_buffer("chamber_features", features)

        if kind == "row":
            self.row = nn.Parameter(torch.zeros(self.groups, self.chamber_count, 4))
        elif kind in {"coxeter", "coxeter_relabel"}:
            self.feature_weight = nn.Parameter(torch.zeros(self.groups, self.feature_rank, self.lifting_steps))
        elif kind == "dense":
            hidden = self.groups // 16
            if 2 * self.dim * hidden != self.groups * self.chamber_count * 4:
                raise ValueError("dimension does not admit the exact dense match")
            self.w1 = nn.Parameter(torch.empty(hidden, self.dim))
            self.w2 = nn.Parameter(torch.zeros(self.dim, hidden))
            nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5), generator=generator)

    @property
    def operator_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def expected_operator_parameters(self) -> int:
        return 0 if self.kind == "noop" else self.groups * self.chamber_count * 4

    def _normalize(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + self.rms_eps).to(x.dtype)

    def chamber_codes(self, x: Tensor) -> Tensor:
        grouped = x.index_select(-1, self.permutation).reshape(*x.shape[:-1], self.groups, 4)
        return permutation_rank4(torch.argsort(grouped, dim=-1, stable=True))

    def _row_delta(self, normalized: Tensor, chamber: Tensor) -> Tensor:
        flat_chamber = chamber.reshape(-1, self.groups)
        offsets = torch.arange(self.groups, device=normalized.device).view(1, self.groups) * self.chamber_count
        rows = self.row.reshape(self.groups * self.chamber_count, 4).index_select(0, (flat_chamber + offsets).reshape(-1))
        grouped = rows.reshape(*normalized.shape[:-1], self.groups, 4)
        return grouped.reshape(*normalized.shape[:-1], self.dim).index_select(-1, self.inverse_permutation)

    def _coxeter_delta(self, normalized: Tensor, chamber: Tensor) -> Tensor:
        prefix = normalized.shape[:-1]
        grouped = normalized.index_select(-1, self.permutation).reshape(-1, self.groups, 4)
        order = torch.argsort(grouped, dim=-1, stable=True)
        sorted_values = torch.gather(grouped, -1, order)
        labels = self.chamber_relabel.index_select(0, chamber.reshape(-1)).reshape(-1, self.groups)
        features = self.chamber_features.index_select(0, labels.reshape(-1)).reshape(-1, self.groups, self.feature_rank)
        coefficients = torch.einsum("bgr,grs->bgs", features, self.feature_weight)
        coefficients = torch.tanh(coefficients)
        values = [sorted_values[..., index] for index in range(4)]
        for step, (target, source) in enumerate(_LIFTING_PAIRS):
            values[target] = values[target] + coefficients[..., step] * values[source]
        lifted = torch.stack(values, dim=-1)
        sorted_delta = lifted - sorted_values
        inverse_order = torch.argsort(order, dim=-1)
        grouped_delta = torch.gather(sorted_delta, -1, inverse_order)
        delta = grouped_delta.reshape(*prefix, self.dim).index_select(-1, self.inverse_permutation)
        return delta

    def forward_with_codes(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if x.shape[-1] != self.dim:
            raise ValueError(f"expected final dimension {self.dim}, got {tuple(x.shape)}")
        normalized = self._normalize(x)
        before = self.chamber_codes(normalized)
        if self.kind == "noop":
            output = x
        elif self.kind == "row":
            output = x + self.residual_scale * self._row_delta(normalized, before)
        elif self.kind in {"coxeter", "coxeter_relabel"}:
            output = x + self.residual_scale * self._coxeter_delta(normalized, before)
        else:
            hidden = F.relu(F.linear(normalized, self.w1))
            output = x + self.residual_scale * F.linear(hidden, self.w2)
        after = self.chamber_codes(self._normalize(output))
        return output, before, after

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_with_codes(x)[0]


class FactorialOrdinalResidualBlock(nn.Module):
    """Minimal constant/live x canonical/relabel ordinal residual factorial."""

    chamber_count = 24
    feature_rank = 8
    action_rank = 4

    def __init__(
        self,
        dim: int,
        *,
        kind: FactorialOrdinalResidualKind,
        seed: int,
        residual_scale: float = 0.25,
        rms_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        valid = {
            "constant_canonical",
            "constant_relabel",
            "live_canonical",
            "live_relabel",
            "dense",
            "noop",
        }
        if dim < 4 or dim % 4:
            raise ValueError("dim must be positive and divisible by four")
        if kind not in valid:
            raise ValueError(f"unsupported residual kind {kind!r}")
        self.dim = int(dim)
        self.groups = self.dim // 4
        self.kind = kind
        self.residual_scale = float(residual_scale)
        self.rms_eps = float(rms_eps)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        permutation = torch.arange(dim) if seed == 0 else torch.randperm(dim, generator=generator)
        self.register_buffer("permutation", permutation)
        self.register_buffer("inverse_permutation", torch.argsort(permutation))
        self.register_buffer("chamber_features", s4_diffusion_features(rank=self.feature_rank))
        if kind.endswith("relabel"):
            relabel_generator = torch.Generator(device="cpu").manual_seed(seed + 0xFA17)
            relabel = torch.randperm(self.chamber_count, generator=relabel_generator)
        else:
            relabel = torch.arange(self.chamber_count)
        self.register_buffer("chamber_relabel", relabel)

        if kind.startswith(("constant", "live")):
            self.feature_weight = nn.Parameter(torch.zeros(self.groups, self.feature_rank, self.action_rank))
        elif kind == "dense":
            hidden = self.action_rank
            self.w1 = nn.Parameter(torch.empty(hidden, self.dim))
            self.w2 = nn.Parameter(torch.zeros(self.dim, hidden))
            nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5), generator=generator)

    @property
    def operator_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def expected_operator_parameters(self) -> int:
        return 0 if self.kind == "noop" else self.groups * self.feature_rank * self.action_rank

    def _normalize(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + self.rms_eps).to(x.dtype)

    def _group_order_chamber(self, normalized: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        grouped = normalized.index_select(-1, self.permutation).reshape(-1, self.groups, 4)
        order = torch.argsort(grouped, dim=-1, stable=True)
        chamber = permutation_rank4(order)
        return grouped, order, chamber

    def chamber_codes(self, x: Tensor) -> Tensor:
        return self._group_order_chamber(x)[2].reshape(*x.shape[:-1], self.groups)

    def _coefficients(self, chamber: Tensor) -> Tensor:
        labels = self.chamber_relabel.index_select(0, chamber.reshape(-1)).reshape(-1, self.groups)
        features = self.chamber_features.index_select(0, labels.reshape(-1)).reshape(-1, self.groups, self.feature_rank)
        return torch.tanh(torch.einsum("bgr,gra->bga", features, self.feature_weight))

    def _factorial_delta(self, normalized: Tensor, *, live: bool) -> tuple[Tensor, Tensor]:
        prefix = normalized.shape[:-1]
        grouped, order, chamber = self._group_order_chamber(normalized)
        sorted_values = torch.gather(grouped, -1, order)
        coefficients = self._coefficients(chamber)
        if live:
            delta = torch.zeros_like(sorted_values)
            gaps = sorted_values[..., 1:] - sorted_values[..., :-1]
            adjacent = coefficients[..., :3] * gaps
            delta[..., :3] += adjacent
            delta[..., 1:] -= adjacent
            centered = sorted_values - sorted_values.mean(dim=-1, keepdim=True)
            delta += coefficients[..., 3:4] * centered
        else:
            delta = coefficients
        inverse_order = torch.argsort(order, dim=-1)
        grouped_delta = torch.gather(delta, -1, inverse_order)
        restored = grouped_delta.reshape(*prefix, self.dim).index_select(-1, self.inverse_permutation)
        return restored, chamber.reshape(*prefix, self.groups)

    def forward_with_codes(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if x.shape[-1] != self.dim:
            raise ValueError(f"expected final dimension {self.dim}, got {tuple(x.shape)}")
        normalized = self._normalize(x)
        if self.kind == "noop":
            before = self.chamber_codes(normalized)
            output = x
        elif self.kind.startswith("constant"):
            delta, before = self._factorial_delta(normalized, live=False)
            output = x + self.residual_scale * delta
        elif self.kind.startswith("live"):
            delta, before = self._factorial_delta(normalized, live=True)
            output = x + self.residual_scale * delta
        else:
            before = self.chamber_codes(normalized)
            output = x + self.residual_scale * F.linear(F.relu(F.linear(normalized, self.w1)), self.w2)
        after = self.chamber_codes(self._normalize(output))
        return output, before, after

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_with_codes(x)[0]
