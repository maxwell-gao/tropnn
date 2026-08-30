from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["IndependentGroupSums", "SumPyramid", "WalshButterfly"]


class WalshButterfly(nn.Module):
    """Fixed randomized Walsh--Hadamard mixing with shared butterfly work.

    A single Rademacher diagonal is applied at the leaves, followed by the
    complete add/subtract butterfly.  Unlike :class:`SumPyramid`, every stage
    retains both the sum and difference branch, so the ``N`` outputs are
    distinct dense signed combinations of all ``N`` inputs.  The transform is
    intentionally unnormalized: its deployed arithmetic is exactly
    ``N * log2(N)`` scalar additions/subtractions plus sign-bit loads.
    """

    def __init__(self, n_features: int, *, seed: int = 0) -> None:
        super().__init__()
        n_features = int(n_features)
        if n_features < 1 or n_features & (n_features - 1):
            raise ValueError("n_features must be a positive power of two")
        self.n_features = n_features
        self.seed = int(seed)
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        signs = 2 * torch.randint(0, 2, (n_features,), generator=generator, dtype=torch.int8) - 1
        self.register_buffer("signs", signs, persistent=True)

    @property
    def output_dim(self) -> int:
        return self.n_features

    @property
    def depth(self) -> int:
        return int(math.log2(self.n_features))

    @property
    def scalar_add_subtracts(self) -> int:
        return self.n_features * self.depth

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim < 1 or x.shape[-1] != self.n_features:
            raise ValueError(f"expected the last input dimension to be {self.n_features}, got {tuple(x.shape)}")
        current = x * self.signs.to(dtype=x.dtype)
        width = 1
        while width < self.n_features:
            blocks = current.reshape(*current.shape[:-1], -1, 2 * width)
            left = blocks[..., :width]
            right = blocks[..., width:]
            current = torch.cat((left + right, left - right), dim=-1).flatten(-2)
            width *= 2
        return current

    def extra_repr(self) -> str:
        return f"n_features={self.n_features}, depth={self.depth}, seed={self.seed}"


class IndependentGroupSums(nn.Module):
    """Fixed independent sparse group sums for relational hash predicates.

    Consecutive output groups form the two sides of one comparison.  Within
    each predicate the positive and negative groups are disjoint; different
    predicates are independently sampled and may reuse coordinates.  The
    layer has no learned parameters and uses only gathers and additions.
    """

    def __init__(
        self,
        n_features: int,
        predicates: int,
        *,
        group_size: int,
        seed: int = 0,
    ) -> None:
        super().__init__()
        n_features = int(n_features)
        predicates = int(predicates)
        group_size = int(group_size)
        if n_features < 2:
            raise ValueError("n_features must be at least two")
        if predicates < 1:
            raise ValueError("predicates must be positive")
        if group_size < 1 or 2 * group_size > n_features:
            raise ValueError("group_size must be positive and at most n_features/2")
        self.n_features = n_features
        self.predicates = predicates
        self.group_size = group_size
        self.seed = int(seed)
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        groups = torch.empty(2 * predicates, group_size, dtype=torch.long)
        for predicate in range(predicates):
            selected = torch.randperm(n_features, generator=generator)[: 2 * group_size]
            groups[2 * predicate : 2 * predicate + 2] = selected.reshape(2, group_size)
        self.register_buffer("groups", groups, persistent=True)

    @property
    def output_dim(self) -> int:
        return 2 * self.predicates

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim < 1 or x.shape[-1] != self.n_features:
            raise ValueError(f"expected the last input dimension to be {self.n_features}, got {tuple(x.shape)}")
        selected = x[..., self.groups.reshape(-1)]
        return selected.reshape(*x.shape[:-1], self.output_dim, self.group_size).sum(dim=-1)

    def extra_repr(self) -> str:
        return f"n_features={self.n_features}, predicates={self.predicates}, group_size={self.group_size}, seed={self.seed}"


class SumPyramid(nn.Module):
    """Fixed sign flips followed by a complete dyadic partial-sum pyramid.

    The final dimension is ordered from leaves to root.  For an input width
    ``N`` this produces ``N + N/2 + ... + 1 = 2N-1`` features.  The module has
    no learned parameters; autograd implements the transpose tree scatter in
    the backward pass.
    """

    def __init__(self, n_features: int, *, signed: bool = False, seed: int = 0) -> None:
        super().__init__()
        n_features = int(n_features)
        if n_features < 1 or n_features & (n_features - 1):
            raise ValueError("n_features must be a positive power of two")
        self.n_features = n_features
        self.signed = bool(signed)
        self.seed = int(seed)
        if self.signed:
            generator = torch.Generator(device="cpu").manual_seed(self.seed)
            signs = 2 * torch.randint(0, 2, (n_features,), generator=generator, dtype=torch.int8) - 1
        else:
            signs = torch.ones(n_features, dtype=torch.int8)
        self.register_buffer("signs", signs, persistent=True)

    @property
    def output_dim(self) -> int:
        return 2 * self.n_features - 1

    @property
    def depth(self) -> int:
        return int(math.log2(self.n_features))

    @property
    def level_sizes(self) -> tuple[int, ...]:
        return tuple(self.n_features >> level for level in range(self.depth + 1))

    @property
    def level_offsets(self) -> tuple[int, ...]:
        offsets: list[int] = []
        offset = 0
        for size in self.level_sizes:
            offsets.append(offset)
            offset += size
        return tuple(offsets)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim < 1 or x.shape[-1] != self.n_features:
            raise ValueError(f"expected the last input dimension to be {self.n_features}, got {tuple(x.shape)}")
        current = x * self.signs.to(dtype=x.dtype) if self.signed else x
        levels = [current]
        while current.shape[-1] > 1:
            current = current.reshape(*current.shape[:-1], current.shape[-1] // 2, 2).sum(dim=-1)
            levels.append(current)
        return torch.cat(levels, dim=-1)

    def extra_repr(self) -> str:
        return f"n_features={self.n_features}, output_dim={self.output_dim}, signed={self.signed}, seed={self.seed}"
