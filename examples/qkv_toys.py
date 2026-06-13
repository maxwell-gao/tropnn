"""Toy tests for non-inner-product substitutes of QKV attention.

The goal is deliberately narrower than language modeling:

1. Compatibility: can a score function learn whether two vectors match?
2. Selective aggregation: can scores select the correct candidate value?
3. Value transform: can a LUT value path approximate a dense value map?

The pairwise score variants use hard comparator hashes in the forward path and a
local one-bit neighbor surrogate in training.  Dense baselines use ordinary
linear projections and dot products.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..layers import AbsDiffLUT, PairwiseLinear
from ..layers.surrogate import ste_heaviside

ScoreFamily = Literal[
    "dense_qk",
    "pairwise_score",
    "joint_concat",
    "additive",
    "cross_lut",
    "diff_lut",
    "aligned_diff_lut",
    "same_coord_lut",
    "same_coord_bidir_lut",
    "absdiff_lut",
    "relation_walsh",
    "binary_code",
]
AggregationMode = Literal["softmax", "smooth_relu", "logsumexp", "hard_top1"]
ValuePayload = Literal["key", "pairwise", "pairwise_delta"]


@dataclass
class ToyConfig:
    dim: int = 32
    train_classes: int = 64
    ood_classes: int = 32
    train_pairs: int = 8192
    test_pairs: int = 2048
    seq_train: int = 4096
    seq_test: int = 1024
    value_train: int = 8192
    value_test: int = 2048
    candidates: int = 8
    value_classes: int = 16
    out_dim: int = 16
    noise: float = 0.15
    tables: int = 16
    comparisons: int = 4
    rank: int = 16
    steps: int = 500
    batch_size: int = 256
    lr: float = 3e-3
    aggregation_mode: AggregationMode = "softmax"
    value_payload: ValuePayload = "key"
    margin_lambda: float = 0.0
    margin: float = 0.25
    teacher_steps: int = 200
    distill_alpha: float = 1.0
    distill_temperature: float = 1.0
    anchor_sample_batches: int = 16
    anchor_candidate_pairs: int = 2048
    seed: int = 0
    device: str = "auto"
    output_dir: str = "python/results/qkv_toys"


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sample_class_vectors(prototypes: Tensor, labels: Tensor, noise: float) -> Tensor:
    x = prototypes.index_select(0, labels)
    return x + torch.randn_like(x) * noise


def make_pair_dataset(
    prototypes: Tensor,
    n_pairs: int,
    *,
    noise: float,
    balanced: bool,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    n_classes = prototypes.shape[0]
    y = torch.arange(n_pairs, device=device) % 2 if balanced else torch.randint(0, 2, (n_pairs,), device=device)
    cls_a = torch.randint(0, n_classes, (n_pairs,), device=device)
    cls_b = torch.randint(0, n_classes, (n_pairs,), device=device)
    cls_b = torch.where(y.bool(), cls_a, (cls_a + 1 + cls_b % max(1, n_classes - 1)) % n_classes)
    x_a = _sample_class_vectors(prototypes, cls_a, noise)
    x_b = _sample_class_vectors(prototypes, cls_b, noise)
    return x_a, x_b, y.float()


def make_aggregation_dataset(
    prototypes: Tensor,
    n_items: int,
    *,
    candidates: int,
    value_classes: int,
    noise: float,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    n_classes = prototypes.shape[0]
    query_cls = torch.randint(0, n_classes, (n_items,), device=device)
    target_pos = torch.randint(0, candidates, (n_items,), device=device)
    cand_cls = torch.randint(0, n_classes, (n_items, candidates), device=device)
    cand_cls = torch.where(torch.arange(candidates, device=device).view(1, -1) == target_pos.view(-1, 1), query_cls.view(-1, 1), cand_cls)
    # Avoid accidental extra positives, so top-1 accuracy is interpretable.
    cand_cls = torch.where(cand_cls == query_cls.view(-1, 1), (cand_cls + 1) % n_classes, cand_cls)
    cand_cls.scatter_(1, target_pos.view(-1, 1), query_cls.view(-1, 1))

    query = _sample_class_vectors(prototypes, query_cls, noise)
    cand = _sample_class_vectors(prototypes, cand_cls.reshape(-1), noise).view(n_items, candidates, -1)
    value_ids = torch.randint(0, value_classes, (n_items, candidates), device=device)
    target_value = value_ids.gather(1, target_pos.view(-1, 1)).squeeze(1)
    values = F.one_hot(value_ids, num_classes=value_classes).float()
    return query, cand, values, target_value


def make_value_dataset(
    prototypes: Tensor,
    teacher: Tensor,
    n_items: int,
    *,
    noise: float,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    labels = torch.randint(0, prototypes.shape[0], (n_items,), device=device)
    x = _sample_class_vectors(prototypes, labels, noise)
    y = torch.tanh(x @ teacher)
    return x, y


class PairwiseHash(nn.Module):
    def __init__(self, dim: int, tables: int, comparisons: int, *, seed: int) -> None:
        super().__init__()
        self.dim = dim
        self.tables = tables
        self.comparisons = comparisons
        self.cells = 1 << comparisons
        gen = torch.Generator(device="cpu").manual_seed(seed)
        anchors = torch.zeros(tables, comparisons, 2, dtype=torch.long)
        for table_idx in range(tables):
            for comp_idx in range(comparisons):
                a = torch.randint(0, dim, (1,), generator=gen).item()
                b = torch.randint(0, dim, (1,), generator=gen).item()
                while a == b:
                    b = torch.randint(0, dim, (1,), generator=gen).item()
                anchors[table_idx, comp_idx, 0] = a
                anchors[table_idx, comp_idx, 1] = b
        self.register_buffer("anchors", anchors)
        self.thresholds = nn.Parameter(torch.zeros(tables, comparisons))
        self.register_buffer("powers", 2 ** torch.arange(comparisons, dtype=torch.long))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        flat = x.reshape(-1, x.shape[-1])
        xa = flat[:, self.anchors[..., 0].reshape(-1)].view(flat.shape[0], self.tables, self.comparisons)
        xb = flat[:, self.anchors[..., 1].reshape(-1)].view(flat.shape[0], self.tables, self.comparisons)
        margins = xa - xb - self.thresholds.to(dtype=flat.dtype, device=flat.device)
        indices = ((margins > 0).long() * self.powers.to(device=flat.device).view(1, 1, -1)).sum(dim=-1)
        r_min = margins.abs().argmin(dim=-1)
        u_min = margins.gather(-1, r_min.unsqueeze(-1)).squeeze(-1)
        shape = (*x.shape[:-1], self.tables)
        return indices.view(shape), margins.view(*x.shape[:-1], self.tables, self.comparisons), r_min.view(shape), u_min.view(shape)


class CrossRelationHash(nn.Module):
    """Hard relation hash with bits H(q_a - k_b - tau).

    Unlike hashing q and k independently, every bit directly cuts the product
    space of a query/key pair.  This is the smallest change that makes the
    comparator see relative q/k geometry before the lookup.
    """

    def __init__(self, dim: int, tables: int, comparisons: int, *, seed: int) -> None:
        super().__init__()
        self.dim = dim
        self.tables = tables
        self.comparisons = comparisons
        self.cells = 1 << comparisons
        gen = torch.Generator(device="cpu").manual_seed(seed)
        q_anchors = torch.randint(0, dim, (tables, comparisons), generator=gen)
        k_anchors = torch.randint(0, dim, (tables, comparisons), generator=gen)
        self.register_buffer("q_anchors", q_anchors)
        self.register_buffer("k_anchors", k_anchors)
        self.thresholds = nn.Parameter(torch.zeros(tables, comparisons))
        self.register_buffer("powers", 2 ** torch.arange(comparisons, dtype=torch.long))

    def margins(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        q_flat = x_i.reshape(-1, x_i.shape[-1])
        k_flat = x_j.reshape(-1, x_j.shape[-1])
        q = q_flat[:, self.q_anchors.reshape(-1)].view(q_flat.shape[0], self.tables, self.comparisons)
        k = k_flat[:, self.k_anchors.reshape(-1)].view(k_flat.shape[0], self.tables, self.comparisons)
        margins = q - k - self.thresholds.to(dtype=q_flat.dtype, device=q_flat.device)
        return margins.view(*x_i.shape[:-1], self.tables, self.comparisons)

    def forward(self, x_i: Tensor, x_j: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        margins = self.margins(x_i, x_j)
        indices = ((margins > 0).long() * self.powers.to(device=margins.device).view(1, 1, -1)).sum(dim=-1)
        r_min = margins.abs().argmin(dim=-1)
        u_min = margins.gather(-1, r_min.unsqueeze(-1)).squeeze(-1)
        return indices, margins, r_min, u_min


class SameCoordinateRelationHash(nn.Module):
    """Hard relation hash over same-coordinate query/key comparisons.

    Modes:
    - forward: bits are H(q_a - k_a - tau)
    - bidir: alternating bits are H(q_a - k_a - tau) and H(k_a - q_a - tau)
    - absdiff: bits are H(width - |q_a - k_a|)

    These comparators test whether score learning needs direct q/k coordinate
    agreement rather than ordering relations between coordinates of q - k.
    """

    def __init__(self, dim: int, tables: int, comparisons: int, *, seed: int, mode: str) -> None:
        super().__init__()
        if mode not in {"forward", "bidir", "absdiff"}:
            raise ValueError(f"unknown same-coordinate relation mode {mode}")
        self.dim = dim
        self.tables = tables
        self.comparisons = comparisons
        self.cells = 1 << comparisons
        self.mode = mode
        gen = torch.Generator(device="cpu").manual_seed(seed)
        coords = torch.randint(0, dim, (tables, comparisons), generator=gen)
        self.register_buffer("coords", coords)
        if mode == "absdiff":
            self.register_buffer("thresholds", torch.zeros(tables, comparisons))
            self.log_widths = nn.Parameter(torch.full((tables, comparisons), -1.6))
        else:
            self.thresholds = nn.Parameter(torch.zeros(tables, comparisons))
            self.register_buffer("log_widths", torch.full((tables, comparisons), -1.6))
        self.register_buffer("directions", torch.where(torch.arange(comparisons) % 2 == 0, 1.0, -1.0).view(1, comparisons))
        self.register_buffer("powers", 2 ** torch.arange(comparisons, dtype=torch.long))

    def margins(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        q_flat = x_i.reshape(-1, x_i.shape[-1])
        k_flat = x_j.reshape(-1, x_j.shape[-1])
        q = q_flat[:, self.coords.reshape(-1)].view(q_flat.shape[0], self.tables, self.comparisons)
        k = k_flat[:, self.coords.reshape(-1)].view(k_flat.shape[0], self.tables, self.comparisons)
        diff = q - k
        if self.mode == "absdiff":
            widths = F.softplus(self.log_widths).to(dtype=diff.dtype, device=diff.device)
            margins = widths - diff.abs()
        else:
            signed = diff
            if self.mode == "bidir":
                signed = signed * self.directions.to(dtype=diff.dtype, device=diff.device).view(1, 1, -1)
            margins = signed - self.thresholds.to(dtype=diff.dtype, device=diff.device)
        return margins.view(*x_i.shape[:-1], self.tables, self.comparisons)

    def forward(self, x_i: Tensor, x_j: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        margins = self.margins(x_i, x_j)
        indices = ((margins > 0).long() * self.powers.to(device=margins.device).view(1, 1, -1)).sum(dim=-1)
        r_min = margins.abs().argmin(dim=-1)
        u_min = margins.gather(-1, r_min.unsqueeze(-1)).squeeze(-1)
        return indices, margins, r_min, u_min


def _binary_metrics(logits: Tensor, y: Tensor) -> dict[str, float]:
    probs = logits.sigmoid()
    pred = probs >= 0.5
    yb = y.bool()
    return {
        "accuracy": float((pred == yb).float().mean().item()),
        "pos_score": float(logits[yb].mean().item()) if yb.any() else float("nan"),
        "neg_score": float(logits[~yb].mean().item()) if (~yb).any() else float("nan"),
        "score_margin": float((logits[yb].mean() - logits[~yb].mean()).item()) if yb.any() and (~yb).any() else float("nan"),
    }


class DenseQKScore(nn.Module):
    def __init__(self, dim: int, rank: int) -> None:
        super().__init__()
        self.q = nn.Linear(dim, rank, bias=False)
        self.k = nn.Linear(dim, rank, bias=False)

    def forward(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        q = self.q(x_i)
        k = self.k(x_j)
        return (q * k).sum(dim=-1) / math.sqrt(q.shape[-1])


class PairwiseScoreLUT(nn.Module):
    def __init__(self, dim: int, tables: int, comparisons: int, *, seed: int) -> None:
        super().__init__()
        self.q_hash = PairwiseHash(dim, tables, comparisons, seed=seed)
        self.k_hash = PairwiseHash(dim, tables, comparisons, seed=seed + 1009)
        self.tables = tables
        self.cells = 1 << comparisons
        self.score = nn.Parameter(torch.randn(tables, self.cells, self.cells) * 0.02)

    def _lookup(self, q_idx: Tensor, k_idx: Tensor) -> Tensor:
        flat_q = q_idx.reshape(-1, self.tables)
        flat_k = k_idx.reshape(-1, self.tables)
        table = self.score.to(device=q_idx.device)
        vals = []
        for table_idx in range(self.tables):
            vals.append(table[table_idx, flat_q[:, table_idx], flat_k[:, table_idx]])
        out = torch.stack(vals, dim=-1).sum(dim=-1) / math.sqrt(self.tables)
        return out.view(q_idx.shape[:-1])

    def forward(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        q_idx, _, q_r, q_u = self.q_hash(x_i)
        k_idx, _, k_r, k_u = self.k_hash(x_j)
        out = self._lookup(q_idx, k_idx)
        if self.training:
            q_neighbor = q_idx ^ (2**q_r).long()
            k_neighbor = k_idx ^ (2**k_r).long()
            q_corr = self._lookup(q_neighbor, k_idx) - out
            k_corr = self._lookup(q_idx, k_neighbor) - out
            q_ste = (ste_heaviside(q_u, "fast_sigmoid_odd") - (q_u > 0).to(q_u.dtype)).sum(dim=-1) / math.sqrt(self.tables)
            k_ste = (ste_heaviside(k_u, "fast_sigmoid_odd") - (k_u > 0).to(k_u.dtype)).sum(dim=-1) / math.sqrt(self.tables)
            out = out + q_ste * q_corr.detach() + k_ste * k_corr.detach()
        return out

    def chamber_stats(self, x_i: Tensor, x_j: Tensor) -> dict[str, float]:
        with torch.no_grad():
            q_idx, _, _, _ = self.q_hash(x_i)
            k_idx, _, _, _ = self.k_hash(x_j)
            joint = q_idx * self.cells + k_idx
            return _index_stats(joint, self.cells * self.cells)


class JointConcatScoreLUT(nn.Module):
    def __init__(self, dim: int, tables: int, comparisons: int, *, seed: int) -> None:
        super().__init__()
        self.hash = PairwiseHash(dim * 2, tables, comparisons, seed=seed)
        self.tables = tables
        self.cells = 1 << comparisons
        self.score = nn.Parameter(torch.randn(tables, self.cells) * 0.02)

    def _lookup(self, idx: Tensor) -> Tensor:
        flat = idx.reshape(-1, self.tables)
        vals = []
        table = self.score.to(device=idx.device)
        for table_idx in range(self.tables):
            vals.append(table[table_idx, flat[:, table_idx]])
        return (torch.stack(vals, dim=-1).sum(dim=-1) / math.sqrt(self.tables)).view(idx.shape[:-1])

    def forward(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        idx, _, r_min, u_min = self.hash(torch.cat([x_i, x_j], dim=-1))
        out = self._lookup(idx)
        if self.training:
            neigh = idx ^ (2**r_min).long()
            corr = self._lookup(neigh) - out
            ste = (ste_heaviside(u_min, "fast_sigmoid_odd") - (u_min > 0).to(u_min.dtype)).sum(dim=-1) / math.sqrt(self.tables)
            out = out + ste * corr.detach()
        return out

    def chamber_stats(self, x_i: Tensor, x_j: Tensor) -> dict[str, float]:
        with torch.no_grad():
            idx, _, _, _ = self.hash(torch.cat([x_i, x_j], dim=-1))
            return _index_stats(idx, self.cells)


class AdditiveScoreLUT(nn.Module):
    def __init__(self, dim: int, tables: int, comparisons: int, *, seed: int) -> None:
        super().__init__()
        self.q_hash = PairwiseHash(dim, tables, comparisons, seed=seed)
        self.k_hash = PairwiseHash(dim, tables, comparisons, seed=seed + 1009)
        self.tables = tables
        self.cells = 1 << comparisons
        self.q_score = nn.Parameter(torch.randn(tables, self.cells) * 0.02)
        self.k_score = nn.Parameter(torch.randn(tables, self.cells) * 0.02)

    def _lookup(self, q_idx: Tensor, k_idx: Tensor) -> Tensor:
        fq = q_idx.reshape(-1, self.tables)
        fk = k_idx.reshape(-1, self.tables)
        vals = []
        for table_idx in range(self.tables):
            vals.append(self.q_score[table_idx, fq[:, table_idx]] + self.k_score[table_idx, fk[:, table_idx]])
        return (torch.stack(vals, dim=-1).sum(dim=-1) / math.sqrt(self.tables)).view(q_idx.shape[:-1])

    def forward(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        q_idx, _, q_r, q_u = self.q_hash(x_i)
        k_idx, _, k_r, k_u = self.k_hash(x_j)
        out = self._lookup(q_idx, k_idx)
        if self.training:
            q_neigh = q_idx ^ (2**q_r).long()
            k_neigh = k_idx ^ (2**k_r).long()
            q_corr = self._lookup(q_neigh, k_idx) - out
            k_corr = self._lookup(q_idx, k_neigh) - out
            q_ste = (ste_heaviside(q_u, "fast_sigmoid_odd") - (q_u > 0).to(q_u.dtype)).sum(dim=-1) / math.sqrt(self.tables)
            k_ste = (ste_heaviside(k_u, "fast_sigmoid_odd") - (k_u > 0).to(k_u.dtype)).sum(dim=-1) / math.sqrt(self.tables)
            out = out + q_ste * q_corr.detach() + k_ste * k_corr.detach()
        return out

    def chamber_stats(self, x_i: Tensor, x_j: Tensor) -> dict[str, float]:
        with torch.no_grad():
            q_idx, _, _, _ = self.q_hash(x_i)
            k_idx, _, _, _ = self.k_hash(x_j)
            joint = q_idx * self.cells + k_idx
            return _index_stats(joint, self.cells * self.cells)


class RelationLookupMixin:
    tables: int
    cells: int
    score: nn.Parameter

    def _lookup(self, idx: Tensor) -> Tensor:
        flat = idx.reshape(-1, self.tables)
        vals = []
        table = self.score.to(device=idx.device)
        for table_idx in range(self.tables):
            vals.append(table[table_idx, flat[:, table_idx]])
        return (torch.stack(vals, dim=-1).sum(dim=-1) / math.sqrt(self.tables)).view(idx.shape[:-1])

    def _lookup_with_minbit_ste(self, idx: Tensor, r_min: Tensor, u_min: Tensor) -> Tensor:
        out = self._lookup(idx)
        if self.training:
            neigh = idx ^ (2**r_min).long()
            corr = self._lookup(neigh) - out
            ste = (ste_heaviside(u_min, "fast_sigmoid_odd") - (u_min > 0).to(u_min.dtype)).sum(dim=-1) / math.sqrt(self.tables)
            out = out + ste * corr.detach()
        return out


class CrossRelationScoreLUT(nn.Module, RelationLookupMixin):
    def __init__(self, dim: int, tables: int, comparisons: int, *, seed: int) -> None:
        super().__init__()
        self.hash = CrossRelationHash(dim, tables, comparisons, seed=seed)
        self.tables = tables
        self.cells = 1 << comparisons
        self.score = nn.Parameter(torch.randn(tables, self.cells) * 0.02)

    def forward(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        idx, _, r_min, u_min = self.hash(x_i, x_j)
        return self._lookup_with_minbit_ste(idx, r_min, u_min)

    def chamber_stats(self, x_i: Tensor, x_j: Tensor) -> dict[str, float]:
        with torch.no_grad():
            idx, _, _, _ = self.hash(x_i, x_j)
            return _index_stats(idx, self.cells)


class DifferenceScoreLUT(nn.Module, RelationLookupMixin):
    def __init__(
        self,
        dim: int,
        tables: int,
        comparisons: int,
        *,
        seed: int,
        anchors: Tensor | None = None,
        thresholds: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.hash = PairwiseHash(dim, tables, comparisons, seed=seed)
        if anchors is not None:
            expected = (tables, comparisons, 2)
            if tuple(anchors.shape) != expected:
                raise ValueError(f"anchors must have shape {expected}, got {tuple(anchors.shape)}")
            with torch.no_grad():
                self.hash.anchors.copy_(anchors.to(dtype=torch.long, device=self.hash.anchors.device))
        if thresholds is not None:
            expected_t = (tables, comparisons)
            if tuple(thresholds.shape) != expected_t:
                raise ValueError(f"thresholds must have shape {expected_t}, got {tuple(thresholds.shape)}")
            with torch.no_grad():
                self.hash.thresholds.copy_(thresholds.to(dtype=self.hash.thresholds.dtype, device=self.hash.thresholds.device))
        self.tables = tables
        self.cells = 1 << comparisons
        self.score = nn.Parameter(torch.randn(tables, self.cells) * 0.02)

    def forward(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        idx, _, r_min, u_min = self.hash(x_i - x_j)
        return self._lookup_with_minbit_ste(idx, r_min, u_min)

    def chamber_stats(self, x_i: Tensor, x_j: Tensor) -> dict[str, float]:
        with torch.no_grad():
            idx, _, _, _ = self.hash(x_i - x_j)
            return _index_stats(idx, self.cells)


class SameCoordinateScoreLUT(nn.Module, RelationLookupMixin):
    def __init__(self, dim: int, tables: int, comparisons: int, *, seed: int, mode: str) -> None:
        super().__init__()
        self.hash = SameCoordinateRelationHash(dim, tables, comparisons, seed=seed, mode=mode)
        self.tables = tables
        self.cells = 1 << comparisons
        self.score = nn.Parameter(torch.randn(tables, self.cells) * 0.02)

    def forward(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        idx, _, r_min, u_min = self.hash(x_i, x_j)
        return self._lookup_with_minbit_ste(idx, r_min, u_min)

    def chamber_stats(self, x_i: Tensor, x_j: Tensor) -> dict[str, float]:
        with torch.no_grad():
            idx, _, _, _ = self.hash(x_i, x_j)
            return _index_stats(idx, self.cells)


class AbsDiffScoreLUT(nn.Module):
    def __init__(self, dim: int, tables: int, comparisons: int, *, seed: int) -> None:
        super().__init__()
        self.layer = AbsDiffLUT(dim, 1, tables=tables, comparisons=comparisons, seed=seed)

    def forward(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        return self.layer(x_i, x_j).squeeze(-1)

    def chamber_stats(self, x_i: Tensor, x_j: Tensor) -> dict[str, float]:
        with torch.no_grad():
            _ = self.layer(x_i, x_j)
            indices = self.layer._last_indices
            if indices is None:
                raise RuntimeError("AbsDiffLUT did not cache route indices")
            return _index_stats(indices, self.layer.table_size)


class RelationWalshScore(nn.Module):
    """Low-order score over cross relation bits.

    This is a structured table on the relation chamber: first-order bit weights
    plus second-order bit interactions.  Neighboring chambers share parameters
    through the Walsh basis instead of learning unrelated free table entries.
    """

    def __init__(self, dim: int, tables: int, comparisons: int, *, seed: int) -> None:
        super().__init__()
        self.hash = CrossRelationHash(dim, tables, comparisons, seed=seed)
        self.tables = tables
        self.comparisons = comparisons
        pairs = list(combinations(range(comparisons), 2))
        self.register_buffer("pair_i", torch.tensor([p[0] for p in pairs], dtype=torch.long))
        self.register_buffer("pair_j", torch.tensor([p[1] for p in pairs], dtype=torch.long))
        self.bias = nn.Parameter(torch.zeros(tables))
        self.linear = nn.Parameter(torch.randn(tables, comparisons) * 0.02)
        self.quadratic = nn.Parameter(torch.randn(tables, len(pairs)) * 0.02)

    def forward(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        margins = self.hash.margins(x_i, x_j)
        hard = (margins > 0).to(margins.dtype)
        bits = hard
        if self.training:
            soft = ste_heaviside(margins, "fast_sigmoid_odd")
            bits = hard + soft - soft.detach()
        z = bits * 2.0 - 1.0
        score = self.bias.to(device=z.device, dtype=z.dtype).view(*([1] * (z.ndim - 2)), self.tables)
        score = score + (z * self.linear.to(device=z.device, dtype=z.dtype)).sum(dim=-1)
        if self.pair_i.numel() > 0:
            zi = z.index_select(-1, self.pair_i.to(device=z.device))
            zj = z.index_select(-1, self.pair_j.to(device=z.device))
            score = score + (zi * zj * self.quadratic.to(device=z.device, dtype=z.dtype)).sum(dim=-1)
        return score.sum(dim=-1) / math.sqrt(self.tables)

    def chamber_stats(self, x_i: Tensor, x_j: Tensor) -> dict[str, float]:
        with torch.no_grad():
            idx, _, _, _ = self.hash(x_i, x_j)
            return _index_stats(idx, 1 << self.comparisons)


class BinaryCodeSimilarity(nn.Module):
    """Learn binary q/k codes, then score agreement with XNOR-like features."""

    def __init__(self, dim: int, tables: int, comparisons: int, *, seed: int) -> None:
        super().__init__()
        self.q_hash = PairwiseHash(dim, tables, comparisons, seed=seed)
        self.k_hash = PairwiseHash(dim, tables, comparisons, seed=seed + 1009)
        self.tables = tables
        self.comparisons = comparisons
        self.weight = nn.Parameter(torch.ones(tables, comparisons) / math.sqrt(tables * comparisons))
        self.bias = nn.Parameter(torch.zeros(()))

    def _bits(self, hash_module: PairwiseHash, x: Tensor) -> Tensor:
        _, margins, _, _ = hash_module(x)
        hard = (margins > 0).to(margins.dtype)
        if self.training:
            soft = ste_heaviside(margins, "fast_sigmoid_odd")
            hard = hard + soft - soft.detach()
        return hard * 2.0 - 1.0

    def forward(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        q = self._bits(self.q_hash, x_i)
        k = self._bits(self.k_hash, x_j)
        agree = q * k
        return (agree * self.weight.to(device=agree.device, dtype=agree.dtype)).sum(dim=(-1, -2)) + self.bias.to(device=agree.device, dtype=agree.dtype)

    def chamber_stats(self, x_i: Tensor, x_j: Tensor) -> dict[str, float]:
        with torch.no_grad():
            q_idx, _, _, _ = self.q_hash(x_i)
            k_idx, _, _, _ = self.k_hash(x_j)
            joint = q_idx * (1 << self.comparisons) + k_idx
            return _index_stats(joint, (1 << self.comparisons) ** 2)


def build_scorer(
    family: ScoreFamily,
    cfg: ToyConfig,
    *,
    anchors: Tensor | None = None,
    thresholds: Tensor | None = None,
) -> nn.Module:
    if family == "dense_qk":
        return DenseQKScore(cfg.dim, cfg.rank)
    if family == "pairwise_score":
        return PairwiseScoreLUT(cfg.dim, cfg.tables, cfg.comparisons, seed=cfg.seed)
    if family == "joint_concat":
        return JointConcatScoreLUT(cfg.dim, cfg.tables, cfg.comparisons, seed=cfg.seed)
    if family == "additive":
        return AdditiveScoreLUT(cfg.dim, cfg.tables, cfg.comparisons, seed=cfg.seed)
    if family == "cross_lut":
        return CrossRelationScoreLUT(cfg.dim, cfg.tables, cfg.comparisons, seed=cfg.seed)
    if family == "diff_lut":
        return DifferenceScoreLUT(cfg.dim, cfg.tables, cfg.comparisons, seed=cfg.seed)
    if family == "aligned_diff_lut":
        return DifferenceScoreLUT(cfg.dim, cfg.tables, cfg.comparisons, seed=cfg.seed, anchors=anchors, thresholds=thresholds)
    if family == "same_coord_lut":
        return SameCoordinateScoreLUT(cfg.dim, cfg.tables, cfg.comparisons, seed=cfg.seed, mode="forward")
    if family == "same_coord_bidir_lut":
        return SameCoordinateScoreLUT(cfg.dim, cfg.tables, cfg.comparisons, seed=cfg.seed, mode="bidir")
    if family == "absdiff_lut":
        return AbsDiffScoreLUT(cfg.dim, cfg.tables, cfg.comparisons, seed=cfg.seed)
    if family == "relation_walsh":
        return RelationWalshScore(cfg.dim, cfg.tables, cfg.comparisons, seed=cfg.seed)
    if family == "binary_code":
        return BinaryCodeSimilarity(cfg.dim, cfg.tables, cfg.comparisons, seed=cfg.seed)
    raise ValueError(f"unknown scorer family {family}")


def _index_stats(indices: Tensor, cells: int) -> dict[str, float]:
    flat = indices.reshape(-1, indices.shape[-1])
    active = []
    entropy = []
    for table_idx in range(flat.shape[-1]):
        counts = torch.bincount(flat[:, table_idx], minlength=cells).float()
        probs = counts / counts.sum().clamp_min(1)
        nz = probs > 0
        ent = -(probs[nz] * probs[nz].log()).sum() / math.log(cells)
        active.append(float(nz.float().sum().item()))
        entropy.append(float(ent.item()))
    return {
        "chamber_active_mean": float(sum(active) / len(active)),
        "chamber_entropy_mean": float(sum(entropy) / len(entropy)),
        "chamber_capacity": float(cells),
        "K_eff_active": float(sum(active) / len(active)),
    }


def _iter_batches(n_items: int, batch_size: int, device: torch.device) -> Tensor:
    return torch.randint(0, n_items, (batch_size,), device=device)


def train_compatibility(
    family: ScoreFamily,
    cfg: ToyConfig,
    train_data: tuple[Tensor, Tensor, Tensor],
    test_data: tuple[Tensor, Tensor, Tensor],
    ood_data: tuple[Tensor, Tensor, Tensor],
    device: torch.device,
) -> dict[str, float | str]:
    model = build_scorer(family, cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    x_i, x_j, y = train_data
    start = time.time()
    for _ in range(cfg.steps):
        idx = _iter_batches(x_i.shape[0], cfg.batch_size, device)
        logits = model(x_i[idx], x_j[idx])
        loss = F.binary_cross_entropy_with_logits(logits, y[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    elapsed = time.time() - start
    model.eval()
    with torch.no_grad():
        test_logits = model(test_data[0], test_data[1])
        ood_logits = model(ood_data[0], ood_data[1])
        out: dict[str, float | str] = {
            "toy": "compatibility",
            "family": family,
            "train_seconds": elapsed,
            "params": float(sum(p.numel() for p in model.parameters())),
            "test_loss": float(F.binary_cross_entropy_with_logits(test_logits, test_data[2]).item()),
            "ood_loss": float(F.binary_cross_entropy_with_logits(ood_logits, ood_data[2]).item()),
        }
        out.update({f"test_{k}": v for k, v in _binary_metrics(test_logits, test_data[2]).items()})
        out.update({f"ood_{k}": v for k, v in _binary_metrics(ood_logits, ood_data[2]).items()})
        if hasattr(model, "chamber_stats"):
            out.update(model.chamber_stats(test_data[0], test_data[1]))  # type: ignore[attr-defined]
    return out


class PairwiseValuePayload(nn.Module):
    def __init__(self, dim: int, value_classes: int, cfg: ToyConfig) -> None:
        super().__init__()
        self.layer = PairwiseLinear(
            dim * 2,
            value_classes,
            tables=cfg.tables,
            comparisons=cfg.comparisons,
            seed=cfg.seed + 2027,
            lut_init_std=0.0,
        )

    def forward(self, query: Tensor, cand: Tensor) -> Tensor:
        bsz, n_cand, dim = cand.shape
        q = query[:, None, :].expand(-1, n_cand, -1)
        x = torch.cat([q, cand], dim=-1)
        return self.layer(x)


def _smooth_relu_weights(scores: Tensor, tau: float) -> Tensor:
    scale = max(tau, 1e-6)
    weights = torch.where(scores <= 0, 1.0 / (1.0 - scores / scale), 1.0 + scores / scale)
    return weights.clamp_min(1e-8) / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def _candidate_payload_probs(
    value_model: nn.Module | None,
    query: Tensor,
    cand: Tensor,
    values: Tensor,
    payload: ValuePayload,
) -> Tensor:
    if payload == "key":
        return values
    if value_model is None:
        raise ValueError(f"value_model is required for payload={payload}")
    logits = value_model(query, cand)
    if payload == "pairwise_delta":
        logits = logits + values.clamp_min(1e-8).log()
    return logits.softmax(dim=-1)


def _aggregation_logits(
    scorer: nn.Module,
    value_model: nn.Module | None,
    query: Tensor,
    cand: Tensor,
    values: Tensor,
    tau: float,
    aggregation_mode: AggregationMode,
    value_payload: ValuePayload,
) -> tuple[Tensor, Tensor, Tensor]:
    bsz, n_cand, dim = cand.shape
    q_flat = query[:, None, :].expand(-1, n_cand, -1).reshape(bsz * n_cand, dim)
    c_flat = cand.reshape(bsz * n_cand, dim)
    scores = scorer(q_flat, c_flat).view(bsz, n_cand)
    payload_probs = _candidate_payload_probs(value_model, query, cand, values, value_payload).clamp_min(1e-8)
    if aggregation_mode == "softmax":
        weights = (scores / tau).softmax(dim=-1)
        label_logits = (weights.unsqueeze(-1) * payload_probs).sum(dim=1).clamp_min(1e-8).log()
    elif aggregation_mode == "smooth_relu":
        weights = _smooth_relu_weights(scores, tau)
        label_logits = (weights.unsqueeze(-1) * payload_probs).sum(dim=1).clamp_min(1e-8).log()
    elif aggregation_mode == "hard_top1":
        soft = (scores / tau).softmax(dim=-1)
        hard = F.one_hot(scores.argmax(dim=-1), num_classes=n_cand).to(scores.dtype)
        weights = hard if not scorer.training else hard + soft - soft.detach()
        label_logits = (weights.unsqueeze(-1) * payload_probs).sum(dim=1).clamp_min(1e-8).log()
    elif aggregation_mode == "logsumexp":
        label_logits = torch.logsumexp(scores.unsqueeze(-1) / tau + payload_probs.log(), dim=1)
    else:
        raise ValueError(f"unknown aggregation mode {aggregation_mode}")
    probs = label_logits.softmax(dim=-1)
    return label_logits, probs, scores


def _score_margin_loss(scores: Tensor, values: Tensor, target: Tensor, margin: float) -> Tensor:
    value_ids = values.argmax(dim=-1)
    is_pos = value_ids == target[:, None]
    pos = scores.masked_fill(~is_pos, -torch.inf).max(dim=-1).values
    neg = scores.masked_fill(is_pos, -torch.inf).max(dim=-1).values
    valid = torch.isfinite(pos) & torch.isfinite(neg)
    if not valid.any():
        return scores.new_zeros(())
    return F.relu(margin - pos[valid] + neg[valid]).mean()


def train_aggregation(
    family: ScoreFamily,
    cfg: ToyConfig,
    train_data: tuple[Tensor, Tensor, Tensor, Tensor],
    test_data: tuple[Tensor, Tensor, Tensor, Tensor],
    ood_data: tuple[Tensor, Tensor, Tensor, Tensor],
    device: torch.device,
) -> dict[str, float | str]:
    scorer = build_scorer(family, cfg).to(device)
    value_model: nn.Module | None = None
    if cfg.value_payload != "key":
        value_model = PairwiseValuePayload(cfg.dim, cfg.value_classes, cfg).to(device)
    params = list(scorer.parameters())
    if value_model is not None:
        params += list(value_model.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr)
    query, cand, values, target = train_data
    tau = 0.5
    start = time.time()
    for _ in range(cfg.steps):
        idx = _iter_batches(query.shape[0], cfg.batch_size, device)
        label_logits, _, scores = _aggregation_logits(
            scorer,
            value_model,
            query[idx],
            cand[idx],
            values[idx],
            tau,
            cfg.aggregation_mode,
            cfg.value_payload,
        )
        loss = F.cross_entropy(label_logits, target[idx])
        if cfg.margin_lambda > 0:
            loss = loss + cfg.margin_lambda * _score_margin_loss(scores, values[idx], target[idx], cfg.margin)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    elapsed = time.time() - start

    def eval_split(data: tuple[Tensor, Tensor, Tensor, Tensor], prefix: str) -> dict[str, float]:
        scorer.eval()
        if value_model is not None:
            value_model.eval()
        with torch.no_grad():
            label_logits, probs, scores = _aggregation_logits(
                scorer,
                value_model,
                data[0],
                data[1],
                data[2],
                tau,
                cfg.aggregation_mode,
                cfg.value_payload,
            )
            pred = probs.argmax(dim=-1)
            score_winner_value = data[2].argmax(dim=-1).gather(1, scores.argmax(dim=-1, keepdim=True)).squeeze(1)
            target = data[3]
            top2 = scores.topk(2, dim=-1).values
            return {
                f"{prefix}_loss": float(F.cross_entropy(label_logits, target).item()),
                f"{prefix}_value_acc": float((pred == target).float().mean().item()),
                f"{prefix}_score_top1_value_acc": float((score_winner_value == target).float().mean().item()),
                f"{prefix}_score_margin": float((top2[:, 0] - top2[:, 1]).mean().item()),
            }

    out: dict[str, float | str] = {
        "toy": "selective_aggregation",
        "family": family,
        "aggregation_mode": cfg.aggregation_mode,
        "value_payload": cfg.value_payload,
        "margin_lambda": cfg.margin_lambda,
        "margin": cfg.margin,
        "train_seconds": elapsed,
        "params": float(sum(p.numel() for p in params)),
    }
    out.update(eval_split(test_data, "test"))
    out.update(eval_split(ood_data, "ood"))
    if hasattr(scorer, "chamber_stats"):
        q = test_data[0][:, None, :].expand(-1, test_data[1].shape[1], -1).reshape(-1, cfg.dim)
        c = test_data[1].reshape(-1, cfg.dim)
        out.update(scorer.chamber_stats(q, c))  # type: ignore[attr-defined]
    return out


def _score_matrix(scorer: nn.Module, query: Tensor, cand: Tensor) -> Tensor:
    bsz, n_cand, dim = cand.shape
    q_flat = query[:, None, :].expand(-1, n_cand, -1).reshape(bsz * n_cand, dim)
    c_flat = cand.reshape(bsz * n_cand, dim)
    return scorer(q_flat, c_flat).view(bsz, n_cand)


def _select_teacher_aligned_diff_anchors(
    teacher: nn.Module,
    cfg: ToyConfig,
    train_data: tuple[Tensor, Tensor, Tensor, Tensor],
    device: torch.device,
) -> tuple[Tensor, Tensor, dict[str, float]]:
    query, cand, _, _ = train_data
    d_chunks = []
    score_chunks = []
    teacher.eval()
    with torch.no_grad():
        for _ in range(cfg.anchor_sample_batches):
            idx = _iter_batches(query.shape[0], cfg.batch_size, device)
            scores = _score_matrix(teacher, query[idx], cand[idx])
            centered_scores = scores - scores.mean(dim=-1, keepdim=True)
            d = query[idx, None, :].expand_as(cand[idx]) - cand[idx]
            d_chunks.append(d.reshape(-1, cfg.dim))
            score_chunks.append(centered_scores.reshape(-1))
    d_all = torch.cat(d_chunks, dim=0)
    y = torch.cat(score_chunks, dim=0)
    y = (y - y.mean()) / y.std().clamp_min(1e-6)

    gen = torch.Generator(device="cpu").manual_seed(cfg.seed + 7919)
    pair_count = max(cfg.anchor_candidate_pairs, cfg.tables * cfg.comparisons)
    a = torch.randint(0, cfg.dim, (pair_count,), generator=gen).to(device)
    b = torch.randint(0, cfg.dim, (pair_count,), generator=gen).to(device)
    same = a == b
    while same.any():
        b[same] = torch.randint(0, cfg.dim, (int(same.sum().item()),), generator=gen).to(device)
        same = a == b

    selected = cfg.tables * cfg.comparisons
    scores = []
    chunk_size = 256
    for start in range(0, pair_count, chunk_size):
        end = min(start + chunk_size, pair_count)
        margins = d_all[:, a[start:end]] - d_all[:, b[start:end]]
        bits = (margins > margins.median(dim=0).values.view(1, -1)).to(d_all.dtype) * 2.0 - 1.0
        corr = (bits * y.view(-1, 1)).mean(dim=0).abs()
        scores.append(corr)
    pair_scores = torch.cat(scores, dim=0)
    order = pair_scores.argsort(descending=True)

    chosen: list[int] = []
    seen: set[tuple[int, int]] = set()
    for idx_t in order.tolist():
        pa = int(a[idx_t].item())
        pb = int(b[idx_t].item())
        key = (pa, pb)
        if key in seen:
            continue
        chosen.append(idx_t)
        seen.add(key)
        if len(chosen) == selected:
            break
    chosen_t = torch.tensor(chosen, dtype=torch.long, device=device)
    anchors = torch.stack([a[chosen_t], b[chosen_t]], dim=-1).view(cfg.tables, cfg.comparisons, 2)
    chosen_margins = d_all[:, anchors.reshape(-1, 2)[:, 0]] - d_all[:, anchors.reshape(-1, 2)[:, 1]]
    thresholds = chosen_margins.median(dim=0).values.view(cfg.tables, cfg.comparisons)
    stats = {
        "aligned_anchor_mean_abs_corr": float(pair_scores[chosen_t].mean().item()),
        "aligned_anchor_max_abs_corr": float(pair_scores[chosen_t].max().item()),
        "aligned_anchor_min_abs_corr": float(pair_scores[chosen_t].min().item()),
        "aligned_anchor_sample_tokens": float(d_all.shape[0]),
        "aligned_anchor_candidate_pairs": float(pair_count),
    }
    return anchors.detach().cpu(), thresholds.detach().cpu(), stats


def train_score_distillation(
    student_family: ScoreFamily,
    cfg: ToyConfig,
    train_data: tuple[Tensor, Tensor, Tensor, Tensor],
    test_data: tuple[Tensor, Tensor, Tensor, Tensor],
    ood_data: tuple[Tensor, Tensor, Tensor, Tensor],
    device: torch.device,
) -> list[dict[str, float | str]]:
    tau = 0.5
    teacher = DenseQKScore(cfg.dim, cfg.rank).to(device)
    teacher_opt = torch.optim.AdamW(teacher.parameters(), lr=cfg.lr)
    query, cand, values, target = train_data

    teacher_start = time.time()
    for _ in range(cfg.teacher_steps):
        idx = _iter_batches(query.shape[0], cfg.batch_size, device)
        label_logits, _, _ = _aggregation_logits(
            teacher,
            None,
            query[idx],
            cand[idx],
            values[idx],
            tau,
            "softmax",
            "key",
        )
        loss = F.cross_entropy(label_logits, target[idx])
        teacher_opt.zero_grad(set_to_none=True)
        loss.backward()
        teacher_opt.step()
    teacher_elapsed = time.time() - teacher_start
    teacher.eval()

    anchor_stats: dict[str, float] = {}
    anchors: Tensor | None = None
    thresholds: Tensor | None = None
    if student_family == "aligned_diff_lut":
        anchors, thresholds, anchor_stats = _select_teacher_aligned_diff_anchors(teacher, cfg, train_data, device)
    student = build_scorer(student_family, cfg, anchors=anchors, thresholds=thresholds).to(device)
    student_opt = torch.optim.AdamW(student.parameters(), lr=cfg.lr)
    distill_temp = cfg.distill_temperature
    student_start = time.time()
    for _ in range(cfg.steps):
        idx = _iter_batches(query.shape[0], cfg.batch_size, device)
        q = query[idx]
        c = cand[idx]
        v = values[idx]
        y = target[idx]
        with torch.no_grad():
            teacher_scores = _score_matrix(teacher, q, c)
            teacher_probs = (teacher_scores / distill_temp).softmax(dim=-1)
        student_scores = _score_matrix(student, q, c)
        student_log_probs = (student_scores / distill_temp).log_softmax(dim=-1)
        distill_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (distill_temp**2)
        label_logits, _, _ = _aggregation_logits(student, None, q, c, v, tau, "softmax", "key")
        ce_loss = F.cross_entropy(label_logits, y)
        loss = cfg.distill_alpha * distill_loss + (1.0 - cfg.distill_alpha) * ce_loss
        student_opt.zero_grad(set_to_none=True)
        loss.backward()
        student_opt.step()
    student_elapsed = time.time() - student_start

    def eval_scorer(scorer: nn.Module, data: tuple[Tensor, Tensor, Tensor, Tensor], prefix: str) -> dict[str, float]:
        scorer.eval()
        with torch.no_grad():
            label_logits, probs, scores = _aggregation_logits(scorer, None, data[0], data[1], data[2], tau, "softmax", "key")
            pred = probs.argmax(dim=-1)
            value_ids = data[2].argmax(dim=-1)
            score_winner_value = value_ids.gather(1, scores.argmax(dim=-1, keepdim=True)).squeeze(1)
            target = data[3]
            top2 = scores.topk(2, dim=-1).values
            return {
                f"{prefix}_loss": float(F.cross_entropy(label_logits, target).item()),
                f"{prefix}_value_acc": float((pred == target).float().mean().item()),
                f"{prefix}_score_top1_value_acc": float((score_winner_value == target).float().mean().item()),
                f"{prefix}_score_margin": float((top2[:, 0] - top2[:, 1]).mean().item()),
            }

    def eval_distill(data: tuple[Tensor, Tensor, Tensor, Tensor], prefix: str) -> dict[str, float]:
        teacher.eval()
        student.eval()
        with torch.no_grad():
            teacher_scores = _score_matrix(teacher, data[0], data[1])
            student_scores = _score_matrix(student, data[0], data[1])
            teacher_probs = (teacher_scores / distill_temp).softmax(dim=-1)
            student_log_probs = (student_scores / distill_temp).log_softmax(dim=-1)
            kl = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (distill_temp**2)
            corr = F.cosine_similarity(
                student_scores - student_scores.mean(dim=-1, keepdim=True),
                teacher_scores - teacher_scores.mean(dim=-1, keepdim=True),
                dim=-1,
            ).mean()
            agree = (student_scores.argmax(dim=-1) == teacher_scores.argmax(dim=-1)).float().mean()
            return {
                f"{prefix}_teacher_student_kl": float(kl.item()),
                f"{prefix}_teacher_student_score_cosine": float(corr.item()),
                f"{prefix}_teacher_student_top1_agree": float(agree.item()),
            }

    teacher_row: dict[str, float | str] = {
        "toy": "score_distillation",
        "family": "dense_qk_teacher",
        "student_family": student_family,
        "aggregation_mode": "softmax",
        "value_payload": "key",
        "distill_alpha": cfg.distill_alpha,
        "distill_temperature": cfg.distill_temperature,
        "teacher_steps": float(cfg.teacher_steps),
        "train_seconds": teacher_elapsed,
        "params": float(sum(p.numel() for p in teacher.parameters())),
    }
    teacher_row.update(eval_scorer(teacher, test_data, "test"))
    teacher_row.update(eval_scorer(teacher, ood_data, "ood"))

    student_row: dict[str, float | str] = {
        "toy": "score_distillation",
        "family": f"{student_family}_distilled",
        "student_family": student_family,
        "aggregation_mode": "softmax",
        "value_payload": "key",
        "distill_alpha": cfg.distill_alpha,
        "distill_temperature": cfg.distill_temperature,
        "teacher_steps": float(cfg.teacher_steps),
        "train_seconds": student_elapsed,
        "params": float(sum(p.numel() for p in student.parameters())),
    }
    student_row.update(eval_scorer(student, test_data, "test"))
    student_row.update(eval_scorer(student, ood_data, "ood"))
    student_row.update(eval_distill(test_data, "test"))
    student_row.update(eval_distill(ood_data, "ood"))
    student_row.update(anchor_stats)
    if hasattr(student, "chamber_stats"):
        q = test_data[0][:, None, :].expand(-1, test_data[1].shape[1], -1).reshape(-1, cfg.dim)
        c = test_data[1].reshape(-1, cfg.dim)
        student_row.update(student.chamber_stats(q, c))  # type: ignore[attr-defined]
    return [teacher_row, student_row]


class DenseValue(nn.Module):
    def __init__(self, dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


class PairwiseValue(nn.Module):
    def __init__(self, dim: int, out_dim: int, cfg: ToyConfig) -> None:
        super().__init__()
        self.layer = PairwiseLinear(dim, out_dim, tables=cfg.tables, comparisons=cfg.comparisons, seed=cfg.seed, lut_init_std=0.0)

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x.unsqueeze(1)).squeeze(1)


def train_value_transform(
    family: Literal["dense_wv", "lut_value"],
    cfg: ToyConfig,
    train_data: tuple[Tensor, Tensor],
    test_data: tuple[Tensor, Tensor],
    ood_data: tuple[Tensor, Tensor],
    device: torch.device,
) -> dict[str, float | str]:
    model: nn.Module = DenseValue(cfg.dim, cfg.out_dim) if family == "dense_wv" else PairwiseValue(cfg.dim, cfg.out_dim, cfg)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    x, y = train_data
    start = time.time()
    for _ in range(cfg.steps):
        idx = _iter_batches(x.shape[0], cfg.batch_size, device)
        pred = model(x[idx])
        loss = F.mse_loss(pred, y[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    elapsed = time.time() - start

    def eval_split(data: tuple[Tensor, Tensor], prefix: str) -> dict[str, float]:
        model.eval()
        with torch.no_grad():
            pred = model(data[0])
            mse = F.mse_loss(pred, data[1])
            cos = F.cosine_similarity(pred, data[1], dim=-1).mean()
            rel = (pred - data[1]).square().sum().sqrt() / data[1].square().sum().sqrt().clamp_min(1e-8)
            return {f"{prefix}_mse": float(mse.item()), f"{prefix}_cosine": float(cos.item()), f"{prefix}_rel_error": float(rel.item())}

    out: dict[str, float | str] = {
        "toy": "value_transform",
        "family": family,
        "train_seconds": elapsed,
        "params": float(sum(p.numel() for p in model.parameters())),
    }
    out.update(eval_split(test_data, "test"))
    out.update(eval_split(ood_data, "ood"))
    return out


def run_all(cfg: ToyConfig) -> list[dict[str, float | str]]:
    _set_seed(cfg.seed)
    dev = _device(cfg.device)
    train_proto = F.normalize(torch.randn(cfg.train_classes, cfg.dim, device=dev), dim=-1)
    ood_proto = F.normalize(torch.randn(cfg.ood_classes, cfg.dim, device=dev), dim=-1)
    teacher = torch.randn(cfg.dim, cfg.out_dim, device=dev) / math.sqrt(cfg.dim)

    pair_train = make_pair_dataset(train_proto, cfg.train_pairs, noise=cfg.noise, balanced=True, device=dev)
    pair_test = make_pair_dataset(train_proto, cfg.test_pairs, noise=cfg.noise, balanced=True, device=dev)
    pair_ood = make_pair_dataset(ood_proto, cfg.test_pairs, noise=cfg.noise, balanced=True, device=dev)

    agg_train = make_aggregation_dataset(
        train_proto, cfg.seq_train, candidates=cfg.candidates, value_classes=cfg.value_classes, noise=cfg.noise, device=dev
    )
    agg_test = make_aggregation_dataset(
        train_proto, cfg.seq_test, candidates=cfg.candidates, value_classes=cfg.value_classes, noise=cfg.noise, device=dev
    )
    agg_ood = make_aggregation_dataset(
        ood_proto, cfg.seq_test, candidates=cfg.candidates, value_classes=cfg.value_classes, noise=cfg.noise, device=dev
    )

    value_train = make_value_dataset(train_proto, teacher, cfg.value_train, noise=cfg.noise, device=dev)
    value_test = make_value_dataset(train_proto, teacher, cfg.value_test, noise=cfg.noise, device=dev)
    value_ood = make_value_dataset(ood_proto, teacher, cfg.value_test, noise=cfg.noise, device=dev)

    rows: list[dict[str, float | str]] = []
    score_families = (
        "dense_qk",
        "pairwise_score",
        "joint_concat",
        "additive",
        "cross_lut",
        "diff_lut",
        "relation_walsh",
        "binary_code",
    )
    for family in score_families:
        rows.append(train_compatibility(family, cfg, pair_train, pair_test, pair_ood, dev))  # type: ignore[arg-type]
    for family in score_families:
        rows.append(train_aggregation(family, cfg, agg_train, agg_test, agg_ood, dev))  # type: ignore[arg-type]
    for family in ("dense_wv", "lut_value"):
        rows.append(train_value_transform(family, cfg, value_train, value_test, value_ood, dev))  # type: ignore[arg-type]
    for row in rows:
        row.update(
            {
                "dim": float(cfg.dim),
                "tables": float(cfg.tables),
                "comparisons": float(cfg.comparisons),
                "rank": float(cfg.rank),
                "cells_per_table": float(1 << cfg.comparisons),
                "nominal_table_cells": float(cfg.tables * (1 << cfg.comparisons)),
            }
        )
    return rows


def run_aggregation_ablation(
    cfg: ToyConfig,
    *,
    score_families: list[ScoreFamily],
    aggregation_modes: list[AggregationMode],
    value_payloads: list[ValuePayload],
    margin_lambdas: list[float],
) -> list[dict[str, float | str]]:
    _set_seed(cfg.seed)
    dev = _device(cfg.device)
    train_proto = F.normalize(torch.randn(cfg.train_classes, cfg.dim, device=dev), dim=-1)
    ood_proto = F.normalize(torch.randn(cfg.ood_classes, cfg.dim, device=dev), dim=-1)
    agg_train = make_aggregation_dataset(
        train_proto, cfg.seq_train, candidates=cfg.candidates, value_classes=cfg.value_classes, noise=cfg.noise, device=dev
    )
    agg_test = make_aggregation_dataset(
        train_proto, cfg.seq_test, candidates=cfg.candidates, value_classes=cfg.value_classes, noise=cfg.noise, device=dev
    )
    agg_ood = make_aggregation_dataset(
        ood_proto, cfg.seq_test, candidates=cfg.candidates, value_classes=cfg.value_classes, noise=cfg.noise, device=dev
    )

    rows: list[dict[str, float | str]] = []
    for family in score_families:
        for aggregation_mode in aggregation_modes:
            for value_payload in value_payloads:
                for margin_lambda in margin_lambdas:
                    run_cfg = ToyConfig(**{**asdict(cfg), "aggregation_mode": aggregation_mode, "value_payload": value_payload, "margin_lambda": margin_lambda})
                    row = train_aggregation(family, run_cfg, agg_train, agg_test, agg_ood, dev)  # type: ignore[arg-type]
                    row.update(
                        {
                            "dim": float(run_cfg.dim),
                            "tables": float(run_cfg.tables),
                            "comparisons": float(run_cfg.comparisons),
                            "rank": float(run_cfg.rank),
                            "cells_per_table": float(1 << run_cfg.comparisons),
                            "nominal_table_cells": float(run_cfg.tables * (1 << run_cfg.comparisons)),
                        }
                    )
                    rows.append(row)
    return rows


def run_score_distillation(
    cfg: ToyConfig,
    *,
    student_families: list[ScoreFamily],
    distill_alphas: list[float],
    distill_temperatures: list[float],
) -> list[dict[str, float | str]]:
    _set_seed(cfg.seed)
    dev = _device(cfg.device)
    train_proto = F.normalize(torch.randn(cfg.train_classes, cfg.dim, device=dev), dim=-1)
    ood_proto = F.normalize(torch.randn(cfg.ood_classes, cfg.dim, device=dev), dim=-1)
    agg_train = make_aggregation_dataset(
        train_proto, cfg.seq_train, candidates=cfg.candidates, value_classes=cfg.value_classes, noise=cfg.noise, device=dev
    )
    agg_test = make_aggregation_dataset(
        train_proto, cfg.seq_test, candidates=cfg.candidates, value_classes=cfg.value_classes, noise=cfg.noise, device=dev
    )
    agg_ood = make_aggregation_dataset(
        ood_proto, cfg.seq_test, candidates=cfg.candidates, value_classes=cfg.value_classes, noise=cfg.noise, device=dev
    )

    rows: list[dict[str, float | str]] = []
    for student_family in student_families:
        for distill_alpha in distill_alphas:
            for distill_temperature in distill_temperatures:
                run_cfg = ToyConfig(
                    **{
                        **asdict(cfg),
                        "distill_alpha": distill_alpha,
                        "distill_temperature": distill_temperature,
                        "aggregation_mode": "softmax",
                        "value_payload": "key",
                    }
                )
                run_rows = train_score_distillation(student_family, run_cfg, agg_train, agg_test, agg_ood, dev)  # type: ignore[arg-type]
                for row in run_rows:
                    row.update(
                        {
                            "dim": float(run_cfg.dim),
                            "tables": float(run_cfg.tables),
                            "comparisons": float(run_cfg.comparisons),
                            "rank": float(run_cfg.rank),
                            "cells_per_table": float(1 << run_cfg.comparisons),
                            "nominal_table_cells": float(run_cfg.tables * (1 << run_cfg.comparisons)),
                        }
                    )
                rows.extend(run_rows)
    return rows


def _write_outputs(rows: list[dict[str, float | str]], cfg: ToyConfig, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with (output_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "summary.json").open("w") as handle:
        json.dump({"config": asdict(cfg), "rows": rows}, handle, indent=2)


def _parse_int_list(value: str | None, default: int) -> list[int]:
    if value is None or value.strip() == "":
        return [default]
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_float_list(value: str | None, default: float) -> list[float]:
    if value is None or value.strip() == "":
        return [default]
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _parse_str_list(value: str | None, default: str) -> list[str]:
    if value is None or value.strip() == "":
        return [default]
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, default=ToyConfig.dim)
    parser.add_argument("--tables", type=int, default=ToyConfig.tables)
    parser.add_argument("--comparisons", type=int, default=ToyConfig.comparisons)
    parser.add_argument("--rank", type=int, default=ToyConfig.rank)
    parser.add_argument("--steps", type=int, default=ToyConfig.steps)
    parser.add_argument("--batch-size", type=int, default=ToyConfig.batch_size)
    parser.add_argument("--lr", type=float, default=ToyConfig.lr)
    parser.add_argument("--seed", type=int, default=ToyConfig.seed)
    parser.add_argument("--device", default=ToyConfig.device)
    parser.add_argument("--output-dir", default=ToyConfig.output_dir)
    parser.add_argument("--tables-list", default=None, help="Comma-separated table counts for a compact scaling sweep.")
    parser.add_argument("--comparisons-list", default=None, help="Comma-separated comparison counts for a compact scaling sweep.")
    parser.add_argument("--aggregation-ablation", action="store_true", help="Run only selective aggregation with aggregation/value/margin axes.")
    parser.add_argument("--score-distill", action="store_true", help="Train dense QK teacher, then distill candidate score distribution into student score family.")
    parser.add_argument("--score-families", default="diff_lut", help="Comma-separated score families for --aggregation-ablation.")
    parser.add_argument("--aggregation-modes", default="softmax", help="Comma-separated aggregation modes: softmax,smooth_relu,logsumexp,hard_top1.")
    parser.add_argument("--value-payloads", default="key", help="Comma-separated value payloads: key,pairwise,pairwise_delta.")
    parser.add_argument("--margin-lambdas", default="0.0", help="Comma-separated score margin loss weights for --aggregation-ablation.")
    parser.add_argument("--margin", type=float, default=ToyConfig.margin)
    parser.add_argument("--teacher-steps", type=int, default=ToyConfig.teacher_steps)
    parser.add_argument("--distill-alphas", default=str(ToyConfig.distill_alpha), help="Comma-separated KL weights: 1.0 means pure teacher-score distillation.")
    parser.add_argument("--distill-temperatures", default=str(ToyConfig.distill_temperature), help="Comma-separated teacher/student score temperatures.")
    args = parser.parse_args()

    tables_list = _parse_int_list(args.tables_list, args.tables)
    comparisons_list = _parse_int_list(args.comparisons_list, args.comparisons)
    all_rows: list[dict[str, float | str]] = []
    root = Path(args.output_dir)
    for tables in tables_list:
        for comparisons in comparisons_list:
            cfg = ToyConfig(
                dim=args.dim,
                tables=tables,
                comparisons=comparisons,
                rank=args.rank,
                steps=args.steps,
                batch_size=args.batch_size,
                lr=args.lr,
                margin=args.margin,
                teacher_steps=args.teacher_steps,
                seed=args.seed,
                device=args.device,
                output_dir=str(root / f"T{tables}_L{comparisons}"),
            )
            if args.score_distill:
                rows = run_score_distillation(
                    cfg,
                    student_families=_parse_str_list(args.score_families, "diff_lut"),  # type: ignore[arg-type]
                    distill_alphas=_parse_float_list(args.distill_alphas, ToyConfig.distill_alpha),
                    distill_temperatures=_parse_float_list(args.distill_temperatures, ToyConfig.distill_temperature),
                )
            elif args.aggregation_ablation:
                rows = run_aggregation_ablation(
                    cfg,
                    score_families=_parse_str_list(args.score_families, "diff_lut"),  # type: ignore[arg-type]
                    aggregation_modes=_parse_str_list(args.aggregation_modes, "softmax"),  # type: ignore[arg-type]
                    value_payloads=_parse_str_list(args.value_payloads, "key"),  # type: ignore[arg-type]
                    margin_lambdas=_parse_float_list(args.margin_lambdas, 0.0),
                )
            else:
                rows = run_all(cfg)
            out = Path(cfg.output_dir)
            _write_outputs(rows, cfg, out)
            all_rows.extend(rows)
            for row in rows:
                print(json.dumps(row, sort_keys=True))
            print(f"wrote {out / 'summary.json'}")

    if len(tables_list) * len(comparisons_list) > 1:
        _write_outputs(all_rows, cfg, root)
        print(f"wrote combined sweep {root / 'summary.json'}")


if __name__ == "__main__":
    main()
