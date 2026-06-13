"""Fit a full-rank dense projection with pairwise LUTs.

This diagnostic tests the hypothesis that a pairwise-comparator LUT needs many
parameters to approximate even a simple dense full-rank channel mixer.

Teacher:
    y = x @ W.T

Student:
    PairwiseLinear(D, D), trained on random Gaussian x.

The decisive metric is normalized MSE:
    nmse = mse(student(x), teacher(x)) / mse(0, teacher(x))

For an orthogonal teacher and standard normal x, the denominator is close to 1.
Dense exact has nmse = 0 with D^2 parameters. If pairwise LUT requires many more
parameters to reach small nmse, the matched-parameter CE gap is plausibly caused
by inefficient simulation of full-rank mixing.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..layers import PairwiseLinear


@dataclass(frozen=True)
class FitRow:
    variant: str
    dim: int
    tables: int
    comparisons: int
    hashes: int
    rank: int
    mix_rounds: int
    local_block: int
    group_size: int
    sketch_scale: float
    fixed_zero_threshold: bool
    params: int
    dense_params: int
    param_ratio: float
    train_steps: int
    lr: float
    final_train_mse: float
    eval_mse: float
    eval_nmse: float
    eval_cos: float
    seconds: float


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _teacher_matrix(dim: int, *, seed: int, kind: str, teacher_rank: int, device: torch.device) -> Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    w = torch.randn(dim, dim, generator=gen)
    if kind == "orthogonal":
        q, r = torch.linalg.qr(w)
        signs = torch.sign(torch.diag(r))
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        w = q * signs.view(1, -1)
    elif kind == "gaussian":
        w = w / math.sqrt(dim)
    elif kind == "common_mode":
        v = torch.randn(dim, generator=gen)
        v = v / v.square().mean().sqrt().clamp_min(1e-12)
        w = torch.outer(v, torch.ones(dim) / dim)
    elif kind == "contrast_only":
        projector = torch.eye(dim) - torch.ones(dim, dim) / dim
        w = (w / math.sqrt(max(dim - 1, 1))) @ projector
    elif kind == "lowrank_residual":
        if teacher_rank < 1:
            raise ValueError("lowrank_residual teacher requires --teacher-rank >= 1")
        u = torch.randn(dim, teacher_rank, generator=gen) / math.sqrt(teacher_rank)
        v = torch.randn(teacher_rank, dim, generator=gen) / math.sqrt(dim)
        w = torch.eye(dim) + u @ v
    else:
        raise ValueError(f"unknown teacher kind: {kind}")
    return w.to(device)


def _sample(batch_size: int, dim: int, *, generator: torch.Generator, device: torch.device) -> Tensor:
    return torch.randn(batch_size, dim, generator=generator, device=device)


def _target(x: Tensor, w: Tensor) -> Tensor:
    return x @ w.T


def _cosine_mean(pred: Tensor, target: Tensor) -> float:
    pred_f = pred.float()
    target_f = target.float()
    return float(F.cosine_similarity(pred_f, target_f, dim=-1).mean().item())


def _make_pairwise(
    *,
    dim: int,
    tables: int,
    comparisons: int,
    fixed_zero_threshold: bool,
    seed: int,
) -> PairwiseLinear:
    return PairwiseLinear(
        dim,
        dim,
        tables=tables,
        comparisons=comparisons,
        backend="torch",
        seed=seed,
        fixed_zero_threshold=fixed_zero_threshold,
        use_output_scaling=True,
    )


class FixedExpanderMixing(nn.Module):
    """Fixed random perfect-matchings with orthogonal 2x2 channel mixing."""

    def __init__(self, dim: int, *, rounds: int, seed: int) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("FixedExpanderMixing requires an even dim")
        gen = torch.Generator(device="cpu").manual_seed(seed)
        left: list[Tensor] = []
        right: list[Tensor] = []
        for _ in range(rounds):
            perm = torch.randperm(dim, generator=gen)
            left.append(perm[: dim // 2])
            right.append(perm[dim // 2 :])
        self.register_buffer("left", torch.stack(left))
        self.register_buffer("right", torch.stack(right))
        self.scale = 1.0 / math.sqrt(2.0)

    def forward(self, x: Tensor) -> Tensor:
        y = x
        for left, right in zip(self.left, self.right):
            mixed = y.clone()
            a = y.index_select(-1, left)
            b = y.index_select(-1, right)
            mixed.index_copy_(-1, left, (a + b) * self.scale)
            mixed.index_copy_(-1, right, (a - b) * self.scale)
            y = mixed
        return y


class FixedButterflyMixing(nn.Module):
    """Fixed Walsh-Hadamard-style butterfly mixing."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim & (dim - 1):
            raise ValueError("FixedButterflyMixing requires power-of-two dim")
        self.dim = dim
        self.scale = 1.0 / math.sqrt(2.0)

    def forward(self, x: Tensor) -> Tensor:
        y = x
        stride = 1
        while stride < self.dim:
            blocks = y.reshape(*y.shape[:-1], -1, stride * 2)
            a = blocks[..., :stride]
            b = blocks[..., stride:]
            y = torch.cat(((a + b) * self.scale, (a - b) * self.scale), dim=-1).reshape_as(y)
            stride *= 2
        return y


class LearnedDiagonalButterflyMixing(nn.Module):
    """Hadamard-style butterfly with learnable diagonal scales."""

    def __init__(self, dim: int, *, rounds: int) -> None:
        super().__init__()
        if dim & (dim - 1):
            raise ValueError("LearnedDiagonalButterflyMixing requires power-of-two dim")
        self.dim = dim
        self.rounds = rounds
        self.diag = nn.Parameter(torch.ones(rounds, dim))
        self.scale = 1.0 / math.sqrt(2.0)

    def _butterfly(self, x: Tensor) -> Tensor:
        y = x
        stride = 1
        while stride < self.dim:
            blocks = y.reshape(*y.shape[:-1], -1, stride * 2)
            a = blocks[..., :stride]
            b = blocks[..., stride:]
            y = torch.cat(((a + b) * self.scale, (a - b) * self.scale), dim=-1).reshape_as(y)
            stride *= 2
        return y

    def forward(self, x: Tensor) -> Tensor:
        y = x
        for round_idx in range(self.rounds):
            y = self._butterfly(y * self.diag[round_idx])
        return y


class AdditiveLiftingMixing(nn.Module):
    """Learnable sparse lifting steps over random channel matchings."""

    def __init__(self, dim: int, *, rounds: int, seed: int) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("AdditiveLiftingMixing requires an even dim")
        gen = torch.Generator(device="cpu").manual_seed(seed)
        left: list[Tensor] = []
        right: list[Tensor] = []
        for _ in range(rounds):
            perm = torch.randperm(dim, generator=gen)
            left.append(perm[: dim // 2])
            right.append(perm[dim // 2 :])
        self.register_buffer("left", torch.stack(left))
        self.register_buffer("right", torch.stack(right))
        self.alpha = nn.Parameter(torch.zeros(rounds, dim // 2))
        self.beta = nn.Parameter(torch.zeros(rounds, dim // 2))

    def forward(self, x: Tensor) -> Tensor:
        y = x
        for round_idx, (left, right) in enumerate(zip(self.left, self.right)):
            mixed = y.clone()
            a = y.index_select(-1, left)
            b = y.index_select(-1, right)
            a_next = a + self.alpha[round_idx] * b
            b_next = b + self.beta[round_idx] * a_next
            mixed.index_copy_(-1, left, a_next)
            mixed.index_copy_(-1, right, b_next)
            y = mixed
        return y


class PermutationLocalMixing(nn.Module):
    """Trainable block-local mixing separated by fixed random permutations."""

    def __init__(self, dim: int, *, block: int, rounds: int, seed: int) -> None:
        super().__init__()
        if dim % block != 0:
            raise ValueError(f"dim={dim} must be divisible by local block={block}")
        self.dim = dim
        self.block = block
        self.rounds = rounds
        self.block_count = dim // block
        gen = torch.Generator(device="cpu").manual_seed(seed)
        perms = [torch.randperm(dim, generator=gen) for _ in range(rounds)]
        self.register_buffer("perms", torch.stack(perms))
        eye = torch.eye(block).view(1, 1, block, block).repeat(rounds, self.block_count, 1, 1)
        noise = torch.randn(rounds, self.block_count, block, block, generator=gen) * 0.02
        self.weight = nn.Parameter(eye + noise)

    def forward(self, x: Tensor) -> Tensor:
        y = x
        for round_idx in range(self.rounds):
            y = y.index_select(-1, self.perms[round_idx])
            blocks = y.reshape(*y.shape[:-1], self.block_count, self.block)
            y = torch.einsum("...nb,nbo->...no", blocks, self.weight[round_idx]).reshape_as(y)
        return y


class LowRankMixing(nn.Module):
    """Small trainable global low-rank linear mixer."""

    def __init__(self, dim: int, *, rank: int, seed: int) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError(f"rank must be >= 1, got {rank}")
        gen = torch.Generator(device="cpu").manual_seed(seed)
        self.down = nn.Parameter(torch.randn(rank, dim, generator=gen) / math.sqrt(dim))
        self.up = nn.Parameter(torch.randn(dim, rank, generator=gen) / math.sqrt(rank))

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(F.linear(x, self.down), self.up)


class CommonModeBypass(nn.Module):
    """Minimal rank-1 common-mode path y += mean(x) * weight."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x.mean(dim=-1, keepdim=True) * self.weight


class TropicalRouteLUT(nn.Module):
    """LUT with richer tropical comparison routes.

    route_kind:
      groupwise: compare max over a channel group against max over another group.
      multiscale: compare one pair against several fixed shifted thresholds.
      tropical_sketch: compare max-plus random sketches.
    """

    def __init__(
        self,
        dim: int,
        *,
        tables: int,
        comparisons: int,
        route_kind: str,
        group_size: int,
        sketch_scale: float,
        fixed_zero_threshold: bool,
        seed: int,
    ) -> None:
        super().__init__()
        if tables < 1:
            raise ValueError(f"tables must be >= 1, got {tables}")
        if comparisons < 1:
            raise ValueError(f"comparisons must be >= 1, got {comparisons}")
        if group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {group_size}")
        if route_kind not in {"groupwise", "multiscale", "tropical_sketch"}:
            raise ValueError(f"unknown tropical route kind: {route_kind}")

        self.dim = dim
        self.tables = tables
        self.comparisons = comparisons
        self.route_kind = route_kind
        self.group_size = group_size
        self.table_size = 1 << comparisons
        self.output_scale = 1.0 / math.sqrt(tables)

        gen = torch.Generator(device="cpu").manual_seed(seed)
        if route_kind == "multiscale":
            pair = torch.zeros(tables, 2, dtype=torch.long)
            for table_idx in range(tables):
                a = torch.randint(0, dim, (1,), generator=gen).item()
                b = torch.randint(0, dim, (1,), generator=gen).item()
                while a == b:
                    b = torch.randint(0, dim, (1,), generator=gen).item()
                pair[table_idx, 0] = a
                pair[table_idx, 1] = b
            thresholds = torch.linspace(-1.5, 1.5, comparisons).view(1, comparisons).repeat(tables, 1)
            self.register_buffer("pair", pair)
            self.register_buffer("thresholds", thresholds)
        else:
            group_a = torch.randint(0, dim, (tables, comparisons, group_size), generator=gen)
            group_b = torch.randint(0, dim, (tables, comparisons, group_size), generator=gen)
            self.register_buffer("group_a", group_a)
            self.register_buffer("group_b", group_b)
            if route_kind == "tropical_sketch":
                offsets = torch.randn(tables, comparisons, 2, group_size, generator=gen) * sketch_scale
            else:
                offsets = torch.zeros(tables, comparisons, 2, group_size)
            self.register_buffer("offsets", offsets)
            thresholds = torch.zeros(tables, comparisons)
            if fixed_zero_threshold:
                self.register_buffer("thresholds", thresholds)
            else:
                self.thresholds = nn.Parameter(thresholds)

        self.lut = nn.Parameter(torch.zeros(tables, self.table_size, dim))
        self.register_buffer("powers", 2 ** torch.arange(comparisons, dtype=torch.long))

    def _route_indices(self, x: Tensor) -> Tensor:
        if self.route_kind == "multiscale":
            pair = self.pair.to(device=x.device)
            diff = x[:, pair[:, 0]] - x[:, pair[:, 1]]
            bits = (diff.unsqueeze(-1) > self.thresholds.to(device=x.device)).to(torch.long)
        else:
            group_a = self.group_a.to(device=x.device)
            group_b = self.group_b.to(device=x.device)
            offsets = self.offsets.to(device=x.device, dtype=x.dtype)
            lhs = (x[:, group_a] + offsets[:, :, 0, :]).amax(dim=-1)
            rhs = (x[:, group_b] + offsets[:, :, 1, :]).amax(dim=-1)
            bits = (lhs - rhs > self.thresholds.to(device=x.device, dtype=x.dtype)).to(torch.long)
        return (bits * self.powers.to(device=x.device).view(1, 1, -1)).sum(dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        batch = x.shape[0]
        indices = self._route_indices(x)
        offsets = (torch.arange(self.tables, device=x.device) * self.table_size).view(1, self.tables)
        linear_idx = (indices + offsets).reshape(-1)
        values = self.lut.reshape(self.tables * self.table_size, self.dim).index_select(0, linear_idx)
        return values.view(batch, self.tables, self.dim).sum(dim=1) * self.output_scale


def _variant_uses_pairwise(variant: str) -> bool:
    return variant not in {
        "lowrank",
        "groupwise",
        "groupwise_bypass",
        "multiscale",
        "multiscale_bypass",
        "tropical_sketch",
        "tropical_sketch_bypass",
    }


def _variant_uses_lowrank(variant: str) -> bool:
    return variant in {"lowrank", "lowrank_pre", "lowrank_post", "lowrank_residual", "lowrank_pre_residual"}


def _variant_uses_tropical_lut(variant: str) -> bool:
    return variant in {
        "groupwise",
        "groupwise_bypass",
        "multiscale",
        "multiscale_bypass",
        "tropical_sketch",
        "tropical_sketch_bypass",
    }


def _variant_uses_lut_tables(variant: str) -> bool:
    return _variant_uses_pairwise(variant) or _variant_uses_tropical_lut(variant)


def _tropical_route_kind(variant: str) -> str:
    if variant.startswith("groupwise"):
        return "groupwise"
    if variant.startswith("multiscale"):
        return "multiscale"
    if variant.startswith("tropical_sketch"):
        return "tropical_sketch"
    raise ValueError(f"variant={variant} is not a tropical LUT variant")


def _variant_uses_common_bypass(variant: str) -> bool:
    return variant.endswith("_bypass") or variant == "plain_bypass"


class DenseProjectionStudent(nn.Module):
    def __init__(
        self,
        *,
        variant: str,
        dim: int,
        tables: int,
        comparisons: int,
        hashes: int,
        rank: int,
        mix_rounds: int,
        local_block: int,
        group_size: int,
        sketch_scale: float,
        fixed_zero_threshold: bool,
        seed: int,
    ) -> None:
        super().__init__()
        self.variant = variant
        self.hashes = hashes if variant == "multihash" else 1
        self.mixer: nn.Module | None = None
        self.lowrank: nn.Module | None = None
        self.tropical_lut: nn.Module | None = None
        self.common_bypass: nn.Module | None = CommonModeBypass(dim) if _variant_uses_common_bypass(variant) else None
        if _variant_uses_lowrank(variant):
            self.lowrank = LowRankMixing(dim, rank=rank, seed=seed + 31)
        if variant == "plain":
            pass
        elif variant == "plain_bypass":
            pass
        elif variant == "residual":
            pass
        elif variant == "lowrank":
            pass
        elif variant == "lowrank_pre":
            pass
        elif variant == "lowrank_post":
            pass
        elif variant == "lowrank_residual":
            pass
        elif variant == "lowrank_pre_residual":
            pass
        elif variant == "expander":
            self.mixer = FixedExpanderMixing(dim, rounds=mix_rounds, seed=seed + 17)
        elif variant == "butterfly":
            self.mixer = FixedButterflyMixing(dim)
        elif variant == "diag_butterfly":
            self.mixer = LearnedDiagonalButterflyMixing(dim, rounds=mix_rounds)
        elif variant == "lifting":
            self.mixer = AdditiveLiftingMixing(dim, rounds=mix_rounds, seed=seed + 19)
        elif variant == "perm_local":
            self.mixer = PermutationLocalMixing(dim, block=local_block, rounds=mix_rounds, seed=seed + 23)
        elif variant == "multihash":
            pass
        elif _variant_uses_tropical_lut(variant):
            self.tropical_lut = TropicalRouteLUT(
                dim,
                tables=tables,
                comparisons=comparisons,
                route_kind=_tropical_route_kind(variant),
                group_size=group_size,
                sketch_scale=sketch_scale,
                fixed_zero_threshold=fixed_zero_threshold,
                seed=seed + 41,
            )
        else:
            raise ValueError(f"unknown variant: {variant}")

        self.branches = nn.ModuleList()
        if _variant_uses_pairwise(variant):
            self.branches = nn.ModuleList(
                [
                    _make_pairwise(
                        dim=dim,
                        tables=tables,
                        comparisons=comparisons,
                        fixed_zero_threshold=fixed_zero_threshold,
                        seed=seed + 1009 * branch_idx,
                    )
                    for branch_idx in range(self.hashes)
                ]
            )

    def forward(self, x: Tensor) -> Tensor:
        if self.variant == "lowrank":
            if self.lowrank is None:
                raise RuntimeError("lowrank variant missing lowrank module")
            return self.lowrank(x)

        if self.variant in {"lowrank_pre", "lowrank_pre_residual"}:
            if self.lowrank is None:
                raise RuntimeError(f"{self.variant} missing lowrank module")
            routed = x + self.lowrank(x)
        else:
            routed = x

        routed = self.mixer(routed) if self.mixer is not None else routed
        if self.tropical_lut is not None:
            out = self.tropical_lut(routed)
        elif len(self.branches) == 1:
            out = self.branches[0](routed.unsqueeze(1)).squeeze(1)
        else:
            values = [branch(routed.unsqueeze(1)).squeeze(1) for branch in self.branches]
            out = torch.stack(values, dim=0).sum(dim=0) / math.sqrt(float(len(values)))
        if self.common_bypass is not None:
            out = out + self.common_bypass(x)
        if self.variant == "residual":
            out = x + out
        elif self.variant == "lowrank_post":
            if self.lowrank is None:
                raise RuntimeError("lowrank_post missing lowrank module")
            out = out + self.lowrank(out)
        elif self.variant == "lowrank_residual":
            if self.lowrank is None:
                raise RuntimeError("lowrank_residual missing lowrank module")
            out = x + self.lowrank(x) + out
        elif self.variant == "lowrank_pre_residual":
            out = routed + out
        return out


def _fit_variant(
    *,
    variant: str,
    dim: int,
    tables: int,
    comparisons: int,
    hashes: int,
    rank: int,
    mix_rounds: int,
    local_block: int,
    group_size: int,
    sketch_scale: float,
    fixed_zero_threshold: bool,
    steps: int,
    batch_size: int,
    eval_batch_size: int,
    lr: float,
    seed: int,
    teacher: Tensor,
    device: torch.device,
) -> FitRow:
    torch.manual_seed(seed)
    model = DenseProjectionStudent(
        variant=variant,
        dim=dim,
        tables=tables,
        comparisons=comparisons,
        hashes=hashes,
        rank=rank,
        mix_rounds=mix_rounds,
        local_block=local_block,
        group_size=group_size,
        sketch_scale=sketch_scale,
        fixed_zero_threshold=fixed_zero_threshold,
        seed=seed,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    gen = torch.Generator(device=device).manual_seed(seed + 1000)

    started = time.perf_counter()
    final_train_mse = float("nan")
    for _ in range(steps):
        x = _sample(batch_size, dim, generator=gen, device=device)
        y = _target(x, teacher)
        pred = model(x)
        loss = F.mse_loss(pred.float(), y.float())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_train_mse = float(loss.item())

    eval_gen = torch.Generator(device=device).manual_seed(seed + 2000)
    with torch.no_grad():
        x_eval = _sample(eval_batch_size, dim, generator=eval_gen, device=device)
        y_eval = _target(x_eval, teacher)
        pred_eval = model(x_eval)
        eval_mse = float(F.mse_loss(pred_eval.float(), y_eval.float()).item())
        zero_mse = float(y_eval.float().square().mean().item())
        eval_nmse = eval_mse / max(zero_mse, 1e-12)
        eval_cos = _cosine_mean(pred_eval, y_eval)

    params = sum(param.numel() for param in model.parameters())
    dense_params = dim * dim
    return FitRow(
        variant=variant,
        dim=dim,
        tables=tables,
        comparisons=comparisons,
        hashes=hashes if variant == "multihash" else 1,
        rank=rank if _variant_uses_lowrank(variant) else 0,
        mix_rounds=mix_rounds if variant in {"expander", "diag_butterfly", "lifting", "perm_local"} else 0,
        local_block=local_block if variant == "perm_local" else 0,
        group_size=group_size if _variant_uses_tropical_lut(variant) else 0,
        sketch_scale=sketch_scale if variant.startswith("tropical_sketch") else 0.0,
        fixed_zero_threshold=fixed_zero_threshold,
        params=params,
        dense_params=dense_params,
        param_ratio=params / dense_params,
        train_steps=steps,
        lr=lr,
        final_train_mse=final_train_mse,
        eval_mse=eval_mse,
        eval_nmse=eval_nmse,
        eval_cos=eval_cos,
        seconds=time.perf_counter() - started,
    )


def _write_csv(path: Path, rows: list[FitRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FitRow.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument(
        "--teacher",
        choices=("orthogonal", "gaussian", "common_mode", "contrast_only", "lowrank_residual"),
        default="orthogonal",
    )
    parser.add_argument("--teacher-rank", type=int, default=32)
    parser.add_argument("--tables-list", type=str, default="4,8,16,32,64,128")
    parser.add_argument("--comparisons-list", type=str, default="4,6,8")
    parser.add_argument(
        "--variants",
        type=str,
        default="plain",
        help=(
            "Comma-separated list: plain,residual,lowrank,lowrank_pre,lowrank_post,"
            "lowrank_residual,lowrank_pre_residual,expander,butterfly,diag_butterfly,"
            "lifting,perm_local,multihash,plain_bypass,groupwise,groupwise_bypass,"
            "multiscale,multiscale_bypass,tropical_sketch,tropical_sketch_bypass."
        ),
    )
    parser.add_argument("--hashes", type=int, default=4)
    parser.add_argument("--rank-list", type=str, default="8")
    parser.add_argument("--mix-rounds", type=int, default=4)
    parser.add_argument("--local-block", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--sketch-scale", type=float, default=0.75)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--fixed-zero-threshold", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    teacher = _teacher_matrix(
        args.dim,
        seed=args.seed,
        kind=args.teacher,
        teacher_rank=args.teacher_rank,
        device=device,
    )
    tables_values = _parse_int_list(args.tables_list)
    comparisons_values = _parse_int_list(args.comparisons_list)
    rank_values = _parse_int_list(args.rank_list)
    variants = _parse_str_list(args.variants)

    rows: list[FitRow] = []
    print(
        "dense_exact,"
        f"dim={args.dim},params={args.dim * args.dim},eval_nmse=0.000000,"
        f"teacher={args.teacher}"
    )
    for variant in variants:
        variant_tables_values = tables_values if _variant_uses_lut_tables(variant) else [0]
        variant_comparisons_values = comparisons_values if _variant_uses_lut_tables(variant) else [0]
        variant_rank_values = rank_values if _variant_uses_lowrank(variant) else [0]
        for rank in variant_rank_values:
            for comparisons in variant_comparisons_values:
                for tables in variant_tables_values:
                    if _variant_uses_lut_tables(variant) and (tables < 1 or comparisons < 1):
                        raise ValueError(f"variant={variant} requires positive tables/comparisons")
                    if _variant_uses_lowrank(variant) and rank < 1:
                        raise ValueError(f"variant={variant} requires positive rank")
                    row = _fit_variant(
                        variant=variant,
                        dim=args.dim,
                        tables=tables,
                        comparisons=comparisons,
                        hashes=args.hashes,
                        rank=rank,
                        mix_rounds=args.mix_rounds,
                        local_block=args.local_block,
                        group_size=args.group_size,
                        sketch_scale=args.sketch_scale,
                        fixed_zero_threshold=args.fixed_zero_threshold,
                        steps=args.steps,
                        batch_size=args.batch_size,
                        eval_batch_size=args.eval_batch_size,
                        lr=args.lr,
                        seed=args.seed + tables * 100 + comparisons + rank * 1000 + len(rows) * 10000,
                        teacher=teacher,
                        device=device,
                    )
                    rows.append(row)
                    print(
                        "pairwise,"
                        f"variant={row.variant},D={row.dim},T={row.tables},C={row.comparisons},"
                        f"hashes={row.hashes},rank={row.rank},mix_rounds={row.mix_rounds},local_block={row.local_block},"
                        f"group_size={row.group_size},sketch_scale={row.sketch_scale:.2f},"
                        f"fixed_thr={row.fixed_zero_threshold},params={row.params},"
                        f"ratio={row.param_ratio:.1f},nmse={row.eval_nmse:.6f},"
                        f"mse={row.eval_mse:.6f},cos={row.eval_cos:.4f},"
                        f"seconds={row.seconds:.1f}",
                        flush=True,
                    )

    out = args.output
    if out is None:
        suffix = "fixedthr" if args.fixed_zero_threshold else "learnthr"
        out = Path("results/experiments/pairwise_dense_projection_fit") / (
            f"d{args.dim}_{args.teacher}_{suffix}_{time.time_ns()}.csv"
        )
    _write_csv(out, rows)
    print(f"metrics -> {out}")


if __name__ == "__main__":
    main()
