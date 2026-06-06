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
    mix_rounds: int
    local_block: int
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


def _teacher_matrix(dim: int, *, seed: int, kind: str, device: torch.device) -> Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    w = torch.randn(dim, dim, generator=gen)
    if kind == "orthogonal":
        q, r = torch.linalg.qr(w)
        signs = torch.sign(torch.diag(r))
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        w = q * signs.view(1, -1)
    elif kind == "gaussian":
        w = w / math.sqrt(dim)
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


class DenseProjectionStudent(nn.Module):
    def __init__(
        self,
        *,
        variant: str,
        dim: int,
        tables: int,
        comparisons: int,
        hashes: int,
        mix_rounds: int,
        local_block: int,
        fixed_zero_threshold: bool,
        seed: int,
    ) -> None:
        super().__init__()
        self.variant = variant
        self.hashes = hashes if variant == "multihash" else 1
        self.mixer: nn.Module | None = None
        if variant == "plain":
            pass
        elif variant == "residual":
            pass
        elif variant == "expander":
            self.mixer = FixedExpanderMixing(dim, rounds=mix_rounds, seed=seed + 17)
        elif variant == "butterfly":
            self.mixer = FixedButterflyMixing(dim)
        elif variant == "perm_local":
            self.mixer = PermutationLocalMixing(dim, block=local_block, rounds=mix_rounds, seed=seed + 23)
        elif variant == "multihash":
            pass
        else:
            raise ValueError(f"unknown variant: {variant}")

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
        routed = self.mixer(x) if self.mixer is not None else x
        if len(self.branches) == 1:
            out = self.branches[0](routed.unsqueeze(1)).squeeze(1)
        else:
            values = [branch(routed.unsqueeze(1)).squeeze(1) for branch in self.branches]
            out = torch.stack(values, dim=0).sum(dim=0) / math.sqrt(float(len(values)))
        if self.variant == "residual":
            out = x + out
        return out


def _fit_variant(
    *,
    variant: str,
    dim: int,
    tables: int,
    comparisons: int,
    hashes: int,
    mix_rounds: int,
    local_block: int,
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
        mix_rounds=mix_rounds,
        local_block=local_block,
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
        mix_rounds=mix_rounds if variant in {"expander", "perm_local"} else 0,
        local_block=local_block if variant == "perm_local" else 0,
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
    parser.add_argument("--teacher", choices=("orthogonal", "gaussian"), default="orthogonal")
    parser.add_argument("--tables-list", type=str, default="4,8,16,32,64,128")
    parser.add_argument("--comparisons-list", type=str, default="4,6,8")
    parser.add_argument(
        "--variants",
        type=str,
        default="plain",
        help="Comma-separated list: plain,residual,expander,butterfly,perm_local,multihash.",
    )
    parser.add_argument("--hashes", type=int, default=4)
    parser.add_argument("--mix-rounds", type=int, default=4)
    parser.add_argument("--local-block", type=int, default=4)
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
    teacher = _teacher_matrix(args.dim, seed=args.seed, kind=args.teacher, device=device)
    tables_values = _parse_int_list(args.tables_list)
    comparisons_values = _parse_int_list(args.comparisons_list)
    variants = _parse_str_list(args.variants)

    rows: list[FitRow] = []
    print(
        "dense_exact,"
        f"dim={args.dim},params={args.dim * args.dim},eval_nmse=0.000000,"
        f"teacher={args.teacher}"
    )
    for variant in variants:
        for comparisons in comparisons_values:
            for tables in tables_values:
                row = _fit_variant(
                    variant=variant,
                    dim=args.dim,
                    tables=tables,
                    comparisons=comparisons,
                    hashes=args.hashes,
                    mix_rounds=args.mix_rounds,
                    local_block=args.local_block,
                    fixed_zero_threshold=args.fixed_zero_threshold,
                    steps=args.steps,
                    batch_size=args.batch_size,
                    eval_batch_size=args.eval_batch_size,
                    lr=args.lr,
                    seed=args.seed + tables * 100 + comparisons + len(rows) * 10000,
                    teacher=teacher,
                    device=device,
                )
                rows.append(row)
                print(
                    "pairwise,"
                    f"variant={row.variant},D={row.dim},T={row.tables},C={row.comparisons},"
                    f"hashes={row.hashes},mix_rounds={row.mix_rounds},local_block={row.local_block},"
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
