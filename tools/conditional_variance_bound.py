"""Estimate conditional-variance lower bounds for pairwise LUT routes.

For a teacher y = W x, the best arbitrary function of a discrete route r(x) is
E[y | r(x)]. Its irreducible error is E[||y - E[y | r]||^2].

This diagnostic estimates two related quantities:

1. joint_route oracle:
   best arbitrary function of the full tuple of PairwiseLinear table indices.
   This is more expressive than the actual additive Pairwise LUT and can become
   sample-limited when the joint route space is huge.

2. additive_route oracle:
   best least-squares additive table model sum_t table_t[route_t].
   This matches the Pairwise LUT value structure more closely and is a better
   optimizer-independent proxy for the actual LUT family.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from ..layers import PairwiseLinear


@dataclass(frozen=True)
class BoundRow:
    teacher: str
    dim: int
    tables: int
    comparisons: int
    teacher_rank: int
    train_samples: int
    eval_samples: int
    route_bits: int
    dense_params: int
    additive_params: int
    unique_train_routes: int
    unseen_eval_frac: float
    target_mse: float
    joint_train_nmse: float
    joint_eval_nmse: float
    additive_train_nmse: float
    additive_eval_nmse: float
    exact_lowrank_residual_nmse: float
    seconds: float


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _sample(batch_size: int, dim: int, *, generator: torch.Generator, device: torch.device) -> Tensor:
    return torch.randn(batch_size, dim, generator=generator, device=device)


def _teacher_matrix(dim: int, *, kind: str, rank: int, seed: int, device: torch.device) -> Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    if kind == "orthogonal":
        w = torch.randn(dim, dim, generator=gen)
        q, r = torch.linalg.qr(w)
        signs = torch.sign(torch.diag(r))
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        w = q * signs.view(1, -1)
    elif kind == "common_mode":
        v = torch.randn(dim, generator=gen)
        v = v / v.square().mean().sqrt().clamp_min(1e-12)
        w = torch.outer(v, torch.ones(dim) / dim)
    elif kind == "contrast_only":
        a = torch.randn(dim, dim, generator=gen) / math.sqrt(max(dim - 1, 1))
        projector = torch.eye(dim) - torch.ones(dim, dim) / dim
        w = a @ projector
    elif kind == "lowrank_residual":
        if rank < 1:
            raise ValueError("lowrank_residual teacher requires rank >= 1")
        u = torch.randn(dim, rank, generator=gen) / math.sqrt(rank)
        v = torch.randn(rank, dim, generator=gen) / math.sqrt(dim)
        w = torch.eye(dim) + u @ v
    else:
        raise ValueError(f"unknown teacher kind: {kind}")
    return w.to(device)


def _target(x: Tensor, w: Tensor) -> Tensor:
    return x @ w.T


def _make_router(
    *,
    dim: int,
    tables: int,
    comparisons: int,
    fixed_zero_threshold: bool,
    seed: int,
    device: torch.device,
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
    ).to(device)


def _routes(x: Tensor, router: PairwiseLinear) -> Tensor:
    anchors = router.anchors.to(device=x.device)
    thresholds = router.thresholds.to(device=x.device)
    powers = router.powers.to(device=x.device)
    a = x[:, anchors[..., 0]]
    b = x[:, anchors[..., 1]]
    bits = (a - b > thresholds).to(torch.long)
    return (bits * powers.view(1, 1, -1)).sum(dim=-1)


def _nmse(pred: Tensor, target: Tensor, target_mse: float) -> float:
    mse = torch.mean((pred.float() - target.float()).square()).item()
    return float(mse / max(target_mse, 1e-12))


def _joint_route_oracle(
    *,
    train_routes: Tensor,
    y_train: Tensor,
    eval_routes: Tensor,
    y_eval: Tensor,
    target_mse: float,
    device: torch.device,
) -> tuple[int, float, float, float]:
    routes_cpu = train_routes.to(device="cpu", dtype=torch.int16)
    unique_cpu, inverse_cpu = torch.unique(routes_cpu, dim=0, return_inverse=True)
    inverse = inverse_cpu.to(device=device, dtype=torch.long)
    group_count = int(unique_cpu.shape[0])
    counts = torch.bincount(inverse, minlength=group_count).to(device=device, dtype=y_train.dtype)
    sums = torch.zeros(group_count, y_train.shape[-1], device=device, dtype=y_train.dtype)
    sums.index_add_(0, inverse, y_train)
    means = sums / counts.clamp_min(1).unsqueeze(-1)
    pred_train = means.index_select(0, inverse)
    train_nmse = _nmse(pred_train, y_train, target_mse)

    route_to_group = {tuple(row.tolist()): idx for idx, row in enumerate(unique_cpu)}
    global_mean = y_train.mean(dim=0)
    pred_eval = global_mean.expand_as(y_eval).clone()
    seen = torch.zeros(y_eval.shape[0], device=device, dtype=torch.bool)
    eval_cpu = eval_routes.to(device="cpu", dtype=torch.int16)
    eval_indices: list[int] = []
    group_indices: list[int] = []
    for sample_idx, row in enumerate(eval_cpu):
        group_idx = route_to_group.get(tuple(row.tolist()))
        if group_idx is not None:
            eval_indices.append(sample_idx)
            group_indices.append(group_idx)
    if eval_indices:
        eval_idx = torch.tensor(eval_indices, device=device, dtype=torch.long)
        group_idx = torch.tensor(group_indices, device=device, dtype=torch.long)
        pred_eval.index_copy_(0, eval_idx, means.index_select(0, group_idx))
        seen.index_fill_(0, eval_idx, True)
    unseen_frac = float((~seen).float().mean().item())
    eval_nmse = _nmse(pred_eval, y_eval, target_mse)
    return group_count, unseen_frac, train_nmse, eval_nmse


def _one_hot_route_features(routes: Tensor, *, tables: int, cells: int, dtype: torch.dtype) -> Tensor:
    samples = routes.shape[0]
    offsets = (torch.arange(tables, device=routes.device, dtype=torch.long) * cells).view(1, tables)
    cols = routes.to(torch.long) + offsets
    features = torch.zeros(samples, tables * cells, device=routes.device, dtype=dtype)
    features.scatter_(1, cols, 1.0)
    return features


def _additive_route_oracle(
    *,
    train_routes: Tensor,
    y_train: Tensor,
    eval_routes: Tensor,
    y_eval: Tensor,
    tables: int,
    comparisons: int,
    ridge: float,
    target_mse: float,
) -> tuple[int, float, float]:
    cells = 1 << comparisons
    features_train = _one_hot_route_features(train_routes, tables=tables, cells=cells, dtype=torch.float32)
    features_eval = _one_hot_route_features(eval_routes, tables=tables, cells=cells, dtype=torch.float32)
    y_train_f = y_train.float()
    gram = features_train.T @ features_train
    gram.diagonal().add_(ridge)
    rhs = features_train.T @ y_train_f
    coeff = torch.linalg.solve(gram, rhs)
    pred_train = features_train @ coeff
    pred_eval = features_eval @ coeff
    return tables * cells * y_train.shape[-1], _nmse(pred_train, y_train, target_mse), _nmse(pred_eval, y_eval, target_mse)


def _estimate_bound(
    *,
    teacher_name: str,
    dim: int,
    tables: int,
    comparisons: int,
    teacher_rank: int,
    train_samples: int,
    eval_samples: int,
    ridge: float,
    seed: int,
    fixed_zero_threshold: bool,
    device: torch.device,
) -> BoundRow:
    started = time.perf_counter()
    teacher = _teacher_matrix(dim, kind=teacher_name, rank=teacher_rank, seed=seed, device=device)
    router = _make_router(
        dim=dim,
        tables=tables,
        comparisons=comparisons,
        fixed_zero_threshold=fixed_zero_threshold,
        seed=seed + 100,
        device=device,
    )
    train_gen = torch.Generator(device=device).manual_seed(seed + 1000)
    eval_gen = torch.Generator(device=device).manual_seed(seed + 2000)
    x_train = _sample(train_samples, dim, generator=train_gen, device=device)
    x_eval = _sample(eval_samples, dim, generator=eval_gen, device=device)
    y_train = _target(x_train, teacher)
    y_eval = _target(x_eval, teacher)
    target_mse = float(y_eval.float().square().mean().item())
    train_routes = _routes(x_train, router)
    eval_routes = _routes(x_eval, router)

    unique_routes, unseen_frac, joint_train_nmse, joint_eval_nmse = _joint_route_oracle(
        train_routes=train_routes,
        y_train=y_train,
        eval_routes=eval_routes,
        y_eval=y_eval,
        target_mse=target_mse,
        device=device,
    )
    additive_params, additive_train_nmse, additive_eval_nmse = _additive_route_oracle(
        train_routes=train_routes,
        y_train=y_train,
        eval_routes=eval_routes,
        y_eval=y_eval,
        tables=tables,
        comparisons=comparisons,
        ridge=ridge,
        target_mse=target_mse,
    )
    exact_lowrank_residual_nmse = 0.0 if teacher_name == "lowrank_residual" else float("nan")
    return BoundRow(
        teacher=teacher_name,
        dim=dim,
        tables=tables,
        comparisons=comparisons,
        teacher_rank=teacher_rank if teacher_name == "lowrank_residual" else 0,
        train_samples=train_samples,
        eval_samples=eval_samples,
        route_bits=tables * comparisons,
        dense_params=dim * dim,
        additive_params=additive_params,
        unique_train_routes=unique_routes,
        unseen_eval_frac=unseen_frac,
        target_mse=target_mse,
        joint_train_nmse=joint_train_nmse,
        joint_eval_nmse=joint_eval_nmse,
        additive_train_nmse=additive_train_nmse,
        additive_eval_nmse=additive_eval_nmse,
        exact_lowrank_residual_nmse=exact_lowrank_residual_nmse,
        seconds=time.perf_counter() - started,
    )


def _write_csv(path: Path, rows: list[BoundRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(BoundRow.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--teachers", type=str, default="common_mode,contrast_only,lowrank_residual")
    parser.add_argument("--teacher-rank", type=int, default=32)
    parser.add_argument("--tables-list", type=str, default="8,32,128")
    parser.add_argument("--comparisons-list", type=str, default="4")
    parser.add_argument("--train-samples", type=int, default=65536)
    parser.add_argument("--eval-samples", type=int, default=65536)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--fixed-zero-threshold", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    teachers = _parse_str_list(args.teachers)
    tables_values = _parse_int_list(args.tables_list)
    comparisons_values = _parse_int_list(args.comparisons_list)

    rows: list[BoundRow] = []
    for teacher in teachers:
        for comparisons in comparisons_values:
            for tables in tables_values:
                row = _estimate_bound(
                    teacher_name=teacher,
                    dim=args.dim,
                    tables=tables,
                    comparisons=comparisons,
                    teacher_rank=args.teacher_rank,
                    train_samples=args.train_samples,
                    eval_samples=args.eval_samples,
                    ridge=args.ridge,
                    seed=args.seed + len(rows) * 10000 + tables * 100 + comparisons,
                    fixed_zero_threshold=args.fixed_zero_threshold,
                    device=device,
                )
                rows.append(row)
                print(
                    "condvar,"
                    f"teacher={row.teacher},D={row.dim},T={row.tables},C={row.comparisons},"
                    f"route_bits={row.route_bits},unique={row.unique_train_routes},"
                    f"unseen={row.unseen_eval_frac:.3f},add_params={row.additive_params},"
                    f"add_ratio={row.additive_params / row.dense_params:.1f},"
                    f"joint_train_nmse={row.joint_train_nmse:.6f},"
                    f"joint_eval_nmse={row.joint_eval_nmse:.6f},"
                    f"add_train_nmse={row.additive_train_nmse:.6f},"
                    f"add_eval_nmse={row.additive_eval_nmse:.6f},"
                    f"lowrank_res_exact={row.exact_lowrank_residual_nmse},"
                    f"seconds={row.seconds:.1f}",
                    flush=True,
                )

    out = args.output
    if out is None:
        out = Path("results/experiments/pairwise_dense_projection_fit") / (
            f"conditional_variance_bound_d{args.dim}_{time.time_ns()}.csv"
        )
    _write_csv(out, rows)
    print(f"metrics -> {out}")


if __name__ == "__main__":
    main()
