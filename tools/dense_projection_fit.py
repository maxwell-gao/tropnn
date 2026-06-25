from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import asdict, dataclass
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
    params: int
    dense_params: int
    train_steps: int
    lr: float
    eval_mse: float
    eval_nmse: float
    eval_cos: float
    seconds: float


class LinearStudent(nn.Module):
    """nn.Linear baseline retained as an experiment baseline."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


def _teacher_matrix(dim: int, *, seed: int, device: torch.device) -> Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    w = torch.randn(dim, dim, generator=gen)
    q, r = torch.linalg.qr(w)
    signs = torch.sign(torch.diag(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return (q * signs.view(1, -1)).to(device)


def _target(x: Tensor, w: Tensor) -> Tensor:
    return x @ w.T


def _cosine_mean(pred: Tensor, target: Tensor) -> float:
    return float(F.cosine_similarity(pred.float(), target.float(), dim=-1).mean().item())


def _build_student(variant: str, dim: int, tables: int, comparisons: int, seed: int) -> nn.Module | None:
    if variant == "dense_exact":
        return None
    if variant == "linear":
        return LinearStudent(dim)
    if variant == "pairwise":
        return PairwiseLinear(dim, dim, tables=tables, comparisons=comparisons, seed=seed, use_output_scaling=True)
    raise ValueError(f"unknown variant {variant!r}")


def fit_variant(variant: str, *, dim: int, tables: int, comparisons: int, steps: int, lr: float, seed: int, batch_size: int, device: str) -> FitRow:
    dev = torch.device(device)
    generator = torch.Generator(device=dev).manual_seed(seed + 17)
    teacher = _teacher_matrix(dim, seed=seed, device=dev)
    dense_params = dim * dim
    t0 = time.perf_counter()
    if variant == "dense_exact":
        params = dense_params
    else:
        model = _build_student(variant, dim, tables, comparisons, seed).to(dev)  # type: ignore[union-attr]
        params = sum(p.numel() for p in model.parameters())
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
        model.train()
        for _ in range(steps):
            x = torch.randn(batch_size, dim, generator=generator, device=dev)
            y = _target(x, teacher)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            pred = pred.squeeze(1) if pred.ndim == 3 and pred.shape[1] == 1 else pred
            loss = F.mse_loss(pred, y)
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        x = torch.randn(batch_size, dim, generator=generator, device=dev)
        y = _target(x, teacher)
        if variant == "dense_exact":
            pred = y
        else:
            pred = model(x)  # type: ignore[possibly-undefined]
            pred = pred.squeeze(1) if pred.ndim == 3 and pred.shape[1] == 1 else pred
        mse = float(F.mse_loss(pred, y).item())
        denom = float(y.square().mean().clamp_min(1e-12).item())
    return FitRow(
        variant=variant,
        dim=dim,
        tables=tables if variant == "pairwise" else 0,
        comparisons=comparisons if variant == "pairwise" else 0,
        params=params,
        dense_params=dense_params,
        train_steps=steps,
        lr=lr,
        eval_mse=mse,
        eval_nmse=mse / denom,
        eval_cos=_cosine_mean(pred, y),
        seconds=time.perf_counter() - t0,
    )


def _parse_str_list(value: str) -> list[str]:
    return [item for item in value.split(",") if item]


def main() -> None:
    parser = argparse.ArgumentParser(description="Dense projection fit with nn.Linear baseline and Pairwise LUT student.")
    parser.add_argument("--variants", default="dense_exact,linear,pairwise")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--tables", type=int, default=64)
    parser.add_argument("--comparisons", type=int, default=6)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="results/experiments/pairwise_dense_projection_fit/summary.csv")
    args = parser.parse_args()

    rows = [
        fit_variant(
            variant,
            dim=args.dim,
            tables=args.tables,
            comparisons=args.comparisons,
            steps=args.steps,
            lr=args.lr,
            seed=args.seed,
            batch_size=args.batch_size,
            device=args.device,
        )
        for variant in _parse_str_list(args.variants)
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
            print(f"variant={row.variant} nmse={row.eval_nmse:.6g} params={row.params}", flush=True)


if __name__ == "__main__":
    main()
