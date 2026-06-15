"""Fit a dense full-rank projection with Pairwise LUT slope-bank variants.

This is a small synthetic probe for the hypothesis that plain Pairwise LUTs
spend many parameters approximating a simple continuous full-rank mixing map.
The shared low-rank slope bank adds a small number of reusable affine atoms to
the compare-lookup-accumulate path without introducing a dense projection.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from tropnn.layers.pairwise import PairwiseLinear


@dataclasses.dataclass
class FitRow:
    variant: str
    dim: int
    tables: int
    comparisons: int
    table_size: int
    lowrank_rank: int
    slope_bank_rank: int
    params: int
    train_mse: float
    eval_mse: float
    steps: int
    lr: float
    seed: int
    target_seed: int


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _count_params(module: nn.Module) -> int:
    return sum(param.numel() for param in module.parameters())


class LowRankResidualPremix(nn.Module):
    def __init__(self, dim: int, rank: int) -> None:
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.normal_(self.down.weight, std=1.0 / math.sqrt(dim))
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.down(x))


class Student(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        variant: str,
        tables: int,
        comparisons: int,
        lowrank_rank: int,
        slope_bank_rank: int,
        fixed_zero_threshold: bool,
        slope_bank_atom_init_std: float,
        slope_bank_coeff_init_std: float,
    ) -> None:
        super().__init__()
        self.variant = variant
        self.premix: nn.Module | None = None
        if variant in {"lowrank_pre", "lowrank_pre_slope_bank"}:
            if lowrank_rank <= 0:
                raise ValueError("lowrank variants require lowrank_rank > 0")
            self.premix = LowRankResidualPremix(dim, lowrank_rank)
        elif variant not in {"plain", "slope_bank"}:
            raise ValueError(f"unknown variant: {variant}")

        active_slope_rank = slope_bank_rank if variant in {"slope_bank", "lowrank_pre_slope_bank"} else 0
        self.pairwise = PairwiseLinear(
            dim,
            dim,
            tables=tables,
            comparisons=comparisons,
            fixed_zero_threshold=fixed_zero_threshold,
            use_min_margin_ste=True,
            slope_bank_rank=active_slope_rank,
            slope_bank_atom_init_std=slope_bank_atom_init_std,
            slope_bank_coeff_init_std=slope_bank_coeff_init_std,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.premix is not None:
            x = self.premix(x)
        out = self.pairwise(x)
        if out.ndim == x.ndim + 1 and out.shape[-2] == 1:
            out = out.squeeze(-2)
        return out


def _fit_once(
    *,
    variant: str,
    dim: int,
    tables: int,
    comparisons: int,
    lowrank_rank: int,
    slope_bank_rank: int,
    steps: int,
    batch_size: int,
    eval_batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    seed: int,
    target_seed: int,
    fixed_zero_threshold: bool,
    slope_bank_atom_init_std: float,
    slope_bank_coeff_init_std: float,
    log_every: int,
) -> FitRow:
    target_gen = torch.Generator()
    target_gen.manual_seed(target_seed)
    target = torch.randn(dim, dim, generator=target_gen).to(device) / math.sqrt(dim)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = Student(
        dim=dim,
        variant=variant,
        tables=tables,
        comparisons=comparisons,
        lowrank_rank=lowrank_rank,
        slope_bank_rank=slope_bank_rank,
        fixed_zero_threshold=fixed_zero_threshold,
        slope_bank_atom_init_std=slope_bank_atom_init_std,
        slope_bank_coeff_init_std=slope_bank_coeff_init_std,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_mse = float("nan")
    for step in range(1, steps + 1):
        x = torch.randn(batch_size, dim, device=device)
        y = x @ target.t()
        pred = model(x)
        loss = F.mse_loss(pred, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        train_mse = float(loss.detach().cpu())
        if log_every > 0 and step % log_every == 0:
            print(
                f"variant={variant} tables={tables} comparisons={comparisons} "
                f"rank={lowrank_rank} slope_rank={slope_bank_rank} "
                f"step={step} train_mse={train_mse:.6f}",
                flush=True,
            )

    with torch.no_grad():
        x = torch.randn(eval_batch_size, dim, device=device)
        y = x @ target.t()
        eval_mse = float(F.mse_loss(model(x), y).detach().cpu())

    return FitRow(
        variant=variant,
        dim=dim,
        tables=tables,
        comparisons=comparisons,
        table_size=1 << comparisons,
        lowrank_rank=lowrank_rank if variant in {"lowrank_pre", "lowrank_pre_slope_bank"} else 0,
        slope_bank_rank=slope_bank_rank if variant in {"slope_bank", "lowrank_pre_slope_bank"} else 0,
        params=_count_params(model),
        train_mse=train_mse,
        eval_mse=eval_mse,
        steps=steps,
        lr=lr,
        seed=seed,
        target_seed=target_seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", default="plain,lowrank_pre,slope_bank,lowrank_pre_slope_bank")
    parser.add_argument("--dim-list", default="64")
    parser.add_argument("--tables-list", default="32,128")
    parser.add_argument("--comparisons-list", default="4")
    parser.add_argument("--rank-list", default="4")
    parser.add_argument("--slope-bank-rank-list", default="4")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-seed", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fixed-zero-threshold", action="store_true")
    parser.add_argument("--slope-bank-atom-init-std", type=float, default=0.02)
    parser.add_argument("--slope-bank-coeff-init-std", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    variants = _parse_str_list(args.variants)
    dims = _parse_int_list(args.dim_list)
    tables_values = _parse_int_list(args.tables_list)
    comparisons_values = _parse_int_list(args.comparisons_list)
    rank_values = _parse_int_list(args.rank_list)
    slope_rank_values = _parse_int_list(args.slope_bank_rank_list)

    rows: list[FitRow] = []
    run_index = 0
    target_seed = args.seed if args.target_seed is None else args.target_seed
    for dim in dims:
        for tables in tables_values:
            for comparisons in comparisons_values:
                for variant in variants:
                    active_ranks = rank_values if variant in {"lowrank_pre", "lowrank_pre_slope_bank"} else [0]
                    active_slope_ranks = (
                        slope_rank_values if variant in {"slope_bank", "lowrank_pre_slope_bank"} else [0]
                    )
                    for rank in active_ranks:
                        for slope_rank in active_slope_ranks:
                            run_seed = args.seed + run_index
                            run_index += 1
                            print(
                                f"start variant={variant} dim={dim} tables={tables} comparisons={comparisons} "
                                f"rank={rank} slope_rank={slope_rank} seed={run_seed}",
                                flush=True,
                            )
                            row = _fit_once(
                                variant=variant,
                                dim=dim,
                                tables=tables,
                                comparisons=comparisons,
                                lowrank_rank=rank,
                                slope_bank_rank=slope_rank,
                                steps=args.steps,
                                batch_size=args.batch_size,
                                eval_batch_size=args.eval_batch_size,
                                lr=args.lr,
                                weight_decay=args.weight_decay,
                                device=device,
                                seed=run_seed,
                                target_seed=target_seed,
                                fixed_zero_threshold=args.fixed_zero_threshold,
                                slope_bank_atom_init_std=args.slope_bank_atom_init_std,
                                slope_bank_coeff_init_std=args.slope_bank_coeff_init_std,
                                log_every=args.log_every,
                            )
                            rows.append(row)
                            print(dataclasses.asdict(row), flush=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[field.name for field in dataclasses.fields(FitRow)])
        writer.writeheader()
        for row in rows:
            writer.writerow(dataclasses.asdict(row))


if __name__ == "__main__":
    main()
