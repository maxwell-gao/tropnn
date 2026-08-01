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

from ..layers import TernaryMarginAction


@dataclass(frozen=True)
class ProjectionFitRow:
    variant: str
    seed: int
    dim: int
    atoms: int
    fan_in: int
    params: int
    semantic_route_terms: int
    semantic_action_terms: int
    train_steps: int
    lr: float
    rank_ceiling_nmse: float
    eval_mse: float
    eval_nmse: float
    eval_r2: float
    eval_cos: float
    effective_rank: int
    weight_nmse: float
    bias_rms: float
    input_zero_fraction: float
    action_zero_fraction: float
    seconds: float


def orthogonal_teacher(dim: int, *, seed: int, device: torch.device) -> Tensor:
    """Return a deterministic full-rank teacher with unit singular values."""

    generator = torch.Generator(device="cpu").manual_seed(seed)
    matrix = torch.randn(dim, dim, generator=generator)
    q, r = torch.linalg.qr(matrix)
    signs = torch.sign(torch.diag(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return (q * signs.view(1, -1)).to(device)


def rank_ceiling_nmse(dim: int, atoms: int) -> float:
    """Best rank-``atoms`` NMSE for an orthogonal square teacher."""

    return 1.0 - min(dim, atoms) / dim


def balanced_supports(input_dim: int, atoms: int, fan_in: int, *, seed: int) -> Tensor:
    if not 1 <= fan_in <= input_dim:
        raise ValueError("fan_in must be in [1, input_dim]")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    permutation = torch.randperm(input_dim, generator=generator)
    positions = torch.arange(atoms * fan_in, dtype=torch.long).remainder(input_dim)
    return permutation.index_select(0, positions).view(atoms, fan_in)


class DenseProjection(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim, dim))
        nn.init.orthogonal_(self.weight)

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight)


class DenseLowRankProjection(nn.Module):
    """Float rank-constrained control with the same atom bottleneck."""

    def __init__(self, dim: int, atoms: int, *, seed: int) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.input_weight = nn.Parameter(torch.randn(atoms, dim, generator=generator) / math.sqrt(dim))
        self.action_weight = nn.Parameter(torch.randn(atoms, dim, generator=generator) / math.sqrt(atoms))

    def forward(self, x: Tensor) -> Tensor:
        return (x @ self.input_weight.T) @ self.action_weight


class SparseFloatMarginAction(nn.Module):
    """Float control sharing the ternary student's fixed sparse support."""

    def __init__(self, dim: int, atoms: int, fan_in: int, *, seed: int) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.register_buffer("support_indices", balanced_supports(dim, atoms, fan_in, seed=seed))
        signs = torch.randint(0, 2, (atoms, fan_in), generator=generator).float().mul_(2).sub_(1)
        self.input_weight = nn.Parameter(signs)
        self.action_weight = nn.Parameter(
            torch.randint(-1, 2, (atoms, dim), generator=generator).float()
        )
        self.atoms = atoms
        self.fan_in = fan_in
        self.output_scale = 1.0 / math.sqrt(float(atoms * fan_in))

    def forward(self, x: Tensor) -> Tensor:
        selected = x.index_select(-1, self.support_indices.reshape(-1)).view(
            *x.shape[:-1], self.atoms, self.fan_in
        )
        margins = (selected * self.input_weight).sum(dim=-1)
        return (margins @ self.action_weight) * self.output_scale


class FixedProjection(nn.Module):
    def __init__(self, weight: Tensor) -> None:
        super().__init__()
        self.register_buffer("weight", weight)

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight)


def truncated_svd_teacher(teacher: Tensor, rank: int) -> Tensor:
    u, s, vh = torch.linalg.svd(teacher.float(), full_matrices=False)
    keep = min(rank, s.numel())
    return (u[:, :keep] * s[:keep]) @ vh[:keep]


def build_student(
    variant: str,
    *,
    teacher: Tensor,
    dim: int,
    atoms: int,
    fan_in: int,
    seed: int,
) -> nn.Module:
    if variant == "dense_exact":
        return FixedProjection(teacher.clone())
    if variant == "svd_oracle":
        return FixedProjection(truncated_svd_teacher(teacher, atoms))
    if variant == "dense":
        return DenseProjection(dim)
    if variant == "float_lowrank":
        return DenseLowRankProjection(dim, atoms, seed=seed)
    if variant == "sparse_float":
        return SparseFloatMarginAction(dim, atoms, fan_in, seed=seed)
    if variant == "ternary_margin":
        return TernaryMarginAction(
            dim,
            dim,
            atoms=atoms,
            fan_in=fan_in,
            mode="linear",
            seed=seed,
            fixed_zero_threshold=True,
            use_output_scaling=True,
        )
    raise ValueError(f"unknown variant {variant!r}")


@torch.no_grad()
def materialize_affine(model: nn.Module, dim: int, *, device: torch.device) -> tuple[Tensor, Tensor]:
    """Materialize ``y = x W^T + b`` from basis-vector evaluations."""

    zero = torch.zeros(1, dim, device=device)
    bias = model(zero).reshape(-1).float()
    basis_outputs = model(torch.eye(dim, device=device)).float() - bias.view(1, -1)
    return basis_outputs.T.contiguous(), bias


def cosine_mean(pred: Tensor, target: Tensor) -> float:
    return float(F.cosine_similarity(pred.float(), target.float(), dim=-1).mean().item())


def _hard_code_fractions(model: nn.Module) -> tuple[float, float]:
    if not isinstance(model, TernaryMarginAction):
        return math.nan, math.nan
    input_codes = model.hard_input_codes()
    action_codes = model.hard_direction_codes()
    return (
        float((input_codes == 0).float().mean().item()),
        float((action_codes == 0).float().mean().item()),
    )


def fit_variant(
    variant: str,
    *,
    dim: int,
    atoms: int,
    fan_in: int,
    steps: int,
    lr: float,
    seed: int,
    batch_size: int,
    eval_size: int,
    device: str,
    log_every: int = 0,
) -> ProjectionFitRow:
    dev = torch.device(device)
    teacher = orthogonal_teacher(dim, seed=10_000 + seed, device=dev)
    model = build_student(
        variant,
        teacher=teacher,
        dim=dim,
        atoms=atoms,
        fan_in=fan_in,
        seed=seed,
    ).to(dev)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    params = sum(parameter.numel() for parameter in trainable)
    generator = torch.Generator(device=dev).manual_seed(20_000 + seed)
    t0 = time.perf_counter()

    if trainable and steps > 0:
        optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)
        model.train()
        for step in range(1, steps + 1):
            x = torch.randn(batch_size, dim, generator=generator, device=dev)
            target = F.linear(x, teacher)
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(x), target)
            loss.backward()
            optimizer.step()
            if log_every and (step == 1 or step % log_every == 0 or step == steps):
                print(f"variant={variant} seed={seed} step={step} loss={loss.item():.7f}", flush=True)

    model.eval()
    with torch.no_grad():
        eval_generator = torch.Generator(device=dev).manual_seed(30_000 + seed)
        x = torch.randn(eval_size, dim, generator=eval_generator, device=dev)
        target = F.linear(x, teacher)
        pred = model(x)
        mse = float(F.mse_loss(pred.float(), target.float()).item())
        target_energy = float(target.float().square().mean().clamp_min(1e-12).item())
        nmse = mse / target_energy
        effective_weight, bias = materialize_affine(model, dim, device=dev)
        weight_nmse = float(
            ((effective_weight - teacher.float()).square().sum() / teacher.float().square().sum()).item()
        )
        effective_rank = int(torch.linalg.matrix_rank(effective_weight, rtol=1e-4).item())
        bias_rms = float(bias.square().mean().sqrt().item())
        input_zero_fraction, action_zero_fraction = _hard_code_fractions(model)

    return ProjectionFitRow(
        variant=variant,
        seed=seed,
        dim=dim,
        atoms=atoms,
        fan_in=fan_in,
        params=params,
        semantic_route_terms=(atoms * fan_in if variant in {"sparse_float", "ternary_margin"} else 0),
        semantic_action_terms=(atoms * dim if variant in {"float_lowrank", "sparse_float", "ternary_margin"} else 0),
        train_steps=(steps if trainable else 0),
        lr=(lr if trainable else 0.0),
        rank_ceiling_nmse=rank_ceiling_nmse(dim, atoms),
        eval_mse=mse,
        eval_nmse=nmse,
        eval_r2=1.0 - nmse,
        eval_cos=cosine_mean(pred, target),
        effective_rank=effective_rank,
        weight_nmse=weight_nmse,
        bias_rms=bias_rms,
        input_zero_fraction=input_zero_fraction,
        action_zero_fraction=action_zero_fraction,
        seconds=time.perf_counter() - t0,
    )


def _parse_variants(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit a full-rank orthogonal linear teacher with dense, low-rank, sparse-float, and ternary students."
    )
    parser.add_argument(
        "--variants",
        default="dense_exact,svd_oracle,dense,float_lowrank,sparse_float,ternary_margin",
    )
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--atoms", type=int, default=64)
    parser.add_argument("--fan-in", type=int, default=8)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument(
        "--out",
        default="results/experiments/ternary_margin_projection_fit/summary.csv",
    )
    args = parser.parse_args()

    rows = [
        fit_variant(
            variant,
            dim=args.dim,
            atoms=args.atoms,
            fan_in=args.fan_in,
            steps=args.steps,
            lr=args.lr,
            seed=args.seed,
            batch_size=args.batch_size,
            eval_size=args.eval_size,
            device=args.device,
            log_every=args.log_every,
        )
        for variant in _parse_variants(args.variants)
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
            print(
                f"variant={row.variant} nmse={row.eval_nmse:.6f} "
                f"weight_nmse={row.weight_nmse:.6f} rank={row.effective_rank}",
                flush=True,
            )


if __name__ == "__main__":
    main()
