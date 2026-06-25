from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ...layers import PairwiseLinear, PairwiseWalshLinear

FAMILIES = (
    "paper",
    "untied_paper",
    "linear",
    "pairwise",
    "tied_pairwise",
    "pairwise_walsh",
    "tied_pairwise_walsh",
)
PAIRWISE_FAMILIES = ("pairwise", "tied_pairwise", "pairwise_walsh", "tied_pairwise_walsh")
WALSH_PAIRWISE_FAMILIES = ("pairwise_walsh", "tied_pairwise_walsh")
TIED_RECOVERY_FAMILIES = ("tied_pairwise", "tied_pairwise_walsh")


def feature_probabilities(n_features: int, alpha: float, activation_density: float, *, device: torch.device) -> Tensor:
    ranks = torch.arange(1, n_features + 1, device=device, dtype=torch.float32)
    weights = ranks.pow(-alpha)
    weights = weights / weights.mean().clamp_min(1e-12)
    return (weights * activation_density).clamp(max=1.0)


def sample_batch(probs: Tensor, batch_size: int) -> Tensor:
    return torch.bernoulli(probs.expand(batch_size, -1))


def squeeze_single_sequence(y: Tensor) -> Tensor:
    return y.squeeze(1) if y.ndim == 3 and y.shape[1] == 1 else y


@dataclass(frozen=True)
class RunConfig:
    family: str
    n_features: int
    model_dim: int
    alpha: float
    activation_density: float
    batch_size: int
    steps: int
    lr: float
    weight_decay: float
    tables: int
    comparisons: int
    walsh_order: int
    seed: int
    device: str


class PaperFeatureRecovery(nn.Module):
    def __init__(self, n_features: int, model_dim: int, *, seed: int) -> None:
        super().__init__()
        gen = torch.Generator(device="cpu").manual_seed(seed)
        self.weight = nn.Parameter(torch.randn(n_features, model_dim, generator=gen) / math.sqrt(model_dim))
        self.bias = nn.Parameter(torch.zeros(n_features))

    def encode(self, x: Tensor) -> Tensor:
        return x @ self.weight

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(self.encode(x) @ self.weight.t() + self.bias)


class UntiedPaperFeatureRecovery(nn.Module):
    def __init__(self, n_features: int, model_dim: int, *, seed: int) -> None:
        super().__init__()
        gen = torch.Generator(device="cpu").manual_seed(seed)
        self.encoder_weight = nn.Parameter(torch.randn(n_features, model_dim, generator=gen) / math.sqrt(model_dim))
        self.decoder_weight = nn.Parameter(torch.randn(n_features, model_dim, generator=gen) / math.sqrt(model_dim))
        self.bias = nn.Parameter(torch.zeros(n_features))

    def encode(self, x: Tensor) -> Tensor:
        return x @ self.encoder_weight

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(self.encode(x) @ self.decoder_weight.t() + self.bias)


class LinearRecovery(nn.Module):
    """nn.Linear baseline retained for report reproduction."""

    def __init__(self, n_features: int, model_dim: int, *, seed: int) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.encoder = nn.Linear(n_features, model_dim, bias=False)
        self.decoder = nn.Linear(model_dim, n_features)

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(self.decoder(self.encoder(x)))


class TiedRecovery(nn.Module):
    def __init__(self, router: nn.Module) -> None:
        super().__init__()
        self.router = router

    def forward(self, x: Tensor) -> Tensor:
        y = self.router(x)
        return y.squeeze(1) if y.ndim == 3 and y.shape[1] == 1 else y


def _pairwise_shape(config: RunConfig) -> tuple[int, int]:
    tables = config.tables if config.tables > 0 else max(1, config.model_dim)
    comparisons = config.comparisons if config.comparisons > 0 else 6
    return tables, comparisons


def _build_pairwise(config: RunConfig) -> nn.Module:
    tables, comparisons = _pairwise_shape(config)
    return PairwiseLinear(config.n_features, config.n_features, tables=tables, comparisons=comparisons, seed=config.seed)


def _build_pairwise_walsh(config: RunConfig) -> nn.Module:
    tables, comparisons = _pairwise_shape(config)
    return PairwiseWalshLinear(
        config.n_features,
        config.n_features,
        tables=tables,
        comparisons=comparisons,
        walsh_order=config.walsh_order,  # type: ignore[arg-type]
        seed=config.seed,
    )


BUILDERS: dict[str, Callable[[RunConfig], nn.Module]] = {
    "paper": lambda c: PaperFeatureRecovery(c.n_features, c.model_dim, seed=c.seed),
    "untied_paper": lambda c: UntiedPaperFeatureRecovery(c.n_features, c.model_dim, seed=c.seed),
    "linear": lambda c: LinearRecovery(c.n_features, c.model_dim, seed=c.seed),
    "pairwise": _build_pairwise,
    "tied_pairwise": lambda c: TiedRecovery(_build_pairwise(c)),
    "pairwise_walsh": _build_pairwise_walsh,
    "tied_pairwise_walsh": lambda c: TiedRecovery(_build_pairwise_walsh(c)),
}


def run_config(config: RunConfig) -> dict[str, float | int | str]:
    if config.family not in BUILDERS:
        raise ValueError(f"unknown family {config.family!r}; expected one of {FAMILIES}")
    device = torch.device(config.device)
    torch.manual_seed(config.seed)
    probs = feature_probabilities(config.n_features, config.alpha, config.activation_density, device=device)
    model = BUILDERS[config.family](config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    t0 = time.perf_counter()
    loss_value = float("nan")
    for _ in range(config.steps):
        x = sample_batch(probs, config.batch_size)
        optimizer.zero_grad(set_to_none=True)
        y = squeeze_single_sequence(model(x))
        loss = F.mse_loss(y, x)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().item())
    seconds = time.perf_counter() - t0
    return {
        **asdict(config),
        "params": sum(p.numel() for p in model.parameters()),
        "final_loss": loss_value,
        "seconds": seconds,
    }


def _parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def _parse_float_list(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item]


def _parse_str_list(value: str) -> list[str]:
    return [item for item in value.split(",") if item]


def main() -> None:
    parser = argparse.ArgumentParser(description="Recovery scaling benchmark with nn.Linear baseline and Pairwise LUT families.")
    parser.add_argument("--families", default="paper,untied_paper,linear,pairwise,tied_pairwise,pairwise_walsh,tied_pairwise_walsh")
    parser.add_argument("--n-features", type=int, default=256)
    parser.add_argument("--model-dims", default="8,16,32,64")
    parser.add_argument("--alphas", default="1.0")
    parser.add_argument("--activation-density", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--tables", type=int, default=32)
    parser.add_argument("--comparisons", type=int, default=6)
    parser.add_argument("--walsh-order", type=int, choices=(1, 2), default=2)
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="results/scaling_benchmark/summary.csv")
    args = parser.parse_args()

    rows: list[dict[str, float | int | str]] = []
    for family in _parse_str_list(args.families):
        for model_dim in _parse_int_list(args.model_dims):
            for alpha in _parse_float_list(args.alphas):
                for seed in _parse_int_list(args.seeds):
                    cfg = RunConfig(
                        family=family,
                        n_features=args.n_features,
                        model_dim=model_dim,
                        alpha=alpha,
                        activation_density=args.activation_density,
                        batch_size=args.batch_size,
                        steps=args.steps,
                        lr=args.lr,
                        weight_decay=args.weight_decay,
                        tables=args.tables,
                        comparisons=args.comparisons,
                        walsh_order=args.walsh_order,
                        seed=seed,
                        device=args.device,
                    )
                    row = run_config(cfg)
                    rows.append(row)
                    print(f"family={family} dim={model_dim} alpha={alpha} seed={seed} loss={row['final_loss']:.6g}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
