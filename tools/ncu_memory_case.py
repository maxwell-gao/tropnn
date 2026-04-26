from __future__ import annotations

import argparse
import math

import torch
import torch.nn as nn
from torch import Tensor

from ..layers import PairwiseLinear, TropLinear


class DenseMlp(nn.Module):
    def __init__(self, in_features: int, out_features: int, *, hidden_features: int, depth: int) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        if depth == 1:
            self.net = nn.Linear(in_features, out_features)
            return

        layers: list[nn.Module] = [nn.Linear(in_features, hidden_features), nn.GELU()]
        for _ in range(depth - 2):
            layers.extend([nn.Linear(hidden_features, hidden_features), nn.GELU()])
        layers.append(nn.Linear(hidden_features, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def _build_model(args: argparse.Namespace) -> nn.Module:
    if args.family == "linear":
        return DenseMlp(args.in_features, args.out_features, hidden_features=args.hidden_features, depth=1)
    if args.family == "mlp":
        return DenseMlp(args.in_features, args.out_features, hidden_features=args.hidden_features, depth=args.mlp_depth)
    if args.family == "tropical":
        return TropLinear(
            args.in_features,
            args.out_features,
            heads=args.heads,
            cells=args.cells,
            code_dim=args.code_dim,
            backend=args.backend,
            seed=args.seed,
        )
    if args.family == "pairwise":
        return PairwiseLinear(
            args.in_features,
            args.out_features,
            tables=args.pairwise_tables,
            comparisons=args.comparisons,
            backend=args.backend,
            seed=args.seed,
        )
    raise ValueError(f"unknown family {args.family!r}")


def _sync() -> None:
    torch.cuda.synchronize()


def _run_forward(model: nn.Module, x: Tensor, repeats: int) -> Tensor:
    last = model(x)
    for _ in range(repeats - 1):
        last = model(x)
    return last


def _run_train(model: nn.Module, x: Tensor, repeats: int) -> None:
    for _ in range(repeats):
        model.zero_grad(set_to_none=True)
        loss = model(x).square().mean()
        loss.backward()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one tropnn case inside a CUDA profiler start/stop range for ncu.")
    parser.add_argument("--family", choices=("linear", "mlp", "tropical", "pairwise"), required=True)
    parser.add_argument("--mode", choices=("forward", "train"), required=True)
    parser.add_argument("--backend", choices=("torch", "auto", "triton", "tilelang"), default="torch")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--in-features", type=int, default=256)
    parser.add_argument("--out-features", type=int, default=512)
    parser.add_argument("--hidden-features", type=int, default=512)
    parser.add_argument("--mlp-depth", type=int, default=2)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--cells", type=int, default=4)
    parser.add_argument("--code-dim", type=int, default=32)
    parser.add_argument("--pairwise-tables", type=int, default=136)
    parser.add_argument("--comparisons", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.set_device(0)
    model = _build_model(args).cuda()
    x = torch.randn(args.batch_size, args.in_features, device="cuda")
    if args.mode == "forward":
        model.eval()
        with torch.no_grad():
            for _ in range(args.warmup):
                _run_forward(model, x, args.repeats)
            _sync()
            torch.cuda.cudart().cudaProfilerStart()
            y = _run_forward(model, x, args.repeats)
            _sync()
            torch.cuda.cudart().cudaProfilerStop()
        checksum = float(y.float().sum().item())
    else:
        model.train()
        for _ in range(args.warmup):
            _run_train(model, x, args.repeats)
        _sync()
        torch.cuda.cudart().cudaProfilerStart()
        _run_train(model, x, args.repeats)
        _sync()
        torch.cuda.cudart().cudaProfilerStop()
        checksum = math.fsum(float(param.grad.float().sum().item()) for param in model.parameters() if param.grad is not None)

    print(
        "case="
        f"{args.family}:{args.backend}:{args.mode} "
        f"batch={args.batch_size} in={args.in_features} out={args.out_features} "
        f"repeats={args.repeats} checksum={checksum:.6f}"
    )


if __name__ == "__main__":
    main()
