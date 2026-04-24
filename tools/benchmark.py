from __future__ import annotations

import argparse
import time

import torch
from torch import Tensor

from ..backend import has_triton, trop_scores, trop_scores_reference
from ..layers import TropLinear


def _copy_weights(dst: TropLinear, src: TropLinear) -> None:
    with torch.no_grad():
        dst.proj.weight.copy_(src.proj.weight)
        dst.router_weight.copy_(src.router_weight)
        dst.router_bias.copy_(src.router_bias)
        dst.affine_weight.copy_(src.affine_weight)
        dst.affine_bias.copy_(src.affine_bias)


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


def _bench_forward(layer: TropLinear, x: Tensor, *, warmup: int, steps: int) -> tuple[Tensor, float]:
    with torch.no_grad():
        for _ in range(warmup):
            layer(x)
        _sync_if_cuda(x.device)
        t0 = time.perf_counter()
        last = None
        for _ in range(steps):
            last = layer(x)
        _sync_if_cuda(x.device)
    assert last is not None
    return last, (time.perf_counter() - t0) * 1000.0 / steps


def _bench_scores(layer: TropLinear, x: Tensor, *, warmup: int, steps: int) -> tuple[float, float, float]:
    if x.ndim == 2:
        x = x.unsqueeze(1)
    compute_dtype = torch.float32 if x.dtype in {torch.float16, torch.bfloat16} else x.dtype
    z = layer.proj(x).to(compute_dtype)
    router_weight = layer.router_weight.to(dtype=compute_dtype, device=x.device)
    router_bias = layer.router_bias.to(dtype=compute_dtype, device=x.device)

    with torch.no_grad():
        for _ in range(warmup):
            trop_scores_reference(z, router_weight, router_bias)
        _sync_if_cuda(x.device)
        t0 = time.perf_counter()
        ref_out = None
        for _ in range(steps):
            ref_out = trop_scores_reference(z, router_weight, router_bias)
        _sync_if_cuda(x.device)
        ref_ms = (time.perf_counter() - t0) * 1000.0 / steps

        for _ in range(warmup):
            tri_out = trop_scores(z, router_weight, router_bias, backend="auto")
        _sync_if_cuda(x.device)
        t0 = time.perf_counter()
        for _ in range(steps):
            tri_out = trop_scores(z, router_weight, router_bias, backend="auto")
        _sync_if_cuda(x.device)
        tri_ms = (time.perf_counter() - t0) * 1000.0 / steps

    assert ref_out is not None
    return ref_ms, tri_ms, float((ref_out - tri_out).abs().max().item())


def benchmark_trop_linear_auto(
    *,
    batch_size: int = 512,
    in_features: int = 28 * 28,
    out_features: int = 128,
    tables: int = 16,
    groups: int = 2,
    cells: int = 4,
    rank: int = 32,
    warmup: int = 20,
    steps: int = 100,
    device: str = "cuda",
    seed: int = 0,
) -> dict[str, float]:
    dev = torch.device(device)
    torch.manual_seed(seed)
    torch_layer = TropLinear(
        in_features,
        out_features,
        tables=tables,
        groups=groups,
        cells=cells,
        rank=rank,
        backend="torch",
        seed=seed,
    ).to(dev)
    auto_layer = TropLinear(
        in_features,
        out_features,
        tables=tables,
        groups=groups,
        cells=cells,
        rank=rank,
        backend="auto",
        seed=seed,
    ).to(dev)
    _copy_weights(auto_layer, torch_layer)
    torch_layer.eval()
    auto_layer.eval()
    x = torch.randn(batch_size, in_features, device=dev)
    score_torch_ms, score_auto_ms, score_max_diff = _bench_scores(torch_layer, x, warmup=warmup, steps=steps)
    torch_out, torch_ms = _bench_forward(torch_layer, x, warmup=warmup, steps=steps)
    auto_out, auto_ms = _bench_forward(auto_layer, x, warmup=warmup, steps=steps)
    max_diff = float((torch_out - auto_out).abs().max().item())
    return {
        "score_torch_ms": score_torch_ms,
        "score_auto_ms": score_auto_ms,
        "score_speedup": score_torch_ms / score_auto_ms,
        "score_max_diff": score_max_diff,
        "torch_ms": torch_ms,
        "auto_ms": auto_ms,
        "speedup": torch_ms / auto_ms,
        "max_diff": max_diff,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark TropLinear backend='auto' against backend='torch'.")
    for name, default in (
        ("--batch-size", 512),
        ("--in-features", 28 * 28),
        ("--out-features", 128),
        ("--tables", 16),
        ("--groups", 2),
        ("--cells", 4),
        ("--rank", 32),
        ("--warmup", 20),
        ("--steps", 100),
    ):
        parser.add_argument(name, type=int, default=default)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"cuda={torch.cuda.is_available()} has_triton={has_triton()} device={args.device}")
    result = benchmark_trop_linear_auto(
        batch_size=args.batch_size,
        in_features=args.in_features,
        out_features=args.out_features,
        tables=args.tables,
        groups=args.groups,
        cells=args.cells,
        rank=args.rank,
        warmup=args.warmup,
        steps=args.steps,
        device=args.device,
        seed=args.seed,
    )
    for key in ("score_torch_ms", "score_auto_ms", "score_speedup", "score_max_diff", "torch_ms", "auto_ms", "speedup", "max_diff"):
        precision = 6 if "diff" in key else 4
        print(f"{key}={result[key]:.{precision}f}")


if __name__ == "__main__":
    main()
