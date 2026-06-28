from __future__ import annotations

import argparse
import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from ..backends.pairwise_tilelang import pairwise_tilelang
from ..backends.pairwise_zig import pairwise_zig_forward, pairwise_zig_tree_tiled_forward
from ..layers import PairwiseLUT


@dataclass(frozen=True)
class TreeKernelBenchRow:
    backend: str
    variant: str
    dim: int
    rows: int
    batch_size: int
    seq_len: int
    tables: int
    comparisons: int
    anchor_policy: str
    dtype: str
    lut_dtype: str
    forward_ms: float
    fwd_bwd_ms: float
    max_abs_diff: float


def _parse_dims(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_ms(fn, *, device: torch.device, warmups: int, iters: int) -> float:
    for _ in range(warmups):
        fn()
    _sync(device)
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize(device)
        return float(start.elapsed_time(end) / max(1, iters))
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return 1000.0 * (time.perf_counter() - t0) / max(1, iters)


def _zig_rows(args: argparse.Namespace, dim: int) -> list[TreeKernelBenchRow]:
    layer = PairwiseLUT(dim, dim, tables=args.tables, comparisons=args.comparisons, backend="torch", seed=args.seed, lut_init_std=0.02, anchor_policy=args.anchor_policy, use_output_scaling=False).eval()
    x = torch.randn(args.batch_size, args.seq_len, dim, dtype=torch.float32)
    thresholds = layer.thresholds.detach().float().contiguous()
    lut = layer.lut.detach().float().contiguous()

    def standard_once() -> torch.Tensor:
        return pairwise_zig_forward(x, layer.anchors, thresholds, lut, lut_dtype="f32")

    def tree_once() -> torch.Tensor:
        return pairwise_zig_tree_tiled_forward(x, layer.anchors, thresholds, lut)

    standard = standard_once()
    tree = tree_once()
    diff = float((standard - tree).abs().max().item())
    standard_ms = _time_ms(standard_once, device=torch.device("cpu"), warmups=args.warmups, iters=args.iters)
    tree_ms = _time_ms(tree_once, device=torch.device("cpu"), warmups=args.warmups, iters=args.iters)
    base = dict(dim=dim, rows=args.batch_size * args.seq_len, batch_size=args.batch_size, seq_len=args.seq_len, tables=args.tables, comparisons=args.comparisons, anchor_policy=args.anchor_policy, dtype="float32", lut_dtype="f32", fwd_bwd_ms=0.0, max_abs_diff=diff)
    return [
        TreeKernelBenchRow(backend="zig", variant="standard", forward_ms=standard_ms, **base),
        TreeKernelBenchRow(backend="zig", variant="tree_tiled", forward_ms=tree_ms, **base),
    ]


def _tilelang_rows(args: argparse.Namespace, dim: int) -> list[TreeKernelBenchRow]:
    device = torch.device(args.device)
    layer = PairwiseLUT(dim, dim, tables=args.tables, comparisons=args.comparisons, backend="torch", seed=args.seed, lut_init_std=0.02, anchor_policy=args.anchor_policy, use_output_scaling=False).to(device)
    x = torch.randn(args.batch_size, args.seq_len, dim, device=device, dtype=torch.float32)
    x_train_standard = x.detach().clone().requires_grad_(True)
    x_train_u8 = x.detach().clone().requires_grad_(True)

    def call(route_index_dtype: str, x_arg: torch.Tensor) -> torch.Tensor:
        return pairwise_tilelang(
            x_arg,
            layer.anchors.to(device),
            layer.thresholds.to(dtype=torch.float32, device=device),
            layer.lut,
            use_min_margin_ste=True,
            surrogate=layer.surrogate,
            lut_dtype="fp32",
            route_index_dtype=route_index_dtype,  # type: ignore[arg-type]
        )[0]

    with torch.no_grad():
        standard = call("int64", x)
        u8 = call("uint8", x)
    diff = float((standard - u8).abs().max().item())

    def standard_forward() -> None:
        with torch.no_grad():
            call("int64", x)

    def u8_forward() -> None:
        with torch.no_grad():
            call("uint8", x)

    def standard_fwd_bwd() -> None:
        layer.zero_grad(set_to_none=True)
        x_train_standard.grad = None
        call("int64", x_train_standard).float().square().mean().backward()

    def u8_fwd_bwd() -> None:
        layer.zero_grad(set_to_none=True)
        x_train_u8.grad = None
        call("uint8", x_train_u8).float().square().mean().backward()

    standard_fwd = _time_ms(standard_forward, device=device, warmups=args.warmups, iters=args.iters)
    u8_fwd = _time_ms(u8_forward, device=device, warmups=args.warmups, iters=args.iters)
    standard_bwd = _time_ms(standard_fwd_bwd, device=device, warmups=args.warmups, iters=args.iters)
    u8_bwd = _time_ms(u8_fwd_bwd, device=device, warmups=args.warmups, iters=args.iters)
    base = dict(dim=dim, rows=args.batch_size * args.seq_len, batch_size=args.batch_size, seq_len=args.seq_len, tables=args.tables, comparisons=args.comparisons, anchor_policy=args.anchor_policy, dtype="float32", lut_dtype="fp32", max_abs_diff=diff)
    return [
        TreeKernelBenchRow(backend="tilelang", variant="standard_int64_route", forward_ms=standard_fwd, fwd_bwd_ms=standard_bwd, **base),
        TreeKernelBenchRow(backend="tilelang", variant="uint8_route", forward_ms=u8_fwd, fwd_bwd_ms=u8_bwd, **base),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark tree-system-inspired Pairwise LUT kernels.")
    parser.add_argument("--backend", choices=("zig", "tilelang", "both"), default="both")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dims", default="512,1024,2048,4096")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--tables", type=int, default=64)
    parser.add_argument("--comparisons", type=int, default=6)
    parser.add_argument("--anchor-policy", default="permuted")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--out", default="results/tree_kernel_benchmark/summary.csv")
    args = parser.parse_args()

    rows: list[TreeKernelBenchRow] = []
    for dim in _parse_dims(args.dims):
        if args.backend in {"zig", "both"}:
            rows.extend(_zig_rows(args, dim))
        if args.backend in {"tilelang", "both"}:
            rows.extend(_tilelang_rows(args, dim))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
            print(f"backend={row.backend} variant={row.variant} dim={row.dim} fwd={row.forward_ms:.3f}ms fwd_bwd={row.fwd_bwd_ms:.3f}ms maxdiff={row.max_abs_diff:.3g}", flush=True)


if __name__ == "__main__":
    main()
