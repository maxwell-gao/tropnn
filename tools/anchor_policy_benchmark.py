from __future__ import annotations

import argparse
import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn

from ..layers import PAIRWISE_ANCHOR_POLICIES, PairwiseLUT


@dataclass(frozen=True)
class AnchorBenchRow:
    policy: str
    backend: str
    device: str
    dtype: str
    batch_size: int
    seq_len: int
    input_dim: int
    output_dim: int
    tables: int
    comparisons: int
    params: int
    anchor_dim_frac: float
    mean_abs_anchor_delta: float
    route_ms: float
    forward_ms: float
    fwd_bwd_ms: float
    peak_mem_mb: float


def _dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(f"unsupported dtype {name!r}")


def _parse_policies(value: str) -> list[str]:
    if value == "all":
        return list(PAIRWISE_ANCHOR_POLICIES)
    policies = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(policies) - set(PAIRWISE_ANCHOR_POLICIES))
    if unknown:
        raise ValueError(f"unknown anchor policies: {unknown}; expected one of {PAIRWISE_ANCHOR_POLICIES}")
    return policies


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
    _sync(device)
    return 1000.0 * (time.perf_counter() - t0) / max(1, iters)


def _timed_with_peak(fn, *, device: torch.device, warmups: int, iters: int) -> tuple[float, float]:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    ms = _time_ms(fn, device=device, warmups=warmups, iters=iters)
    peak = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else 0.0
    return ms, float(peak)


def _anchor_stats(layer: PairwiseLUT) -> tuple[float, float]:
    anchors = layer.anchors.detach().cpu()
    unique_dims = torch.unique(anchors).numel()
    delta = (anchors[..., 0] - anchors[..., 1]).abs().float().mean().item()
    return unique_dims / max(1, layer.input_dim), float(delta)


def benchmark_policy(
    policy: str,
    *,
    backend: str,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    seq_len: int,
    input_dim: int,
    output_dim: int,
    tables: int,
    comparisons: int,
    seed: int,
    warmups: int,
    iters: int,
) -> AnchorBenchRow:
    layer = PairwiseLUT(
        input_dim,
        output_dim,
        tables=tables,
        comparisons=comparisons,
        backend=backend,  # type: ignore[arg-type]
        seed=seed,
        lut_init_std=0.02,
        use_output_scaling=True,
        anchor_policy=policy,
    ).to(device)
    x = torch.randn(batch_size, seq_len, input_dim, device=device, dtype=dtype)
    x_train = x.detach().clone().requires_grad_(True)

    def route_once() -> None:
        with torch.no_grad():
            layer.cache_index(x)

    def forward_once() -> None:
        layer.eval()
        with torch.no_grad():
            layer(x)

    def fwd_bwd_once() -> None:
        layer.train()
        layer.zero_grad(set_to_none=True)
        x_train.grad = None
        loss = layer(x_train).float().square().mean()
        loss.backward()

    route_ms, route_peak = _timed_with_peak(route_once, device=device, warmups=warmups, iters=iters)
    forward_ms, forward_peak = _timed_with_peak(forward_once, device=device, warmups=warmups, iters=iters)
    fwd_bwd_ms, fwd_bwd_peak = _timed_with_peak(fwd_bwd_once, device=device, warmups=warmups, iters=iters)
    anchor_dim_frac, mean_abs_anchor_delta = _anchor_stats(layer)

    return AnchorBenchRow(
        policy=policy,
        backend=backend,
        device=str(device),
        dtype=str(dtype).replace("torch.", ""),
        batch_size=batch_size,
        seq_len=seq_len,
        input_dim=input_dim,
        output_dim=output_dim,
        tables=tables,
        comparisons=comparisons,
        params=sum(p.numel() for p in layer.parameters()),
        anchor_dim_frac=anchor_dim_frac,
        mean_abs_anchor_delta=mean_abs_anchor_delta,
        route_ms=route_ms,
        forward_ms=forward_ms,
        fwd_bwd_ms=fwd_bwd_ms,
        peak_mem_mb=max(route_peak, forward_peak, fwd_bwd_peak),
    )


def benchmark_linear(
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    seq_len: int,
    input_dim: int,
    output_dim: int,
    warmups: int,
    iters: int,
) -> AnchorBenchRow:
    layer = nn.Linear(input_dim, output_dim, bias=False).to(device=device, dtype=dtype)
    x = torch.randn(batch_size, seq_len, input_dim, device=device, dtype=dtype)
    x_train = x.detach().clone().requires_grad_(True)

    def forward_once() -> None:
        layer.eval()
        with torch.no_grad():
            layer(x)

    def fwd_bwd_once() -> None:
        layer.train()
        layer.zero_grad(set_to_none=True)
        x_train.grad = None
        loss = layer(x_train).float().square().mean()
        loss.backward()

    forward_ms, forward_peak = _timed_with_peak(forward_once, device=device, warmups=warmups, iters=iters)
    fwd_bwd_ms, fwd_bwd_peak = _timed_with_peak(fwd_bwd_once, device=device, warmups=warmups, iters=iters)
    return AnchorBenchRow(
        policy="linear",
        backend="torch",
        device=str(device),
        dtype=str(dtype).replace("torch.", ""),
        batch_size=batch_size,
        seq_len=seq_len,
        input_dim=input_dim,
        output_dim=output_dim,
        tables=0,
        comparisons=0,
        params=sum(p.numel() for p in layer.parameters()),
        anchor_dim_frac=1.0,
        mean_abs_anchor_delta=0.0,
        route_ms=0.0,
        forward_ms=forward_ms,
        fwd_bwd_ms=fwd_bwd_ms,
        peak_mem_mb=max(forward_peak, fwd_bwd_peak),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark PairwiseLUT anchor policies.")
    parser.add_argument("--policies", default="all")
    parser.add_argument("--backend", choices=("torch", "tilelang", "triton", "zig"), default="torch")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="float32")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--input-dim", type=int, default=256)
    parser.add_argument("--output-dim", type=int, default=256)
    parser.add_argument("--tables", type=int, default=64)
    parser.add_argument("--comparisons", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--include-linear", action="store_true")
    parser.add_argument("--out", default="results/anchor_policy_benchmark/summary.csv")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = _dtype(args.dtype)
    rows = [
        benchmark_policy(
            policy,
            backend=args.backend,
            device=device,
            dtype=dtype,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            input_dim=args.input_dim,
            output_dim=args.output_dim,
            tables=args.tables,
            comparisons=args.comparisons,
            seed=args.seed,
            warmups=args.warmups,
            iters=args.iters,
        )
        for policy in _parse_policies(args.policies)
    ]
    if args.include_linear:
        rows.append(
            benchmark_linear(
                device=device,
                dtype=dtype,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                input_dim=args.input_dim,
                output_dim=args.output_dim,
                warmups=args.warmups,
                iters=args.iters,
            )
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
            print(
                f"policy={row.policy} route={row.route_ms:.3f}ms "
                f"forward={row.forward_ms:.3f}ms fwd_bwd={row.fwd_bwd_ms:.3f}ms",
                flush=True,
            )


if __name__ == "__main__":
    main()
