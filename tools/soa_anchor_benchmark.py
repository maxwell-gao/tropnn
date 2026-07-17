from __future__ import annotations

import argparse
import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from ..backends.pairwise_zig import pairwise_zig_forward, pairwise_zig_soa_forward
from ..layers import PAIRWISE_ANCHOR_POLICIES, PairwiseLUT


@dataclass(frozen=True)
class SoAAnchorBenchRow:
    policy: str
    batch_size: int
    seq_len: int
    rows: int
    input_dim: int
    output_dim: int
    tables: int
    comparisons: int
    lut_dtype: str
    params: int
    anchor_dim_frac: float
    mean_abs_anchor_delta: float
    standard_ms: float
    soa_ms: float
    speedup_vs_standard: float
    max_abs_diff: float


def _parse_policies(value: str) -> list[str]:
    if value == "all":
        return list(PAIRWISE_ANCHOR_POLICIES)
    policies = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(policies) - set(PAIRWISE_ANCHOR_POLICIES))
    if unknown:
        raise ValueError(f"unknown anchor policies: {unknown}; expected one of {PAIRWISE_ANCHOR_POLICIES}")
    return policies


def _time_ms(fn, *, warmups: int, iters: int) -> float:
    for _ in range(warmups):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return 1000.0 * (time.perf_counter() - t0) / max(1, iters)


def _anchor_stats(layer: PairwiseLUT) -> tuple[float, float]:
    anchors = layer.anchors.detach().cpu()
    unique_dims = torch.unique(anchors).numel()
    delta = (anchors[..., 0] - anchors[..., 1]).abs().float().mean().item()
    return unique_dims / max(1, layer.input_dim), float(delta)


def benchmark_policy(
    policy: str,
    *,
    batch_size: int,
    seq_len: int,
    input_dim: int,
    output_dim: int,
    tables: int,
    comparisons: int,
    lut_dtype: str,
    seed: int,
    warmups: int,
    iters: int,
) -> SoAAnchorBenchRow:
    layer = PairwiseLUT(
        input_dim,
        output_dim,
        tables=tables,
        comparisons=comparisons,
        backend="torch",
        seed=seed,
        lut_init_std=0.02,
        use_output_scaling=False,
        anchor_policy=policy,
        cpu_lut_dtype="f16" if lut_dtype == "f16" else "f32",
    ).eval()
    x = torch.randn(batch_size, seq_len, input_dim, dtype=torch.float32)
    lut = layer.lut.detach().to(torch.float16 if lut_dtype == "f16" else torch.float32).contiguous()
    thresholds = layer.thresholds.detach().float().contiguous()

    def standard_once() -> torch.Tensor:
        return pairwise_zig_forward(x, layer.anchors, thresholds, lut, lut_dtype=lut_dtype)  # type: ignore[arg-type]

    def soa_once() -> torch.Tensor:
        return pairwise_zig_soa_forward(x, layer.anchors, thresholds, lut, lut_dtype=lut_dtype)  # type: ignore[arg-type]

    standard = standard_once()
    soa = soa_once()
    max_abs_diff = float((standard - soa).abs().max().item())
    standard_ms = _time_ms(standard_once, warmups=warmups, iters=iters)
    soa_ms = _time_ms(soa_once, warmups=warmups, iters=iters)
    anchor_dim_frac, mean_abs_anchor_delta = _anchor_stats(layer)

    return SoAAnchorBenchRow(
        policy=policy,
        batch_size=batch_size,
        seq_len=seq_len,
        rows=batch_size * seq_len,
        input_dim=input_dim,
        output_dim=output_dim,
        tables=tables,
        comparisons=comparisons,
        lut_dtype=lut_dtype,
        params=sum(p.numel() for p in layer.parameters()),
        anchor_dim_frac=anchor_dim_frac,
        mean_abs_anchor_delta=mean_abs_anchor_delta,
        standard_ms=standard_ms,
        soa_ms=soa_ms,
        speedup_vs_standard=standard_ms / max(soa_ms, 1e-12),
        max_abs_diff=max_abs_diff,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark CPU Zig SoA-anchor PairwiseLUT forward across anchor policies.")
    parser.add_argument("--policies", default="all")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--input-dim", type=int, default=1024)
    parser.add_argument("--output-dim", type=int, default=1024)
    parser.add_argument("--tables", type=int, default=64)
    parser.add_argument("--comparisons", type=int, default=6)
    parser.add_argument("--lut-dtype", choices=("f32", "f16"), default="f16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--out", default="results/soa_anchor_benchmark/summary.csv")
    args = parser.parse_args()

    rows = [
        benchmark_policy(
            policy,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            input_dim=args.input_dim,
            output_dim=args.output_dim,
            tables=args.tables,
            comparisons=args.comparisons,
            lut_dtype=args.lut_dtype,
            seed=args.seed,
            warmups=args.warmups,
            iters=args.iters,
        )
        for policy in _parse_policies(args.policies)
    ]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
            print(
                f"policy={row.policy} standard={row.standard_ms:.3f}ms soa={row.soa_ms:.3f}ms "
                f"speedup={row.speedup_vs_standard:.3f} maxdiff={row.max_abs_diff:.3g}",
                flush=True,
            )


if __name__ == "__main__":
    main()
