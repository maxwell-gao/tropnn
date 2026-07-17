from __future__ import annotations

import argparse
import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from ..backends.pairwise_zig import pairwise_zig_forward, pairwise_zig_paged_forward
from ..layers import PAIRWISE_ANCHOR_POLICIES, PairwiseLUT


@dataclass(frozen=True)
class PagedAnchorBenchRow:
    policy: str
    batch_size: int
    seq_len: int
    rows: int
    input_dim: int
    output_dim: int
    tables: int
    comparisons: int
    page_size: int
    lut_dtype: str
    params: int
    anchor_dim_frac: float
    mean_abs_anchor_delta: float
    entropy_mean: float
    active_codes_mean: float
    empty_bucket_ratio: float
    max_bucket_p95: float
    tokens_per_active_bucket: float
    standard_ms: float
    paged_ms: float
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


def _route_page_stats(layer: PairwiseLUT, x: torch.Tensor, *, page_size: int) -> tuple[float, float, float, float, float]:
    indices = layer.cache_index(x).indices.reshape(-1, layer.tables)
    rows = indices.shape[0]
    table_size = layer.table_size
    entropies: list[float] = []
    active_codes: list[float] = []
    empty_buckets = 0
    bucket_count = 0
    max_buckets: list[float] = []
    active_token_total = 0.0
    active_bucket_total = 0.0
    for start in range(0, rows, page_size):
        stop = min(start + page_size, rows)
        tile = indices[start:stop]
        for table in range(layer.tables):
            hist = torch.bincount(tile[:, table], minlength=table_size).float()
            total = hist.sum().clamp_min(1.0)
            active = hist > 0
            active_n = int(active.sum().item())
            probs = hist[active] / total
            entropy = float(-(probs * torch.log2(probs)).sum().item()) if active_n else 0.0
            entropies.append(entropy)
            active_codes.append(float(active_n))
            empty_buckets += table_size - active_n
            bucket_count += table_size
            max_buckets.append(float(hist.max().item()))
            active_token_total += float(total.item())
            active_bucket_total += float(active_n)
    max_bucket_p95 = float(torch.tensor(max_buckets).quantile(0.95).item()) if max_buckets else 0.0
    return (
        sum(entropies) / max(1, len(entropies)),
        sum(active_codes) / max(1, len(active_codes)),
        empty_buckets / max(1, bucket_count),
        max_bucket_p95,
        active_token_total / max(1.0, active_bucket_total),
    )


def benchmark_policy(
    policy: str,
    *,
    batch_size: int,
    seq_len: int,
    input_dim: int,
    output_dim: int,
    tables: int,
    comparisons: int,
    page_size: int,
    lut_dtype: str,
    seed: int,
    warmups: int,
    iters: int,
) -> PagedAnchorBenchRow:
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

    def paged_once() -> torch.Tensor:
        return pairwise_zig_paged_forward(x, layer.anchors, thresholds, lut, lut_dtype=lut_dtype, page_size=page_size)  # type: ignore[arg-type]

    standard = standard_once()
    paged = paged_once()
    max_abs_diff = float((standard - paged).abs().max().item())
    standard_ms = _time_ms(standard_once, warmups=warmups, iters=iters)
    paged_ms = _time_ms(paged_once, warmups=warmups, iters=iters)
    anchor_dim_frac, mean_abs_anchor_delta = _anchor_stats(layer)
    entropy_mean, active_codes_mean, empty_bucket_ratio, max_bucket_p95, tokens_per_active_bucket = _route_page_stats(layer, x, page_size=page_size)

    return PagedAnchorBenchRow(
        policy=policy,
        batch_size=batch_size,
        seq_len=seq_len,
        rows=batch_size * seq_len,
        input_dim=input_dim,
        output_dim=output_dim,
        tables=tables,
        comparisons=comparisons,
        page_size=page_size,
        lut_dtype=lut_dtype,
        params=sum(p.numel() for p in layer.parameters()),
        anchor_dim_frac=anchor_dim_frac,
        mean_abs_anchor_delta=mean_abs_anchor_delta,
        entropy_mean=entropy_mean,
        active_codes_mean=active_codes_mean,
        empty_bucket_ratio=empty_bucket_ratio,
        max_bucket_p95=max_bucket_p95,
        tokens_per_active_bucket=tokens_per_active_bucket,
        standard_ms=standard_ms,
        paged_ms=paged_ms,
        speedup_vs_standard=standard_ms / max(paged_ms, 1e-12),
        max_abs_diff=max_abs_diff,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark CPU Zig page-style PairwiseLUT forward across anchor policies.")
    parser.add_argument("--policies", default="all")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--input-dim", type=int, default=1024)
    parser.add_argument("--output-dim", type=int, default=1024)
    parser.add_argument("--tables", type=int, default=64)
    parser.add_argument("--comparisons", type=int, default=6)
    parser.add_argument("--page-size", type=int, default=1024)
    parser.add_argument("--lut-dtype", choices=("f32", "f16"), default="f16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--out", default="results/paged_anchor_benchmark/summary.csv")
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
            page_size=args.page_size,
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
                f"policy={row.policy} standard={row.standard_ms:.3f}ms paged={row.paged_ms:.3f}ms "
                f"speedup={row.speedup_vs_standard:.3f} active_codes={row.active_codes_mean:.1f} "
                f"empty={row.empty_bucket_ratio:.3f} maxdiff={row.max_abs_diff:.3g}",
                flush=True,
            )


if __name__ == "__main__":
    main()
