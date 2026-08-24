from __future__ import annotations

import argparse
import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from ..layers import ComparatorTwoSidedMargin


@dataclass(frozen=True)
class LayoutBenchRow:
    layout: str
    device: str
    dtype: str
    batch_size: int
    seq_len: int
    input_dim: int
    output_dim: int
    tables: int
    comparisons: int
    k_c: int
    output_tile_size: int
    params: int
    csr_entries: int
    csr_max_degree: int
    forward_ms: float
    fwd_bwd_ms: float
    peak_mem_mb: float
    max_abs_fwd_diff: float
    max_abs_grad_x_diff: float
    max_abs_grad_weight_diff: float
    max_abs_grad_threshold_diff: float


def _dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(f"unsupported dtype {name!r}")


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


def _make_layer(
    *,
    layout: str,
    device: torch.device,
    backend: str | None = None,
    input_dim: int,
    output_dim: int,
    tables: int,
    comparisons: int,
    k_c: int,
    output_tile_size: int,
    seed: int,
) -> ComparatorTwoSidedMargin:
    return ComparatorTwoSidedMargin(
        input_dim,
        output_dim,
        tables=tables,
        comparisons=comparisons,
        k_c=k_c,
        backend=backend or ("triton" if device.type == "cuda" else "torch"),
        seed=seed,
        anchor_policy="permuted",
        write_policy="expander",
        reduction_layout=layout,  # type: ignore[arg-type]
        output_tile_size=output_tile_size,
        weight_init="signed",
        use_output_scaling=True,
    ).to(device)


def _clone_reference_state(src: ComparatorTwoSidedMargin, dst: ComparatorTwoSidedMargin) -> None:
    with torch.no_grad():
        dst.anchors.copy_(src.anchors)
        dst.write_indices.copy_(src.write_indices)
        dst.csr_offsets.copy_(src.csr_offsets)
        dst.csr_sources.copy_(src.csr_sources)
        dst.csr_weight_indices.copy_(src.csr_weight_indices)
        dst.write_weight.copy_(src.write_weight)
        dst.thresholds.copy_(src.thresholds)
    dst.csr_max_degree = src.csr_max_degree


def _max_diff(a: torch.Tensor | None, b: torch.Tensor | None) -> float:
    if a is None or b is None:
        return float("nan")
    return float((a.float() - b.float()).abs().max().item())


def _equivalence(
    scatter: ComparatorTwoSidedMargin,
    output_major: ComparatorTwoSidedMargin,
    x: torch.Tensor,
) -> tuple[float, float, float, float]:
    xs = x.detach().clone().requires_grad_(True)
    xo = x.detach().clone().requires_grad_(True)
    scatter.zero_grad(set_to_none=True)
    output_major.zero_grad(set_to_none=True)
    ys = scatter(xs)
    yo = output_major(xo)
    grad = torch.randn_like(ys)
    ys.backward(grad)
    yo.backward(grad)
    return (
        _max_diff(ys, yo),
        _max_diff(xs.grad, xo.grad),
        _max_diff(scatter.write_weight.grad, output_major.write_weight.grad),
        _max_diff(scatter.thresholds.grad, output_major.thresholds.grad),
    )


def _benchmark_layout(
    layer: ComparatorTwoSidedMargin,
    *,
    x: torch.Tensor,
    warmups: int,
    iters: int,
) -> tuple[float, float, float]:
    device = x.device
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
    return forward_ms, fwd_bwd_ms, max(forward_peak, fwd_bwd_peak)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark two-sided comparator margin sparse write layouts.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--input-dim", type=int, default=384)
    parser.add_argument("--output-dim", type=int, default=384)
    parser.add_argument("--tables", type=int, default=64)
    parser.add_argument("--comparisons", type=int, default=6)
    parser.add_argument("--k-c", type=int, default=48)
    parser.add_argument("--output-tile-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = _dtype(args.dtype)
    torch.manual_seed(args.seed)
    scatter = _make_layer(
        layout="scatter",
        device=device,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        tables=args.tables,
        comparisons=args.comparisons,
        k_c=args.k_c,
        output_tile_size=args.output_tile_size,
        seed=args.seed,
    )
    output_major = _make_layer(
        layout="output_major",
        device=device,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        tables=args.tables,
        comparisons=args.comparisons,
        k_c=args.k_c,
        output_tile_size=args.output_tile_size,
        seed=args.seed,
    )
    dense_training = _make_layer(
        layout="dense_training",
        device=device,
        backend="torch",
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        tables=args.tables,
        comparisons=args.comparisons,
        k_c=args.k_c,
        output_tile_size=args.output_tile_size,
        seed=args.seed,
    )
    tile_ref = _make_layer(
        layout="tile_local",
        device=device,
        backend="torch",
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        tables=args.tables,
        comparisons=args.comparisons,
        k_c=args.k_c,
        output_tile_size=args.output_tile_size,
        seed=args.seed,
    )
    tile_local = _make_layer(
        layout="tile_local",
        device=device,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        tables=args.tables,
        comparisons=args.comparisons,
        k_c=args.k_c,
        output_tile_size=args.output_tile_size,
        seed=args.seed,
    )
    _clone_reference_state(scatter, output_major)
    _clone_reference_state(scatter, dense_training)
    _clone_reference_state(tile_ref, tile_local)
    x = torch.randn(args.batch_size, args.seq_len, args.input_dim, device=device, dtype=dtype)

    fwd_diff, grad_x_diff, grad_weight_diff, grad_threshold_diff = _equivalence(scatter, output_major, x)
    dense_fwd_diff, dense_grad_x_diff, dense_grad_weight_diff, dense_grad_threshold_diff = _equivalence(
        scatter,
        dense_training,
        x,
    )
    tile_fwd_diff, tile_grad_x_diff, tile_grad_weight_diff, tile_grad_threshold_diff = _equivalence(tile_ref, tile_local, x)
    rows: list[LayoutBenchRow] = []
    for layer, diffs in (
        (scatter, (0.0, 0.0, 0.0, 0.0)),
        (output_major, (fwd_diff, grad_x_diff, grad_weight_diff, grad_threshold_diff)),
        (
            dense_training,
            (
                dense_fwd_diff,
                dense_grad_x_diff,
                dense_grad_weight_diff,
                dense_grad_threshold_diff,
            ),
        ),
        (tile_local, (tile_fwd_diff, tile_grad_x_diff, tile_grad_weight_diff, tile_grad_threshold_diff)),
    ):
        forward_ms, fwd_bwd_ms, peak_mem_mb = _benchmark_layout(layer, x=x, warmups=args.warmups, iters=args.iters)
        row_fwd_diff, row_grad_x_diff, row_grad_weight_diff, row_grad_threshold_diff = diffs
        rows.append(
            LayoutBenchRow(
                layout=layer.reduction_layout,
                device=str(device),
                dtype=str(dtype).replace("torch.", ""),
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                input_dim=args.input_dim,
                output_dim=args.output_dim,
                tables=args.tables,
                comparisons=args.comparisons,
                k_c=args.k_c,
                output_tile_size=args.output_tile_size,
                params=sum(p.numel() for p in layer.parameters()),
                csr_entries=int(layer.csr_sources.numel()),
                csr_max_degree=int(layer.csr_max_degree),
                forward_ms=forward_ms,
                fwd_bwd_ms=fwd_bwd_ms,
                peak_mem_mb=peak_mem_mb,
                max_abs_fwd_diff=row_fwd_diff,
                max_abs_grad_x_diff=row_grad_x_diff,
                max_abs_grad_weight_diff=row_grad_weight_diff,
                max_abs_grad_threshold_diff=row_grad_threshold_diff,
            )
        )

    for row in rows:
        print(
            f"layout={row.layout} forward={row.forward_ms:.3f}ms fwd_bwd={row.fwd_bwd_ms:.3f}ms "
            f"peak={row.peak_mem_mb:.1f}MB fwd_diff={row.max_abs_fwd_diff:.3g} grad_x_diff={row.max_abs_grad_x_diff:.3g}"
        )

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow(asdict(row))


if __name__ == "__main__":
    main()
