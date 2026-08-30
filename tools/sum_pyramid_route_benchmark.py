from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch
from torch import Tensor

from tropnn.backends.pairwise_tilelang import _pairwise_route_kernel
from tropnn.backends.sum_pyramid_tilelang import (
    sum_pyramid_pairwise_route_tilelang_full,
    sum_pyramid_pairwise_route_torch,
)
from tropnn.layers.accumulation import SumPyramid
from tropnn.layers.hard_lookup import sum_lookup_rows, weighted_neighbor_delta
from tropnn.layers.surrogate import ste_heaviside
from tropnn.tools.zipf_groupsum_pclut_capacity_law import make_pyramid_anchors


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_ms(function: object, *, device: torch.device, warmups: int, iterations: int) -> float:
    callable_function = function  # keep the hot loop free of attribute lookups in the call sites
    for _ in range(warmups):
        callable_function()  # type: ignore[operator]
    _sync(device)
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            callable_function()  # type: ignore[operator]
        end.record()
        torch.cuda.synchronize(device)
        return float(start.elapsed_time(end) / iterations)
    started = time.perf_counter()
    for _ in range(iterations):
        callable_function()  # type: ignore[operator]
    return 1000.0 * (time.perf_counter() - started) / iterations


def _time_repeated_ms(
    function: object,
    *,
    device: torch.device,
    warmups: int,
    iterations: int,
    repeats: int,
) -> tuple[float, list[float]]:
    samples = [_time_ms(function, device=device, warmups=warmups, iterations=iterations) for _ in range(repeats)]
    return float(statistics.median(samples)), samples


def _canonical_route_tilelang(
    latent: Tensor,
    anchors: Tensor,
    thresholds: Tensor,
    *,
    route_block: int = 64,
) -> tuple[Tensor, Tensor, Tensor]:
    items, in_features = latent.shape
    tables, comparisons, _ = anchors.shape
    indices = torch.empty(items, tables, device=latent.device, dtype=torch.int64)
    margins = torch.empty(items, tables, comparisons, device=latent.device, dtype=torch.float32)
    rmins = torch.empty(items, tables, device=latent.device, dtype=torch.uint8)
    kernel = _pairwise_route_kernel(
        items,
        in_features,
        tables,
        comparisons,
        route_block,
        math.ceil(tables / route_block),
        "float32",
        "cuda",
    )
    kernel(latent, anchors, thresholds, indices, margins, rmins)
    return indices, margins, rmins


def _hard_lookup(lut: Tensor, indices: Tensor) -> Tensor:
    return sum_lookup_rows(lut, indices, accumulation_dtype=torch.float32)


def _ste_lookup(lut: Tensor, indices: Tensor, margins: Tensor) -> Tensor:
    hard = _hard_lookup(lut, indices)
    bit = margins.abs().argmin(dim=-1)
    selected_margin = margins.gather(-1, bit.unsqueeze(-1)).squeeze(-1)
    neighbor = indices ^ (2**bit).long()
    ste = ste_heaviside(selected_margin, "fast_sigmoid_odd") - (selected_margin > 0).to(selected_margin.dtype)
    return hard + weighted_neighbor_delta(lut, indices, neighbor, ste)


def benchmark(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("the matched TileLang benchmark requires CUDA")
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    torch.manual_seed(args.seed)
    x = torch.randn(args.items, args.n_features, device=device, dtype=torch.float32)
    pyramid = SumPyramid(args.n_features, signed=args.signed, seed=args.seed + 1).to(device)
    anchors = make_pyramid_anchors(
        args.n_features,
        args.tables,
        args.comparisons,
        policy=args.anchor_policy,
        seed=args.seed + 2,
    ).to(device)
    thresholds = (torch.randn(args.tables, args.comparisons, device=device) * 0.1).contiguous()
    lut = torch.randn(
        args.tables,
        1 << args.comparisons,
        args.output_dim,
        device=device,
        dtype=torch.float32,
    )
    leaf_anchors = make_pyramid_anchors(
        args.n_features,
        args.tables,
        args.comparisons,
        policy="leaf_only",
        seed=args.seed + 3,
    ).to(device)

    def fused_route() -> tuple[Tensor, Tensor, Tensor]:
        return sum_pyramid_pairwise_route_tilelang_full(
            x,
            pyramid.signs,
            anchors,
            thresholds,
            threads=args.threads,
        )

    def unfused_route() -> tuple[Tensor, Tensor, Tensor]:
        return _canonical_route_tilelang(pyramid(x), anchors, thresholds)

    def canonical_route() -> tuple[Tensor, Tensor, Tensor]:
        return _canonical_route_tilelang(x, leaf_anchors, thresholds)

    with torch.no_grad():
        reference_indices, reference_margins = sum_pyramid_pairwise_route_torch(x, pyramid.signs, anchors, thresholds)
        reference_rmins = reference_margins.abs().argmin(dim=-1).to(torch.uint8)
        fused_indices, fused_margins, fused_rmins = fused_route()
        unfused_indices, unfused_margins, unfused_rmins = unfused_route()
    parity = {
        "fused_indices_exact": bool(torch.equal(fused_indices, reference_indices)),
        "unfused_indices_exact": bool(torch.equal(unfused_indices, reference_indices)),
        "fused_margin_max_abs": float((fused_margins - reference_margins).abs().max()),
        "unfused_margin_max_abs": float((unfused_margins - reference_margins).abs().max()),
        "fused_rmins_exact": bool(torch.equal(fused_rmins, reference_rmins)),
        "unfused_rmins_exact": bool(torch.equal(unfused_rmins, reference_rmins)),
    }
    parity["pass"] = bool(
        parity["fused_indices_exact"]
        and parity["unfused_indices_exact"]
        and parity["fused_rmins_exact"]
        and parity["unfused_rmins_exact"]
        and parity["fused_margin_max_abs"] <= 3e-5
        and parity["unfused_margin_max_abs"] <= 3e-5
    )
    if not parity["pass"]:
        raise RuntimeError(f"sum-pyramid route parity failed: {parity}")

    def fused_end_to_end() -> Tensor:
        indices, _, _ = fused_route()
        return _hard_lookup(lut, indices)

    def canonical_end_to_end() -> Tensor:
        indices, _, _ = canonical_route()
        return _hard_lookup(lut, indices)

    def unfused_end_to_end() -> Tensor:
        indices, _, _ = unfused_route()
        return _hard_lookup(lut, indices)

    x_fused = x.detach().clone().requires_grad_(True)
    x_reference = x.detach().clone().requires_grad_(True)
    threshold_fused = thresholds.detach().clone().requires_grad_(True)
    threshold_reference = thresholds.detach().clone().requires_grad_(True)
    lut_fused = lut.detach().clone().requires_grad_(True)
    lut_reference = lut.detach().clone().requires_grad_(True)

    def fused_forward_backward() -> None:
        x_fused.grad = None
        threshold_fused.grad = None
        lut_fused.grad = None
        indices, margins, _ = sum_pyramid_pairwise_route_tilelang_full(
            x_fused,
            pyramid.signs,
            anchors,
            threshold_fused,
            threads=args.threads,
        )
        _ste_lookup(lut_fused, indices, margins).square().mean().backward()

    def reference_forward_backward() -> None:
        x_reference.grad = None
        threshold_reference.grad = None
        lut_reference.grad = None
        indices, margins = sum_pyramid_pairwise_route_torch(x_reference, pyramid.signs, anchors, threshold_reference)
        _ste_lookup(lut_reference, indices, margins).square().mean().backward()

    fused_forward_backward()
    reference_forward_backward()
    backward_parity = {
        "input_gradient_max_abs": float((x_fused.grad - x_reference.grad).abs().max()),
        "threshold_gradient_max_abs": float((threshold_fused.grad - threshold_reference.grad).abs().max()),
        "payload_gradient_max_abs": float((lut_fused.grad - lut_reference.grad).abs().max()),
    }
    backward_parity["pass"] = bool(max(backward_parity.values()) <= 3e-5)
    if not backward_parity["pass"]:
        raise RuntimeError(f"sum-pyramid route backward parity failed: {backward_parity}")

    benchmark_cases = {
        "canonical_leaf_route_ms": (canonical_route, args.warmups, args.iterations),
        "unfused_pyramid_route_ms": (unfused_route, args.warmups, args.iterations),
        "fused_pyramid_route_ms": (fused_route, args.warmups, args.iterations),
        "canonical_route_plus_common_lookup_ms": (canonical_end_to_end, args.warmups, args.iterations),
        "unfused_route_plus_common_lookup_ms": (unfused_end_to_end, args.warmups, args.iterations),
        "fused_route_plus_common_lookup_ms": (fused_end_to_end, args.warmups, args.iterations),
        "torch_reference_forward_backward_ms": (
            reference_forward_backward,
            max(1, args.warmups // 2),
            max(1, args.iterations // 2),
        ),
        "fused_route_common_lookup_forward_backward_ms": (
            fused_forward_backward,
            max(1, args.warmups // 2),
            max(1, args.iterations // 2),
        ),
    }
    timing: dict[str, float] = {}
    timing_samples: dict[str, list[float]] = {}
    for name, (function, warmups, iterations) in benchmark_cases.items():
        timing[name], timing_samples[name] = _time_repeated_ms(
            function,
            device=device,
            warmups=warmups,
            iterations=iterations,
            repeats=args.repeats,
        )
    timing["fused_vs_canonical_route_ratio"] = timing["fused_pyramid_route_ms"] / timing["canonical_leaf_route_ms"]
    timing["fused_vs_canonical_route_plus_lookup_ratio"] = (
        timing["fused_route_plus_common_lookup_ms"] / timing["canonical_route_plus_common_lookup_ms"]
    )
    return {
        "schema": "sum-pyramid-pairwise-route-benchmark-v7",
        "complete": True,
        "config": {
            "items": args.items,
            "n_features": args.n_features,
            "output_dim": args.output_dim,
            "tables": args.tables,
            "threads": args.threads,
            "comparisons": args.comparisons,
            "signed": args.signed,
            "anchor_policy": args.anchor_policy,
            "dtype": "float32",
            "warmups": args.warmups,
            "iterations": args.iterations,
            "repeats": args.repeats,
            "device": str(device),
        },
        "parity": parity,
        "backward_parity": backward_parity,
        "timing": timing,
        "timing_samples_ms": timing_samples,
        "traffic_contract": {
            "fused_pyramid_materialized_hbm_bytes": 0,
            "fused_input_bytes_per_item": 4 * args.n_features,
            "fused_code_bytes_per_item_int64": 8 * args.tables,
            "fused_margin_bytes_per_item": 4 * args.tables * args.comparisons,
            "fused_rmin_bytes_per_item": args.tables,
            "measured_dram_bytes": None,
            "claim": "analytic kernel contract only; no measured HBM traffic claim without profiler counters",
        },
        "environment": {
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(device),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Matched fused SumPyramid route benchmark")
    parser.add_argument("--items", type=int, default=512)
    parser.add_argument("--n-features", type=int, default=1024)
    parser.add_argument("--output-dim", type=int, default=32)
    parser.add_argument("--tables", type=int, default=32)
    parser.add_argument("--comparisons", type=int, default=6)
    parser.add_argument("--threads", type=int, choices=(256, 512), default=256)
    parser.add_argument("--signed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--anchor-policy", default="level_biased")
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
