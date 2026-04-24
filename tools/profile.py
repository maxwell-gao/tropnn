from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from ..examples.emnist import EmnistRoutedClassifier


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


def _build_model(
    family: str,
    *,
    input_dim: int,
    hidden_dim: int,
    out_features: int,
    depth: int,
    heads: int,
    cells: int,
    code_dim: int,
    pairwise_tables: int,
    comparisons: int,
    backend: str,
    seed: int,
    device: torch.device,
) -> EmnistRoutedClassifier:
    return EmnistRoutedClassifier(
        family=family,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=out_features,
        depth=depth,
        heads=heads,
        cells=cells,
        code_dim=code_dim,
        pairwise_tables=pairwise_tables,
        comparisons=comparisons,
        backend=backend if family == "tropical" else "torch",
        seed=seed,
    ).to(device)


def _event_rows(events: list[Any], *, use_cuda: bool, limit: int) -> list[dict[str, float | int | str]]:
    sort_key = "device_time_total" if use_cuda else "self_cpu_time_total"
    top = sorted(events, key=lambda event: getattr(event, sort_key, 0.0), reverse=True)[:limit]
    rows: list[dict[str, float | int | str]] = []
    for event in top:
        rows.append(
            {
                "name": event.key,
                "calls": int(event.count),
                "self_cpu_time_us": float(event.self_cpu_time_total),
                "cpu_time_us": float(event.cpu_time_total),
                "self_device_time_us": float(getattr(event, "self_device_time_total", 0.0)),
                "device_time_us": float(getattr(event, "device_time_total", 0.0)),
                "self_cpu_memory_bytes": int(getattr(event, "self_cpu_memory_usage", 0)),
                "cpu_memory_bytes": int(getattr(event, "cpu_memory_usage", 0)),
                "self_device_memory_bytes": int(getattr(event, "self_device_memory_usage", 0)),
                "device_memory_bytes": int(getattr(event, "device_memory_usage", 0)),
            }
        )
    return rows


def profile_family_forward(
    family: str,
    *,
    batch_size: int = 512,
    input_dim: int = 28 * 28,
    hidden_dim: int = 128,
    out_features: int = 10,
    depth: int = 2,
    heads: int = 32,
    cells: int = 4,
    code_dim: int = 32,
    pairwise_tables: int = 72,
    comparisons: int = 6,
    backend: str = "torch",
    warmup: int = 5,
    seed: int = 0,
    dtype: str = "float32",
    device: str = "cuda",
    top_ops: int = 12,
) -> dict[str, Any]:
    dev = torch.device(device)
    use_cuda = dev.type == "cuda"
    torch.manual_seed(seed)
    model = _build_model(
        family,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        out_features=out_features,
        depth=depth,
        heads=heads,
        cells=cells,
        code_dim=code_dim,
        pairwise_tables=pairwise_tables,
        comparisons=comparisons,
        backend=backend,
        seed=seed,
        device=dev,
    )
    model.eval()

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    x = torch.randn(batch_size, input_dim, device=dev, dtype=dtype_map[dtype])
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(dev)

    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        _sync_if_cuda(dev)

        activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if use_cuda else [])
        t0 = time.perf_counter()
        with profile(activities=activities, profile_memory=True, record_shapes=False, with_stack=False, acc_events=True) as prof:
            with record_function(f"{family}_forward"):
                model(x)
        _sync_if_cuda(dev)
        wall_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "family": family,
        "device": dev.type,
        "dtype": dtype,
        "batch_size": batch_size,
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "out_features": out_features,
        "depth": depth,
        "heads": heads if family == "tropical" else None,
        "cells": cells if family == "tropical" else None,
        "code_dim": code_dim if family == "tropical" else None,
        "pairwise_tables": pairwise_tables if family == "pairwise" else None,
        "comparisons": comparisons if family == "pairwise" else None,
        "backend": backend if family == "tropical" else "torch",
        "params": int(sum(param.numel() for param in model.parameters())),
        "wall_ms": wall_ms,
        "peak_device_memory_bytes": int(torch.cuda.max_memory_allocated(dev)) if use_cuda else 0,
        "top_ops": _event_rows(list(prof.key_averages()), use_cuda=use_cuda, limit=top_ops),
    }


def profile_family_set(
    *,
    batch_size: int = 512,
    input_dim: int = 28 * 28,
    out_features: int = 10,
    depth: int = 2,
    tropical_hidden: int = 128,
    tropical_heads: int = 32,
    tropical_cells: int = 4,
    tropical_code_dim: int = 32,
    tropical_backend: str = "torch",
    pairwise_hidden: int = 128,
    pairwise_tables: int = 72,
    pairwise_comparisons: int = 6,
    warmup: int = 5,
    seed: int = 0,
    dtype: str = "float32",
    device: str = "cuda",
    top_ops: int = 12,
) -> dict[str, Any]:
    return {
        "tropical": profile_family_forward(
            "tropical",
            batch_size=batch_size,
            input_dim=input_dim,
            hidden_dim=tropical_hidden,
            out_features=out_features,
            depth=depth,
            heads=tropical_heads,
            cells=tropical_cells,
            code_dim=tropical_code_dim,
            backend=tropical_backend,
            warmup=warmup,
            seed=seed,
            dtype=dtype,
            device=device,
            top_ops=top_ops,
        ),
        "pairwise": profile_family_forward(
            "pairwise",
            batch_size=batch_size,
            input_dim=input_dim,
            hidden_dim=pairwise_hidden,
            out_features=out_features,
            depth=depth,
            pairwise_tables=pairwise_tables,
            comparisons=pairwise_comparisons,
            warmup=warmup,
            seed=seed,
            dtype=dtype,
            device=device,
            top_ops=top_ops,
        ),
    }


def _print_summary(results: dict[str, Any]) -> None:
    for family, summary in results.items():
        print(
            f"[{family}] params={summary['params']} wall_ms={summary['wall_ms']:.3f} peak_device_memory_bytes={summary['peak_device_memory_bytes']}"
        )
        for row in summary["top_ops"]:
            time_us = row["device_time_us"] if summary["device"] == "cuda" else row["self_cpu_time_us"]
            print(f"  {row['name']:<36} calls={row['calls']:<4d} self_time_us={time_us:>10.1f} self_device_mem={row['self_device_memory_bytes']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile tropical and pairwise forward paths with torch.profiler.")
    for name, default in (
        ("--batch-size", 512),
        ("--input-dim", 28 * 28),
        ("--out-features", 10),
        ("--depth", 2),
        ("--warmup", 5),
        ("--seed", 0),
        ("--top-ops", 12),
    ):
        parser.add_argument(name, type=int, default=default)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--tropical-hidden", type=int, default=128)
    parser.add_argument("--tropical-heads", type=int, default=32)
    parser.add_argument("--tropical-cells", type=int, default=4)
    parser.add_argument("--tropical-code-dim", type=int, default=32)
    parser.add_argument("--tropical-backend", choices=("torch", "auto", "triton"), default="torch")
    parser.add_argument("--pairwise-hidden", type=int, default=128)
    parser.add_argument("--pairwise-tables", type=int, default=72)
    parser.add_argument("--pairwise-comparisons", type=int, default=6)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    results = profile_family_set(
        batch_size=args.batch_size,
        input_dim=args.input_dim,
        out_features=args.out_features,
        depth=args.depth,
        tropical_hidden=args.tropical_hidden,
        tropical_heads=args.tropical_heads,
        tropical_cells=args.tropical_cells,
        tropical_code_dim=args.tropical_code_dim,
        tropical_backend=args.tropical_backend,
        pairwise_hidden=args.pairwise_hidden,
        pairwise_tables=args.pairwise_tables,
        pairwise_comparisons=args.pairwise_comparisons,
        warmup=args.warmup,
        seed=args.seed,
        dtype=args.dtype,
        device=args.device,
        top_ops=args.top_ops,
    )
    _print_summary(results)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
