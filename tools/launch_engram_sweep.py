"""Parallel launcher for the Engram bridge sweep.

Distributes a Cartesian product of ``(engram_heads, engram_table_size)`` configs
across multiple GPUs, running ``tropnn.tools.scaling_benchmark`` once per
config (with all seeds in one invocation). A thread pool with bounded
parallelism manages the GPU pinning and queueing.

Each invocation produces its own ``runs-*.csv`` and ``summary-*.json`` in
``--output-dir``. Use the post-hoc aggregator below (or any JSON merger) to
fit bridge exponents across the full grid.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import cycle, product
from pathlib import Path


def _parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch parallel Engram bridge sweep.")
    parser.add_argument("--n-features", type=int, default=1024)
    parser.add_argument("--model-dim", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--engram-heads-list", type=str, default="1,2,4,8,16,32")
    parser.add_argument("--engram-table-sizes", type=str, default="8,16,32,64,128,256")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5")
    parser.add_argument("--max-parallel", type=int, default=18)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tag-prefix", type=str, default="engram-bridge")
    parser.add_argument("--module", type=str, default="tropnn.tools.scaling_benchmark")
    return parser.parse_args()


def _run_single_config(
    *,
    engram_heads: int,
    engram_table_size: int,
    gpu: int,
    args: argparse.Namespace,
) -> tuple[int, int, int, int, str, str]:
    cmd = [
        sys.executable,
        "-m",
        args.module,
        "--families",
        "tied_engram",
        "--n-features",
        str(args.n_features),
        "--model-dims",
        str(args.model_dim),
        "--alphas",
        str(args.alpha),
        "--batch-size",
        str(args.batch_size),
        "--steps",
        str(args.steps),
        "--heads",
        "32",
        "--cells",
        "4",
        "--route-terms",
        "4",
        "--engram-heads",
        str(engram_heads),
        "--engram-table-size",
        str(engram_table_size),
        "--seeds",
        args.seeds,
        "--device",
        "cuda",
        "--backend",
        "torch",
        "--tag",
        f"{args.tag_prefix}-K{engram_heads}-M{engram_table_size}",
        "--output-dir",
        str(args.output_dir),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    return (
        engram_heads,
        engram_table_size,
        gpu,
        proc.returncode,
        proc.stdout[-400:],
        proc.stderr[-400:],
    )


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    heads_values = _parse_csv_ints(args.engram_heads_list)
    table_sizes = _parse_csv_ints(args.engram_table_sizes)
    gpus = _parse_csv_ints(args.gpus)
    if not gpus:
        raise ValueError("--gpus must list at least one GPU index")

    jobs = list(product(heads_values, table_sizes))
    gpu_iter = cycle(gpus)
    print(f"running {len(jobs)} invocations across {len(gpus)} GPUs with {args.max_parallel} workers")

    started = time.perf_counter()
    completed = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=args.max_parallel) as pool:
        futures = []
        for engram_heads, engram_table_size in jobs:
            gpu = next(gpu_iter)
            futures.append(
                pool.submit(
                    _run_single_config,
                    engram_heads=engram_heads,
                    engram_table_size=engram_table_size,
                    gpu=gpu,
                    args=args,
                )
            )
        for fut in as_completed(futures):
            engram_heads, engram_table_size, gpu, returncode, _stdout, stderr = fut.result()
            completed += 1
            label = "OK" if returncode == 0 else "FAIL"
            print(f"[{completed}/{len(jobs)}] {label} K={engram_heads:>3} M={engram_table_size:>4} gpu={gpu}")
            if returncode != 0:
                failed += 1
                print(f"  stderr tail:\n{stderr.strip()}")

    elapsed = time.perf_counter() - started
    print(f"done: {completed} completed, {failed} failed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
