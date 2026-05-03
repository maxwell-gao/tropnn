"""Parallel launcher for dense Pairwise bridge sweeps.

Distributes a Cartesian product of ``(tables, comparisons)`` across multiple
GPUs, running ``tropnn.tools.scaling_benchmark`` once per configuration (all
seeds in a single invocation). The default grid is intentionally denser than
the original pairwise bridge sweep:

``tables = 4,8,16,32,64,128,256``
``comparisons = 2,3,4,5,6,7,8,9,10``

With the default six GPUs and ``--max-parallel 18``, this gives three
concurrent processes per GPU.
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
    parser = argparse.ArgumentParser(description="Launch parallel Pairwise bridge sweep.")
    parser.add_argument("--n-features", type=int, default=1024)
    parser.add_argument("--model-dim", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--tables-list", type=str, default="4,8,16,32,64,128,256")
    parser.add_argument("--comparisons-list", type=str, default="2,3,4,5,6,7,8,9,10")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5")
    parser.add_argument("--max-parallel", type=int, default=18)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tag-prefix", type=str, default="pairwise-bridge")
    parser.add_argument("--module", type=str, default="tropnn.tools.scaling_benchmark")
    return parser.parse_args()


def _run_single_config(
    *,
    tables: int,
    comparisons: int,
    gpu: int,
    args: argparse.Namespace,
) -> tuple[int, int, int, int, str, str]:
    cmd = [
        sys.executable,
        "-m",
        args.module,
        "--families",
        "tied_pairwise",
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
        "--tables",
        str(tables),
        "--comparisons",
        str(comparisons),
        "--seeds",
        args.seeds,
        "--device",
        "cuda",
        "--backend",
        "torch",
        "--tag",
        f"{args.tag_prefix}-T{tables}-L{comparisons}",
        "--output-dir",
        str(args.output_dir),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    return tables, comparisons, gpu, proc.returncode, proc.stdout[-400:], proc.stderr[-400:]


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tables_values = _parse_csv_ints(args.tables_list)
    comparisons_values = _parse_csv_ints(args.comparisons_list)
    gpus = _parse_csv_ints(args.gpus)
    if not gpus:
        raise ValueError("--gpus must list at least one GPU index")

    jobs = list(product(tables_values, comparisons_values))
    gpu_iter = cycle(gpus)
    print(f"running {len(jobs)} invocations across {len(gpus)} GPUs with {args.max_parallel} workers")

    started = time.perf_counter()
    completed = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=args.max_parallel) as pool:
        futures = []
        for tables, comparisons in jobs:
            gpu = next(gpu_iter)
            futures.append(
                pool.submit(
                    _run_single_config,
                    tables=tables,
                    comparisons=comparisons,
                    gpu=gpu,
                    args=args,
                )
            )
        for fut in as_completed(futures):
            tables, comparisons, gpu, returncode, _stdout, stderr = fut.result()
            completed += 1
            label = "OK" if returncode == 0 else "FAIL"
            print(f"[{completed}/{len(jobs)}] {label} T={tables:>4} L={comparisons:>2} gpu={gpu}")
            if returncode != 0:
                failed += 1
                print(f"  stderr tail:\n{stderr.strip()}")

    elapsed = time.perf_counter() - started
    print(f"done: {completed} completed, {failed} failed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
