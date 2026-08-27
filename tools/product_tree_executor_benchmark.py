"""Bit-exact latency/work benchmark for frozen depth-four product trees."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor

from tropnn.layers.hard_lookup import HardLookupSpec, adaptive_hard_route_lookahead, hard_route, sum_lookup_rows


@dataclass(frozen=True)
class BenchmarkRow:
    device: str
    executor: str
    lookahead: int
    comparison_rounds: int
    comparisons_per_table: int
    batch_size: int
    iterations: int
    route_ms: float
    forward_ms: float
    route_items_per_second: float
    forward_items_per_second: float


def _parse_ints(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item) for item in value.split(",") if item)
    if not parsed or any(item < 1 for item in parsed):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return parsed


def _comparison_work(depth: int, lookahead: int) -> tuple[int, int]:
    widths = [min(lookahead, depth - start) for start in range(0, depth, lookahead)]
    return len(widths), sum(2**width - 1 for width in widths)


def _nmse(prediction: Tensor, target: Tensor) -> float:
    return float((prediction - target).square().mean() / target.square().mean().clamp_min(1e-30))


def _time_ms(fn, *, device: torch.device, warmups: int, iterations: int, repeats: int) -> float:
    samples: list[float] = []
    for _ in range(warmups):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    for _ in range(repeats):
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                fn()
            end.record()
            torch.cuda.synchronize(device)
            samples.append(float(start.elapsed_time(end) / iterations))
        else:
            started = time.perf_counter()
            for _ in range(iterations):
                fn()
            samples.append(1000.0 * (time.perf_counter() - started) / iterations)
    return float(statistics.median(samples))


def _seed_state(bundle: dict[str, object], seed: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    state = bundle["state"]
    if not isinstance(state, dict):
        raise TypeError("artifact state is missing")
    prefix = f"seed{seed}."
    teacher = state[prefix + "teacher"]
    supports = state[prefix + "product_free_task.supports"]
    thresholds = state[prefix + "product_free_task.thresholds"]
    rows = state[prefix + "product_free_task.rows"]
    if not all(isinstance(item, Tensor) for item in (teacher, supports, thresholds, rows)):
        raise TypeError("artifact tensors are malformed")
    return teacher, supports, thresholds, rows


def verify_exact_executors(
    bundle: dict[str, object],
    *,
    seeds: tuple[int, ...],
    samples: int,
) -> dict[str, object]:
    checks: dict[str, object] = {}
    for seed in seeds:
        teacher, supports, thresholds, rows = _seed_state(bundle, seed)
        dim, depth = int(teacher.shape[0]), int(supports.shape[1])
        spec = HardLookupSpec(dim, dim, depth, "unary", "adaptive", support_layout="level", surrogate="none")
        x = torch.randn(samples, dim, generator=torch.Generator().manual_seed(90_000 + seed))
        reference = hard_route(x, supports, thresholds, spec)
        reference_output = sum_lookup_rows(rows, reference.codes)
        target = x @ teacher.T
        seed_checks: dict[str, object] = {
            "reference_r2": 1.0 - _nmse(reference_output, target),
        }
        for lookahead in (1, 2, depth):
            route = adaptive_hard_route_lookahead(x, supports, thresholds, spec, lookahead)
            output = sum_lookup_rows(rows, route.codes)
            rounds, comparisons = _comparison_work(depth, lookahead)
            seed_checks[f"lookahead_{lookahead}"] = {
                "comparison_rounds": rounds,
                "comparisons_per_table": comparisons,
                "code_mismatches": int((route.codes != reference.codes).sum()),
                "branch_mismatches": int((route.branches != reference.branches).sum()),
                "margin_max_abs_difference": float((route.margins - reference.margins).abs().max()),
                "output_max_abs_difference": float((output - reference_output).abs().max()),
                "r2": 1.0 - _nmse(output, target),
            }
            if not torch.equal(route.codes, reference.codes) or not torch.equal(output, reference_output):
                raise AssertionError(f"lookahead={lookahead} is not bit/output exact")
        checks[str(seed)] = seed_checks
    return checks


def benchmark(
    bundle: dict[str, object],
    *,
    device: torch.device,
    batch_sizes: tuple[int, ...],
    warmups: int,
    iterations: int,
    repeats: int,
    target_items: int,
) -> list[BenchmarkRow]:
    teacher, supports, thresholds, rows = _seed_state(bundle, 0)
    dim, depth = int(teacher.shape[0]), int(supports.shape[1])
    supports = supports.to(device)
    thresholds = thresholds.to(device)
    rows = rows.to(device)
    spec = HardLookupSpec(dim, dim, depth, "unary", "adaptive", support_layout="level", surrogate="none")
    result: list[BenchmarkRow] = []
    for batch_size in batch_sizes:
        x = torch.randn(batch_size, dim, generator=torch.Generator().manual_seed(91_000 + batch_size)).to(device)
        count = max(10, min(iterations, max(1, target_items // batch_size)))
        for lookahead in (1, 2, depth):
            rounds, comparisons = _comparison_work(depth, lookahead)

            def execute_route() -> object:
                if lookahead == 1:
                    return hard_route(x, supports, thresholds, spec)
                return adaptive_hard_route_lookahead(x, supports, thresholds, spec, lookahead)

            def route_once() -> Tensor:
                return execute_route().codes  # type: ignore[union-attr]

            def forward_once() -> Tensor:
                codes = execute_route().codes  # type: ignore[union-attr]
                return sum_lookup_rows(rows, codes)

            with torch.inference_mode():
                route_ms = _time_ms(route_once, device=device, warmups=warmups, iterations=count, repeats=repeats)
                forward_ms = _time_ms(forward_once, device=device, warmups=warmups, iterations=count, repeats=repeats)
            result.append(
                BenchmarkRow(
                    device=str(device),
                    executor=f"lookahead_{lookahead}",
                    lookahead=lookahead,
                    comparison_rounds=rounds,
                    comparisons_per_table=comparisons,
                    batch_size=batch_size,
                    iterations=count,
                    route_ms=route_ms,
                    forward_ms=forward_ms,
                    route_items_per_second=1000.0 * batch_size / route_ms,
                    forward_items_per_second=1000.0 * batch_size / forward_ms,
                )
            )
    return result


def run(args: argparse.Namespace) -> dict[str, object]:
    source = Path(args.source_artifact)
    bundle = torch.load(source, map_location="cpu", weights_only=False)
    if bundle.get("schema") != "product-atlas-pc-action-factorial-v1":
        raise ValueError("unexpected source artifact schema")
    protocol = bundle.get("protocol")
    if not isinstance(protocol, dict) or protocol.get("dim") != 64 or protocol.get("tables") != 32 or protocol.get("depth") != 4:
        raise ValueError("source artifact is not the frozen D64/T32/depth4 result")
    exact = verify_exact_executors(bundle, seeds=(0, 1, 2), samples=args.exact_samples)
    original_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(args.cpu_threads)
        cpu_rows = benchmark(
            bundle,
            device=torch.device("cpu"),
            batch_sizes=args.batch_sizes,
            warmups=args.warmups,
            iterations=args.iterations,
            repeats=args.repeats,
            target_items=args.target_items,
        )
    finally:
        torch.set_num_threads(original_threads)
    cuda_rows: list[BenchmarkRow] = []
    if args.device.startswith("cuda"):
        cuda_rows = benchmark(
            bundle,
            device=torch.device(args.device),
            batch_sizes=args.batch_sizes,
            warmups=args.warmups,
            iterations=args.iterations,
            repeats=args.repeats,
            target_items=args.target_items,
        )
    return {
        "schema": "product-tree-executor-benchmark-v1",
        "source_artifact": str(source.resolve()),
        "source_artifact_size": source.stat().st_size,
        "protocol": {
            "dim": 64,
            "tables": 32,
            "depth": 4,
            "rows_per_table": 16,
            "dtype": "float32",
            "batch_sizes": list(args.batch_sizes),
            "warmups": args.warmups,
            "maximum_iterations": args.iterations,
            "repeats": args.repeats,
            "cpu_threads": args.cpu_threads,
            "cuda_reference": args.device,
            "timing_scope": "eager_Torch_semantic_reference_route_and_route_plus_lookup_add",
        },
        "exactness": exact,
        "rows": [asdict(row) for row in (*cpu_rows, *cuda_rows)],
        "ledger": {
            "lookahead_1": {"comparison_rounds": 4, "comparisons_per_table": 4, "total_comparisons": 128},
            "lookahead_2": {"comparison_rounds": 2, "comparisons_per_table": 6, "total_comparisons": 192},
            "lookahead_4": {"comparison_rounds": 1, "comparisons_per_table": 15, "total_comparisons": 480},
            "shared": {"active_row_reads": 32, "active_output_scalar_reads": 2048},
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark bit-exact speculative executors for one frozen product tree")
    parser.add_argument("--source-artifact", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-sizes", type=_parse_ints, default=(1, 32, 1024, 8192))
    parser.add_argument("--exact-samples", type=int, default=8192)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--target-items", type=int, default=1_000_000)
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args()
    if min(args.exact_samples, args.warmups, args.iterations, args.repeats, args.target_items, args.cpu_threads) < 1:
        parser.error("benchmark counts must be positive")
    if Path(args.output).exists():
        parser.error("output must not exist")
    return args


def main() -> None:
    args = parse_args()
    result = run(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"exactness": result["exactness"], "ledger": result["ledger"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
