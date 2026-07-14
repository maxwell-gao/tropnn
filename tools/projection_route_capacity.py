from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor

from tropnn.layers import PairwiseLUT


@dataclass(frozen=True)
class CapacityResult:
    support_kind: str
    anchor_policy: str
    input_dim: int
    support_size: int
    tables: int
    comparisons: int
    unique_routes: int
    singleton_routes: int
    nontrivial_routes: int
    max_fiber_size: int
    route_entropy_bits: float
    sampled_pairs: int
    rank_d_rho: int
    surviving_inner_product_dim: int
    surviving_fraction: float
    cov_trace: float
    cov_min_eig: float
    cov_median_eig: float
    cov_max_eig: float
    spectrum_eps: str
    spectrum_surviving_dims: str
    recovery_gaussian_mse: float
    recovery_gaussian_r2: float
    recovery_binary_mse: float
    recovery_binary_r2: float
    recovery_ternary_mse: float
    recovery_ternary_r2: float
    recovery_low_rank_mse: float
    recovery_low_rank_r2: float
    recovery_coordinate_mse: float
    recovery_coordinate_r2: float
    recovery_all_ones_mse: float
    recovery_all_ones_r2: float

    def as_dict(self) -> dict[str, object]:
        return {
            "support_kind": self.support_kind,
            "anchor_policy": self.anchor_policy,
            "input_dim": self.input_dim,
            "support_size": self.support_size,
            "tables": self.tables,
            "comparisons": self.comparisons,
            "unique_routes": self.unique_routes,
            "singleton_routes": self.singleton_routes,
            "nontrivial_routes": self.nontrivial_routes,
            "max_fiber_size": self.max_fiber_size,
            "route_entropy_bits": self.route_entropy_bits,
            "sampled_pairs": self.sampled_pairs,
            "rank_d_rho": self.rank_d_rho,
            "surviving_inner_product_dim": self.surviving_inner_product_dim,
            "surviving_fraction": self.surviving_fraction,
            "cov_trace": self.cov_trace,
            "cov_min_eig": self.cov_min_eig,
            "cov_median_eig": self.cov_median_eig,
            "cov_max_eig": self.cov_max_eig,
            "spectrum_eps": self.spectrum_eps,
            "spectrum_surviving_dims": self.spectrum_surviving_dims,
            "recovery_gaussian_mse": self.recovery_gaussian_mse,
            "recovery_gaussian_r2": self.recovery_gaussian_r2,
            "recovery_binary_mse": self.recovery_binary_mse,
            "recovery_binary_r2": self.recovery_binary_r2,
            "recovery_ternary_mse": self.recovery_ternary_mse,
            "recovery_ternary_r2": self.recovery_ternary_r2,
            "recovery_low_rank_mse": self.recovery_low_rank_mse,
            "recovery_low_rank_r2": self.recovery_low_rank_r2,
            "recovery_coordinate_mse": self.recovery_coordinate_mse,
            "recovery_coordinate_r2": self.recovery_coordinate_r2,
            "recovery_all_ones_mse": self.recovery_all_ones_mse,
            "recovery_all_ones_r2": self.recovery_all_ones_r2,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate projection route capacity for a finite support X. "
            "For route fibers rho(x)=rho(x'), collect differences x-x', "
            "estimate rank(D_rho), and report input_dim-rank(D_rho)."
        )
    )
    parser.add_argument("--support-kind", choices=["random_int", "binary", "permutation", "file"], default="random_int")
    parser.add_argument("--support-path", type=Path, default=None, help="Tensor file for support-kind=file (.pt/.pth or .npy).")
    parser.add_argument("--support-size", type=int, default=8192)
    parser.add_argument("--input-dim", type=int, default=256)
    parser.add_argument("--max-value", type=int, default=255)
    parser.add_argument("--normalize", choices=["none", "l2", "center_l2", "layernorm"], default="none")
    parser.add_argument("--tables", type=int, default=64)
    parser.add_argument("--comparisons", type=int, default=6)
    parser.add_argument("--anchor-policy", default="random", help="Single policy or comma-separated policies.")
    parser.add_argument("--anchor-seed", type=int, default=0)
    parser.add_argument("--threshold-mode", choices=["zero", "normal"], default="zero")
    parser.add_argument("--threshold-std", type=float, default=1.0)
    parser.add_argument("--pairs-per-route", type=int, default=8)
    parser.add_argument("--max-pairs", type=int, default=200_000)
    parser.add_argument("--max-fiber-store", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--rank-atol", type=float, default=1e-8)
    parser.add_argument("--rank-rtol", type=float, default=1e-5)
    parser.add_argument("--spectrum-eps", default="1e-8,1e-6,1e-4,1e-2")
    parser.add_argument("--skip-spectrum", action="store_true")
    parser.add_argument("--skip-recovery", action="store_true")
    parser.add_argument("--recovery-output-dim", type=int, default=64)
    parser.add_argument("--recovery-low-rank", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, default=None)
    return parser.parse_args()


def make_support(args: argparse.Namespace) -> Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    if args.support_kind == "file":
        if args.support_path is None:
            raise ValueError("--support-path is required when --support-kind=file")
        x = load_support(args.support_path)
    elif args.support_kind == "random_int":
        x = torch.randint(0, args.max_value + 1, (args.support_size, args.input_dim), generator=generator, dtype=torch.int64)
    elif args.support_kind == "binary":
        x = torch.randint(0, 2, (args.support_size, args.input_dim), generator=generator, dtype=torch.int64)
    elif args.support_kind == "permutation":
        template = torch.arange(args.input_dim, dtype=torch.int64)
        rows = [template[torch.randperm(args.input_dim, generator=generator)] for _ in range(args.support_size)]
        x = torch.stack(rows, dim=0)
    else:
        raise ValueError(f"unsupported support kind {args.support_kind!r}")
    if x.ndim != 2:
        raise ValueError(f"support must be a rank-2 tensor, got shape {tuple(x.shape)}")
    x = x.to(torch.float32)
    return normalize_support(x, args.normalize)


def load_support(path: Path) -> Tensor:
    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, Tensor):
            return obj
        if isinstance(obj, dict):
            for key in ("x", "support", "hidden", "states"):
                value = obj.get(key)
                if isinstance(value, Tensor):
                    return value
        raise ValueError(f"could not find a support tensor in {path}")
    if suffix == ".npy":
        import numpy as np

        return torch.from_numpy(np.load(path))
    raise ValueError(f"unsupported support file suffix {suffix!r}")


def normalize_support(x: Tensor, mode: str) -> Tensor:
    if mode == "none":
        return x
    if mode == "l2":
        return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    if mode == "center_l2":
        y = x - x.mean(dim=-1, keepdim=True)
        return y / y.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    if mode == "layernorm":
        y = x - x.mean(dim=-1, keepdim=True)
        return y / y.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-12)
    raise ValueError(f"unsupported normalize mode {mode!r}")


def configure_layer(args: argparse.Namespace, policy: str, input_dim: int) -> PairwiseLUT:
    layer = PairwiseLUT(
        input_dim=input_dim,
        output_dim=1,
        tables=args.tables,
        comparisons=args.comparisons,
        backend="torch",
        seed=args.seed,
        anchor_policy=policy,
        anchor_seed=args.anchor_seed,
        fixed_zero_threshold=args.threshold_mode == "zero",
    )
    if args.threshold_mode == "normal":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(args.seed + 17)
        values = torch.randn(args.tables, args.comparisons, generator=generator) * args.threshold_std
        with torch.no_grad():
            layer.thresholds.copy_(values)
    return layer


def route_key(row: Tensor) -> bytes:
    return row.to(dtype=torch.int32, device="cpu", copy=True).numpy().tobytes()


def collect_route_fibers(
    layer: PairwiseLUT,
    x: Tensor,
    *,
    batch_size: int,
    max_fiber_store: int,
    device: torch.device,
) -> tuple[dict[bytes, int], dict[bytes, list[int]], dict[bytes, Tensor], list[bytes]]:
    counts: dict[bytes, int] = defaultdict(int)
    stored: dict[bytes, list[int]] = defaultdict(list)
    sums: dict[bytes, Tensor] = {}
    row_keys: list[bytes] = [b""] * x.shape[0]
    layer = layer.to(device)
    layer.eval()
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            batch = x[start : start + batch_size].to(device)
            route = layer.route(batch.unsqueeze(1))
            codes = route.indices[:, 0, :]
            for offset, code in enumerate(codes):
                key = route_key(code)
                row_index = start + offset
                row_keys[row_index] = key
                counts[key] += 1
                row = x[row_index].to(torch.float64)
                if key in sums:
                    sums[key] += row
                else:
                    sums[key] = row.clone()
                bucket = stored[key]
                if len(bucket) < max_fiber_store:
                    bucket.append(row_index)
    return dict(counts), dict(stored), sums, row_keys


def route_entropy_bits(fiber_sizes: Iterable[int], total: int) -> float:
    entropy = 0.0
    for size in fiber_sizes:
        p = size / total
        entropy -= p * math.log2(p)
    return entropy


def sample_fiber_differences(
    x: Tensor,
    stored: dict[bytes, list[int]],
    *,
    pairs_per_route: int,
    max_pairs: int,
    seed: int,
) -> Tensor:
    rng = random.Random(seed)
    diffs: list[Tensor] = []
    for indices in stored.values():
        if len(indices) < 2:
            continue
        trials = min(pairs_per_route, max_pairs - len(diffs))
        for _ in range(trials):
            i, j = rng.sample(indices, 2)
            diffs.append(x[i] - x[j])
        if len(diffs) >= max_pairs:
            break
    if not diffs:
        return x.new_zeros((0, x.shape[1]))
    return torch.stack(diffs, dim=0)


def estimate_rank(diffs: Tensor, *, atol: float, rtol: float) -> int:
    if diffs.numel() == 0:
        return 0
    return int(torch.linalg.matrix_rank(diffs.to(torch.float64), atol=atol, rtol=rtol).item())


def parse_eps_list(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("--spectrum-eps must contain at least one numeric tolerance")
    return values


def route_means(counts: dict[bytes, int], sums: dict[bytes, Tensor]) -> dict[bytes, Tensor]:
    return {key: sums[key] / float(count) for key, count in counts.items()}


def within_fiber_covariance(x: Tensor, counts: dict[bytes, int], sums: dict[bytes, Tensor]) -> Tensor:
    x64 = x.to(torch.float64)
    second = x64.T @ x64
    explained = torch.zeros_like(second)
    for key, count in counts.items():
        s = sums[key]
        explained += torch.outer(s, s) / float(count)
    cov = (second - explained) / float(x.shape[0])
    return 0.5 * (cov + cov.T)


def covariance_spectrum_metrics(cov: Tensor, eps_values: list[float]) -> tuple[dict[str, float], dict[str, int]]:
    eigvals = torch.linalg.eigvalsh(cov).clamp_min(0.0)
    metrics = {
        "trace": float(eigvals.sum().item()),
        "min": float(eigvals.min().item()) if eigvals.numel() else 0.0,
        "median": float(eigvals.median().item()) if eigvals.numel() else 0.0,
        "max": float(eigvals.max().item()) if eigvals.numel() else 0.0,
    }
    dims = {format_eps(eps): int((eigvals < eps).sum().item()) for eps in eps_values}
    return metrics, dims


def format_eps(value: float) -> str:
    return f"{value:.0e}" if value != 0 else "0"


def nan_recovery_metrics() -> dict[str, tuple[float, float]]:
    return {kind: (float("nan"), float("nan")) for kind in ("gaussian", "binary", "ternary", "low_rank", "coordinate", "all_ones")}


def make_recovery_matrices(args: argparse.Namespace, input_dim: int) -> dict[str, Tensor]:
    out_dim = min(args.recovery_output_dim, input_dim)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed + 101)
    gaussian = torch.randn(out_dim, input_dim, generator=generator, dtype=torch.float64) / math.sqrt(input_dim)
    binary = (torch.randint(0, 2, (out_dim, input_dim), generator=generator, dtype=torch.int64).to(torch.float64) * 2.0 - 1.0) / math.sqrt(input_dim)
    ternary_raw = torch.randint(0, 3, (out_dim, input_dim), generator=generator, dtype=torch.int64) - 1
    ternary = ternary_raw.to(torch.float64) / math.sqrt(max(1, input_dim * 2 // 3))
    rank = max(1, min(args.recovery_low_rank, input_dim, out_dim))
    left = torch.randn(out_dim, rank, generator=generator, dtype=torch.float64) / math.sqrt(rank)
    right = torch.randn(rank, input_dim, generator=generator, dtype=torch.float64) / math.sqrt(input_dim)
    low_rank = left @ right
    coordinate = torch.eye(input_dim, dtype=torch.float64)[:out_dim]
    all_ones = torch.ones(1, input_dim, dtype=torch.float64) / math.sqrt(input_dim)
    return {
        "gaussian": gaussian,
        "binary": binary,
        "ternary": ternary,
        "low_rank": low_rank,
        "coordinate": coordinate,
        "all_ones": all_ones,
    }


def representative_matrix(row_keys: list[bytes], means: dict[bytes, Tensor]) -> Tensor:
    return torch.stack([means[key] for key in row_keys], dim=0)


def matrix_recovery_metrics(x: Tensor, x_hat: Tensor, matrices: dict[str, Tensor]) -> dict[str, tuple[float, float]]:
    x64 = x.to(torch.float64)
    metrics: dict[str, tuple[float, float]] = {}
    for kind, w in matrices.items():
        y = x64 @ w.T
        pred = x_hat @ w.T
        mse = float(((y - pred) ** 2).mean().item())
        centered = y - y.mean(dim=0, keepdim=True)
        var = float((centered**2).mean().item())
        if var < 1e-24:
            r2 = 1.0 if mse < 1e-24 else float("nan")
        else:
            r2 = 1.0 - mse / var
        metrics[kind] = (mse, float(r2))
    return metrics


def estimate_capacity(args: argparse.Namespace, x: Tensor, policy: str) -> CapacityResult:
    device = torch.device(args.device)
    layer = configure_layer(args, policy, input_dim=x.shape[1])
    counts, stored, sums, row_keys = collect_route_fibers(layer, x, batch_size=args.batch_size, max_fiber_store=args.max_fiber_store, device=device)
    sizes = list(counts.values())
    diffs = sample_fiber_differences(x, stored, pairs_per_route=args.pairs_per_route, max_pairs=args.max_pairs, seed=args.seed)
    rank = estimate_rank(diffs, atol=args.rank_atol, rtol=args.rank_rtol)
    surviving = x.shape[1] - rank
    eps_values = parse_eps_list(args.spectrum_eps)
    if args.skip_spectrum:
        spectrum = {"trace": float("nan"), "min": float("nan"), "median": float("nan"), "max": float("nan")}
        spectrum_dims = {format_eps(eps): -1 for eps in eps_values}
    else:
        spectrum, spectrum_dims = covariance_spectrum_metrics(within_fiber_covariance(x, counts, sums), eps_values)
    if args.skip_recovery:
        recovery = nan_recovery_metrics()
    else:
        means = route_means(counts, sums)
        x_hat = representative_matrix(row_keys, means)
        recovery = matrix_recovery_metrics(x, x_hat, make_recovery_matrices(args, x.shape[1]))
    return CapacityResult(
        support_kind=args.support_kind,
        anchor_policy=policy,
        input_dim=x.shape[1],
        support_size=x.shape[0],
        tables=args.tables,
        comparisons=args.comparisons,
        unique_routes=len(counts),
        singleton_routes=sum(1 for size in sizes if size == 1),
        nontrivial_routes=sum(1 for size in sizes if size > 1),
        max_fiber_size=max(sizes) if sizes else 0,
        route_entropy_bits=route_entropy_bits(sizes, x.shape[0]) if sizes else 0.0,
        sampled_pairs=diffs.shape[0],
        rank_d_rho=rank,
        surviving_inner_product_dim=surviving,
        surviving_fraction=surviving / x.shape[1],
        cov_trace=spectrum["trace"],
        cov_min_eig=spectrum["min"],
        cov_median_eig=spectrum["median"],
        cov_max_eig=spectrum["max"],
        spectrum_eps=json.dumps([format_eps(eps) for eps in eps_values]),
        spectrum_surviving_dims=json.dumps(spectrum_dims, sort_keys=True),
        recovery_gaussian_mse=recovery["gaussian"][0],
        recovery_gaussian_r2=recovery["gaussian"][1],
        recovery_binary_mse=recovery["binary"][0],
        recovery_binary_r2=recovery["binary"][1],
        recovery_ternary_mse=recovery["ternary"][0],
        recovery_ternary_r2=recovery["ternary"][1],
        recovery_low_rank_mse=recovery["low_rank"][0],
        recovery_low_rank_r2=recovery["low_rank"][1],
        recovery_coordinate_mse=recovery["coordinate"][0],
        recovery_coordinate_r2=recovery["coordinate"][1],
        recovery_all_ones_mse=recovery["all_ones"][0],
        recovery_all_ones_r2=recovery["all_ones"][1],
    )


def write_csv(path: Path, rows: list[CapacityResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].as_dict().keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())


def write_json(path: Path, rows: list[CapacityResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([row.as_dict() for row in rows], indent=2) + "\n")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    x = make_support(args)
    policies = [item.strip() for item in args.anchor_policy.split(",") if item.strip()]
    results = [estimate_capacity(args, x, policy) for policy in policies]
    for result in results:
        print(json.dumps(result.as_dict(), sort_keys=True))
    if args.out_csv is not None:
        write_csv(args.out_csv, results)
    if args.out_json is not None:
        write_json(args.out_json, results)


if __name__ == "__main__":
    main()
