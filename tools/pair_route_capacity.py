from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from tropnn.layers import PairwiseLUT


@dataclass(frozen=True)
class PairCapacityResult:
    support_kind: str
    pair_mode: str
    anchor_policy: str
    input_dim: int
    bilinear_dim: int
    support_size: int
    pair_support_size: int
    tables: int
    comparisons: int
    unique_routes: int
    singleton_routes: int
    nontrivial_routes: int
    max_fiber_size: int
    route_entropy_bits: float
    sampled_fiber_pairs: int
    rank_d_pair: int
    surviving_bilinear_dim: int
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
            "pair_mode": self.pair_mode,
            "anchor_policy": self.anchor_policy,
            "input_dim": self.input_dim,
            "bilinear_dim": self.bilinear_dim,
            "support_size": self.support_size,
            "pair_support_size": self.pair_support_size,
            "tables": self.tables,
            "comparisons": self.comparisons,
            "unique_routes": self.unique_routes,
            "singleton_routes": self.singleton_routes,
            "nontrivial_routes": self.nontrivial_routes,
            "max_fiber_size": self.max_fiber_size,
            "route_entropy_bits": self.route_entropy_bits,
            "sampled_fiber_pairs": self.sampled_fiber_pairs,
            "rank_d_pair": self.rank_d_pair,
            "surviving_bilinear_dim": self.surviving_bilinear_dim,
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
            "Estimate pair-space route capacity. For pairs (x,y), route concat[x,y], "
            "collect same-route differences x⊗y - x'⊗y', estimate rank(D_pair), "
            "within-fiber covariance spectrum, and x^T A y recovery."
        )
    )
    parser.add_argument("--support-kind", choices=["random_int", "binary", "permutation", "file"], default="random_int")
    parser.add_argument("--support-path", type=Path, default=None)
    parser.add_argument("--support-size", type=int, default=2048)
    parser.add_argument("--pair-support-size", type=int, default=4096)
    parser.add_argument("--pair-mode", choices=["random", "diagonal", "cartesian_prefix"], default="random")
    parser.add_argument("--input-dim", type=int, default=32)
    parser.add_argument("--max-value", type=int, default=15)
    parser.add_argument("--normalize", choices=["none", "l2", "center_l2", "layernorm"], default="none")
    parser.add_argument("--max-bilinear-dim", type=int, default=4096)
    parser.add_argument("--tables", type=int, default=16)
    parser.add_argument("--comparisons", type=int, default=5)
    parser.add_argument("--anchor-policy", default="random", help="Single policy or comma-separated policies.")
    parser.add_argument("--anchor-seed", type=int, default=0)
    parser.add_argument("--threshold-mode", choices=["zero", "normal"], default="zero")
    parser.add_argument("--threshold-std", type=float, default=1.0)
    parser.add_argument("--pairs-per-route", type=int, default=8)
    parser.add_argument("--max-fiber-pairs", type=int, default=20000)
    parser.add_argument("--max-fiber-store", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--rank-atol", type=float, default=1e-8)
    parser.add_argument("--rank-rtol", type=float, default=1e-5)
    parser.add_argument("--spectrum-eps", default="1e-8,1e-6,1e-4,1e-2")
    parser.add_argument("--skip-spectrum", action="store_true")
    parser.add_argument("--skip-recovery", action="store_true")
    parser.add_argument("--recovery-output-dim", type=int, default=32)
    parser.add_argument("--recovery-low-rank", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, default=None)
    return parser.parse_args()


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
    raise ValueError(f"unsupported support suffix {suffix!r}")


def normalize_support(x: Tensor, mode: str) -> Tensor:
    x = x.to(torch.float32)
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


def make_support(args: argparse.Namespace) -> Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    if args.support_kind == "file":
        if args.support_path is None:
            raise ValueError("--support-path is required with --support-kind=file")
        x = load_support(args.support_path)
    elif args.support_kind == "random_int":
        x = torch.randint(0, args.max_value + 1, (args.support_size, args.input_dim), generator=generator, dtype=torch.int64)
    elif args.support_kind == "binary":
        x = torch.randint(0, 2, (args.support_size, args.input_dim), generator=generator, dtype=torch.int64)
    elif args.support_kind == "permutation":
        template = torch.arange(args.input_dim, dtype=torch.int64)
        x = torch.stack([template[torch.randperm(args.input_dim, generator=generator)] for _ in range(args.support_size)], dim=0)
    else:
        raise ValueError(f"unsupported support kind {args.support_kind!r}")
    if x.ndim != 2:
        raise ValueError(f"support must be rank-2, got {tuple(x.shape)}")
    return normalize_support(x, args.normalize)


def make_pair_indices(args: argparse.Namespace, support_size: int) -> tuple[Tensor, Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed + 11)
    n = args.pair_support_size
    if args.pair_mode == "random":
        left = torch.randint(0, support_size, (n,), generator=generator)
        right = torch.randint(0, support_size, (n,), generator=generator)
        return left, right
    if args.pair_mode == "diagonal":
        base = torch.randint(0, support_size, (n,), generator=generator)
        return base, base.clone()
    if args.pair_mode == "cartesian_prefix":
        side = int(math.ceil(math.sqrt(n)))
        values = torch.arange(min(side, support_size), dtype=torch.long)
        grid_left, grid_right = torch.meshgrid(values, values, indexing="ij")
        return grid_left.reshape(-1)[:n], grid_right.reshape(-1)[:n]
    raise ValueError(f"unsupported pair mode {args.pair_mode!r}")


def bilinear_features(x: Tensor, left: Tensor, right: Tensor, max_bilinear_dim: int) -> Tensor:
    dim = x.shape[1]
    bilinear_dim = dim * dim
    if bilinear_dim > max_bilinear_dim:
        raise ValueError(f"bilinear_dim={bilinear_dim} exceeds --max-bilinear-dim={max_bilinear_dim}")
    a = x[left].to(torch.float64)
    b = x[right].to(torch.float64)
    return (a[:, :, None] * b[:, None, :]).reshape(left.numel(), bilinear_dim)


def route_key(row: Tensor) -> bytes:
    return row.to(dtype=torch.int32, device="cpu", copy=True).numpy().tobytes()


def configure_layer(args: argparse.Namespace, policy: str, route_input_dim: int) -> PairwiseLUT:
    layer = PairwiseLUT(
        input_dim=route_input_dim,
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


def collect_pair_route_fibers(
    layer: PairwiseLUT,
    x: Tensor,
    z: Tensor,
    left: Tensor,
    right: Tensor,
    *,
    batch_size: int,
    max_fiber_store: int,
    device: torch.device,
) -> tuple[dict[bytes, int], dict[bytes, list[int]], dict[bytes, Tensor], list[bytes]]:
    counts: dict[bytes, int] = defaultdict(int)
    stored: dict[bytes, list[int]] = defaultdict(list)
    sums: dict[bytes, Tensor] = {}
    row_keys: list[bytes] = [b""] * left.numel()
    layer = layer.to(device)
    layer.eval()
    with torch.no_grad():
        for start in range(0, left.numel(), batch_size):
            sl = slice(start, min(start + batch_size, left.numel()))
            pair_input = torch.cat([x[left[sl]], x[right[sl]]], dim=-1).to(device)
            route = layer.route(pair_input.unsqueeze(1))
            codes = route.indices[:, 0, :]
            for offset, code in enumerate(codes):
                row_index = start + offset
                key = route_key(code)
                row_keys[row_index] = key
                counts[key] += 1
                row = z[row_index]
                if key in sums:
                    sums[key] += row
                else:
                    sums[key] = row.clone()
                bucket = stored[key]
                if len(bucket) < max_fiber_store:
                    bucket.append(row_index)
    return dict(counts), dict(stored), sums, row_keys


def entropy_bits(sizes: list[int], total: int) -> float:
    entropy = 0.0
    for size in sizes:
        p = size / total
        entropy -= p * math.log2(p)
    return entropy


def sample_fiber_differences(z: Tensor, stored: dict[bytes, list[int]], *, pairs_per_route: int, max_pairs: int, seed: int) -> Tensor:
    rng = random.Random(seed)
    diffs: list[Tensor] = []
    for indices in stored.values():
        if len(indices) < 2:
            continue
        trials = min(pairs_per_route, max_pairs - len(diffs))
        for _ in range(trials):
            i, j = rng.sample(indices, 2)
            diffs.append(z[i] - z[j])
        if len(diffs) >= max_pairs:
            break
    if not diffs:
        return z.new_zeros((0, z.shape[1]))
    return torch.stack(diffs, dim=0)


def estimate_rank(diffs: Tensor, *, atol: float, rtol: float) -> int:
    if diffs.numel() == 0:
        return 0
    return int(torch.linalg.matrix_rank(diffs, atol=atol, rtol=rtol).item())


def route_means(counts: dict[bytes, int], sums: dict[bytes, Tensor]) -> dict[bytes, Tensor]:
    return {key: sums[key] / float(count) for key, count in counts.items()}


def representative_matrix(row_keys: list[bytes], means: dict[bytes, Tensor]) -> Tensor:
    return torch.stack([means[key] for key in row_keys], dim=0)


def within_fiber_covariance(z: Tensor, counts: dict[bytes, int], sums: dict[bytes, Tensor]) -> Tensor:
    second = z.T @ z
    explained = torch.zeros_like(second)
    for key, count in counts.items():
        s = sums[key]
        explained += torch.outer(s, s) / float(count)
    cov = (second - explained) / float(z.shape[0])
    return 0.5 * (cov + cov.T)


def parse_eps_list(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("--spectrum-eps must contain at least one value")
    return values


def format_eps(value: float) -> str:
    return f"{value:.0e}" if value != 0 else "0"


def spectrum_metrics(cov: Tensor, eps_values: list[float]) -> tuple[dict[str, float], dict[str, int]]:
    eigvals = torch.linalg.eigvalsh(cov).clamp_min(0.0)
    metrics = {
        "trace": float(eigvals.sum().item()),
        "min": float(eigvals.min().item()) if eigvals.numel() else 0.0,
        "median": float(eigvals.median().item()) if eigvals.numel() else 0.0,
        "max": float(eigvals.max().item()) if eigvals.numel() else 0.0,
    }
    dims = {format_eps(eps): int((eigvals < eps).sum().item()) for eps in eps_values}
    return metrics, dims


def make_bilinear_matrices(args: argparse.Namespace, dim: int) -> dict[str, Tensor]:
    out_dim = min(args.recovery_output_dim, dim * dim)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed + 101)
    gaussian = torch.randn(out_dim, dim, dim, generator=generator, dtype=torch.float64) / dim
    binary = (torch.randint(0, 2, (out_dim, dim, dim), generator=generator).to(torch.float64) * 2.0 - 1.0) / dim
    ternary_raw = torch.randint(0, 3, (out_dim, dim, dim), generator=generator, dtype=torch.int64) - 1
    ternary = ternary_raw.to(torch.float64) / math.sqrt(max(1, dim * dim * 2 // 3))
    rank = max(1, min(args.recovery_low_rank, dim))
    u = torch.randn(out_dim, rank, dim, generator=generator, dtype=torch.float64) / math.sqrt(rank)
    v = torch.randn(out_dim, rank, dim, generator=generator, dtype=torch.float64) / math.sqrt(dim)
    low_rank = torch.einsum("hrd,hre->hde", u, v)
    coordinate = torch.eye(dim * dim, dtype=torch.float64)[:out_dim].reshape(out_dim, dim, dim)
    all_ones = torch.ones(1, dim, dim, dtype=torch.float64) / dim
    return {
        "gaussian": gaussian.reshape(gaussian.shape[0], dim * dim),
        "binary": binary.reshape(binary.shape[0], dim * dim),
        "ternary": ternary.reshape(ternary.shape[0], dim * dim),
        "low_rank": low_rank.reshape(low_rank.shape[0], dim * dim),
        "coordinate": coordinate.reshape(coordinate.shape[0], dim * dim),
        "all_ones": all_ones.reshape(1, dim * dim),
    }


def nan_recovery_metrics() -> dict[str, tuple[float, float]]:
    return {kind: (float("nan"), float("nan")) for kind in ("gaussian", "binary", "ternary", "low_rank", "coordinate", "all_ones")}


def recovery_metrics(z: Tensor, z_hat: Tensor, matrices: dict[str, Tensor]) -> dict[str, tuple[float, float]]:
    metrics: dict[str, tuple[float, float]] = {}
    for kind, a in matrices.items():
        y = z @ a.T
        pred = z_hat @ a.T
        mse = float(((y - pred) ** 2).mean().item())
        centered = y - y.mean(dim=0, keepdim=True)
        var = float((centered**2).mean().item())
        if var < 1e-24:
            r2 = 1.0 if mse < 1e-24 else float("nan")
        else:
            r2 = 1.0 - mse / var
        metrics[kind] = (mse, float(r2))
    return metrics


def estimate_capacity(args: argparse.Namespace, x: Tensor, left: Tensor, right: Tensor, z: Tensor, policy: str) -> PairCapacityResult:
    device = torch.device(args.device)
    layer = configure_layer(args, policy, route_input_dim=x.shape[1] * 2)
    counts, stored, sums, row_keys = collect_pair_route_fibers(
        layer,
        x,
        z,
        left,
        right,
        batch_size=args.batch_size,
        max_fiber_store=args.max_fiber_store,
        device=device,
    )
    sizes = list(counts.values())
    diffs = sample_fiber_differences(z, stored, pairs_per_route=args.pairs_per_route, max_pairs=args.max_fiber_pairs, seed=args.seed)
    rank = estimate_rank(diffs, atol=args.rank_atol, rtol=args.rank_rtol)
    surviving = z.shape[1] - rank
    eps_values = parse_eps_list(args.spectrum_eps)
    if args.skip_spectrum:
        spectrum = {"trace": float("nan"), "min": float("nan"), "median": float("nan"), "max": float("nan")}
        spectrum_dims = {format_eps(eps): -1 for eps in eps_values}
    else:
        spectrum, spectrum_dims = spectrum_metrics(within_fiber_covariance(z, counts, sums), eps_values)
    if args.skip_recovery:
        recovery = nan_recovery_metrics()
    else:
        means = route_means(counts, sums)
        z_hat = representative_matrix(row_keys, means)
        recovery = recovery_metrics(z, z_hat, make_bilinear_matrices(args, x.shape[1]))
    return PairCapacityResult(
        support_kind=args.support_kind,
        pair_mode=args.pair_mode,
        anchor_policy=policy,
        input_dim=x.shape[1],
        bilinear_dim=z.shape[1],
        support_size=x.shape[0],
        pair_support_size=z.shape[0],
        tables=args.tables,
        comparisons=args.comparisons,
        unique_routes=len(counts),
        singleton_routes=sum(1 for size in sizes if size == 1),
        nontrivial_routes=sum(1 for size in sizes if size > 1),
        max_fiber_size=max(sizes) if sizes else 0,
        route_entropy_bits=entropy_bits(sizes, z.shape[0]) if sizes else 0.0,
        sampled_fiber_pairs=diffs.shape[0],
        rank_d_pair=rank,
        surviving_bilinear_dim=surviving,
        surviving_fraction=surviving / z.shape[1],
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


def write_csv(path: Path, rows: list[PairCapacityResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].as_dict().keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())


def write_json(path: Path, rows: list[PairCapacityResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([row.as_dict() for row in rows], indent=2) + "\n")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    x = make_support(args)
    left, right = make_pair_indices(args, x.shape[0])
    z = bilinear_features(x, left, right, args.max_bilinear_dim)
    policies = [item.strip() for item in args.anchor_policy.split(",") if item.strip()]
    results = [estimate_capacity(args, x, left, right, z, policy) for policy in policies]
    for result in results:
        print(json.dumps(result.as_dict(), sort_keys=True))
    if args.out_csv is not None:
        write_csv(args.out_csv, results)
    if args.out_json is not None:
        write_json(args.out_json, results)


if __name__ == "__main__":
    main()
