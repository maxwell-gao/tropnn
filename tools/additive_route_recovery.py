from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch
from torch import Tensor

from tropnn.layers import PairwiseLUT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit the actual additive PC-LUT feature map "
            "[onehot(c_1), ..., onehot(c_T)] to inputs and BitLinear targets."
        )
    )
    parser.add_argument("--support-size", type=int, default=8192)
    parser.add_argument("--input-dim", type=int, default=128)
    parser.add_argument("--max-value", type=int, default=15)
    parser.add_argument("--budgets", default="T1_C4,T4_C4,T8_C5,T16_C5")
    parser.add_argument("--anchor-policies", default="random,expander")
    parser.add_argument("--bitlinear-output-dim", type=int, default=64)
    parser.add_argument("--train-fraction", type=float, default=0.75)
    parser.add_argument("--rcond", type=float, default=1e-10)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-report", type=Path, default=None)
    return parser.parse_args()


def comma_list(text: str) -> list[str]:
    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("expected a non-empty comma-separated list")
    return values


def parse_budget(text: str) -> tuple[int, int]:
    if not text.startswith("T") or "_C" not in text:
        raise ValueError(f"invalid budget {text!r}; expected T16_C5")
    tables, comparisons = text[1:].split("_C", 1)
    return int(tables), int(comparisons)


def make_support(args: argparse.Namespace) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    return torch.randint(
        0,
        args.max_value + 1,
        (args.support_size, args.input_dim),
        generator=generator,
        dtype=torch.int64,
    ).to(torch.float64)


def make_bitlinear_weight(args: argparse.Namespace) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 101)
    output_dim = min(args.bitlinear_output_dim, args.input_dim)
    signs = torch.randint(0, 2, (output_dim, args.input_dim), generator=generator, dtype=torch.int64)
    return (2.0 * signs.to(torch.float64) - 1.0) / math.sqrt(args.input_dim)


def route_support(
    x: Tensor,
    *,
    tables: int,
    comparisons: int,
    policy: str,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> Tensor:
    layer = PairwiseLUT(
        input_dim=x.shape[1],
        output_dim=1,
        tables=tables,
        comparisons=comparisons,
        backend="torch",
        seed=seed,
        anchor_policy=policy,
        anchor_seed=seed,
        fixed_zero_threshold=True,
    ).to(device)
    layer.eval()
    chunks: list[Tensor] = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            batch = x[start : start + batch_size].to(device=device, dtype=torch.float32)
            chunks.append(layer.route(batch.unsqueeze(1)).indices[:, 0, :].cpu())
    return torch.cat(chunks, dim=0).to(torch.long)


def additive_feature_matrix(codes: Tensor, comparisons: int) -> Tensor:
    table_size = 1 << comparisons
    tables = codes.shape[1]
    offsets = torch.arange(tables, dtype=torch.long).mul(table_size)
    columns = codes + offsets.unsqueeze(0)
    features = torch.zeros(codes.shape[0], tables * table_size, dtype=torch.float64)
    features.scatter_(1, columns, 1.0)
    return features


def minimum_norm_least_squares(
    features: Tensor,
    targets: Tensor,
    *,
    rcond: float,
) -> tuple[Tensor, int, float]:
    gram = features.T @ features
    rhs = features.T @ targets
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    largest = float(eigenvalues[-1].item()) if eigenvalues.numel() else 0.0
    cutoff = rcond * largest
    keep = eigenvalues > cutoff
    if not bool(keep.any()):
        return torch.zeros(features.shape[1], targets.shape[1], dtype=torch.float64), 0, float("inf")
    basis = eigenvectors[:, keep]
    coefficients = basis @ ((basis.T @ rhs) / eigenvalues[keep].unsqueeze(1))
    kept = eigenvalues[keep]
    condition = float((kept[-1] / kept[0]).item())
    return coefficients, int(keep.sum().item()), condition


def metrics(target: Tensor, prediction: Tensor) -> dict[str, float]:
    error = target - prediction
    mse = float(error.square().mean().item())
    centered = target - target.mean(dim=0, keepdim=True)
    variance = float(centered.square().mean().item())
    r2 = 1.0 - mse / variance if variance > 1e-24 else (1.0 if mse < 1e-24 else float("nan"))
    return {
        "mse": mse,
        "r2": float(r2),
        "max_abs": float(error.abs().max().item()),
    }


def joint_route_representatives(x: Tensor, codes: Tensor) -> tuple[Tensor, int, int]:
    _, inverse, counts = torch.unique(codes, dim=0, return_inverse=True, return_counts=True)
    sums = torch.zeros(counts.numel(), x.shape[1], dtype=torch.float64)
    sums.index_add_(0, inverse, x)
    means = sums / counts.to(torch.float64).unsqueeze(1)
    return means[inverse], int(counts.numel()), int(counts.max().item())


def add_metrics(row: dict[str, object], prefix: str, target: Tensor, prediction: Tensor) -> None:
    for name, value in metrics(target, prediction).items():
        row[f"{prefix}_{name}"] = value


def run_case(
    args: argparse.Namespace,
    x: Tensor,
    bit_weight: Tensor,
    budget: str,
    policy: str,
    train_indices: Tensor,
    test_indices: Tensor,
) -> dict[str, object]:
    tables, comparisons = parse_budget(budget)
    codes = route_support(
        x,
        tables=tables,
        comparisons=comparisons,
        policy=policy,
        batch_size=args.batch_size,
        device=torch.device(args.device),
        seed=args.seed,
    )
    features = additive_feature_matrix(codes, comparisons)
    bit_targets = x @ bit_weight.T
    all_targets = torch.cat([x, bit_targets], dim=1)

    full_coefficients, full_rank, full_condition = minimum_norm_least_squares(
        features, all_targets, rcond=args.rcond
    )
    input_coefficients = full_coefficients[:, : x.shape[1]]
    direct_bit_payload = full_coefficients[:, x.shape[1] :]
    composed_bit_payload = input_coefficients @ bit_weight.T

    train_coefficients, train_rank, train_condition = minimum_norm_least_squares(
        features[train_indices], all_targets[train_indices], rcond=args.rcond
    )
    train_input_coefficients = train_coefficients[:, : x.shape[1]]
    train_direct_bit_payload = train_coefficients[:, x.shape[1] :]
    train_composed_bit_payload = train_input_coefficients @ bit_weight.T

    joint_x_hat, unique_routes, max_fiber_size = joint_route_representatives(x, codes)
    row: dict[str, object] = {
        "budget": budget,
        "anchor_policy": policy,
        "support_size": x.shape[0],
        "input_dim": x.shape[1],
        "tables": tables,
        "comparisons": comparisons,
        "table_size": 1 << comparisons,
        "feature_dim": features.shape[1],
        "feature_rank_full": full_rank,
        "feature_rank_train": train_rank,
        "feature_condition_full": full_condition,
        "feature_condition_train": train_condition,
        "unique_joint_routes": unique_routes,
        "max_joint_fiber_size": max_fiber_size,
        "train_size": train_indices.numel(),
        "test_size": test_indices.numel(),
    }
    add_metrics(row, "joint_input_full", x, joint_x_hat)
    add_metrics(row, "joint_bitlinear_full", bit_targets, joint_x_hat @ bit_weight.T)
    add_metrics(row, "add_input_full", x, features @ input_coefficients)
    add_metrics(row, "add_bitlinear_direct_full", bit_targets, features @ direct_bit_payload)
    add_metrics(row, "add_bitlinear_composed_full", bit_targets, features @ composed_bit_payload)
    add_metrics(
        row,
        "add_input_train",
        x[train_indices],
        features[train_indices] @ train_input_coefficients,
    )
    add_metrics(
        row,
        "add_input_test",
        x[test_indices],
        features[test_indices] @ train_input_coefficients,
    )
    add_metrics(
        row,
        "add_bitlinear_direct_train",
        bit_targets[train_indices],
        features[train_indices] @ train_direct_bit_payload,
    )
    add_metrics(
        row,
        "add_bitlinear_direct_test",
        bit_targets[test_indices],
        features[test_indices] @ train_direct_bit_payload,
    )
    add_metrics(
        row,
        "add_bitlinear_composed_test",
        bit_targets[test_indices],
        features[test_indices] @ train_composed_bit_payload,
    )
    row["direct_vs_composed_payload_max_abs"] = float(
        (direct_bit_payload - composed_bit_payload).abs().max().item()
    )
    return row


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2) + "\n")


def write_report(path: Path, args: argparse.Namespace, rows: list[dict[str, object]]) -> None:
    lines = [
        "# Standard PC-LUT Additive Recovery",
        "",
        "This probe fits the actual PC-LUT additive parameterization",
        "",
        "```text",
        "Phi_add(x) = [onehot(c_1(x)), ..., onehot(c_T(x))]",
        "f(x) = Phi_add(x) V = sum_t V_t[c_t(x)]",
        "```",
        "",
        "It contrasts that model with the unrestricted joint-route conditional mean used by the earlier capacity probe.",
        "The route, anchors, and zero thresholds are fixed; only the payload is solved by minimum-norm least squares.",
        "",
        "## Configuration",
        "",
        f"- Support: {args.support_size} random integer vectors in `[0, {args.max_value}]^{args.input_dim}`",
        f"- BitLinear target: `{min(args.bitlinear_output_dim, args.input_dim)} x {args.input_dim}` binary sign matrix",
        f"- Train/test split: {args.train_fraction:.2f}/{1.0 - args.train_fraction:.2f}",
        f"- Least-squares cutoff: `{args.rcond:g}` times the largest Gram eigenvalue",
        "",
        "## Results",
        "",
        "| policy | budget | joint routes | Phi dim | rank | joint input R2 | additive input R2 | BitLinear full R2 | BitLinear test R2 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {anchor_policy} | {budget} | {unique_joint_routes} | {feature_dim} | {feature_rank_full} | "
            "{joint_input_full_r2:.6f} | {add_input_full_r2:.6f} | "
            "{add_bitlinear_direct_full_r2:.6f} | {add_bitlinear_direct_test_r2:.6f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            "`joint input R2` uses one unrestricted representative for the complete `(c_1, ..., c_T)` tuple.",
            "It becomes exactly one when the route tuple is injective on the sampled support.",
            "",
            "`additive input R2` solves `min_R ||Phi_add R - X||_F^2` on the full support.",
            "`BitLinear full R2` independently solves `min_V ||Phi_add V - X W^T||_F^2`.",
            "`BitLinear test R2` fits the additive payload on the training subset and evaluates it on held-out vectors.",
            "",
            "The additive model has at most `1 + T(2^C - 1)` independent columns because every table block sums to the constant feature.",
            "A unique joint route therefore does not imply exact recovery by the standard additive payload.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    if not 0.0 < args.train_fraction < 1.0:
        raise ValueError("--train-fraction must be between zero and one")
    torch.manual_seed(args.seed)
    x = make_support(args)
    bit_weight = make_bitlinear_weight(args)
    split_generator = torch.Generator(device="cpu").manual_seed(args.seed + 211)
    order = torch.randperm(x.shape[0], generator=split_generator)
    train_size = int(round(args.train_fraction * x.shape[0]))
    train_indices = order[:train_size]
    test_indices = order[train_size:]
    rows = [
        run_case(args, x, bit_weight, budget, policy, train_indices, test_indices)
        for policy in comma_list(args.anchor_policies)
        for budget in comma_list(args.budgets)
    ]
    for row in rows:
        print(json.dumps(row, sort_keys=True))
    write_csv(args.out_csv, rows)
    if args.out_json is not None:
        write_json(args.out_json, rows)
    if args.out_report is not None:
        write_report(args.out_report, args, rows)


if __name__ == "__main__":
    main()
