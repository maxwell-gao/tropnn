from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tropnn.layers.pairwise import PairwiseLUT
from tropnn.tools.coxeter_relation_probe import LocalS4Router

READOUTS = ("linear", "pclut_cleanup")
CHAMBERS = 24


@dataclass(frozen=True)
class OrdinalData:
    train_routes: Tensor
    held_routes: Tensor
    feature_counts: Tensor
    route_fingerprint: str


def parse_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not result or any(item <= 0 for item in result):
        raise ValueError("expected a non-empty comma-separated list of positive integers")
    return result


def parse_floats(value: str) -> tuple[float, ...]:
    result = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not result:
        raise ValueError("expected a non-empty comma-separated list of floats")
    return result


def make_ordinal_data(
    *,
    input_dim: int,
    tables: int,
    train_samples: int,
    held_samples: int,
    seed: int,
    device: torch.device,
) -> tuple[LocalS4Router, OrdinalData]:
    """Create one frozen, continuous object set and cache its legal S4 routes."""

    generator = torch.Generator(device="cpu").manual_seed(seed + 101)
    values = torch.randn(train_samples + held_samples, input_dim, generator=generator)
    router = LocalS4Router(input_dim=input_dim, tables=tables, seed=seed).to(device)
    with torch.no_grad():
        routes = router.route(values.to(device))
    train_routes = routes[:train_samples].contiguous()
    held_routes = routes[train_samples:].contiguous()
    offsets = CHAMBERS * torch.arange(tables, device=device).view(1, -1)
    feature_indices = train_routes + offsets
    feature_counts = torch.bincount(feature_indices.flatten(), minlength=tables * CHAMBERS)
    digest = hashlib.sha256(routes.detach().cpu().numpy().tobytes()).hexdigest()
    return router, OrdinalData(train_routes, held_routes, feature_counts, digest)


def routes_to_target(routes: Tensor, feature_count: int) -> Tensor:
    tables = routes.shape[1]
    offsets = CHAMBERS * torch.arange(tables, device=routes.device).view(1, -1)
    target = torch.zeros(routes.shape[0], feature_count, device=routes.device)
    return target.scatter_(1, routes + offsets, 1.0)


class ChamberSuperpositionModel(nn.Module):
    """Tied chamber autoencoder with an optional nonlinear PC-LUT correction."""

    def __init__(
        self,
        *,
        tables: int,
        message_width: int,
        readout: str,
        cleanup_tables: int,
        cleanup_comparisons: int,
        seed: int,
    ) -> None:
        super().__init__()
        if readout not in READOUTS:
            raise ValueError(f"unknown readout {readout!r}")
        self.tables = int(tables)
        self.feature_count = self.tables * CHAMBERS
        self.message_width = int(message_width)
        self.readout = readout
        generator = torch.Generator(device="cpu").manual_seed(seed + 307)
        self.payload = nn.Parameter(
            torch.randn(self.feature_count, self.message_width, generator=generator)
            / math.sqrt(self.message_width)
        )
        self.bias = nn.Parameter(torch.zeros(self.feature_count))
        self.cleanup: PairwiseLUT | None = None
        if readout == "pclut_cleanup":
            self.cleanup = PairwiseLUT(
                input_dim=self.message_width,
                output_dim=self.feature_count,
                tables=cleanup_tables,
                comparisons=cleanup_comparisons,
                backend="torch",
                seed=seed + 401,
                anchor_seed=seed + 409,
                anchor_policy="random_no_replace",
                lut_init_std=0.0,
                lut_dtype="fp32",
                fixed_zero_threshold=True,
                use_output_scaling=True,
                use_min_margin_ste=True,
            )

    def message(self, routes: Tensor) -> Tensor:
        offsets = CHAMBERS * torch.arange(self.tables, device=routes.device).view(1, -1)
        # No 1/sqrt(T) factor: as in the tied superposition autoencoder, a
        # unit-norm active row contributes unit self-gain at decode time.
        return self.payload[routes + offsets].sum(dim=1)

    def forward(self, routes: Tensor) -> Tensor:
        message = self.message(routes)
        prediction = message @ self.payload.T + self.bias
        if self.cleanup is not None:
            prediction = prediction + self.cleanup(message).squeeze(1)
        return prediction


def cosine_lr(step: int, base_lr: float, steps: int, warmup_steps: int) -> float:
    warmup_steps = min(warmup_steps, max(1, steps // 2))
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, steps - warmup_steps)
    return base_lr * (0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress)))


@torch.no_grad()
def apply_superposition_regularizer(payload: Tensor, learning_rate: float, coefficient: float) -> None:
    """Apply the row-wise decay/growth rule from SuperpositionScaling Eq. 3."""

    if coefficient >= 0.0:
        payload.mul_(1.0 - learning_rate * coefficient)
        return
    row_norm = payload.norm(dim=1, keepdim=True).clamp_min(1e-8)
    payload.add_(coefficient * payload * (1.0 - row_norm.reciprocal()), alpha=learning_rate)


def normalized_squared_overlap(vectors: Tensor, mask: Tensor | None = None) -> float:
    if mask is not None:
        vectors = vectors[mask]
    if vectors.shape[0] < 2:
        return float("nan")
    unit = F.normalize(vectors.float(), dim=1)
    gram2 = (unit @ unit.T).square()
    count = vectors.shape[0]
    return float(((gram2.sum() - count) / (count * (count - 1))).item())


@torch.no_grad()
def payload_metrics(payload: Tensor, tables: int) -> dict[str, float | int]:
    rows = payload.detach().float()
    norms = rows.norm(dim=1)
    nonzero = norms > 1e-6
    represented = norms > 0.5
    strong = norms > 1.0
    table_ids = torch.arange(tables, device=rows.device).repeat_interleave(CHAMBERS)
    unit = F.normalize(rows, dim=1)
    gram2 = (unit @ unit.T).square()
    off_diagonal = ~torch.eye(rows.shape[0], device=rows.device, dtype=torch.bool)
    within = (table_ids[:, None] == table_ids[None, :]) & off_diagonal
    cross = table_ids[:, None] != table_ids[None, :]
    quantiles = torch.quantile(norms, torch.tensor([0.1, 0.5, 0.9], device=rows.device))
    return {
        "nonzero_rows": int(nonzero.sum().item()),
        "nonzero_fraction": float(nonzero.float().mean().item()),
        "represented_rows": int(represented.sum().item()),
        "represented_fraction": float(represented.float().mean().item()),
        "strong_rows": int(strong.sum().item()),
        "strong_fraction": float(strong.float().mean().item()),
        "row_norm_mean": float(norms.mean().item()),
        "row_norm_std": float(norms.std(unbiased=False).item()),
        "row_norm_q10": float(quantiles[0].item()),
        "row_norm_median": float(quantiles[1].item()),
        "row_norm_q90": float(quantiles[2].item()),
        "mean_squared_overlap": normalized_squared_overlap(rows, nonzero),
        "represented_mean_squared_overlap": normalized_squared_overlap(rows, represented),
        "strong_mean_squared_overlap": normalized_squared_overlap(rows, strong),
        "within_table_mean_squared_overlap": float(gram2[within].mean().item()),
        "cross_table_mean_squared_overlap": float(gram2[cross].mean().item()),
    }


@torch.no_grad()
def evaluate(
    model: ChamberSuperpositionModel,
    routes: Tensor,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    squared_error = 0.0
    active_squared_error = 0.0
    inactive_squared_error = 0.0
    correct = 0
    total = 0
    tables = routes.shape[1]
    feature_count = tables * CHAMBERS
    for start in range(0, routes.shape[0], batch_size):
        route = routes[start : start + batch_size]
        prediction = model(route)
        target = routes_to_target(route, feature_count)
        error2 = (prediction - target).square()
        offsets = CHAMBERS * torch.arange(tables, device=route.device).view(1, -1)
        active_index = route + offsets
        active_error = error2.gather(1, active_index)
        squared_error += float(error2.sum().item())
        active_squared_error += float(active_error.sum().item())
        inactive_squared_error += float((error2.sum() - active_error.sum()).item())
        predicted_route = prediction.view(route.shape[0], tables, CHAMBERS).argmax(dim=-1)
        correct += int((predicted_route == route).sum().item())
        total += route.shape[0]
    active_total = total * tables
    inactive_total = total * (feature_count - tables)
    return {
        # A zero predictor has held_loss=1. This is ordinary feature MSE
        # multiplied by N/T, a constant common to every width and readout.
        "held_loss": squared_error / active_total,
        "held_feature_mse": squared_error / (total * feature_count),
        "held_active_mse": active_squared_error / active_total,
        "held_inactive_mse": inactive_squared_error / inactive_total,
        "held_chamber_accuracy": correct / active_total,
    }


def run_width(
    *,
    args: argparse.Namespace,
    data: OrdinalData,
    message_width: int,
    device: torch.device,
) -> dict[str, object]:
    torch.manual_seed(args.seed * 10_000 + message_width)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed * 10_000 + message_width)
    model = ChamberSuperpositionModel(
        tables=args.tables,
        message_width=message_width,
        readout=args.readout,
        cleanup_tables=args.cleanup_tables,
        cleanup_comparisons=args.cleanup_comparisons,
        seed=args.seed * 10_000 + message_width,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    batch_generator = torch.Generator(device=device).manual_seed(args.seed + 503)
    started = time.perf_counter()
    final_train_loss = float("nan")
    model.train()
    for step in range(args.steps):
        learning_rate = cosine_lr(step, args.lr, args.steps, args.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        indices = torch.randint(
            0,
            data.train_routes.shape[0],
            (args.batch_size,),
            generator=batch_generator,
            device=device,
        )
        routes = data.train_routes[indices]
        target = routes_to_target(routes, model.feature_count)
        prediction = model(routes)
        loss = (prediction - target).square().sum(dim=-1).mean() / args.tables
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        apply_superposition_regularizer(model.payload, learning_rate, args.weight_decay)
        optimizer.step()
        final_train_loss = float(loss.detach().item())
        if args.log_every and (step + 1) % args.log_every == 0:
            print(
                f"readout={args.readout} wd={args.weight_decay:g} seed={args.seed} "
                f"m={message_width} step={step + 1}/{args.steps} loss={final_train_loss:.7g}",
                flush=True,
            )

    metrics: dict[str, object] = {
        "message_width": message_width,
        "feature_count": model.feature_count,
        "compression_ratio": model.feature_count / message_width,
        "weak_reference_fraction": message_width / model.feature_count,
        "learned_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "payload_parameters": model.payload.numel(),
        "cleanup_parameters": 0 if model.cleanup is None else sum(parameter.numel() for parameter in model.cleanup.parameters()),
        "final_train_loss": final_train_loss,
        "elapsed_seconds": time.perf_counter() - started,
    }
    metrics.update(evaluate(model, data.held_routes, args.eval_batch_size))
    metrics.update(payload_metrics(model.payload, args.tables))
    metrics["overlap_times_width"] = float(metrics["mean_squared_overlap"]) * message_width
    metrics["represented_overlap_times_width"] = float(metrics["represented_mean_squared_overlap"]) * message_width
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ordinal S4 chamber superposition scaling probe.")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run")
    run.add_argument("--readout", choices=READOUTS, required=True)
    run.add_argument("--weight-decay", type=float, required=True)
    run.add_argument("--message-widths", default="4,8,16,32,64,128")
    run.add_argument("--input-dim", type=int, default=32)
    run.add_argument("--tables", type=int, default=16)
    run.add_argument("--train-samples", type=int, default=8192)
    run.add_argument("--held-samples", type=int, default=2048)
    run.add_argument("--steps", type=int, default=10000)
    run.add_argument("--batch-size", type=int, default=512)
    run.add_argument("--eval-batch-size", type=int, default=1024)
    run.add_argument("--lr", type=float, default=0.01)
    run.add_argument("--warmup-steps", type=int, default=1000)
    run.add_argument("--cleanup-tables", type=int, default=4)
    run.add_argument("--cleanup-comparisons", type=int, default=4)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--device", default="cuda")
    run.add_argument("--log-every", type=int, default=1000)
    run.add_argument("--out-dir", type=Path, required=True)

    summarize = commands.add_parser("summarize")
    summarize.add_argument("--result-dir", type=Path, required=True)
    summarize.add_argument("--out-report", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    widths = parse_ints(args.message_widths)
    router, data = make_ordinal_data(
        input_dim=args.input_dim,
        tables=args.tables,
        train_samples=args.train_samples,
        held_samples=args.held_samples,
        seed=args.seed,
        device=device,
    )
    rows = [run_width(args=args, data=data, message_width=width, device=device) for width in widths]
    counts = data.feature_counts.detach().cpu()
    result = {
        "readout": args.readout,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "input_dim": args.input_dim,
        "tables": args.tables,
        "chambers_per_table": CHAMBERS,
        "feature_count": args.tables * CHAMBERS,
        "train_samples": args.train_samples,
        "held_samples": args.held_samples,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "warmup_steps": args.warmup_steps,
        "cleanup_tables": args.cleanup_tables,
        "cleanup_comparisons": args.cleanup_comparisons,
        "message_widths": widths,
        "route_anchor_groups": router.anchors.detach().cpu().tolist(),
        "route_fingerprint": data.route_fingerprint,
        "train_occupied_features": int((counts > 0).sum().item()),
        "train_feature_count_min": int(counts.min().item()),
        "train_feature_count_max": int(counts.max().item()),
        "semantics": {
            "input": "one legal S4 chamber per fixed local table; no raw-coordinate or continuous teacher target",
            "message": "sum of the T active chamber payload rows in R^m",
            "linear": "tied linear decode h W^T + b",
            "pclut_cleanup": "tied linear decode plus a fixed-zero-threshold PairwiseLUT residual G(h)",
            "loss": "one-hot chamber reconstruction squared error divided by T; zero prediction has loss one",
            "regularizer": "SuperpositionScaling row rule on W only: positive decay toward zero; negative coefficient toward unit row norm",
            "represented": "payload row norm greater than 0.5",
            "nonzero": "payload row norm greater than 1e-6",
        },
        "width_results": rows,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"{args.readout}-wd{args.weight_decay:g}-seed{args.seed}.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)


def mean_sem(values: list[float]) -> tuple[float, float]:
    mean = statistics.mean(values)
    sem = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return mean, sem


def fit_power(rows: list[dict[str, object]], metric: str) -> dict[str, float]:
    x = torch.tensor([float(row["message_width"]) for row in rows], dtype=torch.float64)
    y = torch.tensor([float(row[metric]) for row in rows], dtype=torch.float64)
    mask = torch.isfinite(y) & (y > 0)
    x = x[mask]
    y = y[mask]
    if x.numel() < 3:
        return {"coefficient": float("nan"), "exponent": float("nan"), "r2": float("nan")}
    log_x = x.log()
    log_y = y.log()
    centered_x = log_x - log_x.mean()
    slope = ((log_y - log_y.mean()) * centered_x).sum() / centered_x.square().sum()
    intercept = log_y.mean() - slope * log_x.mean()
    prediction = (intercept + slope * log_x).exp()
    ss_res = (y - prediction).square().sum()
    ss_tot = (y - y.mean()).square().sum()
    return {
        "coefficient": float(intercept.exp().item()),
        "exponent": float((-slope).item()),
        "r2": float((1.0 - ss_res / ss_tot).item()) if ss_tot > 0 else float("nan"),
    }


def fit_loss_floor(rows: list[dict[str, object]]) -> dict[str, float]:
    x = torch.tensor([float(row["message_width"]) for row in rows], dtype=torch.float64)
    y = torch.tensor([float(row["held_loss"]) for row in rows], dtype=torch.float64)
    order = torch.argsort(x)
    x = x[order]
    y = y[order]
    minimum = float(y.min().item())
    candidates = torch.linspace(0.0, max(0.0, minimum * 0.999), 4096, dtype=torch.float64)
    best: tuple[float, float, float, float] | None = None
    log_x = x.log()
    centered_x = log_x - log_x.mean()
    denominator = centered_x.square().sum()
    for floor in candidates:
        residual = (y - floor).clamp_min(1e-12)
        log_y = residual.log()
        slope = ((log_y - log_y.mean()) * centered_x).sum() / denominator
        if slope >= 0:
            continue
        intercept = log_y.mean() - slope * log_x.mean()
        prediction = floor + (intercept + slope * log_x).exp()
        sse = float((y - prediction).square().sum().item())
        candidate = (sse, float(floor.item()), float(intercept.exp().item()), float((-slope).item()))
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None:
        return {"loss_floor": float("nan"), "coefficient": float("nan"), "alpha": float("nan"), "r2": float("nan")}
    sse, floor, coefficient, alpha = best
    ss_tot = float((y - y.mean()).square().sum().item())
    return {
        "loss_floor": floor,
        "coefficient": coefficient,
        "alpha": alpha,
        "r2": 1.0 - sse / ss_tot if ss_tot > 0 else float("nan"),
    }


def fit_affine_width(rows: list[dict[str, object]]) -> dict[str, float]:
    feature_count = float(rows[0]["feature_count"])
    x = torch.tensor([float(row["message_width"]) / feature_count for row in rows], dtype=torch.float64)
    y = torch.tensor([float(row["held_loss"]) for row in rows], dtype=torch.float64)
    design = torch.stack((torch.ones_like(x), x), dim=1)
    coefficient = torch.linalg.lstsq(design, y).solution
    prediction = design @ coefficient
    ss_res = (y - prediction).square().sum()
    ss_tot = (y - y.mean()).square().sum()
    return {
        "intercept": float(coefficient[0].item()),
        "slope": float(coefficient[1].item()),
        "r2": float((1.0 - ss_res / ss_tot).item()) if ss_tot > 0 else float("nan"),
    }


def summarize(args: argparse.Namespace) -> None:
    runs = [json.loads(path.read_text()) for path in sorted(args.result_dir.glob("**/*-seed*.json"))]
    if not runs:
        raise RuntimeError(f"no run JSON files found under {args.result_dir}")
    invariant_fields = (
        "input_dim",
        "tables",
        "feature_count",
        "train_samples",
        "held_samples",
        "steps",
        "batch_size",
        "lr",
        "warmup_steps",
        "cleanup_tables",
        "cleanup_comparisons",
    )
    for field in invariant_fields:
        if len({json.dumps(run[field], sort_keys=True) for run in runs}) != 1:
            raise RuntimeError(f"runs do not share fixed {field}")
    for seed in {int(run["seed"]) for run in runs}:
        seed_runs = [run for run in runs if int(run["seed"]) == seed]
        if len({run["route_fingerprint"] for run in seed_runs}) != 1:
            raise RuntimeError(f"seed {seed} does not use fixed source routes")

    flat: list[dict[str, object]] = []
    for run in runs:
        for width_row in run["width_results"]:
            flat.append(
                {
                    "readout": run["readout"],
                    "weight_decay": run["weight_decay"],
                    "seed": run["seed"],
                    "route_fingerprint": run["route_fingerprint"],
                    **width_row,
                }
            )
    fields = sorted({field for row in flat for field in row})
    with (args.result_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)

    metric_names = (
        "held_loss",
        "held_feature_mse",
        "held_active_mse",
        "held_inactive_mse",
        "held_chamber_accuracy",
        "nonzero_fraction",
        "represented_fraction",
        "strong_fraction",
        "row_norm_mean",
        "row_norm_std",
        "mean_squared_overlap",
        "represented_mean_squared_overlap",
        "within_table_mean_squared_overlap",
        "cross_table_mean_squared_overlap",
        "overlap_times_width",
    )
    aggregate: list[dict[str, object]] = []
    groups = sorted({(str(row["readout"]), float(row["weight_decay"]), int(row["message_width"])) for row in flat})
    for readout, weight_decay, width in groups:
        members = [
            row
            for row in flat
            if row["readout"] == readout
            and float(row["weight_decay"]) == weight_decay
            and int(row["message_width"]) == width
        ]
        aggregate_row: dict[str, object] = {
            "readout": readout,
            "weight_decay": weight_decay,
            "message_width": width,
            "seeds": len(members),
            "feature_count": members[0]["feature_count"],
            "weak_reference_fraction": members[0]["weak_reference_fraction"],
            "learned_parameters": members[0]["learned_parameters"],
        }
        for metric in metric_names:
            values = [float(member[metric]) for member in members if math.isfinite(float(member[metric]))]
            aggregate_row[metric], aggregate_row[f"{metric}_sem"] = mean_sem(values) if values else (float("nan"), float("nan"))
        aggregate.append(aggregate_row)

    fits: list[dict[str, object]] = []
    conditions = sorted({(str(row["readout"]), float(row["weight_decay"])) for row in aggregate})
    for readout, weight_decay in conditions:
        members = sorted(
            [row for row in aggregate if row["readout"] == readout and float(row["weight_decay"]) == weight_decay],
            key=lambda row: int(row["message_width"]),
        )
        fits.append(
            {
                "readout": readout,
                "weight_decay": weight_decay,
                "loss_floor_fit": fit_loss_floor(members),
                "affine_width_fit": fit_affine_width(members),
                "overlap_power_fit": fit_power(members, "mean_squared_overlap"),
                "represented_overlap_power_fit": fit_power(members, "represented_mean_squared_overlap"),
                "within_table_overlap_power_fit": fit_power(members, "within_table_mean_squared_overlap"),
                "cross_table_overlap_power_fit": fit_power(members, "cross_table_mean_squared_overlap"),
                "overlap_times_width_mean": statistics.mean(float(row["overlap_times_width"]) for row in members),
                "cross_overlap_times_width_mean": statistics.mean(
                    float(row["cross_table_mean_squared_overlap"]) * int(row["message_width"]) for row in members
                ),
                "represented_fraction_mean": statistics.mean(float(row["represented_fraction"]) for row in members),
            }
        )
    (args.result_dir / "aggregate.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    (args.result_dir / "fits.json").write_text(json.dumps(fits, indent=2, sort_keys=True) + "\n")

    widths = sorted({int(row["message_width"]) for row in aggregate})
    weight_decays = sorted({float(row["weight_decay"]) for row in aggregate})
    seeds = sorted({int(row["seed"]) for row in flat})
    indexed = {(str(row["readout"]), float(row["weight_decay"]), int(row["message_width"])): row for row in aggregate}
    fit_indexed = {(str(row["readout"]), float(row["weight_decay"])): row for row in fits}
    cleanup_gains = []
    for weight_decay in weight_decays:
        for width in widths:
            linear = indexed[("linear", weight_decay, width)]
            cleanup = indexed[("pclut_cleanup", weight_decay, width)]
            cleanup_gains.append(
                (float(linear["held_loss"]) - float(cleanup["held_loss"])) / float(linear["held_loss"])
            )
    loss_power_r2 = [float(row["loss_floor_fit"]["r2"]) for row in fits]
    affine_r2 = [float(row["affine_width_fit"]["r2"]) for row in fits]
    neutral_linear = fit_indexed[("linear", 0.0)]
    neutral_cleanup = fit_indexed[("pclut_cleanup", 0.0)]
    strong_linear = fit_indexed[("linear", -16.0)]
    strong_cleanup = fit_indexed[("pclut_cleanup", -16.0)]
    lines = [
        "# S4 chamber superposition scaling",
        "",
        "This probe asks whether a fixed ordinal feature system exhibits width scaling in the sense of "
        "`ref/SuperpositionScaling`. There is no continuous or bilinear teacher: every example activates "
        "exactly one of 24 legal S4 chambers in each of 16 frozen tables.",
        "",
        "## Controlled protocol",
        "",
        f"Across all conditions, T={runs[0]['tables']}, N={runs[0]['feature_count']}, train/held data="
        f"{runs[0]['train_samples']}/{runs[0]['held_samples']}, and the per-seed S4 anchors and cached routes are fixed. "
        f"Only message width changes over {widths}. Results average {len(seeds)} seeds. The decoder is either tied "
        "linear or tied linear plus a nonlinear PC-LUT residual. The regularizer is applied only to the chamber "
        "payload matrix using the paper's row-wise decay/growth rule.",
        "",
        "A row is represented when its norm exceeds 0.5. A zero reconstruction has normalized held loss 1. "
        "The weak-superposition reference fraction is m/N; strong superposition instead keeps nearly all rows nonzero. "
        "Normalized mean squared overlap is computed off-diagonal after normalizing every nonzero payload row.",
        "",
        "## Main result",
        "",
        "There is a real one-over-width overlap regime, but it is not the paper's complete superposition scaling law. "
        f"At gamma=0, global/cross-table overlap exponents are {neutral_linear['overlap_power_fit']['exponent']:.3f}/"
        f"{neutral_linear['cross_table_overlap_power_fit']['exponent']:.3f} for the linear readout and "
        f"{neutral_cleanup['overlap_power_fit']['exponent']:.3f}/"
        f"{neutral_cleanup['cross_table_overlap_power_fit']['exponent']:.3f} with cleanup. Thus ordinary learned "
        "chamber rows have the expected random-geometry scale. Under activation-count-matched growth gamma=-16, "
        f"all rows have norm about one, but global/cross exponents fall to "
        f"{strong_linear['overlap_power_fit']['exponent']:.3f}/"
        f"{strong_linear['cross_table_overlap_power_fit']['exponent']:.3f} (linear) and "
        f"{strong_cleanup['overlap_power_fit']['exponent']:.3f}/"
        f"{strong_cleanup['cross_table_overlap_power_fit']['exponent']:.3f} (cleanup). Strong row representation "
        "therefore does not imply isotropic 1/m overlap in this structured ordinal ensemble.",
        "",
        f"The held losses do not support L_inf+C/m^alpha on this range: every fitted floor hits zero, alpha lies "
        f"between {min(float(row['loss_floor_fit']['alpha']) for row in fits):.3f} and "
        f"{max(float(row['loss_floor_fit']['alpha']) for row in fits):.3f}, and R2 is only "
        f"{min(loss_power_r2):.3f}-{max(loss_power_r2):.3f}. A simple affine rank law in m/N fits much better "
        f"(R2 {min(affine_r2):.3f}-{max(affine_r2):.3f}). Width is helping mainly by resolving more of the "
        "fixed 384-state categorical system, not by removing isotropic interference with a 1/m loss tail.",
        "",
        "Every trained chamber row remains nonzero. Growth gamma=-16 forces phi_1/2=1 at every width, while "
        "gamma=0, 0.1, and 1 are nearly indistinguishable and do not approach the canonical weak reference phi=m/N. "
        "They form a diffuse, width-dependent norm solution instead. Hence regularization produces a norm-regime "
        "change, but not the clean weak/strong phase transition of the exchangeable Bernoulli toy model.",
        "",
        f"The PC-LUT residual changes held loss by {statistics.mean(cleanup_gains):.1%} on average "
        f"({min(cleanup_gains):.1%} to {max(cleanup_gains):.1%}) and does not change the loss law. It adds a fixed "
        "24,576 parameters: 12.8x the m=4 linear model's parameter count, falling to 0.50x extra at m=128. "
        "This is a small nonlinear cleanup effect, not a capacity-efficient escape from the ordinal bottleneck.",
        "",
        "## Scaling fits",
        "",
        "| Readout | decay/growth gamma | represented fraction | global overlap beta | cross-table beta | "
        "mean m*cross-overlap | loss floor | loss alpha | loss-fit R2 | affine m/N R2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for fit in fits:
        overlap_fit = fit["overlap_power_fit"]
        cross_overlap_fit = fit["cross_table_overlap_power_fit"]
        loss_fit = fit["loss_floor_fit"]
        affine_fit = fit["affine_width_fit"]
        lines.append(
            f"| {fit['readout']} | {float(fit['weight_decay']):g} | {float(fit['represented_fraction_mean']):.3f} | "
            f"{float(overlap_fit['exponent']):.3f} | {float(cross_overlap_fit['exponent']):.3f} | "
            f"{float(fit['cross_overlap_times_width_mean']):.3f} | "
            f"{float(loss_fit['loss_floor']):.5f} | {float(loss_fit['alpha']):.3f} | {float(loss_fit['r2']):.3f} | "
            f"{float(affine_fit['r2']):.3f} |"
        )

    for weight_decay in weight_decays:
        lines.extend(
            [
                "",
                f"## gamma = {weight_decay:g}",
                "",
                "| m | linear loss | cleanup loss | cleanup gain | linear represented | cleanup represented | "
                "linear row norm | cleanup row norm | linear m*overlap | cleanup m*overlap | linear accuracy | cleanup accuracy |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for width in widths:
            linear = indexed.get(("linear", weight_decay, width))
            cleanup = indexed.get(("pclut_cleanup", weight_decay, width))
            if linear is None or cleanup is None:
                continue
            gain = (float(linear["held_loss"]) - float(cleanup["held_loss"])) / float(linear["held_loss"])
            lines.append(
                f"| {width} | {float(linear['held_loss']):.5f} | {float(cleanup['held_loss']):.5f} | "
                f"{gain:.1%} | {float(linear['represented_fraction']):.3f} | "
                f"{float(cleanup['represented_fraction']):.3f} | {float(linear['row_norm_mean']):.3f} | "
                f"{float(cleanup['row_norm_mean']):.3f} | {float(linear['overlap_times_width']):.3f} | "
                f"{float(cleanup['overlap_times_width']):.3f} | {float(linear['held_chamber_accuracy']):.3f} | "
                f"{float(cleanup['held_chamber_accuracy']):.3f} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This is an ordinal-native representation test, not evidence about continuous inner-product teachers. "
            "The S4 route vocabulary and co-activation structure differ from independent Bernoulli features: chamber "
            "states within one table are mutually exclusive, while one state from every table is always active. "
            "Accordingly, within-table and cross-table overlaps are retained in the CSV rather than assuming exchangeable features.",
            "",
            "PC-LUT routing is scale-invariant when thresholds are fixed at zero. Its payload-row norms must therefore be "
            "read together with held reconstruction and chamber accuracy: a small encoder norm need not mean an ordinal "
            "feature has disappeared if the nonlinear route can still distinguish its direction.",
            "",
            "## Artifacts",
            "",
            f"- Raw and aggregate results: `{args.result_dir}`",
            "- Probe: `python/src/tropnn/tools/s4_superposition_scaling_probe.py`",
            "- Launcher: `scripts/run_tropnn_s4_superposition_scaling_4gpu.sh`",
            "- Tests: `python/src/tropnn/tests/test_s4_superposition_scaling_probe.py`",
            "",
        ]
    )
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text("\n".join(lines))


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run":
        run(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
