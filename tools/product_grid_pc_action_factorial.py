"""Adaptive D2 tree versus parallel D2 4x4 grid with free vector actions."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from tropnn.layers.hard_lookup import ProductGridLookupRouter, sum_lookup_rows
from tropnn.layers.maddness import CompiledMaddness, FrozenMaddness, LocalCounterfactualMaddness
from tropnn.tools.maddness_end_to_end_ste_factorial import compile_original_maddness
from tropnn.tools.product_atlas_pc_action_factorial import anisotropic_teacher, cached_action_rows, fit_additive_rows

ARMS = ("tree_tied_frozen", "tree_free_task", "grid_tied_frozen", "grid_free_task")


@dataclass(frozen=True)
class ArmResult:
    seed: int
    arm: str
    held_task_mse: float
    held_task_nmse: float
    held_task_r2: float
    held_atlas_nmse: float
    held_atlas_r2: float
    mean_entropy_bits: float
    minimum_entropy_bits: float
    mean_observed_rows: float
    maximum_row_mass: float
    held_code_flip_fraction: float
    threshold_rms_drift: float
    deploy_parameter_scalars: int
    active_comparisons: int
    comparison_rounds: int
    hard_replay_max_error: float
    training_curve: list[dict[str, float]]
    seconds: float


def _nmse(prediction: Tensor, target: Tensor) -> float:
    return float((prediction - target).square().mean() / target.square().mean().clamp_min(1e-30))


def _route_health(codes: Tensor, rows: int) -> tuple[float, float, float, float]:
    entropies: list[float] = []
    observed: list[float] = []
    maximum = 0.0
    for table in range(codes.shape[-1]):
        counts = torch.bincount(codes[:, table].detach().cpu(), minlength=rows).double()
        probability = counts / counts.sum()
        positive = probability > 0
        observed.append(float(positive.sum()))
        entropies.append(float(-(probability[positive] * probability[positive].log2()).sum()))
        maximum = max(maximum, float(probability.max()))
    return sum(entropies) / len(entropies), min(entropies), sum(observed) / len(observed), maximum


def compile_balanced_product_grid(x: Tensor, tables: int, bins: int = 4) -> tuple[Tensor, Tensor, Tensor]:
    """Compile disjoint D2 axes and empirical quartile walls from train only."""

    if x.ndim != 2 or x.shape[1] != 2 * tables or bins != 4:
        raise ValueError("balanced formal grid requires [N,2*tables] and bins=4")
    supports = torch.arange(x.shape[1], dtype=torch.int64).view(tables, 2)
    thresholds = torch.empty(tables, 2, bins - 1, dtype=x.dtype)
    for table in range(tables):
        for axis in range(2):
            values = x[:, supports[table, axis]].sort(stable=True).values
            for wall in range(1, bins):
                rank = wall * values.numel() // bins
                if rank < 1 or rank >= values.numel():
                    raise ValueError("not enough compiler samples for quartile walls")
                thresholds[table, axis, wall - 1] = (values[rank - 1] + values[rank]) * 0.5
    zero_rows = torch.zeros(tables, bins**2, x.shape[1], dtype=x.dtype)
    route = ProductGridLookupRouter(
        x.shape[1],
        x.shape[1],
        supports=supports,
        thresholds=thresholds,
        rows=zero_rows,
        bins=bins,
        surrogate="none",
        trainable_thresholds=False,
        trainable_rows=False,
    )
    return supports, thresholds, route.hard_codes(x)


def _tree_with_rows(compiled: CompiledMaddness, rows: Tensor) -> CompiledMaddness:
    return CompiledMaddness(
        split_indices=compiled.split_indices.detach().clone(),
        thresholds=compiled.thresholds.detach().clone(),
        encoder_centroids=compiled.encoder_centroids.detach().clone(),
        prototypes=rows.detach().clone(),
    )


def _make_grid(
    dim: int,
    supports: Tensor,
    thresholds: Tensor,
    rows: Tensor,
    *,
    tau: float,
    trainable: bool,
) -> ProductGridLookupRouter:
    return ProductGridLookupRouter(
        dim,
        dim,
        supports=supports,
        thresholds=thresholds,
        rows=rows,
        bins=4,
        surrogate="local_counterfactual" if trainable else "none",
        tau=tau,
        trainable_thresholds=trainable,
        trainable_rows=trainable,
    )


def _evaluate(
    *,
    seed: int,
    arm: str,
    prediction: Tensor,
    target: Tensor,
    atlas: Tensor,
    x: Tensor,
    codes: Tensor,
    initial_codes: Tensor,
    thresholds: Tensor,
    initial_thresholds: Tensor,
    parameter_scalars: int,
    active_comparisons: int,
    rounds: int,
    hard_replay: Tensor,
    curve: list[dict[str, float]],
    seconds: float,
) -> ArmResult:
    task_nmse = _nmse(prediction, target)
    atlas_nmse = _nmse(atlas, x)
    entropy, minimum, observed, maximum = _route_health(codes, 16)
    return ArmResult(
        seed=seed,
        arm=arm,
        held_task_mse=float((prediction - target).square().mean()),
        held_task_nmse=task_nmse,
        held_task_r2=1 - task_nmse,
        held_atlas_nmse=atlas_nmse,
        held_atlas_r2=1 - atlas_nmse,
        mean_entropy_bits=entropy,
        minimum_entropy_bits=minimum,
        mean_observed_rows=observed,
        maximum_row_mass=maximum,
        held_code_flip_fraction=float((codes != initial_codes).float().mean()),
        threshold_rms_drift=float((thresholds - initial_thresholds).square().mean().sqrt()),
        deploy_parameter_scalars=parameter_scalars,
        active_comparisons=active_comparisons,
        comparison_rounds=rounds,
        hard_replay_max_error=float((prediction - hard_replay).abs().max()),
        training_curve=curve,
        seconds=seconds,
    )


def fit_seed(seed: int, args: argparse.Namespace) -> tuple[list[ArmResult], dict[str, object], dict[str, Tensor]]:
    device = torch.device(args.device)
    teacher = anisotropic_teacher(args.dim, seed, device)
    compiler_x = torch.randn(args.compiler_samples, args.dim, generator=torch.Generator().manual_seed(40_000 + seed))
    compiler_target = compiler_x @ teacher.detach().cpu().T
    started = time.perf_counter()

    tree_compiled = compile_original_maddness(compiler_x, args.tables, args.ridge)
    tree_reconstruction_rows = tree_compiled.prototypes
    tree_action_rows = cached_action_rows(tree_compiled, teacher)
    tree_codes = FrozenMaddness(tree_compiled).hard_codes(compiler_x)
    tree_direct_rows = fit_additive_rows(tree_codes, compiler_target, 16, args.ridge)
    tree_tied = FrozenMaddness(_tree_with_rows(tree_compiled, tree_action_rows)).to(device)
    tree_free = LocalCounterfactualMaddness(_tree_with_rows(tree_compiled, tree_action_rows), args.tau).to(device)

    grid_supports, grid_thresholds, grid_codes = compile_balanced_product_grid(compiler_x, args.tables)
    grid_reconstruction_rows = fit_additive_rows(grid_codes, compiler_x, 16, args.ridge)
    grid_action_rows = torch.einsum("tkd,od->tko", grid_reconstruction_rows, teacher.detach().cpu())
    grid_direct_rows = fit_additive_rows(grid_codes, compiler_target, 16, args.ridge)
    grid_tied = _make_grid(
        args.dim,
        grid_supports,
        grid_thresholds,
        grid_action_rows,
        tau=args.tau,
        trainable=False,
    ).to(device)
    grid_free = _make_grid(
        args.dim,
        grid_supports,
        grid_thresholds,
        grid_action_rows,
        tau=args.tau,
        trainable=True,
    ).to(device)
    compiler_seconds = time.perf_counter() - started

    optimizer = torch.optim.AdamW([*tree_free.parameters(), *grid_free.parameters()], lr=args.lr, weight_decay=0)
    curves: dict[str, list[dict[str, float]]] = {"tree_free_task": [], "grid_free_task": []}
    train_generator = torch.Generator(device=device).manual_seed(50_000 + seed)
    train_started = time.perf_counter()
    for step in range(1, args.steps + 1):
        x = torch.randn(args.batch_size, args.dim, generator=train_generator, device=device)
        target = x @ teacher.T
        optimizer.zero_grad(set_to_none=True)
        tree_loss = F.mse_loss(tree_free(x), target)
        grid_loss = F.mse_loss(grid_free(x), target)
        (tree_loss + grid_loss).backward()
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            curves["tree_free_task"].append({"step": float(step), "task_mse": float(tree_loss.detach())})
            curves["grid_free_task"].append({"step": float(step), "task_mse": float(grid_loss.detach())})
    train_seconds = time.perf_counter() - train_started

    held_x = torch.randn(args.held_samples, args.dim, generator=torch.Generator(device=device).manual_seed(70_000 + seed), device=device)
    target = held_x @ teacher.T
    tree_initial_codes = tree_tied.hard_codes(held_x)
    grid_initial_codes = grid_tied.hard_codes(held_x)
    rows: list[ArmResult] = []
    with torch.no_grad():
        tree_tied_prediction = tree_tied(held_x)
        tree_tied_atlas = sum_lookup_rows(tree_reconstruction_rows.to(device), tree_initial_codes)
        rows.append(
            _evaluate(
                seed=seed,
                arm="tree_tied_frozen",
                prediction=tree_tied_prediction,
                target=target,
                atlas=tree_tied_atlas,
                x=held_x,
                codes=tree_initial_codes,
                initial_codes=tree_initial_codes,
                thresholds=tree_tied.thresholds,
                initial_thresholds=tree_compiled.thresholds.to(device),
                parameter_scalars=tree_action_rows.numel() + tree_compiled.thresholds.numel(),
                active_comparisons=args.tables * args.depth,
                rounds=args.depth,
                hard_replay=tree_tied.hard_output(held_x)[0],
                curve=[],
                seconds=compiler_seconds,
            )
        )

        tree_prediction = tree_free(held_x)
        tree_current_codes = tree_free.hard_codes(held_x)
        tree_atlas = sum_lookup_rows(tree_reconstruction_rows.to(device), tree_current_codes)
        rows.append(
            _evaluate(
                seed=seed,
                arm="tree_free_task",
                prediction=tree_prediction,
                target=target,
                atlas=tree_atlas,
                x=held_x,
                codes=tree_current_codes,
                initial_codes=tree_initial_codes,
                thresholds=tree_free.thresholds,
                initial_thresholds=tree_compiled.thresholds.to(device),
                parameter_scalars=tree_action_rows.numel() + tree_compiled.thresholds.numel(),
                active_comparisons=args.tables * args.depth,
                rounds=args.depth,
                hard_replay=tree_free.hard_output(held_x)[0],
                curve=curves["tree_free_task"],
                seconds=train_seconds,
            )
        )

        grid_tied_prediction = grid_tied(held_x)
        grid_tied_atlas = sum_lookup_rows(grid_reconstruction_rows.to(device), grid_initial_codes)
        rows.append(
            _evaluate(
                seed=seed,
                arm="grid_tied_frozen",
                prediction=grid_tied_prediction,
                target=target,
                atlas=grid_tied_atlas,
                x=held_x,
                codes=grid_initial_codes,
                initial_codes=grid_initial_codes,
                thresholds=grid_tied.thresholds,
                initial_thresholds=grid_thresholds.to(device),
                parameter_scalars=grid_action_rows.numel() + grid_thresholds.numel(),
                active_comparisons=args.tables * 6,
                rounds=1,
                hard_replay=grid_tied.hard_output(held_x)[0],
                curve=[],
                seconds=compiler_seconds,
            )
        )

        grid_prediction = grid_free(held_x)
        grid_current_codes = grid_free.hard_codes(held_x)
        grid_atlas = sum_lookup_rows(grid_reconstruction_rows.to(device), grid_current_codes)
        rows.append(
            _evaluate(
                seed=seed,
                arm="grid_free_task",
                prediction=grid_prediction,
                target=target,
                atlas=grid_atlas,
                x=held_x,
                codes=grid_current_codes,
                initial_codes=grid_initial_codes,
                thresholds=grid_free.thresholds,
                initial_thresholds=grid_thresholds.to(device),
                parameter_scalars=grid_action_rows.numel() + grid_thresholds.numel(),
                active_comparisons=args.tables * 6,
                rounds=1,
                hard_replay=grid_free.hard_output(held_x)[0],
                curve=curves["grid_free_task"],
                seconds=train_seconds,
            )
        )

    audit: dict[str, object] = {
        "tree_cached_vs_direct_action_rows_max_abs_difference": float((tree_action_rows - tree_direct_rows).abs().max()),
        "grid_cached_vs_direct_action_rows_max_abs_difference": float((grid_action_rows - grid_direct_rows).abs().max()),
        "all_hard_replays_exact": all(row.hard_replay_max_error == 0 for row in rows),
        "all_finite": all(
            np.isfinite(value) for row in rows for value in (row.held_task_nmse, row.held_atlas_nmse, row.mean_entropy_bits, row.maximum_row_mass)
        ),
    }
    if (
        max(
            float(audit["tree_cached_vs_direct_action_rows_max_abs_difference"]),
            float(audit["grid_cached_vs_direct_action_rows_max_abs_difference"]),
        )
        > 5e-5
    ):
        raise AssertionError("linear action cache does not commute with the reconstruction solve")
    if not audit["all_hard_replays_exact"] or not audit["all_finite"]:
        raise AssertionError("mechanical result gate failed")
    state = {
        "teacher": teacher.detach().cpu(),
        "tree.reconstruction_rows": tree_reconstruction_rows,
        "tree.initial_action_rows": tree_action_rows,
        "grid.supports": grid_supports,
        "grid.initial_thresholds": grid_thresholds,
        "grid.reconstruction_rows": grid_reconstruction_rows,
        "grid.initial_action_rows": grid_action_rows,
        **{f"tree_free_task.{key}": value.detach().cpu() for key, value in tree_free.state_dict().items()},
        **{f"grid_free_task.{key}": value.detach().cpu() for key, value in grid_free.state_dict().items()},
    }
    return rows, audit, state


def summarize(rows: list[ArmResult]) -> dict[str, object]:
    seeds = sorted({row.seed for row in rows})
    lookup = {(row.seed, row.arm): row for row in rows}
    arms = {
        arm: {
            "held_task_r2_mean": sum(lookup[seed, arm].held_task_r2 for seed in seeds) / len(seeds),
            "held_task_nmse_mean": sum(lookup[seed, arm].held_task_nmse for seed in seeds) / len(seeds),
            "held_atlas_r2_mean": sum(lookup[seed, arm].held_atlas_r2 for seed in seeds) / len(seeds),
        }
        for arm in ARMS
    }
    dependency_gaps = [lookup[seed, "tree_free_task"].held_task_r2 - lookup[seed, "grid_free_task"].held_task_r2 for seed in seeds]
    tree_free_gains = [lookup[seed, "tree_free_task"].held_task_r2 - lookup[seed, "tree_tied_frozen"].held_task_r2 for seed in seeds]
    grid_free_gains = [lookup[seed, "grid_free_task"].held_task_r2 - lookup[seed, "grid_tied_frozen"].held_task_r2 for seed in seeds]
    mean_gap = sum(dependency_gaps) / len(dependency_gaps)
    return {
        "arms": arms,
        "effects": {
            "tree_minus_grid_free_task_r2": dependency_gaps,
            "tree_free_action_gain": tree_free_gains,
            "grid_free_action_gain": grid_free_gains,
        },
        "decisions": {
            "grid_is_strong_product_atlas": {
                "pass": min(lookup[seed, "grid_free_task"].held_task_r2 for seed in seeds) >= 0.80,
                "minimum_r2": min(lookup[seed, "grid_free_task"].held_task_r2 for seed in seeds),
            },
            "four_round_dependency_not_required": {
                "pass": mean_gap <= 0.01 and max(dependency_gaps) <= 0.02,
                "mean_tree_minus_grid_r2": mean_gap,
            },
            "adaptive_tree_materially_better": {
                "pass": min(dependency_gaps) > 0 and mean_gap >= 0.03,
                "mean_tree_minus_grid_r2": mean_gap,
            },
            "tree_free_action_adaptation": {
                "pass": min(tree_free_gains) > 0 and sum(tree_free_gains) / len(tree_free_gains) >= 0.01,
                "mean_gain": sum(tree_free_gains) / len(tree_free_gains),
            },
            "grid_free_action_adaptation": {
                "pass": min(grid_free_gains) > 0 and sum(grid_free_gains) / len(grid_free_gains) >= 0.01,
                "mean_gain": sum(grid_free_gains) / len(grid_free_gains),
            },
        },
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    rows: list[ArmResult] = []
    audits: dict[str, object] = {}
    state: dict[str, Tensor] = {}
    for seed in args.seeds:
        seed_rows, seed_audit, seed_state = fit_seed(seed, args)
        rows.extend(seed_rows)
        audits[str(seed)] = seed_audit
        state.update({f"seed{seed}.{key}": value for key, value in seed_state.items()})
        print(f"seed={seed} " + " ".join(f"{row.arm}:R2={row.held_task_r2:.6f}" for row in seed_rows), flush=True)
    result = {
        "schema": "product-grid-pc-action-factorial-v1",
        "protocol": {
            "dim": args.dim,
            "tables": args.tables,
            "rows_per_table": 16,
            "compiler_samples": args.compiler_samples,
            "held_samples": args.held_samples,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "ridge": args.ridge,
            "tau": args.tau,
            "seeds": list(args.seeds),
            "teacher": "random_Haar_SVD_condition16_unit_singular_RMS_linear_map",
            "tree": "balanced_adaptive_D2_binary_depth4",
            "grid": "balanced_parallel_D2_4x4_empirical_quartile",
            "action": "free_D64_rows_initialized_from_reconstruction_rows_times_teacher",
            "surrogate": "local_counterfactual_exact_hard_forward",
        },
        "rows": [asdict(row) for row in rows],
        "audits": audits,
        "summary": summarize(rows),
        "ledger": {
            "adaptive_tree": {
                "active_comparisons": 128,
                "comparison_rounds": 4,
                "stored_thresholds": 480,
                "stored_coordinate_ids": 128,
            },
            "parallel_grid": {
                "active_comparisons": 192,
                "comparison_rounds": 1,
                "stored_thresholds": 192,
                "stored_coordinate_ids": 64,
            },
            "shared": {"active_row_reads": 32, "active_output_scalar_reads": 2048, "action_row_scalars": 32768},
        },
    }
    artifact = Path(args.artifact)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    if artifact.exists():
        raise FileExistsError(artifact)
    torch.save({"schema": result["schema"], "protocol": result["protocol"], "state": state}, artifact)
    return result


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item) for item in value.split(",") if item)
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Matched adaptive D2 tree versus parallel D2 4x4 grid")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--tables", type=int, default=32)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--compiler-samples", type=int, default=32768)
    parser.add_argument("--held-samples", type=int, default=32768)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--seeds", type=_parse_seeds, default=(0, 1, 2))
    parser.add_argument("--log-every", type=int, default=300)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if (args.dim, args.tables, args.depth) != (64, 32, 4):
        parser.error("formal protocol requires D64/T32/depth4")
    if min(args.compiler_samples, args.held_samples, args.steps, args.batch_size) < 1:
        parser.error("sample and training counts must be positive")
    output, artifact = Path(args.output), Path(args.artifact)
    if output.resolve(strict=False) == artifact.resolve(strict=False) or output.exists() or artifact.exists():
        parser.error("output/artifact must differ and must not exist")
    return args


def main() -> None:
    args = parse_args()
    result = run(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(result["summary"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
