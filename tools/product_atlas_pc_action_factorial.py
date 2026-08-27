"""Product-reconstructive MADDNESS addresses with free PC-LUT actions.

The experiment keeps routing and lookup semantics inside ``HardLookupRouter``.
It owns only compilation, the training-only reconstruction auxiliary, metrics,
and artifact serialization.
"""

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

from tropnn.layers.hard_lookup import HardLookupRouter, sum_lookup_rows
from tropnn.layers.maddness import CompiledMaddness, FrozenMaddness, LocalCounterfactualMaddness
from tropnn.tools.maddness_end_to_end_ste_factorial import compile_original_maddness
from tropnn.tools.random_linear_address_action_factorial import orthogonal_teacher, sample_pair_anchors

ARMS = (
    "product_tied_frozen",
    "product_free_task",
    "product_free_task_reconstruction",
    "flat_pair_free_task",
)


@dataclass(frozen=True)
class ArmResult:
    seed: int
    teacher_mode: str
    arm: str
    held_task_mse: float
    held_task_nmse: float
    held_task_r2: float
    held_fixed_atlas_nmse: float
    held_fixed_atlas_r2: float
    linear_inverse_reconstruction_nmse: float
    orthogonal_task_inverse_nmse_gap: float | None
    mean_entropy_bits: float
    minimum_entropy_bits: float
    mean_observed_rows: float
    maximum_row_mass: float
    held_code_flip_fraction: float
    threshold_rms_drift: float
    action_row_rms_drift: float
    deploy_parameter_scalars: int
    training_auxiliary_scalars: int
    hard_replay_max_error: float
    training_curve: list[dict[str, float]]
    seconds: float


def _nmse(prediction: Tensor, target: Tensor) -> float:
    numerator = (prediction - target).square().mean()
    denominator = target.square().mean().clamp_min(1e-30)
    return float(numerator / denominator)


def anisotropic_teacher(dim: int, seed: int, device: torch.device) -> Tensor:
    """Full-rank random linear map with condition 16 and unit singular RMS."""

    left = orthogonal_teacher(dim, 100_000 + seed, torch.device("cpu")).double()
    right = orthogonal_teacher(dim, 200_000 + seed, torch.device("cpu")).double()
    singular = torch.logspace(
        -torch.log10(torch.tensor(4.0)).item(),
        torch.log10(torch.tensor(4.0)).item(),
        dim,
        dtype=torch.float64,
    )
    singular = singular / singular.square().mean().sqrt()
    return (left @ torch.diag(singular) @ right.T).float().to(device)


def _route_health(codes: Tensor, leaves: int) -> tuple[float, float, float, float]:
    entropies: list[float] = []
    observed: list[float] = []
    maximum_mass = 0.0
    for table in range(codes.shape[-1]):
        counts = torch.bincount(codes[:, table].detach().cpu(), minlength=leaves).double()
        probability = counts / counts.sum()
        positive = probability > 0
        entropies.append(float(-(probability[positive] * probability[positive].log2()).sum()))
        observed.append(float(positive.sum()))
        maximum_mass = max(maximum_mass, float(probability.max()))
    return sum(entropies) / len(entropies), min(entropies), sum(observed) / len(observed), maximum_mass


def fit_additive_rows(codes: Tensor, target: Tensor, leaves: int, ridge: float) -> Tensor:
    """Fit all additive table rows with one deterministic CPU float64 solve."""

    if codes.ndim != 2 or target.ndim != 2 or codes.shape[0] != target.shape[0]:
        raise ValueError("codes and target must be [samples,tables] and [samples,output]")
    if ridge <= 0:
        raise ValueError("ridge must be positive")
    code_np = codes.detach().cpu().numpy().astype(np.int64, copy=False)
    target_np = target.detach().cpu().numpy().astype(np.float64, copy=False)
    tables = code_np.shape[1]
    features = tables * leaves
    gram = np.zeros((features, features), dtype=np.float64)
    for left_table in range(tables):
        left = slice(leaves * left_table, leaves * (left_table + 1))
        for right_table in range(left_table, tables):
            right = slice(leaves * right_table, leaves * (right_table + 1))
            joint = leaves * code_np[:, left_table] + code_np[:, right_table]
            block = np.bincount(joint, minlength=leaves * leaves).reshape(leaves, leaves)
            gram[left, right] = block
            gram[right, left] = block.T
    rhs = np.zeros((features, target.shape[1]), dtype=np.float64)
    for table in range(tables):
        np.add.at(rhs[leaves * table : leaves * (table + 1)], code_np[:, table], target_np)
    rows = np.linalg.solve(gram + ridge * np.eye(features), rhs)
    return torch.from_numpy(rows.reshape(tables, leaves, target.shape[1]).astype(np.float32))


def cached_action_rows(compiled: CompiledMaddness, teacher: Tensor) -> Tensor:
    """Cache ``prototype @ teacher.T`` into the deployed wide action rows."""

    return torch.einsum("tkd,od->tko", compiled.prototypes.float(), teacher.detach().cpu().float())


def _with_rows(compiled: CompiledMaddness, rows: Tensor, *, thresholds: Tensor | None = None) -> CompiledMaddness:
    return CompiledMaddness(
        split_indices=compiled.split_indices.detach().clone(),
        thresholds=(compiled.thresholds if thresholds is None else thresholds).detach().clone(),
        encoder_centroids=compiled.encoder_centroids.detach().clone(),
        prototypes=rows.detach().clone(),
    )


def _make_flat_router(
    dim: int,
    tables: int,
    depth: int,
    supports: Tensor,
    thresholds: Tensor,
    rows: Tensor,
    *,
    tau: float,
    trainable: bool,
) -> HardLookupRouter:
    return HardLookupRouter(
        dim,
        dim,
        depth=depth,
        predicate="pair",
        topology="flat",
        support_layout="level",
        supports=supports,
        thresholds=thresholds,
        rows=rows,
        surrogate="local_counterfactual" if trainable else "none",
        tau=tau,
        trainable_thresholds=trainable,
        trainable_rows=trainable,
    )


def _mask_reconstruction_row_gradients(model: LocalCounterfactualMaddness, dim: int) -> None:
    if not isinstance(model.rows, torch.nn.Parameter):
        raise TypeError("combined product rows must be trainable")
    mask = torch.zeros_like(model.rows)
    mask[..., :dim] = 1
    model.rows.register_hook(lambda gradient: gradient * mask)


def _evaluate_arm(
    *,
    seed: int,
    teacher_mode: str,
    arm: str,
    model: torch.nn.Module,
    prediction: Tensor,
    atlas: Tensor,
    target: Tensor,
    x: Tensor,
    teacher: Tensor,
    codes: Tensor,
    initial_codes: Tensor,
    initial_thresholds: Tensor,
    thresholds: Tensor,
    initial_action_rows: Tensor,
    action_rows: Tensor,
    deploy_parameter_scalars: int,
    training_auxiliary_scalars: int,
    hard_replay: Tensor,
    curve: list[dict[str, float]],
    seconds: float,
    leaves: int,
) -> ArmResult:
    del model
    task_nmse = _nmse(prediction, target)
    atlas_nmse = _nmse(atlas, x)
    inverse = torch.linalg.solve(teacher, prediction.T).T
    inverse_nmse = _nmse(inverse, x)
    entropy, minimum_entropy, observed, maximum_mass = _route_health(codes, leaves)
    return ArmResult(
        seed=seed,
        teacher_mode=teacher_mode,
        arm=arm,
        held_task_mse=float((prediction - target).square().mean()),
        held_task_nmse=task_nmse,
        held_task_r2=1 - task_nmse,
        held_fixed_atlas_nmse=atlas_nmse,
        held_fixed_atlas_r2=1 - atlas_nmse,
        linear_inverse_reconstruction_nmse=inverse_nmse,
        orthogonal_task_inverse_nmse_gap=inverse_nmse - task_nmse if teacher_mode == "orthogonal" else None,
        mean_entropy_bits=entropy,
        minimum_entropy_bits=minimum_entropy,
        mean_observed_rows=observed,
        maximum_row_mass=maximum_mass,
        held_code_flip_fraction=float((codes != initial_codes).float().mean()),
        threshold_rms_drift=float((thresholds - initial_thresholds).square().mean().sqrt()),
        action_row_rms_drift=float((action_rows - initial_action_rows).square().mean().sqrt()),
        deploy_parameter_scalars=deploy_parameter_scalars,
        training_auxiliary_scalars=training_auxiliary_scalars,
        hard_replay_max_error=float((prediction - hard_replay).abs().max()),
        training_curve=curve,
        seconds=seconds,
    )


def fit_seed(seed: int, args: argparse.Namespace) -> tuple[list[ArmResult], dict[str, object], dict[str, Tensor]]:
    device = torch.device(args.device)
    leaves = 2**args.depth
    teacher = orthogonal_teacher(args.dim, seed, device) if args.teacher_mode == "orthogonal" else anisotropic_teacher(args.dim, seed, device)
    compiler_generator = torch.Generator(device="cpu").manual_seed(40_000 + seed)
    compiler_x = torch.randn(args.compiler_samples, args.dim, generator=compiler_generator)
    compiler_target = compiler_x @ teacher.detach().cpu().T

    compile_started = time.perf_counter()
    compiled = compile_original_maddness(compiler_x, args.tables, args.ridge)
    tied_rows = cached_action_rows(compiled, teacher)
    product_codes = FrozenMaddness(compiled).hard_codes(compiler_x)
    directly_fitted_product_rows = fit_additive_rows(product_codes, compiler_target, leaves, args.ridge)
    tied_free_max_difference = float((tied_rows - directly_fitted_product_rows).abs().max())

    product_action = _with_rows(compiled, tied_rows)
    product_tied = FrozenMaddness(product_action).to(device)
    product_free = LocalCounterfactualMaddness(product_action, args.tau).to(device)
    combined_rows = torch.cat((tied_rows, compiled.prototypes), dim=-1)
    product_reconstructive = LocalCounterfactualMaddness(_with_rows(compiled, combined_rows), args.tau).to(device)
    _mask_reconstruction_row_gradients(product_reconstructive, args.dim)

    flat_supports = sample_pair_anchors(args.tables * args.depth, args.dim, 80_000 + seed).view(args.tables, args.depth, 2)
    flat_thresholds = torch.zeros(args.tables, args.depth)
    flat_zero_rows = torch.zeros(args.tables, leaves, args.dim)
    flat_compiler_route = _make_flat_router(
        args.dim,
        args.tables,
        args.depth,
        flat_supports,
        flat_thresholds,
        flat_zero_rows,
        tau=args.tau,
        trainable=False,
    )
    flat_codes = flat_compiler_route.hard_codes(compiler_x)
    flat_action_rows = fit_additive_rows(flat_codes, compiler_target, leaves, args.ridge)
    flat_atlas_rows = fit_additive_rows(flat_codes, compiler_x, leaves, args.ridge)
    flat_free = _make_flat_router(
        args.dim,
        args.tables,
        args.depth,
        flat_supports,
        flat_thresholds,
        flat_action_rows,
        tau=args.tau,
        trainable=True,
    ).to(device)
    compiler_seconds = time.perf_counter() - compile_started

    models = {
        "product_free_task": product_free,
        "product_free_task_reconstruction": product_reconstructive,
        "flat_pair_free_task": flat_free,
    }
    optimizer = torch.optim.AdamW(
        [parameter for model in models.values() for parameter in model.parameters()],
        lr=args.lr,
        weight_decay=0,
    )
    curves: dict[str, list[dict[str, float]]] = {arm: [] for arm in models}
    train_generator = torch.Generator(device=device).manual_seed(50_000 + seed)
    train_started = time.perf_counter()
    for step in range(1, args.steps + 1):
        x = torch.randn(args.batch_size, args.dim, generator=train_generator, device=device)
        target = x @ teacher.T
        optimizer.zero_grad(set_to_none=True)
        product_prediction = product_free(x)
        combined_prediction = product_reconstructive(x)
        rec_action, rec_atlas = combined_prediction.split(args.dim, dim=-1)
        flat_prediction = flat_free(x)
        task_loss = F.mse_loss(product_prediction, target)
        rec_task_loss = F.mse_loss(rec_action, target)
        rec_loss = F.mse_loss(rec_atlas, x)
        flat_loss = F.mse_loss(flat_prediction, target)
        total = task_loss + rec_task_loss + args.reconstruction_weight * rec_loss + flat_loss
        total.backward()
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            curves["product_free_task"].append({"step": float(step), "task_mse": float(task_loss.detach())})
            curves["product_free_task_reconstruction"].append(
                {
                    "step": float(step),
                    "task_mse": float(rec_task_loss.detach()),
                    "reconstruction_mse": float(rec_loss.detach()),
                }
            )
            curves["flat_pair_free_task"].append({"step": float(step), "task_mse": float(flat_loss.detach())})
    train_seconds = time.perf_counter() - train_started

    held_generator = torch.Generator(device=device).manual_seed(70_000 + seed)
    held_x = torch.randn(args.held_samples, args.dim, generator=held_generator, device=device)
    held_target = held_x @ teacher.T
    product_initial_codes = product_tied.hard_codes(held_x)
    flat_initial = _make_flat_router(
        args.dim,
        args.tables,
        args.depth,
        flat_supports,
        flat_thresholds,
        flat_action_rows,
        tau=args.tau,
        trainable=False,
    ).to(device)
    flat_initial_codes = flat_initial.hard_codes(held_x)

    rows: list[ArmResult] = []
    with torch.no_grad():
        tied_prediction = product_tied(held_x)
        tied_codes = product_tied.hard_codes(held_x)
        tied_atlas = FrozenMaddness(compiled).to(device)(held_x)
        dense_spelling = tied_atlas @ teacher.T
        cached_dense_max_difference = float((tied_prediction - dense_spelling).abs().max())
        rows.append(
            _evaluate_arm(
                seed=seed,
                teacher_mode=args.teacher_mode,
                arm="product_tied_frozen",
                model=product_tied,
                prediction=tied_prediction,
                atlas=tied_atlas,
                target=held_target,
                x=held_x,
                teacher=teacher,
                codes=tied_codes,
                initial_codes=product_initial_codes,
                initial_thresholds=compiled.thresholds.to(device),
                thresholds=product_tied.thresholds,
                initial_action_rows=tied_rows.to(device),
                action_rows=product_tied.rows,
                deploy_parameter_scalars=tied_rows.numel() + compiled.thresholds.numel(),
                training_auxiliary_scalars=0,
                hard_replay=product_tied.hard_output(held_x)[0],
                curve=[],
                seconds=compiler_seconds,
                leaves=leaves,
            )
        )

        free_prediction = product_free(held_x)
        free_codes = product_free.hard_codes(held_x)
        free_atlas = sum_lookup_rows(compiled.prototypes.to(device), free_codes)
        rows.append(
            _evaluate_arm(
                seed=seed,
                teacher_mode=args.teacher_mode,
                arm="product_free_task",
                model=product_free,
                prediction=free_prediction,
                atlas=free_atlas,
                target=held_target,
                x=held_x,
                teacher=teacher,
                codes=free_codes,
                initial_codes=product_initial_codes,
                initial_thresholds=compiled.thresholds.to(device),
                thresholds=product_free.thresholds,
                initial_action_rows=tied_rows.to(device),
                action_rows=product_free.rows,
                deploy_parameter_scalars=tied_rows.numel() + compiled.thresholds.numel(),
                training_auxiliary_scalars=0,
                hard_replay=product_free.hard_output(held_x)[0],
                curve=curves["product_free_task"],
                seconds=train_seconds,
                leaves=leaves,
            )
        )

        combined_prediction = product_reconstructive(held_x)
        rec_prediction, rec_atlas = combined_prediction.split(args.dim, dim=-1)
        rec_codes = product_reconstructive.hard_codes(held_x)
        rec_action_rows = product_reconstructive.rows[..., : args.dim]
        deployed_rec = FrozenMaddness(_with_rows(compiled, rec_action_rows.cpu(), thresholds=product_reconstructive.thresholds.cpu())).to(device)
        deployed_rec_output = deployed_rec(held_x)
        rows.append(
            _evaluate_arm(
                seed=seed,
                teacher_mode=args.teacher_mode,
                arm="product_free_task_reconstruction",
                model=product_reconstructive,
                prediction=rec_prediction,
                atlas=rec_atlas,
                target=held_target,
                x=held_x,
                teacher=teacher,
                codes=rec_codes,
                initial_codes=product_initial_codes,
                initial_thresholds=compiled.thresholds.to(device),
                thresholds=product_reconstructive.thresholds,
                initial_action_rows=tied_rows.to(device),
                action_rows=rec_action_rows,
                deploy_parameter_scalars=tied_rows.numel() + compiled.thresholds.numel(),
                training_auxiliary_scalars=compiled.prototypes.numel(),
                hard_replay=deployed_rec_output,
                curve=curves["product_free_task_reconstruction"],
                seconds=train_seconds,
                leaves=leaves,
            )
        )

        flat_prediction = flat_free(held_x)
        current_flat_codes = flat_free.hard_codes(held_x)
        flat_atlas = sum_lookup_rows(flat_atlas_rows.to(device), current_flat_codes)
        rows.append(
            _evaluate_arm(
                seed=seed,
                teacher_mode=args.teacher_mode,
                arm="flat_pair_free_task",
                model=flat_free,
                prediction=flat_prediction,
                atlas=flat_atlas,
                target=held_target,
                x=held_x,
                teacher=teacher,
                codes=current_flat_codes,
                initial_codes=flat_initial_codes,
                initial_thresholds=flat_thresholds.to(device),
                thresholds=flat_free.thresholds,
                initial_action_rows=flat_action_rows.to(device),
                action_rows=flat_free.rows,
                deploy_parameter_scalars=flat_action_rows.numel() + flat_thresholds.numel(),
                training_auxiliary_scalars=0,
                hard_replay=flat_free.hard_output(held_x)[0],
                curve=curves["flat_pair_free_task"],
                seconds=train_seconds,
                leaves=leaves,
            )
        )

    frozen_reconstruction_max_difference = float((product_reconstructive.rows[..., args.dim :].detach().cpu() - compiled.prototypes).abs().max())
    audit: dict[str, object] = {
        "tied_vs_direct_free_ridge_max_abs_difference": tied_free_max_difference,
        "cached_action_vs_reconstruct_then_dense_max_abs_difference": cached_dense_max_difference,
        "frozen_reconstruction_rows_max_abs_difference": frozen_reconstruction_max_difference,
        "orthogonal_task_and_inverse_nmse_max_abs_difference": (
            max(abs(row.orthogonal_task_inverse_nmse_gap or 0.0) for row in rows) if args.teacher_mode == "orthogonal" else None
        ),
        "all_hard_replays_exact": all(row.hard_replay_max_error == 0 for row in rows),
        "all_finite": all(
            np.isfinite(value)
            for row in rows
            for value in (
                row.held_task_nmse,
                row.held_fixed_atlas_nmse,
                row.mean_entropy_bits,
                row.maximum_row_mass,
            )
        ),
    }
    if tied_free_max_difference > 5e-5:
        raise AssertionError("linear output-row ridge does not commute with the reconstruction solve")
    if cached_dense_max_difference > 5e-5:
        raise AssertionError("cached action rows disagree with reconstruct-then-dense spelling")
    if frozen_reconstruction_max_difference != 0:
        raise AssertionError("training changed the frozen reconstruction auxiliary rows")
    if not audit["all_hard_replays_exact"]:
        raise AssertionError("a deployed hard action disagrees with its training-module forward")
    if args.teacher_mode == "orthogonal" and float(audit["orthogonal_task_and_inverse_nmse_max_abs_difference"]) > 1e-6:
        raise AssertionError("orthogonal teacher broke task/reconstruction NMSE equivalence")
    if not audit["all_finite"]:
        raise FloatingPointError("nonfinite factorial metric")
    state = {
        "teacher": teacher.cpu(),
        "product.split_indices": compiled.split_indices,
        "product.initial_thresholds": compiled.thresholds,
        "product.encoder_centroids": compiled.encoder_centroids,
        "product.reconstruction_rows": compiled.prototypes,
        "product.initial_action_rows": tied_rows,
        "flat.supports": flat_supports,
        "flat.initial_thresholds": flat_thresholds,
        "flat.reconstruction_rows": flat_atlas_rows,
        "flat.initial_action_rows": flat_action_rows,
        **{f"product_free_task.{key}": value.detach().cpu() for key, value in product_free.state_dict().items()},
        **{f"product_free_task_reconstruction.{key}": value.detach().cpu() for key, value in product_reconstructive.state_dict().items()},
        **{f"flat_pair_free_task.{key}": value.detach().cpu() for key, value in flat_free.state_dict().items()},
    }
    return rows, audit, state


def summarize(rows: list[ArmResult]) -> dict[str, object]:
    seeds = sorted({row.seed for row in rows})
    lookup = {(row.seed, row.arm): row for row in rows}
    arms = {
        arm: {
            "held_task_r2_mean": sum(lookup[seed, arm].held_task_r2 for seed in seeds) / len(seeds),
            "held_task_nmse_mean": sum(lookup[seed, arm].held_task_nmse for seed in seeds) / len(seeds),
            "held_fixed_atlas_r2_mean": sum(lookup[seed, arm].held_fixed_atlas_r2 for seed in seeds) / len(seeds),
        }
        for arm in ARMS
    }
    free_gain = [lookup[seed, "product_free_task"].held_task_r2 - lookup[seed, "product_tied_frozen"].held_task_r2 for seed in seeds]
    rec_task_gain = [lookup[seed, "product_free_task_reconstruction"].held_task_r2 - lookup[seed, "product_free_task"].held_task_r2 for seed in seeds]
    rec_atlas_gain = [
        lookup[seed, "product_free_task_reconstruction"].held_fixed_atlas_r2 - lookup[seed, "product_free_task"].held_fixed_atlas_r2 for seed in seeds
    ]
    product_flat_gain = [
        lookup[seed, "product_free_task_reconstruction"].held_task_r2 - lookup[seed, "flat_pair_free_task"].held_task_r2 for seed in seeds
    ]
    return {
        "arms": arms,
        "effects": {
            "free_action_vs_tied_task_r2": free_gain,
            "reconstruction_aux_vs_task_only_task_r2": rec_task_gain,
            "reconstruction_aux_vs_task_only_fixed_atlas_r2": rec_atlas_gain,
            "product_reconstructive_vs_flat_task_r2": product_flat_gain,
        },
        "decisions": {
            "free_action_adaptation": {
                "pass": min(free_gain) > 0 and sum(free_gain) / len(free_gain) >= 0.01,
                "mean_gain": sum(free_gain) / len(free_gain),
            },
            "reconstruction_preservation": {
                "pass": min(rec_atlas_gain) > 0
                and sum(rec_atlas_gain) / len(rec_atlas_gain) >= 0.01
                and sum(rec_task_gain) / len(rec_task_gain) >= -0.005,
                "mean_atlas_gain": sum(rec_atlas_gain) / len(rec_atlas_gain),
                "mean_task_gain": sum(rec_task_gain) / len(rec_task_gain),
            },
            "combined_product_vs_flat": {
                "pass": min(product_flat_gain) > 0 and sum(product_flat_gain) / len(product_flat_gain) >= 0.10,
                "mean_gain": sum(product_flat_gain) / len(product_flat_gain),
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
        print(
            f"seed={seed} " + " ".join(f"{row.arm}:R2={row.held_task_r2:.6f}" for row in seed_rows),
            flush=True,
        )
    result = {
        "schema": "product-atlas-pc-action-factorial-v1",
        "protocol": {
            "dim": args.dim,
            "tables": args.tables,
            "depth": args.depth,
            "leaves": 2**args.depth,
            "subspace_dim": args.dim // args.tables,
            "active_comparisons": args.tables * args.depth,
            "active_lookups": args.tables,
            "active_vector_additions": args.tables,
            "compiler_samples": args.compiler_samples,
            "held_samples": args.held_samples,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "ridge": args.ridge,
            "tau": args.tau,
            "reconstruction_weight": args.reconstruction_weight,
            "seeds": list(args.seeds),
            "teacher": (
                "QR_Haar_orthogonal_linear_map" if args.teacher_mode == "orthogonal" else "random_Haar_SVD_condition16_unit_singular_RMS_linear_map"
            ),
            "teacher_mode": args.teacher_mode,
            "surrogate": "local_counterfactual",
            "split_coordinates_frozen": True,
            "reconstruction_auxiliary_rows_frozen": True,
            "reconstruction_auxiliary_dropped_at_deployment": True,
            "orthogonal_task_reconstruction_equivalence": args.teacher_mode == "orthogonal",
        },
        "rows": [asdict(row) for row in rows],
        "audits": audits,
        "summary": summarize(rows),
        "semantic_ledger": {
            "product_tree": {
                "stored_thresholds": args.tables * (2**args.depth - 1),
                "stored_level_coordinates": args.tables * args.depth,
                "dependent_steps": args.depth,
            },
            "flat_pair": {
                "stored_thresholds": args.tables * args.depth,
                "stored_pair_endpoints": 2 * args.tables * args.depth,
                "dependent_steps": 1,
            },
            "shared_deployment": {
                "action_row_scalars": args.tables * (2**args.depth) * args.dim,
                "active_comparisons": args.tables * args.depth,
                "active_row_reads": args.tables,
                "active_output_scalar_reads": args.tables * args.dim,
            },
            "reconstruction_auxiliary_training_only_scalars": args.tables * (2**args.depth) * args.dim,
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
    parser = argparse.ArgumentParser(description="Product-atlas address x free PC-LUT action D64 factorial")
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
    parser.add_argument("--reconstruction-weight", type=float, default=1.0)
    parser.add_argument("--teacher-mode", choices=("anisotropic", "orthogonal"), default="anisotropic")
    parser.add_argument("--seeds", type=_parse_seeds, default=(0, 1, 2))
    parser.add_argument("--log-every", type=int, default=300)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.dim != 64 or args.tables != 32 or args.depth != 4:
        parser.error("formal protocol requires D64/T32/depth4")
    if args.dim % args.tables:
        parser.error("input dimension must be divisible by table count")
    if args.reconstruction_weight < 0:
        parser.error("reconstruction weight must be nonnegative")
    output, artifact = Path(args.output), Path(args.artifact)
    if output.resolve(strict=False) == artifact.resolve(strict=False):
        parser.error("output and artifact paths must differ")
    if output.exists() or artifact.exists():
        parser.error("output and artifact paths must not exist")
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
