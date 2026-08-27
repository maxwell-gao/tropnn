"""End-to-end EMNIST task-loss training for one hard MADDNESS layer.

The deployed MADDNESS head always executes four hard tree comparisons per
table followed by one row lookup and vector add.  Training compares a soft-PQ
surrogate with the local sibling-subtree counterfactual inspired by
``ref/spikes.tex``.  A route-calibrated scratch arm receives no labels, dense
teacher actions, or offline output compiler before task training.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from tropnn.layers.maddness import CompiledMaddness, FrozenMaddness, LocalCounterfactualMaddness, SoftPQMaddness
from tropnn.tools.emnist_payload_dtype_sweep import _load_emnist_split
from tropnn.tools.maddness_end_to_end_ste_factorial import (
    _fit_tree,
    _leaf_stats,
)

ARMS = (
    "dense_pretrained",
    "dense_continued",
    "compiled_frozen",
    "soft_pq_task_finetune",
    "local_counterfactual_task_finetune",
    "local_counterfactual_scratch",
)


@dataclass(frozen=True)
class Evaluation:
    arm: str
    held_ce: float
    held_accuracy: float
    mean_entropy_bits: float | None
    minimum_entropy_bits: float | None
    maximum_row_mass: float | None
    code_flip_fraction: float | None
    threshold_rms_drift: float | None
    hard_soft_logit_nmse: float | None
    hard_forward_exact: bool | None


@dataclass(frozen=True)
class SeedResult:
    seed: int
    compiler_logit_nmse: float
    compiler_seconds: float
    dense_pretrain_curve: list[dict[str, float]]
    task_training_curves: dict[str, list[dict[str, float]]]
    first_step_gradient_norms: dict[str, dict[str, float]]
    evaluations: list[Evaluation]
    seconds: float


class DenseStemClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, classes: int) -> None:
        super().__init__()
        self.stem = nn.Linear(input_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, classes)

    def features(self, x: Tensor) -> Tensor:
        return F.gelu(self.stem(x.flatten(1)))

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.features(x))


class MaddnessStemClassifier(nn.Module):
    def __init__(self, stem: nn.Linear, head: nn.Module) -> None:
        super().__init__()
        self.stem = stem
        self.head = head

    def features(self, x: Tensor) -> Tensor:
        return F.gelu(self.stem(x.flatten(1)))

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.features(x))


def _build_gram(codes: np.ndarray, tables: int) -> np.ndarray:
    gram = np.zeros((16 * tables, 16 * tables), dtype=np.float64)
    for left_table in range(tables):
        left = slice(16 * left_table, 16 * (left_table + 1))
        for right_table in range(left_table, tables):
            right = slice(16 * right_table, 16 * (right_table + 1))
            joint = 16 * codes[:, left_table] + codes[:, right_table]
            block = np.bincount(joint, minlength=256).reshape(16, 16)
            gram[left, right] = block
            gram[right, left] = block.T
    return gram


def compile_maddness_targets(x: Tensor, target: Tensor, tables: int, ridge: float = 1.0) -> CompiledMaddness:
    """Compile balanced D2 trees and additive output rows for arbitrary targets."""

    if x.ndim != 2 or target.ndim != 2 or x.shape[0] != target.shape[0]:
        raise ValueError("x and target must be aligned matrices")
    if x.shape[1] % tables:
        raise ValueError("input dimension must be divisible by table count")
    x_np = x.detach().cpu().numpy().astype(np.float32, copy=False)
    target_np = target.detach().cpu().numpy().astype(np.float64, copy=False)
    width = x.shape[1] // tables
    all_indices: list[np.ndarray] = []
    all_thresholds: list[np.ndarray] = []
    all_centroids: list[np.ndarray] = []
    all_codes: list[np.ndarray] = []
    for table in range(tables):
        start = table * width
        indices, thresholds, codes, centroids = _fit_tree(x_np[:, start : start + width])
        all_indices.append(indices + start)
        all_thresholds.append(thresholds)
        all_centroids.append(centroids)
        all_codes.append(codes)
    codes_np = np.stack(all_codes, axis=1)
    gram = _build_gram(codes_np, tables)
    rhs = np.zeros((16 * tables, target.shape[1]), dtype=np.float64)
    for table in range(tables):
        np.add.at(rhs[16 * table : 16 * (table + 1)], codes_np[:, table], target_np)
    prototypes = np.linalg.solve(gram + ridge * np.eye(16 * tables), rhs).reshape(tables, 16, target.shape[1])
    return CompiledMaddness(
        split_indices=torch.from_numpy(np.stack(all_indices)),
        thresholds=torch.from_numpy(np.stack(all_thresholds)),
        encoder_centroids=torch.from_numpy(np.stack(all_centroids)),
        prototypes=torch.from_numpy(prototypes.astype(np.float32)),
    )


def compile_route_only_scratch(x: Tensor, tables: int, output_dim: int, seed: int, prototype_std: float) -> CompiledMaddness:
    """Calibrate balanced unlabeled routes, but initialize every task action randomly."""

    zeros = torch.zeros(x.shape[0], output_dim, dtype=x.dtype)
    route = compile_maddness_targets(x, zeros, tables=tables, ridge=1.0)
    generator = torch.Generator(device="cpu").manual_seed(91_000 + seed)
    prototypes = torch.randn(route.prototypes.shape, generator=generator) * prototype_std
    return CompiledMaddness(
        split_indices=route.split_indices,
        thresholds=route.thresholds,
        encoder_centroids=route.encoder_centroids,
        prototypes=prototypes,
    )


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _ordered_loader(x: Tensor, y: Tensor, batch_size: int, workers: int, pin_memory: bool) -> DataLoader:
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=pin_memory)


def _shuffled_loader(x: Tensor, y: Tensor, batch_size: int, workers: int, pin_memory: bool, seed: int) -> DataLoader:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return DataLoader(
        TensorDataset(x, y),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=workers,
        pin_memory=pin_memory,
    )


def _train_epoch(
    models: dict[str, nn.Module],
    optimizers: dict[str, torch.optim.Optimizer],
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, tuple[float, float]], dict[str, dict[str, float]]]:
    for model in models.values():
        model.train()
    loss_sums = {name: 0.0 for name in models}
    correct = {name: 0 for name in models}
    count = 0
    first_gradients: dict[str, dict[str, float]] = {}
    for batch_index, (x, target) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        logits = {name: model(x) for name, model in models.items()}
        losses = {name: F.cross_entropy(value, target) for name, value in logits.items()}
        sum(losses.values()).backward()
        if batch_index == 0:
            for name, model in models.items():
                gradients = {
                    "stem": _gradient_norm(model.stem.parameters()),  # type: ignore[attr-defined]
                    "all": _gradient_norm(model.parameters()),
                }
                head = getattr(model, "head", None)
                gradients["thresholds"] = _parameter_gradient_norm(getattr(head, "thresholds", None))
                gradients["prototypes"] = _parameter_gradient_norm(getattr(head, "prototypes", None))
                first_gradients[name] = gradients
        for optimizer in optimizers.values():
            optimizer.step()
        batch = target.numel()
        for name, loss in losses.items():
            loss_sums[name] += float(loss.detach()) * batch
            correct[name] += int((logits[name].detach().argmax(-1) == target).sum())
        count += batch
    return {name: (loss_sums[name] / count, correct[name] / count) for name in models}, first_gradients


def _gradient_norm(parameters: object) -> float:
    total = 0.0
    for parameter in parameters:  # type: ignore[union-attr]
        if parameter.grad is not None:
            total += float(parameter.grad.detach().square().sum())
    return math.sqrt(total)


def _parameter_gradient_norm(parameter: Tensor | None) -> float:
    if parameter is None or parameter.grad is None:
        return 0.0
    return float(parameter.grad.detach().square().sum().sqrt())


@torch.no_grad()
def _evaluate_plain(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    count = 0
    for x, target in loader:
        x = x.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        logits = model(x)
        loss_sum += float(F.cross_entropy(logits, target, reduction="sum"))
        correct += int((logits.argmax(-1) == target).sum())
        count += target.numel()
    return loss_sum / count, correct / count


@torch.no_grad()
def _evaluate_maddness(
    arm: str,
    model: MaddnessStemClassifier,
    loader: DataLoader,
    device: torch.device,
    initial: CompiledMaddness,
) -> Evaluation:
    ce, accuracy = _evaluate_plain(model, loader, device)
    all_codes: list[Tensor] = []
    all_initial_codes: list[Tensor] = []
    mismatch_numerator = 0.0
    mismatch_denominator = 0.0
    exact = True
    model.eval()
    for x, _target in loader:
        x = x.to(device, non_blocking=True)
        features = model.features(x)
        head = model.head
        codes = head.hard_codes(features)  # type: ignore[attr-defined]
        initial_codes = FrozenMaddness(initial).to(device).hard_codes(features)
        direct = head.hard_output(features)[0]  # type: ignore[attr-defined]
        exact = exact and torch.equal(model(x), direct)
        all_codes.append(codes.cpu())
        all_initial_codes.append(initial_codes.cpu())
        if isinstance(head, SoftPQMaddness):
            hard, soft = head.outputs(features)
            mismatch_numerator += float((hard - soft).square().sum())
            mismatch_denominator += float(hard.square().sum())
    codes = torch.cat(all_codes)
    initial_codes = torch.cat(all_initial_codes)
    entropy, minimum_entropy, maximum_mass = _leaf_stats(codes)
    threshold_drift = float((model.head.thresholds.detach().cpu() - initial.thresholds).square().mean().sqrt())  # type: ignore[attr-defined]
    mismatch = None if not isinstance(model.head, SoftPQMaddness) else mismatch_numerator / max(mismatch_denominator, 1e-30)
    return Evaluation(
        arm=arm,
        held_ce=ce,
        held_accuracy=accuracy,
        mean_entropy_bits=entropy,
        minimum_entropy_bits=minimum_entropy,
        maximum_row_mass=maximum_mass,
        code_flip_fraction=float((codes != initial_codes).float().mean()),
        threshold_rms_drift=threshold_drift,
        hard_soft_logit_nmse=mismatch,
        hard_forward_exact=exact,
    )


def _plain_evaluation(arm: str, model: nn.Module, loader: DataLoader, device: torch.device) -> Evaluation:
    ce, accuracy = _evaluate_plain(model, loader, device)
    return Evaluation(arm, ce, accuracy, None, None, None, None, None, None, None)


@torch.no_grad()
def _capture_dense_targets(
    model: DenseStemClassifier,
    x: Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    features: list[Tensor] = []
    targets: list[Tensor] = []
    model.eval()
    for start in range(0, x.shape[0], batch_size):
        batch = x[start : start + batch_size].to(device)
        feature = model.features(batch)
        features.append(feature.cpu())
        targets.append(model.head(feature).cpu())
    return torch.cat(features), torch.cat(targets)


def _clone_stem(model: DenseStemClassifier) -> nn.Linear:
    return copy.deepcopy(model.stem)


def _train_models_for_epochs(
    models: dict[str, nn.Module],
    optimizers: dict[str, torch.optim.Optimizer],
    x: Tensor,
    y: Tensor,
    *,
    epochs: int,
    batch_size: int,
    workers: int,
    device: torch.device,
    loader_seed_base: int,
    phase: str,
) -> tuple[dict[str, list[dict[str, float]]], dict[str, dict[str, float]]]:
    curves = {name: [] for name in models}
    first_gradients: dict[str, dict[str, float]] = {}
    for epoch in range(epochs):
        loader = _shuffled_loader(
            x,
            y,
            batch_size,
            workers,
            pin_memory=device.type == "cuda",
            seed=loader_seed_base + epoch,
        )
        metrics, gradients = _train_epoch(models, optimizers, loader, device)
        if epoch == 0:
            first_gradients = gradients
        print(
            f"phase={phase} epoch={epoch + 1}/{epochs} "
            + " ".join(f"{name}:ce={value[0]:.6f},acc={value[1]:.6f}" for name, value in metrics.items()),
            flush=True,
        )
        for name, (ce, accuracy) in metrics.items():
            curves[name].append({"epoch": float(epoch + 1), "train_ce": ce, "train_accuracy": accuracy})
    return curves, first_gradients


def fit_seed(
    seed: int, args: argparse.Namespace, train_x: Tensor, train_y: Tensor, held_loader: DataLoader, classes: int
) -> tuple[SeedResult, dict[str, Tensor]]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(args.device)
    started = time.perf_counter()

    dense_pretrained = DenseStemClassifier(train_x.shape[1], args.hidden_dim, classes).to(device)
    scratch_stem = nn.Linear(train_x.shape[1], args.hidden_dim).to(device)
    with torch.no_grad():
        scratch_features = F.gelu(scratch_stem(train_x[: args.compiler_samples].to(device))).cpu()
    scratch_initial = compile_route_only_scratch(
        scratch_features,
        args.tables,
        classes,
        seed,
        args.scratch_prototype_std,
    )
    scratch_model = MaddnessStemClassifier(
        scratch_stem,
        LocalCounterfactualMaddness(scratch_initial, args.tau).to(device),
    ).to(device)

    pretrain_models: dict[str, nn.Module] = {
        "dense_pretrained": dense_pretrained,
        "local_counterfactual_scratch": scratch_model,
    }
    pretrain_optimizers = {name: torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0) for name, model in pretrain_models.items()}
    pretrain_curves, pretrain_gradients = _train_models_for_epochs(
        pretrain_models,
        pretrain_optimizers,
        train_x,
        train_y,
        epochs=args.pretrain_epochs,
        batch_size=args.batch_size,
        workers=args.workers,
        device=device,
        loader_seed_base=100_000 + seed * 100,
        phase="pretrain",
    )

    compiler_features, compiler_targets = _capture_dense_targets(
        dense_pretrained,
        train_x[: args.compiler_samples],
        args.batch_size,
        device,
    )
    compiler_started = time.perf_counter()
    compiled = compile_maddness_targets(compiler_features, compiler_targets, args.tables, args.ridge)
    compiler_seconds = time.perf_counter() - compiler_started
    with torch.no_grad():
        compiled_logits = FrozenMaddness(compiled)(compiler_features)
        compiler_nmse = float((compiled_logits - compiler_targets).square().mean() / compiler_targets.square().mean().clamp_min(1e-30))

    frozen_model = MaddnessStemClassifier(_clone_stem(dense_pretrained), FrozenMaddness(compiled)).to(device)
    soft_model = MaddnessStemClassifier(
        _clone_stem(dense_pretrained),
        SoftPQMaddness(compiled, args.soft_temperature),
    ).to(device)
    local_model = MaddnessStemClassifier(
        _clone_stem(dense_pretrained),
        LocalCounterfactualMaddness(compiled, args.tau),
    ).to(device)
    dense_continued = copy.deepcopy(dense_pretrained)
    task_models: dict[str, nn.Module] = {
        "dense_continued": dense_continued,
        "soft_pq_task_finetune": soft_model,
        "local_counterfactual_task_finetune": local_model,
        "local_counterfactual_scratch": scratch_model,
    }
    task_optimizers = {name: torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0) for name, model in task_models.items()}
    task_curves, task_gradients = _train_models_for_epochs(
        task_models,
        task_optimizers,
        train_x,
        train_y,
        epochs=args.finetune_epochs,
        batch_size=args.batch_size,
        workers=args.workers,
        device=device,
        loader_seed_base=200_000 + seed * 100,
        phase="task",
    )
    scratch_curve = pretrain_curves["local_counterfactual_scratch"] + [
        {**row, "epoch": row["epoch"] + args.pretrain_epochs} for row in task_curves["local_counterfactual_scratch"]
    ]
    task_curves["local_counterfactual_scratch"] = scratch_curve

    evaluations = [
        _plain_evaluation("dense_pretrained", dense_pretrained, held_loader, device),
        _plain_evaluation("dense_continued", dense_continued, held_loader, device),
        _evaluate_maddness("compiled_frozen", frozen_model, held_loader, device, compiled),
        _evaluate_maddness("soft_pq_task_finetune", soft_model, held_loader, device, compiled),
        _evaluate_maddness("local_counterfactual_task_finetune", local_model, held_loader, device, compiled),
        _evaluate_maddness("local_counterfactual_scratch", scratch_model, held_loader, device, scratch_initial),
    ]
    first_gradients = {**pretrain_gradients, **task_gradients}
    result = SeedResult(
        seed=seed,
        compiler_logit_nmse=compiler_nmse,
        compiler_seconds=compiler_seconds,
        dense_pretrain_curve=pretrain_curves["dense_pretrained"],
        task_training_curves=task_curves,
        first_step_gradient_norms=first_gradients,
        evaluations=evaluations,
        seconds=time.perf_counter() - started,
    )
    state: dict[str, Tensor] = {
        **{f"compiled.{key}": value for key, value in asdict(compiled).items()},
        **{f"scratch_initial.{key}": value for key, value in asdict(scratch_initial).items()},
        **{f"dense_pretrained.{key}": value.detach().cpu() for key, value in dense_pretrained.state_dict().items()},
        **{f"dense_continued.{key}": value.detach().cpu() for key, value in dense_continued.state_dict().items()},
        **{f"soft.{key}": value.detach().cpu() for key, value in soft_model.state_dict().items()},
        **{f"local.{key}": value.detach().cpu() for key, value in local_model.state_dict().items()},
        **{f"scratch.{key}": value.detach().cpu() for key, value in scratch_model.state_dict().items()},
    }
    return result, state


def summarize(rows: list[SeedResult]) -> dict[str, object]:
    by_arm: dict[str, dict[str, float]] = {}
    for arm in ARMS:
        selected = [next(evaluation for evaluation in row.evaluations if evaluation.arm == arm) for row in rows]
        by_arm[arm] = {
            "held_ce_mean": sum(item.held_ce for item in selected) / len(selected),
            "held_accuracy_mean": sum(item.held_accuracy for item in selected) / len(selected),
        }
    frozen_ce = by_arm["compiled_frozen"]["held_ce_mean"]
    soft_ce = by_arm["soft_pq_task_finetune"]["held_ce_mean"]
    local_ce = by_arm["local_counterfactual_task_finetune"]["held_ce_mean"]
    scratch_ce = by_arm["local_counterfactual_scratch"]["held_ce_mean"]
    return {
        "arms": by_arm,
        "effects": {
            "soft_pq_ce_improvement_vs_frozen": frozen_ce - soft_ce,
            "local_ste_ce_improvement_vs_frozen": frozen_ce - local_ce,
            "local_ste_ce_improvement_vs_soft_pq": soft_ce - local_ce,
            "scratch_gap_to_compiled_local": scratch_ce - local_ce,
        },
        "decisions": {
            "local_ste_improves_frozen_all_seeds": all(
                next(item for item in row.evaluations if item.arm == "local_counterfactual_task_finetune").held_ce
                < next(item for item in row.evaluations if item.arm == "compiled_frozen").held_ce
                for row in rows
            ),
            "local_ste_beats_soft_pq_all_seeds": all(
                next(item for item in row.evaluations if item.arm == "local_counterfactual_task_finetune").held_ce
                < next(item for item in row.evaluations if item.arm == "soft_pq_task_finetune").held_ce
                for row in rows
            ),
            "scratch_beats_frozen_all_seeds": all(
                next(item for item in row.evaluations if item.arm == "local_counterfactual_scratch").held_ce
                < next(item for item in row.evaluations if item.arm == "compiled_frozen").held_ce
                for row in rows
            ),
            "all_maddness_hard_forwards_exact": all(
                item.hard_forward_exact is True for row in rows for item in row.evaluations if item.arm not in {"dense_pretrained", "dense_continued"}
            ),
            "all_local_stem_gradients_nonzero": all(
                row.first_step_gradient_norms[name]["stem"] > 0
                for row in rows
                for name in ("local_counterfactual_task_finetune", "local_counterfactual_scratch")
            ),
        },
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    root = Path(args.root).expanduser()
    train_x, train_y = _load_emnist_split(root, "balanced", train=True, limit=args.max_train, seed=0)
    held_x, held_y = _load_emnist_split(root, "balanced", train=False, limit=args.max_test, seed=0)
    classes = int(max(int(train_y.max()), int(held_y.max())) + 1)
    held_loader = _ordered_loader(
        held_x,
        held_y,
        args.batch_size,
        args.workers,
        pin_memory=torch.device(args.device).type == "cuda",
    )
    rows: list[SeedResult] = []
    artifact_state: dict[str, Tensor] = {}
    for seed in args.seeds:
        row, state = fit_seed(seed, args, train_x, train_y, held_loader, classes)
        rows.append(row)
        artifact_state.update({f"seed{seed}.{key}": value for key, value in state.items()})
        print(
            f"seed={seed} " + " ".join(f"{item.arm}:ce={item.held_ce:.6f},acc={item.held_accuracy:.6f}" for item in row.evaluations),
            flush=True,
        )
    protocol = {
        "dataset": "EMNIST Balanced",
        "train_examples": len(train_x),
        "held_examples": len(held_x),
        "input_dim": train_x.shape[1],
        "hidden_dim": args.hidden_dim,
        "classes": classes,
        "tables": args.tables,
        "leaves_per_table": 16,
        "tree_depth": 4,
        "hard_active_comparisons": 4 * args.tables,
        "hard_active_lookups": args.tables,
        "hard_active_output_additions": args.tables * classes,
        "hard_maddness_layer_multiplications": 0,
        "compiler_samples": args.compiler_samples,
        "pretrain_epochs": args.pretrain_epochs,
        "finetune_epochs": args.finetune_epochs,
        "scratch_total_epochs": args.pretrain_epochs + args.finetune_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "ridge": args.ridge,
        "tau": args.tau,
        "soft_initial_temperature": args.soft_temperature,
        "scratch_prototype_std": args.scratch_prototype_std,
        "seeds": list(args.seeds),
        "split_coordinates_frozen_within_each_arm": True,
        "compiled_arms_use_dense_teacher_logits": True,
        "scratch_route_uses_unlabeled_initial_features_only": True,
        "scratch_actions_use_no_offline_teacher_or_labels": True,
        "local_surrogate": "nearest_path_wall_sibling_subtree_action_difference",
        "local_surrogate_inspiration": "ref/spikes.tex equations gi/dLdx and nearest flipped bucket",
        "held_not_used_for_selection_or_early_stopping": True,
        "device": args.device,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    result = {
        "schema": "emnist-maddness-task-ste-v1",
        "protocol": protocol,
        "seeds": [asdict(row) for row in rows],
        "summary": summarize(rows),
    }
    artifact = Path(args.artifact)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    if artifact.exists():
        raise FileExistsError(artifact)
    torch.save({"schema": result["schema"], "protocol": protocol, "state": artifact_state}, artifact)
    return result


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item) for item in value.split(",") if item)
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--tables", type=int, default=32)
    parser.add_argument("--compiler-samples", type=int, default=32768)
    parser.add_argument("--pretrain-epochs", type=int, default=10)
    parser.add_argument("--finetune-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--soft-temperature", type=float, default=0.03)
    parser.add_argument("--scratch-prototype-std", type=float, default=0.02)
    parser.add_argument("--seeds", type=_parse_seeds, default=(0, 1, 2))
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-test", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.hidden_dim != 64 or args.tables != 32:
        parser.error("formal layer requires D64/T32")
    if args.compiler_samples < 32 or args.compiler_samples % 32:
        parser.error("compiler samples must be a positive multiple of 32")
    if args.output == args.artifact or args.output.exists() or args.artifact.exists():
        parser.error("output and artifact must be distinct nonexistent paths")
    return args


def main() -> None:
    args = parse_args()
    result = run(args)
    _atomic_json(args.output, result)
    print(json.dumps(result["summary"], indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
