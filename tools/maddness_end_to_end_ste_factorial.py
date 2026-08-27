from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from tropnn.layers.maddness import CompiledMaddness, FrozenMaddness, LocalCounterfactualMaddness, SoftPQMaddness
from tropnn.tools.random_linear_address_action_factorial import orthogonal_teacher

ARMS = ("original_compiler", "torch_frozen", "soft_pq", "local_counterfactual_ste")


@dataclass(frozen=True)
class ArmResult:
    seed: int
    arm: str
    held_input_nmse: float
    held_input_r2: float
    held_output_nmse: float
    held_output_r2: float
    input_output_nmse_gap: float
    mean_entropy_bits: float
    minimum_entropy_bits: float
    maximum_row_mass: float
    threshold_rms_drift: float
    held_code_flip_fraction: float
    hard_soft_output_nmse: float | None
    training_curve: list[dict[str, float]]
    seconds: float


def _leaf_stats(codes: Tensor) -> tuple[float, float, float]:
    entropies: list[float] = []
    maximum = 0.0
    for table in range(codes.shape[1]):
        counts = torch.bincount(codes[:, table].cpu(), minlength=16).double()
        probabilities = counts / counts.sum()
        positive = probabilities > 0
        entropies.append(float(-(probabilities[positive] * probabilities[positive].log2()).sum()))
        maximum = max(maximum, float(probabilities.max()))
    return sum(entropies) / len(entropies), min(entropies), maximum


def _fit_tree(subspace: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    split_indices: list[int] = []
    thresholds = np.zeros(15, dtype=np.float32)
    buckets = [np.arange(subspace.shape[0], dtype=np.int64)]
    for level in range(4):
        losses = []
        for coordinate in range(subspace.shape[1]):
            loss = 0.0
            for bucket in buckets:
                values = subspace[bucket]
                centered = values - values.mean(axis=0, keepdims=True)
                loss += float(np.square(centered[:, coordinate]).sum())
            losses.append(loss)
        coordinate = int(np.argmax(losses))
        split_indices.append(coordinate)
        next_buckets: list[np.ndarray] = []
        for node_in_level, bucket in enumerate(buckets):
            ordered = bucket[np.argsort(subspace[bucket, coordinate], kind="stable")]
            middle = ordered.size // 2
            left, right = ordered[:middle], ordered[middle:]
            threshold = (subspace[left[-1], coordinate] + subspace[right[0], coordinate]) / 2
            thresholds[2**level - 1 + node_in_level] = threshold
            next_buckets.extend((left, right))
        buckets = next_buckets
    codes = np.empty(subspace.shape[0], dtype=np.int64)
    centroids = []
    for leaf, bucket in enumerate(buckets):
        codes[bucket] = leaf
        centroids.append(subspace[bucket].mean(axis=0))
    return np.asarray(split_indices, dtype=np.int64), thresholds, codes, np.stack(centroids)


def compile_original_maddness(x: Tensor, tables: int, ridge: float = 1.0) -> CompiledMaddness:
    if x.ndim != 2 or x.shape[1] % tables:
        raise ValueError("input dimension must be divisible by table count")
    x_np = x.detach().cpu().numpy().astype(np.float32, copy=False)
    width = x.shape[1] // tables
    all_indices, all_thresholds, all_centroids, all_codes = [], [], [], []
    for table in range(tables):
        start = table * width
        indices, thresholds, codes, centroids = _fit_tree(x_np[:, start : start + width])
        all_indices.append(indices + start)
        all_thresholds.append(thresholds)
        all_centroids.append(centroids)
        all_codes.append(codes)
    codes_np = np.stack(all_codes, axis=1)
    gram = np.zeros((16 * tables, 16 * tables), dtype=np.float64)
    for left_table in range(tables):
        left = slice(16 * left_table, 16 * (left_table + 1))
        for right_table in range(left_table, tables):
            right = slice(16 * right_table, 16 * (right_table + 1))
            joint = 16 * codes_np[:, left_table] + codes_np[:, right_table]
            block = np.bincount(joint, minlength=256).reshape(16, 16)
            gram[left, right] = block
            gram[right, left] = block.T
    rhs = np.zeros((16 * tables, x.shape[1]), dtype=np.float64)
    x64 = x_np.astype(np.float64)
    for table in range(tables):
        np.add.at(rhs[16 * table : 16 * (table + 1)], codes_np[:, table], x64)
    prototypes = np.linalg.solve(gram + ridge * np.eye(16 * tables), rhs).reshape(tables, 16, x.shape[1])
    return CompiledMaddness(
        split_indices=torch.from_numpy(np.stack(all_indices)),
        thresholds=torch.from_numpy(np.stack(all_thresholds)),
        encoder_centroids=torch.from_numpy(np.stack(all_centroids)),
        prototypes=torch.from_numpy(prototypes.astype(np.float32)),
    )


def _nmse(prediction: Tensor, target: Tensor) -> float:
    return float((prediction - target).square().mean() / target.square().mean().clamp_min(1e-30))


def _evaluate(
    seed: int,
    arm: str,
    model: nn.Module,
    initial: CompiledMaddness,
    x: Tensor,
    teacher: Tensor,
    curve: list[dict[str, float]],
    seconds: float,
) -> ArmResult:
    with torch.no_grad():
        reconstruction = model(x)
        prediction = reconstruction @ teacher.T
        target = x @ teacher.T
        input_nmse = _nmse(reconstruction, x)
        output_nmse = _nmse(prediction, target)
        thresholds = model.thresholds  # type: ignore[attr-defined]
        codes = model.hard_codes(x)  # type: ignore[attr-defined]
        initial_codes = FrozenMaddness(initial).to(x.device).hard_codes(x)
        entropy, minimum_entropy, maximum_mass = _leaf_stats(codes)
        hard_soft = None
        if isinstance(model, SoftPQMaddness):
            hard, soft = model.outputs(x)
            hard_soft = _nmse(soft, hard)
        drift = float((thresholds - initial.thresholds.to(x.device)).square().mean().sqrt())
    return ArmResult(
        seed=seed,
        arm=arm,
        held_input_nmse=input_nmse,
        held_input_r2=1 - input_nmse,
        held_output_nmse=output_nmse,
        held_output_r2=1 - output_nmse,
        input_output_nmse_gap=output_nmse - input_nmse,
        mean_entropy_bits=entropy,
        minimum_entropy_bits=minimum_entropy,
        maximum_row_mass=maximum_mass,
        threshold_rms_drift=drift,
        held_code_flip_fraction=float((codes != initial_codes).float().mean()),
        hard_soft_output_nmse=hard_soft,
        training_curve=curve,
        seconds=seconds,
    )


def fit_seed(seed: int, args: argparse.Namespace) -> tuple[list[ArmResult], dict[str, Tensor]]:
    device = torch.device(args.device)
    teacher = orthogonal_teacher(args.dim, seed, device)
    compiler_generator = torch.Generator(device="cpu").manual_seed(40_000 + seed)
    compiler_x = torch.randn(args.compiler_samples, args.dim, generator=compiler_generator)
    started = time.perf_counter()
    compiled = compile_original_maddness(compiler_x, args.tables, args.ridge)
    compiler_seconds = time.perf_counter() - started
    frozen = FrozenMaddness(compiled).to(device)
    soft = SoftPQMaddness(compiled, args.soft_temperature).to(device)
    local = LocalCounterfactualMaddness(compiled, args.tau).to(device)
    optimizers = {
        "soft_pq": torch.optim.AdamW(soft.parameters(), lr=args.lr, weight_decay=0),
        "local_counterfactual_ste": torch.optim.AdamW(local.parameters(), lr=args.lr, weight_decay=0),
    }
    models = {"soft_pq": soft, "local_counterfactual_ste": local}
    curves: dict[str, list[dict[str, float]]] = {arm: [] for arm in models}
    train_generator = torch.Generator(device=device).manual_seed(50_000 + seed)
    train_started = time.perf_counter()
    for step in range(1, args.steps + 1):
        x = torch.randn(args.batch_size, args.dim, generator=train_generator, device=device)
        target = x @ teacher.T
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        losses = {arm: F.mse_loss(model(x) @ teacher.T, target) for arm, model in models.items()}
        sum(losses.values()).backward()
        for optimizer in optimizers.values():
            optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            for arm, loss in losses.items():
                curves[arm].append({"step": float(step), "train_mse": float(loss.detach())})
    train_seconds = time.perf_counter() - train_started
    held_generator = torch.Generator(device=device).manual_seed(70_000 + seed)
    held_x = torch.randn(args.held_samples, args.dim, generator=held_generator, device=device)
    original = FrozenMaddness(compiled).to(device)
    rows = [
        _evaluate(seed, "original_compiler", original, compiled, held_x, teacher, [], compiler_seconds),
        _evaluate(seed, "torch_frozen", frozen, compiled, held_x, teacher, [], 0.0),
        _evaluate(seed, "soft_pq", soft.eval(), compiled, held_x, teacher, curves["soft_pq"], train_seconds),
        _evaluate(
            seed,
            "local_counterfactual_ste",
            local.eval(),
            compiled,
            held_x,
            teacher,
            curves["local_counterfactual_ste"],
            train_seconds,
        ),
    ]
    if abs(rows[0].held_output_nmse - rows[1].held_output_nmse) > 1e-12:
        raise AssertionError("original compiler and frozen import disagree")
    state = {
        "teacher": teacher.cpu(),
        "split_indices": compiled.split_indices,
        "initial_thresholds": compiled.thresholds,
        "initial_encoder_centroids": compiled.encoder_centroids,
        "initial_prototypes": compiled.prototypes,
        **{f"soft_pq.{key}": value.detach().cpu() for key, value in soft.state_dict().items()},
        **{f"local_counterfactual_ste.{key}": value.detach().cpu() for key, value in local.state_dict().items()},
    }
    return rows, state


def summarize(rows: list[ArmResult]) -> dict[str, object]:
    summary: dict[str, object] = {}
    for arm in ARMS:
        selected = [row for row in rows if row.arm == arm]
        summary[arm] = {
            "held_output_r2_mean": sum(row.held_output_r2 for row in selected) / len(selected),
            "held_output_nmse_mean": sum(row.held_output_nmse for row in selected) / len(selected),
            "held_input_r2_mean": sum(row.held_input_r2 for row in selected) / len(selected),
        }
    base = summary["torch_frozen"]["held_output_r2_mean"]  # type: ignore[index]
    soft = summary["soft_pq"]["held_output_r2_mean"]  # type: ignore[index]
    local = summary["local_counterfactual_ste"]["held_output_r2_mean"]  # type: ignore[index]
    return {
        "arms": summary,
        "effects": {"soft_pq_vs_frozen_r2": soft - base, "local_ste_vs_frozen_r2": local - base, "local_ste_vs_soft_pq_r2": local - soft},
        "decisions": {
            "soft_pq_improves_all_seeds": all(
                next(row for row in rows if row.seed == seed and row.arm == "soft_pq").held_output_r2
                > next(row for row in rows if row.seed == seed and row.arm == "torch_frozen").held_output_r2
                for seed in sorted({row.seed for row in rows})
            ),
            "local_ste_improves_all_seeds": all(
                next(row for row in rows if row.seed == seed and row.arm == "local_counterfactual_ste").held_output_r2
                > next(row for row in rows if row.seed == seed and row.arm == "torch_frozen").held_output_r2
                for seed in sorted({row.seed for row in rows})
            ),
            "local_ste_beats_soft_pq_all_seeds": all(
                next(row for row in rows if row.seed == seed and row.arm == "local_counterfactual_ste").held_output_r2
                > next(row for row in rows if row.seed == seed and row.arm == "soft_pq").held_output_r2
                for seed in sorted({row.seed for row in rows})
            ),
        },
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    rows: list[ArmResult] = []
    state: dict[str, Tensor] = {}
    for seed in args.seeds:
        seed_rows, seed_state = fit_seed(seed, args)
        rows.extend(seed_rows)
        state.update({f"seed{seed}.{key}": value for key, value in seed_state.items()})
        print(f"seed={seed} " + " ".join(f"{row.arm}:R2={row.held_output_r2:.6f}" for row in seed_rows), flush=True)
    result = {
        "schema": "maddness-end-to-end-ste-factorial-v1",
        "protocol": {
            "dim": args.dim,
            "tables": args.tables,
            "leaves_per_table": 16,
            "tree_depth": 4,
            "active_comparisons": 4 * args.tables,
            "active_lookups": args.tables,
            "compiler_samples": args.compiler_samples,
            "held_samples": args.held_samples,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "tau": args.tau,
            "soft_initial_temperature": args.soft_temperature,
            "ridge": args.ridge,
            "seeds": list(args.seeds),
            "teacher": "QR_Haar_orthogonal_linear_map",
            "hard_inference_identical": True,
            "split_coordinates_frozen": True,
            "soft_pq_thresholds_frozen": True,
            "local_ste_nearest_path_wall_only": True,
        },
        "rows": [asdict(row) for row in rows],
        "summary": summarize(rows),
    }
    artifact = Path(args.artifact)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    if artifact.exists():
        raise FileExistsError(artifact)
    torch.save({"schema": result["schema"], "protocol": result["protocol"], "state": state}, artifact)
    return result


def _parse_seeds(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item)
    if not result:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MADDNESS offline/soft-PQ/local-STE D64 factorial")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--tables", type=int, default=32)
    parser.add_argument("--compiler-samples", type=int, default=32768)
    parser.add_argument("--held-samples", type=int, default=32768)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--soft-temperature", type=float, default=0.03)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--seeds", type=_parse_seeds, default=(0, 1, 2))
    parser.add_argument("--log-every", type=int, default=300)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.dim != 64 or args.tables != 32:
        parser.error("formal protocol requires D64/T32")
    if Path(args.output).exists() or Path(args.artifact).exists():
        parser.error("output and artifact must not exist")
    return args


def main() -> None:
    args = parse_args()
    result = run(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
