"""Frozen-stem EMNIST comparison of ordinary PQ and a comparison grid."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from tropnn.layers.hard_lookup import ProductGridLookupRouter, sum_lookup_rows
from tropnn.tools.emnist_payload_dtype_sweep import _load_emnist_split
from tropnn.tools.product_atlas_pc_action_factorial import _route_health, fit_additive_rows
from tropnn.tools.product_grid_pc_action_factorial import compile_balanced_product_grid

ARMS = ("pq_tied_frozen", "pq_free_action", "grid_tied_frozen", "grid_free_action")


@dataclass(frozen=True)
class Evaluation:
    seed: int
    arm: str
    held_ce: float
    held_accuracy: float
    mean_entropy_bits: float
    minimum_entropy_bits: float
    mean_observed_rows: float
    maximum_row_mass: float
    hard_replay_max_error: float


class NearestCentroidProductHead(nn.Module):
    """Exact D2 product quantizer followed by additive global action rows."""

    def __init__(self, centroids: Tensor, rows: Tensor, *, trainable_rows: bool) -> None:
        super().__init__()
        if centroids.ndim != 3 or rows.ndim != 3:
            raise ValueError("centroids and rows must be [tables,codes,width]")
        if centroids.shape[:2] != rows.shape[:2]:
            raise ValueError("centroid and row codebooks must align")
        self.register_buffer("centroids", centroids.detach().clone())
        if trainable_rows:
            self.rows = nn.Parameter(rows.detach().clone())
        else:
            self.register_buffer("rows", rows.detach().clone())

    @property
    def tables(self) -> int:
        return int(self.centroids.shape[0])

    @property
    def width(self) -> int:
        return int(self.centroids.shape[-1])

    @property
    def codes_per_table(self) -> int:
        return int(self.centroids.shape[1])

    def hard_codes(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.tables * self.width:
            raise ValueError("input width does not match product codebook")
        local = x.reshape(*x.shape[:-1], self.tables, self.width)
        centroids = self.centroids.to(device=x.device, dtype=x.dtype)
        distances = (local.unsqueeze(-2) - centroids).square().sum(dim=-1)
        return distances.argmin(dim=-1)

    def hard_output(self, x: Tensor) -> tuple[Tensor, Tensor]:
        codes = self.hard_codes(x)
        return sum_lookup_rows(self.rows, codes), codes

    def forward(self, x: Tensor) -> Tensor:
        return self.hard_output(x)[0]


class AdditiveProductGridHead(ProductGridLookupRouter):
    """Product grid whose reference reduction uses the shared bounded gather."""

    def hard_output(self, x: Tensor) -> tuple[Tensor, Tensor]:
        codes = self.hard_codes(x)
        return sum_lookup_rows(self.rows, codes), codes


def fit_lloyd_product_codebook(
    x: Tensor,
    initial_centroids: Tensor,
    *,
    iterations: int = 25,
) -> tuple[Tensor, Tensor, list[int]]:
    """Deterministic CPU-float64 Lloyd refinement for disjoint subspaces."""

    if x.ndim != 2 or initial_centroids.ndim != 3:
        raise ValueError("x and initial_centroids must be matrices/codebooks")
    tables, codes, width = initial_centroids.shape
    if x.shape[1] != tables * width or iterations < 1:
        raise ValueError("input/codebook geometry or iteration count is invalid")
    local = x.detach().cpu().double().reshape(x.shape[0], tables, width)
    centroids = initial_centroids.detach().cpu().double().clone()
    completed: list[int] = []
    all_codes: list[Tensor] = []
    for table in range(tables):
        values = local[:, table]
        current = centroids[table]
        previous: Tensor | None = None
        used = 0
        for step in range(iterations):
            distances = (values[:, None, :] - current[None, :, :]).square().sum(dim=-1)
            assignment = distances.argmin(dim=-1)
            used = step + 1
            if previous is not None and torch.equal(assignment, previous):
                break
            previous = assignment
            counts = torch.bincount(assignment, minlength=codes)
            sums = torch.zeros(codes, width, dtype=torch.float64)
            sums.index_add_(0, assignment, values)
            nonempty = counts > 0
            current[nonempty] = sums[nonempty] / counts[nonempty, None]
            if not bool(nonempty.all()):
                residual = distances.gather(1, assignment[:, None]).squeeze(1)
                order = residual.argsort(descending=True, stable=True)
                cursor = 0
                for empty in (~nonempty).nonzero(as_tuple=False).flatten():
                    current[empty] = values[order[cursor]]
                    cursor += 1
        final_distances = (values[:, None, :] - current[None, :, :]).square().sum(dim=-1)
        final_codes = final_distances.argmin(dim=-1)
        if torch.unique(final_codes).numel() != codes:
            raise RuntimeError(f"PQ table {table} has an empty final centroid")
        centroids[table] = current
        all_codes.append(final_codes)
        completed.append(used)
    return centroids.float(), torch.stack(all_codes, dim=-1), completed


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


@torch.no_grad()
def _capture_features(
    x: Tensor,
    stem: nn.Linear,
    head: nn.Linear,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    features: list[Tensor] = []
    logits: list[Tensor] = []
    for start in range(0, x.shape[0], batch_size):
        batch = x[start : start + batch_size].to(device)
        feature = F.gelu(stem(batch.flatten(1)))
        features.append(feature.cpu())
        logits.append(head(feature).cpu())
    return torch.cat(features), torch.cat(logits)


def _train_action_rows(
    pq_rows: Tensor,
    grid_rows: Tensor,
    pq_codes: Tensor,
    grid_codes: Tensor,
    labels: Tensor,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, dict[str, list[dict[str, float]]]]:
    pq_parameter = nn.Parameter(pq_rows.to(device).clone())
    grid_parameter = nn.Parameter(grid_rows.to(device).clone())
    optimizer = torch.optim.AdamW((pq_parameter, grid_parameter), lr=lr, weight_decay=0)
    pq_codes_device = pq_codes.to(device)
    grid_codes_device = grid_codes.to(device)
    labels_device = labels.to(device)
    curves: dict[str, list[dict[str, float]]] = {"pq_free_action": [], "grid_free_action": []}
    generator = torch.Generator(device=device).manual_seed(300_000 + seed)
    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(labels.numel(), generator=generator, device=device)
        loss_sums = {"pq_free_action": 0.0, "grid_free_action": 0.0}
        correct = {"pq_free_action": 0, "grid_free_action": 0}
        for start in range(0, labels.numel(), batch_size):
            indices = permutation[start : start + batch_size]
            target = labels_device[indices]
            optimizer.zero_grad(set_to_none=True)
            pq_logits = sum_lookup_rows(pq_parameter, pq_codes_device[indices])
            grid_logits = sum_lookup_rows(grid_parameter, grid_codes_device[indices])
            pq_loss = F.cross_entropy(pq_logits, target)
            grid_loss = F.cross_entropy(grid_logits, target)
            (pq_loss + grid_loss).backward()
            optimizer.step()
            count = target.numel()
            loss_sums["pq_free_action"] += float(pq_loss.detach()) * count
            loss_sums["grid_free_action"] += float(grid_loss.detach()) * count
            correct["pq_free_action"] += int((pq_logits.detach().argmax(-1) == target).sum())
            correct["grid_free_action"] += int((grid_logits.detach().argmax(-1) == target).sum())
        for arm in curves:
            curves[arm].append(
                {
                    "epoch": float(epoch),
                    "train_ce": loss_sums[arm] / labels.numel(),
                    "train_accuracy": correct[arm] / labels.numel(),
                }
            )
        print(
            f"seed={seed} epoch={epoch}/{epochs} "
            f"pq:ce={curves['pq_free_action'][-1]['train_ce']:.6f} "
            f"grid:ce={curves['grid_free_action'][-1]['train_ce']:.6f}",
            flush=True,
        )
    return pq_parameter.detach().cpu(), grid_parameter.detach().cpu(), curves


def _evaluate(
    seed: int,
    arm: str,
    rows: Tensor,
    codes: Tensor,
    labels: Tensor,
    *,
    hard_output: Tensor,
) -> Evaluation:
    prediction = sum_lookup_rows(rows, codes)
    entropy, minimum, observed, maximum = _route_health(codes, 16)
    return Evaluation(
        seed=seed,
        arm=arm,
        held_ce=float(F.cross_entropy(prediction, labels)),
        held_accuracy=float((prediction.argmax(-1) == labels).float().mean()),
        mean_entropy_bits=entropy,
        minimum_entropy_bits=minimum,
        mean_observed_rows=observed,
        maximum_row_mass=maximum,
        hard_replay_max_error=float((prediction - hard_output).abs().max()),
    )


def _source_linear(state: dict[str, Tensor], seed: int, prefix: str, in_dim: int, out_dim: int) -> nn.Linear:
    layer = nn.Linear(in_dim, out_dim)
    layer.load_state_dict(
        {
            "weight": state[f"seed{seed}.dense_pretrained.{prefix}.weight"],
            "bias": state[f"seed{seed}.dense_pretrained.{prefix}.bias"],
        }
    )
    layer.requires_grad_(False)
    return layer


def fit_seed(
    seed: int,
    args: argparse.Namespace,
    source_state: dict[str, Tensor],
    train_x: Tensor,
    train_y: Tensor,
    held_x: Tensor,
    held_y: Tensor,
) -> tuple[list[Evaluation], dict[str, object], dict[str, Tensor]]:
    device = torch.device(args.device)
    started = time.perf_counter()
    classes = int(max(int(train_y.max()), int(held_y.max())) + 1)
    stem = _source_linear(source_state, seed, "stem", train_x[0].numel(), args.hidden_dim).to(device)
    dense_head = _source_linear(source_state, seed, "head", args.hidden_dim, classes).to(device)
    train_features, train_dense_logits = _capture_features(train_x, stem, dense_head, batch_size=args.batch_size, device=device)
    held_features, held_dense_logits = _capture_features(held_x, stem, dense_head, batch_size=args.batch_size, device=device)
    compiler_features = train_features[: args.compiler_samples]
    compiler_logits = train_dense_logits[: args.compiler_samples]

    initial_centroids = source_state[f"seed{seed}.compiled.encoder_centroids"]
    pq_centroids, pq_compiler_codes, lloyd_iterations = fit_lloyd_product_codebook(
        compiler_features,
        initial_centroids,
        iterations=args.lloyd_iterations,
    )
    grid_supports, grid_thresholds, grid_compiler_codes = compile_balanced_product_grid(compiler_features, args.tables)
    pq_tied_rows = fit_additive_rows(pq_compiler_codes, compiler_logits, 16, args.ridge)
    grid_tied_rows = fit_additive_rows(grid_compiler_codes, compiler_logits, 16, args.ridge)

    pq_tied = NearestCentroidProductHead(pq_centroids, pq_tied_rows, trainable_rows=False).to(device)
    grid_tied = AdditiveProductGridHead(
        args.hidden_dim,
        classes,
        supports=grid_supports,
        thresholds=grid_thresholds,
        rows=grid_tied_rows,
        bins=4,
        surrogate="none",
        trainable_thresholds=False,
        trainable_rows=False,
    ).to(device)
    with torch.no_grad():
        pq_train_codes = pq_tied.hard_codes(train_features.to(device)).cpu()
        grid_train_codes = grid_tied.hard_codes(train_features.to(device)).cpu()
    pq_free_rows, grid_free_rows, curves = _train_action_rows(
        pq_tied_rows,
        grid_tied_rows,
        pq_train_codes,
        grid_train_codes,
        train_y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=seed,
        device=device,
    )
    pq_free = NearestCentroidProductHead(pq_centroids, pq_free_rows, trainable_rows=False).to(device)
    grid_free = AdditiveProductGridHead(
        args.hidden_dim,
        classes,
        supports=grid_supports,
        thresholds=grid_thresholds,
        rows=grid_free_rows,
        bins=4,
        surrogate="none",
        trainable_thresholds=False,
        trainable_rows=False,
    ).to(device)
    held_features_device = held_features.to(device)
    held_y_device = held_y.to(device)
    rows: list[Evaluation] = []
    with torch.no_grad():
        pq_codes = pq_tied.hard_codes(held_features_device)
        grid_codes = grid_tied.hard_codes(held_features_device)
        models = {
            "pq_tied_frozen": (pq_tied_rows.to(device), pq_tied(held_features_device)),
            "pq_free_action": (pq_free_rows.to(device), pq_free(held_features_device)),
            "grid_tied_frozen": (grid_tied_rows.to(device), grid_tied(held_features_device)),
            "grid_free_action": (grid_free_rows.to(device), grid_free(held_features_device)),
        }
        for arm in ARMS:
            action_rows, hard_output = models[arm]
            codes = pq_codes if arm.startswith("pq_") else grid_codes
            rows.append(_evaluate(seed, arm, action_rows, codes, held_y_device, hard_output=hard_output))
        dense_ce = float(F.cross_entropy(held_dense_logits.to(device), held_y_device))
        dense_accuracy = float((held_dense_logits.argmax(-1) == held_y).float().mean())

    audits: dict[str, object] = {
        "dense_pretrained_held_ce": dense_ce,
        "dense_pretrained_held_accuracy": dense_accuracy,
        "pq_compiler_codes_replay_exact": torch.equal(
            pq_compiler_codes, NearestCentroidProductHead(pq_centroids, pq_tied_rows, trainable_rows=False).hard_codes(compiler_features)
        ),
        "grid_compiler_codes_replay_exact": torch.equal(grid_compiler_codes, grid_tied.cpu().hard_codes(compiler_features)),
        "pq_lloyd_iterations_by_table": lloyd_iterations,
        "all_hard_replays_exact": all(row.hard_replay_max_error == 0 for row in rows),
        "all_finite": all(
            torch.isfinite(torch.tensor(value))
            for row in rows
            for value in (row.held_ce, row.held_accuracy, row.mean_entropy_bits, row.maximum_row_mass)
        ),
        "seconds": time.perf_counter() - started,
        "training_curves": curves,
    }
    state = {
        "pq.centroids": pq_centroids,
        "pq.tied_rows": pq_tied_rows,
        "pq.free_rows": pq_free_rows,
        "grid.supports": grid_supports,
        "grid.thresholds": grid_thresholds,
        "grid.tied_rows": grid_tied_rows,
        "grid.free_rows": grid_free_rows,
    }
    return rows, audits, state


def summarize(rows: list[Evaluation]) -> dict[str, object]:
    arms: dict[str, dict[str, float]] = {}
    for arm in ARMS:
        selected = [row for row in rows if row.arm == arm]
        arms[arm] = {
            "held_ce_mean": sum(row.held_ce for row in selected) / len(selected),
            "held_accuracy_mean": sum(row.held_accuracy for row in selected) / len(selected),
        }
    by_seed = {(row.seed, row.arm): row for row in rows}
    pq_advantage = [
        by_seed[seed, "grid_free_action"].held_ce - by_seed[seed, "pq_free_action"].held_ce for seed in sorted({row.seed for row in rows})
    ]
    pq_gain = [by_seed[seed, "pq_tied_frozen"].held_ce - by_seed[seed, "pq_free_action"].held_ce for seed in sorted({row.seed for row in rows})]
    grid_gain = [by_seed[seed, "grid_tied_frozen"].held_ce - by_seed[seed, "grid_free_action"].held_ce for seed in sorted({row.seed for row in rows})]
    grid_retains = all(
        by_seed[seed, "grid_free_action"].held_ce <= by_seed[seed, "pq_free_action"].held_ce + 0.02
        and by_seed[seed, "grid_free_action"].held_accuracy >= by_seed[seed, "pq_free_action"].held_accuracy - 0.005
        for seed in sorted({row.seed for row in rows})
    )
    return {
        "arms": arms,
        "effects": {
            "pq_ce_advantage_over_grid_by_seed": pq_advantage,
            "pq_free_action_gain_by_seed": pq_gain,
            "grid_free_action_gain_by_seed": grid_gain,
        },
        "decisions": {
            "grid_retains_pq_quality": grid_retains,
            "pq_geometry_materially_better": all(value > 0 for value in pq_advantage) and sum(pq_advantage) / len(pq_advantage) >= 0.03,
            "pq_free_action_helps": all(value > 0 for value in pq_gain) and sum(pq_gain) / len(pq_gain) >= 0.01,
            "grid_free_action_helps": all(value > 0 for value in grid_gain) and sum(grid_gain) / len(grid_gain) >= 0.01,
        },
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    source = torch.load(args.source_artifact, map_location="cpu", weights_only=False)
    if source.get("schema") != "emnist-maddness-task-ste-v1":
        raise ValueError("unexpected source artifact schema")
    train_x, train_y = _load_emnist_split(args.root, "balanced", train=True, limit=args.max_train, seed=0)
    held_x, held_y = _load_emnist_split(args.root, "balanced", train=False, limit=args.max_test, seed=0)
    rows: list[Evaluation] = []
    audits: dict[str, object] = {}
    artifact_state: dict[str, Tensor] = {}
    for seed in args.seeds:
        seed_rows, seed_audit, seed_state = fit_seed(
            seed,
            args,
            source["state"],
            train_x,
            train_y,
            held_x,
            held_y,
        )
        rows.extend(seed_rows)
        audits[str(seed)] = seed_audit
        artifact_state.update({f"seed{seed}.{key}": value for key, value in seed_state.items()})
        print(
            f"seed={seed} " + " ".join(f"{row.arm}:ce={row.held_ce:.6f},acc={row.held_accuracy:.6f}" for row in seed_rows),
            flush=True,
        )
    if not all(audit["all_hard_replays_exact"] and audit["all_finite"] for audit in audits.values()):  # type: ignore[union-attr]
        raise RuntimeError("mechanical hard replay/finite audit failed")
    source_stat = args.source_artifact.stat()
    protocol = {
        "dataset": "EMNIST Balanced",
        "source_artifact": str(args.source_artifact.resolve()),
        "source_artifact_size": source_stat.st_size,
        "source_artifact_mtime_ns": source_stat.st_mtime_ns,
        "train_examples": len(train_x),
        "held_examples": len(held_x),
        "compiler_samples": args.compiler_samples,
        "hidden_dim": args.hidden_dim,
        "tables": args.tables,
        "codes_per_table": 16,
        "classes": int(max(int(train_y.max()), int(held_y.max())) + 1),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "ridge": args.ridge,
        "lloyd_iterations": args.lloyd_iterations,
        "seeds": list(args.seeds),
        "stem_and_recognizers_frozen_during_action_training": True,
        "only_action_rows_train": True,
        "held_used_for_selection": False,
        "device": args.device,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "pq_recognition_squared_terms": args.tables * 16 * 2,
        "grid_recognition_comparisons": args.tables * 6,
        "active_row_reads": args.tables,
        "active_output_scalar_additions": args.tables * 47,
    }
    result = {
        "schema": "emnist-pq-product-grid-factorial-v1",
        "protocol": protocol,
        "rows": [asdict(row) for row in rows],
        "audits": audits,
        "summary": summarize(rows),
    }
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    if args.artifact.exists():
        raise FileExistsError(args.artifact)
    torch.save({"schema": result["schema"], "protocol": protocol, "state": artifact_state}, args.artifact)
    return result


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item) for item in value.split(",") if item)
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-artifact", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--tables", type=int, default=32)
    parser.add_argument("--compiler-samples", type=int, default=32768)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--lloyd-iterations", type=int, default=25)
    parser.add_argument("--seeds", type=_parse_seeds, default=(0, 1, 2))
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-test", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.hidden_dim != 64 or args.tables != 32:
        parser.error("formal protocol requires D64/T32")
    if args.output == args.artifact or args.output.exists() or args.artifact.exists():
        parser.error("output and artifact must be distinct nonexistent paths")
    if not args.source_artifact.is_file():
        parser.error("source artifact is missing")
    return args


def main() -> None:
    args = parse_args()
    result = run(args)
    _atomic_json(args.output, result)
    print(json.dumps(result["summary"], indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
