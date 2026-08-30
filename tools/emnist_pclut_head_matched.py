"""Matched frozen-stem EMNIST comparison with a classic Flat-Pair PC-LUT head.

The source dense stem is frozen.  Recognition is the classic T32/C4 flat
Pair route and action is the sum of 32 selected D47 rows.  The three Pair arms
separate a frozen compiled head, fixed recognition with CE-trained rows, and
joint threshold/row training with the original PC-LUT hard-forward local
counterfactual surrogate.  Previously frozen PQ/grid heads are replayed on the
same features as read-only controls.
"""

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

from tropnn.layers.hard_lookup import sum_lookup_rows
from tropnn.layers.pairwise import PairwiseLUT
from tropnn.tools.emnist_payload_dtype_sweep import _load_emnist_split
from tropnn.tools.emnist_pq_product_grid_factorial import (
    AdditiveProductGridHead,
    NearestCentroidProductHead,
    _capture_features,
    _source_linear,
)
from tropnn.tools.product_atlas_pc_action_factorial import _route_health, fit_additive_rows

SCHEMA = "emnist-pclut-head-matched-v1"
PAIR_ARMS = ("pair_tied_frozen", "pair_free_action", "pair_joint_route_action")
REFERENCE_ARMS = ("pq_tied_frozen", "pq_free_action", "grid_tied_frozen", "grid_free_action")


@dataclass(frozen=True)
class Evaluation:
    seed: int
    arm: str
    held_ce: float
    held_accuracy: float
    mean_entropy_bits: float | None
    minimum_entropy_bits: float | None
    mean_observed_rows: float | None
    maximum_row_mass: float | None
    hard_replay_max_error: float


class ClassicPairHead(nn.Module):
    """D64 -> D47 classic flat PC-LUT with exact hard forward."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        tables: int,
        comparisons: int,
        seed: int,
        rows: Tensor,
        supports: Tensor | None = None,
        thresholds: Tensor | None = None,
        trainable_rows: bool,
        trainable_thresholds: bool,
    ) -> None:
        super().__init__()
        self.layer = PairwiseLUT(
            input_dim,
            output_dim,
            tables=tables,
            comparisons=comparisons,
            backend="torch",
            seed=seed,
            anchor_policy="random_no_replace",
            anchor_seed=900_000 + seed,
            use_min_margin_ste=True,
            use_output_scaling=False,
            fixed_zero_threshold=not trainable_thresholds,
            surrogate="fast_sigmoid_odd",
            lut_dtype="fp32",
        )
        expected_rows = (tables, 1 << comparisons, output_dim)
        if rows.shape != expected_rows:
            raise ValueError(f"rows must be {expected_rows}")
        with torch.no_grad():
            if supports is not None:
                if supports.shape != self.layer.anchors.shape:
                    raise ValueError("support shape mismatch")
                self.layer.anchors.copy_(supports)
            if thresholds is not None:
                if thresholds.shape != self.layer.thresholds.shape:
                    raise ValueError("threshold shape mismatch")
                self.layer.thresholds.copy_(thresholds)
            self.layer.lut.copy_(rows)
        self.layer.lut.requires_grad_(trainable_rows)
        self.layer.thresholds.requires_grad_(trainable_thresholds)

    @property
    def supports(self) -> Tensor:
        return self.layer.anchors

    @property
    def thresholds(self) -> Tensor:
        return self.layer.thresholds

    @property
    def rows(self) -> Tensor:
        return self.layer.lut

    def hard_codes(self, x: Tensor) -> Tensor:
        return self.layer.cache_index(x.unsqueeze(1)).indices.squeeze(1)

    def hard_output(self, x: Tensor) -> tuple[Tensor, Tensor]:
        codes = self.hard_codes(x)
        return sum_lookup_rows(self.rows, codes), codes

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x.unsqueeze(1)).squeeze(1)


def _atomic_json(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _path_metadata(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _evaluate_pair(seed: int, arm: str, model: ClassicPairHead, features: Tensor, labels: Tensor) -> Evaluation:
    model.eval()
    with torch.no_grad():
        hard_output, codes = model.hard_output(features)
        forward = model(features)
        entropy, minimum, observed, maximum = _route_health(codes, 16)
    return Evaluation(
        seed,
        arm,
        float(F.cross_entropy(hard_output, labels)),
        float((hard_output.argmax(-1) == labels).float().mean()),
        entropy,
        minimum,
        observed,
        maximum,
        float((hard_output - forward).abs().max()),
    )


def _evaluate_reference(
    seed: int,
    arm: str,
    reference_state: dict[str, Tensor],
    features: Tensor,
    labels: Tensor,
) -> Evaluation:
    if arm.startswith("pq_"):
        row_kind = "tied_rows" if arm == "pq_tied_frozen" else "free_rows"
        model: nn.Module = NearestCentroidProductHead(
            reference_state[f"seed{seed}.pq.centroids"],
            reference_state[f"seed{seed}.pq.{row_kind}"],
            trainable_rows=False,
        ).to(features.device)
    else:
        row_kind = "tied_rows" if arm == "grid_tied_frozen" else "free_rows"
        model = AdditiveProductGridHead(
            64,
            47,
            supports=reference_state[f"seed{seed}.grid.supports"],
            thresholds=reference_state[f"seed{seed}.grid.thresholds"],
            rows=reference_state[f"seed{seed}.grid.{row_kind}"],
            bins=4,
            surrogate="none",
            trainable_thresholds=False,
            trainable_rows=False,
        ).to(features.device)
    with torch.no_grad():
        hard_output, codes = model.hard_output(features)  # type: ignore[attr-defined]
        forward = model(features)
        entropy, minimum, observed, maximum = _route_health(codes, 16)
    return Evaluation(
        seed,
        arm,
        float(F.cross_entropy(hard_output, labels)),
        float((hard_output.argmax(-1) == labels).float().mean()),
        entropy,
        minimum,
        observed,
        maximum,
        float((hard_output - forward).abs().max()),
    )


def _train_pair_heads(
    fixed: ClassicPairHead,
    joint: ClassicPairHead,
    train_features: Tensor,
    train_labels: Tensor,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, list[dict[str, float]]], dict[str, float]]:
    fixed.train()
    joint.train()
    optimizer = torch.optim.AdamW((*fixed.parameters(), *joint.parameters()), lr=lr, weight_decay=0)
    features = train_features.to(device)
    labels = train_labels.to(device)
    curves = {"pair_free_action": [], "pair_joint_route_action": []}
    first_gradients: dict[str, float] = {}
    generator = torch.Generator(device=device).manual_seed(300_000 + seed)
    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(labels.numel(), generator=generator, device=device)
        loss_sum = {key: 0.0 for key in curves}
        correct = {key: 0 for key in curves}
        for start in range(0, labels.numel(), batch_size):
            indices = permutation[start : start + batch_size]
            target = labels[indices]
            optimizer.zero_grad(set_to_none=True)
            logits = {
                "pair_free_action": fixed(features[indices]),
                "pair_joint_route_action": joint(features[indices]),
            }
            losses = {key: F.cross_entropy(value, target) for key, value in logits.items()}
            sum(losses.values()).backward()
            if epoch == 1 and start == 0:
                first_gradients = {
                    "fixed_rows": float(fixed.rows.grad.norm()),
                    "joint_rows": float(joint.rows.grad.norm()),
                    "joint_thresholds": float(joint.thresholds.grad.norm()),
                }
            optimizer.step()
            count = target.numel()
            for key in curves:
                loss_sum[key] += float(losses[key].detach()) * count
                correct[key] += int((logits[key].detach().argmax(-1) == target).sum())
        for key in curves:
            curves[key].append(
                {
                    "epoch": float(epoch),
                    "train_ce": loss_sum[key] / labels.numel(),
                    "train_accuracy": correct[key] / labels.numel(),
                }
            )
        print(
            f"seed={seed} epoch={epoch}/{epochs} "
            f"fixed:ce={curves['pair_free_action'][-1]['train_ce']:.6f} "
            f"joint:ce={curves['pair_joint_route_action'][-1]['train_ce']:.6f}",
            flush=True,
        )
    return curves, first_gradients


def fit_seed(
    seed: int,
    args: argparse.Namespace,
    source_state: dict[str, Tensor],
    reference_state: dict[str, Tensor],
    reference_rows: dict[tuple[int, str], dict[str, object]],
    train_x: Tensor,
    train_y: Tensor,
    held_x: Tensor,
    held_y: Tensor,
) -> tuple[list[Evaluation], dict[str, object], dict[str, Tensor]]:
    device = torch.device(args.device)
    started = time.perf_counter()
    stem = _source_linear(source_state, seed, "stem", train_x.shape[1], args.hidden_dim).to(device)
    dense_head = _source_linear(source_state, seed, "head", args.hidden_dim, 47).to(device)
    train_features, train_dense_logits = _capture_features(train_x, stem, dense_head, batch_size=args.batch_size, device=device)
    held_features, held_dense_logits = _capture_features(held_x, stem, dense_head, batch_size=args.batch_size, device=device)
    compiler_features = train_features[: args.compiler_samples]
    compiler_logits = train_dense_logits[: args.compiler_samples]

    zero_rows = torch.zeros(args.tables, 1 << args.comparisons, 47)
    route_template = ClassicPairHead(
        args.hidden_dim,
        47,
        tables=args.tables,
        comparisons=args.comparisons,
        seed=seed,
        rows=zero_rows,
        trainable_rows=False,
        trainable_thresholds=False,
    )
    compiler_codes = route_template.hard_codes(compiler_features)
    tied_rows = fit_additive_rows(compiler_codes, compiler_logits, 1 << args.comparisons, args.ridge)
    common = {
        "input_dim": args.hidden_dim,
        "output_dim": 47,
        "tables": args.tables,
        "comparisons": args.comparisons,
        "seed": seed,
        "supports": route_template.supports,
        "thresholds": torch.zeros(args.tables, args.comparisons),
    }
    tied = ClassicPairHead(**common, rows=tied_rows, trainable_rows=False, trainable_thresholds=False).to(device)
    fixed = ClassicPairHead(**common, rows=tied_rows, trainable_rows=True, trainable_thresholds=False).to(device)
    joint = ClassicPairHead(**common, rows=tied_rows, trainable_rows=True, trainable_thresholds=True).to(device)
    initial_joint_codes = joint.hard_codes(train_features.to(device)).detach().cpu()
    curves, first_gradients = _train_pair_heads(
        fixed,
        joint,
        train_features,
        train_y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=seed,
        device=device,
    )
    held_features_device = held_features.to(device)
    held_y_device = held_y.to(device)
    evaluations = [_evaluate_reference(seed, arm, reference_state, held_features_device, held_y_device) for arm in REFERENCE_ARMS]
    evaluations.extend(
        [
            _evaluate_pair(seed, "pair_tied_frozen", tied, held_features_device, held_y_device),
            _evaluate_pair(seed, "pair_free_action", fixed, held_features_device, held_y_device),
            _evaluate_pair(seed, "pair_joint_route_action", joint, held_features_device, held_y_device),
        ]
    )
    reference_error = {
        arm: {
            "held_ce_abs_error": abs(next(row.held_ce for row in evaluations if row.arm == arm) - float(reference_rows[seed, arm]["held_ce"])),
            "held_accuracy_abs_error": abs(
                next(row.held_accuracy for row in evaluations if row.arm == arm) - float(reference_rows[seed, arm]["held_accuracy"])
            ),
        }
        for arm in REFERENCE_ARMS
    }
    with torch.no_grad():
        final_joint_codes = joint.hard_codes(train_features.to(device)).cpu()
        dense_ce = float(F.cross_entropy(held_dense_logits.to(device), held_y_device))
        dense_accuracy = float((held_dense_logits.argmax(-1) == held_y).float().mean())
    audit = {
        "dense_pretrained_held_ce": dense_ce,
        "dense_pretrained_held_accuracy": dense_accuracy,
        "compiler_codes_replay_exact": torch.equal(compiler_codes, route_template.hard_codes(compiler_features)),
        "reference_replay_errors": reference_error,
        "reference_replay_within_1e_6": all(
            value[metric] <= 1e-6 for value in reference_error.values() for metric in ("held_ce_abs_error", "held_accuracy_abs_error")
        ),
        "all_hard_replays_exact": all(row.hard_replay_max_error == 0 for row in evaluations),
        "all_finite": all(
            torch.isfinite(torch.tensor(value)) for row in evaluations for value in (row.held_ce, row.held_accuracy, row.maximum_row_mass or 0.0)
        ),
        "first_step_gradient_norms": first_gradients,
        "joint_train_code_flip_fraction": float((initial_joint_codes != final_joint_codes).float().mean()),
        "joint_threshold_rms": float(joint.thresholds.detach().square().mean().sqrt()),
        "training_curves": curves,
        "seconds": time.perf_counter() - started,
    }
    state = {
        "pair.supports": tied.supports.detach().cpu(),
        "pair.initial_thresholds": torch.zeros(args.tables, args.comparisons),
        "pair.tied_rows": tied.rows.detach().cpu(),
        "pair.fixed_rows": fixed.rows.detach().cpu(),
        "pair.joint_thresholds": joint.thresholds.detach().cpu(),
        "pair.joint_rows": joint.rows.detach().cpu(),
    }
    return evaluations, audit, state


def summarize(rows: list[Evaluation], dense_by_seed: dict[int, tuple[float, float]]) -> dict[str, object]:
    all_arms = (*REFERENCE_ARMS, *PAIR_ARMS)
    arms = {
        arm: {
            "held_ce_mean": sum(row.held_ce for row in rows if row.arm == arm) / len(dense_by_seed),
            "held_accuracy_mean": sum(row.held_accuracy for row in rows if row.arm == arm) / len(dense_by_seed),
        }
        for arm in all_arms
    }
    arms["dense_pretrained"] = {
        "held_ce_mean": sum(value[0] for value in dense_by_seed.values()) / len(dense_by_seed),
        "held_accuracy_mean": sum(value[1] for value in dense_by_seed.values()) / len(dense_by_seed),
    }
    by_seed = {(row.seed, row.arm): row for row in rows}
    seeds = sorted(dense_by_seed)
    joint_over_fixed = [by_seed[s, "pair_free_action"].held_ce - by_seed[s, "pair_joint_route_action"].held_ce for s in seeds]
    fixed_over_tied = [by_seed[s, "pair_tied_frozen"].held_ce - by_seed[s, "pair_free_action"].held_ce for s in seeds]
    pq_over_joint = [by_seed[s, "pair_joint_route_action"].held_ce - by_seed[s, "pq_free_action"].held_ce for s in seeds]
    grid_over_joint = [by_seed[s, "grid_free_action"].held_ce - by_seed[s, "pair_joint_route_action"].held_ce for s in seeds]
    return {
        "arms": arms,
        "effects": {
            "pair_joint_route_gain_over_fixed_by_seed": joint_over_fixed,
            "pair_free_action_gain_over_tied_by_seed": fixed_over_tied,
            "pq_ce_advantage_over_pair_joint_by_seed": pq_over_joint,
            "pair_joint_ce_advantage_over_grid_by_seed": grid_over_joint,
        },
        "decisions": {
            "route_training_materially_helps_pair": all(value > 0 for value in joint_over_fixed)
            and sum(joint_over_fixed) / len(joint_over_fixed) >= 0.01,
            "free_rows_materially_help_pair": all(value > 0 for value in fixed_over_tied) and sum(fixed_over_tied) / len(fixed_over_tied) >= 0.01,
            "pair_joint_within_0p02_ce_of_pq": all(value <= 0.02 for value in pq_over_joint),
            "pair_joint_materially_beats_grid": all(value > 0 for value in grid_over_joint) and sum(grid_over_joint) / len(grid_over_joint) >= 0.01,
        },
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    source = torch.load(args.source_artifact, map_location="cpu", weights_only=False)
    reference_artifact = torch.load(args.reference_artifact, map_location="cpu", weights_only=False)
    reference_result = json.loads(args.reference_result.read_text())
    if source.get("schema") != "emnist-maddness-task-ste-v1":
        raise ValueError("unexpected source artifact schema")
    if reference_artifact.get("schema") != "emnist-pq-product-grid-factorial-v1":
        raise ValueError("unexpected reference artifact schema")
    if reference_result.get("schema") != reference_artifact["schema"]:
        raise ValueError("reference result/artifact schema mismatch")
    if tuple(reference_result["protocol"]["seeds"]) != tuple(args.seeds):
        raise ValueError("reference seeds do not match requested seeds")
    if Path(reference_result["protocol"]["source_artifact"]).resolve() != args.source_artifact.resolve():
        raise ValueError("reference and current source artifacts differ")
    reference_rows = {(int(row["seed"]), str(row["arm"])): row for row in reference_result["rows"]}
    train_x, train_y = _load_emnist_split(args.root, "balanced", train=True, limit=args.max_train, seed=0)
    held_x, held_y = _load_emnist_split(args.root, "balanced", train=False, limit=args.max_test, seed=0)
    rows: list[Evaluation] = []
    audits: dict[str, object] = {}
    state: dict[str, Tensor] = {}
    for seed in args.seeds:
        seed_rows, seed_audit, seed_state = fit_seed(
            seed,
            args,
            source["state"],
            reference_artifact["state"],
            reference_rows,
            train_x,
            train_y,
            held_x,
            held_y,
        )
        rows.extend(seed_rows)
        audits[str(seed)] = seed_audit
        state.update({f"seed{seed}.{key}": value for key, value in seed_state.items()})
        print(
            f"seed={seed} " + " ".join(f"{row.arm}:ce={row.held_ce:.6f},acc={row.held_accuracy:.6f}" for row in seed_rows),
            flush=True,
        )
    if not all(
        bool(audit["all_hard_replays_exact"]) and bool(audit["all_finite"]) and bool(audit["reference_replay_within_1e_6"])
        for audit in audits.values()  # type: ignore[union-attr]
    ):
        raise RuntimeError("mechanical or reference replay audit failed")
    dense_by_seed = {
        int(seed): (float(audit["dense_pretrained_held_ce"]), float(audit["dense_pretrained_held_accuracy"]))
        for seed, audit in audits.items()  # type: ignore[union-attr]
    }
    protocol = {
        "dataset": "EMNIST Balanced",
        "source_artifact": _path_metadata(args.source_artifact),
        "reference_result": _path_metadata(args.reference_result),
        "reference_artifact": _path_metadata(args.reference_artifact),
        "train_examples": len(train_x),
        "held_examples": len(held_x),
        "compiler_samples": args.compiler_samples,
        "hidden_dim": args.hidden_dim,
        "classes": 47,
        "tables": args.tables,
        "comparisons_per_table": args.comparisons,
        "codes_per_table": 1 << args.comparisons,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "ridge": args.ridge,
        "tau_surrogate": "fast_sigmoid_odd",
        "seeds": list(args.seeds),
        "stem_frozen": True,
        "pair_supports_fixed_random_no_replace": True,
        "pair_initial_thresholds_zero": True,
        "held_used_for_selection": False,
        "device": args.device,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "pair_recognition_comparisons": args.tables * args.comparisons,
        "pair_coordinate_reads": 2 * args.tables * args.comparisons,
        "pair_subtractions": args.tables * args.comparisons,
        "active_row_reads": args.tables,
        "active_output_scalar_additions": args.tables * 47,
        "pair_payload_parameters": args.tables * (1 << args.comparisons) * 47,
        "pair_threshold_parameters": args.tables * args.comparisons,
    }
    result = {
        "schema": SCHEMA,
        "protocol": protocol,
        "rows": [asdict(row) for row in rows],
        "audits": audits,
        "summary": summarize(rows, dense_by_seed),
    }
    if args.artifact.exists():
        raise FileExistsError(args.artifact)
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema": SCHEMA, "protocol": protocol, "state": state}, args.artifact)
    reloaded = torch.load(args.artifact, map_location="cpu", weights_only=False)
    result["artifact_roundtrip"] = {
        "all_state_tensors_exact": set(reloaded["state"]) == set(state)
        and all(torch.equal(reloaded["state"][key], value) for key, value in state.items()),
        "tensor_count": len(state),
        "artifact": _path_metadata(args.artifact),
    }
    if not result["artifact_roundtrip"]["all_state_tensors_exact"]:
        raise RuntimeError("artifact roundtrip failed")
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
    parser.add_argument("--reference-result", type=Path, required=True)
    parser.add_argument("--reference-artifact", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--tables", type=int, default=32)
    parser.add_argument("--comparisons", type=int, default=4)
    parser.add_argument("--compiler-samples", type=int, default=32768)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--seeds", type=_parse_seeds, default=(0, 1, 2))
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-test", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.hidden_dim, args.tables, args.comparisons, args.compiler_samples, args.epochs) != (64, 32, 4, 32768, 10):
        parser.error("formal protocol requires D64/T32/C4/compiler32768/epochs10")
    if tuple(args.seeds) != (0, 1, 2):
        parser.error("formal protocol requires seeds 0,1,2")
    paths = (args.source_artifact, args.reference_result, args.reference_artifact)
    if not all(path.is_file() for path in paths):
        parser.error("source/reference files must exist")
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
