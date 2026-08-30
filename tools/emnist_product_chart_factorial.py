"""Frozen-stem EMNIST factorial for hard product codes and local coordinates."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from tropnn.layers.product_chart import ProductChartField
from tropnn.tools.emnist_payload_dtype_sweep import _load_emnist_split
from tropnn.tools.emnist_pq_product_grid_factorial import _capture_features, _source_linear
from tropnn.tools.product_atlas_pc_action_factorial import _route_health

ARMS = (
    "frozen_constant",
    "trained_constant",
    "frozen_shared",
    "trained_shared",
    "frozen_local",
    "trained_local",
)


@dataclass(frozen=True)
class Evaluation:
    seed: int
    arm: str
    held_ce: float
    held_accuracy: float
    quantization_r2: float
    hybrid_reconstruction_max_error: float
    mean_entropy_bits: float
    minimum_entropy_bits: float
    mean_observed_codes: float
    maximum_code_mass: float
    mean_single_table_label_mi_bits: float
    hard_replay_max_error: float
    hard_soft_output_rmse: float
    centroid_rms_motion: float
    initial_code_change_fraction: float
    local_map_rms: float
    mean_neighbor_action_gap: float
    near_boundary_action_gap: float


def factorize_additive_rows(rows: Tensor, rank: int) -> tuple[Tensor, Tensor, float]:
    """Return the best shared rank-r factorization of flattened table rows."""

    if rows.ndim != 3 or rank < 1 or rank > min(rows.shape[0] * rows.shape[1], rows.shape[2]):
        raise ValueError("rows/rank geometry is invalid")
    matrix = rows.detach().cpu().double().reshape(-1, rows.shape[-1])
    _left, _values, right = torch.linalg.svd(matrix, full_matrices=False)
    basis = right[:rank]
    offsets = matrix @ basis.T
    reconstruction = offsets @ basis
    relative_error = float((matrix - reconstruction).square().sum() / matrix.square().sum().clamp_min(1e-30))
    return offsets.reshape(rows.shape[0], rows.shape[1], rank).float(), basis.float(), relative_error


def make_factorial_models(
    centroids: Tensor,
    rows: Tensor,
    *,
    rank: int,
    temperature: float,
    seed: int,
) -> tuple[dict[str, ProductChartField], float]:
    offsets, basis, factorization_error = factorize_additive_rows(rows, rank)
    models: dict[str, ProductChartField] = {}
    for arm in ARMS:
        action = "constant" if arm.endswith("constant") else "shared_linear" if arm.endswith("shared") else "local_linear"
        model = ProductChartField(
            centroids,
            rows.shape[-1],
            rank,
            action=action,
            surrogate="soft_pq",
            temperature=temperature,
            trainable_centroids=arm.startswith("trained"),
            seed=500_000 + seed,
        )
        with torch.no_grad():
            model.offsets.copy_(offsets)
            model.output_basis.copy_(basis)
            model.local_maps.zero_()
        models[arm] = model
    return models, factorization_error


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _single_table_label_mi_bits(codes: Tensor, labels: Tensor, code_count: int) -> float:
    if codes.ndim != 2 or labels.ndim != 1 or codes.shape[0] != labels.shape[0]:
        raise ValueError("codes and labels must be aligned")
    classes = int(labels.max()) + 1
    label_counts = torch.bincount(labels.cpu(), minlength=classes).double()
    label_probabilities = label_counts / label_counts.sum()
    label_entropy = float(-(label_probabilities * label_probabilities.clamp_min(1e-30).log2()).sum())
    values: list[float] = []
    for table in range(codes.shape[1]):
        joint = torch.bincount((codes[:, table].cpu() * classes + labels.cpu()), minlength=code_count * classes).double()
        joint = joint.reshape(code_count, classes)
        code_mass = joint.sum(dim=1)
        nonempty = code_mass > 0
        conditional = joint[nonempty] / code_mass[nonempty, None]
        entropy = -(conditional * conditional.clamp_min(1e-30).log2()).sum(dim=1)
        conditional_entropy = float((code_mass[nonempty] / labels.numel() * entropy).sum())
        values.append(label_entropy - conditional_entropy)
    return sum(values) / len(values)


@torch.no_grad()
def _neighbor_action_gaps(model: ProductChartField, x: Tensor, *, samples: int = 4096) -> tuple[float, float]:
    x = x[:samples]
    local = x.reshape(-1, model.tables, model.block_width)
    centroids = model.centroids.to(device=x.device, dtype=x.dtype)
    distances = (local.unsqueeze(-2) - centroids.unsqueeze(0)).square().sum(dim=-1)
    top = distances.topk(2, largest=False)
    first, second = top.indices[..., 0], top.indices[..., 1]
    table = torch.arange(model.tables, device=x.device).unsqueeze(0).expand(x.shape[0], -1)
    offsets = model.offsets.to(device=x.device, dtype=x.dtype)
    first_action = offsets[table, first]
    second_action = offsets[table, second]
    if model.action != "constant":
        maps = model.local_maps.to(device=x.device, dtype=x.dtype)
        if model.action == "shared_linear":
            first_maps = second_maps = maps[:, 0].unsqueeze(0).expand(x.shape[0], -1, -1, -1)
        else:
            first_maps = maps[table, first]
            second_maps = maps[table, second]
        first_action = first_action + torch.einsum("nts,ntsr->ntr", local - centroids[table, first], first_maps)
        second_action = second_action + torch.einsum("nts,ntsr->ntr", local - centroids[table, second], second_maps)
    delta = (first_action - second_action) @ model.output_basis.to(device=x.device, dtype=x.dtype)
    norms = delta.square().sum(dim=-1).sqrt()
    margins = top.values[..., 1] - top.values[..., 0]
    threshold = torch.quantile(margins.flatten(), 0.1)
    return float(norms.mean()), float(norms[margins <= threshold].mean())


@torch.no_grad()
def evaluate(
    seed: int,
    arm: str,
    model: ProductChartField,
    initial_centroids: Tensor,
    features: Tensor,
    labels: Tensor,
) -> Evaluation:
    device = next(model.parameters()).device
    x = features.to(device)
    target = labels.to(device)
    hard, codes = model.hard_output(x)
    deployed = model(x)
    soft = model.soft_output(x[:4096])
    hard_sample = model.hard_output(x[:4096])[0]
    coordinates = model.chart_coordinates(x)
    reconstructed = model.reconstruct(coordinates)
    local = x.reshape(-1, model.tables, model.block_width)
    centered = local - local.mean(dim=0, keepdim=True)
    quantization_r2 = 1.0 - float(coordinates.residuals.square().sum() / centered.square().sum().clamp_min(1e-30))
    entropy, minimum, observed, maximum = _route_health(codes.cpu(), model.codes)
    initial_local = x.reshape(-1, model.tables, model.block_width)
    initial_distances = (initial_local.unsqueeze(-2) - initial_centroids.to(device=x.device, dtype=x.dtype).unsqueeze(0)).square().sum(-1)
    initial_codes = initial_distances.argmin(dim=-1)
    mean_gap, boundary_gap = _neighbor_action_gaps(model, x)
    return Evaluation(
        seed=seed,
        arm=arm,
        held_ce=float(F.cross_entropy(hard, target)),
        held_accuracy=float((hard.argmax(dim=-1) == target).float().mean()),
        quantization_r2=quantization_r2,
        hybrid_reconstruction_max_error=float((reconstructed - x).abs().max()),
        mean_entropy_bits=entropy,
        minimum_entropy_bits=minimum,
        mean_observed_codes=observed,
        maximum_code_mass=maximum,
        mean_single_table_label_mi_bits=_single_table_label_mi_bits(codes.cpu(), labels.cpu(), model.codes),
        hard_replay_max_error=float((hard - deployed).abs().max()),
        hard_soft_output_rmse=float((hard_sample - soft).square().mean().sqrt()),
        centroid_rms_motion=float((model.centroids.detach().cpu() - initial_centroids).square().mean().sqrt()),
        initial_code_change_fraction=float((codes != initial_codes).float().mean()),
        local_map_rms=float(model.local_maps.detach().square().mean().sqrt()),
        mean_neighbor_action_gap=mean_gap,
        near_boundary_action_gap=boundary_gap,
    )


def summarize(rows: list[Evaluation]) -> dict[str, object]:
    seeds = sorted({row.seed for row in rows})
    by_key = {(row.seed, row.arm): row for row in rows}

    def effect(left: str, right: str) -> list[float]:
        return [by_key[seed, left].held_ce - by_key[seed, right].held_ce for seed in seeds]

    address_constant = effect("frozen_constant", "trained_constant")
    address_shared = effect("frozen_shared", "trained_shared")
    address_local = effect("frozen_local", "trained_local")
    shared_frozen = effect("frozen_constant", "frozen_shared")
    shared_trained = effect("trained_constant", "trained_shared")
    conditioned_frozen = effect("frozen_shared", "frozen_local")
    conditioned_trained = effect("trained_shared", "trained_local")
    local_total_frozen = effect("frozen_constant", "frozen_local")
    local_total_trained = effect("trained_constant", "trained_local")
    interaction = [trained - frozen for trained, frozen in zip(local_total_trained, local_total_frozen)]
    arms = {
        arm: {
            "held_ce_mean": sum(by_key[seed, arm].held_ce for seed in seeds) / len(seeds),
            "held_accuracy_mean": sum(by_key[seed, arm].held_accuracy for seed in seeds) / len(seeds),
        }
        for arm in ARMS
    }
    return {
        "arms": arms,
        "effects": {
            "trained_address_gain_under_constant_by_seed": address_constant,
            "trained_address_gain_under_shared_by_seed": address_shared,
            "trained_address_gain_under_local_by_seed": address_local,
            "shared_linear_gain_under_frozen_address_by_seed": shared_frozen,
            "shared_linear_gain_under_trained_address_by_seed": shared_trained,
            "code_conditioned_slope_gain_under_frozen_address_by_seed": conditioned_frozen,
            "code_conditioned_slope_gain_under_trained_address_by_seed": conditioned_trained,
            "local_field_total_gain_under_frozen_address_by_seed": local_total_frozen,
            "local_field_total_gain_under_trained_address_by_seed": local_total_trained,
            "difference_in_differences_by_seed": interaction,
        },
        "signals": {
            "trained_address_positive_all_actions_all_seeds": all(value > 0 for value in address_constant + address_shared + address_local),
            "shared_linear_positive_both_addresses_all_seeds": all(value > 0 for value in shared_frozen + shared_trained),
            "code_conditioned_slope_positive_both_addresses_all_seeds": all(value > 0 for value in conditioned_frozen + conditioned_trained),
        },
    }


def train_seed(
    seed: int,
    args: argparse.Namespace,
    source_state: dict[str, Tensor],
    pq_state: dict[str, Tensor],
    train_x: Tensor,
    train_y: Tensor,
    held_x: Tensor,
    held_y: Tensor,
) -> tuple[list[Evaluation], dict[str, object], dict[str, Tensor]]:
    device = torch.device(args.device)
    classes = int(max(int(train_y.max()), int(held_y.max())) + 1)
    stem = _source_linear(source_state, seed, "stem", train_x[0].numel(), args.hidden_dim).to(device)
    dense_head = _source_linear(source_state, seed, "head", args.hidden_dim, classes).to(device)
    train_features, _train_dense = _capture_features(train_x, stem, dense_head, batch_size=args.batch_size, device=device)
    held_features, held_dense = _capture_features(held_x, stem, dense_head, batch_size=args.batch_size, device=device)
    centroids = pq_state[f"seed{seed}.pq.centroids"]
    rows = pq_state[f"seed{seed}.pq.free_rows"]
    models, factorization_error = make_factorial_models(
        centroids,
        rows,
        rank=args.rank,
        temperature=args.temperature,
        seed=seed,
    )
    models = {name: model.to(device) for name, model in models.items()}
    parameters = [parameter for model in models.values() for parameter in model.parameters()]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0)
    generator = torch.Generator(device=device).manual_seed(600_000 + seed)
    feature_variance = float(train_features.var(dim=0, unbiased=False).mean().clamp_min(1e-12))
    curves: list[dict[str, object]] = []
    started = time.perf_counter()
    train_features_device = train_features.to(device)
    train_y_device = train_y.to(device)
    for epoch in range(1, args.epochs + 1):
        permutation = torch.randperm(train_y.numel(), generator=generator, device=device)
        loss_sum = {arm: 0.0 for arm in ARMS}
        correct = {arm: 0 for arm in ARMS}
        for start in range(0, train_y.numel(), args.batch_size):
            indices = permutation[start : start + args.batch_size]
            batch = train_features_device[indices]
            target = train_y_device[indices]
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for arm, model in models.items():
                logits = model(batch)
                loss = F.cross_entropy(logits, target)
                if arm.startswith("trained") and args.quantization_weight > 0:
                    residuals = model.chart_coordinates(batch).residuals
                    loss = loss + args.quantization_weight * residuals.square().mean() / feature_variance
                losses.append(loss)
                count = target.numel()
                loss_sum[arm] += float(F.cross_entropy(logits.detach(), target)) * count
                correct[arm] += int((logits.detach().argmax(dim=-1) == target).sum())
            sum(losses).backward()
            optimizer.step()
        row: dict[str, object] = {"epoch": epoch}
        for arm in ARMS:
            row[arm] = {
                "train_ce": loss_sum[arm] / train_y.numel(),
                "train_accuracy": correct[arm] / train_y.numel(),
            }
        curves.append(row)
        print(
            f"seed={seed} epoch={epoch}/{args.epochs} " + " ".join(f"{arm}:ce={row[arm]['train_ce']:.6f}" for arm in ARMS),  # type: ignore[index]
            flush=True,
        )

    evaluations = [evaluate(seed, arm, model, centroids, held_features, held_y) for arm, model in models.items()]
    dense_ce = float(F.cross_entropy(held_dense, held_y))
    audit = {
        "dense_source_held_ce": dense_ce,
        "factorized_initial_row_relative_sse": factorization_error,
        "feature_variance": feature_variance,
        "seconds": time.perf_counter() - started,
        "curves": curves,
        "all_finite": all(
            math.isfinite(value)
            for row in evaluations
            for value in (
                row.held_ce,
                row.held_accuracy,
                row.quantization_r2,
                row.hard_soft_output_rmse,
                row.centroid_rms_motion,
            )
        ),
        "all_hard_replays_exact": all(row.hard_replay_max_error == 0 for row in evaluations),
        "all_hybrid_reconstructions_close": all(row.hybrid_reconstruction_max_error <= 2e-5 for row in evaluations),
    }
    state = {f"{arm}.{key}": value.detach().cpu() for arm, model in models.items() for key, value in model.state_dict().items()}
    return evaluations, audit, state


def run(args: argparse.Namespace) -> dict[str, object]:
    source = torch.load(args.source_artifact, map_location="cpu", weights_only=False)
    pq = torch.load(args.pq_artifact, map_location="cpu", weights_only=False)
    if source.get("schema") != "emnist-maddness-task-ste-v1" or pq.get("schema") != "emnist-pq-product-grid-factorial-v1":
        raise ValueError("unexpected source artifact schema")
    train_x, train_y = _load_emnist_split(args.root, "balanced", train=True, limit=args.max_train, seed=0)
    held_x, held_y = _load_emnist_split(args.root, "balanced", train=False, limit=args.max_test, seed=0)
    rows: list[Evaluation] = []
    audits: dict[str, object] = {}
    artifact_state: dict[str, Tensor] = {}
    for seed in args.seeds:
        seed_rows, audit, state = train_seed(seed, args, source["state"], pq["state"], train_x, train_y, held_x, held_y)
        rows.extend(seed_rows)
        audits[str(seed)] = audit
        artifact_state.update({f"seed{seed}.{key}": value for key, value in state.items()})
        print(
            f"seed={seed} " + " ".join(f"{row.arm}:ce={row.held_ce:.6f},acc={row.held_accuracy:.6f}" for row in seed_rows),
            flush=True,
        )
    if not all(
        bool(audit["all_finite"] and audit["all_hard_replays_exact"] and audit["all_hybrid_reconstructions_close"])
        for audit in audits.values()  # type: ignore[union-attr]
    ):
        raise RuntimeError("mechanical product-chart audit failed")
    protocol = {
        "dataset": "EMNIST Balanced",
        "source_artifact": str(args.source_artifact.resolve()),
        "pq_artifact": str(args.pq_artifact.resolve()),
        "hidden_dim": args.hidden_dim,
        "output_dim": 47,
        "tables": 32,
        "block_width": 2,
        "codes_per_table": 16,
        "rank": args.rank,
        "temperature": args.temperature,
        "quantization_weight": args.quantization_weight,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "seeds": list(args.seeds),
        "train_examples": len(train_x),
        "held_examples": len(held_x),
        "stem_frozen": True,
        "hard_forward_for_every_arm": True,
        "soft_pq_backward_for_every_arm": True,
        "soft_pq_backward_semantics": "exact_hard_action_gradient_plus_soft_mixture_gradient",
        "continuous_residual_counted_as_hybrid_recognizer_output": True,
        "held_used_for_selection": False,
        "device": args.device,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "exact_pq_route_squared_terms": 32 * 16 * 2,
        "hybrid_coordinate_analog_scalars": args.hidden_dim,
        "semantic_parameter_counts": {
            "factorized_constant_including_centroids": args.hidden_dim * 16 + 32 * 16 * args.rank + args.rank * 47,
            "shared_linear_including_centroids": args.hidden_dim * 16 + 32 * 16 * args.rank + args.hidden_dim * args.rank + args.rank * 47,
            "local_field_including_centroids": args.hidden_dim * 16 + 32 * 16 * args.rank + args.hidden_dim * 16 * args.rank + args.rank * 47,
            "full_additive_pq_rows_including_centroids": args.hidden_dim * 16 + 32 * 16 * 47,
            "dense_linear_head_including_bias": args.hidden_dim * 47 + 47,
        },
        "active_constant_action_macs": 47 * args.rank,
        "active_local_field_macs": args.hidden_dim * args.rank + 47 * args.rank,
    }
    result = {
        "schema": "emnist-product-chart-factorial-v2",
        "protocol": protocol,
        "rows": [asdict(row) for row in rows],
        "audits": audits,
        "summary": summarize(rows),
    }
    if args.artifact.exists():
        raise FileExistsError(args.artifact)
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
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
    parser.add_argument("--pq-artifact", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--quantization-weight", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seeds", type=_parse_seeds, default=(0,))
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-test", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.hidden_dim != 64 or args.rank not in {8, 16}:
        parser.error("the first protocol requires D64 and rank 8 or 16")
    if args.temperature <= 0 or args.quantization_weight < 0 or args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("invalid optimization argument")
    if args.output == args.artifact or args.output.exists() or args.artifact.exists():
        parser.error("output and artifact must be distinct nonexistent paths")
    if not args.source_artifact.is_file() or not args.pq_artifact.is_file():
        parser.error("source artifacts are missing")
    return args


def main() -> None:
    args = parse_args()
    _atomic_json(args.output, run(args))


if __name__ == "__main__":
    main()
