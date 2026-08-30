"""Released-stem EMNIST diagnostic for hard product-chart fields.

This is the second stage of the frozen-stem factorial.  Every arm receives an
independent copy of the same pretrained dense stem and trains that stem jointly
with its hard-forward product-chart head.  The dense stem is explicitly
counted: this probe tests coordinate co-adaptation, not a GEMM-free system.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tropnn.tools.emnist_payload_dtype_sweep import _load_emnist_split
from tropnn.tools.emnist_pq_product_grid_factorial import _source_linear
from tropnn.tools.emnist_product_chart_factorial import ARMS, Evaluation, evaluate, make_factorial_models, summarize


class EndToEndProductChart(nn.Module):
    """Dense GELU stem followed by one hard product-chart head."""

    def __init__(self, stem: nn.Linear, head: nn.Module) -> None:
        super().__init__()
        self.stem = stem
        self.head = head

    def features(self, x: Tensor) -> Tensor:
        return F.gelu(self.stem(x.flatten(1)))

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.features(x))


def make_end_to_end_models(
    source_state: dict[str, Tensor],
    seed: int,
    input_dim: int,
    classes: int,
    centroids: Tensor,
    rows: Tensor,
    *,
    rank: int,
    temperature: float,
) -> tuple[dict[str, EndToEndProductChart], float, dict[str, Tensor]]:
    """Create paired arms with exact shared initialization and separate stems."""

    source_stem = _source_linear(source_state, seed, "stem", input_dim, centroids.shape[0] * centroids.shape[2])
    initial_stem = {key: value.detach().clone() for key, value in source_stem.state_dict().items()}
    heads, factorization_error = make_factorial_models(
        centroids,
        rows,
        rank=rank,
        temperature=temperature,
        seed=seed,
    )
    models: dict[str, EndToEndProductChart] = {}
    for arm, head in heads.items():
        stem = copy.deepcopy(source_stem)
        stem.requires_grad_(True)
        models[arm] = EndToEndProductChart(stem, head)
    if classes != rows.shape[-1]:
        raise ValueError("class count and initialized row width disagree")
    return models, factorization_error, initial_stem


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _gradient_norm(parameters: object) -> float:
    total = 0.0
    for parameter in parameters:  # type: ignore[union-attr]
        if parameter.grad is not None:
            total += float(parameter.grad.detach().square().sum())
    return math.sqrt(total)


@torch.no_grad()
def _capture_features(model: EndToEndProductChart, x: Tensor, *, batch_size: int, device: torch.device) -> Tensor:
    result: list[Tensor] = []
    model.eval()
    for start in range(0, x.shape[0], batch_size):
        result.append(model.features(x[start : start + batch_size].to(device)).cpu())
    return torch.cat(result)


@torch.no_grad()
def _feature_variance(stem: nn.Linear, x: Tensor, *, batch_size: int, device: torch.device) -> float:
    values: list[Tensor] = []
    for start in range(0, x.shape[0], batch_size):
        values.append(F.gelu(stem(x[start : start + batch_size].to(device).flatten(1))).cpu())
    return float(torch.cat(values).var(dim=0, unbiased=False).mean().clamp_min(1e-12))


def _stem_motion(model: EndToEndProductChart, initial: dict[str, Tensor]) -> float:
    numerator = 0.0
    count = 0
    for key, value in model.stem.state_dict().items():
        delta = value.detach().cpu() - initial[key]
        numerator += float(delta.square().sum())
        count += delta.numel()
    return math.sqrt(numerator / count)


def _reference_comparison(rows: list[Evaluation], reference: dict[str, object]) -> dict[str, object]:
    if reference.get("schema") != "emnist-product-chart-factorial-v2":
        raise ValueError("unexpected frozen-stem result schema")
    reference_protocol = reference["protocol"]
    if not isinstance(reference_protocol, dict):
        raise TypeError("frozen-stem protocol is malformed")
    by_key = {(int(row["seed"]), str(row["arm"])): row for row in reference["rows"]}  # type: ignore[index]
    gains: dict[str, list[float]] = {}
    for arm in ARMS:
        gains[arm] = [float(by_key[row.seed, arm]["held_ce"]) - row.held_ce for row in rows if row.arm == arm]
    return {
        "reference_schema": reference["schema"],
        "reference_rank": reference_protocol["rank"],
        "released_stem_ce_gain_by_arm_and_seed": gains,
        "released_stem_positive_all_arms_all_seeds": all(value > 0 for values in gains.values() for value in values),
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
    centroids = pq_state[f"seed{seed}.pq.centroids"]
    rows = pq_state[f"seed{seed}.pq.free_rows"]
    models, factorization_error, initial_stem = make_end_to_end_models(
        source_state,
        seed,
        train_x[0].numel(),
        classes,
        centroids,
        rows,
        rank=args.rank,
        temperature=args.temperature,
    )
    models = {name: model.to(device) for name, model in models.items()}
    source_for_variance = copy.deepcopy(models[ARMS[0]].stem).requires_grad_(False)
    feature_variance = _feature_variance(source_for_variance, train_x, batch_size=args.batch_size, device=device)
    optimizer = torch.optim.AdamW(
        [parameter for model in models.values() for parameter in model.parameters()],
        lr=args.lr,
        weight_decay=0,
    )
    generator = torch.Generator(device=device).manual_seed(700_000 + seed)
    train_y_device = train_y.to(device)
    curves: list[dict[str, object]] = []
    first_gradients: dict[str, dict[str, float]] = {}
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        permutation = torch.randperm(train_y.numel(), generator=generator, device=device)
        loss_sum = {arm: 0.0 for arm in ARMS}
        correct = {arm: 0 for arm in ARMS}
        for start in range(0, train_y.numel(), args.batch_size):
            indices = permutation[start : start + args.batch_size]
            batch = train_x[indices.cpu()].to(device)
            target = train_y_device[indices]
            optimizer.zero_grad(set_to_none=True)
            losses: list[Tensor] = []
            for arm, model in models.items():
                features = model.features(batch)
                logits = model.head(features)
                loss = F.cross_entropy(logits, target)
                if arm.startswith("trained") and args.quantization_weight > 0:
                    residuals = model.head.chart_coordinates(features).residuals
                    loss = loss + args.quantization_weight * residuals.square().mean() / feature_variance
                losses.append(loss)
                count = target.numel()
                loss_sum[arm] += float(F.cross_entropy(logits.detach(), target)) * count
                correct[arm] += int((logits.detach().argmax(dim=-1) == target).sum())
            sum(losses).backward()
            if epoch == 1 and start == 0:
                for arm, model in models.items():
                    centroid = getattr(model.head, "centroids")
                    local_maps = getattr(model.head, "local_maps")
                    first_gradients[arm] = {
                        "stem": _gradient_norm(model.stem.parameters()),
                        "centroids": 0.0 if not isinstance(centroid, nn.Parameter) else _gradient_norm((centroid,)),
                        "local_maps": 0.0 if not isinstance(local_maps, nn.Parameter) else _gradient_norm((local_maps,)),
                        "head_all": _gradient_norm(model.head.parameters()),
                    }
            optimizer.step()
        curve: dict[str, object] = {"epoch": epoch}
        for arm in ARMS:
            curve[arm] = {
                "train_ce": loss_sum[arm] / train_y.numel(),
                "train_accuracy": correct[arm] / train_y.numel(),
            }
        curves.append(curve)
        print(
            f"seed={seed} epoch={epoch}/{args.epochs} " + " ".join(f"{arm}:ce={curve[arm]['train_ce']:.6f}" for arm in ARMS),  # type: ignore[index]
            flush=True,
        )

    evaluations: list[Evaluation] = []
    stem_motion: dict[str, float] = {}
    hard_replay: dict[str, float] = {}
    for arm, model in models.items():
        held_features = _capture_features(model, held_x, batch_size=args.batch_size, device=device)
        evaluations.append(evaluate(seed, arm, model.head, centroids, held_features, held_y))
        stem_motion[arm] = _stem_motion(model, initial_stem)
        with torch.no_grad():
            sample = held_x[:4096].to(device)
            explicit = model.head.hard_output(model.features(sample))[0]
            hard_replay[arm] = float((model(sample) - explicit).abs().max())
    audit = {
        "factorized_initial_row_relative_sse": factorization_error,
        "initial_feature_variance": feature_variance,
        "seconds": time.perf_counter() - started,
        "curves": curves,
        "first_step_gradient_norms": first_gradients,
        "stem_rms_motion": stem_motion,
        "wrapper_hard_replay_max_error": hard_replay,
        "all_stem_gradients_nonzero": all(values["stem"] > 0 for values in first_gradients.values()),
        "all_wrapper_hard_replays_exact": all(value == 0 for value in hard_replay.values()),
        "all_finite": all(math.isfinite(row.held_ce) for row in evaluations),
    }
    state = {f"{arm}.{key}": value.detach().cpu() for arm, model in models.items() for key, value in model.state_dict().items()}
    return evaluations, audit, state


def run(args: argparse.Namespace) -> dict[str, object]:
    source = torch.load(args.source_artifact, map_location="cpu", weights_only=False)
    pq = torch.load(args.pq_artifact, map_location="cpu", weights_only=False)
    reference = json.loads(args.frozen_result.read_text())
    if source.get("schema") != "emnist-maddness-task-ste-v1" or pq.get("schema") != "emnist-pq-product-grid-factorial-v1":
        raise ValueError("unexpected source artifact schema")
    if tuple(reference["protocol"]["seeds"]) != args.seeds or int(reference["protocol"]["rank"]) != args.rank:
        raise ValueError("frozen-stem reference does not match seeds/rank")
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
        bool(audit["all_stem_gradients_nonzero"] and audit["all_wrapper_hard_replays_exact"] and audit["all_finite"])
        for audit in audits.values()  # type: ignore[union-attr]
    ):
        raise RuntimeError("end-to-end product-chart audit failed")
    stem_parameters = train_x[0].numel() * args.hidden_dim + args.hidden_dim
    stem_macs = train_x[0].numel() * args.hidden_dim
    protocol = {
        "dataset": "EMNIST Balanced",
        "source_artifact": str(args.source_artifact.resolve()),
        "pq_artifact": str(args.pq_artifact.resolve()),
        "frozen_stem_reference": str(args.frozen_result.resolve()),
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
        "stem_released": True,
        "stem_initialized_from_dense_pretraining": True,
        "from_scratch": False,
        "hard_forward_for_every_arm": True,
        "soft_pq_backward_for_every_arm": True,
        "soft_pq_backward_semantics": "exact_hard_action_gradient_plus_soft_mixture_gradient",
        "held_used_for_selection": False,
        "dense_stem_diagnostic_not_low_cost": True,
        "device": args.device,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "dense_stem_parameters": stem_parameters,
        "dense_stem_macs": stem_macs,
        "exact_pq_route_squared_terms": 32 * 16 * 2,
        "hybrid_coordinate_analog_scalars": args.hidden_dim,
        "head_parameter_counts_including_centroids": reference["protocol"]["semantic_parameter_counts"],
    }
    result = {
        "schema": "emnist-product-chart-end-to-end-v1",
        "protocol": protocol,
        "rows": [asdict(row) for row in rows],
        "audits": audits,
        "summary": summarize(rows),
        "frozen_stem_comparison": _reference_comparison(rows, reference),
    }
    if args.artifact.exists():
        raise FileExistsError(args.artifact)
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": result["schema"], "protocol": protocol, "state": artifact_state}
    torch.save(payload, args.artifact)
    reloaded = torch.load(args.artifact, map_location="cpu", weights_only=False)
    exact = reloaded["schema"] == payload["schema"] and reloaded["protocol"] == payload["protocol"]
    exact = exact and reloaded["state"].keys() == payload["state"].keys()
    exact = exact and all(torch.equal(reloaded["state"][key], value) for key, value in payload["state"].items())
    result["artifact_roundtrip_exact"] = bool(exact)
    if not exact:
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
    parser.add_argument("--pq-artifact", type=Path, required=True)
    parser.add_argument("--frozen-result", type=Path, required=True)
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
        parser.error("this protocol requires D64 and rank 8 or 16")
    if args.temperature <= 0 or args.quantization_weight < 0 or args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("invalid optimization argument")
    if args.output == args.artifact or args.output.exists() or args.artifact.exists():
        parser.error("output and artifact must be distinct nonexistent paths")
    if not all(path.is_file() for path in (args.source_artifact, args.pq_artifact, args.frozen_result)):
        parser.error("source artifacts or frozen result are missing")
    return args


def main() -> None:
    args = parse_args()
    _atomic_json(args.output, run(args))


if __name__ == "__main__":
    main()
