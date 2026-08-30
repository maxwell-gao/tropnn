"""End-to-end EMNIST recognition-by-decoder factorial.

The experiment has no dense stem and no offline compiler.  Five heads consume
the same normalized 784 pixels:

* ``raw_flat`` and ``raw_merge`` use original-coordinate pair codes;
* ``fwht_flat`` and ``fwht_merge`` use a fixed randomized FWHT before the
  otherwise matched pair codes;
* flat decoders add one action row per leaf code, whereas merge decoders map
  adjacent code pairs through learned 12-bit-to-6-bit integer maps first;
* ``dense``: one linear map, included only as a hardware-natural calibration.

Every LUT arm uses an exactly hard forward during training.  Its route credit
comes from local counterfactual action differences; the held split is never
used for optimization or model selection.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from tropnn.layers.fwht_code_merge import DirectCodeMergeLUT, FWHTCodeMergeLUT, FWHTFlatPairLUT, make_disjoint_pair_supports
from tropnn.layers.hard_lookup import sum_lookup_rows, weighted_neighbor_delta
from tropnn.tools.emnist_payload_dtype_sweep import _load_emnist_split

SCHEMA = "emnist-recognition-decoder-factorial-v2"
ARMS = ("raw_flat", "raw_merge", "fwht_flat", "fwht_merge", "dense")


def _pack_lsb(bits: Tensor) -> Tensor:
    powers = 2 ** torch.arange(bits.shape[-1], device=bits.device, dtype=torch.int64)
    return (bits.to(torch.int64) * powers).sum(dim=-1)


def _zero_forward_sigmoid(margins: Tensor, tau: float) -> Tensor:
    soft = torch.sigmoid(margins / tau)
    return soft - soft.detach()


class DirectFlatPairLUT(nn.Module):
    """Flat PC-LUT semantics with the same all-bit counterfactual STE control."""

    def __init__(self, input_dim: int, output_dim: int, supports: Tensor, rows: Tensor, *, tau: float) -> None:
        super().__init__()
        if supports.ndim != 3 or supports.shape[-1] != 2:
            raise ValueError("supports must be [tables,comparisons,2]")
        if rows.shape[:2] != (supports.shape[0], 1 << supports.shape[1]) or rows.shape[-1] != output_dim:
            raise ValueError("rows do not match pair geometry")
        self.input_dim = int(input_dim)
        self.tables = int(supports.shape[0])
        self.comparisons = int(supports.shape[1])
        self.tau = float(tau)
        self.register_buffer("supports", supports.to(torch.int64).contiguous(), persistent=True)
        self.register_buffer("powers", 2 ** torch.arange(self.comparisons, dtype=torch.int64), persistent=False)
        self.thresholds = nn.Parameter(torch.zeros(self.tables, self.comparisons))
        self.action_rows = nn.Parameter(rows.detach().clone())
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def route(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if x.ndim != 2 or x.shape[-1] != self.input_dim:
            raise ValueError("direct pair input shape mismatch")
        supports = self.supports.to(device=x.device)
        margins = x[:, supports[..., 0]] - x[:, supports[..., 1]] - self.thresholds.to(device=x.device, dtype=x.dtype)
        return _pack_lsb(margins > 0), margins

    def hard_codes(self, x: Tensor) -> Tensor:
        return self.route(x)[0]

    def hard_output(self, x: Tensor) -> tuple[Tensor, Tensor]:
        codes, _margins = self.route(x)
        return sum_lookup_rows(self.action_rows, codes, accumulation_dtype=torch.float32) + self.bias, codes

    def forward(self, x: Tensor) -> Tensor:
        codes, margins = self.route(x)
        hard = sum_lookup_rows(self.action_rows, codes, accumulation_dtype=torch.float32) + self.bias
        if not self.training:
            return hard
        neighbors = codes.unsqueeze(-1) ^ self.powers.to(device=x.device)
        weights = _zero_forward_sigmoid(margins, self.tau)
        return hard + weighted_neighbor_delta(self.action_rows, codes, neighbors, weights).to(hard.dtype)


@dataclass(frozen=True)
class Evaluation:
    seed: int
    arm: str
    held_ce: float
    held_accuracy: float
    parameter_count: int
    route_parameter_count: int
    action_parameter_count: int
    mean_code_entropy_bits: float | None
    minimum_code_entropy_bits: float | None
    mean_observed_rows: float | None
    maximum_row_mass: float | None
    leaf_mean_entropy_bits: float | None
    hard_replay_max_error: float
    eager_hard_forward_ms: float


def make_models(args: argparse.Namespace, seed: int) -> dict[str, nn.Module]:
    raw_supports = make_disjoint_pair_supports(
        args.input_dim,
        args.tables,
        args.comparisons,
        seed=10_000 + seed,
    )
    fwht_supports = make_disjoint_pair_supports(
        args.transform_dim,
        args.tables,
        args.comparisons,
        seed=20_000 + seed,
    )
    generator = torch.Generator(device="cpu").manual_seed(30_000 + seed)
    flat_std = args.initial_logit_std / math.sqrt(args.tables)
    common_rows = torch.randn(args.tables, 1 << args.comparisons, args.classes, generator=generator) * flat_std
    raw_flat = DirectFlatPairLUT(args.input_dim, args.classes, raw_supports, common_rows, tau=args.route_temperature)
    fwht_flat = FWHTFlatPairLUT(
        args.input_dim,
        args.transform_dim,
        args.classes,
        fwht_supports,
        seed=40_000 + seed,
        row_init_std=flat_std,
        normalize=True,
        tau=args.route_temperature,
    )
    with torch.no_grad():
        fwht_flat.action_rows.copy_(common_rows)
    merge_std = args.initial_logit_std / math.sqrt(args.tables // 2)
    raw_merge = DirectCodeMergeLUT(
        args.input_dim,
        args.classes,
        raw_supports,
        seed=45_000 + seed,
        row_init_std=merge_std,
        merger_init_logit=args.merger_init_logit,
        merger_initialization=args.merger_initialization,
        tau=args.route_temperature,
    )
    fwht_merge = FWHTCodeMergeLUT(
        args.input_dim,
        args.transform_dim,
        args.classes,
        fwht_supports,
        seed=40_000 + seed,
        row_init_std=merge_std,
        merger_init_logit=args.merger_init_logit,
        merger_initialization=args.merger_initialization,
        normalize=True,
        tau=args.route_temperature,
    )
    with torch.no_grad():
        fwht_merge.initial_merger_map.copy_(raw_merge.initial_merger_map)
        fwht_merge.merger_logits.copy_(raw_merge.merger_logits)
        fwht_merge.action_rows.copy_(raw_merge.action_rows)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(50_000 + seed)
        dense = nn.Linear(args.input_dim, args.classes)
    return {
        "raw_flat": raw_flat,
        "raw_merge": raw_merge,
        "fwht_flat": fwht_flat,
        "fwht_merge": fwht_merge,
        "dense": dense,
    }


def route_and_action_parameters(model: nn.Module, arm: str) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    if arm in {"raw_flat", "fwht_flat"}:
        return [model.thresholds], [model.action_rows, model.bias]  # type: ignore[attr-defined]
    if arm in {"raw_merge", "fwht_merge"}:
        return [model.thresholds, model.merger_logits], [model.action_rows, model.bias]  # type: ignore[attr-defined]
    return [], list(model.parameters())


def hard_output(model: nn.Module, arm: str, x: Tensor) -> tuple[Tensor, Tensor | None]:
    if arm == "dense":
        return model(x), None
    return model.hard_output(x)  # type: ignore[attr-defined,no-any-return]


def _gradient_norm(parameters: list[nn.Parameter]) -> float:
    total = sum(float(parameter.grad.detach().square().sum()) for parameter in parameters if parameter.grad is not None)
    return math.sqrt(total)


def _parameter_rms_motion(parameters: list[nn.Parameter], references: list[Tensor]) -> float:
    total, count = 0.0, 0
    for parameter, reference in zip(parameters, references, strict=True):
        difference = parameter.detach().cpu() - reference
        total += float(difference.square().sum())
        count += difference.numel()
    return math.sqrt(total / max(1, count))


def _route_health(counts: Tensor) -> tuple[float, float, float, float]:
    probabilities = counts.double() / counts.sum(dim=-1, keepdim=True).clamp_min(1)
    entropy = -(probabilities * probabilities.clamp_min(1e-300).log2()).sum(dim=-1)
    return (
        float(entropy.mean()),
        float(entropy.min()),
        float((counts > 0).sum(dim=-1).double().mean()),
        float(probabilities.max()),
    )


@torch.no_grad()
def evaluate(
    seed: int,
    arm: str,
    model: nn.Module,
    x: Tensor,
    y: Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> Evaluation:
    model.eval()
    is_merge = arm in {"raw_merge", "fwht_merge"}
    tables = args.tables // 2 if is_merge else args.tables
    counts = None if arm == "dense" else torch.zeros(tables, 1 << args.comparisons, dtype=torch.int64)
    leaf_counts = torch.zeros(args.tables, 1 << args.comparisons, dtype=torch.int64) if is_merge else None
    loss_sum, correct, replay = 0.0, 0, 0.0
    for start in range(0, len(y), args.batch_size):
        batch = x[start : start + args.batch_size].to(device)
        target = y[start : start + args.batch_size].to(device)
        forward = model(batch)
        explicit, codes = hard_output(model, arm, batch)
        replay = max(replay, float((forward - explicit).abs().max()))
        loss_sum += float(F.cross_entropy(explicit, target, reduction="sum"))
        correct += int((explicit.argmax(dim=-1) == target).sum())
        if counts is not None and codes is not None:
            offset = (1 << args.comparisons) * torch.arange(tables, device=device).view(1, -1)
            counts += (
                torch.bincount((codes + offset).flatten(), minlength=tables * (1 << args.comparisons)).reshape(tables, 1 << args.comparisons).cpu()
            )
        if leaf_counts is not None:
            leaf_codes = model.encoder.route(batch).codes  # type: ignore[attr-defined]
            leaf_offset = (1 << args.comparisons) * torch.arange(args.tables, device=device).view(1, -1)
            leaf_counts += (
                torch.bincount((leaf_codes + leaf_offset).flatten(), minlength=args.tables * (1 << args.comparisons))
                .reshape(args.tables, 1 << args.comparisons)
                .cpu()
            )
    route, action = route_and_action_parameters(model, arm)
    entropy = minimum = observed = maximum = leaf_entropy = None
    if counts is not None:
        entropy, minimum, observed, maximum = _route_health(counts)
    if leaf_counts is not None:
        leaf_entropy = _route_health(leaf_counts)[0]
    sample = x[: min(args.benchmark_batch_size, len(x))].to(device)
    elapsed_ms = benchmark_hard_forward(model, arm, sample, args.benchmark_warmups, args.benchmark_iters, device)
    return Evaluation(
        seed=seed,
        arm=arm,
        held_ce=loss_sum / len(y),
        held_accuracy=correct / len(y),
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        route_parameter_count=sum(parameter.numel() for parameter in route),
        action_parameter_count=sum(parameter.numel() for parameter in action),
        mean_code_entropy_bits=entropy,
        minimum_code_entropy_bits=minimum,
        mean_observed_rows=observed,
        maximum_row_mass=maximum,
        leaf_mean_entropy_bits=leaf_entropy,
        hard_replay_max_error=replay,
        eager_hard_forward_ms=elapsed_ms,
    )


def benchmark_hard_forward(
    model: nn.Module,
    arm: str,
    sample: Tensor,
    warmups: int,
    iterations: int,
    device: torch.device,
) -> float:
    for _ in range(warmups):
        hard_output(model, arm, sample)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(iterations):
            hard_output(model, arm, sample)
        end.record()
        torch.cuda.synchronize(device)
        return float(begin.elapsed_time(end) / iterations)
    started = time.perf_counter()
    for _ in range(iterations):
        hard_output(model, arm, sample)
    return 1000.0 * (time.perf_counter() - started) / iterations


def train_seed(
    seed: int,
    args: argparse.Namespace,
    train_x: Tensor,
    train_y: Tensor,
    held_x: Tensor,
    held_y: Tensor,
) -> tuple[list[Evaluation], dict[str, object], dict[str, Tensor]]:
    device = torch.device(args.device)
    models = {arm: model.to(device) for arm, model in make_models(args, seed).items()}
    if not torch.equal(models["fwht_flat"].encoder.transform.signs, models["fwht_merge"].encoder.transform.signs):  # type: ignore[attr-defined]
        raise RuntimeError("FWHT signs are not shared")
    if not torch.equal(models["fwht_flat"].encoder.supports, models["fwht_merge"].encoder.supports):  # type: ignore[attr-defined]
        raise RuntimeError("FWHT pair supports are not shared")
    if not torch.equal(models["raw_flat"].supports, models["raw_merge"].encoder.supports):  # type: ignore[attr-defined]
        raise RuntimeError("raw pair supports are not shared")
    if not torch.equal(models["raw_flat"].action_rows, models["fwht_flat"].action_rows):  # type: ignore[attr-defined]
        raise RuntimeError("matched flat action initialization differs")
    if not all(
        torch.equal(getattr(models["raw_merge"], field), getattr(models["fwht_merge"], field))
        for field in ("initial_merger_map", "merger_logits", "action_rows", "bias")
    ):
        raise RuntimeError("matched merge initialization differs")

    route_parameters: dict[str, list[nn.Parameter]] = {}
    action_parameters: dict[str, list[nn.Parameter]] = {}
    initial_route: dict[str, list[Tensor]] = {}
    for arm, model in models.items():
        route, action = route_and_action_parameters(model, arm)
        route_parameters[arm] = route
        action_parameters[arm] = action
        initial_route[arm] = [parameter.detach().cpu().clone() for parameter in route]
    optimizer = torch.optim.AdamW(
        [
            {"params": [parameter for arm in ARMS for parameter in action_parameters[arm]], "lr": args.lr},
            {"params": [parameter for arm in ARMS for parameter in route_parameters[arm]], "lr": args.lr * args.route_lr_multiplier},
        ],
        weight_decay=0,
    )
    generator = torch.Generator(device="cpu").manual_seed(60_000 + seed)
    curves: list[dict[str, object]] = []
    first_gradients: dict[str, dict[str, float]] = {}
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        for model in models.values():
            model.train()
        permutation = torch.randperm(len(train_y), generator=generator)
        loss_sum = {arm: 0.0 for arm in ARMS}
        correct = {arm: 0 for arm in ARMS}
        for start in range(0, len(train_y), args.batch_size):
            indices = permutation[start : start + args.batch_size]
            batch = train_x[indices].to(device)
            target = train_y[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = {arm: model(batch) for arm, model in models.items()}
            losses = {arm: F.cross_entropy(output, target) for arm, output in logits.items()}
            sum(losses.values()).backward()
            if epoch == 1 and start == 0:
                first_gradients = {
                    arm: {
                        "route": _gradient_norm(route_parameters[arm]),
                        "action": _gradient_norm(action_parameters[arm]),
                    }
                    for arm in ARMS
                }
                for merge_arm in ("raw_merge", "fwht_merge"):
                    merge = models[merge_arm]
                    first_gradients[merge_arm].update(
                        {
                            "leaf_threshold": _gradient_norm([merge.thresholds]),  # type: ignore[attr-defined]
                            "merger_logits": _gradient_norm([merge.merger_logits]),  # type: ignore[attr-defined]
                        }
                    )
            optimizer.step()
            count = target.numel()
            for arm in ARMS:
                loss_sum[arm] += float(losses[arm].detach()) * count
                correct[arm] += int((logits[arm].detach().argmax(dim=-1) == target).sum())
        curve: dict[str, object] = {"epoch": epoch}
        for arm in ARMS:
            curve[arm] = {
                "train_ce": loss_sum[arm] / len(train_y),
                "train_accuracy": correct[arm] / len(train_y),
            }
        curves.append(curve)
        print(
            f"seed={seed} epoch={epoch}/{args.epochs} " + " ".join(f"{arm}:ce={curve[arm]['train_ce']:.6f}" for arm in ARMS),  # type: ignore[index]
            flush=True,
        )

    evaluations = [evaluate(seed, arm, models[arm], held_x, held_y, args, device) for arm in ARMS]
    route_motion = {arm: _parameter_rms_motion(route_parameters[arm], initial_route[arm]) if route_parameters[arm] else 0.0 for arm in ARMS}
    merger_code_flip = {
        arm: float((models[arm].compiled_merger_map() != models[arm].initial_merger_map).float().mean())  # type: ignore[attr-defined]
        for arm in ("raw_merge", "fwht_merge")
    }
    state = {f"{arm}.{key}": value.detach().cpu() for arm, model in models.items() for key, value in model.state_dict().items()}
    audit: dict[str, object] = {
        "seconds": time.perf_counter() - started,
        "training_curves": curves,
        "first_step_gradient_norms": first_gradients,
        "route_parameter_rms_motion": route_motion,
        "merger_map_fraction_changed_from_initial": merger_code_flip,
        "both_merger_maps_changed_from_initial": all(value > 0 for value in merger_code_flip.values()),
        "recognizer_supports_shared_within_decoder_contrasts": True,
        "fwht_signs_shared_within_decoder_contrast": True,
        "flat_initial_action_rows_exactly_shared": True,
        "merge_initial_maps_logits_and_action_rows_exactly_shared": True,
        "all_route_gradients_nonzero": all(first_gradients[arm]["route"] > 0 for arm in ARMS[:-1]),
        "all_merge_leaf_and_merger_gradients_nonzero": all(
            first_gradients[arm][field] > 0 for arm in ("raw_merge", "fwht_merge") for field in ("leaf_threshold", "merger_logits")
        ),
        "all_action_gradients_nonzero": all(first_gradients[arm]["action"] > 0 for arm in ARMS),
        "all_hard_replays_exact": all(row.hard_replay_max_error == 0 for row in evaluations),
        "all_finite": all(math.isfinite(row.held_ce) for row in evaluations),
    }
    return evaluations, audit, state


def operation_ledger(args: argparse.Namespace) -> dict[str, dict[str, int]]:
    c, t, o, n = args.comparisons, args.tables, args.classes, args.transform_dim
    flat_action_scalars = t * o
    merged_action_scalars = (t // 2) * o
    return {
        "raw_flat": {
            "coordinate_reads": 2 * t * c,
            "pair_subtracts": t * c,
            "threshold_compares": t * c,
            "action_row_lookups": t,
            "active_action_scalar_reads": flat_action_scalars,
        },
        "raw_merge": {
            "coordinate_reads": 2 * t * c,
            "pair_subtracts": t * c,
            "threshold_compares": t * c,
            "compiled_merger_map_lookups": t // 2,
            "action_row_lookups": t // 2,
            "active_action_scalar_reads": merged_action_scalars,
        },
        "fwht_flat": {
            "fwht_add_subtracts": n * int(math.log2(n)),
            "pair_coordinate_reads": 2 * t * c,
            "pair_subtracts": t * c,
            "threshold_compares": t * c,
            "action_row_lookups": t,
            "active_action_scalar_reads": flat_action_scalars,
        },
        "fwht_merge": {
            "fwht_add_subtracts": n * int(math.log2(n)),
            "pair_coordinate_reads": 2 * t * c,
            "pair_subtracts": t * c,
            "threshold_compares": t * c,
            "compiled_merger_map_lookups": t // 2,
            "action_row_lookups": t // 2,
            "active_action_scalar_reads": merged_action_scalars,
        },
        "dense": {
            "multiply_accumulates": args.input_dim * o,
            "weight_scalar_reads": args.input_dim * o,
        },
    }


def summarize(rows: list[Evaluation]) -> dict[str, object]:
    arms: dict[str, dict[str, float]] = {}
    for arm in ARMS:
        selected = [row for row in rows if row.arm == arm]
        arms[arm] = {
            "held_ce_mean": sum(row.held_ce for row in selected) / len(selected),
            "held_accuracy_mean": sum(row.held_accuracy for row in selected) / len(selected),
            "hard_forward_ms_mean": sum(row.eager_hard_forward_ms for row in selected) / len(selected),
        }
    ce = {arm: arms[arm]["held_ce_mean"] for arm in ARMS}
    signed = {
        "fwht_flat_minus_raw_flat": ce["fwht_flat"] - ce["raw_flat"],
        "raw_merge_minus_raw_flat": ce["raw_merge"] - ce["raw_flat"],
        "fwht_merge_minus_fwht_flat": ce["fwht_merge"] - ce["fwht_flat"],
    }
    signed["difference_in_differences"] = signed["fwht_merge_minus_fwht_flat"] - signed["raw_merge_minus_raw_flat"]
    return {
        "arms": arms,
        "ce_order_best_to_worst": sorted(ARMS, key=lambda arm: arms[arm]["held_ce_mean"]),
        "signed_ce_contrasts_requested_order": signed,
        "positive_means_improvement": {
            "fwht_under_flat_decoder": ce["raw_flat"] - ce["fwht_flat"],
            "fwht_under_merge_decoder": ce["raw_merge"] - ce["fwht_merge"],
            "merge_under_raw_recognizer": ce["raw_flat"] - ce["raw_merge"],
            "merge_under_fwht_recognizer": ce["fwht_flat"] - ce["fwht_merge"],
            "synergy": -signed["difference_in_differences"],
        },
    }


def _save_exclusive(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        torch.save(value, handle)


def _write_json_exclusive(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _path_metadata(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def run(args: argparse.Namespace) -> dict[str, object]:
    train_x, train_y = _load_emnist_split(args.root, "balanced", train=True, limit=args.max_train, seed=0)
    held_x, held_y = _load_emnist_split(args.root, "balanced", train=False, limit=args.max_test, seed=0)
    if train_x.shape[-1] != args.input_dim:
        raise ValueError("EMNIST input dimension differs from configured geometry")
    if int(max(int(train_y.max()), int(held_y.max())) + 1) != args.classes:
        raise ValueError("EMNIST class count differs from configured geometry")
    rows: list[Evaluation] = []
    audits: dict[str, object] = {}
    artifact_state: dict[str, Tensor] = {}
    for seed in args.seeds:
        seed_rows, audit, state = train_seed(seed, args, train_x, train_y, held_x, held_y)
        rows.extend(seed_rows)
        audits[str(seed)] = audit
        artifact_state.update({f"seed{seed}.{key}": value for key, value in state.items()})
        print(" ".join(f"{row.arm}:held_ce={row.held_ce:.6f},acc={row.held_accuracy:.6f}" for row in seed_rows), flush=True)
    if not all(
        bool(audit[gate])
        for audit in audits.values()  # type: ignore[union-attr]
        for gate in (
            "recognizer_supports_shared_within_decoder_contrasts",
            "fwht_signs_shared_within_decoder_contrast",
            "flat_initial_action_rows_exactly_shared",
            "merge_initial_maps_logits_and_action_rows_exactly_shared",
            "all_route_gradients_nonzero",
            "all_merge_leaf_and_merger_gradients_nonzero",
            "both_merger_maps_changed_from_initial",
            "all_action_gradients_nonzero",
            "all_hard_replays_exact",
            "all_finite",
        )
    ):
        raise RuntimeError("training semantic audit failed")
    protocol = {
        "dataset": "EMNIST Balanced",
        "data_root": str(args.root.resolve()),
        "train_examples": len(train_x),
        "held_examples": len(held_x),
        "seeds": list(args.seeds),
        "input_dim": args.input_dim,
        "classes": args.classes,
        "transform": "fixed randomized normalized Walsh-Hadamard after zero padding",
        "transform_dim": args.transform_dim,
        "tables": args.tables,
        "comparisons": args.comparisons,
        "leaf_code_bits": args.comparisons,
        "mergers": args.tables // 2,
        "merger_input_rows": 1 << (2 * args.comparisons),
        "merger_output_rows": 1 << args.comparisons,
        "merger_initialization": args.merger_initialization,
        "factorial_axes": {
            "recognition": ["raw pair coordinates", "fixed randomized FWHT pair coordinates"],
            "decoder": ["additive per-leaf action rows", "learned balanced-initialized code merger then action rows"],
        },
        "dense_stem_present": False,
        "offline_compiler_present": False,
        "teacher_present": False,
        "hard_forward_during_training_and_evaluation": True,
        "route_credit": "zero-forward sigmoid STE weighted by local counterfactual action differences",
        "compiled_inference": "optional FWHT -> pair codes -> optional integer merger maps -> action rows",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "route_lr_multiplier": args.route_lr_multiplier,
        "route_temperature": args.route_temperature,
        "merger_init_logit": args.merger_init_logit,
        "optimizer": "AdamW(weight_decay=0)",
        "held_used_for_selection": False,
        "device": args.device,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "operation_ledger": operation_ledger(args),
        "timing_scope": "eager semantic reference; not an optimized-kernel claim",
    }
    result: dict[str, object] = {
        "schema": SCHEMA,
        "protocol": protocol,
        "rows": [asdict(row) for row in rows],
        "audits": audits,
        "summary": summarize(rows),
    }
    artifact = {"schema": SCHEMA, "protocol": protocol, "state": artifact_state}
    _save_exclusive(args.artifact, artifact)
    reloaded = torch.load(args.artifact, map_location="cpu", weights_only=False)
    exact = reloaded["schema"] == artifact["schema"] and reloaded["protocol"] == artifact["protocol"]
    exact = exact and reloaded["state"].keys() == artifact_state.keys()
    exact = exact and all(torch.equal(reloaded["state"][key], value) for key, value in artifact_state.items())
    if not exact:
        raise RuntimeError("artifact roundtrip failed")
    result["artifact"] = _path_metadata(args.artifact)
    result["artifact_roundtrip_exact"] = True
    _write_json_exclusive(args.output, result)
    return result


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item) for item in value.split(",") if item)
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--input-dim", type=int, default=784)
    parser.add_argument("--classes", type=int, default=47)
    parser.add_argument("--transform-dim", type=int, default=1024)
    parser.add_argument("--tables", type=int, default=32)
    parser.add_argument("--comparisons", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--route-lr-multiplier", type=float, default=1.0)
    parser.add_argument("--route-temperature", type=float, default=1.0)
    parser.add_argument("--merger-init-logit", type=float, default=0.005)
    parser.add_argument("--merger-initialization", choices=("balanced_random", "xor"), default="balanced_random")
    parser.add_argument("--initial-logit-std", type=float, default=0.02)
    parser.add_argument("--benchmark-batch-size", type=int, default=512)
    parser.add_argument("--benchmark-warmups", type=int, default=10)
    parser.add_argument("--benchmark-iters", type=int, default=30)
    parser.add_argument("--seeds", type=_parse_seeds, default=(0,))
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-test", type=int, default=0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.input_dim < 1 or args.classes < 2 or args.transform_dim < args.input_dim:
        parser.error("invalid input/classes/transform geometry")
    if args.transform_dim & (args.transform_dim - 1) or args.tables < 2 or args.tables % 2 or args.comparisons < 1:
        parser.error("transform_dim must be power-of-two and tables must be positive/even")
    if 2 * args.tables * args.comparisons > min(args.input_dim, args.transform_dim):
        parser.error("the disjoint-support experiment needs 2*tables*comparisons <= input_dim")
    if args.epochs < 1 or args.batch_size < 1 or min(args.lr, args.route_lr_multiplier, args.route_temperature) <= 0:
        parser.error("invalid optimization argument")
    if args.merger_init_logit <= 0 or args.initial_logit_std <= 0:
        parser.error("invalid initialization argument")
    if args.benchmark_batch_size < 1 or args.benchmark_warmups < 0 or args.benchmark_iters < 1:
        parser.error("invalid benchmark argument")
    if not args.root.is_dir():
        parser.error("EMNIST root does not exist")
    if args.output.resolve() == args.artifact.resolve() or args.output.exists() or args.artifact.exists():
        parser.error("output and artifact must be distinct nonexistent paths")
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
