"""Raw-pixel EMNIST recognizer comparison without a shared dense stem.

Every model consumes the same 784 normalized pixels directly.  Pair, product
grid, and exact-PQ arms share one additive ``T x K x classes`` action shape and
identical initial action rows.  Only their hard recognizers and route-gradient
surrogates differ.  The dense control is one direct ``784 -> classes`` linear
map.  No dense stem, frozen teacher, reconstruction compiler, or live action is
present in this experiment.
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

from tropnn.layers.hard_lookup import ProductGridLookupRouter, sum_lookup_rows
from tropnn.layers.pairwise import PairwiseLUT
from tropnn.tools.emnist_payload_dtype_sweep import _load_emnist_split

SCHEMA = "emnist-raw-recognizer-factorial-v1"
ARMS = ("pair", "grid", "pq", "dense")


@dataclass(frozen=True)
class Evaluation:
    seed: int
    arm: str
    held_ce: float
    held_accuracy: float
    parameter_count: int
    route_parameter_count: int
    action_parameter_count: int
    mean_entropy_bits: float | None
    minimum_entropy_bits: float | None
    mean_observed_rows: float | None
    maximum_row_mass: float | None
    hard_replay_max_error: float
    eager_hard_forward_ms: float


def make_balanced_pair_supports(input_dim: int, comparisons: int, seed: int) -> Tensor:
    """Make unique oriented pair predicates with exact coordinate coverage.

    Each comparison column is a perfect matching of all input coordinates, so
    every coordinate is read exactly once per column.  Pair identity is unique
    up to orientation across all columns.
    """

    if input_dim < 2 or input_dim % 2 or comparisons < 1:
        raise ValueError("input_dim must be positive and even; comparisons must be positive")
    tables = input_dim // 2
    generator = torch.Generator(device="cpu").manual_seed(seed)
    used: set[tuple[int, int]] = set()
    matchings: list[Tensor] = []
    for _comparison in range(comparisons):
        for _attempt in range(10_000):
            oriented = torch.randperm(input_dim, generator=generator).reshape(tables, 2)
            canonical = oriented.sort(dim=-1).values
            keys = [tuple(int(value) for value in pair) for pair in canonical.tolist()]
            if len(set(keys)) != tables or any(key in used for key in keys):
                continue
            used.update(keys)
            matchings.append(oriented)
            break
        else:
            raise RuntimeError("could not construct unique balanced pair supports")
    supports = torch.stack(matchings, dim=1)
    counts = torch.bincount(supports.flatten(), minlength=input_dim)
    if not torch.equal(counts, torch.full_like(counts, comparisons)):
        raise AssertionError("balanced pair supports lost exact coordinate coverage")
    return supports


def grid_initialization(input_dim: int) -> tuple[Tensor, Tensor, Tensor]:
    """Return disjoint D2 supports and matched 4x4 grid/PQ initialization."""

    if input_dim % 2:
        raise ValueError("input_dim must be divisible by two")
    tables = input_dim // 2
    supports = torch.arange(input_dim, dtype=torch.int64).reshape(tables, 2)
    levels = torch.tensor((-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0), dtype=torch.float32)
    thresholds_1d = (levels[:-1] + levels[1:]) * 0.5
    thresholds = thresholds_1d.view(1, 1, 3).expand(tables, 2, 3).clone()
    centroids = (
        torch.tensor(
            [(float(first), float(second)) for first in levels for second in levels],
            dtype=torch.float32,
        )
        .view(1, 16, 2)
        .expand(tables, 16, 2)
        .clone()
    )
    return supports, thresholds, centroids


class RawPairHead(nn.Module):
    """Classic flat PC-LUT recognizer and additive constant-row action."""

    def __init__(self, input_dim: int, classes: int, supports: Tensor, rows: Tensor, *, seed: int, temperature: float) -> None:
        super().__init__()
        tables, comparisons, endpoints = supports.shape
        if endpoints != 2 or rows.shape != (tables, 1 << comparisons, classes):
            raise ValueError("pair supports/rows have incompatible shapes")
        self.layer = PairwiseLUT(
            input_dim,
            classes,
            tables=tables,
            comparisons=comparisons,
            backend="torch",
            seed=seed,
            lut_init_std=0,
            use_min_margin_ste=True,
            use_output_scaling=False,
            fixed_zero_threshold=False,
            surrogate="fast_sigmoid_odd",
            anchor_policy="random_no_replace",
            anchor_seed=seed,
            lut_dtype="fp32",
        )
        del temperature
        with torch.no_grad():
            self.layer.anchors.copy_(supports)
            self.layer.thresholds.zero_()
            self.layer.lut.copy_(rows)

    @property
    def rows(self) -> nn.Parameter:
        return self.layer.lut

    @property
    def thresholds(self) -> nn.Parameter:
        return self.layer.thresholds  # type: ignore[return-value]

    @property
    def supports(self) -> Tensor:
        return self.layer.anchors

    def hard_codes(self, x: Tensor) -> Tensor:
        return self.layer.cache_index(x.unsqueeze(1)).indices.squeeze(1)

    def hard_output(self, x: Tensor) -> tuple[Tensor, Tensor]:
        codes = self.hard_codes(x)
        return sum_lookup_rows(self.rows, codes), codes

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x.unsqueeze(1)).squeeze(1)


class RawGridHead(ProductGridLookupRouter):
    """Parallel D2 four-bin grid with local counterfactual route credit."""

    def hard_output(self, x: Tensor) -> tuple[Tensor, Tensor]:
        codes = self.hard_codes(x)
        return sum_lookup_rows(self.rows, codes), codes

    def forward(self, x: Tensor) -> Tensor:
        if not self.training:
            return self.hard_output(x)[0]
        return super().forward(x)


class RawPQHead(nn.Module):
    """Exact D2/K16 nearest-centroid route with soft-PQ backward credit."""

    def __init__(self, centroids: Tensor, rows: Tensor, *, temperature: float) -> None:
        super().__init__()
        if centroids.ndim != 3 or rows.ndim != 3 or centroids.shape[:2] != rows.shape[:2]:
            raise ValueError("centroids and rows must be aligned [tables,codes,width]")
        if centroids.shape[-1] != 2 or temperature <= 0:
            raise ValueError("this experiment requires D2 blocks and positive temperature")
        self.centroids = nn.Parameter(centroids.detach().clone())
        self.rows = nn.Parameter(rows.detach().clone())
        self.temperature = float(temperature)

    @property
    def tables(self) -> int:
        return int(self.centroids.shape[0])

    def distances(self, x: Tensor) -> Tensor:
        local = x.reshape(x.shape[0], self.tables, 2)
        return (local.unsqueeze(-2) - self.centroids.unsqueeze(0)).square().sum(dim=-1)

    def hard_codes(self, x: Tensor) -> Tensor:
        return self.distances(x).argmin(dim=-1)

    def hard_output(self, x: Tensor) -> tuple[Tensor, Tensor]:
        codes = self.hard_codes(x)
        return sum_lookup_rows(self.rows, codes), codes

    def soft_output(self, x: Tensor) -> Tensor:
        probabilities = torch.softmax(-self.distances(x) / self.temperature, dim=-1)
        return torch.einsum("ntk,tko->no", probabilities, self.rows)

    def forward(self, x: Tensor) -> Tensor:
        hard, _codes = self.hard_output(x)
        soft = self.soft_output(x)
        return hard + (soft - soft.detach())


def make_models(
    input_dim: int,
    classes: int,
    *,
    comparisons: int,
    seed: int,
    row_init_std: float,
    temperature: float,
) -> dict[str, nn.Module]:
    tables = input_dim // 2
    supports, grid_thresholds, centroids = grid_initialization(input_dim)
    pair_supports = make_balanced_pair_supports(input_dim, comparisons, 100_000 + seed)
    generator = torch.Generator(device="cpu").manual_seed(200_000 + seed)
    common_rows = torch.randn(tables, 16, classes, generator=generator) * row_init_std
    pair = RawPairHead(input_dim, classes, pair_supports, common_rows, seed=300_000 + seed, temperature=temperature)
    grid = RawGridHead(
        input_dim,
        classes,
        supports=supports,
        thresholds=grid_thresholds,
        rows=common_rows,
        bins=4,
        tie_break="nonnegative",
        surrogate="local_counterfactual",
        tau=temperature,
        trainable_thresholds=True,
        trainable_rows=True,
    )
    pq = RawPQHead(centroids, common_rows, temperature=temperature)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(400_000 + seed)
        dense = nn.Linear(input_dim, classes)
    return {"pair": pair, "grid": grid, "pq": pq, "dense": dense}


def hard_output(model: nn.Module, arm: str, x: Tensor) -> tuple[Tensor, Tensor | None]:
    if arm == "dense":
        return model(x), None
    output, codes = model.hard_output(x)  # type: ignore[attr-defined]
    return output, codes


def route_and_action_parameters(model: nn.Module, arm: str) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    if arm == "pair":
        return [model.thresholds], [model.rows]  # type: ignore[attr-defined]
    if arm == "grid":
        return [model.thresholds], [model.rows]  # type: ignore[attr-defined]
    if arm == "pq":
        return [model.centroids], [model.rows]  # type: ignore[attr-defined]
    return [], list(model.parameters())


def _gradient_norm(parameters: list[nn.Parameter]) -> float:
    return math.sqrt(sum(float(parameter.grad.detach().square().sum()) for parameter in parameters if parameter.grad is not None))


def _parameter_rms_motion(parameters: list[nn.Parameter], initial: list[Tensor]) -> float:
    total, count = 0.0, 0
    for parameter, reference in zip(parameters, initial, strict=True):
        delta = parameter.detach().cpu() - reference
        total += float(delta.square().sum())
        count += delta.numel()
    return math.sqrt(total / max(count, 1))


def _codes_batched(model: nn.Module, x: Tensor, *, batch_size: int, device: torch.device) -> Tensor:
    pieces: list[Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            pieces.append(model.hard_codes(x[start : start + batch_size].to(device)).cpu())  # type: ignore[attr-defined]
    return torch.cat(pieces)


def train_seed(
    seed: int,
    args: argparse.Namespace,
    train_x: Tensor,
    train_y: Tensor,
    held_x: Tensor,
    held_y: Tensor,
) -> tuple[list[Evaluation], dict[str, object], dict[str, Tensor]]:
    device = torch.device(args.device)
    classes = int(max(int(train_y.max()), int(held_y.max())) + 1)
    row_init_std = args.initial_logit_std / math.sqrt(args.tables)
    models = make_models(
        args.input_dim,
        classes,
        comparisons=args.pair_comparisons,
        seed=seed,
        row_init_std=row_init_std,
        temperature=args.temperature,
    )
    models = {arm: model.to(device) for arm, model in models.items()}
    common_initial_lut_rows_exact = bool(
        torch.equal(models["pair"].rows.detach().cpu(), models["grid"].rows.detach().cpu())  # type: ignore[attr-defined]
        and torch.equal(models["grid"].rows.detach().cpu(), models["pq"].rows.detach().cpu())  # type: ignore[attr-defined]
    )
    diagnostic_x = train_x[: min(args.diagnostic_examples, len(train_x))]
    initial_codes = {arm: _codes_batched(models[arm], diagnostic_x, batch_size=args.batch_size, device=device) for arm in ARMS[:-1]}
    if not torch.equal(initial_codes["grid"], initial_codes["pq"]):
        raise RuntimeError("matched initial grid/PQ codes differ")

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
    generator = torch.Generator(device="cpu").manual_seed(500_000 + seed)
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
            optimizer.step()
            count = target.numel()
            for arm in ARMS:
                loss_sum[arm] += float(losses[arm].detach()) * count
                correct[arm] += int((logits[arm].detach().argmax(-1) == target).sum())
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

    final_codes = {arm: _codes_batched(models[arm], diagnostic_x, batch_size=args.batch_size, device=device) for arm in ARMS[:-1]}
    route_motion = {arm: _parameter_rms_motion(route_parameters[arm], initial_route[arm]) if route_parameters[arm] else 0.0 for arm in ARMS}
    code_flip = {arm: float((initial_codes[arm] != final_codes[arm]).float().mean()) for arm in ARMS[:-1]}
    evaluations = [evaluate(seed, arm, models[arm], held_x, held_y, args, device) for arm in ARMS]
    state = {f"{arm}.{key}": value.detach().cpu() for arm, model in models.items() for key, value in model.state_dict().items()}
    audit: dict[str, object] = {
        "seconds": time.perf_counter() - started,
        "training_curves": curves,
        "first_step_gradient_norms": first_gradients,
        "route_parameter_rms_motion": route_motion,
        "diagnostic_train_code_flip_fraction": code_flip,
        "initial_grid_pq_codes_exact": True,
        "common_initial_lut_rows_exact": common_initial_lut_rows_exact,
        "all_route_gradients_nonzero": all(first_gradients[arm]["route"] > 0 for arm in ARMS[:-1]),
        "all_action_gradients_nonzero": all(first_gradients[arm]["action"] > 0 for arm in ARMS),
        "all_hard_replays_exact": all(row.hard_replay_max_error == 0 for row in evaluations),
        "all_finite": all(math.isfinite(row.held_ce) for row in evaluations),
    }
    return evaluations, audit, state


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
    loss_sum, correct, replay = 0.0, 0, 0.0
    route_counts = None if arm == "dense" else torch.zeros(args.tables, 16, dtype=torch.int64)
    for start in range(0, len(y), args.batch_size):
        batch = x[start : start + args.batch_size].to(device)
        target = y[start : start + args.batch_size].to(device)
        forward = model(batch)
        explicit, codes = hard_output(model, arm, batch)
        replay = max(replay, float((forward - explicit).abs().max()))
        loss_sum += float(F.cross_entropy(explicit, target, reduction="sum"))
        correct += int((explicit.argmax(-1) == target).sum())
        if codes is not None and route_counts is not None:
            offset = 16 * torch.arange(args.tables, device=device).view(1, -1)
            counts = torch.bincount((codes + offset).flatten(), minlength=args.tables * 16)
            route_counts += counts.reshape(args.tables, 16).cpu()
    route, action = route_and_action_parameters(model, arm)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    entropy = minimum = observed = maximum = None
    if route_counts is not None:
        probabilities = route_counts.double() / len(y)
        per_table_entropy = -(probabilities * probabilities.clamp_min(1e-300).log2()).sum(dim=-1)
        entropy = float(per_table_entropy.mean())
        minimum = float(per_table_entropy.min())
        observed = float((route_counts > 0).sum(dim=-1).double().mean())
        maximum = float(probabilities.max())
    sample = x[: min(args.benchmark_batch_size, len(x))].to(device)
    elapsed_ms = benchmark_hard_forward(model, arm, sample, args.benchmark_warmups, args.benchmark_iters, device)
    return Evaluation(
        seed=seed,
        arm=arm,
        held_ce=loss_sum / len(y),
        held_accuracy=correct / len(y),
        parameter_count=parameter_count,
        route_parameter_count=sum(parameter.numel() for parameter in route),
        action_parameter_count=sum(parameter.numel() for parameter in action),
        mean_entropy_bits=entropy,
        minimum_entropy_bits=minimum,
        mean_observed_rows=observed,
        maximum_row_mass=maximum,
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


def operation_ledger(input_dim: int, classes: int, tables: int, comparisons: int) -> dict[str, dict[str, int]]:
    active_scalars = tables * classes
    return {
        "pair": {
            "coordinate_reads": 2 * tables * comparisons,
            "pair_differences": tables * comparisons,
            "threshold_comparisons": tables * comparisons,
            "active_row_scalar_reads": active_scalars,
            "active_row_accumulation_adds": (tables - 1) * classes,
        },
        "grid": {
            "coordinate_reads_if_reused_within_axis": input_dim,
            "threshold_comparisons": tables * 2 * 3,
            "active_row_scalar_reads": active_scalars,
            "active_row_accumulation_adds": (tables - 1) * classes,
        },
        "pq": {
            "coordinate_reads_excluding_centroids": input_dim,
            "centroid_scalar_reads": tables * 16 * 2,
            "squared_distance_terms": tables * 16 * 2,
            "distance_component_adds": tables * 16,
            "argmin_comparisons": tables * 15,
            "active_row_scalar_reads": active_scalars,
            "active_row_accumulation_adds": (tables - 1) * classes,
        },
        "dense": {
            "weight_scalar_reads": input_dim * classes,
            "multiply_accumulates": input_dim * classes,
            "bias_reads": classes,
        },
    }


def summarize(rows: list[Evaluation]) -> dict[str, object]:
    arms: dict[str, dict[str, float]] = {}
    for arm in ARMS:
        selected = [row for row in rows if row.arm == arm]
        arms[arm] = {
            "held_ce_mean": sum(row.held_ce for row in selected) / len(selected),
            "held_ce_min": min(row.held_ce for row in selected),
            "held_ce_max": max(row.held_ce for row in selected),
            "held_accuracy_mean": sum(row.held_accuracy for row in selected) / len(selected),
            "eager_hard_forward_ms_mean": sum(row.eager_hard_forward_ms for row in selected) / len(selected),
        }
    return {
        "arms": arms,
        "ce_order_best_to_worst": sorted(ARMS, key=lambda arm: arms[arm]["held_ce_mean"]),
        "effects": {
            "pair_minus_grid_ce": arms["pair"]["held_ce_mean"] - arms["grid"]["held_ce_mean"],
            "grid_minus_pq_ce": arms["grid"]["held_ce_mean"] - arms["pq"]["held_ce_mean"],
            "dense_minus_pq_ce": arms["dense"]["held_ce_mean"] - arms["pq"]["held_ce_mean"],
        },
    }


def _path_metadata(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _save_exclusive(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        torch.save(value, handle)


def _write_json_exclusive(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def run(args: argparse.Namespace) -> dict[str, object]:
    train_x, train_y = _load_emnist_split(args.root, "balanced", train=True, limit=args.max_train, seed=0)
    held_x, held_y = _load_emnist_split(args.root, "balanced", train=False, limit=args.max_test, seed=0)
    if train_x.shape[1] != args.input_dim or args.tables * 2 != args.input_dim:
        raise ValueError("raw input and T x D2 product geometry do not match")
    if int(max(int(train_y.max()), int(held_y.max())) + 1) != args.classes:
        raise ValueError("class count differs from frozen protocol")
    rows: list[Evaluation] = []
    audits: dict[str, object] = {}
    artifact_state: dict[str, Tensor] = {}
    for seed in args.seeds:
        seed_rows, audit, state = train_seed(seed, args, train_x, train_y, held_x, held_y)
        rows.extend(seed_rows)
        audits[str(seed)] = audit
        artifact_state.update({f"seed{seed}.{key}": value for key, value in state.items()})
        print(
            f"seed={seed} " + " ".join(f"{row.arm}:ce={row.held_ce:.6f},acc={row.held_accuracy:.6f}" for row in seed_rows),
            flush=True,
        )
    if not all(
        bool(audit["initial_grid_pq_codes_exact"])
        and bool(audit["common_initial_lut_rows_exact"])
        and bool(audit["all_route_gradients_nonzero"])
        and bool(audit["all_action_gradients_nonzero"])
        and bool(audit["all_hard_replays_exact"])
        and bool(audit["all_finite"])
        for audit in audits.values()  # type: ignore[union-attr]
    ):
        raise RuntimeError("training or semantic audit failed")
    ledger = operation_ledger(args.input_dim, args.classes, args.tables, args.pair_comparisons)
    protocol = {
        "dataset": "EMNIST Balanced",
        "data_root": str(args.root.resolve()),
        "train_examples": len(train_x),
        "held_examples": len(held_x),
        "seeds": list(args.seeds),
        "input_dim": args.input_dim,
        "classes": args.classes,
        "dense_stem_present": False,
        "all_models_read_raw_pixels_directly": True,
        "tables": args.tables,
        "codes_per_table": 16,
        "block_width_grid_pq": 2,
        "pair_comparisons_per_table": args.pair_comparisons,
        "pair_grid_pq_action_shape": [args.tables, 16, args.classes],
        "pair_grid_pq_initial_rows_exactly_shared": True,
        "grid_pq_initial_codes_exactly_matched": True,
        "initial_grid_levels": [-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0],
        "pair_initial_thresholds": 0.0,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "route_lr_multiplier": args.route_lr_multiplier,
        "temperature": args.temperature,
        "initial_logit_std": args.initial_logit_std,
        "optimizer": "AdamW(weight_decay=0)",
        "no_offline_reconstruction_compiler": True,
        "no_frozen_teacher": True,
        "no_live_action": True,
        "held_used_for_selection": False,
        "hard_forward_during_training_and_evaluation": True,
        "pair_route_surrogate": "classic min-margin local-counterfactual PC-LUT STE",
        "grid_route_surrogate": "nearest active wall local-counterfactual",
        "pq_route_surrogate": "LUT-NN soft-PQ",
        "device": args.device,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "operation_ledger": ledger,
        "timing_scope": "eager reference hard head only; not optimized-kernel evidence",
    }
    result = {
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
    parser.add_argument("--tables", type=int, default=392)
    parser.add_argument("--pair-comparisons", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--route-lr-multiplier", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--initial-logit-std", type=float, default=0.02)
    parser.add_argument("--diagnostic-examples", type=int, default=8192)
    parser.add_argument("--benchmark-batch-size", type=int, default=512)
    parser.add_argument("--benchmark-warmups", type=int, default=10)
    parser.add_argument("--benchmark-iters", type=int, default=30)
    parser.add_argument("--seeds", type=_parse_seeds, default=(0, 1, 2))
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-test", type=int, default=0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.input_dim, args.classes, args.tables, args.pair_comparisons) != (784, 47, 392, 4):
        parser.error("formal geometry is fixed to raw D784, O47, T392, C4/K16")
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.route_lr_multiplier <= 0 or args.temperature <= 0:
        parser.error("invalid optimization argument")
    if args.initial_logit_std <= 0 or args.diagnostic_examples < 1:
        parser.error("invalid initialization/diagnostic argument")
    if args.benchmark_batch_size < 1 or args.benchmark_warmups < 0 or args.benchmark_iters < 1:
        parser.error("invalid benchmark argument")
    if args.output.resolve() == args.artifact.resolve() or args.output.exists() or args.artifact.exists():
        parser.error("output and artifact must be distinct nonexistent paths")
    if not args.root.is_dir():
        parser.error("EMNIST root does not exist")
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
