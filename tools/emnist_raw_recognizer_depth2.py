"""Two-layer raw-pixel EMNIST recognizer comparison without a dense stem."""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from tropnn.tools.emnist_payload_dtype_sweep import _load_emnist_split
from tropnn.tools.emnist_raw_recognizer_factorial import (
    ARMS,
    RawGridHead,
    RawPairHead,
    RawPQHead,
    _parameter_rms_motion,
    _path_metadata,
    _save_exclusive,
    _write_json_exclusive,
    grid_initialization,
    make_balanced_pair_supports,
)

SCHEMA = "emnist-raw-recognizer-depth2-v1"


@dataclass(frozen=True)
class Evaluation:
    seed: int
    arm: str
    held_ce: float
    held_accuracy: float
    parameter_count: int
    route_parameter_count: int
    action_parameter_count: int
    layer1_mean_entropy_bits: float | None
    layer2_mean_entropy_bits: float | None
    layer1_mean_observed_rows: float | None
    layer2_mean_observed_rows: float | None
    hard_replay_max_error: float
    eager_hard_forward_ms: float


class LUTDepthTwo(nn.Module):
    """Two hard lookup layers separated by a common tanh."""

    def __init__(self, first: nn.Module, second: nn.Module) -> None:
        super().__init__()
        self.first = first
        self.second = second

    def forward(self, x: Tensor) -> Tensor:
        return self.second(torch.tanh(self.first(x)))

    def hard_output(self, x: Tensor) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        hidden, first_codes = self.first.hard_output(x)  # type: ignore[attr-defined]
        logits, second_codes = self.second.hard_output(torch.tanh(hidden))  # type: ignore[attr-defined]
        return logits, (first_codes, second_codes)

    def hard_codes(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return self.hard_output(x)[1]


class DenseDepthTwo(nn.Module):
    def __init__(self, input_dim: int, classes: int, seed: int) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.first = nn.Linear(input_dim, input_dim)
            self.second = nn.Linear(input_dim, classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.second(torch.tanh(self.first(x)))

    def hard_output(self, x: Tensor) -> tuple[Tensor, None]:
        return self(x), None


def _make_lut_head(
    arm: str,
    input_dim: int,
    output_dim: int,
    rows: Tensor,
    *,
    comparisons: int,
    seed: int,
    temperature: float,
) -> nn.Module:
    supports, thresholds, centroids = grid_initialization(input_dim)
    if arm == "pair":
        pair_supports = make_balanced_pair_supports(input_dim, comparisons, 100_000 + seed)
        return RawPairHead(input_dim, output_dim, pair_supports, rows, seed=200_000 + seed, temperature=temperature)
    if arm == "grid":
        return RawGridHead(
            input_dim,
            output_dim,
            supports=supports,
            thresholds=thresholds,
            rows=rows,
            bins=4,
            tie_break="nonnegative",
            surrogate="local_counterfactual",
            tau=temperature,
            trainable_thresholds=True,
            trainable_rows=True,
        )
    if arm == "pq":
        return RawPQHead(centroids, rows, temperature=temperature)
    raise ValueError(arm)


def make_models(
    input_dim: int,
    classes: int,
    *,
    tables: int,
    comparisons: int,
    seed: int,
    hidden_initial_std: float,
    logit_initial_std: float,
    temperature: float,
) -> dict[str, nn.Module]:
    generator = torch.Generator(device="cpu").manual_seed(300_000 + seed)
    first_rows = torch.randn(tables, 16, input_dim, generator=generator) * (hidden_initial_std / math.sqrt(tables))
    second_rows = torch.randn(tables, 16, classes, generator=generator) * (logit_initial_std / math.sqrt(tables))
    models: dict[str, nn.Module] = {}
    for index, arm in enumerate(ARMS[:-1]):
        first = _make_lut_head(
            arm,
            input_dim,
            input_dim,
            first_rows,
            comparisons=comparisons,
            seed=seed * 100 + index * 10,
            temperature=temperature,
        )
        second = _make_lut_head(
            arm,
            input_dim,
            classes,
            second_rows,
            comparisons=comparisons,
            seed=seed * 100 + index * 10 + 1,
            temperature=temperature,
        )
        models[arm] = LUTDepthTwo(first, second)
    models["dense"] = DenseDepthTwo(input_dim, classes, 400_000 + seed)
    return models


def route_and_action_parameters(model: nn.Module, arm: str) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    if arm == "dense":
        return [], list(model.parameters())
    route: list[nn.Parameter] = []
    action: list[nn.Parameter] = []
    for layer in (model.first, model.second):  # type: ignore[attr-defined]
        if arm == "pair":
            route.append(layer.thresholds)
            action.append(layer.rows)
        elif arm == "grid":
            route.append(layer.thresholds)
            action.append(layer.rows)
        else:
            route.append(layer.centroids)
            action.append(layer.rows)
    return route, action


@torch.no_grad()
def _codes_batched(model: LUTDepthTwo, x: Tensor, batch_size: int, device: torch.device) -> tuple[Tensor, Tensor]:
    first: list[Tensor] = []
    second: list[Tensor] = []
    model.eval()
    for start in range(0, len(x), batch_size):
        codes = model.hard_codes(x[start : start + batch_size].to(device))
        first.append(codes[0].cpu())
        second.append(codes[1].cpu())
    return torch.cat(first), torch.cat(second)


def _route_health(counts: Tensor, examples: int) -> tuple[float, float]:
    probabilities = counts.double() / examples
    entropy = -(probabilities * probabilities.clamp_min(1e-300).log2()).sum(dim=-1)
    return float(entropy.mean()), float((counts > 0).sum(dim=-1).double().mean())


def _benchmark(model: nn.Module, arm: str, x: Tensor, warmups: int, iterations: int, device: torch.device) -> float:
    def call() -> Tensor:
        return model(x) if arm == "dense" else model.hard_output(x)[0]  # type: ignore[attr-defined]

    for _ in range(warmups):
        call()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(iterations):
            call()
        end.record()
        torch.cuda.synchronize(device)
        return float(begin.elapsed_time(end) / iterations)
    started = time.perf_counter()
    for _ in range(iterations):
        call()
    return 1000 * (time.perf_counter() - started) / iterations


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
    counts = None if arm == "dense" else [torch.zeros(args.tables, 16, dtype=torch.int64) for _ in range(2)]
    for start in range(0, len(y), args.batch_size):
        batch = x[start : start + args.batch_size].to(device)
        target = y[start : start + args.batch_size].to(device)
        forward = model(batch)
        if arm == "dense":
            explicit, codes = forward, None
        else:
            explicit, codes = model.hard_output(batch)  # type: ignore[attr-defined]
        replay = max(replay, float((forward - explicit).abs().max()))
        loss_sum += float(F.cross_entropy(explicit, target, reduction="sum"))
        correct += int((explicit.argmax(-1) == target).sum())
        if counts is not None and codes is not None:
            offset = 16 * torch.arange(args.tables, device=device).view(1, -1)
            for layer in range(2):
                bincount = torch.bincount((codes[layer] + offset).flatten(), minlength=args.tables * 16)
                counts[layer] += bincount.reshape(args.tables, 16).cpu()
    route, action = route_and_action_parameters(model, arm)
    health = [(None, None), (None, None)] if counts is None else [_route_health(value, len(y)) for value in counts]
    sample = x[: min(args.benchmark_batch_size, len(x))].to(device)
    return Evaluation(
        seed,
        arm,
        loss_sum / len(y),
        correct / len(y),
        sum(parameter.numel() for parameter in model.parameters()),
        sum(parameter.numel() for parameter in route),
        sum(parameter.numel() for parameter in action),
        health[0][0],
        health[1][0],
        health[0][1],
        health[1][1],
        replay,
        _benchmark(model, arm, sample, args.benchmark_warmups, args.benchmark_iters, device),
    )


def train_seed(
    seed: int,
    args: argparse.Namespace,
    train_x: Tensor,
    train_y: Tensor,
    held_x: Tensor,
    held_y: Tensor,
) -> tuple[list[Evaluation], dict[str, object], dict[str, Tensor]]:
    device = torch.device(args.device)
    models = make_models(
        args.input_dim,
        args.classes,
        tables=args.tables,
        comparisons=args.pair_comparisons,
        seed=seed,
        hidden_initial_std=args.hidden_initial_std,
        logit_initial_std=args.logit_initial_std,
        temperature=args.temperature,
    )
    models = {arm: model.to(device) for arm, model in models.items()}
    diagnostic = train_x[: min(args.diagnostic_examples, len(train_x))]
    initial_codes = {
        arm: _codes_batched(models[arm], diagnostic, args.batch_size, device)  # type: ignore[arg-type]
        for arm in ARMS[:-1]
    }
    if not all(torch.equal(initial_codes["grid"][layer], initial_codes["pq"][layer]) for layer in range(2)):
        raise RuntimeError("initial two-layer Grid/PQ codes differ")
    common_rows_exact = all(
        torch.equal(
            getattr(getattr(models["pair"], layer), "rows"),
            getattr(getattr(models[arm], layer), "rows"),
        )
        for layer in ("first", "second")
        for arm in ("grid", "pq")
    )
    route_parameters: dict[str, list[nn.Parameter]] = {}
    action_parameters: dict[str, list[nn.Parameter]] = {}
    initial_route: dict[str, list[Tensor]] = {}
    for arm in ARMS:
        route, action = route_and_action_parameters(models[arm], arm)
        route_parameters[arm], action_parameters[arm] = route, action
        initial_route[arm] = [parameter.detach().cpu().clone() for parameter in route]
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for arm in ARMS for p in action_parameters[arm]], "lr": args.lr},
            {"params": [p for arm in ARMS for p in route_parameters[arm]], "lr": args.lr * args.route_lr_multiplier},
        ],
        weight_decay=0,
    )
    generator = torch.Generator(device="cpu").manual_seed(500_000 + seed)
    curves: list[dict[str, object]] = []
    first_gradients: dict[str, dict[str, list[float]]] = {}
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
            logits = {arm: models[arm](batch) for arm in ARMS}
            losses = {arm: F.cross_entropy(logits[arm], target) for arm in ARMS}
            sum(losses.values()).backward()
            if epoch == 1 and start == 0:
                first_gradients = {
                    arm: {
                        "route": [0.0 if p.grad is None else float(p.grad.norm()) for p in route_parameters[arm]],
                        "action": [0.0 if p.grad is None else float(p.grad.norm()) for p in action_parameters[arm]],
                    }
                    for arm in ARMS
                }
            optimizer.step()
            for arm in ARMS:
                loss_sum[arm] += float(losses[arm].detach()) * target.numel()
                correct[arm] += int((logits[arm].detach().argmax(-1) == target).sum())
        curve: dict[str, object] = {"epoch": epoch}
        for arm in ARMS:
            curve[arm] = {"train_ce": loss_sum[arm] / len(train_y), "train_accuracy": correct[arm] / len(train_y)}
        curves.append(curve)
        print(
            f"seed={seed} epoch={epoch}/{args.epochs} " + " ".join(f"{arm}:ce={curve[arm]['train_ce']:.6f}" for arm in ARMS),  # type: ignore[index]
            flush=True,
        )
    final_codes = {
        arm: _codes_batched(models[arm], diagnostic, args.batch_size, device)  # type: ignore[arg-type]
        for arm in ARMS[:-1]
    }
    flips = {arm: [float((initial_codes[arm][layer] != final_codes[arm][layer]).float().mean()) for layer in range(2)] for arm in ARMS[:-1]}
    evaluations = [evaluate(seed, arm, models[arm], held_x, held_y, args, device) for arm in ARMS]
    audit = {
        "seconds": time.perf_counter() - started,
        "training_curves": curves,
        "first_step_gradient_norms_by_layer": first_gradients,
        "route_parameter_rms_motion": {
            arm: _parameter_rms_motion(route_parameters[arm], initial_route[arm]) if route_parameters[arm] else 0.0 for arm in ARMS
        },
        "diagnostic_code_flip_fraction_by_layer": flips,
        "initial_grid_pq_codes_exact_both_layers": True,
        "common_initial_rows_exact_both_layers": common_rows_exact,
        "all_lut_route_parameter_gradients_nonzero": all(value > 0 for arm in ARMS[:-1] for value in first_gradients[arm]["route"]),
        "all_action_parameter_gradients_nonzero": all(value > 0 for arm in ARMS for value in first_gradients[arm]["action"]),
        "all_hard_replays_exact": all(row.hard_replay_max_error == 0 for row in evaluations),
        "all_finite": all(math.isfinite(row.held_ce) for row in evaluations),
    }
    state = {f"{arm}.{key}": value.detach().cpu() for arm, model in models.items() for key, value in model.state_dict().items()}
    return evaluations, audit, state


def summarize(rows: list[Evaluation]) -> dict[str, object]:
    arms = {}
    for arm in ARMS:
        selected = [row for row in rows if row.arm == arm]
        arms[arm] = {
            "held_ce_mean": sum(row.held_ce for row in selected) / len(selected),
            "held_ce_min": min(row.held_ce for row in selected),
            "held_ce_max": max(row.held_ce for row in selected),
            "held_accuracy_mean": sum(row.held_accuracy for row in selected) / len(selected),
            "eager_hard_forward_ms_mean": sum(row.eager_hard_forward_ms for row in selected) / len(selected),
        }
    return {"arms": arms, "ce_order_best_to_worst": sorted(ARMS, key=lambda arm: arms[arm]["held_ce_mean"])}


def operation_ledger(input_dim: int, classes: int, tables: int, comparisons: int) -> dict[str, dict[str, int]]:
    action_reads = tables * (input_dim + classes)
    return {
        "pair": {"threshold_comparisons": 2 * tables * comparisons, "active_row_scalar_reads": action_reads},
        "grid": {"threshold_comparisons": 2 * tables * 2 * 3, "active_row_scalar_reads": action_reads},
        "pq": {"squared_distance_terms": 2 * tables * 16 * 2, "active_row_scalar_reads": action_reads},
        "dense": {"multiply_accumulates": input_dim * input_dim + input_dim * classes},
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    train_x, train_y = _load_emnist_split(args.root, "balanced", train=True, limit=args.max_train, seed=0)
    held_x, held_y = _load_emnist_split(args.root, "balanced", train=False, limit=args.max_test, seed=0)
    if train_x.shape[1] != args.input_dim or args.tables * 2 != args.input_dim:
        raise ValueError("input geometry mismatch")
    rows: list[Evaluation] = []
    audits: dict[str, object] = {}
    state: dict[str, Tensor] = {}
    for seed in args.seeds:
        seed_rows, audit, seed_state = train_seed(seed, args, train_x, train_y, held_x, held_y)
        rows.extend(seed_rows)
        audits[str(seed)] = audit
        state.update({f"seed{seed}.{key}": value for key, value in seed_state.items()})
        print(f"seed={seed} " + " ".join(f"{row.arm}:ce={row.held_ce:.6f}" for row in seed_rows), flush=True)
    if not all(
        audit["initial_grid_pq_codes_exact_both_layers"]
        and audit["common_initial_rows_exact_both_layers"]
        and audit["all_lut_route_parameter_gradients_nonzero"]
        and audit["all_action_parameter_gradients_nonzero"]
        and audit["all_hard_replays_exact"]
        and audit["all_finite"]
        for audit in audits.values()  # type: ignore[union-attr]
    ):
        raise RuntimeError("two-layer semantic audit failed")
    protocol = {
        "dataset": "EMNIST Balanced",
        "data_root": str(args.root.resolve()),
        "train_examples": len(train_x),
        "held_examples": len(held_x),
        "seeds": list(args.seeds),
        "architecture": "raw D784 -> learned D784 -> tanh -> learned D47",
        "dense_stem_present": False,
        "depth": 2,
        "input_dim": args.input_dim,
        "hidden_dim": args.input_dim,
        "classes": args.classes,
        "tables_per_lut_layer": args.tables,
        "codes_per_table": 16,
        "pair_comparisons_per_table": args.pair_comparisons,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "route_lr_multiplier": args.route_lr_multiplier,
        "temperature": args.temperature,
        "hidden_initial_std": args.hidden_initial_std,
        "logit_initial_std": args.logit_initial_std,
        "optimizer": "AdamW(weight_decay=0)",
        "hard_forward": True,
        "held_used_for_selection": False,
        "operation_ledger": operation_ledger(args.input_dim, args.classes, args.tables, args.pair_comparisons),
        "timing_scope": "eager reference hard two-layer model; not optimized-kernel evidence",
        "device": args.device,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    result = {"schema": SCHEMA, "protocol": protocol, "rows": [asdict(row) for row in rows], "audits": audits, "summary": summarize(rows)}
    artifact = {"schema": SCHEMA, "protocol": protocol, "state": state}
    _save_exclusive(args.artifact, artifact)
    reloaded = torch.load(args.artifact, map_location="cpu", weights_only=False)
    exact = reloaded["schema"] == SCHEMA and reloaded["protocol"] == protocol and reloaded["state"].keys() == state.keys()
    exact = exact and all(torch.equal(reloaded["state"][key], value) for key, value in state.items())
    if not exact:
        raise RuntimeError("artifact roundtrip failed")
    result["artifact"] = _path_metadata(args.artifact)
    result["artifact_roundtrip_exact"] = True
    _write_json_exclusive(args.output, result)
    return result


def _parse_seeds(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item)
    if not result:
        raise argparse.ArgumentTypeError("at least one seed required")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--input-dim", type=int, default=784)
    parser.add_argument("--classes", type=int, default=47)
    parser.add_argument("--tables", type=int, default=392)
    parser.add_argument("--pair-comparisons", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--route-lr-multiplier", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--hidden-initial-std", type=float, default=0.5)
    parser.add_argument("--logit-initial-std", type=float, default=0.02)
    parser.add_argument("--diagnostic-examples", type=int, default=2048)
    parser.add_argument("--benchmark-batch-size", type=int, default=256)
    parser.add_argument("--benchmark-warmups", type=int, default=5)
    parser.add_argument("--benchmark-iters", type=int, default=20)
    parser.add_argument("--seeds", type=_parse_seeds, default=(0, 1, 2))
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-test", type=int, default=0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.input_dim, args.classes, args.tables, args.pair_comparisons) != (784, 47, 392, 4):
        parser.error("formal geometry is D784/O47/T392/C4/K16")
    if min(args.epochs, args.batch_size, args.diagnostic_examples, args.benchmark_batch_size, args.benchmark_iters) < 1:
        parser.error("counts must be positive")
    if min(args.lr, args.route_lr_multiplier, args.temperature, args.hidden_initial_std, args.logit_initial_std) <= 0:
        parser.error("optimization/initialization values must be positive")
    if args.output.resolve() == args.artifact.resolve() or args.output.exists() or args.artifact.exists():
        parser.error("output and artifact must be distinct nonexistent paths")
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
