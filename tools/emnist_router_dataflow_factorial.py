"""Compare flat/adaptive and pair/unary hard-router data flows on EMNIST.

The four hard arms form a literal 2x2 factorial:

* flat Pair: classical PC-LUT routing;
* flat unary: random-fern routing;
* adaptive Pair: a relational decision tree;
* adaptive unary: a MADDNESS-style decision tree.

Only the predicate and route topology change.  Every arm uses the same dense
stem, four D64 residual hard layers, one hard D64->D47 head, T32, depth/C4,
16 rows per table, random fixed supports, zero thresholds, random payloads,
and final-task EMNIST cross entropy.  There is no route calibration, offline
action compiler, or learned discrete index.
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from tropnn.layers.hard_lookup import HardLookupRouter, Predicate, Topology
from tropnn.tools.emnist_payload_dtype_sweep import _load_emnist_split
from tropnn.tools.maddness_end_to_end_ste_factorial import _leaf_stats

HARD_ARMS = ("flat_pair", "flat_unary", "adaptive_pair", "adaptive_unary")
ARMS = (*HARD_ARMS, "dense_l4")


@dataclass(frozen=True)
class ArmEvaluation:
    arm: str
    held_ce: float
    held_accuracy: float
    trainable_parameters: int
    hard_forward_exact: bool | None
    route_entropy_bits_mean: float | None
    route_entropy_bits_minimum: float | None
    maximum_row_mass: float | None
    observed_rows_mean: float | None
    threshold_rms: float | None


@dataclass(frozen=True)
class SeedResult:
    seed: int
    initialization_checks: dict[str, bool]
    curves: dict[str, list[dict[str, float]]]
    first_step_gradient_norms: dict[str, dict[str, float]]
    evaluations: list[ArmEvaluation]
    seconds: float


def sample_supports(tables: int, depth: int, input_dim: int, seed: int) -> Tensor:
    """Sample paired coordinates; unary arms use the first endpoint only."""

    generator = torch.Generator(device="cpu").manual_seed(seed)
    first = torch.randint(input_dim, (tables, depth), generator=generator)
    offset = torch.randint(1, input_dim, (tables, depth), generator=generator)
    second = (first + offset).remainder(input_dim)
    return torch.stack((first, second), dim=-1)


def _make_router(
    input_dim: int,
    output_dim: int,
    *,
    tables: int,
    depth: int,
    predicate: Predicate,
    topology: Topology,
    route_seed: int,
    row_seed: int,
    tau: float,
    prototype_std: float,
) -> HardLookupRouter:
    """Create matched experimental tensors, then hand them to the shared core."""

    supports = sample_supports(tables, depth, input_dim, route_seed)
    threshold_count = depth if topology == "flat" else 2**depth - 1
    thresholds = torch.zeros(tables, threshold_count)
    generator = torch.Generator(device="cpu").manual_seed(row_seed)
    rows = torch.randn(tables, 2**depth, output_dim, generator=generator) * prototype_std
    return HardLookupRouter(
        input_dim,
        output_dim,
        depth=depth,
        predicate=predicate,
        topology=topology,
        supports=supports,
        thresholds=thresholds,
        rows=rows,
        surrogate="local_counterfactual",
        tau=tau,
    )


class RouterStackClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        classes: int,
        *,
        hidden_layers: int,
        tables: int,
        depth: int,
        predicate: Predicate,
        topology: Topology,
        seed: int,
        residual_scale: float,
        tau: float,
        prototype_std: float,
    ) -> None:
        super().__init__()
        generator_state = torch.random.get_rng_state()
        torch.manual_seed(seed)
        self.stem = nn.Linear(input_dim, state_dim)
        torch.random.set_rng_state(generator_state)
        self.residual_scale = float(residual_scale)
        self.blocks = nn.ModuleList(
            [
                _make_router(
                    state_dim,
                    state_dim,
                    tables=tables,
                    depth=depth,
                    predicate=predicate,
                    topology=topology,
                    route_seed=seed + 1009 * (layer + 1),
                    row_seed=seed + 2003 * (layer + 1),
                    tau=tau,
                    prototype_std=prototype_std,
                )
                for layer in range(hidden_layers)
            ]
        )
        self.head = _make_router(
            state_dim,
            classes,
            tables=tables,
            depth=depth,
            predicate=predicate,
            topology=topology,
            route_seed=seed + 1009 * (hidden_layers + 1),
            row_seed=seed + 2003 * (hidden_layers + 1),
            tau=tau,
            prototype_std=prototype_std,
        )

    def router_layers(self) -> list[HardLookupRouter]:
        return [*self.blocks, self.head]

    def initial_state(self, x: Tensor) -> Tensor:
        return torch.tanh(self.stem(x.flatten(1)))

    def forward(self, x: Tensor) -> Tensor:
        state = self.initial_state(x)
        for block in self.blocks:
            state = (state + self.residual_scale * block(state)).clamp(-1.0, 1.0)
        return self.head(state)

    def hard_forward_with_trace(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        state = self.initial_state(x)
        codes: list[Tensor] = []
        for block in self.blocks:
            delta, code = block.hard_output(state)
            codes.append(code)
            state = (state + self.residual_scale * delta).clamp(-1.0, 1.0)
        logits, code = self.head.hard_output(state)
        codes.append(code)
        return logits, codes


class DenseStackClassifier(nn.Module):
    def __init__(self, input_dim: int, state_dim: int, classes: int, hidden_layers: int, seed: int, residual_scale: float) -> None:
        super().__init__()
        generator_state = torch.random.get_rng_state()
        torch.manual_seed(seed)
        self.stem = nn.Linear(input_dim, state_dim)
        self.blocks = nn.ModuleList([nn.Linear(state_dim, state_dim) for _ in range(hidden_layers)])
        self.head = nn.Linear(state_dim, classes)
        torch.random.set_rng_state(generator_state)
        self.residual_scale = float(residual_scale)

    def forward(self, x: Tensor) -> Tensor:
        state = torch.tanh(self.stem(x.flatten(1)))
        for block in self.blocks:
            state = (state + self.residual_scale * block(state)).clamp(-1.0, 1.0)
        return self.head(state)


def _arm_spec(arm: str) -> tuple[Topology, Predicate]:
    topology, predicate = arm.split("_", maxsplit=1)
    return topology, predicate  # type: ignore[return-value]


def paired_models(args: argparse.Namespace, classes: int, seed: int) -> tuple[dict[str, nn.Module], dict[str, bool]]:
    models: dict[str, nn.Module] = {}
    common = dict(
        input_dim=784,
        state_dim=args.state_dim,
        classes=classes,
        hidden_layers=4,
        tables=args.tables,
        depth=args.depth,
        seed=seed,
        residual_scale=args.residual_scale,
        tau=args.tau,
        prototype_std=args.prototype_std,
    )
    for arm in HARD_ARMS:
        topology, predicate = _arm_spec(arm)
        models[arm] = RouterStackClassifier(predicate=predicate, topology=topology, **common)
    dense = DenseStackClassifier(784, args.state_dim, classes, 4, seed, args.residual_scale)
    dense.stem.load_state_dict(copy.deepcopy(models["flat_pair"].stem.state_dict()))  # type: ignore[attr-defined]
    models["dense_l4"] = dense

    checks: dict[str, bool] = {}
    for predicate in ("pair", "unary"):
        flat = models[f"flat_{predicate}"]
        adaptive = models[f"adaptive_{predicate}"]
        assert isinstance(flat, RouterStackClassifier) and isinstance(adaptive, RouterStackClassifier)
        checks[f"{predicate}_stem_exact"] = all(
            torch.equal(a, b) for a, b in zip(flat.stem.state_dict().values(), adaptive.stem.state_dict().values(), strict=True)
        )
        checks[f"{predicate}_supports_rows_exact"] = all(
            torch.equal(a.supports, b.supports) and torch.equal(a.rows, b.rows)
            for a, b in zip(flat.router_layers(), adaptive.router_layers(), strict=True)
        )
        generator = torch.Generator(device="cpu").manual_seed(90_000 + seed)
        probe = torch.randn(32, 1, 28, 28, generator=generator)
        with torch.no_grad():
            flat_output, flat_codes = flat.hard_forward_with_trace(probe)
            adaptive_output, adaptive_codes = adaptive.hard_forward_with_trace(probe)
        checks[f"{predicate}_initial_codes_exact"] = all(torch.equal(a, b) for a, b in zip(flat_codes, adaptive_codes, strict=True))
        checks[f"{predicate}_initial_output_exact"] = torch.equal(flat_output, adaptive_output)
    if not all(checks.values()):
        raise AssertionError(f"matched initialization failed: {checks}")
    return models, checks


def _gradient_norm(parameters: list[Tensor]) -> float:
    return math.sqrt(sum(float(parameter.grad.detach().square().sum()) for parameter in parameters if parameter.grad is not None))


def _gradient_summary(model: nn.Module) -> dict[str, float]:
    if isinstance(model, DenseStackClassifier):
        return {"all": _gradient_norm(list(model.parameters()))}
    assert isinstance(model, RouterStackClassifier)
    layers = model.router_layers()
    return {
        "stem": _gradient_norm(list(model.stem.parameters())),
        "thresholds": _gradient_norm([layer.thresholds for layer in layers]),
        "rows": _gradient_norm([layer.rows for layer in layers]),
        "all": _gradient_norm(list(model.parameters())),
    }


def _loader(x: Tensor, y: Tensor, batch_size: int, workers: int, *, shuffle: bool, seed: int, pin: bool) -> DataLoader:
    generator = torch.Generator(device="cpu").manual_seed(seed) if shuffle else None
    return DataLoader(
        TensorDataset(x, y),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=workers,
        pin_memory=pin,
    )


def train_seed(
    models: dict[str, nn.Module],
    train_x: Tensor,
    train_y: Tensor,
    *,
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, list[dict[str, float]]], dict[str, dict[str, float]]]:
    for model in models.values():
        model.to(device).train()
    optimizers = {name: torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0) for name, model in models.items()}
    curves = {name: [] for name in models}
    first_gradients: dict[str, dict[str, float]] = {}
    for epoch in range(args.epochs):
        loader = _loader(
            train_x,
            train_y,
            args.batch_size,
            args.workers,
            shuffle=True,
            seed=410_000 + seed * 100 + epoch,
            pin=device.type == "cuda",
        )
        sums = {name: 0.0 for name in models}
        correct = {name: 0 for name in models}
        count = 0
        for batch_index, (x, target) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            for optimizer in optimizers.values():
                optimizer.zero_grad(set_to_none=True)
            logits = {name: model(x) for name, model in models.items()}
            losses = {name: F.cross_entropy(value, target) for name, value in logits.items()}
            sum(losses.values()).backward()
            if epoch == 0 and batch_index == 0:
                first_gradients = {name: _gradient_summary(model) for name, model in models.items()}
            for optimizer in optimizers.values():
                optimizer.step()
            batch = target.numel()
            for name, loss in losses.items():
                sums[name] += float(loss.detach()) * batch
                correct[name] += int((logits[name].detach().argmax(-1) == target).sum())
            count += batch
        for name in models:
            curves[name].append(
                {
                    "epoch": float(epoch + 1),
                    "train_ce": sums[name] / count,
                    "train_accuracy": correct[name] / count,
                }
            )
        print(
            f"seed={seed} epoch={epoch + 1}/{args.epochs} " + " ".join(f"{name}:ce={curves[name][-1]['train_ce']:.6f}" for name in ARMS),
            flush=True,
        )
    return curves, first_gradients


@torch.no_grad()
def evaluate_plain(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
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
def evaluate_router(arm: str, model: RouterStackClassifier, loader: DataLoader, device: torch.device) -> ArmEvaluation:
    ce, accuracy = evaluate_plain(model, loader, device)
    layers = model.router_layers()
    all_codes: list[list[Tensor]] = [[] for _ in layers]
    exact = True
    for x, _target in loader:
        x = x.to(device, non_blocking=True)
        hard, codes = model.hard_forward_with_trace(x)
        exact = exact and torch.equal(model(x), hard)
        for layer, code in enumerate(codes):
            all_codes[layer].append(code.cpu())
    entropies: list[float] = []
    minima: list[float] = []
    observed: list[float] = []
    maximum_mass = 0.0
    for chunks in all_codes:
        codes = torch.cat(chunks)
        entropy, minimum, maximum = _leaf_stats(codes)
        entropies.append(entropy)
        minima.append(minimum)
        maximum_mass = max(maximum_mass, maximum)
        for table in range(codes.shape[1]):
            observed.append(float((torch.bincount(codes[:, table], minlength=16) > 0).sum()))
    thresholds = torch.cat([layer.thresholds.detach().cpu().reshape(-1) for layer in layers])
    return ArmEvaluation(
        arm=arm,
        held_ce=ce,
        held_accuracy=accuracy,
        trainable_parameters=sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        hard_forward_exact=exact,
        route_entropy_bits_mean=sum(entropies) / len(entropies),
        route_entropy_bits_minimum=min(minima),
        maximum_row_mass=maximum_mass,
        observed_rows_mean=sum(observed) / len(observed),
        threshold_rms=float(thresholds.square().mean().sqrt()),
    )


def fit_seed(
    seed: int,
    args: argparse.Namespace,
    train_x: Tensor,
    train_y: Tensor,
    held_loader: DataLoader,
    classes: int,
) -> tuple[SeedResult, dict[str, Tensor]]:
    started = time.perf_counter()
    device = torch.device(args.device)
    models, checks = paired_models(args, classes, seed)
    curves, gradients = train_seed(models, train_x, train_y, args=args, seed=seed, device=device)
    evaluations: list[ArmEvaluation] = []
    for arm in ARMS:
        model = models[arm]
        if isinstance(model, RouterStackClassifier):
            evaluations.append(evaluate_router(arm, model, held_loader, device))
        else:
            ce, accuracy = evaluate_plain(model, held_loader, device)
            evaluations.append(
                ArmEvaluation(
                    arm,
                    ce,
                    accuracy,
                    sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
            )
    state = {f"{arm}.{key}": value.detach().cpu() for arm, model in models.items() for key, value in model.state_dict().items()}
    return SeedResult(seed, checks, curves, gradients, evaluations, time.perf_counter() - started), state


def summarize(rows: list[SeedResult]) -> dict[str, object]:
    arms: dict[str, dict[str, float]] = {}
    for arm in ARMS:
        selected = [next(item for item in row.evaluations if item.arm == arm) for row in rows]
        arms[arm] = {
            "held_ce_mean": sum(item.held_ce for item in selected) / len(selected),
            "held_accuracy_mean": sum(item.held_accuracy for item in selected) / len(selected),
        }
    per_seed: dict[str, dict[str, float]] = {}
    pair_effects: list[float] = []
    adaptive_effects: list[float] = []
    interactions: list[float] = []
    for row in rows:
        ce = {item.arm: item.held_ce for item in row.evaluations}
        pair_flat = ce["flat_unary"] - ce["flat_pair"]
        pair_adaptive = ce["adaptive_unary"] - ce["adaptive_pair"]
        adaptive_pair = ce["flat_pair"] - ce["adaptive_pair"]
        adaptive_unary = ce["flat_unary"] - ce["adaptive_unary"]
        interaction = adaptive_pair - adaptive_unary
        pair_effects.extend((pair_flat, pair_adaptive))
        adaptive_effects.extend((adaptive_pair, adaptive_unary))
        interactions.append(interaction)
        per_seed[str(row.seed)] = {
            "pair_gain_under_flat_ce": pair_flat,
            "pair_gain_under_adaptive_ce": pair_adaptive,
            "adaptive_gain_under_pair_ce": adaptive_pair,
            "adaptive_gain_under_unary_ce": adaptive_unary,
            "interaction_ce": interaction,
        }
    pair_mean = sum(pair_effects) / len(pair_effects)
    adaptive_mean = sum(adaptive_effects) / len(adaptive_effects)
    interaction_mean = sum(interactions) / len(interactions)
    return {
        "arms": arms,
        "effects_per_seed": per_seed,
        "effects": {
            "pair_predicate_grand_mean_ce_gain": pair_mean,
            "adaptive_topology_grand_mean_ce_gain": adaptive_mean,
            "interaction_mean_ce": interaction_mean,
        },
        "decisions": {
            "pair_predicate_material_gain": pair_mean >= 0.01 and min(pair_effects) > 0,
            "adaptive_topology_material_gain": adaptive_mean >= 0.01 and min(adaptive_effects) > 0,
            "positive_interaction": interaction_mean >= 0.01,
            "all_hard_forwards_exact": all(
                item.hard_forward_exact is True for row in rows for item in row.evaluations if item.hard_forward_exact is not None
            ),
            "all_initialization_checks_exact": all(all(row.initialization_checks.values()) for row in rows),
            "all_router_gradients_nonzero": all(
                row.first_step_gradient_norms[arm][field] > 0 for row in rows for arm in HARD_ARMS for field in ("stem", "thresholds", "rows")
            ),
        },
    }


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, object]:
    root = Path(args.root).expanduser()
    train_x, train_y = _load_emnist_split(root, "balanced", train=True, limit=args.max_train, seed=0)
    held_x, held_y = _load_emnist_split(root, "balanced", train=False, limit=args.max_test, seed=0)
    classes = int(max(int(train_y.max()), int(held_y.max())) + 1)
    held_loader = _loader(
        held_x,
        held_y,
        args.batch_size,
        args.workers,
        shuffle=False,
        seed=0,
        pin=torch.device(args.device).type == "cuda",
    )
    rows: list[SeedResult] = []
    states: dict[str, Tensor] = {}
    for seed in args.seeds:
        row, state = fit_seed(seed, args, train_x, train_y, held_loader, classes)
        rows.append(row)
        states.update({f"seed{seed}.{key}": value for key, value in state.items()})
        print(
            f"held seed={seed} " + " ".join(f"{item.arm}:ce={item.held_ce:.6f},acc={item.held_accuracy:.6f}" for item in row.evaluations),
            flush=True,
        )
    protocol = {
        "dataset": "EMNIST Balanced",
        "train_examples": len(train_x),
        "held_examples": len(held_x),
        "state_dim": args.state_dim,
        "tables": args.tables,
        "depth_or_bits": args.depth,
        "rows_per_table": 2**args.depth,
        "hidden_layers": 4,
        "active_comparisons_per_hard_layer": args.tables * args.depth,
        "active_row_lookups_per_hard_layer": args.tables,
        "unary_coordinate_reads_per_hard_layer": args.tables * args.depth,
        "pair_coordinate_reads_per_hard_layer": 2 * args.tables * args.depth,
        "pair_subtractions_per_hard_layer": args.tables * args.depth,
        "flat_thresholds_per_hard_layer": args.tables * args.depth,
        "adaptive_thresholds_per_hard_layer": args.tables * (2**args.depth - 1),
        "route_calibration_used": False,
        "offline_action_compiler_used": False,
        "discrete_indices_learned": False,
        "support_initialization": "shared_random_fixed_pairs_unary_uses_first_endpoint",
        "threshold_initialization": "all_zero_flat_tree_exact_within_predicate",
        "payload_initialization": "shared_random_normal_within_all_four_arms",
        "state_initialization": "shared_random_dense_stem_then_tanh",
        "hidden_composition": "clamp(state + 0.25 * hard_row_sum, -1, 1)",
        "route_backward": "nearest_executed_wall_local_counterfactual",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "tau": args.tau,
        "prototype_std": args.prototype_std,
        "seeds": list(args.seeds),
        "held_not_used_for_selection_or_early_stopping": True,
        "device": args.device,
        "torch_version": str(torch.__version__),
        "cuda_version": None if torch.version.cuda is None else str(torch.version.cuda),
    }
    result = {
        "schema": "emnist-router-dataflow-factorial-v1",
        "protocol": protocol,
        "seeds": [asdict(row) for row in rows],
        "summary": summarize(rows),
    }
    artifact = Path(args.artifact)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    if artifact.exists():
        raise FileExistsError(artifact)
    torch.save({"schema": result["schema"], "protocol": protocol, "state": states}, artifact)
    return result


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item) for item in value.split(",") if item)
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--state-dim", type=int, default=64)
    parser.add_argument("--tables", type=int, default=32)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--residual-scale", type=float, default=0.25)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--prototype-std", type=float, default=0.02)
    parser.add_argument("--seeds", type=_parse_seeds, default=(0, 1, 2))
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-test", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.state_dim != 64 or args.tables != 32 or args.depth != 4:
        parser.error("formal protocol requires D64/T32/depth4")
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
