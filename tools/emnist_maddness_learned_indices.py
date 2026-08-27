"""Train data-free-initialized hard MADDNESS stacks and discrete split indices."""

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

from tropnn.layers.hard_lookup import HardLookupRouter
from tropnn.tools.emnist_payload_dtype_sweep import _load_emnist_split
from tropnn.tools.maddness_end_to_end_ste_factorial import _leaf_stats

ARMS = ("fixed_indices_l0", "learned_indices_l0", "fixed_indices_l4", "learned_indices_l4", "dense_l4")


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
    selected_index_change_fraction: float | None
    selected_unique_coordinates_mean: float | None
    threshold_rms_drift: float | None
    productive_index_counterfactual_fraction: float | None


@dataclass(frozen=True)
class SeedResult:
    seed: int
    curves: dict[str, list[dict[str, float]]]
    first_step_gradient_norms: dict[str, dict[str, float]]
    evaluations: list[ArmEvaluation]
    seconds: float


def data_free_tree_thresholds(tables: int, *, dtype: torch.dtype = torch.float32) -> Tensor:
    """A fixed balanced binary-search grid on the bounded state interval [-1, 1]."""

    values = [0.0, -0.5, 0.5, -0.75, -0.25, 0.25, 0.75, -0.875, -0.625, -0.375, -0.125, 0.125, 0.375, 0.625, 0.875]
    return torch.tensor(values, dtype=dtype).repeat(tables, 1)


def _make_index_lookup(
    input_dim: int,
    output_dim: int,
    *,
    tables: int,
    seed: int,
    learn_indices: bool,
    tau: float,
    index_tau: float,
    prototype_std: float,
) -> HardLookupRouter:
    if input_dim > 256:
        raise ValueError("deployment index ledger assumes uint8 feature indices")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    scores = torch.randn(tables, 4, input_dim, generator=generator) * 1e-2
    supports = scores.argmax(-1)
    thresholds = data_free_tree_thresholds(tables)
    rows = torch.randn(tables, 16, output_dim, generator=generator) * prototype_std
    return HardLookupRouter(
        input_dim,
        output_dim,
        depth=4,
        predicate="unary",
        topology="adaptive",
        support_layout="level",
        supports=supports,
        thresholds=thresholds,
        rows=rows,
        surrogate="local_counterfactual",
        tau=tau,
        support_scores=scores,
        support_tau=index_tau,
        trainable_supports=learn_indices,
    )


class ScratchMaddnessStackClassifier(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        state_dim: int,
        classes: int,
        hidden_layers: int,
        tables: int,
        seed: int,
        learn_indices: bool,
        residual_scale: float,
        tau: float,
        index_tau: float,
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
                _make_index_lookup(
                    state_dim,
                    state_dim,
                    tables=tables,
                    seed=seed + 1009 * (layer + 1),
                    learn_indices=learn_indices,
                    tau=tau,
                    index_tau=index_tau,
                    prototype_std=prototype_std,
                )
                for layer in range(hidden_layers)
            ]
        )
        self.head = _make_index_lookup(
            state_dim,
            classes,
            tables=tables,
            seed=seed + 1009 * (hidden_layers + 1),
            learn_indices=learn_indices,
            tau=tau,
            index_tau=index_tau,
            prototype_std=prototype_std,
        )

    def index_layers(self) -> list[HardLookupRouter]:
        return [*self.blocks, self.head]

    def set_index_learning(self, enabled: bool) -> None:
        for layer in self.index_layers():
            layer.set_support_learning(enabled)

    def initial_state(self, x: Tensor) -> Tensor:
        return torch.tanh(self.stem(x.flatten(1)))

    def forward(self, x: Tensor) -> Tensor:
        state = self.initial_state(x)
        for block in self.blocks:
            state = (state + self.residual_scale * block(state)).clamp(-1.0, 1.0)
        return self.head(state)

    def hard_forward_with_trace(self, x: Tensor) -> tuple[Tensor, list[Tensor], list[Tensor]]:
        state = self.initial_state(x)
        codes: list[Tensor] = []
        productive: list[Tensor] = []
        for block in self.blocks:
            delta, code, useful = block.hard_output_with_support_counterfactual(state)
            codes.append(code)
            productive.append(useful)
            state = (state + self.residual_scale * delta).clamp(-1.0, 1.0)
        logits, code, useful = self.head.hard_output_with_support_counterfactual(state)
        codes.append(code)
        productive.append(useful)
        return logits, codes, productive


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


def paired_models(args: argparse.Namespace, classes: int, seed: int) -> dict[str, nn.Module]:
    common = dict(
        input_dim=784,
        state_dim=args.state_dim,
        classes=classes,
        tables=args.tables,
        seed=seed,
        learn_indices=True,
        residual_scale=args.residual_scale,
        tau=args.tau,
        index_tau=args.index_tau,
        prototype_std=args.prototype_std,
    )
    learned_l4 = ScratchMaddnessStackClassifier(hidden_layers=4, **common)
    learned_l0 = ScratchMaddnessStackClassifier(hidden_layers=0, **common)
    learned_l0.stem.load_state_dict(copy.deepcopy(learned_l4.stem.state_dict()))
    learned_l0.head.load_state_dict(copy.deepcopy(learned_l4.head.state_dict()))
    fixed_l4 = copy.deepcopy(learned_l4)
    fixed_l0 = copy.deepcopy(learned_l0)
    fixed_l4.set_index_learning(False)
    fixed_l0.set_index_learning(False)
    dense = DenseStackClassifier(784, args.state_dim, classes, 4, seed, args.residual_scale)
    dense.stem.load_state_dict(copy.deepcopy(learned_l4.stem.state_dict()))
    return {
        "fixed_indices_l0": fixed_l0,
        "learned_indices_l0": learned_l0,
        "fixed_indices_l4": fixed_l4,
        "learned_indices_l4": learned_l4,
        "dense_l4": dense,
    }


def _optimizer(model: nn.Module, lr: float, index_lr: float) -> torch.optim.Optimizer:
    selectors = [
        module.support_scores
        for module in model.modules()
        if isinstance(module, HardLookupRouter) and module.support_scores is not None and module.support_scores.requires_grad
    ]
    selector_ids = {id(parameter) for parameter in selectors}
    other = [parameter for parameter in model.parameters() if parameter.requires_grad and id(parameter) not in selector_ids]
    groups: list[dict[str, object]] = [{"params": other, "lr": lr}]
    if selectors:
        groups.append({"params": selectors, "lr": index_lr})
    return torch.optim.AdamW(groups, weight_decay=0.0)


def _gradient_norm(parameters: list[Tensor]) -> float:
    return math.sqrt(sum(float(parameter.grad.detach().square().sum()) for parameter in parameters if parameter.grad is not None))


def _first_gradient_summary(model: nn.Module) -> dict[str, float]:
    routers = [module for module in model.modules() if isinstance(module, HardLookupRouter) and module.support_scores is not None]
    selectors = [module.support_scores for module in routers if module.support_scores is not None]
    thresholds = [module.thresholds for module in routers]
    prototypes = [module.rows for module in routers]
    return {
        "stem": _gradient_norm(list(model.stem.parameters())),  # type: ignore[attr-defined]
        "selectors": _gradient_norm(selectors),
        "thresholds": _gradient_norm(thresholds),
        "prototypes": _gradient_norm(prototypes),
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
    optimizers = {name: _optimizer(model, args.lr, args.index_lr) for name, model in models.items()}
    curves = {name: [] for name in models}
    first_gradients: dict[str, dict[str, float]] = {}
    for epoch in range(args.epochs):
        loader = _loader(
            train_x,
            train_y,
            args.batch_size,
            args.workers,
            shuffle=True,
            seed=310_000 + seed * 100 + epoch,
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
                first_gradients = {name: _first_gradient_summary(model) for name, model in models.items()}
            for optimizer in optimizers.values():
                optimizer.step()
            batch = target.numel()
            for name, loss in losses.items():
                sums[name] += float(loss.detach()) * batch
                correct[name] += int((logits[name].detach().argmax(-1) == target).sum())
            count += batch
        metrics = {name: (sums[name] / count, correct[name] / count) for name in models}
        for name, (ce, accuracy) in metrics.items():
            curves[name].append({"epoch": float(epoch + 1), "train_ce": ce, "train_accuracy": accuracy})
        print(
            f"seed={seed} epoch={epoch + 1}/{args.epochs} "
            + " ".join(f"{name}:ce={value[0]:.6f},acc={value[1]:.6f}" for name, value in metrics.items()),
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
def evaluate_maddness(arm: str, model: ScratchMaddnessStackClassifier, loader: DataLoader, device: torch.device) -> ArmEvaluation:
    ce, accuracy = evaluate_plain(model, loader, device)
    layers = model.index_layers()
    all_codes: list[list[Tensor]] = [[] for _ in layers]
    productive_sum = 0
    productive_count = 0
    exact = True
    for x, _target in loader:
        x = x.to(device, non_blocking=True)
        hard, codes, productive = model.hard_forward_with_trace(x)
        exact = exact and torch.equal(model(x), hard)
        for layer, code in enumerate(codes):
            all_codes[layer].append(code.cpu())
            productive_sum += int(productive[layer].sum())
            productive_count += productive[layer].numel()
    entropies: list[float] = []
    minima: list[float] = []
    maximum_mass = 0.0
    for chunks in all_codes:
        entropy, minimum, maximum = _leaf_stats(torch.cat(chunks))
        entropies.append(entropy)
        minima.append(minimum)
        maximum_mass = max(maximum_mass, maximum)
    current = torch.cat([layer.selected_supports().detach().cpu().reshape(-1) for layer in layers])
    initial = torch.cat([layer.supports[..., 0].cpu().reshape(-1) for layer in layers])
    changes = float((current != initial).float().mean())
    unique = sum(float(layer.selected_supports().unique().numel()) for layer in layers) / len(layers)
    threshold_drift = torch.cat([(layer.thresholds.detach().cpu() - layer.initial_thresholds.cpu()).reshape(-1) for layer in layers])
    return ArmEvaluation(
        arm=arm,
        held_ce=ce,
        held_accuracy=accuracy,
        trainable_parameters=sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        hard_forward_exact=exact,
        route_entropy_bits_mean=sum(entropies) / len(entropies),
        route_entropy_bits_minimum=min(minima),
        maximum_row_mass=maximum_mass,
        selected_index_change_fraction=changes,
        selected_unique_coordinates_mean=unique,
        threshold_rms_drift=float(threshold_drift.square().mean().sqrt()),
        productive_index_counterfactual_fraction=productive_sum / productive_count,
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
    models = paired_models(args, classes, seed)
    curves, gradients = train_seed(models, train_x, train_y, args=args, seed=seed, device=device)
    evaluations: list[ArmEvaluation] = []
    for arm in ARMS:
        model = models[arm]
        if isinstance(model, ScratchMaddnessStackClassifier):
            evaluations.append(evaluate_maddness(arm, model, held_loader, device))
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
                    None,
                    None,
                )
            )
    state = {f"{arm}.{key}": value.detach().cpu() for arm, model in models.items() for key, value in model.state_dict().items()}
    return SeedResult(seed, curves, gradients, evaluations, time.perf_counter() - started), state


def summarize(rows: list[SeedResult]) -> dict[str, object]:
    arms: dict[str, dict[str, float]] = {}
    for arm in ARMS:
        selected = [next(item for item in row.evaluations if item.arm == arm) for row in rows]
        arms[arm] = {
            "held_ce_mean": sum(item.held_ce for item in selected) / len(selected),
            "held_accuracy_mean": sum(item.held_accuracy for item in selected) / len(selected),
        }
    fixed_l0 = arms["fixed_indices_l0"]["held_ce_mean"]
    learned_l0 = arms["learned_indices_l0"]["held_ce_mean"]
    fixed_l4 = arms["fixed_indices_l4"]["held_ce_mean"]
    learned_l4 = arms["learned_indices_l4"]["held_ce_mean"]
    return {
        "arms": arms,
        "effects": {
            "learned_index_gain_l0_ce": fixed_l0 - learned_l0,
            "learned_index_gain_l4_ce": fixed_l4 - learned_l4,
            "stacking_gain_with_learned_indices_ce": learned_l0 - learned_l4,
        },
        "decisions": {
            "learned_indices_improve_l0_all_seeds": all(
                next(item for item in row.evaluations if item.arm == "learned_indices_l0").held_ce
                < next(item for item in row.evaluations if item.arm == "fixed_indices_l0").held_ce
                for row in rows
            ),
            "learned_indices_improve_l4_all_seeds": all(
                next(item for item in row.evaluations if item.arm == "learned_indices_l4").held_ce
                < next(item for item in row.evaluations if item.arm == "fixed_indices_l4").held_ce
                for row in rows
            ),
            "stacking_improves_learned_all_seeds": all(
                next(item for item in row.evaluations if item.arm == "learned_indices_l4").held_ce
                < next(item for item in row.evaluations if item.arm == "learned_indices_l0").held_ce
                for row in rows
            ),
            "all_hard_forwards_exact": all(
                item.hard_forward_exact is True for row in rows for item in row.evaluations if item.hard_forward_exact is not None
            ),
            "learned_selector_gradients_nonzero_all_seeds": all(
                row.first_step_gradient_norms[arm]["selectors"] > 0 for row in rows for arm in ("learned_indices_l0", "learned_indices_l4")
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
        "leaves_per_table": 16,
        "tree_depth": 4,
        "hidden_layers_stacked": 4,
        "active_comparisons_per_hard_layer": 4 * args.tables,
        "active_lookups_per_hard_layer": args.tables,
        "route_calibration_used": False,
        "offline_action_compiler_used": False,
        "labels_or_data_used_for_initial_routes": False,
        "split_indices_learned_by_hard_program_counterfactual": True,
        "threshold_initialization": "fixed_data_free_binary_search_grid_on_minus1_plus1",
        "payload_initialization": "random_normal",
        "state_initialization": "random_dense_stem_then_tanh",
        "hidden_composition": "clamp(state + 0.25 * hard_row_sum, -1, 1)",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "index_learning_rate": args.index_lr,
        "tau": args.tau,
        "index_tau": args.index_tau,
        "prototype_std": args.prototype_std,
        "seeds": list(args.seeds),
        "held_not_used_for_selection_or_early_stopping": True,
        "device": args.device,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    result = {
        "schema": "emnist-maddness-learned-indices-v1",
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
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--index-lr", type=float, default=1e-4)
    parser.add_argument("--residual-scale", type=float, default=0.25)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--index-tau", type=float, default=1.0)
    parser.add_argument("--prototype-std", type=float, default=0.02)
    parser.add_argument("--seeds", type=_parse_seeds, default=(0, 1, 2))
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-test", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.state_dim != 64 or args.tables != 32:
        parser.error("formal protocol requires D64/T32")
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
