"""Matched EMNIST test of Euclidean versus ordinal residual geometry."""

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
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from tropnn.layers.ordinal_residual import MatchedOrdinalResidualBlock, OrdinalResidualKind
from tropnn.tools.emnist_payload_dtype_sweep import _load_emnist_split


@dataclass(frozen=True)
class RunResult:
    schema: str
    family: str
    seed: int
    depth: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    residual_scale: float
    train_examples: int
    held_examples: int
    classes: int
    core_parameters_per_layer: int
    core_parameters_total: int
    head_parameters: int
    total_parameters: int
    final_train_ce: float
    final_train_accuracy: float
    final_held_ce: float
    final_held_accuracy: float
    held_route_entropy_bits_mean: float
    held_effective_chambers_mean: float
    held_transition_fraction_mean: float
    held_transition_distance_mean: float
    seconds: float
    examples_per_second: float
    finite: bool
    device: str
    torch_version: str
    cuda_version: str | None


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _s4_distances() -> Tensor:
    adjacency = torch.full((24, 24), 99, dtype=torch.long)
    adjacency.fill_diagonal_(0)
    import itertools

    permutations = list(itertools.permutations(range(4)))
    lookup = {value: index for index, value in enumerate(permutations)}
    for index, permutation in enumerate(permutations):
        for generator in range(3):
            neighbor = list(permutation)
            neighbor[generator], neighbor[generator + 1] = neighbor[generator + 1], neighbor[generator]
            adjacency[index, lookup[tuple(neighbor)]] = 1
    for pivot in range(24):
        adjacency = torch.minimum(adjacency, adjacency[:, pivot : pivot + 1] + adjacency[pivot : pivot + 1, :])
    return adjacency


class OrdinalResidualEmnistClassifier(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        classes: int,
        depth: int,
        family: OrdinalResidualKind,
        seed: int,
        residual_scale: float,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.blocks = nn.ModuleList(
            [
                MatchedOrdinalResidualBlock(
                    dim,
                    kind=family,
                    seed=0 if layer == 0 else seed + 1009 * layer,
                    residual_scale=residual_scale,
                )
                for layer in range(depth)
            ]
        )
        self.head = nn.Linear(dim, classes)

    def forward_with_codes(self, x: Tensor) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        state = x.flatten(1)
        routes: list[tuple[Tensor, Tensor]] = []
        for block in self.blocks:
            state, before, after = block.forward_with_codes(state)
            routes.append((before, after))
        return self.head(state), routes

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_with_codes(x)[0]


def _loaders(args: argparse.Namespace, device: torch.device) -> tuple[DataLoader, DataLoader, int]:
    train_x, train_y = _load_emnist_split(args.root, args.split, train=True, limit=args.max_train, seed=args.seed)
    held_x, held_y = _load_emnist_split(args.root, args.split, train=False, limit=args.max_test, seed=args.seed)
    classes = int(max(train_y.max().item(), held_y.max().item()) + 1)
    generator = torch.Generator(device="cpu").manual_seed(0xA11CE + args.seed)
    train = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    held = DataLoader(
        TensorDataset(held_x, held_y),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    return train, held, classes


def _epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, float]:
    model.train(optimizer is not None)
    loss_sum = 0.0
    correct = 0
    count = 0
    for x, target in loader:
        x = x.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, target)
        if optimizer is not None:
            loss.backward()
            optimizer.step()
        batch = target.numel()
        loss_sum += float(loss.detach()) * batch
        correct += int((logits.argmax(-1) == target).sum())
        count += batch
    return loss_sum / count, correct / count


@torch.no_grad()
def _route_health(
    model: OrdinalResidualEmnistClassifier,
    loader: DataLoader,
    *,
    device: torch.device,
) -> tuple[float, float, float, float]:
    distance = _s4_distances().to(device)
    counts = [torch.zeros(24, device=device, dtype=torch.float64) for _ in model.blocks]
    changed = [0 for _ in model.blocks]
    total = [0 for _ in model.blocks]
    distance_sum = [0.0 for _ in model.blocks]
    model.eval()
    for x, _target in loader:
        _logits, routes = model.forward_with_codes(x.to(device, non_blocking=True))
        for layer, (before, after) in enumerate(routes):
            counts[layer] += torch.bincount(before.reshape(-1), minlength=24)
            changed[layer] += int((before != after).sum())
            total[layer] += before.numel()
            distance_sum[layer] += float(distance[before.reshape(-1), after.reshape(-1)].sum())
    entropies = []
    effective = []
    transitions = []
    distances = []
    for layer in range(len(model.blocks)):
        probability = counts[layer] / counts[layer].sum().clamp_min(1.0)
        active = probability > 0
        entropy = float(-(probability[active] * probability[active].log2()).sum())
        entropies.append(entropy)
        effective.append(2.0**entropy)
        transitions.append(changed[layer] / max(1, total[layer]))
        distances.append(distance_sum[layer] / max(1, total[layer]))
    return (
        sum(entropies) / len(entropies),
        sum(effective) / len(effective),
        sum(transitions) / len(transitions),
        sum(distances) / len(distances),
    )


def run(args: argparse.Namespace) -> RunResult:
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    train_loader, held_loader, classes = _loaders(args, device)
    model = OrdinalResidualEmnistClassifier(
        dim=784,
        classes=classes,
        depth=args.depth,
        family=args.family,
        seed=args.seed,
        residual_scale=args.residual_scale,
    ).to(device)
    core_per_layer = model.blocks[0].operator_parameters
    expected = model.blocks[0].expected_operator_parameters
    if args.family != "noop" and core_per_layer != expected:
        raise RuntimeError(f"parameter mismatch: got {core_per_layer}, expected {expected}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start = time.perf_counter()
    train_ce = train_accuracy = float("nan")
    for epoch in range(args.epochs):
        train_ce, train_accuracy = _epoch(model, train_loader, device=device, optimizer=optimizer)
        print(
            f"family={args.family} seed={args.seed} epoch={epoch + 1}/{args.epochs} train_ce={train_ce:.6f} train_acc={train_accuracy:.6f}",
            flush=True,
        )
    held_ce, held_accuracy = _epoch(model, held_loader, device=device, optimizer=None)
    entropy, effective, transition, transition_distance = _route_health(model, held_loader, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - start
    head_parameters = sum(parameter.numel() for parameter in model.head.parameters())
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    values = (train_ce, train_accuracy, held_ce, held_accuracy, entropy, effective, transition, transition_distance)
    result = RunResult(
        schema="emnist-ordinal-residual-geometry-v1",
        family=args.family,
        seed=args.seed,
        depth=args.depth,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        residual_scale=args.residual_scale,
        train_examples=len(train_loader.dataset),
        held_examples=len(held_loader.dataset),
        classes=classes,
        core_parameters_per_layer=core_per_layer,
        core_parameters_total=sum(block.operator_parameters for block in model.blocks),
        head_parameters=head_parameters,
        total_parameters=total_parameters,
        final_train_ce=train_ce,
        final_train_accuracy=train_accuracy,
        final_held_ce=held_ce,
        final_held_accuracy=held_accuracy,
        held_route_entropy_bits_mean=entropy,
        held_effective_chambers_mean=effective,
        held_transition_fraction_mean=transition,
        held_transition_distance_mean=transition_distance,
        seconds=seconds,
        examples_per_second=args.epochs * len(train_loader.dataset) / seconds,
        finite=all(math.isfinite(value) for value in values),
        device=str(device),
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
    )
    _atomic_json(args.output, asdict(result))
    print(
        f"held family={args.family} seed={args.seed} ce={held_ce:.6f} acc={held_accuracy:.6f} "
        f"transition={transition:.6f} distance={transition_distance:.6f}",
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", default="balanced")
    parser.add_argument("--family", choices=("noop", "row", "coxeter", "coxeter_relabel", "dense"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--residual-scale", type=float, default=0.25)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-test", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    run(args)


if __name__ == "__main__":
    main()
