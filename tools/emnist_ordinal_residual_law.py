"""EMNIST test of Euclidean versus intrinsic ordinal residual composition."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
from torch import nn

from tropnn.layers.ordinal_residual_law import OrdinalResidualLaw, OrdinalResidualLawBlock
from tropnn.tools.emnist_ordinal_residual_geometry import _epoch, _loaders, _route_health


class OrdinalResidualLawClassifier(nn.Module):
    def __init__(self, *, dim: int, classes: int, depth: int, law: OrdinalResidualLaw, seed: int, residual_scale: float) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                OrdinalResidualLawBlock(
                    dim,
                    law=law,
                    seed=0 if layer == 0 else seed + 1009 * layer,
                    residual_scale=residual_scale,
                )
                for layer in range(depth)
            ]
        )
        self.head = nn.Linear(dim, classes)

    def forward_with_codes(self, x: torch.Tensor) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        state = x.flatten(1)
        routes = []
        for block in self.blocks:
            state, before, after = block.forward_with_codes(state)
            routes.append((before, after))
        return self.head(state), routes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_codes(x)[0]


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, object]:
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    train_loader, held_loader, classes = _loaders(args, device)
    model = OrdinalResidualLawClassifier(
        dim=784,
        classes=classes,
        depth=args.depth,
        law=args.law,
        seed=args.seed,
        residual_scale=args.residual_scale,
    ).to(device)
    per_layer = model.blocks[0].operator_parameters
    if per_layer != model.blocks[0].expected_operator_parameters:
        raise RuntimeError("operator parameter budget mismatch")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    started = time.perf_counter()
    train_ce = train_accuracy = float("nan")
    for epoch in range(args.epochs):
        train_ce, train_accuracy = _epoch(model, train_loader, device=device, optimizer=optimizer)
        print(
            f"law={args.law} seed={args.seed} epoch={epoch + 1}/{args.epochs} train_ce={train_ce:.6f} train_acc={train_accuracy:.6f}",
            flush=True,
        )
    held_ce, held_accuracy = _epoch(model, held_loader, device=device, optimizer=None)
    entropy, effective, transition, distance = _route_health(model, held_loader, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    result: dict[str, object] = {
        "schema": "emnist-ordinal-residual-law-v1",
        "scientific_question": "same ordinal tangent field, Euclidean Euler versus intrinsic multiplicative residual retraction",
        "law": args.law,
        "seed": args.seed,
        "depth": args.depth,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "residual_scale": args.residual_scale,
        "core_parameters_per_layer": per_layer,
        "core_parameters_total": sum(block.operator_parameters for block in model.blocks),
        "head_parameters": sum(parameter.numel() for parameter in model.head.parameters()),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "final_train_ce": train_ce,
        "final_train_accuracy": train_accuracy,
        "final_held_ce": held_ce,
        "final_held_accuracy": held_accuracy,
        "held_route_entropy_bits_mean": entropy,
        "held_effective_chambers_mean": effective,
        "held_transition_fraction_mean": transition,
        "held_transition_distance_mean": distance,
        "seconds": seconds,
        "examples_per_second": args.epochs * len(train_loader.dataset) / seconds,
        "finite": all(math.isfinite(value) for value in (train_ce, train_accuracy, held_ce, held_accuracy, entropy, effective, transition, distance)),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    _atomic_json(args.output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", default="balanced")
    parser.add_argument("--law", choices=("noop", "euclidean_euler", "intrinsic_exp"), required=True)
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
