"""Rapid EMNIST comparison of flat, level-shared, and node-specific Pair routes.

All three arms use the same D64/T32/C4 residual lookup stack, dense stem,
payload initialization, optimizer, and minibatches.  The only route-axis change
is whether an adaptive tree shares one Pair support at each depth or stores an
independently sampled Pair support at every internal node.  Hard inference is
always 128 Pair comparisons and 32 row lookups per hard layer.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from tropnn.layers.hard_lookup import HardLookupRouter
from tropnn.tools.emnist_payload_dtype_sweep import _load_emnist_split
from tropnn.tools.emnist_router_dataflow_factorial import (
    ArmEvaluation,
    RouterStackClassifier,
    _gradient_summary,
    _loader,
    evaluate_router,
    sample_supports,
)

ARMS = ("flat_pair", "adaptive_level_pair", "adaptive_node_pair")


def _common_model(args: argparse.Namespace, classes: int, seed: int, topology: str) -> RouterStackClassifier:
    return RouterStackClassifier(
        784,
        args.state_dim,
        classes,
        hidden_layers=4,
        tables=args.tables,
        depth=args.depth,
        predicate="pair",
        topology=topology,  # type: ignore[arg-type]
        seed=seed,
        residual_scale=args.residual_scale,
        tau=args.tau,
        prototype_std=args.prototype_std,
    )


def _node_router(reference: HardLookupRouter, *, route_seed: int, tau: float) -> HardLookupRouter:
    supports = sample_supports(
        reference.tables,
        2**reference.depth - 1,
        reference.input_dim,
        route_seed,
    )
    return HardLookupRouter(
        reference.input_dim,
        reference.output_dim,
        depth=reference.depth,
        predicate="pair",
        topology="adaptive",
        support_layout="node",
        supports=supports,
        thresholds=torch.zeros(reference.tables, 2**reference.depth - 1),
        rows=reference.rows.detach().clone(),
        surrogate="local_counterfactual",
        tau=tau,
    )


def _node_specific_model(
    reference: RouterStackClassifier,
    *,
    seed: int,
    tau: float,
) -> RouterStackClassifier:
    model = copy.deepcopy(reference)
    model.blocks = nn.ModuleList(
        [_node_router(layer, route_seed=seed + 30_011 * (index + 1), tau=tau) for index, layer in enumerate(reference.blocks)]
    )
    model.head = _node_router(
        reference.head,
        route_seed=seed + 30_011 * (len(reference.blocks) + 1),
        tau=tau,
    )
    return model


def paired_models(
    args: argparse.Namespace,
    classes: int,
    seed: int,
) -> tuple[dict[str, RouterStackClassifier], dict[str, bool]]:
    flat = _common_model(args, classes, seed, "flat")
    level = _common_model(args, classes, seed, "adaptive")
    node = _node_specific_model(flat, seed=seed, tau=args.tau)
    models = {
        "flat_pair": flat,
        "adaptive_level_pair": level,
        "adaptive_node_pair": node,
    }

    flat_layers = flat.router_layers()
    level_layers = level.router_layers()
    node_layers = node.router_layers()
    checks = {
        "stems_exact": all(
            torch.equal(a, b) and torch.equal(a, c)
            for a, b, c in zip(
                flat.stem.state_dict().values(),
                level.stem.state_dict().values(),
                node.stem.state_dict().values(),
                strict=True,
            )
        ),
        "payload_rows_exact": all(
            torch.equal(a.rows, b.rows) and torch.equal(a.rows, c.rows) for a, b, c in zip(flat_layers, level_layers, node_layers, strict=True)
        ),
        "flat_level_supports_exact": all(torch.equal(a.supports, b.supports) for a, b in zip(flat_layers, level_layers, strict=True)),
        "node_support_shape_exact": all(layer.supports.shape == (args.tables, 2**args.depth - 1, 2) for layer in node_layers),
        "node_supports_not_level_repetition": all(
            any(
                not torch.equal(
                    layer.supports[:, 2**depth - 1 : 2 ** (depth + 1) - 1],
                    flat_layer.supports[:, depth : depth + 1].expand(-1, 2**depth, -1),
                )
                for depth in range(args.depth)
            )
            for flat_layer, layer in zip(flat_layers, node_layers, strict=True)
        ),
    }
    if not all(checks.values()):
        raise AssertionError(f"matched construction failed: {checks}")
    return models, checks


def train_models(
    models: dict[str, RouterStackClassifier],
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
    curves = {name: [] for name in ARMS}
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
        loss_sums = {name: 0.0 for name in ARMS}
        correct = {name: 0 for name in ARMS}
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
                loss_sums[name] += float(loss.detach()) * batch
                correct[name] += int((logits[name].detach().argmax(-1) == target).sum())
            count += batch
        for name in ARMS:
            curves[name].append(
                {
                    "epoch": float(epoch + 1),
                    "train_ce": loss_sums[name] / count,
                    "train_accuracy": correct[name] / count,
                }
            )
        print(
            f"seed={seed} epoch={epoch + 1}/{args.epochs} " + " ".join(f"{name}:ce={curves[name][-1]['train_ce']:.6f}" for name in ARMS),
            flush=True,
        )
    return curves, first_gradients


def run(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    root = Path(args.root).expanduser()
    train_x, train_y = _load_emnist_split(root, "balanced", train=True, limit=args.max_train, seed=0)
    held_x, held_y = _load_emnist_split(root, "balanced", train=False, limit=args.max_test, seed=0)
    classes = int(max(int(train_y.max()), int(held_y.max())) + 1)
    device = torch.device(args.device)
    held_loader = _loader(
        held_x,
        held_y,
        args.batch_size,
        args.workers,
        shuffle=False,
        seed=0,
        pin=device.type == "cuda",
    )
    models, checks = paired_models(args, classes, args.seed)
    curves, gradients = train_models(models, train_x, train_y, args=args, seed=args.seed, device=device)
    evaluations: list[ArmEvaluation] = [evaluate_router(name, model, held_loader, device) for name, model in models.items()]
    ce = {item.arm: item.held_ce for item in evaluations}
    summary = {
        "arms": {
            item.arm: {
                "held_ce": item.held_ce,
                "held_accuracy": item.held_accuracy,
                "trainable_parameters": item.trainable_parameters,
                "route_entropy_bits_mean": item.route_entropy_bits_mean,
                "observed_rows_mean": item.observed_rows_mean,
            }
            for item in evaluations
        },
        "adaptive_level_minus_flat_ce_gain": ce["flat_pair"] - ce["adaptive_level_pair"],
        "adaptive_node_minus_flat_ce_gain": ce["flat_pair"] - ce["adaptive_node_pair"],
        "adaptive_node_minus_level_ce_gain": ce["adaptive_level_pair"] - ce["adaptive_node_pair"],
        "all_hard_forwards_exact": all(item.hard_forward_exact is True for item in evaluations),
        "all_initialization_checks_exact": all(checks.values()),
        "all_router_gradients_nonzero": all(gradients[arm][field] > 0 for arm in ARMS for field in ("stem", "thresholds", "rows")),
    }
    protocol = {
        "dataset": "EMNIST Balanced",
        "description": "rapid single-seed node-specific Pair support test",
        "seed": args.seed,
        "train_examples": len(train_x),
        "held_examples": len(held_x),
        "state_dim": args.state_dim,
        "tables": args.tables,
        "depth": args.depth,
        "rows_per_table": 2**args.depth,
        "hidden_layers": 4,
        "active_comparisons_per_hard_layer": args.tables * args.depth,
        "active_row_lookups_per_hard_layer": args.tables,
        "flat_supports_per_hard_layer": args.tables * args.depth,
        "adaptive_level_supports_per_hard_layer": args.tables * args.depth,
        "adaptive_node_supports_per_hard_layer": args.tables * (2**args.depth - 1),
        "support_initialization": "fixed_random; flat/level shared; node independently sampled per internal node",
        "threshold_initialization": "all_zero",
        "payload_initialization": "tensor-identical across arms",
        "route_backward": "nearest_executed_wall_local_counterfactual",
        "discrete_supports_learned": False,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "residual_scale": args.residual_scale,
        "tau": args.tau,
        "prototype_std": args.prototype_std,
        "held_not_used_for_selection_or_early_stopping": True,
        "device": args.device,
        "torch_version": str(torch.__version__),
        "cuda_version": None if torch.version.cuda is None else str(torch.version.cuda),
    }
    result = {
        "schema": "emnist-pair-node-specific-tree-quick-v1",
        "protocol": protocol,
        "initialization_checks": checks,
        "first_step_gradient_norms": gradients,
        "curves": curves,
        "evaluations": [asdict(item) for item in evaluations],
        "summary": summary,
        "seconds": time.perf_counter() - started,
    }
    artifact = Path(args.artifact)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    if artifact.exists():
        raise FileExistsError(artifact)
    torch.save(
        {
            "schema": result["schema"],
            "protocol": protocol,
            "state": {f"{arm}.{key}": value.detach().cpu() for arm, model in models.items() for key, value in model.state_dict().items()},
        },
        artifact,
    )
    return result


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-test", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.state_dim != 64 or args.tables != 32 or args.depth != 4:
        parser.error("protocol requires D64/T32/depth4")
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
