"""EMNIST experiments for local energy smoothing on Pairwise LUT chambers.

The model matches the slide-deck EMNIST anchor-sweep scale by default:
depth=4 total LUT layers, hidden_dim=128, tables=136, comparisons=6,
residual hidden layers, learnable thresholds, and full-vector payload rows.

Variants:
* train_local: smooth only during training; evaluation remains hard LUT.
* forward_local: smooth during both training and evaluation.
* adjacent_reg: keep hard LUT forward, add local adjacent-row regularization.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from tropnn.layers.pairwise import PAIRWISE_ANCHOR_POLICIES, PairwiseLUT, PairwiseRoute
from tropnn.tools.emnist_payload_dtype_sweep import _build_local_loaders, _loader_examples

SmoothingVariant = Literal["train_local", "forward_local", "adjacent_reg"]


@dataclass(frozen=True)
class EpochStats:
    loss: float
    acc: float
    reg_loss: float


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _selected_rows(lut: Tensor, indices: Tensor, *, table_size: int, output_dim: int) -> Tensor:
    prefix, routes = indices.shape[:-1], indices.shape[-1]
    items = max(1, indices.numel() // routes)
    flat_indices = indices.reshape(items, routes)
    flat_lut = lut.reshape(routes * table_size, output_dim)
    output = torch.empty(items, routes, output_dim, device=indices.device, dtype=lut.dtype)
    for start in range(0, routes, 32):
        stop = min(routes, start + 32)
        offsets = (torch.arange(start, stop, device=indices.device) * table_size).view(1, -1)
        rows = (flat_indices[:, start:stop] + offsets).reshape(-1)
        values = flat_lut.index_select(0, rows).view(items, stop - start, output_dim)
        output[:, start:stop] = values
    return output.view(*prefix, routes, output_dim)


def _top_neighbor_delta(
    route: PairwiseRoute,
    lut: Tensor,
    *,
    table_size: int,
    output_dim: int,
    neighbor_count: int,
    temperature: float,
    strength: float,
) -> tuple[Tensor, Tensor]:
    count = max(1, min(int(neighbor_count), int(route.margins.shape[-1])))
    tau = max(float(temperature), 1e-6)
    scale = float(strength) / float(count)
    bits = route.margins.abs().topk(k=count, dim=-1, largest=False).indices
    current = _selected_rows(lut, route.indices, table_size=table_size, output_dim=output_dim).float()
    output_delta = torch.zeros(*route.indices.shape[:-1], output_dim, device=route.indices.device, dtype=torch.float32)
    reg_terms: list[Tensor] = []
    for slot in range(count):
        bit = bits[..., slot]
        margin = route.margins.gather(dim=-1, index=bit.unsqueeze(-1)).squeeze(-1)
        neighbor = route.indices ^ (2**bit).long()
        neighbor_rows = _selected_rows(lut, neighbor, table_size=table_size, output_dim=output_dim).float()
        delta = neighbor_rows - current
        weight = (scale * torch.exp(-margin.abs().float() / tau)).unsqueeze(-1)
        output_delta = output_delta + (weight * delta).sum(dim=-2)
        reg_terms.append((weight.detach() * delta.square()).mean())
    reg = torch.stack(reg_terms).mean() if reg_terms else output_delta.sum() * 0.0
    return output_delta, reg


class EnergySmoothingPairwiseLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        tables: int,
        comparisons: int,
        seed: int,
        backend: str,
        anchor_policy: str,
        lut_dtype: str,
        lut_init_std: float,
        fixed_zero_threshold: bool,
        use_output_scaling: bool,
        neighbor_count: int,
        smoothing_temperature: float,
        smoothing_strength: float,
        variant: SmoothingVariant,
    ) -> None:
        super().__init__()
        self.core = PairwiseLUT(
            input_dim,
            output_dim,
            tables=tables,
            comparisons=comparisons,
            seed=seed,
            anchor_seed=seed,
            lut_init_std=lut_init_std,
            backend=backend,  # type: ignore[arg-type]
            anchor_policy=anchor_policy,
            lut_dtype=lut_dtype,  # type: ignore[arg-type]
            fixed_zero_threshold=fixed_zero_threshold,
            use_output_scaling=use_output_scaling,
        )
        self.variant = variant
        self.neighbor_count = int(neighbor_count)
        self.smoothing_temperature = float(smoothing_temperature)
        self.smoothing_strength = float(smoothing_strength)
        self.last_adjacent_reg = torch.tensor(0.0)

    @property
    def table_size(self) -> int:
        return self.core.table_size

    @property
    def output_dim(self) -> int:
        return self.core.output_dim

    def forward(self, x: Tensor) -> Tensor:
        x_seq = x.unsqueeze(1) if x.ndim == 2 else x
        compute_dtype = torch.float32 if x_seq.dtype in {torch.float16, torch.bfloat16} else x_seq.dtype
        route = self.core.cache_index(x_seq.to(torch.float32 if self.core.backend in {"auto", "tilelang", "triton", "zig"} else compute_dtype))
        lut = self.core.lut_payload(dtype=compute_dtype, device=route.indices.device)
        output = self.core.lut_forward(route, lut, compute_dtype=compute_dtype)
        if self.training and (x_seq.requires_grad or bool(getattr(self.core.thresholds, "requires_grad", False))):
            output = output + self.core.lut_backward_surrogate(route, lut).to(output.dtype)
        delta, reg = _top_neighbor_delta(
            route,
            lut,
            table_size=self.table_size,
            output_dim=self.output_dim,
            neighbor_count=self.neighbor_count,
            temperature=self.smoothing_temperature,
            strength=self.smoothing_strength,
        )
        self.last_adjacent_reg = reg
        if self.variant == "forward_local" or (self.variant == "train_local" and self.training):
            output = output + delta.to(output.dtype)
        if self.core.output_scale != 1.0:
            output = output * self.core.output_scale
        return output.squeeze(1) if x.ndim == 2 else output


class EnergySmoothingEmnistClassifier(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        depth: int,
        tables: int,
        comparisons: int,
        seed: int,
        backend: str,
        anchor_policy: str,
        lut_dtype: str,
        lut_init_std: float,
        fixed_zero_threshold: bool,
        use_output_scaling: bool,
        residual_scale: float,
        neighbor_count: int,
        smoothing_temperature: float,
        smoothing_strength: float,
        variant: SmoothingVariant,
    ) -> None:
        super().__init__()
        dims = [input_dim] + [hidden_dim] * max(0, depth - 1) + [num_classes]
        self.layers = nn.ModuleList(
            [
                EnergySmoothingPairwiseLayer(
                    dims[idx],
                    dims[idx + 1],
                    tables=tables,
                    comparisons=comparisons,
                    seed=seed + idx,
                    backend=backend,
                    anchor_policy=anchor_policy,
                    lut_dtype=lut_dtype,
                    lut_init_std=lut_init_std,
                    fixed_zero_threshold=fixed_zero_threshold,
                    use_output_scaling=use_output_scaling,
                    neighbor_count=neighbor_count,
                    smoothing_temperature=smoothing_temperature,
                    smoothing_strength=smoothing_strength,
                    variant=variant,
                )
                for idx in range(len(dims) - 1)
            ]
        )
        self.residual_scale = float(residual_scale)

    def forward(self, x: Tensor) -> Tensor:
        y = x.flatten(1).float()
        for idx, layer in enumerate(self.layers):
            z = layer(y)
            is_readout = idx + 1 == len(self.layers)
            if is_readout:
                y = z
            elif z.shape == y.shape:
                y = y + self.residual_scale * z
            else:
                y = z
        return y

    def adjacent_regularization(self) -> Tensor:
        regs = [layer.last_adjacent_reg for layer in self.layers]
        return torch.stack([reg.to(next(self.parameters()).device) for reg in regs]).mean()


def _count_params(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _run_epoch(
    model: EnergySmoothingEmnistClassifier,
    loader,
    *,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    regularization_weight: float,
) -> EpochStats:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_reg = 0.0
    total_correct = 0
    total_seen = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        ce = F.cross_entropy(logits, y)
        reg = model.adjacent_regularization() if regularization_weight > 0.0 else ce.detach() * 0.0
        loss = ce + float(regularization_weight) * reg
        if training:
            loss.backward()
            optimizer.step()
        batch = int(y.numel())
        total_loss += float(ce.detach().item()) * batch
        total_reg += float(reg.detach().item()) * batch
        total_correct += int((logits.argmax(dim=-1) == y).sum().item())
        total_seen += batch
    return EpochStats(total_loss / max(1, total_seen), total_correct / max(1, total_seen), total_reg / max(1, total_seen))


def _time_forward_backward(model: nn.Module, loader, *, device: torch.device, iters: int) -> float:
    x, y = next(iter(loader))
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    for _ in range(2):
        model.zero_grad(set_to_none=True)
        F.cross_entropy(model(x), y).backward()
    _sync(device)
    start = time.perf_counter()
    for _ in range(max(1, iters)):
        model.zero_grad(set_to_none=True)
        F.cross_entropy(model(x), y).backward()
    _sync(device)
    return 1000.0 * (time.perf_counter() - start) / max(1, iters)


def _write_csv(path: Path, row: dict[str, float | int | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["train_local", "forward_local", "adjacent_reg"], required=True)
    parser.add_argument("--root", type=Path, default=Path("data/emnist"))
    parser.add_argument("--split", default="balanced")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--tables", type=int, default=136)
    parser.add_argument("--comparisons", type=int, default=6)
    parser.add_argument("--anchor-policy", choices=PAIRWISE_ANCHOR_POLICIES, default="expander")
    parser.add_argument("--backend", choices=["auto", "torch", "tilelang", "triton"], default="torch")
    parser.add_argument("--lut-dtype", choices=["fp32", "bf16", "fp16", "int8", "fp8", "int4", "int2", "fp4", "nf4"], default="bf16")
    parser.add_argument("--lut-init-std", type=float, default=0.02)
    parser.add_argument("--fixed-zero-threshold", action="store_true")
    parser.add_argument("--no-output-scaling", action="store_true")
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--neighbor-count", type=int, default=1)
    parser.add_argument("--smoothing-temperature", type=float, default=0.25)
    parser.add_argument("--smoothing-strength", type=float, default=0.5)
    parser.add_argument("--regularization-weight", type=float, default=1e-3)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-test", type=int, default=0)
    parser.add_argument("--timing-iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _seed_everything(args.seed)
    args.max_train = 0 if args.max_train is None else args.max_train
    args.max_test = 0 if args.max_test is None else args.max_test
    args.root = args.root
    args.workers = args.workers
    device = torch.device(args.device)
    train_loader, valid_loader, classes = _build_local_loaders(args)
    reg_weight = args.regularization_weight if args.variant == "adjacent_reg" else 0.0
    model = EnergySmoothingEmnistClassifier(
        input_dim=28 * 28,
        hidden_dim=args.hidden_dim,
        num_classes=classes,
        depth=args.depth,
        tables=args.tables,
        comparisons=args.comparisons,
        seed=args.seed,
        backend=args.backend,
        anchor_policy=args.anchor_policy,
        lut_dtype=args.lut_dtype,
        lut_init_std=args.lut_init_std,
        fixed_zero_threshold=args.fixed_zero_threshold,
        use_output_scaling=not args.no_output_scaling,
        residual_scale=args.residual_scale,
        neighbor_count=args.neighbor_count,
        smoothing_temperature=args.smoothing_temperature,
        smoothing_strength=args.smoothing_strength,
        variant=args.variant,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train = EpochStats(math.nan, math.nan, math.nan)
    for epoch in range(1, args.epochs + 1):
        train = _run_epoch(model, train_loader, optimizer=optimizer, device=device, regularization_weight=reg_weight)
        valid = _run_epoch(model, valid_loader, optimizer=None, device=device, regularization_weight=0.0)
        print(
            f"epoch={epoch} variant={args.variant} train_loss={train.loss:.6f} train_acc={train.acc:.6f} "
            f"valid_loss={valid.loss:.6f} valid_acc={valid.acc:.6f} reg_loss={train.reg_loss:.6f}",
            flush=True,
        )
    valid = _run_epoch(model, valid_loader, optimizer=None, device=device, regularization_weight=0.0)
    fwd_bwd_ms = _time_forward_backward(model, train_loader, device=device, iters=args.timing_iters)
    row: dict[str, float | int | str] = {
        "variant": args.variant,
        "split": args.split,
        "train_examples": _loader_examples(train_loader),
        "valid_examples": _loader_examples(valid_loader),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "depth": args.depth,
        "hidden_dim": args.hidden_dim,
        "tables": args.tables,
        "comparisons": args.comparisons,
        "anchor_policy": args.anchor_policy,
        "backend": args.backend,
        "lut_dtype": args.lut_dtype,
        "lut_init_std": args.lut_init_std,
        "fixed_zero_threshold": int(bool(args.fixed_zero_threshold)),
        "output_scaling": int(not args.no_output_scaling),
        "neighbor_count": args.neighbor_count,
        "smoothing_temperature": args.smoothing_temperature,
        "smoothing_strength": args.smoothing_strength,
        "regularization_weight": reg_weight,
        "params": _count_params(model),
        "train_loss": train.loss,
        "train_acc": train.acc,
        "train_reg_loss": train.reg_loss,
        "valid_loss": valid.loss,
        "valid_acc": valid.acc,
        "fwd_bwd_ms": fwd_bwd_ms,
        "seed": args.seed,
    }
    _write_csv(args.out, row)
    print(
        "variant={variant} valid_loss={valid_loss:.6f} valid_acc={valid_acc:.6f} "
        "params={params} fwd_bwd_ms={fwd_bwd_ms:.3f} out={out}".format(**row, out=args.out),
        flush=True,
    )


if __name__ == "__main__":
    main()
