"""EMNIST Balanced experiments for symbolic/tropical Pairwise LUT variants.

Variants:

* plain: independent Pairwise tables with full-vector payloads.
* aggregate_code: table payload codes depend on a rolling composition of earlier table codes.
* route_edges: previous-layer route codes select the next layer's comparison-edge bank.
* sparse_maxplus: route coordinates are sparse max-plus transforms before comparison.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from tropnn.layers.pairwise import PAIRWISE_ANCHOR_POLICIES, PairwiseLUT, ste_heaviside
from tropnn.tools.emnist_cross_layer_anchor_sharing import (
    _boundary_density,
    _connected_components,
    _entropy,
    _grid,
    _pca_plane,
    _refinement,
    _route_entropy,
    _route_persistence,
    _signature_ids,
)
from tropnn.tools.emnist_payload_dtype_sweep import _build_local_loaders, _loader_examples

Variant = Literal["plain", "aggregate_code", "route_edges", "sparse_maxplus"]
VARIANTS: tuple[str, ...] = ("plain", "aggregate_code", "route_edges", "sparse_maxplus")


@dataclass(frozen=True)
class EvalResult:
    loss: float
    acc: float
    route_entropy: float
    route_persistence: float


@dataclass(frozen=True)
class RefinementResult:
    unique_signatures: int
    signature_entropy: float
    connected_components: int
    boundary_density: float
    refinement_mean: float
    refinement_max: int


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _anchor_bank(input_dim: int, tables: int, comparisons: int, edge_bank_size: int, anchor_policy: str, seed: int) -> Tensor:
    anchors: list[Tensor] = []
    for bank in range(edge_bank_size):
        layer = PairwiseLUT(
            input_dim,
            1,
            tables=tables,
            comparisons=comparisons,
            anchor_policy=anchor_policy,
            seed=seed + 1009 * bank,
            anchor_seed=seed + 1009 * bank,
            backend="torch",
        )
        anchors.append(layer.anchors.detach().clone())
    return torch.stack(anchors, dim=0)


def _maxplus_indices(input_dim: int, fanout: int, seed: int) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.empty(input_dim, fanout, dtype=torch.long)
    indices[:, 0] = torch.arange(input_dim, dtype=torch.long)
    if fanout == 1:
        return indices
    stride = max(1, input_dim // fanout)
    base = torch.arange(input_dim, dtype=torch.long).view(-1, 1)
    offsets = torch.arange(1, fanout, dtype=torch.long).view(1, -1)
    jitter = torch.randint(0, max(1, stride), (input_dim, fanout - 1), generator=generator, dtype=torch.long)
    indices[:, 1:] = (base * 997 + offsets * stride + jitter) % input_dim
    return indices


class SymbolicTropicalLUTLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        tables: int,
        comparisons: int,
        variant: Variant,
        anchor_policy: str,
        seed: int,
        lut_init_std: float,
        edge_bank_size: int,
        maxplus_fanout: int,
        use_output_scaling: bool,
    ) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant={variant!r}; choices={VARIANTS}")
        if anchor_policy not in PAIRWISE_ANCHOR_POLICIES:
            raise ValueError(f"unknown anchor_policy={anchor_policy!r}; choices={PAIRWISE_ANCHOR_POLICIES}")
        if tables < 1 or comparisons < 1:
            raise ValueError("tables and comparisons must be positive")
        if edge_bank_size < 1:
            raise ValueError("edge_bank_size must be positive")
        if maxplus_fanout < 1:
            raise ValueError("maxplus_fanout must be positive")

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.tables = int(tables)
        self.comparisons = int(comparisons)
        self.table_size = 1 << int(comparisons)
        self.variant = variant
        self.edge_bank_size = int(edge_bank_size)
        self.maxplus_fanout = int(maxplus_fanout)
        self.output_scale = 1.0 / math.sqrt(tables) if use_output_scaling else 1.0

        self.register_buffer("anchors", _anchor_bank(input_dim, tables, comparisons, edge_bank_size, anchor_policy, seed))
        self.register_buffer("powers", 2 ** torch.arange(comparisons, dtype=torch.long))
        self.register_buffer("maxplus_indices", _maxplus_indices(input_dim, maxplus_fanout, seed + 7919))

        if variant == "route_edges":
            self.thresholds = nn.Parameter(torch.zeros(edge_bank_size, tables, comparisons))
        else:
            self.thresholds = nn.Parameter(torch.zeros(tables, comparisons))
        if variant == "sparse_maxplus":
            self.maxplus_offsets = nn.Parameter(torch.zeros(input_dim, maxplus_fanout))
        else:
            self.register_parameter("maxplus_offsets", None)
        self.lut = nn.Parameter(torch.randn(tables, self.table_size, output_dim) * lut_init_std)

    def _route_coordinates(self, x: Tensor) -> Tensor:
        if self.variant != "sparse_maxplus":
            return x
        gathered = x.index_select(1, self.maxplus_indices.reshape(-1).to(device=x.device))
        gathered = gathered.view(x.shape[0], self.input_dim, self.maxplus_fanout)
        shifted = gathered + self.maxplus_offsets.to(device=x.device, dtype=x.dtype).view(1, self.input_dim, self.maxplus_fanout)
        return shifted.max(dim=-1).values

    def _static_route(self, route_x: Tensor) -> tuple[Tensor, Tensor]:
        anchors = self.anchors[0]
        anchor_a = anchors[:, :, 0].flatten().to(device=route_x.device)
        anchor_b = anchors[:, :, 1].flatten().to(device=route_x.device)
        x_a = route_x.index_select(1, anchor_a).view(route_x.shape[0], self.tables, self.comparisons)
        x_b = route_x.index_select(1, anchor_b).view(route_x.shape[0], self.tables, self.comparisons)
        margins = x_a - x_b - self.thresholds.to(device=route_x.device, dtype=route_x.dtype).view(1, self.tables, self.comparisons)
        indices = self._indices_from_margins(margins)
        return indices, margins

    def _route_edge_route(self, route_x: Tensor, context: Tensor | None) -> tuple[Tensor, Tensor]:
        if context is None:
            context = torch.zeros(route_x.shape[0], self.tables, device=route_x.device, dtype=torch.long)
        selector = (context.long() % self.edge_bank_size).to(device=route_x.device)
        bank = self.anchors.to(device=route_x.device)
        thresholds = self.thresholds.to(device=route_x.device, dtype=route_x.dtype)
        anchor_a = torch.empty(route_x.shape[0], self.tables, self.comparisons, device=route_x.device, dtype=torch.long)
        anchor_b = torch.empty_like(anchor_a)
        theta = torch.empty(route_x.shape[0], self.tables, self.comparisons, device=route_x.device, dtype=route_x.dtype)
        for table in range(self.tables):
            selected = selector[:, table]
            anchor_a[:, table, :] = bank[selected, table, :, 0]
            anchor_b[:, table, :] = bank[selected, table, :, 1]
            theta[:, table, :] = thresholds[selected, table, :]
        x_a = route_x.gather(1, anchor_a.reshape(route_x.shape[0], -1)).view(route_x.shape[0], self.tables, self.comparisons)
        x_b = route_x.gather(1, anchor_b.reshape(route_x.shape[0], -1)).view(route_x.shape[0], self.tables, self.comparisons)
        margins = x_a - x_b - theta
        indices = self._indices_from_margins(margins)
        return indices, margins

    def _indices_from_margins(self, margins: Tensor) -> Tensor:
        bits = (margins > 0).to(torch.long)
        powers = self.powers.to(device=margins.device).view(1, 1, -1)
        return (bits * powers).sum(dim=-1)

    def _compose_table_codes(self, raw_indices: Tensor) -> Tensor:
        if self.variant != "aggregate_code":
            return raw_indices
        mask = self.table_size - 1
        summary = torch.zeros(raw_indices.shape[0], device=raw_indices.device, dtype=torch.long)
        composed: list[Tensor] = []
        for table in range(self.tables):
            code = raw_indices[:, table] ^ summary
            code = code & mask
            composed.append(code)
            summary = ((summary << 1) ^ raw_indices[:, table] ^ (summary >> 1)) & mask
        return torch.stack(composed, dim=1)

    def _lookup(self, indices: Tensor) -> Tensor:
        table_offsets = torch.arange(self.tables, device=indices.device, dtype=torch.long).view(1, self.tables) * self.table_size
        flat_indices = (indices + table_offsets).reshape(-1)
        rows = self.lut.reshape(self.tables * self.table_size, self.output_dim).index_select(0, flat_indices)
        return rows.view(indices.shape[0], self.tables, self.output_dim)

    def _payload_to_output(self, payload: Tensor) -> Tensor:
        return payload.sum(dim=1) * self.output_scale

    def _ste_correction(self, selected_indices: Tensor, margins: Tensor, payload: Tensor) -> Tensor:
        bit = margins.abs().argmin(dim=-1)
        margin = margins.gather(dim=-1, index=bit.unsqueeze(-1)).squeeze(-1)
        neighbor_indices = selected_indices ^ (2 ** bit).long()
        ste_delta = ste_heaviside(margin) - (margin > 0).to(margin.dtype)
        delta = self._lookup(neighbor_indices) - payload
        return self._payload_to_output(delta * ste_delta.unsqueeze(-1))

    def compute(self, x: Tensor, context: Tensor | None) -> tuple[Tensor, Tensor]:
        input_dtype = x.dtype
        route_x = self._route_coordinates(x.float())
        if self.variant == "route_edges":
            raw_indices, margins = self._route_edge_route(route_x, context)
        else:
            raw_indices, margins = self._static_route(route_x)
        selected_indices = self._compose_table_codes(raw_indices)
        payload = self._lookup(selected_indices)
        output = self._payload_to_output(payload)
        if self.training and (x.requires_grad or self.thresholds.requires_grad):
            output = output + self._ste_correction(selected_indices, margins, payload)
        return output.to(dtype=input_dtype), selected_indices


class SymbolicTropicalEmnistClassifier(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        classes: int,
        depth: int,
        tables: int,
        comparisons: int,
        variant: Variant,
        anchor_policy: str,
        seed: int,
        lut_init_std: float,
        residual_scale: float,
        edge_bank_size: int,
        maxplus_fanout: int,
        use_output_scaling: bool,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            SymbolicTropicalLUTLayer(
                input_dim,
                input_dim,
                tables=tables,
                comparisons=comparisons,
                variant=variant,
                anchor_policy=anchor_policy,
                seed=seed + 101 * idx,
                lut_init_std=lut_init_std,
                edge_bank_size=edge_bank_size,
                maxplus_fanout=maxplus_fanout,
                use_output_scaling=use_output_scaling,
            )
            for idx in range(depth)
        )
        self.readout = SymbolicTropicalLUTLayer(
            input_dim,
            classes,
            tables=tables,
            comparisons=comparisons,
            variant=variant,
            anchor_policy=anchor_policy,
            seed=seed + 10007,
            lut_init_std=lut_init_std,
            edge_bank_size=edge_bank_size,
            maxplus_fanout=maxplus_fanout,
            use_output_scaling=use_output_scaling,
        )
        self.residual_scale = float(residual_scale)
        self.last_routes: list[Tensor] = []

    def forward(self, x: Tensor) -> Tensor:
        y = x.flatten(start_dim=1).float()
        routes: list[Tensor] = []
        context: Tensor | None = None
        for block in self.blocks:
            output, route = block.compute(y, context)
            y = y + self.residual_scale * output
            routes.append(route.detach())
            context = route.detach()
        logits, _readout_route = self.readout.compute(y, context)
        self.last_routes = routes
        return logits


@torch.no_grad()
def _eval(model: SymbolicTropicalEmnistClassifier, loader, device: torch.device) -> EvalResult:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    entropies: list[float] = []
    persistences: list[float] = []
    table_size = model.blocks[0].table_size if model.blocks else 1
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = F.cross_entropy(logits, labels, reduction="sum")
        total_loss += float(loss.item())
        total_correct += int((logits.argmax(dim=-1) == labels).sum().item())
        total_seen += int(labels.numel())
        entropies.append(_route_entropy(model.last_routes, table_size))
        persistences.append(_route_persistence(model.last_routes))
    model.train()
    return EvalResult(
        loss=total_loss / max(1, total_seen),
        acc=total_correct / max(1, total_seen),
        route_entropy=sum(entropies) / max(1, len(entropies)),
        route_persistence=sum(persistences) / max(1, len(persistences)),
    )


@torch.no_grad()
def _collect_probe_tensor(loader, *, limit: int) -> Tensor:
    xs: list[Tensor] = []
    total = 0
    for x, _y in loader:
        remaining = limit - total if limit > 0 else x.shape[0]
        if remaining <= 0:
            break
        take = min(int(x.shape[0]), int(remaining))
        xs.append(x[:take].cpu())
        total += take
    if not xs:
        return torch.empty(0, 1, 28, 28)
    return torch.cat(xs, dim=0)


@torch.no_grad()
def _batch_signatures(model: SymbolicTropicalEmnistClassifier, points: Tensor, *, device: torch.device, batch_size: int) -> list[Tensor]:
    outputs: list[list[Tensor]] = []
    model.eval()
    for start in range(0, points.shape[0], batch_size):
        model(points[start : start + batch_size].to(device))
        signatures = [route.cpu().to(torch.int16).reshape(route.shape[0], -1) for route in model.last_routes]
        if not outputs:
            outputs = [[] for _ in signatures]
        for idx, signature in enumerate(signatures):
            outputs[idx].append(signature)
    return [torch.cat(parts, dim=0) for parts in outputs]


@torch.no_grad()
def _refinement_probe(model: SymbolicTropicalEmnistClassifier, train_loader, args: argparse.Namespace, *, device: torch.device) -> RefinementResult:
    if args.skip_refinement_probe:
        return RefinementResult(0, math.nan, 0, math.nan, math.nan, 0)
    x_train = _collect_probe_tensor(train_loader, limit=args.pca_samples)
    center, u, v = _pca_plane(x_train, limit=args.pca_samples)
    points, _uu, _vv = _grid(center, u, v, grid_size=args.grid_size, span=args.plane_span)
    signatures = _batch_signatures(model, points, device=device, batch_size=args.probe_batch_size)
    ids = _signature_ids(signatures)
    if ids.numel() == 0:
        ids = torch.zeros(points.shape[0], dtype=torch.long)
    refinement_mean, refinement_max = _refinement(signatures)
    return RefinementResult(
        unique_signatures=int(torch.unique(ids).numel()),
        signature_entropy=_entropy(ids),
        connected_components=_connected_components(ids, args.grid_size),
        boundary_density=_boundary_density(ids, args.grid_size),
        refinement_mean=refinement_mean,
        refinement_max=refinement_max,
    )


def _grad_finite(model: nn.Module) -> bool:
    return all(param.grad is None or bool(torch.isfinite(param.grad).all().item()) for param in model.parameters())


def _count_params(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def _run(args: argparse.Namespace) -> dict[str, float | int | str]:
    _seed_everything(args.seed)
    args.root = args.data_root
    args.max_train = 0 if args.max_train_examples is None else args.max_train_examples
    args.max_test = 0 if args.max_test_examples is None else args.max_test_examples
    args.workers = args.num_workers
    device = torch.device(args.device)
    train_loader, valid_loader, classes = _build_local_loaders(args)
    model = SymbolicTropicalEmnistClassifier(
        input_dim=28 * 28,
        classes=classes,
        depth=args.depth,
        tables=args.tables,
        comparisons=args.comparisons,
        variant=args.variant,
        anchor_policy=args.anchor_policy,
        seed=args.seed,
        lut_init_std=args.lut_init_std,
        residual_scale=args.residual_scale,
        edge_bank_size=args.edge_bank_size,
        maxplus_fanout=args.maxplus_fanout,
        use_output_scaling=not args.no_output_scaling,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loss = math.nan
    train_acc = math.nan
    finite_steps = 0
    nonfinite_loss_steps = 0
    nonfinite_grad_steps = 0
    for _epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                nonfinite_loss_steps += 1
                continue
            loss.backward()
            if not _grad_finite(model):
                nonfinite_grad_steps += 1
                continue
            optimizer.step()
            finite_steps += 1
            total_loss += float(loss.item()) * labels.numel()
            total_correct += int((logits.argmax(dim=-1) == labels).sum().item())
            total_seen += int(labels.numel())
        train_loss = total_loss / max(1, total_seen)
        train_acc = total_correct / max(1, total_seen)
    valid = _eval(model, valid_loader, device)
    refinement = _refinement_probe(model, train_loader, args, device=device)
    return {
        "variant": args.variant,
        "depth": args.depth,
        "tables": args.tables,
        "comparisons": args.comparisons,
        "anchor_policy": args.anchor_policy,
        "edge_bank_size": args.edge_bank_size,
        "maxplus_fanout": args.maxplus_fanout,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "train_examples": _loader_examples(train_loader),
        "valid_examples": _loader_examples(valid_loader),
        "total_params": _count_params(model),
        "train_loss": train_loss,
        "train_acc": train_acc,
        "valid_loss": valid.loss,
        "valid_acc": valid.acc,
        "unique_signatures": refinement.unique_signatures,
        "signature_entropy": refinement.signature_entropy,
        "connected_components": refinement.connected_components,
        "boundary_density": refinement.boundary_density,
        "refinement_mean": refinement.refinement_mean,
        "refinement_max": refinement.refinement_max,
        "route_entropy": valid.route_entropy,
        "route_persistence": valid.route_persistence,
        "finite_steps": finite_steps,
        "nonfinite_loss_steps": nonfinite_loss_steps,
        "nonfinite_grad_steps": nonfinite_grad_steps,
        "seed": args.seed,
    }


def _write_csv(path: Path, row: dict[str, float | int | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--tables", type=int, default=64)
    parser.add_argument("--comparisons", type=int, default=6)
    parser.add_argument("--anchor-policy", choices=PAIRWISE_ANCHOR_POLICIES, default="permuted")
    parser.add_argument("--edge-bank-size", type=int, default=4)
    parser.add_argument("--maxplus-fanout", type=int, default=4)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--lut-init-std", type=float, default=0.0)
    parser.add_argument("--no-output-scaling", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-test-examples", type=int, default=None)
    parser.add_argument("--split", default="balanced")
    parser.add_argument("--grid-size", type=int, default=96)
    parser.add_argument("--plane-span", type=float, default=3.0)
    parser.add_argument("--pca-samples", type=int, default=4096)
    parser.add_argument("--probe-batch-size", type=int, default=2048)
    parser.add_argument("--skip-refinement-probe", action="store_true")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    row = _run(args)
    _write_csv(args.out, row)
    print(
        "variant={variant} depth={depth} valid_loss={valid_loss:.6f} valid_acc={valid_acc:.6f} "
        "refinement={refinement_mean:.3f} params={total_params}".format(**row),
        flush=True,
    )


if __name__ == "__main__":
    main()
