from __future__ import annotations

import argparse
import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..layers.surrogate import ste_heaviside
from .emnist_payload_dtype_sweep import _build_local_loaders, _loader_examples
from .partition_geometry_probe import (
    _boundary_density,
    _collect_dataset_tensor,
    _connected_components,
    _entropy,
    _grid,
    _interpolation_flips,
    _pca_plane,
    _refinement,
    _save_figure,
    _signature_ids,
)

Mechanism = Literal["shared", "braid", "normal", "braid_normal"]
PoolKind = Literal["local", "expander", "sorting"]


@dataclass(frozen=True)
class BraidGeneratorRow:
    mechanism: str
    generator_pool: str
    depth: int
    tables: int
    comparisons: int
    generator_pool_size: int
    exchange_residual_scale: float
    epochs: int
    batch_size: int
    grid_size: int
    params: int
    train_examples: int
    valid_examples: int
    train_loss: float
    train_acc: float
    valid_loss: float
    valid_acc: float
    finite_loss_steps: int
    nonfinite_loss_steps: int
    nonfinite_grad_steps: int
    unique_signatures: int
    signature_entropy: float
    connected_components: int
    boundary_density: float
    refinement_mean: float
    refinement_max: int
    interpolation_flips_mean: float
    route_entropy: float
    route_transition: float
    route_unique_fraction: float
    table_contribution_correlation: float
    generator_reuse_entropy: float
    generator_reuse_fraction: float
    generator_reuse_max: int
    figure_path: str


@dataclass(frozen=True)
class EvalStats:
    loss: float
    acc: float
    route_entropy: float
    route_transition: float
    route_unique_fraction: float
    table_contribution_correlation: float


class GeneratorLUTBlock(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        tables: int,
        comparisons: int,
        edges: Tensor,
        layer_index: int,
        mechanism: Mechanism,
        seed: int,
        lut_init_std: float,
        exchange_residual_scale: float,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.tables = int(tables)
        self.comparisons = int(comparisons)
        self.table_size = 1 << self.comparisons
        self.mechanism = mechanism
        self.exchange = mechanism in {"braid", "braid_normal"}
        self.normal_form = mechanism in {"normal", "braid_normal"}
        self.exchange_residual_scale = float(exchange_residual_scale) if self.exchange and self.output_dim == self.input_dim else 0.0
        self.scale = 1.0 / math.sqrt(self.tables)
        self.register_buffer("edges", edges.long().contiguous())
        self.register_buffer("generator_ids", _assign_generator_ids(len(edges), self.tables, self.comparisons, layer_index=layer_index, seed=seed))
        self.register_buffer("powers", 2 ** torch.arange(self.comparisons, dtype=torch.long))
        self.register_buffer("canonical_order", self.generator_ids.argsort(dim=-1))
        self.thresholds = nn.Parameter(torch.zeros(self.tables, self.comparisons))
        if self.normal_form:
            self.lut = nn.Parameter(torch.randn(self.table_size, self.output_dim) * float(lut_init_std))
            self.table_bias = nn.Parameter(torch.zeros(self.tables, self.output_dim))
        else:
            self.lut = nn.Parameter(torch.randn(self.tables, self.table_size, self.output_dim) * float(lut_init_std))
            self.table_bias = None
        self.last_table_norms: Tensor | None = None

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        route_bits, margins, exchanged = self._route(x)
        lookup_index = self._lookup_index(route_bits)
        payload = self._lookup_payload(lookup_index)
        if self.training:
            payload = payload + self._ste_payload_delta(route_bits, margins, lookup_index).to(payload.dtype)
        self.last_table_norms = payload.detach().float().norm(dim=-1)
        output = payload.sum(dim=1) * self.scale
        if self.exchange_residual_scale:
            output = output + (exchanged - x) * self.exchange_residual_scale
        return output, lookup_index.detach()

    def _route(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch = x.shape[0]
        device = x.device
        generator_ids = self.generator_ids.to(device=device)
        edges = self.edges.to(device=device)
        thresholds = self.thresholds.to(device=device, dtype=x.dtype)
        bits = torch.empty(batch, self.tables, self.comparisons, device=device, dtype=x.dtype)
        margins = torch.empty(batch, self.tables, self.comparisons, device=device, dtype=x.dtype)
        coords = x
        if not self.exchange:
            selected = edges.index_select(0, generator_ids.reshape(-1)).view(self.tables, self.comparisons, 2)
            a = selected[:, :, 0].reshape(-1)
            b = selected[:, :, 1].reshape(-1)
            raw = x[:, a].view(batch, self.tables, self.comparisons) - x[:, b].view(batch, self.tables, self.comparisons)
            margins = raw - thresholds.view(1, self.tables, self.comparisons)
            bits = ste_heaviside(margins)
            return bits, margins, x
        for table in range(self.tables):
            for comp in range(self.comparisons):
                edge = edges[generator_ids[table, comp]]
                a = int(edge[0].item())
                b = int(edge[1].item())
                va = coords[:, a].clone()
                vb = coords[:, b].clone()
                margin = va - vb - thresholds[table, comp]
                bit = ste_heaviside(margin)
                margins[:, table, comp] = margin
                bits[:, table, comp] = bit
                delta = bit * (vb - va)
                coords_next = coords.clone()
                coords_next[:, a] = va + delta
                coords_next[:, b] = vb - delta
                coords = coords_next
        return bits, margins, coords

    def _lookup_index(self, bits: Tensor) -> Tensor:
        if self.normal_form:
            order = self.canonical_order.to(device=bits.device).view(1, self.tables, self.comparisons)
            bits = bits.gather(dim=-1, index=order.expand(bits.shape[0], -1, -1))
        powers = self.powers.to(device=bits.device).view(1, 1, self.comparisons)
        return (bits.to(torch.long) * powers).sum(dim=-1)

    def _lookup_payload(self, index: Tensor) -> Tensor:
        if self.normal_form:
            values = self.lut.to(device=index.device).index_select(0, index.reshape(-1)).view(index.shape[0], self.tables, self.output_dim)
            return values + self.table_bias.to(device=index.device).view(1, self.tables, self.output_dim)
        offsets = (torch.arange(self.tables, device=index.device, dtype=torch.long) * self.table_size).view(1, self.tables)
        rows = (index + offsets).reshape(-1)
        return self.lut.to(device=index.device).reshape(self.tables * self.table_size, self.output_dim).index_select(0, rows).view(index.shape[0], self.tables, self.output_dim)

    def _ste_payload_delta(self, bits: Tensor, margins: Tensor, current_index: Tensor) -> Tensor:
        bit = margins.abs().argmin(dim=-1)
        margin = margins.gather(dim=-1, index=bit.unsqueeze(-1)).squeeze(-1)
        hard = (margin > 0).to(margin.dtype)
        ste = ste_heaviside(margin) - hard
        if self.normal_form:
            neighbor_bits = bits.to(torch.long).clone()
            neighbor_bits.scatter_(dim=-1, index=bit.unsqueeze(-1), src=1 - neighbor_bits.gather(dim=-1, index=bit.unsqueeze(-1)))
            neighbor_index = self._lookup_index(neighbor_bits.to(bits.dtype))
        else:
            neighbor_index = current_index ^ (2**bit).long()
        current = self._lookup_payload(current_index).detach()
        neighbor = self._lookup_payload(neighbor_index).detach()
        return ste.unsqueeze(-1).float() * (neighbor - current).float()


class BraidGeneratorEmnistClassifier(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        classes: int,
        depth: int,
        tables: int,
        comparisons: int,
        generator_pool: PoolKind,
        generator_pool_size: int,
        mechanism: Mechanism,
        seed: int,
        lut_init_std: float,
        exchange_residual_scale: float,
    ) -> None:
        super().__init__()
        self.depth = int(depth)
        self.tables = int(tables)
        self.comparisons = int(comparisons)
        self.table_size = 1 << self.comparisons
        self.mechanism = mechanism
        edges = _make_generator_pool(input_dim, generator_pool, generator_pool_size, seed=seed)
        self.register_buffer("generator_edges", edges.long().contiguous())
        self.blocks = nn.ModuleList(
            [
                GeneratorLUTBlock(
                    input_dim=input_dim,
                    output_dim=input_dim,
                    tables=tables,
                    comparisons=comparisons,
                    edges=edges,
                    layer_index=layer,
                    mechanism=mechanism,
                    seed=seed + layer,
                    lut_init_std=lut_init_std,
                    exchange_residual_scale=exchange_residual_scale,
                )
                for layer in range(depth)
            ]
        )
        self.readout = GeneratorLUTBlock(
            input_dim=input_dim,
            output_dim=classes,
            tables=tables,
            comparisons=comparisons,
            edges=edges,
            layer_index=depth,
            mechanism=mechanism,
            seed=seed + depth,
            lut_init_std=lut_init_std,
            exchange_residual_scale=0.0,
        )
        self.last_routes: list[Tensor] = []
        self.last_table_norms: list[Tensor] = []

    def forward(self, x: Tensor) -> Tensor:
        y = x.flatten(1)
        routes: list[Tensor] = []
        table_norms: list[Tensor] = []
        for block in self.blocks:
            delta, route = block(y)
            y = y + delta
            routes.append(route.detach())
            if block.last_table_norms is not None:
                table_norms.append(block.last_table_norms)
        logits, _ = self.readout(y)
        self.last_routes = routes
        self.last_table_norms = table_norms
        return logits

    def generator_reuse_stats(self) -> tuple[float, float, int]:
        ids = [block.generator_ids.reshape(-1).cpu() for block in self.blocks]
        ids.append(self.readout.generator_ids.reshape(-1).cpu())
        flat = torch.cat(ids)
        counts = torch.bincount(flat, minlength=int(self.generator_edges.shape[0])).float()
        probs = counts / counts.sum().clamp_min(1.0)
        nz = probs > 0
        entropy = float((-(probs[nz] * probs[nz].log()).sum() / math.log(max(2, int(counts.numel())))).item())
        return entropy, float((counts > 0).float().mean().item()), int(counts.max().item())


def _make_generator_pool(dim: int, kind: PoolKind, pool_size: int, *, seed: int) -> Tensor:
    if kind == "local":
        edges = [(i, (i + 1) % dim) for i in range(dim)]
    elif kind == "expander":
        strides = [1, 3, 7, 15, 31, 63, 127, 255]
        edges = [(i, (i + stride) % dim) for stride in strides for i in range(dim) if (i + stride) % dim != i]
    elif kind == "sorting":
        edges = _bitonic_edges(dim)
    else:
        raise ValueError(f"unknown generator pool {kind!r}")
    if not edges:
        raise ValueError("generator pool is empty")
    gen = torch.Generator(device="cpu").manual_seed(seed)
    edge_tensor = torch.tensor(edges, dtype=torch.long)
    if pool_size <= 0:
        pool_size = int(edge_tensor.shape[0])
    if edge_tensor.shape[0] >= pool_size:
        if kind == "sorting":
            return edge_tensor[:pool_size].contiguous()
        perm = torch.randperm(edge_tensor.shape[0], generator=gen)[:pool_size]
        return edge_tensor.index_select(0, perm).contiguous()
    repeats = math.ceil(pool_size / edge_tensor.shape[0])
    return edge_tensor.repeat(repeats, 1)[:pool_size].contiguous()


def _bitonic_edges(dim: int) -> list[tuple[int, int]]:
    size = 1
    while size < dim:
        size *= 2
    edges: list[tuple[int, int]] = []
    k = 2
    while k <= size:
        j = k // 2
        while j > 0:
            for i in range(size):
                other = i ^ j
                if other > i and i < dim and other < dim:
                    edges.append((i, other))
            j //= 2
        k *= 2
    return edges


def _assign_generator_ids(pool_size: int, tables: int, comparisons: int, *, layer_index: int, seed: int) -> Tensor:
    total = tables * comparisons
    base = torch.arange(total, dtype=torch.long)
    shift = (layer_index * (comparisons * 17 + 3) + seed * 13) % max(1, pool_size)
    ids = (base + shift).remainder(pool_size).view(tables, comparisons)
    return ids.contiguous()


def _route_stats(routes: list[Tensor], table_size: int) -> tuple[float, float, float]:
    if not routes:
        return 0.0, 0.0, 0.0
    entropies: list[float] = []
    transitions: list[float] = []
    unique_fracs: list[float] = []
    for route in routes:
        flat = route.reshape(-1).long()
        counts = torch.bincount(flat, minlength=table_size).float()
        probs = counts / counts.sum().clamp_min(1.0)
        nz = probs > 0
        entropies.append(float((-(probs[nz] * probs[nz].log()).sum() / math.log(table_size)).item()))
        unique_fracs.append(float((counts > 0).float().mean().item()))
    for left, right in zip(routes, routes[1:]):
        transitions.append(float((left != right).float().mean().item()))
    return sum(entropies) / len(entropies), sum(transitions) / max(1, len(transitions)), sum(unique_fracs) / len(unique_fracs)


def _table_contribution_correlation(norms: list[Tensor]) -> float:
    if not norms:
        return 0.0
    values: list[float] = []
    for norm in norms:
        x = norm.detach().float().cpu()
        if x.shape[0] < 2 or x.shape[1] < 2:
            continue
        x = x - x.mean(dim=0, keepdim=True)
        std = x.square().mean(dim=0).sqrt().clamp_min(1e-8)
        corr = (x / std).T @ (x / std) / max(1, x.shape[0] - 1)
        offdiag = corr[~torch.eye(corr.shape[0], dtype=torch.bool)]
        values.append(float(offdiag.abs().mean().item()))
    return sum(values) / max(1, len(values))


def _grad_finite(model: nn.Module) -> bool:
    return all(param.grad is None or bool(torch.isfinite(param.grad).all().item()) for param in model.parameters())


def _train(model: BraidGeneratorEmnistClassifier, train_loader, args: argparse.Namespace, *, device: torch.device) -> tuple[float, float, int, int, int]:
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    finite_loss_steps = 0
    nonfinite_loss_steps = 0
    nonfinite_grad_steps = 0
    last_loss = 0.0
    last_acc = 0.0
    for _epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            if torch.isfinite(loss):
                finite_loss_steps += 1
            else:
                nonfinite_loss_steps += 1
            loss.backward()
            if not _grad_finite(model):
                nonfinite_grad_steps += 1
            opt.step()
            batch = int(y.numel())
            total_loss += float(loss.detach().item()) * batch
            total_correct += int((logits.argmax(dim=-1) == y).sum().item())
            total_seen += batch
        last_loss = total_loss / max(1, total_seen)
        last_acc = total_correct / max(1, total_seen)
    return last_loss, last_acc, finite_loss_steps, nonfinite_loss_steps, nonfinite_grad_steps


def _eval(model: BraidGeneratorEmnistClassifier, loader, *, device: torch.device) -> EvalStats:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    route_entropies: list[float] = []
    route_transitions: list[float] = []
    route_unique_fracs: list[float] = []
    corr_values: list[float] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            batch = int(y.numel())
            total_loss += float(loss.item()) * batch
            total_correct += int((logits.argmax(dim=-1) == y).sum().item())
            total_seen += batch
            ent, trans, uniq = _route_stats(model.last_routes, model.table_size)
            route_entropies.append(ent)
            route_transitions.append(trans)
            route_unique_fracs.append(uniq)
            corr_values.append(_table_contribution_correlation(model.last_table_norms))
    return EvalStats(
        loss=total_loss / max(1, total_seen),
        acc=total_correct / max(1, total_seen),
        route_entropy=sum(route_entropies) / max(1, len(route_entropies)),
        route_transition=sum(route_transitions) / max(1, len(route_transitions)),
        route_unique_fraction=sum(route_unique_fracs) / max(1, len(route_unique_fracs)),
        table_contribution_correlation=sum(corr_values) / max(1, len(corr_values)),
    )


def _batch_signatures(model: BraidGeneratorEmnistClassifier, points: Tensor, *, device: torch.device, batch_size: int) -> list[Tensor]:
    outputs: list[list[Tensor]] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, points.shape[0], batch_size):
            x = points[start : start + batch_size].to(device)
            model(x)
            sigs = [route.cpu().to(torch.int16).reshape(route.shape[0], -1) for route in model.last_routes]
            if not outputs:
                outputs = [[] for _ in sigs]
            for idx, sig in enumerate(sigs):
                outputs[idx].append(sig)
    return [torch.cat(parts, dim=0) for parts in outputs]


def run(args: argparse.Namespace) -> BraidGeneratorRow:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    train_loader, valid_loader, classes = _build_local_loaders(args)
    model = BraidGeneratorEmnistClassifier(
        input_dim=28 * 28,
        classes=classes,
        depth=args.depth,
        tables=args.tables,
        comparisons=args.comparisons,
        generator_pool=args.generator_pool,
        generator_pool_size=args.generator_pool_size,
        mechanism=args.mechanism,
        seed=args.seed,
        lut_init_std=args.lut_init_std,
        exchange_residual_scale=args.exchange_residual_scale,
    ).to(device)
    train_loss, train_acc, finite_loss_steps, nonfinite_loss_steps, nonfinite_grad_steps = _train(model, train_loader, args, device=device)
    valid = _eval(model, valid_loader, device=device)
    x_train, y_train = _collect_dataset_tensor(train_loader)
    center, u, v = _pca_plane(x_train, limit=args.pca_samples)
    points, _uu, _vv = _grid(center, u, v, grid_size=args.grid_size, span=args.plane_span)
    signatures = _batch_signatures(model, points, device=device, batch_size=args.probe_batch_size)
    ids = _signature_ids(signatures)
    if ids.numel() == 0:
        ids = torch.zeros(points.shape[0], dtype=torch.long)
    preds: list[Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, points.shape[0], args.probe_batch_size):
            preds.append(model(points[start : start + args.probe_batch_size].to(device)).argmax(dim=-1).cpu())
    pred = torch.cat(preds, dim=0)
    refinement_mean, refinement_max = _refinement(signatures)
    reuse_entropy, reuse_fraction, reuse_max = model.generator_reuse_stats()
    fig = Path(args.figure_dir) / f"{args.mechanism}_{args.generator_pool}_L{args.depth}_s{args.seed}.png"
    written_fig = _save_figure(ids, pred, args.grid_size, fig)
    return BraidGeneratorRow(
        mechanism=args.mechanism,
        generator_pool=args.generator_pool,
        depth=args.depth,
        tables=args.tables,
        comparisons=args.comparisons,
        generator_pool_size=int(model.generator_edges.shape[0]),
        exchange_residual_scale=args.exchange_residual_scale if args.mechanism in {"braid", "braid_normal"} else 0.0,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grid_size=args.grid_size,
        params=sum(p.numel() for p in model.parameters()),
        train_examples=_loader_examples(train_loader),
        valid_examples=_loader_examples(valid_loader),
        train_loss=train_loss,
        train_acc=train_acc,
        valid_loss=valid.loss,
        valid_acc=valid.acc,
        finite_loss_steps=finite_loss_steps,
        nonfinite_loss_steps=nonfinite_loss_steps,
        nonfinite_grad_steps=nonfinite_grad_steps,
        unique_signatures=int(torch.unique(ids).numel()),
        signature_entropy=_entropy(ids),
        connected_components=_connected_components(ids, args.grid_size),
        boundary_density=_boundary_density(ids, args.grid_size),
        refinement_mean=refinement_mean,
        refinement_max=refinement_max,
        interpolation_flips_mean=_interpolation_flips(model, x_train.reshape(x_train.shape[0], -1), y_train, device=device, samples=args.interp_samples, pairs=args.interp_pairs, batch_size=args.probe_batch_size),
        route_entropy=valid.route_entropy,
        route_transition=valid.route_transition,
        route_unique_fraction=valid.route_unique_fraction,
        table_contribution_correlation=valid.table_contribution_correlation,
        generator_reuse_entropy=reuse_entropy,
        generator_reuse_fraction=reuse_fraction,
        generator_reuse_max=reuse_max,
        figure_path=str(written_fig),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="EMNIST balanced shared-generator and braid Pairwise LUT refinement probe.")
    parser.add_argument("--root", default="data/emnist")
    parser.add_argument("--split", default="balanced")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mechanism", choices=("shared", "braid", "normal", "braid_normal"), default="shared")
    parser.add_argument("--generator-pool", choices=("local", "expander", "sorting"), default="expander")
    parser.add_argument("--generator-pool-size", type=int, default=256)
    parser.add_argument("--exchange-residual-scale", type=float, default=0.25)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--tables", type=int, default=64)
    parser.add_argument("--comparisons", type=int, default=6)
    parser.add_argument("--lut-init-std", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-test", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid-size", type=int, default=96)
    parser.add_argument("--plane-span", type=float, default=4.0)
    parser.add_argument("--pca-samples", type=int, default=4096)
    parser.add_argument("--probe-batch-size", type=int, default=1024)
    parser.add_argument("--interp-samples", type=int, default=128)
    parser.add_argument("--interp-pairs", type=int, default=64)
    parser.add_argument("--figure-dir", default="results/braid_generators/figures")
    parser.add_argument("--out", default="results/braid_generators/summary.csv")
    args = parser.parse_args()

    row = run(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(row).keys()))
        writer.writeheader()
        writer.writerow(asdict(row))
    print(
        f"mechanism={row.mechanism} pool={row.generator_pool} L{row.depth} "
        f"valid_loss={row.valid_loss:.4f} valid_acc={row.valid_acc:.4f} refine={row.refinement_mean:.3f} "
        f"components={row.connected_components} route_entropy={row.route_entropy:.3f} "
        f"table_corr={row.table_contribution_correlation:.3f} reuse_entropy={row.generator_reuse_entropy:.3f} "
        f"nonfinite_loss={row.nonfinite_loss_steps} nonfinite_grad={row.nonfinite_grad_steps}",
        flush=True,
    )
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
