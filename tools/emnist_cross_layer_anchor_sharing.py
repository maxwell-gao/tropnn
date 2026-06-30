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

from ..layers import PAIRWISE_ANCHOR_POLICIES, PairwiseLUT
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

SharingMode = Literal["none", "full", "partial_25", "partial_50", "partial_75"]


@dataclass(frozen=True)
class CrossLayerAnchorRow:
    sharing_mode: str
    sharing_fraction: float
    anchor_policy: str
    depth: int
    tables: int
    comparisons: int
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
    route_persistence: float
    relation_flip_rate: float
    table_contribution_correlation: float
    generator_reuse_entropy: float
    generator_reuse_unique_fraction: float
    generator_reuse_max: int
    figure_path: str


@dataclass(frozen=True)
class EvalStats:
    loss: float
    acc: float
    route_entropy: float
    route_persistence: float
    relation_flip_rate: float
    table_contribution_correlation: float


class CrossLayerAnchorSharingClassifier(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        classes: int,
        depth: int,
        tables: int,
        comparisons: int,
        backend: str,
        anchor_policy: str,
        sharing_mode: SharingMode,
        seed: int,
        lut_init_std: float,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.depth = int(depth)
        self.tables = int(tables)
        self.comparisons = int(comparisons)
        self.table_size = 1 << self.comparisons
        self.anchor_policy = anchor_policy
        self.sharing_mode = sharing_mode
        self.sharing_fraction = _sharing_fraction(sharing_mode)
        self.blocks = nn.ModuleList(
            [
                PairwiseLUT(
                    input_dim,
                    input_dim,
                    tables=tables,
                    comparisons=comparisons,
                    seed=seed + layer,
                    anchor_seed=seed + layer,
                    backend=backend,  # type: ignore[arg-type]
                    anchor_policy=anchor_policy,
                    lut_init_std=lut_init_std,
                )
                for layer in range(depth)
            ]
        )
        self.readout = PairwiseLUT(
            input_dim,
            classes,
            tables=tables,
            comparisons=comparisons,
            seed=seed + depth,
            anchor_seed=seed + depth,
            backend=backend,  # type: ignore[arg-type]
            anchor_policy=anchor_policy,
            lut_init_std=lut_init_std,
        )
        self.register_buffer("shared_position_mask", _make_shared_position_mask(tables, comparisons, self.sharing_fraction, seed=seed + 7919))
        self._apply_cross_layer_sharing(input_dim, tables, comparisons, backend, anchor_policy, seed, lut_init_std)
        self.last_routes: list[Tensor] = []
        self.last_route_bits: list[Tensor] = []
        self.last_table_norms: list[Tensor] = []

    def forward(self, x: Tensor) -> Tensor:
        y = x.flatten(1)
        routes: list[Tensor] = []
        route_bits: list[Tensor] = []
        table_norms: list[Tensor] = []
        for block in self.blocks:
            output, route = block.compute(y.unsqueeze(1), compute_dtype=torch.float32, training=self.training)
            y = y + output.squeeze(1)
            indices = route.indices.squeeze(1) if route.indices.ndim == 3 else route.indices
            margins = route.margins.squeeze(1) if route.margins.ndim == 4 else route.margins
            routes.append(indices.detach())
            route_bits.append((margins > 0).detach().to(torch.int16))
            if not self.training:
                table_norms.append(_table_contribution_norm(block, indices))
        logits, _ = self.readout.compute(y.unsqueeze(1), compute_dtype=torch.float32, training=self.training)
        logits = logits.squeeze(1)
        self.last_routes = routes
        self.last_route_bits = route_bits
        self.last_table_norms = table_norms
        return logits

    def generator_reuse_stats(self) -> tuple[float, float, int]:
        anchors = torch.stack([block.anchors.detach().cpu() for block in self.blocks], dim=0)
        edge_ids = anchors[..., 0] * self.input_dim + anchors[..., 1]
        total = max(1, int(edge_ids.numel()))
        _, counts = torch.unique(edge_ids.reshape(-1), return_counts=True)
        probs = counts.float() / counts.sum().float().clamp_min(1.0)
        entropy = float((-(probs * probs.log()).sum() / math.log(max(2, total))).item())
        return entropy, float(counts.numel() / total), int(counts.max().item())

    def _apply_cross_layer_sharing(self, input_dim: int, tables: int, comparisons: int, backend: str, anchor_policy: str, seed: int, lut_init_std: float) -> None:
        if self.sharing_fraction <= 0.0 or not self.blocks:
            return
        template = PairwiseLUT(
            input_dim,
            input_dim,
            tables=tables,
            comparisons=comparisons,
            seed=seed + 104729,
            anchor_seed=seed + 104729,
            backend=backend,  # type: ignore[arg-type]
            anchor_policy=anchor_policy,
            lut_init_std=lut_init_std,
        )
        shared = template.anchors.detach().clone()
        mask = self.shared_position_mask.cpu()
        with torch.no_grad():
            for block in self.blocks:
                anchors = block.anchors.detach().clone().cpu()
                anchors[mask] = shared[mask]
                block.anchors.copy_(anchors.to(device=block.anchors.device))


def _sharing_fraction(mode: SharingMode) -> float:
    if mode == "none":
        return 0.0
    if mode == "full":
        return 1.0
    if mode == "partial_25":
        return 0.25
    if mode == "partial_50":
        return 0.50
    if mode == "partial_75":
        return 0.75
    raise ValueError(f"unknown sharing mode {mode!r}")


def _make_shared_position_mask(tables: int, comparisons: int, fraction: float, *, seed: int) -> Tensor:
    total = tables * comparisons
    count = int(round(total * float(fraction)))
    mask = torch.zeros(total, dtype=torch.bool)
    if count > 0:
        gen = torch.Generator(device="cpu").manual_seed(seed)
        mask[torch.randperm(total, generator=gen)[:count]] = True
    return mask.view(tables, comparisons)


@torch.no_grad()
def _table_contribution_norm(layer: PairwiseLUT, indices: Tensor) -> Tensor:
    lut = layer.lut_payload(dtype=torch.float32, device=indices.device).detach().reshape(layer.tables * layer.table_size, layer.output_dim)
    values: list[Tensor] = []
    for table in range(layer.tables):
        rows = table * layer.table_size + indices[:, table].long()
        values.append(lut.index_select(0, rows).norm(dim=-1))
    return torch.stack(values, dim=1)


def _route_entropy(routes: list[Tensor], table_size: int) -> float:
    if not routes:
        return 0.0
    values: list[float] = []
    for route in routes:
        counts = torch.bincount(route.reshape(-1).long().cpu(), minlength=table_size).float()
        probs = counts / counts.sum().clamp_min(1.0)
        nz = probs > 0
        values.append(float((-(probs[nz] * probs[nz].log()).sum() / math.log(table_size)).item()))
    return sum(values) / len(values)


def _route_persistence(routes: list[Tensor]) -> float:
    if len(routes) < 2:
        return 0.0
    return sum(float((left == right).float().mean().item()) for left, right in zip(routes, routes[1:])) / (len(routes) - 1)


def _relation_flip_rate(bits: list[Tensor], shared_mask: Tensor) -> float:
    if len(bits) < 2:
        return 0.0
    mask = shared_mask.to(device=bits[0].device)
    if not bool(mask.any().item()):
        mask = torch.ones_like(mask, dtype=torch.bool)
    values: list[float] = []
    for left, right in zip(bits, bits[1:]):
        flips = left[:, mask] != right[:, mask]
        values.append(float(flips.float().mean().item()))
    return sum(values) / len(values)


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
        z = x / std
        corr = z.T @ z / max(1, z.shape[0] - 1)
        offdiag = corr[~torch.eye(corr.shape[0], dtype=torch.bool)]
        values.append(float(offdiag.abs().mean().item()))
    return sum(values) / max(1, len(values))


def _grad_finite(model: nn.Module) -> bool:
    return all(param.grad is None or bool(torch.isfinite(param.grad).all().item()) for param in model.parameters())


def _train(model: CrossLayerAnchorSharingClassifier, train_loader, args: argparse.Namespace, *, device: torch.device) -> tuple[float, float, int, int, int]:
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


def _eval(model: CrossLayerAnchorSharingClassifier, loader, *, device: torch.device) -> EvalStats:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    entropies: list[float] = []
    persistences: list[float] = []
    flips: list[float] = []
    correlations: list[float] = []
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
            entropies.append(_route_entropy(model.last_routes, model.table_size))
            persistences.append(_route_persistence(model.last_routes))
            flips.append(_relation_flip_rate(model.last_route_bits, model.shared_position_mask))
            correlations.append(_table_contribution_correlation(model.last_table_norms))
    return EvalStats(
        loss=total_loss / max(1, total_seen),
        acc=total_correct / max(1, total_seen),
        route_entropy=sum(entropies) / max(1, len(entropies)),
        route_persistence=sum(persistences) / max(1, len(persistences)),
        relation_flip_rate=sum(flips) / max(1, len(flips)),
        table_contribution_correlation=sum(correlations) / max(1, len(correlations)),
    )


def _batch_signatures(model: CrossLayerAnchorSharingClassifier, points: Tensor, *, device: torch.device, batch_size: int) -> list[Tensor]:
    outputs: list[list[Tensor]] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, points.shape[0], batch_size):
            model(points[start : start + batch_size].to(device))
            sigs = [route.cpu().to(torch.int16).reshape(route.shape[0], -1) for route in model.last_routes]
            if not outputs:
                outputs = [[] for _ in sigs]
            for idx, sig in enumerate(sigs):
                outputs[idx].append(sig)
    return [torch.cat(parts, dim=0) for parts in outputs]


def run(args: argparse.Namespace) -> CrossLayerAnchorRow:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    train_loader, valid_loader, classes = _build_local_loaders(args)
    model = CrossLayerAnchorSharingClassifier(
        input_dim=28 * 28,
        classes=classes,
        depth=args.depth,
        tables=args.tables,
        comparisons=args.comparisons,
        backend=args.backend,
        anchor_policy=args.anchor_policy,
        sharing_mode=args.sharing_mode,
        seed=args.seed,
        lut_init_std=args.lut_init_std,
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
    reuse_entropy, reuse_unique_fraction, reuse_max = model.generator_reuse_stats()
    fig = Path(args.figure_dir) / f"{args.anchor_policy}_{args.sharing_mode}_L{args.depth}_s{args.seed}.png"
    written_fig = _save_figure(ids, pred, args.grid_size, fig)
    return CrossLayerAnchorRow(
        sharing_mode=args.sharing_mode,
        sharing_fraction=model.sharing_fraction,
        anchor_policy=args.anchor_policy,
        depth=args.depth,
        tables=args.tables,
        comparisons=args.comparisons,
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
        route_persistence=valid.route_persistence,
        relation_flip_rate=valid.relation_flip_rate,
        table_contribution_correlation=valid.table_contribution_correlation,
        generator_reuse_entropy=reuse_entropy,
        generator_reuse_unique_fraction=reuse_unique_fraction,
        generator_reuse_max=reuse_max,
        figure_path=str(written_fig),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="EMNIST balanced cross-layer Pairwise anchor sharing refinement probe.")
    parser.add_argument("--root", default="data/emnist")
    parser.add_argument("--split", default="balanced")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--backend", choices=("auto", "torch", "tilelang", "triton"), default="tilelang")
    parser.add_argument("--sharing-mode", choices=("none", "full", "partial_25", "partial_50", "partial_75"), default="none")
    parser.add_argument("--anchor-policy", choices=PAIRWISE_ANCHOR_POLICIES, default="permuted")
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
    parser.add_argument("--figure-dir", default="results/cross_layer_anchor_sharing/figures")
    parser.add_argument("--out", default="results/cross_layer_anchor_sharing/summary.csv")
    args = parser.parse_args()

    row = run(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(row).keys()))
        writer.writeheader()
        writer.writerow(asdict(row))
    print(
        f"sharing={row.sharing_mode} anchor={row.anchor_policy} L{row.depth} "
        f"valid_loss={row.valid_loss:.4f} valid_acc={row.valid_acc:.4f} refine={row.refinement_mean:.3f} "
        f"routeH={row.route_entropy:.3f} persist={row.route_persistence:.3f} flip={row.relation_flip_rate:.3f} "
        f"corr={row.table_contribution_correlation:.3f} reuseH={row.generator_reuse_entropy:.3f} "
        f"nonfinite_loss={row.nonfinite_loss_steps} nonfinite_grad={row.nonfinite_grad_steps}",
        flush=True,
    )
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
