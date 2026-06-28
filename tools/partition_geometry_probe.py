from __future__ import annotations

import argparse
import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .emnist_payload_dtype_sweep import _build_local_loaders, _eval_model
from .emnist_route_conditioned import RouteConditionedEmnistClassifier

ModelKind = Literal["mlp", "pairwise_plain", "pairwise_sparse_mixing", "pairwise_anchor_transform"]


@dataclass(frozen=True)
class ProbeRow:
    model: str
    depth: int
    hidden_dim: int
    tables: int
    comparisons: int
    grid_size: int
    epochs: int
    params: int
    train_loss: float
    train_acc: float
    valid_loss: float
    valid_acc: float
    unique_signatures: int
    signature_entropy: float
    connected_components: int
    boundary_density: float
    refinement_mean: float
    refinement_max: int
    interpolation_flips_mean: float
    normal_effective_rank: float
    figure_path: str


class ReLUMlpWithSignatures(nn.Module):
    def __init__(self, *, input_dim: int, hidden_dim: int, classes: int, depth: int, seed: int) -> None:
        super().__init__()
        torch.manual_seed(seed)
        layers: list[nn.Linear] = []
        if depth <= 1:
            layers.append(nn.Linear(input_dim, classes))
        else:
            layers.append(nn.Linear(input_dim, hidden_dim))
            for _ in range(depth - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Linear(hidden_dim, classes))
        self.layers = nn.ModuleList(layers)
        self.hidden_count = max(0, len(layers) - 1)
        self.last_signatures: list[Tensor] = []

    def forward(self, x: Tensor) -> Tensor:
        y = x.flatten(1)
        signatures: list[Tensor] = []
        for idx, layer in enumerate(self.layers):
            z = layer(y)
            if idx + 1 < len(self.layers):
                signatures.append((z > 0).detach())
                y = F.relu(z)
            else:
                y = z
        self.last_signatures = signatures
        return y


def _build_probe_model(args: argparse.Namespace, classes: int) -> nn.Module:
    if args.model == "mlp":
        return ReLUMlpWithSignatures(input_dim=28 * 28, hidden_dim=args.hidden_dim, classes=classes, depth=args.depth, seed=args.seed)
    variant = {
        "pairwise_plain": "plain",
        "pairwise_sparse_mixing": "sparse_mixing",
        "pairwise_anchor_transform": "anchor_transform",
    }[args.model]
    return RouteConditionedEmnistClassifier(
        input_dim=28 * 28,
        num_classes=classes,
        depth=args.depth,
        tables=args.tables,
        comparisons=args.comparisons,
        seed=args.seed,
        backend=args.backend,
        anchor_policy=args.anchor_policy,
        lut_init_std=args.lut_init_std,
        variant=variant,  # type: ignore[arg-type]
        mix_strength=args.mix_strength,
    )


def _train(model: nn.Module, train_loader, args: argparse.Namespace, *, device: torch.device) -> tuple[float, float]:
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
            loss.backward()
            opt.step()
            batch = int(y.numel())
            total_loss += float(loss.detach().item()) * batch
            total_correct += int((logits.argmax(dim=-1) == y).sum().item())
            total_seen += batch
        last_loss = total_loss / max(1, total_seen)
        last_acc = total_correct / max(1, total_seen)
    return last_loss, last_acc


def _collect_dataset_tensor(loader) -> tuple[Tensor, Tensor]:
    dataset = loader.dataset
    if hasattr(dataset, "tensors"):
        x, y = dataset.tensors
        return x.float(), y.long()
    xs: list[Tensor] = []
    ys: list[Tensor] = []
    for x, y in loader:
        xs.append(x.cpu().float())
        ys.append(y.cpu().long())
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def _pca_plane(x: Tensor, *, limit: int) -> tuple[Tensor, Tensor, Tensor]:
    flat = x.reshape(x.shape[0], -1).float()
    if limit > 0:
        flat = flat[:limit]
    center = flat.mean(dim=0)
    centered = flat - center
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    return center, vh[0], vh[1]


def _grid(center: Tensor, u: Tensor, v: Tensor, *, grid_size: int, span: float) -> tuple[Tensor, Tensor, Tensor]:
    coords = torch.linspace(-span, span, grid_size)
    uu, vv = torch.meshgrid(coords, coords, indexing="ij")
    points = center.view(1, -1) + uu.reshape(-1, 1) * u.view(1, -1) + vv.reshape(-1, 1) * v.view(1, -1)
    return points.clamp(-1.0, 1.0), uu, vv


def _batch_signatures(model: nn.Module, points: Tensor, *, device: torch.device, batch_size: int) -> list[Tensor]:
    outputs: list[list[Tensor]] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, points.shape[0], batch_size):
            x = points[start : start + batch_size].to(device)
            model(x)
            if isinstance(model, ReLUMlpWithSignatures):
                sigs = [sig.cpu().to(torch.int16) for sig in model.last_signatures]
            else:
                sigs = [sig.cpu().to(torch.int16) for sig in model.last_routes]
            if not outputs:
                outputs = [[] for _ in sigs]
            for idx, sig in enumerate(sigs):
                outputs[idx].append(sig.reshape(sig.shape[0], -1))
    return [torch.cat(parts, dim=0) for parts in outputs]


def _signature_ids(signatures: list[Tensor], upto: int | None = None) -> Tensor:
    if not signatures:
        return torch.zeros(0, dtype=torch.long)
    use = signatures if upto is None else signatures[:upto]
    if not use:
        return torch.zeros(signatures[0].shape[0], dtype=torch.long)
    cols = []
    for sig in use:
        cols.append(sig.reshape(sig.shape[0], -1).to(torch.long))
    mat = torch.cat(cols, dim=1)
    _, inv = torch.unique(mat, dim=0, return_inverse=True)
    return inv.long()


def _entropy(ids: Tensor) -> float:
    if ids.numel() == 0:
        return 0.0
    counts = torch.bincount(ids).float()
    probs = counts / counts.sum().clamp_min(1.0)
    nz = probs > 0
    return float((-(probs[nz] * probs[nz].log()).sum() / math.log(max(2, int(counts.numel())))).item())


def _connected_components(ids: Tensor, grid_size: int) -> int:
    arr = ids.reshape(grid_size, grid_size).numpy()
    seen = np.zeros_like(arr, dtype=bool)
    components = 0
    for i in range(grid_size):
        for j in range(grid_size):
            if seen[i, j]:
                continue
            components += 1
            value = arr[i, j]
            stack = [(i, j)]
            seen[i, j] = True
            while stack:
                x, y = stack.pop()
                for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                    if 0 <= nx < grid_size and 0 <= ny < grid_size and not seen[nx, ny] and arr[nx, ny] == value:
                        seen[nx, ny] = True
                        stack.append((nx, ny))
    return components


def _boundary_density(ids: Tensor, grid_size: int) -> float:
    arr = ids.reshape(grid_size, grid_size)
    right = (arr[:, 1:] != arr[:, :-1]).float().mean()
    down = (arr[1:, :] != arr[:-1, :]).float().mean()
    return float(((right + down) * 0.5).item())


def _refinement(signatures: list[Tensor]) -> tuple[float, int]:
    if len(signatures) < 2:
        return 0.0, 0
    values: list[int] = []
    for layer in range(1, len(signatures)):
        parent = _signature_ids(signatures, upto=layer)
        child = _signature_ids(signatures, upto=layer + 1)
        mapping: dict[int, set[int]] = {}
        for p, c in zip(parent.tolist(), child.tolist()):
            mapping.setdefault(p, set()).add(c)
        values.extend(len(v) for v in mapping.values())
    return sum(values) / max(1, len(values)), max(values) if values else 0


def _interpolation_flips(model: nn.Module, x: Tensor, y: Tensor, *, device: torch.device, samples: int, pairs: int, batch_size: int) -> float:
    if x.shape[0] < 2 or pairs <= 0:
        return 0.0
    gen = torch.Generator(device="cpu").manual_seed(17)
    flips: list[int] = []
    for _ in range(pairs):
        i = int(torch.randint(0, x.shape[0], (1,), generator=gen).item())
        different = torch.where(y != y[i])[0]
        if different.numel() == 0:
            j = int(torch.randint(0, x.shape[0], (1,), generator=gen).item())
        else:
            j = int(different[torch.randint(0, different.numel(), (1,), generator=gen)].item())
        t = torch.linspace(0.0, 1.0, samples).view(-1, 1)
        pts = (1.0 - t) * x[i].reshape(1, -1) + t * x[j].reshape(1, -1)
        sigs = _batch_signatures(model, pts, device=device, batch_size=batch_size)
        ids = _signature_ids(sigs)
        flips.append(int((ids[1:] != ids[:-1]).sum().item()))
    return sum(flips) / max(1, len(flips))


def _normal_effective_rank(model: nn.Module) -> float:
    if not isinstance(model, ReLUMlpWithSignatures):
        return 0.0
    normals: list[Tensor] = []
    prefix = torch.eye(model.layers[0].in_features, device=next(model.parameters()).device)
    for idx, layer in enumerate(model.layers[:-1]):
        w = layer.weight.detach()
        n = w @ prefix
        normals.append(n.cpu())
        prefix = w @ prefix
    if not normals:
        return 0.0
    mat = torch.cat(normals, dim=0).float()
    cov = mat @ mat.T
    eig = torch.linalg.eigvalsh(cov).clamp_min(0)
    if float(eig.sum().item()) <= 0.0:
        return 0.0
    return float((eig.sum().square() / eig.square().sum().clamp_min(1e-12)).item())


def _save_figure(ids: Tensor, labels: Tensor, grid_size: int, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    arr = ids.reshape(grid_size, grid_size).numpy().T
    lab = labels.reshape(grid_size, grid_size).numpy().T
    palette = np.array(
        [
            [31, 119, 180],
            [255, 127, 14],
            [44, 160, 44],
            [214, 39, 40],
            [148, 103, 189],
            [140, 86, 75],
            [227, 119, 194],
            [127, 127, 127],
            [188, 189, 34],
            [23, 190, 207],
            [174, 199, 232],
            [255, 187, 120],
            [152, 223, 138],
            [255, 152, 150],
            [197, 176, 213],
            [196, 156, 148],
            [247, 182, 210],
            [199, 199, 199],
            [219, 219, 141],
            [158, 218, 229],
        ],
        dtype=np.uint8,
    )
    sig_rgb = palette[np.mod(arr, len(palette))]
    lab_rgb = palette[np.mod(lab, len(palette))]
    gap = np.full((grid_size, 4, 3), 255, dtype=np.uint8)
    image = np.concatenate([sig_rgb, gap, lab_rgb], axis=1)
    ppm = out.with_suffix(".ppm")
    with ppm.open("wb") as handle:
        handle.write(f"P6\n{image.shape[1]} {image.shape[0]}\n255\n".encode("ascii"))
        handle.write(np.ascontiguousarray(image).tobytes())
    return ppm


def run(args: argparse.Namespace) -> ProbeRow:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    train_loader, valid_loader, classes = _build_local_loaders(args)
    model = _build_probe_model(args, classes).to(device)
    train_loss, train_acc = _train(model, train_loader, args, device=device)
    valid_loss, valid_acc = _eval_model(model, valid_loader, device=device)
    x_train, y_train = _collect_dataset_tensor(train_loader)
    center, u, v = _pca_plane(x_train, limit=args.pca_samples)
    points, _uu, _vv = _grid(center, u, v, grid_size=args.grid_size, span=args.plane_span)
    signatures = _batch_signatures(model, points, device=device, batch_size=args.probe_batch_size)
    full_ids = _signature_ids(signatures)
    if full_ids.numel() == 0:
        full_ids = torch.zeros(points.shape[0], dtype=torch.long)
    model.eval()
    preds: list[Tensor] = []
    with torch.no_grad():
        for start in range(0, points.shape[0], args.probe_batch_size):
            logits = model(points[start : start + args.probe_batch_size].to(device))
            preds.append(logits.argmax(dim=-1).cpu())
    pred = torch.cat(preds, dim=0)
    refinement_mean, refinement_max = _refinement(signatures)
    fig_path = Path(args.figure_dir) / f"{args.model}_L{args.depth}_s{args.seed}.png"
    written_fig = _save_figure(full_ids, pred, args.grid_size, fig_path)
    return ProbeRow(
        model=args.model,
        depth=args.depth,
        hidden_dim=args.hidden_dim,
        tables=args.tables,
        comparisons=args.comparisons,
        grid_size=args.grid_size,
        epochs=args.epochs,
        params=sum(p.numel() for p in model.parameters()),
        train_loss=train_loss,
        train_acc=train_acc,
        valid_loss=valid_loss,
        valid_acc=valid_acc,
        unique_signatures=int(torch.unique(full_ids).numel()),
        signature_entropy=_entropy(full_ids),
        connected_components=_connected_components(full_ids, args.grid_size),
        boundary_density=_boundary_density(full_ids, args.grid_size),
        refinement_mean=refinement_mean,
        refinement_max=refinement_max,
        interpolation_flips_mean=_interpolation_flips(model, x_train.reshape(x_train.shape[0], -1), y_train, device=device, samples=args.interp_samples, pairs=args.interp_pairs, batch_size=args.probe_batch_size),
        normal_effective_rank=_normal_effective_rank(model),
        figure_path=str(written_fig),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe recursive spatial partitioning on EMNIST PCA planes.")
    parser.add_argument("--root", default="data/emnist")
    parser.add_argument("--split", default="balanced")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model", choices=("mlp", "pairwise_plain", "pairwise_sparse_mixing", "pairwise_anchor_transform"), default="mlp")
    parser.add_argument("--backend", choices=("auto", "torch", "tilelang", "triton"), default="tilelang")
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--tables", type=int, default=64)
    parser.add_argument("--comparisons", type=int, default=6)
    parser.add_argument("--lut-init-std", type=float, default=0.0)
    parser.add_argument("--anchor-policy", default="permuted")
    parser.add_argument("--mix-strength", type=float, default=0.125)
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
    parser.add_argument("--figure-dir", default="results/partition_geometry/figures")
    parser.add_argument("--out", default="results/partition_geometry/summary.csv")
    args = parser.parse_args()

    row = run(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(row).keys()))
        writer.writeheader()
        writer.writerow(asdict(row))
    print(
        f"model={row.model} L{row.depth} valid_loss={row.valid_loss:.4f} valid_acc={row.valid_acc:.4f} "
        f"unique={row.unique_signatures} components={row.connected_components} boundary={row.boundary_density:.4f} "
        f"refine={row.refinement_mean:.3f} flips={row.interpolation_flips_mean:.2f} fig={row.figure_path}",
        flush=True,
    )
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
