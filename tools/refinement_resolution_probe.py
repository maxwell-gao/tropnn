from __future__ import annotations

import argparse
import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from tropnn.tools.emnist_payload_dtype_sweep import _build_local_loaders, _eval_model
from tropnn.tools.emnist_payload_width import PayloadWidthEmnistClassifier, PayloadVariant
from tropnn.tools.partition_geometry_probe import ResidualReLUMlpWithSignatures, _pca_plane

ProbeModel = Literal[
    "residual_mlp",
    "full_vector",
    "pairwise_plain",
    "comparator_signed_margin_kc",
    "group_k16",
    "walsh_affine",
]


@dataclass(frozen=True)
class ResolutionRow:
    model: str
    depth: int
    grid_size: int
    plane_span: float
    epochs: int
    params: int
    mlp_output_dim: int
    train_loss: float
    train_acc: float
    valid_loss: float
    valid_acc: float
    unique_signatures: int
    signature_saturation: float
    connected_components: int
    boundary_density: float
    refinement_mean: float
    refinement_max: int
    parent_cells: int
    parent_support_mean: float
    parent_support_median: float
    parent_support_p10: float
    parent_singleton_fraction: float
    parent_le4_fraction: float
    tables: int
    comparisons: int
    comparator_kc: int
    comparator_write_policy: str
    payload_width: int
    write_degree: int
    walsh_lut_dtype: str
    walsh_order: int
    walsh_slope_order: int
    seed: int


def _seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _parse_grid_sizes(value: str) -> list[int]:
    sizes = [int(part) for part in value.replace(",", " ").split()]
    if not sizes:
        raise ValueError("--grid-sizes must contain at least one integer")
    if any(size < 2 for size in sizes):
        raise ValueError(f"grid sizes must be >= 2, got {sizes}")
    return sizes


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


def _grid(center: Tensor, u: Tensor, v: Tensor, *, grid_size: int, span: float) -> Tensor:
    coords = torch.linspace(-span, span, grid_size)
    uu, vv = torch.meshgrid(coords, coords, indexing="ij")
    points = center.view(1, -1) + uu.reshape(-1, 1) * u.view(1, -1) + vv.reshape(-1, 1) * v.view(1, -1)
    return points.clamp(-1.0, 1.0)


def _build_model(args: argparse.Namespace, classes: int) -> nn.Module:
    if args.model == "residual_mlp":
        return ResidualReLUMlpWithSignatures(
            input_dim=28 * 28,
            hidden_dim=_residual_mlp_output_dim(args),
            classes=classes,
            depth=args.depth,
            seed=args.seed,
            residual_scale=args.mlp_residual_scale,
        )

    variant: PayloadVariant
    if args.model in {"full_vector", "pairwise_plain"}:
        variant = "full_vector"
    elif args.model == "comparator_signed_margin_kc":
        variant = "comparator_signed_margin_kc"
    elif args.model == "group_k16":
        variant = "group_k16"
    elif args.model == "walsh_affine":
        variant = "walsh_affine"
    else:
        raise ValueError(f"unknown model={args.model!r}")

    return PayloadWidthEmnistClassifier(
        input_dim=28 * 28,
        classes=classes,
        depth=args.depth,
        tables=args.tables,
        comparisons=args.comparisons,
        variant=variant,
        anchor_policy=args.anchor_policy,
        seed=args.seed,
        lut_init_std=args.lut_init_std,
        write_degree=args.write_degree,
        walsh_lut_dtype=args.walsh_lut_dtype,
        walsh_order=args.walsh_order,
        walsh_coeff_init_std=args.walsh_coeff_init_std,
        walsh_slope_order=args.walsh_slope_order,
        walsh_slope_coeff_init_std=args.walsh_slope_coeff_init_std,
        walsh_slope_generator_init_std=args.walsh_slope_generator_init_std,
        residual_scale=args.residual_scale,
        use_output_scaling=not args.no_output_scaling,
        use_min_margin_ste=not args.full_ste,
        comparator_kc=args.comparator_kc,
        comparator_write_policy=args.comparator_write_policy,
        comparator_reduction_layout="scatter",
        comparator_output_tile_size=32,
    )


def _residual_mlp_output_dim(args: argparse.Namespace) -> int:
    requested = int(args.residual_mlp_output_dim)
    if requested > 0:
        return requested
    if int(args.depth) == 16:
        return 28 * 28
    return int(args.hidden_dim)


def _train(model: nn.Module, train_loader, args: argparse.Namespace, *, device: torch.device) -> tuple[float, float]:
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    last_loss = math.nan
    last_acc = math.nan
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
        last_loss = total_loss / max(total_seen, 1)
        last_acc = total_correct / max(total_seen, 1)
    return last_loss, last_acc


@torch.no_grad()
def _batch_signatures(model: nn.Module, points: Tensor, *, device: torch.device, batch_size: int) -> list[Tensor]:
    outputs: list[list[Tensor]] = []
    model.eval()
    for start in range(0, points.shape[0], batch_size):
        batch = points[start : start + batch_size].to(device)
        model(batch)
        if isinstance(model, ResidualReLUMlpWithSignatures):
            signatures = [sig.cpu().to(torch.int16).reshape(sig.shape[0], -1) for sig in model.last_signatures]
        elif isinstance(model, PayloadWidthEmnistClassifier):
            signatures = [route.cpu().to(torch.int16).reshape(route.shape[0], -1) for route in model.last_routes]
        else:
            raise TypeError(f"unsupported model type {type(model)!r}")
        if not outputs:
            outputs = [[] for _ in signatures]
        for idx, signature in enumerate(signatures):
            outputs[idx].append(signature)
    return [torch.cat(parts, dim=0) for parts in outputs]


def _row_hash(signature: Tensor) -> Tensor:
    flat = signature.reshape(signature.shape[0], -1).to(torch.int64)
    h = torch.full((flat.shape[0],), 0x1234_5678_1357_2468, dtype=torch.int64)
    for start in range(0, flat.shape[1], 256):
        chunk = flat[:, start : start + 256] + 32768
        weights = torch.arange(start + 1, start + 1 + chunk.shape[1], dtype=torch.int64).mul(1009).add(9176)
        h = h.mul(1_000_003).add((chunk * weights.view(1, -1)).sum(dim=1))
    return h


def _cumulative_hashes(signatures: list[Tensor]) -> list[Tensor]:
    hashes: list[Tensor] = []
    current: Tensor | None = None
    for signature in signatures:
        layer_hash = _row_hash(signature)
        current = layer_hash if current is None else current.mul(1_000_003).add(layer_hash)
        hashes.append(current)
    return hashes


def _dense_ids(values: Tensor) -> Tensor:
    if values.numel() == 0:
        return torch.zeros(0, dtype=torch.long)
    _unique, inverse = torch.unique(values, sorted=False, return_inverse=True)
    return inverse.long()


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


def _support_quantile(values: Tensor, q: float) -> float:
    if values.numel() == 0:
        return 0.0
    sorted_values = values.sort().values.float()
    idx = min(sorted_values.numel() - 1, max(0, int(round(q * (sorted_values.numel() - 1)))))
    return float(sorted_values[idx].item())


@dataclass(frozen=True)
class RefinementMetrics:
    unique_signatures: int
    signature_saturation: float
    connected_components: int
    boundary_density: float
    refinement_mean: float
    refinement_max: int
    parent_cells: int
    parent_support_mean: float
    parent_support_median: float
    parent_support_p10: float
    parent_singleton_fraction: float
    parent_le4_fraction: float


def _refinement_metrics(signatures: list[Tensor], *, grid_size: int) -> RefinementMetrics:
    total_points = grid_size * grid_size
    if not signatures:
        ids = torch.zeros(total_points, dtype=torch.long)
        return RefinementMetrics(
            unique_signatures=1,
            signature_saturation=1.0 / max(total_points, 1),
            connected_components=1,
            boundary_density=0.0,
            refinement_mean=0.0,
            refinement_max=0,
            parent_cells=0,
            parent_support_mean=0.0,
            parent_support_median=0.0,
            parent_support_p10=0.0,
            parent_singleton_fraction=0.0,
            parent_le4_fraction=0.0,
        )

    cumulative = _cumulative_hashes(signatures)
    final_ids = _dense_ids(cumulative[-1])
    unique_signatures = int(torch.unique(final_ids).numel())

    child_counts: list[Tensor] = []
    parent_supports: list[Tensor] = []
    for layer in range(1, len(cumulative)):
        parent_ids = _dense_ids(cumulative[layer - 1])
        child_ids = _dense_ids(cumulative[layer])
        parent_count = int(parent_ids.max().item()) + 1 if parent_ids.numel() else 0
        child_count = int(child_ids.max().item()) + 1 if child_ids.numel() else 0
        if parent_count == 0 or child_count == 0:
            continue
        pair_codes = parent_ids * child_count + child_ids
        unique_pairs = torch.unique(pair_codes)
        pair_parents = unique_pairs // child_count
        child_counts.append(torch.bincount(pair_parents, minlength=parent_count).float())
        parent_supports.append(torch.bincount(parent_ids, minlength=parent_count).float())

    if child_counts:
        all_child_counts = torch.cat(child_counts)
        all_parent_supports = torch.cat(parent_supports)
        refinement_mean = float(all_child_counts.mean().item())
        refinement_max = int(all_child_counts.max().item())
        parent_cells = int(all_parent_supports.numel())
        parent_support_mean = float(all_parent_supports.mean().item())
        parent_support_median = _support_quantile(all_parent_supports, 0.5)
        parent_support_p10 = _support_quantile(all_parent_supports, 0.1)
        parent_singleton_fraction = float((all_parent_supports <= 1).float().mean().item())
        parent_le4_fraction = float((all_parent_supports <= 4).float().mean().item())
    else:
        refinement_mean = 0.0
        refinement_max = 0
        parent_cells = 0
        parent_support_mean = 0.0
        parent_support_median = 0.0
        parent_support_p10 = 0.0
        parent_singleton_fraction = 0.0
        parent_le4_fraction = 0.0

    return RefinementMetrics(
        unique_signatures=unique_signatures,
        signature_saturation=unique_signatures / max(total_points, 1),
        connected_components=_connected_components(final_ids, grid_size),
        boundary_density=_boundary_density(final_ids, grid_size),
        refinement_mean=refinement_mean,
        refinement_max=refinement_max,
        parent_cells=parent_cells,
        parent_support_mean=parent_support_mean,
        parent_support_median=parent_support_median,
        parent_support_p10=parent_support_p10,
        parent_singleton_fraction=parent_singleton_fraction,
        parent_le4_fraction=parent_le4_fraction,
    )


def _payload_attrs(model: nn.Module, args: argparse.Namespace) -> tuple[int, int]:
    if isinstance(model, PayloadWidthEmnistClassifier):
        layer = model.payload_layers()[0]
        return int(layer.payload_width), int(layer.write_degree)
    return 0, 0


def run(args: argparse.Namespace) -> list[ResolutionRow]:
    _seed_all(args.seed)
    args.max_train = args.max_train_examples
    args.max_test = args.max_test_examples
    args.workers = args.num_workers
    device = torch.device(args.device)
    train_loader, valid_loader, classes = _build_local_loaders(args)
    model = _build_model(args, classes).to(device)
    train_loss, train_acc = _train(model, train_loader, args, device=device)
    valid_loss, valid_acc = _eval_model(model, valid_loader, device=device)
    x_train, _y_train = _collect_dataset_tensor(train_loader)
    center, u, v = _pca_plane(x_train, limit=args.pca_samples)
    payload_width, write_degree = _payload_attrs(model, args)

    rows: list[ResolutionRow] = []
    for grid_size in _parse_grid_sizes(args.grid_sizes):
        points = _grid(center, u, v, grid_size=grid_size, span=args.plane_span)
        signatures = _batch_signatures(model, points, device=device, batch_size=args.probe_batch_size)
        metrics = _refinement_metrics(signatures, grid_size=grid_size)
        rows.append(
            ResolutionRow(
                model=args.model,
                depth=args.depth,
                grid_size=grid_size,
                plane_span=args.plane_span,
                epochs=args.epochs,
                params=sum(param.numel() for param in model.parameters()),
                mlp_output_dim=_residual_mlp_output_dim(args) if args.model == "residual_mlp" else 0,
                train_loss=train_loss,
                train_acc=train_acc,
                valid_loss=valid_loss,
                valid_acc=valid_acc,
                unique_signatures=metrics.unique_signatures,
                signature_saturation=metrics.signature_saturation,
                connected_components=metrics.connected_components,
                boundary_density=metrics.boundary_density,
                refinement_mean=metrics.refinement_mean,
                refinement_max=metrics.refinement_max,
                parent_cells=metrics.parent_cells,
                parent_support_mean=metrics.parent_support_mean,
                parent_support_median=metrics.parent_support_median,
                parent_support_p10=metrics.parent_support_p10,
                parent_singleton_fraction=metrics.parent_singleton_fraction,
                parent_le4_fraction=metrics.parent_le4_fraction,
                tables=args.tables,
                comparisons=args.comparisons,
                comparator_kc=args.comparator_kc if args.model == "comparator_signed_margin_kc" else 0,
                comparator_write_policy=args.comparator_write_policy if args.model == "comparator_signed_margin_kc" else "none",
                payload_width=payload_width,
                write_degree=write_degree,
                walsh_lut_dtype=args.walsh_lut_dtype if args.model == "walsh_affine" else "none",
                walsh_order=args.walsh_order if args.model == "walsh_affine" else 0,
                walsh_slope_order=args.walsh_slope_order if args.model == "walsh_affine" else 0,
                seed=args.seed,
            )
        )
    return rows


def _write_rows(path: Path, rows: list[ResolutionRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-resolution EMNIST refinement probe for selected MLP and Pairwise LUT models.")
    parser.add_argument(
        "--model",
        choices=[
            "residual_mlp",
            "full_vector",
            "pairwise_plain",
            "comparator_signed_margin_kc",
            "group_k16",
            "walsh_affine",
        ],
        required=True,
    )
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--root", default="data/emnist")
    parser.add_argument("--split", default="balanced")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-examples", type=int, default=0)
    parser.add_argument("--max-test-examples", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--residual-mlp-output-dim", type=int, default=0)
    parser.add_argument("--mlp-residual-scale", type=float, default=1.0)
    parser.add_argument("--tables", type=int, default=64)
    parser.add_argument("--comparisons", type=int, default=6)
    parser.add_argument("--anchor-policy", default="permuted")
    parser.add_argument("--lut-init-std", type=float, default=0.0)
    parser.add_argument("--write-degree", type=int, default=16)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--full-ste", action="store_true")
    parser.add_argument("--no-output-scaling", action="store_true")
    parser.add_argument("--comparator-kc", type=int, default=16)
    parser.add_argument("--comparator-write-policy", choices=["endpoint", "local-linegraph", "expander"], default="expander")
    parser.add_argument("--walsh-lut-dtype", choices=["fp32", "bf16", "fp16", "int8", "fp8", "int4", "int2", "fp4", "nf4"], default="int2")
    parser.add_argument("--walsh-order", type=int, choices=(1, 2), default=2)
    parser.add_argument("--walsh-coeff-init-std", type=float, default=0.02)
    parser.add_argument("--walsh-slope-order", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--walsh-slope-coeff-init-std", type=float, default=0.02)
    parser.add_argument("--walsh-slope-generator-init-std", type=float, default=0.02)
    parser.add_argument("--grid-sizes", default="96,192,256")
    parser.add_argument("--plane-span", type=float, default=3.0)
    parser.add_argument("--pca-samples", type=int, default=4096)
    parser.add_argument("--probe-batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("results/refinement_resolution_probe/summary.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = run(args)
    _write_rows(args.out, rows)
    for row in rows:
        print(
            f"model={row.model} L{row.depth} grid={row.grid_size} valid={row.valid_loss:.4f} "
            f"refine={row.refinement_mean:.3f} boundary={row.boundary_density:.3f} "
            f"unique={row.unique_signatures} saturation={row.signature_saturation:.3f} "
            f"parent_median={row.parent_support_median:.1f} singleton={row.parent_singleton_fraction:.3f}",
            flush=True,
        )
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
