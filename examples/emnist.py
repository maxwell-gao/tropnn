"""Train a minimal tropnn classifier on a local EMNIST split."""

from __future__ import annotations

import argparse
import csv
import gzip
import struct
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from ..layers import PairwiseLinear, TropLinear

IDX_DTYPES = {
    0x08: np.uint8,
    0x09: np.int8,
    0x0B: np.dtype(">i2"),
    0x0C: np.dtype(">i4"),
    0x0D: np.dtype(">f4"),
    0x0E: np.dtype(">f8"),
}
EMNIST_SPLITS = ("byclass", "bymerge", "balanced", "letters", "digits", "mnist")
ROUTED_FAMILIES = ("tropical", "pairwise")


def _read_idx(path: Path) -> np.ndarray:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as handle:
        header = handle.read(4)
        zero_a, zero_b, dtype_code, ndim = struct.unpack(">BBBB", header)
        if zero_a != 0 or zero_b != 0:
            raise ValueError(f"IDX file {path} has invalid magic prefix")
        shape = struct.unpack(f">{ndim}I", handle.read(4 * ndim))
        data = np.frombuffer(handle.read(), dtype=IDX_DTYPES[dtype_code])
    return data.reshape(shape)


def _find_emnist_file(root: Path, split: str, train: bool, kind: str) -> Path:
    stem = f"emnist-{split}-{'train' if train else 'test'}-{kind}-idx{'3' if kind == 'images' else '1'}-ubyte"
    candidates = sorted(root.rglob(stem)) + sorted(root.rglob(stem + ".gz"))
    if not candidates:
        raise FileNotFoundError(f"Could not find {stem}[.gz] under {root}")
    return candidates[0]


def load_emnist_split(
    root: Path,
    split: str,
    *,
    train: bool,
    limit: Optional[int],
    fix_orientation: bool,
    permute: bool,
    permute_seed: int,
) -> tuple[Tensor, Tensor]:
    image_path = _find_emnist_file(root, split, train=train, kind="images")
    label_path = _find_emnist_file(root, split, train=train, kind="labels")
    images = _read_idx(image_path).astype(np.float32)
    labels = _read_idx(label_path).astype(np.int64)
    if fix_orientation:
        images = np.transpose(images, (0, 2, 1))[:, :, ::-1].copy()
    if split == "letters" and labels.min() == 1:
        labels = labels - 1
    x = torch.from_numpy(images).reshape(images.shape[0], -1).float() / 255.0
    x = x * 2.0 - 1.0
    y = torch.from_numpy(labels.astype(np.int64))
    if permute:
        gen = torch.Generator(device="cpu").manual_seed(permute_seed)
        order = torch.randperm(x.shape[1], generator=gen)
        x = x[:, order]
    if limit is not None:
        x = x[:limit]
        y = y[:limit]
    return x, y


def _make_layer(
    family: str,
    d_in: int,
    d_out: int,
    *,
    heads: int,
    cells: int,
    code_dim: int,
    comparisons: int,
    pairwise_tables: int,
    backend: str,
    seed: int,
) -> nn.Module:
    if family == "tropical":
        return TropLinear(d_in, d_out, heads=heads, cells=cells, code_dim=code_dim, backend=backend, seed=seed)
    return PairwiseLinear(d_in, d_out, tables=pairwise_tables, comparisons=comparisons, backend="torch", seed=seed)


class EmnistRoutedClassifier(nn.Module):
    def __init__(
        self,
        *,
        family: str,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        depth: int,
        heads: int,
        cells: int,
        code_dim: int,
        comparisons: int,
        pairwise_tables: int,
        backend: str,
        seed: int,
    ) -> None:
        super().__init__()
        self.family = family
        dims = [input_dim]
        if depth == 1:
            dims.append(num_classes)
        else:
            dims.extend([hidden_dim] * (depth - 1))
            dims.append(num_classes)
        self.layers = nn.ModuleList(
            [
                _make_layer(
                    family,
                    d_in,
                    d_out,
                    heads=heads,
                    cells=cells,
                    code_dim=code_dim,
                    comparisons=comparisons,
                    pairwise_tables=pairwise_tables,
                    backend=backend,
                    seed=seed + idx,
                )
                for idx, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:]))
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        for layer in self.layers:
            x = layer(x)
        return x.squeeze(1)


class EmnistTropClassifier(EmnistRoutedClassifier):
    def __init__(self, **kwargs) -> None:
        super().__init__(family="tropical", **kwargs)


class EmnistPairwiseClassifier(EmnistRoutedClassifier):
    def __init__(self, **kwargs) -> None:
        super().__init__(family="pairwise", backend="torch", **kwargs)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    model.train(mode=optimizer is not None)
    total_loss = 0.0
    total_correct = 0
    total_items = 0
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item()) * x.shape[0]
            total_correct += int((logits.argmax(dim=-1) == y).sum().item())
            total_items += x.shape[0]
    return total_loss / total_items, total_correct / total_items


def _write_metrics(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", choices=EMNIST_SPLITS, default="digits")
    parser.add_argument("--family", choices=ROUTED_FAMILIES, default="tropical")
    for name, arg_type, default in (
        ("--epochs", int, 10),
        ("--batch-size", int, 256),
        ("--lr", float, 3e-3),
        ("--hidden-dim", int, 256),
        ("--depth", int, 2),
        ("--heads", int, 32),
        ("--cells", int, 4),
        ("--code-dim", int, 32),
        ("--pairwise-tables", int, 72),
        ("--comparisons", int, 6),
    ):
        parser.add_argument(name, type=arg_type, default=default)
    parser.add_argument("--backend", choices=("torch", "auto", "triton"), default="torch")
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--permute", action="store_true")
    parser.add_argument("--permute-seed", type=int, default=0)
    parser.add_argument("--raw-orientation", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    x_train, y_train = load_emnist_split(
        args.root,
        args.split,
        train=True,
        limit=args.max_train,
        fix_orientation=not args.raw_orientation,
        permute=args.permute,
        permute_seed=args.permute_seed,
    )
    x_test, y_test = load_emnist_split(
        args.root,
        args.split,
        train=False,
        limit=args.max_test,
        fix_orientation=not args.raw_orientation,
        permute=args.permute,
        permute_seed=args.permute_seed,
    )
    num_classes = int(max(y_train.max().item(), y_test.max().item()) + 1)
    model = EmnistRoutedClassifier(
        family=args.family,
        input_dim=x_train.shape[1],
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        depth=args.depth,
        heads=args.heads,
        cells=args.cells,
        code_dim=args.code_dim,
        comparisons=args.comparisons,
        pairwise_tables=args.pairwise_tables,
        backend=args.backend,
        seed=args.seed,
    ).to(device)
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=args.batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    config_lines = {
        "root": args.root,
        "split": args.split,
        "family": args.family,
        "depth": args.depth,
        "hidden_dim": args.hidden_dim,
        "heads": args.heads if args.family == "tropical" else "-",
        "cells": args.cells if args.family == "tropical" else "-",
        "code_dim": args.code_dim if args.family == "tropical" else "-",
        "pairwise_tables": args.pairwise_tables if args.family == "pairwise" else "-",
        "comparisons": args.comparisons if args.family == "pairwise" else "-",
        "backend": args.backend if args.family == "tropical" else "torch",
        "train/test": f"{len(x_train)}/{len(x_test)}",
        "device": device.type,
        "params": sum(param.numel() for param in model.parameters()),
    }
    config_text = "\n".join(f"  {key:<15} : {value}" for key, value in config_lines.items())
    print(f"EMNIST tropnn\n{config_text}\n")

    rows: list[dict] = []
    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = _run_epoch(model, train_loader, device, optimizer)
        test_loss, test_acc = _run_epoch(model, test_loader, device)
        rows.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "test_loss": test_loss, "test_acc": test_acc})
        print(f"epoch {epoch:>3d} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} | test_loss={test_loss:.4f} test_acc={test_acc:.4f}")

    repo_root = Path(__file__).resolve().parents[4]
    out_path = repo_root / "results" / "experiments" / "tropnn_emnist" / f"{args.split}_{args.family}_{time.time_ns()}.csv"
    _write_metrics(rows, out_path)
    print(f"\nDone in {time.perf_counter() - t0:.1f}s; metrics -> {out_path}")


if __name__ == "__main__":
    main()
