"""Train a minimal trop_minface classifier on a locally downloaded EMNIST split.

Example:

    uv run python -m tropnn.example_emnist \
        --root /path/to/emnist \
        --split digits \
        --epochs 3 \
        --batch-size 512 \
        --lr 1e-3 \
        --tables 16 \
        --hidden-dim 128 \
        --depth 2 \
        --groups 2 \
        --cells 4 \
        --rank 32
"""

from __future__ import annotations

import argparse
import csv
import gzip
import math
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

from .module import TropLinear

IDX_DTYPES = {
    0x08: np.uint8,
    0x09: np.int8,
    0x0B: np.dtype(">i2"),
    0x0C: np.dtype(">i4"),
    0x0D: np.dtype(">f4"),
    0x0E: np.dtype(">f8"),
}
EMNIST_SPLITS = ("byclass", "bymerge", "balanced", "letters", "digits", "mnist")


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


class EmnistTropClassifier(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        depth: int,
        tables: int,
        groups: int,
        cells: int,
        rank: int,
        backend: str,
        seed: int,
    ) -> None:
        super().__init__()
        dims = [input_dim]
        if depth == 1:
            dims.append(num_classes)
        else:
            dims.extend([hidden_dim] * (depth - 1))
            dims.append(num_classes)
        self.layers = nn.ModuleList(
            [
                TropLinear(d_in, d_out, tables=tables, groups=groups, cells=cells, rank=rank, backend=backend, seed=seed + idx)
                for idx, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:]))
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        for layer in self.layers:
            x = layer(x)
        return x.squeeze(1)


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_items = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            total_loss += float(loss.item()) * x.shape[0]
            total_correct += int((logits.argmax(dim=-1) == y).sum().item())
            total_items += x.shape[0]
    return total_loss / total_items, total_correct / total_items


def _iter_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_items = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
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
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--tables", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--groups", type=int, default=2)
    parser.add_argument("--cells", type=int, default=4)
    parser.add_argument("--rank", type=int, default=32)
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
    model = EmnistTropClassifier(
        input_dim=x_train.shape[1],
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        depth=args.depth,
        tables=args.tables,
        groups=args.groups,
        cells=args.cells,
        rank=args.rank,
        backend=args.backend,
        seed=args.seed,
    ).to(device)
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=args.batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    print(
        f"EMNIST tropnn\n"
        f"  root       : {args.root}\n"
        f"  split      : {args.split}\n"
        f"  layer      : trop_minface\n"
        f"  depth      : {args.depth}\n"
        f"  hidden_dim : {args.hidden_dim}\n"
        f"  tables     : {args.tables}\n"
        f"  groups     : {args.groups}\n"
        f"  cells      : {args.cells}\n"
        f"  rank       : {args.rank}\n"
        f"  backend    : {args.backend}\n"
        f"  train/test : {len(x_train)}/{len(x_test)}\n"
        f"  device     : {device.type}\n"
        f"  params     : {sum(param.numel() for param in model.parameters())}\n"
    )

    rows: list[dict] = []
    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = _iter_epoch(model, train_loader, optimizer, device)
        test_loss, test_acc = _evaluate(model, test_loader, device)
        rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
            }
        )
        print(f"epoch {epoch:>3d} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} | test_loss={test_loss:.4f} test_acc={test_acc:.4f}")

    repo_root = Path(__file__).resolve().parents[3]
    out_path = repo_root / "results" / "experiments" / "tropnn_emnist" / f"{args.split}_trop_minface_{int(time.time())}.csv"
    _write_metrics(rows, out_path)
    print(f"\nDone in {time.perf_counter() - t0:.1f}s; metrics -> {out_path}")


if __name__ == "__main__":
    main()