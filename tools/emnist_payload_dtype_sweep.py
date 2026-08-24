from __future__ import annotations

import argparse
import csv
import gzip
import math
import struct
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from ..examples.emnist import EmnistPairwiseClassifier
from ..layers import PairwiseLUT

PAYLOAD_DTYPES = ("fp32", "bf16", "fp16", "int8", "fp8", "int4", "int2")
IDX_DTYPES = {
    0x08: np.uint8,
    0x09: np.int8,
    0x0B: np.dtype(">i2"),
    0x0C: np.dtype(">i4"),
    0x0D: np.dtype(">f4"),
    0x0E: np.dtype(">f8"),
}


@dataclass(frozen=True)
class PayloadDTypeRow:
    lut_dtype: str
    architecture: str
    lut_init_std: float
    backend: str
    device: str
    split: str
    train_examples: int
    valid_examples: int
    epochs: int
    depth: int
    hidden_dim: int
    tables: int
    comparisons: int
    anchor_policy: str
    params: int
    forward_ms: float
    fwd_bwd_ms: float
    train_loss: float
    train_acc: float
    valid_loss: float
    valid_acc: float
    finite_loss_steps: int
    nonfinite_loss_steps: int
    nonfinite_grad_steps: int
    lut_grad_norm_mean: float
    lut_grad_norm_max: float
    threshold_grad_norm_mean: float
    threshold_grad_norm_max: float
    threshold_to_lut_grad_norm_mean: float


class ResidualPairwiseEmnistClassifier(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        num_classes: int,
        depth: int,
        tables: int,
        comparisons: int,
        seed: int,
        backend: str,
        anchor_policy: str,
        lut_dtype: str,
        lut_init_std: float,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                PairwiseLUT(
                    input_dim,
                    input_dim,
                    tables=tables,
                    comparisons=comparisons,
                    seed=seed + block,
                    backend=backend,  # type: ignore[arg-type]
                    anchor_policy=anchor_policy,
                    lut_dtype=lut_dtype,  # type: ignore[arg-type]
                    lut_init_std=lut_init_std,
                )
                for block in range(depth)
            ]
        )
        self.readout = PairwiseLUT(
            input_dim,
            num_classes,
            tables=tables,
            comparisons=comparisons,
            seed=seed + depth,
            backend=backend,  # type: ignore[arg-type]
            anchor_policy=anchor_policy,
            lut_dtype=lut_dtype,  # type: ignore[arg-type]
            lut_init_std=lut_init_std,
        )

    def forward(self, x: Tensor) -> Tensor:
        y = x.flatten(1)
        for block in self.blocks:
            y = y + block(y).squeeze(1)
        return self.readout(y).squeeze(1)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_ms(fn, *, device: torch.device, warmups: int, iters: int) -> float:
    for _ in range(warmups):
        fn()
    _sync(device)
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize(device)
        return float(start.elapsed_time(end) / max(1, iters))
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync(device)
    return 1000.0 * (time.perf_counter() - t0) / max(1, iters)


def _loader_examples(loader) -> int:
    dataset = getattr(loader, "dataset", None)
    return int(len(dataset)) if dataset is not None else 0


def _read_idx(path: Path) -> np.ndarray:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as handle:
        header = handle.read(4)
        if len(header) != 4:
            raise ValueError(f"IDX file {path} is truncated")
        zero_a, zero_b, dtype_code, ndim = struct.unpack(">BBBB", header)
        if zero_a != 0 or zero_b != 0:
            raise ValueError(f"IDX file {path} has invalid magic prefix")
        if dtype_code not in IDX_DTYPES:
            raise ValueError(f"IDX file {path} uses unsupported dtype code {dtype_code}")
        shape = struct.unpack(f">{ndim}I", handle.read(4 * ndim))
        data = np.frombuffer(handle.read(), dtype=IDX_DTYPES[dtype_code])
    if data.size != math.prod(shape):
        raise ValueError(f"IDX file {path} expected {math.prod(shape)} values, got {data.size}")
    return data.reshape(shape)


def _find_emnist_file(root: Path, split: str, train: bool, kind: str) -> Path:
    stem = f"emnist-{split}-{'train' if train else 'test'}-{kind}-idx{'3' if kind == 'images' else '1'}-ubyte"
    candidates = sorted(root.rglob(stem)) + sorted(root.rglob(stem + ".gz"))
    if not candidates:
        raise FileNotFoundError(f"Could not find {stem}[.gz] under {root}")
    return candidates[0]


def _load_emnist_split(root: Path, split: str, *, train: bool, limit: int, seed: int) -> tuple[Tensor, Tensor]:
    del seed
    images = _read_idx(_find_emnist_file(root, split, train=train, kind="images")).astype(np.float32)
    labels = _read_idx(_find_emnist_file(root, split, train=train, kind="labels")).astype(np.int64)
    if images.ndim != 3:
        raise ValueError(f"expected EMNIST images with rank 3, got {images.shape}")
    if labels.ndim != 1 or labels.shape[0] != images.shape[0]:
        raise ValueError(f"invalid EMNIST labels shape {labels.shape} for images {images.shape}")
    images = np.transpose(images, (0, 2, 1))[:, :, ::-1].copy()
    if split == "letters" and labels.min() == 1:
        labels = labels - 1
    if limit > 0:
        images = images[:limit]
        labels = labels[:limit]
    x = torch.from_numpy(images).reshape(images.shape[0], -1).float() / 255.0
    x = x * 2.0 - 1.0
    y = torch.from_numpy(labels.astype(np.int64))
    return x, y


def _build_local_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, int]:
    root = Path(args.root).expanduser()
    x_train, y_train = _load_emnist_split(root, args.split, train=True, limit=args.max_train, seed=args.seed)
    x_valid, y_valid = _load_emnist_split(root, args.split, train=False, limit=args.max_test, seed=args.seed)
    classes = int(max(int(y_train.max().item()), int(y_valid.max().item())) + 1)
    train_set = TensorDataset(x_train, y_train)
    valid_set = TensorDataset(x_valid, y_valid)
    loader_seed = getattr(args, "loader_seed", None)
    train_generator = None
    if loader_seed is not None:
        train_generator = torch.Generator(device="cpu").manual_seed(int(loader_seed))
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=args.device == "cuda",
        generator=train_generator,
    )
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=args.device == "cuda")
    return train_loader, valid_loader, classes


def _first_batch(loader, device: torch.device) -> tuple[Tensor, Tensor]:
    x, y = next(iter(loader))
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def _build_model(args: argparse.Namespace, classes: int, lut_dtype: str) -> nn.Module:
    if args.architecture == "residual":
        return ResidualPairwiseEmnistClassifier(
            input_dim=28 * 28,
            num_classes=classes,
            depth=args.depth,
            tables=args.tables,
            comparisons=args.comparisons,
            seed=args.seed,
            backend=args.backend,
            anchor_policy=args.anchor_policy,
            lut_dtype=lut_dtype,
            lut_init_std=args.lut_init_std,
        )
    model = EmnistPairwiseClassifier(
        input_dim=28 * 28,
        hidden_dim=args.hidden_dim,
        num_classes=classes,
        depth=args.depth,
        tables=args.tables,
        comparisons=args.comparisons,
        seed=args.seed,
        backend=args.backend,
        anchor_policy=args.anchor_policy,
        lut_dtype=lut_dtype,
        lut_init_std=args.lut_init_std,
    )
    return model


def _benchmark_model(model: nn.Module, batch: tuple[Tensor, Tensor], *, device: torch.device, warmups: int, iters: int) -> tuple[float, float]:
    x, y = batch

    def forward_once() -> None:
        model.eval()
        with torch.no_grad():
            model(x)

    def fwd_bwd_once() -> None:
        model.train()
        model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y)
        loss.backward()

    forward_ms = _time_ms(forward_once, device=device, warmups=warmups, iters=iters)
    fwd_bwd_ms = _time_ms(fwd_bwd_once, device=device, warmups=warmups, iters=iters)
    model.zero_grad(set_to_none=True)
    return forward_ms, fwd_bwd_ms


def _param_grad_norm(model: nn.Module, name_fragment: str) -> tuple[float, bool]:
    total = 0.0
    finite = True
    for name, param in model.named_parameters():
        if name_fragment not in name or param.grad is None:
            continue
        grad = param.grad.detach().float()
        finite = finite and bool(torch.isfinite(grad).all().item())
        total += float(grad.square().sum().item())
    return math.sqrt(total), finite


def _eval_model(model: nn.Module, loader, *, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            batch = int(y.numel())
            total_loss += float(loss.detach().item()) * batch
            total_correct += int((logits.argmax(dim=-1) == y).sum().item())
            total_seen += batch
    return total_loss / max(1, total_seen), total_correct / max(1, total_seen)


def _train_model(
    model: nn.Module,
    train_loader,
    valid_loader,
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> tuple[float, float, float, float, dict[str, float | int]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    finite_loss_steps = 0
    nonfinite_loss_steps = 0
    nonfinite_grad_steps = 0
    lut_grad_norms: list[float] = []
    threshold_grad_norms: list[float] = []
    ratios: list[float] = []
    last_train_loss = 0.0
    last_train_acc = 0.0

    for _epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            batch = int(y.numel())
            if torch.isfinite(loss):
                finite_loss_steps += 1
            else:
                nonfinite_loss_steps += 1
            loss.backward()
            lut_norm, lut_finite = _param_grad_norm(model, "lut")
            threshold_norm, threshold_finite = _param_grad_norm(model, "thresholds")
            if not lut_finite or not threshold_finite:
                nonfinite_grad_steps += 1
            lut_grad_norms.append(lut_norm)
            threshold_grad_norms.append(threshold_norm)
            ratios.append(threshold_norm / max(lut_norm, 1e-12))
            optimizer.step()

            total_loss += float(loss.detach().item()) * batch
            total_correct += int((logits.argmax(dim=-1) == y).sum().item())
            total_seen += batch
        last_train_loss = total_loss / max(1, total_seen)
        last_train_acc = total_correct / max(1, total_seen)

    valid_loss, valid_acc = _eval_model(model, valid_loader, device=device)
    stats: dict[str, float | int] = {
        "finite_loss_steps": finite_loss_steps,
        "nonfinite_loss_steps": nonfinite_loss_steps,
        "nonfinite_grad_steps": nonfinite_grad_steps,
        "lut_grad_norm_mean": sum(lut_grad_norms) / max(1, len(lut_grad_norms)),
        "lut_grad_norm_max": max(lut_grad_norms) if lut_grad_norms else 0.0,
        "threshold_grad_norm_mean": sum(threshold_grad_norms) / max(1, len(threshold_grad_norms)),
        "threshold_grad_norm_max": max(threshold_grad_norms) if threshold_grad_norms else 0.0,
        "threshold_to_lut_grad_norm_mean": sum(ratios) / max(1, len(ratios)),
    }
    return last_train_loss, last_train_acc, valid_loss, valid_acc, stats


def _parse_dtypes(value: str) -> list[str]:
    dtypes = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(dtypes) - set(PAYLOAD_DTYPES))
    if unknown:
        raise ValueError(f"unknown payload dtypes: {unknown}; expected subset of {PAYLOAD_DTYPES}")
    return dtypes


def run(args: argparse.Namespace) -> list[PayloadDTypeRow]:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    train_loader, valid_loader, classes = _build_local_loaders(args)
    batch = _first_batch(train_loader, device)
    rows: list[PayloadDTypeRow] = []
    for lut_dtype in _parse_dtypes(args.lut_dtypes):
        torch.manual_seed(args.seed)
        model = _build_model(args, classes, lut_dtype).to(device)
        forward_ms, fwd_bwd_ms = _benchmark_model(model, batch, device=device, warmups=args.warmups, iters=args.iters)
        train_loss, train_acc, valid_loss, valid_acc, stats = _train_model(model, train_loader, valid_loader, args, device=device)
        row = PayloadDTypeRow(
            lut_dtype=lut_dtype,
            architecture=args.architecture,
            lut_init_std=args.lut_init_std,
            backend=args.backend,
            device=str(device),
            split=args.split,
            train_examples=_loader_examples(train_loader),
            valid_examples=_loader_examples(valid_loader),
            epochs=args.epochs,
            depth=args.depth,
            hidden_dim=args.hidden_dim,
            tables=args.tables,
            comparisons=args.comparisons,
            anchor_policy=args.anchor_policy,
            params=sum(param.numel() for param in model.parameters()),
            forward_ms=forward_ms,
            fwd_bwd_ms=fwd_bwd_ms,
            train_loss=train_loss,
            train_acc=train_acc,
            valid_loss=valid_loss,
            valid_acc=valid_acc,
            finite_loss_steps=int(stats["finite_loss_steps"]),
            nonfinite_loss_steps=int(stats["nonfinite_loss_steps"]),
            nonfinite_grad_steps=int(stats["nonfinite_grad_steps"]),
            lut_grad_norm_mean=float(stats["lut_grad_norm_mean"]),
            lut_grad_norm_max=float(stats["lut_grad_norm_max"]),
            threshold_grad_norm_mean=float(stats["threshold_grad_norm_mean"]),
            threshold_grad_norm_max=float(stats["threshold_grad_norm_max"]),
            threshold_to_lut_grad_norm_mean=float(stats["threshold_to_lut_grad_norm_mean"]),
        )
        rows.append(row)
        print(
            f"dtype={row.lut_dtype} fwd={row.forward_ms:.3f}ms fwd_bwd={row.fwd_bwd_ms:.3f}ms "
            f"train_loss={row.train_loss:.4f} train_acc={row.train_acc:.4f} "
            f"valid_loss={row.valid_loss:.4f} valid_acc={row.valid_acc:.4f} "
            f"nonfinite_loss={row.nonfinite_loss_steps} nonfinite_grad={row.nonfinite_grad_steps} "
            f"thr_grad_mean={row.threshold_grad_norm_mean:.4g} lut_grad_mean={row.lut_grad_norm_mean:.4g}",
            flush=True,
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep PairwiseLUT payload precision on EMNIST balanced.")
    parser.add_argument("--root", default="data/emnist")
    parser.add_argument("--split", default="balanced")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--backend", choices=("auto", "torch", "tilelang", "triton"), default="tilelang")
    parser.add_argument("--architecture", choices=("residual", "plain"), default="residual")
    parser.add_argument("--lut-dtypes", default="fp32,bf16,fp16,int8,fp8")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--tables", type=int, default=64)
    parser.add_argument("--comparisons", type=int, default=6)
    parser.add_argument("--lut-init-std", type=float, default=0.0)
    parser.add_argument("--anchor-policy", default="permuted")
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-test", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--out", default="results/emnist_payload_dtype_sweep/summary.csv")
    args = parser.parse_args()

    rows = run(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
