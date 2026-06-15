"""Train minimal tropnn and ResNet baselines on local CIFAR-100 files."""

from __future__ import annotations

import argparse
import csv
import pickle
import time
import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset

from ..layers import PairwiseLinear
from .emnist import _make_layer

CIFAR100_MEAN = torch.tensor([0.5070758, 0.4865503, 0.4409191], dtype=torch.float32).view(1, 3, 1, 1)
CIFAR100_STD = torch.tensor([0.2673343, 0.2564385, 0.2761505], dtype=torch.float32).view(1, 3, 1, 1)
FAMILIES = ("linear", "dense_mlp", "pairwise", "pairwise_walsh", "resnet20")


def _find_cifar100_dir(root: Path) -> Path:
    if (root / "train").exists() and (root / "test").exists() and (root / "meta").exists():
        return root
    candidate = root / "cifar-100-python"
    if (candidate / "train").exists() and (candidate / "test").exists() and (candidate / "meta").exists():
        return candidate
    raise FileNotFoundError(f"Could not find CIFAR-100 python files under {root}; expected cifar-100-python/{{train,test,meta}}")


def _unpickle(path: Path) -> dict:
    with path.open("rb") as handle:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=getattr(np, "VisibleDeprecationWarning", DeprecationWarning))
            warnings.filterwarnings("ignore", message=r"dtype\(\): align should be passed.*")
            return pickle.load(handle, encoding="latin1")


def load_cifar100_split(
    root: Path,
    *,
    train: bool,
    label_mode: Literal["fine", "coarse"] = "fine",
    limit: int | None = None,
    normalize: bool = True,
    flatten: bool = True,
) -> tuple[Tensor, Tensor]:
    data_dir = _find_cifar100_dir(root)
    obj = _unpickle(data_dir / ("train" if train else "test"))
    images = torch.from_numpy(obj["data"].astype(np.float32)).view(-1, 3, 32, 32) / 255.0
    if normalize:
        images = (images - CIFAR100_MEAN) / CIFAR100_STD
    label_key = "fine_labels" if label_mode == "fine" else "coarse_labels"
    labels = torch.tensor(obj[label_key], dtype=torch.long)
    if limit is not None:
        images = images[:limit]
        labels = labels[:limit]
    return (images.flatten(1) if flatten else images), labels


def load_cifar100_label_names(root: Path, *, label_mode: Literal["fine", "coarse"] = "fine") -> list[str]:
    data_dir = _find_cifar100_dir(root)
    meta = _unpickle(data_dir / "meta")
    return list(meta["fine_label_names" if label_mode == "fine" else "coarse_label_names"])


class CifarTensorDataset(Dataset):
    def __init__(self, x: Tensor, y: Tensor, *, augment: bool = False) -> None:
        self.x = x
        self.y = y
        self.augment = augment

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        x = self.x[index]
        if self.augment:
            x = F.pad(x, (4, 4, 4, 4))
            top = int(torch.randint(0, 9, ()).item())
            left = int(torch.randint(0, 9, ()).item())
            x = x[:, top : top + 32, left : left + 32]
            if bool(torch.rand(()) < 0.5):
                x = torch.flip(x, dims=(-1,))
        return x, self.y[index]


class CifarBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut: nn.Module
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + self.shortcut(x))
        return out


class CifarResNet20(nn.Module):
    """CIFAR ResNet-20: 6n+2 with n=3 basic blocks per stage."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, blocks=3, stride=1)
        self.layer2 = self._make_layer(32, blocks=3, stride=2)
        self.layer3 = self._make_layer(64, blocks=3, stride=2)
        self.fc = nn.Linear(64, num_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _make_layer(self, planes: int, *, blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for block_stride in strides:
            layers.append(CifarBasicBlock(self.in_planes, planes, block_stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.shape[-1])
        return self.fc(out.flatten(1))


class Cifar100RoutedClassifier(nn.Module):
    def __init__(
        self,
        *,
        family: str,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        depth: int,
        comparisons: int,
        pairwise_tables: int,
        walsh_order: int,
        backend: str,
        seed: int,
    ) -> None:
        super().__init__()
        if family not in FAMILIES:
            raise ValueError(f"Unsupported CIFAR-100 family {family!r}")
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")

        self.family = family
        if family == "resnet20":
            self.layers = nn.ModuleList([CifarResNet20(num_classes)])
            return

        dims = [input_dim]
        if depth == 1:
            dims.append(num_classes)
        else:
            dims.extend([hidden_dim] * (depth - 1))
            dims.append(num_classes)

        if family == "linear":
            if depth != 1:
                raise ValueError("family='linear' requires depth=1")
            self.layers = nn.ModuleList([nn.Linear(input_dim, num_classes)])
        elif family == "dense_mlp":
            layers: list[nn.Module] = []
            for idx, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:])):
                layers.append(nn.Linear(d_in, d_out))
                if idx < len(dims) - 2:
                    layers.append(nn.GELU())
            self.layers = nn.ModuleList(layers)
        else:
            self.layers = nn.ModuleList(
                [
                    _make_layer(
                        family,
                        d_in,
                        d_out,
                        heads=1,
                        cells=1,
                        code_dim=1,
                        route_terms=1,
                        fan_value_mode="site",
                        fan_basis_rank=1,
                        comparisons=comparisons,
                        pairwise_tables=pairwise_tables,
                        pairwise_lut_init_std=0.0,
                        pairwise_lut_accumulation="sum",
                        pairwise_max_group_size=4,
                        pairwise_slope_bank_rank=0,
                        pairwise_slope_bank_atom_init_std=0.02,
                        pairwise_slope_bank_coeff_init_std=0.0,
                        fixed_zero_threshold=False,
                        pairwise_route_premix="none",
                        route_premix_block_size=64,
                        route_premix_expander_fanout=4,
                        route_premix_sparse_stages=2,
                        route_premix_lowrank_rank=4,
                        pairwise_hashes=1,
                        walsh_order=walsh_order,
                        backend=backend,
                        seed=seed + idx,
                    )
                    for idx, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:]))
                ]
            )

    def forward(self, x: Tensor) -> Tensor:
        if self.family in {"pairwise", "pairwise_walsh"} and x.ndim == 2:
            x = x.unsqueeze(1)
        for layer in self.layers:
            x = layer(x)
        return x.squeeze(1) if x.ndim == 3 and x.shape[1] == 1 else x


def _topk_correct(logits: Tensor, y: Tensor, k: int) -> int:
    k = min(k, logits.shape[-1])
    return int((logits.topk(k, dim=-1).indices == y.unsqueeze(-1)).any(dim=-1).sum().item())


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float, float, float]:
    model.train(mode=optimizer is not None)
    total_loss = 0.0
    total_top1 = 0
    total_top5 = 0
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
            total_top1 += int((logits.argmax(dim=-1) == y).sum().item())
            total_top5 += _topk_correct(logits, y, 5)
            total_items += x.shape[0]
    return total_loss / total_items, total_top1 / total_items, total_top5 / total_items, total_items


def _write_metrics(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _pairwise_diagnostics(
    model: nn.Module,
    sample_x: Tensor,
    device: torch.device,
    *,
    margin_eps: float = 1e-2,
) -> dict[str, float]:
    captures: list[tuple[PairwiseLinear, Tensor]] = []
    hooks = []
    for module in model.modules():
        if isinstance(module, PairwiseLinear):
            hooks.append(module.register_forward_pre_hook(lambda mod, inputs: captures.append((mod, inputs[0].detach()))))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        _ = model(sample_x.to(device))
    for hook in hooks:
        hook.remove()
    model.train(was_training)

    if not captures:
        return {}

    active_fracs = []
    entropies = []
    max_fracs = []
    samples_per_active = []
    near_zero_fracs = []
    margin_abs_means = []
    threshold_norms = []
    value_norms = []
    for module, latent in captures:
        if latent.ndim == 2:
            latent = latent.unsqueeze(1)
        indices, margins = module._compute_indices(latent.to(device))
        item_count = indices.shape[0] * indices.shape[1]
        for table_idx in range(module.tables):
            counts = torch.bincount(indices[..., table_idx].reshape(-1), minlength=module.table_size).float()
            probs = counts / counts.sum().clamp_min(1.0)
            active = int((counts > 0).sum().item())
            entropy = float((-(probs[probs > 0] * probs[probs > 0].log()).sum() / np.log(module.table_size)).item())
            active_fracs.append(active / module.table_size)
            entropies.append(entropy)
            max_fracs.append(float(probs.max().item()))
            samples_per_active.append(item_count / max(active, 1))
        near_zero_fracs.append(float((margins.abs() < margin_eps).float().mean().item()))
        margin_abs_means.append(float(margins.abs().mean().item()))
        threshold_norms.append(float(module.thresholds.detach().norm().item()))
        if hasattr(module, "lut"):
            value_norms.append(float(module.lut.detach().norm().item()))
        elif hasattr(module, "materialize_lut"):
            value_norms.append(float(module.materialize_lut(dtype=torch.float32, device=device).detach().norm().item()))

    return {
        "diag_pairwise_layers": float(len(captures)),
        "diag_active_cell_frac": float(np.mean(active_fracs)),
        "diag_cell_entropy": float(np.mean(entropies)),
        "diag_max_cell_frac": float(np.mean(max_fracs)),
        "diag_samples_per_active_cell": float(np.mean(samples_per_active)),
        "diag_margin_near_zero_frac": float(np.mean(near_zero_fracs)),
        "diag_margin_abs_mean": float(np.mean(margin_abs_means)),
        "diag_threshold_norm": float(np.mean(threshold_norms)),
        "diag_value_norm": float(np.mean(value_norms)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--label-mode", choices=("fine", "coarse"), default="fine")
    parser.add_argument("--family", choices=FAMILIES, default="pairwise")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--pairwise-tables", type=int, default=128)
    parser.add_argument("--comparisons", type=int, default=6)
    parser.add_argument("--walsh-order", type=int, choices=(1, 2), default=2)
    parser.add_argument("--backend", choices=("torch", "tilelang"), default="torch")
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    flatten = args.family != "resnet20"
    x_train, y_train = load_cifar100_split(
        args.root,
        train=True,
        label_mode=args.label_mode,
        limit=args.max_train,
        normalize=not args.no_normalize,
        flatten=flatten,
    )
    x_test, y_test = load_cifar100_split(
        args.root,
        train=False,
        label_mode=args.label_mode,
        limit=args.max_test,
        normalize=not args.no_normalize,
        flatten=flatten,
    )
    num_classes = int(max(y_train.max().item(), y_test.max().item()) + 1)
    model = Cifar100RoutedClassifier(
        family=args.family,
        input_dim=x_train.shape[1],
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        depth=args.depth,
        comparisons=args.comparisons,
        pairwise_tables=args.pairwise_tables,
        walsh_order=args.walsh_order,
        backend=args.backend,
        seed=args.seed,
    ).to(device)
    train_dataset: Dataset
    test_dataset: Dataset
    if args.augment:
        if flatten:
            raise ValueError("--augment requires an image-shaped family such as resnet20")
        train_dataset = CifarTensorDataset(x_train, y_train, augment=True)
        test_dataset = CifarTensorDataset(x_test, y_test, augment=False)
    else:
        train_dataset = TensorDataset(x_train, y_train)
        test_dataset = TensorDataset(x_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    param_count = sum(param.numel() for param in model.parameters())
    config_lines = {
        "root": args.root,
        "label_mode": args.label_mode,
        "family": args.family,
        "depth": args.depth,
        "hidden_dim": args.hidden_dim if args.family not in {"linear", "resnet20"} else "-",
        "pairwise_tables": args.pairwise_tables if args.family in {"pairwise", "pairwise_walsh"} else "-",
        "comparisons": args.comparisons if args.family in {"pairwise", "pairwise_walsh"} else "-",
        "walsh_order": args.walsh_order if args.family == "pairwise_walsh" else "-",
        "backend": args.backend if args.family in {"pairwise", "pairwise_walsh"} else "torch",
        "normalize": not args.no_normalize,
        "augment": args.augment,
        "train/test": f"{len(x_train)}/{len(x_test)}",
        "device": device.type,
        "params": param_count,
    }
    config_text = "\n".join(f"  {key:<16}: {value}" for key, value in config_lines.items())
    print(f"CIFAR-100 tropnn\n{config_text}\n")

    rows: list[dict] = []
    best_test_acc = 0.0
    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_t0 = time.perf_counter()
        train_loss, train_acc, train_top5, train_items = _run_epoch(model, train_loader, device, optimizer)
        train_elapsed = time.perf_counter() - train_t0
        test_loss, test_acc, test_top5, _ = _run_epoch(model, test_loader, device)
        best_test_acc = max(best_test_acc, test_acc)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_top5_acc": train_top5,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_top5_acc": test_top5,
            "best_test_acc": best_test_acc,
            "params": param_count,
            "images_per_sec": train_items / max(train_elapsed, 1e-9),
        }
        if args.diagnostics:
            row.update(_pairwise_diagnostics(model, x_test[: min(args.batch_size, len(x_test))], device))
        rows.append(row)
        print(
            f"epoch {epoch:>3d} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} top5={train_top5:.4f} | "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} top5={test_top5:.4f}"
        )

    repo_root = Path(__file__).resolve().parents[4]
    out_path = repo_root / "results" / "experiments" / "tropnn_cifar100" / f"{args.label_mode}_{args.family}_{time.time_ns()}.csv"
    _write_metrics(rows, out_path)
    print(f"\nDone in {time.perf_counter() - t0:.1f}s; metrics -> {out_path}")


if __name__ == "__main__":
    main()
