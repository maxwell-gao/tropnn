from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Subset, TensorDataset

from tropnn import PairwiseLUT, PairwiseWalshLUT


class LinearImageClassifier(nn.Module):
    """nn.Linear baseline for experiments, not a core PC-LUT layer."""

    def __init__(self, image_features: int, classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(image_features, classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x.flatten(1))


class EmnistLinearClassifier(nn.Module):
    def __init__(self, *, input_dim: int, hidden_dim: int, num_classes: int, depth: int = 2, seed: int = 0, **_: object) -> None:
        super().__init__()
        torch.manual_seed(seed)
        layers: list[nn.Module] = []
        dim = input_dim
        for _layer in range(max(0, depth - 1)):
            layers.extend([nn.Linear(dim, hidden_dim), nn.GELU()])
            dim = hidden_dim
        layers.append(nn.Linear(dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x.flatten(1))


class EmnistPairwiseClassifier(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        depth: int = 2,
        tables: int = 64,
        comparisons: int = 6,
        seed: int = 0,
        backend: str = "auto",
        anchor_policy: str = "random",
        lut_dtype: str = "bf16",
        lut_init_std: float = 0.0,
        **_: object,
    ) -> None:
        super().__init__()
        dims = [input_dim] + [hidden_dim] * max(0, depth - 1) + [num_classes]
        self.layers = nn.ModuleList(
            [
                PairwiseLUT(
                    dims[i],
                    dims[i + 1],
                    tables=tables,
                    comparisons=comparisons,
                    seed=seed + i,
                    lut_init_std=lut_init_std,
                    backend=backend,  # type: ignore[arg-type]
                    anchor_policy=anchor_policy,
                    lut_dtype=lut_dtype,  # type: ignore[arg-type]
                )
                for i in range(len(dims) - 1)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        y = x.flatten(1)
        for idx, layer in enumerate(self.layers):
            y = layer(y).squeeze(1)
            if idx + 1 < len(self.layers):
                y = torch.tanh(y)
        return y


class PairwiseImageClassifier(nn.Module):
    def __init__(
        self,
        image_features: int,
        classes: int,
        *,
        tables: int,
        comparisons: int,
        seed: int,
        lut_init_std: float,
        anchor_policy: str,
        backend: str,
        lut_dtype: str,
    ) -> None:
        super().__init__()
        self.layer = PairwiseLUT(
            image_features,
            classes,
            tables=tables,
            comparisons=comparisons,
            seed=seed,
            lut_init_std=lut_init_std,
            anchor_policy=anchor_policy,
            backend=backend,  # type: ignore[arg-type]
            lut_dtype=lut_dtype,  # type: ignore[arg-type]
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x.flatten(1)).squeeze(1)


class PairwiseWalshImageClassifier(nn.Module):
    def __init__(
        self,
        image_features: int,
        classes: int,
        *,
        tables: int,
        comparisons: int,
        walsh_order: int,
        seed: int,
        coeff_init_std: float,
        anchor_policy: str,
        slope_order: int = 0,
        lut_dtype: str = "bf16",
    ) -> None:
        super().__init__()
        self.layer = PairwiseWalshLUT(
            image_features,
            classes,
            tables=tables,
            comparisons=comparisons,
            walsh_order=walsh_order,  # type: ignore[arg-type]
            slope_order=slope_order,  # type: ignore[arg-type]
            seed=seed,
            coeff_init_std=coeff_init_std,
            anchor_policy=anchor_policy,
            lut_dtype=lut_dtype,  # type: ignore[arg-type]
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x.flatten(1)).squeeze(1)


class EmnistPairwiseWalshClassifier(PairwiseWalshImageClassifier):
    """Compatibility wrapper used by older pairwise tests and reports."""

    def __init__(
        self,
        *,
        input_dim: int,
        num_classes: int,
        comparisons: int,
        pairwise_tables: int,
        walsh_order: int = 2,
        seed: int = 0,
        coeff_init_std: float = 0.02,
        anchor_policy: str = "random",
        slope_order: int = 0,
        lut_dtype: str = "bf16",
        **_: object,
    ) -> None:
        super().__init__(
            input_dim,
            num_classes,
            tables=pairwise_tables,
            comparisons=comparisons,
            walsh_order=walsh_order,
            slope_order=slope_order,
            seed=seed,
            coeff_init_std=coeff_init_std,
            anchor_policy=anchor_policy,
            lut_dtype=lut_dtype,
        )


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    accuracy: float


def _limited(dataset, limit: int | None):
    if limit is None or limit <= 0 or limit >= len(dataset):
        return dataset
    return Subset(dataset, range(limit))


def _build_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, int]:
    try:
        from torchvision.datasets import EMNIST
        from torchvision.transforms import Compose, Lambda, ToTensor
    except ImportError as exc:
        try:
            from lutflow.experiments.emnist_fit import _load_emnist_split
        except ImportError:
            raise RuntimeError("The EMNIST demo requires torchvision or lutflow's local IDX EMNIST loader.") from exc

        root = Path(args.root).expanduser()
        x_train, y_train = _load_emnist_split(root, args.split, train=True, limit=args.max_train, fix_orientation=True, permute=False, permute_seed=args.seed)
        x_test, y_test = _load_emnist_split(root, args.split, train=False, limit=args.max_test, fix_orientation=True, permute=False, permute_seed=args.seed)
        train_set = TensorDataset(x_train.float(), y_train.long())
        test_set = TensorDataset(x_test.float(), y_test.long())
        classes = int(max(int(y_train.max().item()), int(y_test.max().item())) + 1)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=args.device == "cuda")
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=args.device == "cuda")
        return train_loader, test_loader, classes

    transform = Compose([ToTensor(), Lambda(lambda x: x.flatten())])
    root = Path(args.root).expanduser()
    train_set = EMNIST(root=str(root), split=args.split, train=True, download=args.download, transform=transform)
    test_set = EMNIST(root=str(root), split=args.split, train=False, download=args.download, transform=transform)
    train_set = _limited(train_set, args.max_train)
    test_set = _limited(test_set, args.max_test)
    classes = len(train_set.dataset.classes) if isinstance(train_set, Subset) else len(train_set.classes)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=args.device == "cuda")
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=args.device == "cuda")
    return train_loader, test_loader, classes


def _build_model(args: argparse.Namespace, classes: int) -> nn.Module:
    if args.family == "linear":
        return LinearImageClassifier(28 * 28, classes)
    if args.family == "pairwise":
        return PairwiseImageClassifier(
            28 * 28,
            classes,
            tables=args.tables,
            comparisons=args.comparisons,
            seed=args.seed,
            lut_init_std=args.lut_init_std,
            anchor_policy=args.anchor_policy,
            backend=args.backend,
            lut_dtype=args.lut_dtype,
        )
    if args.family == "pairwise_walsh":
        return PairwiseWalshImageClassifier(
            28 * 28,
            classes,
            tables=args.tables,
            comparisons=args.comparisons,
            walsh_order=args.walsh_order,
            slope_order=args.slope_order,
            seed=args.seed,
            coeff_init_std=args.lut_init_std,
            anchor_policy=args.anchor_policy,
            lut_dtype=args.lut_dtype,
        )
    raise ValueError(f"unknown family {args.family!r}")


def _run_epoch(model: nn.Module, loader: DataLoader, *, optimizer: torch.optim.Optimizer | None, device: torch.device) -> EpochMetrics:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        if training:
            loss.backward()
            optimizer.step()
        batch = int(y.numel())
        total_loss += float(loss.detach().item()) * batch
        total_correct += int((logits.argmax(dim=-1) == y).sum().item())
        total_seen += batch
    return EpochMetrics(loss=total_loss / max(1, total_seen), accuracy=total_correct / max(1, total_seen))


def main() -> None:
    parser = argparse.ArgumentParser(description="EMNIST experiments: nn.Linear baseline and no-GEMM Pairwise LUT models.")
    parser.add_argument("--root", default="data/emnist")
    parser.add_argument("--split", default="digits")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--family", choices=("linear", "pairwise", "pairwise_walsh"), default="pairwise")
    parser.add_argument("--tables", type=int, default=64)
    parser.add_argument("--comparisons", type=int, default=6)
    parser.add_argument("--walsh-order", type=int, choices=(1, 2), default=2)
    parser.add_argument("--slope-order", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--lut-init-std", type=float, default=0.02)
    parser.add_argument("--anchor-policy", default="random")
    parser.add_argument("--backend", choices=("auto", "torch", "tilelang", "triton"), default="auto")
    parser.add_argument("--lut-dtype", choices=("fp32", "bf16", "fp16", "int8", "fp8", "int4", "int2", "fp4", "nf4"), default="bf16")
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-test", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    train_loader, test_loader, classes = _build_loaders(args)
    model = _build_model(args, classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        train = _run_epoch(model, train_loader, optimizer=optimizer, device=device)
        valid = _run_epoch(model, test_loader, optimizer=None, device=device)
        print(
            f"epoch={epoch} family={args.family} train_loss={train.loss:.4f} train_acc={train.accuracy:.4f} "
            f"valid_loss={valid.loss:.4f} valid_acc={valid.accuracy:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
