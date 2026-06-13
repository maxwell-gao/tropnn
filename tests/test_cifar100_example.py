from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from tropnn.examples.cifar100 import Cifar100RoutedClassifier, load_cifar100_label_names, load_cifar100_split


def _write_fake_cifar100(root: Path) -> Path:
    data_dir = root / "cifar-100-python"
    data_dir.mkdir()
    data = (np.arange(4 * 3072, dtype=np.uint16).reshape(4, 3072) % 256).astype(np.uint8)
    train = {
        "data": data,
        "fine_labels": [0, 1, 2, 3],
        "coarse_labels": [0, 0, 1, 1],
    }
    test = {
        "data": data[:2],
        "fine_labels": [4, 5],
        "coarse_labels": [2, 2],
    }
    meta = {
        "fine_label_names": [f"fine_{idx}" for idx in range(100)],
        "coarse_label_names": [f"coarse_{idx}" for idx in range(20)],
    }
    for name, obj in (("train", train), ("test", test), ("meta", meta)):
        with (data_dir / name).open("wb") as handle:
            pickle.dump(obj, handle)
    return data_dir


def test_load_cifar100_split_from_official_pickle_layout(tmp_path: Path) -> None:
    _write_fake_cifar100(tmp_path)

    x, y = load_cifar100_split(tmp_path, train=True, label_mode="coarse", limit=3, normalize=False)
    image_x, _ = load_cifar100_split(tmp_path, train=True, label_mode="fine", limit=2, normalize=False, flatten=False)
    names = load_cifar100_label_names(tmp_path, label_mode="fine")

    assert x.shape == (3, 3072)
    assert image_x.shape == (2, 3, 32, 32)
    assert y.tolist() == [0, 0, 1]
    assert names[:3] == ["fine_0", "fine_1", "fine_2"]
    assert float(x.min()) >= 0.0
    assert float(x.max()) <= 1.0


def test_cifar100_pairwise_classifier_shape() -> None:
    model = Cifar100RoutedClassifier(
        family="pairwise",
        input_dim=3 * 32 * 32,
        hidden_dim=32,
        num_classes=100,
        depth=2,
        comparisons=3,
        pairwise_tables=4,
        walsh_order=2,
        backend="torch",
        seed=0,
    )

    logits = model(torch.randn(5, 3 * 32 * 32))
    assert logits.shape == (5, 100)


def test_cifar100_dense_mlp_classifier_shape() -> None:
    model = Cifar100RoutedClassifier(
        family="dense_mlp",
        input_dim=3 * 32 * 32,
        hidden_dim=32,
        num_classes=100,
        depth=2,
        comparisons=3,
        pairwise_tables=4,
        walsh_order=2,
        backend="torch",
        seed=0,
    )

    logits = model(torch.randn(5, 3 * 32 * 32))
    assert logits.shape == (5, 100)


def test_cifar100_resnet20_classifier_shape() -> None:
    model = Cifar100RoutedClassifier(
        family="resnet20",
        input_dim=3,
        hidden_dim=32,
        num_classes=100,
        depth=2,
        comparisons=3,
        pairwise_tables=4,
        walsh_order=2,
        backend="torch",
        seed=0,
    )

    logits = model(torch.randn(5, 3, 32, 32))
    assert logits.shape == (5, 100)
