"""Probe whether trained EMNIST PC-LUT payloads form reusable data features.

The probe deliberately excludes the 47-dimensional readout payload.  It tests
the 784-dimensional residual writes with four controls:

* visitation-weighted payload spectrum versus norm-matched isotropic rows;
* held-out class selectivity versus within-table row reassignment;
* top-k payload-subspace agreement across independently trained seeds;
* zero-shot functional retention after projecting a target model onto a
  subspace estimated only from the other seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator, Literal, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

from tropnn.tools.emnist_payload_dtype_sweep import _build_local_loaders
from tropnn.tools.emnist_payload_width import PayloadWidthLUTLayer

PayloadMode = Literal["float", "binary01"]


@dataclass(frozen=True)
class ModelConfig:
    depth: int = 4
    tables: int = 64
    comparisons: int = 6
    anchor_policy: str = "permuted"
    residual_scale: float = 1.0
    lut_init_std: float = 0.0
    payload_mode: PayloadMode = "float"


@dataclass
class FeatureMoments:
    sums: Tensor
    norm2: Tensor
    counts: Tensor

    @classmethod
    def zeros(cls, classes: int, features: int) -> "FeatureMoments":
        return cls(
            sums=torch.zeros(classes, features, dtype=torch.float64),
            norm2=torch.zeros(classes, dtype=torch.float64),
            counts=torch.zeros(classes, dtype=torch.long),
        )

    def update(self, values: Tensor, labels: Tensor) -> None:
        values = values.detach().float()
        labels = labels.detach().long()
        class_sums = torch.zeros(self.sums.shape, dtype=torch.float32, device=values.device)
        class_norm2 = torch.zeros(self.norm2.shape, dtype=torch.float32, device=values.device)
        class_counts = torch.zeros(self.counts.shape, dtype=torch.long, device=values.device)
        class_sums.index_add_(0, labels, values)
        class_norm2.index_add_(0, labels, values.square().sum(dim=1))
        class_counts.index_add_(0, labels, torch.ones_like(labels))
        self.sums += class_sums.cpu().double()
        self.norm2 += class_norm2.cpu().double()
        self.counts += class_counts.cpu()

    def centroids(self) -> Tensor:
        return (self.sums / self.counts.clamp_min(1).double().unsqueeze(1)).float()

    def between_total_ratio(self) -> float:
        total_count = int(self.counts.sum().item())
        if total_count == 0:
            return math.nan
        class_means = self.sums / self.counts.clamp_min(1).double().unsqueeze(1)
        global_mean = self.sums.sum(dim=0) / total_count
        between = (
            self.counts.double()
            * (class_means - global_mean).square().sum(dim=1)
        ).sum() / total_count
        total = self.norm2.sum() / total_count - global_mean.square().sum()
        return float((between / total.clamp_min(1e-30)).item())


@dataclass
class TrainProbeState:
    usage: list[Tensor]
    route_class_counts: list[Tensor]
    learned_moments: list[FeatureMoments]
    shuffled_moments: list[FeatureMoments]
    shuffle_permutations: list[Tensor]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FeatureProbeLUTLayer(PayloadWidthLUTLayer):
    """Full-vector layer with either float or binary-{0,1} lookup semantics."""

    def __init__(self, input_dim: int, output_dim: int, *, payload_mode: PayloadMode, **kwargs: object) -> None:
        super().__init__(
            input_dim,
            output_dim,
            variant="full_vector",
            write_degree=output_dim,
            **kwargs,
        )
        self.payload_mode = payload_mode

    def materialized_lut(self) -> Tensor:
        if self.payload_mode == "float":
            return self.lut
        quantized = torch.round(self.lut).clamp(0.0, 1.0)
        return self.lut + (quantized - self.lut).detach()

    def _lookup(self, indices: Tensor) -> Tensor:
        table_offsets = torch.arange(self.tables, device=indices.device, dtype=torch.long).view(1, self.tables) * self.table_size
        flat_indices = (indices + table_offsets).reshape(-1)
        rows = self.materialized_lut().reshape(self.tables * self.table_size, self.payload_width).index_select(0, flat_indices)
        return rows.view(indices.shape[0], self.tables, self.payload_width)


class FeatureProbeEmnistClassifier(nn.Module):
    """Residual classifier kept state-dict compatible with the earlier probe."""

    def __init__(self, config: ModelConfig, *, classes: int, seed: int) -> None:
        super().__init__()

        def layer(output_dim: int, layer_seed: int) -> FeatureProbeLUTLayer:
            return FeatureProbeLUTLayer(
                28 * 28,
                output_dim,
                tables=config.tables,
                comparisons=config.comparisons,
                payload_mode=config.payload_mode,
                anchor_policy=config.anchor_policy,
                seed=layer_seed,
                lut_init_std=config.lut_init_std,
                use_output_scaling=True,
                use_min_margin_ste=True,
            )

        self.blocks = nn.ModuleList(layer(28 * 28, seed + 101 * index) for index in range(config.depth))
        self.readout = layer(classes, seed + 10007)
        self.residual_scale = float(config.residual_scale)
        self.last_routes: list[Tensor] = []

    def forward(self, x: Tensor) -> Tensor:
        y = x.flatten(start_dim=1).float()
        routes: list[Tensor] = []
        for block in self.blocks:
            write, indices = block.compute(y)
            y = y + self.residual_scale * write
            routes.append(indices.detach())
        logits, _indices = self.readout.compute(y)
        self.last_routes = routes
        return logits


def _build_model(config: ModelConfig, *, classes: int, seed: int) -> FeatureProbeEmnistClassifier:
    return FeatureProbeEmnistClassifier(config, classes=classes, seed=seed)


def _data_args(args: argparse.Namespace, *, seed: int, shuffle_batch_size: int | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        root=args.data_root,
        split=args.split,
        max_train=args.max_train_examples,
        max_test=args.max_test_examples,
        batch_size=args.batch_size if shuffle_batch_size is None else shuffle_batch_size,
        workers=args.num_workers,
        device=args.device,
        seed=seed,
    )


def _ordered_loader(loader: DataLoader, *, batch_size: int, workers: int, pin_memory: bool) -> DataLoader:
    return DataLoader(
        loader.dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
    )


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, *, device: torch.device, limit: int = 0) -> tuple[float, float, int]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    seen = 0
    for images, labels in loader:
        if limit > 0 and seen >= limit:
            break
        if limit > 0 and seen + labels.numel() > limit:
            take = limit - seen
            images, labels = images[:take], labels[:take]
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss_sum += float(F.cross_entropy(logits, labels, reduction="sum").item())
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        seen += int(labels.numel())
    return loss_sum / max(1, seen), correct / max(1, seen), seen


def _atomic_torch_save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _append_csv(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def train_checkpoint(args: argparse.Namespace) -> None:
    checkpoint = Path(args.checkpoint)
    config = ModelConfig(
        depth=args.depth,
        tables=args.tables,
        comparisons=args.comparisons,
        anchor_policy=args.anchor_policy,
        residual_scale=args.residual_scale,
        lut_init_std=args.lut_init_std,
        payload_mode=args.payload_mode,
    )
    if checkpoint.exists():
        existing = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if bool(existing.get("complete", False)):
            expected = (args.seed, args.epochs, asdict(config))
            found_config = asdict(ModelConfig(**existing["model_config"]))
            found = (int(existing["seed"]), int(existing["epochs"]), found_config)
            if found != expected:
                raise ValueError(f"completed checkpoint config mismatch: expected {expected}, found {found}")
            optimizer_config = existing.get("optimizer_config")
            if optimizer_config is not None and (
                float(optimizer_config["lr"]) != args.lr
                or float(optimizer_config["weight_decay"]) != args.weight_decay
            ):
                raise ValueError(f"completed checkpoint optimizer mismatch: found {optimizer_config}")
            print(json.dumps({"status": "skipped_complete", "checkpoint": str(checkpoint)}), flush=True)
            return

    _seed_everything(args.seed)
    device = torch.device(args.device)
    train_loader, valid_loader, classes = _build_local_loaders(_data_args(args, seed=args.seed))
    model = _build_model(config, classes=classes, seed=args.seed).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history_path = Path(args.history)
    if history_path.exists():
        failed_history = history_path.with_name(f"{history_path.stem}.incomplete-{int(time.time())}{history_path.suffix}")
        os.replace(history_path, failed_history)

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        correct = 0
        seen = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError(f"non-finite loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().item()) * labels.numel()
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            seen += int(labels.numel())

        row: dict[str, object] = {
            "seed": args.seed,
            "epoch": epoch,
            "train_loss": loss_sum / max(1, seen),
            "train_acc": correct / max(1, seen),
        }
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            valid_loss, valid_acc, valid_seen = _evaluate(model, valid_loader, device=device)
            row.update(valid_loss=valid_loss, valid_acc=valid_acc, valid_examples=valid_seen)
        else:
            row.update(valid_loss=math.nan, valid_acc=math.nan, valid_examples=len(valid_loader.dataset))
        _append_csv(history_path, row)
        print(json.dumps(row), flush=True)

    final_loss, final_acc, valid_seen = _evaluate(model, valid_loader, device=device)
    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    _atomic_torch_save(
        {
            "complete": True,
            "seed": args.seed,
            "classes": classes,
            "model_config": asdict(config),
            "optimizer_config": {"name": "AdamW", "lr": args.lr, "weight_decay": args.weight_decay},
            "epochs": args.epochs,
            "train_examples": len(train_loader.dataset),
            "valid_examples": valid_seen,
            "valid_loss": final_loss,
            "valid_acc": final_acc,
            "state_dict": state,
        },
        checkpoint,
    )
    print(json.dumps({"status": "complete", "checkpoint": str(checkpoint), "valid_loss": final_loss, "valid_acc": final_acc}), flush=True)


def _load_checkpoint(path: Path, *, device: torch.device) -> tuple[FeatureProbeEmnistClassifier, dict[str, object]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not checkpoint.get("complete", False):
        raise ValueError(f"checkpoint is incomplete: {path}")
    config = ModelConfig(**checkpoint["model_config"])
    model = _build_model(config, classes=int(checkpoint["classes"]), seed=int(checkpoint["seed"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, checkpoint


def _make_shuffle_permutations(model: FeatureProbeEmnistClassifier, *, seed: int) -> list[Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    result: list[Tensor] = []
    for block in model.blocks:
        assert isinstance(block, FeatureProbeLUTLayer)
        result.append(torch.stack([torch.randperm(block.table_size, generator=generator) for _ in range(block.tables)]))
    return result


def _shuffled_write(block: FeatureProbeLUTLayer, indices: Tensor, permutation: Tensor) -> Tensor:
    tables = torch.arange(block.tables, device=indices.device).view(1, -1)
    mapped = permutation.to(indices.device)[tables, indices]
    return block._payload_to_output(block._lookup(mapped))


@torch.no_grad()
def _collect_train_probe(
    model: FeatureProbeEmnistClassifier,
    loader: DataLoader,
    *,
    classes: int,
    device: torch.device,
    limit: int,
    shuffle_seed: int,
) -> TrainProbeState:
    model.eval()
    permutations = _make_shuffle_permutations(model, seed=shuffle_seed)
    usage: list[Tensor] = []
    route_class_counts: list[Tensor] = []
    learned = []
    shuffled = []
    for block in model.blocks:
        assert isinstance(block, FeatureProbeLUTLayer)
        usage.append(torch.zeros(block.tables, block.table_size, dtype=torch.long))
        route_class_counts.append(torch.zeros(block.tables, block.table_size, classes, dtype=torch.long))
        learned.append(FeatureMoments.zeros(classes, block.output_dim))
        shuffled.append(FeatureMoments.zeros(classes, block.output_dim))

    seen = 0
    for images, labels in loader:
        if limit > 0 and seen >= limit:
            break
        if limit > 0 and seen + labels.numel() > limit:
            take = limit - seen
            images, labels = images[:take], labels[:take]
        y = images.to(device, non_blocking=True).flatten(1).float()
        labels = labels.to(device, non_blocking=True)
        for layer_index, block in enumerate(model.blocks):
            assert isinstance(block, FeatureProbeLUTLayer)
            write, indices = block.compute(y)
            shuffled_write = _shuffled_write(block, indices, permutations[layer_index])
            learned[layer_index].update(write, labels)
            shuffled[layer_index].update(shuffled_write, labels)

            table_offsets = torch.arange(block.tables, device=device).view(1, -1) * block.table_size
            flat_rows = indices + table_offsets
            usage[layer_index] += torch.bincount(
                flat_rows.reshape(-1), minlength=block.tables * block.table_size
            ).reshape(block.tables, block.table_size).cpu()
            label_grid = labels.view(-1, 1).expand_as(indices)
            joint = flat_rows * classes + label_grid
            route_class_counts[layer_index] += torch.bincount(
                joint.reshape(-1), minlength=block.tables * block.table_size * classes
            ).reshape(block.tables, block.table_size, classes).cpu()
            y = y + model.residual_scale * write
        seen += int(labels.numel())
    return TrainProbeState(usage, route_class_counts, learned, shuffled, permutations)


def _centroid_predictions(values: Tensor, centroids: Tensor, *, cosine: bool) -> Tensor:
    values = values.float()
    centroids = centroids.to(device=values.device, dtype=torch.float32)
    if cosine:
        values = F.normalize(values, dim=1)
        centroids = F.normalize(centroids, dim=1)
        return (values @ centroids.T).argmax(dim=1)
    scores = 2.0 * values @ centroids.T - centroids.square().sum(dim=1).view(1, -1)
    return scores.argmax(dim=1)


@torch.no_grad()
def _evaluate_selectivity(
    model: FeatureProbeEmnistClassifier,
    loader: DataLoader,
    train_state: TrainProbeState,
    *,
    classes: int,
    device: torch.device,
    limit: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    learned_centroids = [moments.centroids() for moments in train_state.learned_moments]
    shuffled_centroids = [moments.centroids() for moments in train_state.shuffled_moments]
    valid_learned = [FeatureMoments.zeros(classes, centroid.shape[1]) for centroid in learned_centroids]
    valid_shuffled = [FeatureMoments.zeros(classes, centroid.shape[1]) for centroid in shuffled_centroids]
    feature_correct = {
        (kind, metric, layer): 0
        for kind in ("learned", "row_reassigned")
        for metric in ("euclidean", "cosine")
        for layer in range(len(model.blocks))
    }

    alpha = 1.0
    route_log_probs: list[Tensor] = []
    for counts in train_state.route_class_counts:
        probabilities = (counts.float() + alpha) / (
            counts.sum(dim=1, keepdim=True).float() + alpha * counts.shape[1]
        )
        route_log_probs.append(probabilities.log().to(device))
    route_correct = [0 for _ in model.blocks]
    route_all_correct = 0
    class_prior = train_state.learned_moments[0].counts.float()
    class_prior = ((class_prior + alpha) / (class_prior.sum() + alpha * classes)).log().to(device)

    seen = 0
    for images, labels in loader:
        if limit > 0 and seen >= limit:
            break
        if limit > 0 and seen + labels.numel() > limit:
            take = limit - seen
            images, labels = images[:take], labels[:take]
        y = images.to(device, non_blocking=True).flatten(1).float()
        labels = labels.to(device, non_blocking=True)
        all_route_scores = class_prior.view(1, -1).expand(labels.shape[0], -1).clone()
        for layer_index, block in enumerate(model.blocks):
            assert isinstance(block, FeatureProbeLUTLayer)
            write, indices = block.compute(y)
            shuffled_write = _shuffled_write(block, indices, train_state.shuffle_permutations[layer_index])
            valid_learned[layer_index].update(write, labels)
            valid_shuffled[layer_index].update(shuffled_write, labels)
            for kind, values, centroid in (
                ("learned", write, learned_centroids[layer_index]),
                ("row_reassigned", shuffled_write, shuffled_centroids[layer_index]),
            ):
                for metric, cosine in (("euclidean", False), ("cosine", True)):
                    predictions = _centroid_predictions(values, centroid, cosine=cosine)
                    feature_correct[(kind, metric, layer_index)] += int((predictions == labels).sum().item())

            layer_scores = class_prior.view(1, -1).expand(labels.shape[0], -1).clone()
            for table in range(block.tables):
                selected = route_log_probs[layer_index][table].index_select(0, indices[:, table])
                layer_scores += selected
                all_route_scores += selected
            route_correct[layer_index] += int((layer_scores.argmax(dim=1) == labels).sum().item())
            y = y + model.residual_scale * write
        route_all_correct += int((all_route_scores.argmax(dim=1) == labels).sum().item())
        seen += int(labels.numel())

    feature_rows: list[dict[str, object]] = []
    for layer in range(len(model.blocks)):
        for kind, moments in (("learned", valid_learned[layer]), ("row_reassigned", valid_shuffled[layer])):
            for metric in ("euclidean", "cosine"):
                feature_rows.append(
                    {
                        "layer": layer,
                        "feature": kind,
                        "metric": metric,
                        "held_accuracy": feature_correct[(kind, metric, layer)] / max(1, seen),
                        "between_total_ratio": moments.between_total_ratio(),
                        "held_examples": seen,
                    }
                )
    route_rows = [
        {"layer": layer, "route_scope": "single_layer", "held_accuracy": route_correct[layer] / max(1, seen), "held_examples": seen}
        for layer in range(len(model.blocks))
    ]
    route_rows.append({"layer": -1, "route_scope": "all_layers", "held_accuracy": route_all_correct / max(1, seen), "held_examples": seen})
    return feature_rows, route_rows


def _spectral_summary(eigenvalues: Tensor) -> dict[str, float | int]:
    values = eigenvalues.detach().double().clamp_min(0).flip(0)
    total = values.sum()
    if not bool((total > 0).item()):
        return {"effective_rank": 0.0, "stable_rank": 0.0, "rank50": 0, "rank90": 0, "rank95": 0, "rank99": 0}
    probabilities = values / total
    effective_rank = torch.exp(-(probabilities * probabilities.clamp_min(1e-300).log()).sum())
    cumulative = probabilities.cumsum(0)

    def energy_rank(threshold: float) -> int:
        target = torch.tensor(threshold, dtype=cumulative.dtype, device=cumulative.device)
        return int(torch.searchsorted(cumulative, target).item()) + 1

    return {
        "effective_rank": float(effective_rank.item()),
        "stable_rank": float((total / values[0].clamp_min(1e-300)).item()),
        "rank50": energy_rank(0.50),
        "rank90": energy_rank(0.90),
        "rank95": energy_rank(0.95),
        "rank99": energy_rank(0.99),
    }


def _symmetric_eigh(matrix: Tensor, *, eigenvectors: bool) -> tuple[Tensor, Tensor | None]:
    """Use a symmetrized float64 matrix so rank-deficient controls stay stable."""

    symmetric = 0.5 * (matrix.double() + matrix.double().T)
    try:
        if eigenvectors:
            values, vectors = torch.linalg.eigh(symmetric)
            return values, vectors
        return torch.linalg.eigvalsh(symmetric), None
    except torch.linalg.LinAlgError:
        # LAPACK divide-and-conquer can reject highly rank-deficient matrices.
        # SVD is equivalent here because the input is a PSD covariance matrix.
        if eigenvectors:
            vectors, values, _right = torch.linalg.svd(symmetric, full_matrices=False)
            return values.flip(0), vectors.flip(1)
        return torch.linalg.svdvals(symmetric).flip(0), None


def _weighted_covariance(rows: Tensor, usage: Tensor, *, centered: bool) -> Tensor:
    rows = rows.float()
    weights = usage.flatten().to(device=rows.device, dtype=torch.float32)
    weights = weights / weights.sum().clamp_min(1.0)
    if centered:
        rows = rows - (weights.unsqueeze(1) * rows).sum(dim=0, keepdim=True)
    return rows.T @ (weights.unsqueeze(1) * rows)


def _isotropic_rows(rows: Tensor, *, seed: int) -> Tensor:
    generator = torch.Generator(device=rows.device)
    generator.manual_seed(seed)
    random_rows = torch.randn(rows.shape, generator=generator, device=rows.device, dtype=torch.float32)
    random_rows = F.normalize(random_rows, dim=1)
    return random_rows * rows.float().norm(dim=1, keepdim=True)


def _covariance_and_spectrum(
    block: FeatureProbeLUTLayer,
    usage: Tensor,
    *,
    seed: int,
) -> tuple[dict[str, Tensor], list[dict[str, object]]]:
    rows = block.materialized_lut().detach().reshape(-1, block.output_dim).float()
    active = int((usage > 0).sum().item())
    total_rows = int(usage.numel())
    records: list[dict[str, object]] = []
    uncentered_covariance = _weighted_covariance(rows, usage, centered=False)
    centered_covariance = _weighted_covariance(rows, usage, centered=True)
    payload_nonzero_fraction = float((rows != 0).float().mean().item())
    payload_mean = float(rows.mean().item())
    payload_std = float(rows.std().item())
    for centered in (False, True):
        covariance = centered_covariance if centered else uncentered_covariance
        control = _weighted_covariance(_isotropic_rows(rows, seed=seed + int(centered)), usage, centered=centered)
        for kind, matrix in (("learned", covariance), ("norm_matched_isotropic", control)):
            eigenvalues, _vectors = _symmetric_eigh(matrix, eigenvectors=False)
            records.append(
                {
                    "spectrum": kind,
                    "centered": centered,
                    "active_rows": active,
                    "total_rows": total_rows,
                    "active_fraction": active / max(1, total_rows),
                    "payload_nonzero_fraction": payload_nonzero_fraction,
                    "payload_mean": payload_mean,
                    "payload_std": payload_std,
                    **_spectral_summary(eigenvalues),
                }
            )
    return {
        "uncentered": uncentered_covariance.detach().cpu(),
        "centered": centered_covariance.detach().cpu(),
    }, records


def _top_basis(covariance: Tensor, rank: int, *, device: torch.device) -> Tensor:
    covariance = covariance.to(device=device, dtype=torch.float32)
    _values, vectors = _symmetric_eigh(covariance, eigenvectors=True)
    assert vectors is not None
    return vectors[:, -rank:].flip(1).float().contiguous()


def _projector_overlap(left: Tensor, right: Tensor) -> float:
    rank = min(left.shape[1], right.shape[1])
    return float((left[:, :rank].T @ right[:, :rank]).square().sum().item() / max(1, rank))


@contextmanager
def _project_hidden_payloads(model: FeatureProbeEmnistClassifier, bases: Sequence[Tensor] | None) -> Iterator[None]:
    if bases is None:
        yield
        return
    originals = [block.lut.detach().clone() for block in model.blocks]
    try:
        with torch.no_grad():
            for block, basis in zip(model.blocks, bases, strict=True):
                assert isinstance(block, FeatureProbeLUTLayer)
                rows = block.materialized_lut().detach().reshape(-1, block.output_dim)
                basis = basis.to(device=rows.device, dtype=rows.dtype)
                block.lut.copy_((rows @ basis @ basis.T).reshape_as(block.lut))
        yield
    finally:
        with torch.no_grad():
            for block, original in zip(model.blocks, originals, strict=True):
                block.lut.copy_(original)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _parse_ranks(text: str, *, dimension: int) -> list[int]:
    ranks = sorted({int(value.strip()) for value in text.split(",") if value.strip()})
    if not ranks or ranks[0] < 1 or ranks[-1] > dimension:
        raise ValueError(f"ranks must be in [1, {dimension}], got {ranks}")
    return ranks


def analyze_checkpoints(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    paths = [Path(value) for value in args.checkpoints.split(",") if value]
    if len(paths) < 3:
        raise ValueError("analysis requires at least three independently trained checkpoints")
    metadata = [torch.load(path, map_location="cpu", weights_only=True) for path in paths]
    seeds = [int(item["seed"]) for item in metadata]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"checkpoint seeds must be distinct, got {seeds}")
    configs = [asdict(ModelConfig(**item["model_config"])) for item in metadata]
    if any(config != configs[0] for config in configs[1:]):
        raise ValueError("all checkpoints must use the same model configuration")
    payload_mode = str(configs[0]["payload_mode"])
    optimizer_configs = [item.get("optimizer_config", {}) for item in metadata]
    if any(config != optimizer_configs[0] for config in optimizer_configs[1:]):
        raise ValueError("all checkpoints must use the same optimizer configuration")
    train_lr = optimizer_configs[0].get("lr", math.nan)
    classes = int(metadata[0]["classes"])
    args.max_train_examples = 0
    args.max_test_examples = 0
    base_train, base_valid, data_classes = _build_local_loaders(_data_args(args, seed=0))
    if data_classes != classes:
        raise ValueError(f"data has {data_classes} classes but checkpoints have {classes}")
    train_loader = _ordered_loader(base_train, batch_size=args.probe_batch_size, workers=args.num_workers, pin_memory=device.type == "cuda")
    valid_loader = _ordered_loader(base_valid, batch_size=args.probe_batch_size, workers=args.num_workers, pin_memory=device.type == "cuda")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    covariances: dict[int, list[dict[str, Tensor]]] = {}
    low_rank_rows: list[dict[str, object]] = []
    selectivity_rows: list[dict[str, object]] = []
    route_rows: list[dict[str, object]] = []
    baseline_rows: list[dict[str, object]] = []

    for path, seed in zip(paths, seeds, strict=True):
        model, checkpoint = _load_checkpoint(path, device=device)
        baseline_loss, baseline_acc, baseline_seen = _evaluate(model, valid_loader, device=device)
        baseline_rows.append(
            {
                "seed": seed,
                "payload_mode": payload_mode,
                "train_lr": train_lr,
                "valid_loss": baseline_loss,
                "valid_acc": baseline_acc,
                "valid_examples": baseline_seen,
                "checkpoint": str(path),
            }
        )
        train_state = _collect_train_probe(
            model,
            train_loader,
            classes=classes,
            device=device,
            limit=args.probe_train_examples,
            shuffle_seed=args.control_seed + seed * 1009,
        )
        feature_records, route_records = _evaluate_selectivity(
            model,
            valid_loader,
            train_state,
            classes=classes,
            device=device,
            limit=args.probe_valid_examples,
        )
        selectivity_rows.extend({"seed": seed, "payload_mode": payload_mode, **record} for record in feature_records)
        route_rows.extend({"seed": seed, "payload_mode": payload_mode, **record} for record in route_records)
        seed_covariances: list[dict[str, Tensor]] = []
        for layer, (block, usage) in enumerate(zip(model.blocks, train_state.usage, strict=True)):
            assert isinstance(block, FeatureProbeLUTLayer)
            covariance, records = _covariance_and_spectrum(block, usage, seed=args.control_seed + seed * 1009 + layer * 17)
            seed_covariances.append(covariance)
            low_rank_rows.extend({"seed": seed, "payload_mode": payload_mode, "layer": layer, **record} for record in records)
        covariances[seed] = seed_covariances
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    dimension = int(configs[0].get("input_dim", 28 * 28)) if "input_dim" in configs[0] else 28 * 28
    alignment_ranks = _parse_ranks(args.alignment_ranks, dimension=dimension)
    alignment_rows: list[dict[str, object]] = []
    bases: dict[tuple[int, int, int, bool], Tensor] = {}
    for seed in seeds:
        for layer, covariance_by_centering in enumerate(covariances[seed]):
            for centered, covariance in ((False, covariance_by_centering["uncentered"]), (True, covariance_by_centering["centered"])):
                full_basis = _top_basis(covariance, max(alignment_ranks), device=device).cpu()
                for rank in alignment_ranks:
                    bases[(seed, layer, rank, centered)] = full_basis[:, :rank]
    for left_index, left_seed in enumerate(seeds):
        for right_seed in seeds[left_index + 1 :]:
            for layer in range(len(covariances[left_seed])):
                for centered in (False, True):
                    for rank in alignment_ranks:
                        covariance_key = "centered" if centered else "uncentered"
                        left_energy = float(torch.trace(covariances[left_seed][layer][covariance_key]).item())
                        right_energy = float(torch.trace(covariances[right_seed][layer][covariance_key]).item())
                        subspace_valid = left_energy > 1e-12 and right_energy > 1e-12
                        overlap = (
                            _projector_overlap(
                                bases[(left_seed, layer, rank, centered)], bases[(right_seed, layer, rank, centered)]
                            )
                            if subspace_valid
                            else math.nan
                        )
                        random_expected = rank / dimension
                        alignment_rows.append(
                            {
                                "seed_a": left_seed,
                                "seed_b": right_seed,
                                "payload_mode": payload_mode,
                                "layer": layer,
                                "centered": centered,
                                "subspace_valid": subspace_valid,
                                "seed_a_energy": left_energy,
                                "seed_b_energy": right_energy,
                                "rank": rank,
                                "projector_overlap": overlap,
                                "random_expected": random_expected,
                                "excess_over_random": overlap - random_expected if subspace_valid else math.nan,
                            }
                        )

    transfer_ranks = _parse_ranks(args.transfer_ranks, dimension=dimension)
    transfer_rows: list[dict[str, object]] = []
    for path, target_seed in zip(paths, seeds, strict=True):
        model, _checkpoint = _load_checkpoint(path, device=device)
        baseline_loss, baseline_acc, baseline_seen = _evaluate(model, valid_loader, device=device, limit=args.transfer_valid_examples)
        transfer_rows.append(
            {
                "target_seed": target_seed,
                "payload_mode": payload_mode,
                "post_projection_mode": "none",
                "basis": "unprojected",
                "rank": dimension,
                "valid_loss": baseline_loss,
                "valid_acc": baseline_acc,
                "accuracy_retention": 1.0,
                "loss_delta": 0.0,
                "valid_examples": baseline_seen,
            }
        )
        shared_covariances = [
            sum(
                (covariances[seed][layer]["uncentered"] for seed in seeds if seed != target_seed),
                torch.zeros_like(covariances[target_seed][layer]["uncentered"]),
            )
            / (len(seeds) - 1)
            for layer in range(len(model.blocks))
        ]
        max_rank = max(rank for rank in transfer_ranks if rank < dimension) if any(rank < dimension for rank in transfer_ranks) else 0
        own_full = [
            _top_basis(covariance["uncentered"], max_rank, device=device) for covariance in covariances[target_seed]
        ] if max_rank else []
        shared_full = [_top_basis(covariance, max_rank, device=device) for covariance in shared_covariances] if max_rank else []
        random_full: list[Tensor] = []
        if max_rank:
            generator = torch.Generator(device=device)
            generator.manual_seed(args.control_seed + target_seed * 65537)
            for layer in range(len(model.blocks)):
                matrix = torch.randn(dimension, max_rank, generator=generator, device=device)
                random_full.append(torch.linalg.qr(matrix, mode="reduced").Q)
        for rank in transfer_ranks:
            if rank == dimension:
                continue
            for name, full_bases in (("target_own", own_full), ("other_seed_shared", shared_full), ("random", random_full)):
                rank_bases = [basis[:, :rank] for basis in full_bases]
                with _project_hidden_payloads(model, rank_bases):
                    loss, acc, seen = _evaluate(model, valid_loader, device=device, limit=args.transfer_valid_examples)
                transfer_rows.append(
                    {
                        "target_seed": target_seed,
                        "payload_mode": payload_mode,
                        "post_projection_mode": "binary_rethreshold" if payload_mode == "binary01" else "continuous",
                        "basis": name,
                        "rank": rank,
                        "valid_loss": loss,
                        "valid_acc": acc,
                        "accuracy_retention": acc / max(baseline_acc, 1e-30),
                        "loss_delta": loss - baseline_loss,
                        "valid_examples": seen,
                    }
                )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    _write_csv(output_dir / "baseline.csv", baseline_rows)
    _write_csv(output_dir / "low_rank.csv", low_rank_rows)
    _write_csv(output_dir / "class_selectivity.csv", selectivity_rows)
    _write_csv(output_dir / "route_selectivity.csv", route_rows)
    _write_csv(output_dir / "cross_seed_alignment.csv", alignment_rows)
    _write_csv(output_dir / "shared_direction_transfer.csv", transfer_rows)
    summary = {
        "checkpoints": [str(path) for path in paths],
        "seeds": seeds,
        "model_config": configs[0],
        "classes": classes,
        "payload_mode": payload_mode,
        "optimizer_config": optimizer_configs[0],
        "probe_train_examples": args.probe_train_examples or len(train_loader.dataset),
        "probe_valid_examples": args.probe_valid_examples or len(valid_loader.dataset),
        "transfer_valid_examples": args.transfer_valid_examples or len(valid_loader.dataset),
        "definitions": {
            "low_rank": "training-route-frequency weighted hidden-payload covariance",
            "class_selectivity": "held-out nearest training-class centroid of each hidden write",
            "alignment": "mean squared cosine between top-k payload subspaces",
            "transfer": "target payload rows projected onto directions learned from other seeds, without fine-tuning",
            "binary_transfer": "binary01 payloads are rethresholded to {0,1} after projection",
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": "complete", "output_dir": str(output_dir), "seeds": seeds}), flush=True)


def _add_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--split", default="balanced")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="train one full-vector checkpoint")
    _add_data_arguments(train)
    train.add_argument("--checkpoint", type=Path, required=True)
    train.add_argument("--history", type=Path, required=True)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--eval-every", type=int, default=1)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--weight-decay", type=float, default=0.0)
    train.add_argument("--depth", type=int, default=4)
    train.add_argument("--tables", type=int, default=64)
    train.add_argument("--comparisons", type=int, default=6)
    train.add_argument("--anchor-policy", default="permuted")
    train.add_argument("--residual-scale", type=float, default=1.0)
    train.add_argument("--lut-init-std", type=float, default=0.0)
    train.add_argument("--payload-mode", choices=("float", "binary01"), default="float")
    train.add_argument("--max-train-examples", type=int, default=0)
    train.add_argument("--max-test-examples", type=int, default=0)

    analyze = subparsers.add_parser("analyze", help="run feature probes over trained checkpoints")
    _add_data_arguments(analyze)
    analyze.add_argument("--checkpoints", required=True, help="comma-separated checkpoint paths")
    analyze.add_argument("--output-dir", type=Path, required=True)
    analyze.add_argument("--probe-batch-size", type=int, default=1024)
    analyze.add_argument("--probe-train-examples", type=int, default=0)
    analyze.add_argument("--probe-valid-examples", type=int, default=0)
    analyze.add_argument("--transfer-valid-examples", type=int, default=10000)
    analyze.add_argument("--alignment-ranks", default="8,16,32,64,128")
    analyze.add_argument("--transfer-ranks", default="8,16,32,64,128,256,512,784")
    analyze.add_argument("--control-seed", type=int, default=1729)

    args = parser.parse_args()
    if args.command == "train":
        train_checkpoint(args)
    else:
        analyze_checkpoints(args)


if __name__ == "__main__":
    main()
