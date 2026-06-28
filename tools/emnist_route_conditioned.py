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

from ..layers import PairwiseLUT, PairwiseRoute
from .emnist_discrete_payload import RowLocalDiscretePayloadOptimizer
from .emnist_payload_dtype_sweep import _build_local_loaders, _eval_model, _loader_examples

RouteVariant = Literal["plain", "sign_permutation", "sparse_mixing", "butterfly", "anchor_transform", "transformed_coordinates"]
OptimizerKind = Literal["adamw", "discrete"]


@dataclass(frozen=True)
class RouteConditionedRow:
    variant: str
    optimizer: str
    discrete_method: str
    depth: int
    tables: int
    comparisons: int
    anchor_policy: str
    mix_strength: float
    lut_init_std: float
    backend: str
    device: str
    split: str
    train_examples: int
    valid_examples: int
    epochs: int
    params: int
    train_loss: float
    train_acc: float
    valid_loss: float
    valid_acc: float
    finite_loss_steps: int
    nonfinite_loss_steps: int
    nonfinite_grad_steps: int
    route_entropy: float
    route_transition: float
    route_unique_fraction: float
    payload_commit_fraction: float
    payload_saturation_fraction: float
    changed_codes: int
    total_codes: int


class RouteConditionedMixer(nn.Module):
    """Cheap route-conditioned coordinate transforms used only for comparisons."""

    def __init__(self, dim: int, tables: int, variant: RouteVariant, *, seed: int, mix_strength: float) -> None:
        super().__init__()
        self.dim = int(dim)
        self.tables = int(tables)
        self.variant = variant
        self.alpha = nn.Parameter(torch.tensor(float(mix_strength), dtype=torch.float32))
        self.register_buffer("coords", torch.arange(dim, dtype=torch.long))
        gen = torch.Generator(device="cpu").manual_seed(seed)
        weights = torch.randint(1, 2**15, (tables,), generator=gen, dtype=torch.long) * 2 + 1
        self.register_buffer("hash_weights", weights)

    def forward(self, x: Tensor, prev_indices: Tensor | None) -> Tensor:
        if self.variant == "plain" or prev_indices is None:
            return x
        if self.variant == "sign_permutation":
            return self._sign_permutation(x, prev_indices)
        if self.variant == "sparse_mixing":
            return self._sparse_mixing(x, prev_indices)
        if self.variant == "butterfly":
            return self._butterfly(x, prev_indices)
        if self.variant == "transformed_coordinates":
            return self._transformed_coordinates(x, prev_indices)
        if self.variant == "anchor_transform":
            return x
        raise AssertionError(f"unreachable variant {self.variant!r}")

    def _hash(self, prev_indices: Tensor, modulus: int) -> Tensor:
        weights = self.hash_weights.to(device=prev_indices.device)[: prev_indices.shape[-1]]
        return (prev_indices.long() * weights.view(1, -1)).sum(dim=-1).remainder(modulus)

    def _shifted(self, x: Tensor, shift: Tensor) -> Tensor:
        idx = (self.coords.to(device=x.device).view(1, -1) + shift.view(-1, 1)).remainder(self.dim)
        return x.gather(1, idx)

    def _route_sign(self, prev_indices: Tensor, width: int) -> Tensor:
        h = self._hash(prev_indices, 2 * width).view(-1, 1)
        base = torch.arange(width, device=prev_indices.device, dtype=torch.long).view(1, -1)
        return torch.where((h + base).remainder(2) == 0, 1.0, -1.0)

    def _sign_permutation(self, x: Tensor, prev_indices: Tensor) -> Tensor:
        shift = self._hash(prev_indices, self.dim)
        sign = self._route_sign(prev_indices, self.dim).to(dtype=x.dtype)
        return self._shifted(x, shift) * sign

    def _sparse_mixing(self, x: Tensor, prev_indices: Tensor) -> Tensor:
        shift = self._hash(prev_indices, self.dim - 1) + 1
        sign = self._route_sign(prev_indices, self.dim).to(dtype=x.dtype)
        return x + self.alpha.tanh().to(dtype=x.dtype) * sign * self._shifted(x, shift)

    def _butterfly(self, x: Tensor, prev_indices: Tensor) -> Tensor:
        pair_count = self.dim // 2
        if pair_count == 0:
            return x
        even = x[:, 0 : 2 * pair_count : 2]
        odd = x[:, 1 : 2 * pair_count : 2]
        sign = self._route_sign(prev_indices, pair_count).to(dtype=x.dtype)
        scale = math.sqrt(0.5)
        y_even = (even + sign * odd) * scale
        y_odd = (even - sign * odd) * scale
        y = torch.empty_like(x)
        y[:, 0 : 2 * pair_count : 2] = y_even
        y[:, 1 : 2 * pair_count : 2] = y_odd
        if self.dim % 2:
            y[:, -1] = x[:, -1]
        return y

    def _transformed_coordinates(self, x: Tensor, prev_indices: Tensor) -> Tensor:
        shift_a = self._hash(prev_indices, self.dim)
        shift_b = (self._hash(prev_indices.roll(shifts=1, dims=-1), self.dim - 1) + 1).remainder(self.dim)
        sign = self._route_sign(prev_indices, self.dim).to(dtype=x.dtype)
        return self._shifted(x, shift_a) * sign + self.alpha.tanh().to(dtype=x.dtype) * self._shifted(x, shift_b)


class RouteConditionedBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        tables: int,
        comparisons: int,
        seed: int,
        backend: str,
        anchor_policy: str,
        lut_init_std: float,
        variant: RouteVariant,
        mix_strength: float,
    ) -> None:
        super().__init__()
        self.variant = variant
        self.layer = PairwiseLUT(
            dim,
            dim,
            tables=tables,
            comparisons=comparisons,
            seed=seed,
            backend=backend,  # type: ignore[arg-type]
            anchor_policy=anchor_policy,
            lut_init_std=lut_init_std,
        )
        self.mixer = RouteConditionedMixer(dim, tables, variant, seed=seed + 1009, mix_strength=mix_strength)

    def forward(self, x: Tensor, prev_indices: Tensor | None) -> tuple[Tensor, Tensor]:
        if self.variant == "anchor_transform" and prev_indices is not None:
            output, route = self._anchor_transform_compute(x, prev_indices)
        else:
            x_cmp = self.mixer(x, prev_indices)
            output, route = self._layer_compute(x_cmp)
        return x + output, route.indices.squeeze(1) if route.indices.ndim == 3 else route.indices

    def _layer_compute(self, x: Tensor) -> tuple[Tensor, PairwiseRoute]:
        output, route = self.layer.compute(x.unsqueeze(1), compute_dtype=torch.float32, training=self.training)
        return output.squeeze(1), PairwiseRoute(route.indices.squeeze(1), route.margins.squeeze(1))

    def _anchor_transform_compute(self, x: Tensor, prev_indices: Tensor) -> tuple[Tensor, PairwiseRoute]:
        route = self._dynamic_anchor_route(x, prev_indices)
        lut = self.layer.lut_payload(dtype=torch.float32, device=x.device)
        output = self.layer.lut_forward(route, lut, compute_dtype=torch.float32)
        if self.training and bool(getattr(self.layer.thresholds, "requires_grad", False)):
            output = output + self.layer.lut_backward_surrogate(route, lut).to(output.dtype)
        return output, route

    def _dynamic_anchor_route(self, x: Tensor, prev_indices: Tensor) -> PairwiseRoute:
        batch = x.shape[0]
        anchors = self.layer.anchors.to(device=x.device)
        tables, comparisons, _ = anchors.shape
        prev_tables = prev_indices.shape[-1]
        t = torch.arange(tables, device=x.device, dtype=torch.long).view(1, tables, 1)
        c = torch.arange(comparisons, device=x.device, dtype=torch.long).view(1, 1, comparisons)
        selector_a = (t + c).remainder(prev_tables).expand(batch, -1, -1)
        selector_b = (t * 3 + c + 1).remainder(prev_tables).expand(batch, -1, -1)
        prev = prev_indices.long().unsqueeze(1).expand(-1, tables, -1)
        prev_a = prev.gather(2, selector_a)
        prev_b = prev.gather(2, selector_b)
        offset_a = (prev_a * 17 + t * 31 + c * 7).remainder(self.layer.input_dim)
        offset_b = (prev_b * 19 + t * 11 + c * 13 + 1).remainder(self.layer.input_dim)
        dyn_a = (anchors[:, :, 0].view(1, tables, comparisons) + offset_a).remainder(self.layer.input_dim)
        dyn_b = (anchors[:, :, 1].view(1, tables, comparisons) + offset_b).remainder(self.layer.input_dim)
        x_for_gather = x.unsqueeze(1).expand(-1, tables, -1)
        margins = x_for_gather.gather(2, dyn_a) - x_for_gather.gather(2, dyn_b)
        margins = margins - self.layer.thresholds.to(dtype=x.dtype, device=x.device).view(1, tables, comparisons)
        powers = self.layer.powers.to(device=x.device).view(1, 1, comparisons)
        indices = ((margins > 0).to(torch.long) * powers).sum(dim=-1)
        return PairwiseRoute(indices, margins)


class RouteConditionedEmnistClassifier(nn.Module):
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
        lut_init_std: float,
        variant: RouteVariant,
        mix_strength: float,
    ) -> None:
        super().__init__()
        self.variant = variant
        self.blocks = nn.ModuleList(
            [
                RouteConditionedBlock(
                    input_dim,
                    tables=tables,
                    comparisons=comparisons,
                    seed=seed + block,
                    backend=backend,
                    anchor_policy=anchor_policy,
                    lut_init_std=lut_init_std,
                    variant=variant,
                    mix_strength=mix_strength,
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
            lut_init_std=lut_init_std,
        )
        self.last_routes: list[Tensor] = []

    def forward(self, x: Tensor) -> Tensor:
        y = x.flatten(1)
        prev: Tensor | None = None
        routes: list[Tensor] = []
        for block in self.blocks:
            y, prev = block(y, prev)
            routes.append(prev.detach())
        self.last_routes = routes
        output, _route = self.readout.compute(y.unsqueeze(1), compute_dtype=torch.float32, training=self.training)
        return output.squeeze(1)


def _pairwise_layers(model: nn.Module) -> list[PairwiseLUT]:
    return [module for module in model.modules() if isinstance(module, PairwiseLUT)]


def _non_lut_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [param for name, param in model.named_parameters() if param.requires_grad and not name.endswith(".lut")]


def _route_stats(routes: list[Tensor], table_size: int) -> tuple[float, float, float]:
    if not routes:
        return 0.0, 0.0, 0.0
    entropies: list[float] = []
    unique_fracs: list[float] = []
    transitions: list[float] = []
    for route in routes:
        flat = route.reshape(-1).long()
        counts = torch.bincount(flat, minlength=table_size).float()
        probs = counts / counts.sum().clamp_min(1.0)
        nz = probs > 0
        entropy = -(probs[nz] * probs[nz].log()).sum() / math.log(table_size)
        entropies.append(float(entropy.detach().cpu().item()))
        unique_fracs.append(float((counts > 0).float().mean().detach().cpu().item()))
    for left, right in zip(routes, routes[1:]):
        transitions.append(float((left != right).float().mean().detach().cpu().item()))
    return sum(entropies) / max(1, len(entropies)), sum(transitions) / max(1, len(transitions)), sum(unique_fracs) / max(1, len(unique_fracs))


def _grad_finite(model: nn.Module) -> bool:
    for param in model.parameters():
        if param.grad is not None and not bool(torch.isfinite(param.grad).all().item()):
            return False
    return True


def _train(model: RouteConditionedEmnistClassifier, train_loader, args: argparse.Namespace, *, device: torch.device) -> tuple[float, float, dict[str, float | int]]:
    if args.optimizer == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        payload_opt = None
        other_opt = None
    else:
        payload_opt = RowLocalDiscretePayloadOptimizer(
            _pairwise_layers(model),
            method=args.discrete_method,
            bitwidth=4,
            lr=args.payload_lr,
            step_size=args.payload_step_size,
            accumulator="float_ef",
            accumulator_unit=args.accumulator_unit,
            beta1=args.beta1,
            beta2=args.beta2,
            eps=args.eps,
            bop_threshold=1.0,
            adam_m_unit=1e-5,
            adam_v_unit=1e-8,
            row_frequency_normalization=False,
            row_frequency_decay=0.95,
            row_frequency_power=0.5,
            row_frequency_eps=1e-3,
        )
        other_opt = torch.optim.AdamW(_non_lut_parameters(model), lr=args.lr, weight_decay=args.weight_decay)
        opt = None

    finite_loss_steps = 0
    nonfinite_loss_steps = 0
    nonfinite_grad_steps = 0
    route_entropies: list[float] = []
    route_transitions: list[float] = []
    route_unique_fracs: list[float] = []
    commit_fracs: list[float] = []
    saturation_fracs: list[float] = []
    changed_codes = 0
    total_codes = sum(layer.lut.numel() for layer in _pairwise_layers(model))
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
            if opt is not None:
                opt.zero_grad(set_to_none=True)
            else:
                other_opt.zero_grad(set_to_none=True)  # type: ignore[union-attr]
                payload_opt.zero_grad()  # type: ignore[union-attr]
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            batch = int(y.numel())
            if torch.isfinite(loss):
                finite_loss_steps += 1
            else:
                nonfinite_loss_steps += 1
            loss.backward()
            if not _grad_finite(model):
                nonfinite_grad_steps += 1
            if opt is not None:
                opt.step()
            else:
                other_opt.step()  # type: ignore[union-attr]
                stats = payload_opt.step()  # type: ignore[union-attr]
                commit_fracs.append(stats.commit_fraction)
                saturation_fracs.append(stats.saturation_fraction)
                changed_codes += stats.changed_codes
            entropy, transition, unique_frac = _route_stats(model.last_routes, 1 << args.comparisons)
            route_entropies.append(entropy)
            route_transitions.append(transition)
            route_unique_fracs.append(unique_frac)
            total_loss += float(loss.detach().item()) * batch
            total_correct += int((logits.argmax(dim=-1) == y).sum().item())
            total_seen += batch
        last_train_loss = total_loss / max(1, total_seen)
        last_train_acc = total_correct / max(1, total_seen)

    return last_train_loss, last_train_acc, {
        "finite_loss_steps": finite_loss_steps,
        "nonfinite_loss_steps": nonfinite_loss_steps,
        "nonfinite_grad_steps": nonfinite_grad_steps,
        "route_entropy": sum(route_entropies) / max(1, len(route_entropies)),
        "route_transition": sum(route_transitions) / max(1, len(route_transitions)),
        "route_unique_fraction": sum(route_unique_fracs) / max(1, len(route_unique_fracs)),
        "payload_commit_fraction": sum(commit_fracs) / max(1, len(commit_fracs)),
        "payload_saturation_fraction": sum(saturation_fracs) / max(1, len(saturation_fracs)),
        "changed_codes": changed_codes,
        "total_codes": total_codes,
    }


def run(args: argparse.Namespace) -> RouteConditionedRow:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    train_loader, valid_loader, classes = _build_local_loaders(args)
    model = RouteConditionedEmnistClassifier(
        input_dim=28 * 28,
        num_classes=classes,
        depth=args.depth,
        tables=args.tables,
        comparisons=args.comparisons,
        seed=args.seed,
        backend=args.backend,
        anchor_policy=args.anchor_policy,
        lut_init_std=args.lut_init_std,
        variant=args.variant,
        mix_strength=args.mix_strength,
    ).to(device)
    train_loss, train_acc, stats = _train(model, train_loader, args, device=device)
    valid_loss, valid_acc = _eval_model(model, valid_loader, device=device)
    return RouteConditionedRow(
        variant=args.variant,
        optimizer=args.optimizer,
        discrete_method=args.discrete_method if args.optimizer == "discrete" else "none",
        depth=args.depth,
        tables=args.tables,
        comparisons=args.comparisons,
        anchor_policy=args.anchor_policy,
        mix_strength=args.mix_strength,
        lut_init_std=args.lut_init_std,
        backend=args.backend,
        device=str(device),
        split=args.split,
        train_examples=_loader_examples(train_loader),
        valid_examples=_loader_examples(valid_loader),
        epochs=args.epochs,
        params=sum(param.numel() for param in model.parameters()),
        train_loss=train_loss,
        train_acc=train_acc,
        valid_loss=valid_loss,
        valid_acc=valid_acc,
        finite_loss_steps=int(stats["finite_loss_steps"]),
        nonfinite_loss_steps=int(stats["nonfinite_loss_steps"]),
        nonfinite_grad_steps=int(stats["nonfinite_grad_steps"]),
        route_entropy=float(stats["route_entropy"]),
        route_transition=float(stats["route_transition"]),
        route_unique_fraction=float(stats["route_unique_fraction"]),
        payload_commit_fraction=float(stats["payload_commit_fraction"]),
        payload_saturation_fraction=float(stats["payload_saturation_fraction"]),
        changed_codes=int(stats["changed_codes"]),
        total_codes=int(stats["total_codes"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="EMNIST balanced route-conditioned PairwiseLUT depth experiments.")
    parser.add_argument("--root", default="data/emnist")
    parser.add_argument("--split", default="balanced")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--backend", choices=("auto", "torch", "tilelang", "triton"), default="tilelang")
    parser.add_argument("--variant", choices=("plain", "sign_permutation", "sparse_mixing", "butterfly", "anchor_transform", "transformed_coordinates"), default="plain")
    parser.add_argument("--optimizer", choices=("adamw", "discrete"), default="adamw")
    parser.add_argument("--discrete-method", choices=("ef_sgd", "adam_ef", "factored_adam_ef", "integer_adam_ef", "scaled_integer_adam_ef"), default="adam_ef")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--payload-lr", type=float, default=0.005)
    parser.add_argument("--payload-step-size", type=float, default=0.05)
    parser.add_argument("--accumulator-unit", type=float, default=0.0002)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--tables", type=int, default=64)
    parser.add_argument("--comparisons", type=int, default=6)
    parser.add_argument("--lut-init-std", type=float, default=0.0)
    parser.add_argument("--mix-strength", type=float, default=0.125)
    parser.add_argument("--anchor-policy", default="permuted")
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-test", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/emnist_route_conditioned/summary.csv")
    args = parser.parse_args()

    row = run(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(row).keys()))
        writer.writeheader()
        writer.writerow(asdict(row))
    print(
        f"variant={row.variant} optimizer={row.optimizer}:{row.discrete_method} depth={row.depth} "
        f"train_loss={row.train_loss:.4f} train_acc={row.train_acc:.4f} "
        f"valid_loss={row.valid_loss:.4f} valid_acc={row.valid_acc:.4f} "
        f"route_entropy={row.route_entropy:.4f} route_transition={row.route_transition:.4f} "
        f"commit={row.payload_commit_fraction:.6f} saturation={row.payload_saturation_fraction:.6f} "
        f"nonfinite_loss={row.nonfinite_loss_steps} nonfinite_grad={row.nonfinite_grad_steps}",
        flush=True,
    )
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
