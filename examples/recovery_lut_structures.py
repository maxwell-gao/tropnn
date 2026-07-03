from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from tropnn.layers import PairwiseLUT, PairwiseWalshLUT
from tropnn.layers.surrogate import ste_heaviside
from tropnn.tools.benchmarking.scaling_benchmark import feature_probabilities, sample_batch

Variant = Literal["free", "walsh1", "walsh2", "coarse", "mlp", "two_sided", "two_sided_g2", "two_sided_g3"]


@dataclass(frozen=True)
class CoarseToFineSpec:
    comparisons: int
    coarse_comparisons: int

    @classmethod
    def from_args(cls, comparisons: int, coarse_comparisons: int) -> "CoarseToFineSpec":
        spec = cls(comparisons=int(comparisons), coarse_comparisons=int(coarse_comparisons))
        if spec.coarse_comparisons < 1 or spec.coarse_comparisons >= spec.comparisons:
            raise ValueError("coarse_comparisons must satisfy 1 <= coarse < comparisons")
        return spec

    @property
    def coarse_table_size(self) -> int:
        return 1 << self.coarse_comparisons


class CoarseToFinePairwiseLUT(PairwiseLUT):
    """PairwiseLUT whose payload table is coarse table plus fine residual."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        tables: int,
        comparisons: int,
        coarse_comparisons: int,
        seed: int,
        init_std: float,
        use_min_margin_ste: bool = True,
        use_output_scaling: bool = True,
        surrogate: str = "fast_sigmoid_odd",
    ) -> None:
        coarse = CoarseToFineSpec.from_args(comparisons, coarse_comparisons)
        super().__init__(
            input_dim,
            output_dim,
            tables=tables,
            comparisons=comparisons,
            backend="torch",
            seed=seed,
            lut_init_std=0.0,
            use_min_margin_ste=use_min_margin_ste,
            use_output_scaling=use_output_scaling,
            surrogate=surrogate,
        )
        del self.lut
        self.coarse = coarse
        generator = torch.Generator(device="cpu").manual_seed(seed + 1)
        self.coarse_lut = nn.Parameter(torch.randn(tables, self.coarse_table_size, output_dim, generator=generator) * init_std)
        self.fine_lut = nn.Parameter(torch.randn(tables, self.table_size, output_dim, generator=generator) * init_std)

    @property
    def coarse_comparisons(self) -> int:
        return self.coarse.coarse_comparisons

    @property
    def coarse_table_size(self) -> int:
        return self.coarse.coarse_table_size

    def materialize_lut(self, *, dtype: torch.dtype | None = None, device: torch.device | None = None) -> Tensor:
        compute_dtype = dtype if dtype is not None else self.fine_lut.dtype
        compute_device = device if device is not None else self.fine_lut.device
        full_indices = torch.arange(self.table_size, device=compute_device, dtype=torch.long)
        coarse_indices = full_indices & (self.coarse_table_size - 1)
        coarse = self.coarse_lut.to(device=compute_device, dtype=compute_dtype).index_select(1, coarse_indices)
        fine = self.fine_lut.to(device=compute_device, dtype=compute_dtype)
        return coarse + fine

    def lut_payload(self, *, dtype: torch.dtype, device: torch.device) -> Tensor:
        return self.materialize_lut(dtype=dtype, device=device)

    def payload_table(self, *, dtype: torch.dtype, device: torch.device) -> Tensor:
        return self.lut_payload(dtype=dtype, device=device)


class RecoveryMLP(nn.Module):
    """Parameter-matched dense nonlinear recovery baseline."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(self.net(x))


class TwoSidedRecovery(nn.Module):
    """Grouped comparator code-state features with sparse trainable expander writes."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        tables: int,
        comparisons: int,
        k_c: int,
        group_size: int,
        seed: int,
        use_output_scaling: bool = True,
    ) -> None:
        super().__init__()
        if tables < 1:
            raise ValueError(f"tables must be >= 1, got {tables}")
        if comparisons < 1:
            raise ValueError(f"comparisons must be >= 1, got {comparisons}")
        if k_c < 1:
            raise ValueError(f"k_c must be >= 1, got {k_c}")
        if group_size < 1 or group_size > comparisons:
            raise ValueError(f"group_size must be in [1, comparisons], got {group_size}")
        template = PairwiseLUT(
            input_dim,
            1,
            tables=tables,
            comparisons=comparisons,
            backend="torch",
            seed=seed,
            anchor_seed=seed,
        )
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.tables = int(tables)
        self.comparisons = int(comparisons)
        self.group_size = int(group_size)
        self.groups_per_table = (self.comparisons + self.group_size - 1) // self.group_size
        self.states_per_table = _two_sided_states_per_table(self.comparisons, self.group_size)
        self.routes = self.tables * self.states_per_table
        self.k_c = int(k_c)
        self.output_scale = 1.0 / math.sqrt(float(self.tables * self.groups_per_table)) if use_output_scaling else 1.0
        self.register_buffer("anchors", template.anchors.detach().clone())
        self.register_buffer("powers", 2 ** torch.arange(comparisons, dtype=torch.long))
        write_indices, write_signs = self._make_expander_writes(seed + 7919)
        self.register_buffer("write_indices", write_indices)
        self.write_weight = nn.Parameter(write_signs / math.sqrt(float(k_c)))
        self.thresholds = nn.Parameter(torch.zeros(tables, comparisons))
        self._last_indices: Tensor | None = None

    def _make_expander_writes(self, seed: int) -> tuple[Tensor, Tensor]:
        indices = torch.empty(self.routes, self.k_c, dtype=torch.long)
        signs = torch.empty(self.routes, self.k_c, dtype=torch.float32)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        for route in range(self.routes):
            for slot in range(self.k_c):
                hashed = (route * 1103515245 + slot * 12345 + 97) & 0x7FFFFFFF
                indices[route, slot] = hashed % self.output_dim
                signs[route, slot] = 1.0 if ((hashed // max(1, self.output_dim)) & 1) == 0 else -1.0
        jitter = torch.randint(0, max(1, self.output_dim), (self.routes, self.k_c), generator=gen, dtype=torch.long)
        return (indices + jitter) % self.output_dim, signs

    def _route(self, x: Tensor) -> tuple[Tensor, Tensor]:
        anchor_a = self.anchors[:, :, 0].flatten()
        anchor_b = self.anchors[:, :, 1].flatten()
        x_a = x.index_select(-1, anchor_a).view(x.shape[0], self.tables, self.comparisons)
        x_b = x.index_select(-1, anchor_b).view(x.shape[0], self.tables, self.comparisons)
        margins = x_a - x_b - self.thresholds.to(device=x.device, dtype=x.dtype)
        bits = (margins > 0).to(torch.long)
        indices = (bits * self.powers.to(device=x.device).view(1, 1, -1)).sum(dim=-1)
        return indices, margins

    def forward(self, x: Tensor) -> Tensor:
        input_dtype = x.dtype
        x32 = x.float()
        indices, margins = self._route(x32)
        group_values: list[Tensor] = []
        for start in range(0, self.comparisons, self.group_size):
            width = min(self.group_size, self.comparisons - start)
            group_margins = margins[:, :, start : start + width]
            group_bits = (group_margins > 0).to(torch.long)
            powers = (2 ** torch.arange(width, device=x.device, dtype=torch.long)).view(1, 1, width)
            group_code = (group_bits * powers).sum(dim=-1)
            group_gain = group_margins.abs().mean(dim=-1)
            group_values.append(F.one_hot(group_code, num_classes=1 << width).to(dtype=x32.dtype) * group_gain.unsqueeze(-1))
        values = torch.cat([value.reshape(x.shape[0], -1) for value in group_values], dim=1)
        batch = values.shape[0]
        weighted = values.reshape(batch, self.routes).unsqueeze(-1) * self.write_weight.to(device=x.device, dtype=x32.dtype).unsqueeze(0)
        output = torch.zeros(batch, self.output_dim, device=x.device, dtype=x32.dtype)
        write_indices = self.write_indices.to(device=x.device).view(1, self.routes, self.k_c).expand(batch, -1, -1)
        output.scatter_add_(1, write_indices.reshape(batch, -1), weighted.reshape(batch, -1))
        self._last_indices = indices.detach()
        return (output * self.output_scale).to(dtype=input_dtype)


@dataclass(frozen=True)
class ExperimentConfig:
    n_features: int
    comparisons: int
    base_tables: int
    coarse_comparisons: int
    alpha: float
    activation_density: float
    batch_size: int
    steps: int
    lr: float
    init_std: float
    two_sided_kc: int
    seed: int
    device: str
    log_losses: bool = False
    log_frequency: int = 1


def _squeeze(y: Tensor) -> Tensor:
    return y.squeeze(1) if y.ndim == 3 and y.shape[1] == 1 else y


def _two_sided_group_size(variant: Variant) -> int:
    if variant == "two_sided_g2":
        return 2
    if variant == "two_sided_g3":
        return 3
    return 1


def _two_sided_states_per_table(comparisons: int, group_size: int) -> int:
    states = 0
    for start in range(0, comparisons, group_size):
        width = min(group_size, comparisons - start)
        states += 1 << width
    return states


def _params_per_table(variant: Variant, comparisons: int, out_features: int, coarse_comparisons: int, two_sided_kc: int) -> int:
    if variant == "free":
        value_params = (1 << comparisons) * out_features
    elif variant == "walsh1":
        value_params = (1 + comparisons) * out_features
    elif variant == "walsh2":
        value_params = (1 + comparisons + comparisons * (comparisons - 1) // 2) * out_features
    elif variant == "coarse":
        value_params = ((1 << coarse_comparisons) + (1 << comparisons)) * out_features
    elif variant in {"two_sided", "two_sided_g2", "two_sided_g3"}:
        value_params = _two_sided_states_per_table(comparisons, _two_sided_group_size(variant)) * two_sided_kc
    elif variant == "mlp":
        return 0
    else:
        raise ValueError(f"unknown variant {variant!r}")
    return value_params + comparisons


def _capacity_for_budget(variant: Variant, cfg: ExperimentConfig, target_params: int) -> int:
    if variant == "mlp":
        return max(1, int(round((target_params - cfg.n_features) / (2 * cfg.n_features + 1))))
    per_table = _params_per_table(variant, cfg.comparisons, cfg.n_features, cfg.coarse_comparisons, cfg.two_sided_kc)
    return max(1, int(round(target_params / per_table)))


def _build_model(variant: Variant, cfg: ExperimentConfig, capacity: int) -> nn.Module:
    if variant == "mlp":
        return RecoveryMLP(cfg.n_features, cfg.n_features, hidden_dim=capacity)
    common = dict(
        input_dim=cfg.n_features,
        output_dim=cfg.n_features,
        tables=capacity,
        comparisons=cfg.comparisons,
        seed=cfg.seed,
        use_min_margin_ste=True,
        use_output_scaling=True,
    )
    if variant == "free":
        return PairwiseLUT(**common, lut_init_std=cfg.init_std)
    if variant == "walsh1":
        return PairwiseWalshLUT(**common, walsh_order=1, coeff_init_std=cfg.init_std)
    if variant == "walsh2":
        return PairwiseWalshLUT(**common, walsh_order=2, coeff_init_std=cfg.init_std)
    if variant == "coarse":
        return CoarseToFinePairwiseLUT(
            **common,
            coarse_comparisons=cfg.coarse_comparisons,
            init_std=cfg.init_std,
        )
    if variant in {"two_sided", "two_sided_g2", "two_sided_g3"}:
        return TwoSidedRecovery(
            cfg.n_features,
            cfg.n_features,
            tables=capacity,
            comparisons=cfg.comparisons,
            k_c=cfg.two_sided_kc,
            group_size=_two_sided_group_size(variant),
            seed=cfg.seed,
            use_output_scaling=True,
        )
    raise ValueError(f"unknown variant {variant!r}")


@torch.no_grad()
def _eval_metrics(model: nn.Module, probs: Tensor, cfg: ExperimentConfig) -> dict[str, float]:
    model.eval()
    eye = torch.eye(cfg.n_features, device=probs.device)
    response = _squeeze(model(eye)).float().cpu()
    diag = response.diag()
    offdiag = response.clone()
    offdiag.fill_diagonal_(0.0)
    weights = (probs.detach().float().cpu() / probs.sum().clamp_min(1e-12).detach().float().cpu())
    indices = getattr(model, "_last_indices", None)
    route_unique = float("nan")
    route_entropy = float("nan")
    if indices is not None and indices.numel() > 0:
        sigs = indices.squeeze(1).detach().cpu()
        _, counts = torch.unique(sigs, dim=0, return_counts=True)
        p = counts.float() / counts.sum()
        route_unique = float(counts.numel())
        route_entropy = float((-(p * p.log()).sum()).item())
    return {
        "self_gain_mean": float(diag.mean().item()),
        "self_gain_weighted_mean": float((weights * diag).sum().item()),
        "self_gain_abs_error": float((diag - 1.0).abs().mean().item()),
        "offdiag_gain": float(offdiag.square().mean().item()),
        "offdiag_weighted_energy": float((weights * offdiag.square().sum(dim=1)).sum().item()),
        "route_unique": route_unique,
        "route_entropy": route_entropy,
    }


def run_variant(variant: Variant, cfg: ExperimentConfig, target_params: int) -> dict[str, float | int | str | list[float]]:
    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    probs = feature_probabilities(cfg.n_features, cfg.alpha, cfg.activation_density, device=device)
    capacity = _capacity_for_budget(variant, cfg, target_params)
    model = _build_model(variant, cfg, capacity).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.0)

    losses: list[float] = []
    loss_curve: list[float | None] = []
    t0 = time.perf_counter()
    model.train()
    for _ in range(cfg.steps):
        x = sample_batch(probs, cfg.batch_size)
        optimizer.zero_grad(set_to_none=True)
        y = _squeeze(model(x))
        loss = F.mse_loss(y, x)
        loss.backward()
        optimizer.step()
        loss_val = float(loss.detach().item())
        losses.append(loss_val)
        step_idx = len(losses)
        should_log = (
            cfg.log_losses
            and (
                (step_idx == 1)
                or (step_idx % cfg.log_frequency == 0)
                or (step_idx == cfg.steps)
            )
        )
        if should_log:
            loss_curve.append(loss_val)
        else:
            loss_curve.append(None)
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)
    train_ms = (time.perf_counter() - t0) * 1000.0

    row: dict[str, float | int | str] = {
        "variant": variant,
        "capacity_kind": "hidden_dim" if variant == "mlp" else "tables",
        "capacity": capacity,
        "tables": 0 if variant == "mlp" else capacity,
        "hidden_dim": capacity if variant == "mlp" else 0,
        "two_sided_kc": cfg.two_sided_kc if variant in {"two_sided", "two_sided_g2", "two_sided_g3"} else 0,
        "two_sided_group_size": _two_sided_group_size(variant) if variant in {"two_sided", "two_sided_g2", "two_sided_g3"} else 0,
        "comparisons": cfg.comparisons,
        "coarse_comparisons": cfg.coarse_comparisons if variant == "coarse" else 0,
        "params": int(sum(p.numel() for p in model.parameters())),
        "target_params": target_params,
        "param_ratio": float(sum(p.numel() for p in model.parameters()) / target_params),
        "final_loss": losses[-1],
        "best_loss": min(losses),
        "train_ms": train_ms,
    }
    row.update(_eval_metrics(model, probs, cfg))
    row["loss_history"] = [float("nan") if value is None else value for value in loss_curve]
    return row


def _extract_curve(rows: list[dict[str, float | int | str | list[float]]], steps: int) -> list[dict[str, float | int | str]]:
    curve_rows: list[dict[str, float | int | str]] = []
    variant_order = [row["variant"] for row in rows]
    for step in range(steps):
        entry: dict[str, float | int | str] = {"step": step}
        for row in rows:
            losses = row.get("loss_history")
            variant = row["variant"]
            assert isinstance(losses, list)
            value = losses[step] if step < len(losses) else float("nan")
            entry[str(variant)] = value
        curve_rows.append(entry)
    return curve_rows


def _write_outputs(rows: list[dict[str, float | int | str]], output_dir: Path, cfg: ExperimentConfig) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w") as handle:
        json.dump(asdict(cfg), handle, indent=2, sort_keys=True)
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(rows, handle, indent=2, sort_keys=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with (output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    if cfg.log_losses:
        variant_order = [str(row["variant"]) for row in rows]
        with (output_dir / "loss_curves.json").open("w") as handle:
            json.dump(_extract_curve(rows, cfg.steps), handle, indent=2)
        with (output_dir / "loss_curves.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["step"] + variant_order)
            for row in _extract_curve(rows, cfg.steps):
                writer.writerow([row["step"]] + [row[str(v)] for v in variant_order])


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare dense MLP, free, Walsh, coarse-to-fine, and grouped two-sided Pairwise structures on the recovery toy.")
    parser.add_argument("--variants", type=str, default="free,walsh1,walsh2,coarse,mlp,two_sided,two_sided_g2,two_sided_g3")
    parser.add_argument("--n-features", type=int, default=128)
    parser.add_argument("--comparisons", type=int, default=6)
    parser.add_argument("--base-tables", type=int, default=16)
    parser.add_argument("--coarse-comparisons", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--activation-density", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--init-std", type=float, default=0.02)
    parser.add_argument("--two-sided-kc", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-losses", action="store_true", help="Record per-step loss curve")
    parser.add_argument("--log-frequency", type=int, default=1, help="Only record every k-th step when log_losses enabled")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--output-dir", type=Path, default=Path("python/results/recovery_lut_structures"))
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    allowed = {"free", "walsh1", "walsh2", "coarse", "mlp", "two_sided", "two_sided_g2", "two_sided_g3"}
    bad = sorted(set(variants) - allowed)
    if bad:
        raise ValueError(f"unknown variants {bad}; expected subset of {sorted(allowed)}")
    cfg = ExperimentConfig(
        n_features=args.n_features,
        comparisons=args.comparisons,
        base_tables=args.base_tables,
        coarse_comparisons=args.coarse_comparisons,
        alpha=args.alpha,
        activation_density=args.activation_density,
        batch_size=args.batch_size,
        steps=args.steps,
        lr=args.lr,
        init_std=args.init_std,
        two_sided_kc=args.two_sided_kc,
        seed=args.seed,
        log_losses=args.log_losses,
        log_frequency=max(1, args.log_frequency),
        device=args.device,
    )
    target_params = args.base_tables * _params_per_table("free", args.comparisons, args.n_features, args.coarse_comparisons, args.two_sided_kc)
    rows = [run_variant(variant, cfg, target_params) for variant in variants]  # type: ignore[arg-type]
    _write_outputs(rows, args.output_dir, cfg)
    for row in rows:
        print(
            f"{row['variant']:>9} {row['capacity_kind']}={row['capacity']:4d} params={row['params']:9d} "
            f"ratio={row['param_ratio']:.3f} final={row['final_loss']:.6f} best={row['best_loss']:.6f} "
            f"self_err={row['self_gain_abs_error']:.4f} offdiag={row['offdiag_gain']:.6f} "
            f"route_unique={row['route_unique']:.0f}"
        )
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
