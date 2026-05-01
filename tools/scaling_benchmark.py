from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..backend import Backend, trop_scores
from ..layers import PairwiseLinear, TropFanZeroDenseLinear, TropLinear, TropZeroDenseLinear
from ..layers.tropical import _minface_mix, _top2_indices

FAMILIES = (
    "paper",
    "untied_paper",
    "linear",
    "tropical_lowrank",
    "tied_tropical_lowrank",
    "tropical_zero_dense",
    "tied_tropical_zero_dense",
    "tropfan_zero_dense",
    "tied_tropfan_zero_dense",
    "pairwise",
    "tied_pairwise",
)
CODE_SCALE_MODES = ("sqrt", "linear", "none")
PAPER_FAMILIES = ("paper", "untied_paper")
LOWRANK_FAMILIES = ("tropical_lowrank", "tied_tropical_lowrank")
COORD_ZERO_DENSE_FAMILIES = ("tropical_zero_dense", "tied_tropical_zero_dense")
FAN_ZERO_DENSE_FAMILIES = ("tropfan_zero_dense", "tied_tropfan_zero_dense")
PAIRWISE_FAMILIES = ("pairwise", "tied_pairwise")
ZERO_DENSE_FAMILIES = COORD_ZERO_DENSE_FAMILIES + FAN_ZERO_DENSE_FAMILIES
ROUTED_FAMILIES = LOWRANK_FAMILIES + ZERO_DENSE_FAMILIES + PAIRWISE_FAMILIES
TIED_RECOVERY_FAMILIES = (
    "tied_tropical_lowrank",
    "tied_tropical_zero_dense",
    "tied_tropfan_zero_dense",
    "tied_pairwise",
)
ROUTE_METRIC_FIELDS = ("route_unique", "route_collision_mean", "route_entropy", "avg_margin")
BRIDGE_CAPACITY_KEYS = (
    "bridge_K_vec_a4",
    "bridge_K_vec_a16",
    "bridge_K_chamb_a4",
    "bridge_K_chamb_a16",
    "bridge_chamber_distinct",
)
GROUP_METRIC_FIELDS = (
    ("mean_loss_per_activation", "loss_per_activation"),
    ("mean_overlap_times_dim", "overlap_times_dim"),
    ("mean_frequency_weighted_overlap_times_dim", "frequency_weighted_overlap_times_dim"),
    ("mean_frequency_pair_weighted_overlap_times_dim", "frequency_pair_weighted_overlap_times_dim"),
    ("mean_route_entropy_norm", "route_entropy_norm"),
    ("mean_represented_fraction", "represented_fraction"),
    ("mean_self_gain_weighted", "self_gain_weighted_mean"),
    ("mean_self_gain_weighted_mse", "self_gain_weighted_mse"),
    ("mean_offdiag_weighted_energy", "offdiag_weighted_energy"),
)

BenchmarkRow = dict[str, float | int | str]
ModelBuilder = Callable[["RunConfig"], nn.Module]
RepresentationExtractor = Callable[[nn.Module, int, torch.device], Tensor]


@dataclass(frozen=True)
class RunConfig:
    family: str
    n_features: int
    model_dim: int
    alpha: float
    activation_density: float
    batch_size: int
    steps: int
    lr: float
    paper_lr: float
    weight_decay: float
    heads: int
    cells: int
    code_scale_mode: str
    route_terms: int
    seed: int
    device: str
    backend: str
    output_lr_mult: float = 1.0
    fan_value_mode: str = "site"
    fan_basis_rank: int = 16
    tables: int = 0
    comparisons: int = 0


def _eye(n_features: int, device: torch.device) -> Tensor:
    return torch.eye(n_features, device=device)


def _flatten_single_sequence(x: Tensor) -> tuple[Tensor, bool]:
    if x.ndim == 3 and x.shape[1] == 1:
        return x.squeeze(1), True
    return x, False


def _squeeze_single_sequence(y: Tensor) -> Tensor:
    return y.squeeze(1) if y.ndim == 3 and y.shape[1] == 1 else y


class PaperFeatureRecovery(nn.Module):
    def __init__(self, n_features: int, model_dim: int, *, seed: int) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.weight = nn.Parameter(torch.randn(n_features, model_dim) / math.sqrt(model_dim))
        self.bias = nn.Parameter(torch.zeros(n_features))

    def encode(self, x: Tensor) -> Tensor:
        return x @ self.weight

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(self.encode(x) @ self.weight.t() + self.bias)


class UntiedPaperFeatureRecovery(nn.Module):
    def __init__(self, n_features: int, model_dim: int, *, seed: int) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.encoder_weight = nn.Parameter(torch.randn(n_features, model_dim) / math.sqrt(model_dim))
        self.decoder_weight = nn.Parameter(torch.randn(n_features, model_dim) / math.sqrt(model_dim))
        self.bias = nn.Parameter(torch.zeros(n_features))

    def encode(self, x: Tensor) -> Tensor:
        return x @ self.encoder_weight

    def readout_representations(self) -> Tensor:
        return self.decoder_weight

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(self.encode(x) @ self.decoder_weight.t() + self.bias)


class LinearRecovery(nn.Module):
    def __init__(self, n_features: int, model_dim: int, *, seed: int) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.encoder = nn.Linear(n_features, model_dim, bias=False)
        self.decoder = nn.Linear(model_dim, n_features)

    def encode(self, x: Tensor) -> Tensor:
        return F.relu(self.encoder(x))

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(self.decoder(self.encode(x)))


class TiedRecoveryBase(nn.Module):
    """Shared paper-style recovery shell for routed feature representations."""

    router: nn.Module

    def __init__(self, n_features: int, model_dim: int) -> None:
        super().__init__()
        self.n_features = n_features
        self.model_dim = model_dim
        self.bias = nn.Parameter(torch.zeros(n_features))

    @property
    def _last_indices(self) -> Tensor | None:
        return getattr(self.router, "_last_indices", None)

    @property
    def _last_margins(self) -> Tensor | None:
        return getattr(self.router, "_last_margins", None)

    def effective_representations(self, *, training: bool | None = None) -> tuple[Tensor, Tensor, Tensor]:
        raise NotImplementedError

    def _identity_router_representations(self, *, score_heads: int | None = None) -> tuple[Tensor, Tensor, Tensor]:
        device = self.bias.device
        reps = _squeeze_single_sequence(self.router(_eye(self.n_features, device))).float()
        heads = int(score_heads if score_heads is not None else getattr(self.router, "heads", 0))
        empty_scores = torch.empty(self.n_features, heads, 0, device=device, dtype=reps.dtype)
        return reps, reps.view(self.n_features, 1, self.model_dim), empty_scores

    def encode(self, x: Tensor) -> Tensor:
        reps, _, _ = self.effective_representations(training=False)
        return x.to(reps.dtype) @ reps

    def forward(self, x: Tensor) -> Tensor:
        x, restore_sequence = _flatten_single_sequence(x)
        reps, _, _ = self.effective_representations(training=self.training)
        hidden = x.to(reps.dtype) @ reps
        output = F.relu(hidden @ reps.t() + self.bias.to(dtype=reps.dtype, device=reps.device))
        return output.unsqueeze(1) if restore_sequence else output


class CodeScaledTiedRecovery(TiedRecoveryBase):
    @property
    def code_scale(self) -> float:
        return self.router.code_scale  # type: ignore[attr-defined]

    @code_scale.setter
    def code_scale(self, value: float) -> None:
        self.router.code_scale = value  # type: ignore[attr-defined]


class TiedTropicalLowRankRecovery(CodeScaledTiedRecovery):
    """Paper-style tied recovery whose feature vectors come from TropLinear codes."""

    router: TropLinear

    def __init__(
        self,
        n_features: int,
        model_dim: int,
        *,
        heads: int,
        cells: int,
        backend: Backend,
        seed: int,
    ) -> None:
        super().__init__(n_features, model_dim)
        torch.manual_seed(seed)
        self.router = TropLinear(n_features, model_dim, heads=heads, cells=cells, code_dim=model_dim, backend=backend, seed=seed)

    @property
    def layer(self) -> TropLinear:
        return self.router

    def effective_representations(self, *, training: bool | None = None) -> tuple[Tensor, Tensor, Tensor]:
        use_training_routes = self.training if training is None else training
        device = self.bias.device
        eye = _eye(self.n_features, device)
        latent = self.router._project_input(eye.unsqueeze(1), torch.float32)
        score_backend = "torch" if self.router.backend == "tilelang" else self.router.backend
        scores = trop_scores(
            latent,
            self.router.router_weight.to(dtype=torch.float32, device=device),
            self.router.router_bias.to(dtype=torch.float32, device=device),
            backend=score_backend,
        )
        winner_idx, runner_idx, margins = _top2_indices(scores)
        if use_training_routes:
            winner_codes = self.router._selected_codes(winner_idx, input_device=device, compute_dtype=torch.float32)
            runner_codes = self.router._selected_codes(runner_idx, input_device=device, compute_dtype=torch.float32)
            codes = _minface_mix(winner_codes, runner_codes, margins)
        else:
            codes = self.router._selected_codes(winner_idx, input_device=device, compute_dtype=torch.float32)
        self.router._last_indices = winner_idx.detach()
        self.router._last_margins = margins.detach()
        reps = (latent + codes.sum(dim=2) * self.router.code_scale).squeeze(1)
        return reps, codes.squeeze(1), scores.squeeze(1)


class TiedZeroDenseRecovery(CodeScaledTiedRecovery):
    """Paper-style tied recovery whose feature vectors come from zero-dense routing."""

    router: TropZeroDenseLinear

    def __init__(
        self,
        n_features: int,
        model_dim: int,
        *,
        heads: int,
        cells: int,
        route_terms: int,
        seed: int,
    ) -> None:
        super().__init__(n_features, model_dim)
        self.router = TropZeroDenseLinear(n_features, model_dim, heads=heads, cells=cells, route_terms=route_terms, seed=seed)

    def effective_representations(self, *, training: bool | None = None) -> tuple[Tensor, Tensor, Tensor]:
        del training
        return self._identity_router_representations()


class TiedFanZeroDenseRecovery(CodeScaledTiedRecovery):
    """Paper-style tied recovery whose feature vectors come from zero-dense fan routing."""

    router: TropFanZeroDenseLinear

    def __init__(
        self,
        n_features: int,
        model_dim: int,
        *,
        heads: int,
        cells: int,
        fan_value_mode: str,
        fan_basis_rank: int,
        seed: int,
    ) -> None:
        super().__init__(n_features, model_dim)
        self.router = TropFanZeroDenseLinear(
            n_features,
            model_dim,
            heads=heads,
            cells=cells,
            code_dim=model_dim,
            fan_value_mode=fan_value_mode,  # type: ignore[arg-type]
            fan_basis_rank=fan_basis_rank,
            seed=seed,
        )

    def effective_representations(self, *, training: bool | None = None) -> tuple[Tensor, Tensor, Tensor]:
        del training
        return self._identity_router_representations()


class TiedPairwiseRecovery(TiedRecoveryBase):
    """Paper-style tied recovery whose feature vectors come from a pairwise comparator LUT."""

    router: PairwiseLinear

    def __init__(
        self,
        n_features: int,
        model_dim: int,
        *,
        tables: int,
        comparisons: int,
        seed: int,
    ) -> None:
        super().__init__(n_features, model_dim)
        self.router = PairwiseLinear(
            n_features,
            model_dim,
            tables=tables,
            comparisons=comparisons,
            seed=seed,
        )

    def effective_representations(self, *, training: bool | None = None) -> tuple[Tensor, Tensor, Tensor]:
        del training
        return self._identity_router_representations(score_heads=0)


def feature_probabilities(n_features: int, alpha: float, activation_density: float, *, device: torch.device) -> Tensor:
    if n_features < 1:
        raise ValueError(f"n_features must be >= 1, got {n_features}")
    if activation_density <= 0:
        raise ValueError(f"activation_density must be positive, got {activation_density}")
    ranks = torch.arange(1, n_features + 1, device=device, dtype=torch.float32)
    probs = ranks.pow(-alpha)
    probs = probs / probs.sum() * activation_density
    max_prob = float(probs.max().item())
    if max_prob > 1.0:
        raise ValueError(
            f"activation_density={activation_density} is too high for n_features={n_features}, alpha={alpha}; max probability is {max_prob:.3f}"
        )
    return probs


def sample_batch(probs: Tensor, batch_size: int, *, generator: torch.Generator | None = None) -> Tensor:
    active = torch.rand((batch_size, probs.numel()), device=probs.device, generator=generator) < probs.view(1, -1)
    values = torch.rand((batch_size, probs.numel()), device=probs.device, generator=generator) * 2.0
    return active.to(values.dtype) * values


def _code_scale(heads: int, mode: str) -> float:
    if mode == "sqrt":
        return 1.0 / math.sqrt(heads)
    if mode == "linear":
        return 1.0 / heads
    if mode == "none":
        return 1.0
    raise ValueError(f"unknown code_scale_mode {mode!r}; expected one of {CODE_SCALE_MODES}")


def _apply_code_scale(model: nn.Module, config: RunConfig) -> nn.Module:
    model.code_scale = _code_scale(config.heads, config.code_scale_mode)  # type: ignore[attr-defined]
    return model


def _pairwise_shape(config: RunConfig) -> tuple[int, int]:
    tables = config.tables if config.tables > 0 else max(4, config.heads)
    comparisons = config.comparisons if config.comparisons > 0 else 4
    return tables, comparisons


def _build_paper(config: RunConfig) -> nn.Module:
    return PaperFeatureRecovery(config.n_features, config.model_dim, seed=config.seed)


def _build_untied_paper(config: RunConfig) -> nn.Module:
    return UntiedPaperFeatureRecovery(config.n_features, config.model_dim, seed=config.seed)


def _build_linear(config: RunConfig) -> nn.Module:
    return LinearRecovery(config.n_features, config.model_dim, seed=config.seed)


def _build_tropical_lowrank(config: RunConfig) -> nn.Module:
    layer = TropLinear(
        config.n_features,
        config.n_features,
        heads=config.heads,
        cells=config.cells,
        code_dim=config.model_dim,
        backend=config.backend,
        seed=config.seed,
    )
    return _apply_code_scale(layer, config)


def _build_tied_tropical_lowrank(config: RunConfig) -> nn.Module:
    model = TiedTropicalLowRankRecovery(
        config.n_features,
        config.model_dim,
        heads=config.heads,
        cells=config.cells,
        backend=config.backend,
        seed=config.seed,
    )
    return _apply_code_scale(model, config)


def _build_tropical_zero_dense(config: RunConfig) -> nn.Module:
    layer = TropZeroDenseLinear(
        config.n_features,
        config.n_features,
        heads=config.heads,
        cells=config.cells,
        route_terms=config.route_terms,
        seed=config.seed,
    )
    return _apply_code_scale(layer, config)


def _build_tied_tropical_zero_dense(config: RunConfig) -> nn.Module:
    model = TiedZeroDenseRecovery(
        config.n_features,
        config.model_dim,
        heads=config.heads,
        cells=config.cells,
        route_terms=config.route_terms,
        seed=config.seed,
    )
    return _apply_code_scale(model, config)


def _build_tropfan_zero_dense(config: RunConfig) -> nn.Module:
    layer = TropFanZeroDenseLinear(
        config.n_features,
        config.n_features,
        heads=config.heads,
        cells=config.cells,
        code_dim=config.model_dim,
        fan_value_mode=config.fan_value_mode,  # type: ignore[arg-type]
        fan_basis_rank=config.fan_basis_rank,
        seed=config.seed,
    )
    return _apply_code_scale(layer, config)


def _build_tied_tropfan_zero_dense(config: RunConfig) -> nn.Module:
    model = TiedFanZeroDenseRecovery(
        config.n_features,
        config.model_dim,
        heads=config.heads,
        cells=config.cells,
        fan_value_mode=config.fan_value_mode,
        fan_basis_rank=config.fan_basis_rank,
        seed=config.seed,
    )
    return _apply_code_scale(model, config)


def _build_pairwise(config: RunConfig) -> nn.Module:
    tables, comparisons = _pairwise_shape(config)
    return PairwiseLinear(
        config.n_features,
        config.n_features,
        tables=tables,
        comparisons=comparisons,
        seed=config.seed,
    )


def _build_tied_pairwise(config: RunConfig) -> nn.Module:
    tables, comparisons = _pairwise_shape(config)
    return TiedPairwiseRecovery(
        config.n_features,
        config.model_dim,
        tables=tables,
        comparisons=comparisons,
        seed=config.seed,
    )


MODEL_BUILDERS: dict[str, ModelBuilder] = {
    "paper": _build_paper,
    "untied_paper": _build_untied_paper,
    "linear": _build_linear,
    "tropical_lowrank": _build_tropical_lowrank,
    "tied_tropical_lowrank": _build_tied_tropical_lowrank,
    "tropical_zero_dense": _build_tropical_zero_dense,
    "tied_tropical_zero_dense": _build_tied_tropical_zero_dense,
    "tropfan_zero_dense": _build_tropfan_zero_dense,
    "tied_tropfan_zero_dense": _build_tied_tropfan_zero_dense,
    "pairwise": _build_pairwise,
    "tied_pairwise": _build_tied_pairwise,
}


def _build_model(config: RunConfig, device: torch.device) -> nn.Module:
    try:
        return MODEL_BUILDERS[config.family](config).to(device)
    except KeyError as exc:
        raise ValueError(f"unknown family {config.family!r}") from exc


@torch.no_grad()
def _row_weight_growth_(weight: Tensor, lr: float, weight_decay: float, eps: float = 1e-8) -> None:
    if weight_decay >= 0:
        weight.mul_(1.0 - lr * weight_decay)
        return
    row_norms = weight.norm(dim=1, keepdim=True).add_(eps)
    weight.add_(weight_decay * weight * (1.0 - 1.0 / row_norms), alpha=lr)


@torch.no_grad()
def _paper_weight_growth_step(model: nn.Module, lr: float, weight_decay: float) -> None:
    if isinstance(model, PaperFeatureRecovery):
        _row_weight_growth_(model.weight, lr=lr, weight_decay=weight_decay)
    elif isinstance(model, UntiedPaperFeatureRecovery):
        _row_weight_growth_(model.encoder_weight, lr=lr, weight_decay=weight_decay)
        _row_weight_growth_(model.decoder_weight, lr=lr, weight_decay=weight_decay)


def _build_optimizer(config: RunConfig, model: nn.Module, lr: float) -> torch.optim.Optimizer:
    if config.family == "tropical_lowrank" and isinstance(model, TropLinear) and config.output_lr_mult != 1.0:
        output_params = list(model.output_proj.parameters())
        output_param_ids = {id(param) for param in output_params}
        base_params = [param for param in model.parameters() if id(param) not in output_param_ids]
        return torch.optim.AdamW(
            [{"params": base_params, "lr": lr}, {"params": output_params, "lr": lr * config.output_lr_mult}],
            weight_decay=0.0,
        )
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)


def _train_one(config: RunConfig) -> tuple[nn.Module, dict[str, float | int]]:
    device = torch.device(config.device)
    torch.manual_seed(config.seed)
    probs = feature_probabilities(config.n_features, config.alpha, config.activation_density, device=device)
    model = _build_model(config, device)
    lr = config.paper_lr if config.family in PAPER_FAMILIES else config.lr
    optimizer = _build_optimizer(config, model, lr)
    task_losses: list[float] = []

    t0 = time.perf_counter()
    model.train()
    for _ in range(config.steps):
        x = sample_batch(probs, config.batch_size)
        optimizer.zero_grad(set_to_none=True)
        y = _squeeze_single_sequence(model(x))
        loss = F.mse_loss(y, x)
        loss.backward()
        _paper_weight_growth_step(model, lr=lr, weight_decay=config.weight_decay)
        optimizer.step()
        task_losses.append(float(loss.detach().item()))
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)
    return model, {
        "final_loss": task_losses[-1],
        "best_loss": min(task_losses),
        "train_ms": (time.perf_counter() - t0) * 1000.0,
        "params": int(sum(param.numel() for param in model.parameters())),
    }


def _lowrank_effective_representations(model: TropLinear, n_features: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    eye = _eye(n_features, device)
    latent = model._project_input(eye.unsqueeze(1), torch.float32)
    score_backend = "torch" if model.backend == "tilelang" else model.backend
    scores = trop_scores(
        latent,
        model.router_weight.to(dtype=torch.float32, device=device),
        model.router_bias.to(dtype=torch.float32, device=device),
        backend=score_backend,
    )
    winner_idx, _, margins = _top2_indices(scores)
    codes = model._selected_codes(winner_idx, input_device=device, compute_dtype=torch.float32)
    model._last_indices = winner_idx.detach()
    model._last_margins = margins.detach()
    reps = (latent + codes.sum(dim=2) * model.code_scale).squeeze(1)
    return reps, codes.squeeze(1), scores.squeeze(1)


def _paper_representations(model: nn.Module, n_features: int, device: torch.device) -> Tensor:
    return model.encode(_eye(n_features, device))  # type: ignore[attr-defined]


def _untied_paper_representations(model: nn.Module, n_features: int, device: torch.device) -> Tensor:
    del n_features, device
    return model.readout_representations()  # type: ignore[attr-defined]


def _identity_model_representations(model: nn.Module, n_features: int, device: torch.device) -> Tensor:
    return _squeeze_single_sequence(model(_eye(n_features, device))).float()


def _lowrank_representations(model: nn.Module, n_features: int, device: torch.device) -> Tensor:
    assert isinstance(model, TropLinear)
    return _lowrank_effective_representations(model, n_features, device)[0]


def _tied_representations(model: nn.Module, n_features: int, device: torch.device) -> Tensor:
    del n_features, device
    assert isinstance(model, TiedRecoveryBase)
    return model.effective_representations(training=False)[0]


REPRESENTATION_EXTRACTORS: dict[str, RepresentationExtractor] = {
    "paper": _paper_representations,
    "untied_paper": _untied_paper_representations,
    "linear": _paper_representations,
    "tropical_lowrank": _lowrank_representations,
    "tied_tropical_lowrank": _tied_representations,
    "tropical_zero_dense": _identity_model_representations,
    "tied_tropical_zero_dense": _tied_representations,
    "tropfan_zero_dense": _identity_model_representations,
    "tied_tropfan_zero_dense": _tied_representations,
    "pairwise": _identity_model_representations,
    "tied_pairwise": _tied_representations,
}


def _effective_representations(model: nn.Module, family: str, n_features: int, device: torch.device) -> Tensor:
    try:
        return REPRESENTATION_EXTRACTORS[family](model, n_features, device)
    except KeyError as exc:
        raise ValueError(f"unknown family {family!r}") from exc


@torch.no_grad()
def _identity_response(model: nn.Module, n_features: int, device: torch.device, *, batch_size: int = 256) -> Tensor:
    responses: list[Tensor] = []
    model.eval()
    eye = _eye(n_features, device)
    for start in range(0, n_features, batch_size):
        stop = min(start + batch_size, n_features)
        responses.append(_squeeze_single_sequence(model(eye[start:stop])).detach().float().cpu())
    return torch.cat(responses, dim=0)


def _overlap_metrics(reps: Tensor) -> dict[str, float]:
    reps = reps.detach().float().cpu()
    norms = reps.norm(dim=1)
    represented = norms > max(1e-6, float(norms.median().item()) * 0.1)
    if int(represented.sum().item()) < 2:
        return {
            "represented_fraction": float(represented.float().mean().item()),
            "mean_squared_overlap": float("nan"),
            "overlap_variance": float("nan"),
        }
    unit = reps[represented] / norms[represented].unsqueeze(1).clamp_min(1e-12)
    gram2 = (unit @ unit.t()).square()
    off_diag = gram2[~torch.eye(gram2.shape[0], dtype=torch.bool)]
    return {
        "represented_fraction": float(represented.float().mean().item()),
        "mean_squared_overlap": float(off_diag.mean().item()),
        "overlap_variance": float(off_diag.var(unbiased=False).item()),
    }


def _frequency_weighted_overlap_metrics(reps: Tensor, probs: Tensor) -> dict[str, float]:
    reps = reps.detach().float().cpu()
    if reps.shape[0] < 2:
        return {"frequency_weighted_overlap": float("nan"), "frequency_pair_weighted_overlap": float("nan")}
    unit = F.normalize(reps, dim=1, eps=1e-12)
    gram2 = (unit @ unit.t()).square()
    mask = ~torch.eye(gram2.shape[0], dtype=torch.bool)
    gram2 = gram2.masked_fill(~mask, 0.0)
    probs_cpu = probs.detach().float().cpu()
    prob_sum = probs_cpu.sum().clamp_min(1e-12)
    feature_weighted = (gram2 * probs_cpu.view(-1, 1)).sum() / (prob_sum * (reps.shape[0] - 1))
    pair_weights = probs_cpu.view(-1, 1) * probs_cpu.view(1, -1)
    pair_weights = pair_weights.masked_fill(~mask, 0.0)
    pair_weighted = (gram2 * pair_weights).sum() / pair_weights.sum().clamp_min(1e-12)
    return {"frequency_weighted_overlap": float(feature_weighted.item()), "frequency_pair_weighted_overlap": float(pair_weighted.item())}


def _recovery_operator_metrics(response: Tensor, probs: Tensor) -> dict[str, float]:
    if response.ndim != 2 or response.shape[0] != response.shape[1]:
        raise ValueError(f"expected square identity response, got shape {tuple(response.shape)}")
    diag = response.diag()
    offdiag = response.clone()
    offdiag.fill_diagonal_(0.0)
    offdiag_energy = offdiag.square().sum(dim=1)
    probs_cpu = probs.detach().float().cpu()
    weights = probs_cpu / probs_cpu.sum().clamp_min(1e-12)
    self_error = diag - 1.0
    return {
        "self_gain_mean": float(diag.mean().item()),
        "self_gain_std": float(diag.std(unbiased=False).item()),
        "self_gain_weighted_mean": float((weights * diag).sum().item()),
        "self_gain_abs_error": float(self_error.abs().mean().item()),
        "self_gain_mse": float(self_error.square().mean().item()),
        "self_gain_weighted_mse": float((weights * self_error.square()).sum().item()),
        "offdiag_gain": float(offdiag.square().mean().item()),
        "offdiag_energy": float(offdiag_energy.mean().item()),
        "offdiag_weighted_energy": float((weights * offdiag_energy).sum().item()),
    }


@torch.no_grad()
def _refresh_route_cache(model: nn.Module, n_features: int, device: torch.device) -> None:
    if isinstance(model, TiedRecoveryBase):
        model.effective_representations(training=False)
    else:
        model(_eye(n_features, device))


@torch.no_grad()
def _route_metrics(model: nn.Module, family: str, n_features: int, device: torch.device) -> dict[str, float]:
    if family not in ROUTED_FAMILIES:
        return {field: float("nan") for field in ROUTE_METRIC_FIELDS}
    model.eval()
    _refresh_route_cache(model, n_features, device)
    indices = getattr(model, "_last_indices", None)
    margins = getattr(model, "_last_margins", None)
    assert indices is not None and margins is not None
    signatures = indices.squeeze(1).cpu()
    unique, counts = torch.unique(signatures, dim=0, return_counts=True)
    probs = counts.float() / counts.sum()
    entropy = -(probs * probs.log()).sum()
    return {
        "route_unique": float(unique.shape[0]),
        "route_collision_mean": float(counts.float().mean().item()),
        "route_entropy": float(entropy.item()),
        "avg_margin": float(margins.float().mean().item()),
    }


def _chamber_signatures(model: nn.Module, family: str) -> Tensor | None:
    if family not in ROUTED_FAMILIES:
        return None
    last = getattr(model, "_last_indices", None)
    if last is None:
        return None
    sigs = last.detach()
    if sigs.ndim == 3:
        sigs = sigs.squeeze(1)
    return sigs.cpu()


def _derived_row_metrics(row: BenchmarkRow, config: RunConfig) -> dict[str, float]:
    metrics = {
        "overlap_times_dim": float(row["mean_squared_overlap"]) * config.model_dim,
        "frequency_weighted_overlap_times_dim": float(row["frequency_weighted_overlap"]) * config.model_dim,
        "frequency_pair_weighted_overlap_times_dim": float(row["frequency_pair_weighted_overlap"]) * config.model_dim,
        "loss_per_activation": float(row["final_loss"]) / config.activation_density,
    }
    route_unique = float(row["route_unique"])
    route_entropy = float(row["route_entropy"])
    if math.isfinite(route_unique) and route_unique > 1 and math.isfinite(route_entropy):
        metrics["route_entropy_norm"] = route_entropy / math.log(route_unique)
    else:
        metrics["route_entropy_norm"] = float("nan")
    return metrics


def run_config(config: RunConfig) -> BenchmarkRow:
    from .bridge_scaling import bridge_diagnostics

    model, train_metrics = _train_one(config)
    device = torch.device(config.device)
    probs = feature_probabilities(config.n_features, config.alpha, config.activation_density, device=device)
    reps = _effective_representations(model, config.family, config.n_features, device)
    response = _identity_response(model, config.n_features, device)
    row: BenchmarkRow = asdict(config)
    row.update(train_metrics)
    row.update(_overlap_metrics(reps))
    row.update(_frequency_weighted_overlap_metrics(reps, probs))
    row.update(_recovery_operator_metrics(response, probs))
    row.update(_route_metrics(model, config.family, config.n_features, device))
    chamber_sigs = _chamber_signatures(model, config.family)
    row.update(
        bridge_diagnostics(
            reps_vector=reps.detach().float().cpu(),
            reps_chamber=chamber_sigs,
        )
    )
    row.update(_derived_row_metrics(row, config))
    return row


SummaryKey = tuple[str, float, int, int, int, str, float, str, int, int, int]


def _summary_key(row: BenchmarkRow) -> SummaryKey:
    family = str(row["family"])
    heads = int(row["heads"])
    if family in COORD_ZERO_DENSE_FAMILIES and heads == int(row["model_dim"]):
        heads = -1
    return (
        family,
        float(row["alpha"]),
        heads,
        int(row["cells"]),
        int(row["route_terms"]),
        str(row["code_scale_mode"]),
        float(row["output_lr_mult"]),
        str(row["fan_value_mode"]),
        int(row["fan_basis_rank"]),
        int(row.get("tables", 0)),
        int(row.get("comparisons", 0)),
    )


def _summary_fields(key: SummaryKey) -> dict[str, float | str]:
    family, alpha, heads, cells, route_terms, code_scale_mode, output_lr_mult, fan_value_mode, fan_basis_rank, tables, comparisons = key
    return {
        "family": family,
        "alpha": alpha,
        "heads": float(heads),
        "cells": float(cells),
        "route_terms": float(route_terms),
        "code_scale_mode": code_scale_mode,
        "output_lr_mult": float(output_lr_mult),
        "fan_value_mode": fan_value_mode,
        "fan_basis_rank": float(fan_basis_rank),
        "tables": float(tables),
        "comparisons": float(comparisons),
    }


def _rows_by_summary_key(rows: list[BenchmarkRow]) -> dict[SummaryKey, list[BenchmarkRow]]:
    groups: dict[SummaryKey, list[BenchmarkRow]] = {}
    for row in rows:
        groups.setdefault(_summary_key(row), []).append(row)
    return groups


def _fit_model_dim_exponent(group: list[BenchmarkRow]) -> tuple[float, float]:
    xs = torch.tensor([float(row["model_dim"]) for row in group])
    ys = torch.tensor([float(row["final_loss"]) for row in group])
    valid = torch.isfinite(xs) & torch.isfinite(ys) & (xs > 0) & (ys > 0)
    if int(valid.sum().item()) < 2:
        return float("nan"), float("nan")

    logx = torch.log(xs[valid])
    logy = torch.log(ys[valid])
    centered_x = logx - logx.mean()
    denom = centered_x.square().sum()
    if float(denom.item()) <= 0.0:
        return float("nan"), float("nan")

    slope = ((centered_x * (logy - logy.mean())).sum() / denom).item()
    pred = logy.mean() + slope * centered_x
    ss_res = (logy - pred).square().sum()
    ss_tot = (logy - logy.mean()).square().sum().clamp_min(1e-12)
    return -float(slope), float((1.0 - ss_res / ss_tot).item())


def _fit_exponents(rows: list[BenchmarkRow]) -> list[dict[str, float | str]]:
    summaries: list[dict[str, float | str]] = []
    for key, group in sorted(_rows_by_summary_key(rows).items()):
        beta, r2 = _fit_model_dim_exponent(group)
        summary = _summary_fields(key)
        summary.update({"beta": beta, "r2": r2, "points": float(len(group))})
        summaries.append(summary)
    return summaries


def _mean_finite(group: list[BenchmarkRow], key: str) -> float:
    values = torch.tensor([float(row[key]) for row in group])
    values = values[torch.isfinite(values)]
    return float(values.mean().item()) if values.numel() > 0 else float("nan")


def _group_metrics(rows: list[BenchmarkRow]) -> list[dict[str, float | str]]:
    summaries: list[dict[str, float | str]] = []
    for key, group in sorted(_rows_by_summary_key(rows).items()):
        summary = _summary_fields(key)
        summary.update({output_key: _mean_finite(group, source_key) for output_key, source_key in GROUP_METRIC_FIELDS})
        summary["points"] = float(len(group))
        summaries.append(summary)
    return summaries


def _parse_csv_numbers(value: str, cast: type = int) -> list:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def _validate_family(family: str) -> None:
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; expected one of {FAMILIES}")


def _heads_for_family(args: argparse.Namespace, family: str, model_dim: int) -> list[int]:
    if family in COORD_ZERO_DENSE_FAMILIES and args.zero_dense_heads_mode == "model_dim":
        return [model_dim]
    return [args.heads]


def _route_terms_for_family(args: argparse.Namespace, family: str, route_terms_values: list[int]) -> list[int]:
    if family in COORD_ZERO_DENSE_FAMILIES:
        return route_terms_values
    return [args.route_terms]


def _run_config_from_args(
    args: argparse.Namespace,
    *,
    family: str,
    alpha: float,
    model_dim: int,
    heads: int,
    route_terms: int,
    seed: int,
) -> RunConfig:
    return RunConfig(
        family=family,
        n_features=args.n_features,
        model_dim=model_dim,
        alpha=alpha,
        activation_density=args.activation_density,
        batch_size=args.batch_size,
        steps=args.steps,
        lr=args.lr,
        paper_lr=args.paper_lr,
        weight_decay=args.weight_decay,
        heads=heads,
        cells=args.cells,
        code_scale_mode=args.code_scale_mode,
        route_terms=route_terms,
        seed=seed,
        device=args.device,
        backend=args.backend,
        output_lr_mult=args.output_lr_mult if family == "tropical_lowrank" else 1.0,
        fan_value_mode=args.fan_value_mode if family in FAN_ZERO_DENSE_FAMILIES else "-",
        fan_basis_rank=args.fan_basis_rank if family in FAN_ZERO_DENSE_FAMILIES else 0,
        tables=args.tables if family in PAIRWISE_FAMILIES else 0,
        comparisons=args.comparisons if family in PAIRWISE_FAMILIES else 0,
    )


def _configs_from_args(args: argparse.Namespace) -> list[RunConfig]:
    families = tuple(_parse_csv_numbers(args.families, str))
    model_dims = _parse_csv_numbers(args.model_dims, int)
    alphas = _parse_csv_numbers(args.alphas, float)
    seeds = _parse_csv_numbers(args.seeds, int)
    route_terms_values = _parse_csv_numbers(args.route_terms_list, int) if args.route_terms_list else [args.route_terms]
    configs: list[RunConfig] = []
    for family in families:
        _validate_family(family)
        for alpha in alphas:
            for model_dim in model_dims:
                heads_values = _heads_for_family(args, family, model_dim)
                family_route_terms = _route_terms_for_family(args, family, route_terms_values)
                for heads in heads_values:
                    for route_terms in family_route_terms:
                        for seed in seeds:
                            configs.append(
                                _run_config_from_args(
                                    args,
                                    family=family,
                                    model_dim=model_dim,
                                    alpha=alpha,
                                    heads=heads,
                                    route_terms=route_terms,
                                    seed=seed,
                                )
                            )
    return configs


def _write_csv(rows: list[BenchmarkRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run sparse feature-recovery scaling benchmarks for tropnn layers.")
    parser.add_argument("--families", type=str, default="paper,untied_paper,linear,tied_tropical_lowrank,tied_tropical_zero_dense")
    parser.add_argument("--n-features", type=int, default=256)
    parser.add_argument("--model-dims", type=str, default="8,16,32,64")
    parser.add_argument("--alphas", type=str, default="0.0,0.5,1.0,1.5")
    parser.add_argument("--activation-density", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--paper-lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=-1.0)
    parser.add_argument("--heads", type=int, default=128)
    parser.add_argument("--cells", type=int, default=4)
    parser.add_argument("--code-scale-mode", choices=CODE_SCALE_MODES, default="sqrt")
    parser.add_argument("--route-terms", type=int, default=2)
    parser.add_argument("--route-terms-list", type=str, default="")
    parser.add_argument("--zero-dense-heads-mode", choices=("model_dim", "fixed"), default="model_dim")
    parser.add_argument("--fan-value-mode", choices=("site", "basis"), default="site")
    parser.add_argument("--fan-basis-rank", type=int, default=16)
    parser.add_argument("--tables", type=int, default=32, help="number of pairwise comparator tables")
    parser.add_argument("--comparisons", type=int, default=6, help="bits per pairwise table (cells = 2**comparisons)")
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--backend", choices=("torch", "auto", "triton", "tilelang"), default="torch")
    parser.add_argument("--output-lr-mult", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--quick", action="store_true")
    return parser


def _option_was_set(raw_argv: list[str], name: str) -> bool:
    return name in raw_argv or any(item.startswith(f"{name}=") for item in raw_argv)


def _apply_quick_defaults(args: argparse.Namespace, raw_argv: list[str]) -> None:
    if not args.quick:
        return
    args.n_features = 32
    args.model_dims = "4,8"
    args.alphas = "0.0"
    if not _option_was_set(raw_argv, "--families"):
        args.families = "paper,untied_paper,linear,tied_tropical_lowrank,tied_tropical_zero_dense,tied_tropfan_zero_dense"
    args.batch_size = 16
    args.steps = 3
    args.heads = 8
    args.route_terms_list = "2"
    args.fan_value_mode = "site"
    args.seeds = "0"
    args.device = "cpu"
    args.backend = "torch"


def _print_run_start(idx: int, total: int, config: RunConfig) -> None:
    print(
        f"[{idx}/{total}] family={config.family} n={config.n_features} m={config.model_dim} "
        f"alpha={config.alpha} heads={config.heads} cells={config.cells} route_terms={config.route_terms} seed={config.seed}"
    )


def _print_run_result(row: BenchmarkRow) -> None:
    print(
        f"  loss={float(row['final_loss']):.6g} overlap*m={float(row['overlap_times_dim']):.6g} self_mse={float(row['self_gain_weighted_mse']):.6g}"
    )


def _run_configs(configs: list[RunConfig]) -> list[BenchmarkRow]:
    rows: list[BenchmarkRow] = []
    for idx, config in enumerate(configs, start=1):
        _print_run_start(idx, len(configs), config)
        row = run_config(config)
        rows.append(row)
        _print_run_result(row)
    return rows


def _output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    repo_root = Path(__file__).resolve().parents[4]
    output_dir = args.output_dir if args.output_dir is not None else repo_root / "results" / "scaling_benchmark"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = f"{args.tag}-{stamp}" if args.tag else stamp
    return output_dir / f"runs-{name}.csv", output_dir / f"summary-{name}.json"


def _build_summary(rows: list[BenchmarkRow]) -> dict[str, object]:
    from .bridge_scaling import fit_bridge_exponents_per_family

    return {
        "runs": rows,
        "exponents": _fit_exponents(rows),
        "group_metrics": _group_metrics(rows),
        "bridge_exponents": fit_bridge_exponents_per_family(rows, capacity_keys=BRIDGE_CAPACITY_KEYS),
    }


def _write_outputs(rows: list[BenchmarkRow], csv_path: Path, json_path: Path) -> None:
    _write_csv(rows, csv_path)
    summary = _build_summary(rows)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"runs_csv={csv_path}")
    print(f"summary_json={json_path}")


def main(argv: Iterable[str] | None = None) -> None:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    parser = _build_parser()
    args = parser.parse_args(raw_argv)
    _apply_quick_defaults(args, raw_argv)

    rows = _run_configs(_configs_from_args(args))
    csv_path, json_path = _output_paths(args)
    _write_outputs(rows, csv_path, json_path)


if __name__ == "__main__":
    main()
