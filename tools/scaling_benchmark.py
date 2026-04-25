from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..backend import Backend, trop_scores
from ..layers import PairwiseLinear, TropLinear
from ..layers.tropical import _minface_mix, _top2_indices

FAMILIES = ("paper", "linear", "tropical", "tied_tropical", "pairwise")
CODE_SCALE_MODES = ("sqrt", "linear", "none")
CODE_GEOMETRY_LOSSES = ("none", "welch")
TROPICAL_FAMILIES = ("tropical", "tied_tropical")


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
    pairwise_tables: int
    comparisons: int
    seed: int
    device: str
    backend: str
    code_geometry_loss: str = "none"
    code_geometry_weight: float = 0.0
    code_norm_weight: float = 0.0
    route_balance_weight: float = 0.0
    recovery_diag_weight: float = 0.0
    recovery_offdiag_weight: float = 0.0
    output_lr_mult: float = 1.0


class PaperFeatureRecovery(nn.Module):
    def __init__(self, n_features: int, model_dim: int, *, seed: int) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.n_features = n_features
        self.model_dim = model_dim
        self.weight = nn.Parameter(torch.randn(n_features, model_dim) / math.sqrt(model_dim))
        self.bias = nn.Parameter(torch.zeros(n_features))

    def encode(self, x: Tensor) -> Tensor:
        return x @ self.weight

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(self.encode(x) @ self.weight.t() + self.bias)


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


class TiedTropicalFeatureRecovery(nn.Module):
    """Paper-style tied autoencoder whose feature vectors are tropical codes."""

    def __init__(
        self,
        n_features: int,
        model_dim: int,
        *,
        heads: int,
        cells: int,
        backend: Backend,
        seed: int,
        code_init_std: float = 0.02,
    ) -> None:
        super().__init__()
        if heads < 1:
            raise ValueError(f"heads must be >= 1, got {heads}")
        if cells < 2:
            raise ValueError(f"cells must be >= 2, got {cells}")
        if model_dim < 1:
            raise ValueError(f"model_dim must be >= 1, got {model_dim}")

        torch.manual_seed(seed)
        self.n_features = n_features
        self.model_dim = model_dim
        self.heads = heads
        self.cells = cells
        self.code_dim = model_dim
        self.backend = backend
        self.code_scale = 1.0 / math.sqrt(heads)

        self.proj = nn.Linear(n_features, model_dim, bias=False)
        nn.init.normal_(self.proj.weight, mean=0.0, std=1.0 / math.sqrt(max(1, n_features)))
        router_std = 1.0 / math.sqrt(max(1, model_dim))
        self.router_weight = nn.Parameter(torch.randn(heads, cells, model_dim) * router_std)
        self.router_bias = nn.Parameter(torch.zeros(heads, cells))
        self.code = nn.Parameter(torch.randn(heads, cells, model_dim) * code_init_std)
        self.bias = nn.Parameter(torch.zeros(n_features))
        self._last_indices: Tensor | None = None
        self._last_margins: Tensor | None = None

    def _project_input(self, x: Tensor, compute_dtype: torch.dtype) -> Tensor:
        return self.proj(x).to(compute_dtype)

    def _scores(self, latent: Tensor, *, input_device: torch.device, compute_dtype: torch.dtype) -> Tensor:
        score_backend = "torch" if self.backend == "tilelang" else self.backend
        return trop_scores(
            latent,
            self.router_weight.to(dtype=compute_dtype, device=input_device),
            self.router_bias.to(dtype=compute_dtype, device=input_device),
            backend=score_backend,
        )

    def _selected_codes(self, winner_idx: Tensor, *, input_device: torch.device, compute_dtype: torch.dtype) -> Tensor:
        code = self.code.to(dtype=compute_dtype, device=input_device).unsqueeze(0).unsqueeze(0)
        gather_idx = winner_idx.unsqueeze(-1).unsqueeze(-1).expand(*winner_idx.shape, 1, self.code_dim)
        return code.expand(*winner_idx.shape[:2], -1, -1, -1).gather(-2, gather_idx).squeeze(-2)

    def effective_representations(self, *, compute_dtype: torch.dtype = torch.float32, training: bool | None = None) -> tuple[Tensor, Tensor, Tensor]:
        use_training_routes = self.training if training is None else training
        device = self.proj.weight.device
        x = torch.eye(self.n_features, device=device)
        latent = self._project_input(x.unsqueeze(1), compute_dtype)
        scores = self._scores(latent, input_device=device, compute_dtype=compute_dtype)
        winner_idx, runner_idx, margins = _top2_indices(scores)
        if use_training_routes:
            winner_codes = self._selected_codes(winner_idx, input_device=device, compute_dtype=compute_dtype)
            runner_codes = self._selected_codes(runner_idx, input_device=device, compute_dtype=compute_dtype)
            codes = _minface_mix(winner_codes, runner_codes, margins)
        else:
            codes = self._selected_codes(winner_idx, input_device=device, compute_dtype=compute_dtype)
        self._last_indices = winner_idx.detach()
        self._last_margins = margins.detach()
        reps = (latent + codes.sum(dim=2) * self.code_scale).squeeze(1)
        return reps, codes.squeeze(1), scores.squeeze(1)

    def encode(self, x: Tensor) -> Tensor:
        reps, _, _ = self.effective_representations(compute_dtype=torch.float32)
        return x.to(reps.dtype) @ reps

    def forward(self, x: Tensor) -> Tensor:
        squeeze_seq = False
        if x.ndim == 3 and x.shape[1] == 1:
            x = x.squeeze(1)
            squeeze_seq = True
        reps, _, _ = self.effective_representations(compute_dtype=torch.float32)
        hidden = x.to(reps.dtype) @ reps
        output = F.relu(hidden @ reps.t() + self.bias.to(dtype=reps.dtype, device=reps.device))
        return output.unsqueeze(1) if squeeze_seq else output


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


def _build_model(config: RunConfig, device: torch.device) -> nn.Module:
    if config.family == "paper":
        return PaperFeatureRecovery(config.n_features, config.model_dim, seed=config.seed).to(device)
    if config.family == "linear":
        return LinearRecovery(config.n_features, config.model_dim, seed=config.seed).to(device)
    if config.family == "tropical":
        layer = TropLinear(
            config.n_features,
            config.n_features,
            heads=config.heads,
            cells=config.cells,
            code_dim=config.model_dim,
            backend=config.backend,
            seed=config.seed,
        )
        layer.code_scale = _code_scale(config.heads, config.code_scale_mode)
        return layer.to(device)
    if config.family == "tied_tropical":
        model = TiedTropicalFeatureRecovery(
            config.n_features,
            config.model_dim,
            heads=config.heads,
            cells=config.cells,
            backend=config.backend,
            seed=config.seed,
        )
        model.code_scale = _code_scale(config.heads, config.code_scale_mode)
        return model.to(device)
    if config.family == "pairwise":
        tables = config.pairwise_tables if config.pairwise_tables > 0 else config.model_dim
        return PairwiseLinear(
            config.n_features,
            config.n_features,
            tables=tables,
            comparisons=config.comparisons,
            seed=config.seed,
        ).to(device)
    raise ValueError(f"unknown family {config.family!r}")


@torch.no_grad()
def _paper_weight_growth_step(model: nn.Module, lr: float, weight_decay: float, eps: float = 1e-8) -> None:
    if not isinstance(model, PaperFeatureRecovery):
        return
    if weight_decay >= 0:
        model.weight.mul_(1.0 - lr * weight_decay)
        return
    row_norms = model.weight.norm(dim=1, keepdim=True).add_(eps)
    model.weight.add_(weight_decay * model.weight * (1.0 - 1.0 / row_norms), alpha=lr)


def _zero_loss(device: torch.device) -> Tensor:
    return torch.zeros((), device=device)


def _tropical_effective_representations(
    model: TropLinear,
    n_features: int,
    device: torch.device,
    *,
    training: bool | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    use_training_routes = model.training if training is None else training
    x = torch.eye(n_features, device=device)
    latent = model._project_input(x.unsqueeze(1), torch.float32)
    scores = model._scores(latent, input_device=device, compute_dtype=torch.float32)
    winner_idx, runner_idx, margins = _top2_indices(scores)
    if use_training_routes:
        winner_codes = model._selected_codes(winner_idx, input_device=device, compute_dtype=torch.float32)
        runner_codes = model._selected_codes(runner_idx, input_device=device, compute_dtype=torch.float32)
        codes = _minface_mix(winner_codes, runner_codes, margins)
    else:
        codes = model._selected_codes(winner_idx, input_device=device, compute_dtype=torch.float32)
    reps = (latent + codes.sum(dim=2) * model.code_scale).squeeze(1)
    return reps, codes.squeeze(1), scores.squeeze(1)


def _effective_representations(
    model: nn.Module,
    n_features: int,
    device: torch.device,
    *,
    training: bool | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    if isinstance(model, TropLinear):
        return _tropical_effective_representations(model, n_features, device, training=training)
    if isinstance(model, TiedTropicalFeatureRecovery):
        return model.effective_representations(training=training)
    raise TypeError(f"expected tropical model, got {type(model).__name__}")


def _welch_loss(reps: Tensor) -> Tensor:
    if reps.shape[0] < 2:
        return _zero_loss(reps.device)
    unit = F.normalize(reps.float(), dim=1, eps=1e-12)
    gram2 = (unit @ unit.t()).square()
    off_diag = gram2[~torch.eye(gram2.shape[0], dtype=torch.bool, device=gram2.device)]
    target = 1.0 / max(1, reps.shape[1])
    return (off_diag - target).square().mean()


def _route_balance_loss(scores: Tensor, *, target_entropy_norm: float = 0.5) -> Tensor:
    if scores.shape[-1] < 2:
        return _zero_loss(scores.device)
    probs = torch.softmax(scores.float(), dim=-1).mean(dim=0)
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
    entropy_norm = entropy / math.log(scores.shape[-1])
    deficit = (target_entropy_norm - entropy_norm).clamp_min(0.0)
    return deficit.square().mean()


def _tropical_regularizers(config: RunConfig, model: nn.Module, device: torch.device) -> tuple[Tensor, dict[str, float]]:
    zero = _zero_loss(device)
    if config.family not in TROPICAL_FAMILIES or not isinstance(model, (TropLinear, TiedTropicalFeatureRecovery)):
        return zero, {
            "geometry_loss": 0.0,
            "code_norm_loss": 0.0,
            "route_balance_loss": 0.0,
            "code_norm_mean": float("nan"),
            "code_norm_std": float("nan"),
            "selected_code_norm_mean": float("nan"),
            "selected_code_norm_std": float("nan"),
        }

    needs_reps = (
        config.code_geometry_weight > 0.0
        or config.code_norm_weight > 0.0
        or config.route_balance_weight > 0.0
    )
    if not needs_reps:
        code_norms = model.code.float().norm(dim=-1)
        return zero, {
            "geometry_loss": 0.0,
            "code_norm_loss": 0.0,
            "route_balance_loss": 0.0,
            "code_norm_mean": float(code_norms.mean().detach().item()),
            "code_norm_std": float(code_norms.std(unbiased=False).detach().item()),
            "selected_code_norm_mean": float("nan"),
            "selected_code_norm_std": float("nan"),
        }

    reps, selected_codes, scores = _effective_representations(model, config.n_features, device, training=True)
    if config.code_geometry_loss == "none":
        geometry_loss = zero
    elif config.code_geometry_loss == "welch":
        geometry_loss = _welch_loss(reps)
    else:
        raise ValueError(f"unknown code_geometry_loss {config.code_geometry_loss!r}; expected one of {CODE_GEOMETRY_LOSSES}")

    selected_norms = selected_codes.float().norm(dim=-1)
    code_norm_loss = (selected_norms.mean(dim=1) - 1.0).square().mean()
    route_loss = _route_balance_loss(scores)
    total = (
        config.code_geometry_weight * geometry_loss
        + config.code_norm_weight * code_norm_loss
        + config.route_balance_weight * route_loss
    )
    code_norms = model.code.float().norm(dim=-1)
    return total, {
        "geometry_loss": float(geometry_loss.detach().item()),
        "code_norm_loss": float(code_norm_loss.detach().item()),
        "route_balance_loss": float(route_loss.detach().item()),
        "code_norm_mean": float(code_norms.mean().detach().item()),
        "code_norm_std": float(code_norms.std(unbiased=False).detach().item()),
        "selected_code_norm_mean": float(selected_norms.mean().detach().item()),
        "selected_code_norm_std": float(selected_norms.std(unbiased=False).detach().item()),
    }


def _squeeze_single_sequence(y: Tensor) -> Tensor:
    return y.squeeze(1) if y.ndim == 3 and y.shape[1] == 1 else y


def _recovery_losses(response: Tensor, probs: Tensor) -> tuple[Tensor, Tensor]:
    response = _squeeze_single_sequence(response)
    if response.ndim != 2 or response.shape[0] != response.shape[1]:
        raise ValueError(f"expected square identity response, got shape {tuple(response.shape)}")
    weights = probs.to(dtype=response.dtype, device=response.device)
    weights = weights / weights.sum().clamp_min(1e-12)
    diag = response.diag()
    diag_loss = (weights * (diag - 1.0).square()).sum()
    offdiag = response.square()
    offdiag = offdiag.masked_fill(torch.eye(response.shape[0], dtype=torch.bool, device=response.device), 0.0)
    offdiag_loss = (weights * offdiag.sum(dim=1)).sum()
    return diag_loss, offdiag_loss


def _recovery_regularizers(config: RunConfig, model: nn.Module, probs: Tensor) -> tuple[Tensor, dict[str, float]]:
    zero = _zero_loss(probs.device)
    empty_metrics = {
        "recovery_diag_loss": 0.0,
        "recovery_offdiag_loss": 0.0,
        "recovery_loss": 0.0,
    }
    if config.family not in TROPICAL_FAMILIES:
        return zero, empty_metrics
    if config.recovery_diag_weight == 0.0 and config.recovery_offdiag_weight == 0.0:
        return zero, empty_metrics

    response = model(torch.eye(config.n_features, device=probs.device))
    diag_loss, offdiag_loss = _recovery_losses(response, probs)
    total = config.recovery_diag_weight * diag_loss + config.recovery_offdiag_weight * offdiag_loss
    return total, {
        "recovery_diag_loss": float(diag_loss.detach().item()),
        "recovery_offdiag_loss": float(offdiag_loss.detach().item()),
        "recovery_loss": float(total.detach().item()),
    }


def _build_optimizer(config: RunConfig, model: nn.Module, lr: float) -> torch.optim.Optimizer:
    if config.family == "tropical" and isinstance(model, TropLinear) and config.output_lr_mult != 1.0:
        output_params = list(model.output_proj.parameters())
        output_param_ids = {id(param) for param in output_params}
        base_params = [param for param in model.parameters() if id(param) not in output_param_ids]
        return torch.optim.AdamW(
            [
                {"params": base_params, "lr": lr},
                {"params": output_params, "lr": lr * config.output_lr_mult},
            ],
            weight_decay=0.0,
        )
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)


def _train_one(config: RunConfig) -> tuple[nn.Module, dict[str, float | int | str]]:
    device = torch.device(config.device)
    torch.manual_seed(config.seed)
    probs = feature_probabilities(config.n_features, config.alpha, config.activation_density, device=device)
    model = _build_model(config, device)
    lr = config.paper_lr if config.family == "paper" else config.lr
    optimizer = _build_optimizer(config, model, lr)

    task_losses: list[float] = []
    total_losses: list[float] = []
    regularizer_metrics: dict[str, float] = {}
    recovery_metrics: dict[str, float] = {}
    t0 = time.perf_counter()
    model.train()
    for _ in range(config.steps):
        x = sample_batch(probs, config.batch_size)
        optimizer.zero_grad(set_to_none=True)
        y = model(x)
        y = _squeeze_single_sequence(y)
        task_loss = F.mse_loss(y, x)
        regularizer_loss, regularizer_metrics = _tropical_regularizers(config, model, device)
        recovery_loss, recovery_metrics = _recovery_regularizers(config, model, probs)
        loss = task_loss + regularizer_loss + recovery_loss
        loss.backward()
        _paper_weight_growth_step(model, lr=lr, weight_decay=config.weight_decay)
        optimizer.step()
        task_losses.append(float(task_loss.detach().item()))
        total_losses.append(float(loss.detach().item()))
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)
    train_ms = (time.perf_counter() - t0) * 1000.0

    return model, {
        "final_loss": task_losses[-1],
        "best_loss": min(task_losses),
        "task_loss": task_losses[-1],
        "total_loss": total_losses[-1],
        "train_ms": train_ms,
        "params": int(sum(param.numel() for param in model.parameters())),
        **regularizer_metrics,
        **recovery_metrics,
    }


@torch.no_grad()
def _representations(model: nn.Module, family: str, n_features: int, device: torch.device, *, batch_size: int = 256) -> Tensor:
    reps: list[Tensor] = []
    model.eval()
    for start in range(0, n_features, batch_size):
        stop = min(start + batch_size, n_features)
        x = torch.eye(n_features, device=device)[start:stop]
        if family == "paper":
            rep = model.encode(x)  # type: ignore[attr-defined]
        elif family == "linear":
            rep = model.encode(x)  # type: ignore[attr-defined]
        elif family in TROPICAL_FAMILIES:
            rep = _effective_representations(model, n_features, device, training=False)[0][start:stop]
        elif family == "pairwise":
            rep = model(x).squeeze(1)
        else:
            raise ValueError(f"unknown family {family!r}")
        reps.append(rep.detach().float().cpu())
    return torch.cat(reps, dim=0)


def _overlap_metrics(reps: Tensor) -> dict[str, float]:
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
    if reps.shape[0] < 2:
        return {
            "frequency_weighted_overlap": float("nan"),
            "frequency_pair_weighted_overlap": float("nan"),
        }

    unit = F.normalize(reps.float(), dim=1, eps=1e-12)
    gram2 = (unit @ unit.t()).square()
    mask = ~torch.eye(gram2.shape[0], dtype=torch.bool)
    gram2 = gram2.masked_fill(~mask, 0.0)

    probs = probs.detach().float().cpu()
    prob_sum = probs.sum().clamp_min(1e-12)
    feature_weighted = (gram2 * probs.view(-1, 1)).sum() / (prob_sum * (reps.shape[0] - 1))

    pair_weights = probs.view(-1, 1) * probs.view(1, -1)
    pair_weights = pair_weights.masked_fill(~mask, 0.0)
    pair_weighted = (gram2 * pair_weights).sum() / pair_weights.sum().clamp_min(1e-12)

    return {
        "frequency_weighted_overlap": float(feature_weighted.item()),
        "frequency_pair_weighted_overlap": float(pair_weighted.item()),
    }


@torch.no_grad()
def _identity_response(model: nn.Module, n_features: int, device: torch.device, *, batch_size: int = 256) -> Tensor:
    responses: list[Tensor] = []
    model.eval()
    eye = torch.eye(n_features, device=device)
    for start in range(0, n_features, batch_size):
        stop = min(start + batch_size, n_features)
        y = model(eye[start:stop])
        responses.append(_squeeze_single_sequence(y).detach().float().cpu())
    return torch.cat(responses, dim=0)


def _recovery_operator_metrics(response: Tensor, probs: Tensor) -> dict[str, float]:
    if response.ndim != 2 or response.shape[0] != response.shape[1]:
        raise ValueError(f"expected square identity response, got shape {tuple(response.shape)}")

    diag = response.diag()
    offdiag = response.clone()
    offdiag.fill_diagonal_(0.0)
    offdiag_energy = offdiag.square().sum(dim=1)

    probs = probs.detach().float().cpu()
    weights = probs / probs.sum().clamp_min(1e-12)
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
def _route_metrics(model: nn.Module, family: str, n_features: int, device: torch.device) -> dict[str, float]:
    if family in TROPICAL_FAMILIES:
        assert isinstance(model, (TropLinear, TiedTropicalFeatureRecovery))
        x = torch.eye(n_features, device=device)
        model.eval()
        model(x)
        indices = model._last_indices
        margins = model._last_margins
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
    if family == "pairwise":
        x = torch.eye(n_features, device=device)
        model.eval()
        model(x)
        indices = model._last_indices
        margins = model._last_margins
        assert indices is not None and margins is not None
        table_entropies = []
        unique_counts = []
        for table in range(indices.shape[-1]):
            _, counts = torch.unique(indices[:, 0, table].cpu(), return_counts=True)
            probs = counts.float() / counts.sum()
            table_entropies.append(float((-(probs * probs.log()).sum()).item()))
            unique_counts.append(float(counts.numel()))
        return {
            "route_unique": sum(unique_counts),
            "route_collision_mean": float(n_features / max(1.0, sum(unique_counts) / len(unique_counts))),
            "route_entropy": float(sum(table_entropies) / len(table_entropies)),
            "avg_margin": float(margins.float().mean().item()),
        }
    return {"route_unique": float("nan"), "route_collision_mean": float("nan"), "route_entropy": float("nan"), "avg_margin": float("nan")}


def run_config(config: RunConfig) -> dict[str, float | int | str]:
    model, train_metrics = _train_one(config)
    device = torch.device(config.device)
    probs = feature_probabilities(config.n_features, config.alpha, config.activation_density, device=device)
    reps = _representations(model, config.family, config.n_features, device)
    response = _identity_response(model, config.n_features, device)
    row: dict[str, float | int | str] = asdict(config)
    row["effective_pairwise_tables"] = config.pairwise_tables if config.pairwise_tables > 0 else config.model_dim
    row.update(train_metrics)
    row.update(_overlap_metrics(reps))
    row.update(_frequency_weighted_overlap_metrics(reps, probs))
    row.update(_recovery_operator_metrics(response, probs))
    row.update(_route_metrics(model, config.family, config.n_features, device))
    row["overlap_times_dim"] = float(row["mean_squared_overlap"]) * config.model_dim
    row["frequency_weighted_overlap_times_dim"] = float(row["frequency_weighted_overlap"]) * config.model_dim
    row["frequency_pair_weighted_overlap_times_dim"] = float(row["frequency_pair_weighted_overlap"]) * config.model_dim
    row["loss_per_activation"] = float(row["final_loss"]) / config.activation_density
    route_unique = float(row["route_unique"])
    route_entropy = float(row["route_entropy"])
    if math.isfinite(route_unique) and route_unique > 1 and math.isfinite(route_entropy):
        row["route_entropy_norm"] = route_entropy / math.log(route_unique)
    else:
        row["route_entropy_norm"] = float("nan")
    return row


def _fit_exponents(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | str]]:
    groups: dict[tuple, list[dict[str, float | int | str]]] = {}
    for row in rows:
        groups.setdefault(_summary_key(row), []).append(row)
    summaries: list[dict[str, float | str]] = []
    for key, group in sorted(groups.items()):
        (
            family,
            alpha,
            heads,
            cells,
            code_scale_mode,
            pairwise_tables,
            comparisons,
            code_geometry_loss,
            code_geometry_weight,
            code_norm_weight,
            route_balance_weight,
            recovery_diag_weight,
            recovery_offdiag_weight,
            output_lr_mult,
        ) = key
        xs = torch.tensor([float(row["model_dim"]) for row in group])
        ys = torch.tensor([float(row["final_loss"]) for row in group])
        valid = torch.isfinite(xs) & torch.isfinite(ys) & (xs > 0) & (ys > 0)
        if int(valid.sum().item()) < 2:
            beta = float("nan")
            r2 = float("nan")
        else:
            logx = torch.log(xs[valid])
            logy = torch.log(ys[valid])
            centered_x = logx - logx.mean()
            slope = ((centered_x * (logy - logy.mean())).sum() / centered_x.square().sum()).item()
            pred = logy.mean() + slope * centered_x
            ss_res = (logy - pred).square().sum()
            ss_tot = (logy - logy.mean()).square().sum().clamp_min(1e-12)
            beta = -float(slope)
            r2 = float((1.0 - ss_res / ss_tot).item())
        summaries.append(
            {
                "family": family,
                "alpha": alpha,
                "heads": float(heads),
                "cells": float(cells),
                "code_scale_mode": code_scale_mode,
                "pairwise_tables": float(pairwise_tables),
                "comparisons": float(comparisons),
                "code_geometry_loss": code_geometry_loss,
                "code_geometry_weight": float(code_geometry_weight),
                "code_norm_weight": float(code_norm_weight),
                "route_balance_weight": float(route_balance_weight),
                "recovery_diag_weight": float(recovery_diag_weight),
                "recovery_offdiag_weight": float(recovery_offdiag_weight),
                "output_lr_mult": float(output_lr_mult),
                "beta": beta,
                "r2": r2,
                "points": float(len(group)),
            }
        )
    return summaries


def _summary_key(row: dict[str, float | int | str]) -> tuple[str, float, int, int, str, int, int, str, float, float, float, float, float, float]:
    return (
        str(row["family"]),
        float(row["alpha"]),
        int(row["heads"]),
        int(row["cells"]),
        str(row["code_scale_mode"]),
        int(row["pairwise_tables"]),
        int(row["comparisons"]),
        str(row["code_geometry_loss"]),
        float(row["code_geometry_weight"]),
        float(row["code_norm_weight"]),
        float(row["route_balance_weight"]),
        float(row["recovery_diag_weight"]),
        float(row["recovery_offdiag_weight"]),
        float(row["output_lr_mult"]),
    )


def _mean_finite(group: list[dict[str, float | int | str]], key: str) -> float:
    values = torch.tensor([float(row[key]) for row in group])
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return float("nan")
    return float(values.mean().item())


def _group_metrics(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | str]]:
    groups: dict[tuple, list[dict[str, float | int | str]]] = {}
    for row in rows:
        groups.setdefault(_summary_key(row), []).append(row)
    summaries: list[dict[str, float | str]] = []
    for key, group in sorted(groups.items()):
        (
            family,
            alpha,
            heads,
            cells,
            code_scale_mode,
            pairwise_tables,
            comparisons,
            code_geometry_loss,
            code_geometry_weight,
            code_norm_weight,
            route_balance_weight,
            recovery_diag_weight,
            recovery_offdiag_weight,
            output_lr_mult,
        ) = key
        summaries.append(
            {
                "family": family,
                "alpha": alpha,
                "heads": float(heads),
                "cells": float(cells),
                "code_scale_mode": code_scale_mode,
                "pairwise_tables": float(pairwise_tables),
                "comparisons": float(comparisons),
                "code_geometry_loss": code_geometry_loss,
                "code_geometry_weight": float(code_geometry_weight),
                "code_norm_weight": float(code_norm_weight),
                "route_balance_weight": float(route_balance_weight),
                "recovery_diag_weight": float(recovery_diag_weight),
                "recovery_offdiag_weight": float(recovery_offdiag_weight),
                "output_lr_mult": float(output_lr_mult),
                "mean_overlap_times_dim": _mean_finite(group, "overlap_times_dim"),
                "mean_frequency_weighted_overlap_times_dim": _mean_finite(group, "frequency_weighted_overlap_times_dim"),
                "mean_frequency_pair_weighted_overlap_times_dim": _mean_finite(group, "frequency_pair_weighted_overlap_times_dim"),
                "mean_route_entropy_norm": _mean_finite(group, "route_entropy_norm"),
                "mean_represented_fraction": _mean_finite(group, "represented_fraction"),
                "mean_loss_per_activation": _mean_finite(group, "loss_per_activation"),
                "mean_self_gain": _mean_finite(group, "self_gain_mean"),
                "mean_self_gain_weighted": _mean_finite(group, "self_gain_weighted_mean"),
                "mean_self_gain_mse": _mean_finite(group, "self_gain_mse"),
                "mean_self_gain_weighted_mse": _mean_finite(group, "self_gain_weighted_mse"),
                "mean_offdiag_gain": _mean_finite(group, "offdiag_gain"),
                "mean_offdiag_energy": _mean_finite(group, "offdiag_energy"),
                "mean_offdiag_weighted_energy": _mean_finite(group, "offdiag_weighted_energy"),
                "mean_geometry_loss": _mean_finite(group, "geometry_loss"),
                "mean_code_norm_loss": _mean_finite(group, "code_norm_loss"),
                "mean_route_balance_loss": _mean_finite(group, "route_balance_loss"),
                "mean_recovery_diag_loss": _mean_finite(group, "recovery_diag_loss"),
                "mean_recovery_offdiag_loss": _mean_finite(group, "recovery_offdiag_loss"),
                "mean_recovery_loss": _mean_finite(group, "recovery_loss"),
                "points": float(len(group)),
            }
        )
    return summaries


def _parse_csv_numbers(value: str, cast: type = int) -> list:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def _configs_from_args(args: argparse.Namespace) -> list[RunConfig]:
    families = tuple(_parse_csv_numbers(args.families, str))
    model_dims = _parse_csv_numbers(args.model_dims, int)
    alphas = _parse_csv_numbers(args.alphas, float)
    seeds = _parse_csv_numbers(args.seeds, int)
    heads_values = _parse_csv_numbers(args.heads_list, int) if args.heads_list else [args.heads]
    cells_values = _parse_csv_numbers(args.cells_list, int) if args.cells_list else [args.cells]
    code_scale_modes = _parse_csv_numbers(args.code_scale_modes, str) if args.code_scale_modes else [args.code_scale_mode]
    if args.code_geometry_loss not in CODE_GEOMETRY_LOSSES:
        raise ValueError(f"unknown code_geometry_loss {args.code_geometry_loss!r}; expected one of {CODE_GEOMETRY_LOSSES}")
    if args.output_lr_mult <= 0.0:
        raise ValueError(f"output_lr_mult must be positive, got {args.output_lr_mult}")
    if args.recovery_diag_weight < 0.0:
        raise ValueError(f"recovery_diag_weight must be non-negative, got {args.recovery_diag_weight}")
    if args.recovery_offdiag_weight < 0.0:
        raise ValueError(f"recovery_offdiag_weight must be non-negative, got {args.recovery_offdiag_weight}")
    configs = []
    for family in families:
        if family not in FAMILIES:
            raise ValueError(f"unknown family {family!r}; expected one of {FAMILIES}")
        is_tropical_family = family in TROPICAL_FAMILIES
        family_heads_values = heads_values if is_tropical_family else [args.heads]
        family_cells_values = cells_values if is_tropical_family else [args.cells]
        family_code_scale_modes = code_scale_modes if is_tropical_family else [args.code_scale_mode]
        for alpha in alphas:
            for model_dim in model_dims:
                for heads in family_heads_values:
                    for cells in family_cells_values:
                        for code_scale_mode in family_code_scale_modes:
                            if code_scale_mode not in CODE_SCALE_MODES:
                                raise ValueError(f"unknown code_scale_mode {code_scale_mode!r}; expected one of {CODE_SCALE_MODES}")
                            for seed in seeds:
                                configs.append(
                                    RunConfig(
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
                                        cells=cells,
                                        code_scale_mode=code_scale_mode,
                                        pairwise_tables=args.pairwise_tables,
                                        comparisons=args.comparisons,
                                        seed=seed,
                                        device=args.device,
                                        backend=args.backend,
                                        code_geometry_loss=args.code_geometry_loss if is_tropical_family else "none",
                                        code_geometry_weight=args.code_geometry_weight if is_tropical_family else 0.0,
                                        code_norm_weight=args.code_norm_weight if is_tropical_family else 0.0,
                                        route_balance_weight=args.route_balance_weight if is_tropical_family else 0.0,
                                        recovery_diag_weight=args.recovery_diag_weight if is_tropical_family else 0.0,
                                        recovery_offdiag_weight=args.recovery_offdiag_weight if is_tropical_family else 0.0,
                                        output_lr_mult=args.output_lr_mult if family == "tropical" else 1.0,
                                    )
                                )
    return configs


def _write_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _apply_presets(args: argparse.Namespace) -> None:
    if args.quick:
        args.n_features = 32
        args.model_dims = "4,8"
        args.alphas = "0.0"
        args.families = "paper,linear,tropical,pairwise"
        args.batch_size = 16
        args.steps = 3
    if args.paper_scale:
        args.n_features = 1000
        args.model_dims = "10,15,25,39,63,100"
        args.alphas = "0.0,0.25,0.5,0.75,1.0"
        args.batch_size = 2048
        args.steps = 20000


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run sparse feature-recovery scaling benchmarks for tropnn layers.")
    parser.add_argument("--families", type=str, default="paper,linear,tropical,pairwise")
    parser.add_argument("--n-features", type=int, default=256)
    parser.add_argument("--model-dims", type=str, default="8,16,32,64")
    parser.add_argument("--alphas", type=str, default="0.0,0.5,1.0,1.5")
    parser.add_argument("--activation-density", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--paper-lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=-1.0)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--cells", type=int, default=4)
    parser.add_argument("--heads-list", type=str, default="", help="Comma-separated tropical heads sweep; overrides --heads for tropical.")
    parser.add_argument("--cells-list", type=str, default="", help="Comma-separated tropical cells sweep; overrides --cells for tropical.")
    parser.add_argument("--code-scale-mode", choices=CODE_SCALE_MODES, default="sqrt")
    parser.add_argument("--code-scale-modes", type=str, default="", help="Comma-separated tropical code scale modes: sqrt,linear,none.")
    parser.add_argument("--code-geometry-loss", choices=CODE_GEOMETRY_LOSSES, default="none")
    parser.add_argument("--code-geometry-weight", type=float, default=0.0)
    parser.add_argument("--code-norm-weight", type=float, default=0.0)
    parser.add_argument("--route-balance-weight", type=float, default=0.0)
    parser.add_argument("--recovery-diag-weight", type=float, default=0.0)
    parser.add_argument("--recovery-offdiag-weight", type=float, default=0.0)
    parser.add_argument("--output-lr-mult", type=float, default=1.0)
    parser.add_argument("--pairwise-tables", type=int, default=0, help="Use model_dim as PairwiseLinear tables when 0.")
    parser.add_argument("--comparisons", type=int, default=4)
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--backend", choices=("torch", "auto", "triton", "tilelang"), default="torch")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--save-every", type=int, default=0, help="Reserved for future checkpointing; currently must be 0.")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--paper-scale", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.save_every != 0:
        raise ValueError("--save-every is reserved for future checkpointing and must be 0")
    _apply_presets(args)

    configs = _configs_from_args(args)
    rows = []
    for idx, config in enumerate(configs, start=1):
        print(
            f"[{idx}/{len(configs)}] family={config.family} n={config.n_features} m={config.model_dim} "
            f"alpha={config.alpha} heads={config.heads} cells={config.cells} scale={config.code_scale_mode} seed={config.seed}"
        )
        row = run_config(config)
        rows.append(row)
        print(
            f"  loss={float(row['final_loss']):.6g} overlap={float(row['mean_squared_overlap']):.6g} "
            f"represented={float(row['represented_fraction']):.3f}"
        )

    repo_root = Path(__file__).resolve().parents[4]
    output_dir = args.output_dir if args.output_dir is not None else repo_root / "results" / "scaling_benchmark"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = f"{args.tag}-{stamp}" if args.tag else stamp
    csv_path = output_dir / f"runs-{name}.csv"
    json_path = output_dir / f"summary-{name}.json"
    _write_csv(rows, csv_path)
    summary = {"runs": rows, "exponents": _fit_exponents(rows), "group_metrics": _group_metrics(rows)}
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"runs_csv={csv_path}")
    print(f"summary_json={json_path}")


if __name__ == "__main__":
    main()
