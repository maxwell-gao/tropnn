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

from ..layers import PairwiseLUT
from .emnist_payload_dtype_sweep import ResidualPairwiseEmnistClassifier, _build_local_loaders, _eval_model

DiscreteMethod = Literal["ef_sgd", "adam_ef", "factored_adam_ef", "integer_adam_ef", "scaled_integer_adam_ef", "bop2_ternary"]
AccumulatorMode = Literal["float_ef", "int_ef"]


@dataclass(frozen=True)
class DiscretePayloadRow:
    method: str
    bitwidth: int
    accumulator: str
    payload_lr: float
    payload_step_size: float
    accumulator_unit: float
    beta1: float
    beta2: float
    eps: float
    bop_threshold: float
    adam_m_unit: float
    adam_v_unit: float
    row_frequency_normalization: bool
    row_frequency_decay: float
    row_frequency_power: float
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
    train_loss: float
    train_acc: float
    valid_loss: float
    valid_acc: float
    finite_loss_steps: int
    nonfinite_loss_steps: int
    nonfinite_grad_steps: int
    payload_grad_norm_mean: float
    threshold_grad_norm_mean: float
    commit_fraction_mean: float
    saturation_fraction_mean: float
    row_hit_mean: float
    row_hit_max: float
    changed_codes: int
    total_codes: int


@dataclass(frozen=True)
class DiscreteStepStats:
    commit_fraction: float = 0.0
    saturation_fraction: float = 0.0
    changed_codes: int = 0
    total_codes: int = 0
    row_hit_mean: float = 0.0
    row_hit_max: float = 0.0


@dataclass
class _LayerDiscreteState:
    layer: PairwiseLUT
    code: Tensor
    residual: Tensor | None
    qmin: int
    qmax: int
    step_size: float
    accumulator_unit: float
    accumulator: AccumulatorMode
    method: DiscreteMethod
    step: int
    m: Tensor | None
    v: Tensor | None
    row_v: Tensor | None
    col_v: Tensor | None
    row_hits: Tensor | None
    m_unit: float
    v_unit: float


class PairwiseRowHitCounter:
    """Forward-hook row hit counter for PairwiseLUT payload rows.

    The optimizer can use the hit histogram to normalize discrete commits by
    row traffic. This deliberately lives in the experiment tool rather than the
    layer implementation because it duplicates route computation and is meant
    for optimizer diagnostics, not fast training.
    """

    def __init__(self, layers: list[PairwiseLUT]) -> None:
        self._counts: dict[PairwiseLUT, Tensor] = {}
        self._handles = [layer.register_forward_hook(self._make_hook(layer)) for layer in layers]

    def reset(self) -> None:
        self._counts.clear()

    def pop_counts(self) -> dict[PairwiseLUT, Tensor]:
        counts = self._counts
        self._counts = {}
        return counts

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._counts.clear()

    def _make_hook(self, layer: PairwiseLUT):
        def hook(_module: nn.Module, inputs: tuple[Tensor, ...], _output: Tensor) -> None:
            if not inputs:
                return
            with torch.no_grad():
                x = inputs[0].detach()
                if x.ndim == 2:
                    x = x.unsqueeze(1)
                route = layer.cache_index(x.to(torch.float32))
                offsets = torch.arange(layer.tables, device=route.indices.device, dtype=torch.long).view(*([1] * (route.indices.ndim - 1)), -1)
                flat_rows = (route.indices + offsets * layer.table_size).reshape(-1)
                counts = torch.bincount(flat_rows, minlength=layer.tables * layer.table_size).view(layer.tables, layer.table_size).float()
                previous = self._counts.get(layer)
                self._counts[layer] = counts if previous is None else previous + counts

        return hook


class RowLocalDiscretePayloadOptimizer:
    """Row-local discrete optimizer for PairwiseLUT payloads.

    The payload state is the integer `code` tensor. `layer.lut` remains a
    materialized floating tensor only because the existing autograd kernels need
    a differentiable payload input. Optimizer variants differ in how they turn
    row-sparse payload gradients into discrete code commits.
    """

    def __init__(
        self,
        layers: list[PairwiseLUT],
        *,
        method: DiscreteMethod,
        bitwidth: int,
        lr: float,
        step_size: float,
        accumulator: AccumulatorMode,
        accumulator_unit: float,
        beta1: float,
        beta2: float,
        eps: float,
        bop_threshold: float,
        adam_m_unit: float,
        adam_v_unit: float,
        row_frequency_normalization: bool,
        row_frequency_decay: float,
        row_frequency_power: float,
        row_frequency_eps: float,
    ) -> None:
        self._validate(method, bitwidth, step_size, accumulator, accumulator_unit, beta1, beta2, eps, bop_threshold, adam_m_unit, adam_v_unit)
        self.method = method
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.bop_threshold = float(bop_threshold)
        self.adam_m_unit = float(adam_m_unit)
        self.adam_v_unit = float(adam_v_unit)
        self.row_frequency_normalization = bool(row_frequency_normalization)
        self.row_frequency_decay = float(row_frequency_decay)
        self.row_frequency_power = float(row_frequency_power)
        self.row_frequency_eps = float(row_frequency_eps)
        self.states = [self._make_state(layer, method, bitwidth, step_size, accumulator, accumulator_unit) for layer in layers]

    @staticmethod
    def _validate(
        method: DiscreteMethod,
        bitwidth: int,
        step_size: float,
        accumulator: AccumulatorMode,
        accumulator_unit: float,
        beta1: float,
        beta2: float,
        eps: float,
        bop_threshold: float,
        adam_m_unit: float,
        adam_v_unit: float,
    ) -> None:
        if method not in {"ef_sgd", "adam_ef", "factored_adam_ef", "integer_adam_ef", "scaled_integer_adam_ef", "bop2_ternary"}:
            raise ValueError(f"unsupported method {method!r}")
        if method == "bop2_ternary" and bitwidth != 2:
            raise ValueError("bop2_ternary uses the ternary {-1, 0, +1} codebook and requires bitwidth=2")
        if method != "bop2_ternary" and (bitwidth < 2 or bitwidth > 8):
            raise ValueError(f"bitwidth must be in [2, 8], got {bitwidth}")
        if step_size <= 0.0 or accumulator_unit <= 0.0:
            raise ValueError("step_size and accumulator_unit must be positive")
        if accumulator not in {"float_ef", "int_ef"}:
            raise ValueError("accumulator must be 'float_ef' or 'int_ef'")
        if not (0.0 <= beta1 < 1.0) or not (0.0 <= beta2 < 1.0):
            raise ValueError("beta1 and beta2 must be in [0, 1)")
        if eps <= 0.0 or bop_threshold <= 0.0 or adam_m_unit <= 0.0 or adam_v_unit <= 0.0:
            raise ValueError("eps, bop_threshold, adam_m_unit, and adam_v_unit must be positive")

    def _make_state(
        self,
        layer: PairwiseLUT,
        method: DiscreteMethod,
        bitwidth: int,
        step_size: float,
        accumulator: AccumulatorMode,
        accumulator_unit: float,
    ) -> _LayerDiscreteState:
        qmin, qmax = (-1, 1) if method == "bop2_ternary" else (-(1 << (bitwidth - 1)), (1 << (bitwidth - 1)) - 1)
        payload = layer.lut.detach().float()
        code = torch.round(payload / step_size).clamp(qmin, qmax).to(torch.int16)
        residual_mode: AccumulatorMode = "int_ef" if method == "integer_adam_ef" else accumulator
        residual: Tensor | None
        if method == "bop2_ternary":
            residual = None
        else:
            residual_dtype = torch.int32 if residual_mode == "int_ef" else torch.float32
            residual = torch.zeros_like(payload, dtype=residual_dtype)
        full_m = method in {"adam_ef", "factored_adam_ef", "bop2_ternary"}
        full_v = method in {"adam_ef", "bop2_ternary"}
        integer_adam = method in {"integer_adam_ef", "scaled_integer_adam_ef"}
        m = torch.zeros_like(payload, dtype=torch.int32 if integer_adam else torch.float32) if full_m or integer_adam else None
        v = torch.zeros_like(payload, dtype=torch.int32 if integer_adam else torch.float32) if full_v or integer_adam else None
        row_v = torch.zeros(payload.shape[:-1], device=payload.device, dtype=torch.float32) if method == "factored_adam_ef" else None
        col_v = torch.zeros(payload.shape[-1], device=payload.device, dtype=torch.float32) if method == "factored_adam_ef" else None
        row_hits = torch.zeros(payload.shape[:-1], device=payload.device, dtype=torch.float32) if self.row_frequency_normalization else None
        state = _LayerDiscreteState(
            layer,
            code,
            residual,
            qmin,
            qmax,
            float(step_size),
            float(accumulator_unit),
            residual_mode,
            method,
            0,
            m,
            v,
            row_v,
            col_v,
            row_hits,
            self.adam_m_unit,
            self.adam_v_unit,
        )
        self._materialize_payload(state)
        return state

    @torch.no_grad()
    def _materialize_payload(self, state: _LayerDiscreteState) -> None:
        state.layer.lut.copy_(state.code.to(device=state.layer.lut.device, dtype=state.layer.lut.dtype) * state.step_size)
        state.layer.clear_packed_payload_cache()

    def zero_grad(self) -> None:
        for state in self.states:
            state.layer.lut.grad = None

    @torch.no_grad()
    def step(self, row_counts: dict[PairwiseLUT, Tensor] | None = None) -> DiscreteStepStats:
        commit_fracs: list[float] = []
        saturation_fracs: list[float] = []
        hit_means: list[float] = []
        hit_maxes: list[float] = []
        changed_codes = 0
        total_codes = 0
        for state in self.states:
            grad = state.layer.lut.grad
            if grad is None:
                continue
            grad, hit_mean, hit_max = self._prepared_grad(state, grad, None if row_counts is None else row_counts.get(state.layer))
            row_mask = grad.ne(0).any(dim=-1)
            if not bool(row_mask.any().item()):
                continue
            before = state.code.clone()
            total_codes += int(state.code.numel())
            hit_means.append(hit_mean)
            hit_maxes.append(hit_max)
            if state.method == "ef_sgd":
                self._sgd_step(state, grad, row_mask)
            elif state.method == "adam_ef":
                self._adam_step(state, grad, row_mask)
            elif state.method == "factored_adam_ef":
                self._factored_adam_step(state, grad, row_mask)
            elif state.method == "integer_adam_ef":
                self._integer_adam_step(state, grad, row_mask)
            elif state.method == "scaled_integer_adam_ef":
                self._scaled_integer_adam_step(state, grad, row_mask)
            elif state.method == "bop2_ternary":
                self._bop2_ternary_step(state, grad, row_mask)
            else:
                raise AssertionError(f"unreachable method {state.method!r}")
            changed = int((state.code != before).sum().item())
            changed_codes += changed
            commit_fracs.append(changed / max(1, state.code.numel()))
            saturation = ((state.code == state.qmin) | (state.code == state.qmax)).float().mean()
            saturation_fracs.append(float(saturation.item()))
            if changed:
                self._materialize_payload(state)
        return DiscreteStepStats(
            commit_fraction=sum(commit_fracs) / max(1, len(commit_fracs)),
            saturation_fraction=sum(saturation_fracs) / max(1, len(saturation_fracs)),
            changed_codes=changed_codes,
            total_codes=total_codes,
            row_hit_mean=sum(hit_means) / max(1, len(hit_means)),
            row_hit_max=max(hit_maxes) if hit_maxes else 0.0,
        )

    def _prepared_grad(self, state: _LayerDiscreteState, grad: Tensor, row_count: Tensor | None) -> tuple[Tensor, float, float]:
        grad = grad.detach().float()
        if not self.row_frequency_normalization or state.row_hits is None:
            return grad, 0.0, 0.0
        if row_count is None:
            row_count = grad.ne(0).any(dim=-1).float()
        row_count = row_count.to(device=grad.device, dtype=torch.float32)
        state.row_hits.mul_(self.row_frequency_decay).add_(row_count, alpha=1.0 - self.row_frequency_decay)
        active = state.row_hits > 0
        mean = state.row_hits[active].mean().clamp_min(self.row_frequency_eps) if bool(active.any().item()) else torch.ones((), device=grad.device)
        scale = (state.row_hits.clamp_min(self.row_frequency_eps) / mean).pow(self.row_frequency_power)
        return grad / scale.unsqueeze(-1), float(row_count.mean().item()), float(row_count.max().item())

    def _sgd_step(self, state: _LayerDiscreteState, grad: Tensor, row_mask: Tensor) -> None:
        self._commit_delta(state, row_mask, -self.lr * grad[row_mask])

    def _adam_step(self, state: _LayerDiscreteState, grad: Tensor, row_mask: Tensor) -> None:
        if state.m is None or state.v is None:
            raise RuntimeError("adam_ef state is missing m/v")
        state.step += 1
        g = grad[row_mask]
        m_rows = state.m[row_mask].float() * self.beta1 + g * (1.0 - self.beta1)
        v_rows = state.v[row_mask].float() * self.beta2 + g.square() * (1.0 - self.beta2)
        state.m[row_mask] = m_rows
        state.v[row_mask] = v_rows
        bias1 = 1.0 - self.beta1**state.step
        bias2 = 1.0 - self.beta2**state.step
        update = (m_rows / bias1) / ((v_rows / bias2).sqrt() + self.eps)
        self._commit_delta(state, row_mask, -self.lr * update)

    def _factored_adam_step(self, state: _LayerDiscreteState, grad: Tensor, row_mask: Tensor) -> None:
        if state.m is None or state.row_v is None or state.col_v is None:
            raise RuntimeError("factored_adam_ef state is missing m/row_v/col_v")
        state.step += 1
        g = grad[row_mask]
        m_rows = state.m[row_mask].float() * self.beta1 + g * (1.0 - self.beta1)
        state.m[row_mask] = m_rows
        g2 = g.square()
        row_v_rows = state.row_v[row_mask] * self.beta2 + g2.mean(dim=-1) * (1.0 - self.beta2)
        state.row_v[row_mask] = row_v_rows
        state.col_v.mul_(self.beta2).add_(g2.mean(dim=0), alpha=1.0 - self.beta2)
        bias1 = 1.0 - self.beta1**state.step
        bias2 = 1.0 - self.beta2**state.step
        row_hat = row_v_rows / bias2
        col_hat = state.col_v / bias2
        col_shape = col_hat / col_hat.mean().clamp_min(self.eps)
        denom = (row_hat.unsqueeze(-1) * col_shape.unsqueeze(0)).sqrt() + self.eps
        update = (m_rows / bias1) / denom
        self._commit_delta(state, row_mask, -self.lr * update)

    def _integer_adam_step(self, state: _LayerDiscreteState, grad: Tensor, row_mask: Tensor) -> None:
        if state.m is None or state.v is None:
            raise RuntimeError("integer_adam_ef state is missing integer m/v")
        state.step += 1
        g = grad[row_mask]
        m_old = state.m[row_mask].float() * self.adam_m_unit
        v_old = state.v[row_mask].float() * self.adam_v_unit
        m_new = m_old * self.beta1 + g * (1.0 - self.beta1)
        v_new = v_old * self.beta2 + g.square() * (1.0 - self.beta2)
        state.m[row_mask] = torch.round(m_new / self.adam_m_unit).clamp(-2_147_483_648, 2_147_483_647).to(torch.int32)
        state.v[row_mask] = torch.round(v_new / self.adam_v_unit).clamp(0, 2_147_483_647).to(torch.int32)
        bias1 = 1.0 - self.beta1**state.step
        bias2 = 1.0 - self.beta2**state.step
        m_hat = state.m[row_mask].float() * self.adam_m_unit / bias1
        v_hat = state.v[row_mask].float() * self.adam_v_unit / bias2
        update = m_hat / (v_hat.sqrt() + self.eps)
        self._commit_delta(state, row_mask, -self.lr * update)

    def _scaled_integer_adam_step(self, state: _LayerDiscreteState, grad: Tensor, row_mask: Tensor) -> None:
        if state.m is None or state.v is None:
            raise RuntimeError("scaled_integer_adam_ef state is missing integer m/v")
        state.step += 1
        m_full = state.m.float() * state.m_unit
        v_full = state.v.float() * state.v_unit
        g = grad[row_mask]
        m_full[row_mask] = m_full[row_mask] * self.beta1 + g * (1.0 - self.beta1)
        v_full[row_mask] = v_full[row_mask] * self.beta2 + g.square() * (1.0 - self.beta2)
        state.m_unit = max(self.adam_m_unit, float(m_full.abs().amax().item()) / 32767.0)
        state.v_unit = max(self.adam_v_unit, float(v_full.amax().item()) / 65535.0)
        state.m.copy_(torch.round(m_full / state.m_unit).clamp(-32768, 32767).to(torch.int32))
        state.v.copy_(torch.round(v_full / state.v_unit).clamp(0, 65535).to(torch.int32))
        bias1 = 1.0 - self.beta1**state.step
        bias2 = 1.0 - self.beta2**state.step
        m_hat = state.m[row_mask].float() * state.m_unit / bias1
        v_hat = state.v[row_mask].float() * state.v_unit / bias2
        update = m_hat / (v_hat.sqrt() + self.eps)
        self._commit_delta(state, row_mask, -self.lr * update)

    def _bop2_ternary_step(self, state: _LayerDiscreteState, grad: Tensor, row_mask: Tensor) -> None:
        if state.m is None or state.v is None:
            raise RuntimeError("bop2_ternary state is missing m/v")
        state.step += 1
        g = grad[row_mask]
        m_rows = state.m[row_mask].float() * self.beta1 + g * (1.0 - self.beta1)
        v_rows = state.v[row_mask].float() * self.beta2 + g.square() * (1.0 - self.beta2)
        bias1 = 1.0 - self.beta1**state.step
        bias2 = 1.0 - self.beta2**state.step
        score = (m_rows / bias1) / ((v_rows / bias2).sqrt() + self.eps)
        flip = score.abs() > self.bop_threshold
        old_code = state.code[row_mask]
        proposal = (old_code - score.sign().to(torch.int16)).clamp(state.qmin, state.qmax).to(torch.int16)
        new_code = torch.where(flip, proposal, old_code)
        changed = new_code != old_code
        state.code[row_mask] = new_code
        state.m[row_mask] = torch.where(changed, torch.zeros_like(m_rows), m_rows)
        state.v[row_mask] = torch.where(changed, torch.zeros_like(v_rows), v_rows)

    def _commit_delta(self, state: _LayerDiscreteState, row_mask: Tensor, desired_delta: Tensor) -> None:
        if state.residual is None:
            raise RuntimeError("residual commit path is unavailable for this optimizer")
        if state.accumulator == "int_ef":
            update_units = torch.round(desired_delta / state.accumulator_unit).to(torch.int32)
            residual_rows = state.residual[row_mask] + update_units
            quantum_units = max(1, int(round(state.step_size / state.accumulator_unit)))
            commit = torch.div(residual_rows.abs(), quantum_units, rounding_mode="floor") * residual_rows.sign()
            commit = commit.to(torch.int16)
            old_code = state.code[row_mask]
            new_code = (old_code + commit).clamp(state.qmin, state.qmax).to(torch.int16)
            residual_rows = residual_rows - (new_code - old_code).to(torch.int32) * quantum_units
            state.residual[row_mask] = residual_rows
        else:
            residual_rows = state.residual[row_mask].float() + desired_delta
            commit = torch.floor(residual_rows.abs() / state.step_size) * residual_rows.sign()
            commit = commit.to(torch.int16)
            old_code = state.code[row_mask]
            new_code = (old_code + commit).clamp(state.qmin, state.qmax).to(torch.int16)
            residual_rows = residual_rows - (new_code - old_code).to(torch.float32) * state.step_size
            state.residual[row_mask] = residual_rows
        state.code[row_mask] = new_code


def _pairwise_layers(model: nn.Module) -> list[PairwiseLUT]:
    return [module for module in model.modules() if isinstance(module, PairwiseLUT)]


def _loader_examples(loader) -> int:
    dataset = getattr(loader, "dataset", None)
    return int(len(dataset)) if dataset is not None else 0


def _grad_norm(model: nn.Module, fragment: str) -> tuple[float, bool]:
    total = 0.0
    finite = True
    for name, param in model.named_parameters():
        if fragment not in name or param.grad is None:
            continue
        grad = param.grad.detach().float()
        finite = finite and bool(torch.isfinite(grad).all().item())
        total += float(grad.square().sum().item())
    return math.sqrt(total), finite


def _build_model(args: argparse.Namespace, classes: int) -> nn.Module:
    if args.architecture != "residual":
        raise ValueError("Only residual architecture is supported for discrete payload training.")
    return ResidualPairwiseEmnistClassifier(
        input_dim=28 * 28,
        num_classes=classes,
        depth=args.depth,
        tables=args.tables,
        comparisons=args.comparisons,
        seed=args.seed,
        backend=args.backend,
        anchor_policy=args.anchor_policy,
        lut_dtype="fp32",
        lut_init_std=args.lut_init_std,
    )


def run(args: argparse.Namespace) -> DiscretePayloadRow:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    method: DiscreteMethod = args.method
    bitwidth = 2 if method == "bop2_ternary" else int(args.bitwidth)
    accumulator: AccumulatorMode = "int_ef" if method in {"integer_adam_ef", "scaled_integer_adam_ef"} else args.accumulator
    train_loader, valid_loader, classes = _build_local_loaders(args)
    model = _build_model(args, classes).to(device)
    payload_layers = _pairwise_layers(model)
    row_counter = PairwiseRowHitCounter(payload_layers) if args.row_frequency_normalization else None
    payload_opt = RowLocalDiscretePayloadOptimizer(
        payload_layers,
        method=method,
        bitwidth=bitwidth,
        lr=args.payload_lr,
        step_size=args.payload_step_size,
        accumulator=accumulator,
        accumulator_unit=args.accumulator_unit,
        beta1=args.beta1,
        beta2=args.beta2,
        eps=args.eps,
        bop_threshold=args.bop_threshold,
        adam_m_unit=args.adam_m_unit,
        adam_v_unit=args.adam_v_unit,
        row_frequency_normalization=args.row_frequency_normalization,
        row_frequency_decay=args.row_frequency_decay,
        row_frequency_power=args.row_frequency_power,
        row_frequency_eps=args.row_frequency_eps,
    )
    threshold_params = [param for name, param in model.named_parameters() if "thresholds" in name]
    threshold_opt = torch.optim.AdamW(threshold_params, lr=args.threshold_lr, weight_decay=0.0)

    finite_loss_steps = 0
    nonfinite_loss_steps = 0
    nonfinite_grad_steps = 0
    payload_grad_norms: list[float] = []
    threshold_grad_norms: list[float] = []
    commit_fracs: list[float] = []
    saturation_fracs: list[float] = []
    row_hit_means: list[float] = []
    row_hit_maxes: list[float] = []
    changed_codes = 0
    total_codes = sum(layer.lut.numel() for layer in payload_layers)
    last_train_loss = 0.0
    last_train_acc = 0.0

    try:
        for _epoch in range(args.epochs):
            model.train()
            total_loss = 0.0
            total_correct = 0
            total_seen = 0
            for x, y in train_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                threshold_opt.zero_grad(set_to_none=True)
                payload_opt.zero_grad()
                if row_counter is not None:
                    row_counter.reset()
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                batch = int(y.numel())
                if torch.isfinite(loss):
                    finite_loss_steps += 1
                else:
                    nonfinite_loss_steps += 1
                loss.backward()
                payload_norm, payload_finite = _grad_norm(model, "lut")
                threshold_norm, threshold_finite = _grad_norm(model, "thresholds")
                if not payload_finite or not threshold_finite:
                    nonfinite_grad_steps += 1
                payload_grad_norms.append(payload_norm)
                threshold_grad_norms.append(threshold_norm)
                threshold_opt.step()
                stats = payload_opt.step(None if row_counter is None else row_counter.pop_counts())
                commit_fracs.append(stats.commit_fraction)
                saturation_fracs.append(stats.saturation_fraction)
                row_hit_means.append(stats.row_hit_mean)
                row_hit_maxes.append(stats.row_hit_max)
                changed_codes += stats.changed_codes
                total_loss += float(loss.detach().item()) * batch
                total_correct += int((logits.argmax(dim=-1) == y).sum().item())
                total_seen += batch
            last_train_loss = total_loss / max(1, total_seen)
            last_train_acc = total_correct / max(1, total_seen)
    finally:
        if row_counter is not None:
            row_counter.close()

    valid_loss, valid_acc = _eval_model(model, valid_loader, device=device)
    return DiscretePayloadRow(
        method=method,
        bitwidth=bitwidth,
        accumulator=accumulator,
        payload_lr=args.payload_lr,
        payload_step_size=args.payload_step_size,
        accumulator_unit=args.accumulator_unit,
        beta1=args.beta1,
        beta2=args.beta2,
        eps=args.eps,
        bop_threshold=args.bop_threshold,
        adam_m_unit=args.adam_m_unit,
        adam_v_unit=args.adam_v_unit,
        row_frequency_normalization=args.row_frequency_normalization,
        row_frequency_decay=args.row_frequency_decay,
        row_frequency_power=args.row_frequency_power,
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
        train_loss=last_train_loss,
        train_acc=last_train_acc,
        valid_loss=valid_loss,
        valid_acc=valid_acc,
        finite_loss_steps=finite_loss_steps,
        nonfinite_loss_steps=nonfinite_loss_steps,
        nonfinite_grad_steps=nonfinite_grad_steps,
        payload_grad_norm_mean=sum(payload_grad_norms) / max(1, len(payload_grad_norms)),
        threshold_grad_norm_mean=sum(threshold_grad_norms) / max(1, len(threshold_grad_norms)),
        commit_fraction_mean=sum(commit_fracs) / max(1, len(commit_fracs)),
        saturation_fraction_mean=sum(saturation_fracs) / max(1, len(saturation_fracs)),
        row_hit_mean=sum(row_hit_means) / max(1, len(row_hit_means)),
        row_hit_max=max(row_hit_maxes) if row_hit_maxes else 0.0,
        changed_codes=changed_codes,
        total_codes=total_codes,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train EMNIST balanced with row-local discrete PairwiseLUT payload updates.")
    parser.add_argument("--root", default="data/emnist")
    parser.add_argument("--split", default="balanced")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--backend", choices=("auto", "torch", "tilelang", "triton"), default="tilelang")
    parser.add_argument("--architecture", choices=("residual",), default="residual")
    parser.add_argument("--method", choices=("ef_sgd", "adam_ef", "factored_adam_ef", "integer_adam_ef", "scaled_integer_adam_ef", "bop2_ternary"), default="ef_sgd")
    parser.add_argument("--bitwidth", type=int, choices=(2, 3, 4, 8), default=4)
    parser.add_argument("--accumulator", choices=("float_ef", "int_ef"), default="float_ef")
    parser.add_argument("--payload-lr", type=float, default=1e-2)
    parser.add_argument("--payload-step-size", type=float, default=0.05)
    parser.add_argument("--accumulator-unit", type=float, default=0.0002)
    parser.add_argument("--threshold-lr", type=float, default=1e-3)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--bop-threshold", type=float, default=0.1)
    parser.add_argument("--adam-m-unit", type=float, default=1e-5)
    parser.add_argument("--adam-v-unit", type=float, default=1e-8)
    parser.add_argument("--row-frequency-normalization", action="store_true")
    parser.add_argument("--row-frequency-decay", type=float, default=0.95)
    parser.add_argument("--row-frequency-power", type=float, default=0.5)
    parser.add_argument("--row-frequency-eps", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--tables", type=int, default=64)
    parser.add_argument("--comparisons", type=int, default=6)
    parser.add_argument("--lut-init-std", type=float, default=0.0)
    parser.add_argument("--anchor-policy", default="permuted")
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-test", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/emnist_discrete_payload/summary.csv")
    args = parser.parse_args()

    row = run(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(row).keys()))
        writer.writeheader()
        writer.writerow(asdict(row))
    print(
        f"method={row.method} bits={row.bitwidth} accumulator={row.accumulator} "
        f"train_loss={row.train_loss:.4f} train_acc={row.train_acc:.4f} "
        f"valid_loss={row.valid_loss:.4f} valid_acc={row.valid_acc:.4f} "
        f"commit_frac={row.commit_fraction_mean:.6f} saturation={row.saturation_fraction_mean:.6f} "
        f"row_hit_mean={row.row_hit_mean:.3f} row_hit_max={row.row_hit_max:.1f} "
        f"nonfinite_loss={row.nonfinite_loss_steps} nonfinite_grad={row.nonfinite_grad_steps}",
        flush=True,
    )
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
