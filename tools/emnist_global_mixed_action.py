"""Test sparse global mixed-action operators on EMNIST Balanced.

Each hidden block is one complete radix-2 butterfly sweep.  Blocks are
near-identity state maps and compose directly; equivalently, each update is
``state += block(state) - state``.  Every pair uses two hard,
branch-conditioned shears.  The ``live_reroute`` arm lets the first shear
change the route selected by the second shear; ``pre_shear_route`` is the
matched control whose second route reads the pre-shear state.  All arms use the
same PairwiseLUT classification head and no learned dense matrix.

The hard forward uses a fixed tanh straight-through derivative.  It is a local
soft-chain surrogate, not exact suffix replay.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from tropnn.layers import PairwiseLUT
from tropnn.tools.emnist_payload_dtype_sweep import _load_emnist_split

Arm = Literal["readout_only", "pre_shear_route", "live_reroute"]


def hard_tanh_ste(margin: Tensor, tau: float) -> Tensor:
    """Return an exactly hard bit with the derivative of a tanh transition."""

    if tau <= 0.0:
        raise ValueError(f"tau must be positive, got {tau}")
    hard = (margin >= 0.0).to(margin.dtype)
    soft = 0.5 * (torch.tanh(margin / tau) + 1.0)
    return hard + soft - soft.detach()


def _inverse_tanh_scaled(value: float, limit: float) -> float:
    ratio = value / limit
    if not -1.0 < ratio < 1.0:
        raise ValueError(f"initial value {value} must lie strictly inside +/-{limit}")
    return math.atanh(ratio)


def _rms(x: Tensor) -> float:
    return float(x.detach().float().square().mean().sqrt().item()) if x.numel() else 0.0


def _tensor_sha256(items: Iterable[tuple[str, Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in items:
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class CoreLedger:
    carrier_dim: int
    depth: int
    rounds: int
    stages: int
    pairs_per_round: int
    trainable_parameters: int
    comparisons_per_example: int
    semantic_macs_per_example: int
    coordinate_writes_per_example: int
    receptive_field: int


class GlobalMixedActionButterfly(nn.Module):
    """A complete butterfly of shared, branch-conditioned two-shear actions."""

    def __init__(
        self,
        *,
        carrier_dim: int = 1024,
        rounds: int | None = None,
        arm: Arm = "live_reroute",
        tau: float = 0.1,
        action_limit: float = 0.25,
        dissipation_span: float = 0.25,
        action_init: float = 0.02,
    ) -> None:
        super().__init__()
        if carrier_dim < 2 or carrier_dim & (carrier_dim - 1):
            raise ValueError(f"carrier_dim must be a power of two >=2, got {carrier_dim}")
        full_rounds = int(math.log2(carrier_dim))
        rounds = full_rounds if rounds is None else int(rounds)
        if not 1 <= rounds <= full_rounds:
            raise ValueError(f"rounds must be in [1,{full_rounds}], got {rounds}")
        if arm not in {"readout_only", "pre_shear_route", "live_reroute"}:
            raise ValueError(f"unknown arm {arm!r}")
        if action_limit <= 0.0 or dissipation_span <= 0.0:
            raise ValueError("action_limit and dissipation_span must be positive")
        if not 0.0 <= action_init < action_limit:
            raise ValueError("action_init must be nonnegative and below action_limit")

        self.carrier_dim = int(carrier_dim)
        self.rounds = rounds
        self.arm: Arm = arm
        self.tau = float(tau)
        self.action_limit = float(action_limit)
        self.dissipation_span = float(dissipation_span)
        self.pairs_per_round = self.carrier_dim // 2

        init_master = _inverse_tanh_scaled(action_init, action_limit)
        raw_a = torch.empty(self.rounds, 2)
        raw_b = torch.empty(self.rounds, 2)
        raw_a[:, 0] = -init_master
        raw_a[:, 1] = init_master
        raw_b[:, 0] = init_master
        raw_b[:, 1] = -init_master
        self.raw_a = nn.Parameter(raw_a)
        self.raw_b = nn.Parameter(raw_b)
        self.raw_d = nn.Parameter(torch.zeros(self.rounds, 2))
        self.theta_q = nn.Parameter(torch.zeros(self.rounds, self.pairs_per_round))
        self.theta_s = nn.Parameter(torch.zeros(self.rounds, self.pairs_per_round))

        self.register_buffer("initial_raw_a", raw_a.clone())
        self.register_buffer("initial_raw_b", raw_b.clone())
        self.register_buffer("initial_raw_d", torch.zeros_like(self.raw_d))
        self.register_buffer("initial_theta_q", torch.zeros_like(self.theta_q))
        self.register_buffer("initial_theta_s", torch.zeros_like(self.theta_s))

        if self.arm == "readout_only":
            for parameter in self.parameters():
                parameter.requires_grad_(False)

    def effective_actions(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        a = self.action_limit * torch.tanh(self.raw_a)
        b = self.action_limit * torch.tanh(self.raw_b)
        d = 1.0 + self.dissipation_span * torch.tanh(self.raw_d)
        return a, b, d[:, 0], d[:, 1]

    def _split_round(self, state: Tensor, round_index: int) -> tuple[Tensor, Tensor, int]:
        stride = 1 << round_index
        paired = state.reshape(state.shape[0], -1, 2, stride)
        return paired[:, :, 0, :].reshape(state.shape[0], -1), paired[:, :, 1, :].reshape(state.shape[0], -1), stride

    @staticmethod
    def _join_round(u: Tensor, v: Tensor, stride: int, carrier_dim: int) -> Tensor:
        batch = u.shape[0]
        blocks = carrier_dim // (2 * stride)
        return torch.stack((u.reshape(batch, blocks, stride), v.reshape(batch, blocks, stride)), dim=2).reshape(batch, carrier_dim)

    def _round(
        self,
        state: Tensor,
        round_index: int,
        *,
        hard_only: bool = False,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        u, v, stride = self._split_round(state, round_index)
        a, b, d_u, d_v = self.effective_actions()
        margin_q = u - v - self.theta_q[round_index].to(dtype=state.dtype)
        q = (margin_q >= 0.0).to(state.dtype) if hard_only else hard_tanh_ste(margin_q, self.tau)
        a_selected = a[round_index, 0] + q * (a[round_index, 1] - a[round_index, 0])
        h = d_v[round_index] * v + a_selected * u

        margin_s_pre = v - u - self.theta_s[round_index].to(dtype=state.dtype)
        margin_s_live = h - u - self.theta_s[round_index].to(dtype=state.dtype)
        margin_s = margin_s_live if self.arm == "live_reroute" else margin_s_pre
        s = (margin_s >= 0.0).to(state.dtype) if hard_only else hard_tanh_ste(margin_s, self.tau)
        b_selected = b[round_index, 0] + s * (b[round_index, 1] - b[round_index, 0])
        new_u = d_u[round_index] * u + b_selected * h
        new_v = h
        output = self._join_round(new_u, new_v, stride, self.carrier_dim)
        return output, {
            "u": u,
            "v": v,
            "q": q,
            "s": s,
            "h": h,
            "margin_s_pre": margin_s_pre,
            "margin_s_live": margin_s_live,
            "new_u": new_u,
            "new_v": new_v,
        }

    def forward(self, state: Tensor) -> Tensor:
        if state.ndim != 2 or state.shape[1] != self.carrier_dim:
            raise ValueError(f"expected [batch,{self.carrier_dim}], got {tuple(state.shape)}")
        if self.arm == "readout_only":
            return state
        for round_index in range(self.rounds):
            state, _ = self._round(state, round_index)
        return state

    @torch.no_grad()
    def hard_route_bits(self, state: Tensor) -> list[tuple[Tensor, Tensor]]:
        """Return the sequential hard q/s routes taken inside this block."""

        routes: list[tuple[Tensor, Tensor]] = []
        if self.arm == "readout_only":
            return routes
        for round_index in range(self.rounds):
            state, values = self._round(state, round_index, hard_only=True)
            routes.append((values["q"], values["s"]))
        return routes

    @torch.no_grad()
    def trace(self, state: Tensor) -> list[dict[str, float | int]]:
        """Trace hard branch behavior without changing the training semantics."""

        if self.arm == "readout_only":
            return [
                {
                    "round": round_index,
                    "q_fraction": math.nan,
                    "s_fraction": math.nan,
                    "joint_00_fraction": math.nan,
                    "joint_01_fraction": math.nan,
                    "joint_10_fraction": math.nan,
                    "joint_11_fraction": math.nan,
                    "live_vs_pre_reroute_fraction": 0.0,
                    "actual_reroute_fraction": 0.0,
                    "forced_q_sensitivity_fraction": 0.0,
                    "state_in_rms": _rms(state),
                    "state_out_rms": _rms(state),
                    "action_rms": 0.0,
                    "mixed_amplitude": math.nan,
                }
                for round_index in range(self.rounds)
            ]

        traces: list[dict[str, float | int]] = []
        a, b, d_u, d_v = self.effective_actions()
        for round_index in range(self.rounds):
            state_in = state
            state, values = self._round(state, round_index, hard_only=True)
            q = values["q"]
            s = values["s"]
            s_pre = (values["margin_s_pre"] >= 0.0).to(s.dtype)
            s_live = (values["margin_s_live"] >= 0.0).to(s.dtype)

            q_flip = 1.0 - q
            a_flip = a[round_index, 0] + q_flip * (a[round_index, 1] - a[round_index, 0])
            h_flip = d_v[round_index] * values["v"] + a_flip * values["u"]
            s_flip = (h_flip - values["u"] - self.theta_s[round_index] >= 0.0).to(s.dtype)
            traces.append(
                {
                    "round": round_index,
                    "q_fraction": float(q.mean().item()),
                    "s_fraction": float(s.mean().item()),
                    "joint_00_fraction": float(((q == 0) & (s == 0)).float().mean().item()),
                    "joint_01_fraction": float(((q == 0) & (s == 1)).float().mean().item()),
                    "joint_10_fraction": float(((q == 1) & (s == 0)).float().mean().item()),
                    "joint_11_fraction": float(((q == 1) & (s == 1)).float().mean().item()),
                    "live_vs_pre_reroute_fraction": float((s_live != s_pre).float().mean().item()),
                    "actual_reroute_fraction": (
                        float((s_live != s_pre).float().mean().item()) if self.arm == "live_reroute" else 0.0
                    ),
                    "forced_q_sensitivity_fraction": (
                        float((s_flip != s_live).float().mean().item()) if self.arm == "live_reroute" else 0.0
                    ),
                    "state_in_rms": _rms(state_in),
                    "state_out_rms": _rms(state),
                    "action_rms": _rms(state - state_in),
                    "a0": float(a[round_index, 0].item()),
                    "a1": float(a[round_index, 1].item()),
                    "b0": float(b[round_index, 0].item()),
                    "b1": float(b[round_index, 1].item()),
                    "d_u": float(d_u[round_index].item()),
                    "d_v": float(d_v[round_index].item()),
                    "determinant": float((d_u[round_index] * d_v[round_index]).item()),
                    "mixed_amplitude": float(
                        ((a[round_index, 1] - a[round_index, 0]) * (b[round_index, 1] - b[round_index, 0])).abs().item()
                    ),
                    "theta_q_mean": float(self.theta_q[round_index].mean().item()),
                    "theta_q_std": float(self.theta_q[round_index].std(unbiased=False).item()),
                    "theta_s_mean": float(self.theta_s[round_index].mean().item()),
                    "theta_s_std": float(self.theta_s[round_index].std(unbiased=False).item()),
                }
            )
        return traces

    def ledger(self) -> CoreLedger:
        pairs = self.pairs_per_round
        return CoreLedger(
            carrier_dim=self.carrier_dim,
            depth=1,
            rounds=self.rounds,
            stages=self.rounds,
            pairs_per_round=pairs,
            trainable_parameters=self.rounds * (2 * pairs + 6),
            comparisons_per_example=self.rounds * 2 * pairs,
            semantic_macs_per_example=self.rounds * 4 * pairs,
            coordinate_writes_per_example=self.rounds * 2 * pairs,
            receptive_field=1 << self.rounds,
        )

    def initial_hash(self) -> str:
        return _tensor_sha256(
            (
                ("raw_a", self.initial_raw_a),
                ("raw_b", self.initial_raw_b),
                ("raw_d", self.initial_raw_d),
                ("theta_q", self.initial_theta_q),
                ("theta_s", self.initial_theta_s),
            )
        )

    def displacement(self) -> dict[str, float]:
        a0 = self.action_limit * torch.tanh(self.initial_raw_a)
        b0 = self.action_limit * torch.tanh(self.initial_raw_b)
        d0 = 1.0 + self.dissipation_span * torch.tanh(self.initial_raw_d)
        a, b, d_u, d_v = self.effective_actions()
        d = torch.stack((d_u, d_v), dim=-1)
        return {
            "a_displacement_rms": _rms(a - a0),
            "b_displacement_rms": _rms(b - b0),
            "d_displacement_rms": _rms(d - d0),
            "theta_q_displacement_rms": _rms(self.theta_q - self.initial_theta_q),
            "theta_s_displacement_rms": _rms(self.theta_s - self.initial_theta_s),
        }


class GlobalMixedActionResidualStack(nn.Module):
    """Serial near-identity mixed-action blocks with independent parameters."""

    def __init__(
        self,
        *,
        depth: int,
        carrier_dim: int,
        rounds: int,
        arm: Arm,
        tau: float,
        action_limit: float,
        dissipation_span: float,
        action_init: float,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >=1, got {depth}")
        self.depth = int(depth)
        self.carrier_dim = int(carrier_dim)
        self.rounds = int(rounds)
        self.arm: Arm = arm
        self.blocks = nn.ModuleList(
            GlobalMixedActionButterfly(
                carrier_dim=carrier_dim,
                rounds=rounds,
                arm=arm,
                tau=tau,
                action_limit=action_limit,
                dissipation_span=dissipation_span,
                action_init=action_init,
            )
            for _ in range(depth)
        )

    def forward(self, state: Tensor) -> Tensor:
        for block in self.blocks:
            # A butterfly core already includes its identity carrier.  Direct
            # composition is the residual update state += block(state)-state.
            state = block(state)
        return state

    @torch.no_grad()
    def trace(self, state: Tensor) -> list[dict[str, float | int]]:
        rows: list[dict[str, float | int]] = []
        input_rms = _rms(state)
        previous_block_input: Tensor | None = None
        for block_index, block in enumerate(self.blocks):
            state_in = state
            upstream_route_change = math.nan
            if previous_block_input is not None and self.arm != "readout_only":
                actual_routes = block.hard_route_bits(state_in)
                skipped_routes = block.hard_route_bits(previous_block_input)
                changed = 0.0
                count = 0
                for (actual_q, actual_s), (skipped_q, skipped_s) in zip(actual_routes, skipped_routes, strict=True):
                    changed += float((actual_q != skipped_q).float().sum().item())
                    changed += float((actual_s != skipped_s).float().sum().item())
                    count += actual_q.numel() + actual_s.numel()
                upstream_route_change = changed / max(1, count)
            block_rows = block.trace(state_in)
            state = block(state_in)
            block_input_rms = _rms(state_in)
            block_output_rms = _rms(state)
            block_delta_rms = _rms(state - state_in)
            block_gain_rms = block_output_rms / max(block_input_rms, 1e-12)
            relative_output_rms = block_output_rms / max(input_rms, 1e-12)
            for row in block_rows:
                rows.append(
                    {
                        "block": block_index,
                        "stage": block_index * self.rounds + int(row["round"]),
                        **row,
                        "block_input_rms": block_input_rms,
                        "block_output_rms": block_output_rms,
                        "block_delta_rms": block_delta_rms,
                        "block_gain_rms": block_gain_rms,
                        "block_output_over_input_rms": relative_output_rms,
                        "upstream_skip_route_change_fraction": upstream_route_change,
                    }
                )
            previous_block_input = state_in
        return rows

    def ledger(self) -> CoreLedger:
        block = self.blocks[0].ledger()
        return CoreLedger(
            carrier_dim=block.carrier_dim,
            depth=self.depth,
            rounds=block.rounds,
            stages=self.depth * block.rounds,
            pairs_per_round=block.pairs_per_round,
            trainable_parameters=self.depth * block.trainable_parameters,
            comparisons_per_example=self.depth * block.comparisons_per_example,
            semantic_macs_per_example=self.depth * block.semantic_macs_per_example,
            coordinate_writes_per_example=self.depth * block.coordinate_writes_per_example,
            receptive_field=block.receptive_field,
        )

    def initial_hash(self) -> str:
        items: list[tuple[str, Tensor]] = []
        for block_index, block in enumerate(self.blocks):
            items.extend(
                (
                    (f"blocks.{block_index}.raw_a", block.initial_raw_a),
                    (f"blocks.{block_index}.raw_b", block.initial_raw_b),
                    (f"blocks.{block_index}.raw_d", block.initial_raw_d),
                    (f"blocks.{block_index}.theta_q", block.initial_theta_q),
                    (f"blocks.{block_index}.theta_s", block.initial_theta_s),
                )
            )
        return _tensor_sha256(items)

    def block_initial_hashes(self) -> list[str]:
        return [block.initial_hash() for block in self.blocks]

    def action_parameters(self) -> Iterable[nn.Parameter]:
        for block in self.blocks:
            yield block.raw_a
            yield block.raw_b
            yield block.raw_d

    def threshold_parameters(self) -> Iterable[nn.Parameter]:
        for block in self.blocks:
            yield block.theta_q
            yield block.theta_s

    def block_grad_norms(self) -> list[float]:
        return [_parameter_grad_norm(block.parameters()) for block in self.blocks]

    def block_displacements(self) -> list[dict[str, float]]:
        return [block.displacement() for block in self.blocks]

    def displacement(self) -> dict[str, float]:
        rows = self.block_displacements()
        return {
            key: math.sqrt(sum(row[key] * row[key] for row in rows) / len(rows))
            for key in rows[0]
        }


class EmnistGlobalMixedActionClassifier(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        carrier_dim: int,
        classes: int,
        arm: Arm,
        seed: int,
        core_depth: int,
        rounds: int,
        tau: float,
        action_limit: float,
        dissipation_span: float,
        action_init: float,
        tables: int,
        comparisons: int,
    ) -> None:
        super().__init__()
        if input_dim > carrier_dim:
            raise ValueError(f"input_dim {input_dim} exceeds carrier_dim {carrier_dim}")
        self.input_dim = int(input_dim)
        self.carrier_dim = int(carrier_dim)
        self.core = GlobalMixedActionResidualStack(
            depth=core_depth,
            carrier_dim=carrier_dim,
            rounds=rounds,
            arm=arm,
            tau=tau,
            action_limit=action_limit,
            dissipation_span=dissipation_span,
            action_init=action_init,
        )
        self.readout = PairwiseLUT(
            input_dim,
            classes,
            tables=tables,
            comparisons=comparisons,
            backend="torch",
            seed=seed + 1000,
            anchor_seed=seed + 2000,
            lut_init_std=0.0,
            use_min_margin_ste=True,
            use_output_scaling=True,
            fixed_zero_threshold=False,
            surrogate="izhikevich",
            anchor_policy="permuted",
            lut_dtype="fp32",
        )
        self._head_initial_hash = _tensor_sha256(
            (
                ("anchors", self.readout.anchors),
                ("thresholds", self.readout.thresholds),
                ("lut", self.readout.lut),
            )
        )

    def carrier(self, x: Tensor) -> Tensor:
        flat = x.flatten(1)
        if flat.shape[1] != self.input_dim:
            raise ValueError(f"expected flattened input {self.input_dim}, got {flat.shape[1]}")
        if self.carrier_dim > self.input_dim:
            flat = F.pad(flat, (0, self.carrier_dim - self.input_dim))
        return self.core(flat)

    def forward(self, x: Tensor) -> Tensor:
        hidden = self.carrier(x)[:, : self.input_dim]
        return self.readout(hidden).squeeze(1)

    def head_initial_hash(self) -> str:
        return self._head_initial_hash


def _parameter_grad_norm(parameters: Iterable[nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach().float()
        total += float(grad.square().sum().item())
    return math.sqrt(total)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _write_csv_atomic(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _make_loaders(args: argparse.Namespace, device: torch.device) -> tuple[DataLoader, DataLoader, int]:
    x_train, y_train = _load_emnist_split(args.root, args.split, train=True, limit=args.max_train, seed=args.seed)
    x_valid, y_valid = _load_emnist_split(args.root, args.split, train=False, limit=args.max_test, seed=args.seed)
    classes = int(max(y_train.max().item(), y_valid.max().item()) + 1)
    generator = torch.Generator(device="cpu").manual_seed(0x5A17 + args.seed)
    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    valid_loader = DataLoader(
        TensorDataset(x_valid, y_valid),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    return train_loader, valid_loader, classes


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        total_loss += float(F.cross_entropy(logits, labels, reduction="sum").item())
        total_correct += int((logits.argmax(dim=-1) == labels).sum().item())
        total += int(labels.numel())
    return total_loss / max(1, total), total_correct / max(1, total)


@torch.no_grad()
def _trace_validation(
    model: EmnistGlobalMixedActionClassifier,
    loader: DataLoader,
    device: torch.device,
    limit: int,
) -> list[dict[str, float | int]]:
    pieces: list[Tensor] = []
    seen = 0
    for images, _labels in loader:
        take = min(int(images.shape[0]), max(0, limit - seen))
        if take <= 0:
            break
        pieces.append(images[:take].flatten(1))
        seen += take
    flat = torch.cat(pieces, dim=0).to(device) if pieces else torch.empty(0, model.input_dim, device=device)
    if model.carrier_dim > model.input_dim:
        flat = F.pad(flat, (0, model.carrier_dim - model.input_dim))
    return model.core.trace(flat)


def _trace_aggregates(trace: list[dict[str, float | int]]) -> dict[str, float]:
    def mean_field(field: str) -> float:
        values = [float(row[field]) for row in trace if math.isfinite(float(row[field]))]
        return sum(values) / len(values) if values else math.nan

    minorities = []
    for row in trace:
        q = float(row["q_fraction"])
        s = float(row["s_fraction"])
        if math.isfinite(q) and math.isfinite(s):
            minorities.append(min(q, 1.0 - q, s, 1.0 - s))
    return {
        "q_fraction_mean": mean_field("q_fraction"),
        "s_fraction_mean": mean_field("s_fraction"),
        "minority_fraction_min": min(minorities) if minorities else math.nan,
        "live_vs_pre_reroute_fraction_mean": mean_field("live_vs_pre_reroute_fraction"),
        "actual_reroute_fraction_mean": mean_field("actual_reroute_fraction"),
        "forced_q_sensitivity_fraction_mean": mean_field("forced_q_sensitivity_fraction"),
        "action_rms_mean": mean_field("action_rms"),
        "state_out_rms_max": max(float(row["state_out_rms"]) for row in trace),
        "block_output_rms_max": max(float(row["block_output_rms"]) for row in trace),
        "block_delta_rms_mean": mean_field("block_delta_rms"),
        "block_gain_rms_max": max(float(row["block_gain_rms"]) for row in trace),
        "stack_output_over_input_rms": float(trace[-1]["block_output_over_input_rms"]),
        "upstream_skip_route_change_fraction_mean": mean_field("upstream_skip_route_change_fraction"),
        "mixed_amplitude_mean": mean_field("mixed_amplitude"),
    }


def _source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _build_model(args: argparse.Namespace, classes: int, device: torch.device) -> EmnistGlobalMixedActionClassifier:
    model = EmnistGlobalMixedActionClassifier(
        input_dim=28 * 28,
        carrier_dim=args.carrier_dim,
        classes=classes,
        arm=args.arm,
        seed=args.seed,
        core_depth=args.core_depth,
        rounds=args.rounds,
        tau=args.tau,
        action_limit=args.action_limit,
        dissipation_span=args.dissipation_span,
        action_init=args.action_init,
        tables=args.tables,
        comparisons=args.comparisons,
    )
    return model.to(device)


def _run_overfit_probe(
    args: argparse.Namespace,
    model: EmnistGlobalMixedActionClassifier,
    train_loader: DataLoader,
    device: torch.device,
    out_dir: Path,
) -> None:
    batches = []
    for images, labels in train_loader:
        batches.append((images.to(device, non_blocking=True), labels.to(device, non_blocking=True)))
        if len(batches) == 2:
            break
    if len(batches) != 2:
        raise RuntimeError("overfit probe requires at least two batches")
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=args.lr, weight_decay=0.0)
    model.train()
    with torch.no_grad():
        initial_loss = sum(float(F.cross_entropy(model(x), y).item()) for x, y in batches) / len(batches)
    action_grad_max = 0.0
    threshold_grad_max = 0.0
    losses: list[float] = []
    for step in range(args.overfit_steps):
        images, labels = batches[step % len(batches)]
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(images), labels)
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite overfit loss at step {step}")
        loss.backward()
        action_grad_max = max(action_grad_max, _parameter_grad_norm(model.core.action_parameters()))
        threshold_grad_max = max(threshold_grad_max, _parameter_grad_norm(model.core.threshold_parameters()))
        torch.nn.utils.clip_grad_norm_((parameter for parameter in model.parameters() if parameter.requires_grad), args.grad_clip)
        optimizer.step()
        losses.append(float(loss.item()))
    model.eval()
    with torch.no_grad():
        final_loss = sum(float(F.cross_entropy(model(x), y).item()) for x, y in batches) / len(batches)
    result = {
        "status": "complete",
        "arm": args.arm,
        "seed": args.seed,
        "steps": args.overfit_steps,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_ratio": final_loss / initial_loss,
        "action_grad_max": action_grad_max,
        "threshold_grad_max": threshold_grad_max,
        "core_initial_hash": model.core.initial_hash(),
        "core_block_initial_hashes": model.core.block_initial_hashes(),
        "head_initial_hash": model.head_initial_hash(),
        "source_sha256": _source_sha256(),
    }
    _write_json_atomic(out_dir / "smoke.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)


def run(args: argparse.Namespace) -> dict[str, object]:
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    complete_path = out_dir / "summary.json"
    if complete_path.exists() and not args.overwrite:
        previous = json.loads(complete_path.read_text())
        if previous.get("status") == "complete":
            raise FileExistsError(f"complete result already exists at {complete_path}; use --overwrite explicitly")

    train_loader, valid_loader, classes = _make_loaders(args, device)
    model = _build_model(args, classes, device)
    ledger = model.core.ledger()
    if ledger.receptive_field != args.carrier_dim:
        raise RuntimeError(f"configuration is not globally connected: receptive field {ledger.receptive_field}")
    core_params = sum(parameter.numel() for parameter in model.core.parameters())
    head_params = sum(parameter.numel() for parameter in model.readout.parameters())
    if core_params != ledger.trainable_parameters:
        raise RuntimeError(f"core parameter ledger mismatch: actual={core_params}, expected={ledger.trainable_parameters}")
    if any(isinstance(module, nn.Linear) for module in model.core.modules()):
        raise RuntimeError("global mixed-action core must not contain nn.Linear")
    if any(parameter.ndim == 2 and tuple(parameter.shape) == (args.carrier_dim, args.carrier_dim) for parameter in model.core.parameters()):
        raise RuntimeError("global mixed-action core contains a forbidden dense carrier matrix")

    config = vars(args).copy()
    config["root"] = str(args.root)
    config["out_dir"] = str(out_dir)
    config["device_resolved"] = str(device)
    config["classes"] = classes
    config["train_examples"] = len(train_loader.dataset)
    config["valid_examples"] = len(valid_loader.dataset)
    config["core_parameters"] = core_params
    config["head_parameters"] = head_params
    config["total_parameters"] = core_params + head_params
    config["trainable_parameters"] = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    config["core_initial_hash"] = model.core.initial_hash()
    config["core_block_initial_hashes"] = model.core.block_initial_hashes()
    config["head_initial_hash"] = model.head_initial_hash()
    config["source_sha256"] = _source_sha256()
    config["ledger"] = asdict(ledger)
    config["command"] = sys.argv
    _write_json_atomic(out_dir / "config.json", config)

    if args.overfit_steps > 0:
        _run_overfit_probe(args, model, train_loader, device, out_dir)
        return {"status": "complete", "kind": "overfit_probe"}

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    metric_rows: list[dict[str, object]] = []
    trace_rows: list[dict[str, object]] = []
    total_start = time.perf_counter()

    valid_loss, valid_acc = _evaluate(model, valid_loader, device)
    trace = _trace_validation(model, valid_loader, device, args.trace_examples)
    trace_agg = _trace_aggregates(trace)
    metric_rows.append(
        {
            "arm": args.arm,
            "seed": args.seed,
            "epoch": 0,
            "train_loss": math.nan,
            "train_acc": math.nan,
            "valid_loss": valid_loss,
            "valid_acc": valid_acc,
            "epoch_seconds": 0.0,
            "train_examples_per_second": math.nan,
            "core_grad_norm_mean": 0.0,
            "core_block_grad_norm_min": 0.0,
            "core_block_grad_norm_max": 0.0,
            "head_grad_norm_mean": 0.0,
            **trace_agg,
            **model.core.displacement(),
        }
    )
    trace_rows.extend({"arm": args.arm, "seed": args.seed, "epoch": 0, **row} for row in trace)
    _write_csv_atomic(out_dir / "metrics.csv", metric_rows)
    _write_csv_atomic(out_dir / "round_trace.csv", trace_rows)
    print(
        f"arm={args.arm} seed={args.seed} epoch=0 valid_loss={valid_loss:.6f} valid_acc={valid_acc:.6f} "
        f"params={core_params + head_params} core={core_params} head={head_params}",
        flush=True,
    )

    nonfinite_batches = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        _sync(device)
        epoch_start = time.perf_counter()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0
        core_grad_sum = 0.0
        block_grad_sums = [0.0 for _ in model.core.blocks]
        head_grad_sum = 0.0
        gradient_steps = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                nonfinite_batches += 1
                raise RuntimeError(f"nonfinite loss at epoch={epoch}, batch={gradient_steps}")
            loss.backward()
            core_grad = _parameter_grad_norm(model.core.parameters())
            block_grads = model.core.block_grad_norms()
            head_grad = _parameter_grad_norm(model.readout.parameters())
            if not math.isfinite(core_grad) or not math.isfinite(head_grad) or not all(
                math.isfinite(value) for value in block_grads
            ):
                raise RuntimeError(f"nonfinite gradient at epoch={epoch}, batch={gradient_steps}")
            core_grad_sum += core_grad
            for block_index, value in enumerate(block_grads):
                block_grad_sums[block_index] += value
            head_grad_sum += head_grad
            gradient_steps += 1
            torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
            optimizer.step()
            batch = int(labels.numel())
            train_loss_sum += float(loss.item()) * batch
            train_correct += int((logits.argmax(dim=-1) == labels).sum().item())
            train_total += batch
        _sync(device)
        epoch_seconds = time.perf_counter() - epoch_start
        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total
        valid_loss, valid_acc = _evaluate(model, valid_loader, device)
        trace = _trace_validation(model, valid_loader, device, args.trace_examples)
        trace_agg = _trace_aggregates(trace)
        block_grad_means = [value / max(1, gradient_steps) for value in block_grad_sums]
        row = {
            "arm": args.arm,
            "seed": args.seed,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "valid_loss": valid_loss,
            "valid_acc": valid_acc,
            "epoch_seconds": epoch_seconds,
            "train_examples_per_second": train_total / epoch_seconds,
            "core_grad_norm_mean": core_grad_sum / max(1, gradient_steps),
            "core_block_grad_norm_min": min(block_grad_means),
            "core_block_grad_norm_max": max(block_grad_means),
            "head_grad_norm_mean": head_grad_sum / max(1, gradient_steps),
            **trace_agg,
            **model.core.displacement(),
        }
        metric_rows.append(row)
        trace_rows.extend({"arm": args.arm, "seed": args.seed, "epoch": epoch, **trace_row} for trace_row in trace)
        _write_csv_atomic(out_dir / "metrics.csv", metric_rows)
        _write_csv_atomic(out_dir / "round_trace.csv", trace_rows)
        print(
            f"arm={args.arm} seed={args.seed} epoch={epoch} train_loss={train_loss:.6f} train_acc={train_acc:.6f} "
            f"valid_loss={valid_loss:.6f} valid_acc={valid_acc:.6f} reroute={trace_agg['actual_reroute_fraction_mean']:.6f} "
            f"stack_rms={trace_agg['stack_output_over_input_rms']:.3f} "
            f"block_grad_min={min(block_grad_means):.3e} samples_s={train_total / epoch_seconds:.1f}",
            flush=True,
        )

    torch.save(model.state_dict(), out_dir / "final_state.pt")
    summary: dict[str, object] = {
        "status": "complete",
        "arm": args.arm,
        "seed": args.seed,
        "epochs": args.epochs,
        "final_train_loss": metric_rows[-1]["train_loss"],
        "final_train_acc": metric_rows[-1]["train_acc"],
        "final_valid_loss": metric_rows[-1]["valid_loss"],
        "final_valid_acc": metric_rows[-1]["valid_acc"],
        "elapsed_seconds": time.perf_counter() - total_start,
        "nonfinite_batches": nonfinite_batches,
        "core_parameters": core_params,
        "head_parameters": head_params,
        "total_parameters": core_params + head_params,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "core_initial_hash": model.core.initial_hash(),
        "core_block_initial_hashes": model.core.block_initial_hashes(),
        "head_initial_hash": model.head_initial_hash(),
        "source_sha256": _source_sha256(),
        "ledger": asdict(ledger),
        "final_trace": trace_agg,
        "final_displacement": model.core.displacement(),
        "final_block_displacements": model.core.block_displacements(),
    }
    _write_json_atomic(out_dir / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("readout_only", "pre_shear_route", "live_reroute"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", default="balanced")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--carrier-dim", type=int, default=1024)
    parser.add_argument("--core-depth", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--action-limit", type=float, default=0.25)
    parser.add_argument("--dissipation-span", type=float, default=0.25)
    parser.add_argument("--action-init", type=float, default=0.02)
    parser.add_argument("--tables", type=int, default=64)
    parser.add_argument("--comparisons", type=int, default=6)
    parser.add_argument("--trace-examples", type=int, default=4096)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-test", type=int, default=0)
    parser.add_argument("--overfit-steps", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.epochs < 0 or args.batch_size < 1 or args.workers < 0:
        parser.error("epochs, batch-size, and workers must be nonnegative/positive")
    if args.overfit_steps < 0:
        parser.error("overfit-steps must be nonnegative")
    if args.core_depth < 1:
        parser.error("core-depth must be positive")
    if args.max_train < 0 or args.max_test < 0:
        parser.error("max-train and max-test must be nonnegative")
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
