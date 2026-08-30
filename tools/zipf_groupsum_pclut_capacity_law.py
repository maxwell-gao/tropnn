from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from tropnn.layers.accumulation import IndependentGroupSums, SumPyramid
from tropnn.layers.hard_lookup import gather_lookup_rows
from tropnn.layers.pairwise import PairwiseLUT, PairwiseRoute
from tropnn.tools.zipf_addressing_capacity_law import (
    DenseTiedRecovery,
    _apply_superposition_weight_decay,
    sample_liu_gore_batch,
    zipf_probabilities,
)

RouteKind = Literal["leaf_only", "pyramid_unsigned", "pyramid_signed", "independent_groups"]
PyramidAnchorPolicy = Literal[
    "node_uniform",
    "level_uniform",
    "level_biased",
    "mixed",
    "same_level_disjoint",
]
FormalArm = Literal[
    "dense",
    "leaf_sum",
    "pyramid_unsigned_sum",
    "pyramid_signed_sum",
    "pyramid_signed_median",
    "independent_group_sum",
    "independent_group_median",
]


def _random_unequal_pairs(width: int, count: int, generator: torch.Generator) -> Tensor:
    left = torch.randint(0, width, (count,), generator=generator)
    right = torch.randint(0, width - 1, (count,), generator=generator)
    right = right + (right >= left).long()
    return torch.stack((left, right), dim=-1)


def make_pyramid_anchors(
    n_features: int,
    tables: int,
    comparisons: int,
    *,
    policy: PyramidAnchorPolicy | Literal["leaf_only"],
    seed: int,
    group_size: int | None = None,
) -> Tensor:
    """Build deterministic explicit anchors over a leaves-first sum pyramid."""

    n_features = int(n_features)
    tables = int(tables)
    comparisons = int(comparisons)
    if n_features < 2 or n_features & (n_features - 1):
        raise ValueError("n_features must be a power of two and at least two")
    if tables < 1 or comparisons < 1:
        raise ValueError("tables and comparisons must be positive")
    count = tables * comparisons
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    if policy == "leaf_only":
        return _random_unequal_pairs(n_features, count, generator).reshape(tables, comparisons, 2)
    if policy == "node_uniform":
        return _random_unequal_pairs(2 * n_features - 1, count, generator).reshape(tables, comparisons, 2)
    if policy == "same_level_disjoint":
        if group_size is None or group_size < 1 or group_size & (group_size - 1):
            raise ValueError("same_level_disjoint requires a positive power-of-two group_size")
        if n_features % group_size:
            raise ValueError("group_size must divide n_features")
        node_count = n_features // group_size
        if node_count < 2 * comparisons:
            raise ValueError(
                f"same_level_disjoint needs at least {2 * comparisons} nodes, got {node_count} for n_features={n_features}, group_size={group_size}"
            )
        level = int(math.log2(group_size))
        base = sum(n_features >> prior for prior in range(level))
        anchors = torch.empty(tables, comparisons, 2, dtype=torch.long)
        for table in range(tables):
            selected = torch.randperm(node_count, generator=generator)[: 2 * comparisons] + base
            anchors[table] = selected.reshape(comparisons, 2)
        return anchors
    if policy not in {"level_uniform", "level_biased", "mixed"}:
        raise ValueError(f"unsupported pyramid anchor policy {policy!r}")

    depth = int(math.log2(n_features))
    level_sizes = [n_features >> level for level in range(depth)]  # exclude the one-node root
    level_offsets: list[int] = []
    offset = 0
    for size in [*level_sizes, 1]:
        level_offsets.append(offset)
        offset += size
    if policy == "level_biased":
        weights = torch.tensor([2.0**level for level in range(depth)], dtype=torch.float64)
        levels = torch.multinomial(weights, count, replacement=True, generator=generator)
    elif policy == "mixed":
        levels = torch.randint(0, depth, (count,), generator=generator)
        levels[::2] = 0
    else:
        levels = torch.randint(0, depth, (count,), generator=generator)

    pairs = torch.empty(count, 2, dtype=torch.long)
    for index, level_tensor in enumerate(levels):
        level = int(level_tensor)
        size = level_sizes[level]
        left = int(torch.randint(0, size, (), generator=generator))
        right = int(torch.randint(0, size - 1, (), generator=generator))
        right += int(right >= left)
        base = level_offsets[level]
        pairs[index] = torch.tensor((base + left, base + right))
    return pairs.reshape(tables, comparisons, 2)


class PyramidPairwiseStage(nn.Module):
    """An experiment wrapper that feeds an explicit feature map to PairwiseLUT."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        tables: int,
        comparisons: int,
        route_kind: RouteKind,
        anchor_policy: PyramidAnchorPolicy,
        anchor_group_size: int | None = None,
        seed: int,
        backend: str,
        aggregation: Literal["sum", "median"] = "sum",
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.route_kind = route_kind
        self.anchor_policy = "leaf_only" if route_kind == "leaf_only" else anchor_policy
        self.anchor_group_size = anchor_group_size
        self.aggregation = aggregation
        if aggregation == "median" and backend != "torch":
            raise ValueError("median aggregation uses the isolated Torch reference path")
        self.pyramid: SumPyramid | None = None
        self.group_projection: IndependentGroupSums | None = None
        if route_kind == "independent_groups":
            if anchor_group_size is None:
                raise ValueError("independent_groups requires anchor_group_size")
            self.group_projection = IndependentGroupSums(
                input_dim,
                tables * comparisons,
                group_size=anchor_group_size,
                seed=seed + 20_000,
            )
            feature_dim = self.group_projection.output_dim
            anchors = torch.arange(feature_dim, dtype=torch.long).reshape(tables, comparisons, 2)
        else:
            self.pyramid = None if route_kind == "leaf_only" else SumPyramid(input_dim, signed=route_kind == "pyramid_signed", seed=seed + 20_000)
            feature_dim = input_dim if self.pyramid is None else self.pyramid.output_dim
            anchors = make_pyramid_anchors(
                input_dim,
                tables,
                comparisons,
                policy=self.anchor_policy,
                seed=seed,
                group_size=anchor_group_size,
            )
        self.lut = PairwiseLUT(
            feature_dim,
            output_dim,
            tables=tables,
            comparisons=comparisons,
            backend=backend,
            seed=seed,
            anchors=anchors,
            lut_init_std=0.02,
            lut_dtype="fp32",
        )

    def features(self, x: Tensor) -> Tensor:
        if self.group_projection is not None:
            return self.group_projection(x)
        return x if self.pyramid is None else self.pyramid(x)

    def route(self, x: Tensor) -> PairwiseRoute:
        return self.lut.route(self.features(x))

    def forward(self, x: Tensor) -> Tensor:
        features = self.features(x)
        if self.aggregation == "sum":
            return self.lut(features).squeeze(1)
        route = self.lut.route(features)
        rows = self.lut.payload_table(dtype=features.dtype, device=features.device)
        selected = gather_lookup_rows(rows, route.indices)
        kth = (self.lut.tables + 1) // 2
        output = selected.kthvalue(kth, dim=-2).values
        if self.training and (features.requires_grad or self.lut.thresholds.requires_grad):
            # The hard action and its payload gradient are exact median semantics.
            # Route credit deliberately reuses the canonical per-table local
            # counterfactual rather than introducing another router surrogate.
            output = output + self.lut.ste_correction(route, rows).to(output.dtype)
        return output


class PyramidRecovery(nn.Module):
    def __init__(
        self,
        n_features: int,
        model_dim: int,
        *,
        tables: int,
        comparisons: int,
        route_kind: RouteKind,
        anchor_policy: PyramidAnchorPolicy,
        anchor_group_size: int | None = None,
        decoder_anchor_policy: PyramidAnchorPolicy | None = None,
        decoder_anchor_group_size: int | None = None,
        seed: int,
        backend: str,
        aggregation: Literal["sum", "median"] = "sum",
    ) -> None:
        super().__init__()
        self.encoder = PyramidPairwiseStage(
            n_features,
            model_dim,
            tables=tables,
            comparisons=comparisons,
            route_kind=route_kind,
            anchor_policy=anchor_policy,
            anchor_group_size=anchor_group_size,
            aggregation=aggregation,
            seed=seed + 1,
            backend=backend,
        )
        self.decoder = PyramidPairwiseStage(
            model_dim,
            n_features,
            tables=tables,
            comparisons=comparisons,
            route_kind=route_kind,
            anchor_policy=decoder_anchor_policy or anchor_policy,
            anchor_group_size=decoder_anchor_group_size if decoder_anchor_policy is not None else anchor_group_size,
            aggregation=aggregation,
            seed=seed + 2,
            backend=backend,
        )
        self.bias = nn.Parameter(torch.zeros(n_features))

    def forward_with_hidden(self, x: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.encoder(x)
        return F.relu(self.decoder(hidden) + self.bias), hidden

    def forward(self, x: Tensor) -> Tensor:
        output, _ = self.forward_with_hidden(x)
        return output


@dataclass(frozen=True)
class Stage1Config:
    route_kind: RouteKind
    anchor_policy: PyramidAnchorPolicy
    anchor_group_size: int | None = None
    decoder_anchor_policy: PyramidAnchorPolicy | None = None
    decoder_anchor_group_size: int | None = None
    n_features: int = 1024
    model_dim: int = 32
    tables: int = 32
    comparisons: int = 6
    alpha: float = 1.0
    activation_density: float = 1.0
    seed: int = 0
    batch_size: int = 512
    steps: int = 10_000
    warmup_steps: int = 500
    learning_rate: float = 0.01
    diagnostic_samples: int = 4096
    diagnostic_batch_size: int = 512
    backend: str = "torch"
    device: str = "cuda:0"

    @property
    def run_key(self) -> str:
        if self.route_kind == "independent_groups":
            policy = f"independent-g{self.anchor_group_size}"
        else:
            policy = "leaf" if self.route_kind == "leaf_only" else self.anchor_policy.replace("_", "-")
        if self.anchor_policy == "same_level_disjoint":
            policy += f"-g{self.anchor_group_size}"
        return f"stage1-{self.route_kind.replace('_', '-')}-{policy}-d{self.model_dim}-t{self.tables}-c{self.comparisons}-s{self.seed}"


@dataclass(frozen=True)
class FormalConfig:
    arm: FormalArm
    n_features: int
    model_dim: int
    alpha: float
    activation_density: float
    tables: int
    comparisons: int
    anchor_policy: PyramidAnchorPolicy
    anchor_group_size: int | None
    backend: str
    weight_decay: float
    learning_rate: float
    batch_size: int
    steps: int
    warmup_steps: int
    eval_samples: int
    eval_batch_size: int
    seed: int
    device: str

    @property
    def run_key(self) -> str:
        density = str(self.activation_density).replace(".", "p")
        alpha = str(self.alpha).replace(".", "p")
        decay = str(self.weight_decay).replace("-", "m").replace(".", "p")
        if self.arm == "dense":
            shape = f"d{self.model_dim}"
        else:
            shape = f"d{self.model_dim}-t{self.tables}-c{self.comparisons}"
            if self.arm.startswith("independent_group_"):
                shape += f"-g{self.anchor_group_size}"
            elif self.anchor_policy == "same_level_disjoint" and self.arm != "leaf_sum":
                shape += f"-g{self.anchor_group_size}"
        return f"{self.arm.replace('_', '-')}-{shape}-a{alpha}-e{density}-wd{decay}-s{self.seed}"


def formal_ledger(config: FormalConfig) -> dict[str, int | str | bool]:
    n = config.n_features
    d = config.model_dim
    if config.arm == "dense":
        learned = n * d + n
        return {
            "deploy_learned_scalars": learned,
            "trainable_scalars": learned,
            "deploy_stored_bytes": 4 * learned,
            "active_model_scalar_reads_unique": n * d + n,
            "active_model_scalar_reads_naive": 2 * n * d + n,
            "active_model_bytes_unique": 4 * (n * d + n),
            "active_model_bytes_naive": 4 * (2 * n * d + n),
            "active_input_coordinate_reads": n,
            "active_comparisons": 0,
            "active_payload_scalar_reads": 0,
            "active_macs": 2 * n * d,
            "recognition_kind": "dense_tied_gemm",
        }
    t = config.tables
    c = config.comparisons
    rows = 1 << c
    payload = t * rows * (n + d)
    thresholds = 2 * t * c
    learned = payload + thresholds + n
    anchors = 8 * t * c
    signed = config.arm in {"pyramid_signed_sum", "pyramid_signed_median"}
    pyramid = config.arm.startswith("pyramid_")
    independent = config.arm.startswith("independent_group_")
    sign_bytes = ((n + 7) // 8 + (d + 7) // 8) if signed else 0
    active_payload = t * (n + d)
    active_thresholds = 2 * t * c
    ledger: dict[str, int | str | bool] = {
        "deploy_learned_scalars": learned,
        "trainable_scalars": learned,
        "deploy_stored_bytes": 4 * learned + anchors + sign_bytes,
        "deploy_selected_pair_index_bytes": anchors,
        "deploy_fixed_sign_bytes": sign_bytes,
        "active_model_scalar_reads_unique": active_payload + active_thresholds,
        "active_model_scalar_reads_naive": active_payload + active_thresholds,
        "active_model_bytes_unique": 4 * (active_payload + active_thresholds),
        "active_model_bytes_naive": 4 * (active_payload + active_thresholds),
        "active_input_coordinate_reads": n + d if pyramid else 4 * t * c,
        "active_pyramid_node_reads": 4 * t * c if pyramid else 0,
        "active_comparisons": 2 * t * c,
        "active_payload_scalar_reads": active_payload,
        "active_additions": t * (n + d),
        "active_pyramid_additions": n + d - 2 if pyramid else 0,
        "active_fixed_sign_flips": n + d if signed else 0,
        "active_macs": 0,
        "rows_per_table": rows,
        "recognition_kind": (
            "independent_sparse_group_sum_pairs"
            if independent
            else "canonical_pair_on_sum_pyramid"
            if pyramid
            else "canonical_fixed_pair_learned_threshold"
        ),
        "aggregation": "median" if config.arm.endswith("_median") else "sum",
        "torch_pyramid_activation_bytes_per_token": 4 * ((2 * n - 1) + (2 * d - 1)) if pyramid else 0,
        "fused_pyramid_hbm_traffic_verified": False,
        "backend": config.backend,
        "table_count_power_of_two": bool(t > 0 and t & (t - 1) == 0),
        "regular_dimension_axes": "N,D,R=2^C; primary T scaling uses powers of two",
    }
    if independent:
        if config.anchor_group_size is None:
            raise ValueError("independent group ledger requires anchor_group_size")
        decoder_group_size = min(config.anchor_group_size, config.model_dim // 2)
        ledger.update(
            {
                "encoder_group_size": config.anchor_group_size,
                "decoder_group_size": decoder_group_size,
                "deploy_fixed_group_index_bytes": 16 * t * c * (config.anchor_group_size + decoder_group_size),
                "active_group_sum_additions": 2 * t * c * (config.anchor_group_size + decoder_group_size - 2),
                "active_group_coordinate_reads_naive": 2 * t * c * (config.anchor_group_size + decoder_group_size),
                "torch_group_feature_bytes_per_token": 16 * t * c,
            }
        )
        ledger["deploy_stored_bytes"] = int(ledger["deploy_stored_bytes"]) + int(ledger["deploy_fixed_group_index_bytes"])
    if config.arm.endswith("_median"):
        ledger["median_selected_payload_values"] = active_payload
        ledger["median_reference_implementation"] = "torch_kthvalue_lower_median"
    return ledger


def _nearest_power_of_two(value: float) -> int:
    return max(1, 1 << max(0, int(round(math.log2(max(1.0, value))))))


def _largest_power_of_two_at_most(value: int) -> int:
    if value < 1:
        raise ValueError("no positive power of two fits the requested bound")
    return 1 << (int(value).bit_length() - 1)


def enumerate_formal_configs(args: argparse.Namespace) -> list[FormalConfig]:
    configs: list[FormalConfig] = []
    n = int(args.n_features)
    dimensions = [int(value) for value in _parse_values(args.model_dims)]
    comparisons_values = [int(value) for value in _parse_values(args.lut_comparisons)]
    seeds = [int(value) for value in _parse_values(args.seeds)]
    dense_decays = [float(value) for value in _parse_values(args.dense_weight_decays)]
    t_sweep = [int(value) for value in _parse_values(args.t_sweep_tables)]
    if n < 2 or n & (n - 1):
        raise ValueError("formal n_features must be a power of two and at least two")
    if not dimensions or any(value < 1 or value & (value - 1) for value in dimensions):
        raise ValueError("formal model_dims must be positive powers of two")
    if not comparisons_values or any(value < 1 or value > 16 for value in comparisons_values):
        raise ValueError("formal lut_comparisons must be in [1,16]")
    if not t_sweep or any(value < 1 or value & (value - 1) for value in t_sweep):
        raise ValueError("formal t_sweep_tables must be positive powers of two")
    if args.t_sweep_dim not in dimensions or args.t_sweep_comparisons not in comparisons_values:
        raise ValueError("formal T-sweep point must lie on the registered D/C axes")
    if args.anchor_group_size is None or args.anchor_group_size < 1 or args.anchor_group_size & (args.anchor_group_size - 1):
        raise ValueError("formal independent-group arms require a power-of-two anchor_group_size")
    if 2 * args.anchor_group_size > n:
        raise ValueError("formal independent-group size must be at most n_features/2")
    if args.anchor_policy == "same_level_disjoint":
        if 2 * max(comparisons_values) * args.anchor_group_size > n:
            raise ValueError("the registered encoder cannot fit disjoint groups at the requested C/group size")
    for seed in seeds:
        for d in dimensions:
            for decay in dense_decays:
                configs.append(
                    FormalConfig(
                        "dense",
                        n,
                        d,
                        args.alpha,
                        args.activation_density,
                        0,
                        0,
                        args.anchor_policy,
                        args.anchor_group_size,
                        "tilelang",
                        decay,
                        args.learning_rate,
                        args.batch_size,
                        args.steps,
                        args.warmup_steps,
                        args.eval_samples,
                        args.eval_batch_size,
                        seed,
                        args.device,
                    )
                )
            for c in comparisons_values:
                rows = 1 << c
                dense_stored_bytes = 4 * (n * d + n)
                dense_unique_reads = n * d + n
                dense_naive_reads = 2 * n * d + n
                per_table_stored_bytes = 4 * rows * (n + d) + 16 * c
                fixed_bias_bytes = 4 * n
                signed_bytes = (n + 7) // 8 + (d + 7) // 8
                active_per_table = n + d + 2 * c
                table_values = {
                    _nearest_power_of_two(d / rows),
                    _nearest_power_of_two(d),
                    _nearest_power_of_two(2 * d),
                    max(1, (dense_stored_bytes - fixed_bias_bytes) // per_table_stored_bytes),
                    max(1, (dense_stored_bytes - fixed_bias_bytes - signed_bytes) // per_table_stored_bytes),
                    max(1, dense_unique_reads // active_per_table),
                    max(1, dense_naive_reads // active_per_table),
                }
                if d == args.t_sweep_dim and c == args.t_sweep_comparisons:
                    table_values.update(t_sweep)
                for t in sorted(table_values):
                    full_arms: tuple[FormalArm, ...] = (
                        "leaf_sum",
                        "pyramid_unsigned_sum",
                        "pyramid_signed_sum",
                        "independent_group_sum",
                    )
                    for arm in full_arms:
                        arm_policy: PyramidAnchorPolicy = "node_uniform" if arm.startswith("independent_group_") else args.anchor_policy
                        configs.append(
                            FormalConfig(
                                arm,
                                n,
                                d,
                                args.alpha,
                                args.activation_density,
                                t,
                                c,
                                arm_policy,
                                args.anchor_group_size,
                                "tilelang",
                                0.0,
                                args.learning_rate,
                                args.batch_size,
                                args.steps,
                                args.warmup_steps,
                                args.eval_samples,
                                args.eval_batch_size,
                                seed,
                                args.device,
                            )
                        )
                    if d == args.t_sweep_dim and c == args.t_sweep_comparisons and t in t_sweep:
                        configs.append(
                            FormalConfig(
                                "independent_group_median",
                                n,
                                d,
                                args.alpha,
                                args.activation_density,
                                t,
                                c,
                                "node_uniform",
                                args.anchor_group_size,
                                "torch",
                                0.0,
                                args.learning_rate,
                                args.batch_size,
                                args.steps,
                                args.warmup_steps,
                                args.eval_samples,
                                args.eval_batch_size,
                                seed,
                                args.device,
                            )
                        )
    unique = {config.run_key: config for config in configs}
    return [unique[key] for key in sorted(unique)]


def _schedule_scale(step: int, config: Stage1Config) -> float:
    if config.warmup_steps > 0 and step < config.warmup_steps:
        return float(step + 1) / float(config.warmup_steps)
    progress = (step - config.warmup_steps) / max(1, config.steps - config.warmup_steps - 1)
    return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))


def _route_health(codes: Tensor, comparisons: int) -> dict[str, object]:
    entropies: list[float] = []
    observed: list[int] = []
    maximum_mass: list[float] = []
    for table in range(codes.shape[1]):
        counts = torch.bincount(codes[:, table], minlength=1 << comparisons).to(torch.float64)
        probabilities = counts[counts > 0] / counts.sum()
        entropies.append(float((-(probabilities * probabilities.log2())).sum()))
        observed.append(int((counts > 0).sum()))
        maximum_mass.append(float(counts.max() / counts.sum()))
    return {
        "entropy_bits_mean": sum(entropies) / len(entropies),
        "entropy_bits_min": min(entropies),
        "entropy_bits_max": max(entropies),
        "observed_rows_mean": sum(observed) / len(observed),
        "observed_rows_min": min(observed),
        "observed_rows_max": max(observed),
        "maximum_cell_mass_mean": sum(maximum_mass) / len(maximum_mass),
    }


@torch.no_grad()
def evaluate_stage1_route_health(
    model: PyramidRecovery,
    probabilities: Tensor,
    *,
    samples: int,
    batch_size: int,
    comparisons: int,
    generator_seed: int,
) -> dict[str, object]:
    model.eval()
    generator = torch.Generator(device=probabilities.device).manual_seed(generator_seed)
    encoder_codes: list[Tensor] = []
    decoder_codes: list[Tensor] = []
    seen = 0
    while seen < samples:
        current = min(batch_size, samples - seen)
        x = sample_liu_gore_batch(probabilities, current, generator=generator)
        _, hidden = model.forward_with_hidden(x)
        encoder_codes.append(model.encoder.route(x).indices.cpu())
        decoder_codes.append(model.decoder.route(hidden).indices.cpu())
        seen += current
    return {
        "samples": samples,
        "encoder": _route_health(torch.cat(encoder_codes), comparisons),
        "decoder": _route_health(torch.cat(decoder_codes), comparisons),
    }


def run_stage1(config: Stage1Config) -> dict[str, object]:
    device = torch.device(config.device)
    torch.manual_seed(config.seed + 1000)
    probabilities = zipf_probabilities(config.n_features, config.alpha, config.activation_density, device=device)
    model = PyramidRecovery(
        config.n_features,
        config.model_dim,
        tables=config.tables,
        comparisons=config.comparisons,
        route_kind=config.route_kind,
        anchor_policy=config.anchor_policy,
        anchor_group_size=config.anchor_group_size,
        decoder_anchor_policy=config.decoder_anchor_policy,
        decoder_anchor_group_size=config.decoder_anchor_group_size,
        seed=config.seed + 1000,
        backend=config.backend,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.0)
    for group in optimizer.param_groups:
        group["initial_lr"] = config.learning_rate
    generator = torch.Generator(device=device).manual_seed(config.seed + 2000)
    loss_history: list[dict[str, float | int]] = []
    model.train()
    started = time.perf_counter()
    for step in range(config.steps):
        scale = _schedule_scale(step, config)
        for group in optimizer.param_groups:
            group["lr"] = float(group["initial_lr"]) * scale
        x = sample_liu_gore_batch(probabilities, config.batch_size, generator=generator)
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(x), x)
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite loss at step {step}: {float(loss)}")
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % max(1, config.steps // 20) == 0 or step + 1 == config.steps:
            loss_history.append({"step": step + 1, "mean_loss": float(loss.detach())})
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - started
    route_health = evaluate_stage1_route_health(
        model,
        probabilities,
        samples=config.diagnostic_samples,
        batch_size=config.diagnostic_batch_size,
        comparisons=config.comparisons,
        generator_seed=60_013,
    )
    return {
        "schema": "zipf-groupsum-pclut-stage1-route-health-v1",
        "complete": True,
        "run_key": config.run_key,
        "config": asdict(config),
        "train_seconds": train_seconds,
        "loss_history": loss_history,
        "route_health": route_health,
        "anchors": {
            "encoder": model.encoder.lut.anchors.cpu().tolist(),
            "decoder": model.decoder.lut.anchors.cpu().tolist(),
        },
        "anchor_group_size": config.anchor_group_size,
        "signs": {
            "encoder": None if model.encoder.pyramid is None else model.encoder.pyramid.signs.cpu().tolist(),
            "decoder": None if model.decoder.pyramid is None else model.decoder.pyramid.signs.cpu().tolist(),
        },
        "independent_groups": {
            "encoder": (None if model.encoder.group_projection is None else model.encoder.group_projection.groups.cpu().tolist()),
            "decoder": (None if model.decoder.group_projection is None else model.decoder.group_projection.groups.cpu().tolist()),
        },
        "environment": {
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        },
    }


def _build_formal_model(config: FormalConfig) -> nn.Module:
    if config.arm == "dense":
        return DenseTiedRecovery(config.n_features, config.model_dim, seed=config.seed + 1000)
    route_kind: RouteKind
    aggregation: Literal["sum", "median"] = "sum"
    if config.arm == "leaf_sum":
        route_kind = "leaf_only"
    elif config.arm == "pyramid_unsigned_sum":
        route_kind = "pyramid_unsigned"
    elif config.arm.startswith("independent_group_"):
        route_kind = "independent_groups"
    else:
        route_kind = "pyramid_signed"
    if config.arm.endswith("_median"):
        aggregation = "median"
    decoder_group_size = config.anchor_group_size
    if route_kind == "independent_groups":
        if config.anchor_group_size is None:
            raise ValueError("independent-group model is missing anchor_group_size")
        decoder_group_size = min(config.anchor_group_size, config.model_dim // 2)
    elif route_kind != "leaf_only" and config.anchor_policy == "same_level_disjoint":
        if config.anchor_group_size is None:
            raise ValueError("same_level_disjoint model is missing anchor_group_size")
        decoder_group_size = min(
            config.anchor_group_size,
            _largest_power_of_two_at_most(config.model_dim // (2 * config.comparisons)),
        )
    return PyramidRecovery(
        config.n_features,
        config.model_dim,
        tables=config.tables,
        comparisons=config.comparisons,
        route_kind=route_kind,
        anchor_policy=config.anchor_policy,
        anchor_group_size=config.anchor_group_size,
        decoder_anchor_policy=config.anchor_policy,
        decoder_anchor_group_size=decoder_group_size,
        seed=config.seed + 1000,
        backend=config.backend,
        aggregation=aggregation,
    )


def _formal_optimizer(model: nn.Module, config: FormalConfig) -> torch.optim.Optimizer:
    if isinstance(model, DenseTiedRecovery):
        return torch.optim.Adam(
            [
                {
                    "params": [model.weight],
                    "superposition_weight_decay": config.weight_decay,
                    "initial_lr": config.learning_rate,
                    "lr": config.learning_rate,
                },
                {
                    "params": [model.bias],
                    "superposition_weight_decay": 0.0,
                    "initial_lr": config.learning_rate,
                    "lr": config.learning_rate,
                },
            ]
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.0)
    for group in optimizer.param_groups:
        group["initial_lr"] = config.learning_rate
    return optimizer


@torch.no_grad()
def evaluate_formal_model(
    model: nn.Module,
    probabilities: Tensor,
    *,
    samples: int,
    batch_size: int,
    comparisons: int,
    generator_seed: int,
) -> dict[str, object]:
    model.eval()
    generator = torch.Generator(device=probabilities.device).manual_seed(generator_seed)
    n = probabilities.numel()
    error_sum = torch.zeros(n, dtype=torch.float64, device=probabilities.device)
    active_error_sum = torch.zeros_like(error_sum)
    inactive_error_sum = torch.zeros_like(error_sum)
    active_count = torch.zeros_like(error_sum)
    target_sum = torch.zeros_like(error_sum)
    target_sq_sum = torch.zeros_like(error_sum)
    output_sum = torch.zeros_like(error_sum)
    output_sq_sum = torch.zeros_like(error_sum)
    encoder_codes: list[Tensor] = []
    decoder_codes: list[Tensor] = []
    seen = 0
    while seen < samples:
        current = min(batch_size, samples - seen)
        x = sample_liu_gore_batch(probabilities, current, generator=generator)
        if isinstance(model, PyramidRecovery):
            output, hidden = model.forward_with_hidden(x)
            if len(encoder_codes) < 8:
                encoder_codes.append(model.encoder.route(x).indices.cpu())
                decoder_codes.append(model.decoder.route(hidden).indices.cpu())
        else:
            output = model(x)
        if not bool(torch.isfinite(output).all()):
            raise FloatingPointError("formal evaluation produced a nonfinite output")
        # Promote before squaring so a large but finite FP32 output cannot turn
        # the audit accumulator into infinity through an FP32 intermediate.
        error = (output.to(torch.float64) - x.to(torch.float64)).square()
        active = x > 0
        error_sum += error.sum(dim=0)
        active_error_sum += (error * active).sum(dim=0)
        inactive_error_sum += (error * ~active).sum(dim=0)
        active_count += active.sum(dim=0)
        target_sum += x.to(torch.float64).sum(dim=0)
        target_sq_sum += x.to(torch.float64).square().sum(dim=0)
        output_sum += output.to(torch.float64).sum(dim=0)
        output_sq_sum += output.to(torch.float64).square().sum(dim=0)
        seen += current
    feature_mse = error_sum / samples
    population_zero_risk = probabilities.to(torch.float64) * (4.0 / 3.0)
    population_constant_risk = population_zero_risk - probabilities.to(torch.float64).square()
    output_second_moment = output_sq_sum / samples
    metrics: dict[str, object] = {
        "samples": samples,
        "total_loss": float(feature_mse.sum()),
        "mean_loss": float(feature_mse.mean()),
        "zero_normalized_loss": float(error_sum.sum() / (samples * population_zero_risk.sum())),
        "constant_normalized_loss": float(error_sum.sum() / (samples * population_constant_risk.sum())),
        "feature_mse": feature_mse.cpu().tolist(),
        "feature_zero_normalized_loss": (feature_mse / population_zero_risk.clamp_min(1e-30)).cpu().tolist(),
        "feature_constant_normalized_loss": (feature_mse / population_constant_risk.clamp_min(1e-30)).cpu().tolist(),
        "active_feature_mse": (active_error_sum / active_count.clamp_min(1.0)).cpu().tolist(),
        "inactive_feature_mse": (inactive_error_sum / (samples - active_count).clamp_min(1.0)).cpu().tolist(),
        "active_count": active_count.cpu().tolist(),
        "target_mean": (target_sum / samples).cpu().tolist(),
        "target_second_moment": (target_sq_sum / samples).cpu().tolist(),
        "output_mean": (output_sum / samples).cpu().tolist(),
        "output_second_moment": output_second_moment.cpu().tolist(),
        "tail_output_nonzero_fraction_1e_12": float((output_second_moment[n // 2 :] > 1e-12).to(torch.float64).mean()),
    }
    if encoder_codes:
        metrics["encoder_route_health"] = _route_health(torch.cat(encoder_codes), comparisons)
        metrics["decoder_route_health"] = _route_health(torch.cat(decoder_codes), comparisons)
    return metrics


def _save_state_exclusive(path: Path, model: nn.Module) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    try:
        torch.save(state, temporary)
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "tensor_count": len(state)}


def train_formal_run(config: FormalConfig, *, checkpoint_path: Path | None = None) -> dict[str, object]:
    device = torch.device(config.device)
    torch.manual_seed(config.seed + 1000)
    probabilities = zipf_probabilities(config.n_features, config.alpha, config.activation_density, device=device)
    model = _build_formal_model(config).to(device)
    optimizer = _formal_optimizer(model, config)
    generator = torch.Generator(device=device).manual_seed(config.seed + 2000)
    loss_history: list[dict[str, float | int]] = []
    divergence: dict[str, object] | None = None
    model.train()
    started = time.perf_counter()
    for step in range(config.steps):
        scale = _schedule_scale(step, config)  # FormalConfig has the same frozen schedule fields.
        for group in optimizer.param_groups:
            group["lr"] = float(group["initial_lr"]) * scale
        x = sample_liu_gore_batch(probabilities, config.batch_size, generator=generator)
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(x), x)
        if not torch.isfinite(loss):
            divergence = {
                "stage": "training_loss",
                "step": step + 1,
                "loss": None,
            }
            break
        loss.backward()
        if isinstance(model, DenseTiedRecovery):
            _apply_superposition_weight_decay(optimizer)
        optimizer.step()
        if step == 0 or (step + 1) % max(1, config.steps // 20) == 0 or step + 1 == config.steps:
            loss_history.append(
                {
                    "step": step + 1,
                    "mean_loss": float(loss.detach()),
                    "learning_rate_max": max(float(group["lr"]) for group in optimizer.param_groups),
                }
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - started
    nonfinite_state = {
        name: int((~torch.isfinite(value)).sum().item())
        for name, value in model.state_dict().items()
        if value.is_floating_point() and not bool(torch.isfinite(value).all())
    }
    if divergence is None and nonfinite_state:
        divergence = {
            "stage": "post_optimizer_state",
            "step": config.steps,
            "nonfinite_tensor_count": len(nonfinite_state),
            "nonfinite_element_count": sum(nonfinite_state.values()),
            "nonfinite_elements_by_tensor": nonfinite_state,
        }
    validation: dict[str, object] | None = None
    test: dict[str, object] | None = None
    if divergence is None:
        try:
            validation = evaluate_formal_model(
                model,
                probabilities,
                samples=config.eval_samples,
                batch_size=config.eval_batch_size,
                comparisons=config.comparisons,
                generator_seed=70_001,
            )
            test = evaluate_formal_model(
                model,
                probabilities,
                samples=config.eval_samples,
                batch_size=config.eval_batch_size,
                comparisons=config.comparisons,
                generator_seed=80_003,
            )
        except FloatingPointError as error:
            divergence = {
                "stage": "evaluation_output",
                "step": config.steps,
                "message": str(error),
            }
    artifact = _save_state_exclusive(checkpoint_path, model) if checkpoint_path is not None and divergence is None else None
    route: dict[str, object] = {}
    if isinstance(model, PyramidRecovery) and divergence is None:
        route = {
            "anchor_policy": config.anchor_policy if config.arm != "leaf_sum" else "leaf_only",
            "encoder_anchor_group_size": model.encoder.anchor_group_size,
            "decoder_anchor_group_size": model.decoder.anchor_group_size,
            "encoder_anchors": model.encoder.lut.anchors.cpu().tolist(),
            "decoder_anchors": model.decoder.lut.anchors.cpu().tolist(),
            "encoder_thresholds": model.encoder.lut.thresholds.detach().cpu().tolist(),
            "decoder_thresholds": model.decoder.lut.thresholds.detach().cpu().tolist(),
            "encoder_signs": None if model.encoder.pyramid is None else model.encoder.pyramid.signs.cpu().tolist(),
            "decoder_signs": None if model.decoder.pyramid is None else model.decoder.pyramid.signs.cpu().tolist(),
            "encoder_independent_group_shape": (
                None if model.encoder.group_projection is None else list(model.encoder.group_projection.groups.shape)
            ),
            "decoder_independent_group_shape": (
                None if model.decoder.group_projection is None else list(model.decoder.group_projection.groups.shape)
            ),
            "anchors_fixed": True,
            "thresholds_learned": True,
        }
    backend_detail = "dense_torch"
    if config.arm != "dense":
        backend_detail = (
            "torch_reference_median_with_canonical_sum_counterfactual_route_credit"
            if config.arm.endswith("_median")
            else (
                "tilelang_pair_route_lookup_with_unfused_torch_independent_group_sums"
                if config.arm.startswith("independent_group_")
                else "tilelang_pair_route_lookup_with_unfused_torch_sum_pyramid"
            )
        )
        if config.backend == "tilelang" and config.model_dim <= 8:
            backend_detail += "_and_exact_torch_short_output_ste"
    return {
        "schema": "zipf-groupsum-pclut-capacity-law-run-v3",
        "complete": True,
        "numerically_valid": divergence is None,
        "divergence": divergence,
        "run_key": config.run_key,
        "config": asdict(config),
        "ledger": formal_ledger(config),
        "actual_trainable_scalars": sum(parameter.numel() for parameter in model.parameters()),
        "train_seconds": train_seconds,
        "loss_history": loss_history,
        "validation": validation,
        "test": test,
        "route": route,
        "checkpoint": artifact,
        "environment": {
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "backend_training_detail": backend_detail,
        },
        "optimizer_contract": {
            "dense": "reference_superposition_adam_with_custom_signed_row_norm_decay",
            "lut": "adamw",
            "minimum_learning_rate_fraction": 0.05,
        },
    }


def run_formal_sweep(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    configs = enumerate_formal_configs(args)
    manifest = {
        "schema": "zipf-groupsum-pclut-capacity-law-manifest-v3",
        "config_count": len(configs),
        "run_keys": [config.run_key for config in configs],
        "command": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        try:
            _write_exclusive(manifest_path, manifest)
        except FileExistsError:
            pass  # A concurrently launched shard won the exclusive manifest race.
    existing = json.loads(manifest_path.read_text())
    if (
        existing["schema"] != manifest["schema"]
        or existing["config_count"] != manifest["config_count"]
        or existing["run_keys"] != manifest["run_keys"]
    ):
        raise RuntimeError("existing formal manifest does not match the frozen matrix")

    include_arms = set(_parse_values(args.include_arms))
    known_arms = {config.arm for config in configs}
    if include_arms - known_arms:
        raise ValueError(f"unknown included arms: {sorted(include_arms - known_arms)}")
    selected = [config for config in configs if config.arm in include_arms]
    if args.run_key_file is not None:
        requested = {line.strip() for line in args.run_key_file.read_text().splitlines() if line.strip() and not line.lstrip().startswith("#")}
        available = {config.run_key for config in selected}
        if requested - available:
            raise ValueError(f"run-key file contains unknown keys: {sorted(requested - available)[:1]}")
        selected = [config for config in selected if config.run_key in requested]
    selected = [config for index, config in enumerate(selected) if index % args.shard_count == args.shard_index]
    t_sweep = {int(value) for value in _parse_values(args.t_sweep_tables)}
    for local_index, config in enumerate(selected, start=1):
        path = output_dir / "runs" / f"{config.run_key}.json"
        checkpoint_path = None
        if (
            config.arm in {"pyramid_signed_sum", "independent_group_sum"}
            and config.model_dim == args.t_sweep_dim
            and config.comparisons == args.t_sweep_comparisons
            and config.tables in t_sweep
        ):
            checkpoint_path = output_dir / "checkpoints" / f"{config.run_key}.pt"
        if path.exists():
            existing = json.loads(path.read_text())
            if existing.get("complete") is True and existing.get("schema") == "zipf-groupsum-pclut-capacity-law-run-v3":
                existing_valid = _run_numerically_valid(existing)
                if not existing_valid and "numerically_valid" not in existing:
                    raise RuntimeError(f"legacy result contains a nonfinite value and must be preserved outside the active run set: {path}")
                if checkpoint_path is not None and existing_valid and not checkpoint_path.exists():
                    raise RuntimeError(f"complete result is missing checkpoint {checkpoint_path}")
                if checkpoint_path is not None and not existing_valid and checkpoint_path.exists():
                    raise RuntimeError(f"diverged result unexpectedly has checkpoint {checkpoint_path}")
                print(f"skip complete {config.run_key}", flush=True)
                continue
            raise RuntimeError(f"refusing to overwrite {path}")
        print(f"[{local_index}/{len(selected)}] start {config.run_key}", flush=True)
        result = train_formal_run(config, checkpoint_path=checkpoint_path)
        _write_exclusive(path, result)
        if result["numerically_valid"]:
            print(
                f"[{local_index}/{len(selected)}] done {config.run_key} "
                f"val={result['validation']['total_loss']:.7g} test={result['test']['total_loss']:.7g} "
                f"seconds={result['train_seconds']:.2f}",
                flush=True,
            )
        else:
            print(
                f"[{local_index}/{len(selected)}] diverged {config.run_key} "
                f"stage={result['divergence']['stage']} seconds={result['train_seconds']:.2f}",
                flush=True,
            )


def _write_exclusive(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _parse_values(raw: str) -> list[str]:
    return [value.strip() for value in raw.split(",") if value.strip()]


def _rank_bins(n_features: int, count: int = 16) -> list[tuple[int, int]]:
    edges = {0, n_features}
    for index in range(1, count):
        edges.add(int(round(math.exp(math.log(n_features) * index / count))))
    ordered = sorted(edges)
    return [(start, end) for start, end in zip(ordered, ordered[1:]) if end > start]


def _bin_loss(run: dict[str, object], split: str, start: int, end: int) -> float:
    return float(sum(run[split]["feature_mse"][start:end]))


def _run_numerically_valid(run: dict[str, object]) -> bool:
    """Treat pre-amendment finite results as valid and explicit divergence as failed."""

    def all_finite(value: object) -> bool:
        if isinstance(value, float):
            return math.isfinite(value)
        if isinstance(value, dict):
            return all(all_finite(item) for item in value.values())
        if isinstance(value, list):
            return all(all_finite(item) for item in value)
        return True

    return bool(run.get("numerically_valid", True)) and all_finite(run)


def summarize_formal(output_dir: Path) -> dict[str, object]:
    run_paths = sorted((output_dir / "runs").glob("*.json"))
    runs = [json.loads(path.read_text()) for path in run_paths]
    if not runs:
        raise RuntimeError(f"no formal runs found under {output_dir / 'runs'}")
    if any(run.get("complete") is not True or run.get("schema") != "zipf-groupsum-pclut-capacity-law-run-v3" for run in runs):
        raise RuntimeError("formal matrix contains an incomplete or invalid result")
    manifest = json.loads((output_dir / "manifest.json").read_text())
    expected_keys = set(manifest["run_keys"])
    actual_keys = {str(run["run_key"]) for run in runs}
    if len(actual_keys) != len(runs) or not actual_keys <= expected_keys:
        raise RuntimeError("formal run keys are duplicated or absent from the frozen manifest")
    matrix_complete = actual_keys == expected_keys
    invalid_runs = sorted(
        (run for run in runs if not _run_numerically_valid(run)),
        key=lambda run: str(run["run_key"]),
    )
    valid_runs = [run for run in runs if _run_numerically_valid(run)]

    dense: dict[tuple[int, int], dict[str, object]] = {}
    for run in valid_runs:
        config = run["config"]
        if config["arm"] != "dense":
            continue
        key = (int(config["seed"]), int(config["model_dim"]))
        if key not in dense or float(run["validation"]["total_loss"]) < float(dense[key]["validation"]["total_loss"]):
            dense[key] = run

    bins = _rank_bins(int(runs[0]["config"]["n_features"]))
    comparisons: list[dict[str, object]] = []
    candidate_arms = sorted({str(run["config"]["arm"]) for run in runs} - {"dense"})
    budget_specs = (
        ("parameter", "deploy_stored_bytes", "deploy_stored_bytes"),
        ("bandwidth_unique", "active_model_bytes_unique", "active_model_bytes_unique"),
        ("bandwidth_naive", "active_model_bytes_naive", "active_model_bytes_naive"),
    )
    for (seed, model_dim), dense_run in sorted(dense.items()):
        n_features = int(dense_run["config"]["n_features"])
        tail_start = n_features // 2
        for arm in candidate_arms:
            candidates = [
                run
                for run in valid_runs
                if run["config"]["arm"] == arm and int(run["config"]["seed"]) == seed and int(run["config"]["model_dim"]) == model_dim
            ]
            if not candidates:
                continue
            for budget_kind, dense_budget_key, candidate_budget_key in budget_specs:
                budget = int(dense_run["ledger"][dense_budget_key])
                eligible = [run for run in candidates if int(run["ledger"][candidate_budget_key]) <= budget]
                if not eligible:
                    continue
                selected = min(eligible, key=lambda run: _bin_loss(run, "validation", tail_start, n_features))
                dense_tail = _bin_loss(dense_run, "test", tail_start, n_features)
                candidate_tail = _bin_loss(selected, "test", tail_start, n_features)
                comparisons_value = int(selected["config"]["comparisons"])
                route_health = selected["validation"].get("encoder_route_health")
                entropy = None if route_health is None else float(route_health["entropy_bits_mean"])
                bin_rows = []
                for start, end in bins:
                    dense_loss = _bin_loss(dense_run, "test", start, end)
                    candidate_loss = _bin_loss(selected, "test", start, end)
                    bin_rows.append(
                        {
                            "start_rank": start + 1,
                            "end_rank": end,
                            "dense_loss": dense_loss,
                            "candidate_loss": candidate_loss,
                            "relative_improvement": (dense_loss - candidate_loss) / max(dense_loss, 1e-30),
                        }
                    )
                comparisons.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "model_dim": model_dim,
                        "budget_kind": budget_kind,
                        "budget_bytes": budget,
                        "dense_run": dense_run["run_key"],
                        "candidate_run": selected["run_key"],
                        "candidate_tables": int(selected["config"]["tables"]),
                        "candidate_comparisons": comparisons_value,
                        "candidate_table_count_power_of_two": bool(selected["ledger"]["table_count_power_of_two"]),
                        "dense_test_loss": float(dense_run["test"]["total_loss"]),
                        "candidate_test_loss": float(selected["test"]["total_loss"]),
                        "dense_tail_loss": dense_tail,
                        "candidate_tail_loss": candidate_tail,
                        "tail_relative_improvement": (dense_tail - candidate_tail) / max(dense_tail, 1e-30),
                        "encoder_entropy_bits": entropy,
                        "encoder_entropy_g1_pass": (entropy is not None and entropy >= 0.5 * comparisons_value),
                        "tail_output_nonzero_fraction": float(selected["test"]["tail_output_nonzero_fraction_1e_12"]),
                        "tail_coverage_g2_pass": float(selected["test"]["tail_output_nonzero_fraction_1e_12"]) >= 0.95,
                        "bins": bin_rows,
                    }
                )

    dimensions: list[dict[str, object]] = []
    grouped: dict[tuple[str, str, int], list[dict[str, object]]] = {}
    for row in comparisons:
        grouped.setdefault((str(row["arm"]), str(row["budget_kind"]), int(row["model_dim"])), []).append(row)
    for (arm, budget_kind, model_dim), rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: int(row["seed"]))
        seeds = [int(row["seed"]) for row in ordered]
        improvements = [float(row["tail_relative_improvement"]) for row in ordered]
        mean_improvement = sum(improvements) / len(improvements)
        dimensions.append(
            {
                "arm": arm,
                "budget_kind": budget_kind,
                "model_dim": model_dim,
                "seeds": seeds,
                "seed_tail_relative_improvements": improvements,
                "mean_tail_relative_improvement": mean_improvement,
                "all_three_seeds_positive": seeds == [0, 1, 2] and all(value > 0.0 for value in improvements),
                "dimension_pass": (seeds == [0, 1, 2] and all(value > 0.0 for value in improvements) and mean_improvement >= 0.05),
                "g1_all_seeds_pass": seeds == [0, 1, 2] and all(bool(row["encoder_entropy_g1_pass"]) for row in ordered),
                "g2_all_seeds_pass": seeds == [0, 1, 2] and all(bool(row["tail_coverage_g2_pass"]) for row in ordered),
            }
        )

    budget_decisions: list[dict[str, object]] = []
    decision_groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in dimensions:
        decision_groups.setdefault((str(row["arm"]), str(row["budget_kind"])), []).append(row)
    for (arm, budget_kind), rows in sorted(decision_groups.items()):
        ordered = sorted(rows, key=lambda row: int(row["model_dim"]))
        adjacent = [
            [int(left["model_dim"]), int(right["model_dim"])]
            for left, right in zip(ordered, ordered[1:])
            if bool(left["dimension_pass"]) and bool(right["dimension_pass"])
        ]
        budget_decisions.append(
            {
                "arm": arm,
                "budget_kind": budget_kind,
                "passing_dimensions": [int(row["model_dim"]) for row in ordered if bool(row["dimension_pass"])],
                "adjacent_passing_dimension_pairs": adjacent,
                "budget_pass": bool(adjacent),
                "g1_passing_dimensions": [int(row["model_dim"]) for row in ordered if bool(row["g1_all_seeds_pass"])],
                "g2_passing_dimensions": [int(row["model_dim"]) for row in ordered if bool(row["g2_all_seeds_pass"])],
            }
        )

    primary = [row for row in budget_decisions if row["arm"] == "independent_group_sum"]
    if not matrix_complete:
        decision = "incomplete"
    elif any(row["budget_pass"] for row in primary):
        decision = "positive_independent_group_tail_crossover"
    else:
        decision = "negative_registered_independent_group_tail_crossover"
    summary = {
        "schema": "zipf-groupsum-pclut-capacity-law-summary-v3",
        "complete": matrix_complete,
        "run_count": len(runs),
        "expected_run_count": len(expected_keys),
        "matrix_complete": matrix_complete,
        "numerically_valid_run_count": len(valid_runs),
        "invalid_run_count": len(invalid_runs),
        "invalid_run_keys": [str(run["run_key"]) for run in invalid_runs],
        "invalid_runs": [
            {
                "run_key": str(run["run_key"]),
                "arm": str(run["config"]["arm"]),
                "model_dim": int(run["config"]["model_dim"]),
                "tables": int(run["config"]["tables"]),
                "comparisons": int(run["config"]["comparisons"]),
                "seed": int(run["config"]["seed"]),
                "divergence": run.get("divergence"),
            }
            for run in invalid_runs
        ],
        "dense_selected_count": len(dense),
        "regular_power_of_two_table_run_count": sum(
            run["config"]["arm"] != "dense" and bool(run["ledger"]["table_count_power_of_two"]) for run in runs
        ),
        "irregular_budget_boundary_run_count": sum(
            run["config"]["arm"] != "dense" and not bool(run["ledger"]["table_count_power_of_two"]) for run in runs
        ),
        "rank_bins": [{"start_rank": start + 1, "end_rank": end} for start, end in bins],
        "comparisons": comparisons,
        "dimension_decisions": dimensions,
        "budget_decisions": budget_decisions,
        "scientific_decision": decision,
    }
    _write_exclusive(output_dir / "summary.json", summary)
    return summary


def summarize_stage1(output_dir: Path) -> dict[str, object]:
    paths = sorted(output_dir.glob("stage1-*.json"))
    runs = [json.loads(path.read_text()) for path in paths]
    expected_policies = {"node_uniform", "level_uniform", "level_biased", "mixed"}
    leaf = [run for run in runs if run["config"]["route_kind"] == "leaf_only"]
    unsigned = [run for run in runs if run["config"]["route_kind"] == "pyramid_unsigned"]
    signed = [run for run in runs if run["config"]["route_kind"] == "pyramid_signed"]
    if (
        len(leaf) != 1
        or {run["config"]["anchor_policy"] for run in unsigned} != expected_policies
        or {run["config"]["anchor_policy"] for run in signed} != expected_policies
    ):
        raise RuntimeError("stage1 summary requires one leaf run and all four unsigned/signed policies")
    if any(run.get("complete") is not True or run.get("schema") != "zipf-groupsum-pclut-stage1-route-health-v1" for run in runs):
        raise RuntimeError("stage1 contains an incomplete or invalid result")

    def entropy(run: dict[str, object]) -> float:
        return float(run["route_health"]["encoder"]["entropy_bits_mean"])

    signed_ordered = sorted(signed, key=lambda run: (-entropy(run), str(run["config"]["anchor_policy"])))
    unsigned_ordered = sorted(unsigned, key=lambda run: (-entropy(run), str(run["config"]["anchor_policy"])))
    selected = signed_ordered[0]
    comparisons = int(selected["config"]["comparisons"])
    threshold = 0.5 * comparisons
    payload = {
        "schema": "zipf-groupsum-pclut-stage1-summary-v1",
        "complete": True,
        "run_count": len(runs),
        "leaf_entropy_bits": entropy(leaf[0]),
        "unsigned_by_policy": {
            str(run["config"]["anchor_policy"]): entropy(run) for run in sorted(unsigned, key=lambda item: item["config"]["anchor_policy"])
        },
        "signed_by_policy": {
            str(run["config"]["anchor_policy"]): entropy(run) for run in sorted(signed, key=lambda item: item["config"]["anchor_policy"])
        },
        "best_unsigned_policy": str(unsigned_ordered[0]["config"]["anchor_policy"]),
        "best_unsigned_entropy_bits": entropy(unsigned_ordered[0]),
        "selected_formal_policy": str(selected["config"]["anchor_policy"]),
        "selected_formal_signed_entropy_bits": entropy(selected),
        "g1_entropy_threshold_bits": threshold,
        "selected_policy_g1_pass": entropy(selected) >= threshold,
        "selection_uses_route_entropy_only": True,
    }
    _write_exclusive(output_dir / "stage1-summary.json", payload)
    return payload


def summarize_stage1_disjoint(output_dir: Path) -> dict[str, object]:
    paths = sorted(output_dir.glob("stage1-pyramid-signed-same-level-disjoint-g*.json"))
    runs = [json.loads(path.read_text()) for path in paths]
    if not runs:
        raise RuntimeError("no same-level-disjoint Stage-1b runs found")
    if any(
        run.get("complete") is not True
        or run.get("schema") != "zipf-groupsum-pclut-stage1-route-health-v1"
        or run["config"]["route_kind"] != "pyramid_signed"
        or run["config"]["anchor_policy"] != "same_level_disjoint"
        for run in runs
    ):
        raise RuntimeError("Stage-1b contains an incomplete or invalid result")

    def entropy(run: dict[str, object]) -> float:
        return float(run["route_health"]["encoder"]["entropy_bits_mean"])

    ordered = sorted(runs, key=lambda run: (-entropy(run), int(run["config"]["anchor_group_size"])))
    selected = ordered[0]
    comparisons = int(selected["config"]["comparisons"])
    threshold = 0.5 * comparisons
    payload = {
        "schema": "zipf-groupsum-pclut-stage1b-disjoint-summary-v1",
        "complete": True,
        "run_count": len(runs),
        "entropy_bits_by_group_size": {
            str(run["config"]["anchor_group_size"]): entropy(run) for run in sorted(runs, key=lambda item: int(item["config"]["anchor_group_size"]))
        },
        "selected_group_size": int(selected["config"]["anchor_group_size"]),
        "selected_entropy_bits": entropy(selected),
        "g1_entropy_threshold_bits": threshold,
        "g1_pass": entropy(selected) >= threshold,
        "selection_uses_encoder_route_entropy_only": True,
        "decoder_control_policy": str(selected["config"]["decoder_anchor_policy"]),
        "claim_boundary": "encoder bits use pairwise leaf-disjoint groups; decoder is a fixed level-biased control",
    }
    _write_exclusive(output_dir / "stage1b-summary.json", payload)
    return payload


def summarize_stage1_independent(output_dir: Path) -> dict[str, object]:
    paths = sorted(output_dir.glob("stage1-*independent-g*-d*.json"))
    runs = [json.loads(path.read_text()) for path in paths]
    if not runs:
        raise RuntimeError("no independent-group Stage-1c runs found")
    if any(
        run.get("complete") is not True
        or run.get("schema") != "zipf-groupsum-pclut-stage1-route-health-v1"
        or run["config"]["route_kind"] != "independent_groups"
        for run in runs
    ):
        raise RuntimeError("Stage-1c contains an incomplete or invalid result")

    def entropy(run: dict[str, object]) -> float:
        return float(run["route_health"]["encoder"]["entropy_bits_mean"])

    ordered = sorted(runs, key=lambda run: (-entropy(run), int(run["config"]["anchor_group_size"])))
    selected = ordered[0]
    threshold = 0.5 * int(selected["config"]["comparisons"])
    payload = {
        "schema": "zipf-groupsum-pclut-stage1c-independent-summary-v1",
        "complete": True,
        "run_count": len(runs),
        "entropy_bits_by_group_size": {
            str(run["config"]["anchor_group_size"]): entropy(run) for run in sorted(runs, key=lambda item: int(item["config"]["anchor_group_size"]))
        },
        "selected_group_size": int(selected["config"]["anchor_group_size"]),
        "selected_entropy_bits": entropy(selected),
        "g1_entropy_threshold_bits": threshold,
        "g1_pass": entropy(selected) >= threshold,
        "selection_uses_encoder_route_entropy_only": True,
        "recognition_cost": "2*T*C independent group reductions plus T*C subtract/compare operations",
    }
    _write_exclusive(output_dir / "stage1c-summary.json", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Zipf group-sum pyramid PC-LUT capacity-law experiment")
    subparsers = parser.add_subparsers(dest="command", required=True)
    stage1 = subparsers.add_parser("stage1")
    stage1.add_argument("--output-dir", type=Path, required=True)
    stage1.add_argument("--route-kinds", default="leaf_only,pyramid_unsigned,pyramid_signed")
    stage1.add_argument("--anchor-policies", default="node_uniform,level_uniform,level_biased,mixed")
    stage1.add_argument("--anchor-group-sizes", default="")
    stage1.add_argument("--n-features", type=int, default=1024)
    stage1.add_argument("--model-dim", type=int, default=32)
    stage1.add_argument("--tables", type=int, default=32)
    stage1.add_argument("--comparisons", type=int, default=6)
    stage1.add_argument("--seed", type=int, default=0)
    stage1.add_argument("--batch-size", type=int, default=512)
    stage1.add_argument("--steps", type=int, default=10_000)
    stage1.add_argument("--warmup-steps", type=int, default=500)
    stage1.add_argument("--learning-rate", type=float, default=0.01)
    stage1.add_argument("--diagnostic-samples", type=int, default=4096)
    stage1.add_argument("--diagnostic-batch-size", type=int, default=512)
    stage1.add_argument("--backend", choices=("torch", "tilelang"), default="torch")
    stage1.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    stage1.add_argument("--shard-index", type=int, default=0)
    stage1.add_argument("--shard-count", type=int, default=1)
    summarize_parser = subparsers.add_parser("summarize-stage1")
    summarize_parser.add_argument("--output-dir", type=Path, required=True)
    summarize_disjoint = subparsers.add_parser("summarize-stage1-disjoint")
    summarize_disjoint.add_argument("--output-dir", type=Path, required=True)
    summarize_independent = subparsers.add_parser("summarize-stage1-independent")
    summarize_independent.add_argument("--output-dir", type=Path, required=True)
    formal = subparsers.add_parser("formal", help="run the frozen v3 quality matrix or one deterministic shard")
    formal.add_argument("--output-dir", type=Path, required=True)
    formal.add_argument(
        "--include-arms",
        default="dense,leaf_sum,pyramid_unsigned_sum,pyramid_signed_sum,pyramid_signed_median",
    )
    formal.add_argument("--run-key-file", type=Path)
    formal.add_argument("--n-features", type=int, default=1024)
    formal.add_argument("--model-dims", default="8,16,32,64,128")
    formal.add_argument("--alpha", type=float, default=1.0)
    formal.add_argument("--activation-density", type=float, default=1.0)
    formal.add_argument("--seeds", default="0,1,2")
    formal.add_argument("--dense-weight-decays", default="-0.2,-0.1,0.0,0.1")
    formal.add_argument("--lut-comparisons", default="2,4,6,8")
    formal.add_argument(
        "--anchor-policy",
        choices=("node_uniform", "level_uniform", "level_biased", "mixed", "same_level_disjoint"),
        default="level_biased",
    )
    formal.add_argument("--anchor-group-size", type=int)
    formal.add_argument("--t-sweep-dim", type=int, default=32)
    formal.add_argument("--t-sweep-comparisons", type=int, default=6)
    formal.add_argument("--t-sweep-tables", default="1,2,4,8,16,32,64,128")
    formal.add_argument("--batch-size", type=int, default=512)
    formal.add_argument("--steps", type=int, default=10_000)
    formal.add_argument("--warmup-steps", type=int, default=500)
    formal.add_argument("--learning-rate", type=float, default=0.01)
    formal.add_argument("--eval-samples", type=int, default=16_384)
    formal.add_argument("--eval-batch-size", type=int, default=512)
    formal.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    formal.add_argument("--shard-index", type=int, default=0)
    formal.add_argument("--shard-count", type=int, default=1)
    summarize_formal_parser = subparsers.add_parser("summarize-formal")
    summarize_formal_parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "summarize-stage1":
        summary = summarize_stage1(args.output_dir)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    if args.command == "summarize-stage1-disjoint":
        summary = summarize_stage1_disjoint(args.output_dir)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    if args.command == "summarize-stage1-independent":
        summary = summarize_stage1_independent(args.output_dir)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    if args.command == "formal":
        if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
            raise ValueError("shard_index must lie in [0, shard_count)")
        run_formal_sweep(args)
        return
    if args.command == "summarize-formal":
        summary = summarize_formal(args.output_dir)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    route_kinds = _parse_values(args.route_kinds)
    anchor_policies = _parse_values(args.anchor_policies)
    unknown_routes = set(route_kinds) - {"leaf_only", "pyramid_unsigned", "pyramid_signed", "independent_groups"}
    unknown_policies = set(anchor_policies) - {
        "node_uniform",
        "level_uniform",
        "level_biased",
        "mixed",
        "same_level_disjoint",
    }
    if unknown_routes or unknown_policies:
        raise ValueError(f"unsupported routes={sorted(unknown_routes)} policies={sorted(unknown_policies)}")
    configs: list[Stage1Config] = []
    group_sizes = [int(value) for value in _parse_values(args.anchor_group_sizes)]
    for route_kind in route_kinds:
        policies = ["node_uniform"] if route_kind in {"leaf_only", "independent_groups"} else anchor_policies
        for anchor_policy in policies:
            selected_group_sizes: list[int | None]
            if anchor_policy == "same_level_disjoint" or route_kind == "independent_groups":
                if not group_sizes:
                    raise ValueError(f"{route_kind}/{anchor_policy} requires --anchor-group-sizes")
                selected_group_sizes = group_sizes
            else:
                selected_group_sizes = [None]
            for group_size in selected_group_sizes:
                configs.append(
                    Stage1Config(
                        route_kind=route_kind,
                        anchor_policy=anchor_policy,
                        anchor_group_size=group_size,
                        decoder_anchor_policy=(
                            "level_biased"
                            if anchor_policy == "same_level_disjoint"
                            else "node_uniform"
                            if route_kind == "independent_groups"
                            else None
                        ),
                        decoder_anchor_group_size=(
                            min(group_size, args.model_dim // 2) if route_kind == "independent_groups" and group_size is not None else None
                        ),
                        n_features=args.n_features,
                        model_dim=args.model_dim,
                        tables=args.tables,
                        comparisons=args.comparisons,
                        seed=args.seed,
                        batch_size=args.batch_size,
                        steps=args.steps,
                        warmup_steps=args.warmup_steps,
                        learning_rate=args.learning_rate,
                        diagnostic_samples=args.diagnostic_samples,
                        diagnostic_batch_size=args.diagnostic_batch_size,
                        backend=args.backend,
                        device=args.device,
                    )
                )
    for index, config in enumerate(configs):
        if index % args.shard_count != args.shard_index:
            continue
        output = args.output_dir / f"{config.run_key}.json"
        if output.exists():
            existing = json.loads(output.read_text())
            if existing.get("complete") is True and existing.get("schema") == "zipf-groupsum-pclut-stage1-route-health-v1":
                print(f"skip complete {config.run_key}", flush=True)
                continue
            raise RuntimeError(f"refusing to overwrite {output}")
        print(f"start {config.run_key}", flush=True)
        result = run_stage1(config)
        _write_exclusive(output, result)
        entropy = result["route_health"]["encoder"]["entropy_bits_mean"]
        print(f"done {config.run_key} encoder_entropy={entropy:.6f} seconds={result['train_seconds']:.2f}", flush=True)


if __name__ == "__main__":
    main()
