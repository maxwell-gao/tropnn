from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from tropnn.layers.pairwise import PairwiseLUT

Family = Literal["dense", "lut"]


def zipf_probabilities(n_features: int, alpha: float, activation_density: float, *, device: torch.device) -> Tensor:
    if n_features < 2:
        raise ValueError("n_features must be at least 2")
    if alpha < 0.0:
        raise ValueError("alpha must be nonnegative")
    if activation_density <= 0.0:
        raise ValueError("activation_density must be positive")
    ranks = torch.arange(1, n_features + 1, dtype=torch.float64, device=device)
    probabilities = ranks.pow(-alpha)
    probabilities.mul_(activation_density / probabilities.sum())
    if float(probabilities.max()) > 1.0 + 1e-12:
        maximum_density = float(ranks.pow(-alpha).sum().item())
        raise ValueError(
            f"activation_density={activation_density} makes p_1>1 for N={n_features}, alpha={alpha}; maximum valid density is {maximum_density:.9g}"
        )
    return probabilities.clamp_max(1.0).to(torch.float32)


def sample_liu_gore_batch(probabilities: Tensor, batch_size: int, *, generator: torch.Generator) -> Tensor:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    shape = (batch_size, probabilities.numel())
    active = torch.rand(shape, device=probabilities.device, generator=generator) < probabilities
    amplitude = 2.0 * torch.rand(shape, device=probabilities.device, generator=generator)
    return active.to(amplitude.dtype) * amplitude


class DenseTiedRecovery(nn.Module):
    def __init__(self, n_features: int, model_dim: int, *, seed: int) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.weight = nn.Parameter(torch.randn(n_features, model_dim, generator=generator) / math.sqrt(model_dim))
        self.bias = nn.Parameter(torch.randn(n_features, generator=generator))

    def forward(self, x: Tensor) -> Tensor:
        return F.relu((x @ self.weight) @ self.weight.t() + self.bias)


class RoutedTableRecovery(nn.Module):
    def __init__(
        self,
        n_features: int,
        model_dim: int,
        *,
        tables: int,
        comparisons: int,
        seed: int,
        backend: str,
    ) -> None:
        super().__init__()
        self.encoder = PairwiseLUT(
            n_features,
            model_dim,
            tables=tables,
            comparisons=comparisons,
            seed=seed + 1,
            backend=backend,
            lut_init_std=0.02,
            lut_dtype="fp32",
        )
        self.decoder = PairwiseLUT(
            model_dim,
            n_features,
            tables=tables,
            comparisons=comparisons,
            seed=seed + 2,
            backend=backend,
            lut_init_std=0.02,
            lut_dtype="fp32",
        )
        self.bias = nn.Parameter(torch.zeros(n_features))

    def forward(self, x: Tensor) -> Tensor:
        hidden = self.encoder(x).squeeze(1)
        return F.relu(self.decoder(hidden).squeeze(1) + self.bias)


@dataclass(frozen=True)
class RunConfig:
    family: Family
    n_features: int
    model_dim: int
    alpha: float
    activation_density: float
    tables: int
    comparisons: int
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
        if self.family == "dense":
            shape = f"d{self.model_dim}"
        else:
            shape = f"d{self.model_dim}-t{self.tables}-c{self.comparisons}"
        return f"{self.family}-{shape}-a{alpha}-e{density}-wd{decay}-s{self.seed}"


def budget_ledger(config: RunConfig) -> dict[str, int | float | str]:
    n = config.n_features
    d = config.model_dim
    if config.family == "dense":
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
    r = 1 << c
    payload = t * r * (n + d)
    thresholds = 2 * t * c
    bias = n
    deploy_learned = payload + thresholds + bias
    selected_pair_index_bytes = 2 * (4 * t * c)
    active_payload = t * (n + d)
    active_thresholds = 2 * t * c
    return {
        "deploy_learned_scalars": deploy_learned,
        "trainable_scalars": deploy_learned,
        "deploy_stored_bytes": 4 * deploy_learned + selected_pair_index_bytes,
        "deploy_selected_pair_index_bytes": selected_pair_index_bytes,
        "active_model_scalar_reads_unique": active_payload + active_thresholds,
        "active_model_scalar_reads_naive": active_payload + active_thresholds,
        "active_model_bytes_unique": 4 * (active_payload + active_thresholds),
        "active_model_bytes_naive": 4 * (active_payload + active_thresholds),
        "active_input_coordinate_reads": 4 * t * c,
        "active_comparisons": 2 * t * c,
        "active_payload_scalar_reads": active_payload,
        "active_additions": t * (n + d),
        "active_macs": 0,
        "rows_per_table": r,
        "recognition_kind": "canonical_fixed_pair_learned_threshold",
        "backend": config.backend,
    }


def _build_model(config: RunConfig) -> nn.Module:
    if config.family == "dense":
        return DenseTiedRecovery(config.n_features, config.model_dim, seed=config.seed + 1000)
    return RoutedTableRecovery(
        config.n_features,
        config.model_dim,
        tables=config.tables,
        comparisons=config.comparisons,
        seed=config.seed + 1000,
        backend=config.backend,
    )


def _optimizer(model: nn.Module, config: RunConfig) -> torch.optim.Optimizer:
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
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if name.endswith("weight") or name.endswith("payload"):
            decay.append(parameter)
        else:
            no_decay.append(parameter)
    return torch.optim.AdamW(
        [
            {
                "params": decay,
                "weight_decay": config.weight_decay,
                "initial_lr": config.learning_rate,
                "lr": config.learning_rate,
            },
            {
                "params": no_decay,
                "weight_decay": 0.0,
                "initial_lr": config.learning_rate,
                "lr": config.learning_rate,
            },
        ],
        lr=config.learning_rate,
    )


def _schedule_scale(step: int, config: RunConfig) -> float:
    if config.warmup_steps > 0 and step < config.warmup_steps:
        return float(step + 1) / float(config.warmup_steps)
    progress = (step - config.warmup_steps) / max(1, config.steps - config.warmup_steps - 1)
    return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))


@torch.no_grad()
def _apply_superposition_weight_decay(optimizer: torch.optim.Optimizer) -> None:
    """Match ref/SuperpositionScaling/exp/adamw.py before the Adam update."""

    for group in optimizer.param_groups:
        decay = float(group.get("superposition_weight_decay", 0.0))
        if decay == 0.0:
            continue
        learning_rate = float(group["lr"])
        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            if decay >= 0.0:
                parameter.mul_(1.0 - decay * learning_rate)
            else:
                row_norms = parameter.norm(dim=1, keepdim=True).add_(float(group["eps"]))
                parameter.add_(decay * parameter * (1.0 - row_norms.reciprocal()), alpha=learning_rate)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    probabilities: Tensor,
    *,
    samples: int,
    batch_size: int,
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
        if isinstance(model, RoutedTableRecovery):
            hidden = model.encoder(x).squeeze(1)
            y = F.relu(model.decoder(hidden).squeeze(1) + model.bias)
        else:
            hidden = None
            y = model(x)
        error = (y - x).square().to(torch.float64)
        active = x > 0
        error_sum += error.sum(dim=0)
        active_error_sum += (error * active).sum(dim=0)
        inactive_error_sum += (error * ~active).sum(dim=0)
        active_count += active.sum(dim=0)
        target_sum += x.to(torch.float64).sum(dim=0)
        target_sq_sum += x.to(torch.float64).square().sum(dim=0)
        output_sum += y.to(torch.float64).sum(dim=0)
        output_sq_sum += y.to(torch.float64).square().sum(dim=0)
        if isinstance(model, RoutedTableRecovery) and len(encoder_codes) < 8:
            assert hidden is not None
            encoder_codes.append(model.encoder.route(x).indices.cpu())
            decoder_codes.append(model.decoder.route(hidden).indices.cpu())
        seen += current
    feature_mse = error_sum / samples
    population_zero_risk = probabilities.to(torch.float64) * (4.0 / 3.0)
    population_constant_risk = population_zero_risk - probabilities.to(torch.float64).square()
    metrics: dict[str, object] = {
        "samples": samples,
        "total_loss": float(feature_mse.sum().item()),
        "mean_loss": float(feature_mse.mean().item()),
        "zero_normalized_loss": float(error_sum.sum().item() / (samples * population_zero_risk.sum()).item()),
        "constant_normalized_loss": float(error_sum.sum().item() / (samples * population_constant_risk.sum()).item()),
        "feature_mse": feature_mse.cpu().tolist(),
        "feature_zero_normalized_loss": (feature_mse / population_zero_risk.clamp_min(1e-30)).cpu().tolist(),
        "feature_constant_normalized_loss": (feature_mse / population_constant_risk.clamp_min(1e-30)).cpu().tolist(),
        "active_feature_mse": (active_error_sum / active_count.clamp_min(1.0)).cpu().tolist(),
        "inactive_feature_mse": (inactive_error_sum / (samples - active_count).clamp_min(1.0)).cpu().tolist(),
        "active_count": active_count.cpu().tolist(),
        "target_mean": (target_sum / samples).cpu().tolist(),
        "target_second_moment": (target_sq_sum / samples).cpu().tolist(),
        "output_mean": (output_sum / samples).cpu().tolist(),
        "output_second_moment": (output_sq_sum / samples).cpu().tolist(),
    }
    for name, codes_list in (("encoder", encoder_codes), ("decoder", decoder_codes)):
        if codes_list:
            codes = torch.cat(codes_list, dim=0)
            entropies = []
            observed = []
            for table in range(codes.shape[1]):
                counts = torch.bincount(codes[:, table], minlength=int(codes.max().item()) + 1).double()
                p = counts[counts > 0] / counts.sum()
                entropies.append(float((-(p * p.log2())).sum().item()))
                observed.append(int((counts > 0).sum().item()))
            metrics[f"{name}_route_entropy_bits_mean"] = sum(entropies) / len(entropies)
            metrics[f"{name}_observed_rows_mean"] = sum(observed) / len(observed)
    return metrics


def train_run(config: RunConfig) -> dict[str, object]:
    device = torch.device(config.device)
    torch.manual_seed(config.seed + 1000)
    probabilities = zipf_probabilities(config.n_features, config.alpha, config.activation_density, device=device)
    model = _build_model(config).to(device)
    optimizer = _optimizer(model, config)
    generator = torch.Generator(device=device).manual_seed(config.seed + 2000)
    loss_history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    model.train()
    for step in range(config.steps):
        scale = _schedule_scale(step, config)
        for group in optimizer.param_groups:
            group["lr"] = float(group["initial_lr"]) * scale
        x = sample_liu_gore_batch(probabilities, config.batch_size, generator=generator)
        optimizer.zero_grad(set_to_none=True)
        y = model(x)
        loss = F.mse_loss(y, x)
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite loss at step {step}: {loss.item()}")
        loss.backward()
        if isinstance(model, DenseTiedRecovery):
            _apply_superposition_weight_decay(optimizer)
        optimizer.step()
        if step == 0 or (step + 1) % max(1, config.steps // 20) == 0 or step + 1 == config.steps:
            loss_history.append(
                {
                    "step": step + 1,
                    "mean_loss": float(loss.detach().item()),
                    "learning_rate_max": max(float(group["lr"]) for group in optimizer.param_groups),
                }
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - started
    validation = evaluate_model(
        model,
        probabilities,
        samples=config.eval_samples,
        batch_size=config.eval_batch_size,
        generator_seed=70_001,
    )
    test = evaluate_model(
        model,
        probabilities,
        samples=config.eval_samples,
        batch_size=config.eval_batch_size,
        generator_seed=80_003,
    )
    route: dict[str, object] = {}
    if isinstance(model, RoutedTableRecovery):
        route = {
            "encoder_anchors": model.encoder.anchors.cpu().tolist(),
            "decoder_anchors": model.decoder.anchors.cpu().tolist(),
            "encoder_thresholds": model.encoder.thresholds.detach().cpu().tolist(),
            "decoder_thresholds": model.decoder.thresholds.detach().cpu().tolist(),
            "anchors_fixed": True,
            "thresholds_learned": True,
        }
    return {
        "schema": "zipf-canonical-pclut-capacity-law-run-v2",
        "complete": True,
        "config": asdict(config),
        "run_key": config.run_key,
        "ledger": budget_ledger(config),
        "actual_trainable_scalars": sum(parameter.numel() for parameter in model.parameters()),
        "train_seconds": train_seconds,
        "loss_history": loss_history,
        "validation": validation,
        "test": test,
        "route": route,
        "environment": {
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "backend_training_detail": (
                "tilelang_hard_route_lookup_and_payload_backward_with_exact_torch_short_output_ste"
                if config.family == "lut" and config.backend == "tilelang" and config.model_dim <= 8
                else "native_backend"
            ),
        },
        "optimizer_contract": {
            "dense": "reference_superposition_adam_with_custom_signed_row_norm_decay",
            "lut": "adamw",
            "minimum_learning_rate_fraction": 0.05,
        },
    }


def _parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def _parse_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def _nearest_power_of_two(value: float) -> int:
    return max(1, 1 << max(0, int(round(math.log2(max(1.0, value))))))


def enumerate_configs(args: argparse.Namespace) -> list[RunConfig]:
    configs: list[RunConfig] = []
    for alpha in _parse_floats(args.alphas):
        for density in _parse_floats(args.activation_densities):
            for seed in _parse_ints(args.seeds):
                for model_dim in _parse_ints(args.model_dims):
                    for decay in _parse_floats(args.dense_weight_decays):
                        configs.append(
                            RunConfig(
                                "dense",
                                args.n_features,
                                model_dim,
                                alpha,
                                density,
                                0,
                                0,
                                args.backend,
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
                    for comparisons in _parse_ints(args.lut_comparisons):
                        rows = 1 << comparisons
                        dense_stored_bytes = 4 * (args.n_features * model_dim + args.n_features)
                        lut_fixed_stored_bytes = 4 * args.n_features
                        lut_per_table_stored_bytes = 4 * rows * (args.n_features + model_dim) + 16 * comparisons
                        exact_parameter_tables = max(
                            1,
                            (dense_stored_bytes - lut_fixed_stored_bytes) // lut_per_table_stored_bytes,
                        )
                        dense_unique_scalar_reads = args.n_features * model_dim + args.n_features
                        dense_naive_scalar_reads = 2 * args.n_features * model_dim + args.n_features
                        lut_active_scalars_per_table = args.n_features + model_dim + 2 * comparisons
                        exact_unique_bandwidth_tables = max(1, dense_unique_scalar_reads // lut_active_scalars_per_table)
                        exact_naive_bandwidth_tables = max(1, dense_naive_scalar_reads // lut_active_scalars_per_table)
                        table_values = {
                            _nearest_power_of_two(model_dim / rows),
                            _nearest_power_of_two(model_dim),
                            _nearest_power_of_two(2 * model_dim),
                            exact_parameter_tables,
                            exact_unique_bandwidth_tables,
                            exact_naive_bandwidth_tables,
                        }
                        if model_dim == args.t_sweep_dim and comparisons == args.t_sweep_comparisons:
                            table_values.update(_parse_ints(args.t_sweep_tables))
                        for tables in sorted(table_values):
                            for decay in _parse_floats(args.lut_weight_decays):
                                configs.append(
                                    RunConfig(
                                        "lut",
                                        args.n_features,
                                        model_dim,
                                        alpha,
                                        density,
                                        tables,
                                        comparisons,
                                        args.backend,
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
    unique: dict[str, RunConfig] = {}
    for config in configs:
        unique[config.run_key] = config
    return [unique[key] for key in sorted(unique)]


def _write_exclusive(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    try:
        os.link(temporary, path)
    except FileExistsError:
        temporary.unlink(missing_ok=True)
        return
    temporary.unlink()


def filter_configs_by_run_key_file(configs: list[RunConfig], run_key_file: Path | None) -> list[RunConfig]:
    if run_key_file is not None:
        requested = {line.strip() for line in Path(run_key_file).read_text().splitlines() if line.strip() and not line.lstrip().startswith("#")}
        available = {config.run_key for config in configs}
        unknown = sorted(requested - available)
        if unknown:
            raise ValueError(f"run-key file contains {len(unknown)} unknown keys; first={unknown[0]}")
        return [config for config in configs if config.run_key in requested]
    return configs


def run_sweep(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    configs = filter_configs_by_run_key_file(enumerate_configs(args), getattr(args, "run_key_file", None))
    manifest = {
        "schema": "zipf-canonical-pclut-capacity-law-manifest-v2",
        "config_count": len(configs),
        "shard_count": args.shard_count,
        "command": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "run_keys": [config.run_key for config in configs],
    }
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        _write_exclusive(manifest_path, manifest)
    selected = [config for index, config in enumerate(configs) if index % args.shard_count == args.shard_index]
    for local_index, config in enumerate(selected, start=1):
        path = output_dir / "runs" / f"{config.run_key}.json"
        if path.exists():
            try:
                existing = json.loads(path.read_text())
            except json.JSONDecodeError:
                existing = {}
            if existing.get("complete") is True and existing.get("schema") == "zipf-canonical-pclut-capacity-law-run-v2":
                print(f"skip complete {config.run_key}", flush=True)
                continue
            raise RuntimeError(f"refusing to overwrite incomplete or invalid result {path}")
        print(f"[{local_index}/{len(selected)}] start {config.run_key}", flush=True)
        result = train_run(config)
        _write_exclusive(path, result)
        print(
            f"[{local_index}/{len(selected)}] done {config.run_key} "
            f"val={result['validation']['total_loss']:.7g} test={result['test']['total_loss']:.7g} "
            f"seconds={result['train_seconds']:.2f}",
            flush=True,
        )


def _rank_bins(n_features: int, count: int = 16) -> list[tuple[int, int]]:
    edges = {0, n_features}
    for index in range(1, count):
        edges.add(int(round(math.exp(math.log(n_features) * index / count))))
    ordered = sorted(edges)
    return [(start, end) for start, end in zip(ordered, ordered[1:]) if end > start]


def _bin_loss(run: dict[str, object], split: str, start: int, end: int) -> float:
    values = run[split]["feature_mse"]
    return float(sum(values[start:end]))


def _best_dense(runs: Iterable[dict[str, object]]) -> dict[tuple[float, float, int, int], dict[str, object]]:
    selected: dict[tuple[float, float, int, int], dict[str, object]] = {}
    for run in runs:
        cfg = run["config"]
        if cfg["family"] != "dense":
            continue
        key = (float(cfg["alpha"]), float(cfg["activation_density"]), int(cfg["seed"]), int(cfg["model_dim"]))
        if key not in selected or run["validation"]["total_loss"] < selected[key]["validation"]["total_loss"]:
            selected[key] = run
    return selected


def summarize(output_dir: Path) -> dict[str, object]:
    run_paths = sorted((output_dir / "runs").glob("*.json"))
    runs = [json.loads(path.read_text()) for path in run_paths]
    if not runs:
        raise RuntimeError(f"no runs found under {output_dir / 'runs'}")
    dense = _best_dense(runs)
    bins = _rank_bins(int(runs[0]["config"]["n_features"]))
    comparisons: list[dict[str, object]] = []
    for key, dense_run in sorted(dense.items()):
        alpha, density, seed, model_dim = key
        candidates = [
            run
            for run in runs
            if run["config"]["family"] == "lut"
            and float(run["config"]["alpha"]) == alpha
            and float(run["config"]["activation_density"]) == density
            and int(run["config"]["seed"]) == seed
            and int(run["config"]["model_dim"]) == model_dim
        ]
        for budget_kind, dense_budget_key, lut_budget_key in (
            ("parameter", "deploy_stored_bytes", "deploy_stored_bytes"),
            ("bandwidth_unique", "active_model_bytes_unique", "active_model_bytes_unique"),
            ("bandwidth_naive", "active_model_bytes_naive", "active_model_bytes_naive"),
        ):
            budget = int(dense_run["ledger"][dense_budget_key])
            eligible = [run for run in candidates if int(run["ledger"][lut_budget_key]) <= budget]
            if not eligible:
                continue
            tail_start = int(dense_run["config"]["n_features"]) // 2
            selected_global = min(eligible, key=lambda run: float(run["validation"]["total_loss"]))
            selected_tail = min(eligible, key=lambda run: _bin_loss(run, "validation", tail_start, int(run["config"]["n_features"])))
            for selection, selected in (("global", selected_global), ("tail_half", selected_tail)):
                bin_rows = []
                for start, end in bins:
                    dense_loss = _bin_loss(dense_run, "test", start, end)
                    lut_loss = _bin_loss(selected, "test", start, end)
                    bin_rows.append(
                        {
                            "start_rank": start + 1,
                            "end_rank": end,
                            "dense_loss": dense_loss,
                            "lut_loss": lut_loss,
                            "lut_minus_dense": lut_loss - dense_loss,
                            "relative_improvement": (dense_loss - lut_loss) / max(dense_loss, 1e-30),
                        }
                    )
                dense_tail = _bin_loss(dense_run, "test", tail_start, int(dense_run["config"]["n_features"]))
                lut_tail = _bin_loss(selected, "test", tail_start, int(selected["config"]["n_features"]))
                comparisons.append(
                    {
                        "alpha": alpha,
                        "activation_density": density,
                        "seed": seed,
                        "model_dim": model_dim,
                        "budget_kind": budget_kind,
                        "selection": selection,
                        "budget_bytes": budget,
                        "dense_run": dense_run["run_key"],
                        "lut_run": selected["run_key"],
                        "dense_test_loss": dense_run["test"]["total_loss"],
                        "lut_test_loss": selected["test"]["total_loss"],
                        "tail_start_rank": tail_start + 1,
                        "dense_tail_loss": dense_tail,
                        "lut_tail_loss": lut_tail,
                        "tail_relative_improvement": (dense_tail - lut_tail) / max(dense_tail, 1e-30),
                        "bins": bin_rows,
                    }
                )
    grouped: dict[tuple[float, float, str, int], list[dict[str, object]]] = {}
    for comparison in comparisons:
        if comparison["selection"] != "tail_half":
            continue
        group_key = (
            float(comparison["alpha"]),
            float(comparison["activation_density"]),
            str(comparison["budget_kind"]),
            int(comparison["model_dim"]),
        )
        grouped.setdefault(group_key, []).append(comparison)
    dimension_decisions: list[dict[str, object]] = []
    for (alpha, density, budget_kind, model_dim), group in sorted(grouped.items()):
        improvements = [float(item["tail_relative_improvement"]) for item in sorted(group, key=lambda item: int(item["seed"]))]
        seeds = [int(item["seed"]) for item in sorted(group, key=lambda item: int(item["seed"]))]
        mean_improvement = sum(improvements) / len(improvements)
        dimension_decisions.append(
            {
                "alpha": alpha,
                "activation_density": density,
                "budget_kind": budget_kind,
                "model_dim": model_dim,
                "seeds": seeds,
                "seed_tail_relative_improvements": improvements,
                "mean_tail_relative_improvement": mean_improvement,
                "all_three_seeds_positive": seeds == [0, 1, 2] and all(value > 0.0 for value in improvements),
                "dimension_pass": seeds == [0, 1, 2] and all(value > 0.0 for value in improvements) and mean_improvement >= 0.05,
            }
        )
    budget_decisions: list[dict[str, object]] = []
    decision_groups: dict[tuple[float, float, str], list[dict[str, object]]] = {}
    for decision in dimension_decisions:
        decision_groups.setdefault((float(decision["alpha"]), float(decision["activation_density"]), str(decision["budget_kind"])), []).append(
            decision
        )
    for (alpha, density, budget_kind), decisions in sorted(decision_groups.items()):
        ordered = sorted(decisions, key=lambda item: int(item["model_dim"]))
        adjacent_passes = [
            [int(left["model_dim"]), int(right["model_dim"])]
            for left, right in zip(ordered, ordered[1:])
            if bool(left["dimension_pass"]) and bool(right["dimension_pass"])
        ]
        budget_decisions.append(
            {
                "alpha": alpha,
                "activation_density": density,
                "budget_kind": budget_kind,
                "passing_dimensions": [int(item["model_dim"]) for item in ordered if bool(item["dimension_pass"])],
                "adjacent_passing_dimension_pairs": adjacent_passes,
                "budget_pass": bool(adjacent_passes),
            }
        )
    manifest_path = output_dir / "manifest.json"
    expected_runs = int(json.loads(manifest_path.read_text())["config_count"]) if manifest_path.exists() else len(runs)
    matrix_complete = len(runs) == expected_runs
    primary_budget_pass = {
        str(item["budget_kind"]): bool(item["budget_pass"])
        for item in budget_decisions
        if float(item["alpha"]) == 1.0 and float(item["activation_density"]) == 1.0
    }
    if not matrix_complete:
        scientific_decision = "incomplete"
    elif primary_budget_pass.get("parameter", False):
        scientific_decision = "positive_parameter_tail_crossover"
    elif primary_budget_pass.get("bandwidth_unique", False):
        scientific_decision = "positive_unique_bandwidth_tail_crossover"
    elif primary_budget_pass.get("bandwidth_naive", False):
        scientific_decision = "positive_naive_bandwidth_tail_crossover_only"
    else:
        scientific_decision = "negative_registered_tail_crossover"
    summary = {
        "schema": "zipf-canonical-pclut-capacity-law-summary-v2",
        "run_count": len(runs),
        "expected_run_count": expected_runs,
        "matrix_complete": matrix_complete,
        "dense_selected_count": len(dense),
        "rank_bins": [{"start_rank": start + 1, "end_rank": end} for start, end in bins],
        "comparisons": comparisons,
        "dimension_decisions": dimension_decisions,
        "budget_decisions": budget_decisions,
        "scientific_decision": scientific_decision,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Liu-Gore Zipf superposition versus canonical PairwiseLUT capacity law")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--n-features", type=int, default=1024)
    run.add_argument("--model-dims", default="8,16,32,64,128")
    run.add_argument("--alphas", default="1.0")
    run.add_argument("--activation-densities", default="1.0")
    run.add_argument("--dense-weight-decays", default="-0.2,-0.1,0.0,0.1")
    run.add_argument("--lut-weight-decays", default="0.0")
    run.add_argument("--lut-comparisons", default="2,4,6,8")
    run.add_argument("--backend", choices=("torch", "tilelang", "triton"), default="tilelang")
    run.add_argument("--t-sweep-dim", type=int, default=32)
    run.add_argument("--t-sweep-comparisons", type=int, default=6)
    run.add_argument("--t-sweep-tables", default="1,2,4,8,16,32,64,128")
    run.add_argument("--seeds", default="0,1,2")
    run.add_argument("--batch-size", type=int, default=512)
    run.add_argument("--steps", type=int, default=10000)
    run.add_argument("--warmup-steps", type=int, default=500)
    run.add_argument("--learning-rate", type=float, default=0.01)
    run.add_argument("--eval-samples", type=int, default=16384)
    run.add_argument("--eval-batch-size", type=int, default=512)
    run.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    run.add_argument(
        "--run-key-file",
        type=Path,
        default=None,
        help="optional newline-delimited subset of enumerated run keys (scheduling only)",
    )
    run.add_argument("--shard-index", type=int, default=0)
    run.add_argument("--shard-count", type=int, default=1)
    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run":
        if not 0 <= args.shard_index < args.shard_count:
            raise ValueError("shard-index must be in [0, shard-count)")
        run_sweep(args)
    else:
        summary = summarize(args.output_dir)
        print(f"wrote {args.output_dir / 'summary.json'} with {summary['run_count']} runs")


if __name__ == "__main__":
    main()
