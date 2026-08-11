"""Evaluate quantized comparison-gated affine circuits on EMNIST Balanced.

The hidden core is a serial stack of globally connected radix-2 sweeps.  A
pairwise comparison selects a small ternary/dyadic affine instruction; it never
indexes a carrier-width payload.  Three modes share the same graph, thresholds,
storage layout, and PC-LUT classifier head:

``constant``
    Keep only branch-dependent scalar offsets (the PC-LUT/MADDNESS-style
    piecewise-constant corner).
``continuous``
    Use wall-matched affine--ReLU instructions.  Their value jump is exactly
    zero, so this is the quantized Neural-Tropical-Geometry sublanguage.
``free``
    Add two explicit branch-jump coefficients.  This is a discontinuous PWA
    extension, with the same forward graph and stored parameter layout.

The hard route is ``margin > 0``.  Continuous hinge terms use PyTorch's exact
ReLU derivative; only the explicit jump term receives the local tanh route
surrogate.  This is a semantic Torch reference, not a fused speed kernel.
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
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from tropnn.layers import PairwiseLUT, QuantizedComparisonAffineMode, QuantizedComparisonAffineStack
from tropnn.tools.emnist_payload_dtype_sweep import _find_emnist_file, _load_emnist_split

SCHEMA_VERSION = "1"
PROTOCOL_ID = "emnist_qcga_caa_l4_v1"


def _rms(value: Tensor) -> float:
    if value.numel() == 0:
        return 0.0
    return float(value.detach().float().square().mean().sqrt().item())


def _tensor_sha256(items: Iterable[tuple[str, Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in items:
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _source_manifest() -> dict[str, str]:
    package = Path(__file__).resolve().parents[1]
    paths = {
        "backend": package / "backend.py",
        "base": package / "layers" / "base.py",
        "tool": Path(__file__).resolve(),
        "core": package / "layers" / "quantized_comparison_affine.py",
        "layers_init": package / "layers" / "__init__.py",
        "pairwise": package / "layers" / "pairwise.py",
        "root_init": package / "__init__.py",
        "surrogate": package / "layers" / "surrogate.py",
        "loader": package / "tools" / "emnist_payload_dtype_sweep.py",
    }
    hashes = {name: _sha256_file(path) for name, path in paths.items()}
    digest = hashlib.sha256()
    for name in sorted(hashes):
        digest.update(name.encode("ascii"))
        digest.update(hashes[name].encode("ascii"))
    return {
        "sha256": digest.hexdigest(),
        "backend_sha256": hashes["backend"],
        "base_sha256": hashes["base"],
        "tool_sha256": hashes["tool"],
        "layer_sha256": hashes["core"],
        "layers_init_sha256": hashes["layers_init"],
        "pairwise_sha256": hashes["pairwise"],
        "root_init_sha256": hashes["root_init"],
        "surrogate_sha256": hashes["surrogate"],
        "loader_sha256": hashes["loader"],
    }


def _data_manifest(root: Path, split: str) -> tuple[str, dict[str, str]]:
    paths = {
        f"{'train' if train else 'test'}_{kind}": _find_emnist_file(
            root,
            split,
            train=train,
            kind=kind,
        )
        for train in (True, False)
        for kind in ("images", "labels")
    }
    hashes = {name: _sha256_file(path) for name, path in paths.items()}
    digest = hashlib.sha256()
    for name in sorted(hashes):
        digest.update(name.encode("ascii"))
        digest.update(hashes[name].encode("ascii"))
    return digest.hexdigest(), hashes


def _strict_json_value(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _strict_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_strict_json_value(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)


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


def _save_state_atomic(path: Path, state: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def _parameter_grad_norm(parameters: Iterable[nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().square().sum().item())
    return math.sqrt(total)


def _core_gradient_components(core: QuantizedComparisonAffineStack) -> dict[str, float]:
    active_total = 0.0
    masked_total = 0.0
    scale_total = 0.0
    threshold_total = 0.0
    for block in core.blocks:
        if block.coefficient_master.grad is not None:
            mask = block.coefficient_mask.unsqueeze(0).to(block.coefficient_master.grad)
            active_total += float((block.coefficient_master.grad * mask).float().square().sum().item())
            masked_total += float((block.coefficient_master.grad * (1.0 - mask)).float().square().sum().item())
        if block.log2_scale_master.grad is not None:
            scale_total += float(block.log2_scale_master.grad.float().square().sum().item())
        if block.thresholds.grad is not None:
            threshold_total += float(block.thresholds.grad.float().square().sum().item())
    return {
        "active_code_grad_norm": math.sqrt(active_total),
        "masked_code_grad_norm": math.sqrt(masked_total),
        "scale_grad_norm": math.sqrt(scale_total),
        "threshold_grad_norm": math.sqrt(threshold_total),
    }


def _masked_master_displacement_max(core: QuantizedComparisonAffineStack) -> float:
    maximum = 0.0
    for block in core.blocks:
        mask = block.coefficient_mask.unsqueeze(0).to(block.coefficient_master)
        delta = (block.coefficient_master - block.initial_coefficient_master) * (1.0 - mask)
        maximum = max(maximum, float(delta.detach().abs().max().item()))
    return maximum


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class EmnistQuantizedComparisonAffineClassifier(nn.Module):
    """Quantized comparison-affine hidden core plus a common PC-LUT head."""

    def __init__(
        self,
        *,
        input_dim: int,
        carrier_dim: int,
        classes: int,
        mode: QuantizedComparisonAffineMode,
        seed: int,
        core_depth: int,
        rounds: int,
        tau: float,
        ternary_threshold: float,
        initial_scale_exponent: int,
        tables: int,
        comparisons: int,
    ) -> None:
        super().__init__()
        if input_dim > carrier_dim:
            raise ValueError(f"input_dim {input_dim} exceeds carrier_dim {carrier_dim}")
        self.input_dim = int(input_dim)
        self.carrier_dim = int(carrier_dim)
        self.mode = mode
        self.core = QuantizedComparisonAffineStack(
            depth=core_depth,
            carrier_dim=carrier_dim,
            rounds=rounds,
            mode=mode,
            tau=tau,
            ternary_threshold=ternary_threshold,
            initial_scale_exponent=initial_scale_exponent,
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

    def carrier(self, images: Tensor) -> Tensor:
        state = images.flatten(1)
        if state.shape[1] != self.input_dim:
            raise ValueError(f"expected flattened input {self.input_dim}, got {state.shape[1]}")
        if self.carrier_dim > self.input_dim:
            state = F.pad(state, (0, self.carrier_dim - self.input_dim))
        return self.core(state)

    def forward(self, images: Tensor) -> Tensor:
        hidden = self.carrier(images)[:, : self.input_dim]
        return self.readout(hidden).squeeze(1)

    def head_initial_hash(self) -> str:
        return self._head_initial_hash


def _make_loaders(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, int]:
    x_train, y_train = _load_emnist_split(
        args.root,
        args.split,
        train=True,
        limit=args.max_train,
        seed=args.seed,
    )
    x_valid, y_valid = _load_emnist_split(
        args.root,
        args.split,
        train=False,
        limit=args.max_test,
        seed=args.seed,
    )
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
    model: EmnistQuantizedComparisonAffineClassifier,
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
    state = torch.cat(pieces, dim=0).to(device) if pieces else torch.empty(0, model.input_dim, device=device)
    if model.carrier_dim > model.input_dim:
        state = F.pad(state, (0, model.carrier_dim - model.input_dim))
    return model.core.trace(state)


def _trace_aggregates(rows: list[dict[str, float | int]]) -> dict[str, float]:
    def values(field: str) -> list[float]:
        return [float(row[field]) for row in rows if math.isfinite(float(row[field]))]

    def mean(field: str) -> float:
        finite = values(field)
        return sum(finite) / len(finite) if finite else math.nan

    return {
        "q_fraction_mean": mean("q_fraction"),
        "minority_fraction_min": min(values("minority_fraction")),
        "pair_branch_coverage_fraction_mean": mean("pair_branch_coverage_fraction"),
        "pair_branch_coverage_fraction_min": min(values("pair_branch_coverage_fraction")),
        "tie_fraction_max": max(values("tie_fraction")),
        "action_rms_mean": mean("action_rms"),
        "wall_jump_rms_mean": mean("wall_jump_rms"),
        "wall_jump_rms_max": max(values("wall_jump_rms")),
        "forced_branch_delta_rms_mean": mean("forced_branch_delta_rms"),
        "state_out_rms_max": max(values("state_out_rms")),
        "block_gain_rms_max": max(values("block_gain_rms")),
        "stack_output_over_input_rms": float(rows[-1]["block_output_over_input_rms"]),
        "scale_mean": mean("scale"),
        "code_zero_fraction_mean": mean("code_zero_fraction"),
        "code_change_fraction_mean": mean("code_change_fraction"),
        "threshold_rms_mean": mean("threshold_rms"),
    }


@torch.no_grad()
def _health_summary(
    *,
    args: argparse.Namespace,
    model: EmnistQuantizedComparisonAffineClassifier,
    aggregate: dict[str, float],
    run_component_grad_max: dict[str, float],
    nonfinite_batches: int,
    run_stack_rms_ratio_max: float,
) -> dict[str, object]:
    code_values: set[int] = set()
    scale_values: list[float] = []
    for block in model.core.blocks:
        code_values.update(int(value) for value in block.hard_coefficient_codes().unique().tolist())
        _coefficients, scales = block.effective_coefficients()
        scale_values.extend(float(value) for value in scales.tolist())
    displacement = model.core.displacement()
    masked_displacement = _masked_master_displacement_max(model.core)
    checks = {
        "no_nonfinite_batches": nonfinite_batches == 0,
        "active_code_gradient_live": run_component_grad_max["active_code_grad_norm"] > 0.0,
        "scale_gradient_live": run_component_grad_max["scale_grad_norm"] > 0.0,
        "threshold_gradient_live": run_component_grad_max["threshold_grad_norm"] > 0.0,
        "masked_code_gradient_zero": run_component_grad_max["masked_code_grad_norm"] <= 1e-12,
        "masked_code_master_unchanged": masked_displacement <= 1e-12,
        "active_master_moved": displacement["coefficient_master_displacement_rms"] > 1e-4,
        "scale_master_moved": displacement["log2_scale_displacement_rms"] > 1e-4,
        "thresholds_moved": displacement["threshold_displacement_rms"] > 1e-4,
        "branch_minority": aggregate["minority_fraction_min"] >= 0.05,
        # The first block necessarily contains some all-zero padded pairs, so
        # coverage is registered as a mean over the 40 stages rather than a
        # per-stage minimum.
        "pair_branch_coverage": aggregate["pair_branch_coverage_fraction_mean"] >= 0.50,
        "forced_branch_delta_live": aggregate["forced_branch_delta_rms_mean"] > 1e-6,
        "state_rms_bounded": run_stack_rms_ratio_max <= args.max_stack_rms_ratio,
        "ternary_alphabet_valid": code_values <= {-1, 0, 1},
        "dyadic_scales_valid": all(
            value > 0.0 and abs(math.log2(value) - round(math.log2(value))) <= 1e-7
            for value in scale_values
        ),
        "continuous_wall_matched": (
            aggregate["wall_jump_rms_max"] <= 1e-8 if args.mode == "continuous" else True
        ),
        "constant_zero_slope_by_construction": args.mode != "constant"
        or all(
            int(value) == 0
            for block in model.core.blocks
            for value in block.hard_coefficient_codes()[:, :, [0, 1, 3, 4]].flatten().tolist()
        ),
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "run_component_grad_max": run_component_grad_max,
        "masked_master_displacement_max": masked_displacement,
        "hard_code_values": sorted(code_values),
        "dyadic_scale_values": sorted(set(scale_values)),
        "run_stack_rms_ratio_max": run_stack_rms_ratio_max,
    }


def _build_model(
    args: argparse.Namespace,
    classes: int,
    device: torch.device,
) -> EmnistQuantizedComparisonAffineClassifier:
    return EmnistQuantizedComparisonAffineClassifier(
        input_dim=28 * 28,
        carrier_dim=args.carrier_dim,
        classes=classes,
        mode=args.mode,
        seed=args.seed,
        core_depth=args.core_depth,
        rounds=args.rounds,
        tau=args.tau,
        ternary_threshold=args.ternary_threshold,
        initial_scale_exponent=args.initial_scale_exponent,
        tables=args.tables,
        comparisons=args.comparisons,
    ).to(device)


def _run_overfit_probe(
    args: argparse.Namespace,
    model: EmnistQuantizedComparisonAffineClassifier,
    train_loader: DataLoader,
    device: torch.device,
    out_dir: Path,
    source_manifest: dict[str, str],
) -> dict[str, object]:
    batches = []
    for images, labels in train_loader:
        batches.append((images.to(device, non_blocking=True), labels.to(device, non_blocking=True)))
        if len(batches) == 2:
            break
    if len(batches) != 2:
        raise RuntimeError("overfit probe requires at least two batches")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    with torch.no_grad():
        initial_loss = sum(float(F.cross_entropy(model(x), y).item()) for x, y in batches) / 2.0
    coefficient_grad_max = 0.0
    threshold_grad_max = 0.0
    for step in range(args.overfit_steps):
        images, labels = batches[step % 2]
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(images), labels)
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite overfit loss at step {step}")
        loss.backward()
        coefficient_grad_max = max(
            coefficient_grad_max,
            _parameter_grad_norm(model.core.coefficient_parameters()),
        )
        threshold_grad_max = max(
            threshold_grad_max,
            _parameter_grad_norm(model.core.threshold_parameters()),
        )
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
    with torch.no_grad():
        final_loss = sum(float(F.cross_entropy(model(x), y).item()) for x, y in batches) / 2.0
    result: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "kind": "overfit_probe",
        "mode": args.mode,
        "seed": args.seed,
        "steps": args.overfit_steps,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_ratio": final_loss / initial_loss,
        "coefficient_grad_max": coefficient_grad_max,
        "threshold_grad_max": threshold_grad_max,
        "core_initial_hash": model.core.initial_hash(),
        "head_initial_hash": model.head_initial_hash(),
        "source_manifest": source_manifest,
        "health": {
            "pass": final_loss / initial_loss <= 0.9
            and coefficient_grad_max > 0.0
            and threshold_grad_max > 0.0,
            "checks": {
                "loss_decreased_10pct": final_loss / initial_loss <= 0.9,
                "coefficient_gradient_live": coefficient_grad_max > 0.0,
                "threshold_gradient_live": threshold_grad_max > 0.0,
            },
        },
    }
    _write_json_atomic(out_dir / "smoke.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def run(args: argparse.Namespace) -> dict[str, object]:
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "summary.json").exists() and not args.overwrite:
        previous = json.loads((out_dir / "summary.json").read_text())
        if previous.get("status") == "complete":
            raise FileExistsError(f"complete result exists in {out_dir}; use --overwrite explicitly")

    source_manifest = _source_manifest()
    data_manifest_sha256, data_hashes = _data_manifest(args.root, args.split)
    train_loader, valid_loader, classes = _make_loaders(args, device)
    model = _build_model(args, classes, device)
    ledger = model.core.ledger()
    core_parameters = sum(parameter.numel() for parameter in model.core.parameters())
    head_parameters = sum(parameter.numel() for parameter in model.readout.parameters())
    if ledger.receptive_field != args.carrier_dim:
        raise RuntimeError(f"core is not globally connected: receptive_field={ledger.receptive_field}")
    if ledger.stored_parameters != core_parameters:
        raise RuntimeError(
            f"core parameter ledger mismatch: actual={core_parameters}, registered={ledger.stored_parameters}"
        )
    if ledger.full_width_payload_scalars != 0:
        raise RuntimeError("core unexpectedly registered a carrier-width payload")
    if any(isinstance(module, nn.Linear) for module in model.core.modules()):
        raise RuntimeError("core contains a forbidden nn.Linear")
    if any(
        parameter.ndim >= 2 and parameter.shape[-1] == args.carrier_dim
        for parameter in model.core.parameters()
    ):
        raise RuntimeError("core contains a forbidden carrier-width parameter tensor")

    config: dict[str, object] = vars(args).copy()
    config.update(
        {
            "root": str(args.root),
            "out_dir": str(out_dir),
            "device_resolved": str(device),
            "classes": classes,
            "train_examples": len(train_loader.dataset),
            "valid_examples": len(valid_loader.dataset),
            "core_parameters_stored": core_parameters,
            "core_parameters_effective": ledger.effective_parameters,
            "head_parameters": head_parameters,
            "total_parameters_stored": core_parameters + head_parameters,
            "total_parameters_effective": ledger.effective_parameters + head_parameters,
            "core_initial_hash": model.core.initial_hash(),
            "core_block_initial_hashes": model.core.block_initial_hashes(),
            "head_initial_hash": model.head_initial_hash(),
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "source_manifest": source_manifest,
            "data_manifest_sha256": data_manifest_sha256,
            "data_hashes": data_hashes,
            "ledger": asdict(ledger),
            "command": sys.argv,
        }
    )
    _write_json_atomic(out_dir / "config.json", config)

    if args.overfit_steps > 0:
        return _run_overfit_probe(
            args,
            model,
            train_loader,
            device,
            out_dir,
            source_manifest,
        )

    parameters = list(model.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    metric_rows: list[dict[str, object]] = []
    trace_rows: list[dict[str, object]] = []
    total_start = time.perf_counter()
    nonfinite_batches = 0
    run_component_grad_max = {
        "active_code_grad_norm": 0.0,
        "masked_code_grad_norm": 0.0,
        "scale_grad_norm": 0.0,
        "threshold_grad_norm": 0.0,
    }

    valid_loss, valid_acc = _evaluate(model, valid_loader, device)
    trace = _trace_validation(model, valid_loader, device, args.trace_examples)
    aggregate = _trace_aggregates(trace)
    run_stack_rms_ratio_max = aggregate["stack_output_over_input_rms"]
    metric_rows.append(
        {
            "mode": args.mode,
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
            "active_code_grad_norm_mean": 0.0,
            "masked_code_grad_norm_mean": 0.0,
            "scale_grad_norm_mean": 0.0,
            "threshold_grad_norm_mean": 0.0,
            "active_code_grad_norm_max": 0.0,
            "masked_code_grad_norm_max": 0.0,
            "scale_grad_norm_max": 0.0,
            "threshold_grad_norm_max": 0.0,
            **aggregate,
            **model.core.displacement(),
        }
    )
    trace_rows.extend(
        {"mode": args.mode, "seed": args.seed, "epoch": 0, **row}
        for row in trace
    )
    _write_csv_atomic(out_dir / "metrics.csv", metric_rows)
    _write_csv_atomic(out_dir / "round_trace.csv", trace_rows)
    print(
        f"mode={args.mode} seed={args.seed} epoch=0 valid_loss={valid_loss:.6f} "
        f"valid_acc={valid_acc:.6f} core={core_parameters} head={head_parameters}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        _sync(device)
        epoch_start = time.perf_counter()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0
        core_grad_sum = 0.0
        head_grad_sum = 0.0
        block_grad_sums = [0.0 for _ in model.core.blocks]
        component_grad_sums = {
            "active_code_grad_norm": 0.0,
            "masked_code_grad_norm": 0.0,
            "scale_grad_norm": 0.0,
            "threshold_grad_norm": 0.0,
        }
        component_grad_max = dict(component_grad_sums)
        gradient_steps = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                nonfinite_batches += 1
                raise RuntimeError(f"nonfinite loss at epoch={epoch}, step={gradient_steps}")
            loss.backward()
            core_grad = _parameter_grad_norm(model.core.parameters())
            head_grad = _parameter_grad_norm(model.readout.parameters())
            block_grads = model.core.block_grad_norms()
            component_grads = _core_gradient_components(model.core)
            if not math.isfinite(core_grad + head_grad) or not all(
                math.isfinite(value) for value in block_grads
            ):
                raise RuntimeError(f"nonfinite gradient at epoch={epoch}, step={gradient_steps}")
            core_grad_sum += core_grad
            head_grad_sum += head_grad
            for block_index, value in enumerate(block_grads):
                block_grad_sums[block_index] += value
            for name, value in component_grads.items():
                component_grad_sums[name] += value
                component_grad_max[name] = max(component_grad_max[name], value)
                run_component_grad_max[name] = max(run_component_grad_max[name], value)
            gradient_steps += 1
            torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
            optimizer.step()
            batch = int(labels.numel())
            train_loss_sum += float(loss.item()) * batch
            train_correct += int((logits.argmax(dim=-1) == labels).sum().item())
            train_total += batch
        _sync(device)
        epoch_seconds = time.perf_counter() - epoch_start
        valid_loss, valid_acc = _evaluate(model, valid_loader, device)
        trace = _trace_validation(model, valid_loader, device, args.trace_examples)
        aggregate = _trace_aggregates(trace)
        run_stack_rms_ratio_max = max(
            run_stack_rms_ratio_max,
            aggregate["stack_output_over_input_rms"],
        )
        block_grad_means = [value / max(1, gradient_steps) for value in block_grad_sums]
        metric_rows.append(
            {
                "mode": args.mode,
                "seed": args.seed,
                "epoch": epoch,
                "train_loss": train_loss_sum / max(1, train_total),
                "train_acc": train_correct / max(1, train_total),
                "valid_loss": valid_loss,
                "valid_acc": valid_acc,
                "epoch_seconds": epoch_seconds,
                "train_examples_per_second": train_total / epoch_seconds,
                "core_grad_norm_mean": core_grad_sum / max(1, gradient_steps),
                "core_block_grad_norm_min": min(block_grad_means),
                "core_block_grad_norm_max": max(block_grad_means),
                "head_grad_norm_mean": head_grad_sum / max(1, gradient_steps),
                **{
                    f"{name}_mean": value / max(1, gradient_steps)
                    for name, value in component_grad_sums.items()
                },
                **{f"{name}_max": value for name, value in component_grad_max.items()},
                **aggregate,
                **model.core.displacement(),
            }
        )
        trace_rows.extend(
            {"mode": args.mode, "seed": args.seed, "epoch": epoch, **row}
            for row in trace
        )
        _write_csv_atomic(out_dir / "metrics.csv", metric_rows)
        _write_csv_atomic(out_dir / "round_trace.csv", trace_rows)
        print(
            f"mode={args.mode} seed={args.seed} epoch={epoch} "
            f"train_loss={metric_rows[-1]['train_loss']:.6f} "
            f"valid_loss={valid_loss:.6f} valid_acc={valid_acc:.6f} "
            f"minority={aggregate['minority_fraction_min']:.4f} "
            f"jump={aggregate['wall_jump_rms_mean']:.4e} "
            f"stack_rms={aggregate['stack_output_over_input_rms']:.3f} "
            f"samples_s={train_total / epoch_seconds:.1f}",
            flush=True,
        )

    state_path = out_dir / "final_state.pt"
    _save_state_atomic(state_path, model.state_dict())
    health = _health_summary(
        args=args,
        model=model,
        aggregate=aggregate,
        run_component_grad_max=run_component_grad_max,
        nonfinite_batches=nonfinite_batches,
        run_stack_rms_ratio_max=run_stack_rms_ratio_max,
    )
    active_fields = {"constant": 2, "continuous": 4, "free": 6}[args.mode]
    parameter_ledger = {
        "core_stored": core_parameters,
        "core_effective": ledger.effective_parameters,
        "thresholds": args.core_depth * args.rounds * (args.carrier_dim // 2),
        "dyadic_scale_masters": args.core_depth * args.rounds,
        "instruction_code_masters_stored": args.core_depth * args.rounds * 2 * 6,
        "instruction_code_masters_effective": args.core_depth * args.rounds * 2 * active_fields,
        "head": head_parameters,
        "total_stored": core_parameters + head_parameters,
        "total_effective": ledger.effective_parameters + head_parameters,
    }
    work_ledger = {
        "core": asdict(ledger),
        "head_comparisons_per_example": args.tables * args.comparisons,
        "head_payload_scalars_read_per_example": args.tables * classes,
        "reference_only": True,
        "fused_kernel_measured": False,
    }
    endpoint = {
        "train_loss": metric_rows[-1]["train_loss"],
        "train_acc": metric_rows[-1]["train_acc"],
        "valid_loss": metric_rows[-1]["valid_loss"],
        "valid_acc": metric_rows[-1]["valid_acc"],
    }
    timing = {
        "elapsed_seconds": time.perf_counter() - total_start,
        "final_epoch_seconds": metric_rows[-1]["epoch_seconds"],
        "final_train_examples_per_second": metric_rows[-1]["train_examples_per_second"],
    }
    artifacts = {
        "config_sha256": _sha256_file(out_dir / "config.json"),
        "metrics_sha256": _sha256_file(out_dir / "metrics.csv"),
        "round_trace_sha256": _sha256_file(out_dir / "round_trace.csv"),
        "final_state_sha256": _sha256_file(state_path),
    }
    summary: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "kind": "formal",
        "mode": args.mode,
        "seed": args.seed,
        "epochs": args.epochs,
        "source_manifest": source_manifest,
        "data_manifest": {"sha256": data_manifest_sha256, **data_hashes},
        "run_config": {
            "carrier_dim": args.carrier_dim,
            "core_depth": args.core_depth,
            "rounds": args.rounds,
            "tau": args.tau,
            "ternary_threshold": args.ternary_threshold,
            "initial_scale_exponent": args.initial_scale_exponent,
            "tables": args.tables,
            "comparisons": args.comparisons,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "max_stack_rms_ratio": args.max_stack_rms_ratio,
        },
        "parameter_ledger": parameter_ledger,
        "work_ledger": work_ledger,
        "init_hashes": {
            "core": model.core.initial_hash(),
            "core_blocks": model.core.block_initial_hashes(),
            "head": model.head_initial_hash(),
        },
        "endpoint": endpoint,
        "health": health,
        "timing": timing,
        "artifacts": artifacts,
        "final_train_loss": metric_rows[-1]["train_loss"],
        "final_train_acc": metric_rows[-1]["train_acc"],
        "final_valid_loss": metric_rows[-1]["valid_loss"],
        "final_valid_acc": metric_rows[-1]["valid_acc"],
        "elapsed_seconds": timing["elapsed_seconds"],
        "nonfinite_batches": nonfinite_batches,
        "core_parameters_stored": core_parameters,
        "core_parameters_effective": ledger.effective_parameters,
        "head_parameters": head_parameters,
        "total_parameters_stored": core_parameters + head_parameters,
        "total_parameters_effective": ledger.effective_parameters + head_parameters,
        "core_initial_hash": model.core.initial_hash(),
        "core_block_initial_hashes": model.core.block_initial_hashes(),
        "head_initial_hash": model.head_initial_hash(),
        "source_manifest_sha256": source_manifest["sha256"],
        "data_manifest_sha256": data_manifest_sha256,
        "ledger": asdict(ledger),
        "final_trace": aggregate,
        "final_displacement": model.core.displacement(),
        "final_block_displacements": model.core.block_displacements(),
    }
    _write_json_atomic(out_dir / "summary.json", summary)
    print(json.dumps(_strict_json_value(summary), sort_keys=True, allow_nan=False), flush=True)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("constant", "continuous", "free"), required=True)
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
    parser.add_argument("--core-depth", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--ternary-threshold", type=float, default=0.5)
    parser.add_argument("--initial-scale-exponent", type=int, default=-4)
    parser.add_argument("--max-stack-rms-ratio", type=float, default=20.0)
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
        parser.error("epochs/workers must be nonnegative and batch-size must be positive")
    if args.core_depth < 1 or args.overfit_steps < 0:
        parser.error("core-depth must be positive and overfit-steps nonnegative")
    if args.max_train < 0 or args.max_test < 0:
        parser.error("max-train and max-test must be nonnegative")
    if args.max_stack_rms_ratio <= 0.0:
        parser.error("max-stack-rms-ratio must be positive")
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
