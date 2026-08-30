from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from tropnn.layers.accumulation import WalshButterfly
from tropnn.layers.pairwise import PairwiseLUT, PairwiseRoute
from tropnn.tools.zipf_addressing_capacity_law import sample_liu_gore_batch, zipf_probabilities
from tropnn.tools.zipf_groupsum_pclut_capacity_law import _route_health, _schedule_scale, make_pyramid_anchors

SCHEMA = "zipf-groupsum-walsh-stage1-run-v1"
COMPLETION_SCHEMA = "zipf-groupsum-walsh-stage1-completion-v1"
REGISTERED_SEEDS = (0, 1, 2)


class WalshPairwiseStage(nn.Module):
    """One shared randomized FWHT followed by canonical PC-LUT comparisons."""

    def __init__(self, input_dim: int, output_dim: int, *, tables: int, comparisons: int, seed: int) -> None:
        super().__init__()
        self.transform = WalshButterfly(input_dim, seed=seed + 20_000)
        anchors = make_pyramid_anchors(input_dim, tables, comparisons, policy="leaf_only", seed=seed)
        self.lut = PairwiseLUT(
            input_dim,
            output_dim,
            tables=tables,
            comparisons=comparisons,
            backend="torch",
            seed=seed,
            anchors=anchors,
            lut_init_std=0.02,
            lut_dtype="fp32",
        )

    def route(self, x: Tensor) -> PairwiseRoute:
        return self.lut.route(self.transform(x))

    def forward(self, x: Tensor) -> Tensor:
        return self.lut(self.transform(x)).squeeze(1)


class WalshRecovery(nn.Module):
    def __init__(self, n_features: int, model_dim: int, *, tables: int, comparisons: int, seed: int) -> None:
        super().__init__()
        self.encoder = WalshPairwiseStage(
            n_features,
            model_dim,
            tables=tables,
            comparisons=comparisons,
            seed=seed + 1,
        )
        self.decoder = WalshPairwiseStage(
            model_dim,
            n_features,
            tables=tables,
            comparisons=comparisons,
            seed=seed + 2,
        )
        self.bias = nn.Parameter(torch.zeros(n_features))

    def forward_with_hidden(self, x: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.encoder(x)
        return F.relu(self.decoder(hidden) + self.bias), hidden

    def forward(self, x: Tensor) -> Tensor:
        output, _ = self.forward_with_hidden(x)
        return output


@dataclass(frozen=True)
class Config:
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
    device: str = "cuda:0"

    @property
    def run_key(self) -> str:
        return f"walsh-stage1-d{self.model_dim}-t{self.tables}-c{self.comparisons}-s{self.seed}"

    def validate_formal(self) -> None:
        expected = Config(seed=self.seed, device=self.device)
        if self != expected or self.seed not in REGISTERED_SEEDS:
            raise ValueError("Walsh Stage-1' formal configuration is frozen except for seed and device")


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


def _save_checkpoint_exclusive(path: Path, model: nn.Module) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(model.state_dict(), temporary)
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


@torch.no_grad()
def evaluate_route_health(model: WalshRecovery, probabilities: Tensor, config: Config) -> dict[str, object]:
    model.eval()
    generator = torch.Generator(device=probabilities.device).manual_seed(60_013)
    encoder_codes: list[Tensor] = []
    decoder_codes: list[Tensor] = []
    seen = 0
    while seen < config.diagnostic_samples:
        current = min(config.diagnostic_batch_size, config.diagnostic_samples - seen)
        x = sample_liu_gore_batch(probabilities, current, generator=generator)
        _, hidden = model.forward_with_hidden(x)
        encoder_codes.append(model.encoder.route(x).indices.cpu())
        decoder_codes.append(model.decoder.route(hidden).indices.cpu())
        seen += current
    return {
        "samples": seen,
        "encoder": _route_health(torch.cat(encoder_codes), config.comparisons),
        "decoder": _route_health(torch.cat(decoder_codes), config.comparisons),
    }


def run(config: Config, *, checkpoint_path: Path | None = None) -> dict[str, object]:
    config.validate_formal()
    device = torch.device(config.device)
    torch.manual_seed(config.seed + 1000)
    probabilities = zipf_probabilities(config.n_features, config.alpha, config.activation_density, device=device)
    model = WalshRecovery(
        config.n_features,
        config.model_dim,
        tables=config.tables,
        comparisons=config.comparisons,
        seed=config.seed + 1000,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.0)
    for group in optimizer.param_groups:
        group["initial_lr"] = config.learning_rate
    generator = torch.Generator(device=device).manual_seed(config.seed + 2000)
    model.train()
    loss_history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for step in range(config.steps):
        scale = _schedule_scale(step, config)  # type: ignore[arg-type]
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
    route_health = evaluate_route_health(model, probabilities, config)
    checkpoint = _save_checkpoint_exclusive(checkpoint_path, model) if checkpoint_path is not None else None
    encoder_adds = model.encoder.transform.scalar_add_subtracts
    decoder_adds = model.decoder.transform.scalar_add_subtracts
    return {
        "schema": SCHEMA,
        "complete": True,
        "run_key": config.run_key,
        "config": asdict(config),
        "train_seconds": train_seconds,
        "checkpoint": checkpoint,
        "loss_history": loss_history,
        "route_health": route_health,
        "deterministic_information_identity": "I(code; input) = H(code) because the hard code is deterministic",
        "frozen_transform": {
            "kind": "randomized_unnormalized_walsh_hadamard_HD",
            "encoder_signs": model.encoder.transform.signs.cpu().tolist(),
            "decoder_signs": model.decoder.transform.signs.cpu().tolist(),
            "encoder_anchors": model.encoder.lut.anchors.cpu().tolist(),
            "decoder_anchors": model.decoder.lut.anchors.cpu().tolist(),
        },
        "arithmetic_ledger_per_example": {
            "shared_butterfly_add_subtracts": encoder_adds + decoder_adds,
            "encoder_butterfly_add_subtracts": encoder_adds,
            "decoder_butterfly_add_subtracts": decoder_adds,
            "pair_coordinate_reads": 4 * config.tables * config.comparisons,
            "pair_subtractions_and_comparisons": 2 * config.tables * config.comparisons,
            "input_sign_bit_loads": config.n_features + config.model_dim,
            "materialized_transform_values": config.n_features + config.model_dim,
            "learned_transform_parameters": 0,
        },
        "environment": {
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        },
    }


def seal(output_dir: Path) -> dict[str, object]:
    paths = sorted((output_dir / "runs").glob("*.json"))
    rows = [json.loads(path.read_text()) for path in paths]
    if len(rows) != len(REGISTERED_SEEDS):
        raise RuntimeError(f"expected {len(REGISTERED_SEEDS)} Walsh runs, found {len(rows)}")
    if {row["config"]["seed"] for row in rows} != set(REGISTERED_SEEDS):
        raise RuntimeError("Walsh seed set is incomplete")
    if any(row.get("schema") != SCHEMA or row.get("complete") is not True for row in rows):
        raise RuntimeError("invalid Walsh run")
    entropies = [row["route_health"]["encoder"]["entropy_bits_mean"] for row in rows]
    payload = {
        "schema": COMPLETION_SCHEMA,
        "complete": True,
        "run_count": len(rows),
        "seeds": list(REGISTERED_SEEDS),
        "encoder_entropy_bits_by_seed": entropies,
        "encoder_entropy_bits_mean": sum(entropies) / len(entropies),
        "g1_threshold_bits": 3.0,
        "g1_all_seeds_pass": all(value >= 3.0 for value in entropies),
        "shared_butterfly_add_subtracts_per_example": rows[0]["arithmetic_ledger_per_example"]["shared_butterfly_add_subtracts"],
        "claim_boundary": "recognition-only Stage-1'; no validation/test capacity result",
    }
    _write_exclusive(output_dir / "completion.json", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Walsh/FWHT shared-compute Stage-1' recognition diagnostic")
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--seed", type=int, required=True)
    run_parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    seal_parser = commands.add_parser("seal")
    seal_parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run":
        config = Config(seed=args.seed, device=args.device)
        output = args.output_dir / "runs" / f"{config.run_key}.json"
        checkpoint = args.output_dir / "checkpoints" / f"{config.run_key}.pt"
        if output.exists():
            existing = json.loads(output.read_text())
            if existing.get("schema") == SCHEMA and existing.get("complete") is True:
                print(f"skip complete {output}")
                return
            raise RuntimeError(f"refusing to overwrite {output}")
        result = run(config, checkpoint_path=checkpoint)
        _write_exclusive(output, result)
        print(json.dumps(result["route_health"], indent=2, sort_keys=True))
        return
    print(json.dumps(seal(args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
