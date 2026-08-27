from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from tropnn.layers.hard_lookup import HardLookupRouter

Address = Literal["flat", "tree"]
Action = Literal["constant", "live"]
ARMS = ("flat_constant", "tree_constant", "flat_live", "tree_live")


@dataclass(frozen=True)
class ArmResult:
    arm: str
    seed: int
    parameters: int
    held_mse: float
    held_nmse: float
    held_r2: float
    held_cosine: float
    observed_rows: int
    route_entropy_bits: float
    maximum_row_mass: float
    threshold_rms: float
    seconds: float
    training_curve: list[dict[str, float]]


def orthogonal_teacher(dim: int, seed: int, device: torch.device) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(10_000 + seed)
    raw = torch.randn(dim, dim, generator=generator, dtype=torch.float64)
    q, r = torch.linalg.qr(raw)
    signs = torch.sign(torch.diag(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return (q * signs.view(1, -1)).float().to(device)


def sample_pair_anchors(count: int, dim: int, seed: int) -> Tensor:
    if dim < 2 or count < 1:
        raise ValueError("pair anchors require dim>=2 and count>=1")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    left = torch.randint(dim, (count,), generator=generator)
    offset = torch.randint(1, dim, (count,), generator=generator)
    right = (left + offset) % dim
    return torch.stack((left, right), dim=-1)


def _make_pair_page_regressor(
    dim: int,
    depth: int,
    address: Address,
    action: Action,
    *,
    anchor_seed: int,
    row_seed: int,
    tau: float,
) -> HardLookupRouter:
    """Construct experiment initialization around the shared router core."""

    support_count = depth if address == "flat" else 2**depth - 1
    supports = sample_pair_anchors(support_count, dim, anchor_seed).unsqueeze(0)
    threshold_count = depth if address == "flat" else 2**depth - 1
    thresholds = torch.zeros(1, threshold_count)
    generator = torch.Generator(device="cpu").manual_seed(row_seed)
    rows = torch.randn(1, 2**depth, dim, generator=generator) * 0.01
    slopes = torch.zeros_like(rows) if action == "live" else None
    return HardLookupRouter(
        dim,
        dim,
        depth=depth,
        predicate="pair",
        topology="flat" if address == "flat" else "adaptive",
        support_layout="level" if address == "flat" else "node",
        supports=supports,
        thresholds=thresholds,
        rows=rows,
        surrogate="soft_product",
        action="constant" if action == "constant" else "diagonal_live",
        slopes=slopes,
        tau=tau,
    )


def _metrics(prediction: Tensor, target: Tensor) -> tuple[float, float, float, float]:
    mse = float(F.mse_loss(prediction, target).item())
    energy = float(target.square().mean().clamp_min(1e-30).item())
    nmse = mse / energy
    cosine = float(F.cosine_similarity(prediction, target, dim=-1).mean().item())
    return mse, nmse, 1.0 - nmse, cosine


def _route_health(codes: Tensor, rows: int) -> tuple[int, float, float]:
    counts = torch.bincount(codes.cpu(), minlength=rows).double()
    probabilities = counts / counts.sum()
    positive = probabilities > 0
    entropy = float(-(probabilities[positive] * probabilities[positive].log2()).sum())
    return int(positive.sum()), entropy, float(probabilities.max())


def _build_models(dim: int, depth: int, seed: int, tau: float, device: torch.device) -> dict[str, HardLookupRouter]:
    models: dict[str, HardLookupRouter] = {}
    for address, anchor_seed, row_seed in (("flat", seed + 1_000, seed + 3_000), ("tree", seed + 2_000, seed + 4_000)):
        for action in ("constant", "live"):
            arm = f"{address}_{action}"
            models[arm] = _make_pair_page_regressor(
                dim,
                depth,
                address,  # type: ignore[arg-type]
                action,  # type: ignore[arg-type]
                anchor_seed=anchor_seed,
                row_seed=row_seed,
                tau=tau,
            ).to(device)
    for address in ("flat", "tree"):
        constant = models[f"{address}_constant"]
        live = models[f"{address}_live"]
        if not torch.equal(constant.supports, live.supports) or not torch.equal(constant.rows, live.rows):
            raise AssertionError("matched constant/live initialization failed")
        with torch.no_grad():
            probe = torch.randn(32, dim, device=device)
            if not torch.equal(constant(probe), live(probe)):
                raise AssertionError("zero-slope live initialization is not nested")
    return models


def fit_seed(
    seed: int,
    *,
    dim: int,
    depth: int,
    steps: int,
    batch_size: int,
    held_samples: int,
    lr: float,
    tau: float,
    device: str,
    log_every: int,
) -> tuple[list[ArmResult], dict[str, object], dict[str, Tensor]]:
    dev = torch.device(device)
    teacher = orthogonal_teacher(dim, seed, dev)
    models = _build_models(dim, depth, seed, tau, dev)
    module_dict = nn.ModuleDict(models)
    optimizer = torch.optim.AdamW(module_dict.parameters(), lr=lr, weight_decay=0.0)
    train_generator = torch.Generator(device=dev).manual_seed(20_000 + seed)
    curves: dict[str, list[dict[str, float]]] = {arm: [] for arm in ARMS}
    started = time.perf_counter()
    for step in range(1, steps + 1):
        x = torch.randn(batch_size, dim, generator=train_generator, device=dev)
        target = x @ teacher.T
        optimizer.zero_grad(set_to_none=True)
        losses = {arm: F.mse_loss(model(x), target) for arm, model in models.items()}
        sum(losses.values()).backward()
        optimizer.step()
        if step == 1 or step % log_every == 0 or step == steps:
            for arm, loss in losses.items():
                curves[arm].append({"step": float(step), "train_mse": float(loss.detach())})
    elapsed = time.perf_counter() - started

    held_generator = torch.Generator(device=dev).manual_seed(30_000 + seed)
    x_held = torch.randn(held_samples, dim, generator=held_generator, device=dev)
    target_held = x_held @ teacher.T
    results: list[ArmResult] = []
    state: dict[str, Tensor] = {"teacher": teacher.detach().cpu()}
    with torch.no_grad():
        for arm in ARMS:
            model = models[arm].eval()
            prediction = model(x_held)
            mse, nmse, r2, cosine = _metrics(prediction, target_held)
            codes = model.hard_codes(x_held).squeeze(-1)
            observed, entropy, maximum_mass = _route_health(codes, 2**depth)
            results.append(
                ArmResult(
                    arm=arm,
                    seed=seed,
                    parameters=sum(parameter.numel() for parameter in model.parameters()),
                    held_mse=mse,
                    held_nmse=nmse,
                    held_r2=r2,
                    held_cosine=cosine,
                    observed_rows=observed,
                    route_entropy_bits=entropy,
                    maximum_row_mass=maximum_mass,
                    threshold_rms=float(model.thresholds.square().mean().sqrt()),
                    seconds=elapsed,
                    training_curve=curves[arm],
                )
            )
            for name, value in model.state_dict().items():
                state[f"{arm}.{name}"] = value.detach().cpu()
        diagonal_prediction = x_held * torch.diag(teacher)
        diagonal_mse, diagonal_nmse, diagonal_r2, diagonal_cosine = _metrics(diagonal_prediction, target_held)
    controls = {
        "dense_exact": {"held_nmse": 0.0, "held_r2": 1.0, "parameters": dim * dim},
        "zero_constant": {"held_nmse": 1.0, "held_r2": 0.0, "parameters": 0},
        "analytic_global_diagonal": {
            "held_mse": diagonal_mse,
            "held_nmse": diagonal_nmse,
            "held_r2": diagonal_r2,
            "held_cosine": diagonal_cosine,
            "parameters": dim,
        },
    }
    return results, controls, state


def summarize(rows: list[ArmResult]) -> dict[str, object]:
    by_seed = {(row.seed, row.arm): row for row in rows}
    seeds = sorted({row.seed for row in rows})
    arm_summary: dict[str, object] = {}
    for arm in ARMS:
        values = torch.tensor([by_seed[(seed, arm)].held_r2 for seed in seeds], dtype=torch.float64)
        nmse = torch.tensor([by_seed[(seed, arm)].held_nmse for seed in seeds], dtype=torch.float64)
        arm_summary[arm] = {
            "held_r2_mean": float(values.mean()),
            "held_r2_sample_std": float(values.std(unbiased=True)) if values.numel() > 1 else 0.0,
            "held_nmse_mean": float(nmse.mean()),
            "held_nmse_sample_std": float(nmse.std(unbiased=True)) if nmse.numel() > 1 else 0.0,
        }
    per_seed: dict[str, object] = {}
    address_effects: list[float] = []
    live_effects: list[float] = []
    interactions: list[float] = []
    for seed in seeds:
        r2 = {arm: by_seed[(seed, arm)].held_r2 for arm in ARMS}
        address_constant = r2["tree_constant"] - r2["flat_constant"]
        address_live = r2["tree_live"] - r2["flat_live"]
        live_flat = r2["flat_live"] - r2["flat_constant"]
        live_tree = r2["tree_live"] - r2["tree_constant"]
        interaction = live_tree - live_flat
        address_effects.extend((address_constant, address_live))
        live_effects.extend((live_flat, live_tree))
        interactions.append(interaction)
        per_seed[str(seed)] = {
            "address_under_constant": address_constant,
            "address_under_live": address_live,
            "live_under_flat": live_flat,
            "live_under_tree": live_tree,
            "interaction": interaction,
        }
    mean_address = sum(address_effects) / len(address_effects)
    mean_live = sum(live_effects) / len(live_effects)
    mean_interaction = sum(interactions) / len(interactions)
    tree_live_mean = float(arm_summary["tree_live"]["held_r2_mean"])  # type: ignore[index]
    return {
        "arms": arm_summary,
        "effects_per_seed": per_seed,
        "decisions": {
            "address_factor": {
                "pass": min(address_effects) > 0 and mean_address >= 0.02,
                "all_simple_effects": address_effects,
                "grand_mean": mean_address,
            },
            "live_action_factor": {
                "pass": min(live_effects) > 0 and mean_live >= 0.02,
                "all_simple_effects": live_effects,
                "grand_mean": mean_live,
            },
            "synergy": {"pass": mean_interaction >= 0.02, "mean_interaction": mean_interaction},
            "strong_tree_live_random_linear_fit": {"pass": tree_live_mean >= 0.90, "mean_r2": tree_live_mean},
        },
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    all_rows: list[ArmResult] = []
    controls: dict[str, object] = {}
    artifact: dict[str, Tensor] = {}
    for seed in args.seeds:
        rows, seed_controls, state = fit_seed(
            seed,
            dim=args.dim,
            depth=args.depth,
            steps=args.steps,
            batch_size=args.batch_size,
            held_samples=args.held_samples,
            lr=args.lr,
            tau=args.tau,
            device=args.device,
            log_every=args.log_every,
        )
        all_rows.extend(rows)
        controls[str(seed)] = seed_controls
        artifact.update({f"seed{seed}.{name}": value for name, value in state.items()})
        print(
            f"seed={seed} " + " ".join(f"{row.arm}:R2={row.held_r2:.5f}" for row in rows),
            flush=True,
        )
    result = {
        "schema": "random-linear-pair-address-action-factorial-v1",
        "protocol": {
            "dim": args.dim,
            "depth": args.depth,
            "active_comparisons": args.depth,
            "rows": 2**args.depth,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "held_samples": args.held_samples,
            "lr": args.lr,
            "tau": args.tau,
            "seeds": list(args.seeds),
            "input": "fresh_standard_Gaussian",
            "teacher": "QR_Haar_orthogonal_linear_map",
            "optimizer": "AdamW_weight_decay_0",
            "offline_compiler_used": False,
            "hard_forward_soft_backward": True,
        },
        "rows": [asdict(row) for row in all_rows],
        "controls": controls,
        "summary": summarize(all_rows),
        "semantic_ledger": {
            "flat_address": {"stored_pair_predicates": args.depth, "active_comparisons": args.depth, "dependent_steps": 1},
            "tree_address": {"stored_pair_predicates": 2**args.depth - 1, "active_comparisons": args.depth, "dependent_steps": args.depth},
            "constant_action": {"active_vector_reads": 1, "active_scalars": args.dim, "multiplications": 0},
            "live_action": {"active_vector_reads": 2, "active_scalars": 2 * args.dim, "multiplications": args.dim},
        },
    }
    artifact_path = Path(args.artifact)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    if artifact_path.exists():
        raise FileExistsError(artifact_path)
    torch.save({"schema": result["schema"], "protocol": result["protocol"], "state": artifact}, artifact_path)
    return result


def _parse_seeds(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item)
    if not result:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct random-linear Flat/Tree x Constant/Live Pair-page factorial")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--held-samples", type=int, default=32768)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--seeds", type=_parse_seeds, default=(0, 1, 2))
    parser.add_argument("--log-every", type=int, default=300)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.output).exists() or Path(args.artifact).exists():
        parser.error("output and artifact paths must not exist")
    if Path(args.output).resolve(strict=False) == Path(args.artifact).resolve(strict=False):
        parser.error("output and artifact paths must differ")
    return args


def main() -> None:
    args = parse_args()
    result = run(args)
    serialized = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(serialized)
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
