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
from tropnn.tools.random_linear_address_action_factorial import (
    _metrics,
    _parse_seeds,
    orthogonal_teacher,
    sample_pair_anchors,
)

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
    mean_observed_rows_per_table: float
    mean_route_entropy_bits_per_table: float
    maximum_row_mass: float
    threshold_rms: float
    seconds: float
    training_curve: list[dict[str, float]]


def _make_multitable_regressor(
    dim: int,
    tables: int,
    depth: int,
    address: Address,
    action: Action,
    *,
    anchor_seed: int,
    row_seed: int,
    tau: float,
) -> HardLookupRouter:
    predicates = depth if address == "flat" else 2**depth - 1
    supports = sample_pair_anchors(tables * predicates, dim, anchor_seed).view(tables, predicates, 2)
    thresholds = torch.zeros(tables, predicates)
    generator = torch.Generator(device="cpu").manual_seed(row_seed)
    rows = torch.randn(tables, 2**depth, dim, generator=generator) * (0.01 / tables**0.5)
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


def _route_health(codes: Tensor, rows: int) -> tuple[float, float, float]:
    observed: list[float] = []
    entropies: list[float] = []
    maximum_mass = 0.0
    for table in range(codes.shape[1]):
        counts = torch.bincount(codes[:, table].cpu(), minlength=rows).double()
        probabilities = counts / counts.sum()
        positive = probabilities > 0
        observed.append(float(positive.sum()))
        entropies.append(float(-(probabilities[positive] * probabilities[positive].log2()).sum()))
        maximum_mass = max(maximum_mass, float(probabilities.max()))
    return sum(observed) / len(observed), sum(entropies) / len(entropies), maximum_mass


def _build_models(dim: int, tables: int, depth: int, seed: int, tau: float, device: torch.device) -> dict[str, HardLookupRouter]:
    models: dict[str, HardLookupRouter] = {}
    for address, anchor_seed, row_seed in (("flat", seed + 1_000, seed + 3_000), ("tree", seed + 2_000, seed + 4_000)):
        for action in ("constant", "live"):
            arm = f"{address}_{action}"
            models[arm] = _make_multitable_regressor(
                dim,
                tables,
                depth,
                address,  # type: ignore[arg-type]
                action,  # type: ignore[arg-type]
                anchor_seed=anchor_seed,
                row_seed=row_seed,
                tau=tau,
            ).to(device)
    for address in ("flat", "tree"):
        constant, live = models[f"{address}_constant"], models[f"{address}_live"]
        if not torch.equal(constant.supports, live.supports) or not torch.equal(constant.rows, live.rows):
            raise AssertionError("constant/live initialization mismatch")
        with torch.no_grad():
            probe = torch.randn(16, dim, device=device)
            if not torch.equal(constant(probe), live(probe)):
                raise AssertionError("zero-slope live is not nested")
    return models


def fit_seed(seed: int, args: argparse.Namespace) -> tuple[list[ArmResult], dict[str, object], dict[str, Tensor]]:
    device = torch.device(args.device)
    teacher = orthogonal_teacher(args.dim, seed, device)
    models = _build_models(args.dim, args.tables, args.depth, seed, args.tau, device)
    optimizer = torch.optim.AdamW(nn.ModuleDict(models).parameters(), lr=args.lr, weight_decay=0)
    generator = torch.Generator(device=device).manual_seed(20_000 + seed)
    curves: dict[str, list[dict[str, float]]] = {arm: [] for arm in ARMS}
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        x = torch.randn(args.batch_size, args.dim, generator=generator, device=device)
        target = x @ teacher.T
        optimizer.zero_grad(set_to_none=True)
        losses = {arm: F.mse_loss(model(x), target) for arm, model in models.items()}
        sum(losses.values()).backward()
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            for arm, loss in losses.items():
                curves[arm].append({"step": float(step), "train_mse": float(loss.detach())})
    elapsed = time.perf_counter() - started
    held_generator = torch.Generator(device=device).manual_seed(30_000 + seed)
    x = torch.randn(args.held_samples, args.dim, generator=held_generator, device=device)
    target = x @ teacher.T
    results: list[ArmResult] = []
    state = {"teacher": teacher.cpu()}
    with torch.no_grad():
        for arm in ARMS:
            model = models[arm].eval()
            mse, nmse, r2, cosine = _metrics(model(x), target)
            observed, entropy, maximum_mass = _route_health(model.hard_codes(x), 2**args.depth)
            results.append(
                ArmResult(
                    arm,
                    seed,
                    sum(p.numel() for p in model.parameters()),
                    mse,
                    nmse,
                    r2,
                    cosine,
                    observed,
                    entropy,
                    maximum_mass,
                    float(model.thresholds.square().mean().sqrt()),
                    elapsed,
                    curves[arm],
                )
            )
            for name, value in model.state_dict().items():
                state[f"{arm}.{name}"] = value.cpu()
        diagonal = _metrics(x * torch.diag(teacher), target)
    controls = {
        "dense_exact": {"held_nmse": 0.0, "held_r2": 1.0, "parameters": args.dim**2},
        "zero_constant": {"held_nmse": 1.0, "held_r2": 0.0, "parameters": 0},
        "analytic_global_diagonal": {
            "held_mse": diagonal[0],
            "held_nmse": diagonal[1],
            "held_r2": diagonal[2],
            "held_cosine": diagonal[3],
            "parameters": args.dim,
        },
    }
    return results, controls, state


def summarize(rows: list[ArmResult]) -> dict[str, object]:
    by_seed = {(row.seed, row.arm): row for row in rows}
    seeds = sorted({row.seed for row in rows})
    arms = {
        arm: {
            "held_r2_mean": sum(by_seed[seed, arm].held_r2 for seed in seeds) / len(seeds),
            "held_nmse_mean": sum(by_seed[seed, arm].held_nmse for seed in seeds) / len(seeds),
        }
        for arm in ARMS
    }
    effects: dict[str, object] = {}
    address, live, interaction = [], [], []
    for seed in seeds:
        r = {arm: by_seed[seed, arm].held_r2 for arm in ARMS}
        ac = r["tree_constant"] - r["flat_constant"]
        al = r["tree_live"] - r["flat_live"]
        lf = r["flat_live"] - r["flat_constant"]
        lt = r["tree_live"] - r["tree_constant"]
        address += [ac, al]
        live += [lf, lt]
        interaction.append(lt - lf)
        effects[str(seed)] = {
            "address_under_constant": ac,
            "address_under_live": al,
            "live_under_flat": lf,
            "live_under_tree": lt,
            "interaction": lt - lf,
        }
    mean_address, mean_live, mean_interaction = sum(address) / len(address), sum(live) / len(live), sum(interaction) / len(interaction)
    return {
        "arms": arms,
        "effects_per_seed": effects,
        "decisions": {
            "address_factor": {"pass": min(address) > 0 and mean_address >= 0.02, "grand_mean": mean_address, "all_simple_effects": address},
            "live_action_factor": {"pass": min(live) > 0 and mean_live >= 0.02, "grand_mean": mean_live, "all_simple_effects": live},
            "synergy": {"pass": mean_interaction >= 0.02, "mean_interaction": mean_interaction},
            "strong_tree_live_random_linear_fit": {"pass": arms["tree_live"]["held_r2_mean"] >= 0.90, "mean_r2": arms["tree_live"]["held_r2_mean"]},
        },
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    rows: list[ArmResult] = []
    controls: dict[str, object] = {}
    state: dict[str, Tensor] = {}
    for seed in args.seeds:
        seed_rows, seed_controls, seed_state = fit_seed(seed, args)
        rows.extend(seed_rows)
        controls[str(seed)] = seed_controls
        state.update({f"seed{seed}.{key}": value for key, value in seed_state.items()})
        print(f"seed={seed} " + " ".join(f"{row.arm}:R2={row.held_r2:.5f}" for row in seed_rows), flush=True)
    result = {
        "schema": "random-linear-multitable-pair-address-action-factorial-v1",
        "protocol": {
            "dim": args.dim,
            "tables": args.tables,
            "depth_per_table": args.depth,
            "active_comparisons": args.tables * args.depth,
            "active_row_reads": args.tables,
            "rows_per_table": 2**args.depth,
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
        "rows": [asdict(row) for row in rows],
        "controls": controls,
        "summary": summarize(rows),
        "semantic_ledger": {
            "flat_address": {
                "stored_pair_predicates": args.tables * args.depth,
                "active_comparisons": args.tables * args.depth,
                "dependent_steps": 1,
            },
            "tree_address": {
                "stored_pair_predicates": args.tables * (2**args.depth - 1),
                "active_comparisons": args.tables * args.depth,
                "dependent_steps": args.depth,
            },
            "constant_action": {"active_vector_reads": args.tables, "active_scalars": args.tables * args.dim, "multiplications": 0},
            "live_action": {"active_vector_reads": 2 * args.tables, "active_scalars": 2 * args.tables * args.dim, "multiplications": args.dim},
        },
    }
    artifact = Path(args.artifact)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    if artifact.exists():
        raise FileExistsError(artifact)
    torch.save({"schema": result["schema"], "protocol": result["protocol"], "state": state}, artifact)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Random-linear multi-table Flat/Tree x Constant/Live factorial")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--tables", type=int, default=32)
    parser.add_argument("--depth", type=int, default=4)
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
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
