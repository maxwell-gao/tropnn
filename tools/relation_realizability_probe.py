from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from tropnn.layers.relation import ComparisonRelationLUT, RelationQuantization
from tropnn.tools.comparison_relation_probe import sample_pair_indices

TEACHERS = ("unary", "additive", "free_float", "free_ternary", "free_binary")
STUDENTS = ("unary", "additive", "constrained", "free_float", "free_ternary", "free_binary")


@dataclass(frozen=True)
class RealizabilityConfig:
    out_dir: Path
    teacher: str = "free_float"
    student: str = "free_float"
    device: str = "cuda"
    input_dim: int = 64
    num_banks: int = 16
    num_codes: int = 32
    relation_rank: int = 16
    train_objects: int = 4096
    test_objects: int = 1024
    batch_size: int = 4096
    eval_pairs: int = 65536
    steps: int = 10000
    eval_every: int = 1000
    learning_rate: float = 0.003
    seed: int = 0


def normalized_integer_support(count: int, dim: int, *, seed: int, device: torch.device) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    samples = torch.randint(-15, 16, (count, dim), generator=generator).float()
    return torch.nn.functional.normalize(samples, dim=-1).to(device)


def base_router(config: RealizabilityConfig, query: Tensor, key: Tensor) -> ComparisonRelationLUT:
    layer = ComparisonRelationLUT(
        config.input_dim,
        num_banks=config.num_banks,
        num_codes=config.num_codes,
        relation_rank=config.relation_rank,
        relation_mode="free",
        relation_init="zeros",
        seed=config.seed,
    ).to(query.device)
    layer.calibrate_routes(query, key)
    return layer


def copy_routes(source: ComparisonRelationLUT, target: ComparisonRelationLUT) -> None:
    target.query_router.load_state_dict(source.query_router.state_dict())
    target.key_router.load_state_dict(source.key_router.state_dict())
    target.set_threshold_training(False)


def make_teacher(config: RealizabilityConfig, router: ComparisonRelationLUT) -> ComparisonRelationLUT:
    teacher = ComparisonRelationLUT(
        config.input_dim,
        num_banks=config.num_banks,
        num_codes=config.num_codes,
        relation_rank=config.relation_rank,
        relation_mode="free",
        relation_init="zeros",
        seed=config.seed + 11,
    ).to(router.relation.device)
    copy_routes(router, teacher)
    generator = torch.Generator(device="cpu").manual_seed(config.seed + 211)
    with torch.no_grad():
        if config.teacher == "unary":
            query = torch.randn(config.num_banks, config.num_codes, generator=generator)
            teacher.relation.copy_(query.unsqueeze(-1).expand_as(teacher.relation))
        elif config.teacher == "additive":
            query = torch.randn(config.num_banks, config.num_codes, generator=generator)
            key = torch.randn(config.num_banks, config.num_codes, generator=generator)
            teacher.relation.copy_(query.unsqueeze(-1) + key.unsqueeze(-2))
        elif config.teacher == "free_float":
            teacher.relation.copy_(
                torch.randn(teacher.relation.shape, generator=generator).to(teacher.relation)
            )
        elif config.teacher == "free_ternary":
            teacher.relation.copy_(
                torch.randint(-1, 2, teacher.relation.shape, generator=generator, dtype=torch.int64).float()
            )
        elif config.teacher == "free_binary":
            teacher.relation.copy_(
                2.0 * torch.randint(0, 2, teacher.relation.shape, generator=generator).float() - 1.0
            )
        else:
            raise ValueError(f"unsupported teacher {config.teacher!r}")
    teacher.requires_grad_(False)
    teacher.eval()
    return teacher


def make_student(config: RealizabilityConfig, router: ComparisonRelationLUT) -> ComparisonRelationLUT:
    mode = "free"
    quantization: RelationQuantization = "float"
    if config.student == "unary" or config.student == "additive":
        mode = "additive"
    elif config.student == "constrained":
        mode = "constrained_gram"
    elif config.student.endswith("ternary"):
        quantization = "ternary"
    elif config.student.endswith("binary"):
        quantization = "binary"
    student = ComparisonRelationLUT(
        config.input_dim,
        num_banks=config.num_banks,
        num_codes=config.num_codes,
        relation_rank=config.relation_rank,
        relation_mode=mode,
        relation_init="random",
        quantization=quantization,
        seed=config.seed + 307,
    ).to(router.relation.device)
    copy_routes(router, student)
    if config.student == "unary":
        with torch.no_grad():
            student.key_values.zero_()
        student.key_values.requires_grad_(False)
    return student


@torch.no_grad()
def split_metrics(
    student: ComparisonRelationLUT,
    teacher: ComparisonRelationLUT,
    query: Tensor,
    key: Tensor,
    *,
    count: int,
    held: bool,
    seed: int,
) -> dict[str, float]:
    qi, ki = sample_pair_indices(
        query.shape[0],
        key.shape[0],
        count,
        held=held,
        seed=seed,
        device=query.device,
    )
    prediction = student.score_aligned(query[qi], key[ki])
    target = teacher.score_aligned(query[qi], key[ki])
    error = prediction - target
    variance = target.var(unbiased=False).clamp_min(1e-12)
    return {
        "mse": float(error.square().mean().item()),
        "normalized_mse": float((error.square().mean() / variance).item()),
        "r2": float((1.0 - error.square().mean() / variance).item()),
    }


def run(config: RealizabilityConfig) -> dict[str, object]:
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    train_query = normalized_integer_support(
        config.train_objects, config.input_dim, seed=config.seed + 401, device=device
    )
    train_key = normalized_integer_support(
        config.train_objects, config.input_dim, seed=config.seed + 409, device=device
    )
    test_query = normalized_integer_support(
        config.test_objects, config.input_dim, seed=config.seed + 419, device=device
    )
    test_key = normalized_integer_support(
        config.test_objects, config.input_dim, seed=config.seed + 421, device=device
    )
    router = base_router(config, train_query, train_key)
    teacher = make_teacher(config, router)
    student = make_student(config, router)
    config.out_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(step: int) -> dict[str, float | int]:
        result: dict[str, float | int] = {"step": step}
        for prefix, query, key, held in (
            ("train_pair", train_query, train_key, False),
            ("held_pair", train_query, train_key, True),
            ("held_object", test_query, test_key, False),
        ):
            metrics = split_metrics(
                student,
                teacher,
                query,
                key,
                count=config.eval_pairs,
                held=held,
                seed=config.seed + 503,
            )
            result.update({f"{prefix}_{name}": value for name, value in metrics.items()})
        return result

    history: list[dict[str, float | int]] = [evaluate(0)]
    optimizer = torch.optim.AdamW(
        [parameter for parameter in student.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )
    started = time.perf_counter()
    for step in range(1, config.steps + 1):
        qi, ki = sample_pair_indices(
            train_query.shape[0],
            train_key.shape[0],
            config.batch_size,
            held=False,
            seed=config.seed + step * 7919,
            device=device,
        )
        prediction = student.score_aligned(train_query[qi], train_key[ki])
        with torch.no_grad():
            target = teacher.score_aligned(train_query[qi], train_key[ki])
        loss = (prediction - target).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % config.eval_every == 0 or step == config.steps:
            student.eval()
            row = evaluate(step)
            row["train_loss"] = float(loss.item())
            row["steps_per_second"] = step / max(time.perf_counter() - started, 1e-9)
            history.append(row)
            student.train()

    summary = {
        "config": {**asdict(config), "out_dir": str(config.out_dir)},
        "parameters": sum(parameter.numel() for parameter in student.parameters()),
        "initial": history[0],
        "final": history[-1],
    }
    with (config.out_dir / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in history for key in row}))
        writer.writeheader()
        writer.writerows(history)
    (config.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    torch.save(student.state_dict(), config.out_dir / "checkpoint.pt")
    print(json.dumps(summary["final"], sort_keys=True))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Realizable fixed-route unary and binary relation controls.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--teacher", choices=TEACHERS, default="free_float")
    parser.add_argument("--student", choices=STUDENTS, default="free_float")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--input-dim", type=int, default=64)
    parser.add_argument("--num-banks", type=int, default=16)
    parser.add_argument("--num-codes", type=int, default=32)
    parser.add_argument("--relation-rank", type=int, default=16)
    parser.add_argument("--train-objects", type=int, default=4096)
    parser.add_argument("--test-objects", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--eval-pairs", type=int, default=65536)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    run(RealizabilityConfig(**vars(build_parser().parse_args())))


if __name__ == "__main__":
    main()
