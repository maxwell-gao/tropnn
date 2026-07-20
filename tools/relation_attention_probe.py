from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from tropnn.layers.relation import ComparisonRelationLUT, RelationQuantization

VARIANTS = (
    "dense",
    "constrained_float",
    "gram_free_float",
    "random_free_float",
    "gram_free_ternary",
    "random_free_ternary",
    "gram_free_binary",
)


@dataclass(frozen=True)
class AttentionConfig:
    out_dir: Path
    variant: str = "gram_free_float"
    device: str = "cuda"
    input_dim: int = 64
    vocabulary: int = 1024
    memories: int = 64
    depth: int = 4
    task_hops: int = 0
    train_vocabulary: int = 768
    num_banks: int = 16
    num_codes: int = 32
    relation_rank: int = 16
    shared_relation: bool = False
    query_noise: float = 0.05
    eval_noises: tuple[float, ...] = (0.0, 0.05, 0.1)
    batch_size: int = 256
    fixed_steps: int = 20000
    joint_steps: int = 20000
    eval_every: int = 1000
    learning_rate: float = 0.003
    threshold_learning_rate: float = 0.0003
    temperature: float = 0.125
    loss_mode: str = "final"
    seed: int = 0


class DenseBilinearRelation(nn.Module):
    def __init__(self, matrix: Tensor) -> None:
        super().__init__()
        self.register_buffer("matrix", matrix)

    def forward(self, query: Tensor, key: Tensor) -> Tensor:
        return query @ self.matrix @ key.transpose(-1, -2)


class RelationChain(nn.Module):
    def __init__(self, scorers: list[nn.Module], *, temperature: float) -> None:
        super().__init__()
        if len(scorers) > 1 and all(scorer is scorers[0] for scorer in scorers):
            self.shared_scorer = scorers[0]
            self.scorers = None
            self.depth = len(scorers)
        else:
            self.shared_scorer = None
            self.scorers = nn.ModuleList(scorers)
            self.depth = len(scorers)
        self.temperature = temperature

    def scorer(self, depth: int) -> nn.Module:
        return self.shared_scorer if self.shared_scorer is not None else self.scorers[depth]

    def forward(self, query: Tensor, keys: Tensor, values: Tensor) -> list[Tensor]:
        states: list[Tensor] = []
        state = query
        for depth in range(self.depth):
            score = self.scorer(depth)(state.unsqueeze(1), keys).squeeze(1)
            probability = torch.softmax(score / self.temperature, dim=-1)
            state = torch.bmm(probability.unsqueeze(1), values).squeeze(1)
            state = torch.nn.functional.normalize(state, dim=-1)
            states.append(state)
        return states

    def set_threshold_training(self, enabled: bool) -> None:
        scorers = [self.shared_scorer] if self.shared_scorer is not None else list(self.scorers)
        for scorer in scorers:
            if isinstance(scorer, ComparisonRelationLUT):
                scorer.set_threshold_training(enabled)


class EpisodeFactory:
    def __init__(self, config: AttentionConfig, device: torch.device) -> None:
        self.config = config
        self.device = device
        generator = torch.Generator(device="cpu").manual_seed(config.seed)
        embedding = torch.randn(config.vocabulary, config.input_dim, generator=generator)
        self.embedding = torch.nn.functional.normalize(embedding, dim=-1).to(device)
        self.query_view = torch.linalg.qr(
            torch.randn(config.input_dim, config.input_dim, generator=generator)
        ).Q.to(device)
        self.key_view = torch.linalg.qr(
            torch.randn(config.input_dim, config.input_dim, generator=generator)
        ).Q.to(device)
        self.query_embedding = self.embedding @ self.query_view
        self.key_embedding = self.embedding @ self.key_view
        self.train_generator = torch.Generator(device=device).manual_seed(config.seed + 101)

    @property
    def dense_matrix(self) -> Tensor:
        return self.query_view.T @ self.key_view

    def sample(
        self,
        batch_size: int,
        *,
        generator: torch.Generator,
        pool_start: int,
        pool_end: int,
        noise: float | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        config = self.config
        hops = config.task_hops or config.depth
        pool_size = pool_end - pool_start
        if hops + 1 > config.memories:
            raise ValueError("memories must contain every chain edge and the terminal self-loop")
        if pool_size < config.memories:
            raise ValueError("node pool must be at least as large as memories")
        nodes = torch.stack(
            [
                torch.randperm(pool_size, generator=generator, device=self.device)[: config.memories] + pool_start
                for _ in range(batch_size)
            ]
        )
        chain = nodes[:, : hops + 1]
        distractor_source = nodes[:, hops + 1 :]
        distractor_target = torch.roll(distractor_source, shifts=1, dims=-1)
        source = torch.cat([chain, distractor_source], dim=-1)
        target = torch.cat([chain[:, 1:], chain[:, -1:], distractor_target], dim=-1)
        permutation = torch.stack(
            [torch.randperm(config.memories, generator=generator, device=self.device) for _ in range(batch_size)]
        )
        source = source.gather(1, permutation)
        target = target.gather(1, permutation)
        keys = self.key_embedding[source]
        values = self.query_embedding[target]
        query = self.query_embedding[chain[:, 0]]
        sigma = config.query_noise if noise is None else noise
        if sigma:
            perturbation = torch.randn(query.shape, generator=generator, device=self.device, dtype=query.dtype)
            query = torch.nn.functional.normalize(query + sigma * perturbation, dim=-1)
        layer_index = torch.arange(1, config.depth + 1, device=self.device).clamp_max(hops)
        targets = self.query_embedding[chain[:, layer_index]]
        final_target = self.query_embedding[chain[:, hops]]
        return query, keys, values, targets, final_target


def initialized_relation(
    config: AttentionConfig,
    factory: EpisodeFactory,
    *,
    quantization: RelationQuantization,
    random_free: bool,
    seed_offset: int,
) -> ComparisonRelationLUT:
    constrained = ComparisonRelationLUT(
        config.input_dim,
        num_banks=config.num_banks,
        num_codes=config.num_codes,
        relation_rank=config.relation_rank,
        relation_mode="constrained_gram",
        relation_init="zeros",
        seed=config.seed + seed_offset,
    ).to(factory.device)
    train_query = factory.query_embedding[: config.train_vocabulary]
    train_key = factory.key_embedding[: config.train_vocabulary]
    constrained.calibrate_routes(train_query, train_key)
    features = factory.embedding[: config.train_vocabulary, : config.relation_rank]
    if features.shape[-1] < config.relation_rank:
        features = torch.nn.functional.pad(features, (0, config.relation_rank - features.shape[-1]))
    constrained.initialize_from_samples(train_query, train_key, features, features)
    if config.variant == "constrained_float":
        return constrained
    return ComparisonRelationLUT.free_from_constrained(
        constrained,
        quantization=quantization,
        random_init=random_free,
        seed=config.seed + seed_offset,
    ).to(factory.device)


def make_model(config: AttentionConfig, factory: EpisodeFactory) -> RelationChain:
    if config.variant == "dense":
        scorer = DenseBilinearRelation(factory.dense_matrix)
        return RelationChain([scorer] * config.depth, temperature=config.temperature)
    quantization: RelationQuantization = "float"
    if config.variant.endswith("ternary"):
        quantization = "ternary"
    elif config.variant.endswith("binary"):
        quantization = "binary"
    random_free = config.variant.startswith("random_free")
    first = initialized_relation(
        config,
        factory,
        quantization=quantization,
        random_free=random_free,
        seed_offset=0,
    )
    scorers = [first]
    if config.shared_relation:
        scorers *= config.depth
    else:
        scorers.extend(
            initialized_relation(
                config,
                factory,
                quantization=quantization,
                random_free=random_free,
                seed_offset=depth * 1000003,
            )
            for depth in range(1, config.depth)
        )
    return RelationChain(scorers, temperature=config.temperature)


@torch.no_grad()
def evaluate(model: RelationChain, factory: EpisodeFactory, *, batches: int = 8) -> dict[str, float]:
    result: dict[str, float] = {}
    config = factory.config
    pools = {
        "seen": (0, config.train_vocabulary),
        "unseen": (config.train_vocabulary, config.vocabulary),
    }
    for split, (pool_start, pool_end) in pools.items():
        for noise_index, noise in enumerate(config.eval_noises):
            generator = torch.Generator(device=factory.device).manual_seed(
                config.seed + 100003 * noise_index + (0 if split == "seen" else 500009)
            )
            layer_accuracy = torch.zeros(model.depth, device=factory.device)
            layer_cosine = torch.zeros(model.depth, device=factory.device)
            final_accuracy = torch.zeros((), device=factory.device)
            final_cosine = torch.zeros((), device=factory.device)
            for _ in range(batches):
                query, keys, values, targets, final_target = factory.sample(
                    config.batch_size,
                    generator=generator,
                    pool_start=pool_start,
                    pool_end=pool_end,
                    noise=noise,
                )
                states = model(query, keys, values)
                for depth, state in enumerate(states):
                    target = targets[:, depth]
                    layer_cosine[depth] += torch.nn.functional.cosine_similarity(state, target, dim=-1).mean()
                    prediction = (state @ factory.query_embedding.T).argmax(dim=-1)
                    expected = (target @ factory.query_embedding.T).argmax(dim=-1)
                    layer_accuracy[depth] += (prediction == expected).float().mean()
                final_state = states[-1]
                final_prediction = (final_state @ factory.query_embedding.T).argmax(dim=-1)
                final_expected = (final_target @ factory.query_embedding.T).argmax(dim=-1)
                final_accuracy += (final_prediction == final_expected).float().mean()
                final_cosine += torch.nn.functional.cosine_similarity(final_state, final_target, dim=-1).mean()
            noise_tag = str(noise).replace(".", "p")
            prefix = f"{split}_noise{noise_tag}"
            result[f"{prefix}_task_accuracy"] = float((final_accuracy / batches).item())
            result[f"{prefix}_task_cosine"] = float((final_cosine / batches).item())
            for depth in range(model.depth):
                result[f"{prefix}_layer{depth + 1}_accuracy"] = float((layer_accuracy[depth] / batches).item())
                result[f"{prefix}_layer{depth + 1}_cosine"] = float((layer_cosine[depth] / batches).item())
    return result


def optimizer_for(model: RelationChain, config: AttentionConfig, *, joint: bool) -> torch.optim.Optimizer:
    threshold_ids = {
        id(router.thresholds)
        for module in model.modules()
        if isinstance(module, ComparisonRelationLUT)
        for router in (module.query_router, module.key_router)
    }
    payload = [p for p in model.parameters() if p.requires_grad and id(p) not in threshold_ids]
    thresholds = [p for p in model.parameters() if p.requires_grad and id(p) in threshold_ids]
    groups: list[dict[str, object]] = [{"params": payload, "lr": config.learning_rate}]
    if joint and thresholds:
        groups.append({"params": thresholds, "lr": config.threshold_learning_rate})
    return torch.optim.AdamW(groups, weight_decay=0.0, betas=(0.9, 0.95))


def run(config: AttentionConfig) -> dict[str, object]:
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    factory = EpisodeFactory(config, device)
    model = make_model(config, factory).to(device)
    config.out_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int | str]] = [
        {"phase": "initial", "step": 0, **evaluate(model, factory)}
    ]
    if config.variant != "dense":
        started = time.perf_counter()
        global_step = 0
        for phase, steps, joint in (("fixed", config.fixed_steps, False), ("joint", config.joint_steps, True)):
            model.set_threshold_training(joint)
            optimizer = optimizer_for(model, config, joint=joint)
            model.train()
            for phase_step in range(1, steps + 1):
                query, keys, values, targets, final_target = factory.sample(
                    config.batch_size,
                    generator=factory.train_generator,
                    pool_start=0,
                    pool_end=config.train_vocabulary,
                )
                states = model(query, keys, values)
                if config.loss_mode == "all":
                    intermediate = torch.stack(
                        [(state - targets[:, depth]).square().mean() for depth, state in enumerate(states)]
                    ).mean()
                    loss = intermediate + (states[-1] - final_target).square().mean()
                else:
                    loss = (states[-1] - final_target).square().mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                global_step += 1
                if phase_step % config.eval_every == 0 or phase_step == steps:
                    model.eval()
                    history.append(
                        {
                            "phase": phase,
                            "step": global_step,
                            "train_loss": float(loss.item()),
                            "steps_per_second": global_step / max(time.perf_counter() - started, 1e-9),
                            **evaluate(model, factory),
                        }
                    )
                    model.train()
            if phase == "fixed":
                torch.save(model.state_dict(), config.out_dir / "checkpoint_fixed.pt")
    model.eval()
    summary = {
        "config": {**asdict(config), "out_dir": str(config.out_dir)},
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "active_comparisons_per_query_key": 2 * config.num_banks * int(torch.tensor(config.num_codes).log2().item()),
        "active_relation_lookups_per_query_key": config.num_banks,
        "float_master_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
        "initial": history[0],
        "final": history[-1],
    }
    with (config.out_dir / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in history for key in row}))
        writer.writeheader()
        writer.writerows(history)
    (config.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    torch.save(model.state_dict(), config.out_dir / "checkpoint.pt")
    print(json.dumps(summary["final"], sort_keys=True))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multi-hop retrieval with directly trained relation LUTs.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, default="gram_free_float")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--input-dim", type=int, default=64)
    parser.add_argument("--vocabulary", type=int, default=1024)
    parser.add_argument("--memories", type=int, default=64)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--task-hops", type=int, default=0)
    parser.add_argument("--train-vocabulary", type=int, default=768)
    parser.add_argument("--num-banks", type=int, default=16)
    parser.add_argument("--num-codes", type=int, default=32)
    parser.add_argument("--relation-rank", type=int, default=16)
    parser.add_argument("--shared-relation", action="store_true")
    parser.add_argument("--query-noise", type=float, default=0.05)
    parser.add_argument(
        "--eval-noises",
        type=lambda value: tuple(float(item) for item in value.split(",")),
        default=(0.0, 0.05, 0.1),
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--fixed-steps", type=int, default=20000)
    parser.add_argument("--joint-steps", type=int, default=20000)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--threshold-learning-rate", type=float, default=0.0003)
    parser.add_argument("--temperature", type=float, default=0.125)
    parser.add_argument("--loss-mode", choices=("final", "all"), default="final")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    run(AttentionConfig(**vars(build_parser().parse_args())))


if __name__ == "__main__":
    main()
