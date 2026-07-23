"""Trainable local composition followed by hard ordinal root retrieval.

The root-transport qualification uses an oracle predecessor lane at every B
row.  This experiment removes that oracle.  A trainable local composer receives
``[predecessor, current]`` and must construct the K state from retrieval labels.

Two composers are tested:

* ``canon``: one learned two-lag softmax per coordinate;
* ``pclut``: a vector PC-LUT over the concatenated predecessor/current row.

The relation kernel is a frozen hard root-32 correspondence learned on disjoint
MQAR episodes.  ``twohop`` episodes contain shuffled A->B and B->C edges and
are evaluated autoregressively: the value selected on hop one becomes the
query token on hop two.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tropnn.layers.pairwise import PairwiseLUT
from tropnn.tools.causal_mqar_induction import (
    EVALUATION_SPECS,
    FixedOrdinalCodebook,
    MQARBatch,
    _clone_state_dict,
    _training_shape,
    atomic_json_write,
    atomic_torch_save,
    seed_everything,
    target_token_loss,
    token_permutation,
)
from tropnn.tools.causal_mqar_role_gauge import (
    coordinate_permutation,
    generate_pool_mqar_batch,
    token_pools,
)
from tropnn.tools.causal_mqar_root_transport import (
    RootTransportMQARRetriever,
    RootTransportRunConfig,
    SignedRootTransportKernel,
    root_signs,
)

COMPOSERS = ("oracle", "current", "canon", "pclut")
TASKS = ("onehop", "twohop")
COMPOSED_EVALUATION_CONDITIONS = ("base", "token_relabel", "current_only")


@dataclass(frozen=True)
class ComposedRunConfig:
    task: str
    composer: str
    seed: int
    relation_checkpoint: str
    steps: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    gradient_clip: float
    validation_interval: int
    validation_episodes: int
    evaluation_episodes: int
    evaluation_batch_size: int
    vocab_size: int
    train_tokens: int
    validation_tokens: int
    test_tokens: int
    relation_dim: int
    composer_tables: int
    composer_comparisons: int
    composer_lut_init_std: float
    root_ste_temperature: float
    data_seed: int
    codebook_seed: int
    token_relabel_seed: int
    device: str


@dataclass(frozen=True)
class TwoHopBatch:
    memory_keys: Tensor
    memory_values: Tensor
    queries: Tensor
    intermediates: Tensor
    targets: Tensor
    first_indices: Tensor
    second_indices: Tensor

    @property
    def batch_size(self) -> int:
        return int(self.memory_keys.shape[0])

    @property
    def pair_count(self) -> int:
        return int(self.memory_keys.shape[1])

    @property
    def query_count(self) -> int:
        return int(self.queries.shape[1])

    def to(self, device: torch.device) -> "TwoHopBatch":
        return TwoHopBatch(
            self.memory_keys.to(device, non_blocking=True),
            self.memory_values.to(device, non_blocking=True),
            self.queries.to(device, non_blocking=True),
            self.intermediates.to(device, non_blocking=True),
            self.targets.to(device, non_blocking=True),
            self.first_indices.to(device, non_blocking=True),
            self.second_indices.to(device, non_blocking=True),
        )


def generate_twohop_batch(
    *,
    token_pool: Tensor,
    batch_size: int,
    chain_count: int,
    query_count: int,
    seed: int,
    pool_permutation: Tensor | None = None,
) -> TwoHopBatch:
    if chain_count < 2:
        raise ValueError("chain_count must be at least two")
    if not 1 <= query_count <= chain_count:
        raise ValueError("query_count must lie in [1, chain_count]")
    pool_size = int(token_pool.numel())
    if pool_size < 3 * chain_count:
        raise ValueError("token pool must hold three distinct tokens per chain")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    priorities = torch.rand(batch_size, pool_size, generator=generator)
    tokens = priorities.topk(
        3 * chain_count,
        dim=1,
        largest=False,
        sorted=False,
    ).indices
    first = tokens[:, :chain_count]
    middle = tokens[:, chain_count : 2 * chain_count]
    last = tokens[:, 2 * chain_count :]
    local_memory_keys = torch.cat((first, middle), dim=1)
    local_memory_values = torch.cat((middle, last), dim=1)

    row_priority = torch.rand(batch_size, 2 * chain_count, generator=generator)
    row_permutation = row_priority.argsort(dim=1)
    local_memory_keys = local_memory_keys.gather(1, row_permutation)
    local_memory_values = local_memory_values.gather(1, row_permutation)
    inverse_rows = torch.empty_like(row_permutation)
    inverse_rows.scatter_(
        1,
        row_permutation,
        torch.arange(2 * chain_count).expand(batch_size, -1),
    )

    query_priority = torch.rand(batch_size, chain_count, generator=generator)
    query_chains = query_priority.topk(
        query_count,
        dim=1,
        largest=False,
        sorted=False,
    ).indices
    queries = first.gather(1, query_chains)
    intermediates = middle.gather(1, query_chains)
    targets = last.gather(1, query_chains)
    first_indices = inverse_rows.gather(1, query_chains)
    second_indices = inverse_rows.gather(1, query_chains + chain_count)

    if pool_permutation is not None:
        if pool_permutation.shape != (pool_size,):
            raise ValueError("pool permutation has the wrong shape")
        local_memory_keys = pool_permutation[local_memory_keys]
        local_memory_values = pool_permutation[local_memory_values]
        queries = pool_permutation[queries]
        intermediates = pool_permutation[intermediates]
        targets = pool_permutation[targets]
    return TwoHopBatch(
        memory_keys=token_pool[local_memory_keys],
        memory_values=token_pool[local_memory_values],
        queries=token_pool[queries],
        intermediates=token_pool[intermediates],
        targets=token_pool[targets],
        first_indices=first_indices,
        second_indices=second_indices,
    )


def ste_root_signs(coordinates: Tensor, edges: Tensor, temperature: float) -> Tensor:
    if temperature <= 0:
        raise ValueError("root STE temperature must be positive")
    differences = coordinates[..., edges[:, 0]] - coordinates[..., edges[:, 1]]
    hard = torch.where(differences > 0, 1.0, -1.0)
    soft = torch.tanh(differences / temperature)
    return hard + soft - soft.detach()


class OracleComposer(nn.Module):
    def forward(self, predecessor: Tensor, current: Tensor) -> Tensor:
        del current
        return predecessor


class CurrentComposer(nn.Module):
    def forward(self, predecessor: Tensor, current: Tensor) -> Tensor:
        del predecessor
        return current


class CanonComposer(nn.Module):
    """Depthwise two-lag Canon mixer."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.lag_logits = nn.Parameter(torch.zeros(dimension, 2))

    def forward(self, predecessor: Tensor, current: Tensor) -> Tensor:
        weights = self.lag_logits.softmax(dim=-1)
        return predecessor * weights[:, 0] + current * weights[:, 1]

    def diagnostics(self) -> dict[str, float]:
        predecessor = self.lag_logits.softmax(dim=-1)[:, 0]
        return {
            "mean_predecessor_weight": float(predecessor.mean().item()),
            "min_predecessor_weight": float(predecessor.min().item()),
            "hard_predecessor_fraction": float((predecessor > 0.5).float().mean().item()),
        }


class PCLUTComposer(nn.Module):
    """Vector PC-LUT over one causal predecessor/current row."""

    def __init__(
        self,
        dimension: int,
        *,
        tables: int,
        comparisons: int,
        init_std: float,
        seed: int,
    ) -> None:
        super().__init__()
        self.layer = PairwiseLUT(
            2 * dimension,
            dimension,
            tables=tables,
            comparisons=comparisons,
            backend="torch",
            anchor_policy="random_no_replace",
            seed=seed + 1201,
            lut_init_std=init_std,
            lut_dtype="fp32",
            fixed_zero_threshold=True,
        )

    def forward(self, predecessor: Tensor, current: Tensor) -> Tensor:
        return self.layer(torch.cat((predecessor, current), dim=-1))


def build_composer(config: ComposedRunConfig) -> nn.Module:
    if config.composer == "oracle":
        return OracleComposer()
    if config.composer == "current":
        return CurrentComposer()
    if config.composer == "canon":
        return CanonComposer(config.relation_dim)
    if config.composer == "pclut":
        return PCLUTComposer(
            config.relation_dim,
            tables=config.composer_tables,
            comparisons=config.composer_comparisons,
            init_std=config.composer_lut_init_std,
            seed=config.seed,
        )
    raise ValueError(f"unsupported composer {config.composer!r}")


def _load_frozen_relation(path: Path) -> tuple[SignedRootTransportKernel, RootTransportRunConfig]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    relation_config = RootTransportRunConfig(**checkpoint["config"])
    if relation_config.mode != "learned" or relation_config.residual_tables != 0:
        raise ValueError("composer qualification requires a learned zero-residual root relation")
    source = RootTransportMQARRetriever(relation_config)
    source.load_state_dict(checkpoint["state_dict"])
    relation = source.relation
    relation.eval()
    for parameter in relation.parameters():
        parameter.requires_grad_(False)
    return relation, relation_config


class ComposedRootRetriever(nn.Module):
    def __init__(self, config: ComposedRunConfig) -> None:
        super().__init__()
        self.config = config
        relation, relation_config = _load_frozen_relation(Path(config.relation_checkpoint))
        matched = {
            "vocab_size": config.vocab_size,
            "train_tokens": config.train_tokens,
            "validation_tokens": config.validation_tokens,
            "test_tokens": config.test_tokens,
            "relation_dim": config.relation_dim,
            "codebook_seed": config.codebook_seed,
        }
        for field, expected in matched.items():
            if getattr(relation_config, field) != expected:
                raise ValueError(f"relation checkpoint mismatch for {field}: {getattr(relation_config, field)} != {expected}")
        self.relation_config = relation_config
        self.codebook = FixedOrdinalCodebook(
            config.vocab_size,
            config.relation_dim,
            config.codebook_seed,
        )
        self.register_buffer(
            "query_permutation",
            coordinate_permutation(config.relation_dim, relation_config.query_gauge_seed),
        )
        self.register_buffer(
            "key_permutation",
            coordinate_permutation(config.relation_dim, relation_config.key_gauge_seed),
        )
        self.relation = relation
        self.composer = build_composer(config)

    def composed_coordinates(
        self,
        memory_keys: Tensor,
        memory_values: Tensor,
        *,
        force_current: bool = False,
    ) -> Tensor:
        predecessor = self.codebook(memory_keys)
        current = self.codebook(memory_values)
        if force_current:
            return current
        return self.composer(predecessor, current)

    def score_tokens(
        self,
        query_tokens: Tensor,
        memory_keys: Tensor,
        memory_values: Tensor,
        *,
        force_current: bool = False,
    ) -> Tensor:
        query_coordinates = self.codebook(query_tokens)[..., self.query_permutation]
        composed = self.composed_coordinates(
            memory_keys,
            memory_values,
            force_current=force_current,
        )
        key_coordinates = composed[..., self.key_permutation]
        query_roots = root_signs(query_coordinates, self.relation.edges)
        if self.training and key_coordinates.requires_grad:
            key_roots = ste_root_signs(
                key_coordinates,
                self.relation.edges,
                self.config.root_ste_temperature,
            )
        else:
            key_roots = root_signs(key_coordinates, self.relation.edges)
        query_active = query_roots[..., self.relation.query_root_subset]
        aligned_key = self.relation.hard_transport(key_roots)
        base = (query_active[:, :, None, :] * aligned_key[:, None, :, :]).sum(dim=-1) / math.sqrt(self.relation.root_budget)
        return self.relation.logit_scale.exp().clamp(max=100.0) * base

    @property
    def composer_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.composer.parameters())

    def composer_diagnostics(self) -> dict[str, float]:
        if isinstance(self.composer, CanonComposer):
            return self.composer.diagnostics()
        return {}


def _onehop_loss(model: ComposedRootRetriever, batch: MQARBatch) -> tuple[Tensor, dict[str, float]]:
    scores = model.score_tokens(batch.queries, batch.keys, batch.values)
    loss = target_token_loss(scores, batch.query_indices)
    return loss, {"hop1_ce": float(loss.detach().item())}


def _twohop_loss(
    model: ComposedRootRetriever,
    batch: TwoHopBatch,
) -> tuple[Tensor, dict[str, float]]:
    first_scores = model.score_tokens(
        batch.queries,
        batch.memory_keys,
        batch.memory_values,
    )
    second_scores = model.score_tokens(
        batch.intermediates,
        batch.memory_keys,
        batch.memory_values,
    )
    first_loss = F.cross_entropy(
        first_scores.flatten(0, 1),
        batch.first_indices.flatten(),
    )
    second_loss = F.cross_entropy(
        second_scores.flatten(0, 1),
        batch.second_indices.flatten(),
    )
    return 0.5 * (first_loss + second_loss), {
        "hop1_ce": float(first_loss.detach().item()),
        "hop2_ce": float(second_loss.detach().item()),
    }


def _training_batch(
    config: ComposedRunConfig,
    token_pool: Tensor,
    step: int,
) -> MQARBatch | TwoHopBatch:
    pair_count, query_count = _training_shape(step, config.data_seed)
    seed = config.data_seed + step * 1_000_003
    if config.task == "onehop":
        return generate_pool_mqar_batch(
            token_pool=token_pool,
            batch_size=config.batch_size,
            pair_count=pair_count,
            query_count=query_count,
            seed=seed,
        )
    chain_count = max(2, pair_count // 2)
    return generate_twohop_batch(
        token_pool=token_pool,
        batch_size=config.batch_size,
        chain_count=chain_count,
        query_count=min(query_count, chain_count),
        seed=seed,
    )


@torch.no_grad()
def evaluate_onehop(
    model: ComposedRootRetriever,
    *,
    token_pool: Tensor,
    pair_count: int,
    query_count: int,
    episodes: int,
    batch_size: int,
    data_seed: int,
    device: torch.device,
    force_current: bool,
    pool_permutation: Tensor | None,
) -> dict[str, float | int]:
    model.eval()
    total_queries = 0
    correct_queries = 0
    exact_episodes = 0
    ce_sum = 0.0
    for offset in range(0, episodes, batch_size):
        current_batch_size = min(batch_size, episodes - offset)
        batch = generate_pool_mqar_batch(
            token_pool=token_pool,
            batch_size=current_batch_size,
            pair_count=pair_count,
            query_count=query_count,
            seed=data_seed + offset * 1009,
            pool_permutation=pool_permutation,
        ).to(device)
        scores = model.score_tokens(
            batch.queries,
            batch.keys,
            batch.values,
            force_current=force_current,
        )
        selected = scores.argmax(dim=-1)
        correct = selected == batch.query_indices
        ce_sum += float(
            F.cross_entropy(
                scores.flatten(0, 1),
                batch.query_indices.flatten(),
                reduction="sum",
            ).item()
        )
        total_queries += current_batch_size * query_count
        correct_queries += int(correct.sum().item())
        exact_episodes += int(correct.all(dim=1).sum().item())
    return {
        "episodes": episodes,
        "queries": total_queries,
        "pair_count": pair_count,
        "query_count": query_count,
        "target_token_accuracy": correct_queries / total_queries,
        "target_token_ce": ce_sum / total_queries,
        "multiquery_exact_accuracy": exact_episodes / episodes,
        "random_token_accuracy": 1.0 / pair_count,
    }


@torch.no_grad()
def evaluate_twohop(
    model: ComposedRootRetriever,
    *,
    token_pool: Tensor,
    pair_count: int,
    query_count: int,
    episodes: int,
    batch_size: int,
    data_seed: int,
    device: torch.device,
    force_current: bool,
    pool_permutation: Tensor | None,
) -> dict[str, float | int]:
    model.eval()
    chain_count = pair_count // 2
    total_queries = 0
    first_correct = 0
    second_teacher_correct = 0
    final_correct = 0
    exact_episodes = 0
    teacher_ce_sum = 0.0
    for offset in range(0, episodes, batch_size):
        current_batch_size = min(batch_size, episodes - offset)
        batch = generate_twohop_batch(
            token_pool=token_pool,
            batch_size=current_batch_size,
            chain_count=chain_count,
            query_count=min(query_count, chain_count),
            seed=data_seed + offset * 1009,
            pool_permutation=pool_permutation,
        ).to(device)
        effective_queries = batch.query_count
        first_scores = model.score_tokens(
            batch.queries,
            batch.memory_keys,
            batch.memory_values,
            force_current=force_current,
        )
        first_selected = first_scores.argmax(dim=-1)
        selected_middle = (
            batch.memory_values[:, None, :]
            .expand(
                -1,
                effective_queries,
                -1,
            )
            .gather(2, first_selected.unsqueeze(-1))
            .squeeze(-1)
        )
        second_scores = model.score_tokens(
            selected_middle,
            batch.memory_keys,
            batch.memory_values,
            force_current=force_current,
        )
        second_selected = second_scores.argmax(dim=-1)
        predicted = (
            batch.memory_values[:, None, :]
            .expand(
                -1,
                effective_queries,
                -1,
            )
            .gather(2, second_selected.unsqueeze(-1))
            .squeeze(-1)
        )

        teacher_second_scores = model.score_tokens(
            batch.intermediates,
            batch.memory_keys,
            batch.memory_values,
            force_current=force_current,
        )
        first = first_selected == batch.first_indices
        teacher_second = teacher_second_scores.argmax(dim=-1) == batch.second_indices
        final = predicted == batch.targets
        teacher_ce_sum += float(
            F.cross_entropy(
                first_scores.flatten(0, 1),
                batch.first_indices.flatten(),
                reduction="sum",
            ).item()
            + F.cross_entropy(
                teacher_second_scores.flatten(0, 1),
                batch.second_indices.flatten(),
                reduction="sum",
            ).item()
        )
        queries = current_batch_size * effective_queries
        total_queries += queries
        first_correct += int(first.sum().item())
        second_teacher_correct += int(teacher_second.sum().item())
        final_correct += int(final.sum().item())
        exact_episodes += int(final.all(dim=1).sum().item())
    return {
        "episodes": episodes,
        "queries": total_queries,
        "pair_count": pair_count,
        "chain_count": chain_count,
        "query_count": min(query_count, chain_count),
        "hop1_accuracy": first_correct / total_queries,
        "teacher_forced_hop2_accuracy": second_teacher_correct / total_queries,
        "final_twohop_accuracy": final_correct / total_queries,
        "teacher_forced_mean_ce": teacher_ce_sum / (2 * total_queries),
        "multiquery_exact_accuracy": exact_episodes / episodes,
        "random_final_token_accuracy": 1.0 / pair_count,
    }


def _validate_config(config: ComposedRunConfig) -> None:
    if config.task not in TASKS:
        raise ValueError(f"unsupported task {config.task!r}")
    if config.composer not in COMPOSERS:
        raise ValueError(f"unsupported composer {config.composer!r}")
    if config.vocab_size != config.train_tokens + config.validation_tokens + config.test_tokens:
        raise ValueError("token counts must exactly partition the vocabulary")
    if min(config.train_tokens, config.validation_tokens, config.test_tokens) < 64:
        raise ValueError("every token pool must contain at least 64 identities")
    if config.composer in {"canon", "pclut"} and config.steps < 1:
        raise ValueError("trainable composers require positive training steps")
    if not Path(config.relation_checkpoint).is_file():
        raise ValueError(f"missing relation checkpoint {config.relation_checkpoint}")


def _build_config(args: argparse.Namespace) -> ComposedRunConfig:
    return ComposedRunConfig(
        task=args.task,
        composer=args.composer,
        seed=args.seed,
        relation_checkpoint=str(args.relation_checkpoint),
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        validation_interval=args.validation_interval,
        validation_episodes=args.validation_episodes,
        evaluation_episodes=args.evaluation_episodes,
        evaluation_batch_size=args.evaluation_batch_size,
        vocab_size=args.vocab_size,
        train_tokens=args.train_tokens,
        validation_tokens=args.validation_tokens,
        test_tokens=args.test_tokens,
        relation_dim=args.relation_dim,
        composer_tables=args.composer_tables,
        composer_comparisons=args.composer_comparisons,
        composer_lut_init_std=args.composer_lut_init_std,
        root_ste_temperature=args.root_ste_temperature,
        data_seed=args.data_seed,
        codebook_seed=args.codebook_seed,
        token_relabel_seed=args.token_relabel_seed,
        device=args.device,
    )


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    config = _build_config(args)
    _validate_config(config)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.out_dir / "result.json"
    if result_path.exists():
        existing = json.loads(result_path.read_text())
        if existing.get("complete") and existing.get("config") == asdict(config):
            print(json.dumps({"status": "skipped_complete", "result": str(result_path)}), flush=True)
            return existing
        if existing.get("complete"):
            raise ValueError(f"completed result at {result_path} has a different configuration")

    seed_everything(config.seed)
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    pools = token_pools(config)
    model = ComposedRootRetriever(config).to(device)
    trainable = [parameter for parameter in model.composer.parameters() if parameter.requires_grad]
    should_train = bool(trainable)
    optimizer = (
        torch.optim.AdamW(
            trainable,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        if should_train
        else None
    )
    best_validation_ce = float("inf")
    best_step = 0
    best_state = _clone_state_dict(model)
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()

    if should_train:
        assert optimizer is not None
        for step in range(1, config.steps + 1):
            batch = _training_batch(config, pools["train"], step).to(device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            if isinstance(batch, MQARBatch):
                loss, components = _onehop_loss(model, batch)
            else:
                loss, components = _twohop_loss(model, batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite composer loss at step {step}")
            loss.backward()
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(trainable, config.gradient_clip).item())
            optimizer.step()
            should_validate = step == 1 or step % config.validation_interval == 0 or step == config.steps
            if should_validate:
                if config.task == "onehop":
                    validation = evaluate_onehop(
                        model,
                        token_pool=pools["validation"],
                        pair_count=8,
                        query_count=4,
                        episodes=config.validation_episodes,
                        batch_size=config.evaluation_batch_size,
                        data_seed=config.data_seed + 50_000_000,
                        device=device,
                        force_current=False,
                        pool_permutation=None,
                    )
                    validation_ce = float(validation["target_token_ce"])
                    validation_accuracy = float(validation["target_token_accuracy"])
                else:
                    validation = evaluate_twohop(
                        model,
                        token_pool=pools["validation"],
                        pair_count=8,
                        query_count=4,
                        episodes=config.validation_episodes,
                        batch_size=config.evaluation_batch_size,
                        data_seed=config.data_seed + 50_000_000,
                        device=device,
                        force_current=False,
                        pool_permutation=None,
                    )
                    validation_ce = float(validation["teacher_forced_mean_ce"])
                    validation_accuracy = float(validation["final_twohop_accuracy"])
                row: dict[str, float | int] = {
                    "step": step,
                    "train_loss": float(loss.item()),
                    "gradient_norm": gradient_norm,
                    "validation_ce": validation_ce,
                    "validation_accuracy": validation_accuracy,
                    **components,
                    **model.composer_diagnostics(),
                }
                history.append(row)
                print(json.dumps(row), flush=True)
                if validation_ce < best_validation_ce:
                    best_validation_ce = validation_ce
                    best_step = step
                    best_state = _clone_state_dict(model)
    else:
        best_validation_ce = 0.0

    training_seconds = time.perf_counter() - started
    model.load_state_dict(best_state)
    test_relabel = token_permutation(config.test_tokens, config.token_relabel_seed)
    evaluation: dict[str, dict[str, dict[str, float | int]]] = {}
    for name, pair_count, query_count in EVALUATION_SPECS:
        evaluation[name] = {}
        for condition in COMPOSED_EVALUATION_CONDITIONS:
            kwargs = {
                "model": model,
                "token_pool": pools["test"],
                "pair_count": pair_count,
                "query_count": query_count,
                "episodes": config.evaluation_episodes,
                "batch_size": config.evaluation_batch_size,
                "data_seed": config.data_seed + 70_000_000 + pair_count * 100_003,
                "device": device,
                "force_current": condition == "current_only",
                "pool_permutation": test_relabel if condition == "token_relabel" else None,
            }
            if config.task == "onehop":
                evaluation[name][condition] = evaluate_onehop(**kwargs)
            else:
                evaluation[name][condition] = evaluate_twohop(**kwargs)
    result: dict[str, Any] = {
        "complete": True,
        "config": asdict(config),
        "architecture": {
            "local_row": "[predecessor token, current token]",
            "composer": config.composer,
            "relation": (f"frozen learned hard root-{model.relation.root_budget} signed transport"),
            "task": ("one-hop A->B retrieval" if config.task == "onehop" else "autoregressive A->B->C retrieval over shuffled edge rows"),
            "trainable_scope": "local composer only; codebook, role gauges, root transport, and values are fixed",
        },
        "composer_parameters": model.composer_parameters,
        "composer_diagnostics": model.composer_diagnostics(),
        "relation_correspondence_accuracy": model.relation.correspondence_accuracy(),
        "best_step": best_step,
        "best_validation_ce": best_validation_ce,
        "training": {
            "seconds": training_seconds,
            "history": history,
        },
        "evaluation": evaluation,
    }
    checkpoint = {
        "config": asdict(config),
        "best_step": best_step,
        "state_dict": best_state,
    }
    atomic_torch_save(checkpoint, args.out_dir / "best.pt")
    atomic_json_write(result, result_path)
    print(
        json.dumps(
            {
                "status": "complete",
                "result": str(result_path),
                "best_step": best_step,
            }
        ),
        flush=True,
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run one composer/task condition")
    run.add_argument("--task", choices=TASKS, required=True)
    run.add_argument("--composer", choices=COMPOSERS, required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--relation-checkpoint", type=Path, required=True)
    run.add_argument("--out-dir", type=Path, required=True)
    run.add_argument("--steps", type=int, default=1500)
    run.add_argument("--batch-size", type=int, default=128)
    run.add_argument("--learning-rate", type=float, default=0.01)
    run.add_argument("--weight-decay", type=float, default=1e-4)
    run.add_argument("--gradient-clip", type=float, default=5.0)
    run.add_argument("--validation-interval", type=int, default=100)
    run.add_argument("--validation-episodes", type=int, default=512)
    run.add_argument("--evaluation-episodes", type=int, default=2048)
    run.add_argument("--evaluation-batch-size", type=int, default=64)
    run.add_argument("--vocab-size", type=int, default=768)
    run.add_argument("--train-tokens", type=int, default=512)
    run.add_argument("--validation-tokens", type=int, default=128)
    run.add_argument("--test-tokens", type=int, default=128)
    run.add_argument("--relation-dim", type=int, default=32)
    run.add_argument("--composer-tables", type=int, default=16)
    run.add_argument("--composer-comparisons", type=int, default=4)
    run.add_argument("--composer-lut-init-std", type=float, default=0.02)
    run.add_argument("--root-ste-temperature", type=float, default=1.0)
    run.add_argument("--data-seed", type=int, default=1729)
    run.add_argument("--codebook-seed", type=int, default=2718)
    run.add_argument("--token-relabel-seed", type=int, default=314159)
    run.add_argument("--device", default="cuda")
    run.set_defaults(function=run_experiment)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
