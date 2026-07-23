"""Causally aligned multi-query associative-recall qualification.

The experiment constructs fresh key/value mappings in every episode.  A
one-token fixed shift writes the predecessor key into the row occupied by its
value:

    prefix positions:  A_j, B_j
    memory row at B_j: [predecessor=A_j, current=B_j]
    K(B_j row):        A_j
    V(B_j row):        B_j

Query and key coordinates therefore refer to the same identity while key and
value remain aligned at one causal row.  The model transports exact one-hot
values; only the relation scorer is trained.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tropnn.layers.pair_kernel import (
    BalancedS4Router,
    CoxeterPairScorer,
    GlobalChamberKernel,
    IntrinsicS4Kernel,
    RootIncidenceKernel,
    S4ObjectFeatures,
)
from tropnn.layers.pairwise import PairwiseLUT

DECODERS = (
    "no_local_dense_qk",
    "local_dense_qk",
    "local_kendall",
    "local_root_incidence",
    "local_global_coxeter",
    "local_jointpair_full",
)
CHAMBER_DECODERS = (
    "local_kendall",
    "local_root_incidence",
    "local_global_coxeter",
)
TRAIN_PAIR_COUNTS = tuple(range(4, 9))
EVALUATION_SPECS = (
    ("id_p8_q4", 8, 4),
    ("length_p16_q8", 16, 8),
    ("length_p32_q8", 32, 8),
)
LOCAL_DENSE_ACCURACY_GATE = 0.95
NO_LOCAL_EXCESS_ACCURACY_GATE = 0.05
ORDINAL_ACCURACY_GATE = 0.80
COXETER_DENSE_GAIN_RETENTION_GATE = 0.80
TOKEN_RELABEL_RETENTION_GATE = 0.95


@dataclass(frozen=True)
class MQARBatch:
    """One batch of abstract causal episodes.

    ``keys`` and ``values`` form the prefix ``A_1 B_1 ... A_P B_P``.
    ``query_indices`` select prefix mappings to query after the prefix.
    """

    keys: Tensor
    values: Tensor
    queries: Tensor
    targets: Tensor
    query_indices: Tensor

    @property
    def batch_size(self) -> int:
        return int(self.keys.shape[0])

    @property
    def pair_count(self) -> int:
        return int(self.keys.shape[1])

    @property
    def query_count(self) -> int:
        return int(self.queries.shape[1])

    def to(self, device: torch.device) -> "MQARBatch":
        return MQARBatch(
            self.keys.to(device, non_blocking=True),
            self.values.to(device, non_blocking=True),
            self.queries.to(device, non_blocking=True),
            self.targets.to(device, non_blocking=True),
            self.query_indices.to(device, non_blocking=True),
        )

    def prefix_tokens(self) -> Tensor:
        """Return the literal interleaved causal prefix."""

        return torch.stack((self.keys, self.values), dim=-1).flatten(1)

    def memory_row_positions(self) -> Tensor:
        """Return B-row positions, where both K and V are read."""

        return torch.arange(1, 2 * self.pair_count, 2, device=self.keys.device)


@dataclass(frozen=True)
class RunConfig:
    decoder: str
    seed: int
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
    relation_dim: int
    dense_rank: int
    relation_tables: int
    relation_coverage: int
    coxeter_rank: int
    jointpair_tables: int
    jointpair_comparisons: int
    data_seed: int
    codebook_seed: int
    token_relabel_seed: int
    chamber_relabel_seed: int
    device: str


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_json_write(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_torch_save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def generate_mqar_batch(
    *,
    batch_size: int,
    pair_count: int,
    query_count: int,
    vocab_size: int,
    seed: int,
    token_permutation: Tensor | None = None,
) -> MQARBatch:
    """Generate fresh injective key/value mappings with multiple queries."""

    if pair_count < 2:
        raise ValueError("pair_count must be at least two")
    if not 1 <= query_count <= pair_count:
        raise ValueError("query_count must lie in [1, pair_count]")
    if vocab_size < 2 * pair_count:
        raise ValueError("vocab_size must hold distinct keys and values")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    token_priority = torch.rand(batch_size, vocab_size, generator=generator)
    episode_tokens = token_priority.topk(2 * pair_count, dim=1, largest=False, sorted=False).indices
    keys = episode_tokens[:, :pair_count]
    values = episode_tokens[:, pair_count:]
    query_priority = torch.rand(batch_size, pair_count, generator=generator)
    query_indices = query_priority.topk(query_count, dim=1, largest=False, sorted=False).indices
    queries = keys.gather(1, query_indices)
    targets = values.gather(1, query_indices)
    if token_permutation is not None:
        if token_permutation.shape != (vocab_size,) or not torch.equal(
            token_permutation.sort().values,
            torch.arange(vocab_size),
        ):
            raise ValueError("token_permutation must be a vocabulary bijection")
        keys = token_permutation[keys]
        values = token_permutation[values]
        queries = token_permutation[queries]
        targets = token_permutation[targets]
    return MQARBatch(keys, values, queries, targets, query_indices)


def token_permutation(vocab_size: int, seed: int) -> Tensor:
    return torch.randperm(vocab_size, generator=torch.Generator(device="cpu").manual_seed(seed))


class FixedOrdinalCodebook(nn.Module):
    """Assign every token a fixed S_D chamber with no amplitude cue."""

    def __init__(self, vocab_size: int, dimension: int, seed: int) -> None:
        super().__init__()
        if dimension < 4:
            raise ValueError("dimension must be at least four")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        centered_ranks = torch.arange(dimension, dtype=torch.float32)
        centered_ranks = centered_ranks - centered_ranks.mean()
        centered_ranks = centered_ranks / centered_ranks.square().mean().sqrt()
        priorities = torch.rand(vocab_size, dimension, generator=generator)
        permutations = priorities.argsort(dim=1, stable=True)
        rows = centered_ranks[permutations]
        self.register_buffer("weight", rows)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weight[token_ids]


class DenseQK(nn.Module):
    def __init__(self, dimension: int, rank: int, seed: int) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(seed + 503)
        self.query = nn.Parameter(torch.randn(dimension, rank, generator=generator) / math.sqrt(dimension))
        self.key = nn.Parameter(torch.randn(dimension, rank, generator=generator) / math.sqrt(dimension))
        self.bias = nn.Parameter(torch.zeros(()))
        self.rank = int(rank)

    def forward(self, query: Tensor, key: Tensor) -> Tensor:
        return ((query @ self.query) * (key @ self.key)).sum(dim=-1) / math.sqrt(self.rank) + self.bias


class JointPairFullLUT(nn.Module):
    """Independent full 6-bit payload tables over the concatenated pair."""

    def __init__(self, dimension: int, tables: int, comparisons: int, seed: int) -> None:
        super().__init__()
        self.layer = PairwiseLUT(
            2 * dimension,
            1,
            tables=tables,
            comparisons=comparisons,
            backend="torch",
            anchor_policy="random_no_replace",
            seed=seed + 701,
            lut_init_std=0.02,
            lut_dtype="fp32",
            fixed_zero_threshold=True,
        )
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, query: Tensor, key: Tensor) -> Tensor:
        return self.layer(torch.cat((query, key), dim=-1)).squeeze(1).squeeze(-1) + self.bias


def _relabel_features(features: S4ObjectFeatures, relabel: Tensor) -> S4ObjectFeatures:
    tables = torch.arange(features.routes.shape[1], device=features.routes.device).view(1, -1)
    routes = relabel[tables, features.routes]
    return S4ObjectFeatures(
        coordinates=features.coordinates,
        routes=routes,
        orders=features.orders,
        adjacent_gaps=features.adjacent_gaps,
        roots=features.roots,
    )


class FixedRelabelledCoxeterScorer(nn.Module):
    """Retraining control with a fixed independent S4 label permutation per chart.

    Relation inputs are frozen, so this control intentionally omits the
    coordinate-wall STE.  Payload/factor gradients remain exact.
    """

    def __init__(self, router: BalancedS4Router, kernel: nn.Module, seed: int) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        relabel = torch.stack([torch.randperm(24, generator=generator) for _ in range(router.tables)])
        self.router = router
        self.kernel = kernel
        self.register_buffer("relabel", relabel)

    def forward(self, query: Tensor, key: Tensor) -> Tensor:
        query_features = _relabel_features(self.router.route(query), self.relabel)
        key_features = _relabel_features(self.router.route(key), self.relabel)
        return self.kernel.hard_score(query_features, key_features)


def build_relation(config: RunConfig) -> tuple[nn.Module, dict[str, Any]]:
    decoder = config.decoder
    if decoder not in DECODERS:
        raise ValueError(f"unsupported decoder {decoder!r}")
    if decoder in {"no_local_dense_qk", "local_dense_qk"}:
        relation = DenseQK(config.relation_dim, config.dense_rank, config.seed)
        return relation, {
            "execution_class": "dense QK diagnostic",
            "dense_products_per_object_pair": 2 * config.relation_dim * config.dense_rank + config.dense_rank,
            "active_lut_reads_per_pair": 0,
        }
    if decoder == "local_jointpair_full":
        relation = JointPairFullLUT(
            config.relation_dim,
            config.jointpair_tables,
            config.jointpair_comparisons,
            config.seed,
        )
        return relation, {
            "execution_class": "nonseparable joint-pair full payload LUT",
            "active_lut_reads_per_pair": config.jointpair_tables,
            "payload_rows": config.jointpair_tables * (1 << config.jointpair_comparisons),
        }

    router = BalancedS4Router(
        config.relation_dim,
        config.relation_tables,
        coverage=config.relation_coverage,
        seed=config.seed,
    )
    if decoder == "local_kendall":
        kernel: nn.Module = IntrinsicS4Kernel(config.relation_tables, "kendall")
        metadata = {
            "execution_class": "fixed intrinsic Kendall S4 kernel",
            "active_lut_reads_per_pair": config.relation_tables,
        }
    elif decoder == "local_root_incidence":
        kernel = RootIncidenceKernel(router, seed=config.seed)
        metadata = {
            "execution_class": "global comparison-root incidence operator",
            "active_lut_reads_per_pair": router.incidence_entries,
            "root_edges": router.roots,
            "root_incidence_entries": router.incidence_entries,
        }
    else:
        kernel = GlobalChamberKernel(
            config.relation_tables,
            config.coxeter_rank,
            shared_coxeter=True,
            seed=config.seed,
        )
        metadata = {
            "execution_class": "separable shared Global Coxeter chamber tower",
            "object_factor_reads": config.relation_tables * 12 * config.coxeter_rank,
            "active_pair_products": config.coxeter_rank,
            "active_lut_reads_per_pair": 0,
        }
    if config.chamber_relabel_seed >= 0:
        relation = FixedRelabelledCoxeterScorer(router, kernel, config.chamber_relabel_seed)
        metadata["chamber_relabel"] = "independent fixed 24-way permutation per table; retrained"
    else:
        relation = CoxeterPairScorer(router, kernel, symmetry="none")
        metadata["chamber_relabel"] = "none"
    metadata.update(
        {
            "chart_anchors": router.anchors.tolist(),
            "relation_tables": config.relation_tables,
            "relation_coverage": config.relation_coverage,
        }
    )
    return relation, metadata


class CausalMQARRetriever(nn.Module):
    """Fixed local write and exact value transport around one relation scorer."""

    def __init__(self, config: RunConfig) -> None:
        super().__init__()
        self.config = config
        self.codebook = FixedOrdinalCodebook(config.vocab_size, config.relation_dim, config.codebook_seed)
        self.relation, self.relation_metadata = build_relation(config)
        self.local_write = config.decoder != "no_local_dense_qk"

    def memory_coordinates(self, batch: MQARBatch) -> tuple[Tensor, Tensor, Tensor]:
        predecessor = self.codebook(batch.keys)
        current = self.codebook(batch.values)
        memory_rows = torch.cat((predecessor, current), dim=-1)
        key = memory_rows[..., : self.config.relation_dim] if self.local_write else memory_rows[..., self.config.relation_dim :]
        return memory_rows, key, batch.values

    def forward(self, batch: MQARBatch) -> tuple[Tensor, Tensor]:
        query = self.codebook(batch.queries)
        _, key, values = self.memory_coordinates(batch)
        batch_size, query_count, dimension = query.shape
        pair_count = key.shape[1]
        query_pairs = query[:, :, None, :].expand(-1, -1, pair_count, -1).reshape(-1, dimension)
        key_pairs = key[:, None, :, :].expand(-1, query_count, -1, -1).reshape(-1, dimension)
        scores = self.relation(query_pairs, key_pairs).view(batch_size, query_count, pair_count)
        return scores, values

    def predicted_tokens(self, batch: MQARBatch) -> Tensor:
        scores, values = self(batch)
        selected = scores.argmax(dim=-1)
        return values[:, None, :].expand(-1, batch.query_count, -1).gather(2, selected.unsqueeze(-1)).squeeze(-1)

    def token_probabilities(self, batch: MQARBatch) -> Tensor:
        """Transport one-hot values into a full vocabulary distribution."""

        scores, values = self(batch)
        attention = scores.softmax(dim=-1)
        probabilities = torch.zeros(
            batch.batch_size,
            batch.query_count,
            self.config.vocab_size,
            device=scores.device,
            dtype=scores.dtype,
        )
        value_ids = values[:, None, :].expand(-1, batch.query_count, -1)
        probabilities.scatter_add_(2, value_ids, attention)
        return probabilities

    @property
    def relation_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.relation.parameters())


def target_token_loss(scores: Tensor, query_indices: Tensor) -> Tensor:
    return F.cross_entropy(scores.flatten(0, 1), query_indices.flatten())


def _training_shape(step: int, data_seed: int) -> tuple[int, int]:
    generator = torch.Generator(device="cpu").manual_seed(data_seed + step * 104729)
    pair_count = int(torch.randint(4, 9, (1,), generator=generator).item())
    query_count = int(torch.randint(1, min(4, pair_count) + 1, (1,), generator=generator).item())
    return pair_count, query_count


@torch.no_grad()
def evaluate(
    model: CausalMQARRetriever,
    *,
    pair_count: int,
    query_count: int,
    episodes: int,
    batch_size: int,
    data_seed: int,
    device: torch.device,
    permutation: Tensor | None = None,
) -> dict[str, float | int]:
    model.eval()
    total_queries = 0
    correct_queries = 0
    top4_queries = 0
    nll_sum = 0.0
    total_episodes = 0
    exact_episodes = 0
    target_probability_sum = 0.0
    for offset in range(0, episodes, batch_size):
        current_batch_size = min(batch_size, episodes - offset)
        batch = generate_mqar_batch(
            batch_size=current_batch_size,
            pair_count=pair_count,
            query_count=query_count,
            vocab_size=model.config.vocab_size,
            seed=data_seed + offset * 1009,
            token_permutation=permutation,
        ).to(device)
        scores, values = model(batch)
        selected = scores.argmax(dim=-1)
        predicted = values[:, None, :].expand(-1, query_count, -1).gather(2, selected.unsqueeze(-1)).squeeze(-1)
        correct = predicted == batch.targets
        order = scores.topk(min(4, pair_count), dim=-1).indices
        top4 = (order == batch.query_indices.unsqueeze(-1)).any(dim=-1)
        log_probability = scores.log_softmax(dim=-1).gather(2, batch.query_indices.unsqueeze(-1)).squeeze(-1)
        total_queries += current_batch_size * query_count
        correct_queries += int(correct.sum().item())
        top4_queries += int(top4.sum().item())
        nll_sum -= float(log_probability.sum().item())
        target_probability_sum += float(log_probability.exp().sum().item())
        total_episodes += current_batch_size
        exact_episodes += int(correct.all(dim=1).sum().item())
    return {
        "episodes": total_episodes,
        "queries": total_queries,
        "pair_count": pair_count,
        "query_count": query_count,
        "target_token_accuracy": correct_queries / total_queries,
        "target_token_top4_accuracy": top4_queries / total_queries,
        "target_token_ce": nll_sum / total_queries,
        "target_token_probability": target_probability_sum / total_queries,
        "multiquery_exact_accuracy": exact_episodes / total_episodes,
        "random_token_accuracy": 1.0 / pair_count,
        "random_multiquery_exact_accuracy": (1.0 / pair_count) ** query_count,
    }


@torch.no_grad()
def benchmark_forward(
    model: CausalMQARRetriever,
    *,
    device: torch.device,
    batch_size: int,
    pair_count: int,
    query_count: int,
    data_seed: int,
    warmups: int = 10,
    iterations: int = 30,
) -> dict[str, float | int | str]:
    model.eval()
    batch = generate_mqar_batch(
        batch_size=batch_size,
        pair_count=pair_count,
        query_count=query_count,
        vocab_size=model.config.vocab_size,
        seed=data_seed,
    ).to(device)

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    for _ in range(warmups):
        model(batch)
    synchronize()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        baseline_memory = torch.cuda.memory_allocated(device)
    else:
        baseline_memory = 0
    started = time.perf_counter()
    for _ in range(iterations):
        model(batch)
    synchronize()
    elapsed = time.perf_counter() - started
    calls = iterations * batch_size
    pair_scores = calls * pair_count * query_count
    peak_increment = 0 if device.type != "cuda" else max(0, torch.cuda.max_memory_allocated(device) - baseline_memory)
    return {
        "mode": "warm_forward_only",
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "dtype": str(model.codebook.weight.dtype),
        "batch_size": batch_size,
        "pair_count": pair_count,
        "query_count": query_count,
        "warmups": warmups,
        "iterations": iterations,
        "episodes_per_second": calls / elapsed,
        "query_tokens_per_second": calls * query_count / elapsed,
        "pair_scores_per_second": pair_scores / elapsed,
        "peak_increment_bytes": peak_increment,
    }


def _clone_state_dict(model: nn.Module) -> dict[str, Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _validate_config(config: RunConfig) -> None:
    if config.decoder not in DECODERS:
        raise ValueError(f"unsupported decoder {config.decoder!r}")
    if config.chamber_relabel_seed >= 0 and config.decoder not in CHAMBER_DECODERS:
        raise ValueError("chamber relabeling is defined only for S4 chamber decoders")
    if 4 * config.relation_tables != config.relation_coverage * config.relation_dim:
        raise ValueError("relation charts must form a balanced cover")
    if config.steps < 1 or config.batch_size < 1:
        raise ValueError("steps and batch_size must be positive")
    if config.validation_interval < 1 or config.validation_episodes < 1 or config.evaluation_episodes < 1:
        raise ValueError("evaluation settings must be positive")


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    config = RunConfig(
        decoder=args.decoder,
        seed=args.seed,
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
        relation_dim=args.relation_dim,
        dense_rank=args.dense_rank,
        relation_tables=args.relation_tables,
        relation_coverage=args.relation_coverage,
        coxeter_rank=args.coxeter_rank,
        jointpair_tables=args.jointpair_tables,
        jointpair_comparisons=args.jointpair_comparisons,
        data_seed=args.data_seed,
        codebook_seed=args.codebook_seed,
        token_relabel_seed=args.token_relabel_seed,
        chamber_relabel_seed=args.chamber_relabel_seed,
        device=args.device,
    )
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
    model = CausalMQARRetriever(config).to(device)
    optimizer = torch.optim.AdamW(
        model.relation.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    best_validation_ce = float("inf")
    best_step = 0
    best_state: dict[str, Tensor] | None = None
    history: list[dict[str, float | int]] = []
    training_queries = 0
    training_pair_scores = 0
    started = time.perf_counter()
    for step in range(1, config.steps + 1):
        pair_count, query_count = _training_shape(step, config.data_seed)
        batch = generate_mqar_batch(
            batch_size=config.batch_size,
            pair_count=pair_count,
            query_count=query_count,
            vocab_size=config.vocab_size,
            seed=config.data_seed + step * 1_000_003,
        ).to(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        scores, _ = model(batch)
        loss = target_token_loss(scores, batch.query_indices)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite training loss at step {step}")
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.relation.parameters(), config.gradient_clip).item())
        optimizer.step()
        training_queries += config.batch_size * query_count
        training_pair_scores += config.batch_size * query_count * pair_count

        should_validate = step == 1 or step % config.validation_interval == 0 or step == config.steps
        if should_validate:
            validation = evaluate(
                model,
                pair_count=8,
                query_count=4,
                episodes=config.validation_episodes,
                batch_size=config.evaluation_batch_size,
                data_seed=config.data_seed + 50_000_000,
                device=device,
            )
            row = {
                "step": step,
                "train_target_token_ce": float(loss.item()),
                "gradient_norm": gradient_norm,
                "validation_target_token_ce": float(validation["target_token_ce"]),
                "validation_target_token_accuracy": float(validation["target_token_accuracy"]),
                "validation_multiquery_exact_accuracy": float(validation["multiquery_exact_accuracy"]),
            }
            history.append(row)
            print(json.dumps(row), flush=True)
            if float(validation["target_token_ce"]) < best_validation_ce:
                best_validation_ce = float(validation["target_token_ce"])
                best_step = step
                best_state = _clone_state_dict(model)

    if best_state is None:
        raise RuntimeError("training produced no validation checkpoint")
    training_seconds = time.perf_counter() - started
    model.load_state_dict(best_state)
    permutation = token_permutation(config.vocab_size, config.token_relabel_seed)
    evaluation: dict[str, dict[str, dict[str, float | int]]] = {}
    for name, pair_count, query_count in EVALUATION_SPECS:
        evaluation[name] = {
            "base": evaluate(
                model,
                pair_count=pair_count,
                query_count=query_count,
                episodes=config.evaluation_episodes,
                batch_size=config.evaluation_batch_size,
                data_seed=config.data_seed + 70_000_000 + pair_count * 100_003,
                device=device,
            ),
            "token_relabel": evaluate(
                model,
                pair_count=pair_count,
                query_count=query_count,
                episodes=config.evaluation_episodes,
                batch_size=config.evaluation_batch_size,
                data_seed=config.data_seed + 70_000_000 + pair_count * 100_003,
                device=device,
                permutation=permutation,
            ),
        }
    benchmark = benchmark_forward(
        model,
        device=device,
        batch_size=min(config.evaluation_batch_size, 64),
        pair_count=32,
        query_count=8,
        data_seed=config.data_seed + 90_000_000,
    )
    result: dict[str, Any] = {
        "complete": True,
        "config": asdict(config),
        "architecture": {
            "prefix": "A_1 B_1 ... A_P B_P followed by one or more A queries",
            "local_write": "fixed one-token shift-concat creates [predecessor=A, current=B] at the B row",
            "key": "first lane A from the B row" if model.local_write else "current-token lane B from the B row",
            "value": "current token B from the same B row as K",
            "query": "fixed ordinal code of the current A",
            "transport": "softmax relation scores over B rows followed by exact one-hot B value transport",
            "trainable_scope": "relation scorer only; token codebook, local write, K selector, and values are fixed",
        },
        "relation_parameters": model.relation_parameters,
        "full_trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "fixed_codebook_values": int(model.codebook.weight.numel()),
        "relation_metadata": model.relation_metadata,
        "best_step": best_step,
        "best_validation_target_token_ce": best_validation_ce,
        "training": {
            "seconds": training_seconds,
            "queries": training_queries,
            "pair_scores": training_pair_scores,
            "queries_per_second": training_queries / training_seconds,
            "pair_scores_per_second": training_pair_scores / training_seconds,
            "history": history,
        },
        "evaluation": evaluation,
        "benchmark": benchmark,
    }
    checkpoint = {
        "config": asdict(config),
        "best_step": best_step,
        "state_dict": best_state,
    }
    atomic_torch_save(checkpoint, args.out_dir / "best.pt")
    atomic_json_write(result, result_path)
    print(json.dumps({"status": "complete", "result": str(result_path), "best_step": best_step}), flush=True)
    return result


def _mean_sem(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    mean = statistics.fmean(values)
    sem = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return mean, sem


def _format_mean_sem(values: list[float], digits: int = 4) -> str:
    mean, sem = _mean_sem(values)
    return f"{mean:.{digits}f} +/- {sem:.{digits}f}"


def _collect_results(result_dir: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for path in sorted(result_dir.rglob("result.json")):
        value = json.loads(path.read_text())
        if value.get("complete"):
            value["_path"] = str(path)
            results.append(value)
    return results


def _metric(results: list[dict[str, Any]], decoder: str, spec: str, condition: str, metric: str, *, chamber: bool) -> list[float]:
    return [
        float(result["evaluation"][spec][condition][metric])
        for result in results
        if result["config"]["decoder"] == decoder and (int(result["config"]["chamber_relabel_seed"]) >= 0) == chamber
    ]


def _write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    results = _collect_results(args.result_dir)
    expected_seeds = tuple(args.expected_seeds)
    expected_base = {(decoder, seed) for decoder in DECODERS for seed in expected_seeds}
    found_base = {
        (result["config"]["decoder"], int(result["config"]["seed"]))
        for result in results
        if int(result["config"]["chamber_relabel_seed"]) < 0
    }
    expected_chamber = {(decoder, seed) for decoder in CHAMBER_DECODERS for seed in expected_seeds}
    found_chamber = {
        (result["config"]["decoder"], int(result["config"]["seed"]))
        for result in results
        if int(result["config"]["chamber_relabel_seed"]) >= 0
    }
    missing_base = sorted(expected_base - found_base)
    missing_chamber = sorted(expected_chamber - found_chamber)
    if missing_base or missing_chamber:
        raise RuntimeError(f"incomplete result matrix: missing_base={missing_base}, missing_chamber={missing_chamber}")

    spec = "length_p32_q8"
    chance = 1.0 / 32.0
    base_accuracy = {
        decoder: _mean_sem(_metric(results, decoder, spec, "base", "target_token_accuracy", chamber=False))[0]
        for decoder in DECODERS
    }
    token_accuracy = {
        decoder: _mean_sem(_metric(results, decoder, spec, "token_relabel", "target_token_accuracy", chamber=False))[0]
        for decoder in DECODERS
    }
    dense_gain = base_accuracy["local_dense_qk"] - chance
    coxeter_gain = base_accuracy["local_global_coxeter"] - chance
    coxeter_retention = coxeter_gain / dense_gain if dense_gain > 0 else float("nan")
    token_relabel_retention = {
        decoder: (token_accuracy[decoder] - chance) / max(base_accuracy[decoder] - chance, 1e-12)
        for decoder in DECODERS
    }
    gates = {
        "local_dense_succeeds": {
            "value": base_accuracy["local_dense_qk"],
            "threshold": LOCAL_DENSE_ACCURACY_GATE,
            "passed": base_accuracy["local_dense_qk"] >= LOCAL_DENSE_ACCURACY_GATE,
        },
        "no_local_dense_is_negative": {
            "value": base_accuracy["no_local_dense_qk"],
            "threshold": chance + NO_LOCAL_EXCESS_ACCURACY_GATE,
            "passed": base_accuracy["no_local_dense_qk"] <= chance + NO_LOCAL_EXCESS_ACCURACY_GATE,
        },
        "global_coxeter_retains_dense_gain": {
            "value": coxeter_retention,
            "threshold": COXETER_DENSE_GAIN_RETENTION_GATE,
            "passed": coxeter_retention >= COXETER_DENSE_GAIN_RETENTION_GATE,
        },
        "global_coxeter_token_relabel": {
            "value": token_relabel_retention["local_global_coxeter"],
            "threshold": TOKEN_RELABEL_RETENTION_GATE,
            "passed": token_relabel_retention["local_global_coxeter"] >= TOKEN_RELABEL_RETENTION_GATE,
        },
    }
    qualified = bool(gates["local_dense_succeeds"]["passed"] and gates["no_local_dense_is_negative"]["passed"])
    ordinal_success = {
        decoder: base_accuracy[decoder] >= ORDINAL_ACCURACY_GATE
        for decoder in ("local_kendall", "local_root_incidence", "local_global_coxeter", "local_jointpair_full")
    }
    if not qualified:
        interpretation = "invalid_relation_comparison"
    elif gates["global_coxeter_retains_dense_gain"]["passed"] and gates["global_coxeter_token_relabel"]["passed"]:
        interpretation = "global_coxeter_routes_causal_mqar"
    elif ordinal_success["local_jointpair_full"] and not ordinal_success["local_global_coxeter"]:
        interpretation = "joint_lut_succeeds_but_coxeter_sharing_fails"
    elif not any(ordinal_success.values()):
        interpretation = "tested_chamber_quotients_lose_required_identity_signal"
    else:
        interpretation = "mixed_ordinal_result"

    rows: list[dict[str, object]] = []
    for decoder in DECODERS:
        sample = next(
            result
            for result in results
            if result["config"]["decoder"] == decoder and int(result["config"]["chamber_relabel_seed"]) < 0
        )
        rows.append(
            {
                "decoder": decoder,
                "relation_parameters": int(sample["relation_parameters"]),
                "target_accuracy_p8": _mean_sem(
                    _metric(results, decoder, "id_p8_q4", "base", "target_token_accuracy", chamber=False)
                )[0],
                "exact_p8": _mean_sem(
                    _metric(results, decoder, "id_p8_q4", "base", "multiquery_exact_accuracy", chamber=False)
                )[0],
                "target_accuracy_p16": _mean_sem(
                    _metric(results, decoder, "length_p16_q8", "base", "target_token_accuracy", chamber=False)
                )[0],
                "target_accuracy_p32": base_accuracy[decoder],
                "exact_p32": _mean_sem(
                    _metric(results, decoder, spec, "base", "multiquery_exact_accuracy", chamber=False)
                )[0],
                "token_relabel_accuracy_p32": token_accuracy[decoder],
                "pair_scores_per_second": _mean_sem(
                    [
                        float(result["benchmark"]["pair_scores_per_second"])
                        for result in results
                        if result["config"]["decoder"] == decoder and int(result["config"]["chamber_relabel_seed"]) < 0
                    ]
                )[0],
            }
        )
    _write_summary_csv(args.out_report.with_suffix(".csv"), rows)

    lines = [
        "# Causally Aligned MQAR Induction",
        "",
        "## Question",
        "",
        "Can an ordinal relation kernel replace dense QK after a causal local write "
        "has already placed the predecessor key and payload value in the same memory row?",
        "",
        "Every episode draws a fresh injective mapping. The prefix is `A_1 B_1 ... A_P B_P`; "
        "a fixed one-token shift constructs `[A_j, B_j]` at the `B_j` row. "
        "K reads the `A_j` lane and V is the `B_j` token from that same row. Only the relation scorer is trainable.",
        "",
        "Training samples 4-8 mappings and 1-4 queries. Evaluation uses held episodes "
        "with 8, 16, and 32 mappings; the 16/32 conditions are length extrapolation. "
        "The token-relabel control applies one fixed vocabulary bijection consistently "
        "to the same abstract episodes. S4 chamber controls retrain after independent "
        "per-table 24-way label permutations.",
        "",
        "## Main results",
        "",
        "| Decoder | Relation params | P8 target acc | P8 all-query exact | "
        "P16 target acc | P32 target acc | P32 all-query exact | "
        "P32 token-relabel acc | Pair scores/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        decoder = str(row["decoder"])
        lines.append(
            f"| {decoder} | {int(row['relation_parameters']):,} | "
            f"{_format_mean_sem(_metric(results, decoder, 'id_p8_q4', 'base', 'target_token_accuracy', chamber=False))} | "
            f"{_format_mean_sem(_metric(results, decoder, 'id_p8_q4', 'base', 'multiquery_exact_accuracy', chamber=False))} | "
            f"{_format_mean_sem(_metric(results, decoder, 'length_p16_q8', 'base', 'target_token_accuracy', chamber=False))} | "
            f"{_format_mean_sem(_metric(results, decoder, spec, 'base', 'target_token_accuracy', chamber=False))} | "
            f"{_format_mean_sem(_metric(results, decoder, spec, 'base', 'multiquery_exact_accuracy', chamber=False))} | "
            f"{_format_mean_sem(_metric(results, decoder, spec, 'token_relabel', 'target_token_accuracy', chamber=False))} | "
            f"{float(row['pair_scores_per_second']):,.0f} |"
        )
    lines.extend(
        [
            "",
            "Random target-token accuracy is `0.125`, `0.0625`, and `0.03125` for P8, P16, and P32. "
            "All-query exact accuracy is the fraction of episodes in which every query retrieves its target value.",
            "",
            "## Chamber-label retraining control",
            "",
            "| Decoder | Original P32 target acc | Relabeled-and-retrained P32 target acc | Delta |",
            "|---|---:|---:|---:|",
        ]
    )
    for decoder in CHAMBER_DECODERS:
        base_values = _metric(results, decoder, spec, "base", "target_token_accuracy", chamber=False)
        relabeled_values = _metric(results, decoder, spec, "base", "target_token_accuracy", chamber=True)
        deltas = [new - old for old, new in zip(base_values, relabeled_values, strict=True)]
        lines.append(
            f"| {decoder} | {_format_mean_sem(base_values)} | {_format_mean_sem(relabeled_values)} | {_format_mean_sem(deltas)} |"
        )
    lines.extend(
        [
            "",
            "Root-incidence uses globally labeled comparison signs, so arbitrary S4 route labels "
            "are not part of its score. Kendall and Shared Global Coxeter do use S4 route labels; "
            "their relabel controls deliberately test whether their group structure, rather than "
            "only categorical capacity, matters.",
            "",
            "## Preregistered qualification",
            "",
        ]
    )
    for name, gate in gates.items():
        lines.append(
            f"- `{name}`: value `{float(gate['value']):.4f}`, threshold `{float(gate['threshold']):.4f}`, passed `{bool(gate['passed'])}`."
        )
    lines.extend(
        [
            "",
            f"Decision: `{interpretation}`.",
            "",
            "The Dense control must first reach at least 0.95 P32 target accuracy while "
            "No-local Dense remains below chance plus 0.05. Only then is a relation-kernel "
            "comparison qualified. Global Coxeter passes its positive gate only if it retains "
            "at least 80% of Dense's chance-adjusted P32 gain "
            "and at least 95% of its own gain after token relabeling.",
            "",
            "## Scope",
            "",
            "This is a causal content-addressing qualification, not a language-model result. "
            "It separates local predecessor construction, global relation scoring, and exact "
            "value transport. Passing establishes that a scorer can route a standard induction/MQAR "
            "primitive on ordinal token codes; it does not establish multi-hop composition, "
            "learned language states, or end-to-end LM quality.",
            "",
            "Dense QK remains a diagnostic GEMM path. Global Coxeter retains a width-12 product "
            "in this reference implementation, and Root-incidence uses sparse signed accumulation. "
            "Throughput is warm forward-only Torch timing at the matched P32/Q8 shape; "
            "it is not a fused-kernel claim.",
            "",
        ]
    )
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text("\n".join(lines))
    decision = {
        "complete": True,
        "expected_seeds": list(expected_seeds),
        "result_count": len(results),
        "qualified_relation_comparison": qualified,
        "ordinal_success": ordinal_success,
        "gates": gates,
        "interpretation": interpretation,
        "report": str(args.out_report),
        "summary_csv": str(args.out_report.with_suffix(".csv")),
    }
    atomic_json_write(decision, args.out_report.with_suffix(".decision.json"))
    print(json.dumps(decision, indent=2, sort_keys=True), flush=True)
    return decision


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="train and evaluate one decoder/seed")
    run.add_argument("--decoder", choices=DECODERS, required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--out-dir", type=Path, required=True)
    run.add_argument("--steps", type=int, default=1500)
    run.add_argument("--batch-size", type=int, default=256)
    run.add_argument("--learning-rate", type=float, default=0.01)
    run.add_argument("--weight-decay", type=float, default=1e-4)
    run.add_argument("--gradient-clip", type=float, default=5.0)
    run.add_argument("--validation-interval", type=int, default=100)
    run.add_argument("--validation-episodes", type=int, default=512)
    run.add_argument("--evaluation-episodes", type=int, default=2048)
    run.add_argument("--evaluation-batch-size", type=int, default=64)
    run.add_argument("--vocab-size", type=int, default=512)
    run.add_argument("--relation-dim", type=int, default=32)
    run.add_argument("--dense-rank", type=int, default=16)
    run.add_argument("--relation-tables", type=int, default=16)
    run.add_argument("--relation-coverage", type=int, default=2)
    run.add_argument("--coxeter-rank", type=int, default=12)
    run.add_argument("--jointpair-tables", type=int, default=144)
    run.add_argument("--jointpair-comparisons", type=int, default=6)
    run.add_argument("--data-seed", type=int, default=1729)
    run.add_argument("--codebook-seed", type=int, default=2718)
    run.add_argument("--token-relabel-seed", type=int, default=314159)
    run.add_argument("--chamber-relabel-seed", type=int, default=-1)
    run.add_argument("--device", default="cuda")
    run.set_defaults(function=run_experiment)

    summary = subparsers.add_parser("summarize", help="validate the matrix and write its report")
    summary.add_argument("--result-dir", type=Path, required=True)
    summary.add_argument("--out-report", type=Path, required=True)
    summary.add_argument("--expected-seeds", type=int, nargs="+", default=(0, 1, 2))
    summary.set_defaults(function=summarize)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
