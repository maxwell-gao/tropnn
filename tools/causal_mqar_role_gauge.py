"""Causal MQAR with role-specific ordinal query/key gauges.

The previous causal MQAR qualification gives query A and the predecessor lane
at the B memory row the identical ordinal code.  That makes the relation
problem chamber equality.  This experiment removes that shortcut:

    q(x) = x[P_Q]
    k(x) = x[P_K],  P_Q != P_K

where the coordinate permutations are fixed, target-free, and shared across
all tokens.  Train, validation, and test token identities are disjoint.  The
only trainable component is the relation scorer; the causal local write and
exact B-value transport remain fixed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from tropnn.layers.pair_kernel import BalancedS4Router
from tropnn.tools.causal_mqar_induction import (
    CHAMBER_DECODERS,
    DECODERS,
    EVALUATION_SPECS,
    FixedOrdinalCodebook,
    MQARBatch,
    _clone_state_dict,
    _training_shape,
    atomic_json_write,
    atomic_torch_save,
    build_relation,
    generate_mqar_batch,
    seed_everything,
    target_token_loss,
    token_permutation,
)

GAUGE_MODES = ("identity", "role_permutation")
IDENTITY_CONTROL_DECODERS = ("local_dense_qk", "local_kendall")
ROLE_CONDITIONS = ("base", "token_relabel", "wrong_key_gauge", "role_swap")
IDENTITY_CONDITIONS = ("base", "token_relabel")

IDENTITY_ACCURACY_GATE = 0.95
LOCAL_DENSE_ACCURACY_GATE = 0.95
NO_LOCAL_EXCESS_ACCURACY_GATE = 0.05
GLOBAL_DENSE_GAIN_RETENTION_GATE = 0.80
TOKEN_RELABEL_RETENTION_GATE = 0.95
WRONG_GAUGE_GAIN_REMOVAL_GATE = 0.80
ORDINAL_ACCURACY_GATE = 0.80
ROUTE_IDENTITY_CEILING = 0.25


@dataclass(frozen=True)
class RoleGaugeRunConfig:
    decoder: str
    gauge_mode: str
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
    train_tokens: int
    validation_tokens: int
    test_tokens: int
    relation_dim: int
    dense_rank: int
    relation_tables: int
    relation_coverage: int
    coxeter_rank: int
    jointpair_tables: int
    jointpair_comparisons: int
    data_seed: int
    codebook_seed: int
    query_gauge_seed: int
    key_gauge_seed: int
    wrong_key_gauge_seed: int
    token_relabel_seed: int
    chamber_relabel_seed: int
    device: str


def coordinate_permutation(dimension: int, seed: int) -> Tensor:
    return torch.randperm(dimension, generator=torch.Generator(device="cpu").manual_seed(seed))


def token_pools(config: RoleGaugeRunConfig) -> dict[str, Tensor]:
    train_end = config.train_tokens
    validation_end = train_end + config.validation_tokens
    test_end = validation_end + config.test_tokens
    return {
        "train": torch.arange(0, train_end),
        "validation": torch.arange(train_end, validation_end),
        "test": torch.arange(validation_end, test_end),
    }


def generate_pool_mqar_batch(
    *,
    token_pool: Tensor,
    batch_size: int,
    pair_count: int,
    query_count: int,
    seed: int,
    pool_permutation: Tensor | None = None,
) -> MQARBatch:
    """Generate an episode using only identities from one disjoint token pool."""

    local = generate_mqar_batch(
        batch_size=batch_size,
        pair_count=pair_count,
        query_count=query_count,
        vocab_size=int(token_pool.numel()),
        seed=seed,
        token_permutation=pool_permutation,
    )
    return MQARBatch(
        keys=token_pool[local.keys],
        values=token_pool[local.values],
        queries=token_pool[local.queries],
        targets=token_pool[local.targets],
        query_indices=local.query_indices,
    )


class RoleGaugeMQARRetriever(nn.Module):
    """Fixed causal write, role gauges, relation scorer, and exact V transport."""

    def __init__(self, config: RoleGaugeRunConfig) -> None:
        super().__init__()
        self.config = config
        self.codebook = FixedOrdinalCodebook(config.vocab_size, config.relation_dim, config.codebook_seed)
        self.relation, self.relation_metadata = build_relation(config)
        self.local_write = config.decoder != "no_local_dense_qk"

        identity = torch.arange(config.relation_dim)
        if config.gauge_mode == "identity":
            query = identity
            key = identity
        else:
            query = coordinate_permutation(config.relation_dim, config.query_gauge_seed)
            key = coordinate_permutation(config.relation_dim, config.key_gauge_seed)
            if torch.equal(query, key):
                raise ValueError("query and key gauge permutations must differ")
        wrong_key = coordinate_permutation(config.relation_dim, config.wrong_key_gauge_seed)
        if torch.equal(wrong_key, key) or torch.equal(wrong_key, query):
            wrong_key = wrong_key.roll(1)
        inverse_query = torch.argsort(query)
        relative = inverse_query[key]
        self.register_buffer("query_permutation", query)
        self.register_buffer("key_permutation", key)
        self.register_buffer("wrong_key_permutation", wrong_key)
        self.register_buffer("relative_query_to_key", relative)

    @staticmethod
    def _apply_gauge(coordinates: Tensor, permutation: Tensor) -> Tensor:
        return coordinates[..., permutation]

    def memory_coordinates(
        self,
        batch: MQARBatch,
        *,
        key_permutation: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        predecessor = self.codebook(batch.keys)
        current = self.codebook(batch.values)
        memory_rows = torch.cat((predecessor, current), dim=-1)
        raw_key = predecessor if self.local_write else current
        permutation = self.key_permutation if key_permutation is None else key_permutation
        return memory_rows, self._apply_gauge(raw_key, permutation), batch.values

    def forward(
        self,
        batch: MQARBatch,
        *,
        query_permutation: Tensor | None = None,
        key_permutation: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        query_coordinates = self.codebook(batch.queries)
        query_gauge = self.query_permutation if query_permutation is None else query_permutation
        query = self._apply_gauge(query_coordinates, query_gauge)
        _, key, values = self.memory_coordinates(batch, key_permutation=key_permutation)
        batch_size, query_count, dimension = query.shape
        pair_count = key.shape[1]
        query_pairs = query[:, :, None, :].expand(-1, -1, pair_count, -1).reshape(-1, dimension)
        key_pairs = key[:, None, :, :].expand(-1, query_count, -1, -1).reshape(-1, dimension)
        scores = self.relation(query_pairs, key_pairs).view(batch_size, query_count, pair_count)
        return scores, values

    @property
    def relation_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.relation.parameters())


def _condition_permutations(model: RoleGaugeMQARRetriever, condition: str) -> tuple[Tensor, Tensor]:
    if condition in {"base", "token_relabel"}:
        return model.query_permutation, model.key_permutation
    if condition == "wrong_key_gauge":
        return model.query_permutation, model.wrong_key_permutation
    if condition == "role_swap":
        return model.key_permutation, model.query_permutation
    raise ValueError(f"unsupported evaluation condition {condition!r}")


@torch.no_grad()
def evaluate(
    model: RoleGaugeMQARRetriever,
    *,
    token_pool: Tensor,
    pair_count: int,
    query_count: int,
    episodes: int,
    batch_size: int,
    data_seed: int,
    device: torch.device,
    condition: str = "base",
    pool_permutation: Tensor | None = None,
) -> dict[str, float | int]:
    model.eval()
    query_gauge, key_gauge = _condition_permutations(model, condition)
    total_queries = 0
    correct_queries = 0
    top4_queries = 0
    nll_sum = 0.0
    total_episodes = 0
    exact_episodes = 0
    target_probability_sum = 0.0
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
        scores, values = model(
            batch,
            query_permutation=query_gauge,
            key_permutation=key_gauge,
        )
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
def gauge_diagnostics(
    model: RoleGaugeMQARRetriever,
    *,
    token_pool: Tensor,
    device: torch.device,
) -> dict[str, float | int | list[int]]:
    token_ids = token_pool.to(device)
    coordinates = model.codebook(token_ids)
    query = model._apply_gauge(coordinates, model.query_permutation)
    key = model._apply_gauge(coordinates, model.key_permutation)
    random_key = key.roll(1, dims=0)
    edges = torch.triu_indices(model.config.relation_dim, model.config.relation_dim, offset=1, device=device)

    def signs(value: Tensor) -> Tensor:
        return value[:, edges[0]] > value[:, edges[1]]

    query_signs = signs(query)
    key_signs = signs(key)
    random_signs = signs(random_key)
    router = BalancedS4Router(
        model.config.relation_dim,
        model.config.relation_tables,
        coverage=model.config.relation_coverage,
        seed=model.config.seed,
    ).to(device)
    query_routes = router.route(query).routes
    key_routes = router.route(key).routes
    random_routes = router.route(random_key).routes
    reconstructed_key = query[..., model.relative_query_to_key]
    return {
        "query_permutation": model.query_permutation.detach().cpu().tolist(),
        "key_permutation": model.key_permutation.detach().cpu().tolist(),
        "relative_query_to_key": model.relative_query_to_key.detach().cpu().tolist(),
        "relative_fixed_coordinates": int((model.query_permutation == model.key_permutation).sum().item()),
        "positive_coordinate_vector_equality": float((query == key).all(dim=-1).float().mean().item()),
        "positive_coordinate_position_agreement": float((query == key).float().mean().item()),
        "relative_map_reconstruction_max_error": float((reconstructed_key - key).abs().max().item()),
        "positive_full_root_agreement": float((query_signs == key_signs).float().mean().item()),
        "random_full_root_agreement": float((query_signs == random_signs).float().mean().item()),
        "positive_s4_table_route_agreement": float((query_routes == key_routes).float().mean().item()),
        "random_s4_table_route_agreement": float((query_routes == random_routes).float().mean().item()),
        "positive_full_route_vector_equality": float((query_routes == key_routes).all(dim=-1).float().mean().item()),
    }


@torch.no_grad()
def benchmark_forward(
    model: RoleGaugeMQARRetriever,
    *,
    token_pool: Tensor,
    device: torch.device,
    batch_size: int,
    pair_count: int,
    query_count: int,
    data_seed: int,
    warmups: int = 10,
    iterations: int = 30,
) -> dict[str, float | int | str]:
    model.eval()
    batch = generate_pool_mqar_batch(
        token_pool=token_pool,
        batch_size=batch_size,
        pair_count=pair_count,
        query_count=query_count,
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
    peak_increment = 0
    if device.type == "cuda":
        peak_increment = max(0, torch.cuda.max_memory_allocated(device) - baseline_memory)
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


def _validate_config(config: RoleGaugeRunConfig) -> None:
    if config.decoder not in DECODERS:
        raise ValueError(f"unsupported decoder {config.decoder!r}")
    if config.gauge_mode not in GAUGE_MODES:
        raise ValueError(f"unsupported gauge mode {config.gauge_mode!r}")
    if config.gauge_mode == "identity" and config.decoder not in IDENTITY_CONTROL_DECODERS:
        raise ValueError("identity gauge is restricted to the Dense and Kendall qualification controls")
    if config.chamber_relabel_seed >= 0 and config.decoder not in CHAMBER_DECODERS:
        raise ValueError("chamber relabeling is defined only for S4 chamber decoders")
    if config.gauge_mode == "identity" and config.chamber_relabel_seed >= 0:
        raise ValueError("identity controls do not use chamber relabeling")
    if 4 * config.relation_tables != config.relation_coverage * config.relation_dim:
        raise ValueError("relation charts must form a balanced cover")
    if config.vocab_size != config.train_tokens + config.validation_tokens + config.test_tokens:
        raise ValueError("train, validation, and test token counts must exactly partition the vocabulary")
    if min(config.train_tokens, config.validation_tokens, config.test_tokens) < 64:
        raise ValueError("every token pool must contain at least 64 identities for the P32 evaluation")
    if config.steps < 1 or config.batch_size < 1:
        raise ValueError("steps and batch_size must be positive")
    if config.validation_interval < 1 or config.validation_episodes < 1 or config.evaluation_episodes < 1:
        raise ValueError("evaluation settings must be positive")


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    config = RoleGaugeRunConfig(
        decoder=args.decoder,
        gauge_mode=args.gauge_mode,
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
        train_tokens=args.train_tokens,
        validation_tokens=args.validation_tokens,
        test_tokens=args.test_tokens,
        relation_dim=args.relation_dim,
        dense_rank=args.dense_rank,
        relation_tables=args.relation_tables,
        relation_coverage=args.relation_coverage,
        coxeter_rank=args.coxeter_rank,
        jointpair_tables=args.jointpair_tables,
        jointpair_comparisons=args.jointpair_comparisons,
        data_seed=args.data_seed,
        codebook_seed=args.codebook_seed,
        query_gauge_seed=args.query_gauge_seed,
        key_gauge_seed=args.key_gauge_seed,
        wrong_key_gauge_seed=args.wrong_key_gauge_seed,
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
    pools = token_pools(config)
    model = RoleGaugeMQARRetriever(config).to(device)
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
        batch = generate_pool_mqar_batch(
            token_pool=pools["train"],
            batch_size=config.batch_size,
            pair_count=pair_count,
            query_count=query_count,
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
                token_pool=pools["validation"],
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
    pool_relabel = token_permutation(config.test_tokens, config.token_relabel_seed)
    conditions = ROLE_CONDITIONS if config.gauge_mode == "role_permutation" else IDENTITY_CONDITIONS
    evaluation: dict[str, dict[str, dict[str, float | int]]] = {}
    for name, pair_count, query_count in EVALUATION_SPECS:
        evaluation[name] = {}
        for condition in conditions:
            evaluation[name][condition] = evaluate(
                model,
                token_pool=pools["test"],
                pair_count=pair_count,
                query_count=query_count,
                episodes=config.evaluation_episodes,
                batch_size=config.evaluation_batch_size,
                data_seed=config.data_seed + 70_000_000 + pair_count * 100_003,
                device=device,
                condition=condition,
                pool_permutation=pool_relabel if condition == "token_relabel" else None,
            )
    benchmark = benchmark_forward(
        model,
        token_pool=pools["test"],
        device=device,
        batch_size=min(config.evaluation_batch_size, 64),
        pair_count=32,
        query_count=8,
        data_seed=config.data_seed + 90_000_000,
    )
    diagnostics = gauge_diagnostics(model, token_pool=pools["test"], device=device)
    result: dict[str, Any] = {
        "complete": True,
        "config": asdict(config),
        "architecture": {
            "prefix": "A_1 B_1 ... A_P B_P followed by one or more A queries",
            "local_write": "fixed one-token shift-concat creates [predecessor=A, current=B] at the B row",
            "key_lane": "predecessor A" if model.local_write else "current B (negative control)",
            "query_gauge": "q(x)=x[P_Q]",
            "key_gauge": "k(x)=x[P_K]",
            "value": "exact current token B from the same memory row as K",
            "transport": "relation-score argmax/softmax over B rows followed by exact B transport",
            "trainable_scope": "relation scorer only; codebook, role gauges, local write, and values are fixed",
            "token_identity_split": {name: [int(pool[0].item()), int(pool[-1].item())] for name, pool in pools.items()},
        },
        "gauge_diagnostics": diagnostics,
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


def _result_key(result: dict[str, Any]) -> tuple[str, bool, str, int]:
    config = result["config"]
    return (
        str(config["gauge_mode"]),
        int(config["chamber_relabel_seed"]) >= 0,
        str(config["decoder"]),
        int(config["seed"]),
    )


def _metric(
    results: list[dict[str, Any]],
    *,
    gauge_mode: str,
    decoder: str,
    spec: str,
    condition: str,
    metric: str,
    chamber: bool = False,
) -> list[float]:
    return [
        float(result["evaluation"][spec][condition][metric])
        for result in results
        if result["config"]["gauge_mode"] == gauge_mode
        and result["config"]["decoder"] == decoder
        and (int(result["config"]["chamber_relabel_seed"]) >= 0) == chamber
    ]


def _write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _gain_retention(value: float, reference: float, chance: float) -> float:
    return (value - chance) / max(reference - chance, 1e-12)


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    results = _collect_results(args.result_dir)
    expected_seeds = tuple(args.expected_seeds)
    expected = {("role_permutation", False, decoder, seed) for decoder in DECODERS for seed in expected_seeds}
    expected |= {("role_permutation", True, decoder, seed) for decoder in CHAMBER_DECODERS for seed in expected_seeds}
    expected |= {("identity", False, decoder, seed) for decoder in IDENTITY_CONTROL_DECODERS for seed in expected_seeds}
    counts: dict[tuple[str, bool, str, int], int] = {}
    for result in results:
        key = _result_key(result)
        counts[key] = counts.get(key, 0) + 1
    found = set(counts)
    missing = sorted(expected - found)
    unexpected = sorted(found - expected)
    duplicates = sorted(key for key, count in counts.items() if count != 1)
    if missing or unexpected or duplicates:
        raise RuntimeError(f"incomplete result matrix: missing={missing}, unexpected={unexpected}, duplicates={duplicates}")

    matched_fields = (
        "steps",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "gradient_clip",
        "validation_interval",
        "validation_episodes",
        "evaluation_episodes",
        "evaluation_batch_size",
        "vocab_size",
        "train_tokens",
        "validation_tokens",
        "test_tokens",
        "relation_dim",
        "dense_rank",
        "relation_tables",
        "relation_coverage",
        "coxeter_rank",
        "jointpair_tables",
        "jointpair_comparisons",
        "data_seed",
        "codebook_seed",
        "query_gauge_seed",
        "key_gauge_seed",
        "wrong_key_gauge_seed",
        "token_relabel_seed",
        "device",
    )
    for field in matched_fields:
        values = {json.dumps(result["config"][field], sort_keys=True) for result in results}
        if len(values) != 1:
            raise RuntimeError(f"formal matrix does not match on config field {field!r}: {sorted(values)}")
    formal_config = results[0]["config"]

    spec = "length_p32_q8"
    chance = 1.0 / 32.0
    role_accuracy = {
        decoder: _mean_sem(
            _metric(
                results,
                gauge_mode="role_permutation",
                decoder=decoder,
                spec=spec,
                condition="base",
                metric="target_token_accuracy",
            )
        )[0]
        for decoder in DECODERS
    }
    token_relabel_accuracy = {
        decoder: _mean_sem(
            _metric(
                results,
                gauge_mode="role_permutation",
                decoder=decoder,
                spec=spec,
                condition="token_relabel",
                metric="target_token_accuracy",
            )
        )[0]
        for decoder in DECODERS
    }
    wrong_gauge_accuracy = {
        decoder: _mean_sem(
            _metric(
                results,
                gauge_mode="role_permutation",
                decoder=decoder,
                spec=spec,
                condition="wrong_key_gauge",
                metric="target_token_accuracy",
            )
        )[0]
        for decoder in DECODERS
    }
    role_swap_accuracy = {
        decoder: _mean_sem(
            _metric(
                results,
                gauge_mode="role_permutation",
                decoder=decoder,
                spec=spec,
                condition="role_swap",
                metric="target_token_accuracy",
            )
        )[0]
        for decoder in DECODERS
    }
    identity_accuracy = {
        decoder: _mean_sem(
            _metric(
                results,
                gauge_mode="identity",
                decoder=decoder,
                spec=spec,
                condition="base",
                metric="target_token_accuracy",
            )
        )[0]
        for decoder in IDENTITY_CONTROL_DECODERS
    }
    dense = role_accuracy["local_dense_qk"]
    global_accuracy = role_accuracy["local_global_coxeter"]
    global_dense_retention = _gain_retention(global_accuracy, dense, chance)
    global_relabel_retention = _gain_retention(token_relabel_accuracy["local_global_coxeter"], global_accuracy, chance)
    global_wrong_retention = _gain_retention(wrong_gauge_accuracy["local_global_coxeter"], global_accuracy, chance)
    global_wrong_removal = 1.0 - global_wrong_retention
    positive_route_agreement = statistics.fmean(
        float(result["gauge_diagnostics"]["positive_s4_table_route_agreement"])
        for result in results
        if result["config"]["gauge_mode"] == "role_permutation" and int(result["config"]["chamber_relabel_seed"]) < 0
    )
    gates = {
        "identity_dense_held_tokens": {
            "value": identity_accuracy["local_dense_qk"],
            "threshold": IDENTITY_ACCURACY_GATE,
            "passed": identity_accuracy["local_dense_qk"] >= IDENTITY_ACCURACY_GATE,
        },
        "identity_kendall_held_tokens": {
            "value": identity_accuracy["local_kendall"],
            "threshold": IDENTITY_ACCURACY_GATE,
            "passed": identity_accuracy["local_kendall"] >= IDENTITY_ACCURACY_GATE,
        },
        "role_dense_held_tokens": {
            "value": dense,
            "threshold": LOCAL_DENSE_ACCURACY_GATE,
            "passed": dense >= LOCAL_DENSE_ACCURACY_GATE,
        },
        "role_no_local_is_negative": {
            "value": role_accuracy["no_local_dense_qk"],
            "threshold": chance + NO_LOCAL_EXCESS_ACCURACY_GATE,
            "passed": role_accuracy["no_local_dense_qk"] <= chance + NO_LOCAL_EXCESS_ACCURACY_GATE,
        },
        "exact_s4_route_identity_removed": {
            "value": positive_route_agreement,
            "threshold": ROUTE_IDENTITY_CEILING,
            "passed": positive_route_agreement <= ROUTE_IDENTITY_CEILING,
        },
        "global_retains_dense_gain": {
            "value": global_dense_retention,
            "threshold": GLOBAL_DENSE_GAIN_RETENTION_GATE,
            "passed": global_dense_retention >= GLOBAL_DENSE_GAIN_RETENTION_GATE,
        },
        "global_token_relabel_transfer": {
            "value": global_relabel_retention,
            "threshold": TOKEN_RELABEL_RETENTION_GATE,
            "passed": global_relabel_retention >= TOKEN_RELABEL_RETENTION_GATE,
        },
        "global_wrong_gauge_removes_gain": {
            "value": global_wrong_removal,
            "threshold": WRONG_GAUGE_GAIN_REMOVAL_GATE,
            "passed": global_wrong_removal >= WRONG_GAUGE_GAIN_REMOVAL_GATE,
        },
    }
    qualified = all(
        bool(gates[name]["passed"])
        for name in (
            "identity_dense_held_tokens",
            "identity_kendall_held_tokens",
            "role_dense_held_tokens",
            "role_no_local_is_negative",
            "exact_s4_route_identity_removed",
        )
    )
    ordinal_success = {
        decoder: role_accuracy[decoder] >= ORDINAL_ACCURACY_GATE
        for decoder in (
            "local_kendall",
            "local_root_incidence",
            "local_global_coxeter",
            "local_jointpair_full",
        )
    }
    global_relabelled_accuracy = _mean_sem(
        _metric(
            results,
            gauge_mode="role_permutation",
            decoder="local_global_coxeter",
            spec=spec,
            condition="base",
            metric="target_token_accuracy",
            chamber=True,
        )
    )[0]
    global_chamber_structure_retention = _gain_retention(global_relabelled_accuracy, global_accuracy, chance)
    global_passes = all(
        bool(gates[name]["passed"])
        for name in (
            "global_retains_dense_gain",
            "global_token_relabel_transfer",
            "global_wrong_gauge_removes_gain",
        )
    )
    if not qualified:
        interpretation = "invalid_role_relation_comparison"
    elif global_passes and global_chamber_structure_retention < 0.80:
        interpretation = "global_coxeter_learns_transferable_role_relation_with_structure_evidence"
    elif global_passes:
        interpretation = "global_factorization_learns_role_relation_but_coxeter_specificity_is_not_identified"
    elif ordinal_success["local_jointpair_full"] and not ordinal_success["local_global_coxeter"]:
        interpretation = "joint_pair_capacity_succeeds_but_global_coxeter_sharing_fails"
    elif not any(ordinal_success.values()):
        interpretation = "tested_ordinal_quotients_lose_the_role_relation"
    else:
        interpretation = "mixed_ordinal_role_relation_result"

    rows: list[dict[str, object]] = []
    for decoder in DECODERS:
        decoder_results = [
            result
            for result in results
            if result["config"]["gauge_mode"] == "role_permutation"
            and result["config"]["decoder"] == decoder
            and int(result["config"]["chamber_relabel_seed"]) < 0
        ]
        parameter_values = [int(result["relation_parameters"]) for result in decoder_results]
        parameter_text = (
            f"{parameter_values[0]:,}" if min(parameter_values) == max(parameter_values) else f"{min(parameter_values):,}-{max(parameter_values):,}"
        )
        rows.append(
            {
                "decoder": decoder,
                "relation_parameters": parameter_text,
                "target_accuracy_p8": _mean_sem(
                    _metric(
                        results,
                        gauge_mode="role_permutation",
                        decoder=decoder,
                        spec="id_p8_q4",
                        condition="base",
                        metric="target_token_accuracy",
                    )
                )[0],
                "target_accuracy_p16": _mean_sem(
                    _metric(
                        results,
                        gauge_mode="role_permutation",
                        decoder=decoder,
                        spec="length_p16_q8",
                        condition="base",
                        metric="target_token_accuracy",
                    )
                )[0],
                "target_accuracy_p32": role_accuracy[decoder],
                "target_ce_p32": _mean_sem(
                    _metric(
                        results,
                        gauge_mode="role_permutation",
                        decoder=decoder,
                        spec=spec,
                        condition="base",
                        metric="target_token_ce",
                    )
                )[0],
                "token_relabel_accuracy_p32": token_relabel_accuracy[decoder],
                "wrong_key_gauge_accuracy_p32": wrong_gauge_accuracy[decoder],
                "role_swap_accuracy_p32": role_swap_accuracy[decoder],
                "pair_scores_per_second": _mean_sem([float(result["benchmark"]["pair_scores_per_second"]) for result in decoder_results])[0],
            }
        )
    _write_summary_csv(args.out_report.with_suffix(".csv"), rows)

    def formatted(
        gauge_mode: str,
        decoder: str,
        evaluation_spec: str,
        condition: str,
        metric: str,
    ) -> str:
        return _format_mean_sem(
            _metric(
                results,
                gauge_mode=gauge_mode,
                decoder=decoder,
                spec=evaluation_spec,
                condition=condition,
                metric=metric,
            )
        )

    lines = [
        "# Causal MQAR with Role-Specific Q/K Gauges",
        "",
        "## Question",
        "",
        "Can a low-complexity ordinal relation kernel learn one fixed relation between "
        "different Q and K coordinate gauges and transfer it to unseen token identities?",
        "",
        "Each token is a fixed permutation rank vector. The fixed local write places A "
        "beside B at the B row, but the relation scorer sees `q(A)=A[P_Q]` and "
        "`k(A)=A[P_K]`, with distinct target-free coordinate permutations. Values are "
        "exact B identities from the same row as K. Only the relation scorer is trained.",
        "",
        f"Token identities are disjoint: train `{formal_config['train_tokens']}`, validation "
        f"`{formal_config['validation_tokens']}`, and test `{formal_config['test_tokens']}`. "
        "Training samples P4-P8/Q1-Q4; P16 and P32 are simultaneous identity and length transfer.",
        "",
        "```text",
        "query A -> ordinal x(A) -> fixed P_Q ----------------------> Q",
        "prefix ... A B -> B-row [predecessor A | current B]",
        "                         |                    |",
        "                         +-> fixed P_K ------> K",
        "                                              B -----------> V",
        "Q,K -> relation scores over B rows -> select one row -> transport its V",
        "```",
        "",
        "Design motivation and primary-source map: [`PHYSICSLM_ROLE_GAUGE_READING.md`](../../doc/PHYSICSLM_ROLE_GAUGE_READING.md).",
        "",
        "## Role-gauge results",
        "",
        "| Decoder | Relation params | P8 acc | P16 acc | P32 acc | P32 CE | Token relabel | Wrong K gauge | Q/K swap | Pair scores/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        decoder = str(row["decoder"])
        lines.append(
            f"| {decoder} | {row['relation_parameters']} | "
            f"{formatted('role_permutation', decoder, 'id_p8_q4', 'base', 'target_token_accuracy')} | "
            f"{formatted('role_permutation', decoder, 'length_p16_q8', 'base', 'target_token_accuracy')} | "
            f"{formatted('role_permutation', decoder, spec, 'base', 'target_token_accuracy')} | "
            f"{formatted('role_permutation', decoder, spec, 'base', 'target_token_ce')} | "
            f"{formatted('role_permutation', decoder, spec, 'token_relabel', 'target_token_accuracy')} | "
            f"{formatted('role_permutation', decoder, spec, 'wrong_key_gauge', 'target_token_accuracy')} | "
            f"{formatted('role_permutation', decoder, spec, 'role_swap', 'target_token_accuracy')} | "
            f"{float(row['pair_scores_per_second']):,.0f} |"
        )
    lines.extend(
        [
            "",
            "Random P32 target accuracy is `0.03125`. Token relabeling is a fixed "
            "bijection inside the unseen test pool. Wrong-K and Q/K-swap evaluations "
            "change the fixed role relation after training; they are mechanism removals, "
            "not unseen-gauge generalization tests.",
            "",
            "## Identity-gauge qualification",
            "",
            "| Decoder | P32 held-token acc | P32 token-relabel acc |",
            "|---|---:|---:|",
        ]
    )
    for decoder in IDENTITY_CONTROL_DECODERS:
        lines.append(
            f"| {decoder} | "
            f"{formatted('identity', decoder, spec, 'base', 'target_token_accuracy')} | "
            f"{formatted('identity', decoder, spec, 'token_relabel', 'target_token_accuracy')} |"
        )
    lines.extend(
        [
            "",
            "## Chamber-label retraining control",
            "",
            "| Decoder | Original role-gauge P32 acc | Relabeled-and-retrained P32 acc | Gain retention |",
            "|---|---:|---:|---:|",
        ]
    )
    for decoder in CHAMBER_DECODERS:
        base_values = _metric(
            results,
            gauge_mode="role_permutation",
            decoder=decoder,
            spec=spec,
            condition="base",
            metric="target_token_accuracy",
        )
        relabelled_values = _metric(
            results,
            gauge_mode="role_permutation",
            decoder=decoder,
            spec=spec,
            condition="base",
            metric="target_token_accuracy",
            chamber=True,
        )
        retention = _gain_retention(statistics.fmean(relabelled_values), statistics.fmean(base_values), chance)
        lines.append(f"| {decoder} | {_format_mean_sem(base_values)} | {_format_mean_sem(relabelled_values)} | {retention:.4f} |")
    lines.extend(["", "## Preregistered gates", ""])
    for name, gate in gates.items():
        lines.append(f"- `{name}`: value `{float(gate['value']):.4f}`, threshold `{float(gate['threshold']):.4f}`, passed `{bool(gate['passed'])}`.")
    lines.extend(
        [
            "",
            f"Decision: `{interpretation}`.",
            "",
            "Dense is a qualification ceiling, not the proposed deployment path. A relation "
            "comparison is valid only if identity Dense/Kendall pass on unseen identities, "
            "role-gauged Dense passes, no-local Dense stays negative, and the measured S4 "
            "route equality shortcut is absent. Global Coxeter must then retain at least 80% "
            "of Dense's chance-adjusted gain, preserve at least 95% under unseen-token "
            "relabeling, and lose at least 80% of its gain under a wrong K gauge.",
            "",
            "Arbitrary 24-way chamber relabeling is interpreted separately. If a successful "
            "Global model survives relabeling and retraining, the evidence supports transferable "
            "low-rank categorical factorization but does not identify Coxeter representation "
            "geometry as the cause.",
            "",
            "## Scope",
            "",
            "This isolates one fixed role relation under causal K/V alignment. Passing does not "
            "establish natural-language semantics, learned Q/K projections, position-dependent "
            "gauges, multi-token composition, or recursive reasoning. It is the smallest test "
            "between chamber identity lookup and an end-to-end language model.",
            "",
            "## Reproduction",
            "",
            "```bash",
            f"GPU_LIST={args.gpu_list} scripts/run_tropnn_causal_mqar_role_gauge_6gpu.sh",
            "```",
            "",
            f"- Results: `{args.result_dir}`.",
            f"- Logs: `{args.result_dir.parent / 'logs'}`.",
            f"- Formal seeds: `{', '.join(str(seed) for seed in expected_seeds)}`; "
            f"{int(formal_config['steps']):,} optimizer steps per run; batch "
            f"{int(formal_config['batch_size'])}.",
            "- The summary requires 18 role-gauge base runs, 9 role-gauge chamber-relabel runs, and 6 identity-gauge qualification runs.",
            "- Semantic implementation and tests: "
            "`python/src/tropnn/tools/causal_mqar_role_gauge.py` and "
            "`python/src/tropnn/tests/test_causal_mqar_role_gauge.py`.",
            "",
        ]
    )
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text("\n".join(lines))
    decision = {
        "complete": True,
        "expected_seeds": list(expected_seeds),
        "result_count": len(results),
        "qualified_role_relation_comparison": qualified,
        "ordinal_success": ordinal_success,
        "gates": gates,
        "global_chamber_structure_retention": global_chamber_structure_retention,
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
    run.add_argument("--gauge-mode", choices=GAUGE_MODES, required=True)
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
    run.add_argument("--vocab-size", type=int, default=768)
    run.add_argument("--train-tokens", type=int, default=512)
    run.add_argument("--validation-tokens", type=int, default=128)
    run.add_argument("--test-tokens", type=int, default=128)
    run.add_argument("--relation-dim", type=int, default=32)
    run.add_argument("--dense-rank", type=int, default=16)
    run.add_argument("--relation-tables", type=int, default=16)
    run.add_argument("--relation-coverage", type=int, default=2)
    run.add_argument("--coxeter-rank", type=int, default=12)
    run.add_argument("--jointpair-tables", type=int, default=144)
    run.add_argument("--jointpair-comparisons", type=int, default=6)
    run.add_argument("--data-seed", type=int, default=1729)
    run.add_argument("--codebook-seed", type=int, default=2718)
    run.add_argument("--query-gauge-seed", type=int, default=202607241)
    run.add_argument("--key-gauge-seed", type=int, default=202607242)
    run.add_argument("--wrong-key-gauge-seed", type=int, default=202607243)
    run.add_argument("--token-relabel-seed", type=int, default=314159)
    run.add_argument("--chamber-relabel-seed", type=int, default=-1)
    run.add_argument("--device", default="cuda")
    run.set_defaults(function=run_experiment)

    summary = subparsers.add_parser("summarize", help="validate the matrix and write its report")
    summary.add_argument("--result-dir", type=Path, required=True)
    summary.add_argument("--out-report", type=Path, required=True)
    summary.add_argument("--expected-seeds", type=int, nargs="+", default=(0, 1, 2))
    summary.add_argument("--gpu-list", default="0,1,2,3,4,5")
    summary.set_defaults(function=summarize)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
