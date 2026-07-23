"""Infer one latent Coxeter transport from support pairs, then retrieve.

This is a closed-world episodic relation-inference gate built on the frozen
generator dictionary learned by ``causal_mqar_variable_transport``.  A held
episode samples one reduced Coxeter word, but does not expose that word to the
inference rule.  Instead, the rule observes paired query/key ordinal states:

    support q_m --one shared hidden transport--> support k_m

It chooses one relation from a same-length held bank by maximizing aggregate
signed-root agreement.  That single inferred relation is then frozen while all
query candidates are ranked.  Candidate-dependent relation selection is never
used.

The experiment measures exact relation/product recovery, size of the surviving
relation version space, and downstream one-hop MQAR retrieval under
order-sensitive orbit negatives.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from tropnn.tools.causal_mqar_induction import (
    atomic_json_write,
    seed_everything,
    token_permutation,
)
from tropnn.tools.causal_mqar_role_gauge import token_pools
from tropnn.tools.causal_mqar_root_transport import root_signs
from tropnn.tools.causal_mqar_variable_transport import (
    CoxeterPathBatch,
    VariableCoxeterTransport,
    VariableTransportConfig,
    actual_word_permutations,
    build_word_pools,
    compose_generator_words,
    dynamic_signed_root_transport,
    symbol_dictionary,
    word_pool_diagnostics,
)

DEFAULT_SUPPORT_COUNTS = (0, 1, 2, 4, 8)
DEFAULT_ROOT_BUDGETS = (16, 32, 64, 96, 192)
HELD_WORD_SPLITS = (
    "unseen_l2",
    "unseen_l4",
    "unseen_l8",
    "unseen_l16",
    "unseen_l31",
)


@dataclass(frozen=True)
class LatentRelationConfig:
    seed: int
    transport_checkpoint: str
    episodes: int
    batch_size: int
    pair_count: int
    query_count: int
    support_counts: tuple[int, ...]
    root_budgets: tuple[int, ...]
    data_seed: int
    token_relabel_seed: int
    device: str


@dataclass(frozen=True)
class LatentRelationBatch:
    """Episode data plus labels retained only for evaluation."""

    support_tokens: Tensor
    query_batch: CoxeterPathBatch
    true_relation_indices: Tensor

    @property
    def batch_size(self) -> int:
        return int(self.support_tokens.shape[0])

    @property
    def max_support_count(self) -> int:
        return int(self.support_tokens.shape[1])

    def to(self, device: torch.device) -> "LatentRelationBatch":
        return LatentRelationBatch(
            support_tokens=self.support_tokens.to(
                device,
                non_blocking=True,
            ),
            query_batch=self.query_batch.to(device),
            true_relation_indices=self.true_relation_indices.to(
                device,
                non_blocking=True,
            ),
        )


@dataclass(frozen=True)
class SupportObservations:
    """The only tensors visible to shared-relation inference."""

    query_coordinates: Tensor
    key_coordinates: Tensor

    def prefix(self, count: int) -> "SupportObservations":
        if not 0 <= count <= self.query_coordinates.shape[1]:
            raise ValueError("support prefix is outside the observed episode")
        return SupportObservations(
            query_coordinates=self.query_coordinates[:, :count],
            key_coordinates=self.key_coordinates[:, :count],
        )


@dataclass(frozen=True)
class RelationInference:
    candidate_scores: Tensor
    selected_indices: Tensor
    selected_permutations: Tensor


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item) for item in value.split(",") if item)
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return parsed


def _inverse_rows(row_permutation: Tensor) -> Tensor:
    inverse = torch.empty_like(row_permutation)
    inverse.scatter_(
        1,
        row_permutation,
        torch.arange(
            row_permutation.shape[1],
            device=row_permutation.device,
        ).expand(row_permutation.shape[0], -1),
    )
    return inverse


def generate_latent_relation_batch(
    *,
    token_pool: Tensor,
    word_pool: Tensor,
    dictionary: Tensor,
    dimension: int,
    batch_size: int,
    pair_count: int,
    query_count: int,
    max_support_count: int,
    seed: int,
    pool_permutation: Tensor | None = None,
) -> LatentRelationBatch:
    """Generate support pairs and one-hop query memory with one shared word."""

    if pair_count < 2 or pair_count % 2:
        raise ValueError("pair count must be even for positive/orbit pairs")
    positive_pair_count = pair_count // 2
    if not 1 <= query_count <= positive_pair_count:
        raise ValueError("query count must not exceed positive pair count")
    if max_support_count < 1:
        raise ValueError("at least one support observation must be generated")
    if word_pool.ndim != 2 or word_pool.shape[0] < 2:
        raise ValueError("word pool must contain at least two relations")

    pool_size = int(token_pool.numel())
    needed_tokens = max_support_count + 3 * positive_pair_count
    if needed_tokens > pool_size:
        raise ValueError("token pool is too small for disjoint episode tokens")
    if pool_permutation is not None and pool_permutation.shape != (pool_size,):
        raise ValueError("pool permutation has the wrong shape")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    token_priority = torch.rand(batch_size, pool_size, generator=generator)
    local_tokens = token_priority.topk(
        needed_tokens,
        dim=1,
        largest=False,
        sorted=False,
    ).indices
    if pool_permutation is not None:
        local_tokens = pool_permutation[local_tokens]
    episode_tokens = token_pool[local_tokens]
    support_tokens = episode_tokens[:, :max_support_count]
    cursor = max_support_count
    positive_keys = episode_tokens[
        :,
        cursor : cursor + positive_pair_count,
    ]
    cursor += positive_pair_count
    positive_values = episode_tokens[
        :,
        cursor : cursor + positive_pair_count,
    ]
    cursor += positive_pair_count
    decoy_values = episode_tokens[
        :,
        cursor : cursor + positive_pair_count,
    ]

    true_relation_indices = torch.randint(
        int(word_pool.shape[0]),
        (batch_size,),
        generator=generator,
    )
    word_symbols = word_pool[true_relation_indices]
    actual_key_permutation = actual_word_permutations(
        word_symbols,
        dictionary=dictionary,
        dimension=dimension,
    )
    reverse_key_permutation = actual_word_permutations(
        word_symbols.flip(dims=(1,)),
        dictionary=dictionary,
        dimension=dimension,
    )
    decoy_prepermutation = torch.empty_like(actual_key_permutation)
    decoy_prepermutation.scatter_(
        1,
        actual_key_permutation,
        reverse_key_permutation,
    )
    identity = torch.arange(dimension).expand(
        batch_size,
        positive_pair_count,
        -1,
    )

    memory_keys = torch.cat((positive_keys, positive_keys), dim=1)
    memory_values = torch.cat((positive_values, decoy_values), dim=1)
    memory_key_prepermutation = torch.cat(
        (
            identity,
            decoy_prepermutation[:, None, :].expand(
                -1,
                positive_pair_count,
                -1,
            ),
        ),
        dim=1,
    )
    row_permutation = torch.rand(
        batch_size,
        pair_count,
        generator=generator,
    ).argsort(dim=1)
    memory_keys = memory_keys.gather(1, row_permutation)
    memory_values = memory_values.gather(1, row_permutation)
    memory_key_prepermutation = memory_key_prepermutation.gather(
        1,
        row_permutation[:, :, None].expand(-1, -1, dimension),
    )
    inverse_rows = _inverse_rows(row_permutation)

    query_rows = torch.rand(
        batch_size,
        positive_pair_count,
        generator=generator,
    ).topk(
        query_count,
        dim=1,
        largest=False,
        sorted=False,
    ).indices
    queries = positive_keys.gather(1, query_rows)
    targets = positive_values.gather(1, query_rows)
    positive_indices = inverse_rows.gather(1, query_rows)
    query_batch = CoxeterPathBatch(
        memory_keys=memory_keys,
        memory_values=memory_values,
        memory_key_prepermutation=memory_key_prepermutation,
        queries=queries,
        hop_targets=targets.unsqueeze(-1),
        hop_indices=positive_indices.unsqueeze(-1),
        word_symbols=word_symbols,
        actual_key_permutation=actual_key_permutation,
    )
    return LatentRelationBatch(
        support_tokens=support_tokens,
        query_batch=query_batch,
        true_relation_indices=true_relation_indices,
    )


@torch.no_grad()
def observe_support(
    model: VariableCoxeterTransport,
    batch: LatentRelationBatch,
) -> SupportObservations:
    """Materialize paired states; do not return the hidden word or permutation."""

    query_coordinates = model.codebook(batch.support_tokens)
    key_coordinates = query_coordinates.gather(
        -1,
        batch.query_batch.actual_key_permutation[:, None, :].expand(
            -1,
            batch.max_support_count,
            -1,
        ),
    )
    return SupportObservations(
        query_coordinates=query_coordinates,
        key_coordinates=key_coordinates,
    )


def relation_candidate_permutations(
    word_pool: Tensor,
    *,
    generator_dictionary: Tensor,
    dimension: int,
) -> Tensor:
    return compose_generator_words(
        generator_dictionary[word_pool],
        dimension,
    )


@torch.no_grad()
def infer_shared_relation(
    model: VariableCoxeterTransport,
    observations: SupportObservations,
    candidate_permutations: Tensor,
) -> RelationInference:
    """Choose one transport for the whole episode using support consensus."""

    batch_size = int(observations.query_coordinates.shape[0])
    candidate_permutations = candidate_permutations.to(
        observations.query_coordinates.device
    )
    if observations.query_coordinates.shape[1] == 0:
        candidate_scores = torch.zeros(
            batch_size,
            candidate_permutations.shape[0],
            device=observations.query_coordinates.device,
        )
    else:
        query_roots = root_signs(
            observations.query_coordinates,
            model.edges,
        )[..., model.query_root_subset]
        key_roots = root_signs(
            observations.key_coordinates,
            model.edges,
        )
        signed_key = torch.cat((-key_roots, key_roots), dim=-1)
        signed_indices = dynamic_signed_root_transport(
            candidate_permutations,
            edges=model.edges,
            query_root_subset=model.query_root_subset,
            edge_index=model.edge_index,
        )
        candidate_count = int(candidate_permutations.shape[0])
        aligned = signed_key[:, :, None, :].expand(
            -1,
            -1,
            candidate_count,
            -1,
        ).gather(
            -1,
            signed_indices[None, None, :, :].expand(
                batch_size,
                observations.query_coordinates.shape[1],
                -1,
                -1,
            ),
        )
        candidate_scores = (
            query_roots[:, :, None, :] * aligned
        ).sum(dim=(1, 3))
    selected_indices = candidate_scores.argmax(dim=-1)
    return RelationInference(
        candidate_scores=candidate_scores,
        selected_indices=selected_indices,
        selected_permutations=candidate_permutations[selected_indices],
    )


@torch.no_grad()
def query_scores_with_transport(
    model: VariableCoxeterTransport,
    batch: CoxeterPathBatch,
    predicted_permutation: Tensor,
) -> Tensor:
    """Rank every candidate with one preselected episode transport."""

    query_roots, key_roots = model._root_features(
        batch.queries,
        batch,
        composer="oracle",
    )
    query_active = query_roots[..., model.query_root_subset]
    signed_indices = dynamic_signed_root_transport(
        predicted_permutation,
        edges=model.edges,
        query_root_subset=model.query_root_subset,
        edge_index=model.edge_index,
    )
    signed_key = torch.cat((-key_roots, key_roots), dim=-1)
    aligned = signed_key.gather(
        -1,
        signed_indices[:, None, :].expand(
            -1,
            batch.pair_count,
            -1,
        ),
    )
    scores = torch.einsum(
        "bqu,bpu->bqp",
        query_active,
        aligned,
    ) / math.sqrt(model.config.root_budget)
    return model.logit_scale.exp().clamp(max=100.0) * scores


def _empty_accumulator() -> dict[str, float]:
    return {
        "episodes": 0.0,
        "queries": 0.0,
        "relation_top1_correct": 0.0,
        "relation_product_correct": 0.0,
        "true_in_max_set": 0.0,
        "unique_relation_correct": 0.0,
        "max_tie_count_sum": 0.0,
        "query_correct": 0.0,
        "query_exact": 0.0,
        "query_ce_sum": 0.0,
    }


def _accumulate_inference(
    accumulator: dict[str, float],
    *,
    inference: RelationInference,
    true_relation_indices: Tensor,
    true_permutations: Tensor,
    query_scores: Tensor,
    query_batch: CoxeterPathBatch,
) -> None:
    batch_size, query_count = query_batch.queries.shape
    maximum = inference.candidate_scores.max(dim=-1).values
    max_mask = inference.candidate_scores == maximum[:, None]
    tie_count = max_mask.sum(dim=-1)
    relation_correct = (
        inference.selected_indices == true_relation_indices
    )
    product_correct = (
        inference.selected_permutations == true_permutations
    ).all(dim=-1)
    true_in_max = max_mask.gather(
        1,
        true_relation_indices[:, None],
    ).squeeze(1)

    selected_rows = query_scores.argmax(dim=-1)
    predicted_values = (
        query_batch.memory_values[:, None, :]
        .expand(-1, query_count, -1)
        .gather(2, selected_rows.unsqueeze(-1))
        .squeeze(-1)
    )
    targets = query_batch.hop_targets[:, :, 0]
    correct = predicted_values == targets
    accumulator["episodes"] += batch_size
    accumulator["queries"] += batch_size * query_count
    accumulator["relation_top1_correct"] += int(
        relation_correct.sum().item()
    )
    accumulator["relation_product_correct"] += int(
        product_correct.sum().item()
    )
    accumulator["true_in_max_set"] += int(true_in_max.sum().item())
    accumulator["unique_relation_correct"] += int(
        (relation_correct & (tie_count == 1)).sum().item()
    )
    accumulator["max_tie_count_sum"] += int(tie_count.sum().item())
    accumulator["query_correct"] += int(correct.sum().item())
    accumulator["query_exact"] += int(correct.all(dim=1).sum().item())
    accumulator["query_ce_sum"] += float(
        F.cross_entropy(
            query_scores.flatten(0, 1),
            query_batch.hop_indices[:, :, 0].flatten(),
            reduction="sum",
        ).item()
    )


def _finalize_accumulator(
    accumulator: dict[str, float],
) -> dict[str, float | int]:
    episodes = int(accumulator["episodes"])
    queries = int(accumulator["queries"])
    return {
        "episodes": episodes,
        "queries": queries,
        "relation_top1_accuracy": (
            accumulator["relation_top1_correct"] / episodes
        ),
        "relation_product_accuracy": (
            accumulator["relation_product_correct"] / episodes
        ),
        "true_relation_in_max_set": (
            accumulator["true_in_max_set"] / episodes
        ),
        "unique_relation_accuracy": (
            accumulator["unique_relation_correct"] / episodes
        ),
        "mean_max_tie_count": (
            accumulator["max_tie_count_sum"] / episodes
        ),
        "query_r1": accumulator["query_correct"] / queries,
        "multiquery_exact_accuracy": (
            accumulator["query_exact"] / episodes
        ),
        "query_ce": accumulator["query_ce_sum"] / queries,
    }


def _query_control_accumulator() -> dict[str, float]:
    return {
        "episodes": 0.0,
        "queries": 0.0,
        "correct": 0.0,
        "exact": 0.0,
    }


def _accumulate_query_control(
    accumulator: dict[str, float],
    scores: Tensor,
    query_batch: CoxeterPathBatch,
) -> None:
    batch_size, query_count = query_batch.queries.shape
    selected_rows = scores.argmax(dim=-1)
    predicted_values = (
        query_batch.memory_values[:, None, :]
        .expand(-1, query_count, -1)
        .gather(2, selected_rows.unsqueeze(-1))
        .squeeze(-1)
    )
    correct = predicted_values == query_batch.hop_targets[:, :, 0]
    accumulator["episodes"] += batch_size
    accumulator["queries"] += batch_size * query_count
    accumulator["correct"] += int(correct.sum().item())
    accumulator["exact"] += int(correct.all(dim=1).sum().item())


def _finalize_query_control(
    accumulator: dict[str, float],
) -> dict[str, float | int]:
    return {
        "episodes": int(accumulator["episodes"]),
        "queries": int(accumulator["queries"]),
        "query_r1": (
            accumulator["correct"] / accumulator["queries"]
        ),
        "multiquery_exact_accuracy": (
            accumulator["exact"] / accumulator["episodes"]
        ),
    }


def _load_model(
    checkpoint_path: Path,
    *,
    root_budget: int,
    device: torch.device,
) -> tuple[
    VariableCoxeterTransport,
    VariableTransportConfig,
    Tensor,
]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    base_config = VariableTransportConfig(**checkpoint["config"])
    config = replace(
        base_config,
        root_budget=root_budget,
        device=str(device),
    )
    model = VariableCoxeterTransport(config)
    state = checkpoint["state_dict"]
    with torch.no_grad():
        model.generator_logits.copy_(state["generator_logits"])
        model.logit_scale.copy_(state["logit_scale"])
    model.eval().to(device)
    true_dictionary = checkpoint.get(
        "true_symbol_dictionary",
        symbol_dictionary(
            config.relation_dim,
            config.generator_dictionary_seed,
        ),
    )
    return model, config, true_dictionary


@torch.no_grad()
def evaluate_split(
    *,
    checkpoint_path: Path,
    word_pool: Tensor,
    root_budget: int,
    support_counts: tuple[int, ...],
    episodes: int,
    batch_size: int,
    pair_count: int,
    query_count: int,
    data_seed: int,
    device: torch.device,
    relabel_seed: int | None,
) -> dict[str, Any]:
    model, config, true_dictionary = _load_model(
        checkpoint_path,
        root_budget=root_budget,
        device=device,
    )
    pools = token_pools(config)
    max_support_count = max(support_counts)
    learned_candidates = relation_candidate_permutations(
        word_pool,
        generator_dictionary=model.hard_dictionary().detach().cpu(),
        dimension=config.relation_dim,
    ).to(device)
    oracle_candidates = actual_word_permutations(
        word_pool,
        dictionary=true_dictionary.cpu(),
        dimension=config.relation_dim,
    ).to(device)
    pool_relabel = (
        token_permutation(config.test_tokens, relabel_seed)
        if relabel_seed is not None
        else None
    )

    learned_accumulators = {
        support_count: _empty_accumulator()
        for support_count in support_counts
    }
    oracle_inference_accumulators = {
        support_count: _empty_accumulator()
        for support_count in support_counts
    }
    controls = {
        name: _query_control_accumulator()
        for name in (
            "oracle_transport",
            "reverse_order",
            "identity_transport",
        )
    }

    for offset in range(0, episodes, batch_size):
        current_batch_size = min(batch_size, episodes - offset)
        latent_batch = generate_latent_relation_batch(
            token_pool=pools["test"],
            word_pool=word_pool,
            dictionary=true_dictionary,
            dimension=config.relation_dim,
            batch_size=current_batch_size,
            pair_count=pair_count,
            query_count=query_count,
            max_support_count=max_support_count,
            seed=data_seed + offset * 1009,
            pool_permutation=pool_relabel,
        ).to(device)
        observations = observe_support(model, latent_batch)
        true_indices = latent_batch.true_relation_indices
        true_permutations = oracle_candidates[true_indices]

        for support_count in support_counts:
            support_prefix = observations.prefix(support_count)
            learned_inference = infer_shared_relation(
                model,
                support_prefix,
                learned_candidates,
            )
            learned_scores = query_scores_with_transport(
                model,
                latent_batch.query_batch,
                learned_inference.selected_permutations,
            )
            _accumulate_inference(
                learned_accumulators[support_count],
                inference=learned_inference,
                true_relation_indices=true_indices,
                true_permutations=true_permutations,
                query_scores=learned_scores,
                query_batch=latent_batch.query_batch,
            )

            oracle_inference = infer_shared_relation(
                model,
                support_prefix,
                oracle_candidates,
            )
            oracle_inference_scores = query_scores_with_transport(
                model,
                latent_batch.query_batch,
                oracle_inference.selected_permutations,
            )
            _accumulate_inference(
                oracle_inference_accumulators[support_count],
                inference=oracle_inference,
                true_relation_indices=true_indices,
                true_permutations=true_permutations,
                query_scores=oracle_inference_scores,
                query_batch=latent_batch.query_batch,
            )

        oracle_scores = query_scores_with_transport(
            model,
            latent_batch.query_batch,
            true_permutations,
        )
        _accumulate_query_control(
            controls["oracle_transport"],
            oracle_scores,
            latent_batch.query_batch,
        )
        reverse_permutations = actual_word_permutations(
            latent_batch.query_batch.word_symbols.flip(dims=(1,)).cpu(),
            dictionary=true_dictionary.cpu(),
            dimension=config.relation_dim,
        ).to(device)
        reverse_scores = query_scores_with_transport(
            model,
            latent_batch.query_batch,
            reverse_permutations,
        )
        _accumulate_query_control(
            controls["reverse_order"],
            reverse_scores,
            latent_batch.query_batch,
        )
        identity = torch.arange(
            config.relation_dim,
            device=device,
        ).expand(current_batch_size, -1)
        identity_scores = query_scores_with_transport(
            model,
            latent_batch.query_batch,
            identity,
        )
        _accumulate_query_control(
            controls["identity_transport"],
            identity_scores,
            latent_batch.query_batch,
        )

    return {
        "candidate_relations": int(word_pool.shape[0]),
        "word_length": int(word_pool.shape[1]),
        "root_budget": root_budget,
        "support_counts": {
            str(count): _finalize_accumulator(
                learned_accumulators[count]
            )
            for count in support_counts
        },
        "oracle_dictionary_support_counts": {
            str(count): _finalize_accumulator(
                oracle_inference_accumulators[count]
            )
            for count in support_counts
        },
        "query_controls": {
            name: _finalize_query_control(accumulator)
            for name, accumulator in controls.items()
        },
    }


def _mean_cells(
    evaluation: dict[str, dict[str, Any]],
    *,
    root_budget: int,
    support_count: int,
    field: str,
    dictionary: str = "learned",
    splits: tuple[str, ...] = HELD_WORD_SPLITS,
) -> float:
    section = (
        "support_counts"
        if dictionary == "learned"
        else "oracle_dictionary_support_counts"
    )
    values = [
        float(
            evaluation[split][str(root_budget)][section][
                str(support_count)
            ][field]
        )
        for split in splits
    ]
    return sum(values) / len(values)


def _build_gates(
    evaluation: dict[str, dict[str, Any]],
    *,
    root_budget: int,
    support_count: int,
) -> dict[str, dict[str, float | bool]]:
    learned_product = _mean_cells(
        evaluation,
        root_budget=root_budget,
        support_count=support_count,
        field="relation_product_accuracy",
    )
    oracle_product = _mean_cells(
        evaluation,
        root_budget=root_budget,
        support_count=support_count,
        field="relation_product_accuracy",
        dictionary="oracle",
    )
    learned_query = _mean_cells(
        evaluation,
        root_budget=root_budget,
        support_count=support_count,
        field="query_r1",
    )
    oracle_query = sum(
        float(
            evaluation[split][str(root_budget)]["query_controls"][
                "oracle_transport"
            ]["query_r1"]
        )
        for split in HELD_WORD_SPLITS
    ) / len(HELD_WORD_SPLITS)
    no_support_product = _mean_cells(
        evaluation,
        root_budget=root_budget,
        support_count=0,
        field="relation_product_accuracy",
    )
    true_in_max = _mean_cells(
        evaluation,
        root_budget=root_budget,
        support_count=support_count,
        field="true_relation_in_max_set",
    )
    long_splits = ("unseen_l8", "unseen_l16", "unseen_l31")
    reverse_query = sum(
        float(
            evaluation[split][str(root_budget)]["query_controls"][
                "reverse_order"
            ]["query_r1"]
        )
        for split in long_splits
    ) / len(long_splits)
    chance = 1.0 / 32
    reverse_removal = 1.0 - (
        (reverse_query - chance)
        / max(learned_query - chance, 1e-12)
    )
    quantities = {
        "latent_relation_product_accuracy": (
            learned_product,
            0.95,
            "min",
        ),
        "learned_dictionary_retains_oracle_inference": (
            learned_product / max(oracle_product, 1e-12),
            0.99,
            "min",
        ),
        "inferred_query_retains_oracle_transport": (
            learned_query / max(oracle_query, 1e-12),
            0.95,
            "min",
        ),
        "true_relation_survives_support_version_space": (
            true_in_max,
            0.99,
            "min",
        ),
        "zero_support_relation_negative": (
            no_support_product,
            0.07,
            "max",
        ),
        "reverse_order_query_gain_removal": (
            reverse_removal,
            0.50,
            "min",
        ),
    }
    output: dict[str, dict[str, float | bool]] = {}
    for name, (value, threshold, direction) in quantities.items():
        passed = (
            value >= threshold
            if direction == "min"
            else value <= threshold
        )
        output[name] = {
            "value": value,
            "threshold": threshold,
            "passed": passed,
        }
    return output


def _validate_config(
    config: LatentRelationConfig,
    base_config: VariableTransportConfig,
) -> None:
    root_count = (
        base_config.relation_dim
        * (base_config.relation_dim - 1)
        // 2
    )
    if config.episodes < 1 or config.batch_size < 1:
        raise ValueError("episodes and batch size must be positive")
    if config.pair_count != 32:
        raise ValueError("formal latent-relation evaluation requires P32")
    if not 1 <= config.query_count <= config.pair_count // 2:
        raise ValueError("query count is outside the positive memory rows")
    if 0 not in config.support_counts:
        raise ValueError("support counts must include the zero-support control")
    if min(config.support_counts) < 0 or max(config.support_counts) < 1:
        raise ValueError("support counts must include positive observations")
    if min(config.root_budgets) < 1 or max(config.root_budgets) > root_count:
        raise ValueError("root budget is outside the full carrier")


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = torch.load(
        args.transport_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    base_config = VariableTransportConfig(**checkpoint["config"])
    config = LatentRelationConfig(
        seed=args.seed,
        transport_checkpoint=str(args.transport_checkpoint),
        episodes=args.episodes,
        batch_size=args.batch_size,
        pair_count=args.pair_count,
        query_count=args.query_count,
        support_counts=args.support_counts,
        root_budgets=args.root_budgets,
        data_seed=args.data_seed,
        token_relabel_seed=args.token_relabel_seed,
        device=args.device,
    )
    _validate_config(config, base_config)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.out_dir / "result.json"
    if result_path.exists():
        existing = json.loads(result_path.read_text())
        if (
            existing.get("complete")
            and existing.get("config") == asdict(config)
        ):
            print(
                json.dumps(
                    {
                        "status": "skipped_complete",
                        "result": str(result_path),
                    }
                ),
                flush=True,
            )
            return existing
        if existing.get("complete"):
            raise ValueError(
                f"completed result at {result_path} has a different config"
            )

    seed_everything(config.seed)
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    true_dictionary = checkpoint.get(
        "true_symbol_dictionary",
        symbol_dictionary(
            base_config.relation_dim,
            base_config.generator_dictionary_seed,
        ),
    )
    word_pools = build_word_pools(
        dimension=base_config.relation_dim,
        dictionary=true_dictionary,
        random_pool_size=base_config.word_pool_size,
        seed=base_config.word_pool_seed,
    )
    diagnostics = word_pool_diagnostics(
        word_pools,
        dictionary=true_dictionary,
        dimension=base_config.relation_dim,
    )

    evaluation: dict[str, dict[str, Any]] = {}
    for split_index, split in enumerate(HELD_WORD_SPLITS):
        evaluation[split] = {}
        for root_budget in config.root_budgets:
            print(
                json.dumps(
                    {
                        "status": "evaluating",
                        "split": split,
                        "root_budget": root_budget,
                    }
                ),
                flush=True,
            )
            evaluation[split][str(root_budget)] = evaluate_split(
                checkpoint_path=args.transport_checkpoint,
                word_pool=word_pools[split],
                root_budget=root_budget,
                support_counts=config.support_counts,
                episodes=config.episodes,
                batch_size=config.batch_size,
                pair_count=config.pair_count,
                query_count=config.query_count,
                data_seed=(
                    config.data_seed
                    + config.seed * 100_000_007
                    + split_index * 10_000_019
                    + root_budget * 1009
                ),
                device=device,
                relabel_seed=None,
            )

    gate_root_budget = max(config.root_budgets)
    gate_support_count = max(config.support_counts)
    gates = _build_gates(
        evaluation,
        root_budget=gate_root_budget,
        support_count=gate_support_count,
    )
    relabel_split = "unseen_l16"
    relabel_split_index = HELD_WORD_SPLITS.index(relabel_split)
    relabel_result = evaluate_split(
        checkpoint_path=args.transport_checkpoint,
        word_pool=word_pools[relabel_split],
        root_budget=gate_root_budget,
        support_counts=(gate_support_count,),
        episodes=config.episodes,
        batch_size=config.batch_size,
        pair_count=config.pair_count,
        query_count=config.query_count,
        data_seed=(
            config.data_seed
            + config.seed * 100_000_007
            + relabel_split_index * 10_000_019
            + gate_root_budget * 1009
        ),
        device=device,
        relabel_seed=config.token_relabel_seed,
    )
    base_relabel_cell = evaluation[relabel_split][
        str(gate_root_budget)
    ]["support_counts"][str(gate_support_count)]
    relabel_cell = relabel_result["support_counts"][
        str(gate_support_count)
    ]
    relation_relabel_retention = float(
        relabel_cell["relation_product_accuracy"]
    ) / max(float(base_relabel_cell["relation_product_accuracy"]), 1e-12)
    query_relabel_retention = float(relabel_cell["query_r1"]) / max(
        float(base_relabel_cell["query_r1"]),
        1e-12,
    )
    gates["token_relabel_relation_retention"] = {
        "value": relation_relabel_retention,
        "threshold": 0.95,
        "passed": relation_relabel_retention >= 0.95,
    }
    gates["token_relabel_query_retention"] = {
        "value": query_relabel_retention,
        "threshold": 0.95,
        "passed": query_relabel_retention >= 0.95,
    }
    all_passed = all(bool(gate["passed"]) for gate in gates.values())

    result: dict[str, Any] = {
        "complete": True,
        "config": asdict(config),
        "source_transport": {
            "checkpoint": str(args.transport_checkpoint),
            "source_trainable_parameters": int(
                base_config.relation_dim - 1
            )
            ** 2
            + 1,
            "latent_inference_trainable_parameters": 0,
            "learned_dictionary_accuracy": float(
                (
                    checkpoint["state_dict"]["generator_logits"].argmax(
                        dim=-1
                    )
                    == true_dictionary
                )
                .float()
                .mean()
                .item()
            ),
        },
        "architecture": {
            "hidden_episode_variable": (
                "one held reduced Coxeter word shared by every support and "
                "query pair"
            ),
            "inference_input": (
                "paired support Q/K ordinal states only; no word symbol, "
                "word index, or query retrieval label"
            ),
            "relation_search": (
                "uniform-prior exhaustive MAP over the same-length held "
                "relation bank"
            ),
            "selection_constraint": (
                "one inferred transport per episode, frozen across all "
                "query candidates"
            ),
            "query_negatives": (
                "one reverse-order orbit decoy per positive memory row"
            ),
            "complexity": "O(candidate_relations * supports * active_roots)",
        },
        "word_pool_diagnostics": {
            split: diagnostics[split]
            for split in HELD_WORD_SPLITS
        },
        "evaluation": evaluation,
        "token_relabel_control": {
            "split": relabel_split,
            "root_budget": gate_root_budget,
            "support_count": gate_support_count,
            "result": relabel_result,
            "relation_product_retention": relation_relabel_retention,
            "query_r1_retention": query_relabel_retention,
        },
        "gates": gates,
        "decision": (
            "advance_latent_relation_inference"
            if all_passed
            else "stop_latent_relation_inference"
        ),
    }
    atomic_json_write(result, result_path)
    print(
        json.dumps(
            {
                "status": "complete",
                "result": str(result_path),
                "decision": result["decision"],
                "gate_root_budget": gate_root_budget,
                "gate_support_count": gate_support_count,
            }
        ),
        flush=True,
    )
    return result


def summarize_results(args: argparse.Namespace) -> dict[str, Any]:
    result_paths = sorted(args.run_dir.glob("runs/seed*/result.json"))
    if not result_paths:
        raise FileNotFoundError(
            f"no complete seed results under {args.run_dir}"
        )
    results = [json.loads(path.read_text()) for path in result_paths]
    if not all(result.get("complete") for result in results):
        raise RuntimeError("at least one seed result is incomplete")
    config = results[0]["config"]
    root_budgets = tuple(config["root_budgets"])
    support_counts = tuple(config["support_counts"])

    summary_matrix: dict[str, dict[str, dict[str, Any]]] = {}
    fields = (
        "relation_product_accuracy",
        "unique_relation_accuracy",
        "mean_max_tie_count",
        "query_r1",
    )
    for split in HELD_WORD_SPLITS:
        summary_matrix[split] = {}
        for root_budget in root_budgets:
            root_key = str(root_budget)
            summary_matrix[split][root_key] = {}
            for support_count in support_counts:
                support_key = str(support_count)
                cell: dict[str, Any] = {}
                for field in fields:
                    values = torch.tensor(
                        [
                            result["evaluation"][split][root_key][
                                "support_counts"
                            ][support_key][field]
                            for result in results
                        ],
                        dtype=torch.float64,
                    )
                    cell[field] = {
                        "mean": float(values.mean().item()),
                        "sem": float(
                            values.std(unbiased=True).item()
                            / math.sqrt(values.numel())
                        )
                        if values.numel() > 1
                        else 0.0,
                    }
                summary_matrix[split][root_key][support_key] = cell

    summary = {
        "complete": True,
        "seed_count": len(results),
        "result_paths": [str(path) for path in result_paths],
        "root_budgets": list(root_budgets),
        "support_counts": list(support_counts),
        "matrix": summary_matrix,
        "gates": {
            name: {
                "values": [
                    float(result["gates"][name]["value"])
                    for result in results
                ],
                "all_passed": all(
                    bool(result["gates"][name]["passed"])
                    for result in results
                ),
            }
            for name in results[0]["gates"]
        },
        "decisions": [result["decision"] for result in results],
    }
    atomic_json_write(summary, args.out)
    print(
        json.dumps(
            {
                "status": "complete",
                "summary": str(args.out),
                "seed_count": len(results),
            }
        ),
        flush=True,
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser(
        "run",
        help="evaluate one frozen transport checkpoint",
    )
    run.add_argument("--seed", type=int, required=True)
    run.add_argument(
        "--transport-checkpoint",
        type=Path,
        required=True,
    )
    run.add_argument("--out-dir", type=Path, required=True)
    run.add_argument("--episodes", type=int, default=512)
    run.add_argument("--batch-size", type=int, default=64)
    run.add_argument("--pair-count", type=int, default=32)
    run.add_argument("--query-count", type=int, default=8)
    run.add_argument(
        "--support-counts",
        type=_parse_int_tuple,
        default=DEFAULT_SUPPORT_COUNTS,
    )
    run.add_argument(
        "--root-budgets",
        type=_parse_int_tuple,
        default=DEFAULT_ROOT_BUDGETS,
    )
    run.add_argument("--data-seed", type=int, default=202607261)
    run.add_argument("--token-relabel-seed", type=int, default=314159)
    run.add_argument("--device", default="cuda")
    run.set_defaults(function=run_experiment)

    summarize = subparsers.add_parser(
        "summarize",
        help="aggregate completed seed results",
    )
    summarize.add_argument("--run-dir", type=Path, required=True)
    summarize.add_argument("--out", type=Path, required=True)
    summarize.set_defaults(function=summarize_results)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
