"""Episode-varying Coxeter transports and long-path causal MQAR.

Every episode supplies a word in abstract generator symbols.  A hidden
bijection assigns those symbols to the adjacent transpositions of S_D.  The
model learns that small dictionary only from single-generator retrieval
episodes, then composes its hard generator choices on previously unseen words.

The key state is transformed by the episode's true word.  Relation scoring
uses a signed transport on a fixed subset of A_{D-1} roots.  Formal evaluation
separates:

* seen singleton generators;
* unseen reduced products of lengths 2, 4, 8, 16, and D-1;
* reversed composition order and identity-transport counterfactuals;
* one-, two-, four-, and eight-hop autoregressive retrieval;
* exact predecessor, frozen Canon, and current-row local composers.
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

from tropnn.tools.causal_mqar_composed_multihop import (
    CanonComposer,
    ComposedRunConfig,
)
from tropnn.tools.causal_mqar_induction import (
    FixedOrdinalCodebook,
    _clone_state_dict,
    _training_shape,
    atomic_json_write,
    atomic_torch_save,
    seed_everything,
    token_permutation,
)
from tropnn.tools.causal_mqar_role_gauge import token_pools
from tropnn.tools.causal_mqar_root_transport import (
    full_root_edges,
    nested_root_subset,
    oracle_signed_root_transport,
    root_signs,
)

WORD_LENGTHS = (1, 2, 4, 8, 16, 31)
HOP_LENGTHS = (1, 2, 4, 8)
EVALUATION_CONDITIONS = (
    "oracle_transport",
    "learned_transport",
    "learned_token_relabel",
    "reverse_order",
    "identity_transport",
    "learned_canon",
    "learned_canon_token_relabel",
    "learned_current",
)


@dataclass(frozen=True)
class VariableTransportConfig:
    seed: int
    canon_checkpoint: str
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
    root_budget: int
    root_subset_seed: int
    evaluation_pair_count: int
    word_pool_size: int
    assignment_temperature_start: float
    assignment_temperature_end: float
    assignment_entropy_weight: float
    assignment_coverage_weight: float
    positive_alignment_weight: float
    data_seed: int
    codebook_seed: int
    generator_dictionary_seed: int
    word_pool_seed: int
    token_relabel_seed: int
    device: str


@dataclass(frozen=True)
class CoxeterPathBatch:
    memory_keys: Tensor
    memory_values: Tensor
    memory_key_prepermutation: Tensor
    queries: Tensor
    hop_targets: Tensor
    hop_indices: Tensor
    word_symbols: Tensor
    actual_key_permutation: Tensor

    @property
    def batch_size(self) -> int:
        return int(self.memory_keys.shape[0])

    @property
    def pair_count(self) -> int:
        return int(self.memory_keys.shape[1])

    @property
    def query_count(self) -> int:
        return int(self.queries.shape[1])

    @property
    def hops(self) -> int:
        return int(self.hop_targets.shape[2])

    def to(self, device: torch.device) -> "CoxeterPathBatch":
        return CoxeterPathBatch(
            self.memory_keys.to(device, non_blocking=True),
            self.memory_values.to(device, non_blocking=True),
            self.memory_key_prepermutation.to(
                device,
                non_blocking=True,
            ),
            self.queries.to(device, non_blocking=True),
            self.hop_targets.to(device, non_blocking=True),
            self.hop_indices.to(device, non_blocking=True),
            self.word_symbols.to(device, non_blocking=True),
            self.actual_key_permutation.to(device, non_blocking=True),
        )


def adjacent_permutation(dimension: int, generator_index: int) -> Tensor:
    if not 0 <= generator_index < dimension - 1:
        raise ValueError("generator index is outside the adjacent-transposition range")
    permutation = torch.arange(dimension)
    permutation[generator_index], permutation[generator_index + 1] = (
        permutation[generator_index + 1].clone(),
        permutation[generator_index].clone(),
    )
    return permutation


def compose_generator_words(generator_indices: Tensor, dimension: int) -> Tensor:
    """Compose adjacent swaps from left to right as coordinate indexing maps."""

    if generator_indices.ndim == 1:
        generator_indices = generator_indices.unsqueeze(0)
    if generator_indices.ndim != 2:
        raise ValueError("generator words must have shape [batch, length]")
    if generator_indices.numel():
        if int(generator_indices.min().item()) < 0 or int(generator_indices.max().item()) >= dimension - 1:
            raise ValueError("generator word contains an invalid adjacent transposition")
    batch_size = int(generator_indices.shape[0])
    permutation = torch.arange(
        dimension,
        device=generator_indices.device,
    ).expand(batch_size, -1).clone()
    rows = torch.arange(batch_size, device=generator_indices.device)
    for column in range(generator_indices.shape[1]):
        left_index = generator_indices[:, column]
        right_index = left_index + 1
        left_value = permutation[rows, left_index].clone()
        right_value = permutation[rows, right_index].clone()
        permutation[rows, left_index] = right_value
        permutation[rows, right_index] = left_value
    return permutation


def coxeter_length(permutation: Tensor) -> Tensor:
    if permutation.ndim == 1:
        permutation = permutation.unsqueeze(0)
    comparisons = permutation[:, :, None] > permutation[:, None, :]
    return comparisons.triu(diagonal=1).sum(dim=(1, 2))


def symbol_dictionary(dimension: int, seed: int) -> Tensor:
    """Return the hidden symbol -> adjacent-generator bijection."""

    return torch.randperm(
        dimension - 1,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    )


def actual_word_permutations(
    words: Tensor,
    *,
    dictionary: Tensor,
    dimension: int,
) -> Tensor:
    return compose_generator_words(dictionary[words], dimension)


def _inverse_symbol_dictionary(dictionary: Tensor) -> Tensor:
    inverse = torch.empty_like(dictionary)
    inverse[dictionary] = torch.arange(dictionary.numel())
    return inverse


def _permutation_key(permutation: Tensor) -> tuple[int, ...]:
    return tuple(int(value) for value in permutation.tolist())


def _random_reduced_word_pool(
    *,
    dimension: int,
    dictionary: Tensor,
    length: int,
    count: int,
    min_reverse_distance: int,
    seed: int,
) -> Tensor:
    inverse_symbols = _inverse_symbol_dictionary(dictionary)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    accepted: list[Tensor] = []
    products: set[tuple[int, ...]] = set()
    attempts = 0
    maximum_attempts = max(20_000, 500 * count)
    while len(accepted) < count and attempts < maximum_attempts:
        attempts += 1
        permutation = torch.arange(dimension)
        actual_word: list[int] = []
        for _ in range(length):
            ascents = torch.nonzero(
                permutation[:-1] < permutation[1:],
                as_tuple=False,
            ).flatten()
            if ascents.numel() == 0:
                break
            selected = int(
                ascents[
                    torch.randint(
                        ascents.numel(),
                        (1,),
                        generator=generator,
                    )
                ].item()
            )
            actual_word.append(selected)
            left = permutation[selected].clone()
            permutation[selected] = permutation[selected + 1]
            permutation[selected + 1] = left
        if len(actual_word) != length:
            continue
        key = _permutation_key(permutation)
        if key in products:
            continue
        reverse = compose_generator_words(
            torch.tensor(actual_word[::-1]),
            dimension,
        )[0]
        reverse_distance = int((permutation != reverse).sum().item())
        if reverse_distance < min_reverse_distance:
            continue
        products.add(key)
        symbols = inverse_symbols[torch.tensor(actual_word)]
        accepted.append(symbols)
    if len(accepted) != count:
        raise RuntimeError(
            f"could construct only {len(accepted)}/{count} reduced words "
            f"of length {length}"
        )
    return torch.stack(accepted)


def build_word_pools(
    *,
    dimension: int,
    dictionary: Tensor,
    random_pool_size: int,
    seed: int,
) -> dict[str, Tensor]:
    generator_count = dimension - 1
    inverse_symbols = _inverse_symbol_dictionary(dictionary)
    pools: dict[str, Tensor] = {
        "seen_l1": torch.arange(generator_count).unsqueeze(1),
    }

    noncommuting: list[Tensor] = []
    for actual_left in range(generator_count - 1):
        actual_right = actual_left + 1
        noncommuting.append(
            inverse_symbols[torch.tensor([actual_left, actual_right])]
        )
        noncommuting.append(
            inverse_symbols[torch.tensor([actual_right, actual_left])]
        )
    pools["unseen_l2"] = torch.stack(noncommuting)

    minimum_distances = {
        4: 3,
        8: 6,
        16: 10,
        dimension - 1: max(12, dimension // 2),
    }
    for length in (4, 8, 16, dimension - 1):
        pools[f"unseen_l{length}"] = _random_reduced_word_pool(
            dimension=dimension,
            dictionary=dictionary,
            length=length,
            count=random_pool_size,
            min_reverse_distance=minimum_distances[length],
            seed=seed + length * 1009,
        )
    return pools


def word_pool_diagnostics(
    pools: dict[str, Tensor],
    *,
    dictionary: Tensor,
    dimension: int,
) -> dict[str, dict[str, float | int]]:
    seen_products = {
        _permutation_key(permutation)
        for permutation in actual_word_permutations(
            pools["seen_l1"],
            dictionary=dictionary,
            dimension=dimension,
        )
    }
    output: dict[str, dict[str, float | int]] = {}
    for name, words in pools.items():
        permutations = actual_word_permutations(
            words,
            dictionary=dictionary,
            dimension=dimension,
        )
        reverse = actual_word_permutations(
            words.flip(dims=(1,)),
            dictionary=dictionary,
            dimension=dimension,
        )
        lengths = coxeter_length(permutations)
        product_keys = {_permutation_key(row) for row in permutations}
        reverse_distance = (permutations != reverse).sum(dim=1).float()
        overlaps = len(product_keys.intersection(seen_products))
        output[name] = {
            "word_length": int(words.shape[1]),
            "word_count": int(words.shape[0]),
            "unique_products": len(product_keys),
            "minimum_coxeter_length": int(lengths.min().item()),
            "maximum_coxeter_length": int(lengths.max().item()),
            "seen_singleton_product_overlap": overlaps if name != "seen_l1" else len(product_keys),
            "minimum_reverse_permutation_distance": int(reverse_distance.min().item()),
            "mean_reverse_permutation_distance": float(reverse_distance.mean().item()),
        }
    return output


def generate_path_batch(
    *,
    token_pool: Tensor,
    word_pool: Tensor,
    dictionary: Tensor,
    dimension: int,
    batch_size: int,
    pair_count: int,
    hops: int,
    query_count: int,
    seed: int,
    pool_permutation: Tensor | None = None,
    order_hard_negatives: bool = False,
) -> CoxeterPathBatch:
    decoy_factor = 2 if order_hard_negatives else 1
    if hops < 1 or pair_count % (hops * decoy_factor):
        raise ValueError(
            "hops times the positive/decoy factor must divide pair_count"
        )
    positive_pair_count = pair_count // decoy_factor
    chain_count = positive_pair_count // hops
    if not 1 <= query_count <= chain_count:
        raise ValueError("query_count must lie in [1, chain_count]")
    pool_size = int(token_pool.numel())
    path_token_count = (hops + 1) * chain_count
    needed_tokens = path_token_count + (
        positive_pair_count
        if order_hard_negatives
        else 0
    )
    if needed_tokens > pool_size:
        raise ValueError("token pool is too small for disjoint path tokens")
    if pool_permutation is not None and pool_permutation.shape != (pool_size,):
        raise ValueError("pool permutation has the wrong shape")
    if word_pool.ndim != 2 or word_pool.shape[0] < 1:
        raise ValueError("word pool must have shape [words, length]")

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
    chain_tokens = token_pool[
        local_tokens[:, :path_token_count]
    ].reshape(
        batch_size,
        chain_count,
        hops + 1,
    )
    positive_keys = chain_tokens[:, :, :-1].reshape(
        batch_size,
        positive_pair_count,
    )
    positive_values = chain_tokens[:, :, 1:].reshape(
        batch_size,
        positive_pair_count,
    )

    word_ids = torch.randint(
        int(word_pool.shape[0]),
        (batch_size,),
        generator=generator,
    )
    word_symbols = word_pool[word_ids]
    actual_key_permutation = actual_word_permutations(
        word_symbols,
        dictionary=dictionary,
        dimension=dimension,
    )
    identity = torch.arange(dimension).expand(
        batch_size,
        positive_pair_count,
        -1,
    )
    if order_hard_negatives:
        reverse_key_permutation = actual_word_permutations(
            word_symbols.flip(dims=(1,)),
            dictionary=dictionary,
            dimension=dimension,
        )
        decoy_prepermutation = torch.empty_like(
            actual_key_permutation
        )
        decoy_prepermutation.scatter_(
            1,
            actual_key_permutation,
            reverse_key_permutation,
        )
        decoy_values = token_pool[
            local_tokens[:, path_token_count:]
        ]
        memory_keys = torch.cat(
            (positive_keys, positive_keys),
            dim=1,
        )
        memory_values = torch.cat(
            (positive_values, decoy_values),
            dim=1,
        )
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
    else:
        memory_keys = positive_keys
        memory_values = positive_values
        memory_key_prepermutation = identity

    row_priority = torch.rand(batch_size, pair_count, generator=generator)
    row_permutation = row_priority.argsort(dim=1)
    memory_keys = memory_keys.gather(1, row_permutation)
    memory_values = memory_values.gather(1, row_permutation)
    memory_key_prepermutation = memory_key_prepermutation.gather(
        1,
        row_permutation[:, :, None].expand(-1, -1, dimension),
    )
    inverse_rows = torch.empty_like(row_permutation)
    inverse_rows.scatter_(
        1,
        row_permutation,
        torch.arange(pair_count).expand(batch_size, -1),
    )

    query_priority = torch.rand(batch_size, chain_count, generator=generator)
    query_chains = query_priority.topk(
        query_count,
        dim=1,
        largest=False,
        sorted=False,
    ).indices
    selected_chains = chain_tokens.gather(
        1,
        query_chains[:, :, None].expand(-1, -1, hops + 1),
    )
    queries = selected_chains[:, :, 0]
    hop_targets = selected_chains[:, :, 1:]
    unshuffled_indices = (
        query_chains[:, :, None] * hops
        + torch.arange(hops)[None, None, :]
    )
    hop_indices = inverse_rows.gather(
        1,
        unshuffled_indices.flatten(1),
    ).reshape(batch_size, query_count, hops)

    return CoxeterPathBatch(
        memory_keys=memory_keys,
        memory_values=memory_values,
        memory_key_prepermutation=memory_key_prepermutation,
        queries=queries,
        hop_targets=hop_targets,
        hop_indices=hop_indices,
        word_symbols=word_symbols,
        actual_key_permutation=actual_key_permutation,
    )


def dynamic_signed_root_transport(
    key_permutation: Tensor,
    *,
    edges: Tensor,
    query_root_subset: Tensor,
    edge_index: Tensor,
) -> Tensor:
    """Return batch-specific signed K-root addresses for identity-gauge Q."""

    if key_permutation.ndim == 1:
        key_permutation = key_permutation.unsqueeze(0)
    inverse_key = torch.argsort(key_permutation, dim=-1)
    active_edges = edges[query_root_subset]
    key_left = inverse_key[:, active_edges[:, 0]]
    key_right = inverse_key[:, active_edges[:, 1]]
    low = torch.minimum(key_left, key_right)
    high = torch.maximum(key_left, key_right)
    root_index = edge_index[low, high]
    orientation = key_left < key_right
    root_count = int(edges.shape[0])
    return root_index + orientation.long() * root_count


def _load_frozen_canon(
    path: Path,
    *,
    dimension: int,
    codebook_seed: int,
) -> tuple[CanonComposer, ComposedRunConfig]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = ComposedRunConfig(**checkpoint["config"])
    if config.composer != "canon":
        raise ValueError("variable transport requires a Canon checkpoint")
    if config.relation_dim != dimension:
        raise ValueError("Canon relation dimension does not match")
    if config.codebook_seed != codebook_seed:
        raise ValueError("Canon codebook seed does not match")
    key = "composer.lag_logits"
    if key not in checkpoint["state_dict"]:
        raise ValueError("Canon checkpoint has no lag logits")
    composer = CanonComposer(dimension)
    with torch.no_grad():
        composer.lag_logits.copy_(checkpoint["state_dict"][key])
    composer.eval()
    for parameter in composer.parameters():
        parameter.requires_grad_(False)
    return composer, config


class VariableCoxeterTransport(nn.Module):
    """Learn generator symbols, then compose their hard root actions."""

    def __init__(self, config: VariableTransportConfig) -> None:
        super().__init__()
        self.config = config
        dimension = config.relation_dim
        generator_count = dimension - 1
        edges = full_root_edges(dimension)
        root_count = int(edges.shape[0])
        subset = nested_root_subset(
            root_count,
            config.root_budget,
            config.root_subset_seed,
        )
        edge_index = torch.full((dimension, dimension), -1, dtype=torch.long)
        root_ids = torch.arange(root_count)
        edge_index[edges[:, 0], edges[:, 1]] = root_ids
        edge_index[edges[:, 1], edges[:, 0]] = root_ids

        candidate_signed: list[Tensor] = []
        identity = torch.arange(dimension)
        for generator_index in range(generator_count):
            key_permutation = adjacent_permutation(
                dimension,
                generator_index,
            )
            key_root, orientation = oracle_signed_root_transport(
                identity,
                key_permutation,
                edges,
            )
            signed = key_root + (orientation > 0).long() * root_count
            candidate_signed.append(signed[subset])

        generator = torch.Generator(device="cpu").manual_seed(config.seed + 1901)
        self.generator_logits = nn.Parameter(
            torch.randn(
                generator_count,
                generator_count,
                generator=generator,
            )
            * 0.001
        )
        self.logit_scale = nn.Parameter(torch.zeros(()))
        self.temperature = 1.0
        self.codebook = FixedOrdinalCodebook(
            config.vocab_size,
            dimension,
            config.codebook_seed,
        )
        canon, canon_config = _load_frozen_canon(
            Path(config.canon_checkpoint),
            dimension=dimension,
            codebook_seed=config.codebook_seed,
        )
        self.canon = canon
        self.canon_config = canon_config
        self.register_buffer("edges", edges)
        self.register_buffer("query_root_subset", subset)
        self.register_buffer("edge_index", edge_index)
        self.register_buffer(
            "candidate_signed_indices",
            torch.stack(candidate_signed),
        )

    @property
    def root_count(self) -> int:
        return int(self.edges.shape[0])

    @property
    def generator_count(self) -> int:
        return self.config.relation_dim - 1

    @property
    def trainable_parameters(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    @property
    def hard_dictionary_bytes(self) -> int:
        return self.generator_count if self.generator_count <= 256 else 2 * self.generator_count

    def assignment_probabilities(self) -> Tensor:
        return (self.generator_logits / self.temperature).softmax(dim=-1)

    def hard_dictionary(self) -> Tensor:
        return self.generator_logits.argmax(dim=-1)

    def dictionary_accuracy(self, true_dictionary: Tensor) -> float:
        return float(
            (
                self.hard_dictionary().detach().cpu()
                == true_dictionary.detach().cpu()
            )
            .float()
            .mean()
            .item()
        )

    def entropy(self) -> Tensor:
        probabilities = self.assignment_probabilities()
        return -(
            probabilities
            * probabilities.clamp_min(1e-12).log()
        ).sum(dim=-1).mean()

    def coverage_loss(self) -> Tensor:
        column_mass = self.assignment_probabilities().sum(dim=0)
        return (column_mass - 1.0).square().mean()

    def _composed_coordinates(
        self,
        batch: CoxeterPathBatch,
        composer: str,
    ) -> Tensor:
        predecessor = self.codebook(batch.memory_keys)
        current = self.codebook(batch.memory_values)
        if composer == "oracle":
            composed = predecessor
        elif composer == "canon":
            composed = self.canon(predecessor, current)
        elif composer == "current":
            composed = current
        else:
            raise ValueError(
                f"unsupported local composer {composer!r}"
            )
        return composed.gather(
            -1,
            batch.memory_key_prepermutation,
        )

    def _root_features(
        self,
        query_tokens: Tensor,
        batch: CoxeterPathBatch,
        *,
        composer: str,
    ) -> tuple[Tensor, Tensor]:
        query_coordinates = self.codebook(query_tokens)
        composed = self._composed_coordinates(batch, composer)
        key_permutation = batch.actual_key_permutation[:, None, :].expand(
            -1,
            batch.pair_count,
            -1,
        )
        key_coordinates = composed.gather(-1, key_permutation)
        return (
            root_signs(query_coordinates, self.edges),
            root_signs(key_coordinates, self.edges),
        )

    def soft_singleton_scores(
        self,
        query_tokens: Tensor,
        batch: CoxeterPathBatch,
    ) -> Tensor:
        scores, _ = self.soft_singleton_objective(
            query_tokens,
            batch,
        )
        return scores

    def soft_singleton_objective(
        self,
        query_tokens: Tensor,
        batch: CoxeterPathBatch,
    ) -> tuple[Tensor, Tensor]:
        """Return retrieval scores and label-selected root alignment loss."""

        if batch.word_symbols.shape[1] != 1:
            raise ValueError("soft training is defined only for singleton words")
        query_roots, key_roots = self._root_features(
            query_tokens,
            batch,
            composer="oracle",
        )
        query_active = query_roots[..., self.query_root_subset]
        signed_key = torch.cat((-key_roots, key_roots), dim=-1)
        batch_size, pair_count = key_roots.shape[:2]
        indices = self.candidate_signed_indices[None, None, :, :].expand(
            batch_size,
            pair_count,
            -1,
            -1,
        )
        aligned = signed_key[:, :, None, :].expand(
            -1,
            -1,
            self.generator_count,
            -1,
        ).gather(-1, indices)
        candidate_scores = torch.einsum(
            "bqu,bpgu->bqpg",
            query_active,
            aligned,
        ) / math.sqrt(self.config.root_budget)
        probabilities = self.assignment_probabilities()[
            batch.word_symbols[:, 0]
        ]
        scores = torch.einsum(
            "bqpg,bg->bqp",
            candidate_scores,
            probabilities,
        )
        positive_indices = batch.hop_indices[:, :, 0]
        positive_aligned = aligned.gather(
            1,
            positive_indices[:, :, None, None].expand(
                -1,
                -1,
                self.generator_count,
                self.config.root_budget,
            ),
        )
        candidate_agreement = (
            query_active[:, :, None, :]
            * positive_aligned
        ).mean(dim=-1)
        expected_agreement = (
            candidate_agreement
            * probabilities[:, None, :]
        ).sum(dim=-1)
        alignment_loss = (1.0 - expected_agreement).mean()
        return (
            self.logit_scale.exp().clamp(max=100.0) * scores,
            alignment_loss,
        )

    def _predicted_permutation(
        self,
        batch: CoxeterPathBatch,
        transport: str,
    ) -> Tensor:
        if transport == "oracle":
            return batch.actual_key_permutation
        if transport == "identity":
            return torch.arange(
                self.config.relation_dim,
                device=batch.word_symbols.device,
            ).expand(batch.batch_size, -1)
        if transport not in {"learned", "reverse"}:
            raise ValueError(f"unsupported transport {transport!r}")
        words = batch.word_symbols
        if transport == "reverse":
            words = words.flip(dims=(1,))
        generator_indices = self.hard_dictionary()[words]
        return compose_generator_words(
            generator_indices,
            self.config.relation_dim,
        )

    def hard_scores(
        self,
        query_tokens: Tensor,
        batch: CoxeterPathBatch,
        *,
        transport: str,
        composer: str,
    ) -> Tensor:
        query_roots, key_roots = self._root_features(
            query_tokens,
            batch,
            composer=composer,
        )
        query_active = query_roots[..., self.query_root_subset]
        predicted_permutation = self._predicted_permutation(
            batch,
            transport,
        )
        signed_indices = dynamic_signed_root_transport(
            predicted_permutation,
            edges=self.edges,
            query_root_subset=self.query_root_subset,
            edge_index=self.edge_index,
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
        ) / math.sqrt(self.config.root_budget)
        return self.logit_scale.exp().clamp(max=100.0) * scores


@torch.no_grad()
def evaluate_path(
    model: VariableCoxeterTransport,
    *,
    token_pool: Tensor,
    word_pool: Tensor,
    dictionary: Tensor,
    pair_count: int,
    hops: int,
    episodes: int,
    batch_size: int,
    data_seed: int,
    device: torch.device,
    transport: str,
    composer: str,
    pool_permutation: Tensor | None,
    order_hard_negatives: bool,
) -> dict[str, Any]:
    model.eval()
    decoy_factor = 2 if order_hard_negatives else 1
    chain_count = pair_count // (hops * decoy_factor)
    query_count = min(8, chain_count)
    total_queries = 0
    hop_correct = [0 for _ in range(hops)]
    final_correct = 0
    all_hops_correct = 0
    exact_episodes = 0
    ce_sum = 0.0
    for offset in range(0, episodes, batch_size):
        current_batch_size = min(batch_size, episodes - offset)
        batch = generate_path_batch(
            token_pool=token_pool,
            word_pool=word_pool,
            dictionary=dictionary,
            dimension=model.config.relation_dim,
            batch_size=current_batch_size,
            pair_count=pair_count,
            hops=hops,
            query_count=query_count,
            seed=data_seed + offset * 1009,
            pool_permutation=pool_permutation,
            order_hard_negatives=order_hard_negatives,
        ).to(device)
        query = batch.queries
        path_correct = torch.ones(
            current_batch_size,
            query_count,
            dtype=torch.bool,
            device=device,
        )
        for hop in range(hops):
            scores = model.hard_scores(
                query,
                batch,
                transport=transport,
                composer=composer,
            )
            selected = scores.argmax(dim=-1)
            predicted = (
                batch.memory_values[:, None, :]
                .expand(-1, query_count, -1)
                .gather(2, selected.unsqueeze(-1))
                .squeeze(-1)
            )
            correct = predicted == batch.hop_targets[:, :, hop]
            path_correct &= correct
            hop_correct[hop] += int(correct.sum().item())
            ce_sum += float(
                F.cross_entropy(
                    scores.flatten(0, 1),
                    batch.hop_indices[:, :, hop].flatten(),
                    reduction="sum",
                ).item()
            )
            query = predicted
        final = query == batch.hop_targets[:, :, -1]
        final_correct += int(final.sum().item())
        all_hops_correct += int(path_correct.sum().item())
        exact_episodes += int(final.all(dim=1).sum().item())
        total_queries += current_batch_size * query_count
    return {
        "episodes": episodes,
        "queries": total_queries,
        "pair_count": pair_count,
        "chain_count": chain_count,
        "query_count": query_count,
        "hops": hops,
        "hop_accuracy": [
            value / total_queries
            for value in hop_correct
        ],
        "final_accuracy": final_correct / total_queries,
        "all_hops_accuracy": all_hops_correct / total_queries,
        "multiquery_final_exact_accuracy": exact_episodes / episodes,
        "autoregressive_mean_ce": ce_sum / (total_queries * hops),
        "random_final_accuracy": 1.0 / pair_count,
    }


def _temperature(config: VariableTransportConfig, step: int) -> float:
    if config.steps <= 1:
        return config.assignment_temperature_end
    fraction = (step - 1) / (config.steps - 1)
    ratio = (
        config.assignment_temperature_end
        / config.assignment_temperature_start
    )
    return config.assignment_temperature_start * ratio**fraction


def _condition_arguments(
    condition: str,
) -> tuple[str, str, bool]:
    if condition == "oracle_transport":
        return "oracle", "oracle", False
    if condition == "learned_transport":
        return "learned", "oracle", False
    if condition == "learned_token_relabel":
        return "learned", "oracle", True
    if condition == "reverse_order":
        return "reverse", "oracle", False
    if condition == "identity_transport":
        return "identity", "oracle", False
    if condition == "learned_canon":
        return "learned", "canon", False
    if condition == "learned_canon_token_relabel":
        return "learned", "canon", True
    if condition == "learned_current":
        return "learned", "current", False
    raise ValueError(f"unsupported evaluation condition {condition!r}")


def _validate_config(config: VariableTransportConfig) -> None:
    if config.relation_dim != 32:
        raise ValueError("the formal word-length protocol currently requires D32")
    root_count = config.relation_dim * (config.relation_dim - 1) // 2
    if not 1 <= config.root_budget <= root_count:
        raise ValueError("root budget is outside the A31 carrier")
    if config.evaluation_pair_count % max(HOP_LENGTHS):
        raise ValueError("evaluation pair count must be divisible by every hop length")
    if config.vocab_size != (
        config.train_tokens
        + config.validation_tokens
        + config.test_tokens
    ):
        raise ValueError("token pools must exactly partition the vocabulary")
    if min(
        config.train_tokens,
        config.validation_tokens,
        config.test_tokens,
    ) < 2 * config.evaluation_pair_count:
        raise ValueError("every token pool must support the one-hop P32 task")
    if config.word_pool_size < 16:
        raise ValueError("word pool size must be at least 16")
    if config.assignment_temperature_start <= 0:
        raise ValueError("assignment temperature must be positive")
    if config.assignment_temperature_end <= 0:
        raise ValueError("assignment temperature must be positive")
    if not Path(config.canon_checkpoint).is_file():
        raise ValueError(f"missing Canon checkpoint {config.canon_checkpoint}")


def _build_config(args: argparse.Namespace) -> VariableTransportConfig:
    return VariableTransportConfig(
        seed=args.seed,
        canon_checkpoint=str(args.canon_checkpoint),
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
        root_budget=args.root_budget,
        root_subset_seed=args.root_subset_seed,
        evaluation_pair_count=args.evaluation_pair_count,
        word_pool_size=args.word_pool_size,
        assignment_temperature_start=args.assignment_temperature_start,
        assignment_temperature_end=args.assignment_temperature_end,
        assignment_entropy_weight=args.assignment_entropy_weight,
        assignment_coverage_weight=args.assignment_coverage_weight,
        positive_alignment_weight=args.positive_alignment_weight,
        data_seed=args.data_seed,
        codebook_seed=args.codebook_seed,
        generator_dictionary_seed=args.generator_dictionary_seed,
        word_pool_seed=args.word_pool_seed,
        token_relabel_seed=args.token_relabel_seed,
        device=args.device,
    )


def _mean_condition(
    evaluation: dict[str, dict[str, dict[str, Any]]],
    *,
    words: tuple[str, ...],
    hops: tuple[int, ...],
    condition: str,
) -> float:
    values = [
        float(evaluation[word][f"h{hop}"][condition]["final_accuracy"])
        for word in words
        for hop in hops
    ]
    return sum(values) / len(values)


def _build_gates(
    evaluation: dict[str, dict[str, dict[str, Any]]],
    *,
    dictionary_accuracy: float,
) -> dict[str, dict[str, float | bool]]:
    chance = 1.0 / 32
    unseen = (
        "unseen_l2",
        "unseen_l4",
        "unseen_l8",
        "unseen_l16",
        "unseen_l31",
    )
    learned_unseen_h1 = _mean_condition(
        evaluation,
        words=unseen,
        hops=(1,),
        condition="learned_transport",
    )
    learned_long_h8 = _mean_condition(
        evaluation,
        words=("unseen_l16", "unseen_l31"),
        hops=(8,),
        condition="learned_transport",
    )
    reverse_long_h1 = _mean_condition(
        evaluation,
        words=("unseen_l8", "unseen_l16", "unseen_l31"),
        hops=(1,),
        condition="reverse_order",
    )
    reverse_removal = 1.0 - (
        (reverse_long_h1 - chance)
        / max(learned_unseen_h1 - chance, 1e-12)
    )
    canon_long_h8 = _mean_condition(
        evaluation,
        words=("unseen_l16", "unseen_l31"),
        hops=(8,),
        condition="learned_canon",
    )
    current_long_h1 = _mean_condition(
        evaluation,
        words=("unseen_l16", "unseen_l31"),
        hops=(1,),
        condition="learned_current",
    )
    oracle_retention_values: list[float] = []
    relabel_values: list[float] = []
    canon_relabel_values: list[float] = []
    for word in unseen:
        for hop in HOP_LENGTHS:
            base = float(
                evaluation[word][f"h{hop}"]["learned_transport"][
                    "final_accuracy"
                ]
            )
            oracle = float(
                evaluation[word][f"h{hop}"]["oracle_transport"][
                    "final_accuracy"
                ]
            )
            relabel = float(
                evaluation[word][f"h{hop}"]["learned_token_relabel"][
                    "final_accuracy"
                ]
            )
            canon = float(
                evaluation[word][f"h{hop}"]["learned_canon"][
                    "final_accuracy"
                ]
            )
            canon_relabel = float(
                evaluation[word][f"h{hop}"][
                    "learned_canon_token_relabel"
                ]["final_accuracy"]
            )
            oracle_retention_values.append(
                base / max(oracle, 1e-12)
            )
            relabel_values.append(relabel / max(base, 1e-12))
            canon_relabel_values.append(
                canon_relabel / max(canon, 1e-12)
            )
    quantities = {
        "generator_dictionary_accuracy": (
            dictionary_accuracy,
            0.95,
            "min",
        ),
        "seen_singleton_h1_accuracy": (
            float(
                evaluation["seen_l1"]["h1"]["learned_transport"][
                    "final_accuracy"
                ]
            ),
            0.99,
            "min",
        ),
        "unseen_product_h1_mean_accuracy": (
            learned_unseen_h1,
            0.98,
            "min",
        ),
        "long_product_h8_mean_accuracy": (
            learned_long_h8,
            0.95,
            "min",
        ),
        "learned_retains_oracle_product_accuracy": (
            sum(oracle_retention_values)
            / len(oracle_retention_values),
            0.99,
            "min",
        ),
        "reverse_order_gain_removal": (
            reverse_removal,
            0.50,
            "min",
        ),
        "canon_long_product_h8_accuracy": (
            canon_long_h8,
            0.80,
            "min",
        ),
        "current_row_long_product_h1_negative": (
            current_long_h1,
            chance + 0.05,
            "max",
        ),
        "unseen_token_relabel_mean_retention": (
            sum(relabel_values) / len(relabel_values),
            0.95,
            "min",
        ),
        "canon_token_relabel_mean_retention": (
            sum(canon_relabel_values)
            / len(canon_relabel_values),
            0.95,
            "min",
        ),
    }
    gates: dict[str, dict[str, float | bool]] = {}
    for name, (value, threshold, direction) in quantities.items():
        passed = (
            value >= threshold
            if direction == "min"
            else value <= threshold
        )
        gates[name] = {
            "value": value,
            "threshold": threshold,
            "passed": passed,
        }
    return gates


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    config = _build_config(args)
    _validate_config(config)
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
    dictionary = symbol_dictionary(
        config.relation_dim,
        config.generator_dictionary_seed,
    )
    word_pools = build_word_pools(
        dimension=config.relation_dim,
        dictionary=dictionary,
        random_pool_size=config.word_pool_size,
        seed=config.word_pool_seed,
    )
    diagnostics = word_pool_diagnostics(
        word_pools,
        dictionary=dictionary,
        dimension=config.relation_dim,
    )
    for name, diagnostic in diagnostics.items():
        expected_length = int(word_pools[name].shape[1])
        if diagnostic["minimum_coxeter_length"] != expected_length:
            raise RuntimeError(f"{name} contains a non-reduced word")
        if diagnostic["maximum_coxeter_length"] != expected_length:
            raise RuntimeError(f"{name} contains a non-reduced word")
        if name != "seen_l1":
            if diagnostic["seen_singleton_product_overlap"] != 0:
                raise RuntimeError(f"{name} overlaps training products")
            if diagnostic["minimum_reverse_permutation_distance"] <= 0:
                raise RuntimeError(f"{name} does not test composition order")

    pools = token_pools(config)
    model = VariableCoxeterTransport(config).to(device)
    trainable = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    best_validation_objective = float("inf")
    best_step = 0
    best_state = _clone_state_dict(model)
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()

    for step in range(1, config.steps + 1):
        model.temperature = _temperature(config, step)
        pair_count, query_count = _training_shape(
            step,
            config.data_seed,
        )
        batch = generate_path_batch(
            token_pool=pools["train"],
            word_pool=word_pools["seen_l1"],
            dictionary=dictionary,
            dimension=config.relation_dim,
            batch_size=config.batch_size,
            pair_count=pair_count,
            hops=1,
            query_count=query_count,
            seed=config.data_seed + step * 1_000_003,
        ).to(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        scores, positive_alignment = model.soft_singleton_objective(
            batch.queries,
            batch,
        )
        retrieval_ce = F.cross_entropy(
            scores.flatten(0, 1),
            batch.hop_indices[:, :, 0].flatten(),
        )
        entropy = model.entropy()
        coverage = model.coverage_loss()
        loss = (
            retrieval_ce
            + config.assignment_entropy_weight * entropy
            + config.assignment_coverage_weight * coverage
            + config.positive_alignment_weight * positive_alignment
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        loss.backward()
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                trainable,
                config.gradient_clip,
            ).item()
        )
        optimizer.step()

        should_validate = (
            step == 1
            or step % config.validation_interval == 0
            or step == config.steps
        )
        if should_validate:
            validation_batch = generate_path_batch(
                token_pool=pools["validation"],
                word_pool=word_pools["seen_l1"],
                dictionary=dictionary,
                dimension=config.relation_dim,
                batch_size=config.validation_episodes,
                pair_count=8,
                hops=1,
                query_count=4,
                seed=config.data_seed + 50_000_000,
            ).to(device)
            model.eval()
            with torch.no_grad():
                (
                    validation_soft_scores,
                    validation_alignment,
                ) = model.soft_singleton_objective(
                    validation_batch.queries,
                    validation_batch,
                )
                validation_soft_ce = F.cross_entropy(
                    validation_soft_scores.flatten(0, 1),
                    validation_batch.hop_indices[:, :, 0].flatten(),
                )
                validation_objective = (
                    validation_soft_ce
                    + config.positive_alignment_weight
                    * validation_alignment
                )
            validation = evaluate_path(
                model,
                token_pool=pools["validation"],
                word_pool=word_pools["seen_l1"],
                dictionary=dictionary,
                pair_count=8,
                hops=1,
                episodes=config.validation_episodes,
                batch_size=config.evaluation_batch_size,
                data_seed=config.data_seed + 50_000_000,
                device=device,
                transport="learned",
                composer="oracle",
                pool_permutation=None,
                order_hard_negatives=False,
            )
            validation_ce = float(validation_soft_ce.item())
            validation_objective_value = float(
                validation_objective.item()
            )
            row = {
                "step": step,
                "temperature": model.temperature,
                "train_retrieval_ce": float(retrieval_ce.item()),
                "assignment_entropy": float(entropy.item()),
                "assignment_coverage_loss": float(coverage.item()),
                "positive_alignment_loss": float(
                    positive_alignment.item()
                ),
                "gradient_norm": gradient_norm,
                "validation_ce": validation_ce,
                "validation_alignment_loss": float(
                    validation_alignment.item()
                ),
                "validation_objective": validation_objective_value,
                "validation_accuracy": float(
                    validation["final_accuracy"]
                ),
                "dictionary_accuracy": model.dictionary_accuracy(
                    dictionary
                ),
            }
            history.append(row)
            print(json.dumps(row), flush=True)
            if validation_objective_value < best_validation_objective:
                best_validation_objective = validation_objective_value
                best_step = step
                best_state = _clone_state_dict(model)

    training_seconds = time.perf_counter() - started
    model.load_state_dict(best_state)
    test_relabel = token_permutation(
        config.test_tokens,
        config.token_relabel_seed,
    )
    evaluation: dict[str, dict[str, dict[str, Any]]] = {}
    for word_name, word_pool in word_pools.items():
        evaluation[word_name] = {}
        for hops in HOP_LENGTHS:
            hop_name = f"h{hops}"
            evaluation[word_name][hop_name] = {}
            for condition in EVALUATION_CONDITIONS:
                transport, composer, relabel = _condition_arguments(
                    condition
                )
                evaluation[word_name][hop_name][
                    condition
                ] = evaluate_path(
                    model,
                    token_pool=pools["test"],
                    word_pool=word_pool,
                    dictionary=dictionary,
                    pair_count=config.evaluation_pair_count,
                    hops=hops,
                    episodes=config.evaluation_episodes,
                    batch_size=config.evaluation_batch_size,
                    data_seed=(
                        config.data_seed
                        + 70_000_000
                        + int(word_pool.shape[1]) * 100_003
                        + hops * 10_007
                    ),
                    device=device,
                    transport=transport,
                    composer=composer,
                    pool_permutation=(
                        test_relabel
                        if relabel
                        else None
                    ),
                    order_hard_negatives=(
                        word_name != "seen_l1"
                    ),
                )

    dictionary_accuracy = model.dictionary_accuracy(dictionary)
    gates = _build_gates(
        evaluation,
        dictionary_accuracy=dictionary_accuracy,
    )
    all_passed = all(
        bool(gate["passed"])
        for gate in gates.values()
    )
    result: dict[str, Any] = {
        "complete": True,
        "config": asdict(config),
        "architecture": {
            "training_products": "31 singleton Coxeter generators only",
            "held_products": "reduced words of lengths 2,4,8,16,31",
            "held_negatives": (
                "one order-orbit decoy per positive edge; reverse order "
                "exactly aligns the decoy"
            ),
            "learned_object": (
                "31-symbol categorical dictionary into adjacent "
                "transpositions"
            ),
            "hard_transport": (
                "compose selected adjacent swaps, derive signed active-root "
                "addresses, XOR/popcount"
            ),
            "local_composers": (
                "exact predecessor, frozen Canon, current-row negative"
            ),
            "path_evaluation": (
                "autoregressive value-to-query feedback for 1/2/4/8 hops"
            ),
        },
        "trainable_parameters": model.trainable_parameters,
        "hard_generator_dictionary_bytes": (
            model.hard_dictionary_bytes
        ),
        "hard_root_products_per_pair": config.root_budget,
        "true_symbol_dictionary": dictionary.tolist(),
        "learned_symbol_dictionary": (
            model.hard_dictionary().detach().cpu().tolist()
        ),
        "dictionary_accuracy": dictionary_accuracy,
        "word_pool_diagnostics": diagnostics,
        "best_step": best_step,
        "best_validation_objective": best_validation_objective,
        "training": {
            "seconds": training_seconds,
            "history": history,
        },
        "evaluation": evaluation,
        "gates": gates,
        "decision": (
            "advance_variable_transport_composition"
            if all_passed
            else "stop_variable_transport_composition"
        ),
    }
    checkpoint = {
        "config": asdict(config),
        "best_step": best_step,
        "state_dict": best_state,
        "true_symbol_dictionary": dictionary,
    }
    atomic_torch_save(checkpoint, args.out_dir / "best.pt")
    atomic_json_write(result, result_path)
    print(
        json.dumps(
            {
                "status": "complete",
                "result": str(result_path),
                "best_step": best_step,
                "dictionary_accuracy": dictionary_accuracy,
                "decision": result["decision"],
            }
        ),
        flush=True,
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser(
        "run",
        help="train one generator dictionary and run the product/path matrix",
    )
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--canon-checkpoint", type=Path, required=True)
    run.add_argument("--out-dir", type=Path, required=True)
    run.add_argument("--steps", type=int, default=1500)
    run.add_argument("--batch-size", type=int, default=128)
    run.add_argument("--learning-rate", type=float, default=0.1)
    run.add_argument("--weight-decay", type=float, default=0.0)
    run.add_argument("--gradient-clip", type=float, default=5.0)
    run.add_argument("--validation-interval", type=int, default=100)
    run.add_argument("--validation-episodes", type=int, default=512)
    run.add_argument("--evaluation-episodes", type=int, default=512)
    run.add_argument("--evaluation-batch-size", type=int, default=64)
    run.add_argument("--vocab-size", type=int, default=768)
    run.add_argument("--train-tokens", type=int, default=512)
    run.add_argument("--validation-tokens", type=int, default=128)
    run.add_argument("--test-tokens", type=int, default=128)
    run.add_argument("--relation-dim", type=int, default=32)
    run.add_argument("--root-budget", type=int, default=64)
    run.add_argument("--root-subset-seed", type=int, default=4242)
    run.add_argument("--evaluation-pair-count", type=int, default=32)
    run.add_argument("--word-pool-size", type=int, default=64)
    run.add_argument(
        "--assignment-temperature-start",
        type=float,
        default=2.0,
    )
    run.add_argument(
        "--assignment-temperature-end",
        type=float,
        default=0.05,
    )
    run.add_argument(
        "--assignment-entropy-weight",
        type=float,
        default=1e-3,
    )
    run.add_argument(
        "--assignment-coverage-weight",
        type=float,
        default=1e-2,
    )
    run.add_argument(
        "--positive-alignment-weight",
        type=float,
        default=1.0,
    )
    run.add_argument("--data-seed", type=int, default=1729)
    run.add_argument("--codebook-seed", type=int, default=2718)
    run.add_argument(
        "--generator-dictionary-seed",
        type=int,
        default=202607251,
    )
    run.add_argument(
        "--word-pool-seed",
        type=int,
        default=202607252,
    )
    run.add_argument("--token-relabel-seed", type=int, default=314159)
    run.add_argument("--device", default="cuda")
    run.set_defaults(function=run_experiment)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
