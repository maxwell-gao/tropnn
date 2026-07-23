"""No-bank Coxeter program induction from partial noisy ordinal supports.

Every episode contains a hidden permutation represented only by paired support
states.  Each support exposes a small, independently sampled subset of
comparison roots, and both Q and K signs can be flipped by observation noise.
There is no bank of complete relation hypotheses.

Controllers must emit a sequence of generator symbols followed by STOP.  The
symbols are composed online into one transport, which is then used for held
P32 retrieval.  Training products have reduced Coxeter length 1-4; evaluation
products have lengths 8 and 16 and never occur in training.

The structured controller uses the defining local property of the A-type root
system: an adjacent value transposition flips exactly one comparison root.
Dense and JointPair PC-LUT controllers receive the same aggregated support
evidence and current program state but learn the next action by teacher-forced
cross entropy.  A random-permutation dictionary tests whether generic
composition alone provides the same length extrapolation.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tropnn.layers import PairwiseLUT
from tropnn.tools.causal_mqar_induction import (
    _clone_state_dict,
    atomic_json_write,
    atomic_torch_save,
    seed_everything,
)
from tropnn.tools.causal_mqar_root_transport import (
    full_root_edges,
    root_signs,
)
from tropnn.tools.causal_mqar_variable_transport import (
    _random_reduced_word_pool,
    coxeter_length,
    dynamic_signed_root_transport,
)

TRAIN_LENGTHS = (1, 2, 3, 4)
TEST_LENGTHS = (8, 16)
CONTROLLER_KINDS = ("dense", "jointpair")


@dataclass(frozen=True)
class ProgramInferenceConfig:
    seed: int
    steps: int
    batch_size: int
    learning_rate: float
    jointpair_learning_rate: float
    weight_decay: float
    gradient_clip: float
    validation_interval: int
    validation_episodes: int
    evaluation_episodes: int
    dimension: int
    support_count: int
    roots_per_support: int
    root_noise: float
    maximum_decode_steps: int
    dense_hidden: int
    jointpair_tables: int
    jointpair_comparisons: int
    retrieval_candidates: int
    train_pool_size: int
    test_pool_size: int
    word_pool_seed: int
    generator_relabel_seed: int
    random_dictionary_seed: int
    data_seed: int
    device: str


@dataclass(frozen=True)
class RootEvidence:
    votes: Tensor
    counts: Tensor
    support_count: int
    roots_per_support: int

    @property
    def batch_size(self) -> int:
        return int(self.votes.shape[0])

    def to(self, device: torch.device) -> "RootEvidence":
        return RootEvidence(
            votes=self.votes.to(device, non_blocking=True),
            counts=self.counts.to(device, non_blocking=True),
            support_count=self.support_count,
            roots_per_support=self.roots_per_support,
        )


@dataclass(frozen=True)
class DecodedPrograms:
    labels: Tensor
    lengths: Tensor
    stopped: Tensor
    permutations: Tensor


def adjacent_action_dictionary(dimension: int) -> Tensor:
    actions = []
    for generator in range(dimension - 1):
        action = torch.arange(dimension)
        action[generator], action[generator + 1] = (
            action[generator + 1].clone(),
            action[generator].clone(),
        )
        actions.append(action)
    return torch.stack(actions)


def compose_left(
    current: Tensor,
    actions: Tensor,
) -> Tensor:
    """Apply action permutations on the left: new = action o current."""

    if actions.ndim == 1:
        actions = actions.unsqueeze(0)
    return actions.gather(1, current)


def compose_right(
    current: Tensor,
    actions: Tensor,
) -> Tensor:
    """Wrong-law control: new = current o action."""

    if actions.ndim == 1:
        actions = actions.unsqueeze(0)
    return current.gather(1, actions)


def compose_label_programs(
    labels: Tensor,
    lengths: Tensor,
    *,
    action_dictionary: Tensor,
    law: Literal["left", "right", "sorted_left"] = "left",
) -> Tensor:
    if labels.ndim != 2:
        raise ValueError("program labels must have shape [batch, steps]")
    batch_size = int(labels.shape[0])
    dimension = int(action_dictionary.shape[1])
    output = torch.arange(
        dimension,
        device=labels.device,
    ).expand(batch_size, -1).clone()
    if law == "sorted_left":
        sorted_labels = labels.clone()
        sentinel = int(action_dictionary.shape[0])
        valid = (
            torch.arange(labels.shape[1], device=labels.device)[None, :]
            < lengths[:, None]
        )
        sorted_labels[~valid] = sentinel
        sorted_labels = sorted_labels.sort(dim=1).values
        sorted_labels[sorted_labels == sentinel] = 0
        labels = sorted_labels
    for step in range(labels.shape[1]):
        active = step < lengths
        if not bool(active.any()):
            break
        action = action_dictionary[labels[active, step]]
        if law in {"left", "sorted_left"}:
            output[active] = compose_left(output[active], action)
        elif law == "right":
            output[active] = compose_right(output[active], action)
        else:
            raise ValueError(f"unsupported composition law {law!r}")
    return output


def _permutation_key(permutation: Tensor) -> tuple[int, ...]:
    return tuple(int(value) for value in permutation.tolist())


def _sample_reduced_pool(
    *,
    dimension: int,
    length: int,
    count: int,
    seed: int,
) -> Tensor:
    if length == 1:
        return torch.arange(dimension - 1).unsqueeze(1)
    identity_dictionary = torch.arange(dimension - 1)
    return _random_reduced_word_pool(
        dimension=dimension,
        dictionary=identity_dictionary,
        length=length,
        count=count,
        min_reverse_distance=0,
        seed=seed,
    )


def _compose_right_words(words: Tensor, dimension: int) -> Tensor:
    batch_size = int(words.shape[0])
    output = torch.arange(dimension).expand(batch_size, -1).clone()
    actions = adjacent_action_dictionary(dimension)
    for step in range(words.shape[1]):
        output = compose_right(output, actions[words[:, step]])
    return output


def build_coxeter_word_pools(
    config: ProgramInferenceConfig,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    maximum_train_counts = {
        1: config.dimension - 1,
        2: min(config.train_pool_size, 24),
        3: min(config.train_pool_size, 64),
        4: config.train_pool_size,
    }
    words: dict[str, Tensor] = {}
    products: dict[str, Tensor] = {}
    seen_products: set[tuple[int, ...]] = set()
    for length in TRAIN_LENGTHS + TEST_LENGTHS:
        count = (
            maximum_train_counts[length]
            if length in TRAIN_LENGTHS
            else config.test_pool_size
        )
        name = f"l{length}"
        pool = _sample_reduced_pool(
            dimension=config.dimension,
            length=length,
            count=count,
            seed=config.word_pool_seed + length * 1009,
        )
        product = _compose_right_words(pool, config.dimension)
        if not bool((coxeter_length(product) == length).all()):
            raise RuntimeError(f"{name} contains a non-reduced word")
        keys = {_permutation_key(row) for row in product}
        if len(keys) != product.shape[0]:
            raise RuntimeError(f"{name} contains duplicate products")
        if length in TEST_LENGTHS and keys.intersection(seen_products):
            raise RuntimeError(f"{name} overlaps a training product")
        if length in TRAIN_LENGTHS:
            seen_products.update(keys)
        words[name] = pool
        products[name] = product
    return words, products


def random_action_dictionary(
    dimension: int,
    seed: int,
) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    identity = tuple(range(dimension))
    seen = {identity}
    actions: list[Tensor] = []
    while len(actions) < dimension - 1:
        candidate = torch.randperm(dimension, generator=generator)
        key = _permutation_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        actions.append(candidate)
    return torch.stack(actions)


def _compose_word_with_dictionary(
    words: Tensor,
    action_dictionary: Tensor,
) -> Tensor:
    output = torch.arange(
        action_dictionary.shape[1]
    ).expand(words.shape[0], -1).clone()
    for step in range(words.shape[1]):
        output = compose_left(
            output,
            action_dictionary[words[:, step]],
        )
    return output


def build_random_word_pools(
    config: ProgramInferenceConfig,
    action_dictionary: Tensor,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    generator = torch.Generator(device="cpu").manual_seed(
        config.word_pool_seed + 700_001
    )
    words: dict[str, Tensor] = {}
    products: dict[str, Tensor] = {}
    seen_train: set[tuple[int, ...]] = set()
    for length in TRAIN_LENGTHS + TEST_LENGTHS:
        train_counts = {
            1: config.dimension - 1,
            2: min(config.train_pool_size, 32),
            3: min(config.train_pool_size, 64),
            4: min(config.train_pool_size, 128),
        }
        target_count = (
            train_counts[length]
            if length in TRAIN_LENGTHS
            else config.test_pool_size
        )
        accepted_words: list[Tensor] = []
        accepted_products: list[Tensor] = []
        local_seen: set[tuple[int, ...]] = set()
        attempts = 0
        while len(accepted_words) < target_count and attempts < 200_000:
            attempts += 1
            word = torch.randint(
                action_dictionary.shape[0],
                (1, length),
                generator=generator,
            )
            product = _compose_word_with_dictionary(
                word,
                action_dictionary,
            )[0]
            key = _permutation_key(product)
            if key in local_seen:
                continue
            if length in TEST_LENGTHS and key in seen_train:
                continue
            local_seen.add(key)
            accepted_words.append(word[0])
            accepted_products.append(product)
        if len(accepted_words) != target_count:
            raise RuntimeError(
                f"could construct only {len(accepted_words)}/"
                f"{target_count} random-dictionary products at L{length}"
            )
        name = f"l{length}"
        words[name] = torch.stack(accepted_words)
        products[name] = torch.stack(accepted_products)
        if length in TRAIN_LENGTHS:
            seen_train.update(local_seen)
    return words, products


def braid_equivalent_words(words: Tensor) -> tuple[Tensor, Tensor]:
    originals: list[Tensor] = []
    alternatives: list[Tensor] = []
    for word in words:
        replacement: Tensor | None = None
        for offset in range(word.numel() - 2):
            left, middle, right = (
                int(word[offset].item()),
                int(word[offset + 1].item()),
                int(word[offset + 2].item()),
            )
            if left == right and abs(left - middle) == 1:
                replacement = word.clone()
                replacement[offset : offset + 3] = torch.tensor(
                    [middle, left, middle]
                )
                break
        if replacement is not None:
            originals.append(word)
            alternatives.append(replacement)
    if not originals:
        raise RuntimeError("word pool contains no braid-equivalent pair")
    return torch.stack(originals), torch.stack(alternatives)


def generate_root_evidence(
    permutations: Tensor,
    *,
    edges: Tensor,
    support_count: int,
    roots_per_support: int,
    noise: float,
    seed: int,
) -> RootEvidence:
    """Generate partial paired Q/K root observations on canonical probes."""

    batch_size = int(permutations.shape[0])
    root_count = int(edges.shape[0])
    if not 1 <= roots_per_support < root_count:
        raise ValueError("each support must expose a strict root subset")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    base = torch.arange(
        permutations.shape[1],
        dtype=torch.float32,
    ).expand(batch_size, support_count, -1)
    query_roots = root_signs(base, edges)
    key_coordinates = base.gather(
        -1,
        permutations[:, None, :].expand(-1, support_count, -1),
    )
    key_roots = root_signs(key_coordinates, edges)
    query_flip = torch.rand(
        batch_size,
        support_count,
        root_count,
        generator=generator,
    ) < noise
    key_flip = torch.rand(
        batch_size,
        support_count,
        root_count,
        generator=generator,
    ) < noise
    query_observed = query_roots * torch.where(
        query_flip,
        -1.0,
        1.0,
    )
    key_observed = key_roots * torch.where(
        key_flip,
        -1.0,
        1.0,
    )
    priorities = torch.rand(
        batch_size,
        support_count,
        root_count,
        generator=generator,
    )
    selected = priorities.topk(
        roots_per_support,
        dim=-1,
        largest=False,
        sorted=False,
    ).indices
    mask = torch.zeros(
        batch_size,
        support_count,
        root_count,
        dtype=torch.bool,
    )
    mask.scatter_(-1, selected, True)

    # Canonical Q roots are -1. Multiplying the observed Q/K signs and then
    # multiplying by that known sign estimates the K chamber root.
    estimated_key = -query_observed * key_observed
    votes = (estimated_key * mask).sum(dim=1)
    counts = mask.sum(dim=1)
    return RootEvidence(
        votes=votes,
        counts=counts,
        support_count=support_count,
        roots_per_support=roots_per_support,
    )


def _edge_index(dimension: int, edges: Tensor) -> Tensor:
    output = torch.full((dimension, dimension), -1, dtype=torch.long)
    root_ids = torch.arange(edges.shape[0])
    output[edges[:, 0], edges[:, 1]] = root_ids
    output[edges[:, 1], edges[:, 0]] = root_ids
    return output


def canonical_left_program(
    permutations: Tensor,
    *,
    action_labels_for_generators: Tensor,
    action_dictionary: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return a deterministic reduced left-action program plus its length."""

    dimension = int(permutations.shape[1])
    maximum_length = dimension * (dimension - 1) // 2
    sorting = permutations.clone()
    records = torch.full(
        (permutations.shape[0], maximum_length),
        -1,
        dtype=torch.long,
        device=permutations.device,
    )
    lengths = torch.zeros(
        permutations.shape[0],
        dtype=torch.long,
        device=permutations.device,
    )
    identity = torch.arange(
        dimension,
        device=permutations.device,
    )
    actual_actions = adjacent_action_dictionary(dimension).to(
        permutations.device
    )
    for step in range(maximum_length):
        active = (sorting != identity).any(dim=1)
        if not bool(active.any()):
            break
        positions = sorting.argsort(dim=1)
        descents = positions[:, :-1] > positions[:, 1:]
        selected = descents.float().argmax(dim=1)
        if not bool(descents[active].any(dim=1).all()):
            raise RuntimeError("non-identity permutation has no left descent")
        records[active, step] = selected[active]
        sorting[active] = compose_left(
            sorting[active],
            actual_actions[selected[active]],
        )
        lengths[active] += 1
    if bool((sorting != identity).any()):
        raise RuntimeError("canonical factorization exceeded maximum length")

    programs = torch.full_like(records, -1)
    for row in range(permutations.shape[0]):
        length = int(lengths[row].item())
        actual = records[row, :length].flip(dims=(0,))
        programs[row, :length] = action_labels_for_generators[actual]
    reconstructed = compose_label_programs(
        programs.clamp_min(0),
        lengths,
        action_dictionary=action_dictionary,
    )
    if not torch.equal(reconstructed, permutations):
        raise RuntimeError("canonical program does not reconstruct product")
    return programs, lengths


def _current_action_gains(
    current: Tensor,
    evidence: RootEvidence,
    *,
    edge_index: Tensor,
) -> Tensor:
    positions = current.argsort(dim=1)
    left_position = positions[:, :-1]
    right_position = positions[:, 1:]
    low = torch.minimum(left_position, right_position)
    high = torch.maximum(left_position, right_position)
    root_ids = edge_index[low, high]
    current_sign = torch.where(
        left_position > right_position,
        1.0,
        -1.0,
    )
    root_votes = evidence.votes.gather(1, root_ids)
    return -2.0 * current_sign * root_votes


def controller_features(
    current: Tensor,
    evidence: RootEvidence,
    *,
    edges: Tensor,
    step: int,
    maximum_steps: int,
) -> Tensor:
    vote_mean = evidence.votes / evidence.counts.clamp_min(1)
    confidence = evidence.counts.float() / evidence.support_count
    current_chamber = root_signs(current.float(), edges)
    progress = torch.full(
        (current.shape[0], 1),
        step / maximum_steps,
        dtype=vote_mean.dtype,
        device=vote_mean.device,
    )
    return torch.cat(
        (vote_mean, confidence, current_chamber, progress),
        dim=-1,
    )


class DenseProgramController(nn.Module):
    def __init__(
        self,
        input_dim: int,
        action_count: int,
        hidden: int,
        seed: int,
    ) -> None:
        super().__init__()
        torch.manual_seed(seed + 101)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, action_count + 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features)


class JointPairProgramController(nn.Module):
    """PC-LUT over joint support evidence and current program state."""

    def __init__(
        self,
        input_dim: int,
        action_count: int,
        *,
        tables: int,
        comparisons: int,
        seed: int,
    ) -> None:
        super().__init__()
        self.layer = PairwiseLUT(
            input_dim,
            action_count + 1,
            tables=tables,
            comparisons=comparisons,
            backend="torch",
            anchor_policy="random_no_replace",
            seed=seed + 211,
            lut_init_std=0.02,
            lut_dtype="fp32",
            fixed_zero_threshold=False,
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.layer(features).squeeze(1)


def _build_controller(
    kind: str,
    config: ProgramInferenceConfig,
) -> nn.Module:
    root_count = config.dimension * (config.dimension - 1) // 2
    input_dim = 3 * root_count + 1
    action_count = config.dimension - 1
    if kind == "dense":
        return DenseProgramController(
            input_dim,
            action_count,
            config.dense_hidden,
            config.seed,
        )
    if kind == "jointpair":
        return JointPairProgramController(
            input_dim,
            action_count,
            tables=config.jointpair_tables,
            comparisons=config.jointpair_comparisons,
            seed=config.seed,
        )
    raise ValueError(f"unsupported controller {kind!r}")


@torch.no_grad()
def decode_structured_coxeter(
    evidence: RootEvidence,
    *,
    action_dictionary: Tensor,
    edge_index: Tensor,
    maximum_steps: int,
) -> DecodedPrograms:
    """Emit locally supported simple reflections and then STOP."""

    device = evidence.votes.device
    batch_size = evidence.batch_size
    dimension = int(action_dictionary.shape[1])
    current = torch.arange(
        dimension,
        device=device,
    ).expand(batch_size, -1).clone()
    labels = torch.zeros(
        batch_size,
        maximum_steps,
        dtype=torch.long,
        device=device,
    )
    lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
    stopped = torch.zeros(batch_size, dtype=torch.bool, device=device)
    for step in range(maximum_steps):
        actual_gains = _current_action_gains(
            current,
            evidence,
            edge_index=edge_index,
        )
        # action_dictionary rows are adjacent permutations. Recover their
        # generator by the first displaced position.
        generator_for_label = (
            action_dictionary
            != torch.arange(dimension, device=device)
        ).float().argmax(dim=1)
        label_gains = actual_gains[:, generator_for_label]
        best_gain, selected_label = label_gains.max(dim=1)
        active = (~stopped) & (best_gain > 0)
        labels[active, step] = selected_label[active]
        if bool(active.any()):
            current[active] = compose_left(
                current[active],
                action_dictionary[selected_label[active]],
            )
            lengths[active] += 1
        stopped |= ~active
        if bool(stopped.all()):
            break
    return DecodedPrograms(
        labels=labels,
        lengths=lengths,
        stopped=stopped,
        permutations=current,
    )


@torch.no_grad()
def decode_generic_greedy(
    evidence: RootEvidence,
    *,
    action_dictionary: Tensor,
    edges: Tensor,
    maximum_steps: int,
) -> DecodedPrograms:
    """Greedy chamber-likelihood controller for arbitrary permutation actions."""

    device = evidence.votes.device
    batch_size = evidence.batch_size
    dimension = int(action_dictionary.shape[1])
    action_count = int(action_dictionary.shape[0])
    current = torch.arange(
        dimension,
        device=device,
    ).expand(batch_size, -1).clone()
    labels = torch.zeros(
        batch_size,
        maximum_steps,
        dtype=torch.long,
        device=device,
    )
    lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
    stopped = torch.zeros(batch_size, dtype=torch.bool, device=device)
    for step in range(maximum_steps):
        current_score = (
            root_signs(current.float(), edges) * evidence.votes
        ).sum(dim=-1)
        candidates = action_dictionary[None, :, :].expand(
            batch_size,
            -1,
            -1,
        ).gather(
            2,
            current[:, None, :].expand(-1, action_count, -1),
        )
        candidate_score = (
            root_signs(candidates.float(), edges)
            * evidence.votes[:, None, :]
        ).sum(dim=-1)
        gain, selected = (
            candidate_score - current_score[:, None]
        ).max(dim=1)
        active = (~stopped) & (gain > 0)
        labels[active, step] = selected[active]
        if bool(active.any()):
            current[active] = candidates[
                active,
                selected[active],
            ]
            lengths[active] += 1
        stopped |= ~active
        if bool(stopped.all()):
            break
    return DecodedPrograms(labels, lengths, stopped, current)


@torch.no_grad()
def decode_learned_controller(
    model: nn.Module,
    evidence: RootEvidence,
    *,
    action_dictionary: Tensor,
    edges: Tensor,
    maximum_steps: int,
) -> DecodedPrograms:
    model.eval()
    device = evidence.votes.device
    batch_size = evidence.batch_size
    dimension = int(action_dictionary.shape[1])
    stop_label = int(action_dictionary.shape[0])
    current = torch.arange(
        dimension,
        device=device,
    ).expand(batch_size, -1).clone()
    labels = torch.zeros(
        batch_size,
        maximum_steps,
        dtype=torch.long,
        device=device,
    )
    lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
    stopped = torch.zeros(batch_size, dtype=torch.bool, device=device)
    for step in range(maximum_steps):
        features = controller_features(
            current,
            evidence,
            edges=edges,
            step=step,
            maximum_steps=maximum_steps,
        )
        selected = model(features).argmax(dim=-1)
        active = (~stopped) & (selected != stop_label)
        labels[active, step] = selected[active]
        if bool(active.any()):
            current[active] = compose_left(
                current[active],
                action_dictionary[selected[active]],
            )
            lengths[active] += 1
        stopped |= selected == stop_label
        if bool(stopped.all()):
            break
    return DecodedPrograms(labels, lengths, stopped, current)


def _sample_products(
    products: dict[str, Tensor],
    *,
    lengths: tuple[int, ...],
    batch_size: int,
    seed: int,
) -> tuple[Tensor, Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    selected_lengths = torch.tensor(lengths)[
        torch.randint(len(lengths), (batch_size,), generator=generator)
    ]
    output = torch.empty(
        batch_size,
        next(iter(products.values())).shape[1],
        dtype=torch.long,
    )
    for length in lengths:
        rows = torch.nonzero(
            selected_lengths == length,
            as_tuple=False,
        ).flatten()
        if rows.numel() == 0:
            continue
        pool = products[f"l{length}"]
        indices = torch.randint(
            pool.shape[0],
            (rows.numel(),),
            generator=generator,
        )
        output[rows] = pool[indices]
    return output, selected_lengths


def _teacher_batch(
    permutations: Tensor,
    evidence: RootEvidence,
    *,
    action_dictionary: Tensor,
    inverse_action_labels: Tensor,
    edges: Tensor,
    maximum_steps: int,
) -> tuple[Tensor, Tensor]:
    programs, lengths = canonical_left_program(
        permutations,
        action_labels_for_generators=inverse_action_labels,
        action_dictionary=action_dictionary,
    )
    current = torch.arange(
        permutations.shape[1],
        device=permutations.device,
    ).expand(permutations.shape[0], -1).clone()
    feature_rows: list[Tensor] = []
    targets: list[Tensor] = []
    stop_label = int(action_dictionary.shape[0])
    maximum_teacher_steps = int(lengths.max().item()) + 1
    for step in range(maximum_teacher_steps):
        active = step <= lengths
        feature_rows.append(
            controller_features(
                current[active],
                RootEvidence(
                    evidence.votes[active],
                    evidence.counts[active],
                    evidence.support_count,
                    evidence.roots_per_support,
                ),
                edges=edges,
                step=step,
                maximum_steps=maximum_steps,
            )
        )
        target = torch.full(
            (int(active.sum().item()),),
            stop_label,
            dtype=torch.long,
            device=permutations.device,
        )
        continuing = step < lengths[active]
        if bool(continuing.any()):
            active_rows = torch.nonzero(
                active,
                as_tuple=False,
            ).flatten()
            target[continuing] = programs[
                active_rows[continuing],
                step,
            ]
        targets.append(target)

        update = step < lengths
        if bool(update.any()):
            current[update] = compose_left(
                current[update],
                action_dictionary[programs[update, step]],
            )
    return torch.cat(feature_rows), torch.cat(targets)


def _program_metrics(
    decoded: DecodedPrograms,
    target: Tensor,
    *,
    canonical_labels: Tensor | None = None,
    canonical_lengths: Tensor | None = None,
) -> dict[str, float]:
    product_correct = (decoded.permutations == target).all(dim=1)
    metrics = {
        "product_accuracy": float(product_correct.float().mean().item()),
        "stop_rate": float(decoded.stopped.float().mean().item()),
        "mean_emitted_length": float(decoded.lengths.float().mean().item()),
        "mean_target_coxeter_length": float(
            coxeter_length(target).float().mean().item()
        ),
    }
    if canonical_labels is not None and canonical_lengths is not None:
        length_match = decoded.lengths == canonical_lengths
        token_match = torch.ones_like(length_match)
        for row in range(decoded.labels.shape[0]):
            length = int(canonical_lengths[row].item())
            token_match[row] = torch.equal(
                decoded.labels[row, :length],
                canonical_labels[row, :length],
            )
        metrics["canonical_program_accuracy"] = float(
            (length_match & token_match).float().mean().item()
        )
    return metrics


@torch.no_grad()
def retrieval_accuracy(
    target: Tensor,
    predicted: Tensor,
    reverse_target: Tensor,
    *,
    edges: Tensor,
    edge_index: Tensor,
    candidate_count: int,
    seed: int,
) -> float:
    batch_size, dimension = target.shape
    generator = torch.Generator(device="cpu").manual_seed(seed)
    query = torch.stack(
        [
            torch.randperm(dimension, generator=generator)
            for _ in range(batch_size)
        ]
    ).float()
    random_keys = torch.stack(
        [
            torch.randperm(dimension, generator=generator)
            for _ in range(batch_size * (candidate_count - 2))
        ]
    ).reshape(batch_size, candidate_count - 2, dimension).float()
    decoy_prepermutation = torch.empty_like(target)
    decoy_prepermutation.scatter_(1, target, reverse_target)
    positive = query
    decoy = query.gather(1, decoy_prepermutation)
    base_keys = torch.cat(
        (positive[:, None, :], decoy[:, None, :], random_keys),
        dim=1,
    )
    key_coordinates = base_keys.gather(
        -1,
        target[:, None, :].expand(-1, candidate_count, -1),
    )
    query_roots = root_signs(query, edges)
    key_roots = root_signs(key_coordinates, edges)
    signed_indices = dynamic_signed_root_transport(
        predicted,
        edges=edges,
        query_root_subset=torch.arange(edges.shape[0]),
        edge_index=edge_index,
    )
    signed_key = torch.cat((-key_roots, key_roots), dim=-1)
    aligned = signed_key.gather(
        -1,
        signed_indices[:, None, :].expand(-1, candidate_count, -1),
    )
    scores = (query_roots[:, None, :] * aligned).sum(dim=-1)
    return float((scores.argmax(dim=1) == 0).float().mean().item())


def _evaluate_one(
    *,
    decoder: Literal["structured", "generic", "dense", "jointpair"],
    model: nn.Module | None,
    permutations: Tensor,
    config: ProgramInferenceConfig,
    action_dictionary: Tensor,
    inverse_action_labels: Tensor | None,
    support_count: int,
    seed: int,
    device: torch.device,
    edges: Tensor,
    edge_index: Tensor,
) -> tuple[dict[str, float], DecodedPrograms, RootEvidence]:
    evidence = generate_root_evidence(
        permutations,
        edges=edges.cpu(),
        support_count=support_count,
        roots_per_support=config.roots_per_support,
        noise=config.root_noise,
        seed=seed,
    ).to(device)
    target = permutations.to(device)
    action_dictionary = action_dictionary.to(device)
    if decoder == "structured":
        decoded = decode_structured_coxeter(
            evidence,
            action_dictionary=action_dictionary,
            edge_index=edge_index.to(device),
            maximum_steps=config.maximum_decode_steps,
        )
    elif decoder == "generic":
        decoded = decode_generic_greedy(
            evidence,
            action_dictionary=action_dictionary,
            edges=edges.to(device),
            maximum_steps=config.maximum_decode_steps,
        )
    elif decoder in {"dense", "jointpair"}:
        if model is None:
            raise ValueError("learned decoder requires a model")
        decoded = decode_learned_controller(
            model,
            evidence,
            action_dictionary=action_dictionary,
            edges=edges.to(device),
            maximum_steps=config.maximum_decode_steps,
        )
    else:
        raise ValueError(f"unsupported decoder {decoder!r}")
    canonical = None
    canonical_lengths = None
    if inverse_action_labels is not None:
        canonical, canonical_lengths = canonical_left_program(
            target,
            action_labels_for_generators=inverse_action_labels.to(device),
            action_dictionary=action_dictionary,
        )
    metrics = _program_metrics(
        decoded,
        target,
        canonical_labels=canonical,
        canonical_lengths=canonical_lengths,
    )
    if canonical is not None and canonical_lengths is not None:
        reverse_labels = canonical.clone()
        for row in range(reverse_labels.shape[0]):
            length = int(canonical_lengths[row].item())
            reverse_labels[row, :length] = canonical[
                row,
                :length,
            ].flip(dims=(0,))
        reverse_target = compose_label_programs(
            reverse_labels,
            canonical_lengths,
            action_dictionary=action_dictionary,
        )
    else:
        reverse_target = target
    metrics["retrieval_r1"] = retrieval_accuracy(
        target.cpu(),
        decoded.permutations.cpu(),
        reverse_target.cpu(),
        edges=edges.cpu(),
        edge_index=edge_index.cpu(),
        candidate_count=config.retrieval_candidates,
        seed=seed + 500_009,
    )
    return metrics, decoded, evidence


def evaluate_controller_matrix(
    *,
    kind: str,
    model: nn.Module | None,
    products: dict[str, Tensor],
    config: ProgramInferenceConfig,
    action_dictionary: Tensor,
    inverse_action_labels: Tensor | None,
    decoder: Literal["structured", "generic", "dense", "jointpair"],
    device: torch.device,
    seed_offset: int,
) -> dict[str, dict[str, float]]:
    edges = full_root_edges(config.dimension)
    edge_index = _edge_index(config.dimension, edges)
    output: dict[str, dict[str, float]] = {}
    for split_index, length in enumerate(TRAIN_LENGTHS + TEST_LENGTHS):
        pool = products[f"l{length}"]
        repeats = math.ceil(config.evaluation_episodes / pool.shape[0])
        target = pool.repeat(repeats, 1)[: config.evaluation_episodes]
        metrics, _, _ = _evaluate_one(
            decoder=decoder,
            model=model,
            permutations=target,
            config=config,
            action_dictionary=action_dictionary,
            inverse_action_labels=inverse_action_labels,
            support_count=config.support_count,
            seed=(
                config.data_seed
                + config.seed * 100_000_007
                + seed_offset
                + split_index * 1_000_003
            ),
            device=device,
            edges=edges,
            edge_index=edge_index,
        )
        output[f"l{length}"] = metrics
    return output


def train_controller(
    kind: str,
    *,
    products: dict[str, Tensor],
    config: ProgramInferenceConfig,
    action_dictionary: Tensor,
    inverse_action_labels: Tensor,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    edges = full_root_edges(config.dimension).to(device)
    model = _build_controller(kind, config).to(device)
    learning_rate = (
        config.learning_rate
        if kind == "dense"
        else config.jointpair_learning_rate
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=config.weight_decay,
    )
    best_validation = -1.0
    best_step = 0
    best_state = _clone_state_dict(model)
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for step in range(1, config.steps + 1):
        target, _ = _sample_products(
            products,
            lengths=TRAIN_LENGTHS,
            batch_size=config.batch_size,
            seed=(
                config.data_seed
                + config.seed * 100_000_007
                + step * 1_000_003
            ),
        )
        evidence = generate_root_evidence(
            target,
            edges=edges.cpu(),
            support_count=config.support_count,
            roots_per_support=config.roots_per_support,
            noise=config.root_noise,
            seed=(
                config.data_seed
                + config.seed * 100_000_007
                + step * 1_000_033
            ),
        ).to(device)
        features, labels = _teacher_batch(
            target.to(device),
            evidence,
            action_dictionary=action_dictionary.to(device),
            inverse_action_labels=inverse_action_labels.to(device),
            edges=edges,
            maximum_steps=config.maximum_decode_steps,
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = F.cross_entropy(logits, labels)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite {kind} loss at step {step}")
        loss.backward()
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.gradient_clip,
            ).item()
        )
        optimizer.step()

        if (
            step == 1
            or step % config.validation_interval == 0
            or step == config.steps
        ):
            validation_target, _ = _sample_products(
                products,
                lengths=TRAIN_LENGTHS,
                batch_size=config.validation_episodes,
                seed=(
                    config.data_seed
                    + config.seed * 100_000_007
                    + 80_000_003
                ),
            )
            validation, _, _ = _evaluate_one(
                decoder=kind,
                model=model,
                permutations=validation_target,
                config=config,
                action_dictionary=action_dictionary,
                inverse_action_labels=inverse_action_labels,
                support_count=config.support_count,
                seed=(
                    config.data_seed
                    + config.seed * 100_000_007
                    + 80_000_019
                ),
                device=device,
                edges=edges.cpu(),
                edge_index=_edge_index(
                    config.dimension,
                    edges.cpu(),
                ),
            )
            row = {
                "step": step,
                "train_ce": float(loss.item()),
                "gradient_norm": gradient_norm,
                "validation_product_accuracy": validation[
                    "product_accuracy"
                ],
                "validation_stop_rate": validation["stop_rate"],
            }
            history.append(row)
            print(
                json.dumps({"controller": kind, **row}),
                flush=True,
            )
            if validation["product_accuracy"] > best_validation:
                best_validation = validation["product_accuracy"]
                best_step = step
                best_state = _clone_state_dict(model)
    model.load_state_dict(best_state)
    return model, {
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "best_step": best_step,
        "best_validation_product_accuracy": best_validation,
        "seconds": time.perf_counter() - started,
        "history": history,
        "state_dict": best_state,
    }


def _structured_support_sweep(
    *,
    products: dict[str, Tensor],
    config: ProgramInferenceConfig,
    action_dictionary: Tensor,
    device: torch.device,
) -> dict[str, dict[str, dict[str, float]]]:
    edges = full_root_edges(config.dimension)
    edge_index = _edge_index(config.dimension, edges)
    output: dict[str, dict[str, dict[str, float]]] = {}
    for support_count in (8, 16, 32, config.support_count):
        output[str(support_count)] = {}
        for split_index, length in enumerate(TEST_LENGTHS):
            pool = products[f"l{length}"]
            repeats = math.ceil(config.evaluation_episodes / pool.shape[0])
            target = pool.repeat(repeats, 1)[
                : config.evaluation_episodes
            ]
            metrics, _, _ = _evaluate_one(
                decoder="structured",
                model=None,
                permutations=target,
                config=config,
                action_dictionary=action_dictionary,
                inverse_action_labels=None,
                support_count=support_count,
                seed=(
                    config.data_seed
                    + config.seed * 100_000_007
                    + 30_000_001
                    + support_count * 10_007
                    + split_index * 1_000_003
                ),
                device=device,
                edges=edges,
                edge_index=edge_index,
            )
            output[str(support_count)][f"l{length}"] = metrics
    return output


def _structured_interventions(
    *,
    products: dict[str, Tensor],
    words: dict[str, Tensor],
    config: ProgramInferenceConfig,
    action_dictionary: Tensor,
    inverse_action_labels: Tensor,
    device: torch.device,
) -> dict[str, Any]:
    edges = full_root_edges(config.dimension)
    edge_index = _edge_index(config.dimension, edges)
    target_pool = products["l16"]
    repeats = math.ceil(config.evaluation_episodes / target_pool.shape[0])
    target = target_pool.repeat(repeats, 1)[: config.evaluation_episodes]
    metrics, decoded, _ = _evaluate_one(
        decoder="structured",
        model=None,
        permutations=target,
        config=config,
        action_dictionary=action_dictionary,
        inverse_action_labels=inverse_action_labels,
        support_count=config.support_count,
        seed=(
            config.data_seed
            + config.seed * 100_000_007
            + 40_000_003
        ),
        device=device,
        edges=edges,
        edge_index=edge_index,
    )
    canonical, canonical_lengths = canonical_left_program(
        target.to(device),
        action_labels_for_generators=inverse_action_labels.to(device),
        action_dictionary=action_dictionary.to(device),
    )
    reverse_labels = canonical.clone()
    for row in range(reverse_labels.shape[0]):
        length = int(canonical_lengths[row].item())
        reverse_labels[row, :length] = canonical[
            row,
            :length,
        ].flip(dims=(0,))
    reverse_product = compose_label_programs(
        reverse_labels,
        canonical_lengths,
        action_dictionary=action_dictionary.to(device),
    )
    sorted_product = compose_label_programs(
        decoded.labels,
        decoded.lengths,
        action_dictionary=action_dictionary.to(device),
        law="sorted_left",
    )
    wrong_law = compose_label_programs(
        decoded.labels,
        decoded.lengths,
        action_dictionary=action_dictionary.to(device),
        law="right",
    )
    identity_actions = adjacent_action_dictionary(config.dimension)
    identity_decoded = decode_structured_coxeter(
        generate_root_evidence(
            target,
            edges=edges,
            support_count=config.support_count,
            roots_per_support=config.roots_per_support,
            noise=config.root_noise,
            seed=(
                config.data_seed
                + config.seed * 100_000_007
                + 40_000_003
            ),
        ).to(device),
        action_dictionary=identity_actions.to(device),
        edge_index=edge_index.to(device),
        maximum_steps=config.maximum_decode_steps,
    )
    braid_original, braid_alternative = braid_equivalent_words(
        words["l16"]
    )
    braid_original_product = _compose_right_words(
        braid_original,
        config.dimension,
    )
    braid_alternative_product = _compose_right_words(
        braid_alternative,
        config.dimension,
    )
    return {
        "base": metrics,
        "generator_relabel_product_retention": (
            float(
                (
                    decoded.permutations
                    == identity_decoded.permutations
                )
                .all(dim=1)
                .float()
                .mean()
                .item()
            )
        ),
        "reverse_order_product_accuracy": float(
            (reverse_product == target.to(device))
            .all(dim=1)
            .float()
            .mean()
            .item()
        ),
        "wrong_right_law_product_accuracy": float(
            (wrong_law == target.to(device))
            .all(dim=1)
            .float()
            .mean()
            .item()
        ),
        "sorted_bag_product_accuracy": float(
            (sorted_product == target.to(device))
            .all(dim=1)
            .float()
            .mean()
            .item()
        ),
        "braid_pair_count": int(braid_original.shape[0]),
        "braid_product_equivalence": float(
            (
                braid_original_product
                == braid_alternative_product
            )
            .all(dim=1)
            .float()
            .mean()
            .item()
        ),
    }


def _build_gates(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    structured = result["evaluation"]["structured_coxeter"]
    dense = result["evaluation"]["dense"]
    jointpair = result["evaluation"]["jointpair"]
    random_dictionary = result["evaluation"][
        "random_permutation_dictionary"
    ]
    interventions = result["interventions"]
    quantities = {
        "structured_l8_product": (
            structured["l8"]["product_accuracy"],
            0.90,
            "min",
        ),
        "structured_l16_product": (
            structured["l16"]["product_accuracy"],
            0.85,
            "min",
        ),
        "structured_l16_retrieval": (
            structured["l16"]["retrieval_r1"],
            0.85,
            "min",
        ),
        "generator_relabel_invariance": (
            interventions["generator_relabel_product_retention"],
            0.99,
            "min",
        ),
        "braid_product_equivalence": (
            interventions["braid_product_equivalence"],
            1.0,
            "min",
        ),
        "reverse_order_negative": (
            interventions["reverse_order_product_accuracy"],
            0.20,
            "max",
        ),
        "wrong_law_negative": (
            interventions["sorted_bag_product_accuracy"],
            0.20,
            "max",
        ),
        "random_dictionary_l16_below_coxeter": (
            random_dictionary["l16"]["product_accuracy"]
            / max(structured["l16"]["product_accuracy"], 1e-12),
            0.50,
            "max",
        ),
    }
    # These are diagnostics, not pass requirements. Their purpose is to
    # compare generic learned controllers against the structured program.
    result["controller_diagnostics"] = {
        "dense_l8_product": dense["l8"]["product_accuracy"],
        "dense_l16_product": dense["l16"]["product_accuracy"],
        "jointpair_l8_product": jointpair["l8"]["product_accuracy"],
        "jointpair_l16_product": jointpair["l16"]["product_accuracy"],
    }
    output: dict[str, dict[str, Any]] = {}
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


def _build_config(args: argparse.Namespace) -> ProgramInferenceConfig:
    return ProgramInferenceConfig(
        seed=args.seed,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        jointpair_learning_rate=args.jointpair_learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        validation_interval=args.validation_interval,
        validation_episodes=args.validation_episodes,
        evaluation_episodes=args.evaluation_episodes,
        dimension=args.dimension,
        support_count=args.support_count,
        roots_per_support=args.roots_per_support,
        root_noise=args.root_noise,
        maximum_decode_steps=args.maximum_decode_steps,
        dense_hidden=args.dense_hidden,
        jointpair_tables=args.jointpair_tables,
        jointpair_comparisons=args.jointpair_comparisons,
        retrieval_candidates=args.retrieval_candidates,
        train_pool_size=args.train_pool_size,
        test_pool_size=args.test_pool_size,
        word_pool_seed=args.word_pool_seed,
        generator_relabel_seed=args.generator_relabel_seed,
        random_dictionary_seed=args.random_dictionary_seed,
        data_seed=args.data_seed,
        device=args.device,
    )


def _validate_config(config: ProgramInferenceConfig) -> None:
    if config.dimension != 8:
        raise ValueError("the formal no-bank protocol currently requires D8")
    root_count = config.dimension * (config.dimension - 1) // 2
    if not 1 <= config.roots_per_support < root_count:
        raise ValueError("roots per support must be a strict root subset")
    if config.support_count < 8:
        raise ValueError("formal protocol requires at least eight supports")
    if not 0.0 <= config.root_noise < 0.5:
        raise ValueError("root noise must lie in [0, 0.5)")
    if config.maximum_decode_steps < max(TEST_LENGTHS):
        raise ValueError("decode horizon is shorter than held programs")
    if config.retrieval_candidates < 3:
        raise ValueError("retrieval requires positive, decoy, and negatives")


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
    words, products = build_coxeter_word_pools(config)
    identity_actions = adjacent_action_dictionary(config.dimension)
    generator_relabel = torch.randperm(
        config.dimension - 1,
        generator=torch.Generator(device="cpu").manual_seed(
            config.generator_relabel_seed + config.seed
        ),
    )
    action_dictionary = identity_actions[generator_relabel]
    inverse_action_labels = torch.empty_like(generator_relabel)
    inverse_action_labels[generator_relabel] = torch.arange(
        config.dimension - 1
    )
    random_actions = random_action_dictionary(
        config.dimension,
        config.random_dictionary_seed + config.seed,
    )
    random_words, random_products = build_random_word_pools(
        config,
        random_actions,
    )

    models: dict[str, nn.Module] = {}
    training: dict[str, Any] = {}
    checkpoints: dict[str, Any] = {}
    for kind in CONTROLLER_KINDS:
        model, training_result = train_controller(
            kind,
            products=products,
            config=config,
            action_dictionary=action_dictionary,
            inverse_action_labels=inverse_action_labels,
            device=device,
        )
        models[kind] = model
        checkpoints[kind] = training_result.pop("state_dict")
        training[kind] = training_result

    evaluation = {
        "structured_coxeter": evaluate_controller_matrix(
            kind="structured_coxeter",
            model=None,
            products=products,
            config=config,
            action_dictionary=action_dictionary,
            inverse_action_labels=inverse_action_labels,
            decoder="structured",
            device=device,
            seed_offset=1_000_001,
        ),
        "dense": evaluate_controller_matrix(
            kind="dense",
            model=models["dense"],
            products=products,
            config=config,
            action_dictionary=action_dictionary,
            inverse_action_labels=inverse_action_labels,
            decoder="dense",
            device=device,
            seed_offset=2_000_003,
        ),
        "jointpair": evaluate_controller_matrix(
            kind="jointpair",
            model=models["jointpair"],
            products=products,
            config=config,
            action_dictionary=action_dictionary,
            inverse_action_labels=inverse_action_labels,
            decoder="jointpair",
            device=device,
            seed_offset=3_000_007,
        ),
        "random_permutation_dictionary": evaluate_controller_matrix(
            kind="random_permutation_dictionary",
            model=None,
            products=random_products,
            config=config,
            action_dictionary=random_actions,
            inverse_action_labels=None,
            decoder="generic",
            device=device,
            seed_offset=4_000_009,
        ),
    }
    support_sweep = _structured_support_sweep(
        products=products,
        config=config,
        action_dictionary=action_dictionary,
        device=device,
    )
    interventions = _structured_interventions(
        products=products,
        words=words,
        config=config,
        action_dictionary=action_dictionary,
        inverse_action_labels=inverse_action_labels,
        device=device,
    )
    result: dict[str, Any] = {
        "complete": True,
        "config": asdict(config),
        "architecture": {
            "relation_bank": "none",
            "support_observation": (
                f"{config.support_count} paired canonical probes; each "
                f"exposes {config.roots_per_support}/"
                f"{config.dimension * (config.dimension - 1) // 2} roots"
            ),
            "root_noise": (
                "independent Q and K sign flips before paired aggregation"
            ),
            "program": (
                "autoregressive generator symbols plus STOP; one composed "
                "transport is reused for all retrieval candidates"
            ),
            "train_lengths": list(TRAIN_LENGTHS),
            "test_lengths": list(TEST_LENGTHS),
            "structured_rule": (
                "A-type adjacent value generator flips one root; emit a "
                "positive-evidence reflection until no positive move remains"
            ),
            "learned_controls": (
                "Dense MLP and JointPair PC-LUT receive identical root votes, "
                "root confidence, and current chamber"
            ),
        },
        "pool_diagnostics": {
            "coxeter": {
                name: {
                    "words": int(pool.shape[0]),
                    "literal_length": int(pool.shape[1]),
                    "unique_products": int(products[name].shape[0]),
                    "coxeter_length_min": int(
                        coxeter_length(products[name]).min().item()
                    ),
                    "coxeter_length_max": int(
                        coxeter_length(products[name]).max().item()
                    ),
                }
                for name, pool in words.items()
            },
            "random_dictionary": {
                name: {
                    "words": int(pool.shape[0]),
                    "literal_length": int(pool.shape[1]),
                    "unique_products": int(random_products[name].shape[0]),
                }
                for name, pool in random_words.items()
            },
            "train_test_product_overlap": 0,
        },
        "generator_relabel": generator_relabel.tolist(),
        "random_action_dictionary": random_actions.tolist(),
        "training": training,
        "evaluation": evaluation,
        "support_sweep": support_sweep,
        "interventions": interventions,
    }
    gates = _build_gates(result)
    result["gates"] = gates
    result["decision"] = (
        "advance_no_bank_program_inference"
        if all(bool(gate["passed"]) for gate in gates.values())
        else "stop_no_bank_program_inference"
    )
    atomic_json_write(result, result_path)
    atomic_torch_save(
        {
            "config": asdict(config),
            "generator_relabel": generator_relabel,
            "dense_state_dict": checkpoints["dense"],
            "jointpair_state_dict": checkpoints["jointpair"],
        },
        args.out_dir / "best.pt",
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "result": str(result_path),
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
        help="train one seed and evaluate no-bank program induction",
    )
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--out-dir", type=Path, required=True)
    run.add_argument("--steps", type=int, default=3000)
    run.add_argument("--batch-size", type=int, default=256)
    run.add_argument("--learning-rate", type=float, default=1e-3)
    run.add_argument("--jointpair-learning-rate", type=float, default=1e-2)
    run.add_argument("--weight-decay", type=float, default=0.0)
    run.add_argument("--gradient-clip", type=float, default=5.0)
    run.add_argument("--validation-interval", type=int, default=250)
    run.add_argument("--validation-episodes", type=int, default=512)
    run.add_argument("--evaluation-episodes", type=int, default=1024)
    run.add_argument("--dimension", type=int, default=8)
    run.add_argument("--support-count", type=int, default=64)
    run.add_argument("--roots-per-support", type=int, default=4)
    run.add_argument("--root-noise", type=float, default=0.05)
    run.add_argument("--maximum-decode-steps", type=int, default=28)
    run.add_argument("--dense-hidden", type=int, default=64)
    run.add_argument("--jointpair-tables", type=int, default=32)
    run.add_argument("--jointpair-comparisons", type=int, default=5)
    run.add_argument("--retrieval-candidates", type=int, default=32)
    run.add_argument("--train-pool-size", type=int, default=128)
    run.add_argument("--test-pool-size", type=int, default=256)
    run.add_argument("--word-pool-seed", type=int, default=202607271)
    run.add_argument(
        "--generator-relabel-seed",
        type=int,
        default=202607272,
    )
    run.add_argument(
        "--random-dictionary-seed",
        type=int,
        default=202607273,
    )
    run.add_argument("--data-seed", type=int, default=1729)
    run.add_argument("--device", default="cuda")
    run.set_defaults(function=run_experiment)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
