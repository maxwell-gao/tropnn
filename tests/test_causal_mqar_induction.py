from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from tropnn.tools.causal_mqar_induction import (
    CHAMBER_DECODERS,
    DECODERS,
    CausalMQARRetriever,
    RunConfig,
    _validate_config,
    generate_mqar_batch,
    target_token_loss,
    token_permutation,
)


def config(decoder: str, *, chamber_relabel_seed: int = -1) -> RunConfig:
    return RunConfig(
        decoder=decoder,
        seed=0,
        steps=2,
        batch_size=8,
        learning_rate=0.01,
        weight_decay=0.0,
        gradient_clip=5.0,
        validation_interval=1,
        validation_episodes=8,
        evaluation_episodes=8,
        evaluation_batch_size=4,
        vocab_size=128,
        relation_dim=8,
        dense_rank=4,
        relation_tables=4,
        relation_coverage=2,
        coxeter_rank=4,
        jointpair_tables=8,
        jointpair_comparisons=4,
        data_seed=1729,
        codebook_seed=2718,
        token_relabel_seed=314159,
        chamber_relabel_seed=chamber_relabel_seed,
        device="cpu",
    )


def test_episode_is_fresh_injective_and_queries_exact_prefix_mappings() -> None:
    batch = generate_mqar_batch(
        batch_size=7,
        pair_count=8,
        query_count=4,
        vocab_size=128,
        seed=11,
    )
    repeated = generate_mqar_batch(
        batch_size=7,
        pair_count=8,
        query_count=4,
        vocab_size=128,
        seed=11,
    )
    assert torch.equal(batch.keys, repeated.keys)
    assert torch.equal(batch.values, repeated.values)
    assert torch.equal(batch.queries, batch.keys.gather(1, batch.query_indices))
    assert torch.equal(batch.targets, batch.values.gather(1, batch.query_indices))
    episode_tokens = torch.cat((batch.keys, batch.values), dim=1)
    assert torch.all(episode_tokens.sort(dim=1).values[:, 1:] != episode_tokens.sort(dim=1).values[:, :-1])
    assert not torch.equal(batch.keys[0], batch.keys[1])


def test_prefix_memory_rows_align_predecessor_key_and_current_value() -> None:
    batch = generate_mqar_batch(
        batch_size=3,
        pair_count=6,
        query_count=2,
        vocab_size=128,
        seed=13,
    )
    prefix = batch.prefix_tokens()
    memory_positions = batch.memory_row_positions()
    assert torch.equal(prefix[:, memory_positions], batch.values)
    assert torch.equal(prefix[:, memory_positions - 1], batch.keys)

    local = CausalMQARRetriever(config("local_dense_qk"))
    rows, key_coordinates, value_tokens = local.memory_coordinates(batch)
    assert torch.equal(rows[..., :8], local.codebook(batch.keys))
    assert torch.equal(rows[..., 8:], local.codebook(batch.values))
    assert torch.equal(key_coordinates, local.codebook(batch.keys))
    assert torch.equal(value_tokens, batch.values)

    no_local = CausalMQARRetriever(config("no_local_dense_qk"))
    _, no_local_key, no_local_values = no_local.memory_coordinates(batch)
    assert torch.equal(no_local_key, no_local.codebook(batch.values))
    assert torch.equal(no_local_values, batch.values)


def test_token_relabel_is_one_consistent_bijection_of_the_same_episode() -> None:
    permutation = token_permutation(128, 17)
    base = generate_mqar_batch(
        batch_size=5,
        pair_count=8,
        query_count=4,
        vocab_size=128,
        seed=19,
    )
    relabeled = generate_mqar_batch(
        batch_size=5,
        pair_count=8,
        query_count=4,
        vocab_size=128,
        seed=19,
        token_permutation=permutation,
    )
    assert torch.equal(relabeled.keys, permutation[base.keys])
    assert torch.equal(relabeled.values, permutation[base.values])
    assert torch.equal(relabeled.queries, permutation[base.queries])
    assert torch.equal(relabeled.targets, permutation[base.targets])
    assert torch.equal(relabeled.query_indices, base.query_indices)


@pytest.mark.parametrize("decoder", DECODERS)
def test_all_relation_scorers_backpropagate_from_target_token_ce(decoder: str) -> None:
    model = CausalMQARRetriever(config(decoder))
    batch = generate_mqar_batch(
        batch_size=4,
        pair_count=5,
        query_count=3,
        vocab_size=128,
        seed=23,
    )
    scores, values = model(batch)
    assert scores.shape == (4, 3, 5)
    assert torch.equal(values, batch.values)
    loss = target_token_loss(scores, batch.query_indices)
    loss.backward()
    gradients = [parameter.grad for parameter in model.relation.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(float(gradient.abs().sum().item()) for gradient in gradients) > 0.0


def test_exact_value_transport_matches_candidate_cross_entropy() -> None:
    model = CausalMQARRetriever(config("local_dense_qk"))
    batch = generate_mqar_batch(
        batch_size=4,
        pair_count=6,
        query_count=3,
        vocab_size=128,
        seed=29,
    )
    scores, _ = model(batch)
    candidate_ce = target_token_loss(scores, batch.query_indices)
    probabilities = model.token_probabilities(batch)
    transported = probabilities.gather(2, batch.targets.unsqueeze(-1)).squeeze(-1)
    token_ce = -transported.clamp_min(1e-30).log().mean()
    torch.testing.assert_close(token_ce, candidate_ce)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones_like(probabilities.sum(dim=-1)))


@pytest.mark.parametrize("decoder", CHAMBER_DECODERS)
def test_chamber_relabel_control_is_a_per_table_bijection(decoder: str) -> None:
    model = CausalMQARRetriever(config(decoder, chamber_relabel_seed=31))
    relabel = model.relation.relabel
    assert relabel.shape == (4, 24)
    expected = torch.arange(24).expand(4, -1)
    assert torch.equal(relabel.sort(dim=1).values, expected)


def test_root_incidence_is_exactly_invariant_to_s4_route_relabeling() -> None:
    base = CausalMQARRetriever(config("local_root_incidence"))
    relabeled = CausalMQARRetriever(config("local_root_incidence", chamber_relabel_seed=37))
    relabeled.load_state_dict(base.state_dict(), strict=False)
    batch = generate_mqar_batch(
        batch_size=4,
        pair_count=6,
        query_count=3,
        vocab_size=128,
        seed=41,
    )
    torch.testing.assert_close(base(batch)[0], relabeled(batch)[0])


def test_fixed_codebook_contains_only_permutations_of_one_rank_vector() -> None:
    model = CausalMQARRetriever(config("local_kendall"))
    sorted_rows = model.codebook.weight.sort(dim=1).values
    torch.testing.assert_close(sorted_rows, sorted_rows[:1].expand_as(sorted_rows))
    assert model.codebook.weight.requires_grad is False
    assert model.relation_parameters == 2


def test_chamber_relabel_is_rejected_for_non_chamber_decoder() -> None:
    invalid = replace(config("local_dense_qk"), chamber_relabel_seed=43)
    with pytest.raises(ValueError, match="defined only for S4 chamber decoders"):
        _validate_config(invalid)
