from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from tropnn.tools.causal_mqar_induction import DECODERS, target_token_loss
from tropnn.tools.causal_mqar_role_gauge import (
    RoleGaugeMQARRetriever,
    RoleGaugeRunConfig,
    _condition_permutations,
    _validate_config,
    gauge_diagnostics,
    generate_pool_mqar_batch,
    token_pools,
)


def config(
    decoder: str,
    *,
    gauge_mode: str = "role_permutation",
    chamber_relabel_seed: int = -1,
) -> RoleGaugeRunConfig:
    return RoleGaugeRunConfig(
        decoder=decoder,
        gauge_mode=gauge_mode,
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
        vocab_size=192,
        train_tokens=64,
        validation_tokens=64,
        test_tokens=64,
        relation_dim=8,
        dense_rank=4,
        relation_tables=4,
        relation_coverage=2,
        coxeter_rank=4,
        jointpair_tables=8,
        jointpair_comparisons=4,
        data_seed=1729,
        codebook_seed=2718,
        query_gauge_seed=101,
        key_gauge_seed=103,
        wrong_key_gauge_seed=107,
        token_relabel_seed=314159,
        chamber_relabel_seed=chamber_relabel_seed,
        device="cpu",
    )


def test_token_pools_are_strictly_disjoint_and_cover_vocabulary() -> None:
    pools = token_pools(config("local_dense_qk"))
    assert torch.equal(pools["train"], torch.arange(0, 64))
    assert torch.equal(pools["validation"], torch.arange(64, 128))
    assert torch.equal(pools["test"], torch.arange(128, 192))
    assert torch.equal(torch.cat(tuple(pools.values())), torch.arange(192))


def test_pool_episode_never_leaks_an_identity_outside_its_split() -> None:
    pool = token_pools(config("local_dense_qk"))["test"]
    batch = generate_pool_mqar_batch(
        token_pool=pool,
        batch_size=7,
        pair_count=16,
        query_count=8,
        seed=11,
    )
    all_tokens = torch.cat((batch.keys, batch.values, batch.queries, batch.targets), dim=1)
    assert int(all_tokens.min().item()) >= 128
    assert int(all_tokens.max().item()) < 192
    assert torch.equal(batch.queries, batch.keys.gather(1, batch.query_indices))
    assert torch.equal(batch.targets, batch.values.gather(1, batch.query_indices))


def test_role_gauges_are_distinct_and_relative_map_exactly_reconstructs_key() -> None:
    model = RoleGaugeMQARRetriever(config("local_dense_qk"))
    assert not torch.equal(model.query_permutation, model.key_permutation)
    token_ids = token_pools(model.config)["test"]
    coordinates = model.codebook(token_ids)
    query = coordinates[..., model.query_permutation]
    key = coordinates[..., model.key_permutation]
    torch.testing.assert_close(query[..., model.relative_query_to_key], key)
    assert not bool((query == key).all(dim=-1).any())


def test_identity_gauge_preserves_exact_chamber_identity() -> None:
    model = RoleGaugeMQARRetriever(config("local_kendall", gauge_mode="identity"))
    expected = torch.arange(model.config.relation_dim)
    assert torch.equal(model.query_permutation, expected)
    assert torch.equal(model.key_permutation, expected)
    diagnostics = gauge_diagnostics(
        model,
        token_pool=token_pools(model.config)["test"],
        device=torch.device("cpu"),
    )
    assert diagnostics["positive_coordinate_vector_equality"] == 1.0
    assert diagnostics["positive_s4_table_route_agreement"] == 1.0


def test_causal_memory_row_applies_key_gauge_only_after_local_lane_selection() -> None:
    model = RoleGaugeMQARRetriever(config("local_dense_qk"))
    batch = generate_pool_mqar_batch(
        token_pool=token_pools(model.config)["train"],
        batch_size=3,
        pair_count=6,
        query_count=2,
        seed=13,
    )
    rows, key, values = model.memory_coordinates(batch)
    predecessor = model.codebook(batch.keys)
    current = model.codebook(batch.values)
    assert torch.equal(rows[..., :8], predecessor)
    assert torch.equal(rows[..., 8:], current)
    assert torch.equal(key, predecessor[..., model.key_permutation])
    assert torch.equal(values, batch.values)

    no_local = RoleGaugeMQARRetriever(config("no_local_dense_qk"))
    _, no_local_key, no_local_values = no_local.memory_coordinates(batch)
    assert torch.equal(no_local_key, no_local.codebook(batch.values)[..., no_local.key_permutation])
    assert torch.equal(no_local_values, batch.values)


def test_wrong_key_and_role_swap_are_post_training_role_interventions() -> None:
    model = RoleGaugeMQARRetriever(config("local_dense_qk"))
    base_query, base_key = _condition_permutations(model, "base")
    wrong_query, wrong_key = _condition_permutations(model, "wrong_key_gauge")
    swap_query, swap_key = _condition_permutations(model, "role_swap")
    assert torch.equal(base_query, wrong_query)
    assert not torch.equal(base_key, wrong_key)
    assert torch.equal(swap_query, base_key)
    assert torch.equal(swap_key, base_query)


@pytest.mark.parametrize("decoder", DECODERS)
def test_all_role_gauge_relation_scorers_backpropagate_from_retrieval_ce(decoder: str) -> None:
    model = RoleGaugeMQARRetriever(config(decoder))
    batch = generate_pool_mqar_batch(
        token_pool=token_pools(model.config)["train"],
        batch_size=4,
        pair_count=5,
        query_count=3,
        seed=17,
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


def test_role_diagnostics_measure_absence_of_exact_identity_shortcut() -> None:
    model = RoleGaugeMQARRetriever(config("local_global_coxeter"))
    diagnostics = gauge_diagnostics(
        model,
        token_pool=token_pools(model.config)["test"],
        device=torch.device("cpu"),
    )
    assert diagnostics["relative_map_reconstruction_max_error"] == 0.0
    assert diagnostics["positive_coordinate_vector_equality"] == 0.0
    assert diagnostics["positive_full_route_vector_equality"] < 1.0


def test_invalid_split_and_identity_matrix_are_rejected() -> None:
    invalid_split = replace(config("local_dense_qk"), train_tokens=63, vocab_size=191)
    with pytest.raises(ValueError, match="at least 64"):
        _validate_config(invalid_split)
    invalid_identity = replace(config("local_global_coxeter"), gauge_mode="identity")
    with pytest.raises(ValueError, match="restricted"):
        _validate_config(invalid_identity)
    invalid_relabel = config("local_dense_qk", chamber_relabel_seed=31)
    with pytest.raises(ValueError, match="defined only"):
        _validate_config(invalid_relabel)
