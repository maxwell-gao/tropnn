from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from tropnn.tools.causal_mqar_composed_multihop import (
    CanonComposer,
    ComposedRunConfig,
)
from tropnn.tools.causal_mqar_role_gauge import token_pools
from tropnn.tools.causal_mqar_root_transport import (
    full_root_edges,
    root_signs,
)
from tropnn.tools.causal_mqar_variable_transport import (
    VariableCoxeterTransport,
    VariableTransportConfig,
    actual_word_permutations,
    build_word_pools,
    compose_generator_words,
    dynamic_signed_root_transport,
    evaluate_path,
    generate_path_batch,
    symbol_dictionary,
    word_pool_diagnostics,
)


def _canon_config(dimension: int) -> ComposedRunConfig:
    return ComposedRunConfig(
        task="twohop",
        composer="canon",
        seed=0,
        relation_checkpoint="unused-in-test",
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
        relation_dim=dimension,
        composer_tables=4,
        composer_comparisons=3,
        composer_lut_init_std=0.02,
        root_ste_temperature=1.0,
        data_seed=1729,
        codebook_seed=2718,
        token_relabel_seed=314159,
        device="cpu",
    )


def _canon_checkpoint(path: Path, dimension: int) -> Path:
    composer = CanonComposer(dimension)
    with torch.no_grad():
        composer.lag_logits[:, 0] = 12.0
        composer.lag_logits[:, 1] = -12.0
    torch.save(
        {
            "config": _canon_config(dimension).__dict__,
            "state_dict": {
                "composer.lag_logits": composer.lag_logits.detach(),
            },
        },
        path,
    )
    return path


def _config(checkpoint: Path, dimension: int = 10) -> VariableTransportConfig:
    return VariableTransportConfig(
        seed=0,
        canon_checkpoint=str(checkpoint),
        steps=2,
        batch_size=8,
        learning_rate=0.1,
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
        relation_dim=dimension,
        root_budget=dimension * (dimension - 1) // 2,
        root_subset_seed=4242,
        evaluation_pair_count=16,
        word_pool_size=16,
        assignment_temperature_start=2.0,
        assignment_temperature_end=0.05,
        assignment_entropy_weight=1e-3,
        assignment_coverage_weight=1e-2,
        positive_alignment_weight=1.0,
        data_seed=1729,
        codebook_seed=2718,
        generator_dictionary_seed=101,
        word_pool_seed=103,
        token_relabel_seed=314159,
        device="cpu",
    )


def test_generator_composition_is_order_sensitive() -> None:
    forward = compose_generator_words(torch.tensor([[0, 1]]), 4)
    reverse = compose_generator_words(torch.tensor([[1, 0]]), 4)
    assert torch.equal(forward, torch.tensor([[1, 2, 0, 3]]))
    assert torch.equal(reverse, torch.tensor([[2, 0, 1, 3]]))
    assert not torch.equal(forward, reverse)


def test_word_pools_are_reduced_disjoint_products() -> None:
    dimension = 32
    dictionary = symbol_dictionary(dimension, 107)
    pools = build_word_pools(
        dimension=dimension,
        dictionary=dictionary,
        random_pool_size=16,
        seed=109,
    )
    diagnostics = word_pool_diagnostics(
        pools,
        dictionary=dictionary,
        dimension=dimension,
    )
    assert set(pools) == {
        "seen_l1",
        "unseen_l2",
        "unseen_l4",
        "unseen_l8",
        "unseen_l16",
        "unseen_l31",
    }
    for name, words in pools.items():
        diagnostic = diagnostics[name]
        assert diagnostic["minimum_coxeter_length"] == words.shape[1]
        assert diagnostic["maximum_coxeter_length"] == words.shape[1]
        assert diagnostic["unique_products"] == words.shape[0]
        if name != "seen_l1":
            assert diagnostic["seen_singleton_product_overlap"] == 0
            assert diagnostic["minimum_reverse_permutation_distance"] > 0


def test_path_generator_preserves_every_shuffled_hop() -> None:
    dictionary = symbol_dictionary(10, 113)
    batch = generate_path_batch(
        token_pool=torch.arange(64),
        word_pool=torch.arange(9).unsqueeze(1),
        dictionary=dictionary,
        dimension=10,
        batch_size=8,
        pair_count=16,
        hops=4,
        query_count=4,
        seed=127,
    )
    flat_indices = batch.hop_indices.flatten(1)
    path_keys = batch.memory_keys.gather(1, flat_indices).reshape(8, 4, 4)
    path_values = batch.memory_values.gather(1, flat_indices).reshape(8, 4, 4)
    assert torch.equal(path_keys[:, :, 0], batch.queries)
    assert torch.equal(path_keys[:, :, 1:], batch.hop_targets[:, :, :-1])
    assert torch.equal(path_values, batch.hop_targets)


def test_path_token_relabel_preserves_words_and_indices() -> None:
    dictionary = symbol_dictionary(10, 131)
    word_pool = torch.arange(9).unsqueeze(1)
    permutation = torch.randperm(64, generator=torch.Generator().manual_seed(137))
    base = generate_path_batch(
        token_pool=torch.arange(128, 192),
        word_pool=word_pool,
        dictionary=dictionary,
        dimension=10,
        batch_size=4,
        pair_count=16,
        hops=4,
        query_count=4,
        seed=139,
    )
    relabelled = generate_path_batch(
        token_pool=torch.arange(128, 192),
        word_pool=word_pool,
        dictionary=dictionary,
        dimension=10,
        batch_size=4,
        pair_count=16,
        hops=4,
        query_count=4,
        seed=139,
        pool_permutation=permutation,
    )
    global_permutation = torch.arange(128, 192)[permutation]
    assert torch.equal(
        relabelled.queries,
        global_permutation[base.queries - 128],
    )
    assert torch.equal(
        relabelled.hop_targets,
        global_permutation[base.hop_targets - 128],
    )
    assert torch.equal(relabelled.hop_indices, base.hop_indices)
    assert torch.equal(relabelled.word_symbols, base.word_symbols)
    assert torch.equal(
        relabelled.actual_key_permutation,
        base.actual_key_permutation,
    )


def test_dynamic_signed_transport_exactly_aligns_composed_words() -> None:
    dimension = 10
    edges = full_root_edges(dimension)
    subset = torch.arange(edges.shape[0])
    edge_index = torch.full((dimension, dimension), -1, dtype=torch.long)
    root_ids = torch.arange(edges.shape[0])
    edge_index[edges[:, 0], edges[:, 1]] = root_ids
    edge_index[edges[:, 1], edges[:, 0]] = root_ids
    words = torch.tensor([[0, 1, 2, 3], [4, 3, 2, 1]])
    key_permutation = compose_generator_words(words, dimension)
    coordinates = torch.stack(
        [
            torch.randperm(
                dimension,
                generator=torch.Generator().manual_seed(seed),
            )
            for seed in range(2)
        ]
    ).float()
    query_roots = root_signs(coordinates, edges)
    key_coordinates = coordinates.gather(1, key_permutation)
    key_roots = root_signs(key_coordinates, edges)
    signed_indices = dynamic_signed_root_transport(
        key_permutation,
        edges=edges,
        query_root_subset=subset,
        edge_index=edge_index,
    )
    signed_key = torch.cat((-key_roots, key_roots), dim=-1)
    aligned = signed_key.gather(1, signed_indices)
    torch.testing.assert_close(aligned, query_roots)


def test_singleton_retrieval_trains_generator_dictionary(
    tmp_path: Path,
) -> None:
    dimension = 10
    checkpoint = _canon_checkpoint(tmp_path / "canon.pt", dimension)
    config = _config(checkpoint, dimension)
    model = VariableCoxeterTransport(config)
    dictionary = symbol_dictionary(
        dimension,
        config.generator_dictionary_seed,
    )
    batch = generate_path_batch(
        token_pool=token_pools(config)["train"],
        word_pool=torch.arange(dimension - 1).unsqueeze(1),
        dictionary=dictionary,
        dimension=dimension,
        batch_size=8,
        pair_count=6,
        hops=1,
        query_count=3,
        seed=149,
    )
    scores = model.soft_singleton_scores(batch.queries, batch)
    loss = F.cross_entropy(
        scores.flatten(0, 1),
        batch.hop_indices[:, :, 0].flatten(),
    )
    loss.backward()
    assert model.generator_logits.grad is not None
    assert float(model.generator_logits.grad.abs().sum().item()) > 0.0


def test_hard_generator_dictionary_composes_unseen_long_paths(
    tmp_path: Path,
) -> None:
    dimension = 10
    checkpoint = _canon_checkpoint(tmp_path / "canon.pt", dimension)
    config = _config(checkpoint, dimension)
    model = VariableCoxeterTransport(config)
    dictionary = symbol_dictionary(
        dimension,
        config.generator_dictionary_seed,
    )
    with torch.no_grad():
        model.generator_logits.fill_(-20.0)
        model.generator_logits.scatter_(
            1,
            dictionary[:, None],
            20.0,
        )
        model.logit_scale.fill_(2.0)
    inverse = torch.empty_like(dictionary)
    inverse[dictionary] = torch.arange(dictionary.numel())
    word_pool = inverse[torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])]
    result = evaluate_path(
        model,
        token_pool=token_pools(config)["test"],
        word_pool=word_pool,
        dictionary=dictionary,
        pair_count=16,
        hops=4,
        episodes=16,
        batch_size=4,
        data_seed=151,
        device=torch.device("cpu"),
        transport="learned",
        composer="oracle",
        pool_permutation=None,
        order_hard_negatives=True,
    )
    assert result["final_accuracy"] == 1.0
    assert result["all_hops_accuracy"] == 1.0
    reverse = evaluate_path(
        model,
        token_pool=token_pools(config)["test"],
        word_pool=word_pool,
        dictionary=dictionary,
        pair_count=16,
        hops=4,
        episodes=16,
        batch_size=4,
        data_seed=151,
        device=torch.device("cpu"),
        transport="reverse",
        composer="oracle",
        pool_permutation=None,
        order_hard_negatives=True,
    )
    assert reverse["all_hops_accuracy"] == 0.0
    assert reverse["final_accuracy"] <= 1.0 / 16


def test_reverse_order_changes_held_product_transport() -> None:
    dimension = 10
    dictionary = symbol_dictionary(dimension, 157)
    inverse = torch.empty_like(dictionary)
    inverse[dictionary] = torch.arange(dictionary.numel())
    words = inverse[torch.tensor([[2, 3], [5, 4]])]
    forward = actual_word_permutations(
        words,
        dictionary=dictionary,
        dimension=dimension,
    )
    reverse = actual_word_permutations(
        words.flip(dims=(1,)),
        dictionary=dictionary,
        dimension=dimension,
    )
    assert bool((forward != reverse).any(dim=1).all())
