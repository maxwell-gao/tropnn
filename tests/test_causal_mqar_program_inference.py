from __future__ import annotations

import torch
from tropnn.tools.causal_mqar_program_inference import (
    ProgramInferenceConfig,
    RootEvidence,
    _compose_right_words,
    _edge_index,
    adjacent_action_dictionary,
    braid_equivalent_words,
    build_coxeter_word_pools,
    canonical_left_program,
    compose_label_programs,
    decode_structured_coxeter,
    generate_root_evidence,
)
from tropnn.tools.causal_mqar_root_transport import (
    full_root_edges,
    root_signs,
)


def _config() -> ProgramInferenceConfig:
    return ProgramInferenceConfig(
        seed=0,
        steps=2,
        batch_size=8,
        learning_rate=1e-3,
        jointpair_learning_rate=1e-2,
        weight_decay=0.0,
        gradient_clip=5.0,
        validation_interval=1,
        validation_episodes=8,
        evaluation_episodes=8,
        dimension=8,
        support_count=64,
        roots_per_support=4,
        root_noise=0.05,
        maximum_decode_steps=28,
        dense_hidden=16,
        jointpair_tables=4,
        jointpair_comparisons=3,
        retrieval_candidates=8,
        train_pool_size=64,
        test_pool_size=64,
        word_pool_seed=107,
        generator_relabel_seed=109,
        random_dictionary_seed=113,
        data_seed=131,
        device="cpu",
    )


def test_train_and_test_products_are_disjoint_and_reduced() -> None:
    config = _config()
    _, products = build_coxeter_word_pools(config)
    train = {
        tuple(row.tolist())
        for length in (1, 2, 3, 4)
        for row in products[f"l{length}"]
    }
    test = {
        tuple(row.tolist())
        for length in (8, 16)
        for row in products[f"l{length}"]
    }
    assert train.isdisjoint(test)


def test_each_support_exposes_only_a_strict_root_subset() -> None:
    config = _config()
    words, products = build_coxeter_word_pools(config)
    del words
    target = products["l8"][:16]
    evidence = generate_root_evidence(
        target,
        edges=full_root_edges(config.dimension),
        support_count=8,
        roots_per_support=4,
        noise=0.05,
        seed=137,
    )
    assert torch.equal(
        evidence.counts.sum(dim=1),
        torch.full((16,), 8 * 4),
    )
    assert bool((evidence.counts < 8).any())


def test_canonical_program_reconstructs_product_and_stop_length() -> None:
    dimension = 8
    actions = adjacent_action_dictionary(dimension)
    relabel = torch.tensor([3, 0, 6, 2, 5, 1, 4])
    dictionary = actions[relabel]
    inverse = torch.empty_like(relabel)
    inverse[relabel] = torch.arange(dimension - 1)
    words = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])
    products = _compose_right_words(words, dimension)
    programs, lengths = canonical_left_program(
        products,
        action_labels_for_generators=inverse,
        action_dictionary=dictionary,
    )
    reconstructed = compose_label_programs(
        programs.clamp_min(0),
        lengths,
        action_dictionary=dictionary,
    )
    assert torch.equal(reconstructed, products)
    assert torch.equal(lengths, torch.tensor([4, 4]))


def test_structured_controller_is_generator_relabel_invariant() -> None:
    dimension = 8
    edges = full_root_edges(dimension)
    edge_index = _edge_index(dimension, edges)
    words = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])
    products = _compose_right_words(words, dimension)
    target_roots = root_signs(products.float(), edges)
    evidence = RootEvidence(
        votes=20.0 * target_roots,
        counts=torch.full_like(target_roots, 20, dtype=torch.long),
        support_count=20,
        roots_per_support=4,
    )
    identity = adjacent_action_dictionary(dimension)
    relabel = torch.tensor([3, 0, 6, 2, 5, 1, 4])
    base = decode_structured_coxeter(
        evidence,
        action_dictionary=identity,
        edge_index=edge_index,
        maximum_steps=28,
    )
    permuted = decode_structured_coxeter(
        evidence,
        action_dictionary=identity[relabel],
        edge_index=edge_index,
        maximum_steps=28,
    )
    assert torch.equal(base.permutations, products)
    assert torch.equal(permuted.permutations, products)


def test_braid_words_have_equal_products_but_different_spelling() -> None:
    words = torch.tensor(
        [
            [0, 1, 0, 3],
            [2, 3, 2, 0],
        ]
    )
    original, alternative = braid_equivalent_words(words)
    assert not torch.equal(original, alternative)
    assert torch.equal(
        _compose_right_words(original, 8),
        _compose_right_words(alternative, 8),
    )


def test_reverse_or_wrong_law_changes_noncommuting_program() -> None:
    dimension = 8
    actions = adjacent_action_dictionary(dimension)
    labels = torch.tensor([[0, 1, 2, 3]])
    lengths = torch.tensor([4])
    correct = compose_label_programs(
        labels,
        lengths,
        action_dictionary=actions,
        law="left",
    )
    wrong = compose_label_programs(
        labels,
        lengths,
        action_dictionary=actions,
        law="right",
    )
    assert not torch.equal(correct, wrong)
