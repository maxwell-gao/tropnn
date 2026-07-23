from __future__ import annotations

from pathlib import Path

import torch
from tropnn.tools.causal_mqar_composed_multihop import (
    CanonComposer,
    ComposedRunConfig,
)
from tropnn.tools.causal_mqar_latent_relation import (
    generate_latent_relation_batch,
    infer_shared_relation,
    observe_support,
    query_scores_with_transport,
    relation_candidate_permutations,
)
from tropnn.tools.causal_mqar_variable_transport import (
    VariableCoxeterTransport,
    VariableTransportConfig,
    actual_word_permutations,
    symbol_dictionary,
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
    root_count = dimension * (dimension - 1) // 2
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
        root_budget=root_count,
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


def _model(
    tmp_path: Path,
    *,
    dimension: int = 10,
) -> tuple[VariableCoxeterTransport, torch.Tensor]:
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
    return model, dictionary


def _word_pool(dictionary: torch.Tensor) -> torch.Tensor:
    inverse = torch.empty_like(dictionary)
    inverse[dictionary] = torch.arange(dictionary.numel())
    actual_words = torch.tensor(
        [
            [0, 1],
            [1, 0],
            [2, 3],
            [3, 2],
            [4, 5],
            [5, 4],
        ]
    )
    return inverse[actual_words]


def test_full_roots_identify_one_shared_relation_from_one_support(
    tmp_path: Path,
) -> None:
    model, dictionary = _model(tmp_path)
    word_pool = _word_pool(dictionary)
    batch = generate_latent_relation_batch(
        token_pool=torch.arange(128, 192),
        word_pool=word_pool,
        dictionary=dictionary,
        dimension=10,
        batch_size=32,
        pair_count=16,
        query_count=4,
        max_support_count=4,
        seed=211,
    )
    observations = observe_support(model, batch).prefix(1)
    candidates = relation_candidate_permutations(
        word_pool,
        generator_dictionary=model.hard_dictionary(),
        dimension=10,
    )
    inference = infer_shared_relation(
        model,
        observations,
        candidates,
    )
    assert torch.equal(
        inference.selected_indices,
        batch.true_relation_indices,
    )
    assert torch.equal(
        inference.selected_permutations,
        batch.query_batch.actual_key_permutation,
    )


def test_zero_support_does_not_read_hidden_relation(
    tmp_path: Path,
) -> None:
    model, dictionary = _model(tmp_path)
    word_pool = _word_pool(dictionary)
    batch = generate_latent_relation_batch(
        token_pool=torch.arange(128, 192),
        word_pool=word_pool,
        dictionary=dictionary,
        dimension=10,
        batch_size=24,
        pair_count=16,
        query_count=4,
        max_support_count=2,
        seed=223,
    )
    observations = observe_support(model, batch).prefix(0)
    candidates = relation_candidate_permutations(
        word_pool,
        generator_dictionary=model.hard_dictionary(),
        dimension=10,
    )
    inference = infer_shared_relation(
        model,
        observations,
        candidates,
    )
    assert torch.equal(
        inference.candidate_scores,
        torch.zeros_like(inference.candidate_scores),
    )
    assert torch.equal(
        inference.selected_indices,
        torch.zeros_like(inference.selected_indices),
    )
    assert not torch.equal(
        inference.selected_indices,
        batch.true_relation_indices,
    )


def test_inferred_transport_is_shared_across_query_candidates(
    tmp_path: Path,
) -> None:
    model, dictionary = _model(tmp_path)
    word_pool = _word_pool(dictionary)
    batch = generate_latent_relation_batch(
        token_pool=torch.arange(128, 192),
        word_pool=word_pool,
        dictionary=dictionary,
        dimension=10,
        batch_size=16,
        pair_count=16,
        query_count=4,
        max_support_count=2,
        seed=227,
    )
    candidates = actual_word_permutations(
        word_pool,
        dictionary=dictionary,
        dimension=10,
    )
    inference = infer_shared_relation(
        model,
        observe_support(model, batch),
        candidates,
    )
    scores = query_scores_with_transport(
        model,
        batch.query_batch,
        inference.selected_permutations,
    )
    assert scores.shape == (16, 4, 16)
    selected = scores.argmax(dim=-1)
    predicted = (
        batch.query_batch.memory_values[:, None, :]
        .expand(-1, 4, -1)
        .gather(2, selected.unsqueeze(-1))
        .squeeze(-1)
    )
    assert torch.equal(
        predicted,
        batch.query_batch.hop_targets[:, :, 0],
    )


def test_reverse_relation_aligns_orbit_decoy_not_positive(
    tmp_path: Path,
) -> None:
    model, dictionary = _model(tmp_path)
    word_pool = _word_pool(dictionary)
    batch = generate_latent_relation_batch(
        token_pool=torch.arange(128, 192),
        word_pool=word_pool,
        dictionary=dictionary,
        dimension=10,
        batch_size=16,
        pair_count=16,
        query_count=4,
        max_support_count=2,
        seed=229,
    )
    reverse = actual_word_permutations(
        batch.query_batch.word_symbols.flip(dims=(1,)),
        dictionary=dictionary,
        dimension=10,
    )
    scores = query_scores_with_transport(
        model,
        batch.query_batch,
        reverse,
    )
    selected = scores.argmax(dim=-1)
    predicted = (
        batch.query_batch.memory_values[:, None, :]
        .expand(-1, 4, -1)
        .gather(2, selected.unsqueeze(-1))
        .squeeze(-1)
    )
    assert bool(
        (
            predicted
            != batch.query_batch.hop_targets[:, :, 0]
        )
        .all()
        .item()
    )
