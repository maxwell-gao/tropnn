from __future__ import annotations

from pathlib import Path

import pytest
import torch
from tropnn.tools.causal_mqar_composed_multihop import (
    ComposedRootRetriever,
    ComposedRunConfig,
    TwoHopBatch,
    _onehop_loss,
    _twohop_loss,
    generate_twohop_batch,
)
from tropnn.tools.causal_mqar_role_gauge import generate_pool_mqar_batch, token_pools
from tropnn.tools.causal_mqar_root_transport import (
    RootTransportMQARRetriever,
    RootTransportRunConfig,
)


def root_config() -> RootTransportRunConfig:
    return RootTransportRunConfig(
        mode="learned",
        seed=0,
        steps=2,
        batch_size=8,
        learning_rate=0.05,
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
        root_budget=28,
        root_subset_seed=4242,
        residual_tables=0,
        residual_roots_per_side=2,
        assignment_temperature_start=2.0,
        assignment_temperature_end=0.1,
        assignment_entropy_weight=1e-3,
        data_seed=1729,
        codebook_seed=2718,
        query_gauge_seed=101,
        key_gauge_seed=103,
        wrong_key_gauge_seed=107,
        token_relabel_seed=314159,
        device="cpu",
    )


def exact_learned_checkpoint(path: Path) -> Path:
    config = root_config()
    model = RootTransportMQARRetriever(config)
    assert model.relation.assignment_logits is not None
    with torch.no_grad():
        model.relation.assignment_logits.fill_(-20.0)
        model.relation.assignment_logits.scatter_(
            1,
            model.relation.oracle_signed_index[:, None],
            20.0,
        )
    checkpoint = {
        "config": config.__dict__,
        "best_step": 1,
        "state_dict": model.state_dict(),
    }
    torch.save(checkpoint, path)
    return path


def composed_config(
    checkpoint: Path,
    *,
    task: str,
    composer: str,
) -> ComposedRunConfig:
    return ComposedRunConfig(
        task=task,
        composer=composer,
        seed=0,
        relation_checkpoint=str(checkpoint),
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
        composer_tables=4,
        composer_comparisons=3,
        composer_lut_init_std=0.02,
        root_ste_temperature=1.0,
        data_seed=1729,
        codebook_seed=2718,
        token_relabel_seed=314159,
        device="cpu",
    )


def test_twohop_generator_builds_exact_shuffled_chains() -> None:
    pool = torch.arange(64)
    batch = generate_twohop_batch(
        token_pool=pool,
        batch_size=8,
        chain_count=8,
        query_count=4,
        seed=11,
    )
    first_values = batch.memory_values.gather(1, batch.first_indices)
    second_keys = batch.memory_keys.gather(1, batch.second_indices)
    second_values = batch.memory_values.gather(1, batch.second_indices)
    assert torch.equal(first_values, batch.intermediates)
    assert torch.equal(second_keys, batch.intermediates)
    assert torch.equal(second_values, batch.targets)
    assert torch.equal(batch.memory_keys.gather(1, batch.first_indices), batch.queries)


def test_twohop_token_relabel_preserves_chain_structure() -> None:
    pool = torch.arange(128, 192)
    permutation = torch.randperm(64, generator=torch.Generator().manual_seed(13))
    base = generate_twohop_batch(
        token_pool=pool,
        batch_size=4,
        chain_count=8,
        query_count=4,
        seed=17,
    )
    relabelled = generate_twohop_batch(
        token_pool=pool,
        batch_size=4,
        chain_count=8,
        query_count=4,
        seed=17,
        pool_permutation=permutation,
    )
    global_permutation = pool[permutation]
    assert torch.equal(relabelled.queries, global_permutation[base.queries - 128])
    assert torch.equal(relabelled.targets, global_permutation[base.targets - 128])
    assert torch.equal(relabelled.first_indices, base.first_indices)
    assert torch.equal(relabelled.second_indices, base.second_indices)


def test_oracle_composer_preserves_onehop_retrieval(tmp_path: Path) -> None:
    checkpoint = exact_learned_checkpoint(tmp_path / "relation.pt")
    model = ComposedRootRetriever(composed_config(checkpoint, task="onehop", composer="oracle"))
    batch = generate_pool_mqar_batch(
        token_pool=token_pools(model.config)["test"],
        batch_size=16,
        pair_count=16,
        query_count=8,
        seed=19,
    )
    scores = model.score_tokens(batch.queries, batch.keys, batch.values)
    assert torch.equal(scores.argmax(dim=-1), batch.query_indices)


def test_oracle_composer_solves_autoregressive_twohop(tmp_path: Path) -> None:
    checkpoint = exact_learned_checkpoint(tmp_path / "relation.pt")
    model = ComposedRootRetriever(composed_config(checkpoint, task="twohop", composer="oracle"))
    batch = generate_twohop_batch(
        token_pool=token_pools(model.config)["test"],
        batch_size=16,
        chain_count=8,
        query_count=4,
        seed=23,
    )
    first_scores = model.score_tokens(
        batch.queries,
        batch.memory_keys,
        batch.memory_values,
    )
    first_selected = first_scores.argmax(dim=-1)
    selected_middle = (
        batch.memory_values[:, None, :]
        .expand(-1, 4, -1)
        .gather(
            2,
            first_selected.unsqueeze(-1),
        )
        .squeeze(-1)
    )
    second_scores = model.score_tokens(
        selected_middle,
        batch.memory_keys,
        batch.memory_values,
    )
    selected_final = second_scores.argmax(dim=-1)
    predicted = (
        batch.memory_values[:, None, :]
        .expand(-1, 4, -1)
        .gather(
            2,
            selected_final.unsqueeze(-1),
        )
        .squeeze(-1)
    )
    assert torch.equal(predicted, batch.targets)


@pytest.mark.parametrize("composer", ("canon", "pclut"))
def test_trainable_composers_receive_onehop_retrieval_gradients(
    tmp_path: Path,
    composer: str,
) -> None:
    checkpoint = exact_learned_checkpoint(tmp_path / f"relation_{composer}.pt")
    model = ComposedRootRetriever(composed_config(checkpoint, task="onehop", composer=composer))
    batch = generate_pool_mqar_batch(
        token_pool=token_pools(model.config)["train"],
        batch_size=8,
        pair_count=6,
        query_count=3,
        seed=29,
    )
    model.train()
    loss, _ = _onehop_loss(model, batch)
    loss.backward()
    gradients = [parameter.grad for parameter in model.composer.parameters() if parameter.grad is not None]
    assert gradients
    assert sum(float(gradient.abs().sum().item()) for gradient in gradients) > 0.0


def test_canon_composer_receives_both_hop_losses(tmp_path: Path) -> None:
    checkpoint = exact_learned_checkpoint(tmp_path / "relation.pt")
    model = ComposedRootRetriever(composed_config(checkpoint, task="twohop", composer="canon"))
    batch: TwoHopBatch = generate_twohop_batch(
        token_pool=token_pools(model.config)["train"],
        batch_size=8,
        chain_count=4,
        query_count=3,
        seed=31,
    )
    model.train()
    loss, components = _twohop_loss(model, batch)
    loss.backward()
    assert components["hop1_ce"] >= 0.0
    assert components["hop2_ce"] >= 0.0
    assert model.composer.lag_logits.grad is not None
    assert float(model.composer.lag_logits.grad.abs().sum().item()) > 0.0
