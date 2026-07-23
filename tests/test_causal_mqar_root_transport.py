from __future__ import annotations

import torch
from tropnn.tools.causal_mqar_induction import target_token_loss
from tropnn.tools.causal_mqar_role_gauge import generate_pool_mqar_batch, token_pools
from tropnn.tools.causal_mqar_root_transport import (
    CrossRootResidualLUT,
    RootTransportMQARRetriever,
    RootTransportRunConfig,
    SignedRootTransportKernel,
    coordinate_permutation,
    full_root_edges,
    nested_root_subset,
    oracle_signed_root_transport,
    root_signs,
)


def config(
    mode: str,
    *,
    root_budget: int = 12,
    residual_tables: int = 0,
) -> RootTransportRunConfig:
    return RootTransportRunConfig(
        mode=mode,
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
        root_budget=root_budget,
        root_subset_seed=4242,
        residual_tables=residual_tables,
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


def test_oracle_signed_transport_exactly_aligns_every_root() -> None:
    dimension = 8
    query_permutation = coordinate_permutation(dimension, 101)
    key_permutation = coordinate_permutation(dimension, 103)
    edges = full_root_edges(dimension)
    key_index, orientation = oracle_signed_root_transport(
        query_permutation,
        key_permutation,
        edges,
    )
    coordinates = torch.stack([torch.randperm(dimension, generator=torch.Generator().manual_seed(seed)) for seed in range(16)]).float()
    query = root_signs(coordinates[..., query_permutation], edges)
    key = root_signs(coordinates[..., key_permutation], edges)
    torch.testing.assert_close(query, key[..., key_index] * orientation)
    assert key_index.unique().numel() == edges.shape[0]


def test_nested_root_budgets_are_prefixes_of_one_seeded_order() -> None:
    small = nested_root_subset(28, 8, 17)
    medium = nested_root_subset(28, 16, 17)
    assert torch.equal(small, medium[:8])
    assert medium.unique().numel() == 16


def test_oracle_root_transport_retrieves_all_targets_without_training() -> None:
    model = RootTransportMQARRetriever(config("oracle", root_budget=28))
    batch = generate_pool_mqar_batch(
        token_pool=token_pools(model.config)["test"],
        batch_size=32,
        pair_count=16,
        query_count=8,
        seed=19,
    )
    scores, values = model(batch)
    selected = scores.argmax(dim=-1)
    predicted = (
        values[:, None, :]
        .expand(-1, 8, -1)
        .gather(
            2,
            selected.unsqueeze(-1),
        )
        .squeeze(-1)
    )
    assert torch.equal(predicted, batch.targets)
    assert model.relation.correspondence_accuracy() == 1.0


def test_learned_transport_has_one_hard_signed_address_per_active_root() -> None:
    model = RootTransportMQARRetriever(config("learned", root_budget=12))
    selected = model.relation.hard_signed_indices()
    assert selected.shape == (12,)
    assert int(selected.min().item()) >= 0
    assert int(selected.max().item()) < 2 * model.relation.root_count
    assert model.relation.hard_inference_bytes == 24


def test_learned_transport_receives_retrieval_ce_gradients() -> None:
    model = RootTransportMQARRetriever(config("learned", root_budget=12))
    batch = generate_pool_mqar_batch(
        token_pool=token_pools(model.config)["train"],
        batch_size=8,
        pair_count=6,
        query_count=3,
        seed=23,
    )
    model.train()
    scores, _ = model(batch)
    loss = target_token_loss(scores, batch.query_indices)
    loss.backward()
    assert model.relation.assignment_logits is not None
    assert model.relation.assignment_logits.grad is not None
    assert float(model.relation.assignment_logits.grad.abs().sum().item()) > 0.0


def test_cross_root_residual_is_joint_and_backpropagates() -> None:
    residual = CrossRootResidualLUT(
        root_count=28,
        tables=4,
        roots_per_side=2,
        seed=29,
    )
    query = torch.where(torch.rand(3, 2, 1, 28) > 0.5, 1.0, -1.0)
    key = torch.where(torch.rand(3, 1, 5, 28) > 0.5, 1.0, -1.0)
    output = residual(query, key)
    assert output.shape == (3, 2, 5)
    output.sum().backward()
    assert residual.weight.grad is not None
    assert float(residual.weight.grad.abs().sum().item()) > 0.0


def test_oracle_kernel_uses_only_budgeted_integer_addresses_at_inference() -> None:
    dimension = 8
    query = coordinate_permutation(dimension, 31)
    key = coordinate_permutation(dimension, 37)
    kernel = SignedRootTransportKernel(
        dimension=dimension,
        root_budget=8,
        root_subset_seed=41,
        query_permutation=query,
        key_permutation=key,
        mode="oracle",
        residual_tables=0,
        residual_roots_per_side=2,
        seed=43,
    )
    assert kernel.assignment_logits is None
    assert kernel.hard_inference_bytes == 16
    assert kernel.correspondence_accuracy() == 1.0
