from __future__ import annotations

import torch
from tropnn.tools.s4_cross_table_kernel_probe import (
    S4_ORDER,
    PairSplit,
    all_table_edges,
    categorical_edge_prediction,
    edge_codes,
    ordinal_coordinates,
    tower_embeddings,
    tower_pair_prediction,
)


def test_same_table_edges_use_only_matched_table_pairs() -> None:
    query_route = torch.tensor([[1, 2], [3, 4]])
    key_route = torch.tensor([[5, 6], [7, 8]])
    split = PairSplit(torch.tensor([0, 1]), torch.tensor([1, 0]), torch.zeros(2))
    edges = all_table_edges(2, diagonal=True)
    codes = edge_codes(query_route, key_route, split.query_index, split.key_index, edges)
    expected = torch.tensor(
        [
            [1 * S4_ORDER + 7, 2 * S4_ORDER + 8],
            [3 * S4_ORDER + 5, 4 * S4_ORDER + 6],
        ]
    )
    assert torch.equal(codes, expected)


def test_global_tower_expands_to_every_cross_table_block() -> None:
    query_route = torch.tensor([[0, 1]])
    key_route = torch.tensor([[2, 3]])
    query_factor = torch.arange(2 * S4_ORDER * 2, dtype=torch.float32).view(2, S4_ORDER, 2) / 100.0
    key_factor = torch.flip(query_factor, dims=(1,))
    query_embedding = tower_embeddings(query_route, query_factor)
    key_embedding = tower_embeddings(key_route, key_factor)
    tower_score = (query_embedding * key_embedding).sum() / 2.0

    explicit = torch.zeros(())
    for query_table in range(2):
        for key_table in range(2):
            explicit += (query_factor[query_table, query_route[0, query_table]] * key_factor[key_table, key_route[0, key_table]]).sum() / 2.0
    torch.testing.assert_close(tower_score, explicit)


def test_rank_twelve_matches_s4_same_table_payload_budget() -> None:
    tables = 16
    rank = 12
    same_table = tables * S4_ORDER * S4_ORDER
    global_tower = 2 * tables * S4_ORDER * rank
    sparse = tables * S4_ORDER * S4_ORDER
    dense = tables * tables * S4_ORDER * S4_ORDER
    assert same_table == global_tower == sparse == 9216
    assert dense == 147456


def test_ordinal_coordinates_depend_only_on_stable_order() -> None:
    first = torch.tensor([[10.0, -3.0, 2.0, 7.0]])
    second = torch.tensor([[100.0, -5.0, 0.0, 9.0]])
    torch.testing.assert_close(ordinal_coordinates(first), ordinal_coordinates(second))


def test_tower_pair_prediction_applies_bias_and_target_scale() -> None:
    query_route = torch.tensor([[0, 1]])
    key_route = torch.tensor([[2, 3]])
    split = PairSplit(torch.tensor([0]), torch.tensor([0]), torch.zeros(1))
    query_factor = torch.zeros(2, S4_ORDER, 1)
    key_factor = torch.zeros(2, S4_ORDER, 1)
    query_factor[0, 0] = 1.0
    query_factor[1, 1] = 2.0
    key_factor[0, 2] = 3.0
    key_factor[1, 3] = 4.0
    prediction = tower_pair_prediction(
        query_route,
        key_route,
        split,
        query_factor,
        key_factor,
        torch.tensor(0.5),
        torch.tensor(2.0),
    )
    torch.testing.assert_close(prediction, torch.tensor([22.0]))


def test_categorical_edge_prediction_uses_unit_norm_edge_scaling() -> None:
    codes = torch.tensor([[0, 1], [2, 3]])
    coefficient = torch.zeros(2, S4_ORDER * S4_ORDER)
    coefficient[0, 0] = 2.0
    coefficient[1, 1] = 4.0
    prediction = categorical_edge_prediction(codes, coefficient, torch.tensor(1.0))
    torch.testing.assert_close(prediction[0], torch.tensor(1.0 + 6.0 / (2.0**0.5)))
    torch.testing.assert_close(prediction[1], torch.tensor(1.0))
