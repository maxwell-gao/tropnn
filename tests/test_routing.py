from __future__ import annotations

import torch
from tropnn.layers.routing import (
    gather_route_margins_from_scores,
    minface_mix,
    pack_route_indices,
    recompute_route_margins_torch,
    top2_indices,
)


def test_trop_linear_exact_route_index_packing_dtypes() -> None:
    indices = torch.tensor([[0, 7]], dtype=torch.long)

    assert pack_route_indices(indices, cells=8).dtype == torch.uint8
    assert pack_route_indices(indices, cells=1000).dtype == torch.int16
    assert pack_route_indices(indices, cells=40000).dtype == torch.int64


def test_top2_indices_returns_winner_runner_and_margin() -> None:
    scores = torch.tensor([[[1.0, 3.0, 2.0], [5.0, -1.0, 4.5]]])

    winner, runner, margins = top2_indices(scores)

    assert torch.equal(winner, torch.tensor([[1, 0]]))
    assert torch.equal(runner, torch.tensor([[2, 2]]))
    assert torch.allclose(margins, torch.tensor([[1.0, 0.5]]))


def test_minface_mix_uses_margin_weighted_runner() -> None:
    winner = torch.tensor([[[10.0, 2.0]]])
    runner = torch.tensor([[[14.0, 6.0]]])
    margins = torch.tensor([[1.0]])

    mixed = minface_mix(winner, runner, margins)

    assert torch.allclose(mixed, torch.tensor([[[11.0, 3.0]]]))


def test_margin_helpers_gather_and_recompute_same_values() -> None:
    latent = torch.tensor([[2.0, -1.0], [0.5, 1.0]])
    router_weight = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
            [[0.0, 1.0], [1.0, 1.0], [1.0, -1.0]],
        ]
    )
    router_bias = torch.tensor([[0.0, 0.5, -0.5], [1.0, -1.0, 0.0]])
    scores = torch.matmul(latent, router_weight.reshape(6, 2).t()).view(2, 2, 3) + router_bias.view(1, 2, 3)
    winner, runner, _ = top2_indices(scores)

    gathered = gather_route_margins_from_scores(scores, winner, runner)
    recomputed = recompute_route_margins_torch(latent, router_weight, router_bias, winner, runner)

    assert torch.allclose(gathered, recomputed)
