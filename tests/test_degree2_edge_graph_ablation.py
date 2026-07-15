from __future__ import annotations

from types import SimpleNamespace

import torch

from tropnn.tools.degree2_edge_graph_ablation import (
    all_degree2_edges,
    edge_masks,
    offline_screened_edges,
    sample_edges,
)


def test_geometry_edge_masks_have_expected_semantics() -> None:
    metadata = {
        "anchor_a": torch.tensor([0, 0, 4, 5]),
        "anchor_b": torch.tensor([1, 2, 5, 6]),
        "table": torch.tensor([0, 1, 2, 3]),
        "q_only": torch.tensor([True, True, False, False]),
        "k_only": torch.tensor([False, False, True, True]),
    }
    edges = all_degree2_edges(4)
    masks = edge_masks(edges, metadata)
    shared = {tuple(edge) for edge in edges[masks["shared_anchor"]].tolist()}
    cross = {tuple(edge) for edge in edges[masks["cross_qk"]].tolist()}
    assert shared == {(0, 1), (2, 3)}
    assert cross == {(0, 2), (0, 3), (1, 2), (1, 3)}


def test_sample_edges_is_unique_and_seeded() -> None:
    candidates = all_degree2_edges(32)
    first = sample_edges(candidates, 128, seed=9)
    second = sample_edges(candidates, 128, seed=9)
    assert torch.equal(first, second)
    assert torch.unique(first, dim=0).shape[0] == 128


def test_offline_screening_recovers_known_pair_feature() -> None:
    generator = torch.Generator().manual_seed(5)
    route = 2.0 * torch.randint(0, 2, (4096, 12), generator=generator).float() - 1.0
    target = route[:, 3] * route[:, 9]
    args = SimpleNamespace(
        ridge=0.001,
        cg_iterations=64,
        cg_tolerance=1e-6,
        batch_size=512,
    )
    supports, _ = offline_screened_edges(route, target, budget=4, args=args)
    assert (supports == torch.tensor([3, 9])).all(dim=-1).any()
