from __future__ import annotations

import torch
from tropnn import GaugeAlignedS4Relation, circulant_relation_edges, s4_fourier_energy, s4_gauge_maps, s4_tables


def test_s4_gauge_map_matches_group_formula_exhaustively() -> None:
    inverse, composition, _ = s4_tables()
    maps = s4_gauge_maps()
    for u in range(24):
        for tau in range(24):
            candidate = u * 24 + tau
            for query in range(24):
                expected = composition[
                    u,
                    composition[
                        composition[composition[inverse[query], tau], torch.arange(24)],
                        inverse[u],
                    ],
                ]
                assert torch.equal(maps[candidate, query], expected)


def test_local_frame_relabeling_is_gauge_covariant() -> None:
    _, composition, _ = s4_tables()
    inverse, _, _ = s4_tables()
    maps = s4_gauge_maps()
    for a in (1, 7, 13, 23):
        for u, tau, query, key in ((3, 5, 8, 11), (17, 2, 4, 21)):
            def relabel(value):
                return composition[a, composition[value, inverse[a]]]

            u_prime = composition[u, inverse[a]]
            tau_prime = relabel(tau)
            original = maps[u * 24 + tau, query, key]
            transformed = maps[u_prime * 24 + tau_prime, relabel(query), relabel(key)]
            assert int(original) == int(transformed)


def test_identity_gauge_with_one_template_is_explicit_shared_relative_sum() -> None:
    layer = GaugeAlignedS4Relation(4, templates=1)
    layer.first_order.data.copy_(torch.arange(24).view(1, 24))
    query = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    key = torch.tensor([[7, 6, 5, 4], [3, 2, 1, 0]])
    inverse, composition, _ = s4_tables()
    relative = composition[inverse[query], key]
    expected = layer.first_order[0, relative].sum(dim=-1) / 2.0
    torch.testing.assert_close(layer.score_aligned_routes(query, key), expected)


def test_zero_second_order_is_exactly_first_order() -> None:
    first = GaugeAlignedS4Relation(16, templates=2)
    second = GaugeAlignedS4Relation(16, templates=2, second_order=True)
    second.first_order.data.copy_(first.first_order)
    second.bias.data.copy_(first.bias)
    query = torch.randint(0, 24, (31, 16))
    key = torch.randint(0, 24, (31, 16))
    assert torch.equal(first.score_aligned_routes(query, key), second.score_aligned_routes(query, key))


def test_degree_four_circulant_has_two_edges_per_table() -> None:
    edges = circulant_relation_edges(16)
    assert edges.shape == (32, 2)
    assert torch.unique(edges, dim=0).shape[0] == 32


def test_s4_fourier_energy_resolves_constant_function() -> None:
    energy = s4_fourier_energy(torch.ones(24))
    assert abs(energy["trivial"] - 1.0) < 1e-12
    assert sum(value for key, value in energy.items() if key not in {"trivial", "coxeter_length"}) < 1e-12
    assert abs(energy["coxeter_length"] - 1.0) < 1e-12
