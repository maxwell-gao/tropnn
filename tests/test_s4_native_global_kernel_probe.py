from __future__ import annotations

import itertools

import torch
from tropnn.tools.coxeter_relation_probe import LocalS4Router
from tropnn.tools.s4_native_global_kernel_probe import (
    PairIndices,
    build_full_representation,
    build_local_representation,
    build_native_layout,
    channel_matrices,
    decode_local_ranks,
    fit_kernel,
    local_root_vertex_scatter,
    pair_channels,
    stable_ranks,
)


def test_s4_route_decodes_back_to_anchor_slot_ranks() -> None:
    values = torch.tensor([[4.0, 1.0, 3.0, 2.0], [0.0, 3.0, 2.0, 1.0]])
    router = LocalS4Router(input_dim=4, tables=1, seed=7)
    router.anchors.copy_(torch.tensor([[0, 1, 2, 3]]))
    actual = decode_local_ranks(router.route(values))[:, 0]
    expected = stable_ranks(values).to(torch.long)
    torch.testing.assert_close(actual, expected)


def test_k4_root_vertex_sum_is_two_times_centered_rank() -> None:
    routes = torch.arange(24).view(24, 1)
    ranks = decode_local_ranks(routes)
    expected = 2.0 * (ranks.to(torch.float32) - 1.5)
    torch.testing.assert_close(local_root_vertex_scatter(ranks), expected)


def test_comparison_root_dot_is_exact_normalized_xor_popcount() -> None:
    anchors = torch.tensor([[0, 1, 2, 3]])
    router = LocalS4Router(input_dim=4, tables=1, seed=3)
    router.anchors.copy_(anchors)
    values = torch.tensor([[0.0, 1.0, 2.0, 3.0], [3.0, 1.0, 2.0, 0.0]])
    representation, _ = build_local_representation(router.route(values), anchors, input_dim=4)
    left, right = representation.comparison_root
    actual = left @ right
    left_bits = left > 0
    right_bits = right > 0
    hamming = torch.logical_xor(left_bits, right_bits).sum()
    expected = 1.0 - 2.0 * hamming / math_comb(4, 2)
    torch.testing.assert_close(actual, expected)


def math_comb(n: int, r: int) -> int:
    return len(tuple(itertools.combinations(range(n), r)))


def test_a2_centered_chamber_kernel_is_one_or_minus_one_fifth() -> None:
    anchors = torch.tensor([[0, 1, 2, 3]])
    router = LocalS4Router(input_dim=4, tables=1, seed=5)
    router.anchors.copy_(anchors)
    values = torch.tensor(
        [
            [0.0, 1.0, 2.0, 3.0],
            [0.0, 1.0, 3.0, 2.0],
            [2.0, 1.0, 0.0, 3.0],
        ]
    )
    representation, _ = build_local_representation(router.route(values), anchors, input_dim=4)
    same = PairIndices(torch.tensor([0]), torch.tensor([0]))
    different_everywhere = PairIndices(torch.tensor([0]), torch.tensor([2]))
    same_score = pair_channels(representation, representation, same, batch_size=1)[0, 2]
    different_score = pair_channels(representation, representation, different_everywhere, batch_size=1)[0, 2]
    torch.testing.assert_close(same_score, torch.tensor(1.0))
    # Reversing the first three coordinates changes every one of the four local triples here.
    assert different_score < 1.0


def test_duplicate_charts_merge_to_identical_native_features_without_ties() -> None:
    anchors = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])
    router = LocalS4Router(input_dim=4, tables=2, seed=11)
    router.anchors.copy_(anchors)
    values = torch.tensor([[0.2, -1.0, 3.0, 0.7], [4.0, 2.0, 1.0, -3.0]])
    representation, diagnostics = build_local_representation(router.route(values), anchors, input_dim=4)
    full = build_full_representation(values)
    torch.testing.assert_close(representation.comparison_root, full.comparison_root)
    torch.testing.assert_close(representation.a2_code, full.a2_code)
    assert diagnostics["edge_duplicate_disagreement_rate"] == 0.0
    assert diagnostics["a2_duplicate_disagreement_rate"] == 0.0


def test_native_layout_uses_global_coordinate_labels() -> None:
    anchors = torch.tensor([[0, 2, 4, 6], [6, 4, 2, 0], [1, 3, 5, 7]])
    layout = build_native_layout(anchors, input_dim=8)
    assert len(layout.edge_keys) == 12
    assert len(layout.triple_keys) == 8
    assert max(len(items) for items in layout.edge_occurrences) == 2
    assert max(len(items) for items in layout.triple_occurrences) == 2


def test_affine_kernel_fit_recovers_matching_full_centered_rank_teacher() -> None:
    generator = torch.Generator().manual_seed(17)
    query_values = torch.randn(31, 6, generator=generator)
    key_values = torch.randn(29, 6, generator=generator)
    query = build_full_representation(query_values)
    key = build_full_representation(key_values)
    indices = PairIndices(
        torch.randint(31, (503,), generator=generator),
        torch.randint(29, (503,), generator=generator),
    )
    channels = pair_channels(query, key, indices, batch_size=64)
    fit = fit_kernel(channels[:400], channels[:400, 0], channels[400:], channels[400:, 0], (0,), (0.0,))
    assert fit.validation_r2 > 1.0 - 1e-10
    torch.testing.assert_close(fit.weight, torch.ones_like(fit.weight), rtol=1e-8, atol=1e-8)
    matrices = channel_matrices(query, key, query_batch=7)
    torch.testing.assert_close(matrices[0], query.centered_rank @ key.centered_rank.T)
