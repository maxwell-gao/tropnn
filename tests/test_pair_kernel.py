from __future__ import annotations

import math

import torch
from tropnn.layers.pair_kernel import (
    BalancedS4Router,
    CoxeterPairScorer,
    GlobalChamberKernel,
    RootIncidenceKernel,
    SameTableFullKernel,
    coxeter_representation_features,
)


def test_balanced_router_is_a_two_cover_with_legal_chambers() -> None:
    router = BalancedS4Router(input_dim=32, tables=16, coverage=2, seed=3)
    counts = torch.bincount(router.anchors.flatten(), minlength=32)
    assert torch.equal(counts, torch.full((32,), 2))
    features = router.route(torch.randn(41, 32, generator=torch.Generator().manual_seed(5)))
    assert features.routes.min() >= 0
    assert features.routes.max() < 24
    torch.testing.assert_close(features.roots.abs(), torch.full_like(features.roots, 1.0 / router.roots**0.5))


def test_root_incidence_support_is_exactly_signed_incidence_support() -> None:
    router = BalancedS4Router(seed=7)
    support = torch.zeros(router.roots, router.roots, dtype=torch.bool)
    support[router.support_rows, router.support_columns] = True
    assert torch.equal(support, router.root_incidence.T @ router.root_incidence != 0.0)
    assert bool((router.support_rows != router.support_columns).any())


def test_root_sparse_score_matches_explicit_dense_operator() -> None:
    router = BalancedS4Router(seed=11)
    kernel = RootIncidenceKernel(router, seed=13)
    query = router.route(torch.randn(17, 32, generator=torch.Generator().manual_seed(17)))
    key = router.route(torch.randn(17, 32, generator=torch.Generator().manual_seed(19)))
    expected = ((query.roots @ kernel.dense_operator(router.roots)) * key.roots).sum(dim=-1) + kernel.bias
    torch.testing.assert_close(kernel.hard_score(query, key), expected)
    torch.testing.assert_close(kernel.cached_score(query.roots, key.roots), expected)
    torch.testing.assert_close(
        kernel.score_from_cache(
            query.roots,
            key.roots,
            kernel.transform_roots(query.roots),
            kernel.transform_roots(key.roots),
        ),
        expected,
    )
    symmetric = 0.5 * (expected + ((key.roots @ kernel.dense_operator(router.roots)) * query.roots).sum(dim=-1) + kernel.bias)
    torch.testing.assert_close(kernel.cached_score(query.roots, key.roots, symmetry="symmetric"), symmetric)


def test_root_signs_decode_the_globally_labelled_coordinate_comparisons() -> None:
    router = BalancedS4Router(seed=21)
    coordinates = torch.randn(13, 32, generator=torch.Generator().manual_seed(22))
    features = router.route(coordinates)
    edges = router.root_edges
    expected = torch.where(coordinates[:, edges[:, 0]] > coordinates[:, edges[:, 1]], 1.0, -1.0)
    expected /= math.sqrt(router.roots)
    torch.testing.assert_close(features.roots, expected)


def test_global_rank_twelve_and_same_table_match_payload_budget() -> None:
    router = BalancedS4Router(seed=23)
    same = SameTableFullKernel(router.tables)
    global_kernel = GlobalChamberKernel(router.tables, 12)
    assert same.weight.numel() == global_kernel.query_factor.numel() + global_kernel.key_factor.numel() == 9216


def test_shared_coxeter_features_are_orthogonal_and_constant_first() -> None:
    features = coxeter_representation_features()
    assert features.shape == (24, 12)
    torch.testing.assert_close(features.T @ features, 24.0 * torch.eye(12), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(features[:, 0], torch.ones(24), rtol=1e-5, atol=1e-5)


def test_pair_scorer_symmetry_and_antisymmetry_are_exact() -> None:
    router = BalancedS4Router(seed=29)
    query = torch.randn(9, 32, generator=torch.Generator().manual_seed(31))
    key = torch.randn(9, 32, generator=torch.Generator().manual_seed(37))
    kernel = GlobalChamberKernel(router.tables, 4, seed=41)
    symmetric = CoxeterPairScorer(router, kernel, symmetry="symmetric").eval()
    torch.testing.assert_close(symmetric(query, key), symmetric(key, query))
    antisymmetric = CoxeterPairScorer(router, kernel, symmetry="antisymmetric").eval()
    torch.testing.assert_close(antisymmetric(query, key), -antisymmetric(key, query))


def test_pair_scorer_ste_preserves_forward_and_supplies_coordinate_gradients() -> None:
    router = BalancedS4Router(input_dim=8, tables=4, coverage=2, seed=43)
    kernel = SameTableFullKernel(router.tables, seed=47)
    scorer = CoxeterPairScorer(router, kernel).train()
    query = torch.randn(5, 8, generator=torch.Generator().manual_seed(53), requires_grad=True)
    key = torch.randn(5, 8, generator=torch.Generator().manual_seed(59), requires_grad=True)
    train_score = scorer(query, key)
    scorer.eval()
    hard_score = scorer(query, key)
    torch.testing.assert_close(train_score.detach(), hard_score.detach())
    train_score.sum().backward()
    assert query.grad is not None and torch.isfinite(query.grad).all() and query.grad.abs().sum() > 0
    assert key.grad is not None and torch.isfinite(key.grad).all() and key.grad.abs().sum() > 0
    assert kernel.weight.grad is not None and torch.isfinite(kernel.weight.grad).all() and kernel.weight.grad.abs().sum() > 0
