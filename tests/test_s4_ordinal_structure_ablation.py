from __future__ import annotations

import torch
from tropnn.tools.s4_ordinal_structure_ablation import (
    coxeter_representation_features,
    feature_embeddings,
    feature_side_normal_matvec,
    feature_side_rhs,
    make_label_permutations,
    make_random_rotations,
    random_shared_features,
    relabel_factors,
    relabel_routes,
    rotated_24way_routes,
    route_feature_rows,
)


def test_per_table_relabeling_is_an_exact_free_factor_reparameterization() -> None:
    generator = torch.Generator().manual_seed(3)
    routes = torch.randint(24, (17, 4), generator=generator)
    factor = torch.randn(4, 24, 5, generator=generator)
    permutation = make_label_permutations(4, 11)
    relabeled_route = relabel_routes(routes, permutation)
    relabeled_factor = relabel_factors(factor, permutation)
    table = torch.arange(4).view(1, -1)
    original = factor[table, routes].sum(dim=1)
    transformed = relabeled_factor[table, relabeled_route].sum(dim=1)
    torch.testing.assert_close(original, transformed)


def test_random_rotated_partition_has_exactly_24_legal_codes() -> None:
    generator = torch.Generator().manual_seed(5)
    values = torch.randn(4096, 8, generator=generator)
    anchors = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    rotations = make_random_rotations(2, 13)
    routes = rotated_24way_routes(values, anchors, rotations)
    assert routes.shape == (4096, 2)
    assert routes.min() == 0
    assert routes.max() == 23
    assert all(torch.unique(routes[:, table]).numel() == 24 for table in range(2))


def test_coxeter_and_random_feature_controls_are_matched_orthonormal_bases() -> None:
    coxeter = coxeter_representation_features()
    random = random_shared_features(17)
    assert coxeter.shape == random.shape == (24, 12)
    expected = 24.0 * torch.eye(12)
    torch.testing.assert_close(coxeter.T @ coxeter, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(random.T @ random, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(coxeter[:, 0], torch.ones(24))
    torch.testing.assert_close(random[:, 0], torch.ones(24))


def test_shared_feature_embedding_equals_materialized_factor_lut() -> None:
    generator = torch.Generator().manual_seed(19)
    feature_table = coxeter_representation_features()
    routes = torch.randint(24, (13, 3), generator=generator)
    coefficient = torch.randn(3, 12, 7, generator=generator)
    features = route_feature_rows(routes, feature_table)
    actual = feature_embeddings(features, coefficient)
    materialized = torch.einsum("pd,tdr->tpr", feature_table, coefficient)
    table = torch.arange(3).view(1, -1)
    expected = materialized[table, routes].sum(dim=1)
    torch.testing.assert_close(actual, expected)


def test_feature_normal_equations_match_explicit_design_matrix() -> None:
    generator = torch.Generator().manual_seed(23)
    objects, samples, tables, dimensions, rank = 7, 11, 3, 4, 2
    route_features = torch.randn(objects, tables, dimensions, generator=generator)
    object_index = torch.randint(objects, (samples,), generator=generator)
    fixed = torch.randn(samples, rank, generator=generator)
    coefficient = torch.randn(tables, dimensions, rank, generator=generator)
    target = torch.randn(samples, generator=generator)
    ridge = 0.7
    design = (route_features[object_index][:, :, :, None] * fixed[:, None, None, :] / tables).reshape(samples, -1)
    explicit_rhs = (design.T @ target / samples).reshape_as(coefficient)
    explicit_normal = ((design.T @ (design @ coefficient.reshape(-1)) + ridge * coefficient.reshape(-1)) / samples).reshape_as(coefficient)
    actual_rhs = feature_side_rhs(
        route_features,
        object_index,
        fixed,
        target,
        tables=tables,
        batch_size=4,
    )
    actual_normal = feature_side_normal_matvec(
        route_features,
        object_index,
        fixed,
        coefficient,
        ridge=ridge,
        tables=tables,
        batch_size=4,
    )
    torch.testing.assert_close(actual_rhs, explicit_rhs, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual_normal, explicit_normal, rtol=1e-5, atol=1e-6)
