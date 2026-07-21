from __future__ import annotations

import torch
from tropnn.tools.coxeter_relation_probe import LocalS4Router, permutation_tables
from tropnn.tools.s4_native_global_kernel_probe import PairIndices, build_native_layout
from tropnn.tools.s4_native_kernel_hierarchy_probe import (
    bilinear_pair_features,
    entry_pair_features,
    fit_linear_features,
    intrinsic_kernel_tables,
    root_geometry,
    structured_teacher_operator,
)


def test_intrinsic_kernels_are_symmetric_and_centered() -> None:
    tables = intrinsic_kernel_tables(mallows_beta=0.75, diffusion_time=0.75)
    for table in (tables.kendall, tables.mallows, tables.cayley_diffusion):
        torch.testing.assert_close(table, table.T, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(table.mean(), torch.tensor(0.0), atol=1e-6, rtol=0.0)
        assert torch.linalg.eigvalsh(table.to(torch.float64)).min() > -1e-5


def test_kendall_and_mallows_depend_only_on_relative_coxeter_length() -> None:
    tables = intrinsic_kernel_tables(mallows_beta=0.75, diffusion_time=0.75)
    inverse, composition, lengths = permutation_tables()
    states = torch.arange(24)
    relative = composition[inverse[:, None], states[None, :]]
    distance = lengths[relative]
    for value in range(7):
        mask = distance == value
        assert torch.unique(tables.kendall[mask]).numel() == 1
        assert torch.unique(tables.mallows[mask]).numel() == 1


def test_cayley_heat_kernel_is_not_only_a_length_exponential() -> None:
    tables = intrinsic_kernel_tables(mallows_beta=0.75, diffusion_time=0.75)
    inverse, composition, lengths = permutation_tables()
    states = torch.arange(24)
    relative = composition[inverse[:, None], states[None, :]]
    distance = lengths[relative]
    assert any(torch.unique(tables.cayley_diffusion[distance == value].round(decimals=6)).numel() > 1 for value in range(1, 6))


def test_root_geometry_has_signed_incidence_support_and_hodge_projectors() -> None:
    anchors = torch.tensor([[0, 1, 2, 3], [2, 3, 4, 5]])
    layout = build_native_layout(anchors, input_dim=6)
    geometry = root_geometry(layout)
    gram = geometry.incidence.T @ geometry.incidence
    support = torch.zeros_like(gram, dtype=torch.bool)
    support[geometry.support_rows, geometry.support_columns] = True
    torch.testing.assert_close(support, gram != 0.0)
    torch.testing.assert_close(
        geometry.hodge_gradient @ geometry.hodge_gradient,
        geometry.hodge_gradient,
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(
        geometry.hodge_residual @ geometry.hodge_residual,
        geometry.hodge_residual,
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(
        geometry.hodge_gradient @ geometry.hodge_residual,
        torch.zeros_like(geometry.identity),
        rtol=1e-5,
        atol=1e-5,
    )


def test_anisotropic_teacher_is_asymmetric_and_incidence_sparse() -> None:
    router = LocalS4Router(input_dim=9, tables=5, seed=13)
    layout = build_native_layout(router.anchors, input_dim=9)
    geometry = root_geometry(layout)
    operator = structured_teacher_operator("root_incidence_anisotropic", geometry, seed=13)
    support = torch.zeros_like(operator, dtype=torch.bool)
    support[geometry.support_rows, geometry.support_columns] = True
    assert torch.count_nonzero(operator[~support]) == 0
    assert torch.linalg.matrix_norm(operator - operator.T) > 0.1 * torch.linalg.matrix_norm(operator)


def test_entry_design_and_basis_design_equal_explicit_bilinear_score() -> None:
    generator = torch.Generator().manual_seed(17)
    query = torch.randn(11, 7, generator=generator)
    key = torch.randn(13, 7, generator=generator)
    indices = PairIndices(torch.tensor([0, 3, 5, 9]), torch.tensor([2, 4, 7, 12]))
    rows = torch.tensor([0, 1, 3, 6])
    columns = torch.tensor([2, 5, 4, 0])
    coefficient = torch.tensor([0.2, -0.7, 1.3, 0.5])
    entry_features = entry_pair_features(query, key, indices, rows, columns, batch_size=2).float()
    operator = torch.zeros(7, 7)
    operator[rows, columns] = coefficient
    expected = ((query[indices.query] @ operator) * key[indices.key]).sum(dim=-1)
    torch.testing.assert_close(entry_features @ coefficient, expected, rtol=2e-3, atol=2e-3)
    basis_features = bilinear_pair_features(query, key, indices, (operator,), batch_size=2)
    torch.testing.assert_close(basis_features[:, 0], expected)


def test_ridge_fit_recovers_a_matched_linear_native_teacher() -> None:
    generator = torch.Generator().manual_seed(19)
    features = torch.randn(600, 5, generator=generator)
    weight = torch.tensor([0.7, -0.2, 1.1, 0.4, -0.8])
    target = features @ weight + 0.35
    fit = fit_linear_features(
        features[:500],
        target[:500],
        features[500:],
        target[500:],
        ridge_grid=(1e-8,),
        iterations=64,
        tolerance=1e-8,
        batch_size=128,
    )
    assert fit.validation_r2 > 1.0 - 1e-10
    torch.testing.assert_close(fit.weight, weight, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(fit.bias, torch.tensor(0.35), rtol=1e-5, atol=1e-5)
