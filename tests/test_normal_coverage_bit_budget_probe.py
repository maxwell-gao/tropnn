import itertools

import torch
from tropnn.layers.hard_lookup import HardLookupSpec, hard_route
from tropnn.tools.normal_coverage_bit_budget_probe import (
    additive_lookup,
    dense_codes,
    fit_additive_lut,
    pair_codes,
    sample_paired_normal_bank,
)


def test_nearest_pair_projection_is_globally_optimal_and_unique() -> None:
    dense, pairs, audit = sample_paired_normal_bank(7, 20, 3)
    keys = {tuple(sorted(pair.tolist())) for pair in pairs}
    assert len(keys) == 20
    assert audit["rejected_projective_pair_duplicates"] >= 0
    roots = []
    for left, right in itertools.combinations(range(7), 2):
        root = torch.zeros(7, dtype=torch.float64)
        root[left], root[right] = 1.0, -1.0
        roots.append(root / 2**0.5)
    root_bank = torch.stack(roots)
    for normal, pair in zip(dense, pairs):
        selected = (normal[pair[0]] - normal[pair[1]]) / 2**0.5
        optimum = (root_bank @ normal).abs().max()
        assert torch.allclose(selected, optimum, atol=1e-12, rtol=0)


def test_pair_codes_delegate_to_shared_hard_router_semantics() -> None:
    x = torch.randn(31, 9, generator=torch.Generator().manual_seed(4), dtype=torch.float64)
    _, pairs, _ = sample_paired_normal_bank(9, 8, 5)
    actual, branches = pair_codes(x, pairs, depth=4)
    supports = pairs.reshape(2, 4, 2)
    expected = hard_route(
        x,
        supports,
        torch.zeros(2, 4, dtype=x.dtype),
        HardLookupSpec(9, 9, 4, "pair", "flat", surrogate="none"),
    )
    assert torch.equal(actual, expected.codes)
    assert torch.equal(branches, expected.branches.reshape(31, 8))


def test_dense_and_pair_widths_are_nested() -> None:
    x = torch.randn(23, 12, generator=torch.Generator().manual_seed(6), dtype=torch.float64)
    dense, pairs, _ = sample_paired_normal_bank(12, 32, 7)
    dense8, _ = dense_codes(x, dense[:8], depth=4)
    dense32, _ = dense_codes(x, dense, depth=4)
    pair8, _ = pair_codes(x, pairs[:8], depth=4)
    pair32, _ = pair_codes(x, pairs, depth=4)
    assert torch.equal(dense8, dense32[:, :2])
    assert torch.equal(pair8, pair32[:, :2])


def test_additive_lut_solver_recovers_an_exact_additive_target() -> None:
    generator = torch.Generator().manual_seed(8)
    codes = torch.randint(4, (4096, 3), generator=generator)
    true_rows = torch.randn(3, 4, 5, generator=generator, dtype=torch.float64)
    target = additive_lookup(codes, true_rows)
    fitted, audit = fit_additive_lut(codes, target, rows=4)
    prediction = additive_lookup(codes, fitted)
    assert audit["rank"] == 10
    assert torch.allclose(prediction, target, atol=1e-10, rtol=1e-10)


def test_additive_projection_commutes_with_a_fixed_linear_teacher() -> None:
    generator = torch.Generator().manual_seed(9)
    codes = torch.randint(4, (2048, 3), generator=generator)
    x = torch.randn(2048, 6, generator=generator, dtype=torch.float64)
    teacher = torch.randn(11, 6, generator=generator, dtype=torch.float64)
    x_rows, _ = fit_additive_lut(codes, x, rows=4)
    y_rows, _ = fit_additive_lut(codes, x @ teacher.T, rows=4)
    transformed_prediction = additive_lookup(codes, x_rows) @ teacher.T
    direct_prediction = additive_lookup(codes, y_rows)
    assert torch.allclose(transformed_prediction, direct_prediction, atol=1e-10, rtol=1e-10)
