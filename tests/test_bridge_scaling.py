from __future__ import annotations

import math

import torch
from tropnn.tools.bridge_scaling import (
    bridge_diagnostics,
    chamber_distinguishability,
    chamber_entropy,
    effective_rank,
    fit_bridge_exponents_per_family,
    fit_bridge_scaling,
    pack_capacity,
    vector_distinguishability,
)


def test_vector_distinguishability_diagonal_is_zero() -> None:
    reps = torch.randn(5, 7)
    D = vector_distinguishability(reps)

    assert D.shape == (5, 5)
    assert torch.allclose(D.diag(), torch.zeros(5))
    assert (D >= 0.0).all() and (D <= 1.0).all()


def test_vector_distinguishability_orthogonal_basis_is_one_off_diag() -> None:
    reps = torch.eye(4, 4)
    D = vector_distinguishability(reps)

    off_diag = D[~torch.eye(4, dtype=torch.bool)]

    assert torch.allclose(off_diag, torch.ones_like(off_diag))


def test_chamber_distinguishability_disagreement_fraction() -> None:
    indices = torch.tensor(
        [
            [0, 1, 2],
            [0, 1, 3],
            [4, 5, 6],
        ],
        dtype=torch.long,
    )
    D = chamber_distinguishability(indices)

    assert D.shape == (3, 3)
    assert torch.allclose(D.diag(), torch.zeros(3))
    assert math.isclose(D[0, 1].item(), 1.0 / 3.0, abs_tol=1e-6)
    assert math.isclose(D[0, 2].item(), 1.0, abs_tol=1e-6)


def test_effective_rank_bounds() -> None:
    n = 8
    zero = torch.zeros(n, n)
    full = 1.0 - torch.eye(n)

    assert math.isclose(effective_rank(zero, alpha=4.0), 1.0, abs_tol=1e-4)
    big_alpha = effective_rank(full, alpha=64.0)
    assert big_alpha > n - 0.5


def test_pack_capacity_endpoints() -> None:
    n = 6
    full = 1.0 - torch.eye(n)

    assert pack_capacity(full, tau=0.5) == n
    assert pack_capacity(torch.zeros(n, n), tau=0.5) == 1


def test_chamber_entropy_unique_vs_collapsed() -> None:
    unique_idx = torch.arange(8).view(8, 1)
    H_unique, distinct_unique = chamber_entropy(unique_idx)
    assert distinct_unique == 8
    assert math.isclose(H_unique, math.log(8), abs_tol=1e-4)

    collapsed = torch.zeros(8, 1, dtype=torch.long)
    H_collapsed, distinct_collapsed = chamber_entropy(collapsed)
    assert distinct_collapsed == 1
    assert math.isclose(H_collapsed, 0.0, abs_tol=1e-6)


def test_bridge_diagnostics_reports_both_views_when_provided() -> None:
    reps = torch.randn(6, 5)
    indices = torch.randint(0, 4, (6, 3))

    diag = bridge_diagnostics(reps_vector=reps, reps_chamber=indices, alphas=(4.0,), taus=(0.3,))

    for key in (
        "bridge_K_vec_a4",
        "bridge_pack_vec_t0.3",
        "bridge_K_chamb_a4",
        "bridge_pack_chamb_t0.3",
        "bridge_chamber_entropy",
        "bridge_chamber_distinct",
    ):
        assert key in diag
        assert math.isfinite(diag[key])


def test_bridge_diagnostics_chamber_only_skips_vector_keys() -> None:
    indices = torch.randint(0, 5, (4, 3))

    diag = bridge_diagnostics(reps_chamber=indices, alphas=(4.0,), taus=(0.3,))

    assert "bridge_K_chamb_a4" in diag
    assert "bridge_K_vec_a4" not in diag


def test_fit_bridge_scaling_recovers_known_slope() -> None:
    rows = []
    for x in (1.0, 2.0, 4.0, 8.0, 16.0):
        rows.append({"capacity": x, "loss": 1.0 / (x**1.5)})
    fit = fit_bridge_scaling(rows, capacity_key="capacity", loss_key="loss")

    assert math.isclose(fit["beta"], 1.5, abs_tol=1e-3)
    assert fit["r2"] > 0.999
    assert fit["points"] == 5.0


def test_fit_bridge_scaling_handles_invalid_rows() -> None:
    rows = [
        {"capacity": -1.0, "loss": 0.5},
        {"capacity": 2.0, "loss": 0.0},
        {"capacity": 4.0, "loss": 0.5},
        {"missing": 1.0},
    ]
    fit = fit_bridge_scaling(rows, capacity_key="capacity", loss_key="loss")

    assert fit["points"] == 1.0
    assert math.isnan(fit["beta"])


def test_fit_bridge_exponents_per_family_emits_pooled_and_per_family() -> None:
    rows = [
        {"family": "A", "K": 1.0, "loss": 1.0},
        {"family": "A", "K": 4.0, "loss": 0.25},
        {"family": "B", "K": 2.0, "loss": 0.5},
        {"family": "B", "K": 8.0, "loss": 0.125},
    ]
    fits = fit_bridge_exponents_per_family(rows, capacity_keys=("K",), loss_key="loss")

    scopes = {(f["scope"], f["family"]) for f in fits}
    assert ("per_family", "A") in scopes
    assert ("per_family", "B") in scopes
    assert ("pooled", "*") in scopes
