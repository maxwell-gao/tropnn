from __future__ import annotations

import pytest
import torch
from tropnn import TropDictLinear


def test_trop_dict_shapes_for_2d_and_3d_inputs() -> None:
    layer = TropDictLinear(5, 3, heads=4, cells=3, route_terms=2, dict_size=8, dict_sparsity=2, seed=0)

    y2 = layer(torch.randn(7, 5))
    y3 = layer(torch.randn(7, 2, 5))

    assert y2.shape == (7, 1, 3)
    assert y3.shape == (7, 2, 3)
    assert layer._last_indices is not None
    assert layer._last_indices.shape == (7, 2, 4)
    assert layer._last_margins is not None


def test_trop_dict_eval_uses_only_winner_code() -> None:
    layer = TropDictLinear(
        2,
        4,
        heads=1,
        cells=3,
        route_terms=1,
        dict_size=4,
        dict_sparsity=1,
        seed=0,
        use_output_scaling=False,
    )
    with torch.no_grad():
        layer.anchors[0, :, 0] = torch.tensor([0, 0, 1])
        layer.router_weight[0, :, 0] = torch.tensor([1.0, -1.0, 1.0])
        layer.router_bias.zero_()
        layer.coeff[0, :, 0] = torch.tensor([1.0, 1.0, 1.0])
        layer.basis.zero_()
        layer.basis[0, 0] = 10.0
        layer.basis[1, 1] = 20.0
        layer.basis[2, 2] = 30.0
        layer.support[0, 0, 0] = 0
        layer.support[0, 1, 0] = 1
        layer.support[0, 2, 0] = 2
        layer.bias.zero_()

    layer.eval()

    out_pos = layer(torch.tensor([[2.0, 0.0]])).squeeze()
    out_neg = layer(torch.tensor([[-2.0, 0.0]])).squeeze()
    assert torch.allclose(out_pos, torch.tensor([10.0, 0.0, 0.0, 0.0]))
    assert torch.allclose(out_neg, torch.tensor([0.0, 20.0, 0.0, 0.0]))


def test_trop_dict_train_uses_minface_interpolation() -> None:
    layer = TropDictLinear(
        1,
        2,
        heads=1,
        cells=3,
        route_terms=1,
        dict_size=4,
        dict_sparsity=1,
        seed=0,
        use_output_scaling=False,
    )
    with torch.no_grad():
        layer.anchors.fill_(0)
        layer.router_weight[0, :, 0] = torch.tensor([1.0, 0.9, -1.0])
        layer.router_bias.zero_()
        layer.coeff[0, :, 0] = torch.tensor([1.0, 1.0, 1.0])
        layer.basis.zero_()
        layer.basis[0, 0] = 10.0
        layer.basis[1, 0] = 20.0
        layer.basis[2, 0] = 30.0
        layer.support[0, 0, 0] = 0
        layer.support[0, 1, 0] = 1
        layer.support[0, 2, 0] = 2
        layer.bias.zero_()

    layer.train()
    y = layer(torch.tensor([[1.0]])).squeeze()
    expected_first = torch.tensor(10.0 + (0.5 / 1.1) * (20.0 - 10.0))
    assert torch.allclose(y, torch.tensor([expected_first.item(), 0.0]), atol=1e-6)


def test_trop_dict_train_propagates_grads_through_dict() -> None:
    layer = TropDictLinear(6, 4, heads=6, cells=4, route_terms=3, dict_size=12, dict_sparsity=2, seed=0)
    x = torch.randn(8, 2, 6, requires_grad=True)

    loss = layer(x).square().mean()
    loss.backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    for param in (layer.router_weight, layer.coeff, layer.basis, layer.bias):
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()
        assert float(param.grad.abs().sum()) > 0.0


def test_trop_dict_dictionary_loss_decreases_offdiag() -> None:
    layer = TropDictLinear(
        4,
        8,
        heads=2,
        cells=3,
        dict_size=6,
        dict_sparsity=2,
        dict_init="gaussian",
        seed=0,
    )
    optim = torch.optim.SGD([layer.basis], lr=0.5)
    initial = float(layer.dictionary_loss().detach())
    for _ in range(20):
        optim.zero_grad()
        layer.dictionary_loss(weight=1.0).backward()
        optim.step()
    final = float(layer.dictionary_loss().detach())

    assert final < initial


def test_trop_dict_dictionary_loss_zero_weight_returns_zero() -> None:
    layer = TropDictLinear(4, 4, dict_size=4, dict_sparsity=2, seed=0)

    loss = layer.dictionary_loss(weight=0.0)

    assert float(loss) == 0.0
    assert loss.shape == ()


def test_trop_dict_supports_disjoint_per_cell_atoms() -> None:
    layer = TropDictLinear(3, 5, heads=2, cells=2, route_terms=1, dict_size=4, dict_sparsity=4, seed=0)

    assert layer.support.shape == (2, 2, 4)
    for h in range(2):
        for k in range(2):
            assert sorted(layer.support[h, k].tolist()) == [0, 1, 2, 3]


def test_trop_dict_orthogonal_init_has_unit_norm_rows() -> None:
    layer = TropDictLinear(4, 16, dict_size=12, dict_sparsity=2, dict_init="orthogonal", seed=0)

    norms = layer.basis.detach().norm(dim=1)

    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_trop_dict_diagnostics_returns_finite_metrics() -> None:
    layer = TropDictLinear(4, 16, dict_size=12, dict_sparsity=4, seed=0)

    diag = layer.dictionary_diagnostics()

    for key in ("dict_norm_mean", "dict_norm_std", "dict_offdiag_max_abs", "dict_offdiag_mean_sq"):
        assert key in diag
        assert torch.isfinite(torch.tensor(diag[key]))


def test_trop_dict_rejects_oversized_sparsity() -> None:
    with pytest.raises(ValueError, match="dict_sparsity"):
        TropDictLinear(4, 4, dict_size=2, dict_sparsity=3)


def test_trop_dict_rejects_invalid_init() -> None:
    with pytest.raises(ValueError, match="unknown dict init"):
        TropDictLinear(4, 4, dict_size=4, dict_sparsity=2, dict_init="bogus")  # type: ignore[arg-type]


def test_trop_dict_rejects_non_torch_backend() -> None:
    with pytest.raises(ValueError, match="backend='torch'"):
        TropDictLinear(5, 3, backend="tilelang")


def test_trop_dict_sketch_route_shapes_for_2d_and_3d_inputs() -> None:
    layer = TropDictLinear(
        12,
        4,
        heads=4,
        cells=3,
        route_source="sketch",
        route_dim=6,
        dict_size=8,
        dict_sparsity=2,
        seed=0,
    )

    y2 = layer(torch.randn(7, 12))
    y3 = layer(torch.randn(7, 2, 12))

    assert y2.shape == (7, 1, 4)
    assert y3.shape == (7, 2, 4)
    assert layer._last_indices is not None
    assert layer._last_indices.shape == (7, 2, 4)


def test_trop_dict_sketch_route_train_propagates_grads() -> None:
    layer = TropDictLinear(
        16,
        5,
        heads=6,
        cells=4,
        route_source="sketch",
        route_dim=8,
        dict_size=12,
        dict_sparsity=3,
        seed=1,
    )
    x = torch.randn(4, 3, 16, requires_grad=True)

    loss = layer(x).square().mean()
    loss.backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    for param in (layer.sites, layer.lifting, layer.coeff, layer.basis, layer.bias):
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()
        assert float(param.grad.abs().sum()) > 0.0


def test_trop_dict_sketch_route_eval_uses_only_winner_code() -> None:
    layer = TropDictLinear(
        2,
        4,
        heads=1,
        cells=3,
        route_source="sketch",
        route_dim=2,
        dict_size=4,
        dict_sparsity=1,
        seed=0,
        use_output_scaling=False,
    )
    with torch.no_grad():
        layer.input_buckets.copy_(torch.tensor([0, 1]))
        layer.input_signs.fill_(1.0)
        layer.input_scale.fill_(1.0)
        layer.sites[0] = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
        layer.lifting.zero_()
        layer.coeff[0, :, 0] = torch.tensor([1.0, 1.0, 1.0])
        layer.basis.zero_()
        layer.basis[0, 0] = 10.0
        layer.basis[1, 1] = 20.0
        layer.basis[2, 2] = 30.0
        layer.support[0, 0, 0] = 0
        layer.support[0, 1, 0] = 1
        layer.support[0, 2, 0] = 2
        layer.bias.zero_()

    layer.eval()

    out_pos = layer(torch.tensor([[2.0, 0.0]])).squeeze()
    out_neg = layer(torch.tensor([[-2.0, 0.0]])).squeeze()
    assert torch.allclose(out_pos, torch.tensor([10.0, 0.0, 0.0, 0.0]))
    assert torch.allclose(out_neg, torch.tensor([0.0, 20.0, 0.0, 0.0]))


def test_trop_dict_sketch_default_route_dim_matches_out_features() -> None:
    layer = TropDictLinear(64, 16, route_source="sketch", dict_size=16, dict_sparsity=2, seed=0)

    assert layer.route_dim == 16
    assert hasattr(layer, "input_buckets")
    assert layer.sites.shape[-1] == 16


def test_trop_dict_rejects_unknown_route_source() -> None:
    with pytest.raises(ValueError, match="route_source"):
        TropDictLinear(4, 4, route_source="bogus")  # type: ignore[arg-type]


def test_trop_dict_route_residual_adds_sketched_input() -> None:
    layer = TropDictLinear(
        4,
        4,
        heads=2,
        cells=3,
        route_source="sketch",
        route_dim=4,
        dict_size=4,
        dict_sparsity=2,
        seed=0,
        use_route_residual=True,
        use_output_scaling=False,
    )
    with torch.no_grad():
        layer.coeff.zero_()
        layer.bias.zero_()

    layer.eval()
    x = torch.randn(3, 4)
    expected_residual = layer._project_input(x.unsqueeze(1), torch.float32).squeeze(1)
    out = layer(x).squeeze(1)

    assert torch.allclose(out, expected_residual, atol=1e-6)


def test_trop_dict_route_residual_requires_sketch() -> None:
    with pytest.raises(ValueError, match="route_source='sketch'"):
        TropDictLinear(4, 4, route_source="anchors", use_route_residual=True)


def test_trop_dict_route_residual_requires_matching_route_dim() -> None:
    with pytest.raises(ValueError, match="use_route_residual"):
        TropDictLinear(4, 8, route_source="sketch", route_dim=6, use_route_residual=True)
