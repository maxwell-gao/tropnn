from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from tropnn import TropFiLMLinear, TropLinear, TropZeroDenseLinear
from tropnn.backend import trop_scores, trop_scores_reference


def test_trop_linear_shapes_for_2d_and_3d_inputs() -> None:
    layer = TropLinear(5, 3, heads=4, cells=3, code_dim=4, seed=0)

    y2 = layer(torch.randn(7, 5))
    y3 = layer(torch.randn(7, 2, 5))

    assert y2.shape == (7, 1, 3)
    assert y3.shape == (7, 2, 3)
    assert layer._last_indices is not None
    assert layer._last_margins is not None


def test_trop_linear_eval_uses_only_winner_code() -> None:
    layer = TropLinear(1, 1, heads=1, cells=3, code_dim=1, seed=0, use_output_scaling=False)
    with torch.no_grad():
        layer.proj.weight.fill_(1.0)
        layer.router_weight[0, :, 0] = torch.tensor([1.0, 0.0, -1.0])
        layer.router_bias.zero_()
        layer.code[0, :, 0] = torch.tensor([10.0, 20.0, 30.0])
        layer.output_proj.weight.fill_(1.0)
        layer.output_proj.bias.zero_()

    layer.eval()

    assert torch.allclose(layer(torch.tensor([[2.0]])).squeeze(), torch.tensor(12.0))
    assert torch.allclose(layer(torch.tensor([[-2.0]])).squeeze(), torch.tensor(28.0))


def test_trop_linear_train_uses_minface_interpolation() -> None:
    layer = TropLinear(1, 1, heads=1, cells=3, code_dim=1, seed=0, use_output_scaling=False)
    with torch.no_grad():
        layer.proj.weight.fill_(1.0)
        layer.router_weight[0, :, 0] = torch.tensor([1.0, 0.9, -1.0])
        layer.router_bias.zero_()
        layer.code[0, :, 0] = torch.tensor([10.0, 20.0, 30.0])
        layer.output_proj.weight.fill_(1.0)
        layer.output_proj.bias.zero_()

    layer.train()
    y = layer(torch.tensor([[1.0]])).squeeze()
    expected_code = torch.tensor(10.0 + (0.5 / 1.1) * (20.0 - 10.0))

    assert torch.allclose(y, torch.tensor(1.0) + expected_code, atol=1e-6)


def test_trop_linear_train_has_router_code_and_output_gradients() -> None:
    layer = TropLinear(6, 4, heads=6, cells=4, code_dim=5, seed=0)
    x = torch.randn(8, 2, 6, requires_grad=True)

    y = layer(x)
    loss = y.square().mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for param in (layer.router_weight, layer.code, layer.output_proj.weight):
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()
        assert float(param.grad.abs().sum()) > 0.0


def test_trop_film_linear_starts_as_trop_linear() -> None:
    base = TropLinear(5, 3, heads=4, cells=3, code_dim=4, seed=0)
    film = TropFiLMLinear(5, 3, heads=4, cells=3, code_dim=4, seed=0)
    x = torch.randn(7, 2, 5)

    assert torch.allclose(film.scale, torch.zeros_like(film.scale))
    assert torch.allclose(film(x), base(x), atol=1e-6)


def test_trop_film_linear_eval_uses_selected_scale_and_offset() -> None:
    layer = TropFiLMLinear(1, 1, heads=1, cells=3, code_dim=1, seed=0, use_output_scaling=False)
    with torch.no_grad():
        layer.proj.weight.fill_(1.0)
        layer.router_weight[0, :, 0] = torch.tensor([1.0, 0.0, -1.0])
        layer.router_bias.zero_()
        layer.code[0, :, 0] = torch.tensor([10.0, 20.0, 30.0])
        layer.scale[0, :, 0] = torch.tensor([0.5, 0.0, -0.5])
        layer.output_proj.weight.fill_(1.0)
        layer.output_proj.bias.zero_()

    layer.eval()

    assert torch.allclose(layer(torch.tensor([[2.0]])).squeeze(), torch.tensor(13.0))
    assert torch.allclose(layer(torch.tensor([[-2.0]])).squeeze(), torch.tensor(29.0))


def test_trop_film_linear_train_has_scale_gradients() -> None:
    layer = TropFiLMLinear(6, 4, heads=6, cells=4, code_dim=5, seed=0)
    x = torch.randn(8, 2, 6, requires_grad=True)

    y = layer(x)
    loss = y.square().mean()
    loss.backward()

    assert x.grad is not None
    assert layer.scale.grad is not None
    assert torch.isfinite(layer.scale.grad).all()
    assert float(layer.scale.grad.abs().sum()) > 0.0


def test_trop_zero_dense_shapes_for_2d_and_3d_inputs() -> None:
    layer = TropZeroDenseLinear(5, 3, heads=4, cells=3, route_terms=2, seed=0)

    y2 = layer(torch.randn(7, 5))
    y3 = layer(torch.randn(7, 2, 5))

    assert y2.shape == (7, 1, 3)
    assert y3.shape == (7, 2, 3)
    assert layer._last_indices is not None
    assert layer._last_margins is not None


def test_trop_zero_dense_has_no_linear_submodules() -> None:
    layer = TropZeroDenseLinear(5, 3, heads=4, cells=3, route_terms=2, seed=0)

    assert not any(isinstance(module, nn.Linear) for module in layer.modules())


def test_trop_zero_dense_eval_uses_only_winner_code() -> None:
    layer = TropZeroDenseLinear(2, 1, heads=1, cells=2, route_terms=1, seed=0, use_output_scaling=False)
    with torch.no_grad():
        layer.anchors[0, 0, 0] = 0
        layer.anchors[0, 1, 0] = 1
        layer.router_weight.fill_(1.0)
        layer.router_bias.zero_()
        layer.code[0, :, 0] = torch.tensor([10.0, 20.0])
        layer.bias.zero_()

    layer.eval()

    assert torch.allclose(layer(torch.tensor([[2.0, 1.0]])).squeeze(), torch.tensor(10.0))
    assert torch.allclose(layer(torch.tensor([[0.0, 3.0]])).squeeze(), torch.tensor(20.0))


def test_trop_zero_dense_train_has_router_code_and_input_gradients() -> None:
    layer = TropZeroDenseLinear(6, 4, heads=6, cells=4, route_terms=2, seed=0)
    x = torch.randn(8, 2, 6, requires_grad=True)

    y = layer(x)
    loss = y.square().mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for param in (layer.router_weight, layer.router_bias, layer.code):
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()
        assert float(param.grad.abs().sum()) > 0.0


def test_trop_zero_dense_rejects_empty_sparse_routes() -> None:
    with pytest.raises(ValueError, match="route_terms"):
        TropZeroDenseLinear(5, 3, route_terms=0)


def test_trop_scores_reference_uses_native_head_cell_shape() -> None:
    z = torch.randn(2, 3, 5)
    router_weight = torch.randn(7, 4, 5)
    router_bias = torch.randn(7, 4)

    scores = trop_scores_reference(z, router_weight, router_bias)

    assert scores.shape == (2, 3, 7, 4)
    assert torch.allclose(scores, torch.einsum("bsr,hkr->bshk", z, router_weight) + router_bias.view(1, 1, 7, 4))


def test_tilelang_score_backend_is_fused_only() -> None:
    z = torch.randn(2, 3, 5)
    router_weight = torch.randn(7, 4, 5)
    router_bias = torch.randn(7, 4)

    with pytest.raises(RuntimeError, match="fused TropLinear inference backend"):
        trop_scores(z, router_weight, router_bias, backend="tilelang")


def test_trop_linear_tilelang_backend_trains_with_torch_fallback() -> None:
    layer = TropLinear(6, 4, heads=3, cells=4, code_dim=5, backend="tilelang", seed=0)
    x = torch.randn(8, 2, 6, requires_grad=True)

    layer.train()
    y = layer(x)
    loss = y.square().mean()
    loss.backward()

    assert x.grad is not None
    assert layer.router_weight.grad is not None
    assert layer.code.grad is not None
