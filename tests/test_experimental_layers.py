from __future__ import annotations

import pytest
import torch
from tropnn import TropFanLinear, TropFanZeroDenseLinear, TropZeroDenseLinear


def test_trop_zero_dense_shapes_for_2d_and_3d_inputs() -> None:
    layer = TropZeroDenseLinear(5, 3, heads=4, cells=3, route_terms=2, seed=0)

    y2 = layer(torch.randn(7, 5))
    y3 = layer(torch.randn(7, 2, 5))

    assert y2.shape == (7, 1, 3)
    assert y3.shape == (7, 2, 3)
    assert layer._last_indices is not None
    assert layer._last_indices.shape == (7, 2, 4)
    assert layer._last_margins is not None


def test_trop_zero_dense_eval_uses_only_winner_code() -> None:
    layer = TropZeroDenseLinear(2, 1, heads=1, cells=3, route_terms=1, seed=0, use_output_scaling=False)
    with torch.no_grad():
        layer.anchors[0, :, 0] = torch.tensor([0, 0, 1])
        layer.router_weight[0, :, 0] = torch.tensor([1.0, -1.0, 1.0])
        layer.router_bias.zero_()
        layer.code[0, :, 0] = torch.tensor([10.0, 20.0, 30.0])
        layer.bias.zero_()

    layer.eval()

    assert torch.allclose(layer(torch.tensor([[2.0, 0.0]])).squeeze(), torch.tensor(10.0))
    assert torch.allclose(layer(torch.tensor([[-2.0, 0.0]])).squeeze(), torch.tensor(20.0))


def test_trop_zero_dense_train_has_router_code_and_bias_gradients() -> None:
    layer = TropZeroDenseLinear(6, 4, heads=6, cells=4, route_terms=3, seed=0)
    x = torch.randn(8, 2, 6, requires_grad=True)

    loss = layer(x).square().mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for param in (layer.router_weight, layer.code, layer.bias):
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()
        assert float(param.grad.abs().sum()) > 0.0


def test_trop_zero_dense_rejects_non_torch_backend() -> None:
    with pytest.raises(ValueError, match="backend='torch'"):
        TropZeroDenseLinear(5, 3, backend="tilelang")


def test_trop_fan_zero_dense_shapes_for_2d_and_3d_inputs() -> None:
    layer = TropFanZeroDenseLinear(5, 3, heads=4, cells=3, code_dim=4, seed=0)

    y2 = layer(torch.randn(7, 5))
    y3 = layer(torch.randn(7, 2, 5))

    assert y2.shape == (7, 1, 3)
    assert y3.shape == (7, 2, 3)
    assert layer._last_indices is not None
    assert layer._last_indices.shape == (7, 2, 4)
    assert layer._last_margins is not None


def test_trop_fan_zero_dense_eval_uses_site_generated_value() -> None:
    layer = TropFanZeroDenseLinear(2, 2, heads=1, cells=3, code_dim=2, seed=0, use_output_scaling=False)
    with torch.no_grad():
        layer.input_buckets.copy_(torch.tensor([0, 1]))
        layer.input_signs.fill_(1.0)
        layer.input_scale.fill_(1.0)
        layer.value_anchors.copy_(torch.tensor([0, 1]))
        layer.value_signs.fill_(1.0)
        layer.sites[0] = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
        layer.lifting.zero_()
        assert layer.value_scale is not None
        layer.value_scale[0] = torch.tensor([10.0, 20.0, 30.0])
        layer.bias.zero_()

    layer.eval()

    assert torch.allclose(layer(torch.tensor([[2.0, 0.0]])).squeeze(), torch.tensor([10.0, 0.0]))
    assert torch.allclose(layer(torch.tensor([[-2.0, 0.0]])).squeeze(), torch.tensor([-20.0, 0.0]))


def test_trop_fan_zero_dense_train_has_site_value_and_bias_gradients() -> None:
    layer = TropFanZeroDenseLinear(6, 4, heads=6, cells=4, code_dim=5, seed=0)
    x = torch.randn(8, 2, 6, requires_grad=True)

    loss = layer(x).square().mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert layer.value_scale is not None
    for param in (layer.sites, layer.lifting, layer.value_scale, layer.bias):
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()
        assert float(param.grad.abs().sum()) > 0.0


def test_trop_fan_zero_dense_rejects_non_torch_backend() -> None:
    with pytest.raises(ValueError, match="backend='torch'"):
        TropFanZeroDenseLinear(5, 3, backend="tilelang")


def test_trop_fan_site_tilelang_backend_uses_torch_fallback() -> None:
    layer = TropFanLinear(6, 4, heads=3, cells=4, code_dim=5, backend="tilelang", fan_value_mode="site", seed=0)
    x = torch.randn(8, 2, 6, requires_grad=True)

    layer.train()
    y = layer(x)
    loss = y.square().mean()
    loss.backward()

    assert x.grad is not None
    assert layer.sites.grad is not None
    assert layer.value_scale is not None
    assert layer.value_scale.grad is not None
