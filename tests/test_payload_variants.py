from __future__ import annotations

import pytest
import torch
from tropnn import TropDeltaLinear, TropLUTLinear, TropSharedLowRankLinear

VARIANT_CLASSES = (TropLUTLinear, TropDeltaLinear, TropSharedLowRankLinear)


@pytest.mark.parametrize("layer_cls", VARIANT_CLASSES)
def test_payload_variant_shapes_for_2d_and_3d_inputs(layer_cls: type[torch.nn.Module]) -> None:
    layer = layer_cls(5, 3, tables=2, groups=2, cells=3, rank=4, seed=0)

    y2 = layer(torch.randn(7, 5))
    y3 = layer(torch.randn(7, 2, 5))

    assert y2.shape == (7, 1, 3)
    assert y3.shape == (7, 2, 3)
    assert layer._last_indices is not None
    assert layer._last_margins is not None


def test_trop_lut_eval_uses_only_winner_payload() -> None:
    layer = TropLUTLinear(1, 1, tables=1, groups=1, cells=3, rank=1, seed=0, use_output_scaling=False)
    with torch.no_grad():
        layer.proj.weight.fill_(1.0)
        layer.router_weight[0, 0, :, 0] = torch.tensor([1.0, 0.0, -1.0])
        layer.router_bias.zero_()
        layer.lut[0, 0, :, 0] = torch.tensor([10.0, 20.0, 30.0])

    layer.eval()

    assert torch.allclose(layer(torch.tensor([[2.0]])).squeeze(), torch.tensor(10.0))
    assert torch.allclose(layer(torch.tensor([[-2.0]])).squeeze(), torch.tensor(30.0))


def test_trop_lut_train_uses_minface_interpolation() -> None:
    layer = TropLUTLinear(1, 1, tables=1, groups=1, cells=3, rank=1, seed=0, use_output_scaling=False)
    with torch.no_grad():
        layer.proj.weight.fill_(1.0)
        layer.router_weight[0, 0, :, 0] = torch.tensor([1.0, 0.9, -1.0])
        layer.router_bias.zero_()
        layer.lut[0, 0, :, 0] = torch.tensor([10.0, 20.0, 30.0])

    layer.train()
    y = layer(torch.tensor([[1.0]])).squeeze()
    expected = torch.tensor(10.0 + (0.5 / 1.1) * (20.0 - 10.0))

    assert torch.allclose(y, expected, atol=1e-6)


@pytest.mark.parametrize("layer_cls", VARIANT_CLASSES)
def test_payload_variant_train_has_router_and_payload_gradients(layer_cls: type[torch.nn.Module]) -> None:
    layer = layer_cls(6, 4, tables=3, groups=2, cells=4, rank=5, seed=0)
    x = torch.randn(8, 2, 6, requires_grad=True)

    y = layer(x)
    loss = y.square().mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert layer.router_weight.grad is not None
    assert torch.isfinite(layer.router_weight.grad).all()
    assert float(layer.router_weight.grad.abs().sum()) > 0.0
    payload_grads = [
        param.grad.abs().sum()
        for name, param in layer.named_parameters()
        if name not in {"proj.weight", "router_weight", "router_bias"} and param.grad is not None
    ]
    assert payload_grads
    assert float(sum(payload_grads)) > 0.0


def test_trop_shared_lowrank_eval_uses_selected_scalar_gated_vector() -> None:
    layer = TropSharedLowRankLinear(1, 1, tables=1, groups=1, cells=2, rank=1, seed=0, use_output_scaling=False)
    with torch.no_grad():
        layer.proj.weight.fill_(1.0)
        layer.router_weight[0, 0, :, 0] = torch.tensor([1.0, -1.0])
        layer.router_bias.zero_()
        layer.gate_weight[0, 0, :, 0] = torch.tensor([2.0, 5.0])
        layer.gate_bias[0, 0, :] = torch.tensor([1.0, 7.0])
        layer.payload_vector[0, 0, :, 0] = torch.tensor([3.0, 11.0])
        layer.payload_bias[0, 0, :, 0] = torch.tensor([13.0, 17.0])

    layer.eval()
    y = layer(torch.tensor([[2.0]])).squeeze()

    assert torch.allclose(y, torch.tensor((2.0 * 2.0 + 1.0) * 3.0 + 13.0))
