from __future__ import annotations

import torch
from tropnn import TropLinear


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


def test_trop_linear_exact_fused_eval_matches_reference() -> None:
    torch.manual_seed(0)
    base = TropLinear(6, 4, heads=3, cells=4, code_dim=5, backend="torch", seed=1).eval()
    fused = TropLinear(6, 4, heads=3, cells=4, code_dim=5, backend="torch", seed=1, exact_fused="eval").eval()
    with torch.no_grad():
        fused.proj.weight.copy_(base.proj.weight)
        fused.router_weight.copy_(base.router_weight)
        fused.router_bias.copy_(base.router_bias)
        fused.code.copy_(base.code)
        fused.output_proj.weight.copy_(base.output_proj.weight)
        fused.output_proj.bias.copy_(base.output_proj.bias)
        # Bias separation keeps top-2 stable under mathematically equivalent reassociation.
        bias = torch.tensor([-3.0, -1.0, 1.0, 3.0], device=base.router_bias.device).expand_as(base.router_bias)
        base.router_bias.copy_(bias)
        fused.router_bias.copy_(bias)
        x = torch.randn(5, 2, 6)

        base_out = base(x)
        fused_out = fused(x)

    assert torch.allclose(base_out, fused_out, atol=1e-5)
    assert base._last_indices is not None and fused._last_indices is not None
    assert torch.equal(base._last_indices, fused._last_indices)
    assert base._last_margins is not None and fused._last_margins is not None
    assert torch.allclose(base._last_margins, fused._last_margins, atol=1e-5)


def test_trop_linear_exact_fused_train_gradients_match_reference() -> None:
    torch.manual_seed(0)
    base = TropLinear(6, 4, heads=3, cells=4, code_dim=5, backend="torch", seed=2).train()
    fused = TropLinear(6, 4, heads=3, cells=4, code_dim=5, backend="torch", seed=2, exact_fused="train").train()
    with torch.no_grad():
        fused.proj.weight.copy_(base.proj.weight)
        fused.router_weight.copy_(base.router_weight)
        fused.router_bias.copy_(base.router_bias)
        fused.code.copy_(base.code)
        fused.output_proj.weight.copy_(base.output_proj.weight)
        fused.output_proj.bias.copy_(base.output_proj.bias)
        bias = torch.tensor([-3.0, -1.0, 1.0, 3.0], device=base.router_bias.device).expand_as(base.router_bias)
        base.router_bias.copy_(bias)
        fused.router_bias.copy_(bias)
    x = torch.randn(5, 2, 6, requires_grad=True)
    x_fused = x.detach().clone().requires_grad_(True)

    loss = base(x).square().mean()
    loss.backward()
    fused_loss = fused(x_fused).square().mean()
    fused_loss.backward()

    assert torch.allclose(base(x.detach()), fused(x.detach()), atol=1e-5)
    assert torch.allclose(x.grad, x_fused.grad, atol=1e-5)
    assert torch.allclose(base.proj.weight.grad, fused.proj.weight.grad, atol=1e-5)
    assert torch.allclose(base.router_weight.grad, fused.router_weight.grad, atol=1e-5)
    assert torch.allclose(base.router_bias.grad, fused.router_bias.grad, atol=1e-5)
    assert torch.allclose(base.code.grad, fused.code.grad, atol=1e-5)
    assert torch.allclose(base.output_proj.weight.grad, fused.output_proj.weight.grad, atol=1e-5)
    assert torch.allclose(base.output_proj.bias.grad, fused.output_proj.bias.grad, atol=1e-5)


def test_trop_linear_exact_fused_train_cells8_no_debug_gradients_match_reference() -> None:
    torch.manual_seed(0)
    base = TropLinear(6, 4, heads=5, cells=8, code_dim=5, backend="torch", seed=7).train()
    fused = TropLinear(6, 4, heads=5, cells=8, code_dim=5, backend="torch", seed=7, exact_fused="train", cache_route_debug=False).train()
    with torch.no_grad():
        fused.proj.weight.copy_(base.proj.weight)
        fused.router_weight.copy_(base.router_weight)
        fused.router_bias.copy_(base.router_bias)
        fused.code.copy_(base.code)
        fused.output_proj.weight.copy_(base.output_proj.weight)
        fused.output_proj.bias.copy_(base.output_proj.bias)
        bias = torch.linspace(-4.0, 3.0, steps=8, device=base.router_bias.device).expand_as(base.router_bias)
        base.router_bias.copy_(bias)
        fused.router_bias.copy_(bias)
    x = torch.randn(5, 2, 6, requires_grad=True)
    x_fused = x.detach().clone().requires_grad_(True)

    loss = base(x).square().mean()
    loss.backward()
    fused_loss = fused(x_fused).square().mean()
    fused_loss.backward()

    assert torch.allclose(base(x.detach()), fused(x.detach()), atol=1e-5)
    assert torch.allclose(x.grad, x_fused.grad, atol=1e-5)
    assert torch.allclose(base.proj.weight.grad, fused.proj.weight.grad, atol=1e-5)
    assert torch.allclose(base.router_weight.grad, fused.router_weight.grad, atol=1e-5)
    assert torch.allclose(base.router_bias.grad, fused.router_bias.grad, atol=1e-5)
    assert torch.allclose(base.code.grad, fused.code.grad, atol=1e-5)
    assert torch.allclose(base.output_proj.weight.grad, fused.output_proj.weight.grad, atol=1e-5)
    assert torch.allclose(base.output_proj.bias.grad, fused.output_proj.bias.grad, atol=1e-5)
    assert fused._last_indices is None
    assert fused._last_margins is None
