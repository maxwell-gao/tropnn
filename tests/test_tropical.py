from __future__ import annotations

import pytest
import torch
from tropnn import PairwiseLinear, TropFanLinear, TropLinear
from tropnn.backend import has_pairwise_zig, has_tilelang, has_triton, has_trop_fan_zig, has_tropical_zig, trop_scores, trop_scores_reference
from tropnn.layers.surrogate import surrogate_gradient


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


@pytest.mark.skipif(
    not torch.cuda.is_available() or not has_tilelang() or not has_triton(),
    reason="requires CUDA TileLang and Triton",
)
def test_trop_linear_exact_fused_train_cuda_gradients_match_reference() -> None:
    torch.manual_seed(0)
    base = TropLinear(6, 4, heads=3, cells=4, code_dim=8, backend="torch", seed=3).cuda().train()
    fused = TropLinear(6, 4, heads=3, cells=4, code_dim=8, backend="tilelang", seed=3, exact_fused="train").cuda().train()
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
    x = torch.randn(5, 2, 6, device="cuda", requires_grad=True)
    x_fused = x.detach().clone().requires_grad_(True)

    loss = base(x).square().mean()
    loss.backward()
    fused_loss = fused(x_fused).square().mean()
    fused_loss.backward()

    assert torch.allclose(base(x.detach()), fused(x.detach()), atol=1e-5)
    assert torch.allclose(x.grad, x_fused.grad, atol=2e-4)
    assert torch.allclose(base.proj.weight.grad, fused.proj.weight.grad, atol=2e-4)
    assert torch.allclose(base.router_weight.grad, fused.router_weight.grad, atol=2e-4)
    assert torch.allclose(base.router_bias.grad, fused.router_bias.grad, atol=2e-4)
    assert torch.allclose(base.code.grad, fused.code.grad, atol=2e-4)
    assert torch.allclose(base.output_proj.weight.grad, fused.output_proj.weight.grad, atol=2e-4)
    assert torch.allclose(base.output_proj.bias.grad, fused.output_proj.bias.grad, atol=2e-4)


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


def test_zig_score_backend_is_fused_only() -> None:
    z = torch.randn(2, 3, 5)
    router_weight = torch.randn(7, 4, 5)
    router_bias = torch.randn(7, 4)

    with pytest.raises(RuntimeError, match="fused TropLinear inference backend"):
        trop_scores(z, router_weight, router_bias, backend="zig")


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


def test_fast_sigmoid_odd_surrogate_has_lut_direction() -> None:
    u = torch.tensor([-2.0, 0.0, 2.0])
    grad = surrogate_gradient(u, "fast_sigmoid_odd")

    assert grad[0] > 0
    assert grad[1] == 0
    assert grad[2] < 0


@pytest.mark.skipif(not has_pairwise_zig(), reason="requires ziglang or TROPNN_ZIG")
def test_pairwise_zig_forward_matches_torch_f32() -> None:
    torch.manual_seed(0)
    base = PairwiseLinear(8, 5, tables=3, comparisons=3, backend="torch", seed=1).eval()
    fast = PairwiseLinear(8, 5, tables=3, comparisons=3, backend="zig", seed=1, cpu_lut_dtype="f32").eval()
    with torch.no_grad():
        fast.anchors.copy_(base.anchors)
        fast.thresholds.copy_(base.thresholds)
        fast.lut.copy_(base.lut)
        x = torch.randn(4, 2, 8)

        assert torch.allclose(base(x), fast(x), atol=1e-6)


@pytest.mark.skipif(not has_pairwise_zig(), reason="requires ziglang or TROPNN_ZIG")
def test_pairwise_zig_forward_matches_torch_f16_lut() -> None:
    torch.manual_seed(0)
    base = PairwiseLinear(8, 5, tables=4, comparisons=3, backend="torch", seed=2).eval()
    fast = PairwiseLinear(8, 5, tables=4, comparisons=3, backend="zig", seed=2, cpu_lut_dtype="f16").eval()
    with torch.no_grad():
        fast.anchors.copy_(base.anchors)
        fast.thresholds.copy_(base.thresholds)
        fast.lut.copy_(base.lut)
        x = torch.randn(7, 8)

        assert torch.allclose(base(x), fast(x), atol=5e-3)


def test_pairwise_zig_backend_is_inference_only() -> None:
    layer = PairwiseLinear(8, 5, tables=3, comparisons=3, backend="zig", seed=1)

    with pytest.raises(RuntimeError, match="inference-only"):
        layer(torch.randn(4, 8))


@pytest.mark.skipif(not has_tropical_zig(), reason="requires ziglang or TROPNN_ZIG")
def test_trop_linear_zig_forward_matches_torch_f32() -> None:
    torch.manual_seed(0)
    base = TropLinear(8, 5, heads=3, cells=4, code_dim=6, backend="torch", seed=1).eval()
    fast = TropLinear(8, 5, heads=3, cells=4, code_dim=6, backend="zig", seed=1, cpu_param_dtype="f32").eval()
    with torch.no_grad():
        fast.proj.weight.copy_(base.proj.weight)
        fast.router_weight.copy_(base.router_weight)
        fast.router_bias.copy_(base.router_bias)
        fast.code.copy_(base.code)
        fast.output_proj.weight.copy_(base.output_proj.weight)
        fast.output_proj.bias.copy_(base.output_proj.bias)
        x = torch.randn(4, 2, 8)

        assert torch.allclose(base(x), fast(x), atol=1e-6)
        assert fast._last_indices is not None
        assert fast._last_indices.shape == (4, 2, 0)


@pytest.mark.skipif(not has_tropical_zig(), reason="requires ziglang or TROPNN_ZIG")
def test_trop_linear_zig_forward_matches_torch_f16_params() -> None:
    torch.manual_seed(0)
    base = TropLinear(8, 5, heads=4, cells=3, code_dim=6, backend="torch", seed=2).eval()
    fast = TropLinear(8, 5, heads=4, cells=3, code_dim=6, backend="zig", seed=2, cpu_param_dtype="f16").eval()
    with torch.no_grad():
        base.router_weight.zero_()
        base.router_bias.copy_(torch.tensor([[0.0, 8.0, 16.0]]).expand(4, 3))
        fast.proj.weight.copy_(base.proj.weight)
        fast.router_weight.copy_(base.router_weight)
        fast.router_bias.copy_(base.router_bias)
        fast.code.copy_(base.code)
        fast.output_proj.weight.copy_(base.output_proj.weight)
        fast.output_proj.bias.copy_(base.output_proj.bias)
        x = torch.randn(7, 8)

        assert torch.allclose(base(x), fast(x), atol=5e-3)


def test_trop_linear_zig_backend_is_inference_only() -> None:
    layer = TropLinear(8, 5, heads=3, cells=4, code_dim=6, backend="zig", seed=1)

    with pytest.raises(RuntimeError, match="inference-only"):
        layer(torch.randn(4, 8))


@pytest.mark.skipif(not has_trop_fan_zig(), reason="requires ziglang or TROPNN_ZIG")
@pytest.mark.parametrize(
    ("fan_value_mode", "cpu_param_dtype", "atol"),
    [
        ("site", "f32", 1e-6),
        ("basis", "f32", 1e-6),
        ("site", "f16", 5e-3),
        ("basis", "f16", 5e-3),
    ],
)
def test_trop_fan_zig_forward_matches_torch(fan_value_mode: str, cpu_param_dtype: str, atol: float) -> None:
    torch.manual_seed(0)
    base = TropFanLinear(8, 5, heads=4, cells=3, code_dim=6, backend="torch", fan_value_mode=fan_value_mode, fan_basis_rank=3, seed=3).eval()
    fast = TropFanLinear(
        8,
        5,
        heads=4,
        cells=3,
        code_dim=6,
        backend="zig",
        fan_value_mode=fan_value_mode,
        fan_basis_rank=3,
        seed=3,
        cpu_param_dtype=cpu_param_dtype,  # type: ignore[arg-type]
    ).eval()
    with torch.no_grad():
        base.lifting.copy_(torch.tensor([[0.0, 8.0, 16.0]]).expand(4, 3))
        fast.proj.weight.copy_(base.proj.weight)
        fast.sites.copy_(base.sites)
        fast.lifting.copy_(base.lifting)
        if fan_value_mode == "site":
            assert base.value_scale is not None and fast.value_scale is not None
            fast.value_scale.copy_(base.value_scale)
        else:
            assert base.value_coeff is not None and fast.value_coeff is not None
            assert base.value_basis is not None and fast.value_basis is not None
            fast.value_coeff.copy_(base.value_coeff)
            fast.value_basis.copy_(base.value_basis)
        fast.output_proj.weight.copy_(base.output_proj.weight)
        fast.output_proj.bias.copy_(base.output_proj.bias)
        x = torch.randn(7, 2, 8)

        assert torch.allclose(base(x), fast(x), atol=atol)
        assert fast._last_indices is not None
        assert fast._last_indices.shape == (7, 2, 0)


def test_trop_fan_zig_backend_is_inference_only() -> None:
    layer = TropFanLinear(8, 5, heads=3, cells=4, code_dim=6, backend="zig", fan_value_mode="basis", seed=1)

    with pytest.raises(RuntimeError, match="inference-only"):
        layer(torch.randn(4, 8))


@pytest.mark.skipif(not torch.cuda.is_available() or not has_tilelang(), reason="requires CUDA TileLang")
@pytest.mark.parametrize("use_min_margin_ste", [True, False])
def test_pairwise_tilelang_forward_backward_matches_torch(use_min_margin_ste: bool) -> None:
    torch.manual_seed(0)
    base = PairwiseLinear(8, 5, tables=3, comparisons=3, backend="torch", seed=1, use_min_margin_ste=use_min_margin_ste).cuda()
    fast = PairwiseLinear(8, 5, tables=3, comparisons=3, backend="tilelang", seed=1, use_min_margin_ste=use_min_margin_ste).cuda()
    with torch.no_grad():
        fast.anchors.copy_(base.anchors)
        fast.thresholds.copy_(base.thresholds)
        fast.lut.copy_(base.lut)
    x = torch.randn(4, 8, device="cuda", requires_grad=True)
    x_fast = x.detach().clone().requires_grad_(True)

    loss = base(x).square().mean()
    loss.backward()
    fast_loss = fast(x_fast).square().mean()
    fast_loss.backward()

    assert torch.allclose(base(x.detach()), fast(x.detach()), atol=1e-6)
    assert torch.allclose(x.grad, x_fast.grad, atol=1e-6)
    assert torch.allclose(base.thresholds.grad, fast.thresholds.grad, atol=1e-6)
    assert torch.allclose(base.lut.grad, fast.lut.grad, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available() or not has_tilelang(), reason="requires CUDA TileLang")
def test_trop_linear_tilelang_forward_backward_matches_torch() -> None:
    torch.manual_seed(0)
    base = TropLinear(8, 5, heads=3, cells=4, code_dim=6, backend="torch", seed=1).cuda()
    fast = TropLinear(8, 5, heads=3, cells=4, code_dim=6, backend="tilelang", seed=1).cuda()
    with torch.no_grad():
        fast.proj.weight.copy_(base.proj.weight)
        fast.router_weight.copy_(base.router_weight)
        fast.router_bias.copy_(base.router_bias)
        fast.code.copy_(base.code)
        fast.output_proj.weight.copy_(base.output_proj.weight)
        fast.output_proj.bias.copy_(base.output_proj.bias)
    x = torch.randn(4, 8, device="cuda", requires_grad=True)
    x_fast = x.detach().clone().requires_grad_(True)

    loss = base(x).square().mean()
    loss.backward()
    fast_loss = fast(x_fast).square().mean()
    fast_loss.backward()

    assert torch.allclose(base(x.detach()), fast(x.detach()), atol=1e-6)
    assert torch.allclose(x.grad, x_fast.grad, atol=1e-6)
    assert torch.allclose(base.router_weight.grad, fast.router_weight.grad, atol=1e-6)
    assert torch.allclose(base.router_bias.grad, fast.router_bias.grad, atol=1e-6)
    assert torch.allclose(base.code.grad, fast.code.grad, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available() or not has_tilelang(), reason="requires CUDA TileLang")
def test_trop_linear_tilelang_parallel_route_matches_torch() -> None:
    torch.manual_seed(0)
    base = TropLinear(16, 7, heads=4, cells=4, code_dim=64, backend="torch", seed=2).cuda()
    fast = TropLinear(16, 7, heads=4, cells=4, code_dim=64, backend="tilelang", seed=2).cuda()
    with torch.no_grad():
        fast.proj.weight.copy_(base.proj.weight)
        fast.router_weight.copy_(base.router_weight)
        fast.router_bias.copy_(base.router_bias)
        fast.code.copy_(base.code)
        fast.output_proj.weight.copy_(base.output_proj.weight)
        fast.output_proj.bias.copy_(base.output_proj.bias)
    x = torch.randn(5, 2, 16, device="cuda", requires_grad=True)
    x_fast = x.detach().clone().requires_grad_(True)

    loss = base(x).square().mean()
    loss.backward()
    fast_loss = fast(x_fast).square().mean()
    fast_loss.backward()

    assert torch.allclose(base(x.detach()), fast(x.detach()), atol=1e-5)
    assert torch.allclose(x.grad, x_fast.grad, atol=1e-5)
    assert torch.allclose(base.proj.weight.grad, fast.proj.weight.grad, atol=1e-5)
    assert torch.allclose(base.router_weight.grad, fast.router_weight.grad, atol=1e-5)
    assert torch.allclose(base.router_bias.grad, fast.router_bias.grad, atol=1e-5)
    assert torch.allclose(base.code.grad, fast.code.grad, atol=1e-5)
    assert torch.allclose(base.output_proj.weight.grad, fast.output_proj.weight.grad, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available() or not has_tilelang(), reason="requires CUDA TileLang")
def test_trop_linear_tilelang_hybrid_score_route_matches_torch() -> None:
    torch.manual_seed(0)
    base = TropLinear(16, 7, heads=4, cells=4, code_dim=128, backend="torch", seed=3).cuda()
    fast = TropLinear(16, 7, heads=4, cells=4, code_dim=128, backend="tilelang", seed=3).cuda()
    with torch.no_grad():
        fast.proj.weight.copy_(base.proj.weight)
        fast.router_weight.copy_(base.router_weight)
        fast.router_bias.copy_(base.router_bias)
        fast.code.copy_(base.code)
        fast.output_proj.weight.copy_(base.output_proj.weight)
        fast.output_proj.bias.copy_(base.output_proj.bias)
    x = torch.randn(3, 2, 16, device="cuda", requires_grad=True)
    x_fast = x.detach().clone().requires_grad_(True)

    loss = base(x).square().mean()
    loss.backward()
    fast_loss = fast(x_fast).square().mean()
    fast_loss.backward()

    assert torch.allclose(base(x.detach()), fast(x.detach()), atol=1e-5)
    assert torch.allclose(x.grad, x_fast.grad, atol=1e-5)
    assert torch.allclose(base.proj.weight.grad, fast.proj.weight.grad, atol=1e-5)
    assert torch.allclose(base.router_weight.grad, fast.router_weight.grad, atol=1e-5)
    assert torch.allclose(base.router_bias.grad, fast.router_bias.grad, atol=1e-5)
    assert torch.allclose(base.code.grad, fast.code.grad, atol=1e-5)
    assert torch.allclose(base.output_proj.weight.grad, fast.output_proj.weight.grad, atol=1e-5)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not has_tilelang() or not has_triton(),
    reason="requires CUDA TileLang and Triton",
)
def test_trop_linear_tilelang_triton_eval_route_matches_torch() -> None:
    torch.manual_seed(0)
    base = TropLinear(16, 7, heads=4, cells=4, code_dim=128, backend="torch", seed=4).cuda().eval()
    fast = TropLinear(16, 7, heads=4, cells=4, code_dim=128, backend="tilelang", seed=4).cuda().eval()
    with torch.no_grad():
        fast.proj.weight.copy_(base.proj.weight)
        fast.router_weight.copy_(base.router_weight)
        fast.router_bias.copy_(base.router_bias)
        fast.code.copy_(base.code)
        fast.output_proj.weight.copy_(base.output_proj.weight)
        fast.output_proj.bias.copy_(base.output_proj.bias)
        x = torch.randn(3, 2, 16, device="cuda")

        base_out = base(x)
        fast_out = fast(x)

    assert torch.allclose(base_out, fast_out, atol=1e-5)
    assert base._last_indices is not None and fast._last_indices is not None
    assert base._last_margins is not None and fast._last_margins is not None
    assert torch.equal(base._last_indices, fast._last_indices)
    assert torch.allclose(base._last_margins, fast._last_margins, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available() or not has_tilelang(), reason="requires CUDA TileLang")
def test_trop_fan_basis_tilelang_forward_backward_matches_torch() -> None:
    torch.manual_seed(0)
    base = TropFanLinear(8, 5, heads=3, cells=4, code_dim=6, backend="torch", fan_value_mode="basis", fan_basis_rank=3, seed=1).cuda()
    fast = TropFanLinear(8, 5, heads=3, cells=4, code_dim=6, backend="tilelang", fan_value_mode="basis", fan_basis_rank=3, seed=1).cuda()
    assert base.value_coeff is not None and base.value_basis is not None
    assert fast.value_coeff is not None and fast.value_basis is not None
    with torch.no_grad():
        fast.proj.weight.copy_(base.proj.weight)
        fast.sites.copy_(base.sites)
        fast.lifting.copy_(base.lifting)
        fast.value_coeff.copy_(base.value_coeff)
        fast.value_basis.copy_(base.value_basis)
        fast.output_proj.weight.copy_(base.output_proj.weight)
        fast.output_proj.bias.copy_(base.output_proj.bias)
    x = torch.randn(4, 8, device="cuda", requires_grad=True)
    x_fast = x.detach().clone().requires_grad_(True)

    loss = base(x).square().mean()
    loss.backward()
    fast_loss = fast(x_fast).square().mean()
    fast_loss.backward()

    assert torch.allclose(base(x.detach()), fast(x.detach()), atol=1e-6)
    assert torch.allclose(x.grad, x_fast.grad, atol=1e-5)
    assert torch.allclose(base.proj.weight.grad, fast.proj.weight.grad, atol=1e-5)
    assert torch.allclose(base.sites.grad, fast.sites.grad, atol=1e-5)
    assert torch.allclose(base.lifting.grad, fast.lifting.grad, atol=1e-5)
    assert torch.allclose(base.value_coeff.grad, fast.value_coeff.grad, atol=1e-5)
    assert torch.allclose(base.value_basis.grad, fast.value_basis.grad, atol=1e-5)
    assert torch.allclose(base.output_proj.weight.grad, fast.output_proj.weight.grad, atol=1e-6)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not has_tilelang() or not has_triton(),
    reason="requires CUDA TileLang and Triton",
)
def test_trop_fan_basis_tilelang_triton_eval_route_matches_torch() -> None:
    torch.manual_seed(0)
    base = (
        TropFanLinear(
            16,
            7,
            heads=4,
            cells=4,
            code_dim=128,
            backend="torch",
            fan_value_mode="basis",
            fan_basis_rank=8,
            seed=5,
        )
        .cuda()
        .eval()
    )
    fast = (
        TropFanLinear(
            16,
            7,
            heads=4,
            cells=4,
            code_dim=128,
            backend="tilelang",
            fan_value_mode="basis",
            fan_basis_rank=8,
            seed=5,
        )
        .cuda()
        .eval()
    )
    assert base.value_coeff is not None and base.value_basis is not None
    assert fast.value_coeff is not None and fast.value_basis is not None
    with torch.no_grad():
        fast.proj.weight.copy_(base.proj.weight)
        fast.sites.copy_(base.sites)
        fast.lifting.copy_(base.lifting)
        fast.value_coeff.copy_(base.value_coeff)
        fast.value_basis.copy_(base.value_basis)
        fast.output_proj.weight.copy_(base.output_proj.weight)
        fast.output_proj.bias.copy_(base.output_proj.bias)
        x = torch.randn(3, 2, 16, device="cuda")

        base_out = base(x)
        fast_out = fast(x)

    assert torch.allclose(base_out, fast_out, atol=1e-5)
    assert base._last_indices is not None and fast._last_indices is not None
    assert base._last_margins is not None and fast._last_margins is not None
    assert torch.equal(base._last_indices, fast._last_indices)
    assert torch.allclose(base._last_margins, fast._last_margins, atol=1e-5)


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
