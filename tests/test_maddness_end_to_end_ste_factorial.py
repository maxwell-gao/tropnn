import torch
from tropnn.layers.maddness import FrozenMaddness, LocalCounterfactualMaddness, SoftPQMaddness
from tropnn.tools.maddness_end_to_end_ste_factorial import (
    compile_original_maddness,
)


def _fixture() -> tuple[torch.Tensor, object]:
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(512, 8, generator=generator)
    return x, compile_original_maddness(x, tables=4)


def test_compiler_shapes_and_balanced_codes() -> None:
    x, compiled = _fixture()
    assert compiled.split_indices.shape == (4, 4)
    assert compiled.thresholds.shape == (4, 15)
    assert compiled.encoder_centroids.shape == (4, 16, 2)
    assert compiled.prototypes.shape == (4, 16, 8)
    codes = FrozenMaddness(compiled).hard_codes(x)
    counts = torch.stack([torch.bincount(codes[:, table], minlength=16) for table in range(4)])
    assert bool((counts > 0).all())


def test_all_trainable_surrogates_have_identical_hard_forward() -> None:
    x, compiled = _fixture()
    frozen = FrozenMaddness(compiled)
    soft = SoftPQMaddness(compiled)
    local = LocalCounterfactualMaddness(compiled, tau=1.0)
    expected = frozen(x)
    assert torch.equal(soft(x), expected)
    assert torch.equal(local(x), expected)


def test_soft_pq_and_local_ste_receive_expected_gradients() -> None:
    x, compiled = _fixture()
    soft = SoftPQMaddness(compiled)
    local = LocalCounterfactualMaddness(compiled, tau=1.0)
    soft(x).square().mean().backward()
    local(x).square().mean().backward()
    assert soft.prototypes.grad is not None
    assert soft.log_temperature.grad is not None
    assert local.prototypes.grad is not None
    assert local.thresholds.grad is not None
    assert float(local.thresholds.grad.abs().sum()) > 0
