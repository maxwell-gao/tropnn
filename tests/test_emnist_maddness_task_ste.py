import torch
import torch.nn.functional as F
from tropnn.layers.maddness import FrozenMaddness, LocalCounterfactualMaddness, SoftPQMaddness
from tropnn.tools.emnist_maddness_task_ste import (
    MaddnessStemClassifier,
    compile_maddness_targets,
    compile_route_only_scratch,
)


def _fixture() -> tuple[torch.Tensor, torch.Tensor, object]:
    generator = torch.Generator().manual_seed(17)
    x = torch.randn(1024, 8, generator=generator)
    weight = torch.randn(5, 8, generator=generator)
    target = x @ weight.T
    return x, target, compile_maddness_targets(x, target, tables=4, ridge=0.1)


def test_target_compiler_emits_balanced_hard_layer_and_beats_global_mean() -> None:
    x, target, compiled = _fixture()
    assert compiled.split_indices.shape == (4, 4)
    assert compiled.thresholds.shape == (4, 15)
    assert compiled.encoder_centroids.shape == (4, 16, 2)
    assert compiled.prototypes.shape == (4, 16, 5)
    prediction = FrozenMaddness(compiled)(x)
    global_prediction = target.mean(0, keepdim=True).expand_as(target)
    assert F.mse_loss(prediction, target) < F.mse_loss(global_prediction, target)
    codes = FrozenMaddness(compiled).hard_codes(x)
    counts = torch.stack([torch.bincount(codes[:, table], minlength=16) for table in range(4)])
    assert bool((counts > 0).all())


def test_soft_and_local_use_identical_hard_forward_but_different_backward() -> None:
    x, _target, compiled = _fixture()
    x_soft = x[:32].clone().requires_grad_()
    x_local = x[:32].clone().requires_grad_()
    frozen = FrozenMaddness(compiled)
    soft = SoftPQMaddness(compiled)
    local = LocalCounterfactualMaddness(compiled, tau=1.0)
    expected = frozen(x[:32])
    assert torch.equal(soft(x_soft), expected)
    assert torch.equal(local(x_local), expected)
    soft(x_soft).square().mean().backward()
    local(x_local).square().mean().backward()
    assert x_soft.grad is not None and float(x_soft.grad.abs().sum()) > 0
    assert x_local.grad is not None and float(x_local.grad.abs().sum()) > 0
    assert local.thresholds.grad is not None and float(local.thresholds.grad.abs().sum()) > 0
    assert not torch.equal(x_soft.grad, x_local.grad)


def test_single_real_layer_passes_task_gradient_to_stem_and_matches_direct_lookup() -> None:
    x, _target, compiled = _fixture()
    stem = torch.nn.Linear(8, 8)
    model = MaddnessStemClassifier(stem, LocalCounterfactualMaddness(compiled, tau=1.0))
    labels = torch.randint(0, 5, (64,), generator=torch.Generator().manual_seed(9))
    features = model.features(x[:64])
    direct, _codes = model.head.hard_output(features)
    assert torch.equal(model(x[:64]), direct)
    F.cross_entropy(model(x[:64]), labels).backward()
    assert model.stem.weight.grad is not None and float(model.stem.weight.grad.abs().sum()) > 0
    assert model.head.thresholds.grad is not None and float(model.head.thresholds.grad.abs().sum()) > 0
    assert model.head.prototypes.grad is not None and float(model.head.prototypes.grad.abs().sum()) > 0


def test_scratch_route_uses_no_target_and_has_random_actions() -> None:
    x, _target, _compiled = _fixture()
    first = compile_route_only_scratch(x, tables=4, output_dim=5, seed=3, prototype_std=0.02)
    second = compile_route_only_scratch(x, tables=4, output_dim=5, seed=3, prototype_std=0.02)
    assert torch.equal(first.split_indices, second.split_indices)
    assert torch.equal(first.thresholds, second.thresholds)
    assert torch.equal(first.prototypes, second.prototypes)
    assert float(first.prototypes.std()) > 0
