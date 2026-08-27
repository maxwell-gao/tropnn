import copy

import torch
import torch.nn.functional as F
from tropnn.layers.hard_lookup import HardLookupRouter
from tropnn.tools.emnist_maddness_learned_indices import (
    ScratchMaddnessStackClassifier,
    _make_index_lookup,
    _optimizer,
    data_free_tree_thresholds,
)


def _layer(*, learned: bool = True) -> HardLookupRouter:
    return _make_index_lookup(
        8,
        5,
        tables=4,
        seed=7,
        learn_indices=learned,
        tau=1.0,
        index_tau=1.0,
        prototype_std=0.1,
    )


def test_data_free_threshold_grid_is_balanced_and_has_no_capture_input() -> None:
    threshold = data_free_tree_thresholds(2)
    assert threshold.shape == (2, 15)
    assert torch.equal(threshold[0], threshold[1])
    assert float(threshold.min()) == -0.875
    assert float(threshold.max()) == 0.875


def test_learned_index_layer_has_exact_hard_forward_and_selector_gradient() -> None:
    layer = _layer(learned=True)
    x = torch.tanh(torch.randn(64, 8, generator=torch.Generator().manual_seed(11))).requires_grad_()
    expected, _ = layer.hard_output(x)
    actual = layer(x)
    assert torch.equal(actual, expected)
    actual.square().mean().backward()
    assert x.grad is not None and float(x.grad.abs().sum()) > 0
    assert layer.support_scores is not None
    assert layer.support_scores.grad is not None and float(layer.support_scores.grad.abs().sum()) > 0
    assert layer.thresholds.grad is not None and float(layer.thresholds.grad.abs().sum()) > 0
    assert layer.rows.grad is not None and float(layer.rows.grad.abs().sum()) > 0


def test_fixed_index_control_matches_initial_hard_program_and_has_no_selector_gradient() -> None:
    learned = _layer(learned=True)
    fixed = copy.deepcopy(learned)
    fixed.set_support_learning(False)
    x = torch.tanh(torch.randn(32, 8, generator=torch.Generator().manual_seed(13)))
    assert torch.equal(learned(x), fixed(x))
    fixed(x).square().mean().backward()
    assert fixed.support_scores is not None and fixed.support_scores.grad is None
    assert fixed.thresholds.grad is not None
    assert fixed.rows.grad is not None


def test_index_optimizer_changes_selected_integer_program_under_counterfactual_loss() -> None:
    layer = _layer(learned=True)
    with torch.no_grad():
        assert layer.support_scores is not None
        layer.support_scores.mul_(1e-3)
    initial = layer.selected_supports().clone()
    optimizer = _optimizer(layer, lr=1e-2, index_lr=0.2)
    generator = torch.Generator().manual_seed(19)
    for _ in range(20):
        x = torch.tanh(torch.randn(128, 8, generator=generator))
        target = torch.randn(128, 5, generator=generator)
        optimizer.zero_grad(set_to_none=True)
        F.mse_loss(layer(x), target).backward()
        optimizer.step()
    assert bool((layer.selected_supports() != initial).any())


def test_four_layer_stack_is_exact_hard_and_passes_ce_to_every_selector() -> None:
    model = ScratchMaddnessStackClassifier(
        input_dim=8,
        state_dim=8,
        classes=5,
        hidden_layers=4,
        tables=4,
        seed=23,
        learn_indices=True,
        residual_scale=0.25,
        tau=1.0,
        index_tau=1.0,
        prototype_std=0.05,
    )
    x = torch.randn(64, 8, generator=torch.Generator().manual_seed(29))
    target = torch.randint(0, 5, (64,), generator=torch.Generator().manual_seed(31))
    hard, codes, productive = model.hard_forward_with_trace(x)
    assert torch.equal(model(x), hard)
    assert len(codes) == len(productive) == 5
    F.cross_entropy(model(x), target).backward()
    assert model.stem.weight.grad is not None and float(model.stem.weight.grad.abs().sum()) > 0
    for layer in model.index_layers():
        assert layer.support_scores is not None
        assert layer.support_scores.grad is not None and float(layer.support_scores.grad.abs().sum()) > 0
        assert layer.thresholds.grad is not None and float(layer.thresholds.grad.abs().sum()) > 0
        assert layer.rows.grad is not None and float(layer.rows.grad.abs().sum()) > 0
