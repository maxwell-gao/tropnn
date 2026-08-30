from __future__ import annotations

import pytest
import torch
from tropnn.layers.accumulation import IndependentGroupSums, SumPyramid, WalshButterfly
from tropnn.layers.pairwise import PairwiseLUT


def test_walsh_butterfly_is_exact_shared_add_subtract_transform() -> None:
    transform = WalshButterfly(4, seed=3)
    transform.signs.fill_(1)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
    expected = torch.tensor([[10.0, -2.0, -4.0, 0.0]])
    output = transform(x)
    assert torch.equal(output, expected)
    assert transform.scalar_add_subtracts == 8
    output.square().sum().backward()
    assert torch.allclose(x.grad, 8 * x.detach())


def test_sum_pyramid_has_exact_leaf_to_root_layout() -> None:
    pyramid = SumPyramid(4)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    assert pyramid.output_dim == 7
    assert pyramid.level_sizes == (4, 2, 1)
    assert pyramid.level_offsets == (0, 4, 6)
    assert torch.equal(pyramid(x), torch.tensor([[1.0, 2.0, 3.0, 4.0, 3.0, 7.0, 10.0]]))


def test_signed_sum_pyramid_is_deterministic_and_has_exact_transpose_gradient() -> None:
    first = SumPyramid(8, signed=True, seed=19)
    second = SumPyramid(8, signed=True, seed=19)
    assert torch.equal(first.signs, second.signs)
    assert set(first.signs.tolist()) <= {-1, 1}
    assert list(first.parameters()) == []

    x = torch.randn(2, 3, 8, requires_grad=True)
    first(x).sum().backward()
    expected = first.signs.to(torch.float32) * 4
    assert torch.equal(x.grad, expected.expand_as(x))


def test_sum_pyramid_rejects_non_power_of_two_width_and_wrong_input() -> None:
    with pytest.raises(ValueError, match="power of two"):
        SumPyramid(6)
    with pytest.raises(ValueError, match="last input dimension"):
        SumPyramid(4)(torch.randn(2, 5))


def test_independent_group_sums_are_disjoint_per_predicate_and_add_only() -> None:
    layer = IndependentGroupSums(16, 3, group_size=4, seed=7)
    assert layer.output_dim == 6
    assert list(layer.parameters()) == []
    for predicate in range(3):
        left, right = layer.groups[2 * predicate : 2 * predicate + 2]
        assert torch.unique(torch.cat((left, right))).numel() == 8
    x = torch.arange(16, dtype=torch.float32).reshape(1, 16).requires_grad_(True)
    expected = torch.stack([x[0, group].sum() for group in layer.groups]).reshape(1, 6)
    assert torch.equal(layer(x), expected)
    layer(x).sum().backward()
    expected_grad = torch.bincount(layer.groups.reshape(-1), minlength=16).to(torch.float32).reshape(1, 16)
    assert torch.equal(x.grad, expected_grad)


def test_pairwise_lut_accepts_validated_explicit_anchors() -> None:
    anchors = torch.tensor([[[0, 1], [4, 6]], [[2, 3], [5, 1]]])
    layer = PairwiseLUT(7, 3, tables=2, comparisons=2, anchors=anchors, backend="torch")
    assert layer.anchor_policy == "explicit"
    assert torch.equal(layer.anchors, anchors)
    anchors.zero_()
    assert not torch.equal(layer.anchors, anchors)


@pytest.mark.parametrize(
    ("anchors", "error"),
    (
        (torch.zeros(2, 2, dtype=torch.long), ValueError),
        (torch.tensor([[[0, 7], [1, 2]], [[2, 3], [4, 5]]]), ValueError),
        (torch.tensor([[[0, 0], [1, 2]], [[2, 3], [4, 5]]]), ValueError),
        (torch.zeros(2, 2, 2, dtype=torch.float32), TypeError),
    ),
)
def test_pairwise_lut_rejects_invalid_explicit_anchors(anchors: torch.Tensor, error: type[Exception]) -> None:
    with pytest.raises(error):
        PairwiseLUT(7, 3, tables=2, comparisons=2, anchors=anchors, backend="torch")


def test_explicit_anchor_policy_cannot_be_combined_with_generated_policy() -> None:
    anchors = torch.tensor([[[0, 1]]])
    with pytest.raises(ValueError, match="cannot be combined"):
        PairwiseLUT(2, 1, tables=1, comparisons=1, anchors=anchors, anchor_policy="local", backend="torch")
