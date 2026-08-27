import torch
from tropnn.tools.random_linear_multitable_address_action_factorial import (
    _build_models,
    _make_multitable_regressor,
)


def test_multitable_hard_routes_and_zero_slope_nesting() -> None:
    models = _build_models(8, 3, 2, 7, 1.0, torch.device("cpu"))
    x = torch.randn(19, 8)
    for address in ("flat", "tree"):
        constant = models[f"{address}_constant"]
        live = models[f"{address}_live"]
        assert torch.equal(constant(x), live(x))
        codes = constant.hard_codes(x)
        assert codes.shape == (19, 3)
        assert bool(((codes >= 0) & (codes < 4)).all())


def test_leaf_probabilities_are_hard_one_hot_in_forward() -> None:
    for address in ("flat", "tree"):
        model = _make_multitable_regressor(8, 3, 2, address, "constant", anchor_seed=1, row_seed=2, tau=1.0)
        x = torch.randn(17, 8)
        leaf = model.leaf_probabilities(x)
        assert leaf.shape == (17, 3, 4)
        assert torch.allclose(leaf.sum(-1), torch.ones(17, 3), atol=1e-6, rtol=0)
        hard = torch.nn.functional.one_hot(model.hard_codes(x), num_classes=4).float()
        assert torch.allclose(leaf, hard, atol=1e-6, rtol=0)
