import torch
import torch.nn.functional as F
from tropnn.tools.emnist_raw_recognizer_depth2 import ARMS, make_models, operation_ledger, route_and_action_parameters


def test_depth2_shapes_shared_initialization_and_hard_replay() -> None:
    models = make_models(
        32,
        7,
        tables=16,
        comparisons=4,
        seed=0,
        hidden_initial_std=0.5,
        logit_initial_std=0.02,
        temperature=1.0,
    )
    x = torch.randn(23, 32)
    for layer in ("first", "second"):
        pair_rows = getattr(getattr(models["pair"], layer), "rows")
        assert torch.equal(pair_rows, getattr(getattr(models["grid"], layer), "rows"))
        assert torch.equal(pair_rows, getattr(getattr(models["pq"], layer), "rows"))
    for layer in range(2):
        assert torch.equal(models["grid"].hard_codes(x)[layer], models["pq"].hard_codes(x)[layer])  # type: ignore[attr-defined]
    for arm in ARMS:
        models[arm].eval()
        explicit = models[arm](x) if arm == "dense" else models[arm].hard_output(x)[0]  # type: ignore[attr-defined]
        assert torch.equal(models[arm](x), explicit)
        assert explicit.shape == (23, 7)


def test_depth2_all_layer_parameters_receive_gradients() -> None:
    models = make_models(
        32,
        7,
        tables=16,
        comparisons=4,
        seed=1,
        hidden_initial_std=0.5,
        logit_initial_std=0.02,
        temperature=1.0,
    )
    x, target = torch.randn(41, 32), torch.randint(0, 7, (41,))
    sum(F.cross_entropy(model(x), target) for model in models.values()).backward()
    for arm in ARMS:
        route, action = route_and_action_parameters(models[arm], arm)
        assert all(parameter.grad is not None and float(parameter.grad.norm()) > 0 for parameter in action)
        assert all(parameter.grad is not None and float(parameter.grad.norm()) > 0 for parameter in route)


def test_depth2_formal_parameter_and_operation_counts() -> None:
    models = make_models(
        784,
        47,
        tables=392,
        comparisons=4,
        seed=0,
        hidden_initial_std=0.5,
        logit_initial_std=0.02,
        temperature=1.0,
    )
    counts = {arm: sum(parameter.numel() for parameter in model.parameters()) for arm, model in models.items()}
    assert counts == {"pair": 5215168, "grid": 5216736, "pq": 5237120, "dense": 652335}
    ledger = operation_ledger(784, 47, 392, 4)
    assert ledger["pair"]["threshold_comparisons"] == 3136
    assert ledger["grid"]["threshold_comparisons"] == 4704
    assert ledger["pq"]["squared_distance_terms"] == 25088
    assert ledger["dense"]["multiply_accumulates"] == 651504
    assert ledger["pair"]["active_row_scalar_reads"] == 325752
