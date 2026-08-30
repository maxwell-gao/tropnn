from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from tropnn.tools.emnist_raw_recognizer_factorial import (
    ARMS,
    SCHEMA,
    RawPQHead,
    grid_initialization,
    hard_output,
    make_balanced_pair_supports,
    make_models,
    operation_ledger,
    route_and_action_parameters,
)
from tropnn.tools.merge_emnist_raw_recognizer_factorial import merge_seed_directories


def test_balanced_pair_supports_are_unique_and_cover_every_coordinate() -> None:
    supports = make_balanced_pair_supports(32, 4, 17)
    assert supports.shape == (16, 4, 2)
    assert torch.equal(torch.bincount(supports.flatten(), minlength=32), torch.full((32,), 4))
    canonical = supports.sort(dim=-1).values.reshape(-1, 2)
    assert torch.unique(canonical, dim=0).shape[0] == canonical.shape[0]
    assert torch.equal(supports, make_balanced_pair_supports(32, 4, 17))


def test_initial_grid_and_pq_codes_are_exactly_equal() -> None:
    models = make_models(32, 7, comparisons=4, seed=3, row_init_std=1e-3, temperature=1.0)
    generator = torch.Generator().manual_seed(9)
    x = torch.rand(257, 32, generator=generator) * 2 - 1
    assert torch.equal(models["grid"].hard_codes(x), models["pq"].hard_codes(x))  # type: ignore[attr-defined]
    grid_rows = models["grid"].rows.detach()  # type: ignore[attr-defined]
    assert torch.equal(models["pair"].rows.detach(), grid_rows)  # type: ignore[attr-defined]
    assert torch.equal(models["pq"].rows.detach(), grid_rows)  # type: ignore[attr-defined]
    models["grid"].eval()
    hard, _codes = models["grid"].hard_output(x)  # type: ignore[attr-defined]
    assert torch.equal(models["grid"](x), hard)


def test_soft_pq_has_exact_hard_forward_and_route_gradient() -> None:
    _supports, _thresholds, centroids = grid_initialization(8)
    rows = torch.randn(4, 16, 5) * 0.01
    model = RawPQHead(centroids, rows, temperature=1.0)
    x = torch.randn(11, 8)
    target = torch.randint(0, 5, (11,))
    hard, _codes = model.hard_output(x)
    output = model(x)
    assert torch.equal(hard, output)
    F.cross_entropy(output, target).backward()
    assert model.centroids.grad is not None and float(model.centroids.grad.norm()) > 0
    assert model.rows.grad is not None and float(model.rows.grad.norm()) > 0


def test_all_four_arms_train_and_hard_replay() -> None:
    models = make_models(32, 7, comparisons=4, seed=5, row_init_std=1e-3, temperature=1.0)
    x = torch.randn(29, 32)
    target = torch.randint(0, 7, (29,))
    parameters = [parameter for model in models.values() for parameter in model.parameters()]
    optimizer = torch.optim.AdamW(parameters, lr=1e-3, weight_decay=0)
    before = {arm: [parameter.detach().clone() for parameter in route_and_action_parameters(model, arm)[0]] for arm, model in models.items()}
    optimizer.zero_grad(set_to_none=True)
    logits = {arm: model(x) for arm, model in models.items()}
    sum(F.cross_entropy(value, target) for value in logits.values()).backward()
    optimizer.step()
    for arm in ARMS:
        explicit, _codes = hard_output(models[arm], arm, x)
        assert torch.equal(models[arm](x), explicit)
        route, action = route_and_action_parameters(models[arm], arm)
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in action)
        if route:
            assert all(parameter.grad is not None and float(parameter.grad.norm()) > 0 for parameter in route)
            assert any(not torch.equal(parameter.detach(), reference) for parameter, reference in zip(route, before[arm], strict=True))


def test_operation_ledger_and_formal_parameter_shapes() -> None:
    ledger = operation_ledger(784, 47, 392, 4)
    assert ledger["pair"]["threshold_comparisons"] == 1568
    assert ledger["grid"]["threshold_comparisons"] == 2352
    assert ledger["pq"]["squared_distance_terms"] == 12544
    assert ledger["dense"]["multiply_accumulates"] == 36848
    models = make_models(784, 47, comparisons=4, seed=0, row_init_std=0.02 / math.sqrt(392), temperature=1.0)
    counts = {arm: sum(parameter.numel() for parameter in model.parameters()) for arm, model in models.items()}
    assert counts == {"pair": 296352, "grid": 297136, "pq": 307328, "dense": 36895}
    assert all(not hasattr(model, "stem") for model in models.values())


def test_parallel_seed_merge_is_mechanical(tmp_path: Path) -> None:
    seed_directories: list[Path] = []
    protocol = {"seeds": [], "dense_stem_present": False, "operation_ledger": {}}
    for seed in (0, 1):
        directory = tmp_path / f"seed{seed}"
        directory.mkdir()
        seed_protocol = {**protocol, "seeds": [seed]}
        rows = []
        for index, arm in enumerate(ARMS):
            rows.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "held_ce": float(seed + index),
                    "held_accuracy": 0.5,
                    "parameter_count": 1,
                    "route_parameter_count": 0,
                    "action_parameter_count": 1,
                    "mean_entropy_bits": None,
                    "minimum_entropy_bits": None,
                    "mean_observed_rows": None,
                    "maximum_row_mass": None,
                    "hard_replay_max_error": 0.0,
                    "eager_hard_forward_ms": 1.0,
                }
            )
        (directory / "result.json").write_text(
            json.dumps(
                {
                    "schema": SCHEMA,
                    "protocol": seed_protocol,
                    "rows": rows,
                    "audits": {str(seed): {"ok": True}},
                    "artifact_roundtrip_exact": True,
                }
            )
        )
        torch.save(
            {"schema": SCHEMA, "protocol": seed_protocol, "state": {f"seed{seed}.weight": torch.tensor([seed])}},
            directory / "artifact.pt",
        )
        seed_directories.append(directory)
    merged = merge_seed_directories(tuple(seed_directories), tmp_path / "result.json", tmp_path / "artifact.pt")
    assert merged["protocol"]["seeds"] == [0, 1]  # type: ignore[index]
    assert len(merged["rows"]) == 8
    assert merged["artifact_roundtrip_exact"] is True
