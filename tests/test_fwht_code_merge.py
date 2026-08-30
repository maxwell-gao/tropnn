from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from tropnn.layers.fwht_code_merge import FWHTCodeMergeLUT, FWHTPairCodeEncoder, make_disjoint_pair_supports
from tropnn.tools.emnist_fwht_code_merge import (
    ARMS,
    Evaluation,
    hard_output,
    make_models,
    operation_ledger,
    route_and_action_parameters,
    summarize,
)


def test_disjoint_pair_supports_are_deterministic_and_globally_unique() -> None:
    supports = make_disjoint_pair_supports(64, 4, 3, seed=7)
    assert supports.shape == (4, 3, 2)
    assert torch.unique(supports).numel() == supports.numel()
    assert torch.equal(supports, make_disjoint_pair_supports(64, 4, 3, seed=7))


def test_fwht_pair_encoder_zero_pads_and_packs_lsb_codes() -> None:
    supports = torch.tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
    encoder = FWHTPairCodeEncoder(6, 8, supports, seed=3, normalize=True)
    x = torch.randn(11, 6)
    route = encoder.route(x)
    assert route.codes.shape == (11, 2)
    assert route.margins.shape == (11, 2, 2)
    expected = ((route.margins > 0).to(torch.int64) * torch.tensor([1, 2])).sum(dim=-1)
    assert torch.equal(route.codes, expected)


def test_code_merge_xor_initialization_and_compiled_hard_forward_are_exact() -> None:
    supports = make_disjoint_pair_supports(16, 4, 2, seed=5)
    model = FWHTCodeMergeLUT(
        16,
        16,
        5,
        supports,
        seed=11,
        row_init_std=0.02,
        merger_init_logit=1.0,
        merger_initialization="xor",
    )
    mapping = model.compiled_merger_map()
    rows = torch.arange(16, dtype=torch.int64)
    expected = (rows // 4) ^ (rows % 4)
    assert torch.equal(mapping, expected.view(1, -1).expand(2, -1))

    x = torch.randn(23, 16)
    model.train()
    forward = model(x)
    hard, codes = model.hard_output(x)
    leaf = model.encoder.route(x).codes.reshape(23, 2, 2)
    assert torch.equal(codes, leaf[..., 0] ^ leaf[..., 1])
    assert torch.equal(forward, hard)


def test_code_merge_local_counterfactual_trains_every_parameter_family() -> None:
    supports = make_disjoint_pair_supports(32, 4, 2, seed=13)
    model = FWHTCodeMergeLUT(32, 32, 7, supports, seed=17, row_init_std=0.02, merger_init_logit=0.5)
    x = torch.randn(41, 32)
    target = torch.randint(0, 7, (41,))
    model.train()
    hard_before, _codes = model.hard_output(x)
    output = model(x)
    assert torch.equal(output, hard_before)
    F.cross_entropy(output, target).backward()
    assert model.thresholds.grad is not None and float(model.thresholds.grad.norm()) > 0
    assert model.merger_logits.grad is not None and float(model.merger_logits.grad.norm()) > 0
    assert model.action_rows.grad is not None and float(model.action_rows.grad.norm()) > 0
    assert model.bias.grad is not None and float(model.bias.grad.norm()) > 0


def test_balanced_random_merger_initialization_is_exactly_balanced_and_not_xor() -> None:
    supports = make_disjoint_pair_supports(32, 4, 2, seed=23)
    model = FWHTCodeMergeLUT(32, 32, 3, supports, seed=29, row_init_std=0.02)
    mapping = model.initial_merger_map
    for merger in range(model.mergers):
        assert torch.equal(torch.bincount(mapping[merger], minlength=4), torch.full((4,), 4))
    inputs = torch.arange(16)
    xor = (inputs // 4) ^ (inputs % 4)
    assert not torch.equal(mapping, xor.view(1, -1).expand_as(mapping))
    assert torch.equal(model.compiled_merger_map(), mapping)


def _tiny_args() -> SimpleNamespace:
    return SimpleNamespace(
        input_dim=32,
        classes=7,
        transform_dim=32,
        tables=4,
        comparisons=2,
        initial_logit_std=0.02,
        route_temperature=1.0,
        merger_init_logit=1.0,
        merger_initialization="balanced_random",
    )


def test_all_emnist_arms_take_a_step_and_replay_their_hard_forward() -> None:
    args = _tiny_args()
    models = make_models(args, seed=19)
    assert torch.equal(models["raw_flat"].supports, models["raw_merge"].encoder.supports)  # type: ignore[attr-defined]
    assert torch.equal(models["fwht_flat"].encoder.supports, models["fwht_merge"].encoder.supports)  # type: ignore[attr-defined]
    for field in ("initial_merger_map", "merger_logits", "action_rows", "bias"):
        assert torch.equal(getattr(models["raw_merge"], field), getattr(models["fwht_merge"], field))
    x = torch.randn(37, args.input_dim)
    target = torch.randint(0, args.classes, (37,))
    optimizer = torch.optim.AdamW([parameter for model in models.values() for parameter in model.parameters()], lr=1e-3, weight_decay=0)
    optimizer.zero_grad(set_to_none=True)
    outputs = {arm: model(x) for arm, model in models.items()}
    sum(F.cross_entropy(output, target) for output in outputs.values()).backward()
    for arm, model in models.items():
        route, action = route_and_action_parameters(model, arm)
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in action)
        if route:
            assert all(parameter.grad is not None and float(parameter.grad.norm()) > 0 for parameter in route)
    optimizer.step()
    for arm in ARMS:
        models[arm].eval()
        explicit, _codes = hard_output(models[arm], arm, x)
        assert torch.equal(models[arm](x), explicit)


def test_formal_parameter_counts_and_inference_ledger() -> None:
    args = SimpleNamespace(
        input_dim=784,
        classes=47,
        transform_dim=1024,
        tables=32,
        comparisons=6,
        initial_logit_std=0.02,
        route_temperature=1.0,
        merger_init_logit=1.0,
        merger_initialization="balanced_random",
    )
    models = make_models(args, seed=0)
    counts = {arm: sum(parameter.numel() for parameter in model.parameters()) for arm, model in models.items()}
    assert counts == {
        "raw_flat": 96_495,
        "raw_merge": 441_583,
        "fwht_flat": 96_495,
        "fwht_merge": 441_583,
        "dense": 36_895,
    }
    ledger = operation_ledger(args)
    assert ledger["fwht_merge"]["fwht_add_subtracts"] == 10_240
    assert ledger["fwht_merge"]["threshold_compares"] == 192
    assert ledger["fwht_merge"]["compiled_merger_map_lookups"] == 16
    assert ledger["fwht_merge"]["action_row_lookups"] == 16
    assert ledger["raw_merge"]["compiled_merger_map_lookups"] == 16
    assert ledger["dense"]["multiply_accumulates"] == 36_848


def test_factorial_summary_reports_requested_contrasts_and_did() -> None:
    ce = {"raw_flat": 4.0, "raw_merge": 3.5, "fwht_flat": 3.0, "fwht_merge": 2.0, "dense": 1.0}
    rows = [
        Evaluation(
            seed=0,
            arm=arm,
            held_ce=value,
            held_accuracy=0.0,
            parameter_count=0,
            route_parameter_count=0,
            action_parameter_count=0,
            mean_code_entropy_bits=None,
            minimum_code_entropy_bits=None,
            mean_observed_rows=None,
            maximum_row_mass=None,
            leaf_mean_entropy_bits=None,
            hard_replay_max_error=0.0,
            eager_hard_forward_ms=0.0,
        )
        for arm, value in ce.items()
    ]
    signed = summarize(rows)["signed_ce_contrasts_requested_order"]
    assert signed == {
        "fwht_flat_minus_raw_flat": -1.0,
        "raw_merge_minus_raw_flat": -0.5,
        "fwht_merge_minus_fwht_flat": -1.0,
        "difference_in_differences": -0.5,
    }
