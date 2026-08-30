from __future__ import annotations

import json

import torch
from torch import nn
from tropnn.layers.pairwise import PairwiseLUT
from tropnn.tools import zipf_groupsum_pclut_capacity_law as capacity_law
from tropnn.tools.zipf_groupsum_pclut_capacity_law import (
    FormalConfig,
    PyramidPairwiseStage,
    PyramidRecovery,
    Stage1Config,
    build_parser,
    enumerate_formal_configs,
    formal_ledger,
    make_pyramid_anchors,
    run_stage1,
    summarize_formal,
    summarize_stage1,
    summarize_stage1_disjoint,
    summarize_stage1_independent,
    train_formal_run,
)


def test_leaf_only_anchors_are_exact_canonical_random_anchors() -> None:
    anchors = make_pyramid_anchors(16, 3, 4, policy="leaf_only", seed=17)
    canonical = PairwiseLUT(16, 2, tables=3, comparisons=4, seed=17, backend="torch")
    assert torch.equal(anchors, canonical.anchors)


def test_pyramid_anchor_policies_are_in_range_and_same_level_when_required() -> None:
    n = 16
    offsets = (0, 16, 24, 28)
    sizes = (16, 8, 4, 2)
    for policy in ("level_uniform", "level_biased", "mixed"):
        anchors = make_pyramid_anchors(n, 4, 6, policy=policy, seed=3)
        assert anchors.shape == (4, 6, 2)
        assert bool((anchors[..., 0] != anchors[..., 1]).all())
        for left, right in anchors.reshape(-1, 2).tolist():
            assert any(offset <= left < offset + size and offset <= right < offset + size for offset, size in zip(offsets, sizes))
    node = make_pyramid_anchors(n, 4, 6, policy="node_uniform", seed=3)
    assert int(node.min()) >= 0 and int(node.max()) < 2 * n - 1


def test_same_level_disjoint_uses_unique_groups_within_each_table() -> None:
    anchors = make_pyramid_anchors(
        64,
        3,
        6,
        policy="same_level_disjoint",
        seed=11,
        group_size=4,
    )
    level_offset = 64 + 32
    assert anchors.shape == (3, 6, 2)
    assert bool((anchors >= level_offset).all())
    assert bool((anchors < level_offset + 16).all())
    for table in anchors:
        assert torch.unique(table).numel() == 12


def test_leaf_stage_matches_canonical_layer_bit_and_output_exactly() -> None:
    stage = PyramidPairwiseStage(
        16,
        5,
        tables=3,
        comparisons=4,
        route_kind="leaf_only",
        anchor_policy="node_uniform",
        seed=23,
        backend="torch",
    )
    canonical = PairwiseLUT(16, 5, tables=3, comparisons=4, seed=23, backend="torch", lut_init_std=0.02, lut_dtype="fp32")
    with torch.no_grad():
        canonical.thresholds.copy_(stage.lut.thresholds)
        canonical.lut.copy_(stage.lut.lut)
    x = torch.randn(7, 16)
    assert torch.equal(stage.route(x).indices, canonical.route(x).indices)
    assert torch.equal(stage(x), canonical(x).squeeze(1))


def test_signed_pyramid_routes_credit_to_input_threshold_and_payload() -> None:
    model = PyramidRecovery(
        8,
        4,
        tables=2,
        comparisons=2,
        route_kind="pyramid_signed",
        anchor_policy="level_uniform",
        seed=5,
        backend="torch",
    )
    x = torch.randn(6, 8, requires_grad=True)
    model(x).sum().backward()
    assert x.grad is not None and float(x.grad.abs().sum()) > 0
    for stage in (model.encoder, model.decoder):
        assert stage.lut.thresholds.grad is not None and float(stage.lut.thresholds.grad.abs().sum()) > 0
        assert stage.lut.lut.grad is not None and float(stage.lut.lut.grad.abs().sum()) > 0


def test_median_aggregation_has_exact_hard_action_and_canonical_route_credit() -> None:
    stage = PyramidPairwiseStage(
        4,
        1,
        tables=3,
        comparisons=1,
        route_kind="leaf_only",
        anchor_policy="node_uniform",
        aggregation="median",
        seed=9,
        backend="torch",
    )
    with torch.no_grad():
        stage.lut.anchors.copy_(torch.tensor([[[0, 1]], [[0, 1]], [[0, 1]]]))
        stage.lut.thresholds.zero_()
        stage.lut.lut.copy_(torch.tensor([[[0.0], [3.0]], [[0.0], [1.0]], [[0.0], [2.0]]]))
    x = torch.tensor([[1.0, 0.0, 0.0, 0.0]], requires_grad=True)
    assert torch.equal(stage(x).detach(), torch.tensor([[2.0]]))
    stage(x).sum().backward()
    assert x.grad is not None and float(x.grad.abs().sum()) > 0
    assert stage.lut.thresholds.grad is not None and float(stage.lut.thresholds.grad.abs().sum()) > 0


def test_tiny_stage1_run_reports_route_health_without_held_loss(tmp_path) -> None:
    result = run_stage1(
        Stage1Config(
            route_kind="pyramid_signed",
            anchor_policy="level_uniform",
            n_features=8,
            model_dim=4,
            tables=2,
            comparisons=2,
            batch_size=8,
            steps=2,
            warmup_steps=1,
            diagnostic_samples=16,
            diagnostic_batch_size=8,
            backend="torch",
            device="cpu",
        )
    )
    assert result["complete"] is True
    assert set(result["route_health"]) == {"samples", "encoder", "decoder"}
    assert "test" not in result and "validation" not in result


def test_independent_group_stage1_uses_sequential_pair_anchors() -> None:
    result = run_stage1(
        Stage1Config(
            route_kind="independent_groups",
            anchor_policy="node_uniform",
            anchor_group_size=4,
            decoder_anchor_policy="node_uniform",
            decoder_anchor_group_size=2,
            n_features=16,
            model_dim=4,
            tables=2,
            comparisons=2,
            batch_size=8,
            steps=2,
            warmup_steps=1,
            diagnostic_samples=16,
            diagnostic_batch_size=8,
            backend="torch",
            device="cpu",
        )
    )
    assert result["complete"] is True
    assert result["independent_groups"]["encoder"] is not None
    assert result["anchors"]["encoder"] == torch.arange(8).reshape(2, 2, 2).tolist()


def test_stage1_summary_selects_signed_policy_by_entropy_only(tmp_path) -> None:
    policies = ("node_uniform", "level_uniform", "level_biased", "mixed")
    rows = [("leaf_only", "node_uniform", 0.1)]
    rows += [("pyramid_unsigned", policy, 0.2 + index) for index, policy in enumerate(policies)]
    rows += [("pyramid_signed", policy, 1.0 + index) for index, policy in enumerate(policies)]
    for index, (route, policy, entropy) in enumerate(rows):
        payload = {
            "schema": "zipf-groupsum-pclut-stage1-route-health-v1",
            "complete": True,
            "config": {"route_kind": route, "anchor_policy": policy, "comparisons": 6},
            "route_health": {"encoder": {"entropy_bits_mean": entropy}},
        }
        (tmp_path / f"stage1-{index}.json").write_text(json.dumps(payload))
    summary = summarize_stage1(tmp_path)
    assert summary["selected_formal_policy"] == "mixed"
    assert summary["selected_formal_signed_entropy_bits"] == 4.0
    assert summary["selected_policy_g1_pass"] is True


def test_stage1b_summary_selects_disjoint_group_size_by_entropy_only(tmp_path) -> None:
    for group_size, entropy in ((1, 2.5), (2, 3.25), (4, 3.0)):
        payload = {
            "schema": "zipf-groupsum-pclut-stage1-route-health-v1",
            "complete": True,
            "config": {
                "route_kind": "pyramid_signed",
                "anchor_policy": "same_level_disjoint",
                "anchor_group_size": group_size,
                "decoder_anchor_policy": "level_biased",
                "comparisons": 6,
            },
            "route_health": {"encoder": {"entropy_bits_mean": entropy}},
        }
        (tmp_path / f"stage1-pyramid-signed-same-level-disjoint-g{group_size}.json").write_text(json.dumps(payload))
    summary = summarize_stage1_disjoint(tmp_path)
    assert summary["selected_group_size"] == 2
    assert summary["selected_entropy_bits"] == 3.25
    assert summary["g1_pass"] is True


def test_stage1c_summary_selects_independent_group_size_by_entropy_only(tmp_path) -> None:
    for group_size, entropy in ((8, 2.5), (16, 3.4), (32, 3.1)):
        payload = {
            "schema": "zipf-groupsum-pclut-stage1-route-health-v1",
            "complete": True,
            "config": {
                "route_kind": "independent_groups",
                "anchor_group_size": group_size,
                "comparisons": 6,
            },
            "route_health": {"encoder": {"entropy_bits_mean": entropy}},
        }
        (tmp_path / f"stage1-independent-g{group_size}-d32.json").write_text(json.dumps(payload))
    summary = summarize_stage1_independent(tmp_path)
    assert summary["selected_group_size"] == 16
    assert summary["selected_entropy_bits"] == 3.4
    assert summary["g1_pass"] is True


def test_formal_matrix_keeps_regular_primary_axes_and_labels_table_counts(tmp_path) -> None:
    args = build_parser().parse_args(
        [
            "formal",
            "--output-dir",
            str(tmp_path),
            "--anchor-group-size",
            "64",
        ]
    )
    configs = enumerate_formal_configs(args)
    assert len(configs) == 1260
    assert all(config.n_features == 1024 and config.n_features & (config.n_features - 1) == 0 for config in configs)
    assert all(config.model_dim & (config.model_dim - 1) == 0 for config in configs)
    regular = next(
        config
        for config in configs
        if config.arm == "pyramid_signed_sum" and config.model_dim == 32 and config.tables == 32 and config.comparisons == 6
    )
    assert formal_ledger(regular)["table_count_power_of_two"] is True


def test_tiny_disjoint_formal_run_saves_an_exclusive_reloadable_state(tmp_path) -> None:
    config = FormalConfig(
        "pyramid_signed_sum",
        64,
        8,
        1.0,
        1.0,
        2,
        4,
        "same_level_disjoint",
        4,
        "torch",
        0.0,
        0.01,
        8,
        2,
        1,
        16,
        8,
        0,
        "cpu",
    )
    checkpoint = tmp_path / "model.pt"
    result = train_formal_run(config, checkpoint_path=checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert result["complete"] is True
    assert result["route"]["encoder_anchor_group_size"] == 4
    assert result["route"]["decoder_anchor_group_size"] == 1
    assert len(state) == result["checkpoint"]["tensor_count"]


def test_nonfinite_formal_run_is_a_json_safe_completed_failure(tmp_path, monkeypatch) -> None:
    class NonfiniteModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(float("nan")))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * self.scale

    monkeypatch.setattr(capacity_law, "_build_formal_model", lambda _config: NonfiniteModel())
    config = FormalConfig(
        arm="leaf_sum",
        n_features=8,
        model_dim=4,
        alpha=1.0,
        activation_density=1.0,
        tables=1,
        comparisons=2,
        anchor_policy="leaf_only",
        anchor_group_size=None,
        backend="torch",
        weight_decay=0.0,
        learning_rate=0.01,
        batch_size=8,
        steps=2,
        warmup_steps=1,
        eval_samples=16,
        eval_batch_size=8,
        seed=0,
        device="cpu",
    )
    checkpoint = tmp_path / "must-not-exist.pt"
    result = train_formal_run(config, checkpoint_path=checkpoint)
    assert result["complete"] is True
    assert result["numerically_valid"] is False
    assert result["divergence"]["stage"] == "training_loss"
    assert result["validation"] is None and result["test"] is None
    assert result["checkpoint"] is None and not checkpoint.exists()
    json.dumps(result, allow_nan=False)


def test_formal_summary_records_divergence_without_selecting_it(tmp_path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    run_key = "invalid-primary"
    payload = {
        "schema": "zipf-groupsum-pclut-capacity-law-run-v3",
        "complete": True,
        "numerically_valid": False,
        "divergence": {"stage": "training_loss", "step": 7, "loss": None},
        "run_key": run_key,
        "config": {
            "arm": "independent_group_sum",
            "n_features": 8,
            "model_dim": 4,
            "tables": 1,
            "comparisons": 2,
            "seed": 0,
        },
        "ledger": {
            "table_count_power_of_two": True,
            "deploy_stored_bytes": 1,
            "active_model_bytes_unique": 1,
            "active_model_bytes_naive": 1,
        },
        "validation": None,
        "test": None,
    }
    (runs_dir / f"{run_key}.json").write_text(json.dumps(payload))
    (tmp_path / "manifest.json").write_text(json.dumps({"run_keys": [run_key], "config_count": 1}))
    summary = summarize_formal(tmp_path)
    assert summary["matrix_complete"] is True
    assert summary["numerically_valid_run_count"] == 0
    assert summary["invalid_run_count"] == 1
    assert summary["invalid_run_keys"] == [run_key]
    assert summary["comparisons"] == []


def test_legacy_result_is_valid_only_when_every_recorded_float_is_finite() -> None:
    finite = {"complete": True, "validation": {"total_loss": 0.25}, "loss_history": [{"mean_loss": 0.5}]}
    nonfinite = {"complete": True, "validation": {"total_loss": 0.25}, "loss_history": [{"mean_loss": float("nan")}]}
    assert capacity_law._run_numerically_valid(finite) is True
    assert capacity_law._run_numerically_valid(nonfinite) is False
