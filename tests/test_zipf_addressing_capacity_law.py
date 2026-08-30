from __future__ import annotations

import json
import math

import pytest
import torch
from tropnn.tools.zipf_addressing_capacity_law import (
    RoutedTableRecovery,
    RunConfig,
    _apply_superposition_weight_decay,
    budget_ledger,
    build_parser,
    enumerate_configs,
    evaluate_model,
    filter_configs_by_run_key_file,
    sample_liu_gore_batch,
    summarize,
    train_run,
    zipf_probabilities,
)


def test_liu_gore_sampler_has_exact_zipf_density_and_uniform_amplitude() -> None:
    probabilities = zipf_probabilities(100, 1.0, 2.0, device=torch.device("cpu"))
    assert math.isclose(float(probabilities.sum()), 2.0, rel_tol=1e-6)
    assert torch.all(probabilities[:-1] > probabilities[1:])
    generator = torch.Generator().manual_seed(7)
    batch = sample_liu_gore_batch(probabilities, 20_000, generator=generator)
    assert float(batch.min()) >= 0.0
    assert float(batch.max()) < 2.0
    assert math.isclose(float((batch > 0).sum(dim=1).float().mean()), 2.0, rel_tol=0.03)
    active_values = batch[batch > 0]
    assert math.isclose(float(active_values.mean()), 1.0, rel_tol=0.02)


def test_canonical_pair_route_uses_all_bits_and_exact_hard_row() -> None:
    from tropnn import PairwiseLUT

    layer = PairwiseLUT(4, 2, tables=1, comparisons=2, backend="torch", seed=3, lut_init_std=0.0, use_output_scaling=False, lut_dtype="fp32")
    with torch.no_grad():
        layer.anchors.copy_(torch.tensor([[[0, 1], [2, 3]]]))
        layer.thresholds.zero_()
        layer.lut.zero_()
        layer.lut[0, 1] = torch.tensor([1.0, 2.0])
        layer.lut[0, 2] = torch.tensor([3.0, 4.0])
    layer.eval()
    output = layer(torch.tensor([[3.0, 1.0, 0.0, 2.0], [0.0, 2.0, 4.0, 1.0]]))
    assert torch.equal(layer.route(torch.tensor([[3.0, 1.0, 0.0, 2.0], [0.0, 2.0, 4.0, 1.0]])).indices, torch.tensor([[1], [2]]))
    assert torch.equal(output.squeeze(1), torch.tensor([[1.0, 2.0], [3.0, 4.0]]))


def test_canonical_route_threshold_payload_and_input_receive_credit() -> None:
    from tropnn import PairwiseLUT

    layer = PairwiseLUT(4, 1, tables=1, comparisons=1, backend="torch", seed=5, lut_init_std=0.0, use_output_scaling=False, lut_dtype="fp32")
    with torch.no_grad():
        layer.anchors.copy_(torch.tensor([[[0, 1]]]))
        layer.thresholds.fill_(0.2)
        layer.lut.copy_(torch.tensor([[[0.0], [2.0]]]))
    layer.train()
    inputs = torch.tensor([[0.8, 0.1, 0.2, 0.9]], requires_grad=True)
    layer(inputs).sum().backward()
    assert layer.lut.grad is not None and float(layer.lut.grad.abs().sum()) > 0.0
    assert layer.thresholds.grad is not None and float(layer.thresholds.grad.abs().sum()) > 0.0
    assert inputs.grad is not None and float(inputs.grad.abs().sum()) > 0.0
    assert "anchors" not in dict(layer.named_parameters())


def test_budget_ledger_matches_actual_trainable_parameters() -> None:
    config = RunConfig("lut", 32, 8, 1.0, 1.0, 4, 4, "torch", 0.0, 0.01, 16, 2, 0, 32, 16, 0, "cpu")
    model = RoutedTableRecovery(32, 8, tables=4, comparisons=4, seed=1000, backend="torch")
    ledger = budget_ledger(config)
    assert int(ledger["deploy_learned_scalars"]) == 4 * 16 * (32 + 8) + 2 * 4 * 4 + 32
    assert int(ledger["trainable_scalars"]) == sum(parameter.numel() for parameter in model.parameters())
    assert int(ledger["active_payload_scalar_reads"]) == 4 * (32 + 8)
    assert int(ledger["active_comparisons"]) == 2 * 4 * 4


def test_signed_superposition_decay_matches_reference_row_norm_update() -> None:
    negative = torch.nn.Parameter(torch.tensor([[2.0, 0.0]]))
    negative.grad = torch.ones_like(negative)
    optimizer = torch.optim.Adam([{"params": [negative], "lr": 0.1, "superposition_weight_decay": -1.0}])
    _apply_superposition_weight_decay(optimizer)
    assert torch.allclose(negative, torch.tensor([[1.9, 0.0]]), atol=1e-7, rtol=0.0)

    positive = torch.nn.Parameter(torch.tensor([[2.0, 0.0]]))
    positive.grad = torch.ones_like(positive)
    optimizer = torch.optim.Adam([{"params": [positive], "lr": 0.1, "superposition_weight_decay": 0.2}])
    _apply_superposition_weight_decay(optimizer)
    assert torch.allclose(positive, torch.tensor([[1.96, 0.0]]), atol=1e-7, rtol=0.0)


def test_default_registered_matrix_has_exact_canonical_size() -> None:
    args = build_parser().parse_args(["run", "--output-dir", "unused", "--device", "cpu"])
    configs = enumerate_configs(args)
    assert len(configs) == 354
    assert sum(config.family == "dense" for config in configs) == 60
    assert sum(config.family == "lut" for config in configs) == 294
    assert {1 << config.comparisons for config in configs if config.family == "lut"} == {4, 16, 64, 256}


def test_identity_has_zero_rank_resolved_risk() -> None:
    probabilities = zipf_probabilities(16, 1.0, 1.0, device=torch.device("cpu"))
    metrics = evaluate_model(torch.nn.Identity(), probabilities, samples=256, batch_size=64, generator_seed=99)
    assert metrics["total_loss"] == 0.0
    assert metrics["zero_normalized_loss"] == 0.0
    assert len(metrics["feature_mse"]) == 16
    assert max(metrics["feature_mse"]) == 0.0


def test_sweep_includes_exact_parameter_and_bandwidth_boundary_points() -> None:
    args = build_parser().parse_args(
        [
            "run",
            "--output-dir",
            "unused",
            "--n-features",
            "100",
            "--model-dims",
            "15",
            "--dense-weight-decays",
            "0",
            "--lut-comparisons",
            "2",
            "--backend",
            "torch",
            "--seeds",
            "0",
        ]
    )
    configs = enumerate_configs(args)
    dense = next(config for config in configs if config.family == "dense")
    luts = [config for config in configs if config.family == "lut"]
    dense_ledger = budget_ledger(dense)
    parameter_eligible = [config for config in luts if int(budget_ledger(config)["deploy_stored_bytes"]) <= int(dense_ledger["deploy_stored_bytes"])]
    bandwidth_eligible = [
        config for config in luts if int(budget_ledger(config)["active_model_bytes_unique"]) <= int(dense_ledger["active_model_bytes_unique"])
    ]
    assert parameter_eligible
    assert bandwidth_eligible
    assert max(int(budget_ledger(config)["deploy_stored_bytes"]) for config in parameter_eligible) <= int(dense_ledger["deploy_stored_bytes"])


def test_run_key_file_is_a_strict_scheduling_filter(tmp_path) -> None:
    args = build_parser().parse_args(
        [
            "run",
            "--output-dir",
            "unused",
            "--n-features",
            "16",
            "--model-dims",
            "4",
            "--dense-weight-decays",
            "0",
            "--lut-comparisons",
            "2",
            "--backend",
            "torch",
            "--seeds",
            "0",
        ]
    )
    configs = enumerate_configs(args)
    selected_key = configs[-1].run_key
    run_keys = tmp_path / "run-keys.txt"
    run_keys.write_text(f"# work-stealing subset\n{selected_key}\n")
    selected = filter_configs_by_run_key_file(configs, run_keys)
    assert [config.run_key for config in selected] == [selected_key]

    run_keys.write_text("not-a-real-run-key\n")
    with pytest.raises(ValueError, match="unknown keys"):
        filter_configs_by_run_key_file(configs, run_keys)


def test_tiny_dense_and_lut_training_runs_are_finite() -> None:
    common = dict(
        n_features=8,
        model_dim=4,
        alpha=1.0,
        activation_density=1.0,
        weight_decay=0.0,
        learning_rate=0.01,
        batch_size=16,
        steps=3,
        warmup_steps=1,
        eval_samples=32,
        eval_batch_size=16,
        seed=0,
        device="cpu",
    )
    dense = train_run(RunConfig("dense", tables=0, comparisons=0, backend="torch", **common))
    lut = train_run(RunConfig("lut", tables=2, comparisons=2, backend="torch", **common))
    assert dense["complete"] is True and lut["complete"] is True
    assert math.isfinite(float(dense["test"]["total_loss"]))
    assert math.isfinite(float(lut["test"]["total_loss"]))
    assert lut["route"]["anchors_fixed"] is True
    assert lut["route"]["thresholds_learned"] is True


def test_summary_requires_three_seed_two_adjacent_dimension_tail_crossover(tmp_path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    run_count = 0
    for model_dim in (8, 16):
        for seed in (0, 1, 2):
            dense_feature = [1.0] * 16
            lut_feature = [1.2] * 8 + [0.9] * 8
            for family, feature, stored, active in (
                ("dense", dense_feature, 1000, 1000),
                ("lut", lut_feature, 900, 900),
            ):
                run_key = f"{family}-d{model_dim}-s{seed}"
                payload = {
                    "schema": "zipf-addressing-capacity-law-run-v1",
                    "complete": True,
                    "run_key": run_key,
                    "config": {
                        "family": family,
                        "alpha": 1.0,
                        "activation_density": 1.0,
                        "seed": seed,
                        "model_dim": model_dim,
                        "n_features": 16,
                    },
                    "ledger": {
                        "deploy_stored_bytes": stored,
                        "active_model_bytes_unique": active,
                        "active_model_bytes_naive": active,
                    },
                    "validation": {"total_loss": sum(feature), "feature_mse": feature},
                    "test": {"total_loss": sum(feature), "feature_mse": feature},
                }
                (runs_dir / f"{run_key}.json").write_text(json.dumps(payload))
                run_count += 1
    (tmp_path / "manifest.json").write_text(json.dumps({"config_count": run_count}))
    result = summarize(tmp_path)
    assert result["matrix_complete"] is True
    assert result["scientific_decision"] == "positive_parameter_tail_crossover"
    parameter = next(item for item in result["budget_decisions"] if item["budget_kind"] == "parameter")
    assert parameter["adjacent_passing_dimension_pairs"] == [[8, 16]]
