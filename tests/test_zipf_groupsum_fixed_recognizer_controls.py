from __future__ import annotations

import json

import torch
from tropnn.tools.zipf_groupsum_fixed_recognizer_controls import (
    CountActionAccumulator,
    RequestedJointMeanAccumulator,
    count_action_predict,
    reconstruction_metrics,
    route_tuple_keys,
    run_control,
)
from tropnn.tools.zipf_groupsum_pclut_capacity_law import FormalConfig, PyramidRecovery, _build_formal_model


def test_feature_owned_count_rows_are_exact_conditional_means() -> None:
    codes = torch.tensor(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ]
    )
    targets = torch.tensor(
        [
            [1.0, 10.0, 3.0, 30.0],
            [1.0, 20.0, 3.0, 40.0],
            [5.0, 10.0, 7.0, 30.0],
            [5.0, 20.0, 7.0, 40.0],
        ]
    )
    accumulator = CountActionAccumulator(tables=2, rows=2, output_dim=4)
    accumulator.update(codes[:2], targets[:2])
    accumulator.update(codes[2:], targets[2:])
    payload, counts, global_mean = accumulator.finalize()
    assert torch.equal(counts, torch.full((2, 2), 2, dtype=torch.int64))
    assert torch.equal(global_mean, torch.tensor([3.0, 15.0, 5.0, 35.0]))
    assert torch.equal(payload[0, 0, ::2], torch.tensor([1.0, 3.0]))
    assert torch.equal(payload[0, 1, ::2], torch.tensor([5.0, 7.0]))
    assert torch.equal(payload[1, 0, 1::2], torch.tensor([10.0, 30.0]))
    assert torch.equal(payload[1, 1, 1::2], torch.tensor([20.0, 40.0]))
    assert not bool(payload[0, :, 1::2].any())
    assert not bool(payload[1, :, ::2].any())
    assert torch.equal(count_action_predict(payload, codes), targets)


def test_unseen_count_row_uses_global_mean_only_for_owned_features() -> None:
    accumulator = CountActionAccumulator(tables=2, rows=2, output_dim=4)
    accumulator.update(torch.zeros(2, 2, dtype=torch.long), torch.tensor([[1.0, 2.0, 3.0, 4.0], [3.0, 4.0, 5.0, 6.0]]))
    payload, counts, global_mean = accumulator.finalize()
    assert torch.equal(counts[:, 1], torch.zeros(2, dtype=torch.int64))
    prediction = count_action_predict(payload, torch.ones(1, 2, dtype=torch.long))
    assert torch.equal(prediction[0], global_mean)


def test_joint_requested_conditional_mean_is_exact_and_unseen_falls_back() -> None:
    requested_codes = torch.tensor([[1, 2], [3, 4], [5, 6]])
    accumulator = RequestedJointMeanAccumulator(route_tuple_keys(requested_codes), output_dim=2)
    fit_codes = torch.tensor([[1, 2], [1, 2], [3, 4], [7, 7]])
    fit_targets = torch.tensor([[1.0, 3.0], [3.0, 5.0], [7.0, 9.0], [100.0, 100.0]])
    accumulator.update(fit_codes[:2], fit_targets[:2])
    accumulator.update(fit_codes[2:], fit_targets[2:])
    prediction, seen = accumulator.predict(requested_codes, torch.tensor([-1.0, -2.0]))
    assert torch.equal(prediction, torch.tensor([[2.0, 4.0], [7.0, 9.0], [-1.0, -2.0]]))
    assert torch.equal(seen, torch.tensor([True, True, False]))


def test_route_tuple_keys_preserve_order_and_width() -> None:
    codes = torch.tensor([[1, 2, 3], [1, 2, 4]], dtype=torch.int64)
    keys = route_tuple_keys(codes)
    assert len(keys) == 2 and len(keys[0]) == 3
    assert keys[0] != keys[1]
    wide = route_tuple_keys(torch.tensor([[256, 1]], dtype=torch.int64))
    assert len(wide[0]) == 4


def test_reconstruction_metrics_match_direct_total_loss() -> None:
    target = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    prediction = torch.zeros_like(target)
    metrics = reconstruction_metrics(target, prediction, torch.tensor([0.5, 0.25]))
    assert metrics["total_loss"] == 2.5
    assert metrics["mean_loss"] == 1.25
    assert metrics["tail_output_nonzero_fraction_1e_12"] == 0.0


def test_tiny_control_strictly_loads_source_and_roundtrips_count_artifact(tmp_path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "controls"
    (source / "runs").mkdir(parents=True)
    (source / "checkpoints").mkdir()
    config = FormalConfig(
        "independent_group_sum",
        1024,
        32,
        1.0,
        1.0,
        1,
        6,
        "node_uniform",
        512,
        "torch",
        0.0,
        0.01,
        4,
        1,
        0,
        4,
        4,
        0,
        "cpu",
    )
    model = _build_formal_model(config)
    assert isinstance(model, PyramidRecovery)
    checkpoint = source / "checkpoints" / f"{config.run_key}.pt"
    torch.save(model.state_dict(), checkpoint)
    stat = checkpoint.stat()
    result = {
        "schema": "zipf-groupsum-pclut-capacity-law-run-v3",
        "complete": True,
        "run_key": config.run_key,
        "config": config.__dict__,
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        },
        "route": {
            "encoder_anchors": model.encoder.lut.anchors.tolist(),
            "decoder_anchors": model.decoder.lut.anchors.tolist(),
            "encoder_thresholds": model.encoder.lut.thresholds.detach().tolist(),
            "decoder_thresholds": model.decoder.lut.thresholds.detach().tolist(),
        },
    }
    (source / "runs" / f"{config.run_key}.json").write_text(json.dumps(result))
    control = run_control(
        source,
        output,
        config.run_key,
        fit_samples=8,
        eval_samples=4,
        batch_size=4,
        device=torch.device("cpu"),
    )
    assert control["complete"] is True
    assert control["artifact"]["strict_roundtrip_exact"] is True
    assert control["source_state_exact_verification"]["all_equal"] is True
    assert control["fit"]["row_count_sum"] == 8
