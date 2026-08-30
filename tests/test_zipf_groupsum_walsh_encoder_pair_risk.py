import json

import pytest
import torch
from tropnn.tools.zipf_groupsum_pair_merge_control import AdjacentPairMeanAccumulator
from tropnn.tools.zipf_groupsum_walsh_encoder_pair_risk import conditional_pair_risk, seal


def test_conditional_pair_risk_detects_pair_information() -> None:
    accumulator = AdjacentPairMeanAccumulator(32, 2)
    samples = 256
    bit = torch.arange(samples) % 2
    codes = torch.zeros(samples, 32, dtype=torch.int64)
    codes[:, 0] = bit
    targets = torch.stack((bit.to(torch.float32), 1.0 - bit.to(torch.float32)), dim=1)
    accumulator.update(codes, targets)
    payload, counts, global_mean = accumulator.finalize()
    result = conditional_pair_risk(payload, counts, global_mean, codes, targets, batch_size=31)
    assert result["individual_pair_r2"][0] == pytest.approx(1.0)
    assert result["individual_pair_r2"][1] == pytest.approx(0.0)
    assert result["mean_pair_r2"] == pytest.approx(1 / 16)
    assert result["any_pair_unseen_token_fraction"] == 0.0


def test_seal_selects_representation_misalignment_only_when_all_seeds_pass(tmp_path) -> None:
    run_dir = tmp_path / "runs"
    run_dir.mkdir()
    for seed, encoder_r2 in enumerate((0.41, 0.39, 0.42)):
        decoder_r2 = 0.24
        row = {
            "schema": "zipf-groupsum-walsh-encoder-pair-risk-v1",
            "complete": True,
            "seed": seed,
            "source_route_health_exact_reproduction": True,
            "source_state_exact_verification": {"all_equal": True},
            "artifact": {"strict_roundtrip_exact": True},
            "decoder_pair_average_test_replay_abs_difference": 0.0,
            "decision": {
                "encoder_pair_r2": encoder_r2,
                "decoder_pair_r2": decoder_r2,
                "encoder_over_decoder_r2": encoder_r2 - decoder_r2,
                "seed_passes_branch_b": True,
            },
            "test": {
                "encoder_equal_average": {"total_loss": 0.6},
                "decoder_equal_average": {"total_loss": 0.8},
                "source_sgd": {"total_loss": 0.7},
            },
        }
        (run_dir / f"seed-{seed}.json").write_text(json.dumps(row))
    result = seal(tmp_path)
    assert result["branch_b_representation_misalignment"] is True
    assert result["branch_a_repeat_mixing"] is False
    assert result["frozen_next_branch"] == "B_code_to_code_merger"


def test_seal_routes_any_failed_seed_to_repeat_mixing(tmp_path) -> None:
    run_dir = tmp_path / "runs"
    run_dir.mkdir()
    for seed in range(3):
        row = {
            "schema": "zipf-groupsum-walsh-encoder-pair-risk-v1",
            "complete": True,
            "seed": seed,
            "source_route_health_exact_reproduction": True,
            "source_state_exact_verification": {"all_equal": True},
            "artifact": {"strict_roundtrip_exact": True},
            "decoder_pair_average_test_replay_abs_difference": 0.0,
            "decision": {
                "encoder_pair_r2": 0.2 if seed == 2 else 0.4,
                "decoder_pair_r2": 0.2,
                "encoder_over_decoder_r2": 0.0 if seed == 2 else 0.2,
                "seed_passes_branch_b": seed != 2,
            },
            "test": {
                "encoder_equal_average": {"total_loss": 0.7},
                "decoder_equal_average": {"total_loss": 0.8},
                "source_sgd": {"total_loss": 0.7},
            },
        }
        (run_dir / f"seed-{seed}.json").write_text(json.dumps(row))
    result = seal(tmp_path)
    assert result["branch_b_representation_misalignment"] is False
    assert result["branch_a_repeat_mixing"] is True
    assert result["frozen_next_branch"] == "A_repeated_HD_mixing"
