import json

import pytest
from tropnn.tools.zipf_groupsum_walsh_pair_merge_bridge import seal


def test_bridge_seal_requires_three_improving_seeds(tmp_path) -> None:
    run_dir = tmp_path / "runs"
    run_dir.mkdir()
    for seed in range(3):
        row = {
            "schema": "zipf-groupsum-walsh-pair-merge-bridge-v1",
            "complete": True,
            "seed": seed,
            "source_route_health_exact_reproduction": True,
            "source_state_exact_verification": {"all_equal": True},
            "artifact": {"strict_roundtrip_exact": True},
            "test": {
                "source_sgd": {"total_loss": 0.7},
                "pair_merge": {"total_loss": 0.6},
                "naive_full_joint": {"total_loss": 1.0},
            },
            "unseen": {
                "test_any_pair_token_fraction": 0.01,
                "test_naive_joint_token_fraction": 0.5,
            },
        }
        (run_dir / f"seed-{seed}.json").write_text(json.dumps(row))
    result = seal(tmp_path)
    assert result["pair_merge_improves_source_all_seeds"] is True
    assert result["pair_merge_relative_improvement_mean"] == pytest.approx(1 / 7)
    assert result["test_naive_joint_unseen_mean"] == 0.5
