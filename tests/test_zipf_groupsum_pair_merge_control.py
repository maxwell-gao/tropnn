import json

import torch
from tropnn.tools.zipf_groupsum_pair_merge_control import (
    AdjacentPairMeanAccumulator,
    pair_mean_predict,
    seal,
)


def test_adjacent_pair_mean_is_exact_conditional_mean_and_global_fallback() -> None:
    accumulator = AdjacentPairMeanAccumulator(4, 3)
    codes = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 5], [2, 2, 3, 4]])
    targets = torch.tensor([[1.0, 0.0, 2.0], [3.0, 1.0, 0.0], [0.0, 5.0, 0.0]])
    accumulator.update(codes, targets)
    payload, counts, global_mean = accumulator.finalize()
    assert counts[0, 1 * 64 + 2] == 2
    assert torch.equal(payload[0, 1 * 64 + 2], torch.tensor([2.0, 0.5, 1.0]))
    assert torch.equal(payload[1, 3 * 64 + 4], torch.tensor([0.5, 2.5, 1.0]))
    assert torch.allclose(global_mean, torch.tensor([4 / 3, 2.0, 2 / 3]))
    unseen = torch.tensor([[63, 63, 63, 63]])
    assert torch.equal(pair_mean_predict(payload, unseen), global_mean.unsqueeze(0))


def test_pair_mean_predict_averages_two_pair_tables() -> None:
    payload = torch.zeros(2, 4096, 2)
    payload[0, 1 * 64 + 2] = torch.tensor([2.0, 4.0])
    payload[1, 3 * 64 + 4] = torch.tensor([6.0, 8.0])
    codes = torch.tensor([[1, 2, 3, 4]])
    assert torch.equal(pair_mean_predict(payload, codes), torch.tensor([[4.0, 6.0]]))


def test_seal_requires_both_arms_and_three_seeds(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source"
    (source / "checkpoints").mkdir(parents=True)
    output = tmp_path / "output"
    (output / "runs").mkdir(parents=True)
    keys = []
    for seed in range(3):
        keys.extend(
            (
                f"independent-group-sum-d32-t4-c6-g512-a1p0-e1p0-wd0p0-s{seed}",
                f"pyramid-signed-sum-d32-t4-c6-a1p0-e1p0-wd0p0-s{seed}",
            )
        )
    for key in keys:
        (source / "checkpoints" / f"{key}.pt").touch()
        arm = "independent_group_sum" if key.startswith("independent") else "pyramid_signed_sum"
        seed = int(key[-1])
        row = {
            "schema": "zipf-groupsum-pair-merge-control-v1",
            "complete": True,
            "source_run_key": key,
            "config": {"arm": arm, "seed": seed},
            "test": {
                "source_sgd": {"total_loss": 0.7},
                "pair_merge": {"total_loss": 0.6},
                "joint_code_oracle": {"total_loss": 0.4},
            },
            "source_state_exact_verification": {"all_equal": True},
            "artifact": {"strict_roundtrip_exact": True},
        }
        (output / "runs" / f"{key}.json").write_text(json.dumps(row))
    result = seal(source, output)
    assert result["run_count"] == 6
    assert all(row["pair_merge_improves_source_all_seeds"] for row in result["rows"])
