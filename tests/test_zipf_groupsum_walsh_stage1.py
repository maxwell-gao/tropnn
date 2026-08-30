import json

import pytest
import torch
from tropnn.tools.zipf_groupsum_walsh_stage1 import Config, WalshRecovery, seal


def test_walsh_recovery_uses_one_shared_transform_per_stage() -> None:
    model = WalshRecovery(16, 8, tables=3, comparisons=2, seed=7)
    x = torch.randn(5, 16)
    output, hidden = model.forward_with_hidden(x)
    assert output.shape == (5, 16)
    assert hidden.shape == (5, 8)
    assert model.encoder.transform.scalar_add_subtracts == 64
    assert model.decoder.transform.scalar_add_subtracts == 24


def test_config_freezes_stage1_prime_surface() -> None:
    Config(seed=0, device="cpu").validate_formal()
    try:
        Config(seed=0, device="cpu", tables=31).validate_formal()
    except ValueError:
        pass
    else:
        raise AssertionError("formal contract accepted a changed table count")


def test_seal_replays_g1_across_all_three_seeds(tmp_path) -> None:
    run_dir = tmp_path / "runs"
    run_dir.mkdir()
    for seed, entropy in enumerate((3.1, 3.2, 2.9)):
        row = {
            "schema": "zipf-groupsum-walsh-stage1-run-v1",
            "complete": True,
            "config": {"seed": seed},
            "route_health": {"encoder": {"entropy_bits_mean": entropy}},
            "arithmetic_ledger_per_example": {"shared_butterfly_add_subtracts": 10_400},
        }
        (run_dir / f"run-{seed}.json").write_text(json.dumps(row))
    result = seal(tmp_path)
    assert result["encoder_entropy_bits_mean"] == pytest.approx(3.0666666666666664)
    assert result["g1_all_seeds_pass"] is False
