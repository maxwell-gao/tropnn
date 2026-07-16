from __future__ import annotations

import torch

from tropnn.tools.dynamic_pc_memory_probe import (
    DotProductLMS,
    DynamicPCMemory,
    PairwiseRouter,
    ProbeConfig,
    RouteConfig,
    episode,
    run_probe,
)


def test_dynamic_pc_memory_read_after_write_and_collision_kernel_are_exact() -> None:
    keys, values = episode(seed=3, facts=2, input_dim=16, value_dim=8)
    router = PairwiseRouter(16, RouteConfig(4, 3), seed=7)
    memory = DynamicPCMemory(router, value_dim=8)

    _, after = memory.write(keys[0], values[0], eta=1.0)
    torch.testing.assert_close(after, values[0], atol=1e-12, rtol=0.0)

    collision = router.paired_similarity(keys[1:2], keys[0:1]).item()
    torch.testing.assert_close(memory.read(keys[1:2]).squeeze(0), collision * values[0], atol=1e-12, rtol=0.0)


def test_dot_lms_read_after_write_and_dot_kernel_are_exact() -> None:
    keys, values = episode(seed=5, facts=2, input_dim=16, value_dim=8)
    memory = DotProductLMS(input_dim=16, value_dim=8)

    _, after = memory.write(keys[0], values[0], eta=1.0)
    torch.testing.assert_close(after, values[0], atol=1e-12, rtol=0.0)

    similarity = torch.dot(keys[1], keys[0])
    torch.testing.assert_close(memory.read(keys[1:2]).squeeze(0), similarity * values[0], atol=1e-12, rtol=0.0)


def test_tiny_cpu_probe_emits_all_four_experiments(tmp_path) -> None:
    summary = run_probe(
        ProbeConfig(
            out_dir=tmp_path,
            input_dim=8,
            value_dim=8,
            routes=(RouteConfig(2, 2),),
            facts=(1, 4),
            noise=(0.0, 0.25),
            seeds=(0,),
            eta=1.0,
            concept_facts=4,
        )
    )

    assert {row["model"] for row in summary["identity"]} == {"dot_lms", "pc_t2_c2"}
    assert {row["model"] for row in summary["capacity"]} == {"dot_lms", "pc_t2_c2"}
    assert {row["model"] for row in summary["concept_drift"]} == {"dot_lms", "pc_t2_c2"}
    for name in (
        "identity.csv",
        "capacity_noise.csv",
        "capacity_noise_summary.csv",
        "concept_drift.csv",
        "concept_drift_summary.csv",
        "summary.json",
        "report.md",
    ):
        assert (tmp_path / name).is_file()
