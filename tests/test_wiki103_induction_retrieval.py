from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from tropnn.tools.wiki103_induction_retrieval import (
    DECODERS,
    build_induction_protocol,
    build_scorer,
    choose_window_starts,
    frozen_hidden_forward,
    ranking_metrics,
    score_candidate_groups,
    summarize,
    validate_completed_cache,
    validate_induction_protocol,
    zero_content_score_luts,
)


def synthetic_tokens() -> torch.Tensor:
    tokens = torch.arange(81, dtype=torch.long) + 1000
    tokens[5:7] = torch.tensor((7, 11))
    tokens[10:12] = torch.tensor((7, 12))
    tokens[40:42] = torch.tensor((7, 11))
    tokens[60:62] = torch.tensor((7, 11))
    return tokens.view(1, -1)


def test_window_starts_are_nonoverlapping_and_in_bounds() -> None:
    starts = choose_window_starts(100, 10_100, context_size=127, count=16)
    assert starts.shape == (16,)
    assert int(starts.min()) >= 100
    assert int(starts.max()) + 128 <= 10_100
    assert torch.all(starts[1:] - starts[:-1] >= 128)


def test_completed_cache_resume_requires_identical_request_and_all_splits(tmp_path) -> None:
    request = {"context_size": 512}
    existing = {"cache_version": 1, "prepare_request": request}
    for split in ("train", "validation", "test"):
        (tmp_path / f"{split}.pt").touch()
    validate_completed_cache(existing, request, tmp_path)
    with pytest.raises(ValueError, match="request mismatch"):
        validate_completed_cache(existing, {"context_size": 256}, tmp_path)
    (tmp_path / "test.pt").unlink()
    with pytest.raises(FileNotFoundError, match="test.pt"):
        validate_completed_cache(existing, request, tmp_path)


def test_induction_protocol_is_causal_exactly_one_positive_and_hard_negative() -> None:
    tokens = synthetic_tokens()
    protocol = build_induction_protocol(tokens, candidate_count=32, max_hard_negatives=16, seed=1729)
    validate_induction_protocol(tokens, protocol)
    assert protocol["queries"] >= 2
    assert torch.all(protocol["relevant_mask"].sum(dim=1) == 1)
    assert torch.all(protocol["candidates"] % 80 < (protocol["query"] % 80)[:, None] - 1)
    assert protocol["hard_negative_mask"].any()
    row = torch.arange(protocol["query"].numel())
    positive = protocol["candidates"][row, protocol["relevant_index"]]
    flat = tokens[:, :-1].reshape(-1)
    successor = tokens[:, 1:].reshape(-1)
    assert torch.equal(flat[positive], flat[protocol["query"]])
    assert torch.equal(successor[positive], protocol["target"])


@pytest.mark.parametrize("decoder", DECODERS)
def test_all_decoders_score_candidate_groups_and_backpropagate(decoder: str) -> None:
    scorer, metadata = build_scorer(decoder, seed=0)
    query = torch.randn(3, 32)
    candidates = torch.randn(3, 5, 32)
    scores = score_candidate_groups(scorer, query, candidates)
    assert scores.shape == (3, 5)
    loss = torch.nn.functional.cross_entropy(scores, torch.tensor((0, 1, 2)))
    loss.backward()
    gradients = [parameter.grad for parameter in scorer.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert metadata["execution_class"]
    parameters = sum(parameter.numel() for parameter in scorer.parameters())
    if decoder == "kendall":
        assert parameters == 2
    elif decoder == "same_table_full":
        assert parameters == 9_217
    elif decoder == "dense_qk":
        assert parameters == 1_025
    else:
        assert 800 < parameters < 1_200


def test_ranking_metrics_measure_relation_and_successor_separately() -> None:
    scores = torch.tensor(((0.9, 0.2, 0.1), (0.3, 0.9, 0.2)))
    relevant = torch.tensor((0, 2))
    values = torch.tensor(((5, 6, 7), (8, 9, 10)))
    target = torch.tensor((5, 9))
    hard = torch.tensor(((False, True, False), (False, True, False)))
    metrics = ranking_metrics(scores, relevant, values, target, hard)
    assert metrics["recall_at_1"] == pytest.approx(0.5)
    assert metrics["recall_at_4"] == pytest.approx(1.0)
    assert metrics["mrr"] == pytest.approx(2.0 / 3.0)
    assert metrics["successor_hit_at_1"] == pytest.approx(1.0)
    assert metrics["hard_negative_top1_rate"] == pytest.approx(0.5)


def test_scoreless_ablation_zeros_every_score_lut() -> None:
    class Branch(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.score_lut = nn.Parameter(torch.ones(2, 3))

    model = nn.Sequential(Branch(), Branch())
    metadata = zero_content_score_luts(model)
    assert metadata["score_lut_tensors"] == 2
    assert metadata["score_lut_l1_before"] == pytest.approx(12.0)
    assert metadata["score_lut_l1_after"] == 0.0
    assert all(torch.count_nonzero(module.score_lut) == 0 for module in model if isinstance(module, Branch))


def test_frozen_hidden_forward_stops_before_unembedding() -> None:
    class Block(nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value + 2

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(8, 4)
            self.embed_proj = None
            self.blocks = nn.ModuleList((Block(), Block()))
            self.final_norm = nn.Identity()

    model = Model()
    tokens = torch.tensor(((1, 2, 3),))
    expected = model.embedding(tokens) + 4
    torch.testing.assert_close(frozen_hidden_forward(model, tokens), expected)


def result_row(decoder: str, seed: int, recall: float) -> dict:
    metrics = {
        "recall_at_1": recall,
        "recall_at_4": min(1.0, recall + 0.2),
        "mrr": min(1.0, recall + 0.1),
        "successor_hit_at_1": min(1.0, recall + 0.05),
        "successor_hit_at_4": min(1.0, recall + 0.25),
        "hard_negative_top1_rate": 0.2,
        "positive_margin": recall - 0.3,
        "listwise_nll": 1.0 - recall,
    }
    return {
        "complete": True,
        "config": {"decoder": decoder, "seed": seed},
        "relation_parameters": {"kendall": 2, "same_table_full": 9217, "root_incidence": 910, "dense_qk": 1025}[decoder],
        "validation": metrics,
        "test": metrics,
    }


def test_summary_enforces_twelve_runs_and_preregistered_gates(tmp_path) -> None:
    result_dir = tmp_path / "runs"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "metadata.json").write_text(
        json.dumps(
            {
                "source": {
                    "boundary": "test boundary",
                }
            }
        )
    )
    recalls = {"kendall": 0.20, "same_table_full": 0.30, "root_incidence": 0.40, "dense_qk": 0.45}
    for decoder, recall in recalls.items():
        for seed in (0, 1, 2):
            path = result_dir / decoder / f"seed{seed}" / "result.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(result_row(decoder, seed, recall)))
    report = tmp_path / "report.md"
    decision = summarize(SimpleNamespace(result_dir=result_dir, cache_dir=cache_dir, out_report=report))
    assert decision["complete"] is True
    assert decision["semantic_gate_passed"] is True
    assert decision["root_vs_kendall"]["mean_delta"] == pytest.approx(0.2)
    assert decision["root_dense_gain_retention"]["retention"] == pytest.approx(0.8)
    assert decision["next_stage"] == "online_warm_start"
    assert report.exists()
