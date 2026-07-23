from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from tropnn.tools.wiki103_hard_induction_retrieval import (
    CANDIDATE_COUNT,
    DECODERS,
    build_hard_induction_protocol,
    build_scorer,
    evaluate_scorer,
    score_candidate_groups,
    summarize,
    validate_hard_induction_protocol,
)


def hard_synthetic_tokens() -> torch.Tensor:
    tokens = torch.arange(121, dtype=torch.long) + 1000
    for index, position in enumerate(range(2, 26, 3)):
        tokens[position] = 7
        tokens[position + 1] = 10 + index
    tokens[80:82] = torch.tensor((7, 10))
    tokens[90:92] = torch.tensor((7, 11))
    tokens[100:102] = torch.tensor((7, 12))
    return tokens.view(1, -1)


def test_hard_protocol_has_only_same_token_distinct_successor_candidates() -> None:
    tokens = hard_synthetic_tokens()
    protocol = build_hard_induction_protocol(tokens)
    validate_hard_induction_protocol(tokens, protocol)

    context_size = tokens.shape[1] - 1
    flat_tokens = tokens[:, :-1].reshape(-1)
    successor = tokens[:, 1:].reshape(-1)
    query = protocol["query"]
    candidates = protocol["candidates"]
    shuffled_query = protocol["shuffled_query"]
    target = protocol["target"]
    values = protocol["candidate_values"]

    assert candidates.shape[1] == CANDIDATE_COUNT
    assert protocol["random_recall_at_1"] == pytest.approx(0.125)
    assert torch.equal(
        flat_tokens[candidates],
        flat_tokens[query][:, None].expand_as(candidates),
    )
    assert torch.all(
        candidates % context_size < (query % context_size)[:, None] - 1
    )
    assert torch.all(values.sort(dim=1).values[:, 1:] != values.sort(dim=1).values[:, :-1])
    assert torch.all((values == target[:, None]).sum(dim=1) == 1)
    assert torch.all(protocol["hard_negative_mask"].sum(dim=1) == 7)
    assert torch.equal(flat_tokens[shuffled_query], flat_tokens[query])
    assert torch.all(successor[shuffled_query] != target)


@pytest.mark.parametrize("decoder", DECODERS)
def test_qualification_scorers_backpropagate(decoder: str) -> None:
    scorer, metadata = build_scorer(decoder, seed=0)
    query = torch.randn(4, 32)
    candidates = torch.randn(4, CANDIDATE_COUNT, 32)
    scores = score_candidate_groups(scorer, query, candidates)
    assert scores.shape == (4, CANDIDATE_COUNT)
    torch.nn.functional.cross_entropy(scores, torch.tensor((0, 1, 2, 3))).backward()
    gradients = [
        parameter.grad
        for parameter in scorer.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert metadata["execution_class"]
    parameters = sum(parameter.numel() for parameter in scorer.parameters())
    assert parameters == {"key_only": 1021, "dense_qk": 1025}[decoder]


def test_key_only_metrics_are_exactly_query_shuffle_invariant() -> None:
    tokens = hard_synthetic_tokens()
    protocol = build_hard_induction_protocol(tokens)
    split = {
        "coordinates": torch.randn(tokens.shape[1] - 1, 32),
        "query": protocol["query"],
        "shuffled_query": protocol["shuffled_query"],
        "candidates": protocol["candidates"],
        "relevant_index": protocol["relevant_index"],
        "target": protocol["target"],
        "candidate_values": protocol["candidate_values"],
        "hard_negative_mask": protocol["hard_negative_mask"],
    }
    scorer, _ = build_scorer("key_only", seed=0)
    base = evaluate_scorer(
        scorer,
        split,
        torch.device("cpu"),
        batch_size=16,
    )
    shuffled = evaluate_scorer(
        scorer,
        split,
        torch.device("cpu"),
        batch_size=16,
        shuffled_query=True,
    )
    assert base == shuffled


def metrics(recall: float) -> dict[str, float]:
    return {
        "recall_at_1": recall,
        "recall_at_4": min(1.0, recall + 0.4),
        "mrr": min(1.0, recall + 0.2),
        "successor_hit_at_1": recall,
        "successor_hit_at_4": min(1.0, recall + 0.4),
        "hard_negative_top1_rate": 1.0 - recall,
        "positive_margin": recall - 0.2,
        "listwise_nll": 2.0 - recall,
    }


def result_row(
    decoder: str,
    seed: int,
    recall: float,
    shuffled_recall: float,
) -> dict:
    return {
        "complete": True,
        "config": {"decoder": decoder, "seed": seed},
        "relation_parameters": {"key_only": 1021, "dense_qk": 1025}[decoder],
        "test": metrics(recall),
        "test_query_shuffle": metrics(shuffled_recall),
        "validation": metrics(recall),
        "validation_query_shuffle": metrics(shuffled_recall),
    }


def write_summary_fixture(
    tmp_path,
    *,
    key_recall: float,
    dense_recall: float,
    shuffled_recall: float,
):
    result_dir = tmp_path / "runs"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "metadata.json").write_text(
        json.dumps(
            {
                "cache_fingerprint": "cache-hash",
                "protocol": {
                    "query_counts": {
                        "train": 100,
                        "validation": 30,
                        "test": 30,
                    }
                },
            }
        )
    )
    for decoder, recall, shuffled in (
        ("key_only", key_recall, key_recall),
        ("dense_qk", dense_recall, shuffled_recall),
    ):
        for seed in (0, 1, 2):
            path = result_dir / decoder / f"seed{seed}" / "result.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(result_row(decoder, seed, recall, shuffled)))
    report = tmp_path / "report.md"
    return summarize(
        SimpleNamespace(
            result_dir=result_dir,
            cache_dir=cache_dir,
            out_report=report,
        )
    )


def test_summary_passes_all_three_dense_qualification_gates(tmp_path) -> None:
    decision = write_summary_fixture(
        tmp_path,
        key_recall=0.15,
        dense_recall=0.30,
        shuffled_recall=0.16,
    )
    assert decision["complete"] is True
    assert decision["dense_qualification_passed"] is True
    assert decision["gates"]["dense_absolute"]["passed"] is True
    assert decision["gates"]["dense_over_key_only"]["passed"] is True
    assert decision["gates"]["query_shuffle_gain_removal"]["passed"] is True
    assert decision["next_stage"] == "compare_ordinal_and_no_gemm_kernels"


def test_summary_stops_when_dense_absolute_recall_is_weak(tmp_path) -> None:
    decision = write_summary_fixture(
        tmp_path,
        key_recall=0.13,
        dense_recall=0.24,
        shuffled_recall=0.13,
    )
    assert decision["complete"] is True
    assert decision["dense_qualification_passed"] is False
    assert decision["gates"]["dense_absolute"]["passed"] is False
    assert decision["next_stage"] == "stop_relation_kernel_search"
