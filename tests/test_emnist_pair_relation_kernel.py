from __future__ import annotations

import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
from tropnn.tools.emnist_pair_relation_kernel import (
    CLASS_TEST,
    CLASS_TRAIN,
    CLASS_VALID,
    PairExperimentConfig,
    PairRelationModel,
    TensorSplit,
    binary_payload_is_valid,
    build_retrieval_set,
    check_frontier_gate,
    digit_relation_metrics,
    improvement_retention,
    pair_metrics,
    paired_metric_gate,
    retrieval_metrics,
    sample_pair_indices,
    sample_training_pair_indices,
    select_learning_rates,
    stratified_half_split,
    validate_completed_config,
)


def test_class_holdout_lists_are_disjoint_and_complete() -> None:
    train, validation, test = set(CLASS_TRAIN), set(CLASS_VALID), set(CLASS_TEST)
    assert not train & validation
    assert not train & test
    assert not validation & test
    assert train | validation | test == set(range(47))


def test_stratified_object_split_is_disjoint_and_balanced() -> None:
    labels = torch.arange(4).repeat_interleave(10)
    images = torch.arange(40).float().view(40, 1)
    validation, test = stratified_half_split(images, labels, 7)
    assert set(validation.images.flatten().tolist()).isdisjoint(test.images.flatten().tolist())
    assert torch.equal(torch.bincount(validation.labels), torch.full((4,), 5))
    assert torch.equal(torch.bincount(test.labels), torch.full((4,), 5))


def test_same_class_pairs_are_balanced_and_never_self_pairs() -> None:
    labels = torch.arange(5).repeat_interleave(20)
    pairs = sample_pair_indices(labels, "same_class", 1000, 11)
    assert pairs.target.sum() == 500
    assert bool((pairs.query != pairs.key).all())
    assert torch.equal(labels[pairs.query] == labels[pairs.key], pairs.target.bool())


def test_default_training_epoch_uses_every_object_once_per_pair_polarity() -> None:
    labels = torch.arange(5).repeat_interleave(20)
    pairs = sample_training_pair_indices(labels, "same_class", 12)
    assert pairs.target.numel() == 2 * labels.numel()
    assert torch.equal(torch.bincount(pairs.query), torch.full((labels.numel(),), 2))
    positive_queries = torch.bincount(pairs.query[pairs.target.bool()], minlength=labels.numel())
    negative_queries = torch.bincount(pairs.query[~pairs.target.bool()], minlength=labels.numel())
    assert torch.equal(positive_queries, torch.ones(labels.numel(), dtype=torch.long))
    assert torch.equal(negative_queries, torch.ones(labels.numel(), dtype=torch.long))
    assert bool((pairs.query != pairs.key).all())
    assert torch.equal(labels[pairs.query] == labels[pairs.key], pairs.target.bool())


def test_digit_relation_pairs_match_label_order() -> None:
    labels = torch.arange(10).repeat_interleave(20)
    pairs = sample_pair_indices(labels, "digit_greater", 1000, 13)
    assert pairs.target.sum() == 500
    assert bool((labels[pairs.query] != labels[pairs.key]).all())
    assert torch.equal(labels[pairs.query] > labels[pairs.key], pairs.target.bool())


def test_retrieval_set_has_exact_positive_count() -> None:
    labels = torch.arange(4).repeat_interleave(30)
    images = torch.randn(120, 8, generator=torch.Generator().manual_seed(17))
    retrieval = build_retrieval_set(TensorSplit(images, labels), queries=8, candidates=24, positives=4, seed=19)
    assert retrieval.candidates.shape == (8, 24)
    assert torch.equal(retrieval.relevant.sum(dim=1), torch.full((8,), 4))


def test_metrics_have_known_perfect_values() -> None:
    target = torch.tensor([0.0, 0.0, 1.0, 1.0])
    prediction = torch.tensor([-3.0, -2.0, 2.0, 3.0])
    pair = pair_metrics(target, prediction)
    assert pair["roc_auc"] == 1.0
    assert pair["pr_auc"] == 1.0
    assert pair["accuracy"] == 1.0
    relevant = torch.tensor([[True, True, False, False]])
    retrieval = retrieval_metrics(relevant, torch.tensor([[4.0, 3.0, 2.0, 1.0]]), top_k=2)
    assert retrieval["recall_at_16"] == 1.0
    assert retrieval["hit_at_1"] == 1.0
    assert retrieval["mrr"] == 1.0


def _config(payload_mode: str = "float", decoder: str = "root_incidence") -> PairExperimentConfig:
    return PairExperimentConfig(
        task="same_class",
        split_mode="object",
        decoder=decoder,
        payload_mode=payload_mode,
        objective="relation_only",
        seed=23,
        epochs=1,
        batch_size=8,
        encoder_lr=1e-3,
        relation_lr=1e-3,
        auxiliary_weight=0.25,
        relation_dim=8,
        relation_tables=4,
        relation_coverage=2,
        encoder_depth=1,
        encoder_tables=2,
        encoder_comparisons=2,
        train_pairs_per_epoch=16,
        eval_pairs=32,
        retrieval_queries=4,
        retrieval_candidates=16,
        retrieval_positives=2,
        hard_reservoir=32,
        data_split_seed=1729,
        max_train_examples=0,
        max_eval_examples=0,
    )


def test_small_model_is_end_to_end_trainable() -> None:
    model = PairRelationModel(_config(), auxiliary_classes=4)
    images = torch.randn(12, 784, generator=torch.Generator().manual_seed(29))
    query, _ = model.encoder(images[:6])
    key, _ = model.encoder(images[6:])
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model.score(query, key), torch.arange(6).remainder(2).float())
    loss.backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in model.encoder.parameters())
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in model.relation.parameters())


def test_small_end_to_end_model_reduces_a_fixed_pair_loss() -> None:
    torch.manual_seed(61)
    model = PairRelationModel(_config(decoder="dense_qk_r4"), auxiliary_classes=4)
    images = torch.randn(16, 784, generator=torch.Generator().manual_seed(67))
    query_index = torch.arange(8)
    key_index = torch.arange(8, 16)
    target = torch.arange(8).remainder(2).float()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

    def loss_value() -> torch.Tensor:
        coordinates, _ = model.encoder(images)
        return torch.nn.functional.binary_cross_entropy_with_logits(
            model.score(coordinates[query_index], coordinates[key_index]),
            target,
        )

    initial = float(loss_value().item())
    for _ in range(30):
        loss = loss_value()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    final = float(loss_value().item())
    assert final < 0.9 * initial


def test_binary_encoder_materializes_only_binary_payloads() -> None:
    model = PairRelationModel(_config("binary01", "dense_qk_r4"), auxiliary_classes=4)
    assert binary_payload_is_valid(model)


def test_joint_pair_controls_match_preregistered_relation_budgets() -> None:
    assert PairRelationModel(_config(decoder="jointpair_t16"), auxiliary_classes=4).relation_parameters == 1025
    assert PairRelationModel(_config(decoder="jointpair_t144"), auxiliary_classes=4).relation_parameters == 9217


def test_digit_metrics_report_macro_query_auc_and_adjacent_accuracy() -> None:
    labels = torch.arange(5).repeat_interleave(8)
    pairs = sample_pair_indices(labels, "digit_greater", 2000, 31)
    prediction = (labels[pairs.query] - labels[pairs.key]).float()
    metrics = digit_relation_metrics(labels, pairs, prediction)
    assert metrics["macro_roc_auc"] == 1.0
    assert metrics["adjacent_accuracy"] == 1.0


def test_completed_result_config_mismatch_is_rejected(tmp_path) -> None:
    config = _config()
    validate_completed_config({"config": asdict(config)}, config, tmp_path / "result.json")
    changed = asdict(config)
    changed["eval_pairs"] = 999
    with pytest.raises(ValueError, match="config mismatch"):
        validate_completed_config({"config": changed}, config, tmp_path / "result.json")


def _fake_result(decoder: str, payload: str, encoder_lr: float, relation_lr: float, score: float, seed: int = 0) -> dict[str, object]:
    return {
        "complete": True,
        "config": {
            "decoder": decoder,
            "payload_mode": payload,
            "encoder_lr": encoder_lr,
            "relation_lr": relation_lr,
            "task": "same_class",
            "split_mode": "object",
            "objective": "relation_only",
            "seed": seed,
        },
        "best_validation_selection_metric": score,
        "validation": {"pair_roc_auc": 0.95, "random_recall_at_16": 0.60},
        "test": {"pair_roc_auc": 0.95, "random_recall_at_16": 0.60},
    }


def test_learning_rate_selection_averages_both_diagnostics_and_tie_breaks_low(tmp_path) -> None:
    for payload in ("float", "binary01"):
        for encoder_lr, score in ((0.001, 0.6), (0.003, 0.6)):
            for decoder in ("root_incidence", "dense_qk_r16"):
                path = tmp_path / payload / str(encoder_lr) / decoder / "result.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(_fake_result(decoder, payload, encoder_lr, 0.003, score)))
    output = tmp_path / "selected.json"
    select_learning_rates(SimpleNamespace(result_dir=tmp_path, output=output))
    selected = json.loads(output.read_text())
    assert selected["float"]["encoder_lr"] == 0.001
    assert selected["binary01"]["encoder_lr"] == 0.001
    assert output.with_suffix(".env").exists()


def test_frontier_gate_requires_three_seeds_and_accepts_one_dense_control(tmp_path) -> None:
    for decoder in ("dense_qk_r16", "concat_mlp"):
        for seed in range(3):
            result = _fake_result(decoder, "float", 0.001, 0.003, 0.6, seed)
            if decoder == "concat_mlp":
                result["validation"] = {"pair_roc_auc": 0.7, "random_recall_at_16": 0.2}
                result["test"] = {"pair_roc_auc": 0.7, "random_recall_at_16": 0.2}
            path = tmp_path / decoder / f"seed{seed}" / "result.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result))
    output = tmp_path / "gate.json"
    decision = check_frontier_gate(SimpleNamespace(result_dir=tmp_path, output=output))
    assert decision["passed"]


def test_paired_gate_and_random_adjusted_retention_use_all_three_seeds() -> None:
    left = ("same_class", "class", "float", "relation_only", "left")
    right = ("same_class", "class", "float", "relation_only", "right")
    indexed = {
        left: {seed: {"random_recall_at_16": 0.55 + 0.01 * seed} for seed in range(3)},
        right: {seed: {"random_recall_at_16": 0.45 + 0.01 * seed} for seed in range(3)},
    }
    gate = paired_metric_gate(indexed, left, right, "random_recall_at_16")
    assert gate["complete"] and gate["passed"] and gate["all_seed_same_sign"]
    retention = improvement_retention(indexed, right, left, "random_recall_at_16", 16.0 / 512.0, 0.7)
    assert retention["complete"] and retention["passed"]
