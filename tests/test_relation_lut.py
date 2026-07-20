from __future__ import annotations

import torch

from tropnn import ComparisonRelationLUT
from tropnn.tools.comparison_relation_probe import SerialRelationScorer
from tropnn.tools.relation_attention_probe import AttentionConfig, EpisodeFactory, evaluate, make_model


def test_relation_lookup_matches_explicit_bank_sum() -> None:
    layer = ComparisonRelationLUT(
        8,
        num_banks=3,
        num_codes=4,
        relation_rank=2,
        relation_mode="free",
        seed=3,
    )
    query = torch.randn(2, 5, 8)
    key = torch.randn(2, 7, 8)
    output, query_route, key_route = layer(query, key, return_routes=True)
    expected = torch.zeros_like(output)
    for batch in range(2):
        for q_index in range(5):
            for k_index in range(7):
                for bank in range(layer.num_banks):
                    expected[batch, q_index, k_index] += layer.relation[
                        bank,
                        query_route.indices[batch, q_index, bank],
                        key_route.indices[batch, k_index, bank],
                    ]
    expected /= layer.num_banks**0.5
    torch.testing.assert_close(output, expected)


def test_materialized_free_relation_recovers_cross_gram_exactly() -> None:
    constrained = ComparisonRelationLUT(
        8,
        num_banks=4,
        num_codes=8,
        relation_rank=4,
        relation_mode="constrained_gram",
        seed=5,
    )
    free = ComparisonRelationLUT.free_from_constrained(constrained)
    query = torch.randn(3, 6, 8)
    key = torch.randn(3, 9, 8)
    torch.testing.assert_close(free(query, key), constrained(query, key), atol=1e-6, rtol=1e-6)


def test_relation_quantizers_have_only_declared_symbols() -> None:
    source = ComparisonRelationLUT(
        8,
        num_banks=2,
        num_codes=8,
        relation_rank=4,
        relation_mode="constrained_gram",
        seed=7,
    )
    ternary = ComparisonRelationLUT.free_from_constrained(source, quantization="ternary")
    binary = ComparisonRelationLUT.free_from_constrained(source, quantization="binary")
    for layer, symbols in ((ternary, 3), (binary, 2)):
        payload = layer.materialized_relation(quantized=True)
        for bank in range(layer.num_banks):
            assert torch.unique(payload[bank]).numel() <= symbols


def test_route_threshold_surrogate_receives_finite_gradients() -> None:
    layer = ComparisonRelationLUT(
        8,
        num_banks=3,
        num_codes=4,
        relation_rank=2,
        relation_mode="free",
        train_thresholds=True,
        seed=11,
    )
    query = torch.randn(4, 2, 8)
    key = torch.randn(4, 3, 8)
    layer(query, key).square().mean().backward()
    assert layer.query_router.thresholds.grad is not None
    assert layer.key_router.thresholds.grad is not None
    assert torch.isfinite(layer.query_router.thresholds.grad).all()
    assert torch.isfinite(layer.key_router.thresholds.grad).all()


def test_route_calibration_balances_non_degenerate_bits() -> None:
    layer = ComparisonRelationLUT(16, num_banks=4, num_codes=8, relation_rank=4, seed=13)
    samples = torch.randn(4097, 16)
    layer.calibrate_routes(samples, samples)
    query_route, key_route = layer.routes(samples, samples)
    for margins in (query_route.margins, key_route.margins):
        positive = (margins > 0).float().mean(dim=0)
        torch.testing.assert_close(positive, torch.full_like(positive, 0.5), atol=0.01, rtol=0.0)


def test_route_code_matches_explicit_comparisons() -> None:
    layer = ComparisonRelationLUT(12, num_banks=3, num_codes=8, relation_rank=4, seed=17)
    samples = torch.randn(19, 12)
    route, _ = layer.routes(samples, samples)
    anchors = layer.query_router.anchors
    explicit = torch.zeros(19, layer.num_banks, dtype=torch.long)
    for bank in range(layer.num_banks):
        for comparison in range(layer.comparisons):
            a, b = anchors[bank, comparison]
            margin = samples[:, a] - samples[:, b] - layer.query_router.thresholds[bank, comparison]
            explicit[:, bank] += (margin > 0).long() * (1 << comparison)
    assert torch.equal(route.indices, explicit)


def test_only_selected_relation_rows_receive_payload_gradient() -> None:
    layer = ComparisonRelationLUT(
        8,
        num_banks=2,
        num_codes=4,
        relation_rank=2,
        relation_mode="free",
        train_thresholds=False,
        seed=19,
    )
    query = torch.randn(11, 8)
    key = torch.randn(11, 8)
    query_route, key_route = layer.routes(query, key)
    layer.score_aligned(query, key).sum().backward()
    expected = torch.zeros_like(layer.relation, dtype=torch.bool)
    for bank in range(layer.num_banks):
        expected[bank, query_route.indices[:, bank], key_route.indices[:, bank]] = True
    assert torch.equal(layer.relation.grad != 0, expected)


def test_boolean_and_additive_masks_preserve_unmasked_scores() -> None:
    layer = ComparisonRelationLUT(8, num_banks=2, num_codes=4, relation_rank=2, seed=23)
    query = torch.randn(1, 2, 8)
    key = torch.randn(1, 3, 8)
    reference = layer(query, key)
    allowed = torch.tensor([[[True, False, True], [False, True, True]]])
    boolean = layer(query, key, mask=allowed)
    additive_mask = torch.where(allowed, torch.zeros_like(reference), torch.full_like(reference, -1000.0))
    additive = layer(query, key, mask=additive_mask)
    torch.testing.assert_close(boolean[allowed], reference[allowed])
    torch.testing.assert_close(additive[allowed], reference[allowed])
    assert torch.all(boolean[~allowed] < -1e20)
    assert torch.all(additive[~allowed] < -900)


def test_seed_reproduces_routers_and_payloads() -> None:
    first = ComparisonRelationLUT(16, num_banks=4, num_codes=8, relation_rank=4, seed=29)
    second = ComparisonRelationLUT(16, num_banks=4, num_codes=8, relation_rank=4, seed=29)
    assert torch.equal(first.query_router.anchors, second.query_router.anchors)
    assert torch.equal(first.key_router.anchors, second.key_router.anchors)
    assert torch.equal(first.relation, second.relation)


def test_zero_conditioned_serial_score_is_layer_sum() -> None:
    layers = [
        ComparisonRelationLUT(8, num_banks=2, num_codes=4, relation_rank=2, seed=31 + depth)
        for depth in range(3)
    ]
    serial = SerialRelationScorer(layers, conditioning=False)
    query = torch.randn(13, 8)
    key = torch.randn(13, 8)
    expected = sum(layer.score_aligned(query, key) for layer in layers)
    torch.testing.assert_close(serial.score_aligned(query, key), expected)


def test_dense_multihop_control_is_solvable_without_noise(tmp_path) -> None:
    config = AttentionConfig(
        out_dir=tmp_path,
        variant="dense",
        device="cpu",
        input_dim=16,
        vocabulary=128,
        train_vocabulary=96,
        memories=16,
        depth=2,
        task_hops=2,
        query_noise=0.0,
        eval_noises=(0.0,),
        batch_size=32,
        seed=37,
    )
    factory = EpisodeFactory(config, torch.device("cpu"))
    metrics = evaluate(make_model(config, factory), factory, batches=4)
    assert metrics["seen_noise0p0_task_accuracy"] > 0.9
    assert metrics["unseen_noise0p0_task_accuracy"] > 0.9
