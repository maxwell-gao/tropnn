from __future__ import annotations

import torch
from tropnn.tools.s4_global_rank_quantization_probe import (
    SPECS,
    fit_symmetric_scale,
    grouped_lut_score_matrix,
    level_tensor,
    quantize_embeddings,
    topk_retention,
)


def test_quantizers_use_exact_requested_alphabets() -> None:
    values = torch.linspace(-3.0, 3.0, 97).view(-1, 1).repeat(1, 12)
    for spec in SPECS:
        levels = level_tensor(spec, values.device)
        scale = fit_symmetric_scale(values, levels)
        indices, _ = quantize_embeddings(values, spec, scale)
        assert indices.min() >= 0
        assert indices.max() < len(spec.levels)
        assert set(levels[indices].unique().tolist()).issubset(set(float(level) for level in spec.levels))


def test_grouped_lut_exactly_matches_quantized_dot_for_every_precision() -> None:
    generator = torch.Generator().manual_seed(7)
    query = torch.randn(7, 12, generator=generator)
    key = torch.randn(11, 12, generator=generator)
    bias = torch.tensor(0.3)
    target_scale = torch.tensor(1.7)
    tables = 16
    for spec in SPECS:
        levels = level_tensor(spec, query.device)
        query_scale = fit_symmetric_scale(query, levels)
        key_scale = fit_symmetric_scale(key, levels)
        query_indices, query_reconstruction = quantize_embeddings(query, spec, query_scale)
        key_indices, key_reconstruction = quantize_embeddings(key, spec, key_scale)
        actual, relation, integer_score = grouped_lut_score_matrix(
            query_indices,
            key_indices,
            spec,
            query_scale,
            key_scale,
            tables=tables,
            bias=bias,
            target_scale=target_scale,
        )
        expected = (query_reconstruction @ key_reconstruction.T / tables + bias) * target_scale
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        assert integer_score.dtype == torch.int32
        assert relation.numel() == len(spec.levels) ** (2 * spec.group_size)


def test_deployment_lut_shapes_and_reads_are_bounded() -> None:
    expected = {
        "binary": (3, 256),
        "ternary": (4, 729),
        "int2": (4, 4096),
        "int4": (12, 256),
    }
    for spec in SPECS:
        pair_reads = 12 // spec.group_size
        entries = len(spec.levels) ** (2 * spec.group_size)
        assert (pair_reads, entries) == expected[spec.name]


def test_topk_retention_is_one_for_identical_rankings() -> None:
    score = torch.tensor([[1.0, 4.0, 3.0, 2.0], [0.0, -1.0, 2.0, 3.0]])
    assert topk_retention(score, score.clone(), 2) == 1.0


def test_topk_retention_uses_stable_key_index_tie_break() -> None:
    reference = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    tied = torch.ones_like(reference)
    assert topk_retention(reference, tied, 2) == 1.0
