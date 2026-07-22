from __future__ import annotations

import json
from types import SimpleNamespace

import torch
from tropnn.layers import BalancedS4Router, RootIncidenceKernel
from tropnn.tools.emnist_pair_relation_quantization import (
    QUANTIZATION_MODES,
    score_integer_cache_pairs,
    summarize_quantization,
    topk_overlap,
)


def test_batched_integer_cache_scoring_matches_quantized_kernel() -> None:
    router = BalancedS4Router(input_dim=8, tables=4, coverage=2, seed=3)
    float_kernel = RootIncidenceKernel(router, seed=5)
    quantized = float_kernel.quantized(router.roots, "int2")
    coordinates = torch.randn(23, 8, generator=torch.Generator().manual_seed(7))
    features = router.route(coordinates)
    cache = quantized.build_cache(features.roots)
    query = torch.tensor((0, 3, 7, 2, 9, 17, 22))
    key = torch.tensor((4, 1, 8, 6, 12, 19, 20))
    logits, integers = score_integer_cache_pairs(
        quantized,
        cache,
        query,
        key,
        symmetry="symmetric",
        batch_size=3,
    )
    forward = quantized.hard_score(router.route(coordinates[query]), router.route(coordinates[key]))
    reverse = quantized.hard_score(router.route(coordinates[key]), router.route(coordinates[query]))
    torch.testing.assert_close(logits, 0.5 * (forward + reverse))
    assert integers.dtype == torch.int32


def test_topk_overlap_uses_set_overlap_with_stable_ranking() -> None:
    reference = torch.tensor(((4.0, 3.0, 2.0, 1.0),))
    prediction = torch.tensor(((4.0, 2.0, 3.0, 1.0),))
    assert topk_overlap(reference, prediction, k=2) == 0.5


def test_quantization_summary_requires_all_36_checkpoint_runs(tmp_path) -> None:
    result_dir = tmp_path / "quant"
    index = 0
    for task, split_mode in (("same_class", "object"), ("same_class", "class"), ("digit_greater", "object")):
        for payload_mode in ("float", "binary01"):
            for objective in ("relation_only", "relation_aux"):
                for seed in range(3):
                    variants = []
                    for mode in QUANTIZATION_MODES:
                        variants.append(
                            {
                                "mode": mode,
                                "storage_bits": 1 if mode == "binary" else 2 if mode in {"ternary", "int2"} else 4,
                                "primary_metric": "random_recall_at_16" if task == "same_class" else "pair_macro_roc_auc",
                                "full_primary": 0.8,
                                "quantized_primary": 0.7,
                                "raw_retention": 0.875,
                                "chance_adjusted_retention": 0.85,
                                "coefficient_mse": 0.01,
                                "packed_coefficient_bytes": 100,
                                "random_full_top16_overlap": 0.75 if task == "same_class" else float("nan"),
                                "execution": {
                                    "cached_integer_pairs_per_second": 2e6,
                                    "direct_integer_pairs_per_second": 1e6,
                                    "float_cached_pairs_per_second": 1.5e6,
                                    "integer_cache_objects_per_second": 3e6,
                                    "integer_cache_bytes_per_object": 450,
                                    "direct_cached_integer_exact": True,
                                },
                            }
                        )
                    result = {
                        "complete": True,
                        "source_identity": {
                            "config": {
                                "task": task,
                                "split_mode": split_mode,
                                "payload_mode": payload_mode,
                                "objective": objective,
                                "seed": seed,
                            }
                        },
                        "variants": variants,
                    }
                    path = result_dir / f"run{index}" / "result.json"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(result))
                    index += 1

    decision = summarize_quantization(SimpleNamespace(result_dir=result_dir, out_report=tmp_path / "quantization.md"))
    assert decision["complete"]
    assert decision["checkpoint_runs"] == 36
    assert decision["quantized_variants"] == 144
    assert decision["all_cached_integer_paths_exact"]
