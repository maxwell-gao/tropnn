"""Post-training relation-coefficient quantization for real EMNIST pair kernels."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch import Tensor

from tropnn.layers import IntegerRootCache, QuantizedRootIncidenceKernel, RootIncidenceKernel
from tropnn.tools.emnist_pair_relation_kernel import (
    PairExperimentConfig,
    PairIndices,
    PairRelationModel,
    TaskSplits,
    TensorSplit,
    _sync,
    build_retrieval_set,
    digit_relation_metrics,
    encode_split,
    index_fingerprint,
    load_task_splits,
    pair_metrics,
    retrieval_metrics,
    sample_pair_indices,
)

QUANTIZATION_MODES = ("binary", "ternary", "int2", "int4")


def mean_sem(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0


def _slice_cache(cache: IntegerRootCache, index: Tensor) -> IntegerRootCache:
    return IntegerRootCache(cache.signs[index], cache.transformed[index])


@torch.no_grad()
def score_integer_cache_pairs(
    kernel: QuantizedRootIncidenceKernel,
    cache: IntegerRootCache,
    query: Tensor,
    key: Tensor,
    *,
    symmetry: str,
    batch_size: int,
) -> tuple[Tensor, Tensor]:
    logits: list[Tensor] = []
    integers: list[Tensor] = []
    device = cache.signs.device
    for start in range(0, query.numel(), batch_size):
        stop = min(start + batch_size, query.numel())
        query_index = query[start:stop].to(device)
        key_index = key[start:stop].to(device)
        query_cache = _slice_cache(cache, query_index)
        key_cache = _slice_cache(cache, key_index)
        integer, _ = kernel.integer_score_from_cache(query_cache, key_cache, symmetry=symmetry)
        logits.append(kernel.score_from_cache(query_cache, key_cache, symmetry=symmetry).cpu())
        integers.append(integer.cpu())
    return torch.cat(logits), torch.cat(integers)


@torch.no_grad()
def score_float_cache_pairs(
    kernel: RootIncidenceKernel,
    roots: Tensor,
    transformed: Tensor,
    query: Tensor,
    key: Tensor,
    *,
    symmetry: str,
    batch_size: int,
) -> Tensor:
    rows: list[Tensor] = []
    device = roots.device
    for start in range(0, query.numel(), batch_size):
        stop = min(start + batch_size, query.numel())
        query_index = query[start:stop].to(device)
        key_index = key[start:stop].to(device)
        rows.append(
            kernel.score_from_cache(
                roots[query_index],
                roots[key_index],
                transformed[query_index],
                transformed[key_index],
                symmetry=symmetry,
            ).cpu()
        )
    return torch.cat(rows)


def topk_overlap(reference: Tensor, prediction: Tensor, k: int = 16) -> float:
    reference_top = torch.argsort(reference, dim=-1, descending=True, stable=True)[..., :k]
    prediction_top = torch.argsort(prediction, dim=-1, descending=True, stable=True)[..., :k]
    matches = (prediction_top.unsqueeze(-1) == reference_top.unsqueeze(-2)).any(dim=-1)
    return float(matches.float().mean().item())


def _truncate_splits(splits: TaskSplits, config: PairExperimentConfig) -> TaskSplits:
    train = splits.train
    validation = splits.validation
    test = splits.test
    if config.max_train_examples > 0:
        train = TensorSplit(train.images[: config.max_train_examples], train.labels[: config.max_train_examples])
    if config.max_eval_examples > 0:
        validation = TensorSplit(validation.images[: config.max_eval_examples], validation.labels[: config.max_eval_examples])
        test = TensorSplit(test.images[: config.max_eval_examples], test.labels[: config.max_eval_examples])
    return TaskSplits(train, validation, test, splits.auxiliary_classes)


@torch.no_grad()
def benchmark_quantized_kernel(
    kernel: QuantizedRootIncidenceKernel,
    float_kernel: RootIncidenceKernel,
    roots: Tensor,
    *,
    symmetry: str,
    pairs: int = 8192,
    warmups: int = 3,
    iterations: int = 10,
) -> dict[str, float | int | str]:
    device = roots.device
    pair_count = min(pairs, roots.shape[0])
    query_roots = roots[:pair_count]
    key_roots = roots[-pair_count:]
    query_cache = kernel.build_cache(query_roots)
    key_cache = kernel.build_cache(key_roots)
    float_query = float_kernel.transform_roots(query_roots)
    float_key = float_kernel.transform_roots(key_roots)

    def direct() -> Tensor:
        forward = kernel.hard_integer_score(query_roots, key_roots)
        if symmetry == "none":
            return forward
        reverse = kernel.hard_integer_score(key_roots, query_roots)
        return forward + reverse if symmetry == "symmetric" else forward - reverse

    def cached() -> Tensor:
        return kernel.integer_score_from_cache(query_cache, key_cache, symmetry=symmetry)[0]

    def float_cached() -> Tensor:
        return float_kernel.score_from_cache(
            query_roots,
            key_roots,
            float_query,
            float_key,
            symmetry=symmetry,
        )

    def build_cache() -> IntegerRootCache:
        return kernel.build_cache(query_roots)

    def time_call(function, units: int) -> float:
        for _ in range(warmups):
            function()
        _sync(device)
        started = time.perf_counter()
        for _ in range(iterations):
            function()
        _sync(device)
        return iterations * units / max(time.perf_counter() - started, 1e-30)

    direct_integer = direct()
    cached_integer = cached()
    if not torch.equal(direct_integer, cached_integer):
        raise RuntimeError(f"{kernel.mode} cached integer accumulator differs from the direct path")
    return {
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "benchmark_pairs": pair_count,
        "benchmark_warmups": warmups,
        "benchmark_iterations": iterations,
        "direct_integer_pairs_per_second": time_call(direct, pair_count),
        "cached_integer_pairs_per_second": time_call(cached, pair_count),
        "float_cached_pairs_per_second": time_call(float_cached, pair_count),
        "integer_cache_objects_per_second": time_call(build_cache, pair_count),
        "direct_relation_code_reads_per_pair": int(kernel.codes.numel()),
        "cache_relation_code_reads_per_object": int(kernel.codes.numel()),
        "cached_pair_integer_add_sub": int(kernel.roots),
        "cached_pair_sign_reads": int(kernel.roots),
        "cached_pair_int32_reads": int(kernel.roots),
        "integer_cache_bytes_per_object": int(
            kernel.roots * (torch.empty((), dtype=torch.int8).element_size() + torch.empty((), dtype=torch.int32).element_size())
        ),
        "float_cache_bytes_per_object": int(2 * kernel.roots * torch.empty((), dtype=torch.float32).element_size()),
        "integer_accumulator_min": int(cached_integer.min().item()),
        "integer_accumulator_max": int(cached_integer.max().item()),
        "direct_cached_integer_exact": True,
        "ranking_path": "int8 signs + int8 coefficients -> int32 cached add/sub accumulation",
    }


def _metric_bundle(task: str, labels: Tensor, pairs: PairIndices, prediction: Tensor) -> dict[str, float]:
    metrics = {f"pair_{key}": value for key, value in pair_metrics(pairs.target, prediction).items()}
    if task == "digit_greater":
        metrics.update({f"pair_{key}": value for key, value in digit_relation_metrics(labels, pairs, prediction).items()})
    return metrics


def run_quantization(args: argparse.Namespace) -> dict[str, object]:
    result_path = args.out_dir / "result.json"
    source_result = json.loads((args.run_dir / "result.json").read_text())
    if not source_result.get("complete"):
        raise ValueError(f"source run is incomplete: {args.run_dir}")
    config = PairExperimentConfig(**source_result["config"])
    if config.decoder != "root_incidence":
        raise ValueError("relation quantization requires a root_incidence checkpoint")
    source_checkpoint = args.run_dir / "best.pt"
    if not source_checkpoint.exists():
        raise FileNotFoundError(source_checkpoint)
    source_identity = {
        "config": asdict(config),
        "best_epoch": source_result["best_epoch"],
        "split_fingerprints": source_result["split_fingerprints"],
    }
    if result_path.exists():
        existing = json.loads(result_path.read_text())
        if existing.get("complete") and existing.get("source_identity") == source_identity:
            print(json.dumps({"status": "skipped_complete", "result": str(result_path)}), flush=True)
            return existing

    device = torch.device(args.device)
    splits = _truncate_splits(
        load_task_splits(args.data_root, config.task, config.split_mode, config.data_split_seed),
        config,
    )
    model = PairRelationModel(config, len(splits.auxiliary_classes))
    checkpoint = torch.load(source_checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    if model.router is None or not hasattr(model.relation, "kernel") or not isinstance(model.relation.kernel, RootIncidenceKernel):
        raise TypeError("loaded model is not a routed Root-incidence scorer")
    router = model.router
    float_kernel = model.relation.kernel
    symmetry = model.relation.symmetry

    coordinates, _ = encode_split(model, splits.test, device, args.eval_batch_size)
    features = router.route(coordinates.to(device))
    float_transformed = float_kernel.transform_roots(features.roots)
    pairs = sample_pair_indices(splits.test.labels, config.task, config.eval_pairs, config.seed + 4001)
    expected_fingerprints = source_result["evaluation_protocol"]["set_fingerprints"]
    pair_fingerprint = index_fingerprint(pairs.query, pairs.key, pairs.target)
    if pair_fingerprint != expected_fingerprints["test_pairs"]:
        raise RuntimeError("quantization pair set does not match the source checkpoint evaluation")
    full_pair_prediction = score_float_cache_pairs(
        float_kernel,
        features.roots,
        float_transformed,
        pairs.query,
        pairs.key,
        symmetry=symmetry,
        batch_size=args.eval_batch_size,
    )
    full_metrics = _metric_bundle(config.task, splits.test.labels, pairs, full_pair_prediction)
    retrieval_sets: dict[str, object] = {}
    full_retrieval_scores: dict[str, Tensor] = {}
    if config.task == "same_class":
        for name, hard in (("random", False), ("hard", True)):
            retrieval = build_retrieval_set(
                splits.test,
                queries=config.retrieval_queries,
                candidates=config.retrieval_candidates,
                positives=config.retrieval_positives,
                seed=config.seed + 4001 + (101 if hard else 0),
                hard=hard,
                hard_reservoir=config.hard_reservoir,
            )
            fingerprint = index_fingerprint(retrieval.query, retrieval.candidates, retrieval.relevant)
            if fingerprint != expected_fingerprints[f"test_{name}_retrieval"]:
                raise RuntimeError(f"quantization {name} retrieval set does not match the source evaluation")
            query = retrieval.query[:, None].expand_as(retrieval.candidates).reshape(-1)
            key = retrieval.candidates.reshape(-1)
            full_score = score_float_cache_pairs(
                float_kernel,
                features.roots,
                float_transformed,
                query,
                key,
                symmetry=symmetry,
                batch_size=args.eval_batch_size,
            ).view_as(retrieval.candidates)
            full_metrics.update({f"{name}_{key}": value for key, value in retrieval_metrics(retrieval.relevant, full_score).items()})
            retrieval_sets[name] = retrieval
            full_retrieval_scores[name] = full_score

    baseline_deltas = {
        key: abs(float(value) - float(source_result["test"][key])) for key, value in full_metrics.items() if key in source_result["test"]
    }
    source_metric_tolerance = 1e-3
    if baseline_deltas and max(baseline_deltas.values()) > source_metric_tolerance:
        raise RuntimeError(f"recomputed full-precision metrics differ from source: {baseline_deltas}")

    variants: list[dict[str, object]] = []
    for mode in QUANTIZATION_MODES:
        quantized = float_kernel.quantized(router.roots, mode).to(device)
        cache = quantized.build_cache(features.roots)
        pair_prediction, pair_integer = score_integer_cache_pairs(
            quantized,
            cache,
            pairs.query,
            pairs.key,
            symmetry=symmetry,
            batch_size=args.eval_batch_size,
        )
        metrics = _metric_bundle(config.task, splits.test.labels, pairs, pair_prediction)
        overlap: dict[str, float] = {}
        if config.task == "same_class":
            for name in ("random", "hard"):
                retrieval = retrieval_sets[name]
                query = retrieval.query[:, None].expand_as(retrieval.candidates).reshape(-1)
                key = retrieval.candidates.reshape(-1)
                score, integer = score_integer_cache_pairs(
                    quantized,
                    cache,
                    query,
                    key,
                    symmetry=symmetry,
                    batch_size=args.eval_batch_size,
                )
                score = score.view_as(retrieval.candidates)
                integer = integer.view_as(retrieval.candidates)
                metrics.update({f"{name}_{key}": value for key, value in retrieval_metrics(retrieval.relevant, score).items()})
                overlap[f"{name}_full_top16_overlap"] = topk_overlap(full_retrieval_scores[name], integer)

        if config.task == "same_class":
            primary_metric = "random_recall_at_16"
            chance = config.retrieval_positives / config.retrieval_candidates
        else:
            primary_metric = "pair_macro_roc_auc"
            chance = 0.5
        full_primary = float(full_metrics[primary_metric])
        quantized_primary = float(metrics[primary_metric])
        raw_retention = quantized_primary / max(full_primary, 1e-12)
        chance_adjusted_retention = (quantized_primary - chance) / max(full_primary - chance, 1e-12)

        verify_count = min(4096, pairs.query.numel())
        query_features = router.route(coordinates[pairs.query[:verify_count]].to(device))
        key_features = router.route(coordinates[pairs.key[:verify_count]].to(device))
        forward = quantized.hard_score(query_features, key_features)
        reverse = quantized.hard_score(key_features, query_features)
        direct = 0.5 * (forward + reverse) if symmetry == "symmetric" else 0.5 * (forward - reverse)
        cached_verify = pair_prediction[:verify_count].to(device)
        max_direct_cached_error = float((direct - cached_verify).abs().max().item())
        if max_direct_cached_error > 2e-6:
            raise RuntimeError(f"{mode} direct/cached dequantized mismatch: {max_direct_cached_error}")

        reconstructed = quantized.reconstructed_coefficients()
        spec = quantized.spec
        variants.append(
            {
                "mode": mode,
                "levels": list(spec.levels),
                "storage_bits": spec.storage_bits,
                "coefficient_scale": float(quantized.scale.item()),
                "coefficient_mse": float((float_kernel.weight - reconstructed).square().mean().item()),
                "coefficient_max_error": float((float_kernel.weight - reconstructed).abs().max().item()),
                "coefficient_zero_fraction": float((quantized.codes == 0).float().mean().item()),
                "packed_coefficient_bytes": math.ceil(quantized.codes.numel() * spec.storage_bits / 8),
                "torch_code_bytes": quantized.codes.numel() * quantized.codes.element_size(),
                "primary_metric": primary_metric,
                "full_primary": full_primary,
                "quantized_primary": quantized_primary,
                "raw_retention": raw_retention,
                "chance_adjusted_retention": chance_adjusted_retention,
                "pair_integer_min": int(pair_integer.min().item()),
                "pair_integer_max": int(pair_integer.max().item()),
                "direct_cached_max_error": max_direct_cached_error,
                **overlap,
                "metrics": metrics,
                "execution": benchmark_quantized_kernel(
                    quantized,
                    float_kernel,
                    features.roots,
                    symmetry=symmetry,
                    pairs=args.benchmark_pairs,
                    warmups=args.benchmark_warmups,
                    iterations=args.benchmark_iterations,
                ),
            }
        )

    result = {
        "complete": True,
        "source_run": str(args.run_dir),
        "source_checkpoint": str(source_checkpoint),
        "source_identity": source_identity,
        "quantization_target": "trained Root-incidence relation coefficients",
        "scale_granularity": "one label-free symmetric scale per relation tensor",
        "evaluation_path": "per-object int8 sign + int32 transformed cache; per-pair integer add/sub accumulation",
        "dequantization_scope": "one common positive scale and bias after accumulation; omitted for ranking",
        "full_metrics": full_metrics,
        "source_metric_max_error": max(baseline_deltas.values(), default=0.0),
        "source_metric_tolerance": source_metric_tolerance,
        "variants": variants,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    temporary = result_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, result_path)
    print(json.dumps({"status": "complete", "result": str(result_path), "modes": list(QUANTIZATION_MODES)}), flush=True)
    return result


def summarize_quantization(args: argparse.Namespace) -> dict[str, object]:
    runs = [json.loads(path.read_text()) for path in sorted(args.result_dir.glob("**/result.json"))]
    runs = [run for run in runs if run.get("complete")]
    if not runs:
        raise RuntimeError(f"no complete quantization results under {args.result_dir}")
    flat: list[dict[str, object]] = []
    for run in runs:
        config = run["source_identity"]["config"]
        for variant in run["variants"]:
            execution = variant["execution"]
            flat.append(
                {
                    "task": config["task"],
                    "split_mode": config["split_mode"],
                    "payload_mode": config["payload_mode"],
                    "objective": config["objective"],
                    "seed": config["seed"],
                    "mode": variant["mode"],
                    "storage_bits": variant["storage_bits"],
                    "primary_metric": variant["primary_metric"],
                    "full_primary": variant["full_primary"],
                    "quantized_primary": variant["quantized_primary"],
                    "raw_retention": variant["raw_retention"],
                    "chance_adjusted_retention": variant["chance_adjusted_retention"],
                    "coefficient_mse": variant["coefficient_mse"],
                    "packed_coefficient_bytes": variant["packed_coefficient_bytes"],
                    "random_full_top16_overlap": variant.get("random_full_top16_overlap", math.nan),
                    "cached_integer_pairs_per_second": execution["cached_integer_pairs_per_second"],
                    "direct_integer_pairs_per_second": execution["direct_integer_pairs_per_second"],
                    "float_cached_pairs_per_second": execution["float_cached_pairs_per_second"],
                    "integer_cache_objects_per_second": execution["integer_cache_objects_per_second"],
                    "integer_cache_bytes_per_object": execution["integer_cache_bytes_per_object"],
                }
            )
    fields = list(flat[0])
    args.result_dir.mkdir(parents=True, exist_ok=True)
    with (args.result_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)

    group_fields = ("task", "split_mode", "payload_mode", "objective", "mode")
    groups: dict[tuple[str, ...], list[dict[str, object]]] = {}
    for row in flat:
        groups.setdefault(tuple(str(row[key]) for key in group_fields), []).append(row)
    aggregate: list[dict[str, object]] = []
    for key, members in sorted(groups.items()):
        record: dict[str, object] = dict(zip(group_fields, key))
        record["seeds"] = len(members)
        record["primary_metric"] = members[0]["primary_metric"]
        record["storage_bits"] = members[0]["storage_bits"]
        record["integer_cache_bytes_per_object"] = members[0]["integer_cache_bytes_per_object"]
        for metric in (
            "full_primary",
            "quantized_primary",
            "raw_retention",
            "chance_adjusted_retention",
            "random_full_top16_overlap",
            "cached_integer_pairs_per_second",
            "direct_integer_pairs_per_second",
            "float_cached_pairs_per_second",
            "integer_cache_objects_per_second",
        ):
            values = [float(member[metric]) for member in members if math.isfinite(float(member[metric]))]
            if values:
                record[metric], record[f"{metric}_sem"] = mean_sem(values)
        aggregate.append(record)
    with (args.result_dir / "aggregate.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({field for row in aggregate for field in row}))
        writer.writeheader()
        writer.writerows(aggregate)

    expected = 3 * 2 * 2 * 3
    complete = len(runs) == expected and len(flat) == expected * len(QUANTIZATION_MODES)
    decision = {
        "complete": complete,
        "checkpoint_runs": len(runs),
        "expected_checkpoint_runs": expected,
        "quantized_variants": len(flat),
        "expected_quantized_variants": expected * len(QUANTIZATION_MODES),
        "all_cached_integer_paths_exact": all(
            bool(variant["execution"]["direct_cached_integer_exact"]) for run in runs for variant in run["variants"]
        ),
    }
    (args.result_dir / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")

    lines = [
        "# Root-Incidence Relation-Coefficient Quantization on EMNIST Pairs",
        "",
        "## Protocol",
        "",
        f"This report aggregates `{len(runs)}` trained Root-incidence checkpoints and `{len(flat)}` post-training variants. "
        "The unary encoder and routes are unchanged. A single label-free scale quantizes only the learned sparse relation "
        "coefficients. Test metrics use an exact per-object integer cache and int32 pair accumulation; the common scale and "
        "bias are applied only after accumulation and can be omitted for ranking.",
        "",
        "## Three-seed retention",
        "",
        "| Task | Split | Payload | Objective | Mode | Full | Quantized | Chance-adjusted retention | Full Top-16 overlap | Cached Mpair/s |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        overlap = float(row.get("random_full_top16_overlap", math.nan))
        overlap_text = f"{overlap:.4f}" if math.isfinite(overlap) else "—"
        lines.append(
            f"| {row['task']} | {row['split_mode']} | {row['payload_mode']} | {row['objective']} | {row['mode']} | "
            f"{float(row['full_primary']):.4f} | {float(row['quantized_primary']):.4f} | "
            f"{float(row['chance_adjusted_retention']):.4f} | {overlap_text} | "
            f"{float(row['cached_integer_pairs_per_second']) / 1e6:.3f} |"
        )
    lines += [
        "",
        "## Systems boundary",
        "",
        "The measured path is a Torch reference, not a fused production kernel. Each object caches one int8 sign and one int32 "
        "transformed value per global root. Each pair reads those cached values and performs sign-controlled integer add/sub. "
        "Direct sparse scoring, cache construction, cached integer scoring, and the matched float cached path are recorded "
        "separately in every machine-readable result.",
        "",
    ]
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text("\n".join(lines))
    print(json.dumps(decision), flush=True)
    return decision


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--out-dir", type=Path, required=True)
    run.add_argument("--data-root", type=Path, default=Path("data"))
    run.add_argument("--device", default="cuda")
    run.add_argument("--eval-batch-size", type=int, default=4096)
    run.add_argument("--benchmark-pairs", type=int, default=8192)
    run.add_argument("--benchmark-warmups", type=int, default=3)
    run.add_argument("--benchmark-iterations", type=int, default=10)
    summarize = commands.add_parser("summarize")
    summarize.add_argument("--result-dir", type=Path, required=True)
    summarize.add_argument("--out-report", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run":
        run_quantization(args)
    else:
        summarize_quantization(args)


if __name__ == "__main__":
    main()
