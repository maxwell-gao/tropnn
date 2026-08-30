from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import Tensor

from tropnn.tools.zipf_addressing_capacity_law import sample_liu_gore_batch, zipf_probabilities
from tropnn.tools.zipf_groupsum_fixed_recognizer_controls import (
    FIT_GENERATOR_SEED,
    TEST_GENERATOR_SEED,
    VALIDATION_GENERATOR_SEED,
    RequestedJointMeanAccumulator,
    _path_metadata,
    _save_artifact_exclusive,
    _source_tensor_sentinel,
    _verify_source_sentinel,
    _write_exclusive,
    reconstruction_metrics,
    route_tuple_keys,
)
from tropnn.tools.zipf_groupsum_pair_merge_control import AdjacentPairMeanAccumulator, pair_mean_predict
from tropnn.tools.zipf_groupsum_walsh_stage1 import Config, WalshRecovery

SCHEMA = "zipf-groupsum-walsh-pair-merge-bridge-v1"
ARTIFACT_SCHEMA = "zipf-groupsum-walsh-pair-merge-artifact-v1"
COMPLETION_SCHEMA = "zipf-groupsum-walsh-pair-merge-bridge-completion-v1"
REGISTERED_SEEDS = (0, 1, 2)


@torch.no_grad()
def capture_split(
    model: WalshRecovery,
    probabilities: Tensor,
    *,
    samples: int,
    batch_size: int,
    generator_seed: int,
) -> tuple[Tensor, Tensor, Tensor]:
    generator = torch.Generator(device=probabilities.device).manual_seed(generator_seed)
    targets: list[Tensor] = []
    codes: list[Tensor] = []
    predictions: list[Tensor] = []
    seen = 0
    model.eval()
    while seen < samples:
        current = min(batch_size, samples - seen)
        x = sample_liu_gore_batch(probabilities, current, generator=generator)
        prediction, hidden = model.forward_with_hidden(x)
        route = model.decoder.route(hidden).indices
        targets.append(x.cpu())
        codes.append(route.to(device="cpu", dtype=torch.uint8))
        predictions.append(prediction.cpu())
        seen += current
    return torch.cat(targets), torch.cat(codes), torch.cat(predictions)


def _load_checkpointed_source(
    source_dir: Path,
    reference_dir: Path,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, object], WalshRecovery, dict[str, object]]:
    run_key = Config(seed=seed, device="cuda:0").run_key
    result_path = source_dir / "runs" / f"{run_key}.json"
    reference_path = reference_dir / "runs" / f"{run_key}.json"
    result = json.loads(result_path.read_text())
    reference = json.loads(reference_path.read_text())
    if result.get("schema") != "zipf-groupsum-walsh-stage1-run-v1" or result.get("complete") is not True:
        raise RuntimeError("invalid checkpointed Walsh source")
    if result.get("checkpoint") is None:
        raise RuntimeError("checkpointed Walsh source has no checkpoint")
    if result["route_health"] != reference["route_health"]:
        raise RuntimeError("checkpointed Walsh rerun did not exactly reproduce Stage-1' route health")
    checkpoint = Path(result["checkpoint"]["path"])
    metadata = _path_metadata(checkpoint)
    if any(metadata[key] != result["checkpoint"][key] for key in ("path", "size", "mtime_ns")):
        raise RuntimeError("Walsh checkpoint metadata mismatch")
    config = Config(**result["config"])
    model = WalshRecovery(
        config.n_features,
        config.model_dim,
        tables=config.tables,
        comparisons=config.comparisons,
        seed=config.seed + 1000,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    frozen = result["frozen_transform"]
    if not torch.equal(model.encoder.transform.signs.cpu(), torch.tensor(frozen["encoder_signs"], dtype=torch.int8)):
        raise RuntimeError("encoder Walsh signs mismatch")
    if not torch.equal(model.decoder.transform.signs.cpu(), torch.tensor(frozen["decoder_signs"], dtype=torch.int8)):
        raise RuntimeError("decoder Walsh signs mismatch")
    if not torch.equal(model.encoder.lut.anchors.cpu(), torch.tensor(frozen["encoder_anchors"], dtype=torch.int64)):
        raise RuntimeError("encoder Walsh anchors mismatch")
    if not torch.equal(model.decoder.lut.anchors.cpu(), torch.tensor(frozen["decoder_anchors"], dtype=torch.int64)):
        raise RuntimeError("decoder Walsh anchors mismatch")
    return result, model, {"result": _path_metadata(result_path), "reference": _path_metadata(reference_path)}


def _entropy_bits(counts: Tensor) -> float:
    positive = counts[counts > 0].to(torch.float64)
    probabilities = positive / positive.sum()
    return float(-(probabilities * probabilities.log2()).sum())


@torch.no_grad()
def run(
    source_dir: Path,
    reference_dir: Path,
    output_dir: Path,
    seed: int,
    *,
    fit_samples: int,
    eval_samples: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    if seed not in REGISTERED_SEEDS:
        raise ValueError("seed is outside the registered bridge diagnostic")
    source, model, source_paths = _load_checkpointed_source(source_dir, reference_dir, seed, device)
    config = Config(**source["config"])
    sentinel = _source_tensor_sentinel(model)
    probabilities = zipf_probabilities(config.n_features, config.alpha, config.activation_density, device=device)
    validation_target, validation_codes, validation_source = capture_split(
        model,
        probabilities,
        samples=eval_samples,
        batch_size=batch_size,
        generator_seed=VALIDATION_GENERATOR_SEED,
    )
    test_target, test_codes, test_source = capture_split(
        model,
        probabilities,
        samples=eval_samples,
        batch_size=batch_size,
        generator_seed=TEST_GENERATOR_SEED,
    )
    pair = AdjacentPairMeanAccumulator(config.tables, config.n_features)
    joint = RequestedJointMeanAccumulator(route_tuple_keys(validation_codes) + route_tuple_keys(test_codes), config.n_features)
    generator = torch.Generator(device=device).manual_seed(FIT_GENERATOR_SEED)
    fitted = 0
    while fitted < fit_samples:
        current = min(batch_size, fit_samples - fitted)
        x = sample_liu_gore_batch(probabilities, current, generator=generator)
        hidden = model.encoder(x)
        codes = model.decoder.route(hidden).indices.to(device="cpu", dtype=torch.uint8)
        x_cpu = x.cpu()
        pair.update(codes, x_cpu)
        joint.update(codes, x_cpu)
        fitted += current
    payload, counts, global_mean = pair.finalize()
    artifact_path = output_dir / "artifacts" / f"walsh-t32-c6-s{seed}.pt"
    artifact = {
        "schema": ARTIFACT_SCHEMA,
        "source_schema": source["schema"],
        "seed": seed,
        "payload": payload,
        "counts": counts,
        "global_mean": global_mean,
        "fit_samples": fit_samples,
        "fit_generator_seed": FIT_GENERATOR_SEED,
        "pairing": torch.arange(config.tables, dtype=torch.int64).reshape(-1, 2),
    }
    artifact_metadata = _save_artifact_exclusive(artifact_path, artifact)
    reloaded = torch.load(artifact_path, map_location="cpu", weights_only=True)
    roundtrip_exact = all(
        reloaded[name] == artifact[name] for name in ("schema", "source_schema", "seed", "fit_samples", "fit_generator_seed")
    ) and all(torch.equal(reloaded[name], artifact[name]) for name in ("payload", "counts", "global_mean", "pairing"))
    if not roundtrip_exact:
        raise RuntimeError("Walsh pair-merge artifact failed strict roundtrip")
    validation_pair = pair_mean_predict(reloaded["payload"], validation_codes, batch_size=batch_size)
    test_pair = pair_mean_predict(reloaded["payload"], test_codes, batch_size=batch_size)
    validation_joint, validation_joint_seen = joint.predict(validation_codes, global_mean)
    test_joint, test_joint_seen = joint.predict(test_codes, global_mean)
    probabilities_cpu = probabilities.cpu()
    splits = {
        "validation": {
            "source_sgd": reconstruction_metrics(validation_target, validation_source, probabilities_cpu),
            "pair_merge": reconstruction_metrics(validation_target, validation_pair, probabilities_cpu),
            "naive_full_joint": reconstruction_metrics(validation_target, validation_joint, probabilities_cpu),
        },
        "test": {
            "source_sgd": reconstruction_metrics(test_target, test_source, probabilities_cpu),
            "pair_merge": reconstruction_metrics(test_target, test_pair, probabilities_cpu),
            "naive_full_joint": reconstruction_metrics(test_target, test_joint, probabilities_cpu),
        },
    }
    test_pair_codes = pair.pair_codes(test_codes)
    test_unseen = counts.gather(1, test_pair_codes.T).T == 0
    verification = _verify_source_sentinel(model, sentinel)
    if not verification["all_equal"]:
        raise RuntimeError("Walsh source changed during bridge diagnostic")
    source_loss = splits["test"]["source_sgd"]["total_loss"]
    pair_loss = splits["test"]["pair_merge"]["total_loss"]
    return {
        "schema": SCHEMA,
        "complete": True,
        "seed": seed,
        "config": source["config"],
        "source_paths": source_paths,
        "source_checkpoint": source["checkpoint"],
        "source_route_health_exact_reproduction": True,
        "source_state_exact_verification": verification,
        "artifact": {**artifact_metadata, "strict_roundtrip_exact": roundtrip_exact},
        "protocol": {
            "fit_samples": fit_samples,
            "eval_samples_per_split": eval_samples,
            "batch_size": batch_size,
            "fit_generator_seed": FIT_GENERATOR_SEED,
            "validation_generator_seed": VALIDATION_GENERATOR_SEED,
            "test_generator_seed": TEST_GENERATOR_SEED,
            "tables": config.tables,
            "comparisons": config.comparisons,
            "pairing": "adjacent fixed pairs (0,1),(2,3),...,(30,31)",
            "estimator": "mean over 16 full-vector adjacent-pair conditional means",
            "optimizer_used": False,
            "recognizer_modified": False,
        },
        "fit": {
            "count_sum": int(counts.sum()),
            "expected_count_sum": fit_samples * (config.tables // 2),
            "observed_cells_by_pair": (counts > 0).sum(dim=1).tolist(),
            "entropy_bits_by_pair": [_entropy_bits(row) for row in counts],
            "joint_requested_keys": len(joint.keys),
            "joint_requested_keys_seen": int((joint.counts > 0).sum()),
        },
        **splits,
        "unseen": {
            "test_selected_pair_cell_fraction": float(test_unseen.to(torch.float64).mean()),
            "test_any_pair_token_fraction": float(test_unseen.any(dim=1).to(torch.float64).mean()),
            "validation_naive_joint_token_fraction": float((~validation_joint_seen).to(torch.float64).mean()),
            "test_naive_joint_token_fraction": float((~test_joint_seen).to(torch.float64).mean()),
        },
        "decision": {
            "pair_merge_improves_source": pair_loss < source_loss,
            "pair_merge_relative_improvement": (source_loss - pair_loss) / source_loss,
        },
        "storage": {
            "pair_tables": config.tables // 2,
            "rows_per_pair_table": 4096,
            "payload_width": config.n_features,
            "stored_payload_scalars": payload.numel(),
            "stored_payload_bytes_fp32": payload.numel() * 4,
        },
    }


def seal(output_dir: Path) -> dict[str, object]:
    paths = sorted((output_dir / "runs").glob("*.json"))
    rows = [json.loads(path.read_text()) for path in paths]
    if len(rows) != 3 or {row["seed"] for row in rows} != set(REGISTERED_SEEDS):
        raise RuntimeError("Walsh bridge result set is incomplete")
    if any(row.get("schema") != SCHEMA or row.get("complete") is not True for row in rows):
        raise RuntimeError("invalid Walsh bridge result")
    source = [row["test"]["source_sgd"]["total_loss"] for row in rows]
    pair = [row["test"]["pair_merge"]["total_loss"] for row in rows]
    joint = [row["test"]["naive_full_joint"]["total_loss"] for row in rows]
    payload = {
        "schema": COMPLETION_SCHEMA,
        "complete": True,
        "run_count": len(rows),
        "source_route_health_exact_all_seeds": all(row["source_route_health_exact_reproduction"] for row in rows),
        "source_state_exact_all_seeds": all(row["source_state_exact_verification"]["all_equal"] for row in rows),
        "artifact_roundtrip_exact_all_seeds": all(row["artifact"]["strict_roundtrip_exact"] for row in rows),
        "source_test_loss_by_seed": source,
        "pair_merge_test_loss_by_seed": pair,
        "naive_full_joint_test_loss_by_seed": joint,
        "source_test_loss_mean": sum(source) / 3,
        "pair_merge_test_loss_mean": sum(pair) / 3,
        "naive_full_joint_test_loss_mean": sum(joint) / 3,
        "pair_merge_improves_source_all_seeds": all(p < s for p, s in zip(pair, source, strict=True)),
        "pair_merge_relative_improvement_mean": (sum(source) - sum(pair)) / sum(source),
        "test_any_pair_unseen_max": max(row["unseen"]["test_any_pair_token_fraction"] for row in rows),
        "test_naive_joint_unseen_mean": sum(row["unseen"]["test_naive_joint_token_fraction"] for row in rows) / 3,
    }
    _write_exclusive(output_dir / "completion.json", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FWHT-code adjacent pair-merge bridge diagnostic")
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--source-dir", type=Path, required=True)
    run_parser.add_argument("--reference-dir", type=Path, required=True)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--seed", type=int, required=True)
    run_parser.add_argument("--fit-samples", type=int, default=1 << 20)
    run_parser.add_argument("--eval-samples", type=int, default=16_384)
    run_parser.add_argument("--batch-size", type=int, default=512)
    run_parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    seal_parser = commands.add_parser("seal")
    seal_parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run":
        if (args.fit_samples, args.eval_samples, args.batch_size) != (1 << 20, 16_384, 512):
            raise ValueError("formal bridge requires fit/eval/batch = 1048576/16384/512")
        output = args.output_dir / "runs" / f"walsh-t32-c6-s{args.seed}.json"
        if output.exists():
            existing = json.loads(output.read_text())
            if existing.get("schema") == SCHEMA and existing.get("complete") is True:
                print(f"skip complete seed {args.seed}")
                return
            raise RuntimeError(f"refusing to overwrite {output}")
        result = run(
            args.source_dir,
            args.reference_dir,
            args.output_dir,
            args.seed,
            fit_samples=args.fit_samples,
            eval_samples=args.eval_samples,
            batch_size=args.batch_size,
            device=torch.device(args.device),
        )
        _write_exclusive(output, result)
        print(json.dumps(result["decision"], indent=2, sort_keys=True))
        return
    print(json.dumps(seal(args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
