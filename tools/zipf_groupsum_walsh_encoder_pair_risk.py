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
    _path_metadata,
    _save_artifact_exclusive,
    _source_tensor_sentinel,
    _verify_source_sentinel,
    _write_exclusive,
    reconstruction_metrics,
)
from tropnn.tools.zipf_groupsum_pair_merge_control import AdjacentPairMeanAccumulator, pair_mean_predict
from tropnn.tools.zipf_groupsum_walsh_pair_merge_bridge import _load_checkpointed_source
from tropnn.tools.zipf_groupsum_walsh_stage1 import Config, WalshRecovery

SCHEMA = "zipf-groupsum-walsh-encoder-pair-risk-v1"
ARTIFACT_SCHEMA = "zipf-groupsum-walsh-encoder-pair-risk-artifact-v1"
COMPLETION_SCHEMA = "zipf-groupsum-walsh-encoder-pair-risk-completion-v1"
DECODER_ARTIFACT_SCHEMA = "zipf-groupsum-walsh-pair-merge-artifact-v1"
REGISTERED_SEEDS = (0, 1, 2)
PAIR_R2_THRESHOLD = 0.25
ENCODER_OVER_DECODER_R2_THRESHOLD = 0.10


@torch.no_grad()
def capture_split(
    model: WalshRecovery,
    probabilities: Tensor,
    *,
    samples: int,
    batch_size: int,
    generator_seed: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    generator = torch.Generator(device=probabilities.device).manual_seed(generator_seed)
    targets: list[Tensor] = []
    encoder_codes: list[Tensor] = []
    decoder_codes: list[Tensor] = []
    predictions: list[Tensor] = []
    seen = 0
    model.eval()
    while seen < samples:
        current = min(batch_size, samples - seen)
        x = sample_liu_gore_batch(probabilities, current, generator=generator)
        prediction, hidden = model.forward_with_hidden(x)
        encoder_route = model.encoder.route(x).indices
        decoder_route = model.decoder.route(hidden).indices
        targets.append(x.cpu())
        encoder_codes.append(encoder_route.to(device="cpu", dtype=torch.uint8))
        decoder_codes.append(decoder_route.to(device="cpu", dtype=torch.uint8))
        predictions.append(prediction.cpu())
        seen += current
    return (
        torch.cat(targets),
        torch.cat(encoder_codes),
        torch.cat(decoder_codes),
        torch.cat(predictions),
    )


def _pair_codes(codes: Tensor) -> Tensor:
    codes = codes.detach().to(device="cpu", dtype=torch.int64)
    if codes.ndim != 2 or codes.shape[1] != 32:
        raise ValueError(f"expected T32 codes, got {tuple(codes.shape)}")
    if codes.numel() and (int(codes.min()) < 0 or int(codes.max()) >= 64):
        raise ValueError("code outside the registered C6 alphabet")
    return codes[:, 0::2] * 64 + codes[:, 1::2]


def _entropy_bits(counts: Tensor) -> float:
    positive = counts[counts > 0].to(torch.float64)
    probabilities = positive / positive.sum()
    return float(-(probabilities * probabilities.log2()).sum())


@torch.no_grad()
def conditional_pair_risk(
    payload: Tensor,
    counts: Tensor,
    global_mean: Tensor,
    codes: Tensor,
    targets: Tensor,
    *,
    batch_size: int,
) -> dict[str, object]:
    payload = payload.detach().to(device="cpu", dtype=torch.float32)
    counts = counts.detach().to(device="cpu", dtype=torch.int64)
    global_mean = global_mean.detach().to(device="cpu", dtype=torch.float32)
    targets = targets.detach().to(device="cpu", dtype=torch.float32)
    pair_codes = _pair_codes(codes)
    if payload.shape != (16, 4096, targets.shape[1]) or counts.shape != (16, 4096):
        raise ValueError("pair payload/count shape mismatch")
    if global_mean.shape != (targets.shape[1],):
        raise ValueError("global mean shape mismatch")

    global_sse = torch.zeros(targets.shape[1], dtype=torch.float64)
    pair_sse = torch.zeros(16, targets.shape[1], dtype=torch.float64)
    for start in range(0, targets.shape[0], batch_size):
        stop = min(start + batch_size, targets.shape[0])
        target = targets[start:stop].to(torch.float64)
        global_sse += (target - global_mean.to(torch.float64)).square().sum(dim=0)
        batch_codes = pair_codes[start:stop]
        for pair in range(16):
            prediction = payload[pair, batch_codes[:, pair]].to(torch.float64)
            pair_sse[pair] += (target - prediction).square().sum(dim=0)

    global_feature_mse = global_sse / targets.shape[0]
    pair_feature_mse = pair_sse / targets.shape[0]
    global_loss = float(global_feature_mse.sum())
    losses = pair_feature_mse.sum(dim=1)
    tail_start = targets.shape[1] // 2
    global_tail_loss = float(global_feature_mse[tail_start:].sum())
    tail_losses = pair_feature_mse[:, tail_start:].sum(dim=1)
    r2 = 1.0 - losses / global_loss
    tail_r2 = 1.0 - tail_losses / global_tail_loss
    unseen = counts.gather(1, pair_codes.T).T == 0
    return {
        "samples": targets.shape[0],
        "global_mean_total_loss": global_loss,
        "global_mean_tail_loss": global_tail_loss,
        "individual_pair_total_loss": losses.tolist(),
        "individual_pair_tail_loss": tail_losses.tolist(),
        "individual_pair_r2": r2.tolist(),
        "individual_pair_tail_r2": tail_r2.tolist(),
        "mean_pair_total_loss": float(losses.mean()),
        "mean_pair_tail_loss": float(tail_losses.mean()),
        "mean_pair_r2": float(r2.mean()),
        "mean_pair_tail_r2": float(tail_r2.mean()),
        "selected_cell_unseen_fraction": float(unseen.to(torch.float64).mean()),
        "any_pair_unseen_token_fraction": float(unseen.any(dim=1).to(torch.float64).mean()),
    }


def _load_decoder_reference(
    decoder_bridge_dir: Path,
    seed: int,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    result_path = decoder_bridge_dir / "runs" / f"walsh-t32-c6-s{seed}.json"
    result = json.loads(result_path.read_text())
    if result.get("schema") != "zipf-groupsum-walsh-pair-merge-bridge-v1" or result.get("complete") is not True:
        raise RuntimeError("invalid decoder-pair reference result")
    artifact_path = Path(result["artifact"]["path"])
    metadata = _path_metadata(artifact_path)
    if any(metadata[key] != result["artifact"][key] for key in ("path", "size", "mtime_ns")):
        raise RuntimeError("decoder-pair artifact metadata mismatch")
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
    if artifact.get("schema") != DECODER_ARTIFACT_SCHEMA or artifact.get("seed") != seed:
        raise RuntimeError("invalid decoder-pair artifact")
    if artifact["payload"].shape != (16, 4096, 1024) or artifact["counts"].shape != (16, 4096):
        raise RuntimeError("decoder-pair artifact tensor shape mismatch")
    return result, artifact, {"result": _path_metadata(result_path), "artifact": metadata}


@torch.no_grad()
def run(
    source_dir: Path,
    reference_dir: Path,
    decoder_bridge_dir: Path,
    output_dir: Path,
    seed: int,
    *,
    fit_samples: int,
    eval_samples: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    if seed not in REGISTERED_SEEDS:
        raise ValueError("seed is outside the registered diagnostic")
    source, model, source_paths = _load_checkpointed_source(source_dir, reference_dir, seed, device)
    decoder_result, decoder_artifact, decoder_paths = _load_decoder_reference(decoder_bridge_dir, seed)
    config = Config(**source["config"])
    sentinel = _source_tensor_sentinel(model)
    probabilities = zipf_probabilities(config.n_features, config.alpha, config.activation_density, device=device)

    validation_target, validation_encoder, validation_decoder, validation_source = capture_split(
        model,
        probabilities,
        samples=eval_samples,
        batch_size=batch_size,
        generator_seed=VALIDATION_GENERATOR_SEED,
    )
    test_target, test_encoder, test_decoder, test_source = capture_split(
        model,
        probabilities,
        samples=eval_samples,
        batch_size=batch_size,
        generator_seed=TEST_GENERATOR_SEED,
    )

    encoder_accumulator = AdjacentPairMeanAccumulator(config.tables, config.n_features)
    generator = torch.Generator(device=device).manual_seed(FIT_GENERATOR_SEED)
    fitted = 0
    while fitted < fit_samples:
        current = min(batch_size, fit_samples - fitted)
        x = sample_liu_gore_batch(probabilities, current, generator=generator)
        encoder_codes = model.encoder.route(x).indices
        encoder_accumulator.update(encoder_codes, x)
        fitted += current
    payload, counts, global_mean = encoder_accumulator.finalize()
    del encoder_accumulator

    if not torch.equal(global_mean, decoder_artifact["global_mean"]):
        difference = float((global_mean - decoder_artifact["global_mean"]).abs().max())
        raise RuntimeError(f"encoder/decoder fits did not reproduce the same global mean: {difference}")

    artifact_path = output_dir / "artifacts" / f"walsh-encoder-t32-c6-s{seed}.pt"
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
    scalar_names = ("schema", "source_schema", "seed", "fit_samples", "fit_generator_seed")
    tensor_names = ("payload", "counts", "global_mean", "pairing")
    roundtrip_exact = all(reloaded[name] == artifact[name] for name in scalar_names) and all(
        torch.equal(reloaded[name], artifact[name]) for name in tensor_names
    )
    if not roundtrip_exact:
        raise RuntimeError("encoder pair-risk artifact failed strict roundtrip")

    probabilities_cpu = probabilities.cpu()
    splits: dict[str, object] = {}
    for split, target, encoder_codes, decoder_codes, source_prediction in (
        ("validation", validation_target, validation_encoder, validation_decoder, validation_source),
        ("test", test_target, test_encoder, test_decoder, test_source),
    ):
        encoder_risk = conditional_pair_risk(
            reloaded["payload"], reloaded["counts"], reloaded["global_mean"], encoder_codes, target, batch_size=batch_size
        )
        decoder_risk = conditional_pair_risk(
            decoder_artifact["payload"],
            decoder_artifact["counts"],
            decoder_artifact["global_mean"],
            decoder_codes,
            target,
            batch_size=batch_size,
        )
        encoder_average = pair_mean_predict(reloaded["payload"], encoder_codes, batch_size=batch_size)
        decoder_average = pair_mean_predict(decoder_artifact["payload"], decoder_codes, batch_size=batch_size)
        splits[split] = {
            "encoder_individual_pair_risk": encoder_risk,
            "decoder_individual_pair_risk": decoder_risk,
            "encoder_equal_average": reconstruction_metrics(target, encoder_average, probabilities_cpu),
            "decoder_equal_average": reconstruction_metrics(target, decoder_average, probabilities_cpu),
            "source_sgd": reconstruction_metrics(target, source_prediction, probabilities_cpu),
            "global_mean": reconstruction_metrics(target, reloaded["global_mean"].expand_as(target), probabilities_cpu),
        }

    decoder_test_replay_difference = abs(splits["test"]["decoder_equal_average"]["total_loss"] - decoder_result["test"]["pair_merge"]["total_loss"])
    if decoder_test_replay_difference > 1e-8:
        raise RuntimeError(f"decoder pair-average metric replay mismatch: {decoder_test_replay_difference}")

    verification = _verify_source_sentinel(model, sentinel)
    if not verification["all_equal"]:
        raise RuntimeError("Walsh source changed during encoder pair-risk diagnostic")
    encoder_r2 = splits["test"]["encoder_individual_pair_risk"]["mean_pair_r2"]
    decoder_r2 = splits["test"]["decoder_individual_pair_risk"]["mean_pair_r2"]
    delta_r2 = encoder_r2 - decoder_r2
    return {
        "schema": SCHEMA,
        "complete": True,
        "seed": seed,
        "config": source["config"],
        "source_paths": source_paths,
        "source_checkpoint": source["checkpoint"],
        "source_route_health_exact_reproduction": True,
        "source_state_exact_verification": verification,
        "decoder_reference_paths": decoder_paths,
        "decoder_pair_average_test_replay_abs_difference": decoder_test_replay_difference,
        "artifact": {**artifact_metadata, "strict_roundtrip_exact": roundtrip_exact},
        "protocol": {
            "fit_samples": fit_samples,
            "eval_samples_per_split": eval_samples,
            "batch_size": batch_size,
            "fit_generator_seed": FIT_GENERATOR_SEED,
            "validation_generator_seed": VALIDATION_GENERATOR_SEED,
            "test_generator_seed": TEST_GENERATOR_SEED,
            "pairing": "adjacent fixed encoder-code pairs",
            "primary_metric": "mean risk of 16 individual conditional-mean predictors",
            "pair_r2_threshold": PAIR_R2_THRESHOLD,
            "encoder_over_decoder_r2_threshold": ENCODER_OVER_DECODER_R2_THRESHOLD,
            "positive_calibration": "one half of the approximately 0.50 independent-group pair R2",
            "optimizer_used": False,
            "recognizer_modified": False,
        },
        "fit": {
            "count_sum": int(counts.sum()),
            "expected_count_sum": fit_samples * 16,
            "observed_cells_by_pair": (counts > 0).sum(dim=1).tolist(),
            "entropy_bits_by_pair": [_entropy_bits(row) for row in counts],
        },
        **splits,
        "decision": {
            "encoder_pair_r2": encoder_r2,
            "decoder_pair_r2": decoder_r2,
            "encoder_over_decoder_r2": delta_r2,
            "seed_passes_branch_b": encoder_r2 >= PAIR_R2_THRESHOLD and delta_r2 >= ENCODER_OVER_DECODER_R2_THRESHOLD,
        },
    }


def seal(output_dir: Path) -> dict[str, object]:
    paths = sorted((output_dir / "runs").glob("*.json"))
    rows = [json.loads(path.read_text()) for path in paths]
    if len(rows) != 3 or {row["seed"] for row in rows} != set(REGISTERED_SEEDS):
        raise RuntimeError("encoder pair-risk result set is incomplete")
    if any(row.get("schema") != SCHEMA or row.get("complete") is not True for row in rows):
        raise RuntimeError("invalid encoder pair-risk result")
    encoder_r2 = [row["decision"]["encoder_pair_r2"] for row in rows]
    decoder_r2 = [row["decision"]["decoder_pair_r2"] for row in rows]
    delta_r2 = [row["decision"]["encoder_over_decoder_r2"] for row in rows]
    branch_b = all(row["decision"]["seed_passes_branch_b"] for row in rows)
    payload = {
        "schema": COMPLETION_SCHEMA,
        "complete": True,
        "run_count": 3,
        "seeds": list(REGISTERED_SEEDS),
        "source_route_health_exact_all_seeds": all(row["source_route_health_exact_reproduction"] for row in rows),
        "source_state_exact_all_seeds": all(row["source_state_exact_verification"]["all_equal"] for row in rows),
        "artifact_roundtrip_exact_all_seeds": all(row["artifact"]["strict_roundtrip_exact"] for row in rows),
        "decoder_pair_average_replay_max_abs": max(row["decoder_pair_average_test_replay_abs_difference"] for row in rows),
        "encoder_pair_r2_by_seed": encoder_r2,
        "decoder_pair_r2_by_seed": decoder_r2,
        "encoder_over_decoder_r2_by_seed": delta_r2,
        "encoder_pair_r2_mean": sum(encoder_r2) / 3,
        "decoder_pair_r2_mean": sum(decoder_r2) / 3,
        "encoder_over_decoder_r2_mean": sum(delta_r2) / 3,
        "pair_r2_threshold": PAIR_R2_THRESHOLD,
        "encoder_over_decoder_r2_threshold": ENCODER_OVER_DECODER_R2_THRESHOLD,
        "branch_b_representation_misalignment": branch_b,
        "branch_a_repeat_mixing": not branch_b,
        "frozen_next_branch": "B_code_to_code_merger" if branch_b else "A_repeated_HD_mixing",
        "encoder_equal_average_test_loss_by_seed": [row["test"]["encoder_equal_average"]["total_loss"] for row in rows],
        "decoder_equal_average_test_loss_by_seed": [row["test"]["decoder_equal_average"]["total_loss"] for row in rows],
        "source_sgd_test_loss_by_seed": [row["test"]["source_sgd"]["total_loss"] for row in rows],
    }
    _write_exclusive(output_dir / "completion.json", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Direct FWHT encoder-code pair conditional-risk diagnostic")
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--source-dir", type=Path, required=True)
    run_parser.add_argument("--reference-dir", type=Path, required=True)
    run_parser.add_argument("--decoder-bridge-dir", type=Path, required=True)
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
            raise ValueError("formal diagnostic requires fit/eval/batch = 1048576/16384/512")
        output = args.output_dir / "runs" / f"walsh-encoder-t32-c6-s{args.seed}.json"
        if output.exists():
            existing = json.loads(output.read_text())
            if existing.get("schema") == SCHEMA and existing.get("complete") is True:
                print(f"skip complete seed {args.seed}")
                return
            raise RuntimeError(f"refusing to overwrite {output}")
        result = run(
            args.source_dir,
            args.reference_dir,
            args.decoder_bridge_dir,
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
