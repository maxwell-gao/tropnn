from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor

from tropnn.tools.zipf_addressing_capacity_law import sample_liu_gore_batch, zipf_probabilities
from tropnn.tools.zipf_groupsum_pclut_capacity_law import FormalConfig, PyramidRecovery, _build_formal_model

SCHEMA = "zipf-groupsum-fixed-recognizer-controls-v1"
ARTIFACT_SCHEMA = "zipf-groupsum-count-action-artifact-v1"
COMPLETION_SCHEMA = "zipf-groupsum-fixed-recognizer-controls-completion-v1"
FIT_GENERATOR_SEED = 90_007
VALIDATION_GENERATOR_SEED = 90_013
TEST_GENERATOR_SEED = 90_019
REGISTERED_TABLES = (1, 2, 4, 8, 16, 32, 64, 128)
REGISTERED_SOURCE_ARMS = ("pyramid_signed_sum", "independent_group_sum")


@dataclass
class CountActionAccumulator:
    """CPU-float64 sufficient statistics for feature-owned table rows."""

    tables: int
    rows: int
    output_dim: int

    def __post_init__(self) -> None:
        if self.tables < 1 or self.rows < 1 or self.output_dim < 1:
            raise ValueError("tables, rows, and output_dim must be positive")
        if self.output_dim % self.tables:
            raise ValueError("feature-owned control requires output_dim divisible by tables")
        self.owner_width = self.output_dim // self.tables
        self.counts = torch.zeros(self.tables * self.rows, dtype=torch.int64)
        self.sums = torch.zeros(self.tables * self.rows, self.owner_width, dtype=torch.float64)
        self.global_sum = torch.zeros(self.output_dim, dtype=torch.float64)
        self.samples = 0

    def update(self, codes: Tensor, targets: Tensor) -> None:
        codes = codes.detach().to(device="cpu", dtype=torch.int64).contiguous()
        targets = targets.detach().to(device="cpu", dtype=torch.float64).contiguous()
        if codes.ndim != 2 or codes.shape[1] != self.tables:
            raise ValueError(f"expected codes [B,{self.tables}], got {tuple(codes.shape)}")
        if targets.shape != (codes.shape[0], self.output_dim):
            raise ValueError(f"expected targets [B,{self.output_dim}], got {tuple(targets.shape)}")
        if codes.numel() and (int(codes.min()) < 0 or int(codes.max()) >= self.rows):
            raise ValueError("route code lies outside the registered row range")
        batch = codes.shape[0]
        offsets = torch.arange(self.tables, dtype=torch.int64).mul(self.rows)
        flat_codes = (codes + offsets.unsqueeze(0)).reshape(-1)
        # Feature k belongs to table k mod T.  Reshaping [B,W,T] exposes the
        # exact owned feature block for each table without copying a dense
        # [B,T,D] target.
        owned = targets.reshape(batch, self.owner_width, self.tables).transpose(1, 2).reshape(batch * self.tables, self.owner_width)
        self.counts.index_add_(0, flat_codes, torch.ones_like(flat_codes))
        self.sums.index_add_(0, flat_codes, owned)
        self.global_sum += targets.sum(dim=0)
        self.samples += batch

    def finalize(self) -> tuple[Tensor, Tensor, Tensor]:
        if self.samples < 1:
            raise RuntimeError("cannot finalize an empty count-action fit")
        counts = self.counts.reshape(self.tables, self.rows)
        sums = self.sums.reshape(self.tables, self.rows, self.owner_width)
        global_mean = self.global_sum / self.samples
        global_owned = global_mean.reshape(self.owner_width, self.tables).transpose(0, 1)
        means = sums / counts.clamp_min(1).to(torch.float64).unsqueeze(-1)
        means = torch.where(counts.unsqueeze(-1) > 0, means, global_owned.unsqueeze(1))
        payload = torch.zeros(self.tables, self.rows, self.output_dim, dtype=torch.float32)
        for table in range(self.tables):
            payload[table, :, table :: self.tables] = means[table].to(torch.float32)
        return payload, counts.clone(), global_mean.to(torch.float32)


class RequestedJointMeanAccumulator:
    """Exact tuple-key conditional means for a frozen set of requested keys.

    Keeping only evaluation-requested keys is output-equivalent to fitting and
    materializing every observed tuple, but bounds memory when the full route is
    nearly injective on the fit sample.
    """

    def __init__(self, requested_keys: Iterable[bytes], output_dim: int) -> None:
        ordered = sorted(set(requested_keys))
        self.keys = ordered
        self.key_to_index = {key: index for index, key in enumerate(ordered)}
        self.output_dim = int(output_dim)
        self.counts = torch.zeros(len(ordered), dtype=torch.int64)
        self.sums = torch.zeros(len(ordered), self.output_dim, dtype=torch.float64)
        self.matched_fit_samples = 0

    def update(self, codes: Tensor, targets: Tensor) -> None:
        targets = targets.detach().to(device="cpu", dtype=torch.float64).contiguous()
        keys = route_tuple_keys(codes)
        if targets.shape != (len(keys), self.output_dim):
            raise ValueError("joint-mean targets do not match route keys")
        sample_indices: list[int] = []
        requested_indices: list[int] = []
        for sample, key in enumerate(keys):
            requested = self.key_to_index.get(key)
            if requested is not None:
                sample_indices.append(sample)
                requested_indices.append(requested)
        if not sample_indices:
            return
        sample_tensor = torch.tensor(sample_indices, dtype=torch.int64)
        requested_tensor = torch.tensor(requested_indices, dtype=torch.int64)
        self.counts.index_add_(0, requested_tensor, torch.ones_like(requested_tensor))
        self.sums.index_add_(0, requested_tensor, targets[sample_tensor])
        self.matched_fit_samples += len(sample_indices)

    def predict(self, codes: Tensor, global_mean: Tensor) -> tuple[Tensor, Tensor]:
        global_mean = global_mean.detach().to(device="cpu", dtype=torch.float64)
        prediction = global_mean.expand(codes.shape[0], -1).clone()
        seen = torch.zeros(codes.shape[0], dtype=torch.bool)
        for sample, key in enumerate(route_tuple_keys(codes)):
            index = self.key_to_index.get(key)
            if index is not None and int(self.counts[index]) > 0:
                prediction[sample] = self.sums[index] / int(self.counts[index])
                seen[sample] = True
        return prediction.to(torch.float32), seen


def route_tuple_keys(codes: Tensor) -> list[bytes]:
    codes = codes.detach().to(device="cpu").contiguous()
    if codes.ndim != 2:
        raise ValueError(f"expected route codes [B,T], got {tuple(codes.shape)}")
    maximum = int(codes.max()) if codes.numel() else 0
    if maximum < 256:
        packed = codes.to(torch.uint8).numpy()
    elif maximum < 65_536:
        packed = codes.to(torch.uint16).numpy()
    else:
        raise ValueError("route tuple code exceeds uint16")
    return [row.tobytes() for row in packed]


def count_action_predict(payload: Tensor, codes: Tensor, *, batch_size: int = 512) -> Tensor:
    payload = payload.detach().to(device="cpu", dtype=torch.float32)
    codes = codes.detach().to(device="cpu", dtype=torch.int64)
    if payload.ndim != 3 or codes.ndim != 2 or payload.shape[0] != codes.shape[1]:
        raise ValueError("payload/codes shape mismatch")
    outputs: list[Tensor] = []
    for start in range(0, codes.shape[0], batch_size):
        batch_codes = codes[start : start + batch_size]
        output = torch.zeros(batch_codes.shape[0], payload.shape[-1], dtype=torch.float32)
        for table in range(payload.shape[0]):
            output += payload[table, batch_codes[:, table]]
        outputs.append(output.relu_())
    return torch.cat(outputs) if outputs else torch.empty(0, payload.shape[-1])


def reconstruction_metrics(target: Tensor, prediction: Tensor, probabilities: Tensor) -> dict[str, object]:
    target = target.to(torch.float64)
    prediction = prediction.to(torch.float64)
    probabilities = probabilities.to(device="cpu", dtype=torch.float64)
    if target.shape != prediction.shape or target.shape[1] != probabilities.numel():
        raise ValueError("metric shapes do not match")
    feature_mse = (target - prediction).square().mean(dim=0)
    zero_risk = probabilities * (4.0 / 3.0)
    constant_risk = zero_risk - probabilities.square()
    output_second_moment = prediction.square().mean(dim=0)
    n = target.shape[1]
    return {
        "samples": target.shape[0],
        "total_loss": float(feature_mse.sum()),
        "mean_loss": float(feature_mse.mean()),
        "zero_normalized_loss": float(feature_mse.sum() / zero_risk.sum()),
        "constant_normalized_loss": float(feature_mse.sum() / constant_risk.sum()),
        "feature_mse": feature_mse.tolist(),
        "output_second_moment": output_second_moment.tolist(),
        "tail_output_nonzero_fraction_1e_12": float((output_second_moment[n // 2 :] > 1e-12).to(torch.float64).mean()),
    }


@torch.no_grad()
def capture_frozen_split(
    model: PyramidRecovery,
    probabilities: Tensor,
    *,
    samples: int,
    batch_size: int,
    generator_seed: int,
) -> tuple[Tensor, Tensor, Tensor]:
    model.eval()
    generator = torch.Generator(device=probabilities.device).manual_seed(generator_seed)
    targets: list[Tensor] = []
    codes: list[Tensor] = []
    base_predictions: list[Tensor] = []
    seen = 0
    while seen < samples:
        current = min(batch_size, samples - seen)
        x = sample_liu_gore_batch(probabilities, current, generator=generator)
        base, hidden = model.forward_with_hidden(x)
        route = model.decoder.route(hidden).indices
        targets.append(x.cpu())
        codes.append(route.to(device="cpu", dtype=torch.uint8))
        base_predictions.append(base.cpu())
        seen += current
    return torch.cat(targets), torch.cat(codes), torch.cat(base_predictions)


def _source_tensor_sentinel(model: PyramidRecovery) -> dict[str, Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _verify_source_sentinel(model: PyramidRecovery, before: dict[str, Tensor]) -> dict[str, object]:
    after = model.state_dict()
    names_match = set(before) == set(after)
    mismatch = []
    if names_match:
        mismatch = [name for name, value in before.items() if not torch.equal(value, after[name].detach().cpu())]
    return {
        "tensor_count": len(before),
        "names_match": names_match,
        "mismatch_count": len(mismatch) if names_match else -1,
        "all_equal": names_match and not mismatch,
    }


def _path_metadata(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _write_exclusive(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _save_artifact_exclusive(path: Path, payload: dict[str, object]) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return _path_metadata(path)


def _load_source(source_dir: Path, run_key: str, device: torch.device) -> tuple[dict[str, object], PyramidRecovery]:
    result_path = source_dir / "runs" / f"{run_key}.json"
    result = json.loads(result_path.read_text())
    if result.get("schema") != "zipf-groupsum-pclut-capacity-law-run-v3" or result.get("complete") is not True:
        raise RuntimeError(f"invalid source result {result_path}")
    config = FormalConfig(**result["config"])
    if config.arm not in REGISTERED_SOURCE_ARMS or config.model_dim != 32 or config.comparisons != 6 or config.tables not in REGISTERED_TABLES:
        raise RuntimeError(f"source run is outside the frozen A3/A4 D32/C6 slice: {run_key}")
    checkpoint = source_dir / "checkpoints" / f"{run_key}.pt"
    source_checkpoint = result.get("checkpoint")
    if source_checkpoint is None:
        raise RuntimeError(f"source run has no checkpoint: {run_key}")
    metadata = _path_metadata(checkpoint)
    for key in ("path", "size", "mtime_ns"):
        if metadata[key] != source_checkpoint[key]:
            raise RuntimeError(f"source checkpoint metadata mismatch for {run_key}: {key}")
    model = _build_formal_model(config)
    if not isinstance(model, PyramidRecovery):
        raise TypeError("diagnostic source is not a PyramidRecovery")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    for stage_name in ("encoder", "decoder"):
        stage = getattr(model, stage_name)
        if not torch.equal(stage.lut.anchors.cpu(), torch.tensor(result["route"][f"{stage_name}_anchors"])):
            raise RuntimeError(f"source {stage_name} anchors do not match result")
        if not torch.equal(stage.lut.thresholds.detach().cpu(), torch.tensor(result["route"][f"{stage_name}_thresholds"])):
            raise RuntimeError(f"source {stage_name} thresholds do not match result")
    return result, model


@torch.no_grad()
def run_control(
    source_dir: Path,
    output_dir: Path,
    run_key: str,
    *,
    fit_samples: int,
    eval_samples: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    source, model = _load_source(source_dir, run_key, device)
    sentinel = _source_tensor_sentinel(model)
    config = FormalConfig(**source["config"])
    probabilities = zipf_probabilities(config.n_features, config.alpha, config.activation_density, device=device)
    validation_target, validation_codes, validation_base = capture_frozen_split(
        model,
        probabilities,
        samples=eval_samples,
        batch_size=batch_size,
        generator_seed=VALIDATION_GENERATOR_SEED,
    )
    test_target, test_codes, test_base = capture_frozen_split(
        model,
        probabilities,
        samples=eval_samples,
        batch_size=batch_size,
        generator_seed=TEST_GENERATOR_SEED,
    )
    requested_keys = route_tuple_keys(validation_codes) + route_tuple_keys(test_codes)
    joint = RequestedJointMeanAccumulator(requested_keys, config.n_features)
    count = CountActionAccumulator(config.tables, 1 << config.comparisons, config.n_features)

    generator = torch.Generator(device=device).manual_seed(FIT_GENERATOR_SEED)
    fitted = 0
    while fitted < fit_samples:
        current = min(batch_size, fit_samples - fitted)
        x = sample_liu_gore_batch(probabilities, current, generator=generator)
        hidden = model.encoder(x)
        codes = model.decoder.route(hidden).indices
        codes_cpu = codes.to(device="cpu", dtype=torch.uint8)
        x_cpu = x.cpu()
        count.update(codes_cpu, x_cpu)
        joint.update(codes_cpu, x_cpu)
        fitted += current

    payload, row_counts, global_mean = count.finalize()
    artifact_path = output_dir / "artifacts" / f"{run_key}.pt"
    artifact = {
        "schema": ARTIFACT_SCHEMA,
        "source_run_key": run_key,
        "payload": payload,
        "row_counts": row_counts,
        "global_mean": global_mean,
        "feature_owner": torch.arange(config.n_features, dtype=torch.int64).remainder(config.tables),
        "fit_samples": fit_samples,
        "fit_generator_seed": FIT_GENERATOR_SEED,
    }
    artifact_metadata = _save_artifact_exclusive(artifact_path, artifact)
    reloaded = torch.load(artifact_path, map_location="cpu", weights_only=True)
    roundtrip_exact = (
        reloaded["schema"] == artifact["schema"]
        and reloaded["source_run_key"] == run_key
        and reloaded["fit_samples"] == fit_samples
        and reloaded["fit_generator_seed"] == FIT_GENERATOR_SEED
        and all(torch.equal(reloaded[name], artifact[name]) for name in ("payload", "row_counts", "global_mean", "feature_owner"))
    )
    if not roundtrip_exact:
        raise RuntimeError("count-action artifact failed strict roundtrip")

    validation_count = count_action_predict(reloaded["payload"], validation_codes, batch_size=batch_size)
    test_count = count_action_predict(reloaded["payload"], test_codes, batch_size=batch_size)
    validation_joint, validation_seen = joint.predict(validation_codes, reloaded["global_mean"])
    test_joint, test_seen = joint.predict(test_codes, reloaded["global_mean"])
    probabilities_cpu = probabilities.cpu()
    validation_metrics = {
        "source_sgd": reconstruction_metrics(validation_target, validation_base, probabilities_cpu),
        "count_action": reconstruction_metrics(validation_target, validation_count, probabilities_cpu),
        "joint_code_oracle": reconstruction_metrics(validation_target, validation_joint, probabilities_cpu),
    }
    test_metrics = {
        "source_sgd": reconstruction_metrics(test_target, test_base, probabilities_cpu),
        "count_action": reconstruction_metrics(test_target, test_count, probabilities_cpu),
        "joint_code_oracle": reconstruction_metrics(test_target, test_joint, probabilities_cpu),
    }
    count_unseen_validation = (row_counts.gather(1, validation_codes.to(torch.int64).T) == 0).T
    count_unseen_test = (row_counts.gather(1, test_codes.to(torch.int64).T) == 0).T
    source_verification = _verify_source_sentinel(model, sentinel)
    if not source_verification["all_equal"]:
        raise RuntimeError("frozen source model changed during the control")
    result = {
        "schema": SCHEMA,
        "complete": True,
        "source_run_key": run_key,
        "source_result": _path_metadata(source_dir / "runs" / f"{run_key}.json"),
        "source_checkpoint": _path_metadata(source_dir / "checkpoints" / f"{run_key}.pt"),
        "config": source["config"],
        "protocol": {
            "fit_samples": fit_samples,
            "eval_samples_per_split": eval_samples,
            "batch_size": batch_size,
            "fit_generator_seed": FIT_GENERATOR_SEED,
            "validation_generator_seed": VALIDATION_GENERATOR_SEED,
            "test_generator_seed": TEST_GENERATOR_SEED,
            "count_action": "feature k owned by table k mod T; conditional mean by frozen decoder code",
            "joint_code": "exact complete decoder-code byte tuple; unseen fallback to fit global mean",
            "optimizer_used": False,
            "recognizer_modified": False,
        },
        "artifact": {**artifact_metadata, "strict_roundtrip_exact": roundtrip_exact},
        "source_state_exact_verification": source_verification,
        "fit": {
            "row_count_sum": int(row_counts.sum()),
            "expected_row_count_sum": fit_samples * config.tables,
            "observed_rows": int((row_counts > 0).sum()),
            "total_rows": row_counts.numel(),
            "joint_requested_key_count": len(joint.keys),
            "joint_requested_keys_seen": int((joint.counts > 0).sum()),
            "joint_matched_fit_samples_with_multiplicity": joint.matched_fit_samples,
        },
        "validation": validation_metrics,
        "test": test_metrics,
        "unseen": {
            "count_action_validation_selected_row_fraction": float(count_unseen_validation.to(torch.float64).mean()),
            "count_action_validation_any_token_fraction": float(count_unseen_validation.any(dim=1).to(torch.float64).mean()),
            "count_action_test_selected_row_fraction": float(count_unseen_test.to(torch.float64).mean()),
            "count_action_test_any_token_fraction": float(count_unseen_test.any(dim=1).to(torch.float64).mean()),
            "joint_validation_token_fraction": float((~validation_seen).to(torch.float64).mean()),
            "joint_test_token_fraction": float((~test_seen).to(torch.float64).mean()),
        },
        "contrasts": {
            split: {
                "count_minus_source_total_loss": metrics["count_action"]["total_loss"] - metrics["source_sgd"]["total_loss"],
                "joint_minus_source_total_loss": metrics["joint_code_oracle"]["total_loss"] - metrics["source_sgd"]["total_loss"],
            }
            for split, metrics in (("validation", validation_metrics), ("test", test_metrics))
        },
    }
    return result


def registered_source_keys(source_dir: Path) -> list[str]:
    keys: list[str] = []
    for pattern in (
        "pyramid-signed-sum-d32-t*-c6-*.pt",
        "independent-group-sum-d32-t*-c6-g512-*.pt",
    ):
        keys.extend(path.stem for path in sorted((source_dir / "checkpoints").glob(pattern)))
    return keys


def run_shard(args: argparse.Namespace) -> None:
    keys = registered_source_keys(args.source_dir)
    expected = len(REGISTERED_SOURCE_ARMS) * len(REGISTERED_TABLES) * 3
    if len(keys) != expected:
        raise RuntimeError(f"expected {expected} registered A3/A4 checkpoints, found {len(keys)}")
    for index, run_key in enumerate(keys):
        if index % args.shard_count != args.shard_index:
            continue
        output = args.output_dir / "runs" / f"{run_key}.json"
        if output.exists():
            existing = json.loads(output.read_text())
            if existing.get("complete") is True and existing.get("schema") == SCHEMA:
                print(f"skip complete {run_key}", flush=True)
                continue
            raise RuntimeError(f"refusing to overwrite {output}")
        print(f"start {run_key}", flush=True)
        result = run_control(
            args.source_dir,
            args.output_dir,
            run_key,
            fit_samples=args.fit_samples,
            eval_samples=args.eval_samples,
            batch_size=args.batch_size,
            device=torch.device(args.device),
        )
        _write_exclusive(output, result)
        print(
            f"done {run_key} source={result['test']['source_sgd']['total_loss']:.7g} "
            f"count={result['test']['count_action']['total_loss']:.7g} "
            f"joint={result['test']['joint_code_oracle']['total_loss']:.7g}",
            flush=True,
        )


def seal(args: argparse.Namespace) -> dict[str, object]:
    keys = registered_source_keys(args.source_dir)
    paths = sorted((args.output_dir / "runs").glob("*.json"))
    results = [json.loads(path.read_text()) for path in paths]
    expected = len(REGISTERED_SOURCE_ARMS) * len(REGISTERED_TABLES) * 3
    if len(keys) != expected or len(results) != expected:
        raise RuntimeError(f"expected {expected} sources/results, found {len(keys)}/{len(results)}")
    if {row["source_run_key"] for row in results} != set(keys):
        raise RuntimeError("control result key set does not match source checkpoints")
    if any(row.get("complete") is not True or row.get("schema") != SCHEMA for row in results):
        raise RuntimeError("invalid control result")
    summary = {
        "schema": COMPLETION_SCHEMA,
        "complete": True,
        "run_count": len(results),
        "source_run_count": len(keys),
        "all_source_states_exact": all(row["source_state_exact_verification"]["all_equal"] for row in results),
        "all_artifact_roundtrips_exact": all(row["artifact"]["strict_roundtrip_exact"] for row in results),
        "results": [
            {
                "source_run_key": row["source_run_key"],
                "source_arm": row["config"]["arm"],
                "tables": row["config"]["tables"],
                "seed": row["config"]["seed"],
                "validation": {arm: row["validation"][arm]["total_loss"] for arm in ("source_sgd", "count_action", "joint_code_oracle")},
                "test": {arm: row["test"][arm]["total_loss"] for arm in ("source_sgd", "count_action", "joint_code_oracle")},
                "joint_test_unseen": row["unseen"]["joint_test_token_fraction"],
            }
            for row in sorted(
                results,
                key=lambda item: (item["config"]["arm"], item["config"]["tables"], item["config"]["seed"]),
            )
        ],
    }
    _write_exclusive(args.output_dir / "completion.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Frozen-recognizer count-action and joint-code controls")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--source-dir", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--fit-samples", type=int, default=1 << 20)
    run.add_argument("--eval-samples", type=int, default=16_384)
    run.add_argument("--batch-size", type=int, default=512)
    run.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    run.add_argument("--shard-index", type=int, default=0)
    run.add_argument("--shard-count", type=int, default=1)
    seal_parser = commands.add_parser("seal")
    seal_parser.add_argument("--source-dir", type=Path, required=True)
    seal_parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run":
        if args.fit_samples != 1 << 20 or args.eval_samples != 16_384 or args.batch_size != 512:
            raise ValueError("formal control requires fit/eval/batch = 1048576/16384/512")
        if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
            raise ValueError("shard_index must lie in [0, shard_count)")
        run_shard(args)
        return
    summary = seal(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
