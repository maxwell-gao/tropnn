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
    _load_source,
    _path_metadata,
    _save_artifact_exclusive,
    _source_tensor_sentinel,
    _verify_source_sentinel,
    _write_exclusive,
    capture_frozen_split,
    reconstruction_metrics,
)
from tropnn.tools.zipf_groupsum_pclut_capacity_law import FormalConfig

SCHEMA = "zipf-groupsum-pair-merge-control-v1"
ARTIFACT_SCHEMA = "zipf-groupsum-pair-merge-artifact-v1"
COMPLETION_SCHEMA = "zipf-groupsum-pair-merge-completion-v1"
REGISTERED_ARMS = ("independent_group_sum", "pyramid_signed_sum")
REGISTERED_SEEDS = (0, 1, 2)
TABLES = 4
COMPARISONS = 6
ROWS = 1 << COMPARISONS
PAIR_ROWS = ROWS * ROWS


class AdjacentPairMeanAccumulator:
    """CPU-float64 sufficient statistics for adjacent 12-bit code pairs.

    Targets in the Liu--Gore task are sparse.  The implementation updates the
    dense pair-cell means only at nonzero target coordinates, avoiding a
    ``B x pair x N`` materialization while preserving exact sufficient
    statistics.
    """

    def __init__(self, tables: int, output_dim: int) -> None:
        if tables < 2 or tables % 2:
            raise ValueError("adjacent pair merge requires a positive even table count")
        self.tables = int(tables)
        self.pairs = self.tables // 2
        self.output_dim = int(output_dim)
        self.counts = torch.zeros(self.pairs, PAIR_ROWS, dtype=torch.int64)
        self.sums = torch.zeros(self.pairs, PAIR_ROWS, self.output_dim, dtype=torch.float64)
        self.global_sum = torch.zeros(self.output_dim, dtype=torch.float64)
        self.samples = 0

    def pair_codes(self, codes: Tensor) -> Tensor:
        codes = codes.detach().to(device="cpu", dtype=torch.int64)
        if codes.ndim != 2 or codes.shape[1] != self.tables:
            raise ValueError(f"expected [B,{self.tables}] codes, got {tuple(codes.shape)}")
        if codes.numel() and (int(codes.min()) < 0 or int(codes.max()) >= ROWS):
            raise ValueError("table code lies outside C6 range")
        return codes[:, 0::2] * ROWS + codes[:, 1::2]

    def update(self, codes: Tensor, targets: Tensor) -> None:
        pair_codes = self.pair_codes(codes)
        targets = targets.detach().to(device="cpu", dtype=torch.float64).contiguous()
        if targets.shape != (pair_codes.shape[0], self.output_dim):
            raise ValueError("target shape does not match pair codes")
        pair_offsets = torch.arange(self.pairs, dtype=torch.int64) * PAIR_ROWS
        flat_cells = pair_codes + pair_offsets.unsqueeze(0)
        flat_counts = self.counts.reshape(-1)
        flat_counts.index_add_(0, flat_cells.reshape(-1), torch.ones(flat_cells.numel(), dtype=torch.int64))

        nonzero = targets.nonzero(as_tuple=False)
        if nonzero.numel():
            sample = nonzero[:, 0]
            feature = nonzero[:, 1]
            cells = flat_cells[sample]
            indices = cells * self.output_dim + feature.unsqueeze(1)
            values = targets[sample, feature].unsqueeze(1).expand(-1, self.pairs)
            self.sums.reshape(-1).index_add_(0, indices.reshape(-1), values.reshape(-1))
        self.global_sum += targets.sum(dim=0)
        self.samples += pair_codes.shape[0]

    def finalize(self) -> tuple[Tensor, Tensor, Tensor]:
        if self.samples < 1:
            raise RuntimeError("cannot finalize an empty pair merge")
        global_mean = self.global_sum / self.samples
        means = self.sums / self.counts.clamp_min(1).to(torch.float64).unsqueeze(-1)
        means = torch.where(self.counts.unsqueeze(-1) > 0, means, global_mean.reshape(1, 1, -1))
        return means.to(torch.float32), self.counts.clone(), global_mean.to(torch.float32)


def pair_mean_predict(payload: Tensor, codes: Tensor, *, batch_size: int = 512) -> Tensor:
    payload = payload.detach().to(device="cpu", dtype=torch.float32)
    if payload.ndim != 3:
        raise ValueError("payload must be [pair,4096,D]")
    accumulator = AdjacentPairMeanAccumulator(payload.shape[0] * 2, payload.shape[-1])
    pair_codes = accumulator.pair_codes(codes)
    outputs: list[Tensor] = []
    for start in range(0, pair_codes.shape[0], batch_size):
        batch = pair_codes[start : start + batch_size]
        selected = torch.stack([payload[pair, batch[:, pair]] for pair in range(payload.shape[0])], dim=1)
        outputs.append(selected.mean(dim=1))
    return torch.cat(outputs) if outputs else torch.empty(0, payload.shape[-1])


def _entropy_bits(counts: Tensor) -> float:
    positive = counts[counts > 0].to(torch.float64)
    probabilities = positive / positive.sum()
    return float(-(probabilities * probabilities.log2()).sum())


def _registered_keys(source_dir: Path) -> list[str]:
    keys: list[str] = []
    for seed in REGISTERED_SEEDS:
        keys.append(f"independent-group-sum-d32-t4-c6-g512-a1p0-e1p0-wd0p0-s{seed}")
        keys.append(f"pyramid-signed-sum-d32-t4-c6-a1p0-e1p0-wd0p0-s{seed}")
    missing = [key for key in keys if not (source_dir / "checkpoints" / f"{key}.pt").exists()]
    if missing:
        raise RuntimeError(f"missing registered source checkpoints: {missing}")
    return sorted(keys)


@torch.no_grad()
def run(
    source_dir: Path,
    prior_control_dir: Path,
    output_dir: Path,
    run_key: str,
    *,
    fit_samples: int,
    eval_samples: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    if run_key not in _registered_keys(source_dir):
        raise ValueError("run key is outside the frozen T4/C6/D32 pair-merge diagnostic")
    source, model = _load_source(source_dir, run_key, device)
    config = FormalConfig(**source["config"])
    if config.arm not in REGISTERED_ARMS or config.tables != TABLES or config.comparisons != COMPARISONS:
        raise RuntimeError("source configuration does not match the pair-merge contract")
    prior_path = prior_control_dir / "runs" / f"{run_key}.json"
    prior = json.loads(prior_path.read_text())
    if prior.get("schema") != "zipf-groupsum-fixed-recognizer-controls-v1" or prior.get("complete") is not True:
        raise RuntimeError("missing completed frozen-recognizer control")
    sentinel = _source_tensor_sentinel(model)
    probabilities = zipf_probabilities(config.n_features, config.alpha, config.activation_density, device=device)
    validation_target, validation_codes, validation_source = capture_frozen_split(
        model,
        probabilities,
        samples=eval_samples,
        batch_size=batch_size,
        generator_seed=VALIDATION_GENERATOR_SEED,
    )
    test_target, test_codes, test_source = capture_frozen_split(
        model,
        probabilities,
        samples=eval_samples,
        batch_size=batch_size,
        generator_seed=TEST_GENERATOR_SEED,
    )
    pair = AdjacentPairMeanAccumulator(config.tables, config.n_features)
    generator = torch.Generator(device=device).manual_seed(FIT_GENERATOR_SEED)
    fitted = 0
    while fitted < fit_samples:
        current = min(batch_size, fit_samples - fitted)
        x = sample_liu_gore_batch(probabilities, current, generator=generator)
        hidden = model.encoder(x)
        codes = model.decoder.route(hidden).indices
        pair.update(codes, x)
        fitted += current
    payload, counts, global_mean = pair.finalize()
    artifact_path = output_dir / "artifacts" / f"{run_key}.pt"
    artifact = {
        "schema": ARTIFACT_SCHEMA,
        "source_run_key": run_key,
        "payload": payload,
        "counts": counts,
        "global_mean": global_mean,
        "fit_samples": fit_samples,
        "fit_generator_seed": FIT_GENERATOR_SEED,
        "pairing": torch.tensor(((0, 1), (2, 3)), dtype=torch.int64),
    }
    artifact_metadata = _save_artifact_exclusive(artifact_path, artifact)
    reloaded = torch.load(artifact_path, map_location="cpu", weights_only=True)
    roundtrip_exact = all(reloaded[name] == artifact[name] for name in ("schema", "source_run_key", "fit_samples", "fit_generator_seed")) and all(
        torch.equal(reloaded[name], artifact[name]) for name in ("payload", "counts", "global_mean", "pairing")
    )
    if not roundtrip_exact:
        raise RuntimeError("pair-merge artifact failed strict roundtrip")
    validation_prediction = pair_mean_predict(reloaded["payload"], validation_codes, batch_size=batch_size)
    test_prediction = pair_mean_predict(reloaded["payload"], test_codes, batch_size=batch_size)
    probabilities_cpu = probabilities.cpu()
    validation_metrics = reconstruction_metrics(validation_target, validation_prediction, probabilities_cpu)
    test_metrics = reconstruction_metrics(test_target, test_prediction, probabilities_cpu)
    regenerated_source = {
        "validation": reconstruction_metrics(validation_target, validation_source, probabilities_cpu),
        "test": reconstruction_metrics(test_target, test_source, probabilities_cpu),
    }
    source_metric_max_abs = max(
        abs(regenerated_source[split]["total_loss"] - prior[split]["source_sgd"]["total_loss"]) for split in ("validation", "test")
    )
    if source_metric_max_abs > 1e-6:
        raise RuntimeError(f"source metric replay disagrees with prior control by {source_metric_max_abs}")
    validation_pair_codes = pair.pair_codes(validation_codes)
    test_pair_codes = pair.pair_codes(test_codes)
    validation_unseen = counts.gather(1, validation_pair_codes.T).T == 0
    test_unseen = counts.gather(1, test_pair_codes.T).T == 0
    verification = _verify_source_sentinel(model, sentinel)
    if not verification["all_equal"]:
        raise RuntimeError("source model changed during pair-merge diagnostic")
    joint_test = prior["test"]["joint_code_oracle"]["total_loss"]
    source_test = prior["test"]["source_sgd"]["total_loss"]
    pair_test = test_metrics["total_loss"]
    return {
        "schema": SCHEMA,
        "complete": True,
        "source_run_key": run_key,
        "config": source["config"],
        "protocol": {
            "fit_samples": fit_samples,
            "eval_samples_per_split": eval_samples,
            "batch_size": batch_size,
            "fit_generator_seed": FIT_GENERATOR_SEED,
            "validation_generator_seed": VALIDATION_GENERATOR_SEED,
            "test_generator_seed": TEST_GENERATOR_SEED,
            "pairing": [[0, 1], [2, 3]],
            "estimator": "mean of the two adjacent-pair conditional target means",
            "optimizer_used": False,
            "recognizer_modified": False,
            "claim_boundary": "one pair-interaction layer; not yet a recursive C-bit code compressor",
        },
        "source_result": _path_metadata(source_dir / "runs" / f"{run_key}.json"),
        "source_checkpoint": _path_metadata(source_dir / "checkpoints" / f"{run_key}.pt"),
        "prior_control": _path_metadata(prior_path),
        "artifact": {**artifact_metadata, "strict_roundtrip_exact": roundtrip_exact},
        "source_state_exact_verification": verification,
        "source_metric_replay_max_abs_difference": source_metric_max_abs,
        "fit": {
            "count_sum": int(counts.sum()),
            "expected_count_sum": fit_samples * (TABLES // 2),
            "observed_cells_by_pair": (counts > 0).sum(dim=1).tolist(),
            "total_cells_by_pair": PAIR_ROWS,
            "entropy_bits_by_pair": [_entropy_bits(row) for row in counts],
            "stored_payload_rows": counts.numel(),
            "stored_payload_scalars": payload.numel(),
        },
        "validation": {
            "pair_merge": validation_metrics,
            "source_sgd": prior["validation"]["source_sgd"],
            "count_action": prior["validation"]["count_action"],
            "joint_code_oracle": prior["validation"]["joint_code_oracle"],
        },
        "test": {
            "pair_merge": test_metrics,
            "source_sgd": prior["test"]["source_sgd"],
            "count_action": prior["test"]["count_action"],
            "joint_code_oracle": prior["test"]["joint_code_oracle"],
        },
        "unseen": {
            "validation_selected_pair_cell_fraction": float(validation_unseen.to(torch.float64).mean()),
            "validation_any_token_fraction": float(validation_unseen.any(dim=1).to(torch.float64).mean()),
            "test_selected_pair_cell_fraction": float(test_unseen.to(torch.float64).mean()),
            "test_any_token_fraction": float(test_unseen.any(dim=1).to(torch.float64).mean()),
        },
        "decision": {
            "test_pair_merge_between_source_and_full_joint": min(source_test, joint_test) <= pair_test <= max(source_test, joint_test),
            "test_pair_merge_improves_source": pair_test < source_test,
            "test_pair_merge_gap_closed_fraction": (source_test - pair_test) / (source_test - joint_test),
        },
        "arithmetic_and_storage": {
            "inference_pair_lookups": TABLES // 2,
            "inference_vector_additions": TABLES // 2 - 1,
            "inference_scalar_scale_by_inverse_pair_count": config.n_features,
            "active_payload_scalars": (TABLES // 2) * config.n_features,
            "stored_payload_scalars": payload.numel(),
            "stored_payload_bytes_fp32": payload.numel() * 4,
        },
    }


def seal(source_dir: Path, output_dir: Path) -> dict[str, object]:
    expected = set(_registered_keys(source_dir))
    paths = sorted((output_dir / "runs").glob("*.json"))
    rows = [json.loads(path.read_text()) for path in paths]
    if {row["source_run_key"] for row in rows} != expected or len(rows) != len(expected):
        raise RuntimeError("pair-merge result set is incomplete")
    if any(row.get("schema") != SCHEMA or row.get("complete") is not True for row in rows):
        raise RuntimeError("invalid pair-merge result")
    summary_rows: list[dict[str, object]] = []
    for arm in REGISTERED_ARMS:
        selected = [row for row in rows if row["config"]["arm"] == arm]
        selected.sort(key=lambda row: row["config"]["seed"])
        source = [row["test"]["source_sgd"]["total_loss"] for row in selected]
        pair = [row["test"]["pair_merge"]["total_loss"] for row in selected]
        joint = [row["test"]["joint_code_oracle"]["total_loss"] for row in selected]
        summary_rows.append(
            {
                "arm": arm,
                "source_test_loss_by_seed": source,
                "pair_merge_test_loss_by_seed": pair,
                "joint_test_loss_by_seed": joint,
                "source_test_loss_mean": sum(source) / len(source),
                "pair_merge_test_loss_mean": sum(pair) / len(pair),
                "joint_test_loss_mean": sum(joint) / len(joint),
                "pair_merge_improves_source_all_seeds": all(p < s for p, s in zip(pair, source, strict=True)),
                "pair_merge_between_source_and_joint_all_seeds": all(
                    min(s, j) <= p <= max(s, j) for s, p, j in zip(source, pair, joint, strict=True)
                ),
            }
        )
    payload = {
        "schema": COMPLETION_SCHEMA,
        "complete": True,
        "run_count": len(rows),
        "all_source_states_exact": all(row["source_state_exact_verification"]["all_equal"] for row in rows),
        "all_artifact_roundtrips_exact": all(row["artifact"]["strict_roundtrip_exact"] for row in rows),
        "rows": summary_rows,
    }
    _write_exclusive(output_dir / "completion.json", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="One-layer adjacent code-pair conditional-mean decoder")
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--source-dir", type=Path, required=True)
    run_parser.add_argument("--prior-control-dir", type=Path, required=True)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--run-key", required=True)
    run_parser.add_argument("--fit-samples", type=int, default=1 << 20)
    run_parser.add_argument("--eval-samples", type=int, default=16_384)
    run_parser.add_argument("--batch-size", type=int, default=512)
    run_parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    seal_parser = commands.add_parser("seal")
    seal_parser.add_argument("--source-dir", type=Path, required=True)
    seal_parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run":
        if (args.fit_samples, args.eval_samples, args.batch_size) != (1 << 20, 16_384, 512):
            raise ValueError("formal pair merge requires fit/eval/batch = 1048576/16384/512")
        output = args.output_dir / "runs" / f"{args.run_key}.json"
        if output.exists():
            existing = json.loads(output.read_text())
            if existing.get("schema") == SCHEMA and existing.get("complete") is True:
                print(f"skip complete {args.run_key}")
                return
            raise RuntimeError(f"refusing to overwrite {output}")
        result = run(
            args.source_dir,
            args.prior_control_dir,
            args.output_dir,
            args.run_key,
            fit_samples=args.fit_samples,
            eval_samples=args.eval_samples,
            batch_size=args.batch_size,
            device=torch.device(args.device),
        )
        _write_exclusive(output, result)
        print(json.dumps(result["decision"], indent=2, sort_keys=True))
        return
    print(json.dumps(seal(args.source_dir, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
