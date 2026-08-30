"""Audit total feature/code drift in the released-stem product-chart run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tropnn.tools.emnist_payload_dtype_sweep import _load_emnist_split
from tropnn.tools.emnist_pq_product_grid_factorial import _source_linear
from tropnn.tools.emnist_product_chart_end_to_end import make_end_to_end_models
from tropnn.tools.emnist_product_chart_factorial import ARMS, _single_table_label_mi_bits


def _atomic_json(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


@torch.no_grad()
def _features(stem: nn.Linear, x: Tensor, *, batch_size: int, device: torch.device) -> Tensor:
    rows: list[Tensor] = []
    for start in range(0, x.shape[0], batch_size):
        rows.append(F.gelu(stem(x[start : start + batch_size].to(device).flatten(1))).cpu())
    return torch.cat(rows)


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, object]:
    source = torch.load(args.source_artifact, map_location="cpu", weights_only=False)
    pq = torch.load(args.pq_artifact, map_location="cpu", weights_only=False)
    artifact = torch.load(args.artifact, map_location="cpu", weights_only=False)
    result = json.loads(args.result.read_text())
    if source.get("schema") != "emnist-maddness-task-ste-v1" or pq.get("schema") != "emnist-pq-product-grid-factorial-v1":
        raise ValueError("unexpected source schema")
    if artifact.get("schema") != "emnist-product-chart-end-to-end-v1" or result.get("schema") != artifact.get("schema"):
        raise ValueError("result/artifact schema mismatch")
    _train_x, _train_y = _load_emnist_split(args.root, "balanced", train=True, limit=1, seed=0)
    held_x, held_y = _load_emnist_split(args.root, "balanced", train=False, limit=0, seed=0)
    device = torch.device(args.device)
    records: list[dict[str, object]] = []
    stored_rows = {(int(row["seed"]), str(row["arm"])): row for row in result["rows"]}
    for seed in result["protocol"]["seeds"]:
        seed = int(seed)
        centroids = pq["state"][f"seed{seed}.pq.centroids"]
        rows = pq["state"][f"seed{seed}.pq.free_rows"]
        initial_stem = _source_linear(source["state"], seed, "stem", held_x[0].numel(), 64).to(device)
        initial_features = _features(initial_stem, held_x, batch_size=args.batch_size, device=device)
        initial_local = initial_features.reshape(-1, 32, 2)
        initial_distances = (initial_local.unsqueeze(-2) - centroids.unsqueeze(0)).square().sum(dim=-1)
        initial_codes = initial_distances.argmin(dim=-1)
        models, _error, _state = make_end_to_end_models(source["state"], seed, held_x[0].numel(), 47, centroids, rows, rank=8, temperature=1.0)
        initial_mi = _single_table_label_mi_bits(initial_codes, held_y, 16)
        for arm in ARMS:
            prefix = f"seed{seed}.{arm}."
            state = {key[len(prefix) :]: value for key, value in artifact["state"].items() if key.startswith(prefix)}
            model = models[arm]
            model.load_state_dict(state, strict=True)
            model.to(device)
            final_features = _features(model.stem, held_x, batch_size=args.batch_size, device=device)
            final_logits, final_codes_device = model.head.hard_output(final_features.to(device))
            final_codes = final_codes_device.cpu()
            replay_ce = float(F.cross_entropy(final_logits, held_y.to(device)))
            replay_accuracy = float((final_logits.argmax(dim=-1) == held_y.to(device)).float().mean())
            stored = stored_rows[seed, arm]
            delta = final_features - initial_features
            initial_norm = initial_features.square().mean().sqrt().clamp_min(1e-30)
            cosine = F.cosine_similarity(initial_features, final_features, dim=-1).mean()
            records.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "feature_relative_rms_drift": float(delta.square().mean().sqrt() / initial_norm),
                    "feature_mean_cosine_to_initial": float(cosine),
                    "per_table_code_change_fraction": float((final_codes != initial_codes).float().mean()),
                    "any_table_code_change_fraction": float((final_codes != initial_codes).any(dim=-1).float().mean()),
                    "initial_single_table_label_mi_bits": initial_mi,
                    "final_single_table_label_mi_bits": _single_table_label_mi_bits(final_codes, held_y, 16),
                    "held_ce_replay_abs_error": abs(replay_ce - float(stored["held_ce"])),
                    "held_accuracy_replay_abs_error": abs(replay_accuracy - float(stored["held_accuracy"])),
                }
            )
    max_ce_error = max(float(record["held_ce_replay_abs_error"]) for record in records)
    max_accuracy_error = max(float(record["held_accuracy_replay_abs_error"]) for record in records)
    if max_ce_error > 1e-7 or max_accuracy_error > 1e-7:
        raise RuntimeError("reloaded artifact does not replay stored held metrics")
    return {
        "schema": "emnist-product-chart-coordinate-drift-audit-v1",
        "source_result": str(args.result.resolve()),
        "source_artifact": str(args.artifact.resolve()),
        "maximum_held_ce_replay_abs_error": max_ce_error,
        "maximum_held_accuracy_replay_abs_error": max_accuracy_error,
        "records": records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-artifact", type=Path, required=True)
    parser.add_argument("--pq-artifact", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size < 1 or not all(path.is_file() for path in (args.source_artifact, args.pq_artifact, args.result, args.artifact)):
        parser.error("invalid batch size or missing input")
    return args


def main() -> None:
    args = parse_args()
    _atomic_json(args.output, run(args))


if __name__ == "__main__":
    main()
