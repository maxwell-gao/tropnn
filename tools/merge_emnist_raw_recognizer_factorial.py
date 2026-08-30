"""Mechanically merge independent seed outputs of the raw recognizer factorial."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from tropnn.tools.emnist_raw_recognizer_factorial import (
    SCHEMA,
    Evaluation,
    _path_metadata,
    _save_exclusive,
    _write_json_exclusive,
    summarize,
)


def merge_seed_directories(seed_directories: tuple[Path, ...], output: Path, artifact: Path) -> dict[str, object]:
    if len(seed_directories) < 1:
        raise ValueError("at least one seed directory is required")
    results: list[dict[str, object]] = []
    artifacts: list[dict[str, object]] = []
    for directory in seed_directories:
        result_path = directory / "result.json"
        artifact_path = directory / "artifact.pt"
        result = json.loads(result_path.read_text())
        seed_artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
        if result.get("schema") != SCHEMA or seed_artifact.get("schema") != SCHEMA:
            raise ValueError(f"unexpected schema under {directory}")
        if not bool(result.get("artifact_roundtrip_exact")):
            raise ValueError(f"seed artifact was not sealed under {directory}")
        results.append(result)
        artifacts.append(seed_artifact)

    seeds: list[int] = []
    rows: list[Evaluation] = []
    audits: dict[str, object] = {}
    state: dict[str, torch.Tensor] = {}
    reference_protocol = copy.deepcopy(results[0]["protocol"])
    if not isinstance(reference_protocol, dict):
        raise TypeError("protocol must be a dictionary")
    for directory, result, seed_artifact in zip(seed_directories, results, artifacts, strict=True):
        protocol = copy.deepcopy(result["protocol"])
        if not isinstance(protocol, dict) or not isinstance(seed_artifact["protocol"], dict):
            raise TypeError("seed protocol must be a dictionary")
        current_seeds = protocol.pop("seeds")
        artifact_protocol = copy.deepcopy(seed_artifact["protocol"])
        artifact_seeds = artifact_protocol.pop("seeds")
        expected = copy.deepcopy(reference_protocol)
        expected.pop("seeds")
        if protocol != expected or protocol != artifact_protocol or current_seeds != artifact_seeds:
            raise ValueError(f"protocol mismatch under {directory}")
        if not isinstance(current_seeds, list) or len(current_seeds) != 1:
            raise ValueError("each source run must contain exactly one seed")
        seed = int(current_seeds[0])
        if seed in seeds:
            raise ValueError("duplicate seed")
        seeds.append(seed)
        rows.extend(Evaluation(**row) for row in result["rows"])  # type: ignore[arg-type]
        seed_audits = result["audits"]
        if not isinstance(seed_audits, dict) or set(seed_audits) != {str(seed)}:
            raise ValueError("seed audit keys do not match")
        audits.update(seed_audits)
        overlap = state.keys() & seed_artifact["state"].keys()
        if overlap:
            raise ValueError(f"duplicate artifact state keys: {sorted(overlap)}")
        state.update(seed_artifact["state"])

    order = sorted(range(len(seeds)), key=seeds.__getitem__)
    seeds = [seeds[index] for index in order]
    rows.sort(key=lambda row: (row.seed, row.arm))
    reference_protocol["seeds"] = seeds
    merged_artifact = {"schema": SCHEMA, "protocol": reference_protocol, "state": state}
    _save_exclusive(artifact, merged_artifact)
    reloaded = torch.load(artifact, map_location="cpu", weights_only=False)
    exact = reloaded["schema"] == SCHEMA and reloaded["protocol"] == reference_protocol
    exact = exact and reloaded["state"].keys() == state.keys()
    exact = exact and all(torch.equal(reloaded["state"][key], value) for key, value in state.items())
    if not exact:
        raise RuntimeError("merged artifact roundtrip failed")
    merged = {
        "schema": SCHEMA,
        "protocol": reference_protocol,
        "rows": [row.__dict__ for row in rows],
        "audits": audits,
        "summary": summarize(rows),
        "source_seed_results": [_path_metadata(directory / "result.json") for directory in seed_directories],
        "artifact": _path_metadata(artifact),
        "artifact_roundtrip_exact": True,
        "mechanical_parallel_seed_merge": True,
    }
    _write_json_exclusive(output, merged)
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-directories", type=Path, nargs="+", required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() == args.artifact.resolve() or args.output.exists() or args.artifact.exists():
        parser.error("output and artifact must be distinct nonexistent paths")
    if not all(directory.is_dir() for directory in args.seed_directories):
        parser.error("all seed directories must exist")
    return args


def main() -> None:
    args = parse_args()
    merge_seed_directories(tuple(args.seed_directories), args.output, args.artifact)


if __name__ == "__main__":
    main()
