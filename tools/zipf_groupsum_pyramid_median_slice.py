from __future__ import annotations

import argparse
import json
from pathlib import Path

from tropnn.tools.zipf_groupsum_pclut_capacity_law import FormalConfig, _write_exclusive, train_formal_run

TABLES = (1, 2, 4, 8, 16, 32, 64, 128)
SEEDS = (0, 1, 2)
SUMMARY_SCHEMA = "zipf-groupsum-pyramid-median-slice-summary-v1"
COMPLETION_SCHEMA = "zipf-groupsum-pyramid-median-slice-completion-v1"


def configs(device: str) -> list[FormalConfig]:
    return [
        FormalConfig(
            "pyramid_signed_median",
            1024,
            32,
            1.0,
            1.0,
            tables,
            6,
            "level_biased",
            512,
            "torch",
            0.0,
            0.01,
            512,
            10_000,
            500,
            16_384,
            512,
            seed,
            device,
        )
        for tables in TABLES
        for seed in SEEDS
    ]


def run(args: argparse.Namespace) -> None:
    frozen = configs(args.device)
    if len(frozen) != 24 or len({config.run_key for config in frozen}) != 24:
        raise RuntimeError("pyramid median slice is not exactly 24 unique runs")
    for index, config in enumerate(frozen):
        if index % args.shard_count != args.shard_index:
            continue
        path = args.output_dir / "runs" / f"{config.run_key}.json"
        if path.exists():
            existing = json.loads(path.read_text())
            if existing.get("complete") is True and existing.get("run_key") == config.run_key:
                print(f"skip complete {config.run_key}", flush=True)
                continue
            raise RuntimeError(f"refusing to overwrite {path}")
        print(f"start {config.run_key}", flush=True)
        result = train_formal_run(config)
        _write_exclusive(path, result)
        print(
            f"done {config.run_key} val={result['validation']['total_loss']:.7g} "
            f"test={result['test']['total_loss']:.7g} seconds={result['train_seconds']:.2f}",
            flush=True,
        )


def summarize(source_dir: Path, output_dir: Path) -> dict[str, object]:
    paths = sorted((output_dir / "runs").glob("*.json"))
    rows = [json.loads(path.read_text()) for path in paths]
    expected = {config.run_key for config in configs("cuda:0")}
    if len(rows) != 24 or {row["run_key"] for row in rows} != expected:
        raise RuntimeError("pyramid median slice is incomplete")
    if any(
        row.get("complete") is not True
        or row.get("schema") != "zipf-groupsum-pclut-capacity-law-run-v3"
        or row["config"]["arm"] != "pyramid_signed_median"
        for row in rows
    ):
        raise RuntimeError("invalid pyramid median result")
    paired: list[dict[str, object]] = []
    for row in sorted(rows, key=lambda item: (item["config"]["tables"], item["config"]["seed"])):
        config = row["config"]
        sum_key = str(row["run_key"]).replace("pyramid-signed-median", "pyramid-signed-sum")
        sum_path = source_dir / "runs" / f"{sum_key}.json"
        sum_row = json.loads(sum_path.read_text())
        if sum_row.get("complete") is not True or sum_row["config"]["arm"] != "pyramid_signed_sum":
            raise RuntimeError(f"invalid paired sum result {sum_path}")
        paired.append(
            {
                "tables": config["tables"],
                "seed": config["seed"],
                "sum_run_key": sum_key,
                "median_run_key": row["run_key"],
                "validation_sum_loss": sum_row["validation"]["total_loss"],
                "validation_median_loss": row["validation"]["total_loss"],
                "test_sum_loss": sum_row["test"]["total_loss"],
                "test_median_loss": row["test"]["total_loss"],
                "test_sum_minus_median": sum_row["test"]["total_loss"] - row["test"]["total_loss"],
            }
        )
    summary = {
        "schema": SUMMARY_SCHEMA,
        "complete": True,
        "run_count": len(rows),
        "paired_count": len(paired),
        "selection_used": False,
        "enters_primary_capacity_gate": False,
        "paired": paired,
    }
    _write_exclusive(output_dir / "summary.json", summary)
    return summary


def seal(args: argparse.Namespace) -> None:
    summary = summarize(args.source_dir, args.output_dir)
    completion = {
        "schema": COMPLETION_SCHEMA,
        "complete": True,
        "run_count": summary["run_count"],
        "paired_count": summary["paired_count"],
    }
    _write_exclusive(args.output_dir / "completion.json", completion)
    print(json.dumps(completion, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Matched original pyramid-signed median D32/C6 T slice")
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--device", default="cuda:0")
    run_parser.add_argument("--shard-index", type=int, default=0)
    run_parser.add_argument("--shard-count", type=int, default=1)
    seal_parser = commands.add_parser("seal")
    seal_parser.add_argument("--source-dir", type=Path, required=True)
    seal_parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run":
        if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
            raise ValueError("shard_index must lie in [0, shard_count)")
        run(args)
        return
    seal(args)


if __name__ == "__main__":
    main()
