from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


def _read_one(path: Path) -> dict[str, str]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-1] if rows else {}


def _float(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    try:
        value = row.get(key, "")
        return default if value == "" else float(value)
    except (TypeError, ValueError):
        return default


def _int(row: dict[str, Any], key: str, default: int = 0) -> int:
    try:
        value = row.get(key, "")
        return default if value == "" else int(float(value))
    except (TypeError, ValueError):
        return default


def _load_rows(results_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("*.csv")):
        if path.name.endswith(".time.csv"):
            continue
        row: dict[str, Any] = dict(_read_one(path))
        if not row:
            continue
        row["file"] = str(path)
        row["name"] = path.stem
        time_path = path.with_name(f"{path.stem}.time.csv")
        if time_path.exists():
            time_row = _read_one(time_path)
            row["elapsed_sec"] = time_row.get("elapsed_sec", "")
            row["gpu"] = time_row.get("gpu", "")
        else:
            row["elapsed_sec"] = ""
            row["gpu"] = ""
        rows.append(row)
    return rows


def _sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _int(row, "depth"),
        _int(row, "write_degree"),
        row.get("payload_variant", ""),
        _int(row, "comparator_kc"),
        row.get("comparator_write_policy", ""),
        row.get("comparator_reduction_layout", ""),
        _int(row, "comparator_output_tile_size"),
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _compact(row: dict[str, Any]) -> str:
    return (
        f"{row.get('payload_variant')} L{_int(row, 'depth')} kc={_int(row, 'comparator_kc')} "
        f"wd={_int(row, 'write_degree')} policy={row.get('comparator_write_policy')} "
        f"layout={row.get('comparator_reduction_layout')} tile={_int(row, 'comparator_output_tile_size')} "
        f"params={_int(row, 'total_params')} valid_acc={_float(row, 'valid_acc'):.4f} "
        f"valid_loss={_float(row, 'valid_loss'):.4f} train_acc={_float(row, 'train_acc'):.4f} "
        f"sec={_float(row, 'elapsed_sec'):.0f}"
    )


def _print_section(title: str, rows: list[dict[str, Any]]) -> None:
    print(f"\n## {title}")
    for row in sorted(rows, key=_sort_key):
        print(_compact(row))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize EMNIST two-sided comparator mechanism runs.")
    parser.add_argument("--results-dir", type=Path, default=Path("results/payload_width/emnist_balanced_twosided_mechanism"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    rows = _load_rows(args.results_dir)
    if args.out is not None:
        _write_csv(args.out, sorted(rows, key=_sort_key))

    core = [r for r in rows if _int(r, "depth") == 16]
    capacity = [
        r
        for r in core
        if r.get("comparator_write_policy") == "expander"
        and r.get("comparator_reduction_layout") == "scatter"
        and r.get("payload_variant") in {"comparator_sign_kc", "comparator_margin_kc", "comparator_signed_margin_kc", "comparator_two_sided_margin_kc"}
    ]
    policy = [
        r
        for r in core
        if r.get("payload_variant") in {"comparator_signed_margin_kc", "comparator_two_sided_margin_kc"}
        and _int(r, "comparator_kc") in {16, 24, 32}
    ]
    depth = [
        r
        for r in rows
        if r.get("payload_variant") in {"comparator_signed_margin_kc", "comparator_two_sided_margin_kc"}
        and _int(r, "depth") in {2, 4, 8, 16}
        and (
            (r.get("comparator_write_policy") == "expander" and r.get("comparator_reduction_layout") == "scatter")
            or r.get("comparator_reduction_layout") == "tile_local"
        )
    ]
    pareto = sorted(rows, key=lambda r: (_float(r, "valid_loss"), _float(r, "elapsed_sec")))[:20]

    _print_section("capacity matched source ablation", capacity)
    _print_section("output write geometry ablation", policy)
    _print_section("depth robustness", depth)
    _print_section("top valid-loss pareto candidates", pareto)


if __name__ == "__main__":
    main()
