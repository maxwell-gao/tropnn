from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

METRICS = (
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "gpu__time_duration.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
)


@dataclass(frozen=True)
class Case:
    family: str
    backend: str
    mode: str


def _float_field(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value else 0.0


def _parse_ncu_raw_csv(text: str) -> dict[str, float]:
    lines = text.splitlines()
    start = next((idx for idx, line in enumerate(lines) if line.startswith('"ID","Process ID"')), None)
    if start is None:
        raise RuntimeError(f"ncu raw CSV header not found. Output tail:\n{os.linesep.join(lines[-20:])}")

    reader = csv.DictReader(StringIO("\n".join(lines[start:])))
    kernels = 0
    read_bytes = 0.0
    write_bytes = 0.0
    duration_ns = 0.0
    pct_weighted = 0.0
    pct_max = 0.0
    for row in reader:
        if not row.get("ID") or not row.get("Kernel Name"):
            continue
        kernels += 1
        read = _float_field(row, "dram__bytes_read.sum")
        write = _float_field(row, "dram__bytes_write.sum")
        duration = _float_field(row, "gpu__time_duration.sum")
        pct = _float_field(row, "dram__throughput.avg.pct_of_peak_sustained_elapsed")
        read_bytes += read
        write_bytes += write
        duration_ns += duration
        pct_weighted += pct * duration
        pct_max = max(pct_max, pct)

    total_bytes = read_bytes + write_bytes
    seconds = duration_ns * 1e-9
    return {
        "kernels": float(kernels),
        "read_bytes": read_bytes,
        "write_bytes": write_bytes,
        "total_bytes": total_bytes,
        "duration_ns": duration_ns,
        "bandwidth_gb_s": total_bytes / seconds / 1e9 if seconds > 0.0 else 0.0,
        "dram_pct_weighted": pct_weighted / duration_ns if duration_ns > 0.0 else 0.0,
        "dram_pct_max": pct_max,
    }


def _run_case(args: argparse.Namespace, case: Case) -> dict[str, float]:
    env = os.environ.copy()
    env.setdefault("CC", "/usr/bin/gcc")
    env.setdefault("CXX", "/usr/bin/g++")
    env.setdefault("CUDA_HOME", "/usr/local/cuda-12.8")
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    cwd = Path.cwd()
    env["PYTHONPATH"] = f"{cwd}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(cwd)

    command = [
        args.ncu_bin,
        "--target-processes",
        "all",
        "--profile-from-start",
        "off",
        "--metrics",
        ",".join(METRICS),
        "--csv",
        "--page",
        "raw",
        "--print-units",
        "base",
        "--print-fp",
        sys.executable,
        "-m",
        "tropnn.tools.ncu_memory_case",
        "--family",
        case.family,
        "--backend",
        case.backend,
        "--mode",
        case.mode,
        "--batch-size",
        str(args.batch_size),
        "--in-features",
        str(args.in_features),
        "--out-features",
        str(args.out_features),
        "--hidden-features",
        str(args.hidden_features),
        "--mlp-depth",
        str(args.mlp_depth),
        "--heads",
        str(args.heads),
        "--cells",
        str(args.cells),
        "--code-dim",
        str(args.code_dim),
        "--pairwise-tables",
        str(args.pairwise_tables),
        "--comparisons",
        str(args.comparisons),
        "--warmup",
        str(args.warmup),
        "--repeats",
        str(args.repeats),
        "--seed",
        str(args.seed),
    ]
    if args.sudo:
        command = ["sudo", "-E", *command]

    proc = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ncu failed for {case.family}:{case.backend}:{case.mode}\n{proc.stdout}")
    result = _parse_ncu_raw_csv(proc.stdout)
    result["case_stdout_bytes"] = float(len(proc.stdout.encode()))
    return result


def _print_row(case: Case, result: dict[str, float]) -> None:
    mib = 1024.0 * 1024.0
    print(
        f"{case.family:8s} {case.backend:8s} {case.mode:7s} "
        f"kernels={int(result['kernels']):2d} "
        f"read_mib={result['read_bytes'] / mib:9.2f} "
        f"write_mib={result['write_bytes'] / mib:9.2f} "
        f"total_mib={result['total_bytes'] / mib:9.2f} "
        f"time_us={result['duration_ns'] / 1000.0:9.2f} "
        f"bw_gb_s={result['bandwidth_gb_s']:9.2f} "
        f"dram_pct_w={result['dram_pct_weighted']:6.2f} "
        f"dram_pct_max={result['dram_pct_max']:6.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ncu DRAM metric sweeps for dense, tropical, and pairwise tropnn cases.")
    parser.add_argument("--ncu-bin", default="/usr/local/cuda-12.8/bin/ncu")
    parser.add_argument("--sudo", action="store_true", help="Run ncu through sudo -E for GPU performance counter access.")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--families", default="linear,mlp,tropical,pairwise")
    parser.add_argument("--modes", default="forward,train")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--in-features", type=int, default=256)
    parser.add_argument("--out-features", type=int, default=512)
    parser.add_argument("--hidden-features", type=int, default=512)
    parser.add_argument("--mlp-depth", type=int, default=2)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--cells", type=int, default=4)
    parser.add_argument("--code-dim", type=int, default=32)
    parser.add_argument("--pairwise-tables", type=int, default=136)
    parser.add_argument("--comparisons", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    families = [family.strip() for family in args.families.split(",") if family.strip()]
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    cases = [
        Case(family=family, backend=("tilelang" if family in {"tropical", "pairwise"} else "torch"), mode=mode)
        for mode in modes
        for family in families
    ]

    print(
        "family   backend  mode    kernels read_mib  write_mib total_mib "
        "time_us   bw_gb_s  dram_pct_w dram_pct_max"
    )
    for case in cases:
        _print_row(case, _run_case(args, case))


if __name__ == "__main__":
    main()
