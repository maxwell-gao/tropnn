from __future__ import annotations

import ctypes
import hashlib
import os
import shlex
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor

_BUILD_LOCK = threading.Lock()
_LIB: ctypes.CDLL | None = None


def _package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _kernel_root() -> Path:
    return _package_root() / "kernels" / "src"


def _zig_command() -> list[str]:
    env_zig = os.environ.get("TROPNN_ZIG")
    if env_zig:
        return shlex.split(env_zig)

    python_zig = shutil.which("python-zig")
    if python_zig:
        return [python_zig]

    return [sys.executable, "-m", "ziglang"]


def has_pairwise_zig() -> bool:
    try:
        subprocess.run([*_zig_command(), "version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return False
    return _kernel_root().joinpath("lib.zig").exists()


def _zig_version(command: list[str]) -> str:
    result = subprocess.run([*command, "version"], check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _source_digest(kernel_root: Path, zig_version: str) -> str:
    digest = hashlib.sha256()
    digest.update(zig_version.encode())
    digest.update(sys.platform.encode())
    digest.update(os.uname().machine.encode() if hasattr(os, "uname") else b"unknown")
    for path in sorted(kernel_root.rglob("*.zig")):
        stat = path.stat()
        digest.update(str(path.relative_to(kernel_root)).encode())
        digest.update(str(stat.st_mtime_ns).encode())
        digest.update(str(stat.st_size).encode())
    return digest.hexdigest()[:16]


def _shared_suffix() -> str:
    if sys.platform == "darwin":
        return ".dylib"
    if sys.platform == "win32":
        return ".dll"
    return ".so"


def _build_library() -> Path:
    kernel_root = _kernel_root()
    lib_zig = kernel_root / "lib.zig"
    if not lib_zig.exists():
        raise RuntimeError(f"Cannot find Zig kernels at {lib_zig}")

    command = _zig_command()
    try:
        version = _zig_version(command)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "Zig is not available for PairwiseLinear backend='zig'. Install the cpu extra with "
            "`uv sync --extra cpu`, or set TROPNN_ZIG to a Zig executable."
        ) from exc

    cache_dir = _package_root() / ".zig-cache" / "pairwise_cpu"
    cache_dir.mkdir(parents=True, exist_ok=True)
    zig_global_cache = cache_dir / "global"
    zig_local_cache = cache_dir / "local"
    zig_global_cache.mkdir(parents=True, exist_ok=True)
    zig_local_cache.mkdir(parents=True, exist_ok=True)
    lib_path = cache_dir / f"libtropnn_pairwise_{_source_digest(kernel_root, version)}{_shared_suffix()}"
    if lib_path.exists():
        return lib_path

    tmp_path = lib_path.with_suffix(lib_path.suffix + ".tmp")
    build_cmd = [
        *command,
        "build-lib",
        "-dynamic",
        "-fPIC",
        "-O",
        "ReleaseFast",
        "-mcpu=native",
        f"-femit-bin={tmp_path}",
        str(lib_zig),
    ]
    env = os.environ.copy()
    env["ZIG_GLOBAL_CACHE_DIR"] = str(zig_global_cache)
    env["ZIG_LOCAL_CACHE_DIR"] = str(zig_local_cache)
    result = subprocess.run(build_cmd, cwd=cache_dir, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(
            "Failed to build Zig pairwise CPU backend.\n"
            f"Command: {' '.join(build_cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    tmp_path.replace(lib_path)
    return lib_path


def _load_library() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    with _BUILD_LOCK:
        if _LIB is not None:
            return _LIB
        lib = ctypes.CDLL(str(_build_library()))
        size = ctypes.c_size_t
        ptr = ctypes.c_void_p
        common_args = [size, size, size, size, size, ptr, ptr, ptr, ptr, ptr]
        lib.lut_forward_batch_with_offsets_no_cache.argtypes = common_args
        lib.lut_forward_batch_with_offsets_no_cache.restype = None
        lib.lut_forward_batch_f16_no_cache.argtypes = common_args
        lib.lut_forward_batch_f16_no_cache.restype = None
        _LIB = lib
        return lib


def _ptr(tensor: Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(tensor.data_ptr())


def pairwise_zig_forward(
    latent: Tensor,
    anchors: Tensor,
    thresholds: Tensor,
    lut: Tensor,
    *,
    lut_dtype: Literal["f32", "f16"] = "f32",
) -> Tensor:
    if latent.device.type != "cpu":
        raise ValueError("PairwiseLinear backend='zig' requires CPU input tensors")
    if latent.dtype != torch.float32:
        raise TypeError(f"PairwiseLinear backend='zig' requires float32 compute tensors, got {latent.dtype}")
    if anchors.dtype != torch.long:
        raise TypeError(f"PairwiseLinear backend='zig' requires int64 anchors, got {anchors.dtype}")
    if thresholds.dtype != torch.float32:
        thresholds = thresholds.to(torch.float32)
    if lut_dtype not in {"f32", "f16"}:
        raise ValueError(f"lut_dtype must be 'f32' or 'f16', got {lut_dtype!r}")

    batch, steps, input_dim = latent.shape
    tables, comparisons, pair_width = anchors.shape
    if pair_width != 2:
        raise ValueError(f"anchors must have shape [tables, comparisons, 2], got {tuple(anchors.shape)}")
    if thresholds.shape != (tables, comparisons):
        raise ValueError(f"thresholds must have shape {(tables, comparisons)}, got {tuple(thresholds.shape)}")
    if lut.shape[:2] != (tables, 1 << comparisons):
        raise ValueError(f"lut has incompatible shape {tuple(lut.shape)} for tables={tables}, comparisons={comparisons}")

    output_dim = lut.shape[-1]
    item_count = batch * steps
    latent_flat = latent.reshape(item_count, input_dim).contiguous()
    anchors_flat = anchors.contiguous()
    thresholds_flat = thresholds.contiguous()
    output = torch.empty((item_count, output_dim), device="cpu", dtype=torch.float32)
    lib = _load_library()

    weights = lut.contiguous()
    if lut_dtype == "f16":
        if weights.dtype != torch.float16:
            weights = weights.to(torch.float16)
        lib.lut_forward_batch_f16_no_cache(
            item_count,
            tables,
            comparisons,
            input_dim,
            output_dim,
            _ptr(weights),
            _ptr(anchors_flat),
            _ptr(thresholds_flat),
            _ptr(latent_flat),
            _ptr(output),
        )
    else:
        if weights.dtype != torch.float32:
            weights = weights.to(torch.float32)
        lib.lut_forward_batch_with_offsets_no_cache(
            item_count,
            tables,
            comparisons,
            input_dim,
            output_dim,
            _ptr(weights),
            _ptr(anchors_flat),
            _ptr(thresholds_flat),
            _ptr(latent_flat),
            _ptr(output),
        )

    return output.view(batch, steps, output_dim)
