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

from torch import Tensor

_BUILD_LOCK = threading.Lock()
_LIB: ctypes.CDLL | None = None
_THREAD_ARGS_REGISTERED = False


def package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def kernel_root() -> Path:
    return package_root() / "kernels" / "src"


def zig_command() -> list[str]:
    env_zig = os.environ.get("TROPNN_ZIG")
    if env_zig:
        return shlex.split(env_zig)

    python_zig = shutil.which("python-zig")
    if python_zig:
        return [python_zig]

    return [sys.executable, "-m", "ziglang"]


def has_zig_backend() -> bool:
    try:
        subprocess.run([*zig_command(), "version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return False
    return kernel_root().joinpath("lib.zig").exists()


def _zig_version(command: list[str]) -> str:
    result = subprocess.run([*command, "version"], check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _source_digest(root: Path, zig_version: str) -> str:
    digest = hashlib.sha256()
    digest.update(zig_version.encode())
    digest.update(b"fno-single-threaded-link-libc")
    digest.update(sys.platform.encode())
    digest.update(os.uname().machine.encode() if hasattr(os, "uname") else b"unknown")
    for path in sorted(root.rglob("*.zig")):
        stat = path.stat()
        digest.update(str(path.relative_to(root)).encode())
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
    root = kernel_root()
    lib_zig = root / "lib.zig"
    if not lib_zig.exists():
        raise RuntimeError(f"Cannot find Zig kernels at {lib_zig}")

    command = zig_command()
    try:
        version = _zig_version(command)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "Zig is not available for tropnn backend='zig'. Install the cpu extra with "
            "`uv sync --extra cpu`, or set TROPNN_ZIG to a Zig executable."
        ) from exc

    cache_dir = package_root() / ".zig-cache" / "cpu"
    cache_dir.mkdir(parents=True, exist_ok=True)
    zig_global_cache = cache_dir / "global"
    zig_local_cache = cache_dir / "local"
    zig_global_cache.mkdir(parents=True, exist_ok=True)
    zig_local_cache.mkdir(parents=True, exist_ok=True)
    lib_path = cache_dir / f"libtropnn_cpu_{_source_digest(root, version)}{_shared_suffix()}"
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
        "-fno-single-threaded",
        "-lc",
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
            "Failed to build Zig CPU backend.\n"
            f"Command: {' '.join(build_cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    tmp_path.replace(lib_path)
    return lib_path


def load_zig_library() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    with _BUILD_LOCK:
        if _LIB is not None:
            return _LIB
        _LIB = ctypes.CDLL(str(_build_library()))
        return _LIB


def zig_num_threads() -> int:
    env_threads = os.environ.get("TROPNN_ZIG_THREADS")
    if env_threads:
        try:
            return max(0, int(env_threads))
        except ValueError as exc:
            raise ValueError(f"TROPNN_ZIG_THREADS must be an integer, got {env_threads!r}") from exc

    try:
        import torch

        return max(1, int(torch.get_num_threads()))
    except Exception:
        return max(1, os.cpu_count() or 1)


def configure_zig_threads(lib: ctypes.CDLL | None = None) -> None:
    global _THREAD_ARGS_REGISTERED
    if lib is None:
        lib = load_zig_library()
    if not _THREAD_ARGS_REGISTERED:
        lib.tropnn_set_num_threads.argtypes = [ctypes.c_size_t]
        lib.tropnn_set_num_threads.restype = None
        lib.tropnn_get_num_threads.argtypes = []
        lib.tropnn_get_num_threads.restype = ctypes.c_size_t
        _THREAD_ARGS_REGISTERED = True
    lib.tropnn_set_num_threads(zig_num_threads())


def tensor_ptr(tensor: Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(tensor.data_ptr())
