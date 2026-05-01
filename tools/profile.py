from __future__ import annotations

from .benchmarking import profile as _impl

globals().update({name: getattr(_impl, name) for name in dir(_impl) if not name.startswith("__")})
__all__ = [name for name in dir(_impl) if not name.startswith("__")]
main = _impl.main


if __name__ == "__main__":
    main()
