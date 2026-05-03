"""Two-axis scaling analysis for routed-LUT bridge sweeps.

Performs three orthogonal analyses on pairwise ``(T, L)`` and engram
``(K, M)`` sweep outputs:

1. **Chinchilla-style 2D fit**: ``L = A * a1^(-alpha) + B * a2^(-beta) + E``
   in the architecture-native coordinates. Compares against a single-axis
   reduction through bridge ``distinct`` count.

2. **Oriented-matroid prediction for pairwise distinct chambers**:
   for one-hot ``eye(n)`` inputs the distinct chamber count of ``T*L``
   random pairwise comparators is closely approximated by

   ``distinct(T, L, n) approx n * (1 - exp(-T*L/n)) + 1``

   which follows from "feature ``i`` is distinguished iff some comparator
   has ``a_k = i``" (with default thresholds).

3. **CRT prediction for engram distinct chambers**:
   ``distinct(K, M, n) approx min(n, prod(consecutive primes >= M, K))``.

The predicted ``distinct`` can then replace the per-architecture axes,
giving a universal capacity unit. Loss-vs-predicted-distinct is fit and
reported alongside the empirical fit.

Usage::

    python -m tropnn.tools.analyze_two_axis \
      --pairwise-glob 'results/scaling_benchmark/pairwise_dense/summary-*.json' \
      --engram-glob   'results/scaling_benchmark/engram/summary-engram-bridge-crt-*.json'
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from collections import defaultdict
from dataclasses import dataclass

try:
    import numpy as np
    from scipy.optimize import minimize  # type: ignore[import-not-found]

    _HAS_SCIPY = True
except ImportError:
    np = None  # type: ignore[assignment]
    _HAS_SCIPY = False


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def _consecutive_primes(start: int, count: int) -> list[int]:
    primes: list[int] = []
    candidate = max(2, start)
    while len(primes) < count:
        while not _is_prime(candidate):
            candidate += 1
        primes.append(candidate)
        candidate += 1
    return primes


def predicted_pairwise_distinct(T: int, L: int, n: int) -> float:
    """Oriented-matroid prediction for pairwise chamber count on eye(n)."""
    coverage = 1.0 - math.exp(-(T * L) / max(1, n))
    return n * coverage + 1.0


def predicted_engram_distinct(K: int, M: int, n: int) -> float:
    """CRT prediction for engram chamber count on eye(n)."""
    primes = _consecutive_primes(M, K)
    log_product = sum(math.log(p) for p in primes)
    log_n = math.log(max(1, n))
    return float(math.exp(min(log_product, log_n)))


@dataclass
class FitResult:
    name: str
    params: dict[str, float]
    log_rmse: float
    n_points: int


def _log_rmse(y_pred: list[float], y_true: list[float]) -> float:
    n = len(y_true)
    if n == 0:
        return float("nan")
    sq = sum((math.log(max(1e-30, p)) - math.log(max(1e-30, t))) ** 2 for p, t in zip(y_pred, y_true))
    return math.sqrt(sq / n)


def fit_loglog_power_law(xs: list[float], ys: list[float]) -> tuple[float, float, int]:
    pts = [(math.log(x), math.log(y)) for x, y in zip(xs, ys) if x > 0 and y > 0 and math.isfinite(x) and math.isfinite(y)]
    n = len(pts)
    if n < 2:
        return float("nan"), float("nan"), n
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    cov = sum((p[0] - mx) * (p[1] - my) for p in pts)
    vx = sum((p[0] - mx) ** 2 for p in pts)
    if vx <= 0:
        return float("nan"), float("nan"), n
    slope = cov / vx
    pred = [my + slope * (p[0] - mx) for p in pts]
    ss_res = sum((p[1] - q) ** 2 for p, q in zip(pts, pred))
    ss_tot = sum((p[1] - my) ** 2 for p in pts)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return -slope, r2, n


def fit_two_axis_chinchilla(
    a1_values: list[float],
    a2_values: list[float],
    losses: list[float],
    *,
    a2_is_exponential: bool = False,
) -> FitResult:
    """Fit ``L = A * a1^(-alpha) + B * f(a2)^(-beta) + E`` via L-BFGS-B in log space.

    ``a2_is_exponential=True`` substitutes ``2^(beta * a2)`` for the second
    term, which is the natural form for pairwise ``L`` (cells = 2^L).
    """
    if not _HAS_SCIPY:
        raise RuntimeError("scipy is required for two-axis Chinchilla fit")

    a1 = np.asarray(a1_values, dtype=np.float64)
    a2 = np.asarray(a2_values, dtype=np.float64)
    y = np.asarray(losses, dtype=np.float64)
    log_y = np.log(y)

    def predict(params):
        log_a, alpha, log_b, beta, log_e = params
        if a2_is_exponential:
            return np.exp(log_a) / np.power(a1, alpha) + np.exp(log_b) / np.power(2.0, beta * a2) + np.exp(log_e)
        return np.exp(log_a) / np.power(a1, alpha) + np.exp(log_b) / np.power(a2, beta) + np.exp(log_e)

    def loss(params):
        pred = predict(params)
        if not np.all(np.isfinite(pred)) or np.any(pred <= 0):
            return 1e9
        return float(np.mean((np.log(pred) - log_y) ** 2))

    best = None
    for log_a in (-2.0, 0.0, 2.0):
        for log_b in (-2.0, 0.0, 2.0):
            x0 = np.array([log_a, 0.4, log_b, 0.4, math.log(min(y) * 0.5)])
            try:
                result = minimize(
                    loss,
                    x0,
                    method="L-BFGS-B",
                    bounds=[(-15, 15), (0.0, 4.0), (-15, 15), (0.0, 4.0), (-30, 0.0)],
                )
            except Exception:
                continue
            if best is None or result.fun < best.fun:
                best = result

    assert best is not None
    log_a, alpha, log_b, beta, log_e = best.x
    pred = predict(best.x).tolist()
    rmse = _log_rmse(pred, list(y))
    return FitResult(
        name="2d_chinchilla",
        params={
            "A": math.exp(log_a),
            "alpha": float(alpha),
            "B": math.exp(log_b),
            "beta": float(beta),
            "E": math.exp(log_e),
        },
        log_rmse=rmse,
        n_points=len(y),
    )


def fit_one_axis_with_floor(xs: list[float], ys: list[float]) -> FitResult:
    """Fit ``L = A * x^(-alpha) + E`` via L-BFGS-B in log space.

    Returns NaN params if scipy is unavailable.
    """
    if not _HAS_SCIPY:
        return FitResult(name="1d_with_floor", params={}, log_rmse=float("nan"), n_points=len(ys))

    x_arr = np.asarray(xs, dtype=np.float64)
    y_arr = np.asarray(ys, dtype=np.float64)
    log_y = np.log(y_arr)

    def predict(params):
        log_a, alpha, log_e = params
        return np.exp(log_a) / np.power(x_arr, alpha) + np.exp(log_e)

    def loss(params):
        pred = predict(params)
        if not np.all(np.isfinite(pred)) or np.any(pred <= 0):
            return 1e9
        return float(np.mean((np.log(pred) - log_y) ** 2))

    best = None
    for log_a in (-2.0, 0.0, 2.0):
        x0 = np.array([log_a, 0.4, math.log(min(y_arr) * 0.5)])
        try:
            result = minimize(loss, x0, method="L-BFGS-B", bounds=[(-15, 15), (0.0, 4.0), (-30, 0.0)])
        except Exception:
            continue
        if best is None or result.fun < best.fun:
            best = result

    assert best is not None
    log_a, alpha, log_e = best.x
    pred = predict(best.x).tolist()
    rmse = _log_rmse(pred, list(y_arr))
    return FitResult(
        name="1d_with_floor",
        params={"A": math.exp(log_a), "alpha": float(alpha), "E": math.exp(log_e)},
        log_rmse=rmse,
        n_points=len(ys),
    )


def _load_runs(pattern: str) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(glob.glob(pattern)):
        try:
            payload = json.load(open(path))
        except Exception:
            continue
        rows.extend(payload.get("runs", []))
    return rows


def _avg(rows: list[dict], key: str) -> float:
    vs = [r.get(key) for r in rows if isinstance(r.get(key), (int, float)) and math.isfinite(r.get(key))]
    return sum(vs) / len(vs) if vs else float("nan")


def analyze_pairwise(rows: list[dict]) -> None:
    if not rows:
        print("(no pairwise rows)")
        return

    print(f"\n=== pairwise: {len(rows)} runs ===")
    by_tl = defaultdict(list)
    for r in rows:
        by_tl[(int(r["tables"]), int(r["comparisons"]))].append(r)

    print("\nOriented-matroid distinct prediction vs measured:")
    print(f"{'T':>5} {'L':>3} {'n':>5} {'pred':>8} {'meas':>8} {'ratio':>7}")
    by_tl_pred: list[tuple[int, int, float, float]] = []
    for (T, L), rs in sorted(by_tl.items()):
        n_features = int(_avg(rs, "n_features"))
        pred = predicted_pairwise_distinct(T, L, n_features)
        meas = _avg(rs, "bridge_chamber_distinct")
        ratio = meas / pred if pred > 0 else float("nan")
        by_tl_pred.append((T, L, pred, meas))
        print(f"{T:>5} {L:>3} {n_features:>5} {pred:>8.1f} {meas:>8.1f} {ratio:>7.3f}")

    print("\n2D Chinchilla fit  L = A*T^(-alpha) + B*2^(-beta*L) + E")
    if _HAS_SCIPY:
        ts: list[float] = []
        ls: list[float] = []
        ys: list[float] = []
        for r in rows:
            T = float(r["tables"])
            L = float(r["comparisons"])
            y = float(r["final_loss"])
            if T > 0 and L > 0 and y > 0:
                ts.append(T)
                ls.append(L)
                ys.append(y)
        fit = fit_two_axis_chinchilla(ts, ls, ys, a2_is_exponential=True)
        print(f"  A={fit.params['A']:.4g}  alpha_T={fit.params['alpha']:.3f}")
        print(f"  B={fit.params['B']:.4g}  beta_L ={fit.params['beta']:.3f}")
        print(f"  E={fit.params['E']:.4g}  log_RMSE={fit.log_rmse:.3f}  pts={fit.n_points}")
    else:
        print("  (scipy not installed; skipping)")

    print("\n1D power law in measured distinct  L = A*distinct^(-alpha) + E")
    distincts = [r["bridge_chamber_distinct"] for r in rows if r.get("bridge_chamber_distinct", 0) > 0]
    losses = [r["final_loss"] for r in rows if r.get("bridge_chamber_distinct", 0) > 0]
    if _HAS_SCIPY:
        fit = fit_one_axis_with_floor(distincts, losses)
        print(f"  A={fit.params['A']:.4g}  alpha={fit.params['alpha']:.3f}  E={fit.params['E']:.4g}  log_RMSE={fit.log_rmse:.3f}")
    beta_meas, r2_meas, _ = fit_loglog_power_law(distincts, losses)
    print(f"  log-log slope only (no floor): beta={beta_meas:.3f} R2={r2_meas:.3f}")

    print("\n1D power law in PREDICTED distinct (oriented-matroid)")
    pred_distincts = [predicted_pairwise_distinct(int(r["tables"]), int(r["comparisons"]), int(r["n_features"])) for r in rows]
    if _HAS_SCIPY:
        fit = fit_one_axis_with_floor(pred_distincts, [r["final_loss"] for r in rows])
        print(f"  A={fit.params['A']:.4g}  alpha={fit.params['alpha']:.3f}  E={fit.params['E']:.4g}  log_RMSE={fit.log_rmse:.3f}")
    beta_pred, r2_pred, _ = fit_loglog_power_law(pred_distincts, [r["final_loss"] for r in rows])
    print(f"  log-log slope only (no floor): beta={beta_pred:.3f} R2={r2_pred:.3f}")


def analyze_engram(rows: list[dict]) -> None:
    if not rows:
        print("(no engram rows)")
        return

    rows = [r for r in rows if int(r.get("engram_heads", 0)) > 0 and int(r.get("engram_table_size", 0)) > 0]
    rows = [r for r in rows if int(r["model_dim"]) // int(r["engram_heads"]) >= 8]
    print(f"\n=== engram (K<=model_dim/8 for non-collapsed regime): {len(rows)} runs ===")

    by_km = defaultdict(list)
    for r in rows:
        by_km[(int(r["engram_heads"]), int(r["engram_table_size"]))].append(r)

    print("\nCRT distinct prediction vs measured:")
    print(f"{'K':>3} {'M':>4} {'n':>5} {'pred':>8} {'meas':>8} {'ratio':>7}")
    for (K, M), rs in sorted(by_km.items()):
        n_features = int(_avg(rs, "n_features"))
        pred = predicted_engram_distinct(K, M, n_features)
        meas = _avg(rs, "bridge_chamber_distinct")
        ratio = meas / pred if pred > 0 else float("nan")
        print(f"{K:>3} {M:>4} {n_features:>5} {pred:>8.1f} {meas:>8.1f} {ratio:>7.3f}")

    print("\n2D Chinchilla fit  L = A*K^(-alpha) + B*M^(-beta) + E")
    if _HAS_SCIPY:
        ks = [float(r["engram_heads"]) for r in rows]
        ms = [float(r["engram_table_size"]) for r in rows]
        ys = [float(r["final_loss"]) for r in rows]
        fit = fit_two_axis_chinchilla(ks, ms, ys, a2_is_exponential=False)
        print(f"  A={fit.params['A']:.4g}  alpha_K={fit.params['alpha']:.3f}")
        print(f"  B={fit.params['B']:.4g}  beta_M ={fit.params['beta']:.3f}")
        print(f"  E={fit.params['E']:.4g}  log_RMSE={fit.log_rmse:.3f}  pts={fit.n_points}")
    else:
        print("  (scipy not installed; skipping)")

    print("\n1D power law in K*M (total embedding rows)")
    tot = [int(r["engram_heads"]) * int(r["engram_table_size"]) for r in rows]
    losses = [float(r["final_loss"]) for r in rows]
    if _HAS_SCIPY:
        fit = fit_one_axis_with_floor(tot, losses)
        print(f"  A={fit.params['A']:.4g}  alpha={fit.params['alpha']:.3f}  E={fit.params['E']:.4g}  log_RMSE={fit.log_rmse:.3f}")
    beta, r2, _ = fit_loglog_power_law(tot, losses)
    print(f"  log-log slope only (no floor): beta={beta:.3f} R2={r2:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-axis scaling analysis for routed-LUT bridge sweeps.")
    parser.add_argument("--pairwise-glob", type=str, default="")
    parser.add_argument("--engram-glob", type=str, default="")
    args = parser.parse_args()

    if args.pairwise_glob:
        analyze_pairwise(_load_runs(args.pairwise_glob))
    if args.engram_glob:
        analyze_engram(_load_runs(args.engram_glob))


if __name__ == "__main__":
    main()
