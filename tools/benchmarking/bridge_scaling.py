"""Architecture-agnostic scaling diagnostics that bridge linear-algebra and
combinatorial-routing kingdoms.

Given any model with

* a vector representation ``r_i in R^m`` for each feature ``i``  (linear-algebra side)
* and/or a chamber signature ``sigma_i in [C]^H``                (combinatorial side)

this module reports a single capacity unit ``K_eff`` that reduces to:

* effective frame size on linear-algebra architectures (paper, TropLinear),
* effective number of distinct chamber tuples on routed architectures
  (PairwiseLinear, TropZeroDenseLinear, TropDictLinear),

so that the SuperpositionScaling-style ``log L vs log K_eff`` fit can be
performed across architectures without privileging one side's metric.

Concrete quantities reported per model:

``K_metric(alpha)`` -- effective rank of the kernel ``M = exp(-alpha D)``,
    where ``D`` is the pairwise distinguishability matrix. Equals
    ``(tr M)^2 / ||M||_F^2``. Reduces to 1 when all features collapse and
    to ``n`` when all features are mutually orthogonal. The ``alpha``
    sweep controls the resolution: small ``alpha`` weights all distances
    softly, large ``alpha`` only counts very close pairs as overlapping.

``K_pack(tau)`` -- greedy anti-clique under ``D >= tau``. The number of
    features that can be picked while keeping every pair at least
    ``tau`` apart. This is the discrete capacity analog.

``chamber_entropy`` -- Shannon entropy (in nats) of the feature-induced
    chamber-tuple distribution, plus the count of distinct tuples. Available
    only when chamber signatures are supplied.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor


def vector_distinguishability(reps: Tensor) -> Tensor:
    """Return ``1 - cos^2`` between every pair of feature representations.

    The diagonal is forced to zero. Off-diagonal entries lie in ``[0, 1]``,
    where 0 means perfectly aligned and 1 means orthogonal.
    """
    if reps.ndim != 2:
        raise ValueError(f"expected (n, m) representation, got shape {tuple(reps.shape)}")
    unit = F.normalize(reps.float(), dim=1, eps=1e-12)
    sim = (unit @ unit.t()).square().clamp(0.0, 1.0)
    distance = (1.0 - sim).clamp_min(0.0)
    distance.fill_diagonal_(0.0)
    return distance


def chamber_distinguishability(indices: Tensor) -> Tensor:
    """Return ``1 - fraction of agreeing heads`` between every pair of features.

    The diagonal is zero (a feature agrees with itself everywhere).
    """
    if indices.ndim != 2:
        raise ValueError(f"expected (n, H) chamber indices, got shape {tuple(indices.shape)}")
    matches = (indices.unsqueeze(0) == indices.unsqueeze(1)).float()
    sim = matches.mean(dim=-1)
    distance = (1.0 - sim).clamp(0.0, 1.0)
    distance.fill_diagonal_(0.0)
    return distance


def effective_rank(distance: Tensor, *, alpha: float) -> float:
    """Effective rank of the similarity kernel ``M = exp(-alpha * D)``.

    Computes ``(tr M)^2 / ||M||_F^2``. With ``D`` as the distinguishability
    matrix this is the participation ratio of the kernel's eigenvalues:

    * 1 when every feature is identical, ``D = 0`` and ``M = J``,
    * approaches ``n`` when every off-diagonal ``D`` is large and the kernel
      decouples toward an identity matrix.
    """
    if distance.ndim != 2 or distance.shape[0] != distance.shape[1]:
        raise ValueError(f"expected square distance matrix, got shape {tuple(distance.shape)}")
    M = (-float(alpha) * distance.float()).exp()
    trace = M.diag().sum()
    frob_sq = M.square().sum().clamp_min(1e-12)
    return float((trace * trace / frob_sq).item())


def pack_capacity(distance: Tensor, *, tau: float) -> int:
    """Greedy anti-clique on the predicate ``D[i, j] >= tau``.

    Picks features in decreasing order of mean distance, keeping only the
    ones that remain at least ``tau`` away from every previously kept index.
    Returns the count.
    """
    if distance.ndim != 2 or distance.shape[0] != distance.shape[1]:
        raise ValueError(f"expected square distance matrix, got shape {tuple(distance.shape)}")
    n = distance.shape[0]
    if n == 0:
        return 0
    avg_dist = distance.float().mean(dim=1)
    order = torch.argsort(avg_dist, descending=True)
    kept = torch.zeros(n, dtype=torch.bool, device=distance.device)
    for idx_t in order.tolist():
        if kept.any():
            kept_idx = kept.nonzero(as_tuple=True)[0]
            if (distance[idx_t, kept_idx].float() < float(tau)).any().item():
                continue
        kept[idx_t] = True
    return int(kept.sum().item())


def chamber_entropy(indices: Tensor) -> tuple[float, int]:
    """Shannon entropy (nats) of the feature-induced chamber-tuple distribution.

    Returns ``(H, distinct_count)``.
    """
    if indices.ndim != 2:
        raise ValueError(f"expected (n, H) chamber indices, got shape {tuple(indices.shape)}")
    if indices.shape[0] == 0:
        return 0.0, 0
    sigs = indices.detach().cpu()
    unique, counts = torch.unique(sigs, dim=0, return_counts=True)
    probs = counts.float() / counts.sum().clamp_min(1)
    entropy = float((-probs * probs.log()).sum().item())
    return entropy, int(unique.shape[0])


def bridge_diagnostics(
    *,
    reps_vector: Tensor | None = None,
    reps_chamber: Tensor | None = None,
    alphas: Iterable[float] = (1.0, 4.0, 16.0),
    taus: Iterable[float] = (0.1, 0.3, 0.5),
) -> dict[str, float]:
    """Compute bridge capacity views on whichever representations are provided.

    At least one of ``reps_vector`` / ``reps_chamber`` must be given.
    Returned keys:

    * ``bridge_K_vec_a{alpha}``   for each alpha (vector kingdom)
    * ``bridge_pack_vec_t{tau}``  for each tau   (vector kingdom)
    * ``bridge_K_chamb_a{alpha}`` for each alpha (chamber kingdom)
    * ``bridge_pack_chamb_t{tau}``for each tau   (chamber kingdom)
    * ``bridge_chamber_entropy``  Shannon entropy of chamber tuples
    * ``bridge_chamber_distinct`` count of distinct chamber tuples
    """
    if reps_vector is None and reps_chamber is None:
        raise ValueError("at least one of reps_vector / reps_chamber must be provided")

    out: dict[str, float] = {}

    if reps_vector is not None:
        Dv = vector_distinguishability(reps_vector)
        for a in alphas:
            out[f"bridge_K_vec_a{float(a):g}"] = effective_rank(Dv, alpha=float(a))
        for t in taus:
            out[f"bridge_pack_vec_t{float(t):g}"] = float(pack_capacity(Dv, tau=float(t)))

    if reps_chamber is not None:
        Dc = chamber_distinguishability(reps_chamber)
        for a in alphas:
            out[f"bridge_K_chamb_a{float(a):g}"] = effective_rank(Dc, alpha=float(a))
        for t in taus:
            out[f"bridge_pack_chamb_t{float(t):g}"] = float(pack_capacity(Dc, tau=float(t)))
        entropy_nats, distinct = chamber_entropy(reps_chamber)
        out["bridge_chamber_entropy"] = entropy_nats
        out["bridge_chamber_distinct"] = float(distinct)

    return out


def fit_bridge_scaling(
    rows: list[dict[str, float | int | str]],
    *,
    capacity_key: str,
    loss_key: str = "final_loss",
) -> dict[str, float]:
    """Least-squares fit of ``log loss = a - beta * log capacity`` on rows.

    Rows missing the capacity or loss key, or with non-positive values, are
    skipped silently. Returns ``beta``, ``r2``, and the number of contributing
    points.
    """
    xs: list[float] = []
    ys: list[float] = []
    for row in rows:
        if capacity_key not in row or loss_key not in row:
            continue
        try:
            x = float(row[capacity_key])
            y = float(row[loss_key])
        except (TypeError, ValueError):
            continue
        if x <= 0 or y <= 0 or not (math.isfinite(x) and math.isfinite(y)):
            continue
        xs.append(x)
        ys.append(y)
    if len(xs) < 2:
        return {"beta": float("nan"), "r2": float("nan"), "points": float(len(xs))}
    xt = torch.tensor(xs).log()
    yt = torch.tensor(ys).log()
    cx = xt - xt.mean()
    cy = yt - yt.mean()
    denom = cx.square().sum().clamp_min(1e-12)
    slope = float((cx * cy).sum().item() / denom.item())
    pred = yt.mean() + slope * cx
    ss_res = float((yt - pred).square().sum().item())
    ss_tot = float(cy.square().sum().clamp_min(1e-12).item())
    return {
        "beta": -slope,
        "r2": 1.0 - ss_res / ss_tot,
        "points": float(len(xs)),
    }


def fit_bridge_exponents_per_family(
    rows: list[dict[str, float | int | str]],
    *,
    capacity_keys: Iterable[str],
    loss_key: str = "final_loss",
) -> list[dict[str, float | str]]:
    """Within-family and pooled bridge scaling fits.

    For each capacity key, emits one fit per family plus a pooled fit across
    families. The pooled fit is the bridge headline: it asks whether all
    architectures sit on the same ``log L vs log K`` line.
    """
    by_family: dict[str, list[dict[str, float | int | str]]] = {}
    for row in rows:
        by_family.setdefault(str(row.get("family", "")), []).append(row)

    out: list[dict[str, float | str]] = []
    for capacity_key in capacity_keys:
        for family, group in sorted(by_family.items()):
            fit = fit_bridge_scaling(group, capacity_key=capacity_key, loss_key=loss_key)
            out.append(
                {
                    "scope": "per_family",
                    "family": family,
                    "capacity": capacity_key,
                    "loss": loss_key,
                    **fit,
                }
            )
        fit = fit_bridge_scaling(rows, capacity_key=capacity_key, loss_key=loss_key)
        out.append(
            {
                "scope": "pooled",
                "family": "*",
                "capacity": capacity_key,
                "loss": loss_key,
                **fit,
            }
        )
    return out
