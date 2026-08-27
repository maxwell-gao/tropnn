from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from tropnn.layers.hard_lookup import HardLookupSpec, hard_route


@dataclass(frozen=True)
class ArmResult:
    seed: int
    family: str
    total_bits: int
    tables: int
    comparisons_per_table: int
    train_r2: float
    held_mse: float
    held_nmse: float
    held_r2: float
    held_cosine: float
    teacher_output_mse: float
    teacher_output_nmse: float
    teacher_output_r2: float
    teacher_output_cosine: float
    reconstruction_teacher_nmse_max_abs_difference: float
    mean_table_entropy_bits: float
    mean_observed_rows: float
    maximum_row_mass: float
    mean_bit_one_probability: float
    maximum_bit_balance_error: float
    mean_absolute_bit_correlation: float
    maximum_absolute_bit_correlation: float
    mean_train_held_table_tv: float
    unseen_held_row_fraction: float
    decoder_rank: int
    decoder_condition_number: float
    fitted_degrees_of_freedom: int
    deployed_row_scalars: int
    seconds: float


def parse_int_tuple(text: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list of positive integers")
    return values


def parse_seed_tuple(text: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if not values or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list of nonnegative integers")
    return values


def orthogonal_teacher_float64(dim: int, seed: int) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(10_000 + seed)
    raw = torch.randn(dim, dim, generator=generator, dtype=torch.float64)
    q, r = torch.linalg.qr(raw)
    signs = torch.sign(torch.diag(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return q * signs.view(1, -1)


def sample_paired_normal_bank(dim: int, count: int, seed: int) -> tuple[Tensor, Tensor, dict[str, float | int]]:
    """Sample dense normals whose nearest oriented pair roots are unique.

    For a unit dense normal ``w``, the pair root maximizing positive cosine is
    ``(e_argmax(w) - e_argmin(w)) / sqrt(2)``.  Rejecting duplicate projective
    roots prevents a pair arm from silently receiving fewer effective bits.
    """

    if dim < 2 or count < 1:
        raise ValueError("normal bank requires dim>=2 and count>=1")
    if count > dim * (dim - 1) // 2:
        raise ValueError("count exceeds the number of projectively unique pair roots")
    generator = torch.Generator(device="cpu").manual_seed(40_000 + seed)
    dense: list[Tensor] = []
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    attempts = 0
    while len(dense) < count:
        attempts += 1
        if attempts > 1_000_000:
            raise RuntimeError("failed to sample enough unique pair projections")
        normal = torch.randn(dim, generator=generator, dtype=torch.float64)
        normal /= normal.norm().clamp_min(1e-30)
        high = int(normal.argmax())
        low = int(normal.argmin())
        canonical = (min(high, low), max(high, low))
        if canonical in seen:
            continue
        seen.add(canonical)
        dense.append(normal)
        pairs.append((high, low))
    dense_tensor = torch.stack(dense)
    pair_tensor = torch.tensor(pairs, dtype=torch.long)
    cosine = (dense_tensor[torch.arange(count), pair_tensor[:, 0]] - dense_tensor[torch.arange(count), pair_tensor[:, 1]]) / math.sqrt(2.0)
    return (
        dense_tensor,
        pair_tensor,
        {
            "sampling_attempts": attempts,
            "rejected_projective_pair_duplicates": attempts - count,
            "mean_nearest_pair_cosine": float(cosine.mean()),
            "minimum_nearest_pair_cosine": float(cosine.min()),
            "maximum_nearest_pair_cosine": float(cosine.max()),
            "mean_nearest_pair_angle_degrees": float(torch.rad2deg(torch.acos(cosine.clamp(-1.0, 1.0))).mean()),
        },
    )


def dense_codes(x: Tensor, normals: Tensor, *, depth: int) -> tuple[Tensor, Tensor]:
    if normals.shape[0] % depth:
        raise ValueError("normal count must be divisible by depth")
    margins = x.to(torch.float64) @ normals.to(torch.float64).T
    branches = margins >= 0
    powers = 2 ** torch.arange(depth - 1, -1, -1, dtype=torch.long)
    codes = (branches.reshape(x.shape[0], -1, depth).long() * powers).sum(-1)
    return codes, branches


def pair_codes(x: Tensor, pairs: Tensor, *, depth: int) -> tuple[Tensor, Tensor]:
    if pairs.shape[0] % depth:
        raise ValueError("pair count must be divisible by depth")
    tables = pairs.shape[0] // depth
    supports = pairs.reshape(tables, depth, 2)
    thresholds = torch.zeros(tables, depth, dtype=x.dtype)
    spec = HardLookupSpec(x.shape[1], x.shape[1], depth, "pair", "flat", surrogate="none")
    route = hard_route(x, supports, thresholds, spec)
    return route.codes, route.branches.reshape(x.shape[0], -1)


def _feature_sufficient_statistics(codes: Tensor, target: Tensor, rows: int) -> tuple[Tensor, Tensor]:
    """Reference-code sufficient statistics for the full additive LUT span."""

    codes = codes.to(device="cpu", dtype=torch.long)
    y = target.to(device="cpu", dtype=torch.float64)
    samples, tables = codes.shape
    nonreference = rows - 1
    features = 1 + tables * nonreference
    gram = torch.zeros(features, features, dtype=torch.float64)
    cross = torch.zeros(features, y.shape[1], dtype=torch.float64)
    gram[0, 0] = samples
    cross[0] = y.sum(0)
    counts: list[Tensor] = []
    sums: list[Tensor] = []
    for table in range(tables):
        count = torch.bincount(codes[:, table], minlength=rows).to(torch.float64)
        value_sum = torch.zeros(rows, y.shape[1], dtype=torch.float64)
        value_sum.index_add_(0, codes[:, table], y)
        counts.append(count)
        sums.append(value_sum)
        block = slice(1 + table * nonreference, 1 + (table + 1) * nonreference)
        gram[0, block] = count[:nonreference]
        gram[block, 0] = count[:nonreference]
        gram[block, block] = torch.diag(count[:nonreference])
        cross[block] = value_sum[:nonreference]
    for left in range(tables):
        left_block = slice(1 + left * nonreference, 1 + (left + 1) * nonreference)
        for right in range(left + 1, tables):
            right_block = slice(1 + right * nonreference, 1 + (right + 1) * nonreference)
            joint = torch.bincount(codes[:, left] * rows + codes[:, right], minlength=rows * rows)
            block = joint.reshape(rows, rows)[:nonreference, :nonreference].to(torch.float64)
            gram[left_block, right_block] = block
            gram[right_block, left_block] = block.T
    return gram, cross


def fit_additive_lut(codes: Tensor, target: Tensor, *, rows: int) -> tuple[Tensor, dict[str, float | int]]:
    """Fit the exact empirical least-squares optimum in the additive LUT class."""

    gram, cross = _feature_sufficient_statistics(codes, target, rows)
    rank = int(torch.linalg.matrix_rank(gram, atol=1e-10, rtol=1e-12))
    if rank != gram.shape[0]:
        raise RuntimeError(f"reference-coded additive design is rank deficient: {rank}/{gram.shape[0]}")
    coefficients = torch.linalg.solve(gram, cross)
    condition = float(torch.linalg.cond(gram))
    tables = codes.shape[1]
    nonreference = rows - 1
    payload = torch.zeros(tables, rows, target.shape[1], dtype=torch.float64)
    payload += coefficients[0].reshape(1, 1, -1) / tables
    for table in range(tables):
        block = slice(1 + table * nonreference, 1 + (table + 1) * nonreference)
        payload[table, :nonreference] += coefficients[block]
    return payload, {"rank": rank, "condition_number": condition, "degrees_of_freedom": gram.shape[0]}


def additive_lookup(codes: Tensor, payload: Tensor) -> Tensor:
    tables = codes.shape[1]
    table = torch.arange(tables, device=codes.device).view(1, tables)
    return payload.to(codes.device)[table, codes].sum(1)


def regression_metrics(prediction: Tensor, target: Tensor) -> tuple[float, float, float, float]:
    prediction = prediction.to(torch.float64)
    target = target.to(torch.float64)
    mse = float(F.mse_loss(prediction, target))
    energy = float(target.square().mean().clamp_min(1e-30))
    nmse = mse / energy
    cosine = float(F.cosine_similarity(prediction, target, dim=-1).mean())
    return mse, nmse, 1.0 - nmse, cosine


def route_health(train_codes: Tensor, held_codes: Tensor, branches: Tensor, rows: int) -> dict[str, float]:
    entropies: list[float] = []
    observed: list[float] = []
    tvs: list[float] = []
    maximum_mass = 0.0
    unseen_tokens = torch.zeros(held_codes.shape[0], dtype=torch.bool)
    for table in range(train_codes.shape[1]):
        train_count = torch.bincount(train_codes[:, table], minlength=rows).double()
        held_count = torch.bincount(held_codes[:, table], minlength=rows).double()
        train_probability = train_count / train_count.sum()
        held_probability = held_count / held_count.sum()
        positive = held_probability > 0
        entropies.append(float(-(held_probability[positive] * held_probability[positive].log2()).sum()))
        observed.append(float(positive.sum()))
        maximum_mass = max(maximum_mass, float(held_probability.max()))
        tvs.append(float(0.5 * (train_probability - held_probability).abs().sum()))
        unseen_rows = train_count == 0
        unseen_tokens |= unseen_rows[held_codes[:, table]]
    branch64 = branches.to(torch.float64)
    probability = branch64.mean(0)
    centered = branch64 - probability
    variance = centered.square().mean(0).clamp_min(1e-30)
    correlation = (centered.T @ centered / branch64.shape[0]) / torch.sqrt(variance[:, None] * variance[None, :])
    mask = ~torch.eye(correlation.shape[0], dtype=torch.bool)
    off_diagonal = correlation[mask].abs()
    return {
        "mean_table_entropy_bits": sum(entropies) / len(entropies),
        "mean_observed_rows": sum(observed) / len(observed),
        "maximum_row_mass": maximum_mass,
        "mean_bit_one_probability": float(probability.mean()),
        "maximum_bit_balance_error": float((probability - 0.5).abs().max()),
        "mean_absolute_bit_correlation": float(off_diagonal.mean()) if off_diagonal.numel() else 0.0,
        "maximum_absolute_bit_correlation": float(off_diagonal.max()) if off_diagonal.numel() else 0.0,
        "mean_train_held_table_tv": sum(tvs) / len(tvs),
        "unseen_held_row_fraction": float(unseen_tokens.double().mean()),
    }


def fit_arm(
    seed: int,
    family: str,
    total_bits: int,
    depth: int,
    train_x: Tensor,
    held_x: Tensor,
    teacher: Tensor,
    dense_normals: Tensor,
    pairs: Tensor,
) -> tuple[ArmResult, Tensor]:
    started = time.perf_counter()
    if family == "dense_real":
        train_codes, _ = dense_codes(train_x, dense_normals[:total_bits], depth=depth)
        held_codes, held_branches = dense_codes(held_x, dense_normals[:total_bits], depth=depth)
    elif family == "pair_root":
        train_codes, _ = pair_codes(train_x, pairs[:total_bits], depth=depth)
        held_codes, held_branches = pair_codes(held_x, pairs[:total_bits], depth=depth)
    else:
        raise ValueError(f"unsupported family {family!r}")
    rows = 1 << depth
    payload, decoder = fit_additive_lut(train_codes, train_x, rows=rows)
    train_prediction = additive_lookup(train_codes, payload)
    held_prediction = additive_lookup(held_codes, payload)
    _, _, train_r2, _ = regression_metrics(train_prediction, train_x)
    mse, nmse, held_r2, cosine = regression_metrics(held_prediction, held_x)
    teacher_target = held_x.to(torch.float64) @ teacher.to(torch.float64).T
    teacher_prediction = held_prediction @ teacher.to(torch.float64).T
    teacher_mse, teacher_nmse, teacher_r2, teacher_cosine = regression_metrics(teacher_prediction, teacher_target)
    health = route_health(train_codes, held_codes, held_branches, rows)
    result = ArmResult(
        seed=seed,
        family=family,
        total_bits=total_bits,
        tables=total_bits // depth,
        comparisons_per_table=depth,
        train_r2=train_r2,
        held_mse=mse,
        held_nmse=nmse,
        held_r2=held_r2,
        held_cosine=cosine,
        teacher_output_mse=teacher_mse,
        teacher_output_nmse=teacher_nmse,
        teacher_output_r2=teacher_r2,
        teacher_output_cosine=teacher_cosine,
        reconstruction_teacher_nmse_max_abs_difference=abs(nmse - teacher_nmse),
        mean_table_entropy_bits=health["mean_table_entropy_bits"],
        mean_observed_rows=health["mean_observed_rows"],
        maximum_row_mass=health["maximum_row_mass"],
        mean_bit_one_probability=health["mean_bit_one_probability"],
        maximum_bit_balance_error=health["maximum_bit_balance_error"],
        mean_absolute_bit_correlation=health["mean_absolute_bit_correlation"],
        maximum_absolute_bit_correlation=health["maximum_absolute_bit_correlation"],
        mean_train_held_table_tv=health["mean_train_held_table_tv"],
        unseen_held_row_fraction=health["unseen_held_row_fraction"],
        decoder_rank=int(decoder["rank"]),
        decoder_condition_number=float(decoder["condition_number"]),
        fitted_degrees_of_freedom=int(decoder["degrees_of_freedom"]),
        deployed_row_scalars=(total_bits // depth) * rows * train_x.shape[1],
        seconds=time.perf_counter() - started,
    )
    return result, payload


def summarize(results: list[ArmResult], bit_budgets: tuple[int, ...], primary_metric: str = "held_r2") -> dict[str, object]:
    if primary_metric not in {"held_r2", "teacher_output_r2"}:
        raise ValueError(f"unsupported primary metric {primary_metric!r}")
    seeds = sorted({row.seed for row in results})
    lookup = {(row.seed, row.family, row.total_bits): row for row in results}

    def metric(row: ArmResult) -> float:
        return float(getattr(row, primary_metric))

    arms: dict[str, object] = {}
    angle_effects: dict[str, object] = {}
    for family in ("pair_root", "dense_real"):
        for bits in bit_budgets:
            values = [metric(lookup[seed, family, bits]) for seed in seeds]
            arms[f"{family}_C{bits}"] = {"primary_r2_mean": sum(values) / len(values), "primary_r2_per_seed": values}
    for bits in bit_budgets:
        deltas = [metric(lookup[seed, "dense_real", bits]) - metric(lookup[seed, "pair_root", bits]) for seed in seeds]
        angle_effects[f"C{bits}"] = {"dense_minus_pair_mean": sum(deltas) / len(deltas), "per_seed": deltas}
    low, high = min(bit_budgets), max(bit_budgets)
    rate_effects = {}
    for family in ("pair_root", "dense_real"):
        gains = [metric(lookup[seed, family, high]) - metric(lookup[seed, family, low]) for seed in seeds]
        rate_effects[family] = {"Cmax_minus_Cmin_mean": sum(gains) / len(gains), "per_seed": gains}
    primary_delta = angle_effects[f"C{high}"]["dense_minus_pair_mean"]
    dense_high = [metric(lookup[seed, "dense_real", high]) for seed in seeds]
    pair_high = [metric(lookup[seed, "pair_root", high]) for seed in seeds]
    return {
        "primary_metric": primary_metric,
        "arms": arms,
        "angle_effects": angle_effects,
        "rate_effects": rate_effects,
        "frozen_decisions": {
            "dense_sign_ceiling_adequate": {
                "pass": min(dense_high) >= 0.50 and sum(dense_high) / len(dense_high) >= 0.55,
                "rule": "all_seed_dense_Cmax_R2>=0.50_and_mean>=0.55",
            },
            "angular_coverage_is_primary": {
                "pass": all(dense > pair for dense, pair in zip(dense_high, pair_high)) and float(primary_delta) >= 0.10,
                "rule": "dense_minus_pair_positive_all_seeds_and_mean>=0.10_at_Cmax",
            },
            "bit_budget_is_material": {
                "pass": all(float(rate_effects[family]["Cmax_minus_Cmin_mean"]) >= 0.20 for family in ("pair_root", "dense_real")),
                "rule": "both_family_mean_Cmax_minus_Cmin_R2>=0.20",
            },
        },
    }


def run(args: argparse.Namespace) -> tuple[dict[str, object], dict[str, Tensor]]:
    if args.depth != 4:
        raise ValueError("the frozen experiment uses C4 tables")
    if tuple(sorted(args.bit_budgets)) != args.bit_budgets or any(bits % args.depth for bits in args.bit_budgets):
        raise ValueError("bit budgets must be sorted and divisible by depth")
    maximum_bits = max(args.bit_budgets)
    all_results: list[ArmResult] = []
    state: dict[str, Tensor] = {}
    bank_audit: dict[str, object] = {}
    for seed in args.seeds:
        dense_normals, pairs, audit = sample_paired_normal_bank(args.dim, maximum_bits, seed)
        teacher = orthogonal_teacher_float64(args.dim, seed) if args.teacher_mode == "orthogonal" else dense_normals
        train_generator = torch.Generator(device="cpu").manual_seed(50_000 + seed)
        held_generator = torch.Generator(device="cpu").manual_seed(60_000 + seed)
        train_x = torch.randn(args.train_samples, args.dim, generator=train_generator, dtype=torch.float64)
        held_x = torch.randn(args.held_samples, args.dim, generator=held_generator, dtype=torch.float64)
        bank_audit[str(seed)] = audit
        state[f"seed{seed}.dense_normals"] = dense_normals
        state[f"seed{seed}.pair_indices"] = pairs
        state[f"seed{seed}.teacher"] = teacher
        for bits in args.bit_budgets:
            for family in ("pair_root", "dense_real"):
                result, payload = fit_arm(
                    seed,
                    family,
                    bits,
                    args.depth,
                    train_x,
                    held_x,
                    teacher,
                    dense_normals,
                    pairs,
                )
                if args.teacher_mode == "orthogonal" and result.reconstruction_teacher_nmse_max_abs_difference > 1e-10:
                    raise AssertionError("orthogonal teacher changed reconstruction NMSE")
                all_results.append(result)
                state[f"seed{seed}.{family}.C{bits}.payload"] = payload
                primary = result.held_r2 if args.teacher_mode == "orthogonal" else result.teacher_output_r2
                print(f"seed={seed} family={family} bits={bits} primary_R2={primary:.6f}", flush=True)
    primary_metric = "held_r2" if args.teacher_mode == "orthogonal" else "teacher_output_r2"
    result = {
        "schema": "normal-coverage-bit-budget-recognition-probe-v2",
        "protocol": {
            "dim": args.dim,
            "input": "independent_standard_Gaussian_train_and_held",
            "teacher_mode": args.teacher_mode,
            "teacher": (
                "QR_Haar_orthogonal_diagnostic; reconstruction_is_the_primary_target"
                if args.teacher_mode == "orthogonal"
                else "the_same_128_dense_normal_bank_as_a_W1_like_linear_score_teacher"
            ),
            "primary_metric": primary_metric,
            "normal_bank": "nested_random_dense_unit_normals_paired_with_unique_nearest_pair_roots",
            "families": ["pair_root", "dense_real"],
            "bit_budgets": list(args.bit_budgets),
            "comparisons_per_table": args.depth,
            "rows_per_table": 1 << args.depth,
            "decoder": "global_empirical_least_squares_optimum_in_additive_C4_LUT_span_CPU_float64",
            "optimizer_used": False,
            "thresholds": "all_zero",
            "train_samples": args.train_samples,
            "held_samples": args.held_samples,
            "seeds": list(args.seeds),
            "dense_real_is_deployable_multiplier_free": False,
        },
        "normal_bank_audit": bank_audit,
        "rows": [asdict(row) for row in all_results],
        "summary": summarize(all_results, args.bit_budgets, primary_metric),
        "semantic_ledger": {
            "pair_root": "C pair comparisons + T C4 row reads/adds",
            "dense_real": "C dense D-dimensional dot/comparisons + T identical C4 row reads/adds; diagnostic only",
            "decoder_payload_scalars": "T*16*D",
        },
    }
    return result, state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pair-root versus dense-real normal coverage at matched C4 additive-LUT bit budgets")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--bit-budgets", type=parse_int_tuple, default=(8, 32, 128))
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--train-samples", type=int, default=32768)
    parser.add_argument("--held-samples", type=int, default=32768)
    parser.add_argument("--seeds", type=parse_seed_tuple, default=(0, 1, 2))
    parser.add_argument("--teacher-mode", choices=("orthogonal", "aligned_dense_scores"), default="orthogonal")
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.dim < 2 or args.train_samples < 1 or args.held_samples < 1:
        parser.error("dim>=2 and positive sample counts are required")
    artifact, output = Path(args.artifact), Path(args.output)
    if artifact.exists() or output.exists():
        parser.error("artifact and output paths must not exist")
    if artifact.resolve(strict=False) == output.resolve(strict=False):
        parser.error("artifact and output paths must differ")
    return args


def main() -> None:
    args = parse_args()
    result, state = run(args)
    artifact = Path(args.artifact)
    output = Path(args.output)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with artifact.open("xb") as handle:
        torch.save({"schema": result["schema"], "protocol": result["protocol"], "state": state}, handle)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(result["summary"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
