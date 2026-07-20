from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from tropnn.layers.s4_relation import S4_ORDER, s4_fourier_energy, s4_gauge_maps
from tropnn.tools.bilinear_retrieval_probe import make_problem, retrieval_metrics, teacher_scores
from tropnn.tools.coxeter_relation_probe import (
    LocalS4Router,
    categorical_prediction,
    make_uniform_pairs,
    pair_route_codes,
    r2_score,
    ridge_cg,
)


@dataclass(frozen=True)
class SharedFit:
    coefficient: Tensor
    bias: Tensor
    gauge_ids: Tensor
    template_ids: Tensor
    validation_mse: float
    rounds: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gauge-aligned shared S4 relation diagnosis.")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--input-dim", type=int, default=32)
    run.add_argument("--train-queries", type=int, default=2048)
    run.add_argument("--train-keys", type=int, default=2048)
    run.add_argument("--test-queries", type=int, default=256)
    run.add_argument("--test-keys", type=int, default=512)
    run.add_argument("--max-value", type=int, default=15)
    run.add_argument("--teacher", choices=("random_bilinear", "permutation_invariant"), default="random_bilinear")
    run.add_argument("--tables", type=int, default=16)
    run.add_argument("--templates", type=int, nargs="+", default=(1, 2, 4))
    run.add_argument("--fit-samples", type=int, default=65536)
    run.add_argument("--validation-samples", type=int, default=16384)
    run.add_argument("--gauge-samples", type=int, default=4096)
    run.add_argument("--gauge-rounds", type=int, default=3)
    run.add_argument("--gauge-restarts", type=int, default=2)
    run.add_argument("--ridge", type=float, default=0.001)
    run.add_argument("--cg-iterations", type=int, default=128)
    run.add_argument("--cg-tolerance", type=float, default=1e-7)
    run.add_argument("--batch-size", type=int, default=8192)
    run.add_argument("--eval-query-batch", type=int, default=32)
    run.add_argument("--top-k", type=int, default=16)
    run.add_argument("--device", default="cpu")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--out-dir", type=Path, required=True)
    summarize = commands.add_parser("summarize")
    summarize.add_argument("--result-dir", type=Path, required=True)
    summarize.add_argument("--out-report", type=Path, required=True)
    summarize.add_argument("--dense-topk-reference", type=float, default=0.6597)
    return parser


def shared_design(codes: Tensor, template_ids: Tensor, templates: int) -> Tensor:
    samples, tables = codes.shape
    feature = template_ids.view(1, tables) * S4_ORDER + codes
    design = torch.zeros(samples, templates * S4_ORDER + 1, device=codes.device, dtype=torch.float32)
    design[:, :-1].scatter_add_(1, feature, torch.full_like(feature, 1.0 / math.sqrt(tables), dtype=torch.float32))
    design[:, -1] = 1.0
    return design


def fit_shared(codes: Tensor, target: Tensor, template_ids: Tensor, templates: int, ridge: float) -> tuple[Tensor, Tensor]:
    design = shared_design(codes, template_ids, templates)
    gram = design.transpose(0, 1) @ design / design.shape[0]
    rhs = design.transpose(0, 1) @ target.float() / design.shape[0]
    diagonal = torch.arange(templates * S4_ORDER, device=design.device)
    gram[diagonal, diagonal] += ridge
    gram[-1, -1] += 1e-8
    solution = torch.linalg.solve(gram, rhs)
    return solution[:-1].view(templates, S4_ORDER), solution[-1]


def shared_prediction(codes: Tensor, coefficient: Tensor, bias: Tensor, template_ids: Tensor) -> Tensor:
    tables = codes.shape[-1]
    return coefficient[template_ids, codes].sum(dim=-1) / math.sqrt(tables) + bias


def gauge_codes(query: Tensor, key: Tensor, gauge_ids: Tensor, candidate_maps: Tensor) -> Tensor:
    table = torch.arange(query.shape[-1], device=query.device)
    maps = candidate_maps[gauge_ids]
    return maps[table, query.long(), key.long()]


def fit_per_table(
    codes: Tensor,
    target: Tensor,
    rows: int,
    args: argparse.Namespace,
) -> tuple[Tensor, Tensor]:
    coefficient, bias, _, _ = ridge_cg(
        codes,
        target,
        rows,
        args.ridge,
        args.cg_iterations,
        args.cg_tolerance,
        args.batch_size,
    )
    return coefficient.view(codes.shape[1], rows), bias


def coordinate_search(
    fit_query: Tensor,
    fit_key: Tensor,
    fit_target: Tensor,
    validation_query: Tensor,
    validation_key: Tensor,
    validation_target: Tensor,
    templates: int,
    args: argparse.Namespace,
) -> SharedFit:
    maps = s4_gauge_maps().to(fit_query.device)
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 7919 + templates)
    screen_count = min(args.gauge_samples, fit_query.shape[0])
    screen_index = torch.randperm(fit_query.shape[0], generator=generator)[:screen_count].to(fit_query.device)
    screen_query = fit_query[screen_index]
    screen_key = fit_key[screen_index]
    screen_target = fit_target[screen_index]
    best: SharedFit | None = None

    for restart in range(args.gauge_restarts):
        if restart == 0:
            gauge_ids = torch.zeros(fit_query.shape[1], dtype=torch.long, device=fit_query.device)
        else:
            gauge_ids = torch.randint(0, S4_ORDER * S4_ORDER, (fit_query.shape[1],), generator=generator).to(fit_query.device)
        template_ids = torch.arange(fit_query.shape[1], device=fit_query.device).remainder(templates)
        if restart:
            template_ids = template_ids[torch.randperm(template_ids.numel(), generator=generator).to(template_ids.device)]

        for _ in range(args.gauge_rounds):
            codes = gauge_codes(fit_query, fit_key, gauge_ids, maps)
            coefficient, bias = fit_shared(codes, fit_target, template_ids, templates, args.ridge)
            screen_codes = gauge_codes(screen_query, screen_key, gauge_ids, maps)
            prediction = shared_prediction(screen_codes, coefficient, bias, template_ids)
            scale = 1.0 / math.sqrt(fit_query.shape[1])
            for table in range(fit_query.shape[1]):
                current = coefficient[template_ids[table], screen_codes[:, table]] * scale
                base = prediction - current
                candidate_codes = maps[:, screen_query[:, table], screen_key[:, table]].transpose(0, 1)
                best_loss = torch.tensor(float("inf"), device=fit_query.device)
                best_gauge = gauge_ids[table]
                best_template = template_ids[table]
                best_value = current
                for template in range(templates):
                    for start in range(0, candidate_codes.shape[1], 48):
                        chunk = candidate_codes[:, start : start + 48]
                        value = coefficient[template, chunk] * scale
                        loss = (base[:, None] + value - screen_target[:, None]).square().mean(dim=0)
                        local_loss, local_index = loss.min(dim=0)
                        if local_loss < best_loss:
                            best_loss = local_loss
                            best_gauge = start + local_index
                            best_template = torch.tensor(template, device=fit_query.device)
                            best_value = value[:, local_index]
                gauge_ids[table] = best_gauge
                template_ids[table] = best_template
                screen_codes[:, table] = candidate_codes[:, best_gauge]
                prediction = base + best_value

        codes = gauge_codes(fit_query, fit_key, gauge_ids, maps)
        coefficient, bias = fit_shared(codes, fit_target, template_ids, templates, args.ridge)
        validation_codes = gauge_codes(validation_query, validation_key, gauge_ids, maps)
        validation_prediction = shared_prediction(validation_codes, coefficient, bias, template_ids)
        validation_mse = float((validation_prediction - validation_target).square().mean().item())
        candidate = SharedFit(
            coefficient.detach().clone(),
            bias.detach().clone(),
            gauge_ids.detach().clone(),
            template_ids.detach().clone(),
            validation_mse,
            args.gauge_rounds,
        )
        if best is None or candidate.validation_mse < best.validation_mse:
            best = candidate
    assert best is not None
    return best


def predict_matrix(
    query_route: Tensor,
    key_route: Tensor,
    coefficient: Tensor,
    bias: Tensor,
    template_ids: Tensor,
    gauge_ids: Tensor | None,
    query_batch: int,
) -> Tensor:
    maps = s4_gauge_maps().to(query_route.device) if gauge_ids is not None else None
    result: list[Tensor] = []
    for start in range(0, query_route.shape[0], query_batch):
        query = query_route[start : start + query_batch, None, :]
        key = key_route[None, :, :].expand(query.shape[0], -1, -1)
        query = query.expand_as(key)
        if gauge_ids is None:
            raise ValueError("gauge_ids are required for shared prediction")
        codes = gauge_codes(query, key, gauge_ids, maps)
        result.append(shared_prediction(codes, coefficient, bias, template_ids))
    return torch.cat(result, dim=0)


def baseline_matrix(
    router: LocalS4Router,
    query_route: Tensor,
    key_route: Tensor,
    decoder: str,
    coefficient: Tensor,
    bias: Tensor,
    query_batch: int,
) -> Tensor:
    result: list[Tensor] = []
    for start in range(0, query_route.shape[0], query_batch):
        query = query_route[start : start + query_batch, None, :].expand(-1, key_route.shape[0], -1)
        key = key_route[None, :, :].expand(query.shape[0], -1, -1)
        codes = router.pair_codes(query, key, decoder)
        flat = codes.reshape(-1, codes.shape[-1])
        prediction = categorical_prediction(flat, coefficient.reshape(-1), bias, router.tables)
        result.append(prediction.view(codes.shape[:-1]))
    return torch.cat(result, dim=0)


def metric_row(
    name: str,
    target: Tensor,
    prediction: Tensor,
    validation_target: Tensor,
    validation_prediction: Tensor,
    parameters: int,
    args: argparse.Namespace,
    **extra: object,
) -> dict[str, object]:
    retrieval = retrieval_metrics(target, prediction, args.top_k, args.seed + 601)
    return {
        "variant": name,
        "parameters": parameters,
        "validation_r2": r2_score(validation_target, validation_prediction),
        **retrieval,
        **extra,
    }


def run(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    problem = make_problem(args)
    train_queries = problem.train_queries.to(device)
    train_keys = problem.train_keys.to(device)
    relation = problem.relation.to(device)
    if args.teacher == "permutation_invariant":
        relation = torch.eye(args.input_dim, device=device) / math.sqrt(args.input_dim)
    router = LocalS4Router(args.input_dim, args.tables, args.seed).to(device)
    train_query_route = router.route(train_queries)
    train_key_route = router.route(train_keys)
    fit = make_uniform_pairs(train_queries, train_keys, relation, args.fit_samples, args.seed + 2003)
    validation = make_uniform_pairs(train_queries, train_keys, relation, args.validation_samples, args.seed + 2017)
    fit_query = train_query_route[fit.query_index]
    fit_key = train_key_route[fit.key_index]
    validation_query = train_query_route[validation.query_index]
    validation_key = train_key_route[validation.key_index]
    test_query_route = router.route(problem.test_queries.to(device))
    test_key_route = router.route(problem.test_keys.to(device))
    test_target = teacher_scores(problem.test_queries.to(device), problem.test_keys.to(device), relation)
    rows: list[dict[str, object]] = []

    for name, decoder, width in (("free_absolute", "absolute", 576), ("relative_per_table", "relative", 24)):
        fit_codes = pair_route_codes(router, train_query_route, train_key_route, fit, decoder)
        validation_codes = pair_route_codes(router, train_query_route, train_key_route, validation, decoder)
        coefficient, bias = fit_per_table(fit_codes, fit.target, width, args)
        validation_prediction = categorical_prediction(validation_codes, coefficient.reshape(-1), bias, args.tables)
        test_prediction = baseline_matrix(
            router, test_query_route, test_key_route, decoder, coefficient, bias, args.eval_query_batch
        )
        rows.append(
            metric_row(
                name,
                test_target,
                test_prediction,
                validation.target,
                validation_prediction,
                coefficient.numel() + 1,
                args,
                fourier=s4_fourier_energy(coefficient) if decoder == "relative" else None,
            )
        )

    relative_fit_codes = pair_route_codes(router, train_query_route, train_key_route, fit, "relative")
    relative_validation_codes = pair_route_codes(router, train_query_route, train_key_route, validation, "relative")
    identity_template = torch.zeros(args.tables, dtype=torch.long, device=device)
    identity_coefficient, identity_bias = fit_shared(relative_fit_codes, fit.target, identity_template, 1, args.ridge)
    identity_validation = shared_prediction(relative_validation_codes, identity_coefficient, identity_bias, identity_template)
    identity_test = predict_matrix(
        test_query_route,
        test_key_route,
        identity_coefficient,
        identity_bias,
        identity_template,
        torch.zeros(args.tables, dtype=torch.long, device=device),
        args.eval_query_batch,
    )
    rows.append(
        metric_row(
            "shared_identity_k1",
            test_target,
            identity_test,
            validation.target,
            identity_validation,
            identity_coefficient.numel() + 1,
            args,
            fourier=s4_fourier_energy(identity_coefficient),
        )
    )

    generator = torch.Generator(device="cpu").manual_seed(args.seed + 3571)
    random_gauge = torch.randint(0, S4_ORDER * S4_ORDER, (args.tables,), generator=generator).to(device)
    maps = s4_gauge_maps().to(device)
    random_fit_codes = gauge_codes(fit_query, fit_key, random_gauge, maps)
    random_validation_codes = gauge_codes(validation_query, validation_key, random_gauge, maps)
    random_coefficient, random_bias = fit_shared(random_fit_codes, fit.target, identity_template, 1, args.ridge)
    rows.append(
        metric_row(
            "shared_random_gauge_k1",
            test_target,
            predict_matrix(
                test_query_route,
                test_key_route,
                random_coefficient,
                random_bias,
                identity_template,
                random_gauge,
                args.eval_query_batch,
            ),
            validation.target,
            shared_prediction(random_validation_codes, random_coefficient, random_bias, identity_template),
            random_coefficient.numel() + 1,
            args,
            fourier=s4_fourier_energy(random_coefficient),
        )
    )

    for templates in args.templates:
        fit_result = coordinate_search(
            fit_query,
            fit_key,
            fit.target,
            validation_query,
            validation_key,
            validation.target,
            templates,
            args,
        )
        validation_codes = gauge_codes(validation_query, validation_key, fit_result.gauge_ids, maps)
        validation_prediction = shared_prediction(
            validation_codes, fit_result.coefficient, fit_result.bias, fit_result.template_ids
        )
        test_prediction = predict_matrix(
            test_query_route,
            test_key_route,
            fit_result.coefficient,
            fit_result.bias,
            fit_result.template_ids,
            fit_result.gauge_ids,
            args.eval_query_batch,
        )
        rows.append(
            metric_row(
                f"gauge_shared_k{templates}",
                test_target,
                test_prediction,
                validation.target,
                validation_prediction,
                fit_result.coefficient.numel() + 1,
                args,
                gauge_nonidentity=float((fit_result.gauge_ids != 0).float().mean().item()),
                template_occupancy=int(torch.unique(fit_result.template_ids).numel()),
                fourier=s4_fourier_energy(fit_result.coefficient),
            )
        )

    output = {
        "seed": args.seed,
        "teacher": args.teacher,
        "input_dim": args.input_dim,
        "tables": args.tables,
        "fit_samples": args.fit_samples,
        "validation_samples": args.validation_samples,
        "test_keys": args.test_keys,
        "top_k": args.top_k,
        "elapsed_seconds": time.perf_counter() - started,
        "variants": rows,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"seed{args.seed}.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, sort_keys=True), flush=True)


def summarize(args: argparse.Namespace) -> None:
    runs = [json.loads(path.read_text()) for path in sorted(args.result_dir.glob("seed*.json"))]
    if not runs:
        raise RuntimeError(f"no seed results in {args.result_dir}")
    flat = [{"seed": run["seed"], "teacher": run["teacher"], **row} for run in runs for row in run["variants"]]
    fields = sorted({key for row in flat for key in row if key != "fourier"})
    with (args.result_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: value for key, value in row.items() if key != "fourier"} for row in flat])
    variants = sorted({row["variant"] for row in flat})

    def values(variant: str, metric: str) -> list[float]:
        return [float(row[metric]) for row in flat if row["variant"] == variant]

    means = {variant: statistics.mean(values(variant, "topk_recall")) for variant in variants}
    identity = means["shared_identity_k1"]
    relative = means["relative_per_table"]
    gauge_variants = [variant for variant in variants if variant.startswith("gauge_shared_k")]
    best_gauge = max(gauge_variants, key=means.__getitem__)
    denominator = relative - identity
    recovery = (means[best_gauge] - identity) / denominator if denominator > 1e-12 else float("-inf")
    identity_sem = statistics.stdev(values("shared_identity_k1", "topk_recall")) / math.sqrt(len(runs)) if len(runs) > 1 else 0.0
    gauge_sem = statistics.stdev(values(best_gauge, "topk_recall")) / math.sqrt(len(runs)) if len(runs) > 1 else 0.0
    significant = means[best_gauge] - identity > 2.0 * math.sqrt(identity_sem**2 + gauge_sem**2)
    pass_gate = recovery >= 0.8 and significant
    random_topk = float(runs[0].get("top_k", 16)) / float(runs[0].get("test_keys", 512))
    free_topk = means["free_absolute"]
    capacity_denominator = args.dense_topk_reference - random_topk
    capacity_recovery = (
        (free_topk - random_topk) / capacity_denominator if capacity_denominator > 1e-12 else float("-inf")
    )
    capacity_pass = capacity_recovery >= 0.8
    advance = capacity_pass and pass_gate
    lines = [
        "# Gauge-Aligned S4 Relation Sharing",
        "",
        "## Mean held retrieval",
        "",
        "| Variant | Top-16 | Top-1 | Hard-negative | Spearman | Valid R2 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in variants:
        lines.append(
            f"| {variant} | {statistics.mean(values(variant, 'topk_recall')):.4f} | "
            f"{statistics.mean(values(variant, 'top1_accuracy')):.4f} | "
            f"{statistics.mean(values(variant, 'hard_negative_preference_accuracy')):.4f} | "
            f"{statistics.mean(values(variant, 'spearman')):.4f} | "
            f"{statistics.mean(values(variant, 'validation_r2')):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Pre-registered decision",
            "",
            f"- Best gauge model: `{best_gauge}`.",
            f"- Recovery of the identity-to-per-table-relative Top-16 gap: `{recovery:.4f}`.",
            f"- Improvement exceeds two pooled standard errors: `{significant}`.",
            f"- Free-absolute recovery relative to dense-route reference: `{capacity_recovery:.4f}`.",
            f"- Raw route-capacity gate: `{capacity_pass}`.",
            f"- Advance to online gauge training: `{advance}`.",
        ]
    )
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text("\n".join(lines) + "\n")
    (args.result_dir / "decision.json").write_text(
        json.dumps(
            {
                "best_gauge_variant": best_gauge,
                "gap_recovery": recovery,
                "significant": significant,
                "sharing_gate": pass_gate,
                "capacity_recovery": capacity_recovery,
                "capacity_gate": capacity_pass,
                "advance_online_training": advance,
            },
            indent=2,
        )
        + "\n"
    )
    print(args.result_dir / "summary.csv")
    print(args.out_report)


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run":
        run(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
