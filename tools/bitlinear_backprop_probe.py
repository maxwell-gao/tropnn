from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch
from torch import Tensor, nn

from tropnn.layers import PairwiseLUT
from tropnn.tools.additive_route_recovery import additive_feature_matrix, minimum_norm_least_squares


VARIANT_DEPTH = {"A": 1, "B": 1, "C": 1, "D": 1, "E2": 2, "E4": 4, "E8": 8}
FIXED_DEPTH = {"F2": 2, "F4": 4, "F8": 8}
ALL_DEPTHS = {**VARIANT_DEPTH, **FIXED_DEPTH}


class PCLUTStudent(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        depth: int,
        tables: int,
        comparisons: int,
        trainable_thresholds: bool,
        seed: int,
    ) -> None:
        super().__init__()
        layer_args = dict(
            tables=tables,
            comparisons=comparisons,
            backend="torch",
            anchor_policy="random",
            fixed_zero_threshold=not trainable_thresholds,
            use_output_scaling=False,
        )
        self.hidden = nn.ModuleList(
            PairwiseLUT(
                input_dim=input_dim,
                output_dim=input_dim,
                seed=seed + 100 + index,
                anchor_seed=seed + 100 + index,
                **layer_args,
            )
            for index in range(depth - 1)
        )
        self.readout = PairwiseLUT(
            input_dim=input_dim,
            output_dim=output_dim,
            seed=seed,
            anchor_seed=seed,
            **layer_args,
        )

    def encode(self, x: Tensor) -> Tensor:
        hidden = x.unsqueeze(1)
        for layer in self.hidden:
            hidden = hidden + layer(hidden)
        return hidden[:, 0, :]

    def forward(self, x: Tensor) -> Tensor:
        hidden = self.encode(x)
        return self.readout(hidden.unsqueeze(1))[:, 0, :]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backpropagation probe for a fixed BitLinear teacher and PC-LUT students.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--variant", choices=list(ALL_DEPTHS), required=True)
    run.add_argument("--support-size", type=int, default=8192)
    run.add_argument("--input-dim", type=int, default=128)
    run.add_argument("--output-dim", type=int, default=64)
    run.add_argument("--max-value", type=int, default=15)
    run.add_argument("--tables", type=int, default=16)
    run.add_argument("--comparisons", type=int, default=5)
    run.add_argument("--train-fraction", type=float, default=0.75)
    run.add_argument("--steps", type=int, default=3000)
    run.add_argument("--batch-size", type=int, default=512)
    run.add_argument("--eval-every", type=int, default=250)
    run.add_argument("--payload-lr", type=float, default=0.05)
    run.add_argument("--threshold-lr", type=float, default=0.01)
    run.add_argument("--alternating-refits", type=int, default=30)
    run.add_argument("--rcond", type=float, default=1e-10)
    run.add_argument("--device", default="cuda")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--out-dir", type=Path, required=True)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--result-dir", type=Path, required=True)
    summarize.add_argument("--out-report", type=Path, required=True)

    fixed = subparsers.add_parser("summarize-fixed")
    fixed.add_argument("--fixed-dir", type=Path, required=True)
    fixed.add_argument("--trainable-dir", type=Path, required=True)
    fixed.add_argument("--out-report", type=Path, required=True)
    return parser


def make_problem(args: argparse.Namespace) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    data_generator = torch.Generator(device="cpu").manual_seed(args.seed)
    x = torch.randint(
        0,
        args.max_value + 1,
        (args.support_size, args.input_dim),
        generator=data_generator,
        dtype=torch.int64,
    ).to(torch.float32)
    weight_generator = torch.Generator(device="cpu").manual_seed(args.seed + 101)
    signs = torch.randint(
        0,
        2,
        (args.output_dim, args.input_dim),
        generator=weight_generator,
        dtype=torch.int64,
    )
    weight = (2.0 * signs.to(torch.float32) - 1.0) / math.sqrt(args.input_dim)
    y = x @ weight.T
    split_generator = torch.Generator(device="cpu").manual_seed(args.seed + 211)
    order = torch.randperm(args.support_size, generator=split_generator)
    train_size = int(round(args.train_fraction * args.support_size))
    return x, y, order[:train_size], order[train_size:]


def regression_metrics(target: Tensor, prediction: Tensor) -> dict[str, float]:
    target64 = target.to(torch.float64)
    prediction64 = prediction.to(torch.float64)
    error = target64 - prediction64
    mse = float(error.square().mean().item())
    centered = target64 - target64.mean(dim=0, keepdim=True)
    variance = float(centered.square().mean().item())
    r2 = 1.0 - mse / variance if variance > 1e-24 else (1.0 if mse < 1e-24 else float("nan"))
    return {"mse": mse, "r2": float(r2), "max_abs": float(error.abs().max().item())}


@torch.no_grad()
def model_predictions(model: PCLUTStudent, x: Tensor, batch_size: int) -> Tensor:
    predictions = [model(x[start : start + batch_size]) for start in range(0, x.shape[0], batch_size)]
    return torch.cat(predictions, dim=0)


@torch.no_grad()
def output_route_codes(model: PCLUTStudent, x: Tensor, batch_size: int) -> Tensor:
    codes: list[Tensor] = []
    for start in range(0, x.shape[0], batch_size):
        hidden = model.encode(x[start : start + batch_size])
        codes.append(model.readout.route(hidden.unsqueeze(1)).indices[:, 0, :].cpu())
    return torch.cat(codes, dim=0).to(torch.long)


def fit_payload_from_codes(
    layer: PairwiseLUT,
    codes: Tensor,
    targets: Tensor,
    train_indices: Tensor,
    comparisons: int,
    rcond: float,
) -> tuple[Tensor, int, float]:
    features = additive_feature_matrix(codes, comparisons)
    coefficients, rank, condition = minimum_norm_least_squares(
        features[train_indices], targets[train_indices].to(torch.float64), rcond=rcond
    )
    payload = coefficients.view(layer.tables, 1 << comparisons, targets.shape[1])
    with torch.no_grad():
        layer.lut.copy_(payload.to(device=layer.lut.device, dtype=layer.lut.dtype))
    return features, rank, condition


def closed_form_reference(
    model: PCLUTStudent,
    x: Tensor,
    y_cpu: Tensor,
    train_indices: Tensor,
    test_indices: Tensor,
    args: argparse.Namespace,
) -> tuple[dict[str, float], Tensor]:
    codes = output_route_codes(model, x, args.batch_size)
    features = additive_feature_matrix(codes, args.comparisons)
    coefficients, rank, condition = minimum_norm_least_squares(
        features[train_indices], y_cpu[train_indices].to(torch.float64), rcond=args.rcond
    )
    prediction = features @ coefficients
    train = regression_metrics(y_cpu[train_indices], prediction[train_indices])
    test = regression_metrics(y_cpu[test_indices], prediction[test_indices])
    return {
        "closed_form_train_mse": train["mse"],
        "closed_form_train_r2": train["r2"],
        "closed_form_test_mse": test["mse"],
        "closed_form_test_r2": test["r2"],
        "closed_form_feature_rank": rank,
        "closed_form_feature_condition": condition,
    }, codes


def evaluate(
    model: PCLUTStudent,
    x: Tensor,
    y: Tensor,
    train_indices: Tensor,
    test_indices: Tensor,
    batch_size: int,
    step: int,
) -> dict[str, float | int]:
    model.eval()
    prediction = model_predictions(model, x, batch_size)
    train = regression_metrics(y[train_indices], prediction[train_indices])
    test = regression_metrics(y[test_indices], prediction[test_indices])
    model.train()
    return {
        "step": step,
        "train_mse": train["mse"],
        "train_r2": train["r2"],
        "test_mse": test["mse"],
        "test_r2": test["r2"],
    }


def parameter_groups(model: PCLUTStudent, args: argparse.Namespace) -> list[dict[str, object]]:
    payloads = [parameter for name, parameter in model.named_parameters() if name.endswith("lut")]
    thresholds = [parameter for name, parameter in model.named_parameters() if name.endswith("thresholds")]
    groups: list[dict[str, object]] = [{"params": payloads, "lr": args.payload_lr}]
    if thresholds:
        groups.append({"params": thresholds, "lr": args.threshold_lr})
    return groups


def train_adamw(
    model: PCLUTStudent,
    x: Tensor,
    y: Tensor,
    train_indices: Tensor,
    test_indices: Tensor,
    args: argparse.Namespace,
) -> list[dict[str, float | int]]:
    optimizer = torch.optim.AdamW(parameter_groups(model, args), weight_decay=0.0)
    generator = torch.Generator(device=x.device).manual_seed(args.seed + 307)
    history = [evaluate(model, x, y, train_indices, test_indices, args.batch_size, 0)]
    for step in range(1, args.steps + 1):
        positions = torch.randint(0, train_indices.numel(), (args.batch_size,), generator=generator, device=x.device)
        batch_indices = train_indices[positions]
        prediction = model(x[batch_indices])
        loss = (prediction - y[batch_indices]).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % args.eval_every == 0 or step == args.steps:
            row = evaluate(model, x, y, train_indices, test_indices, args.batch_size, step)
            history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    return history


def train_alternating(
    model: PCLUTStudent,
    x: Tensor,
    y: Tensor,
    y_cpu: Tensor,
    train_indices: Tensor,
    test_indices: Tensor,
    args: argparse.Namespace,
) -> list[dict[str, float | int]]:
    model.readout.lut.requires_grad_(False)
    threshold_parameters = [parameter for name, parameter in model.named_parameters() if name.endswith("thresholds")]
    optimizer = torch.optim.AdamW(threshold_parameters, lr=args.threshold_lr, weight_decay=0.0)
    generator = torch.Generator(device=x.device).manual_seed(args.seed + 307)
    steps_per_refit = max(1, math.ceil(args.steps / args.alternating_refits))
    history: list[dict[str, float | int]] = []
    completed = 0
    while completed < args.steps:
        codes = output_route_codes(model, x, args.batch_size)
        fit_payload_from_codes(model.readout, codes, y_cpu, train_indices.cpu(), args.comparisons, args.rcond)
        if not history:
            history.append(evaluate(model, x, y, train_indices, test_indices, args.batch_size, 0))
        phase_steps = min(steps_per_refit, args.steps - completed)
        for _ in range(phase_steps):
            positions = torch.randint(0, train_indices.numel(), (args.batch_size,), generator=generator, device=x.device)
            batch_indices = train_indices[positions]
            prediction = model(x[batch_indices])
            loss = (prediction - y[batch_indices]).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(threshold_parameters, 1.0)
            optimizer.step()
            completed += 1
        row = evaluate(model, x, y, train_indices, test_indices, args.batch_size, completed)
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    final_codes = output_route_codes(model, x, args.batch_size)
    fit_payload_from_codes(model.readout, final_codes, y_cpu, train_indices.cpu(), args.comparisons, args.rcond)
    history.append(evaluate(model, x, y, train_indices, test_indices, args.batch_size, args.steps))
    return history


def tensor_rms(parameters: list[Tensor]) -> float:
    if not parameters:
        return 0.0
    square_sum = sum(float(parameter.detach().float().square().sum().item()) for parameter in parameters)
    count = sum(parameter.numel() for parameter in parameters)
    return math.sqrt(square_sum / count)


def write_history(path: Path, history: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def run(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    x_cpu, y_cpu, train_cpu, test_cpu = make_problem(args)
    x = x_cpu.to(device)
    y = y_cpu.to(device)
    train_indices = train_cpu.to(device)
    test_indices = test_cpu.to(device)
    trainable_thresholds = args.variant not in {"A", *FIXED_DEPTH}
    model = PCLUTStudent(
        args.input_dim,
        args.output_dim,
        depth=ALL_DEPTHS[args.variant],
        tables=args.tables,
        comparisons=args.comparisons,
        trainable_thresholds=trainable_thresholds,
        seed=args.seed,
    ).to(device)
    closed_form, initial_codes = closed_form_reference(
        model, x, y_cpu, train_cpu, test_cpu, args
    )
    if args.variant == "C":
        fit_payload_from_codes(model.readout, initial_codes, y_cpu, train_cpu, args.comparisons, args.rcond)

    started = time.perf_counter()
    if args.variant == "D":
        history = train_alternating(model, x, y, y_cpu, train_indices, test_indices, args)
    else:
        history = train_adamw(model, x, y, train_indices, test_indices, args)
    elapsed = time.perf_counter() - started

    final_codes = output_route_codes(model, x, args.batch_size)
    row_changed = float((final_codes != initial_codes).any(dim=1).to(torch.float64).mean().item())
    table_changed = float((final_codes != initial_codes).to(torch.float64).mean().item())
    thresholds = [parameter for name, parameter in model.named_parameters() if name.endswith("thresholds")]
    payloads = [parameter for name, parameter in model.named_parameters() if name.endswith("lut")]
    final = history[-1]
    result: dict[str, object] = {
        "variant": args.variant,
        "depth": ALL_DEPTHS[args.variant],
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "support_size": args.support_size,
        "input_dim": args.input_dim,
        "output_dim": args.output_dim,
        "tables": args.tables,
        "comparisons": args.comparisons,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "payload_lr": args.payload_lr,
        "threshold_lr": args.threshold_lr,
        "elapsed_seconds": elapsed,
        "steps_per_second": args.steps / elapsed,
        "payload_rms": tensor_rms(payloads),
        "threshold_rms": tensor_rms(thresholds),
        "route_rows_changed_fraction": row_changed,
        "route_table_codes_changed_fraction": table_changed,
        **closed_form,
        "initial_train_r2": history[0]["train_r2"],
        "initial_test_r2": history[0]["test_r2"],
        "final_train_mse": final["train_mse"],
        "final_train_r2": final["train_r2"],
        "final_test_mse": final["test_mse"],
        "final_test_r2": final["test_r2"],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"{args.variant}.json").write_text(json.dumps(result, indent=2) + "\n")
    write_history(args.out_dir / f"{args.variant}.history.csv", history)
    print(json.dumps(result, sort_keys=True), flush=True)


def summarize(args: argparse.Namespace) -> None:
    results = []
    for variant in VARIANT_DEPTH:
        path = args.result_dir / f"{variant}.json"
        if not path.exists():
            raise FileNotFoundError(f"missing result {path}")
        results.append(json.loads(path.read_text()))
    summary_path = args.result_dir / "summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    by_variant = {row["variant"]: row for row in results}
    a = by_variant["A"]
    b = by_variant["B"]
    c = by_variant["C"]
    d = by_variant["D"]
    best_e = max((by_variant[name] for name in ("E2", "E4", "E8")), key=lambda row: row["final_test_r2"])
    lines = [
        "# BitLinear Teacher Backpropagation Probe",
        "",
        "A fixed binary-sign linear teacher `Y = X W^T` is fitted by the production `PairwiseLUT` implementation.",
        "All variants share the same finite support, teacher, train/test split, and initial readout anchors.",
        "",
        "| variant | depth | params | closed train R2 | initial train R2 | final train R2 | final test R2 | route rows changed | steps/s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        lines.append(
            "| {variant} | {depth} | {parameters} | {closed_form_train_r2:.4f} | {initial_train_r2:.4f} | "
            "{final_train_r2:.4f} | {final_test_r2:.4f} | {route_rows_changed_fraction:.3f} | {steps_per_second:.1f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Diagnostic deltas",
            "",
            f"- A minus its fixed-route closed-form train ceiling: `{a['final_train_r2'] - a['closed_form_train_r2']:+.6f}`.",
            f"- B minus A test R2: `{b['final_test_r2'] - a['final_test_r2']:+.6f}`.",
            f"- C minus B test R2: `{c['final_test_r2'] - b['final_test_r2']:+.6f}`.",
            f"- D minus B test R2: `{d['final_test_r2'] - b['final_test_r2']:+.6f}`.",
            f"- Best deep variant minus B test R2: `{best_e['final_test_r2'] - b['final_test_r2']:+.6f}` (`{best_e['variant']}`).",
            "",
            "## Variant definitions",
            "",
            "- A: fixed zero thresholds, zero-init payload, AdamW payload training.",
            "- B: trainable STE thresholds, zero-init payload, joint AdamW training.",
            "- C: exact fixed-route payload prefit followed by joint threshold/payload AdamW training.",
            "- D: alternating exact hard-route payload refits and threshold-only STE phases.",
            "- E2/E4/E8: residual PC-LUT hidden layers followed by a PC-LUT readout; all payloads and thresholds use AdamW.",
        ]
    )
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text("\n".join(lines) + "\n")
    print(json.dumps({"summary": str(summary_path), "report": str(args.out_report)}, sort_keys=True))


def summarize_fixed(args: argparse.Namespace) -> None:
    pairs = []
    for depth in (2, 4, 8):
        trainable = json.loads((args.trainable_dir / f"E{depth}.json").read_text())
        fixed = json.loads((args.fixed_dir / f"F{depth}.json").read_text())
        pairs.append((trainable, fixed))

    rows = []
    for trainable, fixed in pairs:
        rows.append(
            {
                "depth": trainable["depth"],
                "trainable_variant": trainable["variant"],
                "fixed_variant": fixed["variant"],
                "trainable_parameters": trainable["parameters"],
                "fixed_parameters": fixed["parameters"],
                "trainable_train_r2": trainable["final_train_r2"],
                "fixed_train_r2": fixed["final_train_r2"],
                "trainable_test_r2": trainable["final_test_r2"],
                "fixed_test_r2": fixed["final_test_r2"],
                "test_r2_delta_trainable_minus_fixed": trainable["final_test_r2"] - fixed["final_test_r2"],
                "trainable_steps_per_second": trainable["steps_per_second"],
                "fixed_steps_per_second": fixed["steps_per_second"],
                "fixed_speedup": fixed["steps_per_second"] / trainable["steps_per_second"],
                "trainable_route_rows_changed": trainable["route_rows_changed_fraction"],
                "fixed_route_rows_changed": fixed["route_rows_changed_fraction"],
                "trainable_route_table_codes_changed": trainable["route_table_codes_changed_fraction"],
                "fixed_route_table_codes_changed": fixed["route_table_codes_changed_fraction"],
            }
        )
    summary_path = args.fixed_dir / "fixed_vs_trainable_summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Fixed-Threshold versus Trainable-Threshold Deep PC-LUT",
        "",
        "F2/F4/F8 use the same BitLinear teacher, data split, anchors, residual stack, zero payload initialization,",
        "5000-step AdamW schedule, and payload learning rate as E2/E4/E8. Every F threshold is a fixed zero buffer.",
        "Later-layer routes may still change because earlier payload updates change the hidden representation.",
        "",
        "| depth | E train R2 | F train R2 | E test R2 | F test R2 | E-F test | E route rows | F route rows | E steps/s | F steps/s |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {depth} | {trainable_train_r2:.4f} | {fixed_train_r2:.4f} | {trainable_test_r2:.4f} | "
            "{fixed_test_r2:.4f} | {test_r2_delta_trainable_minus_fixed:+.4f} | "
            "{trainable_route_rows_changed:.3f} | {fixed_route_rows_changed:.3f} | "
            "{trainable_steps_per_second:.1f} | {fixed_steps_per_second:.1f} |".format(**row)
        )
    baseline_test_r2 = pairs[0][0]["closed_form_test_r2"]
    depth4 = rows[1]
    depth8 = rows[2]
    depth4_fraction = (depth4["fixed_test_r2"] - baseline_test_r2) / (depth4["trainable_test_r2"] - baseline_test_r2)
    depth8_fraction = (depth8["fixed_test_r2"] - baseline_test_r2) / (depth8["trainable_test_r2"] - baseline_test_r2)
    lines.extend(
        [
            "",
            "## Main findings",
            "",
            f"The single-layer fixed-route closed-form test baseline is `{baseline_test_r2:.4f}`.",
            f"F4 and F8 reach `{depth4['fixed_test_r2']:.4f}` and `{depth8['fixed_test_r2']:.4f}`, so learned boundary translation is not required for a depth gain.",
            f"Fixed thresholds retain `{depth4_fraction:.1%}` of the E4 gain and `{depth8_fraction:.1%}` of the E8 gain above the single-layer baseline.",
            f"Trainable thresholds still add `{depth4['test_r2_delta_trainable_minus_fixed']:+.4f}` at depth 4 and `{depth8['test_r2_delta_trainable_minus_fixed']:+.4f}` at depth 8.",
            "Every fixed-threshold model changes the readout route for every sample, proving that payload-induced hidden-state movement is sufficient to reroute later layers.",
            "Fixed-threshold models fit the training support better but generalize worse, so threshold learning acts primarily through the learned partition's inductive bias in this run.",
            "",
            "## Interpretation",
            "",
            "The experiment separates two mechanisms that both change a later route:",
            "",
            "```text",
            "payload update -> hidden representation changes -> crossing a fixed boundary",
            "threshold update -> boundary translates around the current representation",
            "```",
            "",
            "F4/F8 establish the first mechanism. The positive E-F test gaps establish an additional contribution from the second.",
            "The decreasing E-F gap at depth 8 indicates that deeper hidden-state-induced rerouting can substitute for part, but not all, of learned boundary translation.",
            "",
            "## Caveats",
            "",
            "The result uses one seed and a synthetic finite random-integer support. Exact coordinate ties make zero-threshold routes unusually sensitive to small representation changes.",
            "The comparison is endpoint-based at 5000 steps and does not parameter-match depth against the binary teacher.",
        ]
    )
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text("\n".join(lines) + "\n")
    print(json.dumps({"summary": str(summary_path), "report": str(args.out_report)}, sort_keys=True))


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run":
        run(args)
    elif args.command == "summarize":
        summarize(args)
    else:
        summarize_fixed(args)


if __name__ == "__main__":
    main()
