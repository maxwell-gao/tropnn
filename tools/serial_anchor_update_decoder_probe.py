from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor, nn

from tropnn.layers import PairwiseLUT
from tropnn.layers.hard_lookup import HardLookupSpec, hard_route
from tropnn.layers.pairwise import PairwiseRoute, _make_pairwise_anchors
from tropnn.layers.surrogate import ste_heaviside
from tropnn.tools.bilinear_retrieval_probe import retrieval_metrics, teacher_scores
from tropnn.tools.fixed_route_relation_energy_probe import make_relation_pairs, pair_accuracy

DEPTHS = (2, 4, 8, 16)
UPDATE_VARIANTS = ("signed", "two_sided")
UpdateVariant = Literal["signed", "two_sided"]


class AnchorRouteUpdate(nn.Module):
    """Route-conditioned sparse update supported on the route's anchor coordinates."""

    def __init__(
        self,
        input_dim: int,
        *,
        tables: int,
        comparisons: int,
        variant: UpdateVariant,
        seed: int,
    ) -> None:
        super().__init__()
        if variant not in UPDATE_VARIANTS:
            raise ValueError(f"unsupported update variant {variant!r}")
        self.input_dim = input_dim
        self.tables = tables
        self.comparisons = comparisons
        self.variant = variant
        self._route_spec = HardLookupSpec(
            input_dim,
            input_dim,
            comparisons,
            "pair",
            "flat",
            bit_order="lsb",
            tie_break="positive",
            surrogate="none",
        )
        self.output_scale = 1.0 / math.sqrt(tables)
        self.register_buffer(
            "anchors",
            _make_pairwise_anchors(input_dim, tables, comparisons, policy="random", seed=seed),
        )
        self.register_buffer("thresholds", torch.zeros(tables, comparisons))
        self.register_buffer("powers", 2 ** torch.arange(comparisons, dtype=torch.long))
        sides = 1 if variant == "signed" else 2
        self.payload = nn.Parameter(torch.zeros(tables, 1 << comparisons, comparisons, sides))

    def route(self, x: Tensor) -> PairwiseRoute:
        route = hard_route(x, self.anchors, self.thresholds, self._route_spec)
        return PairwiseRoute(route.codes, route.margins)

    def _lookup(self, indices: Tensor) -> Tensor:
        table = torch.arange(self.tables, device=indices.device).view(1, 1, self.tables)
        return self.payload[table, indices]

    def _scatter(self, coefficients: Tensor) -> Tensor:
        batch, sequence = coefficients.shape[:2]
        output = coefficients.new_zeros(batch, sequence, self.input_dim)
        left = self.anchors[..., 0].reshape(1, 1, -1).expand(batch, sequence, -1)
        right = self.anchors[..., 1].reshape(1, 1, -1).expand(batch, sequence, -1)
        if self.variant == "signed":
            values = coefficients[..., 0].reshape(batch, sequence, -1)
            output.scatter_add_(-1, left, values)
            output.scatter_add_(-1, right, -values)
        else:
            output.scatter_add_(-1, left, coefficients[..., 0].reshape(batch, sequence, -1))
            output.scatter_add_(-1, right, coefficients[..., 1].reshape(batch, sequence, -1))
        return output * self.output_scale

    def hard_update(self, route: PairwiseRoute) -> Tensor:
        return self._scatter(self._lookup(route.indices))

    def _ste_update(self, route: PairwiseRoute) -> Tensor:
        bit = route.margins.abs().argmin(dim=-1)
        margin = route.margins.gather(dim=-1, index=bit.unsqueeze(-1)).squeeze(-1)
        neighbor = route.indices ^ (2**bit).long()
        delta = self._lookup(neighbor) - self._lookup(route.indices)
        ste = ste_heaviside(margin, "fast_sigmoid_odd") - (margin > 0).to(margin.dtype)
        return self._scatter(delta * ste[..., None, None])

    def forward(self, x: Tensor) -> Tensor:
        route = self.route(x)
        output = self.hard_update(route)
        if self.training and x.requires_grad:
            output = output + self._ste_update(route).to(output.dtype)
        return output


class SerialAnchorUpdateDecoder(nn.Module):
    """Serial sparse anchor updates followed by one scalar PC-LUT readout."""

    def __init__(
        self,
        input_dim: int,
        *,
        depth: int,
        tables: int,
        comparisons: int,
        variant: UpdateVariant,
        seed: int,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be positive, got {depth}")
        self.variant = variant
        self.blocks = nn.ModuleList(
            AnchorRouteUpdate(
                input_dim,
                tables=tables,
                comparisons=comparisons,
                variant=variant,
                seed=seed + 1009 * index,
            )
            for index in range(depth - 1)
        )
        self.readout = PairwiseLUT(
            input_dim=input_dim,
            output_dim=1,
            tables=tables,
            comparisons=comparisons,
            backend="torch",
            seed=seed + 1009 * (depth - 1),
            anchor_seed=seed + 1009 * (depth - 1),
            anchor_policy="random",
            lut_init_std=0.0,
            use_min_margin_ste=True,
            use_output_scaling=True,
            fixed_zero_threshold=True,
        )

    def hidden(self, pair: Tensor, *, route_source: Literal["serial", "initial"] = "serial") -> Tensor:
        initial = pair.unsqueeze(1)
        hidden = initial
        for block in self.blocks:
            source = hidden if route_source == "serial" else initial
            hidden = hidden + block(source)
        return hidden

    def forward(self, pair: Tensor, *, route_source: Literal["serial", "initial"] = "serial") -> Tensor:
        return self.readout(self.hidden(pair, route_source=route_source))[:, 0, :]

    def prefix_scores(self, pair: Tensor) -> Tensor:
        hidden = pair.unsqueeze(1)
        scores = [self.readout(hidden)[:, 0, :]]
        for block in self.blocks:
            hidden = hidden + block(hidden)
            scores.append(self.readout(hidden)[:, 0, :])
        return torch.cat(scores, dim=-1)

    @torch.no_grad()
    def payload_rms(self) -> list[float]:
        payloads = [block.payload for block in self.blocks]
        return [float(payload.square().mean().sqrt().item()) for payload in payloads] + [
            float(self.readout.lut.square().mean().sqrt().item())
        ]


class CounterfactualDecoder(nn.Module):
    def __init__(self, model: SerialAnchorUpdateDecoder) -> None:
        super().__init__()
        self.model = model

    def forward(self, pair: Tensor) -> Tensor:
        return self.model(pair, route_source="initial")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serial anchor-normal PC-LUT relation decoder sweep.")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run")
    run.add_argument("--variant", choices=UPDATE_VARIANTS, required=True)
    run.add_argument("--depth", choices=DEPTHS, type=int, required=True)
    run.add_argument("--input-dim", type=int, default=32)
    run.add_argument("--train-queries", type=int, default=2048)
    run.add_argument("--train-keys", type=int, default=2048)
    run.add_argument("--test-queries", type=int, default=256)
    run.add_argument("--test-keys", type=int, default=512)
    run.add_argument("--max-value", type=int, default=15)
    run.add_argument("--tables", type=int, default=16)
    run.add_argument("--comparisons", type=int, default=5)
    run.add_argument("--positive-per-query", type=int, default=16)
    run.add_argument("--hard-negative-per-query", type=int, default=8)
    run.add_argument("--steps", type=int, default=10000)
    run.add_argument("--batch-size", type=int, default=512)
    run.add_argument("--eval-batch-size", type=int, default=4096)
    run.add_argument("--eval-every", type=int, default=1000)
    run.add_argument("--diagnostic-pairs", type=int, default=8192)
    run.add_argument("--top-k", type=int, default=16)
    run.add_argument("--lr", type=float, default=0.001)
    run.add_argument("--weight-decay", type=float, default=0.0)
    run.add_argument("--device", default="cuda")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--out-dir", type=Path, required=True)

    summarize = commands.add_parser("summarize")
    summarize.add_argument("--result-dir", type=Path, required=True)
    summarize.add_argument("--full-vector-summary", type=Path, required=True)
    summarize.add_argument("--width-summary", type=Path, required=True)
    summarize.add_argument("--degree-summary", type=Path, required=True)
    summarize.add_argument("--out-report", type=Path, required=True)
    return parser


def train_loss(model: nn.Module, pairs: object, indices: Tensor) -> Tensor:
    positive = model(pairs.positive[indices])
    negative = model(pairs.negative[indices])
    return 0.5 * (
        (positive - pairs.positive_target[indices]).square().mean()
        + (negative - pairs.negative_target[indices]).square().mean()
    )


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


@torch.no_grad()
def predict_prefix_score_matrices(
    model: SerialAnchorUpdateDecoder,
    queries: Tensor,
    keys: Tensor,
    batch_size: int,
) -> list[Tensor]:
    query_count, key_count = queries.shape[0], keys.shape[0]
    depth = len(model.blocks) + 1
    flat = torch.empty(query_count * key_count, depth, device=queries.device)
    for start in range(0, flat.shape[0], batch_size):
        stop = min(start + batch_size, flat.shape[0])
        index = torch.arange(start, stop, device=queries.device)
        pair = torch.cat((queries[index // key_count], keys[index % key_count]), dim=-1)
        flat[start:stop] = model.prefix_scores(pair)
    return [flat[:, index].view(query_count, key_count) for index in range(depth)]


@torch.no_grad()
def predict_counterfactual_score_matrix(
    model: SerialAnchorUpdateDecoder,
    queries: Tensor,
    keys: Tensor,
    batch_size: int,
) -> Tensor:
    query_count, key_count = queries.shape[0], keys.shape[0]
    flat = torch.empty(query_count * key_count, device=queries.device)
    for start in range(0, flat.shape[0], batch_size):
        stop = min(start + batch_size, flat.shape[0])
        index = torch.arange(start, stop, device=queries.device)
        pair = torch.cat((queries[index // key_count], keys[index % key_count]), dim=-1)
        flat[start:stop] = model(pair, route_source="initial")[:, 0]
    return flat.view(query_count, key_count)


@torch.no_grad()
def layer_diagnostics(model: SerialAnchorUpdateDecoder, pair: Tensor) -> list[dict[str, float | int]]:
    initial = pair.unsqueeze(1)
    hidden = initial
    rows: list[dict[str, float | int]] = []
    for index, block in enumerate(model.blocks, start=1):
        route = block.route(hidden)
        base_route = block.route(initial)
        xor = route.indices ^ base_route.indices
        bit_flip = ((xor[..., None] & block.powers.view(1, 1, 1, -1)) != 0).float().mean()
        update = block.hard_update(route)
        hidden_norm = hidden.norm(dim=-1).mean()
        update_norm = update.norm(dim=-1).mean()
        rows.append(
            {
                "residual_block": index,
                "effective_depth": index + 1,
                "route_code_change_rate": float((xor != 0).float().mean().item()),
                "route_bit_flip_rate": float(bit_flip.item()),
                "mean_abs_margin": float(route.margins.abs().mean().item()),
                "hidden_rms_before": float(hidden.square().mean().sqrt().item()),
                "update_rms": float(update.square().mean().sqrt().item()),
                "mean_update_norm": float(update_norm.item()),
                "update_to_hidden_norm": float((update_norm / hidden_norm.clamp_min(1e-12)).item()),
            }
        )
        hidden = hidden + update
    return rows


def sample_uniform_pairs(pairs: object, count: int, seed: int) -> Tensor:
    generator = torch.Generator(device=pairs.test_queries.device).manual_seed(seed)
    query_index = torch.randint(
        pairs.test_queries.shape[0], (count,), generator=generator, device=pairs.test_queries.device
    )
    key_index = torch.randint(
        pairs.test_keys.shape[0], (count,), generator=generator, device=pairs.test_keys.device
    )
    return torch.cat((pairs.test_queries[query_index], pairs.test_keys[key_index]), dim=-1)


def run(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    pairs = make_relation_pairs(args, device)
    model = SerialAnchorUpdateDecoder(
        2 * args.input_dim,
        depth=args.depth,
        tables=args.tables,
        comparisons=args.comparisons,
        variant=args.variant,
        seed=args.seed,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    generator = torch.Generator(device=device).manual_seed(args.seed + 809)
    history: list[dict[str, float | int]] = []

    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        indices = torch.randint(
            0,
            pairs.positive.shape[0],
            (args.batch_size,),
            generator=generator,
            device=device,
        )
        loss = train_loss(model, pairs, indices)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % args.eval_every == 0 or step == args.steps:
            row = {
                "step": step,
                "loss": float(loss.detach().item()),
                "train_pair_accuracy": pair_accuracy(model, pairs, args.eval_batch_size),
            }
            history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    elapsed = time.perf_counter() - started

    model.eval()
    test_target = teacher_scores(pairs.test_queries, pairs.test_keys, pairs.relation)
    prefix_predictions = predict_prefix_score_matrices(
        model, pairs.test_queries, pairs.test_keys, args.eval_batch_size
    )
    prefix_metrics: list[dict[str, float | int]] = []
    for index, prediction in enumerate(prefix_predictions):
        metrics = retrieval_metrics(test_target, prediction, args.top_k, args.seed + 601)
        previous = prefix_metrics[-1] if prefix_metrics else None
        prefix_metrics.append(
            {
                "effective_depth": index + 1,
                **metrics,
                "marginal_topk_recall": metrics["topk_recall"]
                - (float(previous["topk_recall"]) if previous else 0.0),
                "marginal_score_r2": metrics["score_r2"]
                - (float(previous["score_r2"]) if previous else 0.0),
            }
        )
    final_metrics = prefix_metrics[-1]
    counterfactual_prediction = predict_counterfactual_score_matrix(
        model, pairs.test_queries, pairs.test_keys, args.eval_batch_size
    )
    counterfactual_metrics = retrieval_metrics(
        test_target, counterfactual_prediction, args.top_k, args.seed + 601
    )
    train_accuracy = pair_accuracy(model, pairs, args.eval_batch_size)
    counterfactual_train_accuracy = pair_accuracy(CounterfactualDecoder(model), pairs, args.eval_batch_size)
    diagnostics = layer_diagnostics(
        model,
        sample_uniform_pairs(pairs, args.diagnostic_pairs, args.seed + 1229),
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"anchor_{args.variant}_L{args.depth}_seed{args.seed}"
    checkpoint = args.out_dir / f"{stem}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "input_dim": 2 * args.input_dim,
                "depth": args.depth,
                "tables": args.tables,
                "comparisons": args.comparisons,
                "variant": args.variant,
                "seed": args.seed,
            },
        },
        checkpoint,
    )
    result: dict[str, object] = {
        "variant": f"serial_anchor_{args.variant}",
        "update_variant": args.variant,
        "depth": args.depth,
        "residual_blocks": args.depth - 1,
        "hidden_dim": 2 * args.input_dim,
        "tables_per_layer": args.tables,
        "comparisons": args.comparisons,
        "fixed_zero_thresholds": True,
        "zero_payload_initialization": True,
        "parameters": parameter_count(model),
        "steps": args.steps,
        "elapsed_seconds": elapsed,
        "steps_per_second": args.steps / elapsed,
        "seed": args.seed,
        "train_pair_accuracy": train_accuracy,
        "train_held_gap": train_accuracy - float(final_metrics["random_pair_order_accuracy"]),
        "counterfactual_train_pair_accuracy": counterfactual_train_accuracy,
        "payload_rms_by_layer": model.payload_rms(),
        "prefix_metrics": prefix_metrics,
        "layer_diagnostics": diagnostics,
        "checkpoint": str(checkpoint),
        **{
            f"test_{key}": value
            for key, value in final_metrics.items()
            if key not in {"effective_depth", "marginal_topk_recall", "marginal_score_r2"}
        },
        **{f"counterfactual_{key}": value for key, value in counterfactual_metrics.items()},
    }
    (args.out_dir / f"{stem}.json").write_text(json.dumps(result, indent=2) + "\n")
    with (args.out_dir / f"{stem}_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    print(json.dumps(result, sort_keys=True), flush=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def mean(rows: list[dict[str, object]], key: str) -> float:
    return statistics.mean(float(row[key]) for row in rows)


def summarize(args: argparse.Namespace) -> None:
    results = [
        json.loads(path.read_text())
        for path in sorted(args.result_dir.glob("anchor_*_L*_seed*.json"))
    ]
    missing = [
        (variant, depth, seed)
        for variant in UPDATE_VARIANTS
        for depth in DEPTHS
        for seed in (0, 1)
        if not any(
            row["update_variant"] == variant and row["depth"] == depth and row["seed"] == seed
            for row in results
        )
    ]
    if missing:
        raise RuntimeError(f"Missing serial anchor-update runs: {missing}")

    nested = {"payload_rms_by_layer", "prefix_metrics", "layer_diagnostics"}
    scalar_rows = [{key: value for key, value in row.items() if key not in nested} for row in results]
    fieldnames = sorted({key for row in scalar_rows for key in row})
    summary_path = args.result_dir / "summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scalar_rows)

    full_vector = read_csv(args.full_vector_summary)
    width = next(
        row
        for row in read_csv(args.width_summary)
        if row["variant"] == "pc_mse_adamw" and int(row["width"]) == 16
    )
    degree = next(
        row
        for row in read_csv(args.degree_summary)
        if int(row["max_degree"]) == 2 and int(row["support_budget"]) == 8192
    )
    metrics = (
        "train_pair_accuracy",
        "test_random_pair_order_accuracy",
        "test_hard_negative_preference_accuracy",
        "test_topk_recall",
        "test_top1_accuracy",
        "test_spearman",
        "test_score_r2",
        "counterfactual_topk_recall",
        "counterfactual_spearman",
        "steps_per_second",
        "train_held_gap",
    )

    lines = [
        "# Serial Anchor-Normal PC-LUT Relation Decoder",
        "",
        "## Question and architecture",
        "",
        "The full-vector serial decoder established that recursive route composition helps relation decoding, but its free 64-dimensional chamber translations are parameter-inefficient. This experiment restricts each hidden update to the anchor coordinates that define the current comparison geometry:",
        "",
        "```text",
        "signed:    delta_h = sum[t,c] s[t,c](code_t) (e_a - e_b)",
        "two-sided: delta_h = sum[t,c] (s_a[t,c](code_t) e_a + s_b[t,c](code_t) e_b)",
        "h[l+1] = h[l] + delta_h[l]",
        "score = scalar_PC_LUT(h[L-1])",
        "```",
        "",
        "Signed updates move only along comparator normals and preserve the coordinate sum. Two-sided updates add an independent common-mode component. Both use T16/C5, fixed random anchors, fixed zero thresholds, zero payload initialization, min-margin STE, and AdamW score MSE.",
        "",
        "## Runs",
        "",
        "| Update | L | Seed | Params | Train pair | Held pair | Gap | Hard-neg | Top-16 | Top-1 | Spearman | R2 | CF Top-16 | Steps/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(results, key=lambda item: (item["update_variant"], item["depth"], item["seed"])):
        lines.append(
            f"| {row['update_variant']} | {row['depth']} | {row['seed']} | {row['parameters']:,} | "
            f"{row['train_pair_accuracy']:.4f} | {row['test_random_pair_order_accuracy']:.4f} | "
            f"{row['train_held_gap']:.4f} | {row['test_hard_negative_preference_accuracy']:.4f} | "
            f"{row['test_topk_recall']:.4f} | {row['test_top1_accuracy']:.4f} | "
            f"{row['test_spearman']:.4f} | {row['test_score_r2']:.4f} | "
            f"{row['counterfactual_topk_recall']:.4f} | {row['steps_per_second']:.1f} |"
        )

    lines.extend(
        [
            "",
            "## Two-seed means",
            "",
            "| Update | L | Params | Train pair | Held pair | Gap | Hard-neg | Top-16 | Top-1 | Spearman | R2 | CF Top-16 | CF Spearman | Steps/s |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    grouped: dict[tuple[str, int], dict[str, float | int]] = {}
    for variant in UPDATE_VARIANTS:
        for depth in DEPTHS:
            selected = [
                row for row in results if row["update_variant"] == variant and row["depth"] == depth
            ]
            values: dict[str, float | int] = {"parameters": int(selected[0]["parameters"])}
            values.update({metric: mean(selected, metric) for metric in metrics})
            grouped[(variant, depth)] = values
            lines.append(
                f"| {variant} | {depth} | {values['parameters']:,} | {values['train_pair_accuracy']:.4f} | "
                f"{values['test_random_pair_order_accuracy']:.4f} | {values['train_held_gap']:.4f} | "
                f"{values['test_hard_negative_preference_accuracy']:.4f} | {values['test_topk_recall']:.4f} | "
                f"{values['test_top1_accuracy']:.4f} | {values['test_spearman']:.4f} | "
                f"{values['test_score_r2']:.4f} | {values['counterfactual_topk_recall']:.4f} | "
                f"{values['counterfactual_spearman']:.4f} | {values['steps_per_second']:.1f} |"
            )

    lines.extend(
        [
            "",
            "## Controls",
            "",
            "| Model | Params | Held pair | Hard-neg | Top-16 | Top-1 | Spearman | R2 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for depth in DEPTHS:
        selected = [row for row in full_vector if int(row["depth"]) == depth]
        lines.append(
            f"| Full-vector serial L{depth} | {int(selected[0]['parameters']):,} | "
            f"{statistics.mean(float(row['test_random_pair_order_accuracy']) for row in selected):.4f} | "
            f"{statistics.mean(float(row['test_hard_negative_preference_accuracy']) for row in selected):.4f} | "
            f"{statistics.mean(float(row['test_topk_recall']) for row in selected):.4f} | "
            f"{statistics.mean(float(row['test_top1_accuracy']) for row in selected):.4f} | "
            f"{statistics.mean(float(row['test_spearman']) for row in selected):.4f} | "
            f"{statistics.mean(float(row['test_score_r2']) for row in selected):.4f} |"
        )
    lines.extend(
        [
            f"| Parallel W16/T256 additive | {int(width['parameters']):,} | {float(width['test_random_pair_order_accuracy']):.4f} | {float(width['test_hard_negative_preference_accuracy']):.4f} | {float(width['test_topk_recall']):.4f} | {float(width['test_top1_accuracy']):.4f} | {float(width['test_spearman']):.4f} | {float(width['test_score_r2']):.4f} |",
            f"| Offline screened degree-2 8K | {int(degree['parameters']):,} | {float(degree['test_random_pair_order_accuracy']):.4f} | {float(degree['test_hard_negative_preference_accuracy']):.4f} | {float(degree['test_topk_recall']):.4f} | {float(degree['test_top1_accuracy']):.4f} | {float(degree['test_spearman']):.4f} | {float(degree['uniform_validation_r2']):.4f} |",
        ]
    )

    for variant in UPDATE_VARIANTS:
        l16 = [row for row in results if row["update_variant"] == variant and row["depth"] == 16]
        lines.extend(
            [
                "",
                f"## {variant} L16 prefix and route diagnostics",
                "",
                "| Effective depth | Top-16 | Marginal Top-16 | Spearman | R2 | Marginal R2 | Route bit flip | Update/hidden norm |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for effective_depth in range(1, 17):
            prefixes = [row["prefix_metrics"][effective_depth - 1] for row in l16]
            diagnostics = (
                [row["layer_diagnostics"][effective_depth - 2] for row in l16]
                if effective_depth > 1
                else []
            )
            route_flip = statistics.mean(float(row["route_bit_flip_rate"]) for row in diagnostics) if diagnostics else 0.0
            norm_ratio = statistics.mean(float(row["update_to_hidden_norm"]) for row in diagnostics) if diagnostics else 0.0
            lines.append(
                f"| {effective_depth} | {statistics.mean(float(row['topk_recall']) for row in prefixes):.4f} | "
                f"{statistics.mean(float(row['marginal_topk_recall']) for row in prefixes):+.4f} | "
                f"{statistics.mean(float(row['spearman']) for row in prefixes):.4f} | "
                f"{statistics.mean(float(row['score_r2']) for row in prefixes):.4f} | "
                f"{statistics.mean(float(row['marginal_score_r2']) for row in prefixes):+.4f} | "
                f"{route_flip:.4f} | {norm_ratio:.4f} |"
            )

    signed_l16 = grouped[("signed", 16)]
    two_l16 = grouped[("two_sided", 16)]
    full_l16_rows = [row for row in full_vector if int(row["depth"]) == 16]
    full_l16_topk = statistics.mean(float(row["test_topk_recall"]) for row in full_l16_rows)
    full_l16_params = int(full_l16_rows[0]["parameters"])
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"At L16, signed uses {signed_l16['parameters']:,} parameters ({full_l16_params / int(signed_l16['parameters']):.1f}x fewer than full-vector) and reaches Top-16 {signed_l16['test_topk_recall']:.4f}, versus {full_l16_topk:.4f} for full-vector. Two-sided uses {two_l16['parameters']:,} parameters ({full_l16_params / int(two_l16['parameters']):.1f}x fewer) and reaches {two_l16['test_topk_recall']:.4f}.",
            "",
            f"The common-mode contribution is measured by the two-sided minus signed L16 difference: Top-16 {two_l16['test_topk_recall'] - signed_l16['test_topk_recall']:+.4f}, held-pair {two_l16['test_random_pair_order_accuracy'] - signed_l16['test_random_pair_order_accuracy']:+.4f}, and Spearman {two_l16['test_spearman'] - signed_l16['test_spearman']:+.4f}.",
            "",
            f"Forcing every hidden block to route on the original h0 changes L16 Top-16 from {signed_l16['test_topk_recall']:.4f} to {signed_l16['counterfactual_topk_recall']:.4f} for signed and from {two_l16['test_topk_recall']:.4f} to {two_l16['counterfactual_topk_recall']:.4f} for two-sided. This counterfactual isolates whether later decisions use the geometry created by earlier updates rather than merely accumulating independent width.",
            "",
            "The offline screened degree-2 row remains an oracle-style control: it uses target-conditioned support selection and is not an online trainable interaction graph. The next step is justified only if the sparse serial updates retain a meaningful fraction of full-vector depth scaling at substantially lower parameter cost.",
        ]
    )
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text("\n".join(lines) + "\n")
    print(summary_path)
    print(args.out_report)


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run":
        run(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
