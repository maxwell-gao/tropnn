from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from tropnn.layers import PairwiseLUT


@dataclass(frozen=True)
class ProbeRow:
    support_kind: str
    threshold_mode: str
    budget: str
    anchor_policy: str
    input_dim: int
    support_size: int
    tables: int
    comparisons: int
    unique_routes: int
    compression_ratio: float
    route_entropy_bits: float
    kernel_quotient_r2: float
    kernel_quotient_spearman: float
    kernel_quotient_topk_recall: float
    kernel_quotient_ndcg: float
    kernel_hamming_r2: float
    kernel_hamming_spearman: float
    kernel_hamming_topk_recall: float
    kernel_hamming_ndcg: float
    a_kind: str
    pair_unique_routes: int
    pair_compression_ratio: float
    pair_route_entropy_bits: float
    pair_quotient_r2: float
    pair_quotient_spearman: float
    pair_quotient_topk_recall: float
    pair_quotient_ndcg: float
    pair_hamming_r2: float
    pair_hamming_spearman: float
    pair_hamming_topk_recall: float
    pair_hamming_ndcg: float
    pair_hamming_weight_bins: int

    def as_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe quotient-kernel geometry for PC-LUT routes. "
            "Probe 1 compares <x,y>, <m_rho(x),m_rho(y)>, and -Hamming(rho(x),rho(y)). "
            "Probe 2 sweeps route budgets for compression curves. "
            "Probe 3 compares x^T A y with <A,E[x⊗y|rho_pair]> and a Hamming-weight baseline."
        )
    )
    parser.add_argument("--support-kind", choices=["random_int", "binary", "permutation", "file"], default="random_int")
    parser.add_argument("--support-path", type=Path, default=None)
    parser.add_argument("--support-size", type=int, default=2048)
    parser.add_argument("--input-dim", type=int, default=32)
    parser.add_argument("--max-value", type=int, default=15)
    parser.add_argument("--normalize", choices=["none", "l2", "center_l2", "layernorm"], default="none")
    parser.add_argument("--budgets", default="T1_C4,T4_C4,T16_C5,T64_C6")
    parser.add_argument("--anchor-policy", default="random,expander,permuted")
    parser.add_argument("--threshold-mode", default="zero")
    parser.add_argument("--threshold-std", type=float, default=1.0)
    parser.add_argument("--anchor-seed", type=int, default=0)
    parser.add_argument("--query-count", type=int, default=128)
    parser.add_argument("--candidate-count", type=int, default=512)
    parser.add_argument("--pair-query-count", type=int, default=64)
    parser.add_argument("--pair-candidate-count", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--a-kinds", default="gaussian,binary,ternary,low_rank,all_ones")
    parser.add_argument("--low-rank", type=int, default=4)
    parser.add_argument("--max-bilinear-dim", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, default=None)
    return parser.parse_args()


def parse_list(text: str) -> list[str]:
    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("expected a non-empty comma-separated list")
    return values


def parse_budget(text: str) -> tuple[int, int]:
    if text.startswith("T") and "_C" in text:
        left, right = text[1:].split("_C", 1)
        return int(left), int(right)
    if "x" in text:
        left, right = text.split("x", 1)
        return int(left), int(right)
    raise ValueError(f"unsupported budget syntax {text!r}; use T16_C5 or 16x5")


def load_support(path: Path) -> Tensor:
    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, Tensor):
            return obj
        if isinstance(obj, dict):
            for key in ("x", "support", "hidden", "states"):
                value = obj.get(key)
                if isinstance(value, Tensor):
                    return value
        raise ValueError(f"could not find support tensor in {path}")
    if suffix == ".npy":
        import numpy as np

        return torch.from_numpy(np.load(path))
    raise ValueError(f"unsupported support suffix {suffix!r}")


def make_support(args: argparse.Namespace) -> Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    if args.support_kind == "file":
        if args.support_path is None:
            raise ValueError("--support-path is required with --support-kind=file")
        x = load_support(args.support_path)
    elif args.support_kind == "random_int":
        x = torch.randint(0, args.max_value + 1, (args.support_size, args.input_dim), generator=generator, dtype=torch.int64)
    elif args.support_kind == "binary":
        x = torch.randint(0, 2, (args.support_size, args.input_dim), generator=generator, dtype=torch.int64)
    elif args.support_kind == "permutation":
        template = torch.arange(args.input_dim, dtype=torch.int64)
        x = torch.stack([template[torch.randperm(args.input_dim, generator=generator)] for _ in range(args.support_size)], dim=0)
    else:
        raise ValueError(f"unsupported support kind {args.support_kind!r}")
    if x.ndim != 2:
        raise ValueError(f"support must be rank-2, got {tuple(x.shape)}")
    return normalize_support(x.to(torch.float32), args.normalize)


def normalize_support(x: Tensor, mode: str) -> Tensor:
    if mode == "none":
        return x
    if mode == "l2":
        return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    if mode == "center_l2":
        y = x - x.mean(dim=-1, keepdim=True)
        return y / y.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    if mode == "layernorm":
        y = x - x.mean(dim=-1, keepdim=True)
        return y / y.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-12)
    raise ValueError(f"unsupported normalize mode {mode!r}")


def configure_layer(
    *,
    input_dim: int,
    tables: int,
    comparisons: int,
    policy: str,
    threshold_mode: str,
    threshold_std: float,
    anchor_seed: int,
    seed: int,
) -> PairwiseLUT:
    layer = PairwiseLUT(
        input_dim=input_dim,
        output_dim=1,
        tables=tables,
        comparisons=comparisons,
        backend="torch",
        seed=seed,
        anchor_policy=policy,
        anchor_seed=anchor_seed,
        fixed_zero_threshold=threshold_mode == "zero",
    )
    if threshold_mode == "normal":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + 17)
        values = torch.randn(tables, comparisons, generator=generator) * threshold_std
        with torch.no_grad():
            layer.thresholds.copy_(values)
    return layer


def route_support(layer: PairwiseLUT, values: Tensor, *, batch_size: int, device: torch.device) -> Tensor:
    codes: list[Tensor] = []
    layer = layer.to(device)
    layer.eval()
    with torch.no_grad():
        for start in range(0, values.shape[0], batch_size):
            batch = values[start : start + batch_size].to(device)
            route = layer.route(batch.unsqueeze(1))
            codes.append(route.indices[:, 0, :].cpu())
    return torch.cat(codes, dim=0).to(torch.long)


def route_key(row: Tensor) -> tuple[int, ...]:
    return tuple(int(v) for v in row.tolist())


def route_means(x: Tensor, codes: Tensor) -> tuple[Tensor, dict[tuple[int, ...], int], Tensor]:
    sums: dict[tuple[int, ...], Tensor] = {}
    counts: dict[tuple[int, ...], int] = defaultdict(int)
    row_keys: list[tuple[int, ...]] = []
    x64 = x.to(torch.float64)
    for i, code in enumerate(codes):
        key = route_key(code)
        row_keys.append(key)
        counts[key] += 1
        if key in sums:
            sums[key] += x64[i]
        else:
            sums[key] = x64[i].clone()
    means = {key: sums[key] / float(count) for key, count in counts.items()}
    x_hat = torch.stack([means[key] for key in row_keys], dim=0)
    return x_hat, dict(counts), torch.tensor([counts[key] for key in row_keys], dtype=torch.long)


def entropy_bits(counts: dict[tuple[int, ...], int], total: int) -> float:
    entropy = 0.0
    for size in counts.values():
        p = size / total
        entropy -= p * math.log2(p)
    return entropy


def sample_indices(n: int, count: int, seed: int) -> Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    if count >= n:
        return torch.arange(n)
    return torch.randperm(n, generator=generator)[:count]


def popcount_table(comparisons: int) -> Tensor:
    return torch.tensor([int(i).bit_count() for i in range(1 << comparisons)], dtype=torch.long)


def hamming_score_matrix(query_codes: Tensor, candidate_codes: Tensor, comparisons: int) -> Tensor:
    pc = popcount_table(comparisons)
    xor = torch.bitwise_xor(query_codes[:, None, :], candidate_codes[None, :, :])
    dist = pc[xor].sum(dim=-1).to(torch.float64)
    return -dist


def average_ranks(values: Tensor) -> Tensor:
    flat = values.reshape(-1).to(torch.float64)
    order = torch.argsort(flat)
    sorted_values = flat[order]
    ranks_sorted = torch.empty_like(sorted_values)
    n = sorted_values.numel()
    start = 0
    while start < n:
        end = start + 1
        while end < n and sorted_values[end] == sorted_values[start]:
            end += 1
        rank = (start + end - 1) / 2.0
        ranks_sorted[start:end] = rank
        start = end
    ranks = torch.empty_like(ranks_sorted)
    ranks[order] = ranks_sorted
    return ranks.reshape_as(values)


def r2_score(true: Tensor, pred: Tensor) -> float:
    t = true.to(torch.float64)
    p = pred.to(torch.float64)
    mse = ((t - p) ** 2).mean()
    var = ((t - t.mean()) ** 2).mean()
    if float(var.item()) < 1e-24:
        return 1.0 if float(mse.item()) < 1e-24 else float("nan")
    return float((1.0 - mse / var).item())


def spearman_corr(true: Tensor, pred: Tensor) -> float:
    tr = average_ranks(true)
    pr = average_ranks(pred)
    tr = tr - tr.mean()
    pr = pr - pr.mean()
    denom = tr.norm() * pr.norm()
    if float(denom.item()) < 1e-24:
        return float("nan")
    return float((tr.flatten() @ pr.flatten() / denom).item())


def topk_recall(true: Tensor, pred: Tensor, k: int) -> float:
    k = min(k, true.shape[-1])
    true_top = torch.topk(true, k=k, dim=-1).indices
    pred_top = torch.topk(pred, k=k, dim=-1).indices
    recalls = []
    for t, p in zip(true_top, pred_top):
        recalls.append(len(set(t.tolist()) & set(p.tolist())) / k)
    return float(sum(recalls) / len(recalls))


def ndcg(true: Tensor, pred: Tensor, k: int) -> float:
    k = min(k, true.shape[-1])
    pred_top = torch.topk(pred, k=k, dim=-1).indices
    ideal_top = torch.topk(true, k=k, dim=-1).indices
    discounts = 1.0 / torch.log2(torch.arange(k, dtype=torch.float64) + 2.0)
    scores = []
    for row, p_idx, i_idx in zip(true, pred_top, ideal_top):
        rel = row.to(torch.float64)
        rel = rel - rel.min()
        dcg = (rel[p_idx] * discounts).sum()
        idcg = (rel[i_idx] * discounts).sum()
        if float(idcg.item()) < 1e-24:
            scores.append(1.0)
        else:
            scores.append(float((dcg / idcg).item()))
    return float(sum(scores) / len(scores))


def ranking_metrics(true: Tensor, pred: Tensor, top_k: int) -> dict[str, float]:
    return {
        "r2": r2_score(true, pred),
        "spearman": spearman_corr(true, pred),
        "topk_recall": topk_recall(true, pred, top_k),
        "ndcg": ndcg(true, pred, top_k),
    }


def make_bilinear_matrices(kinds: list[str], dim: int, low_rank: int, seed: int) -> dict[str, Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 101)
    mats: dict[str, Tensor] = {}
    for kind in kinds:
        if kind == "gaussian":
            mats[kind] = torch.randn(dim, dim, generator=generator, dtype=torch.float64) / dim
        elif kind == "binary":
            mats[kind] = (torch.randint(0, 2, (dim, dim), generator=generator).to(torch.float64) * 2.0 - 1.0) / dim
        elif kind == "ternary":
            raw = torch.randint(0, 3, (dim, dim), generator=generator, dtype=torch.int64) - 1
            mats[kind] = raw.to(torch.float64) / math.sqrt(max(1, dim * dim * 2 // 3))
        elif kind == "low_rank":
            rank = max(1, min(low_rank, dim))
            u = torch.randn(rank, dim, generator=generator, dtype=torch.float64) / math.sqrt(rank)
            v = torch.randn(rank, dim, generator=generator, dtype=torch.float64) / math.sqrt(dim)
            mats[kind] = u.T @ v
        elif kind == "all_ones":
            mats[kind] = torch.ones(dim, dim, dtype=torch.float64) / dim
        else:
            raise ValueError(f"unsupported A kind {kind!r}")
    return mats


def pair_bilinear_features(x: Tensor, query_idx: Tensor, candidate_idx: Tensor, max_bilinear_dim: int) -> tuple[Tensor, Tensor, Tensor]:
    dim = x.shape[1]
    if dim * dim > max_bilinear_dim:
        raise ValueError(f"bilinear dim {dim * dim} exceeds --max-bilinear-dim={max_bilinear_dim}")
    q = x[query_idx].to(torch.float64)
    c = x[candidate_idx].to(torch.float64)
    left = q[:, None, :].expand(q.shape[0], c.shape[0], dim).reshape(-1, dim)
    right = c[None, :, :].expand(q.shape[0], c.shape[0], dim).reshape(-1, dim)
    z = (left[:, :, None] * right[:, None, :]).reshape(left.shape[0], dim * dim)
    pair_input = torch.cat([left.to(torch.float32), right.to(torch.float32)], dim=-1)
    return pair_input, z, torch.tensor([q.shape[0], c.shape[0]], dtype=torch.long)


def hamming_weight_fit(true: Tensor, pair_codes: Tensor, comparisons: int) -> tuple[Tensor, int]:
    pc = popcount_table(comparisons)
    weights = pc[pair_codes].sum(dim=-1)
    pred = torch.empty_like(true.reshape(-1), dtype=torch.float64)
    flat_true = true.reshape(-1).to(torch.float64)
    for weight in torch.unique(weights):
        mask = weights == weight
        pred[mask] = flat_true[mask].mean()
    return pred.reshape_as(true), int(torch.unique(weights).numel())


def run_budget(args: argparse.Namespace, x: Tensor, threshold_mode: str, budget: str, policy: str, a_kinds: list[str]) -> list[ProbeRow]:
    tables, comparisons = parse_budget(budget)
    device = torch.device(args.device)
    layer = configure_layer(
        input_dim=x.shape[1],
        tables=tables,
        comparisons=comparisons,
        policy=policy,
        threshold_mode=threshold_mode,
        threshold_std=args.threshold_std,
        anchor_seed=args.anchor_seed,
        seed=args.seed,
    )
    codes = route_support(layer, x, batch_size=args.batch_size, device=device)
    x_hat, counts, _ = route_means(x, codes)
    route_entropy = entropy_bits(counts, x.shape[0])
    query_idx = sample_indices(x.shape[0], args.query_count, args.seed + 201)
    candidate_idx = sample_indices(x.shape[0], args.candidate_count, args.seed + 307)
    q = x[query_idx].to(torch.float64)
    c = x[candidate_idx].to(torch.float64)
    q_hat = x_hat[query_idx]
    c_hat = x_hat[candidate_idx]
    true_kernel = q @ c.T
    quotient_kernel = q_hat @ c_hat.T
    hamming_kernel = hamming_score_matrix(codes[query_idx], codes[candidate_idx], comparisons)
    quotient_metrics = ranking_metrics(true_kernel, quotient_kernel, args.top_k)
    hamming_metrics = ranking_metrics(true_kernel, hamming_kernel, args.top_k)

    pair_query_idx = sample_indices(x.shape[0], args.pair_query_count, args.seed + 401)
    pair_candidate_idx = sample_indices(x.shape[0], args.pair_candidate_count, args.seed + 503)
    pair_input, z, shape = pair_bilinear_features(x, pair_query_idx, pair_candidate_idx, args.max_bilinear_dim)
    pair_layer = configure_layer(
        input_dim=x.shape[1] * 2,
        tables=tables,
        comparisons=comparisons,
        policy=policy,
        threshold_mode=threshold_mode,
        threshold_std=args.threshold_std,
        anchor_seed=args.anchor_seed,
        seed=args.seed,
    )
    pair_codes = route_support(pair_layer, pair_input, batch_size=args.batch_size, device=device)
    z_hat, pair_counts, _ = route_means(z.to(torch.float32), pair_codes)
    pair_entropy = entropy_bits(pair_counts, z.shape[0])
    pair_q = int(shape[0].item())
    pair_c = int(shape[1].item())
    matrices = make_bilinear_matrices(a_kinds, x.shape[1], args.low_rank, args.seed)
    rows: list[ProbeRow] = []
    for a_kind, a in matrices.items():
        a_vec = a.reshape(-1)
        true_pair = (z @ a_vec).reshape(pair_q, pair_c)
        quotient_pair = (z_hat @ a_vec).reshape(pair_q, pair_c)
        hamming_pair, bins = hamming_weight_fit(true_pair, pair_codes, comparisons)
        pair_quotient_metrics = ranking_metrics(true_pair, quotient_pair, args.top_k)
        pair_hamming_metrics = ranking_metrics(true_pair, hamming_pair, args.top_k)
        rows.append(
            ProbeRow(
                support_kind=args.support_kind,
                threshold_mode=threshold_mode,
                budget=budget,
                anchor_policy=policy,
                input_dim=x.shape[1],
                support_size=x.shape[0],
                tables=tables,
                comparisons=comparisons,
                unique_routes=len(counts),
                compression_ratio=len(counts) / x.shape[0],
                route_entropy_bits=route_entropy,
                kernel_quotient_r2=quotient_metrics["r2"],
                kernel_quotient_spearman=quotient_metrics["spearman"],
                kernel_quotient_topk_recall=quotient_metrics["topk_recall"],
                kernel_quotient_ndcg=quotient_metrics["ndcg"],
                kernel_hamming_r2=hamming_metrics["r2"],
                kernel_hamming_spearman=hamming_metrics["spearman"],
                kernel_hamming_topk_recall=hamming_metrics["topk_recall"],
                kernel_hamming_ndcg=hamming_metrics["ndcg"],
                a_kind=a_kind,
                pair_unique_routes=len(pair_counts),
                pair_compression_ratio=len(pair_counts) / z.shape[0],
                pair_route_entropy_bits=pair_entropy,
                pair_quotient_r2=pair_quotient_metrics["r2"],
                pair_quotient_spearman=pair_quotient_metrics["spearman"],
                pair_quotient_topk_recall=pair_quotient_metrics["topk_recall"],
                pair_quotient_ndcg=pair_quotient_metrics["ndcg"],
                pair_hamming_r2=pair_hamming_metrics["r2"],
                pair_hamming_spearman=pair_hamming_metrics["spearman"],
                pair_hamming_topk_recall=pair_hamming_metrics["topk_recall"],
                pair_hamming_ndcg=pair_hamming_metrics["ndcg"],
                pair_hamming_weight_bins=bins,
            )
        )
    return rows


def write_csv(path: Path, rows: list[ProbeRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].as_dict().keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())


def write_json(path: Path, rows: list[ProbeRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([row.as_dict() for row in rows], indent=2) + "\n")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    x = make_support(args)
    rows: list[ProbeRow] = []
    for threshold_mode in parse_list(args.threshold_mode):
        for budget in parse_list(args.budgets):
            for policy in parse_list(args.anchor_policy):
                rows.extend(run_budget(args, x, threshold_mode, budget, policy, parse_list(args.a_kinds)))
    for row in rows:
        print(json.dumps(row.as_dict(), sort_keys=True))
    if args.out_csv is not None:
        write_csv(args.out_csv, rows)
    if args.out_json is not None:
        write_json(args.out_json, rows)


if __name__ == "__main__":
    main()
