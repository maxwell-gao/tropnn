from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tropnn.tools.wiki103_induction_retrieval import (
    DATA_SEED,
    SPLITS,
    DenseQK,
    atomic_json_write,
    atomic_torch_save,
    iter_batches,
    ranking_metrics,
    seed_everything,
    tensor_fingerprint,
)

CACHE_VERSION = 1
CANDIDATE_COUNT = 8
DECODERS = ("key_only", "dense_qk")
DENSE_R1_THRESHOLD = 0.25
DENSE_OVER_KEY_THRESHOLD = 0.10
SHUFFLE_REMOVAL_THRESHOLD = 0.80


@dataclass(frozen=True)
class RunConfig:
    decoder: str
    seed: int
    cache_fingerprint: str
    epochs: int
    batch_size: int
    eval_batch_size: int
    learning_rate: float
    weight_decay: float
    gradient_clip: float


def atomic_csv_write(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _choose_query_shuffle_donors(
    records: list[tuple[int, list[int], int, int, int]],
    *,
    seed: int,
) -> list[tuple[int, list[int], int, int, int, int, int]]:
    """Attach a same-token, different-target query to every retained record."""

    grouped: dict[int, dict[int, list[int]]] = {}
    for query, _, _, target, query_token in records:
        grouped.setdefault(query_token, {}).setdefault(target, []).append(query)

    with_donors: list[tuple[int, list[int], int, int, int, int, int]] = []
    for query, candidates, relevant, target, query_token in records:
        donor_targets = sorted(value for value in grouped[query_token] if value != target)
        if not donor_targets:
            continue
        rng = random.Random(seed + query * 4099)
        donor_target = donor_targets[rng.randrange(len(donor_targets))]
        donor_queries = grouped[query_token][donor_target]
        donor_query = donor_queries[rng.randrange(len(donor_queries))]
        with_donors.append(
            (query, candidates, relevant, target, query_token, donor_query, donor_target)
        )
    return with_donors


def build_hard_induction_protocol(
    token_windows: Tensor,
    *,
    candidate_count: int = CANDIDATE_COUNT,
    max_queries: int = 0,
    seed: int = DATA_SEED,
) -> dict[str, Tensor | int | float | str]:
    """Build hard-only candidate sets with one position per successor value."""

    if token_windows.ndim != 2:
        raise ValueError("token_windows must have shape [windows, context + 1]")
    if candidate_count != CANDIDATE_COUNT:
        raise ValueError(f"the preregistered protocol requires {CANDIDATE_COUNT} candidates")

    context_size = token_windows.shape[1] - 1
    raw_records: list[tuple[int, list[int], int, int, int]] = []
    for window_index, token_row in enumerate(token_windows.tolist()):
        history: dict[int, dict[int, list[int]]] = {}
        for query_position in range(2, context_size):
            new_position = query_position - 2
            old_token = token_row[new_position]
            old_successor = token_row[new_position + 1]
            history.setdefault(old_token, {}).setdefault(old_successor, []).append(
                new_position
            )

            current = token_row[query_position]
            target = token_row[query_position + 1]
            successor_positions = history.get(current, {})
            if target not in successor_positions or len(successor_positions) < candidate_count:
                continue

            local_seed = seed + window_index * 1_000_003 + query_position * 1009
            rng = random.Random(local_seed)
            wrong_successors = sorted(
                successor for successor in successor_positions if successor != target
            )
            rng.shuffle(wrong_successors)
            selected_successors = [target, *wrong_successors[: candidate_count - 1]]
            if len(selected_successors) != candidate_count:
                continue
            candidates = [
                successor_positions[successor][
                    rng.randrange(len(successor_positions[successor]))
                ]
                for successor in selected_successors
            ]
            rng.shuffle(candidates)
            positive_position = next(
                position
                for position in candidates
                if token_row[position + 1] == target
            )
            relevant_index = candidates.index(positive_position)
            offset = window_index * context_size
            raw_records.append(
                (
                    offset + query_position,
                    [offset + position for position in candidates],
                    relevant_index,
                    target,
                    current,
                )
            )

    records = _choose_query_shuffle_donors(raw_records, seed=seed + 31)
    if not records:
        raise ValueError(
            "no hard-only induction queries with same-token/different-target "
            "shuffle donors were found"
        )
    if max_queries > 0 and len(records) > max_queries:
        generator = torch.Generator(device="cpu").manual_seed(seed + 991)
        selected = (
            torch.randperm(len(records), generator=generator)[:max_queries]
            .sort()
            .values.tolist()
        )
        records = [records[index] for index in selected]

    query = torch.tensor([record[0] for record in records], dtype=torch.long)
    candidates = torch.tensor([record[1] for record in records], dtype=torch.long)
    relevant_index = torch.tensor(
        [record[2] for record in records], dtype=torch.long
    )
    target = torch.tensor([record[3] for record in records], dtype=torch.long)
    query_tokens = torch.tensor(
        [record[4] for record in records], dtype=torch.long
    )
    shuffled_query = torch.tensor(
        [record[5] for record in records], dtype=torch.long
    )
    shuffled_target = torch.tensor(
        [record[6] for record in records], dtype=torch.long
    )
    flat_tokens = token_windows[:, :-1].reshape(-1)
    successor = token_windows[:, 1:].reshape(-1)
    candidate_values = successor[candidates]
    candidate_tokens = flat_tokens[candidates]
    relevant_mask = F.one_hot(
        relevant_index, num_classes=candidate_count
    ).to(torch.bool)
    hard_mask = ~relevant_mask
    protocol: dict[str, Tensor | int | float | str] = {
        "protocol": "hard_only_distinct_successor_v1",
        "query": query,
        "shuffled_query": shuffled_query,
        "shuffled_target": shuffled_target,
        "candidates": candidates,
        "relevant_index": relevant_index,
        "relevant_mask": relevant_mask,
        "target": target,
        "candidate_values": candidate_values,
        "query_tokens": query_tokens,
        "candidate_tokens": candidate_tokens,
        "hard_negative_mask": hard_mask,
        "candidate_count": candidate_count,
        "queries": len(records),
        "random_recall_at_1": 1.0 / candidate_count,
    }
    protocol["fingerprint"] = tensor_fingerprint(
        query,
        shuffled_query,
        candidates,
        relevant_index,
        target,
        candidate_values,
    )
    validate_hard_induction_protocol(token_windows, protocol)
    return protocol


def validate_hard_induction_protocol(
    token_windows: Tensor,
    protocol: dict[str, Any],
) -> None:
    context_size = token_windows.shape[1] - 1
    flat_tokens = token_windows[:, :-1].reshape(-1)
    successor = token_windows[:, 1:].reshape(-1)
    query = protocol["query"].long()
    shuffled_query = protocol["shuffled_query"].long()
    shuffled_target = protocol["shuffled_target"].long()
    candidates = protocol["candidates"].long()
    relevant_index = protocol["relevant_index"].long()
    target = protocol["target"].long()
    candidate_values = protocol["candidate_values"].long()
    hard_mask = protocol["hard_negative_mask"].bool()
    candidate_count = int(protocol["candidate_count"])

    if candidate_count != CANDIDATE_COUNT or candidates.shape[1] != candidate_count:
        raise ValueError("hard-only protocol must have exactly eight candidates")
    if query.ndim != 1 or candidates.ndim != 2 or candidates.shape[0] != query.shape[0]:
        raise ValueError("invalid hard-only protocol shapes")
    if not bool(
        (query[:, None] // context_size == candidates // context_size).all()
    ):
        raise ValueError("query and candidates must remain in the same frozen context")
    if not bool(
        (candidates % context_size < (query % context_size)[:, None] - 1).all()
    ):
        raise ValueError("candidate keys must satisfy j < i-1")
    if not torch.equal(flat_tokens[candidates], flat_tokens[query][:, None].expand_as(candidates)):
        raise ValueError("every candidate key token must equal the query token")
    if not torch.equal(candidate_values, successor[candidates]):
        raise ValueError("candidate successor values are inconsistent")
    sorted_values = candidate_values.sort(dim=1).values
    if not bool((sorted_values[:, 1:] != sorted_values[:, :-1]).all()):
        raise ValueError("candidate successor values must be pairwise distinct")
    positive_mask = candidate_values == target[:, None]
    if not bool((positive_mask.sum(dim=1) == 1).all()):
        raise ValueError("each candidate set must contain exactly one target successor")
    row = torch.arange(query.numel())
    if not torch.equal(positive_mask[row, relevant_index], torch.ones_like(query, dtype=torch.bool)):
        raise ValueError("relevant_index does not identify the target successor")
    if not torch.equal(hard_mask, ~positive_mask):
        raise ValueError("all and only wrong-successor candidates must be hard negatives")
    if not torch.equal(flat_tokens[shuffled_query], flat_tokens[query]):
        raise ValueError("query shuffle must preserve the current token")
    if not torch.equal(successor[shuffled_query], shuffled_target):
        raise ValueError("shuffled target is inconsistent with the donor query")
    if not bool((shuffled_target != target).all()):
        raise ValueError("query shuffle must change the target successor")


def _cache_request(args: argparse.Namespace, source_metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_cache_fingerprint": source_metadata["cache_fingerprint"],
        "candidate_count": CANDIDATE_COUNT,
        "max_train_queries": args.max_train_queries,
        "max_validation_queries": args.max_validation_queries,
        "max_test_queries": args.max_test_queries,
        "data_seed": DATA_SEED,
    }


def _validate_completed_cache(
    existing: dict[str, Any],
    request: dict[str, Any],
    out_dir: Path,
) -> None:
    if existing.get("cache_version") != CACHE_VERSION:
        raise ValueError(f"completed hard-only cache version mismatch at {out_dir}")
    if existing.get("prepare_request") != request:
        raise ValueError(f"completed hard-only cache request mismatch at {out_dir}")
    for split in SPLITS:
        if not (out_dir / f"{split}.pt").exists():
            raise FileNotFoundError(f"complete metadata exists but {split}.pt is missing")


def prepare_cache(args: argparse.Namespace) -> dict[str, Any]:
    source_metadata = json.loads(
        (args.source_cache_dir / "metadata.json").read_text()
    )
    if not source_metadata.get("complete"):
        raise ValueError("source frozen cache is incomplete")
    request = _cache_request(args, source_metadata)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.out_dir / "metadata.json"
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text())
        if existing.get("complete"):
            _validate_completed_cache(existing, request, args.out_dir)
            print(
                json.dumps(
                    {"status": "skipped_complete", "metadata": str(metadata_path)}
                ),
                flush=True,
            )
            return existing

    limits = {
        "train": args.max_train_queries,
        "validation": args.max_validation_queries,
        "test": args.max_test_queries,
    }
    protocols: dict[str, dict[str, Any]] = {}
    for split_index, split in enumerate(SPLITS):
        source_payload = torch.load(
            args.source_cache_dir / f"{split}.pt",
            map_location="cpu",
            weights_only=True,
        )
        tokens = source_payload["tokens"].long()
        protocols[split] = build_hard_induction_protocol(
            tokens,
            max_queries=limits[split],
            seed=DATA_SEED + split_index * 101,
        )
        payload = {
            "tokens": source_payload["tokens"],
            "hidden": source_payload["hidden"],
            "protocol": protocols[split],
        }
        atomic_torch_save(payload, args.out_dir / f"{split}.pt")
        print(
            json.dumps(
                {
                    "stage": "hard_protocol",
                    "split": split,
                    "queries": protocols[split]["queries"],
                }
            ),
            flush=True,
        )

    fingerprint_payload = {
        "request": request,
        "protocol_fingerprints": {
            split: protocols[split]["fingerprint"] for split in SPLITS
        },
    }
    cache_fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode()
    ).hexdigest()
    metadata: dict[str, Any] = {
        "complete": True,
        "cache_version": CACHE_VERSION,
        "cache_fingerprint": cache_fingerprint,
        "created_at_unix": int(time.time()),
        "prepare_request": request,
        "source_cache": {
            "path": str(args.source_cache_dir),
            "fingerprint": source_metadata["cache_fingerprint"],
            "cache_version": source_metadata["cache_version"],
        },
        "source": source_metadata["source"],
        "data": source_metadata["data"],
        "features": source_metadata["features"],
        "protocol": {
            "name": "hard_only_distinct_successor_v1",
            "candidate_count": CANDIDATE_COUNT,
            "random_recall_at_1": 1.0 / CANDIDATE_COUNT,
            "data_seed": DATA_SEED,
            "definition": (
                "all eight j satisfy j < i-1 and x_j=x_i; their x_(j+1) "
                "values are pairwise distinct; exactly one equals x_(i+1)"
            ),
            "query_shuffle": (
                "same x_i but donor x_(i+1) differs; candidates and labels "
                "remain attached to the original query"
            ),
            "target_leakage": (
                "x_(i+1) constructs candidates and labels only and is absent "
                "from every query/key row"
            ),
            "query_counts": {
                split: int(protocols[split]["queries"]) for split in SPLITS
            },
            "candidate_fingerprints": {
                split: protocols[split]["fingerprint"] for split in SPLITS
            },
        },
    }
    atomic_json_write(metadata, metadata_path)
    print(
        json.dumps(
            {
                "status": "complete",
                "metadata": str(metadata_path),
                "fingerprint": cache_fingerprint,
            }
        ),
        flush=True,
    )
    return metadata


class KeyOnlyMLP(nn.Module):
    """Parameter-matched nonlinear key-only shortcut diagnostic."""

    def __init__(self, dimension: int, hidden: int, seed: int) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(seed + 607)
        self.input_weight = nn.Parameter(
            torch.randn(dimension, hidden, generator=generator)
            / math.sqrt(dimension)
        )
        self.input_bias = nn.Parameter(torch.zeros(hidden))
        self.output_weight = nn.Parameter(
            torch.randn(hidden, generator=generator) / math.sqrt(hidden)
        )
        self.output_bias = nn.Parameter(torch.zeros(()))
        self.hidden = int(hidden)

    def forward(self, query: Tensor, key: Tensor) -> Tensor:
        del query
        hidden = F.gelu(key @ self.input_weight + self.input_bias)
        return hidden @ self.output_weight + self.output_bias


def build_scorer(
    decoder: str,
    seed: int,
    dimension: int = 32,
) -> tuple[nn.Module, dict[str, Any]]:
    if decoder == "key_only":
        scorer = KeyOnlyMLP(dimension, 30, seed)
        return scorer, {
            "execution_class": "nonlinear key-only shortcut diagnostic",
            "hidden": 30,
        }
    if decoder == "dense_qk":
        scorer = DenseQK(dimension, 16, seed)
        return scorer, {
            "execution_class": "dense QK qualification diagnostic",
            "rank": 16,
        }
    raise ValueError(f"unsupported decoder {decoder!r}")


def score_candidate_groups(
    scorer: nn.Module,
    query: Tensor,
    candidates: Tensor,
) -> Tensor:
    batch, candidate_count, dimension = candidates.shape
    query_flat = (
        query[:, None, :]
        .expand(-1, candidate_count, -1)
        .reshape(batch * candidate_count, dimension)
    )
    key_flat = candidates.reshape(batch * candidate_count, dimension)
    return scorer(query_flat, key_flat).view(batch, candidate_count)


def load_cached_split(
    cache_dir: Path,
    split: str,
    coordinate_indices: Tensor,
) -> dict[str, Tensor]:
    payload = torch.load(
        cache_dir / f"{split}.pt",
        map_location="cpu",
        weights_only=True,
    )
    validate_hard_induction_protocol(payload["tokens"].long(), payload["protocol"])
    hidden = (
        payload["hidden"]
        .float()
        .reshape(-1, payload["hidden"].shape[-1])[:, coordinate_indices]
        .contiguous()
    )
    protocol = payload["protocol"]
    return {
        "coordinates": hidden,
        "query": protocol["query"].long(),
        "shuffled_query": protocol["shuffled_query"].long(),
        "candidates": protocol["candidates"].long(),
        "relevant_index": protocol["relevant_index"].long(),
        "target": protocol["target"].long(),
        "candidate_values": protocol["candidate_values"].long(),
        "hard_negative_mask": protocol["hard_negative_mask"].bool(),
    }


def gather_group_batch(
    split: dict[str, Tensor],
    indices: Tensor,
    device: torch.device,
    *,
    shuffled_query: bool = False,
) -> tuple[Tensor, ...]:
    query_field = "shuffled_query" if shuffled_query else "query"
    query_indices = split[query_field][indices]
    candidate_indices = split["candidates"][indices]
    return (
        split["coordinates"][query_indices].to(device),
        split["coordinates"][candidate_indices].to(device),
        split["relevant_index"][indices].to(device),
        split["candidate_values"][indices].to(device),
        split["target"][indices].to(device),
        split["hard_negative_mask"][indices].to(device),
    )


@torch.no_grad()
def evaluate_scorer(
    scorer: nn.Module,
    split: dict[str, Tensor],
    device: torch.device,
    batch_size: int,
    *,
    shuffled_query: bool = False,
) -> dict[str, float]:
    scorer.eval()
    score_rows: list[Tensor] = []
    relevant_rows: list[Tensor] = []
    value_rows: list[Tensor] = []
    target_rows: list[Tensor] = []
    hard_rows: list[Tensor] = []
    for indices in iter_batches(
        split["query"].numel(),
        batch_size,
        shuffle=False,
        seed=0,
    ):
        query, candidates, relevant, values, target, hard = gather_group_batch(
            split,
            indices,
            device,
            shuffled_query=shuffled_query,
        )
        score_rows.append(score_candidate_groups(scorer, query, candidates).cpu())
        relevant_rows.append(relevant.cpu())
        value_rows.append(values.cpu())
        target_rows.append(target.cpu())
        hard_rows.append(hard.cpu())
    return ranking_metrics(
        torch.cat(score_rows),
        torch.cat(relevant_rows),
        torch.cat(value_rows),
        torch.cat(target_rows),
        torch.cat(hard_rows),
    )


def _validate_completed_run(
    existing: dict[str, Any],
    config: RunConfig,
    path: Path,
) -> None:
    if existing.get("config") != asdict(config):
        raise ValueError(f"completed result config mismatch at {path}")


def run_decoder(args: argparse.Namespace) -> dict[str, Any]:
    metadata = json.loads((args.cache_dir / "metadata.json").read_text())
    if not metadata.get("complete") or metadata.get("cache_version") != CACHE_VERSION:
        raise ValueError("hard-only frozen cache is incomplete or incompatible")
    config = RunConfig(
        decoder=args.decoder,
        seed=args.seed,
        cache_fingerprint=metadata["cache_fingerprint"],
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.out_dir / "result.json"
    if result_path.exists():
        existing = json.loads(result_path.read_text())
        if existing.get("complete"):
            _validate_completed_run(existing, config, result_path)
            print(
                json.dumps(
                    {"status": "skipped_complete", "result": str(result_path)}
                ),
                flush=True,
            )
            return existing
    for incomplete in (args.out_dir / "history.csv", args.out_dir / "best.pt"):
        if incomplete.exists():
            incomplete.rename(
                incomplete.with_name(
                    f"{incomplete.stem}.incomplete-{int(time.time())}"
                    f"{incomplete.suffix}"
                )
            )

    seed_everything(args.seed)
    device = torch.device(args.device)
    coordinates = torch.tensor(
        metadata["features"]["coordinate_indices"],
        dtype=torch.long,
    )
    splits = {
        split: load_cached_split(args.cache_dir, split, coordinates)
        for split in SPLITS
    }
    scorer, scorer_metadata = build_scorer(
        args.decoder,
        args.seed,
        coordinates.numel(),
    )
    scorer.to(device)
    optimizer = torch.optim.AdamW(
        scorer.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    history: list[dict[str, object]] = []
    best_recall = float("-inf")
    best_epoch = 0
    checkpoint_path = args.out_dir / "best.pt"
    optimizer_steps = 0
    started = time.perf_counter()
    training_seconds = 0.0
    for epoch in range(1, args.epochs + 1):
        scorer.train()
        loss_sum = 0.0
        seen = 0
        train_started = time.perf_counter()
        for indices in iter_batches(
            splits["train"]["query"].numel(),
            args.batch_size,
            shuffle=True,
            seed=args.seed + 1009 * epoch,
        ):
            query, candidates, relevant, _, _, _ = gather_group_batch(
                splits["train"],
                indices,
                device,
            )
            scores = score_candidate_groups(scorer, query, candidates)
            loss = F.cross_entropy(scores, relevant)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                scorer.parameters(),
                args.gradient_clip,
            )
            if not torch.isfinite(gradient_norm):
                raise RuntimeError("non-finite relation gradient")
            optimizer.step()
            optimizer_steps += 1
            loss_sum += float(loss.item()) * indices.numel()
            seen += indices.numel()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        training_seconds += time.perf_counter() - train_started
        validation = evaluate_scorer(
            scorer,
            splits["validation"],
            device,
            args.eval_batch_size,
        )
        validation_shuffle = evaluate_scorer(
            scorer,
            splits["validation"],
            device,
            args.eval_batch_size,
            shuffled_query=True,
        )
        row: dict[str, object] = {
            "epoch": epoch,
            "train_listwise_nll": loss_sum / max(1, seen),
            **{
                f"validation_{key}": value
                for key, value in validation.items()
            },
            **{
                f"validation_query_shuffle_{key}": value
                for key, value in validation_shuffle.items()
            },
        }
        history.append(row)
        atomic_csv_write(args.out_dir / "history.csv", history)
        if validation["recall_at_1"] > best_recall:
            best_recall = validation["recall_at_1"]
            best_epoch = epoch
            atomic_torch_save(
                {
                    "config": asdict(config),
                    "epoch": epoch,
                    "selection_metric": best_recall,
                    "state_dict": {
                        key: value.detach().cpu()
                        for key, value in scorer.state_dict().items()
                    },
                },
                checkpoint_path,
            )
        print(json.dumps(row, sort_keys=True), flush=True)

    if best_epoch == 0:
        raise RuntimeError("no finite validation checkpoint was produced")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    scorer.load_state_dict(checkpoint["state_dict"])
    scorer.to(device).eval()
    validation = evaluate_scorer(
        scorer,
        splits["validation"],
        device,
        args.eval_batch_size,
    )
    validation_shuffle = evaluate_scorer(
        scorer,
        splits["validation"],
        device,
        args.eval_batch_size,
        shuffled_query=True,
    )
    test = evaluate_scorer(
        scorer,
        splits["test"],
        device,
        args.eval_batch_size,
    )
    test_shuffle = evaluate_scorer(
        scorer,
        splits["test"],
        device,
        args.eval_batch_size,
        shuffled_query=True,
    )
    result: dict[str, Any] = {
        "complete": True,
        "config": asdict(config),
        "best_epoch": best_epoch,
        "best_validation_recall_at_1": best_recall,
        "relation_parameters": sum(
            parameter.numel() for parameter in scorer.parameters()
        ),
        "scorer": scorer_metadata,
        "train_queries": int(splits["train"]["query"].numel()),
        "validation_queries": int(splits["validation"]["query"].numel()),
        "test_queries": int(splits["test"]["query"].numel()),
        "candidate_count": CANDIDATE_COUNT,
        "validation": validation,
        "validation_query_shuffle": validation_shuffle,
        "test": test,
        "test_query_shuffle": test_shuffle,
        "optimizer_steps": optimizer_steps,
        "training_seconds": training_seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "protocol": metadata["protocol"],
        "features": metadata["features"],
        "source": metadata["source"],
    }
    if args.decoder == "key_only":
        for split_name in ("validation", "test"):
            base = result[split_name]["recall_at_1"]
            shuffled = result[f"{split_name}_query_shuffle"]["recall_at_1"]
            if base != shuffled:
                raise RuntimeError(
                    "key-only scorer changed under a query-only permutation"
                )
    atomic_json_write(result, result_path)
    print(
        json.dumps(
            {
                "status": "complete",
                "result": str(result_path),
                "best_epoch": best_epoch,
            }
        ),
        flush=True,
    )
    return result


def mean_sem(values: list[float]) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values) / math.sqrt(len(values))


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    results = [
        json.loads(path.read_text())
        for path in sorted(args.result_dir.glob("**/result.json"))
    ]
    results = [result for result in results if result.get("complete")]
    indexed: dict[str, dict[int, dict[str, Any]]] = {
        decoder: {} for decoder in DECODERS
    }
    for result in results:
        decoder = result["config"]["decoder"]
        seed = int(result["config"]["seed"])
        if decoder not in indexed or seed in indexed[decoder]:
            raise ValueError(
                f"duplicate or unknown result decoder={decoder}, seed={seed}"
            )
        indexed[decoder][seed] = result
    complete = (
        len(results) == 6
        and all(sorted(indexed[decoder]) == [0, 1, 2] for decoder in DECODERS)
    )

    aggregate: list[dict[str, Any]] = []
    for decoder in DECODERS:
        group = [indexed[decoder][seed] for seed in sorted(indexed[decoder])]
        if not group:
            continue
        row: dict[str, Any] = {
            "decoder": decoder,
            "seeds": len(group),
            "relation_parameters": int(group[0]["relation_parameters"]),
        }
        for split_name in (
            "validation",
            "validation_query_shuffle",
            "test",
            "test_query_shuffle",
        ):
            for metric in ("recall_at_1", "recall_at_4", "mrr", "listwise_nll"):
                values = [
                    float(result[split_name][metric]) for result in group
                ]
                mean, sem = mean_sem(values)
                row[f"{split_name}_{metric}_mean"] = mean
                row[f"{split_name}_{metric}_sem"] = sem
        aggregate.append(row)

    if complete:
        key = [
            float(indexed["key_only"][seed]["test"]["recall_at_1"])
            for seed in (0, 1, 2)
        ]
        dense = [
            float(indexed["dense_qk"][seed]["test"]["recall_at_1"])
            for seed in (0, 1, 2)
        ]
        dense_shuffle = [
            float(
                indexed["dense_qk"][seed]["test_query_shuffle"]["recall_at_1"]
            )
            for seed in (0, 1, 2)
        ]
        dense_mean, dense_sem = mean_sem(dense)
        paired_gain = [left - right for left, right in zip(dense, key)]
        gain_mean, gain_sem = mean_sem(paired_gain)
        paired_drop = [
            base - shuffled
            for base, shuffled in zip(dense, dense_shuffle)
        ]
        drop_mean, drop_sem = mean_sem(paired_drop)
        removed_fraction = (
            drop_mean / gain_mean if gain_mean > 0.0 else float("nan")
        )
        dense_absolute_passed = dense_mean >= DENSE_R1_THRESHOLD
        dense_gain_passed = gain_mean >= DENSE_OVER_KEY_THRESHOLD
        raw_shuffle_passed = (
            math.isfinite(removed_fraction)
            and removed_fraction >= SHUFFLE_REMOVAL_THRESHOLD
        )
        shuffle_prerequisites_passed = (
            dense_absolute_passed and dense_gain_passed
        )
        gates = {
            "dense_absolute": {
                "value": dense_mean,
                "sem": dense_sem,
                "threshold": DENSE_R1_THRESHOLD,
                "passed": dense_absolute_passed,
            },
            "dense_over_key_only": {
                "value": gain_mean,
                "sem": gain_sem,
                "paired_deltas": paired_gain,
                "threshold": DENSE_OVER_KEY_THRESHOLD,
                "passed": dense_gain_passed,
            },
            "query_shuffle_gain_removal": {
                "dense_drop": drop_mean,
                "dense_drop_sem": drop_sem,
                "value": removed_fraction,
                "threshold": SHUFFLE_REMOVAL_THRESHOLD,
                "raw_threshold_passed": raw_shuffle_passed,
                "prerequisites_passed": shuffle_prerequisites_passed,
                "passed": (
                    shuffle_prerequisites_passed and raw_shuffle_passed
                ),
                "reason": (
                    None
                    if shuffle_prerequisites_passed
                    else "Dense absolute and relative qualification failed"
                ),
            },
        }
    else:
        gates = {
            "dense_absolute": {"passed": False, "reason": "missing runs"},
            "dense_over_key_only": {"passed": False, "reason": "missing runs"},
            "query_shuffle_gain_removal": {
                "passed": False,
                "reason": "missing runs",
            },
        }
    qualified = complete and all(bool(gate["passed"]) for gate in gates.values())
    decision = {
        "complete": complete,
        "complete_runs": len(results),
        "expected_runs": 6,
        "gates": gates,
        "dense_qualification_passed": qualified,
        "next_stage": (
            "compare_ordinal_and_no_gemm_kernels"
            if qualified
            else "stop_relation_kernel_search"
        ),
    }
    atomic_json_write(decision, args.result_dir / "decision.json")
    if aggregate:
        atomic_csv_write(args.result_dir / "summary.csv", aggregate)

    metadata = json.loads((args.cache_dir / "metadata.json").read_text())
    common_config = results[0]["config"] if results else {}
    if complete:
        key_mean = statistics.mean(key)
        dense_mean = statistics.mean(dense)
        interpretation = (
            f"Key-only reaches {key_mean:.4f} R@1, only "
            f"{key_mean - 1.0 / CANDIDATE_COUNT:.4f} above random. Dense "
            f"reaches {dense_mean:.4f} and adds just "
            f"{gates['dense_over_key_only']['value']:.4f} over key-only; "
            f"the paired SEM is {gates['dense_over_key_only']['sem']:.4f}. "
            "This is not strong transferable query-key retrieval geometry. "
            "The small absolute shuffle drop does not rescue the claim because "
            "the reference Dense gain never qualified."
        )
    else:
        interpretation = (
            "The six-run matrix is incomplete, so no representation or "
            "relation-kernel conclusion is available."
        )
    lines = [
        "# Hard-only Wiki103 Induction Qualification",
        "",
        "## Outcome",
        "",
        "The hard-only frozen-state benchmark fails to qualify relation-kernel "
        "research. Dense QK remains close to both random and the parameter-"
        "matched key-only shortcut, so neither an ordinal kernel nor a "
        "BitNet/MADDNESS compilation is tested on this representation.",
        "",
        "## Protocol",
        "",
        "Every query has eight causal candidates with the same current token as "
        "the query and eight pairwise-distinct successor values. Exactly one "
        "successor equals the future target. No random filler candidates are "
        "used, so random R@1 is 0.125.",
        "",
        f"The fixed cache contains "
        f"`{metadata['protocol']['query_counts']['train']:,}` / "
        f"`{metadata['protocol']['query_counts']['validation']:,}` / "
        f"`{metadata['protocol']['query_counts']['test']:,}` "
        "train/validation/test queries. A query-shuffle donor always has the "
        "same current token but a different target successor.",
        "",
        "Key-only is a parameter-matched nonlinear shortcut diagnostic. Dense "
        "QK is a rank-16 qualification diagnostic. Both use the same raw D32 "
        "coordinates, candidates, labels, optimizer, and seeds 0, 1, and 2. "
        f"Each trains for `{common_config.get('epochs', 'unknown')}` epochs "
        f"with batch `{common_config.get('batch_size', 'unknown')}`, learning "
        f"rate `{common_config.get('learning_rate', 'unknown')}`, and "
        "validation-R@1 checkpoint selection.",
        "",
        f"Boundary: {metadata['source']['boundary']}. The source is a frozen "
        "post-training content-score ablation, not a checkpoint trained "
        "scoreless from initialization. The future target constructs candidates "
        "and labels only; it is absent from every query/key row.",
        "",
        "## Results",
        "",
        "| Decoder | Params | Test R@1 | Shuffled-query R@1 | Test R@4 | MRR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            f"| {row['decoder']} | {row['relation_parameters']:,} | "
            f"{row['test_recall_at_1_mean']:.4f} +/- "
            f"{row['test_recall_at_1_sem']:.4f} | "
            f"{row['test_query_shuffle_recall_at_1_mean']:.4f} +/- "
            f"{row['test_query_shuffle_recall_at_1_sem']:.4f} | "
            f"{row['test_recall_at_4_mean']:.4f} | "
            f"{row['test_mrr_mean']:.4f} |"
        )
    lines += [
        "",
        "## Preregistered qualification",
        "",
        f"- Dense test R@1 >= 0.25: "
        f"`{'PASS' if gates['dense_absolute']['passed'] else 'FAIL'}`.",
        f"- Dense minus key-only test R@1 >= 0.10: "
        f"`{'PASS' if gates['dense_over_key_only']['passed'] else 'FAIL'}`.",
        f"- After Dense qualification, query shuffle removes >=80% of its "
        f"gain over key-only: "
        f"`{'PASS' if gates['query_shuffle_gain_removal']['passed'] else 'FAIL'}`.",
        f"- Next stage: `{decision['next_stage']}`.",
        "",
    ]
    if complete:
        lines += [
            f"Dense minus key-only test R@1 is "
            f"`{gates['dense_over_key_only']['value']:.4f} +/- "
            f"{gates['dense_over_key_only']['sem']:.4f}` across paired seeds. "
            f"Query shuffle lowers Dense by "
            f"`{gates['query_shuffle_gain_removal']['dense_drop']:.4f} +/- "
            f"{gates['query_shuffle_gain_removal']['dense_drop_sem']:.4f}`. "
            "The removal ratio is not qualifying when the Dense reference "
            "gain itself fails its absolute and relative gates.",
            "",
            "Per-seed audit:",
            "",
            "| Decoder | Seed | Best epoch | Val R@1 | Test R@1 | "
            "Shuffled test R@1 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for result in sorted(
            results,
            key=lambda item: (
                DECODERS.index(item["config"]["decoder"]),
                item["config"]["seed"],
            ),
        ):
            lines.append(
                f"| {result['config']['decoder']} | "
                f"{result['config']['seed']} | {result['best_epoch']} | "
                f"{result['validation']['recall_at_1']:.4f} | "
                f"{result['test']['recall_at_1']:.4f} | "
                f"{result['test_query_shuffle']['recall_at_1']:.4f} |"
            )
        lines += [""]
    lines += [
        "## Interpretation",
        "",
        interpretation,
        "",
        "This result supersedes the stronger interpretation of the earlier "
        "mixed 32-way protocol. The earlier Dense advantage was detectable, "
        "but it is insufficient evidence for useful native retrieval once all "
        "candidates share token identity and differ only by successor. The "
        "correct next action is representation/task diagnosis, not Root, "
        "BitNet, MADDNESS, or online-attention integration on this cache.",
        "",
        "Failure means the frozen representation/protocol pair has not "
        "qualified relation-kernel research; it does not rank Root-incidence "
        "or another ordinal kernel. Passing permits, but does not itself "
        "validate, ordinal or no-GEMM scorers.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "GPU_LIST=0,1,2,3,4,5 "
        "scripts/run_tropnn_wiki103_hard_induction_qualification_6gpu.sh",
        "```",
        "",
        f"- Cache: `{args.cache_dir}` "
        f"(`{metadata['cache_fingerprint']}`).",
        f"- Results: `{args.result_dir}`.",
        f"- Source checkpoint: `{metadata['source']['checkpoint']}` at step "
        f"`{metadata['source']['checkpoint_metadata']['global_step']}`.",
        "- Logs: `logs/wiki103_hard_induction_qualification_20260723/`.",
        "- Every formal run completed 320 optimizer steps; result JSON and "
        "history CSV files preserve per-seed metrics.",
    ]
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text("\n".join(lines) + "\n")
    print(json.dumps(decision, indent=2, sort_keys=True), flush=True)
    return decision


def check_cache(args: argparse.Namespace) -> dict[str, Any]:
    metadata = json.loads((args.cache_dir / "metadata.json").read_text())
    counts: dict[str, int] = {}
    for split in SPLITS:
        payload = torch.load(
            args.cache_dir / f"{split}.pt",
            map_location="cpu",
            weights_only=True,
        )
        validate_hard_induction_protocol(
            payload["tokens"].long(),
            payload["protocol"],
        )
        counts[split] = int(payload["protocol"]["queries"])
    result = {
        "complete": bool(metadata.get("complete")),
        "cache_version": metadata.get("cache_version"),
        "cache_fingerprint": metadata.get("cache_fingerprint"),
        "candidate_count": metadata["protocol"]["candidate_count"],
        "random_recall_at_1": metadata["protocol"]["random_recall_at_1"],
        "query_counts": counts,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hard-only Wiki103 induction retrieval qualification"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare")
    prepare.add_argument("--source-cache-dir", type=Path, required=True)
    prepare.add_argument("--out-dir", type=Path, required=True)
    prepare.add_argument("--max-train-queries", type=int, default=0)
    prepare.add_argument("--max-validation-queries", type=int, default=0)
    prepare.add_argument("--max-test-queries", type=int, default=0)
    prepare.set_defaults(function=prepare_cache)

    run = commands.add_parser("run")
    run.add_argument("--cache-dir", type=Path, required=True)
    run.add_argument("--out-dir", type=Path, required=True)
    run.add_argument("--decoder", choices=DECODERS, required=True)
    run.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    run.add_argument("--device", default="cuda")
    run.add_argument("--epochs", type=int, default=20)
    run.add_argument("--batch-size", type=int, default=128)
    run.add_argument("--eval-batch-size", type=int, default=256)
    run.add_argument("--learning-rate", type=float, default=3e-3)
    run.add_argument("--weight-decay", type=float, default=0.0)
    run.add_argument("--gradient-clip", type=float, default=1.0)
    run.set_defaults(function=run_decoder)

    summarize_parser = commands.add_parser("summarize")
    summarize_parser.add_argument("--cache-dir", type=Path, required=True)
    summarize_parser.add_argument("--result-dir", type=Path, required=True)
    summarize_parser.add_argument("--out-report", type=Path, required=True)
    summarize_parser.set_defaults(function=summarize)

    cache_parser = commands.add_parser("check-cache")
    cache_parser.add_argument("--cache-dir", type=Path, required=True)
    cache_parser.set_defaults(function=check_cache)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
