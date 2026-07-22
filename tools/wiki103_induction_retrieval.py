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

from tropnn.layers.pair_kernel import (
    BalancedS4Router,
    CoxeterPairScorer,
    IntrinsicS4Kernel,
    RootIncidenceKernel,
    SameTableFullKernel,
)

CACHE_VERSION = 1
DATA_SEED = 1729
DECODERS = ("kendall", "same_table_full", "root_incidence", "dense_qk")
SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class RunConfig:
    decoder: str
    seed: int
    cache_fingerprint: str
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    gradient_clip: float


def prepare_request(args: argparse.Namespace) -> dict[str, Any]:
    """Describe every input that changes a frozen retrieval cache."""

    return {
        "config_sha256": sha256_file(args.config),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "context_size": args.context_size,
        "train_windows": args.train_windows,
        "validation_windows": args.validation_windows,
        "test_windows": args.test_windows,
        "relation_dim": args.relation_dim,
        "candidate_count": args.candidates,
        "hard_negative_priority": args.hard_negatives,
        "max_train_queries": args.max_train_queries,
        "max_validation_queries": args.max_validation_queries,
        "max_test_queries": args.max_test_queries,
        "data_seed": DATA_SEED,
    }


def validate_completed_cache(existing: dict[str, Any], request: dict[str, Any], out_dir: Path) -> None:
    if existing.get("cache_version") != CACHE_VERSION:
        raise ValueError(f"completed cache version mismatch at {out_dir}")
    if existing.get("prepare_request") != request:
        raise ValueError(f"completed cache request mismatch at {out_dir}")
    for split in SPLITS:
        if not (out_dir / f"{split}.pt").exists():
            raise FileNotFoundError(f"complete metadata exists but {split}.pt is missing")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_fingerprint(*tensors: Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def atomic_torch_save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def atomic_json_write(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def choose_window_starts(region_start: int, region_stop: int, context_size: int, count: int) -> Tensor:
    window = context_size + 1
    available = region_stop - region_start
    if count < 1:
        raise ValueError("window count must be positive")
    if available < count * window:
        raise ValueError(f"region of {available} tokens cannot hold {count} non-overlapping windows of {window}")
    bin_width = available / count
    starts = []
    for index in range(count):
        left = region_start + int(index * bin_width)
        right = region_start + int((index + 1) * bin_width) - window
        starts.append((left + max(left, right)) // 2)
    result = torch.tensor(starts, dtype=torch.long)
    if result.numel() > 1 and not bool((result[1:] - result[:-1] >= window).all()):
        raise RuntimeError("window construction produced overlap")
    return result


def gather_token_windows(tokens: Tensor, starts: Tensor, context_size: int) -> Tensor:
    offsets = torch.arange(context_size + 1, dtype=torch.long)
    return tokens[starts[:, None] + offsets[None, :]].to(torch.long).contiguous()


def zero_content_score_luts(model: nn.Module) -> dict[str, float | int]:
    tensors: list[Tensor] = []
    with torch.no_grad():
        for module in model.modules():
            score_lut = getattr(module, "score_lut", None)
            if isinstance(score_lut, Tensor):
                tensors.append(score_lut)
        before = sum(float(tensor.detach().float().abs().sum().item()) for tensor in tensors)
        for tensor in tensors:
            tensor.zero_()
        after = sum(float(tensor.detach().float().abs().sum().item()) for tensor in tensors)
    if not tensors:
        raise RuntimeError("no score_lut tensors found in frozen tropical model")
    if after != 0.0:
        raise RuntimeError("content score ablation was not exact")
    return {"score_lut_tensors": len(tensors), "score_lut_l1_before": before, "score_lut_l1_after": after}


def frozen_hidden_forward(model: nn.Module, tokens: Tensor) -> Tensor:
    """Return the exact final pre-unembedding state without materializing vocabulary logits."""

    x = model.embedding(tokens)
    if getattr(model, "embed_proj", None) is not None:
        x = model.embed_proj(x)
    for block in model.blocks:
        x = block(x)
    return model.final_norm(x)


def build_induction_protocol(
    token_windows: Tensor,
    *,
    candidate_count: int = 32,
    max_hard_negatives: int = 16,
    max_queries: int = 0,
    seed: int = DATA_SEED,
) -> dict[str, Tensor | int | float | str]:
    """Construct model-independent causal bigram-retrieval candidate sets.

    Exactly one old occurrence of the query bigram is included. Other old
    occurrences of the same bigram are excluded, so Recall@1 is unambiguous.
    Same-current-token/different-successor candidates are sampled first.
    """

    if token_windows.ndim != 2:
        raise ValueError("token_windows must have shape [windows, context + 1]")
    if candidate_count < 2:
        raise ValueError("candidate_count must be at least two")
    context_size = token_windows.shape[1] - 1
    records: list[tuple[int, list[int], int, int, list[int]]] = []
    for window_index, tokens in enumerate(token_windows.tolist()):
        for query_position in range(candidate_count + 1, context_size):
            current = tokens[query_position]
            target = tokens[query_position + 1]
            causal = list(range(0, query_position - 1))
            positives = [position for position in causal if tokens[position] == current and tokens[position + 1] == target]
            if not positives:
                continue
            hard = [position for position in causal if tokens[position] == current and tokens[position + 1] != target]
            local_seed = seed + window_index * 1_000_003 + query_position * 1009
            rng = random.Random(local_seed)
            positive = positives[rng.randrange(len(positives))]
            positive_set = set(positives)
            rng.shuffle(hard)
            chosen_hard = hard[: min(max_hard_negatives, candidate_count - 1)]
            chosen_set = set(chosen_hard)
            random_pool = [position for position in causal if position not in positive_set and position not in chosen_set]
            rng.shuffle(random_pool)
            negatives = chosen_hard + random_pool[: candidate_count - 1 - len(chosen_hard)]
            if len(negatives) != candidate_count - 1:
                continue
            candidates = [positive, *negatives]
            rng.shuffle(candidates)
            relevant_index = candidates.index(positive)
            hard_mask = [int(tokens[position] == current and tokens[position + 1] != target) for position in candidates]
            query_flat = window_index * context_size + query_position
            candidate_flat = [window_index * context_size + position for position in candidates]
            records.append((query_flat, candidate_flat, relevant_index, target, hard_mask))

    if not records:
        raise ValueError("no repeated-bigram induction queries were found")
    if max_queries > 0 and len(records) > max_queries:
        generator = torch.Generator(device="cpu").manual_seed(seed + 991)
        selected = torch.randperm(len(records), generator=generator)[:max_queries].sort().values.tolist()
        records = [records[index] for index in selected]

    query = torch.tensor([record[0] for record in records], dtype=torch.long)
    candidates = torch.tensor([record[1] for record in records], dtype=torch.long)
    relevant_index = torch.tensor([record[2] for record in records], dtype=torch.long)
    target = torch.tensor([record[3] for record in records], dtype=torch.long)
    hard_mask = torch.tensor([record[4] for record in records], dtype=torch.bool)
    flat_tokens = token_windows[:, :-1].reshape(-1)
    successor = token_windows[:, 1:].reshape(-1)
    candidate_values = successor[candidates]
    query_tokens = flat_tokens[query]
    candidate_tokens = flat_tokens[candidates]
    relevant_mask = F.one_hot(relevant_index, num_classes=candidate_count).to(torch.bool)
    result: dict[str, Tensor | int | float | str] = {
        "query": query,
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
        "hard_negative_candidate_fraction": float(hard_mask.float().mean().item()),
        "queries_with_hard_negatives_fraction": float(hard_mask.any(dim=1).float().mean().item()),
    }
    result["fingerprint"] = tensor_fingerprint(
        query,
        candidates,
        relevant_index,
        target,
        candidate_values,
        hard_mask,
    )
    validate_induction_protocol(token_windows, result)
    return result


def validate_induction_protocol(token_windows: Tensor, protocol: dict[str, Any]) -> None:
    context_size = token_windows.shape[1] - 1
    flat_tokens = token_windows[:, :-1].reshape(-1)
    successor = token_windows[:, 1:].reshape(-1)
    query = protocol["query"].long()
    candidates = protocol["candidates"].long()
    relevant_index = protocol["relevant_index"].long()
    target = protocol["target"].long()
    hard_mask = protocol["hard_negative_mask"].bool()
    if query.ndim != 1 or candidates.ndim != 2 or candidates.shape[0] != query.shape[0]:
        raise ValueError("invalid induction protocol shapes")
    if not bool((query[:, None] // context_size == candidates // context_size).all()):
        raise ValueError("query and candidates must remain in the same frozen context")
    if not bool((candidates % context_size < (query % context_size)[:, None] - 1).all()):
        raise ValueError("candidate values must be strictly in the causal past")
    row = torch.arange(query.shape[0])
    positive = candidates[row, relevant_index]
    if not torch.equal(flat_tokens[positive], flat_tokens[query]):
        raise ValueError("positive key token does not match the query token")
    if not torch.equal(successor[positive], target):
        raise ValueError("positive successor does not match the target")
    candidate_bigram_positive = (flat_tokens[candidates] == flat_tokens[query][:, None]) & (successor[candidates] == target[:, None])
    if not torch.equal(candidate_bigram_positive.sum(dim=1), torch.ones_like(relevant_index)):
        raise ValueError("each candidate set must contain exactly one matching old bigram")
    expected_hard = (flat_tokens[candidates] == flat_tokens[query][:, None]) & (successor[candidates] != target[:, None])
    if not torch.equal(hard_mask, expected_hard):
        raise ValueError("hard-negative mask is inconsistent with token/successor identity")
    if not torch.equal(protocol["candidate_values"].long(), successor[candidates]):
        raise ValueError("candidate value tokens are inconsistent")


def _load_tropical_model(config_path: Path, checkpoint_path: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    from lutflow.cli import _load_config
    from lutflow.infer_checkpoint import _migrate_legacy_lut_layer_keys, _normalize_state_dict_keys, _register_checkpoint_safe_globals
    from lutflow.lightning.training import TrainModule

    _register_checkpoint_safe_globals()
    parsed = _load_config(str(config_path))
    module = TrainModule(parsed["model_config"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = _normalize_state_dict_keys(checkpoint.get("state_dict", checkpoint))
    state = _migrate_legacy_lut_layer_keys(state, set(module.model.state_dict()))
    incompatible = module.model.load_state_dict(state, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    missing = [key for key in incompatible.missing_keys if not key.endswith("._bits_matrix")]
    if unexpected or missing:
        raise RuntimeError(f"checkpoint mismatch: unexpected={unexpected[:8]}, missing={missing[:8]}")
    model = module.model
    if not all(hasattr(model, name) for name in ("embedding", "blocks", "final_norm")):
        raise TypeError("checkpoint is not a supported TropAttnTransformer")
    if hasattr(model.config, "gradient_checkpointing"):
        model.config.gradient_checkpointing = False
    ablation = zero_content_score_luts(model)
    model.eval().requires_grad_(False).to(device)
    return model, {
        "global_step": int(checkpoint.get("global_step", -1)),
        "epoch": int(checkpoint.get("epoch", -1)),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        **ablation,
    }


def _load_hf_tokens(config_path: Path) -> tuple[Tensor, Tensor, dict[str, Any]]:
    from lutflow.cli import _load_config
    from lutflow.lightning.training import TokenDataModule

    parsed = _load_config(str(config_path))
    data = parsed["data"]
    module = TokenDataModule(
        train_path=data["train"],
        val_path=data["val"],
        tokenizer_name=data["tokenizer"],
        context_size=data["context_size"],
        batch_size=1,
        num_workers=0,
        val_split=data["val_split"],
        max_samples=data["max_samples"],
        text_column=data["text_column"],
        token_offset=data["token_offset"],
    )
    train = module._load_data(data["train"], split="train")
    validation = module._load_data(data["val"], split="validation")
    if not isinstance(train, Tensor) or not isinstance(validation, Tensor):
        raise TypeError("induction cache preparation requires materialized token tensors")
    return (
        train,
        validation,
        {
            "train_spec": data["train"],
            "validation_spec": data["val"],
            "tokenizer": data["tokenizer"],
            "train_tokens": train.numel(),
            "official_validation_tokens": validation.numel(),
        },
    )


@torch.no_grad()
def extract_hidden_windows(model: nn.Module, token_windows: Tensor, device: torch.device, batch_size: int) -> Tensor:
    hidden: list[Tensor] = []
    for start in range(0, token_windows.shape[0], batch_size):
        tokens = token_windows[start : start + batch_size, :-1].to(device, non_blocking=True)
        context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.autocast(device_type="cpu", enabled=False)
        )
        with context:
            values = frozen_hidden_forward(model, tokens)
        if not torch.isfinite(values).all():
            raise RuntimeError("non-finite frozen hidden state")
        hidden.append(values.detach().to(device="cpu", dtype=torch.float16))
    return torch.cat(hidden, dim=0)


def prepare_cache(args: argparse.Namespace) -> dict[str, Any]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    request = prepare_request(args)
    metadata_path = args.out_dir / "metadata.json"
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text())
        if existing.get("complete"):
            validate_completed_cache(existing, request, args.out_dir)
            print(json.dumps({"status": "skipped_complete", "metadata": str(metadata_path)}), flush=True)
            return existing

    device = torch.device(args.device)
    train_tokens, validation_tokens, data_meta = _load_hf_tokens(args.config)
    validation_midpoint = validation_tokens.numel() // 2
    regions = {
        "train": (train_tokens, 0, train_tokens.numel(), args.train_windows),
        "validation": (validation_tokens, 0, validation_midpoint, args.validation_windows),
        "test": (validation_tokens, validation_midpoint, validation_tokens.numel(), args.test_windows),
    }
    windows: dict[str, Tensor] = {}
    starts_by_split: dict[str, Tensor] = {}
    for split, (tokens, region_start, region_stop, count) in regions.items():
        starts = choose_window_starts(region_start, region_stop, args.context_size, count)
        starts_by_split[split] = starts
        windows[split] = gather_token_windows(tokens, starts, args.context_size)
    del train_tokens, validation_tokens

    model, model_meta = _load_tropical_model(args.config, args.checkpoint, device)
    hidden: dict[str, Tensor] = {}
    for split in SPLITS:
        hidden[split] = extract_hidden_windows(model, windows[split], device, args.extract_batch_size)
        print(json.dumps({"stage": "hidden", "split": split, "shape": list(hidden[split].shape)}), flush=True)
    train_variance = hidden["train"].float().reshape(-1, hidden["train"].shape[-1]).var(dim=0, unbiased=False)
    coordinate_indices = torch.argsort(train_variance, descending=True, stable=True)[: args.relation_dim].sort().values

    protocols: dict[str, dict[str, Any]] = {}
    limits = {"train": args.max_train_queries, "validation": args.max_validation_queries, "test": args.max_test_queries}
    for split_index, split in enumerate(SPLITS):
        protocols[split] = build_induction_protocol(
            windows[split],
            candidate_count=args.candidates,
            max_hard_negatives=args.hard_negatives,
            max_queries=limits[split],
            seed=DATA_SEED + split_index * 101,
        )
        payload = {
            "tokens": windows[split].to(torch.int32),
            "hidden": hidden[split],
            "protocol": protocols[split],
        }
        atomic_torch_save(payload, args.out_dir / f"{split}.pt")
        print(json.dumps({"stage": "protocol", "split": split, "queries": protocols[split]["queries"]}), flush=True)

    cache_fingerprint = tensor_fingerprint(
        coordinate_indices,
        *[starts_by_split[split] for split in SPLITS],
        *[protocols[split]["query"] for split in SPLITS],
        *[protocols[split]["candidates"] for split in SPLITS],
    )
    metadata: dict[str, Any] = {
        "complete": True,
        "cache_version": CACHE_VERSION,
        "cache_fingerprint": cache_fingerprint,
        "prepare_request": request,
        "created_at_unix": int(time.time()),
        "source": {
            "config": str(args.config),
            "config_sha256": request["config_sha256"],
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": request["checkpoint_sha256"],
            "checkpoint_metadata": model_meta,
            "frozen_state": "final pre-unembedding z from a content-scoreless ablation; all score_lut tensors are exactly zero",
            "boundary": (
                "the source checkpoint was trained with AbsDiff content before this frozen score ablation; "
                "it is not a from-scratch scoreless checkpoint"
            ),
        },
        "data": {
            **data_meta,
            "split_policy": "official train; official validation split into disjoint first-half validation and second-half test regions",
            "context_size": args.context_size,
            "window_starts": {split: starts_by_split[split].tolist() for split in SPLITS},
            "window_counts": {split: int(windows[split].shape[0]) for split in SPLITS},
            "token_fingerprints": {split: tensor_fingerprint(windows[split]) for split in SPLITS},
        },
        "protocol": {
            "data_seed": DATA_SEED,
            "candidate_count": args.candidates,
            "hard_negative_priority": args.hard_negatives,
            "definition": "positive j satisfies j < i-1, x_j=x_i, and x_(j+1)=x_(i+1); exactly one positive is included",
            "target_leakage": "x_(i+1) constructs labels only and is never part of q or k",
            "query_counts": {split: int(protocols[split]["queries"]) for split in SPLITS},
            "candidate_fingerprints": {split: protocols[split]["fingerprint"] for split in SPLITS},
            "hard_negative_candidate_fraction": {split: protocols[split]["hard_negative_candidate_fraction"] for split in SPLITS},
            "queries_with_hard_negatives_fraction": {split: protocols[split]["queries_with_hard_negatives_fraction"] for split in SPLITS},
        },
        "features": {
            "hidden_dim": int(hidden["train"].shape[-1]),
            "relation_dim": args.relation_dim,
            "coordinate_policy": "top train-variance raw coordinates; target-free; no learned projection or whitening",
            "coordinate_indices": coordinate_indices.tolist(),
            "coordinate_variances": train_variance[coordinate_indices].tolist(),
        },
    }
    atomic_json_write(metadata, metadata_path)
    print(json.dumps({"status": "complete", "metadata": str(metadata_path), "fingerprint": cache_fingerprint}), flush=True)
    return metadata


class DenseQK(nn.Module):
    def __init__(self, dimension: int, rank: int, seed: int) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(seed + 503)
        self.query = nn.Parameter(torch.randn(dimension, rank, generator=generator) / math.sqrt(dimension))
        self.key = nn.Parameter(torch.randn(dimension, rank, generator=generator) / math.sqrt(dimension))
        self.bias = nn.Parameter(torch.zeros(()))
        self.rank = int(rank)

    def forward(self, query: Tensor, key: Tensor) -> Tensor:
        return ((query @ self.query) * (key @ self.key)).sum(dim=-1) / math.sqrt(self.rank) + self.bias


def build_scorer(decoder: str, seed: int, dimension: int = 32) -> tuple[nn.Module, dict[str, Any]]:
    if decoder not in DECODERS:
        raise ValueError(f"unsupported decoder {decoder!r}")
    if decoder == "dense_qk":
        scorer = DenseQK(dimension, 16, seed)
        return scorer, {"execution_class": "dense QK diagnostic", "rank": 16}
    router = BalancedS4Router(dimension, 16, coverage=2, seed=seed)
    if decoder == "kendall":
        kernel = IntrinsicS4Kernel(16, "kendall")
    elif decoder == "same_table_full":
        kernel = SameTableFullKernel(16, seed=seed)
    else:
        kernel = RootIncidenceKernel(router, seed=seed)
    scorer = CoxeterPairScorer(router, kernel, symmetry="none")
    metadata = {
        "execution_class": {
            "kendall": "fixed intrinsic S4 similarity",
            "same_table_full": "block-diagonal same-chart relation LUT",
            "root_incidence": "global comparison-root incidence operator",
        }[decoder],
        "chart_anchors": router.anchors.tolist(),
        "root_edges": router.roots,
        "root_incidence_entries": router.incidence_entries,
    }
    return scorer, metadata


def score_candidate_groups(scorer: nn.Module, query: Tensor, candidates: Tensor) -> Tensor:
    batch, candidate_count, dimension = candidates.shape
    query_flat = query[:, None, :].expand(-1, candidate_count, -1).reshape(batch * candidate_count, dimension)
    key_flat = candidates.reshape(batch * candidate_count, dimension)
    return scorer(query_flat, key_flat).view(batch, candidate_count)


def ranking_metrics(
    scores: Tensor,
    relevant_index: Tensor,
    candidate_values: Tensor,
    target: Tensor,
    hard_negative_mask: Tensor,
) -> dict[str, float]:
    order = torch.argsort(scores, dim=1, descending=True, stable=True)
    rank = (order == relevant_index[:, None]).nonzero(as_tuple=False)[:, 1] + 1
    top1 = order[:, 0]
    top4 = order[:, : min(4, order.shape[1])]
    row = torch.arange(scores.shape[0], device=scores.device)
    negative_scores = scores.masked_fill(F.one_hot(relevant_index, scores.shape[1]).bool(), float("-inf"))
    return {
        "recall_at_1": float((rank == 1).float().mean().item()),
        "recall_at_4": float((rank <= 4).float().mean().item()),
        "mrr": float(rank.float().reciprocal().mean().item()),
        "successor_hit_at_1": float((candidate_values[row, top1] == target).float().mean().item()),
        "successor_hit_at_4": float((candidate_values.gather(1, top4) == target[:, None]).any(dim=1).float().mean().item()),
        "hard_negative_top1_rate": float(hard_negative_mask[row, top1].float().mean().item()),
        "positive_margin": float((scores[row, relevant_index] - negative_scores.max(dim=1).values).mean().item()),
        "listwise_nll": float(F.cross_entropy(scores, relevant_index).item()),
    }


def load_cached_split(cache_dir: Path, split: str, coordinate_indices: Tensor) -> dict[str, Tensor]:
    payload = torch.load(cache_dir / f"{split}.pt", map_location="cpu", weights_only=True)
    validate_induction_protocol(payload["tokens"].long(), payload["protocol"])
    hidden = payload["hidden"].float().reshape(-1, payload["hidden"].shape[-1])[:, coordinate_indices].contiguous()
    protocol = payload["protocol"]
    return {
        "coordinates": hidden,
        "query": protocol["query"].long(),
        "candidates": protocol["candidates"].long(),
        "relevant_index": protocol["relevant_index"].long(),
        "target": protocol["target"].long(),
        "candidate_values": protocol["candidate_values"].long(),
        "hard_negative_mask": protocol["hard_negative_mask"].bool(),
    }


def iter_batches(total: int, batch_size: int, *, shuffle: bool, seed: int) -> list[Tensor]:
    if shuffle:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        order = torch.randperm(total, generator=generator)
    else:
        order = torch.arange(total)
    return list(order.split(batch_size))


def gather_group_batch(split: dict[str, Tensor], indices: Tensor, device: torch.device) -> tuple[Tensor, ...]:
    query_indices = split["query"][indices]
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
def evaluate_scorer(scorer: nn.Module, split: dict[str, Tensor], device: torch.device, batch_size: int) -> dict[str, float]:
    scorer.eval()
    score_rows: list[Tensor] = []
    relevant_rows: list[Tensor] = []
    value_rows: list[Tensor] = []
    target_rows: list[Tensor] = []
    hard_rows: list[Tensor] = []
    for indices in iter_batches(split["query"].numel(), batch_size, shuffle=False, seed=0):
        query, candidates, relevant, values, target, hard = gather_group_batch(split, indices, device)
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


def validate_completed_run(existing: dict[str, Any], config: RunConfig, path: Path) -> None:
    if existing.get("config") != asdict(config):
        raise ValueError(f"completed result config mismatch at {path}")


def run_decoder(args: argparse.Namespace) -> dict[str, Any]:
    metadata = json.loads((args.cache_dir / "metadata.json").read_text())
    if not metadata.get("complete") or metadata.get("cache_version") != CACHE_VERSION:
        raise ValueError("frozen cache is incomplete or incompatible")
    config = RunConfig(
        decoder=args.decoder,
        seed=args.seed,
        cache_fingerprint=metadata["cache_fingerprint"],
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.out_dir / "result.json"
    if result_path.exists():
        existing = json.loads(result_path.read_text())
        if existing.get("complete"):
            validate_completed_run(existing, config, result_path)
            print(json.dumps({"status": "skipped_complete", "result": str(result_path)}), flush=True)
            return existing
    for incomplete in (args.out_dir / "history.csv", args.out_dir / "best.pt"):
        if incomplete.exists():
            incomplete.rename(incomplete.with_name(f"{incomplete.stem}.incomplete-{int(time.time())}{incomplete.suffix}"))

    seed_everything(args.seed)
    device = torch.device(args.device)
    coordinates = torch.tensor(metadata["features"]["coordinate_indices"], dtype=torch.long)
    splits = {split: load_cached_split(args.cache_dir, split, coordinates) for split in SPLITS}
    scorer, scorer_metadata = build_scorer(args.decoder, args.seed, coordinates.numel())
    scorer.to(device)
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
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
        for indices in iter_batches(splits["train"]["query"].numel(), args.batch_size, shuffle=True, seed=args.seed + 1009 * epoch):
            query, candidates, relevant, _, _, _ = gather_group_batch(splits["train"], indices, device)
            scores = score_candidate_groups(scorer, query, candidates)
            loss = F.cross_entropy(scores, relevant)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(scorer.parameters(), args.gradient_clip)
            if not torch.isfinite(gradient_norm):
                raise RuntimeError("non-finite relation gradient")
            optimizer.step()
            optimizer_steps += 1
            loss_sum += float(loss.item()) * indices.numel()
            seen += indices.numel()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        training_seconds += time.perf_counter() - train_started
        validation = evaluate_scorer(scorer, splits["validation"], device, args.eval_batch_size)
        row: dict[str, object] = {
            "epoch": epoch,
            "train_listwise_nll": loss_sum / max(1, seen),
            **{f"validation_{key}": value for key, value in validation.items()},
        }
        history.append(row)
        write_csv(args.out_dir / "history.csv", history)
        if validation["recall_at_1"] > best_recall:
            best_recall = validation["recall_at_1"]
            best_epoch = epoch
            atomic_torch_save(
                {
                    "config": asdict(config),
                    "epoch": epoch,
                    "selection_metric": best_recall,
                    "state_dict": {key: value.detach().cpu() for key, value in scorer.state_dict().items()},
                },
                checkpoint_path,
            )
        print(json.dumps(row, sort_keys=True), flush=True)
    if best_epoch == 0:
        raise RuntimeError("no finite validation checkpoint was produced")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    scorer.load_state_dict(checkpoint["state_dict"])
    scorer.to(device).eval()
    validation = evaluate_scorer(scorer, splits["validation"], device, args.eval_batch_size)
    test = evaluate_scorer(scorer, splits["test"], device, args.eval_batch_size)
    result: dict[str, Any] = {
        "complete": True,
        "config": asdict(config),
        "best_epoch": best_epoch,
        "best_validation_recall_at_1": best_recall,
        "relation_parameters": sum(parameter.numel() for parameter in scorer.parameters()),
        "scorer": scorer_metadata,
        "train_queries": int(splits["train"]["query"].numel()),
        "validation_queries": int(splits["validation"]["query"].numel()),
        "test_queries": int(splits["test"]["query"].numel()),
        "candidate_count": int(splits["train"]["candidates"].shape[1]),
        "validation": validation,
        "test": test,
        "optimizer_steps": optimizer_steps,
        "training_seconds": training_seconds,
        "train_queries_per_second": args.epochs * splits["train"]["query"].numel() / max(training_seconds, 1e-30),
        "elapsed_seconds": time.perf_counter() - started,
        "frozen_source": metadata["source"],
        "protocol": metadata["protocol"],
        "features": metadata["features"],
    }
    atomic_json_write(result, result_path)
    print(json.dumps({"status": "complete", "result": str(result_path), "best_epoch": best_epoch}), flush=True)
    return result


def mean_sem(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    results = [json.loads(path.read_text()) for path in sorted(args.result_dir.glob("**/result.json"))]
    results = [result for result in results if result.get("complete")]
    indexed: dict[str, dict[int, dict[str, Any]]] = {decoder: {} for decoder in DECODERS}
    for result in results:
        decoder = result["config"]["decoder"]
        seed = int(result["config"]["seed"])
        if decoder not in indexed or seed in indexed[decoder]:
            raise ValueError(f"duplicate or unknown result decoder={decoder}, seed={seed}")
        indexed[decoder][seed] = result
    complete = all(sorted(indexed[decoder]) == [0, 1, 2] for decoder in DECODERS) and len(results) == 12

    aggregate: list[dict[str, Any]] = []
    for decoder in DECODERS:
        group = [indexed[decoder][seed] for seed in sorted(indexed[decoder])]
        if not group:
            continue
        row: dict[str, Any] = {
            "decoder": decoder,
            "seeds": len(group),
            "relation_parameters_min": min(int(result["relation_parameters"]) for result in group),
            "relation_parameters_max": max(int(result["relation_parameters"]) for result in group),
        }
        for split in ("validation", "test"):
            for metric in (
                "recall_at_1",
                "recall_at_4",
                "mrr",
                "successor_hit_at_1",
                "successor_hit_at_4",
                "hard_negative_top1_rate",
                "positive_margin",
                "listwise_nll",
            ):
                values = [float(result[split][metric]) for result in group]
                row[f"{split}_{metric}_mean"], row[f"{split}_{metric}_sem"] = mean_sem(values)
        aggregate.append(row)

    root_vs_kendall: dict[str, Any]
    dense_retention: dict[str, Any]
    if complete:
        root = [float(indexed["root_incidence"][seed]["test"]["recall_at_1"]) for seed in (0, 1, 2)]
        kendall = [float(indexed["kendall"][seed]["test"]["recall_at_1"]) for seed in (0, 1, 2)]
        dense = [float(indexed["dense_qk"][seed]["test"]["recall_at_1"]) for seed in (0, 1, 2)]
        deltas = [left - right for left, right in zip(root, kendall)]
        delta_mean, delta_sem = mean_sem(deltas)
        root_vs_kendall = {
            "complete": True,
            "passed": delta_mean > 0.02,
            "paired_deltas": deltas,
            "mean_delta": delta_mean,
            "sem": delta_sem,
            "threshold": 0.02,
        }
        root_gain = statistics.mean(root) - statistics.mean(kendall)
        dense_gain = statistics.mean(dense) - statistics.mean(kendall)
        retention = root_gain / dense_gain if dense_gain > 0.0 else float("nan")
        dense_retention = {
            "complete": True,
            "passed": math.isfinite(retention) and retention >= 0.8,
            "root_gain_over_kendall": root_gain,
            "dense_gain_over_kendall": dense_gain,
            "retention": retention,
            "threshold": 0.8,
        }
    else:
        root_vs_kendall = {"complete": False, "passed": False, "reason": "missing formal runs"}
        dense_retention = {"complete": False, "passed": False, "reason": "missing formal runs"}
    passed = complete and bool(root_vs_kendall["passed"]) and bool(dense_retention["passed"])
    decision = {
        "complete": complete,
        "complete_runs": len(results),
        "expected_runs": 12,
        "root_vs_kendall": root_vs_kendall,
        "root_dense_gain_retention": dense_retention,
        "semantic_gate_passed": passed,
        "next_stage": "online_warm_start" if passed else "stop_before_online_integration",
    }
    atomic_json_write(decision, args.result_dir / "decision.json")
    if aggregate:
        write_csv(args.result_dir / "summary.csv", aggregate)

    metadata = json.loads((args.cache_dir / "metadata.json").read_text())
    lines = [
        "# Frozen Wiki103 Induction Retrieval in Ordinal Space",
        "",
        "## Protocol",
        "",
        f"The benchmark contains `{len(results)}` complete decoder runs over a shared frozen cache. Query `z_i` and causal key "
        "`z_j` come from the final pre-unembedding state of the local 4k-step WikiText-103 FlashTropical checkpoint after all "
        "content `score_lut` tensors are set exactly to zero. TroPE, ValueLUT, residual blocks, FFNs, and readout-side training "
        "history remain fixed.",
        "",
        f"Boundary: {metadata['source']['boundary']}",
        "",
        "Each query has 32 model-independent candidates and exactly one old matching bigram. Candidates satisfy `j < i-1`; "
        "the old successor `x_(j+1)` is therefore already in the causal past. Same-token/wrong-successor hard negatives are "
        "sampled before random negatives. The future target constructs labels only and is not part of either row vector.",
        "",
        "The four decoders receive the same 32 target-free high-variance raw hidden coordinates. No learned projection or "
        "whitening precedes the ordinal routes. Dense QK is a diagnostic, not a GEMM-free proposal.",
        "",
        "## Three-seed results",
        "",
        "| Decoder | Relation params | Test R@1 | Test R@4 | MRR | Successor hit@1 | Hard-negative top1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        low = int(row["relation_parameters_min"])
        high = int(row["relation_parameters_max"])
        params = f"{low:,}" if low == high else f"{low:,}-{high:,}"
        lines.append(
            f"| {row['decoder']} | {params} | {row['test_recall_at_1_mean']:.4f} ± {row['test_recall_at_1_sem']:.4f} | "
            f"{row['test_recall_at_4_mean']:.4f} ± {row['test_recall_at_4_sem']:.4f} | "
            f"{row['test_mrr_mean']:.4f} | {row['test_successor_hit_at_1_mean']:.4f} | "
            f"{row['test_hard_negative_top1_rate_mean']:.4f} |"
        )
    lines += [
        "",
        "## Preregistered decision",
        "",
        f"- Complete 4 x 3 matrix: `{'YES' if complete else 'NO'}`.",
        f"- Root-incidence exceeds Kendall by more than 0.02 R@1: `{'PASS' if root_vs_kendall.get('passed') else 'FAIL'}`.",
        f"- Root-incidence retains at least 80% of Dense-QK gain over Kendall: `{'PASS' if dense_retention.get('passed') else 'FAIL'}`.",
        f"- Next stage: `{decision['next_stage']}`.",
        "",
    ]
    if complete:
        lines += [
            f"Root-incidence minus Kendall test R@1 is `{root_vs_kendall['mean_delta']:.4f}` with paired SEM "
            f"`{root_vs_kendall['sem']:.4f}`. Its gain-retention ratio relative to Dense QK is "
            f"`{dense_retention['retention']:.4f}`.",
            "",
        ]
    lines += [
        "## Interpretation boundary",
        "",
        "This is a frozen-state relation-selection test, not an online language-model result. Passing would justify a "
        "supervised scorer warm start; failing localizes the problem before online credit assignment. The cache boundary also "
        "prevents claiming that the source model was trained scoreless from initialization.",
        "",
    ]
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text("\n".join(lines))
    print(json.dumps(decision, indent=2, sort_keys=True), flush=True)
    return decision


def check_cache(args: argparse.Namespace) -> dict[str, Any]:
    metadata = json.loads((args.cache_dir / "metadata.json").read_text())
    counts: dict[str, int] = {}
    for split in SPLITS:
        payload = torch.load(args.cache_dir / f"{split}.pt", map_location="cpu", weights_only=True)
        validate_induction_protocol(payload["tokens"].long(), payload["protocol"])
        counts[split] = int(payload["protocol"]["queries"])
    result = {
        "complete": bool(metadata.get("complete")),
        "cache_version": metadata.get("cache_version"),
        "cache_fingerprint": metadata.get("cache_fingerprint"),
        "query_counts": counts,
        "score_lut_l1_after": metadata["source"]["checkpoint_metadata"]["score_lut_l1_after"],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Frozen Wiki103 induction retrieval over ordinal pair kernels")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare")
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--checkpoint", type=Path, required=True)
    prepare.add_argument("--out-dir", type=Path, required=True)
    prepare.add_argument("--device", default="cuda")
    prepare.add_argument("--context-size", type=int, default=512)
    prepare.add_argument("--train-windows", type=int, default=128)
    prepare.add_argument("--validation-windows", type=int, default=32)
    prepare.add_argument("--test-windows", type=int, default=32)
    prepare.add_argument("--extract-batch-size", type=int, default=2)
    prepare.add_argument("--relation-dim", type=int, default=32)
    prepare.add_argument("--candidates", type=int, default=32)
    prepare.add_argument("--hard-negatives", type=int, default=16)
    prepare.add_argument("--max-train-queries", type=int, default=20000)
    prepare.add_argument("--max-validation-queries", type=int, default=4000)
    prepare.add_argument("--max-test-queries", type=int, default=4000)
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
