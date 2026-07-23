"""Role-aligned comparison-root kernels on causally aligned MQAR.

This experiment keeps the fixed causal B-row construction from
``causal_mqar_role_gauge`` but replaces arbitrary Global embedding width with
the natural A_{D-1} comparison-root carrier.  A signed transport aligns K-role
roots to Q-role roots:

    score(q, k) = |E|^{-1/2} sum_e c_Q(q)[e] s_e c_K(k)[pi(e)]

``oracle`` uses the exact transport induced by the known coordinate gauges.
``learned`` fits one signed K-root address per active Q root from retrieval
cross-entropy.  Its dense categorical relaxation is training-only; hard
inference stores one signed integer address per active root.  Optional small
cross-root LUTs provide a deliberately nonseparable residual.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tropnn.tools.causal_mqar_induction import (
    EVALUATION_SPECS,
    FixedOrdinalCodebook,
    MQARBatch,
    _clone_state_dict,
    _training_shape,
    atomic_json_write,
    atomic_torch_save,
    seed_everything,
    target_token_loss,
    token_permutation,
)
from tropnn.tools.causal_mqar_role_gauge import (
    coordinate_permutation,
    generate_pool_mqar_batch,
    token_pools,
)

TRANSPORT_MODES = ("oracle", "learned")
EVALUATION_CONDITIONS = ("base", "token_relabel", "wrong_key_gauge", "role_swap")
ROOT_BUDGETS = (32, 64, 96, 192, 496)


@dataclass(frozen=True)
class RootTransportRunConfig:
    mode: str
    seed: int
    steps: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    gradient_clip: float
    validation_interval: int
    validation_episodes: int
    evaluation_episodes: int
    evaluation_batch_size: int
    vocab_size: int
    train_tokens: int
    validation_tokens: int
    test_tokens: int
    relation_dim: int
    root_budget: int
    root_subset_seed: int
    residual_tables: int
    residual_roots_per_side: int
    assignment_temperature_start: float
    assignment_temperature_end: float
    assignment_entropy_weight: float
    data_seed: int
    codebook_seed: int
    query_gauge_seed: int
    key_gauge_seed: int
    wrong_key_gauge_seed: int
    token_relabel_seed: int
    device: str


def full_root_edges(dimension: int) -> Tensor:
    """Return lexicographically ordered positive A_{D-1} roots."""

    return torch.triu_indices(dimension, dimension, offset=1).T.contiguous()


def nested_root_subset(root_count: int, budget: int, seed: int) -> Tensor:
    if not 1 <= budget <= root_count:
        raise ValueError(f"root budget must lie in [1, {root_count}]")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randperm(root_count, generator=generator)[:budget]


def root_signs(coordinates: Tensor, edges: Tensor) -> Tensor:
    """Return exact {-1,+1} comparison signs."""

    return torch.where(
        coordinates[..., edges[:, 0]] > coordinates[..., edges[:, 1]],
        1.0,
        -1.0,
    )


def oracle_signed_root_transport(
    query_permutation: Tensor,
    key_permutation: Tensor,
    edges: Tensor,
) -> tuple[Tensor, Tensor]:
    """Map every Q-role root to the equivalent signed K-role root."""

    dimension = int(query_permutation.numel())
    if key_permutation.shape != (dimension,):
        raise ValueError("query and key permutations must have equal shape")
    expected = torch.arange(dimension)
    if not torch.equal(query_permutation.sort().values.cpu(), expected):
        raise ValueError("query_permutation must be a bijection")
    if not torch.equal(key_permutation.sort().values.cpu(), expected):
        raise ValueError("key_permutation must be a bijection")
    edge_lookup = {(int(left), int(right)): index for index, (left, right) in enumerate(edges.detach().cpu().tolist())}
    inverse_key = torch.argsort(key_permutation.cpu())
    key_indices: list[int] = []
    orientations: list[float] = []
    for left, right in edges.detach().cpu().tolist():
        key_left = int(inverse_key[int(query_permutation[left])])
        key_right = int(inverse_key[int(query_permutation[right])])
        if key_left < key_right:
            key_indices.append(edge_lookup[(key_left, key_right)])
            orientations.append(1.0)
        else:
            key_indices.append(edge_lookup[(key_right, key_left)])
            orientations.append(-1.0)
    return torch.tensor(key_indices, dtype=torch.long), torch.tensor(orientations)


class CrossRootResidualLUT(nn.Module):
    """Small 3+3-bit joint LUTs over Q and K root signs."""

    def __init__(
        self,
        root_count: int,
        tables: int,
        roots_per_side: int,
        *,
        seed: int,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        if tables < 0:
            raise ValueError("residual table count must be nonnegative")
        if roots_per_side < 1:
            raise ValueError("roots_per_side must be positive")
        self.tables = int(tables)
        self.roots_per_side = int(roots_per_side)
        bits = 2 * roots_per_side
        self.table_size = 1 << bits
        generator = torch.Generator(device="cpu").manual_seed(seed + 811)
        if tables:
            query_roots = torch.stack([torch.randperm(root_count, generator=generator)[:roots_per_side] for _ in range(tables)])
            key_roots = torch.stack([torch.randperm(root_count, generator=generator)[:roots_per_side] for _ in range(tables)])
        else:
            query_roots = torch.empty(0, roots_per_side, dtype=torch.long)
            key_roots = torch.empty(0, roots_per_side, dtype=torch.long)
        self.register_buffer("query_roots", query_roots)
        self.register_buffer("key_roots", key_roots)
        self.register_buffer("powers", 2 ** torch.arange(bits, dtype=torch.long))
        self.weight = nn.Parameter(torch.randn(tables, self.table_size, generator=generator) * init_std)

    def forward(self, query_roots: Tensor, key_roots: Tensor) -> Tensor:
        output_shape = torch.broadcast_shapes(query_roots.shape[:-1], key_roots.shape[:-1])
        if self.tables == 0:
            return torch.zeros(output_shape, device=query_roots.device, dtype=query_roots.dtype)
        query = query_roots[..., self.query_roots].expand(*output_shape, self.tables, self.roots_per_side)
        key = key_roots[..., self.key_roots].expand(*output_shape, self.tables, self.roots_per_side)
        bits = torch.cat((query > 0, key > 0), dim=-1)
        indices = (bits.long() * self.powers).sum(dim=-1)
        tables = torch.arange(self.tables, device=indices.device)
        values = self.weight[tables, indices]
        return values.sum(dim=-1) / math.sqrt(self.tables)


class SignedRootTransportKernel(nn.Module):
    """Oracle or learned sparse signed root correspondence."""

    def __init__(
        self,
        *,
        dimension: int,
        root_budget: int,
        root_subset_seed: int,
        query_permutation: Tensor,
        key_permutation: Tensor,
        mode: str,
        residual_tables: int,
        residual_roots_per_side: int,
        seed: int,
    ) -> None:
        super().__init__()
        if mode not in TRANSPORT_MODES:
            raise ValueError(f"unsupported transport mode {mode!r}")
        edges = full_root_edges(dimension)
        root_count = int(edges.shape[0])
        subset = nested_root_subset(root_count, root_budget, root_subset_seed)
        oracle_index, oracle_orientation = oracle_signed_root_transport(
            query_permutation,
            key_permutation,
            edges,
        )
        oracle_index = oracle_index[subset]
        oracle_orientation = oracle_orientation[subset]
        oracle_signed_index = oracle_index + (oracle_orientation > 0).long() * root_count
        self.mode = mode
        self.dimension = int(dimension)
        self.root_count = root_count
        self.root_budget = int(root_budget)
        self.temperature = 1.0
        self.register_buffer("edges", edges)
        self.register_buffer("query_root_subset", subset)
        self.register_buffer("oracle_key_root_index", oracle_index)
        self.register_buffer("oracle_orientation", oracle_orientation)
        self.register_buffer("oracle_signed_index", oracle_signed_index)
        if mode == "learned":
            generator = torch.Generator(device="cpu").manual_seed(seed + 733)
            logits = torch.randn(root_budget, 2 * root_count, generator=generator) * 0.001
            self.assignment_logits = nn.Parameter(logits)
        else:
            self.register_parameter("assignment_logits", None)
        if mode == "learned" or residual_tables > 0:
            self.logit_scale = nn.Parameter(torch.zeros(()))
        else:
            self.register_buffer("logit_scale", torch.zeros(()))
        self.residual = CrossRootResidualLUT(
            root_count,
            residual_tables,
            residual_roots_per_side,
            seed=seed,
        )

    def assignment_probabilities(self) -> Tensor:
        if self.assignment_logits is None:
            signed = F.one_hot(self.oracle_signed_index, num_classes=2 * self.root_count)
            return signed.to(dtype=torch.float32)
        return (self.assignment_logits / self.temperature).softmax(dim=-1)

    def hard_signed_indices(self) -> Tensor:
        if self.assignment_logits is None:
            return self.oracle_signed_index
        return self.assignment_logits.argmax(dim=-1)

    def correspondence_accuracy(self) -> float:
        return float((self.hard_signed_indices() == self.oracle_signed_index).float().mean().item())

    def hard_transport(self, key_roots: Tensor) -> Tensor:
        signed_key = torch.cat((-key_roots, key_roots), dim=-1)
        return signed_key[..., self.hard_signed_indices()]

    def soft_transport(self, key_roots: Tensor) -> Tensor:
        signed_key = torch.cat((-key_roots, key_roots), dim=-1)
        probabilities = self.assignment_probabilities()
        return torch.einsum("...r,br->...b", signed_key, probabilities)

    def transport(self, key_roots: Tensor) -> Tensor:
        if self.mode == "oracle" or not self.training:
            return self.hard_transport(key_roots)
        return self.soft_transport(key_roots)

    def forward(self, query_roots: Tensor, key_roots: Tensor) -> Tensor:
        query_active = query_roots[..., self.query_root_subset]
        aligned_key = self.transport(key_roots)
        base = (query_active * aligned_key).sum(dim=-1) / math.sqrt(self.root_budget)
        scale = self.logit_scale.exp().clamp(max=100.0)
        return scale * base + self.residual(query_roots, key_roots)

    def entropy(self) -> Tensor:
        if self.assignment_logits is None:
            return torch.zeros((), device=self.logit_scale.device)
        probabilities = self.assignment_probabilities()
        return -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1).mean()

    @property
    def hard_inference_bytes(self) -> int:
        index_bytes = 2 if 2 * self.root_count <= 65536 else 4
        return self.root_budget * index_bytes


class RootTransportMQARRetriever(nn.Module):
    """Causal MQAR with fixed role gauges and a comparison-root kernel."""

    def __init__(self, config: RootTransportRunConfig) -> None:
        super().__init__()
        self.config = config
        self.codebook = FixedOrdinalCodebook(
            config.vocab_size,
            config.relation_dim,
            config.codebook_seed,
        )
        query = coordinate_permutation(config.relation_dim, config.query_gauge_seed)
        key = coordinate_permutation(config.relation_dim, config.key_gauge_seed)
        wrong_key = coordinate_permutation(config.relation_dim, config.wrong_key_gauge_seed)
        if torch.equal(query, key):
            raise ValueError("query and key gauges must differ")
        if torch.equal(wrong_key, key) or torch.equal(wrong_key, query):
            wrong_key = wrong_key.roll(1)
        self.register_buffer("query_permutation", query)
        self.register_buffer("key_permutation", key)
        self.register_buffer("wrong_key_permutation", wrong_key)
        self.relation = SignedRootTransportKernel(
            dimension=config.relation_dim,
            root_budget=config.root_budget,
            root_subset_seed=config.root_subset_seed,
            query_permutation=query,
            key_permutation=key,
            mode=config.mode,
            residual_tables=config.residual_tables,
            residual_roots_per_side=config.residual_roots_per_side,
            seed=config.seed,
        )

    def _root_features(self, token_ids: Tensor, permutation: Tensor) -> Tensor:
        coordinates = self.codebook(token_ids)[..., permutation]
        return root_signs(coordinates, self.relation.edges)

    def forward(
        self,
        batch: MQARBatch,
        *,
        query_permutation: Tensor | None = None,
        key_permutation: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        query_gauge = self.query_permutation if query_permutation is None else query_permutation
        key_gauge = self.key_permutation if key_permutation is None else key_permutation
        query_roots = self._root_features(batch.queries, query_gauge)
        key_roots = self._root_features(batch.keys, key_gauge)
        scores = self.relation(
            query_roots[:, :, None, :],
            key_roots[:, None, :, :],
        )
        return scores, batch.values

    @property
    def trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


def _condition_permutations(model: RootTransportMQARRetriever, condition: str) -> tuple[Tensor, Tensor]:
    if condition in {"base", "token_relabel"}:
        return model.query_permutation, model.key_permutation
    if condition == "wrong_key_gauge":
        return model.query_permutation, model.wrong_key_permutation
    if condition == "role_swap":
        return model.key_permutation, model.query_permutation
    raise ValueError(f"unsupported condition {condition!r}")


@torch.no_grad()
def evaluate(
    model: RootTransportMQARRetriever,
    *,
    token_pool: Tensor,
    pair_count: int,
    query_count: int,
    episodes: int,
    batch_size: int,
    data_seed: int,
    device: torch.device,
    condition: str,
    pool_permutation: Tensor | None = None,
) -> dict[str, float | int]:
    model.eval()
    query_gauge, key_gauge = _condition_permutations(model, condition)
    total_queries = 0
    correct_queries = 0
    exact_episodes = 0
    nll_sum = 0.0
    for offset in range(0, episodes, batch_size):
        current_batch_size = min(batch_size, episodes - offset)
        batch = generate_pool_mqar_batch(
            token_pool=token_pool,
            batch_size=current_batch_size,
            pair_count=pair_count,
            query_count=query_count,
            seed=data_seed + offset * 1009,
            pool_permutation=pool_permutation,
        ).to(device)
        scores, values = model(
            batch,
            query_permutation=query_gauge,
            key_permutation=key_gauge,
        )
        selected = scores.argmax(dim=-1)
        predicted = (
            values[:, None, :]
            .expand(-1, query_count, -1)
            .gather(
                2,
                selected.unsqueeze(-1),
            )
            .squeeze(-1)
        )
        correct = predicted == batch.targets
        log_probability = (
            scores.log_softmax(dim=-1)
            .gather(
                2,
                batch.query_indices.unsqueeze(-1),
            )
            .squeeze(-1)
        )
        total_queries += current_batch_size * query_count
        correct_queries += int(correct.sum().item())
        exact_episodes += int(correct.all(dim=1).sum().item())
        nll_sum -= float(log_probability.sum().item())
    return {
        "episodes": episodes,
        "queries": total_queries,
        "pair_count": pair_count,
        "query_count": query_count,
        "target_token_accuracy": correct_queries / total_queries,
        "target_token_ce": nll_sum / total_queries,
        "multiquery_exact_accuracy": exact_episodes / episodes,
        "random_token_accuracy": 1.0 / pair_count,
    }


def _temperature(config: RootTransportRunConfig, step: int) -> float:
    if config.steps <= 1:
        return config.assignment_temperature_end
    fraction = (step - 1) / (config.steps - 1)
    ratio = config.assignment_temperature_end / config.assignment_temperature_start
    return config.assignment_temperature_start * ratio**fraction


def _validate_config(config: RootTransportRunConfig) -> None:
    if config.mode not in TRANSPORT_MODES:
        raise ValueError(f"unsupported mode {config.mode!r}")
    root_count = config.relation_dim * (config.relation_dim - 1) // 2
    if not 1 <= config.root_budget <= root_count:
        raise ValueError(f"root budget must lie in [1, {root_count}]")
    if config.vocab_size != config.train_tokens + config.validation_tokens + config.test_tokens:
        raise ValueError("token counts must exactly partition the vocabulary")
    if min(config.train_tokens, config.validation_tokens, config.test_tokens) < 64:
        raise ValueError("each token pool must contain at least 64 identities")
    if config.mode == "learned" and config.steps < 1:
        raise ValueError("learned transport requires positive training steps")
    if config.mode == "oracle" and config.residual_tables > 0 and config.steps < 1:
        raise ValueError("trainable residual tables require positive training steps")
    if config.assignment_temperature_start <= 0 or config.assignment_temperature_end <= 0:
        raise ValueError("assignment temperatures must be positive")
    if config.residual_tables < 0:
        raise ValueError("residual table count must be nonnegative")


def _build_config(args: argparse.Namespace) -> RootTransportRunConfig:
    return RootTransportRunConfig(
        mode=args.mode,
        seed=args.seed,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        validation_interval=args.validation_interval,
        validation_episodes=args.validation_episodes,
        evaluation_episodes=args.evaluation_episodes,
        evaluation_batch_size=args.evaluation_batch_size,
        vocab_size=args.vocab_size,
        train_tokens=args.train_tokens,
        validation_tokens=args.validation_tokens,
        test_tokens=args.test_tokens,
        relation_dim=args.relation_dim,
        root_budget=args.root_budget,
        root_subset_seed=args.root_subset_seed,
        residual_tables=args.residual_tables,
        residual_roots_per_side=args.residual_roots_per_side,
        assignment_temperature_start=args.assignment_temperature_start,
        assignment_temperature_end=args.assignment_temperature_end,
        assignment_entropy_weight=args.assignment_entropy_weight,
        data_seed=args.data_seed,
        codebook_seed=args.codebook_seed,
        query_gauge_seed=args.query_gauge_seed,
        key_gauge_seed=args.key_gauge_seed,
        wrong_key_gauge_seed=args.wrong_key_gauge_seed,
        token_relabel_seed=args.token_relabel_seed,
        device=args.device,
    )


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    config = _build_config(args)
    _validate_config(config)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.out_dir / "result.json"
    if result_path.exists():
        existing = json.loads(result_path.read_text())
        if existing.get("complete") and existing.get("config") == asdict(config):
            print(json.dumps({"status": "skipped_complete", "result": str(result_path)}), flush=True)
            return existing
        if existing.get("complete"):
            raise ValueError(f"completed result at {result_path} has a different configuration")

    seed_everything(config.seed)
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    pools = token_pools(config)
    model = RootTransportMQARRetriever(config).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    should_train = config.mode == "learned" or config.residual_tables > 0
    optimizer = (
        torch.optim.AdamW(
            trainable,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        if should_train
        else None
    )
    best_validation_ce = float("inf")
    best_step = 0
    best_state = _clone_state_dict(model)
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()

    if should_train:
        assert optimizer is not None
        for step in range(1, config.steps + 1):
            model.relation.temperature = _temperature(config, step)
            pair_count, query_count = _training_shape(step, config.data_seed)
            batch = generate_pool_mqar_batch(
                token_pool=pools["train"],
                batch_size=config.batch_size,
                pair_count=pair_count,
                query_count=query_count,
                seed=config.data_seed + step * 1_000_003,
            ).to(device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            scores, _ = model(batch)
            retrieval_loss = target_token_loss(scores, batch.query_indices)
            entropy = model.relation.entropy()
            loss = retrieval_loss + config.assignment_entropy_weight * entropy
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}")
            loss.backward()
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(trainable, config.gradient_clip).item())
            optimizer.step()
            should_validate = step == 1 or step % config.validation_interval == 0 or step == config.steps
            if should_validate:
                validation = evaluate(
                    model,
                    token_pool=pools["validation"],
                    pair_count=8,
                    query_count=4,
                    episodes=config.validation_episodes,
                    batch_size=config.evaluation_batch_size,
                    data_seed=config.data_seed + 50_000_000,
                    device=device,
                    condition="base",
                )
                row = {
                    "step": step,
                    "temperature": model.relation.temperature,
                    "train_retrieval_ce": float(retrieval_loss.item()),
                    "assignment_entropy": float(entropy.item()),
                    "gradient_norm": gradient_norm,
                    "validation_target_token_ce": float(validation["target_token_ce"]),
                    "validation_target_token_accuracy": float(validation["target_token_accuracy"]),
                    "correspondence_accuracy": model.relation.correspondence_accuracy(),
                }
                history.append(row)
                print(json.dumps(row), flush=True)
                if float(validation["target_token_ce"]) < best_validation_ce:
                    best_validation_ce = float(validation["target_token_ce"])
                    best_step = step
                    best_state = _clone_state_dict(model)
    else:
        validation = evaluate(
            model,
            token_pool=pools["validation"],
            pair_count=8,
            query_count=4,
            episodes=config.validation_episodes,
            batch_size=config.evaluation_batch_size,
            data_seed=config.data_seed + 50_000_000,
            device=device,
            condition="base",
        )
        best_validation_ce = float(validation["target_token_ce"])

    training_seconds = time.perf_counter() - started
    model.load_state_dict(best_state)
    pool_relabel = token_permutation(config.test_tokens, config.token_relabel_seed)
    evaluation: dict[str, dict[str, dict[str, float | int]]] = {}
    for name, pair_count, query_count in EVALUATION_SPECS:
        evaluation[name] = {}
        for condition in EVALUATION_CONDITIONS:
            evaluation[name][condition] = evaluate(
                model,
                token_pool=pools["test"],
                pair_count=pair_count,
                query_count=query_count,
                episodes=config.evaluation_episodes,
                batch_size=config.evaluation_batch_size,
                data_seed=config.data_seed + 70_000_000 + pair_count * 100_003,
                device=device,
                condition=condition,
                pool_permutation=pool_relabel if condition == "token_relabel" else None,
            )
    selected = model.relation.hard_signed_indices().detach().cpu()
    result: dict[str, Any] = {
        "complete": True,
        "config": asdict(config),
        "architecture": {
            "root_carrier": f"A_{config.relation_dim - 1} with {model.relation.root_count} comparison roots",
            "active_query_roots": config.root_budget,
            "transport": "one signed K-root address per active Q root at hard inference",
            "training_relaxation": (
                "dense categorical softmax over signed K roots" if config.mode == "learned" else "none; exact gauge-induced transport"
            ),
            "residual": (f"{config.residual_tables} independent {config.residual_roots_per_side}+{config.residual_roots_per_side}-root LUTs"),
            "value": "exact B token from the same causal row as predecessor-A K",
        },
        "trainable_parameters": model.trainable_parameters,
        "hard_inference_transport_bytes": model.relation.hard_inference_bytes,
        "hard_inference_root_products": config.root_budget,
        "residual_lut_reads_per_pair": config.residual_tables,
        "correspondence_accuracy": model.relation.correspondence_accuracy(),
        "selected_signed_root_indices": selected.tolist(),
        "oracle_signed_root_indices": model.relation.oracle_signed_index.detach().cpu().tolist(),
        "best_step": best_step,
        "best_validation_target_token_ce": best_validation_ce,
        "training": {
            "seconds": training_seconds,
            "history": history,
        },
        "evaluation": evaluation,
    }
    checkpoint = {
        "config": asdict(config),
        "best_step": best_step,
        "state_dict": best_state,
    }
    atomic_torch_save(checkpoint, args.out_dir / "best.pt")
    atomic_json_write(result, result_path)
    print(
        json.dumps(
            {
                "status": "complete",
                "result": str(result_path),
                "best_step": best_step,
                "correspondence_accuracy": result["correspondence_accuracy"],
            }
        ),
        flush=True,
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run one root-transport condition")
    run.add_argument("--mode", choices=TRANSPORT_MODES, required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--out-dir", type=Path, required=True)
    run.add_argument("--steps", type=int, default=1500)
    run.add_argument("--batch-size", type=int, default=128)
    run.add_argument("--learning-rate", type=float, default=0.05)
    run.add_argument("--weight-decay", type=float, default=0.0)
    run.add_argument("--gradient-clip", type=float, default=5.0)
    run.add_argument("--validation-interval", type=int, default=100)
    run.add_argument("--validation-episodes", type=int, default=512)
    run.add_argument("--evaluation-episodes", type=int, default=2048)
    run.add_argument("--evaluation-batch-size", type=int, default=64)
    run.add_argument("--vocab-size", type=int, default=768)
    run.add_argument("--train-tokens", type=int, default=512)
    run.add_argument("--validation-tokens", type=int, default=128)
    run.add_argument("--test-tokens", type=int, default=128)
    run.add_argument("--relation-dim", type=int, default=32)
    run.add_argument("--root-budget", type=int, default=96)
    run.add_argument("--root-subset-seed", type=int, default=4242)
    run.add_argument("--residual-tables", type=int, default=0)
    run.add_argument("--residual-roots-per-side", type=int, default=3)
    run.add_argument("--assignment-temperature-start", type=float, default=2.0)
    run.add_argument("--assignment-temperature-end", type=float, default=0.05)
    run.add_argument("--assignment-entropy-weight", type=float, default=1e-3)
    run.add_argument("--data-seed", type=int, default=1729)
    run.add_argument("--codebook-seed", type=int, default=2718)
    run.add_argument("--query-gauge-seed", type=int, default=202607241)
    run.add_argument("--key-gauge-seed", type=int, default=202607242)
    run.add_argument("--wrong-key-gauge-seed", type=int, default=202607243)
    run.add_argument("--token-relabel-seed", type=int, default=314159)
    run.add_argument("--device", default="cuda")
    run.set_defaults(function=run_experiment)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
