from __future__ import annotations

import time

import torch
import torch.nn as nn

from tropnn.examples.emnist import EmnistLinearClassifier, EmnistPairwiseClassifier


def _summary(model: nn.Module, x: torch.Tensor) -> dict[str, object]:
    start = time.perf_counter()
    with torch.no_grad():
        model(x)
    wall_ms = (time.perf_counter() - start) * 1000.0
    return {
        "device": str(x.device),
        "params": sum(p.numel() for p in model.parameters()),
        "wall_ms": wall_ms,
        "top_ops": [{"name": type(model).__name__, "self_cpu_time_total": wall_ms}],
    }


def profile_family_set(
    *,
    batch_size: int,
    input_dim: int,
    out_features: int,
    depth: int,
    pairwise_hidden: int,
    pairwise_tables: int,
    pairwise_comparisons: int,
    linear_hidden: int,
    seed: int,
    dtype: str,
    device: str,
    **_: object,
) -> dict[str, dict[str, object]]:
    torch.manual_seed(seed)
    dt = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
    x = torch.randn(batch_size, input_dim, device=device, dtype=dt)
    return {
        "pairwise": _summary(
            EmnistPairwiseClassifier(
                input_dim=input_dim,
                hidden_dim=pairwise_hidden,
                num_classes=out_features,
                depth=depth,
                tables=pairwise_tables,
                comparisons=pairwise_comparisons,
                seed=seed,
            ).to(device=device, dtype=dt),
            x,
        ),
        "linear": _summary(
            EmnistLinearClassifier(input_dim=input_dim, hidden_dim=linear_hidden, num_classes=out_features, depth=depth, seed=seed).to(
                device=device,
                dtype=dt,
            ),
            x,
        ),
    }
