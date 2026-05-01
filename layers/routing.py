from __future__ import annotations

import torch
from torch import Tensor


def top2_indices(scores: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Return winner index, runner-up index, and their score margin."""
    top2_vals, top2_idx = scores.topk(k=2, dim=-1)
    return top2_idx[..., 0], top2_idx[..., 1], top2_vals[..., 0] - top2_vals[..., 1]


def minface_mix(winner_values: Tensor, runner_values: Tensor, margins: Tensor) -> Tensor:
    """Local min-face interpolation between winner and runner-up values."""
    return winner_values + (0.5 / (1.0 + margins.abs())).unsqueeze(-1) * (runner_values - winner_values)


def packed_route_index_dtype(cells: int) -> torch.dtype:
    if cells <= 256:
        return torch.uint8
    if cells <= 32767:
        return torch.int16
    return torch.int64


def pack_route_indices(indices: Tensor, cells: int) -> Tensor:
    return indices.to(dtype=packed_route_index_dtype(cells))


def unpack_route_indices(indices: Tensor) -> Tensor:
    return indices if indices.dtype == torch.int64 else indices.to(dtype=torch.int64)


def gather_route_margins_from_scores(scores: Tensor, winner_idx: Tensor, runner_idx: Tensor) -> Tensor:
    winner_scores = scores.gather(dim=2, index=winner_idx.unsqueeze(-1)).squeeze(-1)
    runner_scores = scores.gather(dim=2, index=runner_idx.unsqueeze(-1)).squeeze(-1)
    return winner_scores - runner_scores


def recompute_route_margins_torch(latent_flat: Tensor, router_weight: Tensor, router_bias: Tensor, winner_idx: Tensor, runner_idx: Tensor) -> Tensor:
    heads, cells, code_dim = router_weight.shape
    scores = torch.matmul(latent_flat, router_weight.reshape(heads * cells, code_dim).t()).view(latent_flat.shape[0], heads, cells)
    scores = scores + router_bias.view(1, heads, cells)
    return gather_route_margins_from_scores(scores, winner_idx, runner_idx)


_top2_indices = top2_indices
_minface_mix = minface_mix
_packed_route_index_dtype = packed_route_index_dtype
_pack_route_indices = pack_route_indices
_unpack_route_indices = unpack_route_indices
_gather_route_margins_from_scores = gather_route_margins_from_scores
_recompute_route_margins_torch = recompute_route_margins_torch


__all__ = [
    "top2_indices",
    "minface_mix",
    "packed_route_index_dtype",
    "pack_route_indices",
    "unpack_route_indices",
    "gather_route_margins_from_scores",
    "recompute_route_margins_torch",
]
