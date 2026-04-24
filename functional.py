from __future__ import annotations

from torch import Tensor


def minface_uncertainty(margins: Tensor) -> Tensor:
    return 0.5 / (1.0 + margins.abs())


def gather_winner_values(cell_values: Tensor, winner_idx: Tensor) -> Tensor:
    out_dim = cell_values.shape[-1]
    return cell_values.gather(
        -2,
        winner_idx.unsqueeze(-1).unsqueeze(-1).expand(*winner_idx.shape, 1, out_dim),
    ).squeeze(-2)


def trop_minface_training_output(scores: Tensor, cell_values: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    top2_vals, top2_idx = scores.topk(k=2, dim=-1)
    winner_idx = top2_idx[..., 0]
    winner_values = gather_winner_values(cell_values, winner_idx)
    runner_values = gather_winner_values(cell_values, top2_idx[..., 1])
    margins = top2_vals[..., 0] - top2_vals[..., 1]
    uncertainty = minface_uncertainty(margins).unsqueeze(-1)
    group_output = winner_values + uncertainty * (runner_values - winner_values)
    return group_output, winner_idx, margins


def trop_minface_eval_output(scores: Tensor, cell_values: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    winner_idx = scores.argmax(dim=-1)
    winner_values = gather_winner_values(cell_values, winner_idx)
    top2_vals = scores.topk(k=2, dim=-1).values
    margins = top2_vals[..., 0] - top2_vals[..., 1]
    return winner_values, winner_idx, margins
