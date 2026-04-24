from __future__ import annotations

import torch
from torch import Tensor


def izhikevich_surrogate(u: Tensor) -> Tensor:
    abs_u = u.abs()
    sig = torch.sigmoid(abs_u)
    sign = torch.where(u > 0, -torch.ones_like(u), torch.ones_like(u))
    return sign * 2.0 * sig * (1.0 - sig)


class StraightThroughHeaviside(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        ctx.save_for_backward(x)
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        (x,) = ctx.saved_tensors
        return grad_output * izhikevich_surrogate(x)


def ste_heaviside(x: Tensor) -> Tensor:
    return StraightThroughHeaviside.apply(x)
