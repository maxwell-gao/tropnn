from __future__ import annotations

import torch
from torch import Tensor

SurrogateName = str


def izhikevich_surrogate(u: Tensor) -> Tensor:
    abs_u = u.abs()
    sig = torch.sigmoid(abs_u)
    sign = torch.where(u > 0, -torch.ones_like(u), torch.ones_like(u))
    return sign * 2.0 * sig * (1.0 - sig)


def fast_sigmoid_odd_surrogate(u: Tensor) -> Tensor:
    return -0.5 * torch.sign(u) / (1.0 + u.abs()).square()


def surrogate_gradient(u: Tensor, surrogate: SurrogateName = "fast_sigmoid_odd") -> Tensor:
    if surrogate == "fast_sigmoid_odd":
        return fast_sigmoid_odd_surrogate(u)
    if surrogate == "izhikevich":
        return izhikevich_surrogate(u)
    raise ValueError(f"unsupported surrogate {surrogate!r}")


class StraightThroughHeaviside(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, surrogate: SurrogateName = "fast_sigmoid_odd") -> Tensor:
        ctx.save_for_backward(x)
        ctx.surrogate = surrogate
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        (x,) = ctx.saved_tensors
        return grad_output * surrogate_gradient(x, ctx.surrogate), None


def ste_heaviside(x: Tensor, surrogate: SurrogateName = "fast_sigmoid_odd") -> Tensor:
    return StraightThroughHeaviside.apply(x, surrogate)
