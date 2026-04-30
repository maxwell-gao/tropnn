"""Tropical layer with shared ETF-regularized dictionary and sparse cell coefficients.

Implements direction A from
``doc/TROPNN_SCALING_GEOMETRY_AND_MEMORY_BOUND.md`` and
``doc/TROPNN_LOW_RANK_ZERO_DENSE_RESULTS.md``: per-cell codes are generated
from a shared dictionary of unit-norm atoms with learnable sparse
coefficients. Two zero-dense routing front-ends are supported:

* ``route_source='anchors'``: same sparse coordinate route as
  :class:`TropZeroDenseLinear`; cheapest read but weak per-feature
  discrimination.
* ``route_source='sketch'``: a fixed CountSketch projects the input into a
  small ``route_dim`` space, then heads score with full dense site
  vectors. No learned dense matmul on the input side, but the routing
  sees a richer view of the feature, which is what makes
  ``tied_tropical_dict`` competitive with ``tied_tropical_lowrank``.

Mathematical form per layer:

.. code-block:: text

    z           = route_project(x)
    score[h, k] = route_score(z, h, k) + lifting[h, k]
    winner_h    = argmax_k score[h, k]
    code[h, k]  = sum_l alpha[h, k, l] * basis[support[h, k, l]]
    y           = bias + code_scale * sum_h code[h, winner_h]

Storage layout:

* ``basis``    : ``(dict_size, out_features)``           shared, ETF-regularized
* ``coeff``    : ``(heads, cells, dict_sparsity)``       learnable mixture weights
* ``support``  : ``(heads, cells, dict_sparsity)``       fixed sparse atom indices
* anchors mode: ``router_weight``, ``router_bias``, ``anchors`` buffer.
* sketch mode: ``sites``, ``lifting``, plus CountSketch buffers.

Active read at inference is ``H * dict_sparsity`` rows of ``basis``, instead of
the ``H * out_features`` free codes used by ``TropZeroDenseLinear``. The
dictionary geometry is exposed through :meth:`TropDictLinear.dictionary_loss`
so that the training loop can add a Frobenius ETF penalty on top of the task
loss.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..backend import Backend
from .base import RoutedLinearBase
from .tropical import _minface_mix, _top2_indices

DictInit = Literal["orthogonal", "gaussian"]
DICT_INITS: tuple[DictInit, ...] = ("orthogonal", "gaussian")

RouteSource = Literal["anchors", "sketch"]
ROUTE_SOURCES: tuple[RouteSource, ...] = ("anchors", "sketch")


def _sample_disjoint_support(
    heads: int,
    cells: int,
    dict_sparsity: int,
    dict_size: int,
    generator: torch.Generator,
) -> Tensor:
    if dict_sparsity > dict_size:
        raise ValueError(f"dict_sparsity={dict_sparsity} cannot exceed dict_size={dict_size}")
    support = torch.empty(heads, cells, dict_sparsity, dtype=torch.long)
    for h in range(heads):
        for k in range(cells):
            support[h, k] = torch.randperm(dict_size, generator=generator)[:dict_sparsity]
    return support


def _init_dict_basis(
    dict_size: int,
    out_features: int,
    init: DictInit,
    init_scale: float,
    generator: torch.Generator,
) -> Tensor:
    basis = torch.empty(dict_size, out_features)
    if init == "orthogonal":
        torch.nn.init.orthogonal_(basis, generator=generator)
    elif init == "gaussian":
        basis = torch.randn(dict_size, out_features, generator=generator)
    else:
        raise ValueError(f"unknown dict init {init!r}; expected one of {DICT_INITS}")
    norms = basis.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return (basis / norms) * init_scale


def _init_count_sketch(
    in_features: int,
    route_dim: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor, float]:
    buckets = torch.randint(0, route_dim, (in_features,), generator=generator, dtype=torch.long)
    if in_features <= route_dim:
        buckets[:in_features] = torch.randperm(route_dim, generator=generator)[:in_features]
    signs = torch.randint(0, 2, (in_features,), generator=generator, dtype=torch.long).float().mul_(2.0).sub_(1.0)
    scale = math.sqrt(route_dim / in_features) if in_features > route_dim else 1.0
    return buckets, signs, scale


class TropDictLinear(RoutedLinearBase):
    """Sparse tropical layer backed by a shared ETF-regularized dictionary."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        heads: int = 32,
        cells: int = 4,
        route_source: RouteSource = "anchors",
        route_terms: int = 2,
        route_dim: int | None = None,
        dict_size: int | None = None,
        dict_sparsity: int = 4,
        dict_init: DictInit = "orthogonal",
        dict_init_scale: float = 1.0,
        coeff_init_std: float | None = None,
        backend: Backend = "torch",
        seed: int = 0,
        use_output_scaling: bool = True,
        use_route_residual: bool = False,
    ) -> None:
        if heads < 1:
            raise ValueError(f"heads must be >= 1, got {heads}")
        if cells < 2:
            raise ValueError(f"cells must be >= 2, got {cells}")
        if route_source not in ROUTE_SOURCES:
            raise ValueError(f"route_source must be one of {ROUTE_SOURCES}, got {route_source!r}")
        if route_source == "anchors" and route_terms < 1:
            raise ValueError(f"route_terms must be >= 1 for route_source='anchors', got {route_terms}")
        if dict_sparsity < 1:
            raise ValueError(f"dict_sparsity must be >= 1, got {dict_sparsity}")
        if backend != "torch":
            raise ValueError(f"TropDictLinear currently supports backend='torch' only, got {backend!r}")

        resolved_dict_size = dict_size if dict_size is not None else max(out_features, 2 * out_features)
        if resolved_dict_size < dict_sparsity:
            raise ValueError(f"dict_size={resolved_dict_size} must be >= dict_sparsity={dict_sparsity}")

        if use_route_residual and route_source != "sketch":
            raise ValueError("use_route_residual=True requires route_source='sketch' (latent must live in out_features)")

        super().__init__(in_features, out_features, backend=backend, output_scale=1.0)
        self.heads = heads
        self.cells = cells
        self.route_source = route_source
        self.route_terms = route_terms
        self.dict_size = resolved_dict_size
        self.dict_sparsity = dict_sparsity
        self.dict_init = dict_init
        self.dict_init_scale = float(dict_init_scale)
        self.use_route_residual = use_route_residual
        self.code_scale = 1.0 / math.sqrt(heads) if use_output_scaling else 1.0

        generator = torch.Generator().manual_seed(seed)

        if route_source == "anchors":
            self.route_dim = route_terms
            anchors = torch.randint(
                0,
                in_features,
                (heads, cells, route_terms),
                generator=generator,
                dtype=torch.long,
            )
            self.register_buffer("anchors", anchors)
            router_std = 1.0 / math.sqrt(route_terms)
            self.router_weight = nn.Parameter(torch.randn(heads, cells, route_terms, generator=generator) * router_std)
            self.router_bias = nn.Parameter(torch.zeros(heads, cells))
        else:
            resolved_route_dim = route_dim if route_dim is not None else out_features
            if resolved_route_dim < 1:
                raise ValueError(f"route_dim must be >= 1 for route_source='sketch', got {resolved_route_dim}")
            if use_route_residual and resolved_route_dim != out_features:
                raise ValueError(f"use_route_residual=True requires route_dim=={out_features} (out_features), got {resolved_route_dim}")
            self.route_dim = resolved_route_dim
            buckets, signs, scale = _init_count_sketch(in_features, resolved_route_dim, generator)
            self.register_buffer("input_buckets", buckets)
            self.register_buffer("input_signs", signs)
            self.register_buffer("input_scale", torch.tensor(scale, dtype=torch.float32))
            site_std = 1.0 / math.sqrt(resolved_route_dim)
            self.sites = nn.Parameter(torch.randn(heads, cells, resolved_route_dim, generator=generator) * site_std)
            self.lifting = nn.Parameter(torch.zeros(heads, cells))

        support = _sample_disjoint_support(heads, cells, dict_sparsity, self.dict_size, generator)
        self.register_buffer("support", support)
        resolved_coeff_std = coeff_init_std if coeff_init_std is not None else 1.0 / math.sqrt(dict_sparsity)
        self.coeff = nn.Parameter(torch.randn(heads, cells, dict_sparsity, generator=generator) * resolved_coeff_std)
        self.basis = nn.Parameter(_init_dict_basis(self.dict_size, out_features, dict_init, self.dict_init_scale, generator))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def extra_repr(self) -> str:
        common = (
            f"in_features={self.in_features}, out_features={self.out_features}, heads={self.heads}, "
            f"cells={self.cells}, route_source={self.route_source!r}, route_dim={self.route_dim}, "
            f"dict_size={self.dict_size}, dict_sparsity={self.dict_sparsity}, dict_init={self.dict_init!r}, "
            f"use_route_residual={self.use_route_residual}, backend={self.backend!r}"
        )
        if self.route_source == "anchors":
            return f"{common}, route_terms={self.route_terms}"
        return common

    def _project_input(self, x: Tensor, compute_dtype: torch.dtype) -> Tensor:
        if self.route_source == "anchors":
            return x.to(compute_dtype)
        x_compute = x.to(dtype=compute_dtype)
        buckets = self.input_buckets.to(device=x.device)
        signs = self.input_signs.to(device=x.device, dtype=compute_dtype)
        latent = x_compute.new_zeros(*x_compute.shape[:-1], self.route_dim)
        scatter_idx = buckets.view(*([1] * (x_compute.ndim - 1)), self.in_features).expand_as(x_compute)
        latent.scatter_add_(-1, scatter_idx, x_compute * signs.view(*([1] * (x_compute.ndim - 1)), self.in_features))
        return latent * self.input_scale.to(device=x.device, dtype=compute_dtype)

    def _scores(self, latent: Tensor, *, compute_dtype: torch.dtype) -> Tensor:
        if self.route_source == "anchors":
            batch, seq, _ = latent.shape
            selected = latent.index_select(-1, self.anchors.flatten()).view(batch, seq, self.heads, self.cells, self.route_terms)
            weight = self.router_weight.to(dtype=compute_dtype, device=latent.device)
            bias = self.router_bias.to(dtype=compute_dtype, device=latent.device)
            return (selected * weight.view(1, 1, self.heads, self.cells, self.route_terms)).sum(dim=-1) + bias.view(1, 1, self.heads, self.cells)
        sites = self.sites.to(dtype=compute_dtype, device=latent.device)
        lifting = self.lifting.to(dtype=compute_dtype, device=latent.device)
        return torch.einsum("bsd,hkd->bshk", latent, sites) + lifting.view(1, 1, self.heads, self.cells)

    def _gather_cell_codes(
        self,
        winner_idx: Tensor,
        *,
        input_device: torch.device,
        compute_dtype: torch.dtype,
    ) -> Tensor:
        batch, seq, heads = winner_idx.shape
        head_offsets = (torch.arange(heads, device=input_device) * self.cells).view(1, 1, heads)
        flat_winner = (winner_idx + head_offsets).reshape(-1)

        flat_support = self.support.view(self.heads * self.cells, self.dict_sparsity).to(device=input_device)
        cell_support = flat_support.index_select(0, flat_winner).view(batch, seq, heads, self.dict_sparsity)

        flat_coeff = self.coeff.view(self.heads * self.cells, self.dict_sparsity).to(dtype=compute_dtype, device=input_device)
        cell_coeff = flat_coeff.index_select(0, flat_winner).view(batch, seq, heads, self.dict_sparsity)

        basis = self.basis.to(dtype=compute_dtype, device=input_device)
        cell_atoms = basis.index_select(0, cell_support.reshape(-1)).view(batch, seq, heads, self.dict_sparsity, self.out_features)
        return (cell_coeff.unsqueeze(-1) * cell_atoms).sum(dim=-2)

    def _output_from_codes(
        self,
        codes: Tensor,
        *,
        input_device: torch.device,
        compute_dtype: torch.dtype,
    ) -> Tensor:
        output = codes.sum(dim=2) * self.code_scale
        return output + self.bias.to(dtype=compute_dtype, device=input_device)

    def _route_output(
        self,
        latent: Tensor,
        *,
        input_device: torch.device,
        compute_dtype: torch.dtype,
        training: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        scores = self._scores(latent, compute_dtype=compute_dtype)
        winner_idx, runner_idx, margins = _top2_indices(scores)
        winner_codes = self._gather_cell_codes(winner_idx, input_device=input_device, compute_dtype=compute_dtype)
        if training:
            runner_codes = self._gather_cell_codes(runner_idx, input_device=input_device, compute_dtype=compute_dtype)
            codes = _minface_mix(winner_codes, runner_codes, margins)
        else:
            codes = winner_codes
        output = self._output_from_codes(codes, input_device=input_device, compute_dtype=compute_dtype)
        if self.use_route_residual:
            output = output + latent
        return output, winner_idx, margins

    def dictionary_loss(self, *, weight: float = 1.0, eps: float = 1e-12) -> Tensor:
        """Mean off-diagonal squared overlap of the unit-normalized dictionary.

        Returns ``weight * (sum_{i!=j} (B_i . B_j)^2) / (K * (K-1))`` where
        ``B`` is the row-normalized basis. With this normalization
        ``weight=1`` is the natural ETF scale: a random unit-norm Gaussian
        basis in ``R^d`` has expected loss ``~1/d``, while an ETF saturates
        the Welch lower bound ``(K - d) / (d * (K - 1))``.
        """
        if weight == 0.0:
            return torch.zeros((), device=self.basis.device, dtype=self.basis.dtype)
        unit = F.normalize(self.basis, dim=1, eps=eps)
        gram = unit @ unit.t()
        eye = torch.eye(self.dict_size, device=unit.device, dtype=unit.dtype)
        offdiag_sq_sum = (gram - eye).square().sum()
        n_offdiag = max(1, self.dict_size * (self.dict_size - 1))
        return weight * offdiag_sq_sum / n_offdiag

    @torch.no_grad()
    def dictionary_diagnostics(self, *, eps: float = 1e-12) -> dict[str, float]:
        """Per-call diagnostic snapshot of dictionary frame geometry."""
        unit = F.normalize(self.basis.detach(), dim=1, eps=eps)
        gram = unit @ unit.t()
        offdiag = gram - torch.eye(self.dict_size, device=unit.device, dtype=unit.dtype)
        offdiag_sq = offdiag.square()
        mask = ~torch.eye(self.dict_size, dtype=torch.bool, device=unit.device)
        norms = self.basis.detach().norm(dim=1)
        return {
            "dict_norm_mean": float(norms.mean().item()),
            "dict_norm_std": float(norms.std(unbiased=False).item()),
            "dict_offdiag_max_abs": float(offdiag.masked_select(mask).abs().max().item()),
            "dict_offdiag_mean_sq": float(offdiag_sq.masked_select(mask).mean().item()),
        }
