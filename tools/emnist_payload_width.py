"""EMNIST payload-width sweep for pairwise-comparator LUT layers.

This experiment keeps the route computation fixed and changes only the payload
written by each routed table row:

* full_vector: each active table row writes the whole output vector.
* group_k*: each active table row writes k fixed output coordinates.
* scalar_k1: each active table row writes one fixed output coordinate.
* scalar_expander: each scalar row writes to a small fixed signed expander fanout.
* scalar_sign: each scalar row writes a fixed dense sign basis vector.
* walsh_affine: low-degree Boolean generators plus margin-affine shared slopes.
* comparator_*_kc: each comparator activation directly writes to k_c output coordinates.
* ternary_margin_*: T sparse C-fan-in ternary margins drive dense ternary
  linear or two-sided live actions, with semantic cost O(TC + TD).
* ladder_*: explicit ablation ladder separating full-code payload width,
  margin-strength output, comparator-side routing, and sparse writes.
* code_bits_k_hidden: hidden full-vector LUT plus explicit comparator-bit
  group coordinates; readout remains full-vector LUT.
* compare_swap_*_hidden: hidden full-vector LUT plus a zero-gated
  compare-swap geometry scaffold; readout remains full-vector LUT.
* full_lut_*_hidden: hidden full-vector LUT plus a zero-gated correction
  map; readout remains full-vector LUT.
* pairwise_glu_*: full-vector pairwise LUT value branch gated by another
  pairwise LUT branch, with either shared or independent route.
* binary_count_gated_lut: shared-route binary payload count value branch gated
  by a second binary count branch.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from tropnn.layers import ComparatorTwoSidedMargin, CoxeterLUT, K4FullLUT, TernaryMarginAction
from tropnn.layers.pairwise import PAIRWISE_ANCHOR_POLICIES, LutDType, PairwiseLUT, PairwiseRoute, PairwiseWalshLUT, ste_heaviside
from tropnn.tools.emnist_discrete_payload import (
    AccumulatorMode,
    DiscreteMethod,
    RowLocalDiscretePayloadOptimizer,
)
from tropnn.tools.emnist_cross_layer_anchor_sharing import (
    _boundary_density,
    _connected_components,
    _entropy,
    _grid,
    _pca_plane,
    _refinement,
    _route_entropy,
    _route_persistence,
    _signature_ids,
)
from tropnn.tools.emnist_payload_dtype_sweep import _build_local_loaders, _loader_examples

PayloadVariant = Literal[
    "full_vector",
    "k4_full_vector",
    "coxeter_full_vector",
    "group_k16",
    "group_k8",
    "group_k4",
    "scalar_k1",
    "scalar_expander",
    "scalar_sign",
    "walsh_affine",
    "comparator_sign_kc",
    "comparator_margin_kc",
    "comparator_signed_margin_kc",
    "comparator_two_sided_margin_kc",
    "ternary_margin_linear",
    "ternary_margin_two_sided",
    "code_bits_k_hidden",
    "compare_swap_independent_hidden",
    "compare_swap_reused_anchor_hidden",
    "compare_swap_anchor_delta_hidden",
    "full_lut_gated_twosided_margin_hidden",
    "full_lut_gated_signed_margin_hidden",
    "full_lut_chamber_diagonal_hidden",
    "full_lut_route_2x2_affine_hidden",
    "pairwise_glu_shared_route",
    "pairwise_glu_dual_route",
    "binary_count_gated_lut",
    "ladder_a_full_code_full_payload",
    "ladder_b_full_code_sparse_payload",
    "ladder_c_full_code_margin_sparse",
    "ladder_d_comparator_side_full_payload",
    "ladder_e_comparator_side_sparse",
]
ComparatorWritePolicy = Literal["endpoint", "local-linegraph", "expander"]
OptimizerName = Literal[
    "adamw",
    "ef_sgd",
    "adam_ef",
    "factored_adam_ef",
    "integer_adam_ef",
    "scaled_integer_adam_ef",
    "bop2_ternary",
]

VARIANT_PAYLOAD_WIDTH: dict[str, int | None] = {
    "full_vector": None,
    "k4_full_vector": None,
    "coxeter_full_vector": None,
    "group_k16": 16,
    "group_k8": 8,
    "group_k4": 4,
    "scalar_k1": 1,
    "scalar_expander": 1,
    "scalar_sign": 1,
    "walsh_affine": None,
    "comparator_sign_kc": None,
    "comparator_margin_kc": None,
    "comparator_signed_margin_kc": None,
    "comparator_two_sided_margin_kc": None,
    "ternary_margin_linear": None,
    "ternary_margin_two_sided": None,
    "code_bits_k_hidden": None,
    "compare_swap_independent_hidden": None,
    "compare_swap_reused_anchor_hidden": None,
    "compare_swap_anchor_delta_hidden": None,
    "full_lut_gated_twosided_margin_hidden": None,
    "full_lut_gated_signed_margin_hidden": None,
    "full_lut_chamber_diagonal_hidden": None,
    "full_lut_route_2x2_affine_hidden": None,
    "pairwise_glu_shared_route": None,
    "pairwise_glu_dual_route": None,
    "binary_count_gated_lut": None,
    "ladder_a_full_code_full_payload": None,
    "ladder_b_full_code_sparse_payload": None,
    "ladder_c_full_code_margin_sparse": None,
    "ladder_d_comparator_side_full_payload": None,
    "ladder_e_comparator_side_sparse": None,
}
DISCRETE_METHODS: dict[str, DiscreteMethod] = {
    "ef_sgd": "ef_sgd",
    "adam_ef": "adam_ef",
    "factored_adam_ef": "factored_adam_ef",
    "integer_adam_ef": "integer_adam_ef",
    "scaled_integer_adam_ef": "scaled_integer_adam_ef",
    "bop2_ternary": "bop2_ternary",
}
DEFAULT_PAYLOAD_LR = {
    "ef_sgd": 0.010,
    "adam_ef": 0.005,
    "factored_adam_ef": 0.005,
    "integer_adam_ef": 0.005,
    "scaled_integer_adam_ef": 0.005,
    "bop2_ternary": 0.005,
}


@dataclass(frozen=True)
class PayloadSpec:
    variant: PayloadVariant
    payload_width: int
    write_degree: int
    dense_sign_basis: bool

    @property
    def payload_label(self) -> str:
        if self.variant.startswith("ladder_"):
            return self.variant
        if self.variant == "walsh_affine":
            return "walsh_affine"
        if self.variant.startswith("comparator_"):
            return self.variant
        if self.variant.startswith("ternary_margin_"):
            return self.variant
        if self.variant == "code_bits_k_hidden":
            return f"code_bits_k{self.write_degree}_hidden"
        if self.variant.startswith("compare_swap_"):
            return self.variant
        if self.variant.startswith("full_lut_"):
            return self.variant
        if self.variant.startswith("pairwise_glu_") or self.variant == "binary_count_gated_lut":
            return self.variant
        if self.variant in {"full_vector", "k4_full_vector", "coxeter_full_vector"}:
            return "full_vector"
        if self.variant.startswith("group_"):
            return f"group_k{self.payload_width}"
        if self.variant == "scalar_expander":
            return f"scalar_expander_w{self.write_degree}"
        if self.variant == "scalar_sign":
            return "scalar_sign_basis"
        return "scalar_k1"


def _payload_spec(variant: PayloadVariant, output_dim: int, write_degree: int) -> PayloadSpec:
    if variant.startswith("ladder_"):
        if variant in {"ladder_a_full_code_full_payload", "ladder_d_comparator_side_full_payload"}:
            return PayloadSpec(variant=variant, payload_width=output_dim, write_degree=output_dim, dense_sign_basis=False)
        return PayloadSpec(variant=variant, payload_width=max(1, min(write_degree, output_dim)), write_degree=max(1, min(write_degree, output_dim)), dense_sign_basis=False)
    if variant == "walsh_affine" or variant.startswith("comparator_"):
        return PayloadSpec(variant=variant, payload_width=0, write_degree=0, dense_sign_basis=False)
    if variant.startswith("ternary_margin_"):
        return PayloadSpec(variant=variant, payload_width=output_dim, write_degree=output_dim, dense_sign_basis=True)
    if variant == "code_bits_k_hidden":
        group_size = max(1, min(write_degree, output_dim))
        return PayloadSpec(variant=variant, payload_width=output_dim, write_degree=group_size, dense_sign_basis=False)
    if variant.startswith("compare_swap_"):
        return PayloadSpec(variant=variant, payload_width=output_dim, write_degree=output_dim, dense_sign_basis=False)
    if variant.startswith("full_lut_"):
        return PayloadSpec(variant=variant, payload_width=output_dim, write_degree=output_dim, dense_sign_basis=False)
    if variant.startswith("pairwise_glu_") or variant == "binary_count_gated_lut":
        return PayloadSpec(variant=variant, payload_width=output_dim, write_degree=output_dim, dense_sign_basis=False)
    width = VARIANT_PAYLOAD_WIDTH[variant]
    if width is None:
        return PayloadSpec(variant=variant, payload_width=output_dim, write_degree=output_dim, dense_sign_basis=False)
    degree = max(1, min(write_degree, output_dim)) if variant == "scalar_expander" else 1
    return PayloadSpec(
        variant=variant,
        payload_width=max(1, min(width, output_dim)),
        write_degree=degree,
        dense_sign_basis=(variant == "scalar_sign"),
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _fixed_signs(shape: tuple[int, ...], seed: int, device: torch.device | None = None) -> Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    signs = torch.randint(0, 2, shape, generator=generator, dtype=torch.int8)
    signs = signs.to(torch.float32).mul_(2.0).sub_(1.0)
    return signs if device is None else signs.to(device)


class PayloadWidthLUTLayer(nn.Module):
    """Pairwise-comparator route with configurable payload writeback."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        tables: int,
        comparisons: int,
        variant: PayloadVariant,
        anchor_policy: str,
        seed: int,
        lut_init_std: float,
        write_degree: int,
        use_output_scaling: bool,
        use_min_margin_ste: bool,
    ) -> None:
        super().__init__()
        if tables < 1:
            raise ValueError(f"tables must be >= 1, got {tables}")
        if comparisons < 1:
            raise ValueError(f"comparisons must be >= 1, got {comparisons}")
        if anchor_policy not in PAIRWISE_ANCHOR_POLICIES:
            raise ValueError(f"unknown anchor_policy={anchor_policy!r}; choices={PAIRWISE_ANCHOR_POLICIES}")

        spec = _payload_spec(variant, output_dim, write_degree)
        template = PairwiseLUT(
            input_dim,
            1,
            tables=tables,
            comparisons=comparisons,
            anchor_policy=anchor_policy,
            seed=seed,
            anchor_seed=seed,
            backend="torch",
        )

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.tables = int(tables)
        self.comparisons = int(comparisons)
        self.table_size = 1 << int(comparisons)
        self.variant = variant
        self.payload_width = int(spec.payload_width)
        self.write_degree = int(spec.write_degree)
        self.output_scale = 1.0 / math.sqrt(tables) if use_output_scaling else 1.0
        self.use_min_margin_ste = bool(use_min_margin_ste)

        self.register_buffer("anchors", template.anchors.detach().clone())
        self.register_buffer("powers", 2 ** torch.arange(comparisons, dtype=torch.long))
        self.register_buffer("output_indices", self._make_output_indices(seed + 17))
        self.register_buffer("write_indices", self._make_write_indices(seed + 31))
        self.register_buffer("write_signs", self._make_write_signs(seed + 43))
        self.register_buffer("sign_basis", self._make_sign_basis(seed + 59))

        self.thresholds = nn.Parameter(torch.zeros(tables, comparisons))
        self.lut = nn.Parameter(torch.randn(tables, self.table_size, self.payload_width) * lut_init_std)

    def clear_packed_payload_cache(self) -> None:
        return None

    def payload_parameters(self) -> list[Tensor]:
        return [self.lut]

    @property
    def payload_params(self) -> int:
        return self.lut.numel()

    @property
    def bias_generator_params(self) -> int:
        return self.lut.numel()

    @property
    def slope_coeff_params(self) -> int:
        return 0

    @property
    def slope_generator_params(self) -> int:
        return 0

    def _make_output_indices(self, seed: int) -> Tensor:
        del seed
        offsets = torch.arange(self.payload_width, dtype=torch.long).view(1, -1)
        base = (torch.arange(self.tables, dtype=torch.long).view(-1, 1) * self.payload_width) % self.output_dim
        return (base + offsets) % self.output_dim

    def _make_write_indices(self, seed: int) -> Tensor:
        if self.variant != "scalar_expander":
            return torch.zeros(self.tables, 1, dtype=torch.long)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        stride = max(1, self.output_dim // max(1, self.write_degree))
        base = torch.arange(self.tables, dtype=torch.long).view(-1, 1)
        offsets = torch.arange(self.write_degree, dtype=torch.long).view(1, -1)
        jitter = torch.randint(0, max(1, stride), (self.tables, self.write_degree), generator=generator, dtype=torch.long)
        return (base * 997 + offsets * stride + jitter) % self.output_dim

    def _make_write_signs(self, seed: int) -> Tensor:
        if self.variant != "scalar_expander":
            return torch.ones(self.tables, 1, dtype=torch.float32)
        return _fixed_signs((self.tables, self.write_degree), seed)

    def _make_sign_basis(self, seed: int) -> Tensor:
        if self.variant != "scalar_sign":
            return torch.zeros(self.tables, self.output_dim, dtype=torch.float32)
        return _fixed_signs((self.tables, self.output_dim), seed) / math.sqrt(float(self.output_dim))

    def _route(self, x: Tensor) -> tuple[Tensor, Tensor]:
        anchor_a = self.anchors[:, :, 0].flatten()
        anchor_b = self.anchors[:, :, 1].flatten()
        x_a = x.index_select(-1, anchor_a).view(x.shape[0], self.tables, self.comparisons)
        x_b = x.index_select(-1, anchor_b).view(x.shape[0], self.tables, self.comparisons)
        margins = x_a - x_b - self.thresholds.to(device=x.device, dtype=x.dtype)
        bits = (margins > 0).to(torch.long)
        indices = (bits * self.powers.to(device=x.device).view(1, 1, -1)).sum(dim=-1)
        return indices, margins

    def _lookup(self, indices: Tensor) -> Tensor:
        table_offsets = torch.arange(self.tables, device=indices.device, dtype=torch.long).view(1, self.tables) * self.table_size
        flat_indices = (indices + table_offsets).reshape(-1)
        rows = self.lut.reshape(self.tables * self.table_size, self.payload_width).index_select(0, flat_indices)
        return rows.view(indices.shape[0], self.tables, self.payload_width)

    def _payload_to_output(self, payload: Tensor) -> Tensor:
        batch = payload.shape[0]
        dtype = payload.dtype
        device = payload.device
        if self.variant == "full_vector":
            output = payload.sum(dim=1)
        elif self.variant == "scalar_sign":
            output = payload[..., 0].matmul(self.sign_basis.to(device=device, dtype=dtype))
        elif self.variant == "scalar_expander":
            output = torch.zeros(batch, self.output_dim, device=device, dtype=dtype)
            values = payload[..., :1] * self.write_signs.to(device=device, dtype=dtype).view(1, self.tables, self.write_degree)
            values = values / math.sqrt(float(self.write_degree))
            indices = self.write_indices.to(device=device).view(1, self.tables, self.write_degree).expand(batch, -1, -1)
            output.scatter_add_(1, indices.reshape(batch, -1), values.reshape(batch, -1))
        else:
            output = torch.zeros(batch, self.output_dim, device=device, dtype=dtype)
            indices = self.output_indices.to(device=device).view(1, self.tables, self.payload_width).expand(batch, -1, -1)
            output.scatter_add_(1, indices.reshape(batch, -1), payload.reshape(batch, -1))
        return output * self.output_scale

    def _ste_correction(self, indices: Tensor, margins: Tensor, payload: Tensor) -> Tensor:
        if self.use_min_margin_ste:
            bit = margins.abs().argmin(dim=-1)
            margin = margins.gather(dim=-1, index=bit.unsqueeze(-1)).squeeze(-1)
            neighbor_indices = indices ^ (2 ** bit).long()
            ste_delta = ste_heaviside(margin) - (margin > 0).to(margin.dtype)
            delta = self._lookup(neighbor_indices) - payload
            return self._payload_to_output(delta * ste_delta.unsqueeze(-1))

        correction = torch.zeros(indices.shape[0], self.output_dim, device=payload.device, dtype=payload.dtype)
        for bit_idx in range(self.comparisons):
            margin = margins[:, :, bit_idx]
            ste_delta = ste_heaviside(margin) - (margin > 0).to(margin.dtype)
            neighbor = self._lookup(indices ^ int(self.powers[bit_idx].item()))
            correction = correction + self._payload_to_output((neighbor - payload) * ste_delta.unsqueeze(-1))
        return correction

    def forward(self, x: Tensor) -> Tensor:
        output, _indices = self.compute(x)
        return output

    def compute(self, x: Tensor) -> tuple[Tensor, Tensor]:
        input_dtype = x.dtype
        x32 = x.float()
        indices, margins = self._route(x32)
        payload = self._lookup(indices)
        output = self._payload_to_output(payload)
        if self.training and (x.requires_grad or self.thresholds.requires_grad):
            output = output + self._ste_correction(indices, margins, payload)
        return output.to(dtype=input_dtype), indices


class SharedRoutePairwiseGLULayer(nn.Module):
    """Full-vector pairwise LUT with a second LUT branch used only as a gate.

    This is the no-extra-comparison analogue of SwiGLU: the same route selects
    value and gate rows, then the layer returns ``v * 2 sigmoid(g)``.  The
    multiplier is exactly one when the gate branch is zero, so the variant can
    start as the ordinary full-vector LUT.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        tables: int,
        comparisons: int,
        anchor_policy: str,
        seed: int,
        lut_init_std: float,
        use_output_scaling: bool,
        use_min_margin_ste: bool,
    ) -> None:
        super().__init__()
        self.value_lut = PayloadWidthLUTLayer(
            input_dim,
            output_dim,
            tables=tables,
            comparisons=comparisons,
            variant="full_vector",
            anchor_policy=anchor_policy,
            seed=seed,
            lut_init_std=lut_init_std,
            write_degree=output_dim,
            use_output_scaling=use_output_scaling,
            use_min_margin_ste=use_min_margin_ste,
        )
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.tables = int(tables)
        self.comparisons = int(comparisons)
        self.table_size = 1 << int(comparisons)
        self.variant: PayloadVariant = "pairwise_glu_shared_route"
        self.payload_width = int(output_dim)
        self.write_degree = int(output_dim)
        self.output_scale = self.value_lut.output_scale
        self.use_min_margin_ste = bool(use_min_margin_ste)
        self.gate_lut = nn.Parameter(torch.zeros(tables, self.table_size, output_dim))

    @property
    def thresholds(self) -> Tensor:
        return self.value_lut.thresholds

    @property
    def payload_params(self) -> int:
        return self.value_lut.payload_params + self.gate_lut.numel()

    @property
    def bias_generator_params(self) -> int:
        return self.payload_params

    @property
    def slope_coeff_params(self) -> int:
        return 0

    @property
    def slope_generator_params(self) -> int:
        return 0

    def threshold_parameters(self) -> list[Tensor]:
        return [self.value_lut.thresholds]

    def payload_parameters(self) -> list[Tensor]:
        return [*self.value_lut.payload_parameters(), self.gate_lut]

    def clear_packed_payload_cache(self) -> None:
        self.value_lut.clear_packed_payload_cache()

    def _lookup_gate(self, indices: Tensor) -> Tensor:
        table_offsets = torch.arange(self.tables, device=indices.device, dtype=torch.long).view(1, self.tables) * self.table_size
        flat_indices = (indices + table_offsets).reshape(-1)
        rows = self.gate_lut.reshape(self.tables * self.table_size, self.output_dim).index_select(0, flat_indices)
        return rows.view(indices.shape[0], self.tables, self.output_dim)

    def _sum_payload(self, payload: Tensor) -> Tensor:
        return payload.sum(dim=1) * self.output_scale

    @staticmethod
    def _compose(value: Tensor, gate: Tensor) -> Tensor:
        return value * (2.0 * torch.sigmoid(gate))

    def _ste_correction(
        self,
        indices: Tensor,
        margins: Tensor,
        value_payload: Tensor,
        gate_payload: Tensor,
        value: Tensor,
        gate: Tensor,
        output: Tensor,
    ) -> Tensor:
        if self.use_min_margin_ste:
            bit = margins.abs().argmin(dim=-1)
            margin = margins.gather(dim=-1, index=bit.unsqueeze(-1)).squeeze(-1)
            neighbor_indices = indices ^ (2 ** bit).long()
            ste_delta = ste_heaviside(margin) - (margin > 0).to(margin.dtype)
            neighbor_value_payload = self.value_lut._lookup(neighbor_indices)
            neighbor_gate_payload = self._lookup_gate(neighbor_indices)
            neighbor_value = value.unsqueeze(1) + (neighbor_value_payload - value_payload) * self.output_scale
            neighbor_gate = gate.unsqueeze(1) + (neighbor_gate_payload - gate_payload) * self.output_scale
            neighbor_output = self._compose(neighbor_value, neighbor_gate)
            return ((neighbor_output - output.unsqueeze(1)) * ste_delta.unsqueeze(-1)).sum(dim=1)

        correction = torch.zeros_like(output)
        for bit_idx in range(self.comparisons):
            margin = margins[:, :, bit_idx]
            neighbor_indices = indices ^ int(self.value_lut.powers[bit_idx].item())
            ste_delta = ste_heaviside(margin) - (margin > 0).to(margin.dtype)
            neighbor_value_payload = self.value_lut._lookup(neighbor_indices)
            neighbor_gate_payload = self._lookup_gate(neighbor_indices)
            neighbor_value = value.unsqueeze(1) + (neighbor_value_payload - value_payload) * self.output_scale
            neighbor_gate = gate.unsqueeze(1) + (neighbor_gate_payload - gate_payload) * self.output_scale
            neighbor_output = self._compose(neighbor_value, neighbor_gate)
            correction = correction + ((neighbor_output - output.unsqueeze(1)) * ste_delta.unsqueeze(-1)).sum(dim=1)
        return correction

    def compute(self, x: Tensor) -> tuple[Tensor, Tensor]:
        input_dtype = x.dtype
        x32 = x.float()
        indices, margins = self.value_lut._route(x32)
        value_payload = self.value_lut._lookup(indices)
        gate_payload = self._lookup_gate(indices)
        value = self._sum_payload(value_payload)
        gate = self._sum_payload(gate_payload)
        output = self._compose(value, gate)
        if self.training and (x.requires_grad or self.thresholds.requires_grad):
            output = output + self._ste_correction(indices, margins, value_payload, gate_payload, value, gate, output)
        return output.to(dtype=input_dtype), indices

    def forward(self, x: Tensor) -> Tensor:
        output, _indices = self.compute(x)
        return output


class DualRoutePairwiseGLULayer(nn.Module):
    """Two independent pairwise LUT routes composed as a GLU gate."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        tables: int,
        comparisons: int,
        anchor_policy: str,
        seed: int,
        lut_init_std: float,
        use_output_scaling: bool,
        use_min_margin_ste: bool,
    ) -> None:
        super().__init__()
        self.value_lut = PayloadWidthLUTLayer(
            input_dim,
            output_dim,
            tables=tables,
            comparisons=comparisons,
            variant="full_vector",
            anchor_policy=anchor_policy,
            seed=seed,
            lut_init_std=lut_init_std,
            write_degree=output_dim,
            use_output_scaling=use_output_scaling,
            use_min_margin_ste=use_min_margin_ste,
        )
        self.gate_lut = PayloadWidthLUTLayer(
            input_dim,
            output_dim,
            tables=tables,
            comparisons=comparisons,
            variant="full_vector",
            anchor_policy=anchor_policy,
            seed=seed + 7919,
            lut_init_std=0.0,
            write_degree=output_dim,
            use_output_scaling=use_output_scaling,
            use_min_margin_ste=use_min_margin_ste,
        )
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.tables = int(tables)
        self.comparisons = int(comparisons)
        self.table_size = 1 << int(comparisons)
        self.variant: PayloadVariant = "pairwise_glu_dual_route"
        self.payload_width = int(output_dim)
        self.write_degree = int(output_dim)
        self.output_scale = self.value_lut.output_scale
        self.use_min_margin_ste = bool(use_min_margin_ste)

    @property
    def thresholds(self) -> Tensor:
        return self.value_lut.thresholds

    @property
    def payload_params(self) -> int:
        return self.value_lut.payload_params + self.gate_lut.payload_params

    @property
    def bias_generator_params(self) -> int:
        return self.payload_params

    @property
    def slope_coeff_params(self) -> int:
        return 0

    @property
    def slope_generator_params(self) -> int:
        return 0

    def threshold_parameters(self) -> list[Tensor]:
        return [self.value_lut.thresholds, self.gate_lut.thresholds]

    def payload_parameters(self) -> list[Tensor]:
        return [*self.value_lut.payload_parameters(), *self.gate_lut.payload_parameters()]

    def clear_packed_payload_cache(self) -> None:
        self.value_lut.clear_packed_payload_cache()
        self.gate_lut.clear_packed_payload_cache()

    def compute(self, x: Tensor) -> tuple[Tensor, Tensor]:
        value, value_indices = self.value_lut.compute(x)
        gate, gate_indices = self.gate_lut.compute(x)
        output = value * (2.0 * torch.sigmoid(gate))
        return output.to(dtype=x.dtype), torch.cat((value_indices, gate_indices), dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        output, _indices = self.compute(x)
        return output


class BinaryCountGatedLUTLayer(nn.Module):
    """Shared-route binary payload counts with a second count branch as gate."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        tables: int,
        comparisons: int,
        anchor_policy: str,
        seed: int,
        lut_init_std: float,
        use_output_scaling: bool,
        use_min_margin_ste: bool,
    ) -> None:
        super().__init__()
        template = PairwiseLUT(
            input_dim,
            1,
            tables=tables,
            comparisons=comparisons,
            anchor_policy=anchor_policy,
            seed=seed,
            anchor_seed=seed,
            backend="torch",
        )
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.tables = int(tables)
        self.comparisons = int(comparisons)
        self.table_size = 1 << int(comparisons)
        self.variant: PayloadVariant = "binary_count_gated_lut"
        self.payload_width = int(output_dim)
        self.write_degree = int(output_dim)
        self.output_scale = 1.0 / math.sqrt(tables) if use_output_scaling else 1.0
        self.use_min_margin_ste = bool(use_min_margin_ste)
        init_std = float(lut_init_std) if lut_init_std > 0.0 else 0.02
        self.register_buffer("anchors", template.anchors.detach().clone())
        self.register_buffer("powers", 2 ** torch.arange(comparisons, dtype=torch.long))
        self.thresholds = nn.Parameter(torch.zeros(tables, comparisons))
        self.value_logits = nn.Parameter(torch.randn(tables, self.table_size, output_dim) * init_std)
        self.gate_logits = nn.Parameter(torch.randn(tables, self.table_size, output_dim) * init_std)

    @property
    def payload_params(self) -> int:
        return self.value_logits.numel() + self.gate_logits.numel()

    @property
    def bias_generator_params(self) -> int:
        return self.payload_params

    @property
    def slope_coeff_params(self) -> int:
        return 0

    @property
    def slope_generator_params(self) -> int:
        return 0

    def threshold_parameters(self) -> list[Tensor]:
        return [self.thresholds]

    def payload_parameters(self) -> list[Tensor]:
        return [self.value_logits, self.gate_logits]

    def clear_packed_payload_cache(self) -> None:
        return None

    @staticmethod
    def _centered_binary(logits: Tensor) -> Tensor:
        prob = torch.sigmoid(logits)
        hard = (logits > 0).to(logits.dtype)
        return hard.detach() - prob.detach() + prob - 0.5

    def _route(self, x: Tensor) -> tuple[Tensor, Tensor]:
        anchor_a = self.anchors[:, :, 0].flatten()
        anchor_b = self.anchors[:, :, 1].flatten()
        x_a = x.index_select(-1, anchor_a).view(x.shape[0], self.tables, self.comparisons)
        x_b = x.index_select(-1, anchor_b).view(x.shape[0], self.tables, self.comparisons)
        margins = x_a - x_b - self.thresholds.to(device=x.device, dtype=x.dtype)
        bits = (margins > 0).to(torch.long)
        indices = (bits * self.powers.to(device=x.device).view(1, 1, -1)).sum(dim=-1)
        return indices, margins

    def _lookup_payload(self, table: Tensor, indices: Tensor) -> Tensor:
        table_offsets = torch.arange(self.tables, device=indices.device, dtype=torch.long).view(1, self.tables) * self.table_size
        flat_indices = (indices + table_offsets).reshape(-1)
        rows = table.reshape(self.tables * self.table_size, self.output_dim).index_select(0, flat_indices)
        return rows.view(indices.shape[0], self.tables, self.output_dim)

    def _payload_table(self, logits: Tensor) -> Tensor:
        return self._centered_binary(logits).to(device=logits.device, dtype=torch.float32)

    def _sum_payload(self, payload: Tensor) -> Tensor:
        return payload.sum(dim=1) * self.output_scale

    @staticmethod
    def _compose(value: Tensor, gate: Tensor) -> Tensor:
        return value * (2.0 * torch.sigmoid(gate))

    def _ste_correction(
        self,
        indices: Tensor,
        margins: Tensor,
        value_payload: Tensor,
        gate_payload: Tensor,
        value: Tensor,
        gate: Tensor,
        output: Tensor,
        value_table: Tensor,
        gate_table: Tensor,
    ) -> Tensor:
        if self.use_min_margin_ste:
            bit = margins.abs().argmin(dim=-1)
            margin = margins.gather(dim=-1, index=bit.unsqueeze(-1)).squeeze(-1)
            neighbor_indices = indices ^ (2 ** bit).long()
            ste_delta = ste_heaviside(margin) - (margin > 0).to(margin.dtype)
            neighbor_value_payload = self._lookup_payload(value_table, neighbor_indices)
            neighbor_gate_payload = self._lookup_payload(gate_table, neighbor_indices)
            neighbor_value = value.unsqueeze(1) + (neighbor_value_payload - value_payload) * self.output_scale
            neighbor_gate = gate.unsqueeze(1) + (neighbor_gate_payload - gate_payload) * self.output_scale
            neighbor_output = self._compose(neighbor_value, neighbor_gate)
            return ((neighbor_output - output.unsqueeze(1)) * ste_delta.unsqueeze(-1)).sum(dim=1)

        correction = torch.zeros_like(output)
        for bit_idx in range(self.comparisons):
            margin = margins[:, :, bit_idx]
            neighbor_indices = indices ^ int(self.powers[bit_idx].item())
            ste_delta = ste_heaviside(margin) - (margin > 0).to(margin.dtype)
            neighbor_value_payload = self._lookup_payload(value_table, neighbor_indices)
            neighbor_gate_payload = self._lookup_payload(gate_table, neighbor_indices)
            neighbor_value = value.unsqueeze(1) + (neighbor_value_payload - value_payload) * self.output_scale
            neighbor_gate = gate.unsqueeze(1) + (neighbor_gate_payload - gate_payload) * self.output_scale
            neighbor_output = self._compose(neighbor_value, neighbor_gate)
            correction = correction + ((neighbor_output - output.unsqueeze(1)) * ste_delta.unsqueeze(-1)).sum(dim=1)
        return correction

    def compute(self, x: Tensor) -> tuple[Tensor, Tensor]:
        input_dtype = x.dtype
        x32 = x.float()
        indices, margins = self._route(x32)
        value_table = self._payload_table(self.value_logits)
        gate_table = self._payload_table(self.gate_logits)
        value_payload = self._lookup_payload(value_table, indices)
        gate_payload = self._lookup_payload(gate_table, indices)
        value = self._sum_payload(value_payload)
        gate = self._sum_payload(gate_payload)
        output = self._compose(value, gate)
        if self.training and (x.requires_grad or self.thresholds.requires_grad):
            output = output + self._ste_correction(indices, margins, value_payload, gate_payload, value, gate, output, value_table, gate_table)
        return output.to(dtype=input_dtype), indices

    def forward(self, x: Tensor) -> Tensor:
        output, _indices = self.compute(x)
        return output


class CodeBitsHiddenLayer(nn.Module):
    """Full-vector LUT plus explicit comparator-bit group coordinates.

    This mirrors the C-reference `code_bits` hidden mode: each comparator sign
    writes one learned scalar per output group, and that scalar is broadcast over
    the coordinates in the group. The coefficients start at zero, so the layer
    initially behaves exactly like the full-vector LUT baseline.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        tables: int,
        comparisons: int,
        anchor_policy: str,
        seed: int,
        lut_init_std: float,
        group_size: int,
        use_output_scaling: bool,
        use_min_margin_ste: bool,
    ) -> None:
        super().__init__()
        self.full_lut = PayloadWidthLUTLayer(
            input_dim,
            output_dim,
            tables=tables,
            comparisons=comparisons,
            variant="full_vector",
            anchor_policy=anchor_policy,
            seed=seed,
            lut_init_std=lut_init_std,
            write_degree=output_dim,
            use_output_scaling=use_output_scaling,
            use_min_margin_ste=use_min_margin_ste,
        )
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.tables = int(tables)
        self.comparisons = int(comparisons)
        self.routes = self.tables * self.comparisons
        self.table_size = 1 << int(comparisons)
        self.variant: PayloadVariant = "code_bits_k_hidden"
        self.payload_width = int(output_dim)
        self.write_degree = max(1, min(int(group_size), self.output_dim))
        self.group_count = math.ceil(self.output_dim / self.write_degree)
        self.output_scale = self.full_lut.output_scale
        self.code_scale = 1.0 / math.sqrt(float(self.routes)) if use_output_scaling else 1.0
        self.use_min_margin_ste = bool(use_min_margin_ste)
        self.code_coeff = nn.Parameter(torch.zeros(self.routes, self.group_count))

    @property
    def thresholds(self) -> Tensor:
        return self.full_lut.thresholds

    @property
    def payload_params(self) -> int:
        return self.full_lut.payload_params + self.code_coeff.numel()

    @property
    def bias_generator_params(self) -> int:
        return self.payload_params

    @property
    def slope_coeff_params(self) -> int:
        return 0

    @property
    def slope_generator_params(self) -> int:
        return 0

    def payload_parameters(self) -> list[Tensor]:
        return [*self.full_lut.payload_parameters(), self.code_coeff]

    def clear_packed_payload_cache(self) -> None:
        self.full_lut.clear_packed_payload_cache()

    def _code_bits_output(self, margins: Tensor) -> Tensor:
        hard = (margins > 0).to(dtype=margins.dtype)
        hard_sign = hard.mul(2.0).sub(1.0)
        if self.training and (margins.requires_grad or self.thresholds.requires_grad):
            signs = hard_sign + 2.0 * (ste_heaviside(margins) - hard)
        else:
            signs = hard_sign
        group_values = signs.reshape(signs.shape[0], self.routes).matmul(self.code_coeff.to(device=margins.device, dtype=margins.dtype))
        group_values = group_values * self.code_scale
        if self.write_degree == 1:
            return group_values[:, : self.output_dim]
        return group_values.repeat_interleave(self.write_degree, dim=-1)[:, : self.output_dim]

    def compute(self, x: Tensor) -> tuple[Tensor, Tensor]:
        input_dtype = x.dtype
        x32 = x.float()
        indices, margins = self.full_lut._route(x32)
        payload = self.full_lut._lookup(indices)
        output = self.full_lut._payload_to_output(payload)
        if self.training and (x.requires_grad or self.thresholds.requires_grad):
            output = output + self.full_lut._ste_correction(indices, margins, payload)
        output = output + self._code_bits_output(margins)
        return output.to(dtype=input_dtype), indices

    def forward(self, x: Tensor) -> Tensor:
        output, _indices = self.compute(x)
        return output


class CompareSwapHiddenLayer(nn.Module):
    """Full-vector LUT plus a compare-swap geometry scaffold.

    The LUT path is exactly the full-vector payload baseline. The scaffold is a
    residual vector field built from fixed pairwise comparisons:

    * independent: a fixed non-overlapping random matching.
    * reused_anchor: a non-overlapping matching greedily extracted from the LUT anchors.
    * anchor_delta: all LUT anchors accumulate compare-swap deltas with degree normalization.

    A scalar gate starts at compare_swap_alpha_init. At zero, the layer is
    functionally identical to the full-vector LUT baseline, while the gate still
    receives gradients through the piecewise-linear compare-swap delta.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        tables: int,
        comparisons: int,
        variant: PayloadVariant,
        anchor_policy: str,
        seed: int,
        lut_init_std: float,
        compare_swap_alpha_init: float,
        compare_swap_pair_count: int,
        use_output_scaling: bool,
        use_min_margin_ste: bool,
    ) -> None:
        super().__init__()
        if input_dim != output_dim:
            raise ValueError("CompareSwapHiddenLayer requires input_dim == output_dim; use it only in hidden blocks")
        if variant not in {
            "compare_swap_independent_hidden",
            "compare_swap_reused_anchor_hidden",
            "compare_swap_anchor_delta_hidden",
        }:
            raise ValueError(f"unknown compare-swap variant {variant!r}")
        self.full_lut = PayloadWidthLUTLayer(
            input_dim,
            output_dim,
            tables=tables,
            comparisons=comparisons,
            variant="full_vector",
            anchor_policy=anchor_policy,
            seed=seed,
            lut_init_std=lut_init_std,
            write_degree=output_dim,
            use_output_scaling=use_output_scaling,
            use_min_margin_ste=use_min_margin_ste,
        )
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.tables = int(tables)
        self.comparisons = int(comparisons)
        self.table_size = 1 << int(comparisons)
        self.variant = variant
        self.payload_width = int(output_dim)
        self.write_degree = int(output_dim)
        self.output_scale = self.full_lut.output_scale
        self.use_min_margin_ste = bool(use_min_margin_ste)
        self.compare_swap_alpha = nn.Parameter(torch.tensor(float(compare_swap_alpha_init), dtype=torch.float32))

        desired_pairs = self._desired_pair_count(compare_swap_pair_count)
        if variant == "compare_swap_independent_hidden":
            pairs = self._independent_matching(desired_pairs, seed + 1009)
            norm = torch.ones(self.input_dim, dtype=torch.float32)
        elif variant == "compare_swap_reused_anchor_hidden":
            pairs = self._reused_anchor_matching(desired_pairs)
            norm = torch.ones(self.input_dim, dtype=torch.float32)
        else:
            pairs = self.full_lut.anchors.detach().cpu().reshape(-1, 2).contiguous()
            if compare_swap_pair_count > 0:
                pairs = pairs[: min(int(compare_swap_pair_count), pairs.shape[0])]
            norm = self._anchor_delta_norm(pairs)
        self.register_buffer("compare_swap_pairs", pairs.to(dtype=torch.long).contiguous())
        self.register_buffer("compare_swap_norm", norm)

    @property
    def thresholds(self) -> Tensor:
        return self.full_lut.thresholds

    @property
    def payload_params(self) -> int:
        return self.full_lut.payload_params + self.compare_swap_alpha.numel()

    @property
    def bias_generator_params(self) -> int:
        return self.payload_params

    @property
    def slope_coeff_params(self) -> int:
        return 0

    @property
    def slope_generator_params(self) -> int:
        return 0

    @property
    def compare_swap_pair_count(self) -> int:
        return int(self.compare_swap_pairs.shape[0])

    def payload_parameters(self) -> list[Tensor]:
        return [*self.full_lut.payload_parameters(), self.compare_swap_alpha]

    def clear_packed_payload_cache(self) -> None:
        self.full_lut.clear_packed_payload_cache()

    def _desired_pair_count(self, requested: int) -> int:
        if requested > 0:
            return max(1, min(int(requested), self.input_dim // 2))
        return self.input_dim // 2

    def _independent_matching(self, pair_count: int, seed: int) -> Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        perm = torch.randperm(self.input_dim, generator=generator)
        return perm[: 2 * pair_count].view(pair_count, 2).contiguous()

    def _reused_anchor_matching(self, pair_count: int) -> Tensor:
        used = torch.zeros(self.input_dim, dtype=torch.bool)
        selected: list[tuple[int, int]] = []
        for left, right in self.full_lut.anchors.detach().cpu().reshape(-1, 2).tolist():
            if left == right or used[left] or used[right]:
                continue
            used[left] = True
            used[right] = True
            selected.append((int(left), int(right)))
            if len(selected) >= pair_count:
                break
        if not selected:
            return self._independent_matching(pair_count, seed=0)
        return torch.tensor(selected, dtype=torch.long)

    def _anchor_delta_norm(self, pairs: Tensor) -> Tensor:
        if pairs.numel() == 0:
            return torch.ones(self.input_dim, dtype=torch.float32)
        counts = torch.zeros(self.input_dim, dtype=torch.float32)
        ones = torch.ones(pairs.shape[0], dtype=torch.float32)
        counts.scatter_add_(0, pairs[:, 0], ones)
        counts.scatter_add_(0, pairs[:, 1], ones)
        return counts.clamp_min(1.0).sqrt()

    def _matching_delta(self, x: Tensor) -> Tensor:
        if self.compare_swap_pairs.numel() == 0:
            return torch.zeros_like(x)
        pairs = self.compare_swap_pairs.to(device=x.device)
        left = pairs[:, 0]
        right = pairs[:, 1]
        x_left = x.index_select(-1, left)
        x_right = x.index_select(-1, right)
        high = torch.maximum(x_left, x_right)
        low = torch.minimum(x_left, x_right)
        delta = torch.zeros_like(x)
        delta.index_copy_(1, left, high - x_left)
        delta.index_copy_(1, right, low - x_right)
        return delta

    def _anchor_delta(self, x: Tensor) -> Tensor:
        if self.compare_swap_pairs.numel() == 0:
            return torch.zeros_like(x)
        pairs = self.compare_swap_pairs.to(device=x.device)
        left = pairs[:, 0]
        right = pairs[:, 1]
        x_left = x.index_select(-1, left)
        x_right = x.index_select(-1, right)
        high = torch.maximum(x_left, x_right)
        low = torch.minimum(x_left, x_right)
        updates = torch.cat((high - x_left, low - x_right), dim=-1)
        indices = torch.cat((left, right), dim=0).view(1, -1).expand(x.shape[0], -1)
        delta = torch.zeros_like(x)
        delta.scatter_add_(1, indices, updates)
        return delta / self.compare_swap_norm.to(device=x.device, dtype=x.dtype).view(1, -1)

    def _compare_swap_delta(self, x: Tensor) -> Tensor:
        if self.variant == "compare_swap_anchor_delta_hidden":
            return self._anchor_delta(x)
        return self._matching_delta(x)

    def compute(self, x: Tensor) -> tuple[Tensor, Tensor]:
        input_dtype = x.dtype
        x32 = x.float()
        indices, margins = self.full_lut._route(x32)
        payload = self.full_lut._lookup(indices)
        output = self.full_lut._payload_to_output(payload)
        if self.training and (x.requires_grad or self.thresholds.requires_grad):
            output = output + self.full_lut._ste_correction(indices, margins, payload)
        output = output + self.compare_swap_alpha.to(device=output.device, dtype=output.dtype) * self._compare_swap_delta(x32)
        return output.to(dtype=input_dtype), indices

    def forward(self, x: Tensor) -> Tensor:
        output, _indices = self.compute(x)
        return output


class FullLutGatedCorrectionLayer(nn.Module):
    """Hidden full-vector LUT plus a zero-gated route/margin correction.

    The full LUT path is unchanged. The correction path is deliberately gated by
    one scalar so gate=0 is exactly the full-vector LUT baseline, while nonzero
    correction parameters still let the gate receive gradients on the first
    update.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        tables: int,
        comparisons: int,
        variant: PayloadVariant,
        anchor_policy: str,
        seed: int,
        lut_init_std: float,
        correction_gate_init: float,
        correction_kc: int,
        correction_init_std: float,
        route_affine_pair_count: int,
        correction_write_policy: ComparatorWritePolicy,
        correction_output_tile_size: int,
        use_output_scaling: bool,
        use_min_margin_ste: bool,
    ) -> None:
        super().__init__()
        if input_dim != output_dim:
            raise ValueError("FullLutGatedCorrectionLayer requires input_dim == output_dim; use it only in hidden blocks")
        if variant not in {
            "full_lut_gated_twosided_margin_hidden",
            "full_lut_gated_signed_margin_hidden",
            "full_lut_chamber_diagonal_hidden",
            "full_lut_route_2x2_affine_hidden",
        }:
            raise ValueError(f"unknown full-LUT correction variant {variant!r}")
        if correction_kc < 1:
            raise ValueError(f"correction_kc must be >= 1, got {correction_kc}")
        if correction_init_std <= 0.0:
            raise ValueError(f"correction_init_std must be > 0, got {correction_init_std}")
        if correction_write_policy not in {"endpoint", "local-linegraph", "expander"}:
            raise ValueError(f"unknown correction write policy {correction_write_policy!r}")
        if correction_output_tile_size not in {16, 32, 64, 128}:
            raise ValueError("correction_output_tile_size must be one of 16, 32, 64, or 128")

        self.full_lut = PayloadWidthLUTLayer(
            input_dim,
            output_dim,
            tables=tables,
            comparisons=comparisons,
            variant="full_vector",
            anchor_policy=anchor_policy,
            seed=seed,
            lut_init_std=lut_init_std,
            write_degree=output_dim,
            use_output_scaling=use_output_scaling,
            use_min_margin_ste=use_min_margin_ste,
        )
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.tables = int(tables)
        self.comparisons = int(comparisons)
        self.routes = self.tables * self.comparisons
        self.table_size = 1 << int(comparisons)
        self.variant = variant
        self.payload_width = int(output_dim)
        self.write_degree = int(output_dim)
        self.output_scale = self.full_lut.output_scale
        self.correction_scale = 1.0 / math.sqrt(float(self.routes)) if use_output_scaling else 1.0
        self.use_min_margin_ste = bool(use_min_margin_ste)
        self.correction_kc = int(correction_kc)
        self.correction_gate = nn.Parameter(torch.tensor(float(correction_gate_init), dtype=torch.float32))

        if variant in {"full_lut_gated_twosided_margin_hidden", "full_lut_gated_signed_margin_hidden"}:
            sides = 2 if variant == "full_lut_gated_twosided_margin_hidden" else 1
            write_indices, write_signs = self._make_sparse_write_pattern(
                seed + 7919,
                sides=sides,
                write_policy=correction_write_policy,
                output_tile_size=correction_output_tile_size,
            )
            self.register_buffer("correction_write_indices", write_indices)
            self.correction_write_weight = nn.Parameter(write_signs / math.sqrt(float(self.correction_kc)))
        elif variant == "full_lut_chamber_diagonal_hidden":
            diag = torch.randn(self.tables, self.table_size, self.output_dim) * float(correction_init_std)
            self.chamber_diag = nn.Parameter(diag)
        else:
            pairs, route_indices = self._route_conditioned_pairs(route_affine_pair_count, seed + 1543)
            self.register_buffer("route_affine_pairs", pairs)
            self.register_buffer("route_affine_indices", route_indices)
            pair_count = int(pairs.shape[0])
            weight = torch.randn(pair_count, 2, 2, 2) * float(correction_init_std)
            bias = torch.randn(pair_count, 2, 2) * float(correction_init_std)
            self.route_affine_weight = nn.Parameter(weight)
            self.route_affine_bias = nn.Parameter(bias)

    @property
    def thresholds(self) -> Tensor:
        return self.full_lut.thresholds

    @property
    def payload_params(self) -> int:
        return self.full_lut.payload_params + sum(param.numel() for param in self._correction_parameters())

    @property
    def bias_generator_params(self) -> int:
        return self.payload_params

    @property
    def slope_coeff_params(self) -> int:
        return 0

    @property
    def slope_generator_params(self) -> int:
        return 0

    @property
    def route_affine_pair_count(self) -> int:
        return int(getattr(self, "route_affine_pairs", torch.empty(0, 2)).shape[0])

    def payload_parameters(self) -> list[Tensor]:
        return [*self.full_lut.payload_parameters(), *self._correction_parameters()]

    def clear_packed_payload_cache(self) -> None:
        self.full_lut.clear_packed_payload_cache()

    def _correction_parameters(self) -> list[Tensor]:
        params: list[Tensor] = [self.correction_gate]
        if hasattr(self, "correction_write_weight"):
            params.append(self.correction_write_weight)
        if hasattr(self, "chamber_diag"):
            params.append(self.chamber_diag)
        if hasattr(self, "route_affine_weight"):
            params.extend([self.route_affine_weight, self.route_affine_bias])
        return params

    def _make_sparse_write_pattern(
        self,
        seed: int,
        *,
        sides: int,
        write_policy: ComparatorWritePolicy,
        output_tile_size: int,
    ) -> tuple[Tensor, Tensor]:
        anchors = self.full_lut.anchors.reshape(self.routes, 2).cpu()
        if sides == 1:
            indices = torch.empty(self.routes, self.correction_kc, dtype=torch.long)
            signs = torch.empty(self.routes, self.correction_kc, dtype=torch.float32)
        else:
            indices = torch.empty(self.routes, 2, self.correction_kc, dtype=torch.long)
            signs = torch.empty(self.routes, 2, self.correction_kc, dtype=torch.float32)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        tiles = max(1, math.ceil(self.output_dim / output_tile_size))
        routes_per_tile = max(1, math.ceil(self.routes / tiles))

        for route in range(self.routes):
            a = int(anchors[route, 0].item()) % self.output_dim
            b = int(anchors[route, 1].item()) % self.output_dim
            tile = min(route // routes_per_tile, tiles - 1)
            tile_start = tile * output_tile_size
            tile_width = max(1, min(output_tile_size, self.output_dim - tile_start))
            for side in range(sides):
                virtual_route = route * sides + side
                side_sign = 1.0 if side == 0 else -1.0
                for slot in range(self.correction_kc):
                    if write_policy == "endpoint":
                        dst = a if slot % 2 == 0 else b
                        sign = side_sign if slot % 2 == 0 else -side_sign
                    elif write_policy == "local-linegraph":
                        neighbor = (virtual_route + slot // 2 + 1) % self.routes
                        na = int(anchors[neighbor, 0].item()) % self.output_dim
                        nb = int(anchors[neighbor, 1].item()) % self.output_dim
                        dst = na if slot % 2 == 0 else nb
                        sign = side_sign if slot % 2 == 0 else -side_sign
                    else:
                        hashed = (virtual_route * 1103515245 + slot * 12345 + 97) & 0x7FFFFFFF
                        jitter = int(torch.randint(0, tile_width, (1,), generator=gen).item())
                        dst = tile_start + ((hashed + jitter) % tile_width)
                        sign = 1.0 if ((hashed // max(1, tile_width)) & 1) == 0 else -1.0
                    if sides == 1:
                        indices[route, slot] = dst
                        signs[route, slot] = sign
                    else:
                        indices[route, side, slot] = dst
                        signs[route, side, slot] = sign
        return indices, signs

    def _route_conditioned_pairs(self, requested: int, seed: int) -> tuple[Tensor, Tensor]:
        desired = max(1, min(int(requested), self.input_dim // 2)) if requested > 0 else self.input_dim // 2
        used = torch.zeros(self.input_dim, dtype=torch.bool)
        selected_pairs: list[tuple[int, int]] = []
        selected_routes: list[int] = []
        for route, (left, right) in enumerate(self.full_lut.anchors.detach().cpu().reshape(-1, 2).tolist()):
            if left == right or used[left] or used[right]:
                continue
            used[left] = True
            used[right] = True
            selected_pairs.append((int(left), int(right)))
            selected_routes.append(int(route))
            if len(selected_pairs) >= desired:
                break
        if selected_pairs:
            return torch.tensor(selected_pairs, dtype=torch.long), torch.tensor(selected_routes, dtype=torch.long)

        generator = torch.Generator(device="cpu").manual_seed(seed)
        perm = torch.randperm(self.input_dim, generator=generator)
        pairs = perm[: 2 * desired].view(desired, 2).contiguous()
        route_indices = torch.arange(desired, dtype=torch.long) % self.routes
        return pairs, route_indices

    def _signed_margin_correction(self, margins: Tensor) -> Tensor:
        batch = margins.shape[0]
        values = margins.reshape(batch, self.routes)
        weighted = values.unsqueeze(-1) * self.correction_write_weight.to(device=margins.device, dtype=margins.dtype).unsqueeze(0)
        output = torch.zeros(batch, self.output_dim, device=margins.device, dtype=margins.dtype)
        indices = self.correction_write_indices.to(device=margins.device).view(1, self.routes, self.correction_kc).expand(batch, -1, -1)
        output.scatter_add_(1, indices.reshape(batch, -1), weighted.reshape(batch, -1))
        return output * self.correction_scale

    def _twosided_margin_correction(self, margins: Tensor) -> Tensor:
        batch = margins.shape[0]
        values = torch.stack((F.relu(margins), F.relu(-margins)), dim=-1).reshape(batch, self.routes, 2)
        weighted = values.unsqueeze(-1) * self.correction_write_weight.to(device=margins.device, dtype=margins.dtype).unsqueeze(0)
        output = torch.zeros(batch, self.output_dim, device=margins.device, dtype=margins.dtype)
        indices = self.correction_write_indices.to(device=margins.device).view(1, self.routes, 2, self.correction_kc).expand(batch, -1, -1, -1)
        output.scatter_add_(1, indices.reshape(batch, -1), weighted.reshape(batch, -1))
        return output * self.correction_scale

    def _chamber_diagonal_correction(self, x: Tensor, indices: Tensor) -> Tensor:
        table_offsets = torch.arange(self.tables, device=indices.device, dtype=torch.long).view(1, self.tables) * self.table_size
        flat_indices = (indices + table_offsets).reshape(-1)
        rows = self.chamber_diag.to(device=x.device, dtype=x.dtype).reshape(self.tables * self.table_size, self.output_dim).index_select(0, flat_indices)
        scale = rows.view(x.shape[0], self.tables, self.output_dim).sum(dim=1) * self.correction_scale
        return x * scale

    def _route_affine_correction(self, x: Tensor, margins: Tensor) -> Tensor:
        if self.route_affine_pair_count == 0:
            return torch.zeros_like(x)
        batch = x.shape[0]
        pairs = self.route_affine_pairs.to(device=x.device)
        route_indices = self.route_affine_indices.to(device=x.device)
        left = pairs[:, 0]
        right = pairs[:, 1]
        pair_x = torch.stack((x.index_select(-1, left), x.index_select(-1, right)), dim=-1)
        sides = (margins.reshape(batch, self.routes).index_select(1, route_indices) > 0).to(torch.long)
        pair_ids = torch.arange(pairs.shape[0], device=x.device).view(1, -1).expand(batch, -1)
        weight = self.route_affine_weight.to(device=x.device, dtype=x.dtype)[pair_ids, sides]
        bias = self.route_affine_bias.to(device=x.device, dtype=x.dtype)[pair_ids, sides]
        delta_pair = torch.einsum("bpij,bpj->bpi", weight, pair_x) + bias
        output = torch.zeros_like(x)
        output.scatter_add_(1, left.view(1, -1).expand(batch, -1), delta_pair[..., 0])
        output.scatter_add_(1, right.view(1, -1).expand(batch, -1), delta_pair[..., 1])
        return output / math.sqrt(float(max(1, pairs.shape[0])))

    def _correction(self, x: Tensor, indices: Tensor, margins: Tensor) -> Tensor:
        if self.variant == "full_lut_gated_twosided_margin_hidden":
            return self._twosided_margin_correction(margins)
        if self.variant == "full_lut_gated_signed_margin_hidden":
            return self._signed_margin_correction(margins)
        if self.variant == "full_lut_chamber_diagonal_hidden":
            return self._chamber_diagonal_correction(x, indices)
        return self._route_affine_correction(x, margins)

    def compute(self, x: Tensor) -> tuple[Tensor, Tensor]:
        input_dtype = x.dtype
        x32 = x.float()
        indices, margins = self.full_lut._route(x32)
        payload = self.full_lut._lookup(indices)
        output = self.full_lut._payload_to_output(payload)
        if self.training and (x.requires_grad or self.thresholds.requires_grad):
            output = output + self.full_lut._ste_correction(indices, margins, payload)
        gate = self.correction_gate.to(device=output.device, dtype=output.dtype)
        output = output + gate * self._correction(x32, indices, margins)
        return output.to(dtype=input_dtype), indices

    def forward(self, x: Tensor) -> Tensor:
        output, _indices = self.compute(x)
        return output


class FullCodeSparsePayloadLayer(nn.Module):
    """Full 2^C table code, but each selected cell writes only k fixed coordinates."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        tables: int,
        comparisons: int,
        anchor_policy: str,
        seed: int,
        lut_init_std: float,
        write_degree: int,
        use_output_scaling: bool,
        use_min_margin_ste: bool,
    ) -> None:
        super().__init__()
        template = PairwiseLUT(
            input_dim,
            1,
            tables=tables,
            comparisons=comparisons,
            anchor_policy=anchor_policy,
            seed=seed,
            anchor_seed=seed,
            backend="torch",
        )
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.tables = int(tables)
        self.comparisons = int(comparisons)
        self.table_size = 1 << int(comparisons)
        self.payload_width = max(1, min(int(write_degree), output_dim))
        self.write_degree = self.payload_width
        self.output_scale = 1.0 / math.sqrt(float(tables)) if use_output_scaling else 1.0
        self.use_min_margin_ste = bool(use_min_margin_ste)
        self.variant: PayloadVariant = "ladder_b_full_code_sparse_payload"
        self.register_buffer("anchors", template.anchors.detach().clone())
        self.register_buffer("powers", 2 ** torch.arange(comparisons, dtype=torch.long))
        self.register_buffer("write_indices", self._make_write_indices(seed + 379))
        self.thresholds = nn.Parameter(torch.zeros(tables, comparisons))
        self.lut = nn.Parameter(torch.randn(tables, self.table_size, self.payload_width) * lut_init_std)

    @property
    def payload_params(self) -> int:
        return self.lut.numel()

    @property
    def bias_generator_params(self) -> int:
        return self.lut.numel()

    @property
    def slope_coeff_params(self) -> int:
        return 0

    @property
    def slope_generator_params(self) -> int:
        return 0

    def payload_parameters(self) -> list[Tensor]:
        return [self.lut]

    def clear_packed_payload_cache(self) -> None:
        return None

    def _make_write_indices(self, seed: int) -> Tensor:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        base = torch.arange(self.tables * self.table_size, dtype=torch.long).view(self.tables, self.table_size, 1)
        offsets = torch.arange(self.payload_width, dtype=torch.long).view(1, 1, self.payload_width)
        jitter = torch.randint(0, max(1, self.output_dim), (self.tables, self.table_size, self.payload_width), generator=generator, dtype=torch.long)
        return (base * 1103515245 + offsets * 12345 + 97 + jitter) % self.output_dim

    def _route(self, x: Tensor) -> tuple[Tensor, Tensor]:
        anchor_a = self.anchors[:, :, 0].flatten()
        anchor_b = self.anchors[:, :, 1].flatten()
        x_a = x.index_select(-1, anchor_a).view(x.shape[0], self.tables, self.comparisons)
        x_b = x.index_select(-1, anchor_b).view(x.shape[0], self.tables, self.comparisons)
        margins = x_a - x_b - self.thresholds.to(device=x.device, dtype=x.dtype)
        bits = (margins > 0).to(torch.long)
        indices = (bits * self.powers.to(device=x.device).view(1, 1, -1)).sum(dim=-1)
        return indices, margins

    def _select_payload_and_writes(self, indices: Tensor) -> tuple[Tensor, Tensor]:
        table_offsets = torch.arange(self.tables, device=indices.device, dtype=torch.long).view(1, self.tables) * self.table_size
        flat_indices = (indices + table_offsets).reshape(-1)
        payload = self.lut.reshape(self.tables * self.table_size, self.payload_width).index_select(0, flat_indices)
        writes = self.write_indices.to(device=indices.device).reshape(self.tables * self.table_size, self.payload_width).index_select(0, flat_indices)
        return payload.view(indices.shape[0], self.tables, self.payload_width), writes.view(indices.shape[0], self.tables, self.payload_width)

    def _payload_to_output(self, payload: Tensor, writes: Tensor) -> Tensor:
        output = torch.zeros(payload.shape[0], self.output_dim, device=payload.device, dtype=payload.dtype)
        output.scatter_add_(1, writes.reshape(payload.shape[0], -1), payload.reshape(payload.shape[0], -1))
        return output * self.output_scale

    def _ste_correction(self, indices: Tensor, margins: Tensor, payload: Tensor, writes: Tensor) -> Tensor:
        if self.use_min_margin_ste:
            bit = margins.abs().argmin(dim=-1)
            margin = margins.gather(dim=-1, index=bit.unsqueeze(-1)).squeeze(-1)
            neighbor_indices = indices ^ (2 ** bit).long()
            neighbor_payload, neighbor_writes = self._select_payload_and_writes(neighbor_indices)
            ste_delta = ste_heaviside(margin) - (margin > 0).to(margin.dtype)
            return self._payload_to_output(neighbor_payload * ste_delta.unsqueeze(-1), neighbor_writes) - self._payload_to_output(
                payload * ste_delta.unsqueeze(-1), writes
            )

        correction = torch.zeros(indices.shape[0], self.output_dim, device=payload.device, dtype=payload.dtype)
        for bit_idx in range(self.comparisons):
            margin = margins[:, :, bit_idx]
            ste_delta = ste_heaviside(margin) - (margin > 0).to(margin.dtype)
            neighbor_payload, neighbor_writes = self._select_payload_and_writes(indices ^ int(self.powers[bit_idx].item()))
            correction = correction + self._payload_to_output(neighbor_payload * ste_delta.unsqueeze(-1), neighbor_writes)
            correction = correction - self._payload_to_output(payload * ste_delta.unsqueeze(-1), writes)
        return correction

    def compute(self, x: Tensor) -> tuple[Tensor, Tensor]:
        input_dtype = x.dtype
        indices, margins = self._route(x.float())
        payload, writes = self._select_payload_and_writes(indices)
        output = self._payload_to_output(payload, writes)
        if self.training and (x.requires_grad or self.thresholds.requires_grad):
            output = output + self._ste_correction(indices, margins, payload, writes)
        return output.to(dtype=input_dtype), indices

    def forward(self, x: Tensor) -> Tensor:
        output, _indices = self.compute(x)
        return output


class FullCodeMarginSparseLayer(FullCodeSparsePayloadLayer):
    """Full 2^C table code; margin magnitude provides scalar output strength."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.variant = "ladder_c_full_code_margin_sparse"
        del self.lut
        self.write_weight = nn.Parameter(_fixed_signs((self.tables, self.table_size, self.payload_width), 7919) / math.sqrt(float(self.payload_width)))

    @property
    def payload_params(self) -> int:
        return self.write_weight.numel()

    @property
    def bias_generator_params(self) -> int:
        return self.write_weight.numel()

    def payload_parameters(self) -> list[Tensor]:
        return [self.write_weight]

    def _select_weight_and_writes(self, indices: Tensor) -> tuple[Tensor, Tensor]:
        table_offsets = torch.arange(self.tables, device=indices.device, dtype=torch.long).view(1, self.tables) * self.table_size
        flat_indices = (indices + table_offsets).reshape(-1)
        weight = self.write_weight.reshape(self.tables * self.table_size, self.payload_width).index_select(0, flat_indices)
        writes = self.write_indices.to(device=indices.device).reshape(self.tables * self.table_size, self.payload_width).index_select(0, flat_indices)
        return weight.view(indices.shape[0], self.tables, self.payload_width), writes.view(indices.shape[0], self.tables, self.payload_width)

    def _ste_correction(self, indices: Tensor, margins: Tensor, payload: Tensor, writes: Tensor) -> Tensor:
        bit = margins.abs().argmin(dim=-1)
        margin = margins.gather(dim=-1, index=bit.unsqueeze(-1)).squeeze(-1)
        neighbor_indices = indices ^ (2 ** bit).long()
        neighbor_weight, neighbor_writes = self._select_weight_and_writes(neighbor_indices)
        ste_delta = ste_heaviside(margin) - (margin > 0).to(margin.dtype)
        amplitude = margins.abs().mean(dim=-1).unsqueeze(-1)
        return self._payload_to_output(neighbor_weight * amplitude * ste_delta.unsqueeze(-1), neighbor_writes) - self._payload_to_output(
            payload * ste_delta.unsqueeze(-1), writes
        )

    def compute(self, x: Tensor) -> tuple[Tensor, Tensor]:
        input_dtype = x.dtype
        indices, margins = self._route(x.float())
        weight, writes = self._select_weight_and_writes(indices)
        payload = margins.abs().mean(dim=-1).unsqueeze(-1) * weight
        output = self._payload_to_output(payload, writes)
        if self.training and (x.requires_grad or self.thresholds.requires_grad):
            output = output + self._ste_correction(indices, margins, payload, writes)
        return output.to(dtype=input_dtype), indices


class ComparatorSideFullPayloadLayer(nn.Module):
    """Independent comparator-side routes, each writing a full output vector."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        tables: int,
        comparisons: int,
        anchor_policy: str,
        seed: int,
        lut_init_std: float,
        use_output_scaling: bool,
    ) -> None:
        super().__init__()
        template = PairwiseLUT(
            input_dim,
            1,
            tables=tables,
            comparisons=comparisons,
            anchor_policy=anchor_policy,
            seed=seed,
            anchor_seed=seed,
            backend="torch",
        )
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.tables = int(tables)
        self.comparisons = int(comparisons)
        self.table_size = 2
        self.payload_width = int(output_dim)
        self.write_degree = int(output_dim)
        self.output_scale = 1.0 / math.sqrt(float(tables * comparisons)) if use_output_scaling else 1.0
        self.variant: PayloadVariant = "ladder_d_comparator_side_full_payload"
        self.register_buffer("anchors", template.anchors.detach().clone())
        self.thresholds = nn.Parameter(torch.zeros(tables, comparisons))
        self.lut = nn.Parameter(torch.randn(tables, comparisons, 2, output_dim) * lut_init_std)

    @property
    def payload_params(self) -> int:
        return self.lut.numel()

    @property
    def bias_generator_params(self) -> int:
        return self.lut.numel()

    @property
    def slope_coeff_params(self) -> int:
        return 0

    @property
    def slope_generator_params(self) -> int:
        return 0

    def payload_parameters(self) -> list[Tensor]:
        return [self.lut]

    def clear_packed_payload_cache(self) -> None:
        return None

    def _route(self, x: Tensor) -> tuple[Tensor, Tensor]:
        anchor_a = self.anchors[:, :, 0].flatten()
        anchor_b = self.anchors[:, :, 1].flatten()
        x_a = x.index_select(-1, anchor_a).view(x.shape[0], self.tables, self.comparisons)
        x_b = x.index_select(-1, anchor_b).view(x.shape[0], self.tables, self.comparisons)
        margins = x_a - x_b - self.thresholds.to(device=x.device, dtype=x.dtype)
        bits = (margins > 0).to(torch.long)
        return bits, margins

    def _select_payload(self, bits: Tensor) -> Tensor:
        table = torch.arange(self.tables, device=bits.device).view(1, self.tables, 1).expand_as(bits)
        comp = torch.arange(self.comparisons, device=bits.device).view(1, 1, self.comparisons).expand_as(bits)
        return self.lut.to(device=bits.device)[table, comp, bits]

    def compute(self, x: Tensor) -> tuple[Tensor, Tensor]:
        input_dtype = x.dtype
        bits, margins = self._route(x.float())
        payload = self._select_payload(bits)
        output = payload.sum(dim=(1, 2)) * self.output_scale
        if self.training and (x.requires_grad or self.thresholds.requires_grad):
            flipped = self._select_payload(1 - bits)
            ste_delta = ste_heaviside(margins) - bits.to(margins.dtype)
            output = output + ((flipped - payload) * ste_delta.unsqueeze(-1)).sum(dim=(1, 2)) * self.output_scale
        return output.to(dtype=input_dtype), bits.reshape(bits.shape[0], -1)

    def forward(self, x: Tensor) -> Tensor:
        output, _indices = self.compute(x)
        return output


class WalshAffinePayloadLayer(nn.Module):
    """Payload-width compatible wrapper for the low-degree Walsh affine layer."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        tables: int,
        comparisons: int,
        anchor_policy: str,
        seed: int,
        coeff_init_std: float,
        walsh_order: int,
        slope_order: int,
        slope_coeff_init_std: float,
        slope_generator_init_std: float,
        use_output_scaling: bool,
        use_min_margin_ste: bool,
        lut_dtype: LutDType,
    ) -> None:
        super().__init__()
        self.layer = PairwiseWalshLUT(
            input_dim,
            output_dim,
            tables=tables,
            comparisons=comparisons,
            walsh_order=walsh_order,  # type: ignore[arg-type]
            slope_order=slope_order,  # type: ignore[arg-type]
            seed=seed,
            coeff_init_std=coeff_init_std,
            slope_coeff_init_std=slope_coeff_init_std,
            slope_generator_init_std=slope_generator_init_std,
            use_min_margin_ste=use_min_margin_ste,
            use_output_scaling=use_output_scaling,
            anchor_policy=anchor_policy,
            anchor_seed=seed,
            lut_dtype=lut_dtype,
        )
        self.variant: PayloadVariant = "walsh_affine"
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.tables = int(tables)
        self.comparisons = int(comparisons)
        self.table_size = 1 << int(comparisons)
        self.payload_width = self.layer.walsh_term_count
        self.write_degree = self.layer.slope_term_count
        self.output_scale = self.layer.output_scale
        self.use_min_margin_ste = bool(use_min_margin_ste)
        self.lut_dtype = lut_dtype

    @property
    def thresholds(self) -> Tensor:
        return self.layer.thresholds

    @property
    def payload_params(self) -> int:
        return sum(param.numel() for param in self.payload_parameters())

    @property
    def bias_generator_params(self) -> int:
        return self.layer.constant.numel() + self.layer.linear_coeff.numel() + self.layer.pair_coeff.numel()

    @property
    def slope_coeff_params(self) -> int:
        params = [self.layer.slope_constant, self.layer.slope_linear_coeff, self.layer.slope_pair_coeff]
        return sum(0 if param is None else param.numel() for param in params)

    @property
    def slope_generator_params(self) -> int:
        return 0 if self.layer.slope_generator is None else self.layer.slope_generator.numel()

    def payload_parameters(self) -> list[Tensor]:
        return [param for name, param in self.layer.named_parameters() if name != "thresholds"]

    def clear_packed_payload_cache(self) -> None:
        self.layer.clear_packed_payload_cache()

    def compute(self, x: Tensor) -> tuple[Tensor, Tensor]:
        input_dtype = x.dtype
        output, route = self.layer.compute(x.float().unsqueeze(1), compute_dtype=torch.float32, training=self.training)
        output = self.layer._finish_output(output, route, input_dtype).squeeze(1)
        return output, route.indices.squeeze(1)

    def forward(self, x: Tensor) -> Tensor:
        output, _indices = self.compute(x)
        return output


class ComparatorGeneratorLayer(nn.Module):
    """Comparator activation layer: route values directly write through a sparse P matrix."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        tables: int,
        comparisons: int,
        source: Literal["sign", "margin", "signed_margin", "two_sided_margin"],
        write_policy: ComparatorWritePolicy,
        reduction_layout: Literal["scatter", "output_major", "tile_local"] = "scatter",
        output_tile_size: int = 32,
        k_c: int,
        anchor_policy: str,
        seed: int,
        use_output_scaling: bool,
        use_min_margin_ste: bool,
    ) -> None:
        super().__init__()
        if k_c < 1:
            raise ValueError(f"k_c must be >= 1, got {k_c}")
        if write_policy not in {"endpoint", "local-linegraph", "expander"}:
            raise ValueError(f"unknown comparator write policy {write_policy!r}")
        if reduction_layout not in {"scatter", "output_major", "tile_local"}:
            raise ValueError(f"unknown comparator reduction layout {reduction_layout!r}")
        if reduction_layout == "output_major":
            raise ValueError("output_major comparator reduction is only implemented for comparator_two_sided_margin_kc")
        if reduction_layout == "tile_local" and write_policy != "expander":
            raise ValueError("tile_local comparator reduction currently requires write_policy='expander'")
        if output_tile_size not in {16, 32, 64, 128}:
            raise ValueError("output_tile_size must be one of 16, 32, 64, or 128")
        template = PairwiseLUT(
            input_dim,
            1,
            tables=tables,
            comparisons=comparisons,
            anchor_policy=anchor_policy,
            seed=seed,
            anchor_seed=seed,
            backend="torch",
        )
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.tables = int(tables)
        self.comparisons = int(comparisons)
        self.table_size = 1 << int(comparisons)
        self.routes = self.tables * self.comparisons
        self.variant: PayloadVariant = f"comparator_{source}_kc"  # type: ignore[assignment]
        self.source = source
        self.sides = 2 if source == "two_sided_margin" else 1
        self.write_policy = write_policy
        self.reduction_layout = reduction_layout
        self.output_tile_size = int(output_tile_size)
        self.k_c = int(k_c)
        self.payload_width = int(k_c)
        self.write_degree = int(k_c * self.sides)
        self.output_scale = 1.0 / math.sqrt(self.routes) if use_output_scaling else 1.0
        self.use_min_margin_ste = bool(use_min_margin_ste)
        self.register_buffer("anchors", template.anchors.detach().clone())
        self.register_buffer("powers", 2 ** torch.arange(comparisons, dtype=torch.long))
        write_indices, write_signs = self._make_write_pattern(seed + 7919)
        self.register_buffer("write_indices", write_indices)
        self.write_weight = nn.Parameter(write_signs / math.sqrt(float(k_c)))
        self.thresholds = nn.Parameter(torch.zeros(tables, comparisons))

    @property
    def payload_params(self) -> int:
        return self.write_weight.numel()

    @property
    def bias_generator_params(self) -> int:
        return self.write_weight.numel()

    @property
    def slope_coeff_params(self) -> int:
        return 0

    @property
    def slope_generator_params(self) -> int:
        return 0

    def payload_parameters(self) -> list[Tensor]:
        return [self.write_weight]

    def clear_packed_payload_cache(self) -> None:
        return None

    def _make_write_pattern(self, seed: int) -> tuple[Tensor, Tensor]:
        if self.sides == 2:
            return self._make_two_sided_write_pattern(seed)

        anchors = self.anchors.reshape(self.routes, 2).cpu()
        indices = torch.empty(self.routes, self.k_c, dtype=torch.long)
        signs = torch.empty(self.routes, self.k_c, dtype=torch.float32)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        tiles = max(1, math.ceil(self.output_dim / self.output_tile_size))
        routes_per_tile = max(1, math.ceil(self.routes / tiles))
        for route in range(self.routes):
            a = int(anchors[route, 0].item()) % self.output_dim
            b = int(anchors[route, 1].item()) % self.output_dim
            tile = min(route // routes_per_tile, tiles - 1)
            tile_start = tile * self.output_tile_size
            tile_width = max(1, min(self.output_tile_size, self.output_dim - tile_start))
            for slot in range(self.k_c):
                if self.reduction_layout == "tile_local":
                    hashed = (route * 1103515245 + slot * 12345 + 97) & 0x7FFFFFFF
                    jitter = int(torch.randint(0, tile_width, (1,), generator=gen).item())
                    indices[route, slot] = tile_start + ((hashed + jitter) % tile_width)
                    signs[route, slot] = 1.0 if ((hashed // max(1, tile_width)) & 1) == 0 else -1.0
                elif self.write_policy == "endpoint":
                    indices[route, slot] = a if slot % 2 == 0 else b
                    signs[route, slot] = 1.0 if slot % 2 == 0 else -1.0
                elif self.write_policy == "local-linegraph":
                    neighbor = (route + slot // 2 + 1) % self.routes
                    na = int(anchors[neighbor, 0].item()) % self.output_dim
                    nb = int(anchors[neighbor, 1].item()) % self.output_dim
                    indices[route, slot] = na if slot % 2 == 0 else nb
                    signs[route, slot] = 1.0 if slot % 2 == 0 else -1.0
                else:
                    hashed = (route * 1103515245 + slot * 12345 + 97) & 0x7FFFFFFF
                    indices[route, slot] = hashed % self.output_dim
                    signs[route, slot] = 1.0 if ((hashed // max(1, self.output_dim)) & 1) == 0 else -1.0
        if self.write_policy == "expander" and self.reduction_layout != "tile_local":
            jitter = torch.randint(0, max(1, self.output_dim), (self.routes, self.k_c), generator=gen, dtype=torch.long)
            indices = (indices + jitter) % self.output_dim
        return indices, signs

    def _make_two_sided_write_pattern(self, seed: int) -> tuple[Tensor, Tensor]:
        anchors = self.anchors.reshape(self.routes, 2).cpu()
        indices = torch.empty(self.routes, 2, self.k_c, dtype=torch.long)
        signs = torch.empty(self.routes, 2, self.k_c, dtype=torch.float32)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        tiles = max(1, math.ceil(self.output_dim / self.output_tile_size))
        routes_per_tile = max(1, math.ceil(self.routes / tiles))
        for route in range(self.routes):
            a = int(anchors[route, 0].item()) % self.output_dim
            b = int(anchors[route, 1].item()) % self.output_dim
            tile = min(route // routes_per_tile, tiles - 1)
            tile_start = tile * self.output_tile_size
            tile_width = max(1, min(self.output_tile_size, self.output_dim - tile_start))
            for side in range(2):
                virtual_route = route * 2 + side
                side_sign = 1.0 if side == 0 else -1.0
                for slot in range(self.k_c):
                    if self.reduction_layout == "tile_local":
                        hashed = (virtual_route * 1103515245 + slot * 12345 + 97) & 0x7FFFFFFF
                        jitter = int(torch.randint(0, tile_width, (1,), generator=gen).item())
                        indices[route, side, slot] = tile_start + ((hashed + jitter) % tile_width)
                        signs[route, side, slot] = 1.0 if ((hashed // max(1, tile_width)) & 1) == 0 else -1.0
                    elif self.write_policy == "endpoint":
                        indices[route, side, slot] = a if slot % 2 == 0 else b
                        signs[route, side, slot] = side_sign if slot % 2 == 0 else -side_sign
                    elif self.write_policy == "local-linegraph":
                        neighbor = (virtual_route + slot // 2 + 1) % self.routes
                        na = int(anchors[neighbor, 0].item()) % self.output_dim
                        nb = int(anchors[neighbor, 1].item()) % self.output_dim
                        indices[route, side, slot] = na if slot % 2 == 0 else nb
                        signs[route, side, slot] = side_sign if slot % 2 == 0 else -side_sign
                    else:
                        hashed = (virtual_route * 1103515245 + slot * 12345 + 97) & 0x7FFFFFFF
                        indices[route, side, slot] = hashed % self.output_dim
                        signs[route, side, slot] = 1.0 if ((hashed // max(1, self.output_dim)) & 1) == 0 else -1.0
        if self.write_policy == "expander" and self.reduction_layout != "tile_local":
            jitter = torch.randint(0, max(1, self.output_dim), (self.routes, 2, self.k_c), generator=gen, dtype=torch.long)
            indices = (indices + jitter) % self.output_dim
        return indices, signs

    def _route(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        anchor_a = self.anchors[:, :, 0].flatten()
        anchor_b = self.anchors[:, :, 1].flatten()
        x_a = x.index_select(-1, anchor_a).view(x.shape[0], self.tables, self.comparisons)
        x_b = x.index_select(-1, anchor_b).view(x.shape[0], self.tables, self.comparisons)
        margins = x_a - x_b - self.thresholds.to(device=x.device, dtype=x.dtype)
        bits = (margins > 0).to(torch.long)
        indices = (bits * self.powers.to(device=x.device).view(1, 1, -1)).sum(dim=-1)
        return indices, margins, bits

    def _activation(self, margins: Tensor, bits: Tensor) -> Tensor:
        hard = bits.to(dtype=margins.dtype)
        hard_sign = hard.mul(2.0).sub(1.0)
        if self.training and (margins.requires_grad or self.thresholds.requires_grad):
            sign = hard_sign + 2.0 * (ste_heaviside(margins) - hard)
        else:
            sign = hard_sign
        if self.source == "sign":
            return sign
        if self.source == "margin":
            return margins
        if self.source == "two_sided_margin":
            return torch.stack((F.relu(margins), F.relu(-margins)), dim=-1)
        return sign * margins

    def _write_output(self, values: Tensor) -> Tensor:
        batch = values.shape[0]
        if self.sides == 2:
            flat_values = values.reshape(batch, self.routes, 2)
            weighted = flat_values.unsqueeze(-1) * self.write_weight.to(device=values.device, dtype=values.dtype).unsqueeze(0)
            output = torch.zeros(batch, self.output_dim, device=values.device, dtype=values.dtype)
            indices = self.write_indices.to(device=values.device).view(1, self.routes, 2, self.k_c).expand(batch, -1, -1, -1)
            output.scatter_add_(1, indices.reshape(batch, -1), weighted.reshape(batch, -1))
            return output * self.output_scale

        flat_values = values.reshape(batch, self.routes)
        weighted = flat_values.unsqueeze(-1) * self.write_weight.to(device=values.device, dtype=values.dtype).unsqueeze(0)
        output = torch.zeros(batch, self.output_dim, device=values.device, dtype=values.dtype)
        indices = self.write_indices.to(device=values.device).view(1, self.routes, self.k_c).expand(batch, -1, -1)
        output.scatter_add_(1, indices.reshape(batch, -1), weighted.reshape(batch, -1))
        return output * self.output_scale

    def compute(self, x: Tensor) -> tuple[Tensor, Tensor]:
        input_dtype = x.dtype
        x32 = x.float()
        indices, margins, bits = self._route(x32)
        output = self._write_output(self._activation(margins, bits))
        return output.to(dtype=input_dtype), indices

    def forward(self, x: Tensor) -> Tensor:
        output, _indices = self.compute(x)
        return output


class PayloadWidthEmnistClassifier(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        classes: int,
        depth: int,
        tables: int,
        comparisons: int,
        variant: PayloadVariant,
        anchor_policy: str,
        seed: int,
        lut_init_std: float,
        write_degree: int,
        walsh_lut_dtype: LutDType,
        walsh_order: int,
        walsh_coeff_init_std: float,
        walsh_slope_order: int,
        walsh_slope_coeff_init_std: float,
        walsh_slope_generator_init_std: float,
        residual_scale: float,
        use_output_scaling: bool,
        use_min_margin_ste: bool,
        comparator_kc: int,
        comparator_write_policy: ComparatorWritePolicy,
        comparator_reduction_layout: Literal["scatter", "output_major", "tile_local"],
        comparator_output_tile_size: int,
        ternary_threshold: float,
        compare_swap_alpha_init: float,
        compare_swap_pair_count: int,
        correction_gate_init: float,
        correction_kc: int,
        correction_init_std: float,
        route_affine_pair_count: int,
    ) -> None:
        super().__init__()
        def make_layer(input_features: int, output_features: int, layer_seed: int, *, is_hidden: bool) -> nn.Module:
            if variant == "k4_full_vector":
                return K4FullLUT(
                    input_features,
                    output_features,
                    tables=tables,
                    comparisons=comparisons,
                    anchor_policy=anchor_policy,
                    seed=layer_seed,
                    lut_init_std=lut_init_std,
                    use_output_scaling=use_output_scaling,
                    use_min_margin_ste=use_min_margin_ste,
                )
            if variant == "coxeter_full_vector":
                return CoxeterLUT(
                    input_features,
                    output_features,
                    tables=tables,
                    comparisons=comparisons,
                    anchor_policy=anchor_policy,
                    seed=layer_seed,
                    lut_init_std=lut_init_std,
                    use_output_scaling=use_output_scaling,
                    use_min_margin_ste=use_min_margin_ste,
                )
            if variant.startswith("full_lut_"):
                if is_hidden:
                    return FullLutGatedCorrectionLayer(
                        input_features,
                        output_features,
                        tables=tables,
                        comparisons=comparisons,
                        variant=variant,
                        anchor_policy=anchor_policy,
                        seed=layer_seed,
                        lut_init_std=lut_init_std,
                        correction_gate_init=correction_gate_init,
                        correction_kc=correction_kc,
                        correction_init_std=correction_init_std,
                        route_affine_pair_count=route_affine_pair_count,
                        correction_write_policy=comparator_write_policy,
                        correction_output_tile_size=comparator_output_tile_size,
                        use_output_scaling=use_output_scaling,
                        use_min_margin_ste=use_min_margin_ste,
                    )
                return PayloadWidthLUTLayer(
                    input_features,
                    output_features,
                    tables=tables,
                    comparisons=comparisons,
                    variant="full_vector",
                    anchor_policy=anchor_policy,
                    seed=layer_seed,
                    lut_init_std=lut_init_std,
                    write_degree=write_degree,
                    use_output_scaling=use_output_scaling,
                        use_min_margin_ste=use_min_margin_ste,
                    )
            if variant == "pairwise_glu_shared_route":
                return SharedRoutePairwiseGLULayer(
                    input_features,
                    output_features,
                    tables=tables,
                    comparisons=comparisons,
                    anchor_policy=anchor_policy,
                    seed=layer_seed,
                    lut_init_std=lut_init_std,
                    use_output_scaling=use_output_scaling,
                    use_min_margin_ste=use_min_margin_ste,
                )
            if variant == "pairwise_glu_dual_route":
                return DualRoutePairwiseGLULayer(
                    input_features,
                    output_features,
                    tables=tables,
                    comparisons=comparisons,
                    anchor_policy=anchor_policy,
                    seed=layer_seed,
                    lut_init_std=lut_init_std,
                    use_output_scaling=use_output_scaling,
                    use_min_margin_ste=use_min_margin_ste,
                )
            if variant == "binary_count_gated_lut":
                return BinaryCountGatedLUTLayer(
                    input_features,
                    output_features,
                    tables=tables,
                    comparisons=comparisons,
                    anchor_policy=anchor_policy,
                    seed=layer_seed,
                    lut_init_std=lut_init_std,
                    use_output_scaling=use_output_scaling,
                    use_min_margin_ste=use_min_margin_ste,
                )
            if variant.startswith("compare_swap_"):
                if is_hidden:
                    return CompareSwapHiddenLayer(
                        input_features,
                        output_features,
                        tables=tables,
                        comparisons=comparisons,
                        variant=variant,
                        anchor_policy=anchor_policy,
                        seed=layer_seed,
                        lut_init_std=lut_init_std,
                        compare_swap_alpha_init=compare_swap_alpha_init,
                        compare_swap_pair_count=compare_swap_pair_count,
                        use_output_scaling=use_output_scaling,
                        use_min_margin_ste=use_min_margin_ste,
                    )
                return PayloadWidthLUTLayer(
                    input_features,
                    output_features,
                    tables=tables,
                    comparisons=comparisons,
                    variant="full_vector",
                    anchor_policy=anchor_policy,
                    seed=layer_seed,
                    lut_init_std=lut_init_std,
                    write_degree=write_degree,
                    use_output_scaling=use_output_scaling,
                    use_min_margin_ste=use_min_margin_ste,
                )
            if variant == "code_bits_k_hidden":
                if is_hidden:
                    return CodeBitsHiddenLayer(
                        input_features,
                        output_features,
                        tables=tables,
                        comparisons=comparisons,
                        anchor_policy=anchor_policy,
                        seed=layer_seed,
                        lut_init_std=lut_init_std,
                        group_size=write_degree,
                        use_output_scaling=use_output_scaling,
                        use_min_margin_ste=use_min_margin_ste,
                    )
                return PayloadWidthLUTLayer(
                    input_features,
                    output_features,
                    tables=tables,
                    comparisons=comparisons,
                    variant="full_vector",
                    anchor_policy=anchor_policy,
                    seed=layer_seed,
                    lut_init_std=lut_init_std,
                    write_degree=write_degree,
                    use_output_scaling=use_output_scaling,
                    use_min_margin_ste=use_min_margin_ste,
                )
            if variant == "ladder_a_full_code_full_payload":
                return PayloadWidthLUTLayer(
                    input_features,
                    output_features,
                    tables=tables,
                    comparisons=comparisons,
                    variant="full_vector",
                    anchor_policy=anchor_policy,
                    seed=layer_seed,
                    lut_init_std=lut_init_std,
                    write_degree=write_degree,
                    use_output_scaling=use_output_scaling,
                    use_min_margin_ste=use_min_margin_ste,
                )
            if variant == "ladder_b_full_code_sparse_payload":
                return FullCodeSparsePayloadLayer(
                    input_features,
                    output_features,
                    tables=tables,
                    comparisons=comparisons,
                    anchor_policy=anchor_policy,
                    seed=layer_seed,
                    lut_init_std=lut_init_std,
                    write_degree=write_degree,
                    use_output_scaling=use_output_scaling,
                    use_min_margin_ste=use_min_margin_ste,
                )
            if variant == "ladder_c_full_code_margin_sparse":
                return FullCodeMarginSparseLayer(
                    input_features,
                    output_features,
                    tables=tables,
                    comparisons=comparisons,
                    anchor_policy=anchor_policy,
                    seed=layer_seed,
                    lut_init_std=lut_init_std,
                    write_degree=write_degree,
                    use_output_scaling=use_output_scaling,
                    use_min_margin_ste=use_min_margin_ste,
                )
            if variant == "ladder_d_comparator_side_full_payload":
                return ComparatorSideFullPayloadLayer(
                    input_features,
                    output_features,
                    tables=tables,
                    comparisons=comparisons,
                    anchor_policy=anchor_policy,
                    seed=layer_seed,
                    lut_init_std=lut_init_std,
                    use_output_scaling=use_output_scaling,
                )
            if variant == "ladder_e_comparator_side_sparse":
                return ComparatorTwoSidedMargin(
                    input_features,
                    output_features,
                    tables=tables,
                    comparisons=comparisons,
                    k_c=comparator_kc,
                    backend="auto",
                    write_policy=comparator_write_policy,
                    reduction_layout=comparator_reduction_layout,
                    output_tile_size=comparator_output_tile_size,
                    anchor_policy=anchor_policy,
                    seed=layer_seed,
                    use_output_scaling=use_output_scaling,
                )
            if variant == "walsh_affine":
                return WalshAffinePayloadLayer(
                    input_features,
                    output_features,
                    tables=tables,
                    comparisons=comparisons,
                    anchor_policy=anchor_policy,
                    seed=layer_seed,
                    coeff_init_std=walsh_coeff_init_std,
                    walsh_order=walsh_order,
                    slope_order=walsh_slope_order,
                    slope_coeff_init_std=walsh_slope_coeff_init_std,
                    slope_generator_init_std=walsh_slope_generator_init_std,
                    use_output_scaling=use_output_scaling,
                    use_min_margin_ste=use_min_margin_ste,
                    lut_dtype=walsh_lut_dtype,
                )
            if variant.startswith("comparator_"):
                source = variant.removeprefix("comparator_").removesuffix("_kc")
                if source not in {"sign", "margin", "signed_margin", "two_sided_margin"}:
                    raise ValueError(f"unknown comparator source from variant {variant!r}")
                if source == "two_sided_margin":
                    return ComparatorTwoSidedMargin(
                        input_features,
                        output_features,
                        tables=tables,
                        comparisons=comparisons,
                        k_c=comparator_kc,
                        backend="auto",
                        write_policy=comparator_write_policy,
                        reduction_layout=comparator_reduction_layout,
                        output_tile_size=comparator_output_tile_size,
                        anchor_policy=anchor_policy,
                        seed=layer_seed,
                        use_output_scaling=use_output_scaling,
                    )
                return ComparatorGeneratorLayer(
                    input_features,
                    output_features,
                    tables=tables,
                    comparisons=comparisons,
                    source=source,  # type: ignore[arg-type]
                    write_policy=comparator_write_policy,
                    reduction_layout=comparator_reduction_layout,
                    output_tile_size=comparator_output_tile_size,
                    k_c=comparator_kc,
                    anchor_policy=anchor_policy,
                    seed=layer_seed,
                    use_output_scaling=use_output_scaling,
                    use_min_margin_ste=use_min_margin_ste,
                )
            if variant.startswith("ternary_margin_"):
                mode = variant.removeprefix("ternary_margin_")
                if mode not in {"linear", "two_sided"}:
                    raise ValueError(f"unknown ternary margin action mode from variant {variant!r}")
                return TernaryMarginAction(
                    input_features,
                    output_features,
                    atoms=tables,
                    fan_in=comparisons,
                    mode=mode,  # type: ignore[arg-type]
                    seed=layer_seed,
                    ternary_threshold=ternary_threshold,
                    use_output_scaling=use_output_scaling,
                )
            return PayloadWidthLUTLayer(
                input_features,
                output_features,
                tables=tables,
                comparisons=comparisons,
                variant=variant,
                anchor_policy=anchor_policy,
                seed=layer_seed,
                lut_init_std=lut_init_std,
                write_degree=write_degree,
                use_output_scaling=use_output_scaling,
                use_min_margin_ste=use_min_margin_ste,
            )

        self.blocks = nn.ModuleList(
            make_layer(
                input_dim,
                input_dim,
                seed + 101 * idx,
                is_hidden=True,
            )
            for idx in range(depth)
        )
        self.readout = make_layer(
            input_dim,
            classes,
            seed + 10007,
            is_hidden=False,
        )
        self.residual_scale = float(residual_scale)
        self.last_routes: list[Tensor] = []

    def payload_layers(self) -> list[nn.Module]:
        return [*self.blocks, self.readout]

    @staticmethod
    def _route_indices(route: Tensor | PairwiseRoute) -> Tensor:
        return route.indices if isinstance(route, PairwiseRoute) else route

    def forward(self, x: Tensor) -> Tensor:
        y = x.flatten(start_dim=1).float()
        routes: list[Tensor] = []
        for block in self.blocks:
            output, indices = block.compute(y)
            y = y + self.residual_scale * output
            routes.append(self._route_indices(indices).detach())
        logits, _readout_indices = self.readout.compute(y)
        self.last_routes = routes
        return logits


def _grad_norm(params: Iterable[Tensor]) -> float:
    total = 0.0
    for param in params:
        if param.grad is None:
            continue
        norm = float(param.grad.detach().float().norm().item())
        total += norm * norm
    return math.sqrt(total)


def _payload_layers(model: PayloadWidthEmnistClassifier) -> list[nn.Module]:
    return model.payload_layers()


def _threshold_params(model: PayloadWidthEmnistClassifier) -> list[Tensor]:
    params: list[Tensor] = []
    for layer in _payload_layers(model):
        if hasattr(layer, "threshold_parameters"):
            params.extend(layer.threshold_parameters())  # type: ignore[attr-defined]
        else:
            params.append(layer.thresholds)
    return params


def _payload_params(model: PayloadWidthEmnistClassifier) -> list[Tensor]:
    params: list[Tensor] = []
    for layer in _payload_layers(model):
        params.extend(layer.payload_parameters())  # type: ignore[attr-defined]
    return params


def _count_params(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def _count_payload_params(model: PayloadWidthEmnistClassifier) -> int:
    return sum(layer.payload_params for layer in _payload_layers(model))


def _count_layer_attr(model: PayloadWidthEmnistClassifier, name: str) -> int:
    return sum(int(getattr(layer, name, 0)) for layer in _payload_layers(model))


def _compare_swap_layers(model: PayloadWidthEmnistClassifier) -> list[CompareSwapHiddenLayer]:
    return [layer for layer in model.blocks if isinstance(layer, CompareSwapHiddenLayer)]


def _compare_swap_alpha_values(model: PayloadWidthEmnistClassifier) -> list[float]:
    return [float(layer.compare_swap_alpha.detach().cpu().item()) for layer in _compare_swap_layers(model)]


def _compare_swap_alpha_mean(model: PayloadWidthEmnistClassifier) -> float:
    values = _compare_swap_alpha_values(model)
    return sum(values) / len(values) if values else 0.0


def _compare_swap_alpha_absmax(model: PayloadWidthEmnistClassifier) -> float:
    values = _compare_swap_alpha_values(model)
    return max((abs(value) for value in values), default=0.0)


def _compare_swap_pair_count(model: PayloadWidthEmnistClassifier) -> int:
    layers = _compare_swap_layers(model)
    return int(layers[0].compare_swap_pair_count) if layers else 0


def _full_lut_correction_layers(model: PayloadWidthEmnistClassifier) -> list[FullLutGatedCorrectionLayer]:
    return [layer for layer in model.blocks if isinstance(layer, FullLutGatedCorrectionLayer)]


def _correction_gate_values(model: PayloadWidthEmnistClassifier) -> list[float]:
    return [float(layer.correction_gate.detach().cpu().item()) for layer in _full_lut_correction_layers(model)]


def _correction_gate_mean(model: PayloadWidthEmnistClassifier) -> float:
    values = _correction_gate_values(model)
    return sum(values) / len(values) if values else 0.0


def _correction_gate_absmax(model: PayloadWidthEmnistClassifier) -> float:
    values = _correction_gate_values(model)
    return max((abs(value) for value in values), default=0.0)


def _route_affine_pair_count(model: PayloadWidthEmnistClassifier) -> int:
    layers = _full_lut_correction_layers(model)
    return int(layers[0].route_affine_pair_count) if layers else 0


@dataclass(frozen=True)
class EvalResult:
    loss: float
    acc: float
    route_entropy: float
    route_persistence: float


@dataclass(frozen=True)
class RefinementResult:
    unique_signatures: int
    signature_entropy: float
    connected_components: int
    boundary_density: float
    refinement_mean: float
    refinement_max: int


@torch.no_grad()
def _eval(model: PayloadWidthEmnistClassifier, loader, device: torch.device) -> EvalResult:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    route_entropies: list[float] = []
    route_persistences: list[float] = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = F.cross_entropy(logits, labels, reduction="sum")
        total_loss += float(loss.item())
        total_correct += int((logits.argmax(dim=-1) == labels).sum().item())
        total_seen += int(labels.numel())
        route_entropies.append(_route_entropy(model.last_routes, model.blocks[0].table_size if model.blocks else 1))
        route_persistences.append(_route_persistence(model.last_routes))
    model.train()
    return EvalResult(
        loss=total_loss / max(total_seen, 1),
        acc=total_correct / max(total_seen, 1),
        route_entropy=sum(route_entropies) / max(1, len(route_entropies)),
        route_persistence=sum(route_persistences) / max(1, len(route_persistences)),
    )


@torch.no_grad()
def _collect_probe_tensor(loader, *, limit: int) -> tuple[Tensor, Tensor]:
    xs: list[Tensor] = []
    ys: list[Tensor] = []
    total = 0
    for x, y in loader:
        remaining = limit - total if limit > 0 else x.shape[0]
        if remaining <= 0:
            break
        take = min(int(x.shape[0]), int(remaining))
        xs.append(x[:take].cpu())
        ys.append(y[:take].cpu())
        total += take
    if not xs:
        return torch.empty(0, 1, 28, 28), torch.empty(0, dtype=torch.long)
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


@torch.no_grad()
def _batch_signatures(model: PayloadWidthEmnistClassifier, points: Tensor, *, device: torch.device, batch_size: int) -> list[Tensor]:
    outputs: list[list[Tensor]] = []
    model.eval()
    for start in range(0, points.shape[0], batch_size):
        model(points[start : start + batch_size].to(device))
        signatures = [route.cpu().to(torch.int16).reshape(route.shape[0], -1) for route in model.last_routes]
        if not outputs:
            outputs = [[] for _ in signatures]
        for idx, signature in enumerate(signatures):
            outputs[idx].append(signature)
    return [torch.cat(parts, dim=0) for parts in outputs]


@torch.no_grad()
def _refinement_probe(model: PayloadWidthEmnistClassifier, train_loader, args: argparse.Namespace, *, device: torch.device) -> RefinementResult:
    if args.skip_refinement_probe:
        return RefinementResult(
            unique_signatures=0,
            signature_entropy=math.nan,
            connected_components=0,
            boundary_density=math.nan,
            refinement_mean=math.nan,
            refinement_max=0,
        )
    x_train, _y_train = _collect_probe_tensor(train_loader, limit=args.pca_samples)
    center, u, v = _pca_plane(x_train, limit=args.pca_samples)
    points, _uu, _vv = _grid(center, u, v, grid_size=args.grid_size, span=args.plane_span)
    signatures = _batch_signatures(model, points, device=device, batch_size=args.probe_batch_size)
    ids = _signature_ids(signatures)
    if ids.numel() == 0:
        ids = torch.zeros(points.shape[0], dtype=torch.long)
    refinement_mean, refinement_max = _refinement(signatures)
    return RefinementResult(
        unique_signatures=int(torch.unique(ids).numel()),
        signature_entropy=_entropy(ids),
        connected_components=_connected_components(ids, args.grid_size),
        boundary_density=_boundary_density(ids, args.grid_size),
        refinement_mean=refinement_mean,
        refinement_max=refinement_max,
    )


def _optimizer_name(args: argparse.Namespace) -> OptimizerName:
    return args.optimizer if args.optimizer == "adamw" else args.discrete_method


def _build_optimizers(args: argparse.Namespace, model: PayloadWidthEmnistClassifier):
    if args.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay), None
    if (
        args.payload_variant == "walsh_affine"
        or args.payload_variant == "code_bits_k_hidden"
        or args.payload_variant.startswith("compare_swap_")
        or args.payload_variant.startswith("full_lut_")
        or args.payload_variant.startswith("pairwise_glu_")
        or args.payload_variant == "binary_count_gated_lut"
        or args.payload_variant.startswith("comparator_")
        or args.payload_variant.startswith("ternary_margin_")
        or args.payload_variant.startswith("ladder_")
    ):
        raise ValueError(f"{args.payload_variant} uses generator parameters and currently supports --optimizer adamw only")

    payload_optimizer = RowLocalDiscretePayloadOptimizer(
        _payload_layers(model),
        method=DISCRETE_METHODS[args.discrete_method],
        bitwidth=2 if args.discrete_method == "bop2_ternary" else args.payload_bitwidth,
        lr=args.payload_lr if args.payload_lr is not None else DEFAULT_PAYLOAD_LR[args.discrete_method],
        step_size=args.payload_step_size,
        accumulator=args.accumulator,
        accumulator_unit=args.accumulator_unit,
        beta1=args.payload_beta1,
        beta2=args.payload_beta2,
        eps=args.payload_eps,
        bop_threshold=args.bop_threshold,
        adam_m_unit=args.adam_m_unit,
        adam_v_unit=args.adam_v_unit,
        row_frequency_normalization=args.row_frequency_normalization,
        row_frequency_decay=args.row_frequency_decay,
        row_frequency_power=args.row_frequency_power,
        row_frequency_eps=args.row_frequency_eps,
    )
    threshold_optimizer = torch.optim.AdamW(_threshold_params(model), lr=args.lr, weight_decay=args.weight_decay)
    return threshold_optimizer, payload_optimizer


def _run(args: argparse.Namespace) -> dict[str, float | int | str | bool]:
    _seed_everything(args.seed)
    args.root = args.data_root
    args.max_train = 0 if args.max_train_examples is None else args.max_train_examples
    args.max_test = 0 if args.max_test_examples is None else args.max_test_examples
    args.workers = args.num_workers
    device = torch.device(args.device)
    train_loader, test_loader, classes = _build_local_loaders(args)
    model = PayloadWidthEmnistClassifier(
        input_dim=28 * 28,
        classes=classes,
        depth=args.depth,
        tables=args.tables,
        comparisons=args.comparisons,
        variant=args.payload_variant,
        anchor_policy=args.anchor_policy,
        seed=args.seed,
        lut_init_std=args.lut_init_std,
        write_degree=args.write_degree,
        walsh_lut_dtype=args.walsh_lut_dtype,
        walsh_order=args.walsh_order,
        walsh_coeff_init_std=args.walsh_coeff_init_std,
        walsh_slope_order=args.walsh_slope_order,
        walsh_slope_coeff_init_std=args.walsh_slope_coeff_init_std,
        walsh_slope_generator_init_std=args.walsh_slope_generator_init_std,
        residual_scale=args.residual_scale,
        use_output_scaling=not args.no_output_scaling,
        use_min_margin_ste=not args.full_ste,
        comparator_kc=args.comparator_kc,
        comparator_write_policy=args.comparator_write_policy,
        comparator_reduction_layout=args.comparator_reduction_layout,
        comparator_output_tile_size=args.comparator_output_tile_size,
        ternary_threshold=args.ternary_threshold,
        compare_swap_alpha_init=args.compare_swap_alpha_init,
        compare_swap_pair_count=args.compare_swap_pair_count,
        correction_gate_init=args.correction_gate_init,
        correction_kc=args.correction_kc,
        correction_init_std=args.correction_init_std,
        route_affine_pair_count=args.route_affine_pair_count,
    ).to(device)
    main_optimizer, payload_optimizer = _build_optimizers(args, model)

    final_train_loss = math.nan
    final_train_acc = math.nan
    train_examples = _loader_examples(train_loader)
    valid_examples = _loader_examples(test_loader)
    nonfinite_batches = 0
    last_payload_grad_norm = math.nan
    last_threshold_grad_norm = math.nan
    last_commit_fraction = math.nan
    last_changed_codes = 0
    last_total_codes = 0
    last_saturation_fraction = math.nan

    for _epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            main_optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                nonfinite_batches += 1
                continue
            loss.backward()
            last_payload_grad_norm = _grad_norm(_payload_params(model))
            last_threshold_grad_norm = _grad_norm(_threshold_params(model))
            if payload_optimizer is None:
                main_optimizer.step()
            else:
                main_optimizer.step()
                stats = payload_optimizer.step()
                last_commit_fraction = stats.commit_fraction
                last_saturation_fraction = stats.saturation_fraction
                last_changed_codes = stats.changed_codes
                last_total_codes = stats.total_codes
            total_loss += float(loss.item()) * labels.numel()
            total_correct += int((logits.argmax(dim=-1) == labels).sum().item())
            total_seen += int(labels.numel())
        final_train_loss = total_loss / max(total_seen, 1)
        final_train_acc = total_correct / max(total_seen, 1)

    valid = _eval(model, test_loader, device)
    refinement = _refinement_probe(model, train_loader, args, device=device)
    first_layer = model.payload_layers()[0]
    spec = _payload_spec(args.payload_variant, 28 * 28, args.write_degree)
    is_ternary_action = isinstance(first_layer, TernaryMarginAction)
    hard_input_density = (
        float((first_layer.hard_input_codes() != 0).float().mean().item()) if is_ternary_action else math.nan
    )
    hard_direction_density = (
        float((first_layer.hard_direction_codes() != 0).float().mean().item()) if is_ternary_action else math.nan
    )
    return {
        "payload_variant": args.payload_variant,
        "payload_label": spec.payload_label,
        "optimizer": _optimizer_name(args),
        "optimizer_family": args.optimizer,
        "discrete_method": args.discrete_method if args.optimizer == "discrete" else "none",
        "depth": args.depth,
        "tables": args.tables,
        "comparisons": args.comparisons,
        "anchor_policy": args.anchor_policy,
        "payload_width": first_layer.payload_width,
        "write_degree": first_layer.write_degree,
        "walsh_lut_dtype": args.walsh_lut_dtype if args.payload_variant == "walsh_affine" else "none",
        "walsh_order": args.walsh_order if args.payload_variant == "walsh_affine" else 0,
        "walsh_slope_order": args.walsh_slope_order if args.payload_variant == "walsh_affine" else 0,
        "comparator_kc": args.comparator_kc if args.payload_variant.startswith("comparator_") or args.payload_variant == "ladder_e_comparator_side_sparse" else 0,
        "comparator_write_policy": args.comparator_write_policy if args.payload_variant.startswith("comparator_") or args.payload_variant == "ladder_e_comparator_side_sparse" else "none",
        "comparator_reduction_layout": args.comparator_reduction_layout if args.payload_variant.startswith("comparator_") or args.payload_variant == "ladder_e_comparator_side_sparse" else "none",
        "comparator_output_tile_size": args.comparator_output_tile_size if args.payload_variant.startswith("comparator_") or args.payload_variant == "ladder_e_comparator_side_sparse" else 0,
        "ternary_action_mode": first_layer.mode if is_ternary_action else "none",
        "ternary_threshold": args.ternary_threshold if is_ternary_action else 0.0,
        "hard_input_density": hard_input_density,
        "hard_direction_density": hard_direction_density,
        "semantic_route_terms": first_layer.semantic_route_terms if is_ternary_action else 0,
        "semantic_action_terms": first_layer.semantic_action_terms if is_ternary_action else 0,
        "compare_swap_alpha_init": args.compare_swap_alpha_init if args.payload_variant.startswith("compare_swap_") else 0.0,
        "compare_swap_pair_count": _compare_swap_pair_count(model),
        "compare_swap_alpha_mean": _compare_swap_alpha_mean(model),
        "compare_swap_alpha_absmax": _compare_swap_alpha_absmax(model),
        "correction_gate_init": args.correction_gate_init if args.payload_variant.startswith("full_lut_") else 0.0,
        "correction_gate_mean": _correction_gate_mean(model),
        "correction_gate_absmax": _correction_gate_absmax(model),
        "correction_kc": args.correction_kc if args.payload_variant.startswith("full_lut_") else 0,
        "correction_init_std": args.correction_init_std if args.payload_variant.startswith("full_lut_") else 0.0,
        "route_affine_pair_count": _route_affine_pair_count(model),
        "output_scale": first_layer.output_scale,
        "residual_scale": args.residual_scale,
        "train_examples": train_examples,
        "valid_examples": valid_examples,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "total_params": _count_params(model),
        "payload_params": _count_payload_params(model),
        "bias_generator_params": _count_layer_attr(model, "bias_generator_params"),
        "slope_coeff_params": _count_layer_attr(model, "slope_coeff_params"),
        "slope_generator_params": _count_layer_attr(model, "slope_generator_params"),
        "threshold_params": sum(p.numel() for p in _threshold_params(model)),
        "train_loss": final_train_loss,
        "train_acc": final_train_acc,
        "valid_loss": valid.loss,
        "valid_acc": valid.acc,
        "unique_signatures": refinement.unique_signatures,
        "signature_entropy": refinement.signature_entropy,
        "connected_components": refinement.connected_components,
        "boundary_density": refinement.boundary_density,
        "refinement_mean": refinement.refinement_mean,
        "refinement_max": refinement.refinement_max,
        "route_entropy": valid.route_entropy,
        "route_persistence": valid.route_persistence,
        "nonfinite_batches": nonfinite_batches,
        "payload_grad_norm": last_payload_grad_norm,
        "threshold_grad_norm": last_threshold_grad_norm,
        "commit_fraction": last_commit_fraction,
        "saturation_fraction": last_saturation_fraction,
        "changed_codes": last_changed_codes,
        "total_codes": last_total_codes,
        "seed": args.seed,
    }


def _write_csv(path: Path, row: dict[str, float | int | str | bool]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload-variant", choices=list(VARIANT_PAYLOAD_WIDTH), required=True)
    parser.add_argument("--optimizer", choices=["adamw", "discrete"], default="adamw")
    parser.add_argument("--discrete-method", choices=list(DISCRETE_METHODS), default="adam_ef")
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--tables", type=int, default=64)
    parser.add_argument("--comparisons", type=int, default=6)
    parser.add_argument("--anchor-policy", choices=PAIRWISE_ANCHOR_POLICIES, default="permuted")
    parser.add_argument("--write-degree", type=int, default=16)
    parser.add_argument("--walsh-lut-dtype", choices=["fp32", "bf16", "fp16", "int8", "fp8", "int4", "int2", "fp4", "nf4"], default="int2")
    parser.add_argument("--walsh-order", type=int, choices=(1, 2), default=2)
    parser.add_argument("--walsh-coeff-init-std", type=float, default=0.02)
    parser.add_argument("--walsh-slope-order", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--walsh-slope-coeff-init-std", type=float, default=0.02)
    parser.add_argument("--walsh-slope-generator-init-std", type=float, default=0.02)
    parser.add_argument("--comparator-kc", type=int, default=4)
    parser.add_argument("--comparator-write-policy", choices=["endpoint", "local-linegraph", "expander"], default="endpoint")
    parser.add_argument("--comparator-reduction-layout", choices=["scatter", "output_major", "tile_local"], default="scatter")
    parser.add_argument("--comparator-output-tile-size", type=int, choices=[16, 32, 64, 128], default=32)
    parser.add_argument("--ternary-threshold", type=float, default=0.5)
    parser.add_argument("--compare-swap-alpha-init", type=float, default=0.0)
    parser.add_argument("--compare-swap-pair-count", type=int, default=0)
    parser.add_argument("--correction-gate-init", type=float, default=0.0)
    parser.add_argument("--correction-kc", type=int, default=48)
    parser.add_argument("--correction-init-std", type=float, default=0.02)
    parser.add_argument("--route-affine-pair-count", type=int, default=0)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--lut-init-std", type=float, default=0.0)
    parser.add_argument("--full-ste", action="store_true")
    parser.add_argument("--no-output-scaling", action="store_true")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-test-examples", type=int, default=None)
    parser.add_argument("--split", default="balanced")
    parser.add_argument("--grid-size", type=int, default=96)
    parser.add_argument("--plane-span", type=float, default=3.0)
    parser.add_argument("--pca-samples", type=int, default=4096)
    parser.add_argument("--probe-batch-size", type=int, default=2048)
    parser.add_argument("--skip-refinement-probe", action="store_true")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--payload-lr", type=float, default=None)
    parser.add_argument("--payload-bitwidth", type=int, default=4)
    parser.add_argument("--payload-step-size", type=float, default=0.05)
    parser.add_argument("--accumulator", choices=["float_ef", "int_ef"], default="float_ef")
    parser.add_argument("--accumulator-unit", type=float, default=0.0002)
    parser.add_argument("--payload-beta1", type=float, default=0.9)
    parser.add_argument("--payload-beta2", type=float, default=0.999)
    parser.add_argument("--payload-eps", type=float, default=1e-8)
    parser.add_argument("--bop-threshold", type=float, default=0.1)
    parser.add_argument("--adam-m-unit", type=float, default=0.0002)
    parser.add_argument("--adam-v-unit", type=float, default=0.0002)
    parser.add_argument("--row-frequency-normalization", action="store_true")
    parser.add_argument("--row-frequency-decay", type=float, default=0.99)
    parser.add_argument("--row-frequency-power", type=float, default=0.5)
    parser.add_argument("--row-frequency-eps", type=float, default=1e-6)
    parser.add_argument("--out", type=Path, default=Path("results/payload_width/emnist_balanced/result.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    row = _run(args)
    _write_csv(args.out, row)
    print(
        "payload={payload_variant} optimizer={optimizer} valid_loss={valid_loss:.6f} "
        "valid_acc={valid_acc:.6f} params={total_params} payload_params={payload_params}".format(**row),
        flush=True,
    )


if __name__ == "__main__":
    main()
