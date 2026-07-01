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

from tropnn.layers.pairwise import PAIRWISE_ANCHOR_POLICIES, LutDType, PairwiseLUT, PairwiseWalshLUT, ste_heaviside
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
        if self.variant == "walsh_affine":
            return "walsh_affine"
        if self.variant.startswith("comparator_"):
            return self.variant
        if self.variant == "full_vector":
            return "full_vector"
        if self.variant.startswith("group_"):
            return f"group_k{self.payload_width}"
        if self.variant == "scalar_expander":
            return f"scalar_expander_w{self.write_degree}"
        if self.variant == "scalar_sign":
            return "scalar_sign_basis"
        return "scalar_k1"


def _payload_spec(variant: PayloadVariant, output_dim: int, write_degree: int) -> PayloadSpec:
    if variant == "walsh_affine" or variant.startswith("comparator_"):
        return PayloadSpec(variant=variant, payload_width=0, write_degree=0, dense_sign_basis=False)
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
        for route in range(self.routes):
            a = int(anchors[route, 0].item()) % self.output_dim
            b = int(anchors[route, 1].item()) % self.output_dim
            for slot in range(self.k_c):
                if self.write_policy == "endpoint":
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
        if self.write_policy == "expander":
            jitter = torch.randint(0, max(1, self.output_dim), (self.routes, self.k_c), generator=gen, dtype=torch.long)
            indices = (indices + jitter) % self.output_dim
        return indices, signs

    def _make_two_sided_write_pattern(self, seed: int) -> tuple[Tensor, Tensor]:
        anchors = self.anchors.reshape(self.routes, 2).cpu()
        indices = torch.empty(self.routes, 2, self.k_c, dtype=torch.long)
        signs = torch.empty(self.routes, 2, self.k_c, dtype=torch.float32)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        for route in range(self.routes):
            a = int(anchors[route, 0].item()) % self.output_dim
            b = int(anchors[route, 1].item()) % self.output_dim
            for side in range(2):
                virtual_route = route * 2 + side
                side_sign = 1.0 if side == 0 else -1.0
                for slot in range(self.k_c):
                    if self.write_policy == "endpoint":
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
        if self.write_policy == "expander":
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
    ) -> None:
        super().__init__()
        def make_layer(input_features: int, output_features: int, layer_seed: int) -> nn.Module:
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
                return ComparatorGeneratorLayer(
                    input_features,
                    output_features,
                    tables=tables,
                    comparisons=comparisons,
                    source=source,  # type: ignore[arg-type]
                    write_policy=comparator_write_policy,
                    k_c=comparator_kc,
                    anchor_policy=anchor_policy,
                    seed=layer_seed,
                    use_output_scaling=use_output_scaling,
                    use_min_margin_ste=use_min_margin_ste,
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
            )
            for idx in range(depth)
        )
        self.readout = make_layer(
            input_dim,
            classes,
            seed + 10007,
        )
        self.residual_scale = float(residual_scale)
        self.last_routes: list[Tensor] = []

    def payload_layers(self) -> list[nn.Module]:
        return [*self.blocks, self.readout]

    def forward(self, x: Tensor) -> Tensor:
        y = x.flatten(start_dim=1).float()
        routes: list[Tensor] = []
        for block in self.blocks:
            output, indices = block.compute(y)
            y = y + self.residual_scale * output
            routes.append(indices.detach())
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
    return [layer.thresholds for layer in _payload_layers(model)]


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
    if args.payload_variant == "walsh_affine" or args.payload_variant.startswith("comparator_"):
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
        "comparator_kc": args.comparator_kc if args.payload_variant.startswith("comparator_") else 0,
        "comparator_write_policy": args.comparator_write_policy if args.payload_variant.startswith("comparator_") else "none",
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
