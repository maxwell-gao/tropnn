from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..backend import Backend
from .pairwise import PAIRWISE_ANCHOR_POLICIES, PairwiseRoute, _compute_dtype_for_lut, _make_pairwise_anchors

ComparatorWritePolicy = Literal[
    "endpoint",
    "local-linegraph",
    "expander",
    "expander_unique",
]
ComparatorReductionLayout = Literal[
    "scatter",
    "output_major",
    "tile_local",
    "dense_training",
]
ComparatorWeightInit = Literal["signed", "zero"]
ComparatorWeightMode = Literal["float", "binary", "ternary"]

__all__ = [
    "ComparatorReductionLayout",
    "ComparatorTwoSidedMargin",
    "ComparatorWeightInit",
    "ComparatorWeightMode",
    "ComparatorWritePolicy",
]


@dataclass(frozen=True)
class ComparatorTwoSidedMarginSpec:
    input_dim: int
    output_dim: int
    tables: int
    comparisons: int
    k_c: int
    backend: Backend
    anchor_policy: str
    anchor_seed: int
    write_policy: ComparatorWritePolicy
    reduction_layout: ComparatorReductionLayout
    output_tile_size: int
    weight_init: ComparatorWeightInit
    weight_mode: ComparatorWeightMode
    ternary_threshold: float
    use_output_scaling: bool

    def __post_init__(self) -> None:
        if self.input_dim < 2 or self.output_dim < 1:
            raise ValueError("input_dim must be >= 2 and output_dim must be positive")
        if self.tables < 1 or self.comparisons < 1 or self.k_c < 1:
            raise ValueError("tables, comparisons, and k_c must be positive")
        if self.backend not in {"auto", "torch", "triton"}:
            raise ValueError(f"unsupported backend {self.backend!r}")
        if self.anchor_policy not in PAIRWISE_ANCHOR_POLICIES:
            raise ValueError(f"unsupported anchor policy {self.anchor_policy!r}")
        if self.write_policy not in {
            "endpoint",
            "local-linegraph",
            "expander",
            "expander_unique",
        }:
            raise ValueError(f"unsupported write policy {self.write_policy!r}")
        if self.write_policy == "expander_unique" and self.k_c > self.output_dim:
            raise ValueError("write_policy='expander_unique' requires k_c <= output_dim")
        if self.reduction_layout not in {
            "scatter",
            "output_major",
            "tile_local",
            "dense_training",
        }:
            raise ValueError(f"unsupported reduction_layout {self.reduction_layout!r}")
        if self.reduction_layout == "tile_local" and self.write_policy != "expander":
            raise ValueError("tile_local reduction currently requires write_policy='expander'")
        if self.reduction_layout == "dense_training" and self.backend == "triton":
            raise ValueError(
                "dense_training materializes the fixed sparse action as a dense "
                "matrix and requires backend='torch' or 'auto'"
            )
        if self.output_tile_size not in {16, 32, 64, 128}:
            raise ValueError("output_tile_size must be one of 16, 32, 64, or 128")
        if self.weight_init not in {"signed", "zero"}:
            raise ValueError(f"unsupported weight_init {self.weight_init!r}")
        if self.weight_mode not in {"float", "binary", "ternary"}:
            raise ValueError(f"unsupported weight_mode {self.weight_mode!r}")
        if self.weight_mode == "binary" and self.weight_init == "zero":
            raise ValueError("binary write weights require weight_init='signed'")
        if not 0.0 < self.ternary_threshold < 1.0:
            raise ValueError("ternary_threshold must be in (0, 1)")

    @property
    def routes(self) -> int:
        return self.tables * self.comparisons

    @property
    def output_scale(self) -> float:
        return 1.0 / math.sqrt(float(self.routes)) if self.use_output_scaling else 1.0


class ComparatorTwoSidedMargin(nn.Module):
    """Comparator-local two-sided hinge generators with sparse expander writes."""

    is_comparator_margin_generator = True

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        tables: int = 64,
        comparisons: int = 6,
        k_c: int = 48,
        backend: Backend = "auto",
        seed: int = 0,
        anchor_policy: str = "permuted",
        anchor_seed: int | None = None,
        write_policy: ComparatorWritePolicy = "expander",
        reduction_layout: ComparatorReductionLayout = "scatter",
        output_tile_size: int = 32,
        weight_init: ComparatorWeightInit = "signed",
        weight_mode: ComparatorWeightMode = "float",
        ternary_threshold: float = 0.5,
        use_output_scaling: bool = True,
        fixed_zero_threshold: bool = False,
    ) -> None:
        super().__init__()
        spec = ComparatorTwoSidedMarginSpec(
            int(input_dim),
            int(output_dim),
            int(tables),
            int(comparisons),
            int(k_c),
            backend,
            anchor_policy,
            seed if anchor_seed is None else int(anchor_seed),
            write_policy,
            reduction_layout,
            int(output_tile_size),
            weight_init,
            weight_mode,
            float(ternary_threshold),
            bool(use_output_scaling),
        )
        self.spec = spec
        self.input_dim = spec.input_dim
        self.output_dim = spec.output_dim
        self.tables = spec.tables
        self.comparisons = spec.comparisons
        self.routes = spec.routes
        self.k_c = spec.k_c
        self.backend = spec.backend
        self.anchor_policy = spec.anchor_policy
        self.write_policy = spec.write_policy
        self.reduction_layout = spec.reduction_layout
        self.output_tile_size = spec.output_tile_size
        self.weight_init = spec.weight_init
        self.weight_mode = spec.weight_mode
        self.ternary_threshold = spec.ternary_threshold
        self.output_scale = spec.output_scale
        self.payload_width = spec.k_c
        self.write_degree = 2 * spec.k_c
        self.table_size = 2

        anchors = _make_pairwise_anchors(spec.input_dim, spec.tables, spec.comparisons, policy=spec.anchor_policy, seed=spec.anchor_seed)
        self.register_buffer("anchors", anchors)
        self.register_buffer("powers", 2 ** torch.arange(spec.comparisons, dtype=torch.long))
        write_indices, write_signs = self._make_write_pattern(seed + 7919)
        self.register_buffer("write_indices", write_indices)
        csr_offsets, csr_sources, csr_weight_indices, csr_max_degree = self._make_output_major_layout(write_indices)
        self.register_buffer("csr_offsets", csr_offsets)
        self.register_buffer("csr_sources", csr_sources)
        self.register_buffer("csr_weight_indices", csr_weight_indices)
        self.csr_max_degree = csr_max_degree
        if spec.weight_init == "zero":
            write_weight = torch.zeros_like(write_signs)
        elif spec.weight_mode in {"binary", "ternary"}:
            write_weight = write_signs * math.atanh(0.75)
        else:
            write_weight = write_signs / math.sqrt(float(spec.k_c))
        self.write_weight = nn.Parameter(write_weight)
        initial_write_codes = (
            write_signs.to(torch.int8) if spec.weight_mode in {"binary", "ternary"} else torch.empty(0, dtype=torch.int8)
        )
        self.register_buffer("initial_write_codes", initial_write_codes)

        thresholds = torch.zeros(spec.tables, spec.comparisons)
        self.register_buffer("thresholds", thresholds) if fixed_zero_threshold else setattr(self, "thresholds", nn.Parameter(thresholds))

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

    @property
    def binary_weight_scale(self) -> float:
        return 1.0 / math.sqrt(float(self.k_c))

    def hard_write_codes(self) -> Tensor:
        if self.weight_mode not in {"binary", "ternary"}:
            raise RuntimeError("hard_write_codes is defined only for quantized write weights")
        bounded = torch.tanh(self.write_weight.detach())
        if self.weight_mode == "binary":
            hard = torch.where(bounded >= 0, torch.ones_like(bounded), -torch.ones_like(bounded))
        else:
            hard = torch.where(
                bounded > self.ternary_threshold,
                torch.ones_like(bounded),
                torch.where(
                    bounded < -self.ternary_threshold,
                    -torch.ones_like(bounded),
                    torch.zeros_like(bounded),
                ),
            )
        return hard.to(torch.int8)

    def quantized_code_change_fraction(self) -> float:
        if self.weight_mode not in {"binary", "ternary"}:
            return math.nan
        return float((self.hard_write_codes().cpu() != self.initial_write_codes.cpu()).float().mean().item())

    def quantized_code_zero_fraction(self) -> float:
        if self.weight_mode not in {"binary", "ternary"}:
            return math.nan
        return float((self.hard_write_codes() == 0).float().mean().item())

    def binary_code_flip_fraction(self) -> float:
        if self.weight_mode != "binary":
            return math.nan
        return self.quantized_code_change_fraction()

    def _materialized_write_weight(self) -> Tensor:
        if self.weight_mode == "float":
            return self.write_weight
        bounded = torch.tanh(self.write_weight)
        if self.weight_mode == "binary":
            hard = torch.where(bounded >= 0, torch.ones_like(bounded), -torch.ones_like(bounded))
        else:
            hard = torch.where(
                bounded > self.ternary_threshold,
                torch.ones_like(bounded),
                torch.where(
                    bounded < -self.ternary_threshold,
                    -torch.ones_like(bounded),
                    torch.zeros_like(bounded),
                ),
            )
        ste = bounded + (hard - bounded).detach()
        return ste * self.binary_weight_scale

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, output_dim={self.output_dim}, tables={self.tables}, "
            f"comparisons={self.comparisons}, k_c={self.k_c}, backend={self.backend!r}, "
            f"anchor_policy={self.anchor_policy!r}, write_policy={self.write_policy!r}, "
            f"reduction_layout={self.reduction_layout!r}, output_tile_size={self.output_tile_size}, "
            f"weight_init={self.weight_init!r}, weight_mode={self.weight_mode!r}, "
            f"ternary_threshold={self.ternary_threshold}"
        )

    def _make_write_pattern(self, seed: int) -> tuple[Tensor, Tensor]:
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
                if self.write_policy == "expander_unique":
                    indices[route, side] = torch.randperm(self.output_dim, generator=gen)[: self.k_c]
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
                    elif self.write_policy == "expander_unique":
                        hashed = (virtual_route * 1103515245 + slot * 12345 + 97) & 0x7FFFFFFF
                        signs[route, side, slot] = 1.0 if ((hashed // max(1, self.output_dim)) & 1) == 0 else -1.0
                    else:
                        hashed = (virtual_route * 1103515245 + slot * 12345 + 97) & 0x7FFFFFFF
                        indices[route, side, slot] = hashed % self.output_dim
                        signs[route, side, slot] = 1.0 if ((hashed // max(1, self.output_dim)) & 1) == 0 else -1.0
        if self.write_policy == "expander" and self.reduction_layout != "tile_local":
            jitter = torch.randint(0, max(1, self.output_dim), (self.routes, 2, self.k_c), generator=gen, dtype=torch.long)
            indices = (indices + jitter) % self.output_dim
        return indices, signs

    def _make_output_major_layout(self, write_indices: Tensor) -> tuple[Tensor, Tensor, Tensor, int]:
        buckets: list[list[tuple[int, int]]] = [[] for _ in range(self.output_dim)]
        flat_indices = write_indices.reshape(self.routes * 2, self.k_c).cpu()
        for source in range(self.routes * 2):
            for slot in range(self.k_c):
                dst = int(flat_indices[source, slot].item())
                buckets[dst].append((source, source * self.k_c + slot))

        offsets = torch.empty(self.output_dim + 1, dtype=torch.long)
        sources: list[int] = []
        weight_indices: list[int] = []
        cursor = 0
        max_degree = 0
        offsets[0] = 0
        for dst, bucket in enumerate(buckets):
            max_degree = max(max_degree, len(bucket))
            for source, weight_idx in bucket:
                sources.append(source)
                weight_indices.append(weight_idx)
            cursor += len(bucket)
            offsets[dst + 1] = cursor

        source_tensor = torch.tensor(sources, dtype=torch.long)
        weight_index_tensor = torch.tensor(weight_indices, dtype=torch.long)
        return offsets, source_tensor, weight_index_tensor, max(1, max_degree)

    def _route(self, x_flat: Tensor) -> PairwiseRoute:
        a = self.anchors[:, :, 0].flatten()
        b = self.anchors[:, :, 1].flatten()
        margins = x_flat.index_select(-1, a).view(x_flat.shape[0], self.tables, self.comparisons)
        margins = margins - x_flat.index_select(-1, b).view(x_flat.shape[0], self.tables, self.comparisons)
        margins = margins - self.thresholds.to(device=x_flat.device, dtype=x_flat.dtype).view(1, self.tables, self.comparisons)
        indices = ((margins > 0).to(torch.long) * self.powers.to(device=x_flat.device).view(1, 1, -1)).sum(dim=-1)
        return PairwiseRoute(indices, margins)

    def _torch_output(self, margins: Tensor, *, compute_dtype: torch.dtype) -> Tensor:
        items = margins.shape[0]
        values = torch.stack((F.relu(margins), F.relu(-margins)), dim=-1).reshape(items, self.routes, 2)
        weighted = values.unsqueeze(-1) * self._materialized_write_weight().to(
            device=margins.device, dtype=compute_dtype
        ).unsqueeze(0)
        output = torch.zeros(items, self.output_dim, device=margins.device, dtype=compute_dtype)
        indices = self.write_indices.to(device=margins.device).view(1, self.routes, 2, self.k_c).expand(items, -1, -1, -1)
        output.scatter_add_(1, indices.reshape(items, -1), weighted.reshape(items, -1))
        return output * self.output_scale

    def _dense_training_output(self, margins: Tensor, *, compute_dtype: torch.dtype) -> Tensor:
        """Evaluate the same sparse action through a materialized dense matrix.

        This spelling exists to make large-batch GPU training practical.  The
        fixed ``write_indices`` still define the learned sparse function and
        gradients flow only to ``write_weight``.  Materializing its equivalent
        ``[2 * routes, output_dim]`` matrix merely changes reduction order and
        execution hardware; it is not a sparse deployment-throughput claim.
        """
        items = margins.shape[0]
        device_type = margins.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            margins_compute = margins.to(dtype=compute_dtype)
            values = torch.stack(
                (F.relu(margins_compute), F.relu(-margins_compute)),
                dim=-1,
            ).reshape(items, self.routes * 2)
            indices = self.write_indices.to(device=margins.device).reshape(
                self.routes * 2,
                self.k_c,
            )
            weights = self._materialized_write_weight().to(
                device=margins.device,
                dtype=compute_dtype,
            ).reshape(self.routes * 2, self.k_c)
            dense_action = torch.zeros(
                self.routes * 2,
                self.output_dim,
                device=margins.device,
                dtype=compute_dtype,
            ).scatter_add(1, indices, weights)
            return (values @ dense_action) * self.output_scale

    def _triton_output(self, x_flat: Tensor) -> tuple[Tensor, Tensor]:
        from ..backends.comparator_margin_triton import (
            comparator_two_sided_margin_output_major_triton,
            comparator_two_sided_margin_tile_local_triton,
            comparator_two_sided_margin_triton,
        )

        materialized_write_weight = self._materialized_write_weight()
        if self.reduction_layout == "output_major":
            return comparator_two_sided_margin_output_major_triton(
                x_flat.contiguous().float(),
                self.anchors.to(device=x_flat.device),
                self.thresholds.to(device=x_flat.device, dtype=torch.float32),
                self.write_indices.to(device=x_flat.device),
                materialized_write_weight.to(device=x_flat.device),
                self.csr_offsets.to(device=x_flat.device),
                self.csr_sources.to(device=x_flat.device),
                self.csr_weight_indices.to(device=x_flat.device),
                csr_max_degree=int(self.csr_max_degree),
                output_dim=self.output_dim,
                output_scale=self.output_scale,
            )
        if self.reduction_layout == "tile_local":
            return comparator_two_sided_margin_tile_local_triton(
                x_flat.contiguous().float(),
                self.anchors.to(device=x_flat.device),
                self.thresholds.to(device=x_flat.device, dtype=torch.float32),
                self.write_indices.to(device=x_flat.device),
                materialized_write_weight.to(device=x_flat.device),
                output_tile_size=int(self.output_tile_size),
                output_dim=self.output_dim,
                output_scale=self.output_scale,
            )

        return comparator_two_sided_margin_triton(
            x_flat.contiguous().float(),
            self.anchors.to(device=x_flat.device),
            self.thresholds.to(device=x_flat.device, dtype=torch.float32),
            self.write_indices.to(device=x_flat.device),
            materialized_write_weight.to(device=x_flat.device),
            output_dim=self.output_dim,
            output_scale=self.output_scale,
        )

    def compute(self, x: Tensor, *, compute_dtype: torch.dtype | None = None, training: bool | None = None) -> tuple[Tensor, PairwiseRoute]:
        del training
        input_dtype = x.dtype
        if x.ndim == 0 or x.shape[-1] != self.input_dim:
            raise ValueError(f"ComparatorTwoSidedMargin expected last dimension {self.input_dim}, got shape {tuple(x.shape)}")
        prefix = x.shape[:-1]
        x_flat = x.reshape(-1, self.input_dim)
        compute_dtype = compute_dtype if compute_dtype is not None else _compute_dtype_for_lut(x, "fp32")
        if self.reduction_layout == "dense_training":
            route_flat = self._route(x_flat.to(compute_dtype))
            output_flat = self._dense_training_output(
                route_flat.margins,
                compute_dtype=compute_dtype,
            )
            route = PairwiseRoute(
                route_flat.indices.view(*prefix, self.tables),
                route_flat.margins.view(*prefix, self.tables, self.comparisons),
            )
            return output_flat.view(*prefix, self.output_dim).to(dtype=input_dtype), route
        backend = "triton" if self.backend == "auto" and x.is_cuda else ("torch" if self.backend == "auto" else self.backend)
        if backend == "triton":
            output_flat, margins_flat = self._triton_output(x_flat)
            margins = margins_flat.view(*prefix, self.tables, self.comparisons)
            indices = ((margins > 0).to(torch.long) * self.powers.to(device=x.device).view(*([1] * len(prefix)), 1, -1)).sum(dim=-1)
            return output_flat.view(*prefix, self.output_dim).to(dtype=input_dtype), PairwiseRoute(indices, margins)
        route_flat = self._route(x_flat.to(compute_dtype))
        output_flat = self._torch_output(route_flat.margins, compute_dtype=compute_dtype)
        route = PairwiseRoute(route_flat.indices.view(*prefix, self.tables), route_flat.margins.view(*prefix, self.tables, self.comparisons))
        return output_flat.view(*prefix, self.output_dim).to(dtype=input_dtype), route

    def forward(self, x: Tensor) -> Tensor:
        output, _route = self.compute(x)
        return output
