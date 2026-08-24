from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .pairwise import PAIRWISE_ANCHOR_POLICIES, PairwiseRoute, _make_pairwise_anchors
from .surrogate import ste_heaviside

__all__ = ["HashSelectedSparseHinge"]


class HashSelectedSparseHinge(nn.Module):
    """Use a cheap joint comparison hash to shortlist sparse live ridge atoms.

    Each table computes one ``comparisons``-bit PC-LUT code.  The selected code
    activates ``candidates`` sparse, task-trainable affine margins.  Positive
    margin amplitudes drive sparse learned writes:

    ``sum[t, k] write[t, code_t, k] * relu(read[t, code_t, k] @ x - beta)``.

    The hard hash is only a candidate selector.  Inside a fixed hash chamber,
    the learned action still depends on the current input amplitude.  Fixed
    read/write supports keep the active arithmetic and memory traffic explicit.
    """

    is_hash_selected_sparse_hinge = True

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        tables: int = 64,
        comparisons: int = 6,
        candidates: int = 4,
        margin_fan_in: int = 8,
        write_fan_out: int = 16,
        seed: int = 0,
        anchor_policy: str = "permuted",
        anchor_seed: int | None = None,
        use_output_scaling: bool = True,
        use_min_margin_ste: bool = True,
        fixed_zero_hash_threshold: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.tables = int(tables)
        self.comparisons = int(comparisons)
        self.candidates = int(candidates)
        self.margin_fan_in = int(margin_fan_in)
        self.write_fan_out = int(write_fan_out)
        self.anchor_policy = str(anchor_policy)
        self.use_min_margin_ste = bool(use_min_margin_ste)
        self.table_size = 1 << self.comparisons
        self.payload_width = self.write_fan_out
        self.write_degree = self.candidates * self.write_fan_out
        self.output_scale = (
            1.0 / math.sqrt(float(self.tables * self.candidates))
            if use_output_scaling
            else 1.0
        )

        if self.input_dim < 1 or self.output_dim < 1:
            raise ValueError("input_dim and output_dim must be positive")
        if self.tables < 1 or self.comparisons < 1 or self.candidates < 1:
            raise ValueError("tables, comparisons, and candidates must be positive")
        if not 1 <= self.margin_fan_in <= self.input_dim:
            raise ValueError("margin_fan_in must be in [1, input_dim]")
        if not 1 <= self.write_fan_out <= self.output_dim:
            raise ValueError("write_fan_out must be in [1, output_dim]")
        if self.anchor_policy not in PAIRWISE_ANCHOR_POLICIES:
            raise ValueError(
                f"unsupported anchor_policy {self.anchor_policy!r}; "
                f"choices={PAIRWISE_ANCHOR_POLICIES}"
            )
        if not self.use_min_margin_ste:
            raise ValueError(
                "HashSelectedSparseHinge currently supports min-margin route STE only"
            )

        resolved_anchor_seed = seed if anchor_seed is None else int(anchor_seed)
        anchors = _make_pairwise_anchors(
            self.input_dim,
            self.tables,
            self.comparisons,
            policy=self.anchor_policy,
            seed=resolved_anchor_seed,
        )
        self.register_buffer("anchors", anchors)
        self.register_buffer(
            "powers",
            2 ** torch.arange(self.comparisons, dtype=torch.long),
        )

        generator = torch.Generator(device="cpu").manual_seed(int(seed) + 104729)
        self.register_buffer(
            "read_indices",
            self._make_sparse_supports(
                width=self.input_dim,
                fan_out=self.margin_fan_in,
                generator=generator,
            ),
        )
        self.register_buffer(
            "write_indices",
            self._make_sparse_supports(
                width=self.output_dim,
                fan_out=self.write_fan_out,
                generator=generator,
            ),
        )

        program_shape = (self.tables, self.table_size, self.candidates)
        read_signs = torch.randint(
            0,
            2,
            (*program_shape, self.margin_fan_in),
            generator=generator,
            dtype=torch.int8,
        ).to(torch.float32).mul_(2.0).sub_(1.0)
        write_signs = torch.randint(
            0,
            2,
            (*program_shape, self.write_fan_out),
            generator=generator,
            dtype=torch.int8,
        ).to(torch.float32).mul_(2.0).sub_(1.0)
        self.read_weight = nn.Parameter(
            read_signs / math.sqrt(float(self.margin_fan_in))
        )
        self.margin_thresholds = nn.Parameter(torch.zeros(program_shape))
        self.write_weight = nn.Parameter(
            write_signs / math.sqrt(float(self.write_fan_out))
        )

        hash_thresholds = torch.zeros(self.tables, self.comparisons)
        if fixed_zero_hash_threshold:
            self.register_buffer("thresholds", hash_thresholds)
        else:
            self.thresholds = nn.Parameter(hash_thresholds)

    @property
    def active_margin_count(self) -> int:
        return self.tables * self.candidates

    @property
    def candidate_bank_size(self) -> int:
        return self.tables * self.table_size * self.candidates

    @property
    def semantic_route_terms(self) -> int:
        return self.tables * self.comparisons

    @property
    def semantic_action_terms(self) -> int:
        return self.tables * self.candidates * (
            self.margin_fan_in + self.write_fan_out
        )

    @property
    def support_index_count(self) -> int:
        return self.candidate_bank_size * (
            self.margin_fan_in + self.write_fan_out
        )

    @property
    def payload_params(self) -> int:
        return (
            self.read_weight.numel()
            + self.margin_thresholds.numel()
            + self.write_weight.numel()
        )

    @property
    def bias_generator_params(self) -> int:
        return self.margin_thresholds.numel()

    @property
    def slope_coeff_params(self) -> int:
        return self.read_weight.numel()

    @property
    def slope_generator_params(self) -> int:
        return self.write_weight.numel()

    def payload_parameters(self) -> list[Tensor]:
        return [self.read_weight, self.write_weight]

    def threshold_parameters(self) -> list[Tensor]:
        parameters: list[Tensor] = [self.margin_thresholds]
        if isinstance(self.thresholds, nn.Parameter):
            parameters.insert(0, self.thresholds)
        return parameters

    def clear_packed_payload_cache(self) -> None:
        return None

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, output_dim={self.output_dim}, "
            f"tables={self.tables}, comparisons={self.comparisons}, "
            f"candidates={self.candidates}, margin_fan_in={self.margin_fan_in}, "
            f"write_fan_out={self.write_fan_out}, "
            f"anchor_policy={self.anchor_policy!r}"
        )

    def _make_sparse_supports(
        self,
        *,
        width: int,
        fan_out: int,
        generator: torch.Generator,
    ) -> Tensor:
        # One base permutation per hash cell gives every candidate a support
        # without replacement.  Supports may overlap across candidates only
        # when candidates * fan_out exceeds the carrier width.
        scores = torch.rand(
            self.tables,
            self.table_size,
            width,
            generator=generator,
        )
        permutation = scores.argsort(dim=-1)
        offsets = (
            torch.arange(self.candidates).view(self.candidates, 1) * fan_out
            + torch.arange(fan_out).view(1, fan_out)
        ) % width
        return permutation[..., offsets]

    def _route(self, x_flat: Tensor) -> PairwiseRoute:
        anchor_a = self.anchors[:, :, 0].flatten()
        anchor_b = self.anchors[:, :, 1].flatten()
        margins = x_flat.index_select(-1, anchor_a).view(
            x_flat.shape[0], self.tables, self.comparisons
        )
        margins = margins - x_flat.index_select(-1, anchor_b).view(
            x_flat.shape[0], self.tables, self.comparisons
        )
        margins = margins - self.thresholds.to(
            device=x_flat.device,
            dtype=x_flat.dtype,
        ).view(1, self.tables, self.comparisons)
        indices = (
            (margins > 0).to(torch.long)
            * self.powers.to(device=x_flat.device).view(1, 1, -1)
        ).sum(dim=-1)
        return PairwiseRoute(indices, margins)

    def _flat_program_indices(self, codes: Tensor) -> Tensor:
        table_offsets = (
            torch.arange(self.tables, device=codes.device, dtype=torch.long)
            .view(1, self.tables)
            .mul(self.table_size)
        )
        return (codes + table_offsets).reshape(-1)

    def _select_program_tensor(self, tensor: Tensor, codes: Tensor) -> Tensor:
        selected = tensor.reshape(self.tables * self.table_size, *tensor.shape[2:])
        selected = selected.index_select(0, self._flat_program_indices(codes))
        return selected.view(codes.shape[0], self.tables, *tensor.shape[2:])

    def _program_payload(
        self,
        x_flat: Tensor,
        codes: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        read_indices = self._select_program_tensor(
            self.read_indices.to(device=x_flat.device),
            codes,
        )
        read_weight = self._select_program_tensor(self.read_weight, codes)
        margin_thresholds = self._select_program_tensor(
            self.margin_thresholds,
            codes,
        )
        gathered = x_flat.gather(1, read_indices.reshape(x_flat.shape[0], -1))
        gathered = gathered.view(
            x_flat.shape[0],
            self.tables,
            self.candidates,
            self.margin_fan_in,
        )
        learned_margins = (gathered * read_weight).sum(dim=-1) - margin_thresholds
        activation = F.relu(learned_margins)

        write_weight = self._select_program_tensor(self.write_weight, codes)
        write_indices = self._select_program_tensor(
            self.write_indices.to(device=x_flat.device),
            codes,
        )
        payload = activation.unsqueeze(-1) * write_weight
        return payload, write_indices, learned_margins

    def _payload_to_output(self, payload: Tensor, write_indices: Tensor) -> Tensor:
        output = torch.zeros(
            payload.shape[0],
            self.output_dim,
            device=payload.device,
            dtype=payload.dtype,
        )
        output.scatter_add_(
            1,
            write_indices.reshape(payload.shape[0], -1),
            payload.reshape(payload.shape[0], -1),
        )
        return output * self.output_scale

    def _route_ste_correction(
        self,
        x_flat: Tensor,
        route: PairwiseRoute,
        payload: Tensor,
        write_indices: Tensor,
    ) -> Tensor:
        bit = route.margins.abs().argmin(dim=-1)
        nearest_margin = route.margins.gather(
            dim=-1,
            index=bit.unsqueeze(-1),
        ).squeeze(-1)
        neighbor_codes = route.indices ^ (2 ** bit).long()
        neighbor_payload, neighbor_write_indices, _ = self._program_payload(
            x_flat,
            neighbor_codes,
        )
        hard_bit = (nearest_margin > 0).to(nearest_margin.dtype)
        ste_delta = ste_heaviside(nearest_margin) - hard_bit
        scale = ste_delta.unsqueeze(-1).unsqueeze(-1)
        return self._payload_to_output(
            neighbor_payload * scale,
            neighbor_write_indices,
        ) - self._payload_to_output(payload * scale, write_indices)

    def compute(self, x: Tensor) -> tuple[Tensor, PairwiseRoute]:
        if x.ndim < 1:
            raise ValueError("HashSelectedSparseHinge expects at least one dimension")
        if not x.is_floating_point():
            raise TypeError("HashSelectedSparseHinge expects floating-point input")
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"HashSelectedSparseHinge expected last dimension {self.input_dim}, "
                f"got shape {tuple(x.shape)}"
            )
        input_dtype = x.dtype
        prefix = x.shape[:-1]
        x_flat = x.reshape(-1, self.input_dim).float()
        route_flat = self._route(x_flat)
        payload, write_indices, _ = self._program_payload(
            x_flat,
            route_flat.indices,
        )
        output = self._payload_to_output(payload, write_indices)
        if self.training and (x.requires_grad or self.thresholds.requires_grad):
            output = output + self._route_ste_correction(
                x_flat,
                route_flat,
                payload,
                write_indices,
            )
        route = PairwiseRoute(
            route_flat.indices.view(*prefix, self.tables),
            route_flat.margins.view(*prefix, self.tables, self.comparisons),
        )
        return output.view(*prefix, self.output_dim).to(dtype=input_dtype), route

    def forward(self, x: Tensor) -> Tensor:
        output, _ = self.compute(x)
        return output
