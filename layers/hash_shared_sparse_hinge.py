from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .pairwise import (
    PAIRWISE_ANCHOR_POLICIES,
    PairwiseRoute,
    _make_pairwise_anchors,
    ste_heaviside,
)

HashSharedSelectionMode = Literal["hash", "fixed", "all"]


class HashSharedSparseHinge(nn.Module):
    """Hash-selected sparse hinge actions drawn from a cross-code shared pool.

    Each table owns ``pool_size`` sparse affine hinge programs.  The program
    parameters are shared across all ``2**comparisons`` hash codes.  A fixed,
    balanced code-to-candidate map selects ``candidates_per_code`` programs for
    ``selection_mode="hash"``.  ``fixed`` evaluates the same candidate subset
    for every code at identical hard-forward active work, while ``all``
    evaluates the whole shared pool as a compute-heavier control.

    The hard hash only selects candidate identities.  Once selected, each
    program reads the current live input through a learned sparse affine margin
    and writes ``ReLU(margin)`` through learned sparse output coefficients.
    The action is affine-ReLU inside a fixed hash chamber, but switching the
    candidate set at a coarse hash wall is not generally continuous.
    """

    is_hash_shared_sparse_hinge = True

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        tables: int = 64,
        comparisons: int = 6,
        pool_size: int = 16,
        candidates_per_code: int = 4,
        margin_fan_in: int = 8,
        write_fan_out: int = 32,
        selection_mode: HashSharedSelectionMode = "hash",
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
        self.pool_size = int(pool_size)
        self.candidates_per_code = int(candidates_per_code)
        self.margin_fan_in = int(margin_fan_in)
        self.write_fan_out = int(write_fan_out)
        self.selection_mode: HashSharedSelectionMode = selection_mode
        self.anchor_policy = str(anchor_policy)
        self.use_min_margin_ste = bool(use_min_margin_ste)
        self.table_size = 1 << self.comparisons
        self.active_candidates = (
            self.pool_size if self.selection_mode == "all" else self.candidates_per_code
        )
        self.payload_width = self.write_fan_out
        self.write_degree = self.active_candidates * self.write_fan_out
        self.output_scale = (
            1.0 / math.sqrt(float(self.tables * self.active_candidates))
            if use_output_scaling
            else 1.0
        )

        if self.input_dim < 1 or self.output_dim < 1:
            raise ValueError("input_dim and output_dim must be positive")
        if self.tables < 1 or self.comparisons < 1 or self.pool_size < 1:
            raise ValueError("tables, comparisons, and pool_size must be positive")
        if not 1 <= self.candidates_per_code <= self.pool_size:
            raise ValueError("candidates_per_code must be in [1, pool_size]")
        if self.pool_size > self.table_size * self.candidates_per_code:
            raise ValueError(
                "pool_size must not exceed table_size * candidates_per_code"
            )
        if not 1 <= self.margin_fan_in <= self.input_dim:
            raise ValueError("margin_fan_in must be in [1, input_dim]")
        if not 1 <= self.write_fan_out <= self.output_dim:
            raise ValueError("write_fan_out must be in [1, output_dim]")
        if self.selection_mode not in {"hash", "fixed", "all"}:
            raise ValueError(
                "selection_mode must be one of 'hash', 'fixed', or 'all'"
            )
        if self.anchor_policy not in PAIRWISE_ANCHOR_POLICIES:
            raise ValueError(
                f"unsupported anchor_policy {self.anchor_policy!r}; "
                f"choices={PAIRWISE_ANCHOR_POLICIES}"
            )
        if not self.use_min_margin_ste:
            raise ValueError(
                "HashSharedSparseHinge currently supports min-margin route STE only"
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

        program_generator = torch.Generator(device="cpu").manual_seed(
            int(seed) + 104729
        )
        map_generator = torch.Generator(device="cpu").manual_seed(
            int(seed) + 130363
        )
        self.register_buffer(
            "read_indices",
            self._make_sparse_supports(
                width=self.input_dim,
                fan_out=self.margin_fan_in,
                generator=program_generator,
            ),
        )
        self.register_buffer(
            "write_indices",
            self._make_sparse_supports(
                width=self.output_dim,
                fan_out=self.write_fan_out,
                generator=program_generator,
            ),
        )
        self.register_buffer(
            "candidate_ranking",
            self._make_balanced_candidate_ranking(map_generator),
        )

        program_shape = (self.tables, self.pool_size)
        read_signs = torch.randint(
            0,
            2,
            (*program_shape, self.margin_fan_in),
            generator=program_generator,
            dtype=torch.int8,
        ).to(torch.float32).mul_(2.0).sub_(1.0)
        write_signs = torch.randint(
            0,
            2,
            (*program_shape, self.write_fan_out),
            generator=program_generator,
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

        self.last_candidate_ids: Tensor | None = None
        self.last_learned_margins: Tensor | None = None

    @property
    def candidates(self) -> int:
        """Compatibility alias for the number of active candidates."""

        return self.active_candidates

    @property
    def active_margin_count(self) -> int:
        return self.tables * self.active_candidates

    @property
    def candidate_bank_size(self) -> int:
        return self.tables * self.pool_size

    @property
    def semantic_route_terms(self) -> int:
        return self.tables * self.comparisons

    @property
    def semantic_action_terms(self) -> int:
        return self.tables * self.active_candidates * (
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

    @property
    def candidate_code_degrees(self) -> Tensor:
        if self.selection_mode == "all":
            return torch.full(
                (self.tables, self.pool_size),
                self.table_size,
                dtype=torch.long,
                device=self.candidate_ranking.device,
            )
        ids = self.candidate_ranking[..., : self.candidates_per_code]
        if self.selection_mode == "fixed":
            ids = ids[:, :1].expand(-1, self.table_size, -1)
        offsets = (
            torch.arange(self.tables, device=ids.device, dtype=torch.long)
            .view(self.tables, 1, 1)
            .mul(self.pool_size)
        )
        counts = torch.bincount(
            (ids + offsets).reshape(-1),
            minlength=self.tables * self.pool_size,
        )
        return counts.view(self.tables, self.pool_size)

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
            f"pool_size={self.pool_size}, candidates_per_code={self.candidates_per_code}, "
            f"margin_fan_in={self.margin_fan_in}, "
            f"write_fan_out={self.write_fan_out}, "
            f"selection_mode={self.selection_mode!r}, "
            f"anchor_policy={self.anchor_policy!r}"
        )

    def _make_sparse_supports(
        self,
        *,
        width: int,
        fan_out: int,
        generator: torch.Generator,
    ) -> Tensor:
        scores = torch.rand(self.tables, width, generator=generator)
        permutation = scores.argsort(dim=-1)
        offsets = (
            torch.arange(self.pool_size).view(self.pool_size, 1) * fan_out
            + torch.arange(fan_out).view(1, fan_out)
        ) % width
        return permutation[:, offsets]

    def _make_balanced_candidate_ranking(self, generator: torch.Generator) -> Tensor:
        result = torch.empty(
            self.tables,
            self.table_size,
            self.pool_size,
            dtype=torch.long,
        )
        for table in range(self.tables):
            code_rank = torch.randperm(self.table_size, generator=generator)
            pool_permutation = torch.randperm(self.pool_size, generator=generator)
            positions = (
                code_rank.view(-1, 1)
                + torch.arange(self.pool_size).view(1, -1)
            ) % self.pool_size
            result[table] = pool_permutation[positions]
        return result

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

    def _candidate_ids_for_codes(self, codes: Tensor) -> Tensor:
        batch = codes.shape[0]
        if self.selection_mode == "all":
            return (
                torch.arange(self.pool_size, device=codes.device, dtype=torch.long)
                .view(1, 1, self.pool_size)
                .expand(batch, self.tables, -1)
            )
        if self.selection_mode == "fixed":
            return self.candidate_ranking[:, 0, : self.candidates_per_code].to(
                device=codes.device
            ).view(
                1, self.tables, self.candidates_per_code
            ).expand(batch, -1, -1)
        table_offsets = (
            torch.arange(self.tables, device=codes.device, dtype=torch.long)
            .view(1, self.tables)
            .mul(self.table_size)
        )
        mapping = self.candidate_ranking[
            ..., : self.candidates_per_code
        ].to(device=codes.device).reshape(
            self.tables * self.table_size,
            self.candidates_per_code,
        )
        selected = mapping.index_select(0, (codes + table_offsets).reshape(-1))
        return selected.view(batch, self.tables, self.candidates_per_code)

    def _select_pool_tensor(self, tensor: Tensor, candidate_ids: Tensor) -> Tensor:
        table_offsets = (
            torch.arange(self.tables, device=candidate_ids.device, dtype=torch.long)
            .view(1, self.tables, 1)
            .mul(self.pool_size)
        )
        flat_indices = (candidate_ids + table_offsets).reshape(-1)
        pool = tensor.reshape(self.tables * self.pool_size, *tensor.shape[2:])
        selected = pool.index_select(0, flat_indices)
        return selected.view(
            candidate_ids.shape[0],
            self.tables,
            candidate_ids.shape[-1],
            *tensor.shape[2:],
        )

    def _program_payload(
        self,
        x_flat: Tensor,
        codes: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        candidate_ids = self._candidate_ids_for_codes(codes)
        read_indices = self._select_pool_tensor(
            self.read_indices.to(device=x_flat.device),
            candidate_ids,
        )
        read_weight = self._select_pool_tensor(self.read_weight, candidate_ids)
        margin_thresholds = self._select_pool_tensor(
            self.margin_thresholds,
            candidate_ids,
        )
        gathered = x_flat.gather(1, read_indices.reshape(x_flat.shape[0], -1))
        gathered = gathered.view(
            x_flat.shape[0],
            self.tables,
            candidate_ids.shape[-1],
            self.margin_fan_in,
        )
        learned_margins = (gathered * read_weight).sum(dim=-1) - margin_thresholds
        activation = F.relu(learned_margins)

        write_weight = self._select_pool_tensor(self.write_weight, candidate_ids)
        write_indices = self._select_pool_tensor(
            self.write_indices.to(device=x_flat.device),
            candidate_ids,
        )
        payload = activation.unsqueeze(-1) * write_weight
        return payload, write_indices, learned_margins, candidate_ids

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
        neighbor_payload, neighbor_write_indices, _, _ = self._program_payload(
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
            raise ValueError("HashSharedSparseHinge expects at least one dimension")
        if not x.is_floating_point():
            raise TypeError("HashSharedSparseHinge expects floating-point input")
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"HashSharedSparseHinge expected last dimension {self.input_dim}, "
                f"got shape {tuple(x.shape)}"
            )
        input_dtype = x.dtype
        prefix = x.shape[:-1]
        x_flat = x.reshape(-1, self.input_dim).float()
        route_flat = self._route(x_flat)
        payload, write_indices, learned_margins, candidate_ids = self._program_payload(
            x_flat,
            route_flat.indices,
        )
        self.last_candidate_ids = candidate_ids.detach()
        self.last_learned_margins = learned_margins.detach()
        output = self._payload_to_output(payload, write_indices)
        if self.selection_mode == "hash" and self.training and (
            x.requires_grad or self.thresholds.requires_grad
        ):
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
