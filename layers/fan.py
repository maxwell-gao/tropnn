from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..backend import Backend, trop_scores
from .base import RoutedLinearBase
from .tropical import _minface_mix, _top2_indices

FanValueMode = Literal["site", "basis"]
FanRecoveryMode = Literal["untied", "tied"]
FAN_VALUE_MODES = ("site", "basis")
FAN_RECOVERY_MODES = ("untied", "tied")


class TropFanLinear(RoutedLinearBase):
    """Tropical normal-fan layer with geometry-generated values.

    The routing sites define the tropical polynomial terms. Values are either
    generated directly from the winning site directions or from a shared value
    basis, binding the cell value field to compact geometric parameters rather
    than a free vector codebook.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        heads: int = 32,
        cells: int = 4,
        code_dim: int = 32,
        backend: Backend = "torch",
        seed: int = 0,
        value_init_std: float = 0.02,
        fan_value_mode: FanValueMode = "site",
        fan_basis_rank: int = 16,
        fan_recovery_mode: FanRecoveryMode = "untied",
        use_output_scaling: bool = True,
    ) -> None:
        if heads < 1:
            raise ValueError(f"heads must be >= 1, got {heads}")
        if cells < 2:
            raise ValueError(f"cells must be >= 2, got {cells}")
        if code_dim < 1:
            raise ValueError(f"code_dim must be >= 1, got {code_dim}")
        if fan_value_mode not in FAN_VALUE_MODES:
            raise ValueError(f"fan_value_mode must be one of {FAN_VALUE_MODES}, got {fan_value_mode!r}")
        if fan_recovery_mode not in FAN_RECOVERY_MODES:
            raise ValueError(f"fan_recovery_mode must be one of {FAN_RECOVERY_MODES}, got {fan_recovery_mode!r}")
        if fan_basis_rank < 1:
            raise ValueError(f"fan_basis_rank must be >= 1, got {fan_basis_rank}")

        super().__init__(in_features, out_features, backend=backend, output_scale=1.0)
        self.heads = heads
        self.cells = cells
        self.code_dim = code_dim
        self.code_scale = 1.0 / math.sqrt(heads) if use_output_scaling else 1.0
        self.fan_value_mode = fan_value_mode
        self.fan_basis_rank = fan_basis_rank
        self.fan_recovery_mode = fan_recovery_mode

        torch.manual_seed(seed)
        self.proj = nn.Linear(in_features, code_dim, bias=False)
        nn.init.normal_(self.proj.weight, mean=0.0, std=1.0 / math.sqrt(max(1, in_features)))

        site_std = 1.0 / math.sqrt(max(1, code_dim))
        self.sites = nn.Parameter(torch.randn(heads, cells, code_dim) * site_std)
        self.lifting = nn.Parameter(torch.zeros(heads, cells))
        if fan_value_mode == "site":
            self.value_scale = nn.Parameter(torch.randn(heads, cells) * value_init_std)
            self.value_coeff = None
            self.value_basis = None
        else:
            self.value_scale = None
            self.value_coeff = nn.Parameter(torch.randn(heads, cells, fan_basis_rank) * value_init_std / math.sqrt(fan_basis_rank))
            self.value_basis = nn.Parameter(torch.randn(fan_basis_rank, code_dim) / math.sqrt(max(1, code_dim)))
        self.output_proj = nn.Linear(code_dim, out_features)
        nn.init.kaiming_uniform_(self.output_proj.weight, a=math.sqrt(5))
        if self.output_proj.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.output_proj.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.output_proj.bias, -bound, bound)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, heads={self.heads}, "
            f"cells={self.cells}, code_dim={self.code_dim}, fan_value_mode={self.fan_value_mode!r}, "
            f"fan_basis_rank={self.fan_basis_rank}, fan_recovery_mode={self.fan_recovery_mode!r}, backend={self.backend!r}"
        )

    def _project_input(self, x: Tensor, compute_dtype: torch.dtype) -> Tensor:
        return self.proj(x.to(dtype=self.proj.weight.dtype)).to(compute_dtype)

    def generated_values(self, *, input_device: torch.device | None = None, compute_dtype: torch.dtype | None = None) -> Tensor:
        device = self.sites.device if input_device is None else input_device
        dtype = self.sites.dtype if compute_dtype is None else compute_dtype
        if self.fan_value_mode == "site":
            sites = self.sites.to(dtype=dtype, device=device)
            unit_sites = F.normalize(sites, dim=-1, eps=1e-12)
            assert self.value_scale is not None
            return unit_sites * self.value_scale.to(dtype=dtype, device=device).unsqueeze(-1)
        assert self.value_coeff is not None and self.value_basis is not None
        coeff = self.value_coeff.to(dtype=dtype, device=device)
        basis = self.value_basis.to(dtype=dtype, device=device)
        return torch.einsum("hkr,rd->hkd", coeff, basis)

    def _score_backend(self, *, training: bool) -> Backend:
        if self.backend in {"tilelang", "zig"}:
            return "torch"
        if training and self.backend == "triton":
            return "torch"
        return self.backend

    def _scores(
        self,
        latent: Tensor,
        *,
        input_device: torch.device,
        compute_dtype: torch.dtype,
        training: bool,
    ) -> Tensor:
        return trop_scores(
            latent,
            self.sites.to(dtype=compute_dtype, device=input_device),
            self.lifting.to(dtype=compute_dtype, device=input_device),
            backend=self._score_backend(training=training),
        )

    def _selected_values(
        self,
        winner_idx: Tensor,
        *,
        input_device: torch.device,
        compute_dtype: torch.dtype,
    ) -> Tensor:
        values = self.generated_values(input_device=input_device, compute_dtype=compute_dtype).unsqueeze(0).unsqueeze(0)
        gather_idx = winner_idx.unsqueeze(-1).unsqueeze(-1).expand(*winner_idx.shape, 1, self.code_dim)
        return values.expand(*winner_idx.shape[:2], -1, -1, -1).gather(-2, gather_idx).squeeze(-2)

    def _output_from_hidden(self, hidden: Tensor, *, input_device: torch.device, compute_dtype: torch.dtype) -> Tensor:
        if self.fan_recovery_mode != "untied":
            raise RuntimeError("TropFanLinear fan_recovery_mode='tied' is only supported by the scaling benchmark tied recovery wrapper")
        weight = self.output_proj.weight.to(dtype=compute_dtype, device=input_device)
        bias = self.output_proj.bias
        output = torch.matmul(hidden, weight.t())
        if bias is not None:
            output = output + bias.to(dtype=compute_dtype, device=input_device)
        return output

    def _hidden_from_latent(
        self,
        latent: Tensor,
        *,
        input_device: torch.device,
        compute_dtype: torch.dtype,
        training: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        scores = self._scores(latent, input_device=input_device, compute_dtype=compute_dtype, training=training)
        winner_idx, runner_idx, margins = _top2_indices(scores)
        winner_values = self._selected_values(winner_idx, input_device=input_device, compute_dtype=compute_dtype)
        if training:
            runner_values = self._selected_values(runner_idx, input_device=input_device, compute_dtype=compute_dtype)
            values = _minface_mix(winner_values, runner_values, margins)
        else:
            values = winner_values
        hidden = latent + values.sum(dim=2) * self.code_scale
        return hidden, values, scores

    def _route_output(
        self,
        latent: Tensor,
        *,
        input_device: torch.device,
        compute_dtype: torch.dtype,
        training: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if self.backend == "tilelang" and self.fan_value_mode == "basis" and latent.is_cuda and compute_dtype == torch.float32:
            assert self.value_coeff is not None and self.value_basis is not None
            sites = self.sites.to(dtype=compute_dtype, device=input_device)
            lifting = self.lifting.to(dtype=compute_dtype, device=input_device)
            value_coeff = self.value_coeff.to(dtype=compute_dtype, device=input_device)
            value_basis = self.value_basis.to(dtype=compute_dtype, device=input_device)
            score_bytes = latent.shape[0] * latent.shape[1] * self.heads * self.cells * 4
            if not training and not torch.is_grad_enabled() and self.code_dim >= 128 and score_bytes <= 128 * 1024 * 1024:
                from ..backends import has_triton, trop_fan_basis_hidden_triton_eval

                if has_triton():
                    hidden, winner_idx, margins = trop_fan_basis_hidden_triton_eval(
                        latent,
                        sites,
                        lifting,
                        value_coeff,
                        value_basis,
                        code_scale=self.code_scale,
                    )
                    return self._output_from_hidden(hidden, input_device=input_device, compute_dtype=compute_dtype), winner_idx, margins

            from ..backends import trop_fan_basis_route_hidden_tilelang

            hidden, winner_idx, margins = trop_fan_basis_route_hidden_tilelang(
                latent,
                sites,
                lifting,
                value_coeff,
                value_basis,
                code_scale=self.code_scale,
                training=training,
            )
            return self._output_from_hidden(hidden, input_device=input_device, compute_dtype=compute_dtype), winner_idx, margins

        hidden, _, scores = self._hidden_from_latent(
            latent,
            input_device=input_device,
            compute_dtype=compute_dtype,
            training=training,
        )
        winner_idx, _, margins = _top2_indices(scores)
        return self._output_from_hidden(hidden, input_device=input_device, compute_dtype=compute_dtype), winner_idx, margins

    def effective_representations(
        self,
        n_features: int,
        *,
        device: torch.device,
        training: bool | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        use_training_routes = self.training if training is None else training
        x = torch.eye(n_features, device=device)
        latent = self._project_input(x.unsqueeze(1), torch.float32)
        hidden, values, scores = self._hidden_from_latent(
            latent,
            input_device=device,
            compute_dtype=torch.float32,
            training=use_training_routes,
        )
        self._last_indices = scores.argmax(dim=-1).detach()
        if scores.shape[-1] >= 2:
            top2_vals = scores.topk(k=2, dim=-1).values
            self._last_margins = (top2_vals[..., 0] - top2_vals[..., 1]).detach()
        else:
            self._last_margins = torch.empty((*scores.shape[:3], 0), device=device, dtype=scores.dtype)
        return hidden.squeeze(1), values.squeeze(1), scores.squeeze(1)
