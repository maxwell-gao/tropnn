from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

PackedLutDType = Literal["fp32", "bf16", "fp16", "int8", "fp8", "int4", "int2", "fp4", "nf4"]


@dataclass(frozen=True)
class _PackedPayload:
    mode: PackedLutDType
    data: Tensor
    scales: Tensor
    codebook: Tensor
    table_size: int
    out_features: int


def _empty_payload_tensor(device: torch.device) -> Tensor:
    return torch.empty(0, device=device, dtype=torch.float32)


def _pack_lut_int8(lut: Tensor, *, qmax: float = 127.0) -> tuple[Tensor, Tensor]:
    lut = lut.detach().float().contiguous()
    scales = lut.abs().amax(dim=-1).clamp_min(1e-8) / qmax
    codes = torch.round(lut / scales.unsqueeze(-1)).clamp(min=-qmax, max=qmax).to(torch.int8).contiguous()
    return codes, scales.contiguous()


def _pack_lut_4bit(lut: Tensor, codebook: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    lut = lut.detach().float().contiguous()
    codebook = codebook.to(device=lut.device, dtype=torch.float32).contiguous()
    scale = lut.abs().amax(dim=-1).clamp_min(1e-8) / codebook.abs().amax().clamp_min(1e-8)
    normalized = lut / scale.unsqueeze(-1)
    distances = (normalized.unsqueeze(-1) - codebook.view(1, 1, 1, 16)).abs()
    codes = distances.argmin(dim=-1).to(torch.uint8)
    if codes.shape[-1] % 2:
        codes = torch.cat((codes, torch.zeros(*codes.shape[:-1], 1, device=codes.device, dtype=torch.uint8)), dim=-1)
    low = codes[..., 0::2]
    high = codes[..., 1::2]
    packed = (low | (high << 4)).contiguous()
    return packed, scale.contiguous(), codebook


def _pack_lut_int4(lut: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    lut = lut.detach().float().contiguous()
    codebook = torch.arange(-8, 8, device=lut.device, dtype=torch.float32).contiguous()
    scale = lut.abs().amax(dim=-1).clamp_min(1e-8) / 7.0
    normalized = lut / scale.unsqueeze(-1)
    distances = (normalized.unsqueeze(-1) - codebook.view(1, 1, 1, 16)).abs()
    codes = distances.argmin(dim=-1).to(torch.uint8)
    if codes.shape[-1] % 2:
        codes = torch.cat((codes, torch.zeros(*codes.shape[:-1], 1, device=codes.device, dtype=torch.uint8)), dim=-1)
    low = codes[..., 0::2]
    high = codes[..., 1::2]
    packed = (low | (high << 4)).contiguous()
    return packed, scale.contiguous(), codebook


def _pack_lut_int2(lut: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    lut = lut.detach().float().contiguous()
    quant_codebook = torch.tensor([-2.0, -1.0, 0.0, 1.0], device=lut.device, dtype=torch.float32)
    codebook = torch.tensor([-2.0, -1.0, 0.0, 1.0] + [0.0] * 12, device=lut.device, dtype=torch.float32)
    scale = lut.abs().amax(dim=-1).clamp_min(1e-8)
    normalized = lut / scale.unsqueeze(-1)
    distances = (normalized.unsqueeze(-1) - quant_codebook.view(1, 1, 1, 4)).abs()
    codes = distances.argmin(dim=-1).to(torch.uint8)
    pad = (-codes.shape[-1]) % 4
    if pad:
        codes = torch.cat((codes, torch.zeros(*codes.shape[:-1], pad, device=codes.device, dtype=torch.uint8)), dim=-1)
    packed = (codes[..., 0::4] | (codes[..., 1::4] << 2) | (codes[..., 2::4] << 4) | (codes[..., 3::4] << 6)).contiguous()
    return packed, scale.contiguous(), codebook


def _fp4_codebook(device: torch.device) -> Tensor:
    return torch.tensor(
        [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0],
        device=device,
        dtype=torch.float32,
    )


def _nf4_codebook(device: torch.device) -> Tensor:
    return torch.tensor(
        [
            -1.0,
            -0.6961928,
            -0.5250731,
            -0.3949175,
            -0.2844414,
            -0.1847734,
            -0.0910500,
            0.0,
            0.0795803,
            0.1609302,
            0.2461123,
            0.3379152,
            0.4407098,
            0.5626170,
            0.7229568,
            1.0,
        ],
        device=device,
        dtype=torch.float32,
    )


def _pack_lut_payload(lut: Tensor, mode: PackedLutDType) -> _PackedPayload:
    if lut.ndim != 3:
        raise ValueError(f"lut must have shape [tables, table_size, out_features], got {tuple(lut.shape)}")
    table_size = int(lut.shape[1])
    out_features = int(lut.shape[2])
    empty = _empty_payload_tensor(lut.device)
    if mode == "fp32":
        return _PackedPayload(mode, lut.detach().to(torch.float32).contiguous(), empty, empty, table_size, out_features)
    if mode == "bf16":
        return _PackedPayload(mode, lut.detach().to(torch.bfloat16).contiguous(), empty, empty, table_size, out_features)
    if mode == "fp16":
        return _PackedPayload(mode, lut.detach().to(torch.float16).contiguous(), empty, empty, table_size, out_features)
    if mode in {"int8", "fp8"}:
        codes, scales = _pack_lut_int8(lut, qmax=127.0)
        return _PackedPayload(mode, codes, scales, empty, table_size, out_features)
    if mode == "int4":
        packed, scales, codebook = _pack_lut_int4(lut)
        return _PackedPayload(mode, packed, scales, codebook, table_size, out_features)
    if mode == "int2":
        packed, scales, codebook = _pack_lut_int2(lut)
        return _PackedPayload(mode, packed, scales, codebook, table_size, out_features)
    codebook = _fp4_codebook(lut.device) if mode == "fp4" else _nf4_codebook(lut.device)
    packed, scales, codebook = _pack_lut_4bit(lut, codebook)
    return _PackedPayload(mode, packed, scales, codebook, table_size, out_features)
