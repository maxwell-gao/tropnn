"""Train a minimal tropnn classifier on a local EMNIST split."""

from __future__ import annotations

import argparse
import csv
import gzip
import math
import struct
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from ..layers import PairwiseLinear, PairwiseWalshLinear, TropFanZeroDenseLinear, TropLinear, TropZeroDenseLinear

IDX_DTYPES = {
    0x08: np.uint8,
    0x09: np.int8,
    0x0B: np.dtype(">i2"),
    0x0C: np.dtype(">i4"),
    0x0D: np.dtype(">f4"),
    0x0E: np.dtype(">f8"),
}
EMNIST_SPLITS = ("byclass", "bymerge", "balanced", "letters", "digits", "mnist")
ROUTED_FAMILIES = ("tropical", "tropical_lowrank", "tropical_zero_dense", "tropfan_zero_dense", "pairwise", "pairwise_walsh")
TROPICAL_FAMILIES = ("tropical", "tropical_lowrank")
HEAD_ROUTED_FAMILIES = ("tropical", "tropical_lowrank", "tropical_zero_dense", "tropfan_zero_dense")
PAIRWISE_ROUTE_PREMIXES = (
    "none",
    "block_hadamard",
    "cyclic_expander",
    "learned_butterfly",
    "multi_hash_structured",
    "learnable_cyclic_expander",
    "hadamard_diag_sandwich",
    "givens_butterfly",
    "sparse_product",
    "lowrank",
)


def _read_idx(path: Path) -> np.ndarray:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as handle:
        header = handle.read(4)
        zero_a, zero_b, dtype_code, ndim = struct.unpack(">BBBB", header)
        if zero_a != 0 or zero_b != 0:
            raise ValueError(f"IDX file {path} has invalid magic prefix")
        shape = struct.unpack(f">{ndim}I", handle.read(4 * ndim))
        data = np.frombuffer(handle.read(), dtype=IDX_DTYPES[dtype_code])
    return data.reshape(shape)


def _find_emnist_file(root: Path, split: str, train: bool, kind: str) -> Path:
    stem = f"emnist-{split}-{'train' if train else 'test'}-{kind}-idx{'3' if kind == 'images' else '1'}-ubyte"
    candidates = sorted(root.rglob(stem)) + sorted(root.rglob(stem + ".gz"))
    if not candidates:
        raise FileNotFoundError(f"Could not find {stem}[.gz] under {root}")
    return candidates[0]


def load_emnist_split(
    root: Path,
    split: str,
    *,
    train: bool,
    limit: Optional[int],
    fix_orientation: bool,
    permute: bool,
    permute_seed: int,
) -> tuple[Tensor, Tensor]:
    image_path = _find_emnist_file(root, split, train=train, kind="images")
    label_path = _find_emnist_file(root, split, train=train, kind="labels")
    images = _read_idx(image_path).astype(np.float32)
    labels = _read_idx(label_path).astype(np.int64)
    if fix_orientation:
        images = np.transpose(images, (0, 2, 1))[:, :, ::-1].copy()
    if split == "letters" and labels.min() == 1:
        labels = labels - 1
    x = torch.from_numpy(images).reshape(images.shape[0], -1).float() / 255.0
    x = x * 2.0 - 1.0
    y = torch.from_numpy(labels.astype(np.int64))
    if permute:
        gen = torch.Generator(device="cpu").manual_seed(permute_seed)
        order = torch.randperm(x.shape[1], generator=gen)
        x = x[:, order]
    if limit is not None:
        x = x[:limit]
        y = y[:limit]
    return x, y


def _make_layer(
    family: str,
    d_in: int,
    d_out: int,
    *,
    heads: int,
    cells: int,
    code_dim: int,
    route_terms: int,
    fan_value_mode: str,
    fan_basis_rank: int,
    comparisons: int,
    pairwise_tables: int,
    pairwise_lut_init_std: float,
    pairwise_lut_accumulation: str,
    pairwise_max_group_size: int,
    fixed_zero_threshold: bool,
    pairwise_route_premix: str,
    route_premix_block_size: int,
    route_premix_expander_fanout: int,
    route_premix_sparse_stages: int,
    route_premix_lowrank_rank: int,
    pairwise_hashes: int,
    walsh_order: int,
    backend: str,
    seed: int,
) -> nn.Module:
    if family in TROPICAL_FAMILIES:
        return TropLinear(d_in, d_out, heads=heads, cells=cells, code_dim=code_dim, backend=backend, seed=seed)
    if family == "tropical_zero_dense":
        return TropZeroDenseLinear(d_in, d_out, heads=heads, cells=cells, route_terms=route_terms, seed=seed)
    if family == "tropfan_zero_dense":
        return TropFanZeroDenseLinear(
            d_in,
            d_out,
            heads=heads,
            cells=cells,
            code_dim=code_dim,
            fan_value_mode=fan_value_mode,  # type: ignore[arg-type]
            fan_basis_rank=fan_basis_rank,
            seed=seed,
        )
    if family == "pairwise_walsh":
        return PairwiseWalshLinear(
            d_in,
            d_out,
            tables=pairwise_tables,
            comparisons=comparisons,
            walsh_order=walsh_order,  # type: ignore[arg-type]
            backend=backend,
            seed=seed,
        )
    return _make_pairwise_route_layer(
        d_in,
        d_out,
        tables=pairwise_tables,
        comparisons=comparisons,
        backend=backend,
        seed=seed,
        lut_init_std=pairwise_lut_init_std,
        accumulation=pairwise_lut_accumulation,  # type: ignore[arg-type]
        max_group_size=pairwise_max_group_size,
        fixed_zero_threshold=fixed_zero_threshold,
        route_premix=pairwise_route_premix,
        block_size=route_premix_block_size,
        expander_fanout=route_premix_expander_fanout,
        sparse_stages=route_premix_sparse_stages,
        lowrank_rank=route_premix_lowrank_rank,
        hashes=pairwise_hashes,
    )


def _next_power_of_two(n: int) -> int:
    return 1 << (max(1, n) - 1).bit_length()


def _fwht_last_dim(x: Tensor) -> Tensor:
    original_shape = x.shape
    batch_shape = x.shape[:-1]
    width = x.shape[-1]
    h = 1
    y = x
    while h < width:
        y = y.reshape(*batch_shape, -1, h * 2)
        a = y[..., :h]
        b = y[..., h : h * 2]
        y = torch.cat((a + b, a - b), dim=-1)
        h *= 2
    return y.reshape(*original_shape)


class BlockHadamardRouteMix(nn.Module):
    """Fixed sign/permutation + block FWHT route pre-mixer."""

    def __init__(self, features: int, *, block_size: int, seed: int) -> None:
        super().__init__()
        if block_size < 2 or block_size & (block_size - 1):
            raise ValueError(f"block_size must be a power of two >= 2, got {block_size}")
        self.features = int(features)
        self.block_size = int(block_size)
        self.padded_features = ((features + block_size - 1) // block_size) * block_size
        gen = torch.Generator(device="cpu").manual_seed(seed)
        perm = torch.randperm(self.padded_features, generator=gen)
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(self.padded_features)
        sign = torch.randint(0, 2, (self.padded_features,), generator=gen, dtype=torch.float32) * 2.0 - 1.0
        self.register_buffer("perm", perm)
        self.register_buffer("inv_perm", inv_perm)
        self.register_buffer("sign", sign)
        self.scale = 1.0 / math.sqrt(block_size)

    def forward(self, x: Tensor) -> Tensor:
        pad = self.padded_features - x.shape[-1]
        y = F.pad(x, (0, pad)) if pad > 0 else x
        y = y.index_select(-1, self.perm.to(device=x.device))
        y = y * self.sign.to(device=x.device, dtype=y.dtype)
        y = y.reshape(*y.shape[:-1], -1, self.block_size)
        y = _fwht_last_dim(y) * self.scale
        y = y.reshape(*x.shape[:-1], self.padded_features)
        y = y.index_select(-1, self.inv_perm.to(device=x.device))
        return y[..., : self.features]


class CyclicExpanderRouteMix(nn.Module):
    """Fixed cyclic k-sparse route pre-mixer."""

    def __init__(self, features: int, *, fanout: int, seed: int) -> None:
        super().__init__()
        if fanout < 1:
            raise ValueError(f"fanout must be >= 1, got {fanout}")
        self.features = int(features)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        candidates = torch.arange(1, max(2, features), dtype=torch.long)
        if candidates.numel() >= fanout:
            offsets = candidates[torch.randperm(candidates.numel(), generator=gen)[:fanout]]
        else:
            offsets = torch.arange(1, fanout + 1, dtype=torch.long).remainder(max(1, features))
            offsets[offsets == 0] = 1
        signs = torch.randint(0, 2, (fanout,), generator=gen, dtype=torch.float32) * 2.0 - 1.0
        self.register_buffer("offsets", offsets)
        self.register_buffer("signs", signs)
        self.scale = 1.0 / math.sqrt(fanout + 1)

    def forward(self, x: Tensor) -> Tensor:
        y = x
        signs = self.signs.to(device=x.device, dtype=x.dtype)
        for idx, offset in enumerate(self.offsets.tolist()):
            y = y + signs[idx] * torch.roll(x, shifts=int(offset), dims=-1)
        return y * self.scale


class LearnableCyclicExpanderRouteMix(nn.Module):
    """Learnable fixed-offset cyclic sparse route pre-mixer.

    Initialized near identity: the direct path starts at one, while shifted
    sparse edges start near zero.
    """

    def __init__(self, features: int, *, fanout: int, seed: int, init_std: float = 0.02) -> None:
        super().__init__()
        if fanout < 1:
            raise ValueError(f"fanout must be >= 1, got {fanout}")
        self.features = int(features)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        candidates = torch.arange(1, max(2, features), dtype=torch.long)
        if candidates.numel() >= fanout:
            offsets = candidates[torch.randperm(candidates.numel(), generator=gen)[:fanout]]
        else:
            offsets = torch.arange(1, fanout + 1, dtype=torch.long).remainder(max(1, features))
            offsets[offsets == 0] = 1
        self.register_buffer("offsets", offsets)
        self.direct = nn.Parameter(torch.ones(features))
        self.edge = nn.Parameter(torch.randn(fanout, features, generator=gen) * init_std)

    def forward(self, x: Tensor) -> Tensor:
        y = x * self.direct.to(device=x.device, dtype=x.dtype)
        edge = self.edge.to(device=x.device, dtype=x.dtype)
        for idx, offset in enumerate(self.offsets.tolist()):
            y = y + edge[idx] * torch.roll(x, shifts=int(offset), dims=-1)
        return y


class HadamardDiagonalSandwichRouteMix(nn.Module):
    """Residual fixed block-Hadamard sandwich with learnable diagonals."""

    def __init__(self, features: int, *, block_size: int, seed: int) -> None:
        super().__init__()
        self.h1 = BlockHadamardRouteMix(features, block_size=block_size, seed=seed)
        self.h2 = BlockHadamardRouteMix(features, block_size=block_size, seed=seed + 1009)
        self.gamma1 = nn.Parameter(torch.ones(features))
        self.gamma2 = nn.Parameter(torch.ones(features))
        self.alpha = nn.Parameter(torch.zeros(features))

    def forward(self, x: Tensor) -> Tensor:
        y = x * self.gamma1.to(device=x.device, dtype=x.dtype)
        y = self.h1(y)
        y = y * self.gamma2.to(device=x.device, dtype=x.dtype)
        y = self.h2(y)
        return x + self.alpha.to(device=x.device, dtype=x.dtype) * y


class LearnedButterflyRouteMix(nn.Module):
    """Learned full-width butterfly route pre-mixer initialized near identity."""

    def __init__(self, features: int, *, seed: int, init_std: float = 0.02) -> None:
        super().__init__()
        self.features = int(features)
        self.padded_features = _next_power_of_two(features)
        self.stages = int(math.log2(self.padded_features))
        gen = torch.Generator(device="cpu").manual_seed(seed)
        delta = torch.randn(self.stages, self.padded_features // 2, 2, 2, generator=gen) * init_std
        self.delta = nn.Parameter(delta)
        eye = torch.eye(2).view(1, 1, 2, 2).expand(self.stages, self.padded_features // 2, 2, 2).clone()
        self.register_buffer("eye", eye)

    def forward(self, x: Tensor) -> Tensor:
        pad = self.padded_features - x.shape[-1]
        y = F.pad(x, (0, pad)) if pad > 0 else x
        for stage in range(self.stages):
            stride = 1 << stage
            pairs = y.reshape(*y.shape[:-1], -1, 2, stride).transpose(-2, -1)
            flat_pairs = pairs.reshape(*y.shape[:-1], self.padded_features // 2, 2)
            matrix = (self.eye[stage] + self.delta[stage]).to(device=x.device, dtype=y.dtype)
            mixed = torch.einsum("...pi,pio->...po", flat_pairs, matrix)
            y = mixed.reshape(*y.shape[:-1], -1, stride, 2).transpose(-2, -1).reshape(*y.shape[:-1], self.padded_features)
        return y[..., : self.features]


class GivensButterflyRouteMix(nn.Module):
    """Orthogonal butterfly route pre-mixer with one learned angle per pair."""

    def __init__(self, features: int, *, seed: int, init_std: float = 0.02) -> None:
        super().__init__()
        self.features = int(features)
        self.padded_features = _next_power_of_two(features)
        self.stages = int(math.log2(self.padded_features))
        gen = torch.Generator(device="cpu").manual_seed(seed)
        theta = torch.randn(self.stages, self.padded_features // 2, generator=gen) * init_std
        self.theta = nn.Parameter(theta)

    def forward(self, x: Tensor) -> Tensor:
        pad = self.padded_features - x.shape[-1]
        y = F.pad(x, (0, pad)) if pad > 0 else x
        for stage in range(self.stages):
            stride = 1 << stage
            pairs = y.reshape(*y.shape[:-1], -1, 2, stride).transpose(-2, -1)
            flat_pairs = pairs.reshape(*y.shape[:-1], self.padded_features // 2, 2)
            a = flat_pairs[..., 0]
            b = flat_pairs[..., 1]
            theta = self.theta[stage].to(device=x.device, dtype=y.dtype)
            c = torch.cos(theta)
            s = torch.sin(theta)
            mixed = torch.stack((c * a - s * b, s * a + c * b), dim=-1)
            y = mixed.reshape(*y.shape[:-1], -1, stride, 2).transpose(-2, -1).reshape(*y.shape[:-1], self.padded_features)
        return y[..., : self.features]


class SparseProductRouteMix(nn.Module):
    """Product of fixed-pattern sparse learned cyclic mixers."""

    def __init__(self, features: int, *, stages: int, fanout: int, seed: int, init_std: float = 0.02) -> None:
        super().__init__()
        if stages < 1:
            raise ValueError(f"stages must be >= 1, got {stages}")
        if fanout < 1:
            raise ValueError(f"fanout must be >= 1, got {fanout}")
        self.features = int(features)
        self.stages = int(stages)
        self.fanout = int(fanout)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        offsets = []
        for _ in range(stages):
            candidates = torch.arange(1, max(2, features), dtype=torch.long)
            if candidates.numel() >= fanout:
                stage_offsets = candidates[torch.randperm(candidates.numel(), generator=gen)[:fanout]]
            else:
                stage_offsets = torch.arange(1, fanout + 1, dtype=torch.long).remainder(max(1, features))
                stage_offsets[stage_offsets == 0] = 1
            offsets.append(stage_offsets)
        self.register_buffer("offsets", torch.stack(offsets, dim=0))
        self.direct = nn.Parameter(torch.ones(stages, features))
        self.edge = nn.Parameter(torch.randn(stages, fanout, features, generator=gen) * init_std)

    def forward(self, x: Tensor) -> Tensor:
        y = x
        direct = self.direct.to(device=x.device, dtype=x.dtype)
        edge = self.edge.to(device=x.device, dtype=x.dtype)
        for stage in range(self.stages):
            z = y * direct[stage]
            for idx, offset in enumerate(self.offsets[stage].tolist()):
                z = z + edge[stage, idx] * torch.roll(y, shifts=int(offset), dims=-1)
            y = z
        return y


class LowRankRouteMix(nn.Module):
    """Tiny residual low-rank route pre-mixer, used as a diagnostic upper bound."""

    def __init__(self, features: int, *, rank: int, seed: int) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError(f"rank must be >= 1, got {rank}")
        gen = torch.Generator(device="cpu").manual_seed(seed)
        self.down = nn.Parameter(torch.randn(features, rank, generator=gen) / math.sqrt(max(1, features)))
        self.up = nn.Parameter(torch.zeros(rank, features))
        self.scale = 1.0 / math.sqrt(rank)

    def forward(self, x: Tensor) -> Tensor:
        return x + (x.matmul(self.down.to(device=x.device, dtype=x.dtype))).matmul(self.up.to(device=x.device, dtype=x.dtype)) * self.scale


class RoutePreMixPairwiseLayer(nn.Module):
    """Apply a cheap structured mixer before PairwiseLinear routing."""

    def __init__(self, mix: nn.Module, layer: PairwiseLinear) -> None:
        super().__init__()
        self.mix = mix
        self.layer = layer

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(self.mix(x))


class MultiHashStructuredPairwiseLayer(nn.Module):
    """Multiple independently mixed PairwiseLinear branches summed together."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        tables: int,
        comparisons: int,
        backend: str,
        seed: int,
        lut_init_std: float,
        accumulation: str,
        max_group_size: int,
        fixed_zero_threshold: bool,
        block_size: int,
        hashes: int,
    ) -> None:
        super().__init__()
        if hashes < 2:
            raise ValueError(f"multi_hash_structured requires hashes >= 2, got {hashes}")
        branches: list[nn.Module] = []
        for hash_idx in range(hashes):
            branch_seed = seed + hash_idx * 1009
            layer = PairwiseLinear(
                in_features,
                out_features,
                tables=tables,
                comparisons=comparisons,
                backend=backend,
                seed=branch_seed,
                lut_init_std=lut_init_std,
                accumulation=accumulation,  # type: ignore[arg-type]
                max_group_size=max_group_size,
                fixed_zero_threshold=fixed_zero_threshold,
            )
            mix = BlockHadamardRouteMix(in_features, block_size=block_size, seed=branch_seed + 17)
            branches.append(RoutePreMixPairwiseLayer(mix, layer))
        self.branches = nn.ModuleList(branches)
        self.scale = 1.0 / math.sqrt(hashes)

    def forward(self, x: Tensor) -> Tensor:
        y = self.branches[0](x)
        for branch in self.branches[1:]:
            y = y + branch(x)
        return y * self.scale


def _make_pairwise_route_layer(
    d_in: int,
    d_out: int,
    *,
    tables: int,
    comparisons: int,
    backend: str,
    seed: int,
    lut_init_std: float,
    accumulation: str,
    max_group_size: int,
    fixed_zero_threshold: bool,
    route_premix: str,
    block_size: int,
    expander_fanout: int,
    sparse_stages: int,
    lowrank_rank: int,
    hashes: int,
) -> nn.Module:
    if route_premix == "multi_hash_structured":
        return MultiHashStructuredPairwiseLayer(
            d_in,
            d_out,
            tables=tables,
            comparisons=comparisons,
            backend=backend,
            seed=seed,
            lut_init_std=lut_init_std,
            accumulation=accumulation,
            max_group_size=max_group_size,
            fixed_zero_threshold=fixed_zero_threshold,
            block_size=block_size,
            hashes=hashes,
        )

    layer = PairwiseLinear(
        d_in,
        d_out,
        tables=tables,
        comparisons=comparisons,
        backend=backend,
        seed=seed,
        lut_init_std=lut_init_std,
        accumulation=accumulation,  # type: ignore[arg-type]
        max_group_size=max_group_size,
        fixed_zero_threshold=fixed_zero_threshold,
    )
    if route_premix == "none":
        return layer
    if route_premix == "block_hadamard":
        mix = BlockHadamardRouteMix(d_in, block_size=block_size, seed=seed + 17)
    elif route_premix == "cyclic_expander":
        mix = CyclicExpanderRouteMix(d_in, fanout=expander_fanout, seed=seed + 17)
    elif route_premix == "learned_butterfly":
        mix = LearnedButterflyRouteMix(d_in, seed=seed + 17)
    elif route_premix == "learnable_cyclic_expander":
        mix = LearnableCyclicExpanderRouteMix(d_in, fanout=expander_fanout, seed=seed + 17)
    elif route_premix == "hadamard_diag_sandwich":
        mix = HadamardDiagonalSandwichRouteMix(d_in, block_size=block_size, seed=seed + 17)
    elif route_premix == "givens_butterfly":
        mix = GivensButterflyRouteMix(d_in, seed=seed + 17)
    elif route_premix == "sparse_product":
        mix = SparseProductRouteMix(d_in, stages=sparse_stages, fanout=expander_fanout, seed=seed + 17)
    elif route_premix == "lowrank":
        mix = LowRankRouteMix(d_in, rank=lowrank_rank, seed=seed + 17)
    else:
        raise ValueError(f"unsupported pairwise route pre-mix {route_premix!r}")
    return RoutePreMixPairwiseLayer(mix, layer)


class CommonModeBypassLayer(nn.Module):
    """Add a minimal common-mode bypass to a routed layer."""

    def __init__(self, layer: nn.Module, *, out_features: int) -> None:
        super().__init__()
        self.layer = layer
        self.common_weight = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x) + x.mean(dim=-1, keepdim=True) * self.common_weight


class EmnistRoutedClassifier(nn.Module):
    def __init__(
        self,
        *,
        family: str,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        depth: int,
        heads: int,
        cells: int,
        code_dim: int,
        route_terms: int,
        fan_value_mode: str,
        fan_basis_rank: int,
        comparisons: int,
        pairwise_tables: int,
        pairwise_lut_init_std: float,
        pairwise_lut_accumulation: str,
        pairwise_max_group_size: int,
        fixed_zero_threshold: bool,
        pairwise_route_premix: str,
        route_premix_block_size: int,
        route_premix_expander_fanout: int,
        route_premix_sparse_stages: int,
        route_premix_lowrank_rank: int,
        pairwise_hashes: int,
        walsh_order: int,
        backend: str,
        seed: int,
        residual: bool = False,
        common_mode_bypass: bool = False,
    ) -> None:
        super().__init__()
        self.family = family
        self.residual = residual
        dims = [input_dim]
        if depth == 1:
            dims.append(num_classes)
        else:
            dims.extend([hidden_dim] * (depth - 1))
            dims.append(num_classes)
        layer_count = len(dims) - 1
        self.residual_mask = [residual and idx < layer_count - 1 for idx in range(layer_count)]
        layers: list[nn.Module] = []
        for idx, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:])):
            layer = _make_layer(
                family,
                d_in,
                d_out,
                heads=heads,
                cells=cells,
                code_dim=code_dim,
                route_terms=route_terms,
                fan_value_mode=fan_value_mode,
                fan_basis_rank=fan_basis_rank,
                comparisons=comparisons,
                pairwise_tables=pairwise_tables,
                pairwise_lut_init_std=pairwise_lut_init_std,
                pairwise_lut_accumulation=pairwise_lut_accumulation,
                pairwise_max_group_size=pairwise_max_group_size,
                fixed_zero_threshold=fixed_zero_threshold,
                pairwise_route_premix=pairwise_route_premix,
                route_premix_block_size=route_premix_block_size,
                route_premix_expander_fanout=route_premix_expander_fanout,
                route_premix_sparse_stages=route_premix_sparse_stages,
                route_premix_lowrank_rank=route_premix_lowrank_rank,
                pairwise_hashes=pairwise_hashes,
                walsh_order=walsh_order,
                backend=backend,
                seed=seed + idx,
            )
            if common_mode_bypass and family == "pairwise":
                layer = CommonModeBypassLayer(layer, out_features=d_out)
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

    @staticmethod
    def _adapt_residual(x: Tensor, out_features: int) -> Tensor:
        in_features = x.shape[-1]
        if in_features == out_features:
            return x
        if in_features > out_features:
            return x[..., :out_features]
        return F.pad(x, (0, out_features - in_features))

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        for use_residual, layer in zip(self.residual_mask, self.layers):
            y = layer(x)
            x = self._adapt_residual(x, y.shape[-1]) + y if use_residual else y
        return x.squeeze(1)


class EmnistTropClassifier(EmnistRoutedClassifier):
    def __init__(self, **kwargs) -> None:
        super().__init__(family="tropical", **kwargs)


class EmnistTropLowRankClassifier(EmnistRoutedClassifier):
    def __init__(self, **kwargs) -> None:
        super().__init__(family="tropical_lowrank", **kwargs)


class EmnistTropZeroDenseClassifier(EmnistRoutedClassifier):
    def __init__(self, **kwargs) -> None:
        super().__init__(family="tropical_zero_dense", **kwargs)


class EmnistTropFanZeroDenseClassifier(EmnistRoutedClassifier):
    def __init__(self, **kwargs) -> None:
        super().__init__(family="tropfan_zero_dense", **kwargs)


class EmnistPairwiseClassifier(EmnistRoutedClassifier):
    def __init__(self, **kwargs) -> None:
        super().__init__(family="pairwise", **kwargs)


class EmnistPairwiseWalshClassifier(EmnistRoutedClassifier):
    def __init__(self, **kwargs) -> None:
        super().__init__(family="pairwise_walsh", **kwargs)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    model.train(mode=optimizer is not None)
    total_loss = 0.0
    total_correct = 0
    total_items = 0
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item()) * x.shape[0]
            total_correct += int((logits.argmax(dim=-1) == y).sum().item())
            total_items += x.shape[0]
    return total_loss / total_items, total_correct / total_items


def _write_metrics(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", choices=EMNIST_SPLITS, default="digits")
    parser.add_argument("--family", choices=ROUTED_FAMILIES, default="tropical")
    for name, arg_type, default in (
        ("--epochs", int, 10),
        ("--batch-size", int, 256),
        ("--lr", float, 3e-3),
        ("--hidden-dim", int, 256),
        ("--depth", int, 2),
        ("--heads", int, 32),
        ("--cells", int, 4),
        ("--code-dim", int, 32),
        ("--route-terms", int, 2),
        ("--fan-basis-rank", int, 16),
        ("--pairwise-tables", int, 72),
        ("--pairwise-lut-init-std", float, 0.0),
        ("--pairwise-max-group-size", int, 4),
        ("--comparisons", int, 6),
    ):
        parser.add_argument(name, type=arg_type, default=default)
    parser.add_argument("--walsh-order", type=int, choices=(1, 2), default=2)
    parser.add_argument("--backend", choices=("torch", "auto", "triton", "tilelang"), default="torch")
    parser.add_argument("--fan-value-mode", choices=("site", "basis"), default="site")
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--residual", action="store_true", help="Add a fixed crop/pad residual path on all non-output layers.")
    parser.add_argument("--fixed-zero-threshold", action="store_true", help="Use fixed zero pairwise thresholds instead of learnable offsets.")
    parser.add_argument("--common-mode-bypass", action="store_true", help="Add y += mean(x) * v to each PairwiseLinear layer.")
    parser.add_argument("--pairwise-lut-accumulation", choices=("sum", "two_bank_max"), default="sum")
    parser.add_argument("--pairwise-route-premix", choices=PAIRWISE_ROUTE_PREMIXES, default="none")
    parser.add_argument("--route-premix-block-size", type=int, default=64)
    parser.add_argument("--route-premix-expander-fanout", type=int, default=4)
    parser.add_argument("--route-premix-sparse-stages", type=int, default=2)
    parser.add_argument("--route-premix-lowrank-rank", type=int, default=4)
    parser.add_argument("--pairwise-hashes", type=int, default=4)
    parser.add_argument("--permute", action="store_true")
    parser.add_argument("--permute-seed", type=int, default=0)
    parser.add_argument("--raw-orientation", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    x_train, y_train = load_emnist_split(
        args.root,
        args.split,
        train=True,
        limit=args.max_train,
        fix_orientation=not args.raw_orientation,
        permute=args.permute,
        permute_seed=args.permute_seed,
    )
    x_test, y_test = load_emnist_split(
        args.root,
        args.split,
        train=False,
        limit=args.max_test,
        fix_orientation=not args.raw_orientation,
        permute=args.permute,
        permute_seed=args.permute_seed,
    )
    num_classes = int(max(y_train.max().item(), y_test.max().item()) + 1)
    model = EmnistRoutedClassifier(
        family=args.family,
        input_dim=x_train.shape[1],
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        depth=args.depth,
        heads=args.heads,
        cells=args.cells,
        code_dim=args.code_dim,
        route_terms=args.route_terms,
        fan_value_mode=args.fan_value_mode,
        fan_basis_rank=args.fan_basis_rank,
        comparisons=args.comparisons,
        pairwise_tables=args.pairwise_tables,
        pairwise_lut_init_std=args.pairwise_lut_init_std,
        pairwise_lut_accumulation=args.pairwise_lut_accumulation,
        pairwise_max_group_size=args.pairwise_max_group_size,
        fixed_zero_threshold=args.fixed_zero_threshold,
        pairwise_route_premix=args.pairwise_route_premix,
        route_premix_block_size=args.route_premix_block_size,
        route_premix_expander_fanout=args.route_premix_expander_fanout,
        route_premix_sparse_stages=args.route_premix_sparse_stages,
        route_premix_lowrank_rank=args.route_premix_lowrank_rank,
        pairwise_hashes=args.pairwise_hashes,
        walsh_order=args.walsh_order,
        backend=args.backend,
        seed=args.seed,
        residual=args.residual,
        common_mode_bypass=args.common_mode_bypass,
    ).to(device)
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=args.batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    config_lines = {
        "root": args.root,
        "split": args.split,
        "family": args.family,
        "depth": args.depth,
        "hidden_dim": args.hidden_dim,
        "heads": args.heads if args.family in HEAD_ROUTED_FAMILIES else "-",
        "cells": args.cells if args.family in HEAD_ROUTED_FAMILIES else "-",
        "code_dim": args.code_dim if args.family in TROPICAL_FAMILIES or args.family == "tropfan_zero_dense" else "-",
        "route_terms": args.route_terms if args.family == "tropical_zero_dense" else "-",
        "fan_value_mode": args.fan_value_mode if args.family == "tropfan_zero_dense" else "-",
        "fan_basis_rank": args.fan_basis_rank if args.family == "tropfan_zero_dense" else "-",
        "pairwise_tables": args.pairwise_tables if args.family in {"pairwise", "pairwise_walsh"} else "-",
        "pairwise_lut_init_std": args.pairwise_lut_init_std if args.family == "pairwise" else "-",
        "pairwise_lut_accum": args.pairwise_lut_accumulation if args.family == "pairwise" else "-",
        "pairwise_max_group": args.pairwise_max_group_size if args.family == "pairwise" and args.pairwise_lut_accumulation == "two_bank_max" else "-",
        "comparisons": args.comparisons if args.family in {"pairwise", "pairwise_walsh"} else "-",
        "fixed_zero_threshold": args.fixed_zero_threshold if args.family == "pairwise" else "-",
        "common_bypass": args.common_mode_bypass if args.family == "pairwise" else "-",
        "route_premix": args.pairwise_route_premix if args.family == "pairwise" else "-",
        "premix_block": args.route_premix_block_size if args.family == "pairwise" else "-",
        "premix_fanout": args.route_premix_expander_fanout if args.family == "pairwise" else "-",
        "premix_stages": args.route_premix_sparse_stages if args.family == "pairwise" and args.pairwise_route_premix == "sparse_product" else "-",
        "premix_rank": args.route_premix_lowrank_rank if args.family == "pairwise" and args.pairwise_route_premix == "lowrank" else "-",
        "pairwise_hashes": args.pairwise_hashes if args.family == "pairwise" and args.pairwise_route_premix == "multi_hash_structured" else "-",
        "walsh_order": args.walsh_order if args.family == "pairwise_walsh" else "-",
        "backend": args.backend if args.family in TROPICAL_FAMILIES or args.family in {"pairwise", "pairwise_walsh"} else "torch",
        "residual": args.residual,
        "train/test": f"{len(x_train)}/{len(x_test)}",
        "device": device.type,
        "params": sum(param.numel() for param in model.parameters()),
    }
    config_text = "\n".join(f"  {key:<15} : {value}" for key, value in config_lines.items())
    print(f"EMNIST tropnn\n{config_text}\n")

    rows: list[dict] = []
    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = _run_epoch(model, train_loader, device, optimizer)
        test_loss, test_acc = _run_epoch(model, test_loader, device)
        rows.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "test_loss": test_loss, "test_acc": test_acc})
        print(f"epoch {epoch:>3d} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} | test_loss={test_loss:.4f} test_acc={test_acc:.4f}")

    repo_root = Path(__file__).resolve().parents[4]
    out_path = repo_root / "results" / "experiments" / "tropnn_emnist" / f"{args.split}_{args.family}_{time.time_ns()}.csv"
    _write_metrics(rows, out_path)
    print(f"\nDone in {time.perf_counter() - t0:.1f}s; metrics -> {out_path}")


if __name__ == "__main__":
    main()
