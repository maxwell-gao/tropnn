from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .s4_relation import s4_tables
from .surrogate import ste_heaviside

__all__ = [
    "BalancedS4Router",
    "CoxeterPairScorer",
    "GlobalChamberKernel",
    "IntrinsicS4Kernel",
    "RootIncidenceKernel",
    "S4ObjectFeatures",
    "SameTableFullKernel",
    "coxeter_representation_features",
]


S4_ORDER = 24
_PERMUTATIONS = tuple(itertools.permutations(range(4)))
_K4_EDGES = tuple(itertools.combinations(range(4), 2))


def _permutation_rank(order: Tensor) -> Tensor:
    rank = torch.zeros(order.shape[:-1], device=order.device, dtype=torch.long)
    for position, factorial in enumerate((6, 2, 1)):
        smaller = (order[..., position + 1 :] < order[..., position : position + 1]).sum(dim=-1)
        rank += smaller * factorial
    return rank


def _orthonormalize_feature_columns(raw: Tensor) -> Tensor:
    if int(torch.linalg.matrix_rank(raw).item()) != raw.shape[1]:
        raise ValueError("feature columns must be linearly independent")
    orthogonal, _ = torch.linalg.qr(raw, mode="reduced")
    if orthogonal[:, 0].sum() < 0:
        orthogonal[:, 0] *= -1
    return orthogonal * math.sqrt(raw.shape[0])


def coxeter_representation_features(device: torch.device | str | None = None) -> Tensor:
    """Twelve shared S4 features used by the Global Coxeter factorization."""

    basis = torch.tensor(
        (
            (1.0 / math.sqrt(2.0), 1.0 / math.sqrt(6.0), 1.0 / math.sqrt(12.0)),
            (-1.0 / math.sqrt(2.0), 1.0 / math.sqrt(6.0), 1.0 / math.sqrt(12.0)),
            (0.0, -2.0 / math.sqrt(6.0), 1.0 / math.sqrt(12.0)),
            (0.0, 0.0, -3.0 / math.sqrt(12.0)),
        ),
        dtype=torch.float64,
    )
    _, _, lengths = s4_tables()
    rows: list[Tensor] = []
    for index, permutation in enumerate(_PERMUTATIONS):
        matrix = torch.zeros(4, 4, dtype=torch.float64)
        matrix[torch.arange(4), torch.tensor(permutation)] = 1.0
        standard = basis.T @ matrix @ basis
        rows.append(
            torch.cat(
                (
                    torch.ones(1, dtype=torch.float64),
                    standard.reshape(-1),
                    torch.tensor(((-1.0) ** int(lengths[index]),), dtype=torch.float64),
                    (lengths[index].to(torch.float64) - 3.0).view(1),
                )
            )
        )
    return _orthonormalize_feature_columns(torch.stack(rows)).to(dtype=torch.float32, device=device)


@dataclass(frozen=True)
class S4ObjectFeatures:
    coordinates: Tensor
    routes: Tensor
    orders: Tensor
    adjacent_gaps: Tensor
    roots: Tensor


class BalancedS4Router(nn.Module):
    """Balanced overlapping K4 charts over a compact ordinal coordinate vector.

    The default D32/T16 layout is made from two independently seeded
    permutations.  Every coordinate therefore occurs in exactly two charts,
    which makes coordinate-incidence support cross table boundaries.
    """

    def __init__(self, input_dim: int = 32, tables: int = 16, *, coverage: int = 2, seed: int = 0) -> None:
        super().__init__()
        if input_dim < 4 or tables < 1 or coverage < 1:
            raise ValueError("input_dim >= 4, tables >= 1, and coverage >= 1 are required")
        if 4 * tables != coverage * input_dim:
            raise ValueError("balanced charts require 4 * tables == coverage * input_dim")
        self.input_dim = int(input_dim)
        self.tables = int(tables)
        self.coverage = int(coverage)

        generator = torch.Generator(device="cpu").manual_seed(seed + 1709)
        anchors = torch.cat([torch.randperm(input_dim, generator=generator) for _ in range(coverage)]).view(tables, 4)
        orders = torch.tensor(_PERMUTATIONS, dtype=torch.long)
        neighbours = torch.empty(S4_ORDER, 3, dtype=torch.long)
        lookup = {permutation: index for index, permutation in enumerate(_PERMUTATIONS)}
        for state, permutation in enumerate(_PERMUTATIONS):
            for generator_index in range(3):
                neighbour = list(permutation)
                neighbour[generator_index], neighbour[generator_index + 1] = (
                    neighbour[generator_index + 1],
                    neighbour[generator_index],
                )
                neighbours[state, generator_index] = lookup[tuple(neighbour)]

        edge_keys = sorted(
            {
                tuple(sorted((int(row[left]), int(row[right]))))
                for row in anchors
                for left, right in _K4_EDGES
            }
        )
        edge_lookup = {edge: index for index, edge in enumerate(edge_keys)}
        adjacent_root_edge = torch.empty(tables, S4_ORDER, 3, dtype=torch.long)
        for table in range(tables):
            for state, permutation in enumerate(_PERMUTATIONS):
                for generator_index in range(3):
                    left = int(anchors[table, permutation[generator_index]])
                    right = int(anchors[table, permutation[generator_index + 1]])
                    adjacent_root_edge[table, state, generator_index] = edge_lookup[tuple(sorted((left, right)))]

        edge_tensor = torch.tensor(edge_keys, dtype=torch.long)
        incidence = torch.zeros(input_dim, len(edge_keys), dtype=torch.float32)
        incidence[edge_tensor[:, 0], torch.arange(len(edge_keys))] = -1.0
        incidence[edge_tensor[:, 1], torch.arange(len(edge_keys))] = 1.0
        support_rows, support_columns = torch.nonzero(incidence.T @ incidence != 0.0, as_tuple=True)

        self.register_buffer("anchors", anchors)
        self.register_buffer("permutation_orders", orders)
        self.register_buffer("neighbours", neighbours)
        self.register_buffer("root_edges", edge_tensor)
        self.register_buffer("root_incidence", incidence)
        self.register_buffer("support_rows", support_rows)
        self.register_buffer("support_columns", support_columns)
        self.register_buffer("adjacent_root_edge", adjacent_root_edge)

    @property
    def roots(self) -> int:
        return int(self.root_edges.shape[0])

    @property
    def incidence_entries(self) -> int:
        return int(self.support_rows.numel())

    def route(self, coordinates: Tensor) -> S4ObjectFeatures:
        if coordinates.ndim != 2 or coordinates.shape[-1] != self.input_dim:
            raise ValueError(f"expected coordinates [batch, {self.input_dim}], got {tuple(coordinates.shape)}")
        selected = coordinates[:, self.anchors.flatten()].view(coordinates.shape[0], self.tables, 4)
        order = torch.argsort(selected, dim=-1, stable=True)
        sorted_values = selected.gather(-1, order)
        routes = _permutation_rank(order)
        edges = self.root_edges
        signs = torch.where(
            coordinates[:, edges[:, 0]] > coordinates[:, edges[:, 1]],
            1.0,
            -1.0,
        )
        signs = signs / math.sqrt(max(1, self.roots))
        return S4ObjectFeatures(coordinates, routes, order, sorted_values[..., 1:] - sorted_values[..., :-1], signs)

    def neighbour(self, features: S4ObjectFeatures, table: int, generator: Tensor) -> S4ObjectFeatures:
        if generator.shape != (features.routes.shape[0],):
            raise ValueError("generator must contain one adjacent generator per object")
        batch = torch.arange(features.routes.shape[0], device=features.routes.device)
        old_route = features.routes[:, table]
        new_routes = features.routes.clone()
        new_routes[:, table] = self.neighbours[old_route, generator]
        edge = self.adjacent_root_edge[table, old_route, generator]
        new_roots = features.roots.clone()
        new_roots[batch, edge] *= -1.0
        return S4ObjectFeatures(
            features.coordinates,
            new_routes,
            features.orders,
            features.adjacent_gaps,
            new_roots,
        )


class PairKernelBase(nn.Module):
    def hard_score(self, query: S4ObjectFeatures, key: S4ObjectFeatures) -> Tensor:
        raise NotImplementedError


class IntrinsicS4Kernel(PairKernelBase):
    def __init__(self, tables: int, kind: str = "kendall", *, mallows_beta: float = 0.75) -> None:
        super().__init__()
        if kind not in {"kendall", "mallows"}:
            raise ValueError("kind must be kendall or mallows")
        inverse, composition, length = s4_tables()
        states = torch.arange(S4_ORDER)
        relative = composition[inverse[:, None], states[None, :]]
        distance = length[relative].to(torch.float32)
        table = 1.0 - distance / 3.0 if kind == "kendall" else torch.exp(-mallows_beta * distance)
        table = (table - table.mean()) / table.std(unbiased=False).clamp_min(1e-12)
        self.tables = int(tables)
        self.kind = kind
        self.register_buffer("kernel_table", table)
        self.scale = nn.Parameter(torch.ones(()))
        self.bias = nn.Parameter(torch.zeros(()))

    def hard_score(self, query: S4ObjectFeatures, key: S4ObjectFeatures) -> Tensor:
        values = self.kernel_table[query.routes, key.routes].mean(dim=-1)
        return self.scale * values + self.bias


class SameTableFullKernel(PairKernelBase):
    def __init__(self, tables: int, *, init_std: float = 0.02, seed: int = 0) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(seed + 211)
        self.tables = int(tables)
        self.weight = nn.Parameter(torch.randn(tables, S4_ORDER, S4_ORDER, generator=generator) * init_std)
        self.bias = nn.Parameter(torch.zeros(()))

    def hard_score(self, query: S4ObjectFeatures, key: S4ObjectFeatures) -> Tensor:
        table = torch.arange(self.tables, device=query.routes.device).view(1, -1)
        values = self.weight[table, query.routes, key.routes]
        return values.sum(dim=-1) / math.sqrt(self.tables) + self.bias


class GlobalChamberKernel(PairKernelBase):
    def __init__(
        self,
        tables: int,
        rank: int,
        *,
        shared_coxeter: bool = False,
        init_std: float = 0.02,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be positive")
        self.tables = int(tables)
        self.rank = int(rank)
        self.shared_coxeter = bool(shared_coxeter)
        generator = torch.Generator(device="cpu").manual_seed(seed + 307)
        if shared_coxeter:
            features = coxeter_representation_features()
            self.register_buffer("feature_table", features)
            shape = (tables, features.shape[1], rank)
        else:
            self.register_buffer("feature_table", torch.empty(0))
            shape = (tables, S4_ORDER, rank)
        scale = init_std / math.sqrt(max(1, tables))
        self.query_factor = nn.Parameter(torch.randn(shape, generator=generator) * scale)
        self.key_factor = nn.Parameter(torch.randn(shape, generator=generator) * scale)
        self.bias = nn.Parameter(torch.zeros(()))

    def _rows(self, routes: Tensor, factor: Tensor) -> Tensor:
        table = torch.arange(self.tables, device=routes.device).view(1, -1)
        if self.shared_coxeter:
            route_features = self.feature_table[routes]
            rows = torch.einsum("ntd,tdr->ntr", route_features, factor)
        else:
            rows = factor[table, routes]
        return rows.sum(dim=1) / math.sqrt(self.tables)

    def object_embeddings(self, features: S4ObjectFeatures) -> tuple[Tensor, Tensor]:
        return self._rows(features.routes, self.query_factor), self._rows(features.routes, self.key_factor)

    def hard_score(self, query: S4ObjectFeatures, key: S4ObjectFeatures) -> Tensor:
        query_embedding = self._rows(query.routes, self.query_factor)
        key_embedding = self._rows(key.routes, self.key_factor)
        return (query_embedding * key_embedding).sum(dim=-1) / math.sqrt(self.rank) + self.bias


class RootIncidenceKernel(PairKernelBase):
    def __init__(self, router: BalancedS4Router, *, diagonal: bool = False, init_std: float = 0.02, seed: int = 0) -> None:
        super().__init__()
        self.diagonal = bool(diagonal)
        if diagonal:
            rows = torch.arange(router.roots)
            columns = rows.clone()
        else:
            rows = router.support_rows.detach().cpu()
            columns = router.support_columns.detach().cpu()
        generator = torch.Generator(device="cpu").manual_seed(seed + 401)
        self.register_buffer("rows", rows)
        self.register_buffer("columns", columns)
        self.weight = nn.Parameter(torch.randn(rows.numel(), generator=generator) * init_std)
        self.bias = nn.Parameter(torch.zeros(()))

    def hard_score(self, query: S4ObjectFeatures, key: S4ObjectFeatures) -> Tensor:
        products = query.roots[:, self.rows] * key.roots[:, self.columns]
        return products @ self.weight + self.bias

    def dense_operator(self, roots: int) -> Tensor:
        operator = torch.zeros(roots, roots, device=self.weight.device, dtype=self.weight.dtype)
        operator[self.rows, self.columns] = self.weight
        return operator

    def transform_roots(self, roots: Tensor, *, transpose: bool = False) -> Tensor:
        """Cache ``M c`` using only supported entries and scatter-adds."""

        source = self.rows if transpose else self.columns
        destination = self.columns if transpose else self.rows
        result = torch.zeros_like(roots)
        result.scatter_add_(1, destination.view(1, -1).expand(roots.shape[0], -1), roots[:, source] * self.weight)
        return result

    def cached_score(self, query_roots: Tensor, key_roots: Tensor, *, symmetry: str = "none") -> Tensor:
        return self.score_from_cache(
            query_roots,
            key_roots,
            self.transform_roots(query_roots),
            self.transform_roots(key_roots),
            symmetry=symmetry,
        )

    def score_from_cache(
        self,
        query_roots: Tensor,
        key_roots: Tensor,
        transformed_query_roots: Tensor,
        transformed_key_roots: Tensor,
        *,
        symmetry: str = "none",
    ) -> Tensor:
        """Score pairs after each object's sparse ``M c`` transform is cached."""

        forward = (query_roots * transformed_key_roots).sum(dim=-1) + self.bias
        if symmetry == "none":
            return forward
        reverse = (key_roots * transformed_query_roots).sum(dim=-1) + self.bias
        if symmetry == "symmetric":
            return 0.5 * (forward + reverse)
        if symmetry == "antisymmetric":
            return 0.5 * (forward - reverse)
        raise ValueError("symmetry must be none, symmetric, or antisymmetric")


class CoxeterPairScorer(nn.Module):
    """Apply a hard S4 pair kernel with an exact-forward adjacent-wall STE."""

    def __init__(self, router: BalancedS4Router, kernel: PairKernelBase, *, symmetry: str = "none") -> None:
        super().__init__()
        if symmetry not in {"none", "symmetric", "antisymmetric"}:
            raise ValueError("symmetry must be none, symmetric, or antisymmetric")
        self.router = router
        self.kernel = kernel
        self.symmetry = symmetry

    def _score_features(self, query: S4ObjectFeatures, key: S4ObjectFeatures) -> Tensor:
        forward = self.kernel.hard_score(query, key)
        if self.symmetry == "none":
            return forward
        reverse = self.kernel.hard_score(key, query)
        if self.symmetry == "symmetric":
            return 0.5 * (forward + reverse)
        return 0.5 * (forward - reverse)

    def _ste_correction(
        self,
        query: S4ObjectFeatures,
        key: S4ObjectFeatures,
        hard_score: Tensor,
    ) -> Tensor:
        correction = torch.zeros_like(hard_score)
        for table in range(self.router.tables):
            query_generator = query.adjacent_gaps[:, table].argmin(dim=-1)
            query_gap = query.adjacent_gaps[
                torch.arange(query.routes.shape[0], device=query.routes.device), table, query_generator
            ]
            query_neighbour = self.router.neighbour(query, table, query_generator)
            query_delta = self._score_features(query_neighbour, key) - hard_score
            query_gate = ste_heaviside(query_gap) - (query_gap > 0).to(query_gap.dtype)
            correction = correction + query_gate * query_delta

            key_generator = key.adjacent_gaps[:, table].argmin(dim=-1)
            key_gap = key.adjacent_gaps[
                torch.arange(key.routes.shape[0], device=key.routes.device), table, key_generator
            ]
            key_neighbour = self.router.neighbour(key, table, key_generator)
            key_delta = self._score_features(query, key_neighbour) - hard_score
            key_gate = ste_heaviside(key_gap) - (key_gap > 0).to(key_gap.dtype)
            correction = correction + key_gate * key_delta
        return correction

    def forward(self, query_coordinates: Tensor, key_coordinates: Tensor) -> Tensor:
        query = self.router.route(query_coordinates)
        key = self.router.route(key_coordinates)
        hard_score = self._score_features(query, key)
        if self.training and (query_coordinates.requires_grad or key_coordinates.requires_grad):
            hard_score = hard_score + self._ste_correction(query, key, hard_score)
        return hard_score
