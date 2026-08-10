"""Prepared single-particle group types and archetype-aware planning."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, cast

import numpy as np

from pyceles.core.indexing import n_modes
from pyceles.core.particles import Particle, ParticleCollection, ParticleTRepresentation

Array = np.ndarray
COMPLEX128_DTYPE = np.dtype(np.complex128)


@dataclass(frozen=True)
class ParticleTGroupPlan:
    """Planned instances sharing one prepared ``T`` representation.

    ``archetype_indices`` identifies the unique collection archetypes that need
    one prepared operator each. ``operator_indices`` is aligned with
    ``particle_indices`` and maps every instance to a local prepared operator.
    """

    representation: ParticleTRepresentation
    particle_indices: Array
    archetype_indices: Array
    operator_indices: Array


@dataclass(frozen=True)
class ParticleTPreparationContext:
    """Shared preparation inputs for representation-specific group factories."""

    lmax: int
    k: float
    particles: ParticleCollection
    n_medium: complex
    dtype: np.dtype

    @property
    def n_modes(self) -> int:
        return n_modes(self.lmax)

    def archetypes_for(self, plan: ParticleTGroupPlan) -> tuple[Particle, ...]:
        return tuple(
            self.particles.archetypes[int(index)]
            for index in np.asarray(plan.archetype_indices, dtype=np.int64)
        )

    def instances_for(self, plan: ParticleTGroupPlan) -> tuple[Particle, ...]:
        return tuple(
            self.particles[int(index)]
            for index in np.asarray(plan.particle_indices, dtype=np.int64)
        )


class PreparedParticleTGroup(Protocol):
    """Prepared subset of particles sharing one ``T`` representation."""

    particle_indices: Array
    operator_indices: Array
    dtype: np.dtype

    def apply_subset(self, x_subset: Array) -> Array: ...

    def rhs_subset(self, b_subset: Array) -> Array: ...

    def apply_local_block(self, local_particle_index: int, block: Array) -> Array: ...

    def mode_diagonal(self) -> Array | None: ...

    def degree_diagonals(self) -> tuple[Array, Array] | None: ...


type ParticleTGroupFactory = Callable[
    [ParticleTGroupPlan, ParticleTPreparationContext], PreparedParticleTGroup
]
type DenseBlockProvider = Callable[[Sequence[Particle], ParticleTPreparationContext], Array]
type AxisymmetricSubsetApply = Callable[
    [Array, Sequence[Particle], ParticleTPreparationContext], Array
]
type AxisymmetricLocalBlockApply = Callable[
    [int, Array, Sequence[Particle], ParticleTPreparationContext], Array
]
type AxisymmetricMetadataBuilder = Callable[
    [Sequence[Particle], ParticleTPreparationContext], object | None
]


@dataclass(frozen=True)
class ParticleTGroupFactories:
    """Optional representation-specific group factories for ``prepare_matvec``."""

    diagonal: ParticleTGroupFactory | None = None
    axisymmetric: ParticleTGroupFactory | None = None
    dense: ParticleTGroupFactory | None = None

    def for_representation(
        self, representation: ParticleTRepresentation
    ) -> ParticleTGroupFactory | None:
        if representation == "diagonal":
            return self.diagonal
        if representation == "axisymmetric":
            return self.axisymmetric
        if representation == "dense":
            return self.dense
        raise ValueError(f"Unsupported particle-T representation {representation!r}.")


def _normalized_operator_indices(
    particle_indices: Array,
    operator_indices: Array | None,
    *,
    n_operators: int,
) -> tuple[Array, Array]:
    particles = np.asarray(particle_indices, dtype=np.int64).reshape(-1)
    if operator_indices is None:
        if n_operators == particles.size:
            operators = np.arange(particles.size, dtype=np.int64)
        elif n_operators == 1:
            operators = np.zeros(particles.size, dtype=np.int64)
        else:
            raise ValueError(
                "A shared T group requires an explicit instance-to-operator map. "
                f"Got {particles.size} particles and {n_operators} operators."
            )
    else:
        operators = np.asarray(operator_indices, dtype=np.int64).reshape(-1)
    if operators.size != particles.size:
        raise ValueError(
            "`operator_indices` must align with `particle_indices`. "
            f"Got {operators.size} and {particles.size}."
        )
    if operators.size and (int(operators.min()) < 0 or int(operators.max()) >= int(n_operators)):
        raise IndexError("`operator_indices` contains an out-of-range operator id.")
    return particles, operators


def _mapped_rows(values: Array, operator_indices: Array) -> Array:
    """Expand mapped rows only for diagnostics that explicitly require them."""
    rows = np.asarray(values)
    ids = np.asarray(operator_indices, dtype=np.int64)
    if rows.shape[0] == ids.size and np.array_equal(ids, np.arange(ids.size)):
        return rows
    return rows[ids]


def _apply_shared_diagonal(values: Array, operator_indices: Array, x_subset: Array) -> Array:
    """Apply shared diagonal rows without retaining an expanded diagonal table."""
    rows = np.asarray(values)
    ids = np.asarray(operator_indices, dtype=np.int64)
    arr = np.asarray(x_subset)
    trailing = (1,) * max(0, arr.ndim - 2)
    if rows.shape[0] == 1:
        return cast(Array, rows[0].reshape((1, rows.shape[1], *trailing)) * arr)
    if rows.shape[0] == ids.size and np.array_equal(ids, np.arange(ids.size)):
        return cast(Array, rows.reshape((*rows.shape, *trailing)) * arr)
    out = np.empty_like(arr)
    row_shape = (1, rows.shape[1], *trailing)
    for operator_index in range(rows.shape[0]):
        selected = ids == operator_index
        out[selected] = rows[operator_index].reshape(row_shape) * arr[selected]
    return out


def _apply_shared_dense(blocks: Array, operator_indices: Array, x_subset: Array) -> Array:
    """Apply shared dense blocks without constructing ``blocks[operator_indices]``."""
    block_rows = np.asarray(blocks)
    ids = np.asarray(operator_indices, dtype=np.int64)
    arr = np.asarray(x_subset)
    if arr.ndim not in {2, 3}:
        raise ValueError(f"Dense particle-T subsets must be 2D or 3D. Got {arr.shape}.")
    shared_subscripts = "ij,gj->gi" if arr.ndim == 2 else "ij,gjr->gir"
    mapped_subscripts = "gij,gj->gi" if arr.ndim == 2 else "gij,gjr->gir"
    if block_rows.shape[0] == 1:
        return cast(
            Array,
            np.einsum(shared_subscripts, block_rows[0], arr, optimize=True),
        )
    if block_rows.shape[0] == ids.size and np.array_equal(ids, np.arange(ids.size)):
        return cast(Array, np.einsum(mapped_subscripts, block_rows, arr, optimize=True))
    out = np.empty_like(arr)
    for operator_index in range(block_rows.shape[0]):
        selected = ids == operator_index
        out[selected] = np.einsum(
            shared_subscripts,
            block_rows[operator_index],
            arr[selected],
            optimize=True,
        )
    return out


@dataclass
class DiagonalTGroup:
    particle_indices: Array
    T_M: Array
    T_N: Array
    T_diag: Array
    operator_indices: Array = field(default_factory=lambda: np.zeros((0,), dtype=np.int64))
    dtype: np.dtype = COMPLEX128_DTYPE

    def __post_init__(self) -> None:
        self.T_M = np.asarray(self.T_M, dtype=self.dtype)
        self.T_N = np.asarray(self.T_N, dtype=self.dtype)
        self.T_diag = np.asarray(self.T_diag, dtype=self.dtype)
        if self.T_M.shape[0] != self.T_N.shape[0] or self.T_M.shape[0] != self.T_diag.shape[0]:
            raise ValueError("Diagonal T data must have one row per prepared archetype.")
        self.particle_indices, self.operator_indices = _normalized_operator_indices(
            self.particle_indices,
            None if self.operator_indices.size == 0 else self.operator_indices,
            n_operators=self.T_diag.shape[0],
        )

    def apply_subset(self, x_subset: Array) -> Array:
        return np.asarray(
            _apply_shared_diagonal(
                self.T_diag,
                self.operator_indices,
                np.asarray(x_subset, dtype=self.dtype),
            ),
            dtype=self.dtype,
        )

    def rhs_subset(self, b_subset: Array) -> Array:
        return self.apply_subset(b_subset)

    def apply_local_block(self, local_particle_index: int, block: Array) -> Array:
        operator_index = int(self.operator_indices[int(local_particle_index)])
        return np.asarray(
            self.T_diag[operator_index][:, None] * np.asarray(block, dtype=self.dtype),
            dtype=self.dtype,
        )

    def mode_diagonal(self) -> Array | None:
        return np.asarray(_mapped_rows(self.T_diag, self.operator_indices), dtype=self.dtype)

    def degree_diagonals(self) -> tuple[Array, Array] | None:
        return (
            np.asarray(_mapped_rows(self.T_M, self.operator_indices), dtype=self.dtype),
            np.asarray(_mapped_rows(self.T_N, self.operator_indices), dtype=self.dtype),
        )


@dataclass
class DenseTGroup:
    particle_indices: Array
    T_blocks: Array
    operator_indices: Array = field(default_factory=lambda: np.zeros((0,), dtype=np.int64))
    dtype: np.dtype = COMPLEX128_DTYPE

    def __post_init__(self) -> None:
        blocks = np.asarray(self.T_blocks, dtype=self.dtype)
        if blocks.ndim != 3 or blocks.shape[1] != blocks.shape[2]:
            raise ValueError(f"`T_blocks` must have shape (Nu, Nm, Nm). Got {blocks.shape}.")
        self.particle_indices, self.operator_indices = _normalized_operator_indices(
            self.particle_indices,
            None if self.operator_indices.size == 0 else self.operator_indices,
            n_operators=blocks.shape[0],
        )
        self.T_blocks = blocks

    def apply_subset(self, x_subset: Array) -> Array:
        return np.asarray(
            _apply_shared_dense(
                self.T_blocks,
                self.operator_indices,
                np.asarray(x_subset, dtype=self.dtype),
            ),
            dtype=self.dtype,
        )

    def rhs_subset(self, b_subset: Array) -> Array:
        return self.apply_subset(b_subset)

    def apply_local_block(self, local_particle_index: int, block: Array) -> Array:
        operator_index = int(self.operator_indices[int(local_particle_index)])
        return np.asarray(
            self.T_blocks[operator_index] @ np.asarray(block, dtype=self.dtype),
            dtype=self.dtype,
        )

    def mode_diagonal(self) -> Array | None:
        return None

    def degree_diagonals(self) -> tuple[Array, Array] | None:
        return None


@dataclass
class AxisymmetricTGroup:
    particle_indices: Array
    T_blocks: Array | None = None
    operator_indices: Array = field(default_factory=lambda: np.zeros((0,), dtype=np.int64))
    apply_subset_fn: Callable[[Array], Array] | None = None
    rhs_subset_fn: Callable[[Array], Array] | None = None
    apply_local_block_fn: Callable[[int, Array], Array] | None = None
    body_metadata: object | None = None
    dtype: np.dtype = COMPLEX128_DTYPE

    def __post_init__(self) -> None:
        if self.T_blocks is not None:
            blocks = np.asarray(self.T_blocks, dtype=self.dtype)
            if blocks.ndim != 3 or blocks.shape[1] != blocks.shape[2]:
                raise ValueError(
                    "Axisymmetric dense fallback must provide blocks of shape (Nu, Nm, Nm)."
                )
            self.particle_indices, self.operator_indices = _normalized_operator_indices(
                self.particle_indices,
                None if self.operator_indices.size == 0 else self.operator_indices,
                n_operators=blocks.shape[0],
            )
            self.T_blocks = blocks
        else:
            self.particle_indices = np.asarray(self.particle_indices, dtype=np.int64).reshape(-1)
            self.operator_indices = np.arange(self.particle_indices.size, dtype=np.int64)

    def apply_subset(self, x_subset: Array) -> Array:
        if self.T_blocks is not None:
            return np.asarray(
                _apply_shared_dense(
                    self.T_blocks,
                    self.operator_indices,
                    np.asarray(x_subset, dtype=self.dtype),
                ),
                dtype=self.dtype,
            )
        if self.apply_subset_fn is None:
            raise NotImplementedError(
                "Axisymmetric particle-T operators are planned but not implemented yet."
            )
        return np.asarray(
            self.apply_subset_fn(np.asarray(x_subset, dtype=self.dtype)), dtype=self.dtype
        )

    def rhs_subset(self, b_subset: Array) -> Array:
        if self.T_blocks is not None:
            return self.apply_subset(b_subset)
        if self.rhs_subset_fn is not None:
            return np.asarray(
                self.rhs_subset_fn(np.asarray(b_subset, dtype=self.dtype)), dtype=self.dtype
            )
        return self.apply_subset(b_subset)

    def apply_local_block(self, local_particle_index: int, block: Array) -> Array:
        if self.T_blocks is not None:
            operator_index = int(self.operator_indices[int(local_particle_index)])
            return np.asarray(
                self.T_blocks[operator_index] @ np.asarray(block, dtype=self.dtype),
                dtype=self.dtype,
            )
        if self.apply_local_block_fn is None:
            raise NotImplementedError(
                "Axisymmetric particle-T operators are planned but not implemented yet."
            )
        return np.asarray(
            self.apply_local_block_fn(
                int(local_particle_index), np.asarray(block, dtype=self.dtype)
            ),
            dtype=self.dtype,
        )

    def mode_diagonal(self) -> Array | None:
        return None

    def degree_diagonals(self) -> tuple[Array, Array] | None:
        return None


def plan_particle_t_groups(particles: Sequence[Particle]) -> tuple[ParticleTGroupPlan, ...]:
    """Plan particle-local groups and retain shared-archetype mappings."""
    collection = ParticleCollection.from_particles(particles)
    return tuple(
        ParticleTGroupPlan(
            representation=group.representation,
            particle_indices=np.asarray(group.particle_indices, dtype=np.int64),
            archetype_indices=np.asarray(group.archetype_indices, dtype=np.int64),
            operator_indices=np.asarray(group.operator_indices, dtype=np.int64),
        )
        for group in collection.archetype_groups()
    )


def make_dense_group_factory(block_provider: DenseBlockProvider) -> ParticleTGroupFactory:
    """Build a dense group from one block per unique particle archetype."""

    def factory(
        plan: ParticleTGroupPlan, context: ParticleTPreparationContext
    ) -> PreparedParticleTGroup:
        ids = np.asarray(plan.particle_indices, dtype=np.int64)
        archetypes = context.archetypes_for(plan)
        return DenseTGroup(
            particle_indices=ids,
            operator_indices=np.asarray(plan.operator_indices, dtype=np.int64),
            T_blocks=np.asarray(block_provider(archetypes, context), dtype=context.dtype),
            dtype=context.dtype,
        )

    return factory


def make_axisymmetric_group_factory(
    *,
    apply_subset: AxisymmetricSubsetApply,
    apply_local_block: AxisymmetricLocalBlockApply,
    rhs_subset: AxisymmetricSubsetApply | None = None,
    metadata_builder: AxisymmetricMetadataBuilder | None = None,
) -> ParticleTGroupFactory:
    """Build an axisymmetric group from high-level per-instance callbacks."""

    def factory(
        plan: ParticleTGroupPlan, context: ParticleTPreparationContext
    ) -> PreparedParticleTGroup:
        ids = np.asarray(plan.particle_indices, dtype=np.int64)
        group_particles = context.instances_for(plan)
        metadata = None if metadata_builder is None else metadata_builder(group_particles, context)

        def apply_subset_bound(x_subset: Array) -> Array:
            return np.asarray(
                apply_subset(np.asarray(x_subset, dtype=context.dtype), group_particles, context),
                dtype=context.dtype,
            )

        def apply_local_block_bound(local_particle_index: int, block: Array) -> Array:
            return np.asarray(
                apply_local_block(
                    int(local_particle_index),
                    np.asarray(block, dtype=context.dtype),
                    group_particles,
                    context,
                ),
                dtype=context.dtype,
            )

        def rhs_subset_bound(b_subset: Array) -> Array:
            if rhs_subset is None:
                return apply_subset_bound(b_subset)
            return np.asarray(
                rhs_subset(np.asarray(b_subset, dtype=context.dtype), group_particles, context),
                dtype=context.dtype,
            )

        return AxisymmetricTGroup(
            particle_indices=ids,
            apply_subset_fn=apply_subset_bound,
            rhs_subset_fn=rhs_subset_bound,
            apply_local_block_fn=apply_local_block_bound,
            body_metadata=metadata,
            dtype=context.dtype,
        )

    return factory


def make_axisymmetric_block_group_factory(
    block_provider: DenseBlockProvider,
    *,
    metadata_builder: AxisymmetricMetadataBuilder | None = None,
) -> ParticleTGroupFactory:
    """Build an axisymmetric group from one dense block per unique archetype."""

    def factory(
        plan: ParticleTGroupPlan, context: ParticleTPreparationContext
    ) -> PreparedParticleTGroup:
        ids = np.asarray(plan.particle_indices, dtype=np.int64)
        archetypes = context.archetypes_for(plan)
        metadata = None if metadata_builder is None else metadata_builder(archetypes, context)
        return AxisymmetricTGroup(
            particle_indices=ids,
            operator_indices=np.asarray(plan.operator_indices, dtype=np.int64),
            T_blocks=np.asarray(block_provider(archetypes, context), dtype=context.dtype),
            body_metadata=metadata,
            dtype=context.dtype,
        )

    return factory


__all__ = [
    "Array",
    "AxisymmetricLocalBlockApply",
    "AxisymmetricMetadataBuilder",
    "AxisymmetricSubsetApply",
    "AxisymmetricTGroup",
    "DenseBlockProvider",
    "DenseTGroup",
    "DiagonalTGroup",
    "ParticleTGroupFactories",
    "ParticleTGroupFactory",
    "ParticleTGroupPlan",
    "ParticleTPreparationContext",
    "PreparedParticleTGroup",
    "make_axisymmetric_block_group_factory",
    "make_axisymmetric_group_factory",
    "make_dense_group_factory",
    "plan_particle_t_groups",
]
