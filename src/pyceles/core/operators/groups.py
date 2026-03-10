from __future__ import annotations

"""Prepared single-particle group types and group-factory planning."""

from dataclasses import dataclass
from typing import Callable, Protocol, Sequence, TypeAlias

import numpy as np

from pyceles.core.indexing import n_modes
from pyceles.core.particles import Particle, ParticleTRepresentation

Array = np.ndarray


@dataclass(frozen=True)
class ParticleTGroupPlan:
    """Planned subset of particles sharing one prepared `T` representation."""

    representation: ParticleTRepresentation
    particle_indices: Array


@dataclass(frozen=True)
class ParticleTPreparationContext:
    """Shared preparation inputs for representation-specific group factories."""

    lmax: int
    k: float
    particles: tuple[Particle, ...]
    n_medium: complex
    dtype: np.dtype

    @property
    def n_modes(self) -> int:
        return n_modes(self.lmax)


class PreparedParticleTGroup(Protocol):
    """Prepared subset of particles sharing one `T` representation."""

    particle_indices: Array
    dtype: np.dtype

    def apply_subset(self, x_subset: Array) -> Array: ...

    def rhs_subset(self, b_subset: Array) -> Array: ...

    def apply_local_block(self, local_particle_index: int, block: Array) -> Array: ...

    def mode_diagonal(self) -> Array | None: ...

    def degree_diagonals(self) -> tuple[Array, Array] | None: ...


ParticleTGroupFactory: TypeAlias = Callable[
    [ParticleTGroupPlan, ParticleTPreparationContext], PreparedParticleTGroup
]
DenseBlockProvider: TypeAlias = Callable[[Sequence[Particle], ParticleTPreparationContext], Array]
AxisymmetricSubsetApply: TypeAlias = Callable[
    [Array, Sequence[Particle], ParticleTPreparationContext], Array
]
AxisymmetricLocalBlockApply: TypeAlias = Callable[
    [int, Array, Sequence[Particle], ParticleTPreparationContext], Array
]
AxisymmetricMetadataBuilder: TypeAlias = Callable[
    [Sequence[Particle], ParticleTPreparationContext], object | None
]


@dataclass(frozen=True)
class ParticleTGroupFactories:
    """Optional representation-specific group factories for `prepare_matvec`."""

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


@dataclass
class DiagonalTGroup:
    particle_indices: Array
    T_M: Array
    T_N: Array
    T_diag: Array
    dtype: np.dtype = np.dtype(np.complex128)

    def apply_subset(self, x_subset: Array) -> Array:
        arr = np.asarray(x_subset, dtype=self.dtype)
        return self.T_diag * arr

    def rhs_subset(self, b_subset: Array) -> Array:
        return self.apply_subset(b_subset)

    def apply_local_block(self, local_particle_index: int, block: Array) -> Array:
        return self.T_diag[int(local_particle_index)][:, None] * np.asarray(block, dtype=self.dtype)

    def mode_diagonal(self) -> Array | None:
        return np.asarray(self.T_diag, dtype=self.dtype)

    def degree_diagonals(self) -> tuple[Array, Array] | None:
        return np.asarray(self.T_M, dtype=self.dtype), np.asarray(self.T_N, dtype=self.dtype)


@dataclass
class DenseTGroup:
    particle_indices: Array
    T_blocks: Array
    dtype: np.dtype = np.dtype(np.complex128)

    def __post_init__(self) -> None:
        blocks = np.asarray(self.T_blocks, dtype=self.dtype)
        if blocks.ndim != 3 or blocks.shape[1] != blocks.shape[2]:
            raise ValueError(f"`T_blocks` must have shape (Ng, Nm, Nm). Got {blocks.shape}.")
        ids = np.asarray(self.particle_indices, dtype=np.int64).reshape(-1)
        if blocks.shape[0] != ids.size:
            raise ValueError(
                "Dense particle-T group must have one T block per particle. "
                f"Got {blocks.shape[0]} blocks for {ids.size} particles."
            )
        self.particle_indices = ids
        self.T_blocks = blocks

    def apply_subset(self, x_subset: Array) -> Array:
        arr = np.asarray(x_subset, dtype=self.dtype)
        return np.einsum("gij,gj->gi", self.T_blocks, arr, optimize=True)

    def rhs_subset(self, b_subset: Array) -> Array:
        return self.apply_subset(b_subset)

    def apply_local_block(self, local_particle_index: int, block: Array) -> Array:
        return self.T_blocks[int(local_particle_index)] @ np.asarray(block, dtype=self.dtype)

    def mode_diagonal(self) -> Array | None:
        return None

    def degree_diagonals(self) -> tuple[Array, Array] | None:
        return None


@dataclass
class AxisymmetricTGroup:
    particle_indices: Array
    T_blocks: Array | None = None
    apply_subset_fn: Callable[[Array], Array] | None = None
    rhs_subset_fn: Callable[[Array], Array] | None = None
    apply_local_block_fn: Callable[[int, Array], Array] | None = None
    body_metadata: object | None = None
    dtype: np.dtype = np.dtype(np.complex128)

    def __post_init__(self) -> None:
        self.particle_indices = np.asarray(self.particle_indices, dtype=np.int64).reshape(-1)
        if self.T_blocks is not None:
            blocks = np.asarray(self.T_blocks, dtype=self.dtype)
            ids = np.asarray(self.particle_indices, dtype=np.int64).reshape(-1)
            if blocks.ndim != 3 or blocks.shape[1] != blocks.shape[2]:
                raise ValueError(
                    "Axisymmetric dense fallback must provide blocks of shape (Ng, Nm, Nm)."
                )
            if blocks.shape[0] != ids.size:
                raise ValueError(
                    "Axisymmetric dense fallback must provide one T block per particle. "
                    f"Got {blocks.shape[0]} blocks for {ids.size} particles."
                )
            self.T_blocks = blocks

    def apply_subset(self, x_subset: Array) -> Array:
        if self.T_blocks is not None:
            arr = np.asarray(x_subset, dtype=self.dtype)
            return np.einsum("gij,gj->gi", self.T_blocks, arr, optimize=True)
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
            return self.T_blocks[int(local_particle_index)] @ np.asarray(block, dtype=self.dtype)
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
    """Plan particle-local `T` groups before preparing concrete operators."""
    if len(particles) == 0:
        return ()

    grouped: dict[ParticleTRepresentation, list[int]] = {}
    order: list[ParticleTRepresentation] = []
    for idx, particle in enumerate(particles):
        rep = particle.t_operator_representation
        if rep not in grouped:
            grouped[rep] = []
            order.append(rep)
        grouped[rep].append(int(idx))

    return tuple(
        ParticleTGroupPlan(
            representation=rep,
            particle_indices=np.asarray(grouped[rep], dtype=np.int64),
        )
        for rep in order
    )


def make_dense_group_factory(block_provider: DenseBlockProvider) -> ParticleTGroupFactory:
    """Build a dense-group factory from a particle-subset T-block provider."""

    def factory(
        plan: ParticleTGroupPlan, context: ParticleTPreparationContext
    ) -> PreparedParticleTGroup:
        ids = np.asarray(plan.particle_indices, dtype=np.int64)
        group_particles = tuple(context.particles[int(i)] for i in ids)
        return DenseTGroup(
            particle_indices=ids,
            T_blocks=np.asarray(block_provider(group_particles, context), dtype=context.dtype),
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
    """Build an axisymmetric-group factory from high-level apply callbacks."""

    def factory(
        plan: ParticleTGroupPlan, context: ParticleTPreparationContext
    ) -> PreparedParticleTGroup:
        ids = np.asarray(plan.particle_indices, dtype=np.int64)
        group_particles = tuple(context.particles[int(i)] for i in ids)
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
    """Build an axisymmetric group from full spherical-basis T blocks."""

    def factory(
        plan: ParticleTGroupPlan, context: ParticleTPreparationContext
    ) -> PreparedParticleTGroup:
        ids = np.asarray(plan.particle_indices, dtype=np.int64)
        group_particles = tuple(context.particles[int(i)] for i in ids)
        metadata = None if metadata_builder is None else metadata_builder(group_particles, context)
        return AxisymmetricTGroup(
            particle_indices=ids,
            T_blocks=np.asarray(block_provider(group_particles, context), dtype=context.dtype),
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
