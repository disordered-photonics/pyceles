"""Many-sphere matvec assembly (CPU reference, CELES conventions).

We solve the multiple-scattering system in the CELES form:

Let x be the stacked *outgoing/scattered* multipole coefficients for all spheres.
Let W_ij be the CELES translation block mapping sphere j coefficients into the
*incident* coefficients at i.
Let T_i be the diagonal single-sphere T operator (scattered = T * incident).

Self-consistent equation:
  x_i = T_i ( b_i + sum_{j!=i} W_ij x_j )

Linear system:
  (I - T W) x = T b

This file provides a correctness-first O(N^2) implementation:
- apply_W_numpy: y = W x
- apply_A_numpy: y = (I - T W) x
- rhs_Tb_numpy:  r = T b
- assemble_dense_A_numpy: explicit dense A = I - T W (small systems only)

Architecture note
-----------------
The operator is intentionally split into:
- particle-local (single-particle) scattering `T`
- inter-particle coupling `W`

That boundary is the key refactor for future work. Spheres and layered spheres
still use a diagonal fast path today, but the solver no longer assumes that all
particles do. Future axisymmetric and fully general particles can slot in by
implementing new particle-T groups without rewriting `Simulation`, the direct
solver cache, or the block preconditioner.

Performance notes
-----------------
Even for the O(N^2) reference, it is crucial to precompute reusable quantities:
- per-sphere T-diagonal entries (`precompute_T_diagonal`)
- angular translation table `ab5`
- radial translation LUT (`RadialLUT`, always used)

Current implementation is exact (no distance binning approximation) and supports:
- matrix-free translation blocks on the fly (CELES-style default)
- optional exact block caching (`cache_translation_blocks=True`) for small systems

Future speedups:
- block-diagonal preconditioner (CELES)
- GPU backend (CuPy) for hot paths

Practical usage
---------------
Use `prepare_matvec(...)` once, then repeatedly call:
- `prepared.apply_A(x)` inside GMRES
- `prepared.rhs_Tb(b)` for the right-hand side

This avoids accidental recomputation of T-diagonal entries, ab5 tables, LUTs,
and pair translation blocks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol, Sequence, TypeAlias

import numpy as np
import numpy.typing as npt
from tqdm.auto import tqdm

from .geometry_bounds import conservative_set_diameter
from .indexing import n_modes
from .particles import Particle, ParticleTRepresentation, Sphere, particle_t_signature
from .tmatrix import particle_T_diagonal, particle_T_matrix_blocks, sphere_T_diagonal
from .translation import (
    RadialLUT,
    translation_ab5_table,
    translation_block,
)

Array = np.ndarray


@dataclass(frozen=True)
class Geometry:
    """Minimal geometry container for sphere-center coordinates."""

    positions: Array  # (Ns,3)


@dataclass(frozen=True)
class ParticleTGroupPlan:
    """Planned subset of particles sharing one prepared `T` representation.

    Planning happens before expensive preparation. This keeps representation
    policy in one place so future axisymmetric and dense groups can be added
    without scattering `isinstance(...)` branches through the solver stack.
    """

    representation: ParticleTRepresentation
    particle_indices: Array


@dataclass(frozen=True)
class ParticleTPreparationContext:
    """Shared preparation inputs for representation-specific group factories.

    This keeps extension points high-level: future backends can inspect the
    particle subset, truncation, medium, and dtype policy without reaching back
    into `Simulation` or global module state.
    """

    lmax: int
    k: float
    particles: tuple[Particle, ...]
    n_medium: complex
    dtype: np.dtype

    @property
    def n_modes(self) -> int:
        """Number of SVWF modes per particle at the requested truncation."""
        return n_modes(self.lmax)


class PreparedParticleTGroup(Protocol):
    """Prepared subset of particles sharing one `T` representation.

    Every group exposes the same small surface:
    - apply `T` to stacked coefficients on its own particle subset,
    - left-apply one local `T_i` block during dense assembly/preconditioning,
    - optionally expose cheap diagonal views when the representation has them.

    In T-matrix language this is the prepared single-particle scattering
    operator for one particle subset, not a cluster-wide aggregate operator.
    """

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
    """Optional representation-specific group factories for `prepare_matvec`.

    The diagonal path has a built-in default because it is the current core
    solver path. Axisymmetric and dense paths can be injected here before their
    physics-specific preparation is part of the main codebase.
    """

    diagonal: ParticleTGroupFactory | None = None
    axisymmetric: ParticleTGroupFactory | None = None
    dense: ParticleTGroupFactory | None = None

    def for_representation(
        self, representation: ParticleTRepresentation
    ) -> ParticleTGroupFactory | None:
        """Return the factory configured for one representation label."""
        if representation == "diagonal":
            return self.diagonal
        if representation == "axisymmetric":
            return self.axisymmetric
        if representation == "dense":
            return self.dense
        raise ValueError(f"Unsupported particle-T representation {representation!r}.")


@dataclass
class DiagonalTGroup:
    """Diagonal particle-local / single-particle T operator.

    This is the main performance path for spheres and layered spheres. The
    prepared data stay compact and `T` application remains elementwise, which is
    why the general operator refactor does not need to penalize spherical runs.
    """

    particle_indices: Array
    T_M: Array
    T_N: Array
    T_diag: Array
    dtype: np.dtype = np.dtype(np.complex128)

    def apply_subset(self, x_subset: Array) -> Array:
        """Apply `T` to coefficients already sliced to this group."""
        arr = np.asarray(x_subset, dtype=self.dtype)
        return self.T_diag * arr

    def rhs_subset(self, b_subset: Array) -> Array:
        """Apply `T` to incident coefficients on this group subset."""
        return self.apply_subset(b_subset)

    def apply_local_block(self, local_particle_index: int, block: Array) -> Array:
        """Left-multiply one pair block by the local particle T operator."""
        return self.T_diag[int(local_particle_index)][:, None] * np.asarray(block, dtype=self.dtype)

    def mode_diagonal(self) -> Array | None:
        """Return the stored per-particle mode diagonals."""
        return np.asarray(self.T_diag, dtype=self.dtype)

    def degree_diagonals(self) -> tuple[Array, Array] | None:
        """Return the stored per-particle `(T_M, T_N)` diagonals."""
        return np.asarray(self.T_M, dtype=self.dtype), np.asarray(self.T_N, dtype=self.dtype)


@dataclass
class DenseTGroup:
    """Dense particle-local / single-particle T operator.

    This is the generic fallback for imported or non-axisymmetric particles.
    It is intentionally separate from the diagonal path so mixed clusters can
    keep spherical particles on the cheap representation while only the truly
    general subset pays dense-block storage and matvec costs.
    """

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
        """Apply dense per-particle T blocks to one subset of coefficients."""
        arr = np.asarray(x_subset, dtype=self.dtype)
        return np.einsum("gij,gj->gi", self.T_blocks, arr, optimize=True)

    def rhs_subset(self, b_subset: Array) -> Array:
        """Apply dense per-particle T blocks to incident coefficients."""
        return self.apply_subset(b_subset)

    def apply_local_block(self, local_particle_index: int, block: Array) -> Array:
        """Left-multiply one pair block by the local dense particle T block."""
        return self.T_blocks[int(local_particle_index)] @ np.asarray(block, dtype=self.dtype)

    def mode_diagonal(self) -> Array | None:
        """Dense groups do not expose cheap diagonal-mode views."""
        return None

    def degree_diagonals(self) -> tuple[Array, Array] | None:
        """Dense groups do not expose degree-diagonal TE/TM factors."""
        return None


@dataclass
class AxisymmetricTGroup:
    """Prepared axisymmetric particle-local T operator.

    The long-term optimized form is expected to use body-frame `m` blocks and
    optional SVWF rotations. For the first spherical-basis spheroid backend,
    however, it is useful to also support a dense-block fallback here directly.
    That lets non-diagonal axisymmetric particles enter the default solver path
    as soon as they can produce spherical-basis T blocks, without waiting for
    the narrower `m`-block implementation.
    """

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


class ParticleTOperator(Protocol):
    """Prepared particle-local (single-particle) scattering operator `T`."""

    lmax: int
    n_particles: int
    dtype: np.dtype

    def apply(self, x: Array) -> Array: ...

    def rhs(self, b: Array) -> Array: ...

    def apply_particle_block(self, particle_index: int, block: Array) -> Array: ...

    def mode_diagonal(self) -> Array | None: ...

    def degree_diagonals(self) -> tuple[Array, Array] | None: ...


@dataclass
class CompositeParticleTOperator:
    """Composite particle-local / single-particle T operator.

    A single simulation can therefore combine, for example, diagonal spheres
    with future axisymmetric spheroids or dense imported T-matrices while
    preserving one solver and one preconditioner interface.
    """

    lmax: int
    n_particles: int
    groups: Sequence[PreparedParticleTGroup]
    dtype: np.dtype = np.dtype(np.complex128)
    _particle_to_group: np.ndarray = field(init=False, repr=False)
    _particle_to_local: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        Ns = int(self.n_particles)
        group_of = np.full(Ns, -1, dtype=np.int64)
        local_of = np.full(Ns, -1, dtype=np.int64)
        for gidx, group in enumerate(self.groups):
            ids = np.asarray(group.particle_indices, dtype=np.int64).reshape(-1)
            if ids.size == 0:
                raise ValueError("Particle-T operator groups must be non-empty.")
            if np.any(ids < 0) or np.any(ids >= Ns):
                raise ValueError("Particle-T operator group indices out of bounds.")
            if np.any(group_of[ids] != -1):
                raise ValueError("Particle-T operator groups must not overlap.")
            group_of[ids] = gidx
            local_of[ids] = np.arange(ids.size, dtype=np.int64)
        if np.any(group_of < 0):
            raise ValueError("Particle-T operator groups must cover all particles.")
        self._particle_to_group = group_of
        self._particle_to_local = local_of

    @property
    def n_modes(self) -> int:
        """Number of SVWF modes per particle."""
        return n_modes(self.lmax)

    def apply(self, x: Array) -> Array:
        """Apply `T` to stacked coefficients."""
        arr = np.asarray(x, dtype=self.dtype).reshape(self.n_particles, self.n_modes)
        out = np.zeros_like(arr, dtype=self.dtype)
        for group in self.groups:
            ids = np.asarray(group.particle_indices, dtype=np.int64)
            out[ids] = group.apply_subset(arr[ids])
        return out.reshape(self.n_particles * self.n_modes)

    def rhs(self, b: Array) -> Array:
        """Apply `T` to stacked incident coefficients."""
        arr = np.asarray(b, dtype=self.dtype).reshape(self.n_particles, self.n_modes)
        out = np.zeros_like(arr, dtype=self.dtype)
        for group in self.groups:
            ids = np.asarray(group.particle_indices, dtype=np.int64)
            out[ids] = group.rhs_subset(arr[ids])
        return out.reshape(self.n_particles * self.n_modes)

    def apply_particle_block(self, particle_index: int, block: Array) -> Array:
        """Return `T_i @ block` for one destination particle."""
        i = int(particle_index)
        gidx = int(self._particle_to_group[i])
        local = int(self._particle_to_local[i])
        return self.groups[gidx].apply_local_block(local, block)

    def mode_diagonal(self) -> Array | None:
        """Return per-particle mode diagonals if every group is diagonal."""
        out = np.empty((self.n_particles, self.n_modes), dtype=self.dtype)
        for group in self.groups:
            ids = np.asarray(group.particle_indices, dtype=np.int64)
            diag = group.mode_diagonal()
            if diag is None:
                return None
            out[ids] = np.asarray(diag, dtype=self.dtype)
        return out

    def degree_diagonals(self) -> tuple[Array, Array] | None:
        """Return per-particle `(T_M, T_N)` diagonals if every group is diagonal."""
        out_M = np.empty((self.n_particles, self.lmax + 1), dtype=self.dtype)
        out_N = np.empty((self.n_particles, self.lmax + 1), dtype=self.dtype)
        for group in self.groups:
            ids = np.asarray(group.particle_indices, dtype=np.int64)
            diags = group.degree_diagonals()
            if diags is None:
                return None
            out_M[ids] = np.asarray(diags[0], dtype=self.dtype)
            out_N[ids] = np.asarray(diags[1], dtype=self.dtype)
        return out_M, out_N


@dataclass
class PairwiseCouplingOperator:
    """Prepared pairwise free-space coupling operator `W`."""

    lmax: int
    k: float
    positions: Array
    ab5: Array
    radial_lut: RadialLUT
    dtype: np.dtype = np.dtype(np.complex128)
    cache_translation_blocks: bool = False
    _W_cache: dict[tuple[int, int], Array] = field(default_factory=dict)

    def apply_W(self, x: Array) -> Array:
        """Apply inter-particle translation operator `W` to coefficient vector."""
        return apply_W_numpy(
            self.lmax,
            self.k,
            self.positions,
            x,
            self.ab5,
            dtype=self.dtype,
            radial_lut=self.radial_lut,
            block_cache=self._W_cache if self.cache_translation_blocks else None,
        )

    def apply(self, x: Array) -> Array:
        """Protocol-friendly alias for `apply_W`."""
        return self.apply_W(x)

    def populate_translation_cache(self, *, show_progress: bool = False) -> None:
        """Precompute and cache all pair translation blocks W_ij (i!=j)."""
        if not self.cache_translation_blocks:
            return

        Ns = self.positions.shape[0]
        pair_iter = [(i, j) for i in range(Ns) for j in range(Ns) if i != j]
        if show_progress:
            pair_iter = tqdm(pair_iter, desc="Precompute W_ij")

        for i, j in pair_iter:
            key = (i, j)
            if key in self._W_cache:
                continue
            rvec = self.positions[i] - self.positions[j]
            self._W_cache[key] = translation_block(
                self.lmax,
                self.k,
                rvec,
                ab5=self.ab5,
                radial_lut=self.radial_lut,
            )


@dataclass
class PreparedOperator:
    """Prepared linear operator `A = I - T W` with explicit `T`/`W` boundaries."""

    lmax: int
    k: float
    positions: Array
    particle_t: ParticleTOperator
    coupling: PairwiseCouplingOperator
    dtype: np.dtype = np.dtype(np.complex128)

    def apply_W(self, x: Array) -> Array:
        """Apply inter-particle translation operator `W`."""
        return self.coupling.apply(x)

    def apply_A(self, x: Array) -> Array:
        """Apply the full linear operator `A = I - T W`."""
        return np.asarray(x, dtype=self.dtype) - self.particle_t.apply(self.apply_W(x))

    def rhs(self, b: Array) -> Array:
        """Apply right-hand side mapping `b -> T b`."""
        return self.particle_t.rhs(b)

    def rhs_Tb(self, b: Array) -> Array:
        """Compatibility wrapper for the CELES-form RHS map."""
        return self.rhs(b)

    def apply_particle_block(self, particle_index: int, block: Array) -> Array:
        """Return `T_i @ block` for one destination-particle block row."""
        return self.particle_t.apply_particle_block(particle_index, block)

    def populate_translation_cache(self, *, show_progress: bool = False) -> None:
        """Precompute and cache all pair translation blocks W_ij."""
        self.coupling.populate_translation_cache(show_progress=show_progress)

    @property
    def ab5(self) -> Array:
        return self.coupling.ab5

    @property
    def radial_lut(self) -> RadialLUT:
        return self.coupling.radial_lut

    @property
    def cache_translation_blocks(self) -> bool:
        return self.coupling.cache_translation_blocks

    @property
    def _W_cache(self) -> dict[tuple[int, int], Array]:
        return self.coupling._W_cache

    @property
    def T_diag(self) -> Array:
        diag = self.particle_t.mode_diagonal()
        if diag is None:
            raise NotImplementedError("Prepared operator does not expose diagonal per-mode T data.")
        return diag

    @property
    def T_M(self) -> Array:
        diags = self.particle_t.degree_diagonals()
        if diags is None:
            raise NotImplementedError("Prepared operator does not expose diagonal T_M/T_N data.")
        return diags[0]

    @property
    def T_N(self) -> Array:
        diags = self.particle_t.degree_diagonals()
        if diags is None:
            raise NotImplementedError("Prepared operator does not expose diagonal T_M/T_N data.")
        return diags[1]


def plan_particle_t_groups(particles: Sequence[Particle]) -> tuple[ParticleTGroupPlan, ...]:
    """Plan particle-local `T` groups before preparing concrete operators.

    Grouping by representation keeps the current diagonal sphere path intact and
    provides a stable insertion point for future axisymmetric and dense
    implementations. The order is deterministic and follows first appearance in
    the particle list so diagnostics remain easy to read.
    """

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
    """Build a dense-group factory from a particle-subset T-block provider.

    This is the intended high-level hook for future imported/database-backed
    T-matrix workflows: the caller only needs to provide one `(Ng, Nm, Nm)`
    block stack for the selected particle subset.
    """

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
    """Build an axisymmetric-group factory from high-level apply callbacks.

    The eventual spheroid backend can use this by binding body-frame data and
    SVWF rotations into the returned callbacks, while `prepare_matvec()` remains
    unchanged.
    """

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
    """Build an axisymmetric group from full spherical-basis T blocks.

    This is the intended bridge for the first spheroid backend in `pyceles`:
    generate canonical spherical-basis T blocks from physics code, keep the
    solver path unchanged, and optimize storage/application later by replacing
    the dense fallback inside `AxisymmetricTGroup`.
    """

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


def _prepare_diagonal_group(
    *,
    plan: ParticleTGroupPlan,
    lmax: int,
    k: float,
    particles: Sequence[Particle],
    n_medium: complex,
    dtype: np.dtype,
) -> DiagonalTGroup:
    """Prepare one diagonal particle-T group for diagonal-capable particles.

    This helper isolates the current sphere/layered-sphere preparation logic so
    later group types can be added alongside it instead of replacing it.
    """

    ids = np.asarray(plan.particle_indices, dtype=np.int64)
    group_particles = [particles[int(i)] for i in ids]
    T_M, T_N = precompute_T_diagonal(
        lmax=int(lmax),
        k=float(k),
        particles=group_particles,
        n_medium=n_medium,
        dtype=dtype,
    )
    return DiagonalTGroup(
        particle_indices=ids,
        T_M=T_M,
        T_N=T_N,
        T_diag=_build_T_mode_diagonal(int(lmax), T_M, T_N),
        dtype=dtype,
    )


def _default_group_factory(
    *,
    plan: ParticleTGroupPlan,
    context: ParticleTPreparationContext,
) -> PreparedParticleTGroup:
    """Prepare one group using built-in support when available."""

    if plan.representation == "diagonal":
        return _prepare_diagonal_group(
            plan=plan,
            lmax=context.lmax,
            k=context.k,
            particles=context.particles,
            n_medium=context.n_medium,
            dtype=context.dtype,
        )

    if plan.representation == "dense":
        ids = np.asarray(plan.particle_indices, dtype=np.int64)
        group_particles = [context.particles[int(i)] for i in ids]
        try:
            blocks = particle_T_matrix_blocks(
                lmax=context.lmax,
                k_medium=context.k,
                particles=group_particles,
                n_medium=context.n_medium,
            )
        except NotImplementedError as exc:
            raise NotImplementedError(str(exc)) from exc
        except TypeError as exc:
            raise NotImplementedError(
                "dense particle-T preparation requires canonical spherical-basis "
                "T blocks for every particle in the selected group."
            ) from exc
        return DenseTGroup(
            particle_indices=ids,
            T_blocks=blocks.astype(context.dtype, copy=False),
            dtype=context.dtype,
        )

    if plan.representation == "axisymmetric":
        ids = np.asarray(plan.particle_indices, dtype=np.int64)
        axis_particles = tuple(context.particles[int(i)] for i in ids)
        try:
            blocks = particle_T_matrix_blocks(
                lmax=context.lmax,
                k_medium=context.k,
                particles=list(axis_particles),
                n_medium=context.n_medium,
            )
        except NotImplementedError as exc:
            raise NotImplementedError(str(exc)) from exc
        except TypeError as exc:
            raise NotImplementedError(
                "axisymmetric particle-T preparation requires canonical spherical-basis "
                "T blocks for every particle in the selected group."
            ) from exc
        return AxisymmetricTGroup(
            particle_indices=ids,
            T_blocks=blocks.astype(context.dtype, copy=False),
            body_metadata={"storage": "spherical_basis_dense_blocks"},
            dtype=context.dtype,
        )

    particle_labels = ", ".join(
        f"{int(i)}:{type(context.particles[int(i)]).__name__}"
        for i in np.asarray(plan.particle_indices)
    )
    raise NotImplementedError(
        "Single-body operator planning selected the "
        f"'{plan.representation}' representation for particle(s) {particle_labels}, "
        "but no preparation factory was provided for that representation."
    )


def _prepare_particle_t_operator(
    *,
    lmax: int,
    k: float,
    particles: Sequence[Particle],
    n_medium: complex,
    dtype: np.dtype,
    group_factories: ParticleTGroupFactories | None = None,
) -> CompositeParticleTOperator:
    """Prepare the particle-local operator using planned representation groups.

    Only the diagonal path is active today. Axisymmetric and dense groups are
    deliberately rejected here, at the representation boundary, so unsupported
    particles fail early with an architectural message instead of deep inside a
    diagonal-only kernel.
    """

    part = tuple(particles)
    context = ParticleTPreparationContext(
        lmax=int(lmax),
        k=float(k),
        particles=part,
        n_medium=complex(n_medium),
        dtype=dtype,
    )
    plans = plan_particle_t_groups(part)
    factories = ParticleTGroupFactories() if group_factories is None else group_factories
    groups: list[PreparedParticleTGroup] = []
    for plan in plans:
        factory = factories.for_representation(plan.representation)
        groups.append(
            _default_group_factory(plan=plan, context=context)
            if factory is None
            else factory(plan, context)
        )

    return CompositeParticleTOperator(
        lmax=int(lmax),
        n_particles=len(part),
        groups=tuple(groups),
        dtype=dtype,
    )


def _build_T_mode_diagonal(lmax: int, T_M: Array, T_N: Array) -> Array:
    """Expand per-(sphere,l) diagonal entries to per-(sphere,mode) factors."""

    lmax = int(lmax)
    T_M = np.asarray(T_M)
    T_N = np.asarray(T_N)

    if T_M.shape != T_N.shape:
        raise ValueError(
            f"T_M and T_N must have identical shapes. Got {T_M.shape} and {T_N.shape}."
        )

    Ns = T_M.shape[0]
    Nm = n_modes(lmax)
    Nscl = lmax * (lmax + 2)
    out_dtype = np.result_type(T_M.dtype, T_N.dtype, np.complex64)
    T_diag = np.zeros((Ns, Nm), dtype=out_dtype)

    for l in range(1, lmax + 1):
        start = (l - 1) * (l + 1)
        end = start + (2 * l + 1)
        T_diag[:, start:end] = T_M[:, l : l + 1]
        T_diag[:, Nscl + start : Nscl + end] = T_N[:, l : l + 1]

    return T_diag


def _infer_rmax(positions: Array) -> float:
    """Conservative upper bound on center-to-center separation for LUT sizing.

    Uses the axis-aligned bounding-box diagonal. This is O(N) in both time and
    memory and safely upper-bounds the true maximum pair distance.
    """

    return conservative_set_diameter(np.asarray(positions, dtype=float))


def prepare_matvec(
    *,
    lmax: int,
    k: float,
    particles: Sequence[Particle],
    n_medium: complex = 1.0 + 0j,
    radial_lut_dr: float,
    cache_translation_blocks: bool = False,
    operator_dtype: npt.DTypeLike = np.complex128,
    particle_t_group_factories: ParticleTGroupFactories | None = None,
) -> PreparedOperator:
    """Prepare reusable `A = I - T W` data from explicit particle descriptors.

    The returned object is intentionally backend-neutral at the top level:
    `Simulation` only sees one prepared operator, while the internal particle-T
    representation can stay diagonal for spheres or later switch to
    axisymmetric/dense groups on a subset of particles.

    Advanced callers can inject `particle_t_group_factories` to prepare dense
    or axisymmetric groups before those backends are part of the default path.
    """
    part = list(particles)
    positions = np.asarray(
        [np.asarray(p.position, dtype=float) for p in part], dtype=float
    ).reshape(-1, 3)
    op_dtype = np.dtype(operator_dtype)
    k_f = float(k)
    ab5 = translation_ab5_table(int(lmax), dtype=op_dtype)

    dr_user = float(radial_lut_dr)
    if dr_user < 0.0:
        raise ValueError(f"radial_lut_dr must be >= 0, got {dr_user}.")
    k_abs = float(abs(k_f))
    if k_abs <= 0.0:
        raise ValueError(f"`k` must be non-zero for radial LUT setup. Got {k_f!r}.")
    # `radial_lut_dr` semantics:
    # - > 0: explicit absolute spacing in geometry units
    # - = 0: auto spacing from fixed `delta(kr)=1e-2`
    dr = (1.0e-2 / k_abs) if dr_user == 0.0 else dr_user
    lut = RadialLUT(lmax=int(lmax), k=k_f, r_max=_infer_rmax(positions), dr=dr, dtype=op_dtype)

    particle_t = _prepare_particle_t_operator(
        lmax=int(lmax),
        k=k_f,
        particles=part,
        n_medium=n_medium,
        dtype=op_dtype,
        group_factories=particle_t_group_factories,
    )
    coupling = PairwiseCouplingOperator(
        lmax=int(lmax),
        k=k_f,
        positions=positions,
        ab5=ab5,
        radial_lut=lut,
        dtype=op_dtype,
        cache_translation_blocks=bool(cache_translation_blocks),
    )

    return PreparedOperator(
        lmax=int(lmax),
        k=k_f,
        positions=positions,
        particle_t=particle_t,
        coupling=coupling,
        dtype=op_dtype,
    )


def estimate_translation_cache_bytes(
    N: int, lmax: int, *, dtype: npt.DTypeLike = np.complex128
) -> int:
    """Estimate memory of storing all pair blocks W_ij (i!=j).

    This excludes Python-dictionary/object overhead from cache bookkeeping.
    """

    N = int(N)
    lmax = int(lmax)
    if N < 0:
        raise ValueError("N must be non-negative")
    Nm = n_modes(lmax)
    pairs = N * (N - 1)
    block_entries = Nm * Nm
    return pairs * block_entries * np.dtype(dtype).itemsize


def make_prepared_A_and_rhs(
    prepared: PreparedOperator, b: Array
) -> tuple[Callable[[Array], Array], Array]:
    """Return `(A_mv, rhs)` from a prepared system and incident coefficients."""

    rhs = prepared.rhs_Tb(b)

    def A_mv(x: Array) -> Array:
        """Matrix-free apply of `A = I - T W` using precomputed prepared data."""
        return prepared.apply_A(x)

    return A_mv, rhs


def assemble_dense_A_numpy(
    prepared: PreparedOperator,
    *,
    show_progress: bool = False,
    use_cache: bool = False,
    store_blocks: bool = False,
) -> Array:
    """Assemble dense A = I - T W from a prepared system.

    This is intended for small systems where a direct dense solve is feasible.
    Assembly is blockwise over sphere pairs:
      A_ii = I
      A_ij = -diag(T_i) @ W_ij, i != j
    """

    Ns = prepared.positions.shape[0]
    Nm = n_modes(prepared.lmax)
    n = Ns * Nm
    A = np.zeros((n, n), dtype=prepared.dtype)
    A[np.arange(n), np.arange(n)] = 1.0 + 0.0j

    pair_iter = ((i, j) for i in range(Ns) for j in range(Ns) if i != j)
    if show_progress:
        pair_iter = tqdm(pair_iter, total=Ns * (Ns - 1), desc="Assemble A (blockwise)")

    cache = prepared._W_cache if use_cache else None
    for i, j in pair_iter:
        key = (i, j)
        Wij = cache.get(key) if cache is not None else None
        if Wij is None:
            rvec = prepared.positions[i] - prepared.positions[j]
            Wij = translation_block(
                prepared.lmax,
                prepared.k,
                rvec,
                ab5=prepared.ab5,
                radial_lut=prepared.radial_lut,
            )
            if store_blocks:
                prepared._W_cache[key] = Wij

        blk = -prepared.apply_particle_block(i, Wij)
        rs = slice(i * Nm, (i + 1) * Nm)
        cs = slice(j * Nm, (j + 1) * Nm)
        A[rs, cs] = blk

    return A


def precompute_T_diagonal(
    *,
    lmax: int,
    k: float,
    particles: Sequence[Particle],
    n_medium: complex = 1.0 + 0j,
    dtype: npt.DTypeLike = np.complex128,
) -> tuple[Array, Array]:
    """Precompute per-particle diagonal T entries for the current geometry."""
    lmax = int(lmax)
    out_dtype = np.dtype(dtype)
    Ns = len(particles)
    T_M = np.zeros((Ns, lmax + 1), dtype=out_dtype)
    T_N = np.zeros((Ns, lmax + 1), dtype=out_dtype)

    # Reuse one particle-local diagonal kernel across repeated particles that
    # differ only by position. This keeps large repeated arrays cheap for both
    # homogeneous and layered spheres, and later extends naturally to other
    # diagonal-capable particle families.
    MAX_TMEMO_ENTRIES = 100_000
    diagonal_memo: dict[tuple[object, ...], tuple[Array, Array]] = {}
    for i, p in enumerate(particles):
        key = (
            particle_t_signature(p),
            complex(k),
            complex(n_medium),
            int(lmax),
        )
        cached = diagonal_memo.get(key)
        if cached is None:
            if isinstance(p, Sphere):
                Td = sphere_T_diagonal(lmax, k, p.radius, p.refractive_index, n_medium)
            else:
                Td = particle_T_diagonal(lmax=lmax, k_medium=k, particle=p, n_medium=n_medium)
            cached = (
                np.asarray(Td[1], dtype=out_dtype).copy(),
                np.asarray(Td[2], dtype=out_dtype).copy(),
            )
            if len(diagonal_memo) < MAX_TMEMO_ENTRIES:
                diagonal_memo[key] = cached

        T_M[i, :] = cached[0]
        T_N[i, :] = cached[1]

    return T_M, T_N


def apply_W_numpy(
    lmax: int,
    k: float,
    positions: Array,
    x: Array,
    ab5: Array,
    *,
    dtype: npt.DTypeLike = np.complex128,
    radial_lut: Optional[RadialLUT],
    block_cache: Optional[dict[tuple[int, int], Array]] = None,
) -> Array:
    """Compute y = W x (NumPy), excluding self-interaction.

    If `radial_lut` is `None`, the translation radial functions are evaluated
    exactly (no LUT approximation).

    This routine is the matrix-free hot path: W_ij blocks are formed per pair
    and applied directly, rather than materializing a global W matrix.
    """

    out_dtype = np.dtype(dtype)
    Ns = positions.shape[0]
    Nm = n_modes(lmax)
    x = np.asarray(x, dtype=out_dtype).reshape(Ns, Nm)
    y = np.zeros_like(x, dtype=out_dtype)

    for i in range(Ns):
        for j in range(Ns):
            if i == j:
                continue
            key = (i, j)
            Wij = block_cache.get(key) if block_cache is not None else None
            if Wij is None:
                rvec = positions[i] - positions[j]
                Wij = translation_block(lmax, k, rvec, ab5=ab5, radial_lut=radial_lut)
                if block_cache is not None:
                    block_cache[key] = Wij
            y[i] += Wij @ x[j]

    return y.reshape(Ns * Nm)


def apply_A_numpy(
    lmax: int,
    k: float,
    positions: Array,
    x: Array,
    *,
    T_M: Array,
    T_N: Array,
    T_diag: Optional[Array] = None,
    ab5: Optional[Array] = None,
    dtype: npt.DTypeLike = np.complex128,
    radial_lut: Optional[RadialLUT],
    block_cache: Optional[dict[tuple[int, int], Array]] = None,
) -> Array:
    """Compute y = (I - T W) x with precomputed diagonal T entries.

    If `radial_lut` is `None`, the translation radial functions are evaluated
    exactly (no LUT approximation).
    """

    out_dtype = np.dtype(dtype)
    Ns = positions.shape[0]
    Nm = n_modes(lmax)
    if ab5 is None:
        ab5 = translation_ab5_table(lmax)

    Wx = apply_W_numpy(
        lmax,
        k,
        positions,
        x,
        ab5,
        dtype=out_dtype,
        radial_lut=radial_lut,
        block_cache=block_cache,
    ).reshape(Ns, Nm)

    y = np.asarray(x, dtype=out_dtype).reshape(Ns, Nm).copy()

    if T_diag is None:
        T_diag = _build_T_mode_diagonal(lmax, T_M, T_N)
    else:
        T_diag = np.asarray(T_diag, dtype=out_dtype)

    y -= T_diag * Wx

    return y.reshape(Ns * Nm)


def rhs_Tb_numpy(
    lmax: int,
    b: Array,
    *,
    T_M: Array,
    T_N: Array,
    T_diag: Optional[Array] = None,
    dtype: npt.DTypeLike = np.complex128,
) -> Array:
    """Compute r = T b (NumPy) where b is stacked incident coefficients per sphere."""

    out_dtype = np.dtype(dtype)
    lmax = int(lmax)
    Nm = n_modes(lmax)

    b = np.asarray(b, dtype=out_dtype)
    Ns = b.size // Nm
    b = b.reshape(Ns, Nm)

    if T_diag is None:
        T_diag = _build_T_mode_diagonal(lmax, T_M, T_N)
    else:
        T_diag = np.asarray(T_diag, dtype=out_dtype)

    r = T_diag * b

    return r.reshape(Ns * Nm)
