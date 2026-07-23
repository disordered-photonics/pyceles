from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, fields
from typing import ClassVar, Literal, overload

import numpy as np

ParticleTRepresentation = Literal["diagonal", "axisymmetric", "dense"]


@dataclass(frozen=True, slots=True)
class Particle:
    """Base particle descriptor.

    Notes
    -----
    `Sphere`, `PECSphere`, `LayeredSphere`, and `Spheroid` are supported by
    active solver kernels, though not every downstream postprocessing path is
    equally mature for every particle family.
    """

    position: tuple[float, float, float]

    def pos_array(self, dtype=float) -> np.ndarray:
        """Return particle center as numeric array for kernel consumption."""
        return np.asarray(self.position, dtype=dtype)

    @property
    def kind(self) -> str:
        """Lower-case particle type label used by dispatch utilities."""
        return type(self).__name__.lower()

    def circumscribing_radius(self) -> float:
        """Radius of a sphere enclosing the particle (for broad-phase geometry)."""
        raise NotImplementedError(f"{type(self).__name__} must implement circumscribing_radius().")

    @property
    def t_operator_representation(self) -> ParticleTRepresentation:
        """Preferred prepared-operator representation for this particle family.

        The representation only describes how the particle-local `T` operator
        should be stored/applied. It does not decide the coupling backend.
        Keeping this declaration on the particle descriptor lets operator
        preparation choose specialized groups in one place instead of growing
        solver-wide type checks.
        """
        return "dense"


@dataclass(frozen=True, slots=True)
class Sphere(Particle):
    """Homogeneous sphere supported by current scattering kernels."""

    radius: float
    refractive_index: complex = 1.5 + 0j

    def circumscribing_radius(self) -> float:
        """For spheres the circumscribing radius is the physical radius."""
        return float(self.radius)

    @property
    def t_operator_representation(self) -> ParticleTRepresentation:
        """Spheres use the diagonal Mie fast path."""
        return "diagonal"


@dataclass(frozen=True, slots=True)
class PECSphere(Particle):
    """Perfect-electric-conductor sphere.

    PEC spheres use the closed-form conducting-sphere Mie limit instead of a
    finite refractive index. They are lossless scatterers and their physical
    interior field is zero.
    """

    radius: float

    def __post_init__(self) -> None:
        if float(self.radius) <= 0.0:
            raise ValueError("radius must be positive.")

    def circumscribing_radius(self) -> float:
        """For PEC spheres the circumscribing radius is the physical radius."""
        return float(self.radius)

    @property
    def t_operator_representation(self) -> ParticleTRepresentation:
        """PEC spheres use the diagonal Mie fast path."""
        return "diagonal"


@dataclass(frozen=True, slots=True)
class LayeredSphere(Particle):
    """Concentric multilayer sphere for exact multilayer Mie kernels."""

    layer_radii: tuple[float, ...]
    layer_refractive_indices: tuple[complex, ...]

    def __post_init__(self) -> None:
        if len(self.layer_radii) == 0:
            raise ValueError("layer_radii must be non-empty.")
        if len(self.layer_radii) != len(self.layer_refractive_indices):
            raise ValueError("layer_radii and layer_refractive_indices must have the same length.")
        if any(float(r) <= 0.0 for r in self.layer_radii):
            raise ValueError("All layer radii must be positive.")
        if any(
            self.layer_radii[i] <= self.layer_radii[i - 1] for i in range(1, len(self.layer_radii))
        ):
            raise ValueError("layer_radii must be strictly increasing (inner to outer).")

    def circumscribing_radius(self) -> float:
        """Outermost shell radius, useful for overlap checks and bounding boxes."""
        return float(self.layer_radii[-1])

    @property
    def t_operator_representation(self) -> ParticleTRepresentation:
        """Layered spheres remain diagonal in the SVWF basis."""
        return "diagonal"


@dataclass(frozen=True, slots=True)
class Spheroid(Particle):
    """Homogeneous axisymmetric particle with spherical-basis T-block support."""

    equatorial_radius: float
    polar_radius: float
    refractive_index: complex = 1.5 + 0j
    euler_angles: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if float(self.equatorial_radius) <= 0.0:
            raise ValueError("equatorial_radius must be positive.")
        if float(self.polar_radius) <= 0.0:
            raise ValueError("polar_radius must be positive.")

    def circumscribing_radius(self) -> float:
        """Largest semi-axis of the spheroid."""
        return float(max(float(self.equatorial_radius), float(self.polar_radius)))

    @property
    def t_operator_representation(self) -> ParticleTRepresentation:
        """Axisymmetric particles admit a narrower-than-dense T representation."""
        return "axisymmetric"


def _rotation_matrix_zyz_lab_to_body(euler_angles: tuple[float, float, float]) -> np.ndarray:
    """Return the lab-to-body rotation matrix for particle Euler angles.

    The convention matches the particle-orientation usage elsewhere in
    `pyceles`: a `z-y'-z''` Euler triplet with the body frame aligned to the
    particle at zero angles.
    """

    alpha = float(euler_angles[0])
    beta = float(euler_angles[1])
    gamma = float(euler_angles[2])
    rot_1 = np.array(
        [
            [np.cos(alpha), np.sin(alpha), 0.0],
            [-np.sin(alpha), np.cos(alpha), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    rot_2 = np.array(
        [
            [np.cos(beta), 0.0, -np.sin(beta)],
            [0.0, 1.0, 0.0],
            [np.sin(beta), 0.0, np.cos(beta)],
        ],
        dtype=float,
    )
    rot_3 = np.array(
        [
            [np.cos(gamma), np.sin(gamma), 0.0],
            [-np.sin(gamma), np.cos(gamma), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return np.asarray(rot_3 @ rot_2 @ rot_1, dtype=float)


def particle_contains_points(
    particle: Particle,
    points: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """Return a boolean mask for points inside the physical particle boundary."""

    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    center = np.asarray(particle.position, dtype=float).reshape(3)
    rel = pts - center[None, :]

    if isinstance(particle, (Sphere, PECSphere)):
        radius = float(particle.radius)
        return np.asarray(np.sum(rel * rel, axis=1) < (radius**2), dtype=bool)

    if isinstance(particle, LayeredSphere):
        radius = float(particle.layer_radii[-1])
        return np.asarray(np.sum(rel * rel, axis=1) < (radius**2), dtype=bool)

    if isinstance(particle, Spheroid):
        body = rel @ _rotation_matrix_zyz_lab_to_body(particle.euler_angles).T
        a = float(particle.equatorial_radius)
        c = float(particle.polar_radius)
        rho2 = (body[:, 0] / a) ** 2 + (body[:, 1] / a) ** 2 + (body[:, 2] / c) ** 2
        return np.asarray(rho2 < 1.0, dtype=bool)

    raise TypeError(f"Unsupported particle instance: {type(particle)!r}")


def particle_t_signature(particle: Particle) -> tuple[object, ...]:
    """Return a position-independent cache key for solver-facing particle-T data.

    The particle-local `T` operator depends on shape, material, and orientation,
    but not on the particle center. This helper lets preparation paths reuse one
    computed particle kernel across many identical particles placed at different
    positions in the cluster.

    Notes
    -----
    This signature intentionally includes every particle field except
    `position`, so a rotated spheroid is treated as distinct from the same
    spheroid at another orientation. That is the correct cache key when the
    reused object is the final lab-frame operator consumed by the solver.
    """

    payload = []
    for field in fields(type(particle)):
        if field.name == "position":
            continue
        payload.append(getattr(particle, field.name))
    return (type(particle), *payload)


def particle_intrinsic_t_signature(particle: Particle) -> tuple[object, ...]:
    """Return a position/orientation-independent key for intrinsic particle data.

    This narrower signature is useful when particle preparation naturally splits
    into:
    - an intrinsic body-frame object that depends only on shape/material, and
    - a solver-facing lab-frame object that additionally depends on orientation.

    Spheres and layered spheres do not currently need that distinction, but
    axisymmetric particles do: one aligned spheroid model can be reused across
    many differently oriented copies before the final SVWF rotation step.
    """

    payload = []
    for field in fields(type(particle)):
        if field.name in {"position", "euler_angles"}:
            continue
        payload.append(getattr(particle, field.name))
    return (type(particle), *payload)


def _owned_read_only_array(values: np.ndarray, *, dtype: np.dtype) -> np.ndarray:
    out = np.array(values, dtype=dtype, copy=True, order="C")
    out.setflags(write=False)
    return out


class _ParticleBatch:
    """Internal homogeneous or descriptor-backed particle segment."""

    representation: ClassVar[ParticleTRepresentation | None] = None

    @property
    def positions(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def circumscribing_radii(self) -> np.ndarray:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def particle_at(self, index: int) -> Particle:
        raise NotImplementedError

    def iter_particles(self) -> Iterator[Particle]:
        for index in range(len(self)):
            yield self.particle_at(index)

    def outer_refractive_indices(self) -> np.ndarray | None:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class _ArrayParticleBatch(_ParticleBatch):
    _positions: np.ndarray
    _circumscribing_radii: np.ndarray

    @property
    def positions(self) -> np.ndarray:
        return self._positions

    @property
    def circumscribing_radii(self) -> np.ndarray:
        return self._circumscribing_radii

    def __len__(self) -> int:
        return int(self._positions.shape[0])


@dataclass(frozen=True, slots=True)
class _DescriptorBatch(_ArrayParticleBatch):
    descriptors: tuple[Particle, ...]

    @classmethod
    def from_particles(cls, particles: Sequence[Particle]) -> _DescriptorBatch:
        descriptors = tuple(particles)
        if not all(isinstance(particle, Particle) for particle in descriptors):
            bad = [
                type(particle).__name__
                for particle in descriptors
                if not isinstance(particle, Particle)
            ]
            raise TypeError(f"All entries in `particles` must be Particle instances. Got {bad}.")
        if descriptors:
            positions = np.asarray([particle.position for particle in descriptors], dtype=float)
            radii = np.asarray(
                [float(particle.circumscribing_radius()) for particle in descriptors],
                dtype=float,
            )
        else:
            positions = np.zeros((0, 3), dtype=float)
            radii = np.zeros((0,), dtype=float)
        return cls(
            _positions=_owned_read_only_array(positions, dtype=np.dtype(float)),
            _circumscribing_radii=_owned_read_only_array(radii, dtype=np.dtype(float)),
            descriptors=descriptors,
        )

    def __len__(self) -> int:
        return len(self.descriptors)

    def particle_at(self, index: int) -> Particle:
        return self.descriptors[int(index)]

    def iter_particles(self) -> Iterator[Particle]:
        return iter(self.descriptors)

    def outer_refractive_indices(self) -> np.ndarray | None:
        values: list[complex] = []
        for particle in self.descriptors:
            if isinstance(particle, Sphere):
                values.append(complex(particle.refractive_index))
            elif isinstance(particle, PECSphere):
                continue
            elif isinstance(particle, LayeredSphere):
                values.append(complex(particle.layer_refractive_indices[-1]))
            elif isinstance(particle, Spheroid):
                values.append(complex(particle.refractive_index))
            else:
                raise TypeError(
                    f"Unsupported particle type {type(particle).__name__!r} "
                    "for refractive-index checks."
                )
        return np.asarray(values, dtype=np.complex128)


@dataclass(frozen=True, slots=True)
class _SphereBatch(_ArrayParticleBatch):
    refractive_indices: np.ndarray
    representation: ClassVar[ParticleTRepresentation] = "diagonal"

    def particle_at(self, index: int) -> Particle:
        idx = int(index)
        position = self._positions[idx]
        return Sphere(
            position=(float(position[0]), float(position[1]), float(position[2])),
            radius=float(self._circumscribing_radii[idx]),
            refractive_index=complex(self.refractive_indices[idx]),
        )

    def outer_refractive_indices(self) -> np.ndarray:
        return self.refractive_indices


@dataclass(frozen=True, slots=True)
class _PECSphereBatch(_ArrayParticleBatch):
    representation: ClassVar[ParticleTRepresentation] = "diagonal"

    def particle_at(self, index: int) -> Particle:
        idx = int(index)
        position = self._positions[idx]
        return PECSphere(
            position=(float(position[0]), float(position[1]), float(position[2])),
            radius=float(self._circumscribing_radii[idx]),
        )

    def outer_refractive_indices(self) -> None:
        return None


class ParticleCollection(Sequence[Particle]):
    """Immutable particle sequence with compact sphere-family batches.

    Homogeneous and PEC sphere constructors avoid one persistent Python object
    per particle. Explicit descriptors and metadata-rich particle families
    share the same sequence API through a descriptor-backed fallback.
    """

    __slots__ = ("_batches", "_offsets", "_positions", "_radii")

    def __init__(self, batches: Sequence[_ParticleBatch] = ()) -> None:
        self._batches = tuple(batch for batch in batches if len(batch) > 0)
        offsets: list[int] = []
        count = 0
        for batch in self._batches:
            count += len(batch)
            offsets.append(count)
        self._offsets = tuple(offsets)

        if not self._batches:
            positions_owner = np.zeros((0, 3), dtype=float)
            radii_owner = np.zeros((0,), dtype=float)
        elif len(self._batches) == 1:
            positions_owner = self._batches[0].positions
            radii_owner = self._batches[0].circumscribing_radii
        else:
            positions_owner = np.concatenate([batch.positions for batch in self._batches], axis=0)
            radii_owner = np.concatenate(
                [batch.circumscribing_radii for batch in self._batches], axis=0
            )
        positions_owner.setflags(write=False)
        radii_owner.setflags(write=False)
        self._positions = positions_owner.view()
        self._radii = radii_owner.view()

    @classmethod
    def from_particles(
        cls, particles: Sequence[Particle] | ParticleCollection
    ) -> ParticleCollection:
        if isinstance(particles, cls):
            return particles
        batch = _DescriptorBatch.from_particles(particles)
        return cls((batch,)) if len(batch) else cls()

    @classmethod
    def concatenate(
        cls, *collections: Sequence[Particle] | ParticleCollection
    ) -> ParticleCollection:
        batches: list[_ParticleBatch] = []
        for collection in collections:
            normalized = cls.from_particles(collection)
            batches.extend(normalized._batches)
        return cls(batches)

    @property
    def positions(self) -> np.ndarray:
        return self._positions

    @property
    def circumscribing_radii(self) -> np.ndarray:
        return self._radii

    def __len__(self) -> int:
        return 0 if not self._offsets else int(self._offsets[-1])

    @overload
    def __getitem__(self, index: int) -> Particle: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[Particle, ...]: ...

    def __getitem__(self, index: int | slice) -> Particle | tuple[Particle, ...]:
        if isinstance(index, slice):
            return tuple(self[position] for position in range(*index.indices(len(self))))
        idx = int(index)
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError("particle index out of range")
        batch_index = bisect_right(self._offsets, idx)
        batch_start = 0 if batch_index == 0 else self._offsets[batch_index - 1]
        return self._batches[batch_index].particle_at(idx - batch_start)

    def __iter__(self) -> Iterator[Particle]:
        for batch in self._batches:
            yield from batch.iter_particles()

    def representation_groups(
        self,
    ) -> tuple[tuple[ParticleTRepresentation, np.ndarray], ...]:
        grouped: dict[ParticleTRepresentation, list[np.ndarray]] = {}
        order: list[ParticleTRepresentation] = []
        start = 0
        for batch in self._batches:
            stop = start + len(batch)
            segments: Iterable[tuple[ParticleTRepresentation, np.ndarray]]
            if batch.representation is None:
                local_groups: dict[ParticleTRepresentation, list[int]] = {}
                local_order: list[ParticleTRepresentation] = []
                for local, particle in enumerate(batch.iter_particles()):
                    representation = particle.t_operator_representation
                    if representation not in local_groups:
                        local_groups[representation] = []
                        local_order.append(representation)
                    local_groups[representation].append(start + local)
                segments = (
                    (
                        representation,
                        np.asarray(local_groups[representation], dtype=np.int64),
                    )
                    for representation in local_order
                )
            else:
                segments = (
                    (
                        batch.representation,
                        np.arange(start, stop, dtype=np.int64),
                    ),
                )
            for representation, indices in segments:
                if representation not in grouped:
                    grouped[representation] = []
                    order.append(representation)
                grouped[representation].append(indices)
            start = stop
        return tuple(
            (
                representation,
                np.concatenate(grouped[representation])
                if len(grouped[representation]) > 1
                else grouped[representation][0],
            )
            for representation in order
        )

    def sphere_parameters(
        self, particle_indices: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray] | None:
        indices = np.asarray(particle_indices, dtype=np.int64).reshape(-1)
        if indices.size == 0:
            return np.zeros((0,), dtype=float), np.zeros((0,), dtype=np.complex128)
        if int(indices.min()) < 0 or int(indices.max()) >= len(self):
            raise IndexError("sphere-parameter particle index out of range")
        if (
            len(self._batches) == 1
            and isinstance(self._batches[0], _SphereBatch)
            and indices.size == len(self)
            and int(indices[0]) == 0
            and int(indices[-1]) == len(self) - 1
            and bool(np.all(indices[1:] == indices[:-1] + 1))
        ):
            sphere_batch = self._batches[0]
            return sphere_batch.circumscribing_radii, sphere_batch.refractive_indices

        batch_indices = np.searchsorted(
            np.asarray(self._offsets, dtype=np.int64),
            indices,
            side="right",
        )
        batch_starts = np.asarray((0, *self._offsets[:-1]), dtype=np.int64)
        radii = np.empty((indices.size,), dtype=float)
        refractive_indices = np.empty((indices.size,), dtype=np.complex128)
        for batch_index in np.unique(batch_indices):
            selected_batch = self._batches[int(batch_index)]
            if not isinstance(selected_batch, _SphereBatch):
                return None
            selected = batch_indices == batch_index
            local = indices[selected] - batch_starts[int(batch_index)]
            radii[selected] = selected_batch.circumscribing_radii[local]
            refractive_indices[selected] = selected_batch.refractive_indices[local]
        return radii, refractive_indices

    def homogeneous_sphere_arrays(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Return compact sphere arrays when the collection is one sphere batch."""
        if len(self._batches) != 1 or not isinstance(self._batches[0], _SphereBatch):
            return None
        batch = self._batches[0]
        return batch.positions, batch.circumscribing_radii, batch.refractive_indices

    def homogeneous_pec_sphere_radii(self) -> np.ndarray | None:
        """Return compact radii when the collection is one PEC-sphere batch."""
        if len(self._batches) != 1 or not isinstance(self._batches[0], _PECSphereBatch):
            return None
        return self._batches[0].circumscribing_radii

    def outer_refractive_index_batches(self) -> Iterator[np.ndarray]:
        for batch in self._batches:
            values = batch.outer_refractive_indices()
            if values is not None and values.size:
                yield np.asarray(values, dtype=np.complex128)


def _as_positions_array(positions: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """Normalize particle-center inputs to shape `(N, 3)` float arrays."""
    pos = np.asarray(positions, dtype=float)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"`positions` must have shape (N, 3). Got {pos.shape}.")
    if not np.all(np.isfinite(pos)):
        raise ValueError("`positions` must contain only finite values.")
    return pos


def _as_complex_vector(
    values: Sequence[complex] | np.ndarray | complex,
    *,
    n: int,
    name: str,
) -> np.ndarray:
    """Normalize scalar-or-vector complex inputs to length `n`."""
    arr = np.asarray(values, dtype=np.complex128)
    if arr.ndim == 0:
        out = np.full((n,), complex(arr), dtype=np.complex128)
    else:
        out = arr.reshape(-1).astype(np.complex128, copy=False)
        if out.shape[0] != n:
            raise ValueError(
                f"`{name}` length ({out.shape[0]}) must match number of particles ({n})."
            )
    if not np.all(np.isfinite(out.real)) or not np.all(np.isfinite(out.imag)):
        raise ValueError(f"`{name}` must contain only finite values.")
    if np.any(out.real <= 0.0):
        raise ValueError(f"Real part of `{name}` must be strictly positive.")
    return out


def spheres_from_arrays(
    *,
    positions: Sequence[Sequence[float]] | np.ndarray,
    radii: Sequence[float] | np.ndarray,
    refractive_indices: Sequence[complex] | np.ndarray | complex,
) -> ParticleCollection:
    """Create an array-native homogeneous-sphere collection."""
    pos = _as_positions_array(positions)
    n = int(pos.shape[0])
    rad = np.asarray(radii, dtype=float).reshape(-1)
    if rad.shape[0] != n:
        raise ValueError(f"`radii` length ({rad.shape[0]}) must match number of particles ({n}).")
    if np.any(~np.isfinite(rad)) or np.any(rad <= 0.0):
        raise ValueError("`radii` must be finite and strictly positive.")
    n_part = _as_complex_vector(refractive_indices, n=n, name="refractive_indices")
    return ParticleCollection(
        (
            _SphereBatch(
                _positions=_owned_read_only_array(pos, dtype=np.dtype(float)),
                _circumscribing_radii=_owned_read_only_array(rad, dtype=np.dtype(float)),
                refractive_indices=_owned_read_only_array(n_part, dtype=np.dtype(np.complex128)),
            ),
        )
    )


def pec_spheres_from_arrays(
    *,
    positions: Sequence[Sequence[float]] | np.ndarray,
    radii: Sequence[float] | np.ndarray,
) -> ParticleCollection:
    """Create an array-native perfect-electric-conductor sphere collection.

    Unlike :func:`spheres_from_arrays`, this helper intentionally has no
    refractive-index argument: the particle response is the analytic PEC limit.
    """
    pos = _as_positions_array(positions)
    n = int(pos.shape[0])
    rad = np.asarray(radii, dtype=float).reshape(-1)
    if rad.shape[0] != n:
        raise ValueError(f"`radii` length ({rad.shape[0]}) must match number of particles ({n}).")
    if np.any(~np.isfinite(rad)) or np.any(rad <= 0.0):
        raise ValueError("`radii` must be finite and strictly positive.")
    return ParticleCollection(
        (
            _PECSphereBatch(
                _positions=_owned_read_only_array(pos, dtype=np.dtype(float)),
                _circumscribing_radii=_owned_read_only_array(rad, dtype=np.dtype(float)),
            ),
        )
    )


def _as_layer_matrix(
    values: Sequence[Sequence[float | complex]] | Sequence[float | complex] | np.ndarray,
    *,
    n_particles: int,
    name: str,
) -> np.ndarray:
    """Normalize layer-value inputs to shape `(N, L)` with optional broadcast."""
    arr = np.asarray(values)
    if arr.ndim == 1:
        if arr.size == 0:
            raise ValueError(f"`{name}` must contain at least one layer.")
        return np.broadcast_to(arr.reshape(1, -1), (n_particles, int(arr.size))).copy()
    if arr.ndim == 2:
        if arr.shape[0] != n_particles:
            raise ValueError(
                f"`{name}` first dimension ({arr.shape[0]}) must match number of particles ({n_particles})."
            )
        if arr.shape[1] == 0:
            raise ValueError(f"`{name}` must contain at least one layer.")
        return np.asarray(arr).copy()
    raise ValueError(f"`{name}` must be shaped (L,) or (N, L). Got shape {arr.shape}.")


def layered_spheres_from_arrays(
    *,
    positions: Sequence[Sequence[float]] | np.ndarray,
    layer_radii: Sequence[Sequence[float]] | Sequence[float] | np.ndarray,
    layer_refractive_indices: Sequence[Sequence[complex]] | Sequence[complex] | np.ndarray,
) -> ParticleCollection:
    """Create an immutable concentric layered-sphere collection.

    For convenience, `(L,)` layer inputs are broadcast to all particles.
    Combine batches with :meth:`ParticleCollection.concatenate`.
    """
    pos = _as_positions_array(positions)
    n = int(pos.shape[0])
    lr = _as_layer_matrix(layer_radii, n_particles=n, name="layer_radii").astype(float, copy=False)
    li = _as_layer_matrix(
        layer_refractive_indices,
        n_particles=n,
        name="layer_refractive_indices",
    ).astype(np.complex128, copy=False)

    if lr.shape != li.shape:
        raise ValueError(
            f"`layer_radii` and `layer_refractive_indices` must share shape. Got {lr.shape} vs {li.shape}."
        )
    if np.any(~np.isfinite(lr)) or np.any(lr <= 0.0):
        raise ValueError("`layer_radii` entries must be finite and strictly positive.")
    if np.any(np.diff(lr, axis=1) <= 0.0):
        raise ValueError("Each particle `layer_radii` row must be strictly increasing.")
    if not np.all(np.isfinite(li.real)) or not np.all(np.isfinite(li.imag)):
        raise ValueError("`layer_refractive_indices` must contain only finite values.")
    if np.any(li.real <= 0.0):
        raise ValueError("Real part of `layer_refractive_indices` must be strictly positive.")

    return ParticleCollection.from_particles(
        tuple(
            LayeredSphere(
                position=(float(p[0]), float(p[1]), float(p[2])),
                layer_radii=tuple(float(value) for value in radii_row),
                layer_refractive_indices=tuple(complex(value) for value in index_row),
            )
            for p, radii_row, index_row in zip(pos, lr, li, strict=True)
        )
    )


def _as_positive_float_vector(
    values: Sequence[float] | np.ndarray | float,
    *,
    n: int,
    name: str,
) -> np.ndarray:
    """Normalize scalar-or-vector positive float inputs to length `n`."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        out = np.full((n,), float(arr), dtype=float)
    else:
        out = arr.reshape(-1).astype(float, copy=False)
        if out.shape[0] != n:
            raise ValueError(
                f"`{name}` length ({out.shape[0]}) must match number of particles ({n})."
            )
    if np.any(~np.isfinite(out)) or np.any(out <= 0.0):
        raise ValueError(f"`{name}` must contain finite strictly positive values.")
    return out


def spheroids_from_arrays(
    *,
    positions: Sequence[Sequence[float]] | np.ndarray,
    equatorial_radii: Sequence[float] | np.ndarray | float,
    polar_radii: Sequence[float] | np.ndarray | float,
    refractive_indices: Sequence[complex] | np.ndarray | complex,
    euler_angles: Sequence[Sequence[float]] | Sequence[float] | np.ndarray = (0.0, 0.0, 0.0),
) -> ParticleCollection:
    """Create an immutable homogeneous-spheroid collection."""
    pos = _as_positions_array(positions)
    n = int(pos.shape[0])
    eq = _as_positive_float_vector(equatorial_radii, n=n, name="equatorial_radii")
    po = _as_positive_float_vector(polar_radii, n=n, name="polar_radii")

    eul = np.asarray(euler_angles, dtype=float)
    if eul.ndim == 1:
        if eul.shape[0] != 3:
            raise ValueError(f"`euler_angles` must have shape (3,) or (N,3). Got {eul.shape}.")
        eul = np.broadcast_to(eul.reshape(1, 3), (n, 3)).copy()
    elif eul.ndim == 2 and eul.shape == (n, 3):
        eul = eul.copy()
    else:
        raise ValueError(f"`euler_angles` must have shape (3,) or (N,3). Got {eul.shape}.")
    if np.any(~np.isfinite(eul)):
        raise ValueError("`euler_angles` must contain only finite values.")

    n_part = _as_complex_vector(refractive_indices, n=n, name="refractive_indices")
    return ParticleCollection.from_particles(
        tuple(
            Spheroid(
                position=(float(p[0]), float(p[1]), float(p[2])),
                equatorial_radius=float(a_eq),
                polar_radius=float(a_po),
                refractive_index=complex(index),
                euler_angles=(float(angles[0]), float(angles[1]), float(angles[2])),
            )
            for p, a_eq, a_po, index, angles in zip(
                pos,
                eq,
                po,
                n_part,
                eul,
                strict=True,
            )
        )
    )
