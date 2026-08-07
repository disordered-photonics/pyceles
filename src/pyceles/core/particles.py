from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, fields, replace
from types import MappingProxyType
from typing import Literal, overload

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


def _freeze_signature_value(value: object) -> object:
    """Convert descriptor metadata into a deterministic hashable cache key."""
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return ("ndarray", array.dtype.str, array.shape, array.tobytes())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        frozen_items = [
            (
                _freeze_signature_value(key),
                _freeze_signature_value(item),
            )
            for key, item in value.items()
        ]
        return tuple(sorted(frozen_items, key=lambda item: repr(item[0])))
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_signature_value(item) for item in value)
    try:
        hash(value)
    except TypeError as exc:
        raise TypeError(
            f"Particle metadata value of type {type(value).__name__!r} is not hashable. "
            "Use immutable scalars, tuples, mappings, or NumPy arrays."
        ) from exc
    return value


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
        payload.append(_freeze_signature_value(getattr(particle, field.name)))
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
        payload.append(_freeze_signature_value(getattr(particle, field.name)))
    return (type(particle), *payload)


def _owned_read_only_array(values: np.ndarray, *, dtype: np.dtype) -> np.ndarray:
    out = np.array(values, dtype=dtype, copy=True, order="C")
    out.setflags(write=False)
    return out


def _compact_index_dtype(n_values: int) -> np.dtype:
    """Return the smallest practical unsigned dtype for non-negative indices."""
    maximum = max(0, int(n_values) - 1)
    if maximum <= np.iinfo(np.uint8).max:
        return np.dtype(np.uint8)
    if maximum <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    if maximum <= np.iinfo(np.uint32).max:
        return np.dtype(np.uint32)
    return np.dtype(np.uint64)


def _stable_unique_rows(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return unique rows in first-occurrence order and a compact inverse map."""
    rows = np.ascontiguousarray(values)
    if rows.ndim != 2:
        raise ValueError(f"Expected a 2D row table. Got shape {rows.shape}.")
    if rows.shape[0] == 0:
        return rows.copy(), np.zeros((0,), dtype=np.uint8)
    first_row = rows[0]
    chunk_size = 65_536
    if all(
        bool(np.all(rows[start : start + chunk_size] == first_row))
        for start in range(0, rows.shape[0], chunk_size)
    ):
        return rows[:1].copy(), np.zeros((rows.shape[0],), dtype=np.uint8)
    unique, first, inverse = np.unique(rows, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first, kind="stable")
    old_to_new = np.empty(order.size, dtype=np.int64)
    old_to_new[order] = np.arange(order.size, dtype=np.int64)
    mapped = old_to_new[inverse]
    return unique[order], mapped.astype(_compact_index_dtype(order.size), copy=False)


def _owned_archetype_value(value: object) -> object:
    """Copy mutable metadata into an immutable collection-owned form."""
    if isinstance(value, np.ndarray):
        array = np.array(value, copy=True, order="C")
        array.setflags(write=False)
        return array
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                _owned_archetype_value(key): _owned_archetype_value(item)
                for key, item in value.items()
            }
        )
    if isinstance(value, (tuple, list)):
        return tuple(_owned_archetype_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_owned_archetype_value(item) for item in value)
    return value


def _particle_at_origin(particle: Particle) -> Particle:
    """Return a collection-owned position-independent immutable archetype."""
    overrides = {
        field.name: _owned_archetype_value(getattr(particle, field.name))
        for field in fields(type(particle))
        if field.name != "position"
    }
    return replace(particle, position=(0.0, 0.0, 0.0), **overrides)


@dataclass(frozen=True, slots=True)
class ParticleArchetypeGroup:
    """Instances sharing one prepared single-body representation.

    ``archetype_indices`` addresses the collection-wide archetype table. The
    aligned ``operator_indices`` array maps each particle in ``particle_indices``
    to the local unique-archetype/operator row used by the prepared group.
    """

    representation: ParticleTRepresentation
    particle_indices: np.ndarray
    archetype_indices: np.ndarray
    operator_indices: np.ndarray


class ParticleCollection(Sequence[Particle]):
    """Immutable columnar particle instances backed by shared archetypes.

    Every particle family uses the same storage contract:

    - ``positions`` stores one center per instance;
    - ``archetype_indices`` stores one compact integer tag per instance;
    - ``archetypes`` stores each distinct position-independent descriptor once.

    Materializing ``collection[i]`` remains convenient for small heterogeneous
    workflows, while solver preparation can consume the archetype table and the
    instance map without allocating one descriptor or one T matrix per particle.
    """

    __slots__ = ("_archetype_indices", "_archetypes", "_positions", "_radii")

    def __init__(
        self,
        positions: Sequence[Sequence[float]] | np.ndarray = (),
        archetypes: Sequence[Particle] = (),
        archetype_indices: Sequence[int] | np.ndarray = (),
    ) -> None:
        pos_arr = np.asarray(positions, dtype=float)
        if pos_arr.size == 0:
            pos_arr = np.zeros((0, 3), dtype=float)
        if pos_arr.ndim != 2 or pos_arr.shape[1] != 3:
            raise ValueError(f"`positions` must have shape (N, 3). Got {pos_arr.shape}.")
        if not np.all(np.isfinite(pos_arr)):
            raise ValueError("`positions` must contain only finite values.")

        archetype_tuple = tuple(archetypes)
        if not all(isinstance(particle, Particle) for particle in archetype_tuple):
            raise TypeError("All archetypes must be Particle instances.")
        raw_archetypes = tuple(_particle_at_origin(particle) for particle in archetype_tuple)

        ids = np.asarray(archetype_indices)
        if ids.size == 0:
            ids = np.zeros((0,), dtype=np.int64)
        if ids.ndim != 1 or ids.shape[0] != pos_arr.shape[0]:
            raise ValueError(
                "`archetype_indices` must be a length-N vector aligned with `positions`. "
                f"Got {ids.shape} for N={pos_arr.shape[0]}."
            )
        if not np.issubdtype(ids.dtype, np.integer):
            raise TypeError("`archetype_indices` must contain integers.")
        ids_i64 = ids.astype(np.int64, copy=False)
        if ids_i64.size and (int(ids_i64.min()) < 0 or int(ids_i64.max()) >= len(raw_archetypes)):
            raise IndexError("`archetype_indices` contains an out-of-range archetype id.")
        if pos_arr.shape[0] and not raw_archetypes:
            raise ValueError("Non-empty particle instances require at least one archetype.")

        # Canonicalize duplicate archetypes even when callers provide a redundant
        # table. This keeps concatenation and third-party constructors on the same
        # compact representation as the built-in array helpers.
        canonical: list[Particle] = []
        canonical_by_signature: dict[tuple[object, ...], int] = {}
        old_to_new = np.empty((len(raw_archetypes),), dtype=np.int64)
        for old_index, particle in enumerate(raw_archetypes):
            signature = particle_t_signature(particle)
            new_index = canonical_by_signature.get(signature)
            if new_index is None:
                new_index = len(canonical)
                canonical_by_signature[signature] = new_index
                canonical.append(particle)
            old_to_new[old_index] = new_index
        mapped = old_to_new[ids_i64] if ids_i64.size else ids_i64

        # Retain only referenced archetypes. This prevents an externally supplied
        # redundant table from leaking unused metadata into validation, storage,
        # or prepared-operator planning.
        if mapped.size:
            used, first, inverse = np.unique(mapped, return_index=True, return_inverse=True)
            order = np.argsort(first, kind="stable")
            canonical = [canonical[int(used[index])] for index in order]
            old_to_used = np.empty(order.size, dtype=np.int64)
            old_to_used[order] = np.arange(order.size, dtype=np.int64)
            mapped = old_to_used[inverse]
        else:
            canonical = []

        radii_by_archetype = np.asarray(
            [float(particle.circumscribing_radius()) for particle in canonical], dtype=float
        )
        radii = radii_by_archetype[mapped] if mapped.size else np.zeros((0,), dtype=float)
        if np.any(~np.isfinite(radii)) or np.any(radii <= 0.0):
            raise ValueError("Particle circumscribing radii must be finite and strictly positive.")

        self._positions = _owned_read_only_array(pos_arr, dtype=np.dtype(float))
        self._archetypes = tuple(canonical)
        self._archetype_indices = _owned_read_only_array(
            mapped, dtype=_compact_index_dtype(len(canonical))
        )
        self._radii = _owned_read_only_array(radii, dtype=np.dtype(float))

    @classmethod
    def from_archetypes(
        cls,
        *,
        positions: Sequence[Sequence[float]] | np.ndarray,
        archetypes: Sequence[Particle],
        archetype_indices: Sequence[int] | np.ndarray,
    ) -> ParticleCollection:
        """Build a scalable collection from shared immutable archetypes."""
        return cls(positions, archetypes, archetype_indices)

    @classmethod
    def from_particles(
        cls, particles: Sequence[Particle] | ParticleCollection
    ) -> ParticleCollection:
        """Normalize explicit descriptors into the same shared-archetype model."""
        if isinstance(particles, cls):
            return particles
        descriptors = tuple(particles)
        if not all(isinstance(particle, Particle) for particle in descriptors):
            bad = [
                type(particle).__name__
                for particle in descriptors
                if not isinstance(particle, Particle)
            ]
            raise TypeError(f"All entries in `particles` must be Particle instances. Got {bad}.")
        if not descriptors:
            return cls()

        positions = np.asarray([particle.position for particle in descriptors], dtype=float)
        archetypes: list[Particle] = []
        archetype_by_signature: dict[tuple[object, ...], int] = {}
        ids = np.empty((len(descriptors),), dtype=np.int64)
        for index, particle in enumerate(descriptors):
            signature = particle_t_signature(particle)
            archetype_index = archetype_by_signature.get(signature)
            if archetype_index is None:
                archetype_index = len(archetypes)
                archetype_by_signature[signature] = archetype_index
                archetypes.append(particle)
            ids[index] = archetype_index
        return cls.from_archetypes(
            positions=positions,
            archetypes=archetypes,
            archetype_indices=ids,
        )

    @classmethod
    def concatenate(
        cls, *collections: Sequence[Particle] | ParticleCollection
    ) -> ParticleCollection:
        """Concatenate collections while globally deduplicating archetypes."""
        normalized = [cls.from_particles(collection) for collection in collections]
        normalized = [collection for collection in normalized if len(collection)]
        if not normalized:
            return cls()
        positions = np.concatenate([collection.positions for collection in normalized], axis=0)
        archetypes: list[Particle] = []
        ids: list[np.ndarray] = []
        offset = 0
        for collection in normalized:
            archetypes.extend(collection.archetypes)
            ids.append(collection.archetype_indices.astype(np.int64, copy=False) + offset)
            offset += collection.n_archetypes
        return cls.from_archetypes(
            positions=positions,
            archetypes=archetypes,
            archetype_indices=np.concatenate(ids),
        )

    @property
    def positions(self) -> np.ndarray:
        return self._positions

    @property
    def circumscribing_radii(self) -> np.ndarray:
        return self._radii

    @property
    def archetypes(self) -> tuple[Particle, ...]:
        """Distinct position-independent particle descriptors."""
        return self._archetypes

    @property
    def archetype_indices(self) -> np.ndarray:
        """Compact instance-to-archetype map aligned with ``positions``."""
        return self._archetype_indices

    @property
    def n_archetypes(self) -> int:
        return len(self._archetypes)

    def __len__(self) -> int:
        return int(self._positions.shape[0])

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
        position = self._positions[idx]
        archetype = self._archetypes[int(self._archetype_indices[idx])]
        return replace(
            archetype,
            position=(float(position[0]), float(position[1]), float(position[2])),
        )

    def __iter__(self) -> Iterator[Particle]:
        for index in range(len(self)):
            yield self[index]

    def representation_groups(self) -> tuple[tuple[ParticleTRepresentation, np.ndarray], ...]:
        """Return particle indices grouped by prepared T representation."""
        return tuple(
            (group.representation, group.particle_indices) for group in self.archetype_groups()
        )

    def archetype_groups(self) -> tuple[ParticleArchetypeGroup, ...]:
        """Plan representation groups with local shared-archetype mappings."""
        if not len(self):
            return ()
        representations = tuple(particle.t_operator_representation for particle in self._archetypes)
        ids_i64 = self._archetype_indices.astype(np.int64, copy=False)
        used_ids, first_positions = np.unique(ids_i64, return_index=True)
        first_order = np.argsort(first_positions, kind="stable")
        first_used_ids = used_ids[first_order]
        ordered_representations: list[ParticleTRepresentation] = []
        for archetype_id in first_used_ids:
            representation = representations[int(archetype_id)]
            if representation not in ordered_representations:
                ordered_representations.append(representation)

        groups: list[ParticleArchetypeGroup] = []
        for representation in ordered_representations:
            matching_archetypes = np.asarray(
                [rep == representation for rep in representations], dtype=bool
            )
            particle_indices = np.flatnonzero(matching_archetypes[ids_i64]).astype(
                np.int64, copy=False
            )
            global_ids = ids_i64[particle_indices]
            group_archetype_mask = np.asarray(
                [matching_archetypes[int(archetype_id)] for archetype_id in first_used_ids],
                dtype=bool,
            )
            group_archetype_ids = first_used_ids[group_archetype_mask]
            global_to_local = np.full(len(representations), -1, dtype=np.int64)
            global_to_local[group_archetype_ids] = np.arange(group_archetype_ids.size)
            local_ids = global_to_local[global_ids]
            groups.append(
                ParticleArchetypeGroup(
                    representation=representation,
                    particle_indices=_owned_read_only_array(
                        particle_indices, dtype=np.dtype(np.int64)
                    ),
                    archetype_indices=_owned_read_only_array(
                        group_archetype_ids, dtype=np.dtype(np.int64)
                    ),
                    operator_indices=_owned_read_only_array(
                        local_ids, dtype=_compact_index_dtype(group_archetype_ids.size)
                    ),
                )
            )
        return tuple(groups)

    def indices_of_type(self, particle_type: type[Particle]) -> np.ndarray:
        """Return instance indices whose archetype is an instance of ``particle_type``."""
        matching = np.asarray(
            [isinstance(archetype, particle_type) for archetype in self._archetypes],
            dtype=bool,
        )
        ids = self._archetype_indices.astype(np.int64, copy=False)
        out = np.flatnonzero(matching[ids]).astype(np.int64, copy=False)
        out.setflags(write=False)
        return out

    def homogeneous_sphere_arrays(
        self,
        particle_indices: Sequence[int] | np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Expose a homogeneous-sphere compute view for all or selected instances."""
        if particle_indices is None:
            sphere_archetypes = tuple(
                archetype for archetype in self._archetypes if isinstance(archetype, Sphere)
            )
            if len(sphere_archetypes) != len(self._archetypes):
                return None
            selected_positions = self._positions
            inverse = self._archetype_indices
        else:
            selected = np.asarray(particle_indices, dtype=np.int64).reshape(-1)
            if selected.size and (int(selected.min()) < 0 or int(selected.max()) >= len(self)):
                raise IndexError("particle index out of range")
            selected_positions = self._positions[selected]
            selected_archetype_ids = self._archetype_indices[selected]
            used_archetype_ids, inverse = np.unique(selected_archetype_ids, return_inverse=True)
            selected_archetypes = tuple(
                self._archetypes[int(archetype_id)] for archetype_id in used_archetype_ids
            )
            sphere_archetypes = tuple(
                archetype for archetype in selected_archetypes if isinstance(archetype, Sphere)
            )
            if len(sphere_archetypes) != used_archetype_ids.size:
                return None

        radii_by_archetype = np.asarray(
            [float(archetype.radius) for archetype in sphere_archetypes], dtype=np.float64
        )
        refractive_indices_by_archetype = np.asarray(
            [complex(archetype.refractive_index) for archetype in sphere_archetypes],
            dtype=np.complex128,
        )
        return (
            selected_positions,
            radii_by_archetype[inverse],
            refractive_indices_by_archetype[inverse],
        )

    def outer_refractive_index_batches(self) -> Iterator[np.ndarray]:
        """Yield unique outer refractive indices for validation."""
        values: list[complex] = []
        for particle in self._archetypes:
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
        if values:
            yield np.asarray(values, dtype=np.complex128)


def _as_positions_array(positions: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """Normalize particle-center inputs to shape ``(N, 3)`` float arrays."""
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
    """Normalize scalar-or-vector complex inputs to length ``n``."""
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
    """Create homogeneous-sphere instances in the shared-archetype model."""
    pos = _as_positions_array(positions)
    n = int(pos.shape[0])
    rad = np.asarray(radii, dtype=float).reshape(-1)
    if rad.shape[0] != n:
        raise ValueError(f"`radii` length ({rad.shape[0]}) must match number of particles ({n}).")
    if np.any(~np.isfinite(rad)) or np.any(rad <= 0.0):
        raise ValueError("`radii` must be finite and strictly positive.")
    n_part = _as_complex_vector(refractive_indices, n=n, name="refractive_indices")
    rows, ids = _stable_unique_rows(np.column_stack((rad, n_part.real, n_part.imag)))
    archetypes = tuple(
        Sphere(
            position=(0.0, 0.0, 0.0),
            radius=float(radius),
            refractive_index=complex(float(n_real), float(n_imag)),
        )
        for radius, n_real, n_imag in rows
    )
    return ParticleCollection.from_archetypes(
        positions=pos,
        archetypes=archetypes,
        archetype_indices=ids,
    )


def pec_spheres_from_arrays(
    *,
    positions: Sequence[Sequence[float]] | np.ndarray,
    radii: Sequence[float] | np.ndarray,
) -> ParticleCollection:
    """Create PEC-sphere instances in the shared-archetype model."""
    pos = _as_positions_array(positions)
    n = int(pos.shape[0])
    rad = np.asarray(radii, dtype=float).reshape(-1)
    if rad.shape[0] != n:
        raise ValueError(f"`radii` length ({rad.shape[0]}) must match number of particles ({n}).")
    if np.any(~np.isfinite(rad)) or np.any(rad <= 0.0):
        raise ValueError("`radii` must be finite and strictly positive.")
    rows, ids = _stable_unique_rows(rad[:, None])
    archetypes = tuple(PECSphere(position=(0.0, 0.0, 0.0), radius=float(row[0])) for row in rows)
    return ParticleCollection.from_archetypes(
        positions=pos,
        archetypes=archetypes,
        archetype_indices=ids,
    )


def _as_layer_matrix(
    values: Sequence[Sequence[float | complex]] | Sequence[float | complex] | np.ndarray,
    *,
    n_particles: int,
    name: str,
) -> np.ndarray:
    """Normalize layer-value inputs to shape ``(N, L)`` with optional broadcast."""
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
    """Create layered-sphere instances in the shared-archetype model."""
    pos = _as_positions_array(positions)
    n = int(pos.shape[0])
    lr_input = np.asarray(layer_radii)
    li_input = np.asarray(layer_refractive_indices)
    if lr_input.ndim == 1 and li_input.ndim == 1:
        lr_single = lr_input.astype(float, copy=False).reshape(-1)
        li_single = li_input.astype(np.complex128, copy=False).reshape(-1)
        if lr_single.size == 0 or li_single.size == 0:
            raise ValueError("Layer metadata must contain at least one layer.")
        if lr_single.shape != li_single.shape:
            raise ValueError(
                "`layer_radii` and `layer_refractive_indices` must share shape. "
                f"Got {lr_single.shape} vs {li_single.shape}."
            )
        if np.any(~np.isfinite(lr_single)) or np.any(lr_single <= 0.0):
            raise ValueError("`layer_radii` entries must be finite and strictly positive.")
        if np.any(np.diff(lr_single) <= 0.0):
            raise ValueError("Each particle `layer_radii` row must be strictly increasing.")
        if not np.all(np.isfinite(li_single.real)) or not np.all(np.isfinite(li_single.imag)):
            raise ValueError("`layer_refractive_indices` must contain only finite values.")
        if np.any(li_single.real <= 0.0):
            raise ValueError("Real part of `layer_refractive_indices` must be strictly positive.")
        return ParticleCollection.from_archetypes(
            positions=pos,
            archetypes=(
                LayeredSphere(
                    position=(0.0, 0.0, 0.0),
                    layer_radii=tuple(float(value) for value in lr_single),
                    layer_refractive_indices=tuple(complex(value) for value in li_single),
                ),
            ),
            archetype_indices=np.zeros((n,), dtype=np.uint8),
        )

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

    n_layers = int(lr.shape[1])
    rows, ids = _stable_unique_rows(np.concatenate((lr, li.real, li.imag), axis=1))
    archetypes = tuple(
        LayeredSphere(
            position=(0.0, 0.0, 0.0),
            layer_radii=tuple(float(value) for value in row[:n_layers]),
            layer_refractive_indices=tuple(
                complex(float(real), float(imag))
                for real, imag in zip(
                    row[n_layers : 2 * n_layers],
                    row[2 * n_layers :],
                    strict=True,
                )
            ),
        )
        for row in rows
    )
    return ParticleCollection.from_archetypes(
        positions=pos,
        archetypes=archetypes,
        archetype_indices=ids,
    )


def _as_positive_float_vector(
    values: Sequence[float] | np.ndarray | float,
    *,
    n: int,
    name: str,
) -> np.ndarray:
    """Normalize scalar-or-vector positive float inputs to length ``n``."""
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
    """Create spheroid instances in the shared-archetype model."""
    pos = _as_positions_array(positions)
    n = int(pos.shape[0])
    eq_input = np.asarray(equatorial_radii, dtype=float)
    po_input = np.asarray(polar_radii, dtype=float)
    index_input = np.asarray(refractive_indices, dtype=np.complex128)
    eul = np.asarray(euler_angles, dtype=float)
    if (
        eq_input.ndim == 0
        and po_input.ndim == 0
        and index_input.ndim == 0
        and eul.ndim == 1
        and eul.shape == (3,)
    ):
        eq_single = float(eq_input)
        po_single = float(po_input)
        index_single = complex(index_input)
        if not np.isfinite(eq_single) or eq_single <= 0.0:
            raise ValueError("`equatorial_radii` must contain finite strictly positive values.")
        if not np.isfinite(po_single) or po_single <= 0.0:
            raise ValueError("`polar_radii` must contain finite strictly positive values.")
        if not np.isfinite(index_single.real) or not np.isfinite(index_single.imag):
            raise ValueError("`refractive_indices` must contain only finite values.")
        if index_single.real <= 0.0:
            raise ValueError("Real part of `refractive_indices` must be strictly positive.")
        if np.any(~np.isfinite(eul)):
            raise ValueError("`euler_angles` must contain only finite values.")
        return ParticleCollection.from_archetypes(
            positions=pos,
            archetypes=(
                Spheroid(
                    position=(0.0, 0.0, 0.0),
                    equatorial_radius=eq_single,
                    polar_radius=po_single,
                    refractive_index=index_single,
                    euler_angles=(float(eul[0]), float(eul[1]), float(eul[2])),
                ),
            ),
            archetype_indices=np.zeros((n,), dtype=np.uint8),
        )

    eq = _as_positive_float_vector(equatorial_radii, n=n, name="equatorial_radii")
    po = _as_positive_float_vector(polar_radii, n=n, name="polar_radii")

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
    rows, ids = _stable_unique_rows(np.column_stack((eq, po, n_part.real, n_part.imag, eul)))
    archetypes = tuple(
        Spheroid(
            position=(0.0, 0.0, 0.0),
            equatorial_radius=float(row[0]),
            polar_radius=float(row[1]),
            refractive_index=complex(float(row[2]), float(row[3])),
            euler_angles=(float(row[4]), float(row[5]), float(row[6])),
        )
        for row in rows
    )
    return ParticleCollection.from_archetypes(
        positions=pos,
        archetypes=archetypes,
        archetype_indices=ids,
    )
