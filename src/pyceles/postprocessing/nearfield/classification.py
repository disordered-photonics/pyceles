from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from pyceles.core.particles import (
    LayeredSphere,
    Particle,
    ParticleCollection,
    PECSphere,
    Sphere,
    Spheroid,
    particle_contains_points,
)


@dataclass(frozen=True, slots=True)
class InternalPointClassification:
    """Sparse particle ownership of near-field sample points.

    `inside_any` marks points inside at least one physical particle.
    Only particles that own at least one point appear in the compact CSR-like
    arrays, avoiding one Python/NumPy object per particle on large systems.
    """

    n_particles: int
    inside_any: np.ndarray
    active_particle_indices: np.ndarray
    point_offsets: np.ndarray
    point_indices: np.ndarray

    @classmethod
    def from_active_points(
        cls,
        *,
        n_particles: int,
        inside_any: np.ndarray,
        entries: Sequence[tuple[int, np.ndarray]],
    ) -> InternalPointClassification:
        """Build compact storage from non-empty `(particle, points)` entries."""
        active: list[int] = []
        chunks: list[np.ndarray] = []
        offsets = [0]
        for particle_index, point_indices in entries:
            points = np.asarray(point_indices, dtype=np.intp).reshape(-1)
            if points.size == 0:
                continue
            active.append(int(particle_index))
            chunks.append(points)
            offsets.append(offsets[-1] + int(points.size))
        active_array = np.asarray(active, dtype=np.int64)
        if active_array.size > 1 and np.any(np.diff(active_array) <= 0):
            raise ValueError("Active particle entries must be strictly increasing.")
        return cls(
            n_particles=int(n_particles),
            inside_any=np.asarray(inside_any, dtype=bool).reshape(-1),
            active_particle_indices=active_array,
            point_offsets=np.asarray(offsets, dtype=np.int64),
            point_indices=(
                np.concatenate(chunks).astype(np.intp, copy=False)
                if chunks
                else np.zeros((0,), dtype=np.intp)
            ),
        )

    def points_for_particle(self, particle_index: int) -> np.ndarray:
        """Return the point-index view owned by one particle."""
        index = int(particle_index)
        if index < 0 or index >= self.n_particles:
            raise IndexError("particle index out of range")
        active_index = int(np.searchsorted(self.active_particle_indices, index))
        if (
            active_index >= self.active_particle_indices.size
            or int(self.active_particle_indices[active_index]) != index
        ):
            return self.point_indices[:0]
        start = int(self.point_offsets[active_index])
        stop = int(self.point_offsets[active_index + 1])
        return self.point_indices[start:stop]


def _supported_particle_archetypes(part: ParticleCollection) -> None:
    supported = (Sphere, PECSphere, LayeredSphere, Spheroid)
    bad = [
        type(particle).__name__
        for particle in part.archetypes
        if not isinstance(particle, supported)
    ]
    if bad:
        raise TypeError(
            "Internal point classification currently supports Sphere, PECSphere, LayeredSphere, "
            f"and Spheroid. Got {bad}."
        )


def _active_particle_mask(part: ParticleCollection, *, n_medium: complex) -> np.ndarray:
    """Return instance-aligned particles whose physical interior differs from the host."""
    n_medium_c = complex(n_medium)
    active_archetypes = np.asarray(
        [
            not (
                (isinstance(particle, Sphere) and complex(particle.refractive_index) == n_medium_c)
                or (
                    isinstance(particle, Spheroid)
                    and complex(particle.refractive_index) == n_medium_c
                )
                or (
                    isinstance(particle, LayeredSphere)
                    and all(
                        complex(n_layer) == n_medium_c
                        for n_layer in particle.layer_refractive_indices
                    )
                )
            )
            for particle in part.archetypes
        ],
        dtype=bool,
    )
    return active_archetypes[part.archetype_indices.astype(np.int64, copy=False)]


def classify_internal_points(
    field_points: np.ndarray,
    particles: Sequence[Particle],
    *,
    n_medium: complex = 1.0 + 0j,
) -> InternalPointClassification:
    """Classify near-field points against physical particle interiors."""
    pts = np.asarray(field_points, dtype=float).reshape(-1, 3)
    n_points = pts.shape[0]
    inside_any = np.zeros((n_points,), dtype=bool)
    part = ParticleCollection.from_particles(particles)
    _supported_particle_archetypes(part)

    if not len(part) or not n_points:
        return InternalPointClassification.from_active_points(
            n_particles=len(part),
            inside_any=inside_any,
            entries=(),
        )

    active_particles = _active_particle_mask(part, n_medium=n_medium)

    from scipy.spatial import cKDTree

    tree = cKDTree(part.positions)
    candidate_lists = tree.query_ball_point(
        pts,
        r=float(np.max(part.circumscribing_radii)),
        return_sorted=True,
    )
    points_by_particle: dict[int, list[int]] = {}
    archetype_ids = part.archetype_indices.astype(np.int64, copy=False)
    for point_index, candidates_raw in enumerate(candidate_lists):
        candidates = np.asarray(candidates_raw, dtype=np.int64)
        if candidates.size == 0:
            continue
        candidates = candidates[active_particles[candidates]]
        if candidates.size == 0:
            continue
        delta = pts[point_index] - part.positions[candidates]
        distance_squared = np.einsum("ij,ij->i", delta, delta)
        radii = part.circumscribing_radii[candidates]
        candidates = candidates[distance_squared < radii * radii]
        for particle_index in candidates:
            index = int(particle_index)
            archetype = part.archetypes[int(archetype_ids[index])]
            if isinstance(archetype, Spheroid) and not bool(
                particle_contains_points(part[index], pts[point_index])[0]
            ):
                continue
            points_by_particle.setdefault(index, []).append(point_index)
            inside_any[point_index] = True

    return InternalPointClassification.from_active_points(
        n_particles=len(part),
        inside_any=inside_any,
        entries=tuple(
            (particle_index, np.asarray(indices, dtype=np.intp))
            for particle_index, indices in sorted(points_by_particle.items())
        ),
    )


def _wrap_points_to_nearest_rectangular_image(
    *,
    points: np.ndarray,
    center: np.ndarray,
    lattice_ax: float,
    lattice_ay: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Wrap points to the nearest rectangular-lattice image of one center."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    ctr = np.asarray(center, dtype=float).reshape(3)
    dx = pts[:, 0] - float(ctr[0])
    dy = pts[:, 1] - float(ctr[1])
    nx = np.rint(dx / float(lattice_ax)).astype(np.int64, copy=False)
    ny = np.rint(dy / float(lattice_ay)).astype(np.int64, copy=False)
    wrapped = np.asarray(pts, dtype=float).copy()
    wrapped[:, 0] -= nx.astype(float) * float(lattice_ax)
    wrapped[:, 1] -= ny.astype(float) * float(lattice_ay)
    return wrapped, nx, ny


def _classify_periodic_internal_points_reference(
    *,
    points: np.ndarray,
    particles: ParticleCollection,
    lattice_ax: float,
    lattice_ay: float,
    k_parallel: np.ndarray,
    n_medium: complex,
) -> tuple[InternalPointClassification, np.ndarray, np.ndarray]:
    """Reference periodic classifier used when the bounded image tree is unsuitable."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    n_points = int(pts.shape[0])
    wrapped_points = np.asarray(pts, dtype=float).copy()
    bloch_phase = np.ones((n_points,), dtype=np.complex128)
    inside_any = np.zeros((n_points,), dtype=bool)
    active_entries: list[tuple[int, np.ndarray]] = []
    kpar = np.asarray(k_parallel, dtype=float).reshape(2)
    active_particles = _active_particle_mask(particles, n_medium=n_medium)

    for particle_index in np.flatnonzero(active_particles):
        index = int(particle_index)
        remaining = np.flatnonzero(~inside_any)
        if remaining.size == 0:
            break
        particle = particles[index]
        wrapped_remain, nx, ny = _wrap_points_to_nearest_rectangular_image(
            points=pts[remaining],
            center=np.asarray(particle.position, dtype=float),
            lattice_ax=float(lattice_ax),
            lattice_ay=float(lattice_ay),
        )
        mask = np.asarray(particle_contains_points(particle, wrapped_remain), dtype=bool)
        if not np.any(mask):
            continue
        owned = remaining[mask]
        active_entries.append((index, owned.astype(np.intp, copy=False)))
        inside_any[owned] = True
        wrapped_points[owned] = wrapped_remain[mask]
        phase_arg = kpar[0] * nx[mask].astype(float) * float(lattice_ax) + kpar[1] * ny[
            mask
        ].astype(float) * float(lattice_ay)
        bloch_phase[owned] = np.exp(1j * phase_arg)

    return (
        InternalPointClassification.from_active_points(
            n_particles=len(particles),
            inside_any=inside_any,
            entries=active_entries,
        ),
        wrapped_points,
        bloch_phase,
    )


def _classify_periodic_internal_points_tree(
    *,
    points: np.ndarray,
    particles: ParticleCollection,
    lattice_ax: float,
    lattice_ay: float,
    k_parallel: np.ndarray,
    n_medium: complex,
) -> tuple[InternalPointClassification, np.ndarray, np.ndarray] | None:
    """Use a 3x3 periodic image tree as a bounded broad phase when safe."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    n_points = int(pts.shape[0])
    n_particles = len(particles)
    inside_any = np.zeros((n_points,), dtype=bool)
    if n_points == 0 or n_particles == 0:
        return (
            InternalPointClassification.from_active_points(
                n_particles=n_particles,
                inside_any=inside_any,
                entries=(),
            ),
            np.asarray(pts, dtype=float).copy(),
            np.ones((n_points,), dtype=np.complex128),
        )

    active_particles = _active_particle_mask(particles, n_medium=n_medium)
    active_indices = np.flatnonzero(active_particles).astype(np.int64, copy=False)
    if active_indices.size == 0:
        return (
            InternalPointClassification.from_active_points(
                n_particles=n_particles,
                inside_any=inside_any,
                entries=(),
            ),
            np.asarray(pts, dtype=float).copy(),
            np.ones((n_points,), dtype=np.complex128),
        )

    ax = abs(float(lattice_ax))
    ay = abs(float(lattice_ay))
    if not np.isfinite(ax) or not np.isfinite(ay) or ax <= 0.0 or ay <= 0.0:
        raise ValueError("Periodic classification requires finite positive lattice periods.")

    positions = particles.positions
    radii = particles.circumscribing_radii
    active_pos = positions[active_indices]
    max_radius = float(np.max(radii[active_indices]))
    # A 3x3 image stencil is sufficient for ordinary non-self-overlapping
    # periodic particles and query points within roughly one neighbouring cell.
    # Retain the exact nearest-image reference path outside that bounded case.
    if 2.0 * max_radius >= min(ax, ay):
        return None
    x_extent = max(
        float(np.max(pts[:, 0]) - np.min(active_pos[:, 0])),
        float(np.max(active_pos[:, 0]) - np.min(pts[:, 0])),
    )
    y_extent = max(
        float(np.max(pts[:, 1]) - np.min(active_pos[:, 1])),
        float(np.max(active_pos[:, 1]) - np.min(pts[:, 1])),
    )
    if x_extent >= 1.5 * ax or y_extent >= 1.5 * ay:
        return None

    centers: list[np.ndarray] = []
    owners: list[np.ndarray] = []
    shifts: list[np.ndarray] = []
    for ix in (-1, 0, 1):
        for iy in (-1, 0, 1):
            shift = np.asarray([ix * float(lattice_ax), iy * float(lattice_ay), 0.0])
            centers.append(active_pos + shift[None, :])
            owners.append(active_indices)
            shifts.append(np.broadcast_to(shift, (active_indices.size, 3)).copy())
    image_centers = np.concatenate(centers, axis=0)
    image_owners = np.concatenate(owners, axis=0)
    image_shifts = np.concatenate(shifts, axis=0)

    from scipy.spatial import cKDTree

    candidates = cKDTree(image_centers).query_ball_point(
        pts,
        r=max_radius,
        return_sorted=True,
        workers=-1,
    )
    wrapped_points = np.asarray(pts, dtype=float).copy()
    bloch_phase = np.ones((n_points,), dtype=np.complex128)
    points_by_particle: dict[int, list[int]] = {}
    kpar = np.asarray(k_parallel, dtype=float).reshape(2)
    radius_squared = np.square(radii)
    archetype_ids = particles.archetype_indices.astype(np.int64, copy=False)

    for point_index, candidate_indices in enumerate(candidates):
        selected_image: int | None = None
        selected_particle = n_particles
        for image_index_raw in candidate_indices:
            image_index = int(image_index_raw)
            particle_index = int(image_owners[image_index])
            delta = pts[point_index] - image_centers[image_index]
            if float(delta @ delta) >= float(radius_squared[particle_index]):
                continue
            archetype = particles.archetypes[int(archetype_ids[particle_index])]
            if isinstance(archetype, Spheroid):
                wrapped_candidate = pts[point_index] - image_shifts[image_index]
                if not bool(
                    particle_contains_points(
                        particles[particle_index],
                        wrapped_candidate[None, :],
                    )[0]
                ):
                    continue
            if particle_index < selected_particle:
                selected_particle = particle_index
                selected_image = image_index
        if selected_image is None:
            continue
        inside_any[point_index] = True
        points_by_particle.setdefault(selected_particle, []).append(point_index)
        shift = image_shifts[selected_image]
        wrapped_points[point_index] = pts[point_index] - shift
        bloch_phase[point_index] = np.exp(1j * (kpar[0] * shift[0] + kpar[1] * shift[1]))

    return (
        InternalPointClassification.from_active_points(
            n_particles=n_particles,
            inside_any=inside_any,
            entries=tuple(
                (particle_index, np.asarray(point_indices, dtype=np.intp))
                for particle_index, point_indices in sorted(points_by_particle.items())
            ),
        ),
        wrapped_points,
        bloch_phase,
    )


def classify_periodic_internal_points(
    points: np.ndarray,
    particles: Sequence[Particle],
    *,
    lattice_ax: float,
    lattice_ay: float,
    k_parallel: np.ndarray,
    n_medium: complex = 1.0 + 0j,
) -> tuple[InternalPointClassification, np.ndarray, np.ndarray]:
    """Classify physical particle interiors across rectangular periodic images.

    The common path uses a compact 3x3 image-tree broad phase and exact shape
    predicates only for the few candidate particles near each query point.
    Unusual large-particle or far-outside-cell queries transparently retain the
    reference nearest-image scan.
    """
    part = ParticleCollection.from_particles(particles)
    _supported_particle_archetypes(part)
    tree_result = _classify_periodic_internal_points_tree(
        points=points,
        particles=part,
        lattice_ax=float(lattice_ax),
        lattice_ay=float(lattice_ay),
        k_parallel=k_parallel,
        n_medium=n_medium,
    )
    if tree_result is not None:
        return tree_result
    return _classify_periodic_internal_points_reference(
        points=points,
        particles=part,
        lattice_ax=float(lattice_ax),
        lattice_ay=float(lattice_ay),
        k_parallel=k_parallel,
        n_medium=n_medium,
    )


__all__ = [
    "InternalPointClassification",
    "classify_internal_points",
    "classify_periodic_internal_points",
]
