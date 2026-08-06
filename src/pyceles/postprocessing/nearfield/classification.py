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

    supported = (Sphere, PECSphere, LayeredSphere, Spheroid)
    bad = [type(p).__name__ for p in part.archetypes if not isinstance(p, supported)]
    if bad:
        raise TypeError(
            "Internal point classification currently supports Sphere, PECSphere, LayeredSphere, "
            f"and Spheroid. Got {bad}."
        )

    if not len(part) or not n_points:
        return InternalPointClassification.from_active_points(
            n_particles=len(part),
            inside_any=inside_any,
            entries=(),
        )

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
    archetype_ids = part.archetype_indices.astype(np.int64, copy=False)
    active_particles = active_archetypes[archetype_ids]

    from scipy.spatial import cKDTree

    tree = cKDTree(part.positions)
    candidate_lists = tree.query_ball_point(
        pts,
        r=float(np.max(part.circumscribing_radii)),
        return_sorted=True,
    )
    points_by_particle: dict[int, list[int]] = {}
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


__all__ = ["InternalPointClassification", "classify_internal_points"]
