from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from pyceles.core.particles import (
    LayeredSphere,
    Particle,
    Sphere,
    Spheroid,
    particle_contains_points,
)


@dataclass(frozen=True)
class InternalPointClassification:
    """Broad-phase particle ownership of near-field sample points.

    `inside_any` marks points inside at least one physical particle.
    `point_indices_by_particle[j]` stores the global point indices associated
    with particle `j` under the same broad-phase rule.
    """

    inside_any: np.ndarray
    point_indices_by_particle: tuple[np.ndarray, ...]


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
    by_particle: list[np.ndarray] = []

    supported = (Sphere, LayeredSphere, Spheroid)
    bad = [type(p).__name__ for p in particles if not isinstance(p, supported)]
    if bad:
        raise TypeError(
            "Internal point classification currently supports Sphere, LayeredSphere, "
            f"and Spheroid. Got {bad}."
        )

    n_medium_c = complex(n_medium)
    for particle in particles:
        if isinstance(particle, Sphere) and complex(particle.refractive_index) == n_medium_c:
            idx = np.zeros((0,), dtype=np.intp)
        elif isinstance(particle, Spheroid) and complex(particle.refractive_index) == n_medium_c:
            idx = np.zeros((0,), dtype=np.intp)
        elif isinstance(particle, LayeredSphere) and all(
            complex(n_layer) == n_medium_c for n_layer in particle.layer_refractive_indices
        ):
            idx = np.zeros((0,), dtype=np.intp)
        else:
            idx = np.flatnonzero(particle_contains_points(particle, pts)).astype(
                np.intp, copy=False
            )
        by_particle.append(idx)
        inside_any[idx] = True

    return InternalPointClassification(
        inside_any=inside_any,
        point_indices_by_particle=tuple(by_particle),
    )


__all__ = ["InternalPointClassification", "classify_internal_points"]
