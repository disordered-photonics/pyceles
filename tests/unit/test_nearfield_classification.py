from __future__ import annotations

import numpy as np

from pyceles.core.particles import ParticleCollection, Sphere
from pyceles.postprocessing.nearfield.classification import classify_internal_points


def test_internal_point_classification_stores_only_active_particles() -> None:
    n_particles = 10_000
    positions = np.column_stack(
        (
            10.0 * np.arange(n_particles, dtype=float),
            np.zeros(n_particles, dtype=float),
            np.zeros(n_particles, dtype=float),
        )
    )
    particles = ParticleCollection.from_archetypes(
        positions=positions,
        archetypes=[Sphere(position=(0.0, 0.0, 0.0), radius=1.0, refractive_index=1.5 + 0.0j)],
        archetype_indices=np.zeros(n_particles, dtype=np.uint8),
    )
    points = np.asarray(
        [
            [0.25, 0.0, 0.0],
            [50_000.25, 0.0, 0.0],
            [-5.0, 0.0, 0.0],
        ]
    )

    classification = classify_internal_points(points, particles)

    np.testing.assert_array_equal(classification.inside_any, [True, True, False])
    np.testing.assert_array_equal(classification.active_particle_indices, [0, 5_000])
    np.testing.assert_array_equal(classification.point_offsets, [0, 1, 2])
    np.testing.assert_array_equal(classification.points_for_particle(0), [0])
    np.testing.assert_array_equal(classification.points_for_particle(5_000), [1])
    assert classification.points_for_particle(9_999).size == 0


def test_internal_point_classification_excludes_index_matched_particles() -> None:
    particles = ParticleCollection.from_archetypes(
        positions=np.asarray([[0.0, 0.0, 0.0]]),
        archetypes=[Sphere(position=(0.0, 0.0, 0.0), radius=2.0, refractive_index=1.0 + 0.0j)],
        archetype_indices=np.asarray([0], dtype=np.uint8),
    )

    classification = classify_internal_points(
        np.asarray([[0.0, 0.0, 0.0]]),
        particles,
        n_medium=1.0 + 0.0j,
    )

    assert not classification.inside_any[0]
    assert classification.active_particle_indices.size == 0
    assert classification.point_indices.size == 0
