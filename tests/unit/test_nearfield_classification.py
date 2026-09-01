from __future__ import annotations

from typing import Any, cast

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


def test_periodic_internal_classifier_wraps_reference_cell_images() -> None:
    from pyceles.postprocessing.nearfield.classification import (
        _classify_periodic_internal_points_reference,
        classify_periodic_internal_points,
    )

    particles = ParticleCollection.from_archetypes(
        positions=np.asarray([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
        archetypes=[Sphere(position=(0.0, 0.0, 0.0), radius=0.6, refractive_index=1.5 + 0.0j)],
        archetype_indices=np.zeros(2, dtype=np.uint8),
    )
    points = np.asarray(
        [
            [10.2, 0.0, 0.0],
            [-6.8, 0.0, 0.0],
            [1.5, 0.0, 0.0],
        ]
    )
    kwargs = dict(
        points=points,
        particles=particles,
        lattice_ax=10.0,
        lattice_ay=9.0,
        k_parallel=np.asarray([0.13, -0.07]),
        n_medium=1.0 + 0.0j,
    )

    actual = classify_periodic_internal_points(**cast(Any, kwargs))
    expected = _classify_periodic_internal_points_reference(**cast(Any, kwargs))

    np.testing.assert_array_equal(actual[0].inside_any, expected[0].inside_any)
    np.testing.assert_array_equal(
        actual[0].active_particle_indices,
        expected[0].active_particle_indices,
    )
    np.testing.assert_array_equal(actual[0].point_offsets, expected[0].point_offsets)
    np.testing.assert_array_equal(actual[0].point_indices, expected[0].point_indices)
    np.testing.assert_allclose(actual[1], expected[1])
    np.testing.assert_allclose(actual[2], expected[2])


def test_periodic_internal_classifier_falls_back_for_large_particles() -> None:
    from pyceles.postprocessing.nearfield.classification import (
        _classify_periodic_internal_points_tree,
    )

    particles = ParticleCollection.from_archetypes(
        positions=np.asarray([[0.0, 0.0, 0.0]]),
        archetypes=[Sphere(position=(0.0, 0.0, 0.0), radius=5.1, refractive_index=1.5 + 0.0j)],
        archetype_indices=np.zeros(1, dtype=np.uint8),
    )

    assert (
        _classify_periodic_internal_points_tree(
            points=np.asarray([[0.0, 0.0, 0.0]]),
            particles=particles,
            lattice_ax=10.0,
            lattice_ay=12.0,
            k_parallel=np.zeros(2),
            n_medium=1.0 + 0.0j,
        )
        is None
    )
