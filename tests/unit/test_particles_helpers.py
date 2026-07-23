from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from pyceles.core.particles import (
    LayeredSphere,
    Particle,
    ParticleCollection,
    PECSphere,
    Sphere,
    Spheroid,
    layered_spheres_from_arrays,
    particle_contains_points,
    particle_intrinsic_t_signature,
    particle_t_signature,
    pec_spheres_from_arrays,
    spheres_from_arrays,
    spheroids_from_arrays,
)

pytestmark = pytest.mark.api_contract


def test_particle_contains_points_handles_supported_particle_families():
    pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    sphere = Sphere(position=(0.0, 0.0, 0.0), radius=2.0, refractive_index=1.5 + 0j)
    layered = LayeredSphere(
        position=(0.0, 0.0, 0.0),
        layer_radii=(1.0, 2.5),
        layer_refractive_indices=(1.4 + 0j, 1.5 + 0j),
    )
    spheroid = Spheroid(
        position=(0.0, 0.0, 0.0),
        equatorial_radius=1.0,
        polar_radius=3.0,
        refractive_index=1.5 + 0j,
        euler_angles=(0.0, np.pi / 2.0, 0.0),
    )
    assert particle_contains_points(sphere, pts).tolist() == [True, True, False]
    assert particle_contains_points(
        PECSphere(position=(0.0, 0.0, 0.0), radius=2.0),
        pts,
    ).tolist() == [True, True, False]
    assert particle_contains_points(layered, pts).tolist() == [True, True, False]
    assert particle_contains_points(spheroid, pts).tolist() == [True, True, False]


def test_particle_base_api_and_representation_defaults():
    particle = Particle(position=(1.0, 2.0, 3.0))
    np.testing.assert_allclose(
        particle.pos_array(dtype=np.float32), np.array([1.0, 2.0, 3.0], dtype=np.float32)
    )
    assert particle.kind == "particle"
    assert particle.t_operator_representation == "dense"
    with pytest.raises(NotImplementedError, match="must implement circumscribing_radius"):
        particle.circumscribing_radius()

    sphere = Sphere(position=(0.0, 0.0, 0.0), radius=1.0, refractive_index=1.5 + 0j)
    layered = LayeredSphere(
        position=(0.0, 0.0, 0.0),
        layer_radii=(1.0, 2.0),
        layer_refractive_indices=(1.3 + 0j, 1.4 + 0j),
    )
    spheroid = Spheroid(
        position=(0.0, 0.0, 0.0),
        equatorial_radius=1.0,
        polar_radius=2.0,
        refractive_index=1.5 + 0j,
    )
    assert sphere.t_operator_representation == "diagonal"
    assert PECSphere(position=(0.0, 0.0, 0.0), radius=1.0).t_operator_representation == "diagonal"
    assert layered.t_operator_representation == "diagonal"
    assert spheroid.t_operator_representation == "axisymmetric"


def test_particle_signatures_ignore_position_and_intrinsic_signature_ignores_orientation():
    sphere0 = Sphere(position=(0.0, 0.0, 0.0), radius=2.0, refractive_index=1.5 + 0j)
    sphere1 = Sphere(position=(5.0, -3.0, 2.0), radius=2.0, refractive_index=1.5 + 0j)
    pec0 = PECSphere(position=(0.0, 0.0, 0.0), radius=2.0)
    pec1 = PECSphere(position=(5.0, -3.0, 2.0), radius=2.0)
    spheroid0 = Spheroid(
        position=(0.0, 0.0, 0.0),
        equatorial_radius=2.0,
        polar_radius=3.0,
        refractive_index=1.5 + 0j,
        euler_angles=(0.1, 0.2, 0.3),
    )
    spheroid1 = Spheroid(
        position=(1.0, 2.0, 3.0),
        equatorial_radius=2.0,
        polar_radius=3.0,
        refractive_index=1.5 + 0j,
        euler_angles=(0.4, -0.2, 0.1),
    )
    assert particle_t_signature(sphere0) == particle_t_signature(sphere1)
    assert particle_t_signature(pec0) == particle_t_signature(pec1)
    assert particle_t_signature(spheroid0) != particle_t_signature(spheroid1)
    assert particle_intrinsic_t_signature(spheroid0) == particle_intrinsic_t_signature(spheroid1)


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (
            lambda: PECSphere(
                position=(0.0, 0.0, 0.0),
                radius=0.0,
            ),
            "radius must be positive",
        ),
        (
            lambda: LayeredSphere(
                position=(0.0, 0.0, 0.0),
                layer_radii=(),
                layer_refractive_indices=(),
            ),
            "non-empty",
        ),
        (
            lambda: LayeredSphere(
                position=(0.0, 0.0, 0.0),
                layer_radii=(1.0,),
                layer_refractive_indices=(1.4 + 0j, 1.5 + 0j),
            ),
            "same length",
        ),
        (
            lambda: LayeredSphere(
                position=(0.0, 0.0, 0.0),
                layer_radii=(0.0, 1.0),
                layer_refractive_indices=(1.4 + 0j, 1.5 + 0j),
            ),
            "positive",
        ),
        (
            lambda: LayeredSphere(
                position=(0.0, 0.0, 0.0),
                layer_radii=(2.0, 1.0),
                layer_refractive_indices=(1.4 + 0j, 1.5 + 0j),
            ),
            "strictly increasing",
        ),
        (
            lambda: Spheroid(
                position=(0.0, 0.0, 0.0),
                equatorial_radius=0.0,
                polar_radius=1.0,
                refractive_index=1.5 + 0j,
            ),
            "equatorial_radius must be positive",
        ),
        (
            lambda: Spheroid(
                position=(0.0, 0.0, 0.0),
                equatorial_radius=1.0,
                polar_radius=0.0,
                refractive_index=1.5 + 0j,
            ),
            "polar_radius must be positive",
        ),
    ],
)
def test_particle_dataclass_validation(factory, match):
    with pytest.raises(ValueError, match=match):
        factory()


def test_particle_constructors_broadcast_inputs_and_concatenate_batches():
    initial = ParticleCollection.from_particles(
        [Sphere(position=(9.0, 9.0, 9.0), radius=1.0, refractive_index=1.4 + 0j)]
    )
    positions = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=float)
    out = ParticleCollection.concatenate(
        initial,
        spheres_from_arrays(
            positions=positions,
            radii=np.array([1.0, 2.0], dtype=float),
            refractive_indices=1.6 + 0.0j,
        ),
    )
    assert len(out) == 3

    out = ParticleCollection.concatenate(
        out,
        pec_spheres_from_arrays(
            positions=positions,
            radii=np.array([1.5, 2.5], dtype=float),
        ),
    )
    assert len(out) == 5
    assert isinstance(out[-1], PECSphere)
    assert out[-1].radius == 2.5

    out = ParticleCollection.concatenate(
        out,
        layered_spheres_from_arrays(
            positions=positions,
            layer_radii=np.array([0.5, 1.5], dtype=float),
            layer_refractive_indices=np.array([1.3 + 0j, 1.5 + 0j], dtype=np.complex128),
        ),
    )
    assert len(out) == 7
    assert isinstance(out[-1], LayeredSphere)
    assert out[-1].layer_radii == (0.5, 1.5)

    out = ParticleCollection.concatenate(
        out,
        spheroids_from_arrays(
            positions=positions,
            equatorial_radii=1.0,
            polar_radii=np.array([2.0, 3.0], dtype=float),
            refractive_indices=1.7 + 0.0j,
            euler_angles=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=float),
        ),
    )
    assert len(out) == 9
    assert isinstance(out[-1], Spheroid)
    spheroid_prev = cast(Spheroid, out[-2])
    assert spheroid_prev.euler_angles == (0.1, 0.2, 0.3)
    assert out[-1].euler_angles == (0.4, 0.5, 0.6)


def test_array_particle_collection_owns_read_only_inputs_and_materializes_lazily():
    positions = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=float)
    radii = np.array([10.0, 20.0], dtype=float)
    refractive_indices = np.array([1.5 + 0.01j, 1.7 + 0.02j])
    particles = spheres_from_arrays(
        positions=positions,
        radii=radii,
        refractive_indices=refractive_indices,
    )

    positions[:] = -1.0
    radii[:] = -1.0
    refractive_indices[:] = 9.0

    assert isinstance(particles, ParticleCollection)
    np.testing.assert_array_equal(
        particles.positions,
        np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]),
    )
    np.testing.assert_array_equal(particles.circumscribing_radii, [10.0, 20.0])
    assert particles[1] == Sphere(
        position=(3.0, 4.0, 5.0),
        radius=20.0,
        refractive_index=1.7 + 0.02j,
    )
    with pytest.raises(ValueError, match="read-only"):
        particles.positions[0, 0] = 99.0
    with pytest.raises(ValueError, match="read-only"):
        particles.circumscribing_radii[0] = 99.0


@pytest.mark.parametrize(
    ("builder", "kwargs", "match"),
    [
        (
            spheres_from_arrays,
            {
                "positions": np.array([[np.nan, 0.0, 0.0]], dtype=float),
                "radii": np.array([1.0], dtype=float),
                "refractive_indices": 1.5 + 0j,
            },
            "finite values",
        ),
        (
            spheres_from_arrays,
            {
                "positions": np.array([[0.0, 0.0]], dtype=float),
                "radii": np.array([1.0], dtype=float),
                "refractive_indices": 1.5 + 0j,
            },
            "shape \\(N, 3\\)",
        ),
        (
            spheres_from_arrays,
            {
                "positions": np.array([[0.0, 0.0, 0.0]], dtype=float),
                "radii": np.array([-1.0], dtype=float),
                "refractive_indices": 1.5 + 0j,
            },
            "strictly positive",
        ),
        (
            spheres_from_arrays,
            {
                "positions": np.array([[0.0, 0.0, 0.0]], dtype=float),
                "radii": np.array([1.0, 2.0], dtype=float),
                "refractive_indices": 1.5 + 0j,
            },
            "`radii` length",
        ),
        (
            spheres_from_arrays,
            {
                "positions": np.array([[0.0, 0.0, 0.0]], dtype=float),
                "radii": np.array([1.0], dtype=float),
                "refractive_indices": 0.0 + 0j,
            },
            "Real part of `refractive_indices` must be strictly positive",
        ),
        (
            pec_spheres_from_arrays,
            {
                "positions": np.array([[0.0, 0.0, 0.0]], dtype=float),
                "radii": np.array([1.0, 2.0], dtype=float),
            },
            "`radii` length",
        ),
        (
            pec_spheres_from_arrays,
            {
                "positions": np.array([[0.0, 0.0, 0.0]], dtype=float),
                "radii": np.array([0.0], dtype=float),
            },
            "strictly positive",
        ),
        (
            layered_spheres_from_arrays,
            {
                "positions": np.array([[0.0, 0.0, 0.0]], dtype=float),
                "layer_radii": np.array([], dtype=float),
                "layer_refractive_indices": np.array([1.4 + 0j], dtype=np.complex128),
            },
            "at least one layer",
        ),
        (
            layered_spheres_from_arrays,
            {
                "positions": np.array([[0.0, 0.0, 0.0]], dtype=float),
                "layer_radii": np.array([[1.0, 2.0]], dtype=float),
                "layer_refractive_indices": np.array([[1.4 + 0j]], dtype=np.complex128),
            },
            "must share shape",
        ),
        (
            layered_spheres_from_arrays,
            {
                "positions": np.array([[0.0, 0.0, 0.0]], dtype=float),
                "layer_radii": np.array([[1.0, np.nan]], dtype=float),
                "layer_refractive_indices": np.array([[1.4 + 0j, 1.5 + 0j]], dtype=np.complex128),
            },
            "strictly positive",
        ),
        (
            layered_spheres_from_arrays,
            {
                "positions": np.array([[0.0, 0.0, 0.0]], dtype=float),
                "layer_radii": np.array([[1.0, 2.0]], dtype=float),
                "layer_refractive_indices": np.array(
                    [[np.nan + 0j, 1.5 + 0j]], dtype=np.complex128
                ),
            },
            "finite values",
        ),
        (
            layered_spheres_from_arrays,
            {
                "positions": np.array([[0.0, 0.0, 0.0]], dtype=float),
                "layer_radii": np.array([[1.0, 2.0]], dtype=float),
                "layer_refractive_indices": np.array([[0.0 + 0j, 1.5 + 0j]], dtype=np.complex128),
            },
            "strictly positive",
        ),
        (
            layered_spheres_from_arrays,
            {
                "positions": np.array([[0.0, 0.0, 0.0]], dtype=float),
                "layer_radii": np.array([[2.0, 1.0]], dtype=float),
                "layer_refractive_indices": np.array([[1.4 + 0j, 1.5 + 0j]], dtype=np.complex128),
            },
            "strictly increasing",
        ),
        (
            spheroids_from_arrays,
            {
                "positions": np.array([[0.0, 0.0, 0.0]], dtype=float),
                "equatorial_radii": np.array([1.0, 2.0], dtype=float),
                "polar_radii": 2.0,
                "refractive_indices": 1.5 + 0j,
            },
            "`equatorial_radii` length",
        ),
        (
            spheroids_from_arrays,
            {
                "positions": np.array([[0.0, 0.0, 0.0]], dtype=float),
                "equatorial_radii": 1.0,
                "polar_radii": np.array([np.nan], dtype=float),
                "refractive_indices": 1.5 + 0j,
            },
            "finite strictly positive values",
        ),
        (
            spheroids_from_arrays,
            {
                "positions": np.array([[0.0, 0.0, 0.0]], dtype=float),
                "equatorial_radii": 1.0,
                "polar_radii": 2.0,
                "refractive_indices": 1.5 + 0j,
                "euler_angles": np.zeros((2, 2), dtype=float),
            },
            "`euler_angles` must have shape",
        ),
        (
            spheroids_from_arrays,
            {
                "positions": np.array([[0.0, 0.0, 0.0]], dtype=float),
                "equatorial_radii": 1.0,
                "polar_radii": 2.0,
                "refractive_indices": 1.5 + 0j,
                "euler_angles": np.array([[0.0, np.nan, 0.0]], dtype=float),
            },
            "finite values",
        ),
    ],
)
def test_particle_constructors_validate_public_array_inputs(builder, kwargs, match):
    fn = cast(Any, builder)
    with pytest.raises(ValueError, match=match):
        fn(**kwargs)
