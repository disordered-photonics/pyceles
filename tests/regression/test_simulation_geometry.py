from __future__ import annotations

import numpy as np
import pytest

from pyceles.core.angular import (
    is_uniform_periodic_azimuth,
    uniform_periodic_azimuth_grid,
    uniform_polar_grid,
)
from pyceles.core.lattice import RectangularLattice2D
from pyceles.core.particles import (
    LayeredSphere,
    Sphere,
    spheres_from_arrays,
)
from pyceles.core.periodic import PeriodicSpec
from pyceles.simulation import Simulation, SimulationConfig
from pyceles.simulation.helpers import first_overlapping_circumscribing_pair


def test_simulation_rejects_overlapping_circumscribing_spheres_by_default() -> None:
    cfg = SimulationConfig()
    positions = np.array([[0.0, 0.0, 0.0], [150.0, 0.0, 0.0]], dtype=float)
    radii = np.array([100.0, 100.0], dtype=float)

    with pytest.raises(ValueError, match="circumscribing spheres overlap"):
        Simulation(
            cfg,
            particles=spheres_from_arrays(
                positions=positions,
                radii=radii,
                refractive_indices=1.5 + 0j,
            ),
        )


def test_simulation_can_skip_overlap_check_when_requested() -> None:
    cfg = SimulationConfig(check_circumscribing_sphere_overlap=False)
    positions = np.array([[0.0, 0.0, 0.0], [150.0, 0.0, 0.0]], dtype=float)
    radii = np.array([100.0, 100.0], dtype=float)

    sim = Simulation(
        cfg,
        particles=spheres_from_arrays(
            positions=positions,
            radii=radii,
            refractive_indices=1.5 + 0j,
        ),
    )
    np.testing.assert_allclose(sim.positions, positions, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(sim.circumscribing_radii, radii, rtol=0.0, atol=0.0)


def test_overlap_tolerance_allows_small_roundoff_level_penetration() -> None:
    positions = np.array([[0.0, 0.0, 0.0], [199.999999999, 0.0, 0.0]], dtype=float)
    radii = np.array([100.0, 100.0], dtype=float)

    with pytest.raises(ValueError):
        Simulation(
            SimulationConfig(circumscribing_sphere_overlap_atol=0.0),
            particles=spheres_from_arrays(
                positions=positions,
                radii=radii,
                refractive_indices=1.5 + 0j,
            ),
        )

    sim = Simulation(
        SimulationConfig(circumscribing_sphere_overlap_atol=1e-8),
        particles=spheres_from_arrays(
            positions=positions,
            radii=radii,
            refractive_indices=1.5 + 0j,
        ),
    )
    np.testing.assert_allclose(sim.circumscribing_radii, radii, rtol=0.0, atol=0.0)


def test_periodic_overlap_check_uses_xy_minimum_image_but_finite_z() -> None:
    lattice = RectangularLattice2D(100.0, 120.0)
    positions = np.array(
        [
            [5.0, 10.0, 0.0],
            [95.0, 10.0, 0.0],
            [5.0, 10.0, 30.0],
        ],
        dtype=float,
    )
    radii = np.array([6.0, 6.0, 6.0], dtype=float)

    overlap = first_overlapping_circumscribing_pair(
        positions,
        radii,
        lattice=lattice,
    )

    assert overlap is not None
    assert (overlap.particle_i, overlap.particle_j) == (0, 1)
    assert overlap.lattice_shift == (-1, 0)
    assert overlap.distance == pytest.approx(10.0)
    assert overlap.required_minimum == pytest.approx(12.0)


def test_periodic_overlap_check_keeps_z_nonperiodic() -> None:
    overlap = first_overlapping_circumscribing_pair(
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 30.0]], dtype=float),
        np.array([10.0, 10.0], dtype=float),
        lattice=RectangularLattice2D(100.0, 100.0),
    )

    assert overlap is None


def test_periodic_overlap_check_scales_to_large_sparse_cells() -> None:
    n_particles = 10_000
    positions = np.zeros((n_particles, 3), dtype=float)
    positions[:, 2] = 3.0 * np.arange(n_particles, dtype=float)
    particles = spheres_from_arrays(
        positions=positions,
        radii=np.ones((n_particles,), dtype=float),
        refractive_indices=1.5 + 0j,
    )
    config = SimulationConfig(
        periodic=PeriodicSpec(lattice=RectangularLattice2D(100.0, 100.0)),
        verbose=False,
    )

    sim = Simulation(config, particles=particles)

    assert sim.n_particles == n_particles


def test_simulation_accepts_empty_particle_geometry() -> None:
    cfg = SimulationConfig(check_circumscribing_sphere_overlap=True, verbose=False)
    sim = Simulation(cfg, particles=[])
    assert sim.n_particles == 0
    assert sim.positions.size == 0
    assert sim.circumscribing_radii.size == 0


def test_simulation_accepts_explicit_particle_descriptors() -> None:
    cfg = SimulationConfig(check_circumscribing_sphere_overlap=True, verbose=False)
    particles = [
        Sphere(position=(0.0, 0.0, 0.0), radius=50.0, refractive_index=1.5 + 0j),
        LayeredSphere(
            position=(200.0, 0.0, 0.0),
            layer_radii=(40.0, 80.0),
            layer_refractive_indices=(1.8 + 0j, 1.3 + 0.02j),
        ),
    ]
    sim = Simulation(cfg, particles=particles)
    assert sim.n_particles == 2
    assert sim.positions.shape == (2, 3)
    np.testing.assert_allclose(
        sim.circumscribing_radii,
        np.array([50.0, 80.0], dtype=float),
        rtol=0.0,
        atol=0.0,
    )
    assert sim.particles is not None
    assert len(sim.particles) == 2


def test_simulation_reuses_array_collection_as_canonical_geometry_owner() -> None:
    particles = spheres_from_arrays(
        positions=np.array([[0.0, 0.0, 0.0], [200.0, 0.0, 0.0]]),
        radii=np.array([50.0, 60.0]),
        refractive_indices=np.array([1.5 + 0j, 1.6 + 0j]),
    )
    sim = Simulation(
        SimulationConfig(
            check_circumscribing_sphere_overlap=False,
            verbose=False,
        ),
        particles=particles,
    )

    assert sim.particles is particles
    assert np.shares_memory(sim.positions, particles.positions)
    assert np.shares_memory(
        sim.circumscribing_radii,
        particles.circumscribing_radii,
    )


def test_simulation_geometry_and_config_cannot_invalidate_prepared_caches() -> None:
    cfg = SimulationConfig(verbose=False)
    particles = [
        Sphere(position=(0.0, 0.0, 0.0), radius=50.0, refractive_index=1.5 + 0j),
        Sphere(position=(200.0, 0.0, 0.0), radius=50.0, refractive_index=1.5 + 0j),
    ]
    sim = Simulation(cfg, particles=particles)

    assert not sim.positions.flags.writeable
    assert not sim.circumscribing_radii.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        sim.positions[1, 0] = 260.0
    with pytest.raises(ValueError, match="WRITEABLE"):
        sim.positions.flags.writeable = True
    with pytest.raises(ValueError, match="read-only"):
        sim.circumscribing_radii[0] = 60.0

    for name, value in (
        ("config", SimulationConfig(verbose=False)),
        ("particles", tuple(particles)),
        ("positions", np.zeros((2, 3), dtype=float)),
        ("circumscribing_radii", np.ones((2,), dtype=float)),
    ):
        with pytest.raises(AttributeError):
            setattr(sim, name, value)


def test_simulation_requires_explicit_particles_argument() -> None:
    cfg = SimulationConfig(verbose=False)
    with pytest.raises(TypeError, match="missing 1 required keyword-only argument"):
        Simulation(cfg)  # type: ignore[call-arg]


def test_simulation_rejects_non_particle_entries() -> None:
    cfg = SimulationConfig(verbose=False)
    with pytest.raises(TypeError, match="Particle instances"):
        Simulation(cfg, particles=[object()])  # type: ignore[list-item]


def test_default_azimuth_grid_is_uniform_periodic_open_interval() -> None:
    cfg = SimulationConfig(verbose=False)
    alpha = np.asarray(cfg.azimuthal_angles, dtype=float)
    assert np.isclose(alpha[0], 0.0, rtol=0.0, atol=0.0)
    assert alpha[-1] < 2.0 * np.pi
    assert is_uniform_periodic_azimuth(alpha)


def test_warns_on_redundant_periodic_azimuth_endpoint() -> None:
    with pytest.warns(UserWarning, match="endpoint=False"):
        _ = SimulationConfig(
            azimuthal_angles=np.linspace(0.0, 2.0 * np.pi, 201, endpoint=True),
            verbose=False,
        )


def test_uniform_grid_helpers_return_expected_conventions() -> None:
    beta = uniform_polar_grid(11)
    alpha = uniform_periodic_azimuth_grid(12)
    assert beta.shape == (11,)
    assert alpha.shape == (12,)
    assert np.isclose(beta[0], 0.0)
    assert np.isclose(beta[-1], np.pi)
    assert np.isclose(alpha[0], 0.0)
    assert alpha[-1] < 2.0 * np.pi
    assert is_uniform_periodic_azimuth(alpha)
