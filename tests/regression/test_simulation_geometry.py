from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from pyceles.core.angular import (
    is_uniform_periodic_azimuth,
    uniform_periodic_azimuth_grid,
    uniform_polar_grid,
)
from pyceles.core.particles import LayeredSphere, Sphere
from pyceles.simulation import Simulation, SimulationConfig


def test_simulation_rejects_overlapping_circumscribing_spheres_by_default() -> None:
    cfg = SimulationConfig()
    positions = np.array([[0.0, 0.0, 0.0], [150.0, 0.0, 0.0]], dtype=float)
    radii = np.array([100.0, 100.0], dtype=float)

    with pytest.raises(ValueError, match="circumscribing spheres overlap"):
        Simulation(cfg, positions=positions, radii=radii, n_particle=1.5 + 0j)


def test_simulation_can_skip_overlap_check_when_requested() -> None:
    cfg = SimulationConfig(check_circumscribing_sphere_overlap=False)
    positions = np.array([[0.0, 0.0, 0.0], [150.0, 0.0, 0.0]], dtype=float)
    radii = np.array([100.0, 100.0], dtype=float)

    sim = Simulation(cfg, positions=positions, radii=radii, n_particle=1.5 + 0j)
    np.testing.assert_allclose(sim.positions, positions, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(sim.radii, radii, rtol=0.0, atol=0.0)


def test_overlap_tolerance_allows_small_roundoff_level_penetration() -> None:
    positions = np.array([[0.0, 0.0, 0.0], [199.999999999, 0.0, 0.0]], dtype=float)
    radii = np.array([100.0, 100.0], dtype=float)

    with pytest.raises(ValueError):
        Simulation(
            SimulationConfig(circumscribing_sphere_overlap_atol=0.0),
            positions=positions,
            radii=radii,
            n_particle=1.5 + 0j,
        )

    sim = Simulation(
        SimulationConfig(circumscribing_sphere_overlap_atol=1e-8),
        positions=positions,
        radii=radii,
        n_particle=1.5 + 0j,
    )
    np.testing.assert_allclose(sim.radii, radii, rtol=0.0, atol=0.0)


def test_simulation_accepts_empty_particle_geometry() -> None:
    cfg = SimulationConfig(check_circumscribing_sphere_overlap=True, verbose=False)
    sim = Simulation(
        cfg,
        positions=np.zeros((0, 3), dtype=float),
        radii=np.zeros((0,), dtype=float),
        n_particle=np.zeros((0,), dtype=np.complex128),
    )
    assert sim.n_particles == 0
    assert sim.positions.size == 0
    assert sim.radii.size == 0
    assert sim.n_particle.size == 0


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
    sim = Simulation.from_particles(cfg, particles=particles)
    assert sim.n_particles == 2
    assert sim.positions.shape == (2, 3)
    np.testing.assert_allclose(sim.radii, np.array([50.0, 80.0], dtype=float), rtol=0.0, atol=0.0)
    assert sim.particles is not None
    assert len(sim.particles) == 2


def test_simulation_rejects_none_geometry_inputs() -> None:
    cfg = SimulationConfig(verbose=False)
    with pytest.raises(ValueError, match="`positions` cannot be None"):
        Simulation(
            cfg,
            positions=cast(Any, None),
            radii=np.zeros((0,), dtype=float),
            n_particle=np.zeros((0,), dtype=np.complex128),
        )
    with pytest.raises(ValueError, match="`radii` cannot be None"):
        Simulation(
            cfg,
            positions=np.zeros((0, 3), dtype=float),
            radii=cast(Any, None),
            n_particle=np.zeros((0,), dtype=np.complex128),
        )
    with pytest.raises(ValueError, match="`n_particle` cannot be None"):
        Simulation(
            cfg,
            positions=np.zeros((0, 3), dtype=float),
            radii=np.zeros((0,), dtype=float),
            n_particle=cast(Any, None),
        )


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
