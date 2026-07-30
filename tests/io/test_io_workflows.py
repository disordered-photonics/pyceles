from typing import Any, cast

import numpy as np
import pytest

from pyceles.core.fields import PlaneWave
from pyceles.core.particles import ParticleCollection, Sphere, spheres_from_arrays
from pyceles.io.workflows import load_simulation_h5, save_simulation_h5
from pyceles.postprocessing.nearfield import NearFieldSlice
from pyceles.simulation import (
    ResultRetention,
    Simulation,
    SimulationConfig,
)

pytestmark = [pytest.mark.filesystem, pytest.mark.hdf5]


def test_save_simulation_h5_serializes_one_explicit_polarization_channel(tmp_path):
    source = PlaneWave(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        polarization=(1.0 + 0j, 1.0j),
        polar_angle=0.2,
        azimuthal_angle=0.3,
    )
    sim = Simulation(
        SimulationConfig(
            wavelength=550.0,
            n_medium=1.0 + 0j,
            lmax=1,
            solver_method="direct",
            verbose=False,
        ),
        particles=[
            Sphere(
                position=(0.0, 0.0, 0.0),
                radius=100.0,
                refractive_index=1.5 + 0.0j,
            )
        ],
    )
    polarized = sim.run_polarizations(source)

    X, Z = np.meshgrid(np.linspace(-1.0, 1.0, 4), np.linspace(-1.0, 1.0, 3), indexing="xy")
    zero = np.zeros((*X.shape, 3), dtype=np.complex128)
    near = NearFieldSlice(
        axis_0=X,
        axis_1=Z,
        inside=np.zeros_like(X, dtype=bool),
        field_maps={"total": (zero, zero)},
        plane="y",
        plane_value=0.0,
        axis_0_label="x",
        axis_1_label="z",
    )

    te_path = save_simulation_h5(polarized.te, near, tmp_path / "te.h5")
    mixed_path = save_simulation_h5(polarized.mixed, near, tmp_path / "mixed.h5")

    import h5py

    with h5py.File(te_path, "r") as h5:
        assert "solution/coeffs" in h5
        assert "far_field" in h5
        assert "diagnostics/cross_sections" in h5
        assert "solver" not in h5["solution"].attrs
        assert "solution_basis" not in h5
        assert "far_field_basis" not in h5
        assert "power_basis" not in h5["diagnostics"]
        assert "unpolarized" not in h5["diagnostics"]

    with h5py.File(mixed_path, "r") as h5:
        assert "solution/coeffs" in h5
        assert "far_field" in h5
        assert "solver" not in h5["solution"].attrs

    loaded = load_simulation_h5(te_path)
    assert set(loaded) >= {"geometry", "solution", "far_field", "diagnostics"}


def test_no_scatterer_run_roundtrip_io_workflow(tmp_path):
    source = PlaneWave(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        polarization="TE",
        polar_angle=0.2,
        azimuthal_angle=0.3,
    )
    cfg = SimulationConfig(
        wavelength=550.0,
        n_medium=1.0 + 0j,
        lmax=3,
        solver_method="gmres",
        verbose=False,
    )
    sim = Simulation(
        cfg,
        particles=[],
    )
    run = sim.run(source)

    X, Z = np.meshgrid(np.linspace(-1.0, 1.0, 4), np.linspace(-1.0, 1.0, 3), indexing="xy")
    zero = np.zeros((*X.shape, 3), dtype=np.complex128)
    near = NearFieldSlice(
        axis_0=X,
        axis_1=Z,
        inside=np.zeros_like(X, dtype=bool),
        field_maps={"initial": (zero, zero), "scattered": (zero, zero), "total": (zero, zero)},
        plane="y",
        plane_value=0.0,
        axis_0_label="x",
        axis_1_label="z",
    )

    out = save_simulation_h5(run, near, tmp_path / "no_scatter.h5")
    loaded = cast(dict[str, Any], load_simulation_h5(out))
    geometry = cast(dict[str, Any], loaded["geometry"])
    solution = cast(dict[str, Any], loaded["solution"])

    geometry_particles = geometry["particles"]
    assert isinstance(geometry_particles, ParticleCollection)
    assert len(geometry_particles) == 0
    assert "positions" not in geometry
    assert np.asarray(solution["coeffs"]).shape[0] == 0
    assert "far_field" in loaded
    assert "diagnostics" in loaded


def test_save_simulation_h5_geometry_loads_particles(tmp_path):
    source = PlaneWave(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        polarization="TE",
        polar_angle=0.0,
        azimuthal_angle=0.0,
    )
    cfg = SimulationConfig(
        wavelength=550.0,
        n_medium=1.0 + 0j,
        lmax=2,
        solver_method="direct",
        verbose=False,
    )
    particles = spheres_from_arrays(
        positions=np.array([[0.0, 0.0, 0.0]], dtype=float),
        radii=np.array([120.0], dtype=float),
        refractive_indices=np.array([1.5 + 0.01j], dtype=np.complex128),
    )
    sim = Simulation(cfg, particles=particles)
    run = sim.run(source, include_farfield=False)

    X, Z = np.meshgrid(np.linspace(-1.0, 1.0, 3), np.linspace(-1.0, 1.0, 3), indexing="xy")
    zero = np.zeros((*X.shape, 3), dtype=np.complex128)
    near = NearFieldSlice(
        axis_0=X,
        axis_1=Z,
        inside=np.zeros_like(X, dtype=bool),
        field_maps={"initial": (zero, zero)},
        plane="y",
        plane_value=0.0,
        axis_0_label="x",
        axis_1_label="z",
    )
    out = save_simulation_h5(run, near, tmp_path / "particle_roundtrip.h5")
    loaded = cast(dict[str, Any], load_simulation_h5(out))
    geometry = cast(dict[str, Any], loaded["geometry"])
    geometry_particles = cast(ParticleCollection, geometry["particles"])

    assert isinstance(geometry_particles, ParticleCollection)
    assert len(geometry_particles) == 1
    assert geometry_particles[0] == particles[0]


def test_save_simulation_h5_accepts_minimal_result_retention(tmp_path):
    source = PlaneWave(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        polarization="TE",
        polar_angle=0.0,
        azimuthal_angle=0.0,
    )
    sim = Simulation(
        SimulationConfig(
            wavelength=550.0,
            n_medium=1.0 + 0j,
            lmax=1,
            solver_method="direct",
            verbose=False,
        ),
        particles=[
            Sphere(
                position=(0.0, 0.0, 0.0),
                radius=50.0,
                refractive_index=1.5 + 0.0j,
            )
        ],
    )
    run = sim.run(
        source,
        include_farfield=False,
        retention=ResultRetention.minimal(),
    )
    axis_0, axis_1 = np.meshgrid(
        np.linspace(-1.0, 1.0, 2),
        np.linspace(-1.0, 1.0, 2),
        indexing="xy",
    )
    zero = np.zeros((*axis_0.shape, 3), dtype=np.complex128)
    near = NearFieldSlice(
        axis_0=axis_0,
        axis_1=axis_1,
        inside=np.zeros_like(axis_0, dtype=bool),
        field_maps={"total": (zero, zero)},
        plane="y",
        plane_value=0.0,
        axis_0_label="x",
        axis_1_label="z",
    )

    out = save_simulation_h5(run, near, tmp_path / "minimal.h5")

    import h5py

    with h5py.File(out, "r") as h5:
        assert "solution/coeffs" in h5
        assert "solution/rhs" not in h5
        assert "solution/initial_coeffs" not in h5
        assert "solution/residual_history" not in h5
