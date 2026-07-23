from typing import Any, cast

import numpy as np
import pytest

from pyceles.core.fields import PlaneWave
from pyceles.core.particles import Sphere, spheres_from_arrays
from pyceles.io.workflows import load_simulation_h5, save_simulation_h5
from pyceles.linear.solvers import LinearSolveResult
from pyceles.postprocessing.farfield import FarFieldPatterns
from pyceles.postprocessing.nearfield import NearFieldSlice
from pyceles.simulation import (
    ResultRetention,
    Simulation,
    SimulationConfig,
    SimulationResult,
)

pytestmark = [pytest.mark.filesystem, pytest.mark.hdf5]


def _dummy_pwp(alpha: np.ndarray, beta: np.ndarray) -> dict:
    coeff = np.ones((alpha.size, beta.size), dtype=np.complex128)
    agrid = alpha[:, None]
    bgrid = beta[None, :]
    return {
        "alpha": alpha,
        "beta": beta,
        "kx": np.sin(bgrid) * np.cos(agrid),
        "ky": np.sin(bgrid) * np.sin(agrid),
        "kz": np.cos(bgrid) * np.ones_like(agrid),
        "coeff": coeff,
    }


def test_save_simulation_h5_writes_basis_and_diagnostics(tmp_path):
    alpha = np.linspace(0.0, 2.0 * np.pi, 13, endpoint=False)
    beta = np.linspace(0.0, np.pi, 21)
    pwp = _dummy_pwp(alpha, beta)

    ff = FarFieldPatterns(
        initial_te=pwp,
        initial_tm=pwp,
        scattered_te=pwp,
        scattered_tm=pwp,
        total_te=pwp,
        total_tm=pwp,
    )
    ff_basis = {"te": ff, "tm": ff}

    source = PlaneWave(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        polarization=(1.0 + 0j, 1.0j),
        polar_angle=0.2,
        azimuthal_angle=0.3,
    )
    cfg = SimulationConfig(
        wavelength=550.0,
        n_medium=1.0 + 0j,
        lmax=1,
        source=source,
        solver_method="direct",
        verbose=False,
    )

    run = SimulationResult(
        config=cfg,
        particles=spheres_from_arrays(
            positions=np.array([[0.0, 0.0, 0.0]], dtype=float),
            radii=np.array([100.0], dtype=float),
            refractive_indices=np.array([1.5 + 0.0j], dtype=np.complex128),
        ),
        k=2.0 * np.pi / 550.0,
        k0=2.0 * np.pi / 550.0,
        coeffs=np.ones((1, 6), dtype=np.complex128),
        rhs=np.ones((1, 6), dtype=np.complex128),
        initial_coeffs=np.ones((1, 6), dtype=np.complex128),
        initial_coeffs_basis={
            "te": np.ones((1, 6), dtype=np.complex128),
            "tm": np.ones((1, 6), dtype=np.complex128),
        },
        coeffs_basis={
            "te": np.ones((1, 6), dtype=np.complex128),
            "tm": np.ones((1, 6), dtype=np.complex128),
        },
        solver_result=LinearSolveResult(
            x=np.ones((6, 2), dtype=np.complex128),
            info=np.array([0, 0], dtype=int),
            residual_norm=np.array([1e-6, 1e-6], dtype=float),
            relative_residual=np.array([1e-5, 1e-5], dtype=float),
            iterations=np.array([5, 5], dtype=int),
            method="gmres",
            residual_history=[np.array([1.0, 0.5], dtype=float), np.array([1.0, 0.5], dtype=float)],
            rhs_count=2,
        ),
        solver_result_basis=None,
        farfield=ff,
        farfield_basis=ff_basis,
        power={"T": 1.0, "R": 0.0},
        power_basis={"te": {"T": 1.0}, "tm": {"T": 1.0}},
        cross_sections={"C_sca": 1.0, "C_ext": 2.0, "C_abs": 1.0},
        cross_sections_basis={"te": {"C_sca": 1.0}, "tm": {"C_sca": 1.0}},
        unpolarized={"cross_sections": {"C_sca": 1.0}},
        decomposition_forward={"P_total": 1.0},
        decomposition_backward={"P_total": 0.0},
        decomposition_forward_basis={"te": {"P_total": 1.0}, "tm": {"P_total": 1.0}},
        decomposition_backward_basis={"te": {"P_total": 0.0}, "tm": {"P_total": 0.0}},
        polarization_jones=(1.0 + 0j, 1.0j),
        compute_dtype="complex128",
        accum_dtype="complex128",
    )

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

    out = save_simulation_h5(run, near, tmp_path / "run_full.h5")

    import h5py

    with h5py.File(out, "r") as h5:
        assert "solution_basis" in h5
        assert "solution_basis/te" in h5
        assert "far_field_basis/te" in h5
        assert "far_field_basis/tm" in h5
        assert "diagnostics" in h5
        assert "cross_sections" in h5["diagnostics"]
        assert "power_basis" in h5["diagnostics"]
        assert "unpolarized" in h5["diagnostics"]

    loaded = load_simulation_h5(out)
    assert "geometry" in loaded
    assert "solution" in loaded
    assert "far_field" in loaded
    assert "diagnostics" in loaded
    assert "solution_basis" in loaded
    assert "far_field_basis" in loaded


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
        source=source,
        solver_method="gmres",
        verbose=False,
    )
    sim = Simulation(
        cfg,
        particles=[],
    )
    run = sim.run()

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

    assert geometry["particles"] == tuple()
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
        source=source,
        solver_method="direct",
        verbose=False,
    )
    particles = spheres_from_arrays(
        positions=np.array([[0.0, 0.0, 0.0]], dtype=float),
        radii=np.array([120.0], dtype=float),
        refractive_indices=np.array([1.5 + 0.01j], dtype=np.complex128),
    )
    sim = Simulation(cfg, particles=particles)
    run = sim.run(include_farfield=False)

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
    geometry_particles = cast(tuple[Sphere, ...], geometry["particles"])

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
            source=source,
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
