from __future__ import annotations

import numpy as np
import pytest

from pyceles.core.fields import PlaneWave
from pyceles.core.operators import (
    MLFMMCouplingOperator,
    MLFMMOptions,
    assemble_dense_A_numpy,
    prepare_matvec,
)
from pyceles.core.particles import Particle, spheres_from_arrays
from pyceles.simulation import Simulation, SimulationConfig


def _mlfmm_single_level_problem() -> tuple[int, float, tuple[Particle, ...]]:
    lmax = 2
    k = 2 * np.pi / 550.0
    positions = np.array(
        [
            [-90.0, -90.0, -90.0],
            [-72.0, -74.0, -88.0],
            [-90.0, -90.0, 90.0],
            [-72.0, -88.0, 74.0],
            [90.0, 90.0, -90.0],
            [72.0, 88.0, -74.0],
            [90.0, 90.0, 90.0],
            [88.0, 72.0, 74.0],
        ],
        dtype=float,
    )
    particles = tuple(
        spheres_from_arrays(
            positions=positions,
            radii=np.full((positions.shape[0],), 11.0, dtype=float),
            refractive_indices=np.full((positions.shape[0],), 1.59 + 0.0j, dtype=np.complex128),
        )
    )
    return lmax, k, particles


def _plane_wave() -> PlaneWave:
    return PlaneWave(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        polarization="TE",
        polar_angle=0.0,
        azimuthal_angle=0.0,
        amplitude=1.0,
    )


def test_prepare_matvec_accepts_mlfmm_coupling_backend() -> None:
    lmax, k, particles = _mlfmm_single_level_problem()
    prepared = prepare_matvec(
        lmax=lmax,
        k=k,
        particles=particles,
        n_medium=1.0 + 0j,
        radial_lut_dr=1.0,
        cache_translation_blocks=False,
        coupling_backend="mlfmm",
        mlfmm_options=MLFMMOptions(max_leaf_particles=1, max_depth=2),
    )

    assert isinstance(prepared.coupling, MLFMMCouplingOperator)
    assert prepared.coupling.resolved_plan.stage == "single_level"


def test_dense_assembly_rejects_true_mlfmm_coupling_backend() -> None:
    lmax, k, particles = _mlfmm_single_level_problem()
    prepared = prepare_matvec(
        lmax=lmax,
        k=k,
        particles=particles,
        n_medium=1.0 + 0j,
        radial_lut_dr=1.0,
        cache_translation_blocks=False,
        coupling_backend="mlfmm",
        mlfmm_options=MLFMMOptions(max_leaf_particles=1, max_depth=2),
    )

    with pytest.raises(TypeError, match="PairwiseCouplingOperator"):
        assemble_dense_A_numpy(prepared, show_progress=False, use_cache=False, store_blocks=False)


def test_simulation_iterative_solve_runs_with_mlfmm_coupling() -> None:
    lmax, _, particles = _mlfmm_single_level_problem()
    cfg = SimulationConfig(
        wavelength=550.0,
        n_medium=1.0 + 0j,
        lmax=lmax,
        source=_plane_wave(),
        polar_angles=np.linspace(0.0, np.pi, 41),
        azimuthal_angles=np.linspace(0.0, 2.0 * np.pi, 40, endpoint=False),
        solver_method="gmres",
        solver_rtol=1e-4,
        solver_restart=20,
        solver_maxiter=80,
        coupling_backend="mlfmm",
        mlfmm_options=MLFMMOptions(max_leaf_particles=1, max_depth=2),
        verbose=False,
    )
    sim = Simulation(cfg, particles=particles)
    run = sim.run(include_farfield=False)

    assert run.solver_result.info == 0
    assert sim._prepared_operator_cache is not None
    assert isinstance(sim._prepared_operator_cache.coupling, MLFMMCouplingOperator)


def test_simulation_direct_solve_rejects_true_mlfmm_coupling() -> None:
    lmax, _, particles = _mlfmm_single_level_problem()
    cfg = SimulationConfig(
        wavelength=550.0,
        n_medium=1.0 + 0j,
        lmax=lmax,
        source=_plane_wave(),
        polar_angles=np.linspace(0.0, np.pi, 41),
        azimuthal_angles=np.linspace(0.0, 2.0 * np.pi, 40, endpoint=False),
        solver_method="direct",
        coupling_backend="mlfmm",
        mlfmm_options=MLFMMOptions(max_leaf_particles=1, max_depth=2),
        verbose=False,
    )
    sim = Simulation(cfg, particles=particles)

    with pytest.raises(NotImplementedError, match="pairwise coupling backend"):
        sim.run(include_farfield=False)


def test_prepare_matvec_rejects_numpy_mlfmm_complex64_operator_dtype() -> None:
    lmax, k, particles = _mlfmm_single_level_problem()
    with pytest.raises(ValueError, match="operator_dtype=complex128"):
        prepare_matvec(
            lmax=lmax,
            k=k,
            particles=particles,
            n_medium=1.0 + 0j,
            radial_lut_dr=1.0,
            cache_translation_blocks=False,
            coupling_backend="mlfmm",
            mlfmm_options=MLFMMOptions(max_leaf_particles=1, max_depth=2),
            operator_dtype=np.complex64,
        )
