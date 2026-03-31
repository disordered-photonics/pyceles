from __future__ import annotations

import numpy as np
import pytest

from pyceles.core.fields import PlaneWave
from pyceles.core.operators import (
    MLFMMCouplingOperator,
    MLFMMOptions,
    prepare_matvec,
)
from pyceles.core.particles import Particle, spheres_from_arrays
from pyceles.simulation import Simulation, SimulationConfig
from pyceles.simulation.solve import solve_sources_core


def _mlfmm_single_level_problem() -> tuple[int, float, tuple[Particle, ...]]:
    lmax = 1
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


def _mlfmm_multilevel_problem() -> tuple[int, float, tuple[Particle, ...]]:
    lmax = 1
    k = 2 * np.pi / 550.0
    gx, gy, gz = np.meshgrid(np.arange(8), np.arange(8), np.arange(8), indexing="ij")
    # Keep the fixture geometry compact so far-order table setup stays fast
    # while preserving a true multilevel partition with max_leaf_particles=1.
    positions = (40.0 * np.stack((gx.ravel(), gy.ravel(), gz.ravel()), axis=1)[:9]).astype(float)
    particles = tuple(
        spheres_from_arrays(
            positions=positions,
            radii=np.full((positions.shape[0],), 8.0, dtype=float),
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


@pytest.fixture(scope="module")
def prepared_multilevel_mlfmm():
    lmax, k, particles = _mlfmm_multilevel_problem()
    return prepare_matvec(
        lmax=lmax,
        k=k,
        particles=particles,
        n_medium=1.0 + 0j,
        radial_lut_dr=1.0,
        cache_translation_blocks=False,
        coupling_backend="mlfmm",
        mlfmm_options=MLFMMOptions(max_leaf_particles=1, max_depth=3),
        operator_dtype=np.complex128,
    )


def test_simulation_direct_solve_rejects_true_mlfmm_coupling(prepared_multilevel_mlfmm) -> None:
    lmax, _, particles = _mlfmm_multilevel_problem()
    cfg = SimulationConfig(
        wavelength=550.0,
        n_medium=1.0 + 0j,
        lmax=lmax,
        source=_plane_wave(),
        polar_angles=np.linspace(0.0, np.pi, 5),
        azimuthal_angles=np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False),
        solver_method="direct",
        coupling_backend="mlfmm",
        mlfmm_options=MLFMMOptions(max_leaf_particles=1, max_depth=2),
        verbose=False,
    )
    sim = Simulation(cfg, particles=particles)
    sim._prepared_operator_cache = prepared_multilevel_mlfmm
    sim._prepared_operator_dtype = np.dtype(np.complex128)

    with pytest.raises(NotImplementedError, match="pairwise coupling backend"):
        solve_sources_core(sim, {"source": _plane_wave()})


def test_prepare_matvec_accepts_numpy_mlfmm_complex64_operator_dtype() -> None:
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
        operator_dtype=np.complex64,
    )
    assert isinstance(prepared.coupling, MLFMMCouplingOperator)
    coupling = prepared.coupling
    assert coupling.dtype == np.dtype(np.complex64)
    assert coupling.near_dtype == np.dtype(np.complex64)
    assert coupling.far_dtype == np.dtype(np.complex128)


@pytest.mark.parametrize(
    ("fixture_name", "expected_stage"),
    [
        ("prepared_multilevel_mlfmm", "multilevel"),
    ],
)
def test_prepare_matvec_mlfmm_internal_arrays_stay_complex128(
    fixture_name: str,
    expected_stage: str,
    request: pytest.FixtureRequest,
) -> None:
    prepared = request.getfixturevalue(fixture_name)

    assert isinstance(prepared.coupling, MLFMMCouplingOperator)
    coupling = prepared.coupling
    assert coupling.resolved_plan.stage == expected_stage
    assert coupling.dtype == np.dtype(np.complex128)
    assert coupling.radial_lut.dtype == np.dtype(np.complex128)

    if coupling.single_level is not None:
        assert coupling.single_level.aggregation[0].dtype == np.dtype(np.complex128)
        assert coupling.single_level.receive[0].dtype == np.dtype(np.complex128)
        assert coupling.single_level.directional.Fth.dtype == np.dtype(np.complex128)
        assert coupling.single_level.directional.Gth.dtype == np.dtype(np.complex128)
        assert next(iter(coupling.single_level.offset_diagonals.values())).dtype == np.dtype(
            np.complex128
        )

    if coupling.multilevel is not None:
        assert coupling.multilevel.aggregation[0].dtype == np.dtype(np.complex128)
        assert coupling.multilevel.receive[0].dtype == np.dtype(np.complex128)
        leaf_level = coupling.multilevel.leaf_level
        leaf_data = coupling.multilevel.levels[leaf_level]
        assert leaf_data.directional.Fth.dtype == np.dtype(np.complex128)
        assert leaf_data.directional.Gth.dtype == np.dtype(np.complex128)
        assert next(iter(leaf_data.offset_diagonals.values())).dtype == np.dtype(np.complex128)


def test_prepare_matvec_mlfmm_complex64_keeps_far_internal_complex128() -> None:
    lmax, k, particles = _mlfmm_multilevel_problem()
    prepared = prepare_matvec(
        lmax=lmax,
        k=k,
        particles=particles,
        n_medium=1.0 + 0j,
        radial_lut_dr=1.0,
        cache_translation_blocks=False,
        coupling_backend="mlfmm",
        mlfmm_options=MLFMMOptions(max_leaf_particles=1, max_depth=3),
        operator_dtype=np.complex64,
    )

    assert isinstance(prepared.coupling, MLFMMCouplingOperator)
    coupling = prepared.coupling
    assert coupling.dtype == np.dtype(np.complex64)
    assert coupling.near_dtype == np.dtype(np.complex64)
    assert coupling.far_dtype == np.dtype(np.complex128)
    assert coupling.radial_lut.dtype == np.dtype(np.complex128)

    assert coupling.multilevel is not None
    assert coupling.multilevel.aggregation[0].dtype == np.dtype(np.complex128)
    assert coupling.multilevel.receive[0].dtype == np.dtype(np.complex128)
