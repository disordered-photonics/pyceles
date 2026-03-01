from __future__ import annotations

import numpy as np
import pytest

from pyceles.core.fields import PlaneWave
from pyceles.core.indexing import n_modes
from pyceles.core.matvec import assemble_dense_A_numpy, prepare_matvec
from pyceles.linear.preconditioner import make_grid_block_preconditioner, regular_grid_partition
from pyceles.simulation import Simulation, SimulationConfig


def _sample_prepared():
    lmax = 2
    k = 2.0 * np.pi / 550.0
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [120.0, 15.0, -40.0],
            [-60.0, 45.0, 35.0],
        ],
        dtype=float,
    )
    radii = np.array([80.0, 82.0, 79.0], dtype=float)
    n_particle = np.array([1.59 + 0.0j, 1.61 + 0.0j, 1.58 + 0.0j], dtype=np.complex128)
    prepared = prepare_matvec(
        lmax=lmax,
        k=k,
        positions=positions,
        radii=radii,
        n_particle=n_particle,
        n_medium=1.0 + 0j,
        radial_lut_dr=1.0,
        cache_translation_blocks=False,
        operator_dtype=np.complex128,
    )
    return prepared


def test_regular_grid_partition_int_and_tuple_cover_all_particles():
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    blocks_a = regular_grid_partition(positions, subdivisions=2, cubic_bbox=True)
    blocks_b = regular_grid_partition(positions, subdivisions=(2, 2, 2), cubic_bbox=False)
    idx_a = np.concatenate(blocks_a)
    idx_b = np.concatenate(blocks_b)
    np.testing.assert_array_equal(np.sort(idx_a), np.arange(positions.shape[0]))
    np.testing.assert_array_equal(np.sort(idx_b), np.arange(positions.shape[0]))


def test_grid_block_preconditioner_exact_if_single_block_contains_all_particles():
    prepared = _sample_prepared()
    A = assemble_dense_A_numpy(prepared, show_progress=False, use_cache=False, store_blocks=False)
    precond = make_grid_block_preconditioner(
        prepared,
        subdivisions=1,  # one spatial block -> local block is full matrix
        cubic_bbox=True,
        show_progress=False,
    )

    n = prepared.positions.shape[0] * n_modes(prepared.lmax)
    rng = np.random.default_rng(12)
    x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    y = A @ x
    x_rec = precond(y)
    np.testing.assert_allclose(x_rec, x, rtol=1e-10, atol=1e-10)


def test_simulation_supports_builtin_grid_block_preconditioner():
    source = PlaneWave(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        polarization="TE",
        polar_angle=0.0,
        azimuthal_angle=0.0,
        amplitude=1.0,
    )
    cfg = SimulationConfig(
        wavelength=550.0,
        n_medium=1.0 + 0j,
        lmax=2,
        source=source,
        polar_angles=np.linspace(0.0, np.pi, 51),
        azimuthal_angles=np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False),
        solver_method="gmres",
        solver_rtol=1e-4,
        solver_preconditioner_kind="grid_block",
        solver_preconditioner_subdivisions=2,
        verbose=False,
        compute_dtype="complex64",
        accum_dtype="complex128",
    )
    sim = Simulation(
        cfg,
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [240.0, 15.0, -40.0],
                [-210.0, 45.0, 35.0],
            ],
            dtype=float,
        ),
        radii=np.array([70.0, 72.0, 68.0], dtype=float),
        n_particle=np.array([1.59 + 0.0j, 1.61 + 0.0j, 1.58 + 0.0j], dtype=np.complex128),
    )
    run = sim.run()
    assert run.solver_result.info == 0


def test_simulation_config_rejects_ambiguous_preconditioner_settings():
    source = PlaneWave(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        polarization="TE",
        polar_angle=0.0,
        azimuthal_angle=0.0,
        amplitude=1.0,
    )
    with pytest.raises(ValueError):
        SimulationConfig(
            wavelength=550.0,
            n_medium=1.0 + 0j,
            lmax=1,
            source=source,
            solver_method="gmres",
            solver_preconditioner=lambda v: np.asarray(v),
            solver_preconditioner_kind="grid_block",
            verbose=False,
        )
