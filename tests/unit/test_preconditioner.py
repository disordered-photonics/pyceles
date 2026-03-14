from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pytest

from pyceles._optional import import_cupy
from pyceles.core.fields import PlaneWave
from pyceles.core.indexing import n_modes
from pyceles.core.operators import (
    CompositeParticleTOperator,
    DenseTGroup,
    DiagonalTGroup,
    PreparedOperator,
    assemble_dense_A_numpy,
    prepare_matvec,
)
from pyceles.core.particles import layered_spheres_from_arrays, spheres_from_arrays
from pyceles.linear.preconditioner import make_grid_block_preconditioner, regular_grid_partition
from pyceles.simulation import Simulation, SimulationConfig


def _cupy_available() -> bool:
    try:
        cupy, _ = import_cupy()
    except RuntimeError:
        return False
    try:
        x = cupy.arange(1, dtype=cupy.float32)
        cupy.cuda.Stream.null.synchronize()
        return int(cupy.asnumpy(x)[0]) == 0
    except Exception:
        return False


cupy_available = pytest.mark.skipif(not _cupy_available(), reason="CuPy runtime unavailable")


def _configure_cupy_tempdir() -> None:
    temp_root = Path("outputs/test_cupy_tmp").resolve()
    temp_root.mkdir(parents=True, exist_ok=True)
    os.environ["TMP"] = str(temp_root)
    os.environ["TEMP"] = str(temp_root)
    tempfile.tempdir = str(temp_root)


@pytest.fixture(scope="module", autouse=True)
def _cupy_tempdir_env() -> Generator[None, None, None]:
    prev_tmp = os.environ.get("TMP")
    prev_temp = os.environ.get("TEMP")
    prev_tempdir = tempfile.tempdir
    try:
        _configure_cupy_tempdir()
    except PermissionError as exc:
        pytest.skip(f"Local CuPy temp-directory permission issue: {exc}")
    yield
    if prev_tmp is None:
        os.environ.pop("TMP", None)
    else:
        os.environ["TMP"] = prev_tmp
    if prev_temp is None:
        os.environ.pop("TEMP", None)
    else:
        os.environ["TEMP"] = prev_temp
    tempfile.tempdir = prev_tempdir


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
        particles=spheres_from_arrays(
            positions=positions,
            radii=radii,
            refractive_indices=n_particle,
        ),
        n_medium=1.0 + 0j,
        radial_lut_dr=1.0,
        cache_translation_blocks=False,
        operator_dtype=np.complex128,
    )
    return prepared


def _sample_layered_prepared():
    lmax = 2
    k = 2.0 * np.pi / 550.0
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [150.0, -10.0, 35.0],
            [-125.0, 55.0, -25.0],
        ],
        dtype=float,
    )
    prepared = prepare_matvec(
        lmax=lmax,
        k=k,
        particles=layered_spheres_from_arrays(
            positions=positions,
            layer_radii=np.array(
                [
                    [40.0, 80.0],
                    [35.0, 82.0],
                    [38.0, 79.0],
                ],
                dtype=float,
            ),
            layer_refractive_indices=np.array(
                [
                    [1.70 + 0.0j, 1.59 + 0.0j],
                    [1.68 + 0.0j, 1.61 + 0.0j],
                    [1.72 + 0.0j, 1.58 + 0.0j],
                ],
                dtype=np.complex128,
            ),
        ),
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


def test_grid_block_preconditioner_exact_for_layered_spheres_too():
    prepared = _sample_layered_prepared()
    A = assemble_dense_A_numpy(prepared, show_progress=False, use_cache=False, store_blocks=False)
    precond = make_grid_block_preconditioner(
        prepared,
        subdivisions=1,
        cubic_bbox=True,
        show_progress=False,
    )

    n = prepared.positions.shape[0] * n_modes(prepared.lmax)
    rng = np.random.default_rng(21)
    x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    y = A @ x
    x_rec = precond(y)
    np.testing.assert_allclose(x_rec, x, rtol=1e-10, atol=1e-10)


def test_grid_block_preconditioner_handles_mixed_diagonal_and_dense_groups():
    prepared = _sample_prepared()
    Nm = n_modes(prepared.lmax)

    diagonal_group = DiagonalTGroup(
        particle_indices=np.array([0], dtype=np.int64),
        T_M=prepared.T_M[[0]],
        T_N=prepared.T_N[[0]],
        T_diag=prepared.T_diag[[0]],
        dtype=np.dtype(prepared.dtype),
    )
    dense_group = DenseTGroup(
        particle_indices=np.array([1, 2], dtype=np.int64),
        T_blocks=np.stack(
            [
                np.diag(prepared.T_diag[1]),
                np.diag(prepared.T_diag[2]),
            ],
            axis=0,
        ),
        dtype=np.dtype(prepared.dtype),
    )
    mixed_particle_t = CompositeParticleTOperator(
        lmax=prepared.lmax,
        n_particles=prepared.positions.shape[0],
        groups=(diagonal_group, dense_group),
        dtype=np.dtype(prepared.dtype),
    )
    mixed_prepared = PreparedOperator(
        lmax=prepared.lmax,
        k=prepared.k,
        positions=prepared.positions,
        particle_t=mixed_particle_t,
        coupling=prepared.coupling,
        dtype=np.dtype(prepared.dtype),
    )

    A = assemble_dense_A_numpy(
        mixed_prepared, show_progress=False, use_cache=False, store_blocks=False
    )
    precond = make_grid_block_preconditioner(
        mixed_prepared,
        subdivisions=1,
        cubic_bbox=True,
        show_progress=False,
    )

    rng = np.random.default_rng(31)
    x = rng.standard_normal(3 * Nm) + 1j * rng.standard_normal(3 * Nm)
    y = A @ x
    x_rec = precond(y)
    np.testing.assert_allclose(x_rec, x, rtol=1e-10, atol=1e-10)


@cupy_available
def test_grid_block_preconditioner_exact_for_single_cupy_block() -> None:
    _configure_cupy_tempdir()
    prepared = _sample_prepared()
    prepared_cupy = prepare_matvec(
        lmax=prepared.lmax,
        k=prepared.k,
        particles=spheres_from_arrays(
            positions=prepared.positions,
            radii=np.array([80.0, 82.0, 79.0], dtype=float),
            refractive_indices=np.array(
                [1.59 + 0.0j, 1.61 + 0.0j, 1.58 + 0.0j], dtype=np.complex128
            ),
        ),
        n_medium=1.0 + 0j,
        radial_lut_dr=1.0,
        cache_translation_blocks=False,
        operator_dtype=np.complex128,
        backend="cupy",
    )
    A = assemble_dense_A_numpy(prepared, show_progress=False, use_cache=False, store_blocks=False)
    try:
        precond = make_grid_block_preconditioner(
            prepared_cupy,
            backend="cupy",
            subdivisions=1,
            cubic_bbox=True,
            show_progress=False,
        )
    except PermissionError as exc:
        pytest.skip(f"Local CuPy NVRTC temp-dir cleanup issue on this machine: {exc}")

    n = prepared.positions.shape[0] * n_modes(prepared.lmax)
    rng = np.random.default_rng(44)
    x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    y = A @ x
    x_rec = precond(y)
    np.testing.assert_allclose(x_rec, x, rtol=1e-10, atol=1e-10)


@cupy_available
def test_cupy_grid_block_preconditioner_matches_numpy_apply() -> None:
    _configure_cupy_tempdir()
    prepared_numpy = _sample_prepared()
    prepared_cupy = prepare_matvec(
        lmax=prepared_numpy.lmax,
        k=prepared_numpy.k,
        particles=spheres_from_arrays(
            positions=prepared_numpy.positions,
            radii=np.array([80.0, 82.0, 79.0], dtype=float),
            refractive_indices=np.array(
                [1.59 + 0.0j, 1.61 + 0.0j, 1.58 + 0.0j], dtype=np.complex128
            ),
        ),
        n_medium=1.0 + 0j,
        radial_lut_dr=1.0,
        cache_translation_blocks=False,
        operator_dtype=np.complex128,
        backend="cupy",
    )
    try:
        precond_numpy = make_grid_block_preconditioner(
            prepared_numpy,
            backend="numpy",
            subdivisions=2,
            cubic_bbox=True,
            show_progress=False,
        )
        precond_cupy = make_grid_block_preconditioner(
            prepared_cupy,
            backend="cupy",
            subdivisions=2,
            cubic_bbox=True,
            show_progress=False,
        )
    except PermissionError as exc:
        pytest.skip(f"Local CuPy NVRTC temp-dir cleanup issue on this machine: {exc}")

    n = prepared_numpy.positions.shape[0] * n_modes(prepared_numpy.lmax)
    rng = np.random.default_rng(45)
    x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    try:
        cupy_out = precond_cupy(x)
    except PermissionError as exc:
        pytest.skip(f"Local CuPy NVRTC temp-dir cleanup issue on this machine: {exc}")
    np.testing.assert_allclose(cupy_out, precond_numpy(x), rtol=1e-10, atol=1e-10)


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
        particles=spheres_from_arrays(
            positions=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [240.0, 15.0, -40.0],
                    [-210.0, 45.0, 35.0],
                ],
                dtype=float,
            ),
            radii=np.array([70.0, 72.0, 68.0], dtype=float),
            refractive_indices=np.array(
                [1.59 + 0.0j, 1.61 + 0.0j, 1.58 + 0.0j],
                dtype=np.complex128,
            ),
        ),
    )
    run = sim.run()
    assert run.solver_result.info == 0


@cupy_available
def test_simulation_supports_builtin_grid_block_preconditioner_with_cupy_backend():
    _configure_cupy_tempdir()
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
        solver_restart=10,
        solver_maxiter=80,
        solver_preconditioner_kind="grid_block",
        solver_preconditioner_subdivisions=2,
        verbose=False,
        compute_dtype="complex64",
        accum_dtype="complex128",
        operator_backend="cupy",
    )
    sim = Simulation(
        cfg,
        particles=spheres_from_arrays(
            positions=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [240.0, 15.0, -40.0],
                    [-210.0, 45.0, 35.0],
                ],
                dtype=float,
            ),
            radii=np.array([70.0, 72.0, 68.0], dtype=float),
            refractive_indices=np.array(
                [1.59 + 0.0j, 1.61 + 0.0j, 1.58 + 0.0j],
                dtype=np.complex128,
            ),
        ),
    )
    try:
        run = sim.run()
    except PermissionError as exc:
        pytest.skip(f"Local CuPy NVRTC temp-dir cleanup issue on this machine: {exc}")
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
