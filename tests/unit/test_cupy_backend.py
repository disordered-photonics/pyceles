from __future__ import annotations

from typing import Literal

import numpy as np
import pytest

import pyceles as pcl
from pyceles._optional import import_cupy
from pyceles.core.indexing import n_modes
from pyceles.core.operators import prepare_matvec
from pyceles.core.particles import spheres_from_arrays


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


pytestmark = pytest.mark.skipif(not _cupy_available(), reason="CuPy runtime unavailable")


@pytest.mark.parametrize(
    ("operator_dtype", "rtol", "atol"),
    [
        (np.complex64, 2e-5, 2e-6),
        (np.complex128, 1e-12, 1e-12),
    ],
)
def test_cupy_prepared_operator_matches_numpy_for_diagonal_spheres(
    operator_dtype: np.dtype, rtol: float, atol: float
) -> None:
    lmax = 3
    wavelength = 550.0
    n_medium = 1.0 + 0j
    k = 2.0 * np.pi / wavelength
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [220.0, 25.0, -60.0],
            [-180.0, 90.0, 70.0],
        ],
        dtype=float,
    )
    radii = np.array([80.0, 82.0, 79.0], dtype=float)
    n_particle = np.array([1.59 + 0.0j, 1.61 + 0.0j, 1.58 + 0.0j], dtype=np.complex128)
    particles = spheres_from_arrays(
        positions=positions,
        radii=radii,
        refractive_indices=n_particle,
    )
    nm = n_modes(lmax)
    rng = np.random.default_rng(7)
    x = np.asarray(
        rng.standard_normal(positions.shape[0] * nm)
        + 1j * rng.standard_normal(positions.shape[0] * nm),
        dtype=operator_dtype,
    )
    b = np.asarray(
        rng.standard_normal(positions.shape[0] * nm)
        + 1j * rng.standard_normal(positions.shape[0] * nm),
        dtype=operator_dtype,
    )

    prepared_numpy = prepare_matvec(
        lmax=lmax,
        k=k,
        particles=particles,
        n_medium=n_medium,
        radial_lut_dr=0.5,
        cache_translation_blocks=False,
        operator_dtype=operator_dtype,
        backend="numpy",
    )
    prepared_cupy = prepare_matvec(
        lmax=lmax,
        k=k,
        particles=particles,
        n_medium=n_medium,
        radial_lut_dr=0.5,
        cache_translation_blocks=False,
        operator_dtype=operator_dtype,
        backend="cupy",
    )

    np.testing.assert_allclose(
        prepared_cupy.apply_A(x), prepared_numpy.apply_A(x), rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        prepared_cupy.rhs_Tb(b), prepared_numpy.rhs_Tb(b), rtol=rtol, atol=atol
    )


@pytest.mark.parametrize(
    ("operator_dtype", "rtol", "atol"),
    [
        (np.complex64, 3e-5, 3e-6),
        (np.complex128, 2e-5, 2e-8),
    ],
)
def test_cupy_simulation_run_matches_numpy_for_farfield_observables(
    operator_dtype: np.dtype, rtol: float, atol: float
) -> None:
    lmax = 3
    wavelength = 550.0
    n_medium = 1.0 + 0j
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [220.0, 25.0, -60.0],
            [-180.0, 90.0, 70.0],
        ],
        dtype=float,
    )
    radii = np.array([80.0, 82.0, 79.0], dtype=float)
    n_particle = np.array([1.59 + 0.0j, 1.61 + 0.0j, 1.58 + 0.0j], dtype=np.complex128)
    particles = tuple(
        spheres_from_arrays(
            positions=positions,
            radii=radii,
            refractive_indices=n_particle,
        )
    )
    source = pcl.PlaneWave(
        wavelength=wavelength,
        medium_n=n_medium,
        polarization="TE",
        polar_angle=0.3,
        azimuthal_angle=0.2,
        amplitude=1.0,
    )

    compute_dtype: Literal["complex64", "complex128"] = (
        "complex64" if operator_dtype == np.complex64 else "complex128"
    )
    cfg_numpy = pcl.SimulationConfig(
        wavelength=wavelength,
        n_medium=n_medium,
        lmax=lmax,
        source=source,
        polar_angles=pcl.core.uniform_polar_grid(181),
        azimuthal_angles=pcl.core.uniform_periodic_azimuth_grid(36),
        radial_lut_dr=0.5,
        solver_method="gmres",
        solver_rtol=1e-6,
        solver_restart=10,
        solver_maxiter=80,
        operator_backend="numpy",
        compute_dtype=compute_dtype,
        accum_dtype="complex128",
        verbose=False,
    )
    cfg_cupy = pcl.SimulationConfig(
        wavelength=wavelength,
        n_medium=n_medium,
        lmax=lmax,
        source=source,
        polar_angles=pcl.core.uniform_polar_grid(181),
        azimuthal_angles=pcl.core.uniform_periodic_azimuth_grid(36),
        radial_lut_dr=0.5,
        solver_method="gmres",
        solver_rtol=1e-6,
        solver_restart=10,
        solver_maxiter=80,
        operator_backend="cupy",
        compute_dtype=compute_dtype,
        accum_dtype="complex128",
        verbose=False,
    )

    run_numpy = pcl.Simulation(cfg_numpy, particles=particles).run(include_farfield=True)
    run_cupy = pcl.Simulation(cfg_cupy, particles=particles).run(include_farfield=True)

    np.testing.assert_allclose(run_cupy.coeffs, run_numpy.coeffs, rtol=rtol, atol=atol)
    np.testing.assert_allclose(
        run_cupy.farfield.scattered_te["coeff"],
        run_numpy.farfield.scattered_te["coeff"],
        rtol=rtol,
        atol=atol,
    )
    np.testing.assert_allclose(
        run_cupy.farfield.scattered_tm["coeff"],
        run_numpy.farfield.scattered_tm["coeff"],
        rtol=rtol,
        atol=atol,
    )
