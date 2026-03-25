from __future__ import annotations

import os
import pickle
import tempfile
from pathlib import Path
from typing import Generator, Literal

import numpy as np
import pytest

import pyceles as pcl
from pyceles._optional import asnumpy, import_cupy
from pyceles.core.indexing import n_modes
from pyceles.core.operators import (
    CuPyMLFMMCouplingOperator,
    CuPyMLFMMHostCachePolicy,
    MLFMMCouplingOperator,
    MLFMMOptions,
    build_mlfmm_cupy_host_cache,
    prepare_matvec,
    prepare_mlfmm_cupy_data,
)
from pyceles.core.operators.mlfmm_cupy import _upload_offset_batches
from pyceles.core.particles import Particle, spheres_from_arrays
from pyceles.io import far_field_intensity


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


def _configure_cupy_tempdir() -> None:
    tmp_root = Path.cwd() / "outputs" / "test_cupy_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    os.environ["TMP"] = str(tmp_root)
    os.environ["TEMP"] = str(tmp_root)
    tempfile.tempdir = str(tmp_root)


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


def _skip_on_cupy_temp_permission(exc: Exception) -> None:
    if (
        isinstance(exc, PermissionError)
        or "Permission denied" in str(exc)
        or "Accesso negato" in str(exc)
    ):
        pytest.skip(f"Local CuPy temp-directory permission issue: {exc}")


def _small_cluster_particles() -> tuple[Particle, ...]:
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
    return tuple(
        spheres_from_arrays(
            positions=positions,
            radii=radii,
            refractive_indices=n_particle,
        )
    )


def _plane_wave_source(wavelength: float, n_medium: complex) -> pcl.PlaneWave:
    return pcl.PlaneWave(
        wavelength=wavelength,
        medium_n=n_medium,
        polarization="TE",
        polar_angle=0.3,
        azimuthal_angle=0.2,
        amplitude=1.0,
    )


def _mixed_cluster_particles() -> tuple[Particle, ...]:
    return (
        pcl.Sphere(
            position=(-220.0, 0.0, -40.0),
            radius=70.0,
            refractive_index=1.52 + 0.01j,
        ),
        pcl.LayeredSphere(
            position=(40.0, 0.0, 30.0),
            layer_radii=(45.0, 85.0),
            layer_refractive_indices=(1.35 + 0.0j, 1.68 + 0.02j),
        ),
        pcl.Spheroid(
            position=(250.0, 0.0, -20.0),
            equatorial_radius=60.0,
            polar_radius=95.0,
            refractive_index=1.47 + 0.03j,
            euler_angles=(0.1, 0.35, -0.2),
        ),
    )


def test_cupy_mlfmm_upload_offset_batches_rejects_nonunique() -> None:
    cupy, _ = import_cupy()
    with pytest.raises(ValueError, match="violates grouped uniqueness contract"):
        _upload_offset_batches(
            {(0, 0, 0): (np.array([1, 1], dtype=np.int64), np.array([2, 3], dtype=np.int64))},
            cupy=cupy,
            name="test_batches",
        )
    with pytest.raises(ValueError, match="violates grouped uniqueness contract"):
        _upload_offset_batches(
            {(0, 0, 0): (np.array([1, 2], dtype=np.int64), np.array([3, 3], dtype=np.int64))},
            cupy=cupy,
            name="test_batches",
        )


def _transition_numpy_mlfmm_coupling() -> MLFMMCouplingOperator:
    prepared = prepare_matvec(
        lmax=3,
        k=2.0 * np.pi / 550.0,
        particles=list(_mlfmm_transition_particles()),
        n_medium=1.0 + 0j,
        radial_lut_dr=0.5,
        cache_translation_blocks=False,
        coupling_backend="mlfmm",
        mlfmm_options=MLFMMOptions(max_leaf_particles=4, max_depth=8),
        backend="numpy",
    )
    coupling = prepared.coupling
    if not isinstance(coupling, MLFMMCouplingOperator):
        raise AssertionError("Transition fixture unexpectedly resolved to direct stage.")
    return coupling


def test_cupy_mlfmm_host_cache_policy_low_host_memory_recomputes_static_tables() -> None:
    coupling = _transition_numpy_mlfmm_coupling()
    policy = CuPyMLFMMHostCachePolicy(memory_budget="low_host_memory")
    host_cache = build_mlfmm_cupy_host_cache(coupling, host_cache_policy=policy)
    assert host_cache.host_memory_budget == "low_host_memory"
    assert host_cache.near_plm_coeffs is None
    assert host_cache.near_compact_re_ab is None
    assert host_cache.near_compact_im_ab is None
    assert host_cache.near_mode_m is None
    assert host_cache.near_pair_offset is None
    assert host_cache.near_pair_pmin is None
    assert host_cache.near_pair_pcount is None

    prepared = prepare_mlfmm_cupy_data(host_cache)
    assert prepared.stage in {"single_level", "multilevel"}
    assert int(prepared.near_pairs.mode_m.size) > 0
    assert int(prepared.near_pairs.pair_offset.size) > 0


def test_cupy_mlfmm_runtime_operator_is_non_picklable() -> None:
    prepared = prepare_matvec(
        lmax=3,
        k=2.0 * np.pi / 550.0,
        particles=list(_mlfmm_transition_particles()),
        n_medium=1.0 + 0j,
        radial_lut_dr=0.5,
        cache_translation_blocks=False,
        coupling_backend="mlfmm",
        mlfmm_options=MLFMMOptions(max_leaf_particles=4, max_depth=8),
        backend="cupy",
    )
    coupling = prepared.coupling
    if not isinstance(coupling, CuPyMLFMMCouplingOperator):
        raise AssertionError("Transition fixture unexpectedly resolved to direct CuPy fallback.")
    with pytest.raises(TypeError, match="non-picklable"):
        _ = pickle.dumps(coupling)


def _mlfmm_transition_particles() -> tuple[Particle, ...]:
    rng = np.random.default_rng(4)
    positions = rng.uniform(-1000.0, 1000.0, size=(60, 3))
    radii = np.full((positions.shape[0],), 20.0, dtype=float)
    n_particle = np.full((positions.shape[0],), 1.59 + 0.0j, dtype=np.complex128)
    return tuple(
        spheres_from_arrays(
            positions=positions,
            radii=radii,
            refractive_indices=n_particle,
        )
    )


def _sim_cfg(
    *,
    operator_backend: Literal["numpy", "cupy"],
    compute_dtype: Literal["complex64", "complex128"],
    wavelength: float,
    n_medium: complex,
    source: pcl.PlaneWave,
) -> pcl.SimulationConfig:
    return pcl.SimulationConfig(
        wavelength=wavelength,
        n_medium=n_medium,
        lmax=3,
        source=source,
        polar_angles=pcl.core.uniform_polar_grid(181),
        azimuthal_angles=pcl.core.uniform_periodic_azimuth_grid(36),
        radial_lut_dr=0.5,
        solver_method="gmres",
        solver_rtol=1e-10 if compute_dtype == "complex128" else 1e-6,
        solver_restart=10,
        solver_maxiter=120,
        operator_backend=operator_backend,
        compute_dtype=compute_dtype,
        accum_dtype="complex128",
        verbose=False,
    )


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
        (np.complex64, 2e-5, 2e-6),
        (np.complex128, 1e-12, 1e-12),
    ],
)
def test_cupy_prepared_operator_block_rhs_matches_columnwise(
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
    rng = np.random.default_rng(17)
    n_unknowns = positions.shape[0] * nm
    x_block = np.asarray(
        rng.standard_normal((n_unknowns, 3)) + 1j * rng.standard_normal((n_unknowns, 3)),
        dtype=operator_dtype,
    )
    b_block = np.asarray(
        rng.standard_normal((n_unknowns, 3)) + 1j * rng.standard_normal((n_unknowns, 3)),
        dtype=operator_dtype,
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

    y_block = np.asarray(prepared_cupy.apply_A(x_block))
    y_cols = np.column_stack([np.asarray(prepared_cupy.apply_A(x_block[:, j])) for j in range(3)])
    np.testing.assert_allclose(y_block, y_cols, rtol=rtol, atol=atol)

    rhs_block = np.asarray(prepared_cupy.rhs_Tb(b_block))
    rhs_cols = np.column_stack([np.asarray(prepared_cupy.rhs_Tb(b_block[:, j])) for j in range(3)])
    np.testing.assert_allclose(rhs_block, rhs_cols, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    ("max_leaf_particles", "expected_stage"),
    [
        (8, "single_level"),
        (4, "multilevel"),
    ],
)
def test_cupy_mlfmm_prepared_operator_matches_numpy_reference(
    max_leaf_particles: int, expected_stage: str
) -> None:
    lmax = 1
    wavelength = 550.0
    n_medium = 1.0 + 0j
    k = 2.0 * np.pi / wavelength
    particles = _mlfmm_transition_particles()
    options = MLFMMOptions(max_leaf_particles=max_leaf_particles, max_depth=4)

    prepared_numpy = prepare_matvec(
        lmax=lmax,
        k=k,
        particles=particles,
        n_medium=n_medium,
        radial_lut_dr=0.5,
        cache_translation_blocks=False,
        operator_dtype=np.complex128,
        coupling_backend="mlfmm",
        mlfmm_options=options,
        backend="numpy",
    )
    prepared_cupy = prepare_matvec(
        lmax=lmax,
        k=k,
        particles=particles,
        n_medium=n_medium,
        radial_lut_dr=0.5,
        cache_translation_blocks=False,
        operator_dtype=np.complex128,
        coupling_backend="mlfmm",
        mlfmm_options=options,
        backend="cupy",
    )

    assert isinstance(prepared_numpy.coupling, MLFMMCouplingOperator)
    assert isinstance(prepared_cupy.coupling, CuPyMLFMMCouplingOperator)
    assert str(prepared_numpy.coupling.resolved_plan.stage) == expected_stage
    assert str(prepared_cupy.coupling.prepared_data.stage) == expected_stage

    nm = n_modes(lmax)
    n_particles = len(particles)
    rng = np.random.default_rng(20260323 + int(max_leaf_particles))
    x = np.asarray(
        rng.standard_normal(n_particles * nm) + 1j * rng.standard_normal(n_particles * nm),
        dtype=np.complex128,
    )
    y_numpy = np.asarray(prepared_numpy.apply_W(x), dtype=np.complex128)
    y_cupy = np.asarray(asnumpy(prepared_cupy.apply_W(x)), dtype=np.complex128)
    np.testing.assert_allclose(y_cupy, y_numpy, rtol=1e-10, atol=1e-10)


def test_cupy_mlfmm_prepared_operator_block_rhs_matches_columnwise() -> None:
    lmax = 1
    wavelength = 550.0
    n_medium = 1.0 + 0j
    k = 2.0 * np.pi / wavelength
    particles = _mlfmm_transition_particles()
    options = MLFMMOptions(max_leaf_particles=8, max_depth=4)
    prepared_cupy = prepare_matvec(
        lmax=lmax,
        k=k,
        particles=particles,
        n_medium=n_medium,
        radial_lut_dr=0.5,
        cache_translation_blocks=False,
        operator_dtype=np.complex128,
        coupling_backend="mlfmm",
        mlfmm_options=options,
        backend="cupy",
    )
    assert isinstance(prepared_cupy.coupling, CuPyMLFMMCouplingOperator)
    nm = n_modes(lmax)
    n_particles = len(particles)
    rng = np.random.default_rng(202603231)
    x_block = np.asarray(
        rng.standard_normal((n_particles * nm, 2))
        + 1j * rng.standard_normal((n_particles * nm, 2)),
        dtype=np.complex128,
    )
    y_block = np.asarray(asnumpy(prepared_cupy.apply_W(x_block)), dtype=np.complex128)
    y_cols = np.column_stack(
        [
            np.asarray(asnumpy(prepared_cupy.apply_W(x_block[:, j])), dtype=np.complex128)
            for j in range(2)
        ]
    )
    np.testing.assert_allclose(y_block, y_cols, rtol=1e-10, atol=1e-10)


def test_cupy_mlfmm_complex64_request_matches_numpy_with_far_complex128() -> None:
    lmax = 1
    wavelength = 550.0
    n_medium = 1.0 + 0j
    k = 2.0 * np.pi / wavelength
    particles = _mlfmm_transition_particles()
    options = MLFMMOptions(max_leaf_particles=8, max_depth=4)

    prepared_numpy = prepare_matvec(
        lmax=lmax,
        k=k,
        particles=particles,
        n_medium=n_medium,
        radial_lut_dr=0.5,
        cache_translation_blocks=False,
        operator_dtype=np.complex64,
        coupling_backend="mlfmm",
        mlfmm_options=options,
        backend="numpy",
    )
    prepared_cupy = prepare_matvec(
        lmax=lmax,
        k=k,
        particles=particles,
        n_medium=n_medium,
        radial_lut_dr=0.5,
        cache_translation_blocks=False,
        operator_dtype=np.complex64,
        coupling_backend="mlfmm",
        mlfmm_options=options,
        backend="cupy",
    )

    assert isinstance(prepared_numpy.coupling, MLFMMCouplingOperator)
    assert isinstance(prepared_cupy.coupling, CuPyMLFMMCouplingOperator)
    assert prepared_numpy.coupling.near_dtype == np.dtype(np.complex64)
    assert prepared_numpy.coupling.far_dtype == np.dtype(np.complex128)
    assert prepared_cupy.coupling.near_dtype == np.dtype(np.complex64)
    assert prepared_cupy.coupling.far_dtype == np.dtype(np.complex128)

    nm = n_modes(lmax)
    n_particles = len(particles)
    rng = np.random.default_rng(20260324)
    x = np.asarray(
        rng.standard_normal(n_particles * nm) + 1j * rng.standard_normal(n_particles * nm),
        dtype=np.complex64,
    )
    y_numpy = np.asarray(prepared_numpy.apply_W(x), dtype=np.complex64)
    y_cupy = np.asarray(asnumpy(prepared_cupy.apply_W(x)), dtype=np.complex64)
    assert y_cupy.dtype == np.dtype(np.complex64)
    np.testing.assert_allclose(y_cupy, y_numpy, rtol=3e-4, atol=3e-5)


@pytest.mark.parametrize(
    (
        "operator_dtype",
        "coeff_rtol",
        "coeff_atol",
        "ff_back_rtol",
        "ff_back_atol",
        "nf_rtol",
        "nf_atol",
    ),
    [
        (np.complex64, 3e-5, 3e-6, 4e-5, 2e-6, 3e-5, 3e-6),
        (np.complex128, 5e-9, 5e-10, 5e-9, 5e-10, 1e-8, 1e-9),
    ],
)
def test_cupy_simulation_run_matches_numpy_for_coeffs_farfield_and_nearfield(
    operator_dtype: np.dtype,
    coeff_rtol: float,
    coeff_atol: float,
    ff_back_rtol: float,
    ff_back_atol: float,
    nf_rtol: float,
    nf_atol: float,
) -> None:
    wavelength = 550.0
    n_medium = 1.0 + 0j
    particles = _small_cluster_particles()
    source = _plane_wave_source(wavelength, n_medium)

    compute_dtype: Literal["complex64", "complex128"] = (
        "complex64" if operator_dtype == np.complex64 else "complex128"
    )
    cfg_numpy = _sim_cfg(
        operator_backend="numpy",
        compute_dtype=compute_dtype,
        wavelength=wavelength,
        n_medium=n_medium,
        source=source,
    )
    cfg_cupy = _sim_cfg(
        operator_backend="cupy",
        compute_dtype=compute_dtype,
        wavelength=wavelength,
        n_medium=n_medium,
        source=source,
    )

    run_numpy = pcl.Simulation(cfg_numpy, particles=particles).run(include_farfield=True)
    try:
        run_cupy = pcl.Simulation(cfg_cupy, particles=particles).run(include_farfield=True)
    except Exception as exc:
        _skip_on_cupy_temp_permission(exc)
        raise

    np.testing.assert_allclose(
        run_cupy.coeffs,
        run_numpy.coeffs,
        rtol=coeff_rtol,
        atol=coeff_atol,
    )
    np.testing.assert_allclose(
        run_cupy.farfield.scattered_te["coeff"],
        run_numpy.farfield.scattered_te["coeff"],
        rtol=coeff_rtol,
        atol=coeff_atol,
    )
    np.testing.assert_allclose(
        run_cupy.farfield.scattered_tm["coeff"],
        run_numpy.farfield.scattered_tm["coeff"],
        rtol=coeff_rtol,
        atol=coeff_atol,
    )

    intensity_numpy = far_field_intensity(
        run_numpy.farfield.scattered_te, run_numpy.farfield.scattered_tm
    )
    intensity_cupy = far_field_intensity(
        run_cupy.farfield.scattered_te, run_cupy.farfield.scattered_tm
    )
    kz = np.asarray(run_numpy.farfield.scattered_te["kz"], dtype=float)
    backward_mask = kz <= 0.0
    np.testing.assert_allclose(
        intensity_cupy[backward_mask],
        intensity_numpy[backward_mask],
        rtol=ff_back_rtol,
        atol=ff_back_atol,
    )

    nearfield_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [220.0, 25.0, -60.0],
            [82.0, 0.0, 0.0],
            [0.0, 84.0, 0.0],
            [220.0 + 83.5, 25.0, -60.0],
            [-180.0, 90.0 + 81.5, 70.0],
            [30.0, -15.0, 110.0],
            [220.0, 25.0, -60.0 + 86.0],
        ],
        dtype=float,
    )
    nf_numpy = pcl.compute_near_field(
        run_numpy, points=nearfield_points, channel="mixed", show_progress=False
    )
    try:
        nf_cupy = pcl.compute_near_field(
            run_cupy, points=nearfield_points, channel="mixed", show_progress=False
        )
    except Exception as exc:
        _skip_on_cupy_temp_permission(exc)
        raise

    np.testing.assert_array_equal(np.asarray(nf_cupy.inside_mask), np.asarray(nf_numpy.inside_mask))
    np.testing.assert_allclose(
        np.asarray(nf_cupy.E_scattered),
        np.asarray(nf_numpy.E_scattered),
        rtol=nf_rtol,
        atol=nf_atol,
    )
    np.testing.assert_allclose(
        np.asarray(nf_cupy.H_scattered),
        np.asarray(nf_numpy.H_scattered),
        rtol=nf_rtol,
        atol=nf_atol,
    )
    np.testing.assert_allclose(
        np.asarray(nf_cupy.E_initial),
        np.asarray(nf_numpy.E_initial),
        rtol=nf_rtol,
        atol=nf_atol,
    )
    np.testing.assert_allclose(
        np.asarray(nf_cupy.H_initial),
        np.asarray(nf_numpy.H_initial),
        rtol=nf_rtol,
        atol=nf_atol,
    )
    np.testing.assert_allclose(
        np.asarray(nf_cupy.E_total),
        np.asarray(nf_numpy.E_total),
        rtol=nf_rtol,
        atol=nf_atol,
    )
    np.testing.assert_allclose(
        np.asarray(nf_cupy.H_total),
        np.asarray(nf_numpy.H_total),
        rtol=nf_rtol,
        atol=nf_atol,
    )


def test_cupy_mixed_particle_groups_match_numpy_for_solve_and_backscatter() -> None:
    wavelength = 550.0
    n_medium = 1.0 + 0j
    particles = _mixed_cluster_particles()
    source = _plane_wave_source(wavelength, n_medium)

    cfg_numpy = _sim_cfg(
        operator_backend="numpy",
        compute_dtype="complex128",
        wavelength=wavelength,
        n_medium=n_medium,
        source=source,
    )
    cfg_cupy = _sim_cfg(
        operator_backend="cupy",
        compute_dtype="complex128",
        wavelength=wavelength,
        n_medium=n_medium,
        source=source,
    )

    run_numpy = pcl.Simulation(cfg_numpy, particles=particles).run(include_farfield=True)
    try:
        run_cupy = pcl.Simulation(cfg_cupy, particles=particles).run(include_farfield=True)
    except Exception as exc:
        _skip_on_cupy_temp_permission(exc)
        raise

    np.testing.assert_allclose(run_cupy.coeffs, run_numpy.coeffs, rtol=5e-8, atol=5e-10)

    intensity_numpy = far_field_intensity(
        run_numpy.farfield.scattered_te,
        run_numpy.farfield.scattered_tm,
    )
    intensity_cupy = far_field_intensity(
        run_cupy.farfield.scattered_te,
        run_cupy.farfield.scattered_tm,
    )
    kz = np.asarray(run_numpy.farfield.scattered_te["kz"], dtype=float)
    backward_mask = kz <= 0.0
    np.testing.assert_allclose(
        intensity_cupy[backward_mask],
        intensity_numpy[backward_mask],
        rtol=5e-8,
        atol=5e-10,
    )
