from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

import pyceles as pcl
import pyceles.core.operators.coupling_periodic_cupy as periodic_cupy_module
from pyceles._cupy_memory import CuPyAllocatorSnapshot
from pyceles.core.indexing import n_modes
from pyceles.core.lattice import RectangularLattice2D
from pyceles.core.operators import (
    CuPyPeriodicCouplingOperator,
    PeriodicCouplingOperator,
    prepare_matvec,
)
from pyceles.core.periodic import PeriodicOptions, PeriodicSpec
from pyceles.core.periodic.rayleigh import _scan_far_numpy, build_rayleigh_plan
from pyceles.core.periodic.rayleigh_cupy import (
    _scan_far_indexed_cupy,
    apply_rayleigh_far_to_points_cupy,
    apply_sparse_near_coupling_cupy,
    scan_far_cupy,
)
from pyceles.core.translation import translation_ab5_table
from pyceles.postprocessing.nearfield.periodic_interior import (
    _periodic_local_regular_l1_coeffs,
)
from pyceles.postprocessing.nearfield.periodic_interior_cupy import (
    periodic_local_regular_l1_coeffs_cupy,
)
from pyceles.simulation.solve import _assemble_dense_operator_for_prepared

pytestmark = pytest.mark.gpu


def _small_periodic_case(
    *,
    cache_blocks: bool = False,
    dtype: Any = np.complex128,
    off_plane: bool = False,
    lmax: int = 1,
) -> tuple[PeriodicCouplingOperator, CuPyPeriodicCouplingOperator]:
    k = 2.0 * np.pi / 550.0
    positions = np.asarray(
        [
            [0.0, 0.0, 40.0],
            [210.0, -120.0, 185.0 if off_plane else 40.0],
        ],
        dtype=float,
    )
    periodic = PeriodicSpec(
        lattice=RectangularLattice2D(ax=900.0, ay=850.0),
        options=PeriodicOptions(
            method="ewald",
            eta=0.002,
            real_shells=1,
            reciprocal_shells=1,
            max_shells=4,
            shell_tolerance=1e-9,
        ),
    )
    k_parallel = np.asarray([0.0, 0.0], dtype=float)
    ab5 = translation_ab5_table(lmax, dtype=dtype)
    cpu = PeriodicCouplingOperator(
        lmax=lmax,
        k=k,
        positions=positions,
        ab5=ab5,
        periodic=periodic,
        k_parallel=k_parallel,
        dtype=np.dtype(dtype),
        cache_blocks=cache_blocks,
    )
    gpu = CuPyPeriodicCouplingOperator(
        lmax=lmax,
        k=k,
        positions=positions,
        ab5=ab5,
        periodic=periodic,
        k_parallel=k_parallel,
        dtype=np.dtype(dtype),
        cache_blocks=cache_blocks,
    )
    return cpu, gpu


def test_periodic_cupy_coupling_apply_matches_numpy_same_plane_two_particle_cell(
    cupy_runtime: tuple[Any, Any],
) -> None:
    cp, _ = cupy_runtime
    cpu, gpu = _small_periodic_case(cache_blocks=False)
    rng = np.random.default_rng(12)
    x = rng.normal(size=12) + 1j * rng.normal(size=12)

    got = gpu.apply(x.astype(np.complex128))
    want = cpu.apply(x.astype(np.complex128))
    assert not hasattr(got, "get")
    np.testing.assert_allclose(np.asarray(got), want, rtol=1e-8, atol=1e-9)

    got_device = gpu.apply(cp.asarray(x.astype(np.complex128)))
    assert hasattr(got_device, "get")
    np.testing.assert_allclose(cp.asnumpy(got_device), want, rtol=1e-8, atol=1e-9)


def test_periodic_cupy_coupling_apply_matches_numpy_off_plane_two_particle_cell(
    cupy_runtime: tuple[Any, Any],
) -> None:
    cp, _ = cupy_runtime
    cpu, gpu = _small_periodic_case(cache_blocks=False, off_plane=True)
    rng = np.random.default_rng(18)
    x = rng.normal(size=12) + 1j * rng.normal(size=12)

    got = gpu.apply(cp.asarray(x.astype(np.complex128)))
    want = cpu.apply(x.astype(np.complex128))

    np.testing.assert_allclose(cp.asnumpy(got), want, rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [
        (np.complex128, 1e-8, 1e-9),
        (np.complex64, 3e-5, 3e-6),
    ],
)
def test_periodic_cupy_shifted_reciprocal_tables_match_numpy_at_lmax3(
    cupy_runtime: tuple[Any, Any], dtype: Any, rtol: float, atol: float
) -> None:
    cp, _ = cupy_runtime
    cpu, gpu = _small_periodic_case(cache_blocks=False, dtype=dtype, off_plane=True, lmax=3)
    rng = np.random.default_rng(20260714)
    size = 2 * gpu.n_modes
    x = rng.normal(size=size) + 1j * rng.normal(size=size)

    got = gpu.apply(cp.asarray(x, dtype=dtype))
    want = cpu.apply(x.astype(dtype))

    np.testing.assert_allclose(cp.asnumpy(got), want, rtol=rtol, atol=atol)


def test_periodic_cupy_coupling_cache_apply_parity(cupy_runtime: tuple[Any, Any]) -> None:
    cp, _ = cupy_runtime
    uncached_cpu, uncached_gpu = _small_periodic_case(cache_blocks=False)
    _cpu_cached, cached_gpu = _small_periodic_case(cache_blocks=True)
    rng = np.random.default_rng(25)
    x = rng.normal(size=12) + 1j * rng.normal(size=12)
    x_device = cp.asarray(x.astype(np.complex128))

    uncached = uncached_gpu.apply(x_device)
    cached_first = cached_gpu.apply(x_device)
    dense_cache = cached_gpu._dense_w_cache_gpu
    assert dense_cache is not None
    assert dense_cache.shape == (12, 12)
    cached_second = cached_gpu.apply(x_device)

    want = uncached_cpu.apply(x.astype(np.complex128))
    np.testing.assert_allclose(cp.asnumpy(uncached), want, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(cp.asnumpy(cached_first), want, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(cp.asnumpy(cached_second), want, rtol=1e-8, atol=1e-9)

    rhs = cp.stack([x_device, 0.5j * x_device], axis=1)
    want_rhs = np.column_stack([want, 0.5j * want])
    np.testing.assert_allclose(
        cp.asnumpy(cached_gpu.apply(rhs)),
        want_rhs,
        rtol=1e-8,
        atol=1e-9,
    )

    # Explicit cache population is idempotent and preserves the contiguous matrix.
    cached_gpu.populate(show_progress=False)
    assert cached_gpu._dense_w_cache_gpu is dense_cache
    np.testing.assert_allclose(cp.asnumpy(cached_gpu.apply(x_device)), want, rtol=1e-8, atol=1e-9)


def test_periodic_cupy_dense_cache_rejects_device_oversubscription(
    cupy_runtime: tuple[Any, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _cp, _ = cupy_runtime
    _cpu, gpu = _small_periodic_case(cache_blocks=True, dtype=np.complex64, lmax=3)
    snapshot = CuPyAllocatorSnapshot(
        raw_free_bytes=1024,
        raw_total_bytes=4096,
        pool_used_bytes=0,
        pool_total_bytes=0,
        pool_free_bytes=0,
        pool_limit_bytes=4096,
        effective_device_limit_bytes=4096,
        active_headroom_bytes=1024,
        guaranteed_fresh_allocation_bytes=1024,
        pool_limit_applied=False,
        pool_trimmed_to_limit=False,
        pool_trimmed_for_fragmentation=False,
    )
    monkeypatch.setattr(
        periodic_cupy_module,
        "cupy_allocator_snapshot",
        lambda *_args, **_kwargs: snapshot,
    )

    with pytest.raises(MemoryError, match="dense W cache requires"):
        gpu.populate(show_progress=False)


@pytest.mark.parametrize("cache_blocks", [False, True])
@pytest.mark.parametrize(
    ("dtype", "tolerance"),
    ((np.complex128, 2.0e-8), (np.complex64, 3.0e-5)),
)
def test_periodic_cupy_ewald_adjoint_matches_inner_product(
    cupy_runtime: tuple[Any, Any], cache_blocks: bool, dtype: Any, tolerance: float
) -> None:
    cp, _ = cupy_runtime
    _cpu, gpu = _small_periodic_case(cache_blocks=cache_blocks, off_plane=True, dtype=dtype)
    rng = np.random.default_rng(20260912)
    x = rng.standard_normal(12) + 1j * rng.standard_normal(12)
    y = rng.standard_normal(12) + 1j * rng.standard_normal(12)
    x_device = cp.asarray(x, dtype=dtype)
    y_device = cp.asarray(y, dtype=dtype)
    forward = gpu.apply(x_device)
    adjoint = gpu.apply_adjoint(y_device)
    lhs = cp.vdot(forward, y_device)
    rhs = cp.vdot(x_device, adjoint)
    discrepancy = float(cp.asnumpy(cp.abs(lhs - rhs)))
    scale = max(float(cp.asnumpy(cp.abs(lhs))), float(cp.asnumpy(cp.abs(rhs))), 1.0)
    assert discrepancy / scale < tolerance


def test_periodic_cupy_dense_assembly_from_cached_blocks_matches_matvec(
    cupy_runtime: tuple[Any, Any],
) -> None:
    cp, _ = cupy_runtime
    spec = PeriodicSpec(
        lattice=RectangularLattice2D(ax=900.0, ay=850.0),
        options=PeriodicOptions(
            method="ewald",
            eta=0.002,
            real_shells=1,
            reciprocal_shells=1,
            max_shells=4,
            shell_tolerance=1e-9,
        ),
    )
    source = pcl.PlaneWave(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        polarization="TE",
        polar_angle=0.0,
        azimuthal_angle=0.0,
    )
    prepared = prepare_matvec(
        lmax=1,
        k=2.0 * np.pi / 550.0,
        particles=[
            pcl.Sphere(position=(0.0, 0.0, 40.0), radius=55.0, refractive_index=1.5 + 0j),
            pcl.Sphere(position=(210.0, -120.0, 185.0), radius=45.0, refractive_index=1.4 + 0j),
        ],
        n_medium=1.0 + 0j,
        radial_lut_dr=1.0,
        cache_translation_blocks=False,
        periodic=spec,
        k_parallel=pcl.core.plane_wave_k_parallel(source),
        backend="cupy",
        show_progress=False,
    )
    assert isinstance(prepared.coupling, CuPyPeriodicCouplingOperator)

    n = 2 * 6
    dense = _assemble_dense_operator_for_prepared(
        prepared=prepared,
        A_mv=prepared.apply_A,
        n=n,
        dtype=np.dtype(np.complex128),
        show_progress=False,
    )
    eye = cp.eye(n, dtype=cp.complex128)
    expected = cp.stack([prepared.apply_A(eye[:, j]) for j in range(n)], axis=1)

    np.testing.assert_allclose(cp.asnumpy(dense), cp.asnumpy(expected), rtol=1e-8, atol=1e-9)
    assert prepared.coupling.cache_blocks is False
    assert prepared.coupling._dense_w_cache_gpu is None


@pytest.mark.parametrize(
    ("dtype", "accum_dtype", "rtol", "atol"),
    [
        (np.complex128, np.complex128, 2e-9, 2e-10),
        (np.complex64, np.complex64, 4e-5, 4e-6),
        (np.complex64, np.complex128, 4e-5, 4e-6),
    ],
)
def test_periodic_cupy_rayleigh_hybrid_matches_numpy_scan_and_near_cache(
    cupy_runtime: tuple[Any, Any],
    monkeypatch: pytest.MonkeyPatch,
    dtype: Any,
    accum_dtype: Any,
    rtol: float,
    atol: float,
) -> None:
    cp, _ = cupy_runtime
    k = 2.0 * np.pi / 550.0
    positions = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [170.0, -80.0, 140.0],
            [-90.0, 60.0, 820.0],
        ],
        dtype=float,
    )
    periodic = PeriodicSpec(
        lattice=RectangularLattice2D(ax=900.0, ay=850.0),
        options=PeriodicOptions(
            method="rayleigh",
            eta=0.002,
            real_shells=2,
            reciprocal_shells=2,
            rayleigh_z_cut=300.0,
            rayleigh_reciprocal_shells=6,
        ),
    )
    ab5 = translation_ab5_table(1, dtype=dtype)
    kwargs = dict(
        lmax=1,
        k=k,
        positions=positions,
        ab5=ab5,
        periodic=periodic,
        k_parallel=np.asarray([0.0003, -0.0002]),
        dtype=np.dtype(dtype),
        cache_blocks=False,
        circumscribing_radii=np.full(positions.shape[0], 40.0),
    )
    cpu = PeriodicCouplingOperator(**cast(Any, kwargs), accum_dtype=np.dtype(accum_dtype))
    gpu = CuPyPeriodicCouplingOperator(**cast(Any, kwargs), accum_dtype=np.dtype(accum_dtype))
    monkeypatch.setattr(gpu, "_near_apply_batch_size", lambda **_kwargs: 1)
    rng = np.random.default_rng(20260725)
    x = rng.normal(size=(18, 2)) + 1j * rng.normal(size=(18, 2))

    expected = cpu.apply(x.astype(dtype))
    actual = gpu.apply(cp.asarray(x, dtype=dtype))
    repeated = gpu.apply(cp.asarray(x, dtype=dtype))

    np.testing.assert_allclose(cp.asnumpy(actual), expected, rtol=rtol, atol=atol)
    np.testing.assert_allclose(cp.asnumpy(repeated), expected, rtol=rtol, atol=atol)
    assert gpu._near_cache_memory_plan is not None
    cache = gpu._near_structural_sums_gpu
    assert cache is not None
    assert cache.shape[1] == 9
    assert cache.dtype == np.dtype(dtype)
    assert gpu._self_block_gpu is not None
    assert gpu._rayleigh_plan_cache is not None


def test_periodic_cupy_sparse_rayleigh_near_kernel_matches_dense_lmax4(
    cupy_runtime: tuple[Any, Any],
) -> None:
    cp, _ = cupy_runtime
    lmax = 4
    nm = n_modes(lmax)
    _cpu, gpu = _small_periodic_case(dtype=np.complex64, lmax=lmax)
    row_ptr, input_modes, channels, values = gpu._near_sparse_contraction_device()
    rng = np.random.default_rng(20260809)
    sources = np.asarray([0, 1, 0], dtype=np.int32)
    destinations = np.asarray([1, 0, 1], dtype=np.int32)
    structural = (
        rng.normal(size=(sources.size, 81)) + 1j * rng.normal(size=(sources.size, 81))
    ).astype(np.complex64)
    coefficients = (rng.normal(size=(2, nm, 2)) + 1j * rng.normal(size=(2, nm, 2))).astype(
        np.complex64
    )

    target = cp.zeros_like(cp.asarray(coefficients))
    apply_sparse_near_coupling_cupy(
        target=target,
        structural=cp.asarray(structural),
        coefficients=cp.asarray(coefficients),
        sources=cp.asarray(sources),
        destinations=cp.asarray(destinations),
        row_ptr=row_ptr,
        input_modes=input_modes,
        structural_channels=channels,
        values=values,
        cupy=cp,
    )
    cp.cuda.Stream.null.synchronize()

    dense = gpu._near_sparse_contraction_device()
    pointers_np, input_np, channels_np, values_np = (cp.asnumpy(item) for item in dense)
    expected = np.zeros_like(coefficients)
    for pair, (source, destination) in enumerate(zip(sources, destinations, strict=True)):
        for output_mode in range(nm):
            start = int(pointers_np[output_mode])
            stop = int(pointers_np[output_mode + 1])
            factors = values_np[start:stop] * structural[pair, channels_np[start:stop]]
            expected[destination, output_mode] += np.sum(
                factors[:, None] * coefficients[source, input_np[start:stop]], axis=0
            )

    np.testing.assert_allclose(cp.asnumpy(target), expected, rtol=5e-5, atol=8e-5)


def test_periodic_cupy_rayleigh_wide_scan_state_matches_c128_recurrence(
    cupy_runtime: tuple[Any, Any],
) -> None:
    cp, _ = cupy_runtime
    rng = np.random.default_rng(20260904)
    n_particles = 257
    n_q = 7
    z = np.sort(rng.uniform(-2200.0, 2200.0, size=n_particles))
    k = 2.0 * np.pi / 366.6666666666667
    rho = np.linspace(0.25 * k, 2.5 * k, n_q)
    gamma = np.sqrt((k * k - rho * rho) + 0.0j).astype(np.complex128)
    source = (
        rng.normal(size=(n_particles, n_q, 2, 1)) + 1j * rng.normal(size=(n_particles, n_q, 2, 1))
    ).astype(np.complex64)

    for upward in (True, False):
        reference = _scan_far_numpy(
            source_amplitudes=source.astype(np.complex128),
            z=z,
            gamma=gamma,
            z_cut=366.6666666666667,
            upward=upward,
        ).astype(np.complex64)
        standard = scan_far_cupy(
            source_amplitudes=cp.asarray(source),
            z=cp.asarray(z),
            gamma=cp.asarray(gamma),
            z_cut=366.6666666666667,
            upward=upward,
            cupy=cp,
        )
        wide = scan_far_cupy(
            source_amplitudes=cp.asarray(source),
            z=cp.asarray(z),
            gamma=cp.asarray(gamma),
            z_cut=366.6666666666667,
            upward=upward,
            cupy=cp,
            accumulation_dtype=np.complex128,
        )
        cp.cuda.Stream.null.synchronize()

        standard_error = np.linalg.norm(cp.asnumpy(standard).astype(np.complex128) - reference)
        wide_error = np.linalg.norm(cp.asnumpy(wide).astype(np.complex128) - reference)
        np.testing.assert_allclose(cp.asnumpy(wide), reference, rtol=2e-6, atol=2e-6)
        assert wide_error < standard_error


def test_periodic_cupy_rayleigh_indexed_wide_scan_overwrites_only_selected_modes(
    cupy_runtime: tuple[Any, Any],
) -> None:
    cp, _ = cupy_runtime
    rng = np.random.default_rng(17)
    n_particles = 97
    n_q = 7
    z = np.sort(rng.uniform(-1500.0, 1500.0, size=n_particles))
    k = 2.0 * np.pi / 366.6666666666667
    rho = np.linspace(0.2 * k, 2.0 * k, n_q)
    gamma = np.sqrt((k * k - rho * rho) + 0.0j).astype(np.complex128)
    source = (
        rng.normal(size=(n_particles, n_q, 2, 1)) + 1j * rng.normal(size=(n_particles, n_q, 2, 1))
    ).astype(np.complex64)
    source_gpu = cp.asarray(source)
    z_gpu = cp.asarray(z)
    gamma_gpu = cp.asarray(gamma)
    selected = np.asarray([0, 3, 6], dtype=np.int32)

    standard = scan_far_cupy(
        source_amplitudes=source_gpu,
        z=z_gpu,
        gamma=gamma_gpu,
        z_cut=366.6666666666667,
        upward=True,
        cupy=cp,
    )
    expected_wide = scan_far_cupy(
        source_amplitudes=source_gpu,
        z=z_gpu,
        gamma=gamma_gpu,
        z_cut=366.6666666666667,
        upward=True,
        cupy=cp,
        accumulation_dtype=np.complex128,
    )
    actual = standard.copy()
    _scan_far_indexed_cupy(
        source_amplitudes=source_gpu,
        z=z_gpu,
        gamma=gamma_gpu,
        z_cut=366.6666666666667,
        upward=True,
        q_indices=cp.asarray(selected),
        output=actual,
        cupy=cp,
        accumulation_dtype=np.complex128,
    )
    cp.cuda.Stream.null.synchronize()

    selected_host = cp.asnumpy(actual[:, selected])
    expected_host = cp.asnumpy(expected_wide[:, selected])
    np.testing.assert_allclose(selected_host, expected_host, rtol=2e-6, atol=2e-6)
    unselected = np.asarray([1, 2, 4, 5], dtype=np.int32)
    np.testing.assert_array_equal(
        cp.asnumpy(actual[:, unselected]),
        cp.asnumpy(standard[:, unselected]),
    )


def test_periodic_cupy_rayleigh_auto_eta_stays_finite_for_large_cell(
    cupy_runtime: tuple[Any, Any],
) -> None:
    cp, _ = cupy_runtime
    k = 2.0 * np.pi / 366.6666666666667
    positions = np.asarray(
        [[0.0, 0.0, 0.0], [125.0, -80.0, 180.0]],
        dtype=float,
    )
    periodic = PeriodicSpec(
        lattice=RectangularLattice2D(ax=3544.8765, ay=3544.8765),
        options=PeriodicOptions(
            method="rayleigh",
            shell_tolerance=1.0e-8,
            max_shells=32,
            rayleigh_reciprocal_shells=12,
        ),
    )
    gpu = CuPyPeriodicCouplingOperator(
        lmax=2,
        k=k,
        positions=positions,
        ab5=translation_ab5_table(2, dtype=np.complex64),
        periodic=periodic,
        k_parallel=np.zeros(2),
        dtype=np.dtype(np.complex64),
        cache_blocks=False,
        circumscribing_radii=np.full(2, 100.0),
    )

    gpu.populate(show_progress=False)

    assert gpu._resolved_ewald_eta is not None
    assert gpu._resolved_ewald_eta > 2.0 * np.sqrt(np.pi / periodic.lattice.area)
    assert gpu._self_block_gpu is not None
    assert bool(cp.all(cp.isfinite(gpu._self_block_gpu)))
    assert gpu._near_structural_sums_gpu is not None
    assert bool(cp.all(cp.isfinite(gpu._near_structural_sums_gpu)))


@pytest.mark.parametrize(
    ("compute_dtype", "accum_dtype", "rtol", "atol"),
    [
        (np.complex128, np.complex128, 2e-9, 2e-10),
        (np.complex64, np.complex128, 5e-5, 5e-6),
        (np.complex64, np.complex64, 5e-5, 5e-6),
    ],
)
def test_periodic_cupy_rayleigh_interior_points_match_numpy(
    cupy_runtime: tuple[Any, Any],
    compute_dtype: type[np.complexfloating[Any, Any]],
    accum_dtype: type[np.complexfloating[Any, Any]],
    rtol: float,
    atol: float,
) -> None:
    cp, _ = cupy_runtime
    lmax = 2
    k = 2.0 * np.pi / 550.0
    positions = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [170.0, -80.0, 140.0],
            [-90.0, 60.0, 820.0],
        ]
    )
    points = np.asarray(
        [
            [20.0, 10.0, 50.0],
            [30.0, -40.0, 500.0],
            [-20.0, 50.0, 900.0],
        ]
    )
    rng = np.random.default_rng(20260728)
    coeffs = (
        rng.normal(size=(positions.shape[0], n_modes(lmax)))
        + 1j * rng.normal(size=(positions.shape[0], n_modes(lmax)))
    ).astype(compute_dtype)
    periodic = PeriodicSpec(
        lattice=RectangularLattice2D(ax=900.0, ay=850.0),
        options=PeriodicOptions(
            method="rayleigh",
            eta=0.002,
            real_shells=5,
            reciprocal_shells=8,
            rayleigh_z_cut=300.0,
            rayleigh_reciprocal_shells=16,
        ),
    )
    kwargs = dict(
        points=points,
        positions=positions,
        coeffs=coeffs,
        lmax=lmax,
        k=k,
        periodic=periodic,
        k_parallel=np.asarray([0.0003, -0.0002]),
        circumscribing_radii=np.full(positions.shape[0], 40.0),
    )

    expected = _periodic_local_regular_l1_coeffs(**cast(Any, kwargs))
    actual = periodic_local_regular_l1_coeffs_cupy(
        **cast(Any, kwargs),
        compute_dtype=compute_dtype,
        accum_dtype=accum_dtype,
    )
    cp.cuda.Stream.null.synchronize()

    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


def test_periodic_cupy_rayleigh_repeated_horizontal_plane_matches_numpy(
    cupy_runtime: tuple[Any, Any],
) -> None:
    """The plane-factorized near band remains CPU-parity accurate in c64."""
    cp, _ = cupy_runtime
    lmax = 2
    k = 2.0 * np.pi / 550.0
    positions = np.asarray(
        [[0.0, 0.0, 0.0], [170.0, -80.0, 140.0], [-90.0, 60.0, 820.0]],
        dtype=float,
    )
    xy = np.asarray([-80.0, 10.0, 90.0])
    xx, yy = np.meshgrid(xy, xy, indexing="xy")
    points = np.column_stack((xx.reshape(-1), yy.reshape(-1), np.full(9, 50.0)))
    rng = np.random.default_rng(20260903)
    coeffs = (
        rng.normal(size=(positions.shape[0], n_modes(lmax)))
        + 1j * rng.normal(size=(positions.shape[0], n_modes(lmax)))
    ).astype(np.complex64)
    periodic = PeriodicSpec(
        lattice=RectangularLattice2D(ax=900.0, ay=850.0),
        options=PeriodicOptions(
            method="rayleigh",
            eta=0.002,
            real_shells=5,
            reciprocal_shells=8,
            rayleigh_z_cut=300.0,
            rayleigh_reciprocal_shells=16,
        ),
    )
    kwargs = dict(
        points=points,
        positions=positions,
        coeffs=coeffs,
        lmax=lmax,
        k=k,
        periodic=periodic,
        k_parallel=np.asarray([0.0003, -0.0002]),
        circumscribing_radii=np.full(positions.shape[0], 40.0),
    )
    expected = _periodic_local_regular_l1_coeffs(**cast(Any, kwargs))
    actual = periodic_local_regular_l1_coeffs_cupy(
        **cast(Any, kwargs),
        compute_dtype=np.complex64,
        accum_dtype=np.complex128,
    )
    cp.cuda.Stream.null.synchronize()

    np.testing.assert_allclose(actual, expected, rtol=6e-5, atol=8e-6)


def test_periodic_cupy_ewald_repeated_horizontal_plane_matches_numpy(
    cupy_runtime: tuple[Any, Any],
) -> None:
    """The same shifted-plane factorization is valid for pure Ewald fields."""
    cp, _ = cupy_runtime
    lmax = 2
    k = 2.0 * np.pi / 550.0
    positions = np.asarray(
        [[0.0, 0.0, 0.0], [170.0, -80.0, 140.0], [-90.0, 60.0, 820.0]],
        dtype=float,
    )
    xy = np.asarray([-80.0, 10.0, 90.0])
    xx, yy = np.meshgrid(xy, xy, indexing="xy")
    points = np.column_stack((xx.reshape(-1), yy.reshape(-1), np.full(9, 50.0)))
    rng = np.random.default_rng(20260904)
    coeffs = (
        rng.normal(size=(positions.shape[0], n_modes(lmax)))
        + 1j * rng.normal(size=(positions.shape[0], n_modes(lmax)))
    ).astype(np.complex64)
    periodic = PeriodicSpec(
        lattice=RectangularLattice2D(ax=900.0, ay=850.0),
        options=PeriodicOptions(
            method="ewald",
            eta=0.002,
            real_shells=5,
            reciprocal_shells=8,
        ),
    )
    kwargs = dict(
        points=points,
        positions=positions,
        coeffs=coeffs,
        lmax=lmax,
        k=k,
        periodic=periodic,
        k_parallel=np.asarray([0.0003, -0.0002]),
    )
    expected = _periodic_local_regular_l1_coeffs(**cast(Any, kwargs))
    actual = periodic_local_regular_l1_coeffs_cupy(
        **cast(Any, kwargs),
        compute_dtype=np.complex64,
        accum_dtype=np.complex128,
    )
    cp.cuda.Stream.null.synchronize()

    np.testing.assert_allclose(actual, expected, rtol=6e-5, atol=8e-6)


def test_periodic_cupy_rayleigh_cartesian_points_match_numpy(
    cupy_runtime: tuple[Any, Any],
) -> None:
    cp, _ = cupy_runtime
    lmax = 2
    k = 2.0 * np.pi / 550.0
    positions = np.asarray([[0.0, 0.0, 0.0], [170.0, -80.0, 140.0], [-90.0, 60.0, 820.0]])
    xy = np.asarray([[20.0, 10.0], [-30.0, 40.0], [55.0, -25.0]])
    z = np.asarray([-420.0, 500.0, 900.0])
    points = np.asarray([[x, y, zz] for zz in z for x, y in xy], dtype=float)
    points = points[[5, 0, 7, 2, 8, 1, 3, 6, 4]]
    rng = np.random.default_rng(20260901)
    coeffs = (
        rng.normal(size=(positions.shape[0], n_modes(lmax)))
        + 1j * rng.normal(size=(positions.shape[0], n_modes(lmax)))
    ).astype(np.complex64)
    plan = build_rayleigh_plan(
        lmax=lmax,
        k=k,
        positions=positions,
        lattice=RectangularLattice2D(ax=900.0, ay=850.0),
        k_parallel=np.asarray([0.0002, -0.0001]),
        z_cut=300.0,
        half_width=8,
        dtype=np.complex64,
    )

    from pyceles.core.periodic.rayleigh import apply_rayleigh_far_to_points_numpy

    expected = apply_rayleigh_far_to_points_numpy(plan, coeffs, points)
    actual = apply_rayleigh_far_to_points_cupy(plan, coeffs, points, cupy=cp)
    cp.cuda.Stream.null.synchronize()

    np.testing.assert_allclose(cp.asnumpy(actual), expected, rtol=4e-5, atol=4e-6)
