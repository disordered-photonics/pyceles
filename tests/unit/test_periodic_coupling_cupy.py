from __future__ import annotations

from typing import Any, Literal, cast

import numpy as np
import pytest

import pyceles as pcl
import pyceles.core.operators.coupling_periodic_cupy as periodic_cupy_module
import pyceles.core.periodic.ewald_cupy as ewald_cupy_module
from pyceles._cupy_memory import CuPyAllocatorSnapshot
from pyceles.core.indexing import n_modes
from pyceles.core.lattice import RectangularLattice2D
from pyceles.core.operators import (
    CuPyPeriodicCouplingOperator,
    PeriodicCouplingOperator,
    prepare_matvec,
)
from pyceles.core.periodic import PeriodicOptions, PeriodicSpec
from pyceles.core.periodic.ewald import (
    EwaldShellWorkspace,
    _shifted_real_sum,
    _shifted_reciprocal_sum,
    ewald_structural_sums_2d_batch,
)
from pyceles.core.periodic.ewald_cupy import (
    CupyEwaldShellWorkspace,
    _add_real_space_structural_sums_cupy,
    _add_shifted_reciprocal_structural_sums_cupy,
)
from pyceles.core.periodic.rayleigh import (
    _scan_far_numpy,
    apply_rayleigh_far_adjoint_numpy,
    apply_rayleigh_far_numpy,
    build_rayleigh_plan,
)
from pyceles.core.periodic.rayleigh_cupy import (
    _scan_far_indexed_cupy,
    apply_rayleigh_far_to_points_cupy,
    apply_sparse_near_adjoint_cupy,
    apply_sparse_near_coupling_cupy,
    scan_far_cupy,
)
from pyceles.core.periodic.scalar import structural_sum_m_normalization
from pyceles.core.translation import translation_ab5_table
from pyceles.postprocessing.nearfield.periodic_interior import (
    _periodic_local_regular_l1_coeffs,
)
from pyceles.postprocessing.nearfield.periodic_interior_cupy import (
    periodic_local_regular_l1_coeffs_cupy,
)
from pyceles.simulation import Simulation, SimulationConfig
from pyceles.simulation.solve import _assemble_dense_operator_for_prepared

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("method", ("ewald", "rayleigh"))
def test_periodic_cupy_complex_host_simulation_matches_numpy(
    cupy_runtime: tuple[Any, Any],
    method: Literal["ewald", "rayleigh"],
) -> None:
    del cupy_runtime
    host = 1.33 + 0.08j
    wavelength = 550.0
    lattice = RectangularLattice2D(ax=900.0, ay=850.0)
    options = PeriodicOptions(
        method=method,
        eta=np.sqrt(np.pi / lattice.area),
        real_shells=8,
        reciprocal_shells=8,
        output_bmax=0.16,
        rayleigh_z_cut=300.0 if method == "rayleigh" else None,
        rayleigh_reciprocal_shells=8 if method == "rayleigh" else None,
    )
    periodic = PeriodicSpec(lattice=lattice, options=options)
    particles = [
        pcl.Sphere(position=(0.0, 0.0, 0.0), radius=20.0, refractive_index=1.5 + 0.0j),
        pcl.Sphere(position=(140.0, -90.0, 110.0), radius=20.0, refractive_index=1.5 + 0.0j),
    ]
    source = pcl.BlochPlaneWave(
        wavelength=wavelength,
        medium_n=host,
        k_parallel=(0.0003, -0.0002),
        polarization="TE",
        direction="down",
    )
    runs: dict[str, Any] = {}
    for backend in ("numpy", "cupy"):
        config = SimulationConfig(
            wavelength=wavelength,
            n_medium=host,
            lmax=1,
            periodic=periodic,
            operator_backend=backend,
            solver_method="gmres",
            solver_rtol=1e-10,
            solver_maxiter=100,
            compute_dtype="complex128",
            accum_dtype="complex128",
            verbose=False,
        )
        runs[backend] = Simulation(config, particles=particles).run(
            source,
            include_farfield=False,
        )

    np.testing.assert_allclose(
        runs["cupy"].rhs,
        runs["numpy"].rhs,
        rtol=1e-13,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        runs["cupy"].coeffs,
        runs["numpy"].coeffs,
        rtol=2e-8,
        atol=2e-10,
    )
    assert runs["cupy"].solver_result.info == 0
    assert runs["numpy"].solver_result.info == 0

    points = np.asarray([[300.0, 100.0, 70.0]])
    fields = {
        backend: pcl.compute_periodic_near_field(
            run,
            points=points,
            field_bmax=0.16,
        )
        for backend, run in runs.items()
    }
    field_rtol = 3e-8
    field_atol = 3e-10
    for component in ("E_scattered", "H_scattered", "E_total", "H_total"):
        np.testing.assert_allclose(
            getattr(fields["cupy"], component),
            getattr(fields["numpy"], component),
            rtol=field_rtol,
            atol=field_atol,
        )


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


def test_periodic_cupy_complex_host_ewald_real_space_matches_numpy(
    cupy_runtime: tuple[Any, Any],
) -> None:
    cp, _ = cupy_runtime
    k = (2.0 * np.pi / 550.0) * (1.0 + 0.025j)
    lattice = RectangularLattice2D(ax=900.0, ay=850.0)
    k_parallel = np.asarray([0.0004, -0.0002])
    eta = 0.002
    deltas = np.asarray([[50.0, -70.0, 130.0], [160.0, 40.0, -225.0]])
    order = 3
    shell_count = 1
    workspace_gpu = CupyEwaldShellWorkspace(
        cupy=cp,
        lattice=lattice,
        k=k,
        k_parallel=k_parallel,
        eta=eta,
    )
    sums_gpu = cp.zeros((deltas.shape[0], order + 1, 2 * order + 1), dtype=cp.complex128)
    _add_real_space_structural_sums_cupy(
        c=cp.asarray(deltas),
        same_plane=cp.zeros(deltas.shape[0], dtype=cp.bool_),
        sums=sums_gpu,
        workspace=workspace_gpu,
        real_shell_count=shell_count,
        order=order,
        eta=eta,
        k=k,
    )

    workspace_cpu = EwaldShellWorkspace(
        lattice=lattice,
        k=k,
        k_parallel=k_parallel,
        eta=eta,
    )
    expected = np.zeros((deltas.shape[0], order + 1, 2 * order + 1), dtype=np.complex128)
    for pair, delta in enumerate(deltas):
        for degree in range(order + 1):
            for azimuthal_order in range(-degree, degree + 1):
                expected[pair, degree, azimuthal_order + order] = structural_sum_m_normalization(
                    azimuthal_order
                ) * _shifted_real_sum(
                    degree,
                    azimuthal_order,
                    rvec=-delta,
                    k=k,
                    k_parallel=k_parallel,
                    lattice=lattice,
                    eta=eta,
                    shells=shell_count,
                    workspace=workspace_cpu,
                )

    np.testing.assert_allclose(cp.asnumpy(sums_gpu), expected, rtol=2e-9, atol=2e-10)


def test_periodic_cupy_complex_host_ewald_shifted_reciprocal_matches_numpy(
    cupy_runtime: tuple[Any, Any],
) -> None:
    cp, _ = cupy_runtime
    k = (2.0 * np.pi / 550.0) * (1.0 + 0.025j)
    lattice = RectangularLattice2D(ax=900.0, ay=850.0)
    k_parallel = np.asarray([0.0004, -0.0002])
    eta = 0.002
    deltas = np.asarray([[50.0, -70.0, 130.0], [160.0, 40.0, -225.0]])
    order = 2
    workspace_gpu = CupyEwaldShellWorkspace(
        cupy=cp,
        lattice=lattice,
        k=k,
        k_parallel=k_parallel,
        eta=eta,
    )
    sums_gpu = cp.zeros((deltas.shape[0], order + 1, 2 * order + 1), dtype=cp.complex128)
    _add_shifted_reciprocal_structural_sums_cupy(
        c=cp.asarray(deltas),
        same_plane=cp.zeros(deltas.shape[0], dtype=cp.bool_),
        sums=sums_gpu,
        workspace=workspace_gpu,
        reciprocal_shell_count=0,
        order=order,
        k=k,
    )

    workspace_cpu = EwaldShellWorkspace(
        lattice=lattice,
        k=k,
        k_parallel=k_parallel,
        eta=eta,
    )
    expected = np.zeros((deltas.shape[0], order + 1, 2 * order + 1), dtype=np.complex128)
    for pair, delta in enumerate(deltas):
        for degree in range(order + 1):
            for azimuthal_order in range(-degree, degree + 1):
                expected[pair, degree, azimuthal_order + order] = structural_sum_m_normalization(
                    azimuthal_order
                ) * _shifted_reciprocal_sum(
                    degree,
                    azimuthal_order,
                    rvec=-delta,
                    k=k,
                    k_parallel=k_parallel,
                    lattice=lattice,
                    eta=eta,
                    shells=0,
                    workspace=workspace_cpu,
                )

    np.testing.assert_allclose(cp.asnumpy(sums_gpu), expected, rtol=3e-9, atol=3e-10)


def test_periodic_cupy_complex_host_ewald_structural_sums_match_numpy(
    cupy_runtime: tuple[Any, Any],
) -> None:
    cp, _ = cupy_runtime
    k = (2.0 * np.pi / 550.0) * (1.0 + 0.025j)
    lattice = RectangularLattice2D(ax=900.0, ay=850.0)
    k_parallel = np.asarray([0.0004, -0.0002])
    eta = 0.002
    deltas = np.asarray([[50.0, -70.0, 0.0], [50.0, -70.0, 130.0], [160.0, 40.0, -225.0]])
    workspace_gpu = CupyEwaldShellWorkspace(
        cupy=cp,
        lattice=lattice,
        k=k,
        k_parallel=k_parallel,
        eta=eta,
    )
    actual = ewald_cupy_module.ewald_structural_sums_2d_fixed_cupy(
        relative_source_minus_destination=cp.asarray(deltas),
        lmax_struct=2,
        workspace=workspace_gpu,
        real_shell_count=1,
        reciprocal_shell_count=1,
    )

    workspace_cpu = EwaldShellWorkspace(
        lattice=lattice,
        k=k,
        k_parallel=k_parallel,
        eta=eta,
    )
    expected = ewald_structural_sums_2d_batch(
        lmax_struct=2,
        k=k,
        destinations=-deltas,
        source=np.zeros(3),
        lattice=lattice,
        k_parallel=k_parallel,
        eta=eta,
        real_shells=1,
        reciprocal_shells=1,
        workspace=workspace_cpu,
    )

    np.testing.assert_allclose(cp.asnumpy(actual), expected, rtol=8e-9, atol=8e-10)


@pytest.mark.parametrize("host_index", (1.0, 1.0 + 0.025j))
def test_periodic_cupy_ewald_reused_gamma_table_respects_row_stride(
    cupy_runtime: tuple[Any, Any],
    host_index: complex,
) -> None:
    """A high-order cache must not corrupt a later lower-order shifted sum."""
    cp, _ = cupy_runtime
    k = (2.0 * np.pi / 550.0) * host_index
    lattice = RectangularLattice2D(ax=900.0, ay=850.0)
    offsets = cp.asarray([[50.0, -70.0, 130.0], [-40.0, 30.0, -95.0]])
    kwargs = dict(
        cupy=cp,
        lattice=lattice,
        k=k,
        k_parallel=np.array([0.0004, -0.0002]),
        eta=0.002,
    )
    reused = CupyEwaldShellWorkspace(**kwargs)
    reused.upper_gamma_terms(1, 40)
    fresh = CupyEwaldShellWorkspace(**kwargs)

    def evaluate(workspace: CupyEwaldShellWorkspace) -> Any:
        return ewald_cupy_module.ewald_structural_sums_2d_fixed_cupy(
            relative_source_minus_destination=offsets,
            lmax_struct=1,
            workspace=workspace,
            real_shell_count=1,
            reciprocal_shell_count=1,
        )

    np.testing.assert_allclose(
        cp.asnumpy(evaluate(reused)),
        cp.asnumpy(evaluate(fresh)),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    ("host_index", "lmax_struct"),
    ((1.0, 12), (1.5 + 0.01j, 10), (1.5 + 0.1j, 12)),
)
def test_periodic_cupy_ewald_high_order_structural_sums_match_numpy(
    cupy_runtime: tuple[Any, Any], host_index: complex, lmax_struct: int
) -> None:
    cp, _ = cupy_runtime
    k = (2.0 * np.pi / 550.0) * host_index
    lattice = RectangularLattice2D(ax=900.0, ay=850.0)
    k0 = 2.0 * np.pi / 550.0
    k_parallel = np.asarray([0.2 * k0, -0.11 * k0])
    eta = 0.002
    deltas = np.asarray(
        [[50.0, -70.0, 130.0], [-40.0, 30.0, -95.0], [170.0, 110.0, 600.0]],
        dtype=np.float64,
    )
    workspace_gpu = CupyEwaldShellWorkspace(
        cupy=cp,
        lattice=lattice,
        k=k,
        k_parallel=k_parallel,
        eta=eta,
    )
    actual = ewald_cupy_module.ewald_structural_sums_2d_fixed_cupy(
        relative_source_minus_destination=cp.asarray(deltas),
        lmax_struct=lmax_struct,
        workspace=workspace_gpu,
        real_shell_count=1,
        reciprocal_shell_count=1,
    )
    expected = ewald_structural_sums_2d_batch(
        lmax_struct=lmax_struct,
        k=k,
        destinations=-deltas,
        source=np.zeros(3),
        lattice=lattice,
        k_parallel=k_parallel,
        eta=eta,
        real_shells=1,
        reciprocal_shells=1,
        dtype=np.complex128,
    )
    np.testing.assert_allclose(cp.asnumpy(actual), expected, rtol=2e-8, atol=2e-9)


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


@pytest.mark.parametrize("cache_blocks", (False, True))
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    ((np.complex128, 2e-8, 2e-9), (np.complex64, 3e-5, 3e-6)),
)
def test_periodic_cupy_ewald_complex_host_apply_and_adjoint_match_numpy(
    cupy_runtime: tuple[Any, Any], cache_blocks: bool, dtype: Any, rtol: float, atol: float
) -> None:
    cp, _ = cupy_runtime
    lmax = 1
    k = (2.0 * np.pi / 550.0) * (1.0 + 0.025j)
    positions = np.asarray(
        [[0.0, 0.0, 40.0], [210.0, -120.0, 40.0], [-90.0, 60.0, 185.0]],
        dtype=float,
    )
    periodic = PeriodicSpec(
        lattice=RectangularLattice2D(ax=900.0, ay=850.0),
        options=PeriodicOptions(
            method="ewald",
            eta=0.002,
            real_shells=1,
            reciprocal_shells=1,
        ),
    )
    kwargs = dict(
        lmax=lmax,
        k=k,
        positions=positions,
        ab5=translation_ab5_table(lmax, dtype=dtype),
        periodic=periodic,
        k_parallel=np.asarray([0.0003, -0.0002]),
        dtype=np.dtype(dtype),
        cache_blocks=cache_blocks,
    )
    cpu = PeriodicCouplingOperator(**cast(Any, kwargs))
    gpu = CuPyPeriodicCouplingOperator(**cast(Any, kwargs))
    rng = np.random.default_rng(20261023)
    x = (
        rng.normal(size=positions.shape[0] * 6) + 1j * rng.normal(size=positions.shape[0] * 6)
    ).astype(dtype)
    y = (
        rng.normal(size=positions.shape[0] * 6) + 1j * rng.normal(size=positions.shape[0] * 6)
    ).astype(dtype)
    x_gpu = cp.asarray(x, dtype=dtype)
    y_gpu = cp.asarray(y, dtype=dtype)

    actual = gpu.apply(x_gpu)
    expected = cpu.apply(x)
    actual_adjoint = gpu.apply_adjoint(y_gpu)
    cp.cuda.Stream.null.synchronize()

    np.testing.assert_allclose(cp.asnumpy(actual), expected, rtol=rtol, atol=atol)
    lhs = cp.vdot(actual, y_gpu)
    rhs = cp.vdot(x_gpu, actual_adjoint)
    discrepancy = float(cp.asnumpy(cp.abs(lhs - rhs)))
    scale = max(float(cp.asnumpy(cp.abs(lhs))), float(cp.asnumpy(cp.abs(rhs))), 1.0)
    assert discrepancy / scale < max(rtol, 2e-8)


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
        backend="cupy",
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


@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    ((np.complex128, 2e-8, 2e-9), (np.complex64, 3e-5, 3e-6)),
)
def test_periodic_cupy_rayleigh_complex_host_apply_and_adjoint_match_numpy(
    cupy_runtime: tuple[Any, Any], dtype: Any, rtol: float, atol: float
) -> None:
    cp, _ = cupy_runtime
    lmax = 1
    k = (2.0 * np.pi / 550.0) * (1.0 + 0.025j)
    positions = np.asarray(
        [[0.0, 0.0, 0.0], [170.0, -80.0, 140.0], [-90.0, 60.0, 820.0]],
        dtype=float,
    )
    radii = np.full(positions.shape[0], 40.0)
    lattice = RectangularLattice2D(ax=900.0, ay=850.0)
    periodic = PeriodicSpec(
        lattice=lattice,
        options=PeriodicOptions(
            method="rayleigh",
            eta=0.002,
            real_shells=1,
            reciprocal_shells=1,
            rayleigh_z_cut=300.0,
            rayleigh_reciprocal_shells=5,
        ),
    )
    kwargs = dict(
        lmax=lmax,
        k=k,
        positions=positions,
        ab5=translation_ab5_table(lmax, dtype=dtype),
        periodic=periodic,
        k_parallel=np.asarray([0.0003, -0.0002]),
        dtype=np.dtype(dtype),
        cache_blocks=False,
        circumscribing_radii=radii,
    )
    cpu = PeriodicCouplingOperator(**cast(Any, kwargs))
    gpu = CuPyPeriodicCouplingOperator(**cast(Any, kwargs))
    rng = np.random.default_rng(20261022)
    x = (
        rng.normal(size=positions.shape[0] * 6) + 1j * rng.normal(size=positions.shape[0] * 6)
    ).astype(dtype)
    y = (
        rng.normal(size=positions.shape[0] * 6) + 1j * rng.normal(size=positions.shape[0] * 6)
    ).astype(dtype)
    x_gpu = cp.asarray(x, dtype=dtype)
    y_gpu = cp.asarray(y, dtype=dtype)

    actual = gpu.apply(x_gpu)
    expected = cpu.apply(x)
    actual_adjoint = gpu.apply_adjoint(y_gpu)
    cp.cuda.Stream.null.synchronize()

    np.testing.assert_allclose(cp.asnumpy(actual), expected, rtol=rtol, atol=atol)
    lhs = cp.vdot(actual, y_gpu)
    rhs = cp.vdot(x_gpu, actual_adjoint)
    discrepancy = float(cp.asnumpy(cp.abs(lhs - rhs)))
    scale = max(float(cp.asnumpy(cp.abs(lhs))), float(cp.asnumpy(cp.abs(rhs))), 1.0)
    assert discrepancy / scale < max(rtol, 2e-8)


@pytest.mark.parametrize("dtype", (np.complex64, np.complex128))
def test_periodic_cupy_rayleigh_noncoplanar_far_action_and_adjoint(
    cupy_runtime: tuple[Any, Any], monkeypatch: pytest.MonkeyPatch, dtype: Any
) -> None:
    cp, _ = cupy_runtime
    lmax = 2
    k = (2.0 * np.pi / 550.0) * (1.5 + 0.05j)
    k_parallel = np.asarray([0.2, -0.13]) * (2.0 * np.pi / 550.0)
    positions = np.asarray(
        [
            [0.0, 0.0, -250.0],
            [170.0, -80.0, -30.0],
            [-90.0, 60.0, 210.0],
            [240.0, 130.0, 470.0],
            [-210.0, -170.0, 910.0],
        ],
        dtype=np.float64,
    )
    periodic = PeriodicSpec(
        lattice=RectangularLattice2D(ax=900.0, ay=850.0),
        options=PeriodicOptions(
            method="rayleigh",
            eta=0.002,
            real_shells=1,
            reciprocal_shells=1,
            rayleigh_z_cut=100.0,
            rayleigh_reciprocal_shells=4,
        ),
    )
    operator = CuPyPeriodicCouplingOperator(
        lmax=lmax,
        k=k,
        positions=positions,
        ab5=translation_ab5_table(lmax, dtype=dtype),
        periodic=periodic,
        k_parallel=k_parallel,
        dtype=np.dtype(dtype),
        accum_dtype=np.dtype(np.complex128),
        cache_blocks=False,
        circumscribing_radii=np.full(positions.shape[0], 35.0),
    )
    rng = np.random.default_rng(20261024)
    shape = (positions.shape[0] * operator.n_modes,)
    x = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(dtype)
    y = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(dtype)
    x_device = cp.asarray(x)
    y_device = cp.asarray(y)

    # Isolate the far Rayleigh action while exercising the operator's real
    # forward path, then use its explicit far adjoint for the inner-product test.
    monkeypatch.setattr(
        operator,
        "_self_block_device",
        lambda: cp.zeros((operator.n_modes, operator.n_modes), dtype=dtype),
    )
    monkeypatch.setattr(
        operator, "_near_pairs", lambda: (np.empty(0, np.int32), np.empty(0, np.int32))
    )
    monkeypatch.setattr(
        operator,
        "_populate_near_structural_sums_device",
        lambda show_progress=False: setattr(
            operator, "_near_structural_sums_gpu", cp.empty((0, 0), dtype=dtype)
        ),
    )

    plan = operator._rayleigh_plan()
    expected = apply_rayleigh_far_numpy(
        plan,
        x.reshape(positions.shape[0], operator.n_modes),
        accumulation_dtype=np.complex128,
    ).reshape(-1)
    actual = operator._apply_rayleigh_gpu(x_device)
    adjoint_reshaped, _ = operator._reshape_input_device(y_device)
    actual_adjoint = operator._apply_rayleigh_far_adjoint_reshaped_gpu(adjoint_reshaped).reshape(-1)
    expected_adjoint = apply_rayleigh_far_adjoint_numpy(
        plan,
        y.reshape(positions.shape[0], operator.n_modes),
        accumulation_dtype=np.complex128,
    ).reshape(-1)
    cp.cuda.Stream.null.synchronize()

    tolerance = 5e-5 if np.dtype(dtype) == np.dtype(np.complex64) else 2e-9
    np.testing.assert_allclose(cp.asnumpy(actual), expected, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(
        cp.asnumpy(actual_adjoint), expected_adjoint, rtol=tolerance, atol=tolerance
    )
    lhs = cp.vdot(actual, y_device)
    rhs = cp.vdot(x_device, actual_adjoint)
    scale = max(float(cp.asnumpy(cp.abs(lhs))), float(cp.asnumpy(cp.abs(rhs))), 1.0)
    assert float(cp.asnumpy(cp.abs(lhs - rhs))) / scale < tolerance


@pytest.mark.parametrize("dtype", (np.complex64, np.complex128))
@pytest.mark.parametrize("nrhs", (1, 3, 5))
def test_periodic_cupy_sparse_rayleigh_near_kernel_matches_dense_lmax4(
    cupy_runtime: tuple[Any, Any],
    dtype: Any,
    nrhs: int,
) -> None:
    cp, _ = cupy_runtime
    lmax = 4
    nm = n_modes(lmax)
    _cpu, gpu = _small_periodic_case(dtype=dtype, lmax=lmax)
    row_ptr, input_modes, channels, values = gpu._near_sparse_contraction_device()
    rng = np.random.default_rng(20260809)
    sources = np.asarray([0, 1, 0], dtype=np.int32)
    destinations = np.asarray([1, 0, 1], dtype=np.int32)
    structural = (
        rng.normal(size=(sources.size, 81)) + 1j * rng.normal(size=(sources.size, 81))
    ).astype(dtype)
    coefficients = (rng.normal(size=(2, nm, nrhs)) + 1j * rng.normal(size=(2, nm, nrhs))).astype(
        dtype
    )
    pair_order = np.argsort(destinations, kind="stable").astype(np.int32, copy=False)
    counts = np.bincount(destinations, minlength=coefficients.shape[0]).astype(np.int64)
    owner_ptr = np.empty(coefficients.shape[0] + 1, dtype=np.int64)
    owner_ptr[0] = 0
    np.cumsum(counts, out=owner_ptr[1:])
    owners = np.flatnonzero(counts).astype(np.int32, copy=False)

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
        owner_ptr=cp.asarray(owner_ptr),
        pair_order=cp.asarray(pair_order),
        owners=cp.asarray(owners),
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

    tolerance = 8e-5 if np.dtype(dtype) == np.dtype(np.complex64) else 2e-11
    np.testing.assert_allclose(cp.asnumpy(target), expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("dtype", (np.complex64, np.complex128))
@pytest.mark.parametrize("nrhs", (1, 3, 5))
def test_periodic_cupy_sparse_rayleigh_owned_adjoint_matches_dense_lmax4(
    cupy_runtime: tuple[Any, Any],
    dtype: Any,
    nrhs: int,
) -> None:
    cp, _ = cupy_runtime
    lmax = 4
    nm = n_modes(lmax)
    _cpu, gpu = _small_periodic_case(dtype=dtype, lmax=lmax)
    row_ptr, input_modes, channels, values = gpu._near_sparse_adjoint_contraction_device()
    rng = np.random.default_rng(20260810)
    sources = np.asarray([0, 1, 0], dtype=np.int32)
    destinations = np.asarray([1, 0, 1], dtype=np.int32)
    structural = (
        rng.normal(size=(sources.size, 81)) + 1j * rng.normal(size=(sources.size, 81))
    ).astype(dtype)
    coefficients = (rng.normal(size=(2, nm, nrhs)) + 1j * rng.normal(size=(2, nm, nrhs))).astype(
        dtype
    )
    pair_order = np.argsort(sources, kind="stable").astype(np.int32, copy=False)
    counts = np.bincount(sources, minlength=coefficients.shape[0]).astype(np.int64)
    owner_ptr = np.empty(coefficients.shape[0] + 1, dtype=np.int64)
    owner_ptr[0] = 0
    np.cumsum(counts, out=owner_ptr[1:])
    owners = np.flatnonzero(counts).astype(np.int32, copy=False)

    target = cp.zeros_like(cp.asarray(coefficients))
    apply_sparse_near_adjoint_cupy(
        target=target,
        structural=cp.asarray(structural),
        coefficients=cp.asarray(coefficients),
        input_particles=cp.asarray(destinations),
        output_particles=cp.asarray(sources),
        row_ptr=row_ptr,
        input_modes=input_modes,
        structural_channels=channels,
        values=values,
        cupy=cp,
        owner_ptr=cp.asarray(owner_ptr),
        pair_order=cp.asarray(pair_order),
        owners=cp.asarray(owners),
    )
    cp.cuda.Stream.null.synchronize()

    pointers_np, input_np, channels_np, values_np = (
        cp.asnumpy(item) for item in (row_ptr, input_modes, channels, values)
    )
    expected = np.zeros_like(coefficients)
    for pair, (source, destination) in enumerate(zip(sources, destinations, strict=True)):
        for output_mode in range(nm):
            start = int(pointers_np[output_mode])
            stop = int(pointers_np[output_mode + 1])
            factors = np.conj(values_np[start:stop] * structural[pair, channels_np[start:stop]])
            expected[source, output_mode] += np.sum(
                factors[:, None] * coefficients[destination, input_np[start:stop]], axis=0
            )

    tolerance = 8e-5 if np.dtype(dtype) == np.dtype(np.complex64) else 2e-11
    np.testing.assert_allclose(cp.asnumpy(target), expected, rtol=tolerance, atol=tolerance)


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


@pytest.mark.parametrize("dtype", (np.complex64, np.complex128))
def test_periodic_cupy_rayleigh_scan_preserves_complex_gamma_phase(
    cupy_runtime: tuple[Any, Any], dtype: Any
) -> None:
    cp, _ = cupy_runtime
    rng = np.random.default_rng(20261020)
    z = np.sort(rng.uniform(-45.0, 45.0, size=19))
    gamma = np.asarray(
        [0.018 + 0.002j, 0.011 + 0.0j, 0.0 + 0.014j, 0.007 + 0.004j],
        dtype=np.complex128,
    )
    source = (
        rng.normal(size=(z.size, gamma.size, 2, 3))
        + 1j * rng.normal(size=(z.size, gamma.size, 2, 3))
    ).astype(dtype)
    tolerance = 3e-5 if np.dtype(dtype) == np.dtype(np.complex64) else 2e-12

    for upward in (True, False):
        expected = _scan_far_numpy(
            source_amplitudes=source.astype(np.complex128),
            z=z,
            gamma=gamma,
            z_cut=13.0,
            upward=upward,
        ).astype(dtype)
        actual = scan_far_cupy(
            source_amplitudes=cp.asarray(source),
            z=cp.asarray(z),
            gamma=cp.asarray(gamma),
            z_cut=13.0,
            upward=upward,
            cupy=cp,
        )
        np.testing.assert_allclose(cp.asnumpy(actual), expected, rtol=tolerance, atol=tolerance)

        selected = np.asarray([0, 2, 3], dtype=np.int32)
        indexed = cp.zeros_like(cp.asarray(source))
        _scan_far_indexed_cupy(
            source_amplitudes=cp.asarray(source),
            z=cp.asarray(z),
            gamma=cp.asarray(gamma),
            z_cut=13.0,
            upward=upward,
            q_indices=cp.asarray(selected),
            output=indexed,
            cupy=cp,
            accumulation_dtype=np.complex128,
        )
        np.testing.assert_allclose(
            cp.asnumpy(indexed[:, selected]),
            expected[:, selected],
            rtol=tolerance,
            atol=tolerance,
        )


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


@pytest.mark.parametrize("method", ("ewald", "rayleigh"))
def test_periodic_cupy_complex_host_interior_fields_match_numpy(
    cupy_runtime: tuple[Any, Any], method: str
) -> None:
    cp, _ = cupy_runtime
    lmax = 1
    k = (2.0 * np.pi / 550.0) * (1.0 + 0.025j)
    positions = np.asarray(
        [[0.0, 0.0, 0.0], [170.0, -80.0, 140.0], [-90.0, 60.0, 820.0]],
        dtype=float,
    )
    xy = np.asarray([-80.0, 10.0, 90.0])
    xx, yy = np.meshgrid(xy, xy, indexing="xy")
    points = np.column_stack((xx.reshape(-1), yy.reshape(-1), np.full(9, 50.0)))
    rng = np.random.default_rng(20261024)
    coeffs = (
        rng.normal(size=(positions.shape[0], n_modes(lmax)))
        + 1j * rng.normal(size=(positions.shape[0], n_modes(lmax)))
    ).astype(np.complex128)
    options = dict(
        method=method,
        eta=0.002,
        real_shells=1,
        reciprocal_shells=1,
    )
    if method == "rayleigh":
        options.update(rayleigh_z_cut=300.0, rayleigh_reciprocal_shells=5)
    periodic = PeriodicSpec(
        lattice=RectangularLattice2D(ax=900.0, ay=850.0),
        options=PeriodicOptions(**cast(Any, options)),
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
        compute_dtype=np.complex128,
        accum_dtype=np.complex128,
    )
    cp.cuda.Stream.null.synchronize()

    np.testing.assert_allclose(actual, expected, rtol=2e-8, atol=2e-9)


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


def test_periodic_cupy_rayleigh_complex_host_points_match_numpy(
    cupy_runtime: tuple[Any, Any],
) -> None:
    cp, _ = cupy_runtime
    lmax = 2
    k = (2.0 * np.pi / 550.0) * (1.0 + 0.025j)
    positions = np.asarray([[0.0, 0.0, 0.0], [170.0, -80.0, 140.0], [-90.0, 60.0, 820.0]])
    points = np.asarray(
        [[20.0, 10.0, -420.0], [-30.0, 40.0, 500.0], [55.0, -25.0, 900.0]],
        dtype=float,
    )
    rng = np.random.default_rng(20261021)
    coeffs = (
        rng.normal(size=(positions.shape[0], n_modes(lmax)))
        + 1j * rng.normal(size=(positions.shape[0], n_modes(lmax)))
    ).astype(np.complex128)
    plan = build_rayleigh_plan(
        lmax=lmax,
        k=k,
        positions=positions,
        lattice=RectangularLattice2D(ax=900.0, ay=850.0),
        k_parallel=np.asarray([0.0002, -0.0001]),
        z_cut=300.0,
        half_width=8,
        dtype=np.complex128,
    )

    from pyceles.core.periodic.rayleigh import apply_rayleigh_far_to_points_numpy

    expected = apply_rayleigh_far_to_points_numpy(plan, coeffs, points)
    actual = apply_rayleigh_far_to_points_cupy(plan, coeffs, points, cupy=cp)
    cp.cuda.Stream.null.synchronize()

    np.testing.assert_allclose(cp.asnumpy(actual), expected, rtol=3e-11, atol=3e-12)
