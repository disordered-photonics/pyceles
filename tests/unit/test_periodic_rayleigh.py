from __future__ import annotations

from typing import Any, Literal, cast

import numpy as np
import pytest

import pyceles as pcl
from pyceles._cupy_memory import CuPyAllocatorSnapshot
from pyceles.core.indexing import index_vswf, n_modes
from pyceles.core.operators import (
    CuPyPeriodicCouplingOperator,
    PeriodicCouplingOperator,
    prepare_matvec,
)
from pyceles.core.operators import (
    coupling_periodic as coupling_periodic_module,
)
from pyceles.core.operators.coupling_periodic_cupy import (
    _resolve_rayleigh_near_cache_memory_plan,
)
from pyceles.core.periodic import PeriodicOptions, PeriodicSpec
from pyceles.core.periodic import rayleigh as rayleigh_module
from pyceles.core.periodic.ewald import ewald_structural_sums_2d_batch, periodic_ewald_block
from pyceles.core.periodic.rayleigh import (
    apply_rayleigh_far_numpy,
    apply_rayleigh_far_to_points_numpy,
    build_rayleigh_plan,
    near_pair_csr,
    near_point_source_csr,
    rayleigh_block,
    rayleigh_near_cache_estimate,
    rayleigh_structural_sums_2d_batch,
    resolve_rayleigh_half_width,
    resolve_rayleigh_mode_chunk_size,
    resolve_rayleigh_z_cut,
)
from pyceles.core.translation import translation_ab5_table
from pyceles.postprocessing.nearfield import periodic_interior as periodic_interior_module
from pyceles.postprocessing.nearfield.periodic_interior import (
    _periodic_local_regular_l1_coeffs,
)


def test_shifted_rayleigh_structural_batch_matches_ewald_reference() -> None:
    lattice = pcl.RectangularLattice2D(900.0, 950.0)
    k = 2.0 * np.pi / 550.0
    k_parallel = np.asarray([1.2e-3, -0.7e-3], dtype=float)
    displacements = np.asarray(
        [
            [120.0, -80.0, 250.0],
            [-100.0, 50.0, -330.0],
            [10.0, 15.0, 700.0],
        ],
        dtype=float,
    )
    expected = ewald_structural_sums_2d_batch(
        lmax_struct=3,
        k=k,
        destinations=displacements,
        source=np.zeros(3),
        lattice=lattice,
        k_parallel=k_parallel,
        eta=2.5e-3,
        real_shells=8,
        reciprocal_shells=8,
        shell_tolerance=1.0e-12,
        max_shells=16,
        dtype=np.complex128,
    )
    actual = rayleigh_structural_sums_2d_batch(
        max_degree=6,
        k=k,
        displacements=displacements,
        lattice=lattice,
        k_parallel=k_parallel,
        tolerance=1.0e-12,
        max_shells=64,
        requested_half_width=32,
        matmul_backend="numpy",
    )

    np.testing.assert_allclose(actual, expected, rtol=2.0e-11, atol=5.0e-12)


def test_default_rayleigh_band_is_wavelength_and_particle_safe() -> None:
    k = 2.0 * np.pi / 600.0

    assert resolve_rayleigh_z_cut(
        k=k,
        circumscribing_radii=np.asarray([40.0, 90.0]),
        requested=None,
    ) == pytest.approx(600.0)
    assert resolve_rayleigh_z_cut(
        k=k,
        circumscribing_radii=np.asarray([340.0, 90.0]),
        requested=None,
    ) == pytest.approx(680.0)
    with pytest.raises(ValueError, match="twice the largest"):
        resolve_rayleigh_z_cut(
            k=k,
            circumscribing_radii=np.asarray([90.0]),
            requested=150.0,
        )


def test_near_pair_csr_and_far_scan_partition_boundary_exactly() -> None:
    positions = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [10.0, 5.0, 100.0],
            [-20.0, 8.0, 200.0],
        ]
    )
    indptr, destinations, sources = near_pair_csr(positions, 100.0)

    np.testing.assert_array_equal(indptr, np.asarray([0, 1, 3, 4]))
    np.testing.assert_array_equal(destinations, np.asarray([1, 0, 2, 1]))
    np.testing.assert_array_equal(sources, np.asarray([0, 1, 1, 2]))
    assert indptr.dtype == np.dtype(np.int64)
    assert destinations.dtype == np.dtype(np.int32)
    assert sources.dtype == np.dtype(np.int32)


def test_near_point_source_csr_keeps_boundary_pairs_exact() -> None:
    points = np.asarray(
        [
            [0.0, 0.0, -100.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 100.0],
            [0.0, 0.0, 201.0],
        ]
    )
    sources = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 101.0]])

    indptr, destinations, source_indices = near_point_source_csr(points, sources, 100.0)

    np.testing.assert_array_equal(indptr, np.asarray([0, 3, 5]))
    np.testing.assert_array_equal(destinations, np.asarray([0, 1, 2, 2, 3]))
    np.testing.assert_array_equal(source_indices, np.asarray([0, 0, 0, 1, 1]))
    assert indptr.dtype == np.dtype(np.int64)
    assert destinations.dtype == np.dtype(np.int32)
    assert source_indices.dtype == np.dtype(np.int32)


def test_default_wavelength_band_keeps_exact_boundary_near() -> None:
    k = 2.0 * np.pi / 500.0
    z_cut = resolve_rayleigh_z_cut(
        k=k,
        circumscribing_radii=np.asarray([40.0]),
        requested=None,
    )
    positions = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 500.0]])

    _indptr, destinations, sources = near_pair_csr(positions, z_cut)

    np.testing.assert_array_equal(destinations, np.asarray([1, 0]))
    np.testing.assert_array_equal(sources, np.asarray([0, 1]))


def test_near_pair_csr_returns_source_major_nonself_pairs() -> None:
    positions = np.asarray(
        [
            [0.0, 0.0, 200.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 40.0],
            [0.0, 0.0, 120.0],
        ]
    )

    indptr, destinations, sources = near_pair_csr(positions, 100.0)

    np.testing.assert_array_equal(indptr, np.asarray([0, 1, 2, 4, 6]))
    np.testing.assert_array_equal(destinations, np.asarray([3, 2, 1, 3, 2, 0]))
    np.testing.assert_array_equal(sources, np.asarray([0, 1, 2, 2, 3, 3]))


def test_rayleigh_workspace_chunk_is_a_memory_aware_upper_bound() -> None:
    assert (
        resolve_rayleigh_mode_chunk_size(
            n_modes_reciprocal=3025,
            n_particles=500,
            n_rhs=1,
            dtype=np.complex128,
        )
        == 3025
    )
    assert (
        resolve_rayleigh_mode_chunk_size(
            n_modes_reciprocal=3025,
            n_particles=1_000_000,
            n_rhs=1,
            dtype=np.complex128,
        )
        == 4
    )
    assert (
        resolve_rayleigh_mode_chunk_size(
            n_modes_reciprocal=7,
            n_particles=10,
            n_rhs=2,
            dtype=np.complex64,
        )
        == 7
    )


@pytest.mark.parametrize(
    ("angle_deg", "lmax", "z_cut", "max_degree", "max_shells", "reference_width"),
    [
        (0.0, 26, 2186.8097898, 52, 64, 64),
        (25.0, 26, 2186.8097898, 52, 64, 64),
        # Accuracy-6 production MLFMM uses closure order 59, mapping to
        # structural lmax=30 for the adaptive Rayleigh envelope.
        (0.0, 30, 2186.8097898, 59, 64, 64),
        (25.0, 30, 2186.8097898, 59, 64, 64),
        (0.0, 4, 366.6666667, 8, 128, 96),
        (25.0, 4, 366.6666667, 8, 128, 96),
    ],
)
def test_rayleigh_half_width_accounts_for_complex_angular_growth(
    angle_deg: float,
    lmax: int,
    z_cut: float,
    max_degree: int,
    max_shells: int,
    reference_width: int,
) -> None:
    """Adaptive windows resolve closure and particle-level degrees."""

    lattice = pcl.RectangularLattice2D(3544.8765, 3544.8765)
    k = 2.0 * np.pi * 1.5 / 550.0
    angle = np.deg2rad(float(angle_deg))
    k_parallel = np.asarray([k * np.sin(angle), 0.0])
    width = resolve_rayleigh_half_width(
        lattice=lattice,
        k=k,
        k_parallel=k_parallel,
        lmax=int(lmax),
        z_cut=float(z_cut),
        tolerance=1.0e-8,
        max_shells=int(max_shells),
        requested=None,
    )

    displacement = np.asarray([[0.0, 0.0, float(z_cut)]])
    selected = rayleigh_structural_sums_2d_batch(
        max_degree=int(max_degree),
        k=k,
        displacements=displacement,
        lattice=lattice,
        k_parallel=k_parallel,
        tolerance=1.0e-8,
        max_shells=max(int(max_shells), int(reference_width)),
        requested_half_width=int(width),
    )
    reference = rayleigh_structural_sums_2d_batch(
        max_degree=int(max_degree),
        k=k,
        displacements=displacement,
        lattice=lattice,
        tolerance=1.0e-8,
        k_parallel=k_parallel,
        max_shells=max(int(max_shells), int(reference_width)),
        requested_half_width=int(reference_width),
    )
    relative_error = np.linalg.norm(selected - reference) / np.linalg.norm(reference)

    # The original radial-power heuristic selected h=16 for the degree-52
    # production closure. The complex-angle envelope must remain accurate not
    # only there but also for the shorter particle-level exact-near band and
    # for oblique incidence.
    assert width < reference_width
    assert relative_error < 1.0e-8


def test_automatic_rayleigh_half_width_is_memoized() -> None:
    rayleigh_module._resolve_rayleigh_half_width_cached.cache_clear()
    kwargs: dict[str, Any] = dict(
        lattice=pcl.RectangularLattice2D(3544.8765, 3544.8765),
        k=2.0 * np.pi * 1.5 / 550.0,
        k_parallel=np.asarray([0.0002, -0.0001]),
        lmax=4,
        z_cut=366.6666667,
        tolerance=1.0e-8,
        max_shells=64,
        requested=None,
    )

    first = resolve_rayleigh_half_width(**kwargs)
    info_after_first = rayleigh_module._resolve_rayleigh_half_width_cached.cache_info()
    second = resolve_rayleigh_half_width(**kwargs)
    info_after_second = rayleigh_module._resolve_rayleigh_half_width_cached.cache_info()

    assert second == first
    assert info_after_second.hits == info_after_first.hits + 1


def test_rayleigh_half_width_rejects_negative_lmax() -> None:
    with pytest.raises(ValueError, match=r"lmax.*non-negative"):
        resolve_rayleigh_half_width(
            lattice=pcl.RectangularLattice2D(900.0, 900.0),
            k=2.0 * np.pi / 550.0,
            k_parallel=np.zeros(2),
            lmax=-1,
            z_cut=550.0,
            tolerance=1.0e-8,
            max_shells=32,
            requested=None,
        )


def test_rayleigh_plan_rejects_exact_wood_anomaly() -> None:
    wavelength = 550.0
    with pytest.raises(ValueError, match="Wood anomaly"):
        build_rayleigh_plan(
            lmax=1,
            k=2.0 * np.pi / wavelength,
            positions=np.zeros((1, 3)),
            lattice=pcl.RectangularLattice2D(wavelength, wavelength),
            k_parallel=np.zeros(2),
            z_cut=wavelength,
            half_width=1,
            dtype=np.complex128,
        )


def test_semiseparable_rayleigh_scan_matches_explicit_far_blocks() -> None:
    k = 2.0 * np.pi / 550.0
    lattice = pcl.RectangularLattice2D(ax=900.0, ay=850.0)
    positions = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [120.0, -50.0, 170.0],
            [-80.0, 90.0, 410.0],
            [40.0, 30.0, 760.0],
        ],
        dtype=float,
    )
    plan = build_rayleigh_plan(
        lmax=2,
        k=k,
        positions=positions,
        lattice=lattice,
        k_parallel=np.asarray([0.0004, -0.0002]),
        z_cut=220.0,
        half_width=5,
        dtype=np.complex128,
    )
    assert plan.sort_order.dtype == np.dtype(np.int32)
    assert plan.inverse_order.dtype == np.dtype(np.int32)

    rng = np.random.default_rng(20260725)
    x = rng.normal(size=(positions.shape[0], 16)) + 1j * rng.normal(size=(positions.shape[0], 16))

    expected = np.zeros_like(x)
    for destination in range(positions.shape[0]):
        for source in range(positions.shape[0]):
            if abs(positions[destination, 2] - positions[source, 2]) <= plan.z_cut:
                continue
            expected[destination] += (
                rayleigh_block(
                    plan=plan,
                    destination_index=destination,
                    source_index=source,
                )
                @ x[source]
            )

    actual = apply_rayleigh_far_numpy(plan, x)

    np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=3e-13)


def test_rayleigh_point_scan_matches_explicit_far_blocks() -> None:
    lmax = 2
    k = 2.0 * np.pi / 550.0
    source_positions = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [170.0, -80.0, 180.0],
            [-90.0, 60.0, 900.0],
        ]
    )
    points = np.asarray(
        [
            [20.0, 10.0, 620.0],
            [-30.0, 40.0, -420.0],
        ]
    )
    lattice = pcl.RectangularLattice2D(ax=900.0, ay=850.0)
    k_parallel = np.asarray([0.0002, -0.0001])
    z_cut = 300.0
    rng = np.random.default_rng(20260726)
    coeffs = rng.normal(size=(source_positions.shape[0], n_modes(lmax))) + 1j * rng.normal(
        size=(source_positions.shape[0], n_modes(lmax))
    )
    plan = build_rayleigh_plan(
        lmax=lmax,
        k=k,
        positions=source_positions,
        lattice=lattice,
        k_parallel=k_parallel,
        z_cut=z_cut,
        half_width=8,
        dtype=np.complex128,
    )

    actual = apply_rayleigh_far_to_points_numpy(plan, coeffs, points)

    combined = np.vstack([source_positions, points])
    combined_plan = build_rayleigh_plan(
        lmax=lmax,
        k=k,
        positions=combined,
        lattice=lattice,
        k_parallel=k_parallel,
        z_cut=z_cut,
        half_width=8,
        dtype=np.complex128,
    )
    l1_indices = [index_vswf(1, m, tau, lmax) for tau in (1, 2) for m in (-1, 0, 1)]
    expected = np.zeros_like(actual)
    for destination, point in enumerate(points):
        for source, source_position in enumerate(source_positions):
            if abs(float(point[2] - source_position[2])) <= z_cut:
                continue
            block = rayleigh_block(
                plan=combined_plan,
                destination_index=source_positions.shape[0] + destination,
                source_index=source,
            )
            expected[destination] += (block @ coeffs[source])[l1_indices]

    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)


def test_rayleigh_scan_accumulation_dtype_preserves_c64_storage() -> None:
    rng = np.random.default_rng(20260904)
    n_particles = 257
    z = np.sort(rng.uniform(-2200.0, 2200.0, size=n_particles))
    k = 2.0 * np.pi / 366.6666666666667
    gamma = np.asarray([0.8 * k + 0.0j, 0.015 * k + 0.0j], dtype=np.complex128)
    source = (
        rng.normal(size=(n_particles, 2, 2, 1)) + 1j * rng.normal(size=(n_particles, 2, 2, 1))
    ).astype(np.complex64)

    reference = rayleigh_module._scan_far_numpy(
        source_amplitudes=source.astype(np.complex128),
        z=z,
        gamma=gamma,
        z_cut=366.6666666666667,
        upward=True,
    ).astype(np.complex64)
    standard = rayleigh_module._scan_far_numpy(
        source_amplitudes=source,
        z=z,
        gamma=gamma,
        z_cut=366.6666666666667,
        upward=True,
    )
    wide = rayleigh_module._scan_far_numpy(
        source_amplitudes=source,
        z=z,
        gamma=gamma,
        z_cut=366.6666666666667,
        upward=True,
        accumulation_dtype=np.complex128,
    )

    assert wide.dtype == np.dtype(np.complex64)
    standard_error = np.linalg.norm(standard.astype(np.complex128) - reference)
    wide_error = np.linalg.norm(wide.astype(np.complex128) - reference)
    np.testing.assert_allclose(wide, reference, rtol=5e-7, atol=5e-7)
    assert wide_error < 0.05 * standard_error


def test_rayleigh_long_lived_scan_modes_follow_physical_damping() -> None:
    gamma = np.asarray(
        [
            0.01 + 0.0j,
            0.0 + 2.0e-4j,
            0.0 + 4.0e-3j,
        ],
        dtype=np.complex128,
    )
    indices = rayleigh_module._long_lived_scan_mode_indices(
        gamma,
        z_span=10_000.0,
        z_cut=366.0,
        storage_dtype=np.complex64,
    )
    np.testing.assert_array_equal(indices, np.asarray([0, 1], dtype=np.int64))
    assert (
        rayleigh_module._long_lived_scan_mode_indices(
            gamma,
            z_span=10_000.0,
            z_cut=366.0,
            storage_dtype=np.complex128,
        ).size
        == 0
    )


def test_hybrid_interior_local_l1_matches_exact_ewald() -> None:
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
    rng = np.random.default_rng(20260727)
    coeffs = rng.normal(size=(positions.shape[0], n_modes(lmax))) + 1j * rng.normal(
        size=(positions.shape[0], n_modes(lmax))
    )
    lattice = pcl.RectangularLattice2D(ax=900.0, ay=850.0)
    k_parallel = np.asarray([0.0003, -0.0002])
    shared: dict[str, Any] = dict(
        eta=0.002,
        real_shells=5,
        reciprocal_shells=8,
        shell_tolerance=1.0e-10,
        max_shells=32,
    )
    exact = PeriodicSpec(
        lattice=lattice,
        options=PeriodicOptions(method="ewald", **shared),
    )
    hybrid = PeriodicSpec(
        lattice=lattice,
        options=PeriodicOptions(
            method="rayleigh",
            rayleigh_z_cut=300.0,
            rayleigh_reciprocal_shells=16,
            **shared,
        ),
    )
    kwargs = dict(
        points=points,
        positions=positions,
        coeffs=coeffs,
        lmax=lmax,
        k=k,
        k_parallel=k_parallel,
        circumscribing_radii=np.full(positions.shape[0], 40.0),
    )

    expected = _periodic_local_regular_l1_coeffs(periodic=exact, **cast(Any, kwargs))
    actual = _periodic_local_regular_l1_coeffs(periodic=hybrid, **cast(Any, kwargs))

    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-11)


def test_rayleigh_nearfield_eta_preflight_uses_resolved_exact_band(monkeypatch) -> None:
    k = 2.0 * np.pi / 550.0
    positions = np.asarray([[0.0, 0.0, 0.0], [80.0, -30.0, 900.0]], dtype=float)
    points = np.asarray([[20.0, 10.0, 50.0]], dtype=float)
    coeffs = np.ones((positions.shape[0], n_modes(1)), dtype=np.complex128)
    periodic = PeriodicSpec(
        lattice=pcl.RectangularLattice2D(ax=900.0, ay=850.0),
        options=PeriodicOptions(
            method="rayleigh",
            eta=0.002,
            real_shells=4,
            reciprocal_shells=6,
            rayleigh_reciprocal_shells=12,
        ),
    )
    seen: list[float | None] = []
    original = periodic_interior_module.resolve_ewald_eta

    def capture_eta(**kwargs: Any) -> float:
        seen.append(kwargs.get("max_vertical_offset"))
        return float(original(**kwargs))

    monkeypatch.setattr(periodic_interior_module, "resolve_ewald_eta", capture_eta)
    _periodic_local_regular_l1_coeffs(
        points=points,
        positions=positions,
        coeffs=coeffs,
        lmax=1,
        k=k,
        periodic=periodic,
        k_parallel=np.zeros(2),
        circumscribing_radii=np.full(positions.shape[0], 320.0),
    )

    assert seen == [pytest.approx(640.0)]


def test_rayleigh_far_apply_preserves_complex64() -> None:
    k = 2.0 * np.pi / 550.0
    lattice = pcl.RectangularLattice2D(ax=900.0, ay=850.0)
    positions = np.asarray(
        [[0.0, 0.0, 0.0], [80.0, -30.0, 700.0], [-50.0, 60.0, 1400.0]],
        dtype=float,
    )
    common = dict(
        lmax=1,
        k=k,
        positions=positions,
        lattice=lattice,
        k_parallel=np.asarray([0.0004, -0.0002]),
        z_cut=550.0,
        half_width=8,
    )
    plan64 = build_rayleigh_plan(dtype=np.complex64, **cast(Any, common))
    plan128 = build_rayleigh_plan(dtype=np.complex128, **cast(Any, common))
    rng = np.random.default_rng(20260727)
    x = (
        rng.normal(size=(positions.shape[0], 6)) + 1j * rng.normal(size=(positions.shape[0], 6))
    ).astype(np.complex64)

    actual = apply_rayleigh_far_numpy(plan64, x)
    expected = apply_rayleigh_far_numpy(plan128, x.astype(np.complex128))

    assert actual.dtype == np.dtype(np.complex64)
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-6)


@pytest.mark.reference
def test_lmax3_rayleigh_block_matches_ewald_one_wavelength_off_plane() -> None:
    wavelength = 632.8
    k = 2.0 * np.pi / wavelength
    period = 7.447941497572213 * wavelength
    lattice = pcl.RectangularLattice2D(period, period)
    positions = np.asarray(
        [[0.0, 0.0, 0.0], [111.0, -233.0, wavelength]],
        dtype=float,
    )
    lmax = 3
    ab5 = translation_ab5_table(lmax, dtype=np.complex128)
    plan = build_rayleigh_plan(
        lmax=lmax,
        k=k,
        positions=positions,
        lattice=lattice,
        k_parallel=np.zeros(2),
        z_cut=0.75 * wavelength,
        half_width=32,
        dtype=np.complex128,
    )

    actual = rayleigh_block(plan=plan, destination_index=1, source_index=0)
    expected = periodic_ewald_block(
        lmax=lmax,
        k=k,
        destination=positions[1],
        source=positions[0],
        lattice=lattice,
        k_parallel=np.zeros(2),
        eta=0.0015,
        real_shells=20,
        reciprocal_shells=20,
        ab5=ab5,
        dtype=np.complex128,
    )

    relative_error = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
    assert relative_error < 2.0e-8


def _operator(
    *,
    positions: np.ndarray,
    method: Literal["ewald", "rayleigh"],
    z_cut: float | None = None,
    reciprocal_shells: int | None = None,
) -> PeriodicCouplingOperator:
    k = 2.0 * np.pi / 550.0
    lmax = 1
    return PeriodicCouplingOperator(
        lmax=lmax,
        k=k,
        positions=positions,
        ab5=translation_ab5_table(lmax, dtype=np.complex128),
        periodic=PeriodicSpec(
            lattice=pcl.RectangularLattice2D(ax=900.0, ay=850.0),
            options=PeriodicOptions(
                method=method,
                eta=0.002,
                real_shells=6,
                reciprocal_shells=6,
                rayleigh_z_cut=z_cut,
                rayleigh_reciprocal_shells=reciprocal_shells,
            ),
        ),
        k_parallel=np.zeros(2),
        dtype=np.dtype(np.complex128),
        circumscribing_radii=np.full(positions.shape[0], 40.0),
    )


def test_hybrid_operator_is_exact_when_every_pair_is_in_near_band() -> None:
    positions = np.asarray(
        [[0.0, 0.0, 0.0], [100.0, 30.0, 100.0], [40.0, -20.0, 300.0]],
        dtype=float,
    )
    exact = _operator(positions=positions, method="ewald")
    hybrid = _operator(
        positions=positions,
        method="rayleigh",
        z_cut=500.0,
        reciprocal_shells=4,
    )
    rng = np.random.default_rng(4)
    x = rng.normal(size=18) + 1j * rng.normal(size=18)

    np.testing.assert_allclose(hybrid.apply(x), exact.apply(x), rtol=2e-13, atol=2e-13)
    assert hybrid._near_destinations is not None
    assert hybrid._near_destinations.size == positions.shape[0] * (positions.shape[0] - 1)
    assert hybrid._self_block_cache is not None


def test_complex64_hybrid_stores_near_structural_sums_in_compute_dtype() -> None:
    positions = np.asarray(
        [[0.0, 0.0, 0.0], [100.0, 30.0, 100.0], [40.0, -20.0, 900.0]],
        dtype=float,
    )
    k = 2.0 * np.pi / 550.0
    lmax = 1
    hybrid = PeriodicCouplingOperator(
        lmax=lmax,
        k=k,
        positions=positions,
        ab5=translation_ab5_table(lmax, dtype=np.complex64),
        periodic=PeriodicSpec(
            lattice=pcl.RectangularLattice2D(ax=900.0, ay=850.0),
            options=PeriodicOptions(
                method="rayleigh",
                eta=0.002,
                real_shells=6,
                reciprocal_shells=6,
                rayleigh_z_cut=550.0,
                rayleigh_reciprocal_shells=8,
            ),
        ),
        k_parallel=np.zeros(2),
        dtype=np.dtype(np.complex64),
        circumscribing_radii=np.full(positions.shape[0], 40.0),
    )

    hybrid.populate()

    assert hybrid._near_structural_sums is not None
    assert hybrid._near_structural_sums.dtype == np.dtype(np.complex64)
    assert hybrid._near_contraction_tensor_cache is not None
    assert hybrid._ewald_shell_workspace is None
    assert hybrid._structural_contraction_tensor is None


def test_hybrid_operator_uses_rayleigh_only_for_far_pairs() -> None:
    positions = np.asarray(
        [[0.0, 0.0, 0.0], [100.0, 30.0, 100.0], [40.0, -20.0, 900.0]],
        dtype=float,
    )
    exact = _operator(positions=positions, method="ewald")
    hybrid = _operator(
        positions=positions,
        method="rayleigh",
        z_cut=550.0,
    )
    rng = np.random.default_rng(8)
    x = rng.normal(size=(18, 2)) + 1j * rng.normal(size=(18, 2))

    actual = hybrid.apply(x)
    expected = np.column_stack([exact.apply(x[:, column]) for column in range(x.shape[1])])

    np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-11)
    assert hybrid._near_destinations is not None
    assert hybrid._near_destinations.size == 2
    assert hybrid._near_structural_sums is not None
    assert hybrid._near_structural_sums.shape == (2, 9)
    assert hybrid._rayleigh_plan().half_width > 0


def test_vertically_sparse_hybrid_prepares_self_ewald_only_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    positions = np.asarray(
        [[0.0, 0.0, 0.0], [40.0, 20.0, 700.0], [-30.0, 60.0, 1400.0]],
        dtype=float,
    )
    hybrid = _operator(
        positions=positions,
        method="rayleigh",
        z_cut=550.0,
        reciprocal_shells=12,
    )
    original = coupling_periodic_module.ewald_structural_sums_2d_batch
    calls = 0

    def counting_ewald(**kwargs: Any) -> np.ndarray:
        nonlocal calls
        calls += 1
        return original(**kwargs)

    monkeypatch.setattr(
        coupling_periodic_module,
        "ewald_structural_sums_2d_batch",
        counting_ewald,
    )
    x = np.arange(18, dtype=float).astype(np.complex128)

    hybrid.apply(x)
    hybrid.apply(x)

    assert calls == 1
    assert hybrid._near_destinations is not None
    assert hybrid._near_destinations.size == 0


def _allocator_snapshot(*, available: int, guaranteed_fresh: int) -> CuPyAllocatorSnapshot:
    return CuPyAllocatorSnapshot(
        raw_free_bytes=available,
        raw_total_bytes=8 * 1024**3,
        pool_used_bytes=0,
        pool_total_bytes=0,
        pool_free_bytes=0,
        pool_limit_bytes=available,
        effective_device_limit_bytes=available,
        active_headroom_bytes=available,
        guaranteed_fresh_allocation_bytes=guaranteed_fresh,
        pool_limit_applied=False,
        pool_trimmed_to_limit=False,
        pool_trimmed_for_fragmentation=False,
    )


def test_rayleigh_near_cache_estimate_honors_compute_dtype() -> None:
    estimate64 = rayleigh_near_cache_estimate(pair_count=3_633_788, lmax=3, dtype=np.complex64)
    estimate128 = rayleigh_near_cache_estimate(pair_count=3_633_788, lmax=3, dtype=np.complex128)

    assert estimate64.structural_channels == 49
    assert estimate64.structural_bytes == 3_633_788 * 49 * 8
    assert estimate128.structural_bytes == 2 * estimate64.structural_bytes


def test_rayleigh_near_cache_memory_plan_rejects_guarded_device_oversubscription() -> None:
    estimate = rayleigh_near_cache_estimate(pair_count=4_000_000, lmax=3, dtype=np.complex64)
    remaining = 512 * 1024**2
    available = 2 * 1024**3

    with pytest.raises(MemoryError, match="does not automatically spill"):
        _resolve_rayleigh_near_cache_memory_plan(
            estimate=estimate,
            snapshot=_allocator_snapshot(
                available=available,
                guaranteed_fresh=available,
            ),
            remaining_rayleigh_device_bytes=remaining,
        )


@pytest.mark.fake_gpu
def test_cupy_rayleigh_release_drops_only_ewald_preparation_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = CuPyPeriodicCouplingOperator(
        lmax=1,
        k=2.0 * np.pi / 550.0,
        positions=np.zeros((1, 3), dtype=float),
        ab5=translation_ab5_table(1, dtype=np.complex64),
        periodic=PeriodicSpec(
            lattice=pcl.RectangularLattice2D(900.0, 850.0),
            options=PeriodicOptions(
                method="rayleigh",
                eta=0.0015,
                real_shells=1,
                reciprocal_shells=1,
                rayleigh_z_cut=550.0,
                rayleigh_reciprocal_shells=1,
            ),
        ),
        k_parallel=np.zeros(2),
        dtype=np.dtype(np.complex64),
        circumscribing_radii=np.asarray([40.0]),
    )
    compact_sparse = (
        np.asarray([0, 1], dtype=np.int64),
        np.asarray([0], dtype=np.int32),
        np.asarray([0], dtype=np.int32),
        np.asarray([1.0 + 0.0j], dtype=np.complex64),
    )
    monkeypatch.setattr(operator, "_near_sparse_contraction_device", lambda: compact_sparse)
    operator._workspace = cast(Any, object())
    operator._contraction_tensor_gpu = cast(Any, object())
    operator._self_correction_gpu = cast(Any, object())
    operator._self_block_gpu = np.ones((1, 1), dtype=np.complex64)
    operator._near_structural_sums_gpu = np.ones((1, 1), dtype=np.complex64)

    operator._release_rayleigh_preparation_state()

    assert operator._workspace is None
    assert operator._contraction_tensor_gpu is None
    assert operator._self_correction_gpu is None
    assert operator._self_block_gpu is not None
    assert operator._near_structural_sums_gpu is not None


def test_rayleigh_near_cache_memory_plan_keeps_small_cache_on_device() -> None:
    estimate = rayleigh_near_cache_estimate(pair_count=20_000, lmax=3, dtype=np.complex64)
    available = 2 * 1024**3

    plan = _resolve_rayleigh_near_cache_memory_plan(
        estimate=estimate,
        snapshot=_allocator_snapshot(
            available=available,
            guaranteed_fresh=available,
        ),
        remaining_rayleigh_device_bytes=128 * 1024**2,
    )

    assert plan.required_device_bytes <= plan.available_device_bytes


@pytest.mark.fake_gpu
def test_cupy_near_apply_batch_is_bounded_by_structural_row_size() -> None:
    positions = np.zeros((700, 3), dtype=float)
    operator = CuPyPeriodicCouplingOperator(
        lmax=3,
        k=2.0 * np.pi / 550.0,
        positions=positions,
        ab5=translation_ab5_table(3, dtype=np.complex64),
        periodic=PeriodicSpec(
            lattice=pcl.RectangularLattice2D(6000.0, 6000.0),
            options=PeriodicOptions(
                method="rayleigh",
                eta=0.0015,
                real_shells=1,
                reciprocal_shells=1,
                rayleigh_z_cut=550.0,
                rayleigh_reciprocal_shells=1,
            ),
        ),
        k_parallel=np.zeros(2),
        dtype=np.dtype(np.complex64),
        circumscribing_radii=np.full(positions.shape[0], 40.0),
    )

    total = 2_000_000
    batch = operator._near_apply_batch_size(total=total)
    expected = (256 * 1024**2) // (((2 * operator.lmax + 1) ** 2) * operator.dtype.itemsize)

    assert batch == expected
    assert 1 <= batch < total


@pytest.mark.reference
def test_lmax3_hybrid_full_action_matches_exact_ewald() -> None:
    wavelength = 632.8
    k = 2.0 * np.pi / wavelength
    lmax = 3
    positions = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [100.0, 20.0, 150.0],
            [-200.0, 50.0, 700.0],
            [70.0, -80.0, 1400.0],
        ],
        dtype=float,
    )
    lattice = pcl.RectangularLattice2D(1500.0, 1400.0)

    def make_operator(options: PeriodicOptions) -> PeriodicCouplingOperator:
        return PeriodicCouplingOperator(
            lmax=lmax,
            k=k,
            positions=positions,
            ab5=translation_ab5_table(lmax, dtype=np.complex128),
            periodic=PeriodicSpec(lattice=lattice, options=options),
            k_parallel=np.asarray([0.0002, -0.0001]),
            dtype=np.dtype(np.complex128),
            cache_blocks=False,
            circumscribing_radii=np.full(positions.shape[0], 60.0),
        )

    exact = make_operator(
        PeriodicOptions(
            method="ewald",
            eta=0.0015,
            real_shells=8,
            reciprocal_shells=8,
        )
    )
    hybrid = make_operator(
        PeriodicOptions(
            method="rayleigh",
            eta=0.0015,
            real_shells=8,
            reciprocal_shells=8,
            rayleigh_z_cut=wavelength,
            rayleigh_reciprocal_shells=18,
        )
    )
    rng = np.random.default_rng(20260726)
    x = rng.normal(size=positions.shape[0] * 30) + 1j * rng.normal(size=positions.shape[0] * 30)

    actual = hybrid.apply(x)
    expected = exact.apply(x)

    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-12)
    assert hybrid._near_structural_sums is not None
    assert hybrid._near_structural_sums.shape[1] == 49


def test_prepare_matvec_passes_particle_radii_to_default_rayleigh_band() -> None:
    wavelength = 550.0
    positions = np.asarray([[0.0, 0.0, 0.0], [40.0, 10.0, 900.0]], dtype=float)
    particles = pcl.spheres_from_arrays(
        positions=positions,
        radii=np.asarray([40.0, 60.0]),
        refractive_indices=np.asarray([1.5 + 0.0j, 1.4 + 0.0j]),
    )
    prepared = prepare_matvec(
        lmax=1,
        k=2.0 * np.pi / wavelength,
        particles=particles,
        n_medium=1.0 + 0.0j,
        radial_lut_dr=1.0,
        cache_translation_blocks=False,
        operator_dtype=np.complex64,
        accum_dtype=np.complex128,
        periodic=PeriodicSpec(
            lattice=pcl.RectangularLattice2D(900.0, 850.0),
            options=PeriodicOptions(
                method="rayleigh",
                eta=0.002,
                real_shells=4,
                reciprocal_shells=4,
                rayleigh_reciprocal_shells=8,
            ),
        ),
        k_parallel=np.zeros(2),
        backend="numpy",
        show_progress=False,
    )

    assert isinstance(prepared.coupling, PeriodicCouplingOperator)
    assert prepared.coupling.accum_dtype == np.dtype(np.complex128)
    assert prepared.coupling._rayleigh_plan().z_cut == pytest.approx(wavelength)


def test_rayleigh_cartesian_numpy_path_matches_pointwise_fallback() -> None:
    lmax = 2
    k = 2.0 * np.pi / 550.0
    positions = np.asarray([[0.0, 0.0, 0.0], [170.0, -80.0, 140.0], [-90.0, 60.0, 820.0]])
    xy = np.asarray([[20.0, 10.0], [-30.0, 40.0], [55.0, -25.0]])
    z = np.asarray([-420.0, 500.0, 900.0])
    points = np.asarray([[x, y, zz] for zz in z for x, y in xy], dtype=float)
    points = points[[5, 0, 7, 2, 8, 1, 3, 6, 4]]
    rng = np.random.default_rng(20260901)
    coeffs = rng.normal(size=(positions.shape[0], n_modes(lmax))) + 1j * rng.normal(
        size=(positions.shape[0], n_modes(lmax))
    )
    plan = build_rayleigh_plan(
        lmax=lmax,
        k=k,
        positions=positions,
        lattice=pcl.RectangularLattice2D(ax=900.0, ay=850.0),
        k_parallel=np.asarray([0.0002, -0.0001]),
        z_cut=300.0,
        half_width=8,
        dtype=np.complex128,
    )

    fast = apply_rayleigh_far_to_points_numpy(plan, coeffs, points)
    # Duplicating one point breaks the Cartesian-product uniqueness condition and
    # forces the generic pointwise path without changing the first n outputs.
    fallback = apply_rayleigh_far_to_points_numpy(
        plan,
        coeffs,
        np.vstack([points, points[0]]),
    )[: points.shape[0]]

    np.testing.assert_allclose(fast, fallback, rtol=2e-12, atol=2e-12)
