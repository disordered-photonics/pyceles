from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

import pyceles as pcl
from pyceles._optional import asnumpy
from pyceles.core.indexing import n_modes
from pyceles.core.operators import (
    CuPyMLFMMCouplingOperator,
    MLFMMCouplingOperator,
    build_mlfmm_cupy_host_cache,
    prepare_matvec,
)
from pyceles.core.operators.coupling_periodic import PeriodicCouplingOperator
from pyceles.core.operators.mlfmm import (
    MLFMMOptions,
    _exact_periodic_leaf_near_apply,
    _sampled_rokhlin_translator,
)
from pyceles.core.operators.mlfmm_directional import directional_transforms
from pyceles.core.operators.mlfmm_periodic import (
    _periodic_nonzero_structural_sums,
    prepare_periodized_mlfmm_coupling,
    sampled_transfer_from_structural_sums,
)
from pyceles.core.periodic.ewald import (
    ewald_self_correction,
    ewald_structural_sums_2d_batch,
)
from pyceles.core.periodic.scalar import structural_sum_m_normalization
from pyceles.core.periodic.structural import free_space_structural_sums
from pyceles.core.translation import RadialLUT, translation_ab5_table


def _periodic_fixture() -> tuple[np.ndarray, np.ndarray, float, np.ndarray, pcl.PeriodicSpec]:
    positions = np.array(
        [
            [-110.0, -90.0, -35.0],
            [-95.0, 105.0, 40.0],
            [100.0, -100.0, 55.0],
            [115.0, 95.0, -45.0],
        ],
        dtype=float,
    )
    radii = np.ones((positions.shape[0],), dtype=float)
    k = 2.0 * np.pi / 550.0
    k_parallel = np.array([4.0e-4, -2.0e-4], dtype=float)
    periodic = pcl.PeriodicSpec(
        lattice=pcl.RectangularLattice2D(300.0, 320.0),
        options=pcl.PeriodicOptions(
            method="ewald",
            eta=2.5e-3,
            real_shells=6,
            reciprocal_shells=6,
        ),
    )
    return positions, radii, k, k_parallel, periodic


def _radial_lut(*, positions: np.ndarray, lmax: int, k: float) -> RadialLUT:
    separations = positions[:, None, :] - positions[None, :, :]
    return RadialLUT(
        lmax=int(lmax),
        k=float(k),
        r_max=float(np.max(np.linalg.norm(separations, axis=2), initial=1.0)),
        dr=0.25,
        dtype=np.complex128,
    )


def _reference_periodic_operator(
    *,
    lmax: int,
    k: float,
    positions: np.ndarray,
    radii: np.ndarray,
    periodic: pcl.PeriodicSpec,
    k_parallel: np.ndarray,
) -> PeriodicCouplingOperator:
    return PeriodicCouplingOperator(
        lmax=int(lmax),
        k=float(k),
        positions=positions,
        ab5=translation_ab5_table(int(lmax), dtype=np.complex128),
        periodic=periodic,
        k_parallel=k_parallel,
        dtype=np.dtype(np.complex128),
        circumscribing_radii=radii,
    )


def test_structural_sum_map_matches_sampled_rokhlin_translator() -> None:
    k = 2.0 * np.pi / 550.0
    displacement = np.array([350.0, -220.0, 150.0], dtype=float)
    order = 6
    directional = directional_transforms(order, grid_order=order)

    expected = _sampled_rokhlin_translator(
        displacement,
        k=complex(k),
        truncation_order=order,
        directions=directional.grid.directions,
        weights=directional.grid.weights,
        dtype=np.dtype(np.complex128),
    )
    structural = free_space_structural_sums(
        max_degree=order,
        k=k,
        displacement=displacement,
    )
    actual = sampled_transfer_from_structural_sums(
        structural,
        directions=directional.grid.directions,
        weights=directional.grid.weights,
    )

    np.testing.assert_allclose(actual, expected, rtol=2.0e-13, atol=2.0e-13)


def test_periodic_box_closure_regularizes_lattice_equivalent_centers() -> None:
    lattice = pcl.RectangularLattice2D(300.0, 320.0)
    k_parallel = np.array([4.0e-4, -2.0e-4], dtype=float)
    k = 2.0 * np.pi / 550.0
    eta = 2.5e-3
    periodic = pcl.PeriodicSpec(
        lattice=lattice,
        options=pcl.PeriodicOptions(
            method="ewald",
            eta=eta,
            real_shells=8,
            reciprocal_shells=8,
        ),
    )
    offset = (1, 0, 0)
    displacement = np.array([[lattice.ax, 0.0, 0.0]], dtype=float)

    actual_by_offset, singular_by_offset, _eta = _periodic_nonzero_structural_sums(
        max_degree=4,
        k=k,
        positions=np.zeros((1, 3), dtype=float),
        periodic=periodic,
        k_parallel=k_parallel,
        offsets=[offset],
        displacements=displacement,
    )

    self_raw = ewald_structural_sums_2d_batch(
        lmax_struct=2,
        k=k,
        destinations=np.zeros((1, 3), dtype=float),
        source=np.zeros((3,), dtype=float),
        lattice=lattice,
        k_parallel=k_parallel,
        eta=eta,
        real_shells=8,
        reciprocal_shells=8,
        dtype=np.complex128,
    )[0]
    self_raw = np.asarray(self_raw, dtype=np.complex128).copy()
    self_raw[0, 4] += structural_sum_m_normalization(0) * ewald_self_correction(k, eta)
    phase = np.exp(1j * k_parallel[0] * lattice.ax)
    expected = phase * self_raw - free_space_structural_sums(
        max_degree=4,
        k=k,
        displacement=displacement[0],
    )

    assert singular_by_offset[offset] == (1, 0)
    np.testing.assert_allclose(actual_by_offset[offset], expected, rtol=3.0e-13, atol=3.0e-13)


def test_one_particle_periodized_mlfmm_matches_ewald_self_images() -> None:
    positions = np.array([[0.1, -0.2, 0.3]], dtype=float)
    radii = np.array([0.01], dtype=float)
    lmax = 1
    k = 2.0 * np.pi / 550.0
    k_parallel = np.array([7.0e-4, -4.0e-4], dtype=float)
    periodic = pcl.PeriodicSpec(
        lattice=pcl.RectangularLattice2D(900.0, 800.0),
        options=pcl.PeriodicOptions(
            method="ewald",
            eta=2.5e-3,
            real_shells=8,
            reciprocal_shells=8,
        ),
    )
    x = np.arange(n_modes(lmax), dtype=float).astype(np.complex128)
    x += 1j * np.arange(n_modes(lmax), dtype=float)[::-1]
    lut = RadialLUT(lmax=lmax, k=k, r_max=1.0, dr=0.01, dtype=np.complex128)

    actual_operator = prepare_periodized_mlfmm_coupling(
        lmax=lmax,
        k=k,
        positions=positions,
        particle_circumscribing_radii=radii,
        radial_lut=lut,
        periodic=periodic,
        k_parallel=k_parallel,
        options=MLFMMOptions(
            max_leaf_particles=8,
            max_depth=0,
            leaf_size_radius_factor=1.0,
            accuracy_level=1,
        ),
        box_order=2,
    )
    reference_operator = _reference_periodic_operator(
        lmax=lmax,
        k=k,
        positions=positions,
        radii=radii,
        periodic=periodic,
        k_parallel=k_parallel,
    )

    np.testing.assert_allclose(
        actual_operator.apply(x),
        reference_operator.apply(x),
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    assert actual_operator.periodization is not None
    assert actual_operator.periodization.exact_particle_pair_count == 0


def test_periodized_mlfmm_converges_to_pairwise_ewald_with_near_images() -> None:
    positions, radii, k, k_parallel, periodic = _periodic_fixture()
    lmax = 1
    nm = n_modes(lmax)
    rng = np.random.default_rng(5)
    x = rng.normal(size=positions.shape[0] * nm) + 1j * rng.normal(size=positions.shape[0] * nm)
    lut = _radial_lut(positions=positions, lmax=lmax, k=k)
    options = MLFMMOptions(
        max_leaf_particles=1,
        max_depth=2,
        leaf_size_radius_factor=1.0,
        accuracy_level=1,
    )
    reference = _reference_periodic_operator(
        lmax=lmax,
        k=k,
        positions=positions,
        radii=radii,
        periodic=periodic,
        k_parallel=k_parallel,
    ).apply(x)

    order4 = prepare_periodized_mlfmm_coupling(
        lmax=lmax,
        k=k,
        positions=positions,
        particle_circumscribing_radii=radii,
        radial_lut=lut,
        periodic=periodic,
        k_parallel=k_parallel,
        options=options,
        box_order=4,
    )
    order8 = prepare_periodized_mlfmm_coupling(
        lmax=lmax,
        k=k,
        positions=positions,
        particle_circumscribing_radii=radii,
        radial_lut=lut,
        periodic=periodic,
        k_parallel=k_parallel,
        options=options,
        box_order=8,
    )
    error4 = np.linalg.norm(order4.apply(x) - reference) / np.linalg.norm(reference)
    error8 = np.linalg.norm(order8.apply(x) - reference) / np.linalg.norm(reference)

    assert order8.periodization is not None
    assert order8.periodization.near_image_count > 0
    assert order8.periodization.exact_leaf_box_pair_count > 0
    assert order8.periodization.exact_particle_pair_count > 0
    assert error8 < 2.0e-4
    assert error8 < error4 / 20.0

    hierarchy = order8.hierarchy_diagnostics()
    memory = order8.memory_diagnostics()
    assert hierarchy["periodization"] == order8.periodization.summary()
    periodization_memory = memory.get("periodization")
    assert isinstance(periodization_memory, dict)
    assert periodization_memory["total_bytes"] > 0


def test_periodic_exact_leaf_apply_accumulates_into_supplied_output() -> None:
    positions, radii, k, k_parallel, periodic = _periodic_fixture()
    lmax = 1
    coupling = prepare_periodized_mlfmm_coupling(
        lmax=lmax,
        k=k,
        positions=positions,
        particle_circumscribing_radii=radii,
        radial_lut=_radial_lut(positions=positions, lmax=lmax, k=k),
        periodic=periodic,
        k_parallel=k_parallel,
        options=MLFMMOptions(
            max_leaf_particles=1,
            max_depth=2,
            leaf_size_radius_factor=1.0,
            accuracy_level=1,
        ),
        box_order=6,
    )
    assert coupling.multilevel is not None
    assert coupling.periodization is not None
    assert coupling.periodization.leaf_batches

    nm = n_modes(lmax)
    x = np.arange(positions.shape[0] * nm, dtype=float).astype(np.complex128)
    expected = _exact_periodic_leaf_near_apply(
        lmax=lmax,
        k=k,
        positions=positions,
        x=x,
        partition=coupling.multilevel.partition,
        batches=coupling.periodization.leaf_batches,
        radial_lut=coupling.radial_lut,
        dtype=np.dtype(np.complex128),
    )
    out = np.full_like(x, 2.0 - 3.0j)
    actual = _exact_periodic_leaf_near_apply(
        lmax=lmax,
        k=k,
        positions=positions,
        x=x,
        partition=coupling.multilevel.partition,
        batches=coupling.periodization.leaf_batches,
        radial_lut=coupling.radial_lut,
        dtype=np.dtype(np.complex128),
        out=out,
    )

    assert np.shares_memory(actual, out)
    np.testing.assert_allclose(actual, expected + (2.0 - 3.0j), rtol=2.0e-15, atol=2.0e-15)


def test_prepare_matvec_accepts_numpy_periodic_mlfmm() -> None:
    positions, _radii, k, k_parallel, periodic = _periodic_fixture()
    particles = pcl.spheres_from_arrays(
        positions=positions,
        radii=np.full((positions.shape[0],), 5.0, dtype=float),
        refractive_indices=1.5 + 0.0j,
    )

    prepared = prepare_matvec(
        lmax=1,
        k=k,
        particles=particles,
        n_medium=1.0 + 0.0j,
        radial_lut_dr=0.5,
        coupling_backend="mlfmm",
        mlfmm_options=MLFMMOptions(
            max_leaf_particles=1,
            max_depth=1,
            leaf_size_radius_factor=1.0,
            accuracy_level=1,
            order_additive=0,
        ),
        periodic=periodic,
        k_parallel=k_parallel,
        backend="numpy",
    )

    assert isinstance(prepared.coupling, MLFMMCouplingOperator)
    assert prepared.coupling.periodization is not None
    result = prepared.apply_W(np.ones((positions.shape[0] * n_modes(1),), dtype=np.complex128))
    assert result.shape == (positions.shape[0] * n_modes(1),)


@pytest.mark.fake_gpu
def test_periodized_mlfmm_cupy_host_staging_is_compact_and_precision_aligned() -> None:
    positions, radii, k, k_parallel, periodic = _periodic_fixture()
    lmax = 1
    coupling = prepare_periodized_mlfmm_coupling(
        lmax=lmax,
        k=k,
        positions=positions,
        particle_circumscribing_radii=radii,
        radial_lut=_radial_lut(positions=positions, lmax=lmax, k=k),
        periodic=periodic,
        k_parallel=k_parallel,
        options=MLFMMOptions(
            max_leaf_particles=1,
            max_depth=2,
            leaf_size_radius_factor=1.0,
            accuracy_level=1,
        ),
        dtype=np.dtype(np.complex64),
        box_order=6,
        leaf_map_backend="cupy",
        build_leaf_maps=False,
    )
    assert coupling.multilevel is not None
    assert coupling.periodization is not None
    assert coupling.periodization.leaf_batches
    assert len(coupling.multilevel.aggregation) == 0
    assert len(coupling.multilevel.receive) == 0

    leaves = coupling.multilevel.partition.leaves
    actual_r_max = 0.0
    for batch in coupling.periodization.leaf_batches:
        shift = np.asarray(batch.lattice_shift, dtype=float).reshape(1, 1, 3)
        for source_leaf, destination_leaf in zip(
            batch.source_leaf_indices, batch.destination_leaf_indices, strict=True
        ):
            source_particles = leaves[int(source_leaf)].particle_indices
            destination_particles = leaves[int(destination_leaf)].particle_indices
            displacement = (
                positions[destination_particles, None, :]
                - positions[None, source_particles, :]
                - shift
            )
            actual_r_max = max(
                actual_r_max,
                float(np.max(np.linalg.norm(displacement, axis=2), initial=0.0)),
            )
    assert coupling.periodization.exact_leaf_r_max_bound >= actual_r_max
    assert float(coupling.radial_lut.r_grid[-1]) >= coupling.periodization.exact_leaf_r_max_bound

    host_cache = build_mlfmm_cupy_host_cache(coupling)
    assert host_cache.near_lut_re.dtype == np.dtype(np.float32)
    assert host_cache.near_lut_im.dtype == np.dtype(np.float32)
    expected_h = np.asarray(coupling.radial_lut.h).T.reshape(-1)
    np.testing.assert_array_equal(host_cache.near_lut_re, expected_h.real.astype(np.float32))
    np.testing.assert_array_equal(host_cache.near_lut_im, expected_h.imag.astype(np.float32))
    assert host_cache.multilevel is not None
    assert host_cache.multilevel.aggregation is None
    assert host_cache.multilevel.leaf_groups_otf is not None
    assert host_cache.multilevel.leaf_translation_tables is not None
    assert host_cache.multilevel.leaf_translation_tables.last_index < host_cache.near_last_index
    assert host_cache.periodization is not None
    assert host_cache.periodization.summary == coupling.periodization.summary()

    far_batches = [
        batch for batches in host_cache.periodization.far_batches_by_level for batch in batches
    ]
    assert far_batches
    assert all(batch.diagonal.dtype == np.dtype(np.complex128) for batch in far_batches)
    assert all(batch.source_indices.dtype == np.dtype(np.int32) for batch in far_batches)
    assert all(batch.destination_indices.dtype == np.dtype(np.int32) for batch in far_batches)
    assert all(
        batch.source_leaf_indices.dtype == np.dtype(np.int32)
        and batch.destination_leaf_indices.dtype == np.dtype(np.int32)
        for batch in host_cache.periodization.leaf_batches
    )


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("operator_dtype", "rtol", "atol"),
    [
        (np.complex64, 1.0e-4, 1.0e-5),
        (np.complex128, 2.0e-9, 2.0e-10),
    ],
)
def test_prepare_matvec_cupy_periodized_mlfmm_matches_numpy(
    cupy_runtime: tuple[Any, Any],
    operator_dtype: npt.DTypeLike,
    rtol: float,
    atol: float,
) -> None:
    cupy, _ = cupy_runtime
    positions, _radii, k, k_parallel, periodic = _periodic_fixture()
    particles = pcl.spheres_from_arrays(
        positions=positions,
        radii=np.full((positions.shape[0],), 5.0, dtype=float),
        refractive_indices=1.5 + 0.0j,
    )
    options = MLFMMOptions(
        max_leaf_particles=1,
        max_depth=2,
        leaf_size_radius_factor=1.0,
        accuracy_level=1,
        order_additive=0,
    )
    common: dict[str, Any] = dict(
        lmax=1,
        k=k,
        particles=particles,
        n_medium=1.0 + 0.0j,
        radial_lut_dr=0.25,
        cache_translation_blocks=False,
        operator_dtype=operator_dtype,
        coupling_backend="mlfmm",
        mlfmm_options=options,
        periodic=periodic,
        k_parallel=k_parallel,
    )
    prepared_numpy = prepare_matvec(**common, backend="numpy")
    prepared_cupy = prepare_matvec(**common, backend="cupy")

    assert isinstance(prepared_numpy.coupling, MLFMMCouplingOperator)
    assert prepared_numpy.coupling.periodization is not None
    assert isinstance(prepared_cupy.coupling, CuPyMLFMMCouplingOperator)
    assert prepared_cupy.coupling.prepared_data.periodization is not None
    assert prepared_cupy.coupling.prepared_data.periodization.leaf_batches

    rng = np.random.default_rng(20260728)
    n_unknowns = positions.shape[0] * n_modes(1)
    x = np.asarray(
        rng.normal(size=(n_unknowns, 2)) + 1j * rng.normal(size=(n_unknowns, 2)),
        dtype=operator_dtype,
    )
    # The NumPy MLFMM reference currently accepts one flat RHS at a time;
    # retain the block-RHS CuPy coverage by assembling its reference columns.
    expected = np.column_stack(
        [np.asarray(prepared_numpy.apply_W(x[:, j]), dtype=operator_dtype) for j in range(2)]
    )
    actual = np.asarray(
        asnumpy(prepared_cupy.apply_W(cupy.asarray(x))),
        dtype=operator_dtype,
    )
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)

    hierarchy = prepared_cupy.coupling.hierarchy_diagnostics()
    periodization = hierarchy.get("periodization")
    assert isinstance(periodization, dict)
    assert int(periodization["sampled_far_box_pair_count"]) > 0
    memory = prepared_cupy.coupling.memory_diagnostics()
    periodization_memory = memory.get("periodization")
    assert isinstance(periodization_memory, dict)
    assert int(periodization_memory["total_bytes"]) > 0
