from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from pyceles.core.indexing import iter_modes, n_modes
from pyceles.core.periodic.reception_cupy import (
    _periodic_near_pair_batch_size_for_workspace,
    _periodic_near_pair_workspace_bytes_per_pair,
)
from pyceles.core.periodic.structural import rectangular_projection_data
from pyceles.core.translation import translation_ab5_table


def _dense_l1_kernel(lmax):
    """Independent dense mode/channel selection for the rectangular contraction."""
    modes = tuple(iter_modes(lmax))
    rows = [(m, index) for _, degree, m, index in modes if degree == 1]
    order = 2 * ((lmax + 2) // 2)
    p_count = lmax + 2
    table = translation_ab5_table(lmax, dtype=np.complex128)
    kernel = np.zeros((6, n_modes(lmax), p_count, 2 * order + 1), dtype=complex)
    for row, (m_dst, dst) in enumerate(rows):
        for _, _, m_src, src in modes:
            kernel[row, src, :, m_src - m_dst + order] = table[dst, src, :p_count]
    return order, kernel


@pytest.mark.parametrize("lmax", [1, 2, 3, 4, 5, 8])
def test_compact_l1_projection_matches_dense_reference(lmax: int) -> None:
    rng = np.random.default_rng(1800 + lmax)
    maximum_order, dense_kernel = _dense_l1_kernel(lmax)
    structural_order = lmax + 1
    degrees, orders, indptr, inputs, values = rectangular_projection_data(1, lmax)
    structural = rng.normal(size=(7, maximum_order + 1, 2 * maximum_order + 1)) + 1j * rng.normal(
        size=(7, maximum_order + 1, 2 * maximum_order + 1)
    )
    # Invalid (p,m) slots do not occur in an Ewald table. Zero them so the
    # reference and compact representations compare the same mathematical data.
    for degree in range(maximum_order + 1):
        for column in range(2 * maximum_order + 1):
            order_m = column - maximum_order
            if abs(order_m) > degree or degree > structural_order:
                structural[:, degree, column] = 0.0
    coeffs = rng.normal(size=n_modes(lmax)) + 1j * rng.normal(size=n_modes(lmax))

    want = np.einsum(
        "dpm,rcpm,c->dr", structural[:, : lmax + 2], dense_kernel, coeffs, optimize=True
    )
    compact_structural = structural[
        :,
        : structural_order + 1,
        maximum_order - structural_order : maximum_order + structural_order + 1,
    ]
    projection = csr_matrix((values, inputs, indptr), shape=(len(degrees) * 6, n_modes(lmax)))
    weighted_source = (projection @ coeffs).reshape(len(degrees), 6)
    got = compact_structural[:, degrees, orders] @ weighted_source
    np.testing.assert_allclose(got, want, rtol=5e-13, atol=5e-13)


@pytest.mark.parametrize("lmax", [2, 3, 4, 5, 8])
def test_l1_translation_has_no_support_above_triangle_limit(lmax: int) -> None:
    """The compact p<=lmax+1 cutoff follows the exact Wigner triangle rule."""
    l1_rows = [index for _, degree, _, index in iter_modes(lmax) if degree == 1]
    ab5 = translation_ab5_table(lmax, dtype=np.complex128)
    unsupported = np.asarray(ab5[l1_rows, :, lmax + 2 :])
    np.testing.assert_array_equal(unsupported, np.zeros_like(unsupported))


@pytest.mark.parametrize("projection_dtype", [np.complex64, np.complex128])
def test_periodic_near_pair_batch_planner_respects_budget(projection_dtype: Any) -> None:
    per_pair = _periodic_near_pair_workspace_bytes_per_pair(
        lmax=4,
        projection_dtype=np.dtype(projection_dtype),
    )
    batch = _periodic_near_pair_batch_size_for_workspace(
        total_pairs=1_000_000,
        lmax=4,
        projection_dtype=np.dtype(projection_dtype),
        workspace_bytes=100 * per_pair,
    )
    assert batch == 100


def test_periodic_near_pair_batch_planner_caps_large_workspaces() -> None:
    batch = _periodic_near_pair_batch_size_for_workspace(
        total_pairs=1_000_000,
        lmax=4,
        projection_dtype=np.dtype(np.complex64),
        workspace_bytes=10**12,
    )
    assert batch == 65_536


def test_ewald_structural_order_truncation_matches_full_table() -> None:
    import pyceles as pcl
    from pyceles.core.periodic.ewald import ewald_structural_sums_2d_batch

    kwargs = dict(
        lmax_struct=3,
        k=2.0 * np.pi / 550.0,
        destinations=np.asarray([[70.0, 25.0, 31.0], [-42.0, 88.0, 0.0]]),
        source=np.asarray([0.0, 0.0, 0.0]),
        lattice=pcl.RectangularLattice2D(430.0, 510.0),
        k_parallel=np.asarray([8.0e-4, -5.0e-4]),
        eta=0.02,
        real_shells=4,
        reciprocal_shells=4,
    )
    full = ewald_structural_sums_2d_batch(**cast(Any, kwargs))
    truncated = ewald_structural_sums_2d_batch(structural_order=5, **cast(Any, kwargs))

    # The compact table has a different m-offset, so compare physical (p,m)
    # entries rather than rectangular storage columns.
    for degree in range(6):
        for order_m in range(-degree, degree + 1):
            np.testing.assert_allclose(
                truncated[:, degree, order_m + 5],
                full[:, degree, order_m + 6],
                rtol=2e-11,
                atol=2e-11,
            )


def test_cartesian_xy_z_layout_detects_complete_grid() -> None:
    from pyceles.core.periodic.rayleigh_cupy import _cartesian_xy_z_layout

    xy = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
    z = np.asarray([-1.0, 3.0, 7.0, 9.0])
    points = np.asarray([[x, y, zz] for zz in z for x, y in xy], dtype=float)
    order = np.asarray([7, 0, 10, 3, 5, 1, 11, 4, 8, 2, 9, 6])
    points = points[order]

    layout = _cartesian_xy_z_layout(points)
    assert layout is not None
    unique_xy, unique_z, xy_inverse, z_inverse = layout
    reconstructed = np.column_stack([unique_xy[xy_inverse], unique_z[z_inverse]])
    np.testing.assert_allclose(reconstructed, points)


def test_cartesian_xy_z_layout_rejects_incomplete_product() -> None:
    from pyceles.core.periodic.rayleigh_cupy import _cartesian_xy_z_layout

    points = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert _cartesian_xy_z_layout(points) is None


def test_shifted_plane_source_reduction_commutes_with_lateral_synthesis() -> None:
    """Source reduction may precede destination Fourier synthesis exactly."""
    rng = np.random.default_rng(20260903)
    phase = rng.normal(size=(11, 7)) + 1j * rng.normal(size=(11, 7))
    projected = rng.normal(size=(5, 7, 6)) + 1j * rng.normal(size=(5, 7, 6))

    per_source = np.einsum("dt,str->sdr", phase, projected, optimize=True)
    reference = np.sum(per_source, axis=0, dtype=np.complex128)
    reduced_first = phase @ np.sum(projected, axis=0, dtype=np.complex128)

    np.testing.assert_allclose(reduced_first, reference, rtol=2e-15, atol=2e-15)


@pytest.mark.parametrize("orders", [(1, 4), (4, 1), (3, 3), (2, 4)])
def test_sparse_rectangular_source_projection_matches_explicit_mode_pairs(orders):
    lout, lin = orders
    full = max(orders)
    degree_max = lout + lin
    degrees, columns, indptr, inputs, values = rectangular_projection_data(lout, lin)
    matrix = csr_matrix(
        (values, inputs, indptr), shape=(len(degrees) * n_modes(lout), n_modes(lin))
    )
    rng = np.random.default_rng(2718)
    coeffs = rng.normal(size=(3, n_modes(lin))) + 1j * rng.normal(size=(3, n_modes(lin)))
    expected = np.zeros((3, len(degrees), n_modes(lout)), dtype=complex)
    table = translation_ab5_table(full, dtype=np.complex128)
    global_modes = {(tau, l, m): idx for tau, l, m, idx in iter_modes(full)}
    for tau_out, l_out, m_out, out_idx in iter_modes(lout):
        for tau_in, l_in, m_in, in_idx in iter_modes(lin):
            delta = m_in - m_out
            for p in range(abs(delta), degree_max + 1):
                global_out = global_modes[tau_out, l_out, m_out]
                global_in = global_modes[tau_in, l_in, m_in]
                expected[:, p * p + p + delta, out_idx] += (
                    table[global_out, global_in, p] * coeffs[:, in_idx]
                )
    actual = (matrix @ coeffs.T).T.reshape(3, len(degrees), n_modes(lout))
    np.testing.assert_allclose(actual, expected, rtol=5e-13, atol=5e-13)
    assert np.all(np.abs(columns - degree_max) <= degrees)
    assert all(a.ndim == 1 for a in (degrees, columns, indptr, inputs, values))
    # Structural storage is bounded by degree-only mode pairs, not all azimuths.
    assert values.size <= n_modes(lout) * n_modes(lin) * (degree_max + 1)
